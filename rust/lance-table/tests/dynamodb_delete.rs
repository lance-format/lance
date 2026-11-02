// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `DynamoDBExternalManifestStore::delete` against a scripted DynamoDB endpoint.
//!
//! The store's paging and batching are only observable in the sequence of
//! requests it issues, so this serves canned JSON responses to the real SDK and
//! asserts on what it asked for. It covers multi-page deletes, the 25-request
//! batch limit, unprocessed-row retries, and empty partitions.

#![cfg(feature = "dynamodb")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::config::{Credentials, Region, retry::RetryConfig};
use lance_core::Error;
use lance_table::io::commit::dynamodb::DynamoDBExternalManifestStore;
use rstest::rstest;
use serde_json::{Value, json};

/// One recorded request: the operation name from `X-Amz-Target` and its body.
type Recorded = (String, Value);

/// Serve `responses` in order, one per connection, recording each request.
///
/// Each response closes the connection, so the client opens a new one per
/// request and the pairing stays obvious. The thread exits once the script is
/// exhausted; a request past the end of the script means the store asked for
/// more than it should have, which the assertions catch as a missing request.
fn scripted_endpoint(responses: Vec<Value>) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = recorded.clone();

    std::thread::spawn(move || {
        for response in responses {
            let Ok((mut socket, _)) = listener.accept() else {
                return;
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();

            // Read the headers, then exactly Content-Length bytes of body.
            let mut raw = Vec::new();
            let (head_len, body_len, target) = loop {
                let mut chunk = [0u8; 4096];
                let read = socket.read(&mut chunk).unwrap();
                assert_ne!(read, 0, "client closed before sending a request");
                raw.extend_from_slice(&chunk[..read]);
                if let Some(at) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&raw[..at]).to_string();
                    let header = |name: &str| {
                        head.lines()
                            .find_map(|line| line.strip_prefix(name))
                            .map(|rest| rest.trim().to_string())
                    };
                    let len: usize = header("content-length:")
                        .or_else(|| header("Content-Length:"))
                        .expect("request has a content length")
                        .parse()
                        .unwrap();
                    let target = header("x-amz-target:")
                        .or_else(|| header("X-Amz-Target:"))
                        .expect("request has an x-amz-target")
                        .rsplit('.')
                        .next()
                        .unwrap()
                        .to_string();
                    break (at + 4, len, target);
                }
            };
            while raw.len() < head_len + body_len {
                let mut chunk = [0u8; 4096];
                let read = socket.read(&mut chunk).unwrap();
                assert_ne!(read, 0, "client closed mid-body");
                raw.extend_from_slice(&chunk[..read]);
            }
            let body: Value = serde_json::from_slice(&raw[head_len..head_len + body_len]).unwrap();
            sink.lock().unwrap().push((target, body));

            let payload = response.to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-amz-json-1.0\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            socket.write_all(reply.as_bytes()).unwrap();
            let _ = socket.flush();
        }
    });

    (endpoint, recorded)
}

fn client(endpoint: &str) -> Arc<Client> {
    let config = aws_sdk_dynamodb::config::Builder::new()
        .behavior_version_latest()
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::for_tests())
        .endpoint_url(endpoint)
        // The script pairs one response with one request, so a retry would
        // consume the next response and hide what actually happened.
        .retry_config(RetryConfig::disabled())
        .build();
    Arc::new(Client::from_conf(config))
}

/// The table description the store's sanity check accepts.
fn describe_table(table: &str) -> Value {
    json!({"Table": {
        "TableName": table,
        "KeySchema": [
            {"AttributeName": "base_uri", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "base_uri", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
    }})
}

fn row(base_uri: &str, version: &str) -> Value {
    json!({
        "base_uri": {"S": base_uri},
        "version": {"N": version},
        "path": {"S": format!("_versions/{version}.manifest")},
    })
}

/// The complete keys carried by a recorded `BatchWriteItem` body, in order.
fn deleted_keys(body: &Value, table: &str) -> Vec<(String, String)> {
    let request_items = body["RequestItems"]
        .as_object()
        .expect("batch carries request items");
    assert_eq!(
        request_items.keys().map(String::as_str).collect::<Vec<_>>(),
        vec![table],
        "batch must target only the configured table"
    );

    request_items[table]
        .as_array()
        .expect("batch carries requests for the table")
        .iter()
        .map(|request| {
            let key = &request["DeleteRequest"]["Key"];
            let base_uri = key["base_uri"]["S"]
                .as_str()
                .expect("delete request keys a string base URI")
                .to_string();
            let version = key["version"]["N"]
                .as_str()
                .expect("delete request keys a numeric version")
                .to_string();
            (base_uri, version)
        })
        .collect()
}

#[tokio::test]
async fn delete_drains_every_page_and_stops_on_an_empty_terminal_key() {
    let table = "lance_delete_paging";
    let base_uri = "s3://bucket/table";
    // Page one hands back a continuation key; page two reports the end of the
    // results the way the Query contract allows, as an empty map rather than an
    // absent one. Reading that as "there is more" queries forever.
    let (endpoint, recorded) = scripted_endpoint(vec![
        describe_table(table),
        json!({
            "Items": [row(base_uri, "1"), row(base_uri, "2")],
            "LastEvaluatedKey": {"base_uri": {"S": base_uri}, "version": {"N": "2"}},
        }),
        json!({"UnprocessedItems": {}}),
        json!({"Items": [row(base_uri, "3")], "LastEvaluatedKey": {}}),
        json!({"UnprocessedItems": {}}),
    ]);

    let store = DynamoDBExternalManifestStore::new_external_store(
        client(&endpoint),
        table,
        "test-committer",
    )
    .await
    .unwrap();
    store.delete(base_uri).await.unwrap();

    let recorded = recorded.lock().unwrap();
    let targets: Vec<&str> = recorded.iter().map(|(target, _)| target.as_str()).collect();
    assert_eq!(
        targets,
        vec![
            "DescribeTable",
            "Query",
            "BatchWriteItem",
            "Query",
            "BatchWriteItem"
        ],
        "the empty terminal key must end the paging"
    );
    assert_eq!(
        deleted_keys(&recorded[2].1, table),
        vec![
            (base_uri.to_string(), "1".to_string()),
            (base_uri.to_string(), "2".to_string())
        ]
    );
    assert_eq!(
        deleted_keys(&recorded[4].1, table),
        vec![(base_uri.to_string(), "3".to_string())]
    );
}

#[tokio::test]
async fn delete_resends_unprocessed_rows() {
    let table = "lance_delete_unprocessed";
    let base_uri = "s3://bucket/table";
    // BatchWriteItem reports a successful call that did not do all of the work;
    // the leftover row has to be sent again.
    let leftover = json!({table: [{"DeleteRequest": {"Key": {
        "base_uri": {"S": base_uri},
        "version": {"N": "2"},
    }}}]});
    let (endpoint, recorded) = scripted_endpoint(vec![
        describe_table(table),
        json!({"Items": [row(base_uri, "1"), row(base_uri, "2")]}),
        json!({"UnprocessedItems": leftover}),
        json!({"UnprocessedItems": {}}),
    ]);

    let store = DynamoDBExternalManifestStore::new_external_store(
        client(&endpoint),
        table,
        "test-committer",
    )
    .await
    .unwrap();
    store.delete(base_uri).await.unwrap();

    let recorded = recorded.lock().unwrap();
    let targets: Vec<&str> = recorded.iter().map(|(target, _)| target.as_str()).collect();
    assert_eq!(
        targets,
        vec!["DescribeTable", "Query", "BatchWriteItem", "BatchWriteItem"],
        "the unprocessed row must be sent again"
    );
    assert_eq!(
        deleted_keys(&recorded[2].1, table),
        vec![
            (base_uri.to_string(), "1".to_string()),
            (base_uri.to_string(), "2".to_string())
        ]
    );
    assert_eq!(
        deleted_keys(&recorded[3].1, table),
        vec![(base_uri.to_string(), "2".to_string())]
    );
}

#[tokio::test]
async fn delete_splits_each_query_page_into_batches_of_25() {
    let table = "lance_delete_batch_limit";
    let base_uri = "s3://bucket/table";
    let items = (1..=26)
        .map(|version| row(base_uri, &version.to_string()))
        .collect::<Vec<_>>();
    let (endpoint, recorded) = scripted_endpoint(vec![
        describe_table(table),
        json!({"Items": items}),
        json!({"UnprocessedItems": {}}),
        json!({"UnprocessedItems": {}}),
    ]);

    let store = DynamoDBExternalManifestStore::new_external_store(
        client(&endpoint),
        table,
        "test-committer",
    )
    .await
    .unwrap();
    store.delete(base_uri).await.unwrap();

    let recorded = recorded.lock().unwrap();
    let targets = recorded
        .iter()
        .map(|(target, _)| target.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        targets,
        vec!["DescribeTable", "Query", "BatchWriteItem", "BatchWriteItem"]
    );

    let first_batch = (1..=25)
        .map(|version| (base_uri.to_string(), version.to_string()))
        .collect::<Vec<_>>();
    assert_eq!(deleted_keys(&recorded[2].1, table), first_batch);
    assert_eq!(
        deleted_keys(&recorded[3].1, table),
        vec![(base_uri.to_string(), "26".to_string())]
    );
}

#[tokio::test(start_paused = true)]
async fn delete_errors_when_unprocessed_rows_exhaust_retries() {
    let table = "lance_delete_unprocessed_exhausted";
    let base_uri = "s3://bucket/table";
    let leftover = json!({table: [{"DeleteRequest": {"Key": {
        "base_uri": {"S": base_uri},
        "version": {"N": "2"},
    }}}]});
    let mut responses = vec![
        describe_table(table),
        json!({"Items": [row(base_uri, "1"), row(base_uri, "2")]}),
    ];
    // The production limit is the initial send plus eight re-sends.
    let batch_attempts = 9;
    responses.extend((0..batch_attempts).map(|_| json!({"UnprocessedItems": leftover.clone()})));
    let (endpoint, recorded) = scripted_endpoint(responses);

    let store = DynamoDBExternalManifestStore::new_external_store(
        client(&endpoint),
        table,
        "test-committer",
    )
    .await
    .unwrap();
    let error = store
        .delete(base_uri)
        .await
        .expect_err("unprocessed rows must eventually fail");
    assert!(
        matches!(&error, Error::IO { .. }),
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains(&format!(
            "dynamodb table: {table} left 1 manifest rows unprocessed after re-sending them 8 times"
        )),
        "unexpected error: {error}"
    );

    let recorded = recorded.lock().unwrap();
    let targets = recorded
        .iter()
        .map(|(target, _)| target.as_str())
        .collect::<Vec<_>>();
    let mut expected_targets = vec!["DescribeTable", "Query"];
    expected_targets.extend(std::iter::repeat_n("BatchWriteItem", batch_attempts));
    assert_eq!(targets, expected_targets);
    assert_eq!(
        deleted_keys(&recorded[2].1, table),
        vec![
            (base_uri.to_string(), "1".to_string()),
            (base_uri.to_string(), "2".to_string())
        ]
    );
    for record in &recorded[3..] {
        assert_eq!(
            deleted_keys(&record.1, table),
            vec![(base_uri.to_string(), "2".to_string())],
            "only the unprocessed key must be re-sent"
        );
    }
}

#[rstest]
#[case::missing_items("lance_delete_missing_items", json!({}))]
#[case::empty_items("lance_delete_empty_items", json!({"Items": []}))]
#[tokio::test]
async fn delete_empty_partition_skips_batch(#[case] table: &str, #[case] query_response: Value) {
    let base_uri = "s3://bucket/table";
    let (endpoint, recorded) = scripted_endpoint(vec![describe_table(table), query_response]);

    let store = DynamoDBExternalManifestStore::new_external_store(
        client(&endpoint),
        table,
        "test-committer",
    )
    .await
    .unwrap();
    store.delete(base_uri).await.unwrap();

    let recorded = recorded.lock().unwrap();
    let targets = recorded
        .iter()
        .map(|(target, _)| target.as_str())
        .collect::<Vec<_>>();
    assert_eq!(targets, vec!["DescribeTable", "Query"]);
}
