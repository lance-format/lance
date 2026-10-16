// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `DynamoDBExternalManifestStore::delete` against a scripted DynamoDB endpoint.
//!
//! The store's paging and batching are only observable in the sequence of
//! requests it issues, so this serves canned JSON responses to the real SDK and
//! asserts on what it asked for. It covers the two shapes that cannot be
//! reached from a unit test: a second page, and a terminal `LastEvaluatedKey`
//! that arrives as an empty map rather than being absent.

#![cfg(feature = "dynamodb")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::config::{Credentials, Region, retry::RetryConfig};
use lance_table::io::commit::dynamodb::DynamoDBExternalManifestStore;
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

/// The versions carried by a recorded `BatchWriteItem` body, in order.
fn deleted_versions(body: &Value, table: &str) -> Vec<String> {
    body["RequestItems"][table]
        .as_array()
        .expect("batch carries requests for the table")
        .iter()
        .map(|request| {
            request["DeleteRequest"]["Key"]["version"]["N"]
                .as_str()
                .expect("delete request keys a numeric version")
                .to_string()
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
    assert_eq!(deleted_versions(&recorded[2].1, table), vec!["1", "2"]);
    assert_eq!(deleted_versions(&recorded[4].1, table), vec!["3"]);
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
    assert_eq!(deleted_versions(&recorded[2].1, table), vec!["1", "2"]);
    assert_eq!(deleted_versions(&recorded[3].1, table), vec!["2"]);
}
