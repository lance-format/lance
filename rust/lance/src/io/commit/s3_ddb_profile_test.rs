// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use lance_io::object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor};
use lance_table::io::commit::{CommitHandler, commit_handler_from_url};
use object_store::path::Path;
use wiremock::matchers::header;
use wiremock::{Mock, MockServer, ResponseTemplate};

const CHILD_ENV: &str = "LANCE_TEST_S3_DDB_PROFILE_CHILD";
const ENDPOINT_ENV: &str = "LANCE_TEST_S3_DDB_PROFILE_ENDPOINT";
const SCRUBBED_ENV: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL_DYNAMODB",
    "DYNAMODB_ENDPOINT",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_EC2_METADATA_DISABLED",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
];

const DESCRIBE_TABLE_RESPONSE: &str = r#"{
  "Table": {
    "AttributeDefinitions": [
      {"AttributeName": "base_uri", "AttributeType": "S"},
      {"AttributeName": "version", "AttributeType": "N"}
    ],
    "KeySchema": [
      {"AttributeName": "base_uri", "KeyType": "HASH"},
      {"AttributeName": "version", "KeyType": "RANGE"}
    ],
    "TableStatus": "ACTIVE"
  }
}"#;

const QUERY_RESPONSE: &str = r#"{"Items":[],"Count":0,"ScannedCount":0}"#;

#[tokio::test]
async fn test_s3_ddb_profile_selection_through_commit_handler() {
    if std::env::var_os(CHILD_ENV).is_none() {
        run_s3_ddb_profile_parent().await;
    } else {
        run_s3_ddb_profile_child().await;
    }
}

async fn run_s3_ddb_profile_parent() {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "DynamoDB_20120810.DescribeTable"))
        .respond_with(dynamodb_response(DESCRIBE_TABLE_RESPONSE))
        .mount(&server)
        .await;
    Mock::given(header("x-amz-target", "DynamoDB_20120810.Query"))
        .respond_with(dynamodb_response(QUERY_RESPONSE))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config");
    let credentials_path = dir.path().join("credentials");
    std::fs::write(
        &config_path,
        "[profile secondary]\n\
         region = ap-southeast-2\n\
         endpoint_url = http://127.0.0.1:1\n\
         \n\
         [profile primary]\n\
         region = eu-west-1\n\
         endpoint_url = http://127.0.0.1:1\n",
    )
    .unwrap();
    std::fs::write(
        &credentials_path,
        "[secondary]\n\
         aws_access_key_id = TESTSECONDARY\n\
         aws_secret_access_key = secondary-secret\n\
         aws_session_token = session-secondary\n\
         \n\
         [primary]\n\
         aws_access_key_id = TESTPRIMARY\n\
         aws_secret_access_key = primary-secret\n\
         aws_session_token = session-primary\n",
    )
    .unwrap();

    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        std::thread::current().name().unwrap(),
        "--nocapture",
    ]);
    for &key in SCRUBBED_ENV {
        command.env_remove(key);
    }
    let output = command
        .env(CHILD_ENV, "1")
        .env(ENDPOINT_ENV, server.uri())
        .env("AWS_CONFIG_FILE", &config_path)
        .env("AWS_SHARED_CREDENTIALS_FILE", &credentials_path)
        .env("AWS_PROFILE", "secondary")
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child test failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed"),
        "child did not run exactly one test:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let requests = server
        .received_requests()
        .await
        .expect("wiremock request recording is enabled");
    let describe_requests = dynamodb_requests(&requests, "DynamoDB_20120810.DescribeTable");
    let query_requests = dynamodb_requests(&requests, "DynamoDB_20120810.Query");
    assert_eq!(describe_requests.len(), 3, "{describe_requests:?}");
    assert_eq!(query_requests.len(), 3, "{query_requests:?}");

    let expected = [
        ("TESTPRIMARY", "eu-west-1", "session-primary"),
        ("TESTPRIMARY", "ca-central-1", "session-primary"),
        ("TESTSECONDARY", "ap-southeast-2", "session-secondary"),
    ];
    for (requests, operation) in [
        (&describe_requests, "DescribeTable"),
        (&query_requests, "Query"),
    ] {
        for (request, (access_key, region, token)) in requests.iter().copied().zip(expected) {
            assert_signed_ddb_request(request, operation, access_key, region, token);
        }
    }
}

async fn run_s3_ddb_profile_child() {
    let endpoint = std::env::var(ENDPOINT_ENV).unwrap();
    let cases = [
        ("fixture-commits-primary", Some("primary"), None),
        (
            "fixture-commits-region",
            Some("primary"),
            Some("ca-central-1"),
        ),
        ("fixture-commits-ambient", None, None),
    ];

    let object_store = ObjectStore::memory();
    for (table_name, profile, region) in cases {
        let handler = open_dynamodb_handler(&endpoint, profile, region, table_name).await;
        let _ = handler
            .resolve_latest_location(&Path::from("db"), &object_store)
            .await;
    }
}

async fn open_dynamodb_handler(
    endpoint: &str,
    profile: Option<&str>,
    region: Option<&str>,
    table_name: &str,
) -> Arc<dyn CommitHandler> {
    let mut options = HashMap::from([("dynamodb_endpoint".to_string(), endpoint.to_string())]);
    if let Some(profile) = profile {
        options.insert("aws_profile".to_string(), profile.to_string());
    }
    if let Some(region) = region {
        options.insert("region".to_string(), region.to_string());
    }
    let params = ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            options,
        ))),
        ..Default::default()
    };
    let url = format!("s3+ddb://bucket/db?ddbTableName={table_name}");
    commit_handler_from_url(&url, &Some(params)).await.unwrap()
}

fn dynamodb_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "application/x-amz-json-1.0")
        .set_body_string(body)
}

fn dynamodb_requests<'a>(
    requests: &'a [wiremock::Request],
    operation: &str,
) -> Vec<&'a wiremock::Request> {
    requests
        .iter()
        .filter(|request| {
            request
                .headers
                .get("x-amz-target")
                .and_then(|value| value.to_str().ok())
                == Some(operation)
        })
        .collect()
}

fn assert_signed_ddb_request(
    request: &wiremock::Request,
    operation: &str,
    access_key: &str,
    region: &str,
    token: &str,
) {
    let authorization = request
        .headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_else(|| panic!("{operation} request should be signed: {request:?}"));
    assert!(
        authorization.contains(&format!("Credential={access_key}/")),
        "expected access key {access_key} in {authorization}"
    );
    assert!(
        authorization.contains(&format!("/{region}/dynamodb/aws4_request")),
        "expected region {region} in {authorization}"
    );
    assert_eq!(
        request
            .headers
            .get("x-amz-security-token")
            .and_then(|value| value.to_str().ok()),
        Some(token),
        "expected session token {token} on {operation}"
    );
}
