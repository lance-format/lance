// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DynamoDB based external manifest store
//!

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::error::SdkError;
use aws_sdk_dynamodb::operation::RequestId;
use aws_sdk_dynamodb::operation::{
    get_item::builders::GetItemFluentBuilder, put_item::builders::PutItemFluentBuilder,
    query::builders::QueryFluentBuilder,
};
use aws_sdk_dynamodb::types::{
    AttributeValue, DeleteRequest, KeyType, ScalarAttributeType, WriteRequest,
};
use object_store::path::Path;
use snafu::OptionExt;
use tokio::sync::RwLock;
use tracing::warn;

use crate::io::commit::external_manifest::ExternalManifestStore;
use lance_core::error::NotFoundSnafu;
use lance_core::error::box_error;
use lance_core::{Error, Result};

use super::ManifestLocation;
use super::external_manifest::detect_naming_scheme_from_path;

#[derive(Debug)]
struct WrappedSdkError<E>(SdkError<E>);

impl<E> From<WrappedSdkError<E>> for Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(e: WrappedSdkError<E>) -> Self {
        Self::io_source(box_error(e))
    }
}

impl<E> std::fmt::Display for WrappedSdkError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let request_id = self.0.request_id().unwrap_or("unknown");
        let service_err = &self.0.raw_response();
        write!(f, "WrappedSdkError: request_id: {}", request_id)?;
        if let Some(err) = service_err {
            write!(f, ", service_error: {:?}", err)
        } else {
            write!(f, ", no service error")
        }
    }
}

impl<E> std::error::Error for WrappedSdkError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    // Implement the necessary methods for the Error trait here.
    // For example, you can delegate to the inner SdkError:

    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

trait SdkResultExt<T> {
    fn wrap_err(self) -> Result<T>;
}

impl<T, E> SdkResultExt<T> for std::result::Result<T, SdkError<E>>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn wrap_err(self) -> Result<T> {
        self.map_err(|err| {
            warn!(
                target: "lance::dynamodb",
                request_id = err.request_id().unwrap_or("unknown"),
                "DynamoDB SDK error: {err:?}",
            );
            Error::from(WrappedSdkError(err))
        })
    }
}

/// An external manifest store backed by DynamoDB
///
/// When calling DynamoDBExternalManifestStore::new_external_store()
/// the key schema, (PK, SK), is checked. If the table does not exist,
/// or the key schema is not as expected, an error is returned.
///
/// The table schema is expected as follows:
/// PK: base_uri -- string
/// SK: version -- number
/// path -- string
/// committer -- string
///
/// Consistency: This store is expected to have read-after-write consistency
/// consistent_read should always be set to true
///
/// Transaction Safety: This store uses DynamoDB conditional write to ensure
/// only one writer can win per version.
///
/// Deleting manifest entries requires the `dynamodb:BatchWriteItem` IAM
/// permission for the table.
#[derive(Debug)]
pub struct DynamoDBExternalManifestStore {
    client: Arc<Client>,
    table_name: String,
    committer_name: String,
}

// these are in macro because I want to use them in a match statement
macro_rules! base_uri {
    () => {
        "base_uri"
    };
}
macro_rules! version {
    () => {
        "version"
    };
}
macro_rules! path {
    () => {
        "path"
    };
}
macro_rules! committer {
    () => {
        "committer"
    };
}

/// Read the `size` attribute of a manifest row.
///
/// Rows written before the column existed omit it, so an absent attribute is not
/// an error: the reader falls back to a `HEAD` request for the manifest length.
/// A present value is a different case - both writers store it as
/// [`AttributeValue::N`] alongside `path` - so report a malformed one instead of
/// dropping the hint and paying for that `HEAD` on every read of the row.
fn parse_size_attribute(item: &HashMap<String, AttributeValue>) -> Result<Option<u64>> {
    let Some(attribute) = item.get("size") else {
        return Ok(None);
    };
    let AttributeValue::N(size) = attribute else {
        return Err(Error::invalid_input(format!(
            "dynamodb error: the size column is {attribute:?}, expected a number"
        )));
    };
    size.parse().map(Some).map_err(|e| {
        Error::invalid_input(format!(
            "dynamodb error: could not parse the size returned {size}, error: {e}"
        ))
    })
}

impl DynamoDBExternalManifestStore {
    pub async fn new_external_store(
        client: Arc<Client>,
        table_name: &str,
        committer_name: &str,
    ) -> Result<Arc<dyn ExternalManifestStore>> {
        static SANITY_CHECK_CACHE: LazyLock<RwLock<HashSet<String>>> =
            LazyLock::new(|| RwLock::new(HashSet::new()));

        let store = Arc::new(Self {
            client: client.clone(),
            table_name: table_name.to_string(),
            committer_name: committer_name.to_string(),
        });

        // already checked this table before, skip
        // this is to avoid checking the table schema every time
        // because it's expensive to call DescribeTable
        if SANITY_CHECK_CACHE.read().await.contains(table_name) {
            return Ok(store);
        }

        // Check if the table schema is correct
        let describe_result = client
            .describe_table()
            .table_name(table_name)
            .send()
            .await
            .wrap_err()?;
        let table = describe_result
            .table
            .ok_or_else(|| Error::io(format!("dynamodb table: {table_name} does not exist")))?;
        let mut schema = table.key_schema.ok_or_else(|| {
            Error::io(format!(
                "dynamodb table: {table_name} does not have a key schema"
            ))
        })?;

        let mut has_hash_key = false;
        let mut has_range_key = false;

        // there should be two keys, HASH(base_uri) and RANGE(version)
        for _ in 0..2 {
            let key = schema.pop().ok_or_else(|| {
                Error::io(format!(
                    "dynamodb table: {table_name} must have HASH and RANGE keys"
                ))
            })?;
            match (key.key_type, key.attribute_name.as_str()) {
                (KeyType::Hash, base_uri!()) => {
                    has_hash_key = true;
                }
                (KeyType::Range, version!()) => {
                    has_range_key = true;
                }
                _ => {
                    return Err(Error::io(format!(
                        "dynamodb table: {} unknown key type encountered name:{}",
                        table_name, key.attribute_name
                    )));
                }
            }
        }

        // Both keys must be present
        if !(has_hash_key && has_range_key) {
            return Err(Error::io(format!(
                "dynamodb table: {} must have HASH and RANGE keys, named `{}` and `{}` respectively",
                table_name,
                base_uri!(),
                version!()
            )));
        }

        for (name, expected_type) in [
            (base_uri!(), ScalarAttributeType::S),
            (version!(), ScalarAttributeType::N),
        ] {
            let attribute_type = table
                .attribute_definitions
                .as_deref()
                .unwrap_or_default()
                .iter()
                .find(|attribute| attribute.attribute_name == name)
                .map(|attribute| &attribute.attribute_type);
            if attribute_type != Some(&expected_type) {
                return Err(Error::invalid_input(format!(
                    "dynamodb table: {table_name} key `{name}` must have type {}, found {}",
                    expected_type.as_str(),
                    attribute_type.map_or("missing", ScalarAttributeType::as_str)
                )));
            }
        }

        SANITY_CHECK_CACHE
            .write()
            .await
            .insert(table_name.to_string());

        Ok(store)
    }

    fn ddb_put(&self) -> PutItemFluentBuilder {
        self.client.put_item().table_name(&self.table_name)
    }

    fn ddb_get(&self) -> GetItemFluentBuilder {
        self.client
            .get_item()
            .table_name(&self.table_name)
            .consistent_read(true)
    }

    fn ddb_query(&self) -> QueryFluentBuilder {
        self.client
            .query()
            .table_name(&self.table_name)
            .consistent_read(true)
    }

    async fn delete_batch(&self, requests: Vec<WriteRequest>) -> Result<()> {
        let mut request_items = HashMap::from([(self.table_name.clone(), requests)]);
        // Throttled writes are returned in a successful response. Retry only
        // those writes with exponential backoff, bounded to five attempts.
        for attempt in 0..5 {
            let result = self
                .client
                .batch_write_item()
                .set_request_items(Some(request_items))
                .send()
                .await
                .wrap_err()?;
            request_items = result.unprocessed_items.unwrap_or_default();
            request_items.retain(|_, requests| !requests.is_empty());
            if request_items.is_empty() {
                return Ok(());
            }
            if attempt < 4 {
                tokio::time::sleep(Duration::from_millis(50 << attempt)).await;
            }
        }
        Err(Error::io(format!(
            "dynamodb table: {} has {} unprocessed manifest deletes after 5 attempts",
            self.table_name,
            request_items.values().map(Vec::len).sum::<usize>()
        )))
    }
}

#[async_trait]
impl ExternalManifestStore for DynamoDBExternalManifestStore {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
        let get_item_result = self
            .ddb_get()
            .key(base_uri!(), AttributeValue::S(base_uri.into()))
            .key(version!(), AttributeValue::N(version.to_string()))
            .send()
            .await
            .wrap_err()?;

        let item = get_item_result.item.context(NotFoundSnafu {
            uri: format!(
                "dynamodb not found: base_uri: {}; version: {}",
                base_uri, version
            ),
        })?;

        let path = item
            .get(path!())
            .ok_or_else(|| Error::not_found(format!("key {} is not present", path!())))?;

        match path {
            AttributeValue::S(path) => Ok(path.clone()),
            _ => Err(Error::invalid_input(format!(
                "key {} is not a string",
                path!()
            ))),
        }
    }

    async fn get_manifest_location(
        &self,
        base_uri: &str,
        version: u64,
    ) -> Result<ManifestLocation> {
        let get_item_result = self
            .ddb_get()
            .key(base_uri!(), AttributeValue::S(base_uri.into()))
            .key(version!(), AttributeValue::N(version.to_string()))
            .send()
            .await
            .wrap_err()?;

        let item = get_item_result.item.context(NotFoundSnafu {
            uri: format!(
                "dynamodb not found: base_uri: {}; version: {}",
                base_uri, version
            ),
        })?;

        let path = item
            .get(path!())
            .ok_or_else(|| Error::not_found(format!("key {} is not present", path!())))?
            .as_s()
            .map_err(|_| Error::invalid_input(format!("key {} is not a string", path!())))?
            .as_str();
        let path = Path::from(path);

        let size = parse_size_attribute(&item)?;

        let naming_scheme = detect_naming_scheme_from_path(&path)?;

        Ok(ManifestLocation {
            version,
            path,
            size,
            naming_scheme,
            // DynamoDB coordinates the logical version but does not own an
            // object generation. Older rows may still contain `e_tag`; ignore
            // it and let the commit handler obtain the current token from the
            // authoritative object store when it validates the final path.
            e_tag: None,
            identity: None,
        })
    }

    /// Get the latest version of a dataset at the base_uri
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
        self.get_latest_manifest_location(base_uri)
            .await
            .map(|location| location.map(|loc| (loc.version, loc.path.to_string())))
    }

    async fn get_latest_manifest_location(
        &self,
        base_uri: &str,
    ) -> Result<Option<ManifestLocation>> {
        let query_result = self
            .ddb_query()
            .key_condition_expression(format!("{} = :{}", base_uri!(), base_uri!()))
            .expression_attribute_values(
                format!(":{}", base_uri!()),
                AttributeValue::S(base_uri.into()),
            )
            .scan_index_forward(false)
            .limit(1)
            .send()
            .await
            .wrap_err()?;

        match query_result.items {
            Some(mut items) => {
                if items.is_empty() {
                    return Ok(None);
                }
                if items.len() > 1 {
                    return Err(Error::invalid_input(format!(
                        "dynamodb table: {} returned unexpected number of items",
                        self.table_name
                    )));
                }

                let item = items.pop().expect("length checked");
                let version_attribute = item
                    .get(version!())
                    .ok_or_else(|| Error::not_found(
                        format!("dynamodb error: found entries for {} but the returned data does not contain {} column", base_uri, version!())
                    ))?;

                let path_attribute = item
                    .get(path!())
                    .ok_or_else(|| Error::not_found(
                        format!("dynamodb error: found entries for {} but the returned data does not contain {} column", base_uri, path!())
                    ))?;

                let size = parse_size_attribute(&item)?;

                match (version_attribute, path_attribute) {
                    (AttributeValue::N(version), AttributeValue::S(path)) => {
                        let version = version.parse().map_err(|e| Error::invalid_input(format!("dynamodb error: could not parse the version number returned {}, error: {}", version, e)))?;
                        let path = Path::from(path.as_str());
                        let naming_scheme = detect_naming_scheme_from_path(&path)?;
                        let location = ManifestLocation {
                            version,
                            path,
                            size,
                            naming_scheme,
                            // See `get_manifest_location`: legacy DDB ETags
                            // are physical-generation observations, not
                            // version identity, and are intentionally ignored.
                            e_tag: None,
                            identity: None,
                        };
                        Ok(Some(location))
                    }
                    _ => Err(Error::invalid_input(format!(
                        "dynamodb error: found entries for {base_uri} but the returned data is not number type"
                    ))),
                }
            }
            _ => Ok(None),
        }
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        // Do not persist an object-store ETag. Staging paths are immutable and
        // uniquely selected by this conditional write; finalized paths are
        // validated against object storage. Persisting an ETag adds no DDB
        // concurrency or content-integrity guarantee and can make the row stale
        // after an identical copy. The commit handler returns the destination
        // ETag separately as an ephemeral runtime cache discriminator.
        self.ddb_put()
            .item(base_uri!(), AttributeValue::S(base_uri.into()))
            .item(version!(), AttributeValue::N(version.to_string()))
            .item(path!(), AttributeValue::S(path.to_string()))
            .item(committer!(), AttributeValue::S(self.committer_name.clone()))
            .item("size", AttributeValue::N(size.to_string()))
            .condition_expression(format!(
                "attribute_not_exists({}) AND attribute_not_exists({})",
                base_uri!(),
                version!(),
            ))
            .send()
            .await
            .wrap_err()?;

        Ok(())
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        _e_tag: Option<String>,
    ) -> Result<()> {
        // Replacing the staging pointer with the canonical path publishes the
        // same generation-independent `(path, size)` tuple from every helper.
        // Each helper still returns the canonical ETag it observed to its caller.
        self.ddb_put()
            .item(base_uri!(), AttributeValue::S(base_uri.into()))
            .item(version!(), AttributeValue::N(version.to_string()))
            .item(path!(), AttributeValue::S(path.to_string()))
            .item(committer!(), AttributeValue::S(self.committer_name.clone()))
            .item("size", AttributeValue::N(size.to_string()))
            .condition_expression(format!(
                "attribute_exists({}) AND attribute_exists({})",
                base_uri!(),
                version!(),
            ))
            .send()
            .await
            .wrap_err()?;

        Ok(())
    }

    /// Delete the manifest information for the given base_uri in dynamodb
    async fn delete(&self, base_uri: &str) -> Result<()> {
        let mut last_evaluated_key = None;
        loop {
            let query_result = self
                .ddb_query()
                .key_condition_expression(format!("{} = :{}", base_uri!(), base_uri!()))
                .expression_attribute_values(
                    format!(":{}", base_uri!()),
                    AttributeValue::S(base_uri.into()),
                )
                .projection_expression("#base_uri, #version")
                .expression_attribute_names("#base_uri", base_uri!())
                .expression_attribute_names("#version", version!())
                .set_exclusive_start_key(last_evaluated_key.take())
                .send()
                .await
                .wrap_err()?;

            let items = query_result.items.unwrap_or_default();
            // DynamoDB accepts at most 25 writes in one batch.
            for batch in items.chunks(25) {
                let requests = batch
                    .iter()
                    .map(|item| {
                        let request = DeleteRequest::builder()
                            .set_key(Some(item.clone()))
                            .build()
                            .map_err(|error| {
                                Error::invalid_input(format!(
                                    "dynamodb table: {} invalid manifest key: {error}",
                                    self.table_name
                                ))
                            })?;
                        Ok(WriteRequest::builder().delete_request(request).build())
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.delete_batch(requests).await?;
            }
            last_evaluated_key = query_result
                .last_evaluated_key
                .filter(|key| !key.is_empty());
            if last_evaluated_key.is_none() {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use aws_credential_types::Credentials;
    use aws_sdk_dynamodb::config::{Region, retry::RetryConfig};
    use rstest::rstest;
    use serde_json::{Value, json};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers::method};

    use super::*;

    async fn mock_client(responses: Vec<Value>) -> (Arc<Client>, MockServer) {
        let server = MockServer::start().await;
        let count = responses.len() as u64;
        let responses = Mutex::new(VecDeque::from(responses));
        Mock::given(method("POST"))
            .respond_with(move |_: &Request| {
                ResponseTemplate::new(200).set_body_json(
                    responses
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("unexpected DynamoDB request"),
                )
            })
            .expect(count)
            .up_to_n_times(count)
            .mount(&server)
            .await;
        let config = aws_sdk_dynamodb::config::Builder::new()
            .behavior_version_latest()
            .endpoint_url(server.uri())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("test", "test", None, None, "test"))
            .retry_config(RetryConfig::standard().with_max_attempts(1))
            .build();
        (Arc::new(Client::from_conf(config)), server)
    }

    async fn assert_requests(server: &MockServer, expected: Vec<(&str, Value)>) {
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), expected.len());
        for (request, (operation, body)) in requests.iter().zip(expected) {
            assert_eq!(
                request.headers["x-amz-target"].to_str().unwrap(),
                format!("DynamoDB_20120810.{operation}")
            );
            assert_eq!(request.body_json::<Value>().unwrap(), body);
        }
    }

    #[rstest]
    #[case::valid(Some("S"), Some("N"), None)]
    #[case::numeric_base_uri(Some("N"), Some("N"), Some("base_uri"))]
    #[case::binary_base_uri(Some("B"), Some("N"), Some("base_uri"))]
    #[case::string_version(Some("S"), Some("S"), Some("version"))]
    #[case::binary_version(Some("S"), Some("B"), Some("version"))]
    #[case::missing_base_uri(None, Some("N"), Some("base_uri"))]
    #[case::missing_version(Some("S"), None, Some("version"))]
    #[tokio::test]
    async fn test_key_attribute_types(
        #[case] base_uri_type: Option<&str>,
        #[case] version_type: Option<&str>,
        #[case] invalid_key: Option<&str>,
    ) {
        let definitions: Vec<_> = [("base_uri", base_uri_type), ("version", version_type)]
            .into_iter()
            .filter_map(|(name, scalar_type)| {
                scalar_type
                    .map(|scalar_type| json!({"AttributeName": name, "AttributeType": scalar_type}))
            })
            .collect();
        let response = json!({"Table": {
            "KeySchema": [
                {"AttributeName": "base_uri", "KeyType": "HASH"},
                {"AttributeName": "version", "KeyType": "RANGE"}
            ],
            "AttributeDefinitions": definitions
        }});
        let responses = if invalid_key.is_some() {
            vec![response.clone(), response]
        } else {
            vec![response]
        };
        let (client, server) = mock_client(responses).await;
        let table_name = uuid::Uuid::new_v4().to_string();
        // Only a successful validation may populate the sanity-check cache.
        for _ in 0..2 {
            let result = DynamoDBExternalManifestStore::new_external_store(
                client.clone(),
                &table_name,
                "test",
            )
            .await;
            if let Some(key) = invalid_key {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }));
                let expected_type = if key == "base_uri" { "S" } else { "N" };
                let actual_type = if key == "base_uri" {
                    base_uri_type
                } else {
                    version_type
                }
                .unwrap_or("missing");
                assert!(error.to_string().contains(&format!(
                    "dynamodb table: {table_name} key `{key}` must have type {expected_type}, found {actual_type}"
                )));
            } else {
                assert!(result.is_ok());
            }
        }
        let request = ("DescribeTable", json!({"TableName": table_name}));
        assert_requests(
            &server,
            if invalid_key.is_some() {
                vec![request.clone(), request]
            } else {
                vec![request]
            },
        )
        .await;
    }

    fn manifest_key(version: u64) -> Value {
        json!({"base_uri": {"S": "s3://bucket/table"}, "version": {"N": version.to_string()}})
    }

    fn delete_requests(versions: std::ops::RangeInclusive<u64>) -> Value {
        Value::Array(
            versions
                .map(|version| json!({"DeleteRequest": {"Key": manifest_key(version)}}))
                .collect(),
        )
    }

    fn delete_query() -> Value {
        json!({
            "TableName": "manifests",
            "ConsistentRead": true,
            "KeyConditionExpression": "base_uri = :base_uri",
            "ExpressionAttributeValues": {":base_uri": {"S": "s3://bucket/table"}},
            "ProjectionExpression": "#base_uri, #version",
            "ExpressionAttributeNames": {"#base_uri": "base_uri", "#version": "version"}
        })
    }

    #[tokio::test]
    async fn test_delete_all_pages_in_batches() {
        let (client, server) = mock_client(vec![
            json!({"Items": (1..=26).map(manifest_key).collect::<Vec<_>>(),
                "LastEvaluatedKey": manifest_key(26)}),
            json!({}),
            json!({}),
            json!({"Items": [manifest_key(27)], "LastEvaluatedKey": {}}),
            json!({}),
        ])
        .await;
        let store = DynamoDBExternalManifestStore {
            client,
            table_name: "manifests".into(),
            committer_name: "test".into(),
        };
        store.delete("s3://bucket/table").await.unwrap();
        let mut next_query = delete_query();
        next_query["ExclusiveStartKey"] = manifest_key(26);
        assert_requests(
            &server,
            vec![
                ("Query", delete_query()),
                (
                    "BatchWriteItem",
                    json!({"RequestItems": {"manifests": delete_requests(1..=25)}}),
                ),
                (
                    "BatchWriteItem",
                    json!({"RequestItems": {"manifests": delete_requests(26..=26)}}),
                ),
                ("Query", next_query),
                (
                    "BatchWriteItem",
                    json!({"RequestItems": {"manifests": delete_requests(27..=27)}}),
                ),
            ],
        )
        .await;
    }

    #[rstest]
    #[case::no_items(json!({}))]
    #[case::empty_items(json!({"Items": []}))]
    #[tokio::test]
    async fn test_delete_empty_dataset(#[case] response: Value) {
        let (client, server) = mock_client(vec![response]).await;
        let store = DynamoDBExternalManifestStore {
            client,
            table_name: "manifests".into(),
            committer_name: "test".into(),
        };
        store.delete("s3://bucket/table").await.unwrap();
        assert_requests(&server, vec![("Query", delete_query())]).await;
    }

    #[rstest]
    #[case::eventual_success(false)]
    #[case::retry_exhaustion(true)]
    #[tokio::test]
    async fn test_delete_unprocessed_items(#[case] is_exhausted: bool) {
        let unprocessed = json!({"UnprocessedItems": {"manifests": delete_requests(2..=2)}});
        let mut responses = vec![
            json!({"Items": [manifest_key(1), manifest_key(2)]}),
            unprocessed.clone(),
        ];
        if is_exhausted {
            responses.extend(vec![unprocessed; 4]);
        } else {
            responses.push(json!({"UnprocessedItems": {"manifests": []}}));
        }
        let (client, server) = mock_client(responses).await;
        let store = DynamoDBExternalManifestStore {
            client,
            table_name: "manifests".into(),
            committer_name: "test".into(),
        };
        let result = store.delete("s3://bucket/table").await;
        if is_exhausted {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains(
                "dynamodb table: manifests has 1 unprocessed manifest deletes after 5 attempts"
            ));
        } else {
            result.unwrap();
        }
        let mut requests = vec![
            ("Query", delete_query()),
            (
                "BatchWriteItem",
                json!({"RequestItems": {"manifests": delete_requests(1..=2)}}),
            ),
        ];
        requests.extend(vec![
            (
                "BatchWriteItem",
                json!({"RequestItems": {"manifests": delete_requests(2..=2)}}),
            );
            if is_exhausted { 4 } else { 1 }
        ]);
        assert_requests(&server, requests).await;
    }

    fn item(size: Option<AttributeValue>) -> HashMap<String, AttributeValue> {
        let mut item = HashMap::from([
            (
                base_uri!().to_string(),
                AttributeValue::S("s3://bucket/table".into()),
            ),
            (version!().to_string(), AttributeValue::N("7".into())),
            (
                path!().to_string(),
                AttributeValue::S("_versions/7.manifest".into()),
            ),
        ]);
        if let Some(size) = size {
            item.insert("size".to_string(), size);
        }
        item
    }

    /// Rows predate the column, so an absent value keeps the `HEAD` fallback
    /// rather than becoming an error.
    #[test]
    fn test_absent_size_is_not_an_error() {
        assert_eq!(parse_size_attribute(&item(None)).unwrap(), None);
    }

    #[test]
    fn test_size_roundtrips_through_the_attribute() {
        for size in [0u64, 1, 4096, u64::MAX] {
            let attribute = AttributeValue::N(size.to_string());
            assert_eq!(
                parse_size_attribute(&item(Some(attribute))).unwrap(),
                Some(size),
                "size {size}"
            );
        }
    }

    /// A present value is written by both put helpers, so a malformed one means a
    /// corrupt row: report it instead of silently dropping the length hint.
    #[test]
    fn test_unreadable_size_is_reported() {
        for raw in ["", "abc", "-1", "1.5", "99999999999999999999999"] {
            let attribute = AttributeValue::N(raw.to_string());
            let err = parse_size_attribute(&item(Some(attribute)))
                .expect_err(&format!("size {raw:?} should be rejected"));
            assert!(
                err.to_string().contains("could not parse the size"),
                "size {raw:?}: unexpected message {err}"
            );
        }
    }

    #[test]
    fn test_non_numeric_size_attribute_is_reported() {
        for attribute in [
            AttributeValue::S("4096".into()),
            AttributeValue::Bool(true),
            AttributeValue::Null(true),
        ] {
            let err = parse_size_attribute(&item(Some(attribute.clone())))
                .expect_err(&format!("{attribute:?} should be rejected"));
            assert!(
                err.to_string().contains("expected a number"),
                "{attribute:?}: unexpected message {err}"
            );
        }
    }
}
