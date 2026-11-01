// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DynamoDB based external manifest store
//!

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::error::SdkError;
use aws_sdk_dynamodb::operation::RequestId;
use aws_sdk_dynamodb::operation::{
    get_item::builders::GetItemFluentBuilder, put_item::builders::PutItemFluentBuilder,
    query::builders::QueryFluentBuilder,
};
use aws_sdk_dynamodb::types::{
    AttributeDefinition, AttributeValue, DeleteRequest, KeyType, WriteRequest,
};
use object_store::path::Path;
use snafu::OptionExt;
use tokio::sync::RwLock;
use tracing::warn;

use crate::io::commit::external_manifest::ExternalManifestStore;
use lance_core::error::NotFoundSnafu;
use lance_core::error::box_error;
use lance_core::utils::backoff::Backoff;
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
/// The credentials need `dynamodb:DescribeTable` for that check, `GetItem`,
/// `PutItem` and `Query` for the commit protocol, and `BatchWriteItem` for
/// [`ExternalManifestStore::delete`], which removes a dataset's rows in batches.
///
/// Consistency: This store is expected to have read-after-write consistency
/// consistent_read should always be set to true
///
/// Transaction Safety: This store uses DynamoDB conditional write to ensure
/// only one writer can win per version.
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

/// The maximum number of write requests `BatchWriteItem` accepts.
const BATCH_WRITE_ITEM_LIMIT: usize = 25;

/// How many times to re-send the rows `BatchWriteItem` reports as unprocessed
/// before giving up and reporting them.
const MAX_BATCH_WRITE_ATTEMPTS: u32 = 8;

/// Check that the table's key attributes have the types this store writes.
///
/// The key schema check only compares names, and the range key's type decides
/// how DynamoDB orders it. [`DynamoDBExternalManifestStore::get_latest_manifest_location`]
/// takes the first row of a descending scan, so a string-typed version orders
/// "9" above "12" and silently resolves the wrong latest version.
fn validate_key_attribute_types(
    table_name: &str,
    attribute_definitions: Option<&[AttributeDefinition]>,
) -> Result<()> {
    let declared: HashMap<&str, &str> = attribute_definitions
        .unwrap_or_default()
        .iter()
        .map(|def| (def.attribute_name.as_str(), def.attribute_type.as_str()))
        .collect();
    for (key, expected) in [(base_uri!(), "S"), (version!(), "N")] {
        match declared.get(key) {
            Some(actual) if *actual == expected => {}
            Some(actual) => {
                return Err(Error::io(format!(
                    "dynamodb table: {table_name} key `{key}` must have attribute type \
                     {expected}, but the table declares {actual}"
                )));
            }
            None => {
                return Err(Error::io(format!(
                    "dynamodb table: {table_name} does not declare an attribute type for \
                     key `{key}`, expected {expected}"
                )));
            }
        }
    }
    Ok(())
}

/// Build the delete requests for one page of manifest rows.
///
/// A row without a numeric `version` is not one this store wrote, so it is left
/// alone rather than deleted on a guess.
fn page_delete_requests(
    base_uri: &str,
    items: &[HashMap<String, AttributeValue>],
) -> Result<Vec<WriteRequest>> {
    let mut requests = Vec::with_capacity(items.len());
    for item in items {
        if let Some(AttributeValue::N(version)) = item.get(version!()) {
            let delete = DeleteRequest::builder()
                .key(base_uri!(), AttributeValue::S(base_uri.to_string()))
                .key(version!(), AttributeValue::N(version.clone()))
                .build()
                .map_err(|e| Error::io(format!("failed to build DynamoDB delete request: {e}")))?;
            requests.push(WriteRequest::builder().delete_request(delete).build());
        }
    }
    Ok(requests)
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

        validate_key_attribute_types(table_name, table.attribute_definitions.as_deref())?;

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

    /// Send one `BatchWriteItem` and re-send whatever the service reports as
    /// unprocessed.
    ///
    /// The client's retry policy does not cover these: the call succeeded, it
    /// just did not do all of the work (a throttled table). Bounded with a
    /// backoff so a throttled table reports what is left instead of spinning
    /// against the service.
    async fn batch_delete(&self, requests: &[WriteRequest]) -> Result<()> {
        let mut pending = requests.to_vec();
        let mut backoff = Backoff::default();
        while !pending.is_empty() {
            let request_items = HashMap::from([(self.table_name.clone(), pending)]);
            let result = self
                .client
                .batch_write_item()
                .set_request_items(Some(request_items))
                .send()
                .await
                .wrap_err()?;
            pending = result
                .unprocessed_items
                .and_then(|mut items| items.remove(&self.table_name))
                .unwrap_or_default();
            if pending.is_empty() {
                break;
            }
            if backoff.attempt() >= MAX_BATCH_WRITE_ATTEMPTS {
                return Err(Error::io(format!(
                    "dynamodb table: {} left {} manifest rows unprocessed after re-sending them {} times",
                    self.table_name,
                    pending.len(),
                    MAX_BATCH_WRITE_ATTEMPTS
                )));
            }
            tokio::time::sleep(backoff.next_backoff()).await;
        }
        Ok(())
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
        // A single Query returns at most 1MB of items; a long-versioned
        // dataset exceeds that, and the leftover highest-version row would
        // shadow a dataset recreated at the same base_uri. Page through the
        // whole partition instead.
        let mut pages = self
            .ddb_query()
            .key_condition_expression(format!("{} = :{}", base_uri!(), base_uri!()))
            .expression_attribute_values(
                format!(":{}", base_uri!()),
                AttributeValue::S(base_uri.into()),
            )
            .into_paginator()
            .send();

        // The paginator owns the continuation key, which is what keeps an empty
        // terminal key from being read as "there is more".
        while let Some(page) = pages.next().await {
            let page = page.wrap_err()?;
            let items = page.items.unwrap_or_default();
            let delete_requests = page_delete_requests(base_uri, &items)?;

            // DeleteItem is one round trip per row; batch them instead.
            for chunk in delete_requests.chunks(BATCH_WRITE_ITEM_LIMIT) {
                self.batch_delete(chunk).await?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_dynamodb::types::ScalarAttributeType;

    fn attr(name: &str, ty: ScalarAttributeType) -> AttributeDefinition {
        AttributeDefinition::builder()
            .attribute_name(name)
            .attribute_type(ty)
            .build()
            .unwrap()
    }

    /// The key schema check only compares names. A string-typed version key
    /// orders "9" above "12", and the latest-version lookup takes the first row
    /// of a descending scan, so the wrong manifest would be resolved silently.
    #[test]
    fn test_key_attribute_types_are_checked() {
        let good = [
            attr(base_uri!(), ScalarAttributeType::S),
            attr(version!(), ScalarAttributeType::N),
        ];
        validate_key_attribute_types("t", Some(&good)).unwrap();

        let string_version = [
            attr(base_uri!(), ScalarAttributeType::S),
            attr(version!(), ScalarAttributeType::S),
        ];
        let err = validate_key_attribute_types("t", Some(&string_version))
            .expect_err("a string-typed version key must be rejected");
        assert!(
            err.to_string().contains(version!()) && err.to_string().contains("type N"),
            "unexpected message: {err}"
        );

        let numeric_base_uri = [
            attr(base_uri!(), ScalarAttributeType::N),
            attr(version!(), ScalarAttributeType::N),
        ];
        validate_key_attribute_types("t", Some(&numeric_base_uri))
            .expect_err("a number-typed base_uri key must be rejected");

        let only_base_uri = [attr(base_uri!(), ScalarAttributeType::S)];
        validate_key_attribute_types("t", Some(&only_base_uri))
            .expect_err("a missing version definition must be rejected");

        validate_key_attribute_types("t", None)
            .expect_err("a table without attribute definitions must be rejected");
    }

    /// `delete` deletes by key, so a row that does not carry a numeric version
    /// is not one this store wrote and must be left alone.
    #[test]
    fn test_page_delete_requests_uses_rows_with_a_numeric_version() {
        let mut no_version = item(None);
        no_version.remove(version!());
        let mut string_version = item(None);
        string_version.insert(version!().to_string(), AttributeValue::S("7".into()));
        let page = [item(None), no_version, string_version];

        let requests = page_delete_requests("s3://bucket/table", &page).unwrap();
        assert_eq!(requests.len(), 1);
        let key = &requests[0].delete_request().unwrap().key;
        assert_eq!(
            key.get(base_uri!()),
            Some(&AttributeValue::S("s3://bucket/table".into()))
        );
        assert_eq!(key.get(version!()), Some(&AttributeValue::N("7".into())));
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
