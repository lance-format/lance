// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc};

use arrow::datatypes::Int32Type;

use crate::{
    dataset::{
        CommitBuilder, InsertBuilder, ReadParams, WriteMode, WriteParams, builder::DatasetBuilder,
    },
    io::{ObjectStoreParams, StorageOptionsAccessor},
};
use aws_config::{BehaviorVersion, ConfigLoader, Region, SdkConfig};
use aws_sdk_dynamodb::types::AttributeValue;
use aws_sdk_s3::{Client as S3Client, config::Credentials};
use futures::future::try_join_all;
use lance_datagen::{RowCount, array, gen_batch};
use lance_io::assert_io_eq;
use lance_io::utils::tracking_store::IOTracker;

const CONFIG: &[(&str, &str)] = &[
    ("access_key_id", "ACCESS_KEY"),
    ("secret_access_key", "SECRET_KEY"),
    ("endpoint", "http://127.0.0.1:4566"),
    ("dynamodb_endpoint", "http://127.0.0.1:4566"),
    ("allow_http", "true"),
    ("region", "us-east-1"),
];

async fn aws_config() -> SdkConfig {
    let credentials = Credentials::new(CONFIG[0].1, CONFIG[1].1, None, None, "static");
    ConfigLoader::default()
        .credentials_provider(credentials)
        .endpoint_url(CONFIG[2].1)
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(CONFIG[5].1))
        .load()
        .await
}

struct S3Bucket(String);

impl S3Bucket {
    async fn new(bucket: &str) -> Self {
        let config = aws_config().await;
        let client = S3Client::new(&config);

        // In case it wasn't deleted earlier
        Self::delete_bucket(client.clone(), bucket).await;

        client.create_bucket().bucket(bucket).send().await.unwrap();

        Self(bucket.to_string())
    }

    async fn delete_bucket(client: S3Client, bucket: &str) {
        // Before we delete the bucket, we need to delete all objects in it
        let res = client
            .list_objects_v2()
            .bucket(bucket)
            .send()
            .await
            .map_err(|err| err.into_service_error());
        match res {
            Err(e) if e.is_no_such_bucket() => return,
            Err(e) => panic!("Failed to list objects in bucket: {}", e),
            _ => {}
        }
        let objects = res.unwrap().contents.unwrap_or_default();
        for object in objects {
            client
                .delete_object()
                .bucket(bucket)
                .key(object.key.unwrap())
                .send()
                .await
                .unwrap();
        }
        client.delete_bucket().bucket(bucket).send().await.unwrap();
    }
}

impl Drop for S3Bucket {
    fn drop(&mut self) {
        let bucket_name = self.0.clone();
        tokio::task::spawn(async move {
            let config = aws_config().await;
            let client = S3Client::new(&config);
            Self::delete_bucket(client, &bucket_name).await;
        });
    }
}

struct DynamoDBCommitTable(String);

impl DynamoDBCommitTable {
    async fn new(name: &str) -> Self {
        let config = aws_config().await;
        let client = aws_sdk_dynamodb::Client::new(&config);

        // In case it wasn't deleted earlier
        Self::delete_table(client.clone(), name).await;
        // Dynamodb table drop is async, so we need to wait a bit
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        use aws_sdk_dynamodb::types::*;

        client
            .create_table()
            .table_name(name)
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("base_uri")
                    .attribute_type(ScalarAttributeType::S)
                    .build()
                    .unwrap(),
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("version")
                    .attribute_type(ScalarAttributeType::N)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("base_uri")
                    .key_type(KeyType::Hash)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("version")
                    .key_type(KeyType::Range)
                    .build()
                    .unwrap(),
            )
            .provisioned_throughput(
                ProvisionedThroughput::builder()
                    .read_capacity_units(1)
                    .write_capacity_units(1)
                    .build()
                    .unwrap(),
            )
            .send()
            .await
            .unwrap();

        Self(name.to_string())
    }

    async fn item_for_version(&self, expected_version: u64) -> HashMap<String, AttributeValue> {
        let config = aws_config().await;
        let client = aws_sdk_dynamodb::Client::new(&config);
        let expected_version_string = expected_version.to_string();
        client
            .scan()
            .table_name(&self.0)
            .consistent_read(true)
            .send()
            .await
            .unwrap()
            .items
            .unwrap_or_default()
            .into_iter()
            .find(|item| {
                item.get("version").and_then(|value| value.as_n().ok())
                    == Some(&expected_version_string)
            })
            .unwrap_or_else(|| panic!("DynamoDB row for version {expected_version} not found"))
    }

    async fn set_legacy_etag(&self, version: u64, e_tag: &str) {
        let item = self.item_for_version(version).await;
        let base_uri = item
            .get("base_uri")
            .and_then(|value| value.as_s().ok())
            .expect("DynamoDB row must contain a string base_uri")
            .clone();
        let config = aws_config().await;
        aws_sdk_dynamodb::Client::new(&config)
            .update_item()
            .table_name(&self.0)
            .key("base_uri", AttributeValue::S(base_uri))
            .key("version", AttributeValue::N(version.to_string()))
            .update_expression("SET e_tag = :e_tag")
            .expression_attribute_values(":e_tag", AttributeValue::S(e_tag.to_string()))
            .send()
            .await
            .unwrap();
    }

    async fn delete_table(client: aws_sdk_dynamodb::Client, name: &str) {
        match client
            .delete_table()
            .table_name(name)
            .send()
            .await
            .map_err(|err| err.into_service_error())
        {
            Ok(_) => {}
            Err(e) if e.is_resource_not_found_exception() => {}
            Err(e) => panic!("Failed to delete table: {}", e),
        };
    }
}

impl Drop for DynamoDBCommitTable {
    fn drop(&mut self) {
        let table_name = self.0.clone();
        tokio::task::spawn(async move {
            let config = aws_config().await;
            let client = aws_sdk_dynamodb::Client::new(&config);
            Self::delete_table(client, &table_name).await;
        });
    }
}

#[tokio::test]
async fn test_concurrent_writers() {
    let datagen = gen_batch().col("values", array::step::<Int32Type>());
    let data = datagen.into_batch_rows(RowCount::from(100)).unwrap();

    // We want to track IOs prior to creating the dataset, so need to explicitly create the tracker
    let io_tracker = Arc::new(IOTracker::default());

    // Create a table
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(io_tracker.clone()),
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            CONFIG
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ))),
        ..Default::default()
    };
    let write_params = WriteParams {
        store_params: Some(store_params.clone()),
        mode: WriteMode::Append,
        ..Default::default()
    };
    let bucket = S3Bucket::new("test-concurrent-writers").await;
    let uri = format!("s3://{}/test", bucket.0);
    let transaction = InsertBuilder::new(&uri)
        .with_params(&write_params)
        .execute_uncommitted(vec![data.clone()])
        .await
        .unwrap();

    // 1 IOPS for uncommitted write
    let io_stats = io_tracker.incremental_stats();
    assert_io_eq!(io_stats, write_iops, 1);

    let dataset = CommitBuilder::new(&uri)
        .with_store_params(store_params.clone())
        .execute(transaction)
        .await
        .unwrap();
    // Commit: 2 IOPs. 1 for transaction file, 1 for manifest file
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, write_iops, 2);
    let dataset = Arc::new(dataset);
    let old_version = dataset.manifest().version;

    let concurrency = 10;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let ds_ref = dataset.clone();
        let data_ref = data.clone();
        let task = tokio::spawn(async move {
            InsertBuilder::new(ds_ref)
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute(vec![data_ref])
                .await
                .unwrap();
        });
        tasks.push(task);
    }
    try_join_all(tasks).await.unwrap();

    let mut dataset = dataset.as_ref().clone();
    dataset.checkout_latest().await.unwrap();
    assert_eq!(old_version + concurrency as u64, dataset.manifest().version);

    let num_rows = dataset.count_rows(None).await.unwrap();
    assert_eq!(num_rows, data.num_rows() * (concurrency + 1));

    dataset.validate().await.unwrap();
    let half_rows = dataset
        .count_rows(Some("values >= 50".into()))
        .await
        .unwrap();
    assert_eq!(half_rows, num_rows / 2);
}

#[tokio::test]
async fn test_ddb_open_iops() {
    let bucket = S3Bucket::new("test-ddb-iops").await;
    let ddb_table = DynamoDBCommitTable::new("test-ddb-iops").await;
    let uri = format!("s3+ddb://{}/test?ddbTableName={}", bucket.0, ddb_table.0);

    let datagen = gen_batch().col("values", array::step::<Int32Type>());
    let data = datagen.into_batch_rows(RowCount::from(100)).unwrap();

    let io_tracker = Arc::new(IOTracker::default());

    // Create a table
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(io_tracker.clone()),
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            CONFIG
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ))),
        ..Default::default()
    };
    let write_params = WriteParams {
        store_params: Some(store_params.clone()),
        mode: WriteMode::Append,
        ..Default::default()
    };
    let transaction = InsertBuilder::new(&uri)
        .with_params(&write_params)
        .execute_uncommitted(vec![data.clone()])
        .await
        .unwrap();

    // 1 IOPS for uncommitted write
    let io_stats = io_tracker.incremental_stats();
    assert_io_eq!(io_stats, write_iops, 1);

    let committed_ds = CommitBuilder::new(&uri)
        .with_store_params(store_params.clone())
        .execute(transaction)
        .await
        .unwrap();
    // Commit: 4 write IOPs:
    // * 1 for transaction file
    // * 3 for manifest file
    //    * write staged file
    //    * copy to final file
    //    * delete staged file
    // Commit: 2 read IOPs: one to list versions before creating the dataset and
    // one to HEAD the canonical manifest after COPY. DynamoDB does not persist
    // that ETag, but the freshly committed Dataset needs the observed physical
    // generation so downstream caches cannot reuse an older Dataset at the
    // same URI and version.
    let io_stats = committed_ds.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, write_iops, 4);
    assert_io_eq!(io_stats, read_iops, 2);
    assert!(committed_ds.manifest_location().e_tag.is_some());

    let committed_row = ddb_table.item_for_version(1).await;
    assert!(
        !committed_row.contains_key("e_tag"),
        "DynamoDB must not persist a physical object generation"
    );
    // Simulate a row written by an older Lance version. New DynamoDB readers
    // ignore this physical-generation token instead of treating it as logical
    // manifest identity.
    ddb_table.set_legacy_etag(1, "legacy-stale-etag").await;

    let dataset = DatasetBuilder::from_uri(&uri)
        .with_read_params(ReadParams {
            store_options: Some(store_params.clone()),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    assert_ne!(
        dataset.manifest_location().e_tag.as_deref(),
        Some("legacy-stale-etag")
    );
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    // Open dataset can be read with 2 IOPs: HEAD verifies that the
    // finalized path returned by DynamoDB still exists, then the manifest
    // itself is read. Looking up the latest manifest is handled in DynamoDB.
    assert_io_eq!(io_stats, read_iops, 2);
    assert_io_eq!(io_stats, write_iops, 0);

    // Append
    let dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![data.clone()])
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    // Append: 5 IOPS: data file, transaction file, 3x manifest file
    assert_io_eq!(io_stats, write_iops, 5);
    // Append reads once to list versions and once to observe the canonical
    // generation after COPY. DDB stores only the stable final path and size;
    // the returned Dataset retains the observed ETag.
    // TODO: we can reduce this by implementing a specialized CommitHandler::list_manifest_locations()
    // for the DDB commit handler.
    assert_io_eq!(io_stats, read_iops, 2);
    assert!(dataset.manifest_location().e_tag.is_some());

    // Checkout original version
    dataset.checkout_version(1).await.unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    // Checkout: 1 read IOP. Version resolution HEAD-checks the finalized path,
    // while the manifest body is served from the Session metadata cache.
    assert_io_eq!(io_stats, read_iops, 1);
    assert_io_eq!(io_stats, write_iops, 0);
}
