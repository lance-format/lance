use std::collections::HashMap;
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction, UpdateMode, UpdatedFragmentOffsets};
use crate::dataset::{
    ColumnAlteration, ManifestWriteConfig, NewColumnTransform, TRANSACTIONS_DIR,
    write_manifest_file,
};
use crate::io::ObjectStoreParams;
use crate::session::Session;
use crate::{Dataset, Result};
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::ManifestNamingScheme;

use crate::dataset::write::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Int32Array, RecordBatchIterator, StringArray, StructArray,
    types::{Int32Type, Int64Type},
};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use lance_core::Error;
use lance_core::datatypes::{Field as LanceCoreField, LogicalType, Schema as LanceSchema};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::{TempDir, TempStrDir};
use lance_core::{ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};
use lance_datagen::{BatchCount, RowCount, array};

use crate::datafusion::LanceTableProvider;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance_datafusion::udf::register_functions;
use object_store::ObjectStoreExt;

#[tokio::test]
async fn test_read_transaction_properties() {
    const LANCE_COMMIT_MESSAGE_KEY: &str = "__lance_commit_message";
    // Create a test dataset
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("value", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let test_uri = TempStrDir::default();

    // Create WriteParams with properties
    let mut properties1 = HashMap::new();
    properties1.insert(
        LANCE_COMMIT_MESSAGE_KEY.to_string(),
        "First commit".to_string(),
    );
    properties1.insert("custom_prop".to_string(), "custom_value".to_string());

    let write_params = WriteParams {
        transaction_properties: Some(Arc::new(properties1)),
        ..Default::default()
    };

    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
        &test_uri,
        Some(write_params),
    )
    .await
    .unwrap();

    let transaction = dataset.read_transaction_by_version(1).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"First commit".to_string())
    );
    assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

    let mut properties2 = HashMap::new();
    properties2.insert(
        LANCE_COMMIT_MESSAGE_KEY.to_string(),
        "Second commit".to_string(),
    );
    properties2.insert("another_prop".to_string(), "another_value".to_string());

    let write_params = WriteParams {
        transaction_properties: Some(Arc::new(properties2)),
        mode: WriteMode::Append,
        ..Default::default()
    };

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(StringArray::from(vec!["d", "e"])),
        ],
    )
    .unwrap();

    let mut dataset = dataset;
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch2)], schema.clone()),
            Some(write_params),
        )
        .await
        .unwrap();

    let transaction = dataset.read_transaction_by_version(2).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"Second commit".to_string())
    );
    assert_eq!(
        props.get("another_prop"),
        Some(&"another_value".to_string())
    );

    let transaction = dataset.read_transaction_by_version(1).await.unwrap();
    assert!(transaction.is_some());
    let props = transaction.unwrap().transaction_properties.unwrap();
    assert_eq!(props.len(), 2);
    assert_eq!(
        props.get(LANCE_COMMIT_MESSAGE_KEY),
        Some(&"First commit".to_string())
    );
    assert_eq!(props.get("custom_prop"), Some(&"custom_value".to_string()));

    let result = dataset.read_transaction_by_version(999).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_session_store_registry() {
    // Create a session
    let session = Arc::new(Session::default());
    let registry = session.store_registry();
    assert!(registry.active_stores().is_empty());

    // Create a dataset with memory store
    let write_params = WriteParams {
        session: Some(session.clone()),
        ..Default::default()
    };
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let dataset = InsertBuilder::new("memory://test")
        .with_params(&write_params)
        .execute(vec![batch.clone()])
        .await
        .unwrap();

    // Assert there is one active store.
    assert_eq!(registry.active_stores().len(), 1);

    // If we create another dataset also in memory, it should re-use the
    // existing store.
    let dataset2 = InsertBuilder::new("memory://test2")
        .with_params(&write_params)
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    assert_eq!(registry.active_stores().len(), 1);
    assert_eq!(
        Arc::as_ptr(&dataset.object_store.as_ref().inner),
        Arc::as_ptr(&dataset2.object_store.as_ref().inner)
    );

    // If we create another with **different parameters**, it should create a new store.
    let write_params2 = WriteParams {
        session: Some(session.clone()),
        store_params: Some(ObjectStoreParams {
            block_size: Some(10_000),
            ..Default::default()
        }),
        ..Default::default()
    };
    let dataset3 = InsertBuilder::new("memory://test3")
        .with_params(&write_params2)
        .execute(vec![batch.clone()])
        .await
        .unwrap();
    assert_eq!(registry.active_stores().len(), 2);
    assert_ne!(
        Arc::as_ptr(&dataset.object_store.as_ref().inner),
        Arc::as_ptr(&dataset3.object_store.as_ref().inner)
    );

    // Remove both datasets
    drop(dataset3);
    assert_eq!(registry.active_stores().len(), 1);
    drop(dataset2);
    drop(dataset);
    assert_eq!(registry.active_stores().len(), 0);
}

#[tokio::test]
async fn test_migrate_v2_manifest_paths() {
    let test_uri = TempStrDir::default();

    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &test_uri,
        Some(WriteParams {
            enable_v2_manifest_paths: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        dataset.manifest_location().naming_scheme,
        ManifestNamingScheme::V1
    );

    dataset.migrate_manifest_paths_v2().await.unwrap();
    assert_eq!(
        dataset.manifest_location().naming_scheme,
        ManifestNamingScheme::V2
    );
}

pub(super) async fn execute_sql(
    sql: &str,
    table: String,
    dataset: Arc<Dataset>,
) -> Result<Vec<RecordBatch>> {
    let ctx = SessionContext::new();
    ctx.register_table(
        table,
        Arc::new(LanceTableProvider::new(dataset, false, false)),
    )?;
    register_functions(&ctx);

    let df = ctx.sql(sql).await?;
    Ok(df
        .execute_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await?)
}

pub(super) fn assert_results<T: Array + PartialEq + 'static>(
    results: Vec<RecordBatch>,
    values: &T,
) {
    assert_eq!(results.len(), 1);
    let results = results.into_iter().next().unwrap();
    assert_eq!(results.num_columns(), 1);

    assert_eq!(
        results.column(0).as_any().downcast_ref::<T>().unwrap(),
        values
    )
}

fn gen_rows() -> impl arrow_array::RecordBatchReader + Send + 'static {
    lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1))
}

/// Write a dataset with `versions` versions of 10 rows each.
async fn write_versions(uri: &str, versions: usize, enable_v2_manifest_paths: bool) -> Dataset {
    let mut ds = Dataset::write(
        gen_rows(),
        uri,
        Some(WriteParams {
            enable_v2_manifest_paths,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for _ in 1..versions {
        ds.append(
            gen_rows(),
            Some(WriteParams {
                mode: WriteMode::Append,
                enable_v2_manifest_paths,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }
    ds
}

#[tokio::test]
async fn test_inline_transaction() {
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    async fn create_dataset(rows: i32) -> Arc<Dataset> {
        let dir = TempDir::default();
        let uri = dir.path_str();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..rows))],
        )
        .unwrap();
        let ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri.as_str(),
            None,
        )
        .await
        .unwrap();
        Arc::new(ds)
    }

    fn make_tx(read_version: u64) -> Transaction {
        Transaction::new(read_version, Operation::Append { fragments: vec![] }, None)
    }

    async fn delete_external_tx_file(ds: &Dataset) {
        if let Some(tx_file) = ds.manifest.transaction_file.as_ref() {
            let tx_path = ds
                .base
                .clone()
                .join(TRANSACTIONS_DIR)
                .join(tx_file.as_str());
            let _ = ds.object_store.inner.delete(&tx_path).await; // ignore errors
        }
    }

    let session = Arc::new(Session::default());

    // Case 1: Default write_flag=true, delete external transaction file, read should use inline transaction
    let ds = create_dataset(5).await;
    let read_version = ds.manifest().version;
    let tx = make_tx(read_version);
    let ds2 = CommitBuilder::new(ds.clone())
        .execute(tx.clone())
        .await
        .unwrap();
    delete_external_tx_file(&ds2).await;
    let read_tx = ds2.read_transaction().await.unwrap().unwrap();
    assert_eq!(read_tx, tx.clone());

    // Case 2: reading small manifest caches transaction data, eliminating transaction reading IO.
    let read_ds2 = DatasetBuilder::from_uri(ds2.uri.clone())
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    let stats = read_ds2.object_store.as_ref().io_stats_incremental(); // Reset
    assert!(stats.read_bytes < 64 * 1024);
    // Because the manifest is so small, we should have opportunistically
    // cached the transaction in memory already.
    let inline_tx = read_ds2.read_transaction().await.unwrap().unwrap();
    let stats = read_ds2.object_store.as_ref().io_stats_incremental();
    assert_eq!(stats.read_iops, 0);
    assert_eq!(stats.read_bytes, 0);
    assert_eq!(inline_tx, tx);

    // Case 3: manifest does not contain inline transaction, read should fall back to external transaction file
    let ds = create_dataset(2).await;
    let tx = make_tx(ds.manifest().version);
    let tx_file = crate::io::commit::write_transaction_file(
        ds.object_store.as_ref(),
        &ds.base,
        &lance_table::format::pb::Transaction::from(&tx),
    )
    .await
    .unwrap();
    let (mut manifest, indices) = tx
        .build_manifest(
            Some(ds.manifest.as_ref()),
            ds.load_indices().await.unwrap().as_ref().clone(),
            &tx_file,
            &ManifestWriteConfig::default().to_build_config(),
        )
        .unwrap();
    let location = write_manifest_file(
        ds.object_store.as_ref(),
        ds.commit_handler.as_ref(),
        &ds.base,
        &mut manifest,
        if indices.is_empty() {
            None
        } else {
            Some(indices.clone())
        },
        &ManifestWriteConfig::default(),
        ds.manifest_location.naming_scheme,
        None,
    )
    .await
    .unwrap();
    let ds_new = ds.checkout_version(location.version).await.unwrap();
    assert!(ds_new.manifest.transaction_section.is_none());
    assert!(ds_new.manifest.transaction_file.is_some());
    let read_tx = ds_new.read_transaction().await.unwrap().unwrap();
    assert_eq!(read_tx, tx);

    // The direct read takes the same external-file fallback.
    let version_transaction = ds_new
        .read_version_transaction(location.version)
        .await
        .unwrap();
    assert_eq!(version_transaction.transaction, Some(tx));
}

#[tokio::test]
async fn test_read_version_transaction_does_not_populate_caches() {
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    let test_uri = TempStrDir::default();
    let mut dataset = write_versions(&test_uri, 1, true).await;
    // Index the table so historical manifests carry an IndexSection that a
    // caching read path would decode.
    dataset
        .create_index(
            &["key"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap(); // version 2
    for _ in 0..18 {
        dataset
            .append(
                gen_rows(),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }
    let latest_version = dataset.version().version;
    assert_eq!(latest_version, 20);

    // Fresh session so any cache insertion by the API under test shows as growth.
    let session = Arc::new(Session::default());
    let dataset = DatasetBuilder::from_uri(&test_uri)
        .with_session(session.clone())
        .load()
        .await
        .unwrap();

    let metadata_stats_before = session.metadata_cache_stats().await;
    let index_stats_before = session.index_cache_stats().await;

    let mut actual = Vec::with_capacity(latest_version as usize);
    for version in 1..=latest_version {
        let version_transaction = dataset.read_version_transaction(version).await.unwrap();
        assert_eq!(version_transaction.version, version);
        actual.push(version_transaction);
    }

    let metadata_stats_after = session.metadata_cache_stats().await;
    let index_stats_after = session.index_cache_stats().await;
    assert_eq!(
        metadata_stats_after.num_entries,
        metadata_stats_before.num_entries
    );
    assert_eq!(
        metadata_stats_after.size_bytes,
        metadata_stats_before.size_bytes
    );
    assert_eq!(
        index_stats_after.num_entries,
        index_stats_before.num_entries
    );
    assert_eq!(index_stats_after.size_bytes, index_stats_before.size_bytes);

    // Results match a full checkout.
    for version_transaction in &actual {
        let checked_out = dataset
            .checkout_version(version_transaction.version)
            .await
            .unwrap();
        assert_eq!(
            version_transaction.transaction,
            checked_out.read_transaction().await.unwrap()
        );
        assert_eq!(
            version_transaction.timestamp,
            checked_out.version().timestamp
        );
        assert!(version_transaction.transaction.is_some());
    }

    // A missing (e.g. cleaned up) version errors as DatasetNotFound, matching
    // the historical checkout_version-based contract of the public API.
    let err = dataset.read_version_transaction(9999).await.unwrap_err();
    assert!(
        matches!(err, crate::Error::DatasetNotFound { .. }),
        "expected DatasetNotFound for a missing version, got {err:?}"
    );
}

#[tokio::test]
async fn test_read_transaction_recovers_from_stale_manifest_size() {
    let test_uri = TempStrDir::default();
    let ds = write_versions(&test_uri, 1, true).await;
    let manifest = ds.manifest().clone();
    // Only meaningful for the inline path; a plain write inlines the transaction.
    assert!(manifest.transaction_section.is_some());

    // A size at/under the transaction offset makes the first read_message fail
    // "file size is too small"; only the retry at the true size can recover.
    let mut stale = ds.manifest_location().clone();
    stale.size = Some(1);
    let recovered = ds
        .read_transaction_from_storage(&manifest, &stale)
        .await
        .unwrap();
    assert_eq!(recovered, ds.read_transaction().await.unwrap());
    assert!(recovered.is_some());
}

#[tokio::test]
async fn test_read_version_transaction_v1_manifest_naming() {
    let test_uri = TempStrDir::default();
    let ds = write_versions(&test_uri, 3, false).await;
    assert_eq!(
        ds.manifest_location().naming_scheme,
        ManifestNamingScheme::V1
    );

    for version in 1..=3 {
        let version_transaction = ds.read_version_transaction(version).await.unwrap();
        let checked_out = ds.checkout_version(version).await.unwrap();
        assert_eq!(
            version_transaction.transaction,
            checked_out.read_transaction().await.unwrap()
        );
        assert_eq!(
            version_transaction.timestamp,
            checked_out.version().timestamp
        );
    }
}

#[tokio::test]
async fn test_read_version_transaction_on_branch() {
    let test_uri = TempStrDir::default();
    let mut main_ds = write_versions(&test_uri, 1, true).await;
    let branch_ds = main_ds.create_branch("dev", 1, None).await.unwrap();

    // Commit on the branch.
    let branch_ds = Dataset::write(
        gen_rows(),
        branch_ds.uri(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(branch_ds.manifest().branch.as_deref(), Some("dev"));

    // Versions resolve against the branch chain and match a full checkout.
    for version in branch_ds.versions().await.unwrap() {
        let version_transaction = branch_ds
            .read_version_transaction(version.version)
            .await
            .unwrap();
        assert_eq!(version_transaction.version, version.version);
        assert_eq!(version_transaction.timestamp, version.timestamp);
        let checked_out = branch_ds.checkout_version(version.version).await.unwrap();
        assert_eq!(checked_out.manifest().branch.as_deref(), Some("dev"));
        assert_eq!(
            version_transaction.transaction,
            checked_out.read_transaction().await.unwrap()
        );
    }

    // The append on the branch is the branch's own transaction.
    let latest = branch_ds.version().version;
    let version_transaction = branch_ds.read_version_transaction(latest).await.unwrap();
    assert!(matches!(
        version_transaction.transaction,
        Some(Transaction {
            operation: Operation::Append { .. },
            ..
        })
    ));
}

#[tokio::test]
async fn test_list_detached_manifests() {
    let test_uri = TempStrDir::default();

    // Create initial dataset
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
            &test_uri,
            None,
        )
        .await
        .unwrap(),
    );

    // Initially there should be no detached manifests
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert!(detached.is_empty());

    // Create a detached transaction with properties
    let mut properties = HashMap::new();
    properties.insert("detached_key".to_string(), "detached_value".to_string());

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
    )
    .unwrap();

    // Use execute_uncommitted + CommitBuilder with_detached(true)
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            transaction_properties: Some(Arc::new(properties.clone())),
            ..Default::default()
        })
        .execute_uncommitted(vec![batch2])
        .await
        .unwrap();

    CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(transaction)
        .await
        .unwrap();

    // Now there should be one detached manifest
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert_eq!(detached.len(), 1);
    assert_eq!(
        dataset
            .version_refs()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![1]
    );

    // The detached version should have the high bit set
    let detached_version = detached[0].version;
    assert!(lance_table::format::is_detached_version(detached_version));

    // We should be able to checkout the detached version and read transaction properties
    let checked_out = dataset.checkout_version(detached_version).await.unwrap();
    let tx = checked_out.read_transaction().await.unwrap().unwrap();
    let tx_props = tx.transaction_properties.unwrap();
    assert_eq!(
        tx_props.get("detached_key"),
        Some(&"detached_value".to_string())
    );

    // The detached dataset should have more rows
    assert_eq!(checked_out.count_rows(None).await.unwrap(), 6);

    // Create another detached transaction
    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![7, 8, 9]))],
    )
    .unwrap();

    let mut properties2 = HashMap::new();
    properties2.insert("second_key".to_string(), "second_value".to_string());

    let transaction2 = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            transaction_properties: Some(Arc::new(properties2)),
            ..Default::default()
        })
        .execute_uncommitted(vec![batch3])
        .await
        .unwrap();

    CommitBuilder::new(dataset.clone())
        .with_detached(true)
        .execute(transaction2)
        .await
        .unwrap();

    // Now there should be two detached manifests
    let detached = dataset.list_detached_manifests().await.unwrap();
    assert_eq!(detached.len(), 2);

    // Both should be detached versions
    for loc in &detached {
        assert!(lance_table::format::is_detached_version(loc.version));
    }

    // Regular versions() should not include detached manifests
    let versions = dataset.versions().await.unwrap();
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0].version, 1);
}

/// Transaction properties large enough to push the transaction over the
/// inline threshold.
fn large_props(key: &str) -> Option<Arc<HashMap<String, String>>> {
    use crate::io::commit::MAX_INLINE_TRANSACTION_BYTES;
    let mut props = HashMap::new();
    props.insert(
        key.to_string(),
        "x".repeat(2 * MAX_INLINE_TRANSACTION_BYTES),
    );
    Some(Arc::new(props))
}

fn spill_test_batch() -> (Arc<ArrowSchema>, RecordBatch) {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..10))],
    )
    .unwrap();
    (schema, batch)
}

/// Load the dataset with a fresh session so assertions hit storage, not caches.
async fn reopen(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_large_transaction_spills_to_external_file() {
    use crate::io::commit::MAX_INLINE_TRANSACTION_BYTES;

    let (schema, batch) = spill_test_batch();
    let test_uri = TempStrDir::default();

    // New-dataset commit path: a transaction too large to inline is written
    // only to the external transaction file.
    Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
        &test_uri,
        Some(WriteParams {
            transaction_properties: large_props("payload"),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let ds = reopen(&test_uri).await;
    assert!(ds.manifest.transaction_section.is_none());
    assert!(matches!(ds.manifest.transaction_file.as_deref(), Some(f) if !f.is_empty()));
    let tx = ds.read_transaction().await.unwrap().unwrap();
    assert_eq!(
        tx.transaction_properties
            .unwrap()
            .get("payload")
            .unwrap()
            .len(),
        2 * MAX_INLINE_TRANSACTION_BYTES
    );

    // Normal commit path: a small append is still inlined.
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let ds = reopen(&test_uri).await;
    assert!(ds.manifest.transaction_section.is_some());
}

#[tokio::test]
async fn test_spilled_restore_and_deep_clone_read_own_transaction() {
    // Restore and deep clone both rebuild the new manifest from an existing
    // manifest file, inheriting its inline transaction offset and external
    // transaction file name. When the new transaction is too large to inline,
    // readers fall back to exactly those fields, so the stale inherited values
    // must not leak into the new manifest.
    use crate::dataset::transaction::TransactionBuilder;

    let (schema, batch) = spill_test_batch();
    let source_uri = TempStrDir::default();
    Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
        &source_uri,
        None,
    )
    .await
    .unwrap();
    let ds = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &source_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // The manifests restored from / cloned from below carry an inline
    // transaction offset and their own transaction file.
    assert!(ds.manifest.transaction_section.is_some());
    let source_version = ds.manifest().version;
    let source_ref_path = ds.uri().to_string();
    let source_tx_file = ds.manifest.transaction_file.clone();
    assert!(matches!(source_tx_file.as_deref(), Some(f) if !f.is_empty()));

    // Restore with a spilled transaction: the stale inline offset must be
    // cleared and the transaction read back from the external file.
    let restore_tx = TransactionBuilder::new(source_version, Operation::Restore { version: 1 })
        .transaction_properties(large_props("payload"))
        .build();
    CommitBuilder::new(Arc::new(ds))
        .execute(restore_tx.clone())
        .await
        .unwrap();
    let restored = reopen(&source_uri).await;
    assert!(restored.manifest.transaction_section.is_none());
    assert_eq!(
        restored.read_transaction().await.unwrap().unwrap(),
        restore_tx
    );

    // Deep clone with a spilled transaction: the manifest must reference the
    // clone's own transaction file, not the source's.
    let clone_tx = TransactionBuilder::new(
        source_version,
        Operation::Clone {
            is_shallow: false,
            ref_name: None,
            ref_version: source_version,
            ref_path: source_ref_path,
            branch_name: None,
        },
    )
    .transaction_properties(large_props("payload"))
    .build();
    let clone_uri = TempStrDir::default();
    CommitBuilder::new(&clone_uri)
        .execute(clone_tx.clone())
        .await
        .unwrap();
    let cloned = reopen(&clone_uri).await;
    assert!(cloned.manifest.transaction_section.is_none());
    assert_ne!(cloned.manifest.transaction_file, source_tx_file);
    assert_eq!(cloned.read_transaction().await.unwrap().unwrap(), clone_tx);
}

/// Partial RewriteColumns refresh in `build_manifest`: only matched physical
/// rows get `last_updated_at_version` bumped; same-fragment unmatched rows and
/// untouched fragments keep both version sequences.
#[tokio::test]
async fn test_build_manifest_partial_last_updated_rewrite_columns_stable_row_ids() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new("x", DataType::Int32, false),
    ]));
    let batch0 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..8)),
            Arc::new(Int32Array::from(vec![0_i32; 8])),
        ],
    )
    .unwrap();
    let reader0 = RecordBatchIterator::new(vec![Ok(batch0)], schema.clone());
    let write_params = WriteParams {
        enable_stable_row_ids: true,
        data_storage_version: Some(LanceFileVersion::Stable),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader0, uri, Some(write_params))
        .await
        .unwrap();

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(100..108)),
            Arc::new(Int32Array::from(vec![0_i32; 8])),
        ],
    )
    .unwrap();
    let reader1 = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
    dataset.append(reader1, None).await.unwrap();

    let frags = dataset.get_fragments();
    assert_eq!(
        frags.len(),
        2,
        "expected two fragments (append creates a new fragment)"
    );

    async fn scan_row_versions(ds: &Dataset) -> HashMap<(u32, u32), (u64, u64)> {
        let mut scanner = ds.scan();
        scanner
            .project(&[
                ROW_ADDR,
                ROW_LAST_UPDATED_AT_VERSION,
                ROW_CREATED_AT_VERSION,
            ])
            .unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut out = HashMap::new();
        for batch in batches {
            let addrs = batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_primitive::<UInt64Type>();
            let last = batch
                .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
                .unwrap()
                .as_primitive::<UInt64Type>();
            let created = batch
                .column_by_name(ROW_CREATED_AT_VERSION)
                .unwrap()
                .as_primitive::<UInt64Type>();
            for row in 0..batch.num_rows() {
                let addr = RowAddress::from(addrs.value(row));
                out.insert(
                    (addr.fragment_id(), addr.row_offset()),
                    (last.value(row), created.value(row)),
                );
            }
        }
        out
    }

    let before = scan_row_versions(&dataset).await;
    assert_eq!(before.len(), 16);

    // Update only rows i in {2, 4, 6} within fragment 0 (physical offsets 2, 4, 6).
    let update_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new("x", DataType::Int32, false),
    ]));
    let update_batch = RecordBatch::try_new(
        update_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2, 4, 6])),
            Arc::new(Int32Array::from(vec![99, 99, 99])),
        ],
    )
    .unwrap();
    let right: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(update_batch)].into_iter(),
        update_schema,
    ));

    let mut frag0 = dataset.get_fragment(0).unwrap();
    let u = frag0
        .update_columns_with_offsets(right, "i", "i")
        .await
        .unwrap();
    assert_eq!(u.matched_offsets.iter().count(), 3);
    for off in [2_u32, 4, 6] {
        assert!(u.matched_offsets.contains(off));
    }

    let updated_fragment_offsets = Some(UpdatedFragmentOffsets(HashMap::from([(
        u.fragment.id,
        u.matched_offsets,
    )])));

    let op = Operation::Update {
        removed_fragment_ids: vec![],
        updated_fragments: vec![u.fragment],
        new_fragments: vec![],
        fields_modified: u.fields_modified,
        compacted_sstables: Vec::new(),
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteColumns),
        inserted_rows_filter: None,
        updated_fragment_offsets,
    };

    let read_v = dataset.version().version;
    let dataset = Dataset::commit(
        uri,
        op,
        Some(read_v),
        None,
        None,
        Arc::new(Session::default()),
        true,
    )
    .await
    .unwrap();

    let new_v = dataset.version().version;
    assert_eq!(new_v, read_v + 1);

    let after = scan_row_versions(&dataset).await;
    for off in 0..8_u32 {
        let key = (0, off);
        let (last_before, created_before) = before[&key];
        let (last_after, created_after) = after[&key];
        assert_eq!(created_after, created_before);
        if off == 2 || off == 4 || off == 6 {
            assert_eq!(
                last_after, new_v,
                "matched row offset {off} should advance last_updated to new version"
            );
        } else {
            assert_eq!(
                last_after, last_before,
                "unmatched row offset {off} in fragment 0 should keep last_updated"
            );
        }
    }

    for off in 0..8_u32 {
        let key = (1, off);
        assert_eq!(
            after[&key], before[&key],
            "fragment 1 row offset {off}: both version columns unchanged"
        );
    }
}

/// Repro shape for issue 7700: write a, b, c; drop one column; add d. The
/// dropped id stays referenced by the data files and max_field_id stays 3.
async fn dataset_with_dropped_column(uri: &str, dropped: &str) -> Dataset {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
        ArrowField::new("c", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![10, 20])),
            Arc::new(Int32Array::from(vec![100, 200])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema);
    let mut dataset = Dataset::write(reader, uri, None).await.unwrap();
    dataset.drop_columns(&[dropped]).await.unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("d".into(), "CAST(5 AS INT)".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let dropped_id = ["a", "b", "c"].iter().position(|c| *c == dropped).unwrap() as i32;
    let mut expected_ids: Vec<i32> = (0..3).filter(|id| *id != dropped_id).collect();
    expected_ids.push(3);
    assert_eq!(dataset.schema().field_ids(), expected_ids);
    assert_eq!(dataset.manifest.max_field_id(), 3);
    dataset
}

/// Expected values of every column surviving `dropped`, plus d.
fn surviving_columns(dropped: &str) -> Vec<(&'static str, [i32; 2])> {
    [
        ("a", [1, 2]),
        ("b", [10, 20]),
        ("c", [100, 200]),
        ("d", [5, 5]),
    ]
    .into_iter()
    .filter(|(name, _)| *name != dropped)
    .collect()
}

fn assert_columns(batch: &RecordBatch, cols: &[(&str, [i32; 2])]) {
    for (name, expected) in cols {
        let col = &batch[*name];
        assert_eq!(
            col.as_primitive::<Int32Type>().values(),
            expected,
            "column {}",
            name
        );
    }
}

async fn commit_merge(dataset: &Dataset, schema: LanceSchema) -> Result<Dataset> {
    let fragments = dataset
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect();
    Dataset::commit(
        Arc::new(dataset.clone()),
        Operation::Merge {
            fragments,
            schema,
            preserves_nullability: true,
        },
        Some(dataset.manifest.version),
        None,
        None,
        dataset.session(),
        false,
    )
    .await
}

// Which clause rejects the lossy round-trip depends on the hole's
// position: a hole before the last field remaps a shared id, while a
// hole at the end reuses the dropped id for the new field.
#[rstest::rstest]
#[case::drop_a_remaps_shared_id("a", "remaps field id 1 from \"b\" to \"c\"")]
#[case::drop_b_remaps_shared_id("b", "remaps field id 2 from \"c\" to \"d\"")]
#[case::drop_c_reuses_dropped_id("c", "assigns id 2 to new field \"d\"")]
#[tokio::test]
async fn test_merge_rejects_renumbered_field_ids(#[case] dropped: &str, #[case] expected: &str) {
    let dataset = dataset_with_dropped_column("memory://", dropped).await;

    let arrow_schema = ArrowSchema::from(dataset.schema());
    let renumbered = LanceSchema::try_from(&arrow_schema).unwrap();
    assert_eq!(renumbered.field_ids(), vec![0, 1, 2]);

    let err = commit_merge(&dataset, renumbered).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
    let message = err.to_string();
    assert!(message.contains(expected), "unexpected error: {}", message);
}

#[tokio::test]
async fn test_merge_rejects_dropped_field_id_reuse() {
    // Deliberate reuse of a tombstoned id, as opposed to the renumbering
    // accident covered above.
    let dataset = dataset_with_dropped_column("memory://", "b").await;

    let mut schema = dataset.schema().clone();
    let mut field = LanceCoreField::try_from(&ArrowField::new("e", DataType::Int32, true)).unwrap();
    field.id = 1;
    schema.fields.push(field);

    let err = commit_merge(&dataset, schema).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
    let message = err.to_string();
    assert!(
        message.contains("assigns id 1 to new field \"e\"")
            && message.contains("must use ids of at least 4"),
        "unexpected error: {}",
        message
    );
}

#[tokio::test]
async fn test_merge_rejects_renumbered_nested_field_ids() {
    // A hole inside a struct shifts a nested leaf's id onto a field
    // outside the struct on renumbering; the full-path comparison must
    // catch the cross-parent remap.
    let struct_fields = Fields::from(vec![
        ArrowField::new("x", DataType::Int32, true),
        ArrowField::new("y", DataType::Int32, true),
    ]);
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", DataType::Struct(struct_fields.clone()), true),
        ArrowField::new("z", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(StructArray::new(
                struct_fields,
                vec![
                    Arc::new(Int32Array::from(vec![1, 2])),
                    Arc::new(Int32Array::from(vec![10, 20])),
                ],
                None,
            )),
            Arc::new(Int32Array::from(vec![100, 200])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema);
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    dataset.drop_columns(&["s.x"]).await.unwrap();
    assert_eq!(dataset.schema().field_ids(), vec![0, 2, 3]);

    let arrow_schema = ArrowSchema::from(dataset.schema());
    let renumbered = LanceSchema::try_from(&arrow_schema).unwrap();
    assert_eq!(renumbered.field_ids(), vec![0, 1, 2]);

    let err = commit_merge(&dataset, renumbered).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
    let message = err.to_string();
    assert!(
        message.contains("remaps field id 2 from \"s.y\" to \"z\""),
        "unexpected error: {}",
        message
    );
}

#[rstest::rstest]
#[case::drop_a("a")]
#[case::drop_b("b")]
#[case::drop_c("c")]
#[tokio::test]
async fn test_merge_allows_id_preserving_schema_change(#[case] dropped: &str) {
    let dataset = dataset_with_dropped_column("memory://", dropped).await;

    let survivors = surviving_columns(dropped);
    let first_id = dataset.schema().field(survivors[0].0).unwrap().id;
    let mut schema = dataset.schema().clone();
    schema
        .mut_field_by_id(first_id)
        .unwrap()
        .metadata
        .insert("wm".into(), "42".into());

    let dataset = commit_merge(&dataset, schema).await.unwrap();
    assert_eq!(
        dataset
            .schema()
            .field(survivors[0].0)
            .unwrap()
            .metadata
            .get("wm"),
        Some(&"42".to_string())
    );

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_columns(&batch, &survivors);
}

#[rstest::rstest]
#[case::drop_a("a")]
#[case::drop_b("b")]
#[case::drop_c("c")]
#[tokio::test]
async fn test_merge_allows_dropping_field(#[case] dropped: &str) {
    let dataset = dataset_with_dropped_column("memory://", dropped).await;

    let mut survivors = surviving_columns(dropped);
    let omitted = survivors.remove(0);
    let names: Vec<&str> = survivors.iter().map(|(n, _)| *n).collect();
    let schema = dataset.schema().project(&names).unwrap();

    let dataset = commit_merge(&dataset, schema).await.unwrap();
    assert!(dataset.schema().field(omitted.0).is_none());

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_columns(&batch, &survivors);
}

#[tokio::test]
async fn test_merge_rejects_schema_only_path_remap() {
    let dataset = dataset_with_dropped_column("memory://", "c").await;
    let prior_id = dataset.schema().field("a").unwrap().id;
    let fresh_id = dataset.manifest.max_field_id() + 1;

    let mut schema = dataset.schema().clone();
    schema.mut_field_by_id(prior_id).unwrap().id = fresh_id;

    let err = commit_merge(&dataset, schema).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
    let message = err.to_string();
    assert!(
        message.contains(&format!(
            "remaps existing field \"a\" from id {} to id {}",
            prior_id, fresh_id
        )) && message.contains("base data file"),
        "unexpected error: {}",
        message
    );
}

#[rstest::rstest]
#[case::logical_type(DataType::Float32, true, "logical type")]
#[case::nullability(DataType::Int32, false, "nullable")]
#[tokio::test]
async fn test_merge_rejects_shared_id_type_or_nullability_change(
    #[case] data_type: DataType,
    #[case] nullable: bool,
    #[case] expected: &str,
) {
    let dataset = dataset_with_dropped_column("memory://", "c").await;
    let field_id = dataset.schema().field("a").unwrap().id;

    let mut schema = dataset.schema().clone();
    let field = schema.mut_field_by_id(field_id).unwrap();
    field.logical_type = LogicalType::try_from(&data_type).unwrap();
    field.nullable = nullable;

    let err = commit_merge(&dataset, schema).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {:?}", err);
    let message = err.to_string();
    assert!(
        message.contains(&format!("changes field id {} (\"a\")", field_id))
            && message.contains(expected),
        "unexpected error: {}",
        message
    );
}

/// The pure schema/fragment-shape half of this check lives in
/// `lance_table::transaction`'s `test_merge_allows_rewritten_fresh_field_id`;
/// this covers the `Dataset`-level effect of a rewrite that assigns a fresh
/// field id.
#[tokio::test]
async fn test_alter_columns_materializes_fresh_field_id_in_every_fragment() {
    let mut dataset = dataset_with_dropped_column("memory://", "c").await;
    let prior_id = dataset.schema().field("a").unwrap().id;
    dataset
        .alter_columns(&[ColumnAlteration::new("a".into()).cast_to(DataType::Int64)])
        .await
        .unwrap();
    let new_id = dataset.schema().field("a").unwrap().id;
    assert_ne!(new_id, prior_id);
    assert!(
        dataset.get_fragments().iter().all(|fragment| {
            fragment
                .metadata()
                .files
                .iter()
                .any(|file| file.fields.contains(&new_id))
        }),
        "alter_columns must materialize the fresh id in every fragment base file"
    );
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch["a"].as_primitive::<Int64Type>().values(), &[1, 2]);
}

mod composite {
    //! End-to-end coverage for committing an action set against a real dataset.
    //!
    //! Each of these is a single commit that does work a named operation would
    //! have needed several commits for: a fragment is added and then modified, a
    //! field is added and then filled, all inside one version. That is what the
    //! action vocabulary buys -- steps that reference each other's minted ids can
    //! be squashed into one atomic manifest change.
    //!
    //! The commit path checks that a referenced data file exists, so these tests
    //! point their actions at files the fixture dataset already wrote. The
    //! resulting datasets are inspected through their manifests rather than read --
    //! the files hold the wrong columns for where they end up attached.

    use std::sync::Arc;

    use crate::Dataset;
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteParams};
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_table::format::DataFile;
    use lance_table::transaction::action::{
        Action, AddDataFile, AddField, AddFragment, CompositeOperation, DropField, Ref,
        TombstoneFieldData, UserAction,
    };
    use lance_table::transaction::{Operation, Transaction};

    /// A two-fragment dataset, so its two data files can stand in for the files an
    /// action set would otherwise have had to write.
    async fn test_dataset(enable_stable_row_ids: bool) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let data =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(0..10))])
                .unwrap();

        InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids,
                max_rows_per_file: 5,
                ..Default::default()
            })
            .execute(vec![data])
            .await
            .unwrap()
    }

    fn existing_data_file(dataset: &Dataset, fragment: usize) -> DataFile {
        dataset.fragments()[fragment].files[0].clone()
    }

    async fn commit(dataset: Dataset, actions: Vec<Action>) -> Dataset {
        let read_version = dataset.version().version;
        CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                    "step", actions,
                )])),
                None,
            ))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_one_commit_adds_a_fragment_and_then_modifies_it() {
        let dataset = test_dataset(false).await;
        let before = dataset.version().version;
        let first_file = existing_data_file(&dataset, 0);
        let second_file = existing_data_file(&dataset, 1);
        let second_path = second_file.path.clone();

        let dataset = commit(
            dataset,
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file: first_file,
                    field_ids: vec![Ref::Committed(0)],
                    data_change: true,
                }),
                // The same commit then replaces the data it just added, naming the
                // fragment by the token it was minted under.
                Action::TombstoneFieldData(TombstoneFieldData {
                    fragment: Ref::Local(0),
                    field_ids: vec![0],
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file: second_file,
                    field_ids: vec![Ref::Committed(0)],
                    data_change: true,
                }),
            ],
        )
        .await;

        assert_eq!(dataset.version().version, before + 1);
        let fragments = dataset.fragments();
        assert_eq!(fragments.len(), 3);
        let added = fragments.last().unwrap();
        assert_eq!(added.physical_rows, Some(4));
        // The tombstoned file is gone; only the replacement survives the commit.
        let paths = added
            .files
            .iter()
            .map(|file| file.path.clone())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec![second_path]);
    }

    #[tokio::test]
    async fn test_one_commit_adds_a_field_and_then_fills_it() {
        let dataset = test_dataset(false).await;
        let fragment_id = dataset.fragments()[0].id;
        let file = existing_data_file(&dataset, 1);
        let path = file.path.clone();

        let dataset = commit(
            dataset,
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: lance_core::datatypes::Field::try_from(Field::new(
                        "b",
                        DataType::Int32,
                        true,
                    ))
                    .unwrap(),
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(fragment_id),
                    file,
                    field_ids: vec![Ref::Local(0)],
                    data_change: true,
                }),
            ],
        )
        .await;

        let field = dataset.schema().field("b").expect("field b was added");
        let fragment = &dataset.fragments()[0];
        let added = fragment
            .files
            .iter()
            .find(|file| file.path == path)
            .expect("the new field's data file was attached");
        // The file points at the id the commit minted, which the caller never knew.
        assert_eq!(added.fields.as_ref(), &[field.id]);
    }

    #[tokio::test]
    async fn test_one_commit_swaps_a_field_for_a_new_one() {
        let dataset = test_dataset(false).await;
        let fragment_id = dataset.fragments()[0].id;
        let file = existing_data_file(&dataset, 1);

        let dataset = commit(
            dataset,
            vec![
                Action::DropField(DropField { field: 0 }),
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: lance_core::datatypes::Field::try_from(Field::new(
                        "a",
                        DataType::Int64,
                        true,
                    ))
                    .unwrap(),
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Committed(fragment_id),
                    file,
                    field_ids: vec![Ref::Local(0)],
                    data_change: true,
                }),
            ],
        )
        .await;

        // Dropping "a" and adding a new "a" of a different type is one version, and
        // the new field gets a fresh id rather than inheriting the dropped one.
        let schema = dataset.schema();
        assert_eq!(schema.fields.len(), 1);
        let field = schema.field("a").unwrap();
        assert_ne!(field.id, 0);
        assert_eq!(field.logical_type.to_string(), "int64");
    }

    #[tokio::test]
    async fn test_one_commit_assigns_row_ids_to_the_fragments_it_mints() {
        let dataset = test_dataset(true).await;
        let next_row_id = dataset.manifest().next_row_id;
        assert_eq!(next_row_id, 10);

        let dataset = commit(
            dataset,
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddFragment(AddFragment {
                    local: 1,
                    physical_rows: 6,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
            ],
        )
        .await;

        assert_eq!(dataset.manifest().next_row_id, 20);
        for fragment in dataset.fragments().iter().skip(2) {
            assert!(
                fragment.row_id_meta.is_some(),
                "fragment {} was minted without row ids",
                fragment.id
            );
            assert!(fragment.created_at_version_meta.is_some());
        }
    }

    #[tokio::test]
    async fn test_two_action_sets_on_disjoint_coordinates_both_commit() {
        let dataset = Arc::new(test_dataset(false).await);
        let read_version = dataset.version().version;

        let append = |local| {
            vec![Action::AddFragment(AddFragment {
                local,
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })]
        };

        let first = CommitBuilder::new(dataset.clone())
            .execute(Transaction::new(
                read_version,
                Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                    "step",
                    append(0),
                )])),
                None,
            ))
            .await
            .unwrap();

        // The second commit still reads the original version, so it has to be
        // checked against the first. Both only mint, so neither writes anything
        // the other does.
        let second = CommitBuilder::new(dataset)
            .execute(Transaction::new(
                read_version,
                Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                    "step",
                    append(0),
                )])),
                None,
            ))
            .await
            .unwrap();

        assert_eq!(second.version().version, first.version().version + 1);
        assert_eq!(second.fragments().len(), 4);
    }

    #[tokio::test]
    async fn test_two_action_sets_writing_the_same_field_data_conflict() {
        let dataset = Arc::new(test_dataset(false).await);
        let read_version = dataset.version().version;
        let fragment_id = dataset.fragments()[0].id;

        let tombstone = || {
            Transaction::new(
                read_version,
                Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                    "step",
                    vec![Action::TombstoneFieldData(TombstoneFieldData {
                        fragment: Ref::Committed(fragment_id),
                        field_ids: vec![0],
                        data_change: true,
                    })],
                )])),
                None,
            )
        };

        CommitBuilder::new(dataset.clone())
            .execute(tombstone())
            .await
            .unwrap();

        let error = CommitBuilder::new(dataset)
            .with_max_retries(0)
            .execute(tombstone())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("preempted"),
            "unexpected error: {error}"
        );
    }
}
