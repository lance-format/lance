// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
};
use std::time::Duration;
use std::vec;

use crate::Dataset;
use crate::dataset::UpdateBuilder;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::datatypes::Schema;
use crate::io::commit::load_and_sort_new_transactions;
use lance_table::io::commit::ManifestNamingScheme;

use crate::dataset::write::{CommitBuilder, WriteMode, WriteParams};
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{RecordBatchIterator, UInt32Array, types::Int32Type};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::{TempDir, TempStdDir, TempStrDir};
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_table::feature_flags::FLAG_STABLE_ROW_IDS;
use mock_instant::thread_local::MockClock;
use tokio::sync::Barrier;

use crate::dataset::refs::{base_branches_contents_path, branch_contents_path};
use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure, copy_test_data_to_tmp};
use futures::TryStreamExt;
use futures::stream::BoxStream;
use lance_core::Error;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, ObjectStoreExt as _, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, path::Path,
};
use rstest::rstest;
use std::cmp::Ordering;

fn assert_all_manifests_use_scheme(test_dir: &TempStdDir, scheme: ManifestNamingScheme) {
    let entries_names = test_dir
        .join("_versions")
        .read_dir()
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        // Ignore the version hint file, which is not a manifest.
        .filter(|name| !name.starts_with("latest_version_hint"))
        .collect::<Vec<_>>();
    assert!(
        entries_names
            .iter()
            .all(|name| ManifestNamingScheme::detect_scheme(name) == Some(scheme)),
        "Entries: {:?}",
        entries_names
    );
}

#[rstest]
#[case::one_io(1)]
#[case::two_io(2)]
fn test_transaction_reads_with_low_io_parallelism(#[case] io_parallelism: usize) {
    // Isolate the process-wide I/O setting from concurrently running tests.
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "dataset::tests::dataset_versioning::test_low_io_transaction_child",
            "--nocapture",
        ])
        .env("LANCE_LOW_IO_TRANSACTION_TEST", "1")
        .env("LANCE_IO_THREADS", io_parallelism.to_string())
        .output()
        .unwrap();
    assert!(output.status.success(), "child process failed: {output:?}");
}

#[tokio::test]
async fn test_low_io_transaction_child() {
    if std::env::var_os("LANCE_LOW_IO_TRANSACTION_TEST").is_none() {
        return;
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        let data = gen_batch()
            .col("key", array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(4))
            .unwrap();
        let reader = || RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(
            reader(),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        // An empty history must finish too: no sender will publish a manifest.
        let (current, transactions) = load_and_sort_new_transactions(&dataset).await.unwrap();
        assert_eq!(current.version().version, 1);
        assert!(transactions.is_empty());

        let stale = dataset.clone();
        dataset.append(reader(), None).await.unwrap();
        let (current, transactions) = load_and_sort_new_transactions(&stale).await.unwrap();
        assert_eq!(current.version().version, 2);
        assert_eq!(transactions.len(), 1);
        assert_eq!(transactions[0].0, 2);

        dataset.delete("key = 1").await.unwrap();
        assert_eq!(dataset.version().version, 3);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
        assert_eq!(dataset.count_rows(Some("key = 1".into())).await.unwrap(), 0);
    })
    .await
    .expect("transaction reads and delete must progress with one I/O slot");
}

#[tokio::test]
async fn test_list_manifest_locations_rejects_explicit_refs() {
    let test_dir = TempStdDir::default();
    let test_uri = test_dir.to_str().unwrap();
    let builders = [
        DatasetBuilder::from_uri(test_uri).with_version(1),
        DatasetBuilder::from_uri(test_uri).with_branch("dev", None),
        DatasetBuilder::from_uri(test_uri).with_tag("release"),
    ];

    for builder in builders {
        let err = builder.list_manifest_locations().await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("does not support an explicit"));
    }
}

#[tokio::test]
async fn test_v2_manifest_path_create() {
    // Can create a dataset, using V2 paths
    let data = lance_datagen::gen_batch()
        .col("key", array::step::<Int32Type>())
        .into_batch_rows(RowCount::from(10))
        .unwrap();
    let test_dir = TempStdDir::default();
    let test_uri = test_dir.to_str().unwrap();
    Dataset::write(
        RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
        test_uri,
        Some(WriteParams {
            enable_v2_manifest_paths: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

    // Appending to it will continue to use those paths
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(data.clone())], data.schema().clone()),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);

    UpdateBuilder::new(Arc::new(dataset))
        .update_where("key = 5")
        .unwrap()
        .set("key", "200")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
}

#[tokio::test]
async fn test_v2_manifest_path_commit() {
    let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
    .unwrap();
    let operation = Operation::Overwrite {
        fragments: vec![],
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let test_dir = TempStdDir::default();
    let test_uri = test_dir.to_str().unwrap();
    let dataset = Dataset::commit(
        test_uri,
        operation,
        None,
        None,
        None,
        Default::default(),
        true, // enable_v2_manifest_paths
    )
    .await
    .unwrap();

    assert!(dataset.manifest_location.naming_scheme == ManifestNamingScheme::V2);

    assert_all_manifests_use_scheme(&test_dir, ManifestNamingScheme::V2);
}

#[tokio::test]
async fn test_strict_overwrite() {
    let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        "x",
        DataType::Int32,
        false,
    )]))
    .unwrap();
    let operation = Operation::Overwrite {
        fragments: vec![],
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let test_uri = TempStrDir::default();
    let read_version_0_transaction = Transaction::new(0, operation, None);
    let strict_builder = CommitBuilder::new(&test_uri).with_max_retries(0);
    let unstrict_builder = CommitBuilder::new(&test_uri).with_max_retries(1);
    strict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect("Strict overwrite should succeed when writing a new dataset");
    strict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect_err("Strict overwrite should fail when committing to a stale version");
    unstrict_builder
        .clone()
        .execute(read_version_0_transaction.clone())
        .await
        .expect("Unstrict overwrite should succeed when committing to a stale version");
}

#[tokio::test]
async fn test_version_id_fast_path() {
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..5))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());

    let original = Dataset::write(reader, &test_uri, None).await.unwrap();
    assert_eq!(original.version_id(), 1);
    assert_eq!(original.version_id(), original.version().version);
    assert_eq!(original.latest_version_id().await.unwrap(), 1);

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(5..10))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema);
    let updated = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(updated.version_id(), 2);
    assert_eq!(updated.version_id(), updated.version().version);
    assert_eq!(updated.latest_version_id().await.unwrap(), 2);

    let historical = updated.checkout_version(1).await.unwrap();
    assert_eq!(historical.version_id(), 1);
    assert_eq!(historical.version_id(), historical.version().version);
    assert_eq!(historical.latest_version_id().await.unwrap(), 2);
}

#[rstest]
#[tokio::test]
async fn test_stale_checks_cover_fast_successor_and_latest_version(
    #[values(false, true)] enable_v2_manifest_paths: bool,
) {
    let expected_scheme = if enable_v2_manifest_paths {
        ManifestNamingScheme::V2
    } else {
        ManifestNamingScheme::V1
    };
    let test_uri = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..5))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());

    let original = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            enable_v2_manifest_paths,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(original.manifest_location().naming_scheme, expected_scheme);
    assert!(!original.is_stale().await.unwrap());
    assert!(!original.has_successor_version().await.unwrap());

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(5..10))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema);
    let updated = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            enable_v2_manifest_paths,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(original.is_stale().await.unwrap());
    assert!(original.has_successor_version().await.unwrap());
    assert_eq!(updated.manifest_location().naming_scheme, expected_scheme);
    assert!(!updated.is_stale().await.unwrap());
    assert!(!updated.has_successor_version().await.unwrap());

    let historical = updated.checkout_version(1).await.unwrap();
    assert_eq!(
        historical.manifest_location().naming_scheme,
        expected_scheme
    );
    assert!(historical.is_stale().await.unwrap());
    assert!(historical.has_successor_version().await.unwrap());
}

/// All row ids visible in `dataset`, in scan order.
async fn scan_row_ids(dataset: &Dataset) -> Vec<u64> {
    let batch = dataset
        .scan()
        .with_row_id()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    batch["_rowid"]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

fn u32_batch(values: std::ops::Range<u32>) -> RecordBatch {
    arrow_array::record_batch!(("i", UInt32, values.collect::<Vec<u32>>())).unwrap()
}

/// Restoring past activation would turn stable row ids off, putting row
/// addresses back into a namespace this table has already issued ids from.
#[tokio::test]
async fn test_restore_rejects_crossing_stable_id_activation() {
    let test_uri = TempStrDir::default();
    let batch = u32_batch(0..10);
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        test_uri.as_str(),
        None,
    )
    .await
    .unwrap();
    dataset.migrate_to_stable_row_ids().await.unwrap();

    let mut restored = dataset.checkout_version(1).await.unwrap();
    let err = restored.restore().await.unwrap_err();
    assert!(
        err.to_string()
            .contains("stable row ids were enabled after"),
        "{err}"
    );
}

/// A restore must not rewind the row-id high-water mark, or the next append reuses old ids.
#[tokio::test]
async fn test_restore_preserves_row_id_high_water_mark() {
    let test_uri = TempStrDir::default();
    let write = |values: std::ops::Range<u32>, mode| {
        let uri = test_uri.as_str().to_string();
        async move {
            let batch = u32_batch(values);
            Dataset::write(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                &uri,
                Some(WriteParams {
                    mode,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
        }
    };

    write(0..10, WriteMode::Create).await;
    let appended = write(10..20, WriteMode::Append).await;
    let mark = appended.manifest.next_row_id;

    let mut restored = appended.checkout_version(1).await.unwrap();
    restored.restore().await.unwrap();
    assert!(
        restored.manifest.next_row_id >= mark,
        "restore rewound the row id high-water mark: {} < {mark}",
        restored.manifest.next_row_id
    );

    // The rows appended after the restore must not reuse the dropped rows' ids.
    let reused = write(20..30, WriteMode::Append).await;
    let ids = scan_row_ids(&reused).await;
    assert_eq!(
        ids.iter().filter(|id| **id >= mark).count(),
        10,
        "appended rows did not all take fresh ids past {mark}: {ids:?}"
    );
}

#[derive(Debug, Clone, Copy)]
enum EmptyTableCopy {
    ShallowClone,
    Branch,
    DeepClone,
}

/// Once every row is deleted no fragment is left to show that the table uses
/// stable row ids, so only the flag records it. A copy that lost the flag would
/// hand out row addresses from the namespace the source has already issued ids
/// from.
#[rstest]
#[case::shallow_clone(EmptyTableCopy::ShallowClone)]
#[case::branch(EmptyTableCopy::Branch)]
#[case::deep_clone(EmptyTableCopy::DeepClone)]
#[tokio::test]
async fn test_copy_of_emptied_table_keeps_stable_row_ids(#[case] copy: EmptyTableCopy) {
    let test_dir = TempStrDir::default();
    let source_uri = format!("{}/source", test_dir.as_str());
    let copy_uri = format!("{}/copy", test_dir.as_str());
    let batch = u32_batch(0..20);
    let mut source = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        &source_uri,
        Some(WriteParams {
            max_rows_per_file: 10,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    source.delete("i >= 0").await.unwrap();
    assert!(source.manifest.fragments.is_empty());
    assert!(source.manifest.uses_stable_row_ids());
    let next_row_id = source.manifest.next_row_id;
    assert_eq!(next_row_id, 20);

    let version = source.version().version;
    match copy {
        EmptyTableCopy::ShallowClone => {
            source
                .shallow_clone(&copy_uri, version, None)
                .await
                .unwrap();
        }
        EmptyTableCopy::Branch => {
            source.create_branch("copy", version, None).await.unwrap();
        }
        EmptyTableCopy::DeepClone => {
            source.deep_clone(&copy_uri, version, None).await.unwrap();
        }
    }
    let open_copy = async || match copy {
        EmptyTableCopy::Branch => Dataset::open(&source_uri)
            .await
            .unwrap()
            .checkout_branch("copy")
            .await
            .unwrap(),
        EmptyTableCopy::ShallowClone | EmptyTableCopy::DeepClone => {
            Dataset::open(&copy_uri).await.unwrap()
        }
    };
    let mut copied = open_copy().await;
    assert!(copied.manifest.fragments.is_empty());
    assert!(
        copied.manifest.uses_stable_row_ids(),
        "the copy lost stable row ids: reader flags {}",
        copied.manifest.reader_feature_flags
    );
    assert_ne!(
        copied.manifest.writer_feature_flags & FLAG_STABLE_ROW_IDS,
        0,
        "the copy lost stable row ids: writer flags {}",
        copied.manifest.writer_feature_flags
    );
    assert_eq!(copied.manifest.next_row_id, next_row_id);

    let batch = u32_batch(20..25);
    copied
        .append(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        scan_row_ids(&open_copy().await).await,
        (next_row_id..next_row_id + 5).collect::<Vec<_>>()
    );
}

#[rstest]
#[tokio::test]
async fn test_restore(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a table
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let test_uri = TempStrDir::default();

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..100))],
    );
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, 1);
    let original_manifest = dataset.manifest.clone();

    // Delete some rows
    dataset.delete("i > 50").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    // Checkout a previous version
    let mut dataset = dataset.checkout_version(1).await.unwrap();
    assert_eq!(dataset.manifest.version, 1);
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(dataset.count_fragments(), 1);
    assert_eq!(fragments[0].metadata.deletion_file, None);
    assert_eq!(dataset.manifest, original_manifest);

    // Checkout latest and then go back.
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.manifest.version, 2);
    let mut dataset = dataset.checkout_version(1).await.unwrap();

    // Restore to a previous version
    dataset.restore().await.unwrap();
    assert_eq!(dataset.manifest.version, 3);
    assert_eq!(dataset.manifest.fragments, original_manifest.fragments);
    assert_eq!(dataset.manifest.schema, original_manifest.schema);

    // Delete some rows again (make sure we can still write as usual)
    dataset.delete("i > 30").await.unwrap();
    assert_eq!(dataset.manifest.version, 4);
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert_eq!(dataset.count_fragments(), 1);
    assert!(fragments[0].metadata.deletion_file.is_some());
}

#[rstest]
#[tokio::test]
async fn test_tag(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a table
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    let test_uri = TempStrDir::default();

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..100))],
    );
    let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, 1);

    // delete some rows
    dataset.delete("i > 50").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

    let bad_tag_creation = dataset.tags().create("tag1", 3).await;
    assert_eq!(
        bad_tag_creation.err().unwrap().to_string(),
        "Version not found error: version main:3 does not exist"
    );

    let bad_tag_deletion = dataset.tags().delete("tag1").await;
    assert_eq!(
        bad_tag_deletion.err().unwrap().to_string(),
        "Ref not found error: tag tag1 does not exist"
    );

    MockClock::set_system_time(std::time::Duration::from_secs(1));
    dataset.tags().create("tag1", 1).await.unwrap();
    let mut tag_map = dataset.tags().list().await.unwrap();
    let tag1_meta = tag_map.remove("tag1").unwrap();
    assert_eq!(tag1_meta.created_at, tag1_meta.updated_at);
    assert!(tag1_meta.created_at.is_some());

    assert_eq!(dataset.tags().list().await.unwrap().len(), 1);

    let another_bad_tag_creation = dataset.tags().create("tag1", 1).await;
    assert_eq!(
        another_bad_tag_creation.err().unwrap().to_string(),
        "Ref conflict error: tag tag1 already exists"
    );

    dataset.tags().delete("tag1").await.unwrap();

    assert_eq!(dataset.tags().list().await.unwrap().len(), 0);

    dataset.tags().create("tag1", 1).await.unwrap();
    dataset.tags().create("tag2", 1).await.unwrap();
    dataset.tags().create("v1.0.0-rc1", 2).await.unwrap();

    let default_order = dataset.tags().list_tags_ordered(None).await.unwrap();
    let default_names: Vec<_> = default_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        default_names,
        ["v1.0.0-rc1", "tag1", "tag2"],
        "Default ordering mismatch"
    );

    let asc_order = dataset
        .tags()
        .list_tags_ordered(Some(Ordering::Less))
        .await
        .unwrap();
    let asc_names: Vec<_> = asc_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        asc_names,
        ["tag1", "tag2", "v1.0.0-rc1"],
        "Ascending ordering mismatch"
    );

    let desc_order = dataset
        .tags()
        .list_tags_ordered(Some(Ordering::Greater))
        .await
        .unwrap();
    let desc_names: Vec<_> = desc_order.iter().map(|t| &t.0).collect();
    assert_eq!(
        desc_names,
        ["v1.0.0-rc1", "tag1", "tag2"],
        "Descending ordering mismatch"
    );

    assert_eq!(dataset.tags().list().await.unwrap().len(), 3);

    let bad_checkout = dataset.checkout_version("tag3").await;
    assert_eq!(
        bad_checkout.err().unwrap().to_string(),
        "Ref not found error: tag tag3 does not exist"
    );

    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 1);

    let first_ver = DatasetBuilder::from_uri(&test_uri)
        .with_tag("tag1")
        .load()
        .await
        .unwrap();
    assert_eq!(first_ver.version().version, 1);

    // test update tag
    let bad_tag_update = dataset.tags().update("tag3", 1).await;
    assert_eq!(
        bad_tag_update.err().unwrap().to_string(),
        "Ref not found error: tag tag3 does not exist"
    );

    let another_bad_tag_update = dataset.tags().update("tag1", 3).await;
    assert_eq!(
        another_bad_tag_update.err().unwrap().to_string(),
        "Version not found error: version main:3 does not exist"
    );

    let tag1_before_update = dataset.tags().get("tag1").await.unwrap();
    MockClock::set_system_time(std::time::Duration::from_secs(2));
    dataset.tags().update("tag1", 2).await.unwrap();
    let tag1_after_update = dataset.tags().get("tag1").await.unwrap();
    assert_eq!(tag1_after_update.created_at, tag1_before_update.created_at);
    assert!(tag1_after_update.updated_at > tag1_before_update.updated_at);
    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 2);

    let tag1_before_second_update = dataset.tags().get("tag1").await.unwrap();
    MockClock::set_system_time(std::time::Duration::from_secs(3));
    dataset.tags().update("tag1", 1).await.unwrap();
    let tag1_after_second_update = dataset.tags().get("tag1").await.unwrap();
    assert_eq!(
        tag1_after_second_update.created_at,
        tag1_before_second_update.created_at
    );
    assert!(tag1_after_second_update.updated_at > tag1_before_second_update.updated_at);
    dataset = dataset.checkout_version("tag1").await.unwrap();
    assert_eq!(dataset.manifest.version, 1);
}

#[tokio::test]
async fn test_concurrent_tag_creation_conflict() {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..10))],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(data)], schema),
        &test_uri,
        None,
    )
    .await
    .unwrap();
    dataset.delete("i >= 5").await.unwrap();

    let dataset = Arc::new(dataset);
    let concurrency = 32;
    let barrier = Arc::new(Barrier::new(concurrency));
    let handles = (0..concurrency)
        .map(|attempt| {
            let dataset = dataset.clone();
            let barrier = barrier.clone();
            let version = (attempt % 2 + 1) as u64;
            tokio::spawn(async move {
                barrier.wait().await;
                (version, dataset.tags().create("race", version).await)
            })
        })
        .collect::<Vec<_>>();

    let mut successful_version = None;
    let mut conflicts = 0;
    for handle in handles {
        let (version, result) = handle.await.unwrap();
        match result {
            Ok(()) => successful_version = Some(version),
            Err(Error::RefConflict { .. }) => conflicts += 1,
            Err(error) => panic!("unexpected tag creation error: {error}"),
        }
    }

    assert_eq!(conflicts, concurrency - 1);
    assert_eq!(
        dataset.tags().get_version("race").await.unwrap(),
        successful_version.unwrap()
    );
}

#[rstest]
#[tokio::test]
async fn test_fragment_id_zero_not_reused() {
    // Test case 1: Fragment id zero isn't re-used
    // 1. Create a dataset with 1 fragment
    // 2. Delete all rows
    // 3. Append another fragment
    // 4. Assert new fragment has id 1 not 0

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    // Create dataset with 1 fragment
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..10))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(batches, &test_uri, None).await.unwrap();

    // Verify we have 1 fragment with id 0
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].id(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    // Delete all rows
    dataset.delete("true").await.unwrap();

    // After deletion, dataset should be empty but max_fragment_id preserved
    assert_eq!(dataset.get_fragments().len(), 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(0));

    // Append another fragment
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(20..30))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![data].into_iter().map(Ok), schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert new fragment has id 1, not 0
    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.get_fragments()[0].id(), 1);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
}

#[rstest]
#[tokio::test]
async fn test_fragment_id_never_reset() {
    // Test case 2: Fragment id is never reset, even if all rows are deleted
    // 1. Create dataset with N fragments
    // 2. Delete all rows
    // 3. Append more fragments
    // 4. Assert new fragments have ids >= N

    let test_uri = TempStrDir::default();

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::UInt32,
        false,
    )]));

    // Create dataset with 3 fragments (N=3)
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(0..30))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 10, // Force multiple fragments
        ..Default::default()
    };
    let mut dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Verify we have 3 fragments with ids 0, 1, 2
    assert_eq!(dataset.get_fragments().len(), 3);
    assert_eq!(dataset.get_fragments()[0].id(), 0);
    assert_eq!(dataset.get_fragments()[1].id(), 1);
    assert_eq!(dataset.get_fragments()[2].id(), 2);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

    // Delete all rows
    dataset.delete("true").await.unwrap();

    // After deletion, dataset should be empty but max_fragment_id preserved
    assert_eq!(dataset.get_fragments().len(), 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(2));

    // Append more fragments (2 new fragments)
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from_iter_values(100..120))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 10, // Force multiple fragments
        ..Default::default()
    };
    let dataset = Dataset::write(batches, &test_uri, Some(write_params))
        .await
        .unwrap();

    // Assert new fragments have ids >= N (3, 4)
    assert_eq!(dataset.get_fragments().len(), 2);
    assert_eq!(dataset.get_fragments()[0].id(), 3);
    assert_eq!(dataset.get_fragments()[1].id(), 4);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(4));
}

#[tokio::test]
async fn test_overwrite_does_not_reuse_fragment_ids() {
    // An overwrite replaces every fragment, but the ids it hands out must still
    // continue from the dataset's high water mark: an id that named one set of
    // rows must never name another.
    let test_dir = TempStrDir::default();
    let write = |mode: WriteMode, rows: u64| {
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
        let params = WriteParams {
            mode,
            max_rows_per_file: 10,
            ..Default::default()
        };
        Dataset::write(reader, test_dir.as_str(), Some(params))
    };
    let fragment_ids = |dataset: &Dataset| {
        dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>()
    };

    let dataset = write(WriteMode::Create, 30).await.unwrap();
    assert_eq!(fragment_ids(&dataset), vec![0, 1, 2]);

    let dataset = write(WriteMode::Overwrite, 20).await.unwrap();
    assert_eq!(fragment_ids(&dataset), vec![3, 4]);
    assert_eq!(dataset.manifest.max_fragment_id(), Some(4));

    let dataset = write(WriteMode::Append, 10).await.unwrap();
    assert_eq!(fragment_ids(&dataset), vec![3, 4, 5]);
}

#[rstest]
#[tokio::test]
async fn test_overwrite_rejects_fragment_with_deletion_file(
    // Every form of the operation must be rejected: upserting config alongside
    // the overwrite does not make renumbering the fragment any safer.
    #[values(None, Some(HashMap::from([("key".to_string(), "value".to_string())])))]
    config_upsert_values: Option<HashMap<String, String>>,
) {
    // A deletion file's path embeds the fragment id, so it cannot follow its
    // fragment to the fresh id an overwrite assigns. Such a fragment belongs to
    // the dataset being replaced, so the operation is rejected rather than
    // silently losing the deletions.
    let test_dir = TempStrDir::default();
    let reader = gen_batch()
        .col("i", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();
    dataset.delete("i < 3").await.unwrap();

    let fragment = dataset.manifest.fragments[0].clone();
    assert!(fragment.deletion_file.is_some());
    let schema = dataset.schema().clone();
    let read_version = dataset.manifest.version;

    let err = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Overwrite {
                fragments: vec![fragment],
                schema,
                config_upsert_values,
                initial_bases: None,
            },
            None,
        ))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::InvalidInput { .. }),
        "expected invalid input, got {err:?}"
    );
    assert!(
        err.to_string().contains("must be newly written"),
        "unexpected message: {err}"
    );
}

#[tokio::test]
async fn test_commit_rejects_duplicate_fragment_ids() {
    // Append honors an id a fragment arrives with, so a caller can still hand in
    // one that an existing fragment already uses. Committing that would leave two
    // fragments sharing everything keyed by the id.
    let test_dir = TempStrDir::default();
    let write = |mode: WriteMode| {
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(1));
        let params = WriteParams {
            mode,
            max_rows_per_file: 5,
            ..Default::default()
        };
        Dataset::write(reader, test_dir.as_str(), Some(params))
    };
    let dataset = write(WriteMode::Create).await.unwrap();
    let existing = dataset.manifest.fragments[1].clone();
    assert_eq!(existing.id, 1);

    let err = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            1,
            Operation::Append {
                fragments: vec![existing],
            },
            None,
        ))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("two fragments with id 1"),
        "unexpected message: {err}"
    );
}

#[tokio::test]
async fn test_commit_on_dataset_with_mixed_file_versions() {
    // A v0.16 dataset that has both v1 and v2 files also has two fragments with
    // id 1, because the id allocation of that era could hand out an id a caller
    // had already supplied. The mixture is the more actionable diagnosis, so the
    // duplicate check must not preempt it during commit.
    let test_dir = copy_test_data_to_tmp("v0.16.0/wrong_data_version_no_fix.lance").unwrap();
    let mut dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let ids = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![0, 1, 1, 2]);

    let err = dataset.delete("false").await.unwrap_err();
    assert!(
        err.to_string()
            .contains("The dataset contains a mixture of file versions"),
        "unexpected message: {err}"
    );
}

/// create_branch and shallow_clone must read the SOURCE ref's chain, not the
/// receiver's. Both chains get a version 2 with diverged row counts so a clone
/// that wrongly resolves the version under the receiver succeeds silently with
/// the wrong data.
#[tokio::test]
async fn test_create_branch_and_shallow_clone_from_other_branch() {
    let tempdir = TempDir::default();
    let test_uri = tempdir.path_str();

    let gen_rows = |start: i32, rows: u64| {
        gen_batch()
            .col("id", array::step_custom::<Int32Type>(start, 1))
            .into_reader_rows(RowCount::from(rows), BatchCount::from(1))
    };
    let write = |uri: String, start: i32, rows: u64, mode: WriteMode| async move {
        Dataset::write(
            gen_rows(start, rows),
            uri.as_str(),
            Some(WriteParams {
                mode,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    };

    // main v1: 50 rows.
    let mut main_ds = write(test_uri.clone(), 0, 50, WriteMode::Create).await;
    // dev: forked at v1, appended 30 rows -> dev v2 has 80 rows.
    let dev_ds = main_ds.create_branch("dev", 1, None).await.unwrap();
    write(dev_ds.uri().to_string(), 1000, 30, WriteMode::Append).await;
    // Diverge main to the same version number with a different row count.
    let mut main_ds = write(test_uri.clone(), 5000, 10, WriteMode::Append).await; // main v2: 60 rows

    // Cross-source create_branch: receiver is main, source is dev.
    let child_ds = main_ds
        .create_branch("child", ("dev", 2), None)
        .await
        .unwrap();
    assert_eq!(
        child_ds.count_rows(None).await.unwrap(),
        80,
        "child must clone dev@2, not main@2"
    );

    // Cross-source shallow_clone: same rule.
    let clone_uri = format!("{}_clone", test_uri);
    let cloned_ds = main_ds
        .shallow_clone(&clone_uri, ("dev", 2), None)
        .await
        .unwrap();
    assert_eq!(
        cloned_ds.count_rows(None).await.unwrap(),
        80,
        "shallow clone must read dev@2, not main@2"
    );
}

#[tokio::test]
async fn test_main_branch_management() {
    let tempdir = TempDir::default();
    let test_uri = tempdir.path_str();
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();

    let main_branch = dataset.checkout_branch("main").await.unwrap();
    assert_eq!(main_branch.version().version, dataset.version().version);
    assert_eq!(main_branch.manifest.branch, None);

    let main_ref = dataset.checkout_version(("main", None)).await.unwrap();
    assert_eq!(main_ref.version().version, dataset.version().version);
    assert_eq!(main_ref.manifest.branch, None);

    let Err(create_branch_err) = dataset.create_branch("main", ("main", None), None).await else {
        panic!("creating a branch named main should fail");
    };
    assert!(matches!(create_branch_err, Error::InvalidRef { .. }));
    assert!(
        create_branch_err
            .to_string()
            .contains("\"main\" is reserved"),
        "{create_branch_err}"
    );

    assert!(!tempdir.std_path().join("tree").join("main").exists());
    assert!(dataset.list_branches().await.unwrap().is_empty());

    let get_branch_err = dataset.branches().get("main").await.unwrap_err();
    assert!(matches!(get_branch_err, Error::InvalidRef { .. }));
    assert!(
        get_branch_err.to_string().contains("\"main\" is reserved"),
        "{get_branch_err}"
    );

    let delete_branch_err = dataset.delete_branch("main").await.unwrap_err();
    assert!(matches!(delete_branch_err, Error::InvalidRef { .. }));
    assert!(
        delete_branch_err
            .to_string()
            .contains("\"main\" is reserved"),
        "{delete_branch_err}"
    );
}

#[tokio::test]
async fn test_cannot_delete_branch_referenced_by_tag() {
    let tempdir = TempDir::default();
    let test_uri = tempdir.path_str();
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(1), BatchCount::from(1));
    let mut dataset = Dataset::write(data, &test_uri, None).await.unwrap();
    let branch = dataset
        .create_branch("dev", /*version=*/ 1, /*store_params=*/ None)
        .await
        .unwrap();
    dataset
        .tags()
        .create("keep-dev", ("dev", branch.version().version))
        .await
        .unwrap();

    let error = dataset.delete_branch("dev").await.unwrap_err();
    assert!(matches!(&error, Error::RefConflict { .. }));
    assert!(error.to_string().contains("keep-dev"));
    dataset.checkout_version("keep-dev").await.unwrap();
}

#[rstest]
#[case::older_source(1, 2)]
#[case::newer_source(2, 1)]
#[tokio::test]
async fn test_duplicate_branch_creation_preserves_dataset(
    #[case] source_version: u64,
    #[case] conflicting_version: u64,
) {
    let tempdir = TempStrDir::default();
    let data = || {
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(2), BatchCount::from(1))
    };
    let mut dataset = Dataset::write(
        data(),
        &tempdir,
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.append(data(), None).await.unwrap();

    let branch = dataset
        .create_branch("race", source_version, None)
        .await
        .unwrap();
    let expected = branch.scan().try_into_batch().await.unwrap();
    let contents = dataset.branches().get("race").await.unwrap();

    let error = dataset
        .create_branch("race", conflicting_version, None)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::RefConflict { .. }), "{error}");
    assert!(error.to_string().contains("branch race already exists"));

    let branch = dataset.checkout_branch("race").await.unwrap();
    assert_eq!(branch.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(
        branch
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![source_version]
    );
    assert_eq!(
        serde_json::to_value(dataset.branches().get("race").await.unwrap()).unwrap(),
        serde_json::to_value(contents).unwrap()
    );
}

#[tokio::test]
async fn test_concurrent_branch_creation_preserves_winner() {
    let tempdir = TempStrDir::default();
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(4), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &tempdir,
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete("id >= 2").await.unwrap();

    let barrier = Arc::new(Barrier::new(2));
    let handles = (1..=2)
        .map(|version| {
            let mut dataset = dataset.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                (version, dataset.create_branch("race", version, None).await)
            })
        })
        .collect::<Vec<_>>();

    let mut successful_version = None;
    let mut conflicts = 0;
    for handle in handles {
        let (version, result) = handle.await.unwrap();
        match result {
            Ok(_) => {
                assert!(successful_version.replace(version).is_none());
            }
            Err(error) => {
                assert!(matches!(error, Error::RefConflict { .. }), "{error}");
                assert!(error.to_string().contains("branch race already exists"));
                conflicts += 1;
            }
        }
    }
    assert_eq!(conflicts, 1);
    let successful_version = successful_version.unwrap();
    let branch = dataset.checkout_branch("race").await.unwrap();
    let source = dataset.checkout_version(successful_version).await.unwrap();
    assert_eq!(
        branch.scan().try_into_batch().await.unwrap(),
        source.scan().try_into_batch().await.unwrap()
    );
    assert_eq!(
        branch
            .versions()
            .await
            .unwrap()
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![successful_version]
    );
    let contents = dataset.branches().get("race").await.unwrap();
    assert_eq!(contents.parent_version, successful_version);
    assert_eq!(contents.parent_branch, None);
}

#[tokio::test]
async fn test_failed_branch_clone_can_be_deleted_and_recreated() {
    let tempdir = TempStrDir::default();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(2), BatchCount::from(1));
    let mut dataset = Dataset::write(
        data,
        &tempdir,
        Some(WriteParams {
            commit_handler: Some(handler.clone()),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    handler.fail_next(AmbiguousFailure::FailOutright);

    let error = dataset.create_branch("retry", 1, None).await.unwrap_err();
    assert!(matches!(error, Error::IO { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("simulated outright commit failure")
    );
    // The name stays reserved even if the clone fails, so another creator
    // cannot race with an interrupted or ambiguously successful commit.
    let contents = dataset.branches().get("retry").await.unwrap();
    assert_eq!(contents.parent_version, 1);
    let error = dataset.create_branch("retry", 1, None).await.unwrap_err();
    assert!(matches!(error, Error::RefConflict { .. }), "{error}");
    assert!(error.to_string().contains("branch retry already exists"));

    dataset.force_delete_branch("retry").await.unwrap();
    let branch = dataset.create_branch("retry", 1, None).await.unwrap();
    assert_eq!(
        branch.scan().try_into_batch().await.unwrap(),
        dataset.scan().try_into_batch().await.unwrap()
    );
}

#[tokio::test]
async fn test_list_hierarchical_branches() {
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(1), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

    dataset.create_branch("release", 1, None).await.unwrap();
    dataset
        .create_branch("features/nested", 1, None)
        .await
        .unwrap();

    // OpenDAL-backed stores such as OSS may decode the '%2F' path part and
    // expose the branch metadata as a nested object during listing.
    let root_path = dataset.refs.root().unwrap().path;
    let encoded_path = branch_contents_path(&root_path, "features/nested");
    let nested_path = Path::parse(format!(
        "{}/features/nested.json",
        base_branches_contents_path(&root_path)
    ))
    .unwrap();
    dataset
        .object_store
        .inner
        .copy(&encoded_path, &nested_path)
        .await
        .unwrap();
    dataset.object_store.delete(&encoded_path).await.unwrap();

    let branches = dataset.list_branches().await.unwrap();
    assert_eq!(branches.len(), 2);
    assert!(branches.contains_key("release"));
    assert!(branches.contains_key("features/nested"));
}

#[derive(Debug)]
struct DirectoryMarkerStore {
    inner: Arc<dyn OSObjectStore>,
    marker: Path,
    listings: AtomicUsize,
}

impl std::fmt::Display for DirectoryMarkerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DirectoryMarkerStore")
    }
}

#[async_trait::async_trait]
impl OSObjectStore for DirectoryMarkerStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(path, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let mut entries = self.inner.list_with_delimiter(prefix).await?;
        if prefix == Some(&self.marker) {
            // Fail promptly if cleanup requeues the marker instead of hanging.
            if self.listings.fetch_add(1, AtomicOrdering::SeqCst) > 0 {
                return Err(object_store::Error::Generic {
                    store: "DirectoryMarkerStore",
                    source: "shared directory was listed more than once".into(),
                });
            }
            // Hierarchical OpenDAL stores can return the listed directory itself.
            entries.common_prefixes.push(self.marker.clone());
        }
        Ok(entries)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[rstest]
#[case::child("a/b", false, false)]
#[case::nested_child("a/b/c", false, false)]
#[case::data_directory("a/data/child", false, true)]
#[case::force("a/b", true, false)]
#[case::sibling("ab", false, false)]
#[tokio::test]
async fn test_delete_branch_with_shared_directory(
    #[case] remaining_branch: &str,
    #[case] force: bool,
    #[case] use_legacy_branch_creation: bool,
    #[values(false, true)] use_object_store: bool,
    #[values(false, true)] has_directory_marker: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        let path_prefix = if cfg!(windows) { "/" } else { "" };
        format!("file-object-store://{path_prefix}{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let data = |start| {
        gen_batch()
            .col("id", array::step_custom::<Int32Type>(start, 1))
            .into_reader_rows(RowCount::from(2), BatchCount::from(1))
    };
    let params = || WriteParams {
        max_rows_per_file: 1,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data(0), &uri, Some(params())).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let mut deleted = dataset.create_branch("a", 1, None).await.unwrap();
    deleted.append(data(10), Some(params())).await.unwrap();
    if use_legacy_branch_creation {
        create_branch_before_name_check(&dataset, remaining_branch).await;
    } else {
        dataset
            .create_branch(remaining_branch, 1, None)
            .await
            .unwrap();
    }
    let mut remaining = dataset.checkout_branch(remaining_branch).await.unwrap();
    remaining.append(data(20), Some(params())).await.unwrap();

    let marker_store = has_directory_marker.then(|| {
        let mut store = dataset.object_store.as_ref().clone();
        let marker_store = Arc::new(DirectoryMarkerStore {
            inner: store.inner.clone(),
            marker: deleted.base.clone(),
            listings: AtomicUsize::new(0),
        });
        store.inner = marker_store.clone();
        let store = Arc::new(store);
        dataset.object_store = store.clone();
        dataset.refs.object_store = store;
        marker_store
    });
    if force {
        let metadata = branch_contents_path(&dataset.base, "a");
        dataset.object_store.delete(&metadata).await.unwrap();
        dataset.force_delete_branch("a").await.unwrap();
    } else {
        dataset.delete_branch("a").await.unwrap();
    }
    if let Some(marker_store) = marker_store {
        assert_eq!(
            marker_store.listings.load(AtomicOrdering::SeqCst),
            usize::from(remaining_branch.starts_with("a/"))
        );
    }
    let branches = dataset.list_branches().await.unwrap();
    assert_eq!(branches.len(), 1);
    assert!(branches.contains_key(remaining_branch));
    for version in [None, Some(1), Some(2)] {
        let error = dataset.checkout_version(("a", version)).await.unwrap_err();
        assert!(matches!(
            error,
            Error::NotFound { .. } | Error::DatasetNotFound { .. }
        ));
        let message = error.to_string().replace('\\', "/");
        assert!(message.contains(deleted.base.as_ref()), "{error}");
    }
    let files = dataset
        .object_store
        .list(Some(deleted.base.clone()))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert!(
        files
            .iter()
            .all(|file| file.location.prefix_matches(&remaining.base))
    );

    let mut recreated = dataset.create_branch("a", 1, None).await.unwrap();
    assert_eq!(recreated.count_rows(None).await.unwrap(), 2);
    recreated.append(data(30), Some(params())).await.unwrap();
    assert_eq!(recreated.count_rows(None).await.unwrap(), 4);

    let mut remaining = dataset.checkout_branch(remaining_branch).await.unwrap();
    remaining.append(data(40), Some(params())).await.unwrap();
    let batch = remaining.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values().as_ref(),
        &[0, 1, 20, 21, 40, 41]
    );
}

#[rstest]
#[case::direct(false, false)]
#[case::object_store(true, false)]
#[case::rooted(false, true)]
#[tokio::test]
async fn test_delete_branch_preserves_case_aliased_shared_directories(
    #[case] use_object_store: bool,
    #[case] is_rooted_store: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        let path_prefix = if cfg!(windows) { "/" } else { "" };
        format!("file-object-store://{path_prefix}{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let mut deleted = root.create_branch("a", 1, None).await.unwrap();
    deleted.append(id_rows(10, 1), None).await.unwrap();
    // Exercise physical aliases on case-sensitive hosts as well as Windows.
    #[cfg(unix)]
    if !tempdir.std_path().join("tree/A").exists() {
        std::os::unix::fs::symlink(
            tempdir.std_path().join("tree/a"),
            tempdir.std_path().join("tree/A"),
        )
        .unwrap();
    }
    let mut lower_child = root.create_branch("a/b", 1, None).await.unwrap();
    lower_child.append(id_rows(20, 1), None).await.unwrap();
    let mut case_child = root.create_branch("A/c", 1, None).await.unwrap();
    case_child.append(id_rows(30, 1), None).await.unwrap();
    if is_rooted_store {
        // A provider without native path resolution must preserve ambiguous subtrees.
        let mut store = root.object_store.as_ref().clone();
        store.store_prefix = "file$rooted-test".to_owned();
        assert!(!store.has_direct_local_paths());
        let store = Arc::new(store);
        root.object_store = store.clone();
        root.refs.object_store = store;
    }

    root.delete_branch("a").await.unwrap();

    let branches = root.list_branches().await.unwrap();
    assert_eq!(branches.len(), 2);
    assert!(branches.contains_key("a/b"));
    assert!(branches.contains_key("A/c"));

    for (branch, expected_before, appended, expected_after) in [
        ("a/b", [0, 20], 40, [0, 20, 40]),
        ("A/c", [0, 30], 50, [0, 30, 50]),
    ] {
        let mut dataset = root.checkout_branch(branch).await.unwrap();
        assert_eq!(
            dataset.scan().try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .as_ref(),
            &expected_before
        );
        dataset.append(id_rows(appended, 1), None).await.unwrap();
        assert_eq!(
            root.checkout_branch(branch)
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .as_ref(),
            &expected_after
        );
    }
}

#[rstest]
#[case::same_name(false, false)]
#[case::alias(true, false)]
#[case::alias_object_store(true, true)]
#[tokio::test]
async fn test_delete_branch_preserves_aliased_parent_data(
    #[case] has_alias: bool,
    #[case] use_object_store: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        let path_prefix = if cfg!(windows) { "/" } else { "" };
        format!("file-object-store://{path_prefix}{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let parent_branch = if has_alias { "A" } else { "a" };
    let mut parent = root.create_branch(parent_branch, 1, None).await.unwrap();
    parent.append(id_rows(10, 1), None).await.unwrap();
    #[cfg(unix)]
    if has_alias && !tempdir.std_path().join("tree/a").exists() {
        std::os::unix::fs::symlink(
            tempdir.std_path().join("tree/A"),
            tempdir.std_path().join("tree/a"),
        )
        .unwrap();
    }
    create_branch_before_name_check(&parent, "a/data").await;
    create_branch_before_name_check(&root, "a/data/x").await;

    root.delete_branch("a/data").await.unwrap();

    let mut parent = Dataset::open(&uri)
        .await
        .unwrap()
        .checkout_branch(parent_branch)
        .await
        .unwrap();
    let batch = parent.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values().as_ref(),
        &[0, 10]
    );
    parent.append(id_rows(20, 1), None).await.unwrap();
    assert_eq!(parent.count_rows(None).await.unwrap(), 3);
    assert!(root.list_branches().await.unwrap().contains_key("a/data/x"));
}

#[cfg(unix)]
#[tokio::test]
async fn test_delete_branch_preserves_unicode_aliased_parent_data() {
    let tempdir = TempDir::default();
    let uri = tempdir.path_str();
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let mut parent = root.create_branch("a", 1, None).await.unwrap();
    parent.append(id_rows(10, 1), None).await.unwrap();

    let versions = tempdir.std_path().join("tree/a/_versions");
    let unicode_alias = tempdir.std_path().join("tree/a/_ver\u{017f}ions");
    if !unicode_alias.exists() {
        std::os::unix::fs::symlink(&versions, &unicode_alias).unwrap();
    }

    let middle_name = "a/_ver\u{017f}ions";
    let mut middle = root.create_branch(middle_name, 1, None).await.unwrap();
    middle.append(id_rows(20, 1), None).await.unwrap();
    let mut leaf = root
        .create_branch("a/_ver\u{017f}ions/x", 1, None)
        .await
        .unwrap();
    leaf.append(id_rows(30, 1), None).await.unwrap();

    root.delete_branch(middle_name).await.unwrap();

    assert!(
        root.list_branches()
            .await
            .unwrap()
            .contains_key("a/_ver\u{017f}ions/x")
    );
    let mut parent = Dataset::open(&uri)
        .await
        .unwrap()
        .checkout_branch("a")
        .await
        .unwrap();
    let batch = parent.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values().as_ref(),
        &[0, 10]
    );
    parent.append(id_rows(40, 1), None).await.unwrap();
    assert_eq!(parent.count_rows(None).await.unwrap(), 3);
}

#[cfg(unix)]
#[rstest]
#[case::data("data", "data")]
#[case::unicode_versions("_versions", "_ver\u{017f}ions")]
#[tokio::test]
async fn test_delete_branch_preserves_relocated_parent_data(
    #[case] dataset_dir: &str,
    #[case] alias: &str,
    #[values(false, true)] use_object_store: bool,
    #[values(false, true)] has_child: bool,
    #[values(false, true)] has_sibling: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let mut parent = root.create_branch("A", 1, None).await.unwrap();
    parent.append(id_rows(10, 1), None).await.unwrap();

    let owned_path = tempdir.std_path().join("tree/A").join(dataset_dir);
    let relocated_path = tempdir.std_path().join("relocated");
    std::fs::rename(&owned_path, &relocated_path).unwrap();
    std::os::unix::fs::symlink(&relocated_path, &owned_path).unwrap();
    let alias_path = tempdir.std_path().join("tree/A").join(alias);
    if !alias_path.exists() {
        std::os::unix::fs::symlink(&owned_path, &alias_path).unwrap();
    }
    if !tempdir.std_path().join("tree/a").exists() {
        std::os::unix::fs::symlink(
            tempdir.std_path().join("tree/A"),
            tempdir.std_path().join("tree/a"),
        )
        .unwrap();
    }
    let deleted_branch = format!("a/{alias}");
    create_branch_before_name_check(&parent, &deleted_branch).await;
    if has_sibling {
        // Keep recursive cleanup rooted at the dataset directory without a child branch.
        root.create_branch("a/sibling", 1, None).await.unwrap();
    }
    let child_branch = format!("{deleted_branch}/x");
    if has_child {
        create_branch_before_name_check(&root, &child_branch).await;
    }

    for after_deletion in [false, true] {
        if after_deletion {
            root.delete_branch(&deleted_branch).await.unwrap();
            let branches = root.list_branches().await.unwrap();
            assert!(!branches.contains_key(&deleted_branch));
            assert!(branches.contains_key("A"));
            assert_eq!(branches.contains_key("a/sibling"), has_sibling);
            assert_eq!(branches.contains_key(&child_branch), has_child);
        }
        let mut parent = Dataset::open(&uri)
            .await
            .unwrap()
            .checkout_branch("A")
            .await
            .unwrap();
        let batch = parent.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 10]
        );
        if after_deletion {
            parent.append(id_rows(20, 1), None).await.unwrap();
            assert_eq!(parent.count_rows(None).await.unwrap(), 3);
        }
    }
    for branch in [
        has_sibling.then_some("a/sibling"),
        has_child.then_some(child_branch.as_str()),
    ]
    .into_iter()
    .flatten()
    {
        let mut survivor = root.checkout_branch(branch).await.unwrap();
        survivor.append(id_rows(30, 1), None).await.unwrap();
        let batch = survivor.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 30]
        );
    }
}

#[cfg(unix)]
#[rstest]
#[case::direct_nested(false, true)]
#[case::object_store_nested(true, true)]
#[case::direct_alias(false, false)]
#[case::object_store_alias(true, false)]
#[tokio::test]
async fn test_delete_branch_preserves_nested_dataset_data(
    #[case] use_object_store: bool,
    #[case] relocate_into_deleted: bool,
    #[values(false, true)] deep_alias: bool,
    #[values(false, true)] has_child: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let mut parent = root.create_branch("retained", 1, None).await.unwrap();
    parent.append(id_rows(10, 1), None).await.unwrap();
    let mut deleted = root.create_branch("a", 1, None).await.unwrap();
    deleted.append(id_rows(20, 1), None).await.unwrap();
    if has_child {
        root.create_branch("a/b", 1, None).await.unwrap();
    }

    let parent_data = tempdir.std_path().join("tree/retained/data");
    let container = tempdir.std_path().join(if deep_alias {
        "tree/a/container"
    } else {
        "tree/a"
    });
    std::fs::create_dir_all(&container).unwrap();
    let nested_data = container.join("parent-data");
    if relocate_into_deleted {
        std::fs::rename(&parent_data, &nested_data).unwrap();
        std::os::unix::fs::symlink(&nested_data, &parent_data).unwrap();
    } else {
        std::os::unix::fs::symlink(&parent_data, &nested_data).unwrap();
    }

    for after_deletion in [false, true] {
        if after_deletion {
            root.delete_branch("a").await.unwrap();
            assert!(!root.list_branches().await.unwrap().contains_key("a"));
        }
        let mut parent = Dataset::open(&uri)
            .await
            .unwrap()
            .checkout_branch("retained")
            .await
            .unwrap();
        let batch = parent.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 10]
        );
        if after_deletion {
            parent.append(id_rows(30, 1), None).await.unwrap();
            assert_eq!(parent.count_rows(None).await.unwrap(), 3);
        }
    }
    let error = root.checkout_branch("a").await.unwrap_err();
    assert!(matches!(
        error,
        Error::NotFound { .. } | Error::DatasetNotFound { .. }
    ));
    assert!(
        error
            .to_string()
            .replace('\\', "/")
            .contains(deleted.base.as_ref()),
        "{error}"
    );
    let mut recreated = root.create_branch("a", 1, None).await.unwrap();
    recreated.append(id_rows(40, 1), None).await.unwrap();
    assert_eq!(recreated.count_rows(None).await.unwrap(), 2);
    if has_child {
        let mut child = root.checkout_branch("a/b").await.unwrap();
        child.append(id_rows(50, 1), None).await.unwrap();
        let batch = child.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 50]
        );
    }
}

#[cfg(unix)]
#[rstest]
#[case::data("data")]
#[case::versions("_versions")]
#[tokio::test]
async fn test_delete_branch_preserves_aliased_main_dataset(
    #[case] dataset_dir: &str,
    #[values(false, true)] use_object_store: bool,
    #[values(false, true)] has_child: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    root.append(id_rows(10, 1), None).await.unwrap();
    std::fs::create_dir_all(tempdir.std_path().join("tree")).unwrap();
    std::os::unix::fs::symlink(
        tempdir.std_path().join(dataset_dir),
        tempdir.std_path().join("tree/a"),
    )
    .unwrap();
    root.create_branch("a", 1, None).await.unwrap();
    if has_child {
        root.create_branch("a/b", 1, None).await.unwrap();
    }

    for after_deletion in [false, true] {
        if after_deletion {
            root.delete_branch("a").await.unwrap();
            assert!(!root.list_branches().await.unwrap().contains_key("a"));
        }
        let mut main = Dataset::open(&uri).await.unwrap();
        let batch = main.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 10]
        );
        if after_deletion {
            main.append(id_rows(20, 1), None).await.unwrap();
            let batch = main.scan().try_into_batch().await.unwrap();
            assert_eq!(
                batch["id"].as_primitive::<Int32Type>().values().as_ref(),
                &[0, 10, 20]
            );
        }
    }
    if has_child {
        let mut child = root.checkout_branch("a/b").await.unwrap();
        child.append(id_rows(30, 1), None).await.unwrap();
        let batch = child.scan().try_into_batch().await.unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values().as_ref(),
            &[0, 30]
        );
    }
}

#[derive(Debug)]
struct GateSpelledRefsStore {
    inner: Arc<dyn OSObjectStore>,
    root: Path,
    spelling: &'static str,
}

impl std::fmt::Display for GateSpelledRefsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GateSpelledRefsStore")
    }
}

#[async_trait::async_trait]
impl OSObjectStore for GateSpelledRefsStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(path, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        self.inner.get_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let mut entries = self.inner.list_with_delimiter(prefix).await?;
        if prefix == Some(&self.root) {
            for directory in &mut entries.common_prefixes {
                if directory.filename() == Some("_refs") {
                    *directory = self.root.clone().join(self.spelling);
                }
            }
        }
        Ok(entries)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[cfg(unix)]
#[rstest]
#[tokio::test]
async fn test_delete_branch_preserves_shared_refs(
    #[values(false, true)] use_object_store: bool,
    #[values(false, true)] deep_alias: bool,
    #[values(false, true)] has_child: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    root.tags().create("checkpoint", 1).await.unwrap();
    let mut survivor = root.create_branch("retained", 1, None).await.unwrap();
    survivor.append(id_rows(10, 1), None).await.unwrap();
    let metadata = HashMap::from([("owner".to_string(), "retained".to_string())]);
    root.refs
        .branches()
        .replace_metadata("retained", metadata.clone())
        .await
        .unwrap();
    root.refs
        .tags()
        .replace_metadata("checkpoint", metadata)
        .await
        .unwrap();
    if !deep_alias {
        std::os::unix::fs::symlink(
            tempdir.std_path().join("_refs"),
            tempdir.std_path().join("tree/a"),
        )
        .unwrap();
    }
    root.create_branch("a", 1, None).await.unwrap();
    if has_child {
        root.create_branch("a/b", 1, None).await.unwrap();
    }
    if deep_alias {
        let container = tempdir.std_path().join("tree/a/container");
        std::fs::create_dir_all(&container).unwrap();
        std::os::unix::fs::symlink(
            tempdir.std_path().join("_refs"),
            container.join("shared-refs"),
        )
        .unwrap();
    }
    let mut expected_branches = root.list_branches().await.unwrap();
    expected_branches.remove("a").unwrap();
    let expected_tags = root.tags().list().await.unwrap();

    root.delete_branch("a").await.unwrap();

    let fresh = Dataset::open(&uri).await.unwrap();
    assert_eq!(
        serde_json::to_value(fresh.list_branches().await.unwrap()).unwrap(),
        serde_json::to_value(expected_branches).unwrap()
    );
    assert_eq!(
        serde_json::to_value(fresh.tags().list().await.unwrap()).unwrap(),
        serde_json::to_value(expected_tags).unwrap()
    );
    let mut survivor = fresh.checkout_branch("retained").await.unwrap();
    survivor.append(id_rows(20, 1), None).await.unwrap();
    let batch = survivor.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values().as_ref(),
        &[0, 10, 20]
    );
}

#[cfg(unix)]
#[rstest]
#[case::normal("_refs")]
#[case::uppercase("_REFS")]
#[case::unicode("_ref\u{017f}")]
#[tokio::test]
async fn test_gatekeeper_case_spelled_shared_refs(
    #[case] spelling: &'static str,
    #[values(false, true)] use_object_store: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    root.tags().create("checkpoint", 1).await.unwrap();
    let mut survivor = root.create_branch("retained", 1, None).await.unwrap();
    survivor.append(id_rows(10, 1), None).await.unwrap();
    std::os::unix::fs::symlink(
        tempdir.std_path().join("_refs"),
        tempdir.std_path().join("tree/a"),
    )
    .unwrap();
    root.create_branch("a", 1, None).await.unwrap();
    root.create_branch("a/b", 1, None).await.unwrap();
    let mut expected_branches = root.list_branches().await.unwrap();
    expected_branches.remove("a").unwrap();
    let expected_tags = root.tags().list().await.unwrap();
    assert_eq!(
        Dataset::open(&uri)
            .await
            .unwrap()
            .checkout_branch("retained")
            .await
            .unwrap()
            .scan()
            .try_into_batch()
            .await
            .unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .as_ref(),
        &[0, 10],
    );

    let mut store = root.object_store.as_ref().clone();
    store.inner = Arc::new(GateSpelledRefsStore {
        inner: store.inner.clone(),
        root: root.refs.root().unwrap().path,
        spelling,
    });
    let store = Arc::new(store);
    root.object_store = store.clone();
    root.refs.object_store = store;
    assert!(root.list_branches().await.unwrap().contains_key("retained"));
    root.delete_branch("a").await.unwrap();

    let fresh = Dataset::open(&uri).await.unwrap();
    let branches = fresh.list_branches().await.unwrap();
    let tags = fresh.tags().list().await.unwrap();
    assert_eq!(
        serde_json::to_value(&branches).unwrap(),
        serde_json::to_value(expected_branches).unwrap(),
        "shared branch records lost: {:?}",
        branches.keys().collect::<Vec<_>>(),
    );
    assert_eq!(
        serde_json::to_value(tags).unwrap(),
        serde_json::to_value(expected_tags).unwrap(),
    );
}

#[cfg(unix)]
#[rstest]
#[tokio::test]
async fn test_delete_branch_preserves_nested_blob_target(
    #[values(false, true)] use_object_store: bool,
    #[values(false, true)] relocate_owned_child: bool,
    #[values(false, true)] is_file_alias: bool,
    #[values(false, true)] has_child: bool,
) {
    let tempdir = TempDir::default();
    let uri = if use_object_store {
        format!("file-object-store://{}", tempdir.path_str())
    } else {
        tempdir.path_str()
    };
    let owned_dir = tempdir.std_path().join("_blobs/nested/deeper");
    std::fs::create_dir_all(&owned_dir).unwrap();
    std::fs::write(owned_dir.join("payload.blob"), b"payload").unwrap();
    let mut values = crate::blob::BlobDescriptorArrayBuilder::new("blob");
    values
        .push(crate::blob::BlobDescriptor::Managed {
            base_id: None,
            uri: "_blobs/nested/deeper/payload.blob".to_string(),
            offset: 0,
            size: 7,
        })
        .unwrap();
    let (field, values) = values.finish().unwrap().into_parts();
    let schema = Arc::new(crate::dataset::blob::test_util::managed_blob_schema(vec![
        field,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
    let mut root = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    root.create_branch("a", 1, None).await.unwrap();
    if has_child {
        root.create_branch("a/b", 1, None).await.unwrap();
    }
    let container = tempdir.std_path().join("tree/a/container");
    std::fs::create_dir_all(&container).unwrap();
    let target = container.join("blob-payload");
    let owned_path = if is_file_alias {
        owned_dir.join("payload.blob")
    } else {
        owned_dir
    };
    if relocate_owned_child {
        std::fs::rename(&owned_path, &target).unwrap();
        std::os::unix::fs::symlink(&target, &owned_path).unwrap();
    } else {
        std::os::unix::fs::symlink(&owned_path, &target).unwrap();
    }
    std::os::unix::fs::symlink(
        tempdir.std_path().join("_blobs"),
        tempdir.std_path().join("_blobs/cycle"),
    )
    .unwrap();
    for after_deletion in [false, true] {
        if after_deletion {
            root.delete_branch("a").await.unwrap();
        }
        let fresh = Arc::new(Dataset::open(&uri).await.unwrap());
        let handles = fresh.take_blobs_by_indices(&[0], "blob").await.unwrap();
        let bytes = handles[0].as_ref().unwrap().read().await.unwrap();
        assert_eq!(bytes.as_ref(), b"payload");
    }
}

#[cfg(unix)]
#[tokio::test]
async fn test_delete_branch_preserves_symlink_routed_child_namespace() {
    let tempdir = TempDir::default();
    let uri = tempdir.path_str();
    let mut root = Dataset::write(id_rows(0, 1), &uri, None).await.unwrap();
    let mut parent = root.create_branch("p", 1, None).await.unwrap();
    parent.append(id_rows(10, 2), None).await.unwrap();
    let mut removed = root.create_branch("a", 1, None).await.unwrap();
    removed.append(id_rows(20, 1), None).await.unwrap();

    let data_path = tempdir.std_path().join("tree/p/data");
    let relocated_data = tempdir.std_path().join("relocated-data");
    std::fs::rename(&data_path, &relocated_data).unwrap();
    std::os::unix::fs::symlink(&relocated_data, &data_path).unwrap();
    let route = tempdir.std_path().join("tree/a/b");
    std::os::unix::fs::symlink(&relocated_data, &route).unwrap();

    let mut child = root.create_branch("a/b/x", 1, None).await.unwrap();
    child.append(id_rows(30, 1), None).await.unwrap();

    root.delete_branch("a").await.unwrap();

    assert!(route.is_symlink(), "the surviving branch route was removed");
    assert!(root.list_branches().await.unwrap().contains_key("a/b/x"));
    let parent = Dataset::open(&uri)
        .await
        .unwrap()
        .checkout_branch("p")
        .await
        .unwrap();
    assert_eq!(
        parent.scan().try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .as_ref(),
        &[0, 10, 11]
    );
    let child = Dataset::open(&uri)
        .await
        .unwrap()
        .checkout_branch("a/b/x")
        .await
        .unwrap();
    assert_eq!(
        child.scan().try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .as_ref(),
        &[0, 30]
    );
    match Dataset::open(&uri)
        .await
        .unwrap()
        .checkout_version(("a", None))
        .await
    {
        Err(Error::NotFound { .. } | Error::DatasetNotFound { .. }) => {}
        Err(error) => panic!("unexpected checkout error: {error}"),
        Ok(still_readable) => panic!(
            "deleted branch is still readable: {:?}",
            still_readable.scan().try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .as_ref()
        ),
    }
}

#[tokio::test]
async fn test_branch() {
    let tempdir = TempDir::default();
    let test_uri = tempdir.path_str();
    let data_storage_version = LanceFileVersion::Stable;

    // Generate consistent test data batches
    let generate_data = |prefix: &str, start_id: i32, row_count: u64| {
        gen_batch()
            .col("id", array::step_custom::<Int32Type>(start_id, 1))
            .col("value", array::fill_utf8(format!("{prefix}_data")))
            .into_reader_rows(RowCount::from(row_count), BatchCount::from(1))
    };

    // Reusable dataset writer with configurable mode
    async fn write_dataset(
        uri: &str,
        data_reader: impl RecordBatchReader + Send + 'static,
        mode: WriteMode,
        version: LanceFileVersion,
    ) -> Dataset {
        let params = WriteParams {
            max_rows_per_file: 100,
            max_rows_per_group: 20,
            data_storage_version: Some(version),
            mode,
            ..Default::default()
        };
        Dataset::write(data_reader, uri, Some(params))
            .await
            .unwrap()
    }

    // Unified dataset scanning and row counting
    async fn collect_rows(dataset: &Dataset) -> (usize, Vec<RecordBatch>) {
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        (batches.iter().map(|b| b.num_rows()).sum(), batches)
    }

    // Phase 1: Create empty dataset, write data batch 1, create branch1 based on version_number, write data batch 2
    let mut dataset = write_dataset(
        &test_uri,
        generate_data("batch1", 0, 50),
        WriteMode::Create,
        data_storage_version,
    )
    .await;

    let original_version = dataset.version().version;
    assert_eq!(original_version, 1);

    // Create branch1 on the latest version and write data batch 2
    let mut branch1_dataset = dataset
        .create_branch("branch1", original_version, None)
        .await
        .unwrap();
    assert_eq!(branch1_dataset.uri, format!("{}/tree/branch1", test_uri));

    branch1_dataset = write_dataset(
        branch1_dataset.uri(),
        generate_data("batch2", 50, 30),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Phase 2: Create branch2 based on branch1's latest version_number, write data batch 3
    let mut branch2_dataset = branch1_dataset
        .create_branch(
            "dev/branch2",
            ("branch1", branch1_dataset.version().version),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        branch2_dataset.uri,
        format!("{}/tree/dev/branch2", test_uri)
    );

    branch2_dataset = write_dataset(
        branch2_dataset.uri(),
        generate_data("batch3", 80, 20),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Phase 3: Create a tag on branch2, the actual tag content is under root dataset
    // create branch3 based on that tag, write data batch 4
    branch2_dataset
        .tags()
        .create("tag1", ("dev/branch2", branch2_dataset.version().version))
        .await
        .unwrap();

    let mut branch3_dataset = branch2_dataset
        .create_branch("feature/nathan/branch3", "tag1", None)
        .await
        .unwrap();
    assert_eq!(
        branch3_dataset.uri,
        format!("{}/tree/feature/nathan/branch3", test_uri)
    );

    branch3_dataset = write_dataset(
        branch3_dataset.uri(),
        generate_data("batch4", 100, 25),
        WriteMode::Append,
        data_storage_version,
    )
    .await;

    // Verify data correctness and independence of each branch
    // Main branch only has data 1 (50 rows)
    let main_dataset = Dataset::open(&test_uri).await.unwrap();
    let (main_rows, _) = collect_rows(&main_dataset).await;
    assert_eq!(main_rows, 50); // only batch1
    assert_eq!(main_dataset.version().version, 1);
    let main_versions = main_dataset.version_refs().await.unwrap();
    assert_eq!(
        main_versions
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(
        main_dataset.latest_version_id().await.unwrap(),
        main_versions.last().unwrap().version
    );

    // branch1 has data 1 + 2 (80 rows)
    let updated_branch1 = Dataset::open(branch1_dataset.uri()).await.unwrap();
    let (branch1_rows, _) = collect_rows(&updated_branch1).await;
    assert_eq!(branch1_rows, 80); // batch1+batch2
    assert_eq!(updated_branch1.version().version, 2);
    let _ = updated_branch1.object_store.as_ref().io_stats_incremental();
    let branch1_versions = updated_branch1.version_refs().await.unwrap();
    let io_stats = updated_branch1.object_store.as_ref().io_stats_incremental();
    assert_eq!(io_stats.read_bytes, 0);
    assert_eq!(
        branch1_versions
            .iter()
            .map(|version| version.version)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        updated_branch1.latest_version_id().await.unwrap(),
        branch1_versions.last().unwrap().version
    );

    // branch2 has data 1 + 2 + 3 (100 rows)
    let updated_branch2 = Dataset::open(branch2_dataset.uri()).await.unwrap();
    let (branch2_rows, _) = collect_rows(&updated_branch2).await;
    assert_eq!(branch2_rows, 100); // batch1+batch2+batch3
    assert_eq!(updated_branch2.version().version, 3);

    // branch3 has data 1 + 2 + 3 + 4 (125 rows)
    let updated_branch3 = Dataset::open(branch3_dataset.uri()).await.unwrap();
    let (branch3_rows, _) = collect_rows(&updated_branch3).await;
    assert_eq!(branch3_rows, 125); // batch1+batch2+batch3+batch4
    assert_eq!(updated_branch3.version().version, 4);

    // Use list_branches to get branch list and verify each field of branch_content
    let branches = dataset.list_branches().await.unwrap();
    assert_eq!(branches.len(), 3);
    assert!(branches.contains_key("branch1"));
    assert!(branches.contains_key("dev/branch2"));
    assert!(branches.contains_key("feature/nathan/branch3"));

    // Verify branch1 content
    let branch1_content = branches.get("branch1").unwrap();
    assert_eq!(branch1_content.parent_branch, None); // Created based on main branch
    assert_eq!(branch1_content.parent_version, 1);
    assert!(branch1_content.create_at > 0);
    assert!(branch1_content.manifest_size > 0);

    // Verify branch2 content
    let branch2_content = branches.get("dev/branch2").unwrap();
    assert_eq!(branch2_content.parent_branch.as_deref().unwrap(), "branch1");
    assert_eq!(branch2_content.parent_version, 2);
    assert!(branch2_content.create_at > 0);
    assert!(branch2_content.manifest_size > 0);
    assert!(branch2_content.create_at >= branch1_content.create_at);

    // Verify branch3 content
    let branch3_content = branches.get("feature/nathan/branch3").unwrap();
    // Created based on tag pointed to branch2
    assert_eq!(
        branch3_content.parent_branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(branch3_content.parent_version, 3);
    assert!(branch3_content.create_at > 0);
    assert!(branch3_content.manifest_size > 0);
    assert!(branch3_content.create_at >= branch2_content.create_at);

    // Verify checkout_branch
    let checkout_branch1 = main_dataset.checkout_branch("branch1").await.unwrap();
    let checkout_branch2 = checkout_branch1
        .checkout_branch("dev/branch2")
        .await
        .unwrap();
    let checkout_branch2_tag = checkout_branch1.checkout_version("tag1").await.unwrap();
    let checkout_branch3 = checkout_branch2_tag
        .checkout_branch("feature/nathan/branch3")
        .await
        .unwrap();
    let checkout_branch3_at_version3 = checkout_branch2
        .checkout_version(("feature/nathan/branch3", 3))
        .await
        .unwrap();
    assert_eq!(checkout_branch3.version().version, 4);
    assert_eq!(checkout_branch3_at_version3.version().version, 3);
    assert_eq!(checkout_branch2.version().version, 3);
    assert_eq!(checkout_branch2_tag.version().version, 3);
    assert_eq!(checkout_branch1.version().version, 2);
    assert_eq!(checkout_branch3.count_rows(None).await.unwrap(), 125);
    assert_eq!(
        checkout_branch3_at_version3.count_rows(None).await.unwrap(),
        100
    );
    assert_eq!(checkout_branch2.count_rows(None).await.unwrap(), 100);
    assert_eq!(checkout_branch2_tag.count_rows(None).await.unwrap(), 100);
    assert_eq!(checkout_branch1.count_rows(None).await.unwrap(), 80);
    assert_eq!(
        checkout_branch3.manifest.branch.as_deref().unwrap(),
        "feature/nathan/branch3"
    );
    assert_eq!(
        checkout_branch3_at_version3
            .manifest
            .branch
            .as_deref()
            .unwrap(),
        "feature/nathan/branch3"
    );
    assert_eq!(
        checkout_branch2.manifest.branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(
        checkout_branch2_tag.manifest.branch.as_deref().unwrap(),
        "dev/branch2"
    );
    assert_eq!(
        checkout_branch1.manifest.branch.as_deref().unwrap(),
        "branch1"
    );

    // Opening a branch version that main does not have (main only has version 1)
    // must resolve the version on the branch chain.
    let branch_version_open = DatasetBuilder::from_uri(&test_uri)
        .with_branch("feature/nathan/branch3", Some(3))
        .load()
        .await
        .unwrap();
    assert_eq!(
        branch_version_open.manifest.branch.as_deref(),
        Some("feature/nathan/branch3")
    );
    assert_eq!(branch_version_open.version().version, 3);
    assert_eq!(
        branch_version_open.count_rows(None).await.unwrap(),
        checkout_branch3_at_version3.count_rows(None).await.unwrap()
    );
    // A version the branch does not have is still an error, not its latest version.
    let err = DatasetBuilder::from_uri(&test_uri)
        .with_branch("feature/nathan/branch3", Some(99))
        .load()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::DatasetNotFound { .. }), "{err}");
    assert!(err.to_string().contains("feature/nathan/branch3"), "{err}");

    // From the branch's own directory, an older version of that branch is checked out on it.
    let branch_dir_open = DatasetBuilder::from_uri(branch1_dataset.uri())
        .with_branch("branch1", Some(1))
        .load()
        .await
        .unwrap();
    assert_eq!(branch_dir_open.manifest.branch.as_deref(), Some("branch1"));
    assert_eq!(branch_dir_open.version().version, 1);
    assert_eq!(branch_dir_open.count_rows(None).await.unwrap(), 50);

    // Opening at a branch-pointing tag through the builder must check out the
    // tag's branch chain, not main's chain at the tag's version number.
    let tag_open = DatasetBuilder::from_uri(&test_uri)
        .with_tag("tag1")
        .load()
        .await
        .unwrap();
    assert_eq!(tag_open.manifest.branch.as_deref(), Some("dev/branch2"));
    assert_eq!(tag_open.version().version, 3);
    assert_eq!(tag_open.count_rows(None).await.unwrap(), 100);

    // Opening a branch URI with a tag pointing to a non-latest version on that same branch must check out the tag's version.
    main_dataset
        .tags()
        .create("tag_branch1_v1", ("branch1", 1))
        .await
        .unwrap();
    let branch_tag_open = DatasetBuilder::from_uri(branch1_dataset.uri())
        .with_tag("tag_branch1_v1")
        .load()
        .await
        .unwrap();
    assert_eq!(branch_tag_open.manifest.branch.as_deref(), Some("branch1"));
    assert_eq!(branch_tag_open.version().version, 1);
    assert_eq!(branch_tag_open.count_rows(None).await.unwrap(), 50);

    // Malformed branch names are rejected at the boundary
    for bad_name in ["", "branch1/"] {
        let err = main_dataset
            .checkout_version((Some(bad_name), None::<u64>))
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidRef { .. }),
            "checkout of {:?} must be rejected as InvalidRef, got: {}",
            bad_name,
            err
        );
        let err = DatasetBuilder::from_uri(&test_uri)
            .with_branch(bad_name, None)
            .load()
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidRef { .. }),
            "open of {:?} must be rejected as InvalidRef, got: {}",
            bad_name,
            err
        );
    }

    // "main" stays a valid spelling of the main branch on checkout; the JNI
    // bindings construct Ref::Version(Some("main"), _) directly.
    let main_by_name = checkout_branch1.checkout_branch("main").await.unwrap();
    assert_eq!(main_by_name.manifest.branch, None);
    assert_eq!(main_by_name.version().version, 1);
    let main_by_ref = checkout_branch1
        .checkout_version(crate::dataset::refs::Ref::Version(
            Some("main".to_string()),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(main_by_ref.manifest.branch, None);

    // A checkout whose resolved manifest is not on the requested branch must
    // error loudly instead of handing back another branch's data: stage main's
    // manifest under a branch path that was never created, so resolution finds
    // a manifest belonging to main.
    let staged_manifest = main_dataset.manifest_location().path.clone();
    let staged_copy = Path::parse(format!(
        "{}/tree/ghost/_versions/{}",
        test_uri,
        staged_manifest.filename().unwrap()
    ))
    .unwrap();
    main_dataset
        .object_store
        .inner
        .copy(&staged_manifest, &staged_copy)
        .await
        .unwrap();
    let err = main_dataset.checkout_branch("ghost").await.unwrap_err();
    assert!(
        err.to_string().contains("resolved a manifest belonging to"),
        "expected the branch-mismatch guardrail, got: {}",
        err
    );
    main_dataset
        .object_store
        .remove_dir_all(Path::parse(format!("{}/tree/ghost", test_uri)).unwrap())
        .await
        .unwrap();

    let mut dataset = main_dataset;
    // Finally delete all branches
    assert!(matches!(
        dataset.delete_branch("branch1").await,
        Err(Error::RefConflict { message: _ })
    ));
    // Test deleting zombie branch
    let root_location = dataset.refs.root().unwrap();
    let branch_file = branch_contents_path(&root_location.path, "feature/nathan/branch3");
    dataset.object_store.delete(&branch_file).await.unwrap();
    // Now "feature/nathan/branch3" is a zombie branch
    // Use delete_branch to verify if the directory is cleaned up
    dataset
        .force_delete_branch("feature/nathan/branch3")
        .await
        .unwrap();
    let cleaned_path = Path::parse(format!("{}/tree/feature", test_uri)).unwrap();
    assert!(!dataset.object_store.exists(&cleaned_path).await.unwrap());

    dataset.tags().delete("tag1").await.unwrap();
    dataset.tags().delete("tag_branch1_v1").await.unwrap();
    dataset.delete_branch("dev/branch2").await.unwrap();
    dataset.delete_branch("branch1").await.unwrap();

    // Verify list_branches is empty
    let branches_after_delete = dataset.list_branches().await.unwrap();
    assert!(branches_after_delete.is_empty());

    // Verify branch directories are all deleted cleanly
    let test_path = tempdir.obj_path();
    let branches = dataset
        .object_store
        .read_dir(test_path.clone().join("tree"))
        .await
        .unwrap();
    assert!(branches.is_empty());
}

fn id_rows(start: i32, rows: u64) -> impl RecordBatchReader + Send + 'static {
    gen_batch()
        .col("id", array::step_custom::<Int32Type>(start, 1))
        .into_reader_rows(RowCount::from(rows), BatchCount::from(1))
}

// Simulate a branch created before the new name validation.
async fn create_branch_before_name_check(source: &Dataset, branch: &str) {
    let version = source.version().version;
    let source_branch = source.manifest.branch.clone();
    let source_location = source
        .branch_location()
        .find_branch(source_branch.as_deref())
        .unwrap();
    let location = source.branch_location().find_branch(Some(branch)).unwrap();
    let clone = Operation::Clone {
        is_shallow: true,
        ref_name: source_branch.clone(),
        ref_version: version,
        ref_path: source_location.uri,
        branch_name: Some(branch.to_string()),
    };
    CommitBuilder::new(location.uri.as_str())
        .with_store_params(source.store_params.as_deref().cloned().unwrap_or_default())
        .with_object_store(Arc::new(source.object_store.as_ref().clone()))
        .with_commit_handler(source.commit_handler.clone())
        .with_exact_storage_format(source.manifest.data_storage_format.lance_file_format())
        .execute(Transaction::new(version, clone, None))
        .await
        .unwrap();
    source
        .branches()
        .create(branch, version, source_branch.as_deref())
        .await
        .unwrap();
}

#[rstest]
#[case::data_directory("data", true)]
#[case::data_directory_case_alias("Data", true)]
#[case::versions_directory("_versions", true)]
#[case::deletions_directory("_deletions", true)]
#[case::dir_id_mem_wal_directory("_a1b2c3d4_mem_wal", true)]
#[case::ordinary_name("child", false)]
#[tokio::test]
async fn deleting_nested_branch_preserves_parent(
    #[case] segment: &str,
    #[case] directory_stays: bool,
) {
    let tempdir = TempStrDir::default();
    let uri = tempdir.as_str();
    let mut root = Dataset::write(id_rows(0, 4), uri, None).await.unwrap();
    let mut outer = root
        .create_branch("outer", root.version().version, None)
        .await
        .unwrap();
    outer.delete("id = 0").await.unwrap();
    outer.append(id_rows(4, 2), None).await.unwrap();
    let nested = format!("outer/{segment}");
    create_branch_before_name_check(&outer, &nested).await;
    assert!(root.list_branches().await.unwrap().contains_key(&nested));
    let nested_rows = root.checkout_branch(&nested).await.unwrap();
    assert_eq!(nested_rows.count_rows(None).await.unwrap(), 5);

    root.delete_branch(&nested).await.unwrap();

    assert!(!root.list_branches().await.unwrap().contains_key(&nested));
    let nested_dir = std::path::Path::new(uri).join("tree/outer").join(segment);
    if directory_stays {
        let preserved = root.checkout_branch(&nested).await.unwrap_or_else(|error| {
            panic!("deleted branch {nested} should stay openable: {error}")
        });
        assert_eq!(preserved.count_rows(None).await.unwrap(), 5);
    } else {
        assert!(
            !nested_dir.exists(),
            "ordinary nested branch directory should be removed: {}",
            nested_dir.display()
        );
    }
    let outer = Dataset::open(uri)
        .await
        .unwrap()
        .checkout_branch("outer")
        .await
        .unwrap_or_else(|error| panic!("branch outer is gone: {error}"));
    let rows = outer
        .count_rows(Some("id >= 0".to_string()))
        .await
        .unwrap_or_else(|error| panic!("branch outer cannot be scanned: {error}"));
    assert_eq!(rows, 5);
}

#[rstest]
#[case::data_directory("data")]
#[case::versions_directory("_versions")]
#[case::deletions_directory("_deletions")]
#[tokio::test]
async fn create_branch_refuses_a_nested_name_of_a_dataset_directory(#[case] segment: &str) {
    let tempdir = TempStrDir::default();
    let uri = tempdir.as_str();
    let mut root = Dataset::write(id_rows(0, 4), uri, None).await.unwrap();
    let mut outer = root
        .create_branch("outer", root.version().version, None)
        .await
        .unwrap();
    let nested = format!("outer/{segment}");

    let error = outer
        .create_branch(&nested, outer.version().version, None)
        .await
        .unwrap_err();

    assert!(matches!(error, Error::InvalidRef { .. }), "{error}");
    assert!(!root.list_branches().await.unwrap().contains_key(&nested));
    let nested_versions = std::path::Path::new(uri)
        .join("tree/outer")
        .join(segment)
        .join("_versions");
    assert!(!nested_versions.exists(), "{}", nested_versions.display());
    let outer = root.checkout_branch("outer").await.unwrap();
    assert_eq!(outer.count_rows(None).await.unwrap(), 4);
}
