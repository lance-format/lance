// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::blob::test_util::{
    BlobTestFixture, complete_blob_v2_array, complete_blob_v2_field,
    create_multi_base_blob_v2_fixture, managed_blob_schema, nested_blob_v2_batch,
};
use crate::dataset::blob::{BlobRangeRequest, BlobReadRange};
use crate::{
    Dataset,
    blob::{
        BlobArrayBuilder, BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange, PackedBlobWriter,
        blob_field,
    },
    dataset::{
        CommitBuilder, ExternalBlobMode, WriteMode, WriteParams,
        transaction::{DataReplacementGroup, Operation, Transaction},
    },
};
use arrow::{
    array::AsArray,
    datatypes::{UInt8Type, UInt32Type, UInt64Type},
};
use arrow_array::RecordBatch;
use arrow_array::{
    Array, ArrayRef, Int32Array, LargeBinaryArray, RecordBatchIterator, StringArray, StructArray,
    UInt8Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chrono::Utc;
use futures::TryStreamExt;
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_META_KEY, BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
    BLOB_V2_EXT_NAME, DataTypeExt,
};
use lance_core::{
    Error, ROW_ID, Result,
    utils::tempfile::{TempDir, TempStrDir},
};
use lance_core::{
    datatypes::{BLOB_V2_LOGICAL_FIELDS, BlobHandling, BlobKind},
    utils::blob::blob_path,
};
use lance_datagen::{BatchCount, RowCount, array};
use lance_file::{version::LanceFileVersion, writer::FileWriterOptions};
use lance_io::stream::RecordBatchStream;
use lance_table::feature_flags::{FLAG_MANAGED_BLOBS, MANAGED_BLOBS_META_KEY};
use lance_table::format::BasePath;
use rstest::rstest;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[rstest]
#[tokio::test]
async fn managed_append_preserves_primary_base_root_semantics(
    #[values(false, true)] is_dataset_root: bool,
) {
    let test_dir = TempStrDir::default();
    let payload = vec![7; 100 * 1024];
    let make_reader = || {
        let mut blobs = BlobArrayBuilder::new(1);
        blobs.push_bytes(&payload).unwrap();
        let schema = Arc::new(managed_blob_schema(vec![blob_field("blob", false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        RecordBatchIterator::new(vec![Ok(batch)], schema)
    };
    Dataset::write(
        make_reader(),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: Some(vec![BasePath::new(
                7,
                test_dir.to_string(),
                None,
                is_dataset_root,
            )]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            make_reader(),
            &test_dir,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let blobs = dataset
        .take_blobs_by_indices(&[0, 1], "blob")
        .await
        .unwrap();
    for blob in blobs {
        assert_eq!(blob.unwrap().read().await.unwrap().as_ref(), payload);
    }
    assert_eq!(dataset.manifest.base_paths.len(), 1);
    let flags = FLAG_MANAGED_BLOBS
        | lance_table::feature_flags::FLAG_BASE_PATHS
        | lance_table::feature_flags::FLAG_MIXED_DATA_FILE_VERSIONS;
    assert_eq!(dataset.manifest.reader_feature_flags & flags, flags);
    assert_eq!(dataset.manifest.writer_feature_flags & flags, flags);
    assert_eq!(
        dataset.manifest.base_paths[&7].is_dataset_root,
        is_dataset_root
    );
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata.files[0].base_id.is_none())
    );
}

#[rstest]
#[case::local(None)]
#[case::registered(Some(7))]
#[case::maximum(Some(u32::MAX))]
#[tokio::test]
async fn managed_prepared_write_validates_registered_base(
    #[case] base_id: Option<u32>,
    #[values(false, true)] registered: bool,
    #[values(false, true)] nested: bool,
    #[values(false, true)] enabled: bool,
) {
    let table = TempStrDir::default();
    let objects = TempStrDir::default();
    let root: &str = if base_id.is_some() {
        objects.as_ref()
    } else {
        table.as_ref()
    };
    std::fs::create_dir_all(std::path::Path::new(root).join("_blobs")).unwrap();
    std::fs::write(
        std::path::Path::new(root).join("_blobs/test.blob"),
        b"payload",
    )
    .unwrap();
    let mut values = BlobDescriptorArrayBuilder::new("blob");
    values
        .push(BlobDescriptor::Managed {
            base_id,
            uri: "_blobs/test.blob".to_string(),
            offset: 1,
            size: 3,
        })
        .unwrap();
    let (field, values) = values.finish().unwrap().into_parts();
    let (field, values): (Field, ArrayRef) = if nested {
        let fields: arrow_schema::Fields = vec![field].into();
        (
            Field::new("nested", DataType::Struct(fields.clone()), false),
            Arc::new(StructArray::new(fields, vec![values], None)),
        )
    } else {
        (field, values)
    };
    let schema = Arc::new(if enabled {
        managed_blob_schema(vec![field])
    } else {
        Schema::new(vec![field])
    });
    let batch = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
    let result = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &table,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: base_id
                .filter(|_| registered)
                .map(|id| vec![BasePath::new(id, objects.to_string(), None, false)]),
            ..Default::default()
        }),
    )
    .await;
    if !enabled || (base_id.is_some() && !registered) {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        let message = if enabled {
            "unknown base_id"
        } else {
            "lance.blob.managed=true"
        };
        assert!(error.to_string().contains(message), "{error}");
    } else {
        let dataset = Arc::new(result.unwrap());
        assert_eq!(
            dataset.manifest.base_paths.len(),
            usize::from(base_id.is_some())
        );
        let blobs = dataset
            .take_blobs_by_indices(&[0], if nested { "nested.blob" } else { "blob" })
            .await
            .unwrap();
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"ayl"
        );
    }
}

#[rstest]
#[case::adopt_stable(true)]
#[case::reuse_managed(false)]
#[tokio::test]
async fn managed_compaction_preserves_sidecars_after_source_gc(
    #[case] released: bool,
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    mock_instant::thread_local::MockClock::set_system_time(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap(),
    );
    let test_dir = TempStrDir::default();
    let payload = vec![b'p'; 16];
    let dedicated = vec![b'd'; 32];
    let mut blobs = BlobArrayBuilder::new(6);
    blobs.push_bytes(&payload).unwrap();
    blobs.push_bytes(b"inline").unwrap();
    blobs.push_null().unwrap();
    blobs.push_bytes(b"").unwrap();
    blobs.push_bytes(&dedicated).unwrap();
    blobs.push_bytes(b"inline").unwrap();
    let mut field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.extend(HashMap::from([
        (
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
            "8".to_string(),
        ),
        (
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            "24".to_string(),
        ),
    ]));
    field = field.with_metadata(metadata);
    let schema = Arc::new(managed_blob_schema(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let old_data = crate::utils::test::copy_test_data_to_tmp("v11.0.0/blob_sidecars").unwrap();
    let mut dataset = if released {
        let mut dataset = Dataset::open(old_data.path_str().as_str()).await.unwrap();
        assert!(!dataset.manifest.has_managed_blobs());
        dataset
            .update_config([("test", "metadata-only")])
            .await
            .unwrap();
        assert!(!dataset.manifest.has_managed_blobs());
        dataset
            .update_schema_metadata([(MANAGED_BLOBS_META_KEY, "true")])
            .await
            .unwrap();
        dataset
    } else {
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 3,
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    };
    assert_eq!(dataset.manifest.fragments.len(), 2);
    let before = dataset
        .object_store
        .read_dir_all(&dataset.base, None)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let sidecars = before
        .iter()
        .filter(|object| object.location.extension() == Some("blob"))
        .map(|object| (object.location.clone(), object.size))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert!(!sidecars.is_empty());
    mock_instant::thread_local::MockClock::set_system_time(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap(),
    );
    crate::dataset::optimize::compact_files(
        &mut dataset,
        crate::dataset::optimize::CompactionOptions {
            data_storage_version: Some(version),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.fragments.len(), 1);
    assert!(dataset.manifest.has_managed_blobs());
    let cleanup_stats = crate::dataset::cleanup::cleanup_old_versions(
        &dataset,
        crate::dataset::cleanup::CleanupPolicy {
            before_timestamp: Some(Utc::now() + chrono::TimeDelta::seconds(1)),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let after = dataset
        .object_store
        .read_dir_all(&dataset.base, None)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let remaining_sidecars = after
        .iter()
        .filter(|object| object.location.extension() == Some("blob"))
        .map(|object| (object.location.clone(), object.size))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(sidecars, remaining_sidecars);
    for source in before
        .iter()
        .filter(|object| object.location.extension() == Some("lance"))
    {
        assert!(
            !after
                .iter()
                .any(|object| object.location == source.location),
            "source={source:?}, stats={cleanup_stats:?}, fragments={:?}",
            dataset.manifest.fragments
        );
    }
    // Move the complete table and remove the old location. Both adopted
    // sidecars and newly written Managed objects must follow the table.
    let moved_dir = TempStrDir::default();
    std::fs::remove_dir(moved_dir.as_ref()).unwrap();
    let old_path = if released {
        old_data.path_str()
    } else {
        test_dir.to_string()
    };
    drop(dataset);
    std::fs::rename(&old_path, moved_dir.as_ref()).unwrap();
    let dataset = Dataset::open(&moved_dir).await.unwrap();
    assert!(dataset.manifest.base_paths.is_empty());
    crate::dataset::cleanup::cleanup_old_versions(
        &dataset,
        crate::dataset::cleanup::CleanupPolicy {
            before_timestamp: Some(Utc::now() + chrono::TimeDelta::seconds(1)),
            delete_unverified: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut scan = dataset.scan();
    scan.blob_handling(BlobHandling::AllBinary);
    let scanned = scan.try_into_batch().await.unwrap();
    assert_eq!(scanned.num_rows(), 6);
    assert_eq!(scanned["blob"].as_binary::<i64>().value(0), payload);
    let values = Arc::new(dataset)
        .take_blobs_by_indices(&[0, 1, 2, 3, 4, 5], "blob")
        .await
        .unwrap();
    for index in [0, 4] {
        assert_eq!(values[index].as_ref().unwrap().kind(), BlobKind::Managed);
        assert_eq!(
            values[index]
                .as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .as_ref(),
            if index == 0 { &payload } else { &dedicated }.as_slice()
        );
    }
    for index in [1, 5] {
        assert_eq!(values[index].as_ref().unwrap().kind(), BlobKind::Inline);
        assert_eq!(
            values[index]
                .as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .as_ref(),
            b"inline"
        );
    }
    assert!(values[2].is_none());
    assert!(values[3].as_ref().unwrap().read().await.unwrap().is_empty());
}

/// Assert the latest commit came from merge_insert's row-rewrite path. Blob
/// input in a layout other than the table's must not reach the column updater,
/// which cannot write it; which path a merge takes depends on how many rows of
/// each fragment it updates, so a passing result alone does not show it.
async fn assert_rewrote_rows(dataset: &Dataset) {
    use crate::dataset::transaction::UpdateMode;
    let transaction = dataset.read_transaction().await.unwrap().unwrap();
    assert!(
        matches!(
            transaction.operation,
            Operation::Update {
                update_mode: Some(UpdateMode::RewriteRows),
                ..
            }
        ),
        "{:?}",
        transaction.operation
    );
}

/// `column`, a blob column named `blob`, as written: inside a `meta` struct
/// when `nested`.
fn place_blob(nested: bool, field: Field, column: ArrayRef) -> (Field, ArrayRef) {
    if !nested {
        return (field, column);
    }
    let fields: arrow_schema::Fields = vec![field].into();
    (
        Field::new("meta", DataType::Struct(fields.clone()), true),
        Arc::new(StructArray::new(fields, vec![column], None)),
    )
}

/// A table with managed blobs of rows `id` 0 and 1, each with blob bytes
/// "seed" in the minimal or `complete` logical layout, and `other` = 6, 7 when
/// `other`.
async fn seed_blob_table(
    dir: &str,
    complete: bool,
    nested: bool,
    other: bool,
    params: WriteParams,
) -> Dataset {
    let (blob, seed) = if complete {
        let seed = complete_blob_v2_array(
            vec![Some(b"seed".to_vec()); 2],
            vec![None; 2],
            vec![None; 2],
            vec![None; 2],
            None,
        );
        (complete_blob_v2_field("blob", true), seed)
    } else {
        let mut seed = BlobArrayBuilder::new(2);
        seed.push_bytes(b"seed").unwrap();
        seed.push_bytes(b"seed").unwrap();
        (blob_field("blob", true), seed.finish().unwrap())
    };
    let (blob, seed) = place_blob(nested, blob, seed);
    let mut fields = vec![Field::new("id", DataType::Int32, false), blob];
    let mut columns = vec![Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef, seed];
    if other {
        fields.push(Field::new("other", DataType::Int32, true));
        columns.push(Arc::new(Int32Array::from(vec![6, 7])));
    }
    let schema = Arc::new(managed_blob_schema(fields));
    let seed = RecordBatch::try_new(schema.clone(), columns).unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(seed)], schema),
        dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..params
        }),
    )
    .await
    .unwrap()
}

/// Upsert `source` by `id`, checking the merge took the row-rewrite path.
async fn merge_by_id(
    dataset: Dataset,
    source: RecordBatch,
) -> (
    Arc<Dataset>,
    crate::dataset::write::merge_insert::MergeStats,
) {
    use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    let schema = source.schema();
    let mut builder =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll);
    let merged = builder
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new(vec![Ok(source)], schema))
        .await
        .unwrap();
    assert_rewrote_rows(&merged.0).await;
    merged
}

/// Each row's blob bytes and `other` value (`None` when null or absent), by id.
async fn rows_by_id(dataset: &Dataset, nested: bool) -> HashMap<i32, (Vec<u8>, Option<i32>)> {
    use arrow_array::types::Int32Type;
    let mut scan = dataset.scan();
    scan.blob_handling(BlobHandling::AllBinary);
    let rows = scan.try_into_batch().await.unwrap();
    let ids = rows["id"].as_primitive::<Int32Type>();
    let blobs = if nested {
        rows["meta"].as_struct()["blob"].as_binary::<i64>()
    } else {
        rows["blob"].as_binary::<i64>()
    };
    let other = rows
        .column_by_name("other")
        .map(|other| other.as_primitive::<Int32Type>());
    (0..rows.num_rows())
        .map(|row| {
            let other = other.and_then(|other| other.is_valid(row).then(|| other.value(row)));
            (ids.value(row), (blobs.value(row).to_vec(), other))
        })
        .collect()
}

/// merge_insert takes writer-prepared input as append does, so a Managed
/// reference to an object already under the table is adopted, for an
/// updated row and an inserted one, without its bytes passing through the
/// writer. A source may also leave out table columns, which then keep their
/// values, the table may declare the blob in either logical layout, and the
/// blob may sit inside a struct.
#[rstest]
#[tokio::test]
async fn merge_insert_adopts_prepared_managed_references(
    #[values(false, true)] partial: bool,
    #[values(false, true)] complete: bool,
    #[values(false, true)] nested: bool,
) {
    let test_dir = TempStrDir::default();
    let dataset =
        seed_blob_table(&test_dir, complete, nested, partial, WriteParams::default()).await;
    let blobs_dir = std::path::Path::new(test_dir.as_ref()).join("_blobs");
    std::fs::create_dir_all(&blobs_dir).unwrap();
    std::fs::write(blobs_dir.join("adopted.blob"), vec![b'v'; 1000]).unwrap();

    // Two rows share one object, as rows from one packed file do.
    let mut prepared = BlobDescriptorArrayBuilder::new("blob");
    for (offset, size) in [(0, 400), (400, 200)] {
        prepared
            .push(BlobDescriptor::Managed {
                base_id: None,
                uri: "_blobs/adopted.blob".to_string(),
                offset,
                size,
            })
            .unwrap();
    }
    let (blob, prepared) = prepared.finish().unwrap().into_parts();
    let (blob, prepared) = place_blob(nested, blob, prepared);
    let source = RecordBatch::try_new(
        Arc::new(managed_blob_schema(vec![
            Field::new("id", DataType::Int32, false),
            blob,
        ])),
        vec![Arc::new(Int32Array::from(vec![1, 2])), prepared],
    )
    .unwrap();
    let (dataset, stats) = merge_by_id(dataset, source).await;
    assert_eq!((stats.num_updated_rows, stats.num_inserted_rows), (1, 1));

    let other = |value| partial.then_some(value);
    assert_eq!(
        rows_by_id(&dataset, nested).await,
        HashMap::from([
            (0, (b"seed".to_vec(), other(6))),
            (1, (vec![b'v'; 400], other(7))),
            (2, (vec![b'v'; 200], None)),
        ])
    );
    let path = if nested { "meta.blob" } else { "blob" };
    for blob in dataset.take_blobs_by_indices(&[1, 2], path).await.unwrap() {
        let blob = blob.unwrap();
        assert_eq!(blob.kind(), BlobKind::Managed);
        assert_eq!(blob.uri(), Some("_blobs/adopted.blob"));
    }
}

/// A logical blob's byte range survives a merge_insert that leaves out other
/// table columns: the row reads back only its range of the object.
#[tokio::test]
async fn merge_insert_keeps_a_logical_blob_range() {
    let test_dir = TempStrDir::default();
    let external_dir = TempDir::default();
    let external = external_dir.std_path().join("external.bin");
    std::fs::write(&external, b"0123456789").unwrap();
    let base = BasePath {
        id: 1,
        name: Some("external".to_string()),
        path: format!("file://{}", external_dir.std_path().display()),
        is_dataset_root: false,
    };
    let params = WriteParams {
        initial_bases: Some(vec![base]),
        ..Default::default()
    };
    let dataset = seed_blob_table(&test_dir, false, false, true, params).await;

    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            complete_blob_v2_field("blob", true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            complete_blob_v2_array(
                vec![None],
                vec![Some(format!("file://{}", external.display()))],
                vec![Some(2)],
                vec![Some(3)],
                None,
            ),
        ],
    )
    .unwrap();
    let (dataset, _) = merge_by_id(dataset, source).await;
    assert_eq!(
        rows_by_id(&dataset, false).await,
        HashMap::from([
            (0, (b"seed".to_vec(), Some(6))),
            (1, (b"234".to_vec(), Some(7)))
        ])
    );
}

#[tokio::test]
async fn managed_flag_survives_restore_of_unflagged_snapshot() {
    let dir = crate::utils::test::copy_test_data_to_tmp("v11.0.0/blob_sidecars").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    let mut old = dataset.checkout_version(1).await.unwrap();
    assert!(!old.manifest.has_managed_blobs());
    dataset
        .update_schema_metadata([(MANAGED_BLOBS_META_KEY, "true")])
        .await
        .unwrap();
    crate::dataset::optimize::compact_files(&mut dataset, Default::default(), None)
        .await
        .unwrap();
    assert!(dataset.manifest.has_managed_blobs());
    old.restore().await.unwrap();
    assert!(old.manifest.has_managed_blobs());
    assert_ne!(old.manifest.writer_feature_flags & FLAG_MANAGED_BLOBS, 0);
    let values = Arc::new(old)
        .take_blobs_by_indices(&[0, 4], "blob")
        .await
        .unwrap();
    assert_eq!(values[0].as_ref().unwrap().kind(), BlobKind::Packed);
    assert_eq!(values[1].as_ref().unwrap().kind(), BlobKind::Dedicated);
    assert_eq!(
        values[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"pppppppppppppppp"
    );
}

#[rstest]
#[case::append(WriteMode::Append)]
#[case::overwrite(WriteMode::Overwrite)]
#[tokio::test]
async fn managed_flag_activates_only_on_opted_in_blob_write(
    #[case] mode: WriteMode,
    #[values(false, true)] enabled: bool,
    #[values(6, 16, 32)] size: usize,
) {
    let dir = crate::utils::test::copy_test_data_to_tmp("v11.0.0/blob_sidecars").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    let files = dataset.manifest.fragments[0].files.clone();
    dataset.delete("_rowid = 1").await.unwrap();
    assert!(!dataset.manifest.has_managed_blobs());
    assert_eq!(dataset.manifest.fragments[0].files, files);
    assert!(dataset.manifest.fragments[0].deletion_file.is_some());

    dataset
        .update_schema_metadata([(
            MANAGED_BLOBS_META_KEY,
            if enabled { "true" } else { "false" },
        )])
        .await
        .unwrap();
    assert_eq!(dataset.manifest.has_managed_blobs(), enabled);
    let metadata = dataset.schema().field("blob").unwrap().metadata.clone();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![blob_field("blob", true).with_metadata(metadata)],
        dataset.schema().metadata.clone(),
    ));
    let payload = vec![b'p'; size];
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_bytes(&payload).unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &dir.path_str(),
        Some(WriteParams {
            mode,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let flag = FLAG_MANAGED_BLOBS;
    let expected_flags = if enabled { flag } else { 0 };
    assert_eq!(dataset.manifest.reader_feature_flags & flag, expected_flags);
    assert_eq!(dataset.manifest.writer_feature_flags & flag, expected_flags);
    let mut scan = dataset.scan();
    scan.blob_handling(BlobHandling::AllBinary);
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(
        batch["blob"].as_binary::<i64>().value(batch.num_rows() - 1),
        payload.as_slice()
    );
}

#[rstest]
#[tokio::test]
async fn blob_compaction_does_not_enable_managed_by_default(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    let dir = crate::utils::test::copy_test_data_to_tmp("v11.0.0/blob_sidecars").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    crate::dataset::optimize::compact_files(
        &mut dataset,
        crate::dataset::optimize::CompactionOptions {
            data_storage_version: Some(version),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.fragments.len(), 1);
    let flag = FLAG_MANAGED_BLOBS;
    assert_eq!(dataset.manifest.reader_feature_flags & flag, 0);
    assert_eq!(dataset.manifest.writer_feature_flags & flag, 0);
    let blobs = Arc::new(dataset)
        .take_blobs_by_indices(&[0, 4], "blob")
        .await
        .unwrap();
    assert_eq!(blobs[0].as_ref().unwrap().kind(), BlobKind::Packed);
    assert_eq!(blobs[1].as_ref().unwrap().kind(), BlobKind::Dedicated);
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"pppppppppppppppp"
    );
}

#[rstest]
#[case::drop_column(false)]
#[case::overwrite(true)]
#[tokio::test]
async fn managed_flag_survives_removing_blob_columns(#[case] overwrite: bool) {
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_bytes(b"inline").unwrap();
    let schema = Arc::new(managed_blob_schema(vec![
        Field::new("id", DataType::Int32, false),
        blob_field("blob", false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1])), blobs.finish().unwrap()],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if overwrite {
        let batch = arrow_array::record_batch!(("id", Int32, [2])).unwrap();
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            Arc::new(dataset.clone()),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    } else {
        dataset.drop_columns(&["blob"]).await.unwrap();
    }
    dataset
        .update_schema_metadata([(MANAGED_BLOBS_META_KEY, "false")])
        .await
        .unwrap();
    dataset
        .update_config([("test", "metadata-only")])
        .await
        .unwrap();
    let flag = FLAG_MANAGED_BLOBS;
    assert_eq!(dataset.manifest.reader_feature_flags & flag, flag);
    assert_eq!(dataset.manifest.writer_feature_flags & flag, flag);
    assert!(dataset.schema().field("blob").is_none());
    assert_eq!(dataset.scan().try_into_batch().await.unwrap().num_rows(), 1);
}

#[rstest]
#[tokio::test]
async fn managed_clone_preserves_existing_retention_contract(
    #[values(false, true)] deep: bool,
    #[values(false, true)] registered: bool,
    #[values(None, Some("false"), Some("true"))] opt_in: Option<&str>,
) {
    let source_dir = TempStrDir::default();
    let clone_dir = TempStrDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    std::fs::write(&external_path, b"external").unwrap();
    let payload = vec![9; 16];
    let dedicated_payload = vec![7; 32];
    let mut blobs = BlobArrayBuilder::new(6);
    for row in 0..6 {
        match row {
            2 => blobs
                .push_uri(format!("file://{}", external_path.display()))
                .unwrap(),
            4 => blobs.push_bytes(b"inline").unwrap(),
            _ => blobs.push_bytes(&payload).unwrap(),
        }
    }
    let schema = Arc::new(managed_blob_schema(vec![
        Field::new("id", DataType::Int32, false),
        blob_field("blob", true).with_metadata(HashMap::from([
            (ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string()),
            (
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
                "8".to_string(),
            ),
            (
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                "24".to_string(),
            ),
        ])),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            blobs.finish().unwrap(),
        ],
    )
    .unwrap();
    let mut source = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &source_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_3),
            initial_bases: registered
                .then(|| vec![BasePath::new(7, source_dir.to_string(), None, true)]),
            target_bases: registered.then_some(vec![7]),
            max_rows_per_file: 3,
            allow_external_blob_outside_bases: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    source.delete("id IN (1, 3, 5)").await.unwrap();
    source
        .update_schema_metadata([(MANAGED_BLOBS_META_KEY, opt_in)])
        .await
        .unwrap();
    if opt_in != Some("true") {
        let mut blobs = BlobArrayBuilder::new(2);
        blobs.push_bytes(&payload).unwrap();
        blobs.push_bytes(&dedicated_payload).unwrap();
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int32, false),
                blob_field("blob", true)
                    .with_metadata(source.schema().field("blob").unwrap().metadata.clone()),
            ],
            source.schema().metadata.clone(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![6, 7])),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap();
        source = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(source.clone()),
            Some(WriteParams {
                mode: WriteMode::Append,
                target_bases: registered.then_some(vec![7]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    }
    let version = source.version_id();
    let mut cloned = if deep {
        source.deep_clone(&clone_dir, version, None).await.unwrap()
    } else {
        source.tags().create("clone_source", version).await.unwrap();
        source
            .shallow_clone(&clone_dir, "clone_source", None)
            .await
            .unwrap()
    };
    assert!(cloned.manifest.has_managed_blobs());
    assert_eq!(cloned.schema().metadata, source.schema().metadata);
    if deep && !registered {
        assert_eq!(source.manifest.fragments, cloned.manifest.fragments);
        for fragment in source.get_fragments() {
            for file in &fragment.metadata.files {
                let relative = format!("data/{}", file.path);
                assert_eq!(
                    std::fs::read(std::path::Path::new(source_dir.as_ref()).join(&relative))
                        .unwrap(),
                    std::fs::read(std::path::Path::new(clone_dir.as_ref()).join(&relative))
                        .unwrap(),
                );
            }
        }
    }
    if !deep {
        crate::dataset::optimize::compact_files(&mut cloned, Default::default(), None)
            .await
            .unwrap();
    }
    source.checkout_latest().await.unwrap();
    assert_eq!(source.version_id(), version);
    if deep {
        std::fs::remove_dir_all(source_dir.as_ref()).unwrap();
    } else {
        source.delete("true").await.unwrap();
        let version = source.version_id();
        crate::dataset::cleanup::cleanup_old_versions(
            &source,
            crate::dataset::cleanup::CleanupPolicy {
                before_version: Some(version),
                error_if_tagged_old_versions: false,
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        source.checkout_latest().await.unwrap();
        assert_eq!(source.version_id(), version);
        assert_eq!(source.count_rows(None).await.unwrap(), 0);
    }
    let num_rows = if opt_in == Some("true") { 3 } else { 5 };
    assert_eq!(cloned.count_rows(None).await.unwrap(), num_rows);
    let blobs = Arc::new(cloned)
        .take_blobs_by_indices(&(0..num_rows as u64).collect::<Vec<_>>(), "blob")
        .await
        .unwrap();
    let mut expected = vec![
        (
            if deep || opt_in == Some("true") {
                BlobKind::Managed
            } else {
                BlobKind::Packed
            },
            payload.as_slice(),
        ),
        (BlobKind::External, b"external".as_slice()),
        (BlobKind::Inline, b"inline".as_slice()),
    ];
    if opt_in != Some("true") {
        expected.extend([
            (
                if deep && registered {
                    BlobKind::Managed
                } else {
                    BlobKind::Packed
                },
                payload.as_slice(),
            ),
            (
                if deep && registered {
                    BlobKind::Managed
                } else {
                    BlobKind::Dedicated
                },
                dedicated_payload.as_slice(),
            ),
        ]);
    }
    for (blob, (kind, bytes)) in blobs.into_iter().zip(expected) {
        let blob = blob.unwrap();
        assert_eq!(blob.kind(), kind);
        assert_eq!(blob.read().await.unwrap().as_ref(), bytes);
    }
}

async fn stable_row_id_blob_dataset_with_deleted_row() -> (TempStrDir, Arc<Dataset>, Vec<u64>) {
    let test_dir = TempStrDir::default();
    let mut blob_builder = BlobArrayBuilder::new(3);
    blob_builder.push_bytes(b"AAaa").unwrap();
    blob_builder.push_bytes(b"BBbb").unwrap();
    blob_builder.push_bytes(b"CCcc").unwrap();
    let schema = Arc::new(Schema::new(vec![
        blob_field("blob", true),
        Field::new("idx", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            blob_builder.finish().unwrap(),
            Arc::new(UInt64Array::from(vec![0, 1, 2])),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let with_row_ids = dataset
        .scan()
        .project(&["idx"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let indices = with_row_ids["idx"].as_primitive::<UInt64Type>();
    let ids = with_row_ids[ROW_ID].as_primitive::<UInt64Type>();
    let mut row_ids = vec![0; with_row_ids.num_rows()];
    for (index, row_id) in indices.values().iter().zip(ids.values()) {
        row_ids[*index as usize] = *row_id;
    }

    dataset.delete("idx = 1").await.unwrap();
    (test_dir, Arc::new(dataset), row_ids)
}

#[tokio::test]
pub async fn test_take_blobs() {
    let fixture = BlobTestFixture::new().await;

    let row_ids = fixture
        .dataset
        .scan()
        .project::<String>(&[])
        .unwrap()
        .filter("filterme >= 10")
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().values();
    let row_ids = vec![row_ids[5], row_ids[17], row_ids[23]];

    let blobs = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

    for (actual_idx, (expected_batch_idx, expected_row_idx)) in
        [(1, 5), (2, 7), (3, 3)].iter().enumerate()
    {
        let val = blobs[actual_idx].as_ref().unwrap().read().await.unwrap();
        let expected = fixture.data[*expected_batch_idx]
            .column(1)
            .as_binary::<i64>()
            .value(*expected_row_idx);

        assert_eq!(&val, expected);
    }
}

#[tokio::test]
async fn test_read_blobs_requires_selection() {
    let fixture = BlobTestFixture::new().await;

    let err = fixture.dataset.read_blobs("blobs").unwrap().execute().await;

    assert!(matches!(err, Err(Error::InvalidInput { .. })));
    assert!(
        err.unwrap_err()
            .to_string()
            .contains("requires a row selection")
    );
}

#[tokio::test]
async fn test_read_blobs_by_indices_execute() {
    let fixture = BlobTestFixture::new().await;
    let indices = vec![2, 12, 22];

    let blobs = fixture
        .dataset
        .read_blobs("blobs")
        .unwrap()
        .with_row_indices(indices)
        .execute()
        .await
        .unwrap();

    assert_eq!(blobs.len(), 3);
    for (actual_idx, (expected_batch_idx, expected_row_idx)) in
        [(0, 2), (1, 2), (2, 2)].iter().enumerate()
    {
        let expected = fixture.data[*expected_batch_idx]
            .column(1)
            .as_binary::<i64>()
            .value(*expected_row_idx);
        assert_eq!(blobs[actual_idx].data.as_deref(), Some(expected));
    }
}

#[tokio::test]
pub async fn test_take_blobs_by_indices() {
    let fixture = BlobTestFixture::new().await;

    let fragments = fixture.dataset.fragments();
    assert!(fragments.len() >= 2);
    let mut indices = Vec::with_capacity(fragments.len());
    let mut last = 2;

    for frag in fragments.iter() {
        indices.push(last as u64);
        last += frag.num_rows().unwrap_or(0);
    }
    indices.pop();

    // Row indices
    assert_eq!(indices, [2, 12, 22]);
    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&indices, "blobs")
        .await
        .unwrap();

    // Row IDs
    let row_ids = fragments
        .iter()
        .map(|frag| (frag.id << 32) + 2)
        .collect::<Vec<_>>();
    let blobs2 = fixture.dataset.take_blobs(&row_ids, "blobs").await.unwrap();

    for (blob1, blob2) in blobs.iter().zip(blobs2.iter()) {
        let blob1 = blob1.as_ref().unwrap();
        let blob2 = blob2.as_ref().unwrap();
        assert_eq!(blob1.position(), blob2.position());
        assert_eq!(blob1.size(), blob2.size());
        assert_eq!(blob1.data_path(), blob2.data_path());
    }

    // Unsorted indices spanning fragments use the take remapping path, which
    // carries _rowaddr internally and must still preserve the requested order.
    let indices = [33_u64, 17, 5, 28, 12, 39];
    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&indices, "blobs")
        .await
        .unwrap();
    for (blob, index) in blobs.iter().zip(indices) {
        let actual = blob.as_ref().unwrap().read().await.unwrap();
        let index = index as usize;
        let expected = fixture.data[index / 10]
            .column(1)
            .as_binary::<i64>()
            .value(index % 10);
        assert_eq!(actual.as_ref(), expected);
    }
}

#[rstest]
#[case::all_valid_first(false)]
#[case::nullable_first(true)]
#[tokio::test]
async fn test_write_blob_batches_with_mixed_nullability(#[case] nulls_first: bool) {
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![
        Field::new("blob", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            BLOB_META_KEY.to_string(),
            "true".to_string(),
        )])),
    ]));
    let batch = |values: Vec<Option<&[u8]>>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeBinaryArray::from(values))],
        )
        .unwrap()
    };

    // Definition-level semantics come from each batch's validity bitmap. Both
    // transitions must start a new descriptor page so nulls remain distinct
    // from valid empty blobs.
    let all_valid = batch(vec![Some(b"a".as_slice()), Some(b"".as_slice())]);
    let with_null = batch(vec![Some(b"c".as_slice()), None]);
    let (batches, expected) = if nulls_first {
        (
            vec![with_null, all_valid],
            vec![Some(b"c".as_slice()), None, Some(b"a"), Some(b"")],
        )
    } else {
        (
            vec![all_valid, with_null],
            vec![Some(b"a".as_slice()), Some(b""), Some(b"c"), None],
        )
    };

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    let dataset = Arc::new(
        Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let blobs = dataset
        .take_blobs_by_indices(&[0, 1, 2, 3], "blob")
        .await
        .unwrap();
    assert_eq!(blobs.len(), expected.len());
    for (row_idx, (blob, expected)) in blobs.into_iter().zip(expected).enumerate() {
        let actual = match blob {
            Some(blob) => Some(blob.read().await.unwrap()),
            None => None,
        };
        assert_eq!(actual.as_deref(), expected, "row {row_idx}");
    }
}

#[tokio::test]
pub async fn test_take_blob_id_not_exist() {
    let fixture = BlobTestFixture::new().await;

    let err = fixture.dataset.take_blobs(&[1000], "blobs").await;

    assert!(matches!(err, Err(Error::InvalidInput { .. })));
}

#[tokio::test]
async fn test_take_blobs_by_ids_rejects_deleted_stable_row_id() {
    let (_test_dir, dataset, row_ids) = stable_row_id_blob_dataset_with_deleted_row().await;

    let err = dataset.take_blobs(&row_ids, "blob").await.unwrap_err();

    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("Could not resolve all requested row IDs")
    );
}

#[tokio::test]
async fn test_read_blobs_by_ids_rejects_deleted_stable_row_id() {
    let (_test_dir, dataset, row_ids) = stable_row_id_blob_dataset_with_deleted_row().await;

    let err = dataset
        .read_blobs("blob")
        .unwrap()
        .with_row_ids(row_ids)
        .execute()
        .await
        .unwrap_err();

    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("Could not resolve all requested row IDs")
    );
}

#[tokio::test]
async fn test_read_blob_ranges_by_ids_rejects_deleted_stable_row_id() {
    let (_test_dir, dataset, row_ids) = stable_row_id_blob_dataset_with_deleted_row().await;
    let requests = [
        BlobRangeRequest::new(row_ids[0], 0, 2),
        BlobRangeRequest::new(row_ids[1], 0, 2),
        BlobRangeRequest::new(row_ids[2], 2, 2),
    ];

    let err = dataset
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_ids(requests)
        .execute()
        .await
        .unwrap_err();

    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(
        err.to_string()
            .contains("Could not resolve all requested row IDs")
    );
}

#[tokio::test]
pub async fn test_take_blob_not_blob_col() {
    let fixture = BlobTestFixture::new().await;

    let err = fixture.dataset.take_blobs(&[0], "filterme").await;

    assert!(matches!(err, Err(Error::InvalidInput { .. })));
    assert!(err.unwrap_err().to_string().contains("not a blob column"));
}

#[tokio::test]
pub async fn test_scan_blobs() {
    let fixture = BlobTestFixture::new().await;

    // By default, scanning a blob column will load descriptions
    let batches = fixture
        .dataset
        .scan()
        .project(&["blobs"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();

    let schema = batches.schema();

    assert!(schema.fields[0].data_type().is_struct());

    let batches = batches.try_collect::<Vec<_>>().await.unwrap();

    assert_eq!(batches.len(), 4);
    for batch in batches.iter() {
        assert_eq!(batch.num_columns(), 1);
        assert!(batch.column(0).data_type().is_struct());
    }

    // Should also be able to scan with filter
    let batches = fixture
        .dataset
        .scan()
        .project(&["blobs"])
        .unwrap()
        .filter("filterme = 30")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap();

    let schema = batches.schema();

    assert!(schema.fields[0].data_type().is_struct());

    let batches = batches.try_collect::<Vec<_>>().await.unwrap();

    assert_eq!(batches.len(), 1);
    for batch in batches.iter() {
        assert_eq!(batch.num_columns(), 1);
        assert!(batch.column(0).data_type().is_struct());
    }
}

/// Test that take_blobs_by_indices works correctly with enable_stable_row_ids=true.
///
/// This is a regression test for a bug where take_blobs_by_indices would fail
/// with "index out of bounds" for fragment 1+ when stable row IDs are enabled.
/// The bug was caused by passing row addresses (from row_offsets_to_row_addresses)
/// to blob::take_blobs which expected row IDs. When stable row IDs are enabled,
/// row addresses (fragment_id << 32 | offset) are different from row IDs
/// (sequential integers), causing the row ID index lookup to fail for fragment 1+.
#[tokio::test]
pub async fn test_take_blobs_by_indices_with_stable_row_ids() {
    use crate::dataset::WriteParams;
    use arrow_array::RecordBatchIterator;

    let test_dir = TempStrDir::default();

    // Create test data with blob column
    let data = lance_datagen::gen_batch()
        .col("filterme", array::step::<UInt64Type>())
        .col("blobs", array::blob())
        .into_reader_rows(RowCount::from(6), BatchCount::from(1))
        .map(|batch| Ok(batch.unwrap()))
        .collect::<Result<Vec<_>>>()
        .unwrap();

    // Write with enable_stable_row_ids=true and force multiple fragments
    let write_params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_1),
        enable_stable_row_ids: true,
        max_rows_per_file: 3, // Force 2 fragments with 3 rows each
        ..Default::default()
    };

    let reader = RecordBatchIterator::new(data.clone().into_iter().map(Ok), data[0].schema());
    let dataset = Arc::new(
        Dataset::write(reader, &test_dir, Some(write_params))
            .await
            .unwrap(),
    );

    // Verify we have multiple fragments
    let fragments = dataset.fragments();
    assert!(
        fragments.len() >= 2,
        "Expected at least 2 fragments, got {}",
        fragments.len()
    );

    // Test first fragment (indices 0, 1, 2) - this always worked
    let blobs = dataset
        .take_blobs_by_indices(&[0, 1, 2], "blobs")
        .await
        .unwrap();
    assert_eq!(blobs.len(), 3, "First fragment blobs should have 3 items");

    // Verify we can read the blob content
    for blob in &blobs {
        let content = blob.as_ref().unwrap().read().await.unwrap();
        assert!(!content.is_empty(), "Blob content should not be empty");
    }

    // Test second fragment (indices 3, 4, 5) - this was failing before the fix
    let blobs = dataset
        .take_blobs_by_indices(&[3, 4, 5], "blobs")
        .await
        .unwrap();
    assert_eq!(blobs.len(), 3, "Second fragment blobs should have 3 items");

    // Verify we can read the blob content from second fragment
    for blob in &blobs {
        let content = blob.as_ref().unwrap().read().await.unwrap();
        assert!(!content.is_empty(), "Blob content should not be empty");
    }

    // Test mixed indices from both fragments
    let blobs = dataset
        .take_blobs_by_indices(&[1, 4], "blobs")
        .await
        .unwrap();
    assert_eq!(blobs.len(), 2, "Mixed fragment blobs should have 2 items");
}

/// Regression test: scanning the same blob column twice on a single
/// `Dataset` (and therefore a single cache scope) with different
/// `BlobHandling` values must not panic in the structural decoder.
///
/// Background: the page-data cache key previously keyed cached page data
/// only by `column_index`. A blob column has two valid decoder shapes — the
/// descriptor view (`Struct<position, size>`) used when scanning with
/// `BlobHandling::BlobsDescriptions`, and the bytes view (`LargeBinary`)
/// used when scanning with `BlobHandling::AllBinary`. Both views go through
/// the same `StructuralPrimitiveFieldScheduler` but instantiate different
/// page-level schedulers, which cache different concrete `CachedPageData`
/// types under the same column index. When the second view hit the cache
/// populated by the first, it downcast the wrong state type and panicked.
#[tokio::test]
async fn test_blob_cache_key_distinguishes_views() {
    use crate::dataset::WriteParams;
    use arrow_array::RecordBatchIterator;
    use arrow_array::{LargeBinaryArray, UInt64Array};
    use arrow_schema::{Field, Schema};
    use lance_arrow::BLOB_META_KEY;
    use lance_core::datatypes::BlobHandling;

    let test_dir = TempStrDir::default();
    let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("blobs", DataType::LargeBinary, true).with_metadata(blob_meta),
        Field::new("idx", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"foo".as_slice()),
                Some(b"bar".as_slice()),
                Some(b"baz".as_slice()),
            ])),
            Arc::new(UInt64Array::from(vec![0u64, 1, 2])),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Open once and reuse the same Dataset (single Session, single cache)
    // across both scans. Reopening would defeat the test.
    let dataset = Arc::new(Dataset::open(test_dir.as_str()).await.unwrap());

    // Pass 1: descriptor view (Struct<position, size>). Default for scan().
    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::BlobsDescriptions);
    let descriptors = scanner
        .project(&["blobs"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(descriptors.column(0).data_type().is_struct());

    let read_payload = || {
        let dataset = dataset.clone();
        async move {
            let mut scanner = dataset.scan();
            scanner.blob_handling(BlobHandling::AllBinary);
            scanner
                .project(&["blobs"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
        }
    };

    // Pass 2: bytes view (LargeBinary). Used by compact_files.
    // Without the fix this used to panic in BlobPageScheduler::load
    // when it downcast the cached BlobDescriptionPageScheduler state.
    let bytes = read_payload().await;
    assert_eq!(bytes.column(0).data_type(), &DataType::LargeBinary);
    let blobs = bytes.column(0).as_binary::<i64>();
    assert_eq!(blobs.value(0), b"foo");
    assert_eq!(blobs.value(1), b"bar");
    assert_eq!(blobs.value(2), b"baz");

    // Pass 3: back to descriptor view to verify both directions are safe.
    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::BlobsDescriptions);
    let descriptors = scanner
        .project(&["blobs"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(descriptors.column(0).data_type().is_struct());

    // Pass 4: repeat the bytes view to exercise the payload cache hit.
    let bytes = read_payload().await;
    let blobs = bytes.column(0).as_binary::<i64>();
    assert_eq!(blobs.value(0), b"foo");
    assert_eq!(blobs.value(1), b"bar");
    assert_eq!(blobs.value(2), b"baz");
}

#[tokio::test]
async fn test_v2_0_legacy_blob_descriptor_projection_and_reads() {
    let test_dir = TempStrDir::default();
    let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("blob", DataType::LargeBinary, true).with_metadata(blob_meta),
    ]));
    let payloads = [
        b"abc".as_slice(),
        b"defgh".as_slice(),
        b"ijklmnop".as_slice(),
    ];
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2])) as ArrayRef,
            Arc::new(LargeBinaryArray::from_iter_values(payloads)) as ArrayRef,
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    for blob_handling in [
        None,
        Some(BlobHandling::BlobsDescriptions),
        Some(BlobHandling::AllDescriptions),
    ] {
        let mut scanner = dataset.scan();
        if let Some(blob_handling) = blob_handling {
            scanner.blob_handling(blob_handling);
        }
        let descriptors = scanner
            .project(&["blob"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let descriptor = descriptors.column(0).as_struct();
        assert_eq!(descriptor.fields().len(), 2);
        assert_eq!(descriptor.fields()[0].name(), "position");
        assert_eq!(descriptor.fields()[1].name(), "size");
        let sizes = descriptor
            .column_by_name("size")
            .unwrap()
            .as_primitive::<UInt64Type>();
        assert_eq!(sizes.values(), &[3, 5, 8]);
    }

    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let bytes = scanner
        .project(&["blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let bytes = bytes.column(0).as_binary::<i64>();
    for (idx, expected) in payloads.iter().enumerate() {
        assert_eq!(bytes.value(idx), *expected);
    }

    let blob_files = dataset
        .take_blobs_by_indices(&[0, 1, 2], "blob")
        .await
        .unwrap();
    assert_eq!(blob_files.len(), payloads.len());
    for (blob_file, expected) in blob_files.iter().zip(payloads) {
        assert_eq!(
            blob_file.as_ref().unwrap().read().await.unwrap().as_ref(),
            expected
        );
    }

    let read_blobs = dataset
        .read_blobs("blob")
        .unwrap()
        .with_row_indices(vec![0, 1, 2])
        .execute()
        .await
        .unwrap();
    assert_eq!(read_blobs.len(), payloads.len());
    for (read_blob, expected) in read_blobs.iter().zip(payloads) {
        assert_eq!(read_blob.data.as_deref(), Some(expected));
    }
}

#[rstest]
#[case::v20("v2.0.lance")]
#[case::v21("v2.1.lance")]
#[tokio::test]
async fn test_legacy_blob_append_without_metadata(#[case] fixture: &str) {
    let test_dir =
        crate::utils::test::copy_test_data_to_tmp(&format!("v8.0.0/blobs/{fixture}")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("blob", DataType::LargeBinary, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![3])),
            Arc::new(LargeBinaryArray::from(vec![Some(b"appended".as_slice())])),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &test_dir.path_str(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let handles = dataset.take_blobs_by_indices(&[3], "blob").await.unwrap();
    assert_eq!(
        handles[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"appended"
    );
    assert!(!dataset.schema().field("blob").unwrap().is_blob_v2());
}

#[rstest]
#[case::v20_bytes("v2.0.lance", false)]
#[case::v20_struct("v2.0.lance", true)]
#[case::v21_bytes("v2.1.lance", false)]
#[case::v21_struct("v2.1.lance", true)]
#[tokio::test]
async fn test_mixed_blob_versions(#[case] fixture: &str, #[case] struct_input: bool) {
    let test_dir =
        crate::utils::test::copy_test_data_to_tmp(&format!("v8.0.0/blobs/{fixture}")).unwrap();
    let dir = test_dir.path_str();
    let dataset = Dataset::open(&dir).await.unwrap();
    let schema = Arc::new(Schema::from(dataset.schema()));
    let id_field = schema.field_with_name("id").unwrap().clone();
    let legacy_field = schema.field_with_name("blob").unwrap();
    let old_values = [Some(b"legacy bytes".as_slice()), None, Some(b"".as_slice())];
    let old_snapshot = dataset.version_id();
    let field_id = dataset.schema().field("blob").unwrap().id;
    let old_fragment = dataset.manifest.fragments[0].clone();
    let old_path = std::path::Path::new(dir.as_str())
        .join("data")
        .join(&old_fragment.files[0].path);
    let old_file_bytes = std::fs::read(&old_path).unwrap();

    let new_values = [
        Some(b"packed!!".as_slice()),
        Some(b"dedicated blob bytes".as_slice()),
        None,
        Some(b"".as_slice()),
    ];
    let batch = if struct_input {
        let mut builder = BlobArrayBuilder::new(new_values.len());
        for value in new_values {
            match value {
                Some(bytes) => builder.push_bytes(bytes).unwrap(),
                None => builder.push_null().unwrap(),
            }
        }
        let mut field = lance_core::datatypes::Field::try_from(legacy_field).unwrap();
        field.promote_blob_v2().unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![id_field, Field::from(&field)])),
            vec![
                Arc::new(UInt32Array::from(vec![3, 4, 5, 6])),
                builder.finish().unwrap(),
            ],
        )
        .unwrap()
    } else {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![3, 4, 5, 6])),
                Arc::new(LargeBinaryArray::from(new_values.to_vec())),
            ],
        )
        .unwrap()
    };
    Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        &dir,
        Some(WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut dataset = Dataset::open(&dir).await.unwrap();
    assert_eq!(dataset.schema().field("blob").unwrap().id, field_id);
    assert_eq!(dataset.manifest.fragments[0], old_fragment);
    assert_eq!(std::fs::read(&old_path).unwrap(), old_file_bytes);
    assert_eq!(dataset.manifest.fragments.len(), 2);
    dataset.validate().await.unwrap();
    // The table default remains the original version. A subsequent byte append
    // must preserve the promoted schema while writing another legacy file.
    let legacy_again = Some(b"legacy again".as_slice());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![7])),
            Arc::new(LargeBinaryArray::from(vec![legacy_again])),
        ],
    )
    .unwrap();
    dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert!(dataset.schema().field("blob").unwrap().is_blob_v2());
    assert_eq!(dataset.schema().field("blob").unwrap().id, field_id);
    assert_eq!(dataset.manifest.fragments.len(), 3);
    dataset.validate().await.unwrap();
    let expected = old_values
        .into_iter()
        .chain(new_values)
        .chain([legacy_again])
        .collect::<Vec<_>>();

    for compact in [false, true] {
        if compact {
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_eq!(dataset.manifest.fragments.len(), 1);
            dataset = Dataset::open(&dir).await.unwrap();
        }
        let dataset = Arc::new(dataset.clone());
        let descriptors = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(
            descriptors["blob"].as_struct().fields(),
            &*lance_core::datatypes::BLOB_V2_DESC_FIELDS
        );
        for fragment in dataset.get_fragments() {
            let batch = fragment.scan().try_into_batch().await.unwrap();
            assert_eq!(batch.schema(), descriptors.schema());
        }
        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let batch = scanner.try_into_batch().await.unwrap();
        assert_eq!(
            batch["blob"].as_binary::<i64>().iter().collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            batch["id"]
                .as_primitive::<arrow_array::types::UInt32Type>()
                .values(),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        let indices = [4, 0, 6, 1, 3, 2, 5, 7];
        let handles = dataset
            .take_blobs_by_indices(&indices, "blob")
            .await
            .unwrap();
        for (index, handle) in indices.into_iter().zip(handles) {
            match (expected[index as usize], handle) {
                (Some(bytes), Some(handle)) => {
                    assert_eq!(handle.read().await.unwrap().as_ref(), bytes)
                }
                (None, None) => {}
                _ => panic!("blob nullability changed at row {index}"),
            }
        }
        let ranges = dataset
            .read_blob_ranges("blob")
            .unwrap()
            .with_row_indices(vec![
                BlobRangeRequest::new(4, 1, 3),
                BlobRangeRequest::new(0, 1, 3),
            ])
            .execute()
            .await
            .unwrap();
        assert_eq!(ranges[0].data.as_deref(), Some(&b"edi"[..]));
        assert_eq!(ranges[1].data.as_deref(), Some(&b"ega"[..]));
    }
    let historical = Arc::new(dataset.checkout_version(old_snapshot).await.unwrap());
    assert!(!historical.schema().field("blob").unwrap().is_blob_v2());
    let handles = historical
        .take_blobs_by_indices(&[0, 1, 2], "blob")
        .await
        .unwrap();
    assert_eq!(
        handles[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"legacy bytes"
    );
    assert!(handles[1].is_none());
    assert_eq!(handles[2].as_ref().unwrap().size(), 0);
    assert_eq!(std::fs::read(&old_path).unwrap(), old_file_bytes);
}

#[tokio::test]
async fn test_write_and_take_blobs_with_blob_descriptor_array_builder() {
    let test_dir = TempStrDir::default();

    // Build a blob column with the new BlobArrayBuilder
    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(b"hello").unwrap();
    blob_builder.push_bytes(b"world").unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

    let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0, 1]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());

    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    let dataset = Arc::new(
        Dataset::write(reader, &test_dir, Some(params))
            .await
            .unwrap(),
    );

    let blobs = dataset
        .take_blobs_by_indices(&[0, 1], "blob")
        .await
        .unwrap();

    assert_eq!(blobs.len(), 2);
    let first = blobs[0].as_ref().unwrap().read().await.unwrap();
    let second = blobs[1].as_ref().unwrap().read().await.unwrap();
    assert_eq!(first.as_ref(), b"hello");
    assert_eq!(second.as_ref(), b"world");
}

#[tokio::test]
async fn test_write_prepared_blob_column_normalizes_schema() {
    let test_dir = TempStrDir::default();
    let mut prepared_writer = BlobDescriptorArrayBuilder::new("blob");
    prepared_writer
        .push_inline(Bytes::from_static(b"prepared-inline"))
        .unwrap();
    let (prepared_field, prepared_array) = prepared_writer.finish().unwrap().into_parts();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        prepared_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            prepared_array,
        ],
    )
    .unwrap();

    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let dataset_schema = Schema::from(dataset.schema());
    let blob_field = dataset_schema.field_with_name("blob").unwrap();
    let DataType::Struct(fields) = blob_field.data_type() else {
        panic!("expected blob field to be a struct");
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name(), "data");
    assert_eq!(fields[1].name(), "uri");

    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(blobs.len(), 1);
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"prepared-inline"
    );
}

#[tokio::test]
async fn test_data_replacement_uses_blob_descriptor_array_builder_prepared_packed_blob_column() {
    let test_dir = TempStrDir::default();
    let logical_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(b"initial").unwrap();
    let initial_batch = RecordBatch::try_new(
        logical_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            initial_builder.finish().unwrap(),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let file_id = Uuid::new_v4().to_string();
    let data_file_name = format!("{file_id}.lance");
    let data_file_path = dataset.data_dir().join(data_file_name.as_str());
    let blob_id = 1;
    let packed_path = blob_path(&dataset.data_dir(), &file_id, blob_id);
    let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
    let mut packed = PackedBlobWriter::try_new(
        dataset.object_store.as_ref().clone(),
        data_file_path.clone(),
        blob_id,
    )
    .await
    .unwrap();
    assert_eq!(packed.path(), &packed_path);
    packed.write_blob(b"prepared-packed").await.unwrap();
    blob_writer.extend(packed.finish().await.unwrap()).unwrap();
    let prepared_field = blob_writer.field().clone();
    let prepared_array = blob_writer.finish().unwrap().into_parts().1;
    let append_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        prepared_field,
    ]));
    let replacement_batch = RecordBatch::try_new(
        append_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
            prepared_array,
        ],
    )
    .unwrap();

    let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
    let mut file_writer = lance_file::versions::v2_2::create_writer(
        object_writer,
        crate::datatypes::Schema::try_from(append_schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    file_writer.write_batch(&replacement_batch).await.unwrap();
    file_writer.finish().await.unwrap();

    let data_file = dataset
        .create_data_file(&data_file_name, None)
        .await
        .unwrap();
    let transaction = Transaction {
        read_version: dataset.manifest.version,
        uuid: Uuid::new_v4().hyphenated().to_string(),
        operation: Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, data_file)],
        },
        tag: None,
        transaction_properties: None,
    };
    let dataset = Arc::new(
        CommitBuilder::new(dataset)
            .execute(transaction)
            .await
            .unwrap(),
    );

    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.read().await.unwrap().as_ref(), b"prepared-packed");
    assert_eq!(blob.kind(), BlobKind::Packed);
}

#[tokio::test]
async fn test_append_uses_blob_descriptor_array_builder_prepared_blob_column() {
    let test_dir = TempStrDir::default();
    let logical_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(b"initial").unwrap();
    let initial_batch = RecordBatch::try_new(
        logical_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
            initial_builder.finish().unwrap(),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
    blob_writer
        .push_inline(Bytes::from_static(b"append"))
        .unwrap();
    let prepared_column = blob_writer.finish().unwrap();
    let (prepared_field, prepared_array) = prepared_column.into_parts();
    let append_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        prepared_field,
    ]));
    let append_batch = RecordBatch::try_new(
        append_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
            prepared_array,
        ],
    )
    .unwrap();

    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(append_batch)], append_schema),
            dataset,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let ids = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let id_values = ids.column(0).as_primitive::<UInt32Type>();
    assert_eq!(id_values.values(), &[0, 1]);

    let blobs = dataset
        .take_blobs_by_indices(&[0, 1], "blob")
        .await
        .unwrap();
    assert_eq!(blobs.len(), 2);
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"initial"
    );
    assert_eq!(
        blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"append"
    );
}

#[tokio::test]
async fn test_data_replacement_uses_blob_descriptor_array_builder_prepared_nested_blob_column() {
    let test_dir = TempStrDir::default();
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(b"initial").unwrap();
    let (logical_schema, initial_batch) = nested_blob_v2_batch(initial_builder.finish().unwrap());
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let file_id = Uuid::new_v4().to_string();
    let data_file_name = format!("{file_id}.lance");
    let data_file_path = dataset.data_dir().join(data_file_name.as_str());
    let blob_id = 1;
    let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
    let mut packed = PackedBlobWriter::try_new(
        dataset.object_store.as_ref().clone(),
        data_file_path.clone(),
        blob_id,
    )
    .await
    .unwrap();
    packed.write_blob(b"nested-replacement").await.unwrap();
    blob_writer.extend(packed.finish().await.unwrap()).unwrap();
    let prepared_column = blob_writer.finish().unwrap();
    let (prepared_field, prepared_array) = prepared_column.into_parts();

    let info_fields = vec![Field::new("name", DataType::Utf8, false), prepared_field];
    let info_array = Arc::new(
        StructArray::try_new(
            info_fields.clone().into(),
            vec![
                Arc::new(StringArray::from(vec!["replacement"])) as ArrayRef,
                prepared_array,
            ],
            None,
        )
        .unwrap(),
    ) as ArrayRef;
    let replacement_schema = Arc::new(Schema::new(vec![Field::new(
        "info",
        DataType::Struct(info_fields.into()),
        true,
    )]));
    let replacement_batch =
        RecordBatch::try_new(replacement_schema.clone(), vec![info_array]).unwrap();

    let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
    let mut file_writer = lance_file::versions::v2_2::create_writer(
        object_writer,
        crate::datatypes::Schema::try_from(replacement_schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    file_writer.write_batch(&replacement_batch).await.unwrap();
    file_writer.finish().await.unwrap();

    let data_file = dataset
        .create_data_file(&data_file_name, None)
        .await
        .unwrap();
    let transaction = Transaction {
        read_version: dataset.manifest.version,
        uuid: Uuid::new_v4().hyphenated().to_string(),
        operation: Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(0, data_file)],
        },
        tag: None,
        transaction_properties: None,
    };
    let dataset = Arc::new(
        CommitBuilder::new(dataset)
            .execute(transaction)
            .await
            .unwrap(),
    );

    let blobs = dataset
        .take_blobs_by_indices(&[0], "info.blob")
        .await
        .unwrap();
    assert_eq!(blobs.len(), 1);
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"nested-replacement"
    );
    assert_eq!(blobs[0].as_ref().unwrap().kind(), BlobKind::Packed);
}

#[tokio::test]
async fn test_create_data_file_skips_blob_orphans_with_top_level_name_collision() {
    let test_dir = TempStrDir::default();
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(b"initial").unwrap();
    let (_, initial_info_batch) = nested_blob_v2_batch(initial_builder.finish().unwrap());
    let info_field = initial_info_batch.schema().field(0).as_ref().clone();
    let logical_schema = Arc::new(Schema::new(vec![
        info_field,
        Field::new("kind", DataType::UInt8, false),
    ]));
    let initial_batch = RecordBatch::try_new(
        logical_schema.clone(),
        vec![
            initial_info_batch.column(0).clone(),
            Arc::new(UInt8Array::from(vec![7])) as ArrayRef,
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial_batch)], logical_schema.clone()),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let file_id = Uuid::new_v4().to_string();
    let data_file_name = format!("{file_id}.lance");
    let data_file_path = dataset.data_dir().join(data_file_name.as_str());
    let mut blob_writer = BlobDescriptorArrayBuilder::new("blob");
    blob_writer
        .push_inline(Bytes::from_static(b"replacement"))
        .unwrap();
    let prepared_column = blob_writer.finish().unwrap();
    let (prepared_field, prepared_array) = prepared_column.into_parts();

    let info_fields = vec![Field::new("name", DataType::Utf8, false), prepared_field];
    let info_array = Arc::new(
        StructArray::try_new(
            info_fields.clone().into(),
            vec![
                Arc::new(StringArray::from(vec!["replacement"])) as ArrayRef,
                prepared_array,
            ],
            None,
        )
        .unwrap(),
    ) as ArrayRef;
    let replacement_schema = Arc::new(Schema::new(vec![Field::new(
        "info",
        DataType::Struct(info_fields.into()),
        true,
    )]));
    let replacement_batch =
        RecordBatch::try_new(replacement_schema.clone(), vec![info_array]).unwrap();

    let object_writer = dataset.object_store.create(&data_file_path).await.unwrap();
    let mut file_writer = lance_file::versions::v2_2::create_writer(
        object_writer,
        crate::datatypes::Schema::try_from(replacement_schema.as_ref()).unwrap(),
        FileWriterOptions::default(),
    )
    .unwrap();
    file_writer.write_batch(&replacement_batch).await.unwrap();
    file_writer.finish().await.unwrap();

    let data_file = dataset
        .create_data_file(&data_file_name, None)
        .await
        .unwrap();

    assert_eq!(data_file.fields.len(), 2);
    assert_eq!(data_file.column_indices.as_ref(), &[0, 1]);
}

#[rstest]
#[tokio::test]
async fn test_write_and_scan_struct_nested_list_blob_v2(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();

    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(b"nested").unwrap();
    blob_builder.push_null().unwrap();
    let blob_values = blob_builder.finish().unwrap();

    let item_field = Arc::new(blob_field("item", true));
    let list_field = Field::new("blobs", DataType::List(item_field.clone()), true);
    let list_array: ArrayRef = Arc::new(
        arrow_array::ListArray::try_new(
            item_field,
            arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 2, 2])),
            blob_values,
            None,
        )
        .unwrap(),
    );
    let info_fields = vec![Field::new("name", DataType::Utf8, false), list_field];
    let info_array: ArrayRef = Arc::new(
        StructArray::try_new(
            info_fields.clone().into(),
            vec![
                Arc::new(StringArray::from(vec!["row-0", "row-1"])) as ArrayRef,
                list_array,
            ],
            None,
        )
        .unwrap(),
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "info",
        DataType::Struct(info_fields.into()),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![info_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let descriptions = dataset
        .scan()
        .project(&["info"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let info = descriptions.column(0).as_struct();
    assert_eq!(
        info.column_by_name("name")
            .unwrap()
            .as_string::<i32>()
            .value(0),
        "row-0"
    );
    let lists = info.column_by_name("blobs").unwrap().as_list::<i32>();
    assert_eq!(lists.offsets().inner().as_ref(), &[0, 2, 2]);
    let DataType::List(descriptor_field) = lists.data_type() else {
        panic!("unexpected nested list type: {}", lists.data_type());
    };
    assert!(matches!(descriptor_field.data_type(), DataType::Struct(_)));
    assert!(!descriptor_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
    let descriptors = lists.values().as_struct();
    assert_eq!(descriptors.fields().len(), 5);
    assert!(descriptors.is_valid(0));
    assert!(descriptors.is_null(1));

    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let bytes = scanner
        .project(&["info"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let info = bytes.column(0).as_struct();
    let lists = info.column_by_name("blobs").unwrap().as_list::<i32>();
    assert_eq!(lists.offsets().inner().as_ref(), &[0, 2, 2]);
    let DataType::List(value_field) = lists.data_type() else {
        panic!("unexpected nested list type: {}", lists.data_type());
    };
    assert_eq!(value_field.data_type(), &DataType::LargeBinary);
    assert!(!value_field.metadata().contains_key(ARROW_EXT_NAME_KEY));
    let values = lists.values().as_binary::<i64>();
    assert_eq!(values.value(0), b"nested");
    assert!(values.is_null(1));
}

#[tokio::test]
async fn test_nested_blob_v2_requires_v2_2() {
    let test_dir = TempStrDir::default();

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_bytes(b"hello").unwrap();
    let blob_array: ArrayRef = blob_builder.finish().unwrap();

    let (schema, batch) = nested_blob_v2_batch(blob_array);
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let result = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await;

    assert!(
        result.is_err(),
        "Nested blob v2 should be rejected for file version 2.1"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Blob v2 requires file version >= 2.2")
    );
}

#[tokio::test]
async fn test_read_blob_ranges_preserves_request_order_across_fragments() {
    let fixture = BlobTestFixture::new().await;
    let selections = [
        (22_u64, 2_usize, 2_usize),
        (2, 0, 2),
        (12, 1, 2),
        (12, 1, 2),
    ];
    let ranges = vec![
        BlobReadRange::new(2, 3),
        BlobReadRange::new(0, 2),
        BlobReadRange::new(3, 2),
        BlobReadRange::new(0, 3),
    ];

    let results = fixture
        .dataset
        .read_blob_ranges("blobs")
        .unwrap()
        .with_row_indices(
            selections
                .iter()
                .zip(&ranges)
                .map(|((row_index, _, _), range)| {
                    BlobRangeRequest::new(*row_index, range.offset, range.length)
                })
                .collect::<Vec<_>>(),
        )
        .execute()
        .await
        .unwrap();

    assert_eq!(results.len(), selections.len());
    for (request_index, (((_, batch_index, row_index), range), result)) in
        selections.iter().zip(&ranges).zip(&results).enumerate()
    {
        let payload = fixture.data[*batch_index]
            .column(1)
            .as_binary::<i64>()
            .value(*row_index);
        let start = range.offset as usize;
        let end = start + range.length as usize;
        assert_eq!(result.request_index, request_index);
        assert_eq!(result.range, *range);
        assert_eq!(result.data.as_deref(), Some(&payload[start..end]));
    }
}

#[tokio::test]
async fn test_blob_selection_apis_preserve_v2_nulls_and_empty_values() {
    let test_dir = TempStrDir::default();
    let mut blob_builder = BlobArrayBuilder::new(3);
    blob_builder.push_empty().unwrap();
    blob_builder.push_null().unwrap();
    blob_builder.push_bytes(b"abc").unwrap();
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_builder.finish().unwrap()]).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let blob_files = dataset
        .take_blobs_by_indices(&[2, 1, 0], "blob")
        .await
        .unwrap();
    assert_eq!(blob_files.len(), 3);
    assert_eq!(
        blob_files[0]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .as_ref(),
        b"abc"
    );
    assert!(blob_files[1].is_none());
    assert!(
        blob_files[2]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .is_empty()
    );

    let blobs = dataset
        .read_blobs("blob")
        .unwrap()
        .with_row_indices([2, 1, 0])
        .execute()
        .await
        .unwrap();
    assert_eq!(
        blobs
            .iter()
            .map(|blob| (blob.row_address, blob.data.as_deref()))
            .collect::<Vec<_>>(),
        vec![
            (2, Some(b"abc".as_slice())),
            (1, None),
            (0, Some(b"".as_slice())),
        ]
    );

    let results = dataset
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_indices([
            BlobRangeRequest::new(0, 0, 0),
            BlobRangeRequest::new(1, 0, 0),
            BlobRangeRequest::new(1, 0, 1),
            BlobRangeRequest::new(2, 1, 1),
        ])
        .execute()
        .await
        .unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].request_index, 0);
    assert_eq!(results[0].data.as_deref(), Some(b"".as_slice()));
    assert_eq!(results[1].request_index, 1);
    assert!(results[1].data.is_none());
    assert_eq!(results[2].request_index, 2);
    assert!(results[2].data.is_none());
    assert_eq!(results[3].request_index, 3);
    assert_eq!(results[3].data.as_deref(), Some(b"b".as_slice()));
}

#[tokio::test]
async fn test_read_blob_ranges_rejects_overflow_and_out_of_bounds() {
    let fixture = BlobTestFixture::new().await;

    let err = fixture
        .dataset
        .read_blob_ranges("blobs")
        .unwrap()
        .execute()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains("requires requests"));

    let err = fixture
        .dataset
        .read_blob_ranges("blobs")
        .unwrap()
        .with_row_indices([BlobRangeRequest::new(0, u64::MAX, 1)])
        .execute()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains("offset + length overflowed"));

    let err = fixture
        .dataset
        .read_blob_ranges("blobs")
        .unwrap()
        .with_row_indices([BlobRangeRequest::new(0, 0, u64::MAX)])
        .execute()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains("exceeds blob size"));
    assert!(err.to_string().contains("request 0"));
}

#[tokio::test]
async fn test_take_blob_v2_from_non_default_base_inline() {
    let fixture = create_multi_base_blob_v2_fixture(b"inline".to_vec(), None, true).await;

    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&[0], "blob")
        .await
        .unwrap();

    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Inline);
    assert_eq!(
        blob.read().await.unwrap().as_ref(),
        fixture.expected.as_slice()
    );
}

#[tokio::test]
async fn test_take_blob_v2_from_non_default_base_dedicated() {
    let fixture = create_multi_base_blob_v2_fixture(vec![0xA5; 4096], Some(1), true).await;

    let blobs = fixture
        .dataset
        .take_blobs_by_indices(&[0], "blob")
        .await
        .unwrap();

    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Dedicated);
    assert_eq!(
        blob.read().await.unwrap().as_ref(),
        fixture.expected.as_slice()
    );
}

#[tokio::test]
async fn test_blob_v2_external_outside_base_denied_by_default() {
    let dataset_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    std::fs::write(&external_path, b"outside").unwrap();
    let external_uri = format!("file://{}", external_path.display());

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_uri(external_uri).unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let result = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await;

    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("outside registered external bases"),
        "{err:?}"
    );
}

#[tokio::test]
async fn test_blob_v2_external_under_dataset_root_denied_by_default() {
    let test_dir = TempDir::default();
    let dataset_path = test_dir.std_path().join("dataset");
    std::fs::create_dir_all(dataset_path.join("media")).unwrap();
    let external_path = dataset_path.join("media").join("external.bin");
    std::fs::write(&external_path, b"root-local").unwrap();
    let external_uri = format!("file://{}", external_path.display());

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_uri(external_uri).unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let result = Dataset::write(
        reader,
        dataset_path.to_str().unwrap(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await;

    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("outside registered external bases"),
        "{err:?}"
    );
}

#[rstest]
#[case::whole_object(None, b"mapped")]
#[case::slice(Some(BlobRange { offset: 1, size: 4 }), b"appe")]
#[tokio::test]
async fn test_blob_v2_external_mapped_to_registered_base(
    #[case] range: Option<BlobRange>,
    #[case] expected: &[u8],
) {
    let test_dir = TempDir::default();
    let dataset_uri = test_dir.std_path().join("dataset");
    let external_base = test_dir.std_path().join("external_base");
    let external_obj_dir = external_base.join("objects");
    std::fs::create_dir_all(&external_obj_dir).unwrap();
    let external_path = external_obj_dir.join("mapped.bin");
    std::fs::write(&external_path, b"mapped").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let base_uri = format!("file://{}", external_base.display());

    let blob_array = complete_blob_v2_array(
        vec![None],
        vec![Some(external_uri)],
        vec![range.map(|r| r.offset)],
        vec![range.map(|r| r.size)],
        None,
    );
    let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            dataset_uri.to_str().unwrap(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("external".to_string()),
                    path: base_uri,
                    is_dataset_root: false,
                }]),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let desc = dataset
        .scan()
        .project(&["blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
        .column(0)
        .as_struct()
        .to_owned();
    assert_eq!(
        desc.column(0).as_primitive::<UInt8Type>().value(0),
        BlobKind::External as u8
    );
    assert_eq!(desc.column(3).as_primitive::<UInt32Type>().value(0), 1);
    assert_eq!(
        desc.column(4).as_string::<i32>().value(0),
        "objects/mapped.bin"
    );

    let external_store = dataset.object_store(Some(1)).await.unwrap();
    let _ = external_store.io_stats_incremental();
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(blobs.len(), 1);
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        expected
    );
    let results = dataset
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_indices([BlobRangeRequest::new(0, 1, 2)])
        .execute()
        .await
        .unwrap();
    assert_eq!(results[0].data.as_deref(), Some(&expected[1..3]));
    let stats = external_store.io_stats_incremental();
    let size_queries = stats
        .requests
        .iter()
        .filter(|request| request.method == "head")
        .count();
    // A descriptor without a length still needs one size query per selection.
    let expected_size_queries = if desc.column(2).as_primitive::<UInt64Type>().value(0) == 0 {
        2
    } else {
        0
    };
    assert_eq!(size_queries, expected_size_queries, "{stats:?}");
}

#[tokio::test]
async fn test_blob_v2_external_ingest_inline_slice() {
    let dataset_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    std::fs::write(&external_path, b"prefix-inline-suffix").unwrap();
    let external_uri = format!("file://{}", external_path.display());

    let metadata = [(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]
        .into_iter()
        .collect();
    let blob_field = Field::new(
        "blob",
        DataType::Struct(
            vec![
                Field::new("data", DataType::LargeBinary, true),
                Field::new("uri", DataType::Utf8, true),
                Field::new("position", DataType::UInt64, true),
                Field::new("size", DataType::UInt64, true),
            ]
            .into(),
        ),
        true,
    )
    .with_metadata(metadata);
    let blob_array: ArrayRef = Arc::new(
        StructArray::try_new(
            vec![
                Field::new("data", DataType::LargeBinary, true),
                Field::new("uri", DataType::Utf8, true),
                Field::new("position", DataType::UInt64, true),
                Field::new("size", DataType::UInt64, true),
            ]
            .into(),
            vec![
                Arc::new(arrow_array::LargeBinaryArray::from(vec![None::<&[u8]>])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some(external_uri.as_str())])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![Some(7)])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![Some(6)])) as ArrayRef,
            ],
            None,
        )
        .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![blob_field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                external_blob_mode: ExternalBlobMode::Ingest,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let desc = dataset
        .scan()
        .project(&["blob"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
        .column(0)
        .as_struct()
        .to_owned();
    assert_eq!(
        desc.column_by_name("kind")
            .unwrap()
            .as_primitive::<UInt8Type>()
            .value(0),
        BlobKind::Inline as u8
    );

    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Inline);
    assert_eq!(blob.read().await.unwrap().as_ref(), b"inline");
}

#[tokio::test]
async fn test_complete_blob_v2_schema_survives_create() {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![complete_blob_v2_array(
            vec![Some(b"created".to_vec())],
            vec![None],
            vec![None],
            vec![None],
            None,
        )],
    )
    .unwrap();

    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let dataset_schema = Schema::from(dataset.schema());
    let DataType::Struct(fields) = dataset_schema.field_with_name("blob").unwrap().data_type()
    else {
        panic!("expected complete logical blob struct after create");
    };
    assert_eq!(fields.as_ref(), BLOB_V2_LOGICAL_FIELDS.as_ref());

    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"created"
    );
}

#[tokio::test]
async fn test_complete_blob_v2_schema_survives_append() {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
    let initial_batch = RecordBatch::try_new(
        schema.clone(),
        vec![complete_blob_v2_array(
            vec![Some(b"initial".to_vec())],
            vec![None],
            vec![None],
            vec![None],
            None,
        )],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_batch)], schema.clone()),
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let append_batch = RecordBatch::try_new(
        schema.clone(),
        vec![complete_blob_v2_array(
            vec![Some(b"appended".to_vec())],
            vec![None],
            vec![None],
            vec![None],
            None,
        )],
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(append_batch)], schema),
            None,
        )
        .await
        .unwrap();

    let dataset = Arc::new(dataset);
    let dataset_schema = Schema::from(dataset.schema());
    let DataType::Struct(fields) = dataset_schema.field_with_name("blob").unwrap().data_type()
    else {
        panic!("expected complete logical blob struct after append");
    };
    assert_eq!(fields.as_ref(), BLOB_V2_LOGICAL_FIELDS.as_ref());

    let blobs = dataset
        .take_blobs_by_indices(&[0, 1], "blob")
        .await
        .unwrap();
    assert_eq!(
        blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"initial"
    );
    assert_eq!(
        blobs[1].as_ref().unwrap().read().await.unwrap().as_ref(),
        b"appended"
    );
}

#[rstest]
#[case::reference_missing_size(
    ExternalBlobMode::Reference,
    Some("file:///source.bin"),
    Some(3),
    None,
    "both `position` and `size`"
)]
#[case::reference_missing_position(
    ExternalBlobMode::Reference,
    Some("file:///source.bin"),
    None,
    Some(2),
    "both `position` and `size`"
)]
#[case::reference_range_without_uri(
    ExternalBlobMode::Reference,
    None,
    Some(3),
    Some(2),
    "`uri` is null"
)]
#[case::ingest_missing_size(
    ExternalBlobMode::Ingest,
    Some("file:///source.bin"),
    Some(3),
    None,
    "both `position` and `size`"
)]
#[case::ingest_missing_position(
    ExternalBlobMode::Ingest,
    Some("file:///source.bin"),
    None,
    Some(2),
    "both `position` and `size`"
)]
#[case::ingest_range_without_uri(ExternalBlobMode::Ingest, None, Some(3), Some(2), "`uri` is null")]
#[tokio::test]
async fn test_complete_blob_v2_rejects_invalid_ranges(
    #[case] external_blob_mode: ExternalBlobMode,
    #[case] uri: Option<&str>,
    #[case] position: Option<u64>,
    #[case] size: Option<u64>,
    #[case] expected_message: &str,
) {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![complete_blob_v2_array(
            vec![None],
            vec![uri.map(str::to_string)],
            vec![position],
            vec![size],
            None,
        )],
    )
    .unwrap();

    let error = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            allow_external_blob_outside_bases: matches!(
                external_blob_mode,
                ExternalBlobMode::Reference
            ),
            external_blob_mode,
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();

    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(expected_message));
}

#[rstest]
#[case::reference(ExternalBlobMode::Reference)]
#[case::ingest(ExternalBlobMode::Ingest)]
#[tokio::test]
async fn test_complete_blob_v2_rejects_zero_size_range(
    #[case] external_blob_mode: ExternalBlobMode,
) {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![complete_blob_v2_field("blob", true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![complete_blob_v2_array(
            vec![None],
            vec![Some("file:///source.bin".to_string())],
            vec![Some(3)],
            vec![Some(0)],
            None,
        )],
    )
    .unwrap();

    let error = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            allow_external_blob_outside_bases: matches!(
                external_blob_mode,
                ExternalBlobMode::Reference
            ),
            external_blob_mode,
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();

    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("greater than zero"));
}

#[tokio::test]
async fn test_blob_v2_external_ingest_respects_inline_threshold() {
    let dataset_dir = TempDir::default();
    let external_dir = TempDir::default();
    let external_path = external_dir.std_path().join("external.bin");
    let payload = vec![0x5A; 2048];
    std::fs::write(&external_path, &payload).unwrap();
    let external_uri = format!("file://{}", external_path.display());

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_uri(external_uri).unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

    let mut field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        "1024".to_string(),
    );
    field = field.with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            &dataset_dir.path_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                external_blob_mode: ExternalBlobMode::Ingest,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(blobs.len(), 1);
    let blob = blobs[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Packed);
    assert_eq!(blob.read().await.unwrap().as_ref(), payload.as_slice());
}

#[tokio::test]
async fn test_blob_v2_requires_v2_2() {
    let test_dir = TempStrDir::default();

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_bytes(b"hello").unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

    let id_array: arrow_array::ArrayRef = Arc::new(UInt32Array::from(vec![0]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, blob_array]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let result = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await;

    assert!(
        result.is_err(),
        "Blob v2 should be rejected for file version 2.1"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Blob v2 requires file version >= 2.2")
    );
}

#[tokio::test]
async fn test_blob_v2_write_rejects_invalid_inline_threshold_metadata() {
    let dataset_dir = TempDir::default();
    let mut field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        "not-a-number".to_string(),
    );
    field = field.with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));

    let mut blob_builder = BlobArrayBuilder::new(1);
    blob_builder.push_bytes(vec![0u8; 256]).unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(blob_builder.finish().unwrap()) as ArrayRef],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let result = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await;
    let Err(err) = result else {
        panic!("write with invalid blob threshold metadata should fail");
    };
    assert!(
        err.to_string()
            .contains("expected a non-negative integer that fits in usize")
    );
}

#[rstest]
#[case::negative_inline(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "-5")]
#[case::invalid_dedicated(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "abc")]
#[case::zero_dedicated(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "0")]
#[case::zero_pack_file(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, "0")]
#[tokio::test]
async fn test_blob_v2_empty_create_rejects_invalid_threshold_metadata(
    #[case] key: &str,
    #[case] value: &str,
) {
    let dataset_dir = TempDir::default();
    let mut field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.insert(key.to_string(), value.to_string());
    field = field.with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema);

    let err = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
    assert!(err.to_string().contains(&format!("{key}=\"{value}\"")));
    assert!(Dataset::open(&dataset_dir.path_str()).await.is_err());
}

#[rstest]
#[case::negative_inline(BLOB_INLINE_SIZE_THRESHOLD_META_KEY, "-5")]
#[case::invalid_dedicated(BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, "abc")]
#[case::zero_pack_file(BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY, "0")]
#[tokio::test]
async fn test_blob_v2_field_metadata_rejects_invalid_threshold(
    #[case] key: &str,
    #[case] value: &str,
) {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema);
    let mut dataset = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let version = dataset.version_id();

    let err = dataset
        .update_field_metadata()
        .update("blob", [(key, value)])
        .unwrap()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
    assert!(err.to_string().contains(&format!("{key}=\"{value}\"")));
    assert_eq!(dataset.version_id(), version);
    let reopened = Dataset::open(&dataset_dir.path_str()).await.unwrap();
    assert_eq!(reopened.version_id(), version);
    assert!(
        !reopened
            .schema()
            .field("blob")
            .unwrap()
            .metadata
            .contains_key(key)
    );
}

#[tokio::test]
async fn test_blob_v2_overwrite_rejects_invalid_threshold_metadata() {
    let dataset_dir = TempDir::default();
    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema);
    let dataset = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        "-5".to_string(),
    );
    field = field.with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema);
    let err = Dataset::write(
        reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
    assert!(
        err.to_string()
            .contains(BLOB_INLINE_SIZE_THRESHOLD_META_KEY)
    );
    assert_eq!(
        Dataset::open(&dataset_dir.path_str())
            .await
            .unwrap()
            .version_id(),
        dataset.version_id()
    );
}

#[tokio::test]
async fn test_blob_v2_append_rejects_explicit_inline_threshold_mismatch() {
    let dataset_dir = TempDir::default();
    let payload = vec![0u8; 2048];

    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(payload.clone()).unwrap();
    let initial_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
    )
    .unwrap();
    let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
    let dataset = Dataset::write(
        initial_reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut append_field = blob_field("blob", true);
    let mut append_metadata = append_field.metadata().clone();
    append_metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        "1024".to_string(),
    );
    append_field = append_field.with_metadata(append_metadata);
    let append_schema = Arc::new(Schema::new(vec![append_field]));
    let mut append_builder = BlobArrayBuilder::new(1);
    append_builder.push_bytes(payload).unwrap();
    let append_batch = RecordBatch::try_new(
        append_schema.clone(),
        vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
    )
    .unwrap();
    let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

    let result = Dataset::write(
        append_reader,
        Arc::new(dataset),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await;
    let Err(err) = result else {
        panic!("append with explicit blob threshold mismatch should fail");
    };
    let message = err.to_string();
    assert!(message.contains("Cannot append data with blob threshold metadata"));
    assert!(message.contains(BLOB_INLINE_SIZE_THRESHOLD_META_KEY));
}

#[tokio::test]
async fn test_blob_v2_append_rejects_threshold_mismatch_with_non_blob_input_extension() {
    let dataset_dir = TempDir::default();
    let payload = vec![0u8; 2048];

    let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
    let mut initial_builder = BlobArrayBuilder::new(1);
    initial_builder.push_bytes(payload.clone()).unwrap();
    let initial_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(initial_builder.finish().unwrap()) as ArrayRef],
    )
    .unwrap();
    let initial_reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema);
    let dataset = Dataset::write(
        initial_reader,
        &dataset_dir.path_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let mut append_field = blob_field("blob", true);
    let mut append_metadata = append_field.metadata().clone();
    append_metadata.insert(
        ARROW_EXT_NAME_KEY.to_string(),
        "some.other.extension".to_string(),
    );
    append_metadata.insert(
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
        "1024".to_string(),
    );
    append_field = append_field.with_metadata(append_metadata);
    let append_schema = Arc::new(Schema::new(vec![append_field]));
    let mut append_builder = BlobArrayBuilder::new(1);
    append_builder.push_bytes(payload).unwrap();
    let append_batch = RecordBatch::try_new(
        append_schema.clone(),
        vec![Arc::new(append_builder.finish().unwrap()) as ArrayRef],
    )
    .unwrap();
    let append_reader = RecordBatchIterator::new(vec![Ok(append_batch)], append_schema);

    let result = Dataset::write(
        append_reader,
        Arc::new(dataset),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await;
    let Err(err) = result else {
        panic!("append with ignored blob threshold metadata should fail");
    };
    let message = err.to_string();
    assert!(message.contains("Cannot append data with blob threshold metadata"));
    assert!(message.contains(BLOB_INLINE_SIZE_THRESHOLD_META_KEY));
}
