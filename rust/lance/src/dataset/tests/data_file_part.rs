// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, fs, ops::Range, sync::Arc, time::SystemTime};

use arrow::array::AsArray;
use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, LargeListArray, ListArray, RecordBatch, RecordBatchIterator,
    StringArray, StructArray, UInt64Array, new_null_array, types::Int32Type,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use futures::{TryStreamExt, stream};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use lance_core::{
    Error,
    datatypes::{BLOB_V2_LOGICAL_FIELDS, BlobHandling, BlobKind},
    utils::{
        blob::blob_path,
        tempfile::{TempDir, TempStrDir},
    },
};
use lance_file::concat::EncodedFileInput;
use lance_file::reader::FileReaderOptions;
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::feature_flags::MANAGED_BLOBS_META_KEY;
use lance_table::format::BasePath;
use mock_instant::thread_local::MockClock;
use object_store::path::Path;
use rstest::rstest;
use sha2::{Digest, Sha256};

use crate::blob::{
    BlobArrayBuilder, BlobDescriptorArrayBuilder, BlobFile, BlobWriteSources, blob_field,
};
use crate::dataset::blob::test_util::managed_blob_schema;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::write::{WriteMode, WriteParams};
use crate::dataset::{DataFilePart, DataFileTarget, WriteDestination};
use crate::{Dataset, Result};

async fn dataset_of(batch: RecordBatch, version: LanceFileVersion) -> Dataset {
    let schema = batch.schema();
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn complete_logical_blob_batch(uri: &str, position: u64, size: u64) -> RecordBatch {
    let field = Field::new(
        "blob",
        DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()),
        true,
    )
    .with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]));
    let array = StructArray::try_new(
        BLOB_V2_LOGICAL_FIELDS.clone(),
        vec![
            Arc::new(LargeBinaryArray::from(vec![None::<&[u8]>])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some(uri)])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![Some(position)])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![Some(size)])) as ArrayRef,
        ],
        None,
    )
    .unwrap();
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![field])),
        vec![Arc::new(array)],
    )
    .unwrap()
}

fn only_fragment(dataset: &Dataset) -> FileFragment {
    dataset.get_fragments().into_iter().next().unwrap()
}

fn part_file_name(part: &DataFilePart) -> String {
    serde_json::to_value(part).unwrap()["file_name"]
        .as_str()
        .unwrap()
        .to_owned()
}

async fn write_part(
    dataset: &Dataset,
    target: &DataFileTarget,
    blob_ids: Option<Range<u32>>,
    batch: RecordBatch,
) -> DataFilePart {
    dataset
        .write_data_file_part(target, blob_ids, stream::iter([Ok(batch)]))
        .await
        .unwrap()
}

async fn commit(dataset: &Dataset, replacement: DataReplacementGroup) -> Result<Dataset> {
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Operation::DataReplacement {
            replacements: vec![replacement],
            data_change: true,
        },
        Some(dataset.version_id()),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
}

#[tokio::test]
async fn concatenates_parts_in_caller_order_without_reusing_staging_files() {
    let original = arrow_array::record_batch!(("id", Int32, [0, 1, 2, 3])).unwrap();
    let dataset = dataset_of(original, LanceFileVersion::V2_1).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let first = write_part(
        &dataset,
        &target,
        None,
        arrow_array::record_batch!(("id", Int32, [10, 11])).unwrap(),
    )
    .await;
    let second = write_part(
        &dataset,
        &target,
        None,
        arrow_array::record_batch!(("id", Int32, [12, 13])).unwrap(),
    )
    .await;

    let ordered_parts = [second, first];
    // The caller checks fragment coverage before assembling its replacement.
    assert_eq!(
        ordered_parts
            .iter()
            .map(DataFilePart::num_rows)
            .sum::<u64>(),
        only_fragment(&dataset).physical_rows().await.unwrap() as u64,
    );
    let assembly_dataset = dataset.clone();
    let assembly_target = target.clone();
    let data_file = tokio::spawn(async move {
        assembly_dataset
            .concat_data_file_parts(&assembly_target, &ordered_parts)
            .await
    })
    .await
    .unwrap()
    .unwrap();
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    assert_eq!(replacement.1.path, target.0.file_name());
    let dataset = commit(&dataset, replacement).await.unwrap();
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values(),
        &[12, 13, 10, 11]
    );
}

#[tokio::test]
async fn ordinary_blob_parts_survive_retained_version_cleanup() {
    let dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        vec![blob_field("blob", false)],
        HashMap::from([(MANAGED_BLOBS_META_KEY.to_string(), "false".to_string())]),
    ));
    let make_batch = |payloads: &[Vec<u8>]| {
        let mut blobs = BlobArrayBuilder::new(payloads.len());
        for payload in payloads {
            blobs.push_bytes(payload).unwrap();
        }
        RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap()
    };
    let payloads: Vec<Vec<u8>> = [100 * 1024, 100 * 1024, 128 * 1024, 137 * 1024]
        .into_iter()
        .enumerate()
        .map(|(row, size)| {
            (0..size)
                .map(|offset| ((offset * 17 + offset / 251 + row * 37) % 256) as u8)
                .collect()
        })
        .collect();
    let expected = payloads
        .iter()
        .map(|bytes| (bytes.len(), Sha256::digest(bytes)))
        .collect::<Vec<_>>();

    // The original version has its own ordinary sidecar, which cleanup can reclaim.
    let original = make_batch(&[vec![0; 100 * 1024], vec![], vec![], vec![]]);
    let params = WriteParams {
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(original)], schema.clone()),
        dir.as_str(),
        Some(params.clone()),
    )
    .await
    .unwrap();
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let parts = [
        write_part(&dataset, &target, Some(1..10), make_batch(&payloads[..2])).await,
        write_part(&dataset, &target, Some(10..20), make_batch(&payloads[2..])).await,
    ];
    let data_file = dataset
        .concat_data_file_parts(&target, &parts)
        .await
        .unwrap();
    // The local store reports real mtimes, which cleanup compares with manifest times.
    MockClock::set_system_time(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap(),
    );
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    let published = commit(&dataset, replacement).await.unwrap();
    let retained_version = published.version_id();
    target.finish(&published).await.unwrap();

    // Only the retained manifest references the assembled file after the overwrite.
    let latest = Dataset::write(
        RecordBatchIterator::new([Ok(make_batch(&[vec![255]]))], schema.clone()),
        dir.as_str(),
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            ..params
        }),
    )
    .await
    .unwrap();
    assert_eq!(latest.version_id(), retained_version + 1);
    assert_ne!(
        only_fragment(&latest).metadata().files[0].path,
        target.0.file_name()
    );
    let policy = CleanupPolicyBuilder::default()
        .retain_n_versions(&latest, 2)
        .await
        .unwrap()
        .build();
    let removed = latest.cleanup(policy).execute().await.unwrap();
    assert_eq!(removed.old_versions, 1);
    assert_eq!(removed.data_files_removed, 2); // Original data file and its sidecar.
    assert_eq!(removed.failed_deletes, 0);

    let retained = Arc::new(
        DatasetBuilder::from_uri(dir.as_str())
            .with_version(retained_version)
            .load()
            .await
            .unwrap(),
    );
    let blobs = retained
        .take_blobs_by_indices(&[0, 1, 2, 3], "blob")
        .await
        .unwrap();
    assert_eq!(blobs.len(), expected.len());
    let sidecar_dir = retained.data_dir().join(target.data_file_key());
    for (row, (blob, (length, digest))) in blobs.into_iter().zip(expected).enumerate() {
        let blob = blob.unwrap();
        assert!(matches!(
            blob.kind(),
            BlobKind::Packed | BlobKind::Dedicated
        ));
        assert!(blob.data_path().prefix_matches(&sidecar_dir));
        let bytes = blob.read().await.unwrap();
        assert_eq!(bytes.len(), length, "row {row}");
        assert_eq!(Sha256::digest(&bytes), digest, "row {row}");
    }
}

#[rstest]
#[case::rows("num_rows", serde_json::json!(3), "physical rows")]
#[case::size("size_bytes", serde_json::json!(1), "bytes")]
#[case::missing("file_name", serde_json::json!("missing.part"), "missing.part")]
#[tokio::test]
async fn managed_part_checks_real_file_before_creating_output(
    #[case] field: &str,
    #[case] value: serde_json::Value,
    #[case] message: &str,
) {
    let batch = arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap();
    let dataset = dataset_of(batch.clone(), LanceFileVersion::V2_2).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let part = write_part(&dataset, &target, None, batch.clone()).await;
    let mut description = serde_json::to_value(&part).unwrap();
    description[field] = value;
    let invalid = serde_json::from_value::<DataFilePart>(description).unwrap();
    let valid = write_part(&dataset, &target, None, batch).await;
    let error = dataset
        .concat_data_file_parts(&target, &[valid, invalid])
        .await
        .unwrap_err();
    if field == "file_name" {
        assert!(
            matches!(error.find_root(), Error::NotFound { .. }),
            "{error}"
        );
    } else {
        assert!(
            matches!(error.find_root(), Error::InvalidInput { .. }),
            "{error}"
        );
    }
    assert!(error.to_string().contains(message), "{error}");
    assert!(
        !dataset
            .table_state
            .object_store
            .exists(&dataset.data_dir().join(target.0.file_name()))
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn managed_part_rejects_invalid_descriptions_and_duplicate_inputs() {
    let batch = arrow_array::record_batch!(("id", Int32, [1])).unwrap();
    let dataset = dataset_of(batch.clone(), LanceFileVersion::V2_2).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let part = write_part(&dataset, &target, None, batch).await;
    let description = serde_json::to_value(&part).unwrap();
    let mut zero_size = description.clone();
    zero_size["size_bytes"] = serde_json::json!(0);
    let error = serde_json::from_value::<DataFilePart>(zero_size).unwrap_err();
    assert!(error.to_string().contains("nonzero"), "{error}");
    for (key, replacement, message) in [
        (
            "file_name",
            serde_json::json!("../outside.part"),
            "part identity",
        ),
        ("file_name", serde_json::json!(""), "child object"),
        (
            "target_file_name",
            serde_json::json!("../outside.lance"),
            "belongs to target",
        ),
        (
            "blob_ids",
            serde_json::json!({"start": 0, "end": 1}),
            "Blob ID range",
        ),
        (
            "blob_ids",
            serde_json::json!({"start": 2, "end": 2}),
            "Blob ID range",
        ),
        ("base_id", serde_json::json!(0), "belongs to target"),
    ] {
        let mut invalid = description.clone();
        invalid[key] = replacement;
        let part = serde_json::from_value::<DataFilePart>(invalid).unwrap();
        let error = dataset
            .concat_data_file_parts(&target, &[part])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }
    let error = dataset
        .concat_data_file_parts(&target, &[part.clone(), part])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("duplicate part"), "{error}");
    assert!(
        !dataset
            .table_state
            .object_store
            .exists(&dataset.data_dir().join(target.0.file_name()))
            .await
            .unwrap()
    );
}

#[rstest]
#[case::plain_primary(false, None)]
#[case::plain_registered(false, Some(7))]
#[case::blob_primary(true, None)]
#[case::blob_registered(true, Some(7))]
#[tokio::test]
async fn abandoned_target_cleanup_includes_failed_writes_and_preserves_other_targets(
    #[case] has_blob: bool,
    #[case] base_id: Option<u32>,
) {
    let dir = TempDir::default();
    let batch = if has_blob {
        let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
        let mut values = BlobArrayBuilder::new(1);
        values.push_bytes(b"payload").unwrap();
        RecordBatch::try_new(schema, vec![values.finish().unwrap()]).unwrap()
    } else {
        arrow_array::record_batch!(("id", Int32, [1])).unwrap()
    };
    let lease = |start| has_blob.then_some(start..start + 10);
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        format!("{}/dataset", dir.path_str()).as_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: base_id.map(|id| {
                vec![BasePath {
                    id,
                    name: Some("output".to_string()),
                    path: format!("{}/output", dir.path_str()),
                    is_dataset_root: false,
                }]
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let target = DataFileTarget::new(
        base_id,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let mut neighbor = serde_json::to_value(&target).unwrap();
    neighbor["file_name"] = serde_json::json!(format!("{}.neighbor.lance", target.0.file_name()));
    let neighbor: DataFileTarget = serde_json::from_value(neighbor).unwrap();
    let first = write_part(&dataset, &target, lease(1), batch.clone()).await;
    let assembled = dataset
        .concat_data_file_parts(&target, std::slice::from_ref(&first))
        .await
        .unwrap();
    let assembled_path = dataset
        .data_file_dir_for_base(base_id)
        .unwrap()
        .join(assembled.path.as_str());
    assert!(
        dataset
            .object_store(base_id)
            .await
            .unwrap()
            .exists(&assembled_path)
            .await
            .unwrap()
    );
    let other = write_part(&dataset, &neighbor, lease(1), batch.clone()).await;
    let error = dataset
        .write_data_file_part(
            &target,
            lease(20),
            stream::iter([
                Ok(batch),
                Err(Error::invalid_input("injected input failure")),
            ]),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("injected input failure"));
    let store = dataset.object_store(base_id).await.unwrap();
    let data_dir = dataset.data_file_dir_for_base(base_id).unwrap();
    let staging = data_dir.clone().join("_parts").join(target.0.file_name());
    // Simulate durable objects left when a worker exits without returning a part.
    store
        .put(&staging.clone().join("incomplete.part"), b"incomplete")
        .await
        .unwrap();
    let blob_dir = data_dir.clone().join(target.data_file_key());
    if has_blob {
        let orphan = blob_path(&data_dir, target.data_file_key(), 99);
        store.put(&orphan, b"orphan").await.unwrap();
    }
    assert!(
        store
            .exists(&staging.clone().join(part_file_name(&first).as_str()))
            .await
            .unwrap()
    );
    assert_eq!(
        !store
            .list(Some(blob_dir.clone()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .is_empty(),
        has_blob,
    );
    let restored: DataFileTarget =
        serde_json::from_slice(&serde_json::to_vec(&target).unwrap()).unwrap();
    restored.cleanup(&dataset).await.unwrap();
    assert!(!store.exists(&assembled_path).await.unwrap());
    restored.cleanup(&dataset).await.unwrap();
    assert!(
        store
            .list(Some(staging))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .is_empty()
    );
    let prefix = format!("{blob_dir}/");
    assert!(
        !store
            .list(Some(blob_dir))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .iter()
            .any(|object| object.location.as_ref().starts_with(&prefix))
    );
    let data_file = dataset
        .concat_data_file_parts(&neighbor, &[other])
        .await
        .unwrap();
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    let dataset = commit(&dataset, replacement).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let result = scanner.try_into_batch().await.unwrap();
    if has_blob {
        assert_eq!(result["blob"].as_binary::<i64>().value(0), b"payload");
    } else {
        assert_eq!(result["id"].as_primitive::<Int32Type>().value(0), 1);
    }
}

#[rstest]
#[case::plain_primary(false, None)]
#[case::plain_registered(false, Some(7))]
#[case::blob_primary(true, None)]
#[case::blob_registered(true, Some(7))]
#[tokio::test]
async fn restores_target_and_completed_parts_from_checkpoint(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[case] has_blob: bool,
    #[case] base_id: Option<u32>,
) {
    let dir = TempDir::default();
    let dataset_uri = format!("{}/dataset", dir.path_str());
    let make_batch = |ids: Vec<i32>| {
        let batch = arrow_array::record_batch!(("id", Int32, ids.clone())).unwrap();
        if !has_blob {
            return batch;
        }
        let mut blobs = BlobArrayBuilder::new(ids.len());
        for id in ids {
            blobs.push_bytes(format!("blob-{id}").as_bytes()).unwrap();
        }
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                batch.schema().field(0).clone(),
                blob_field("blob", true),
            ])),
            vec![batch["id"].clone(), blobs.finish().unwrap()],
        )
        .unwrap()
    };
    let lease = |start| has_blob.then_some(start..start + 10);

    let checkpoint = {
        let original = make_batch(vec![0, 1, 2, 3]);
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(original.clone())], original.schema()),
            dataset_uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(version),
                max_rows_per_file: 2,
                initial_bases: base_id.map(|id| {
                    vec![BasePath {
                        id,
                        name: Some("output".to_string()),
                        path: format!("{}/output", dir.path_str()),
                        is_dataset_root: false,
                    }]
                }),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let target = DataFileTarget::new(
            base_id,
            Arc::new(dataset.schema().clone()),
            dataset.manifest.data_storage_format.lance_file_format(),
        )
        .unwrap();
        let part = write_part(&dataset, &target, lease(1), make_batch(vec![10])).await;
        serde_json::to_vec(&(target, part, dataset.version_id())).unwrap()
    };

    let (target, first, snapshot): (DataFileTarget, DataFilePart, u64) =
        serde_json::from_slice(&checkpoint).unwrap();
    let dataset = Dataset::open(&dataset_uri)
        .await
        .unwrap()
        .checkout_version(snapshot)
        .await
        .unwrap();
    assert_eq!(target.0.schema(), dataset.schema());
    assert_eq!(target.0.base_id(), base_id);
    let data_dir = dataset.data_file_dir_for_base(base_id).unwrap();
    let store = dataset.object_store(base_id).await.unwrap();
    assert!(
        !store
            .exists(&data_dir.clone().join(target.0.file_name()))
            .await
            .unwrap()
    );
    let second = write_part(&dataset, &target, lease(11), make_batch(vec![11])).await;
    assert_ne!(
        part_file_name(&first).as_str(),
        part_file_name(&second).as_str()
    );
    let checkpoint = serde_json::to_vec(&(target, vec![first, second], snapshot)).unwrap();
    drop(store);
    drop(dataset);

    // Assembly reconstructs its own target and opens files only from descriptors.
    let (target, parts, snapshot): (DataFileTarget, Vec<DataFilePart>, u64) =
        serde_json::from_slice(&checkpoint).unwrap();
    let dataset = Dataset::open(&dataset_uri)
        .await
        .unwrap()
        .checkout_version(snapshot)
        .await
        .unwrap();
    let store = dataset.object_store(base_id).await.unwrap();
    let blob_dir = data_dir.clone().join(target.data_file_key());
    let blobs_before = store
        .list(Some(blob_dir.clone()))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let data_file = dataset
        .concat_data_file_parts(&target, &parts)
        .await
        .unwrap();
    let replacement = DataReplacementGroup(dataset.get_fragments()[0].id() as u64, data_file);
    assert_eq!(replacement.1.path, target.0.file_name());
    assert_eq!(replacement.1.base_id, base_id);
    assert_eq!(
        blobs_before,
        store
            .list(Some(blob_dir))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    );
    let dataset = commit(&dataset, replacement).await.unwrap();
    target.finish(&dataset).await.unwrap();
    target.finish(&dataset).await.unwrap();
    let staging_dir = data_dir.clone().join("_parts").join(target.0.file_name());
    assert!(
        store
            .list(Some(staging_dir))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .is_empty()
    );

    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values(),
        &[10, 11, 2, 3]
    );
    if has_blob {
        let values = batch["blob"].as_binary::<i64>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![
                Some(b"blob-10".as_slice()),
                Some(b"blob-11".as_slice()),
                Some(b"blob-2".as_slice()),
                Some(b"blob-3".as_slice()),
            ]
        );
    }
}

#[tokio::test]
async fn blob_part_requires_an_id_lease_before_writing() {
    let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_bytes(b"old").unwrap();
    let original = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
    let dataset = dataset_of(original, LanceFileVersion::V2_2).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let mut replacement = BlobArrayBuilder::new(1);
    replacement.push_bytes(b"new").unwrap();
    let batch = RecordBatch::try_new(schema, vec![replacement.finish().unwrap()]).unwrap();

    let error = dataset
        .write_data_file_part(&target, None, stream::iter([Ok(batch)]))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires a non-empty Blob ID range"),
        "{error}"
    );
}

#[tokio::test]
async fn data_file_part_rejects_non_empty_file_relative_inline_blob() {
    let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_bytes(b"ordinary-inline").unwrap();
    let batch = RecordBatch::try_new(schema, vec![blobs.finish().unwrap()]).unwrap();
    let dataset = dataset_of(batch, LanceFileVersion::V2_2).await;
    let fragment = only_fragment(&dataset);
    let data_file = &fragment.metadata().files[0];
    let scheduler = ScanScheduler::new(
        dataset.table_state.object_store.clone(),
        SchedulerConfig::default_for_testing(),
    );
    let file = scheduler
        .open_file(
            &dataset.data_dir().join(data_file.path.as_str()),
            &data_file.file_size_bytes,
        )
        .await
        .unwrap();

    let error = lance_file::concat::DataFilePart::open(EncodedFileInput::new(file), None, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("non-empty Inline"), "{error}");
}

#[tokio::test]
async fn complete_logical_blob_schema_and_external_range_survive_assembly() {
    let test_dir = TempDir::default();
    let dataset_path = test_dir.std_path().join("dataset");
    let external_base = test_dir.std_path().join("external");
    let external_objects = external_base.join("objects");
    fs::create_dir_all(&external_objects).unwrap();
    let external_path = external_objects.join("blob.bin");
    fs::write(&external_path, b"prefix-selected-suffix").unwrap();
    let external_uri = format!("file://{}", external_path.display());
    let external_base_uri = format!("file://{}", external_base.display());
    let original = complete_logical_blob_batch(&external_uri, 7, 8);
    let schema = original.schema();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(original)], schema),
        dataset_path.to_str().unwrap(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: Some(vec![BasePath {
                id: 1,
                name: Some("external".to_string()),
                path: external_base_uri,
                is_dataset_root: false,
            }]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    assert_eq!(
        target.0.schema().fields[0]
            .children
            .iter()
            .map(|child| child.name.as_str())
            .collect::<Vec<_>>(),
        ["data", "uri", "position", "size"]
    );

    let part = write_part(
        &dataset,
        &target,
        Some(1..10),
        complete_logical_blob_batch(&external_uri, 7, 8),
    )
    .await;
    assert_eq!(part.num_rows(), 1);
    let data_file = dataset
        .concat_data_file_parts(&target, &[part])
        .await
        .unwrap();
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    let dataset = commit(&dataset, replacement).await.unwrap();
    assert_eq!(dataset.schema().fields[0].children.len(), 4);

    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let batch = scanner.try_into_batch().await.unwrap();
    let values = batch["blob"].as_binary::<i64>();
    assert_eq!(values.value(0), b"selected");
}

#[tokio::test]
async fn blob_parts_write_sidecars_in_final_namespace_and_concat_descriptors() {
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        vec![
            Field::new("id", DataType::Int32, false),
            blob_field("blob", true),
        ],
        HashMap::from([(MANAGED_BLOBS_META_KEY.to_string(), "true".to_string())]),
    ));
    let make_batch = |ids: Vec<i32>, values: Vec<&'static [u8]>| {
        let mut blobs = BlobArrayBuilder::new(values.len());
        for value in values {
            blobs.push_bytes(value).unwrap();
        }
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(ids)),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap()
    };
    let make_prepared_batch = |id: i32, value: &'static [u8]| {
        let mut blobs = BlobDescriptorArrayBuilder::new("blob");
        blobs.push_inline(Bytes::from_static(value)).unwrap();
        let (blob_field, blob_array) = blobs.finish().unwrap().into_parts();
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int32, false),
                blob_field,
            ])),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![id])),
                blob_array,
            ],
        )
        .unwrap()
    };
    let dataset = dataset_of(
        make_batch(vec![0, 1], vec![b"old-0", b"old-1"]),
        LanceFileVersion::V2_2,
    )
    .await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let first = write_part(
        &dataset,
        &target,
        Some(1..10),
        make_prepared_batch(10, b"replacement-0"),
    )
    .await;
    let mut invalid = serde_json::to_value(&first).unwrap();
    invalid["blob_ids"] = serde_json::json!({"start": 20, "end": 30});
    let invalid: DataFilePart = serde_json::from_value(invalid).unwrap();
    // Managed references use base IDs, so their identity is independent of the
    // file-local sidecar lease recorded in the checkpoint.
    dataset
        .concat_data_file_parts(&target, &[invalid])
        .await
        .unwrap();
    let second = write_part(
        &dataset,
        &target,
        Some(10..20),
        make_batch(vec![11], vec![b"replacement-1"]),
    )
    .await;

    let other_target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let error = dataset
        .concat_data_file_parts(&other_target, &[first.clone(), second.clone()])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("belongs to target"), "{error}");
    assert!(
        !dataset
            .table_state
            .object_store
            .exists(&dataset.data_dir().join(other_target.0.file_name()))
            .await
            .unwrap()
    );

    let target =
        serde_json::from_slice::<DataFileTarget>(&serde_json::to_vec(&target).unwrap()).unwrap();
    let data_file = dataset
        .concat_data_file_parts(&target, &[first, second])
        .await
        .unwrap();
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    let dataset = commit(&dataset, replacement).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let batch = scanner.try_into_batch().await.unwrap();
    let values = batch["blob"].as_binary::<i64>();
    assert_eq!(values.value(0), b"replacement-0");
    assert_eq!(values.value(1), b"replacement-1");
}

#[rstest]
#[case::truncate(false)]
#[case::grow(true)]
#[tokio::test]
async fn managed_part_rejects_changed_physical_size_before_output(#[case] grow: bool) {
    let batch = arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap();
    let dataset = dataset_of(batch.clone(), LanceFileVersion::V2_2).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        ConcreteFileVersion::V2_2,
    )
    .unwrap();
    let part = write_part(&dataset, &target, None, batch).await;
    let path = target
        .0
        .parts_dir(&dataset.data_dir())
        .join(part_file_name(&part).as_str());
    let store = dataset.object_store(None).await.unwrap();
    let original = store
        .read_object(&path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let mut changed = original.to_vec();
    if grow {
        changed.push(0);
    } else {
        changed.pop();
    }
    store
        .put_payload(&path, Bytes::from(changed).into())
        .await
        .unwrap();
    let error = dataset
        .concat_data_file_parts(&target, std::slice::from_ref(&part))
        .await
        .unwrap_err();
    assert!(
        matches!(error.find_root(), Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(
        error.to_string().contains("expected complete file"),
        "{error}"
    );
    assert!(
        !store
            .exists(&dataset.data_dir().join(target.0.file_name()))
            .await
            .unwrap()
    );
    store.put_payload(&path, original.into()).await.unwrap();
    dataset
        .concat_data_file_parts(&target, &[part])
        .await
        .unwrap();
}

fn nested_blob_source_batch(item: Field, values: ArrayRef, managed: bool) -> RecordBatch {
    let item = Arc::new(item);
    let list_validity = Some(NullBuffer::from(vec![true, false, true]));
    let list = ListArray::try_new(
        item.clone(),
        OffsetBuffer::new(vec![0i32, 3, 4, 5].into()),
        values.clone(),
        list_validity.clone(),
    )
    .unwrap();
    let large_list = LargeListArray::try_new(
        item,
        OffsetBuffer::new(vec![0i64, 3, 4, 5].into()),
        values,
        list_validity,
    )
    .unwrap();
    let nested = StructArray::try_new(
        vec![
            Field::new("small", list.data_type().clone(), true),
            Field::new("large", large_list.data_type().clone(), true),
        ]
        .into(),
        vec![Arc::new(list), Arc::new(large_list)],
        Some(NullBuffer::from(vec![true, true, false])),
    )
    .unwrap();
    let fields = vec![Field::new("nested", nested.data_type().clone(), true)];
    let schema = if managed {
        managed_blob_schema(fields)
    } else {
        ArrowSchema::new(fields)
    };
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(nested)]).unwrap()
}

#[rstest]
#[case::ordinary_fragment(false, false, None)]
#[case::ordinary_parts(false, true, None)]
#[case::managed_fragment(true, false, None)]
#[case::managed_parts(true, true, None)]
#[case::managed_parts_registered_base(true, true, Some(7))]
#[tokio::test]
async fn completed_blob_uploads_round_trip_through_nested_columns(
    #[case] managed: bool,
    #[case] parts: bool,
    #[case] base_id: Option<u32>,
) {
    let directory = TempDir::default();
    let uri = directory.std_path().join("table");
    let uri = uri.to_str().unwrap();
    let item = blob_field("item", true);
    let original =
        nested_blob_source_batch(item.clone(), new_null_array(item.data_type(), 5), managed);
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(original.clone())], original.schema()),
        uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: base_id.map(|id| {
                vec![BasePath {
                    id,
                    name: Some("output".into()),
                    path: format!("{}/output", directory.path_str()),
                    is_dataset_root: false,
                }]
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let root = dataset.table_view().blob_base_path(base_id).unwrap();
    let relative = Path::from("_blobs/ingest/attempt-1/value.payload");
    let absolute: Path = root.parts().chain(relative.parts()).collect();
    let empty_relative = Path::from("_blobs/ingest/attempt-1/empty.payload");
    let empty_absolute: Path = root.parts().chain(empty_relative.parts()).collect();
    let payload = Bytes::from(vec![0x6D; 128 * 1024]);
    let store = dataset.object_store(base_id).await.unwrap();
    store
        .put_payload(&absolute, payload.clone().into())
        .await
        .unwrap();
    store
        .put_payload(&empty_absolute, Bytes::new().into())
        .await
        .unwrap();

    let mut sources = BlobWriteSources::default();
    let uploaded = sources
        .add_uploaded_blob(relative.clone(), payload.len() as u64)
        .unwrap();
    let empty = sources.add_uploaded_blob(empty_relative, 0).unwrap();
    // The object would fail to resolve if either kind of null parent lost its validity.
    let ignored = sources
        .add_uploaded_blob(Path::from("_blobs/ingest/attempt-1/missing.payload"), 10)
        .unwrap();
    let mut replacement = BlobDescriptorArrayBuilder::for_field(&item).unwrap();
    replacement
        .extend([
            uploaded,
            crate::BlobDescriptor::Null,
            empty,
            ignored.clone(),
            ignored,
        ])
        .unwrap();
    let (item, values) = replacement.finish().unwrap().into_parts();
    let batch = nested_blob_source_batch(item, values, managed);
    let fragment = only_fragment(&dataset);
    let options = FileWriterOptions::default();
    let replacement = if parts {
        let target = DataFileTarget::new(
            base_id,
            Arc::new(dataset.schema().clone()),
            ConcreteFileVersion::V2_2,
        )
        .unwrap();
        let ids = (!managed).then_some(99..105);
        let part = dataset
            .write_data_file_part_with_blob_sources(
                &target,
                ids,
                stream::iter([Ok((batch, sources))]),
                options,
            )
            .await
            .unwrap();
        let (target, part): (DataFileTarget, DataFilePart) =
            serde_json::from_slice(&serde_json::to_vec(&(target, part)).unwrap()).unwrap();
        let data_file = dataset
            .concat_data_file_parts(&target, &[part])
            .await
            .unwrap();
        DataReplacementGroup(fragment.id() as u64, data_file)
    } else {
        // The caller may project the fields without carrying schema metadata.
        let mut schema = dataset.schema().clone();
        schema.metadata.clear();
        fragment
            .write_columns_with_blob_sources(stream::iter([Ok((batch, sources))]), &schema, options)
            .await
            .unwrap()
    };
    let published = commit(&dataset, replacement).await.unwrap();
    if !managed {
        // Published ordinary values must have their own storage.
        store.delete(&absolute).await.unwrap();
        store.delete(&empty_absolute).await.unwrap();
    }
    let reopened = Dataset::open(uri).await.unwrap();
    assert_eq!(reopened.version_id(), published.version_id());
    assert_eq!(reopened.manifest.has_managed_blobs(), managed);
    let descriptors = reopened.scan().try_into_batch().await.unwrap();
    let nested = descriptors["nested"].as_struct();
    let list = nested.column_by_name("small").unwrap().as_list::<i32>();
    let descriptors = list.values().as_struct();
    let kind = descriptors
        .column_by_name("kind")
        .unwrap()
        .as_primitive::<arrow_array::types::UInt8Type>();
    let uri = descriptors
        .column_by_name("blob_uri")
        .unwrap()
        .as_string::<i32>();
    let ids = descriptors
        .column_by_name("blob_id")
        .unwrap()
        .as_primitive::<arrow_array::types::UInt32Type>();
    if let Some(id) = base_id {
        assert_eq!(kind.value(0), BlobKind::ManagedWithBase as u8);
        assert_eq!(ids.value(0), id);
        assert_eq!(uri.value(0), relative.as_ref());
    } else if managed {
        assert_eq!(kind.value(0), BlobKind::Managed as u8);
        assert_eq!(uri.value(0), relative.as_ref());
    } else {
        assert_eq!(kind.value(0), BlobKind::Packed as u8);
        if parts {
            assert_eq!(ids.value(0), 99);
        }
    }
    let decoded = reopened
        .scan()
        .blob_handling(BlobHandling::AllBinary)
        .try_into_batch()
        .await
        .unwrap();
    let nested = decoded["nested"].as_struct();
    assert!(nested.is_null(2));
    let small = nested.column_by_name("small").unwrap().as_list::<i32>();
    let large = nested.column_by_name("large").unwrap().as_list::<i64>();
    assert!(small.is_null(1));
    assert!(large.is_null(1));
    for list in [small.value(0), large.value(0)] {
        assert_eq!(
            list.as_binary::<i64>().iter().collect::<Vec<_>>(),
            [Some(payload.as_ref()), None, Some(b"".as_slice()),]
        );
    }
}

#[tokio::test]
async fn fragment_copies_registered_blob_ranges_with_bounded_requests() {
    const MAX_REQUEST: usize = 8 * 1024 * 1024;
    let field = blob_field("blob", true);
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![field.clone()])),
        vec![new_null_array(field.data_type(), 3)],
    )
    .unwrap();
    let dataset = dataset_of(batch, LanceFileVersion::V2_2).await;
    let mut source_data = vec![0x13; MAX_REQUEST + 64];
    source_data[MAX_REQUEST..].fill(0x71);
    let source_data = Bytes::from(source_data);
    let (store, _) = ObjectStore::from_uri("memory://").await.unwrap();
    let path = Path::from("existing-sidecar");
    store.put(&path, &source_data).await.unwrap();
    store.io_stats_incremental();
    let ranges = [5..9, 7..(128 * 1024 + 7), 9..(MAX_REQUEST + 27)];
    let mut sources = BlobWriteSources::default();
    let mut blobs = BlobDescriptorArrayBuilder::for_field(&field).unwrap();
    for range in &ranges {
        let file = BlobFile::new_packed(
            store.clone(),
            path.clone(),
            range.start as u64,
            range.len() as u64,
        );
        blobs.push(sources.add_blob_file(file).unwrap()).unwrap();
    }
    let (prepared_field, prepared_values) = blobs.finish().unwrap().into_parts();
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![prepared_field])),
        vec![prepared_values],
    )
    .unwrap();
    let replacement = only_fragment(&dataset)
        .write_columns_with_blob_sources(
            stream::iter([Ok((batch, sources))]),
            dataset.schema(),
            FileWriterOptions::default(),
        )
        .await
        .unwrap();
    let requests = store
        .io_stats_incremental()
        .requests
        .into_iter()
        .filter(|request| request.path == path)
        .filter_map(|request| request.range)
        .collect::<Vec<_>>();
    assert!(
        requests
            .iter()
            .all(|range| range.end - range.start <= MAX_REQUEST as u64),
        "{requests:?}"
    );
    assert!(
        requests
            .iter()
            .any(|range| range.end == (MAX_REQUEST + 27) as u64),
        "{requests:?}"
    );
    let dataset = Arc::new(commit(&dataset, replacement).await.unwrap());
    let files = dataset
        .take_blobs_by_indices(&[0, 1, 2], "blob")
        .await
        .unwrap();
    for ((file, range), expected) in
        files
            .into_iter()
            .zip(ranges)
            .zip([BlobKind::Inline, BlobKind::Packed, BlobKind::Dedicated])
    {
        let file = file.unwrap();
        assert_eq!(file.kind(), expected);
        assert_eq!(file.read().await.unwrap(), source_data.slice(range));
    }
}

/// Pages a single part holds when its writer is given `options`.
async fn part_page_count(
    dataset: &Dataset,
    target: &DataFileTarget,
    options: FileWriterOptions,
) -> usize {
    let batches = (0..8).map(|batch| {
        Ok(arrow_array::record_batch!((
            "value",
            Int32,
            (batch * 32..batch * 32 + 32).collect::<Vec<i32>>()
        ))
        .unwrap())
    });
    let part = dataset
        .write_data_file_part_with_file_writer_options(target, None, stream::iter(batches), options)
        .await
        .unwrap();
    let path = target
        .0
        .parts_dir(&dataset.data_file_dir_for_base(target.0.base_id()).unwrap())
        .join(part_file_name(&part).as_str());
    let scheduler = ScanScheduler::new(
        dataset.table_state.object_store.clone(),
        SchedulerConfig::default_for_testing(),
    );
    let file = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = lance_file::reader::FileReader::try_open(
        file,
        None,
        Default::default(),
        &dataset
            .table_state
            .metadata_cache
            .file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    reader.metadata().column_metadatas[0].pages.len()
}

/// A part written through the dataset API must honour the caller's writer
/// budget: without it every part buffers the writer's own default allowance,
/// which a caller bounding a 100 MB+ Function output cannot account for.
#[rstest]
#[case(LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_3)]
#[tokio::test]
async fn part_write_applies_file_writer_cache_option(#[case] version: LanceFileVersion) {
    let dataset = dataset_of(
        arrow_array::record_batch!(("value", Int32, (0..256).collect::<Vec<_>>())).unwrap(),
        version,
    )
    .await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();

    let default_pages = part_page_count(&dataset, &target, FileWriterOptions::default()).await;
    let bounded_pages = part_page_count(
        &dataset,
        &target,
        FileWriterOptions {
            data_cache_bytes: Some(1),
            ..Default::default()
        },
    )
    .await;

    assert!(
        bounded_pages > default_pages,
        "the part writer must not discard FileWriterOptions: default={default_pages}, bounded={bounded_pages}"
    );
}
