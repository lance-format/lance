// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, fs, ops::Range, sync::Arc};

use arrow::array::AsArray;
use arrow_array::{
    ArrayRef, LargeBinaryArray, RecordBatch, RecordBatchIterator, StringArray, StructArray,
    UInt64Array, types::Int32Type,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use futures::{TryStreamExt, stream};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use lance_core::{
    Error,
    datatypes::{BLOB_V2_LOGICAL_FIELDS, BlobHandling},
    utils::{blob::blob_path, tempfile::TempDir},
};
use lance_file::concat::EncodedFileInput;
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::BasePath;
use rstest::rstest;

use crate::blob::{BlobArrayBuilder, BlobDescriptorArrayBuilder, blob_field};
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::write::WriteParams;
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
    let data_file = dataset
        .concat_data_file_parts(&target, &ordered_parts)
        .await
        .unwrap();
    let replacement = DataReplacementGroup(only_fragment(&dataset).id() as u64, data_file);
    assert_eq!(replacement.1.path, target.file_name.as_str());
    let dataset = commit(&dataset, replacement).await.unwrap();
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values(),
        &[12, 13, 10, 11]
    );
}

#[tokio::test]
async fn target_uses_an_ordinary_generated_data_file_name() {
    let original = arrow_array::record_batch!(("id", Int32, [0, 1])).unwrap();
    let dataset = dataset_of(original, LanceFileVersion::V2_1).await;
    let first = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let second = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();

    assert_ne!(first.file_name.as_str(), second.file_name.as_str());
    assert_eq!(first.file_name.len(), 56);
    assert!(first.file_name.ends_with(".lance"));
    assert!(!first.file_name.contains('/'));

    let restored =
        serde_json::from_slice::<DataFileTarget>(&serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(restored.file_name.as_str(), first.file_name.as_str());
    assert_eq!(restored.base_id, first.base_id);
    assert_eq!(restored.schema, first.schema);
    assert_eq!(restored.version, first.version);
}

#[test]
fn target_serialization_preserves_schema_and_rejects_invalid_identity() {
    let arrow_schema = ArrowSchema::new(vec![
        Field::new(
            "nested",
            DataType::Struct(vec![Field::new("value", DataType::Int32, true)].into()),
            true,
        ),
        Field::new(
            "category",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        ),
    ]);
    let mut schema = lance_core::datatypes::Schema::try_from(&arrow_schema).unwrap();
    schema
        .metadata
        .insert("owner".to_string(), "test".to_string());
    schema.fields[0].id = 10;
    schema.fields[0].children[0].id = 11;
    schema.fields[0].children[0].parent_id = 10;
    schema.fields[0].children[0]
        .metadata
        .insert("unit".to_string(), "count".to_string());
    schema.fields[1].id = 20;
    schema.fields[1].dictionary = Some(lance_core::datatypes::Dictionary {
        offset: 12,
        length: 2,
        values: Some(Arc::new(StringArray::from(vec!["a", "b"]))),
    });
    let target = DataFileTarget::new(Some(7), Arc::new(schema), ConcreteFileVersion::V2_2).unwrap();
    let value = serde_json::to_value(&target).unwrap();
    let restored: DataFileTarget = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(restored.schema, target.schema);
    assert_eq!(restored.file_name.as_str(), target.file_name.as_str());
    assert_eq!(restored.base_id, target.base_id);
    let dictionary = restored.schema.fields[1].dictionary.as_ref().unwrap();
    assert_eq!((dictionary.offset, dictionary.length), (12, 2));
    for (key, replacement, message) in [
        (
            "format_version",
            serde_json::json!(99),
            "checkpoint version",
        ),
        ("file_version", serde_json::json!("stable"), "version"),
        (
            "file_name",
            serde_json::json!("../outside.lance"),
            "target identity",
        ),
        ("file_name", serde_json::json!(""), "target identity"),
        ("file_name", serde_json::json!(".lance"), "target identity"),
        ("file_name", serde_json::json!(".."), "target identity"),
        ("base_id", serde_json::json!(0), "base_id"),
    ] {
        let mut invalid = value.clone();
        invalid[key] = replacement;
        let error = serde_json::from_value::<DataFileTarget>(invalid).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
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
    let part = write_part(&dataset, &target, None, batch).await;
    let mut description = serde_json::to_value(&part).unwrap();
    description[field] = value;
    let invalid = serde_json::from_value::<DataFilePart>(description).unwrap();
    let error = dataset
        .concat_data_file_parts(&target, &[invalid])
        .await
        .unwrap_err();
    if field == "file_name" {
        assert!(matches!(error, Error::NotFound { .. }), "{error}");
    } else {
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
    assert!(error.to_string().contains(message), "{error}");
    assert!(
        !dataset
            .object_store
            .exists(&dataset.data_dir().join(target.file_name.as_str()))
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
            .object_store
            .exists(&dataset.data_dir().join(target.file_name.as_str()))
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
    neighbor["file_name"] = serde_json::json!(format!("{}.neighbor.lance", target.file_name));
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
    let staging = data_dir
        .clone()
        .join("_parts")
        .join(target.file_name.as_str());
    // Simulate durable objects left when a worker exits without returning a part.
    store
        .put(&staging.clone().join("incomplete.part"), b"incomplete")
        .await
        .unwrap();
    let blob_dir = data_dir
        .clone()
        .join(target.file_name.strip_suffix(".lance").unwrap());
    if has_blob {
        let orphan = blob_path(&data_dir, target.data_file_key(), 99);
        store.put(&orphan, b"orphan").await.unwrap();
    }
    assert!(
        store
            .exists(&staging.clone().join(first.file_name.as_str()))
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
                data_storage_version: Some(LanceFileVersion::V2_2),
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
    assert_eq!(target.schema.as_ref(), dataset.schema());
    assert_eq!(target.base_id, base_id);
    let data_dir = dataset.data_file_dir_for_base(base_id).unwrap();
    let store = dataset.object_store(base_id).await.unwrap();
    assert!(
        !store
            .exists(&data_dir.clone().join(target.file_name.as_str()))
            .await
            .unwrap()
    );
    let second = write_part(&dataset, &target, lease(11), make_batch(vec![11])).await;
    assert_ne!(first.file_name.as_str(), second.file_name.as_str());
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
    let blob_dir = data_dir
        .clone()
        .join(target.file_name.strip_suffix(".lance").unwrap());
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
    assert_eq!(replacement.1.path, target.file_name.as_str());
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
    let staging_dir = data_dir
        .clone()
        .join("_parts")
        .join(target.file_name.as_str());
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
    let data_file = &only_fragment(&dataset).metadata.files[0];
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
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
        target.schema.fields[0]
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
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        blob_field("blob", true),
    ]));
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
    let error = dataset
        .concat_data_file_parts(&target, &[invalid])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("outside declared range"),
        "{error}"
    );
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
            .object_store
            .exists(&dataset.data_dir().join(other_target.file_name.as_str()))
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
