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
use futures::stream;
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use lance_core::{
    datatypes::{BLOB_V2_LOGICAL_FIELDS, BlobHandling},
    utils::tempfile::TempDir,
};
use lance_file::concat::EncodedFileInput;
use lance_file::version::LanceFileVersion;
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use lance_table::format::BasePath;

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
    staging_name: &str,
    blob_ids: Option<Range<u32>>,
    batch: RecordBatch,
) -> DataFilePart {
    let path = dataset.data_dir().join(staging_name);
    let output = dataset.object_store.create(&path).await.unwrap();
    let summary = dataset
        .write_data_file_part(target, output, blob_ids.clone(), stream::iter([Ok(batch)]))
        .await
        .unwrap();
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::default_for_testing(),
    );
    let file = scheduler
        .open_file(&path, &CachedFileSize::new(summary.size_bytes))
        .await
        .unwrap();
    target
        .open_part(
            EncodedFileInput::new(file).with_expected_num_rows(summary.num_rows),
            blob_ids,
        )
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
        "part-1.lance",
        None,
        arrow_array::record_batch!(("id", Int32, [10, 11])).unwrap(),
    )
    .await;
    let second = write_part(
        &dataset,
        &target,
        "part-2.lance",
        None,
        arrow_array::record_batch!(("id", Int32, [12, 13])).unwrap(),
    )
    .await;

    let replacement = only_fragment(&dataset)
        .write_columns_from_parts(&target, &[second, first])
        .await
        .unwrap();
    assert_eq!(replacement.1.path, target.file_name());
    let dataset = commit(&dataset, replacement).await.unwrap();
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values(),
        &[12, 13, 10, 11]
    );
}

#[tokio::test]
async fn fragment_adapter_rejects_incomplete_physical_coverage() {
    let original = arrow_array::record_batch!(("id", Int32, [0, 1, 2])).unwrap();
    let dataset = dataset_of(original, LanceFileVersion::V2_1).await;
    let target = DataFileTarget::new(
        None,
        Arc::new(dataset.schema().clone()),
        dataset.manifest.data_storage_format.lance_file_format(),
    )
    .unwrap();
    let part = write_part(
        &dataset,
        &target,
        "short-part.lance",
        None,
        arrow_array::record_batch!(("id", Int32, [10, 11])).unwrap(),
    )
    .await;
    let error = only_fragment(&dataset)
        .write_columns_from_parts(&target, &[part])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("2 physical rows"), "{error}");
    assert!(error.to_string().contains("contains 3"), "{error}");
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

    assert_ne!(first.file_name(), second.file_name());
    assert_eq!(first.file_name().len(), 56);
    assert!(first.file_name().ends_with(".lance"));
    assert!(!first.file_name().contains('/'));
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
    let output = dataset
        .object_store
        .create(&dataset.data_dir().join("missing-lease-part.lance"))
        .await
        .unwrap();
    let mut replacement = BlobArrayBuilder::new(1);
    replacement.push_bytes(b"new").unwrap();
    let batch = RecordBatch::try_new(schema, vec![replacement.finish().unwrap()]).unwrap();

    let error = dataset
        .write_data_file_part(&target, output, None, stream::iter([Ok(batch)]))
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

    let error = DataFilePart::open(EncodedFileInput::new(file), None, None)
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
        target.schema().fields[0]
            .children
            .iter()
            .map(|child| child.name.as_str())
            .collect::<Vec<_>>(),
        ["data", "uri", "position", "size"]
    );

    let part = write_part(
        &dataset,
        &target,
        "complete-logical-part.lance",
        Some(1..10),
        complete_logical_blob_batch(&external_uri, 7, 8),
    )
    .await;
    assert_eq!(part.num_rows(), 1);
    let replacement = only_fragment(&dataset)
        .write_columns_from_parts(&target, &[part])
        .await
        .unwrap();
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
        "blob-part-1.lance",
        Some(1..10),
        make_prepared_batch(10, b"replacement-0"),
    )
    .await;
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::default_for_testing(),
    );
    let file = scheduler
        .open_file(
            &dataset.data_dir().join("blob-part-1.lance"),
            &CachedFileSize::unknown(),
        )
        .await
        .unwrap();
    let error = target
        .open_part(EncodedFileInput::new(file), Some(20..30))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("outside declared range"),
        "{error}"
    );
    let second = write_part(
        &dataset,
        &target,
        "blob-part-2.lance",
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
    assert!(error.to_string().contains("Blob target ID"), "{error}");
    assert!(
        !dataset
            .object_store
            .exists(&dataset.data_dir().join(other_target.file_name()))
            .await
            .unwrap()
    );

    let replacement = only_fragment(&dataset)
        .write_columns_from_parts(&target, &[first, second])
        .await
        .unwrap();
    let dataset = commit(&dataset, replacement).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.blob_handling(BlobHandling::AllBinary);
    let batch = scanner.try_into_batch().await.unwrap();
    let values = batch["blob"].as_binary::<i64>();
    assert_eq!(values.value(0), b"replacement-0");
    assert_eq!(values.value(1), b"replacement-1");
}
