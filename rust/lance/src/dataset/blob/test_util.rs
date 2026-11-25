// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    Dataset,
    blob::{BlobArrayBuilder, blob_field},
    dataset::WriteParams,
    utils::test::TestDatasetGenerator,
};
use arrow::datatypes::UInt64Type;
use arrow_array::RecordBatch;
use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, RecordBatchIterator, StringArray, StructArray, UInt32Array,
    UInt64Array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME};
use lance_core::datatypes::BLOB_V2_LOGICAL_FIELDS;
use lance_core::{
    Result,
    utils::tempfile::{TempDir, TempStrDir},
};
use lance_datagen::{BatchCount, RowCount, array};
use lance_file::version::LanceFileVersion;
use lance_table::feature_flags::MANAGED_BLOBS_META_KEY;
use lance_table::format::BasePath;
use std::collections::HashMap;
use std::sync::Arc;

pub fn managed_blob_schema(fields: Vec<Field>) -> Schema {
    Schema::new_with_metadata(
        fields,
        HashMap::from([(MANAGED_BLOBS_META_KEY.to_string(), "true".to_string())]),
    )
}

pub struct BlobTestFixture {
    pub _test_dir: TempStrDir,
    pub dataset: Arc<Dataset>,
    pub data: Vec<RecordBatch>,
}

impl BlobTestFixture {
    pub async fn new() -> Self {
        let test_dir = TempStrDir::default();

        let data = lance_datagen::gen_batch()
            .col("filterme", array::step::<UInt64Type>())
            .col("blobs", array::blob())
            .into_reader_rows(RowCount::from(10), BatchCount::from(4))
            .map(|batch| Ok(batch?))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        let dataset = Arc::new(
            TestDatasetGenerator::new(data.clone(), LanceFileVersion::V2_1)
                .make_hostile(&test_dir)
                .await,
        );

        Self {
            _test_dir: test_dir,
            dataset,
            data,
        }
    }
}

pub struct MultiBaseBlobFixture {
    pub _test_dir: TempDir,
    pub dataset: Arc<Dataset>,
    pub expected: Vec<u8>,
}

pub fn complete_blob_v2_field(name: &str, nullable: bool) -> Field {
    Field::new(
        name,
        DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()),
        nullable,
    )
    .with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]))
}

pub fn complete_blob_v2_array(
    data: Vec<Option<Vec<u8>>>,
    uris: Vec<Option<String>>,
    positions: Vec<Option<u64>>,
    sizes: Vec<Option<u64>>,
    validity: Option<NullBuffer>,
) -> ArrayRef {
    Arc::new(
        StructArray::try_new(
            BLOB_V2_LOGICAL_FIELDS.clone(),
            vec![
                Arc::new(LargeBinaryArray::from_iter(
                    data.iter().map(|value| value.as_deref()),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    uris.iter().map(|value| value.as_deref()),
                )) as ArrayRef,
                Arc::new(UInt64Array::from(positions)) as ArrayRef,
                Arc::new(UInt64Array::from(sizes)) as ArrayRef,
            ],
            validity,
        )
        .unwrap(),
    )
}

pub fn nested_blob_v2_batch_with_field(
    blob_field: Field,
    blob_array: ArrayRef,
) -> (Arc<Schema>, RecordBatch) {
    let info_fields = vec![Field::new("name", DataType::Utf8, false), blob_field];
    let info_array: ArrayRef = Arc::new(
        StructArray::try_new(
            info_fields.clone().into(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (0..blob_array.len()).map(|idx| format!("name-{idx}")),
                )) as ArrayRef,
                blob_array,
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
    (schema, batch)
}

pub fn nested_blob_v2_batch(blob_array: ArrayRef) -> (Arc<Schema>, RecordBatch) {
    nested_blob_v2_batch_with_field(blob_field("blob", true), blob_array)
}

pub async fn create_multi_base_blob_v2_fixture(
    payload: Vec<u8>,
    dedicated_threshold: Option<usize>,
    is_dataset_root: bool,
) -> MultiBaseBlobFixture {
    let test_dir = TempDir::default();
    let primary_uri = test_dir.path_str();
    let base_dir = test_dir.std_path().join("blob_base");
    std::fs::create_dir_all(&base_dir).unwrap();
    let base_uri = format!("file://{}", base_dir.display());

    let mut blob_builder = BlobArrayBuilder::new(2);
    blob_builder.push_bytes(payload.clone()).unwrap();
    blob_builder.push_bytes(payload.clone()).unwrap();
    let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();

    let mut blob_column = blob_field("blob", true);
    if let Some(threshold) = dedicated_threshold {
        let mut metadata = blob_column.metadata().clone();
        metadata.insert(
            BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
            threshold.to_string(),
        );
        blob_column = blob_column.with_metadata(metadata);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_column,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(UInt32Array::from(vec![0, 1])), blob_array],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    let dataset = Arc::new(
        Dataset::write(
            reader,
            &primary_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("blob_base".to_string()),
                    path: base_uri,
                    is_dataset_root,
                }]),
                target_bases: Some(vec![1]),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    assert!(
        dataset
            .fragments()
            .iter()
            .all(|frag| frag.files.iter().all(|file| file.base_id == Some(1)))
    );
    assert_eq!(dataset.fragments().len(), 2);

    MultiBaseBlobFixture {
        _test_dir: test_dir,
        dataset,
        expected: payload,
    }
}
