// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::NewColumnTransform;
use arrow_array::{Decimal128Array, UInt64Array};

const NON_LEGACY_VERSIONS: [LanceFileVersion; 4] = [
    LanceFileVersion::V2_0,
    LanceFileVersion::V2_1,
    LanceFileVersion::V2_2,
    LanceFileVersion::V2_3,
];

async fn write_two_column_schema_evolution_dataset(uri: &str) -> Dataset {
    let batch =
        arrow_array::record_batch!(("a", Int32, [1, 2, 3, 4]), ("b", Int32, [10, 20, 30, 40]))
            .unwrap();
    let schema = batch.schema();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn test_binary_copy_merge_small_files() {
    for version in NON_LEGACY_VERSIONS {
        do_test_binary_copy_merge_small_files(version).await;
    }
}

async fn do_test_binary_copy_merge_small_files(version: LanceFileVersion) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let reader2 = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 2_500,
        max_rows_per_group: 1_000,
        data_storage_version: Some(version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    dataset.append(reader2, Some(write_params)).await.unwrap();

    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000_000,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_added >= 1);
    assert_eq!(
        dataset.count_rows(None).await.unwrap() as usize,
        before.num_rows()
    );
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
}

#[rstest]
#[case::default_inputs(LanceFileVersion::V2_0, LanceFileVersion::V2_0)]
#[case::non_default_inputs(LanceFileVersion::V2_1, LanceFileVersion::V2_1)]
#[case::mixed_inputs(LanceFileVersion::V2_0, LanceFileVersion::V2_2)]
#[tokio::test]
async fn test_compaction_exact_target(
    #[case] first: LanceFileVersion,
    #[case] second: LanceFileVersion,
    #[values(
        None,
        Some(LanceFileVersion::V2_0),
        Some(LanceFileVersion::V2_1),
        Some(LanceFileVersion::V2_2),
        Some(LanceFileVersion::V2_3)
    )]
    target: Option<LanceFileVersion>,
    #[values(
        CompactionMode::Reencode,
        CompactionMode::TryBinaryCopy,
        CompactionMode::ForceBinaryCopy
    )]
    mode: CompactionMode,
) {
    let batch = arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(RecordBatch::new_empty(batch.schema()))], batch.schema()),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for version in [first, second] {
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }
    let before = dataset.scan().try_into_batch().await.unwrap();
    let read_version = dataset.manifest.version;
    let options = CompactionOptions {
        target_rows_per_fragment: 8,
        compaction_mode: Some(mode),
        data_storage_version: target,
        ..Default::default()
    };
    let expected = target.unwrap_or(LanceFileVersion::V2_0).resolve();
    let versions_match = first.resolve() == expected && second.resolve() == expected;
    assert_eq!(
        can_use_binary_copy(&dataset, &options, &dataset.manifest.fragments).await,
        versions_match && mode != CompactionMode::Reencode,
    );
    let result = compact_files(&mut dataset, options, None).await;
    if mode == CompactionMode::ForceBinaryCopy && !versions_match {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        let message = error.to_string();
        assert!(message.contains(&format!("target is {expected}")));
        assert!(message.contains("uses"));
        assert!(message.contains(".lance"));
        assert_eq!(dataset.manifest.version, read_version);
    } else {
        result.unwrap();
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert!(
            dataset
                .manifest
                .fragments
                .iter()
                .flat_map(Fragment::referenced_lance_files)
                .all(|file| file.file_version().unwrap() == expected)
        );
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
        dataset.validate().await.unwrap();
    }
    assert_eq!(
        dataset.manifest.data_storage_format.lance_file_format(),
        ConcreteFileVersion::V2_0
    );
}

#[tokio::test]
async fn test_binary_copy_falls_back_for_non_schema_column_order() {
    let decimal_type = DataType::Decimal128(38, 10);
    let dataset_schema = Arc::new(Schema::new(vec![
        Field::new("v_dec", decimal_type.clone(), true),
        Field::new("v_u64", DataType::UInt64, true),
    ]));
    let write_params = WriteParams {
        max_rows_per_file: 1,
        data_storage_version: Some(LanceFileVersion::V2_3),
        ..Default::default()
    };
    let test_dir = TempStrDir::default();
    let empty_batch = RecordBatch::new_empty(dataset_schema.clone());
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(empty_batch)], dataset_schema),
        &test_dir,
        Some(write_params.clone()),
    )
    .await
    .unwrap();

    let decimal_values = Decimal128Array::from_iter_values([
        201_000_000_000_000_000_000_000_i128,
        202_000_000_000_000_000_000_000_i128,
    ])
    .with_precision_and_scale(38, 10)
    .unwrap();
    let swapped_schema = Arc::new(Schema::new(vec![
        Field::new("v_u64", DataType::UInt64, true),
        Field::new("v_dec", decimal_type, true),
    ]));
    let swapped_batch = RecordBatch::try_new(
        swapped_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![201, 202])),
            Arc::new(decimal_values),
        ],
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(swapped_batch)], swapped_schema),
            Some(write_params),
        )
        .await
        .unwrap();

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(fragments.len(), 2);
    for fragment in &fragments {
        assert_eq!(fragment.files[0].fields.as_ref(), &[1, 0]);
        assert_eq!(fragment.files[0].column_indices.as_ref(), &[0, 1]);
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 8,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
    let compacted_file = &dataset.manifest.fragments[0].files[0];
    assert_eq!(compacted_file.fields.as_ref(), &[0, 1]);
    assert_eq!(compacted_file.column_indices.as_ref(), &[0, 1]);
}

#[tokio::test]
async fn test_binary_copy_packed_struct_column_mapping() {
    for version in NON_LEGACY_VERSIONS {
        do_test_binary_copy_packed_struct_column_mapping(version).await;
    }
}

async fn do_test_binary_copy_packed_struct_column_mapping(version: LanceFileVersion) {
    use arrow_array::StructArray;
    use arrow_schema::Fields;
    use std::collections::HashMap;

    let packed_fields = Fields::from(vec![Field::new("child", DataType::Int32, true)]);
    let packed_field = Field::new("packed", DataType::Struct(packed_fields.clone()), true)
        .with_metadata(HashMap::from([("packed".to_string(), "true".to_string())]));
    let schema = Arc::new(Schema::new(vec![
        packed_field,
        Field::new("tail", DataType::Int32, true),
    ]));
    let packed: ArrayRef = Arc::new(StructArray::new(
        packed_fields,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
        None,
    ));
    let tail: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));
    let batch = RecordBatch::try_new(schema.clone(), vec![packed, tail]).unwrap();

    let test_dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            data_storage_version: Some(version),
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let before = dataset.scan().try_into_batch().await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 100_000,
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);

    let data_file = &dataset.manifest.fragments[0].files[0];
    assert_eq!(data_file.fields.len(), 2);
    assert_eq!(data_file.column_indices.as_ref(), &[0, 1]);
}

#[tokio::test]
async fn test_binary_copy_empty_string_scalar_index() {
    for version in NON_LEGACY_VERSIONS {
        do_test_binary_copy_empty_string_scalar_index(version).await;
    }
}

async fn do_test_binary_copy_empty_string_scalar_index(version: LanceFileVersion) {
    use arrow_array::StringArray;

    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from_iter_values(
            std::iter::repeat_n("", 4_000),
        ))],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 1_000,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    // Scanning the compacted data must round-trip the zero-length buffers
    // untouched; a corrupted offsets table would surface as mismatched data here.
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);

    dataset
        .create_index(
            &["text"],
            IndexType::Scalar,
            Some("text_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // The scalar index over the empty strings must also return the full set.
    let indexed = dataset
        .scan()
        .filter("text = ''")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(indexed.num_rows(), 4_000);
}

#[tokio::test]
async fn test_binary_copy_with_defer_remap() {
    for version in NON_LEGACY_VERSIONS {
        do_test_binary_copy_with_defer_remap(version).await;
    }
}

async fn do_test_binary_copy_with_defer_remap(version: LanceFileVersion) {
    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
    use std::sync::Arc;

    let fixed_list_dt =
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4);

    let meta_fields = Fields::from(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", fixed_list_dt.clone(), true),
    ]);

    let inner_fields = Fields::from(vec![
        Field::new("x", DataType::UInt32, true),
        Field::new("y", DataType::LargeUtf8, true),
    ]);
    let nested_fields = Fields::from(vec![
        Field::new("inner", DataType::Struct(inner_fields.clone()), true),
        Field::new("fsb", DataType::FixedSizeBinary(8), true),
    ]);

    let event_fields = Fields::from(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("payload", DataType::Binary, true),
    ]);

    let reader = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
        .col("i", array::step::<Int32Type>())
        .col("meta", array::rand_struct(meta_fields))
        .col("nested", array::rand_struct(nested_fields))
        .col(
            "events",
            array::rand_list_any(array::rand_struct(event_fields), true),
        )
        .into_reader_rows(RowCount::from(6_000), BatchCount::from(1));

    let mut dataset = Dataset::write(
        reader,
        "memory://test/binary_copy_nested",
        Some(WriteParams {
            max_rows_per_file: 1_000,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let before_batch = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        defer_index_remap: true,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after_batch = dataset.scan().try_into_batch().await.unwrap();

    assert_eq!(before_batch, after_batch);
}

#[rstest::rstest]
#[case(LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_1)]
#[case(LanceFileVersion::V2_2)]
#[case(LanceFileVersion::V2_3)]
#[tokio::test]
async fn test_binary_copy_preserves_stable_row_ids(#[case] version: LanceFileVersion) {
    do_binary_copy_preserves_stable_row_ids(version).await;
}

async fn do_binary_copy_preserves_stable_row_ids(version: LanceFileVersion) {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(8).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(1_024),
        format!("memory://test/binary_copy_stable_row_ids_{}", version).as_str(),
        Some(WriteParams {
            enable_stable_row_ids: true,
            data_storage_version: Some(version),
            max_rows_per_file: 256,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vector".into()),
            &params,
            false,
        )
        .await
        .unwrap();

    async fn index_set(dataset: &Dataset) -> HashSet<Uuid> {
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.uuid)
            .collect()
    }
    let indices = index_set(&dataset).await;

    async fn vector_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        let query = Float32Array::from(vec![0.0f32; 8]);
        scanner
            .nearest("vec", &query, 10)
            .unwrap()
            .project(&["i"])
            .unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    async fn scalar_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    let before_vec_result = vector_query(&dataset).await;
    let before_scalar_result = scalar_query(&dataset).await;

    let before_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 512,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let current_indices = index_set(&dataset).await;
    assert_eq!(indices, current_indices);

    let after_vec_result = vector_query(&dataset).await;
    assert_eq!(before_vec_result, after_vec_result);

    let after_scalar_result = scalar_query(&dataset).await;
    assert_eq!(before_scalar_result, after_scalar_result);

    let after_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let before_idx = arrow_ord::sort::sort_to_indices(
        before_batch.column_by_name(lance_core::ROW_ID).unwrap(),
        None,
        None,
    )
    .unwrap();
    let after_idx = arrow_ord::sort::sort_to_indices(
        after_batch.column_by_name(lance_core::ROW_ID).unwrap(),
        None,
        None,
    )
    .unwrap();
    let before = arrow::compute::take_record_batch(&before_batch, &before_idx).unwrap();
    let after = arrow::compute::take_record_batch(&after_batch, &after_idx).unwrap();

    assert_eq!(before, after);
}

#[rstest::rstest]
#[case(LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_1)]
#[case(LanceFileVersion::V2_2)]
#[case(LanceFileVersion::V2_3)]
#[tokio::test]
async fn test_binary_copy_remaps_unstable_row_ids(#[case] version: LanceFileVersion) {
    do_binary_copy_remaps_unstable_row_ids(version).await;
}

async fn do_binary_copy_remaps_unstable_row_ids(version: LanceFileVersion) {
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(8).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(1_024),
        format!("memory://test/binary_copy_no_stable_{version}").as_str(),
        Some(WriteParams {
            enable_stable_row_ids: false,
            data_storage_version: Some(version),
            max_rows_per_file: 256,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vector".into()),
            &params,
            false,
        )
        .await
        .unwrap();

    async fn vector_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        let query = Float32Array::from(vec![0.0f32; 8]);
        scanner
            .nearest("vec", &query, 10)
            .unwrap()
            .project(&["i"])
            .unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    async fn scalar_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();
        scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    let before_vec_result = vector_query(&dataset).await;
    let before_scalar_result = scalar_query(&dataset).await;
    let before_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 512,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after_vec_result = vector_query(&dataset).await;
    assert_eq!(before_vec_result, after_vec_result);

    let after_scalar_result = scalar_query(&dataset).await;
    assert_eq!(before_scalar_result, after_scalar_result);

    let after_batch = dataset
        .scan()
        .project(&["vec", "i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before_batch, after_batch);
}

#[tokio::test]
async fn test_binary_copy_preserves_zonemap_queries() {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    let mut data_gen = BatchGenerator::new()
        .col(Box::new(IncrementingInt32::new().named("a".to_owned())))
        .col(Box::new(IncrementingInt32::new().named("b".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(5_000),
        "memory://test/binary_copy_zonemap",
        Some(WriteParams {
            max_rows_per_file: 500,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let zonemap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(
            &["a"],
            IndexType::Scalar,
            Some("zonemap".into()),
            &zonemap_params,
            false,
        )
        .await
        .unwrap();

    let predicate = "a >= 2500 AND b < 4000";
    let before = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_preserves_bloom_filter_queries() {
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

    let mut data_gen = BatchGenerator::new()
        .col(Box::new(IncrementingInt32::new().named("id".to_owned())))
        .col(Box::new(IncrementingInt32::new().named("val".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(6_000),
        "memory://test/binary_copy_bloom",
        Some(WriteParams {
            max_rows_per_file: 500,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    #[derive(serde::Serialize)]
    struct BloomParams {
        number_of_items: u64,
        probability: f64,
    }
    let bloom_params =
        ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter).with_params(&BloomParams {
            number_of_items: 500,
            probability: 0.01,
        });
    dataset
        .create_index(
            &["val"],
            IndexType::Scalar,
            Some("bloom".into()),
            &bloom_params,
            false,
        )
        .await
        .unwrap();

    let predicate = "val IN (123, 124, 125, 126)";
    let before = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(before, after);
}

#[tokio::test]
async fn test_binary_copy_fallback_to_common_compaction() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 500,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    dataset.delete("a < 100").await.unwrap();

    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };

    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(!can_use_binary_copy(&dataset, &options, &frags).await);

    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
}

#[rstest]
#[case::unstable_row_ids(false)]
#[case::stable_row_ids(true)]
#[tokio::test]
async fn test_try_binary_copy_falls_back_for_file_less_fragment(
    #[case] enable_stable_row_ids: bool,
) {
    let test_dir = TempStrDir::default();
    let initial_batch = arrow_array::record_batch!(("a", Int32, [1, 2])).unwrap();
    let initial_schema = initial_batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_batch)], initial_schema),
        &test_dir,
        Some(WriteParams {
            enable_stable_row_ids,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Put b in its own file so dropping a leaves the first fragment with an
    // exact schema mapping and no extra physical columns.
    let added_batch = arrow_array::record_batch!(("b", Int32, [Some(10), Some(20)])).unwrap();
    let added_schema = added_batch.schema();
    assert!(added_schema.field_with_name("b").unwrap().is_nullable());
    dataset
        .add_columns(
            NewColumnTransform::Reader(Box::new(RecordBatchIterator::new(
                vec![Ok(added_batch)],
                added_schema,
            ))),
            None,
            None,
        )
        .await
        .unwrap();

    let append_batch = arrow_array::record_batch!(("a", Int32, [3, 4])).unwrap();
    let append_schema = append_batch.schema();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(append_batch)], append_schema),
            Some(WriteParams {
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset.drop_columns(&["a"]).await.unwrap();

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(fragments.len(), 2);
    assert!(!fragments[0].files.is_empty());
    assert!(fragments[1].files.is_empty());

    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(can_use_binary_copy(&dataset, &options, &fragments[..1]).await);
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);
    let error = compact_files(
        &mut dataset,
        CompactionOptions {
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..options.clone()
        },
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("binary copy is not supported"));

    let before = dataset.scan().try_into_batch().await.unwrap();
    let expected =
        arrow_array::record_batch!(("b", Int32, [Some(10), Some(20), None, None])).unwrap();
    assert_eq!(before, expected);
    let row_ids_before = if enable_stable_row_ids {
        Some(dataset.scan().with_row_id().try_into_batch().await.unwrap())
    } else {
        None
    };

    compact_files(&mut dataset, options, None).await.unwrap();

    dataset.validate().await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    let dataset = Dataset::open(&test_dir).await.unwrap();
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(after, expected);
    if let Some(row_ids_before) = row_ids_before {
        let row_ids_after = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        assert_eq!(row_ids_after, row_ids_before);
    }
}

#[tokio::test]
async fn test_try_binary_copy_falls_back_after_all_null_column_addition() {
    let test_dir = TempStrDir::default();
    let batch = arrow_array::record_batch!(("a", Int32, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![Field::new(
                "nulls",
                DataType::Int32,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(fragments.len(), 2);
    assert!(fragments.iter().all(|fragment| fragment.files.len() == 1));
    assert!(
        fragments
            .iter()
            .flat_map(|fragment| &fragment.files)
            .all(|file| file.fields.len() == 1)
    );

    let expected = arrow_array::record_batch!(
        ("a", Int32, [1, 2, 3, 4]),
        (
            "nulls",
            Int32,
            [None::<i32>, None::<i32>, None::<i32>, None::<i32>]
        )
    )
    .unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);

    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(dataset.get_fragments().len(), 1);
}

#[tokio::test]
async fn test_try_binary_copy_falls_back_after_physical_column_drop() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_two_column_schema_evolution_dataset(&test_dir).await;
    dataset.drop_columns(&["a"]).await.unwrap();

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(fragments.len(), 2);
    assert!(fragments.iter().all(|fragment| !fragment.files.is_empty()));

    let expected = arrow_array::record_batch!(("b", Int32, [10, 20, 30, 40])).unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);

    let force_options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let error = compact_files(&mut dataset, force_options, None)
        .await
        .unwrap_err();
    assert!(matches!(&error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("binary copy is not supported"));

    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(dataset.get_fragments().len(), 1);
}

#[tokio::test]
async fn test_can_use_binary_copy_rejects_extra_physical_columns() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_two_column_schema_evolution_dataset(&test_dir).await;
    dataset.drop_columns(&["b"]).await.unwrap();
    let mut fragments = dataset.manifest.fragments.as_ref().clone();
    let version = dataset.manifest.data_storage_format.lance_file_format();
    let (fields, column_indices) =
        lance_file::versions::data_file_columns(version, dataset.schema());
    for fragment in &mut fragments {
        // Even a mapping that only lists the surviving column cannot make
        // binary copy discard the extra physical column in the source file.
        fragment.files[0].fields = fields.clone().into();
        fragment.files[0].column_indices = column_indices.clone().into();
    }
    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);
}

#[tokio::test]
async fn test_try_binary_copy_rejects_same_count_schema_replacement() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_two_column_schema_evolution_dataset(&test_dir).await;
    dataset.drop_columns(&["a"]).await.unwrap();
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![Field::new(
                "c",
                DataType::Int32,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    let current_field_ids = dataset.schema().field_ids();
    for file in fragments.iter().flat_map(|fragment| &fragment.files) {
        assert_eq!(file.fields.len(), current_field_ids.len());
        assert_ne!(file.fields.as_ref(), current_field_ids.as_slice());
    }

    let expected = arrow_array::record_batch!(
        ("b", Int32, [10, 20, 30, 40]),
        (
            "c",
            Int32,
            [None::<i32>, None::<i32>, None::<i32>, None::<i32>]
        )
    )
    .unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);

    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    assert_eq!(dataset.get_fragments().len(), 1);
}

#[tokio::test]
async fn test_try_binary_copy_materializes_data_overlay() {
    let test_dir = TempStrDir::default();
    let dataset = create_base_dataset(&test_dir).await;
    let mut dataset = commit_overlay(
        dataset,
        0,
        &[1],
        OverlayCoverage::dense(bitmap([1, 4])),
        vec![i32_array([Some(111), None])],
    )
    .await;

    let fragments: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert_eq!(fragments[0].overlays.len(), 1);
    let before = id_val_map(&dataset).await;
    assert_eq!(before.get(&1), Some(&Some(111)));
    assert_eq!(before.get(&4), Some(&None));

    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    assert!(!can_use_binary_copy(&dataset, &options, &fragments).await);

    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(id_val_map(&dataset).await, before);
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().overlays.is_empty())
    );
}

#[tokio::test]
async fn test_can_use_binary_copy_schema_consistency_ok() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader1 = RecordBatchIterator::new(vec![Ok(data.slice(0, 5_000))], data.schema());
    let reader2 = RecordBatchIterator::new(vec![Ok(data.slice(5_000, 5_000))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader1, test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    dataset.append(reader2, Some(write_params)).await.unwrap();

    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };
    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(can_use_binary_copy(&dataset, &options, &frags).await);
}

#[tokio::test]
async fn test_can_use_binary_copy_schema_mismatch() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    let mut frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    // Introduce a column index mismatch in the first data file
    if let Some(df) = frags.get_mut(0).and_then(|f| f.files.get_mut(0)) {
        let mut indices = df.column_indices.to_vec();
        if let Some(first) = indices.get_mut(0) {
            *first = -*first - 1;
        } else {
            indices.push(-1);
        }
        df.column_indices = indices.into();
    }
    assert!(!can_use_binary_copy(&dataset, &options, &frags).await);

    // Also introduce a version mismatch and ensure rejection
    if let Some(df) = frags.get_mut(0).and_then(|f| f.files.get_mut(0)) {
        df.file_minor_version = if df.file_minor_version == 1 { 2 } else { 1 };
    }
    assert!(!can_use_binary_copy(&dataset, &options, &frags).await);
}

#[tokio::test]
async fn test_can_use_binary_copy_version_mismatch() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 500,
        data_storage_version: Some(LanceFileVersion::V2_0),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Append additional data and then mark its files as a newer format version (v2.1).
    let reader_append = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    dataset.append(reader_append, None).await.unwrap();

    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    let mut frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(
        frags.len() >= 2,
        "expected multiple fragments for version mismatch test"
    );

    // Simulate mixed file versions by marking the second fragment as v2.1.
    let (v21_major, v21_minor) =
        lance_file::version::ConcreteFileVersion::V2_1.to_data_file_numbers();
    for file in &mut frags[1].files {
        file.file_major_version = v21_major;
        file.file_minor_version = v21_minor;
    }

    assert!(!can_use_binary_copy(&dataset, &options, &frags).await);
}

#[tokio::test]
async fn test_can_use_binary_copy_reject_deletions() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    dataset.delete("a < 10").await.unwrap();

    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    let frags: Vec<Fragment> = dataset
        .get_fragments()
        .into_iter()
        .map(Into::into)
        .collect();
    assert!(!can_use_binary_copy(&dataset, &options, &frags).await);
}

#[rstest::rstest]
#[case(LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_1)]
#[case(LanceFileVersion::V2_2)]
#[case(LanceFileVersion::V2_3)]
#[tokio::test]
async fn test_binary_copy_compaction_with_complex_schema(#[case] version: LanceFileVersion) {
    do_test_binary_copy_compaction_with_complex_schema(version).await;
}

#[test]
fn test_binary_copy_complex_schema_covers_every_non_legacy_version() {
    assert_eq!(
        [
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::V2_3
        ],
        NON_LEGACY_VERSIONS
    );
}

async fn do_test_binary_copy_compaction_with_complex_schema(version: LanceFileVersion) {
    use arrow_schema::{DataType, Field, Fields, TimeUnit};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};

    let row_num: u64 = 1_000;
    let batches: u32 = 10;
    const NUM_FRAGMENTS: usize = 100;
    let rows_per_file = (row_num as usize * batches as usize) / NUM_FRAGMENTS;

    let inner_fields = Fields::from(vec![
        Field::new("x", DataType::UInt32, true),
        Field::new("y", DataType::LargeUtf8, true),
    ]);
    let nested_fields = Fields::from(vec![
        Field::new("inner", DataType::Struct(inner_fields.clone()), true),
        Field::new("fsb", DataType::FixedSizeBinary(16), true),
        Field::new("bin", DataType::Binary, true),
    ]);
    let event_fields = Fields::from(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("payload", DataType::Binary, true),
    ]);

    let reader_full = gen_batch()
        .col("vec1", array::rand_vec::<Float32Type>(Dimension::from(12)))
        .col("vec2", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .col("i32", array::step::<Int32Type>())
        .col("i64", array::step::<Int64Type>())
        .col("f32", array::rand::<Float32Type>())
        .col("f64", array::rand::<Float64Type>())
        .col("bool", array::cycle_bool(vec![false, true]))
        .col("date32", array::rand_date32())
        .col("date64", array::rand_date64())
        .col(
            "ts_ms",
            array::rand_timestamp(&DataType::Timestamp(TimeUnit::Millisecond, None)),
        )
        .col(
            "utf8",
            array::rand_utf8(lance_datagen::ByteCount::from(16), false),
        )
        .col("large_utf8", array::random_sentence(1, 6, true))
        .col(
            "bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), false),
        )
        .col(
            "large_bin",
            array::rand_fixedbin(lance_datagen::ByteCount::from(24), true),
        )
        .col(
            "varbin",
            array::rand_varbin(
                lance_datagen::ByteCount::from(8),
                lance_datagen::ByteCount::from(32),
            ),
        )
        .col("fsb16", array::rand_fsb(16))
        .col(
            "fsl4",
            array::cycle_vec(array::rand::<Float32Type>(), Dimension::from(4)),
        )
        .col("struct_simple", array::rand_struct(inner_fields.clone()))
        .col("struct_nested", array::rand_struct(nested_fields))
        .col(
            "events",
            array::rand_list_any(array::rand_struct(event_fields.clone()), true),
        )
        .into_reader_rows(RowCount::from(row_num), BatchCount::from(batches));

    let full_dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        reader_full,
        &*full_dir,
        Some(WriteParams {
            enable_stable_row_ids: true,
            data_storage_version: Some(version),
            max_rows_per_file: rows_per_file,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        dataset.get_fragments().len(),
        NUM_FRAGMENTS,
        "compaction must have many input fragments to merge"
    );

    let opt_full = CompactionOptions {
        compaction_mode: Some(CompactionMode::Reencode),
        ..Default::default()
    };
    let opt_binary = CompactionOptions {
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    };

    let _ = compact_files(&mut dataset, opt_full, None).await.unwrap();
    let before = dataset.count_rows(None).await.unwrap();
    let batch_before = dataset.scan().try_into_batch().await.unwrap();

    let mut dataset = dataset.checkout_version(1).await.unwrap();

    // rollback and trigger another binary copy compaction
    dataset.restore().await.unwrap();
    let _ = compact_files(&mut dataset, opt_binary, None).await.unwrap();
    let after = dataset.count_rows(None).await.unwrap();
    let batch_after = dataset.scan().try_into_batch().await.unwrap();

    assert_eq!(before, after);
    assert_eq!(batch_before, batch_after);
}
