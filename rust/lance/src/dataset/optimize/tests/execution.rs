// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the execution contract area.

use super::*;
use lance_core::datatypes::Field as LanceField;

#[tokio::test]
async fn test_compact_v2_0_file_missing_evolved_struct_child() {
    let old_child = Arc::new(Field::new("old", DataType::Int32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "values",
            DataType::Struct(vec![old_child.clone()].into()),
            true,
        ),
        Field::new("id", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StructArray::from(vec![(
                old_child,
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as ArrayRef,
            )])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Model a snapshot whose nullable struct retained its field ID while its
    // old child was replaced by a later-added child. Existing V2.0 files
    // still record the struct header and old child IDs.
    let mut evolved_schema = dataset.schema().clone();
    let parent_id = evolved_schema.fields[0].id;
    let mut later_child = LanceField::new_arrow("later", DataType::Int32, true).unwrap();
    later_child.id = evolved_schema.max_field_id().unwrap() + 1;
    later_child.parent_id = parent_id;
    evolved_schema.fields[0].children = vec![later_child];
    let read_version = dataset.version().version;
    dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::Project {
            schema: evolved_schema,
            preserves_nullability: true,
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 4);
    assert_eq!(metrics.fragments_added, 1);

    let compacted = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(compacted.num_rows(), 4);
    assert_eq!(
        compacted["id"].as_ref(),
        &Int32Array::from(vec![0, 1, 2, 3])
    );
    let values = compacted["values"].as_struct();
    assert_eq!(values.null_count(), 0);
    assert_eq!(values.column_by_name("later").unwrap().null_count(), 4);
}
#[rstest]
#[tokio::test]
async fn test_compact_all_good(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Compact a table with nothing to do
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    // Just one file
    let write_params = WriteParams {
        max_rows_per_file: 10_000,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // There's only one file, so we can't compact any more if we wanted to.
    let plan = plan_compaction(&dataset, &CompactionOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.tasks().len(), 0);

    // Now split across multiple files
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 3_000,
        max_rows_per_group: 1_000,
        data_storage_version: Some(data_storage_version),
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 3_000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 0);
}
/// Compaction is a read->write roundtrip and the decoder materializes the
/// children of null structs as nulls. A dataset containing null structs
/// with non-nullable children (e.g. GeoArrow point columns, where the
/// children must be non-nullable and validity is carried by the struct)
/// must therefore stay compactable.
///
/// Reproduces <https://github.com/lance-format/lance/issues/7826>
///
/// 2.1+ only: the 2.0 encoding does not store struct validity, so masked
/// child nulls stay rejected at write time there.
#[tokio::test]
async fn test_compact_null_structs_with_non_nullable_children() {
    let data_storage_version = LanceFileVersion::V2_1;
    use arrow_array::{Float64Array, StructArray, cast::AsArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::Fields;

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let child_fields = Fields::from(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("position", DataType::Struct(child_fields.clone()), true),
    ]));

    // Every third row is a null struct; the children physically hold 0.0
    // fillers at those slots ("validity-first" layout).
    let num_rows = 100_usize;
    let ids = Int64Array::from_iter_values(0..num_rows as i64);
    let is_valid = |i: usize| !i.is_multiple_of(3);
    let xs = Float64Array::from_iter_values(
        (0..num_rows).map(|i| if is_valid(i) { i as f64 } else { 0.0 }),
    );
    let ys = Float64Array::from_iter_values(
        (0..num_rows).map(|i| if is_valid(i) { i as f64 * 10.0 } else { 0.0 }),
    );
    let position = StructArray::try_new(
        child_fields,
        vec![Arc::new(xs) as ArrayRef, Arc::new(ys) as ArrayRef],
        Some(NullBuffer::from_iter((0..num_rows).map(is_valid))),
    )
    .unwrap();
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids) as ArrayRef, Arc::new(position) as ArrayRef],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 25,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 4);

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 4);
    assert_eq!(metrics.fragments_added, 1);

    // The rewritten data is intact: null structs stay null and valid rows
    // keep their values.
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), num_rows);
    let ids = batch.column_by_name("id").unwrap();
    let ids = ids.as_primitive::<Int64Type>();
    let position = batch.column_by_name("position").unwrap().as_struct();
    let xs = position.column(0).as_primitive::<Float64Type>();
    let ys = position.column(1).as_primitive::<Float64Type>();
    for row in 0..num_rows {
        let i = ids.value(row) as usize;
        assert_eq!(position.is_valid(row), is_valid(i));
        if is_valid(i) {
            assert_eq!((xs.value(row), ys.value(row)), (i as f64, i as f64 * 10.0));
        }
    }
}
#[rstest]
#[tokio::test]
async fn test_compact_many(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();

    // Create a table with 3 small fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 400,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Append 2 large fragments (1k rows)
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1000,
        data_storage_version: Some(data_storage_version),
        mode: WriteMode::Append,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Delete 1 row from first large fragment
    dataset.delete("a = 1300").await.unwrap();

    // Delete 20% of rows from second large fragment
    dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

    // Append 2 small fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 300,
        data_storage_version: Some(data_storage_version),
        mode: WriteMode::Append,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let first_new_frag_idx = 7;
    // The tasks execute concurrently, so either one may reserve the first
    // output fragment id.
    let remap_a = expect_remap(
        &[
            vec![
                // 3 small fragments are rewritten to frag 7
                (row_addrs(0, 0..400), true),
                (row_addrs(1, 0..400), true),
                (row_addrs(2, 0..400), true),
            ],
            // frag 3 is skipped since it does not have enough missing data
            // Frags 4, 5, and 6 are rewritten to frag 8
            vec![
                (row_addrs(4, 0..200), true),
                (row_addrs(4, 200..400), false),
                (row_addrs(4, 400..1000), true),
                (row_addrs(5, 0..300), true),
                (row_addrs(6, 0..300), true),
            ],
        ],
        first_new_frag_idx,
    );
    let remap_b = expect_remap(
        &[
            // Frags 4, 5, and 6 are rewritten to frag 7
            vec![
                (row_addrs(4, 0..200), true),
                (row_addrs(4, 200..400), false),
                (row_addrs(4, 400..1000), true),
                (row_addrs(5, 0..300), true),
                (row_addrs(6, 0..300), true),
            ],
            // 3 small fragments rewritten to frag 8
            vec![
                (row_addrs(0, 0..400), true),
                (row_addrs(1, 0..400), true),
                (row_addrs(2, 0..400), true),
            ],
        ],
        first_new_frag_idx,
    );

    // Create compaction plan
    let options = CompactionOptions {
        target_rows_per_fragment: 1000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 2);
    assert_eq!(plan.tasks()[0].fragments.len(), 3);
    assert_eq!(plan.tasks()[1].fragments.len(), 3);

    assert_eq!(
        plan.tasks()[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        plan.tasks()[1]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );

    let mock_remapper = MockIndexRemapper::in_any_order(&[remap_a, remap_b]);

    // Run compaction
    let metrics = compact_files(&mut dataset, options, Some(Arc::new(mock_remapper)))
        .await
        .unwrap();

    // Assert on metrics
    assert_eq!(metrics.fragments_removed, 6);
    assert_eq!(metrics.fragments_added, 2);
    assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
    assert_eq!(metrics.files_added, 2);

    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id())
        .collect::<Vec<_>>();
    assert_eq!(fragment_ids, vec![3, 7, 8]);
}
#[rstest]
#[tokio::test]
async fn test_compaction_does_not_strand_small_remainders(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let data = sample_data().slice(0, 2_000);
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 200,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 500,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();

    assert_eq!(metrics.fragments_removed, 10);
    assert_eq!(metrics.fragments_added, 3);
    let mut fragment_sizes = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.metadata.physical_rows.unwrap())
        .collect::<Vec<_>>();
    fragment_sizes.sort_unstable();
    assert_eq!(fragment_sizes, vec![600, 600, 800]);

    let second_metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(second_metrics, CompactionMetrics::default());
}
#[rstest]
#[case::legacy(LanceFileVersion::Legacy)]
#[case::stable(LanceFileVersion::Stable)]
#[tokio::test]
async fn test_compaction_rebalances_oversized_task(#[case] data_storage_version: LanceFileVersion) {
    let test_dir = TempStrDir::default();
    let data = sample_data().slice(0, 5_100);
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 5_000))], data.schema());
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 5_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(5_000, 100))], data.schema());
    dataset.append(reader, None).await.unwrap();

    dataset.delete("a < 1000").await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].fragments.len(), 2);

    let metrics = compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 4);
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.metadata.physical_rows.unwrap())
            .collect::<Vec<_>>(),
        vec![1_025; 4]
    );

    let second_metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(second_metrics, CompactionMetrics::default());
}
#[rstest]
#[tokio::test]
async fn test_compact_data_files(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();

    // Create a table with 2 small fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 5_000,
        max_rows_per_group: 1_000,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    // Add a column
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("x", DataType::Float32, false),
    ]);

    let data = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..10_000)),
            Arc::new(Float32Array::from_iter_values(
                (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
            )),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

    dataset.merge(reader, "a", "a").await.unwrap();

    let expected_remap = expect_remap(
        &[vec![
            // 3 small fragments are rewritten entirely
            (row_addrs(0, 0..5000), true),
            (row_addrs(1, 0..5000), true),
        ]],
        2,
    );

    let plan = plan_compaction(
        &dataset,
        &CompactionOptions {
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(plan.tasks().len(), 1);
    assert_eq!(plan.tasks()[0].fragments.len(), 2);

    let metrics = compact_files(&mut dataset, plan.options, Some(Arc::new(expected_remap)))
        .await
        .unwrap();

    assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
    assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);

    // Assert order unchanged and data is all there.
    let scanner = dataset.scan();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

    assert_eq!(scanned_data, data);
}
#[rstest]
#[tokio::test]
async fn test_compact_with_io_buffer_size(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Compaction should succeed and produce correct results when an
    // explicit io_buffer_size is provided via CompactionOptions.
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();

    // Create a table with 2 small fragments so there is something to compact.
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 5_000,
        max_rows_per_group: 1_000,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let options = CompactionOptions {
        // A generous buffer so the read does not deadlock on large batches.
        io_buffer_size: Some(256 * 1024 * 1024),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1);

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);

    // All rows are preserved after compaction.
    let scanner = dataset.scan();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();
    assert_eq!(scanned_data.num_rows(), data.num_rows());
}
#[rstest]
#[case::default_cache(None, 1)]
// 1k-row Int64 batches (8 KiB) overflow a 4 KiB cache, so each batch becomes a page.
#[case::small_cache(Some(4 * 1024), 10)]
#[tokio::test]
async fn test_compact_with_data_cache_bytes(
    #[case] data_cache_bytes: Option<u64>,
    #[case] expected_pages: usize,
) {
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 5_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let options = CompactionOptions {
        batch_size: Some(1_000),
        data_cache_bytes,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    // Tasks may run on another machine, so the option must survive serialization.
    let tasks = plan
        .compaction_tasks()
        .map(|task| {
            let json = serde_json::to_string(&task).unwrap();
            serde_json::from_str::<CompactionTask>(&json).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].options.data_cache_bytes, data_cache_bytes);

    let result = tasks[0].execute(&dataset).await.unwrap();
    commit_compaction(
        &mut dataset,
        vec![result],
        Arc::new(IgnoreRemap::default()),
        &options,
    )
    .await
    .unwrap();

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    let data_file = &fragments[0].metadata().files[0];
    let path = dataset
        .data_file_dir(data_file)
        .unwrap()
        .join(data_file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&path, &data_file.file_size_bytes)
        .await
        .unwrap();
    let metadata = LFReader::read_all_metadata(&file_scheduler).await.unwrap();
    assert_eq!(metadata.column_infos[0].page_infos.len(), expected_pages);
}
#[rstest]
#[tokio::test]
async fn test_compact_deletions(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // For files that have few rows, we don't want to compact just 1 since
    // that won't do anything. But if there are deletions to materialize,
    // we want to do groups of 1. This test checks that.
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();

    // Create a table with 1 fragment
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1000))], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 1000,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    dataset.delete("a <= 500").await.unwrap();

    // Threshold must be satisfied
    let mut options = CompactionOptions {
        materialize_deletions_threshold: 0.8,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 0);

    // Ignore deletions if materialize_deletions is false
    options.materialize_deletions_threshold = 0.1;
    options.materialize_deletions = false;
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 0);

    // Materialize deletions if threshold is met
    options.materialize_deletions = true;
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1);

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.files_removed, 2);
    assert_eq!(metrics.fragments_added, 1);

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    assert!(fragments[0].metadata.deletion_file.is_none());
}
