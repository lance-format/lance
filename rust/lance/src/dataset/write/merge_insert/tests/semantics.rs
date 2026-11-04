// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_explain_plan() {
    // Set up test data using lance_datagen
    let dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col("name", array::cycle_utf8_literals(&["a", "b", "c"]))
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(3))
        .await
        .unwrap();

    // Create merge insert job
    let merge_insert_job =
        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

    // Test explain_plan with default schema (None)
    let plan = merge_insert_job.explain_plan(None, false).await.unwrap();

    // Also validate the full string structure with pattern matching
    let expected_pattern = "\
MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep...
  CoalescePartitionsExec...
    HashJoinExec...
      LanceRead...
      StreamingTableExec: partition_sizes=1, projection=[id, name]";
    assert_string_matches(&plan, expected_pattern).unwrap();

    // Test with explicit schema
    let source_schema = arrow_schema::Schema::from(dataset.schema());
    let explicit_plan = merge_insert_job
        .explain_plan(Some(&source_schema), false)
        .await
        .unwrap();
    assert_eq!(plan, explicit_plan); // Should be the same as default

    // Test verbose mode produces different (likely longer) output
    let verbose_plan = merge_insert_job.explain_plan(None, true).await.unwrap();
    assert!(verbose_plan.contains("MergeInsert"));
    // Verbose should also match the expected pattern
    assert_string_matches(&verbose_plan, expected_pattern).unwrap();
}

/// Asserts that `explain_plan()` is supported for a default find-or-create
/// configuration (`WhenMatched::DoNothing` + `WhenNotMatched::InsertAll`).
/// Before lance-format/lance#6441 this returned `Error::NotSupported`
/// because the job fell back to the legacy v1 path.
#[tokio::test]
async fn test_explain_plan_find_or_create() {
    let dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col("name", array::cycle_utf8_literals(&["a", "b", "c"]))
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(3))
        .await
        .unwrap();

    // Default builder config == find-or-create.
    let merge_insert_job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .try_build()
        .unwrap();

    let plan = merge_insert_job.explain_plan(None, false).await.unwrap();

    let expected_pattern = "\
MergeInsert: on=[id], when_matched=DoNothing, when_not_matched=InsertAll, when_not_matched_by_source=Keep...
  CoalescePartitionsExec...
    HashJoinExec...join_type=Right...
      LanceRead...
      StreamingTableExec: partition_sizes=1, projection=[id, name]";
    assert_string_matches(&plan, expected_pattern).unwrap();
}

#[tokio::test]
async fn test_explain_plan_full_schema_delete_by_source_with_fsl() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let dataset_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![
                        1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3,
                    ]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let dataset = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(dataset_batch)],
            schema.clone(),
        )),
        "memory://test_explain_plan_full_schema_delete_by_source_with_fsl",
        None,
    )
    .await
    .unwrap();

    let merge_insert_job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .use_index(false)
        .try_build()
        .unwrap();

    let plan = merge_insert_job.explain_plan(None, false).await.unwrap();
    assert!(plan.contains("HashJoinExec"));
    assert!(plan.contains("join_type=Full"));
    assert!(
        plan.lines().any(|line| line.contains("HashJoinExec")
            && line.contains("projection=[")
            && line.contains("_rowid")),
        "join should push down a projection that retains _rowid: {plan}"
    );
    assert!(
        plan.contains("LanceRead: uri=") && plan.contains("projection=[id]"),
        "target-side scan should prune the FSL payload from the join build side: {plan}"
    );
    assert!(
        !plan.contains("LanceRead: uri=test_explain_plan_full_schema_delete_by_source_with_fsl/data, projection=[id, vec]"),
        "target-side scan should not include the FSL payload in the join build side: {plan}"
    );
}

#[tokio::test]
async fn test_explain_plan_full_schema_delete_by_source_with_fsl_and_scalar_index() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let dataset_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![
                        1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3,
                    ]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let mut dataset = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(dataset_batch)],
            schema.clone(),
        )),
        "memory://test_explain_plan_full_schema_delete_by_source_with_fsl_and_scalar_index",
        None,
    )
    .await
    .unwrap();

    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(&["id"], IndexType::Scalar, None, &scalar_params, false)
        .await
        .unwrap();

    let merge_insert_job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();

    let plan = merge_insert_job.explain_plan(None, false).await.unwrap();
    assert!(plan.contains("HashJoinExec"));
    assert!(plan.contains("join_type=Full"));
    assert!(
        plan.lines().any(|line| line.contains("HashJoinExec")
            && line.contains("projection=[")
            && line.contains("_rowid")),
        "join should push down a projection that retains _rowid: {plan}"
    );
    assert!(
        plan.contains("LanceRead: uri=") && plan.contains("projection=[id]"),
        "target-side scan should prune the FSL payload from the join build side even when a scalar index exists: {plan}"
    );
    assert!(
        !plan.contains(
            "LanceRead: uri=test_explain_plan_full_schema_delete_by_source_with_fsl_and_scalar_index/data, projection=[id, vec]"
        ),
        "target-side scan should not include the FSL payload in the join build side: {plan}"
    );
}

#[tokio::test]
async fn test_merge_insert_full_schema_delete_by_source_with_fsl() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let dataset_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![
                        1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3,
                    ]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let dataset = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(dataset_batch)],
            schema.clone(),
        )),
        "memory://test_merge_insert_full_schema_delete_by_source_with_fsl",
        None,
    )
    .await
    .unwrap();

    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2, 4])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![20.0, 20.1, 20.2, 20.3, 40.0, 40.1, 40.2, 40.3]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let (merged_dataset, stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source_batch)],
                schema.clone(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 2);
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);

    let merged = merged_dataset.scan().try_into_batch().await.unwrap();
    let ids = merged["id"].as_primitive::<Int32Type>().values().to_vec();
    assert_eq!(ids, vec![2, 4]);

    let vecs = merged["vec"].as_fixed_size_list();
    let actual = vecs
        .values()
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    assert_eq!(actual, vec![20.0, 20.1, 20.2, 20.3, 40.0, 40.1, 40.2, 40.3]);
}

#[tokio::test]
async fn test_merge_insert_full_schema_delete_by_source_with_fsl_and_scalar_index() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let dataset_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![
                        1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3,
                    ]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let mut dataset = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(dataset_batch)],
            schema.clone(),
        )),
        "memory://test_merge_insert_full_schema_delete_by_source_with_fsl_and_scalar_index",
        None,
    )
    .await
    .unwrap();

    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(&["id"], IndexType::Scalar, None, &scalar_params, false)
        .await
        .unwrap();

    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2, 4])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![20.0, 20.1, 20.2, 20.3, 40.0, 40.1, 40.2, 40.3]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();

    let (merged_dataset, stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source_batch)],
                schema.clone(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 2);
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);

    let merged = merged_dataset.scan().try_into_batch().await.unwrap();
    let ids = merged["id"].as_primitive::<Int32Type>().values().to_vec();
    assert_eq!(ids, vec![2, 4]);

    let vecs = merged["vec"].as_fixed_size_list();
    let actual = vecs
        .values()
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    assert_eq!(actual, vec![20.0, 20.1, 20.2, 20.3, 40.0, 40.1, 40.2, 40.3]);
}

#[tokio::test]
async fn test_analyze_plan() {
    // Set up test data using lance_datagen
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col("name", array::cycle_utf8_literals(&["a", "b", "c"]))
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(3))
        .await
        .unwrap();

    // Capture the original version before analyze_plan
    let original_version = dataset.version().version;

    // Create merge insert job
    let merge_insert_job =
        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

    // Create source data stream with exact same schema
    let schema = Arc::new(arrow_schema::Schema::from(dataset.schema()));
    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 4])), // 1 matches, 4 is new
            Arc::new(StringArray::from(vec!["updated_a", "d"])),
        ],
    )
    .unwrap();

    let source_stream = RecordBatchStreamAdapter::new(
        schema,
        futures::stream::once(async { Ok(source_batch) }).boxed(),
    );

    // Test analyze_plan. We enclose the analysis output string in brackets to make it easier
    // to use assert_string_matches.  (That function requires a known string at the beginning
    // and end.)
    let mut analysis = String::from("[");
    analysis.push_str(
        &merge_insert_job
            .analyze_plan(Box::pin(source_stream))
            .await
            .unwrap(),
    );
    analysis.push_str(&String::from("]"));

    // Verify the analysis contains expected components
    assert!(analysis.contains("MergeInsert"));
    assert!(analysis.contains("metrics"));
    // Note: AnalyzeExec is no longer in the output

    // Should show execution metrics including new write metrics
    assert!(analysis.contains("bytes_written"));
    assert!(analysis.contains("num_files_written"));

    // IMPORTANT: Verify that no new version was created
    // analyze_plan should not commit the transaction
    dataset.checkout_latest().await.unwrap();
    assert_eq!(
        dataset.version().version,
        original_version,
        "analyze_plan should not create a new dataset version"
    );

    // Also validate the full string structure with pattern matching
    let expected_pattern = "[...MergeInsert: elapsed=..., on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep, metrics=...bytes_written=...num_deleted_rows=0, num_files_written=...num_inserted_rows=1, num_skipped_duplicates=0, num_updated_rows=1]
    ...
    StreamingTableExec: partition_sizes=1, projection=[id, name], metrics=[]...]";
    assert_string_matches(&analysis, expected_pattern).unwrap();
    assert!(analysis.contains("bytes_written"));
    assert!(analysis.contains("num_files_written"));
    assert!(analysis.contains("elapsed_compute"));
}

#[tokio::test]
async fn test_merge_insert_with_action_column() {
    // Test that merge_insert works when the user has a column named "action"
    // This reproduces issue #4498

    // Create a dataset with an "action" column
    let initial_data = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("action", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("value", arrow_schema::DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["create", "update", "delete"])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();

    let tempdir = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data.clone())], initial_data.schema()),
        &tempdir,
        None,
    )
    .await
    .unwrap();

    // Create new data for merge with matching "action" column
    let new_data = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int32, false),
            arrow_schema::Field::new("action", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("value", arrow_schema::DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![2, 4])),
            Arc::new(StringArray::from(vec!["modify", "insert"])),
            Arc::new(Int32Array::from(vec![25, 40])),
        ],
    )
    .unwrap();

    // Perform merge insert - this should work despite having "action" column
    let merge_insert_job =
        MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, _) = merge_insert_job.execute(new_stream).await.unwrap();

    // Verify the merge worked correctly
    let result_batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let result_batch = concat_batches(&result_batches[0].schema(), &result_batches).unwrap();

    // Should have 4 rows: 1 (unchanged), 2 (updated), 3 (unchanged), 4 (inserted)
    assert_eq!(result_batch.num_rows(), 4);

    // Verify the "action" column values are preserved correctly
    let id_col = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let action_col = result_batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let value_col = result_batch
        .column(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    // Find each row by ID and verify
    for i in 0..result_batch.num_rows() {
        match id_col.value(i) {
            1 => {
                assert_eq!(action_col.value(i), "create");
                assert_eq!(value_col.value(i), 10);
            }
            2 => {
                assert_eq!(action_col.value(i), "modify"); // Updated
                assert_eq!(value_col.value(i), 25); // Updated
            }
            3 => {
                assert_eq!(action_col.value(i), "delete");
                assert_eq!(value_col.value(i), 30);
            }
            4 => {
                assert_eq!(action_col.value(i), "insert"); // New row
                assert_eq!(value_col.value(i), 40); // New row
            }
            _ => panic!("Unexpected id: {}", id_col.value(i)),
        }
    }
}

#[tokio::test]
#[rstest::rstest]
async fn test_duplicate_rowid_detection(
    #[values(false, true)] is_full_schema: bool,
    #[values(true, false)] enable_stable_row_ids: bool,
    #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = "memory://test_duplicate_rowid_multi_fragment.lance";

    // Create initial dataset with multiple fragments to test cross-fragment duplicate detection
    let dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::step_custom::<UInt32Type>(10, 10))
        .into_dataset_with_params(
            test_uri,
            FragmentCount(3),
            FragmentRowCount(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids,
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(dataset.get_fragments().len(), 3, "Should have 3 fragments");

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, is_full_schema),
        Field::new("value", DataType::UInt32, is_full_schema),
    ]));

    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2, 2, 6, 6, 10, 10, 15])),
            Arc::new(UInt32Array::from(vec![100, 200, 300, 400, 500, 600, 700])),
        ],
    )
    .unwrap();

    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();

    let reader = Box::new(RecordBatchIterator::new([Ok(source_batch)], schema.clone()));
    let stream = reader_to_stream(reader);

    let result = job.execute(stream).await;

    assert!(
        result.is_err(),
        "Expected merge insert to fail due to duplicate rows on key column."
    );

    assert!(
        matches!(&result, &Err(Error::InvalidInput { ref source, .. }) if source.to_string().contains("Ambiguous merge insert") && source.to_string().contains("multiple source rows")),
        "Expected error to be InvalidInput with message about ambiguous merge insert and multiple source rows, got: {:?}",
        result
    );
}

#[tokio::test]
#[rstest::rstest]
async fn test_source_dedupe_behavior_first_seen(
    #[values(false, true)] is_full_schema: bool,
    #[values(true, false)] enable_stable_row_ids: bool,
    #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
    data_storage_version: LanceFileVersion,
) {
    let test_uri = format!(
        "memory://test_dedupe_first_seen_{}_{}.lance",
        is_full_schema, enable_stable_row_ids
    );

    // Create initial dataset with keys 1, 2, 3, 4
    let dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::step_custom::<UInt32Type>(10, 10))
        .into_dataset_with_params(
            &test_uri,
            FragmentCount(1),
            FragmentRowCount(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids,
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // Initial data: key=1,value=10; key=2,value=20; key=3,value=30; key=4,value=40
    let initial_data: Vec<(u32, u32)> = dataset
        .scan()
        .try_into_batch()
        .await
        .unwrap()
        .columns()
        .iter()
        .map(|c| c.as_primitive::<UInt32Type>().values().to_vec())
        .collect::<Vec<_>>()
        .into_iter()
        .fold(Vec::new(), |mut acc, vals| {
            if acc.is_empty() {
                acc = vals.into_iter().map(|v| (v, 0)).collect();
            } else {
                for (i, v) in vals.into_iter().enumerate() {
                    acc[i].1 = v;
                }
            }
            acc
        });
    assert_eq!(
        initial_data,
        vec![(1, 10), (2, 20), (3, 30), (4, 40)],
        "Initial data should be correct"
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, is_full_schema),
        Field::new("value", DataType::UInt32, is_full_schema),
    ]));

    // Source data with duplicates:
    // - key=2 appears 3 times with values 100, 200, 300 (first seen: 100)
    // - key=3 appears 2 times with values 400, 500 (first seen: 400)
    // - key=5 is a new insert (value=600)
    // Total duplicates: 3 (2 extra for key=2, 1 extra for key=3)
    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2, 2, 2, 3, 3, 5])),
            Arc::new(UInt32Array::from(vec![100, 200, 300, 400, 500, 600])),
        ],
    )
    .unwrap();

    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .source_dedupe_behavior(SourceDedupeBehavior::FirstSeen)
        .try_build()
        .unwrap();

    let reader = Box::new(RecordBatchIterator::new([Ok(source_batch)], schema.clone()));
    let stream = reader_to_stream(reader);

    let (dataset, stats) = job.execute(stream).await.unwrap();

    // Verify stats
    assert_eq!(
        stats.num_skipped_duplicates, 3,
        "Should have skipped 3 duplicate rows (2 extra for key=2, 1 extra for key=3)"
    );
    assert_eq!(
        stats.num_updated_rows, 2,
        "Should have updated 2 rows (key=2 and key=3)"
    );
    assert_eq!(
        stats.num_inserted_rows, 1,
        "Should have inserted 1 row (key=5)"
    );

    // Verify the actual data - first seen values should be kept
    let result_batch = dataset.scan().try_into_batch().await.unwrap();
    let keys = result_batch.column(0).as_primitive::<UInt32Type>();
    let values = result_batch.column(1).as_primitive::<UInt32Type>();

    let result_data: std::collections::HashMap<u32, u32> = keys
        .values()
        .iter()
        .zip(values.values().iter())
        .map(|(&k, &v)| (k, v))
        .collect();

    assert_eq!(result_data.len(), 5, "Should have 5 rows total");
    assert_eq!(
        result_data.get(&1),
        Some(&10),
        "key=1 should be unchanged (original value)"
    );
    assert_eq!(
        result_data.get(&2),
        Some(&100),
        "key=2 should have first seen value (100, not 200 or 300)"
    );
    assert_eq!(
        result_data.get(&3),
        Some(&400),
        "key=3 should have first seen value (400, not 500)"
    );
    assert_eq!(
        result_data.get(&4),
        Some(&40),
        "key=4 should be unchanged (original value)"
    );
    assert_eq!(
        result_data.get(&5),
        Some(&600),
        "key=5 should be inserted with value 600"
    );
}

#[rstest::rstest]
#[case::v2(false)]
#[case::indexed_scan(true)]
#[tokio::test]
async fn test_first_seen_dedupes_unmatched_source_rows(#[case] with_index: bool) {
    let initial = record_batch!(("id", Int32, [Some(1)]), ("value", Int32, [Some(10)])).unwrap();
    let initial = if with_index {
        initial
    } else {
        // Match the reported failure's empty-target setup on the v2 path.
        initial.slice(0, 0)
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(initial.clone())], initial.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    if with_index {
        dataset
            .create_index(
                &["id"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }

    // Exercise enough indexed duplicates to force the source sort to
    // rearrange keys. The first 64 rows contain each key exactly once, so
    // their values are the expected FirstSeen winners after the sort.
    let (source_batches, expected_inserted, expected_updated, expected_skipped) = if with_index {
        let ids = (0..4096).map(|row| (row * 37) % 64).collect::<Vec<i32>>();
        let values = (0..4096).collect::<Vec<i32>>();
        let source = RecordBatch::try_new(
            initial.schema(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();
        let source_batches = (0..source.num_rows())
            .step_by(40)
            .map(|offset| Ok(source.slice(offset, (source.num_rows() - offset).min(40))))
            .collect();
        (source_batches, 63, 1, 4096 - 64)
    } else {
        let first = record_batch!(
            ("id", Int32, [Some(108), None]),
            ("value", Int32, [Some(1), Some(3)])
        )
        .unwrap();
        let second = record_batch!(
            ("id", Int32, [Some(108), None]),
            ("value", Int32, [Some(2), Some(4)])
        )
        .unwrap();
        (vec![Ok(first), Ok(second)], 3, 0, 1)
    };
    let source_schema = source_batches[0].as_ref().unwrap().schema();

    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .source_dedupe_behavior(SourceDedupeBehavior::FirstSeen)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            source_batches,
            source_schema,
        )))
        .await
        .unwrap();

    assert_eq!(stats.num_inserted_rows, expected_inserted);
    assert_eq!(stats.num_updated_rows, expected_updated);
    assert_eq!(stats.num_skipped_duplicates, expected_skipped);

    if with_index {
        let result = dataset.scan().try_into_batch().await.unwrap();
        let ids = result["id"].as_primitive::<Int32Type>();
        let values = result["value"].as_primitive::<Int32Type>();
        let actual = ids
            .values()
            .iter()
            .zip(values.values().iter())
            .map(|(&id, &value)| (id, value))
            .collect::<HashMap<_, _>>();
        assert_eq!(actual.len(), 64);
        for first_row in 0..64 {
            let id = (first_row * 37) % 64;
            assert_eq!(actual.get(&id), Some(&first_row));
        }
        return;
    }

    let inserted = dataset
        .scan()
        .filter("id = 108")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(inserted.num_rows(), 1);
    assert_eq!(inserted["value"].as_primitive::<Int32Type>().value(0), 1);
}

#[tokio::test]
async fn test_when_matched_fail() {
    let dataset = create_test_dataset("memory://test_fail", LanceFileVersion::V2_0, true).await;

    // Create new data with some existing keys (should fail)
    let new_data = RecordBatch::try_new(
        create_test_schema(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 2, 10, 11])), // Keys: 1,2 exist, 10,11 are new
            Arc::new(UInt32Array::from(vec![100, 200, 1000, 1100])),
            Arc::new(StringArray::from(vec!["X", "Y", "Z", "W"])),
        ],
    )
    .unwrap();

    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let new_stream = reader_to_stream(reader);

    let result = MergeInsertBuilder::try_new(dataset.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Fail)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute(new_stream)
        .await;

    // Should fail because keys 1 and 2 already exist
    match result {
        Ok((_dataset, stats)) => {
            panic!(
                "Expected merge insert to fail, but it succeeded. Stats: {:?}",
                stats
            );
        }
        Err(e) => {
            let error_msg = e.to_string();
            assert!(error_msg.contains("Merge insert failed"));
            assert!(error_msg.contains("found matching row"));
        }
    }

    // Create new data with only new keys (should succeed)
    let new_data = RecordBatch::try_new(
        create_test_schema(),
        vec![
            Arc::new(UInt32Array::from(vec![10, 11, 12])), // All new keys
            Arc::new(UInt32Array::from(vec![1000, 1100, 1200])),
            Arc::new(StringArray::from(vec!["X", "Y", "Z"])),
        ],
    )
    .unwrap();

    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let new_stream = reader_to_stream(reader);

    let (updated_dataset, stats) =
        MergeInsertBuilder::try_new(dataset.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::Fail)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute(new_stream)
            .await
            .unwrap();

    // Should succeed with 3 new rows inserted
    assert_eq!(stats.num_inserted_rows, 3);
    assert_eq!(stats.num_updated_rows, 0);
    assert_eq!(stats.num_deleted_rows, 0);

    // Verify the data was inserted correctly
    let count = updated_dataset
        .count_rows(Some("key >= 10".to_string()))
        .await
        .unwrap();
    assert_eq!(count, 3);
}

/// Test case for Issue #4654: merge_insert should handle nullable source fields
/// when target is non-nullable, as long as there are no actual null values.
///
/// This test verifies that:
/// - Dataset has non-nullable fields
/// - Source data has nullable fields BUT no actual null values
/// - merge_insert() succeeds (same behavior as insert)
#[tokio::test]
async fn test_merge_insert_permissive_nullability() {
    // Step 1: Create dataset with NON-NULLABLE schema
    let non_nullable_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false), // nullable=False
        Field::new("value", DataType::Int64, false), // nullable=False
    ]));

    let initial_data = RecordBatch::try_new(
        non_nullable_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();

    let test_uri = "memory://test_nullable_issue_4654";
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data)], non_nullable_schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();

    // Step 2: Create new data with NULLABLE schema but NO actual null values
    let nullable_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),    // nullable=True
        Field::new("value", DataType::Int64, true), // nullable=True
    ]));

    let new_data = RecordBatch::try_new(
        nullable_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![2, 4, 5])), // id=2 exists (update), 4,5 new (insert)
            Arc::new(Int64Array::from(vec![999, 400, 500])), // No nulls
        ],
    )
    .unwrap();

    // Step 3: Test merge_insert()
    let merge_result = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(new_data.clone())],
            nullable_schema.clone(),
        )))
        .await;

    assert!(
        merge_result.is_ok(),
        "merge_insert() should succeed with nullable fields but no actual nulls. \
             This is the same behavior as insert/append. Error: {:?}",
        merge_result.err()
    );

    // Step 4: Verify the results
    let (merged_dataset, stats) = merge_result.unwrap();

    // Should have: 1 updated row (id=2), 2 new rows (id=4,5)
    assert_eq!(stats.num_updated_rows, 1, "Should update 1 row (id=2)");
    assert_eq!(
        stats.num_inserted_rows, 2,
        "Should insert 2 new rows (id=4,5)"
    );

    // Total: 3 original (id=1,2,3) + 2 new (id=4,5) = 5 rows
    let count = merged_dataset.count_rows(None).await.unwrap();
    assert_eq!(count, 5, "Should have 5 total rows");

    // Verify the updated value for id=2
    let result = merged_dataset
        .scan()
        .filter("id = 2")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let batch = concat_batches(&result[0].schema(), &result).unwrap();
    assert_eq!(batch.num_rows(), 1);
    let value_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        value_array.value(0),
        999,
        "Value for id=2 should be updated to 999"
    );
}

/// Test case for Issue #4644: merge_insert should NOT skip source rows whose ON
/// columns contain NULL.
///
/// With standard SQL equality NULL != NULL, so a source row with a NULL key will
/// never match any target row.  It must therefore be treated as "not matched" and
/// inserted when `when_not_matched = InsertAll`.  The previous implementation
/// incorrectly required all ON columns to be non-null before even considering the
/// row, causing it to be silently dropped (Action::Nothing).
#[tokio::test]
async fn test_merge_insert_null_on_column_inserts() {
    // Initial dataset: one row with a NULL record_type.
    let initial_data = record_batch!(
        ("id", Int32, [0]),
        ("record_type", Utf8, [Option::<&str>::None]),
        ("value", Int32, [10])
    )
    .unwrap();

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data.clone())], initial_data.schema()),
        "memory://test_null_on_column",
        None,
    )
    .await
    .unwrap();

    // New data: a row with a different id AND a NULL record_type.
    // Because id differs (2 vs 0) no match should be found even with NULL-safe
    // semantics, so this row must be INSERTED.
    let new_data = record_batch!(
        ("id", Int32, [Some(2)]),
        ("record_type", Utf8, [Option::<&str>::None]),
        ("value", Int32, [Some(99)])
    )
    .unwrap();

    let (merged_dataset, stats) = MergeInsertBuilder::try_new(
        Arc::new(dataset),
        vec!["id".to_string(), "record_type".to_string()],
    )
    .unwrap()
    .when_matched(WhenMatched::UpdateAll)
    .when_not_matched(WhenNotMatched::InsertAll)
    .try_build()
    .unwrap()
    .execute_reader(Box::new(RecordBatchIterator::new(
        vec![Ok(new_data.clone())],
        new_data.schema(),
    )))
    .await
    .unwrap();

    // The source row (id=2, record_type=NULL) must be inserted, NOT silently skipped.
    assert_eq!(
        stats.num_inserted_rows, 1,
        "row with NULL ON column should be inserted"
    );
    assert_eq!(stats.num_updated_rows, 0, "no row should be updated");

    let count = merged_dataset.count_rows(None).await.unwrap();
    assert_eq!(
        count, 2,
        "dataset should have the original row plus the newly inserted row"
    );
}

/// Partial composite key match: the non-null part of the ON key (id) matches an
/// existing target row, but the second ON column (record_type) is NULL in the source.
/// Standard SQL equality treats NULL != NULL, so the composite key does NOT match
/// and the source row must be inserted, not updated and not silently dropped.
#[tokio::test]
async fn test_merge_insert_partial_composite_key_null() {
    // Target: one row where id=1 and record_type="A".
    let initial_data = record_batch!(
        ("id", Int32, [Some(1)]),
        ("record_type", Utf8, [Some("A")]),
        ("value", Int32, [Some(10)])
    )
    .unwrap();

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data.clone())], initial_data.schema()),
        "memory://test_partial_composite_null",
        None,
    )
    .await
    .unwrap();

    // Source: one row where id=1 (matches target) but record_type=NULL.
    // The composite key (1, NULL) does NOT match (1, "A") under standard equality,
    // so this is a "not matched" row that should be inserted.
    let new_data = record_batch!(
        ("id", Int32, [Some(1)]),
        ("record_type", Utf8, [Option::<&str>::None]),
        ("value", Int32, [Some(99)])
    )
    .unwrap();

    let (merged_dataset, stats) = MergeInsertBuilder::try_new(
        Arc::new(dataset),
        vec!["id".to_string(), "record_type".to_string()],
    )
    .unwrap()
    .when_matched(WhenMatched::UpdateAll)
    .when_not_matched(WhenNotMatched::InsertAll)
    .try_build()
    .unwrap()
    .execute_reader(Box::new(RecordBatchIterator::new(
        vec![Ok(new_data.clone())],
        new_data.schema(),
    )))
    .await
    .unwrap();

    // Source row (id=1, record_type=NULL) must be inserted, not updated and not dropped.
    assert_eq!(
        stats.num_inserted_rows, 1,
        "row with partial NULL composite key should be inserted"
    );
    assert_eq!(
        stats.num_updated_rows, 0,
        "existing (id=1, record_type=A) row must not be updated"
    );

    // Dataset: original (1, "A") row + newly inserted (1, NULL) row = 2 rows.
    let count = merged_dataset.count_rows(None).await.unwrap();
    assert_eq!(
        count, 2,
        "both the original and the new row must be present"
    );
}

/// NULL source keys must be inserted separately from NULL target keys in both
/// indexed and full-scan merges, including when every source key is NULL.
#[rstest::rstest]
#[case::no_index(None, true)]
#[case::btree(Some(IndexType::BTree), true)]
#[case::btree_full_scan(Some(IndexType::BTree), false)]
#[case::bitmap(Some(IndexType::Bitmap), true)]
#[case::bitmap_full_scan(Some(IndexType::Bitmap), false)]
#[tokio::test]
async fn test_merge_insert_null_single_on_column(
    #[case] index_type: Option<IndexType>,
    #[case] use_index: bool,
    #[values(WhenMatched::UpdateAll, WhenMatched::DoNothing)] when_matched: WhenMatched,
    #[values(false, true)] all_null_source: bool,
) {
    let initial_data = record_batch!(
        ("id", Int32, [None, Some(1), Some(2)]),
        ("value", Int32, [1, 2, 3])
    )
    .unwrap();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data.clone())], initial_data.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if let Some(index_type) = index_type {
        dataset
            .create_index(
                &["id"],
                index_type,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }

    let new_data = if all_null_source {
        record_batch!(
            ("id", Int32, [Option::<i32>::None, None]),
            ("value", Int32, [99, 50])
        )
        .unwrap()
    } else {
        record_batch!(
            ("id", Int32, [None, Some(1), Some(5)]),
            ("value", Int32, [99, 20, 50])
        )
        .unwrap()
    };

    let has_update = !all_null_source && when_matched == WhenMatched::UpdateAll;
    let (merged_dataset, stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(when_matched)
            .when_not_matched(WhenNotMatched::InsertAll)
            .use_index(use_index)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(new_data.clone())],
                new_data.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_inserted_rows, 2);
    assert_eq!(stats.num_updated_rows, u64::from(has_update));
    assert_eq!(stats.num_deleted_rows, 0);

    let actual = merged_dataset
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("value".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let inserted_key = (!all_null_source).then_some(5);
    let expected = if has_update {
        record_batch!(
            ("id", Int32, [None, Some(2), Some(1), inserted_key, None]),
            ("value", Int32, [1, 3, 20, 50, 99])
        )
        .unwrap()
    } else {
        record_batch!(
            ("id", Int32, [None, Some(1), Some(2), inserted_key, None]),
            ("value", Int32, [1, 2, 3, 50, 99])
        )
        .unwrap()
    };
    assert_eq!(actual, expected);
}

/// Test case for Issue #3634: merge_insert should provide a helpful error
/// message when a subschema with a mismatched type is provided.
#[tokio::test]
async fn test_merge_insert_subschema_invalid_type_error() {
    // Step 1: Create a dataset with a multi-column schema.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, true), // The target type is Float64.
        Field::new("extra", DataType::Utf8, true),
    ]));

    let initial_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let test_uri = "memory://test_issue_3634";
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data)], schema),
        test_uri,
        None,
    )
    .await
    .unwrap();

    // Step 2: Create source data with a subschema where one field has a wrong type.
    let subschema_with_wrong_type = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    let new_data = RecordBatch::try_new(
        subschema_with_wrong_type.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2, 4])),
            Arc::new(Int32Array::from(vec![22, 44])),
        ],
    )
    .unwrap();

    // Step 3: Execute the merge_insert operation, which should fail.
    let merge_result = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(new_data)],
            subschema_with_wrong_type,
        )))
        .await;

    // Step 4: Verify that the operation failed with the correct error type and message.
    let err = merge_result.expect_err("Merge insert should have failed but it succeeded.");
    assert!(
        matches!(err, lance_core::Error::SchemaMismatch { .. }),
        "Expected a SchemaMismatch error, but got a different error type: {:?}",
        err
    );

    let error_message = err.to_string();
    assert!(
        error_message.contains("`value` should have type double but type was int32"),
        "Error message should specify the expected (double) and actual (int32) types for 'value', but was: {}",
        error_message
    );

    assert!(
        !error_message.contains("missing="),
        "Error message should NOT complain about missing fields for a subschema check, but was: {}",
        error_message
    );
}

/// Test that merge_insert works with mixed-case column names as keys.
/// This is a regression test for the fix in assign_action.rs that wraps
/// column names in double quotes to preserve case in DataFusion expressions.
#[tokio::test]
async fn test_merge_insert_mixed_case_key() {
    // Create a schema with a mixed-case column name
    let schema = Arc::new(Schema::new(vec![
        Field::new("userId", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, true),
    ]));

    // Initial data
    let initial_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 2, 3])),
            Arc::new(UInt32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();

    // Write initial dataset
    let test_uri = "memory://test_mixed_case.lance";
    let ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_batch)], schema.clone()),
        test_uri,
        None,
    )
    .await
    .unwrap();

    // New data to merge (updates userId=2, inserts userId=4)
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2, 4])),
            Arc::new(UInt32Array::from(vec![200, 400])),
        ],
    )
    .unwrap();

    // Perform merge_insert using "userId" as the key
    let job = MergeInsertBuilder::try_new(Arc::new(ds), vec!["userId".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (merged_ds, _merge_stats) = job.execute(new_stream).await.unwrap();

    // Verify the merge succeeded
    let result = merged_ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let result_batch = concat_batches(&schema, &result).unwrap();
    assert_eq!(result_batch.num_rows(), 4); // 3 original + 1 inserted

    // Verify that userId=2 was updated to value=200
    let user_ids = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let values = result_batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();

    // Find the row with userId=2 and check its value
    for i in 0..result_batch.num_rows() {
        if user_ids.value(i) == 2 {
            assert_eq!(
                values.value(i),
                200,
                "userId=2 should have been updated to value=200"
            );
        }
    }
}

mod external_error {
    use super::*;
    use arrow_schema::{ArrowError, Field as ArrowField, Schema as ArrowSchema};
    use std::fmt;

    #[derive(Debug)]
    struct MyTestError {
        code: i32,
        details: String,
    }

    impl fmt::Display for MyTestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "MyTestError({}): {}", self.code, self.details)
        }
    }

    impl std::error::Error for MyTestError {}

    #[tokio::test]
    async fn test_merge_insert_execute_reader_preserves_error_message() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        // Create initial dataset
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Arc::new(
            Dataset::write(reader, "memory://test_merge_external", None)
                .await
                .unwrap(),
        );

        // Try merge insert with failing source
        let error_code = 789;
        let iter = std::iter::once(Err(ArrowError::ExternalError(Box::new(MyTestError {
            code: error_code,
            details: "merge insert failure".to_string(),
        }))));
        let reader = RecordBatchIterator::new(iter, schema);

        let result = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()])
            .unwrap()
            .try_build()
            .unwrap()
            .execute_reader(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
            .await;

        // The source error is routed through the merge plan, which shares it
        // across join partitions, so its concrete type is not recoverable. The
        // message must still reach the caller.
        let err = result.expect_err("expected the source error to surface");
        assert!(
            err.to_string().contains("merge insert failure"),
            "source error message should be preserved; got: {err}"
        );
    }
}

/// `execute_provider` is the canonical entry point; a `MemTable` source merges
/// the same way a stream does.
#[tokio::test]
async fn test_merge_insert_execute_provider() {
    let initial = record_batch!(("id", UInt32, [0, 1, 2]), ("value", UInt32, [0, 0, 0])).unwrap();
    let dataset = Arc::new(
        InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );

    // Update id=1, insert id=3.
    let new_data = record_batch!(("id", UInt32, [1, 3]), ("value", UInt32, [10, 30])).unwrap();
    let provider: Arc<dyn TableProvider> = Arc::new(
        datafusion::datasource::MemTable::try_new(new_data.schema(), vec![vec![new_data]]).unwrap(),
    );

    let (merged, stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_provider(provider)
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);

    let batch = merged.scan().try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<UInt32Type>();
    let values = batch["value"].as_primitive::<UInt32Type>();
    let merged_rows: HashMap<u32, u32> = ids
        .values()
        .iter()
        .zip(values.values().iter())
        .map(|(id, value)| (*id, *value))
        .collect();
    assert_eq!(
        merged_rows,
        HashMap::from([(0, 0), (1, 10), (2, 0), (3, 30)])
    );
}

/// `execute_batches` merges materialized batches; multiple batches are spread
/// across partitions and merged correctly.
#[tokio::test]
async fn test_merge_insert_execute_batches() {
    let initial = record_batch!(("id", UInt32, [0, 1, 2]), ("value", UInt32, [0, 0, 0])).unwrap();
    let dataset = Arc::new(
        InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );

    // Two batches: update id=1 (batch 0), insert id=3 (batch 1).
    let batch0 = record_batch!(("id", UInt32, [1]), ("value", UInt32, [10])).unwrap();
    let batch1 = record_batch!(("id", UInt32, [3]), ("value", UInt32, [30])).unwrap();

    let (merged, stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_batches(vec![batch0, batch1])
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);

    let batch = merged.scan().try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<UInt32Type>();
    let values = batch["value"].as_primitive::<UInt32Type>();
    let merged_rows: HashMap<u32, u32> = ids
        .values()
        .iter()
        .zip(values.values().iter())
        .map(|(id, value)| (*id, *value))
        .collect();
    assert_eq!(
        merged_rows,
        HashMap::from([(0, 0), (1, 10), (2, 0), (3, 30)])
    );
}

/// An empty batch list still produces a valid (single, empty) partition, so the
/// merge is a no-op and the target is unchanged.
#[tokio::test]
async fn test_merge_insert_execute_batches_empty() {
    let initial = record_batch!(("id", UInt32, [0, 1, 2]), ("value", UInt32, [0, 0, 0])).unwrap();
    let dataset = Arc::new(
        InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );

    let (merged, stats) = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_batches(vec![])
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, 0);
    assert_eq!(stats.num_inserted_rows, 0);

    let batch = merged.scan().try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<UInt32Type>();
    let values = batch["value"].as_primitive::<UInt32Type>();
    let merged_rows: HashMap<u32, u32> = ids
        .values()
        .iter()
        .zip(values.values().iter())
        .map(|(id, value)| (*id, *value))
        .collect();
    assert_eq!(merged_rows, HashMap::from([(0, 0), (1, 0), (2, 0)]));
}

/// With a one-shot stream source and `spill_for_retry(false)`, a commit
/// conflict fails fast instead of replaying the stream. The non-replayable
/// one-shot provider must be scanned exactly once (scanning it twice would
/// panic), proving retries are disabled even though `conflict_retries > 0`.
#[tokio::test]
async fn test_merge_insert_spill_for_retry_false_fails_fast() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, false),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(UInt32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );

    // Merge insert job based on version 1, with retries enabled but spilling off.
    let new_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100])),
            Arc::new(UInt32Array::from(vec![1])),
        ],
    )
    .unwrap();
    let job = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(10)
        .spill_for_retry(false)
        .try_build()
        .unwrap();

    // An append commits first (version 2), so the merge built on version 1 hits
    // an unresolvable conflict on commit.
    let append_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![50])),
            Arc::new(UInt32Array::from(vec![2])),
        ],
    )
    .unwrap();
    InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![append_batch])
        .await
        .unwrap();

    let source =
        RecordBatchStreamAdapter::new(schema.clone(), futures::stream::iter(vec![Ok(new_data)]));
    let merge_result = job
        .execute(Box::pin(source) as SendableRecordBatchStream)
        .await;

    assert!(
        matches!(
            merge_result,
            Err(crate::Error::TooMuchWriteContention { .. })
        ),
        "Expected fail-fast TooMuchWriteContention, got: {:?}",
        merge_result
    );
}

/// Test WhenMatched::Delete combined with deleting rows not matched by source,
/// when the source carries dataset columns besides the key. This is not
/// delete-only, so the general write exec runs and needs those columns.
#[rstest::rstest]
#[tokio::test]
async fn test_when_matched_delete_with_delete_not_matched_by_source(
    #[values(true, false)] full_source: bool,
    #[values(true, false)] delete_if: bool,
) {
    let test_uri = "memory://test_delete_with_delete_by_source.lance";

    // Create dataset with keys 1-6 (full schema: key, value, filterme)
    let ds = create_test_dataset(test_uri, LanceFileVersion::V2_0, false).await;
    // Keys 4, 5, 6 match and are deleted; 1, 2, 3 are only in the target
    let full_batch = create_new_batch(create_test_schema());
    let new_batch = if full_source {
        full_batch
    } else {
        full_batch.project(&[0, 1]).unwrap()
    };

    let when_not_matched_by_source = if delete_if {
        WhenNotMatchedBySource::delete_if(&ds, "key < 3").unwrap()
    } else {
        WhenNotMatchedBySource::Delete
    };
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(when_not_matched_by_source)
        .try_build()
        .unwrap();

    let new_schema = new_batch.schema();
    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], new_schema));
    let (merged_dataset, merge_stats) = job.execute(reader_to_stream(new_reader)).await.unwrap();

    let expected_keys: Vec<u32> = if delete_if { vec![3] } else { vec![] };
    assert_eq!(merge_stats.num_deleted_rows, 6 - expected_keys.len() as u64);
    assert_eq!(merge_stats.num_inserted_rows, 0);
    assert_eq!(merge_stats.num_updated_rows, 0);

    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let merged = concat_batches(&create_test_schema(), &batches).unwrap();
    let mut remaining_keys: Vec<u32> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    remaining_keys.sort();
    assert_eq!(remaining_keys, expected_keys);
}

/// A source column the dataset lacks is never written, so it is not carried
/// through the join even when the merge is not delete-only.
#[rstest::rstest]
#[tokio::test]
async fn test_delete_by_source_drops_source_columns_not_in_dataset(
    #[values(WhenMatched::Delete, WhenMatched::DoNothing)] when_matched: WhenMatched,
    #[values(true, false)] delete_if: bool,
) {
    let ds = create_test_dataset(
        "memory://test_delete_by_source_drops_columns.lance",
        LanceFileVersion::V2_0,
        false,
    )
    .await;
    // Keys 4, 5, 6 match; 1, 2, 3 are only in the target
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, true),
        Field::new("extra", DataType::Utf8, true),
    ]));
    let new_batch = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![4, 5, 6, 7, 8, 9])),
            Arc::new(StringArray::from(vec!["x"; 6])),
        ],
    )
    .unwrap();

    let when_not_matched_by_source = if delete_if {
        WhenNotMatchedBySource::delete_if(&ds, "key < 3").unwrap()
    } else {
        WhenNotMatchedBySource::Delete
    };
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(when_matched.clone())
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(when_not_matched_by_source)
        .try_build()
        .unwrap();

    let plan = job
        .explain_plan(Some(source_schema.as_ref()), false)
        .await
        .unwrap();
    assert!(!plan.contains("extra"), "{plan}");

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], source_schema));
    let (_, merge_stats) = job.execute(reader_to_stream(new_reader)).await.unwrap();
    let matched_deleted = if when_matched == WhenMatched::Delete {
        3
    } else {
        0
    };
    let unmatched_deleted = if delete_if { 2 } else { 3 };
    assert_eq!(
        merge_stats.num_deleted_rows,
        matched_deleted + unmatched_deleted
    );
}
