// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// An update-style merge_insert leaves the source and new fragments with
// overlapping id ranges; a scattered delete punches holes in that range. A
// filtered `with_row_id` scan must still resolve every id (round-tripped via take).
#[tokio::test(flavor = "multi_thread")]
async fn merge_insert_then_delete_resolves_overlapping_row_ids() {
    use arrow_array::ArrayRef;
    use arrow_array::types::UInt64Type;
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let schema = Arc::new(Schema::new(vec![
        Field::new("slug", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let mk = |slugs: Vec<String>, titles: Vec<String>| {
        let cats: Vec<String> = (0..slugs.len())
            .map(|i| ["A", "B", "C", "D", "E"][i % 5].to_string())
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(slugs)) as ArrayRef,
                Arc::new(StringArray::from(titles)) as ArrayRef,
                Arc::new(StringArray::from(cats)) as ArrayRef,
            ],
        )
        .unwrap()
    };

    // Empty dataset with stable row ids; write-seed 40 rows (ids 0..40).
    let params = WriteParams {
        mode: WriteMode::Create,
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(mk(vec![], vec![]))], schema.clone()),
        uri,
        Some(params),
    )
    .await
    .unwrap();
    ds.append(
        RecordBatchIterator::new(
            vec![Ok(mk(
                (1..=40).map(|i| format!("t{i}")).collect(),
                (1..=40).map(|i| format!("r{i}")).collect(),
            ))],
            schema.clone(),
        ),
        None,
    )
    .await
    .unwrap();

    // Update every other row (so the new fragment's ids interleave with -- and
    // its range overlaps -- the source's) plus a few inserts.
    let mut slugs: Vec<String> = (1..=40).step_by(2).map(|i| format!("t{i}")).collect();
    let mut titles: Vec<String> = (1..=40).step_by(2).map(|i| format!("e{i}")).collect();
    for i in 41..=45 {
        slugs.push(format!("t{i}"));
        titles.push(format!("e{i}"));
    }
    let mut b = MergeInsertBuilder::try_new(Arc::new(ds), vec!["slug".into()]).unwrap();
    b.when_matched(WhenMatched::UpdateAll);
    b.when_not_matched(WhenNotMatched::InsertAll);
    let (ds, _) = b
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new(
            vec![Ok(mk(slugs, titles))],
            schema.clone(),
        ))
        .await
        .unwrap();

    // Scattered delete -> interior holes in the overlapped id range.
    let mut ds = (*ds).clone();
    let ds = (*ds.delete("category = 'A'").await.unwrap().new_dataset).clone();

    // The `with_row_id` scan builds the RowIdIndex; every scanned id must
    // round-trip through take_rows.
    let batches: Vec<RecordBatch> = ds
        .scan()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let row_ids: Vec<u64> = batches
        .iter()
        .flat_map(|b| {
            b.column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec()
        })
        .collect();
    let scanned_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let taken = ds.take_rows(&row_ids, ds.schema().clone()).await.unwrap();
    assert_eq!(taken.num_rows(), scanned_rows);

    // A point lookup on a surviving row resolves to exactly one row.
    let filtered: Vec<RecordBatch> = ds
        .scan()
        .with_row_id()
        .filter("slug = 't30'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let n: usize = filtered.iter().map(|b| b.num_rows()).sum();
    assert_eq!(n, 1, "expected exactly one row for slug='t30'");
}

#[tokio::test]
async fn test_merge_insert_requires_on_or_primary_key() {
    let test_uri = "memory://merge_insert_requires_keys";

    let ds = create_test_dataset(test_uri, LanceFileVersion::V2_0, false).await;

    let err = MergeInsertBuilder::try_new(ds, Vec::new()).unwrap_err();
    if let crate::Error::InvalidInput { source, .. } = err {
        let msg = source.to_string();
        assert!(
            msg.contains("requires join keys") && msg.contains("primary key"),
            "unexpected error message: {}",
            msg
        );
    } else {
        panic!("expected InvalidInput error");
    }
}

#[tokio::test]
async fn test_merge_insert_defaults_to_unenforced_primary_key() {
    // Define a simple schema with an unenforced primary key on `id`.
    let id_field = Field::new("id", DataType::Int32, false).with_metadata(
        [(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        )]
        .into(),
    );
    let value_field = Field::new("value", DataType::Int32, false);
    let schema = Arc::new(Schema::new(vec![id_field, value_field]));

    let initial_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(initial_batch)], schema.clone());
    let dataset = Dataset::write(
        reader,
        "memory://merge_insert_pk_default",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let dataset = Arc::new(dataset);

    // New data: update ids 2 and 3, insert id 4.
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![2, 3, 4])),
            Arc::new(Int32Array::from(vec![200, 300, 400])),
        ],
    )
    .unwrap();

    let mut builder = MergeInsertBuilder::try_new(dataset.clone(), Vec::new()).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll);
    let job = builder.try_build().unwrap();

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (updated_dataset, stats) = job.execute(new_stream).await.unwrap();

    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(stats.num_deleted_rows, 0);

    let result_batch = updated_dataset.scan().try_into_batch().await.unwrap();
    let ids = result_batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let values = result_batch
        .column_by_name("value")
        .unwrap()
        .as_primitive::<Int32Type>();

    let mut pairs = (0..ids.len())
        .map(|i| (ids.value(i), values.value(i)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();

    assert_eq!(pairs, vec![(1, 10), (2, 200), (3, 300), (4, 400)]);
}

#[rstest::rstest]
#[case::full(false, false)]
#[case::column_patch(true, false)]
#[case::partial_column_patch(true, true)]
#[tokio::test]
async fn merge_insert_uses_explicit_exact_version(
    #[case] partial: bool,
    #[case] partial_rows: bool,
    #[values(false, true)] indexed: bool,
    #[values(
        None,
        Some(LanceFileVersion::V2_1),
        Some(LanceFileVersion::V2_2),
        Some(LanceFileVersion::V2_3)
    )]
    target: Option<LanceFileVersion>,
) {
    let mut dataset = create_test_dataset("memory://", LanceFileVersion::V2_0, false).await;
    if indexed {
        Arc::make_mut(&mut dataset)
            .create_index(
                &["key"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }
    let original_paths = dataset
        .manifest
        .fragments
        .iter()
        .flat_map(Fragment::referenced_lance_files)
        .map(|file| file.path.clone())
        .collect::<HashSet<_>>();
    let mut new_batch = create_new_batch(create_test_schema());
    if partial {
        new_batch = new_batch.project(&[0, 1]).unwrap();
    }
    if partial_rows {
        new_batch = new_batch.slice(1, new_batch.num_rows() - 1);
    }
    let schema = new_batch.schema();
    let mut builder = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()]).unwrap();
    builder.when_matched(WhenMatched::UpdateAll);
    if partial {
        builder.when_not_matched(WhenNotMatched::DoNothing);
        if !indexed {
            builder.write_mode(MergeInsertWriteMode::RewriteColumns);
        }
    }
    if let Some(target) = target {
        builder.data_storage_version(target);
    }
    let (dataset, stats) = builder
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(new_batch)], schema))
        .await
        .unwrap();

    assert_eq!(stats.num_inserted_rows, if partial { 0 } else { 3 });
    let updated_rows = if partial_rows { 2 } else { 3 };
    assert_eq!(stats.num_updated_rows, updated_rows);
    assert_eq!(
        dataset
            .count_rows(Some("value = 2".to_string()))
            .await
            .unwrap(),
        if partial { updated_rows as usize } else { 6 }
    );
    assert_eq!(
        dataset.manifest.data_storage_format.lance_file_format(),
        ConcreteFileVersion::V2_0
    );
    assert!(
        dataset
            .manifest
            .fragments
            .iter()
            .flat_map(Fragment::referenced_lance_files)
            .filter(|file| !original_paths.contains(&file.path))
            .all(|file| file.file_version().unwrap()
                == target.unwrap_or(LanceFileVersion::V2_0).resolve())
    );
    assert_eq!(
        dataset.manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS != 0,
        target.is_some()
    );
}

#[rstest::rstest]
#[case(LanceFileVersion::Legacy, LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_0, LanceFileVersion::Legacy)]
#[tokio::test]
async fn merge_insert_rejects_cross_family_target(
    #[case] source: LanceFileVersion,
    #[case] target: LanceFileVersion,
) {
    let dataset = create_test_dataset("memory://", source, false).await;
    let error = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()])
        .unwrap()
        .data_storage_version(target)
        .try_build()
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("V1 and V2 storage versions cannot be mixed")
    );
}

#[rstest::rstest]
#[tokio::test]
async fn test_basic_merge(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
) {
    let schema = create_test_schema();
    let new_batch = create_new_batch(schema.clone());

    let test_uri = "memory://test.lance";

    let ds = create_test_dataset(test_uri, version, false).await;

    // Quick test that no on-keys is not valid and fails
    assert!(MergeInsertBuilder::try_new(ds.clone(), vec![]).is_err());

    let keys = vec!["key".to_string()];
    // find-or-create, no delete
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3, 4, 5, 6],
        &[7, 8, 9],
        &[3, 0, 0],
    ))
    .await;

    // upsert, no delete
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3],
        &[4, 5, 6, 7, 8, 9],
        &[3, 3, 0],
    ))
    .await;

    // conditional upsert, no delete
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap())
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3, 4, 5],
        &[6, 7, 8, 9],
        &[3, 1, 0],
    ))
    .await;

    // conditional update, no matches
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::update_if(&ds, "target.filterme = 'z'").unwrap())
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3, 4, 5, 6],
        &[],
        &[0, 0, 0],
    ))
    .await;

    // update only, no delete (useful for bulk update)
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3],
        &[4, 5, 6],
        &[0, 3, 0],
    ))
    .await;

    // Conditional update
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "source.filterme == target.filterme").unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 2, 3, 6],
        &[4, 5],
        &[0, 2, 0],
    ))
    .await;

    // No-op (will raise an error)
    assert!(
        MergeInsertBuilder::try_new(ds.clone(), keys.clone())
            .unwrap()
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .is_err()
    );

    // find-or-create, with delete all
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[4, 5, 6],
        &[7, 8, 9],
        &[3, 0, 3],
    ))
    .await;

    // upsert, with delete all
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[],
        &[4, 5, 6, 7, 8, 9],
        &[3, 3, 3],
    ))
    .await;

    // conditional upsert, with delete all
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap())
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[4, 5],
        &[6, 7, 8, 9],
        &[3, 1, 3],
    ))
    .await;

    // update only, with delete all (unusual)
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[],
        &[4, 5, 6],
        &[0, 3, 3],
    ))
    .await;

    // just delete all (not real case, just use delete)
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[4, 5, 6],
        &[],
        &[0, 0, 3],
    ))
    .await;

    // For the "delete some" tests we use key > 1
    let condition = create_delete_condition();
    // find-or-create, with delete some
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 4, 5, 6],
        &[7, 8, 9],
        &[3, 0, 2],
    ))
    .await;

    // upsert, with delete some
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1],
        &[4, 5, 6, 7, 8, 9],
        &[3, 3, 2],
    ))
    .await;

    // conditional upsert, with delete some
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap())
        .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 4, 5],
        &[6, 7, 8, 9],
        &[3, 1, 2],
    ))
    .await;

    // update only, witxh delete some (unusual)
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1],
        &[4, 5, 6],
        &[0, 3, 2],
    ))
    .await;

    // just delete some (not real case, just use delete)
    let job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition.clone()))
        .try_build()
        .unwrap();
    // Boxed: the dataset and job held across the await put this
    // future over the `large_futures` threshold.
    Box::pin(check_then_refresh_dataset(
        new_batch.clone(),
        job,
        &[1, 4, 5, 6],
        &[],
        &[0, 0, 2],
    ))
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_upsert_and_delete_all_with_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    // Boxed: the builder holds several `Fragment`s, which puts the
    // future over the `large_futures` threshold.
    Box::pin(
        MergeInsertTestBuilder::new()
            .with_version(version)
            .with_stable_row_ids(enable_stable_row_ids)
            .with_test_keys(&[4, 5, 6])
            .with_expected_left_keys(&[])
            .with_expected_right_keys(&[4, 5, 6, 7, 8, 9])
            .with_expected_stats(&[3, 3, 3])
            .with_job_builder(|ds| {
                MergeInsertBuilder::try_new(ds, vec!["key".to_string()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
                    .try_build()
                    .unwrap()
            })
            .run_test(),
    )
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_upsert_only_with_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    // Boxed: the builder holds several `Fragment`s, which puts the
    // future over the `large_futures` threshold.
    Box::pin(
        MergeInsertTestBuilder::new()
            .with_version(version)
            .with_stable_row_ids(enable_stable_row_ids)
            .with_test_keys(&[4, 5, 6])
            .with_expected_left_keys(&[1, 2, 3])
            .with_expected_right_keys(&[4, 5, 6, 7, 8, 9])
            .with_expected_stats(&[3, 3, 0])
            .with_job_builder(|ds| {
                MergeInsertBuilder::try_new(ds, vec!["key".to_string()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .try_build()
                    .unwrap()
            })
            .run_test(),
    )
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_conditional_update_with_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    // Boxed: the builder holds several `Fragment`s, which puts the
    // future over the `large_futures` threshold.
    Box::pin(
        MergeInsertTestBuilder::new()
            .with_version(version)
            .with_stable_row_ids(enable_stable_row_ids)
            .with_test_keys(&[6])
            .with_expected_left_keys(&[1, 2, 3, 4, 5])
            .with_expected_right_keys(&[6, 7, 8, 9])
            .with_expected_stats(&[3, 1, 0])
            .with_job_builder(|ds| {
                let keys = vec!["key".to_string()];
                MergeInsertBuilder::try_new(ds.clone(), keys)
                    .unwrap()
                    .when_matched(
                        WhenMatched::update_if(&ds, "source.filterme != target.filterme").unwrap(),
                    )
                    .try_build()
                    .unwrap()
            })
            .run_test(),
    )
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_update_only_with_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    // Boxed: the builder holds several `Fragment`s, which puts the
    // future over the `large_futures` threshold.
    Box::pin(
        MergeInsertTestBuilder::new()
            .with_version(version)
            .with_stable_row_ids(enable_stable_row_ids)
            .with_test_keys(&[4, 5, 6])
            .with_expected_left_keys(&[1, 2, 3])
            .with_expected_right_keys(&[4, 5, 6])
            .with_expected_stats(&[0, 3, 0])
            .with_job_builder(|ds| {
                let keys = vec!["key".to_string()];
                MergeInsertBuilder::try_new(ds, keys)
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::DoNothing)
                    .try_build()
                    .unwrap()
            })
            .run_test(),
    )
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_upsert_with_conditional_delete_and_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    // Boxed: the builder holds several `Fragment`s, which puts the
    // future over the `large_futures` threshold.
    Box::pin(
        MergeInsertTestBuilder::new()
            .with_version(version)
            .with_stable_row_ids(enable_stable_row_ids)
            .with_test_keys(&[1, 4, 5, 6])
            .with_expected_left_keys(&[1])
            .with_expected_right_keys(&[4, 5, 6, 7, 8, 9])
            .with_expected_stats(&[3, 3, 2])
            .with_job_builder(|ds| {
                let keys = vec!["key".to_string()];
                let condition = create_delete_condition();
                MergeInsertBuilder::try_new(ds, keys)
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched_by_source(WhenNotMatchedBySource::DeleteIf(condition))
                    .try_build()
                    .unwrap()
            })
            .run_test(),
    )
    .await;
}

#[rstest::rstest]
#[tokio::test]
async fn test_multiple_merge_insert_stable_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    let schema = create_test_schema();
    let test_uri = "memory://test_multiple_merge.lance";

    let ds = create_test_dataset(test_uri, version, enable_stable_row_ids).await;

    let target_key = 2u32;
    let target_keys = vec![target_key];

    let initial_row_ids = get_row_ids_for_keys(&ds, &target_keys).await;
    let initial_row_id = initial_row_ids.value(0);

    let mut current_ds = ds;

    for iteration in 1..=3 {
        let new_value = 1000u32 + iteration * 10;
        let new_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![target_key])), // key
                Arc::new(UInt32Array::from(vec![new_value])),  // value
                Arc::new(StringArray::from(vec![format!("iteration_{}", iteration)])), // filterme
            ],
        )
        .unwrap();

        let job = MergeInsertBuilder::try_new(current_ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();

        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
        let new_stream = reader_to_stream(new_reader);
        let (updated_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

        assert_eq!(
            merge_stats.num_updated_rows, 1,
            "Iteration {}: Expected 1 updated row",
            iteration
        );
        assert_eq!(
            merge_stats.num_inserted_rows, 0,
            "Iteration {}: Expected 0 inserted rows",
            iteration
        );
        assert_eq!(
            merge_stats.num_deleted_rows, 0,
            "Iteration {}: Expected 0 deleted rows",
            iteration
        );

        let updated_row_ids = get_row_ids_for_keys(&updated_dataset, &target_keys).await;
        let updated_row_id = updated_row_ids.value(0);

        let updated_batch = updated_dataset
            .scan()
            .filter(&format!("key = {}", target_key))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let value_col = updated_batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let filterme_col = updated_batch
            .column_by_name("filterme")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(
            value_col.value(0),
            new_value,
            "Iteration {}: Value should be updated to {}",
            iteration,
            new_value
        );
        assert_eq!(filterme_col.value(0), format!("iteration_{}", iteration));

        if enable_stable_row_ids {
            assert_eq!(
                updated_row_id, initial_row_id,
                "Iteration {}: Row ID should remain stable across merge inserts when stable_row_ids is enabled. Initial: {}, Current: {}",
                iteration, initial_row_id, updated_row_id
            );
        }

        current_ds = updated_dataset;
    }

    let final_batch = current_ds
        .scan()
        .filter(&format!("key = {}", target_key))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(
        final_batch.num_rows(),
        1,
        "Should have exactly one row for the target key"
    );

    let final_value = final_batch
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .value(0);
    let final_filterme = final_batch
        .column_by_name("filterme")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);

    assert_eq!(
        final_value, 1030u32,
        "Final value should be from last iteration"
    );
    assert_eq!(
        final_filterme, "iteration_3",
        "Final filterme should be from last iteration"
    );
}

#[rstest::rstest]
#[case::updates_before_inserts(1, 1, 7, 2)]
#[case::insert_batches_before_updates(1000, 1, 0, 256)]
#[case::insert_batches_between_updates(128, 256, 0, 1024)]
#[tokio::test]
async fn test_multi_batch_upsert_preserves_stable_row_ids(
    #[case] first_key: u32,
    #[case] key_step: u32,
    #[case] first_insert: u32,
    #[case] insert_range_len: u32,
    #[values(true, false)] use_index: bool,
    #[values(64, 1024)] max_rows_per_file: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let mut dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(first_key, key_step))
        .col("value", array::fill::<UInt32Type>(1))
        .col("filterme", array::fill_utf8("original".to_string()))
        .into_dataset_with_params(
            uri,
            FragmentCount(2),
            FragmentRowCount(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["key"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    let initial_keys = (0..6).map(|i| first_key + i * key_step).collect::<Vec<_>>();
    let initial_row_ids = get_row_ids_for_keys(&dataset, &initial_keys).await;
    let updated_keys = [initial_keys[1], initial_keys[3]];
    let inserted_keys = (first_insert..first_insert + insert_range_len)
        .filter(|key| !initial_keys.contains(key))
        .collect::<Vec<_>>();
    let midpoint = inserted_keys.len() / 2;

    // Input batch order differs from key order. The indexed join must also
    // handle entire output batches of inserts before or between updates.
    let first_insert_batch = record_batch!(
        ("key", UInt32, inserted_keys[..midpoint].to_vec()),
        ("value", UInt32, vec![70; midpoint]),
        ("filterme", Utf8, vec!["inserted before"; midpoint])
    )
    .unwrap();
    let update_batch = record_batch!(
        ("key", UInt32, updated_keys.to_vec()),
        ("value", UInt32, [20, 20]),
        ("filterme", Utf8, ["updated", "updated"])
    )
    .unwrap();
    let second_insert_batch = record_batch!(
        ("key", UInt32, inserted_keys[midpoint..].to_vec()),
        ("value", UInt32, vec![80; inserted_keys.len() - midpoint]),
        (
            "filterme",
            Utf8,
            vec!["inserted after"; inserted_keys.len() - midpoint]
        )
    )
    .unwrap();
    let source_schema = first_insert_batch.schema();
    let source = Box::new(RecordBatchIterator::new(
        [
            Ok(first_insert_batch),
            Ok(update_batch),
            Ok(second_insert_batch),
        ],
        source_schema,
    ));

    let (dataset, stats) = MergeInsertBuilder::try_new(dataset, vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .use_index(use_index)
        .with_params(&WriteParams {
            max_rows_per_file,
            ..Default::default()
        })
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, updated_keys.len() as u64);
    assert_eq!(stats.num_inserted_rows, inserted_keys.len() as u64);
    drop(dataset);
    let dataset = Dataset::open(uri).await.unwrap();
    assert_eq!(
        dataset.count_rows(None).await.unwrap(),
        initial_keys.len() + inserted_keys.len()
    );
    let row_ids_after = get_row_ids_for_keys(&dataset, &initial_keys).await;
    assert_eq!(row_ids_after, initial_row_ids);
    let original_rows = dataset
        .take_rows(initial_row_ids.values(), dataset.schema().clone())
        .await
        .unwrap();
    assert_eq!(
        original_rows["key"].as_primitive::<UInt32Type>().values(),
        &initial_keys
    );
    assert_eq!(
        original_rows["value"].as_primitive::<UInt32Type>().values(),
        &[1, 20, 1, 20, 1, 1]
    );

    let inserted_row_ids = get_row_ids_for_keys(&dataset, &inserted_keys).await;
    assert!(
        inserted_row_ids
            .values()
            .iter()
            .all(|row_id| !initial_row_ids.values().contains(row_id))
    );
}

#[rstest::rstest]
#[tokio::test]
async fn test_row_id_stability_across_update_and_merge_insert(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    let schema = create_test_schema();
    let test_uri = "memory://test_row_id_stability.lance";

    let mut dataset = create_test_dataset(test_uri, version, enable_stable_row_ids).await;

    let target_key = 2u32;
    let target_keys = vec![target_key];

    let initial_row_ids = get_row_ids_for_keys(&dataset, &target_keys).await;
    let initial_row_id = initial_row_ids.value(0);

    let initial_batch = dataset
        .scan()
        .filter(&format!("key = {}", target_key))
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let initial_value = initial_batch
        .column_by_name("value")
        .unwrap()
        .as_primitive::<UInt32Type>()
        .value(0);

    let update_result = crate::dataset::UpdateBuilder::new(Arc::new((*dataset).clone()))
        .update_where(&format!("key = {}", target_key))
        .unwrap()
        .set("value", "value + 100")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();

    dataset = update_result.new_dataset.clone();

    let after_update_row_ids = get_row_ids_for_keys(&dataset, &target_keys).await;
    let after_update_row_id = after_update_row_ids.value(0);

    let after_update_batch = dataset
        .scan()
        .filter(&format!("key = {}", target_key))
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let after_update_value = after_update_batch
        .column_by_name("value")
        .unwrap()
        .as_primitive::<UInt32Type>()
        .value(0);

    if enable_stable_row_ids {
        assert_eq!(
            initial_row_id, after_update_row_id,
            "Row ID should remain stable after update"
        );
    } else {
        assert_ne!(
            initial_row_id, after_update_row_id,
            "Row ID should change after update when stable row IDs are disabled"
        );
    }
    assert_eq!(
        after_update_value,
        initial_value + 100,
        "Value should be updated correctly"
    );

    let merge_new_value = 500u32;
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![target_key])),
            Arc::new(UInt32Array::from(vec![merge_new_value])),
            Arc::new(StringArray::from(vec!["UPDATED"])),
        ],
    )
    .unwrap();

    let job = MergeInsertBuilder::try_new(dataset.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

    let after_merge_row_ids = get_row_ids_for_keys(&merged_dataset, &target_keys).await;
    let after_merge_row_id = after_merge_row_ids.value(0);

    let after_merge_batch = merged_dataset
        .scan()
        .filter(&format!("key = {}", target_key))
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();

    let after_merge_value = after_merge_batch
        .column_by_name("value")
        .unwrap()
        .as_primitive::<UInt32Type>()
        .value(0);

    let after_merge_filterme = after_merge_batch
        .column_by_name("filterme")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);

    if enable_stable_row_ids {
        assert_eq!(
            initial_row_id, after_merge_row_id,
            "Row ID should remain stable after merge insert"
        );
        assert_eq!(
            after_update_row_id, after_merge_row_id,
            "Row ID should remain the same across update and merge insert"
        );
    } else {
        assert_ne!(
            after_update_row_id, after_merge_row_id,
            "Row ID should change after merge insert when stable row IDs are disabled"
        );
    }

    assert_eq!(
        after_merge_value, merge_new_value,
        "Value should be updated by merge insert"
    );
    assert_eq!(
        after_merge_filterme, "UPDATED",
        "Filterme should be updated by merge insert"
    );

    assert_eq!(
        merge_stats.num_updated_rows, 1,
        "Should update exactly 1 row"
    );
    assert_eq!(
        merge_stats.num_inserted_rows, 0,
        "Should not insert any new rows"
    );
    assert_eq!(
        merge_stats.num_deleted_rows, 0,
        "Should not delete any rows"
    );

    if enable_stable_row_ids {
        assert_eq!(
            initial_row_id, after_merge_row_id,
            "Row ID should remain stable throughout the entire process of update and merge insert"
        );
    }
}

/// Reproduces https://github.com/lancedb/lancedb/issues/3515:
/// a partial-schema `merge_insert` with a scalar index on the join key,
/// where every fragment is covered by the index (no unindexed data),
/// silently updates 0 rows instead of the expected matches.
#[rstest::rstest]
#[tokio::test]
async fn test_repro_3515_partial_schema_fully_indexed(
    #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
    version: LanceFileVersion,
) {
    const N: usize = 1000;
    const UPD: usize = 128;
    let vec_field = Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
        true,
    );
    let full_schema = Arc::new(Schema::new(vec![
        vec_field.clone(),
        Field::new("path", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, true),
        Field::new("file_size", DataType::Int64, true),
    ]));

    // 1000 rows: vector all-null, path "/img/{i}.jpg", status "pending".
    let paths = StringArray::from((0..N).map(|i| format!("/img/{i}.jpg")).collect::<Vec<_>>());
    let statuses = StringArray::from(vec!["pending"; N]);
    let file_sizes = Int64Array::from((0..N as i64).map(|i| 1000 + i).collect::<Vec<_>>());
    let null_vectors = arrow_array::new_null_array(vec_field.data_type(), N);
    let batch = RecordBatch::try_new(
        full_schema.clone(),
        vec![
            null_vectors,
            Arc::new(paths),
            Arc::new(statuses),
            Arc::new(file_sizes),
        ],
    )
    .unwrap();

    let mut ds = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], full_schema.clone()),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Scalar index on the merge key, covering every fragment.
    ds.create_index(
        &["path"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);

    // Partial-schema source (no `file_size`): update the first 128 rows.
    let upd_schema = Arc::new(Schema::new(vec![
        vec_field,
        Field::new("path", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, true),
    ]));
    let upd_paths = StringArray::from(
        (0..UPD)
            .map(|i| format!("/img/{i}.jpg"))
            .collect::<Vec<_>>(),
    );
    let upd_vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.1f32; 4 * UPD]), 4)
            .unwrap();
    let upd_statuses = StringArray::from(vec!["indexed"; UPD]);
    let updates = RecordBatch::try_new(
        upd_schema.clone(),
        vec![
            Arc::new(upd_vectors),
            Arc::new(upd_paths),
            Arc::new(upd_statuses),
        ],
    )
    .unwrap();

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["path".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    let job_disk_manager = job.spill_session_context.runtime_env().disk_manager.clone();
    let fragment_update_context =
        fragment_update_session_context(&job.spill_session_context, &job.spill_execution_options);
    assert!(
        Arc::ptr_eq(
            &job_disk_manager,
            &fragment_update_context.runtime_env().disk_manager
        ),
        "indexed execution and partial-update sorting must share one disk manager"
    );

    let (ds, stats) = job
        .execute_reader(RecordBatchIterator::new([Ok(updates)], upd_schema))
        .await
        .unwrap();

    assert_eq!(
        stats.num_updated_rows, UPD as u64,
        "expected {UPD} updated rows on {version:?}, got {}",
        stats.num_updated_rows
    );
    let n_indexed = ds
        .count_rows(Some("status = 'indexed'".to_string()))
        .await
        .unwrap();
    assert_eq!(n_indexed, UPD, "expected {UPD} rows flipped to 'indexed'");
}

#[rstest::rstest]
#[case::reordered_null_payload(false, Some(42))]
#[case::target_name_collision(true, None)]
#[tokio::test]
async fn test_indexed_partial_merge_with_reordered_source(
    #[case] has_target_name_collision: bool,
    #[case] expected_payload: Option<i32>,
) {
    let (target, source, payload_column) = if has_target_name_collision {
        (
            record_batch!(
                ("id", UInt64, [1]),
                ("target_id", Int32, [9]),
                ("b", Int32, [7])
            )
            .unwrap(),
            record_batch!(("target_id", Int32, [None]), ("id", UInt64, [1])).unwrap(),
            "target_id",
        )
    } else {
        (
            record_batch!(("id", UInt64, [1]), ("a", Int32, [None]), ("b", Int32, [7])).unwrap(),
            record_batch!(("a", Int32, [42]), ("id", UInt64, [1])).unwrap(),
            "a",
        )
    };
    let mut dataset = InsertBuilder::new("memory://")
        .execute(vec![target])
        .await
        .unwrap();
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

    let source_schema = source.schema();
    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(source)], source_schema))
        .await
        .unwrap();

    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_deleted_rows, 0);
    let result = dataset
        .scan()
        .project(&[payload_column])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let payload = result[payload_column].as_primitive::<Int32Type>();
    let actual_payload = (!payload.is_null(0)).then(|| payload.value(0));
    assert_eq!(actual_payload, expected_payload);
}

/// merge_insert matches float keys per IEEE 754, like filters do, so a source
/// key of `+0.0` matches both the `-0.0` and the `+0.0` row. Both settings of
/// `use_index` are exercised; which one the planner picks for a one-row source
/// is not asserted.
#[rstest::rstest]
#[tokio::test]
async fn test_merge_insert_on_float_zero_key(#[values(true, false)] use_index: bool) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let target = record_batch!(
        (
            "key",
            Float64,
            [Some(-1.0), Some(-0.0), Some(0.0), Some(1.0)]
        ),
        ("value", Int32, [10, 20, 30, 40])
    )
    .unwrap();
    let schema = target.schema();
    let reader = RecordBatchIterator::new(vec![Ok(target)], schema.clone());
    let mut ds = Dataset::write(reader, test_uri, None).await.unwrap();
    ds.create_index(
        &["key"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    let source = record_batch!(("key", Float64, [Some(0.0)]), ("value", Int32, [99])).unwrap();
    let source = Box::new(RecordBatchIterator::new(vec![Ok(source)], schema.clone()));

    let (ds, _) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["key".to_string()])
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::UpdateAll)
        .use_index(use_index)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    // Both zero rows are updated; `value = 20` was the -0.0 row and
    // `value = 30` the +0.0 row.
    assert_eq!(ds.count_rows(None).await.unwrap(), 4);
    for (filter, expected) in [("value = 99", 2), ("value = 30", 0), ("value = 20", 0)] {
        assert_eq!(
            ds.count_rows(Some(filter.to_string())).await.unwrap(),
            expected,
            "{filter}"
        );
    }
}

/// `-0.0` and `+0.0` are one key, so a source holding both has a duplicate
/// key: each zero target row is matched twice. The default `Fail` rejects
/// that as ambiguous; `FirstSeen` updates each zero row once, with either
/// source row's value depending on join order.
#[rstest::rstest]
#[tokio::test]
async fn test_merge_insert_source_with_both_float_zeros(
    #[values(true, false)] use_index: bool,
    #[values(SourceDedupeBehavior::Fail, SourceDedupeBehavior::FirstSeen)]
    behavior: SourceDedupeBehavior,
) {
    let target = record_batch!(
        (
            "key",
            Float64,
            [Some(-1.0), Some(-0.0), Some(0.0), Some(1.0)]
        ),
        ("value", Int32, [10, 20, 30, 40])
    )
    .unwrap();
    let schema = target.schema();
    let reader = RecordBatchIterator::new(vec![Ok(target)], schema.clone());
    let mut ds = Dataset::write(reader, "memory://", None).await.unwrap();
    ds.create_index(
        &["key"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    let source = record_batch!(
        ("key", Float64, [Some(-0.0), Some(0.0)]),
        ("value", Int32, [98, 99])
    )
    .unwrap();
    let result = MergeInsertBuilder::try_new(Arc::new(ds), vec!["key".to_string()])
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::UpdateAll)
        .use_index(use_index)
        .source_dedupe_behavior(behavior)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(source)],
            schema.clone(),
        )))
        .await;

    if behavior == SourceDedupeBehavior::Fail {
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Ambiguous merge inserts"),
            "Fail must reject a source holding both zeros, got: {err}"
        );
        return;
    }

    let (ds, _) = result.unwrap();
    assert_eq!(ds.count_rows(None).await.unwrap(), 4);
    for (filter, expected) in [
        ("value = 98 OR value = 99", 2),
        ("value = 20", 0),
        ("value = 30", 0),
    ] {
        assert_eq!(
            ds.count_rows(Some(filter.to_string())).await.unwrap(),
            expected,
            "{filter}"
        );
    }
}

/// `try_build` refuses write parameters a merge insert sets on its own
/// builder or cannot honor.
#[tokio::test]
async fn test_merge_insert_with_params_refuses_what_merge_insert_owns() {
    let initial = record_batch!(("id", Int64, [0])).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(initial.clone())], initial.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    for (params, needle) in [
        (
            WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            },
            "mode=Overwrite",
        ),
        (
            WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            },
            "mode=Append",
        ),
        (
            WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            },
            "data_storage_version",
        ),
        (
            WriteParams {
                target_bases: Some(vec![0]),
                ..Default::default()
            },
            "target_bases=Some([0])",
        ),
        (
            WriteParams {
                target_all_bases: Some(true),
                ..Default::default()
            },
            "target_all_bases=Some(true)",
        ),
        (
            WriteParams {
                external_blob_mode: ExternalBlobMode::Ingest,
                ..Default::default()
            },
            "ingest",
        ),
    ] {
        let error = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
            .unwrap()
            .with_params(&params)
            .try_build()
            .err()
            .unwrap_or_else(|| panic!("{needle} must be refused"));
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains(needle), "{needle}: {error}");
    }
}
