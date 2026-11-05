// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use async_trait::async_trait;

#[tokio::test]
async fn test_indexed_merge_insert() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));
    let schema = data.schema();

    // Create an input dataset with a scalar index on key
    let mut ds = Dataset::write(data, test_uri, None).await.unwrap();
    let index_params = ScalarIndexParams::default();
    ds.create_index(&["key"], IndexType::Scalar, None, &index_params, false)
        .await
        .unwrap();

    // Create some new (unindexed) data
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(8));
    let ds = Dataset::write(
        data,
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let ds = Arc::new(ds);

    let just_index_col = Schema::new(vec![Field::new("key", DataType::Utf8, false)]);

    // Sample 2048 random indices and then paste on a column of 9999999's
    let some_indices = ds
        .sample(2048, &(&just_index_col).try_into().unwrap(), None)
        .await
        .unwrap();
    let some_indices = some_indices.column(0).clone();
    let some_vals = lance_datagen::gen_batch()
        .anon_col(array::fill::<UInt32Type>(9999999))
        .into_batch_rows(RowCount::from(2048))
        .unwrap();
    let some_vals = some_vals.column(0).clone();
    let source_batch = RecordBatch::try_new(schema.clone(), vec![some_vals, some_indices]).unwrap();
    // To make things more interesting, lets make the input a stream of four batches
    let source_batches = vec![
        source_batch.slice(0, 512),
        source_batch.slice(512, 512),
        source_batch.slice(1024, 512),
        source_batch.slice(1536, 512),
    ];
    let source = Box::new(RecordBatchIterator::new(
        source_batches.clone().into_iter().map(Ok),
        schema.clone(),
    ));

    // Run merge_insert
    let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    // Check that the data is as expected
    let updated = ds
        .count_rows(Some("value = 9999999".to_string()))
        .await
        .unwrap();
    assert_eq!(updated, 2048);

    // Make sure we don't use an indexed scan if there is a delete criteria
    let source = Box::new(RecordBatchIterator::new(
        source_batches.clone().into_iter().map(Ok),
        schema.clone(),
    ));
    // Run merge_insert
    let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    // Check that the data is as expected
    assert_eq!(ds.count_rows(None).await.unwrap(), 2048);

    let source = Box::new(RecordBatchIterator::new(
        source_batches.clone().into_iter().map(Ok),
        schema.clone(),
    ));
    // Run merge_insert one last time.  The index is now completely out of date.  Every
    // row it points to is a deleted row.  Make sure that doesn't break.
    let (ds, _) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_not_matched(WhenNotMatched::DoNothing)
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap()
        .execute_reader(source)
        .await
        .unwrap();

    assert_eq!(ds.count_rows(None).await.unwrap(), 2048);
}

/// Multi-column (composite key) merge_insert when one or more join
/// columns have a scalar index.  Before this change the indexed path
/// was hard-gated to single-column joins; composite-key merges fell
/// through to a full target scan even when every key column was
/// indexed.  Now each indexed column contributes an AND-ed `IsIn`
/// probe inside one `MapIndexExec`, and the downstream hash join trims
/// the candidates to the exact composite-key matches — rows that
/// happen to match one key column but not the other must NOT be
/// touched.
#[rstest::rstest]
#[case::index_on_first(true, false)]
#[case::index_on_second(false, true)]
#[case::index_on_both(true, true)]
#[tokio::test]
async fn test_indexed_merge_insert_composite_key(
    #[case] index_on_a: bool,
    #[case] index_on_b: bool,
) {
    // Target rows: every (a, b) combination from {1,2} x {10,20}.
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    if index_on_a {
        ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
    }
    if index_on_b {
        ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
    }

    // Update (1, 10) and insert (3, 30).  A naive single-column probe
    // on `a` would also pull (1, 20) into the candidate set; the
    // composite hash join must keep (1, 20) untouched.
    let source = record_batch!(
        ("a", Int32, [1, 3]),
        ("b", Int32, [10, 30]),
        ("value", Int32, [999, 333])
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 5);

    let untouched = updated_ds
        .count_rows(Some("a = 1 AND b = 20 AND value = 200".to_string()))
        .await
        .unwrap();
    assert_eq!(
        untouched, 1,
        "(1, 20) must not be clobbered by an `a`-only probe"
    );

    let updated = updated_ds
        .count_rows(Some("a = 1 AND b = 10 AND value = 999".to_string()))
        .await
        .unwrap();
    assert_eq!(updated, 1);

    let inserted = updated_ds
        .count_rows(Some("a = 3 AND b = 30 AND value = 333".to_string()))
        .await
        .unwrap();
    assert_eq!(inserted, 1);
}

/// A composite index probe can over-match the exact join key, but a target
/// row reached by more than one source batch must still enter the join once.
#[tokio::test]
async fn test_indexed_merge_insert_deduplicates_cross_batch_candidates() {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial)], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    dataset
        .create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    dataset
        .create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    let first = record_batch!(
        ("a", Int32, [1]),
        ("b", Int32, [10]),
        ("value", Int32, [901])
    )
    .unwrap();
    // This batch probes `a IN (1, 2) AND b IN (20, 10)`, which reaches
    // (1, 10) again even though that tuple is not present in this batch.
    let second = record_batch!(
        ("a", Int32, [1, 2]),
        ("b", Int32, [20, 10]),
        ("value", Int32, [902, 903])
    )
    .unwrap();

    let (dataset, stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(first), Ok(second)],
                schema,
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_updated_rows, 3);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
    for (a, b, value) in [(1, 10, 901), (1, 20, 902), (2, 10, 903), (2, 20, 400)] {
        assert_eq!(
            dataset
                .count_rows(Some(format!("a = {a} AND b = {b} AND value = {value}")))
                .await
                .unwrap(),
            1,
        );
    }
}

/// Composite key merge_insert with no scalar index on any join column
/// must keep working via the full-scan fallback.  Guards against the
/// indexed path becoming a hard requirement after this optimization.
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_no_index() {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2]),
        ("b", Int32, [10, 20, 10]),
        ("value", Int32, [100, 200, 300])
    )
    .unwrap();
    let schema = initial.schema();

    let ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let source = record_batch!(
        ("a", Int32, [1]),
        ("b", Int32, [20]),
        ("value", Int32, [999])
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 0);
    let count = updated_ds
        .count_rows(Some("a = 1 AND b = 20 AND value = 999".to_string()))
        .await
        .unwrap();
    assert_eq!(count, 1);
}

/// Composite-key merge_insert must use standard SQL NULL semantics
/// (NULL != NULL) on the post-filter hash join so its behavior is
/// identical to the full-scan path; otherwise enabling the indexed
/// path for multi-column joins would silently change semantics.
#[rstest::rstest]
#[case::btree(IndexType::BTree)]
#[case::bitmap(IndexType::Bitmap)]
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_null_semantics(#[case] index_type: IndexType) {
    let initial = record_batch!(
        ("a", Int32, [Some(1), None]),
        ("b", Utf8, [Option::<&str>::None, None]),
        ("value", Int32, [10, 11])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    for key in ["a", "b"] {
        ds.create_index(
            &[key],
            index_type,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    }

    let source = record_batch!(
        ("a", Int32, [Some(1), None]),
        ("b", Utf8, [Option::<&str>::None, None]),
        ("value", Int32, [99, 100])
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_inserted_rows, 2);
    assert_eq!(stats.num_updated_rows, 0);
    let actual = updated_ds
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("value".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let expected = record_batch!(
        ("a", Int32, [Some(1), None, Some(1), None]),
        ("b", Utf8, [Option::<&str>::None, None, None, None]),
        ("value", Int32, [10, 11, 99, 100])
    )
    .unwrap();
    assert_eq!(actual, expected);
}

/// Composite-key merge_insert where new (unindexed) fragments are
/// appended after the indices were built.  The indexed take only sees
/// fragments covered by every chosen index, so the unindexed remainder
/// must be unioned in via a full scan — otherwise updates to rows
/// that live in those fragments are silently dropped.
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_unindexed_fragments() {
    let first = record_batch!(
        ("a", Int32, [1, 2]),
        ("b", Int32, [10, 20]),
        ("value", Int32, [100, 200])
    )
    .unwrap();
    let schema = first.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(first.clone())], schema.clone()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 64,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    // Append a fragment AFTER both indices are built.  The new (3, 30)
    // row lives in a fragment neither index covers, so the indexed
    // take alone would miss it.
    let appended = record_batch!(
        ("a", Int32, [3]),
        ("b", Int32, [30]),
        ("value", Int32, [300])
    )
    .unwrap();
    ds.append(
        RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
        None,
    )
    .await
    .unwrap();

    // Source updates one row in the indexed fragment AND one row in
    // the appended (unindexed) fragment.
    let source = record_batch!(
        ("a", Int32, [1, 3]),
        ("b", Int32, [10, 30]),
        ("value", Int32, [999, 333])
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(
        stats.num_updated_rows, 2,
        "row in the unindexed fragment must also be updated"
    );
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 3);
}

/// Composite-key delete-only merge_insert (`when_matched(Delete)`,
/// `when_not_matched_by_source(Keep)`) removes the matched rows for every
/// combination of which join columns carry a scalar index, including when
/// every column is indexed.
#[rstest::rstest]
#[case::index_on_both(true, true)]
#[case::index_on_first(true, false)]
#[case::index_on_second(false, true)]
#[case::no_index(false, false)]
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_delete(
    #[case] index_on_a: bool,
    #[case] index_on_b: bool,
) {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    if index_on_a {
        ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
    }
    if index_on_b {
        ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
            .await
            .unwrap();
    }

    // Delete (1, 10) by composite key.  Only key columns in the source.
    let source = record_batch!(("a", Int32, [1]), ("b", Int32, [10])).unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 1, "matched row must be deleted");
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 3);
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1 AND b = 10".to_string()))
            .await
            .unwrap(),
        0,
        "(1, 10) must be gone after the delete"
    );
    // The sibling key (1, 20) must remain untouched.
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1 AND b = 20 AND value = 200".to_string()))
            .await
            .unwrap(),
        1,
    );
}

/// Delete-only merge_insert on a single indexed key removes the matched
/// rows (the indexed path is not exclusive to composite keys).
#[tokio::test]
async fn test_indexed_merge_insert_single_key_delete() {
    let initial = record_batch!(
        ("id", Int32, [1, 2, 3, 4]),
        ("value", Int32, [10, 20, 30, 40])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    ds.create_index(
        &["id"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    let source = record_batch!(("id", Int32, [2, 4])).unwrap();

    let (updated_ds, stats) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(source.clone())],
            source.schema(),
        )))
        .await
        .unwrap();

    assert_eq!(stats.num_deleted_rows, 2);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 2);
    assert_eq!(
        updated_ds
            .count_rows(Some("id = 2 OR id = 4".to_string()))
            .await
            .unwrap(),
        0,
    );
}

/// `when_matched(Fail)` on an indexed key aborts the operation when a
/// source row matches an existing key, and inserts cleanly when none do.
#[tokio::test]
async fn test_indexed_merge_insert_when_matched_fail() {
    let initial = record_batch!(("id", Int32, [1, 2, 3]), ("value", Int32, [10, 20, 30])).unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    ds.create_index(
        &["id"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);

    // A source row matching an existing key must fail the operation.
    let matching = record_batch!(("id", Int32, [2]), ("value", Int32, [999])).unwrap();
    let err = MergeInsertBuilder::try_new(ds.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Fail)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(matching.clone())],
            matching.schema(),
        )))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Merge insert failed"), "got: {msg}");
    assert!(msg.contains("found matching row"), "got: {msg}");

    // A source with no matching key inserts without failing.
    let new_rows = record_batch!(("id", Int32, [4]), ("value", Int32, [40])).unwrap();
    let (updated_ds, stats) = MergeInsertBuilder::try_new(ds.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Fail)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(new_rows.clone())],
            new_rows.schema(),
        )))
        .await
        .unwrap();
    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(stats.num_updated_rows, 0);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 4);
}

/// Fully-indexed composite-key `when_matched(Delete)` combined with
/// `when_not_matched(InsertAll)` must both delete matched rows and write
/// the inserted rows.
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_delete_with_insert() {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    // Source matches (1, 10) -> delete, and (3, 30) is new -> insert.
    let source = record_batch!(
        ("a", Int32, [1, 3]),
        ("b", Int32, [10, 30]),
        ("value", Int32, [999, 333])
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    // 4 - 1 deleted + 1 inserted = 4.
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 4);
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1 AND b = 10".to_string()))
            .await
            .unwrap(),
        0,
        "matched row must be deleted, not updated"
    );
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 3 AND b = 30 AND value = 333".to_string()))
            .await
            .unwrap(),
        1,
        "unmatched source row must be inserted"
    );
}

/// A delete whose source contains duplicate keys matching the same target
/// row applies `source_dedupe_behavior` on the indexed-scan path, exactly
/// like an update: the default `Fail` aborts (naming the ambiguous key),
/// while `FirstSeen` removes and counts the row once and reports the extra
/// match as a skipped duplicate.
#[rstest::rstest]
#[case::fail(SourceDedupeBehavior::Fail)]
#[case::first_seen(SourceDedupeBehavior::FirstSeen)]
#[tokio::test]
async fn test_indexed_merge_insert_delete_source_duplicates(
    #[case] behavior: SourceDedupeBehavior,
) {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    // Index every join column so the merge takes the indexed-scan delete path.
    let params = ScalarIndexParams::default();
    ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    // Two source rows collide on the same target key (1, 10).
    let source = record_batch!(("a", Int32, [1, 1]), ("b", Int32, [10, 10])).unwrap();

    let result = MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .source_dedupe_behavior(behavior)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(source.clone())],
            source.schema(),
        )))
        .await;

    if behavior == SourceDedupeBehavior::Fail {
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Ambiguous merge inserts") && err.contains("a = 1"),
            "Fail must abort naming the ambiguous key, got: {err}"
        );
        return;
    }

    let (updated_ds, stats) = result.unwrap();
    assert_eq!(stats.num_deleted_rows, 1);
    assert_eq!(stats.num_skipped_duplicates, 1);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 3);
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1 AND b = 10".to_string()))
            .await
            .unwrap(),
        0,
        "the matched row must be removed exactly once"
    );
}

/// The v2 plans apply the same `source_dedupe_behavior` to deletes when the
/// source has duplicate keys matching one target row — covering both
/// `FullSchemaMergeInsertExec` (`Delete + InsertAll`) and
/// `DeleteOnlyMergeInsertExec` (pure delete). No scalar index, so routing
/// stays on the v2 path.
#[rstest::rstest]
#[case::full_schema_fail(true, SourceDedupeBehavior::Fail)]
#[case::full_schema_first_seen(true, SourceDedupeBehavior::FirstSeen)]
#[case::delete_only_fail(false, SourceDedupeBehavior::Fail)]
#[case::delete_only_first_seen(false, SourceDedupeBehavior::FirstSeen)]
#[tokio::test]
async fn test_v2_merge_insert_delete_source_duplicates(
    #[case] with_insert: bool,
    #[case] behavior: SourceDedupeBehavior,
) {
    let initial = record_batch!(("a", Int32, [1, 2, 3]), ("value", Int32, [10, 20, 30])).unwrap();
    let schema = initial.schema();

    let ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    // Two source rows collide on target key a=1. With insert, a=4 is new.
    let (source, when_not_matched, expected_inserted, expected_total) = if with_insert {
        (
            record_batch!(("a", Int32, [1, 1, 4]), ("value", Int32, [99, 99, 40])).unwrap(),
            WhenNotMatched::InsertAll,
            1,
            3, // 3 - 1 deleted + 1 inserted
        )
    } else {
        (
            record_batch!(("a", Int32, [1, 1])).unwrap(),
            WhenNotMatched::DoNothing,
            0,
            2, // 3 - 1 deleted
        )
    };

    let result = MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string()])
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(when_not_matched)
        .source_dedupe_behavior(behavior)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            vec![Ok(source.clone())],
            source.schema(),
        )))
        .await;

    if behavior == SourceDedupeBehavior::Fail {
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Ambiguous merge inserts") && err.contains("a = 1"),
            "Fail must abort naming the ambiguous key, got: {err}"
        );
        return;
    }

    let (updated_ds, stats) = result.unwrap();
    assert_eq!(stats.num_deleted_rows, 1, "the matched row is removed once");
    assert_eq!(stats.num_skipped_duplicates, 1);
    assert_eq!(stats.num_inserted_rows, expected_inserted);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), expected_total);
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1".to_string()))
            .await
            .unwrap(),
        0,
        "the matched row must be removed exactly once"
    );
}

/// A partial-schema source that combines `when_matched(Delete)` with
/// `when_not_matched(InsertAll)` must succeed even when every join key is
/// indexed. The indexed-scan delete path cannot fold a delete into a
/// partial write, so this case routes to the v2 plan (which fills omitted
/// nullable target columns) instead of being rejected.
#[tokio::test]
async fn test_indexed_merge_insert_partial_schema_delete_with_insert() {
    // Target carries two nullable non-key columns; the source omits `note`.
    let full_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
        Field::new("note", DataType::Utf8, true),
    ]));
    let full_batch = RecordBatch::try_new(
        full_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            Arc::new(Int32Array::from(vec![10, 20, 10, 20])),
            Arc::new(Int32Array::from(vec![100, 200, 300, 400])),
            Arc::new(StringArray::from(vec!["w", "x", "y", "z"])),
        ],
    )
    .unwrap();

    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(full_batch)], full_schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    // Source deletes matched (1, 10) and inserts new (3, 30), omitting `note`.
    let partial_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));
    let source = RecordBatch::try_new(
        partial_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 3])),
            Arc::new(Int32Array::from(vec![10, 30])),
            Arc::new(Int32Array::from(vec![999, 333])),
        ],
    )
    .unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    // 4 - 1 deleted + 1 inserted = 4.
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 4);
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 1 AND b = 10".to_string()))
            .await
            .unwrap(),
        0,
        "matched row must be deleted, not updated"
    );
    // Inserted row carries the omitted `note` column as NULL.
    assert_eq!(
        updated_ds
            .count_rows(Some(
                "a = 3 AND b = 30 AND value = 333 AND note IS NULL".to_string()
            ))
            .await
            .unwrap(),
        1,
        "unmatched source row must be inserted with omitted column NULL-filled"
    );
}

/// Fully-indexed composite-key delete across multiple fragments, with
/// stable row ids on/off.  Exercises the indexed-scan delete commit
/// path: matched row ids are resolved to addresses (via the row-id
/// index when stable) and removed without rewriting any fragments.
/// Also covers an appended fragment that neither index covers, so the
/// delete must reach rows via the unindexed-remainder union too.
#[rstest::rstest]
#[case(true)]
#[case(false)]
#[tokio::test]
async fn test_indexed_merge_insert_composite_key_delete_multi_fragment(
    #[case] enable_stable_row_ids: bool,
) {
    let initial = record_batch!(
        ("a", Int32, [1, 1, 2, 2]),
        ("b", Int32, [10, 20, 10, 20]),
        ("value", Int32, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = initial.schema();

    // One row per fragment so the delete spans multiple fragments.
    let mut ds = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial.clone())], schema.clone()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    ds.create_index(&["a"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();
    ds.create_index(&["b"], IndexType::Scalar, None, &params, false)
        .await
        .unwrap();

    // Append a row AFTER the indices are built so it lives in a fragment
    // neither index covers.  The delete must still reach it.
    let appended = record_batch!(
        ("a", Int32, [3]),
        ("b", Int32, [30]),
        ("value", Int32, [500])
    )
    .unwrap();
    ds.append(
        RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
        None,
    )
    .await
    .unwrap();

    // Delete an indexed row (1, 10) and the unindexed appended row (3, 30).
    let source = record_batch!(("a", Int32, [1, 3]), ("b", Int32, [10, 30])).unwrap();

    let (updated_ds, stats) =
        MergeInsertBuilder::try_new(Arc::new(ds), vec!["a".to_string(), "b".to_string()])
            .unwrap()
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::DoNothing)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(source.clone())],
                source.schema(),
            )))
            .await
            .unwrap();

    assert_eq!(stats.num_deleted_rows, 2);
    assert_eq!(updated_ds.count_rows(None).await.unwrap(), 3);
    assert_eq!(
        updated_ds
            .count_rows(Some("(a = 1 AND b = 10) OR (a = 3 AND b = 30)".to_string()))
            .await
            .unwrap(),
        0,
        "both matched rows must be gone"
    );
    // Untouched rows survive with their original values.
    assert_eq!(
        updated_ds
            .count_rows(Some("a = 2 AND b = 20 AND value = 400".to_string()))
            .await
            .unwrap(),
        1,
    );
}

/// Composite-key `MapIndexExec` formats its Display so plans expose
/// every probed column, and `with_new_children` round-trips the full
/// lookup list rather than collapsing back to a single-column form.
/// Lives here (not in scalar_index.rs's tests) so the new lines don't
/// pile up in a file with pending upstream conflicts.
#[test]
fn map_index_exec_multi_lookup_plan_shape() {
    use crate::io::exec::scalar_index::{IndexLookup, MapIndexExec, ScalarIndexExec};
    use crate::utils::test::NoContextTestFixture;
    use datafusion::physical_plan::{ExecutionPlan, displayable};
    use datafusion::scalar::ScalarValue;
    use lance_index::scalar::{
        SargableQuery,
        expression::{ScalarIndexExpr, ScalarIndexSearch},
    };
    use lance_select::result::IndexExprResultWireFormat;

    let fixture = NoContextTestFixture::new();
    let dataset = Arc::new(fixture.dataset);

    let dummy_input: Arc<dyn ExecutionPlan> = Arc::new(ScalarIndexExec::new(
        dataset.clone(),
        ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "ordered".to_string(),
            index_name: "ordered_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::UInt64(Some(1)))),
            needs_recheck: false,
            fragment_bitmap: None,
        }),
        IndexExprResultWireFormat::default(),
    ));

    let lookups = vec![
        IndexLookup::new("a", "a_idx"),
        IndexLookup::new("b", "b_idx"),
    ];
    let plan: Arc<dyn ExecutionPlan> = Arc::new(MapIndexExec::new_multi(
        dataset.clone(),
        lookups,
        dummy_input.clone(),
    ));

    let rendered = format!("{}", displayable(plan.as_ref()).indent(false));
    assert!(
        rendered.contains("IndexedLookup [a, b]"),
        "multi-lookup Display must list every probed column, got: {rendered}",
    );

    let rebuilt = plan
        .replace_children(
            vec![dummy_input.clone()],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .expect("replace_children must accept exactly one child");
    let rebuilt_rendered = format!("{}", displayable(rebuilt.as_ref()).indent(false));
    assert!(
        rebuilt_rendered.contains("IndexedLookup [a, b]"),
        "with_new_children must preserve every lookup, got: {rebuilt_rendered}",
    );

    // The single-lookup convenience constructor still renders without
    // the column list, so existing EXPLAIN output is unchanged for
    // single-column joins.
    let single: Arc<dyn ExecutionPlan> = Arc::new(MapIndexExec::new(
        dataset,
        "ordered".to_string(),
        "ordered_idx".to_string(),
        dummy_input,
    ));
    let single_rendered = format!("{}", displayable(single.as_ref()).indent(false));
    assert!(
        single_rendered.contains("IndexedLookup") && !single_rendered.contains("IndexedLookup ["),
        "single-lookup Display must not include the column list, got: {single_rendered}",
    );
}

/// The probe loop orders join keys by how many distinct values the source
/// batch holds for them, so writing a composite key coarse-to-fine
/// (`["bucket", "id"]`) does not make the coarse probe run first and
/// materialize a candidate set the size of the table.
///
/// Asserted on the emitted candidate count, which is the only observable
/// that says which probe ran: `IndexMetrics` has no per-probe counter and
/// is shared across lookups. Probing the selective key first leaves 4
/// candidates and stops, because 4 is already down to the source batch
/// size; following the caller's order instead would probe `bucket` first
/// (8 candidates, no stop), then intersect down to the 2 exact matches. So
/// the larger count is the cheaper plan — one probe instead of two — and
/// the surplus is trimmed by the full-key join downstream. A regression in
/// the ordering shows up here as 2.
///
/// Sits with the other composite-key merge_insert tests, next to
/// `map_index_exec_multi_lookup_plan_shape`: probe ordering only matters on
/// the composite-key indexed path, and this is where that path is covered.
#[tokio::test]
async fn map_index_exec_probes_most_selective_key_first() {
    use crate::io::exec::scalar_index::{IndexLookup, MapIndexExec};
    use arrow_array::types::UInt64Type;
    use datafusion::physical_plan::ExecutionPlan;
    use lance_datafusion::exec::OneShotExec;

    // `id` is unique; `bucket` takes two values, so a `bucket` probe alone
    // reaches half the table while pruning almost nothing.
    let initial = record_batch!(
        (
            "id",
            Int32,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        ),
        (
            "bucket",
            Int32,
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        )
    )
    .unwrap();
    let schema = initial.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::default();
    for column in ["id", "bucket"] {
        dataset
            .create_index(
                &[column],
                IndexType::Scalar,
                Some(format!("{column}_idx")),
                &params,
                false,
            )
            .await
            .unwrap();
    }

    // Columns are in lookup order, which is the deliberately bad one: the
    // coarse key first. Only ids 0 and 2 also sit in bucket 0, so the exact
    // composite match is 2 rows.
    let probe =
        record_batch!(("bucket", Int32, [0, 0, 0, 0]), ("id", Int32, [0, 1, 2, 3])).unwrap();
    let source_rows = probe.num_rows();
    let plan = MapIndexExec::new_multi(
        Arc::new(dataset),
        vec![
            IndexLookup::new("bucket", "bucket_idx"),
            IndexLookup::new("id", "id_idx"),
        ],
        Arc::new(OneShotExec::from_batch(probe)),
    );

    let mut stream = plan
        .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
        .unwrap();
    let mut candidates = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.unwrap();
        candidates.extend(
            batch
                .column(0)
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied(),
        );
    }

    assert_eq!(
        candidates.len(),
        source_rows,
        "the selective key must be probed first and stop the loop, leaving \
             one candidate per source row; {} means the probes ran in `on` order",
        candidates.len()
    );
    // Whatever the order, the candidate set has to cover the exact matches.
    for row_addr in [0_u64, 2] {
        assert!(
            candidates.contains(&row_addr),
            "candidate set must contain exact match at row {row_addr}: {candidates:?}"
        );
    }
}

#[tokio::test]
async fn test_merge_insert_use_index() {
    let data = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col("value", array::step::<UInt32Type>());
    let data = data.into_reader_rows(RowCount::from(100), BatchCount::from(1));
    let schema = data.schema();
    let mut ds = Dataset::write(data, "memory://", None).await.unwrap();

    // Create a scalar index on id column
    let index_params = ScalarIndexParams::default();
    ds.create_index(&["id"], IndexType::Scalar, None, &index_params, false)
        .await
        .unwrap();

    let source_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 101])), // Two matches, one new
            Arc::new(UInt32Array::from(vec![999, 999, 999])),
        ],
    )
    .unwrap();

    // Test 1: use_index=false should allow explain_plan to succeed
    let merge_job_no_index =
        MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .use_index(false) // Force not using index
            .try_build()
            .unwrap();

    // With use_index=false, explain_plan should succeed even with an index present
    let plan = merge_job_no_index.explain_plan(None, false).await;
    assert!(
        plan.is_ok(),
        "explain_plan should succeed with use_index=false"
    );
    let plan_str = plan.unwrap();
    assert!(plan_str.contains("MergeInsert"));
    assert!(plan_str.contains("HashJoinExec")); // Should use hash join, not index scan

    // Test 2: use_index=true (default) should fail explain_plan with index present
    let merge_job_with_index =
        MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .use_index(true) // Explicitly set to use index (though it's the default)
            .try_build()
            .unwrap();

    // With use_index=true and an index present, explain_plan should fail
    let plan_result = merge_job_with_index.explain_plan(None, false).await;
    assert!(
        plan_result.is_err(),
        "explain_plan should fail with use_index=true when index exists"
    );

    match plan_result {
        Err(Error::NotSupported { source, .. }) => {
            assert!(source.to_string().contains("does not support explain_plan"));
        }
        _ => panic!("Expected NotSupported error"),
    }

    // Test 3: Verify actual execution works without index
    let source = Box::new(RecordBatchIterator::new(
        vec![Ok(source_batch.clone())],
        schema.clone(),
    ));
    let (result_ds, stats) = merge_job_no_index.execute_reader(source).await.unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(stats.num_inserted_rows, 1);

    // Verify the data was updated correctly
    let updated_count = result_ds
        .count_rows(Some("value = 999".to_string()))
        .await
        .unwrap();
    assert_eq!(updated_count, 3);
}

#[tokio::test]
async fn test_full_schema_upsert_fragment_bitmap() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, true),
        Field::new("value", DataType::UInt32, true),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let mut dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::step_custom::<UInt32Type>(10, 10))
        .col(
            "vec",
            array::cycle_vec(
                array::cycle::<Float32Type>(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                ]),
                Dimension::from(4),
            ),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("value_idx".to_string()),
            &scalar_params,
            true,
        )
        .await
        .unwrap();

    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let value_index = indices.iter().find(|idx| idx.name == "value_idx").unwrap();
    let vec_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

    assert_eq!(
        value_index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        vec_index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    // update keys: 2,5
    let upsert_keys = UInt32Array::from(vec![2, 5]);
    let upsert_values = UInt32Array::from(vec![200, 500]);
    let upsert_vecs = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]),
        4,
    )
    .unwrap();

    let upsert_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(upsert_keys),
            Arc::new(upsert_values),
            Arc::new(upsert_vecs),
        ],
    )
    .unwrap();

    let upsert_stream = RecordBatchStreamAdapter::new(
        schema.clone(),
        futures::stream::once(async { Ok(upsert_batch) }).boxed(),
    );

    let (updated_dataset, _stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
            .try_build()
            .unwrap()
            .execute(Box::pin(upsert_stream))
            .await
            .unwrap();

    let fragments = updated_dataset.get_fragments();
    assert_eq!(fragments.len(), 3);
}

#[tokio::test]
async fn test_sub_schema_upsert_fragment_bitmap() {
    let mut dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::step_custom::<UInt32Type>(10, 10))
        .col(
            "vec",
            array::cycle_vec(
                array::cycle::<Float32Type>(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                ]),
                Dimension::from(4),
            ),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("value_idx".to_string()),
            &scalar_params,
            true,
        )
        .await
        .unwrap();

    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let value_index = indices.iter().find(|idx| idx.name == "value_idx").unwrap();
    let vec_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

    assert_eq!(
        value_index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        vec_index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    let sub_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, true),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    let upsert_keys = UInt32Array::from(vec![2, 5]);
    let upsert_vecs = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]),
        4,
    )
    .unwrap();

    let upsert_batch = RecordBatch::try_new(
        sub_schema.clone(),
        vec![Arc::new(upsert_keys), Arc::new(upsert_vecs)],
    )
    .unwrap();

    let upsert_stream = RecordBatchStreamAdapter::new(
        sub_schema.clone(),
        futures::stream::once(async { Ok(upsert_batch) }).boxed(),
    );

    let (updated_dataset, _stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
            .try_build()
            .unwrap()
            .execute(Box::pin(upsert_stream))
            .await
            .unwrap();

    let fragments = updated_dataset.get_fragments();
    // v2 path: partial-schema upsert goes through FullSchemaMergeInsertExec
    // which writes a new fragment containing the updated rows. Fragments
    // 0 and 1 keep 2 rows each (with deletion vectors covering the
    // matched keys), fragment 2 is the new one holding the 2 updated
    // rows. The in-place RewriteColumns alternative is opt-in; see
    // `test_sub_schema_upsert_in_place_fragment_bitmap`.
    assert_eq!(fragments.len(), 3);

    let updated_indices = updated_dataset.load_indices().await.unwrap();
    // Both indices remain after the v2 upsert. The vector index still
    // covers the old fragments' non-deleted rows (the deleted rows are
    // filtered by deletion vectors), and queries that need the new
    // row values fall back to scanning the unindexed new fragment.
    // This is a behavior difference from v1, which eagerly invalidated
    // the vec index when any row in a fragment was updated.
    assert_eq!(updated_indices.len(), 2);
    let updated_value_index = updated_indices
        .iter()
        .find(|idx| idx.name == "value_idx")
        .unwrap();

    // The scalar index on `value` must still cover fragments 0 and 1 —
    // even though those fragments now carry deletion vectors, the
    // `value` column itself was not modified, so the existing index
    // entries remain valid for the rows that were not deleted.
    let value_bitmap = updated_value_index.fragment_bitmap.as_ref().unwrap();
    assert!(value_bitmap.contains(0));
    assert!(value_bitmap.contains(1));
}

/// The opt-in counterpart of `test_sub_schema_upsert_fragment_bitmap`.
/// Patching `vec` in place keeps the fragments and their ids, but it does
/// rewrite that column's data, so every index over `vec` must lose the
/// patched fragments while an index over an untouched column keeps them.
/// `fields_modified` is what drives that pruning.
#[tokio::test]
async fn test_sub_schema_upsert_in_place_fragment_bitmap() {
    let mut dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::step_custom::<UInt32Type>(10, 10))
        .col(
            "vec",
            array::cycle_vec(
                array::cycle::<Float32Type>(vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                ]),
                Dimension::from(4),
            ),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let scalar_params = ScalarIndexParams::default();
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("value_idx".to_string()),
            &scalar_params,
            true,
        )
        .await
        .unwrap();
    let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".to_string()),
            &vector_params,
            true,
        )
        .await
        .unwrap();

    let sub_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, true),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));
    let upsert_batch = RecordBatch::try_new(
        sub_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![2, 5])),
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    Float32Array::from(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]),
                    4,
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let upsert_stream = RecordBatchStreamAdapter::new(
        sub_schema.clone(),
        futures::stream::once(async { Ok(upsert_batch) }).boxed(),
    );

    let (updated_dataset, _stats) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute(Box::pin(upsert_stream))
            .await
            .unwrap();

    let fragments = updated_dataset.get_fragments();
    assert_eq!(
        fragments.len(),
        2,
        "in-place patches must not add or remove fragments"
    );
    for fragment in &fragments {
        assert!(
            fragment.metadata().deletion_file.is_none(),
            "in-place patches must not produce deletion vectors"
        );
    }

    let updated_indices = updated_dataset.load_indices().await.unwrap();
    assert_eq!(updated_indices.len(), 2);
    let value_bitmap = updated_indices
        .iter()
        .find(|idx| idx.name == "value_idx")
        .unwrap()
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .clone();
    let vec_bitmap = updated_indices
        .iter()
        .find(|idx| idx.name == "vec_idx")
        .unwrap()
        .fragment_bitmap
        .as_ref()
        .unwrap()
        .clone();

    // `value` was not patched, so its index still describes what is stored.
    assert!(value_bitmap.contains(0));
    assert!(value_bitmap.contains(1));
    // `vec` was rewritten in both fragments, so its index no longer does.
    assert!(
        vec_bitmap.is_empty(),
        "index over a patched column must be invalidated for the patched fragments, got {:?}",
        vec_bitmap.iter().collect::<Vec<_>>()
    );
}

/// Creates a 3-fragment dataset (100 rows each) with columns (id: Utf8, category: Utf8,
/// value_a: Float64, value_b: Float64) and a BTree index on `id`.
///
/// Fragment 0: id-0000..id-0099
/// Fragment 1: id-0100..id-0199
/// Fragment 2: id-0200..id-0299
async fn create_indexed_3frag_dataset() -> Arc<Dataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
        Field::new("value_b", DataType::Float64, false),
    ]));

    let make_batch = |frag_idx: usize| {
        let start = frag_idx * 100;
        let ids: Vec<String> = (start..start + 100).map(|j| format!("id-{j:04}")).collect();
        let categories: Vec<&str> = vec!["A"; 100];
        let value_a: Vec<f64> = (0..100)
            .map(|i| i as f64 + frag_idx as f64 * 100.0)
            .collect();
        let value_b: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(categories)),
                Arc::new(Float64Array::from(value_a)),
                Arc::new(Float64Array::from(value_b)),
            ],
        )
        .unwrap()
    };

    // Write first fragment
    let batch0 = make_batch(0);
    let reader = Box::new(RecordBatchIterator::new([Ok(batch0)], schema.clone()));
    let mut ds = Dataset::write(reader, "memory://indexed_3frag", None)
        .await
        .unwrap();

    // Append fragments 1 and 2
    for frag_idx in 1..3 {
        let batch = make_batch(frag_idx);
        let reader = Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()));
        ds.append(reader, None).await.unwrap();
    }

    // Create BTree index on id
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    Arc::new(ds)
}

// Regression test: partial-schema merge_insert followed by another partial merge_insert
// on the same rows should not produce "Ambiguous merge inserts" errors.
//
// The bug: the first partial merge_insert drops the touched fragment from the index bitmap
// but leaves stale btree entries. The second merge_insert finds the same rows via both
// the stale btree lookup AND the unindexed fragment scan, causing duplicates.
#[tokio::test]
async fn test_partial_merge_insert_stale_index_ambiguous() {
    let dataset = create_indexed_3frag_dataset().await;

    // Step 2: Partial merge_insert on fragment 1 rows -> fragment 1 drops from bitmap
    let dataset = partial_merge_insert(dataset, 100..200, 999.0).await;

    // Step 3: Another partial merge_insert on the same rows.
    // This should succeed, not fail with "Ambiguous merge inserts".
    let dataset = partial_merge_insert(dataset, 100..200, 888.0).await;

    // Verify correctness: all 300 rows present, updated values correct
    let batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let all_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
        Field::new("value_b", DataType::Float64, false),
    ]));
    let combined = concat_batches(&all_schema, &batches).unwrap();
    assert_eq!(combined.num_rows(), 300);

    // Check the updated rows have value_a = 888.0
    let result = dataset
        .scan()
        .filter("id >= 'id-0100' AND id < 'id-0200'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let result = concat_batches(&all_schema, &result).unwrap();
    assert_eq!(result.num_rows(), 100);
    let values = result
        .column_by_name("value_a")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..100 {
        assert_eq!(values.value(i), 888.0, "row {i} should have value_a=888.0");
    }
}

// Regression test for GitHub issue #6877.
//
// Two sequential full-schema merge_insert UpdateAll calls against the same
// target row, on a dataset with stable_row_ids enabled and a BTREE scalar
// index on the join column, used to fail on the second call with
// "Ambiguous merge inserts are prohibited" — even though each call's
// source had exactly one row per key.
//
// Mechanism: with stable row ids the BTREE stores stable_row_ids (not
// physical addresses). After the first merge_insert, A's stable_row_id is
// preserved but its physical home moves to an unindexed fragment. The
// BTREE-side TakeExec resolves the stable_row_id to A's new location and
// emits a row; the unindexed-fragments scan also covers the new fragment
// and emits the same logical row. Both surface the same `_rowid`, so the
// merge_insert source-dedup HashSet sees a duplicate and aborts.
//
// Fix: thread `restrict_to_fragments` into `do_create_deletion_mask_row_id`
// so the allow-list only contains stable_row_ids whose current physical
// home is inside the index's fragment_bitmap.
#[tokio::test]
async fn test_issue_6877_repeated_merge_insert_stable_row_ids() {
    use arrow_array::Int32Array;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["A", "B", "C"])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();

    let mut ds = Dataset::write(
        Box::new(RecordBatchIterator::new([Ok(initial)], schema.clone())),
        "memory://test_6877",
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    ds.create_index(
        &["id"],
        IndexType::Scalar,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();

    // First merge_insert: A 1 -> 11.
    let update_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(Int32Array::from(vec![11])),
        ],
    )
    .unwrap();
    let (ds, _) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(update_a)],
            schema.clone(),
        )))
        .await
        .unwrap();

    // Second merge_insert: A 11 -> 22. Used to fail before the fix.
    let update_a_again = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(Int32Array::from(vec![22])),
        ],
    )
    .unwrap();
    let (ds, _) = MergeInsertBuilder::try_new(ds, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(update_a_again)],
            schema.clone(),
        )))
        .await
        .unwrap();

    // Sanity check: A's value is now 22.
    let batches = ds
        .scan()
        .filter("id = 'A'")
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let combined = concat_batches(&schema, &batches).unwrap();
    assert_eq!(combined.num_rows(), 1);
    let values = combined
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(values.value(0), 22);
}

// Regression test: partial-schema merge_insert followed by update (deleting all rows
// in a fragment) followed by partial merge_insert should not produce
// "fragment id N does not exist" errors.
//
// The bug: stale btree entries reference the deleted fragment. The deletion mask doesn't
// block those addresses because the fragment isn't in the index bitmap. TakeExec tries
// to read from a non-existent fragment.
#[tokio::test]
async fn test_partial_merge_insert_stale_index_fragment_not_exist() {
    let dataset = create_indexed_3frag_dataset().await;

    // Step 2: Partial merge_insert on fragment 1 rows -> fragment 1 drops from bitmap
    let dataset = partial_merge_insert(dataset, 100..200, 999.0).await;

    // Step 3: Update all rows that were in fragment 1, causing fragment 1 to be
    // fully deleted and replaced by a new fragment.
    let update_result = crate::dataset::UpdateBuilder::new(Arc::new((*dataset).clone()))
        .update_where("id >= 'id-0100' AND id < 'id-0200'")
        .unwrap()
        .set("category", "'B'")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    let dataset = update_result.new_dataset;

    // Step 4: Partial merge_insert on the same rows.
    // This should succeed, not fail with "fragment does not exist".
    let dataset = partial_merge_insert(dataset, 100..200, 888.0).await;

    // Verify correctness
    let batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let all_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
        Field::new("value_b", DataType::Float64, false),
    ]));
    let combined = concat_batches(&all_schema, &batches).unwrap();
    assert_eq!(combined.num_rows(), 300);
}

// Regression test: partial-schema merge_insert followed by update (deleting SOME rows
// in a fragment) followed by partial merge_insert should not produce
// "RecordBatch size mismatch" errors.
//
// The bug: stale btree entries reference deleted rows in a fragment that still exists.
// The deletion mask doesn't block those addresses (fragment not in bitmap). TakeExec
// reads the fragment but the rows have deletion markers, returning 0 rows where N
// were expected.
#[tokio::test]
async fn test_partial_merge_insert_stale_index_batch_size_mismatch() {
    let dataset = create_indexed_3frag_dataset().await;

    // Step 2: Partial merge_insert on fragment 1 rows -> fragment 1 drops from bitmap
    let dataset = partial_merge_insert(dataset, 100..200, 999.0).await;

    // Step 3: Update HALF of the rows that were in fragment 1. Fragment 1 survives
    // but the updated rows are deleted from it (moved to a new fragment).
    let update_result = crate::dataset::UpdateBuilder::new(Arc::new((*dataset).clone()))
        .update_where("id >= 'id-0100' AND id < 'id-0150'")
        .unwrap()
        .set("category", "'B'")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    let dataset = update_result.new_dataset;

    // Step 4: Partial merge_insert targeting the rows that were updated (and thus
    // deleted from fragment 1). Should succeed, not fail with batch size mismatch.
    let dataset = partial_merge_insert(dataset, 100..150, 888.0).await;

    // Verify correctness
    let batches = dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let all_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
        Field::new("value_b", DataType::Float64, false),
    ]));
    let combined = concat_batches(&all_schema, &batches).unwrap();
    assert_eq!(combined.num_rows(), 300);
}

// Regression test: after a partial-schema merge_insert drops a fragment from the vector
// index bitmap, a vector search should not return duplicate rows. The stale vector index
// data still references the dropped fragment, and the scanner also flat-scans unindexed
// fragments, causing the same rows to appear from both paths.
#[tokio::test]
async fn test_partial_merge_insert_stale_vector_index_duplicates() {
    let dim = 4i32;
    let rows_per_frag = 10usize;
    let num_frags = 3usize;
    let total_rows = rows_per_frag * num_frags;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    let make_batch = |frag_idx: usize, offset: f32| {
        let start = frag_idx * rows_per_frag;
        let ids: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("id-{j:04}"))
            .collect();
        let cats: Vec<&str> = vec!["A"; rows_per_frag];
        let values: Vec<f32> = (0..rows_per_frag * dim as usize)
            .map(|i| (start * dim as usize + i) as f32 + offset)
            .collect();
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim).unwrap();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(cats)),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    };

    // Write 3 fragments
    let batch0 = make_batch(0, 0.0);
    let reader = Box::new(RecordBatchIterator::new([Ok(batch0)], schema.clone()));
    let mut ds = Dataset::write(reader, "memory://vector_stale_test", None)
        .await
        .unwrap();
    for frag_idx in 1..num_frags {
        let batch = make_batch(frag_idx, 0.0);
        let reader = Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()));
        ds.append(reader, None).await.unwrap();
    }

    // Create IVF_FLAT vector index on vec
    let params = VectorIndexParams::ivf_flat(1, MetricType::L2);
    ds.create_index(&["vec"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let ds = Arc::new(ds);

    // Partial merge_insert with (id, vec) on fragment 1 rows - slightly different vectors.
    // This drops fragment 1 from the vector index bitmap.
    let frag1_start = rows_per_frag;
    let ids: Vec<String> = (frag1_start..frag1_start + rows_per_frag)
        .map(|j| format!("id-{j:04}"))
        .collect();
    let sub_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));
    let values: Vec<f32> = (0..rows_per_frag * dim as usize)
        .map(|i| (frag1_start * dim as usize + i) as f32 + 0.5)
        .collect();
    let vectors = FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim).unwrap();
    let update_batch = RecordBatch::try_new(
        sub_schema.clone(),
        vec![Arc::new(StringArray::from(ids)), Arc::new(vectors)],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(update_batch)], sub_schema));
    let (ds, _) = MergeInsertBuilder::try_new(ds, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();

    // KNN search with k = total_rows to retrieve all rows
    let query: Float32Array = (0..dim)
        .map(|i| (frag1_start * dim as usize + i as usize) as f32 + 0.5)
        .collect();
    let results = ds
        .scan()
        .nearest("vec", &query, total_rows)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Check no duplicate ids
    let ids = results
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let unique_ids: std::collections::HashSet<&str> =
        (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(
        unique_ids.len(),
        ids.len(),
        "Found duplicate ids in KNN results: {} unique out of {} total",
        unique_ids.len(),
        ids.len()
    );
}

// Regression test: after a partial-schema merge_insert drops a fragment from the FTS
// index bitmap, a full text search should not return duplicate rows. The stale inverted
// index data still references the dropped fragment, and the scanner also flat-scans
// unindexed fragments, causing the same rows to appear from both paths.
#[tokio::test]
async fn test_partial_merge_insert_stale_fts_index_duplicates() {
    let rows_per_frag = 10usize;
    let num_frags = 3usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let make_batch = |frag_idx: usize| {
        let start = frag_idx * rows_per_frag;
        let ids: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("id-{j:04}"))
            .collect();
        let cats: Vec<&str> = vec!["A"; rows_per_frag];
        // Every row contains "common" so we can search for it and expect all rows
        let texts: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("common unique{j:04}"))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(cats)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap()
    };

    // Write 3 fragments
    let batch0 = make_batch(0);
    let reader = Box::new(RecordBatchIterator::new([Ok(batch0)], schema.clone()));
    let mut ds = Dataset::write(reader, "memory://fts_stale_test", None)
        .await
        .unwrap();
    for frag_idx in 1..num_frags {
        let batch = make_batch(frag_idx);
        let reader = Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()));
        ds.append(reader, None).await.unwrap();
    }

    // Create inverted index on text
    let params = InvertedIndexParams::default();
    ds.create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    let ds = Arc::new(ds);

    // Partial merge_insert with (id, text) on fragment 1 rows.
    // Text still contains "common" so FTS will find them via both paths.
    // This drops fragment 1 from the inverted index bitmap.
    let frag1_start = rows_per_frag;
    let ids: Vec<String> = (frag1_start..frag1_start + rows_per_frag)
        .map(|j| format!("id-{j:04}"))
        .collect();
    let texts: Vec<String> = (frag1_start..frag1_start + rows_per_frag)
        .map(|j| format!("common updated{j:04}"))
        .collect();
    let sub_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let update_batch = RecordBatch::try_new(
        sub_schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(update_batch)], sub_schema));
    let (ds, _) = MergeInsertBuilder::try_new(ds, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();

    // FTS search for "common" — every row should match exactly once
    let query = FullTextSearchQuery::new("common".to_string());
    let results = ds
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Check no duplicate ids
    let ids = results
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let unique_ids: std::collections::HashSet<&str> =
        (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(
        unique_ids.len(),
        ids.len(),
        "Found duplicate ids in FTS results: {} unique out of {} total",
        unique_ids.len(),
        ids.len()
    );
    // Also verify we got all rows
    assert_eq!(
        unique_ids.len(),
        rows_per_frag * num_frags,
        "Expected {} rows but got {}",
        rows_per_frag * num_frags,
        unique_ids.len()
    );
}

// Companion regression test for issue #6877 on the FTS path.
//
// The FTS prefilter shares `do_create_deletion_mask_row_id` with the
// scalar-index path, so the same stable-row-id bypass that produced
// duplicate rows in merge_insert can produce duplicate hits in FTS search
// after a merge_insert moves rows to unindexed fragments. This test pins
// the contract for the FTS consumer.
#[tokio::test]
async fn test_issue_6877_fts_no_duplicates_stable_row_ids() {
    let rows_per_frag = 10usize;
    let num_frags = 3usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let make_batch = |frag_idx: usize| {
        let start = frag_idx * rows_per_frag;
        let ids: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("id-{j:04}"))
            .collect();
        let texts: Vec<String> = (start..start + rows_per_frag)
            .map(|j| format!("common unique{j:04}"))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap()
    };

    let batch0 = make_batch(0);
    let reader = Box::new(RecordBatchIterator::new([Ok(batch0)], schema.clone()));
    let mut ds = Dataset::write(
        reader,
        "memory://fts_stable_row_id_test",
        Some(WriteParams {
            mode: WriteMode::Overwrite,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for frag_idx in 1..num_frags {
        let batch = make_batch(frag_idx);
        let reader = Box::new(RecordBatchIterator::new([Ok(batch)], schema.clone()));
        ds.append(reader, None).await.unwrap();
    }

    let params = InvertedIndexParams::default();
    ds.create_index(&["text"], IndexType::Inverted, None, &params, true)
        .await
        .unwrap();

    // Full-schema merge_insert rewriting fragment 1's rows. After this,
    // the original locations are tombstoned and the new locations live in
    // a new (unindexed) fragment; the stable_row_ids are preserved.
    let frag1_start = rows_per_frag;
    let ids: Vec<String> = (frag1_start..frag1_start + rows_per_frag)
        .map(|j| format!("id-{j:04}"))
        .collect();
    let texts: Vec<String> = (frag1_start..frag1_start + rows_per_frag)
        .map(|j| format!("common updated{j:04}"))
        .collect();
    let update_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(texts)),
        ],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(update_batch)], schema.clone()));
    let (ds, _) = MergeInsertBuilder::try_new(Arc::new(ds), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();

    // FTS search for "common" — every row should match exactly once.
    let query = FullTextSearchQuery::new("common".to_string());
    let results = ds
        .scan()
        .full_text_search(query)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let ids = results
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let unique_ids: std::collections::HashSet<&str> =
        (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(
        unique_ids.len(),
        ids.len(),
        "Found duplicate ids in FTS results: {} unique out of {} total",
        unique_ids.len(),
        ids.len()
    );
    assert_eq!(
        unique_ids.len(),
        rows_per_frag * num_frags,
        "Expected {} rows but got {}",
        rows_per_frag * num_frags,
        unique_ids.len()
    );
}

/// Returns rows only from its first scan, simulating a provider whose
/// snapshot changes between scans.
#[derive(Debug)]
struct ChangingScanProvider {
    first_batch: RecordBatch,
    scan_count: Arc<AtomicUsize>,
}

#[async_trait]
impl TableProvider for ChangingScanProvider {
    fn schema(&self) -> Arc<Schema> {
        self.first_batch.schema()
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        datafusion::logical_expr::TableType::Base
    }

    async fn scan(
        &self,
        session: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let schema = self.first_batch.schema();
        let scan_number = self.scan_count.fetch_add(1, Ordering::SeqCst);
        let batch = if scan_number == 0 {
            self.first_batch.clone()
        } else {
            RecordBatch::new_empty(schema.clone())
        };
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter([Ok(batch)]),
        ));
        let partition = Arc::new(OneShotPartitionStream::new(stream));
        let provider = StreamingTable::try_new(schema, vec![partition])?;
        provider.scan(session, projection, filters, limit).await
    }
}

#[tokio::test]
async fn test_indexed_merge_provider_scans_must_share_a_snapshot() {
    let test_dir = TempStrDir::default();
    let initial = record_batch!(("id", UInt32, [2]), ("value", UInt32, [20])).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(initial.clone())], initial.schema()),
        test_dir.as_str(),
        None,
    )
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
    let dataset = Arc::new(dataset);
    let source = record_batch!(("id", UInt32, [2]), ("value", UInt32, [200])).unwrap();
    let scan_count = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn TableProvider> = Arc::new(ChangingScanProvider {
        first_batch: source,
        scan_count: scan_count.clone(),
    });

    let (dataset, stats) = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_provider(provider)
        .await
        .unwrap();

    assert_eq!(scan_count.load(Ordering::SeqCst), 1);
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 0);
    drop(dataset);

    let reopened = Dataset::open(test_dir.as_str()).await.unwrap();
    let batch = reopened.scan().try_into_batch().await.unwrap();
    let ids = batch["id"].as_primitive::<UInt32Type>();
    let values = batch["value"].as_primitive::<UInt32Type>();
    let mut rows = ids
        .values()
        .iter()
        .zip(values.values())
        .map(|(id, value)| (*id, *value))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![(2, 200)]);
}

#[rstest::rstest]
#[case::single_key(vec!["key".to_string()], Some(1))]
#[case::null_key(vec!["key".to_string()], None)]
#[case::composite_key(vec!["key".to_string(), "other_key".to_string()], Some(1))]
#[tokio::test]
async fn test_indexed_merge_insert_overlaid_keys(
    #[case] on: Vec<String>,
    #[case] base_key: Option<i32>,
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] use_index: bool,
) {
    let initial = record_batch!(
        ("key", Int32, [Some(0), base_key, Some(2), Some(3)]),
        ("other_key", Int32, [10, 20, 30, 40]),
        ("tag", Utf8, ["base0", "base1", "base2", "base3"])
    )
    .unwrap();
    let schema = initial.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(initial)], schema.clone()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            data_storage_version: Some(LanceFileVersion::V2_1),
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for column in &on {
        dataset
            .create_index(
                &[column],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }

    let appended = record_batch!(
        ("key", Int32, [4, 5]),
        ("other_key", Int32, [50, 60]),
        ("tag", Utf8, ["base4", "base5"])
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(appended)], schema.clone()),
        WriteDestination::Dataset(Arc::new(dataset)),
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    // Deleted offsets must not shift the overlay's physical row selection.
    dataset.delete("key = 0").await.unwrap();
    let dataset = Arc::new(dataset);
    let key_schema = dataset
        .schema()
        .project_by_ids(&[dataset.schema().field("key").unwrap().id], true);
    let mut groups = Vec::new();
    for (fragment_id, offsets, keys) in [
        (0, vec![1], vec![100]),
        // The unchanged key must not be returned by both the index and stale-row scan.
        (1, vec![0, 1], vec![101, 3]),
        // An overlay on an unindexed fragment must still be scanned only once.
        (2, vec![0], vec![200]),
    ] {
        let row_addrs = offsets
            .into_iter()
            .map(|offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
            .collect::<Vec<_>>();
        let batch = record_batch!(("_rowaddr", UInt64, row_addrs), ("key", Int32, keys)).unwrap();
        let mut writer = dataset
            .get_fragment(fragment_id as usize)
            .unwrap()
            .write_overlay(&key_schema)
            .await
            .unwrap();
        writer.write_batch(&batch).await.unwrap();
        groups.push(writer.finish().await.unwrap().unwrap());
    }
    let read_version = dataset.version().version;
    let dataset = Dataset::commit(
        WriteDestination::Dataset(dataset),
        Operation::DataOverlay { groups },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();

    let source = record_batch!(
        ("key", Int32, [100, 101, 3, 200, 999]),
        ("other_key", Int32, [20, 30, 40, 50, 99]),
        (
            "tag",
            Utf8,
            ["merged", "merged", "merged", "merged", "inserted"]
        )
    )
    .unwrap();
    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), on)
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .use_index(use_index)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(source.slice(0, 2)), Ok(source.slice(2, 3))],
            schema,
        )))
        .await
        .unwrap();

    assert_eq!(stats.num_updated_rows, 4);
    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
    for (key, other_key, tag) in [
        (100, 20, "merged"),
        (101, 30, "merged"),
        (3, 40, "merged"),
        (200, 50, "merged"),
        (5, 60, "base5"),
        (999, 99, "inserted"),
    ] {
        assert_eq!(
            dataset
                .count_rows(Some(format!(
                    "key = {key} AND other_key = {other_key} AND tag = '{tag}'"
                )))
                .await
                .unwrap(),
            1,
        );
    }
}
