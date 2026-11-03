// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;
use rstest::rstest;

async fn dataset(uri: &str) -> Dataset {
    let params = WriteParams {
        enable_stable_row_ids: true,
        max_rows_per_file: 1,
        ..Default::default()
    };
    let initial =
        record_batch!(("key", UInt32, [0, 1]), ("value", UInt32, [Some(10), None])).unwrap();
    let mut dataset = InsertBuilder::new(uri)
        .with_params(&params)
        .execute(vec![initial])
        .await
        .unwrap();
    let appended =
        record_batch!(("key", UInt32, [2, 3]), ("value", UInt32, [Some(20), None])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
            Some(params),
        )
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 4);
    dataset
}
async fn rows(dataset: &Dataset) -> RecordBatch {
    dataset
        .scan()
        .project(&[
            "key",
            "value",
            ROW_CREATED_AT_VERSION,
            ROW_LAST_UPDATED_AT_VERSION,
        ])
        .unwrap()
        .with_row_id()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("key".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
}
type Row = (u64, Option<u32>, Option<u32>, u64, u64);
fn row_values(batch: &RecordBatch) -> Vec<Row> {
    (0..batch.num_rows())
        .map(|i| {
            (
                batch[ROW_ID].as_primitive::<UInt64Type>().value(i),
                (!batch["key"].is_null(i))
                    .then(|| batch["key"].as_primitive::<UInt32Type>().value(i)),
                (!batch["value"].is_null(i))
                    .then(|| batch["value"].as_primitive::<UInt32Type>().value(i)),
                batch[ROW_CREATED_AT_VERSION]
                    .as_primitive::<UInt64Type>()
                    .value(i),
                batch[ROW_LAST_UPDATED_AT_VERSION]
                    .as_primitive::<UInt64Type>()
                    .value(i),
            )
        })
        .collect()
}
async fn check_rows(dataset: &Dataset, expected: &[Row]) {
    let mut actual = row_values(&rows(dataset).await);
    actual.sort_by_key(|row| row.0);
    assert_eq!(actual, expected);
    // Taking in reverse id order exercises a different reader path and
    // verifies that hidden lineage columns follow the requested rows.
    let ids = expected.iter().rev().map(|row| row.0).collect::<Vec<_>>();
    let taken = dataset
        .take_rows(
            &ids,
            ProjectionRequest::from_columns(
                [
                    "key",
                    "value",
                    ROW_ID,
                    ROW_ADDR,
                    ROW_CREATED_AT_VERSION,
                    ROW_LAST_UPDATED_AT_VERSION,
                ],
                dataset.schema(),
            ),
        )
        .await
        .unwrap();
    assert_eq!(
        row_values(&taken),
        expected.iter().rev().copied().collect::<Vec<_>>()
    );
    dataset.validate().await.unwrap();
}
fn job(dataset: Arc<Dataset>) -> MergeInsertJob {
    MergeInsertBuilder::try_new(dataset, vec!["key".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .write_mode(MergeInsertWriteMode::RewriteRows)
        .use_index(false)
        .try_build()
        .unwrap()
}
#[rstest]
#[case::update_only(vec![0, 2], true)]
#[case::mixed(vec![0, 2, 4], true)]
#[case::insert_only(vec![4], true)]
#[case::default_off(vec![0, 2, 4], false)]
#[tokio::test]
async fn merge_insert_spills_carried_lineage(#[case] keys: Vec<u32>, #[case] spill: bool) {
    let mut dataset = dataset("memory://").await;
    if spill {
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
    }
    let source = RecordBatch::try_from_iter([
        ("key", Arc::new(UInt32Array::from(keys.clone())) as ArrayRef),
        (
            "value",
            Arc::new(UInt32Array::from(vec![None; keys.len()])) as ArrayRef,
        ),
    ])
    .unwrap();
    let dataset = Arc::new(dataset);
    let job = MergeInsertBuilder::try_new(dataset.clone(), vec!["key".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .write_mode(MergeInsertWriteMode::RewriteRows)
        .use_index(false)
        .try_build()
        .unwrap();
    assert!(
        job.can_use_create_plan(source.schema().as_ref())
            .await
            .unwrap()
    );
    let prepared = job.execute_uncommitted_batches(vec![source]).await.unwrap();
    let Operation::Update { new_fragments, .. } = &prepared.transaction.operation else {
        panic!("expected a V2 row rewrite");
    };
    let updates = keys.iter().filter(|key| **key < 4).count();
    assert_eq!(
        new_fragments.iter().any(|f| f.has_spilled_row_lineage()),
        spill && updates > 0,
    );
    for fragment in new_fragments {
        assert!(fragment.last_updated_at_version_meta.is_none());
        if fragment.has_spilled_row_lineage() {
            assert!(matches!(fragment.row_id_meta, Some(RowIdMeta::Column)));
            assert!(matches!(
                fragment.created_at_version_meta,
                Some(RowDatasetVersionMeta::Column)
            ));
        }
    }
    let merged = CommitBuilder::new(dataset)
        .execute(prepared.transaction)
        .await
        .unwrap();
    let batch = rows(&merged).await;
    assert_eq!(batch.num_rows(), 4 + usize::from(keys.contains(&4)));
    for i in 0..batch.num_rows() {
        let key = batch["key"].as_primitive::<UInt32Type>().value(i);
        assert_eq!(
            batch[ROW_ID].as_primitive::<UInt64Type>().value(i),
            key as u64
        );
        let created = if key < 2 {
            1
        } else if key < 4 {
            2
        } else {
            merged.version().version
        };
        assert_eq!(
            batch[ROW_CREATED_AT_VERSION]
                .as_primitive::<UInt64Type>()
                .value(i),
            created
        );
        assert_eq!(
            batch[ROW_LAST_UPDATED_AT_VERSION]
                .as_primitive::<UInt64Type>()
                .value(i),
            if keys.contains(&key) {
                merged.version().version
            } else {
                created
            }
        );
        assert_eq!(
            batch["value"].is_null(i),
            keys.contains(&key) || key % 2 == 1
        );
    }
    merged.validate().await.unwrap();
}
#[rstest]
#[case::explicitly_disabled("false", "0")]
#[case::under_budget("true", "204800")]
#[tokio::test]
async fn merge_insert_keeps_small_or_disabled_lineage_inline(
    #[case] enabled: &str,
    #[case] budget: &str,
) {
    let mut dataset = dataset("memory://").await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, enabled),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, budget),
        ])
        .await
        .unwrap();
    let source = record_batch!(
        ("key", UInt32, [0, 2, 4]),
        ("value", UInt32, [100, 200, 400])
    )
    .unwrap();
    let dataset = Arc::new(dataset);
    let prepared = job(dataset.clone())
        .execute_uncommitted_batches(vec![source])
        .await
        .unwrap();
    let Operation::Update { new_fragments, .. } = &prepared.transaction.operation else {
        panic!("expected update")
    };
    for fragment in new_fragments {
        assert!(!fragment.has_spilled_row_lineage());
        assert!(fragment.created_at_version_meta.is_none());
        assert!(fragment.last_updated_at_version_meta.is_none());
    }
    let merged = CommitBuilder::new(dataset)
        .execute(prepared.transaction)
        .await
        .unwrap();
    let version = merged.version().version;
    check_rows(
        &merged,
        &[
            (0, Some(0), Some(100), 1, version),
            (1, Some(1), None, 1, 1),
            (2, Some(2), Some(200), 2, version),
            (3, Some(3), None, 2, 2),
            (4, Some(4), Some(400), version, version),
        ],
    )
    .await;
}
/// A concurrent append consumes the ids predicted by the first attempt.
/// The commit must allocate different ids and creation versions to the
/// insert group while keeping the spilled update group's lineage.
#[tokio::test]
async fn merge_insert_spill_rebases_insert_lineage() {
    let mut dataset = dataset("memory://").await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
    let source = record_batch!(
        ("key", UInt32, [2, 4, 0]),
        ("value", UInt32, [200, 400, 100])
    )
    .unwrap();
    let original = Arc::new(dataset.clone());
    let prepared = job(original.clone())
        .execute_uncommitted_batches(vec![source])
        .await
        .unwrap();
    let Operation::Update { new_fragments, .. } = &prepared.transaction.operation else {
        panic!("expected update")
    };
    assert_eq!(new_fragments.len(), 2);
    assert!(new_fragments[0].has_spilled_row_lineage());
    assert!(new_fragments[1].row_id_meta.is_none());
    assert!(new_fragments[1].created_at_version_meta.is_none());
    let appended = record_batch!(("key", UInt32, [8]), ("value", UInt32, [800])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    let append_version = dataset.version().version;
    let merged = CommitBuilder::new(original)
        .execute(prepared.transaction)
        .await
        .unwrap();
    let version = merged.version().version;
    assert_eq!(version, append_version + 1);
    check_rows(
        &merged,
        &[
            (0, Some(0), Some(100), 1, version),
            (1, Some(1), None, 1, 1),
            (2, Some(2), Some(200), 2, version),
            (3, Some(3), None, 2, 2),
            (4, Some(8), Some(800), append_version, append_version),
            (5, Some(4), Some(400), version, version),
        ],
    )
    .await;
}
/// The competing merge turns a source-only row into an existing row.
/// A real conflict retry must reclassify it and carry its new creation
/// version, including when its neighboring updates were already spilled.
#[rstest]
#[case::keep_spilling(true)]
#[case::stop_spilling(false)]
#[tokio::test]
async fn merge_insert_spill_retry_and_repeated_operations(#[case] keep_spilling: bool) {
    let directory = TempStrDir::default();
    let mut dataset = dataset(directory.as_str()).await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
    let original = Arc::new(dataset);
    let stale_job = job(original.clone());
    let concurrent = record_batch!(("key", UInt32, [0, 4]), ("value", UInt32, [90, 94])).unwrap();
    let (other, _) = job(original)
        .execute_batches(vec![concurrent])
        .await
        .unwrap();
    let created_four = other.version().version;
    assert!(
        other
            .fragments()
            .iter()
            .any(|f| f.has_spilled_row_lineage())
    );
    let source = record_batch!(
        ("key", UInt32, [2, 4, 0]),
        ("value", UInt32, [200, 400, 100])
    )
    .unwrap();
    let (merged, stats) = stale_job.execute_batches(vec![source]).await.unwrap();
    assert_eq!(stats.num_attempts, 2);
    assert_eq!(stats.num_updated_rows, 3);
    assert_eq!(stats.num_inserted_rows, 0);
    let first_version = merged.version().version;
    let merged = if keep_spilling {
        merged
    } else {
        let mut disabled = merged.as_ref().clone();
        disabled
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "false")])
            .await
            .unwrap();
        Arc::new(disabled)
    };
    let source = record_batch!(
        ("key", UInt32, [4, 2]),
        ("value", UInt32, [None, Some(222)])
    )
    .unwrap();
    let (merged, _) = job(merged).execute_batches(vec![source]).await.unwrap();
    let version = merged.version().version;
    let expected = vec![
        (0, Some(0), Some(100), 1, first_version),
        (1, Some(1), None, 1, 1),
        (2, Some(2), Some(222), 2, version),
        (3, Some(3), None, 2, 2),
        (4, Some(4), None, created_four, version),
    ];
    check_rows(&merged, &expected).await;
    let mut compacted = merged.as_ref().clone();
    compact_files(
        &mut compacted,
        CompactionOptions {
            target_rows_per_fragment: 5,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let reopened = Dataset::open(directory.as_str()).await.unwrap();
    check_rows(&reopened, &expected).await;
    let updated = UpdateBuilder::new(Arc::new(reopened))
        .update_where("key = 4")
        .unwrap()
        .set("value", "444")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    let mut expected = expected;
    expected[4].2 = Some(444);
    expected[4].4 = updated.new_dataset.version().version;
    check_rows(&updated.new_dataset, &expected).await;
}
#[tokio::test]
async fn merge_insert_spill_preserves_lineage_through_deferred_condition() {
    let mut dataset = dataset("memory://").await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
    let source = record_batch!(
        ("value", UInt32, [Some(100), Some(11), None, Some(400)]),
        ("key", UInt32, [0, 1, 2, 4])
    )
    .unwrap();
    let (merged, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateIf("source.value > target.value".into()))
        .when_not_matched(WhenNotMatched::InsertAll)
        .write_mode(MergeInsertWriteMode::RewriteRows)
        .use_index(false)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    let version = merged.version().version;
    check_rows(
        &merged,
        &[
            (0, Some(0), Some(100), 1, version),
            (1, Some(1), None, 1, 1),
            (2, Some(2), Some(20), 2, 2),
            (3, Some(3), None, 2, 2),
            (4, Some(4), Some(400), version, version),
        ],
    )
    .await;
}
#[rstest]
#[case::inner(false, false)]
#[case::right(true, false)]
#[case::left(false, true)]
#[case::full(true, true)]
#[tokio::test]
async fn merge_insert_spill_preserves_nulls_across_joins(
    #[case] insert: bool,
    #[case] delete_unmatched: bool,
    #[values(false, true)] partial_source: bool,
) {
    let initial = record_batch!(
        ("key", UInt32, [Some(0), None, Some(2), Some(3)]),
        ("value", UInt32, [10, 11, 20, 30])
    )
    .unwrap();
    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        })
        .execute(vec![initial])
        .await
        .unwrap();
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
    let source = record_batch!(
        ("value", UInt32, [None, Some(900), Some(100)]),
        ("key", UInt32, [Some(2), None, Some(0)])
    )
    .unwrap();
    let source = if partial_source {
        source.project(&[1]).unwrap()
    } else {
        source
    };
    let (merged, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["key".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(if insert {
            WhenNotMatched::InsertAll
        } else {
            WhenNotMatched::DoNothing
        })
        .when_not_matched_by_source(if delete_unmatched {
            WhenNotMatchedBySource::Delete
        } else {
            WhenNotMatchedBySource::Keep
        })
        .write_mode(MergeInsertWriteMode::RewriteRows)
        .use_index(false)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(stats.num_inserted_rows, u64::from(insert));
    assert_eq!(stats.num_deleted_rows, if delete_unmatched { 2 } else { 0 });
    assert!(
        merged
            .fragments()
            .iter()
            .any(|fragment| fragment.has_spilled_row_lineage())
    );
    let version = merged.version().version;
    let mut expected = vec![
        (
            0,
            Some(0),
            Some(if partial_source { 10 } else { 100 }),
            1,
            version,
        ),
        (
            2,
            Some(2),
            if partial_source { Some(20) } else { None },
            1,
            version,
        ),
    ];
    if !delete_unmatched {
        expected.extend([(1, None, Some(11), 1, 1), (3, Some(3), Some(30), 1, 1)]);
    }
    if insert {
        expected.push((
            4,
            None,
            if partial_source { None } else { Some(900) },
            version,
            version,
        ));
    }
    expected.sort_by_key(|row| row.0);
    check_rows(&merged, &expected).await;
}
#[tokio::test]
async fn merge_insert_spill_rejects_invalid_budget_before_execution() {
    let mut dataset = dataset("memory://").await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "invalid"),
        ])
        .await
        .unwrap();
    let source = record_batch!(("key", UInt32, [0]), ("value", UInt32, [100])).unwrap();
    let schema = source.schema();
    let error = job(Arc::new(dataset))
        .execute_uncommitted(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async {
                Err(DataFusionError::Execution("input must not execute".into()))
            }),
        )) as SendableRecordBatchStream)
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains(INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY),
        "{error}"
    );
}
