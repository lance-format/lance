// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column repacking through `compact_files`.

use super::*;
use crate::dataset::NewColumnTransform;
use arrow_array::StringArray;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

fn abc_batch(start: i32, rows: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(start..start + rows)),
            Arc::new(Int32Array::from_iter_values(
                (start..start + rows).map(|v| v * 10),
            )),
            Arc::new(StringArray::from_iter_values(
                (start..start + rows).map(|v| format!("row-{v}")),
            )),
        ],
    )
    .unwrap()
}

/// Two fragments of four rows, each `a, b, c` in one file.
async fn write(uri: &str, version: LanceFileVersion) -> Dataset {
    let data = abc_batch(0, 8);
    let reader = RecordBatchIterator::new([Ok(data.clone())], data.schema());
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 4,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// `write`, then two backfills: every fragment holds its columns in three
/// files, `[a, b, c]`, `[d]` and `[e]`.
async fn write_backfilled() -> Dataset {
    let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
    for (name, expr) in [("d", "a + 1"), ("e", "a + 2")] {
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(name.into(), expr.into())]),
                None,
                None,
            )
            .await
            .unwrap();
    }
    dataset
}

fn repack_options(max_files: Option<usize>, groups: Vec<Vec<&str>>) -> CompactionOptions {
    CompactionOptions {
        max_data_files_per_fragment: max_files,
        column_groups: groups
            .into_iter()
            .map(|group| group.into_iter().map(String::from).collect())
            .collect(),
        scope: CompactionScope::RepackColumns,
        ..Default::default()
    }
}

fn layout(dataset: &Dataset) -> Vec<Vec<Vec<i32>>> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| {
            fragment
                .metadata()
                .files
                .iter()
                .map(|file| file.fields.to_vec())
                .collect()
        })
        .collect()
}

fn live_file_counts(dataset: &Dataset) -> Vec<usize> {
    dataset
        .column_layout_stats()
        .iter()
        .map(|stats| stats.live_file_count)
        .collect()
}

#[tokio::test]
async fn repack_collapses_backfilled_files() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;
    assert_eq!(live_file_counts(&dataset), vec![3, 3]);

    let metrics = compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    assert_eq!(dataset.manifest.version, version + 1, "one commit");
    assert_eq!(layout(&dataset), vec![vec![vec![0, 1, 2, 3, 4]]; 2]);
    assert_eq!(metrics.files_added, 2);
    assert_eq!(metrics.files_removed, 6);
    assert_eq!(metrics.fragments_added, 0);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();

    // Nothing is left to do, so nothing is committed.
    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version + 1);
}

/// Above the limit, the columns of the biggest file stay where they are and
/// only the others are rewritten.
#[tokio::test]
async fn repack_keeps_the_biggest_file() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(2), vec![]), None)
        .await
        .unwrap();

    for files in layout(&dataset) {
        assert_eq!(files.len(), 2, "{files:?}");
        assert_eq!(files[0], vec![0, 1, 2], "the base file is untouched");
        assert_eq!(files[1], vec![3, 4]);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_follows_column_groups() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(
        &mut dataset,
        repack_options(None, vec![vec!["c", "e"]]),
        None,
    )
    .await
    .unwrap();

    for files in layout(&dataset) {
        let mut files = files
            .into_iter()
            .map(|fields| {
                fields
                    .into_iter()
                    .filter(|id| *id != TOMBSTONE_FIELD_ID)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        files.sort();
        // Only the group moves; the other columns stay where they were.
        assert_eq!(files, vec![vec![0, 1], vec![2, 4], vec![3]], "{files:?}");
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();

    // The layout matches the groups now.
    let version = dataset.manifest.version;
    compact_files(
        &mut dataset,
        repack_options(None, vec![vec!["c", "e"]]),
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, version);
}

/// Deleted rows stay deleted and the new files line up with the old ones.
#[tokio::test]
async fn repack_keeps_deletions() {
    let mut dataset = write_backfilled().await;
    dataset.delete("a = 1 OR a = 6").await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        // Neither small nor deleted enough to be rewritten.
        target_rows_per_fragment: 4,
        materialize_deletions_threshold: 0.5,
        max_data_files_per_fragment: Some(1),
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().deletion_file.is_some())
    );
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_keeps_scalar_index() {
    let mut dataset = write_backfilled().await;
    dataset
        .create_index(
            &["d"],
            IndexType::BTree,
            Some("d_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let before = dataset.load_indices().await.unwrap()[0].clone();

    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    let after = dataset.load_indices().await.unwrap()[0].clone();
    assert_eq!(after.uuid, before.uuid, "the index is not rebuilt");
    assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
    let mut scanner = dataset.scan();
    scanner.filter("d = 6").unwrap();
    let plan = scanner.explain_plan(true).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    let filtered = dataset
        .scan()
        .filter("d = 6")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(filtered.num_rows(), 1);
    dataset.validate().await.unwrap();
}

/// `drop_columns` leaves a dropped column's id in its file. A repack that
/// moves the other columns out drops that file, as `drop_columns` would.
#[tokio::test]
async fn repack_drops_file_left_with_only_dropped_columns() {
    let mut dataset = write_backfilled().await;
    dataset.drop_columns(&["c"]).await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    assert_eq!(layout(&dataset), vec![vec![vec![0, 1, 3, 4]]; 2]);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

/// A run that rewrites some fragments and repacks others commits the
/// rewrite first and the repacks second, and never touches one fragment both
/// ways.
#[tokio::test]
async fn compaction_rewrites_and_repacks_in_one_run() {
    let mut dataset = write_backfilled().await;
    // Fragment 0 qualifies for a rewrite, fragment 1 only for a repack.
    dataset.delete("a = 0 OR a = 1").await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let options = CompactionOptions {
        target_rows_per_fragment: 3,
        max_data_files_per_fragment: Some(1),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 2, "{plan:?}");
    assert_eq!(plan.tasks[0].kind, CompactionTaskKind::RewriteFragments);
    assert_eq!(plan.tasks[0].fragments[0].id, 0);
    assert!(matches!(
        plan.tasks[1].kind,
        CompactionTaskKind::RepackColumns { .. }
    ));
    assert_eq!(plan.tasks[1].fragments[0].id, 1);

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    let last = dataset.manifest.version;
    assert!(last >= version + 2);
    for (version, operation) in [(last - 1, "Rewrite"), (last, "DataReplacement")] {
        let transaction = dataset
            .checkout_version(version)
            .await
            .unwrap()
            .read_transaction()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(transaction.operation.to_string(), operation);
    }
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    // The rewritten fragment gets a new id, so it now scans last; the repack
    // changes no value.
    let rewritten = dataset.checkout_version(last - 1).await.unwrap();
    let after_rewrite = rewritten.scan().try_into_batch().await.unwrap();
    assert_eq!(after_rewrite.num_rows(), before.num_rows());
    assert_eq!(
        dataset.scan().try_into_batch().await.unwrap(),
        after_rewrite
    );
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn compaction_scope_selects_task_kinds() {
    let mut dataset = write_backfilled().await;
    dataset.delete("a = 0 OR a = 1").await.unwrap();
    let options = |scope| CompactionOptions {
        target_rows_per_fragment: 2,
        max_data_files_per_fragment: Some(1),
        scope,
        ..Default::default()
    };

    let rewrites = plan_compaction(&dataset, &options(CompactionScope::RewriteFragments))
        .await
        .unwrap();
    assert_eq!(rewrites.tasks.len(), 1);
    assert_eq!(rewrites.tasks[0].kind, CompactionTaskKind::RewriteFragments);

    let repacks = plan_compaction(&dataset, &options(CompactionScope::RepackColumns))
        .await
        .unwrap();
    assert_eq!(repacks.tasks.len(), 2, "both fragments are repacked");
    assert!(
        repacks
            .tasks
            .iter()
            .all(|task| matches!(task.kind, CompactionTaskKind::RepackColumns { .. }))
    );
}

/// Without `max_data_files_per_fragment` or `column_groups`, nothing is
/// repacked.
#[tokio::test]
async fn compaction_plans_no_repack_by_default() {
    let dataset = write_backfilled().await;
    // Every default but a target that leaves the 4-row fragments alone, so
    // that no rewrite claims them first.
    let plan = plan_compaction(
        &dataset,
        &CompactionOptions {
            target_rows_per_fragment: 4,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
    let plan = plan_compaction(
        &dataset,
        &CompactionOptions {
            scope: CompactionScope::RepackColumns,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

/// The limit is an upper bound: a fragment already at it is left alone.
#[tokio::test]
async fn repack_is_not_planned_at_the_limit() {
    let dataset = write_backfilled().await;
    assert_eq!(live_file_counts(&dataset), vec![3, 3]);
    let plan = plan_compaction(&dataset, &repack_options(Some(3), vec![]))
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

/// Repack tasks run on another machine: they are serialized, executed on
/// their own, and their results committed together.
#[tokio::test]
async fn repack_tasks_run_distributed() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();
    let plan = plan_compaction(&dataset, &repack_options(Some(1), vec![]))
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 2);

    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        let task: CompactionTask =
            serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
        let result = task.execute(&dataset).await.unwrap();
        let result: RewriteResult =
            serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
        results.push(result);
    }
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        plan.options(),
    )
    .await
    .unwrap();

    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

/// A task serialized before `kind` existed rewrites its fragments.
#[test]
fn task_data_without_kind_rewrites_fragments() {
    let task: TaskData = serde_json::from_str(r#"{"fragments": []}"#).unwrap();
    assert_eq!(task.kind, CompactionTaskKind::RewriteFragments);
}

/// A result serialized before `repacked_files` existed is a fragment rewrite.
#[test]
fn rewrite_result_without_repacked_files_is_a_rewrite() {
    let json = r#"{
        "metrics": {"fragments_removed": 0, "fragments_added": 0, "files_removed": 0, "files_added": 0},
        "new_fragments": [],
        "read_version": 1,
        "original_fragments": [],
        "row_addrs": null
    }"#;
    let result: RewriteResult = serde_json::from_str(json).unwrap();
    assert!(result.repacked_files.is_none());
}

/// Run a repack of every fragment to the point of commit.
async fn stale_repack_results(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Vec<RewriteResult> {
    let plan = plan_compaction(dataset, options).await.unwrap();
    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        results.push(task.execute(dataset).await.unwrap());
    }
    results
}

async fn commit_results(
    dataset: &mut Dataset,
    results: Vec<RewriteResult>,
) -> Result<CompactionMetrics> {
    commit_compaction(
        dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &CompactionOptions::default(),
    )
    .await
}

/// A delete that removes rows of a repacked fragment leaves the new files
/// aligned (a deletion only marks rows), so the repack still commits; one
/// that removes the whole fragment makes it retry.
#[rstest]
#[case::one_row("a = 1", true)]
#[case::whole_fragment("a < 4", false)]
#[tokio::test]
async fn repack_against_concurrent_delete(#[case] predicate: &str, #[case] commits: bool) {
    let mut dataset = write_backfilled().await;
    let stale = stale_repack_results(&dataset, &repack_options(Some(1), vec![])).await;

    dataset.delete(predicate).await.unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let result = commit_results(&mut dataset, stale).await;
    if commits {
        result.unwrap();
        assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    } else {
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, version);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

/// A concurrent `drop_columns` of a column the repack moves makes it retry.
/// One that drops a column the repack leaves alone does not: the new file is
/// applied to the fragment as the drop left it.
#[rstest]
#[case::moved_column("d", false)]
#[case::other_column("b", true)]
#[tokio::test]
async fn repack_against_concurrent_drop(#[case] dropped: &str, #[case] commits: bool) {
    let mut dataset = write_backfilled().await;
    // Moves d and e into one file and leaves the base file alone.
    let stale = stale_repack_results(&dataset, &repack_options(Some(2), vec![])).await;

    dataset.drop_columns(&[dropped]).await.unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let result = commit_results(&mut dataset, stale).await;
    if commits {
        result.unwrap();
        assert_eq!(live_file_counts(&dataset), vec![2, 2]);
    } else {
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, version);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

/// How a concurrent commit writes `d = 100` into the row `a = 2` without
/// moving the row out of its fragment.
#[derive(Debug, Clone, Copy)]
enum NewValueOfD {
    /// An `Update` that rewrites `d` in place (`UpdateMode::RewriteColumns`).
    InPlaceUpdate,
    /// A `DataReplacement` of `d` with `data_change: true`.
    Replacement,
}

async fn write_new_value_of_d(dataset: &mut Dataset, how: NewValueOfD) {
    use crate::dataset::transaction::UpdateMode;
    use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};

    match how {
        NewValueOfD::InPlaceUpdate => {
            let schema = Arc::new(Schema::from(
                &dataset.schema().project(&["a", "d"]).unwrap(),
            ));
            let source = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(vec![2])),
                    Arc::new(Int32Array::from(vec![100])),
                ],
            )
            .unwrap();
            MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["a".into()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(MergeInsertWriteMode::RewriteColumns)
                .try_build()
                .unwrap()
                .execute_batches(vec![source])
                .await
                .unwrap();
        }
        NewValueOfD::Replacement => {
            // Fragment 0 holds a = 0..4, so d = a + 1 = [1, 2, 3, 4].
            let schema = dataset.schema().project(&["d"]).unwrap();
            let batch = RecordBatch::try_new(
                Arc::new(Schema::from(&schema)),
                vec![Arc::new(Int32Array::from(vec![1, 2, 100, 4]))],
            )
            .unwrap();
            let group = dataset
                .get_fragment(0)
                .unwrap()
                .write_columns(futures::stream::iter([Ok(batch)]), &schema)
                .await
                .unwrap();
            let read_version = dataset.manifest.version;
            Dataset::commit(
                WriteDestination::Dataset(Arc::new(dataset.clone())),
                Operation::DataReplacement {
                    replacements: vec![group],
                    data_change: true,
                },
                Some(read_version),
                None,
                None,
                Arc::new(Default::default()),
                false,
            )
            .await
            .unwrap();
        }
    }
    dataset.checkout_latest().await.unwrap();

    // Each case has to keep testing the commit shape it names.
    let d_id = dataset.schema().field("d").unwrap().id as u32;
    let operation = dataset.read_transaction().await.unwrap().unwrap().operation;
    let shape_matches = match how {
        NewValueOfD::InPlaceUpdate => matches!(
            &operation,
            Operation::Update {
                update_mode: Some(UpdateMode::RewriteColumns),
                fields_modified,
                ..
            } if fields_modified.contains(&d_id)
        ),
        NewValueOfD::Replacement => matches!(
            operation,
            Operation::DataReplacement {
                data_change: true,
                ..
            }
        ),
    };
    assert!(shape_matches, "{how:?}: {operation:?}");
}

/// A concurrent commit that writes a new value into a moved column of the
/// fragment makes the repack retry rather than publish the old value over the
/// new one.
#[rstest]
#[case::in_place_update(NewValueOfD::InPlaceUpdate)]
#[case::data_changing_replacement(NewValueOfD::Replacement)]
#[tokio::test]
async fn repack_against_concurrent_new_value_of_moved_column(#[case] how: NewValueOfD) {
    let mut dataset = write_backfilled().await;
    // Moves every column of each fragment, d included, into one file.
    let stale = stale_repack_results(&dataset, &repack_options(Some(1), vec![])).await;

    write_new_value_of_d(&mut dataset, how).await;
    let expected = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(dataset.count_rows(Some("d = 100".into())).await.unwrap(), 1);

    let err = commit_results(&mut dataset, stale).await.unwrap_err();
    assert!(
        matches!(err, Error::RetryableCommitConflict { .. }),
        "{err}"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_skips_legacy_files() {
    let dir = TempStrDir::default();
    let dataset = write(&dir, LanceFileVersion::Legacy).await;
    let plan = plan_compaction(&dataset, &repack_options(None, vec![vec!["c"]]))
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

/// One legacy (V1) file is enough to leave the whole fragment alone, since
/// its fields cannot be tombstoned one by one.
#[test]
fn repack_skips_a_fragment_holding_a_legacy_file() {
    let schema = lance_schema(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        lance_table::format::DataFile::new_legacy_from_fields("legacy.lance", vec![0], None),
        v2_0_file("b.lance", vec![1]),
        v2_0_file("c.lance", vec![2]),
    ];
    let plan = |fragment: &Fragment| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            fragment,
            fragment.files.len(),
            None,
            Some(1),
        )
    };
    assert_eq!(plan(&fragment), None);
    // The same layout with a V2 file in its place is repacked.
    fragment.files[0] = v2_0_file("a.lance", vec![0]);
    assert_eq!(plan(&fragment), Some(vec![vec![0, 1, 2]]));
}

#[test]
fn max_data_files_per_fragment_must_be_positive() {
    let mut options = CompactionOptions {
        max_data_files_per_fragment: Some(0),
        ..Default::default()
    };
    assert!(options.validate().is_err());
}

#[test]
fn repack_options_parse_from_config() {
    let config = HashMap::from([(
        "lance.compaction.max_data_files_per_fragment".to_string(),
        "4".to_string(),
    )]);
    let options = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(options.max_data_files_per_fragment, Some(4));

    let config = HashMap::from([(
        "lance.compaction.max_data_files_per_fragment".to_string(),
        "lots".to_string(),
    )]);
    let err = CompactionOptions::from_dataset_config(&config).unwrap_err();
    assert!(
        err.to_string()
            .contains("lance.compaction.max_data_files_per_fragment")
            && err.to_string().contains("lots"),
        "{err}"
    );
}

#[test]
fn compaction_scope_rejects_an_unknown_name() {
    let err = CompactionScope::try_from("nonsense").unwrap_err();
    assert!(
        err.to_string().contains("Invalid compaction scope"),
        "{err}"
    );
    assert_eq!(
        CompactionScope::try_from("Repack_Columns").unwrap(),
        CompactionScope::RepackColumns
    );
}

/// The file holding a fragment's spilled row lineage keeps its columns: the
/// new files carry no lineage, so emptying that file would strand it.
#[test]
fn repack_keeps_the_spilled_lineage_file() {
    use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
    let schema = lance_core::datatypes::Schema::try_from(&Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]))
    .unwrap();
    let mut fragment = Fragment::new(0);
    fragment.add_file(
        "lineage.lance",
        vec![0, ROW_ID_FIELD_ID],
        vec![0, 1],
        lance_file::version::ConcreteFileVersion::V2_0,
        None,
    );
    for (path, field) in [("b.lance", 1), ("c.lance", 2)] {
        fragment.add_file(
            path,
            vec![field],
            vec![0],
            lance_file::version::ConcreteFileVersion::V2_0,
            None,
        );
    }
    fragment.row_id_meta = Some(RowIdMeta::Column);

    // Above the limit, the lineage file's column stays and the others move.
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(2)
        ),
        Some(vec![vec![1, 2]])
    );
    // With a limit of 1 as well: a file a move cannot empty stays anyway, so
    // the lineage file keeps its column.
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(1)
        ),
        Some(vec![vec![1, 2]])
    );
    // A group can name the lineage file's column, which a merge never moves.
    // Moving it out with another column would leave that file holding the
    // lineage alone, so the group is not planned; a group of the other
    // columns is.
    let plan = |groups: &[Vec<i32>]| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            Some(groups),
            None,
        )
    };
    assert_eq!(plan(&[vec![0, 1]]), None);
    assert_eq!(plan(&[vec![1, 2]]), Some(vec![vec![1, 2]]));
}

/// Over the file limit with groups set, the columns no group names are merged
/// into one file as well.
#[tokio::test]
async fn repack_merges_unclaimed_columns_over_the_limit() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(2), vec![vec!["e"]]), None)
        .await
        .unwrap();

    for files in layout(&dataset) {
        let mut files = files
            .into_iter()
            .map(|fields| {
                fields
                    .into_iter()
                    .filter(|id| *id != TOMBSTONE_FIELD_ID)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        files.sort();
        assert_eq!(files, vec![vec![0, 1, 2, 3], vec![4]], "{files:?}");
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

fn lance_schema(arrow: Schema) -> lance_core::datatypes::Schema {
    lance_core::datatypes::Schema::try_from(&arrow).unwrap()
}

fn v2_0_file(path: &str, fields: Vec<i32>) -> lance_table::format::DataFile {
    let indices = (0..fields.len() as i32).collect();
    lance_table::format::DataFile::new(
        path,
        fields,
        indices,
        lance_file::version::ConcreteFileVersion::V2_0,
        None,
        None,
    )
}

/// A V2.0 file keeps a struct's header after the struct's last child in it
/// is dropped. That header holds no data, so the file does not hold the
/// struct: the struct sits alone in the file with its live child, and no
/// repack is planned (one would be planned again on every run).
#[test]
fn repack_ignores_struct_header_without_children() {
    use arrow_schema::Fields;
    let arrow = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ])),
            true,
        ),
    ]);
    // a=0, s=1, s.x=2 (dropped), s.y=3.
    let schema = lance_schema(arrow).project_by_ids(&[0, 1, 3], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("a.lance", vec![0, 1, 2]),
        v2_0_file("s.lance", vec![1, 3]),
    ];

    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            2,
            Some(&[vec![1]]),
            None
        ),
        None
    );
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            2,
            None,
            Some(1)
        ),
        Some(vec![vec![0, 1]])
    );
}

/// A blob column never moves, so with a limit of 1 the other columns are
/// merged next to it instead of nothing being planned.
#[test]
fn repack_merges_around_a_blob_column() {
    let mut blob = Field::new("img", DataType::LargeBinary, true);
    blob.set_metadata(std::collections::HashMap::from([(
        lance_arrow::BLOB_META_KEY.to_string(),
        "true".to_string(),
    )]));
    let schema = lance_schema(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        blob,
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("base.lance", vec![0, 1]),
        v2_0_file("d.lance", vec![2]),
        v2_0_file("e.lance", vec![3]),
    ];
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(1)
        ),
        Some(vec![vec![2, 3]])
    );
}

/// A file a move cannot empty (here one holding a blob column) stays in any
/// case, so the files that can go all merge, even above a limit of 1: keeping
/// one of them as well would leave the fragment over the limit.
#[test]
fn repack_merges_every_movable_file_next_to_one_that_stays() {
    let mut blob = Field::new("img", DataType::LargeBinary, true);
    blob.set_metadata(std::collections::HashMap::from([(
        lance_arrow::BLOB_META_KEY.to_string(),
        "true".to_string(),
    )]));
    let schema = lance_schema(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        blob,
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int32, true),
        Field::new("f", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("base.lance", vec![0, 1]),
        v2_0_file("d.lance", vec![2]),
        v2_0_file("e.lance", vec![3]),
        v2_0_file("f.lance", vec![4]),
    ];
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            4,
            None,
            Some(2)
        ),
        Some(vec![vec![2, 3, 4]])
    );
}

/// A group holding a blob column stays where it is; a group without one
/// moves.
#[test]
fn repack_leaves_a_group_holding_a_blob_column() {
    let mut blob = Field::new("img", DataType::LargeBinary, true);
    blob.set_metadata(std::collections::HashMap::from([(
        lance_arrow::BLOB_META_KEY.to_string(),
        "true".to_string(),
    )]));
    let schema = lance_schema(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        blob,
        Field::new("d", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("base.lance", vec![0, 1]),
        v2_0_file("d.lance", vec![2]),
    ];
    let plan = |groups: &[Vec<i32>]| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            2,
            Some(groups),
            None,
        )
    };
    assert_eq!(plan(&[vec![1, 2]]), None);
    assert_eq!(plan(&[vec![0, 2]]), Some(vec![vec![0, 2]]));
}

#[tokio::test]
async fn repack_is_not_planned_under_force_binary_copy() {
    let dataset = write_backfilled().await;
    let options = CompactionOptions {
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..repack_options(Some(1), vec![])
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

/// `max_source_bytes` counts only the files a repack reads.
#[tokio::test]
async fn repack_counts_only_the_files_it_reads_against_the_byte_budget() {
    let dataset = write_backfilled().await;
    let fragment = &dataset.manifest.fragments[0];
    let size = |index: usize| fragment.files[index].file_size_bytes.get().unwrap().get();
    // Each repack leaves the base file alone and merges d and e, so a budget
    // for two of those fits both, though not one whole fragment.
    let options = CompactionOptions {
        max_source_bytes: Some(2 * (size(1) + size(2))),
        ..repack_options(Some(2), vec![])
    };
    assert!(size(0) + size(1) + size(2) > 2 * (size(1) + size(2)));
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 2, "{plan:?}");
}

/// A fragment both rewritten and repacked is refused before either commit.
#[tokio::test]
async fn commit_rejects_a_fragment_in_both_kinds_of_task() {
    let mut dataset = write_backfilled().await;
    let version = dataset.manifest.version;
    let repacks = stale_repack_results(&dataset, &repack_options(Some(1), vec![])).await;
    let rewrite = CompactionTask {
        task: TaskData::rewrite_fragments(vec![dataset.manifest.fragments[0].clone()]),
        read_version: version,
        options: CompactionOptions::default(),
    }
    .execute(&dataset)
    .await
    .unwrap();

    let mut results = repacks;
    results.push(rewrite);
    let err = commit_results(&mut dataset, results).await.unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.manifest.version, version);
}

/// The lineage file and a blob file both stay as they are; the columns that
/// can move are still merged.
#[test]
fn repack_merges_around_a_blob_outside_the_kept_file() {
    use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
    let mut blob = Field::new("img", DataType::LargeBinary, true);
    blob.set_metadata(std::collections::HashMap::from([(
        lance_arrow::BLOB_META_KEY.to_string(),
        "true".to_string(),
    )]));
    let schema = lance_schema(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        blob,
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("lineage.lance", vec![0, ROW_ID_FIELD_ID]),
        v2_0_file("img.lance", vec![1]),
        v2_0_file("d.lance", vec![2]),
        v2_0_file("e.lance", vec![3]),
    ];
    fragment.row_id_meta = Some(RowIdMeta::Column);
    for max_files in [1, 2] {
        assert_eq!(
            crate::dataset::optimize::repack::plan_fragment_repack(
                &schema,
                &fragment,
                4,
                None,
                Some(max_files)
            ),
            Some(vec![vec![2, 3]]),
            "max_files={max_files}"
        );
    }
}

/// A file holding a column that cannot move is never partly emptied: moving
/// its other columns out would add a file instead of removing one.
#[test]
fn repack_never_adds_a_file() {
    use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
    let blob = |name: &str| {
        let mut field = Field::new(name, DataType::LargeBinary, true);
        field.set_metadata(std::collections::HashMap::from([(
            lance_arrow::BLOB_META_KEY.to_string(),
            "true".to_string(),
        )]));
        field
    };
    let plan = |schema: &lance_core::datatypes::Schema, fragment: &Fragment, max| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            schema,
            fragment,
            fragment.files.len(),
            None,
            Some(max),
        )
    };

    // Lineage in one file, a blob next to a movable column in the other.
    let schema = lance_schema(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        blob("img"),
        Field::new("d", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("lineage.lance", vec![0, ROW_ID_FIELD_ID]),
        v2_0_file("mixed.lance", vec![1, 2]),
    ];
    fragment.row_id_meta = Some(RowIdMeta::Column);
    assert_eq!(plan(&schema, &fragment, 1), None);

    // Two files, each a blob next to a movable column.
    let schema = lance_schema(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        blob("img1"),
        blob("img2"),
        Field::new("cap", DataType::Utf8, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("base.lance", vec![0, 1]),
        v2_0_file("extra.lance", vec![2, 3]),
    ];
    assert_eq!(plan(&schema, &fragment, 1), None);

    // A blob next to a movable column, and two movable files: only those two
    // merge.
    let schema = lance_schema(Schema::new(vec![
        blob("img"),
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int32, true),
        Field::new("f", DataType::Int32, true),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("mixed.lance", vec![0, 1]),
        v2_0_file("e.lance", vec![2]),
        v2_0_file("f.lance", vec![3]),
    ];
    assert_eq!(plan(&schema, &fragment, 2), Some(vec![vec![2, 3]]));
}

/// A file holding a struct's header next to another column is emptied only
/// when the struct moves too, so a merge is planned around the file whose
/// staying leaves the others emptied.
#[test]
fn repack_moves_a_struct_with_its_header() {
    use arrow_schema::Fields;
    let plan = |schema: &lance_core::datatypes::Schema, fragment: &Fragment| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            schema,
            fragment,
            fragment.files.len(),
            None,
            Some(2),
        )
    };
    let child = |name: &str| Field::new(name, DataType::Int32, true);

    // a=0, s=1 {x=2 (dropped), y=3}, b=4, c=5.
    let arrow = Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y")])),
            true,
        ),
        child("b"),
        child("c"),
    ]);
    let schema = lance_schema(arrow).project_by_ids(&[0, 1, 3, 4, 5], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("a.lance", vec![0, 1, 2]),
        v2_0_file("s.lance", vec![1, 3, 5]),
        v2_0_file("b.lance", vec![4]),
    ];
    // Keeping s.lance would leave a.lance alive on the header; keeping
    // a.lance lets s.lance and b.lance both go.
    assert_eq!(plan(&schema, &fragment), Some(vec![vec![1, 4, 5]]));

    // a=0, s=1 {x=2, y=3, z=4} (x, z dropped), b=5, c=6: both a.lance and
    // b.lance hold the header.
    let arrow = Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y"), child("z")])),
            true,
        ),
        child("b"),
        child("c"),
    ]);
    let schema = lance_schema(arrow).project_by_ids(&[0, 1, 3, 5, 6], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("a.lance", vec![0, 1, 2]),
        v2_0_file("b.lance", vec![5, 1, 4]),
        v2_0_file("s.lance", vec![1, 3, 6]),
    ];
    assert_eq!(plan(&schema, &fragment), Some(vec![vec![1, 5, 6]]));
}

/// A fragment whose first file still holds a dropped struct child reaches one
/// file, and the next run plans nothing.
#[tokio::test]
async fn repack_with_a_dropped_struct_child_converges() {
    use arrow_array::{ArrayRef, StructArray};
    use arrow_schema::Fields;
    let fields = Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Int32, true),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("s", DataType::Struct(fields.clone()), true),
    ]));
    let values =
        |offset: i32| Arc::new(Int32Array::from_iter_values(offset..offset + 4)) as ArrayRef;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            values(0),
            Arc::new(StructArray::new(fields, vec![values(10), values(20)], None)),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("b".into(), "a + 1".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    dataset.drop_columns(&["s.x"]).await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();
    assert_eq!(live_file_counts(&dataset), vec![1]);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();

    let version = dataset.manifest.version;
    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version);
}

/// Struct-header layouts for the merges the other planner tests don't reach:
/// the unclaimed columns under `column_groups`, a struct split with the
/// lineage file, and the file kept when every file could go.
#[test]
fn repack_merges_only_files_it_empties() {
    use arrow_schema::Fields;
    use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    let plan = |schema: &lance_core::datatypes::Schema,
                fragment: &Fragment,
                groups: Option<&[Vec<i32>]>,
                max| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            schema,
            fragment,
            fragment.files.len(),
            groups,
            Some(max),
        )
    };

    // a=0, s=1 {x=2 (dropped), y=3, z=4 (dropped)}, b=5, groups [[s]].
    // a.lance and b.lance both keep s's header, so merging a and b would
    // drop neither.
    let arrow = Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y"), child("z")])),
            true,
        ),
        child("b"),
    ]);
    let schema = lance_schema(arrow).project_by_ids(&[0, 1, 3, 5], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("a.lance", vec![0, 1, 2]),
        v2_0_file("b.lance", vec![5, 1, 4]),
        v2_0_file("s.lance", vec![1, 3]),
    ];
    assert_eq!(plan(&schema, &fragment, Some(&[vec![1]]), 2), None);

    // s=0 {y=1, z=2}, b=3, c=4. s is split between the lineage file and
    // c1.lance; moving s would empty the lineage file's columns, so only b
    // and c merge.
    let arrow = Schema::new(vec![
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("y"), child("z")])),
            true,
        ),
        child("b"),
        child("c"),
    ]);
    let schema = lance_schema(arrow);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("lineage.lance", vec![0, 1, ROW_ID_FIELD_ID]),
        v2_0_file("c1.lance", vec![0, 2]),
        v2_0_file("b.lance", vec![3]),
        v2_0_file("c.lance", vec![4]),
    ];
    fragment.row_id_meta = Some(RowIdMeta::Column);
    assert_eq!(plan(&schema, &fragment, None, 2), Some(vec![vec![3, 4]]));

    // a=0, s=1 {x=2, y=3}, b=4, c=5. k.lance is the largest and shares s with
    // m1.lance; keeping k leaves s and m1 alone and merges b and c.
    let arrow = Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y")])),
            true,
        ),
        child("b"),
        child("c"),
    ]);
    let schema = lance_schema(arrow);
    let sized = |path: &str, fields: Vec<i32>, size: u64| {
        let mut file = v2_0_file(path, fields);
        file.file_size_bytes = lance_io::utils::CachedFileSize::new(size);
        file
    };
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        sized("k.lance", vec![1, 2], 100),
        sized("m1.lance", vec![0, 1, 3], 10),
        sized("m2.lance", vec![4], 10),
        sized("m3.lance", vec![5], 10),
    ];
    assert_eq!(plan(&schema, &fragment, None, 2), Some(vec![vec![4, 5]]));
}

/// When every file holds the struct, the largest file is kept with its other
/// column and the struct's other files merge; and when only some files share
/// a column with the largest, the merge falls back to every other file.
#[test]
fn repack_keeps_the_largest_file_that_shares_a_struct() {
    use arrow_schema::Fields;
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    let sized = |path: &str, fields: Vec<i32>, size: u64| {
        let mut file = v2_0_file(path, fields);
        file.file_size_bytes = lance_io::utils::CachedFileSize::new(size);
        file
    };
    let plan = |schema: &lance_core::datatypes::Schema, fragment: &Fragment| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            schema,
            fragment,
            fragment.files.len(),
            None,
            Some(2),
        )
    };
    // a=0, s=1 {x=2, y=3, z=4}, b=5.
    let schema = lance_schema(Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y"), child("z")])),
            true,
        ),
        child("b"),
    ]));

    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        sized("k.lance", vec![0, 1, 2], 100),
        sized("m1.lance", vec![1, 3], 10),
        sized("m2.lance", vec![1, 4], 10),
    ];
    assert_eq!(plan(&schema, &fragment), Some(vec![vec![1]]));

    fragment.files = vec![
        sized("k.lance", vec![0, 1, 2], 100),
        sized("m1.lance", vec![1, 3, 4], 10),
        sized("m2.lance", vec![5], 10),
    ];
    assert_eq!(plan(&schema, &fragment), Some(vec![vec![1, 5]]));
}

/// When the search finds no file to keep while two others merge, the files
/// all merge, as under a limit of 1: a looser limit must not leave the
/// fragment over it while a limit of 1 fixes it.
#[test]
fn repack_merges_all_files_when_none_can_stay() {
    use arrow_schema::Fields;
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    let plan = |schema: &lance_core::datatypes::Schema, fragment: &Fragment, limit| {
        crate::dataset::optimize::repack::plan_fragment_repack(
            schema,
            fragment,
            fragment.files.len(),
            None,
            Some(limit),
        )
    };

    // s=0 {x=1, y=2, z=3}, one child per file.
    let schema = lance_schema(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![child("x"), child("y"), child("z")])),
        true,
    )]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("x.lance", vec![0, 1]),
        v2_0_file("y.lance", vec![0, 2]),
        v2_0_file("z.lance", vec![0, 3]),
    ];
    assert_eq!(plan(&schema, &fragment, 2), Some(vec![vec![0]]));
    assert_eq!(plan(&schema, &fragment, 2), plan(&schema, &fragment, 1));

    // s=0 {x=1, y=2}, t=3 {p=4, q=5}: the first file shares s with the
    // second and t with the third.
    let schema = lance_schema(Schema::new(vec![
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y")])),
            true,
        ),
        Field::new(
            "t",
            DataType::Struct(Fields::from(vec![child("p"), child("q")])),
            true,
        ),
    ]));
    fragment.files = vec![
        v2_0_file("xp.lance", vec![0, 1, 3, 4]),
        v2_0_file("y.lance", vec![0, 2]),
        v2_0_file("q.lance", vec![3, 5]),
    ];
    assert_eq!(plan(&schema, &fragment, 2), Some(vec![vec![0, 3]]));
    assert_eq!(plan(&schema, &fragment, 2), plan(&schema, &fragment, 1));

    // a=0, s=1 {x=2 (dropped), y=3}: two files hold a column and a third
    // holds only s's header, so it counts toward the limit. Keeping either
    // column file merges one file, so both merge and the header file goes
    // with s.
    let schema = lance_schema(Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y")])),
            true,
        ),
    ]))
    .project_by_ids(&[0, 1, 3], false);
    fragment.files = vec![
        v2_0_file("h.lance", vec![1, 2]),
        v2_0_file("a.lance", vec![0]),
        v2_0_file("sy.lance", vec![1, 3]),
    ];
    assert_eq!(plan(&schema, &fragment, 2), Some(vec![vec![0, 1]]));
    assert_eq!(plan(&schema, &fragment, 2), plan(&schema, &fragment, 1));
}

/// A V2.0 file holding only a struct header holds no column, so it does not
/// stop the largest file from being kept; it goes when the struct moves.
#[test]
fn repack_keeps_the_largest_file_next_to_a_header_only_file() {
    use arrow_schema::Fields;
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    let sized = |path: &str, fields: Vec<i32>, size: u64| {
        let mut file = v2_0_file(path, fields);
        file.file_size_bytes = lance_io::utils::CachedFileSize::new(size);
        file
    };
    // a=0, s=1 {x=2 (dropped), y=3}, b=4.
    let schema = lance_schema(Schema::new(vec![
        child("a"),
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y")])),
            true,
        ),
        child("b"),
    ]))
    .project_by_ids(&[0, 1, 3, 4], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        sized("h.lance", vec![1, 2], 10),
        sized("base.lance", vec![0], 100),
        sized("sy.lance", vec![1, 3], 10),
        sized("bf.lance", vec![4], 10),
    ];
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            4,
            None,
            Some(2)
        ),
        Some(vec![vec![1, 4]])
    );
}

/// A merge that leaves the largest file in place wins over one that merges
/// it away, even when a smaller file is the one kept.
#[test]
fn repack_prefers_leaving_the_largest_file_in_place() {
    use arrow_schema::Fields;
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    let sized = |path: &str, fields: Vec<i32>, size: u64| {
        let mut file = v2_0_file(path, fields);
        file.file_size_bytes = lance_io::utils::CachedFileSize::new(size);
        file
    };
    // s=0 {x=1, y=2, z=3}, t=4 {p=5, q=6}, c=7.
    let schema = lance_schema(Schema::new(vec![
        Field::new(
            "s",
            DataType::Struct(Fields::from(vec![child("x"), child("y"), child("z")])),
            true,
        ),
        Field::new(
            "t",
            DataType::Struct(Fields::from(vec![child("p"), child("q")])),
            true,
        ),
        child("c"),
    ]));
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        sized("k.lance", vec![0, 1, 4, 5], 100),
        sized("m1.lance", vec![0, 2, 7], 10),
        sized("m2.lance", vec![0, 3], 10),
        sized("m3.lance", vec![4, 6], 10),
    ];
    // Keeping m3 merges m1 and m2 (s and c) and leaves k holding t.
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            4,
            None,
            Some(3)
        ),
        Some(vec![vec![0, 7]])
    );
}

/// A fragment whose files all hold only a struct header has no column to
/// move, and nothing is planned.
#[test]
fn repack_plans_nothing_for_header_only_files() {
    use arrow_schema::Fields;
    let child = |name: &str| Field::new(name, DataType::Int32, true);
    // s=0 {a=1, b=2, c=3, d=4}; only d is left.
    let schema = lance_schema(Schema::new(vec![Field::new(
        "s",
        DataType::Struct(Fields::from(vec![
            child("a"),
            child("b"),
            child("c"),
            child("d"),
        ])),
        true,
    )]))
    .project_by_ids(&[0, 4], false);
    let mut fragment = Fragment::new(0);
    fragment.files = vec![
        v2_0_file("a.lance", vec![0, 1]),
        v2_0_file("b.lance", vec![0, 2]),
        v2_0_file("c.lance", vec![0, 3]),
    ];
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(2)
        ),
        None
    );
}
