// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the planning contract area.

use super::*;
use std::collections::BTreeSet;

use crate::dataset::optimize::planning::{
    CandidateBin, CompactionCandidacy, FragmentMetrics, HELD_BACK_SAMPLE_LIMIT, describe_held,
    fragment_file_bytes, limit_tasks_to_source_budget,
};

/// The hold-back log names at most [`HELD_BACK_SAMPLE_LIMIT`] fragment ids,
/// in ascending order, and summarizes the rest as a count.
#[test]
fn held_back_description_truncates_to_a_sample_and_a_count() {
    assert_eq!(describe_held(&BTreeSet::new()), "fragment(s) ");
    assert_eq!(describe_held(&BTreeSet::from([7])), "fragment(s) 7");
    assert_eq!(
        describe_held(&BTreeSet::from([3, 1, 2])),
        "fragment(s) 1, 2, 3",
        "ids are named in ascending order"
    );

    let at_limit = (0..HELD_BACK_SAMPLE_LIMIT as u64).collect::<BTreeSet<_>>();
    assert_eq!(
        describe_held(&at_limit),
        "fragment(s) 0, 1, 2, 3, 4, 5, 6, 7"
    );

    let over_limit = (0..HELD_BACK_SAMPLE_LIMIT as u64 + 1).collect::<BTreeSet<_>>();
    assert_eq!(
        describe_held(&over_limit),
        "fragment(s) 0, 1, 2, 3, 4, 5, 6, 7 and 1 more"
    );

    let wide = (100..1_100).collect::<BTreeSet<_>>();
    assert_eq!(
        describe_held(&wide),
        "fragment(s) 100, 101, 102, 103, 104, 105, 106, 107 and 992 more",
        "a wide index cannot produce an unbounded line"
    );
}

#[rstest::rstest]
#[case::below_default(Some(949), Some(1000), None, true)]
#[case::at_default(Some(950), Some(1000), None, false)]
#[case::above_default(Some(951), Some(1000), None, false)]
#[case::below_explicit(Some(899), Some(1000), Some(900), true)]
#[case::at_explicit(Some(900), Some(1000), Some(900), false)]
#[case::higher_threshold(Some(950), Some(1000), Some(1000), true)]
#[case::round_up(Some(95), Some(101), None, true)]
#[case::at_rounded_threshold(Some(96), Some(101), None, false)]
#[case::unknown(None, Some(1000), Some(900), true)]
#[case::writer_default(
    Some((WriteParams::default().max_bytes_per_file as u128 * 95).div_ceil(100) as u64),
    None,
    None,
    false
)]
#[case::max_integer(Some(usize::MAX as u64), Some(usize::MAX), None, false)]
fn merge_candidate_byte_threshold(
    #[case] bytes: Option<u64>,
    #[case] cap: Option<usize>,
    #[case] min_bytes: Option<usize>,
    #[case] expected: bool,
) {
    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1000,
        max_bytes_per_file: cap,
        min_bytes_per_file: min_bytes,
        ..Default::default()
    })
    .unwrap();
    let fragment = Fragment::new(0).with_file(
        "data.lance",
        vec![0],
        vec![0],
        ConcreteFileVersion::V2_2,
        bytes.and_then(std::num::NonZero::new),
    );
    let fields = HashSet::from([0]);
    let metrics = FragmentMetrics {
        physical_rows: 10,
        num_deletions: 0,
    };
    assert_eq!(
        planner
            .candidacy_of(&fragment, &metrics, &fields, false)
            .map(|reason| matches!(reason, CompactionCandidacy::CompactWithNeighbors)),
        expected.then_some(true)
    );
}

#[test]
fn test_candidate_bin() {
    let empty_bin = CandidateBin {
        fragments: vec![],
        pos_range: 0..0,
        candidacy: vec![],
        row_counts: vec![],
        indices: vec![],
    };
    assert!(empty_bin.is_noop());

    let fragment = Fragment {
        id: 0,
        files: vec![],
        overlays: vec![],
        deletion_file: None,
        row_id_meta: None,
        physical_rows: Some(0),
        last_updated_at_version_meta: None,
        created_at_version_meta: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let single_bin = CandidateBin {
        fragments: vec![fragment.clone()],
        pos_range: 0..1,
        candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
        row_counts: vec![100],
        indices: vec![],
    };
    assert!(single_bin.is_noop());

    let single_bin = CandidateBin {
        fragments: vec![fragment.clone()],
        pos_range: 0..1,
        candidacy: vec![CompactionCandidacy::CompactItself],
        row_counts: vec![100],
        indices: vec![],
    };
    // Not a no-op because it's CompactItself
    assert!(!single_bin.is_noop());

    let big_bin = CandidateBin {
        fragments: std::iter::repeat_n(fragment, 8).collect(),
        pos_range: 0..8,
        candidacy: std::iter::repeat_n(CompactionCandidacy::CompactItself, 8).collect(),
        row_counts: vec![100, 400, 200, 200, 400, 300, 300, 100],
        indices: vec![],
        // Will group into: [[100, 400], [200, 200, 400], [300, 300, 100]]
        // with size = 500
    };
    assert!(!big_bin.is_noop());
    let split = big_bin.split_for_size(500);
    assert_eq!(split.len(), 3);
    assert_eq!(split[0].pos_range, 0..2);
    assert_eq!(split[1].pos_range, 2..5);
    assert_eq!(split[2].pos_range, 5..8);

    let zero_min_split_bin = CandidateBin {
        fragments: std::iter::repeat_n(
            Fragment {
                id: 0,
                files: vec![],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: Some(0),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                clustering_id: None,
                metadata: Default::default(),
            },
            3,
        )
        .collect(),
        pos_range: 0..3,
        candidacy: std::iter::repeat_n(CompactionCandidacy::CompactItself, 3).collect(),
        row_counts: vec![100, 200, 300],
        indices: vec![],
    };
    let split = zero_min_split_bin.split_for_size(0);
    assert_eq!(split.len(), 3);
    assert!(split.iter().all(|bin| !bin.fragments.is_empty()));
    assert_eq!(split[0].pos_range, 0..1);
    assert_eq!(split[1].pos_range, 1..2);
    assert_eq!(split[2].pos_range, 2..3);
}
/// A task's output properties are independent: outputs carry exactly the
/// clustering id and the metadata the planner set, in any of the four
/// combinations, whatever the inputs carried; metadata needs no
/// definition; an id whose definition the manifest lacks is refused at
/// commit and no replacement is published.
#[tokio::test]
async fn compaction_outputs_carry_the_tasks_output_properties() {
    use crate::dataset::CommitBuilder;
    use crate::dataset::transaction::{Operation, Transaction};
    use lance_table::format::ClusteringDefinition;

    struct Stamping {
        inner: DefaultCompactionPlanner,
        clustering_id: Option<u32>,
        metadata: HashMap<String, String>,
    }

    #[async_trait::async_trait]
    impl CompactionPlanner for Stamping {
        async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
            let mut plan = self.inner.plan(dataset).await?;
            for task in &mut plan.tasks {
                task.output_clustering_id = self.clustering_id;
                task.output_metadata = self.metadata.clone();
            }
            Ok(plan)
        }
    }

    fn stamping(clustering_id: Option<u32>, metadata: &[(&str, &str)]) -> Stamping {
        Stamping {
            inner: DefaultCompactionPlanner::new(CompactionOptions {
                target_rows_per_fragment: 1_000,
                ..Default::default()
            })
            .unwrap(),
            clustering_id,
            metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    async fn three_fragments(uri: &str) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..300))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 3);
        dataset
    }

    async fn with_definition(dataset: Dataset, id: u32) -> Dataset {
        let definition = ClusteringDefinition {
            id,
            columns: vec![0],
            provider: "org.example.cluster".into(),
            version: 1,
        };
        CommitBuilder::new(Arc::new(dataset.clone()))
            .with_clustering_definitions(vec![definition])
            .execute(Transaction::new(
                dataset.manifest.version,
                Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: None,
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                },
                None,
            ))
            .await
            .unwrap()
    }

    async fn append_two_unstamped(dataset: &Dataset, uri: &str) -> Dataset {
        let appended = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from_iter_values(300..320))],
        )
        .unwrap();
        let mut dataset = dataset.clone();
        for _ in 0..2 {
            dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
                uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        }
        dataset
    }

    fn assert_outputs(dataset: &Dataset, id: Option<u32>, metadata: &[(&str, &str)]) {
        let expected: HashMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        for output in dataset.fragments().iter() {
            assert_eq!(output.clustering_id, id, "{output:?}");
            assert_eq!(output.metadata, expected, "{output:?}");
        }
    }

    // Both: id under a defined definition plus opaque metadata.
    let dir = TempStrDir::default();
    let mut dataset = with_definition(three_fragments(&dir).await, 1).await;
    compact_files_with_planner(&mut dataset, None, &stamping(Some(1), &[("bucket", "7")]))
        .await
        .unwrap();
    assert!(dataset.fragments().len() < 3, "{:?}", dataset.fragments());
    assert_outputs(&dataset, Some(1), &[("bucket", "7")]);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 300);

    // Neither: stamped and unstamped inputs compact into outputs that carry
    // no id and no metadata; nothing is inferred or copied from the inputs.
    let mut dataset = append_two_unstamped(&dataset, &dir).await;
    assert!(
        dataset
            .fragments()
            .iter()
            .any(|f| f.clustering_id == Some(1))
    );
    assert!(
        dataset
            .fragments()
            .iter()
            .any(|f| f.clustering_id.is_none())
    );
    compact_files_with_planner(&mut dataset, None, &stamping(None, &[]))
        .await
        .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 340);
    assert_outputs(&dataset, None, &[]);

    // Metadata only: no definition exists on this table and none is needed.
    let dir = TempStrDir::default();
    let mut dataset = three_fragments(&dir).await;
    assert!(dataset.manifest.clustering.is_empty());
    compact_files_with_planner(
        &mut dataset,
        None,
        &stamping(None, &[("owner", "x"), ("bucket", "3")]),
    )
    .await
    .unwrap();
    assert!(dataset.fragments().len() < 3, "{:?}", dataset.fragments());
    assert_outputs(&dataset, None, &[("owner", "x"), ("bucket", "3")]);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 300);

    // Id only: outputs carry the definition and empty metadata, replacing
    // the metadata the previous round stamped. Definition ids continue the
    // manifest's maximum, so this table's first definition is 1.
    let mut dataset = with_definition(append_two_unstamped(&dataset, &dir).await, 1).await;
    compact_files_with_planner(&mut dataset, None, &stamping(Some(1), &[]))
        .await
        .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 340);
    assert_outputs(&dataset, Some(1), &[]);

    // Serialized tasks carry both properties and omit them when unset.
    let fragments = dataset.manifest.fragments.as_ref().clone();
    let task = TaskData {
        fragments: fragments.clone(),
        output_clustering_id: Some(1),
        output_metadata: [("k".to_string(), "v".to_string())].into(),
    };
    let json = serde_json::to_string(&task).unwrap();
    assert!(json.contains("\"output_clustering_id\":1"), "{json}");
    assert!(json.contains("\"output_metadata\""), "{json}");
    let back: TaskData = serde_json::from_str(&json).unwrap();
    assert_eq!(back, task);
    let plain = TaskData {
        fragments,
        output_clustering_id: None,
        output_metadata: HashMap::new(),
    };
    let json = serde_json::to_string(&plain).unwrap();
    assert!(!json.contains("output_"), "{json}");

    // A stale id, one at or below the watermark that the table no longer
    // defines, is refused before any task rewrites data: no file is staged
    // and no version is added. Definition 1 is allocated, then cleared.
    let dir = TempStrDir::default();
    let dataset = with_definition(three_fragments(&dir).await, 1).await;
    let mut dataset = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_clustering_definitions(Vec::new())
        .execute(Transaction::new(
            dataset.manifest.version,
            Operation::UpdateConfig {
                config_updates: None,
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates: HashMap::new(),
            },
            None,
        ))
        .await
        .unwrap();
    assert!(dataset.manifest.clustering.is_empty());
    assert_eq!(dataset.manifest.max_clustering_id, 1);
    let version = dataset.manifest.version;
    let error = compact_files_with_planner(&mut dataset, None, &stamping(Some(1), &[]))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("definition 1, which the table no longer defines"),
        "{error}"
    );
    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(reopened.manifest.version, version);
    assert_eq!(reopened.fragments().len(), 3);
    assert_outputs(&reopened, None, &[]);
    // A worker given such a task refuses it the same way.
    let error = rewrite_files(
        Cow::Borrowed(&reopened),
        TaskData {
            fragments: reopened.manifest.fragments.as_ref().clone(),
            output_clustering_id: Some(1),
            output_metadata: HashMap::new(),
        },
        &CompactionOptions::default(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("no longer defines"), "{error}");

    // An id above the watermark may have been allocated after this
    // snapshot was read, so it is left to the commit, which refuses it
    // against the landing manifest: no replacement is published, though
    // the id reservation may have advanced the version and output files
    // may have been staged.
    let error = compact_files_with_planner(&mut dataset, None, &stamping(Some(9), &[]))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("clustering"), "{error}");
    let reopened = Dataset::open(&dir).await.unwrap();
    assert!(reopened.manifest.version <= version + 1);
    assert_eq!(reopened.fragments().len(), 3);
    assert_outputs(&reopened, None, &[]);
    assert_eq!(reopened.count_rows(None).await.unwrap(), 300);
}
/// Compacting explicitly chosen, non-adjacent fragments replaces exactly
/// those sources through the public plan -> execute -> commit path: the
/// untouched fragments keep their ids, files and metadata, the output takes
/// a fresh id at the end, carries the group's properties and holds the
/// sources' rows in source order; refusals happen before anything is
/// written; a group over the fragment budget is not planned.
#[tokio::test]
async fn grouped_compaction_replaces_only_the_selected_fragments() {
    struct Groups(Vec<CompactionGroup>, CompactionOptions);

    #[async_trait::async_trait]
    impl CompactionPlanner for Groups {
        async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
            DefaultCompactionPlanner::new(self.1.clone())?
                .plan_groups(dataset, self.0.clone())
                .await
        }
    }

    fn group(fragments: &[&Fragment], metadata: &[(&str, &str)]) -> CompactionGroup {
        CompactionGroup {
            fragments: fragments.iter().map(|f| (*f).clone()).collect(),
            output_clustering_id: None,
            output_metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    async fn values_of(dataset: &Dataset, fragments: Vec<Fragment>) -> Vec<i64> {
        let mut scanner = dataset.scan();
        scanner
            .with_fragments(fragments)
            .scan_in_order(true)
            .project(&["a"])
            .unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    // [A1, B1, A2, B2]: fragment i holds rows 100i..100i+100.
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..400))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let before = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(
        before.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };

    // Refusals, before anything is written.
    for (groups, expected) in [
        (vec![group(&[], &[])], "at least one fragment"),
        (
            vec![group(&[&before[2], &before[0]], &[])],
            "ascending id order",
        ),
        (
            vec![
                group(&[&before[0], &before[2]], &[]),
                group(&[&before[2]], &[]),
            ],
            "more than one compaction group",
        ),
    ] {
        let error =
            compact_files_with_planner(&mut dataset, None, &Groups(groups, options.clone()))
                .await
                .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let mut missing = before[3].clone();
    missing.id = 99;
    let error = compact_files_with_planner(
        &mut dataset,
        None,
        &Groups(vec![group(&[&missing], &[])], options.clone()),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("not in the dataset"), "{error}");
    let error = compact_files_with_planner(
        &mut dataset,
        None,
        &Groups(
            vec![group(&[&before[1], &before[3]], &[])],
            CompactionOptions {
                excluded_fragment_ids: vec![3],
                ..options.clone()
            },
        ),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("excluded"), "{error}");
    assert_eq!(dataset.manifest.fragments.as_ref(), &before);

    // A1 + A2 compact together; B1 and B2 are left exactly as they were.
    let metrics = compact_files_with_planner(
        &mut dataset,
        None,
        &Groups(
            vec![group(&[&before[0], &before[2]], &[("group", "a")])],
            options.clone(),
        ),
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);
    let after = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(
        after.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![1, 3, 4]
    );
    assert_eq!(after[0], before[1]);
    assert_eq!(after[1], before[3]);
    let output = &after[2];
    assert_eq!(output.physical_rows, Some(200));
    assert_eq!(output.clustering_id, None);
    assert_eq!(output.metadata.get("group").map(String::as_str), Some("a"));
    assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
    let expected: Vec<i64> = (0..100).chain(200..300).collect();
    assert_eq!(values_of(&dataset, vec![output.clone()]).await, expected);
    let mut all = values_of(&dataset, after.clone()).await;
    all.sort_unstable();
    assert_eq!(all, (0..400).collect::<Vec<_>>());

    // The non-adjacent remainder under a fragment budget: over budget is
    // not planned and nothing is written; within budget it compacts.
    let mut missing_size = dataset.clone();
    let manifest = Arc::make_mut(&mut missing_size.manifest);
    Arc::make_mut(&mut manifest.fragments)[0].files[0].file_size_bytes = CachedFileSize::unknown();
    let groups = vec![group(&[&after[0], &after[1]], &[("group", "b")])];

    let soft_plan = DefaultCompactionPlanner::new(CompactionOptions {
        max_source_fragments: Some(1),
        max_source_bytes: Some(u64::MAX),
        source_budget_mode: SourceBudgetMode::Soft,
        ..options.clone()
    })
    .unwrap()
    .plan_groups(&missing_size, groups.clone())
    .await
    .unwrap();
    assert_eq!(soft_plan.tasks.len(), 1);
    assert_eq!(soft_plan.tasks[0].fragments.len(), 2);

    let hard_plan = DefaultCompactionPlanner::new(CompactionOptions {
        max_source_fragments: Some(1),
        max_source_bytes: Some(u64::MAX),
        ..options.clone()
    })
    .unwrap()
    .plan_groups(&missing_size, groups)
    .await
    .unwrap();
    assert!(hard_plan.tasks.is_empty(), "{:?}", hard_plan.tasks);

    let metrics = compact_files_with_planner(
        &mut dataset,
        None,
        &Groups(
            vec![group(&[&after[0], &after[1]], &[("group", "b")])],
            CompactionOptions {
                max_source_fragments: Some(1),
                ..options.clone()
            },
        ),
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(dataset.manifest.fragments.as_ref(), &after);
    compact_files_with_planner(
        &mut dataset,
        None,
        &Groups(
            vec![group(&[&after[0], &after[1]], &[("group", "b")])],
            CompactionOptions {
                max_source_fragments: Some(2),
                ..options.clone()
            },
        ),
    )
    .await
    .unwrap();
    let final_ids = dataset
        .manifest
        .fragments
        .iter()
        .map(|f| f.id)
        .collect::<Vec<_>>();
    assert_eq!(final_ids, vec![4, 5]);
    assert_eq!(dataset.manifest.fragments[0], *output);
    assert_eq!(
        dataset.manifest.fragments[1]
            .metadata
            .get("group")
            .map(String::as_str),
        Some("b")
    );
    assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
}
/// A group is a boundary, not an order: ordinary candidacy decides which
/// members compact. Two small fragments compact together while a member
/// already at the target stays untouched, and a group of one small
/// fragment plans nothing.
#[tokio::test]
async fn grouped_compaction_applies_ordinary_candidacy_inside_the_group() {
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let mut dataset: Option<Dataset> = None;
    for (start, rows) in [(0_i64, 100_i64), (100, 200), (300, 1_000)] {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(start..start + rows))],
        )
        .unwrap();
        dataset = Some(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                &dir,
                Some(WriteParams {
                    mode: if dataset.is_none() {
                        WriteMode::Create
                    } else {
                        WriteMode::Append
                    },
                    max_rows_per_file: 1_000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
    }
    let mut dataset = dataset.unwrap();
    let before = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(
        before.iter().map(|f| f.physical_rows).collect::<Vec<_>>(),
        vec![Some(100), Some(200), Some(1_000)]
    );
    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap();
    let plan = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: before.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    // A lone small fragment has no neighbour in its group: nothing to do.
    let alone = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: vec![before[0].clone()],
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert!(alone.tasks.is_empty(), "{:?}", alone.tasks);

    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        results.push(task.execute(&dataset).await.unwrap());
    }
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap();
    let after = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(after.iter().map(|f| f.id).collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(after[0], before[2], "the full fragment is untouched");
    assert_eq!(after[1].physical_rows, Some(300));
    assert_eq!(dataset.count_rows(None).await.unwrap(), 1_300);
}
/// Three fragments of 100, 200 and 1,000 rows, in that order.
async fn three_sizes(uri: &str) -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let mut dataset: Option<Dataset> = None;
    for (start, rows) in [(0_i64, 100_i64), (100, 200), (300, 1_000)] {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(start..start + rows))],
        )
        .unwrap();
        dataset = Some(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                uri,
                Some(WriteParams {
                    mode: if dataset.is_none() {
                        WriteMode::Create
                    } else {
                        WriteMode::Append
                    },
                    max_rows_per_file: 1_000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
    }
    dataset.unwrap()
}

/// Replace a readable index on `column` with one of a type this build has no
/// reader for, covering exactly `covered`.
async fn plant_unreadable_index(dataset: &mut Dataset, column: &str, covered: &[u32]) {
    use crate::dataset::transaction::{Operation, Transaction};
    use lance_table::format::IndexMetadata;

    dataset
        .create_index(
            &[column],
            IndexType::BTree,
            Some("a_idx".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    let current = dataset.load_indices().await.unwrap();
    let unknown = IndexMetadata {
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/example.ForeignIndexDetails".to_string(),
            value: Vec::new(),
        })),
        fragment_bitmap: Some(RoaringBitmap::from_iter(covered.iter().copied())),
        ..current[0].clone()
    };
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![unknown],
                    removed_indices: current.to_vec(),
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
}

/// A source covered by an index this build cannot read is held out of every
/// task instead of refusing the plan: a covered member candidacy leaves alone
/// never mattered, a covered member that would have been rewritten is simply
/// left uncompacted, and the members on either side of the gap are not planned
/// together. Stable row ids and deferred remapping hold nothing back: there an
/// index's coverage is fixed up from manifest metadata alone.
#[tokio::test]
async fn grouped_compaction_holds_unreadable_coverage_back_from_its_group() {
    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap();

    // The covered member is the full-size one; candidacy leaves it alone,
    // so the two small members still plan.
    let dir = TempStrDir::default();
    let mut dataset = three_sizes(&dir).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[2]).await;
    let plan = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: before.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    // Both members of the group are covered: the plan is empty rather than
    // refused, and nothing is written.
    let dir = TempStrDir::default();
    let mut dataset = three_sizes(&dir).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[0, 1]).await;
    let version = dataset.manifest.version;
    let plan = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: before[..2].to_vec(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{:?}", plan.tasks);
    assert_eq!(dataset.manifest.version, version);

    // The source budget drops the first group's task; the second group is
    // fully covered and contributes nothing either way.
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..400))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let four = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[2, 3]).await;
    let groups = vec![
        CompactionGroup {
            fragments: four[..2].to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        CompactionGroup {
            fragments: four[2..].to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
    ];
    let budgeted = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        max_source_fragments: Some(2),
        ..Default::default()
    })
    .unwrap();
    let plan = budgeted
        .plan_groups(&dataset, groups.clone())
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    // Without the budget the covered group is still held back, not refused.
    let plan = planner.plan_groups(&dataset, groups).await.unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    // One of two small members is covered: the other is left alone rather
    // than compacted on its own.
    let dir = TempStrDir::default();
    let mut dataset = three_sizes(&dir).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[0]).await;
    let plan = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: before[..2].to_vec(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{:?}", plan.tasks);

    // Stable row ids hold nothing back: coverage is fixed up from manifest
    // metadata alone, so the same coverage plans both small members.
    let mut dataset = stable_row_id_dataset("memory://", 3).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "i", &[0, 1]).await;
    let plan = planner
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: before[..2].to_vec(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );

    // Deferred remapping holds nothing back either: it remaps no index here.
    let dir = TempStrDir::default();
    let mut dataset = three_sizes(&dir).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[0, 1]).await;
    let plan = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        ..Default::default()
    })
    .unwrap()
    .plan_groups(
        &dataset,
        vec![CompactionGroup {
            fragments: before[..2].to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        }],
    )
    .await
    .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}
/// A group fully covered by an index this build cannot read does not stop an
/// uncovered group: the plan holds exactly the uncovered group's task, and
/// none of the covered group's fragments appear anywhere in it. The held-back
/// index name and fragment ids go to the info log, which this test cannot
/// observe without a log capture crate, so only the plan shape is asserted.
#[tokio::test]
async fn grouped_compaction_plans_an_uncovered_group_beside_a_covered_one() {
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..400))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let four = dataset.manifest.fragments.as_ref().clone();
    // Group A is fragments 0 and 1, both covered; group B is 2 and 3.
    plant_unreadable_index(&mut dataset, "a", &[0, 1]).await;

    let plan = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap()
    .plan_groups(
        &dataset,
        vec![
            CompactionGroup {
                fragments: four[..2].to_vec(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            },
            CompactionGroup {
                fragments: four[2..].to_vec(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            },
        ],
    )
    .await
    .unwrap();

    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert!(
        !plan
            .tasks
            .iter()
            .flat_map(|task| &task.fragments)
            .any(|fragment| fragment.id == 0 || fragment.id == 1),
        "{:?}",
        plan.tasks
    );
}
/// The uncovered group's plan actually executes and commits while the covered
/// group is left exactly as it was: its fragments keep their ids and files, the
/// index this build cannot read keeps its coverage, and every row of the table
/// survives. A plan whose every source is held back writes nothing at all.
#[tokio::test]
async fn grouped_compaction_commit_leaves_the_held_back_group_untouched() {
    async fn four_fragments(dir: &TempStrDir) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..400))],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir,
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn values(dataset: &Dataset, predicate: Option<&str>) -> Vec<i64> {
        let mut scanner = dataset.scan();
        scanner.project(&["a"]).unwrap();
        if let Some(predicate) = predicate {
            scanner.filter(predicate).unwrap();
        }
        let batch = scanner.try_into_batch().await.unwrap();
        let mut values = batch
            .column_by_name("a")
            .unwrap()
            .as_primitive::<Int64Type>()
            .values()
            .to_vec();
        values.sort_unstable();
        values
    }

    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap();

    // Group A is fragments 0 and 1, both covered; group B is 2 and 3.
    let dir = TempStrDir::default();
    let mut dataset = four_fragments(&dir).await;
    let four = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[0, 1]).await;
    let planted = load_all_indices(&Dataset::open(&dir).await.unwrap())
        .await
        .unwrap()
        .as_ref()
        .clone();
    assert_eq!(planted.len(), 1);

    let groups = vec![
        CompactionGroup {
            fragments: four[..2].to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        CompactionGroup {
            fragments: four[2..].to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
    ];
    let planned = Dataset::open(&dir).await.unwrap();
    let plan = planner.plan_groups(&planned, groups).await.unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);

    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        results.push(task.execute(&planned).await.unwrap());
    }
    let mut committer = Dataset::open(&dir).await.unwrap();
    commit_compaction(
        &mut committer,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap();

    // The covered group's fragments are byte-for-byte what they were.
    let after = committer.manifest.fragments.as_ref().clone();
    assert_eq!(
        after.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 1, 4]
    );
    assert_eq!(after[0], four[0], "the covered fragment 0 is untouched");
    assert_eq!(after[1], four[1], "the covered fragment 1 is untouched");
    assert_eq!(after[2].physical_rows, Some(200), "2 and 3 compacted");

    // The index this build cannot read is still there, byte-for-byte the
    // metadata that was planted: nothing of it had to be remapped.
    let indices = load_all_indices(&committer).await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0], planted[0],
        "the unreadable index's metadata is untouched"
    );

    // Every row survives, and the uncovered group's own rows are exact.
    assert_eq!(committer.count_rows(None).await.unwrap(), 400);
    assert_eq!(values(&committer, None).await, (0..400).collect::<Vec<_>>());
    assert_eq!(
        values(&committer, Some("a >= 200")).await,
        (200..400).collect::<Vec<_>>(),
        "the rewritten group's rows are preserved exactly"
    );

    // Every source held back: no task, so nothing is rewritten and the commit
    // of the empty plan leaves the manifest where it was.
    let dir = TempStrDir::default();
    let mut dataset = four_fragments(&dir).await;
    let four = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[0, 1, 2, 3]).await;
    let planned = Dataset::open(&dir).await.unwrap();
    let plan = planner
        .plan_groups(
            &planned,
            vec![CompactionGroup {
                fragments: four.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{:?}", plan.tasks);

    let mut committer = Dataset::open(&dir).await.unwrap();
    let version = committer.manifest.version;
    let metrics = commit_compaction(
        &mut committer,
        Vec::new(),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(metrics.fragments_added, 0);
    assert_eq!(
        committer.manifest.version, version,
        "an empty plan commits nothing"
    );
    assert_eq!(committer.manifest.fragments.as_ref(), &four);
    assert_eq!(committer.count_rows(None).await.unwrap(), 400);
}
/// Grouped planning leaves exactly the gap ordinary planning leaves at a
/// source held back for unreadable index coverage. Ordinary planning folds
/// that coverage into its own exclusions and terminates the current bin at
/// one, so the candidates on either side are never planned together; the same
/// group handed to `plan_groups`, with the same options, has to select the
/// same sources.
#[tokio::test]
async fn grouped_planning_keeps_the_same_gap_at_a_held_back_source_as_ordinary_planning() {
    /// A dataset of `sizes` fragments in order, one fragment per entry.
    async fn fragments_of(dir: &TempStrDir, sizes: &[i64]) -> (Dataset, Vec<Fragment>) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let mut dataset: Option<Dataset> = None;
        let mut start = 0_i64;
        for rows in sizes {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from_iter_values(start..start + rows))],
            )
            .unwrap();
            dataset = Some(
                Dataset::write(
                    RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                    dir,
                    Some(WriteParams {
                        mode: if dataset.is_none() {
                            WriteMode::Create
                        } else {
                            WriteMode::Append
                        },
                        max_rows_per_file: 1_000,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
            start += rows;
        }
        let dataset = dataset.unwrap();
        let fragments = dataset.manifest.fragments.as_ref().clone();
        assert_eq!(fragments.len(), sizes.len());
        (dataset, fragments)
    }

    fn sources(plan: &CompactionPlan) -> Vec<Vec<u64>> {
        plan.tasks
            .iter()
            .map(|task| task.fragments.iter().map(|f| f.id).collect())
            .collect()
    }

    // The one set of options both planners run under: the large middle
    // fragment is over the target, so `candidacy_of` rejects it either way.
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };

    // Two small, one over-target and covered, two small. Both planners plan
    // the two sides as separate tasks and leave the covered member alone.
    let dir = TempStrDir::default();
    let (mut dataset, five) = fragments_of(&dir, &[100, 100, 1_000, 100, 100]).await;
    plant_unreadable_index(&mut dataset, "a", &[2]).await;
    let dataset = Dataset::open(&dir).await.unwrap();

    let ordinary = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(
        sources(&ordinary),
        vec![vec![0, 1], vec![3, 4]],
        "ordinary planning splits its bin at the excluded source"
    );

    let grouped = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: five.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert_eq!(
        sources(&grouped),
        sources(&ordinary),
        "grouped planning selects the same sources as ordinary planning"
    );

    // One small on each side of the covered member: the gap leaves each a
    // lone fragment that would only gain from neighbours, which neither
    // planner rewrites.
    let dir = TempStrDir::default();
    let (mut dataset, three) = fragments_of(&dir, &[100, 1_000, 100]).await;
    plant_unreadable_index(&mut dataset, "a", &[1]).await;
    let dataset = Dataset::open(&dir).await.unwrap();

    let ordinary = plan_compaction(&dataset, &options).await.unwrap();
    assert!(ordinary.tasks.is_empty(), "{:?}", ordinary.tasks);
    let grouped = DefaultCompactionPlanner::new(options)
        .unwrap()
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: three.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert!(grouped.tasks.is_empty(), "{:?}", grouped.tasks);
}
/// Only a held-back source terminates a grouped run; an ordinary
/// non-candidate does not. Ordinary planning ends its bin at every
/// non-candidate, so an over-target fragment splits its plan too. Grouped
/// planning skips that fragment and keeps the candidates around it in one
/// task, because naming them in a group is an explicit request to compact
/// them together; only coverage this build cannot remap overrides it.
#[tokio::test]
async fn grouped_planning_splits_at_a_held_back_source_but_not_at_an_over_target_one() {
    /// A dataset of `sizes` fragments in order, one fragment per entry.
    async fn fragments_of(dir: &TempStrDir, sizes: &[i64]) -> (Dataset, Vec<Fragment>) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let mut dataset: Option<Dataset> = None;
        let mut start = 0_i64;
        for rows in sizes {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from_iter_values(start..start + rows))],
            )
            .unwrap();
            dataset = Some(
                Dataset::write(
                    RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
                    dir,
                    Some(WriteParams {
                        mode: if dataset.is_none() {
                            WriteMode::Create
                        } else {
                            WriteMode::Append
                        },
                        max_rows_per_file: 1_000,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap(),
            );
            start += rows;
        }
        let dataset = dataset.unwrap();
        let fragments = dataset.manifest.fragments.as_ref().clone();
        assert_eq!(fragments.len(), sizes.len());
        (dataset, fragments)
    }

    fn sources(plan: &CompactionPlan) -> Vec<Vec<u64>> {
        plan.tasks
            .iter()
            .map(|task| task.fragments.iter().map(|f| f.id).collect())
            .collect()
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    };

    // Fragment 2 is over the target, so neither planner rewrites it, and it is
    // not held back. Fragment 4 is small and covered, so it is held back.
    let dir = TempStrDir::default();
    let (mut dataset, six) = fragments_of(&dir, &[100, 100, 1_000, 100, 100, 100]).await;
    plant_unreadable_index(&mut dataset, "a", &[4]).await;
    let dataset = Dataset::open(&dir).await.unwrap();

    let grouped = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: six.clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
    assert_eq!(
        sources(&grouped),
        vec![vec![0, 1, 3]],
        "the over-target member 2 does not split the run, the held-back member 4 does, and \
         fragment 5 is left alone past the split"
    );

    // The same layout through ordinary planning: its bin ends at the
    // over-target member as well, so the run before it is a task of its own.
    let ordinary = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(
        sources(&ordinary),
        vec![vec![0, 1]],
        "ordinary planning splits at the over-target member too, leaving 3 alone between the \
         two gaps"
    );
}
/// A group that mixes covered and uncovered sources still plans the uncovered
/// ones: the covered member is held back and the two uncovered members on one
/// side of it compact together.
#[tokio::test]
async fn grouped_compaction_plans_the_uncovered_sources_of_a_mixed_group() {
    let dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..400))],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let four = dataset.manifest.fragments.as_ref().clone();
    plant_unreadable_index(&mut dataset, "a", &[1]).await;

    let plan = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap()
    .plan_groups(
        &dataset,
        vec![CompactionGroup {
            fragments: four.clone(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        }],
    )
    .await
    .unwrap();

    // Fragment 0 is left alone: the gap at 1 terminates its run, and a lone
    // neighbours-only fragment is not rewritten.
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}
/// No metric of a held-back member is read: its data file is gone, so any
/// measurement of it would fail the plan, and the manifest's cached counts are
/// made untrustworthy so that every measured member really does open its file.
#[tokio::test]
async fn grouped_compaction_reads_no_metrics_for_a_held_back_source() {
    /// Four fragments of 100 rows, with fragment 1's data file removed and the
    /// writer version cleared so that the cached counts cannot be trusted and
    /// measuring fragment 1 must fail.
    async fn four_with_an_unreadable_second(dir: &TempStrDir) -> (Dataset, Vec<Fragment>) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..400))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            dir,
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let four = dataset.manifest.fragments.as_ref().clone();
        (dataset, four)
    }

    /// Returns the path removed, so a caller can relate it to the fragment the
    /// planner is expected to hold back.
    fn sabotage(
        dataset: &mut Dataset,
        fragment: &Fragment,
        dir: &TempStrDir,
    ) -> std::path::PathBuf {
        Arc::make_mut(&mut dataset.manifest).writer_version = None;
        let path = std::path::Path::new(dir.as_ref())
            .join("data")
            .join(&fragment.files[0].path);
        std::fs::remove_file(&path).unwrap();
        path
    }

    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..Default::default()
    })
    .unwrap();
    let whole_group = |four: &[Fragment]| {
        vec![CompactionGroup {
            fragments: four.to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        }]
    };

    // The control: with no index holding it back, fragment 1 is measured and
    // its missing file fails the plan. This is what makes the case below a
    // proof that the held-back member is never measured.
    let dir = TempStrDir::default();
    let (mut dataset, four) = four_with_an_unreadable_second(&dir).await;
    sabotage(&mut dataset, &four[1], &dir);
    assert!(
        planner
            .plan_groups(&dataset, whole_group(&four))
            .await
            .is_err()
    );

    // The same unreadable fragment, now covered by an index this build cannot
    // read: no metric of it is read, the plan succeeds, and the uncovered
    // members on the far side of the gap it leaves still compact together.
    let dir = TempStrDir::default();
    let (mut dataset, four) = four_with_an_unreadable_second(&dir).await;
    plant_unreadable_index(&mut dataset, "a", &[1]).await;
    let mut dataset = Dataset::open(&dir).await.unwrap();
    let held = dataset
        .fragments()
        .iter()
        .find(|fragment| fragment.id == four[1].id)
        .cloned()
        .expect("the held-back fragment survives the index commit");
    let removed = sabotage(&mut dataset, &held, &dir);
    // The sabotage only proves anything if the index the reopened manifest
    // carries is the unreadable one, and if the fragment whose file was
    // removed is exactly the fragment that index's coverage holds back.
    // `load_all_indices`, not `load_indices`: the latter drops exactly the
    // unusable entry this case is about.
    let indices = load_all_indices(&dataset).await.unwrap();
    assert_eq!(indices.len(), 1, "{indices:?}");
    assert!(
        indices[0]
            .index_details
            .as_ref()
            .is_some_and(|details| details.type_url.ends_with("example.ForeignIndexDetails")),
        "{indices:?}"
    );
    assert_eq!(
        indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![four[1].id as u32]
    );
    assert_eq!(held.files[0].path, four[1].files[0].path);
    assert!(removed.ends_with(&held.files[0].path), "{removed:?}");
    let plan = planner
        .plan_groups(&dataset, whole_group(&four))
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 1, "{:?}", plan.tasks);
    assert_eq!(
        plan.tasks[0]
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}
/// With stable row ids and deferred remapping, a non-adjacent group's
/// moves are recorded in the fragment reuse index and every row keeps its
/// id and value.
#[tokio::test]
async fn grouped_compaction_records_moves_for_non_adjacent_sources() {
    let mut dataset = stable_row_id_dataset("memory://", 4).await;
    let before = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(before.len(), 4);

    async fn rows_by_id(dataset: &Dataset) -> Vec<(u64, i32)> {
        let mut scanner = dataset.scan();
        scanner.with_row_id().project(&["i"]).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap();
        let values = batch
            .column_by_name("i")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut rows = ids
            .values()
            .iter()
            .copied()
            .zip(values.values().iter().copied())
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    }
    let expected = rows_by_id(&dataset).await;

    let plan = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        ..Default::default()
    })
    .unwrap()
    .plan_groups(
        &dataset,
        vec![CompactionGroup {
            fragments: vec![before[0].clone(), before[2].clone()],
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        }],
    )
    .await
    .unwrap();
    assert_eq!(plan.tasks.len(), 1);
    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        let result = task.execute(&dataset).await.unwrap();
        assert!(result.row_addrs.is_some());
        results.push(result);
    }
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap();

    let after = dataset.manifest.fragments.as_ref().clone();
    assert_eq!(
        after.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![1, 3, 4]
    );
    assert_eq!(after[0], before[1]);
    assert_eq!(after[1], before[3]);
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "a deferred rewrite over non-adjacent sources must record its moves"
    );
    assert_eq!(rows_by_id(&dataset).await, expected);
}
/// Between planning and commit, an unrelated append lands alongside grouped
/// results and keeps its own fragment, while a change to one of the
/// group's own sources is refused and no deleted row comes back.
#[tokio::test]
async fn grouped_compaction_commit_lands_with_an_append_and_refuses_a_source_change() {
    async fn four_fragments(uri: &str) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..400))],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn plan_and_execute(dataset: &Dataset) -> (CompactionPlan, Vec<RewriteResult>) {
        let before = dataset.manifest.fragments.as_ref().clone();
        let plan = DefaultCompactionPlanner::new(CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        })
        .unwrap()
        .plan_groups(
            dataset,
            vec![CompactionGroup {
                fragments: vec![before[0].clone(), before[2].clone()],
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        let mut results = Vec::new();
        for task in plan.compaction_tasks() {
            results.push(task.execute(dataset).await.unwrap());
        }
        (plan, results)
    }

    // A harmless append between plan and commit: both land.
    let dir = TempStrDir::default();
    let planned = four_fragments(&dir).await;
    let (plan, results) = plan_and_execute(&planned).await;
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let appended = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(400..450))],
    )
    .unwrap();
    let other = Dataset::write(
        RecordBatchIterator::new(vec![Ok(appended)], schema),
        &dir,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let appended_fragment = other.manifest.fragments.last().unwrap().clone();
    assert_eq!(appended_fragment.id, 4);
    let mut committer = Dataset::open(&dir).await.unwrap();
    commit_compaction(
        &mut committer,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap();
    let after = committer.manifest.fragments.as_ref().clone();
    assert_eq!(
        after.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![1, 3, 4, 5]
    );
    assert_eq!(after[2], appended_fragment, "the append keeps its fragment");
    assert_eq!(after[3].physical_rows, Some(200));
    assert_eq!(committer.count_rows(None).await.unwrap(), 450);

    // A delete on one of the group's sources between plan and commit: the
    // commit is refused and the deleted rows stay deleted.
    let dir = TempStrDir::default();
    let planned = four_fragments(&dir).await;
    let (plan, results) = plan_and_execute(&planned).await;
    let mut other = Dataset::open(&dir).await.unwrap();
    other.delete("a < 10").await.unwrap();
    let mut committer = Dataset::open(&dir).await.unwrap();
    let error = commit_compaction(
        &mut committer,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("retryable") || error.contains("conflict") || error.contains("preempted"),
        "{error}"
    );
    let latest = Dataset::open(&dir).await.unwrap();
    assert_eq!(
        latest
            .manifest
            .fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "no rewrite landed"
    );
    assert_eq!(
        latest.count_rows(Some("a < 10".to_string())).await.unwrap(),
        0
    );
    assert_eq!(latest.count_rows(None).await.unwrap(), 390);
}
#[rstest]
#[tokio::test]
async fn test_compact_empty(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Compact an empty table
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

    let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
    let mut dataset = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let plan = plan_compaction(&dataset, &CompactionOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.tasks().len(), 0);

    let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    assert_eq!(metrics, CompactionMetrics::default());
    assert_eq!(dataset.manifest.version, 1);
}
#[rstest]
#[case::default(None, LanceFileVersion::V2_0)]
#[case::stable(Some(LanceFileVersion::Stable), LanceFileVersion::V2_2)]
#[case::next(Some(LanceFileVersion::Next), LanceFileVersion::V2_3)]
#[tokio::test]
async fn plan_compaction_freezes_storage_version_selector(
    #[case] selector: Option<LanceFileVersion>,
    #[case] exact: LanceFileVersion,
) {
    let data = arrow_array::record_batch!(("id", Int32, [1, 2, 3, 4, 5, 6])).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(data.clone())], data.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let options = CompactionOptions {
        data_storage_version: selector,
        excluded_fragment_ids: vec![2],
        ..Default::default()
    };

    let plan = plan_compaction(&dataset, &options).await.unwrap();

    assert_eq!(plan.options.data_storage_version, Some(exact));
    let serialized = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        serialized["options"]["data_storage_version"],
        exact.to_string()
    );
    assert_eq!(
        serde_json::from_value::<CompactionPlan>(serialized).unwrap(),
        plan
    );
    assert_eq!(plan.num_tasks(), 1);
    let retained = dataset.manifest.fragments[2].clone();
    let task = plan.compaction_tasks().next().unwrap();
    let task: CompactionTask = serde_json::from_slice(&serde_json::to_vec(&task).unwrap()).unwrap();
    let result = task.execute(&dataset).await.unwrap();
    assert!(
        result
            .new_fragments
            .iter()
            .flat_map(Fragment::referenced_lance_files)
            .all(|file| file.file_version().unwrap() == exact.resolve())
    );
    let result: RewriteResult =
        serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
    commit_compaction(
        &mut dataset,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    assert!(
        dataset
            .manifest
            .fragments
            .iter()
            .any(|fragment| fragment == &retained)
    );
    assert_eq!(
        dataset.manifest.data_storage_format.lance_file_format(),
        ConcreteFileVersion::V2_0
    );
    let actual = dataset
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("id".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(actual, data);
}
#[rstest]
#[case(LanceFileVersion::Legacy, LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_0, LanceFileVersion::Legacy)]
#[tokio::test]
async fn compaction_rejects_cross_family_target(
    #[case] source: LanceFileVersion,
    #[case] target: LanceFileVersion,
) {
    let batch = arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(source),
            max_rows_per_file: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let options = CompactionOptions {
        data_storage_version: Some(target),
        ..Default::default()
    };
    let error = plan_compaction(&dataset, &options).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("V1 and V2 storage versions cannot be mixed")
    );
    // Standalone distributed tasks also validate, independently of the planner.
    let mut task = plan_compaction(&dataset, &CompactionOptions::default())
        .await
        .unwrap()
        .compaction_tasks()
        .next()
        .unwrap();
    task.options = options;
    let error = task.execute(&dataset).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("V1 and V2 storage versions cannot be mixed")
    );
}
#[rstest]
#[case::pq("pq", 1_000)]
#[case::rq("rq", 1_000)]
#[case::hnsw("hnsw", 1_000)]
#[case::no_window("flat", 1)]
#[tokio::test]
async fn test_compaction_planning_skips_ineligible_index_io(
    #[case] layout: &str,
    #[case] target_rows_per_fragment: usize,
) {
    const DIM: u32 = 8;
    let dir = TempStrDir::default();
    let reader = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_reader_rows(RowCount::from(16), BatchCount::from(1));
    let mut dataset = Dataset::write(
        reader,
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 8,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from(vec![0.0; DIM as usize]),
        DIM as i32,
    )
    .unwrap();
    let ivf = IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap();
    let params = match layout {
        "pq" => VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf,
            PQBuildParams {
                num_bits: 4,
                num_sub_vectors: 2,
                max_iters: 1,
                ..Default::default()
            },
        ),
        "rq" => {
            VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf, RQBuildParams::default())
        }
        "hnsw" => VectorIndexParams::ivf_hnsw(DistanceType::L2, ivf, HnswBuildParams::default()),
        _ => VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf),
    };
    let template = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
        .name("vec_idx".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    // Readable files make accidental opens observable: failed reads alone
    // would not increment the object store's successful-read counter.
    let mut indices = Vec::new();
    for fragment in 0..2 {
        let mut segment = template.clone();
        segment.uuid = Uuid::new_v4();
        segment.fragment_bitmap = Some(RoaringBitmap::from_iter([fragment]));
        for file in template.files.as_ref().unwrap() {
            dataset
                .object_store
                .copy(
                    &dataset
                        .indices_dir()
                        .join(template.uuid.to_string())
                        .join(file.path.as_str()),
                    &dataset
                        .indices_dir()
                        .join(segment.uuid.to_string())
                        .join(file.path.as_str()),
                )
                .await
                .unwrap();
        }
        indices.push(segment);
    }
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: indices,
                    removed_indices: Vec::new(),
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    // Keep metadata-loading I/O separate from grouping (legacy inference can
    // read index files; these fixtures have explicit modern manifest details).
    load_all_indices(&dataset).await.unwrap();
    dataset.object_store.io_stats_incremental();
    let plan = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment,
        ..Default::default()
    })
    .unwrap()
    .with_index_remapper(&DatasetIndexRemapperOptions::default())
    .plan(&dataset)
    .await
    .unwrap();
    assert_eq!(plan.num_tasks(), 0);
    assert_eq!(dataset.object_store.io_stats_incremental().read_iops, 0);
    dataset
        .open_logical_vector_index("vec", "vec_idx")
        .await
        .unwrap();
    assert!(
        dataset.object_store.io_stats_incremental().read_iops > 0,
        "opening the cold, readable segments must register index-file I/O"
    );
}
#[tokio::test]
async fn test_default_compaction_planner() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();
    let schema = data.schema();

    // Create dataset with multiple small fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 2000,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    assert_eq!(dataset.get_fragments().len(), 5);

    // Test default planner
    let options = CompactionOptions {
        target_rows_per_fragment: 5000,
        materialize_deletions_threshold: 2.0,
        ..Default::default()
    };

    let planner = DefaultCompactionPlanner::new(options).unwrap();
    let plan = planner.plan(&dataset).await.unwrap();

    // Should create tasks to compact small fragments
    assert!(!plan.tasks.is_empty());
    assert_eq!(plan.read_version, dataset.manifest.version);
    // make sure options.validate() worked
    assert!(!plan.options.materialize_deletions);
}
/// `num_threads` caps how many rewrite tasks run at once, and a cap of zero
/// never starts one, so compaction would wait forever instead of failing.
#[tokio::test]
async fn test_compact_files_rejects_zero_threads() {
    let test_dir = TempStrDir::default();
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let write_params = WriteParams {
        max_rows_per_file: 2000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, &test_dir, Some(write_params))
        .await
        .unwrap();

    let options = CompactionOptions {
        num_threads: Some(0),
        ..Default::default()
    };
    let error = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        compact_files(&mut dataset, options, None),
    )
    .await
    .expect("compaction with zero threads must fail instead of hanging")
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("num_threads must be greater than 0"),
        "{error}"
    );
}
#[test]
fn test_from_dataset_config() {
    let config = HashMap::from([
        (
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        ),
        (
            "lance.compaction.max_rows_per_group".to_string(),
            "2048".to_string(),
        ),
        (
            "lance.compaction.max_bytes_per_file".to_string(),
            "1000000".to_string(),
        ),
        (
            "lance.compaction.min_bytes_per_file".to_string(),
            "950000".to_string(),
        ),
        (
            "lance.compaction.materialize_deletions".to_string(),
            "false".to_string(),
        ),
        (
            "lance.compaction.materialize_deletions_threshold".to_string(),
            "0.25".to_string(),
        ),
        (
            "lance.compaction.defer_index_remap".to_string(),
            "true".to_string(),
        ),
        (
            "lance.compaction.batch_size".to_string(),
            "4096".to_string(),
        ),
        (
            "lance.compaction.io_buffer_size".to_string(),
            "1073741824".to_string(),
        ),
        (
            "lance.compaction.compaction_mode".to_string(),
            "try_binary_copy".to_string(),
        ),
        (
            "lance.compaction.binary_copy_read_batch_bytes".to_string(),
            "8388608".to_string(),
        ),
        (
            "lance.compaction.index_remap_mode".to_string(),
            "compact".to_string(),
        ),
        (
            "lance.compaction.max_source_fragments".to_string(),
            "20".to_string(),
        ),
        (
            "lance.compaction.max_source_rows".to_string(),
            "1000000".to_string(),
        ),
        (
            "lance.compaction.max_source_bytes".to_string(),
            "1073741824".to_string(),
        ),
        (
            "lance.compaction.source_budget_mode".to_string(),
            "soft".to_string(),
        ),
    ]);

    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(opts.target_rows_per_fragment, 500_000);
    assert_eq!(opts.max_rows_per_group, 2048);
    assert_eq!(opts.max_bytes_per_file, Some(1_000_000));
    assert_eq!(opts.min_bytes_per_file, Some(950_000));
    assert!(!opts.materialize_deletions);
    assert!((opts.materialize_deletions_threshold - 0.25).abs() < f32::EPSILON);
    assert!(opts.defer_index_remap);
    assert_eq!(opts.batch_size, Some(4096));
    assert_eq!(opts.io_buffer_size, Some(1_073_741_824));
    assert_eq!(opts.compaction_mode, Some(CompactionMode::TryBinaryCopy));
    assert_eq!(opts.binary_copy_read_batch_bytes, Some(8_388_608));
    // A non-default value proves the config string was actually parsed.
    assert_eq!(opts.index_remap_mode, IndexRemapMode::Compact);
    assert_eq!(opts.max_source_fragments, Some(20));
    assert_eq!(opts.max_source_rows, Some(1_000_000));
    assert_eq!(opts.max_source_bytes, Some(1_073_741_824));
    assert_eq!(opts.source_budget_mode, SourceBudgetMode::Soft);

    let mut serialized = serde_json::to_value(&opts).unwrap();
    assert_eq!(
        serde_json::from_value::<CompactionOptions>(serialized.clone()).unwrap(),
        opts
    );
    // Older serialized options omit newer defaults.
    serialized
        .as_object_mut()
        .unwrap()
        .remove("min_bytes_per_file");
    serialized
        .as_object_mut()
        .unwrap()
        .remove("source_budget_mode");
    let decoded = serde_json::from_value::<CompactionOptions>(serialized).unwrap();
    assert_eq!(decoded.min_bytes_per_file, None);
    assert_eq!(decoded.source_budget_mode, SourceBudgetMode::Hard);
}
#[test]
fn test_from_dataset_config_empty() {
    let config = HashMap::new();
    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    let defaults = CompactionOptions::default();
    assert_eq!(opts.min_bytes_per_file, None);
    assert_eq!(
        opts.target_rows_per_fragment,
        defaults.target_rows_per_fragment
    );
    assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
    assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
    assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
    assert_eq!(
        opts.materialize_deletions_threshold,
        defaults.materialize_deletions_threshold
    );
    assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
    assert_eq!(opts.index_remap_mode, defaults.index_remap_mode);
    assert_eq!(opts.index_remap_mode, IndexRemapMode::Direct);
    assert_eq!(opts.batch_size, defaults.batch_size);
    assert_eq!(opts.compaction_mode, defaults.compaction_mode);
    assert_eq!(
        opts.binary_copy_read_batch_bytes,
        defaults.binary_copy_read_batch_bytes
    );
}
#[test]
fn test_from_dataset_config_partial() {
    let config = HashMap::from([(
        "lance.compaction.target_rows_per_fragment".to_string(),
        "500000".to_string(),
    )]);

    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(opts.target_rows_per_fragment, 500_000);
    // Other fields should remain at defaults
    let defaults = CompactionOptions::default();
    assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
    assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
    assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
    assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
    assert_eq!(opts.batch_size, defaults.batch_size);
    assert_eq!(opts.compaction_mode, defaults.compaction_mode);
    assert_eq!(
        opts.binary_copy_read_batch_bytes,
        defaults.binary_copy_read_batch_bytes
    );
}
#[test]
fn test_from_dataset_config_ignores_other_keys() {
    let config = HashMap::from([
        (
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        ),
        (
            "lance.auto_cleanup.interval".to_string(),
            "3600".to_string(),
        ),
        ("some.other.key".to_string(), "value".to_string()),
    ]);

    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(opts.target_rows_per_fragment, 500_000);
}
#[rstest::rstest]
#[case::invalid_rows("target_rows_per_fragment", "not_a_number")]
#[case::zero_min_bytes("min_bytes_per_file", "0")]
#[case::negative_min_bytes("min_bytes_per_file", "-1")]
#[case::min_bytes_above_max("min_bytes_per_file", "1001")]
#[case::fractional_min_bytes("min_bytes_per_file", "0.95")]
fn test_from_dataset_config_invalid_value(#[case] key: &str, #[case] value: &str) {
    let config = HashMap::from([
        (format!("lance.compaction.{key}"), value.to_string()),
        (
            "lance.compaction.max_bytes_per_file".to_string(),
            "1000".to_string(),
        ),
    ]);
    let error = CompactionOptions::from_dataset_config(&config)
        .and_then(|mut options| options.validate())
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    let message = error.to_string();
    assert!(message.contains(key));
    assert!(message.contains(value));
}
#[test]
fn test_from_dataset_config_invalid_bool() {
    let config = HashMap::from([(
        "lance.compaction.materialize_deletions".to_string(),
        "yes".to_string(),
    )]);

    let result = CompactionOptions::from_dataset_config(&config);
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("materialize_deletions"));
    assert!(err_msg.contains("yes"));
}
#[test]
fn test_from_dataset_config_unknown_compaction_key() {
    // Unknown keys should be ignored (with a warning) for forwards compatibility
    let config = HashMap::from([(
        "lance.compaction.unknown_key".to_string(),
        "value".to_string(),
    )]);

    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    // Should return defaults since the unknown key is skipped
    let defaults = CompactionOptions::default();
    assert_eq!(
        opts.target_rows_per_fragment,
        defaults.target_rows_per_fragment
    );
}
#[test]
fn test_from_dataset_config_invalid_compaction_mode() {
    let config = HashMap::from([(
        "lance.compaction.compaction_mode".to_string(),
        "invalid_mode".to_string(),
    )]);

    let result = CompactionOptions::from_dataset_config(&config);
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("invalid_mode"));
}

#[test]
fn test_from_dataset_config_invalid_source_budget_mode() {
    let config = HashMap::from([(
        "lance.compaction.source_budget_mode".to_string(),
        "invalid".to_string(),
    )]);

    let err = CompactionOptions::from_dataset_config(&config).unwrap_err();
    assert!(err.to_string().contains("Invalid source budget mode"));
}
#[test]
fn test_from_dataset_config_max_overlays_per_fragment() {
    let key = "lance.compaction.max_overlays_per_fragment".to_string();

    // An integer sets the threshold.
    let config = HashMap::from([(key.clone(), "3".to_string())]);
    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(opts.max_overlays_per_fragment, Some(3));

    // "none" (case-insensitive) disables the trigger, overriding the Some(10) default.
    let config = HashMap::from([(key.clone(), "None".to_string())]);
    let opts = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(opts.max_overlays_per_fragment, None);

    // Anything else is rejected.
    let config = HashMap::from([(key, "not_a_number".to_string())]);
    let err_msg = CompactionOptions::from_dataset_config(&config)
        .unwrap_err()
        .to_string();
    assert!(err_msg.contains("max_overlays_per_fragment"));
    assert!(err_msg.contains("not_a_number"));
}
#[test]
fn test_apply_dataset_config_overrides() {
    let config = HashMap::from([(
        "lance.compaction.target_rows_per_fragment".to_string(),
        "500000".to_string(),
    )]);

    let mut opts = CompactionOptions {
        max_rows_per_group: 4096,
        ..Default::default()
    };
    opts.apply_dataset_config(&config).unwrap();

    // Config value should be applied
    assert_eq!(opts.target_rows_per_fragment, 500_000);
    // Explicitly set value should be preserved (config didn't have this key)
    assert_eq!(opts.max_rows_per_group, 4096);
}
#[test]
fn test_apply_dataset_config_overwrites_matching_field() {
    let config = HashMap::from([(
        "lance.compaction.max_rows_per_group".to_string(),
        "2048".to_string(),
    )]);

    let mut opts = CompactionOptions {
        max_rows_per_group: 4096,
        ..Default::default()
    };
    opts.apply_dataset_config(&config).unwrap();

    // Config value should overwrite the pre-set value
    assert_eq!(opts.max_rows_per_group, 2048);
}
#[tokio::test]
async fn test_max_source_fragments() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;

    // Plan without limit - all 10 fragments should be candidates.
    // Use a target that splits the 10 fragments into multiple tasks.
    let opts_no_limit = CompactionOptions {
        target_rows_per_fragment: 250,
        ..Default::default()
    };
    let plan_all = plan_compaction(&dataset, &opts_no_limit).await.unwrap();
    let total_source_frags: usize = plan_all.tasks().iter().map(|t| t.fragments.len()).sum();
    assert_eq!(total_source_frags, 10);
    assert!(
        plan_all.num_tasks() > 2,
        "need multiple tasks to test bounding, got {}",
        plan_all.num_tasks()
    );

    // Plan with max_source_fragments=4 should include tasks covering <= 4
    // source fragments
    let opts_bounded = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_fragments: Some(4),
        ..Default::default()
    };
    let plan_bounded = plan_compaction(&dataset, &opts_bounded).await.unwrap();
    let bounded_source_frags: usize = plan_bounded.tasks().iter().map(|t| t.fragments.len()).sum();
    assert!(
        bounded_source_frags <= 4,
        "expected at most 4 source fragments, got {bounded_source_frags}"
    );
    assert!(
        bounded_source_frags > 0,
        "expected at least 1 source fragment in bounded plan"
    );
    assert!(
        plan_bounded.num_tasks() < plan_all.num_tasks(),
        "bounded plan ({}) should have fewer tasks than unbounded ({})",
        plan_bounded.num_tasks(),
        plan_all.num_tasks()
    );

    // Execute bounded compaction incrementally
    let mut dataset = dataset;
    compact_files(&mut dataset, opts_bounded, None)
        .await
        .unwrap();
    let after_first = dataset.get_fragments().len();
    assert!(
        after_first < 10,
        "expected fewer than 10 fragments after first compaction, got {after_first}"
    );
    assert!(
        after_first > 1,
        "expected partial compaction (not fully compacted), got {after_first}"
    );

    // Run again to make more progress
    let opts_bounded = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_fragments: Some(4),
        ..Default::default()
    };
    compact_files(&mut dataset, opts_bounded, None)
        .await
        .unwrap();
    let after_second = dataset.get_fragments().len();
    assert!(
        after_second <= after_first,
        "expected progress: {after_second} should be <= {after_first}"
    );
}
#[tokio::test]
async fn test_excluded_fragments_are_planning_boundaries() {
    let test_dir = TempStrDir::default();
    let mut dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let excluded_fragment_id = 4;
    let options = CompactionOptions {
        target_rows_per_fragment: 250,
        excluded_fragment_ids: vec![excluded_fragment_id, excluded_fragment_id, u32::MAX],
        ..Default::default()
    };

    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let planned_fragment_ids = plan
        .tasks()
        .iter()
        .map(|task| {
            task.fragments
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(
        planned_fragment_ids,
        vec![vec![0, 1, 2, 3], vec![5, 6, 7, 8, 9]]
    );
    assert!(
        planned_fragment_ids
            .iter()
            .flatten()
            .all(|fragment_id| *fragment_id != excluded_fragment_id)
    );

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, 9);
    let remaining_fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    assert!(remaining_fragment_ids.contains(&excluded_fragment_id));
}
#[tokio::test]
async fn test_max_source_rows() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;

    // The first task covers fragments 0..=2: 250 live rows (fragment 0
    // keeps 50 after the deletes, and its overlay adds none) but 300
    // physical rows. A budget between the two admits it and only it,
    // proving the budget is enforced against live rows.
    let opts = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_rows: Some(270),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &opts).await.unwrap();
    assert_eq!(plan.num_tasks(), 1);
    let physical_rows: usize = plan
        .tasks()
        .iter()
        .flat_map(|t| &t.fragments)
        .map(|f| f.physical_rows.unwrap())
        .sum();
    assert_eq!(physical_rows, 300);
}

#[tokio::test]
async fn test_soft_source_budget_admits_one_oversized_task() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;

    // The first task contains three fragments and 250 live rows, and its
    // source files are larger than one byte. In soft mode it is admitted
    // so compaction can make progress, but the next task is not.
    let opts = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_fragments: Some(1),
        max_source_rows: Some(1),
        max_source_bytes: Some(1),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    };
    let result = DefaultCompactionPlanner::new(opts)
        .unwrap()
        .plan_with_budget(&dataset)
        .await
        .unwrap();
    let plan = result.plan;
    let summary = result.source_budget;
    assert_eq!(summary.selected_source_rows, 250);
    assert_eq!(summary.selected_source_fragments, 3);
    assert!(summary.selected_source_bytes.is_some_and(|bytes| bytes > 1));
    assert!(summary.budget_exhausted());
    assert_eq!(
        summary.limiting_budgets,
        vec![
            SourceBudgetLimit::Fragments,
            SourceBudgetLimit::Rows,
            SourceBudgetLimit::Bytes
        ]
    );

    assert_eq!(plan.num_tasks(), 1);
    assert_eq!(plan.tasks()[0].fragments.len(), 3);
}

#[tokio::test]
async fn test_soft_source_budget_missing_first_task_size() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let unbounded = CompactionOptions {
        target_rows_per_fragment: 250,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &unbounded).await.unwrap();
    assert!(plan.tasks.len() > 1);

    let mut all_tasks = plan
        .tasks
        .into_iter()
        .map(|task| (task, 0))
        .collect::<Vec<_>>();
    all_tasks[0].0.fragments[0].files[0].file_size_bytes = CachedFileSize::unknown();

    let hard = CompactionOptions {
        max_source_bytes: Some(1),
        ..Default::default()
    };
    let err = limit_tasks_to_source_budget(&hard, dataset.schema(), all_tasks.clone()).unwrap_err();
    assert!(err.to_string().contains("has no size recorded"));

    let soft = CompactionOptions {
        max_source_bytes: Some(1),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    };
    let (selected, summary) =
        limit_tasks_to_source_budget(&soft, dataset.schema(), all_tasks).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(summary.selected_tasks, 1);
    assert!(summary.budget_exhausted());
    assert!(!summary.is_blocked());
    assert_eq!(summary.selected_source_bytes, None);
    assert_eq!(summary.next_task_source_bytes, None);
    assert_eq!(summary.limiting_budgets, vec![SourceBudgetLimit::Bytes]);
}

#[tokio::test]
async fn test_soft_source_budget_keeps_prefix_before_unknown_later_task() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let unbounded = CompactionOptions {
        target_rows_per_fragment: 250,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &unbounded).await.unwrap();
    assert!(plan.tasks.len() > 1);

    let mut all_tasks = plan
        .tasks
        .into_iter()
        .map(|task| (task, 0))
        .collect::<Vec<_>>();
    assert!(
        all_tasks[0].0.fragments[0].files[0]
            .file_size_bytes
            .get()
            .is_some()
    );
    all_tasks[1].0.fragments[0].files[0].file_size_bytes = CachedFileSize::unknown();

    let hard = CompactionOptions {
        max_source_bytes: Some(u64::MAX),
        ..Default::default()
    };
    let err = limit_tasks_to_source_budget(&hard, dataset.schema(), all_tasks.clone()).unwrap_err();
    assert!(err.to_string().contains("has no size recorded"));

    let soft = CompactionOptions {
        max_source_bytes: Some(u64::MAX),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    };
    let (selected, summary) =
        limit_tasks_to_source_budget(&soft, dataset.schema(), all_tasks).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(summary.selected_tasks, 1);
    assert!(summary.budget_exhausted());
    assert!(!summary.is_blocked());
    assert!(summary.selected_source_bytes.is_some_and(|bytes| bytes > 0));
    assert_eq!(summary.next_task_source_bytes, None);
    assert_eq!(summary.limiting_budgets, vec![SourceBudgetLimit::Bytes]);
}

#[tokio::test]
async fn test_soft_source_budget_stops_after_oversized_first_task() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let unbounded = CompactionOptions {
        target_rows_per_fragment: 250,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &unbounded).await.unwrap();
    assert!(plan.tasks.len() > 1);

    let mut all_tasks = plan
        .tasks
        .into_iter()
        .map(|task| (task, 0))
        .collect::<Vec<_>>();
    // The first task has known source sizes and exceeds the row budget.
    // The second task's unknown size must not be inspected after soft mode
    // admits the first task, because the plan must stop at that point.
    assert!(
        all_tasks[0].0.fragments[0].files[0]
            .file_size_bytes
            .get()
            .is_some()
    );
    all_tasks[0].1 = 2;
    all_tasks[1].0.fragments[0].files[0].file_size_bytes = CachedFileSize::unknown();

    let soft = CompactionOptions {
        max_source_rows: Some(1),
        max_source_bytes: Some(u64::MAX),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    };
    let (selected, summary) =
        limit_tasks_to_source_budget(&soft, dataset.schema(), all_tasks).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(summary.selected_tasks, 1);
    assert!(summary.budget_exhausted());
    assert!(!summary.is_blocked());
    assert_eq!(summary.selected_source_rows, 2);
    assert!(summary.selected_source_bytes.is_some_and(|bytes| bytes > 0));
    assert_eq!(summary.next_task_source_bytes, None);
    assert_eq!(summary.limiting_budgets, vec![SourceBudgetLimit::Rows]);
}

#[tokio::test]
async fn test_soft_source_budget_uses_cumulative_totals_after_first_task() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let unbounded = CompactionOptions {
        target_rows_per_fragment: 250,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &unbounded).await.unwrap();
    assert!(plan.tasks.len() > 2);

    // Each task's synthetic row count fits the budget independently. The
    // first two fit cumulatively, but adding the third exceeds the total.
    let all_tasks = plan
        .tasks
        .into_iter()
        .take(3)
        .map(|task| (task, 2))
        .collect::<Vec<_>>();
    let hard = CompactionOptions {
        max_source_rows: Some(5),
        ..Default::default()
    };
    let soft = CompactionOptions {
        max_source_rows: Some(5),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    };

    let hard_selected =
        limit_tasks_to_source_budget(&hard, dataset.schema(), all_tasks.clone()).unwrap();
    let soft_selected = limit_tasks_to_source_budget(&soft, dataset.schema(), all_tasks).unwrap();
    assert_eq!(hard_selected.0.len(), 2);
    assert_eq!(hard_selected.1.selected_source_rows, 4);
    assert_eq!(hard_selected.1.next_task_source_rows, Some(2));
    assert_eq!(
        hard_selected.1.limiting_budgets,
        vec![SourceBudgetLimit::Rows]
    );
    assert_eq!(soft_selected, hard_selected);
}

#[tokio::test]
async fn test_max_source_bytes() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;

    let first_task_base_bytes: u64 = dataset.get_fragments()[..3]
        .iter()
        .flat_map(|f| f.metadata.files.iter())
        .map(|df| df.file_size_bytes.get().unwrap().get())
        .sum();
    let overlay_bytes = dataset.get_fragment(0).unwrap().metadata().overlays[0]
        .data_file
        .file_size_bytes
        .get()
        .unwrap()
        .get();
    assert!(first_task_base_bytes > 0 && overlay_bytes > 0);

    // The first task covers fragments 0..=2 (target 250 rows). A budget of
    // exactly their base bytes is exceeded once fragment 0's overlay file
    // is counted, so the plan is empty: the budget is a hard upper bound,
    // overlay bytes are part of a task's source, and deletion files are
    // never counted.
    let opts = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_bytes: Some(first_task_base_bytes),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &opts).await.unwrap();
    assert_eq!(plan.num_tasks(), 0);

    // Widening the budget by the overlay's bytes admits exactly the first
    // task and nothing more.
    let budget = first_task_base_bytes + overlay_bytes;
    let opts = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_bytes: Some(budget),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &opts).await.unwrap();
    assert_eq!(plan.num_tasks(), 1);
    let source_bytes: u64 = plan
        .tasks()
        .iter()
        .flat_map(|t| &t.fragments)
        .flat_map(|f| {
            f.files
                .iter()
                .chain(f.overlays.iter().map(|o| &o.data_file))
        })
        .map(|df| df.file_size_bytes.get().unwrap().get())
        .sum();
    assert_eq!(source_bytes, budget);
}
#[tokio::test]
async fn test_compaction_uses_manifest_config() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = sample_data();
    let schema = data.schema();

    // Create dataset with small fragments
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
    let write_params = WriteParams {
        max_rows_per_file: 2000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    assert_eq!(dataset.get_fragments().len(), 5);

    // Set compaction config in manifest
    dataset
        .update_config([
            ("lance.compaction.target_rows_per_fragment", "5000"),
            ("lance.compaction.materialize_deletions_threshold", "2.0"),
        ])
        .await
        .unwrap();

    // Build options from the dataset config (as the bindings do)
    let opts = CompactionOptions::from_dataset_config(&dataset.manifest.config).unwrap();
    assert_eq!(opts.target_rows_per_fragment, 5000);
    assert!((opts.materialize_deletions_threshold - 2.0).abs() < f32::EPSILON);

    // Verify the config flows through plan_compaction
    let plan = plan_compaction(&dataset, &opts).await.unwrap();
    assert!(!plan.tasks.is_empty());
    assert_eq!(plan.options.target_rows_per_fragment, 5000);
    // validate() should have turned off materialize_deletions since threshold >= 1.0
    assert!(!plan.options.materialize_deletions);
}

#[rstest]
#[case("0.1", Some(LanceFileVersion::Legacy))]
#[case("2.0", Some(LanceFileVersion::V2_0))]
#[case("2.1", Some(LanceFileVersion::V2_1))]
#[case("2.2", Some(LanceFileVersion::V2_2))]
#[case("2.3", Some(LanceFileVersion::V2_3))]
#[case("stable", Some(LanceFileVersion::Stable))]
#[case("next", Some(LanceFileVersion::Next))]
#[case("invalid", None)]
fn compaction_storage_version_config(
    #[case] value: &str,
    #[case] expected: Option<LanceFileVersion>,
) {
    let config = HashMap::from([(
        "lance.compaction.data_storage_version".to_string(),
        value.to_string(),
    )]);
    let result = CompactionOptions::from_dataset_config(&config);
    if let Some(expected) = expected {
        let options = result.unwrap();
        assert_eq!(options.data_storage_version, Some(expected));
        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(json["data_storage_version"], value);
        assert_eq!(
            serde_json::from_value::<CompactionOptions>(json).unwrap(),
            options
        );
    } else {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("lance.compaction.data_storage_version")
        );
        assert!(error.to_string().contains(value));
        let error =
            serde_json::from_value::<LanceFileVersion>(serde_json::json!(value)).unwrap_err();
        assert!(error.to_string().contains(value));
    }
}

#[tokio::test]
async fn compaction_skips_fragments_above_min_bytes() {
    let dir = TempStrDir::default();
    let reader = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(64)),
        )
        .into_reader_rows(RowCount::from(128), BatchCount::from(1));
    let mut dataset = Dataset::write(
        reader,
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 16,
            max_rows_per_group: 1,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    let fields = dataset.schema().field_ids().into_iter().collect();
    let fragment = &dataset.fragments()[0];
    assert!(fragment_file_bytes(fragment, &fields).is_some());
    let mut multi_file = fragment.clone();
    multi_file.files.push(fragment.files[0].clone());
    assert_eq!(fragment_file_bytes(&multi_file, &fields), None);
    let mut obsolete = fragment.clone();
    let mut obsolete_fields = obsolete.files[0].fields.to_vec();
    obsolete_fields.push(i32::MAX);
    obsolete.files[0].fields = obsolete_fields.into();
    assert_eq!(fragment_file_bytes(&obsolete, &fields), None);
    let before = dataset.scan().try_into_batch().await.unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        max_bytes_per_file: Some(16 * 1024),
        // Flush within this small fixture so the byte cap, rather than the
        // final close, splits the output into multiple files below the row target.
        batch_size: Some(16),
        data_cache_bytes: Some(1024),
        ..Default::default()
    };
    let planner = DefaultCompactionPlanner::new(options.clone()).unwrap();
    assert!(planner.plan(&dataset).await.unwrap().num_tasks() > 0);
    compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    assert!(dataset.fragments().len() > 1);
    assert_eq!(
        plan_compaction(&dataset, &options)
            .await
            .unwrap()
            .num_tasks(),
        0
    );
    let settled_version = dataset.version().version;
    compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    assert_eq!(dataset.version().version, settled_version);
    let mut scan = dataset.scan();
    scan.filter("id >= 64").unwrap();
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 64);
    dataset.delete("id % 2 = 0").await.unwrap();
    assert!(planner.plan(&dataset).await.unwrap().num_tasks() > 0);
}

#[tokio::test]
async fn byte_full_fragment_preserves_small_runs_on_both_sides() {
    let dir = TempStrDir::default();
    let mut dataset = None;
    for rows in [4, 4, 64, 4, 4] {
        let reader = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(64)),
            )
            .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
        let params = Some(WriteParams {
            max_rows_per_file: rows as usize,
            ..Default::default()
        });
        if let Some(ds) = &mut dataset {
            Dataset::append(ds, reader, params).await.unwrap();
        } else {
            dataset = Some(Dataset::write(reader, dir.as_str(), params).await.unwrap());
        }
    }
    let dataset = dataset.unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 10_000,
        max_bytes_per_file: Some(8 * 1024),
        ..Default::default()
    };
    let planner = DefaultCompactionPlanner::new(options).unwrap();
    let plan = planner.plan(&dataset).await.unwrap();
    assert_eq!(plan.num_tasks(), 2);
    let middle_id = dataset.fragments()[2].id;
    assert!(
        plan.compaction_tasks()
            .all(|task| task.task.fragments.iter().all(|f| f.id != middle_id))
    );
}

#[tokio::test]
async fn test_source_budget_progress_and_blocked_diagnostics() {
    // Whole tasks fit the run cap and drain over separate atomic commits.
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount(5), FragmentRowCount(50))
        .await
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        max_source_rows: Some(200),
        ..Default::default()
    };
    let mut committed = 0;
    loop {
        let result = DefaultCompactionPlanner::new(options.clone())
            .unwrap()
            .plan_with_budget(&dataset)
            .await
            .unwrap();
        assert!(result.source_budget.selected_source_rows <= 200);
        assert!(!result.source_budget.is_blocked());
        if result.plan.num_tasks() == 0 {
            break;
        }
        assert_eq!(result.plan.options.target_rows_per_fragment, 100);
        let mut results = Vec::new();
        for task in result.plan.compaction_tasks() {
            results.push(task.execute(&dataset).await.unwrap());
        }
        commit_compaction(
            &mut dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        committed += 1;
        assert!(committed <= 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 250);
    }
    assert_eq!(committed, 2);

    // A task larger than the cap is held back whole, even when some of its
    // source fragments fit. Report that boundary rather than a silent no-op.
    let dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount(3), FragmentRowCount(99))
        .await
        .unwrap();
    let blocked = DefaultCompactionPlanner::new(CompactionOptions {
        max_source_rows: Some(200),
        ..options
    })
    .unwrap()
    .plan_with_budget(&dataset)
    .await
    .unwrap();
    assert_eq!(blocked.plan.num_tasks(), 0);
    assert!(blocked.source_budget.is_blocked());
    assert_eq!(
        blocked.source_budget.limiting_budgets,
        vec![SourceBudgetLimit::Rows]
    );
    assert_eq!(blocked.source_budget.next_task_source_rows, Some(297));
    assert_eq!(blocked.source_budget.next_task_source_fragments, Some(3));
}

#[tokio::test]
async fn test_source_budget_reports_remaining_work_and_live_rows() {
    let test_dir = TempStrDir::default();
    let dataset = dataset_with_ten_small_fragments(&test_dir).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 250,
        max_source_rows: Some(270),
        ..Default::default()
    };
    let result = DefaultCompactionPlanner::new(options)
        .unwrap()
        .plan_with_budget(&dataset)
        .await
        .unwrap();
    assert!(result.source_budget.budget_exhausted());
    assert!(!result.source_budget.is_blocked());
    assert_eq!(result.source_budget.selected_tasks, 1);
    assert_eq!(result.source_budget.selected_source_rows, 250);
    assert!(result.source_budget.candidate_tasks > 1);
    assert_eq!(result.source_budget.next_task_source_rows, Some(300));
    assert_eq!(result.source_budget.next_task_source_fragments, Some(3));
}

#[derive(Debug, Clone, Copy)]
enum ByteBudgetCase {
    Unset,
    BelowFirstTask,
    FirstTask,
}

#[derive(Debug, Clone, Copy)]
enum BudgetPlanningPath {
    Ordinary,
    Grouped,
}

#[rstest::rstest]
#[tokio::test]
async fn test_soft_budget_report_admits_first_task_in_both_planning_paths(
    #[values(BudgetPlanningPath::Ordinary, BudgetPlanningPath::Grouped)] path: BudgetPlanningPath,
    #[values(2, 4)] fragments: u32,
) {
    let dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount(fragments), FragmentRowCount(50))
        .await
        .unwrap();
    let planner = DefaultCompactionPlanner::new(CompactionOptions {
        target_rows_per_fragment: 100,
        max_source_rows: Some(99),
        max_source_fragments: Some(1),
        source_budget_mode: SourceBudgetMode::Soft,
        ..Default::default()
    })
    .unwrap();
    let result = match path {
        BudgetPlanningPath::Ordinary => planner.plan_with_budget(&dataset).await,
        BudgetPlanningPath::Grouped => {
            let groups = dataset
                .manifest
                .fragments
                .chunks(2)
                .map(|fragments| CompactionGroup {
                    fragments: fragments.to_vec(),
                    output_clustering_id: None,
                    output_metadata: HashMap::new(),
                })
                .collect();
            planner.plan_groups_with_budget(&dataset, groups).await
        }
    }
    .unwrap();
    let summary = result.source_budget;
    assert_eq!(result.plan.num_tasks(), 1);
    assert_eq!(summary.candidate_tasks, fragments as usize / 2);
    assert_eq!(summary.selected_tasks, 1);
    assert_eq!(summary.selected_source_rows, 100);
    assert_eq!(summary.selected_source_fragments, 2);
    assert_eq!(summary.selected_source_bytes, None);
    assert!(!summary.is_blocked());
    assert_eq!(summary.budget_exhausted(), fragments == 4);
    assert_eq!(
        summary.next_task_source_rows,
        (fragments == 4).then_some(100)
    );
    assert_eq!(
        summary.next_task_source_fragments,
        (fragments == 4).then_some(2)
    );
    assert_eq!(
        summary.limiting_budgets,
        if fragments == 4 {
            vec![SourceBudgetLimit::Fragments, SourceBudgetLimit::Rows]
        } else {
            vec![]
        }
    );
}

#[rstest::rstest]
#[case::rows(Some(99), None, ByteBudgetCase::Unset, vec![SourceBudgetLimit::Rows], 0)]
#[case::fragments(None, Some(1), ByteBudgetCase::Unset, vec![SourceBudgetLimit::Fragments], 0)]
#[case::bytes(None, None, ByteBudgetCase::BelowFirstTask, vec![SourceBudgetLimit::Bytes], 0)]
#[case::all_limits(Some(99), Some(1), ByteBudgetCase::BelowFirstTask,
    vec![SourceBudgetLimit::Fragments, SourceBudgetLimit::Rows, SourceBudgetLimit::Bytes], 0)]
#[case::bytes_after_progress(None, None, ByteBudgetCase::FirstTask, vec![SourceBudgetLimit::Bytes], 1)]
#[case::all_limits_after_progress(Some(100), Some(2), ByteBudgetCase::FirstTask,
    vec![SourceBudgetLimit::Fragments, SourceBudgetLimit::Rows, SourceBudgetLimit::Bytes], 1)]
#[case::unlimited(None, None, ByteBudgetCase::Unset, vec![], 2)]
#[tokio::test]
async fn test_source_budget_reports_limits_bytes_and_warning_context(
    #[case] max_source_rows: Option<usize>,
    #[case] max_source_fragments: Option<usize>,
    #[case] byte_budget: ByteBudgetCase,
    #[case] limiting_budgets: Vec<SourceBudgetLimit>,
    #[case] selected_tasks: usize,
    #[values(BudgetPlanningPath::Ordinary, BudgetPlanningPath::Grouped)] path: BudgetPlanningPath,
) {
    let dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount(4), FragmentRowCount(50))
        .await
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        ..Default::default()
    };
    let unlimited = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .plan(&dataset)
        .await
        .unwrap();
    assert_eq!(unlimited.num_tasks(), 2);
    let task_bytes: Vec<u64> = unlimited
        .tasks()
        .iter()
        .map(|task| {
            task.fragments
                .iter()
                .flat_map(|fragment| &fragment.files)
                .map(|file| file.file_size_bytes.get().unwrap().get())
                .sum()
        })
        .collect();
    assert!(task_bytes[0] > 1);
    let options = CompactionOptions {
        max_source_rows,
        max_source_fragments,
        max_source_bytes: match byte_budget {
            ByteBudgetCase::Unset => None,
            ByteBudgetCase::BelowFirstTask => Some(task_bytes[0] - 1),
            ByteBudgetCase::FirstTask => Some(task_bytes[0]),
        },
        ..options
    };
    let logs = tempfile::NamedTempFile::new().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_max_level(tracing::Level::WARN)
        .with_writer(logs.reopen().unwrap())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let planner = DefaultCompactionPlanner::new(options.clone()).unwrap();
    let result = match path {
        BudgetPlanningPath::Ordinary => planner.plan_with_budget(&dataset).await,
        BudgetPlanningPath::Grouped => {
            let groups = dataset
                .manifest
                .fragments
                .chunks(2)
                .map(|fragments| CompactionGroup {
                    fragments: fragments.to_vec(),
                    output_clustering_id: None,
                    output_metadata: HashMap::new(),
                })
                .collect();
            planner.plan_groups_with_budget(&dataset, groups).await
        }
    }
    .unwrap();
    let summary = &result.source_budget;
    assert_eq!(result.plan.num_tasks(), selected_tasks);
    assert_eq!(summary.candidate_tasks, 2);
    assert_eq!(summary.selected_tasks, selected_tasks);
    assert_eq!(summary.selected_source_rows, selected_tasks * 100);
    assert_eq!(summary.selected_source_fragments, selected_tasks * 2);
    assert_eq!(
        summary.selected_source_bytes,
        options
            .max_source_bytes
            .map(|_| task_bytes[..selected_tasks].iter().sum())
    );
    assert_eq!(summary.limiting_budgets, limiting_budgets);
    assert_eq!(summary.budget_exhausted(), selected_tasks < 2);
    assert_eq!(summary.is_blocked(), selected_tasks == 0);
    assert_eq!(
        summary.next_task_source_rows,
        (selected_tasks < 2).then_some(100)
    );
    assert_eq!(
        summary.next_task_source_fragments,
        (selected_tasks < 2).then_some(2)
    );
    assert_eq!(
        summary.next_task_source_bytes,
        options
            .max_source_bytes
            .and_then(|_| task_bytes.get(selected_tasks).copied())
    );

    let output = std::fs::read_to_string(logs.path()).unwrap();
    if summary.is_blocked() {
        for expected in [
            "candidate_tasks=2".to_string(),
            format!("max_source_rows={max_source_rows:?}"),
            format!("max_source_fragments={max_source_fragments:?}"),
            format!("max_source_bytes={:?}", options.max_source_bytes),
            format!("limiting_budgets={limiting_budgets:?}"),
            "first_task_source_rows=Some(100)".to_string(),
            "first_task_source_fragments=Some(2)".to_string(),
            format!(
                "first_task_source_bytes={:?}",
                summary.next_task_source_bytes
            ),
        ] {
            assert!(output.contains(&expected), "missing {expected}: {output}");
        }
    } else {
        assert!(
            !output.contains("Compaction source budget cannot admit"),
            "{output}"
        );
    }
}

#[tokio::test]
async fn test_grouped_budget_report_preserves_fragment_cutoff_before_byte_validation() {
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount(4), FragmentRowCount(50))
        .await
        .unwrap();
    let mut manifest = dataset.manifest.as_ref().clone();
    let mut fragments = manifest.fragments.as_ref().clone();
    fragments[2].files[0].file_size_bytes = Default::default();
    manifest.fragments = Arc::new(fragments);
    dataset.manifest = Arc::new(manifest);
    let groups: Vec<_> = dataset
        .manifest
        .fragments
        .chunks(2)
        .map(|fragments| CompactionGroup {
            fragments: fragments.to_vec(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        })
        .collect();
    let options = CompactionOptions {
        target_rows_per_fragment: 100,
        max_source_fragments: Some(2),
        max_source_bytes: Some(u64::MAX),
        ..Default::default()
    };
    let result = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .plan_groups_with_budget(&dataset, groups.clone())
        .await
        .unwrap();
    assert_eq!(result.plan.num_tasks(), 1);
    assert_eq!(result.source_budget.candidate_tasks, 2);
    assert_eq!(result.source_budget.selected_tasks, 1);
    assert_eq!(result.source_budget.next_task_source_rows, Some(100));
    assert_eq!(result.source_budget.next_task_source_fragments, Some(2));
    assert_eq!(result.source_budget.next_task_source_bytes, None);
    assert_eq!(
        result.source_budget.limiting_budgets,
        vec![SourceBudgetLimit::Fragments]
    );
    assert!(result.source_budget.budget_exhausted());

    let error = DefaultCompactionPlanner::new(CompactionOptions {
        max_source_fragments: None,
        ..options
    })
    .unwrap()
    .plan_groups_with_budget(&dataset, groups)
    .await
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("no size recorded"), "{error}");
}
