// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::dataset_overlay_index_masking::commit_overlay;
use super::*;
use crate::io::exec::PreFilterSource;
use crate::io::exec::fts::{
    CombinedFieldsQueryExec, FlatCombinedFieldsExec, FlatStatsCoverage, FlatStatsScope,
    SharedFtsScorerExec,
};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use futures::TryStreamExt;
use lance_core::utils::address::RowAddress;
use lance_datafusion::exec::execute_plan;
use lance_index::scalar::inverted::{CombinedFieldsBM25Scorer, query::FtsSearchParams};
use lance_select::RowSetOps;
use lance_table::format::overlay::OverlayCoverage;
use roaring::RoaringBitmap;

fn reconstruct_indexed(exec: &CombinedFieldsQueryExec) -> CombinedFieldsQueryExec {
    let mut rebuilt = CombinedFieldsQueryExec::new(
        exec.dataset().clone(),
        exec.query().clone(),
        exec.params().clone(),
        exec.prefilter_source().clone(),
    )
    .with_external_mask(exec.external_mask().cloned());
    if let Some(fragments) = exec.covered_fragments() {
        rebuilt = rebuilt.with_covered_fragments(fragments.clone());
    }
    if let Some(mask) = exec.overlay_block() {
        rebuilt = rebuilt.with_overlay_block(mask.clone());
    }
    if let Some(scorer) = exec.base_scorer() {
        // Recreate the values a codec can encode, without reusing its Arc.
        rebuilt = rebuilt.with_base_scorer(Arc::new(CombinedFieldsBM25Scorer::new(
            scorer.doc_count(),
            scorer.avg_doc_length(),
            scorer.doc_freq().clone(),
        )));
    }
    rebuilt
}

fn reconstruct_flat(
    exec: &FlatCombinedFieldsExec,
    decoded_input: Arc<dyn ExecutionPlan>,
) -> FlatCombinedFieldsExec {
    let mut rebuilt = FlatCombinedFieldsExec::try_new(
        exec.dataset().clone(),
        exec.query().clone(),
        exec.params().clone(),
        decoded_input,
        exec.scanned_fragments().clone(),
        exec.stats_scope().clone(),
        exec.emit_prefilter().clone(),
    )
    .unwrap();
    if let Some(scorer) = exec.base_scorer() {
        rebuilt = rebuilt.with_base_scorer(Arc::new(CombinedFieldsBM25Scorer::new(
            scorer.doc_count(),
            scorer.avg_doc_length(),
            scorer.doc_freq().clone(),
        )));
    }
    rebuilt
}

/// Reconstruct leaf configuration through its public API, then let each native
/// owner reconnect its producer and consumer using the exported binding.
pub(super) fn reconstruct_plan(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let mut leaves = 0;
    let mut owners = 0;
    let rebuilt = plan
        .transform_down_up(
            |plan| {
                if let Some(owner) = plan.downcast_ref::<SharedFtsScorerExec>() {
                    owners += 1;
                    let binding = owner.binding().unwrap();
                    let input = reconstruct_plan(owner.children()[0].clone());
                    let rebuilt = SharedFtsScorerExec::try_new(input, binding).unwrap();
                    return Ok(Transformed::new(
                        Arc::new(rebuilt) as Arc<dyn ExecutionPlan>,
                        true,
                        TreeNodeRecursion::Jump,
                    ));
                }
                Ok(Transformed::no(plan))
            },
            |plan| {
                if let Some(exec) = plan.downcast_ref::<CombinedFieldsQueryExec>() {
                    leaves += 1;
                    return Ok(Transformed::yes(
                        Arc::new(reconstruct_indexed(exec)) as Arc<dyn ExecutionPlan>
                    ));
                }
                if let Some(exec) = plan.downcast_ref::<FlatCombinedFieldsExec>() {
                    leaves += 1;
                    return Ok(Transformed::yes(Arc::new(reconstruct_flat(
                        exec,
                        exec.children()[0].clone(),
                    ))
                        as Arc<dyn ExecutionPlan>));
                }
                Ok(Transformed::no(plan))
            },
        )
        .unwrap()
        .data;
    assert!(leaves > 0 || owners > 0);
    rebuilt
}

async fn batches(plan: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    execute_plan(plan, Default::default())
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

pub(super) async fn id_scores(plan: Arc<dyn ExecutionPlan>) -> Vec<(i32, f32)> {
    batches(plan)
        .await
        .iter()
        .flat_map(|batch| {
            let ids = batch["id"].as_primitive::<Int32Type>();
            let scores = batch["_score"].as_primitive::<Float32Type>();
            (0..batch.num_rows())
                .map(|row| (ids.value(row), scores.value(row)))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn append(dataset: &mut Dataset, batch: RecordBatch) {
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
}

fn row_id(row: u32, stable: bool) -> u64 {
    if stable {
        u64::from(row)
    } else {
        row_addr(row)
    }
}

/// The fixture's physical address for `row`, which is what an FTS node emits
/// whether or not the dataset has stable row ids (two rows per fragment).
fn row_addr(row: u32) -> u64 {
    RowAddress::new_from_parts(row / 2, row % 2).into()
}

#[rstest]
#[tokio::test]
async fn bm25f_plan_roundtrip_partial_coverage(
    #[values(false, true)] stable: bool,
    #[values(false, true)] overlay: bool,
    #[values(false, true)] scalar_prefilter: bool,
) {
    let mut titles = vec!["aa bb", "cc", "aa aa", "dd", "aa", "zz"];
    let bodies = vec!["cc dd", "aa bb", "dd", "aa bb bb", "bb", "aa"];
    let batch = |rows: std::ops::Range<usize>| {
        combined_fields_batch(
            rows.clone().map(|row| row as i32).collect(),
            titles[rows.clone()].to_vec(),
            bodies[rows].to_vec(),
        )
    };
    let mut dataset = write_fts_dataset(
        "memory://",
        batch(0..2),
        Some(WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        }),
    )
    .await;
    let params = combined_fields_test_params();
    create_inverted_indices(&mut dataset, &["title", "body"], &params).await;
    append(&mut dataset, batch(2..4)).await;
    create_inverted_indices(&mut dataset, &["title"], &params).await;
    append(&mut dataset, batch(4..6)).await;
    if scalar_prefilter {
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }
    if overlay {
        for (fragment, offset, text) in [(0, 1, "aa aa zz"), (1, 0, "bb zz")] {
            dataset = commit_overlay(
                dataset,
                &format!("roundtrip_{fragment}"),
                fragment,
                &[1],
                OverlayCoverage::dense(RoaringBitmap::from_iter([offset])),
                vec![Arc::new(StringArray::from(vec![text]))],
            )
            .await;
        }
        titles[1] = "aa aa zz";
        titles[2] = "bb zz";
    }
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .prefilter(true)
        .filter("id != 0 AND id != 4")
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query_with_boosts(
            "aa bb",
            Operator::Or,
            Some(vec![3.0, 1.0]),
        )))
        .unwrap();
    let original = scanner.create_plan().await.unwrap();
    let mut seen_flat = false;
    original
        .apply(|plan| {
            if let Some(flat) = plan.downcast_ref::<FlatCombinedFieldsExec>() {
                seen_flat = true;
                assert_eq!(
                    flat.scanned_fragments(),
                    &RoaringBitmap::from_iter(if overlay { vec![0, 1, 2] } else { vec![1, 2] })
                );
                let FlatStatsScope::PerColumn(coverage) = flat.stats_scope() else {
                    panic!("per-column statistics expected")
                };
                assert_eq!(coverage[0].fragments, RoaringBitmap::from_iter([2]));
                assert_eq!(coverage[1].fragments, RoaringBitmap::from_iter([1, 2]));
                let expected_stale = if overlay {
                    RowAddrTreeMap::from_iter([row_addr(1), row_addr(2)])
                } else {
                    RowAddrTreeMap::new()
                };
                assert_eq!(coverage[0].stale_rows.as_ref(), &expected_stale);
                assert!(coverage[1].stale_rows.is_empty());
                assert!(flat.has_shared_scorer());
                assert!(!matches!(flat.emit_prefilter(), PreFilterSource::None));
            }
            if let Some(indexed) = plan.downcast_ref::<CombinedFieldsQueryExec>() {
                assert_eq!(
                    indexed.covered_fragments(),
                    Some(&RoaringBitmap::from_iter([0]))
                );
                assert_eq!(indexed.overlay_block().is_some(), overlay);
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
    assert!(seen_flat);
    let expected = id_scores(original).await;
    // Use a separate native plan because overlay prefilters can be one-shot.
    let actual = id_scores(reconstruct_plan(scanner.create_plan().await.unwrap())).await;
    assert_eq!(actual, expected);
    let mut oracle = brute_force_bm25f(&[(3.0, titles), (1.0, bodies)], "aa bb", false);
    oracle[0] = None;
    oracle[4] = None;
    assert_matches_brute_force(
        &actual,
        &oracle,
        "reconstructed partial coverage and overlays",
    );
}

#[rstest]
#[tokio::test]
async fn bm25f_plan_roundtrip_indexed_overrides(#[values(false, true)] stable: bool) {
    let mut dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch((0..6).collect(), vec!["aa"; 6], vec!["aa bb"; 6]),
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: stable,
            ..Default::default()
        }),
    )
    .await;
    create_inverted_indices(
        &mut dataset,
        &["title", "body"],
        &combined_fields_test_params(),
    )
    .await;
    let mask =
        |rows: &[u32]| RowAddrTreeMap::from_iter(rows.iter().map(|&row| row_id(row, stable)));
    let candidates = record_batch!((
        ROW_ID,
        UInt64,
        (0..5).map(|row| row_id(row, stable)).collect::<Vec<_>>()
    ))
    .unwrap();
    let source =
        MemorySourceConfig::try_new_exec(&[vec![candidates.clone()]], candidates.schema(), None)
            .unwrap();
    let exec = CombinedFieldsQueryExec::new(
        Arc::new(dataset),
        CombinedFieldsQuery::try_new("aa bb".into(), vec!["title".into(), "body".into()]).unwrap(),
        FtsSearchParams::default().with_limit(Some(2)),
        PreFilterSource::FilteredRowIds(source),
    )
    .with_covered_fragments(RoaringBitmap::from_iter([1, 2]))
    .with_overlay_block(RowAddrMask::from_block(mask(&[2])))
    .with_external_mask(Some(Arc::new(RowAddrMask::from_allowed(mask(&[
        0, 1, 2, 3, 5,
    ])))))
    .with_base_scorer(Arc::new(CombinedFieldsBM25Scorer::new(
        100,
        12.0,
        HashMap::from([("aa".into(), 2), ("bb".into(), 1)]),
    )));
    let rebuilt = reconstruct_indexed(&exec);
    let original = batches(Arc::new(exec)).await;
    let actual = batches(Arc::new(rebuilt)).await;
    assert_eq!(actual, original);
    assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let result = actual.iter().find(|batch| batch.num_rows() > 0).unwrap();
    assert_eq!(
        result[ROW_ADDR].as_primitive::<UInt64Type>().value(0),
        row_addr(3)
    );
    let override_scorer = CombinedFieldsBM25Scorer::new(
        100,
        12.0,
        HashMap::from([("aa".into(), 2), ("bb".into(), 1)]),
    );
    let expected_score = override_scorer.query_weight("aa") * override_scorer.doc_weight(2.0, 3.0)
        + override_scorer.query_weight("bb") * override_scorer.doc_weight(1.0, 3.0);
    assert!(
        (result["_score"].as_primitive::<Float32Type>().value(0) - expected_score).abs() < 1e-5
    );
}

#[tokio::test]
async fn bm25f_plan_roundtrip_whole_corpus() {
    let mut dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch(vec![0, 1], vec!["aa", "zz"], vec!["bb", "aa aa"]),
        None,
    )
    .await;
    create_inverted_indices(&mut dataset, &["title"], &combined_fields_test_params()).await;
    append(
        &mut dataset,
        combined_fields_batch(vec![2, 3], vec!["aa", "bb"], vec!["aa", "aa bb"]),
    )
    .await;
    let mut scanner = dataset.scan();
    scanner
        .project(&["title", "body"])
        .unwrap()
        .with_row_address();
    let flat = FlatCombinedFieldsExec::try_new(
        Arc::new(dataset),
        CombinedFieldsQuery::try_new("aa".into(), vec!["title".into(), "body".into()]).unwrap(),
        FtsSearchParams::default(),
        scanner.create_plan().await.unwrap(),
        RoaringBitmap::from_iter([0, 1]),
        FlatStatsScope::WholeCorpus,
        PreFilterSource::None,
    )
    .unwrap();
    let rebuilt = reconstruct_flat(&flat, scanner.create_plan().await.unwrap());
    let actual = batches(Arc::new(rebuilt)).await;
    assert_eq!(actual, batches(Arc::new(flat)).await);
    let oracle = brute_force_bm25f(
        &[
            (1.0, vec!["aa", "zz", "aa", "bb"]),
            (1.0, vec!["bb", "aa aa", "aa", "aa bb"]),
        ],
        "aa",
        false,
    );
    assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    for batch in actual {
        for row in 0..batch.num_rows() {
            let address = RowAddress::from(batch[ROW_ADDR].as_primitive::<UInt64Type>().value(row));
            let id = (address.fragment_id() * 2 + address.row_offset()) as usize;
            assert!(
                (batch["_score"].as_primitive::<Float32Type>().value(row) - oracle[id].unwrap())
                    .abs()
                    < 1e-3
            );
        }
    }
}

#[tokio::test]
async fn bm25f_plan_roundtrip_list_documents() {
    let mut dataset = write_fts_dataset(
        "memory://",
        list_column_batch::<i32, i32>(&[0, 1], &[Some(vec!["aa", "bb"]), None], &["zz", "aa"]),
        None,
    )
    .await;
    create_inverted_indices(
        &mut dataset,
        &["tags", "body"],
        &combined_fields_test_params(),
    )
    .await;
    append(
        &mut dataset,
        list_column_batch::<i32, i32>(
            &[2, 3, 4],
            &[Some(vec!["aa", "bb aa"]), Some(vec![]), None],
            &["bb", "aa aa", "aa"],
        ),
    )
    .await;
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(combined_query_over(
            &["tags", "body"],
            "aa bb",
            Operator::Or,
            Some(vec![2.0, 1.0]),
        )))
        .unwrap();
    let expected = id_scores(scanner.create_plan().await.unwrap()).await;
    let actual = id_scores(reconstruct_plan(scanner.create_plan().await.unwrap())).await;
    assert_eq!(actual, expected);
    let oracle = brute_force_bm25f(
        &[
            (2.0, vec!["aa bb", "", "aa bb aa", "", ""]),
            (1.0, vec!["zz", "aa", "bb", "aa aa", "aa"]),
        ],
        "aa bb",
        false,
    );
    assert_matches_brute_force(
        &actual,
        &oracle,
        "list traversal reconstructed from the pinned schema",
    );
}

#[rstest]
#[case::coverage("coverage")]
#[case::missing_row_addr("missing _rowaddr")]
#[case::rowid_type("must be UInt64")]
#[case::missing_column("missing column 'body'")]
#[case::column_type("incompatible type")]
#[tokio::test]
async fn bm25f_plan_rejects_invalid_input(#[case] expected: &str) {
    let dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch(vec![0], vec!["aa"], vec!["bb"]),
        None,
    )
    .await;
    let mut fields = vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("title", DataType::Utf8, true),
        ArrowField::new("body", DataType::Utf8, true),
    ];
    let mut coverage = vec![FlatStatsCoverage::default(); 2];
    match expected {
        "coverage" => {
            coverage.pop();
        }
        "missing _rowaddr" => {
            fields.remove(0);
        }
        "must be UInt64" => {
            fields[0] = ArrowField::new(ROW_ADDR, DataType::Int32, true);
        }
        "missing column 'body'" => {
            fields.pop();
        }
        "incompatible type" => {
            fields[2] = ArrowField::new("body", DataType::Int32, true);
        }
        _ => unreachable!(),
    }
    let error = FlatCombinedFieldsExec::try_new(
        Arc::new(dataset),
        CombinedFieldsQuery::try_new("aa".into(), vec!["title".into(), "body".into()]).unwrap(),
        FtsSearchParams::default(),
        Arc::new(EmptyExec::new(Arc::new(ArrowSchema::new(fields)))),
        RoaringBitmap::from_iter([0]),
        FlatStatsScope::PerColumn(coverage),
        PreFilterSource::None,
    )
    .unwrap_err();
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}

async fn row_scores(plan: Arc<dyn ExecutionPlan>) -> Vec<(u64, f32)> {
    batches(plan)
        .await
        .iter()
        .flat_map(|batch| {
            let row_ids = batch[ROW_ADDR].as_primitive::<UInt64Type>();
            let scores = batch["_score"].as_primitive::<Float32Type>();
            (0..batch.num_rows())
                .map(|row| (row_ids.value(row), scores.value(row)))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[rstest]
#[tokio::test]
async fn bm25f_scoped_prescored_opens_only_overlapping_segments(
    #[values(false, true)] stable: bool,
) {
    use crate::index::loading::DatasetIndexInternalExt;
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::inverted::{InvertedIndex, build_combined_bm25_scorer_from_totals};

    let mut dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch(
            vec![0, 1],
            vec!["alpha beta", "gamma"],
            vec!["delta alpha", "beta delta"],
        ),
        Some(WriteParams {
            enable_stable_row_ids: stable,
            ..Default::default()
        }),
    )
    .await;
    let idx_params = combined_fields_test_params();
    create_inverted_indices(&mut dataset, &["title", "body"], &idx_params).await;

    append(
        &mut dataset,
        combined_fields_batch(
            vec![2, 3],
            vec!["beta", "alpha alpha"],
            vec!["alpha gamma", "delta"],
        ),
    )
    .await;
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let dataset = Arc::new(dataset);
    let query = CombinedFieldsQuery::try_new(
        "alpha beta".to_string(),
        vec!["title".to_string(), "body".to_string()],
    )
    .unwrap()
    .try_with_boosts(vec![2.0, 1.0])
    .unwrap();
    let params = FtsSearchParams::default().with_limit(Some(10));

    // Reference full-corpus single-node scores.
    let reference = Arc::new(CombinedFieldsQueryExec::new(
        dataset.clone(),
        query.clone(),
        params.clone(),
        PreFilterSource::None,
    ));
    let expected = row_scores(reference).await;

    // Collect exact global stats from each segment independently (simulating Phase 1).
    let terms = vec!["alpha".to_string(), "beta".to_string()];
    let mut doc_counts = vec![0usize; 2];
    let mut total_tokens = vec![0u64; 2];
    let mut doc_freqs = vec![vec![0usize; 2]; 2];
    for (slot, col) in ["title", "body"].iter().enumerate() {
        let segs = crate::index::scalar::inverted::load_segments(
            &dataset,
            col,
            lance_index::scalar::inverted::DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(segs.len(), 2, "expected two segments per column");
        for seg in segs {
            let idx = dataset
                .open_generic_index(col, &seg.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let inv = idx.as_any().downcast_ref::<InvertedIndex>().unwrap();
            let (toks, docs, dfs) = inv
                .bm25_row_stats_for_terms(&terms, None, None)
                .await
                .unwrap();
            total_tokens[slot] += toks;
            doc_counts[slot] += docs;
            for (t, df) in dfs.into_iter().enumerate() {
                doc_freqs[slot][t] += df;
            }
        }
    }
    let global_scorer = Arc::new(
        build_combined_bm25_scorer_from_totals(
            &[2.0, 1.0],
            &doc_counts,
            &total_tokens,
            &terms,
            &doc_freqs,
        )
        .unwrap(),
    );

    // Execute two disjoint fragment shards with base_scorer. Each shard must open
    // only the overlapping segment per column (2 partitions total, not 4).
    let mut merged = Vec::new();
    for frag_id in [0u32, 1u32] {
        let exec = Arc::new(
            CombinedFieldsQueryExec::new(
                dataset.clone(),
                query.clone(),
                params.clone(),
                PreFilterSource::None,
            )
            .with_covered_fragments(RoaringBitmap::from_iter([frag_id]))
            .with_base_scorer(global_scorer.clone()),
        );
        let shard_rows = row_scores(exec.clone()).await;
        let parts = exec
            .metrics()
            .unwrap()
            .sum_by_name(PARTITIONS_SEARCHED_METRIC)
            .map(|m| m.as_usize())
            .unwrap_or(0);
        assert_eq!(
            parts, 2,
            "shard {frag_id} must open only its 1 overlapping segment per column"
        );
        merged.extend(shard_rows);
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(
        merged, expected,
        "sharded pre-scored results must be bit-identical"
    );

    // Empty covered_fragments with base_scorer opens zero segments and returns empty.
    let empty_exec = Arc::new(
        CombinedFieldsQueryExec::new(dataset.clone(), query, params, PreFilterSource::None)
            .with_covered_fragments(RoaringBitmap::new())
            .with_base_scorer(global_scorer),
    );
    assert!(row_scores(empty_exec.clone()).await.is_empty());
    let empty_parts = empty_exec
        .metrics()
        .unwrap()
        .sum_by_name(PARTITIONS_SEARCHED_METRIC)
        .map(|m| m.as_usize())
        .unwrap_or(0);
    assert_eq!(empty_parts, 0);
}

#[tokio::test]
async fn bm25f_scoped_prescored_still_rejects_omitted_mismatched_tokenizer() {
    let mut dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch(vec![0], vec!["alpha beta"], vec!["alpha"]),
        None,
    )
    .await;
    dataset
        .create_index(
            &["title"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().stem(true),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["body"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().stem(false),
            true,
        )
        .await
        .unwrap();
    let dataset = Arc::new(dataset);
    let query = CombinedFieldsQuery::try_new(
        "alpha".to_string(),
        vec!["title".to_string(), "body".to_string()],
    )
    .unwrap();
    let scorer = Arc::new(CombinedFieldsBM25Scorer::new(1, 2.0, Default::default()));
    let exec = Arc::new(
        CombinedFieldsQueryExec::new(
            dataset,
            query,
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .with_covered_fragments(RoaringBitmap::new())
        .with_base_scorer(scorer),
    );
    let err = execute_plan(exec, Default::default())
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("same tokenizer"),
        "expected tokenizer mismatch error even when segments are omitted, got: {err}"
    );
}

#[tokio::test]
async fn bm25f_flat_prescored_streams_without_opening_index_segments() {
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_index::scalar::inverted::{
        build_combined_bm25_scorer_from_totals, collect_flat_combined_field_stats,
    };

    let mut dataset = write_fts_dataset(
        "memory://",
        combined_fields_batch(
            vec![0, 1],
            vec!["alpha beta", "gamma"],
            vec!["delta", "beta"],
        ),
        None,
    )
    .await;
    let idx_params = combined_fields_test_params();
    create_inverted_indices(&mut dataset, &["title", "body"], &idx_params).await;

    // Append an unindexed fragment (fragment 1).
    append(
        &mut dataset,
        combined_fields_batch(
            vec![2, 3],
            vec!["alpha", "beta alpha"],
            vec!["gamma", "alpha"],
        ),
    )
    .await;
    let dataset = Arc::new(dataset);
    let query = CombinedFieldsQuery::try_new(
        "alpha beta".to_string(),
        vec!["title".to_string(), "body".to_string()],
    )
    .unwrap()
    .try_with_boosts(vec![2.0, 1.0])
    .unwrap();

    // Reference full-corpus single-node mixed query.
    let make_plan = || async {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(FtsQuery::CombinedFields(
                query.clone(),
            )))
            .unwrap();
        scanner.create_plan().await.unwrap()
    };
    let expected = id_scores(make_plan().await).await;

    // Extract fresh FlatCombinedFieldsExec instances from unexecuted plans because
    // FilteredReadExec's underlying stream state is single-use once executed.
    let extract_flat = |plan: &Arc<dyn ExecutionPlan>| {
        let mut flat_node: Option<FlatCombinedFieldsExec> = None;
        plan.apply(|node| {
            if let Some(flat) = node.downcast_ref::<FlatCombinedFieldsExec>() {
                flat_node = Some(flat.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        flat_node.expect("mixed plan must contain FlatCombinedFieldsExec")
    };
    let flat_for_stats = extract_flat(&make_plan().await);
    let flat_for_score = extract_flat(&make_plan().await);

    // Collect global stats: indexed segment 0 + flat fragment 1.
    let terms = vec!["alpha".to_string(), "beta".to_string()];
    let mut doc_counts = vec![0usize; 2];
    let mut total_tokens = vec![0u64; 2];
    let mut doc_freqs = vec![vec![0usize; 2]; 2];
    for (slot, col) in ["title", "body"].iter().enumerate() {
        let segs = crate::index::scalar::inverted::load_segments(
            &dataset,
            col,
            lance_index::scalar::inverted::DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .unwrap();
        for seg in segs {
            let idx = crate::index::loading::DatasetIndexInternalExt::open_generic_index(
                dataset.as_ref(),
                col,
                &seg.uuid,
                &lance_index::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
            let inv = idx
                .as_any()
                .downcast_ref::<lance_index::scalar::inverted::InvertedIndex>()
                .unwrap();
            let (toks, docs, dfs) = inv
                .bm25_row_stats_for_terms(&terms, None, None)
                .await
                .unwrap();
            total_tokens[slot] += toks;
            doc_counts[slot] += docs;
            for (t, df) in dfs.into_iter().enumerate() {
                doc_freqs[slot][t] += df;
            }
        }
    }
    let flat_input = flat_for_stats.children()[0]
        .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
        .unwrap();
    let flat_schema = flat_input.schema();
    let col_indices = vec![
        flat_schema.index_of("title").unwrap(),
        flat_schema.index_of("body").unwrap(),
    ];
    let flat_stats = collect_flat_combined_field_stats(
        flat_input,
        col_indices,
        &[
            Arc::new(RowAddrMask::all_rows()),
            Arc::new(RowAddrMask::all_rows()),
        ],
        &lance_index::scalar::inverted::query::Tokens::new(
            terms.clone(),
            lance_index::scalar::inverted::tokenizer::document_tokenizer::DocType::Text,
        ),
        idx_params.build().unwrap(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(flat_stats.doc_counts, vec![2, 2]);
    for ((dc, tt), (dfs, (flat_dc, (flat_tt, flat_dfs)))) in
        doc_counts.iter_mut().zip(&mut total_tokens).zip(
            doc_freqs.iter_mut().zip(
                flat_stats
                    .doc_counts
                    .iter()
                    .zip(flat_stats.total_tokens.iter().zip(&flat_stats.doc_freqs)),
            ),
        )
    {
        *dc += *flat_dc;
        *tt += *flat_tt;
        for (df, flat_df) in dfs.iter_mut().zip(flat_dfs) {
            *df += *flat_df;
        }
    }
    let global_scorer = Arc::new(
        build_combined_bm25_scorer_from_totals(
            &[2.0, 1.0],
            &doc_counts,
            &total_tokens,
            &terms,
            &doc_freqs,
        )
        .unwrap(),
    );

    let prescored_flat = Arc::new(
        reconstruct_flat(&flat_for_score, flat_for_score.children()[0].clone())
            .with_base_scorer(global_scorer.clone()),
    );
    let flat_rows = row_scores(prescored_flat.clone()).await;
    assert_eq!(flat_rows.len(), 2, "expected both unindexed rows to match");
    let parts = prescored_flat
        .metrics()
        .unwrap()
        .sum_by_name(PARTITIONS_SEARCHED_METRIC)
        .map(|m| m.as_usize())
        .unwrap_or(0);
    assert_eq!(
        parts, 0,
        "pre-scored flat scan must not open any index segment"
    );

    let indexed_shard = Arc::new(
        CombinedFieldsQueryExec::new(
            dataset.clone(),
            query,
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .with_covered_fragments(RoaringBitmap::from_iter([0]))
        .with_base_scorer(global_scorer),
    );
    let indexed_rows = row_scores(indexed_shard).await;

    let mut combined_rows: Vec<(i32, f32)> = indexed_rows
        .into_iter()
        .chain(flat_rows)
        .map(|(row_id, score)| {
            let id = ((row_id >> 32) as i32) * 2 + (row_id & 0xffff_ffff) as i32;
            (id, score)
        })
        .collect();
    combined_rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(combined_rows, expected);
}
