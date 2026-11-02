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
    FlatCombinedFieldsExec::try_new(
        exec.dataset().clone(),
        exec.query().clone(),
        exec.params().clone(),
        decoded_input,
        exec.scanned_fragments().clone(),
        exec.stats_scope().clone(),
        exec.emit_prefilter().clone(),
    )
    .unwrap()
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
        RowAddress::new_from_parts(row / 2, row % 2).into()
    }
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
                    RowAddrTreeMap::from_iter([row_id(1, stable), row_id(2, stable)])
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
        result[ROW_ID].as_primitive::<UInt64Type>().value(0),
        row_id(3, stable)
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
    scanner.project(&["title", "body"]).unwrap().with_row_id();
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
            let address = RowAddress::from(batch[ROW_ID].as_primitive::<UInt64Type>().value(row));
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
#[case::missing_rowid("missing _rowid")]
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
        ArrowField::new(ROW_ID, DataType::UInt64, true),
        ArrowField::new("title", DataType::Utf8, true),
        ArrowField::new("body", DataType::Utf8, true),
    ];
    let mut coverage = vec![FlatStatsCoverage::default(); 2];
    match expected {
        "coverage" => {
            coverage.pop();
        }
        "missing _rowid" => {
            fields.remove(0);
        }
        "must be UInt64" => {
            fields[0] = ArrowField::new(ROW_ID, DataType::Int32, true);
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
