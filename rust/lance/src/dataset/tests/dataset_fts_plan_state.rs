// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod compound;

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, record_batch,
};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance_core::ROW_ID;
use lance_core::utils::address::RowAddress;
use lance_datafusion::exec::execute_plan;
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::document_tokenizer::DocType;
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsSearchParams, MatchQuery, Operator, PhraseQuery, Tokens,
};
use lance_index::scalar::inverted::{
    DOC_INDEX_COL, DocumentGranularity, InvertedIndex, InvertedIndexParams, MemBM25Scorer,
    PreparedBm25Query, prepare_bm25_query,
};
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::OverlayCoverage;
use roaring::RoaringBitmap;
use rstest::rstest;

use super::dataset_overlay_index_masking::commit_overlay;
use crate::Dataset;
use crate::dataset::{WriteMode, WriteParams};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::io::exec::PreFilterSource;
use crate::io::exec::fts::{
    FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec, SharedFtsScorerExec,
};

#[derive(Debug, Clone, Copy)]
enum Kind {
    Match,
    Phrase,
}

#[derive(Debug, Clone, Copy)]
enum Selection {
    All,
    Resolved,
    Uuids,
}

fn restore_prepared(prepared: &PreparedBm25Query) -> Arc<PreparedBm25Query> {
    let tokens = prepared.tokens();
    let decoded = Tokens::try_from_parts(
        (0..tokens.len())
            .map(|i| tokens.get_token(i).to_owned())
            .collect(),
        (0..tokens.len()).map(|i| tokens.position(i)).collect(),
        (0..tokens.len()).map(|i| tokens.term_group(i)).collect(),
        tokens.token_type().clone(),
    )
    .unwrap();
    let scorer = prepared.scorer();
    Arc::new(PreparedBm25Query::from_parts(
        Arc::new(decoded),
        Arc::new(MemBM25Scorer::new(
            scorer.total_tokens,
            scorer.num_docs,
            scorer.token_docs.clone(),
        )),
        prepared.has_all_query_positions(),
    ))
}

// Both leaf types share this public constructor/mask contract. Prepared Match
// state is restored separately, after the scorer-only override.
macro_rules! restore_leaf {
    ($exec:expr, $ty:ty) => {{
        let exec = $exec;
        let mut rebuilt = if let Some(segments) = exec.preset_segments() {
            <$ty>::new_with_segments_and_document_granularity(
                exec.dataset().clone(),
                exec.query().clone(),
                exec.params().clone(),
                exec.prefilter_source().clone(),
                segments.to_vec(),
                exec.document_granularity(),
            )
        } else if let Some(uuids) = exec.explicit_segment_uuids() {
            <$ty>::new_with_segment_uuids_and_document_granularity(
                exec.dataset().clone(),
                exec.query().clone(),
                exec.params().clone(),
                exec.prefilter_source().clone(),
                uuids,
                exec.document_granularity(),
            )
        } else {
            <$ty>::new_with_document_granularity(
                exec.dataset().clone(),
                exec.query().clone(),
                exec.params().clone(),
                exec.prefilter_source().clone(),
                exec.document_granularity(),
            )
        }
        .with_external_mask(exec.external_mask().cloned());
        if let Some(mask) = exec.overlay_block() {
            rebuilt = rebuilt.with_overlay_block(mask.clone());
        }
        if let Some(scorer) = exec.base_scorer() {
            rebuilt = rebuilt.with_base_scorer(Arc::new(MemBM25Scorer::new(
                scorer.total_tokens,
                scorer.num_docs,
                scorer.token_docs.clone(),
            )));
        }
        rebuilt
    }};
}

fn restore_match(exec: &MatchQueryExec) -> MatchQueryExec {
    let mut rebuilt = restore_leaf!(exec, MatchQueryExec);
    if let Some(prepared) = exec.prepared_query() {
        rebuilt = rebuilt.with_prepared_query(restore_prepared(prepared));
    }
    rebuilt
}

fn restore_plan(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    plan.transform_down_up(
        |plan| {
            if let Some(owner) = plan.downcast_ref::<SharedFtsScorerExec>() {
                let binding = owner.binding().unwrap();
                let input = restore_plan(owner.children()[0].clone());
                return Ok(Transformed::new(
                    Arc::new(SharedFtsScorerExec::try_new(input, binding).unwrap())
                        as Arc<dyn ExecutionPlan>,
                    true,
                    TreeNodeRecursion::Jump,
                ));
            }
            Ok(Transformed::no(plan))
        },
        |plan| {
            if let Some(exec) = plan.downcast_ref::<MatchQueryExec>() {
                return Ok(Transformed::yes(
                    Arc::new(restore_match(exec)) as Arc<dyn ExecutionPlan>
                ));
            }
            if let Some(exec) = plan.downcast_ref::<PhraseQueryExec>() {
                return Ok(Transformed::yes(
                    Arc::new(restore_leaf!(exec, PhraseQueryExec)) as Arc<dyn ExecutionPlan>,
                ));
            }
            Ok(Transformed::no(plan))
        },
    )
    .unwrap()
    .data
}

async fn results(plan: Arc<dyn ExecutionPlan>) -> Vec<(u64, Vec<u32>, f32)> {
    let batches: Vec<RecordBatch> = execute_plan(plan, Default::default())
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut documents: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            let ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            let scores = batch["_score"].as_primitive::<Float32Type>();
            (0..batch.num_rows())
                .map(|row| {
                    let coordinates = batch
                        .column_by_name(DOC_INDEX_COL)
                        .map(|column| {
                            column
                                .as_list::<i32>()
                                .value(row)
                                .as_primitive::<UInt32Type>()
                                .values()
                                .to_vec()
                        })
                        .unwrap_or_default();
                    (ids.value(row), coordinates, scores.value(row))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // Leaf plans do not promise tie order. Keep duplicates and compare every
    // document's exact score rather than the completion order of segment reads.
    documents.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    documents
}

fn row_id(row: u32, is_stable: bool) -> u64 {
    if is_stable {
        u64::from(row)
    } else {
        RowAddress::new_from_parts(row / 2, row % 2).into()
    }
}

async fn write(batch: RecordBatch, is_stable: bool) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: is_stable,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn build_segments(
    dataset: &mut Dataset,
    granularity: DocumentGranularity,
) -> Vec<IndexMetadata> {
    let params = InvertedIndexParams::default()
        .with_position(true)
        .stem(false)
        .remove_stop_words(false)
        .document_granularity(granularity);
    let fragments = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        segments.push(
            dataset
                .create_index_builder(&["text"], IndexType::Inverted, &params)
                .name("text_idx".to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("text_idx", "text", segments.clone())
        .await
        .unwrap();
    segments
}

#[rstest]
#[tokio::test]
async fn list_fts_corpus_owner(
    #[values(Kind::Match, Kind::Phrase)] kind: Kind,
    #[values(false, true)] is_stable: bool,
    #[values(false, true)] filtered: bool,
) {
    let batch = |start| {
        let mut text = ListBuilder::new(StringBuilder::new());
        for row in [
            Some(vec![Some("alpha beta"), Some("alpha")]),
            Some(vec![None, Some("alpha beta")]),
            Some(vec![Some("alpha x beta"), Some("other")]),
            Some(vec![Some("alpha"), Some("beta")]),
            Some(vec![]),
            None,
        ] {
            if let Some(values) = row {
                for value in values {
                    text.values().append_option(value);
                }
                text.append(true);
            } else {
                text.append(false);
            }
        }
        RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(start..start + 6)) as ArrayRef,
            ),
            ("text", Arc::new(text.finish()) as ArrayRef),
        ])
        .unwrap()
    };
    let mut dataset = write(batch(0), is_stable).await;
    let granularity = DocumentGranularity::ListElement;
    build_segments(&mut dataset, granularity).await;
    let residual = batch(6);
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(residual.clone())], residual.schema()),
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let query = match kind {
        Kind::Match => FtsQuery::Match(
            MatchQuery::new("alpha beta".into())
                .with_column(Some("text".into()))
                .with_operator(Operator::And)
                .with_document_granularity(granularity),
        ),
        Kind::Phrase => FtsQuery::Phrase(
            PhraseQuery::new("alpha beta".into())
                .with_column(Some("text".into()))
                .with_document_granularity(granularity),
        ),
    };
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .full_text_search(FullTextSearchQuery::new_query(query))
        .unwrap();
    if filtered {
        scanner.filter("id % 3 != 1").unwrap().prefilter(true);
    }
    let plan = scanner.create_plan().await.unwrap();
    let mut owners = Vec::new();
    plan.apply(|node| {
        if node.is::<SharedFtsScorerExec>() {
            owners.push(node.clone());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert_eq!(
        owners.len(),
        1,
        "mixed list plan needs an explicit corpus owner"
    );
    let owner = owners[0].downcast_ref::<SharedFtsScorerExec>().unwrap();
    let binding = owner.binding().unwrap();
    let at_path = |path: &[usize]| {
        let mut node = owner.children()[0].clone();
        for &child in path {
            node = node.children()[child].clone();
        }
        node
    };
    assert!(at_path(&binding.producer_path).is::<FlatMatchQueryExec>());
    let consumer = at_path(&binding.consumer_path);
    assert!(match kind {
        Kind::Match => consumer.is::<MatchQueryExec>(),
        Kind::Phrase => consumer.is::<PhraseQueryExec>(),
    });

    let expected: Vec<_> = (0..12)
        .filter(|row| !filtered || row % 3 != 1)
        .filter_map(|row| match row % 6 {
            0 => Some((row_id(row, is_stable), vec![0])),
            1 => Some((row_id(row, is_stable), vec![1])),
            2 if matches!(kind, Kind::Match) => Some((row_id(row, is_stable), vec![0])),
            _ => None,
        })
        .collect();
    let first = results(plan.clone()).await;
    assert_eq!(
        first
            .iter()
            .map(|(row, doc, _)| (*row, doc.clone()))
            .collect::<Vec<_>>(),
        expected
    );
    // Exercise scanner-created ownership and reconstructed ownership independently.
    // Reusing either plan must not retain a consumed producer or stale scorer state.
    let rebuilt = restore_plan(plan.clone());
    for replay in [plan, rebuilt.clone(), rebuilt] {
        assert_eq!(results(replay).await, first);
    }
}

#[rstest]
#[case::all(Selection::All, false)]
#[case::resolved(Selection::Resolved, true)]
#[case::uuids(Selection::Uuids, false)]
#[tokio::test]
async fn indexed_fts_state_list_granularity(
    #[values(Kind::Match, Kind::Phrase)] kind: Kind,
    #[case] selection: Selection,
    #[case] is_stable: bool,
) {
    let rows = [
        Some(vec![Some("alpha beta"), Some("alpha")]),
        Some(vec![None, Some("alpha beta")]),
        Some(vec![Some("alpha x beta"), Some("other")]),
        Some(vec![Some("alpha"), Some("beta")]),
        Some(vec![]),
        None,
    ];
    let mut text = ListBuilder::new(StringBuilder::new());
    for row in rows {
        if let Some(values) = row {
            for value in values {
                text.values().append_option(value);
            }
            text.append(true);
        } else {
            text.append(false);
        }
    }
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from_iter_values(0..6)) as ArrayRef,
        ),
        ("text", Arc::new(text.finish()) as ArrayRef),
    ])
    .unwrap();
    let mut dataset = write(batch, is_stable).await;
    let granularity = DocumentGranularity::ListElement;
    let segments = build_segments(&mut dataset, granularity).await;
    let dataset = Arc::new(dataset);
    macro_rules! make_leaf {
        ($ty:ty, $query:expr) => {{
            let query = $query;
            assert_eq!(query.document_granularity, None);
            match selection {
                Selection::All => <$ty>::new_with_document_granularity(
                    dataset.clone(),
                    query,
                    FtsSearchParams::default(),
                    PreFilterSource::None,
                    granularity,
                ),
                Selection::Resolved => <$ty>::new_with_segments_and_document_granularity(
                    dataset.clone(),
                    query,
                    FtsSearchParams::default(),
                    PreFilterSource::None,
                    vec![segments[1].clone()],
                    granularity,
                ),
                Selection::Uuids => <$ty>::new_with_segment_uuids_and_document_granularity(
                    dataset.clone(),
                    query,
                    FtsSearchParams::default(),
                    PreFilterSource::None,
                    vec![segments[0].uuid, segments[0].uuid],
                    granularity,
                ),
            }
        }};
    }
    let original: Arc<dyn ExecutionPlan> = match kind {
        Kind::Match => Arc::new(make_leaf!(
            MatchQueryExec,
            MatchQuery::new("alpha beta".into())
                .with_column(Some("text".into()))
                .with_operator(Operator::And)
        )),
        Kind::Phrase => Arc::new(make_leaf!(
            PhraseQueryExec,
            PhraseQuery::new("alpha beta".into())
                .with_column(Some("text".into()))
                .with_slop(1)
        )),
    };
    let rebuilt = restore_plan(original.clone());
    assert_eq!(rebuilt.schema(), original.schema());
    let original = results(original).await;
    let actual = results(rebuilt).await;
    assert_eq!(actual, original);
    let mut keys = actual
        .into_iter()
        .map(|(row, coords, _)| (row, coords))
        .collect::<Vec<_>>();
    keys.sort();
    let expected_rows = match selection {
        Selection::All => vec![(0, 0), (1, 1), (2, 0)],
        Selection::Resolved => vec![(2, 0)],
        Selection::Uuids => vec![(0, 0), (1, 1)],
    };
    assert_eq!(
        keys,
        expected_rows
            .into_iter()
            .map(|(row, doc)| (row_id(row, is_stable), vec![doc]))
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[tokio::test]
async fn indexed_fts_state_overlay_and_masks(
    #[values(Kind::Match, Kind::Phrase)] kind: Kind,
    #[values(false, true)] is_stable: bool,
) {
    let mut dataset = write(
        record_batch!(
            ("id", Int32, vec![0, 1, 2, 3, 4, 5]),
            (
                "text",
                Utf8,
                vec![
                    "alpha beta",
                    "alpha beta",
                    "alpha beta",
                    "alpha beta",
                    "alpha beta",
                    "other"
                ]
            )
        )
        .unwrap(),
        is_stable,
    )
    .await;
    build_segments(&mut dataset, DocumentGranularity::Row).await;
    let dataset = commit_overlay(
        dataset,
        "indexed_state",
        1,
        &[1],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(StringArray::from(vec!["missing"]))],
    )
    .await;
    let query = match kind {
        Kind::Match => FtsQuery::Match(
            MatchQuery::new("alpha beta".into())
                .with_column(Some("text".into()))
                .with_operator(Operator::And),
        ),
        Kind::Phrase => {
            FtsQuery::Phrase(PhraseQuery::new("alpha beta".into()).with_column(Some("text".into())))
        }
    };
    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .with_row_id()
        .prefilter(true)
        .filter("id != 0")
        .unwrap()
        .with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
            [0, 1, 2, 3, 5].map(|row| row_id(row, is_stable)),
        )))
        .full_text_search(FullTextSearchQuery::new_query(query))
        .unwrap();
    let native = scanner.create_plan().await.unwrap();
    let mut checked = false;
    native
        .apply(|node| {
            let state = if let Some(exec) = node.downcast_ref::<MatchQueryExec>() {
                Some((
                    exec.overlay_block().is_some(),
                    exec.external_mask().is_some(),
                    matches!(exec.prefilter_source(), PreFilterSource::None),
                ))
            } else {
                node.downcast_ref::<PhraseQueryExec>().map(|exec| {
                    (
                        exec.overlay_block().is_some(),
                        exec.external_mask().is_some(),
                        matches!(exec.prefilter_source(), PreFilterSource::None),
                    )
                })
            };
            if let Some(state) = state {
                assert_eq!(state, (true, true, false));
                checked = true;
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
    assert!(checked);
    let expected = results(native).await;
    let actual = results(restore_plan(scanner.create_plan().await.unwrap())).await;
    assert_eq!(actual, expected);
    let mut ids = actual
        .into_iter()
        .map(|(row, _, _)| row)
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, vec![row_id(1, is_stable), row_id(3, is_stable)]);
}

#[rstest]
#[tokio::test]
async fn indexed_fts_state_prepared_canonical_vocabulary(
    #[values(false, true)] has_missing_group: bool,
) {
    let mut dataset = write(
        record_batch!((
            "text",
            Utf8,
            vec!["lance alpha", "lance beta", "lancd alpha", "lancd beta"]
        ))
        .unwrap(),
        false,
    )
    .await;
    let segments = build_segments(&mut dataset, DocumentGranularity::Row).await;
    let mut indices = Vec::new();
    for segment in &segments {
        let generic = dataset
            .open_scalar_index("text", &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        indices.push(Arc::new(
            generic
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .unwrap()
                .clone(),
        ));
    }
    let terms = if has_missing_group {
        vec!["lancx".into(), "zzzzzz".into()]
    } else {
        vec!["lancx".into()]
    };
    let query = MatchQuery::new(terms.join(" "))
        .with_column(Some("text".into()))
        .with_fuzziness(Some(1))
        .with_max_expansions(1)
        .with_operator(Operator::And);
    let params = FtsSearchParams::default()
        .with_fuzziness(Some(1))
        .with_max_expansions(1);
    let prepared = Arc::new(
        prepare_bm25_query(
            &indices,
            Tokens::new(terms, DocType::Text),
            &params,
            None,
            None,
        )
        .await
        .unwrap(),
    );
    assert_eq!(prepared.tokens().get_token(0), "lancd");
    assert_eq!(prepared.has_all_query_positions(), !has_missing_group);
    let dataset = Arc::new(dataset);
    let leaf = |selected: Vec<IndexMetadata>| {
        MatchQueryExec::new_with_segments_and_document_granularity(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            selected,
            DocumentGranularity::Row,
        )
    };
    let original = leaf(segments.clone()).with_prepared_query(prepared.clone());
    let decoded = restore_match(&original);
    assert!(!Arc::ptr_eq(
        original.prepared_query().unwrap(),
        decoded.prepared_query().unwrap()
    ));
    let expected = results(Arc::new(original)).await;
    let actual = results(Arc::new(decoded)).await;
    assert_eq!(actual, expected);
    if has_missing_group {
        assert!(actual.is_empty());
        let lost_completeness = PreparedBm25Query::from_parts(
            prepared.tokens().clone(),
            prepared.scorer().clone(),
            true,
        );
        assert_eq!(
            results(Arc::new(
                leaf(segments).with_prepared_query(Arc::new(lost_completeness))
            ))
            .await
            .len(),
            2
        );
    } else {
        assert_eq!(actual.len(), 2);
        let worker = leaf(vec![segments[0].clone()]).with_prepared_query(prepared.clone());
        assert!(results(Arc::new(restore_match(&worker))).await.is_empty());
        assert_eq!(
            results(Arc::new(leaf(vec![segments[0].clone()])))
                .await
                .len(),
            2,
            "worker-local expansion selects a different vocabulary"
        );
        let local = results(Arc::new(leaf(vec![segments[1].clone()]))).await;
        assert_eq!(
            local.iter().map(|r| r.0).collect::<Vec<_>>(),
            actual.iter().map(|r| r.0).collect::<Vec<_>>()
        );
        assert_ne!(local, actual, "worker-local BM25 statistics change scores");
    }
}

#[tokio::test]
async fn indexed_fts_state_preserves_match_groups() {
    let mut dataset = write(
        record_batch!(("text", Utf8, vec!["alpha", "beta", "alpha beta"])).unwrap(),
        false,
    )
    .await;
    build_segments(&mut dataset, DocumentGranularity::Row).await;
    let tokens = Tokens::try_from_parts(
        vec!["alpha".into(), "beta".into()],
        vec![0, 0],
        vec![0, 1],
        DocType::Text,
    )
    .unwrap();
    let scorer = Arc::new(MemBM25Scorer::new(
        4,
        3,
        std::collections::HashMap::from([("alpha".into(), 2), ("beta".into(), 2)]),
    ));
    let prepared = PreparedBm25Query::from_parts(Arc::new(tokens), scorer.clone(), true);
    let leaf = MatchQueryExec::new_with_document_granularity(
        Arc::new(dataset),
        MatchQuery::new("alpha beta".into())
            .with_column(Some("text".into()))
            .with_operator(Operator::And),
        FtsSearchParams::default(),
        PreFilterSource::None,
        DocumentGranularity::Row,
    )
    .with_prepared_query(Arc::new(prepared));
    let actual = results(Arc::new(restore_match(&leaf))).await;
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].0, row_id(2, false));
    let original = Arc::new(leaf);
    assert_eq!(actual, results(original.clone()).await);
    let lost_groups = PreparedBm25Query::from_parts(
        Arc::new(Tokens::with_positions(
            vec!["alpha".into(), "beta".into()],
            vec![0, 0],
            DocType::Text,
        )),
        scorer,
        true,
    );
    assert_eq!(
        results(Arc::new(
            restore_match(&original).with_prepared_query(Arc::new(lost_groups))
        ))
        .await
        .len(),
        3
    );
}

#[rstest]
#[case::positions(vec![0, 1], vec![0])]
#[case::groups(vec![0], vec![])]
#[test]
fn indexed_fts_state_rejects_malformed_tokens(
    #[case] positions: Vec<u32>,
    #[case] groups: Vec<u32>,
) {
    let error = Tokens::try_from_parts(vec!["alpha".into()], positions, groups, DocType::Text)
        .err()
        .expect("invalid token shape");
    assert!(
        matches!(error, lance_core::Error::InvalidInput { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("lengths must agree"), "{error}");
}
