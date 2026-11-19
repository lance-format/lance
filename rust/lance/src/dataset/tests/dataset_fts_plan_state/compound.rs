// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::io::exec::fts::CompoundQueryExec;
use datafusion::datasource::memory::MemorySourceConfig;
use lance_index::scalar::inverted::build_global_bm25_scorer;
use lance_index::scalar::inverted::query::{BooleanQuery, BoostQuery, Occur};

fn restore(exec: &CompoundQueryExec) -> CompoundQueryExec {
    let mut rebuilt = match exec.preset_segments() {
        Some(segments) => CompoundQueryExec::new_with_segments(
            exec.dataset().clone(),
            exec.query().clone(),
            exec.params().clone(),
            exec.prefilter_source().clone(),
            segments.to_vec(),
        ),
        None => CompoundQueryExec::new_with_segment_uuids(
            exec.dataset().clone(),
            exec.query().clone(),
            exec.params().clone(),
            exec.prefilter_source().clone(),
            exec.explicit_segment_uuids().unwrap(),
        ),
    }
    .with_external_mask(
        exec.external_mask()
            .map(|mask| Arc::new(mask.as_ref().clone())),
    );
    if let Some(scorer) = exec.base_scorer() {
        rebuilt = rebuilt.with_base_scorer(Arc::new(MemBM25Scorer::new(
            scorer.total_tokens,
            scorer.num_docs,
            scorer.token_docs.clone(),
        )));
    }
    if let Some(prepared) = exec.prepared_match() {
        // Reuse the portable-parts reconstruction verified by the parent API PR.
        rebuilt = rebuilt.with_prepared_match(restore_prepared(prepared));
    }
    rebuilt
}

async fn open_indices(dataset: &Dataset, segments: &[IndexMetadata]) -> Vec<Arc<InvertedIndex>> {
    let mut indices = Vec::with_capacity(segments.len());
    for segment in segments {
        let index = dataset
            .open_scalar_index("text", &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        indices.push(Arc::new(
            index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .unwrap()
                .clone(),
        ));
    }
    indices
}

#[rstest]
#[case::resolved_boolean(Selection::Resolved, false, false)]
#[case::uuid_boolean(Selection::Uuids, true, false)]
#[case::resolved_boost(Selection::Resolved, true, true)]
#[case::uuid_boost(Selection::Uuids, false, true)]
#[tokio::test]
async fn compound_leaf_state_masks_and_scorer(
    #[case] selection: Selection,
    #[case] is_stable: bool,
    #[case] is_boost: bool,
) {
    let mut dataset = write(
        record_batch!((
            "text",
            Utf8,
            vec![
                "alpha beta",
                "alpha beta alpha",
                "alpha beta",
                "alpha beta beta",
                "alpha beta alpha",
                "alpha beta beta alpha",
            ]
        ))
        .unwrap(),
        is_stable,
    )
    .await;
    let segments = build_segments(&mut dataset, DocumentGranularity::Row).await;
    let indices = open_indices(&dataset, &segments).await;
    let scorer = Arc::new(
        build_global_bm25_scorer(
            &indices,
            &Tokens::new(vec!["alpha".into(), "beta".into()], DocType::Text),
            &FtsSearchParams::default().with_fuzziness(Some(0)),
            None,
        )
        .await
        .unwrap(),
    );
    let dataset = Arc::new(dataset);
    let selected = vec![segments[2].clone(), segments[1].clone()];
    let alpha = MatchQuery::new("alpha".into())
        .with_column(Some("text".into()))
        .with_fuzziness(Some(0));
    let beta = MatchQuery::new("beta".into())
        .with_column(Some("text".into()))
        .with_fuzziness(Some(0));
    let phrase = PhraseQuery::new("alpha beta".into()).with_column(Some("text".into()));
    let query: FtsQuery = if is_boost {
        BoostQuery::new(
            alpha.clone().with_boost(2.0).into(),
            phrase.clone().into(),
            Some(0.5),
        )
        .into()
    } else {
        BooleanQuery::new([
            (Occur::Must, alpha.clone().into()),
            (Occur::Should, beta.clone().with_boost(2.0).into()),
        ])
        .into()
    };
    let params = FtsSearchParams::default().with_limit(Some(2));
    let candidates = record_batch!((
        ROW_ID,
        UInt64,
        (0..6)
            .filter(|row| *row != 3)
            .map(|row| row_id(row, is_stable))
            .collect::<Vec<_>>()
    ))
    .unwrap();
    let input =
        MemorySourceConfig::try_new_exec(&[vec![candidates.clone()]], candidates.schema(), None)
            .unwrap();
    let prefilter = PreFilterSource::FilteredRowIds(input);
    let original = match selection {
        Selection::Resolved => CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query,
            params,
            prefilter,
            selected.clone(),
        ),
        Selection::Uuids => CompoundQueryExec::new_with_segment_uuids(
            dataset.clone(),
            query,
            params,
            prefilter,
            vec![selected[0].uuid, selected[1].uuid, selected[0].uuid],
        ),
        Selection::All => unreachable!(),
    }
    .with_external_mask(Some(Arc::new(RowAddrMask::from_block(
        RowAddrTreeMap::from_iter([row_id(4, is_stable)]),
    ))))
    .with_base_scorer(scorer.clone());
    let restored = restore(&original);
    assert_eq!(
        restored.preset_segments().is_some(),
        matches!(selection, Selection::Resolved)
    );
    assert_eq!(
        restored.explicit_segment_uuids(),
        Some(selected.iter().map(|s| s.uuid).collect())
    );
    let actual = results(Arc::new(restored)).await;
    assert_eq!(actual, results(Arc::new(original)).await);
    assert_eq!(
        actual.iter().map(|row| row.0).collect::<Vec<_>>(),
        vec![row_id(2, is_stable), row_id(5, is_stable)]
    );

    // Compose independent leaf scores over the selected domain using the same
    // full-corpus statistics, then apply candidate restrictions outside scoring.
    let alpha_scores = results(Arc::new(
        MatchQueryExec::new_with_segments_and_document_granularity(
            dataset.clone(),
            alpha,
            FtsSearchParams::default(),
            PreFilterSource::None,
            selected.clone(),
            DocumentGranularity::Row,
        )
        .with_base_scorer(scorer.clone()),
    ))
    .await;
    let other: Arc<dyn ExecutionPlan> = if is_boost {
        Arc::new(
            PhraseQueryExec::new_with_segments_and_document_granularity(
                dataset.clone(),
                phrase,
                FtsSearchParams::default(),
                PreFilterSource::None,
                selected,
                DocumentGranularity::Row,
            )
            .with_base_scorer(scorer),
        )
    } else {
        Arc::new(
            MatchQueryExec::new_with_segments_and_document_granularity(
                dataset,
                beta,
                FtsSearchParams::default(),
                PreFilterSource::None,
                selected,
                DocumentGranularity::Row,
            )
            .with_base_scorer(scorer),
        )
    };
    let other_scores = results(other).await;
    for (id, _, score) in actual {
        let alpha = alpha_scores.iter().find(|row| row.0 == id).unwrap().2;
        let other = other_scores.iter().find(|row| row.0 == id).unwrap().2;
        let expected = if is_boost {
            2.0 * alpha - 0.5 * other
        } else {
            alpha + 2.0 * other
        };
        assert!(
            (score - expected).abs() < 1e-5,
            "row {id}: {score} != {expected}"
        );
    }
}

#[rstest]
#[tokio::test]
async fn compound_leaf_state_prepared_match(
    #[values(None, Some(2))] limit: Option<usize>,
    #[values(false, true)] has_missing_group: bool,
) {
    let mut dataset = write(
        record_batch!((
            "text",
            Utf8,
            vec!["lance alpha", "lance beta", "lancd alpha", "lancd beta",]
        ))
        .unwrap(),
        false,
    )
    .await;
    let segments = build_segments(&mut dataset, DocumentGranularity::Row).await;
    let indices = open_indices(&dataset, &segments).await;
    let terms = if has_missing_group {
        vec!["lancx".into(), "zzzzzz".into()]
    } else {
        vec!["lancx".into()]
    };
    let query: FtsQuery = MatchQuery::new(terms.join(" "))
        .with_column(Some("text".into()))
        .with_fuzziness(Some(1))
        .with_max_expansions(1)
        .with_operator(Operator::And)
        .into();
    let params = FtsSearchParams::default()
        .with_limit(limit)
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
    let leaf = |chosen: Vec<IndexMetadata>| {
        CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            chosen,
        )
    };
    let original = leaf(segments.clone()).with_prepared_match(prepared.clone());
    let restored = restore(&original);
    assert!(!Arc::ptr_eq(
        original.prepared_match().unwrap(),
        restored.prepared_match().unwrap()
    ));
    let actual = results(Arc::new(restored)).await;
    assert_eq!(actual, results(Arc::new(original)).await);
    if has_missing_group {
        assert!(actual.is_empty());
        let lost_flag = PreparedBm25Query::from_parts(
            prepared.tokens().clone(),
            prepared.scorer().clone(),
            true,
        );
        assert_eq!(
            results(Arc::new(
                leaf(segments).with_prepared_match(Arc::new(lost_flag))
            ))
            .await
            .len(),
            2
        );
    } else {
        assert_eq!(actual.len(), 2);
        let worker = leaf(vec![segments[0].clone()]).with_prepared_match(prepared);
        assert!(results(Arc::new(restore(&worker))).await.is_empty());
        assert_eq!(
            results(Arc::new(leaf(vec![segments[0].clone()])))
                .await
                .len(),
            2,
            "losing prepared state would select the worker-local vocabulary"
        );
    }
}

#[rstest]
#[case::resolved(true)]
#[case::uuids(false)]
#[tokio::test]
async fn compound_leaf_state_empty_selection(#[case] is_resolved: bool) {
    let mut dataset = write(
        record_batch!(("text", Utf8, ["alpha", "beta"])).unwrap(),
        false,
    )
    .await;
    build_segments(&mut dataset, DocumentGranularity::Row).await;
    let query: FtsQuery = MatchQuery::new("alpha".into())
        .with_column(Some("text".into()))
        .into();
    let original = if is_resolved {
        CompoundQueryExec::new_with_segments(
            Arc::new(dataset),
            query,
            FtsSearchParams::default(),
            PreFilterSource::None,
            vec![],
        )
    } else {
        CompoundQueryExec::new_with_segment_uuids(
            Arc::new(dataset),
            query,
            FtsSearchParams::default(),
            PreFilterSource::None,
            vec![],
        )
    };
    let restored = restore(&original);
    assert_eq!(restored.preset_segments().is_some(), is_resolved);
    assert_eq!(restored.explicit_segment_uuids(), Some(vec![]));
    let error = execute_plan(Arc::new(restored), Default::default())
        .unwrap()
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            datafusion::error::DataFusionError::Execution(_)
                | datafusion::error::DataFusionError::External(_)
        ),
        "{error}"
    );
    let native_error = execute_plan(Arc::new(original), Default::default())
        .unwrap()
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), native_error.to_string());
}

#[tokio::test]
async fn compound_leaf_state_rejects_non_match_preparation() {
    let mut dataset = write(
        record_batch!(("text", Utf8, ["alpha", "beta"])).unwrap(),
        false,
    )
    .await;
    let segments = build_segments(&mut dataset, DocumentGranularity::Row).await;
    let query: FtsQuery = BooleanQuery::new([(
        Occur::Must,
        MatchQuery::new("alpha".into())
            .with_column(Some("text".into()))
            .into(),
    )])
    .into();
    let prepared = PreparedBm25Query::from_parts(
        Arc::new(Tokens::new(vec!["alpha".into()], DocType::Text)),
        Arc::new(MemBM25Scorer::new(
            2,
            2,
            std::collections::HashMap::from([("alpha".into(), 1)]),
        )),
        true,
    );
    let original = CompoundQueryExec::new_with_segments(
        Arc::new(dataset),
        query,
        FtsSearchParams::default(),
        PreFilterSource::None,
        segments,
    )
    .with_prepared_match(Arc::new(prepared));
    let error = execute_plan(Arc::new(restore(&original)), Default::default())
        .unwrap()
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap_err();
    assert!(
        matches!(error, datafusion::error::DataFusionError::Execution(_)),
        "{error}"
    );
    assert!(
        error.to_string().contains("requires a root Match query"),
        "{error}"
    );
}
