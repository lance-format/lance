// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::DatasetIndexExt;
use crate::io::exec::fts::FtsCorpusBinding;
use crate::io::exec::fts::tests::{assert_execution_error, execute_results};
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use arrow_array::RecordBatchIterator;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use lance_datafusion::exec::OneShotExec;
use lance_index::scalar::inverted::query::CombinedFieldsQuery;
use lance_index::scalar::inverted::query::{FtsQuery, Operator, PhraseQuery};
use lance_index::{
    IndexType,
    scalar::{FullTextSearchQuery, InvertedIndexParams},
};

fn binding() -> FtsCorpusBinding {
    FtsCorpusBinding {
        producer_path: vec![0, 1],
        consumer_path: vec![0, 0],
    }
}

fn branches(children: Vec<Arc<dyn ExecutionPlan>>) -> Arc<dyn ExecutionPlan> {
    Arc::new(CoalescePartitionsExec::new(
        UnionExec::try_new(children).unwrap(),
    ))
}

fn match_pair(
    dataset: Arc<Dataset>,
    query: MatchQuery,
    input: Arc<dyn ExecutionPlan>,
) -> (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) {
    let indexed = Arc::new(
        MatchQueryExec::new(
            dataset.clone(),
            query.clone(),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap(),
    );
    let flat = Arc::new(
        FlatMatchQueryExec::new(
            Arc::new(dataset.as_ref().clone()),
            query,
            FtsSearchParams::default(),
            input,
        )
        .unwrap(),
    );
    (indexed, flat)
}

fn query() -> MatchQuery {
    MatchQuery::new("alpha beta".to_string())
        .with_column(Some("text".to_string()))
        .with_document_granularity(DocumentGranularity::Row)
}

fn residual_batch() -> RecordBatch {
    record_batch!(("text", Utf8, ["alpha"]), (ROW_ID, UInt64, [100])).unwrap()
}

fn assert_binding_error(result: DataFusionResult<SharedFtsScorerExec>, message: &str) {
    let error = result.unwrap_err();
    assert!(matches!(error, DataFusionError::Plan(_)), "{error}");
    assert!(error.to_string().contains(message), "{error}");
}

#[rstest]
#[case::out_of_bounds(vec![0, 7], vec![0, 0], "has no child")]
#[case::wrong_roles(vec![0, 0], vec![0, 1], "flat producer and indexed consumer")]
#[case::same_node(vec![0, 1], vec![0, 1], "separate branches")]
#[case::descendant(vec![0, 1], vec![0, 1, 0], "separate branches")]
#[tokio::test]
async fn shared_corpus_binding_rejects_invalid_paths(
    #[case] producer_path: Vec<usize>,
    #[case] consumer_path: Vec<usize>,
    #[case] message: &str,
) {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset, query(), memory_input(residual_batch()));
    assert_binding_error(
        SharedFtsScorerExec::try_new(
            branches(vec![indexed, flat]),
            FtsCorpusBinding {
                producer_path,
                consumer_path,
            },
        ),
        message,
    );
}

#[rstest]
#[case::query(false)]
#[case::manifest_view(true)]
#[tokio::test]
async fn shared_corpus_binding_rejects_different_corpora(#[case] is_manifest_view: bool) {
    let dataset = indexed_match_dataset().await;
    let (indexed, _) = match_pair(dataset.clone(), query(), memory_input(residual_batch()));
    let (producer_dataset, producer_query, message) = if is_manifest_view {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.fragments = Arc::new(Vec::new());
        let view = Arc::new(dataset.with_manifest_view(Arc::new(manifest)));
        assert_eq!(dataset.uri(), view.uri());
        assert_eq!(dataset.version_id(), view.version_id());
        (view, query(), "opened dataset snapshot")
    } else {
        (
            dataset,
            MatchQuery {
                terms: "beta".into(),
                ..query()
            },
            "same Match query",
        )
    };
    let (_, flat) = match_pair(
        producer_dataset,
        producer_query,
        memory_input(residual_batch()),
    );
    assert_binding_error(
        SharedFtsScorerExec::try_new(branches(vec![indexed, flat]), binding()),
        message,
    );
}

#[tokio::test]
async fn shared_corpus_binding_rejects_unowned_shared_leaf() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset.clone(), query(), memory_input(residual_batch()));
    let orphan = Arc::new(
        MatchQueryExec::new(
            dataset,
            query(),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap()
        .with_shared_scorer(Arc::new(SharedFtsScorer::new())),
    );
    assert_binding_error(
        SharedFtsScorerExec::try_new(branches(vec![indexed, flat, orphan]), binding()),
        "unbound shared-scorer leaf",
    );
}

#[tokio::test]
async fn shared_corpus_binding_preserves_nested_owners() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset.clone(), query(), memory_input(residual_batch()));
    let (inner_indexed, inner_flat) = match_pair(
        dataset,
        MatchQuery {
            terms: "beta".into(),
            ..query()
        },
        memory_input(residual_batch()),
    );
    let inner: Arc<dyn ExecutionPlan> = Arc::new(
        SharedFtsScorerExec::try_new(branches(vec![inner_indexed, inner_flat]), binding()).unwrap(),
    );
    let outer_reference =
        SharedFtsScorerExec::try_new(branches(vec![indexed.clone(), flat.clone()]), binding())
            .unwrap();
    let mut expected = execute_results(&outer_reference).await.unwrap();
    expected.extend(execute_results(inner.as_ref()).await.unwrap());
    let owner = SharedFtsScorerExec::try_new(
        branches(vec![indexed.clone(), flat, inner.clone()]),
        binding(),
    )
    .unwrap();
    assert_eq!(owner.binding().unwrap(), binding());
    let mut actual = execute_results(&owner).await.unwrap();
    let order = |a: &(u64, f32), b: &(u64, f32)| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1));
    expected.sort_by(order);
    actual.sort_by(order);
    assert_eq!(actual, expected);

    assert_binding_error(
        SharedFtsScorerExec::try_new(
            branches(vec![indexed, inner]),
            FtsCorpusBinding {
                producer_path: vec![0, 1, 0, 0, 1],
                consumer_path: vec![0, 0],
            },
        ),
        "crosses a nested owner",
    );
}

#[tokio::test]
async fn shared_corpus_binding_tracks_child_rewrites() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset, query(), memory_input(residual_batch()));
    let owner: Arc<dyn ExecutionPlan> =
        Arc::new(SharedFtsScorerExec::try_new(branches(vec![indexed, flat]), binding()).unwrap());
    let expected = execute_results(owner.as_ref()).await.unwrap();
    let new_input = Arc::new(CoalescePartitionsExec::new(owner.children()[0].clone()));
    let owner = owner
        .replace_children(
            vec![new_input],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let owner = owner.downcast_ref::<SharedFtsScorerExec>().unwrap();
    let current = owner.binding().unwrap();
    assert_eq!(current.producer_path, vec![0, 0, 1]);
    assert_eq!(current.consumer_path, vec![0, 0, 0]);
    let rebuilt = SharedFtsScorerExec::try_new(owner.children()[0].clone(), current).unwrap();
    assert_eq!(execute_results(&rebuilt).await.unwrap(), expected);
    assert_eq!(execute_results(&rebuilt).await.unwrap(), expected);
}

#[tokio::test]
async fn shared_corpus_binding_does_not_replay_one_shot_children() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(
        dataset,
        query(),
        Arc::new(OneShotExec::from_batch(residual_batch())),
    );
    let owner = SharedFtsScorerExec::try_new(branches(vec![indexed, flat]), binding()).unwrap();
    assert!(!execute_results(&owner).await.unwrap().is_empty());
    let error = tokio::time::timeout(Duration::from_secs(1), execute_results(&owner))
        .await
        .expect("a consumed producer must return an error, not leave its consumer waiting")
        .unwrap_err();
    assert!(matches!(error, DataFusionError::Execution(_)), "{error}");
    assert!(
        error.to_string().contains("already been executed"),
        "{error}"
    );
}

fn find_owner(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.is::<SharedFtsScorerExec>() {
        return Some(plan.clone());
    }
    plan.children().into_iter().find_map(find_owner)
}

async fn combined_owner() -> Arc<dyn ExecutionPlan> {
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "title",
            lance_datagen::array::cycle_utf8_literals(&["alpha alpha", "beta"]),
        )
        .col(
            "body",
            lance_datagen::array::cycle_utf8_literals(&["beta", "beta beta"]),
        )
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
        .await
        .unwrap();
    for column in ["title", "body"] {
        dataset
            .create_index(
                &[column],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .stem(false)
                    .remove_stop_words(false),
                true,
            )
            .await
            .unwrap();
    }
    let tail = record_batch!(
        ("title", Utf8, vec!["alpha"; 10]),
        ("body", Utf8, vec!["alpha"; 10])
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(tail.clone())], tail.schema()),
            None,
        )
        .await
        .unwrap();
    let mut query = FullTextSearchQuery::new_query(FtsQuery::CombinedFields(
        CombinedFieldsQuery::try_new("alpha beta".into(), vec!["title".into(), "body".into()])
            .unwrap(),
    ));
    query.limit = Some(1);
    let mut scanner = dataset.scan();
    scanner.full_text_search(query).unwrap();
    let plan = scanner.create_plan().await.unwrap();
    find_owner(&plan).expect("partial cross-field coverage needs a shared corpus owner")
}

#[rstest]
#[case::execute(ResidualFailure::Execute)]
#[case::stream(ResidualFailure::Stream)]
#[case::cancel(ResidualFailure::Cancel)]
#[tokio::test]
async fn shared_corpus_combined_reconstruction_replays_after_failure(
    #[case] failure: ResidualFailure,
) {
    let native = combined_owner().await;
    let expected = execute_results(native.as_ref()).await.unwrap();
    assert_eq!(expected.len(), 1);
    assert_eq!(
        expected[0].0, 1,
        "the residual alpha documents must affect the indexed winner"
    );
    let native_owner = native.downcast_ref::<SharedFtsScorerExec>().unwrap();
    let state = native_owner.binding().unwrap();
    let mut residual = None;
    // Model decoded leaves with independently allocated coordination, sharing
    // no original scorer. Keep native coverage, field order and statistics scope.
    let input = native_owner.children()[0]
        .clone()
        .transform_up(|plan| {
            if let Some(node) = plan.downcast_ref::<FlatCombinedFieldsExec>() {
                let mut children = node.children().into_iter().cloned().collect::<Vec<_>>();
                let source = Arc::new(FailingResidualExec::new(children[0].clone(), failure));
                children[0] = source.clone();
                residual = Some(source);
                let replaced = plan.replace_children(
                    children,
                    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
                )?;
                let node = replaced
                    .downcast_ref::<FlatCombinedFieldsExec>()
                    .unwrap()
                    .clone()
                    .with_shared_scorer(Arc::new(SharedFtsScorer::new()));
                return Ok(Transformed::yes(Arc::new(node) as Arc<dyn ExecutionPlan>));
            }
            if let Some(node) = plan.downcast_ref::<CombinedFieldsQueryExec>() {
                let node = node
                    .clone()
                    .with_shared_scorer(Arc::new(SharedFtsScorer::new()));
                return Ok(Transformed::yes(Arc::new(node) as Arc<dyn ExecutionPlan>));
            }
            Ok(Transformed::no(plan))
        })
        .unwrap()
        .data;
    let residual = residual.unwrap();
    let rebuilt = SharedFtsScorerExec::try_new(input, state.clone()).unwrap();
    assert_eq!(rebuilt.binding().unwrap(), state);
    let output = rebuilt
        .execute(0, Arc::new(TaskContext::default()))
        .unwrap();
    if matches!(failure, ResidualFailure::Cancel) {
        tokio::time::timeout(Duration::from_secs(1), residual.started.notified())
            .await
            .unwrap();
        drop(output);
        tokio::time::timeout(Duration::from_secs(1), async {
            while !residual.dropped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancellation must release the producer while its owner is retained");
    } else {
        let error = tokio::time::timeout(Duration::from_secs(1), output.try_collect::<Vec<_>>())
            .await
            .unwrap()
            .unwrap_err();
        assert_execution_error(error, "injected residual");
    }
    for _ in 0..2 {
        let actual = tokio::time::timeout(Duration::from_secs(1), execute_results(&rebuilt))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[tokio::test]
async fn shared_corpus_binding_rejects_duplicate_or_missing_roles() {
    let missing = SharedFtsScorerExec::new(
        Arc::new(EmptyExec::new(FTS_SCHEMA.clone())),
        Arc::new(SharedFtsScorer::new()),
    );
    let error = missing.binding().unwrap_err();
    assert!(matches!(error, DataFusionError::Plan(_)));
    assert!(error.to_string().contains("got 0 and 0"), "{error}");

    let dataset = indexed_match_dataset().await;
    let shared = Arc::new(SharedFtsScorer::new());
    let indexed: Arc<dyn ExecutionPlan> = Arc::new(
        MatchQueryExec::new(
            dataset.clone(),
            query(),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap()
        .with_shared_scorer(shared.clone()),
    );
    let flat = Arc::new(
        FlatMatchQueryExec::new(
            dataset,
            query(),
            FtsSearchParams::default(),
            memory_input(residual_batch()),
        )
        .unwrap()
        .with_shared_scorer(shared.clone()),
    );
    let duplicate =
        SharedFtsScorerExec::new(branches(vec![indexed.clone(), flat, indexed]), shared);
    let error = duplicate.binding().unwrap_err();
    assert!(matches!(error, DataFusionError::Plan(_)));
    assert!(error.to_string().contains("got 1 and 2"), "{error}");
}

#[tokio::test]
async fn shared_corpus_binding_rejects_different_field_weights() {
    let native = combined_owner().await;
    let owner = native.downcast_ref::<SharedFtsScorerExec>().unwrap();
    let input = owner.children()[0]
        .clone()
        .transform_up(|plan| {
            if let Some(node) = plan.downcast_ref::<CombinedFieldsQueryExec>() {
                let query = node
                    .query()
                    .clone()
                    .try_with_boosts(vec![2.0, 1.0])
                    .unwrap();
                let node = CombinedFieldsQueryExec::new(
                    node.dataset().clone(),
                    query,
                    node.params().clone(),
                    node.prefilter_source().clone(),
                );
                return Ok(Transformed::yes(Arc::new(node) as Arc<dyn ExecutionPlan>));
            }
            Ok(Transformed::no(plan))
        })
        .unwrap()
        .data;
    assert_binding_error(
        SharedFtsScorerExec::try_new(input, owner.binding().unwrap()),
        "same combined_fields query",
    );
}

#[tokio::test]
async fn shared_corpus_binding_rejects_different_resolved_granularity() {
    let dataset = indexed_match_dataset().await;
    let logical = MatchQuery {
        document_granularity: None,
        ..query()
    };
    let indexed = Arc::new(MatchQueryExec::new_with_document_granularity(
        dataset.clone(),
        logical.clone(),
        FtsSearchParams::default(),
        PreFilterSource::None,
        DocumentGranularity::Row,
    ));
    let flat = Arc::new(FlatMatchQueryExec::new_with_document_granularity(
        dataset,
        logical,
        FtsSearchParams::default(),
        memory_input(
            record_batch!(
                ("text", Utf8, ["alpha"]),
                (ROW_ID, UInt64, [100]),
                ("_doc_index_0", UInt32, [0])
            )
            .unwrap(),
        ),
        DocumentGranularity::ListElement,
        "text".to_string(),
    ));
    // A projection can make the output schemas agree while the statistical
    // document units still differ. The owner must validate the actual leaves.
    let expressions: Vec<_> = [ROW_ID, SCORE_COL]
        .into_iter()
        .map(|name| {
            (
                Arc::new(Column::new(name, flat.schema().index_of(name).unwrap()))
                    as Arc<dyn PhysicalExpr>,
                name.to_string(),
            )
        })
        .collect();
    let projected = Arc::new(ProjectionExec::try_new(expressions, flat).unwrap());
    assert_binding_error(
        SharedFtsScorerExec::try_new(
            branches(vec![indexed, projected]),
            FtsCorpusBinding {
                producer_path: vec![0, 1, 0],
                consumer_path: vec![0, 0],
            },
        ),
        "document granularity",
    );
}

#[tokio::test]
async fn shared_corpus_executions_do_not_publish_to_each_other() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset, query(), memory_input(residual_batch()));
    let owner = SharedFtsScorerExec::try_new(branches(vec![indexed, flat]), binding()).unwrap();
    let first = owner.prepare_execution().unwrap();
    let second = owner.prepare_execution().unwrap();
    let first_union = first.children()[0].clone();
    let second_union = second.children()[0].clone();
    let first_consumer = first_union.children()[0].clone();
    let second_consumer = second_union.children()[0].clone();
    let first_producer = first_union.children()[1].clone();
    let second_producer = second_union.children()[1].clone();
    let context = Arc::new(TaskContext::default());
    let mut first_stream = first_consumer.execute(0, context.clone()).unwrap();
    // Drive the real index reader to its scorer wait, as in the existing
    // failure/replay test, before another execution publishes statistics.
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            tokio::select! {
                result = first_stream.try_next() => panic!("consumer completed before its producer: {result:?}"),
                _ = tokio::task::yield_now() => {}
            }
            if metric_value(first_consumer.as_ref(), PARTITIONS_SEARCHED_METRIC) > 0 { break; }
        }
    }).await.unwrap();
    assert!(first_stream.try_next().now_or_never().is_none());
    execute_results(second_producer.as_ref()).await.unwrap();
    let expected = second_consumer
        .execute(0, context.clone())
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(expected.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert!(
        first_stream.try_next().now_or_never().is_none(),
        "publishing another execution's scorer must not unblock this consumer"
    );
    execute_results(first_producer.as_ref()).await.unwrap();
    let actual = first_stream.try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn shared_corpus_cloned_nested_owner_keeps_its_scope() {
    let dataset = indexed_match_dataset().await;
    let (indexed, flat) = match_pair(dataset, query(), memory_input(residual_batch()));
    let original: Arc<dyn ExecutionPlan> =
        Arc::new(SharedFtsScorerExec::try_new(branches(vec![indexed, flat]), binding()).unwrap());
    let mut expected = execute_results(original.as_ref()).await.unwrap();
    expected.extend(expected.clone());
    expected.sort_by_key(|(row, _)| *row);
    let union = original.children()[0].children()[0].clone();
    let input = branches(vec![
        union.children()[0].clone(),
        union.children()[1].clone(),
        original.clone(),
    ]);
    // A normal child rewrite can reuse the owner's template Arc in a nested
    // owner. The boundaries, rather than pointer equality alone, define scope.
    let rewritten = original
        .replace_children(
            vec![input],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let owner = rewritten.downcast_ref::<SharedFtsScorerExec>().unwrap();
    assert_eq!(owner.binding().unwrap(), binding());
    assert_eq!(execute_results(owner).await.unwrap(), expected);
    let rebuilt =
        SharedFtsScorerExec::try_new(owner.children()[0].clone(), owner.binding().unwrap())
            .unwrap();
    assert_eq!(execute_results(&rebuilt).await.unwrap(), expected);
}

#[tokio::test]
async fn shared_corpus_binding_accepts_phrase_pair() {
    let dataset = indexed_match_dataset().await;
    let phrase = PhraseQuery::new("alpha beta".to_string())
        .with_column(Some("text".to_string()))
        .with_document_granularity(DocumentGranularity::Row);
    let match_query = MatchQuery::new(phrase.terms.clone())
        .with_column(phrase.column.clone())
        .with_operator(Operator::And)
        .with_document_granularity(DocumentGranularity::Row);
    let flat_params = FtsSearchParams::default().with_phrase_slop(Some(phrase.slop));
    let consumer: Arc<dyn ExecutionPlan> = Arc::new(
        PhraseQueryExec::new(
            dataset.clone(),
            phrase,
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap(),
    );
    let producer: Arc<dyn ExecutionPlan> = Arc::new(
        FlatMatchQueryExec::new(
            Arc::new(dataset.as_ref().clone()),
            match_query,
            flat_params,
            memory_input(residual_batch()),
        )
        .unwrap(),
    );
    let owner =
        SharedFtsScorerExec::try_new(branches(vec![consumer, producer]), binding()).unwrap();
    assert_eq!(owner.binding().unwrap(), binding());
}
