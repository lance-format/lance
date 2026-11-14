// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{RecordBatch, RecordBatchIterator, record_batch};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};

const DOCS: [&str; 3] = ["alpha beta", "gamma delta", "alpha"];

fn corpus() -> RecordBatch {
    record_batch!(
        ("text", Utf8, DOCS.to_vec()),
        ("_rowid", UInt64, vec![0_u64, 1, 2])
    )
    .unwrap()
}

fn input(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
    MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], batch.schema(), None).unwrap()
}

async fn flat_query_exec() -> FlatMatchQueryExec {
    flat_query_exec_with("alpha beta", InvertedIndexParams::default().stem(false)).await
}

async fn flat_query_exec_with(terms: &str, params: InvertedIndexParams) -> FlatMatchQueryExec {
    let batch = corpus();
    // `_rowid` is reserved in datasets; only the in-memory input carries it.
    let text = batch.project(&[0]).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(text.clone())], text.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    let mut query = MatchQuery::new(terms.into()).with_column(Some("text".into()));
    query.document_granularity = Some(DocumentGranularity::Row);
    FlatMatchQueryExec::new(dataset, query, FtsSearchParams::new(), input(&batch))
        .unwrap()
        .with_tokenizer_params(params)
}

async fn row_scores(exec: FlatMatchQueryExec) -> Vec<(u64, f32)> {
    let batches = exec
        .execute(0, Arc::new(TaskContext::default()))
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for batch in batches {
        let row_ids = batch["_rowid"].as_primitive::<UInt64Type>();
        let scores = batch["_score"].as_primitive::<Float32Type>();
        rows.extend(
            row_ids
                .values()
                .iter()
                .copied()
                .zip(scores.values().iter().copied()),
        );
    }
    rows.sort_by_key(|(row_id, _)| *row_id);
    rows
}

fn base() -> MemBM25Scorer {
    MemBM25Scorer::new(
        40,
        10,
        HashMap::from([("alpha".into(), 3), ("beta".into(), 6)]),
    )
}

fn noop_exchange() -> FlatScorerExchange {
    Arc::new(|scorer: MemBM25Scorer| {
        async move { Ok::<_, DataFusionError>(Arc::new(scorer)) }.boxed()
    })
}

#[tokio::test]
async fn flat_query_scores_with_exchanged_scorer() {
    let reference = row_scores(flat_query_exec().await.with_base_scorer(Arc::new(base()))).await;
    assert!(!reference.is_empty());

    let calls = Arc::new(AtomicUsize::new(0));
    let exchange: FlatScorerExchange = {
        let calls = calls.clone();
        Arc::new(move |local: MemBM25Scorer| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(local.num_docs(), DOCS.len());
            let base = base();
            let token_docs = ["alpha", "beta"]
                .into_iter()
                .map(|token| {
                    (
                        token.to_string(),
                        base.num_docs_containing_token(token)
                            + local.num_docs_containing_token(token),
                    )
                })
                .collect();
            let scorer = MemBM25Scorer::new(
                base.total_tokens + local.total_tokens,
                base.num_docs() + local.num_docs(),
                token_docs,
            );
            async move { Ok::<_, DataFusionError>(Arc::new(scorer)) }.boxed()
        })
    };
    let exec = flat_query_exec().await.with_scorer_exchange(exchange);
    assert!(exec.has_scorer_exchange());
    assert!(exec.tokenizer_params().is_some());
    let actual = row_scores(exec).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(actual, reference);
}

#[tokio::test]
async fn flat_query_ignores_tokenizer_params_without_statistics() {
    // These params drop "alpha" as too long; without statistics they are not used.
    let exec = flat_query_exec_with(
        "alpha",
        InvertedIndexParams::default()
            .stem(false)
            .max_token_length(Some(3)),
    )
    .await;
    let rows = row_scores(exec).await;
    assert_eq!(
        rows.iter().map(|(row_id, _)| *row_id).collect::<Vec<_>>(),
        vec![0, 2]
    );
}

#[tokio::test]
async fn flat_query_rejects_scorer_exchange_with_shared_scorer() {
    let exec = flat_query_exec()
        .await
        .with_scorer_exchange(noop_exchange())
        .with_shared_scorer(Arc::new(SharedFtsScorer::new()));
    assert!(exec.has_shared_scorer());
    let Err(error) = exec.execute(0, Arc::new(TaskContext::default())) else {
        panic!("a scorer exchange combined with a shared scorer must fail");
    };
    assert!(
        error
            .to_string()
            .contains("cannot be combined with a shared corpus scorer"),
        "{error}"
    );
}

#[tokio::test]
async fn flat_query_rebuilds_preserve_scorer_settings() {
    let exec = flat_query_exec()
        .await
        .with_shared_scorer(Arc::new(SharedFtsScorer::new()));
    assert!(exec.has_shared_scorer());
    assert!(!exec.has_scorer_exchange());

    let rebuilt = exec.rebuilt_with_scorer_exchange(noop_exchange());
    assert!(rebuilt.has_scorer_exchange());
    assert!(!rebuilt.has_shared_scorer());
    assert!(rebuilt.tokenizer_params().is_some());

    let children = rebuilt.children().into_iter().cloned().collect::<Vec<_>>();
    let rebuilt: Arc<dyn ExecutionPlan> = Arc::new(rebuilt);
    let rebuilt = rebuilt
        .replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let rebuilt = rebuilt.downcast_ref::<FlatMatchQueryExec>().unwrap();
    assert!(rebuilt.has_scorer_exchange());
    assert!(!rebuilt.has_shared_scorer());
    assert_eq!(
        rebuilt.tokenizer_params(),
        Some(&InvertedIndexParams::default().stem(false))
    );

    let params = InvertedIndexParams::default().stem(false).lower_case(false);
    let rebuilt = rebuilt.rebuilt_with_tokenizer_params(params.clone());
    assert_eq!(rebuilt.tokenizer_params(), Some(&params));
    assert!(rebuilt.has_scorer_exchange());

    let without_exchange = flat_query_exec().await;
    assert!(!without_exchange.has_scorer_exchange());
    let display = datafusion::physical_plan::displayable(&without_exchange)
        .one_line()
        .to_string();
    assert!(!display.contains("scorer_exchange"), "{display}");
    let display = datafusion::physical_plan::displayable(&rebuilt)
        .one_line()
        .to_string();
    assert!(display.contains(", scorer_exchange=true"), "{display}");
}
