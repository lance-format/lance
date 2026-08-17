// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The unindexed `combined_fields` plan: blend the scanned column values into
//! `dl'`/`tf'` and score the rows no target column's index covers.

use std::sync::Arc;

use arrow::array::{Float32Builder, UInt64Builder};
use arrow_array::{ArrayRef, RecordBatch};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::Time;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{FutureExt, stream};
use lance_core::error::DataFusionResult;
use lance_core::utils::tokio::spawn_cpu;
use lance_select::RowAddrMask;

use super::super::index::{BlendedRows, FTS_SCHEMA, slice_into_batches, tokenize_and_blend_multi};
use super::super::query::{Operator, Tokens};
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::super::tokenizer::document_tokenizer::LanceTokenizer;
use super::stats::{CombinedCorpusStats, FlatFieldStats, build_combined_bm25_scorer};
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;

/// Exact cross-field BM25F search over rows that no index fully covers, scored
/// straight from their column values. Together with the indexed
/// [`combined_fields_search`](super::combined_fields_search) this covers the
/// whole dataset, with the same term deduplication and `operator` semantics.
///
/// - `input` carries `_rowid` plus every target column; `doc_col_indices` locates
///   them, in `columns` order.
/// - `stats_masks[column]` selects the rows folded into that column's corpus
///   statistics, so a row indexed for some columns is not counted twice.
/// - `emit_mask`, when set, selects the rows to emit from an unfiltered `input`.
/// - `flat_covers_whole_corpus` selects [`CombinedCorpusStats::FlatOnly`].
/// - `metrics` receives only the scorer build's index reads.
///
/// Returns the scorer alongside the stream. The whole input is consumed before
/// the first output batch, so an indexed sibling can score against the same
/// statistics.
#[allow(clippy::too_many_arguments)]
pub async fn flat_combined_fields_search_stream(
    input: SendableRecordBatchStream,
    columns: &[CombinedFieldColumn],
    doc_col_indices: Vec<usize>,
    stats_masks: &[Arc<RowAddrMask>],
    emit_mask: Option<Arc<RowAddrMask>>,
    flat_covers_whole_corpus: bool,
    tokens: &Tokens,
    tokenizer: Box<dyn LanceTokenizer>,
    operator: Operator,
    target_batch_size: usize,
    elapsed_compute: Option<Time>,
    metrics: Option<&dyn MetricsCollector>,
) -> DataFusionResult<(SendableRecordBatchStream, Arc<CombinedFieldsBM25Scorer>)> {
    let terms = unique_terms(tokens);
    debug_assert_eq!(doc_col_indices.len(), columns.len());
    debug_assert_eq!(stats_masks.len(), columns.len());
    // A query that tokenizes to nothing matches nothing, as on the indexed path.
    // The scorer is still returned, because an indexed sibling waits for it.
    if terms.is_empty() || columns.is_empty() {
        let empty = FlatFieldStats::zeros(columns.len(), 0);
        let scorer = Arc::new(
            build_combined_bm25_scorer(
                columns,
                tokens,
                CombinedCorpusStats::for_flat_scan(&empty, flat_covers_whole_corpus),
                metrics,
            )
            .boxed()
            .await?,
        );
        return Ok((
            Box::pin(RecordBatchStreamAdapter::new(
                FTS_SCHEMA.clone(),
                stream::empty::<DataFusionResult<RecordBatch>>(),
            )),
            scorer,
        ));
    }

    let input_schema = input.schema();
    // `tf'` is one value per deduplicated term, so the counter must see the same
    // deduplicated list the indexed scan scores.
    let unique_tokens = Arc::new(Tokens::new(terms.clone(), tokens.token_type().clone()));
    // Same thresholds as the single-column flat path: tokenization is CPU-bound,
    // so batches are accumulated before a task is dispatched.
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    const SLICE_BYTES: usize = 512 * 1024;
    let chunked = lance_arrow::stream::rechunk_stream_by_size(
        input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    );

    let weights: Vec<f32> = columns.iter().map(|column| column.weight).collect();
    let (blended, flat_stats) = tokenize_and_blend_multi(
        chunked,
        tokenizer,
        unique_tokens,
        Arc::new(doc_col_indices),
        Arc::new(weights),
        Arc::new(stats_masks.to_vec()),
        elapsed_compute.clone(),
    )
    .await?;

    // Time the scorer build and the scoring loop together.
    let post_await_start = std::time::Instant::now();
    let scorer = Arc::new(
        build_combined_bm25_scorer(
            columns,
            tokens,
            CombinedCorpusStats::for_flat_scan(&flat_stats, flat_covers_whole_corpus),
            metrics,
        )
        .boxed()
        .await?,
    );
    // `rows x terms` synchronous work, offloaded like the indexed scoring loop so a
    // large flat scan cannot hold a DataFusion worker.
    let scores = {
        let terms_for_scoring = terms.clone();
        let scorer_for_scoring = scorer.clone();
        let require_all_terms = operator == Operator::And;
        spawn_cpu(move || {
            flat_combined_score(
                &terms_for_scoring,
                blended,
                scorer_for_scoring.as_ref(),
                require_all_terms,
                emit_mask.as_deref(),
            )
        })
        .await?
    };

    let batches = slice_into_batches(scores, target_batch_size);
    if let Some(t) = &elapsed_compute {
        t.add_duration(post_await_start.elapsed());
    }
    Ok((
        Box::pin(RecordBatchStreamAdapter::new(
            FTS_SCHEMA.clone(),
            stream::iter(batches),
        )),
        scorer,
    ))
}

/// Score every retained flat row from its blended `dl'`/`tf'`, emitting
/// `(ROW_ID, SCORE)` in [`FTS_SCHEMA`] for the rows `emit_mask` selects.
///
/// Takes the chunks by value and drops each one as it is scored, so the blend and
/// the output arrays are never both fully resident.
fn flat_combined_score(
    terms: &[String],
    blended: Vec<BlendedRows>,
    scorer: &CombinedFieldsBM25Scorer,
    require_all_terms: bool,
    emit_mask: Option<&RowAddrMask>,
) -> DataFusionResult<RecordBatch> {
    let num_terms = terms.len();
    let num_rows = blended.iter().map(|chunk| chunk.row_ids.len()).sum();
    let mut row_ids = UInt64Builder::with_capacity(num_rows);
    let mut scores = Float32Builder::with_capacity(num_rows);
    for chunk in blended {
        for (row, (input_row_id, dl_prime)) in
            chunk.row_ids.iter().zip(&chunk.doc_lengths).enumerate()
        {
            if emit_mask.is_some_and(|mask| !mask.selected(*input_row_id)) {
                continue;
            }
            let tf_prime = &chunk.term_freqs[row * num_terms..(row + 1) * num_terms];
            // `And` spans the virtual field, so it checks the blended `tf'`.
            if require_all_terms && tf_prime.iter().any(|tf| *tf <= 0.0) {
                continue;
            }
            let score: f32 = terms
                .iter()
                .zip(tf_prime)
                .map(|(term, tf)| scorer.query_weight(term) * scorer.doc_weight(*tf, *dl_prime))
                .sum();
            if score > 0.0 {
                row_ids.append_value(*input_row_id);
                scores.append_value(score);
            }
        }
    }

    Ok(RecordBatch::try_new(
        FTS_SCHEMA.clone(),
        vec![
            Arc::new(row_ids.finish()) as ArrayRef,
            Arc::new(scores.finish()) as ArrayRef,
        ],
    )?)
}
