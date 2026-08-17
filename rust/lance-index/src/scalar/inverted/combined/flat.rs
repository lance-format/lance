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
use lance_core::Error;
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

/// Exact cross-field BM25F search over rows that no index covers.
///
/// The indexed [`combined_fields_search`](super::combined_fields_search) can
/// only score a row whose fragment
/// every target column's index covers; otherwise `dl'` and `tf'` would be missing
/// a column's contribution. This scores the remaining rows straight from their
/// column values, so the two plans together cover the whole dataset.
///
/// `input` must carry `_rowid` plus every target column, in `columns` order.
/// Query terms are deduplicated exactly as
/// [`combined_fields_search`](super::combined_fields_search) does, and
/// `operator` applies across the virtual field: `And` requires every term to
/// appear in at least one target column.
///
/// Unless `base_scorer` overrides it, the blended scorer folds these rows' own
/// per-column statistics in through [`FlatFieldStats`](super::FlatFieldStats),
/// so `avgdl'` and `idf'`
/// reflect the scanned rows rather than only the indexed ones. `stats_masks`
/// keeps that fold from double counting rows a column's index already holds.
/// `metrics` covers only that scorer build's index reads; the plan already
/// accounts for the input stream's IO.
///
/// `emit_mask`, when set, selects the rows to emit. The caller reads `input`
/// unfiltered in that case, so the fold sees rows the query filtered out and a folded
/// row's contribution does not move with the filter.
///
/// `flat_covers_whole_corpus` drops the index statistics in favour of this scan's own;
/// see [`CombinedCorpusStats::FlatOnly`].
///
/// Returns the scorer alongside the stream. The whole input is consumed before
/// the first output batch, so by then the blend describes the entire scanned
/// corpus and an indexed sibling can score against the same statistics.
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
    base_scorer: Option<Arc<CombinedFieldsBM25Scorer>>,
    target_batch_size: usize,
    elapsed_compute: Option<Time>,
    metrics: Option<&dyn MetricsCollector>,
) -> DataFusionResult<(SendableRecordBatchStream, Arc<CombinedFieldsBM25Scorer>)> {
    let terms = unique_terms(tokens);
    // Check the caller's arity first: a call with the wrong number of columns is a
    // bug whatever the query tokenizes to, and reporting it only for non-empty
    // queries would let it through intermittently.
    if doc_col_indices.len() != columns.len() || stats_masks.len() != columns.len() {
        return Err(Error::invalid_input(format!(
            "combined_fields flat scan got {} document columns and {} statistics masks for {} \
             target columns",
            doc_col_indices.len(),
            stats_masks.len(),
            columns.len()
        ))
        .into());
    }
    // A query that tokenizes to nothing matches nothing; mirrors the indexed path.
    // The scorer is still built and returned: an indexed sibling waits for it, and
    // over no terms it costs nothing.
    if terms.is_empty() || columns.is_empty() {
        // An empty fold, so whole-corpus mode reads no index statistics here either.
        // Over no terms both variants give the same scorer.
        let empty = FlatFieldStats::zeros(columns.len(), 0);
        let scorer = match base_scorer {
            Some(scorer) => scorer,
            None => Arc::new(
                build_combined_bm25_scorer(
                    columns,
                    tokens,
                    CombinedCorpusStats::for_flat_scan(&empty, flat_covers_whole_corpus),
                    metrics,
                )
                .boxed()
                .await?,
            ),
        };
        return Ok((
            Box::pin(RecordBatchStreamAdapter::new(
                FTS_SCHEMA.clone(),
                stream::empty::<DataFusionResult<RecordBatch>>(),
            )),
            scorer,
        ));
    }

    // Pre-await synchronous work: chunk-stream setup.
    let pre_await_start = std::time::Instant::now();
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
    if let Some(t) = &elapsed_compute {
        t.add_duration(pre_await_start.elapsed());
    }

    let weights: Vec<f32> = columns.iter().map(|column| column.weight).collect();
    // The per-column statistics fold happens inside the tokenization tasks, so the
    // masks only travel when a scorer still has to be built from them. The copy is
    // of the `Arc`s alone: a `RowAddrMask` wraps a roaring bitmap, so cloning one
    // per column per query would be a deep copy.
    let fold_masks = base_scorer
        .is_none()
        .then(|| Arc::new(stats_masks.to_vec()));
    let (blended, flat_stats) = tokenize_and_blend_multi(
        chunked,
        tokenizer,
        unique_tokens,
        Arc::new(doc_col_indices),
        Arc::new(weights),
        fold_masks,
        elapsed_compute.clone(),
    )
    .await?;

    // Time the scorer build and the scoring loop together.
    let post_await_start = std::time::Instant::now();
    let scorer = match base_scorer {
        Some(scorer) => scorer,
        None => {
            let flat =
                flat_stats.expect("statistics fold requested whenever no base scorer is preset");
            Arc::new(
                build_combined_bm25_scorer(
                    columns,
                    tokens,
                    CombinedCorpusStats::for_flat_scan(&flat, flat_covers_whole_corpus),
                    metrics,
                )
                .boxed()
                .await?,
            )
        }
    };
    // `rows x terms` synchronous work. Offload it so a large flat scan cannot
    // occupy a DataFusion worker past a stream drop or task cancellation, matching
    // how the indexed per-partition scoring loop is dispatched.
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
/// `(ROW_ID, SCORE)` in [`FTS_SCHEMA`].
///
/// Takes the chunks by value and drops each one as it is scored, so the blend and
/// the output arrays are never both fully resident.
///
/// `emit_mask` drops rows the query filtered out. It is applied here rather than to
/// `input` so that the statistics the scorer was built from still cover the corpus.
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
            // `And` spans the virtual field: a term counts when it occurs in at
            // least one target column, so the check runs on the blended `tf'`,
            // the same condition `combined_maxscore` applies through
            // `missing_term`.
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
