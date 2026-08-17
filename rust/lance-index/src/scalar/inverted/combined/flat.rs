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

#[cfg(test)]
mod tests {
    use super::super::super::tokenizer::InvertedIndexParams;
    use super::super::super::tokenizer::document_tokenizer::DocType;
    use super::super::testing::{flat_columns, flat_input, flat_scores};
    use super::*;
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;

    /// Golden `(row_id, score)` for the flat BM25F scan, compared bit-for-bit.
    ///
    /// The blend `dl' = Σ_f w_f·dl_f` and `tf'_t = Σ_f w_f·tf_f,t` accumulates in
    /// f32, which is not associative, and the corpus statistics it feeds are
    /// derived from the same rows. Any change to how or where the per-column
    /// contributions are summed has to leave these exact bits alone, so they are
    /// pinned here rather than compared with a tolerance.
    ///
    /// The fixture exercises the parts that a plausible rewrite gets wrong:
    /// non-unit and unequal weights, a row empty in one column but not the other,
    /// a null, a row empty everywhere (dropped), and a `stats_masks` entry that
    /// keeps one row out of one column's corpus totals.
    #[tokio::test]
    async fn test_flat_combined_fields_golden_scores() {
        let row_ids: Vec<u64> = (0..7).collect();
        let docs = vec![
            vec![
                Some("cat"),
                Some("dog cat"),
                None,
                Some("bird"),
                Some("cat cat dog"),
                Some(""),
                Some("cat dog bird"),
            ],
            vec![
                Some("dog"),
                None,
                Some("cat dog dog"),
                Some("fish"),
                Some("cat"),
                Some(""),
                Some("bird bird"),
            ],
        ];
        let weights = [1.0f32, 2.5];
        let columns = flat_columns(&weights);
        let tokens = Tokens::new(vec!["cat".to_string(), "dog".to_string()], DocType::Text);
        let tokenizer = InvertedIndexParams::default().build().unwrap();
        // Column 1 must not fold row 4 into its corpus totals (its index already
        // holds that row); column 0 folds everything.
        let stats_masks = vec![
            Arc::new(RowAddrMask::all_rows()),
            Arc::new(RowAddrMask::all_rows().also_block(RowAddrTreeMap::from_iter([4u64]))),
        ];

        let stream = flat_combined_fields_search_stream(
            flat_input(&row_ids, &docs, 1),
            &columns,
            vec![1, 2],
            &stats_masks,
            /*emit_mask=*/ None,
            /*flat_covers_whole_corpus=*/ false,
            &tokens,
            tokenizer,
            Operator::Or,
            None,
            3,
            None,
            None,
        )
        .await
        .unwrap()
        .0;
        let (batch_sizes, scored) = flat_scores(stream).await;

        // Row 5 is empty in both columns, so it never reaches the scorer, and row 3
        // ("bird" / "fish") carries no query term, so it scores 0 and is dropped.
        // `target_batch_size` of 3 splits the 5 survivors into 3 + 2.
        assert_eq!(batch_sizes, vec![3, 2]);
        assert_eq!(
            scored,
            vec![
                (0, 0x3f9b_c3cf),
                (1, 0x3f8f_0e96),
                (2, 0x3fa6_8e68),
                (4, 0x3f84_f2a6),
                (6, 0x3f32_7287),
            ]
        );
    }

    /// The retained per-row payload must not grow with the number of target
    /// columns. That independence is the whole point of blending during
    /// tokenization: a `combined_fields` query over a wide column list routes the
    /// entire dataset down this path as soon as one column's index is stale, and
    /// the accumulation lives until the corpus statistics are complete.
    ///
    /// The input arrives as several batches, so this also covers the per-batch
    /// statistics fold and that the retained rows stay in scan order across chunks.
    #[tokio::test]
    async fn test_flat_combined_fields_retention_is_independent_of_column_count() {
        let row_ids: Vec<u64> = (0..64).collect();
        let column = vec![Some("cat dog bird"); 64];
        let tokens = Arc::new(Tokens::new(
            vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
            DocType::Text,
        ));
        let num_terms = tokens.len();

        let mut footprints = Vec::new();
        for num_columns in 1..=4usize {
            let docs = vec![column.clone(); num_columns];
            let (chunks, stats) = tokenize_and_blend_multi(
                flat_input(&row_ids, &docs, 5),
                InvertedIndexParams::default().build().unwrap(),
                tokens.clone(),
                Arc::new((1..=num_columns).collect()),
                Arc::new(vec![1.0; num_columns]),
                Some(Arc::new(
                    (0..num_columns)
                        .map(|_| Arc::new(RowAddrMask::all_rows()))
                        .collect(),
                )),
                None,
            )
            .await
            .unwrap();

            let retained: usize = chunks
                .iter()
                .map(|chunk| {
                    chunk.row_ids.len() * size_of::<u64>()
                        + (chunk.doc_lengths.len() + chunk.term_freqs.len()) * size_of::<f32>()
                })
                .sum();
            let retained_row_ids: Vec<u64> = chunks
                .iter()
                .flat_map(|chunk| &chunk.row_ids)
                .copied()
                .collect();
            assert!(chunks.len() > 1, "expected several chunks to fold");
            assert_eq!(retained_row_ids, row_ids);
            let rows = retained_row_ids.len();
            // 8 bytes of row id plus `dl'` and one `tf'` per term, all f32.
            assert_eq!(retained, rows * (8 + 4 * (1 + num_terms)));
            footprints.push(retained);

            // Every column sees the same three-token doc, so the statistics scale
            // with the column count while the retained payload does not.
            let stats = stats.unwrap();
            assert_eq!(stats.doc_counts, vec![rows; num_columns]);
            assert_eq!(stats.total_tokens, vec![3 * rows as u64; num_columns]);
            assert_eq!(stats.doc_freqs, vec![vec![rows; num_terms]; num_columns]);
        }
        assert!(footprints.windows(2).all(|pair| pair[0] == pair[1]));
    }

    /// A flat scan can legitimately see no batches at all (every scanned fragment
    /// empty, or a prefilter that keeps nothing). The statistics fold still owes the
    /// caller a scorer, so it reports all-zero totals rather than nothing.
    #[tokio::test]
    async fn test_flat_combined_fields_empty_scan_still_builds_a_scorer() {
        let docs = vec![Vec::<Option<&str>>::new(), Vec::new()];
        let stream = flat_combined_fields_search_stream(
            flat_input(&[], &docs, 1),
            &flat_columns(&[1.0, 2.5]),
            vec![1, 2],
            &[
                Arc::new(RowAddrMask::all_rows()),
                Arc::new(RowAddrMask::all_rows()),
            ],
            /*emit_mask=*/ None,
            /*flat_covers_whole_corpus=*/ false,
            &Tokens::new(vec!["cat".to_string()], DocType::Text),
            InvertedIndexParams::default().build().unwrap(),
            Operator::Or,
            None,
            16,
            None,
            None,
        )
        .await
        .unwrap()
        .0;

        let (batch_sizes, scored) = flat_scores(stream).await;
        assert!(batch_sizes.is_empty());
        assert!(scored.is_empty());
    }

    /// A caller that hands over the wrong number of document columns or statistics
    /// masks has a bug, and it must surface whatever the query tokenizes to. The
    /// empty-token case is the one that used to slip through, because matching
    /// nothing short circuits before any per-column work happens.
    #[rstest]
    #[case::empty_query(Vec::new())]
    #[case::one_term(vec!["cat".to_string()])]
    #[tokio::test]
    async fn test_flat_combined_fields_rejects_mismatched_arity(#[case] tokens: Vec<String>) {
        let docs = vec![Vec::<Option<&str>>::new(), Vec::new()];
        let result = flat_combined_fields_search_stream(
            flat_input(&[], &docs, 1),
            &flat_columns(&[1.0, 2.5]),
            // Two target columns, but one document column and one mask.
            vec![1],
            &[Arc::new(RowAddrMask::all_rows())],
            /*emit_mask=*/ None,
            /*flat_covers_whole_corpus=*/ false,
            &Tokens::new(tokens, DocType::Text),
            InvertedIndexParams::default().build().unwrap(),
            Operator::Or,
            None,
            16,
            None,
            None,
        )
        .await;

        let Err(error) = result else {
            panic!("mismatched arity must be rejected");
        };
        assert!(
            error
                .to_string()
                .contains("got 1 document columns and 1 statistics masks for 2 target columns"),
            "unexpected error: {error}",
        );
    }
}
