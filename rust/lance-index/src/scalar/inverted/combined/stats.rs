// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Corpus statistics for `combined_fields`: the cross-column BM25F scorer
//! build.

use std::collections::HashMap;

use lance_core::Result;

use super::super::query::Tokens;
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::{CombinedFieldColumn, unique_terms};
use crate::metrics::MetricsCollector;

/// Fold the target columns' segment statistics into a single [`CombinedFieldsBM25Scorer`].
///
/// Generalizes [`build_global_bm25_scorer`](super::super::build_global_bm25_scorer)
/// (which folds segments of one column) to fold across columns too, with
/// per-column weights and the BM25F blend:
/// `docCount'`/`docFreq'` take the max across columns while `sumTotalTermFreq'`
/// is the weighted sum. The query terms are deduplicated the same way the scan
/// deduplicates them, so the resulting per-term `docFreq'` covers exactly the
/// terms [`combined_fields_search`](super::combined_fields_search) scores.
///
/// `metrics`, when provided, receives the per-token posting-metadata cache
/// lookups this fold triggers, exactly as
/// [`build_global_bm25_scorer`](super::super::build_global_bm25_scorer)
/// reports them on the single-column path. Without it a cold cross-field query
/// undercounts `index_cache_misses` by one lookup per (term, partition, column).
pub async fn build_combined_bm25_scorer(
    columns: &[CombinedFieldColumn],
    tokens: &Tokens,
    metrics: Option<&dyn MetricsCollector>,
) -> Result<CombinedFieldsBM25Scorer> {
    let terms = unique_terms(tokens);
    let mut doc_count = 0usize;
    let mut sum_total_term_freq = 0f64;
    let mut doc_freq: HashMap<String, usize> = terms.iter().map(|t| (t.clone(), 0)).collect();

    for column in columns {
        let mut column_num_docs = 0usize;
        let mut column_total_tokens = 0u64;
        let mut column_doc_freq = vec![0usize; terms.len()];
        {
            for index in &column.indices {
                // Row granularity, not document granularity: `combined_fields_search`
                // blends every posting a row owns into one `tf'` and every document
                // length it owns into one `dl'`, and released V1/V2 indexes may hold
                // one document per list element. See
                // [`InvertedIndex::bm25_row_stats_for_terms`].
                let (total_tokens, num_docs, token_docs) =
                    index.bm25_row_stats_for_terms(&terms, metrics).await?;
                column_total_tokens += total_tokens;
                column_num_docs += num_docs;
                for (slot, df) in token_docs.into_iter().enumerate() {
                    column_doc_freq[slot] += df;
                }
            }
        }

        doc_count = doc_count.max(column_num_docs);
        sum_total_term_freq += column.weight as f64 * column_total_tokens as f64;
        for (term, df) in terms.iter().zip(column_doc_freq) {
            let entry = doc_freq
                .get_mut(term)
                .expect("doc_freq initialized for every term");
            *entry = (*entry).max(df);
        }
    }

    let avg_doc_length = if doc_count > 0 {
        (sum_total_term_freq / doc_count as f64) as f32
    } else {
        0.0
    };
    Ok(CombinedFieldsBM25Scorer::new(
        doc_count,
        avg_doc_length,
        doc_freq,
    ))
}
