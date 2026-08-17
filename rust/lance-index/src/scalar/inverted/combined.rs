// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cross-field BM25F scoring (`combined_fields`).
//!
//! The target columns are treated as one virtual field so that term statistics
//! are blended across fields instead of scored independently (contrast the
//! field-centric `best_fields` fusion of a `MultiMatch` query). The blend rules
//! are Lucene's `CombinedFieldQuery`:
//!
//! ```text
//! tf'(t, d)      = Σ_f w_f · tf_f(t, d)              (weighted sum)
//! dl'(d)         = Σ_f w_f · dl_f(d)                 (weighted sum)
//! docFreq'(t)    = max_f docFreq_f(t)                (max)
//! docCount'      = max_f docCount_f                  (max)
//! sumTotalTermFreq' = Σ_f w_f · sumTotalTermFreq_f   (weighted sum)
//! avgdl'         = sumTotalTermFreq' / docCount'
//! score(t, d)    = idf'(t) · (k1 + 1) · tf' / (tf' + k1·(1 - b + b·dl'/avgdl'))
//! ```
//!
//! Scoring uses exact (non-quantized) document lengths and a shared tokenizer
//! across columns. Every candidate in the union of the query terms' postings is
//! scored; see [`combined_fields_search`].

mod cursor;
mod search;
mod stats;

use std::sync::Arc;

use lance_core::{Error, Result};

pub use search::combined_fields_search;
pub use stats::build_combined_bm25_scorer;

use super::index::InvertedIndex;
use super::query::Tokens;

/// One target column of a `combined_fields` query: its per-column weight and
/// the opened FTS segments (one per committed segment; usually a single one).
pub struct CombinedFieldColumn {
    /// Column name, used only for error messages.
    pub column: String,
    /// Per-column BM25F weight `w_f` (`>= 1`, validated at query construction).
    pub weight: f32,
    /// Opened inverted-index segments for this column.
    pub indices: Vec<Arc<InvertedIndex>>,
}

/// Deduplicate the query tokens into the unique terms of the virtual field,
/// preserving first-seen order. Duplicate terms collapse to one (mirrors the
/// per-column `load_posting_lists` dedup), so each term is scored once.
fn unique_terms(tokens: &Tokens) -> Vec<String> {
    let mut terms = Vec::with_capacity(tokens.len());
    let mut seen = std::collections::HashSet::new();
    for token in tokens {
        if seen.insert(token.as_str()) {
            terms.push(token.clone());
        }
    }
    terms
}

/// Reject a `combined_fields` query whose target columns do not share an
/// identical index/tokenizer configuration. BM25F is only well-defined when the
/// fields tokenize the same way, so mixing configurations is an error rather
/// than a silently wrong score. The error lists the offending columns.
pub fn validate_combined_tokenizers(columns: &[CombinedFieldColumn]) -> Result<()> {
    // A column with no index has no tokenizer to disagree with, so it is skipped.
    let mut indexed = columns
        .iter()
        .filter_map(|column| Some((column, column.indices.first()?)));
    let Some((reference, reference_index)) = indexed.next() else {
        return Ok(());
    };
    // Compare only the tokenization-affecting params: two columns may differ in
    // storage/layout knobs (e.g. `with_position`) yet still tokenize identically,
    // which is all BM25F requires.
    let offending: Vec<&str> = indexed
        .filter(|(_, index)| !reference_index.params().same_tokenization(index.params()))
        .map(|(column, _)| column.column.as_str())
        .collect();
    if !offending.is_empty() {
        return Err(Error::invalid_input(format!(
            "combined_fields requires every target column to share the same tokenizer/index \
             configuration; column(s) {:?} differ from column '{}'",
            offending, reference.column
        )));
    }
    Ok(())
}
