// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-term cross-column postings for `combined_fields`: one term's postings
//! merged across every target column into the shared row-id space, and the
//! loaded posting sources they are built from.

use std::collections::HashMap;
use std::sync::Arc;

use lance_select::RowAddrMask;

use super::super::documents::AddressKeyedDocuments;
use super::super::index::{PostingList, live_posting_rows};
use super::super::scorer::CombinedFieldsBM25Scorer;

/// One query term's postings, merged across every target column/partition into
/// the shared row-id space.
///
/// Entries are unique row ids sorted ascending, each carrying the blended term
/// frequency `tf'(t, d) = Σ_f w_f · freq_f(t, d)`.
pub(super) struct CombinedTermPostings {
    pub(super) idf: f32,
    pub(super) postings: Vec<(u64, f32)>,
}

impl CombinedTermPostings {
    /// `tf'` for `row_id`, or 0 when the term does not occur in the document.
    #[inline]
    pub(super) fn tf_prime(&self, row_id: u64) -> f32 {
        match self.postings.binary_search_by_key(&row_id, |(id, _)| *id) {
            Ok(idx) => self.postings[idx].1,
            Err(_) => 0.0,
        }
    }
}

/// A `(column, index, partition)` posting source loaded for one term.
pub(super) struct LoadedSource {
    pub(super) weight: f32,
    pub(super) docs: AddressKeyedDocuments,
    pub(super) is_legacy: bool,
    pub(super) posting: PostingList,
}

/// Merge every source's postings for `term` into the shared row-id space,
/// accumulating `tf'` in the canonical order.
pub(super) fn build_term_postings(
    term: &str,
    sources: Vec<LoadedSource>,
    mask: &Arc<RowAddrMask>,
    scorer: &CombinedFieldsBM25Scorer,
) -> CombinedTermPostings {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for source in &sources {
        for (row_id, freq) in live_posting_rows(&source.posting, &source.docs, source.is_legacy) {
            if !mask.selected(row_id) {
                continue;
            }
            *acc.entry(row_id).or_insert(0.0) += source.weight * freq as f32;
        }
    }
    let idf = scorer.query_weight(term);
    let mut postings: Vec<(u64, f32)> = acc.into_iter().collect();
    postings.sort_unstable_by_key(|(row_id, _)| *row_id);
    CombinedTermPostings { idf, postings }
}
