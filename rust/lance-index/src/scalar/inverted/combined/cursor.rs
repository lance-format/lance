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

#[cfg(test)]
mod tests {
    use super::super::testing::{compressed_list, modern_identity_docs};
    use super::*;
    use lance_core::utils::address::RowAddress;
    use lance_select::RowAddrTreeMap;

    #[tokio::test]
    async fn test_build_term_postings_merges_legacy_and_compressed() {
        // A legacy (Plain, row-id-keyed, list-multiplicity)
        // source and a compressed source merge into one ordered `tf'` stream,
        // masked rows dropped and contributions summed in column order.
        use super::super::super::index::PlainPostingList;
        use arrow::buffer::ScalarBuffer;

        let scorer = CombinedFieldsBM25Scorer::new(1000, 12.0, HashMap::new());
        // Legacy column (weight 2): the posting keys directly on row ids; row 20
        // appears twice (list multiplicity), so its contributions sum.
        let legacy = PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(vec![10u64, 20, 20, 30]),
            ScalarBuffer::from(vec![1.0f32, 2.0, 3.0, 1.0]),
            Some(0.0),
            None,
        ));
        // Compressed column (weight 1): doc id 20 maps through the modern
        // projection (the only representation a compressed posting is loaded
        // alongside) to row 20; row 42 is blocked by the mask below.
        let compressed = PostingList::Compressed(compressed_list(&[(20, 5), (42, 7)]));
        let docs = modern_identity_docs(&vec![1u32; 64], &[]).await;
        let sources = vec![
            LoadedSource {
                weight: 2.0,
                docs: docs.clone(),
                is_legacy: true,
                posting: legacy,
            },
            LoadedSource {
                weight: 1.0,
                docs,
                is_legacy: false,
                posting: compressed,
            },
        ];
        let mask = Arc::new(RowAddrMask::all_rows().also_block(RowAddrTreeMap::from_iter([42u64])));
        let term = build_term_postings("t", sources, &mask, &scorer);

        // row 10: 2*1 = 2; row 20: 2*2 + 2*3 + 1*5 = 15; row 30: 2*1 = 2.
        // Row 42 is masked out entirely.
        assert_eq!(term.postings, vec![(10, 2.0), (20, 15.0), (30, 2.0)]);
    }

    #[tokio::test]
    async fn test_build_term_postings_skips_tombstoned_addresses() {
        // A remapped partition keeps a deleted document's DocId slot so the
        // posting lists stay aligned and answers `TOMBSTONE_ROW` for its address.
        // Nothing else stops that address: a default mask is an empty block list,
        // which selects it, and `doc_length_at(TOMBSTONE_ROW) == 0` would give it
        // the largest `doc_weight` there is, so it must be dropped at the source.
        const DEAD_ROW: u64 = 20;
        let scorer = CombinedFieldsBM25Scorer::new(1000, 12.0, HashMap::new());
        let docs = modern_identity_docs(&[4u32; 40], &[DEAD_ROW]).await;
        assert_eq!(
            docs.row_address(DEAD_ROW as u32),
            RowAddress::TOMBSTONE_ROW,
            "the deleted document must keep its slot as a tombstone"
        );
        assert_eq!(docs.doc_length_at(RowAddress::TOMBSTONE_ROW), 0);
        let sources = vec![LoadedSource {
            weight: 2.0,
            docs,
            is_legacy: false,
            posting: PostingList::Compressed(compressed_list(&[
                (10, 1),
                (DEAD_ROW as u32, 7),
                (30, 3),
            ])),
        }];
        let term = build_term_postings("t", sources, &Arc::new(RowAddrMask::default()), &scorer);
        assert_eq!(
            term.postings,
            vec![(10, 2.0), (30, 6.0)],
            "the tombstoned address must never be accumulated, and the live rows \
             must keep their exact contributions"
        );
    }
}
