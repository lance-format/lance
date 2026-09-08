// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-term cross-column cursors for `combined_fields`: the [`TermCursor`]
//! abstraction, its full-read implementation, and the loaded posting sources it
//! is built from.

use std::collections::HashMap;
use std::sync::Arc;

use lance_select::RowAddrMask;

use super::super::documents::AddressKeyedDocuments;
use super::super::index::{PostingList, live_posting_rows};
use super::super::scorer::CombinedFieldsBM25Scorer;
use super::maxscore::term_upper_bound;

/// The MAXSCORE loop's view of one query term. [`MaterializedTerm`] implements it
/// by reading every posting up front; the trait keeps
/// [`combined_maxscore`](super::maxscore::combined_maxscore) independent of how a
/// `tf'` is fetched, so a lazier cursor can be substituted without touching the
/// algorithm.
///
/// A term exposes a merged cross-column cursor over the shared row-id space:
/// [`head`](Self::head) is the smallest not-yet-consumed row id, advanced by
/// the essential terms during discovery; [`probe`](Self::probe) reads `tf'` at
/// an arbitrary (non-decreasing) target for the non-essential terms.
pub(super) trait TermCursor {
    /// The constant per-term ceiling `idf · MAX_DOC_WEIGHT` (clamped, see
    /// [`term_upper_bound`]).
    fn upper_bound(&self) -> f32;
    /// The blended IDF `idf'(t)`.
    fn idf(&self) -> f32;
    /// The smallest not-yet-consumed row id across the term's columns, or
    /// `None` when the term is exhausted.
    fn head(&self) -> Option<u64>;
    /// `tf'` at [`head`](Self::head) (the weighted sum across the columns that
    /// carry the head row id). Only meaningful while `head()` is `Some`.
    fn head_tf(&self) -> f32;
    /// Consume `row_id`: advance every column cursor currently sitting on it.
    /// Called only for `row_id == head()`.
    fn consume(&mut self, row_id: u64);
    /// `tf'` at `target`, or 0 when the term is absent there. `target` never
    /// decreases across calls to one cursor, so the lazy path can skip blocks.
    fn probe(&mut self, target: u64) -> f32;
}

/// One query term's postings, merged across every target column/partition into
/// the shared row-id space.
///
/// Entries are unique row ids sorted ascending, each carrying the blended term
/// frequency `tf'(t, d) = Σ_f w_f · freq_f(t, d)`.
pub(super) struct CombinedTermPostings {
    pub(super) idf: f32,
    pub(super) upper_bound: f32,
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

/// [`TermCursor`] over a [`CombinedTermPostings`] with an owned scan cursor.
pub(super) struct MaterializedTerm {
    pub(super) postings: CombinedTermPostings,
    cursor: usize,
}

impl MaterializedTerm {
    pub(super) fn new(postings: CombinedTermPostings) -> Self {
        Self {
            postings,
            cursor: 0,
        }
    }
}

impl TermCursor for MaterializedTerm {
    #[inline]
    fn upper_bound(&self) -> f32 {
        self.postings.upper_bound
    }

    #[inline]
    fn idf(&self) -> f32 {
        self.postings.idf
    }

    #[inline]
    fn head(&self) -> Option<u64> {
        self.postings.postings.get(self.cursor).map(|(id, _)| *id)
    }

    #[inline]
    fn head_tf(&self) -> f32 {
        self.postings
            .postings
            .get(self.cursor)
            .map(|(_, tf)| *tf)
            .unwrap_or(0.0)
    }

    #[inline]
    fn consume(&mut self, row_id: u64) {
        if self.head() == Some(row_id) {
            self.cursor += 1;
        }
    }

    #[inline]
    fn probe(&mut self, target: u64) -> f32 {
        self.postings.tf_prime(target)
    }
}

/// A `(column, index, partition)` posting source loaded for one term.
pub(super) struct LoadedSource {
    pub(super) weight: f32,
    pub(super) docs: AddressKeyedDocuments,
    pub(super) is_legacy: bool,
    pub(super) posting: PostingList,
}

/// Build the full-read cursor for `term` by merging every source's postings into
/// the shared row-id space, accumulating `tf'` in the canonical order.
pub(super) fn build_materialized_term(
    term: &str,
    sources: Vec<LoadedSource>,
    mask: &Arc<RowAddrMask>,
    scorer: &CombinedFieldsBM25Scorer,
) -> MaterializedTerm {
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
    MaterializedTerm::new(CombinedTermPostings {
        idf,
        upper_bound: term_upper_bound(idf),
        postings,
    })
}

#[cfg(test)]
mod tests {
    use super::super::maxscore::combined_maxscore;
    use super::super::testing::{compressed_list, modern_identity_docs};
    use super::*;
    use lance_core::utils::address::RowAddress;
    use lance_select::RowAddrTreeMap;
    use std::cmp::Reverse;

    #[tokio::test]
    async fn test_build_materialized_term_merges_legacy_and_compressed() {
        // Fallback data path: a legacy (Plain, row-id-keyed, list-multiplicity)
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
        let term = build_materialized_term("t", sources, &mask, &scorer);

        // row 10: 2*1 = 2; row 20: 2*2 + 2*3 + 1*5 = 15; row 30: 2*1 = 2.
        // Row 42 is masked out entirely.
        assert_eq!(
            term.postings.postings,
            vec![(10, 2.0), (20, 15.0), (30, 2.0)]
        );
    }

    #[tokio::test]
    async fn test_build_materialized_term_skips_tombstoned_addresses() {
        // A remapped partition keeps a deleted document's DocId slot so the
        // posting lists stay aligned and answers `TOMBSTONE_ROW` for its address.
        // Nothing else stops that address: a default mask is an empty block list,
        // which selects it, and `doc_length_at(TOMBSTONE_ROW) == 0` would give it
        // the largest `doc_weight` there is. It stays out of the result today only
        // because `TOMBSTONE_ROW == u64::MAX` collides with `combined_maxscore`'s
        // exhausted-cursor sentinel, so it must be dropped at the source.
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
            docs: docs.clone(),
            is_legacy: false,
            posting: PostingList::Compressed(compressed_list(&[
                (10, 1),
                (DEAD_ROW as u32, 7),
                (30, 3),
            ])),
        }];
        let term =
            build_materialized_term("t", sources, &Arc::new(RowAddrMask::default()), &scorer);
        assert_eq!(
            term.postings.postings,
            vec![(10, 2.0), (30, 6.0)],
            "the tombstoned address must never be accumulated, and the live rows \
             must keep their exact contributions"
        );

        // Nor may it be collected: the two live rows are the whole top-k, scored
        // exactly as if the dead slot's posting did not exist.
        //
        // `test_scorer` carries no document frequencies, so `query_weight` is 0 for
        // every term and both rows score 0. The order is therefore the tiebreak,
        // ascending by row id.
        let mut cursors = vec![term];
        let dl_prime = |row_id: u64| -> f32 { 2.0 * docs.doc_length_at(row_id) as f32 };
        let (top, _) = combined_maxscore(&mut cursors, dl_prime, 10, false, &scorer);
        let hits: Vec<(u64, u32)> = top
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(doc)| (doc.row_id.0, doc.score.0.to_bits()))
            .collect();
        let expected = |tf: f32| (scorer.query_weight("t") * scorer.doc_weight(tf, 8.0)).to_bits();
        assert_eq!(hits, vec![(10, expected(2.0)), (30, expected(6.0))]);
    }
}
