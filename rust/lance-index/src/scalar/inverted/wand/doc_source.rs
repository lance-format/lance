// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Document-side view a `Wand` searches: lengths, visibility and candidate keys.

use std::fmt::Debug;

use lance_core::utils::address::RowAddress;
use lance_select::RowAddrMask;

use crate::scalar::inverted::documents::{DocId, DocLengths, DocVisibility};
use crate::scalar::inverted::{DocInfo, DocSet};

#[derive(Debug)]
pub struct DocCandidate<C> {
    pub document: C,
    /// The document key used by the posting lists: doc_id for compressed
    /// postings, row_id for legacy plain postings.
    pub posting_doc_id: u64,
    /// (term_index, freq)
    pub freqs: Vec<(u32, u32)>,
    pub doc_length: u32,
}

/// Document-side contract consumed by WAND.  Implementations fix candidate
/// identity and visibility before the CPU executor starts.
pub(in crate::scalar::inverted) type FlatDocuments<'a> =
    (usize, Box<dyn Iterator<Item = (u64, u64)> + 'a>);

pub(in crate::scalar::inverted) trait WandDocuments {
    type Candidate: Copy + Debug;

    fn len(&self) -> usize;
    /// Conservative upper bound on documents that can survive visibility.
    /// Used only to estimate scorer cost; iteration remains authoritative.
    fn visible_cost_upper_bound(&self) -> usize {
        self.len()
    }
    /// Whether this partition can hold a document that is not visible --
    /// deleted, or outside a selection. When it cannot, asking about
    /// visibility per document can only ever answer "visible", so callers may
    /// skip the question. Conservative default: assume it can.
    fn any_invisible(&self) -> bool {
        true
    }
    fn scoring_norms(&self) -> Option<&[u8]>;
    fn scoring_num_tokens(&self, doc_id: u32) -> u32;
    fn doc_length(&self, doc: &DocInfo) -> u32;
    fn document_key(&self, doc: &DocInfo) -> Option<u64>;
    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64>;
    fn candidate_from_key(&self, key: u64) -> Self::Candidate;
    fn flat_documents(&self) -> Option<FlatDocuments<'_>>;
    fn flat_doc_length(&self, doc_id: u64, document_key: u64, compressed: bool) -> u32;
}

pub(in crate::scalar::inverted) trait ModernVisibility {
    fn selected(&self, doc_id: DocId) -> bool;
    fn len(&self, total_docs: usize) -> usize;
    /// Whether some document of the partition can be invisible. `len` cannot
    /// answer this: `DocVisibility::Filtered` -- a deletion vector too large
    /// to materialize -- reports the full document count even though it hides
    /// rows, because counting the survivors is not worth the work up front.
    fn any_invisible(&self) -> bool {
        true
    }
    fn iter(&self) -> Option<Box<dyn Iterator<Item = DocId> + '_>>;
}

pub(in crate::scalar::inverted) struct AllModernDocuments;

impl ModernVisibility for AllModernDocuments {
    #[inline]
    fn selected(&self, _doc_id: DocId) -> bool {
        true
    }

    fn len(&self, total_docs: usize) -> usize {
        total_docs
    }

    fn any_invisible(&self) -> bool {
        false
    }

    fn iter(&self) -> Option<Box<dyn Iterator<Item = DocId> + '_>> {
        None
    }
}

impl ModernVisibility for &DocVisibility {
    #[inline]
    fn selected(&self, doc_id: DocId) -> bool {
        DocVisibility::selected(self, doc_id)
    }

    fn len(&self, total_docs: usize) -> usize {
        DocVisibility::len(self, total_docs)
    }

    fn any_invisible(&self) -> bool {
        !DocVisibility::is_all(self)
    }

    fn iter(&self) -> Option<Box<dyn Iterator<Item = DocId> + '_>> {
        DocVisibility::iter(self)
            .map(|doc_ids| Box::new(doc_ids) as Box<dyn Iterator<Item = DocId>>)
    }
}

pub(in crate::scalar::inverted) struct ModernWandDocuments<'a, V> {
    lengths: &'a DocLengths,
    visibility: V,
}

impl<'a> ModernWandDocuments<'a, AllModernDocuments> {
    pub(crate) fn all(lengths: &'a DocLengths) -> Self {
        Self {
            lengths,
            visibility: AllModernDocuments,
        }
    }
}

impl<'a> ModernWandDocuments<'a, &'a DocVisibility> {
    pub(crate) fn filtered(lengths: &'a DocLengths, visibility: &'a DocVisibility) -> Self {
        Self {
            lengths,
            visibility,
        }
    }
}

impl<V: ModernVisibility> WandDocuments for ModernWandDocuments<'_, V> {
    type Candidate = DocId;

    fn len(&self) -> usize {
        self.lengths.len()
    }

    fn visible_cost_upper_bound(&self) -> usize {
        self.visibility.len(self.lengths.len())
    }

    fn any_invisible(&self) -> bool {
        self.visibility.any_invisible()
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        self.lengths.scoring_norms()
    }

    fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        self.lengths.scoring(DocId::new(doc_id))
    }

    fn doc_length(&self, doc: &DocInfo) -> u32 {
        match doc {
            DocInfo::Raw(doc) => self.scoring_num_tokens(doc.doc_id),
            DocInfo::Located(_) => unreachable!("modern posting lists contain dense DocIds"),
        }
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        match doc {
            DocInfo::Raw(doc) if self.visibility.selected(DocId::new(doc.doc_id)) => {
                Some(u64::from(doc.doc_id))
            }
            DocInfo::Raw(_) => None,
            DocInfo::Located(_) => unreachable!("modern posting lists contain dense DocIds"),
        }
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        self.visibility
            .selected(DocId::new(doc_id))
            .then_some(u64::from(doc_id))
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        DocId::new(key as u32)
    }

    fn flat_documents(&self) -> Option<(usize, Box<dyn Iterator<Item = (u64, u64)> + '_>)> {
        self.visibility.iter().map(|doc_ids| {
            let len = self.visibility.len(self.lengths.len());
            let docs = doc_ids.map(|doc_id| {
                let value = u64::from(doc_id.get());
                (value, value)
            });
            (len, Box::new(docs) as Box<dyn Iterator<Item = (u64, u64)>>)
        })
    }

    fn flat_doc_length(&self, doc_id: u64, _document_key: u64, _compressed: bool) -> u32 {
        self.scoring_num_tokens(doc_id as u32)
    }
}

pub(in crate::scalar::inverted) struct LegacyWandDocuments<'a> {
    docs: &'a DocSet,
    mask: &'a RowAddrMask,
}

impl<'a> LegacyWandDocuments<'a> {
    pub(crate) fn new(docs: &'a DocSet, mask: &'a RowAddrMask) -> Self {
        Self { docs, mask }
    }
}

impl WandDocuments for LegacyWandDocuments<'_> {
    type Candidate = u64;

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        self.docs.scoring_norms()
    }

    fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        self.docs.scoring_num_tokens(doc_id)
    }

    fn doc_length(&self, doc: &DocInfo) -> u32 {
        match doc {
            DocInfo::Raw(doc) => self.docs.scoring_num_tokens(doc.doc_id),
            DocInfo::Located(doc) => self.docs.num_tokens_by_row_id(doc.row_id),
        }
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        let row_id = match doc {
            DocInfo::Raw(doc) => self.docs.row_id(doc.doc_id),
            DocInfo::Located(doc) => doc.row_id,
        };
        (row_id != RowAddress::TOMBSTONE_ROW && self.mask.selected(row_id)).then_some(row_id)
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        let row_id = self.docs.row_id(doc_id);
        (row_id != RowAddress::TOMBSTONE_ROW && self.mask.selected(row_id)).then_some(row_id)
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        key
    }

    fn flat_documents(&self) -> Option<(usize, Box<dyn Iterator<Item = (u64, u64)> + '_>)> {
        let count = self.mask.max_len()? as usize;
        let row_ids = self.mask.iter_addrs()?;
        let docs = row_ids.flat_map(|row_addr| {
            let row_id: u64 = row_addr.into();
            self.docs
                .doc_ids(row_id)
                .map(move |doc_id| (doc_id, row_id))
        });
        Some((count, Box::new(docs)))
    }

    fn flat_doc_length(&self, doc_id: u64, document_key: u64, compressed: bool) -> u32 {
        if compressed {
            self.docs.scoring_num_tokens(doc_id as u32)
        } else {
            self.docs.num_tokens_by_row_id(document_key)
        }
    }
}

// Most unit tests exercise WAND in isolation with an in-memory complete
// DocSet.  Production code must select one of the explicit modern/legacy
// adapters above, so this convenience implementation is test-only.
#[cfg(test)]
impl WandDocuments for DocSet {
    type Candidate = u64;

    fn len(&self) -> usize {
        self.len()
    }

    fn scoring_norms(&self) -> Option<&[u8]> {
        self.scoring_norms()
    }

    fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        self.scoring_num_tokens(doc_id)
    }

    fn doc_length(&self, doc: &DocInfo) -> u32 {
        match doc {
            DocInfo::Raw(doc) => self.scoring_num_tokens(doc.doc_id),
            DocInfo::Located(doc) => self.num_tokens_by_row_id(doc.row_id),
        }
    }

    fn document_key(&self, doc: &DocInfo) -> Option<u64> {
        Some(match doc {
            DocInfo::Raw(doc) if self.has_row_ids() => self.row_id(doc.doc_id),
            DocInfo::Raw(doc) => u64::from(doc.doc_id),
            DocInfo::Located(doc) => doc.row_id,
        })
    }

    fn document_key_for_doc_id(&self, doc_id: u32) -> Option<u64> {
        Some(if self.has_row_ids() {
            self.row_id(doc_id)
        } else {
            u64::from(doc_id)
        })
    }

    fn candidate_from_key(&self, key: u64) -> Self::Candidate {
        key
    }

    fn flat_documents(&self) -> Option<(usize, Box<dyn Iterator<Item = (u64, u64)> + '_>)> {
        None
    }

    fn flat_doc_length(&self, doc_id: u64, document_key: u64, compressed: bool) -> u32 {
        if compressed {
            self.scoring_num_tokens(doc_id as u32)
        } else {
            self.num_tokens_by_row_id(document_key)
        }
    }
}
