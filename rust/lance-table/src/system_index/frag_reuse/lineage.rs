// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The lineage a tagged fragment reuse entry records, and the one question
//! every write path asks of it: may a segment keep the provenance it stores?
//!
//! A translating segment's bitmap names the fragments it was built from,
//! retired by later rewrites; its rows reach the live destinations by
//! following the recorded transitions. Three places must agree on which of
//! that provenance an in-place column rewrite invalidates: the manifest
//! build pruning committed indices, the conflict resolver rebasing a new
//! index over committed writes, and the replay that validates staged
//! segments. They all walk this type (`withdraw_rewritten_coverage`).

use super::ledger::FragReuseLedger;
use super::metadata::is_tagged;
use crate::format::IndexMetadata;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::sync::Arc;

/// The tagged entry's recorded lineage plus, when it can be walked, the
/// transitions themselves.
#[derive(Debug)]
pub struct TaggedLineage {
    /// The entry's bitmap: every source and destination of every transition
    /// plus the coverage it inherited. `None` is unknown, not empty.
    fragments: Option<RoaringBitmap>,
    /// The decoded history; `None` only when a caller built the lineage
    /// without one (records this build cannot interpret count as none).
    /// Without it every answer is the conservative one.
    ledger: Option<Arc<FragReuseLedger>>,
}

impl TaggedLineage {
    /// The lineage of `entry`, walked through `ledger` when one is given.
    pub fn new(entry: &IndexMetadata, ledger: Option<Arc<FragReuseLedger>>) -> Self {
        Self {
            fragments: entry.fragment_bitmap.clone(),
            ledger: ledger.filter(|ledger| !ledger.has_unsupported_transitions()),
        }
    }

    /// The lineage of the tagged entry among `indices`, or `None` on a table
    /// without one. The history is `ledger` when the caller supplied it (the
    /// commit path decodes it once per attempt), else decoded from the
    /// entry's inline details. An undecodable history is an error, and so is
    /// an external one nobody read: a commit must not guess at the lineage.
    pub fn from_indices_with_ledger(
        indices: &[IndexMetadata],
        ledger: Option<Arc<FragReuseLedger>>,
    ) -> Result<Option<Self>> {
        let Some(entry) = indices.iter().find(|index| is_tagged(index)) else {
            return Ok(None);
        };
        let ledger = match ledger {
            Some(ledger) => ledger,
            None => {
                let details = entry.index_details.as_ref().ok_or_else(|| {
                    Error::invalid_input(
                        "the tagged fragment reuse entry carries no details; its history \
                         cannot be read",
                    )
                })?;
                FragReuseLedger::decode_inline(entry.index_version, details)?
                    .map(Arc::new)
                    .ok_or_else(|| {
                        Error::invalid_input(
                            "the tagged fragment reuse history is stored externally and was \
                             not supplied to the manifest build; commit through the dataset \
                             so the history is decoded first",
                        )
                    })?
            }
        };
        Ok(Some(Self::new(entry, Some(ledger))))
    }

    /// Whether the transitions can be walked (`destinations_of` answers).
    pub fn has_ledger(&self) -> bool {
        self.ledger.is_some()
    }

    /// Whether the walkable history mentions `fragment` as a source or a
    /// destination of some transition: the reader translates a segment
    /// whose bitmap names such a fragment instead of loading it as it is.
    /// `None` when the history cannot be walked.
    pub fn mentions(&self, fragment: u32) -> Option<bool> {
        self.ledger
            .as_ref()
            .map(|ledger| ledger.contains_fragment(fragment))
    }

    /// Whether `fragment` is on the recorded lineage. Unknown lineage is not
    /// empty lineage: without a bitmap every fragment may be on it.
    pub fn contains(&self, fragment: u32) -> bool {
        self.fragments
            .as_ref()
            .is_none_or(|fragments| fragments.contains(fragment))
    }

    /// The fragments the transition consuming `source` produced: one hop.
    /// `None` when the history cannot be walked; empty when no transition
    /// consumed `source`.
    pub fn immediate_destinations_of(&self, source: u32) -> Option<Vec<u32>> {
        let ledger = self.ledger.as_ref()?;
        let Some(position) = ledger.consumer(source) else {
            return Some(Vec::new());
        };
        ledger.transitions()[position]
            .destinations()
            .iter()
            .map(|digest| u32::try_from(digest.id).ok())
            .collect()
    }

    /// Every fragment the rows of `source` were moved into, following the
    /// transitions transitively (retired intermediates included, in walk
    /// order). `None` when the history cannot be walked.
    pub fn destinations_of(&self, source: u32) -> Option<Vec<u32>> {
        let ledger = self.ledger.as_ref()?;
        let mut reached = Vec::new();
        let mut seen = HashSet::new();
        let mut pending = vec![source];
        while let Some(fragment) = pending.pop() {
            let Some(position) = ledger.consumer(fragment) else {
                continue;
            };
            for digest in ledger.transitions()[position].destinations() {
                let destination = u32::try_from(digest.id).ok()?;
                if seen.insert(destination) {
                    reached.push(destination);
                    pending.push(destination);
                }
            }
        }
        Some(reached)
    }
}
