// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Composing a tagged history into the translation an index rewrite applies
//! to one segment's files, and the bitmap the rewritten segment publishes.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_select::remapper::{RemapperBacking, TranslateFuture};
use roaring::RoaringBitmap;

use super::super::FriImpl;
use crate::format::pb::fragment_reuse_index_details as pb;
use crate::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use crate::system_index::frag_reuse::mapping::{
    MappingRef, MappingSource, translate_stable_partition,
};
use crate::system_index::frag_reuse::plan::{
    Composition, RewriteContext, RewriteHop, RewritePath, SegmentSummary, TranslationPlan,
};

pub(super) fn compose(
    history: Arc<dyn FriImpl>,
    context: &RewriteContext,
    segment: &SegmentSummary,
) -> Result<Composition> {
    let ledger = history.records().ledger.clone();
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the tagged FRI history carries transitions this client cannot interpret; \
             upgrade to a newer version of Lance before remapping",
        ));
    }
    let stored = segment.fragment_bitmap.as_ref().ok_or_else(|| {
        Error::invalid_input(format!(
            "segment {} records no fragment bitmap, so its lineage through the history \
             cannot be determined",
            segment.uuid
        ))
    })?;
    // A segment that predates a lifted v0 deferred compaction still holds
    // the group's source addresses behind a bitmap v0 swapped to the
    // destinations; plan from the fragments the file really addresses.
    let provenance = effective_provenance(&ledger, stored, segment.dataset_version);
    Ok(
        match walk(
            &ledger,
            &provenance,
            context.live_fragments,
            context.lineage,
        ) {
            Walk::Identity => Composition::CaughtUp,
            Walk::Blocked { fragment } => Composition::Blocked { fragment },
            Walk::Remap {
                hops,
                coverage,
                admitted,
            } => Composition::Rewrite(TranslationPlan::rewrite(
                history,
                coverage,
                RewritePath {
                    hops,
                    admitted,
                    fragment_rows: known_fragment_rows(&ledger, context.fragment_rows),
                },
            )),
        },
    )
}

/// The outcome of walking a segment's provenance forward through the
/// ledger's consumer edges.
#[derive(Debug)]
enum Walk {
    /// No transition consumes any covered fragment.
    Identity,
    /// Every applied hop was fully covered: the bitmap swaps to `coverage`.
    /// A fully covered stable partition takes this path too: the segment
    /// owns every source, so the destinations are wholly its rows and the
    /// restamp arithmetic is exactly the compaction one.
    Remap {
        hops: Vec<RewriteHop>,
        coverage: RoaringBitmap,
        /// The stored provenance plus the destinations of every applied hop,
        /// intermediates included. A merged segment keeps its sources'
        /// provenance while its pages hold the addresses the translating
        /// loader produced, so its rows can be anywhere on that path. A
        /// withdrawn source's transition is never applied, so nothing past it
        /// is admitted.
        admitted: RoaringBitmap,
    },
    /// See [`Composition::Blocked`].
    Blocked { fragment: u32 },
}

fn ids(digests: &[pb::FragmentDigest]) -> RoaringBitmap {
    digests.iter().map(|digest| digest.id as u32).collect()
}

/// Walk FORWARD from the segment's provenance along consumer edges. A hop
/// with `sources ⊆ coverage` (compaction or stable partition) applies:
/// coverage evolves to the hop's destinations. A partial-overlap compaction
/// takes the v0 straddle fallback, but only before any stable-partition hop
/// has been applied: that coverage is dropped from the bitmap (sources
/// removed, destinations NOT added) and the walk continues; the dropped rows
/// are served by scan. A partial-overlap stable partition cannot be restamped
/// and blocks the whole plan. Once a stable-partition hop has been applied,
/// ANY later partial overlap blocks as well: the fallback is only sound while
/// the segment's stored addresses are untouched. A restamped file would carry
/// addresses inside a fragment the later hop consumes while its bitmap no
/// longer names the partition's sources, so nothing would pin the partition
/// and a later trim could drop the mapping those addresses need. A remap is
/// published whole or not at all.
///
/// What the walk leaves retired in the coverage was consumed by no recorded
/// transition. On a tagged table a covered fragment leaves the manifest only
/// through a recorded transition or a whole-fragment delete, so such a
/// fragment is gone (deleted, or a destination deleted since; the bitmap
/// swap narrows to what the reader still serves), unless `lineage` records it
/// while the ledger no longer mentions it: its mapping was trimmed away, the
/// addresses that need it cannot be translated, and the plan blocks rather
/// than treating them as deletions.
///
/// Iron law (asserted): a swapped bitmap only ever claims destination
/// fragments the segment fully owns -- destinations are added only by a hop
/// applied with ALL of its sources covered, so no destination mixing rows
/// from uncovered sources can be claimed.
fn walk(
    ledger: &FragReuseLedger,
    provenance: &RoaringBitmap,
    live: &RoaringBitmap,
    lineage: &RoaringBitmap,
) -> Walk {
    let mut coverage = provenance.clone();
    let mut admitted = provenance.clone();
    let mut hops = Vec::new();
    let mut applied = vec![false; ledger.transitions().len()];
    let mut changed = false;
    // Set once a stable-partition hop is applied: from then on the segment's
    // stored addresses are being moved, and no partial hop may be skipped.
    let mut restamped = false;
    // Transitions are in lineage order (producers before consumers), so one
    // pass reaches a fixpoint.
    for (position, transition) in ledger.transitions().iter().enumerate() {
        let sources = ids(transition.sources());
        let overlap = &sources & &coverage;
        if overlap.is_empty() {
            continue;
        }
        if overlap == sources {
            coverage -= &sources;
            let destinations = ids(transition.destinations());
            coverage |= &destinations;
            admitted |= &destinations;
            restamped |= matches!(transition.mapping(), Mapping::StablePartition(_));
            hops.push(RewriteHop {
                transition: position,
                enter: overlap,
            });
            applied[position] = true;
        } else {
            match transition.mapping() {
                Mapping::OrderedCompaction(_) if !restamped => {
                    coverage -= &overlap;
                }
                Mapping::OrderedCompaction(_) | Mapping::StablePartition(_) => {
                    return Walk::Blocked {
                        fragment: overlap.min().unwrap(),
                    };
                }
            }
        }
        changed = true;
    }
    debug_assert!(
        ledger
            .transitions()
            .iter()
            .enumerate()
            .all(|(position, transition)| {
                applied[position]
                    || transition.destinations().iter().all(|digest| {
                        !coverage.contains(digest.id as u32)
                            || provenance.contains(digest.id as u32)
                    })
            }),
        "a remapped fragment bitmap may only claim destinations the segment fully owns"
    );
    for fragment in (&coverage - live).iter() {
        if lineage.contains(fragment) && !ledger.contains_fragment(fragment) {
            return Walk::Blocked { fragment };
        }
    }
    if !changed {
        Walk::Identity
    } else {
        Walk::Remap {
            hops,
            coverage,
            admitted,
        }
    }
}

/// The fragments a segment's FILE still addresses, recovered from its
/// stored bitmap.
///
/// Under the v0 reuse index a deferred compaction left an index file holding
/// SOURCE addresses while the v0 reader swapped the segment's stored
/// `fragment_bitmap` to the group's DESTINATIONS and the next commit
/// persisted that swap; the v0 remap planner compensates with its
/// `data_predates_version` rule (`index.dataset_version <=
/// version.dataset_version` means the data still needs the group). A tagged
/// table lifts the v0 bytes verbatim and treats the stored bitmap as
/// provenance, so a segment that predates a lifted compaction would either
/// be skipped as already caught up, or, once a later transition consumes the
/// destinations, enter the remap with an `admitted` set that rejects every
/// address its file actually holds.
///
/// This walks the lifted legacy transitions in REVERSE lineage order and,
/// for each transition stamped `v_t` (`legacy_dataset_version`) where
/// `segment_dataset_version <= v_t`, whose destinations are all in the bitmap
/// and whose sources are absent from it, replaces the destinations by the
/// sources (`B = (B - destinations) | sources`). A segment stamped after
/// `v_t` is left alone (it was remapped, or built, after that compaction),
/// so nothing is reversed blindly. Reverse order unwinds multi-round chains:
/// with v1 `{2,3} -> D`, v2 `{0,1,D} -> X` and a stored `{X}`, the v2
/// transition first yields `{0,1,D}`, then the v1 transition `{0,1,2,3}`.
///
/// The result only seeds the walk; the walk's `coverage` is what gets
/// published. Should a file already hold destination addresses despite the
/// stamp, a compaction hop passes an address already in its destination
/// through unchanged, so it cannot be damaged.
fn effective_provenance(
    ledger: &FragReuseLedger,
    stored: &RoaringBitmap,
    segment_dataset_version: u64,
) -> RoaringBitmap {
    let mut bitmap = stored.clone();
    for transition in ledger.transitions().iter().rev() {
        let Some(legacy_version) = transition.legacy_dataset_version() else {
            continue;
        };
        // The legacy watermark is the version immediately before the rewrite
        // committed (#9620), so a segment stamped at it still predates the
        // compaction and holds source addresses: the boundary is inclusive,
        // as in the v0 remap (`data_predates_version`) and cleanup rules.
        if segment_dataset_version > legacy_version {
            continue;
        }
        let destinations = ids(transition.destinations());
        let sources = ids(transition.sources());
        if destinations.is_empty()
            || !destinations.is_subset(&bitmap)
            || !(&sources & &bitmap).is_empty()
        {
            continue;
        }
        bitmap -= &destinations;
        bitmap |= &sources;
    }
    bitmap
}

/// The physical row count of every fragment the history or the snapshot
/// knows. A digest records the count at the transition, which is the
/// physical size the fragment keeps.
fn known_fragment_rows(
    ledger: &FragReuseLedger,
    snapshot: &HashMap<u32, u64>,
) -> HashMap<u32, u64> {
    let mut rows = HashMap::new();
    for transition in ledger.transitions() {
        for digest in transition
            .sources()
            .iter()
            .chain(transition.destinations().iter())
        {
            if let Ok(id) = u32::try_from(digest.id) {
                rows.insert(id, digest.physical_rows);
            }
        }
    }
    rows.extend(snapshot.iter().map(|(&id, &count)| (id, count)));
    rows
}

/// Addresses are handed to the hops in batches of at most this many,
/// bounding the buffers one batch holds.
const REWRITE_BATCH: usize = 64 * 1024;

/// The translation a rewrite applies: each batch of addresses is pushed
/// through the planned hops one hop at a time, in ledger order, so a
/// compaction after a stable partition sees the partition's output. Nothing
/// is materialized up front: memory is the batch itself plus what the
/// mapping source holds.
///
/// An address enters the hops only if its fragment is admitted; an address
/// whose fragment a hop does not enter passes through it unchanged (a
/// destination address is never a source of the hop that produced it, so an
/// address a merged segment already translated passes through every hop),
/// and a row a hop deletes stays deleted. After the last hop an address is
/// kept only if its fragment is claimed, the coverage the rewritten segment
/// publishes: a row a partial compaction ceded to the scan, a row a direct
/// sibling took over, or a row in a fragment deleted since is dropped, so
/// the rewritten files carry no contribution their bitmap does not claim.
#[derive(Debug)]
pub(super) struct RewriteRemapper {
    ledger: Arc<FragReuseLedger>,
    path: Arc<RewritePath>,
    claimed: RoaringBitmap,
    /// Fragments some hop enters: their addresses move.
    entered: RoaringBitmap,
    mappings: Arc<dyn MappingSource>,
}

impl RewriteRemapper {
    pub(super) fn new(
        ledger: Arc<FragReuseLedger>,
        path: Arc<RewritePath>,
        claimed: RoaringBitmap,
        mappings: Arc<dyn MappingSource>,
    ) -> Self {
        let entered = path.hops.iter().map(|hop| &hop.enter).fold(
            RoaringBitmap::new(),
            |mut entered, enter| {
                entered |= enter;
                entered
            },
        );
        Self {
            ledger,
            path,
            claimed,
            entered,
            mappings,
        }
    }

    async fn translate_batch(&self, addrs: &mut [Option<u64>]) -> Result<()> {
        let fragment = |addr: u64| RowAddress::from(addr).fragment_id();
        for slot in addrs.iter_mut() {
            if slot.is_some_and(|addr| !self.path.admitted.contains(fragment(addr))) {
                *slot = None;
            }
        }
        for hop in &self.path.hops {
            let transition = &self.ledger.transitions()[hop.transition];
            match transition.mapping() {
                Mapping::OrderedCompaction(remap) => {
                    for slot in addrs.iter_mut() {
                        if let Some(addr) = *slot {
                            *slot = remap.get(addr).unwrap_or(Some(addr));
                        }
                    }
                }
                Mapping::StablePartition(partition) => {
                    let positions: Vec<usize> = (0..addrs.len())
                        .filter(|&position| {
                            addrs[position].is_some_and(|addr| hop.enter.contains(fragment(addr)))
                        })
                        .collect();
                    if positions.is_empty() {
                        continue;
                    }
                    let rows: Vec<u64> = positions
                        .iter()
                        .filter_map(|&position| addrs[position])
                        .collect();
                    let translated = translate_stable_partition(
                        self.mappings.as_ref(),
                        &MappingRef::of(partition),
                        transition.sources(),
                        transition.destinations(),
                        &rows,
                    )
                    .await?;
                    for (position, addr) in positions.into_iter().zip(translated) {
                        addrs[position] = addr;
                    }
                }
            }
        }
        for slot in addrs.iter_mut() {
            if slot.is_some_and(|addr| !self.claimed.contains(fragment(addr))) {
                *slot = None;
            }
        }
        Ok(())
    }
}

impl RemapperBacking for RewriteRemapper {
    fn fragment_needs_translation(&self, fragment: u32) -> bool {
        !self.claimed.contains(fragment) || self.entered.contains(fragment)
    }

    fn translate_fragments(&self, fragments: &RoaringBitmap) -> RoaringBitmap {
        let mut reached = fragments & &self.path.admitted;
        for hop in &self.path.hops {
            let entering = &reached & &hop.enter;
            if entering.is_empty() {
                continue;
            }
            reached -= &entering;
            reached |= ids(self.ledger.transitions()[hop.transition].destinations());
        }
        reached & &self.claimed
    }

    fn translate_in_place<'a>(&'a self, addrs: &'a mut [Option<u64>]) -> TranslateFuture<'a> {
        Box::pin(async move {
            for batch in addrs.chunks_mut(REWRITE_BATCH) {
                self.translate_batch(batch).await?;
            }
            Ok(())
        })
    }

    fn fragment_physical_rows(&self, fragment: u32) -> Option<u64> {
        self.path.fragment_rows.get(&fragment).copied()
    }
}

#[cfg(test)]
mod tests;
