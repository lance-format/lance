// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which records of a fragment reuse history some index still needs.
//!
//! Computed from the history's digests, the stored index metadata and the
//! live fragment set alone, with no payload IO, so both the maintenance trim
//! and an action-based commit's `TrimFragReuse` reach the same verdict.

use std::collections::{HashMap, VecDeque};

use lance_core::{Error, Result};
use log::warn;
use prost::Message;
use roaring::RoaringBitmap;

use super::FragReuseVersion;
use crate::format::IndexMetadata;
use crate::format::pb::fragment_reuse_index_details as pb_fri;
use crate::system_index::is_system_index;

/// Every fragment the reuse chain touches (old + new) across all versions. An
/// index disjoint from this set holds no row address the FRI remaps, so trimming
/// can never strand it (fragment ids are never reused).
pub fn reuse_chain_frag_bitmap(versions: &[FragReuseVersion]) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for version in versions {
        bitmap.extend(version.old_frag_ids().iter().map(|&id| id as u32));
        bitmap.extend(version.new_frag_ids().iter().map(|&id| id as u32));
    }
    bitmap
}

pub fn is_index_remap_caught_up(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
    chain_frag_bitmap: &RoaringBitmap,
) -> Result<bool> {
    if is_system_index(index_meta) {
        return Ok(true);
    }

    // Disjoint coverage => caught up regardless of dataset_version, bypassing the
    // stale-version gate below (see fn docs). The chain includes NEW fragments
    // deliberately: a deferred-remap commit advances a covering index's bitmap
    // onto them before its data is remapped, so an old-frag-only check would
    // clear a still-stale index and trim a version it needs.
    if let Some(index_frag_bitmap) = &index_meta.fragment_bitmap
        && index_frag_bitmap.is_disjoint(chain_frag_bitmap)
    {
        return Ok(true);
    }

    index_postdates_rewrite(frag_reuse_version, index_meta)
}

/// Built after the rewrite and covering none of its sources. No disjoint
/// shortcut: coverage can be withdrawn while the index data keeps the rows.
pub fn index_postdates_rewrite(
    frag_reuse_version: &FragReuseVersion,
    index_meta: &IndexMetadata,
) -> Result<bool> {
    // Inclusive: an index built at dataset_version predates the rewrite.
    if index_meta.dataset_version <= frag_reuse_version.dataset_version {
        return Ok(false);
    }

    match &index_meta.fragment_bitmap {
        Some(index_frag_bitmap) => {
            for group in frag_reuse_version.groups.iter() {
                let mut old_frag_in_index = 0;
                for old_frag in group.old_frags.iter() {
                    if index_frag_bitmap.contains(old_frag.id as u32) {
                        old_frag_in_index += 1;
                    }
                }

                if old_frag_in_index > 0 {
                    if old_frag_in_index != group.old_frags.len() {
                        // This should never happen because we always commit a full rewrite group
                        // and we always reindex either the entire group or nothing.
                        // We use invalid input to be consistent with
                        // dataset::transaction::recalculate_fragment_bitmap
                        return Err(Error::invalid_input(format!(
                            "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                            group.old_frags
                        )));
                    }
                    return Ok(false);
                }
            }
            Ok(true)
        }
        None => {
            warn!(
                "Index {} ({}) missing fragment bitmap, cannot determine if it is caught up with the fragment reuse version, consider retraining the index",
                index_meta.name, index_meta.uuid
            );
            Ok(true)
        }
    }
}

/// One splice unit of the entry's content: a legacy version (field 1) or a
/// transition (field 2), kept with its exact wire form so retained records
/// are carried forward verbatim.
pub enum TrimRecord {
    LegacyVersion(FragReuseVersion),
    Transition(pb_fri::Transition),
}

pub struct TrimElement {
    pub raw: Vec<u8>,
    pub record: TrimRecord,
}

pub fn split_trim_elements(content: &[u8]) -> Result<Vec<TrimElement>> {
    use bytes::Buf;
    use prost::encoding::{WireType, decode_key, decode_varint};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = content.len();
    let mut buf = bytes::Bytes::copy_from_slice(content);
    let mut elements = Vec::new();
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        // Fields 1 and 2 are the only known records and both are messages. An
        // unknown record may participate in lineage in ways this client cannot
        // reason about, so refuse to trim rather than silently dropping or
        // blindly retaining it.
        if wire_type != WireType::LengthDelimited || !matches!(tag, 1 | 2) {
            return Err(Error::not_supported(format!(
                "the tagged FRI history carries an unknown record (field {tag}); \
                 upgrade to a newer version of Lance before trimming"
            )));
        }
        let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if length > buf.remaining() as u64 {
            return Err(corrupt(format!(
                "field {tag} length {length} exceeds remaining {} bytes",
                buf.remaining()
            )));
        }
        let payload = buf.split_to(length as usize);
        let end = total - buf.remaining();
        let record = match tag {
            1 => TrimRecord::LegacyVersion(
                pb_fri::Version::decode(payload)
                    .map_err(|e| corrupt(e.to_string()))?
                    .try_into()?,
            ),
            _ => TrimRecord::Transition(
                pb_fri::Transition::decode(payload).map_err(|e| corrupt(e.to_string()))?,
            ),
        };
        elements.push(TrimElement {
            raw: content[start..end].to_vec(),
            record,
        });
    }
    Ok(elements)
}

/// One record's lineage edge as the trim walks it: a transition, or one group
/// of a lifted legacy version.
struct Node {
    element: usize,
    sources: Vec<u64>,
    destinations: Vec<u64>,
}

/// One logical index's coverage as the trim sees it: provenance is the union
/// of every stored segment bitmap (any segment, usable by this build or not,
/// pins the records its addresses need), while direct coverage counts only
/// segments this build can actually serve (`index_is_usable`) -- an unusable
/// segment answers no query, so its bitmap must not mask anyone's need for a
/// translation.
#[derive(Default)]
struct IndexGroupCoverage {
    provenance: RoaringBitmap,
    direct: RoaringBitmap,
}

/// Bitmaps hold u32 ids, so an id outside that range is in none of them.
fn bitmap_contains(bitmap: &RoaringBitmap, fragment: u64) -> bool {
    u32::try_from(fragment).is_ok_and(|id| bitmap.contains(id))
}

/// Every record naming each fragment as a destination. A valid history has
/// one producer per fragment, but only tagged transitions are checked for a
/// duplicate producer (`order_lineage`); lifted legacy groups are not, so
/// every producer is kept and walked.
fn producers_of(nodes: &[Node]) -> HashMap<u64, Vec<usize>> {
    let mut producer_of: HashMap<u64, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for destination in &node.destinations {
            producer_of
                .entry(*destination)
                .or_default()
                .push(node_index);
        }
    }
    producer_of
}

/// The producers the walk follows out of `fragment`: none once it is directly
/// covered.
fn upstream_producers<'a>(
    fragment: u64,
    direct: &RoaringBitmap,
    producer_of: &'a HashMap<u64, Vec<usize>>,
) -> &'a [usize] {
    if bitmap_contains(direct, fragment) {
        return &[];
    }
    producer_of
        .get(&fragment)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

/// One node's verdict in the walk of one logical index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    InProgress,
    Pinned,
    NotPinned,
}

struct IndexWalk {
    /// `None` for a node the walk did not reach.
    verdicts: Vec<Option<Verdict>>,
    /// Node resolutions performed, so tests can check the memoisation.
    #[cfg(test)]
    resolutions: usize,
}

/// Which records lie on a translation path one logical index still needs,
/// and whether that index pins them.
///
/// The walk starts at every live fragment that some record produces and the
/// index does not cover directly, visits its producers, and from each visited
/// record continues to the producers of its sources outside direct coverage;
/// it never crosses a directly covered fragment. A visited record is pinned
/// when one of its sources is in the index's provenance, or a source outside
/// direct coverage has a pinned producer. Every such producer is visited even
/// once the record is known pinned, since upstream records need their own
/// verdict.
///
/// This mirrors the reader's `FragmentReuseIndex::segment_plans` (per index,
/// direct coverage first, otherwise back through producers) with two
/// differences: every stored bitmap pins, while only usable segments stop the
/// walk.
///
/// Post-order with an explicit stack so long chains cannot overflow, and
/// memoised so each node is resolved at most once per index. A node reached
/// again while in progress (a cycle, which a valid history cannot hold) counts
/// as pinned, so every node on the cycle resolves pinned and the trim retains.
fn walk_index(
    nodes: &[Node],
    producer_of: &HashMap<u64, Vec<usize>>,
    coverage: &IndexGroupCoverage,
    live_fragments: &RoaringBitmap,
) -> IndexWalk {
    struct Frame {
        node: usize,
        next_source: usize,
        next_producer: usize,
        pinned: bool,
    }
    let open = |node: usize| Frame {
        node,
        next_source: 0,
        next_producer: 0,
        pinned: nodes[node]
            .sources
            .iter()
            .any(|&source| bitmap_contains(&coverage.provenance, source)),
    };

    let mut verdicts: Vec<Option<Verdict>> = vec![None; nodes.len()];
    #[cfg(test)]
    let mut resolutions = 0usize;
    let mut stack: Vec<Frame> = Vec::new();
    for (&fragment, producers) in producer_of {
        if bitmap_contains(&coverage.direct, fragment) || !bitmap_contains(live_fragments, fragment)
        {
            continue;
        }
        for &root in producers {
            if verdicts[root].is_some() {
                continue;
            }
            verdicts[root] = Some(Verdict::InProgress);
            stack.push(open(root));
            while let Some(frame) = stack.last_mut() {
                let sources = &nodes[frame.node].sources;
                let mut descend = None;
                while let Some(&source) = sources.get(frame.next_source) {
                    let producers = upstream_producers(source, &coverage.direct, producer_of);
                    let Some(&producer) = producers.get(frame.next_producer) else {
                        frame.next_source += 1;
                        frame.next_producer = 0;
                        continue;
                    };
                    frame.next_producer += 1;
                    match verdicts[producer] {
                        None => {
                            descend = Some(producer);
                            break;
                        }
                        Some(Verdict::Pinned | Verdict::InProgress) => frame.pinned = true,
                        Some(Verdict::NotPinned) => {}
                    }
                }
                if let Some(child) = descend {
                    verdicts[child] = Some(Verdict::InProgress);
                    stack.push(open(child));
                    continue;
                }
                let (node, pinned) = (frame.node, frame.pinned);
                stack.pop();
                verdicts[node] = Some(if pinned {
                    Verdict::Pinned
                } else {
                    Verdict::NotPinned
                });
                #[cfg(test)]
                {
                    resolutions += 1;
                }
                if pinned && let Some(parent) = stack.last_mut() {
                    parent.pinned = true;
                }
            }
        }
    }
    IndexWalk {
        verdicts,
        #[cfg(test)]
        resolutions,
    }
}

/// Which elements of a tagged history are still needed, from the entry's
/// digests, the stored index metadata and the live fragment set alone (zero
/// payload IO). `live_fragments` must be the fragments of the same manifest
/// `indices` was read from.
///
/// Seeds. Legacy versions keep the exact v0 caught-up predicate. A tagged
/// transition is a seed when, for some logical index (its stored segments
/// grouped by name, system indices exempt), the producer walk of
/// `walk_index` visits it and finds it pinned: every stored segment bitmap
/// of the index pins (a segment without a bitmap imposes nothing, mirroring
/// the v0 leniency), and only the bitmaps of usable segments stop the walk.
/// Only an index's own direct coverage stops its walk: the reader gives direct
/// coverage precedence per logical index, so another index's segments never
/// stand in for its translation. Lifted legacy groups pass pins through the
/// walk but their verdict does not decide a legacy version.
///
/// A transition none of whose destinations leads to a live fragment outside
/// direct coverage is released: a retired destination that no record
/// consumes had its rows deleted, since the trim does not run on a history
/// with unsupported transitions, and the forward closure never releases a
/// consumer before the record producing its sources. A fan-in whose sources
/// are only partly in provenance stays retained, since a later segment over
/// the other sources may complete it.
///
/// Final retention is the forward closure of all seeds: a retained record's
/// translation path walks through every downstream consumer of its
/// destinations, so those records survive too, and legacy versions retain at
/// whole-version granularity (their groups stand or fall together, as in
/// v0). A record can therefore be retained although no walk visited or pinned
/// it. Each index's walk resolves every node at most once.
pub fn compute_tagged_retention(
    elements: &[TrimElement],
    indices: &[IndexMetadata],
    live_fragments: &RoaringBitmap,
    is_usable: impl Fn(&IndexMetadata) -> bool,
) -> Vec<bool> {
    // The v0 predicate's disjointness rule is scoped to the legacy chain, as
    // in v0: an index touching only tagged-transition fragments holds nothing
    // a legacy version remaps.
    let legacy_versions: Vec<FragReuseVersion> = elements
        .iter()
        .filter_map(|element| match &element.record {
            TrimRecord::LegacyVersion(version) => Some(version.clone()),
            TrimRecord::Transition(_) => None,
        })
        .collect();
    let chain_frag_bitmap = reuse_chain_frag_bitmap(&legacy_versions);

    // Stored segments grouped by logical index name. Every segment's bitmap
    // counts as provenance (pinning); only usable segments' bitmaps count as
    // direct coverage (masking).
    let mut index_groups: HashMap<&str, IndexGroupCoverage> = HashMap::new();
    for index in indices.iter() {
        if is_system_index(index) {
            continue;
        }
        let group = index_groups.entry(index.name.as_str()).or_default();
        match &index.fragment_bitmap {
            Some(bitmap) => {
                group.provenance |= bitmap;
                if is_usable(index) {
                    group.direct |= bitmap;
                }
            }
            None => warn!(
                "Index {} ({}) missing fragment bitmap, it cannot pin tagged fragment reuse records, consider retraining the index",
                index.name, index.uuid
            ),
        }
    }

    // Lineage edges, shared by the producer walk and the forward closure.
    let mut nodes = Vec::new();
    for (position, element) in elements.iter().enumerate() {
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
                for group in version.groups.iter() {
                    nodes.push(Node {
                        element: position,
                        sources: group.old_frags.iter().map(|f| f.id).collect(),
                        destinations: group.new_frags.iter().map(|f| f.id).collect(),
                    });
                }
            }
            TrimRecord::Transition(transition) => nodes.push(Node {
                element: position,
                sources: transition.sources.iter().map(|d| d.id).collect(),
                destinations: transition.destinations.iter().map(|d| d.id).collect(),
            }),
        }
    }
    // Relies on each fragment being consumed by at most one record; a repeated
    // source would be last-wins. The ledger rejects duplicate consumers among
    // tagged transitions (`order_lineage`); lifted legacy groups are not
    // checked there.
    let mut consumer_of: HashMap<u64, usize> = HashMap::new();
    let mut element_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for source in &node.sources {
            consumer_of.insert(*source, node_index);
        }
        element_nodes
            .entry(node.element)
            .or_default()
            .push(node_index);
    }

    let producer_of = producers_of(&nodes);

    let mut retained = vec![false; elements.len()];
    for (position, element) in elements.iter().enumerate() {
        if let TrimRecord::LegacyVersion(version) = &element.record {
            let check_results = indices
                .iter()
                .map(|idx| is_index_remap_caught_up(version, idx, &chain_frag_bitmap))
                .collect::<Vec<_>>();
            if check_results
                .iter()
                .any(|r| matches!(r, Err(Error::InvalidInput { .. })))
            {
                // If the check fails, the reuse version is likely corrupted,
                // do not retain it (v0 behavior).
                continue;
            }
            if !check_results.into_iter().all(|r| r.unwrap()) {
                retained[position] = true;
            }
        }
    }
    for coverage in index_groups.values() {
        let walk = walk_index(&nodes, &producer_of, coverage, live_fragments);
        for (node, verdict) in nodes.iter().zip(walk.verdicts) {
            if verdict == Some(Verdict::Pinned)
                && matches!(elements[node.element].record, TrimRecord::Transition(_))
            {
                retained[node.element] = true;
            }
        }
    }

    // Transitive chain retention (forward closure over producer->consumer
    // edges).
    let mut node_retained = vec![false; nodes.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if retained[node.element] {
            node_retained[node_index] = true;
            queue.push_back(node_index);
        }
    }
    while let Some(node_index) = queue.pop_front() {
        for destination in nodes[node_index].destinations.clone() {
            let Some(&consumer) = consumer_of.get(&destination) else {
                continue;
            };
            if node_retained[consumer] {
                continue;
            }
            node_retained[consumer] = true;
            queue.push_back(consumer);
            let element = nodes[consumer].element;
            if !retained[element] {
                retained[element] = true;
                for &sibling in element_nodes[&element].iter() {
                    if !node_retained[sibling] {
                        node_retained[sibling] = true;
                        queue.push_back(sibling);
                    }
                }
            }
        }
    }

    retained
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(
        sources: impl IntoIterator<Item = u64>,
        destinations: impl IntoIterator<Item = u64>,
    ) -> Node {
        Node {
            element: 0,
            sources: sources.into_iter().collect(),
            destinations: destinations.into_iter().collect(),
        }
    }

    fn pinning(fragments: &[u32]) -> IndexGroupCoverage {
        IndexGroupCoverage {
            provenance: fragments.iter().copied().collect(),
            direct: RoaringBitmap::new(),
        }
    }

    /// Every destination of a wide transition is a root, yet the transition
    /// is resolved once.
    #[test]
    fn wide_transition_is_resolved_once() {
        const SOURCES: u64 = 4_096;
        const DESTINATIONS: u64 = 2_048;
        let nodes = vec![node(1..=SOURCES, SOURCES + 1..=SOURCES + DESTINATIONS)];
        let live: RoaringBitmap = (SOURCES as u32 + 1..=(SOURCES + DESTINATIONS) as u32).collect();
        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[1]), &live);
        assert_eq!(walk.resolutions, 1);
        assert_eq!(walk.verdicts, vec![Some(Verdict::Pinned)]);
    }

    /// Many roots behind one shared upstream chain: each reachable node is
    /// resolved once, not once per root.
    #[test]
    fn shared_upstream_chain_is_resolved_once_per_node() {
        const HOPS: u64 = 1_000;
        const ROOTS: u64 = 500;
        // Chain `k -> k + 1` for k in 1..=HOPS, a split of HOPS + 1 into
        // ROOTS fragments, and one hop from each of them onto a live root.
        let mut nodes: Vec<Node> = (1..=HOPS).map(|hop| node([hop], [hop + 1])).collect();
        let split_base = HOPS + 2;
        nodes.push(node([HOPS + 1], split_base..split_base + ROOTS));
        let root_base = split_base + ROOTS;
        nodes.extend((0..ROOTS).map(|i| node([split_base + i], [root_base + i])));
        let live: RoaringBitmap = (root_base as u32..(root_base + ROOTS) as u32).collect();

        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[1]), &live);
        assert_eq!(walk.resolutions, nodes.len());
        assert!(
            walk.verdicts
                .iter()
                .all(|verdict| *verdict == Some(Verdict::Pinned))
        );
    }

    /// A cycle (which a valid history cannot hold) resolves every node on it
    /// pinned, even without any provenance.
    #[test]
    fn cycle_resolves_pinned() {
        let nodes = vec![node([2], [1]), node([1], [2]), node([1], [3])];
        let live: RoaringBitmap = [3u32].into_iter().collect();
        let walk = walk_index(&nodes, &producers_of(&nodes), &pinning(&[]), &live);
        assert_eq!(walk.resolutions, 3);
        assert!(
            walk.verdicts
                .iter()
                .all(|verdict| *verdict == Some(Verdict::Pinned))
        );
    }
}
