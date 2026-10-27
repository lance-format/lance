// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which records of a fragment reuse history some index still needs.
//!
//! Computed from the history's digests and the stored index metadata alone,
//! with no payload IO, so both the maintenance trim and an action-based
//! commit's `TrimFragReuse` reach the same verdict.

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

    // Inclusive: an index built at dataset_version predates the rewrite.
    if index_meta.dataset_version <= frag_reuse_version.dataset_version {
        return Ok(false);
    }

    match index_meta.fragment_bitmap.clone() {
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

/// Whether some index still needs this transition to translate its rows.
///
/// RETAIN iff there is a logical index I (its stored segments grouped by
/// name, system indices exempt) such that:
/// * some segment's stored provenance bitmap intersects the transition's
///   sources (I holds addresses the transition moves), AND
/// * some destination is not directly covered by any USABLE segment of I
///   (I has not fully re-derived onto the transition's output in a form this
///   build can serve).
///
/// A segment without a stored bitmap imposes no constraint, mirroring the v0
/// leniency for missing bitmaps.
fn is_transition_needed(
    transition: &pb_fri::Transition,
    index_groups: &HashMap<&str, IndexGroupCoverage>,
) -> bool {
    index_groups.values().any(|group| {
        let sources_hit = transition
            .sources
            .iter()
            .any(|source| group.provenance.contains(source.id as u32));
        sources_hit
            && transition
                .destinations
                .iter()
                .any(|destination| !group.direct.contains(destination.id as u32))
    })
}

/// Which elements of a tagged history are still needed, from the entry's
/// digests and the stored index metadata alone (zero payload IO).
///
/// Legacy versions keep the exact v0 caught-up predicate; transitions use
/// [`is_transition_needed`]; then transitive chain retention closes the set
/// forward: a retained record's translation path walks through every
/// downstream consumer of its destinations, so those records must survive
/// too. Legacy versions retain at whole-version granularity (their groups
/// stand or fall together, as in v0).
pub fn compute_tagged_retention(
    elements: &[TrimElement],
    indices: &[IndexMetadata],
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

    let mut retained = vec![false; elements.len()];
    for (position, element) in elements.iter().enumerate() {
        match &element.record {
            TrimRecord::LegacyVersion(version) => {
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
            TrimRecord::Transition(transition) => {
                retained[position] = is_transition_needed(transition, &index_groups);
            }
        }
    }

    // Transitive chain retention (forward closure over producer->consumer
    // edges).
    struct Node {
        element: usize,
        sources: Vec<u64>,
        destinations: Vec<u64>,
    }
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
