// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which records of a tagged history the live index segments still need.

use std::collections::{HashMap, VecDeque};

use roaring::RoaringBitmap;

use super::super::StoredRecord;
use super::super::legacy::{is_caught_up_with_version, warn_missing_bitmap};
use crate::format::pb::fragment_reuse_index_details as pb;
use crate::system_index::frag_reuse::history::Record;
use crate::system_index::frag_reuse::plan::{SegmentSummary, TranslationSupport};

/// One logical index's coverage as the trim sees it: provenance is the union
/// of every stored segment bitmap (any segment, usable by this build or not,
/// pins the records its addresses need), while direct coverage counts only
/// segments this build can serve -- an unusable segment answers no query, so
/// its bitmap must not mask anyone's need for a translation.
#[derive(Default)]
struct GroupCoverage {
    provenance: RoaringBitmap,
    direct: RoaringBitmap,
}

/// Whether some logical index still needs `transition` to translate its
/// rows: some segment's stored provenance intersects the transition's
/// sources (the index holds addresses the transition moves), and some
/// destination is not directly covered by a usable segment of the index (it
/// has not fully re-derived onto the transition's output in a form this
/// build can serve). A segment without a stored bitmap imposes no
/// constraint, mirroring the v0 leniency for missing bitmaps.
fn is_transition_needed(transition: &pb::Transition, groups: &[GroupCoverage]) -> bool {
    groups.iter().any(|group| {
        transition
            .sources
            .iter()
            .any(|source| group.provenance.contains(source.id as u32))
            && transition
                .destinations
                .iter()
                .any(|destination| !group.direct.contains(destination.id as u32))
    })
}

/// Which of `records` are still needed, in serialized order, from the
/// records' digests and the segments' metadata alone.
///
/// Lifted legacy versions keep the exact v0 caught-up predicate;
/// transitions use [`is_transition_needed`]; then transitive chain retention
/// closes the set forward: a retained record's translation path walks
/// through every downstream consumer of its destinations, so those records
/// survive too. Legacy versions retain at whole-version granularity (their
/// groups stand or fall together, as in v0). Every record must be
/// interpretable; the caller refuses a history that is not.
pub(super) fn retention(records: &[StoredRecord], live: &[Vec<SegmentSummary>]) -> Vec<bool> {
    // The v0 predicate's disjointness rule is scoped to the legacy chain, as
    // in v0: an index touching only tagged-transition fragments holds nothing
    // a legacy version remaps.
    let mut legacy_chain = RoaringBitmap::new();
    for stored in records {
        if let Record::LegacyVersion(version) = &stored.record {
            for group in &version.groups {
                legacy_chain.extend(group.old_fragments.iter().map(|digest| digest.id as u32));
                legacy_chain.extend(group.new_fragments.iter().map(|digest| digest.id as u32));
            }
        }
    }
    let groups: Vec<GroupCoverage> = live
        .iter()
        .map(|segments| {
            let mut group = GroupCoverage::default();
            for segment in segments {
                if let Some(bitmap) = &segment.fragment_bitmap {
                    group.provenance |= bitmap;
                    if segment.translation != TranslationSupport::Uninterpretable {
                        group.direct |= bitmap;
                    }
                }
            }
            group
        })
        .collect();

    let mut retained = vec![false; records.len()];
    for (position, stored) in records.iter().enumerate() {
        retained[position] = match &stored.record {
            Record::LegacyVersion(version) => {
                let mut caught_up = true;
                let mut corrupt = false;
                for segment in live.iter().flatten() {
                    warn_missing_bitmap(segment, version.dataset_version);
                    match is_caught_up_with_version(
                        version.dataset_version,
                        version.groups.iter().map(|group| {
                            group.old_fragments.iter().map(|digest| digest.id).collect()
                        }),
                        segment.fragment_bitmap.as_ref(),
                        segment.dataset_version,
                        &legacy_chain,
                    ) {
                        Ok(true) => {}
                        Ok(false) => caught_up = false,
                        Err(_) => corrupt = true,
                    }
                }
                // A version some segment covers part of a group of is
                // likely corrupt: it is not retained (v0 behavior).
                !corrupt && !caught_up
            }
            Record::Transition(transition) => is_transition_needed(transition, &groups),
            Record::Uninterpretable { .. } => true,
        };
    }

    // Transitive chain retention (forward closure over producer->consumer
    // edges).
    struct Node {
        record: usize,
        sources: Vec<u64>,
        destinations: Vec<u64>,
    }
    let mut nodes = Vec::new();
    for (position, stored) in records.iter().enumerate() {
        match &stored.record {
            Record::LegacyVersion(version) => {
                for group in &version.groups {
                    nodes.push(Node {
                        record: position,
                        sources: group.old_fragments.iter().map(|f| f.id).collect(),
                        destinations: group.new_fragments.iter().map(|f| f.id).collect(),
                    });
                }
            }
            Record::Transition(transition) => nodes.push(Node {
                record: position,
                sources: transition.sources.iter().map(|d| d.id).collect(),
                destinations: transition.destinations.iter().map(|d| d.id).collect(),
            }),
            Record::Uninterpretable { .. } => {}
        }
    }
    let mut consumer_of: HashMap<u64, usize> = HashMap::new();
    let mut record_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for source in &node.sources {
            consumer_of.insert(*source, node_index);
        }
        record_nodes
            .entry(node.record)
            .or_default()
            .push(node_index);
    }
    let mut node_retained = vec![false; nodes.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node_index, node) in nodes.iter().enumerate() {
        if retained[node.record] {
            node_retained[node_index] = true;
            queue.push_back(node_index);
        }
    }
    while let Some(node_index) = queue.pop_front() {
        for destination in &nodes[node_index].destinations {
            let Some(&consumer) = consumer_of.get(destination) else {
                continue;
            };
            if node_retained[consumer] {
                continue;
            }
            node_retained[consumer] = true;
            queue.push_back(consumer);
            let record = nodes[consumer].record;
            if !retained[record] {
                retained[record] = true;
                for &sibling in &record_nodes[&record] {
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
    use prost::Message;
    use uuid::Uuid;

    use super::*;
    use crate::system_index::frag_reuse::history::tests::{digest, legacy, whole};
    use crate::system_index::frag_reuse::history::{
        FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, RawHistory, TRANSITION_FIELD,
        length_delimited,
    };
    use crate::system_index::frag_reuse::plan::RowDomain;

    fn transition(sources: &[u64], destinations: &[u64]) -> pb::Transition {
        pb::Transition {
            sources: sources.iter().map(|&id| digest(id, 4, 0)).collect(),
            destinations: destinations.iter().map(|&id| digest(id, 4, 0)).collect(),
            mapping: Some(pb::transition::Mapping::StablePartition(
                pb::StablePartition {
                    map_id: Uuid::new_v4().to_string(),
                    map_size_bytes: 1,
                    base_id: None,
                },
            )),
        }
    }

    enum Element {
        Legacy(pb::Version),
        Transition(pb::Transition),
    }

    fn history(elements: Vec<Element>) -> FragReuseHistory {
        let mut content = Vec::new();
        for element in elements {
            content.extend(match element {
                Element::Legacy(version) => {
                    length_delimited(LEGACY_VERSION_FIELD, &version.encode_to_vec())
                }
                Element::Transition(transition) => {
                    length_delimited(TRANSITION_FIELD, &transition.encode_to_vec())
                }
            });
        }
        RawHistory::new(HistoryFormat::Tagged, content.into())
            .validate()
            .unwrap()
    }

    fn legacy_element(dataset_version: u64, old: &[u64], new: &[u64]) -> Element {
        let sources: Vec<_> = old.iter().map(|&id| whole(id, 4)).collect();
        let destinations: Vec<_> = new.iter().map(|&id| (id, 4)).collect();
        Element::Legacy(legacy(dataset_version, &sources, &destinations))
    }

    fn segment(dataset_version: u64, covered: Option<&[u32]>) -> SegmentSummary {
        SegmentSummary {
            uuid: Uuid::new_v4(),
            fragment_bitmap: covered.map(|covered| covered.iter().copied().collect()),
            dataset_version,
            domain: RowDomain::Address,
            translation: TranslationSupport::Supported,
        }
    }

    fn retain(history: &FragReuseHistory, live: &[Vec<SegmentSummary>]) -> Vec<bool> {
        retention(&history.imp().records().stored, live)
    }

    /// Retain iff some logical index still derives coverage from the
    /// transition's sources while not directly covering every destination.
    #[test]
    fn transition_retention_rules() {
        let history = history(vec![Element::Transition(transition(&[1, 2], &[5, 6]))]);
        let retain = |live: Vec<Vec<SegmentSummary>>| retain(&history, &live)[0];
        let at = |version, covered: &[u32]| segment(version, Some(covered));

        // A disjoint index does not block the trim.
        assert!(!retain(vec![vec![at(1, &[3, 4])]]));
        // Still deriving from the sources, destinations uncovered.
        assert!(retain(vec![vec![at(1, &[1, 2])]]));
        // A partial drain (one destination still uncovered) retains.
        assert!(retain(vec![vec![at(1, &[1, 5])]]));
        // Full direct coverage across the index's segments trims, even while
        // another segment still lists the sources.
        assert!(!retain(vec![vec![at(1, &[1, 2]), at(2, &[5, 6])]]));
        // Segments rebuilt onto the destinations (sources gone) trim.
        assert!(!retain(vec![vec![at(2, &[5, 6])]]));
        // One index caught up, another still hanging off the sources.
        assert!(retain(vec![vec![at(2, &[5, 6])], vec![at(1, &[1])]]));
        // No indices left: nothing pins the record.
        assert!(!retain(vec![]));
        // A missing bitmap imposes no constraint (v0 leniency) ...
        assert!(!retain(vec![vec![segment(1, None)]]));
        // ... but a sibling segment with a bitmap still pins it.
        assert!(retain(vec![vec![segment(1, None), at(1, &[1])]]));
        // A segment this build cannot serve pins, but never masks.
        let mut unusable = at(2, &[5, 6]);
        unusable.translation = TranslationSupport::Uninterpretable;
        assert!(retain(vec![vec![at(1, &[1, 2]), unusable]]));
    }

    /// A segment hanging off the head of a chain pins every downstream hop
    /// its translation path traverses; hops upstream of where it enters are
    /// not needed.
    #[test]
    fn chain_retention_closes_forward() {
        let history = history(vec![
            Element::Transition(transition(&[1], &[2])),
            Element::Transition(transition(&[2], &[3])),
        ]);
        let retain = |covered: &[u32]| retain(&history, &[vec![segment(1, Some(covered))]]);
        assert_eq!(retain(&[1]), vec![true, true]);
        assert_eq!(retain(&[2]), vec![false, true]);
        assert_eq!(retain(&[3]), vec![false, false]);
    }

    /// Chain retention traverses lifted legacy groups too, and a legacy
    /// version retains at whole-version granularity.
    #[test]
    fn chain_retention_spans_legacy_versions() {
        let chained = history(vec![
            legacy_element(10, &[1], &[2]),
            Element::Transition(transition(&[2], &[3])),
        ]);
        // A stale index over the legacy sources pins the version (exact v0
        // predicate) and, transitively, the downstream transition.
        assert_eq!(
            retain(&chained, &[vec![segment(5, Some(&[1]))]]),
            vec![true, true]
        );
        // Caught up past the legacy version and rebuilt onto the final
        // fragments: everything trims.
        assert_eq!(
            retain(&chained, &[vec![segment(11, Some(&[3]))]]),
            vec![false, false]
        );

        let grouped = history(vec![
            legacy_element(10, &[1, 5], &[2, 6]),
            Element::Transition(transition(&[6], &[7])),
        ]);
        assert_eq!(
            retain(&grouped, &[vec![segment(5, Some(&[1]))]]),
            vec![true, true]
        );
    }
}
