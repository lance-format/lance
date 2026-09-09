// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata-only reader for the unified fragment reuse history.
//!
//! Decoding validates lineage without opening mapping files. Unknown alternatives
//! retain only their common fragment metadata for conservative query coverage.
//! Operations that carry an index forward keep its original `Any` separately.
//!
//! ```
//! use bytes::Bytes;
//! use lance_table::system_index::frag_reuse::ledger::FragReuseLedger;
//!
//! let ledger = FragReuseLedger::decode(1, Bytes::new())?;
//! assert!(ledger.transitions().is_empty());
//! # Ok::<(), lance_core::Error>(())
//! ```

use std::collections::{HashMap, VecDeque};
use std::io::Cursor;

use bytes::{Buf, Bytes};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::{GroupInputWithLayout, RowAddrRemap};
use lance_core::{Error, Result};
use prost::Message;
use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use crate::format::pb::fragment_reuse_index_details::{self as pb, transition};

/// A decoded mapping. External labels remain unopened until address resolution.
#[derive(Debug)]
pub enum Mapping {
    /// Bitmap/rank translation, including lifted legacy compaction groups.
    OrderedCompaction(RowAddrRemap),
    /// Immutable row-map reference, with the base selected by its optional base ID.
    StablePartition(pb::StablePartition),
    /// An unrecognized alternative; it cannot provide address translation.
    Unknown {
        /// Protobuf field number identifying the future mapping alternative.
        field_number: u32,
    },
}

/// One whole-fragment rewrite, with fragment lists in mapping order.
#[derive(Debug)]
pub struct Transition {
    sources: Vec<pb::FragmentDigest>,
    destinations: Vec<pb::FragmentDigest>,
    mapping: Mapping,
}

impl Transition {
    /// Source digests at rewrite time, including deleted physical positions.
    pub fn sources(&self) -> &[pb::FragmentDigest] {
        &self.sources
    }

    /// Destination digests at creation time.
    pub fn destinations(&self) -> &[pb::FragmentDigest] {
        &self.destinations
    }

    /// Mapping semantics, or an opaque future alternative.
    pub fn mapping(&self) -> &Mapping {
        &self.mapping
    }
}

/// Validated history in fragment-lineage order, independent of serialization order.
#[derive(Debug)]
pub struct FragReuseLedger {
    transitions: Vec<Transition>,
    consumers: HashMap<u32, usize>,
    producers: HashMap<u32, usize>,
}

impl FragReuseLedger {
    /// Decode serialized `InlineContent`, after resolving the outer inline/external
    /// wrapper. Versions 0 (legacy only) and 1 (mixed history) are supported.
    /// Unsupported versions and malformed or ambiguous lineage are rejected.
    pub fn decode(index_version: i32, content: Bytes) -> Result<Self> {
        if !matches!(index_version, 0 | 1) {
            return Err(Error::not_supported(format!(
                "FRI index_version {index_version}; supported versions are 0 and 1"
            )));
        }
        let mut remaining = content;
        let mut transitions = Vec::new();
        while remaining.has_remaining() {
            let (tag, payload) = next_field(&mut remaining)?;
            match tag {
                1 => {
                    let version = pb::Version::decode(require_message(tag, payload)?)
                        .map_err(|e| corrupt(e.to_string()))?;
                    for group in version.groups {
                        transitions.push(decode_transition(
                            pb::Transition {
                                sources: group.old_fragments,
                                destinations: group.new_fragments,
                                encoding: Some(transition::Encoding::OrderedCompaction(
                                    pb::OrderedCompaction {
                                        changed_row_addrs: group.changed_row_addrs,
                                    },
                                )),
                            },
                            5,
                        )?);
                    }
                }
                2 => {
                    if index_version == 0 {
                        return Err(corrupt("tagged transitions require FRI index_version 1"));
                    }
                    let raw = require_message(tag, payload)?;
                    let mut fields = raw.clone();
                    let mut mapping_tag = None;
                    while fields.has_remaining() {
                        let (tag, payload) = next_field(&mut fields)?;
                        if tag >= 5 {
                            require_message(tag, payload)?;
                            if mapping_tag.replace(tag).is_some() {
                                return Err(corrupt(
                                    "transition contains multiple mapping alternatives",
                                ));
                            }
                        }
                    }
                    let mapping_tag =
                        mapping_tag.ok_or_else(|| corrupt("transition has no mapping"))?;
                    let decoded =
                        pb::Transition::decode(raw).map_err(|e| corrupt(e.to_string()))?;
                    transitions.push(decode_transition(decoded, mapping_tag)?);
                }
                _ => {} // Unknown envelope fields do not participate in address resolution.
            }
        }
        let transitions = order_lineage(transitions)?;
        let consumers = transitions
            .iter()
            .enumerate()
            .flat_map(|(position, transition)| {
                transition
                    .sources()
                    .iter()
                    .map(move |source| (source.id as u32, position))
            })
            .collect();
        let producers = transitions
            .iter()
            .enumerate()
            .flat_map(|(position, transition)| {
                transition
                    .destinations()
                    .iter()
                    .map(move |destination| (destination.id as u32, position))
            })
            .collect();
        Ok(Self {
            transitions,
            consumers,
            producers,
        })
    }

    /// Find the transition consuming a fragment in expected constant time.
    /// The returned position indexes [`Self::transitions`]. Unaffected fragments return `None`.
    pub fn consumer(&self, fragment_id: u32) -> Option<usize> {
        self.consumers.get(&fragment_id).copied()
    }

    /// Whether a fragment occurs anywhere in the retained lineage.
    /// Destination coverage can still describe an index storing source addresses.
    pub fn contains_fragment(&self, fragment_id: u32) -> bool {
        self.consumers.contains_key(&fragment_id) || self.producers.contains_key(&fragment_id)
    }

    /// Rewrites ordered so every producer precedes its consumers.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }
}

impl DeepSizeOf for FragReuseLedger {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.consumers.deep_size_of_children(context)
            + self.producers.deep_size_of_children(context)
            + self.transitions.capacity() * std::mem::size_of::<Transition>()
            + self
                .transitions
                .iter()
                .map(|transition| {
                    (transition.sources.capacity() + transition.destinations.capacity())
                        * std::mem::size_of::<pb::FragmentDigest>()
                        + match &transition.mapping {
                            Mapping::OrderedCompaction(remap) => {
                                remap.deep_size_of_children(context)
                            }
                            Mapping::StablePartition(reference) => reference.map_id.capacity(),
                            Mapping::Unknown { .. } => 0,
                        }
                })
                .sum::<usize>()
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("FRI details", message)
}

// Keep length-delimited payloads as zero-copy slices of the original history.
fn next_field(input: &mut Bytes) -> Result<(u32, Option<Bytes>)> {
    let (tag, wire) = decode_key(input).map_err(|e| corrupt(e.to_string()))?;
    let payload = if wire == WireType::LengthDelimited {
        let length = decode_varint(input).map_err(|e| corrupt(e.to_string()))?;
        if length > input.remaining() as u64 {
            return Err(corrupt(format!(
                "field {tag} length {length} exceeds remaining {} bytes",
                input.remaining()
            )));
        }
        Some(input.split_to(length as usize))
    } else {
        skip_field(wire, tag, input, DecodeContext::default())
            .map_err(|e| corrupt(e.to_string()))?;
        None
    };
    Ok((tag, payload))
}

fn require_message(tag: u32, payload: Option<Bytes>) -> Result<Bytes> {
    payload.ok_or_else(|| corrupt(format!("field {tag} must be length-delimited")))
}

fn validate_digests(digests: &[pb::FragmentDigest], is_destination: bool) -> Result<u64> {
    let mut ids = RoaringBitmap::new();
    let mut live_rows = 0_u64;
    for digest in digests {
        if digest.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
            || digest.physical_rows > u32::MAX as u64
            || digest.num_deleted_rows > digest.physical_rows
        {
            return Err(corrupt(format!("invalid fragment digest {digest:?}")));
        }
        if !ids.insert(digest.id as u32) {
            return Err(corrupt(format!(
                "duplicate fragment {} in transition",
                digest.id
            )));
        }
        if is_destination && digest.num_deleted_rows != 0 {
            return Err(corrupt(format!(
                "destination fragment {} has creation-time deletions",
                digest.id
            )));
        }
        live_rows = live_rows
            .checked_add(digest.physical_rows - digest.num_deleted_rows)
            .ok_or_else(|| corrupt("fragment row count overflow"))?;
    }
    Ok(live_rows)
}

fn decode_transition(value: pb::Transition, mapping_tag: u32) -> Result<Transition> {
    if value.sources.is_empty() {
        return Err(corrupt("transition has no source fragments"));
    }
    let source_rows = validate_digests(&value.sources, false)?;
    let destination_rows = validate_digests(&value.destinations, true)?;
    if source_rows != destination_rows {
        return Err(corrupt(format!(
            "transition row counts differ: {source_rows} source rows, {destination_rows} destination rows"
        )));
    }
    let mapping = match value.encoding {
        Some(transition::Encoding::OrderedCompaction(ordered)) => {
            let mut cursor = Cursor::new(&ordered.changed_row_addrs);
            let bitmap = RoaringTreemap::deserialize_from(&mut cursor)
                .map_err(|e| corrupt(e.to_string()))?;
            if cursor.position() != ordered.changed_row_addrs.len() as u64 {
                return Err(corrupt("trailing bytes in ordered compaction bitmap"));
            }
            for source in &value.sources {
                let survivors =
                    bitmap.range_cardinality(RowAddress::address_range(source.id as u32));
                if survivors != source.physical_rows - source.num_deleted_rows {
                    return Err(corrupt(format!(
                        "fragment {} bitmap has {survivors} survivors inconsistent with its digest",
                        source.id
                    )));
                }
            }
            let layout = |fragments: &[pb::FragmentDigest]| {
                fragments
                    .iter()
                    .map(|f| (f.id as u32, f.physical_rows as u32))
                    .collect()
            };
            let remap = RowAddrRemap::compact_with_layout([GroupInputWithLayout {
                rewritten_old_row_addrs: bitmap,
                old_frags: layout(&value.sources),
                new_frags: layout(&value.destinations),
            }])
            .map_err(|e| corrupt(e.to_string()))?;
            Mapping::OrderedCompaction(remap)
        }
        Some(transition::Encoding::StablePartition(partition)) => {
            Uuid::parse_str(&partition.map_id).map_err(|e| {
                corrupt(format!(
                    "invalid stable partition map_id {:?}: {e}",
                    partition.map_id
                ))
            })?;
            if partition.map_size_bytes == 0 {
                return Err(corrupt("stable partition map_size_bytes must be positive"));
            }
            Mapping::StablePartition(partition)
        }
        None => Mapping::Unknown {
            field_number: mapping_tag,
        },
    };
    Ok(Transition {
        sources: value.sources,
        destinations: value.destinations,
        mapping,
    })
}

fn order_lineage(transitions: Vec<Transition>) -> Result<Vec<Transition>> {
    let mut producers = HashMap::new();
    let mut consumers = RoaringBitmap::new();
    for (index, transition) in transitions.iter().enumerate() {
        for destination in &transition.destinations {
            if producers
                .insert(destination.id, (index, destination.physical_rows))
                .is_some()
            {
                return Err(corrupt(format!(
                    "duplicate producer for fragment {}",
                    destination.id
                )));
            }
        }
        for source in &transition.sources {
            if !consumers.insert(source.id as u32) {
                return Err(corrupt(format!(
                    "duplicate consumer for fragment {}",
                    source.id
                )));
            }
        }
    }
    let mut incoming = vec![0; transitions.len()];
    let mut outgoing = vec![Vec::new(); transitions.len()];
    for (consumer, transition) in transitions.iter().enumerate() {
        for source in &transition.sources {
            if let Some(&(producer, physical_rows)) = producers.get(&source.id) {
                if source.physical_rows != physical_rows {
                    return Err(corrupt(format!(
                        "inconsistent physical_rows for fragment {}",
                        source.id
                    )));
                }
                incoming[consumer] += 1;
                outgoing[producer].push(consumer);
            }
        }
    }
    let mut ready: VecDeque<_> = incoming
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n == 0).then_some(i))
        .collect();
    let mut ordered = Vec::with_capacity(transitions.len());
    let mut transitions: Vec<_> = transitions.into_iter().map(Some).collect();
    while let Some(index) = ready.pop_front() {
        let transition = transitions[index]
            .take()
            .ok_or_else(|| corrupt("lineage visited a transition twice"))?;
        ordered.push(transition);
        for &consumer in &outgoing[index] {
            incoming[consumer] -= 1;
            if incoming[consumer] == 0 {
                ready.push_back(consumer);
            }
        }
    }
    if ordered.len() != transitions.len() {
        return Err(corrupt("fragment lineage contains a cycle"));
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn digest(id: u64, rows: u64, deleted: u64) -> pb::FragmentDigest {
        pb::FragmentDigest {
            id,
            physical_rows: rows,
            num_deleted_rows: deleted,
        }
    }

    fn partition(source: u64, destination: u64) -> pb::Transition {
        pb::Transition {
            sources: vec![digest(source, 2, 0)],
            destinations: vec![digest(destination, 2, 0)],
            encoding: Some(transition::Encoding::StablePartition(pb::StablePartition {
                map_id: Uuid::nil().to_string(),
                map_size_bytes: 100,
                base_id: Some(7),
            })),
        }
    }

    fn history(transitions: Vec<pb::Transition>) -> Bytes {
        pb::InlineContent {
            legacy_versions: vec![],
            transitions,
        }
        .encode_to_vec()
        .into()
    }

    fn message_field(tag: u32, payload: &[u8], output: &mut Vec<u8>) {
        prost::encoding::encode_key(tag, WireType::LengthDelimited, output);
        prost::encoding::encode_varint(payload.len() as u64, output);
        output.extend_from_slice(payload);
    }

    fn ordered(
        sources: Vec<pb::FragmentDigest>,
        destinations: Vec<pb::FragmentDigest>,
        addresses: &[u64],
    ) -> pb::Transition {
        let bitmap: RoaringTreemap = addresses.iter().copied().collect();
        let mut changed_row_addrs = Vec::new();
        bitmap.serialize_into(&mut changed_row_addrs).unwrap();
        pb::Transition {
            sources,
            destinations,
            encoding: Some(transition::Encoding::OrderedCompaction(
                pb::OrderedCompaction { changed_row_addrs },
            )),
        }
    }

    fn address(fragment: u32, offset: u32) -> u64 {
        RowAddress::new_from_parts(fragment, offset).into()
    }

    fn assert_corrupt(result: Result<FragReuseLedger>, message: &str) {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn mixed_history_uses_lineage_and_preserves_source_order() {
        let old = ordered(
            vec![digest(2, 3, 1), digest(1, 1, 0)],
            vec![digest(3, 3, 0)],
            &[address(2, 0), address(2, 2), address(1, 0)],
        );
        let Some(transition::Encoding::OrderedCompaction(mapping)) = old.encoding else {
            unreachable!()
        };
        let mut next = partition(3, 4);
        next.sources[0].physical_rows = 3;
        next.destinations[0].physical_rows = 3;
        let content: Bytes = pb::InlineContent {
            legacy_versions: vec![pb::Version {
                dataset_version: 100,
                groups: vec![pb::Group {
                    old_fragments: old.sources,
                    new_fragments: old.destinations,
                    changed_row_addrs: mapping.changed_row_addrs,
                }],
            }],
            transitions: vec![next],
        }
        .encode_to_vec()
        .into();
        let ledger = FragReuseLedger::decode(1, content).unwrap();
        for (position, transition) in ledger.transitions().iter().enumerate() {
            for source in transition.sources() {
                assert_eq!(ledger.consumer(source.id as u32), Some(position));
            }
        }
        assert_eq!(ledger.consumer(u32::MAX), None);
        assert!(!ledger.contains_fragment(u32::MAX));
        for transition in ledger.transitions() {
            for destination in transition.destinations() {
                assert!(ledger.contains_fragment(destination.id as u32));
            }
        }

        assert_eq!(ledger.transitions().len(), 2);
        let Mapping::OrderedCompaction(remap) = ledger.transitions()[0].mapping() else {
            unreachable!()
        };
        assert_eq!(remap.get(address(2, 0)), Some(Some(address(3, 0))));
        assert_eq!(remap.get(address(2, 1)), Some(None));
        assert_eq!(remap.get(address(2, 2)), Some(Some(address(3, 1))));
        assert_eq!(remap.get(address(1, 0)), Some(Some(address(3, 2))));
        assert_eq!(remap.get(address(9, 0)), None);
        let Mapping::StablePartition(reference) = ledger.transitions()[1].mapping() else {
            unreachable!()
        };
        assert_eq!(reference.base_id, Some(7));
        assert_eq!(reference.map_size_bytes, 100);
    }

    #[test]
    fn unknown_mapping_retains_common_metadata_and_lineage() {
        let mut unknown = partition(1, 2);
        unknown.encoding = None;
        let mut raw = unknown.encode_to_vec();
        message_field(17, b"future external mapping", &mut raw);
        let mut content = history(vec![partition(2, 3), partition(10, 11)]).to_vec();
        message_field(2, &raw, &mut content);
        message_field(19, b"future envelope metadata", &mut content);
        let content = Bytes::from(content);
        let ledger = FragReuseLedger::decode(1, content).unwrap();
        let nodes = ledger.transitions();
        let unknown_index = nodes
            .iter()
            .position(|t| matches!(t.mapping(), Mapping::Unknown { field_number: 17 }))
            .unwrap();
        let dependent_index = nodes.iter().position(|t| t.sources()[0].id == 2).unwrap();
        assert!(unknown_index < dependent_index);
        assert_eq!(nodes[unknown_index].sources()[0].id, 1);
        assert_eq!(nodes[unknown_index].destinations()[0].id, 2);
    }

    #[rstest]
    #[case::duplicate_producer(vec![partition(1, 3), partition(2, 3)], "duplicate producer")]
    #[case::duplicate_consumer(vec![partition(1, 2), partition(1, 3)], "duplicate consumer")]
    #[case::cycle(vec![partition(1, 2), partition(2, 1)], "cycle")]
    #[case::self_cycle(vec![partition(1, 1)], "cycle")]
    fn rejects_invalid_lineage(#[case] transitions: Vec<pb::Transition>, #[case] message: &str) {
        assert_corrupt(FragReuseLedger::decode(1, history(transitions)), message);
    }

    #[test]
    fn multi_fragment_edges_and_intervening_deletes() {
        let mut first = partition(1, 2);
        first.sources = vec![digest(1, 4, 0)];
        first.destinations = vec![digest(2, 2, 0), digest(3, 2, 0)];
        let second = ordered(
            vec![digest(3, 2, 1), digest(2, 2, 0)],
            vec![digest(4, 3, 0)],
            &[address(3, 1), address(2, 0), address(2, 1)],
        );
        let ledger = FragReuseLedger::decode(1, history(vec![second, first])).unwrap();
        assert_eq!(ledger.transitions()[0].sources()[0].id, 1);
        assert_eq!(ledger.transitions()[1].sources()[0].id, 3);
    }

    #[rstest]
    #[case::missing(vec![], "no mapping")]
    #[case::duplicate_known(vec![6, 6], "multiple mapping")]
    #[case::conflicting_known(vec![5, 6], "multiple mapping")]
    #[case::known_unknown(vec![6, 9], "multiple mapping")]
    #[case::duplicate_unknown(vec![9, 9], "multiple mapping")]
    fn rejects_ambiguous_mapping(#[case] tags: Vec<u32>, #[case] message: &str) {
        let mut transition = partition(1, 2);
        transition.encoding = None;
        let mut raw = transition.encode_to_vec();
        for tag in tags {
            message_field(tag, &[], &mut raw);
        }
        let mut content = Vec::new();
        message_field(2, &raw, &mut content);
        assert_corrupt(FragReuseLedger::decode(1, content.into()), message);
    }

    #[rstest]
    #[case::truncated(vec![0x12, 10, 0], "exceeds remaining")]
    #[case::wrong_transition_wire(vec![0x10, 0], "length-delimited")]
    #[case::wrong_mapping_wire(vec![0x12, 2, 0x48, 0], "length-delimited")]
    #[case::invalid_key(vec![0], "invalid tag")]
    fn rejects_malformed_wire(#[case] content: Vec<u8>, #[case] message: &str) {
        assert_corrupt(FragReuseLedger::decode(1, content.into()), message);
    }

    #[rstest]
    #[case::invalid_id(digest(u32::MAX as u64, 2, 0), "invalid fragment digest")]
    #[case::invalid_rows(digest(1, u32::MAX as u64 + 1, 0), "invalid fragment digest")]
    #[case::invalid_deletions(digest(1, 2, 3), "invalid fragment digest")]
    #[case::row_conservation(digest(1, 3, 0), "row counts differ")]
    fn rejects_invalid_digests(#[case] source: pb::FragmentDigest, #[case] message: &str) {
        let mut transition = partition(1, 2);
        transition.sources = vec![source];
        assert_corrupt(
            FragReuseLedger::decode(1, history(vec![transition])),
            message,
        );
    }

    #[test]
    fn rejects_inconsistent_lineage_counts() {
        let mut second = partition(2, 3);
        second.sources[0] = digest(2, 3, 1);
        assert_corrupt(
            FragReuseLedger::decode(1, history(vec![partition(1, 2), second])),
            "inconsistent physical_rows",
        );
    }

    #[test]
    fn rejects_bad_bitmap_and_reference() {
        let transition = ordered(
            vec![digest(1, 2, 1)],
            vec![digest(2, 1, 0)],
            &[address(1, 2)],
        );
        assert_corrupt(
            FragReuseLedger::decode(1, history(vec![transition])),
            "outside",
        );
        let transition = ordered(
            vec![digest(1, 2, 1)],
            vec![digest(2, 1, 0)],
            &[address(9, 0)],
        );
        assert_corrupt(
            FragReuseLedger::decode(1, history(vec![transition])),
            "survivors",
        );
        let mut transition = partition(1, 2);
        let Some(transition::Encoding::StablePartition(reference)) = &mut transition.encoding
        else {
            unreachable!()
        };
        reference.map_id = "../escape".into();
        assert_corrupt(
            FragReuseLedger::decode(1, history(vec![transition])),
            "map_id",
        );
    }

    #[test]
    fn all_deleted_source_has_no_destinations() {
        let transition = ordered(vec![digest(1, 2, 2)], vec![], &[]);
        let ledger = FragReuseLedger::decode(1, history(vec![transition])).unwrap();
        let Mapping::OrderedCompaction(remap) = ledger.transitions()[0].mapping() else {
            unreachable!()
        };
        assert_eq!(remap.get(address(1, 0)), Some(None));
    }

    #[test]
    fn version_gate() {
        assert!(
            FragReuseLedger::decode(0, Bytes::new())
                .unwrap()
                .transitions()
                .is_empty()
        );
        let error = FragReuseLedger::decode(2, Bytes::new()).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("index_version 2"));
        assert_corrupt(
            FragReuseLedger::decode(0, history(vec![partition(1, 2)])),
            "index_version 1",
        );
    }
}
