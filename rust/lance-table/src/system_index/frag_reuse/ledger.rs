// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata-only reader for the unified fragment reuse history.
//!
//! Decoding validates lineage without opening row-map files. Unsupported index
//! versions are rejected before interpreting their content.
//! Operations that carry an index forward keep its original `Any` separately.
//!
//! ```
//! use lance_table::system_index::frag_reuse::ledger::FragReuseLedger;
//! use prost::Message;
//! use lance_table::format::pb;
//!
//! # async fn example() -> lance_core::Result<()> {
//! let details = prost_types::Any {
//!     type_url: "/lance.table.FragmentReuseIndexDetails".into(),
//!     value: pb::FragmentReuseIndexDetails {
//!         content: Some(pb::fragment_reuse_index_details::Content::Inline(Default::default())),
//!     }.encode_to_vec(),
//! };
//! let ledger = FragReuseLedger::decode(1, &details, |_| async {
//!     Err(lance_core::Error::not_supported("external content is unavailable"))
//! }).await?;
//! assert!(ledger.transitions().is_empty());
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, VecDeque};
use std::future::Future;
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

    /// Mapping semantics for a supported index version.
    pub fn mapping(&self) -> &Mapping {
        &self.mapping
    }
}

/// Supported transitions in fragment-lineage order, independent of serialization order.
#[derive(Debug)]
pub struct FragReuseLedger {
    transitions: Vec<Transition>,
    consumers: HashMap<u32, usize>,
    producers: HashMap<u32, usize>,
    has_unsupported_transitions: bool,
}

impl FragReuseLedger {
    /// Decode FRI details, resolving external history through `read_external`.
    /// The callback reads the exact byte range described by the external reference;
    /// it is never used for inline history or stable-partition row maps.
    /// Index versions 0 and 1 are supported. Unsupported versions require an upgrade.
    pub async fn decode<F, Fut>(
        index_version: i32,
        details: &prost_types::Any,
        read_external: F,
    ) -> Result<Self>
    where
        F: FnOnce(crate::format::pb::ExternalFile) -> Fut,
        Fut: Future<Output = Result<Bytes>>,
    {
        validate_index_version(index_version)?;
        if details.type_url.rsplit('/').next() != Some("lance.table.FragmentReuseIndexDetails") {
            return Err(corrupt(format!(
                "unexpected FRI details type {:?}",
                details.type_url
            )));
        }
        let mut wire = Bytes::copy_from_slice(&details.value);
        let mut content = None;
        while wire.has_remaining() {
            let (tag, payload) = next_field(&mut wire)?;
            if matches!(tag, 1 | 2) {
                let payload = require_message(tag, payload)?;
                if content.replace((tag, payload)).is_some() {
                    return Err(corrupt("multiple FRI content fields"));
                }
            }
        }
        let (tag, content) = content.ok_or_else(|| corrupt("missing FRI content"))?;
        let content = if tag == 1 {
            content
        } else {
            let file = crate::format::pb::ExternalFile::decode(content)
                .map_err(|e| corrupt(e.to_string()))?;
            file.offset
                .checked_add(file.size)
                .and_then(|end| usize::try_from(end).ok())
                .ok_or_else(|| corrupt("external FRI range overflow"))?;
            let expected = file.size;
            let bytes = read_external(file).await?;
            if bytes.len() as u64 != expected {
                return Err(corrupt(format!(
                    "external FRI size mismatch: expected {expected}, received {}",
                    bytes.len()
                )));
            }
            bytes
        };
        Self::decode_content(index_version, content)
    }

    // Each legacy group owns a compaction remap. Address resolution follows its
    // fragment lineage, rather than searching every group in a legacy version.
    fn decode_content(index_version: i32, content: Bytes) -> Result<Self> {
        validate_index_version(index_version)?;
        let mut remaining = content;
        let mut transitions = Vec::new();
        let mut has_unsupported_transitions = false;
        while remaining.has_remaining() {
            let (tag, payload) = next_field(&mut remaining)?;
            match tag {
                1 => {
                    let version = pb::Version::decode(require_message(tag, payload)?)
                        .map_err(|e| corrupt(e.to_string()))?;
                    for group in version.groups {
                        transitions.push(decode_transition(pb::Transition {
                            sources: group.old_fragments,
                            destinations: group.new_fragments,
                            mapping: Some(transition::Mapping::OrderedCompaction(
                                pb::OrderedCompaction {
                                    changed_row_addrs: group.changed_row_addrs,
                                },
                            )),
                        })?);
                    }
                }
                2 => {
                    if index_version == 0 {
                        return Err(corrupt("tagged transitions require FRI index_version 1"));
                    }
                    let raw = require_message(tag, payload)?;
                    let mut fields = raw.clone();
                    let mut has_mapping = false;
                    let mut has_unknown_fields = false;
                    while fields.has_remaining() {
                        let (tag, payload) = next_field(&mut fields)?;
                        if matches!(tag, ORDERED_COMPACTION_FIELD | STABLE_PARTITION_FIELD) {
                            require_message(tag, payload)?;
                            if has_mapping {
                                return Err(corrupt(
                                    "transition contains multiple mapping alternatives",
                                ));
                            }
                            has_mapping = true;
                        } else if !matches!(tag, 1 | 2) {
                            has_unknown_fields = true;
                        }
                    }
                    if has_unknown_fields {
                        // The field number does not tell us whether this is metadata
                        // or a mapping. Neither is safe to partially interpret.
                        has_unsupported_transitions = true;
                        continue;
                    }
                    if !has_mapping {
                        return Err(corrupt("transition has no mapping"));
                    }
                    let decoded =
                        pb::Transition::decode(raw).map_err(|e| corrupt(e.to_string()))?;
                    transitions.push(decode_transition(decoded)?);
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
            has_unsupported_transitions,
        })
    }

    /// Whether decoding omitted transitions this implementation cannot interpret.
    /// Such a ledger is insufficient for maintenance; carrying the original
    /// serialized details forward without interpreting them is still possible.
    pub fn has_unsupported_transitions(&self) -> bool {
        self.has_unsupported_transitions
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
                        }
                })
                .sum::<usize>()
    }
}

// Exact protobuf alternatives, not the start of a reserved field-number range.
const ORDERED_COMPACTION_FIELD: u32 = 3;
const STABLE_PARTITION_FIELD: u32 = 4;

fn validate_index_version(index_version: i32) -> Result<()> {
    if !matches!(index_version, 0 | 1) {
        return Err(Error::not_supported(format!(
            "Unsupported FRI index_version {index_version}; supported versions are 0 and 1. Please upgrade to a newer version of Lance."
        )));
    }
    Ok(())
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

fn decode_transition(value: pb::Transition) -> Result<Transition> {
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
    let mapping = match value.mapping {
        Some(transition::Mapping::OrderedCompaction(ordered)) => {
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
        Some(transition::Mapping::StablePartition(partition)) => {
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
        None => return Err(corrupt("transition has no mapping")),
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
        // Each transition is queued once, when its incoming count reaches zero.
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
    use crate::system_index::frag_reuse::{
        CompactFragReuseIndex, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
    };
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
            mapping: Some(transition::Mapping::StablePartition(pb::StablePartition {
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
            mapping: Some(transition::Mapping::OrderedCompaction(
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
        let Some(transition::Mapping::OrderedCompaction(mapping)) = old.mapping else {
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
        let ledger = FragReuseLedger::decode_content(1, content).unwrap();
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
        assert!(!ledger.has_unsupported_transitions());
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
    fn unknown_fields_do_not_partially_apply_a_known_mapping() {
        let mut raw = partition(1, 2).encode_to_vec();
        message_field(17, b"future metadata", &mut raw);
        prost::encoding::encode_key(7, WireType::Varint, &mut raw);
        prost::encoding::encode_varint(100, &mut raw);
        let mut content = Vec::new();
        message_field(2, &raw, &mut content);
        let ledger = FragReuseLedger::decode_content(1, content.into()).unwrap();
        assert!(ledger.transitions().is_empty());
        assert!(ledger.has_unsupported_transitions());
    }

    #[test]
    fn unknown_mapping_is_omitted_without_discarding_supported_transitions() {
        let mut unknown = partition(2, 3);
        unknown.mapping = None;
        let mut raw = unknown.encode_to_vec();
        message_field(17, b"future mapping", &mut raw);
        let mut content =
            history(vec![partition(3, 4), partition(1, 2), partition(10, 11)]).to_vec();
        message_field(2, &raw, &mut content);
        let ledger = FragReuseLedger::decode_content(1, content.into()).unwrap();
        assert!(ledger.has_unsupported_transitions());
        assert_eq!(ledger.transitions().len(), 3);
        assert!(ledger.consumer(1).is_some());
        assert_eq!(ledger.consumer(2), None);
        assert!(ledger.consumer(3).is_some());
        assert!(ledger.consumer(10).is_some());
    }

    #[rstest]
    #[case::duplicate_producer(vec![partition(1, 3), partition(2, 3)], "duplicate producer")]
    #[case::duplicate_consumer(vec![partition(1, 2), partition(1, 3)], "duplicate consumer")]
    #[case::cycle(vec![partition(1, 2), partition(2, 1)], "cycle")]
    #[case::self_cycle(vec![partition(1, 1)], "cycle")]
    fn rejects_invalid_lineage(#[case] transitions: Vec<pb::Transition>, #[case] message: &str) {
        assert_corrupt(
            FragReuseLedger::decode_content(1, history(transitions)),
            message,
        );
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
        let ledger = FragReuseLedger::decode_content(1, history(vec![second, first])).unwrap();
        assert_eq!(ledger.transitions()[0].sources()[0].id, 1);
        assert_eq!(ledger.transitions()[1].sources()[0].id, 3);
        let Mapping::OrderedCompaction(remap) = ledger.transitions()[1].mapping() else {
            unreachable!()
        };
        assert_eq!(remap.get(address(3, 0)), Some(None));
        for (source, destination) in [(address(3, 1), 0), (address(2, 0), 1), (address(2, 1), 2)] {
            assert_eq!(remap.get(source), Some(Some(address(4, destination))));
        }
    }

    #[rstest]
    #[case::missing(vec![], "no mapping")]
    #[case::duplicate_known(vec![4, 4], "multiple mapping")]
    #[case::conflicting_known(vec![3, 4], "multiple mapping")]
    fn rejects_ambiguous_mapping(#[case] tags: Vec<u32>, #[case] message: &str) {
        let mut transition = partition(1, 2);
        transition.mapping = None;
        let mut raw = transition.encode_to_vec();
        for tag in tags {
            message_field(tag, &[], &mut raw);
        }
        let mut content = Vec::new();
        message_field(2, &raw, &mut content);
        assert_corrupt(FragReuseLedger::decode_content(1, content.into()), message);
    }

    #[rstest]
    #[case::truncated(vec![0x12, 10, 0], "exceeds remaining")]
    #[case::wrong_transition_wire(vec![0x10, 0], "length-delimited")]
    #[case::wrong_mapping_wire(vec![0x12, 2, 0x20, 0], "length-delimited")]
    #[case::invalid_key(vec![0], "invalid tag")]
    fn rejects_malformed_wire(#[case] content: Vec<u8>, #[case] message: &str) {
        assert_corrupt(FragReuseLedger::decode_content(1, content.into()), message);
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
            FragReuseLedger::decode_content(1, history(vec![transition])),
            message,
        );
    }

    #[test]
    fn rejects_inconsistent_lineage_counts() {
        let mut second = partition(2, 3);
        second.sources[0] = digest(2, 3, 1);
        assert_corrupt(
            FragReuseLedger::decode_content(1, history(vec![partition(1, 2), second])),
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
            FragReuseLedger::decode_content(1, history(vec![transition])),
            "outside",
        );
        let transition = ordered(
            vec![digest(1, 2, 1)],
            vec![digest(2, 1, 0)],
            &[address(9, 0)],
        );
        assert_corrupt(
            FragReuseLedger::decode_content(1, history(vec![transition])),
            "survivors",
        );
        let mut transition = partition(1, 2);
        let Some(transition::Mapping::StablePartition(reference)) = &mut transition.mapping else {
            unreachable!()
        };
        reference.map_id = "../escape".into();
        assert_corrupt(
            FragReuseLedger::decode_content(1, history(vec![transition])),
            "map_id",
        );
    }

    #[test]
    fn all_deleted_source_has_no_destinations() {
        let transition = ordered(vec![digest(1, 2, 2)], vec![], &[]);
        let ledger = FragReuseLedger::decode_content(1, history(vec![transition])).unwrap();
        let Mapping::OrderedCompaction(remap) = ledger.transitions()[0].mapping() else {
            unreachable!()
        };
        assert_eq!(remap.get(address(1, 0)), Some(None));
    }

    #[test]
    fn version_gate() {
        assert!(
            FragReuseLedger::decode_content(0, Bytes::new())
                .unwrap()
                .transitions()
                .is_empty()
        );
        let error = FragReuseLedger::decode_content(2, Bytes::new()).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("index_version 2"));
        assert_corrupt(
            FragReuseLedger::decode_content(0, history(vec![partition(1, 2)])),
            "index_version 1",
        );
    }
    #[rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn legacy_writer_round_trip_and_remap_equivalence(#[case] external: bool) {
        let group = |sources, destinations, addresses: &[u64]| {
            let transition = ordered(sources, destinations, addresses);
            let Some(transition::Mapping::OrderedCompaction(mapping)) = transition.mapping else {
                unreachable!()
            };
            FragReuseGroup::try_from(pb::Group {
                old_fragments: transition.sources,
                new_fragments: transition.destinations,
                changed_row_addrs: mapping.changed_row_addrs,
            })
            .unwrap()
        };
        let details = FragReuseIndexDetails {
            versions: vec![
                FragReuseVersion {
                    dataset_version: 10,
                    groups: vec![
                        group(
                            vec![digest(2, 3, 1), digest(1, 1, 0)],
                            vec![digest(3, 3, 0)],
                            &[address(2, 0), address(2, 2), address(1, 0)],
                        ),
                        group(
                            vec![digest(10, 1, 0)],
                            vec![digest(11, 1, 0)],
                            &[address(10, 0)],
                        ),
                    ],
                },
                FragReuseVersion {
                    dataset_version: 11,
                    groups: vec![group(
                        vec![digest(3, 3, 1)],
                        vec![digest(4, 2, 0)],
                        &[address(3, 0), address(3, 2)],
                    )],
                },
            ],
        };
        // Use the same serializer as the index-version-0 writer.
        let content = pb::InlineContent::from(&details).encode_to_vec();
        let old = CompactFragReuseIndex::try_new(Uuid::nil(), details).unwrap();
        let file = crate::format::pb::ExternalFile {
            path: "details.binpb".into(),
            offset: 7,
            size: content.len() as u64,
        };
        let mut envelope = Vec::new();
        if external {
            message_field(2, &file.encode_to_vec(), &mut envelope);
        } else {
            message_field(1, &content, &mut envelope);
        }
        let any = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: envelope,
        };
        let ledger = FragReuseLedger::decode(1, &any, |actual| async move {
            assert!(external);
            assert_eq!(actual, file);
            Ok(content.into())
        })
        .await
        .unwrap();
        assert_eq!(ledger.transitions().len(), 3);
        for source in [
            address(2, 0),
            address(2, 1),
            address(2, 2),
            address(1, 0),
            address(10, 0),
            address(99, 0),
        ] {
            let mut translated = Some(source);
            while let Some(current) = translated {
                let fragment = RowAddress::from(current).fragment_id();
                let Some(index) = ledger.consumer(fragment) else {
                    break;
                };
                let Mapping::OrderedCompaction(remap) = ledger.transitions()[index].mapping()
                else {
                    unreachable!()
                };
                translated = remap.get(current).unwrap();
            }
            assert_eq!(translated, old.remap_row_id(source), "address {source}");
        }
    }

    #[test]
    fn independent_transitions_keep_serialized_order() {
        let ledger =
            FragReuseLedger::decode_content(1, history(vec![partition(10, 11), partition(1, 2)]))
                .unwrap();
        assert_eq!(
            ledger
                .transitions()
                .iter()
                .map(|t| t.sources()[0].id)
                .collect::<Vec<_>>(),
            vec![10, 1]
        );
    }

    #[rstest]
    #[case::missing(vec![], "missing FRI content")]
    #[case::duplicate(vec![0x0a, 0, 0x0a, 0], "multiple FRI content")]
    #[case::conflicting(vec![0x0a, 0, 0x12, 0], "multiple FRI content")]
    #[case::truncated(vec![0x0a, 10, 0], "exceeds remaining")]
    #[case::oversized(vec![0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01], "exceeds remaining")]
    #[tokio::test]
    async fn rejects_invalid_envelope(#[case] value: Vec<u8>, #[case] message: &str) {
        let any = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value,
        };
        let result = FragReuseLedger::decode(1, &any, |_| async {
            panic!("invalid envelope must not read external content")
        })
        .await;
        assert_corrupt(result, message);
    }

    #[tokio::test]
    async fn version_gate_precedes_envelope_and_io() {
        let error = FragReuseLedger::decode(2, &prost_types::Any::default(), |_| async {
            panic!("unsupported version must not read external content")
        })
        .await
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("Please upgrade"));
    }

    #[rstest]
    #[case::overflow(u64::MAX, 1, "range overflow")]
    #[case::short_read(0, 2, "size mismatch")]
    #[tokio::test]
    async fn rejects_invalid_external_range(
        #[case] offset: u64,
        #[case] size: u64,
        #[case] message: &str,
    ) {
        let file = crate::format::pb::ExternalFile {
            path: "details.binpb".into(),
            offset,
            size,
        };
        let mut value = Vec::new();
        message_field(2, &file.encode_to_vec(), &mut value);
        let details = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value,
        };
        let result = FragReuseLedger::decode(1, &details, |_| async move {
            assert_ne!(offset, u64::MAX, "overflow must be rejected before IO");
            Ok(Bytes::from_static(&[0]))
        })
        .await;
        assert_corrupt(result, message);
    }

    proptest::proptest! {
        #[test]
        fn arbitrary_inline_bytes_do_not_panic(raw in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..4096)) {
            let _ = FragReuseLedger::decode_content(1, raw.into());
        }

        #[test]
        fn arbitrary_envelopes_do_not_panic(raw in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..4096)) {
            let any = prost_types::Any { type_url: "/lance.table.FragmentReuseIndexDetails".into(), value: raw };
            let _ = futures::executor::block_on(FragReuseLedger::decode(1, &any, |_| async {
                Err(Error::not_supported("no external data in parser property test"))
            }));
        }
    }
}
