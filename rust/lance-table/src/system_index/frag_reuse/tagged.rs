// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The tagged (`index_version` 1) fragment reuse history: compaction versions
//! and tagged transitions. Every writer of this format validates it, so a
//! tagged history is never repaired.

use std::collections::HashSet;
use std::sync::Arc;

use bytes::{Buf, Bytes, BytesMut};
use lance_core::{Error, Result};
use prost::Message;
use prost::encoding::{WireType, decode_key, decode_varint};
use roaring::RoaringBitmap;

use super::history::{HistoryFormat, Record, TRANSITION_FIELD, length_delimited};
use super::imp::{FriImpl, Records, StoredRecord, concat, invalid_edit};
use super::ledger::{STABLE_PARTITION_FIELD, corrupt, next_field, require_message};
use super::plan::{
    ReadContext, ReadPlan, SegmentSummary, TranslationPlan, TranslationSupport, segment_plans,
};
use crate::format::pb::fragment_reuse_index_details::{self as pb, transition};

/// `StablePartition.base_id`.
const STABLE_PARTITION_BASE_ID_FIELD: u32 = 3;

#[derive(Debug, Default)]
pub struct TaggedHistory(Records);

impl TaggedHistory {
    pub fn decode(content: Bytes) -> Result<Self> {
        let records = Records::decode(HistoryFormat::Tagged, content)?;
        validate_records(&records.stored)?;
        Ok(Self(records))
    }
}

impl TaggedHistory {
    /// Whether the segment's stored addresses are usable as they are: every
    /// fragment it covers is live and it is caught up with the history.
    /// Dead fragments may belong to an omitted unknown mapping; those must
    /// not get an identity read merely because they are absent from the
    /// graph.
    fn is_direct(&self, context: &ReadContext, segment: &SegmentSummary) -> bool {
        segment
            .fragment_bitmap
            .as_ref()
            .is_some_and(|bitmap| bitmap.is_subset(context.live_fragments))
            && self.is_caught_up(context.entry_version, segment)
    }
}

impl FriImpl for TaggedHistory {
    fn format(&self) -> HistoryFormat {
        HistoryFormat::Tagged
    }

    fn records(&self) -> &Records {
        &self.0
    }

    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>> {
        Ok(Arc::new(Self::decode(content)?))
    }

    /// Every fragment of the retained lineage, sources included, so a rewrite
    /// of a fragment the history still translates from is seen as covered.
    fn fragment_bitmap(&self) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        let mut insert = |digests: &[pb::FragmentDigest]| {
            bitmap.extend(digests.iter().map(|digest| digest.id as u32));
        };
        for stored in &self.0.stored {
            match &stored.record {
                Record::LegacyVersion(version) => {
                    for group in &version.groups {
                        insert(&group.old_fragments);
                        insert(&group.new_fragments);
                    }
                }
                Record::Transition(transition) => {
                    insert(&transition.sources);
                    insert(&transition.destinations);
                }
                Record::Uninterpretable { .. } => {}
            }
        }
        bitmap
    }

    fn into_tagged(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(self)
    }

    fn append_transitions(&self, transitions: &[pb::Transition]) -> Result<Arc<dyn FriImpl>> {
        self.0
            .require_fully_interpretable("append transitions to")?;
        let mut content = BytesMut::from(concat(&self.0.stored));
        for transition in transitions {
            content.extend_from_slice(&length_delimited(
                TRANSITION_FIELD,
                &transition.encode_to_vec(),
            ));
        }
        self.rebuilt(content.freeze()).map_err(invalid_edit)
    }

    fn insert_legacy_version(&self, _version: pb::Version) -> Result<Arc<dyn FriImpl>> {
        Err(Error::invalid_input(
            "a compaction version cannot be added to a tagged fragment reuse history; append \
             transitions instead",
        ))
    }

    fn remap_stable_partition_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0.require_fully_interpretable("relocate")?;
        let mut records = Vec::with_capacity(self.0.stored.len());
        for stored in &self.0.stored {
            let raw = match &stored.record {
                Record::Transition(pb::Transition {
                    mapping: Some(transition::Mapping::StablePartition(_)),
                    ..
                }) => {
                    let mut field = stored.raw.clone();
                    let (tag, payload) = next_field(&mut field)?;
                    let patched =
                        patch_transition_stable_partition(&require_message(tag, payload)?, remap)?;
                    length_delimited(TRANSITION_FIELD, &patched).into()
                }
                _ => stored.raw.clone(),
            };
            records.push(StoredRecord {
                raw,
                record: stored.record.clone(),
            });
        }
        self.rebuilt(concat(&records))
    }

    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Err(Error::not_supported(
            "a tagged fragment reuse history cannot describe a table with stable row IDs, which \
             supports only index_version 0",
        ))
    }

    /// A projected bitmap can name only destinations while the files still
    /// hold source addresses, so a segment is caught up only when it covers
    /// no fragment of the lineage. A dropped unknown transition leaves no
    /// record of which fragments it touched, so no segment can be proven
    /// untouched by it.
    fn is_caught_up(&self, _entry_version: u64, segment: &SegmentSummary) -> bool {
        let ledger = &self.0.ledger;
        !ledger.has_unsupported_transitions()
            && segment.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                bitmap
                    .iter()
                    .all(|fragment| !ledger.contains_fragment(fragment))
            })
    }

    fn needs_translation_support(&self, context: &ReadContext, segment: &SegmentSummary) -> bool {
        !self.is_direct(context, segment)
    }

    fn plan(&self, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        let ledger = &self.0.ledger;
        let mut plans = vec![ReadPlan::Excluded; group.len()];
        // Only segments the reader can use establish coverage: a destination
        // needs all of its contributing sources.
        let mut members = Vec::with_capacity(group.len());
        let mut provenance = Vec::with_capacity(group.len());
        for (position, segment) in group.iter().enumerate() {
            let Some(bitmap) = &segment.fragment_bitmap else {
                continue;
            };
            if ledger.has_unsupported_transitions()
                || (!self.is_direct(context, segment)
                    && segment.translation != TranslationSupport::Supported)
            {
                continue;
            }
            members.push(position);
            provenance.push(bitmap.clone());
        }
        let parts = segment_plans(ledger, context.live_fragments, &provenance);
        for ((position, bitmap), parts) in members.into_iter().zip(&provenance).zip(parts) {
            plans[position] = if bitmap.is_empty() {
                // Nothing needs translating, but the segment's files may still
                // hold the rows whose coverage was withdrawn: never direct.
                ReadPlan::Translate(TranslationPlan::nothing(ledger))
            } else if self.is_direct(context, &group[position]) {
                ReadPlan::Direct
            } else if parts.coverage.is_empty() {
                ReadPlan::Translate(TranslationPlan::nothing(ledger))
            } else {
                ReadPlan::Translate(TranslationPlan::new(
                    ledger,
                    parts.coverage,
                    parts.excluded,
                    &parts.path,
                ))
            };
        }
        plans
    }
}

/// Rules the lineage decode does not see. Each stable-partition transition
/// owns its row map: cleanup deletes a map once no retained transition
/// references it, so a shared map would be deleted while still in use.
fn validate_records(records: &[StoredRecord]) -> Result<()> {
    let mut map_ids = HashSet::new();
    for stored in records {
        if let Record::Transition(pb::Transition {
            mapping: Some(transition::Mapping::StablePartition(partition)),
            ..
        }) = &stored.record
            && !map_ids.insert(partition.map_id.as_str())
        {
            return Err(corrupt(format!(
                "stable-partition row map {} is referenced by more than one transition",
                partition.map_id
            )));
        }
    }
    Ok(())
}

/// Patch a transition's stable-partition mapping, copying every other
/// field's bytes as written.
fn patch_transition_stable_partition(
    raw: &Bytes,
    remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
) -> Result<Vec<u8>> {
    let mut remaining = raw.clone();
    let mut output = Vec::with_capacity(raw.len() + 8);
    while remaining.has_remaining() {
        let start = raw.len() - remaining.len();
        let (tag, payload) = next_field(&mut remaining)?;
        if tag == STABLE_PARTITION_FIELD {
            let patched = patch_stable_partition_base(&require_message(tag, payload)?, remap)?;
            output.extend_from_slice(&length_delimited(STABLE_PARTITION_FIELD, &patched));
        } else {
            output.extend_from_slice(&raw[start..raw.len() - remaining.len()]);
        }
    }
    Ok(output)
}

/// Drop `base_id` wherever it occurs (the last occurrence is the effective
/// value) and append the remapped value; every other byte is kept.
fn patch_stable_partition_base(
    raw: &Bytes,
    remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
) -> Result<Vec<u8>> {
    let mut remaining = raw.clone();
    let mut output = Vec::with_capacity(raw.len() + 8);
    let mut existing = None;
    while remaining.has_remaining() {
        let start = raw.len() - remaining.len();
        let mut peek = remaining.clone();
        let (tag, wire_type) = decode_key(&mut peek).map_err(|e| corrupt(e.to_string()))?;
        if tag == STABLE_PARTITION_BASE_ID_FIELD {
            if wire_type != WireType::Varint {
                return Err(corrupt("stable partition base_id must be a varint"));
            }
            let value = decode_varint(&mut peek).map_err(|e| corrupt(e.to_string()))?;
            existing =
                Some(u32::try_from(value).map_err(|_| {
                    corrupt(format!("stable partition base_id {value} overflows u32"))
                })?);
            remaining = peek;
        } else {
            next_field(&mut remaining)?;
            output.extend_from_slice(&raw[start..raw.len() - remaining.len()]);
        }
    }
    if let Some(base_id) = remap(existing)? {
        prost::encoding::encode_key(
            STABLE_PARTITION_BASE_ID_FIELD,
            WireType::Varint,
            &mut output,
        );
        prost::encoding::encode_varint(u64::from(base_id), &mut output);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::frag_reuse::history::tests::{
        MAP_A, MAP_B, digest, ordered, partition, whole,
    };
    use crate::system_index::frag_reuse::history::{FragReuseHistory, RawHistory};
    use crate::system_index::frag_reuse::ledger::Mapping;

    #[test]
    fn appended_transitions_are_validated_against_the_whole_lineage() {
        let history = FragReuseHistory::empty(HistoryFormat::Tagged)
            .append_transitions(&[ordered(&[whole(0, 2)], &[(1, 2)])])
            .unwrap();
        let error = history
            .append_transitions(&[ordered(&[whole(0, 2)], &[(2, 2)])])
            .unwrap_err();
        assert!(error.to_string().contains("duplicate consumer"), "{error}");
    }

    #[test]
    fn a_row_map_belongs_to_one_transition() {
        let history = FragReuseHistory::empty(HistoryFormat::Tagged)
            .append_transitions(&[partition(0, 1, MAP_A)])
            .unwrap();
        let error = history
            .append_transitions(&[partition(2, 3, MAP_A)])
            .unwrap_err();
        assert!(
            error.to_string().contains("more than one transition"),
            "{error}"
        );
        history
            .append_transitions(&[partition(2, 3, MAP_B)])
            .unwrap();
    }

    #[test]
    fn base_relocation_keeps_unknown_nested_fields() {
        let mut partition_payload = pb::StablePartition {
            map_id: MAP_A.to_string(),
            map_size_bytes: 64,
            base_id: Some(2),
        }
        .encode_to_vec();
        partition_payload.extend_from_slice(&length_delimited(9, b"future"));
        let mut transition = pb::Transition {
            sources: vec![digest(0, 4, 0)],
            destinations: vec![digest(1, 4, 0)],
            mapping: None,
        }
        .encode_to_vec();
        transition.extend_from_slice(&length_delimited(
            STABLE_PARTITION_FIELD,
            &partition_payload,
        ));
        let history = RawHistory::new(
            HistoryFormat::Tagged,
            length_delimited(TRANSITION_FIELD, &transition).into(),
        )
        .validate()
        .unwrap();

        let relocated = history
            .remap_stable_partition_bases(&|base| {
                assert_eq!(base, Some(2));
                Ok(Some(7))
            })
            .unwrap();
        let Mapping::StablePartition(partition) = relocated.ledger().transitions()[0].mapping()
        else {
            panic!("not a stable partition");
        };
        assert_eq!(partition.base_id, Some(7));
        let content = relocated.content();
        let unknown = length_delimited(9, b"future");
        assert!(
            content
                .windows(unknown.len())
                .any(|window| window == unknown),
            "the unknown nested field was dropped"
        );
    }
}
