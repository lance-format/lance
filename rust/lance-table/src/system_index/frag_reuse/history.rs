// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The fragment reuse history as one validated value.
//!
//! [`FragReuseHistory`] is the only way to read or edit the records an FRI
//! entry holds. It keeps every record's exact wire bytes in serialized order,
//! unknown records included, so carrying a history forward never reorders or
//! drops what an older or newer writer recorded. Each edit rebuilds the
//! history from the edited bytes, so every value has passed the same
//! validation as a freshly decoded one.
//!
//! ```
//! use lance_table::format::pb::fragment_reuse_index_details as pb;
//! use lance_table::system_index::frag_reuse::history::{FragReuseHistory, HistoryFormat};
//!
//! # fn example(transition: pb::Transition) -> lance_core::Result<()> {
//! let history = FragReuseHistory::empty(HistoryFormat::Tagged).append_transitions(&[transition])?;
//! assert_eq!(history.ledger().transitions().len(), 1);
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::{Buf, Bytes};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use prost::Message;
use prost::encoding::WireType;
use roaring::RoaringBitmap;

use super::imp::{FriImpl, StoredRecord, concat};
use super::ledger::{FragReuseLedger, corrupt, is_interpretable_transition, next_field};
use super::legacy::{self, LegacyHistory};
use super::plan::{ReadContext, ReadPlan, SegmentSummary};
use super::tagged::TaggedHistory;
use crate::format::pb::fragment_reuse_index_details as pb;

/// `InlineContent.legacy_versions`.
pub(super) const LEGACY_VERSION_FIELD: u32 = 1;
/// `InlineContent.transitions`.
pub(super) const TRANSITION_FIELD: u32 = 2;

/// How an FRI entry's records are interpreted, as its `index_version` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryFormat {
    /// `index_version` 0: compaction versions only, which every released
    /// reader and writer understands.
    Legacy,
    /// `index_version` 1: compaction versions and tagged transitions. Once a
    /// table carries them it keeps the format (the feature flag is sticky).
    Tagged,
}

impl HistoryFormat {
    /// The format of an entry stamped `index_version`. A version this build
    /// does not know is refused with `NotSupported`: only a newer Lance can
    /// interpret it.
    pub fn from_index_version(index_version: i32) -> Result<Self> {
        match index_version {
            0 => Ok(Self::Legacy),
            1 => Ok(Self::Tagged),
            _ => Err(Error::not_supported(format!(
                "Unsupported FRI index_version {index_version}; supported versions are 0 and 1. Please upgrade to a newer version of Lance."
            ))),
        }
    }

    /// The `IndexMetadata::index_version` of an entry in this format.
    pub fn index_version(self) -> i32 {
        match self {
            Self::Legacy => 0,
            Self::Tagged => 1,
        }
    }
}

/// One record of the history, decoded.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    /// An index_version 0 compaction version.
    LegacyVersion(pb::Version),
    /// A tagged transition this implementation interprets.
    Transition(pb::Transition),
    /// A record this implementation cannot interpret: an unknown envelope
    /// field, or a transition carrying fields it does not know. Its bytes are
    /// carried forward unchanged, but no edit that needs to reason about the
    /// whole history (trim, relocation, appending) is allowed past it.
    Uninterpretable { field: u32 },
}

/// The serialized records of an FRI entry, not yet checked against the
/// history's rules. Whether a history that breaks them is refused or repaired
/// depends on who wrote it, so the caller decides by how it consumes this:
/// [`Self::validate`] for an entry a caller proposes, [`Self::repair`] for an
/// entry already committed.
#[derive(Debug, Clone)]
pub struct RawHistory {
    format: HistoryFormat,
    content: Bytes,
}

impl RawHistory {
    /// The serialized `InlineContent` `content`, interpreted under `format`.
    pub fn new(format: HistoryFormat, content: Bytes) -> Self {
        Self { format, content }
    }

    /// The history, refusing one that breaks the rules with `CorruptFile`.
    /// An entry a caller proposes is never repaired: an invalid proposal is
    /// the caller's error.
    pub fn validate(self) -> Result<FragReuseHistory> {
        FragReuseHistory::from_content(self.format, self.content)
    }

    /// The history a committed entry stores, repairing a version 0 history
    /// that an older writer left invalid (see
    /// [`FragReuseHistory::repair_legacy_versions`]). Returns the history and
    /// the dataset versions repair dropped. A tagged history is never
    /// repaired, since every writer of that format validates it: one that
    /// breaks the rules is refused as by [`Self::validate`].
    pub fn repair(self) -> Result<(FragReuseHistory, Vec<u64>)> {
        match self.format {
            HistoryFormat::Legacy => LegacyHistory::decode_repairing(self.content),
            HistoryFormat::Tagged => Ok((self.validate()?, Vec::new())),
        }
    }
}

/// A validated fragment reuse history and its lineage view. Cheap to clone.
#[derive(Debug, Clone)]
pub struct FragReuseHistory(Arc<dyn FriImpl>);

impl FragReuseHistory {
    pub(super) fn from_impl(imp: Arc<dyn FriImpl>) -> Self {
        Self(imp)
    }

    /// A history with no records, in `format`.
    pub fn empty(format: HistoryFormat) -> Self {
        match format {
            HistoryFormat::Legacy => Self(Arc::new(LegacyHistory::default())),
            HistoryFormat::Tagged => Self(Arc::new(TaggedHistory::default())),
        }
    }

    /// A version 0 history holding `versions`, ordered by dataset version.
    pub fn from_legacy_versions(versions: Vec<pb::Version>) -> Result<Self> {
        legacy::from_versions(versions)
    }

    /// A version 0 history of the `versions` that can be kept, and the dataset
    /// versions of the ones dropped.
    ///
    /// Versions are taken in dataset version order, and one is dropped when it
    /// is invalid on its own or when it would break the lineage of the ones
    /// already kept (consuming a fragment twice, producing one twice, or
    /// producing a fragment that is already consumed). Dropping a version only
    /// loses the translation it records: rows an index reaches only through it
    /// are scanned instead.
    pub fn repair_legacy_versions(versions: Vec<pb::Version>) -> Result<(Self, Vec<u64>)> {
        legacy::repair_versions(versions)
    }

    fn from_content(format: HistoryFormat, content: Bytes) -> Result<Self> {
        Ok(Self(match format {
            HistoryFormat::Legacy => Arc::new(LegacyHistory::decode(content)?),
            HistoryFormat::Tagged => Arc::new(TaggedHistory::decode(content)?),
        }))
    }

    /// The FRI format these records are interpreted under.
    pub fn format(&self) -> HistoryFormat {
        self.0.format()
    }

    /// Whether the history uses the tagged format.
    pub fn is_tagged(&self) -> bool {
        self.format() == HistoryFormat::Tagged
    }

    /// The lineage view used for translation.
    pub fn ledger(&self) -> &Arc<FragReuseLedger> {
        &self.0.records().ledger
    }

    /// Records in serialized order.
    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.0.records().stored.iter().map(|stored| &stored.record)
    }

    /// Whether every record is one this implementation interprets. Edits
    /// that reason about the whole history require it.
    pub fn is_fully_interpretable(&self) -> bool {
        self.0.records().is_fully_interpretable()
    }

    /// The serialized `InlineContent`: every record's bytes, unchanged and in
    /// order.
    pub fn content(&self) -> Bytes {
        concat(&self.0.records().stored)
    }

    /// The `fragment_bitmap` for this history's FRI entry.
    ///
    /// Which fragments it names is an encoding rule of the format. Version 0
    /// keeps what released writers stored: the destinations of every
    /// compaction version. The tagged format names every fragment of the
    /// retained lineage, sources included, so a rewrite of a fragment the
    /// history still translates from is seen as covered.
    pub fn fragment_bitmap(&self) -> RoaringBitmap {
        self.0.fragment_bitmap()
    }

    /// The same records interpreted under the tagged format. Version 0
    /// records are valid tagged records, so the bytes are unchanged.
    pub fn into_tagged(self) -> Result<Self> {
        self.0.into_tagged().map(Self)
    }

    /// Append `transitions` after every existing record.
    pub fn append_transitions(&self, transitions: &[pb::Transition]) -> Result<Self> {
        self.0.append_transitions(transitions).map(Self)
    }

    /// Insert a compaction version into a version 0 history, keeping
    /// versions ordered by `dataset_version` as version 0 readers expect.
    pub fn insert_legacy_version(&self, version: pb::Version) -> Result<Self> {
        self.0.insert_legacy_version(version).map(Self)
    }

    /// Keep only the records `keep` accepts.
    pub fn retain(&self, keep: &mut dyn FnMut(&Record) -> bool) -> Result<Self> {
        self.0.records().require_fully_interpretable("trim")?;
        let records: Vec<StoredRecord> = self
            .0
            .records()
            .stored
            .iter()
            .filter(|stored| keep(&stored.record))
            .cloned()
            .collect();
        self.0.rebuilt(concat(&records)).map(Self)
    }

    /// Restamp every stable-partition `base_id` through `remap`, leaving
    /// every other byte as written: unknown fields nested inside a mapping or
    /// a digest survive.
    pub fn remap_stable_partition_bases(
        &self,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Self> {
        self.0.clone().remap_stable_partition_bases(remap).map(Self)
    }

    /// The same history, read on a table with stable row IDs. Its read plans
    /// also exclude every segment that does not store row addresses or whose
    /// kind cannot apply the history, whether or not it is caught up: the
    /// legacy reader applies the history to every segment it opens, so it
    /// would rewrite stable row IDs as if they were addresses. Only a legacy
    /// history can describe such a table.
    pub fn on_stable_row_ids(self) -> Result<Self> {
        self.0.on_stable_row_ids().map(Self)
    }

    /// Whether `segment` holds no address the history still moves, so it can
    /// be used without translation, and no record is needed for it.
    ///
    /// `entry_version` is the `dataset_version` of the FRI entry holding the
    /// history (see [`ReadContext::entry_version`]). A segment without a
    /// recorded fragment bitmap is caught up only under the legacy rule, which
    /// released writers relied on.
    pub fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool {
        self.0.is_caught_up(entry_version, segment)
    }

    /// Whether [`Self::plan`] consults `segment.translation`. Computing it can
    /// require reading the segment's files, so a caller may skip it (and pass
    /// any value) for a segment this returns false for.
    pub fn needs_translation_support(
        &self,
        context: &ReadContext,
        segment: &SegmentSummary,
    ) -> bool {
        self.0.needs_translation_support(context, segment)
    }

    /// How a reader uses each segment of one logical index, in `group`
    /// order. A logical index is planned as a whole: a fragment one segment
    /// covers directly is never translated into for another.
    pub fn plan(&self, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        self.0.plan(context, group)
    }

    /// The live fragments each segment of `group` answers for, as planned by
    /// [`Self::plan`]: every live fragment of a direct segment, the remaining
    /// coverage of a translated one, nothing for an excluded one. A fragment
    /// no segment claims is scanned.
    pub fn effective_coverage(
        &self,
        context: &ReadContext,
        group: &[SegmentSummary],
    ) -> Vec<RoaringBitmap> {
        self.plan(context, group)
            .into_iter()
            .zip(group)
            .map(|(plan, segment)| match plan {
                ReadPlan::Direct => segment
                    .fragment_bitmap
                    .as_ref()
                    .map(|bitmap| bitmap & context.live_fragments)
                    .unwrap_or_default(),
                ReadPlan::Translate(plan) => plan.coverage().clone(),
                ReadPlan::Excluded => RoaringBitmap::new(),
            })
            .collect()
    }

    /// The records `self` adds to `base`, refusing anything else: a rewrite
    /// appends history, so every record `base` holds must still be present
    /// with the same content. Transitions are identified by their source and
    /// destination fragments, compaction versions by their dataset version.
    pub fn records_added_since(&self, base: &Self) -> Result<Vec<Record>> {
        let mut proposed: HashMap<RecordIdentity, &Record> = HashMap::new();
        for record in self.records() {
            if let Some(identity) = RecordIdentity::of(record)
                && proposed.insert(identity.clone(), record).is_some()
            {
                return Err(Error::invalid_input(format!(
                    "the fragment reuse entry records {identity} more than once"
                )));
            }
        }
        let mut recorded = HashSet::new();
        for record in base.records() {
            let Some(identity) = RecordIdentity::of(record) else {
                continue;
            };
            match proposed.get(&identity) {
                None => {
                    return Err(Error::invalid_input(format!(
                        "the fragment reuse entry drops {identity}; a rewrite only appends \
                         history, retire records through the fragment reuse index cleanup instead"
                    )));
                }
                Some(&proposed) if proposed != record => {
                    return Err(Error::invalid_input(format!(
                        "the fragment reuse entry changes {identity}; a recorded entry cannot be \
                         modified"
                    )));
                }
                Some(_) => {}
            }
            recorded.insert(identity);
        }
        Ok(self
            .records()
            .filter(|record| {
                RecordIdentity::of(record).is_some_and(|identity| !recorded.contains(&identity))
            })
            .cloned()
            .collect())
    }
}

impl DeepSizeOf for FragReuseHistory {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let records = self.0.records();
        records
            .stored
            .iter()
            .map(|stored| stored.raw.len())
            .sum::<usize>()
            + records.ledger.deep_size_of_children(context)
    }
}

impl PartialEq for FragReuseHistory {
    fn eq(&self, other: &Self) -> bool {
        self.format() == other.format() && self.content() == other.content()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RecordIdentity {
    Transition {
        sources: Vec<u64>,
        destinations: Vec<u64>,
    },
    LegacyVersion(u64),
}

impl RecordIdentity {
    fn of(record: &Record) -> Option<Self> {
        let ids = |digests: &[pb::FragmentDigest]| digests.iter().map(|digest| digest.id).collect();
        match record {
            Record::Transition(transition) => Some(Self::Transition {
                sources: ids(&transition.sources),
                destinations: ids(&transition.destinations),
            }),
            Record::LegacyVersion(version) => Some(Self::LegacyVersion(version.dataset_version)),
            Record::Uninterpretable { .. } => None,
        }
    }
}

impl std::fmt::Display for RecordIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transition { sources, .. } => {
                write!(f, "the recorded transition from fragments {sources:?}")
            }
            Self::LegacyVersion(version) => {
                write!(f, "the recorded compaction version {version}")
            }
        }
    }
}

pub(super) fn split_records(content: Bytes) -> Result<Vec<StoredRecord>> {
    let mut remaining = content.clone();
    let mut records = Vec::new();
    while remaining.has_remaining() {
        let start = content.len() - remaining.len();
        let (tag, payload) = next_field(&mut remaining)?;
        let payload = payload.unwrap_or_default();
        let end = content.len() - remaining.len();
        let record = match tag {
            LEGACY_VERSION_FIELD => Record::LegacyVersion(
                pb::Version::decode(payload).map_err(|e| corrupt(e.to_string()))?,
            ),
            TRANSITION_FIELD if is_interpretable_transition(&payload)? => Record::Transition(
                pb::Transition::decode(payload).map_err(|e| corrupt(e.to_string()))?,
            ),
            field => Record::Uninterpretable { field },
        };
        records.push(StoredRecord {
            raw: content.slice(start..end),
            record,
        });
    }
    Ok(records)
}

/// One length-delimited protobuf field.
pub fn length_delimited(tag: u32, payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(payload.len() + 8);
    prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut output);
    prost::encoding::encode_varint(payload.len() as u64, &mut output);
    output.extend_from_slice(payload);
    output
}

#[cfg(test)]
pub(super) mod tests;
