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
//! use lance_table::system_index::frag_reuse::history::FragReuseHistory;
//!
//! # fn example(transition: pb::Transition) -> lance_core::Result<()> {
//! let history = FragReuseHistory::empty(1)?.append_transitions(&[transition])?;
//! assert_eq!(history.ledger().transitions().len(), 1);
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use bytes::{Buf, Bytes, BytesMut};
use lance_core::{Error, Result};
use prost::Message;
use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};
use roaring::RoaringBitmap;

use super::ledger::{
    FragReuseLedger, STABLE_PARTITION_FIELD, corrupt, is_interpretable_transition, load_content,
    validate_index_version,
};
use crate::format::pb::ExternalFile;
use crate::format::pb::fragment_reuse_index_details::{self as pb, transition};

/// `InlineContent.legacy_versions`.
const LEGACY_VERSION_FIELD: u32 = 1;
/// `InlineContent.transitions`.
const TRANSITION_FIELD: u32 = 2;
/// `StablePartition.base_id`.
const STABLE_PARTITION_BASE_ID_FIELD: u32 = 3;

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

#[derive(Debug, Clone)]
struct StoredRecord {
    /// The complete field: key, length and payload.
    raw: Bytes,
    record: Record,
}

/// A validated fragment reuse history and its lineage view.
#[derive(Debug, Clone)]
pub struct FragReuseHistory {
    index_version: i32,
    records: Vec<StoredRecord>,
    ledger: Arc<FragReuseLedger>,
}

impl FragReuseHistory {
    /// A history with no records, in the format of `index_version`.
    pub fn empty(index_version: i32) -> Result<Self> {
        Self::from_content(index_version, Bytes::new())
    }

    /// A version 0 history holding `versions`, ordered by dataset version.
    pub fn from_legacy_versions(versions: impl IntoIterator<Item = pb::Version>) -> Result<Self> {
        let mut versions: Vec<pb::Version> = versions.into_iter().collect();
        versions.sort_by_key(|version| version.dataset_version);
        let mut content = BytesMut::new();
        for version in &versions {
            content.extend_from_slice(&length_delimited(
                LEGACY_VERSION_FIELD,
                &version.encode_to_vec(),
            ));
        }
        Self::from_content(0, content.freeze())
    }

    /// Decode FRI `index_details`, resolving external content through
    /// `read_external`, which must return exactly the referenced byte range.
    pub async fn decode<F, Fut>(
        index_version: i32,
        details: &prost_types::Any,
        read_external: F,
    ) -> Result<Self>
    where
        F: FnOnce(ExternalFile) -> Fut,
        Fut: Future<Output = Result<Bytes>>,
    {
        validate_index_version(index_version)?;
        let content = load_content(details, read_external).await?;
        Self::from_content(index_version, content)
    }

    /// Decode the history a committed entry stores, repairing a version 0
    /// history that violates the rules (see [`Self::repair_legacy_versions`]).
    /// Returns the history and the dataset versions repair dropped. A tagged
    /// history is never repaired: every writer of that format validates it.
    pub async fn decode_stored<F, Fut>(
        index_version: i32,
        details: &prost_types::Any,
        read_external: F,
    ) -> Result<(Self, Vec<u64>)>
    where
        F: FnOnce(ExternalFile) -> Fut,
        Fut: Future<Output = Result<Bytes>>,
    {
        validate_index_version(index_version)?;
        let content = load_content(details, read_external).await?;
        match Self::from_content(index_version, content.clone()) {
            Ok(history) => Ok((history, Vec::new())),
            Err(Error::CorruptFile { .. }) if index_version == 0 => {
                let versions =
                    split_records(content)?
                        .into_iter()
                        .filter_map(|stored| match stored.record {
                            Record::LegacyVersion(version) => Some(version),
                            _ => None,
                        });
                Self::repair_legacy_versions(versions)
            }
            Err(error) => Err(error),
        }
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
    pub fn repair_legacy_versions(
        versions: impl IntoIterator<Item = pb::Version>,
    ) -> Result<(Self, Vec<u64>)> {
        let mut versions: Vec<pb::Version> = versions.into_iter().collect();
        versions.sort_by_key(|version| version.dataset_version);
        let mut kept = Vec::with_capacity(versions.len());
        let mut dropped = Vec::new();
        let mut produced: HashMap<u64, u64> = HashMap::new();
        let mut consumed: HashSet<u64> = HashSet::new();
        for version in versions {
            let fits_lineage = version.groups.iter().all(|group| {
                group.old_fragments.iter().all(|source| {
                    !consumed.contains(&source.id)
                        && produced
                            .get(&source.id)
                            .is_none_or(|&rows| rows == source.physical_rows)
                }) && group.new_fragments.iter().all(|destination| {
                    !produced.contains_key(&destination.id) && !consumed.contains(&destination.id)
                })
            });
            if !fits_lineage || Self::from_legacy_versions([version.clone()]).is_err() {
                dropped.push(version.dataset_version);
                continue;
            }
            for group in &version.groups {
                consumed.extend(group.old_fragments.iter().map(|source| source.id));
                produced.extend(
                    group
                        .new_fragments
                        .iter()
                        .map(|destination| (destination.id, destination.physical_rows)),
                );
            }
            kept.push(version);
        }
        Ok((Self::from_legacy_versions(kept)?, dropped))
    }

    /// Build a history from serialized `InlineContent` bytes.
    pub fn from_content(index_version: i32, content: Bytes) -> Result<Self> {
        validate_index_version(index_version)?;
        let ledger = FragReuseLedger::decode_content(index_version, content.clone())?;
        let records = split_records(content)?;
        validate_records(index_version, &records)?;
        Ok(Self {
            index_version,
            records,
            ledger: Arc::new(ledger),
        })
    }

    /// The FRI format these records are interpreted under.
    pub fn index_version(&self) -> i32 {
        self.index_version
    }

    /// Whether the history uses the tagged format (`index_version >= 1`).
    pub fn is_tagged(&self) -> bool {
        self.index_version >= 1
    }

    /// The lineage view used for translation.
    pub fn ledger(&self) -> &Arc<FragReuseLedger> {
        &self.ledger
    }

    /// Records in serialized order.
    pub fn records(&self) -> impl Iterator<Item = &Record> {
        self.records.iter().map(|stored| &stored.record)
    }

    /// Whether every record is one this implementation interprets. Edits
    /// that reason about the whole history require it.
    pub fn is_fully_interpretable(&self) -> bool {
        !self.ledger.has_unsupported_transitions()
    }

    /// The serialized `InlineContent`: every record's bytes, unchanged and in
    /// order.
    pub fn content(&self) -> Bytes {
        let mut content = BytesMut::with_capacity(self.records.iter().map(|r| r.raw.len()).sum());
        for stored in &self.records {
            content.extend_from_slice(&stored.raw);
        }
        content.freeze()
    }

    /// The fragment bitmap the entry stores.
    ///
    /// Version 0 records the destinations of every compaction version, as
    /// every released writer has. The tagged format records every fragment
    /// the retained lineage names, sources included, so a rewrite of a
    /// fragment the history still translates from is seen as covered.
    pub fn fragment_bitmap(&self) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        let mut insert = |digests: &[pb::FragmentDigest]| {
            bitmap.extend(digests.iter().map(|digest| digest.id as u32));
        };
        for record in self.records() {
            match record {
                Record::LegacyVersion(version) => {
                    for group in &version.groups {
                        if self.is_tagged() {
                            insert(&group.old_fragments);
                        }
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

    /// The same records interpreted under the tagged format. Version 0
    /// records are valid tagged records, so the bytes are unchanged.
    pub fn into_tagged(self) -> Result<Self> {
        if self.is_tagged() {
            return Ok(self);
        }
        Self::from_content(1, self.content())
    }

    /// Append `transitions` after every existing record.
    pub fn append_transitions(&self, transitions: &[pb::Transition]) -> Result<Self> {
        if !self.is_tagged() {
            return Err(Error::invalid_input(
                "transitions require the tagged fragment reuse format; convert the history with \
                 `into_tagged` first",
            ));
        }
        self.require_fully_interpretable("append transitions to")?;
        let mut content = BytesMut::from(self.content());
        for transition in transitions {
            content.extend_from_slice(&length_delimited(
                TRANSITION_FIELD,
                &transition.encode_to_vec(),
            ));
        }
        Self::from_content(self.index_version, content.freeze()).map_err(invalid_edit)
    }

    /// Insert a compaction version into a version 0 history, keeping
    /// versions ordered by `dataset_version` as version 0 readers expect.
    pub fn insert_legacy_version(&self, version: pb::Version) -> Result<Self> {
        if self.is_tagged() {
            return Err(Error::invalid_input(
                "a compaction version cannot be added to a tagged fragment reuse history; append \
                 transitions instead",
            ));
        }
        let position = self
            .records
            .iter()
            .position(|stored| {
                matches!(&stored.record, Record::LegacyVersion(existing)
                    if existing.dataset_version > version.dataset_version)
            })
            .unwrap_or(self.records.len());
        let mut records = self.records.clone();
        records.insert(
            position,
            StoredRecord {
                raw: length_delimited(LEGACY_VERSION_FIELD, &version.encode_to_vec()).into(),
                record: Record::LegacyVersion(version),
            },
        );
        self.rebuilt(&records).map_err(invalid_edit)
    }

    /// Keep only the records `keep` accepts.
    pub fn retain(&self, mut keep: impl FnMut(&Record) -> bool) -> Result<Self> {
        self.require_fully_interpretable("trim")?;
        let records: Vec<StoredRecord> = self
            .records
            .iter()
            .filter(|stored| keep(&stored.record))
            .cloned()
            .collect();
        self.rebuilt(&records)
    }

    /// Restamp every stable-partition `base_id` through `remap`, leaving
    /// every other byte as written: unknown fields nested inside a mapping or
    /// a digest survive.
    pub fn remap_stable_partition_bases(
        &self,
        remap: impl Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Self> {
        self.require_fully_interpretable("relocate")?;
        let mut records = Vec::with_capacity(self.records.len());
        for stored in &self.records {
            let raw = match &stored.record {
                Record::Transition(pb::Transition {
                    mapping: Some(transition::Mapping::StablePartition(_)),
                    ..
                }) => {
                    let mut field = stored.raw.clone();
                    let (_, payload) = split_field(&mut field)?;
                    let patched = patch_transition_stable_partition(&payload, &remap)?;
                    length_delimited(TRANSITION_FIELD, &patched).into()
                }
                _ => stored.raw.clone(),
            };
            records.push(StoredRecord {
                raw,
                record: stored.record.clone(),
            });
        }
        self.rebuilt(&records)
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

    fn require_fully_interpretable(&self, operation: &str) -> Result<()> {
        if self.is_fully_interpretable() {
            return Ok(());
        }
        Err(Error::not_supported(format!(
            "the fragment reuse history contains records this writer cannot maintain, so it \
             cannot {operation} it; upgrade to a newer version of Lance"
        )))
    }

    fn rebuilt(&self, records: &[StoredRecord]) -> Result<Self> {
        let mut content = BytesMut::with_capacity(records.iter().map(|r| r.raw.len()).sum());
        for stored in records {
            content.extend_from_slice(&stored.raw);
        }
        Self::from_content(self.index_version, content.freeze())
    }
}

impl PartialEq for FragReuseHistory {
    fn eq(&self, other: &Self) -> bool {
        self.index_version == other.index_version && self.content() == other.content()
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

/// Split one field off `input`, returning its tag and payload. Only
/// length-delimited fields carry a payload; other wire types are skipped.
fn split_field(input: &mut Bytes) -> Result<(u32, Bytes)> {
    let (tag, wire_type) = decode_key(input).map_err(|e| corrupt(e.to_string()))?;
    if wire_type != WireType::LengthDelimited {
        skip_field(wire_type, tag, input, DecodeContext::default())
            .map_err(|e| corrupt(e.to_string()))?;
        return Ok((tag, Bytes::new()));
    }
    let length = decode_varint(input).map_err(|e| corrupt(e.to_string()))?;
    if length > input.remaining() as u64 {
        return Err(corrupt(format!(
            "field {tag} length {length} exceeds remaining {} bytes",
            input.remaining()
        )));
    }
    Ok((tag, input.split_to(length as usize)))
}

fn split_records(content: Bytes) -> Result<Vec<StoredRecord>> {
    let mut remaining = content.clone();
    let mut records = Vec::new();
    while remaining.has_remaining() {
        let start = content.len() - remaining.len();
        let (tag, payload) = split_field(&mut remaining)?;
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

/// Rules the lineage decode does not see. Each stable-partition transition
/// owns its row map: cleanup deletes a map once no retained transition
/// references it, so a shared map would be deleted while still in use.
fn validate_records(index_version: i32, records: &[StoredRecord]) -> Result<()> {
    if index_version == 0 {
        return Ok(());
    }
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

/// A validated history became invalid by an edit: the records the caller
/// added are what is wrong, not the stored history.
fn invalid_edit(error: Error) -> Error {
    match error {
        Error::CorruptFile { source, .. } => Error::invalid_input(format!(
            "the records added to the fragment reuse history are invalid: {source}"
        )),
        other => other,
    }
}

/// One length-delimited protobuf field.
pub fn length_delimited(tag: u32, payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(payload.len() + 8);
    prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut output);
    prost::encoding::encode_varint(payload.len() as u64, &mut output);
    output.extend_from_slice(payload);
    output
}

/// Patch a transition's stable-partition mapping, copying every other
/// field's bytes as written.
fn patch_transition_stable_partition(
    raw: &Bytes,
    remap: &impl Fn(Option<u32>) -> Result<Option<u32>>,
) -> Result<Vec<u8>> {
    let mut remaining = raw.clone();
    let mut output = Vec::with_capacity(raw.len() + 8);
    while remaining.has_remaining() {
        let start = raw.len() - remaining.len();
        let (tag, payload) = split_field(&mut remaining)?;
        if tag == STABLE_PARTITION_FIELD {
            let patched = patch_stable_partition_base(&payload, remap)?;
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
    remap: &impl Fn(Option<u32>) -> Result<Option<u32>>,
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
            split_field(&mut remaining)?;
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
mod tests;
