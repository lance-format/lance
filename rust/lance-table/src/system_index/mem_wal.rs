// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL index data structures and metadata helpers.
//!
//! The MemWAL Index stores:
//! - Configuration (sharding_specs, maintained_indexes)
//! - SSTable compaction progress
//! - Shard state snapshots (eventually consistent)
//!
//! Writers no longer update the index on every write. Instead, they update
//! shard manifests directly. This module provides functions to:
//! - Load the MemWAL index
//! - Update compacted SSTables (called during merge-insert commits)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::format::{IndexMetadata, pb};

pub const MEM_WAL_INDEX_NAME: &str = "__lance_mem_wal";

/// Type alias for shard identifier (UUID v4).
pub type ShardId = Uuid;

/// An SSTable: the immutable result of flushing a MemTable, stored as a Lance dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct SsTable {
    pub generation: u64,
    pub path: String,
    /// Payload size of the rows, as the writer's MemTable accounted for them.
    ///
    /// An estimate of the payload, not a bound on what reading the SSTable
    /// costs: it excludes the per-array structure a reader materializes, so a
    /// consumer budgeting memory from it must add its own headroom.
    ///
    /// `None` on an SSTable written before these fields existed, which a reader
    /// must carry as unmeasured rather than as zero. The same applies to the two
    /// below.
    pub in_memory_bytes: Option<u64>,
    /// Rows held, counting the older duplicates of a primary key that the
    /// generation's deletion vector masks.
    pub physical_rows: Option<u64>,
    /// Total payload size of the primary-key columns over every row in
    /// [`Self::physical_rows`] -- not a per-row size, which varies for a
    /// variable-length key. Carries the same estimate caveat as
    /// [`Self::in_memory_bytes`], and is also `None` on a table with no primary
    /// key.
    pub primary_key_bytes: Option<u64>,
}

impl SsTable {
    /// An SSTable whose contents were not measured at flush.
    pub fn unmeasured(generation: u64, path: String) -> Self {
        Self {
            generation,
            path,
            in_memory_bytes: None,
            physical_rows: None,
            primary_key_bytes: None,
        }
    }
}

impl From<&SsTable> for pb::SsTable {
    fn from(sstable: &SsTable) -> Self {
        Self {
            generation: sstable.generation,
            path: sstable.path.clone(),
            in_memory_bytes: sstable.in_memory_bytes,
            physical_rows: sstable.physical_rows,
            primary_key_bytes: sstable.primary_key_bytes,
        }
    }
}

impl From<pb::SsTable> for SsTable {
    fn from(sstable: pb::SsTable) -> Self {
        Self {
            generation: sstable.generation,
            path: sstable.path,
            in_memory_bytes: sstable.in_memory_bytes,
            physical_rows: sstable.physical_rows,
            primary_key_bytes: sstable.primary_key_bytes,
        }
    }
}

/// A pointer to the latest SSTable compacted for a shard.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CompactedSsTable {
    pub shard_id: Uuid,
    pub generation: u64,
}

impl DeepSizeOf for CompactedSsTable {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        0 // UUID is 16 bytes fixed size, no heap allocations
    }
}

impl CompactedSsTable {
    pub fn new(shard_id: Uuid, generation: u64) -> Self {
        Self {
            shard_id,
            generation,
        }
    }
}

impl From<&CompactedSsTable> for pb::CompactedSsTable {
    fn from(sstable: &CompactedSsTable) -> Self {
        Self {
            shard_id: Some((&sstable.shard_id).into()),
            generation: sstable.generation,
        }
    }
}

impl TryFrom<pb::CompactedSsTable> for CompactedSsTable {
    type Error = Error;

    fn try_from(sstable: pb::CompactedSsTable) -> lance_core::Result<Self> {
        let shard_id = sstable
            .shard_id
            .as_ref()
            .map(Uuid::try_from)
            .ok_or_else(|| Error::invalid_input("Missing shard_id in CompactedSsTable"))??;
        Ok(Self {
            shard_id,
            generation: sstable.generation,
        })
    }
}

/// Tracks which compacted SSTable generation a base table index covers.
/// Used to determine whether to read from SSTable indexes or base table.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct IndexCatchupProgress {
    pub index_name: String,
    pub caught_up_generations: Vec<CompactedSsTable>,
}

impl IndexCatchupProgress {
    pub fn new(index_name: String, caught_up_generations: Vec<CompactedSsTable>) -> Self {
        Self {
            index_name,
            caught_up_generations,
        }
    }

    /// Get the caught up generation for a specific shard.
    /// Returns None if the shard is not present (assumed fully caught up).
    pub fn caught_up_generation_for_shard(&self, shard_id: &Uuid) -> Option<u64> {
        self.caught_up_generations
            .iter()
            .find(|sstable| &sstable.shard_id == shard_id)
            .map(|sstable| sstable.generation)
    }
}

impl From<&IndexCatchupProgress> for pb::IndexCatchupProgress {
    fn from(icp: &IndexCatchupProgress) -> Self {
        Self {
            index_name: icp.index_name.clone(),
            caught_up_generations: icp
                .caught_up_generations
                .iter()
                .map(|sstable| sstable.into())
                .collect(),
        }
    }
}

impl TryFrom<pb::IndexCatchupProgress> for IndexCatchupProgress {
    type Error = Error;

    fn try_from(icp: pb::IndexCatchupProgress) -> lance_core::Result<Self> {
        Ok(Self {
            index_name: icp.index_name,
            caught_up_generations: icp
                .caught_up_generations
                .into_iter()
                .map(CompactedSsTable::try_from)
                .collect::<lance_core::Result<_>>()?,
        })
    }
}

/// Lifecycle status of a WAL shard, persisted in [`ShardManifest`].
///
/// `Sealed` is the durable in-doubt record for drop-table two-phase
/// commit: a sealed shard refuses new writer claims (enforced in
/// `claim_epoch`) but is reversible back to `Active` on rollback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Normal: the shard accepts writer claims.
    #[default]
    Active,
    /// A drop is in flight: claims are refused. Reversible.
    Sealed,
}

impl ShardStatus {
    /// Map to the protobuf enum discriminant (`pb::ShardStatus`).
    fn to_i32(self) -> i32 {
        match self {
            Self::Active => 0,
            Self::Sealed => 1,
        }
    }

    /// Map from the protobuf enum discriminant; unknown values decode as
    /// `Active` (forward-compatible default).
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Sealed,
            _ => Self::Active,
        }
    }
}

/// Shard manifest containing epoch-based fencing and WAL state.
/// Each shard has exactly one active writer at any time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifest {
    pub shard_id: Uuid,
    pub version: u64,
    pub shard_spec_id: u32,
    /// Computed shard field values as raw Arrow scalar bytes, keyed by field id.
    /// The byte encoding follows Arrow's little-endian convention: int32 is 4 LE
    /// bytes, utf8 is raw UTF-8 bytes, etc. The result_type in the corresponding
    /// ShardingField from the ShardingSpec determines how to interpret each value.
    pub shard_field_values: HashMap<String, Vec<u8>>,
    pub writer_epoch: u64,
    /// The most recent WAL entry position flushed to a MemTable.
    /// Recovery replays from `replay_after_wal_entry_position + 1`. The
    /// default value 0 means "no flush has ever stamped this shard" — WAL
    /// positions themselves are 1-based, so 0 is never a valid covered
    /// position.
    pub replay_after_wal_entry_position: u64,
    /// The most recent WAL entry position observed at manifest write time.
    /// Default 0 means "no entry has been written yet"; WAL positions are
    /// 1-based.
    pub wal_entry_position_last_seen: u64,
    pub current_generation: u64,
    pub sstables: Vec<SsTable>,
    /// Lifecycle status (drop-table 2PC). Defaults to `Active`; preserved
    /// across claims via `..base` so only fresh constructions set it.
    pub status: ShardStatus,
}

impl ShardManifest {
    /// The version a manifest built on this one must carry.
    ///
    /// Manifest versions are CAS-allocated and must stay gap-free: a reader
    /// scans forward and stops at the first version it cannot find, so a gap
    /// hides everything past it.
    pub fn next_version(&self) -> u64 {
        self.version + 1
    }
}

impl DeepSizeOf for ShardManifest {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.shard_field_values.deep_size_of_children(context)
            + self.sstables.deep_size_of_children(context)
    }
}

impl From<&ShardManifest> for pb::ShardManifest {
    fn from(rm: &ShardManifest) -> Self {
        Self {
            shard_id: Some((&rm.shard_id).into()),
            version: rm.version,
            shard_spec_id: rm.shard_spec_id,
            shard_field_entries: rm
                .shard_field_values
                .iter()
                .map(|(k, v)| pb::ShardFieldEntry {
                    field_id: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            writer_epoch: rm.writer_epoch,
            replay_after_wal_entry_position: rm.replay_after_wal_entry_position,
            wal_entry_position_last_seen: rm.wal_entry_position_last_seen,
            current_generation: rm.current_generation,
            sstables: rm.sstables.iter().map(|sstable| sstable.into()).collect(),
            status: rm.status.to_i32(),
        }
    }
}

impl TryFrom<pb::ShardManifest> for ShardManifest {
    type Error = Error;

    fn try_from(rm: pb::ShardManifest) -> lance_core::Result<Self> {
        let shard_id = rm
            .shard_id
            .as_ref()
            .map(Uuid::try_from)
            .ok_or_else(|| Error::invalid_input("Missing shard_id in ShardManifest"))??;
        let shard_field_values = rm
            .shard_field_entries
            .into_iter()
            .map(|e| (e.field_id, e.value))
            .collect();
        Ok(Self {
            shard_id,
            version: rm.version,
            shard_spec_id: rm.shard_spec_id,
            shard_field_values,
            writer_epoch: rm.writer_epoch,
            replay_after_wal_entry_position: rm.replay_after_wal_entry_position,
            wal_entry_position_last_seen: rm.wal_entry_position_last_seen,
            current_generation: rm.current_generation,
            sstables: rm.sstables.into_iter().map(SsTable::from).collect(),
            status: ShardStatus::from_i32(rm.status),
        })
    }
}

/// Sharding field definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct ShardingField {
    pub field_id: String,
    pub source_ids: Vec<i32>,
    pub transform: Option<String>,
    pub expression: Option<String>,
    pub result_type: String,
    pub parameters: HashMap<String, String>,
}

impl From<&ShardingField> for pb::ShardingField {
    fn from(rf: &ShardingField) -> Self {
        Self {
            field_id: rf.field_id.clone(),
            source_ids: rf.source_ids.clone(),
            transform: rf.transform.clone(),
            expression: rf.expression.clone(),
            result_type: rf.result_type.clone(),
            parameters: rf.parameters.clone(),
        }
    }
}

impl From<pb::ShardingField> for ShardingField {
    fn from(rf: pb::ShardingField) -> Self {
        Self {
            field_id: rf.field_id,
            source_ids: rf.source_ids,
            transform: rf.transform,
            expression: rf.expression,
            result_type: rf.result_type,
            parameters: rf.parameters,
        }
    }
}

/// Sharding spec definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct ShardingSpec {
    pub spec_id: u32,
    pub fields: Vec<ShardingField>,
}

impl From<&ShardingSpec> for pb::ShardingSpec {
    fn from(rs: &ShardingSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.iter().map(|f| f.into()).collect(),
        }
    }
}

impl From<pb::ShardingSpec> for ShardingSpec {
    fn from(rs: pb::ShardingSpec) -> Self {
        Self {
            spec_id: rs.spec_id,
            fields: rs.fields.into_iter().map(ShardingField::from).collect(),
        }
    }
}

/// Index details for MemWAL Index, stored in IndexMetadata.index_details.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct MemWalIndexDetails {
    pub snapshot_ts_millis: i64,
    pub num_shards: u32,
    pub inline_snapshots: Option<Vec<u8>>,
    pub sharding_specs: Vec<ShardingSpec>,
    pub maintained_indexes: Vec<String>,
    pub compacted_sstables: Vec<CompactedSsTable>,
    pub index_catchup: Vec<IndexCatchupProgress>,
    /// Default `ShardWriter` configuration values for this MemWAL index.
    ///
    /// Persisted so every writer — across processes and restarts — starts
    /// from the same default writer configuration. These are defaults only;
    /// an individual writer may still override any value at runtime in its
    /// own (non-persisted) `ShardWriterConfig`.
    pub writer_config_defaults: HashMap<String, String>,
}

impl From<&MemWalIndexDetails> for pb::MemWalIndexDetails {
    fn from(details: &MemWalIndexDetails) -> Self {
        Self {
            snapshot_ts_millis: details.snapshot_ts_millis,
            num_shards: details.num_shards,
            inline_snapshots: details.inline_snapshots.clone(),
            sharding_specs: details.sharding_specs.iter().map(|rs| rs.into()).collect(),
            maintained_indexes: details.maintained_indexes.clone(),
            compacted_sstables: details
                .compacted_sstables
                .iter()
                .map(|sstable| sstable.into())
                .collect(),
            index_catchup: details.index_catchup.iter().map(|icp| icp.into()).collect(),
            writer_config_defaults: details.writer_config_defaults.clone(),
        }
    }
}

impl TryFrom<pb::MemWalIndexDetails> for MemWalIndexDetails {
    type Error = Error;

    fn try_from(details: pb::MemWalIndexDetails) -> lance_core::Result<Self> {
        Ok(Self {
            snapshot_ts_millis: details.snapshot_ts_millis,
            num_shards: details.num_shards,
            inline_snapshots: details.inline_snapshots,
            sharding_specs: details
                .sharding_specs
                .into_iter()
                .map(ShardingSpec::from)
                .collect(),
            maintained_indexes: details.maintained_indexes,
            compacted_sstables: details
                .compacted_sstables
                .into_iter()
                .map(CompactedSsTable::try_from)
                .collect::<lance_core::Result<_>>()?,
            index_catchup: details
                .index_catchup
                .into_iter()
                .map(IndexCatchupProgress::try_from)
                .collect::<lance_core::Result<_>>()?,
            writer_config_defaults: details.writer_config_defaults,
        })
    }
}

/// MemWAL Index provides access to MemWAL configuration and state.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct MemWalIndex {
    pub details: MemWalIndexDetails,
}

impl MemWalIndex {
    pub fn new(details: MemWalIndexDetails) -> Self {
        Self { details }
    }

    pub fn compacted_generation_for_shard(&self, shard_id: &Uuid) -> Option<u64> {
        self.details
            .compacted_sstables
            .iter()
            .find(|sstable| &sstable.shard_id == shard_id)
            .map(|sstable| sstable.generation)
    }

    /// Get the caught up generation for a specific index and shard.
    /// Returns None if the index is not tracked (assumed fully caught up).
    pub fn index_caught_up_generation(&self, index_name: &str, shard_id: &Uuid) -> Option<u64> {
        self.details
            .index_catchup
            .iter()
            .find(|icp| icp.index_name == index_name)
            .and_then(|icp| icp.caught_up_generation_for_shard(shard_id))
    }
}

// Reading and updating the `IndexMetadata` entry that carries the details above.

/// Load MemWalIndexDetails from an IndexMetadata.
pub fn load_mem_wal_index_details(index: IndexMetadata) -> Result<MemWalIndexDetails> {
    if let Some(details_any) = index.index_details.as_ref() {
        if !details_any.type_url.ends_with("MemWalIndexDetails") {
            return Err(Error::index(format!(
                "Index details is not for the MemWAL index, but {}",
                details_any.type_url
            )));
        }

        Ok(MemWalIndexDetails::try_from(
            details_any.to_msg::<pb::MemWalIndexDetails>()?,
        )?)
    } else {
        Err(Error::index("Index details not found for the MemWAL index"))
    }
}

/// Open the MemWAL index from its metadata.
pub fn open_mem_wal_index(index: IndexMetadata) -> Result<Arc<MemWalIndex>> {
    Ok(Arc::new(MemWalIndex::new(load_mem_wal_index_details(
        index,
    )?)))
}

/// Update `compacted_sstables` in the MemWAL index.
///
/// Called from the final data-changing merge-insert commit for a compaction
/// target, so the rows and the generation that describes them publish
/// together.
///
/// A proposed generation must be **strictly greater** than the one the latest
/// state records for that shard, and a stale one fails the whole transaction.
/// Accepting it while keeping the larger marker would publish that worker's row
/// mutations under a generation it did not produce, and anything reading only
/// the marker could then stop serving SSTables whose rows were never inserted.
///
/// Every other `MemWalIndexDetails` field is carried through untouched.
pub fn update_mem_wal_index_compacted_sstables(
    indices: &mut [IndexMetadata],
    dataset_version: u64,
    new_compacted_sstables: Vec<CompactedSsTable>,
) -> Result<()> {
    if new_compacted_sstables.is_empty() {
        return Ok(());
    }

    let mut seen_shards = HashSet::with_capacity(new_compacted_sstables.len());
    for sstable in &new_compacted_sstables {
        if !seen_shards.insert(sstable.shard_id) {
            return Err(Error::invalid_input(format!(
                "Duplicate shard {} in one SSTable compaction update; each shard \
                 may advance at most once per transaction",
                sstable.shard_id
            )));
        }
    }

    // Default details would describe a table with no MemWAL shards at all, so
    // the recorded generation would name a shard nothing can corroborate.
    // Refuse instead of inventing metadata.
    let pos = indices
        .iter()
        .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Cannot record SSTable compaction progress: the {} system index \
                 does not exist on this table",
                MEM_WAL_INDEX_NAME
            ))
        })?;

    // Validated against a copy so a rejected update leaves `indices` exactly as
    // the caller passed it.
    let mut details = load_mem_wal_index_details(indices[pos].clone())?;

    for new_sstable in new_compacted_sstables {
        match details
            .compacted_sstables
            .iter_mut()
            .find(|sstable| sstable.shard_id == new_sstable.shard_id)
        {
            Some(existing) if new_sstable.generation <= existing.generation => {
                return Err(Error::invalid_input(format!(
                    "Stale SSTable compaction for shard {}: proposed generation {} \
                     is not greater than the recorded generation {}",
                    new_sstable.shard_id, new_sstable.generation, existing.generation
                )));
            }
            Some(existing) => existing.generation = new_sstable.generation,
            None => details.compacted_sstables.push(new_sstable),
        }
    }

    // Replaced in place so the index list keeps its order.
    indices[pos] = new_mem_wal_index_meta(dataset_version, details)?;
    Ok(())
}

/// Create a new MemWAL index metadata entry.
///
/// A fresh UUID is minted on every rewrite, including metadata-only updates.
/// The decoded-details cache is keyed on that UUID, so the change of identity
/// is what invalidates it; holding the UUID steady would leave a warmed reader
/// answering with the state from before the update.
pub fn new_mem_wal_index_meta(
    dataset_version: u64,
    details: MemWalIndexDetails,
) -> Result<IndexMetadata> {
    Ok(IndexMetadata {
        uuid: Uuid::new_v4(),
        name: MEM_WAL_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any::from_msg(
            &pb::MemWalIndexDetails::from(&details),
        )?)),
        index_version: 0,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // Memory WAL index is inline (no files)
        files: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An SSTable written before the size fields existed decodes as
    /// unmeasured, not as zero. A reader that saw zero would treat a full
    /// generation as costing nothing.
    #[test]
    fn an_sstable_without_the_size_fields_decodes_as_unmeasured() {
        let legacy = pb::SsTable {
            generation: 7,
            path: "aaa_gen_7".to_string(),
            in_memory_bytes: None,
            physical_rows: None,
            primary_key_bytes: None,
        };
        let decoded = SsTable::from(legacy);
        assert_eq!(decoded.generation, 7);
        assert_eq!(decoded.in_memory_bytes, None);
        assert_eq!(decoded.physical_rows, None);
        assert_eq!(decoded.primary_key_bytes, None);
    }

    /// All three survive the round trip, so a reader sees what the flush
    /// measured rather than a default.
    #[test]
    fn the_recorded_size_survives_the_round_trip() {
        let recorded = SsTable {
            generation: 7,
            path: "aaa_gen_7".to_string(),
            in_memory_bytes: Some(4_096),
            physical_rows: Some(10),
            primary_key_bytes: Some(80),
        };
        let encoded = pb::SsTable::from(&recorded);
        assert_eq!(encoded.in_memory_bytes, Some(4_096));
        assert_eq!(encoded.physical_rows, Some(10));
        assert_eq!(encoded.primary_key_bytes, Some(80));
        assert_eq!(SsTable::from(encoded), recorded);
    }
}
