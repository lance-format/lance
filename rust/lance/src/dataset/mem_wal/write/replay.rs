// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WAL replay into memtables and generations.

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_index::mem_wal::ShardManifest;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use uuid::Uuid;

use super::schema::conform_to_storage_schema;
use super::writer::{memtable_cannot_take, memtable_reached_flush_threshold};
use crate::dataset::blob::prepared_blob_ids;
use crate::dataset::mem_wal::MemTableDataTarget;
use crate::dataset::mem_wal::index::MemIndexSpec;
use crate::dataset::mem_wal::memtable::MemTable;
use crate::dataset::mem_wal::memtable::batch_store::StoredBatch;
use crate::dataset::mem_wal::memtable::flush::MemTableFlusher;
use crate::dataset::mem_wal::wal::{WalFlusher, WalTailer};

/// Replay WAL entries written after the last successfully-flushed SSTable
/// into the freshly-built MemTable. Updates any in-memory indexes attached to
/// the MemTable so replayed rows are immediately searchable.
///
/// Returns the WAL position the next `WalAppender::append` should use — i.e.
/// one past the highest replayed position, or the original start position if
/// the loop found nothing (we proved that position is empty by getting
/// `None` back from the tailer). The caller can pass this directly to
/// `WalAppender::seed_next_position` unconditionally so the first post-open
/// append skips its own discovery probe.
///
/// Aborts with an error if any replayed entry's `writer_epoch` is strictly
/// greater than `our_epoch` — that indicates a successor writer claimed the
/// shard between our `claim_epoch` and this replay, fencing us.
/// Outcome of replaying a shard's WAL into memory.
pub(super) struct ReplayResult {
    /// The active memtable — the final, partial one replay left unfrozen. A fresh
    /// shard yields an empty one; every frozen memtable was flushed to a Lance
    /// generation during replay and is not returned.
    pub(super) active: MemTable,
    /// One past the highest WAL entry position observed — the next write position.
    pub(super) next_wal_position: u64,
}

/// Replay a shard's WAL into memory, flushing frozen memtables as the batch store
/// fills.
///
/// A single memtable holds at most `max_memtable_batches` batches, but a WAL is
/// unbounded — so replay must rotate exactly as the live write path does. It
/// freezes a full memtable and, because the data is already durable, flushes it to
/// a Lance generation right here (the same `MemTableFlusher::flush` the live path
/// uses), rather than holding every frozen memtable in memory until open
/// finishes. That bounds resident memory to ~two memtables and truncates the WAL
/// as it goes, so a later reopen replays only the unflushed tail. Only the final
/// partial memtable is returned, as the active one.
///
/// `make_memtable(generation, global_offset, target)` builds a fresh,
/// cursor-bound memtable. Target-bearing entries are rotated at their persisted
/// generation boundary, never by reconstructed size thresholds, so the data
/// file descriptors always remain paired with the sidecars written before the
/// WAL append.
#[allow(clippy::too_many_arguments)]
pub(super) async fn replay_memtable_from_wal(
    object_store: Arc<ObjectStore>,
    base_path: Path,
    dir_id: &str,
    shard_id: Uuid,
    our_epoch: u64,
    manifest: &ShardManifest,
    base_generation: u64,
    mut make_memtable: impl FnMut(u64, usize, Option<MemTableDataTarget>) -> Result<MemTable>,
    // Conforming a replayed entry needs these: a primary key the entry does not
    // carry cannot be filled with a null.
    pk_columns: &[String],
    flusher: &MemTableFlusher,
    wal_flusher: &WalFlusher,
    index_specs: &[MemIndexSpec],
    max_memtable_size: usize,
    max_memtable_rows: usize,
    max_resident_bytes: usize,
) -> Result<ReplayResult> {
    // WAL positions are 1-based (see `FIRST_WAL_ENTRY_POSITION`), so a
    // cursor of 0 means "no flush has ever stamped this shard" and replay
    // starts at position 1. After flushing position N the cursor holds N
    // and replay starts at N+1. The arithmetic collapses to a single
    // saturating_add(1) in both cases — we deliberately do not consult
    // `sstables` here, since an external compactor may
    // legitimately drain that vector back to empty after merging its
    // contents into the base table.
    let start_position = manifest.replay_after_wal_entry_position.saturating_add(1);

    let tailer = WalTailer::new(object_store, base_path, dir_id, shard_id);
    let mut position = start_position;

    let mut active = make_memtable(base_generation, 0, None)?;
    let mut last_replayed_generation: Option<u64> = None;

    loop {
        match tailer.read_entry(position).await? {
            // The first NotFound proves the WAL tip is at `position`, which
            // is the next write position to hand back.
            None => break,
            Some(entry) => {
                if entry.writer_epoch > our_epoch {
                    return Err(Error::fenced_by_peer(format!(
                        "WAL replay aborted: entry at position {} has writer_epoch {} > our claimed epoch {} for shard {} (writer was fenced during open)",
                        position, entry.writer_epoch, our_epoch, shard_id
                    )));
                }
                if let Some(target) = entry.target.as_ref()
                    && target.creator_epoch > entry.writer_epoch
                {
                    return Err(Error::io(format!(
                        "WAL entry at position {} has writer_epoch {} older than target creator_epoch {}",
                        position, entry.writer_epoch, target.creator_epoch
                    )));
                }
                // An entry of a generation below the one replay starts from is
                // already in that generation's SSTable. A flush's covered
                // position can trail its generation's last entry, so replay
                // reaches such entries.
                let already_flushed = entry
                    .generation
                    .is_some_and(|generation| generation < base_generation);
                if let (Some(generation), Some(last)) = (entry.generation, last_replayed_generation)
                    && generation < last
                {
                    return Err(Error::io(format!(
                        "WAL entry at position {} has generation {} below the generation {} \
                         replayed before it",
                        position, generation, last
                    )));
                }
                // Fence sentinels deserialize to zero batches and are skipped
                // here — they carry only a position, no rows.
                if !entry.batches.is_empty() && !already_flushed {
                    // Start a new memtable where the entry's recorded generation
                    // changes. Entries with no generation use the size rule below.
                    if let Some(generation) = entry.generation
                        && generation > active.generation()
                        && !active.batch_store().is_empty()
                    {
                        let global_end = active.batch_store().global_end();
                        wal_flusher.advance_durable(global_end);
                        flush_replayed_memtable(
                            flusher,
                            &active,
                            our_epoch,
                            position.saturating_sub(1),
                            global_end,
                            index_specs,
                        )
                        .await?;
                        active = make_memtable(generation, global_end, None)?;
                    }

                    if let Some(target) = entry.target.as_ref()
                        && active.target() != Some(target)
                    {
                        let expected_generation = if active.batch_store().is_empty() {
                            active.generation()
                        } else {
                            active.generation() + 1
                        };
                        // The memtable takes the target's generation number. If
                        // that number is not the next one, it was already used.
                        if target.generation != expected_generation {
                            return Err(Error::io(format!(
                                "WAL target generation {} at position {} does not follow active generation {}; the memtable replaying this WAL is smaller than the one that wrote it",
                                target.generation,
                                position,
                                active.generation()
                            )));
                        }
                        let global_end = active.batch_store().global_end();
                        if !active.batch_store().is_empty() {
                            wal_flusher.advance_durable(global_end);
                            flush_replayed_memtable(
                                flusher,
                                &active,
                                our_epoch,
                                position.saturating_sub(1),
                                global_end,
                                index_specs,
                            )
                            .await?;
                        }
                        active =
                            make_memtable(target.generation, global_end, Some(target.clone()))?;
                    }

                    // Re-label to the current storage schema; entries written
                    // before deletes existed also need `_tombstone = false`.
                    let storage_schema = active.schema().clone();
                    let batches = entry
                        .batches
                        .into_iter()
                        .map(|b| conform_to_storage_schema(b, &storage_schema, pk_columns))
                        .collect::<Result<Vec<_>>>()?;
                    // Reconstruct the target's allocator state from the
                    // prepared descriptors persisted in the WAL. Reservations
                    // are idempotent across rows and batches that share a pack,
                    // while the allocator skips every recovered ID for new
                    // sidecars written by this successor.
                    if active.target().is_some() {
                        let allocator = active.blob_id_allocator();
                        for batch in &batches {
                            for blob_id in prepared_blob_ids(batch)? {
                                allocator.reserve(blob_id)?;
                            }
                        }
                    }

                    let entry_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                    // Replay cannot split a recorded generation: the rest would
                    // reuse its number. Fail if the memtable cannot hold it.
                    // Byte thresholds do not count here, since replay widens old
                    // rows to the current schema.
                    if let Some(generation) = entry.generation
                        && !active.batch_store().is_empty()
                        && memtable_cannot_take(
                            &active,
                            max_memtable_rows,
                            batches.len(),
                            entry_rows,
                        )
                    {
                        return Err(Error::io(format!(
                            "WAL entry at position {} continues generation {} past what the \
                             memtable replaying it can hold ({} batches, {} rows); the shard \
                             was written under a larger max_memtable_batches or \
                             max_memtable_rows",
                            position,
                            generation,
                            active.batch_store().capacity(),
                            max_memtable_rows,
                        )));
                    }
                    // Measured over the whole entry, so no entry spans two memtables.
                    if entry.target.is_none()
                        && entry.generation.is_none()
                        && !active.batch_store().is_empty()
                        && memtable_reached_flush_threshold(
                            &active,
                            max_memtable_size,
                            max_memtable_rows,
                            max_resident_bytes,
                            batches.len(),
                            entry_rows,
                        )
                    {
                        let store = active.batch_store();
                        // The last entry this memtable fully absorbed is the one
                        // before the entry about to be inserted.
                        let covered = position.saturating_sub(1);
                        let generation = active.generation() + 1;
                        let global_end = store.global_end();

                        // The frozen data is already durable in the WAL — mark it
                        // so the flush's `all_flushed_to_wal` precondition holds and
                        // no WAL re-append is attempted.
                        wal_flusher.advance_durable(global_end);
                        flush_replayed_memtable(
                            flusher,
                            &active,
                            our_epoch,
                            covered,
                            global_end,
                            index_specs,
                        )
                        .await?;

                        active = make_memtable(generation, global_end, None)?;
                    }

                    active.insert_batches_only(batches).await?;
                    if entry.generation.is_some() {
                        last_replayed_generation = entry.generation;
                    }
                }
                position = position.checked_add(1).ok_or_else(|| {
                    Error::io(format!(
                        "WAL position overflow during replay for shard {}",
                        shard_id
                    ))
                })?;
            }
        }
    }

    index_replayed_batches(&active).await?;

    Ok(ReplayResult {
        active,
        next_wal_position: position,
    })
}

/// Index the batches a replayed memtable holds.
///
/// Replay inserts without indexing, and the flush builds the primary-key
/// sidecar from these indexes, so skipping this breaks lookup by key.
async fn index_replayed_batches(memtable: &MemTable) -> Result<()> {
    let Some(indexes) = memtable.indexes_arc() else {
        return Ok(());
    };
    let batch_count = memtable.batch_count();
    if batch_count == 0 {
        return Ok(());
    }
    let store = memtable.batch_store();
    let stored: Vec<StoredBatch> = (0..batch_count)
        .filter_map(|pos| store.get(pos).cloned())
        .collect();
    tokio::task::spawn_blocking(move || indexes.insert_batches(&stored))
        .await
        .map_err(|e| Error::internal(format!("WAL replay index update task panicked: {}", e)))??;
    Ok(())
}

/// Flush a frozen replay memtable to a generation, with its indexes if any.
/// `covered` becomes the generation's `replay_after_wal_entry_position`.
async fn flush_replayed_memtable(
    flusher: &MemTableFlusher,
    memtable: &MemTable,
    epoch: u64,
    covered: u64,
    durable: usize,
    index_specs: &[MemIndexSpec],
) -> Result<()> {
    index_replayed_batches(memtable).await?;
    if index_specs.is_empty() {
        flusher.flush(memtable, epoch, covered, durable).await?;
    } else {
        Box::pin(flusher.flush_with_indexes(memtable, epoch, index_specs, covered, durable))
            .await?;
    }
    Ok(())
}
