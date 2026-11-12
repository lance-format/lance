// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shard writer lifecycle and write coordination.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::Duration;

use arc_swap::ArcSwap;
use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_index::mem_wal::ShardManifest;
use lance_io::object_store::ObjectStore;
use log::{debug, info, warn};
use object_store::path::Path;
use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::instrument;
use uuid::Uuid;

use super::backpressure::{
    BackpressureController, BackpressureStatsSnapshot, DurabilityWatcher, ShardMemory,
    resolve_backpressure,
};
use super::config::ShardWriterConfig;
use super::handlers::{
    IndexApplyHandler, MemTableFlushHandler, WalFlushHandler, sweep_expired_frozen,
};
use super::replay::{ReplayResult, replay_memtable_from_wal};
use super::schema::{
    Incoming, WriterSchema, pk_index_columns, same_index_set, validate_against_logical_schema,
};
use super::stats::{
    MemTableStats, SharedWriteStats, WalStats, WriteStatsSnapshot, new_shared_stats,
};
use super::tasks::TaskExecutor;
use crate::dataset::DATA_DIR;
use crate::dataset::blob::BlobPreprocessor;
use crate::dataset::mem_wal::MemTableDataTarget;
use crate::dataset::mem_wal::MemTableScanner;
use crate::dataset::mem_wal::index::{IndexStore, MemIndexSpec};
use crate::dataset::mem_wal::manifest::ShardManifestStore;
use crate::dataset::mem_wal::memtable::MemTable;
use crate::dataset::mem_wal::memtable::batch_store::BatchStore;
use crate::dataset::mem_wal::memtable::flush::{MemTableFlusher, TriggerMemTableFlush};
use crate::dataset::mem_wal::reconcile::without_field_ids;
use crate::dataset::mem_wal::scanner::InMemoryMemTableRef;
use crate::dataset::mem_wal::util::WatchableOnceCell;
use crate::dataset::mem_wal::wal::{
    BatchDurableWatcher, TriggerIndexApply, TriggerWalFlush, WalAppender, WalFlushFailure,
    WalFlushResult, WalFlushSource, WalFlusher, WalOnlyState, WalRetryConfig, WriterCursors,
};
use crate::dataset::write::ExternalBlobMode;

/// Result of a write operation.
#[derive(Debug)]
pub struct WriteResult {
    /// Range of batch positions [start, end) for inserted batches.
    /// For a single batch, this is [pos, pos+1).
    pub batch_positions: std::ops::Range<usize>,
}

/// A frozen memtable kept queryable in memory. `flushed_at_ms` is `None` while
/// the generation is still awaiting (or retrying) its flush, and `Some(t)` once
/// the flush commits — after which it lingers for `frozen_memtable_grace` so
/// in-flight as-of reads keep batch-resolved membership, then is swept.
pub(super) struct FrozenMemTable {
    pub(super) memtable: Arc<MemTable>,
    pub(super) flushed_at_ms: Option<u64>,
}

/// What one shard holds in memory: detached size handles, plus the flush a
/// waiter should park on.
///
/// Data only. The byte arithmetic lives on [`ShardMemory`], which is the single
/// place it exists — this is just what gets published.
///
/// Lives outside [`WriterState`] on purpose. A caller deciding whether to admit
/// a write must read this *while* the writer holds the write lock, which is
/// exactly when a `try_read()` on that lock fails — and tokio's `RwLock` is
/// write-preferring, so it fails whenever a writer is merely queued. Reading
/// through the lock would therefore report zero precisely under load.
///
/// Replaced wholesale by [`publish_memory`] at the moments the memtable set
/// changes, and **derived** from `WriterState` each time rather than adjusted.
/// So there is no counter to keep paired with anything, nothing on the per-put
/// path, and no way for this and [`ShardWriter::memtable_stats`] to disagree
/// about what the shard holds: they read the same memtables through the same
/// filter.
///
/// The handles stay live, so byte totals track a memtable that is still growing
/// without anyone republishing.
#[derive(Default)]
pub(super) struct ResidentMemTables {
    /// The active memtable, or `None` before the first publish.
    pub(super) active: Option<InMemoryMemTableRef>,
    /// Frozen memtables whose flush has not committed — including any left
    /// resident by a *failed* flush, which are the ones most worth metering.
    pub(super) frozen: Vec<InMemoryMemTableRef>,
    /// Frozen memtables whose flush *did* commit, lingering out
    /// `frozen_memtable_grace` before [`sweep_expired_frozen`] drops them.
    ///
    /// Held apart from `frozen` rather than dropped from the view: no flush can
    /// reclaim these, so metering the flush valve on them would throttle against
    /// memory that is going away on a timer. They are still resident, though,
    /// and a process-wide budget has to see them — hence
    /// [`ShardMemory::retained_bytes`] alongside
    /// [`ShardMemory::unflushed_bytes`].
    pub(super) grace: Vec<InMemoryMemTableRef>,
    /// The oldest flush still outstanding, or `None` when none is.
    ///
    /// Deliberately not backfilled with the active memtable's watcher. That
    /// watcher only fires when the active memtable is frozen, which only a put
    /// does — so offering it to a waiter that is itself holding puts out names
    /// an event that cannot arrive. `None` is the honest answer, and
    /// [`ShardMemory::drain`] turns it into one.
    pub(super) oldest_flush: Option<DurabilityWatcher>,
}

/// Re-derive a shard's resident-memtable set from its writer state.
///
/// Call under the write lock after any change to that set: `open`,
/// `freeze_memtable`, a flush commit, and [`sweep_expired_frozen`] — which changes it by
/// evicting grace-expired generations that are still counted until it runs.
///
/// Cheap: two `Arc` clones per live memtable, no byte walk — the totals are
/// computed on read.
pub(super) fn publish_memory(memory: &ArcSwap<ResidentMemTables>, state: &WriterState) {
    let (grace, frozen) = state
        .frozen_memtables
        .iter()
        .partition::<Vec<_>, _>(|frozen| frozen.flushed_at_ms.is_some());
    let refs = |tables: Vec<&FrozenMemTable>| {
        tables
            .into_iter()
            .map(|frozen| in_memory_ref(&frozen.memtable))
            .collect()
    };
    memory.store(Arc::new(ResidentMemTables {
        active: Some(in_memory_ref(&state.memtable)),
        frozen: refs(frozen),
        grace: refs(grace),
        // Oldest first, so the front of the queue is what a waiter parks on.
        oldest_flush: state.frozen_flush_watchers.front().cloned(),
    }));
}

/// ShardWriter state shared across tasks.
pub(super) struct WriterState {
    pub(super) memtable: MemTable,
    /// The schema `memtable` was created under. Kept under the same lock so a
    /// write is checked against the schema of the memtable it lands in.
    pub(super) schema: Arc<WriterSchema>,
    pub(super) last_flushed_wal_entry_position: u64,
    /// Flush watchers for frozen memtables, oldest first. Carries no byte
    /// count: sizes are read live off `frozen_memtables` (see
    /// [`ResidentMemTables`]), so there is nothing here to keep paired.
    pub(super) frozen_flush_watchers: VecDeque<DurabilityWatcher>,
    /// Frozen memtables, kept queryable so a concurrent reader sees no hole
    /// between `freeze_memtable` and the flush task's manifest commit, and for
    /// `frozen_memtable_grace` beyond it so as-of reads stay batch-resolved.
    /// Pushed in `freeze_memtable`; stamped `flushed_at_ms` by `flush_memtable`
    /// on commit success only (retained un-stamped on failure until a later
    /// flush or WAL replay on reopen); swept after the grace by
    /// [`sweep_expired_frozen`].
    pub(super) frozen_memtables: VecDeque<FrozenMemTable>,
    /// Flag to prevent duplicate memtable flush requests.
    flush_requested: bool,
    /// Current memtable's claimed WAL-size threshold crossings.
    wal_flush_trigger_count: usize,
    /// Last write-path interval trigger. Its boundary is resolved when handled,
    /// so this recovers sequential-write latency without defeating group commit.
    last_wal_flush_trigger_time: u64,
}

/// Capture a point-in-time handle to one in-memory memtable (active or frozen
/// — same shape).
///
/// The single projection of a memtable in this crate: the read path scans
/// through it, and [`ShardMemory`] sizes through it. Everything it holds is an
/// `Arc` or a copy, so a handle is cheap and stays live as the memtable grows —
/// which is what lets the memory view be read without the writer lock.
fn in_memory_ref(mt: &MemTable) -> InMemoryMemTableRef {
    InMemoryMemTableRef {
        batch_store: mt.batch_store(),
        index_store: mt
            .indexes_arc()
            .unwrap_or_else(|| Arc::new(IndexStore::new())),
        schema: mt.scan_schema().clone(),
        generation: mt.generation(),
    }
}

fn start_time() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

pub(super) fn now_millis() -> u64 {
    start_time().elapsed().as_millis() as u64
}

/// Whether the freeze a put needed actually happened. Pre-insert callers turn
/// `Blocked` into a refusal; the post-insert caller leaves the memtable full
/// for the next put to be refused on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FreezeOutcome {
    /// Nothing needed freezing, or the memtable was frozen.
    Settled,
    /// A freeze was required and the controller is holding it back.
    Blocked,
}

/// The refusal a [`FreezeOutcome::Blocked`] turns into on a put that needed the
/// freeze. Retryable: the controller admits freezes again once the tier drains.
fn freeze_blocked_error() -> Error {
    Error::backpressure(
        "memtable is full and freezing is held back because the tier below it has \
         no room; retry once compaction drains it"
            .to_string(),
    )
}

/// Whether a memtable has reached the threshold at which it should be frozen and
/// flushed.
///
/// The single source of truth for the flush trigger, shared by the live put path
/// (`maybe_trigger_memtable_flush`) and by replay so the two cannot drift.
///
/// `incoming_batches` / `incoming_rows` are what is about to be inserted.
/// Pre-insert callers pass the real counts; post-insert callers pass `(1, 1)`,
/// asking whether there is room for one more batch.
///
/// Four arms, each answering a different question:
///
/// - **Row window** against `max_memtable_size`. The knob an operator sizes: it
///   measures what a flush actually writes, so a generation stays a predictable
///   fragment in the base dataset. Deliberately the *only* thing charged to this
///   threshold — index memory and buffer padding are not, or fragment size would
///   start depending on index configuration and Arrow's allocator.
/// - **Resident total** against `max_resident_bytes`, the backpressure ceiling.
///   The row window bounds neither what the batches *pin* (a one-row slice holds
///   its whole parent) nor what the indexes hold, so without this arm a memtable
///   can carry a shard past its ceiling with no freeze reachable — and the valve,
///   finding nothing outstanding to wait on, would refuse writes that can never
///   succeed. This is the drain path that makes the ceiling live rather than a
///   trap.
/// - **Batch-store capacity**, room for `incoming_batches` more.
/// - **Row count** against `max_memtable_rows`, room for `incoming_rows` more.
///   A hard capacity rather than a target: the in-memory indexes are
///   pre-allocated to exactly this many rows, so an overshoot fails the index
///   apply. The live path checks this arm pre-insert as well.
pub(super) fn memtable_reached_flush_threshold(
    memtable: &MemTable,
    max_memtable_size: usize,
    max_memtable_rows: usize,
    max_resident_bytes: usize,
    incoming_batches: usize,
    incoming_rows: usize,
) -> bool {
    fill_reached_flush_threshold(
        &memtable.batch_store(),
        memtable_resident_bytes(memtable),
        max_memtable_size,
        max_memtable_rows,
        max_resident_bytes,
        incoming_batches,
        incoming_rows,
    )
}

/// Whether a memtable has no room for `incoming` batches and rows.
///
/// Only the batch and row caps count. The byte limits in
/// [`fill_reached_flush_threshold`] mean "time to freeze", not "full".
pub(super) fn memtable_cannot_take(
    memtable: &MemTable,
    max_memtable_rows: usize,
    incoming_batches: usize,
    incoming_rows: usize,
) -> bool {
    let store = memtable.batch_store();
    store.remaining_capacity() < incoming_batches
        || store.total_rows().saturating_add(incoming_rows) > max_memtable_rows
}

/// [`memtable_reached_flush_threshold`] over a memtable's contents rather than
/// the memtable itself, so admission can evaluate the same arms against the
/// published snapshot without the write lock. One predicate, so a put cannot be
/// refused for a freeze the writer would not have made.
#[allow(clippy::too_many_arguments)]
fn fill_reached_flush_threshold(
    store: &BatchStore,
    resident_bytes: usize,
    max_memtable_size: usize,
    max_memtable_rows: usize,
    max_resident_bytes: usize,
    incoming_batches: usize,
    incoming_rows: usize,
) -> bool {
    store.row_bytes() >= max_memtable_size
        || resident_bytes >= max_resident_bytes
        || store.remaining_capacity() < incoming_batches
        || store.total_rows().saturating_add(incoming_rows) > max_memtable_rows
}

/// What this memtable holds in memory: the heap its batches pin plus its
/// in-memory indexes. The same quantity [`ShardMemory`] reports for the active
/// memtable, so the freeze trigger and the ceiling that gates writes measure the
/// same thing.
fn memtable_resident_bytes(memtable: &MemTable) -> usize {
    memtable.batch_store().retained_bytes()
        + memtable.indexes().map_or(0, IndexStore::resident_bytes)
        + super::super::memtable::pk_bloom_filter_bytes()
}

/// Shared state for writer operations.
struct SharedWriterState {
    /// Detached size handles for every memtable this shard holds, shared with
    /// the memtable flush handler (which re-derives them on commit).
    memory: Arc<ArcSwap<ResidentMemTables>>,
    wal_flusher: Arc<WalFlusher>,
    wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
    /// The index-apply task's channel. Separate from the WAL flusher's on
    /// purpose: `TaskDispatcher::run` awaits `handle()` inline, so sharing one
    /// would put a ~100ms S3 PUT in front of every latency-sensitive index apply.
    index_apply_tx: mpsc::UnboundedSender<TriggerIndexApply>,
    memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
    config: ShardWriterConfig,
    epoch: u64,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    shard_id: Uuid,
    max_memtable_batches: usize,
    max_memtable_rows: usize,
    /// Cancelled by `close`, so a failing memtable flush stops retrying.
    stop_flush_retries: CancellationToken,
}

impl SharedWriterState {
    /// Bundle the handles the writer's put, freeze, and close paths share.
    #[allow(clippy::too_many_arguments)]
    fn new(
        memory: Arc<ArcSwap<ResidentMemTables>>,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        index_apply_tx: mpsc::UnboundedSender<TriggerIndexApply>,
        memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
        config: ShardWriterConfig,
        epoch: u64,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        shard_id: Uuid,
        stop_flush_retries: CancellationToken,
    ) -> Self {
        Self {
            memory,
            wal_flusher,
            wal_flush_tx,
            index_apply_tx,
            memtable_flush_tx,
            max_memtable_batches: config.max_memtable_batches,
            max_memtable_rows: config.max_memtable_rows,
            config,
            epoch,
            object_store,
            base_path,
            shard_id,
            stop_flush_retries,
        }
    }

    /// Write the batches' blob payloads into the memtable's generation; batches
    /// pass through untouched when it has no data target. The flag is true when
    /// the payloads sit in the generation's shared pack, which the WAL flush finishes.
    async fn prepare_batches(
        &self,
        memtable: &MemTable,
        schema: &WriterSchema,
        batches: Vec<RecordBatch>,
    ) -> Result<(Vec<RecordBatch>, Option<BlobPreprocessor>, bool)> {
        let Some(target) = memtable.target() else {
            return Ok((batches, None, false));
        };
        let logical_schema = Schema::try_from(schema.storage.as_ref())?;
        let data_dir = target
            .generation_path(&self.base_path, &self.config.dir_id, &self.shard_id)
            .join(DATA_DIR);
        let source_store_registry = self
            .config
            .session
            .as_ref()
            .map(|session| session.store_registry())
            .unwrap_or_default();
        let source_store_params = self.config.store_params.clone().unwrap_or_default();
        let new_preprocessor = || {
            BlobPreprocessor::new(
                self.object_store.as_ref().clone(),
                data_dir.clone(),
                target.data_file_key().to_string(),
                &logical_schema,
                None,
                true,
                ExternalBlobMode::Reference,
                source_store_registry,
                source_store_params,
                None,
            )
            .map(|preprocessor| preprocessor.for_mem_wal(memtable.blob_id_allocator()))
        };

        // A non-durable put becomes visible when its in-memory index apply
        // finishes, so its sidecar must already be readable at that point.
        // Preserve the original per-put finalization for that mode.
        if !self.config.durable_write {
            let mut preprocessor = new_preprocessor()?;

            for batch in &batches {
                preprocessor.validate_batch(batch).await?;
            }
            let prepared = async {
                let mut prepared = Vec::with_capacity(batches.len());
                for batch in batches {
                    prepared.push(preprocessor.preprocess_batch(&batch).await?);
                }
                preprocessor.finish().await?;
                Ok(prepared)
            }
            .await;
            if prepared.is_err() {
                preprocessor.abort();
            }
            return prepared.map(|batches| (batches, Some(preprocessor), false));
        }

        // Durable puts share one preprocessor per MemTable generation. Keeping
        // its partial pack open lets sequential put_no_wait calls use one object;
        // WalFlusher finishes it before appending the corresponding descriptors.
        let batch_store = memtable.batch_store();
        let mut guard = batch_store.blob_preprocessor().lock().await;
        {
            let preprocessor = guard.get_or_insert_with(new_preprocessor)?;
            for batch in &batches {
                preprocessor.validate_batch(batch).await?;
            }
        }
        let preprocessor = guard.begin_operation()?;
        let prepared = async {
            let mut prepared = Vec::with_capacity(batches.len());
            for batch in batches {
                prepared.push(preprocessor.preprocess_batch(&batch).await?);
            }
            Ok(prepared)
        }
        .await;
        match prepared {
            Ok(batches) => {
                guard.complete_operation();
                Ok((batches, None, true))
            }
            Err(error) => {
                // The open pack may contain payloads from earlier
                // unacknowledged puts. Latch a terminal pack failure so even a
                // flush that was already queued cannot append their descriptors
                // after the upload is abandoned.
                guard.fail(&error);
                self.wal_flusher.poison(&error);
                Err(error)
            }
        }
    }

    /// Ask the index-apply task to cover `[indexed, end_batch_position)` of this
    /// store. Cheap and idempotent: a range already covered is a no-op, which is
    /// the common case under load, since one apply coalesces the puts queued
    /// behind it.
    fn trigger_index_apply(
        &self,
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        end_batch_position: usize,
    ) -> Result<()> {
        self.index_apply_tx
            .send(TriggerIndexApply {
                batch_store,
                indexes,
                end_batch_position,
            })
            .map_err(|_| Error::io("index apply channel closed"))
    }

    /// Freeze the current memtable and send it to the flush handler.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn freeze_memtable(&self, state: &mut WriterState) -> Result<u64> {
        let schema = Arc::clone(&state.schema);
        self.rotate_memtable(state, schema)
    }

    /// Freeze the current memtable under its own schema, and start the next one
    /// under `next_schema`.
    fn rotate_memtable(
        &self,
        state: &mut WriterState,
        next_schema: Arc<WriterSchema>,
    ) -> Result<u64> {
        let durable = self.wal_flusher.durable();
        let pending_wal_range = state
            .memtable
            .batch_store()
            .pending_wal_flush_range(durable);
        let last_wal_entry_position = state.last_flushed_wal_entry_position;

        let old_batch_store = state.memtable.batch_store();

        let next_generation = state.memtable.generation() + 1;
        // The incoming memtable's batch 0 continues the writer's batch sequence
        // where the outgoing one ends. Without this coordinate, local positions
        // (which restart at 0 every rotation) cannot be mapped onto the
        // writer-global durability cursor.
        let next_global_offset = old_batch_store.global_end();
        let new_memtable = next_schema.new_memtable(
            next_generation,
            next_global_offset,
            None,
            self.epoch,
            self.max_memtable_rows,
            self.max_memtable_batches,
            self.wal_flusher.cursors(),
        )?;

        let mut old_memtable = std::mem::replace(&mut state.memtable, new_memtable);
        state.wal_flush_trigger_count = 0;
        // The outgoing memtable flushes with the indexes it was built with.
        let old_schema = std::mem::replace(&mut state.schema, next_schema);
        old_memtable.freeze(last_wal_entry_position);

        // Set up completion tracking on the outgoing table before it is retained
        // and before any fallible dispatch, so the retained table already carries
        // its cells and a failed send below can poison-and-return without leaving
        // partial state to unwind.
        let _memtable_flush_watcher = old_memtable.create_memtable_flush_completion();

        // The outgoing memtable may still owe an index apply — the puts that
        // filled it triggered one, but the task need not have drained yet, and
        // this is the last chance to name that store. Its SSTable flush is gated on
        // the WAL append (below), not on indexing, so without this its tail could
        // stay unindexed and invisible for the rest of its life.
        let pending_index_apply = match old_memtable.indexes_arc() {
            Some(old_indexes) if old_indexes.indexed_count() < old_batch_store.len() => {
                Some((old_batch_store.clone(), old_indexes, old_batch_store.len()))
            }
            _ => None,
        };

        let pending_wal_flush = if pending_wal_range.is_some() {
            let completion_cell: WatchableOnceCell<
                std::result::Result<WalFlushResult, WalFlushFailure>,
            > = WatchableOnceCell::new();
            old_memtable.set_wal_flush_completion(completion_cell.reader());
            Some((old_batch_store.len(), completion_cell))
        } else {
            None
        };

        let flush_watcher = old_memtable
            .get_memtable_flush_watcher()
            .expect("Flush watcher should exist after create_memtable_flush_completion");
        state.frozen_flush_watchers.push_back(flush_watcher);

        let frozen_memtable = Arc::new(old_memtable);

        // Retain the outgoing table in the read view *before* the fallible
        // dispatches below. `state.memtable` was already replaced, so a failed
        // send that returned here without this push would drop the table and its
        // accepted rows would silently vanish from every scan. Keep it queryable
        // past its manifest commit too (swept after the grace by
        // `sweep_expired_frozen`);
        // Arc refcount, not a copy — the flush task holds it alive anyway.
        state.frozen_memtables.push_back(FrozenMemTable {
            memtable: frozen_memtable.clone(),
            flushed_at_ms: None,
        });

        // The memtable set changed: re-derive. Before the fallible dispatches
        // below, so a poisoned writer still reports the bytes it is holding.
        publish_memory(&self.memory, state);

        // Dispatch can only fail if a background task's channel is already closed,
        // i.e. the writer is being torn down. Poison so the read path fails fast
        // with the typed error instead of serving the retained-but-never-durable
        // tail, then return — the table stays in the read view.
        if let Some((batch_store, indexes, end_batch_position)) = pending_index_apply {
            self.trigger_index_apply(batch_store, indexes, end_batch_position)
                .inspect_err(|e| self.wal_flusher.poison(e))?;
        }

        if let Some((end_batch_position, completion_cell)) = pending_wal_flush {
            self.wal_flusher
                .trigger_flush(
                    WalFlushSource::BatchStore {
                        batch_store: old_batch_store,
                    },
                    end_batch_position,
                    Some(completion_cell),
                )
                .inspect_err(|e| self.wal_flusher.poison(e))?;
        }

        debug!(
            "Frozen memtable generation {}, pending_count = {}",
            next_generation - 1,
            state.frozen_flush_watchers.len()
        );

        let _ = self.memtable_flush_tx.send(TriggerMemTableFlush::Flush {
            memtable: frozen_memtable,
            index_specs: Arc::clone(&old_schema.index_specs),
            done: None,
        });

        Ok(next_generation)
    }

    /// Watch for a write to become visible: indexed, and — in durable mode —
    /// WAL-durable too.
    ///
    /// `target_indexed` is memtable-local; `target_durable` is writer-global.
    /// Different coordinate spaces on purpose — see `WalFlusher::track_batch`.
    fn track_batch_for_wal(
        &self,
        indexes: Option<Arc<IndexStore>>,
        target_indexed: usize,
        target_durable: usize,
    ) -> BatchDurableWatcher {
        self.wal_flusher
            .track_batch(indexes, target_indexed, target_durable)
    }

    /// Check if memtable flush is needed and trigger if so.
    ///
    /// `incoming_batches` / `incoming_rows`: see [`memtable_reached_flush_threshold`].
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_memtable_flush(
        &self,
        state: &mut WriterState,
        incoming_batches: usize,
        incoming_rows: usize,
    ) -> Result<FreezeOutcome> {
        if state.flush_requested {
            return Ok(FreezeOutcome::Settled);
        }

        // An empty memtable has nothing to freeze, and freezing one would spin: its
        // indexes alone can sit above the ceiling.
        if state.memtable.batch_count() == 0 {
            return Ok(FreezeOutcome::Settled);
        }

        let should_flush = memtable_reached_flush_threshold(
            &state.memtable,
            self.config.max_memtable_size,
            self.config.max_memtable_rows,
            self.config.max_unflushed_memtable_bytes,
            incoming_batches,
            incoming_rows,
        );

        if !should_flush {
            return Ok(FreezeOutcome::Settled);
        }

        // The tier a flush lands in has no room for another generation. The
        // memtable stays at its cap, so a put that needed this freeze is refused:
        // its rows would overrun the capacity the indexes are allocated to.
        if !self.may_freeze_memtable() {
            return Ok(FreezeOutcome::Blocked);
        }

        state.flush_requested = true;
        self.freeze_memtable(state)?;
        state.flush_requested = false;
        Ok(FreezeOutcome::Settled)
    }

    /// Whether the installed controller is admitting freezes right now. An
    /// unset one resolves to [`super::backpressure::LocalBackpressureController`], which takes the
    /// trait default and admits everything.
    fn may_freeze_memtable(&self) -> bool {
        match &self.config.backpressure {
            Some(controller) => controller.may_freeze_memtable(),
            None => true,
        }
    }

    /// Whether a put of this shape can only land after a freeze — the predicate
    /// [`Self::maybe_trigger_memtable_flush`] acts on, off the published snapshot
    /// so admission can decide without the write lock.
    ///
    /// Racy against a concurrent freeze in both directions; the writer re-checks
    /// under the lock and a parked controller re-reads on its next poll.
    fn freeze_required(&self, incoming_batches: usize, incoming_rows: usize) -> bool {
        let tables = self.memory.load();
        let Some(active) = tables.active.as_ref() else {
            return false;
        };
        // Matches the empty-memtable early-out above: nothing to freeze.
        if active.batch_store.is_empty() {
            return false;
        }
        fill_reached_flush_threshold(
            &active.batch_store,
            active.resident_bytes(),
            self.config.max_memtable_size,
            self.config.max_memtable_rows,
            self.config.max_unflushed_memtable_bytes,
            incoming_batches,
            incoming_rows,
        )
    }

    /// Check whether the current memtable crossed a WAL size or time threshold.
    ///
    /// Both signals are best-effort and resolve the latest pending suffix when
    /// the handler drains the message. This preserves group commit while the
    /// write-path clock prevents a sequential durable client from waiting for a
    /// fresh ticker period after every completed append. Freeze and close use
    /// their own exact, completion-bearing triggers.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_wal_flush(&self, state: &mut WriterState) {
        let now = now_millis();
        let time_trigger = claim_wal_time_trigger(
            &mut state.last_wal_flush_trigger_time,
            now,
            self.config.max_wal_flush_interval,
        );
        let total_bytes = state.memtable.batch_store().row_bytes();
        let size_trigger = claim_wal_size_trigger(
            &mut state.wal_flush_trigger_count,
            total_bytes,
            self.config.max_wal_buffer_size,
        );
        if !time_trigger && !size_trigger {
            return;
        }
        if size_trigger {
            state.last_wal_flush_trigger_time = now;
        }

        let _ = self.wal_flush_tx.send(TriggerWalFlush {
            source: WalFlushSource::NextPending,
            end_batch_position: 0,
            done: None,
        });
    }
}

/// Claim one write-path interval trigger without fixing its flush boundary.
fn claim_wal_time_trigger(
    last_trigger_ms: &mut u64,
    now_ms: u64,
    interval: Option<Duration>,
) -> bool {
    let Some(interval_ms) = interval
        .filter(|interval| !interval.is_zero())
        .and_then(|interval| u64::try_from(interval.as_millis()).ok())
    else {
        return false;
    };
    if *last_trigger_ms == 0 {
        *last_trigger_ms = now_ms;
        return false;
    }
    if now_ms.saturating_sub(*last_trigger_ms) < interval_ms {
        return false;
    }
    *last_trigger_ms = now_ms;
    true
}

/// Claim all WAL-size thresholds crossed by the current put as one trigger.
///
/// A single append covers the whole pending suffix, so queueing one message per
/// crossed threshold only creates stale boundaries under load. The counter is
/// generation-local and is reset when the active memtable rotates.
fn claim_wal_size_trigger(
    claimed_thresholds: &mut usize,
    total_bytes: usize,
    threshold: usize,
) -> bool {
    if threshold == 0 {
        return false;
    }
    let crossed = total_bytes / threshold;
    if crossed <= *claimed_thresholds {
        return false;
    }
    *claimed_thresholds = crossed;
    true
}

/// Claim WAL-size thresholds for a queue whose byte count shrinks on drain.
fn claim_pending_wal_size_trigger(
    claimed_bytes: &mut usize,
    pending_bytes: usize,
    threshold: usize,
) -> bool {
    if threshold == 0 {
        return false;
    }
    if pending_bytes < *claimed_bytes {
        *claimed_bytes = 0;
    }
    let crossed = (pending_bytes - *claimed_bytes) / threshold;
    if crossed == 0 {
        return false;
    }
    *claimed_bytes += crossed * threshold;
    true
}

/// Trigger-tracking state for WAL-only mode (no MemTable).
///
/// MemTable mode keeps these counters inside `WriterState`. WAL-only mode
/// has no `WriterState`, so the same counters live here, with one
/// adjustment: instead of a monotonic threshold-crossing count (which
/// can't decrement when the pending queue drains), we record the
/// `pending_bytes` value at the time of the last size-trigger. A trigger
/// fires whenever `pending_bytes` has grown by at least one
/// `max_wal_buffer_size` since `last_trigger_pending_bytes`. When the
/// pending queue is drained, `pending_bytes` drops below the recorded
/// baseline, and the next push detects this via `pending_bytes < baseline`
/// and resets the baseline.
#[derive(Debug, Default)]
struct WalOnlyTriggerState {
    /// `pending_bytes` value recorded the last time a size-based trigger
    /// fired. Resets to 0 when `pending_bytes` drops below it (drain
    /// happened since the last trigger).
    last_trigger_pending_bytes: usize,
    /// Last write-path interval trigger. See [`claim_wal_time_trigger`].
    last_wal_flush_trigger_time: u64,
}

/// Per-mode state for `ShardWriter`.
enum WriterMode {
    /// Default mode: MemTable + indexes + Lance file flushing on top of
    /// the WAL.
    MemTable {
        state: Arc<RwLock<WriterState>>,
        writer_state: Arc<SharedWriterState>,
        backpressure: Arc<dyn BackpressureController>,
    },
    /// WAL-only mode: drainable pending-batch queue + WAL pipeline. No
    /// MemTable, no indexes, no Lance file flushing.
    WalOnly {
        state: Arc<WalOnlyState>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: StdRwLock<WalOnlyTriggerState>,
        backpressure: Arc<dyn BackpressureController>,
        /// The base table's schema as the caller passed it, without field ids.
        /// Caller input is checked against it. Unlike memtable mode, it never
        /// changes.
        logical_schema: Arc<ArrowSchema>,
    },
}

/// Main writer for a MemWAL shard.
///
/// Transient SSTable flush failures are retried before any later generation flushes.
/// A terminal failure, including when close cancels retries, poisons the writer:
/// subsequent reads, writes, freezes, drains, and close report a terminal error.
/// Later generations cannot publish past that failure. Reopen the writer to
/// recover durable rows through WAL replay.
pub struct ShardWriter {
    config: ShardWriterConfig,
    epoch: u64,
    wal_flusher: Arc<WalFlusher>,
    task_executor: Arc<TaskExecutor>,
    manifest_store: Arc<ShardManifestStore>,
    stats: SharedWriteStats,
    mode: WriterMode,
}

impl ShardWriter {
    /// Open or create a ShardWriter.
    ///
    /// The `base_path` should come from `ObjectStore::from_uri()` to ensure
    /// WAL files are written inside the dataset directory.
    ///
    /// `schema` should carry each field's id under `lance:field_id` in its
    /// field metadata; without them a replayed entry is matched by name, which
    /// a rename loses.
    #[instrument(name = "sw_open", level = "info", skip_all, fields(shard_id = %config.shard_id, index_count = index_specs.len()))]
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        index_specs: Vec<MemIndexSpec>,
    ) -> Result<Self> {
        Box::pin(Self::open_inner(
            object_store,
            base_path,
            base_uri.into(),
            config,
            schema,
            index_specs,
        ))
        .await
    }

    /// Body of [`Self::open`].
    async fn open_inner(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: String,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        index_specs: Vec<MemIndexSpec>,
    ) -> Result<Self> {
        if !config.enable_memtable && !index_specs.is_empty() {
            return Err(Error::invalid_input(
                "indexes require enable_memtable = true; \
                 WAL-only mode does not maintain in-memory indexes",
            ));
        }

        // A durable writer needs a flush ticker to make progress, in either
        // mode. With `durable_write` on, a put becomes durable only once its WAL
        // append lands, and neither mode self-triggers that append per put — the
        // background ticker drives it. Without an interval (or with a zero one,
        // which tokio cannot schedule), a small put that never fills the
        // size-triggered buffer would block until close. Reject the config here
        // rather than let a put hang.
        if config.durable_write && config.max_wal_flush_interval.is_none_or(|d| d.is_zero()) {
            return Err(Error::invalid_input(
                "durable_write requires a positive max_wal_flush_interval: with no \
                 flush ticker a durable put has nothing to drive its WAL append and \
                 would block until close",
            ));
        }

        let shard_id = config.shard_id;
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            &config.dir_id,
            shard_id,
            config.manifest_scan_batch_size,
        ));

        // Derive PK metadata and run every side-effect-free validation *before*
        // claiming the epoch. `claim_epoch` durably bumps the stored epoch and,
        // for a successor, `write_fence_sentinel` fences the predecessor — so an
        // open doomed by purely local input (an index spec that disagrees with
        // the schema) must fail here, before it can knock the healthy incumbent off
        // the shard. Memtable-only: WAL-only mode has no indexes to validate.
        let writer_schema = if config.enable_memtable {
            let writer_schema = WriterSchema::try_new(&schema, index_specs)?;

            // An HNSW graph reserves its fixed part before the first insert,
            // but the freeze trigger only measures row bytes. A reservation with no
            // room left under the ceiling puts the shard over budget at zero rows
            // — nothing to freeze, so nothing to flush, so every put stalls and then
            // fails as `Error::Backpressure`, which is supposed to mean "retry
            // later". Reject the config instead; only the built-in valve reads
            // this ceiling.
            if config.backpressure.is_none() {
                // The headroom the check below reserves for rows. At zero it
                // reserves nothing, so a fresh memtable's index reservation may
                // *equal* the ceiling — over budget before its first row, with an
                // empty memtable there is nothing to freeze, and the writer refuses
                // its first write forever. A zero threshold is degenerate anyway:
                // it freezes every memtable at every insert.
                if config.max_memtable_size == 0 {
                    return Err(Error::invalid_input(
                        "max_memtable_size must be greater than zero: it is both the \
                         freeze threshold for row data and the headroom reserved for rows \
                         under max_unflushed_memtable_bytes, and at zero a writer with \
                         in-memory indexes can be at its ceiling before its first row",
                    ));
                }

                // Built the way `make_bound_memtable` builds it below, so this is
                // the figure the controller will actually read.
                let mut indexes = IndexStore::from_specs(
                    &writer_schema.index_specs,
                    &writer_schema.lance_schema,
                    config.max_memtable_rows,
                    config.max_memtable_batches,
                )?;
                if !writer_schema.pk_columns.is_empty() {
                    indexes.enable_pk_index(&pk_index_columns(
                        &writer_schema.pk_columns,
                        &writer_schema.pk_field_ids,
                    ));
                }
                let reserved =
                    indexes.resident_bytes() + super::super::memtable::pk_bloom_filter_bytes();
                // Room for a full memtable of rows on top, or the ceiling is
                // crossed before `max_memtable_size` can freeze.
                let needed = reserved.saturating_add(config.max_memtable_size);
                if needed > config.max_unflushed_memtable_bytes {
                    return Err(Error::invalid_input(format!(
                        "in-memory indexes reserve {reserved} bytes at \
                         max_memtable_rows={}, and max_memtable_size={} must fit alongside them, \
                         needing {needed} bytes; max_unflushed_memtable_bytes={} is below that, \
                         so the active memtable would cross the backpressure ceiling before \
                         accruing enough row bytes to freeze, stalling every write. Raise \
                         max_unflushed_memtable_bytes to at least {needed}, or lower \
                         max_memtable_rows / max_memtable_size",
                        config.max_memtable_rows,
                        config.max_memtable_size,
                        config.max_unflushed_memtable_bytes,
                    )));
                }
            }

            Some(writer_schema)
        } else {
            None
        };

        // Claim the shard (epoch-based fencing) — done once, then shared
        // with the WalAppender via `with_claimed_epoch`.
        let (epoch, manifest) = manifest_store.claim_epoch(config.shard_spec_id).await?;

        info!(
            "Opened ShardWriter for shard {} (epoch {}, generation {}, enable_memtable {})",
            shard_id, epoch, manifest.current_generation, config.enable_memtable
        );

        // Create WAL appender (owns object store, epoch, and position
        // state). Seed the appender's stats hint from the higher of the
        // manifest's two cursors: `replay_after_wal_entry_position` is
        // updated authoritatively at every MemTable flush, while
        // `wal_entry_position_last_seen` is a best-effort hint that may
        // lag behind. Either can lead the other depending on which was
        // updated last, so take the max (then +1) to get the most
        // accurate post-recovery cursor for `wal_stats()`.
        let position_hint_seed = manifest
            .wal_entry_position_last_seen
            .max(manifest.replay_after_wal_entry_position)
            .saturating_add(1);
        let wal_appender = Arc::new(WalAppender::with_claimed_epoch(
            object_store.clone(),
            base_path.clone(),
            &config.dir_id,
            shard_id,
            manifest_store.clone(),
            epoch,
            position_hint_seed,
            WalRetryConfig {
                max_retries: config.max_wal_persist_retries,
                base_delay: config.wal_persist_retry_base_delay,
            },
        ));

        // Fence the predecessor before replay (see `write_fence_sentinel`).
        // Epoch 1 is a fresh shard with no predecessor to fence.
        if epoch >= 2 {
            wal_appender.write_fence_sentinel().await?;
        }

        // Create WAL flusher backed by the shared appender.
        // Build the cursors from *this writer's* config. `durable_write` is what
        // decides whether durability is part of visibility, so a flusher that
        // defaulted it would leave a non-durable put waiting on a durability
        // cursor nothing ever advances.
        let cursors = Arc::new(WriterCursors::new(config.durable_write));
        let mut wal_flusher = WalFlusher::with_cursors(wal_appender, cursors);

        let (wal_flush_tx, wal_flush_rx) = mpsc::unbounded_channel();
        wal_flusher.set_flush_channel(wal_flush_tx.clone());
        let wal_flusher = Arc::new(wal_flusher);

        let stats = new_shared_stats();
        let task_executor = Arc::new(TaskExecutor::new());

        let mode = if let Some(writer_schema) = writer_schema {
            Box::pin(Self::open_memtable_mode(
                &config,
                Arc::new(writer_schema),
                &manifest,
                wal_flusher.clone(),
                wal_flush_tx,
                wal_flush_rx,
                object_store.clone(),
                base_path,
                base_uri,
                shard_id,
                epoch,
                manifest_store.clone(),
                stats.clone(),
                &task_executor,
            ))
            .await?
        } else {
            Self::open_wal_only_mode(
                &config,
                Arc::new(without_field_ids(&schema)),
                wal_flusher.clone(),
                wal_flush_tx,
                wal_flush_rx,
                stats.clone(),
                &task_executor,
            )?
        };

        Ok(Self {
            config,
            epoch,
            wal_flusher,
            task_executor,
            manifest_store,
            stats,
            mode,
        })
    }

    /// Replay the WAL into memtables, then start the WAL-flush, index-apply,
    /// and memtable-flush tasks.
    #[allow(clippy::too_many_arguments)]
    async fn open_memtable_mode(
        config: &ShardWriterConfig,
        schema: Arc<WriterSchema>,
        manifest: &ShardManifest,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        wal_flush_rx: mpsc::UnboundedReceiver<TriggerWalFlush>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: String,
        shard_id: Uuid,
        epoch: u64,
        manifest_store: Arc<ShardManifestStore>,
        stats: SharedWriteStats,
        task_executor: &Arc<TaskExecutor>,
    ) -> Result<WriterMode> {
        // PK metadata and index/interval validation were resolved in `open`
        // before the epoch was claimed (a doomed open must not fence the
        // incumbent first).

        // Replay creates every memtable under the schema the writer opens with.
        // Entries written under an older schema are converted as they are read.
        let make_bound_memtable = |generation: u64,
                                   global_offset: usize,
                                   target: Option<MemTableDataTarget>|
         -> Result<MemTable> {
            schema.new_memtable(
                generation,
                global_offset,
                target,
                epoch,
                config.max_memtable_rows,
                config.max_memtable_batches,
                wal_flusher.cursors(),
            )
        };

        // The flusher writes frozen memtables to Lance generations — both the
        // ones replay freezes below and the ones the live path freezes later.
        let flusher = Arc::new(
            MemTableFlusher::new(
                object_store.clone(),
                base_path.clone(),
                base_uri.clone(),
                config.dir_id.clone(),
                shard_id,
                manifest_store.clone(),
            )
            .with_warmer(config.warmer.clone())
            .with_storage_context(config.store_params.clone(), config.session.clone()),
        );

        // Replay any WAL entries written after the last successfully-flushed
        // SSTable, flushing frozen memtables to Lance SSTables as the batch
        // store fills. Each entry's writer_epoch is checked against ours; an entry
        // with a strictly greater epoch means a successor claimed the shard
        // between our `claim_epoch` and replay, so we abort with a fence error.
        // Replay walks the tailer to the WAL tip and returns the discovered
        // next-write position, so the appender's first append skips the
        // discover_next_position probe.
        let ReplayResult {
            active: memtable,
            next_wal_position,
        } = replay_memtable_from_wal(
            object_store.clone(),
            base_path.clone(),
            &config.dir_id,
            shard_id,
            epoch,
            manifest,
            manifest.current_generation,
            make_bound_memtable,
            &schema.pk_columns,
            &flusher,
            &wal_flusher,
            &schema.index_specs,
            config.max_memtable_size,
            config.max_memtable_rows,
            config.max_unflushed_memtable_bytes,
        )
        .await?;

        // Publish the read cursor once, now that replay knows the tip. The
        // tailer used to write this per entry, but a tailer holds no claim;
        // here it rides the epoch holder's normal commit path. Best-effort:
        // it is a hint for other readers, and `position_hint_seed` already
        // tolerates it lagging behind `replay_after_wal_entry_position`.
        let replayed_through = next_wal_position.saturating_sub(1);
        if replayed_through > manifest.wal_entry_position_last_seen
            && let Err(error) = manifest_store
                .commit_update(epoch, |current| ShardManifest {
                    version: current.next_version(),
                    wal_entry_position_last_seen: current
                        .wal_entry_position_last_seen
                        .max(replayed_through),
                    ..current.clone()
                })
                .await
        {
            warn!(
                "failed to publish WAL read cursor {} for shard {}: {}",
                replayed_through, shard_id, error
            );
        }

        // Mark the active memtable's replayed batches durable. They came *from*
        // the WAL, and replay has already re-derived its indexes over them.
        //
        // Without this the durability cursor stays at the last frozen generation,
        // so the next WAL flush re-covers the active tail: it re-appends the
        // already-durable rows *and* re-inserts every replayed row into the
        // indexes. None of the three in-memory indexes is idempotent (HNSW mints
        // fresh node ids for the same row, FTS increments doc_count/df rather than
        // recomputing them, BTree is a multiset), so a full scan keeps looking
        // healthy while every index-accelerated query silently returns duplicates
        // — and it compounds, because the WAL now holds those rows twice.
        //
        // `global_end()` is the writer-global batch count through this memtable,
        // since its coordinate continues where the last frozen generation ended.
        wal_flusher.advance_durable(memtable.batch_store().global_end());

        wal_flusher
            .wal_appender()
            .seed_next_position(next_wal_position)
            .await;

        // Seed the writer's covered-WAL cursor from the post-replay tip:
        // `next_wal_position` is one past the highest WAL entry we just
        // replayed into the active memtable, so everything strictly below
        // it is durably reflected in this writer's memtable. We can't
        // seed from `manifest.wal_entry_position_last_seen` — that field
        // is bumped on every successful tailer read by other readers, so
        // it may sit above what's actually covered by any
        // SSTable. Subtracting 1 from a fresh shard's `next_wal_position`
        // of `FIRST_WAL_ENTRY_POSITION` (= 1) yields 0, which correctly
        // means "no entry covered yet."
        let initial_covered_wal_entry_position = next_wal_position.saturating_sub(1);

        let memory = Arc::new(ArcSwap::<ResidentMemTables>::default());

        let state = WriterState {
            memtable,
            schema,
            last_flushed_wal_entry_position: initial_covered_wal_entry_position,
            frozen_flush_watchers: VecDeque::new(),
            frozen_memtables: VecDeque::new(),
            flush_requested: false,
            wal_flush_trigger_count: 0,
            last_wal_flush_trigger_time: 0,
        };
        // Seed before the first freeze: replay above may already have filled the
        // memtable, and nothing else publishes until it freezes.
        publish_memory(&memory, &state);
        let state = Arc::new(RwLock::new(state));

        let (memtable_flush_tx, memtable_flush_rx) = mpsc::unbounded_channel();

        let flusher = Arc::new(
            MemTableFlusher::new(
                object_store.clone(),
                base_path.clone(),
                base_uri,
                config.dir_id.clone(),
                shard_id,
                manifest_store,
            )
            .with_warmer(config.warmer.clone())
            .with_storage_context(config.store_params.clone(), config.session.clone()),
        );

        // Background WAL flush handler — parallel WAL I/O + index updates.
        let wal_handler = WalFlushHandler::new(
            wal_flusher.clone(),
            Some(state.clone()),
            None,
            config.max_wal_flush_interval,
            stats.clone(),
            config.observer.clone(),
        );
        task_executor.add_handler(
            "wal_flusher".to_string(),
            Box::new(wal_handler),
            wal_flush_rx,
        )?;

        // Background MemTable flush handler — frozen memtable to Lance file.
        // It rebuilds the same secondary indexes on each SSTable.
        let stop_flush_retries = task_executor.child_token();
        let memtable_handler = MemTableFlushHandler::new(
            state.clone(),
            memory.clone(),
            flusher,
            wal_flusher.clone(),
            epoch,
            stats.clone(),
            config.observer.clone(),
            config.frozen_memtable_grace,
            stop_flush_retries.clone(),
        );
        task_executor.add_handler(
            "memtable_flusher".to_string(),
            Box::new(memtable_handler),
            memtable_flush_rx,
        )?;

        // On its own task: only this sweep reclaims a frozen memtable's bytes,
        // and a ticker on the flush handler can be starved by its mailbox. See
        // `TaskExecutor::add_periodic`.
        //
        // Zero grace evicts on flush commit, so there is nothing to sweep.
        if !config.frozen_memtable_grace.is_zero() {
            // Sweep often enough that eviction lags the grace by at most ~1/3,
            // so a generation lives no more than ~grace * 4/3 past its commit.
            let tick = (config.frozen_memtable_grace / 3).max(Duration::from_millis(100));
            let sweep_state = state.clone();
            let sweep_memory = memory.clone();
            let grace = config.frozen_memtable_grace;
            task_executor.add_periodic("memtable_grace_sweeper".to_string(), tick, move || {
                let state = sweep_state.clone();
                let memory = sweep_memory.clone();
                async move { sweep_expired_frozen(&state, &memory, grace).await }
            })?;
        }

        // The index-apply task. Its own channel and its own dispatcher: the
        // dispatcher awaits `handle()` inline, so sharing the WAL flusher's
        // channel would queue every index apply behind a ~100ms S3 PUT.
        let (index_apply_tx, index_apply_rx) = mpsc::unbounded_channel();
        let index_handler = IndexApplyHandler {
            cursors: Arc::clone(wal_flusher.cursors()),
            wal_flusher: wal_flusher.clone(),
            stats,
        };
        task_executor.add_handler(
            "index_applier".to_string(),
            Box::new(index_handler),
            index_apply_rx,
        )?;

        // Shared state used by `put()` to dispatch trigger checks.
        let writer_state = Arc::new(SharedWriterState::new(
            memory,
            wal_flusher,
            wal_flush_tx,
            index_apply_tx,
            memtable_flush_tx,
            config.clone(),
            epoch,
            object_store,
            base_path,
            shard_id,
            stop_flush_retries,
        ));

        let backpressure = resolve_backpressure(config);

        Ok(WriterMode::MemTable {
            state,
            writer_state,
            backpressure,
        })
    }

    fn open_wal_only_mode(
        config: &ShardWriterConfig,
        logical_schema: Arc<ArrowSchema>,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        wal_flush_rx: mpsc::UnboundedReceiver<TriggerWalFlush>,
        stats: SharedWriteStats,
        task_executor: &Arc<TaskExecutor>,
    ) -> Result<WriterMode> {
        // The pending queue is shared with the flush handler so a background
        // tick can resolve to the batches still owed an append — the WAL-only
        // analog of resolving a tick against the durability cursor in MemTable
        // mode. A durable put waits on the durability cursor the handler's
        // append advances, so the ticker must run (`open` rejects durable +
        // no-interval); a non-durable writer may pass no interval and rely on
        // the size/close triggers alone.
        let state = Arc::new(WalOnlyState::default());

        // Background WAL flush handler — no MemTable state to consult, so
        // pass `None` for the frozen-vs-active detection; the pending queue and
        // the flush interval drive the background append instead.
        let wal_handler = WalFlushHandler::new(
            wal_flusher,
            None,
            Some(state.clone()),
            config.max_wal_flush_interval,
            stats,
            config.observer.clone(),
        );
        task_executor.add_handler(
            "wal_flusher".to_string(),
            Box::new(wal_handler),
            wal_flush_rx,
        )?;

        // Reuse the memtable valve (keyed off `max_unflushed_memtable_bytes`)
        // as the WAL-only budget, fed `WalOnlyState::queue_bytes()`. Keeps
        // the config knob meaningful in WAL-only mode and prevents the pending
        // queue from growing unbounded under non-durable writes.
        //
        // The *same* `state` the flush handler was given above: a second
        // `WalOnlyState` here would leave writes queuing on one and the
        // background append draining the other, forever empty.
        let backpressure = resolve_backpressure(config);

        Ok(WriterMode::WalOnly {
            state,
            wal_flush_tx,
            trigger: StdRwLock::new(WalOnlyTriggerState::default()),
            backpressure,
            logical_schema,
        })
    }

    /// Write record batches to the shard.
    ///
    /// All batches are inserted atomically with a single lock acquisition.
    /// This is more efficient than calling put() multiple times for Arrow IPC
    /// streams that contain multiple batches.
    ///
    /// # Arguments
    ///
    /// * `batches` - The record batches to write
    ///
    /// # Returns
    ///
    /// A WriteResult with batch position range and optional durability watcher.
    ///
    /// # Note
    ///
    /// Fencing is detected lazily during WAL flush via atomic writes.
    /// If another writer has taken over, the WAL flush will fail with
    /// `AlreadyExists`, indicating this writer has been fenced.
    #[instrument(name = "sw_put", level = "info", skip_all, fields(batch_count = batches.len(), shard_id = %self.config.shard_id))]
    pub async fn put(&self, batches: Vec<RecordBatch>) -> Result<WriteResult> {
        Self::validate_non_empty(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                self.put_memtable(Incoming::Rows(batches), state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger,
                backpressure,
                logical_schema,
            } => {
                validate_against_logical_schema(logical_schema, &batches)?;
                self.put_wal_only(batches, state, wal_flush_tx, trigger, backpressure)
                    .await
            }
        }
    }

    /// Delete rows by primary key.
    ///
    /// Each batch in `keys` must carry the shard's primary key column(s); other
    /// columns are ignored. lance builds a tombstone row per key — the primary
    /// key plus `_tombstone = true` and null in every other column — and
    /// appends it like an ordinary write. The tombstone is the newest value for
    /// its key: it wins newest-per-PK resolution (suppressing the older real
    /// row) and is then dropped from query results.
    ///
    /// Only supported in memtable mode. Works against non-nullable base columns:
    /// tombstones live in the storage schema, which widens them to nullable.
    ///
    /// ```
    /// # use lance::Result;
    /// # use lance::dataset::mem_wal::ShardWriter;
    /// # use arrow_array::RecordBatch;
    /// # async fn doc(writer: &ShardWriter, keys: Vec<RecordBatch>) -> Result<()> {
    /// writer.delete(keys).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "sw_delete", level = "info", skip_all, fields(batch_count = keys.len(), shard_id = %self.config.shard_id))]
    pub async fn delete(&self, keys: Vec<RecordBatch>) -> Result<WriteResult> {
        let (result, watcher) = self.delete_no_wait(keys).await?;
        // Wait for durability if configured (mirrors `put` → `put_memtable`).
        if let Some(mut watcher) = watcher {
            watcher.wait().await?;
        }
        Ok(result)
    }

    /// Like [`Self::delete`], but returns the durability watcher *without*
    /// awaiting it — the tombstone lands in the in-memory tier the instant this
    /// returns, but the index-driven LSM read only folds it once the watcher
    /// resolves the flush (which advances the visibility watermark and updates
    /// the PK index). The delete analog of [`Self::put_no_wait`], so a caller
    /// can hold an external lock across only the in-memory insert and await
    /// durability after releasing it. MemTable mode only.
    #[instrument(name = "sw_delete_no_wait", level = "info", skip_all, fields(batch_count = keys.len(), shard_id = %self.config.shard_id))]
    pub async fn delete_no_wait(
        &self,
        keys: Vec<RecordBatch>,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        if keys.is_empty() {
            return Err(Error::invalid_input("Cannot delete with empty key list"));
        }
        for (i, batch) in keys.iter().enumerate() {
            if batch.num_rows() == 0 {
                return Err(Error::invalid_input(format!("Key batch {} is empty", i)));
            }
        }

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                self.put_memtable_no_wait(
                    Incoming::Tombstones(keys),
                    state,
                    writer_state,
                    backpressure,
                )
                .await
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "delete is only supported in memtable mode (enable_memtable = true)",
            )),
        }
    }

    /// Like [`Self::put`], but returns the durability watcher *without* awaiting
    /// it. The row lands in the in-memory tier the instant this returns, but the
    /// index-driven LSM read only surfaces it once the watcher resolves the
    /// flush (which advances the visibility watermark and updates the indexes);
    /// the caller awaits durability via the watcher (`None` when `durable_write`
    /// is off).
    ///
    /// This lets a caller hold an *external* lock across only the in-memory
    /// read-merge-insert and await durability after releasing it, so concurrent
    /// flushes still coalesce. The insert stays guarded by the internal
    /// `state_lock`, so `BatchStore`'s single-writer invariant holds regardless.
    ///
    /// MemTable mode only; errors in WAL-only mode (no in-memory tier).
    #[instrument(name = "sw_put_no_wait", level = "info", skip_all, fields(batch_count = batches.len(), shard_id = %self.config.shard_id))]
    pub async fn put_no_wait(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        Self::validate_non_empty(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                self.put_memtable_no_wait(
                    Incoming::Rows(batches),
                    state,
                    writer_state,
                    backpressure,
                )
                .await
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "put_no_wait is only supported in MemTable mode",
            )),
        }
    }

    fn validate_non_empty(batches: &[RecordBatch]) -> Result<()> {
        if batches.is_empty() {
            return Err(Error::invalid_input("Cannot write empty batch list"));
        }
        for (i, batch) in batches.iter().enumerate() {
            if batch.num_rows() == 0 {
                return Err(Error::invalid_input(format!("Batch {} is empty", i)));
            }
        }
        Ok(())
    }

    async fn put_memtable(
        &self,
        incoming: Incoming,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &Arc<dyn BackpressureController>,
    ) -> Result<WriteResult> {
        let (result, watcher) = self
            .put_memtable_no_wait(incoming, state_lock, writer_state, backpressure)
            .await?;
        // Wait for durability if configured (outside the lock).
        if let Some(mut watcher) = watcher {
            watcher.wait().await?;
        }
        Ok(result)
    }

    /// In-memory half of [`Self::put_memtable`]: insert under `state_lock`,
    /// trigger the WAL flush, and return the watcher un-awaited for the caller
    /// to wait on. `None` when `durable_write` is off. See [`Self::put_no_wait`].
    async fn put_memtable_no_wait(
        &self,
        incoming: Incoming,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &Arc<dyn BackpressureController>,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        // Reject writes on a fenced writer before mutating the memtable, so a
        // poisoned writer can't drift further from the durable WAL.
        self.wal_flusher.check_poisoned()?;

        // A write lands whole in one memtable, so one larger than the cap fits
        // nowhere — a fresh memtable overflows the same way. Deletes arrive here
        // as tombstone rows and are bounded the same way.
        let incoming_batches = incoming.batch_count();
        let incoming_rows = incoming.row_count();
        if incoming_rows > self.config.max_memtable_rows {
            return Err(Error::invalid_input(format!(
                "write of {incoming_rows} rows across {} batches exceeds \
                 max_memtable_rows={}: a write is never split across memtables, and the \
                 in-memory indexes are sized to that cap. Split the write, or raise \
                 max_memtable_rows",
                incoming_batches, self.config.max_memtable_rows,
            )));
        }

        // The freeze check runs inside the lock immediately after an insert, but the
        // index apply that follows it runs *outside* — and replay hands back a
        // memtable whose indexes were built after its last check too. Either way
        // the ceiling can be reached with nothing frozen, and the valve below
        // would then refuse a write that a freeze would have admitted. Re-run the
        // check here so the wait has a flush to end on.
        //
        // The read is two relaxed loads; the lock is taken only when the shard is
        // actually at its ceiling, which is the path already about to block.
        if ShardMemory::memtables(writer_state.memory.clone()).unflushed_bytes()
            >= self.config.max_unflushed_memtable_bytes
        {
            let mut state = state_lock.write().await;
            self.wal_flusher.check_poisoned()?;
            if writer_state.maybe_trigger_memtable_flush(&mut state, 1, 1)?
                == FreezeOutcome::Blocked
            {
                return Err(freeze_blocked_error());
            }
        }

        // Apply backpressure if needed (before acquiring main lock). The reading
        // carries whether this put needs a freeze, so a controller bounding the
        // tier below can refuse only the puts that would grow it.
        let cursors = self.wal_flusher.cursors();
        tokio::select! {
            biased;
            result = backpressure.maybe_apply_backpressure(
                ShardMemory::memtables(writer_state.memory.clone()).with_freeze_required(
                    writer_state.freeze_required(incoming_batches, incoming_rows),
                ),
            ) => result?,
            error = cursors.wait_for_failure() => return Err(error),
        }

        let start = std::time::Instant::now();

        // Acquire write lock for entire operation (atomic approach)
        let (batch_positions, durable_watcher, batch_store, indexes) = {
            let mut state = state_lock.write().await;
            // Failure is latched under this same lock. Recheck after admission
            // and lock waits before preparing or inserting any more rows.
            self.wal_flusher.check_poisoned()?;

            // 0. Freeze first if this put would not fit: the row cap is a hard
            //    index capacity, so an overshoot cannot be undone afterwards.
            //    A held-back freeze refuses the put; admission above catches
            //    that first except when it read before the memtable filled.
            if writer_state.maybe_trigger_memtable_flush(
                &mut state,
                incoming_batches,
                incoming_rows,
            )? == FreezeOutcome::Blocked
            {
                return Err(freeze_blocked_error());
            }

            // 1. Shape the batches to the memtable's schema, spill Blob v2
            //    payloads into the active target, then keep only prepared
            //    descriptors in the memtable and WAL.
            let batches = incoming.shape(&state.schema)?;
            let batch_store = state.memtable.batch_store();
            let (batches, mut blob_preprocessor, shared_blob_preprocessor) = writer_state
                .prepare_batches(&state.memtable, &state.schema, batches)
                .await?;

            // 2. Insert all batches into memtable atomically
            let results = match state.memtable.insert_batches_only(batches).await {
                Ok(results) => results,
                Err(error) => {
                    if let Some(preprocessor) = blob_preprocessor.as_mut() {
                        preprocessor.abort();
                    } else if shared_blob_preprocessor {
                        // Shared preparation may include payloads from earlier
                        // unacknowledged puts, so abandoning it is a terminal
                        // group-commit failure rather than a per-put rollback.
                        batch_store.fail_blob_pack(&error).await;
                        writer_state.wal_flusher.poison(&error);
                    }
                    return Err(error);
                }
            };

            // 2. Capture the store the batches actually landed in, *before* step
            //    6 below can freeze and swap the active memtable. Reading it
            //    afterwards hands the flush trigger the **new** store paired with
            //    the **old** store's end position, so the new store's watermark
            //    jumps past batches that were never appended.
            let indexes = state.memtable.indexes_arc();

            let start_pos = results.first().map(|(pos, _, _)| *pos).unwrap_or(0);
            let end_pos = results.last().map(|(pos, _, _)| pos + 1).unwrap_or(0);
            let batch_positions = start_pos..end_pos;

            // 4. Watch for this write to become *visible*: indexed, and — under
            //    `durable_write` — WAL-durable too.
            //
            //    The two targets live in different coordinate spaces. `end_pos`
            //    is memtable-local, which is what the index apply works in. The
            //    durability cursor is writer-global, because batch positions
            //    restart at 0 in every memtable while that cursor spans the
            //    writer's whole life — a local durable target would already be
            //    satisfied by a *previous* memtable's appends, so the first N
            //    puts into every post-rotation memtable would ack as durable with
            //    no WAL append ever happening.
            let durable_watcher = writer_state.track_batch_for_wal(
                indexes.clone(),
                end_pos,
                batch_store.global_offset() + end_pos,
            );

            // 5. Check if WAL flush should be triggered
            writer_state.maybe_trigger_wal_flush(&mut state);

            // 6. Check if memtable flush is needed (may freeze and rotate).
            //    `Blocked` is fine here: the rows already landed, and the next
            //    put is the one refused.
            if let Err(e) = writer_state.maybe_trigger_memtable_flush(&mut state, 1, 1) {
                warn!("Failed to trigger memtable flush: {}", e);
            }

            (batch_positions, durable_watcher, batch_store, indexes)
        }; // Lock released here

        self.stats.record_put(start.elapsed());

        // Trigger the index apply, in **both** modes. This is what makes reads
        // read-your-writes regardless of `durable_write`: skipping durability now
        // costs the caller durability only, not visibility. It is cheap
        // (in-memory, ~ms), so there is no reason to batch it onto the WAL's
        // schedule — that schedule exists to bound S3 API cost, which an
        // in-memory index apply does not incur.
        if let Some(indexes) = indexes {
            writer_state.trigger_index_apply(batch_store, indexes, batch_positions.end)?;
        }

        // The write-path size and interval signals above name no fixed boundary:
        // the handler resolves the pending suffix when it receives them. That
        // lets concurrent puts group while a sequential durable client can
        // submit promptly once an interval has elapsed. The background ticker
        // remains the backstop when no later put arrives to emit a signal.

        // The watcher is returned in both modes now. A non-durable put still
        // waits — for its index apply (~ms), not for an S3 PUT (~100ms).
        let watcher = Some(durable_watcher);

        Ok((WriteResult { batch_positions }, watcher))
    }

    async fn put_wal_only(
        &self,
        batches: Vec<RecordBatch>,
        state: &Arc<WalOnlyState>,
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: &StdRwLock<WalOnlyTriggerState>,
        backpressure: &Arc<dyn BackpressureController>,
    ) -> Result<WriteResult> {
        // Reject writes on a fenced writer before enqueuing — see
        // `put_memtable_no_wait`.
        self.wal_flusher.check_poisoned()?;

        // Apply backpressure against the pending queue before pushing. The
        // budget reuses `max_unflushed_memtable_bytes` since WAL-only mode
        // shares the same "in-memory bytes waiting for durable storage"
        // shape as MemTable mode. WAL-only mode has no per-frozen-MemTable
        // watcher, so the backpressure loop falls back to its short sleep.
        backpressure
            .maybe_apply_backpressure(ShardMemory::queue(state.clone()))
            .await?;

        let start = std::time::Instant::now();

        // Push batches into the pending queue and capture the assigned
        // [start, end) range. `next_batch_position` is monotonic across the
        // writer's lifetime; positions are not BatchStore indices but they
        // are used the same way for durability tracking. Because the queue is
        // strict FIFO with contiguous positions and its front always sits at
        // the durability cursor, `end` is exactly the writer-global durable
        // count this put reaches once its append lands — no globalizing offset
        // as in MemTable mode, which restarts positions per generation.
        let batch_positions = state.push(batches);

        // Under `durable_write` the put becomes durable only once its WAL
        // append lands. Track it on the writer-global durability cursor *before*
        // triggering, so a flush that completes between here and the wait is not
        // missed: the watcher recomputes visibility from the cursor rather than
        // latching a one-shot wake. WAL-only mode has no indexes, so the
        // index-visibility half of the watcher is a no-op (`None`, target 0).
        let durable_watcher = self
            .config
            .durable_write
            .then(|| self.wal_flusher.track_batch(None, 0, batch_positions.end));

        // Best-effort size and interval triggers run on the write path for
        // durable and non-durable puts alike. The background ticker remains the
        // idle-write backstop.
        self.maybe_trigger_wal_flush_wal_only(wal_flush_tx, trigger, state.queue_bytes());

        self.stats.record_put(start.elapsed());

        // Durable writes wait on the durability cursor, advanced by the append
        // the ticker (or the trigger above) drives — the same watermark path
        // MemTable mode uses. A terminal flush failure (a peer fence or an
        // exhausted-retry persistence self-fence) poisons the writer and wakes
        // the waiter with that typed error instead of leaving it to hang. This
        // is why `open` rejects `durable_write` with no flush ticker: nothing
        // else would advance the cursor a small put is parked on.
        if let Some(mut watcher) = durable_watcher {
            watcher.wait().await?;
        }

        Ok(WriteResult { batch_positions })
    }

    /// WAL-only-mode best-effort size and interval triggers.
    fn maybe_trigger_wal_flush_wal_only(
        &self,
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: &StdRwLock<WalOnlyTriggerState>,
        pending_bytes: usize,
    ) {
        let mut t = trigger.write().unwrap();
        let now = now_millis();
        let time_trigger = claim_wal_time_trigger(
            &mut t.last_wal_flush_trigger_time,
            now,
            self.config.max_wal_flush_interval,
        );

        // Claim every newly crossed `max_wal_buffer_size` boundary with one
        // dynamically resolved trigger. If the pending queue shrank below the
        // recorded baseline (a drain happened), reset the baseline first so the
        // next crossing fires correctly.
        let size_trigger = claim_pending_wal_size_trigger(
            &mut t.last_trigger_pending_bytes,
            pending_bytes,
            self.config.max_wal_buffer_size,
        );
        if time_trigger || size_trigger {
            if size_trigger {
                t.last_wal_flush_trigger_time = now;
            }
            let _ = wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::NextPending,
                end_batch_position: 0,
                done: None,
            });
        }
    }

    /// Get a snapshot of current write statistics.
    pub fn stats(&self) -> WriteStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the shared stats handle (for external monitoring).
    pub fn stats_handle(&self) -> SharedWriteStats {
        self.stats.clone()
    }

    /// What this writer is holding in memory, in both write modes.
    ///
    /// The same [`ShardMemory`] the admission controller is handed, so an
    /// embedder ranking shards by size and the controller gating a write read
    /// one implementation rather than two that can disagree.
    ///
    /// Only a flush reclaims `unflushed_bytes`, so that is the pool to bound
    /// against OOM; `retained_bytes` adds what is resident but already flushed
    /// and waiting out `frozen_memtable_grace`. Either can far exceed
    /// `max_memtable_size`, which gates row data alone.
    pub fn memory(&self) -> ShardMemory {
        match &self.mode {
            WriterMode::MemTable { writer_state, .. } => {
                ShardMemory::memtables(writer_state.memory.clone())
            }
            WriterMode::WalOnly { state, .. } => ShardMemory::queue(state.clone()),
        }
    }

    /// Get the current shard manifest.
    ///
    /// Served from this writer's own last manifest commit, so a peer's commit
    /// may not appear. Fencing does not rely on this — [`Self::check_fenced`]
    /// always reads storage.
    pub async fn manifest(&self) -> Result<Option<ShardManifest>> {
        self.manifest_store.latest().await
    }

    /// The shard's manifest store.
    ///
    /// Embedders must commit through this instance: a second
    /// `ShardManifestStore` over the same shard keeps its own cache, and
    /// neither would see the other's commits.
    pub fn manifest_store(&self) -> Arc<ShardManifestStore> {
        self.manifest_store.clone()
    }

    /// Get the writer's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The configuration this writer was opened with.
    pub fn config(&self) -> &ShardWriterConfig {
        &self.config
    }

    /// The indexes memtables are currently built with, sorted.
    ///
    /// This is the set in force now, which a replacement can move; the one an
    /// already-frozen memtable was built with is not reported here. Empty in
    /// WAL-only mode, which holds no memtable to index.
    pub async fn maintained_index_names(&self) -> Vec<String> {
        match &self.mode {
            WriterMode::MemTable { state, .. } => {
                let mut names: Vec<String> = state
                    .read()
                    .await
                    .schema
                    .index_specs
                    .iter()
                    .map(|spec| spec.name.clone())
                    .collect();
                names.sort_unstable();
                names
            }
            WriterMode::WalOnly { .. } => Vec::new(),
        }
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.config.shard_id
    }

    /// Return `Err` if a successor writer has claimed a higher epoch.
    pub async fn check_fenced(&self) -> Result<()> {
        self.manifest_store.check_fenced(self.epoch).await
    }

    /// Get current MemTable statistics. Returns an error in WAL-only mode
    /// (no MemTable exists).
    ///
    /// Deliberately does *not* `check_poisoned`, unlike the read and write
    /// paths: a poisoned writer is exactly when an operator most needs to see
    /// its state, and the caller deciding whether to evict reads these stats.
    pub async fn memtable_stats(&self) -> Result<MemTableStats> {
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        let batch_store = state.memtable.batch_store();
        let durable = self.wal_flusher.durable();
        let pending_wal = batch_store.pending_wal_flush_stats(durable);
        Ok(MemTableStats {
            row_count: state.memtable.row_count(),
            batch_count: state.memtable.batch_count(),
            generation: state.memtable.generation(),
            max_buffered_batch_position: batch_store.max_buffered_batch_position(),
            durable_batch_count: durable,
            global_offset: batch_store.global_offset(),
            pending_wal_start_batch_position: pending_wal.start_batch_position,
            pending_wal_end_batch_position: pending_wal.end_batch_position,
            pending_wal_batch_count: pending_wal.batch_count,
            pending_wal_row_count: pending_wal.row_count,
            pending_wal_estimated_bytes: pending_wal.estimated_bytes,
            frozen_count: state.frozen_memtables.len(),
        })
    }

    /// Snapshot of the backpressure counters. Both writer modes answer, so a
    /// caller asking "am I throttled" need not know which mode it is in.
    pub fn backpressure_stats(&self) -> BackpressureStatsSnapshot {
        match &self.mode {
            WriterMode::MemTable { backpressure, .. }
            | WriterMode::WalOnly { backpressure, .. } => backpressure.stats_snapshot(),
        }
    }

    /// Create a scanner for querying the current MemTable data.
    ///
    /// The scanner provides read access to all data currently in the MemTable,
    /// with optional filtering, projection, and index support.
    ///
    /// The scanner captures the current `visible_count` from the
    /// `IndexStore` at construction time to ensure consistent visibility.
    ///
    /// Returns an error in WAL-only mode, or if the writer is poisoned.
    pub async fn scan(&self) -> Result<MemTableScanner> {
        self.wal_flusher.check_poisoned()?;
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(state.memtable.scan())
    }

    /// A handle to just the active memtable, for unified LSM scanning.
    /// Prefer [`Self::in_memory_memtable_refs`] on the read path — it also
    /// carries frozen-awaiting-flush generations.
    ///
    /// Returns an error in WAL-only mode, or if the writer is poisoned.
    pub async fn active_memtable_ref(
        &self,
    ) -> Result<crate::dataset::mem_wal::scanner::InMemoryMemTableRef> {
        self.wal_flusher.check_poisoned()?;
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(in_memory_ref(&state.memtable))
    }

    /// The active memtable plus every frozen-awaiting-flush memtable,
    /// captured atomically under one `state.read()` (no torn freeze).
    /// Mirrors `WriterState { memtable, frozen_memtables }`. The WAL read
    /// path uses this instead of [`Self::active_memtable_ref`] so a
    /// concurrent reader sees no hole while a flush drains.
    ///
    /// Returns an error in WAL-only mode, or if the writer is poisoned.
    pub async fn in_memory_memtable_refs(
        &self,
    ) -> Result<crate::dataset::mem_wal::scanner::InMemoryMemTables> {
        self.wal_flusher.check_poisoned()?;
        let state_lock = self.memtable_state_lock()?;
        let state = state_lock.read().await;
        Ok(crate::dataset::mem_wal::scanner::InMemoryMemTables {
            active: in_memory_ref(&state.memtable),
            frozen: state
                .frozen_memtables
                .iter()
                .map(|m| in_memory_ref(&m.memtable))
                .collect(),
        })
    }

    /// Returns the MemTable-mode state lock or a clear invalid_input error
    /// when running in WAL-only mode.
    fn memtable_state_lock(&self) -> Result<&Arc<RwLock<WriterState>>> {
        match &self.mode {
            WriterMode::MemTable { state, .. } => Ok(state),
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "MemTable accessor not available when enable_memtable = false (WAL-only mode)",
            )),
        }
    }

    /// Get WAL statistics.
    pub fn wal_stats(&self) -> WalStats {
        WalStats {
            next_wal_entry_position: self.wal_flusher.next_wal_entry_position(),
        }
    }

    /// Freeze the active memtable and start the next under the schema `next` returns.
    ///
    /// `next` runs under the `state` write lock on the current schema, so a
    /// caller changing only the schema or only the indexes keeps the other.
    /// `Ok(None)` when nothing would change.
    async fn replace_writer_schema(
        &self,
        op: &'static str,
        next: impl FnOnce(&WriterSchema) -> Result<WriterSchema>,
    ) -> Result<Option<FreezeFence>> {
        let WriterMode::MemTable {
            state,
            writer_state,
            ..
        } = &self.mode
        else {
            return Err(Error::invalid_input(format!(
                "{op} is not available in WAL-only mode (no MemTable)"
            )));
        };
        self.check_fenced().await?;
        self.wal_flusher.check_poisoned()?;

        let mut state = state.write().await;
        self.wal_flusher.check_poisoned()?;
        let current = Arc::clone(&state.schema);
        let next = Arc::new(next(&current)?);
        if next.is_equivalent(&current) {
            return Ok(None);
        }
        if next.pk_field_ids != current.pk_field_ids {
            return Err(Error::invalid_input(format!(
                "{op} cannot change the primary key: the writer holds primary key field ids \
                 {:?} ({:?}) and the new schema declares {:?} ({:?})",
                current.pk_field_ids, current.pk_columns, next.pk_field_ids, next.pk_columns,
            )));
        }

        let frozen_generation = if state.memtable.batch_count() == 0 {
            // Nothing written yet: replace it in place under the same generation.
            let generation = state.memtable.generation();
            let global_offset = state.memtable.batch_store().global_end();
            state.memtable = next.new_memtable(
                generation,
                global_offset,
                None,
                writer_state.epoch,
                writer_state.max_memtable_rows,
                writer_state.max_memtable_batches,
                self.wal_flusher.cursors(),
            )?;
            state.schema = next;
            publish_memory(&writer_state.memory, &state);
            None
        } else {
            let frozen = state.memtable.generation();
            writer_state.rotate_memtable(&mut state, next)?;
            Some(frozen)
        };
        info!(
            "{op} moved shard {} (epoch {}) onto a new schema: active generation {}, frozen {:?}",
            self.config.shard_id,
            self.epoch,
            state.memtable.generation(),
            frozen_generation,
        );
        Ok(Some(FreezeFence {
            frozen_generation,
            watchers: state.frozen_flush_watchers.iter().cloned().collect(),
            cursors: self.wal_flusher.cursors().clone(),
        }))
    }

    /// Switch this writer to `schema` and `index_specs` without reopening it.
    ///
    /// The active memtable is frozen and flushed under its old schema. Reads
    /// match columns by field id, so older rows show up under the new names.
    ///
    /// `schema` should carry field ids under `lance:field_id`, as for
    /// [`Self::open`]. The primary key cannot change. Repeating the same change
    /// is a no-op, so concurrent callers freeze only once.
    ///
    /// MemTable mode only. `Ok(None)` when nothing would change; otherwise a
    /// [`FreezeFence`] covering whatever was frozen.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::Schema;
    /// # use lance::Result;
    /// # use lance::dataset::mem_wal::ShardWriter;
    /// # async fn doc(writer: &ShardWriter, renamed: Arc<Schema>) -> Result<()> {
    /// writer.evolve_schema(renamed, Vec::new()).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "sw_evolve_schema", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn evolve_schema(
        &self,
        schema: Arc<ArrowSchema>,
        index_specs: Vec<MemIndexSpec>,
    ) -> Result<Option<FreezeFence>> {
        self.replace_writer_schema("evolve_schema", |_current| {
            WriterSchema::try_new(&schema, index_specs)
        })
        .await
    }

    /// Freeze the active memtable so it's queued for SSTable flush. Errors in
    /// WAL-only mode or if this writer has been fenced by a successor.
    ///
    /// The returned [`FreezeFence`] is what makes a *bounded* wait possible:
    /// a caller that needs "everything written before my freeze is an SSTable"
    /// awaits [`FreezeFence::wait`], which covers the flushes outstanding at
    /// freeze time and nothing else. [`Self::wait_for_flush_drain`] instead
    /// loops until the frozen set is *empty*, re-collecting it each round,
    /// so it also waits on every memtable frozen *while it waits*. Under
    /// sustained writes that set may never empty.
    ///
    /// Beyond test setup where deterministic flush points are required,
    /// this is the primary lever for callers that need to drive flushes
    /// out-of-band from the size/interval triggers — for example, to cap
    /// resident memtable bytes across many shards in a multi-table WAL
    /// writer process, or to drain the WAL ahead of a format change so
    /// the next epoch starts with no replayable entries from the old
    /// layout.
    #[instrument(name = "sw_force_freeze_active", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn force_freeze_active(&self) -> Result<FreezeFence> {
        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                self.check_fenced().await?;
                self.wal_flusher.check_poisoned()?;
                let mut state = state.write().await;
                self.wal_flusher.check_poisoned()?;
                let frozen_generation = if state.memtable.batch_count() == 0 {
                    None
                } else {
                    let generation = state.memtable.generation();
                    writer_state.freeze_memtable(&mut state)?;
                    Some(generation)
                };
                // Capture the outstanding set under the same lock that froze,
                // so no freeze can slip between the two and widen the fence.
                // It covers more than the generation just frozen: a
                // size/interval trigger freezes generations asynchronously, so
                // an empty active memtable does *not* mean every pre-call write
                // already became an SSTable. Watchers are popped as flushes settle, so
                // whatever remains here is exactly what is still owed.
                Ok(FreezeFence {
                    frozen_generation,
                    watchers: state.frozen_flush_watchers.iter().cloned().collect(),
                    cursors: self.wal_flusher.cursors().clone(),
                })
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "force_freeze_active not available in WAL-only mode (no MemTable)",
            )),
        }
    }

    /// Replace the indexes memtables are built with, and freeze so the next one
    /// carries them.
    ///
    /// The swap and the freeze happen under one hold of the `state` write lock,
    /// the same lock a write takes, so a memtable is never built from a
    /// half-applied set and no write lands in a memtable whose indexes have
    /// already been superseded.
    ///
    /// The outgoing memtable keeps the indexes it was built with, and its SSTable
    /// is written without the new vector or full-text ones, so its
    /// rows answer neither search until compaction.
    ///
    /// `Ok(None)` when the set is already current; nothing is frozen.
    #[instrument(name = "sw_replace_index_configs", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch, index_count = specs.len()))]
    pub async fn replace_index_configs(
        &self,
        specs: Vec<MemIndexSpec>,
    ) -> Result<Option<FreezeFence>> {
        // Usually already current: check under the read lock and skip the
        // fence check, which reads the manifest.
        if let WriterMode::MemTable { state, .. } = &self.mode
            && same_index_set(&state.read().await.schema.index_specs, &specs)
        {
            return Ok(None);
        }
        self.replace_writer_schema("replace_index_configs", |current| {
            current.with_index_specs(specs)
        })
        .await
    }

    /// Block until every frozen memtable in the SSTable flush queue has
    /// landed and been recorded in the manifest. Does not wait on the
    /// active memtable — call [`Self::force_freeze_active`] first if you
    /// want everything-on-disk. Errors in WAL-only mode, or if any
    /// awaited flush reports `DurabilityResult::Failed`.
    ///
    /// Useful in tests for deterministic post-flush assertions. In
    /// production prefer [`FreezeFence::wait`] from the freeze itself: this
    /// drains the queue to empty, including memtables frozen after the
    /// call, so under sustained writes it may not return.
    #[instrument(name = "sw_wait_for_flush_drain", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn wait_for_flush_drain(&self) -> Result<()> {
        let state_lock = match &self.mode {
            WriterMode::MemTable { state, .. } => state,
            WriterMode::WalOnly { .. } => {
                return Err(Error::invalid_input(
                    "wait_for_flush_drain not available in WAL-only mode (no MemTable)",
                ));
            }
        };

        loop {
            let watchers: Vec<DurabilityWatcher> = {
                let st = state_lock.read().await;
                self.wal_flusher.check_poisoned()?;
                st.frozen_flush_watchers.iter().cloned().collect()
            };
            if watchers.is_empty() {
                return Ok(());
            }
            for mut w in watchers {
                let durability = tokio::select! {
                    biased;
                    durability = w.await_value() => durability,
                    error = self.wal_flusher.cursors().wait_for_failure() => return Err(error),
                };
                self.wal_flusher.check_poisoned()?;
                match durability {
                    Some(durability) => durability.into_result()?,
                    None => {
                        return Err(Error::io(
                            "MemTable flush handler exited before reporting completion",
                        ));
                    }
                }
            }
        }
    }

    /// Abort the writer without flushing.
    ///
    /// Shuts down the background flush tasks and leaves all buffered
    /// memtable state to be dropped with the writer. Unlike
    /// [`Self::close`], no WAL/MemTable flush is issued: pending in-memory
    /// rows are discarded, not made durable, and no object-store IO is
    /// performed. Used on drop-table, where the dataset directory is about
    /// to be removed and a flush would only race fresh files back into a
    /// doomed path.
    ///
    /// Caller-quiesce contract: `abort` takes `&self` (so it can be called
    /// through the `Arc<ShardWriter>` callers hold) and therefore cannot
    /// structurally bar a concurrent or subsequent `put` the way consuming
    /// `close(self)` does. After abort the dispatchers are gone, so a later
    /// `put` would buffer data that never flushes. Callers MUST stop
    /// issuing writes before calling abort.
    ///
    /// Blocks until any flush already mid-`handle()` settles —
    /// cancellation only fires between messages — so no flush task lingers
    /// after abort returns. Cursor waiters are failed before cancellation so
    /// they cannot be stranded when another dispatcher discards work they
    /// depend on. Idempotent: a second call re-cancels an already-cancelled
    /// token and joins an already-emptied task set.
    #[instrument(name = "sw_abort", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn abort(&self) -> Result<()> {
        info!(
            "Aborting ShardWriter for shard {} (no flush)",
            self.config.shard_id
        );
        // A memtable flush may be waiting for an index apply that is still
        // queued on another dispatcher. Cancellation is biased ahead of queued
        // messages, so wake the dependent cursor waiter before cancellation can
        // discard the apply and leave shutdown joining the waiter forever.
        self.wal_flusher.poison(&Error::writer_poisoned(
            "ShardWriter aborted while background work was pending",
        ));
        self.task_executor.shutdown_all().await?;
        Ok(())
    }

    /// Send the close-time final WAL flush and await its completion.
    ///
    /// Sends directly on the flush channel rather than via
    /// [`WalFlusher::trigger_flush`]: the latter silently returns `Ok` when the
    /// flusher's `flush_tx` is unset, which would let close report success
    /// without ever persisting the final WAL entry. A closed send channel must
    /// surface as an error here so close never acknowledges durability it did
    /// not achieve.
    async fn flush_final_wal(
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        source: WalFlushSource,
        end_batch_position: usize,
    ) -> Result<()> {
        let done = WatchableOnceCell::new();
        let mut reader = done.reader();
        if wal_flush_tx
            .send(TriggerWalFlush {
                source,
                end_batch_position,
                done: Some(done),
            })
            .is_err()
        {
            return Err(Error::io("WAL flush channel closed during close"));
        }

        match reader.await_value().await {
            Some(Ok(_)) => Ok(()),
            Some(Err(failure)) => Err(failure.into_error()),
            None => Err(Error::io(
                "WAL flush handler exited before reporting durability during close",
            )),
        }
    }

    fn merge_close_stage(
        close_result: Result<()>,
        stage: &str,
        stage_result: Result<()>,
    ) -> Result<()> {
        if let (Err(_), Err(stage_error)) = (&close_result, &stage_result) {
            warn!("Close stage '{stage}' also failed: {stage_error}");
        }
        close_result.and(stage_result)
    }

    async fn drain_for_close(&self) -> Result<()> {
        self.wal_flusher.check_poisoned()?;
        let mut close_result: Result<()> = Ok(());

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                // Else a failing flush retries forever, and an early close
                // error leaves it running; replay recovers its generation.
                writer_state.stop_flush_retries.cancel();

                // Drain *both* tasks against the active memtable. The index apply
                // and the WAL append are independent now, so closing has to
                // settle both: the SSTable flush below turns this memtable into a
                // Lance generation, and a generation whose indexes never saw the
                // tail is a generation with a hole in it.
                let st = state.read().await;
                let batch_store = st.memtable.batch_store();
                let indexes = st.memtable.indexes_arc();
                let batch_count = st.memtable.batch_count();
                drop(st);

                let index_drain = async {
                    if batch_count > 0
                        && let Some(indexes) = indexes
                        && indexes.indexed_count() < batch_count
                    {
                        let mut watcher = self.wal_flusher.track_batch(
                            Some(Arc::clone(&indexes)),
                            batch_count,
                            0, // durability is settled by the WAL flush below
                        );
                        writer_state.trigger_index_apply(
                            Arc::clone(&batch_store),
                            indexes,
                            batch_count,
                        )?;
                        watcher.wait().await?;
                    }
                    Ok(())
                }
                .await;
                close_result = Self::merge_close_stage(
                    close_result,
                    "active MemTable index drain",
                    index_drain,
                );

                // The flush builds the SSTable's indexes from the memtable's, so
                // skip it when the drain failed; replay recovers the rows.
                if close_result.is_ok() {
                    if batch_count > 0 {
                        // Append-only source: on this branch the index apply is a
                        // separate task (drained above), so the final WAL flush carries
                        // no indexes. #7769's failure propagation still applies.
                        let stage_result = Self::flush_final_wal(
                            &writer_state.wal_flush_tx,
                            WalFlushSource::BatchStore { batch_store },
                            batch_count,
                        )
                        .await;
                        close_result =
                            Self::merge_close_stage(close_result, "final WAL flush", stage_result);
                    }

                    // Freeze the active memtable (if any rows) so it joins the
                    // pending-flush queue, then wait for every frozen
                    // memtable's flush to complete. Without this, close() left
                    // any rows that hadn't crossed the size/batch trigger
                    // sitting in memory at shutdown — they were durable in the
                    // WAL but never produced a Lance fragment, which is
                    // surprising for callers who reasonably expect close() to
                    // make all data fully durable in the LSM sense (and which
                    // makes flush-cost benchmarks impossible to measure).
                    //
                    // Propagate any freeze error: at close time the caller
                    // has explicitly asked for full durability, so silently
                    // dropping a freeze failure would lose data without any
                    // signal. If freeze fails, its error is recorded as the
                    // first causal failure, but close still drains any
                    // pre-existing frozen MemTable watchers so a successor
                    // failure is logged without replacing the first error.
                    let watchers: Vec<_> = {
                        let mut st = state.write().await;
                        if st.memtable.row_count() > 0 {
                            let freeze_result = writer_state.freeze_memtable(&mut st).map(|_| ());
                            close_result = Self::merge_close_stage(
                                close_result,
                                "active MemTable freeze",
                                freeze_result,
                            );
                        }
                        st.frozen_flush_watchers.iter().cloned().collect()
                    };
                    for mut watcher in watchers {
                        let stage_result = match watcher.await_value().await {
                            Some(durability) => durability.into_result(),
                            None => Err(Error::io(
                                "MemTable flush handler exited before reporting completion during close",
                            )),
                        };
                        close_result = Self::merge_close_stage(
                            close_result,
                            "frozen MemTable flush watcher",
                            stage_result,
                        );
                    }
                }
            }
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger: _,
                backpressure: _,
                logical_schema: _,
            } => {
                // Drain any pending batches via a final flush; wait for completion.
                let pending = state.batch_count();
                let end_position = state.next_batch_position();
                if pending > 0 {
                    let stage_result = Self::flush_final_wal(
                        wal_flush_tx,
                        WalFlushSource::WalOnly {
                            state: state.clone(),
                        },
                        end_position,
                    )
                    .await;
                    close_result =
                        Self::merge_close_stage(close_result, "final WAL flush", stage_result);
                }
            }
        }

        close_result
    }

    /// Close the writer gracefully.
    ///
    /// Flushes pending data and shuts down background tasks, even if draining fails.
    ///
    /// # Errors
    ///
    /// Returns an error if pending WAL data cannot be persisted, an active or
    /// frozen MemTable cannot be flushed, a flush handler exits before reporting
    /// completion, or background tasks cannot be shut down.
    #[instrument(name = "sw_close", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn close(self) -> Result<()> {
        info!("Closing ShardWriter for shard {}", self.config.shard_id);
        let close_result = self.drain_for_close().await;
        // A drain error must not leave dispatchers retaining the old writer's
        // memtables while the caller reopens and replays the WAL.
        let shutdown_result = self.task_executor.shutdown_all().await;
        let close_result = Self::merge_close_stage(close_result, "task shutdown", shutdown_result);
        // The failed flush may already have removed its watcher, or may have
        // failed while close was draining. Its terminal state outlives both.
        let close_result = Self::merge_close_stage(
            self.wal_flusher.check_poisoned(),
            "close drain",
            close_result,
        );

        match &close_result {
            Ok(()) => info!("ShardWriter closed for shard {}", self.config.shard_id),
            Err(error) => warn!(
                "ShardWriter close for shard {} failed: {error}",
                self.config.shard_id
            ),
        }
        close_result
    }
}

/// The SSTable flushes outstanding when [`ShardWriter::force_freeze_active`]
/// returned — the exact predicate for "everything written before that call
/// is an SSTable".
///
/// The set is fixed at freeze time, so [`Self::wait`] is bounded no matter how
/// many memtables freeze while it waits, and each entry reports the outcome of
/// its own flush. The captured set remains valid independently of manifest
/// changes, including compaction and a successor writer's commits.
#[derive(Debug)]
pub struct FreezeFence {
    frozen_generation: Option<u64>,
    watchers: Vec<DurabilityWatcher>,
    /// Keeps a typed terminal error observable after flush watchers retire.
    cursors: Arc<WriterCursors>,
}

impl FreezeFence {
    /// The generation this call froze, or `None` when the active memtable
    /// was empty (a no-op freeze). Independent of what the fence covers: an
    /// empty active memtable says nothing about generations frozen earlier
    /// and still awaiting flush, which the fence still waits on.
    pub fn frozen_generation(&self) -> Option<u64> {
        self.frozen_generation
    }

    /// Block until every flush this fence covers has landed as an SSTable.
    ///
    /// Errors if the writer is poisoned, any flush reports
    /// `DurabilityResult::Failed`, or the flush handler exited without reporting.
    pub async fn wait(self) -> Result<()> {
        self.cursors.check_poisoned()?;
        for mut watcher in self.watchers {
            let durability = tokio::select! {
                biased;
                durability = watcher.await_value() => durability,
                error = self.cursors.wait_for_failure() => return Err(error),
            };
            self.cursors.check_poisoned()?;
            match durability {
                Some(durability) => durability.into_result()?,
                None => {
                    return Err(Error::io(
                        "MemTable flush handler exited before reporting completion",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
