// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Background handlers for WAL, index, and memtable flush work.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use lance_core::{Error, Result};
use log::{info, warn};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::instrument;

use super::backpressure::DurabilityResult;
use super::stats::SharedWriteStats;
use super::tasks::{MessageFactory, MessageHandler};
use super::writer::{ResidentMemTables, WriterState, now_millis, publish_memory};
use crate::dataset::mem_wal::index::MemIndexSpec;
use crate::dataset::mem_wal::memtable::MemTable;
use crate::dataset::mem_wal::memtable::batch_store::BatchStore;
use crate::dataset::mem_wal::memtable::flush::{MemTableFlusher, TriggerMemTableFlush};
use crate::dataset::mem_wal::observer::WalObserver;
use crate::dataset::mem_wal::wal::{
    TriggerIndexApply, TriggerWalFlush, WalFlushFailure, WalFlushResult, WalFlushSource,
    WalFlusher, WalOnlyState, WriterCursors, apply_index_range, empty_flush_result,
};

/// The oldest store that still owes the WAL an append, or `None` when everything
/// is durable.
///
/// Ordering is the whole point. WAL entry positions are assigned in append-call
/// order; replay walks them ascending; row positions follow; and primary-key
/// recency is "newest visible row position wins". So appending a newer memtable
/// ahead of an older one's tail silently inverts dedup after a crash.
///
/// Taking "the active memtable" would do exactly that, because a timer tick
/// enqueued before a freeze is handled after it and would resolve to the incoming
/// memtable. Selecting by cursor instead makes the target a function of what is
/// actually durable, not of when the timer happened to fire.
fn next_pending_store(
    frozen: impl Iterator<Item = Arc<BatchStore>>,
    active: Arc<BatchStore>,
    durable: usize,
) -> Option<Arc<BatchStore>> {
    frozen
        .chain(std::iter::once(active))
        .find(|store| store.global_end() > durable)
}

/// The index-apply task: one sequential consumer of the index-apply channel.
///
/// Sequential consumption is the safety property. `HnswGraph::insert_batch` hard-
/// rejects any range whose start is not its `indexed_len`, so the apply must see
/// contiguous, in-order ranges — and a single consumer guarantees that
/// regardless of how many putters race behind it. Ordering comes from the task,
/// not from a flush interval, which is why triggering per-put is exactly as safe
/// as triggering on a timer, and lets a put become visible in milliseconds
/// instead of waiting on an S3 round-trip.
pub(super) struct IndexApplyHandler {
    pub(super) cursors: Arc<WriterCursors>,
    pub(super) wal_flusher: Arc<WalFlusher>,
    pub(super) stats: SharedWriteStats,
}

#[async_trait]
impl MessageHandler<TriggerIndexApply> for IndexApplyHandler {
    async fn handle(&mut self, message: TriggerIndexApply) -> Result<()> {
        match apply_index_range(&self.cursors, message).await {
            Ok(applied) => {
                // A coalesced no-op indexes nothing (`rows_indexed == 0`);
                // recording it would inflate the count and skew avg latency.
                if applied.rows_indexed > 0 {
                    self.stats
                        .record_index_update(applied.duration, applied.rows_indexed);
                }
                Ok(())
            }
            // An index apply cannot be partially rolled back, so a failure is
            // terminal: poison, and let reopen rebuild the indexes from the WAL.
            // See the note in `WalFlusher::flush_from_batch_store`.
            Err(e) => {
                self.wal_flusher.poison(&e);
                Err(e)
            }
        }
    }
}

pub(super) struct WalFlushHandler {
    wal_flusher: Arc<WalFlusher>,
    /// MemTable-mode writer state, used to detect "frozen vs active" flushes
    /// via Arc::ptr_eq on the active batch_store. `None` when running in
    /// WAL-only mode (no MemTable, no frozen-vs-active distinction).
    memtable_state: Option<Arc<RwLock<WriterState>>>,
    /// WAL-only-mode pending queue, so a background tick can resolve to the
    /// batches still owed an append. `None` in MemTable mode. Exactly one of
    /// `memtable_state` / `wal_only_state` is `Some`.
    wal_only_state: Option<Arc<WalOnlyState>>,
    /// How often to append in the background. `None` disables the ticker, leaving
    /// the append size-triggered (and freeze/close-triggered) only.
    flush_interval: Option<Duration>,
    stats: SharedWriteStats,
    observer: Option<Arc<dyn WalObserver>>,
}

impl WalFlushHandler {
    pub(super) fn new(
        wal_flusher: Arc<WalFlusher>,
        memtable_state: Option<Arc<RwLock<WriterState>>>,
        wal_only_state: Option<Arc<WalOnlyState>>,
        flush_interval: Option<Duration>,
        stats: SharedWriteStats,
        observer: Option<Arc<dyn WalObserver>>,
    ) -> Self {
        Self {
            wal_flusher,
            memtable_state,
            wal_only_state,
            flush_interval,
            stats,
            observer,
        }
    }
}

#[async_trait]
impl MessageHandler<TriggerWalFlush> for WalFlushHandler {
    /// Append periodically in the background.
    ///
    /// This is what the flush interval was always supposed to mean. It routed to
    /// a timer that was only ever *evaluated on the write path*, so it could add
    /// a redundant trigger but never delay or batch one — with every durable put
    /// triggering its own append, tuning the knob did nothing at all.
    ///
    /// The ticker exists to bound S3 API cost: an append is a PUT, billed per
    /// call, and it is the only thing on this schedule. The index apply is not —
    /// it is in-memory and free to batch, so it runs per-put on its own task.
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<TriggerWalFlush>)> {
        // No interval => no ticker. A zero interval would panic in tokio.
        let Some(interval) = self.flush_interval.filter(|d| !d.is_zero()) else {
            return vec![];
        };
        // The tick names no store: `MessageFactory` is synchronous and cannot take
        // the async state lock. `handle()` resolves it against the cursor.
        vec![(
            interval,
            Box::new(|| TriggerWalFlush {
                source: WalFlushSource::NextPending,
                end_batch_position: 0,
                done: None,
            }),
        )]
    }

    async fn handle(&mut self, message: TriggerWalFlush) -> Result<()> {
        let TriggerWalFlush {
            source,
            end_batch_position,
            done,
        } = message;

        // A timer tick names no store — resolve it now, at handle time.
        let (source, end_batch_position) = match source {
            WalFlushSource::NextPending => match self.resolve_next_pending().await {
                Some(resolved) => resolved,
                // Everything is already durable; the tick has nothing to do.
                None => return Ok(()),
            },
            other => (other, end_batch_position),
        };

        let result = self.do_flush(source, end_batch_position).await;

        // Propagate the just-appended WAL entry position back into the
        // writer state so a subsequent MemTable freeze can stamp the
        // correct `covered_wal_entry_position` into the manifest. Without
        // this, `replay_after_wal_entry_position` stays at 0 and replay
        // re-reads already-flushed entries after restart.
        //
        // Always update state before signalling the completion cell so any
        // waiter that reads state immediately after the cell fires sees
        // the new position.
        if let (Ok(flush_result), Some(state_lock)) = (&result, &self.memtable_state)
            && let Some(entry) = &flush_result.entry
        {
            let mut state = state_lock.write().await;
            state.last_flushed_wal_entry_position =
                state.last_flushed_wal_entry_position.max(entry.position);
        }

        // Notify completion if requested. Carry the typed fence reason through
        // the cell (not just a string) so a waiter rebuilds the right error.
        if let Some(cell) = done {
            cell.write(result.map_err(|e| WalFlushFailure::from_error(&e)));
        }

        Ok(())
    }
}

impl WalFlushHandler {
    /// Pick the store the WAL still owes an append, oldest first.
    ///
    /// **WAL entries must be appended in global batch-position order for the
    /// writer's lifetime.** `WalAppender::append` assigns each entry's position
    /// from its own counter, in call order; replay walks those positions
    /// ascending and assigns row positions in that order; and primary-key recency
    /// is "newest visible row position wins". So append order fixes dedup order.
    /// Append two memtables out of order and replay silently hands the dedup to
    /// the *stale* row — corruption that survives the crash that caused it, and
    /// that a full scan cannot see.
    ///
    /// Resolving to "the active memtable" would break exactly that: a tick
    /// enqueued before a freeze is handled after it, resolves to the incoming
    /// memtable, and appends its batches ahead of the outgoing memtable's tail.
    /// So the target is a function of `durable`, not of when the timer fired.
    ///
    /// Safe to walk the frozen list because a store that still owes an append
    /// cannot be swept: its SSTable flush is blocked on the completion cell that only
    /// that append fires.
    ///
    /// In WAL-only mode there is a single FIFO pending queue and no memtable
    /// rotation, so the ordering hazard above cannot arise: the tick resolves to
    /// the queue whenever it holds un-appended batches.
    async fn resolve_next_pending(&self) -> Option<(WalFlushSource, usize)> {
        if let Some(state_lock) = self.memtable_state.as_ref() {
            let state = state_lock.read().await;
            let durable = self.wal_flusher.durable();

            return next_pending_store(
                state
                    .frozen_memtables
                    .iter()
                    .map(|frozen| frozen.memtable.batch_store()),
                state.memtable.batch_store(),
                durable,
            )
            .map(|store| {
                let end = store.len();
                (WalFlushSource::BatchStore { batch_store: store }, end)
            });
        }

        let state = self.wal_only_state.as_ref()?;
        if state.batch_count() == 0 {
            // Everything already appended; the tick has nothing to do.
            return None;
        }
        // `flush_from_wal_only` snapshots the whole queue, so the end position
        // is informational here; carry the next position for symmetry.
        let end = state.next_batch_position();
        Some((
            WalFlushSource::WalOnly {
                state: Arc::clone(state),
            },
            end,
        ))
    }

    /// Unified flush method for both active and frozen memtables and for
    /// WAL-only mode.
    ///
    /// For BatchStore sources, detects frozen vs active flush by comparing
    /// the passed batch_store with the current active memtable's
    /// batch_store. If different, it's a frozen memtable flush.
    #[instrument(
        name = "wal_do_flush",
        level = "debug",
        skip_all,
        fields(end_batch_position)
    )]
    async fn do_flush(
        &self,
        source: WalFlushSource,
        end_batch_position: usize,
    ) -> Result<WalFlushResult> {
        let start = Instant::now();

        // Early-out for BatchStore sources where the watermark already
        // covers the requested end position. Detection of "frozen flush"
        // requires the active memtable's batch_store; WAL-only handlers
        // don't have one (`memtable_state` is `None`) and never receive a
        // BatchStore source, so the early-out simplifies to the watermark
        // comparison.
        if let WalFlushSource::BatchStore { batch_store, .. } = &source {
            let flushed_up_to = batch_store.local_end(self.wal_flusher.durable());
            let is_frozen_flush = if let Some(state_lock) = &self.memtable_state {
                let state = state_lock.read().await;
                !Arc::ptr_eq(batch_store, &state.memtable.batch_store())
            } else {
                false
            };
            if !is_frozen_flush && flushed_up_to >= end_batch_position {
                return Ok(empty_flush_result());
            }
        }

        // Delegate the actual flush to the WalFlusher.
        let flush_result = self.wal_flusher.flush(&source, end_batch_position).await?;

        let batches_flushed = flush_result
            .entry
            .as_ref()
            .map(|e| e.num_batches)
            .unwrap_or(0);

        if batches_flushed > 0 {
            // One reading for both sinks, so the cumulative total and the
            // per-flush observation cannot disagree.
            let elapsed = start.elapsed();
            self.stats.record_wal_flush(elapsed, flush_result.wal_bytes);
            self.stats.record_wal_io(flush_result.wal_io_duration);
            if let Some(observer) = &self.observer {
                observer.on_wal_flush(elapsed, flush_result.wal_bytes);
            }
        }

        Ok(flush_result)
    }
}

/// First delay before a failed memtable flush is retried; it doubles per attempt.
const FLUSH_RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

/// Longest delay between attempts at a failing memtable flush.
const FLUSH_RETRY_MAX_DELAY: Duration = Duration::from_secs(10 * 60);

/// Background handler for MemTable flush operations.
///
/// This handler receives frozen memtables directly via messages and flushes them to Lance storage.
/// Freezing is done by the writer (via SharedWriterState::freeze_memtable) to ensure
/// immediate memtable switching, so writes can continue on the new memtable while this
/// handler flushes in the background.
pub(super) struct MemTableFlushHandler {
    state: Arc<RwLock<WriterState>>,
    /// Shared with `SharedWriterState`; this handler re-derives it once a
    /// flush commits and the memtable set changes.
    memory: Arc<ArcSwap<ResidentMemTables>>,
    flusher: Arc<MemTableFlusher>,
    /// Source of the writer-global durability cursor, which the SSTable flush asserts
    /// covers the whole frozen memtable before it writes a generation.
    wal_flusher: Arc<WalFlusher>,
    epoch: u64,
    stats: SharedWriteStats,
    observer: Option<Arc<dyn WalObserver>>,
    /// How long a frozen memtable lingers in memory after its flush commits
    /// before [`sweep_expired_frozen`] evicts it. See
    /// `ShardWriterConfig::frozen_memtable_grace`.
    grace: Duration,
    /// Cancelled by `close`, so a failing flush stops retrying.
    stop_retrying: CancellationToken,
}

impl MemTableFlushHandler {
    /// `stop_retrying` ends the retries of a failing flush when cancelled.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        state: Arc<RwLock<WriterState>>,
        memory: Arc<ArcSwap<ResidentMemTables>>,
        flusher: Arc<MemTableFlusher>,
        wal_flusher: Arc<WalFlusher>,
        epoch: u64,
        stats: SharedWriteStats,
        observer: Option<Arc<dyn WalObserver>>,
        grace: Duration,
        stop_retrying: CancellationToken,
    ) -> Self {
        Self {
            state,
            memory,
            flusher,
            wal_flusher,
            epoch,
            stats,
            observer,
            grace,
            stop_retrying,
        }
    }
}

#[cfg(test)]
mod tests;

/// Evict frozen memtables whose post-flush grace has elapsed. Un-stamped
/// (not-yet-flushed) entries are always kept.
///
/// A free function so the sweeper task can call it without owning the flush
/// handler — see [`TaskExecutor::add_periodic`] for why it must not run there.
pub(super) async fn sweep_expired_frozen(
    state: &Arc<RwLock<WriterState>>,
    memory: &Arc<ArcSwap<ResidentMemTables>>,
    grace: Duration,
) {
    let now = now_millis();
    let grace_ms = grace.as_millis() as u64;
    let mut state = state.write().await;
    let before = state.frozen_memtables.len();
    state
        .frozen_memtables
        .retain(|frozen| match frozen.flushed_at_ms {
            Some(flushed_at) => now.saturating_sub(flushed_at) < grace_ms,
            None => true,
        });
    // Eviction is the only thing that reclaims a grace-retained generation,
    // so this is where its bytes leave the memory view.
    if state.frozen_memtables.len() != before {
        publish_memory(memory, &state);
    }
}

#[async_trait]
impl MessageHandler<TriggerMemTableFlush> for MemTableFlushHandler {
    async fn handle(&mut self, message: TriggerMemTableFlush) -> Result<()> {
        match message {
            TriggerMemTableFlush::Flush {
                memtable,
                index_specs,
                done,
            } => {
                let result = self.flush_memtable(memtable, &index_specs).await;
                if let Some(tx) = done {
                    // Send result through the channel - caller is waiting for it
                    let _ = tx.send(result);
                } else {
                    // No done channel, propagate errors
                    result?;
                }
            }
        }
        Ok(())
    }
}

impl MemTableFlushHandler {
    /// Flush the given frozen memtable to Lance storage.
    ///
    /// This method waits for the WAL flush to complete (sent at freeze time),
    /// then flushes to Lance storage. The WAL flush is already queued by
    /// freeze_memtable to ensure strict ordering of WAL entries.
    ///
    /// A failed write is retried with backoff until it lands, fails for good
    /// (fence, refused precondition), or `close` cancels; nothing later flushes meanwhile.
    ///
    /// Whether the flush succeeds or fails, the memtable's flush-completion
    /// watcher is always signaled and the backpressure queue is always drained
    /// for this memtable. Otherwise `wait_for_flush_drain` would observe a
    /// dropped watch channel and return `Err` instead of the actual outcome.
    ///
    /// `index_specs` are the indexes the memtable was built with. They are
    /// rebuilt on the generation, without which `fast_search` misses its rows.
    #[instrument(name = "mt_flush", level = "info", skip_all, fields(generation = memtable.generation(), row_count = memtable.row_count()))]
    async fn flush_memtable(
        &mut self,
        memtable: Arc<MemTable>,
        index_specs: &[MemIndexSpec],
    ) -> Result<crate::dataset::mem_wal::memtable::flush::FlushResult> {
        let start = Instant::now();
        let generation = memtable.generation();

        let flush_result = async {
            // A terminal failure is sticky: later queued generations must
            // stay in the WAL for replay rather than advance the replay cursor.
            self.wal_flusher.check_poisoned()?;
            // Step 1: Wait for WAL flush completion (already queued at freeze time).
            // The TriggerWalFlush message was sent by freeze_memtable to ensure
            // strict ordering of WAL entries. If the freeze didn't trigger a
            // flush (no pending WAL range), there's no completion cell and the
            // memtable was already WAL-flushed by an earlier put.
            let wal_flushed_position =
                if let Some(mut completion_reader) = memtable.take_wal_flush_completion() {
                    match completion_reader.await_value().await {
                        Some(Ok(flush_result)) => flush_result.entry.map(|e| e.position),
                        // Rebuild the typed error so a fence/poison reason
                        // propagates to the memtable-flush caller too.
                        Some(Err(e)) => return Err(e.into_error()),
                        None => {
                            return Err(Error::io(
                                "WAL flush handler exited before reporting completion",
                            ));
                        }
                    }
                } else {
                    None
                };

            // Step 1b: Wait until index application covers this whole memtable.
            //
            // Freeze queues the apply and this flush on separate channels, so
            // without waiting the export can run while the indexes are still
            // behind the batch store. The primary-key sidecar and every
            // secondary index are exported from that asynchronously maintained
            // state. A short PK index can make superseded rows visible, while
            // an empty one publishes no sidecar at all and makes the LSM scan
            // fail when it opens the generation. A short vector index similarly
            // loses rows because SSTable vector search is index-only --
            // `fast_search`, with no brute-force scan.
            //
            // `batch_count` is fixed at freeze, so this waits for a target that
            // cannot move, and the watcher surfaces a poisoned writer rather
            // than blocking on a cursor that will never arrive.
            if let Some(indexes) = memtable.indexes_arc() {
                let target_indexed = memtable.batch_count();
                self.wal_flusher
                    .track_batch(Some(indexes), target_indexed, 0)
                    .wait()
                    .await?;
            }

            // Step 2: Flush the memtable to Lance storage. The covered WAL
            // entry position is either the one we just appended (per-memtable,
            // from the completion cell — authoritative even when concurrent
            // flushes have raced ahead in `state.last_flushed_wal_entry_position`)
            // or, when no flush was triggered at freeze time, the memtable's
            // frozen-at marker captured at freeze. Stamping this into the
            // manifest is what lets replay-on-reopen skip entries this
            // generation covers. The marker can trail the last entry holding
            // this memtable's batches; replay skips those entries by generation.
            let covered_wal_entry_position = wal_flushed_position
                .or_else(|| memtable.frozen_at_wal_entry_position())
                .unwrap_or(0);
            // Rebuild secondary indexes on the SSTable so later
            // queries hit an index instead of scanning. Skip the extra
            // dataset open when there are no indexes to build. The indexed
            // path's future is boxed to keep this async block's nesting
            // under the type-layout recursion limit.
            // Read the durability cursor *after* the WAL-append completion above,
            // not before: the append that makes this memtable durable is the very
            // thing we just waited on, so a cursor sampled earlier would still be
            // short of it and trip the flush precondition.
            let durable = self.wal_flusher.durable();

            let mut delay = FLUSH_RETRY_BASE_DELAY;
            loop {
                let attempt = if index_specs.is_empty() {
                    self.flusher
                        .flush(&memtable, self.epoch, covered_wal_entry_position, durable)
                        .await
                } else {
                    Box::pin(self.flusher.flush_with_indexes(
                        &memtable,
                        self.epoch,
                        index_specs,
                        covered_wal_entry_position,
                        durable,
                    ))
                    .await
                };
                let error = match attempt {
                    Ok(result) => return Ok(result),
                    Err(error) => error,
                };
                self.stats.record_memtable_flush_failure();
                if let Some(observer) = &self.observer {
                    observer.on_memtable_flush_failure(generation, &error);
                }
                // A fence or a refused precondition fails the same way every attempt.
                if matches!(
                    error,
                    Error::Fenced { .. } | Error::Internal { .. } | Error::InvalidInput { .. }
                ) {
                    return Err(error);
                }
                warn!(
                    "MemTable generation {generation} failed to flush; retrying in {delay:?} \
                     before any later generation flushes: {error}"
                );
                tokio::select! {
                    _ = self.stop_retrying.cancelled() => return Err(error),
                    _ = tokio::time::sleep(delay) => {}
                }
                delay = (delay * 2).min(FLUSH_RETRY_MAX_DELAY);
            }
        }
        .await;

        // Step 3: Always signal completion (with the outcome) and drain
        // backpressure state for this memtable, even on failure.
        let flush_result = {
            let mut state = self.state.write().await;
            // Serialize poison with write admission, and publish it before
            // waking any fence or removing the failed generation's watcher.
            let flush_result = flush_result.map_err(|error| {
                let error = if error.fence_reason().is_some() {
                    error
                } else {
                    Error::writer_poisoned(format!(
                        "SSTable flush failed for generation {}: {error}; reopen the writer to replay the WAL",
                        memtable.generation()
                    ))
                };
                self.wal_flusher.poison(&error);
                error
            });
            let durability = match &flush_result {
                Ok(_) => DurabilityResult::ok(),
                Err(error) => DurabilityResult::err(error.to_string()),
            };
            memtable.signal_memtable_flush_complete(durability);
            // Backpressure drain: unconditional so `wait_for_flush_drain`
            // sees the watcher's error signal, not a dropped channel. Which
            // entry comes off the front does not matter — flushes complete out
            // of order, and the snapshot re-derived at the end of this block
            // reads the frozen queue itself rather than tracking a charge.
            state.frozen_flush_watchers.pop_front();
            // Retire the frozen handle on commit success, keyed by generation
            // (non-FIFO completion is fine). Zero grace evicts here; otherwise
            // stamp the grace clock so it lingers for multi-part as-of reads
            // until the grace sweep. On failure leave it un-stamped: its rows
            // stay in the read union until a reopen replays them.
            if flush_result.is_ok() {
                let sstable = memtable.generation();
                if self.grace.is_zero() {
                    state
                        .frozen_memtables
                        .retain(|frozen| frozen.memtable.generation() != sstable);
                } else {
                    let now = now_millis();
                    for frozen in state.frozen_memtables.iter_mut() {
                        if frozen.memtable.generation() == sstable {
                            frozen.flushed_at_ms = Some(now);
                        }
                    }
                }
            }
            // Re-derive after both branches above. A *failed* flush leaves its
            // memtable un-stamped and so still counted, which is the point: its
            // bytes are resident and only another flush can reclaim them.
            publish_memory(&self.memory, &state);
            flush_result
        };

        let result = flush_result?;

        let elapsed = start.elapsed();
        self.stats
            .record_memtable_flush(elapsed, result.rows_flushed);
        if let Some(observer) = &self.observer {
            observer.on_memtable_flush(elapsed, result.rows_flushed);
        }

        info!(
            "Flushed frozen memtable generation {} ({} rows in {:?})",
            result.sstable.generation,
            result.rows_flushed,
            start.elapsed()
        );

        Ok(result)
    }
}
