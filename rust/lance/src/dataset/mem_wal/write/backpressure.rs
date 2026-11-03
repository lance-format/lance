// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Durability notifications, admission control, and shard memory views.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use lance_core::{Error, Result};
use log::{debug, warn};

use super::config::ShardWriterConfig;
use super::writer::ResidentMemTables;
use crate::dataset::mem_wal::scanner::InMemoryMemTableRef;
use crate::dataset::mem_wal::util::{WatchableOnceCell, WatchableOnceCellReader};
use crate::dataset::mem_wal::wal::WalOnlyState;

/// Result of a durability notification.
///
/// This is a simple enum that can be cloned, unlike `Result<(), Error>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DurabilityResult {
    /// Write is now durable.
    Durable,
    /// Write failed with an error message.
    Failed(String),
}

impl DurabilityResult {
    /// Create a successful durability result.
    pub fn ok() -> Self {
        Self::Durable
    }

    /// Create a failed durability result.
    pub fn err(msg: impl Into<String>) -> Self {
        Self::Failed(msg.into())
    }

    /// Check if the result is durable.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Durable)
    }

    /// Convert to a Result.
    pub fn into_result(self) -> Result<()> {
        match self {
            Self::Durable => Ok(()),
            Self::Failed(msg) => Err(Error::io(msg)),
        }
    }
}

/// Type alias for durability watchers.
pub type DurabilityWatcher = WatchableOnceCellReader<DurabilityResult>;

/// Type alias for durability cells.
pub type DurabilityCell = WatchableOnceCell<DurabilityResult>;

/// Statistics for backpressure monitoring.
#[derive(Debug, Default)]
pub struct BackpressureStats {
    /// Total number of *completed* waits.
    total_count: AtomicU64,
    /// Total time completed waits spent parked (in milliseconds).
    total_wait_ms: AtomicU64,
    /// Writers parked in `maybe_apply_backpressure` right now.
    active_count: AtomicU64,
}

impl BackpressureStats {
    /// Create new backpressure stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed backpressure wait.
    pub fn record(&self, wait_ms: u64) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_wait_ms.fetch_add(wait_ms, Ordering::Relaxed);
    }

    /// Count a writer as parked until the returned guard drops. Drop-based
    /// because the caller's future can be cancelled mid-wait, which would
    /// otherwise strand a waiter that never returns.
    pub fn begin_wait(&self) -> BackpressureWaitGuard<'_> {
        self.active_count.fetch_add(1, Ordering::Relaxed);
        BackpressureWaitGuard(self)
    }

    /// Get the completed-wait count.
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Get the total time completed waits spent parked.
    pub fn total_wait_ms(&self) -> u64 {
        self.total_wait_ms.load(Ordering::Relaxed)
    }

    /// Get a snapshot of all stats.
    pub fn snapshot(&self) -> BackpressureStatsSnapshot {
        BackpressureStatsSnapshot {
            total_count: self.total_count.load(Ordering::Relaxed),
            total_wait_ms: self.total_wait_ms.load(Ordering::Relaxed),
            active_count: self.active_count.load(Ordering::Relaxed),
        }
    }
}

/// Keeps its writer counted in `active_count` for as long as it is held.
#[derive(Debug)]
pub struct BackpressureWaitGuard<'a>(&'a BackpressureStats);

impl Drop for BackpressureWaitGuard<'_> {
    fn drop(&mut self) {
        self.0.active_count.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Snapshot of backpressure statistics.
#[derive(Debug, Clone, Default)]
pub struct BackpressureStatsSnapshot {
    /// Number of waits that have *finished*. A wait in progress is not counted
    /// here, and a cancelled one never is.
    pub total_count: u64,
    /// Total time finished waits spent parked (in milliseconds), on the same
    /// denominator as `total_count`.
    pub total_wait_ms: u64,
    /// Writers parked right now. This is the field that answers "am I being
    /// throttled at this instant" — the totals only move once a wait ends, so
    /// they read zero throughout a first, still-ongoing stall.
    pub active_count: u64,
}

/// A **live** view of what one shard is holding in memory.
///
/// The single place a shard's byte totals are computed. Everything that wants
/// them — the admission controller,
/// [`ShardWriter::memory`](super::writer::ShardWriter::memory), an operator gauge
/// — goes through this, so there is no second implementation to drift from it.
///
/// Re-read it on every poll rather than reading once: a controller that delays
/// is waiting for exactly these numbers to fall, so a captured copy would never
/// observe the drain and the wait would never end. A read is one `ArcSwap` load
/// and a sum over the live memtables, so polling is cheap.
#[derive(Clone)]
pub struct ShardMemory {
    source: ShardMemorySource,
    /// Whether the put this reading was taken for can only land after a freeze.
    /// `false` outside a put: the freeze thresholds are per-table config only the
    /// writer can evaluate.
    freeze_required: bool,
}

/// Where a [`ShardMemory`] reads from. A dispatch over the two write modes, not
/// a second accounting: every arm is a field read, and the arithmetic that
/// combines them lives once, in `ShardMemory`.
#[derive(Clone)]
enum ShardMemorySource {
    /// Memtable mode: the published set of resident memtables.
    MemTables(Arc<ArcSwap<ResidentMemTables>>),
    /// WAL-only mode has no memtable; the pending queue is the whole pool, and
    /// there is no flush to await, so a waiter falls back to a short sleep.
    Queue(Arc<WalOnlyState>),
    /// Test-only: synthetic unflushed bytes, re-read each poll, so a controller
    /// can be driven without standing up a writer. Carries no watcher — a
    /// waiter falls back to its sleep, which keeps one poll to one call so a
    /// test can count them.
    #[cfg(test)]
    Fake(Arc<dyn Fn() -> usize + Send + Sync>),
}

impl ShardMemory {
    pub(super) fn memtables(tables: Arc<ArcSwap<ResidentMemTables>>) -> Self {
        Self {
            source: ShardMemorySource::MemTables(tables),
            freeze_required: false,
        }
    }

    pub(super) fn queue(state: Arc<WalOnlyState>) -> Self {
        Self {
            source: ShardMemorySource::Queue(state),
            freeze_required: false,
        }
    }

    /// Mark this reading as taken for a put that cannot land without a freeze.
    pub(super) fn with_freeze_required(mut self, freeze_required: bool) -> Self {
        self.freeze_required = freeze_required;
        self
    }

    /// Whether the put this reading was taken for can only land after a freeze.
    ///
    /// A put that still fits in the active memtable adds nothing to the tier
    /// below; one that must freeze first adds a whole generation to it. A
    /// controller bounding that tier refuses only the latter.
    pub fn freeze_required(&self) -> bool {
        self.freeze_required
    }

    /// Resident bytes of the active memtable — row data plus its in-memory
    /// indexes. In WAL-only mode, the pending queue's bytes.
    pub fn active_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => t
                .load()
                .active
                .as_ref()
                .map_or(0, InMemoryMemTableRef::resident_bytes),
            ShardMemorySource::Queue(q) => q.queue_bytes(),
            #[cfg(test)]
            ShardMemorySource::Fake(f) => f(),
        }
    }

    /// Row-data bytes of the active memtable: the flush unit, summed over the
    /// windows the batches read through. In WAL-only mode, the pending queue's
    /// bytes.
    ///
    /// Deliberately *not* `active_bytes() - index_bytes()`. That difference is
    /// what the batches pin — whole parent buffers, unbounded above this figure
    /// once any batch is a zero-copy slice — and is what the ceiling is built
    /// on. Use this to reason about when a memtable freezes, not about what it
    /// costs.
    pub fn row_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => t
                .load()
                .active
                .as_ref()
                .map_or(0, InMemoryMemTableRef::row_bytes),
            ShardMemorySource::Queue(q) => q.queue_bytes(),
            #[cfg(test)]
            ShardMemorySource::Fake(f) => f(),
        }
    }

    /// The part of [`Self::active_bytes`] held by the active memtable's
    /// in-memory indexes (its PK bloom filter included). A **subset** of that
    /// figure, not another term to add. `0` in WAL-only mode, which has none.
    ///
    /// Broken out because it does not behave like row data and is usually what
    /// explains a shard near its ceiling with few rows in it: an HNSW graph
    /// allocates a fixed part sized by `max_memtable_rows` on the first insert.
    pub fn index_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => t
                .load()
                .active
                .as_ref()
                .map_or(0, InMemoryMemTableRef::index_bytes),
            ShardMemorySource::Queue(_) => 0,
            #[cfg(test)]
            ShardMemorySource::Fake(_) => 0,
        }
    }

    /// Resident bytes of frozen memtables whose flush has not committed.
    /// Always `0` in WAL-only mode.
    pub fn frozen_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => t
                .load()
                .frozen
                .iter()
                .map(InMemoryMemTableRef::resident_bytes)
                .sum(),
            ShardMemorySource::Queue(_) => 0,
            #[cfg(test)]
            ShardMemorySource::Fake(_) => 0,
        }
    }

    /// Resident bytes of frozen memtables that have flushed and are lingering
    /// out `frozen_memtable_grace` so in-flight as-of reads stay batch-resolved.
    ///
    /// Real memory, but no flush reclaims it — the sweeper does, on a timer. A
    /// waiter that blocks on this is waiting for the clock, not for a flush.
    /// `0` in WAL-only mode, and `0` under the default zero grace.
    pub fn grace_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => t
                .load()
                .grace
                .iter()
                .map(InMemoryMemTableRef::resident_bytes)
                .sum(),
            ShardMemorySource::Queue(_) => 0,
            #[cfg(test)]
            ShardMemorySource::Fake(_) => 0,
        }
    }

    /// Bytes only a flush can reclaim: the pool to bound against OOM.
    ///
    /// One load covers both terms, so this cannot cross a freeze and
    /// double-count or lose a memtable the way two separate reads could — which
    /// is why this is not `active_bytes() + frozen_bytes()`.
    pub fn unflushed_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => {
                let tables = t.load();
                tables
                    .active
                    .as_ref()
                    .map_or(0, InMemoryMemTableRef::resident_bytes)
                    + tables
                        .frozen
                        .iter()
                        .map(InMemoryMemTableRef::resident_bytes)
                        .sum::<usize>()
            }
            ShardMemorySource::Queue(q) => q.queue_bytes(),
            #[cfg(test)]
            ShardMemorySource::Fake(f) => f(),
        }
    }

    /// Every resident byte the shard is holding: [`Self::unflushed_bytes`] plus
    /// the generations already flushed but still inside `frozen_memtable_grace`.
    ///
    /// This is the figure a process-wide budget wants. `unflushed_bytes` is the
    /// narrower one — what a flush can still reclaim — and is what the per-shard
    /// valve throttles on, because throttling on grace-retained memory would
    /// stall the writer waiting for a sweeper tick. The two differ only when a
    /// grace is configured; under the default zero grace they are equal.
    pub fn retained_bytes(&self) -> usize {
        match &self.source {
            ShardMemorySource::MemTables(t) => {
                let tables = t.load();
                tables
                    .active
                    .as_ref()
                    .map_or(0, InMemoryMemTableRef::resident_bytes)
                    + tables
                        .frozen
                        .iter()
                        .chain(tables.grace.iter())
                        .map(InMemoryMemTableRef::resident_bytes)
                        .sum::<usize>()
            }
            ShardMemorySource::Queue(q) => q.queue_bytes(),
            #[cfg(test)]
            ShardMemorySource::Fake(f) => f(),
        }
    }

    /// What could reclaim memory here while a writer waits on it.
    ///
    /// A blocking controller must consult this rather than assume that waiting
    /// eventually works: a shard can be over its ceiling with nothing running
    /// that would bring it back down. See [`Drain`].
    pub fn drain(&self) -> Drain {
        match &self.source {
            ShardMemorySource::MemTables(t) => {
                let tables = t.load();
                match tables.oldest_flush.clone() {
                    Some(flush) => Drain::Flush(flush),
                    // The sweeper will drop these once their grace elapses.
                    None if !tables.grace.is_empty() => Drain::Background,
                    None => Drain::Stalled,
                }
            }
            // The WAL flusher drains the queue on its own schedule.
            ShardMemorySource::Queue(_) => Drain::Background,
            #[cfg(test)]
            ShardMemorySource::Fake(_) => Drain::Background,
        }
    }
}

/// What can bring a shard back under its ceiling while a writer waits.
///
/// Returned by [`ShardMemory::drain`] so a blocking controller can tell a wait
/// that ends from one that cannot. The distinction is not academic: a flush
/// that fails leaves its generation resident and charged with nothing queued to
/// retry it, and index memory is charged to the ceiling while the freeze trigger
/// measures row bytes — either can put a shard over budget with no flush in
/// flight, and only a new write would start one.
#[derive(Debug)]
pub enum Drain {
    /// Park on this flush. Completing it retires a whole generation.
    Flush(DurabilityWatcher),
    /// Nothing to park on, but something is draining the shard on its own
    /// schedule — the WAL flusher, or the grace sweeper. Poll.
    Background,
    /// Nothing outstanding. Only a new write would start a flush, and a waiter
    /// here is precisely what is keeping writes out, so waiting cannot end.
    Stalled,
}

impl Debug for ShardMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardMemory")
            .field("active_bytes", &self.active_bytes())
            .field("frozen_bytes", &self.frozen_bytes())
            .field("grace_bytes", &self.grace_bytes())
            .finish()
    }
}

/// Admission control for [`ShardWriter::put`](super::writer::ShardWriter::put),
/// consulted before each write.
///
/// Two implementations ship in-tree and exactly one runs per writer:
/// [`LocalBackpressureController`] by default, or whatever an embedder installs
/// via [`ShardWriterConfig::backpressure`]. They do not layer — the injected
/// one owns the whole policy, per-shard rules included, which is what
/// [`ShardMemory`] is for.
///
/// An embedder replaces the default when it has budgets lance cannot see: a
/// process-wide memtable total across shards, a page-cache working set. Because
/// [`ShardMemory`] is self-sufficient, a replacement that also wants the
/// built-in per-shard behaviour can call [`LocalBackpressureController`] from
/// inside its own implementation rather than reimplementing it.
#[async_trait::async_trait]
pub trait BackpressureController: Send + Sync + Debug {
    /// Decide whether to admit a write into a shard currently holding `shard`.
    ///
    /// May await to delay the writer, or return [`Error::Backpressure`] to
    /// refuse it. Any other error is a real failure.
    ///
    /// Deliberately not told the incoming batch's size. Batches are already
    /// decoded and resident by the time this runs, so there is nothing to
    /// reserve against — refusing does not un-allocate them. Bounding a single
    /// write's memory is the ingress's job, not this one's.
    async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()>;

    /// Whether a freeze may proceed right now. Read under the writer lock, so
    /// it must answer from already-published state without blocking.
    ///
    /// `false` pins the active memtable at its cap and refuses the puts that
    /// needed the freeze. Not consulted by
    /// [`ShardWriter::force_freeze_active`](super::writer::ShardWriter::force_freeze_active),
    /// which drain and drop rely on to freeze whatever the tier below looks like.
    fn may_freeze_memtable(&self) -> bool {
        true
    }

    /// Throttling counters for
    /// [`ShardWriter::backpressure_stats`](super::writer::ShardWriter::backpressure_stats).
    /// An injected
    /// controller keeps its own metrics, so the default reports zeros rather
    /// than requiring it to maintain lance's.
    fn stats_snapshot(&self) -> BackpressureStatsSnapshot {
        BackpressureStatsSnapshot::default()
    }
}

/// The controller guarding this writer: the embedder's if one was injected,
/// otherwise lance's own [`LocalBackpressureController`].
pub(super) fn resolve_backpressure(config: &ShardWriterConfig) -> Arc<dyn BackpressureController> {
    match &config.backpressure {
        Some(injected) => injected.clone(),
        None => Arc::new(LocalBackpressureController::new(config)),
    }
}

/// Poll cadence while a shard is over its ceiling and something other than a
/// flush is expected to bring it back down — there is no watcher to park on.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long a shard must keep reading as [`Drain::Stalled`] before the valve
/// gives up on it rather than waiting.
///
/// The classification is momentarily wrong under concurrency: another writer
/// holding the state lock has already grown the active memtable past the
/// ceiling but has not yet reached the `freeze_memtable` that publishes a
/// watcher, so a waiter sampling in between sees over-budget with nothing
/// outstanding. That window is one locked section of in-memory work, orders of
/// magnitude under this. A real stall never closes.
const STALL_GRACE: Duration = Duration::from_secs(1);

/// The per-shard memtable valve: lance's default when nothing is injected.
///
/// Soft and blocking — it stalls the producer until a flush drains the pool.
/// It refuses a write only when the pool *cannot* drain (see [`Drain`]);
/// waiting there would park the writer for good. Replaced wholesale by
/// [`ShardWriterConfig::backpressure`], so an embedder that injects a
/// controller takes on the per-shard ceiling this provides (see
/// [`ShardMemory`]).
#[derive(Debug)]
pub struct LocalBackpressureController {
    max_unflushed_memtable_bytes: usize,
    log_interval: Duration,
    stats: Arc<BackpressureStats>,
}

impl LocalBackpressureController {
    pub(super) fn new(config: &ShardWriterConfig) -> Self {
        Self {
            max_unflushed_memtable_bytes: config.max_unflushed_memtable_bytes,
            log_interval: config.backpressure_log_interval,
            stats: Arc::new(BackpressureStats::new()),
        }
    }

    /// Get backpressure stats.
    pub fn stats(&self) -> &Arc<BackpressureStats> {
        &self.stats
    }
}

#[async_trait::async_trait]
impl BackpressureController for LocalBackpressureController {
    fn stats_snapshot(&self) -> BackpressureStatsSnapshot {
        self.stats.snapshot()
    }

    /// Blocks while this shard's unflushed bytes are at or above
    /// `max_unflushed_memtable_bytes`, waiting on the oldest flush.
    ///
    /// Returns [`Error::Backpressure`] in the one case where blocking would
    /// never end: over the ceiling with no flush outstanding and nothing
    /// draining in the background, for `STALL_GRACE` running. Only a write
    /// starts a flush, and this valve is what holds writes out, so the wait
    /// would have no event to end on. The
    /// error names the breakdown, because the two ways to get there — a flush
    /// that failed and left its generation charged, or index memory carrying a
    /// memtable past the ceiling while the freeze trigger still sees small row
    /// bytes — are both configuration or operational conditions an operator has
    /// to act on rather than wait out.
    async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()> {
        let start = std::time::Instant::now();
        let mut iteration = 0u32;
        // Held for the whole stall so an operator polling mid-wait sees it; the
        // totals below cannot, since they only move once the wait ends.
        let mut active_wait = None;
        // When the shard first read as un-drainable, cleared the moment it
        // stops. See `STALL_GRACE`.
        let mut stalled_since: Option<std::time::Instant> = None;

        loop {
            // Re-read every iteration: this loop is waiting for exactly these
            // bytes to fall, so a value captured once would never see the drain.
            let unflushed_memtable_bytes = shard.unflushed_bytes();

            // Check if under threshold
            if unflushed_memtable_bytes < self.max_unflushed_memtable_bytes {
                if iteration > 0 {
                    let wait_ms = start.elapsed().as_millis() as u64;
                    self.stats.record(wait_ms);
                }
                return Ok(());
            }

            iteration += 1;
            if active_wait.is_none() {
                active_wait = Some(self.stats.begin_wait());
            }

            debug!(
                "Backpressure triggered: unflushed_bytes={}, max={}, iteration={}",
                unflushed_memtable_bytes, self.max_unflushed_memtable_bytes, iteration
            );

            match shard.drain() {
                Drain::Flush(mut mem_watcher) => {
                    stalled_since = None;
                    tokio::select! {
                        _ = mem_watcher.await_value() => {}
                        _ = tokio::time::sleep(self.log_interval) => {
                            warn!(
                                "Backpressure wait timeout, continuing to wait: unflushed_bytes={}, interval={}s, iteration={}",
                                unflushed_memtable_bytes,
                                self.log_interval.as_secs(),
                                iteration
                            );
                        }
                    }
                }
                // Someone else is draining on a schedule of their own; poll
                // rather than busy-loop.
                Drain::Background => {
                    stalled_since = None;
                    tokio::time::sleep(DRAIN_POLL_INTERVAL).await
                }
                Drain::Stalled => {
                    let since = *stalled_since.get_or_insert_with(std::time::Instant::now);
                    if since.elapsed() < STALL_GRACE {
                        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
                        continue;
                    }
                    return Err(Error::backpressure(format!(
                        "shard is at its memtable ceiling with no flush outstanding, so waiting \
                         cannot drain it: unflushed_bytes={}, max={}, active_bytes={} (of which \
                         index_bytes={}), frozen_bytes={}. The active memtable freezes on resident \
                         bytes, so reaching here means a flush failed and left its generation \
                         charged with nothing queued to retry it",
                        unflushed_memtable_bytes,
                        self.max_unflushed_memtable_bytes,
                        shard.active_bytes(),
                        shard.index_bytes(),
                        shard.frozen_bytes(),
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
