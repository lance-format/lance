// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stderr)]

//! Write path for MemWAL.
//!
//! This module contains all components for the write path:
//! - [`ShardWriter`] - Main writer interface for a single shard
//! - [`MemTable`] - In-memory table storing Arrow RecordBatches
//! - [`WalFlusher`] - Write-ahead log buffer for durability (Arrow IPC format)
//! - [`IndexStore`] - In-memory index management
//! - [`MemTableFlusher`] - Flush MemTable to storage as single Lance file

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_index::mem_wal::ShardManifest;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use log::{debug, error, info, warn};
use object_store::path::Path;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{Interval, interval_at};
use tokio_util::sync::CancellationToken;
use tracing::instrument;
use uuid::Uuid;

pub use super::index::{
    BTreeIndexConfig, BTreeMemIndex, FtsIndexConfig, HnswIndexConfig, IndexStore, MemIndexConfig,
    MemIndexKind, validate_index_configs,
};
pub use super::memtable::CacheConfig;
pub use super::memtable::MemTable;
pub use super::memtable::batch_store::{BatchStore, StoreFull, StoredBatch};
pub use super::memtable::flush::MemTableFlusher;
pub use super::memtable::scanner::MemTableScanner;
pub use super::util::{WatchableOnceCell, WatchableOnceCellReader};
pub use super::wal::{WalEntry, WalEntryData, WalFlushFailure, WalFlushResult, WalFlusher};

use super::memtable::flush::TriggerMemTableFlush;
use super::observer::WalObserver;
use super::scanner::InMemoryMemTableRef;
use super::scanner::SsTableWarmer;
use super::wal::{
    BatchDurableWatcher, TriggerIndexApply, TriggerWalFlush, WalAppender, WalFlushSource,
    WalOnlyState, WalRetryConfig, WalTailer, WriterCursors, apply_index_range, empty_flush_result,
};
use super::{TOMBSTONE, relax_non_pk_nullability, schema_with_tombstone};
use crate::session::Session;

use super::manifest::ShardManifestStore;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a shard writer.
#[derive(Debug, Clone)]
pub struct ShardWriterConfig {
    /// Unique identifier for this shard (UUID v4).
    pub shard_id: Uuid,

    /// Shard spec ID this shard was created with.
    /// A value of 0 indicates a manually-created shard not governed by any spec.
    pub shard_spec_id: u32,

    /// Whether to wait for WAL flush before returning from writes.
    ///
    /// When true (durable writes):
    /// - Each write waits for WAL persistence before returning
    /// - Guarantees no data loss on crash
    /// - Higher latency due to object storage writes
    ///
    /// When false (non-durable writes):
    /// - Writes return immediately after buffering in memory
    /// - Potential data loss if process crashes before flush
    /// - Lower latency, batched S3 operations
    pub durable_write: bool,

    /// Maximum WAL buffer size in bytes before triggering a flush.
    ///
    /// This is a soft threshold - write batches are atomic and won't be split.
    /// WAL flushes when buffer exceeds this size OR when `max_wal_flush_interval` elapses.
    /// Default: 10MB
    pub max_wal_buffer_size: usize,

    /// Time-based WAL flush interval.
    ///
    /// WAL buffer will be flushed after this duration even if size threshold
    /// hasn't been reached. This ensures bounded data loss window in non-durable mode
    /// and prevents accumulating too much data before flushing to object storage.
    /// Default: 100ms
    pub max_wal_flush_interval: Option<Duration>,

    /// Times a failed WAL PUT is retried (same position, exponential backoff)
    /// before the writer self-fences. On exhaustion the writer is poisoned
    /// (`Error::writer_poisoned`) and must be reopened to replay. Absorbs
    /// transient errors beyond `object_store`'s own retries. Default: 3.
    pub max_wal_persist_retries: usize,

    /// Base backoff before the first WAL persistence retry; subsequent retries
    /// back off exponentially (capped). Default: 50ms.
    pub wal_persist_retry_base_delay: Duration,

    /// Maximum MemTable size in bytes before triggering a flush to storage.
    ///
    /// MemTable size is checked every `max_wal_flush_interval` (during WAL flush ticks).
    /// Default: 256MB
    pub max_memtable_size: usize,

    /// Maximum number of rows in a MemTable.
    ///
    /// Used to pre-allocate the in-memory HNSW graph and vector storage
    /// capacity. When the memtable reaches capacity, it will be flushed.
    /// Default: 100,000 rows
    pub max_memtable_rows: usize,

    /// Maximum number of batches in a MemTable.
    ///
    /// Used to pre-allocate batch storage. When this limit is reached,
    /// memtable will be flushed. Sized for typical ML workloads with
    /// 1024-dim vectors (~82KB per 20-row batch).
    /// Default: 8,000 batches
    pub max_memtable_batches: usize,

    /// Batch size for parallel HEAD requests when scanning for manifest versions.
    ///
    /// Higher values scan faster but use more parallel requests.
    /// Default: 2
    pub manifest_scan_batch_size: usize,

    /// Maximum unflushed bytes before applying backpressure.
    ///
    /// When total unflushed data (active memtable + frozen memtables) exceeds this,
    /// new writes will block until some data is flushed to storage.
    /// This prevents unbounded memory growth during write spikes.
    ///
    /// Default: 1GB
    pub max_unflushed_memtable_bytes: usize,

    /// Interval for logging warnings when writes are blocked by backpressure.
    ///
    /// When a write is blocked waiting for WAL flush, memtable flush, or index
    /// updates to complete, a warning is logged after this duration. The write
    /// will continue waiting indefinitely (it never fails due to backpressure),
    /// but warnings are logged at this interval to help diagnose slow flushes.
    ///
    /// Default: 30 seconds
    pub backpressure_log_interval: Duration,

    /// Interval for periodic stats logging.
    ///
    /// Stats (write throughput, backpressure events, memtable size) are logged
    /// at this interval. Set to None to disable periodic stats logging.
    ///
    /// Default: 60 seconds
    pub stats_log_interval: Option<Duration>,

    /// How long a frozen memtable lingers in memory after its flush commits,
    /// before it is evicted and served only from the on-disk SSTable dataset.
    ///
    /// `Duration::ZERO` (the default) disables retention: evict on commit, no
    /// sweep ticker. Correct for single-shot queries, which can't observe a
    /// generation evicted mid-read.
    ///
    /// A non-zero value is required only for queries split across reads (e.g.
    /// fresh tier and base table read separately, then deduped): the SSTable
    /// dataset loses the per-batch boundaries that bound as-of membership
    /// (see [`crate::dataset::mem_wal::scanner::FreshTierWatermark`]), so a
    /// generation evicted between a query's reads can serve a stale row. Set it
    /// above the worst-case multi-part query latency, with margin.
    pub frozen_memtable_grace: Duration,

    /// Whether to maintain an in-memory MemTable on top of the WAL.
    ///
    /// When `true` (default), the writer maintains an in-memory `MemTable`,
    /// optionally with indexes, and asynchronously flushes frozen MemTables
    /// to Lance files alongside the WAL.
    ///
    /// When `false`, the writer skips the MemTable layer entirely:
    /// - No MemTable / BatchStore / IndexStore is allocated.
    /// - `index_configs` must be empty (validated at open time).
    /// - No MemTable freezing or Lance file flushing happens.
    /// - `max_unflushed_memtable_bytes` is reused as the backpressure
    ///   budget for the WAL-only pending-batch queue: `put` blocks while
    ///   the queue's estimated bytes meet or exceed this threshold.
    /// - The async batched WAL pipeline still runs, driven by the same
    ///   `max_wal_buffer_size`, `max_wal_flush_interval`, and
    ///   `durable_write` settings as MemTable mode.
    ///
    /// MemTable-tied tunables (`max_memtable_size`, `max_memtable_rows`,
    /// `max_memtable_batches`) are ignored when `enable_memtable == false`.
    ///
    /// For raw single-entry synchronous atomic appends with no buffering and
    /// no background tasks, use `WalAppender` directly — it is a strictly
    /// lower-level primitive.
    pub enable_memtable: bool,

    /// Per-index HNSW build-parameter overrides for maintained vector indexes,
    /// keyed by index name.
    ///
    /// These control the in-memory HNSW graph this writer builds for its
    /// MemTable (and, on flush, the on-disk graph serialized from it). They are
    /// a property of the writer that builds the MemTable, not of the index
    /// definition: each SSTable is independent, so different writers
    /// may use different parameters. An index without an entry uses the default
    /// build parameters. `num_edges` is the HNSW graph degree (level 0 retains
    /// `2 * num_edges`), equivalent to FAISS's `M`.
    ///
    /// Default: empty.
    pub hnsw_params: HashMap<String, HnswBuildParams>,

    /// Optional warmer fired pre-commit for each new generation (zero cold reads
    /// on first query). Wired to the flusher; supplied by the consumer (e.g. the
    /// WAL pod). Default: `None`.
    pub warmer: Option<Arc<dyn SsTableWarmer>>,

    /// Optional sink for write-path events, currently flush latency. Wired to
    /// the flush handlers; supplied by the consumer (e.g. the WAL pod), which
    /// owns the aggregation Lance would otherwise have to pick for it.
    /// Default: `None`.
    pub observer: Option<Arc<dyn WalObserver>>,

    /// Store params the base dataset was opened with, reused for the flusher's
    /// opens + writes (base + generations). Injected by `mem_wal_writer`; set
    /// these to the params of the dataset at `base_uri`, not to params bound to
    /// some other path — generation URIs are derived from them.
    /// Default: `None` (open by URI alone).
    pub store_params: Option<ObjectStoreParams>,

    /// Session for those opens, injected alongside `store_params`.
    /// Default: `None`.
    pub session: Option<Arc<Session>>,

    /// Admission control for every `put`, **replacing** lance's built-in
    /// per-shard valve ([`LocalBackpressureController`]).
    ///
    /// For embedders whose budgets lance cannot see: a process-wide memtable
    /// total across shards, a page-cache working set. Because it replaces
    /// rather than layers, the injected controller owns the whole policy — a
    /// per-shard ceiling included, for which it is handed [`ShardMemory`].
    /// Unlike the built-in valve it may also reject: see
    /// [`Error::Backpressure`].
    /// Default: `None` (use the built-in valve).
    pub backpressure: Option<Arc<dyn BackpressureController>>,
}

impl Default for ShardWriterConfig {
    fn default() -> Self {
        Self {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 10 * 1024 * 1024, // 10MB
            max_wal_flush_interval: Some(Duration::from_millis(100)), // 100ms
            max_wal_persist_retries: 3,
            wal_persist_retry_base_delay: Duration::from_millis(50),
            max_memtable_size: 256 * 1024 * 1024, // 256MB
            max_memtable_rows: 100_000,           // 100k rows
            max_memtable_batches: 8_000,          // 8k batches
            manifest_scan_batch_size: 2,
            max_unflushed_memtable_bytes: 1024 * 1024 * 1024, // 1GB
            backpressure_log_interval: Duration::from_secs(30),
            stats_log_interval: Some(Duration::from_secs(60)), // 1 minute
            frozen_memtable_grace: Duration::ZERO,
            enable_memtable: true,
            hnsw_params: HashMap::new(),
            warmer: None,
            observer: None,
            store_params: None,
            session: None,
            backpressure: None,
        }
    }
}

impl ShardWriterConfig {
    /// Create a new configuration with the given shard ID.
    pub fn new(shard_id: Uuid) -> Self {
        Self {
            shard_id,
            ..Default::default()
        }
    }

    /// Set the shard spec ID.
    pub fn with_shard_spec_id(mut self, spec_id: u32) -> Self {
        self.shard_spec_id = spec_id;
        self
    }

    /// Set durable writes mode.
    pub fn with_durable_write(mut self, durable: bool) -> Self {
        self.durable_write = durable;
        self
    }

    /// Set maximum WAL buffer size.
    pub fn with_max_wal_buffer_size(mut self, size: usize) -> Self {
        self.max_wal_buffer_size = size;
        self
    }

    /// Set maximum flush interval.
    pub fn with_max_wal_flush_interval(mut self, interval: Duration) -> Self {
        self.max_wal_flush_interval = Some(interval);
        self
    }

    /// Set the number of WAL persistence retries before the writer self-fences.
    /// See [`ShardWriterConfig::max_wal_persist_retries`].
    pub fn with_max_wal_persist_retries(mut self, retries: usize) -> Self {
        self.max_wal_persist_retries = retries;
        self
    }

    /// Set the base backoff before the first WAL persistence retry.
    /// See [`ShardWriterConfig::wal_persist_retry_base_delay`].
    pub fn with_wal_persist_retry_base_delay(mut self, delay: Duration) -> Self {
        self.wal_persist_retry_base_delay = delay;
        self
    }

    /// Set maximum MemTable size.
    pub fn with_max_memtable_size(mut self, size: usize) -> Self {
        self.max_memtable_size = size;
        self
    }

    /// Set maximum MemTable rows for index pre-allocation.
    pub fn with_max_memtable_rows(mut self, rows: usize) -> Self {
        self.max_memtable_rows = rows;
        self
    }

    /// Set maximum MemTable batches for batch store pre-allocation.
    pub fn with_max_memtable_batches(mut self, batches: usize) -> Self {
        self.max_memtable_batches = batches;
        self
    }

    /// Set manifest scan batch size.
    pub fn with_manifest_scan_batch_size(mut self, size: usize) -> Self {
        self.manifest_scan_batch_size = size;
        self
    }

    /// Set maximum unflushed bytes for backpressure.
    pub fn with_max_unflushed_memtable_bytes(mut self, size: usize) -> Self {
        self.max_unflushed_memtable_bytes = size;
        self
    }

    /// Replace the built-in per-shard valve with `controller`. See
    /// [`Self::backpressure`].
    pub fn with_backpressure(mut self, controller: Arc<dyn BackpressureController>) -> Self {
        self.backpressure = Some(controller);
        self
    }

    /// Set backpressure log interval.
    pub fn with_backpressure_log_interval(mut self, interval: Duration) -> Self {
        self.backpressure_log_interval = interval;
        self
    }

    /// Set stats logging interval. Use None to disable periodic stats logging.
    pub fn with_stats_log_interval(mut self, interval: Option<Duration>) -> Self {
        self.stats_log_interval = interval;
        self
    }

    /// Set how long an SSTable lingers in memory before eviction. MUST
    /// exceed the maximum query elapsed time — see `frozen_memtable_grace`.
    pub fn with_frozen_memtable_grace(mut self, grace: Duration) -> Self {
        self.frozen_memtable_grace = grace;
        self
    }

    /// Toggle the in-memory MemTable layer. See `enable_memtable` for the
    /// full WAL-only-mode contract. Defaults to `true`.
    pub fn with_enable_memtable(mut self, enable: bool) -> Self {
        self.enable_memtable = enable;
        self
    }

    /// Override the HNSW build parameters for a maintained vector index.
    ///
    /// Applies to the in-memory HNSW graph this writer builds for `index_name`
    /// (and the on-disk graph serialized from it on flush). Calling this
    /// repeatedly for the same index replaces the previous value.
    pub fn with_hnsw_params(
        mut self,
        index_name: impl Into<String>,
        params: HnswBuildParams,
    ) -> Self {
        self.hnsw_params.insert(index_name.into(), params);
        self
    }
}

// ============================================================================
// Background Task Infrastructure
// ============================================================================

/// Factory function for creating ticker messages.
type MessageFactory<T> = Box<dyn Fn() -> T + Send + Sync>;

/// Handler trait for processing messages in a background task.
#[async_trait]
pub trait MessageHandler<T: Send + Debug + 'static>: Send {
    /// Define periodic tickers that generate messages.
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<T>)> {
        vec![]
    }

    /// Handle a single message.
    async fn handle(&mut self, message: T) -> Result<()>;

    /// Cleanup on shutdown.
    async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
        Ok(())
    }
}

/// Dispatcher that runs the event loop for a single message handler.
struct TaskDispatcher<T: Send + Debug> {
    handler: Box<dyn MessageHandler<T>>,
    rx: mpsc::UnboundedReceiver<T>,
    cancellation_token: CancellationToken,
    name: String,
}

impl<T: Send + Debug + 'static> TaskDispatcher<T> {
    async fn run(mut self) -> Result<()> {
        let tickers = self.handler.tickers();
        let mut ticker_intervals: Vec<(Interval, MessageFactory<T>)> = tickers
            .into_iter()
            .map(|(duration, factory)| {
                let mut interval = interval_at(tokio::time::Instant::now() + duration, duration);
                // `Burst` (the default) replays every tick missed while `handle()`
                // was running. A WAL append can easily outlast its own interval, so
                // the missed ticks pile up, the ticker arm below is always ready,
                // and — being `biased` — it starves `rx` indefinitely: freeze
                // completion cells and `close()`'s final append never get handled.
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                (interval, factory)
            })
            .collect();

        // A single failing message must not bring down the dispatcher:
        // the WAL flusher and MemTable flusher both run on this loop,
        // and dropping their channels deadlocks all subsequent puts
        // (and panics any task waiting on the corresponding watch).
        // Log and keep draining; the worst case for a transient flush
        // failure is replay from the WAL on next open.
        let result = loop {
            if ticker_intervals.is_empty() {
                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                }
            } else {
                let first_ticker = ticker_intervals.first_mut().unwrap();
                let first_interval = &mut first_ticker.0;

                tokio::select! {
                    biased;
                    _ = self.cancellation_token.cancelled() => {
                        debug!("Task '{}' received cancellation", self.name);
                        break Ok(());
                    }
                    // Explicit messages outrank the ticker. A tick is a backstop
                    // — it only ever *adds* an append that a real trigger would
                    // have made anyway — whereas a message may be a freeze's
                    // completion cell or `close()`'s final append, which nothing
                    // else will deliver. Polling the ticker first (as this did)
                    // lets a ready tick starve them.
                    msg = self.rx.recv() => {
                        match msg {
                            Some(message) => {
                                if let Err(e) = self.handler.handle(message).await {
                                    error!("Task '{}' error handling message: {}", self.name, e);
                                }
                            }
                            None => {
                                debug!("Task '{}' channel closed", self.name);
                                break Ok(());
                            }
                        }
                    }
                    _ = first_interval.tick() => {
                        let message = (ticker_intervals[0].1)();
                        if let Err(e) = self.handler.handle(message).await {
                            error!("Task '{}' error handling ticker message: {}", self.name, e);
                        }
                    }
                }
            }
        };

        let cleanup_ok = result.is_ok();
        self.handler.cleanup(cleanup_ok).await?;

        info!("Task dispatcher '{}' stopped", self.name);
        result
    }
}

/// Executor that manages multiple background tasks.
pub struct TaskExecutor {
    tasks: StdRwLock<Vec<(String, JoinHandle<Result<()>>)>>,
    cancellation_token: CancellationToken,
}

impl TaskExecutor {
    pub fn new() -> Self {
        Self {
            tasks: StdRwLock::new(Vec::new()),
            cancellation_token: CancellationToken::new(),
        }
    }

    pub fn add_handler<T: Send + Debug + 'static>(
        &self,
        name: String,
        handler: Box<dyn MessageHandler<T>>,
        rx: mpsc::UnboundedReceiver<T>,
    ) -> Result<()> {
        let dispatcher = TaskDispatcher {
            handler,
            rx,
            cancellation_token: self.cancellation_token.clone(),
            name: name.clone(),
        };

        let handle = tokio::spawn(async move { dispatcher.run().await });
        self.tasks.write().unwrap().push((name, handle));
        Ok(())
    }

    /// Cancel and join every handler registered by [`Self::add_handler`].
    ///
    /// Cancellation causes each handler's dispatcher to stop accepting messages and call
    /// [`MessageHandler::cleanup`]. This method waits for every dispatcher to finish, even if
    /// cleanup fails or a dispatcher panics, and then returns the first such failure. It returns
    /// `Ok(())` only after every registered handler has been cleaned up successfully.
    pub async fn shutdown_all(&self) -> Result<()> {
        info!("Shutting down all tasks");
        self.cancellation_token.cancel();

        let tasks = std::mem::take(&mut *self.tasks.write().unwrap());
        let mut first_error = None;
        for (name, handle) in tasks {
            match handle.await {
                Ok(Ok(())) => debug!("Task '{}' completed successfully", name),
                Ok(Err(e)) => {
                    warn!("Task '{}' completed with error: {}", name, e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    error!("Task '{}' panicked: {}", name, e);
                    if first_error.is_none() {
                        first_error = Some(Error::internal(format!(
                            "Task '{name}' panicked during shutdown: {e}"
                        )));
                    }
                }
            }
        }

        first_error.map_or(Ok(()), Err)
    }
}

impl Default for TaskExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Durability and Backpressure Types
// ============================================================================

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
/// them — the admission controller, [`ShardWriter::memory`], an operator gauge
/// — goes through this, so there is no second implementation to drift from it.
///
/// Re-read it on every poll rather than reading once: a controller that delays
/// is waiting for exactly these numbers to fall, so a captured copy would never
/// observe the drain and the wait would never end. A read is one `ArcSwap` load
/// and a sum over the live memtables, so polling is cheap.
#[derive(Clone)]
pub struct ShardMemory(ShardMemorySource);

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
    fn memtables(tables: Arc<ArcSwap<ResidentMemTables>>) -> Self {
        Self(ShardMemorySource::MemTables(tables))
    }

    fn queue(state: Arc<WalOnlyState>) -> Self {
        Self(ShardMemorySource::Queue(state))
    }

    /// Resident bytes of the active memtable — row data plus its in-memory
    /// indexes. In WAL-only mode, the pending queue's bytes.
    pub fn active_bytes(&self) -> usize {
        match &self.0 {
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
    /// on. Use this to reason about when a memtable seals, not about what it
    /// costs.
    pub fn row_bytes(&self) -> usize {
        match &self.0 {
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
    /// explains a shard near its ceiling with few rows in it: an HNSW graph is
    /// pre-allocated in full on the first insert.
    pub fn index_bytes(&self) -> usize {
        match &self.0 {
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

    /// Resident bytes of sealed memtables whose flush has not committed.
    /// Always `0` in WAL-only mode.
    pub fn frozen_bytes(&self) -> usize {
        match &self.0 {
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

    /// Resident bytes of sealed memtables that have flushed and are lingering
    /// out `frozen_memtable_grace` so in-flight as-of reads stay batch-resolved.
    ///
    /// Real memory, but no flush reclaims it — the sweeper does, on a timer. A
    /// waiter that blocks on this is waiting for the clock, not for a flush.
    /// `0` in WAL-only mode, and `0` under the default zero grace.
    pub fn grace_bytes(&self) -> usize {
        match &self.0 {
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
        match &self.0 {
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
        match &self.0 {
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
        match &self.0 {
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
/// retry it, and index memory is charged to the ceiling while the seal trigger
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

/// Admission control for [`ShardWriter::put`], consulted before each write.
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

    /// Throttling counters for [`ShardWriter::backpressure_stats`]. An injected
    /// controller keeps its own metrics, so the default reports zeros rather
    /// than requiring it to maintain lance's.
    fn stats_snapshot(&self) -> BackpressureStatsSnapshot {
        BackpressureStatsSnapshot::default()
    }
}

/// The controller guarding this writer: the embedder's if one was injected,
/// otherwise lance's own [`LocalBackpressureController`].
fn resolve_backpressure(config: &ShardWriterConfig) -> Arc<dyn BackpressureController> {
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
    fn new(config: &ShardWriterConfig) -> Self {
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
    /// memtable past the ceiling while the seal trigger still sees small row
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
                         index_bytes={}), frozen_bytes={}. The active memtable seals on resident \
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

/// Result of a write operation.
#[derive(Debug)]
pub struct WriteResult {
    /// Range of batch positions [start, end) for inserted batches.
    /// For a single batch, this is [pos, pos+1).
    pub batch_positions: std::ops::Range<usize>,
}

/// A sealed memtable kept queryable in memory. `flushed_at_ms` is `None` while
/// the generation is still awaiting (or retrying) its flush, and `Some(t)` once
/// the flush commits — after which it lingers for `frozen_memtable_grace` so
/// in-flight as-of reads keep batch-resolved membership, then is swept.
struct FrozenMemTable {
    memtable: Arc<MemTable>,
    flushed_at_ms: Option<u64>,
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
struct ResidentMemTables {
    /// The active memtable, or `None` before the first publish.
    active: Option<InMemoryMemTableRef>,
    /// Sealed memtables whose flush has not committed — including any left
    /// resident by a *failed* flush, which are the ones most worth metering.
    frozen: Vec<InMemoryMemTableRef>,
    /// Sealed memtables whose flush *did* commit, lingering out
    /// `frozen_memtable_grace` before `SweepExpired` drops them.
    ///
    /// Held apart from `frozen` rather than dropped from the view: no flush can
    /// reclaim these, so metering the flush valve on them would throttle against
    /// memory that is going away on a timer. They are still resident, though,
    /// and a process-wide budget has to see them — hence
    /// [`ShardMemory::retained_bytes`] alongside
    /// [`ShardMemory::unflushed_bytes`].
    grace: Vec<InMemoryMemTableRef>,
    /// The oldest flush still outstanding, or `None` when none is.
    ///
    /// Deliberately not backfilled with the active memtable's watcher. That
    /// watcher only fires when the active memtable is sealed, which only a put
    /// does — so offering it to a waiter that is itself holding puts out names
    /// an event that cannot arrive. `None` is the honest answer, and
    /// [`ShardMemory::drain`] turns it into one.
    oldest_flush: Option<DurabilityWatcher>,
}

/// Re-derive a shard's resident-memtable set from its writer state.
///
/// Call under the write lock after any change to that set: `open`,
/// `freeze_memtable`, a flush commit, and `SweepExpired` — which changes it by
/// evicting grace-expired generations that are still counted until it runs.
///
/// Cheap: two `Arc` clones per live memtable, no byte walk — the totals are
/// computed on read.
fn publish_memory(memory: &ArcSwap<ResidentMemTables>, state: &WriterState) {
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
struct WriterState {
    memtable: MemTable,
    last_flushed_wal_entry_position: u64,
    /// Flush watchers for frozen memtables, oldest first. Carries no byte
    /// count: sizes are read live off `frozen_memtables` (see
    /// [`ResidentMemTables`]), so there is nothing here to keep paired.
    frozen_flush_watchers: VecDeque<DurabilityWatcher>,
    /// Sealed memtables, kept queryable so a concurrent reader sees no hole
    /// between `freeze_memtable` and the flush task's manifest commit, and for
    /// `frozen_memtable_grace` beyond it so as-of reads stay batch-resolved.
    /// Pushed in `freeze_memtable`; stamped `flushed_at_ms` by `flush_memtable`
    /// on commit success only (retained un-stamped on failure until a later
    /// flush or WAL replay on reopen); swept after the grace by `SweepExpired`.
    frozen_memtables: VecDeque<FrozenMemTable>,
    /// Flag to prevent duplicate memtable flush requests.
    flush_requested: bool,
    /// Counter for WAL flush threshold crossings.
    wal_flush_trigger_count: usize,
    /// Last time a WAL flush was triggered (for time-based flush).
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
        schema: mt.schema().clone(),
        generation: mt.generation(),
    }
}

fn start_time() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

fn now_millis() -> u64 {
    start_time().elapsed().as_millis() as u64
}

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
struct ReplayResult {
    /// The active memtable — the final, partial one replay left unsealed. A fresh
    /// shard yields an empty one; every sealed memtable was flushed to a Lance
    /// generation during replay and is not returned.
    active: MemTable,
    /// One past the highest WAL entry position observed — the next write position.
    next_wal_position: u64,
}

/// Replay a shard's WAL into memory, flushing sealed memtables as the batch store
/// fills.
///
/// A single memtable holds at most `max_memtable_batches` batches, but a WAL is
/// unbounded — so replay must rotate exactly as the live write path does. It
/// seals a full memtable and, because the data is already durable, flushes it to
/// a Lance generation right here (the same `MemTableFlusher::flush` the live path
/// uses), rather than holding every sealed memtable in memory until open
/// finishes. That bounds resident memory to ~two memtables and truncates the WAL
/// as it goes, so a later reopen replays only the unflushed tail. Only the final
/// partial memtable is returned, as the active one.
///
/// `make_memtable(generation, global_offset)` builds a fresh, cursor-bound
/// memtable. Rotation happens at WAL-entry boundaries, never mid-entry, so each
/// sealed memtable covers a clean range of complete entries and stamps the last
/// one as its SSTable's `replay_after_wal_entry_position`.
#[allow(clippy::too_many_arguments)]
async fn replay_memtable_from_wal(
    object_store: Arc<ObjectStore>,
    base_path: Path,
    shard_id: Uuid,
    our_epoch: u64,
    manifest: &ShardManifest,
    base_generation: u64,
    mut make_memtable: impl FnMut(u64, usize) -> Result<MemTable>,
    flusher: &MemTableFlusher,
    wal_flusher: &WalFlusher,
    index_configs: &[MemIndexConfig],
    max_memtable_size: usize,
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

    let tailer = WalTailer::new(object_store, base_path, shard_id);
    let mut position = start_position;

    let mut active = make_memtable(base_generation, 0)?;

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
                // Fence sentinels deserialize to zero batches and are skipped
                // here — they carry only a position, no rows.
                if !entry.batches.is_empty() {
                    // Re-label to the current storage schema; entries written
                    // before deletes existed also need `_tombstone = false`.
                    let storage_schema = active.schema().clone();
                    let batches = entry
                        .batches
                        .into_iter()
                        .map(|b| ensure_tombstone_column(b, &storage_schema))
                        .collect::<Result<Vec<_>>>()?;

                    // Seal + flush at the entry boundary on the *same* criteria the
                    // live path uses (`maybe_trigger_memtable_flush`): the memtable
                    // is at or over `max_memtable_size` bytes, or this whole entry
                    // won't fit the batch store. The byte trigger is the one that
                    // matters beyond avoiding overflow — it is what keeps a memtable
                    // under `max_memtable_rows`, and therefore keeps the in-memory
                    // HNSW index (sized to `max_memtable_rows`) from exhausting its
                    // capacity when the final active memtable is indexed.
                    //
                    // Rotate at the entry boundary so no entry is split across two
                    // memtables and each sealed one covers a clean range of complete
                    // entries. Never rotate an empty memtable — if a single entry
                    // has more batches than a memtable can hold, a fresh one would
                    // overflow too, the same hard limit the live put path has, left
                    // to the insert below to surface.
                    if !active.batch_store().is_empty()
                        && memtable_reached_flush_threshold(
                            &active,
                            max_memtable_size,
                            max_resident_bytes,
                            batches.len(),
                        )
                    {
                        let store = active.batch_store();
                        // The last entry this memtable fully absorbed is the one
                        // before the entry about to be inserted.
                        let covered = position.saturating_sub(1);
                        let generation = active.generation() + 1;
                        let global_end = store.global_end();

                        // The sealed data is already durable in the WAL — mark it
                        // so the flush's `all_flushed_to_wal` precondition holds and
                        // no WAL re-append is attempted.
                        wal_flusher.advance_durable(global_end);
                        flush_replayed_memtable(
                            flusher,
                            &active,
                            our_epoch,
                            covered,
                            global_end,
                            index_configs,
                        )
                        .await?;

                        active = make_memtable(generation, global_end)?;
                    }

                    active.insert_batches_only(batches).await?;
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

    // Rebuild the active memtable's in-memory indexes from the batches just
    // replayed, so readers see them through the index path — matching what the
    // pre-crash writer's flush would have done. Sealed memtables needed no
    // in-memory index build: they were flushed straight to disk and are gone.
    if let Some(indexes) = active.indexes_arc() {
        let batch_count = active.batch_count();
        if batch_count > 0 {
            let store = active.batch_store();
            let stored: Vec<StoredBatch> = (0..batch_count)
                .filter_map(|pos| store.get(pos).cloned())
                .collect();
            tokio::task::spawn_blocking(move || indexes.insert_batches(&stored))
                .await
                .map_err(|e| {
                    Error::internal(format!("WAL replay index update task panicked: {}", e))
                })??;
        }
    }

    Ok(ReplayResult {
        active,
        next_wal_position: position,
    })
}

/// Whether a memtable has reached the threshold at which it should be sealed and
/// flushed.
///
/// The single source of truth for the flush trigger, shared by the live put path
/// (`maybe_trigger_memtable_flush`, checking post-insert with `incoming_batches =
/// 1` — "is there room for the next batch") and by replay (checking pre-insert
/// with the next WAL entry's batch count). Keeping one predicate is what stops the
/// two from drifting — e.g. someone adding a third criterion to one and not the
/// other, which is the exact class of bug this whole change set is about.
///
/// Three arms, each answering a different question:
///
/// - **Row window** against `max_memtable_size`. The knob an operator sizes: it
///   measures what a flush actually writes, so a generation stays a predictable
///   fragment in the base dataset. Deliberately the *only* thing charged to this
///   threshold — index memory and buffer padding are not, or fragment size would
///   start depending on index configuration and Arrow's allocator.
/// - **Resident total** against `max_resident_bytes`, the backpressure ceiling.
///   The row window bounds neither what the batches *pin* (a one-row slice holds
///   its whole parent) nor what the indexes hold, so without this arm a memtable
///   can carry a shard past its ceiling with no seal reachable — and the valve,
///   finding nothing outstanding to wait on, would refuse writes that can never
///   succeed. This is the drain path that makes the ceiling live rather than a
///   trap.
/// - **Batch-store capacity**, room for `incoming_batches` more.
fn memtable_reached_flush_threshold(
    memtable: &MemTable,
    max_memtable_size: usize,
    max_resident_bytes: usize,
    incoming_batches: usize,
) -> bool {
    let store = memtable.batch_store();
    store.row_bytes() >= max_memtable_size
        || memtable_resident_bytes(memtable) >= max_resident_bytes
        || store.remaining_capacity() < incoming_batches
}

/// What this memtable holds in memory: the heap its batches pin plus its
/// in-memory indexes. The same quantity [`ShardMemory`] reports for the active
/// memtable, so the seal trigger and the ceiling that gates writes measure the
/// same thing.
fn memtable_resident_bytes(memtable: &MemTable) -> usize {
    memtable.batch_store().retained_bytes()
        + memtable.indexes().map_or(0, IndexStore::resident_bytes)
        + super::memtable::pk_bloom_filter_bytes()
}

/// Flush a sealed replay memtable to a Lance generation, choosing the indexed
/// path when secondary indexes are configured (mirroring the live memtable-flush
/// handler). Commits the manifest, stamping `covered` as the generation's
/// `replay_after_wal_entry_position` so a later reopen skips these entries.
async fn flush_replayed_memtable(
    flusher: &MemTableFlusher,
    memtable: &MemTable,
    epoch: u64,
    covered: u64,
    durable: usize,
    index_configs: &[MemIndexConfig],
) -> Result<()> {
    if index_configs.is_empty() {
        flusher.flush(memtable, epoch, covered, durable).await?;
    } else {
        Box::pin(flusher.flush_with_indexes(memtable, epoch, index_configs, covered, durable))
            .await?;
    }
    Ok(())
}

/// Pair each primary-key column name with its field id (both derived from the
/// schema's primary key, in the same order) for [`IndexStore::enable_pk_index`].
fn pk_index_columns(pk_columns: &[String], pk_field_ids: &[i32]) -> Vec<(String, i32)> {
    pk_columns
        .iter()
        .cloned()
        .zip(pk_field_ids.iter().copied())
        .collect()
}

/// Re-label `batch` to the storage schema, injecting `_tombstone = false` when
/// absent — callers pass logical-shaped batches, and WAL entries written before
/// deletes existed lack the column.
///
/// A batch that already carries `_tombstone` is re-labeled too, so an entry
/// written under an older storage schema replays into the current one.
fn ensure_tombstone_column(
    batch: RecordBatch,
    storage_schema: &Arc<ArrowSchema>,
) -> Result<RecordBatch> {
    let n = batch.num_rows();
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    if batch.schema().column_with_name(TOMBSTONE).is_none() {
        columns.push(Arc::new(BooleanArray::from(vec![false; n])));
    }
    RecordBatch::try_new(storage_schema.clone(), columns).map_err(|e| {
        Error::invalid_input(format!(
            "failed to inject _tombstone column (does the batch match the base table schema?): {}",
            e
        ))
    })
}

/// Build a tombstone batch from a key-only `keys` batch: primary keys carried
/// through, `_tombstone` true, every other column null.
///
/// Non-PK columns are nullable in the storage schema however the base table
/// declares them — that is what lets a strict table have tombstones at all.
/// Primary keys are not, so the validation below still rejects a null,
/// mistyped, or missing key.
fn build_tombstone_batch(
    keys: &RecordBatch,
    storage_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
) -> Result<RecordBatch> {
    let n = keys.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(storage_schema.fields().len());
    for field in storage_schema.fields() {
        let name = field.name();
        if name == TOMBSTONE {
            columns.push(Arc::new(BooleanArray::from(vec![true; n])));
        } else if pk_columns.iter().any(|c| c == name) {
            let col = keys.column_by_name(name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "delete keys batch is missing primary key column '{}'",
                    name
                ))
            })?;
            columns.push(col.clone());
        } else {
            columns.push(new_null_array(field.data_type(), n));
        }
    }
    RecordBatch::try_new(storage_schema.clone(), columns).map_err(|e| {
        Error::invalid_input(format!(
            "failed to build tombstone batch (do the delete keys match the primary key?): {}",
            e
        ))
    })
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
    schema: Arc<ArrowSchema>,
    pk_field_ids: Vec<i32>,
    /// Primary-key column names, used to (re)enable the PK-position index on
    /// each fresh active memtable created at freeze.
    pk_columns: Vec<String>,
    max_memtable_batches: usize,
    max_memtable_rows: usize,
    index_configs: Vec<MemIndexConfig>,
}

impl SharedWriterState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        memory: Arc<ArcSwap<ResidentMemTables>>,
        wal_flusher: Arc<WalFlusher>,
        wal_flush_tx: mpsc::UnboundedSender<TriggerWalFlush>,
        index_apply_tx: mpsc::UnboundedSender<TriggerIndexApply>,
        memtable_flush_tx: mpsc::UnboundedSender<TriggerMemTableFlush>,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        pk_field_ids: Vec<i32>,
        pk_columns: Vec<String>,
        max_memtable_batches: usize,
        max_memtable_rows: usize,
        index_configs: Vec<MemIndexConfig>,
    ) -> Self {
        Self {
            memory,
            wal_flusher,
            wal_flush_tx,
            index_apply_tx,
            memtable_flush_tx,
            config,
            schema,
            pk_field_ids,
            pk_columns,
            max_memtable_batches,
            max_memtable_rows,
            index_configs,
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
        let mut new_memtable = MemTable::with_capacity_at(
            self.schema.clone(),
            next_generation,
            self.pk_field_ids.clone(),
            CacheConfig::default(),
            self.max_memtable_batches,
            next_global_offset,
        )?;

        // Always build and bind an IndexStore, even with no user indexes and no
        // primary key. It is what carries the memtable's `indexed_count`, and
        // binding it to the writer's cursors is what lets a reader derive the
        // visible prefix — so an index-less memtable that skipped this would fall
        // back to `visible == indexed` and publish rows before they were durable.
        // (A PK memtable also needs the PK dedup index and its flushed sidecar.)
        let mut indexes = IndexStore::from_configs(
            &self.index_configs,
            self.max_memtable_rows,
            self.max_memtable_batches,
        )?;
        if !self.pk_columns.is_empty() {
            indexes.enable_pk_index(&pk_index_columns(&self.pk_columns, &self.pk_field_ids));
        }
        indexes.set_durability(Arc::clone(self.wal_flusher.cursors()), next_global_offset);
        new_memtable.set_indexes_arc(Arc::new(indexes));

        let mut old_memtable = std::mem::replace(&mut state.memtable, new_memtable);
        old_memtable.freeze(last_wal_entry_position);

        // Set up completion tracking on the outgoing table before it is retained
        // and before any fallible dispatch, so the retained table already carries
        // its cells and a failed send below can poison-and-return without leaving
        // partial state to unwind.
        let _memtable_flush_watcher = old_memtable.create_memtable_flush_completion();

        // The outgoing memtable may still owe an index apply — the puts that
        // filled it triggered one, but the task need not have drained yet, and
        // this is the last chance to name that store. Its L0 flush is gated on
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
        // past its manifest commit too (swept after the grace by `SweepExpired`);
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
    ) -> super::wal::BatchDurableWatcher {
        self.wal_flusher
            .track_batch(indexes, target_indexed, target_durable)
    }

    /// Check if memtable flush is needed and trigger if so.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_memtable_flush(&self, state: &mut WriterState) -> Result<()> {
        if state.flush_requested {
            return Ok(());
        }

        // Checked post-insert: flush if there is no longer room for even one more
        // batch (or the byte threshold is crossed). Same predicate replay uses.
        let should_flush = memtable_reached_flush_threshold(
            &state.memtable,
            self.config.max_memtable_size,
            self.config.max_unflushed_memtable_bytes,
            1,
        );

        if should_flush {
            state.flush_requested = true;
            self.freeze_memtable(state)?;
            state.flush_requested = false;
        }
        Ok(())
    }

    /// Check if WAL flush is needed and trigger if so.
    ///
    /// Takes `&mut WriterState` directly since caller already holds the lock.
    fn maybe_trigger_wal_flush(&self, state: &mut WriterState) {
        let threshold = self.config.max_wal_buffer_size;

        let batch_count = state.memtable.batch_count();
        let total_bytes = state.memtable.batch_store().row_bytes();
        let batch_store = state.memtable.batch_store();

        // Check if there are any unflushed batches
        let has_pending = batch_store.pending_wal_flush_count(self.wal_flusher.durable()) > 0;

        // Check time-based trigger first
        let time_trigger = if let Some(interval) = self.config.max_wal_flush_interval {
            let interval_millis = interval.as_millis() as u64;
            let last_trigger = state.last_wal_flush_trigger_time;
            let now = now_millis();

            // If last_trigger is 0, this is the first write - start the timer but don't flush
            if last_trigger == 0 {
                state.last_wal_flush_trigger_time = now;
                None
            } else {
                let elapsed = now.saturating_sub(last_trigger);

                if elapsed >= interval_millis && has_pending {
                    state.last_wal_flush_trigger_time = now;
                    Some(now)
                } else {
                    None
                }
            }
        } else {
            None
        };

        // If time trigger fired, send a flush message
        if time_trigger.is_some() {
            let _ = self.wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::BatchStore { batch_store },
                end_batch_position: batch_count,
                done: None,
            });
            return;
        }

        // Check size-based trigger
        if threshold == 0 {
            return;
        }

        // Calculate how many thresholds have been crossed (1 at 10MB, 2 at 20MB, etc.)
        let thresholds_crossed = total_bytes / threshold;

        // Trigger flush for each unclaimed threshold crossing
        while state.wal_flush_trigger_count < thresholds_crossed {
            state.wal_flush_trigger_count += 1;
            // Update last trigger time so time-based trigger doesn't fire immediately after
            state.last_wal_flush_trigger_time = now_millis();

            // Trigger WAL flush with captured batch range
            let _ = self.wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::BatchStore {
                    batch_store: batch_store.clone(),
                },
                end_batch_position: batch_count,
                done: None,
            });
        }
    }
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
    /// Last time a WAL flush was triggered (for time-based trigger).
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
    },
}

/// Main writer for a MemWAL shard.
pub struct ShardWriter {
    config: ShardWriterConfig,
    epoch: u64,
    wal_flusher: Arc<WalFlusher>,
    task_executor: Arc<TaskExecutor>,
    manifest_store: Arc<ShardManifestStore>,
    stats: SharedWriteStats,
    mode: WriterMode,
    /// The base table's schema as the caller passed it — no `_tombstone`,
    /// nullability untouched. Caller input is held to it (see
    /// [`Self::validate_against_logical_schema`]) and the scan narrows back to
    /// it; the memtable, WAL, and SSTables carry the widened storage schema
    /// ([`relax_non_pk_nullability`]) instead.
    logical_schema: Arc<ArrowSchema>,
}

impl ShardWriter {
    /// Open or create a ShardWriter.
    ///
    /// The `base_path` should come from `ObjectStore::from_uri()` to ensure
    /// WAL files are written inside the dataset directory.
    #[instrument(name = "sw_open", level = "info", skip_all, fields(shard_id = %config.shard_id, index_count = index_configs.len()))]
    pub async fn open(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        config: ShardWriterConfig,
        schema: Arc<ArrowSchema>,
        index_configs: Vec<MemIndexConfig>,
    ) -> Result<Self> {
        if !config.enable_memtable && !index_configs.is_empty() {
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

        // The caller's schema is the shard's logical schema; the storage schema
        // is derived below, once the primary key is known. lance owns
        // `_tombstone` and appends it here — idempotent across reopens.
        let logical_schema = schema;
        let tombstoned = schema_with_tombstone(&logical_schema);

        let base_uri = base_uri.into();
        let shard_id = config.shard_id;
        let manifest_store = Arc::new(ShardManifestStore::new(
            object_store.clone(),
            &base_path,
            shard_id,
            config.manifest_scan_batch_size,
        ));

        // Derive PK metadata and run every side-effect-free validation *before*
        // claiming the epoch. `claim_epoch` durably bumps the stored epoch and,
        // for a successor, `write_fence_sentinel` fences the predecessor — so an
        // open doomed by purely local input (an index config that disagrees with
        // the schema) must fail here, before it can knock the healthy incumbent off
        // the shard. Memtable-only: WAL-only mode has no indexes to validate.
        let memtable_validation = if config.enable_memtable {
            let lance_schema = Schema::try_from(tombstoned.as_ref())?;
            let pk_fields = lance_schema.unenforced_primary_key();
            let pk_field_ids: Vec<i32> = pk_fields.iter().map(|f| f.id).collect();
            let pk_columns: Vec<String> = pk_fields.iter().map(|f| f.name.clone()).collect();

            // Reject an index config that disagrees with the schema *before* a
            // single row is accepted. Such a config fails deterministically on
            // every insert, including inserts replayed from the WAL — so once a row
            // is durable the shard can never reopen. Fail the open instead.
            validate_index_configs(
                &index_configs,
                tombstoned.as_ref(),
                &lance_schema,
                &pk_columns,
            )?;

            // An HNSW graph reserves its whole capacity before the first insert,
            // but the seal trigger only measures row bytes. A reservation with no
            // room left under the ceiling puts the shard over budget at zero rows
            // — nothing to seal, so nothing to flush, so every put stalls and then
            // fails as `Error::Backpressure`, which is supposed to mean "retry
            // later". Reject the config instead; only the built-in valve reads
            // this ceiling.
            if config.backpressure.is_none() {
                // The headroom the check below reserves for rows. At zero it
                // reserves nothing, so a fresh memtable's index reservation may
                // *equal* the ceiling — over budget before its first row, with an
                // empty memtable there is nothing to seal, and the writer refuses
                // its first write forever. A zero threshold is degenerate anyway:
                // it seals every memtable at every insert.
                if config.max_memtable_size == 0 {
                    return Err(Error::invalid_input(
                        "max_memtable_size must be greater than zero: it is both the \
                         seal threshold for row data and the headroom reserved for rows \
                         under max_unflushed_memtable_bytes, and at zero a writer with \
                         in-memory indexes can be at its ceiling before its first row",
                    ));
                }

                // Built the way `make_bound_memtable` builds it below, so this is
                // the figure the controller will actually read.
                let mut indexes = IndexStore::from_configs(
                    &index_configs,
                    config.max_memtable_rows,
                    config.max_memtable_batches,
                )?;
                if !pk_columns.is_empty() {
                    indexes.enable_pk_index(&pk_index_columns(&pk_columns, &pk_field_ids));
                }
                let reserved = indexes.resident_bytes() + super::memtable::pk_bloom_filter_bytes();
                // Room for a full memtable of rows on top, or the ceiling is
                // crossed before `max_memtable_size` can seal.
                let needed = reserved.saturating_add(config.max_memtable_size);
                if needed > config.max_unflushed_memtable_bytes {
                    return Err(Error::invalid_input(format!(
                        "in-memory indexes reserve {reserved} bytes at \
                         max_memtable_rows={}, and max_memtable_size={} must fit alongside them, \
                         needing {needed} bytes; max_unflushed_memtable_bytes={} is below that, \
                         so the active memtable would cross the backpressure ceiling before \
                         accruing enough row bytes to seal, stalling every write. Raise \
                         max_unflushed_memtable_bytes to at least {needed}, or lower \
                         max_memtable_rows / max_memtable_size",
                        config.max_memtable_rows,
                        config.max_memtable_size,
                        config.max_unflushed_memtable_bytes,
                    )));
                }
            }

            // Widen only now that the primary key is known — a tombstone nulls
            // every non-PK column, and PK detection needs the strict schema.
            let storage_schema = relax_non_pk_nullability(&tombstoned, &pk_columns);
            Some((pk_field_ids, pk_columns, storage_schema))
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

        let mode = if config.enable_memtable {
            let (pk_field_ids, pk_columns, storage_schema) = memtable_validation
                .expect("memtable_validation is Some when enable_memtable is true");
            Self::open_memtable_mode(
                &config,
                &storage_schema,
                &manifest,
                &index_configs,
                pk_field_ids,
                pk_columns,
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
            )
            .await?
        } else {
            Self::open_wal_only_mode(
                &config,
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
            logical_schema,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn open_memtable_mode(
        config: &ShardWriterConfig,
        schema: &Arc<ArrowSchema>,
        manifest: &ShardManifest,
        index_configs: &[MemIndexConfig],
        pk_field_ids: Vec<i32>,
        pk_columns: Vec<String>,
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

        // Build a fresh, cursor-bound memtable at a given generation and
        // writer-global coordinate. Replay calls this for the first memtable and
        // after every rotation. Always builds and binds an `IndexStore`, even
        // with no user indexes and no primary key — see the note in
        // `freeze_memtable` for why an index-less memtable still needs one.
        let make_bound_memtable = |generation: u64, global_offset: usize| -> Result<MemTable> {
            let mut memtable = MemTable::with_capacity_at(
                schema.clone(),
                generation,
                pk_field_ids.clone(),
                CacheConfig::default(),
                config.max_memtable_batches,
                global_offset,
            )?;
            let mut indexes = IndexStore::from_configs(
                index_configs,
                config.max_memtable_rows,
                config.max_memtable_batches,
            )?;
            if !pk_columns.is_empty() {
                indexes.enable_pk_index(&pk_index_columns(&pk_columns, &pk_field_ids));
            }
            indexes.set_durability(Arc::clone(wal_flusher.cursors()), global_offset);
            memtable.set_indexes_arc(Arc::new(indexes));
            Ok(memtable)
        };

        // The flusher writes sealed memtables to Lance generations — both the
        // ones replay seals below and the ones the live path freezes later.
        let flusher = Arc::new(
            MemTableFlusher::new(
                object_store.clone(),
                base_path.clone(),
                base_uri.clone(),
                shard_id,
                manifest_store.clone(),
            )
            .with_warmer(config.warmer.clone())
            .with_storage_context(config.store_params.clone(), config.session.clone()),
        );

        // Replay any WAL entries written after the last successfully-flushed
        // SSTable, flushing sealed memtables to Lance SSTables as the batch
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
            shard_id,
            epoch,
            manifest,
            manifest.current_generation,
            make_bound_memtable,
            &flusher,
            &wal_flusher,
            index_configs,
            config.max_memtable_size,
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
        // Without this the durability cursor stays at the last sealed generation,
        // so the next WAL flush re-covers the active tail: it re-appends the
        // already-durable rows *and* re-inserts every replayed row into the
        // indexes. None of the three in-memory indexes is idempotent (HNSW mints
        // fresh node ids for the same row, FTS increments doc_count/df rather than
        // recomputing them, BTree is a multiset), so a full scan keeps looking
        // healthy while every index-accelerated query silently returns duplicates
        // — and it compounds, because the WAL now holds those rows twice.
        //
        // `global_end()` is the writer-global batch count through this memtable,
        // since its coordinate continues where the last sealed generation ended.
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
            last_flushed_wal_entry_position: initial_covered_wal_entry_position,
            frozen_flush_watchers: VecDeque::new(),
            frozen_memtables: VecDeque::new(),
            flush_requested: false,
            wal_flush_trigger_count: 0,
            last_wal_flush_trigger_time: 0,
        };
        // Seed before the first freeze: replay above may already have filled the
        // memtable, and nothing else publishes until it seals.
        publish_memory(&memory, &state);
        let state = Arc::new(RwLock::new(state));

        let (memtable_flush_tx, memtable_flush_rx) = mpsc::unbounded_channel();

        let flusher = Arc::new(
            MemTableFlusher::new(object_store, base_path, base_uri, shard_id, manifest_store)
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
        let memtable_handler = MemTableFlushHandler::new(
            state.clone(),
            memory.clone(),
            flusher,
            wal_flusher.clone(),
            epoch,
            index_configs.to_vec(),
            stats.clone(),
            config.observer.clone(),
            config.frozen_memtable_grace,
        );
        task_executor.add_handler(
            "memtable_flusher".to_string(),
            Box::new(memtable_handler),
            memtable_flush_rx,
        )?;

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
            schema.clone(),
            pk_field_ids,
            pk_columns,
            config.max_memtable_batches,
            config.max_memtable_rows,
            index_configs.to_vec(),
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
        self.validate_against_logical_schema(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                // Callers pass logical-shaped batches and never name
                // `_tombstone`.
                let batches = batches
                    .into_iter()
                    .map(|b| ensure_tombstone_column(b, &writer_state.schema))
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable(batches, state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger,
                backpressure,
            } => {
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
                if writer_state.pk_columns.is_empty() {
                    return Err(Error::invalid_input(
                        "delete requires a primary key, but this shard has no primary key columns",
                    ));
                }
                let tombstones = keys
                    .into_iter()
                    .map(|k| {
                        build_tombstone_batch(&k, &writer_state.schema, &writer_state.pk_columns)
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable_no_wait(tombstones, state, writer_state, backpressure)
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
        self.validate_against_logical_schema(&batches)?;

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                backpressure,
            } => {
                // Mirrors `put`.
                let batches = batches
                    .into_iter()
                    .map(|b| ensure_tombstone_column(b, &writer_state.schema))
                    .collect::<Result<Vec<_>>>()?;
                self.put_memtable_no_wait(batches, state, writer_state, backpressure)
                    .await
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "put_no_wait is only supported in MemTable mode",
            )),
        }
    }

    /// Reject caller input that violates the logical schema: wrong column names,
    /// order, count, or types, or a null where the base table declares
    /// non-nullable.
    ///
    /// The *only* gate on that contract — the storage schema accepts the null,
    /// append and `merge_insert` compare with `NullabilityComparison::Ignore`,
    /// and the encoder takes validity from the array, not the field — so a null
    /// that gets past here reaches the base table silently.
    ///
    /// Runs before the WAL append: a batch rejected only afterwards would fail
    /// identically on every replay, leaving the shard unable to reopen.
    fn validate_against_logical_schema(&self, batches: &[RecordBatch]) -> Result<()> {
        for (i, batch) in batches.iter().enumerate() {
            // Everything downstream matches columns by position, so a swapped
            // pair of same-typed columns would be stored under each other's
            // names unless caught here.
            for (col, (expected, actual)) in self
                .logical_schema
                .fields()
                .iter()
                .zip(batch.schema().fields())
                .enumerate()
            {
                if expected.name() != actual.name() {
                    return Err(Error::invalid_input(format!(
                        "batch {i} column {col} is named '{}', but the base table schema \
                         declares '{}' at that position",
                        actual.name(),
                        expected.name()
                    )));
                }
            }
            RecordBatch::try_new(self.logical_schema.clone(), batch.columns().to_vec()).map_err(
                |e| {
                    Error::invalid_input(format!(
                        "batch {i} does not match the base table schema: {e}"
                    ))
                },
            )?;
        }
        Ok(())
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
        batches: Vec<RecordBatch>,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &Arc<dyn BackpressureController>,
    ) -> Result<WriteResult> {
        let (result, watcher) = self
            .put_memtable_no_wait(batches, state_lock, writer_state, backpressure)
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
        batches: Vec<RecordBatch>,
        state_lock: &Arc<RwLock<WriterState>>,
        writer_state: &Arc<SharedWriterState>,
        backpressure: &Arc<dyn BackpressureController>,
    ) -> Result<(WriteResult, Option<BatchDurableWatcher>)> {
        // Reject writes on a fenced writer before mutating the memtable, so a
        // poisoned writer can't drift further from the durable WAL.
        self.wal_flusher.check_poisoned()?;

        // The seal check runs inside the lock immediately after an insert, but the
        // index apply that follows it runs *outside* — and replay hands back a
        // memtable whose indexes were built after its last check too. Either way
        // the ceiling can be reached with nothing sealed, and the valve below
        // would then refuse a write that a seal would have admitted. Re-run the
        // check here so the wait has a flush to end on.
        //
        // The read is two relaxed loads; the lock is taken only when the shard is
        // actually at its ceiling, which is the path already about to block.
        if ShardMemory::memtables(writer_state.memory.clone()).unflushed_bytes()
            >= self.config.max_unflushed_memtable_bytes
        {
            let mut state = state_lock.write().await;
            // Nothing to seal in an empty memtable, and freezing one would spin:
            // an injected controller skips the open-time reservation check, so a
            // fresh memtable can sit above this ceiling on its indexes alone.
            if state.memtable.batch_count() > 0 {
                writer_state.maybe_trigger_memtable_flush(&mut state)?;
            }
        }

        // Apply backpressure if needed (before acquiring main lock)
        backpressure
            .maybe_apply_backpressure(ShardMemory::memtables(writer_state.memory.clone()))
            .await?;

        let start = std::time::Instant::now();

        // Acquire write lock for entire operation (atomic approach)
        let (batch_positions, durable_watcher, batch_store, indexes) = {
            let mut state = state_lock.write().await;

            // 1. Insert all batches into memtable atomically
            let results = state.memtable.insert_batches_only(batches).await?;

            // 2. Capture the store the batches actually landed in, *before* step
            //    6 below can freeze and swap the active memtable. Reading it
            //    afterwards hands the flush trigger the **new** store paired with
            //    the **old** store's end position, so the new store's watermark
            //    jumps past batches that were never appended.
            let batch_store = state.memtable.batch_store();
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

            // 6. Check if memtable flush is needed (may freeze and rotate)
            if let Err(e) = writer_state.maybe_trigger_memtable_flush(&mut state) {
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

        // The WAL append is *not* triggered here. It happens on the background
        // ticker (and on the size trigger, and at freeze/close), which is the only
        // way the flush interval can mean anything: while every durable put
        // triggered its own append, the interval could add a redundant trigger but
        // never delay or batch one.
        //
        // The cost is real and accepted: a single client's sequential *durable*
        // throughput drops from ~10 writes/sec (one PUT round-trip) to roughly one
        // per tick. That is a policy choice — the interval should mean what it
        // says, and S3 API cost should be bounded. Latency-sensitive callers want
        // `durable_write: false`, which now costs them durability only, not
        // visibility.

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

        // Time- and size-based triggers on the write path, for durable and
        // non-durable puts alike — mirroring MemTable mode's
        // `maybe_trigger_wal_flush`. The background ticker drives the append
        // too; whichever fires first wins, and a redundant trigger is a cheap
        // no-op because the flush snapshot/commit is idempotent.
        self.maybe_trigger_wal_flush_wal_only(
            state,
            wal_flush_tx,
            trigger,
            batch_positions.end,
            state.queue_bytes(),
        );

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

    /// WAL-only-mode size+time trigger. Mirrors `SharedWriterState::maybe_trigger_wal_flush`
    /// but reads its inputs from `WalOnlyState` (pending queue) instead of
    /// the active MemTable.
    fn maybe_trigger_wal_flush_wal_only(
        &self,
        state: &Arc<WalOnlyState>,
        wal_flush_tx: &mpsc::UnboundedSender<TriggerWalFlush>,
        trigger: &StdRwLock<WalOnlyTriggerState>,
        end_batch_position: usize,
        pending_bytes: usize,
    ) {
        let threshold = self.config.max_wal_buffer_size;
        let has_pending = state.batch_count() > 0;

        let mut t = trigger.write().unwrap();

        // Time-based trigger.
        if let Some(interval) = self.config.max_wal_flush_interval {
            let interval_millis = interval.as_millis() as u64;
            let now = now_millis();
            if t.last_wal_flush_trigger_time == 0 {
                t.last_wal_flush_trigger_time = now;
            } else {
                let elapsed = now.saturating_sub(t.last_wal_flush_trigger_time);
                if elapsed >= interval_millis && has_pending {
                    t.last_wal_flush_trigger_time = now;
                    let _ = wal_flush_tx.send(TriggerWalFlush {
                        source: WalFlushSource::WalOnly {
                            state: state.clone(),
                        },
                        end_batch_position,
                        done: None,
                    });
                    return;
                }
            }
        }

        if threshold == 0 {
            return;
        }

        // Size-based trigger: fire one trigger per `max_wal_buffer_size`
        // crossed since the last time we triggered. If the pending queue
        // shrank below the recorded baseline (a drain happened), reset the
        // baseline first so the next crossing fires correctly.
        if pending_bytes < t.last_trigger_pending_bytes {
            t.last_trigger_pending_bytes = 0;
        }
        while pending_bytes >= t.last_trigger_pending_bytes + threshold {
            t.last_trigger_pending_bytes += threshold;
            t.last_wal_flush_trigger_time = now_millis();
            let _ = wal_flush_tx.send(TriggerWalFlush {
                source: WalFlushSource::WalOnly {
                    state: state.clone(),
                },
                end_batch_position,
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

    /// Seal the active memtable so it's queued for L0 flush. Errors in
    /// WAL-only mode or if this writer has been fenced by a successor.
    ///
    /// The returned [`SealFence`] is what makes a *bounded* wait possible:
    /// a caller that needs "everything written before my seal is in L0"
    /// awaits [`SealFence::wait`], which covers the flushes outstanding at
    /// seal time and nothing else. [`Self::wait_for_flush_drain`] instead
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
    #[instrument(name = "sw_force_seal_active", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn force_seal_active(&self) -> Result<SealFence> {
        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                self.check_fenced().await?;
                self.wal_flusher.check_poisoned()?;
                let mut state = state.write().await;
                let sealed_generation = if state.memtable.batch_count() == 0 {
                    None
                } else {
                    let generation = state.memtable.generation();
                    writer_state.freeze_memtable(&mut state)?;
                    Some(generation)
                };
                // Capture the outstanding set under the same lock that froze,
                // so no freeze can slip between the two and widen the fence.
                // It covers more than the generation just sealed: a
                // size/interval trigger freezes generations asynchronously, so
                // an empty active memtable does *not* mean every pre-call write
                // already reached L0. Watchers are popped as flushes settle, so
                // whatever remains here is exactly what is still owed.
                Ok(SealFence {
                    sealed_generation,
                    watchers: state.frozen_flush_watchers.iter().cloned().collect(),
                })
            }
            WriterMode::WalOnly { .. } => Err(Error::invalid_input(
                "force_seal_active not available in WAL-only mode (no MemTable)",
            )),
        }
    }

    /// Block until every frozen memtable in the L0 flush queue has
    /// landed and been recorded in the manifest. Does not wait on the
    /// active memtable — call [`Self::force_seal_active`] first if you
    /// want everything-on-disk. Errors in WAL-only mode, or if any
    /// awaited flush reports `DurabilityResult::Failed`.
    ///
    /// Useful in tests for deterministic post-flush assertions. In
    /// production prefer [`SealFence::wait`] from the seal itself: this
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
                st.frozen_flush_watchers.iter().cloned().collect()
            };
            if watchers.is_empty() {
                return Ok(());
            }
            for mut w in watchers {
                match w.await_value().await {
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
    /// after abort returns. Idempotent: a second call re-cancels an
    /// already-cancelled token and joins an already-emptied task set.
    #[instrument(name = "sw_abort", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn abort(&self) -> Result<()> {
        info!(
            "Aborting ShardWriter for shard {} (no flush)",
            self.config.shard_id
        );
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

    /// Close the writer gracefully.
    ///
    /// Flushes pending data and shuts down background tasks.
    ///
    /// # Errors
    ///
    /// Returns an error if pending WAL data cannot be persisted, an active or
    /// frozen MemTable cannot be flushed, a flush handler exits before reporting
    /// completion, or background tasks cannot be shut down.
    #[instrument(name = "sw_close", level = "info", skip_all, fields(shard_id = %self.config.shard_id, epoch = self.epoch))]
    pub async fn close(self) -> Result<()> {
        info!("Closing ShardWriter for shard {}", self.config.shard_id);
        let mut close_result: Result<()> = Ok(());

        match &self.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                // Drain *both* tasks against the active memtable. The index apply
                // and the WAL append are independent now, so closing has to
                // settle both: the L0 flush below turns this memtable into a
                // Lance generation, and a generation whose indexes never saw the
                // tail is a generation with a hole in it.
                let st = state.read().await;
                let batch_store = st.memtable.batch_store();
                let indexes = st.memtable.indexes_arc();
                let batch_count = st.memtable.batch_count();
                drop(st);

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
            WriterMode::WalOnly {
                state,
                wal_flush_tx,
                trigger: _,
                backpressure: _,
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

        // Shutdown background tasks
        let shutdown_result = self.task_executor.shutdown_all().await;
        let close_result = Self::merge_close_stage(close_result, "task shutdown", shutdown_result);

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

/// MemTable statistics: rows, generation, and what the memtable still owes the
/// WAL.
///
/// Deliberately carries **no byte totals**. [`ShardWriter::memory`] is the one
/// way to ask what a shard is holding — it answers without the writer lock, and
/// a second set of byte fields here would be a second implementation of the
/// same filter and sum, free to disagree with the gate.
#[derive(Debug, Clone)]
pub struct MemTableStats {
    pub row_count: usize,
    pub batch_count: usize,
    pub generation: u64,
    pub max_buffered_batch_position: Option<usize>,
    /// Writer-global count of WAL-durable batches. Exclusive: 0 means none.
    /// Compare against `global_offset + batch_count` to see what this memtable
    /// still owes the WAL.
    pub durable_batch_count: usize,
    /// Writer-global coordinate of this memtable's batch 0.
    pub global_offset: usize,
    pub pending_wal_start_batch_position: Option<usize>,
    pub pending_wal_end_batch_position: Option<usize>,
    pub pending_wal_batch_count: usize,
    pub pending_wal_row_count: usize,
    pub pending_wal_estimated_bytes: usize,
    /// Frozen memtables in the read view: sealed-awaiting-flush, plus flushed
    /// ones still inside `frozen_memtable_grace`.
    pub frozen_count: usize,
}

/// WAL statistics.
#[derive(Debug, Clone)]
pub struct WalStats {
    /// Next WAL entry position to be used.
    pub next_wal_entry_position: u64,
}

/// The L0 flushes outstanding when [`ShardWriter::force_seal_active`]
/// returned — the exact predicate for "everything written before that call
/// is in L0".
///
/// The set is fixed at seal time, so [`Self::wait`] is bounded no matter how
/// many memtables freeze while it waits, and each entry reports the outcome of
/// its own flush. That is why the fence is a captured watcher set and not a
/// generation number compared against `ShardManifest::current_generation`: the
/// manifest advances to `generation + 1` on every committed flush without
/// checking for a gap, so a later generation's success moves it past a
/// generation that failed and never reached L0 — the watermark would report
/// durability that does not exist.
#[derive(Debug)]
pub struct SealFence {
    sealed_generation: Option<u64>,
    watchers: Vec<DurabilityWatcher>,
}

impl SealFence {
    /// The generation the seal froze, or `None` when the active memtable
    /// was empty (a no-op seal). Independent of what the fence covers: an
    /// empty active memtable says nothing about generations frozen earlier
    /// and still awaiting flush, which the fence still waits on.
    pub fn sealed_generation(&self) -> Option<u64> {
        self.sealed_generation
    }

    /// Block until every flush this fence covers has landed in L0.
    ///
    /// Errors if any of them reports `DurabilityResult::Failed`, or if the
    /// flush handler exited without reporting.
    pub async fn wait(self) -> Result<()> {
        for mut watcher in self.watchers {
            match watcher.await_value().await {
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
struct IndexApplyHandler {
    cursors: Arc<WriterCursors>,
    wal_flusher: Arc<WalFlusher>,
    stats: SharedWriteStats,
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

struct WalFlushHandler {
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
    fn new(
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
    /// cannot be swept: its L0 flush is blocked on the completion cell that only
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

/// Background handler for MemTable flush operations.
///
/// This handler receives frozen memtables directly via messages and flushes them to Lance storage.
/// Freezing is done by the writer (via SharedWriterState::freeze_memtable) to ensure
/// immediate memtable switching, so writes can continue on the new memtable while this
/// handler flushes in the background.
struct MemTableFlushHandler {
    state: Arc<RwLock<WriterState>>,
    /// Shared with `SharedWriterState`; this handler re-derives it once a
    /// flush commits and the memtable set changes.
    memory: Arc<ArcSwap<ResidentMemTables>>,
    flusher: Arc<MemTableFlusher>,
    /// Source of the writer-global durability cursor, which the L0 flush asserts
    /// covers the whole frozen memtable before it writes a generation.
    wal_flusher: Arc<WalFlusher>,
    epoch: u64,
    /// Secondary index configs to rebuild on each SSTable. When
    /// non-empty the handler flushes via [`MemTableFlusher::flush_with_indexes`]
    /// so queries over SSTables use index lookups instead of full
    /// scans — and so vector search's index-only `fast_search` can see the data
    /// at all.
    index_configs: Vec<MemIndexConfig>,
    stats: SharedWriteStats,
    observer: Option<Arc<dyn WalObserver>>,
    /// How long a frozen memtable lingers in memory after its flush commits
    /// before `SweepExpired` evicts it. See `ShardWriterConfig::frozen_memtable_grace`.
    grace: Duration,
}

impl MemTableFlushHandler {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state: Arc<RwLock<WriterState>>,
        memory: Arc<ArcSwap<ResidentMemTables>>,
        flusher: Arc<MemTableFlusher>,
        wal_flusher: Arc<WalFlusher>,
        epoch: u64,
        index_configs: Vec<MemIndexConfig>,
        stats: SharedWriteStats,
        observer: Option<Arc<dyn WalObserver>>,
        grace: Duration,
    ) -> Self {
        Self {
            state,
            memory,
            flusher,
            wal_flusher,
            epoch,
            index_configs,
            stats,
            observer,
            grace,
        }
    }

    /// Evict frozen memtables whose post-flush grace has elapsed. Un-stamped
    /// (not-yet-flushed) entries are always kept.
    async fn sweep_expired_frozen(&self) {
        let now = now_millis();
        let grace_ms = self.grace.as_millis() as u64;
        let mut state = self.state.write().await;
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
            publish_memory(&self.memory, &state);
        }
    }
}

#[async_trait]
impl MessageHandler<TriggerMemTableFlush> for MemTableFlushHandler {
    fn tickers(&mut self) -> Vec<(Duration, MessageFactory<TriggerMemTableFlush>)> {
        // Zero grace evicts on commit, so no sweeper is needed.
        if self.grace.is_zero() {
            return vec![];
        }
        // Sweep often enough that eviction lags the grace by at most ~1/3, so a
        // generation lives no more than ~grace * 4/3 past its flush commit.
        let tick = (self.grace / 3).max(Duration::from_millis(100));
        vec![(tick, Box::new(|| TriggerMemTableFlush::SweepExpired))]
    }

    async fn handle(&mut self, message: TriggerMemTableFlush) -> Result<()> {
        match message {
            TriggerMemTableFlush::Flush { memtable, done } => {
                let result = self.flush_memtable(memtable).await;
                if let Some(tx) = done {
                    // Send result through the channel - caller is waiting for it
                    let _ = tx.send(result);
                } else {
                    // No done channel, propagate errors
                    result?;
                }
            }
            TriggerMemTableFlush::SweepExpired => self.sweep_expired_frozen().await,
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
    /// Whether the flush succeeds or fails, the memtable's flush-completion
    /// watcher is always signaled and the backpressure queue is always drained
    /// for this memtable. Otherwise `wait_for_flush_drain` would observe a
    /// dropped watch channel and return `Err` instead of the actual outcome.
    #[instrument(name = "mt_flush", level = "info", skip_all, fields(generation = memtable.generation(), row_count = memtable.row_count()))]
    async fn flush_memtable(
        &mut self,
        memtable: Arc<MemTable>,
    ) -> Result<super::memtable::flush::FlushResult> {
        let start = Instant::now();

        let flush_result = async {
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

            // Step 2: Flush the memtable to Lance storage. The covered WAL
            // entry position is either the one we just appended (per-memtable,
            // from the completion cell — authoritative even when concurrent
            // flushes have raced ahead in `state.last_flushed_wal_entry_position`)
            // or, when no flush was triggered at freeze time, the memtable's
            // frozen-at marker captured at freeze. Stamping this into the
            // manifest is what lets replay-on-reopen skip entries this
            // generation covers.
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

            if self.index_configs.is_empty() {
                self.flusher
                    .flush(&memtable, self.epoch, covered_wal_entry_position, durable)
                    .await
            } else {
                Box::pin(self.flusher.flush_with_indexes(
                    &memtable,
                    self.epoch,
                    &self.index_configs,
                    covered_wal_entry_position,
                    durable,
                ))
                .await
            }
        }
        .await;

        // Step 3: Always signal completion (with the outcome) and drain
        // backpressure state for this memtable, even on failure.
        let durability = match &flush_result {
            Ok(_) => DurabilityResult::ok(),
            Err(e) => DurabilityResult::err(e.to_string()),
        };
        memtable.signal_memtable_flush_complete(durability);

        {
            let mut state = self.state.write().await;
            // Backpressure drain: unconditional so `wait_for_flush_drain`
            // sees the watcher's error signal, not a dropped channel. Which
            // entry comes off the front does not matter — flushes complete out
            // of order, and the snapshot re-derived at the end of this block
            // reads the frozen queue itself rather than tracking a charge.
            state.frozen_flush_watchers.pop_front();
            // Retire the frozen handle on commit success, keyed by generation
            // (non-FIFO completion is fine). Zero grace evicts here; otherwise
            // stamp the grace clock so it lingers for multi-part as-of reads
            // until `SweepExpired`. On failure leave it un-stamped: rows stay in
            // the read union until a later flush or WAL replay, else a transient
            // error reopens the hole.
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
        }

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

// ============================================================================
// Write Statistics
// ============================================================================

/// Write performance statistics.
///
/// All fields use atomic operations for thread-safe updates.
/// Use `snapshot()` to get a consistent view of all stats.
#[derive(Debug, Default)]
pub struct WriteStats {
    // Put operation stats
    put_count: AtomicU64,
    put_time_nanos: AtomicU64,

    // WAL flush stats (total time = max(wal_io, index_update) due to parallel execution)
    wal_flush_count: AtomicU64,
    wal_flush_time_nanos: AtomicU64,
    wal_flush_bytes: AtomicU64,

    // WAL flush sub-component stats (for diagnosing bottlenecks)
    wal_io_time_nanos: AtomicU64,
    wal_io_count: AtomicU64,
    index_update_time_nanos: AtomicU64,
    index_update_count: AtomicU64,
    index_update_rows: AtomicU64,

    // MemTable flush stats
    memtable_flush_count: AtomicU64,
    memtable_flush_time_nanos: AtomicU64,
    memtable_flush_rows: AtomicU64,
}

/// Snapshot of write statistics at a point in time.
#[derive(Debug, Clone)]
pub struct WriteStatsSnapshot {
    pub put_count: u64,
    pub put_time: Duration,

    pub wal_flush_count: u64,
    pub wal_flush_time: Duration,
    pub wal_flush_bytes: u64,

    // WAL flush sub-component stats
    pub wal_io_time: Duration,
    pub wal_io_count: u64,
    pub index_update_time: Duration,
    pub index_update_count: u64,
    pub index_update_rows: u64,

    pub memtable_flush_count: u64,
    pub memtable_flush_time: Duration,
    pub memtable_flush_rows: u64,
}

impl WriteStats {
    /// Create a new stats collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a put operation.
    pub fn record_put(&self, duration: Duration) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record a WAL flush operation (total time including parallel I/O and index).
    pub fn record_wal_flush(&self, duration: Duration, bytes: usize) {
        self.wal_flush_count.fetch_add(1, Ordering::Relaxed);
        self.wal_flush_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.wal_flush_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record WAL I/O duration (sub-component of WAL flush).
    pub fn record_wal_io(&self, duration: Duration) {
        self.wal_io_count.fetch_add(1, Ordering::Relaxed);
        self.wal_io_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record index update duration (sub-component of WAL flush).
    pub fn record_index_update(&self, duration: Duration, rows: usize) {
        self.index_update_count.fetch_add(1, Ordering::Relaxed);
        self.index_update_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.index_update_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Record a MemTable flush operation.
    pub fn record_memtable_flush(&self, duration: Duration, rows: usize) {
        self.memtable_flush_count.fetch_add(1, Ordering::Relaxed);
        self.memtable_flush_time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.memtable_flush_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Get a snapshot of current statistics.
    pub fn snapshot(&self) -> WriteStatsSnapshot {
        WriteStatsSnapshot {
            put_count: self.put_count.load(Ordering::Relaxed),
            put_time: Duration::from_nanos(self.put_time_nanos.load(Ordering::Relaxed)),

            wal_flush_count: self.wal_flush_count.load(Ordering::Relaxed),
            wal_flush_time: Duration::from_nanos(self.wal_flush_time_nanos.load(Ordering::Relaxed)),
            wal_flush_bytes: self.wal_flush_bytes.load(Ordering::Relaxed),

            wal_io_time: Duration::from_nanos(self.wal_io_time_nanos.load(Ordering::Relaxed)),
            wal_io_count: self.wal_io_count.load(Ordering::Relaxed),
            index_update_time: Duration::from_nanos(
                self.index_update_time_nanos.load(Ordering::Relaxed),
            ),
            index_update_count: self.index_update_count.load(Ordering::Relaxed),
            index_update_rows: self.index_update_rows.load(Ordering::Relaxed),

            memtable_flush_count: self.memtable_flush_count.load(Ordering::Relaxed),
            memtable_flush_time: Duration::from_nanos(
                self.memtable_flush_time_nanos.load(Ordering::Relaxed),
            ),
            memtable_flush_rows: self.memtable_flush_rows.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.put_count.store(0, Ordering::Relaxed);
        self.put_time_nanos.store(0, Ordering::Relaxed);

        self.wal_flush_count.store(0, Ordering::Relaxed);
        self.wal_flush_time_nanos.store(0, Ordering::Relaxed);
        self.wal_flush_bytes.store(0, Ordering::Relaxed);

        self.wal_io_time_nanos.store(0, Ordering::Relaxed);
        self.wal_io_count.store(0, Ordering::Relaxed);
        self.index_update_time_nanos.store(0, Ordering::Relaxed);
        self.index_update_count.store(0, Ordering::Relaxed);
        self.index_update_rows.store(0, Ordering::Relaxed);

        self.memtable_flush_count.store(0, Ordering::Relaxed);
        self.memtable_flush_time_nanos.store(0, Ordering::Relaxed);
        self.memtable_flush_rows.store(0, Ordering::Relaxed);
    }
}

impl WriteStatsSnapshot {
    /// Get average put latency.
    pub fn avg_put_latency(&self) -> Option<Duration> {
        if self.put_count > 0 {
            Some(self.put_time / self.put_count as u32)
        } else {
            None
        }
    }

    /// Get put throughput (puts per second based on time spent in puts).
    pub fn put_throughput(&self) -> f64 {
        if self.put_time.as_secs_f64() > 0.0 {
            self.put_count as f64 / self.put_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average WAL flush latency.
    pub fn avg_wal_flush_latency(&self) -> Option<Duration> {
        if self.wal_flush_count > 0 {
            Some(self.wal_flush_time / self.wal_flush_count as u32)
        } else {
            None
        }
    }

    /// Get average WAL flush size in bytes.
    pub fn avg_wal_flush_bytes(&self) -> Option<u64> {
        self.wal_flush_bytes.checked_div(self.wal_flush_count)
    }

    /// Get WAL write throughput (bytes per second based on WAL flush time).
    pub fn wal_throughput_bytes(&self) -> f64 {
        if self.wal_flush_time.as_secs_f64() > 0.0 {
            self.wal_flush_bytes as f64 / self.wal_flush_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average WAL I/O latency.
    pub fn avg_wal_io_latency(&self) -> Option<Duration> {
        if self.wal_io_count > 0 {
            Some(self.wal_io_time / self.wal_io_count as u32)
        } else {
            None
        }
    }

    /// Get average index update latency.
    pub fn avg_index_update_latency(&self) -> Option<Duration> {
        if self.index_update_count > 0 {
            Some(self.index_update_time / self.index_update_count as u32)
        } else {
            None
        }
    }

    /// Get average rows per index update.
    pub fn avg_index_update_rows(&self) -> Option<u64> {
        self.index_update_rows.checked_div(self.index_update_count)
    }

    /// Get average MemTable flush latency.
    pub fn avg_memtable_flush_latency(&self) -> Option<Duration> {
        if self.memtable_flush_count > 0 {
            Some(self.memtable_flush_time / self.memtable_flush_count as u32)
        } else {
            None
        }
    }

    /// Get average MemTable flush size in rows.
    pub fn avg_memtable_flush_rows(&self) -> Option<u64> {
        self.memtable_flush_rows
            .checked_div(self.memtable_flush_count)
    }

    /// Log stats summary using tracing (for structured telemetry).
    pub fn log_summary(&self, prefix: &str) {
        tracing::info!(
            prefix = prefix,
            put_count = self.put_count,
            put_throughput = self.put_throughput(),
            put_avg_latency_us = self.avg_put_latency().unwrap_or_default().as_micros() as u64,
            wal_flush_count = self.wal_flush_count,
            wal_flush_bytes = self.wal_flush_bytes,
            wal_avg_latency_us =
                self.avg_wal_flush_latency().unwrap_or_default().as_micros() as u64,
            memtable_flush_count = self.memtable_flush_count,
            memtable_flush_rows = self.memtable_flush_rows,
            memtable_avg_latency_us = self
                .avg_memtable_flush_latency()
                .unwrap_or_default()
                .as_micros() as u64,
            "MemWAL stats summary"
        );
    }

    /// Log detailed WAL flush breakdown (WAL I/O vs index update) using tracing.
    pub fn log_wal_breakdown(&self, prefix: &str) {
        if self.wal_flush_count > 0 {
            tracing::info!(
                prefix = prefix,
                wal_total_latency_us =
                    self.avg_wal_flush_latency().unwrap_or_default().as_micros() as u64,
                wal_io_latency_us =
                    self.avg_wal_io_latency().unwrap_or_default().as_micros() as u64,
                index_update_latency_us = self
                    .avg_index_update_latency()
                    .unwrap_or_default()
                    .as_micros() as u64,
                index_update_rows = self.index_update_rows,
                "MemWAL WAL flush breakdown"
            );
        }
    }
}

/// Shared stats handle for use across components.
pub type SharedWriteStats = Arc<WriteStats>;

/// Create a new shared stats collector.
pub fn new_shared_stats() -> SharedWriteStats {
    Arc::new(WriteStats::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::test_util::failing_memory_store;
    use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field};
    use lance_core::FenceReason;
    use rstest::rstest;
    use std::sync::atomic::AtomicUsize;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, uri, temp_dir)
    }

    #[test]
    fn test_merge_close_stage_preserves_first_error() {
        let result = ShardWriter::merge_close_stage(
            Err(Error::io("primary close error")),
            "secondary close stage",
            Err(Error::io("secondary close error")),
        );

        let error = result.expect_err("close must preserve the first error");
        assert!(matches!(&error, Error::IO { .. }));
        assert!(
            error.to_string().contains("primary close error"),
            "unexpected error: {error}"
        );
        assert!(
            !error.to_string().contains("secondary close error"),
            "secondary error replaced the primary error: {error}"
        );
    }

    /// Base schema with `id` marked as the unenforced primary key (delete needs
    /// a PK). `name` is nullable so a tombstone can null it.
    fn create_pk_test_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = std::collections::HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);
        Arc::new(ArrowSchema::new(vec![
            id,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// [`create_pk_test_schema`] with a non-nullable `name` — the shape that
    /// used to make `delete` fail.
    fn create_strict_pk_test_schema() -> Arc<ArrowSchema> {
        let fields: Vec<Field> = create_pk_test_schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(false))
            .collect();
        Arc::new(ArrowSchema::new(fields))
    }

    fn id_only_keys(ids: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(ids.to_vec()))],
        )
        .unwrap()
    }

    /// Two same-typed columns handed over swapped pass every positional check
    /// downstream, so the logical-schema gate has to catch them by name.
    #[tokio::test]
    async fn test_put_rejects_swapped_same_typed_columns() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("email", DataType::Utf8, true),
        ]));
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            ShardWriterConfig {
                shard_id: Uuid::new_v4(),
                ..Default::default()
            },
            schema,
            vec![],
        )
        .await
        .unwrap();

        let swapped = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("email", DataType::Utf8, true),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a@example.com"])),
                Arc::new(StringArray::from(vec!["a"])),
            ],
        )
        .unwrap();

        let error = writer.put(vec![swapped]).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("column 0") && message.contains("email") && message.contains("name"),
            "error should name the position and both columns: {message}"
        );

        writer.close().await.unwrap();
    }

    #[test]
    fn test_ensure_tombstone_column_injects_false() {
        let base = create_test_schema();
        let storage = schema_with_tombstone(&base);
        let out = ensure_tombstone_column(create_test_batch(&base, 0, 3), &storage).unwrap();
        assert_eq!(out.schema(), storage);
        let ts = out
            .column_by_name(TOMBSTONE)
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(
            (0..3).all(|i| !ts.value(i)),
            "put injects _tombstone = false"
        );
        // Idempotent: a batch already carrying the column passes through.
        let again = ensure_tombstone_column(out.clone(), &storage).unwrap();
        assert_eq!(again.schema(), out.schema());
    }

    #[test]
    fn test_build_tombstone_batch_shape() {
        let storage = schema_with_tombstone(&create_test_schema());
        let tomb =
            build_tombstone_batch(&id_only_keys(&[5, 7]), &storage, &["id".to_string()]).unwrap();
        assert_eq!(tomb.schema(), storage);
        assert_eq!(tomb.num_rows(), 2);
        let ids = tomb
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[5, 7]);
        let name = tomb.column_by_name("name").unwrap();
        assert!(
            name.is_null(0) && name.is_null(1),
            "non-PK columns are null in a tombstone"
        );
        let ts = tomb
            .column_by_name(TOMBSTONE)
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(ts.value(0) && ts.value(1), "_tombstone = true");
    }

    #[test]
    fn test_build_tombstone_batch_missing_pk_errors() {
        let storage = schema_with_tombstone(&create_test_schema());
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "other",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        assert!(build_tombstone_batch(&keys, &storage, &["id".to_string()]).is_err());
    }

    #[test]
    fn test_build_tombstone_batch_nulls_non_nullable_base_column() {
        // The point of the storage schema: a tombstone nulls `v` even though the
        // base table declares it non-nullable.
        let pk = ["id".to_string()];
        let base = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let storage = relax_non_pk_nullability(&schema_with_tombstone(&base), &pk);

        let batch = build_tombstone_batch(&id_only_keys(&[1]), &storage, &pk).unwrap();

        assert!(batch["v"].is_null(0), "the tombstone must null `v`");
        assert!(!batch["id"].is_null(0), "the primary key survives");
        assert!(
            batch[TOMBSTONE]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
    }

    #[test]
    fn test_build_tombstone_batch_rejects_null_primary_key() {
        // Primary keys are never relaxed, so the storage schema still rejects a
        // null key — the delete path needs no separate check for it.
        let pk = ["id".to_string()];
        let base = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let storage = relax_non_pk_nullability(&schema_with_tombstone(&base), &pk);
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from(vec![None::<i32>]))],
        )
        .unwrap();

        let error = build_tombstone_batch(&keys, &storage, &pk).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        assert!(
            error.to_string().contains("non-nullable"),
            "error should name the nullability violation: {error}"
        );
    }

    #[tokio::test]
    async fn test_shard_writer_delete_round_trip() {
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: true,
            ..Default::default()
        };
        let shard_id = config.shard_id;
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // ids 0..5, then delete id=2.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.delete(vec![id_only_keys(&[2])]).await.unwrap();

        let refs = writer.in_memory_memtable_refs().await.unwrap();
        let scanner = LsmScanner::without_base_table(
            schema.clone(),
            base_uri,
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(shard_id, refs);
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let arr = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..arr.len()).map(|i| arr.value(i)));
        }
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![0, 1, 3, 4],
            "id=2 deleted; tombstone not surfaced"
        );

        writer.close().await.unwrap();
    }

    /// Delete works against a base table with non-nullable non-PK columns, and
    /// survivors come back through the narrowing egress relabel intact.
    #[tokio::test]
    async fn test_delete_against_non_nullable_base_column_round_trip() {
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_strict_pk_test_schema();
        assert!(
            !schema.field_with_name("name").unwrap().is_nullable(),
            "the point of this test is a non-nullable non-PK column"
        );
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: true,
            ..Default::default()
        };
        let shard_id = config.shard_id;
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.delete(vec![id_only_keys(&[2])]).await.unwrap();

        let refs = writer.in_memory_memtable_refs().await.unwrap();
        let scanner = LsmScanner::without_base_table(
            schema.clone(),
            base_uri,
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(shard_id, refs);
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let mut rows: Vec<(i32, String)> = Vec::new();
        for b in &batches {
            assert!(
                !b.schema().field_with_name("name").unwrap().is_nullable(),
                "egress must narrow back to the logical schema"
            );
            let ids = b["id"].as_any().downcast_ref::<Int32Array>().unwrap();
            let names = b["name"].as_any().downcast_ref::<StringArray>().unwrap();
            rows.extend((0..ids.len()).map(|i| (ids.value(i), names.value(i).to_string())));
        }
        rows.sort_unstable();

        assert_eq!(
            rows,
            vec![
                (0, "name_0".to_string()),
                (1, "name_1".to_string()),
                (3, "name_3".to_string()),
                (4, "name_4".to_string()),
            ],
            "id=2 deleted; every survivor keeps its non-nullable value"
        );

        writer.close().await.unwrap();
    }

    /// The storage schema no longer rejects a caller's null, so `put` is the
    /// only thing standing between a null and a non-nullable base column.
    #[tokio::test]
    async fn test_put_rejects_null_in_non_nullable_base_column() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_strict_pk_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            ShardWriterConfig {
                shard_id: Uuid::new_v4(),
                ..Default::default()
            },
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let error = writer.put(vec![null_name_batch()]).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );
        assert!(
            error.to_string().contains("base table schema"),
            "error should point at the schema contract: {error}"
        );

        writer.close().await.unwrap();
    }

    /// WAL-only mode validates too — it has no memtable, so before this gate
    /// nothing checked its input at all.
    #[tokio::test]
    async fn test_wal_only_put_rejects_null_in_non_nullable_base_column() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_strict_pk_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let error = writer.put(vec![null_name_batch()]).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error:?}"
        );

        writer.close().await.unwrap();
    }

    /// A caller-shaped batch that declares `name` nullable and carries a null —
    /// legal Arrow, illegal against a base table that declares it non-nullable.
    fn null_name_batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(StringArray::from(vec![Some("a"), None])),
            ],
        )
        .unwrap()
    }

    /// `delete_no_wait` lands the tombstone in the in-memory tier (visible at
    /// the batch-store level the instant it returns) and hands back the
    /// durability watcher *without* awaiting it. Index-driven LSM read
    /// visibility — folding the tombstone out — follows once the watcher
    /// resolves the flush that advances the visibility watermark and updates
    /// the PK index. The delete analog of
    /// `test_put_no_wait_durable_visible_then_durable`.
    #[tokio::test]
    async fn test_shard_writer_delete_no_wait_durable_visible_after_watcher() {
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: true,
            ..Default::default()
        };
        let shard_id = config.shard_id;
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();

        // Tombstone id=2 without blocking on durability.
        let (_result, watcher) = writer
            .delete_no_wait(vec![id_only_keys(&[2])])
            .await
            .unwrap();

        // The tombstone is in the in-memory tier immediately (5 rows + 1
        // tombstone), even though the index-driven read can't fold it until the
        // flush behind the watcher lands. `durable_write` is on, so a watcher is
        // returned to await.
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 6);
        let mut watcher = watcher.expect("durable_write returns a watcher");

        // Awaiting the watcher waits for the flush, which advances the
        // visibility watermark and updates the PK index — only then does the LSM
        // read fold the delete.
        watcher.wait().await.unwrap();

        let refs = writer.in_memory_memtable_refs().await.unwrap();
        let scanner = LsmScanner::without_base_table(
            schema.clone(),
            base_uri,
            vec![],
            vec!["id".to_string()],
        )
        .with_in_memory_memtables(shard_id, refs);
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let arr = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..arr.len()).map(|i| arr.value(i)));
        }
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![0, 1, 3, 4],
            "delete folded by the LSM read once the watcher resolves"
        );

        writer.close().await.unwrap();
    }

    /// With `durable_write` off, `delete_no_wait` returns no watcher (nothing to
    /// await), but the tombstone still lands in the in-memory tier. The delete
    /// analog of `test_put_no_wait_non_durable_returns_no_watcher`.
    #[tokio::test]
    async fn test_non_durable_delete_is_read_your_writes() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: false,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();

        let (_result, watcher) = writer
            .delete_no_wait(vec![id_only_keys(&[2])])
            .await
            .unwrap();
        // As with a put: a non-durable delete awaits its index apply, not an S3
        // round-trip. It is read-your-writes, just not durable.
        let mut watcher = watcher.expect("a non-durable delete awaits its index apply");
        watcher.wait().await.unwrap();

        // Tombstone landed in the in-memory tier (5 rows + 1 tombstone).
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 6);

        writer.close().await.unwrap();
    }

    /// Read every surviving `id` through the full LSM read path (which folds
    /// `NOT _tombstone` per source) over the writer's *flushed* generations,
    /// with an optional filter. Mirrors how a query reads a WAL table after a
    /// flush — the path the wallop fuzz exercised when it caught a deleted row
    /// resurfacing.
    async fn read_sstable_ids_via_lsm(
        writer: &ShardWriter,
        schema: Arc<ArrowSchema>,
        base_uri: &str,
        shard_id: Uuid,
        filter: Option<&str>,
    ) -> Vec<i32> {
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use futures::TryStreamExt;

        let manifest = writer.manifest().await.unwrap().unwrap();
        let mut snapshot =
            ShardSnapshot::new(shard_id).with_current_generation(manifest.current_generation);
        for sstable in &manifest.sstables {
            snapshot = snapshot.with_sstable(sstable.generation, sstable.path.clone());
        }
        let mut scanner = LsmScanner::without_base_table(
            schema,
            base_uri.to_string(),
            vec![snapshot],
            vec!["id".to_string()],
        );
        if let Some(predicate) = filter {
            scanner = scanner.filter(predicate).unwrap();
        }
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let arr = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            ids.extend((0..arr.len()).map(|i| arr.value(i)));
        }
        ids.sort_unstable();
        ids
    }

    fn flush_test_config(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            durable_write: false,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    /// Delete a key, then flush: the tombstone and the live row land in the
    /// *same* SSTable, so flush-time dedup must keep the tombstone
    /// (newest) and the read must fold it away. Regression for the wallop
    /// phantom (deleted row resurfacing in a filtered read after flush).
    #[tokio::test]
    async fn test_shard_writer_delete_then_flush_round_trip() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // ids 0..5, delete id=2, THEN flush (insert + tombstone in one gen).
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.delete(vec![id_only_keys(&[2])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![0, 1, 3, 4],
            "id=2 deleted before flush; tombstone must not surface in an SSTable scan"
        );
        // The filtered read path (folds NOT _tombstone into the predicate) must
        // also drop it — this is the exact wallop failure shape (`id < 3`).
        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 3"))
                .await,
            vec![0, 1],
            "filtered read after flush must not resurface deleted id=2"
        );

        writer.close().await.unwrap();
    }

    /// Insert+flush, then delete+flush: the tombstone lands in a *newer*
    /// generation than the live row, so the cross-generation block-list must
    /// mask the older row by PK. This is the wallop scenario (seed flushed,
    /// then delete, then flush).
    #[tokio::test]
    async fn test_shard_writer_delete_across_sstables() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // gen 1: ids 0..5.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // gen 2: tombstone for id=0 (a later generation than the live row).
        writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![1, 2, 3, 4],
            "id=0 tombstoned in a newer gen must mask the older gen's live row"
        );
        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
                .await,
            Vec::<i32>::new(),
            "filtered read 'id < 1' must not resurface cross-gen deleted id=0"
        );

        writer.close().await.unwrap();
    }

    /// Same as the cross-generation case, but the SSTables carry a
    /// BTree index on `id` (as every wallop table does). A filtered read
    /// `id < 1` resolves through the scalar index; the `NOT _tombstone` residual
    /// must still be applied or the deleted row leaks. This is the exact wallop
    /// failure (BTree id + `FilteredRead 'id < 1'` resurfacing deleted id=0).
    #[tokio::test]
    async fn test_shard_writer_delete_across_sstables_indexed() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = create_pk_test_schema();
        let shard_id = Uuid::new_v4();
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            flush_test_config(shard_id),
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        // gen 1: ids 0..5 (indexed). gen 2: tombstone for id=0.
        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();
        writer.delete(vec![id_only_keys(&[0])]).await.unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, None).await,
            vec![1, 2, 3, 4],
            "indexed cross-gen: full scan must mask deleted id=0"
        );
        assert_eq!(
            read_sstable_ids_via_lsm(&writer, schema.clone(), &base_uri, shard_id, Some("id < 1"))
                .await,
            Vec::<i32>::new(),
            "indexed filtered read 'id < 1' must not resurface deleted id=0 (wallop repro)"
        );

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_validation_errors() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;

        // No primary key on the schema → delete is rejected.
        let no_pk = create_test_schema();
        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            ShardWriterConfig {
                shard_id: Uuid::new_v4(),
                durable_write: false,
                ..Default::default()
            },
            no_pk,
            vec![],
        )
        .await
        .unwrap();
        assert!(
            writer.delete(vec![id_only_keys(&[1])]).await.is_err(),
            "delete without a primary key must error"
        );
        // Empty key list is rejected.
        assert!(writer.delete(vec![]).await.is_err());
        writer.close().await.unwrap();
    }

    /// The compaction Phase-1 hard-delete primitive: a key-only `merge_insert`
    /// with `WhenMatched::Delete` + `use_index` over a COMPOSITE join key
    /// `(id, shard_key)` where only `id` carries a scalar index. The partial
    /// index makes `indexed_join_keys` non-empty, so the scalar-index merge path
    /// engages — and it must still delete exactly the `(0, 0)` row. If it
    /// no-ops (or mismatches on the partial key), the compactor trims the WAL
    /// tombstone while the base row survives → the wallop resurface phantom.
    #[tokio::test]
    async fn test_indexed_composite_key_delete_removes_row() {
        use crate::Dataset;
        use crate::dataset::{WhenMatched, WhenNotMatched, WhenNotMatchedBySource};
        use crate::index::DatasetIndexExt;
        use arrow_array::{Int64Array, RecordBatchIterator};
        use lance_index::IndexType;
        use lance_index::scalar::ScalarIndexParams;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("shard_key", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]));
        // ids 0..5, shard_key = id % 8 (the wallop composite PK), value = id*100.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..5)),
                Arc::new(Int64Array::from_iter_values((0..5).map(|i| i % 8))),
                Arc::new(Int64Array::from_iter_values((0..5).map(|i| i * 100))),
            ],
        )
        .unwrap();

        let uri = format!("shared-memory://phase1-delete-{}/", Uuid::new_v4().simple());
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            &uri,
            None,
        )
        .await
        .unwrap();
        // Scalar index on `id` only — `shard_key` is unindexed, mirroring the
        // WAL base table (BTree on `id`, composite PK `(id, shard_key)`).
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Phase-1 call: key-only source `(id=0, shard_key=0)`, delete-when-matched.
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("shard_key", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![0_i64])),
                Arc::new(Int64Array::from(vec![0_i64])),
            ],
        )
        .unwrap();
        let mut builder = crate::dataset::MergeInsertBuilder::try_new(
            Arc::new(dataset),
            vec!["id".to_string(), "shard_key".to_string()],
        )
        .unwrap();
        builder
            .when_matched(WhenMatched::Delete)
            .when_not_matched(WhenNotMatched::DoNothing)
            .when_not_matched_by_source(WhenNotMatchedBySource::Keep)
            .use_index(true);
        let job = builder.try_build().unwrap();
        let keys_schema = keys.schema();
        let (dataset, _stats) = job
            .execute_reader(RecordBatchIterator::new([Ok(keys)], keys_schema))
            .await
            .unwrap();

        // id=0 must be gone — both from a full scan and from an indexed filter.
        let all = dataset.scan().try_into_batch().await.unwrap();
        let mut ids: Vec<i64> = all
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4],
            "Phase-1 must hard-delete (0, 0) from base"
        );

        let filtered = dataset
            .scan()
            .filter("id < 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            filtered.num_rows(),
            0,
            "indexed filter must not resurface the deleted composite key"
        );
    }

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, start_id: i32, num_rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    start_id..start_id + num_rows as i32,
                )),
                Arc::new(StringArray::from_iter_values(
                    (0..num_rows).map(|i| format!("name_{}", start_id as usize + i)),
                )),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_shard_writer_basic_write() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Write a batch
        let batch = create_test_batch(&schema, 0, 10);
        let result = writer.put(vec![batch]).await.unwrap();

        assert_eq!(result.batch_positions, 0..1);

        // Check stats
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);
        assert_eq!(stats.batch_count, 1);

        // Close writer
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_put_no_wait_durable_visible_then_durable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let batch = create_test_batch(&schema, 0, 10);
        let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
        assert_eq!(result.batch_positions, 0..1);

        // Row is visible in memory before durability is awaited.
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        // durable_write is on, so a watcher is returned and resolves once the
        // triggered flush lands.
        let mut watcher = watcher.expect("durable_write returns a watcher");
        watcher.wait().await.unwrap();

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_non_durable_put_is_read_your_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let batch = create_test_batch(&schema, 0, 10);
        let (result, watcher) = writer.put_no_wait(vec![batch]).await.unwrap();
        assert_eq!(result.batch_positions, 0..1);

        // A non-durable put still has something to await: its *index apply*.
        // Skipping durability now costs the caller durability only — not
        // visibility. Before the index apply was split off the WAL flush, a
        // non-durable write was not read-your-writes at all: the row stayed
        // invisible until some later flush happened to index it.
        let mut watcher = watcher.expect("a non-durable put awaits its index apply");
        watcher.wait().await.unwrap();

        let scanned = writer.scan().await.unwrap().try_into_batch().await.unwrap();
        assert_eq!(
            scanned.num_rows(),
            10,
            "a non-durable put must be readable as soon as it returns"
        );

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_shard_writer_multiple_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        // Write multiple batches in a single put call
        let batches: Vec<_> = (0..5)
            .map(|i| create_test_batch(&schema, i * 10, 10))
            .collect();
        let result = writer.put(batches).await.unwrap();
        assert_eq!(result.batch_positions, 0..5);

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 50);
        assert_eq!(stats.batch_count, 5);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_shard_writer_with_indexes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        // Write a batch
        let batch = create_test_batch(&schema, 0, 10);
        writer.put(vec![batch]).await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 10);

        writer.close().await.unwrap();
    }

    /// End-to-end check that the background flush handler rebuilds secondary
    /// indexes on every SSTable. Before this, the handler flushed
    /// via plain `flush`, leaving SSTables unindexed — point
    /// lookups had to full-scan and vector search's index-only `fast_search`
    /// couldn't see the data at all.
    #[tokio::test]
    async fn test_sstable_is_indexed() {
        use crate::index::DatasetIndexExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let config = ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri.clone(),
            config,
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // Freeze the active memtable and wait until it lands on disk.
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Resolve the SSTable recorded in the manifest.
        let manifest = writer.manifest().await.unwrap().unwrap();
        assert_eq!(manifest.sstables.len(), 1, "expected exactly one SSTable");
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, shard_id, manifest.sstables[0].path
        );

        // The SSTable must carry the BTree index built during flush.
        let dataset = crate::Dataset::open(&gen_uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "SSTable should have one index");
        assert_eq!(indices[0].name, "id_idx");

        // A PK filter over it must resolve through the index, not a full scan.
        let mut scan = dataset.scan();
        scan.filter("id = 5").unwrap();
        scan.prefilter(true);
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "LanceRead: ...full_filter=id = Int32(5)...
  ScalarIndexQuery: query=[id = 5]@id_idx(BTree)",
        )
        .await
        .unwrap();

        // And the index returns the correct row.
        let batch = dataset
            .scan()
            .filter("id = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);

        writer.close().await.unwrap();
    }

    /// Test memtable auto-flush triggered by size threshold.
    #[tokio::test]
    async fn test_shard_writer_auto_flush_by_size() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Use a small memtable size to trigger auto-flush
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 1024, // Very small - will trigger flush quickly
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // Write batches until auto-flush triggers
        for i in 0..20 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Give time for background flush to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that generation increased (indicating flush happened)
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation > initial_gen,
            "Generation should increment after auto-flush"
        );

        writer.close().await.unwrap();
    }

    /// `MemTableStats::frozen_count` and `ShardMemory::frozen_bytes` count
    /// different things on purpose: bytes drop on flush commit, the handle
    /// lingers for `frozen_memtable_grace`. A long grace therefore leaves count
    /// non-zero with bytes back at zero — and pins that the two surfaces are
    /// answering different questions, not disagreeing about one.
    #[tokio::test]
    async fn test_memtable_stats_frozen_count_outlives_frozen_bytes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 1024, // small enough to seal within the loop below
            frozen_memtable_grace: Duration::from_secs(600),
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let fresh = writer.memtable_stats().await.unwrap();
        assert_eq!(fresh.frozen_count, 0);
        assert_eq!(writer.memory().frozen_bytes(), 0);

        for i in 0..20 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }
        writer.wait_for_flush_drain().await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.frozen_count > 0,
            "flushed memtables must stay in the read view for the grace window"
        );
        let memory = writer.memory();
        assert_eq!(
            memory.frozen_bytes(),
            0,
            "every seal flushed, so nothing is owed to flush"
        );

        // Owing nothing to flush is not the same as holding nothing. Those
        // generations stay resident for the whole grace window, and a
        // process-wide budget has to see them even though the per-shard flush
        // valve deliberately does not meter them.
        assert!(
            memory.grace_bytes() > 0,
            "flushed-but-retained generations hold real memory"
        );
        assert_eq!(
            memory.retained_bytes(),
            memory.unflushed_bytes() + memory.grace_bytes(),
            "retained is the whole footprint; unflushed is only what a flush can reclaim"
        );

        writer.close().await.unwrap();
    }

    /// The controller sits in a private `WriterMode` variant, reachable only
    /// through the writer. Both modes must answer.
    #[rstest::rstest]
    #[case::memtable(true)]
    #[case::wal_only(false)]
    #[tokio::test]
    async fn test_backpressure_stats_reachable_in_both_modes(#[case] enable_memtable: bool) {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            enable_memtable,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        // A writer well under the threshold has never throttled.
        let stats = writer.backpressure_stats();
        assert_eq!(stats.total_count, 0);
        assert_eq!(stats.total_wait_ms, 0);
        assert_eq!(stats.active_count, 0);

        writer.close().await.unwrap();
    }

    /// Regression for #6713: a single failing `handle()` must not kill
    /// the dispatcher. Earlier the loop would `break Err(e)` on the
    /// first message error, dropping the rx side and stranding
    /// subsequent senders. The flusher tasks need to survive transient
    /// errors so the writer keeps making forward progress.
    #[tokio::test]
    async fn test_task_dispatcher_survives_handle_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct FlakyHandler {
            call_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MessageHandler<u32> for FlakyHandler {
            async fn handle(&mut self, message: u32) -> Result<()> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(Error::io("first message intentionally fails"))
                } else {
                    let _ = message;
                    Ok(())
                }
            }
        }

        let executor = TaskExecutor::new();
        let call_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::unbounded_channel::<u32>();
        executor
            .add_handler(
                "flaky".to_string(),
                Box::new(FlakyHandler {
                    call_count: call_count.clone(),
                }),
                rx,
            )
            .unwrap();

        // Send three messages: the first errors, the next two should
        // still be delivered to the (still-alive) handler.
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();

        for _ in 0..50 {
            if call_count.load(Ordering::SeqCst) >= 3 {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        }

        assert!(
            call_count.load(Ordering::SeqCst) >= 3,
            "dispatcher exited after the first error; only {} message(s) were handled",
            call_count.load(Ordering::SeqCst)
        );

        executor
            .shutdown_all()
            .await
            .expect("dispatcher should shut down successfully");
    }

    #[tokio::test]
    async fn test_task_executor_shutdown_propagates_cleanup_error_and_joins_all_tasks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CleanupHandler {
            cleanup_count: Arc<AtomicUsize>,
            error_message: Option<&'static str>,
        }

        #[async_trait]
        impl MessageHandler<u32> for CleanupHandler {
            async fn handle(&mut self, _message: u32) -> Result<()> {
                Ok(())
            }

            async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
                self.cleanup_count.fetch_add(1, Ordering::SeqCst);
                match self.error_message {
                    Some(message) => Err(Error::io(message)),
                    None => Ok(()),
                }
            }
        }

        let executor = TaskExecutor::new();
        let cleanup_count = Arc::new(AtomicUsize::new(0));
        let (_failing_tx, failing_rx) = mpsc::unbounded_channel::<u32>();
        executor
            .add_handler(
                "failing-cleanup".to_string(),
                Box::new(CleanupHandler {
                    cleanup_count: cleanup_count.clone(),
                    error_message: Some("intentional cleanup failure"),
                }),
                failing_rx,
            )
            .unwrap();
        let (_successful_tx, successful_rx) = mpsc::unbounded_channel::<u32>();
        executor
            .add_handler(
                "successful-cleanup".to_string(),
                Box::new(CleanupHandler {
                    cleanup_count: cleanup_count.clone(),
                    error_message: None,
                }),
                successful_rx,
            )
            .unwrap();

        let error = executor
            .shutdown_all()
            .await
            .expect_err("shutdown must propagate the handler cleanup failure");
        assert!(matches!(&error, Error::IO { .. }));
        assert!(
            error.to_string().contains("intentional cleanup failure"),
            "unexpected error: {error}"
        );
        assert_eq!(
            cleanup_count.load(Ordering::SeqCst),
            2,
            "shutdown must join and clean up every task after the first failure"
        );
        assert!(executor.tasks.read().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_task_executor_shutdown_propagates_task_panic() {
        struct PanickingCleanupHandler;

        #[async_trait]
        impl MessageHandler<u32> for PanickingCleanupHandler {
            async fn handle(&mut self, _message: u32) -> Result<()> {
                Ok(())
            }

            async fn cleanup(&mut self, _shutdown_ok: bool) -> Result<()> {
                panic!("intentional cleanup panic");
            }
        }

        let executor = TaskExecutor::new();
        let (_tx, rx) = mpsc::unbounded_channel::<u32>();
        executor
            .add_handler(
                "panicking-cleanup".to_string(),
                Box::new(PanickingCleanupHandler),
                rx,
            )
            .unwrap();

        let error = executor
            .shutdown_all()
            .await
            .expect_err("shutdown must propagate the task panic");
        assert!(matches!(&error, Error::Internal { .. }));
        assert!(
            error.to_string().contains("panicking-cleanup")
                && error.to_string().contains("panicked during shutdown"),
            "unexpected error: {error}"
        );
    }

    /// Regression for #6713 against memory://, which is closer to S3
    /// semantics (conditional PUT and list-prefix consistency).
    #[tokio::test]
    async fn test_shard_writer_auto_flush_repeatedly_memory_store() {
        let base_uri = "memory:///bench_test_flush";
        let (store, base_path) = ObjectStore::from_uri(base_uri).await.unwrap();
        let base_uri = base_uri.to_string();
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // The bug appeared on the second generation because repeated flushes
        // reused the generation-1 path. Queue several flushes before draining
        // so both path uniqueness and background sequencing are exercised.
        for i in 0..8 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        writer.wait_for_flush_drain().await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation >= initial_gen + 3,
            "expected repeated successful flushes; generation went {} → {}",
            initial_gen,
            stats.generation
        );

        writer.close().await.unwrap();
    }

    /// Regression for #6713: with durable_write=true and a memtable
    /// size threshold that fires frequently, the memtable flush task
    /// hit "Dataset already exists: …_gen_1" once the second flush
    /// started.
    #[tokio::test]
    async fn test_shard_writer_auto_flush_repeatedly_local_store() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            // Tiny size threshold — every batch crosses it.
            max_memtable_size: 64,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        // Queue multiple generations before waiting. The original failure was
        // deterministic on the second flush, so three committed generations
        // are sufficient to prove that paths and generation IDs advance.
        for i in 0..8 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        writer.wait_for_flush_drain().await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation >= initial_gen + 3,
            "expected repeated successful flushes; generation went {} → {}",
            initial_gen,
            stats.generation
        );

        writer.close().await.unwrap();
    }

    /// Regression: `close()` must flush the active memtable (not just
    /// drain the WAL). Earlier, with a `max_memtable_size` set well
    /// above the workload, no auto-flush would fire and `close()`
    /// would return without producing a Lance fragment — data was
    /// durable in the WAL but no LSM-level generation existed,
    /// surprising callers and making flush-cost benchmarks impossible.
    ///
    /// The test verifies, end-to-end:
    /// 1. close() returns Ok (a freeze/flush error must propagate, not
    ///    be silently dropped).
    /// 2. The persisted shard manifest's `current_generation` has
    ///    advanced past the initial generation — direct evidence that
    ///    a MemTable flush + manifest commit happened during close()
    ///    rather than the active memtable being dropped on the floor.
    /// (Verifying replay's post-flush behavior is tangled with
    /// independent replay logic and is exercised by the dedicated
    /// `test_memtable_replay_*` tests.)
    #[tokio::test]
    async fn test_close_flushes_active_memtable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Reuse the same shard_id when reopening so we observe state
        // produced by the first writer's `close()`.
        let shard_id = Uuid::new_v4();
        // Huge size threshold so puts never trigger an auto-flush.
        let config = ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: usize::MAX,
            max_unflushed_memtable_bytes: usize::MAX,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;

        for i in 0..50 {
            let batch = create_test_batch(&schema, i * 10, 10);
            writer.put(vec![batch]).await.unwrap();
        }

        // Pre-close sanity: no auto-flush should have fired.
        let stats_before = writer.memtable_stats().await.unwrap();
        assert_eq!(
            stats_before.generation, initial_gen,
            "no flush should have fired during puts (size threshold is usize::MAX)"
        );
        assert!(
            stats_before.row_count > 0,
            "memtable should hold the rows we just inserted"
        );

        // close() must succeed; any freeze/flush error must propagate.
        writer
            .close()
            .await
            .expect("close() must succeed and propagate any freeze/flush error");

        // Reopen the same shard and read the persisted manifest. The
        // active memtable from the closed writer was frozen + flushed
        // inside close(), which must have committed a new manifest
        // recording the advanced generation.
        let reopened =
            ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
                .await
                .unwrap();
        let manifest = reopened
            .manifest()
            .await
            .unwrap()
            .expect("reopened shard must have a persisted manifest");
        assert!(
            manifest.current_generation > initial_gen,
            "expected manifest current_generation to advance past {} after close() flushed the active memtable; got {}",
            initial_gen,
            manifest.current_generation,
        );

        reopened.close().await.unwrap();
    }

    #[rstest]
    #[case::memtable(true)]
    #[case::wal_only(false)]
    #[tokio::test]
    async fn test_close_propagates_final_wal_persistence_failure(#[case] enable_memtable: bool) {
        let (store, base_path, controls) = failing_memory_store().await;
        let base_uri = "memory:///";
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            enable_memtable,
            max_wal_buffer_size: usize::MAX,
            max_wal_flush_interval: None,
            max_wal_persist_retries: 0,
            max_memtable_size: usize::MAX,
            max_unflushed_memtable_bytes: usize::MAX,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();
        let task_executor = writer.task_executor.clone();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        controls.fail_wal_puts(usize::MAX);

        let error = writer
            .close()
            .await
            .expect_err("close must propagate the final WAL persistence failure");
        assert_eq!(error.fence_reason(), Some(FenceReason::PersistenceFailure));
        assert!(
            error
                .to_string()
                .contains("injected transient WAL put failure"),
            "unexpected error: {error}"
        );
        assert!(
            task_executor.tasks.read().unwrap().is_empty(),
            "close must join background tasks before returning an error"
        );
    }

    /// Recompute what the shard is holding straight from `WriterState`, the
    /// way [`ShardWriter::memtable_stats`] reads it — independent of the
    /// publish mechanism, which is the thing that can drift.
    async fn ground_truth(writer: &ShardWriter) -> (usize, usize) {
        let state = writer.memtable_state_lock().unwrap().read().await;
        let active = in_memory_ref(&state.memtable).resident_bytes();
        let frozen = state
            .frozen_memtables
            .iter()
            .filter(|frozen| frozen.flushed_at_ms.is_none())
            .map(|frozen| in_memory_ref(&frozen.memtable).resident_bytes())
            .sum();
        (active, frozen)
    }

    async fn assert_no_drift(writer: &ShardWriter, after: &str) {
        let (active, frozen) = ground_truth(writer).await;
        let memory = writer.memory();
        assert_eq!(
            (memory.active_bytes(), memory.frozen_bytes()),
            (active, frozen),
            "published memory drifted from the writer state after {after}"
        );
        assert_eq!(
            memory.unflushed_bytes(),
            active + frozen,
            "unflushed must be the sum of the two terms after {after}"
        );
    }

    /// The keystone invariant: the published snapshot is **derived** from
    /// `WriterState`, never adjusted, so it cannot drift from it — across every
    /// event that changes the memtable set.
    ///
    /// This is what replaced a pair of incremented counters. A counter has no
    /// way back: one missed decrement is permanent, and the pod eventually
    /// refuses every write with a memtable that reads empty. So this walks the
    /// writer through open, puts, a seal, a flush commit, and puts into the
    /// fresh memtable, checking the invariant after each.
    #[rstest]
    #[case::grace_keeps_handles(Duration::from_secs(600))]
    #[case::zero_grace_evicts_on_commit(Duration::ZERO)]
    #[tokio::test]
    async fn test_memory_snapshot_never_drifts_from_writer_state(#[case] grace: Duration) {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: false,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            // Small enough that the loop below seals several times.
            max_memtable_size: 2048,
            frozen_memtable_grace: grace,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        // Seeded at open, before anything has been written.
        assert_no_drift(&writer, "open").await;

        for i in 0..24 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
            assert_no_drift(&writer, &format!("put {i}")).await;
        }

        // Seals happened above; make sure at least one did, or this test proves
        // nothing about the freeze path.
        assert!(
            writer.memory().frozen_bytes() > 0
                || writer.memtable_stats().await.unwrap().frozen_count > 0,
            "the loop must have sealed at least once for this to cover freeze"
        );

        writer.wait_for_flush_drain().await.unwrap();
        assert_no_drift(&writer, "flush drain").await;

        // A long grace keeps the flushed handles in the read view and a zero
        // grace evicts them on commit — two different publish paths. Either way
        // they are reclaimable, so neither may keep metering.
        assert_eq!(
            writer.memory().frozen_bytes(),
            0,
            "flushed memtables are reclaimable and must stop metering (grace {grace:?})"
        );

        for i in 24..32 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
            assert_no_drift(&writer, &format!("post-flush put {i}")).await;
        }

        writer.close().await.unwrap();
    }

    /// The bug the whole design exists to avoid: the old accounting read
    /// `WriterState` through `try_read()`, which fails while a writer holds the
    /// lock — and, because tokio's `RwLock` is write-preferring, also while one
    /// is merely queued. It therefore reported **zero** on exactly the shards
    /// taking writes, so a pod under load looked empty.
    ///
    /// Holding the write lock outright is the deterministic version of that
    /// race.
    #[tokio::test]
    async fn test_memory_is_readable_while_a_writer_holds_the_lock() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            ShardWriterConfig {
                shard_id: Uuid::new_v4(),
                durable_write: false,
                ..Default::default()
            },
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 100)])
            .await
            .unwrap();
        let unlocked = writer.memory().unflushed_bytes();
        assert!(unlocked > 0, "rows are resident, so this cannot be zero");

        let state_lock = writer.memtable_state_lock().unwrap().clone();
        let held = state_lock.write().await;
        assert_eq!(
            writer.memory().unflushed_bytes(),
            unlocked,
            "a writer holding the lock must not zero the memory view"
        );
        // The drain classification comes off the same published snapshot, so it
        // answers under the lock as well. Nothing is frozen here, so the honest
        // answer is that no flush is outstanding.
        assert!(
            matches!(writer.memory().drain(), Drain::Stalled),
            "the drain classification is published too, so it reads under the lock"
        );
        drop(held);

        writer.close().await.unwrap();
    }

    /// Two seal predicates exist — `MemTable::should_flush` and
    /// `memtable_reached_flush_threshold` — and their *row-window* arms must trip
    /// at the same byte. They did not: one counted the PK bloom filter and the
    /// other did not, so they disagreed by a fixed offset on every memtable.
    ///
    /// Only that arm is shared. `memtable_reached_flush_threshold` also seals on
    /// resident bytes, which `should_flush` knows nothing about, so the ceiling
    /// below is held out of range to compare like with like.
    #[tokio::test]
    async fn test_both_seal_predicates_share_one_byte_arm() {
        let schema = create_test_schema();
        let mut memtable =
            MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 64).unwrap();
        memtable
            .insert(create_test_batch(&schema, 0, 50))
            .await
            .unwrap();

        let at = memtable.batch_store().row_bytes();
        assert!(at > 0, "the memtable must hold something to be a test");
        // Owned batches, so the pinned footprint sits just under the window sum
        // (no per-batch header in the allocation walk). Asserted rather than
        // assumed: if it ever exceeded `at`, the retained arm would fire and this
        // would be comparing something other than the row arm.

        // `incoming_batches` of 1 against a capacity of 64 keeps the batch-count
        // arm out of it, and `usize::MAX` keeps the resident arm out, so only the
        // row-window arms are being compared.
        for (bytes, expected) in [(at, true), (at + 1, false)] {
            assert_eq!(
                memtable.should_flush(bytes),
                expected,
                "should_flush at {bytes}"
            );
            assert_eq!(
                memtable_reached_flush_threshold(&memtable, bytes, usize::MAX, 1),
                expected,
                "the two seal predicates disagree at {bytes}; a bloom-sized offset \
                 between them makes every memtable seal early on one path"
            );
        }
    }

    /// A handle taken before any rows exist must still see them arrive: the
    /// memory view is read without the writer lock, so a handle that captured a
    /// value instead of a live counter would gate writes on a stale reading.
    /// Index heap is counted, row bytes are not inflated by it.
    #[tokio::test]
    async fn test_memory_handle_tracks_the_live_memtable() {
        let schema = create_test_schema();
        let mut memtable =
            MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 8).unwrap();

        let handle = in_memory_ref(&memtable);
        let empty_rows = handle.row_bytes();
        for _ in 0..3 {
            memtable
                .insert(create_test_batch(&schema, 0, 10))
                .await
                .unwrap();
        }

        assert!(
            handle.row_bytes() > empty_rows,
            "a handle taken before the writes must still see them"
        );
        assert_eq!(
            handle.row_bytes(),
            in_memory_ref(&memtable).row_bytes(),
            "a handle and a freshly taken one must agree"
        );
        // The bloom filter is fixed-size and lives in the index term, so it
        // never moves the flush unit.
        assert_eq!(
            handle.row_bytes(),
            memtable.batch_store().row_bytes(),
            "row bytes are the flush unit: batches only"
        );
        assert_eq!(
            handle.index_bytes(),
            super::super::memtable::pk_bloom_filter_bytes(),
            "an unindexed memtable still holds its PK bloom filter"
        );
        // The ceiling is built on what the batches keep alive, not on the flush
        // unit. The two row measures are close but never identical even for
        // owned batches — buffer capacity is padded, and `row_bytes` charges a
        // `RecordBatch` header per batch that an allocation walk does not see.
        // How far they diverge for zero-copy slices, which is the case that
        // matters, is pinned in `BatchStore`'s own tests.
        assert_eq!(
            handle.resident_bytes(),
            handle.retained_row_bytes() + handle.index_bytes()
        );
    }

    /// A `ShardMemory` backed by a closure instead of a live writer, re-read on
    /// every poll exactly as the real one is.
    fn fake_memory(read: impl Fn() -> usize + Send + Sync + 'static) -> ShardMemory {
        ShardMemory(ShardMemorySource::Fake(Arc::new(read)))
    }

    fn fixed_memory(unflushed: usize) -> ShardMemory {
        fake_memory(move || unflushed)
    }

    fn empty_shard_memory() -> ShardMemory {
        fixed_memory(0)
    }

    #[tokio::test]
    async fn test_no_backpressure_when_under_threshold() {
        let config = ShardWriterConfig::default().with_max_unflushed_memtable_bytes(1024 * 1024); // 1MB

        let controller = LocalBackpressureController::new(&config);

        // Should return immediately - well under threshold (100 bytes < 1MB)
        controller
            .maybe_apply_backpressure(fixed_memory(100))
            .await
            .unwrap();

        assert_eq!(controller.stats().count(), 0);
    }

    #[tokio::test]
    async fn test_backpressure_loops_until_under_threshold() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;

        let config = ShardWriterConfig::default()
            .with_max_unflushed_memtable_bytes(100) // Very low threshold
            .with_backpressure_log_interval(Duration::from_millis(50));

        // Simulate: starts at 1000 bytes, drops by 400 each call (simulating memtable flushes)
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let controller = LocalBackpressureController::new(&config);
        let draining = fake_memory(move || {
            let count = call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // 1000 -> 600 -> 200 -> under threshold (need 3 iterations)
            1000usize.saturating_sub(count * 400)
        });

        controller.maybe_apply_backpressure(draining).await.unwrap();

        // Should have read the shard 4 times (initial + 3 waits until under 100)
        assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 4);
        // Should have recorded backpressure wait time (waited 3 times)
        assert_eq!(controller.stats().count(), 1);
        assert_eq!(
            controller.stats().snapshot().active_count,
            0,
            "the wait is over, so nobody is parked"
        );
    }

    /// The totals only move when a wait ends, so a first stall would be
    /// invisible to a poller if `active_count` did not report it live.
    #[tokio::test]
    async fn test_backpressure_in_progress_wait_is_observable() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering as AtomicOrdering;
        use std::time::Duration;

        let config = ShardWriterConfig::default()
            .with_max_unflushed_memtable_bytes(100)
            .with_backpressure_log_interval(Duration::from_millis(50));

        let unflushed = Arc::new(AtomicUsize::new(1000));
        let release = unflushed.clone();

        let controller = LocalBackpressureController::new(&config);
        let stats = controller.stats().clone();

        let parked = controller
            .maybe_apply_backpressure(fake_memory(move || unflushed.load(AtomicOrdering::Relaxed)));

        let observer = async {
            // Bounded so a regression that never publishes the park fails here
            // instead of hanging the suite.
            let deadline = Instant::now() + Duration::from_secs(5);
            while stats.snapshot().active_count == 0 {
                assert!(
                    Instant::now() < deadline,
                    "an ongoing wait was never published"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            let mid = stats.snapshot();
            assert_eq!(mid.active_count, 1);
            assert_eq!(
                mid.total_count, 0,
                "a wait still in progress is not a completed one"
            );
            assert_eq!(mid.total_wait_ms, 0);

            release.store(0, AtomicOrdering::Relaxed);
        };

        let (result, ()) = tokio::join!(parked, observer);
        result.unwrap();

        let after = stats.snapshot();
        assert_eq!(after.active_count, 0, "the guard drops when the wait ends");
        assert_eq!(after.total_count, 1);
    }

    /// A `put` whose caller times out drops the wait future mid-park. The gauge
    /// must come back down, or a cancelled writer is throttled forever on paper.
    #[tokio::test]
    async fn test_backpressure_cancelled_wait_does_not_leak_active_count() {
        use std::time::Duration;

        let config = ShardWriterConfig::default()
            .with_max_unflushed_memtable_bytes(100)
            .with_backpressure_log_interval(Duration::from_millis(50));

        let controller = LocalBackpressureController::new(&config);

        // Never drops below the threshold, so the timeout is what ends the wait.
        assert!(
            tokio::time::timeout(
                Duration::from_millis(50),
                controller.maybe_apply_backpressure(fixed_memory(1000)),
            )
            .await
            .is_err()
        );

        let after = controller.stats().snapshot();
        assert_eq!(after.active_count, 0, "cancellation must release the guard");
        assert_eq!(
            after.total_count, 0,
            "a cancelled wait never completed, so it is not a completed wait"
        );
    }

    /// Records what it saw and answers with a fixed verdict.
    #[derive(Debug)]
    struct SpyController {
        seen: Arc<StdRwLock<Vec<(usize, usize)>>>,
        reject: bool,
    }

    #[async_trait::async_trait]
    impl BackpressureController for SpyController {
        async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()> {
            self.seen
                .write()
                .unwrap()
                .push((shard.active_bytes(), shard.frozen_bytes()));
            if self.reject {
                return Err(Error::backpressure("full"));
            }
            Ok(())
        }
    }

    /// An injected controller *replaces* the built-in valve rather than
    /// stacking on it, and is handed what the calling shard holds, split the
    /// way relief cares about — everything it needs to own the whole policy,
    /// per-shard rules included.
    #[tokio::test]
    async fn test_injected_controller_replaces_default_and_sees_shard_memory() {
        let seen = Arc::new(StdRwLock::new(Vec::new()));
        let spy = Arc::new(SpyController {
            seen: seen.clone(),
            reject: false,
        });
        // A budget the built-in valve would trip on instantly, to prove it is
        // not the thing being consulted.
        let config = ShardWriterConfig::default()
            .with_max_unflushed_memtable_bytes(1)
            .with_backpressure(spy);

        let controller = resolve_backpressure(&config);
        controller
            .maybe_apply_backpressure(fixed_memory(1000))
            .await
            .unwrap();

        assert_eq!(
            *seen.read().unwrap(),
            vec![(1000, 0)],
            "the injected controller ran, and the built-in valve did not park the write"
        );
    }

    /// `ShardMemory` must stay live across polls: a controller that delays is
    /// waiting for these bytes to fall, so a captured copy would spin forever.
    #[tokio::test]
    async fn test_shard_memory_reflects_drain_across_polls() {
        #[derive(Debug)]
        struct DrainWaiter {
            polls: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl BackpressureController for DrainWaiter {
            async fn maybe_apply_backpressure(&self, shard: ShardMemory) -> Result<()> {
                while shard.unflushed_bytes() > 0 {
                    self.polls.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
                Ok(())
            }
        }

        let resident = Arc::new(AtomicUsize::new(4_096));
        let drain = resident.clone();
        let view = fake_memory(move || resident.load(Ordering::Relaxed));

        let controller = Arc::new(DrainWaiter {
            polls: AtomicUsize::new(0),
        });
        let gate = controller.clone();
        let waiting = tokio::spawn(async move { gate.maybe_apply_backpressure(view).await });

        // Let the gate observe the full pool, then drain it as a flush commit
        // would. A stale copy would never see this and the task would hang.
        tokio::task::yield_now().await;
        drain.store(0, Ordering::Relaxed);

        waiting.await.unwrap().unwrap();
        assert!(controller.polls.load(Ordering::Relaxed) > 0);
    }

    /// With nothing injected, lance keeps its own per-shard valve.
    #[tokio::test]
    async fn test_default_backpressure_is_used_when_none_injected() {
        let config = ShardWriterConfig::default().with_max_unflushed_memtable_bytes(100);
        let controller = resolve_backpressure(&config);

        let polls = Arc::new(AtomicUsize::new(0));
        let polls_clone = polls.clone();
        controller
            .maybe_apply_backpressure(fake_memory(move || {
                polls_clone.fetch_add(1, Ordering::Relaxed);
                0
            }))
            .await
            .unwrap();

        assert_eq!(polls.load(Ordering::Relaxed), 1);
    }

    /// A rejecting controller surfaces as `Error::Backpressure`, which is
    /// distinguishable from a real failure without matching on the message.
    #[tokio::test]
    async fn test_injected_controller_rejects_with_backpressure_error() {
        let spy = Arc::new(SpyController {
            seen: Arc::new(StdRwLock::new(Vec::new())),
            reject: true,
        });
        let config = ShardWriterConfig::default().with_backpressure(spy);
        let controller = resolve_backpressure(&config);

        let err = controller
            .maybe_apply_backpressure(empty_shard_memory())
            .await
            .unwrap_err();
        assert!(err.is_backpressure(), "expected backpressure, got {err:?}");
    }

    #[test]
    fn test_record_put() {
        let stats = WriteStats::new();
        stats.record_put(Duration::from_millis(10));
        stats.record_put(Duration::from_millis(20));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.put_count, 2);
        assert_eq!(snapshot.put_time, Duration::from_millis(30));
        assert_eq!(snapshot.avg_put_latency(), Some(Duration::from_millis(15)));
    }

    #[test]
    fn test_record_wal_flush() {
        let stats = WriteStats::new();
        stats.record_wal_flush(Duration::from_millis(100), 1024);
        stats.record_wal_flush(Duration::from_millis(200), 2048);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.wal_flush_count, 2);
        assert_eq!(snapshot.wal_flush_time, Duration::from_millis(300));
        assert_eq!(snapshot.wal_flush_bytes, 3072);
        assert_eq!(snapshot.avg_wal_flush_bytes(), Some(1536));
    }

    #[test]
    fn test_record_memtable_flush() {
        let stats = WriteStats::new();
        stats.record_memtable_flush(Duration::from_secs(1), 10000);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.memtable_flush_count, 1);
        assert_eq!(snapshot.memtable_flush_time, Duration::from_secs(1));
        assert_eq!(snapshot.memtable_flush_rows, 10000);
    }

    #[test]
    fn test_stats_reset() {
        let stats = WriteStats::new();
        stats.record_put(Duration::from_millis(10));
        stats.record_wal_flush(Duration::from_millis(100), 1024);

        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.put_count, 0);
        assert_eq!(snapshot.wal_flush_count, 0);
    }

    // ----- WAL-only mode (`enable_memtable = false`) -----

    fn wal_only_config(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            enable_memtable: false,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_wal_only_durable_round_trip() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let shard_id = Uuid::new_v4();
        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Two durable puts → two WAL entries. Each `put` awaits its own append
        // (driven by the background ticker) before returning, so the second
        // batch is only pushed after the first is durable — they never coalesce.
        let r1 = writer
            .put(vec![create_test_batch(&schema, 0, 4)])
            .await
            .unwrap();
        let r2 = writer
            .put(vec![create_test_batch(&schema, 100, 2)])
            .await
            .unwrap();
        assert_eq!(r1.batch_positions, 0..1);
        assert_eq!(r2.batch_positions, 1..2);

        writer.close().await.unwrap();

        // Read back via WalTailer. WAL positions are 1-based, so two
        // entries from a fresh shard land at 1 and 2.
        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.first_position().await.unwrap(), 1);
        assert_eq!(tailer.next_position().await.unwrap(), 3);

        let e0 = tailer.read_entry(1).await.unwrap().unwrap();
        let e1 = tailer.read_entry(2).await.unwrap().unwrap();
        assert_eq!(e0.batches.len(), 1);
        assert_eq!(e0.batches[0].num_rows(), 4);
        assert_eq!(e1.batches.len(), 1);
        assert_eq!(e1.batches[0].num_rows(), 2);
        // Both entries from the same writer should have the same epoch.
        assert_eq!(e0.writer_epoch, e1.writer_epoch);
        assert!(e0.writer_epoch >= 1);
    }

    /// A durable WAL-only put is driven by the background ticker, not an inline
    /// per-put flush: `put().await` must not return until the ticker's append
    /// advances the durability watermark it waits on. With a long interval and a
    /// buffer the write cannot cross, the ticker is the *only* thing that can
    /// make the put durable — so if the WAL entry is present the instant `put`
    /// returns (before any `close()`), the watermark-wait is doing its job.
    #[tokio::test]
    async fn test_wal_only_durable_put_waits_for_ticker_append() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let mut config = wal_only_config(shard_id);
        config.max_wal_flush_interval = Some(Duration::from_millis(100));
        config.max_wal_buffer_size = 100 * 1024 * 1024; // never crossed

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 4)])
            .await
            .unwrap();

        // `put` returned, so the batch must already be durable — read the WAL
        // directly, without closing the writer.
        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.next_position().await.unwrap(), 2);
        let entry = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(entry.batches.len(), 1);
        assert_eq!(entry.batches[0].num_rows(), 4);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_rejects_index_configs() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let err = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema,
            index_configs,
        )
        .await
        .err()
        .expect("expected invalid_input");
        assert!(
            err.to_string().contains("indexes require enable_memtable"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_wal_only_rejects_empty_batches() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Empty list.
        let err = writer.put(vec![]).await.err().unwrap();
        assert!(err.to_string().contains("empty batch list"));

        // Single empty batch.
        let zero = arrow_array::RecordBatch::new_empty(schema);
        let err = writer.put(vec![zero]).await.err().unwrap();
        assert!(err.to_string().contains("Batch 0 is empty"));

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_memtable_accessors_error() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema,
            vec![],
        )
        .await
        .unwrap();

        let err = writer.memtable_stats().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.scan().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.active_memtable_ref().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));
        let err = writer.in_memory_memtable_refs().await.err().unwrap();
        assert!(err.to_string().contains("WAL-only mode"));

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_async_batches_multiple_puts() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let mut config = wal_only_config(Uuid::new_v4());
        let shard_id = config.shard_id;
        // Non-durable: puts should accumulate in the pending queue until
        // close() drains them with a single WAL entry.
        config.durable_write = false;
        config.max_wal_flush_interval = None;
        config.max_wal_buffer_size = 100 * 1024 * 1024; // never crossed

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        for i in 0..3 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
        }

        writer.close().await.unwrap();

        // All three puts should be in a single WAL entry at position 1
        // (WAL positions are 1-based).
        let tailer = WalTailer::new(store, base_path, shard_id);
        assert_eq!(tailer.next_position().await.unwrap(), 2);
        let entry = tailer.read_entry(1).await.unwrap().unwrap();
        assert_eq!(entry.batches.len(), 3);
        for (i, batch) in entry.batches.iter().enumerate() {
            assert_eq!(batch.num_rows(), 10, "batch {i}");
        }
    }

    #[tokio::test]
    async fn test_wal_only_fencing() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        // Writer B claims a higher epoch and writes — fences A.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());
        writer_b
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap();

        // A's next durable put must fail with a fence error (the underlying
        // WalAppender::append surfaces it via atomic_put).
        let err = writer_a
            .put(vec![create_test_batch(&schema, 2, 1)])
            .await
            .expect_err("expected fence error");
        assert!(
            err.to_string().contains("Writer fenced"),
            "unexpected error: {err}"
        );

        writer_b.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_check_fenced_detects_successor_claim() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Not yet fenced.
        writer_a.check_fenced().await.unwrap();

        // Successor claims a higher epoch.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());

        // A's check_fenced surfaces the fence without needing a put round-trip.
        let err = writer_a
            .check_fenced()
            .await
            .expect_err("expected fence error");
        assert!(
            err.to_string().contains("Writer fenced"),
            "unexpected error: {err}"
        );

        // B is the current writer and is not fenced.
        writer_b.check_fenced().await.unwrap();
        writer_b.close().await.unwrap();
    }

    // ----- MemTable replay on open -----

    fn memtable_config_with_pk(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            max_wal_buffer_size: 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    fn schema_with_pk() -> Arc<ArrowSchema> {
        use arrow_schema::Field;
        // Mark `id` as the unenforced primary key.
        let pk_meta: std::collections::HashMap<String, String> = [(
            "lance-schema:unenforced-primary-key".to_string(),
            "1".to_string(),
        )]
        .into_iter()
        .collect();
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(pk_meta);
        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    // A durable write whose WAL PUT keeps failing poisons the writer with a
    // typed persistence failure; the next write *and every read* fail fast with
    // the same reason; and once storage heals, reopening replays the WAL and
    // writes resume.
    #[tokio::test]
    async fn test_writer_poisons_on_persistence_failure_and_recovers_on_reopen() {
        let (store, base_path, controls) = failing_memory_store().await;
        let base_uri = "memory:///";
        let shard_id = Uuid::new_v4();
        let schema = schema_with_pk();
        controls.fail_wal_puts(usize::MAX);

        let config = ShardWriterConfig {
            max_wal_persist_retries: 1,
            wal_persist_retry_base_delay: Duration::from_millis(1),
            ..memtable_config_with_pk(shard_id)
        };

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // The durable put waits on the flush, which poisons after its retries.
        let err = writer
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap_err();
        assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));

        // A poisoned writer rejects further writes fast, same typed reason.
        let err = writer
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap_err();
        assert_eq!(err.fence_reason(), Some(FenceReason::PersistenceFailure));

        // ...and rejects *reads* too. Batch 0 was committed to the BatchStore
        // before its WAL PUT failed, so a poisoned writer that still served
        // reads would hand out a row that is not durable and that replay will
        // not reproduce — a divergent snapshot. Mirrors SlateDB's
        // `check_closed()` at the top of every read.
        for reason in [
            writer.scan().await.err().and_then(|e| e.fence_reason()),
            writer
                .active_memtable_ref()
                .await
                .err()
                .and_then(|e| e.fence_reason()),
            writer
                .in_memory_memtable_refs()
                .await
                .err()
                .and_then(|e| e.fence_reason()),
        ] {
            assert_eq!(reason, Some(FenceReason::PersistenceFailure));
        }

        // Stats stay readable: this is what an operator (and the eviction path)
        // inspects to decide what to do about the poisoned shard.
        writer.memtable_stats().await.unwrap();
        drop(writer);

        // Storage heals: reopening replays the WAL and accepts writes again.
        controls.recover();
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 2, 1)])
            .await
            .unwrap();
    }

    /// A doomed open must fail on local validation *before* it claims the epoch,
    /// so it cannot fence the healthy writer already serving the shard. The
    /// index-config check used to run *after* `claim_epoch` (and, for a successor,
    /// after `write_fence_sentinel`), so a rejected open still bumped the stored
    /// epoch and fenced the incumbent.
    #[tokio::test]
    async fn test_doomed_open_does_not_fence_incumbent() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        // An index config that disagrees with the schema (FTS on the Int32 `id`
        // column) is rejected on local validation. On the old path this rejection
        // landed only after the epoch had already been claimed.
        let bad_fts = MemIndexConfig::Fts(FtsIndexConfig::new(
            "bad_fts".to_string(),
            0,
            "id".to_string(),
        ));
        let err = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![bad_fts],
        )
        .await
        .map(|_| ())
        .expect_err("an FTS index on a non-Utf8 column must be rejected");
        assert!(
            err.to_string().contains("bad_fts") && err.to_string().contains("Utf8"),
            "unexpected error: {err}"
        );

        // The incumbent is untouched: not fenced, still accepting writes.
        writer_a.check_fenced().await.unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap();
        writer_a.close().await.unwrap();
    }

    /// A failed dispatch during `freeze_memtable` must not drop the outgoing
    /// table's rows from the read view. The active memtable is replaced before
    /// the WAL-flush and index-apply sends; a send that failed (background tasks
    /// gone) used to return before the outgoing table was retained in
    /// `frozen_memtables`, so its accepted rows silently vanished — a scan
    /// returned 0 rows with no error. The writer must instead retain the table
    /// and poison, so reads fail fast rather than serve a divergent snapshot.
    #[tokio::test]
    async fn test_freeze_dispatch_failure_retains_rows_and_poisons() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Non-durable + no ticker: the put is read-your-writes (waits for its
        // index apply) but nothing is WAL-flushed, so the freeze below still owes
        // a WAL append.
        let config = ShardWriterConfig {
            durable_write: false,
            max_wal_flush_interval: None,
            ..memtable_config_with_pk(shard_id)
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);

        // Tear the background tasks down out from under the writer, so the
        // freeze's dispatch sends hit closed channels.
        writer.abort().await.unwrap();

        let err = writer
            .force_seal_active()
            .await
            .expect_err("force_seal_active must surface the failed dispatch");
        assert!(
            err.to_string().contains("channel closed"),
            "unexpected error: {err}"
        );

        // The failure poisoned the writer: reads fail fast instead of returning a
        // silent zero-row snapshot of a shard whose rows were dropped.
        assert!(
            writer.scan().await.is_err(),
            "a poisoned writer must reject reads, not serve a divergent snapshot"
        );
        assert!(writer.in_memory_memtable_refs().await.is_err());
    }

    /// A WAL holding more batches than one memtable's capacity must reopen.
    ///
    /// One memtable holds at most `max_memtable_batches` batches, but a WAL is
    /// unbounded, so replay has to rotate — seal the full memtable, start a fresh
    /// one — exactly as the live write path does. Before, replay stuffed
    /// everything into a single memtable and `open()` failed outright with
    /// "MemTable batch store is full", leaving the shard permanently unopenable.
    #[tokio::test]
    async fn test_replay_rotates_when_wal_exceeds_one_memtable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        const N: i32 = 8;

        // Writer A has a *large* capacity, so its eight one-batch puts all land in
        // a single memtable and it never freezes or flushes a generation of its
        // own. Dropping it without close leaves an eight-entry WAL and no
        // generations — a WAL that no single small memtable could hold.
        let writer_a_config = ShardWriterConfig {
            max_memtable_batches: 1000,
            ..memtable_config_with_pk(shard_id)
        };
        // Writer B has a *two-batch* capacity, so replaying that eight-entry WAL is
        // exactly what must rotate. Keeping the configs distinct isolates replay
        // rotation from the live rotation writer A would otherwise do concurrently.
        let config = ShardWriterConfig {
            max_memtable_batches: 2,
            ..memtable_config_with_pk(shard_id)
        };

        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                writer_a_config,
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            for id in 0..N {
                writer_a
                    .put(vec![create_test_batch(&schema, id, 1)])
                    .await
                    .unwrap();
            }
            // Drop without close: only the WAL survives, and it holds more batches
            // than writer B's memtable can.
        }

        // Total rows across the active memtable plus every SSTable.
        // Distinct ids, so no cross-generation dedup — a plain sum is exact.
        async fn total_rows(writer: &ShardWriter, base_uri: &str, shard_id: Uuid) -> usize {
            let mut rows = writer.memtable_stats().await.unwrap().row_count;
            let manifest = writer.manifest().await.unwrap().unwrap();
            for sstable in &manifest.sstables {
                let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, sstable.path);
                let dataset = crate::Dataset::open(&gen_uri).await.unwrap();
                rows += dataset.count_rows(None).await.unwrap();
            }
            rows
        }

        // Reopen. This used to fail with a full-batch-store error.
        let writer_b = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            config.clone(),
            schema.clone(),
            vec![],
        )
        .await
        .expect("a WAL larger than one memtable must still reopen");

        // Rotation produced sealed memtables, and replay flushed each to a Lance
        // generation rather than holding it in memory or leaving it in the WAL.
        let manifest = writer_b.manifest().await.unwrap().unwrap();
        assert!(
            !manifest.sstables.is_empty(),
            "replay must have sealed and flushed at least one full memtable"
        );

        // Every row survived, split between the SSTables and the
        // active (partial) memtable.
        assert_eq!(
            total_rows(&writer_b, &base_uri, shard_id).await as i32,
            N,
            "every replayed row must be durable, across generations and the active memtable"
        );
        writer_b.close().await.unwrap();

        // Because the sealed memtables were flushed, the manifest's replay cursor
        // advanced past their WAL entries — so a second reopen replays only the
        // tail and still accounts for every row. The WAL truncates across reopens
        // rather than growing without bound.
        let writer_c =
            ShardWriter::open(store, base_path, base_uri.clone(), config, schema, vec![])
                .await
                .unwrap();
        assert_eq!(total_rows(&writer_c, &base_uri, shard_id).await as i32, N);
        writer_c.close().await.unwrap();
    }

    /// Replay-on-open recovers durable WAL entries that were never flushed
    /// to a Lance generation. Setup: writer A durably writes batches, drops
    /// without close (so MemTable freeze never runs); writer B reopens and
    /// must see A's rows in its MemTable scan.
    #[tokio::test]
    async fn test_memtable_replay_recovers_unflushed_writes() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write two durable batches, then drop without close.
        // The WAL files persist; the in-memory MemTable does not.
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 5)])
                .await
                .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 100, 3)])
                .await
                .unwrap();
            // intentionally drop without close()
        }

        // Writer B reopens. Replay must rehydrate A's two batches into the
        // active MemTable.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();

        let stats = writer_b.memtable_stats().await.unwrap();
        assert_eq!(
            stats.row_count, 8,
            "expected replay to insert 5 + 3 = 8 rows, got {}",
            stats.row_count
        );
        assert_eq!(
            stats.batch_count, 2,
            "expected replay to insert 2 batches, got {}",
            stats.batch_count
        );

        writer_b.close().await.unwrap();
    }

    /// Replayed batches are already WAL-durable, so the first flush after a
    /// reopen must not re-append them to the WAL or re-insert them into the
    /// indexes. Before replay stamped the durability cursor it stayed at
    /// "nothing flushed", so the next flush re-covered `[0, end)`: it appended
    /// the already-durable rows a second time *and* re-indexed them. None of
    /// the in-memory indexes is idempotent, so an indexed PK lookup returned
    /// the row twice while a full scan still looked healthy.
    #[tokio::test]
    async fn test_replay_does_not_reappend_or_reindex() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: two durable batches (5 + 3 = 8 rows), dropped without close.
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 5)])
                .await
                .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 100, 3)])
                .await
                .unwrap();
        }

        // Writer B reopens and replays A's two batches.
        let writer_b = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let stats = writer_b.memtable_stats().await.unwrap();
        assert_eq!(stats.batch_count, 2);
        assert_eq!(
            stats.durable_batch_count, 2,
            "replayed batches came from the WAL, so the durability cursor must already cover them"
        );

        // One more durable put. Its flush must cover only the new batch.
        writer_b
            .put(vec![create_test_batch(&schema, 200, 2)])
            .await
            .unwrap();

        let tailer = WalTailer::new(store, base_path, shard_id);
        let first = tailer.first_position().await.unwrap();
        let next = tailer.next_position().await.unwrap();
        let mut wal_rows = 0;
        for position in first..next {
            if let Some(entry) = tailer.read_entry(position).await.unwrap() {
                wal_rows += entry.batches.iter().map(|b| b.num_rows()).sum::<usize>();
            }
        }
        assert_eq!(
            wal_rows, 10,
            "WAL must hold 8 replayed + 2 new rows; a re-covering flush re-appends the replayed 8"
        );

        // The indexed arm must not see a replayed row twice.
        let mut scanner = writer_b.scan().await.unwrap();
        scanner.filter("id = 0").unwrap();
        let hit = scanner.try_into_batch().await.unwrap();
        assert_eq!(
            hit.num_rows(),
            1,
            "indexed PK lookup returned the replayed row more than once"
        );

        let all = writer_b
            .scan()
            .await
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(all.num_rows(), 10);

        writer_b.close().await.unwrap();
    }

    /// A non-durable put is readable through the **index-backed** arms the moment
    /// it returns, not just through a full scan.
    ///
    /// This is what splitting the index apply off the WAL flush buys. Before, the
    /// index apply only ran as one arm of the flush, so with `durable_write:
    /// false` nothing triggered it on the put path at all: the row sat in the
    /// batch store, unindexed, until some later flush happened along. A full scan
    /// (which reads the batch store directly) could still find it, while every
    /// index-accelerated query could not — the tiers disagreed. Now the apply is
    /// triggered per-put in both modes, so `put` returning means "indexed".
    #[tokio::test]
    async fn test_non_durable_put_is_visible_through_the_index() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        let config = ShardWriterConfig {
            durable_write: false,
            ..memtable_config_with_pk(shard_id)
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 5)])
            .await
            .unwrap();

        // The indexed PK lookup must find it — this is the arm that saw nothing
        // before, because the index apply had never run.
        let mut scanner = writer.scan().await.unwrap();
        scanner.filter("id = 3").unwrap();
        let hit = scanner.try_into_batch().await.unwrap();
        assert_eq!(
            hit.num_rows(),
            1,
            "an index-backed lookup must see a non-durable put as soon as it returns"
        );

        // ...and so must the unindexed full scan, i.e. the tiers agree.
        let all = writer.scan().await.unwrap().try_into_batch().await.unwrap();
        assert_eq!(all.num_rows(), 5);

        writer.close().await.unwrap();
    }

    /// The durability cursor is writer-global, so a put into a *post-rotation*
    /// memtable must still wait for its own WAL append.
    ///
    /// Batch positions restart at 0 in every memtable, but the durability watch
    /// channel spans the writer's whole life and is never reset. When the put
    /// path targeted a memtable-local position, the first N puts into every
    /// memtable after the first were already "satisfied" by the *previous*
    /// memtable's N appends: they acked instantly, with no WAL append, and
    /// `durable_write: true` silently degraded to non-durable. Worse, the next
    /// append then sent a *smaller* value, walking the watermark backwards and
    /// hanging any watcher still waiting on the old, higher one.
    ///
    /// The cursor is now a writer-global exclusive count and every target is
    /// lifted through the store's `global_offset`, so it only ever moves forward
    /// and a post-rotation put can only be acked by its own append.
    #[tokio::test]
    async fn test_durable_ack_after_rotation_requires_its_own_wal_append() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // A two-batch memtable, so the third put forces a freeze + rotation.
        let config = ShardWriterConfig {
            max_memtable_batches: 2,
            ..memtable_config_with_pk(shard_id)
        };

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Fill and rotate the first memtable.
        for i in 0..3 {
            writer
                .put(vec![create_test_batch(&schema, i * 5, 5)])
                .await
                .unwrap();
        }
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.global_offset > 0,
            "expected a rotation; the active memtable is still the writer's first"
        );

        // Every batch acked so far must be genuinely durable, and the cursor must
        // cover the active memtable's entire prefix rather than lagging inside it.
        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.durable_batch_count >= stats.global_offset + stats.batch_count,
            "durable_write acked a put the WAL never received: durable={} but the active \
             memtable spans [{}, {})",
            stats.durable_batch_count,
            stats.global_offset,
            stats.global_offset + stats.batch_count
        );

        // And the WAL really holds every row we acked (3 puts x 5 rows).
        writer.close().await.unwrap();
        let tailer = WalTailer::new(store, base_path, shard_id);
        let first = tailer.first_position().await.unwrap();
        let next = tailer.next_position().await.unwrap();
        let mut wal_rows = 0;
        for position in first..next {
            if let Some(entry) = tailer.read_entry(position).await.unwrap() {
                wal_rows += entry.batches.iter().map(|b| b.num_rows()).sum::<usize>();
            }
        }
        assert_eq!(
            wal_rows, 15,
            "every acked row must be in the WAL; a post-rotation put that acked without an \
             append would leave rows missing"
        );
    }

    /// A background tick must append the **oldest** store that still owes the WAL,
    /// never "whatever memtable is active".
    ///
    /// A tick carries no store: it is enqueued by a timer and resolved when it is
    /// handled. So a tick enqueued before a freeze is handled *after* it, and
    /// resolving to the active memtable would append the incoming memtable's
    /// batches ahead of the outgoing memtable's tail. WAL entry positions are
    /// assigned in append-call order, replay walks them ascending, row positions
    /// follow, and primary-key recency is "newest visible row position wins" — so
    /// that inverts dedup after a crash, handing the key to the stale row. It
    /// survives the crash that caused it, and a full scan cannot see it.
    #[test]
    fn test_next_pending_store_picks_the_oldest_owing_an_append() {
        let schema = create_test_schema();

        // A frozen store of 2 batches at coordinate 0, and the active store that
        // rotated in behind it at coordinate 2.
        let frozen = Arc::new(BatchStore::with_capacity(4));
        frozen.append(create_test_batch(&schema, 0, 1)).unwrap();
        frozen.append(create_test_batch(&schema, 1, 1)).unwrap();
        let active = Arc::new(BatchStore::with_capacity_at(4, 2));
        active.append(create_test_batch(&schema, 2, 1)).unwrap();

        let frozen_list = || std::iter::once(Arc::clone(&frozen));

        // Nothing durable: the frozen store owes the oldest append, so it wins —
        // even though the active memtable also has un-appended batches.
        let picked = next_pending_store(frozen_list(), Arc::clone(&active), 0).unwrap();
        assert!(
            Arc::ptr_eq(&picked, &frozen),
            "the outgoing memtable's tail must be appended before the incoming one's head"
        );

        // Still true partway through the frozen store.
        let picked = next_pending_store(frozen_list(), Arc::clone(&active), 1).unwrap();
        assert!(Arc::ptr_eq(&picked, &frozen));

        // Once the frozen store is fully durable, the active one is next.
        let picked = next_pending_store(frozen_list(), Arc::clone(&active), 2).unwrap();
        assert!(Arc::ptr_eq(&picked, &active));

        // Everything durable: nothing to do.
        assert!(next_pending_store(frozen_list(), Arc::clone(&active), 3).is_none());
    }

    /// A durable writer with no flush ticker cannot make progress in either
    /// mode — the ticker is the only thing that drives the WAL append the put
    /// waits on — so `open()` rejects it rather than letting a put block forever.
    #[rstest]
    #[case::memtable(true)]
    #[case::wal_only(false)]
    #[tokio::test]
    async fn test_open_rejects_durable_write_without_a_ticker(#[case] enable_memtable: bool) {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        let config = ShardWriterConfig {
            durable_write: true,
            enable_memtable,
            max_wal_flush_interval: None,
            ..memtable_config_with_pk(shard_id)
        };

        let Err(err) = ShardWriter::open(store, base_path, base_uri, config, schema, vec![]).await
        else {
            panic!("durable_write with no ticker must be rejected");
        };
        assert!(
            err.to_string().contains("max_wal_flush_interval"),
            "the error must name the knob, got: {err}"
        );
    }

    /// WAL entries must be appended in global batch-position order across a
    /// memtable rotation, because append order *is* primary-key recency order.
    ///
    /// `WalAppender::append` assigns each entry's position from its own counter,
    /// in call order. Replay walks those positions ascending and assigns row
    /// positions in that order. Primary-key recency is "newest visible row
    /// position wins". So an out-of-order append silently inverts dedup after a
    /// crash — the stale row wins — and a full scan cannot see it.
    ///
    /// The hazard is the background ticker. If it resolved to "whatever memtable
    /// is active" rather than to the oldest store still owing an append, a tick
    /// enqueued before a freeze but handled after it would append the *incoming*
    /// memtable's batches ahead of the outgoing memtable's tail. So the target is
    /// resolved from the durability cursor, not from wall-clock timing.
    #[tokio::test]
    async fn test_wal_append_order_preserves_pk_recency_across_rotation() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Two batches per memtable, so the second put fills it and rotates.
        let config = ShardWriterConfig {
            max_memtable_batches: 2,
            ..memtable_config_with_pk(shard_id)
        };

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Memtable 1: ids 7 then 20 (the second fills it and triggers the freeze).
        writer
            .put(vec![create_test_batch(&schema, 7, 1)])
            .await
            .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 20, 1)])
            .await
            .unwrap();
        // Memtable 2 overwrites id=7 with a newer row. Its append must land in the
        // WAL *after* memtable 1's, or replay would resolve id=7 to the stale copy.
        writer
            .put(vec![create_test_batch(&schema, 30, 1)])
            .await
            .unwrap();
        writer.close().await.unwrap();

        // Walk the WAL in entry order and collect the ids as replay would see them.
        let tailer = WalTailer::new(store, base_path, shard_id);
        let first = tailer.first_position().await.unwrap();
        let next = tailer.next_position().await.unwrap();
        let mut ids: Vec<i32> = Vec::new();
        for position in first..next {
            let Some(entry) = tailer.read_entry(position).await.unwrap() else {
                continue;
            };
            for batch in &entry.batches {
                let column = batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                ids.extend((0..column.len()).map(|i| column.value(i)));
            }
        }

        assert_eq!(
            ids,
            vec![7, 20, 30],
            "WAL entries must follow global batch-position order; memtable 1's rows must \
             precede memtable 2's, or replay inverts primary-key recency"
        );
    }

    /// An index config that disagrees with the schema fails `open()` outright.
    /// It must not be allowed to accept writes: the insert would fail
    /// deterministically on every batch, including batches replayed from the
    /// WAL, so once a row was durable the shard could never reopen. Before this
    /// check, an FTS index on a non-Utf8 column silently indexed nothing and the
    /// shard reported healthy.
    #[tokio::test]
    async fn test_open_rejects_index_config_that_disagrees_with_schema() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // `id` is Int32, not a string column.
        let bad_fts = MemIndexConfig::Fts(FtsIndexConfig::new(
            "bad_fts".to_string(),
            0,
            "id".to_string(),
        ));

        let Err(err) = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![bad_fts],
        )
        .await
        else {
            panic!("open must reject an FTS index on a non-Utf8 column");
        };

        let message = err.to_string();
        assert!(
            message.contains("bad_fts") && message.contains("Utf8"),
            "error must name the index and the constraint, got: {message}"
        );
    }

    /// Replay is a no-op on a fresh shard: the MemTable starts empty.
    #[tokio::test]
    async fn test_memtable_replay_no_op_on_fresh_shard() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.row_count, 0);
        assert_eq!(stats.batch_count, 0);
        writer.close().await.unwrap();
    }

    /// Regression for the OSS-WAL compactor-drain bug: after a flush
    /// records its generation in the manifest and an external compactor
    /// later drains `sstables` back to empty (the legitimate
    /// outcome after compacting the SSTable into the base table), reopening
    /// the writer must not re-replay the already-flushed WAL entry into
    /// the active memtable.
    ///
    /// Under the pre-fix logic, replay disambiguated "fresh shard" from
    /// "flushed-then-compacted" with `sstables.is_empty()`,
    /// which collapsed both cases into start-at-0. With 1-based WAL
    /// positions and a default cursor of 0 meaning "no flush stamped",
    /// the flush-then-drain sequence leaves `replay_after_wal_entry_position`
    /// pinned at the flushed position, so replay correctly starts past it.
    #[tokio::test]
    async fn test_memtable_replay_skips_entries_after_external_compaction() {
        use crate::dataset::mem_wal::ShardManifestStore;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write 5 rows, close (forces a flush of the active
        // memtable). The manifest now records an SSTable and
        // pins `replay_after_wal_entry_position` to the covered WAL entry.
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 5)])
                .await
                .unwrap();
            writer_a.close().await.unwrap();
        }

        // Simulate an external compactor compacting the SSTable.
        // into the base table: drain `sstables` to empty via a
        // direct manifest commit. The cursor stays where the flush put it.
        let manifest_store = ShardManifestStore::new(store.clone(), &base_path, shard_id, 2);
        let pre = manifest_store.latest().await.unwrap().unwrap();
        assert!(
            !pre.sstables.is_empty(),
            "writer A's close() should have stamped an SSTable"
        );
        let cursor_at_flush = pre.replay_after_wal_entry_position;
        assert!(
            cursor_at_flush >= 1,
            "expected cursor to land on a 1-based WAL position after flush, got {cursor_at_flush}"
        );
        // Bump the epoch (claim_epoch) so we can commit_update without
        // being fenced; this also mirrors how a compactor process would
        // hold its own writer claim.
        let (compactor_epoch, _) = manifest_store.claim_epoch(pre.shard_spec_id).await.unwrap();
        manifest_store
            .commit_update(compactor_epoch, |current| ShardManifest {
                version: current.next_version(),
                sstables: vec![],
                ..current.clone()
            })
            .await
            .unwrap();
        let post = manifest_store.latest().await.unwrap().unwrap();
        assert!(
            post.sstables.is_empty(),
            "compactor drain should have left sstables empty"
        );
        assert_eq!(
            post.replay_after_wal_entry_position, cursor_at_flush,
            "compactor must not touch the replay cursor"
        );

        // Writer B reopens. Pre-fix: replay saw sstables empty,
        // restarted at WAL position 0, and re-inserted writer A's rows.
        // Post-fix: replay starts at cursor + 1, finds no entry, and the
        // memtable stays empty.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let stats = writer_b.memtable_stats().await.unwrap();
        assert_eq!(
            stats.row_count, 0,
            "memtable must not re-replay compacted WAL entries; got {} rows",
            stats.row_count
        );
        assert_eq!(stats.batch_count, 0);
        writer_b.close().await.unwrap();
    }

    /// Replay aborts the open with a clear fence error if it encounters a
    /// WAL entry written with an epoch strictly greater than ours. Simulate
    /// the race where another writer wrote an entry with a higher epoch
    /// between our `claim_epoch` and our replay by injecting a high-epoch
    /// entry directly via `WalAppender::with_claimed_epoch` (which
    /// bypasses `claim_epoch` and so does not bump the manifest).
    #[tokio::test]
    async fn test_memtable_replay_fenced_aborts_open() {
        use crate::dataset::mem_wal::ShardManifestStore;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();

        // Writer A: write one durable batch (claims epoch 1, writes entry at position 1).
        {
            let writer_a = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                memtable_config_with_pk(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            writer_a
                .put(vec![create_test_batch(&schema, 0, 1)])
                .await
                .unwrap();
            // drop without close
        }

        // Inject a WAL entry written with epoch 100 — far above whatever
        // claim_epoch will hand the next opener. The manifest is not
        // updated since we use `with_claimed_epoch` directly.
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let high_epoch_appender = WalAppender::with_claimed_epoch(
            store.clone(),
            base_path.clone(),
            shard_id,
            manifest_store,
            100,
            // hint seed irrelevant; the real position counter is discovered
            // lazily on the first append.
            0,
            WalRetryConfig::default(),
        );
        high_epoch_appender
            .append(vec![create_test_batch(&schema, 999, 1)])
            .await
            .unwrap();

        // Writer B opens. claim_epoch returns 2 (manifest's writer_epoch
        // was 1 before this open). Replay reads the injected entry, sees
        // epoch 100 > 2, and aborts with a fence error.
        let result = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await;
        let Err(err) = result else {
            panic!("expected open to fail with fence error during replay");
        };
        // Assert the *typed* fence reason, not just the message: a regression
        // reverting this to `Error::io` would still carry a message containing
        // "fenced" and slip past a string check, but must not report a
        // `FenceReason`.
        assert_eq!(
            err.fence_reason(),
            Some(FenceReason::PeerClaimedEpoch),
            "replay must abort with a typed peer-fence error, got: {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("WAL replay aborted") && msg.contains("fenced"),
            "unexpected error: {msg}"
        );
    }

    /// Regression: `wal_stats().next_wal_entry_position` must reflect the
    /// post-recovery cursor immediately on reopen, not 0 until the first
    /// append discovers the tip. Pre-fix the appender's hint was seeded at
    /// 0 and only updated after the first successful append, so external
    /// monitors saw 0 between open and first put on a shard with prior
    /// entries.
    #[tokio::test]
    async fn test_wal_stats_seeded_from_manifest_on_reopen() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        // First writer creates a shard, writes one entry, closes.
        let writer1 = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer1
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();
        writer1.close().await.unwrap();

        // Reopen: stats must reflect the post-recovery cursor immediately,
        // before any put has happened on this writer.
        let writer2 = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        let next = writer2.wal_stats().next_wal_entry_position;
        assert!(
            next >= 1,
            "expected wal_stats to reflect post-recovery cursor (>= 1) on reopen, got {next}"
        );

        writer2.close().await.unwrap();
    }

    /// Regression test for the size-based trigger after a drain.
    ///
    /// Earlier the WAL-only size trigger used a monotonic counter
    /// (`wal_flush_trigger_count`) which never reset across drains. After
    /// the first crossing the counter was >= 1 and `pending_bytes / threshold`
    /// could never grow past 1 (because pending_bytes resets on drain), so
    /// the size trigger silently stopped firing. This test pushes batches
    /// to cross the size threshold multiple times across drains and asserts
    /// every crossing produces a WAL entry.
    #[tokio::test]
    async fn test_wal_only_size_trigger_fires_repeatedly() {
        use crate::dataset::mem_wal::wal::WalTailer;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let mut config = wal_only_config(Uuid::new_v4());
        let shard_id = config.shard_id;
        // Non-durable so puts don't auto-flush. Time trigger off so only
        // the size trigger drives flushes.
        config.durable_write = false;
        config.max_wal_flush_interval = None;
        // Pick a tiny threshold so a single batch crosses it.
        config.max_wal_buffer_size = 1;

        let writer = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri,
            config,
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        // Three puts, each large enough to cross the (1-byte) threshold.
        // Without the fix, only the first would trigger; the rest would
        // sit in the queue until close().
        for i in 0..3 {
            writer
                .put(vec![create_test_batch(&schema, i * 10, 10)])
                .await
                .unwrap();
            // Yield, then sleep, so the background flush handler can
            // drain the trigger queue before the next push — otherwise
            // multiple pending triggers can coalesce into a single drain.
            // 50ms historically failed on slow Windows CI runners; 250ms
            // gives a comfortable margin without making the suite slow.
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }

        writer.close().await.unwrap();

        // Each put should have produced its own WAL entry — three crossings,
        // three entries. Without the regression fix, all three batches end
        // up in a single entry written by `close()`.
        let tailer = WalTailer::new(store, base_path, shard_id);
        let next = tailer.next_position().await.unwrap();
        assert!(
            next >= 3,
            "expected at least 3 WAL entries (one per crossing), got next_position = {next}"
        );
    }

    /// Regression test for concurrent durable WAL-only puts on a fenced
    /// writer. Earlier `flush_from_wal_only` did a destructive `drain()`
    /// before calling `wal_appender.append`. If the append failed (e.g.
    /// fence), the drained batches were dropped — the next concurrent put
    /// would then see an empty pending queue and spuriously return Ok,
    /// hiding the data loss. With the snapshot/commit fix, the failed flush
    /// leaves the batches in the queue for retry, and — because the flush is
    /// terminal — it poisons the writer, waking *both* parked durability
    /// waiters with the typed fence error instead of either one hanging.
    #[tokio::test]
    async fn test_wal_only_fenced_concurrent_puts_do_not_silently_succeed() {
        use std::sync::Arc;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        // Writer A claims epoch 1, writes one entry (takes WAL position 1,
        // caches its next-position as 2 internally).
        let writer_a = Arc::new(
            ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                wal_only_config(shard_id),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap(),
        );
        writer_a
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        // Writer B claims epoch 2 and writes its own entry (takes WAL
        // position 1). A is now fenced: A's next put will attempt WAL
        // position 1 (its cached next), collide with B's entry, and
        // surface a "Writer fenced" error from `check_fenced`.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_b
            .put(vec![create_test_batch(&schema, 1, 1)])
            .await
            .unwrap();

        // Two concurrent durable puts on the (now-fenced) writer A. With
        // the destructive-drain bug, the first flush would consume both
        // pending batches into a failing append; the second flush would
        // see an empty queue and return spurious success, silently losing
        // the second put's data. Now both puts park on the durability
        // watermark; the ticker's append fails with the fence, poisons the
        // writer, and both waiters wake with the fence error — batches intact
        // in the queue.
        let a1 = writer_a.clone();
        let a2 = writer_a.clone();
        let schema1 = schema.clone();
        let schema2 = schema.clone();
        let h1 = tokio::spawn(async move { a1.put(vec![create_test_batch(&schema1, 2, 1)]).await });
        let h2 = tokio::spawn(async move { a2.put(vec![create_test_batch(&schema2, 3, 1)]).await });

        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();

        assert!(
            r1.is_err() && r2.is_err(),
            "expected both concurrent puts on a fenced writer to fail, got r1={r1:?} r2={r2:?}",
        );

        writer_b.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_only_stats_no_memtable_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            wal_only_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 0, 1)])
            .await
            .unwrap();

        let stats_handle = writer.stats_handle();
        writer.close().await.unwrap();

        let snapshot = stats_handle.snapshot();
        assert!(snapshot.put_count >= 1, "expected at least one put");
        assert!(
            snapshot.wal_flush_count >= 1,
            "expected at least one WAL flush"
        );
        assert_eq!(
            snapshot.memtable_flush_count, 0,
            "WAL-only mode must never trigger a memtable flush"
        );
        assert_eq!(
            snapshot.index_update_count, 0,
            "WAL-only mode must never trigger an index update"
        );
    }

    #[tokio::test]
    async fn test_memtable_stats_record_index_update() {
        // MemTable mode with a BTree index: index application runs on its own
        // task and must record an index-update stat. Regression for the stat
        // silently reading zero after index apply moved off the WAL-flush path.
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_pk_test_schema();
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_idx".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            flush_test_config(Uuid::new_v4()),
            schema.clone(),
            index_configs,
        )
        .await
        .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 0, 3)])
            .await
            .unwrap();

        let stats_handle = writer.stats_handle();
        // `close()` drains the index-apply task, so the apply is settled here.
        writer.close().await.unwrap();

        let snapshot = stats_handle.snapshot();
        assert!(
            snapshot.index_update_count >= 1,
            "the index apply must record an index-update stat, got {}",
            snapshot.index_update_count
        );
        assert_eq!(
            snapshot.index_update_rows, 3,
            "every indexed row must be counted exactly once, got {}",
            snapshot.index_update_rows
        );
        assert!(
            snapshot.avg_index_update_latency().is_some(),
            "a recorded index update must expose an average latency"
        );
    }

    #[tokio::test]
    async fn test_force_seal_active_and_wait_for_flush_drain() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Thresholds high enough that auto-flush won't fire; the seal is
        // the only thing that should rotate the memtable.
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        let flushed_before = writer
            .manifest()
            .await
            .unwrap()
            .map(|m| m.sstables.len())
            .unwrap_or(0);

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        let fence = writer.force_seal_active().await.unwrap();
        assert_eq!(fence.sealed_generation(), Some(initial_gen));
        fence.wait().await.unwrap();

        let stats = writer.memtable_stats().await.unwrap();
        assert_eq!(stats.generation, initial_gen + 1);
        assert_eq!(stats.batch_count, 0);

        let manifest = writer
            .manifest()
            .await
            .unwrap()
            .expect("manifest should exist after flush");
        assert_eq!(manifest.sstables.len(), flushed_before + 1);

        writer.close().await.unwrap();
    }

    /// A durable put returns only once its WAL flush landed, and the seal
    /// fence resolves only once the sealed memtable reached L0 — so both
    /// callbacks have fired by the time this asserts, without sleeping.
    #[tokio::test]
    async fn test_observer_sees_both_flush_kinds() {
        #[derive(Debug, Default)]
        struct CountingObserver {
            wal_flushes: AtomicU64,
            wal_bytes: AtomicU64,
            memtable_flushes: AtomicU64,
            memtable_rows: AtomicU64,
        }

        impl WalObserver for CountingObserver {
            fn on_wal_flush(&self, _duration: Duration, bytes: usize) {
                self.wal_flushes.fetch_add(1, Ordering::Relaxed);
                self.wal_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
            }

            fn on_memtable_flush(&self, _duration: Duration, rows: usize) {
                self.memtable_flushes.fetch_add(1, Ordering::Relaxed);
                self.memtable_rows.fetch_add(rows as u64, Ordering::Relaxed);
            }
        }

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        let observer = Arc::new(CountingObserver::default());
        let sink: Arc<dyn WalObserver> = observer.clone();
        let config = ShardWriterConfig {
            observer: Some(sink),
            ..seal_fence_test_config(Uuid::new_v4())
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer
            .force_seal_active()
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();

        assert!(observer.wal_flushes.load(Ordering::Relaxed) > 0);
        assert!(observer.wal_bytes.load(Ordering::Relaxed) > 0);
        assert_eq!(observer.memtable_flushes.load(Ordering::Relaxed), 1);
        assert_eq!(observer.memtable_rows.load(Ordering::Relaxed), 10);

        writer.close().await.unwrap();
    }

    /// Durable writes so `put` returns only once the row is indexed and
    /// WAL-durable. Both fence tests tear the background tasks down before
    /// freezing, and a freeze still owing an index apply or a WAL append
    /// would dispatch onto a closed channel and poison the writer.
    fn seal_fence_test_config(shard_id: Uuid) -> ShardWriterConfig {
        ShardWriterConfig {
            shard_id,
            shard_spec_id: 0,
            durable_write: true,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        }
    }

    /// An empty active memtable does not mean every pre-call write is in
    /// L0: a size/interval trigger swaps generation N for an empty N+1
    /// while N's flush is still in flight. The seal must fence that
    /// generation anyway — reporting "nothing sealed, nothing to wait for"
    /// lets a caller acknowledge a flush that has not happened.
    #[tokio::test]
    async fn test_force_seal_active_fences_pending_generation_when_active_is_empty() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            seal_fence_test_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // Stop the flush tasks, then freeze by hand: this is the post-rotation
        // state held still — generation N frozen and unflushed, N+1 active and
        // empty.
        writer.task_executor.shutdown_all().await.unwrap();
        let pending_generation = match &writer.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                let mut state = state.write().await;
                writer_state.freeze_memtable(&mut state).unwrap() - 1
            }
            WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
        };

        let fence = writer.force_seal_active().await.unwrap();
        assert_eq!(
            fence.sealed_generation(),
            None,
            "the active memtable was empty, so this seal froze nothing"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(200), fence.wait())
                .await
                .is_err(),
            "generation {pending_generation} is still awaiting flush; the fence must not be satisfied"
        );
    }

    /// The fence is tied to each flush's own outcome, never to
    /// `ShardManifest::current_generation`. The manifest advances to
    /// `generation + 1` on every committed flush without checking for a gap,
    /// so a later generation's success moves it past one that failed — a
    /// watermark comparison would report durability that does not exist.
    #[tokio::test]
    async fn test_force_seal_active_fence_ignores_manifest_generation_advance() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            seal_fence_test_config(Uuid::new_v4()),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // No flush task, so the sealed generation never reaches L0.
        writer.task_executor.shutdown_all().await.unwrap();
        let fence = writer.force_seal_active().await.unwrap();
        let sealed = fence
            .sealed_generation()
            .expect("the active memtable held rows");

        // Advance the manifest past the sealed generation, as a later
        // generation's successful flush would.
        writer
            .manifest_store
            .commit_update(writer.epoch(), |current| ShardManifest {
                version: current.next_version(),
                current_generation: sealed + 2,
                ..current.clone()
            })
            .await
            .unwrap();

        assert!(
            tokio::time::timeout(Duration::from_millis(200), fence.wait())
                .await
                .is_err(),
            "generation {sealed} never reached L0; a manifest advance past it must not satisfy its fence"
        );
    }

    /// `abort` tears down the background flush tasks WITHOUT flushing —
    /// buffered memtable rows are discarded, not sealed into an L0
    /// generation the way `close` would. Idempotent on a second call.
    #[tokio::test]
    async fn test_abort_discards_without_flushing_and_is_idempotent() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();

        // Thresholds high enough that nothing auto-flushes; the rows stay
        // in the active memtable until abort discards them.
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        let flushed_before = writer
            .manifest()
            .await
            .unwrap()
            .map(|m| m.sstables.len())
            .unwrap_or(0);

        writer.abort().await.unwrap();

        // No generation was sealed — contrast with `close`, which flushes
        // the 10 buffered rows into a new L0 generation.
        let flushed_after = writer
            .manifest()
            .await
            .unwrap()
            .map(|m| m.sstables.len())
            .unwrap_or(0);
        assert_eq!(
            flushed_after, flushed_before,
            "abort must not flush a new L0 generation"
        );

        // Idempotent: re-cancels the already-cancelled token, joins an
        // already-emptied task set.
        writer.abort().await.unwrap();
    }

    /// On a successful flush commit the sealed generation's rows land in the
    /// manifest immediately, but the in-memory handle is NOT dropped — it
    /// lingers for `frozen_memtable_grace` (so in-flight as-of reads keep
    /// batch-resolved membership), then is swept by the `SweepExpired` ticker.
    #[tokio::test]
    async fn test_frozen_retained_during_grace_then_swept() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            // Short grace so the sweep is observable without a slow test.
            frozen_memtable_grace: Duration::from_millis(50),
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Recorded in the manifest at commit time.
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert!(
            manifest
                .sstables
                .iter()
                .any(|g| g.generation == initial_gen),
            "SSTable must be recorded in the manifest"
        );

        // Still queryable in memory immediately after commit (within grace).
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(refs.active.generation, initial_gen + 1);
        assert!(
            refs.frozen.iter().any(|f| f.generation == initial_gen),
            "SSTable must stay queryable during the grace window"
        );

        // After the grace elapses (plus a sweep tick) the handle is evicted.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert!(
            refs.frozen.is_empty(),
            "frozen handle must be swept once the grace elapses"
        );

        writer.close().await.unwrap();
    }

    /// With zero grace (the default) a frozen handle is evicted synchronously on
    /// flush commit — no sweep tick, no lingering window.
    #[tokio::test]
    async fn test_frozen_evicted_immediately_with_zero_grace() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            shard_spec_id: 0,
            durable_write: false,
            max_wal_buffer_size: 64 * 1024 * 1024,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_size: 64 * 1024 * 1024,
            manifest_scan_batch_size: 2,
            frozen_memtable_grace: Duration::ZERO,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        let initial_gen = writer.memtable_stats().await.unwrap().generation;
        writer
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // Rows are durably in the manifest...
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert!(
            manifest
                .sstables
                .iter()
                .any(|g| g.generation == initial_gen),
            "SSTable must be recorded in the manifest"
        );

        // ...and the in-memory handle is already gone, no sweep tick needed.
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert!(
            refs.frozen.is_empty(),
            "frozen handle must be evicted on commit when grace is zero"
        );

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_close_propagates_frozen_memtable_flush_failure() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();
        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        writer_a
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema,
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());

        let error = writer_a
            .close()
            .await
            .expect_err("close must propagate the fenced MemTable flush");
        assert!(
            matches!(error, Error::IO { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("Writer fenced"),
            "unexpected error: {error}"
        );

        writer_b.close().await.unwrap();
    }

    /// Regression: a transient flush failure must NOT reopen the
    /// concurrent-read-vs-flush hole. The sealed generation stays in the
    /// queryable set (rows intact) until a later flush or WAL replay.
    /// Failure is induced deterministically by fencing the writer with a
    /// successor before the seal, so the flush's `check_fenced` rejects it.
    #[tokio::test]
    async fn test_frozen_retained_after_failed_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = create_test_schema();
        let shard_id = Uuid::new_v4();

        let writer_a = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();

        let initial_gen = writer_a.memtable_stats().await.unwrap().generation;
        writer_a
            .put(vec![create_test_batch(&schema, 0, 10)])
            .await
            .unwrap();

        // Successor claims a higher epoch, fencing A.
        let writer_b = ShardWriter::open(
            store,
            base_path,
            base_uri,
            memtable_config_with_pk(shard_id),
            schema.clone(),
            vec![],
        )
        .await
        .unwrap();
        assert!(writer_b.epoch() > writer_a.epoch());

        // `force_seal_active` would reject up-front on a fenced writer;
        // freeze directly so the failure surfaces at flush-commit time —
        // exactly the freeze/flush race the fix guards.
        match &writer_a.mode {
            WriterMode::MemTable {
                state,
                writer_state,
                ..
            } => {
                let mut st = state.write().await;
                writer_state.freeze_memtable(&mut st).unwrap();
            }
            WriterMode::WalOnly { .. } => unreachable!("opened in memtable mode"),
        }

        // The fenced flush fails; the drain surfaces that error.
        assert!(
            writer_a.wait_for_flush_drain().await.is_err(),
            "fenced flush should fail the drain"
        );

        // The hole did not reopen: the sealed generation is still queryable
        // with its rows, alongside the new (empty) active generation.
        let refs = writer_a.in_memory_memtable_refs().await.unwrap();
        assert_eq!(refs.frozen.len(), 1, "sealed generation must be retained");
        assert_eq!(refs.frozen[0].generation, initial_gen);
        assert!(
            !refs.frozen[0].batch_store.is_empty(),
            "retained sealed memtable must still hold its rows"
        );
        assert_eq!(refs.active.generation, initial_gen + 1);

        // Nor did it vanish from the accounting. A failed flush leaves its
        // memtable un-stamped, so it stays in the owed-to-flush set and keeps
        // metering — those bytes are resident and only another flush reclaims
        // them. This is also what an operator reads to decide whether to evict.
        let stats = writer_a.memtable_stats().await.unwrap();
        assert_eq!(stats.frozen_count, 1);
        let frozen_bytes = writer_a.memory().frozen_bytes();
        assert!(
            frozen_bytes >= refs.frozen[0].batch_store.row_bytes(),
            "a failed flush must keep owing its resident bytes, got {frozen_bytes}"
        );

        // Charging those bytes must not become a trap. Nothing retries the
        // failed generation, and its watcher came off the queue when the flush
        // reported, so there is no event left that would drain the pool — while
        // the valve itself is what keeps the puts that could seal a new
        // generation from arriving. Waiting here would never end, so the
        // controller refuses instead.
        assert!(
            matches!(writer_a.memory().drain(), Drain::Stalled),
            "a failed flush leaves nothing outstanding to wait on"
        );
        let controller = LocalBackpressureController::new(&ShardWriterConfig {
            max_unflushed_memtable_bytes: writer_a.memory().unflushed_bytes(),
            ..Default::default()
        });
        let refused = tokio::time::timeout(
            STALL_GRACE * 10,
            controller.maybe_apply_backpressure(writer_a.memory()),
        )
        .await
        .expect("a shard nothing can drain must refuse, not park the writer");
        assert!(
            refused.is_err_and(|e| e.is_backpressure()),
            "the refusal must be the retryable backpressure signal"
        );

        writer_b.close().await.unwrap();
    }

    /// A one-row slice pins its whole parent, so a memtable can hold megabytes
    /// while its row window reads a few dozen bytes. With only the row arm the
    /// shard crossed its ceiling with nothing sealed and no seal reachable — the
    /// valve then found `Drain::Stalled` and refused every put, permanently.
    /// The resident arm is what makes that shard drain instead.
    #[tokio::test]
    async fn test_pinned_parents_seal_before_the_ceiling_traps_the_writer() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let chunk = 1_000_000; // ~4MB parent per put
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            // Far above anything the one-row windows will ever sum to, so the row
            // arm cannot be what seals here.
            max_memtable_size: 1024 * 1024,
            // Never filled, so the capacity arm cannot be it either.
            max_memtable_batches: 1024,
            max_unflushed_memtable_bytes: 8 * 1024 * 1024,
            ..Default::default()
        };
        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), vec![])
            .await
            .unwrap();

        for i in 0..6i32 {
            let parent = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(
                    (0..chunk).map(|v| v + i).collect::<Vec<_>>(),
                ))],
            )
            .unwrap();
            tokio::time::timeout(STALL_GRACE * 10, writer.put(vec![parent.slice(0, 1)]))
                .await
                .expect("a shard the resident arm can seal must not park forever")
                .unwrap_or_else(|e| {
                    panic!("put {i} was refused, so the ceiling is still a trap: {e}")
                });
        }

        let stats = writer.memtable_stats().await.unwrap();
        assert!(
            stats.generation > 0,
            "the pinned parents must have sealed a generation; row bytes never \
             came close to max_memtable_size"
        );
        assert!(
            writer.memory().row_bytes() < 1024,
            "the row window must still be tiny — otherwise the row arm did the \
             sealing and this proves nothing"
        );

        writer.close().await.unwrap();
    }

    /// Index memory is not charged to `max_memtable_size`, so it is the resident
    /// arm that has to notice a memtable whose indexes rather than its rows are
    /// filling the ceiling.
    #[tokio::test]
    async fn test_resident_arm_seals_on_index_memory() {
        let schema = create_test_schema();
        let mut memtable =
            MemTable::with_capacity(schema.clone(), 1, vec![], CacheConfig::default(), 64).unwrap();
        memtable
            .insert(create_test_batch(&schema, 0, 50))
            .await
            .unwrap();

        let resident = memtable_resident_bytes(&memtable);
        let rows = memtable.batch_store().row_bytes();
        assert!(
            resident > rows,
            "the fixture needs non-row memory to be measuring anything: \
             resident {resident} vs rows {rows}"
        );

        // Row arm way out of range; only the resident arm can fire.
        assert!(
            memtable_reached_flush_threshold(&memtable, usize::MAX, resident, 1),
            "resident bytes at the ceiling must seal"
        );
        assert!(
            !memtable_reached_flush_threshold(&memtable, usize::MAX, resident + 1, 1),
            "and must not seal below it"
        );
    }

    /// The post-insert seal check runs inside the writer lock; the index apply
    /// that follows it runs outside. So a put's index growth is invisible to the
    /// only check that put makes, and the *next* put is gated by the valve before
    /// it can insert and check again — leaving a shard over its ceiling with
    /// nothing sealed and every write refused. Replay reaches the same state by
    /// building its final memtable's indexes after its last check.
    #[tokio::test]
    async fn test_index_growth_after_the_seal_check_still_drains() {
        let (store, base_path, base_uri, _t) = create_local_store().await;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        // Keys long enough to spill out of the skiplist nodes, so the index heap
        // grows with the column instead of staying a fixed reservation.
        let btree = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "text_idx".to_string(),
            field_id: 1,
            column: "text".to_string(),
        })];

        let rows = 2_000usize;
        let width = 512usize;
        let payload = rows * width;
        let ceiling = payload + payload / 2;

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            // As high as the open-time check allows, keeping the row arm out of
            // reach of the rows below so only the resident arm can seal.
            max_memtable_size: ceiling - 64 * 1024,
            max_memtable_batches: 1024,
            // Above the rows alone, below rows plus the index heap they build.
            max_unflushed_memtable_bytes: ceiling,
            ..Default::default()
        };

        let writer = ShardWriter::open(store, base_path, base_uri, config, schema.clone(), btree)
            .await
            .unwrap();

        for i in 0..3i32 {
            let ids: Vec<i32> = (0..rows as i32).map(|v| v + i * rows as i32).collect();
            let texts: Vec<String> = ids
                .iter()
                .map(|v| format!("{v:07}{}", "z".repeat(width - 7)))
                .collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(StringArray::from(
                        texts.iter().map(|t| Some(t.as_str())).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();

            tokio::time::timeout(STALL_GRACE * 10, writer.put(vec![batch]))
                .await
                .expect("a shard with a sealable memtable must not park forever")
                .unwrap_or_else(|e| {
                    panic!("put {i} was refused; index growth outran the seal check: {e}")
                });
        }

        assert!(
            writer.memory().index_bytes() > writer.memory().row_bytes(),
            "the fixture must be index-dominated, or it is not testing this path"
        );

        writer.close().await.unwrap();
    }

    /// `max_memtable_size` is the headroom the reservation check reserves for
    /// rows. At zero it reserves none, so a fresh memtable's index reservation
    /// could equal the ceiling exactly — admissible by the check, yet over budget
    /// before its first row, with an empty memtable offering nothing to seal. The
    /// writer refused its first write and never recovered.
    #[tokio::test]
    async fn test_open_rejects_a_zero_row_headroom() {
        let sizing = ShardWriterConfig {
            max_memtable_rows: 2_000,
            ..Default::default()
        };
        let reserved = reserved_index_bytes(&sizing, &hnsw_configs());
        let (store, base_path, base_uri, _t) = create_local_store().await;
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            max_memtable_rows: sizing.max_memtable_rows,
            max_memtable_size: 0,
            // Exactly the fresh reservation: the boundary the old `>` admitted.
            max_unflushed_memtable_bytes: reserved,
            ..Default::default()
        };

        let err = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            hnsw_schema(8),
            hnsw_configs(),
        )
        .await
        .err()
        .expect("a zero row headroom must be rejected at open");

        assert!(
            err.to_string()
                .contains("max_memtable_size must be greater than zero"),
            "the error must name the knob, got: {err}"
        );
    }

    /// Schema for the reservation tests: `vector` is field 1, the id 0 is `id`.
    fn hnsw_schema(dim: i32) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]))
    }

    fn hnsw_configs() -> Vec<MemIndexConfig> {
        vec![MemIndexConfig::Hnsw(Box::new(HnswIndexConfig::new(
            "vec_idx".to_string(),
            1,
            "vector".to_string(),
            lance_linalg::distance::DistanceType::L2,
        )))]
    }

    /// What the configured indexes owe before a single row arrives — the figure
    /// `open` validates against, computed the way the memtable will build them.
    fn reserved_index_bytes(config: &ShardWriterConfig, configs: &[MemIndexConfig]) -> usize {
        IndexStore::from_configs(
            configs,
            config.max_memtable_rows,
            config.max_memtable_batches,
        )
        .unwrap()
        .resident_bytes()
            + super::super::memtable::pk_bloom_filter_bytes()
    }

    /// An HNSW graph is charged from `max_memtable_rows` before the first
    /// insert, while only row bytes seal a memtable. Sized past the ceiling it
    /// would put the shard over budget at zero rows with nothing to seal and so
    /// nothing to flush — every put stalling, then failing as `Backpressure`,
    /// which means "retry later" and never comes true. That is a config error,
    /// so it has to land at `open`, not on put #1.
    #[tokio::test]
    async fn test_open_rejects_indexes_that_cannot_fit_under_the_ceiling() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            max_memtable_rows: 100_000,
            max_memtable_size: 1024 * 1024,
            max_unflushed_memtable_bytes: 1024 * 1024,
            ..Default::default()
        };
        let reserved = reserved_index_bytes(&config, &hnsw_configs());
        assert!(
            reserved > config.max_unflushed_memtable_bytes,
            "the fixture must actually over-subscribe the ceiling, got {reserved}"
        );

        let err = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            hnsw_schema(32),
            hnsw_configs(),
        )
        .await
        .err()
        .expect("an over-subscribed ceiling must be rejected at open");

        let message = err.to_string();
        for fragment in [
            "in-memory indexes reserve",
            "max_unflushed_memtable_bytes",
            "stalling every write",
        ] {
            assert!(
                message.contains(fragment),
                "the error must name {fragment}, got: {message}"
            );
        }
        assert!(
            !err.is_backpressure(),
            "a config error must not masquerade as the retryable busy signal"
        );
    }

    /// The other side of the gate: a ceiling with room for the reservation *and*
    /// a full memtable of rows on top opens, and keeps taking writes across a
    /// seal — the frozen generation gives the valve a flush to park on, so the
    /// wait ends instead of refusing.
    #[tokio::test]
    async fn test_writer_with_indexes_under_the_ceiling_keeps_accepting_writes() {
        let (store, base_path, base_uri, _temp) = create_local_store().await;
        let dim = 8;
        let sizing = ShardWriterConfig {
            max_memtable_rows: 2_000,
            ..Default::default()
        };
        let reserved = reserved_index_bytes(&sizing, &hnsw_configs());
        let max_memtable_size = 4 * 1024;

        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            max_memtable_rows: sizing.max_memtable_rows,
            max_memtable_size,
            // Two generations' worth, so a seal does not immediately re-park the
            // writer on a ceiling it cannot clear.
            max_unflushed_memtable_bytes: 2 * (reserved + max_memtable_size),
            ..Default::default()
        };

        let schema = hnsw_schema(dim);
        let writer = ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            schema.clone(),
            hnsw_configs(),
        )
        .await
        .expect("a reservation that leaves room under the ceiling must open");

        // Enough rows to carry row bytes past `max_memtable_size` several times
        // over, so the run spans seals rather than sitting in one memtable.
        for round in 0..8i32 {
            let rows = 64;
            let ids: Vec<i32> = (0..rows).map(|i| round * rows + i).collect();
            let values: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
            let vectors = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim,
                Arc::new(Float32Array::from(values)),
                None,
            )
            .unwrap();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(vectors)],
            )
            .unwrap();

            tokio::time::timeout(STALL_GRACE * 10, writer.put(vec![batch]))
                .await
                .expect("a drainable shard must not park the writer indefinitely")
                .unwrap_or_else(|e| panic!("put in round {round} was refused: {e}"));
        }

        // Without a seal this would only prove one memtable fits, which is not
        // the case that stalls.
        assert!(
            writer.memtable_stats().await.unwrap().generation > 0,
            "the run must cross a seal for the drain path to have been exercised"
        );

        writer.close().await.unwrap();
    }
}

#[cfg(test)]
mod shard_writer_tests {
    use std::sync::Arc;

    use crate::index::DatasetIndexExt;
    use arrow_array::{
        FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::IndexType;
    use lance_index::scalar::inverted::{InvertedIndexParams, InvertedListFormatVersion};
    use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::builder::PQBuildParams;
    use lance_linalg::distance::MetricType;
    use uuid::Uuid;

    use crate::dataset::mem_wal::DatasetMemWalExt;
    use crate::dataset::{Dataset, WriteParams};
    use crate::index::vector::VectorIndexParams;

    use super::super::ShardWriterConfig;

    fn create_test_schema(vector_dim: i32) -> Arc<ArrowSchema> {
        use std::collections::HashMap;

        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_append_only_schema(vector_dim: i32) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_dim,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(
        schema: &ArrowSchema,
        start_id: i64,
        num_rows: usize,
        vector_dim: i32,
    ) -> RecordBatch {
        let vectors: Vec<f32> = (0..num_rows)
            .flat_map(|i| {
                let seed = (start_id as usize + i) as f32;
                (0..vector_dim as usize).map(move |d| (seed * 0.1 + d as f32 * 0.01).sin())
            })
            .collect();

        let vector_array =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vectors), vector_dim)
                .unwrap();

        let texts: Vec<String> = (0..num_rows)
            .map(|i| format!("Sample text for row {}", start_id as usize + i))
            .collect();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    start_id..start_id + num_rows as i64,
                )),
                Arc::new(vector_array),
                Arc::new(StringArray::from_iter_values(texts)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_records_writer_config_defaults() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_writer_config_defaults_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        let writer_config = ShardWriterConfig::default()
            .with_durable_write(false)
            .with_max_memtable_size(8 * 1024 * 1024);

        dataset
            .initialize_mem_wal()
            .writer_config_defaults(writer_config)
            .add_writer_config_default("custom_knob", "custom_value")
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Defaults must survive the manifest round-trip so all writers share them.
        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        let defaults = &details.writer_config_defaults;
        // ShardWriterConfig tunables are recorded under their field names.
        assert_eq!(
            defaults.get("durable_write").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            defaults.get("max_memtable_size").map(String::as_str),
            Some("8388608")
        );
        // Duration knobs are recorded in milliseconds with a `_ms` suffix.
        assert_eq!(
            defaults
                .get("max_wal_flush_interval_ms")
                .map(String::as_str),
            Some("100")
        );
        // Every tunable field is present.
        assert!(defaults.contains_key("enable_memtable"));
        // add_writer_config_default records arbitrary keys.
        assert_eq!(
            defaults.get("custom_knob").map(String::as_str),
            Some("custom_value")
        );
        // Shard identity is not a configuration default.
        assert!(!defaults.contains_key("shard_id"));
        assert!(!defaults.contains_key("shard_spec_id"));
    }

    /// A maintained index can be split across multiple physical segments once a
    /// delta is appended over previously uncovered fragments (the distributed
    /// indexer / `optimize_indices(append)` flow). `mem_wal_writer` must resolve
    /// such an index by name without tripping the singular loader's "multiple
    /// indices of the same name" error — it only reads the shared type/params,
    /// which every segment carries identically.
    #[tokio::test]
    async fn test_mem_wal_writer_with_multi_segment_index() {
        use lance_index::optimize::OptimizeOptions;

        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        // The generation flusher reopens by URI, so this independent open must
        // resolve to the same in-memory backend. The unique authority isolates the test.
        let uri = format!(
            "shared-memory://multi-segment-index-{}/",
            Uuid::new_v4().simple()
        );

        // Initial fragment + an IVF vector index covering it.
        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create vector index");

        // Append a second fragment and index it as a *delta* (no merge), so the
        // index ends up with two physical segments sharing the name "vector_idx".
        let appended = create_test_batch(&schema, 256, 256, vector_dim);
        let append_batches = RecordBatchIterator::new([Ok(appended)], schema.clone());
        dataset
            .append(append_batches, None)
            .await
            .expect("Failed to append fragment");
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .expect("Failed to append index delta");

        // Precondition: the index is genuinely multi-segment, so the singular
        // `load_index_by_name` would error here.
        assert_eq!(
            dataset
                .load_indices_by_name("vector_idx")
                .await
                .unwrap()
                .len(),
            2,
            "expected two physical segments for the maintained index"
        );

        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // The regression: loading the multi-segment maintained index must succeed.
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id).with_durable_write(false),
            )
            .await
            .expect("mem_wal_writer must accept a multi-segment maintained index");

        // And the resulting writer is functional.
        writer
            .put(vec![create_test_batch(&schema, 200, 10, vector_dim)])
            .await
            .unwrap();
        assert_eq!(writer.memtable_stats().await.unwrap().row_count, 10);
        writer.close().await.unwrap();
    }

    /// A generation of only tombstones (delete nulls the vector, and the HNSW
    /// skips nulls) has an empty vector index. Flushing it must succeed —
    /// writing the data with no HNSW index for that generation — not fail the
    /// drain with "HnswMemIndex is empty". Regression for the wallop fuzz
    /// `flush drain failed: HnswMemIndex is empty` on a delete-only memtable.
    #[tokio::test]
    async fn test_flush_all_tombstone_generation_skips_empty_hnsw() {
        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        // `shared-memory` (not `memory`): the flush writes the generation dataset
        // and the drain reopens it by URI, which a plain in-memory store wouldn't
        // share across opens. A unique authority isolates this test.
        let uri = format!("shared-memory://tombstone-{}/", Uuid::new_v4().simple());

        // Base dataset + maintained vector index, so the writer carries an HNSW
        // mem index.
        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .unwrap();

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id).with_durable_write(false),
            )
            .await
            .unwrap();

        // Delete only — the memtable holds a single tombstone with a null vector,
        // so its HNSW index is empty.
        let keys = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![0_i64]))],
        )
        .unwrap();
        writer.delete(vec![keys]).await.unwrap();

        // The drain must not error on the empty HNSW.
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        // The tombstone-only generation still flushed (data without an HNSW index).
        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert_eq!(
            manifest.sstables.len(),
            1,
            "the all-tombstone generation must still flush"
        );
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_mem_wal_maintained_fts_v1_flush_preserves_format() {
        use tempfile::TempDir;

        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let uri = format!("file://{}", temp_dir.path().display());

        let initial = create_test_batch(&schema, 0, 16, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        let fts_params =
            InvertedIndexParams::default().format_version(InvertedListFormatVersion::V1);
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &fts_params,
                false,
            )
            .await
            .expect("Failed to create v1 FTS index");
        let base_indices = dataset.load_indices().await.unwrap();
        assert_eq!(base_indices.len(), 1);
        assert_eq!(base_indices[0].index_version, 1);

        dataset
            .initialize_mem_wal()
            .maintained_indexes(["text_fts"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id).with_durable_write(true);
        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create MemWAL writer");
        writer
            .put(vec![create_test_batch(&schema, 1_000, 3, vector_dim)])
            .await
            .expect("Failed to write MemWAL batch");
        writer.close().await.expect("Failed to close writer");

        let (store, base_path) = lance_io::object_store::ObjectStore::from_uri(&uri)
            .await
            .expect("Failed to open store");
        let manifest_store =
            super::super::manifest::ShardManifestStore::new(store, &base_path, shard_id, 2);
        let manifest = manifest_store
            .latest()
            .await
            .expect("Failed to read manifest")
            .expect("Manifest should exist");
        assert_eq!(manifest.sstables.len(), 1);

        let sstable = &manifest.sstables[0];
        let gen_uri = format!("{}/_mem_wal/{}/{}", uri, shard_id, sstable.path);
        let sstable = Dataset::open(&gen_uri)
            .await
            .expect("Failed to open SSTable");
        let sstable_indices = sstable.load_indices().await.unwrap();
        assert_eq!(sstable_indices.len(), 1);
        assert_eq!(sstable_indices[0].name, "text_fts");
        assert_eq!(
            sstable_indices[0].index_version, 1,
            "maintained v1 FTS index must flush as v1"
        );

        let results = sstable
            .scan()
            .full_text_search(FullTextSearchQuery::new("Sample".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 3);
    }

    #[tokio::test]
    async fn test_writer_hnsw_params_override() {
        use lance_index::vector::hnsw::builder::HnswBuildParams;

        let vector_dim = 32;
        let schema = create_test_schema(vector_dim);
        // The generation flusher reopens by URI, so this independent open must
        // resolve to the same in-memory backend. The unique authority isolates the test.
        let uri = format!(
            "shared-memory://writer-hnsw-params-{}/",
            Uuid::new_v4().simple()
        );

        let initial = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create vector index");

        // Persisting a ShardWriterConfig that carries HNSW params records them
        // in the writer-config defaults map under `hnsw.<index>.<field>` keys.
        let configured = ShardWriterConfig::new(Uuid::new_v4()).with_hnsw_params(
            "vector_idx",
            HnswBuildParams::default().num_edges(7).ef_construction(48),
        );
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .writer_config_defaults(configured)
            .execute()
            .await
            .expect("Failed to initialize MemWAL");
        let defaults = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("MemWAL details should exist")
            .writer_config_defaults;
        assert_eq!(
            defaults
                .get("hnsw.vector_idx.num_edges")
                .map(String::as_str),
            Some("7")
        );
        assert_eq!(
            defaults
                .get("hnsw.vector_idx.ef_construction")
                .map(String::as_str),
            Some("48")
        );

        // The writer's per-index HNSW override flows into the in-memory index.
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(false)
            .with_hnsw_params(
                "vector_idx",
                HnswBuildParams::default().num_edges(7).ef_construction(48),
            );
        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("mem_wal_writer must accept the HNSW-param override");
        writer
            .put(vec![create_test_batch(&schema, 300, 10, vector_dim)])
            .await
            .unwrap();

        let active = writer.active_memtable_ref().await.unwrap();
        let hnsw = active
            .index_store
            .get_hnsw("vector_idx")
            .expect("maintained HNSW index should exist in the memtable");
        assert_eq!(hnsw.build_params().m, 7);
        assert_eq!(hnsw.build_params().ef_construction, 48);
        drop(active);

        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_bucket_sharding() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_bucket_sharding_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // num_buckets out of range is rejected.
        let result = dataset
            .initialize_mem_wal()
            .bucket_sharding("id", 0)
            .execute()
            .await;
        assert!(result.is_err(), "num_buckets = 0 should be rejected");

        dataset
            .initialize_mem_wal()
            .bucket_sharding("text", 8)
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 8);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("bucket"));
        assert_eq!(
            field.parameters.get("num_buckets").map(String::as_str),
            Some("8")
        );
        assert_eq!(field.source_ids.len(), 1);
        let source_id = field.source_ids[0];
        let source_field = dataset.schema().field("text").expect("text field exists");
        assert_eq!(source_id, source_field.id);
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_bucket_sharding_without_primary_key() {
        let vector_dim = 128;
        let schema = create_append_only_schema(vector_dim);
        let uri = format!(
            "memory://test_bucket_sharding_no_primary_key_{}",
            Uuid::new_v4()
        );

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        dataset
            .initialize_mem_wal()
            .bucket_sharding("id", 8)
            .execute()
            .await
            .expect("Failed to initialize append-only MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 8);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("bucket"));
        assert_eq!(
            field.parameters.get("num_buckets").map(String::as_str),
            Some("8")
        );
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_unsharded() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_unsharded_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        assert_eq!(details.num_shards, 1);
        assert_eq!(details.sharding_specs.len(), 1);
        assert_eq!(
            details.sharding_specs[0].fields[0].transform.as_deref(),
            Some("unsharded")
        );
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_identity_sharding() {
        let vector_dim = 128;
        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_identity_sharding_{}", Uuid::new_v4());

        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // A column that does not exist is rejected.
        let result = dataset
            .initialize_mem_wal()
            .identity_sharding("nonexistent")
            .execute()
            .await;
        assert!(
            result.is_err(),
            "an unknown identity column should be rejected"
        );

        // A non-scalar column cannot be a shard key.
        let result = dataset
            .initialize_mem_wal()
            .identity_sharding("vector")
            .execute()
            .await;
        assert!(
            result.is_err(),
            "a non-scalar identity column should be rejected"
        );

        dataset
            .initialize_mem_wal()
            .identity_sharding("text")
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let details = dataset
            .mem_wal_index_details()
            .await
            .expect("Failed to read MemWAL index details")
            .expect("MemWAL index details should exist");

        // Identity sharding has an open-ended shard count.
        assert_eq!(details.num_shards, 0);
        assert_eq!(details.sharding_specs.len(), 1);
        let field = &details.sharding_specs[0].fields[0];
        assert_eq!(field.transform.as_deref(), Some("identity"));
        assert_eq!(field.result_type.as_str(), "utf8");
        assert_eq!(field.source_ids.len(), 1);
    }

    /// Quick smoke test for shard writer - runs against memory://
    /// Run with: cargo test -p lance shard_writer_tests::test_shard_writer_smoke -- --nocapture
    #[tokio::test]
    async fn test_shard_writer_smoke() {
        let vector_dim = 128;
        let batch_size = 20;
        let num_batches = 100;

        let schema = create_test_schema(vector_dim);
        let uri = format!("memory://test_shard_writer_{}", Uuid::new_v4());

        // Create initial dataset
        let initial_batch = create_test_batch(&schema, 0, 100, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Initialize MemWAL (no indexes for smoke test)
        dataset
            .initialize_mem_wal()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id).with_durable_write(false);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        // Pre-generate batches
        let batches: Vec<RecordBatch> = (0..num_batches)
            .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
            .collect();

        // Write all batches in a single put call for efficiency
        writer.put(batches).await.expect("Failed to write");

        writer.close().await.expect("Failed to close");
    }

    #[tokio::test]
    async fn test_shard_writer_with_vector_index_searches_active_memtable() {
        let vector_dim = 32;
        let batch_size = 20;
        let target_id = 1_000i64 + 37;

        let schema = create_test_schema(vector_dim);
        // The generation flusher reopens by URI, so this independent open must
        // resolve to the same in-memory backend. The unique authority isolates the test.
        let uri = format!(
            "shared-memory://shard-writer-hnsw-{}/",
            Uuid::new_v4().simple()
        );

        let initial_batch = create_test_batch(&schema, 0, 256, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create base vector index");

        dataset
            .initialize_mem_wal()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id).with_durable_write(true);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        let batches: Vec<RecordBatch> = (0..4)
            .map(|i| {
                create_test_batch(
                    &schema,
                    1_000 + (i * batch_size) as i64,
                    batch_size,
                    vector_dim,
                )
            })
            .collect();
        writer.put(batches).await.expect("Failed to write");

        let query = Float32Array::from_iter_values(
            (0..vector_dim as usize).map(|d| (target_id as f32 * 0.1 + d as f32 * 0.01).sin()),
        );
        let mut scanner = writer.scan().await.unwrap();
        scanner.nearest("vector", &query, 80).unwrap();
        let result = scanner.try_into_batch().await.expect("Failed to scan");

        assert!(result.num_rows() > 0, "vector query returned no rows");
        let id_col = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let dist_col = result
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let target_idx = (0..result.num_rows())
            .find(|&idx| id_col.value(idx) == target_id)
            .expect("target vector was not returned by active MemTable HNSW search");
        assert!(
            dist_col.value(target_idx) < 1e-6,
            "expected self-match distance near zero, got {}",
            dist_col.value(target_idx)
        );

        writer.close().await.expect("Failed to close");
    }

    /// Test shard writer against S3 with IVF-PQ, BTree, and FTS indexes (requires DATASET_PREFIX env var)
    /// Run with: DATASET_PREFIX=s3://bucket/path cargo test -p lance --release shard_writer_tests::test_shard_writer_s3_ivfpq -- --nocapture --ignored
    #[tokio::test]
    #[ignore]
    async fn test_shard_writer_s3_ivfpq() {
        let prefix = std::env::var("DATASET_PREFIX").expect("DATASET_PREFIX not set");

        let vector_dim = 512;
        let batch_size = 20;
        let num_batches = 10000;
        let num_partitions = 16;
        let num_sub_vectors = 64; // 512 / 8 = 64 subvectors

        let schema = create_test_schema(vector_dim);
        let uri = format!(
            "{}/test_s3_{}",
            prefix.trim_end_matches('/'),
            Uuid::new_v4()
        );

        // Create initial dataset with enough data for IVF-PQ training
        let initial_batch = create_test_batch(&schema, 0, 1000, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Create BTree index on id column
        let scalar_params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &scalar_params,
                false,
            )
            .await
            .expect("Failed to create BTree index");

        // Create FTS index on text column
        let fts_params = InvertedIndexParams::default();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &fts_params,
                false,
            )
            .await
            .expect("Failed to create FTS index");

        // Create IVF-PQ index on dataset

        let ivf_params = IvfBuildParams {
            num_partitions: Some(num_partitions),
            ..Default::default()
        };
        let pq_params = PQBuildParams {
            num_sub_vectors,
            num_bits: 8,
            ..Default::default()
        };
        let vector_params =
            VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .expect("Failed to create IVF-PQ index");

        // Initialize MemWAL with all three indexes
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["id_btree", "text_fts", "vector_idx"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer with default config
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id).with_durable_write(false);

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        // Pre-generate batches
        let batches: Vec<RecordBatch> = (0..num_batches)
            .map(|i| create_test_batch(&schema, (i * batch_size) as i64, batch_size, vector_dim))
            .collect();

        // Write all batches in a single put call for efficiency
        writer.put(batches).await.expect("Failed to write");

        writer.close().await.expect("Failed to close");
    }

    /// End-to-end correctness test for ShardWriter with multiple memtable flushes.
    ///
    /// This test verifies:
    /// 1. Multiple memtable flushes are triggered via small memtable size
    /// 2. File system layout is correct (WAL files, manifest, generation directories)
    /// 3. WAL entries contain expected data
    /// 4. Data can be read after each flush cycle
    /// 5. Manifest tracks SSTables correctly
    ///
    /// Run with: cargo test -p lance shard_writer_tests::test_shard_writer_e2e_correctness -- --nocapture
    #[tokio::test]
    async fn test_shard_writer_e2e_correctness() {
        use std::time::Duration;
        use tempfile::TempDir;

        let vector_dim = 32;
        let rows_per_batch = 50;
        // Write enough to trigger ~3 memtable flushes with 50KB memtable size
        // Each batch is ~6KB (50 rows * 32 dims * 4 bytes/float + overhead)
        let num_write_rounds = 3;
        let batches_per_round = 3;

        // Create temp directory for the test
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let uri = format!("file://{}", temp_dir.path().display());

        let schema = create_test_schema(vector_dim);

        // Create initial dataset with enough rows for IVF-PQ training
        let initial_batch = create_test_batch(&schema, 0, 500, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial_batch)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Create BTree index
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .expect("Failed to create BTree index");

        // Initialize MemWAL with BTree index only (simpler for this test)
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["id_btree"])
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Create shard writer with small memtable size to trigger flushes
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_durable_write(true) // Ensure WAL files are written
            .with_max_memtable_size(50 * 1024) // 50KB - triggers flush after ~8 batches
            .with_max_wal_buffer_size(10 * 1024) // 10KB WAL buffer
            .with_max_wal_flush_interval(Duration::from_millis(50)); // Fast flush

        let writer = dataset
            .mem_wal_writer(shard_id, config)
            .await
            .expect("Failed to create writer");

        let mut total_rows_written = 0i64;

        // Write data in rounds
        for _round in 0..num_write_rounds {
            let start_id = 500 + total_rows_written;
            let batches_to_write: Vec<RecordBatch> = (0..batches_per_round)
                .map(|i| {
                    create_test_batch(
                        &schema,
                        start_id + (i * rows_per_batch) as i64,
                        rows_per_batch,
                        vector_dim,
                    )
                })
                .collect();

            writer.put(batches_to_write).await.expect("Failed to write");

            total_rows_written += (batches_per_round * rows_per_batch) as i64;

            // Give time for WAL flush and potential memtable flush
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        // Close writer to ensure final flush
        writer.close().await.expect("Failed to close");

        // === VERIFY FILE SYSTEM LAYOUT ===
        let mem_wal_dir = temp_dir.path().join("_mem_wal").join(shard_id.to_string());
        assert!(mem_wal_dir.exists(), "MemWAL directory should exist");

        // Check WAL directory
        let wal_dir = mem_wal_dir.join("wal");
        assert!(wal_dir.exists(), "WAL directory should exist");
        let wal_files: Vec<_> = std::fs::read_dir(&wal_dir)
            .expect("Failed to read WAL dir")
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !wal_files.is_empty(),
            "WAL directory should contain at least one file"
        );

        // Check manifest directory
        let manifest_dir = mem_wal_dir.join("manifest");
        assert!(manifest_dir.exists(), "Manifest directory should exist");
        let manifest_files: Vec<_> = std::fs::read_dir(&manifest_dir)
            .expect("Failed to read manifest dir")
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            !manifest_files.is_empty(),
            "Manifest directory should contain at least one file"
        );

        // Read and verify manifest
        let (store, base_path) = lance_io::object_store::ObjectStore::from_uri(&uri)
            .await
            .expect("Failed to open store");
        let manifest_store =
            super::super::manifest::ShardManifestStore::new(store, &base_path, shard_id, 2);
        let manifest = manifest_store
            .latest()
            .await
            .expect("Failed to read manifest")
            .expect("Manifest should exist");

        // Verify SSTables exist on disk
        assert!(
            !manifest.sstables.is_empty(),
            "Should have at least one SSTable"
        );
        for sstable in &manifest.sstables {
            // The path stored in manifest is relative to the shard directory
            // Construct full path: temp_dir/_mem_wal/shard_id/generation_folder
            let gen_path = temp_dir
                .path()
                .join("_mem_wal")
                .join(shard_id.to_string())
                .join(&sstable.path);

            // The generation directory should exist
            assert!(
                gen_path.exists(),
                "SSTable directory should exist at {:?}",
                gen_path
            );

            // Verify generation directory has files
            let gen_contents_count = std::fs::read_dir(&gen_path)
                .expect("Failed to read gen dir")
                .filter_map(|e| e.ok())
                .count();
            assert!(
                gen_contents_count > 0,
                "Generation directory should have files"
            );
        }

        // === VERIFY WAL ENTRIES ===
        // Verify WAL files have correct extension
        for wal_file in wal_files.iter().take(1) {
            let wal_path = wal_file.path();
            let file_name = wal_path.file_name().unwrap().to_string_lossy();
            assert!(
                file_name.ends_with(".arrow"),
                "WAL file should have .arrow extension"
            );
        }

        // === VERIFY DATA CAN BE READ FROM NEW WRITER ===
        // Re-open dataset and create new writer to verify recovery
        let dataset = Dataset::open(&uri).await.expect("Failed to reopen dataset");
        let new_shard_id = Uuid::new_v4();
        // `durable_write(true)` so the put waits for its flush, which is what
        // currently publishes the rows. A non-durable put is *not* yet
        // read-your-writes: the index apply is welded to the WAL flush, so the
        // rows stay invisible until the next flush. This test used to pass with
        // `durable_write(false)` only because an un-advanced cursor of 0 was
        // misread as "batch 0 is visible" — it was asserting the dirty read.
        let new_config = ShardWriterConfig::new(new_shard_id).with_durable_write(true);

        let new_writer = dataset
            .mem_wal_writer(new_shard_id, new_config)
            .await
            .expect("Failed to create new writer");

        // Write a test batch to verify the new shard works
        let verify_batch = create_test_batch(&schema, 10000, 10, vector_dim);
        new_writer
            .put(vec![verify_batch])
            .await
            .expect("Failed to write to new shard");

        let scanner = new_writer.scan().await.unwrap();
        let result = scanner.try_into_batch().await.expect("Failed to scan");
        assert_eq!(result.num_rows(), 10, "New shard should have 10 rows");

        new_writer
            .close()
            .await
            .expect("Failed to close new writer");
    }

    /// Regression: a base opened with a *path-bound* store binding (the
    /// deprecated `ObjectStoreParams::object_store`) must still flush and read
    /// generations at their own paths.
    ///
    /// The binding pins a store to one location, and both
    /// `ObjectStore::from_uri_and_params` and `DatasetBuilder::build_object_store`
    /// take the path from it while ignoring the URI they were handed. Reusing the
    /// base's params verbatim therefore aimed every generation write and open at
    /// the base table itself: the flush failed ("dataset already exists") and any
    /// derived open returned base rows as generation rows.
    #[tokio::test]
    async fn test_flush_and_read_with_path_bound_object_store() {
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use futures::TryStreamExt;
        use lance_io::object_store::ObjectStoreParams;
        use tempfile::TempDir;

        let vector_dim = 8;
        let schema = create_test_schema(vector_dim);
        let temp_dir = TempDir::new().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());

        let initial = create_test_batch(&schema, 0, 16, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        dataset
            .initialize_mem_wal()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        // Re-bind the base to a store pinned at the base's own path — what
        // `DatasetBuilder::with_object_store` leaves on an opened dataset.
        #[allow(deprecated)]
        let store_params = ObjectStoreParams {
            object_store: Some((
                Arc::new(object_store::local::LocalFileSystem::new()),
                url::Url::parse(&uri).unwrap(),
            )),
            ..Default::default()
        };
        let dataset = dataset.with_object_store(dataset.object_store.clone(), Some(store_params));

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .expect("Failed to create writer");
        writer
            .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
            .await
            .expect("Failed to write");
        writer.force_seal_active().await.unwrap();
        writer
            .wait_for_flush_drain()
            .await
            .expect("flush must not be redirected at the base table");

        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert_eq!(manifest.sstables.len(), 1);
        let sstable = manifest.sstables[0].clone();

        // The generation landed under `_mem_wal/`, and the base table is untouched.
        let gen_uri = format!("{}/_mem_wal/{}/{}", uri, shard_id, sstable.path);
        let generation = Dataset::open(&gen_uri)
            .await
            .expect("generation must exist at its own path");
        assert_eq!(generation.count_rows(None).await.unwrap(), 8);
        let base = Dataset::open(&uri).await.unwrap();
        assert_eq!(
            base.count_rows(None).await.unwrap(),
            16,
            "the generation write must not land in the base table"
        );

        // The read path resolves the generation, not the base: 16 base + 8 flushed.
        // Opening the base instead would dedup back down to 16 rows.
        let snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(manifest.current_generation)
            .with_sstable(sstable.generation, sstable.path.clone());
        let scanner = LsmScanner::new(Arc::new(dataset), vec![snapshot], vec!["id".to_string()]);
        let rows: usize = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .expect("scan must open the generation, not the base")
            .iter()
            .map(|batch| batch.num_rows())
            .sum();
        assert_eq!(rows, 24);

        writer.close().await.unwrap();
    }

    /// The store params a base was opened with must reach every *derived* open:
    /// the flush that writes a generation and the scan that reads it back.
    ///
    /// This is the point of threading them at all. A namespace-vended store
    /// exists only on the params (credentials, endpoint, wrapper), so a
    /// generation resolved by URI alone would silently sign with the ambient
    /// identity instead — succeeding against a local store and failing against
    /// the vended one. Asserting on the *generation folder* rather than
    /// `_mem_wal/` is what makes this bite: WAL entries are written through the
    /// base dataset's own store, so they would show up here either way.
    #[tokio::test]
    async fn test_store_params_reach_generation_write_and_read() {
        use crate::dataset::builder::DatasetBuilder;
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use crate::dataset::mem_wal::test_util::observable_store_params;
        use futures::TryStreamExt;
        use tempfile::TempDir;

        let vector_dim = 8;
        let schema = create_test_schema(vector_dim);
        let temp_dir = TempDir::new().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());

        let initial = create_test_batch(&schema, 0, 16, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");

        // Open the base through an observable store, exactly as a namespace
        // client would hand in a vended-credential store.
        let (store_params, controls) = observable_store_params();
        let mut dataset = DatasetBuilder::from_uri(&uri)
            .with_store_params(store_params)
            .load()
            .await
            .expect("Failed to open dataset");
        dataset
            .initialize_mem_wal()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .expect("Failed to create writer");
        writer
            .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
            .await
            .expect("Failed to write");
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.expect("flush failed");

        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        assert_eq!(manifest.sstables.len(), 1);
        let sstable = manifest.sstables[0].clone();

        // The generation's own Lance manifest is the signal to key on. Keying on
        // the generation folder alone would pass vacuously: sidecars like
        // `{gen}/bloom_filter.bin` are written through the *base* dataset's
        // store, which is observable no matter what the params do. And the
        // fragments can't be used either — `ObjectStore::create` writes local
        // files through `tokio::fs`, bypassing the object store entirely, so
        // `{gen}/data/` never reaches a wrapper under `file://`. The manifest
        // goes through `put_opts`, and only the flusher's `Dataset::write` /
        // `open_generation` writes it — both of which must carry the params.
        let gen_manifest = format!("{}/_versions", sstable.path);

        assert!(
            controls.wrote_under(&gen_manifest),
            "the flush must write the generation through the base's store params, \
             not a store resolved from the generation URI alone"
        );

        // And the read path must resolve the generation through them too.
        let snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(manifest.current_generation)
            .with_sstable(sstable.generation, sstable.path.clone());
        let scanner = LsmScanner::new(Arc::new(dataset), vec![snapshot], vec!["id".to_string()]);
        let rows: usize = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .expect("scan failed")
            .iter()
            .map(|batch| batch.num_rows())
            .sum();
        assert_eq!(rows, 24);

        // Reads key on the data files, not the manifest: the flusher already
        // pulled the generation's manifest into the shared session cache, so the
        // scan's open serves it from memory and never touches the store. The
        // fragments are read through it (reads have no local bypass), as is the
        // generation's standalone PK index.
        assert!(
            controls.read_under(&format!("{}/data/", sstable.path)),
            "the scan must read the generation through the base's store params"
        );

        writer.close().await.unwrap();
    }

    /// A fresh-tier-only scanner reaches its store params through
    /// `with_store_params`, not `new()`, so the setter must strip the path-bound
    /// store binding too. Left raw, it redirects the generation open at the base
    /// table and the scan silently returns base rows as WAL rows.
    #[tokio::test]
    async fn test_fresh_tier_scan_with_path_bound_object_store() {
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use futures::TryStreamExt;
        use lance_io::object_store::ObjectStoreParams;
        use tempfile::TempDir;

        let vector_dim = 8;
        let schema = create_test_schema(vector_dim);
        let temp_dir = TempDir::new().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());

        // 16 base rows with ids 0..16; the WAL gets 8 rows with ids 1000..1008,
        // so a redirected generation open is unambiguous in the output.
        let initial = create_test_batch(&schema, 0, 16, vector_dim);
        let batches = RecordBatchIterator::new([Ok(initial)], schema.clone());
        let mut dataset = Dataset::write(batches, &uri, Some(WriteParams::default()))
            .await
            .expect("Failed to create dataset");
        dataset
            .initialize_mem_wal()
            .execute()
            .await
            .expect("Failed to initialize MemWAL");

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .expect("Failed to create writer");
        writer
            .put(vec![create_test_batch(&schema, 1_000, 8, vector_dim)])
            .await
            .expect("Failed to write");
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.expect("flush failed");

        let manifest = writer.manifest().await.unwrap().expect("manifest exists");
        let sstable = manifest.sstables[0].clone();
        let snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(manifest.current_generation)
            .with_sstable(sstable.generation, sstable.path.clone());

        // What `DatasetBuilder::with_object_store` leaves on an opened dataset:
        // a store pinned at the base's own path.
        #[allow(deprecated)]
        let store_params = ObjectStoreParams {
            object_store: Some((
                Arc::new(object_store::local::LocalFileSystem::new()),
                url::Url::parse(&uri).unwrap(),
            )),
            ..Default::default()
        };

        let arrow_schema: Arc<ArrowSchema> = schema.clone();
        let batches = LsmScanner::without_base_table(
            arrow_schema,
            uri.clone(),
            vec![snapshot],
            vec!["id".to_string()],
        )
        .with_session(dataset.session())
        .with_store_params(store_params)
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .expect("scan must open the generation, not the base");

        let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(
            rows, 8,
            "fresh tier holds only the 8 WAL rows; 16 means the generation open \
             was redirected at the base table"
        );
        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(
            ids.iter().all(|id| (1_000..1_008).contains(id)),
            "expected the WAL's own rows, got {ids:?}"
        );

        writer.close().await.unwrap();
    }
}
