// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shard writer configuration and its builder methods.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lance_core::Result;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_io::object_store::ObjectStoreParams;
use uuid::Uuid;

use super::backpressure::BackpressureController;
use crate::dataset::mem_wal::index::{MemIndexPlugin, MemIndexRegistry};
use crate::dataset::mem_wal::observer::WalObserver;
use crate::dataset::mem_wal::scanner::SsTableWarmer;
use crate::session::Session;

/// Configuration for a shard writer.
#[derive(Debug, Clone)]
pub struct ShardWriterConfig {
    /// Unique identifier for this shard (UUID v4).
    pub shard_id: Uuid,

    /// The MemWAL's `dir_id`; this writer's WAL, manifests, and generations live
    /// under its directory. Empty for the legacy `_mem_wal`.
    pub dir_id: String,

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
    /// Sizes the in-memory index pre-allocation. The memtable freezes before a
    /// write that would carry it past this, and a single write larger than the
    /// cap is rejected.
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
    /// - `index_specs` must be empty (validated at open time).
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
    /// may use different parameters. An index with no entry here or in
    /// [`index_overrides`](Self::index_overrides) uses the default build
    /// parameters. `num_edges` is the HNSW graph degree (level 0 retains
    /// `2 * num_edges`), equivalent to FAISS's `M`.
    ///
    /// Default: empty.
    pub hnsw_params: HashMap<String, HnswBuildParams>,

    /// Settings for the plugin maintaining an index, by index name, as the
    /// type the plugin reads through
    /// [`ResolveContext::overrides`](crate::dataset::mem_wal::index::ResolveContext::overrides).
    /// Naming an index here and in [`hnsw_params`](Self::hnsw_params) is an
    /// error. Each writer passes its own: these are not recorded as writer
    /// defaults.
    ///
    /// Default: empty.
    pub index_overrides: HashMap<String, Arc<dyn std::any::Any + Send + Sync>>,

    /// The memtable index plugins this writer can maintain. Defaults to the
    /// kinds Lance builds in.
    pub mem_index_registry: MemIndexRegistry,

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
    /// per-shard valve ([`super::backpressure::LocalBackpressureController`]).
    ///
    /// For embedders whose budgets lance cannot see: a process-wide memtable
    /// total across shards, a page-cache working set. Because it replaces
    /// rather than layers, the injected controller owns the whole policy — a
    /// per-shard ceiling included, for which it is handed
    /// [`ShardMemory`](super::backpressure::ShardMemory).
    /// Unlike the built-in valve it may also reject: see
    /// [`Error::Backpressure`](lance_core::Error::Backpressure).
    /// Default: `None` (use the built-in valve).
    pub backpressure: Option<Arc<dyn BackpressureController>>,
}

impl Default for ShardWriterConfig {
    fn default() -> Self {
        Self {
            shard_id: Uuid::new_v4(),
            dir_id: String::new(),
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
            index_overrides: HashMap::new(),
            mem_index_registry: MemIndexRegistry::default(),
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

    /// Register a memtable index plugin. Fails if another plugin already
    /// maintains the same base-table index.
    pub fn with_mem_index_plugin(mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<Self> {
        self.mem_index_registry.add_plugin(plugin)?;
        Ok(self)
    }

    /// The plugins this writer maintains indexes with.
    pub fn with_mem_index_registry(mut self, registry: MemIndexRegistry) -> Self {
        self.mem_index_registry = registry;
        self
    }

    /// Give the plugin maintaining `index_name` these settings, replacing any
    /// given before.
    pub fn with_index_override(
        mut self,
        index_name: impl Into<String>,
        settings: impl std::any::Any + Send + Sync,
    ) -> Self {
        self.index_overrides
            .insert(index_name.into(), Arc::new(settings));
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
