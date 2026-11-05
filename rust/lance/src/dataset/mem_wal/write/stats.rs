// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL write, WAL, and memtable statistics.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// MemTable statistics: rows, generation, and what the memtable still owes the
/// WAL.
///
/// Deliberately carries **no byte totals**.
/// [`ShardWriter::memory`](super::writer::ShardWriter::memory) is the one
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
    /// Frozen memtables in the read view: frozen-awaiting-flush, plus flushed
    /// ones still inside `frozen_memtable_grace`.
    pub frozen_count: usize,
}

/// WAL statistics.
#[derive(Debug, Clone)]
pub struct WalStats {
    /// Next WAL entry position to be used.
    pub next_wal_entry_position: u64,
}

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
    memtable_flush_failures: AtomicU64,
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
    /// Failed MemTable flush attempts, each retry counted separately.
    pub memtable_flush_failures: u64,
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

    /// Record a failed MemTable flush attempt.
    pub fn record_memtable_flush_failure(&self) {
        self.memtable_flush_failures.fetch_add(1, Ordering::Relaxed);
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
            memtable_flush_failures: self.memtable_flush_failures.load(Ordering::Relaxed),
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
        self.memtable_flush_failures.store(0, Ordering::Relaxed);
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
mod tests;
