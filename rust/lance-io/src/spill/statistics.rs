// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Counters do not own files, quotas, memory pools, or cleanup jobs.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// A logical scratch-usage snapshot, including pending I/O and deferred cleanup.
/// These are accounted file bytes, not filesystem allocation blocks or free space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillUsage {
    /// Bytes currently reserved by live scratch backing owners.
    pub reserved_bytes: u64,
    /// Highest simultaneous reservation, including writes awaiting completion.
    pub peak_reserved_bytes: u64,
    /// Scratch paths still owned, including creation and cleanup in progress.
    pub live_files: u64,
    /// Highest simultaneous number of owned scratch paths.
    pub peak_live_files: u64,
    /// Total scratch files admitted, including failed creation attempts.
    pub total_files: u64,
    /// Total bytes accepted by writers, including queued writes; saturates at u64::MAX.
    pub accepted_bytes: u64,
}

#[derive(Debug, Default)]
pub(super) struct Counters {
    pub(super) used: Arc<AtomicU64>,
    pub(super) peak: AtomicU64,
    pub(super) files: AtomicU64,
    pub(super) peak_files: AtomicU64,
    pub(super) total_files: AtomicU64,
    pub(super) accepted: AtomicU64,
}

/// Retain scratch diagnostics without retaining scratch storage or its grant.
///
/// ```
/// # use lance_io::spill::SpillStatistics;
/// let statistics = SpillStatistics::default();
/// assert_eq!(statistics.snapshot().reserved_bytes, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SpillStatistics {
    pub(super) inner: Arc<Counters>,
}

impl SpillStatistics {
    /// Read independently updated counters; concurrent snapshots are approximate.
    pub fn snapshot(&self) -> SpillUsage {
        SpillUsage {
            reserved_bytes: self.inner.used.load(Ordering::Acquire),
            peak_reserved_bytes: self.inner.peak.load(Ordering::Acquire),
            live_files: self.inner.files.load(Ordering::Acquire),
            peak_live_files: self.inner.peak_files.load(Ordering::Acquire),
            total_files: self.inner.total_files.load(Ordering::Acquire),
            accepted_bytes: self.inner.accepted.load(Ordering::Acquire),
        }
    }
}
