// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicU64, Ordering};

use lance_io::scheduler::IoStats;
use serde::Serialize;

/// Cumulative counters from all streams of one reader; clones share counters.
/// Snapshots during concurrent work are not atomic.
///
/// Descriptor scans and context creation precede this session and are excluded.
/// Physical counters measure scheduler submissions after coalescing and splitting,
/// rather than successful bytes or backend retries. Reservation counters cover the
/// shared materialization context, including other readers and retained outputs.
///
/// ```
/// # use lance::blob::reference::BlobRefReader;
/// # fn example(reader: &BlobRefReader) -> lance::Result<()> {
/// // Compare logical totals after the reader's streams finish.
/// let metrics = reader.metrics()?;
/// assert!(metrics.returned_logical_bytes <= metrics.requested_logical_bytes);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct BlobRefMetrics {
    /// Descriptor requests issued by this reader: always zero.
    pub descriptor_requests: u64,
    /// Attempts to resolve unknown external lengths, excluding cache hits.
    pub metadata_requests: u64,
    /// Physical ranges submitted after scheduler coalescing and splitting.
    pub payload_requests: u64,
    /// Physical bytes submitted to storage, including coalescing gaps.
    pub payload_bytes: u64,
    /// Resolved logical demand bytes in successfully planned operations; duplicates count separately.
    pub requested_logical_bytes: u64,
    /// Logical bytes yielded to consumers, counting duplicates separately.
    pub returned_logical_bytes: u64,
    /// Planned physical footprint of admitted batches, including coalescing gaps.
    pub coalesced_bytes: u64,
    /// Repeated nonempty physical ranges in each operation, after the first demand.
    pub duplicate_demands: u64,
    /// Current materialization reservation, including physical and logical charges.
    pub reserved_bytes: u64,
    /// Peak materialization reservation since the shared context was created.
    pub peak_reserved_bytes: u64,
    /// Current physical I/O reservation retained by outstanding reads.
    pub io_reserved_bytes: u64,
    /// Peak physical I/O reservation since the shared context was created.
    pub peak_io_reserved_bytes: u64,
}

#[derive(Debug, Default)]
pub struct ReaderMetrics {
    pub(crate) payload: IoStats,
    pub(crate) metadata_requests: AtomicU64,
    pub(crate) requested_logical_bytes: AtomicU64,
    pub(crate) returned_logical_bytes: AtomicU64,
    pub(crate) coalesced_bytes: AtomicU64,
    pub(crate) duplicate_demands: AtomicU64,
}

impl ReaderMetrics {
    pub(crate) fn snapshot(&self) -> BlobRefMetrics {
        let payload = self.payload.snapshot();
        BlobRefMetrics {
            metadata_requests: self.metadata_requests.load(Ordering::Relaxed),
            payload_requests: payload.iops,
            payload_bytes: payload.bytes_read,
            requested_logical_bytes: self.requested_logical_bytes.load(Ordering::Relaxed),
            returned_logical_bytes: self.returned_logical_bytes.load(Ordering::Relaxed),
            coalesced_bytes: self.coalesced_bytes.load(Ordering::Relaxed),
            duplicate_demands: self.duplicate_demands.load(Ordering::Relaxed),
            ..Default::default()
        }
    }
}
