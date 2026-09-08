// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Consumer-supplied sink for MemWAL write-path events.

use std::fmt::Debug;
use std::time::Duration;

/// Sink for individual write-path events, supplied by the consumer via
/// [`ShardWriterConfig::observer`](super::write::ShardWriterConfig::observer).
///
/// Cumulative counts stay on
/// [`WriteStats`](super::write::WriteStats), which an embedder polls: a total
/// loses nothing to aggregation. A duration does — an average reconstructed
/// from a running total cannot show a tail — so each flush is reported here as
/// it completes and the consumer decides how to aggregate it.
///
/// Observers run inline on the flush task. Do the aggregation, not the export.
///
/// Every method defaults to a no-op, so adding an event is not a breaking
/// change for existing implementors.
pub trait WalObserver: Send + Sync + Debug {
    /// A WAL buffer flush landed in object storage. This is the latency a
    /// `durable_write` put waits on.
    fn on_wal_flush(&self, _duration: Duration, _bytes: usize) {}

    /// A frozen memtable became an L0 SSTable. Orders of magnitude longer
    /// than a WAL flush.
    fn on_memtable_flush(&self, _duration: Duration, _rows: usize) {}
}
