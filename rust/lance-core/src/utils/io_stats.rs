// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, time::Duration};

/// A sink that records I/O requests as they are submitted to storage.
///
/// This lives in `lance-core` so that the encoding layer (`lance-encoding`) and
/// the I/O layer (`lance-io`) can both refer to it without depending on one
/// another.  It lets a caller attach a lightweight counter to a file reader and
/// measure scheduler-requested bytes/IOPS for a bounded scope (e.g. a single
/// query); see `lance_io::scheduler::IoStats` for the concrete implementation.
///
/// # When to use this
///
/// Lance also exposes two *process-wide, cumulative* I/O accounting facilities:
/// the global scheduler counters (`lance_io::scheduler::iops_counter` /
/// `bytes_read_counter`) and the object-store `IOTracker` wrapper used in tests.
/// Both aggregate every read in the process and cannot attribute I/O to a single
/// bounded scope.  Prefer an `IoStatsRecorder` when you need the scoped I/O of
/// one operation (e.g. a single query): attach it to a reader with
/// `with_io_stats`, then read the snapshot when the scope ends.  It re-uses the
/// reader's cached metadata, so measuring costs no extra file opens and does not
/// disturb the global counters.
pub trait IoStatsRecorder: std::fmt::Debug + Send + Sync {
    /// Record how many fragments were selected for the operation.
    ///
    /// Selection happens before asynchronous reads begin, so cancellation may
    /// leave some selected fragments unread. The default is a no-op for
    /// recorders that are not attached to a fragment-oriented operation.
    fn record_fragments_selected(&self, _count: usize) {}

    /// Record one scheduler request, given the byte ranges as actually
    /// submitted to storage (i.e. after any coalescing/splitting).
    ///
    /// One call counts one scheduler request. Each range counts one physical
    /// I/O operation, and its length counts scheduler-requested bytes. These
    /// are not necessarily wire-level requests or bytes.
    fn record_request(&self, ranges: &[Range<u64>]);

    /// Record one scheduler request before and after range transformation.
    ///
    /// `logical_ranges` are the ranges received by `FileScheduler`. Earlier
    /// encoding layers may already have split oversized ranges or skipped
    /// empty ones, so these are not raw decoder ranges.
    /// `physical_ranges` are the ranges submitted to storage after the file
    /// scheduler has coalesced nearby ranges and split oversized ranges.  The
    /// default preserves the existing physical-I/O-only contract.
    ///
    /// Implementors that need coalescing diagnostics can override this method
    /// without making every recorder pay to inspect logical ranges.
    fn record_request_transform(
        &self,
        _logical_ranges: &[Range<u64>],
        physical_ranges: &[Range<u64>],
    ) {
        self.record_request(physical_ranges);
    }

    /// Record one cloud-reader attempt and the response-body bytes Lance
    /// observed before that attempt completed or failed.
    ///
    /// This covers Lance's request and partial-body retry loops. Retries hidden
    /// inside a provider SDK are not observable here. The default is a no-op.
    fn record_cloud_reader_attempt(&self, _body_bytes: u64, _failed: bool) {}

    /// Whether this recorder needs transport-attempt and response-body accounting.
    ///
    /// The default preserves the reader's ordinary body collection path for
    /// recorders that only count scheduler submissions or read lifecycles.
    fn records_cloud_reader_attempts(&self) -> bool {
        false
    }

    /// Whether this recorder needs per-read queue and execution timings.
    ///
    /// The default avoids lifecycle instrumentation for recorders that only
    /// count scheduler submissions and physical ranges.
    fn records_read_lifecycle(&self) -> bool {
        false
    }

    /// Records when one physical read leaves the scheduler queue.
    ///
    /// `queue_time` covers time from submission until the scheduler starts the
    /// storage read. Implementors that only need request counts can keep the
    /// default no-op.
    fn record_read_started(&self, _queue_time: Duration) {}

    /// Records when one admitted physical read finishes or is cancelled.
    ///
    /// `read_time` starts when the scheduler starts the storage read.
    /// Implementors that only need request counts can keep the default no-op.
    fn record_read_completed(&self, _read_time: Duration) {}
}
