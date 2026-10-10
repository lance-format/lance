// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

pub const AND_CANDIDATES_SEEN_METRIC: &str = "and_candidates_seen";
pub const AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC: &str = "and_candidates_pruned_before_return";
pub const AND_FULL_SCORES_METRIC: &str = "and_full_scores";
pub const FREQS_COLLECTED_METRIC: &str = "freqs_collected";
pub const FTS_BLOCK_METADATA_ROWS_METRIC: &str = "fts_block_metadata_rows";
pub const FTS_PAYLOAD_BLOCKS_READ_METRIC: &str = "fts_payload_blocks_read";
pub const FTS_POSITION_BLOCKS_READ_METRIC: &str = "fts_position_blocks_read";
pub const FTS_BLOCKS_PRUNED_METRIC: &str = "fts_blocks_pruned";
pub const FTS_PAYLOAD_READ_RANGES_METRIC: &str = "fts_payload_read_ranges";
pub const FTS_REFINEMENT_BATCHES_METRIC: &str = "fts_refinement_batches";
pub const FTS_PAYLOAD_READ_BATCHES_METRIC: &str = "fts_payload_read_batches";
pub const FTS_POSITION_READ_BATCHES_METRIC: &str = "fts_position_read_batches";

/// A trait used by the index to report metrics
///
/// Callers can implement this trait to collect metrics
pub trait MetricsCollector: Send + Sync {
    /// Record partition loads
    ///
    /// Many indices consist of partitions that may need to be loaded
    /// into cache.  For example, an inverted index or ngram index has a
    /// posting list for each token.
    ///
    /// In the ideal case, these shards are in the cache and will not need
    /// to be loaded from disk.  This method should not be called if the
    /// shard is in the cache.
    fn record_parts_loaded(&self, num_parts: usize);

    /// Record a shard load
    fn record_part_load(&self) {
        self.record_parts_loaded(1);
    }

    /// Record an index load
    ///
    /// This should be called when a scalar index is loaded from storage.
    /// It should not be called if the index is already in memory.
    fn record_index_loads(&self, num_indexes: usize);

    /// Record an index load
    fn record_index_load(&self) {
        self.record_index_loads(1);
    }

    /// Record the number of "comparisons" made by the index
    ///
    /// What exactly constitutes a comparison depends on the index type.
    /// For example, a B-tree index may make comparisons while searching for a value.
    /// On the other hand, a bitmap index makes comparisons when computing the intersection
    /// of two bitmaps.
    ///
    /// The goal is to provide some visibility into the compute cost of the search
    fn record_comparisons(&self, num_comparisons: usize);

    /// Record AND candidates returned from WAND alignment to the scoring loop.
    ///
    /// This excludes candidates pruned before `next()` returns. Use this with
    /// `record_and_candidates_pruned_before_return` to recover total aligned
    /// AND candidates.
    fn record_and_candidates_seen(&self, _num_candidates: usize) {}

    /// Record AND candidates pruned during WAND alignment before `next()` returns.
    fn record_and_candidates_pruned_before_return(&self, _num_candidates: usize) {}

    fn record_and_full_scores(&self, _num_scores: usize) {}

    fn record_freqs_collected(&self, _num_collections: usize) {}

    fn record_fts_block_metadata_rows(&self, _num_rows: usize) {}

    fn record_fts_payload_blocks_read(&self, _num_blocks: usize) {}

    fn record_fts_position_blocks_read(&self, _num_blocks: usize) {}

    fn record_fts_blocks_pruned(&self, _num_blocks: usize) {}

    fn record_fts_payload_read_ranges(&self, _num_ranges: usize) {}

    fn record_fts_refinement_batches(&self, _num_batches: usize) {}

    fn record_fts_payload_read_batches(&self, _num_batches: usize) {}

    fn record_fts_position_read_batches(&self, _num_batches: usize) {}

    /// Returns an optional sink for recording exact I/O statistics (bytes read,
    /// IOPS, and requests) performed on behalf of this collector.
    ///
    /// Index implementations that read from a
    /// [`lance_io::scheduler::ScanScheduler`] can attach the returned handle to
    /// their file readers so the I/O performed for a single query is measured
    /// and attributed here.  The default returns `None`, meaning the caller does
    /// not want I/O measured (and index implementations should then take their
    /// normal, uninstrumented read path).
    fn io_stats(&self) -> Option<lance_io::scheduler::IoStats> {
        None
    }
}

/// A no-op metrics collector that does nothing
pub struct NoOpMetricsCollector;

impl MetricsCollector for NoOpMetricsCollector {
    fn record_parts_loaded(&self, _num_parts: usize) {}
    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
}

#[derive(Default)]
pub struct LocalMetricsCollector {
    pub parts_loaded: AtomicUsize,
    pub index_loads: AtomicUsize,
    pub comparisons: AtomicUsize,
    pub fts_block_metadata_rows: AtomicUsize,
    pub fts_payload_blocks_read: AtomicUsize,
    pub fts_position_blocks_read: AtomicUsize,
    pub fts_blocks_pruned: AtomicUsize,
    pub fts_payload_read_ranges: AtomicUsize,
    pub fts_refinement_batches: AtomicUsize,
    pub fts_payload_read_batches: AtomicUsize,
    pub fts_position_read_batches: AtomicUsize,
}

impl LocalMetricsCollector {
    pub fn dump_into(self, other: &dyn MetricsCollector) {
        other.record_parts_loaded(self.parts_loaded.load(Ordering::Relaxed));
        other.record_index_loads(self.index_loads.load(Ordering::Relaxed));
        other.record_comparisons(self.comparisons.load(Ordering::Relaxed));
        other.record_fts_block_metadata_rows(self.fts_block_metadata_rows.load(Ordering::Relaxed));
        other.record_fts_payload_blocks_read(self.fts_payload_blocks_read.load(Ordering::Relaxed));
        other
            .record_fts_position_blocks_read(self.fts_position_blocks_read.load(Ordering::Relaxed));
        other.record_fts_blocks_pruned(self.fts_blocks_pruned.load(Ordering::Relaxed));
        other.record_fts_payload_read_ranges(self.fts_payload_read_ranges.load(Ordering::Relaxed));
        other.record_fts_refinement_batches(self.fts_refinement_batches.load(Ordering::Relaxed));
        other.record_fts_position_read_batches(
            self.fts_position_read_batches.load(Ordering::Relaxed),
        );
        other
            .record_fts_payload_read_batches(self.fts_payload_read_batches.load(Ordering::Relaxed));
    }
}

impl MetricsCollector for LocalMetricsCollector {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.parts_loaded.fetch_add(num_parts, Ordering::Relaxed);
    }

    fn record_index_loads(&self, num_indexes: usize) {
        self.index_loads.fetch_add(num_indexes, Ordering::Relaxed);
    }

    fn record_comparisons(&self, num_comparisons: usize) {
        self.comparisons
            .fetch_add(num_comparisons, Ordering::Relaxed);
    }

    fn record_fts_block_metadata_rows(&self, num_rows: usize) {
        self.fts_block_metadata_rows
            .fetch_add(num_rows, Ordering::Relaxed);
    }

    fn record_fts_payload_blocks_read(&self, num_blocks: usize) {
        self.fts_payload_blocks_read
            .fetch_add(num_blocks, Ordering::Relaxed);
    }

    fn record_fts_position_blocks_read(&self, num_blocks: usize) {
        self.fts_position_blocks_read
            .fetch_add(num_blocks, Ordering::Relaxed);
    }

    fn record_fts_blocks_pruned(&self, num_blocks: usize) {
        self.fts_blocks_pruned
            .fetch_add(num_blocks, Ordering::Relaxed);
    }

    fn record_fts_payload_read_ranges(&self, num_ranges: usize) {
        self.fts_payload_read_ranges
            .fetch_add(num_ranges, Ordering::Relaxed);
    }

    fn record_fts_refinement_batches(&self, num_batches: usize) {
        self.fts_refinement_batches
            .fetch_add(num_batches, Ordering::Relaxed);
    }

    fn record_fts_payload_read_batches(&self, num_batches: usize) {
        self.fts_payload_read_batches
            .fetch_add(num_batches, Ordering::Relaxed);
    }

    fn record_fts_position_read_batches(&self, num_batches: usize) {
        self.fts_position_read_batches
            .fetch_add(num_batches, Ordering::Relaxed);
    }
}
