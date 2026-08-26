// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

pub const AND_CANDIDATES_SEEN_METRIC: &str = "and_candidates_seen";
pub const AND_CANDIDATES_PRUNED_BEFORE_RETURN_METRIC: &str = "and_candidates_pruned_before_return";
pub const AND_FULL_SCORES_METRIC: &str = "and_full_scores";
pub const FREQS_COLLECTED_METRIC: &str = "freqs_collected";
pub const COMPOUND_ADDRESSES_RESOLVED_METRIC: &str = "compound_addresses_resolved";
pub const COMPOUND_ADDRESS_RESOLUTION_BATCHES_METRIC: &str = "compound_address_resolution_batches";
pub const COMPOUND_PEAK_ADDRESS_RESOLUTION_BATCH_SIZE_METRIC: &str =
    "compound_peak_address_resolution_batch_size";
pub const COMPOUND_SCORE_FLOOR_OVERFLOWS_METRIC: &str = "compound_score_floor_overflows";
pub const COMPOUND_PEAK_BUFFERED_CANDIDATES_METRIC: &str = "compound_peak_buffered_candidates";
pub const COMPOUND_SHOULD_SKIPPED_WINDOWS_METRIC: &str = "compound_should_skipped_windows";
pub const COMPOUND_SHOULD_BOUND_RECOMPUTATIONS_METRIC: &str =
    "compound_should_bound_recomputations";
pub const COMPOUND_SHOULD_ESSENTIAL_EVALUATIONS_METRIC: &str =
    "compound_should_essential_evaluations";
pub const COMPOUND_SHOULD_NON_ESSENTIAL_EVALUATIONS_METRIC: &str =
    "compound_should_non_essential_evaluations";
pub const CROSS_COLUMN_STAGED_ATTEMPTS_METRIC: &str = "cross_column_staged_attempts";
pub const CROSS_COLUMN_STAGED_SUCCESSES_METRIC: &str = "cross_column_staged_successes";
pub const CROSS_COLUMN_STAGED_FALLBACKS_METRIC: &str = "cross_column_staged_fallbacks";
pub const CROSS_COLUMN_STAGED_CANDIDATES_METRIC: &str = "cross_column_staged_candidates";
pub const WAND_EXACTNESS_CERTIFICATE_ATTEMPTS_METRIC: &str = "wand_exactness_certificate_attempts";
pub const WAND_EXACTNESS_CERTIFICATE_STRICT_METRIC: &str = "wand_exactness_certificate_strict";
pub const WAND_EXACTNESS_CERTIFICATE_EXHAUSTIVE_METRIC: &str =
    "wand_exactness_certificate_exhaustive";
pub const WAND_EXACTNESS_CERTIFICATE_FALLBACKS_METRIC: &str =
    "wand_exactness_certificate_fallbacks";
pub const WAND_EXACTNESS_CERTIFICATE_CANDIDATES_METRIC: &str =
    "wand_exactness_certificate_candidates";
pub const WAND_EXACTNESS_PROBE_MS_METRIC: &str = "wand_exactness_probe_ms";
pub const WAND_EXACTNESS_PROBE_COMPARISONS_METRIC: &str = "wand_exactness_probe_comparisons";
pub const WAND_TIE_COMPLETION_ATTEMPTS_METRIC: &str = "wand_tie_completion_attempts";
pub const WAND_TIE_COMPLETION_SUCCESSES_METRIC: &str = "wand_tie_completion_successes";
pub const WAND_TIE_COMPLETION_OVERFLOWS_METRIC: &str = "wand_tie_completion_overflows";
pub const WAND_TIE_COMPLETION_CANDIDATES_METRIC: &str = "wand_tie_completion_candidates";
pub const WAND_TIE_COMPLETION_ROW_ID_REPLACEMENTS_METRIC: &str =
    "wand_tie_completion_row_id_replacements";
pub const WAND_TIE_COMPLETION_MS_METRIC: &str = "wand_tie_completion_ms";
pub const WAND_TIE_COMPLETION_COMPARISONS_METRIC: &str = "wand_tie_completion_comparisons";
pub const WAND_SEEDED_FALLBACKS_METRIC: &str = "wand_seeded_fallbacks";
pub const WAND_SEEDED_FALLBACK_MS_METRIC: &str = "wand_seeded_fallback_ms";
pub const WAND_SEEDED_FALLBACK_COMPARISONS_METRIC: &str = "wand_seeded_fallback_comparisons";

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

    /// Record index cache hits observed while serving this query.
    ///
    /// A "hit" is one page-level lookup (partition, posting list, BTree page, etc.)
    /// that was served from the in-memory index cache without touching storage.
    fn record_index_cache_hits(&self, _num_hits: usize) {}

    /// Convenience for a single cache hit.
    fn record_index_cache_hit(&self) {
        self.record_index_cache_hits(1);
    }

    /// Record index cache misses observed while serving this query.
    ///
    /// A "miss" is one page-level lookup that had to be loaded from storage
    /// because it was not present in the cache.
    fn record_index_cache_misses(&self, _num_misses: usize) {}

    /// Convenience for a single cache miss.
    fn record_index_cache_miss(&self) {
        self.record_index_cache_misses(1);
    }

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

    /// Record compound FTS document addresses resolved for final row-ID ties.
    fn record_compound_addresses_resolved(&self, _num_addresses: usize) {}

    /// Record bounded compound FTS address-resolution batches.
    fn record_compound_address_resolution_batches(&self, _num_batches: usize) {}

    /// Record the largest compound FTS address-resolution batch.
    fn record_compound_peak_address_resolution_batch_size(&self, _num_addresses: usize) {}

    /// Record unresolved score floors that required a resolved-key retry.
    fn record_compound_score_floor_overflows(&self, _num_overflows: usize) {}

    /// Record a candidate-buffer high-water mark for compound FTS.
    fn record_compound_peak_buffered_candidates(&self, _num_candidates: usize) {}

    /// Record pure-SHOULD compound FTS windows skipped using score bounds.
    fn record_compound_should_skipped_windows(&self, _num_windows: usize) {}

    /// Record score-bound recomputations for pure-SHOULD compound FTS windows.
    fn record_compound_should_bound_recomputations(&self, _num_recomputations: usize) {}

    /// Record essential-clause evaluations for pure-SHOULD compound FTS.
    fn record_compound_should_essential_evaluations(&self, _num_evaluations: usize) {}

    /// Record non-essential-clause evaluations for pure-SHOULD compound FTS.
    fn record_compound_should_non_essential_evaluations(&self, _num_evaluations: usize) {}

    /// Record cross-column queries that attempted candidate-driven staging.
    fn record_cross_column_staged_attempts(&self, _num_attempts: usize) {}

    /// Record staged executions that produced a complete candidate set.
    fn record_cross_column_staged_successes(&self, _num_successes: usize) {}

    /// Record staged executions abandoned in favor of exact eager execution.
    fn record_cross_column_staged_fallbacks(&self, _num_fallbacks: usize) {}

    /// Record unique row-address candidates produced by successful staging.
    fn record_cross_column_staged_candidates(&self, _num_candidates: usize) {}

    /// Record root Match WAND executions that attempted a k+1 exactness certificate.
    fn record_wand_exactness_certificate_attempts(&self, _num_attempts: usize) {}

    /// Record certificates proven by a strict score gap after the kth result.
    fn record_wand_exactness_certificate_strict(&self, _num_certificates: usize) {}

    /// Record certificates proven because WAND exhausted all matching documents.
    fn record_wand_exactness_certificate_exhaustive(&self, _num_certificates: usize) {}

    /// Record ambiguous certificates that fell back to the exact compound scorer.
    fn record_wand_exactness_certificate_fallbacks(&self, _num_fallbacks: usize) {}

    /// Record WAND candidates returned to the certificate classifier.
    fn record_wand_exactness_certificate_candidates(&self, _num_candidates: usize) {}

    /// Record bounded WAND attempts to complete an ambiguous kth-score tie.
    fn record_wand_tie_completion_attempts(&self, _num_attempts: usize) {}

    /// Record kth-score ties completed without exact replay.
    fn record_wand_tie_completion_successes(&self, _num_successes: usize) {}

    /// Record kth-score ties that exceeded the bounded completion budget.
    fn record_wand_tie_completion_overflows(&self, _num_overflows: usize) {}

    /// Record candidates returned by bounded kth-score tie completion.
    fn record_wand_tie_completion_candidates(&self, _num_candidates: usize) {}

    /// Record exact replays seeded with an inclusive kth-score floor.
    fn record_wand_seeded_fallbacks(&self, _num_fallbacks: usize) {}

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
    // Kept `pub(crate)` so that adding new metric fields to this public struct
    // does not break downstream callers that construct or destructure the
    // existing three fields. Callers can still read cumulative values via
    // [`Self::index_cache_hits`] / [`Self::index_cache_misses`].
    pub(crate) index_cache_hits: AtomicUsize,
    pub(crate) index_cache_misses: AtomicUsize,
}

impl LocalMetricsCollector {
    pub fn dump_into(self, other: &dyn MetricsCollector) {
        other.record_parts_loaded(self.parts_loaded.load(Ordering::Relaxed));
        other.record_index_loads(self.index_loads.load(Ordering::Relaxed));
        other.record_comparisons(self.comparisons.load(Ordering::Relaxed));
        other.record_index_cache_hits(self.index_cache_hits.load(Ordering::Relaxed));
        other.record_index_cache_misses(self.index_cache_misses.load(Ordering::Relaxed));
    }

    /// Cumulative index cache hits recorded so far.
    pub fn index_cache_hits(&self) -> usize {
        self.index_cache_hits.load(Ordering::Relaxed)
    }

    /// Cumulative index cache misses recorded so far.
    pub fn index_cache_misses(&self) -> usize {
        self.index_cache_misses.load(Ordering::Relaxed)
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

    fn record_index_cache_hits(&self, num_hits: usize) {
        self.index_cache_hits.fetch_add(num_hits, Ordering::Relaxed);
    }

    fn record_index_cache_misses(&self, num_misses: usize) {
        self.index_cache_misses
            .fetch_add(num_misses, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SumSink {
        parts: AtomicUsize,
        loads: AtomicUsize,
        comparisons: AtomicUsize,
        hits: AtomicUsize,
        misses: AtomicUsize,
    }

    impl MetricsCollector for SumSink {
        fn record_parts_loaded(&self, n: usize) {
            self.parts.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_loads(&self, n: usize) {
            self.loads.fetch_add(n, Ordering::Relaxed);
        }
        fn record_comparisons(&self, n: usize) {
            self.comparisons.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_cache_hits(&self, n: usize) {
            self.hits.fetch_add(n, Ordering::Relaxed);
        }
        fn record_index_cache_misses(&self, n: usize) {
            self.misses.fetch_add(n, Ordering::Relaxed);
        }
    }

    #[test]
    fn local_metrics_collector_forwards_cache_counts() {
        let local = LocalMetricsCollector::default();
        local.record_index_cache_hit();
        local.record_index_cache_hit();
        local.record_index_cache_misses(3);
        local.record_part_load();
        local.record_index_load();
        local.record_comparisons(5);

        let sink = SumSink {
            parts: AtomicUsize::new(0),
            loads: AtomicUsize::new(0),
            comparisons: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        };
        local.dump_into(&sink);

        assert_eq!(sink.parts.load(Ordering::Relaxed), 1);
        assert_eq!(sink.loads.load(Ordering::Relaxed), 1);
        assert_eq!(sink.comparisons.load(Ordering::Relaxed), 5);
        assert_eq!(sink.hits.load(Ordering::Relaxed), 2);
        assert_eq!(sink.misses.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn no_op_metrics_collector_ignores_cache_counts() {
        // Ensures existing implementors that do not override cache-count methods
        // remain sound (default impl is a no-op).
        let collector = NoOpMetricsCollector;
        collector.record_index_cache_hit();
        collector.record_index_cache_miss();
        collector.record_index_cache_hits(10);
        collector.record_index_cache_misses(20);
    }
}
