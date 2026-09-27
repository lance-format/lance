// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Process-wide diagnostics of the lazy layered full-precision scan.
//!
//! The counters only advance while a query runs the lazy scan (or is checked
//! for it), so production queries with the scan disabled touch at most the
//! `ineligible_disabled` counter. Readers take deltas with
//! [`snapshot_and_reset`].

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of probe-rank buckets: ranks 0, 1-3, 4-15, 16-63 and 64+.
pub const RANK_BUCKETS: usize = 5;

/// Exclusive upper rank of every bucket but the last.
const RANK_BUCKET_ENDS: [usize; RANK_BUCKETS - 1] = [1, 4, 16, 64];

/// The bucket a probe rank (0 = the nearest partition) is reported in.
pub fn rank_bucket(rank: usize) -> usize {
    RANK_BUCKET_ENDS
        .iter()
        .position(|&end| rank < end)
        .unwrap_or(RANK_BUCKETS - 1)
}

/// A cumulative counter.
#[derive(Debug)]
pub struct Counter(AtomicU64);

impl Counter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn add(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }

    pub fn incr(&self) {
        self.add(1);
    }

    /// Add the nanoseconds elapsed since `start`.
    pub fn add_elapsed(&self, start: std::time::Instant) {
        self.add(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }

    fn take(&self) -> u64 {
        self.0.swap(0, Ordering::Relaxed)
    }
}

/// A cumulative counter split by probe-rank bucket (see [`rank_bucket`]).
#[derive(Debug)]
pub struct RankCounter([AtomicU64; RANK_BUCKETS]);

impl RankCounter {
    const fn new() -> Self {
        Self([const { AtomicU64::new(0) }; RANK_BUCKETS])
    }

    pub fn add(&self, rank: usize, value: u64) {
        self.0[rank_bucket(rank)].fetch_add(value, Ordering::Relaxed);
    }

    pub fn incr(&self, rank: usize) {
        self.add(rank, 1);
    }

    fn take(&self) -> [u64; RANK_BUCKETS] {
        std::array::from_fn(|bucket| self.0[bucket].swap(0, Ordering::Relaxed))
    }
}

/// The largest value observed since the last snapshot.
#[derive(Debug)]
pub struct MaxCounter(AtomicU64);

impl MaxCounter {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn observe(&self, value: u64) {
        self.0.fetch_max(value, Ordering::Relaxed);
    }

    fn take(&self) -> u64 {
        self.0.swap(0, Ordering::Relaxed)
    }
}

macro_rules! layered_lazy_counters {
    (
        counters { $($(#[$counter_doc:meta])* $counter:ident,)* }
        ranked { $($(#[$ranked_doc:meta])* $ranked:ident,)* }
        maxima { $($(#[$max_doc:meta])* $max:ident,)* }
    ) => {
        /// Live counters; see [`LayeredLazyStats`] for their meaning.
        #[derive(Debug)]
        pub struct LayeredLazyCounters {
            $(pub $counter: Counter,)*
            $(pub $ranked: RankCounter,)*
            $(pub $max: MaxCounter,)*
        }

        impl LayeredLazyCounters {
            const fn new() -> Self {
                Self {
                    $($counter: Counter::new(),)*
                    $($ranked: RankCounter::new(),)*
                    $($max: MaxCounter::new(),)*
                }
            }

            fn take(&self) -> LayeredLazyStats {
                LayeredLazyStats {
                    $($counter: self.$counter.take(),)*
                    $($ranked: self.$ranked.take(),)*
                    $($max: self.$max.take(),)*
                }
            }
        }

        /// Counters accumulated since the previous [`snapshot_and_reset`].
        /// Rank-split counters hold one value per [`rank_bucket`]; `*_ns`
        /// counters are summed nanoseconds.
        #[derive(Debug, Default, Clone, PartialEq, Eq)]
        pub struct LayeredLazyStats {
            $($(#[$counter_doc])* pub $counter: u64,)*
            $($(#[$ranked_doc])* pub $ranked: [u64; RANK_BUCKETS],)*
            $($(#[$max_doc])* pub $max: u64,)*
        }
    };
}

layered_lazy_counters! {
    counters {
        /// Queries that ran the lazy scan.
        lazy_queries,
        /// Eligible queries whose probed ex planes were all resident, so the
        /// eager scan ran unchanged.
        lazy_all_resident_skips,
        /// Layered queries checked while the lazy scan is disabled.
        ineligible_disabled,
        /// Layered queries on an index with a row-id remapper.
        ineligible_remapper,
        /// Layered queries below full precision.
        ineligible_precision,
        /// Layered queries in Fast approx mode.
        ineligible_fast,
        /// Layered queries without a raw-query estimator context.
        ineligible_residual,
        /// Layered queries with a cascade factor.
        ineligible_cascade,
        /// Layered queries with a refine factor.
        ineligible_refine,
        /// Layered queries with `k == 0`, which the eager scan serves.
        ineligible_k_zero,
        /// Rows the live heap still needed that the gather did not fetch.
        /// Always 0 unless the lazy scan has a bug; the query then fails.
        needed_not_fetched,
        /// Sparse gathers whose rows came from the origin file because the
        /// cache had no persistent entry for the plane.
        origin_row_reads,
        /// Sum over gathered probes of probes still unscored at issue.
        staleness_sum,
        /// Issues deferred because the heap was not yet full at the gate.
        deferred_issues,
        /// Deferred issues released only when every earlier probe was scored.
        serial_waits,
        /// Loading sign planes.
        sign_load_ns,
        /// Stage-1 CPU dispatches.
        stage1_dispatches,
        /// Stage-1 work, measured on the CPU pool.
        stage1_cpu_ns,
        /// Waiting for the threshold gate before issuing a gather.
        gate_wait_ns,
        /// Gathering resident ex planes.
        fetch_resident_ns,
        /// Gathering whole ex planes from the cache or origin.
        fetch_whole_ns,
        /// Gathering selected ex-plane rows.
        fetch_sparse_ns,
        /// Scorer idle time waiting for the next probe: I/O on the critical path.
        scorer_wait_ns,
        /// Stage-2 and eager scoring work.
        stage2_cpu_ns,
        /// Scoring batches run on the query task.
        dispatch_inline,
        /// Scoring batches dispatched to the CPU pool.
        dispatch_spawn,
        /// Time from query start until the heap first held `k` rows.
        time_to_first_full_ns,
        /// Background whole-plane promotions started.
        promotions_issued,
        /// Promotions not started because the plane was already in flight.
        promotions_deduped,
        /// Promotions not started for lack of an in-flight permit, failed, or
        /// whose plane was no longer resident once loaded.
        promotions_skipped,
        /// Promotions whose plane was resident once loaded.
        promotions_completed,
        /// Bytes of planes loaded by promotions, resident afterwards or not,
        /// apart from critical-path reads.
        promotion_bytes,
    }
    ranked {
        /// Probes gathered lazily.
        lazy_probes,
        /// Probes of non-empty partitions scored eagerly because both ex
        /// planes were resident.
        eager_resident,
        /// Probes scored eagerly because lower-bound gating is disabled.
        gating_off,
        /// Probes of empty partitions, which have no ex rows and are scored
        /// by the eager scan.
        empty,
        /// Probes issued at once because the rows of all earlier probes
        /// cannot fill the heap.
        certain_dense,
        /// High planes served from RAM.
        high_resident,
        /// High planes read whole.
        high_whole,
        /// High planes read by selected rows.
        high_sparse,
        /// Low planes served from RAM.
        low_resident,
        /// Low planes read whole.
        low_whole,
        /// Low planes read by selected rows.
        low_sparse,
        /// Rows accepted by the prefilter.
        stage1_candidates,
        /// Rows stage 1 pruned against the query's upper bound.
        stage1_pruned_ub,
        /// Rows stage 1 pruned against the issue-time heap threshold.
        stage1_pruned_heap,
        /// Stage-1 survivors whose ex rows were gathered.
        rows_fetched,
        /// Survivors scored exactly.
        stage2_exact,
        /// Survivors pruned against the live heap: over-fetch from a stale
        /// issue-time threshold.
        stage2_pruned_live,
        /// Exactly scored survivors outside the query's distance range.
        stage2_exact_rejected,
    }
    maxima {
        /// Largest number of probes still unscored at a gather's issue.
        staleness_max,
    }
}

static COUNTERS: LayeredLazyCounters = LayeredLazyCounters::new();
static PROMOTIONS_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

/// The live counters, for recording.
pub fn counters() -> &'static LayeredLazyCounters {
    &COUNTERS
}

/// Read and zero every counter.
pub fn snapshot_and_reset() -> LayeredLazyStats {
    COUNTERS.take()
}

/// Background promotions currently running. A gauge, not reset by snapshots.
pub fn promotions_in_flight() -> u64 {
    PROMOTIONS_IN_FLIGHT.load(Ordering::Relaxed)
}

pub(crate) fn promotion_started() {
    PROMOTIONS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn promotion_finished() {
    PROMOTIONS_IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_buckets_match_their_documented_ranges() {
        let buckets = [0, 1, 3, 4, 15, 16, 63, 64, 10_000].map(rank_bucket);
        assert_eq!(buckets, [0, 1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn counters_reset_on_snapshot() {
        let counters = LayeredLazyCounters::new();
        counters.lazy_queries.add(3);
        counters.rows_fetched.add(5, 7);
        counters.staleness_max.observe(4);
        counters.staleness_max.observe(2);
        let stats = counters.take();
        assert_eq!(stats.lazy_queries, 3);
        assert_eq!(stats.rows_fetched, [0, 0, 7, 0, 0]);
        assert_eq!(stats.staleness_max, 4);
        assert_eq!(counters.take(), LayeredLazyStats::default());
    }
}
