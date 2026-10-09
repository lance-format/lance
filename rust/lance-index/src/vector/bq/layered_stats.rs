// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Process-wide diagnostics of the lazy layered full-precision scan.
//!
//! The counters only advance while a query runs the lazy scan (or is checked
//! for it), so production queries with the scan disabled touch at most the
//! `ineligible_disabled` counter. The `resident_columns_*`,
//! `resident_attach_*`, `code_only_partition_*`, `resident_store_evictions`
//! and `pinned_overflow` counters are the exception: they advance when an
//! IVF_RQ index, layered or not, opens with, loads or reads through its
//! resident columns. So is
//! `storage_construct_repacks`, which advances when any RaBitQ storage is
//! built from codes it must rewrite, and `plane_rows_unpack_*`, which advance
//! on every read of a plane-row IVF_RQ file.
//! Readers take deltas with [`snapshot_and_reset`].

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

    /// The count so far, without resetting it: tests that run beside others
    /// can only bound the growth of a process-wide counter from below.
    #[cfg(test)]
    pub(crate) fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
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
        /// Origin requests of those reads after coalescing within the lazy
        /// origin gap (`LANCE_RQ_LAZY_ORIGIN_GAP_BYTES`) and splitting, which
        /// are GETs on an object store. A fallback load of the resident
        /// columns that such a read starts, for a storage whose store no
        /// index open loaded, is counted in `resident_columns_load_requests`
        /// instead.
        origin_sparse_requests,
        /// Bytes those requests read, the gaps they span included.
        origin_sparse_bytes,
        /// Rows those reads gathered, which the gap does not change. Over one
        /// plane, `origin_sparse_bytes / (origin_sparse_rows * row width)` is
        /// the byte amplification of the gap.
        origin_sparse_rows,
        /// Sum over gathered probes of probes still unscored at issue.
        staleness_sum,
        /// Issues deferred because the heap was not yet full at the gate.
        deferred_issues,
        /// Deferred issues released only when every earlier probe was scored.
        serial_waits,
        /// Gathers issued before the heap was full and before their turn,
        /// selecting every accepted row; see `LANCE_RQ_LAZY_EAGER_BEFORE_FULL`.
        eager_before_full,
        /// Sparse gathers of planes the persistent tier did not hold that
        /// loaded the whole plane instead of more origin row runs than
        /// `LANCE_RQ_LAZY_ORIGIN_MAX_RUNS` (unlimited by default).
        origin_whole_fallbacks,
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
        /// Bytes of values each load of an index file's resident columns
        /// keeps in memory (see `LANCE_RQ_RESIDENT_COLUMNS`). The first index
        /// of the file to open loads them once, so over every snapshot this
        /// sums to the store's size, plus a size per reload after an idle
        /// eviction.
        resident_columns_bytes,
        /// Memory the arrays of those loads hold: the capacity of their
        /// buffers, counted per column. It exceeds `resident_columns_bytes`
        /// when an array keeps a buffer larger than its values.
        resident_columns_alloc_bytes,
        /// Origin requests of those loads after coalescing and splitting,
        /// which are GETs on an object store.
        resident_columns_load_requests,
        /// Bytes those loads read from the origin.
        resident_columns_load_bytes,
        /// Time those loads took.
        resident_columns_load_ns,
        /// Classifying staged probes' ex planes by the cache tier that holds
        /// them (only RAM residency on a low-latency origin).
        tier_peek_ns,
        /// Gathers issued beyond the ordinary window with a permit of their
        /// index's pool, because their probe reads an ex plane from a
        /// high-latency origin; see `LANCE_RQ_LAZY_FAR_WINDOW`.
        far_early_issues,
        /// Gathers beyond the ordinary window that found no free permit and
        /// waited for one or for the ordinary window, whichever came first.
        far_permit_waits,
        /// Lazy scans that published the scoring of their first probe; see
        /// `lazy_rank0_scored_ns`.
        lazy_rank0_scored_queries,
        /// Time from the start of each of those scans until it published the
        /// scoring of its first probe (rank 0) to the gathers: once the probe
        /// was scored or, with `LANCE_RQ_LAZY_PARTIAL_PUBLISH=on`, earlier,
        /// once its rows filled the heap. No gather that waits for the
        /// threshold or its turn is released before.
        lazy_rank0_scored_ns,
        /// Lazy scans that issued a gather of a probe that is not
        /// `certain_dense`, whose issue waits on scoring progress; see
        /// `lazy_first_gather_issue_ns`. Certain-dense gathers, the first
        /// probe's always among them, are issued as soon as staged.
        lazy_first_gather_issue_queries,
        /// Time from the start of each of those scans until the first such
        /// gather was issued, its gate waits and permit included.
        lazy_first_gather_issue_ns,
        /// Gathers whose probe was further ahead of scoring than their
        /// staleness window (the far window for a probe that reads an ex
        /// plane from a high-latency origin) when they reached the gate.
        lazy_window_waits,
        /// Time those gathers waited for scoring to bring their probe within
        /// the window. Part of `gate_wait_ns`, as are
        /// `lazy_release_wait_ns` and `far_permit_wait_ns`.
        lazy_window_wait_ns,
        /// Time the `deferred_issues` waited for the heap to fill, their turn
        /// or, with eager-before-full, the scoring of the probes holding `k`
        /// rows.
        lazy_release_wait_ns,
        /// Time the `far_permit_waits` waited for a permit or the ordinary
        /// window.
        far_permit_wait_ns,
        /// Loads of an index file's resident store: one per store when an
        /// index of its file opens while no live index or cache holds it,
        /// and so one more for each reload after an idle eviction.
        resident_columns_loads,
        /// Of those loads, the ones a read started because no index open had
        /// loaded the store: only a storage built outside an index open
        /// reads before its store loads. 0 for every index opened through a
        /// `Dataset`.
        resident_columns_read_loads,
        /// Index opens that bound a resident store a live index of the file
        /// had loaded, found through the process's weak registry of stores.
        resident_columns_registry_reuses,
        /// Index opens that bound a resident store charged in the index cache.
        resident_columns_binds,
        /// Loaded resident stores admitted to the index cache again by an
        /// index open that found them gone: evicted while overflowed,
        /// cleared, or refused at admission.
        resident_columns_recharges,
        /// Index opens whose resident store would take more of the index
        /// cache's largest admissible entry than `auto` allows, so `auto`
        /// kept the small columns in the file (`on` keeps them resident).
        resident_columns_oversize,
        /// Index opens under `auto` whose resident store would fit the index
        /// cache but whose cache has no pin budget (a backend that never
        /// pins, such as Moka), so `auto` kept the small columns in the
        /// file rather than keep a store a lease cannot pin (`on` keeps
        /// them resident).
        resident_columns_unpinnable,
        /// Index opens that leased the cached resident store before their
        /// first index-cache access.
        resident_columns_preopen_leases,
        /// Resident store entries the index cache dropped: evicted while
        /// idle, cleared, or refused at admission.
        resident_store_evictions,
        /// Leases of a resident store that found no room under the index
        /// cache's pinned cap, leaving the store evictable.
        pinned_overflow,
        /// RaBitQ storages, layered or not, whose construction rewrote the
        /// stored codes: packed sign codes stored unpacked, or repacked
        /// legacy sequential ex codes into the blocked layout. A storage
        /// built from codes stored packed and blocked, as the index builder
        /// writes them and cache entries keep them, rewrites nothing.
        storage_construct_repacks,
        /// Batches assembled with an index's resident rows, copied or shared:
        /// every read of a plane or partition from the file that takes its
        /// small columns from the resident store, and every read of a
        /// code-only cache entry (`LANCE_RQ_ENTRY_COLUMNS=codes`), hits
        /// included.
        resident_attach_calls,
        /// Rows those batches hold.
        resident_attach_rows,
        /// Bytes of the resident rows those batches hold, copied or shared,
        /// each row at its columns' fixed widths: what a full entry would
        /// have kept in the cache.
        resident_attach_bytes,
        /// Time spent assembling them once their rows are resolved: the
        /// copies (or views) of the resident rows, the other columns' clones
        /// and the batch. Summed over every attach, including those of
        /// partitions prepared at once, so it is CPU time rather than time
        /// on a query's critical path.
        resident_attach_ns,
        /// Of those batches, whole partitions and planes in which every
        /// resident column shares the store's buffers.
        resident_attach_shares,
        /// Whole partitions and planes that copied a resident column: reads
        /// that become cache entries (full entries, or partitions of a native
        /// index with a graph), partitions streamed out of the index, and
        /// columns a view cannot cover.
        resident_attach_whole_copies,
        /// Sparse gathers, whose rows are always copied.
        resident_attach_gathers,
        /// Bytes the whole copies and the gathers copied.
        resident_attach_copied_bytes,
        /// Storages of native partitions built from their code-only entry
        /// (`LANCE_RQ_ENTRY_COLUMNS=codes`), once per read of the partition.
        code_only_partition_builds,
        /// Time those builds took: the attach of the resident rows and the
        /// storage's construction, without a fallback load of the store.
        code_only_partition_build_ns,
        /// Bytes of packed row columns that reads of plane-row IVF_RQ files
        /// unpacked into the column layout's fields.
        plane_rows_unpack_bytes,
        /// Time spent unpacking them.
        plane_rows_unpack_ns,
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
        /// Probes of non-empty partitions whose ex planes were not both
        /// resident, predicted from `k` and the partition sizes to be
        /// gathered whole, and so loaded and scored by the eager scan, as
        /// `LANCE_RQ_LAZY_DENSE_TO_EAGER` routes them for their planes' tiers.
        dense_to_eager,
        /// Probes of non-empty partitions with a high or low plane that no
        /// cache tier held when they were staged, on an index whose origin
        /// latency class is high: reading the plane is a request to the
        /// origin, an object store such as S3. Counted whether the probe was
        /// then gathered lazily or routed to the eager scan; always 0 on a
        /// low-latency origin.
        s3_bound_probes,
        /// Lazy probes issued at once because the rows of all earlier probes
        /// cannot fill the heap. The probes that
        /// `LANCE_RQ_LAZY_DENSE_TO_EAGER` routes are counted in
        /// `dense_to_eager` instead.
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
        /// Lazy probes whose survivors first filled the heap to `k` rows
        /// while they were scored, and that published the heap's top then,
        /// before the probe was scored whole, which only
        /// `LANCE_RQ_LAZY_PARTIAL_PUBLISH=on` does. That threshold is the
        /// `k`-th best of the rows scored so far, looser than the probe's
        /// final one.
        mid_probe_full_publishes,
        /// Gathers that selected their survivors against such a mid-probe
        /// threshold: issued after it was published and before its probe
        /// was scored whole.
        partial_threshold_issues,
        /// Survivors of those gathers whose ex rows were gathered, part of
        /// `rows_fetched`.
        partial_threshold_rows,
    }
    maxima {
        /// Largest number of probes still unscored at a gather's issue.
        staleness_max,
        /// Most permits of one index's pool taken at once, observed as each
        /// gather beyond the ordinary window takes one; at most
        /// `LANCE_RQ_LAZY_FAR_INFLIGHT`.
        far_in_flight_max,
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
