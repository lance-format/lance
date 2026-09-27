// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy full-precision scan of a layered IVF_RQ index.
//!
//! The eager global-heap scan loads the sign, high and low planes of every
//! probed partition before scoring. This scan loads only the sign plane,
//! bounds every row with the native 1-bit lower bound (stage 1), and gathers
//! the ex-plane rows of the survivors only, which it then scores with the
//! eager scan's per-row step on the same global heap (stage 2).
//!
//! Stage 1 prunes against a threshold `T` taken from that heap: the top of a
//! heap holding `k` rows at some point no later than the probe's scoring.
//! The heap's top never increases once full, so every row stage 1 prunes is
//! one the eager scan's per-row step would prune too, and stage 2 replays the
//! remaining rows in the same order with the same inputs. Heap contents,
//! distances, ties and prune counters are therefore identical to the eager
//! scan; `T` only decides how many rows are read.
//!
//! Pipeline, in probe order:
//! - a producer task stages probes (sign plane, prefilter, stage 1 on the CPU
//!   pool), then gathers each probe's survivors once the probe is at most
//!   `window` probes ahead of scoring, reading the threshold the scorer
//!   publishes over a `watch` channel;
//! - the query task scores ready probes on one heap and publishes progress.
//!
//! Every staging and gather step runs as its own task, so it makes progress
//! whether or not the producer is polling the buffer that holds it; see
//! `spawn_scan_step`.
//!
//! Probes whose ex planes are both resident, and probes that cannot gate on
//! the lower bound, are scored by the eager scan in their probe position.

use std::collections::BinaryHeap;
use std::future::Future;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt32Array};
use futures::StreamExt;
use futures::prelude::stream;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::bq::layered::{PlaneKey, RQPrecision};
use lance_index::vector::bq::layered_stats;
use lance_index::vector::bq::storage::{
    ExRows, RabitQuantizationStorage, SignStage, StagePruneCounts, SurvivorRow,
    select_full_survivors,
};
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::storage::{
    DistanceCalculatorOptions, ExIndex, GatherPlan, GatheredEx, LayeredExLayout, LayeredLazyConfig,
    LazyPromotionTicket, PlaneSource, QueryScratchPool, RabitRawQueryContext, VectorStore,
    plan_plane_gather,
};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{ApproxMode, Query};
use tokio::sync::{mpsc, watch};

use super::{
    GLOBAL_TOPK_INLINE_HEAP_LEN, IVFIndex, LAYERED_LAZY_CONFIG, PartitionEntry,
    PreparedPartitionGuard, PreparedPartitionSearch, RabitSearchCache,
    rotated_partition_centroid_slice,
};

/// Ready probes buffered between the producer and the scorer. Gathers beyond
/// it wait in the producer, where the staleness gate bounds them.
const LAZY_READY_CHANNEL_CAPACITY: usize = 2;
/// Most ready probes one scoring dispatch takes.
const LAZY_SCORE_BATCH_MAX_PROBES: usize = 64;
/// Gathers kept in flight beyond the staleness window, so the reads of the
/// next probes overlap the scoring of the current one.
const LAZY_FETCH_EXTRA_SLOTS: usize = 2;

/// Scoring progress the scorer publishes to the gathers.
#[derive(Debug, Clone, Copy)]
struct LazyProgress {
    /// Probes scored so far, in probe order.
    scored: usize,
    /// Whether the heap has held `k` rows; it then stays full.
    full: bool,
    /// The heap's top when `full`, otherwise +inf.
    threshold: f32,
}

/// A probe after its sign stage.
enum LazyStaged<S: IvfSubIndex, Q: Quantization> {
    Eager(PreparedPartitionSearch<S, Q>),
    /// Lower-bound gating is disabled for the partition, so it is scored by
    /// the eager scan, which scores every row.
    GatingOff {
        partition_id: usize,
        query: Query,
    },
    Empty,
    Lazy(Box<LazySignProbe<S, Q>>),
}

struct LazySignProbe<S: IvfSubIndex, Q: Quantization> {
    rank: usize,
    partition_id: usize,
    dist_q_c: f32,
    /// The sub-index and the sign-plane-only storage.
    entry: Arc<PartitionEntry<S, Q>>,
    stage: SignStage,
    /// Prefilter acceptance by partition offset; `None` accepts every row.
    accept: Option<Vec<bool>>,
    _in_flight: PreparedPartitionGuard,
}

/// A probe ready for scoring.
enum LazyProbe<S: IvfSubIndex, Q: Quantization> {
    Eager(PreparedPartitionSearch<S, Q>),
    Empty,
    Ready(Box<LazyReadyProbe<S, Q>>),
}

struct LazyReadyProbe<S: IvfSubIndex, Q: Quantization> {
    sign: LazySignProbe<S, Q>,
    /// Sorted partition offsets whose ex rows were gathered.
    survivors: Vec<u32>,
    stage1: StagePruneCounts,
    gathered: GatheredEx,
}

impl<S: IvfSubIndex, Q: Quantization> LazyProbe<S, Q> {
    fn survivor_rows(&self) -> usize {
        match self {
            Self::Ready(ready) => ready.survivors.len(),
            Self::Eager(_) | Self::Empty => 0,
        }
    }
}

/// Query-wide inputs of the gathers.
struct LazyFetchContext {
    config: LayeredLazyConfig,
    heap_capacity: usize,
    /// Rows of the probes before each rank.
    rows_before: Vec<usize>,
    upper_bound: Option<f32>,
    layout: LayeredExLayout,
    progress: watch::Receiver<LazyProgress>,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<dyn MetricsCollector>,
    raw_query_context: Option<Arc<RabitRawQueryContext>>,
}

/// Query-wide inputs of the scorer.
#[derive(Clone)]
struct LazyScorer {
    key: ArrayRef,
    heap_capacity: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    approx_mode: ApproxMode,
    layout: LayeredExLayout,
    use_query_residual: bool,
    use_residual_scratch: bool,
    rq_search_cache: Option<Arc<RabitSearchCache>>,
    raw_query_context: Option<Arc<RabitRawQueryContext>>,
    scratch_pool: Arc<QueryScratchPool>,
    metrics: Arc<dyn MetricsCollector>,
    progress: Arc<watch::Sender<LazyProgress>>,
}

/// Aborts a task of the scan when its owner is dropped: the producer when the
/// query is dropped or fails, and a staging or gather step when the producer
/// drops its buffers. Promotions are not tracked and keep running on their own.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn scan_cancelled() -> Error {
    Error::internal("lazy layered scan ended before scoring every probe")
}

/// Run one staging or gather step as its own task, returning a future of its
/// result that aborts the task when dropped.
///
/// The producer keeps the steps in nested ordered buffers and stops polling
/// the staging buffer while the gather buffer is full. A step polled only
/// through its buffer would then freeze inside whatever fair resource it was
/// queued on: a cache that hands a freed slot of a bounded spill queue to its
/// oldest waiter can hand it to a frozen staging step, which never uses it,
/// while the head gather waits behind it forever. As a task, every step runs
/// to completion on its own. The buffers still start a step only when they
/// have room, so they bound the steps in flight and keep results in probe
/// order exactly as before.
fn spawn_scan_step<T: Send + 'static>(
    step: impl Future<Output = Result<T>> + Send + 'static,
) -> impl Future<Output = Result<T>> + Send + 'static {
    let task = tokio::spawn(step);
    let abort = AbortOnDrop(task.abort_handle());
    async move {
        let joined = task.await;
        drop(abort);
        match joined {
            Ok(result) => result,
            Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
            Err(_) => Err(scan_cancelled()),
        }
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> IVFIndex<S, Q> {
    /// The lazy scan settings for a newly opened index. Only layered indexes
    /// read (and validate) the environment.
    pub(super) fn layered_lazy_config_at_open(layered_rq: bool) -> Result<LayeredLazyConfig> {
        if !layered_rq {
            return Ok(LayeredLazyConfig::default());
        }
        LAYERED_LAZY_CONFIG.clone().map_err(Error::invalid_input)
    }

    /// Replace this index's lazy scan settings, avoiding the process-wide
    /// environment in concurrent tests.
    #[cfg(test)]
    pub(crate) fn set_layered_lazy_config_for_test(&self, config: LayeredLazyConfig) {
        *self
            .layered_lazy
            .lock()
            .unwrap_or_else(|err| err.into_inner()) = config;
    }

    /// Probes the lazy scan stages at once: the compute pool's width, like
    /// the eager scan's prepare window.
    fn lazy_prepare_parallelism(&self) -> usize {
        #[cfg(test)]
        {
            let parallelism = self
                .lazy_prepare_parallelism_for_test
                .load(std::sync::atomic::Ordering::Relaxed);
            if parallelism > 0 {
                return parallelism;
            }
        }
        get_num_compute_intensive_cpus().max(1)
    }

    /// Override the lazy scan's staging parallelism, or restore the default
    /// with 0, so tests stage several probes at once on hosts with few cores.
    #[cfg(test)]
    pub(crate) fn set_lazy_prepare_parallelism_for_test(&self, parallelism: usize) {
        self.lazy_prepare_parallelism_for_test
            .store(parallelism, std::sync::atomic::Ordering::Relaxed);
    }

    /// The settings for `query` when it can run the lazy scan. Queries that
    /// cannot run it take the eager scan and are counted by reason.
    pub(super) fn lazy_full_config(
        &self,
        query: &Query,
        has_raw_query_context: bool,
    ) -> Option<LayeredLazyConfig> {
        if !self.layered_rq {
            return None;
        }
        let config = *self
            .layered_lazy
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let stats = layered_stats::counters();
        let ineligible = if !config.enabled {
            &stats.ineligible_disabled
        } else if query.k == 0 {
            // Nothing to gather; the eager scan still waits for the prefilter
            // and loads every probe, and a `k == 0` query must keep doing so.
            &stats.ineligible_k_zero
        } else if !self.storage.supports_candidate_reads() {
            &stats.ineligible_remapper
        } else if query.rq_cascade_factor.is_some() {
            &stats.ineligible_cascade
        } else if query.rq_precision != RQPrecision::Full {
            &stats.ineligible_precision
        } else if query.refine_factor.is_some() {
            &stats.ineligible_refine
        } else if query.approx_mode == ApproxMode::Fast {
            &stats.ineligible_fast
        } else if self.use_query_residual || !has_raw_query_context {
            &stats.ineligible_residual
        } else {
            return Some(config);
        };
        ineligible.incr();
        None
    }

    /// Whether the high and low planes of every probed partition are in RAM,
    /// checked without counting as accesses.
    pub(super) async fn all_ex_planes_resident(
        &self,
        partitions: &UInt32Array,
        probes: Range<usize>,
    ) -> bool {
        for idx in probes {
            if !self
                .ex_planes_resident(partitions.value(idx) as usize)
                .await
            {
                return false;
            }
        }
        true
    }

    /// Score probes `probes` of `partitions` into one global heap with the
    /// lazy scan and return the heap as a result batch.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn search_partitions_lazy_full(
        self: Arc<Self>,
        query: Query,
        partitions: Arc<UInt32Array>,
        q_c_dists: Arc<Float32Array>,
        probes: Range<usize>,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
        raw_query_context: Option<Arc<RabitRawQueryContext>>,
        config: LayeredLazyConfig,
    ) -> Result<RecordBatch> {
        let stats = layered_stats::counters();
        stats.lazy_queries.incr();
        let started = Instant::now();
        pre_filter.wait_for_ready().await?;
        let heap_capacity = query.k;
        let probe_count = probes.len();
        let layout = self.storage.layered_ex_layout()?;
        let rows_before = probes
            .clone()
            .scan(0usize, |rows, idx| {
                let before = *rows;
                *rows += self.storage.partition_size(partitions.value(idx) as usize);
                Some(before)
            })
            .collect();
        let (progress_tx, progress_rx) = watch::channel(LazyProgress {
            scored: 0,
            full: false,
            threshold: f32::INFINITY,
        });
        let scorer = LazyScorer {
            key: query.key.clone(),
            heap_capacity,
            lower_bound: query.lower_bound,
            upper_bound: query.upper_bound,
            approx_mode: query.approx_mode,
            layout,
            use_query_residual: self.use_query_residual,
            use_residual_scratch: self.use_residual_scratch,
            rq_search_cache: self.rq_search_cache.clone(),
            raw_query_context: raw_query_context.clone(),
            scratch_pool: self.scratch_pool.clone(),
            metrics: metrics.clone(),
            progress: Arc::new(progress_tx),
        };
        let fetch_context = Arc::new(LazyFetchContext {
            config,
            heap_capacity,
            rows_before,
            upper_bound: query.upper_bound,
            layout,
            progress: progress_rx,
            pre_filter: pre_filter.clone(),
            metrics: metrics.clone(),
            raw_query_context: raw_query_context.clone(),
        });

        let (ready_tx, mut ready_rx) = mpsc::channel(LAZY_READY_CHANNEL_CAPACITY);
        let prepare_parallelism = self.lazy_prepare_parallelism();
        let stage_index = self.clone();
        let fetch_index = self.clone();
        let probe_start = probes.start;
        let producer = tokio::spawn(async move {
            // `map` spawns a step only when `buffered` pulls it, so at most
            // `prepare_parallelism` probes are staged or held staged, and the
            // gather window bounds the gathers, as with plain futures.
            let fetched = stream::iter(probes)
                .map(move |idx| {
                    let mut query = query.clone();
                    query.dist_q_c = q_c_dists.value(idx);
                    spawn_scan_step(stage_index.clone().lazy_stage_probe(
                        idx - probe_start,
                        partitions.value(idx) as usize,
                        query,
                        pre_filter.clone(),
                        metrics.clone(),
                        raw_query_context.clone(),
                    ))
                })
                .buffered(prepare_parallelism)
                .map(move |staged| {
                    let index = fetch_index.clone();
                    let context = fetch_context.clone();
                    spawn_scan_step(async move { index.lazy_fetch(staged?, &context).await })
                })
                .buffered(config.window.saturating_add(LAZY_FETCH_EXTRA_SLOTS));
            futures::pin_mut!(fetched);
            while let Some(probe) = fetched.next().await {
                let failed = probe.is_err();
                if ready_tx.send(probe).await.is_err() || failed {
                    break;
                }
            }
        });
        let _abort_producer = AbortOnDrop(producer.abort_handle());

        let mut heap = BinaryHeap::with_capacity(heap_capacity);
        let mut scored = 0usize;
        let mut first_full = false;
        loop {
            let waiting = Instant::now();
            let Some(probe) = ready_rx.recv().await else {
                break;
            };
            stats.scorer_wait_ns.add_elapsed(waiting);
            let mut batch = vec![probe?];
            while batch.len() < LAZY_SCORE_BATCH_MAX_PROBES {
                match ready_rx.try_recv() {
                    Ok(probe) => batch.push(probe?),
                    Err(_) => break,
                }
            }
            let survivor_rows: usize = batch.iter().map(LazyProbe::survivor_rows).sum();
            let has_eager = batch
                .iter()
                .any(|probe| matches!(probe, LazyProbe::Eager(_)));
            if !has_eager && survivor_rows <= config.inline_rows {
                stats.dispatch_inline.incr();
                Self::score_lazy_batch(&scorer, batch, &mut heap, &mut scored)?;
            } else {
                stats.dispatch_spawn.incr();
                let batch_scorer = scorer.clone();
                (heap, scored) = spawn_cpu(move || {
                    let (mut heap, mut scored) = (heap, scored);
                    Self::score_lazy_batch(&batch_scorer, batch, &mut heap, &mut scored)?;
                    Ok::<_, Error>((heap, scored))
                })
                .await?;
            }
            if !first_full && heap.len() >= heap_capacity {
                first_full = true;
                stats.time_to_first_full_ns.add_elapsed(started);
            }
        }
        match producer.await {
            Ok(()) => {}
            Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
            Err(_) => return Err(scan_cancelled()),
        }
        if scored != probe_count {
            return Err(Error::internal(format!(
                "lazy layered scan scored {scored} of {probe_count} probes"
            )));
        }
        if heap.len() <= GLOBAL_TOPK_INLINE_HEAP_LEN {
            Self::global_heap_to_batch(heap)
        } else {
            spawn_cpu(move || Self::global_heap_to_batch(heap)).await
        }
    }

    /// Load a probe's sign plane and run stage 1, or prepare it for the eager
    /// scan when both ex planes are resident.
    async fn lazy_stage_probe(
        self: Arc<Self>,
        rank: usize,
        partition_id: usize,
        query: Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
        raw_query_context: Option<Arc<RabitRawQueryContext>>,
    ) -> Result<LazyStaged<S, Q>> {
        let stats = layered_stats::counters();
        if self.ex_planes_resident(partition_id).await {
            if self.storage.partition_size(partition_id) == 0 {
                stats.empty.incr(rank);
            } else {
                stats.eager_resident.incr(rank);
            }
            return Ok(LazyStaged::Eager(
                self.prepare_partition_without_prefilter_wait(
                    partition_id,
                    &query,
                    pre_filter,
                    metrics.as_ref(),
                    raw_query_context,
                )
                .await?,
            ));
        }
        let loading = Instant::now();
        let io_stats = metrics.io_stats();
        let (index, sign) = tokio::try_join!(
            self.load_sub_index(partition_id, io_stats.clone()),
            self.storage
                .load_sign_stage(partition_id, &self.index_cache, io_stats),
        )?;
        stats.sign_load_ns.add_elapsed(loading);
        let mut entry = PartitionEntry::new(index, sign);
        // Like the eager layered load, a query-assembled entry is never cached whole.
        entry.cache_whole_partition = false;
        let entry = Arc::new(entry);
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &entry, pre_filter)
                .await?;
        if entry.storage.is_empty() {
            stats.empty.incr(rank);
            return Ok(LazyStaged::Empty);
        }
        let in_flight = self.prepared_partitions.track();
        stats.stage1_dispatches.incr();
        let stage_entry = entry.clone();
        let stage_query = query.clone();
        let rq_search_cache = self.rq_search_cache.clone();
        let scratch_pool = self.scratch_pool.clone();
        let stage = spawn_cpu(move || {
            let computing = Instant::now();
            let stage = Self::lazy_sign_stage(
                &stage_entry,
                partition_id,
                &stage_query,
                pre_filter.as_ref(),
                rq_search_cache.as_deref(),
                raw_query_context.as_deref(),
                &scratch_pool,
            );
            layered_stats::counters()
                .stage1_cpu_ns
                .add_elapsed(computing);
            stage
        })
        .await?;
        let Some((stage, accept)) = stage else {
            stats.gating_off.incr(rank);
            return Ok(LazyStaged::GatingOff {
                partition_id,
                query,
            });
        };
        stats.lazy_probes.incr(rank);
        Ok(LazyStaged::Lazy(Box::new(LazySignProbe {
            rank,
            partition_id,
            dist_q_c: query.dist_q_c,
            entry,
            stage,
            accept,
            _in_flight: in_flight,
        })))
    }

    /// Whether a partition's high and low planes are in RAM, checked without
    /// counting as accesses. An empty partition has no ex rows to gather, so
    /// it counts as resident: the lazy scan never touches its (empty) plane
    /// entries, and once they were evicted it would otherwise never take the
    /// eager scan again.
    async fn ex_planes_resident(&self, partition: usize) -> bool {
        if self.storage.partition_size(partition) == 0 {
            return true;
        }
        for plane in [1, 2] {
            if !self
                .index_cache
                .peek_resident_with_key(&PlaneKey { partition, plane })
                .await
            {
                return false;
            }
        }
        true
    }

    /// Stage 1 of one partition: the binary inner products and lower bounds
    /// of every row, computed with the calculator the eager scan builds, and
    /// the prefilter's row acceptance. `None` when the partition cannot gate
    /// on the lower bound.
    #[allow(clippy::type_complexity)]
    fn lazy_sign_stage(
        entry: &PartitionEntry<S, Q>,
        partition_id: usize,
        query: &Query,
        pre_filter: &dyn PreFilter,
        rq_search_cache: Option<&RabitSearchCache>,
        raw_query_context: Option<&RabitRawQueryContext>,
        scratch_pool: &QueryScratchPool,
    ) -> Result<Option<(SignStage, Option<Vec<bool>>)>> {
        let storage = rabit_storage(&entry.storage)?;
        let residual = Self::query_context_for_scratch(
            false,
            false,
            partition_id,
            None,
            rotated_partition_centroid_slice(rq_search_cache, partition_id),
            raw_query_context,
        )?;
        let mut stage = SignStage::default();
        let gated = scratch_pool.with_scratch(|scratch| {
            let calc = storage.dist_calculator_with_scratch(
                query.key.clone(),
                query.dist_q_c,
                residual,
                &mut scratch.query_f32,
                DistanceCalculatorOptions {
                    rq_precision: query.rq_precision,
                    approx_mode: query.approx_mode,
                },
            );
            calc.full_sign_stage_with_scratch(
                &mut stage,
                &mut scratch.u16,
                &mut scratch.u8,
                &mut scratch.u32,
            )
            .is_ok()
        });
        if !gated {
            return Ok(None);
        }
        // The eager scan takes its filtered path exactly when the prefilter is not empty.
        let accept = (!pre_filter.is_empty()).then(|| {
            let mask = pre_filter.mask();
            storage
                .row_ids()
                .map(|&row_id| mask.selected(row_id))
                .collect()
        });
        Ok(Some((stage, accept)))
    }

    /// Wait until a probe may be issued, select its survivors against the
    /// published threshold, and gather their ex rows.
    async fn lazy_fetch(
        self: Arc<Self>,
        staged: LazyStaged<S, Q>,
        context: &LazyFetchContext,
    ) -> Result<LazyProbe<S, Q>> {
        let probe = match staged {
            LazyStaged::Eager(prepared) => return Ok(LazyProbe::Eager(prepared)),
            LazyStaged::Empty => return Ok(LazyProbe::Empty),
            LazyStaged::GatingOff {
                partition_id,
                query,
            } => {
                return Ok(LazyProbe::Eager(
                    self.prepare_partition_without_prefilter_wait(
                        partition_id,
                        &query,
                        context.pre_filter.clone(),
                        context.metrics.as_ref(),
                        context.raw_query_context.clone(),
                    )
                    .await?,
                ));
            }
            LazyStaged::Lazy(probe) => *probe,
        };
        let stats = layered_stats::counters();
        let rank = probe.rank;
        let gating = Instant::now();
        let mut progress = context.progress.clone();
        let issue = if context.rows_before[rank] < context.heap_capacity {
            // The earlier probes cannot fill the heap, so the threshold is +inf at scoring.
            stats.certain_dense.incr(rank);
            *progress.borrow()
        } else {
            let window = context.config.window;
            let gate = *progress
                .wait_for(|progress| progress.scored.saturating_add(window) >= rank)
                .await
                .map_err(|_| scan_cancelled())?;
            if gate.full || gate.scored >= rank {
                gate
            } else {
                // An infinite threshold would gather every row; wait for a
                // finite one, or for this probe's turn.
                stats.deferred_issues.incr();
                let issue = *progress
                    .wait_for(|progress| progress.full || progress.scored >= rank)
                    .await
                    .map_err(|_| scan_cancelled())?;
                if !issue.full {
                    stats.serial_waits.incr();
                }
                issue
            }
        };
        stats.gate_wait_ns.add_elapsed(gating);
        let staleness = rank.saturating_sub(issue.scored) as u64;
        stats.staleness_sum.add(staleness);
        stats.staleness_max.observe(staleness);

        let mut survivors = Vec::new();
        let mut stage1 = StagePruneCounts::default();
        select_full_survivors(
            &probe.stage.lower_bounds,
            probe.accept.as_deref(),
            context.upper_bound,
            issue.full.then_some(issue.threshold),
            &mut survivors,
            &mut stage1,
        );
        stats.stage1_candidates.add(rank, stage1.candidates as u64);
        stats
            .stage1_pruned_ub
            .add(rank, stage1.pruned_upper_bound as u64);
        stats
            .stage1_pruned_heap
            .add(rank, stage1.pruned_heap as u64);
        stats.rows_fetched.add(rank, survivors.len() as u64);

        let partition_rows = probe.entry.storage.len();
        let mut planes = [PlaneSource::Sparse; 2];
        for (slot, plane) in [1u8, 2].into_iter().enumerate() {
            planes[slot] = if self
                .index_cache
                .peek_resident_with_key(&PlaneKey {
                    partition: probe.partition_id,
                    plane,
                })
                .await
            {
                PlaneSource::Resident
            } else {
                plan_plane_gather(
                    &survivors,
                    partition_rows,
                    context.layout.row_bytes[slot],
                    &context.config,
                )
            };
        }
        let mut gathered = self
            .storage
            .gather_ex_rows(
                probe.partition_id,
                &survivors,
                GatherPlan { planes },
                &context.config,
                &self.index_cache,
                context.metrics.io_stats(),
            )
            .await?;
        record_gather_sources(rank, &gathered);
        stats.origin_row_reads.add(gathered.origin_row_reads as u64);
        for ticket in std::mem::take(&mut gathered.promotions) {
            self.spawn_lazy_promotion(ticket);
        }
        Ok(LazyProbe::Ready(Box::new(LazyReadyProbe {
            sign: probe,
            survivors,
            stage1,
            gathered,
        })))
    }

    /// Load a whole ex plane in the background, off the query's critical
    /// path, admitting it as the eager load would.
    fn spawn_lazy_promotion(self: &Arc<Self>, ticket: LazyPromotionTicket) {
        let index = self.clone();
        tokio::spawn(async move {
            let stats = layered_stats::counters();
            match index
                .storage
                .load_plane_entry(ticket.partition, ticket.plane, &index.index_cache, None)
                .await
            {
                Ok(batch) => {
                    stats
                        .promotion_bytes
                        .add(batch.as_ref().deep_size_of() as u64);
                    let resident = index
                        .index_cache
                        .peek_resident_with_key(&PlaneKey {
                            partition: ticket.partition,
                            plane: ticket.plane,
                        })
                        .await;
                    if resident {
                        stats.promotions_completed.incr();
                    } else {
                        // Loaded but evicted before it could serve a query.
                        stats.promotions_skipped.incr();
                    }
                }
                Err(err) => {
                    stats.promotions_skipped.incr();
                    tracing::warn!(
                        partition = ticket.partition,
                        plane = ticket.plane,
                        "lazy layered plane promotion failed: {err}"
                    );
                }
            }
            drop(ticket);
        });
    }

    /// Score `batch` in probe order into `heap`, publishing progress after
    /// every probe and the threshold as soon as the heap fills.
    fn score_lazy_batch(
        scorer: &LazyScorer,
        batch: Vec<LazyProbe<S, Q>>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        scored: &mut usize,
    ) -> Result<()> {
        let stats = layered_stats::counters();
        let scoring = Instant::now();
        let result = batch.into_iter().try_for_each(|probe| {
            match probe {
                LazyProbe::Eager(prepared) => scorer.scratch_pool.with_scratch(|scratch| {
                    Self::accumulate_prepared_partition_search(
                        scorer.use_query_residual,
                        scorer.use_residual_scratch,
                        prepared,
                        heap,
                        scratch,
                        scorer.metrics.as_ref(),
                    )
                })?,
                LazyProbe::Empty => {}
                LazyProbe::Ready(ready) => Self::score_ready_probe(scorer, *ready, heap)?,
            }
            *scored += 1;
            let top = (heap.len() >= scorer.heap_capacity)
                .then(|| heap.peek().map(|node| node.dist.0))
                .flatten();
            let scored = *scored;
            scorer.progress.send_modify(|progress| {
                progress.scored = scored;
                if let Some(top) = top {
                    progress.full = true;
                    progress.threshold = top;
                }
            });
            Ok(())
        });
        stats.stage2_cpu_ns.add_elapsed(scoring);
        result
    }

    /// Stage 2 of one partition: check that the gather covers every row the
    /// live heap can still take, then replay the survivors through the eager
    /// scan's per-row step.
    fn score_ready_probe(
        scorer: &LazyScorer,
        ready: LazyReadyProbe<S, Q>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
    ) -> Result<()> {
        let stats = layered_stats::counters();
        let LazyReadyProbe {
            sign,
            survivors,
            stage1,
            gathered,
        } = ready;
        let rank = sign.rank;
        let storage = rabit_storage(&sign.entry.storage)?;
        scorer.metrics.record_comparisons(storage.len());

        let live_threshold = (heap.len() >= scorer.heap_capacity)
            .then(|| heap.peek().map(|node| node.dist.0))
            .flatten();
        let mut needed = Vec::new();
        let mut needed_counts = StagePruneCounts::default();
        select_full_survivors(
            &sign.stage.lower_bounds,
            sign.accept.as_deref(),
            scorer.upper_bound,
            live_threshold,
            &mut needed,
            &mut needed_counts,
        );
        let missing = count_missing(&needed, &survivors);
        if missing > 0 {
            stats.needed_not_fetched.add(missing as u64);
            return Err(Error::internal(format!(
                "lazy layered scan of partition {} did not gather {missing} rows the heap still needs",
                sign.partition_id
            )));
        }

        let expected_rows = match gathered.index {
            ExIndex::Identity => storage.len(),
            ExIndex::Compact => survivors.len(),
        };
        if gathered.high.num_rows() != expected_rows || gathered.low.num_rows() != expected_rows {
            return Err(Error::internal(format!(
                "lazy layered gather of partition {} returned {} high and {} low rows, expected {expected_rows}",
                sign.partition_id,
                gathered.high.num_rows(),
                gathered.low.num_rows()
            )));
        }
        let rows = survivors
            .iter()
            .enumerate()
            .map(|(position, &offset)| SurvivorRow {
                ex_index: match gathered.index {
                    ExIndex::Identity => offset,
                    ExIndex::Compact => position as u32,
                },
                row_id: storage.row_id(offset),
                binary_ip: sign.stage.binary_ips[offset as usize],
                lower_bound: sign.stage.lower_bounds[offset as usize],
            })
            .collect::<Vec<_>>();
        let ex_rows = ExRows::from_plane_batches(
            &gathered.high,
            &gathered.low,
            scorer.layout.rotated_dim,
            scorer.layout.num_bits,
        )?;
        let residual = Self::query_context_for_scratch(
            scorer.use_query_residual,
            scorer.use_residual_scratch,
            sign.partition_id,
            None,
            rotated_partition_centroid_slice(scorer.rq_search_cache.as_deref(), sign.partition_id),
            scorer.raw_query_context.as_deref(),
        )?;
        let counters = scorer.scratch_pool.with_scratch(|scratch| {
            let calc = storage.dist_calculator_with_ex_rows(
                scorer.key.clone(),
                sign.dist_q_c,
                residual,
                &mut scratch.query_f32,
                DistanceCalculatorOptions {
                    rq_precision: RQPrecision::Full,
                    approx_mode: scorer.approx_mode,
                },
                ex_rows,
            )?;
            Ok::<_, Error>(calc.accumulate_survivor_rows(
                scorer.heap_capacity,
                scorer.lower_bound,
                scorer.upper_bound,
                &rows,
                stage1,
                heap,
                |threshold| {
                    scorer.progress.send_modify(|progress| {
                        progress.full = true;
                        progress.threshold = threshold;
                    });
                },
            ))
        })?;
        stats.stage2_exact.add(rank, counters.exact as u64);
        stats.stage2_pruned_live.add(
            rank,
            counters.pruned_heap.saturating_sub(stage1.pruned_heap) as u64,
        );
        stats
            .stage2_exact_rejected
            .add(rank, counters.exact_rejected as u64);
        Ok(())
    }
}

fn rabit_storage<T: VectorStore>(storage: &T) -> Result<&RabitQuantizationStorage> {
    storage
        .as_any()
        .downcast_ref::<RabitQuantizationStorage>()
        .ok_or_else(|| Error::internal("lazy layered scan requires RaBitQ storage"))
}

/// Rows of the sorted `needed` missing from the sorted `gathered`.
fn count_missing(needed: &[u32], gathered: &[u32]) -> usize {
    let mut gathered = gathered.iter().peekable();
    needed
        .iter()
        .filter(|&&row| {
            while gathered.next_if(|&&candidate| candidate < row).is_some() {}
            gathered.next_if_eq(&&row).is_none()
        })
        .count()
}

fn record_gather_sources(rank: usize, gathered: &GatheredEx) {
    let stats = layered_stats::counters();
    let counters = [
        [&stats.high_resident, &stats.high_whole, &stats.high_sparse],
        [&stats.low_resident, &stats.low_whole, &stats.low_sparse],
    ];
    for (source, counters) in gathered.sources.iter().zip(counters) {
        let counter = match source {
            Some(PlaneSource::Resident) => counters[0],
            Some(PlaneSource::Whole) => counters[1],
            Some(PlaneSource::Sparse) => counters[2],
            None => continue,
        };
        counter.incr(rank);
    }
}

#[cfg(test)]
mod tests {
    use super::count_missing;

    #[test]
    fn count_missing_compares_sorted_offsets() {
        assert_eq!(count_missing(&[], &[1, 2]), 0);
        assert_eq!(count_missing(&[1, 3], &[1, 2, 3]), 0);
        assert_eq!(count_missing(&[0, 3, 7], &[1, 3, 5]), 2);
        assert_eq!(count_missing(&[4], &[]), 1);
    }
}
