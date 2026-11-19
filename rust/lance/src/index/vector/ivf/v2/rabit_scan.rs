// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Session-scoped execution strategies for native RaBitQ scans.
//!
//! Strategies choose how to schedule partition reads. The context keeps
//! filtering, query preparation and scoring on the same implementation as
//! the built-in full scan. No strategy or session state is serialized into
//! an index or its cached file metadata.

use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Float32Array, RecordBatch, UInt32Array};
use async_trait::async_trait;
use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, stream};
use lance_core::cache::WeakLanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::metrics::{IndexTimer, IndexTiming, MetricsCollector};
use lance_index::prefilter::PreFilter;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::scan_stats;
use lance_index::vector::bq::storage::{
    ExRows, RabitPruneCounters, SignStage, StagePruneCounts, SurvivorRow, select_full_survivors,
};
use lance_index::vector::flat::index::FlatIndex;
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::storage::{
    DistanceCalculatorOptions, ExIndex, GatheredRabitEx, IndexFileKey, IvfQuantizationStorage,
    OriginLatencyClass, RabitExLayout, RabitRawQueryContext, VectorStore, sequential_plane_loads,
};
use lance_index::vector::{
    ApproxMode, PartitionSearchControl, Query, VECTOR_RESULT_SCHEMA, VectorIndex,
};

use super::{
    IVFIndex, PartitionEntry, PreparedPartitionGuard, PreparedPartitionSearch,
    global_heap_to_batch, rotated_partition_centroid_slice,
};

pub(super) type RabitIndex = IVFIndex<FlatIndex, RabitQuantizer>;

/// Creates runtime execution state for an opened RaBitQ index with current plane storage.
///
/// Installed with [`crate::session::Session::with_rabit_scan_factory`].
/// Called on fresh opens and on reconstruction from cached file metadata;
/// implementations must not assume that an earlier session's strategy remains installed.
pub trait RabitScanFactory: Debug + Send + Sync {
    /// Bind the current session's strategy to this file and resolved origin class.
    /// Errors fail index opening rather than silently selecting another implementation.
    fn create(&self, file: &IndexFileKey, origin: OriginLatencyClass)
    -> Result<Arc<dyn RabitScan>>;
}

/// Cache granularity requested by an installed strategy for native RaBitQ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RabitCacheLayout {
    /// Preserve the ordinary native whole-partition cache and prewarm path.
    #[default]
    Partition,
    /// Share sign/ex entries between prewarm, whole-partition fallback and
    /// candidate gathers, without changing the persisted index format.
    Planes,
}

/// An optional execution strategy selected once per multi-partition query.
#[async_trait]
pub trait RabitScan: Debug + Send + Sync {
    /// Select native cache granularity once the index binds this strategy.
    /// The default preserves existing strategy and OSS behavior.
    fn cache_layout(&self) -> RabitCacheLayout {
        RabitCacheLayout::Partition
    }

    /// Return a result batch, or `None` to use the built-in full scan.
    ///
    /// Returning `None` must leave the query unexecuted. Execution errors must
    /// be returned as errors, since retrying on another path can hide failures.
    async fn search(&self, context: Arc<RabitScanContext>) -> Result<Option<RecordBatch>>;

    /// Construct a stream with one batch per probe for partition-controlled search.
    ///
    /// Put asynchronous preparation in the returned stream so Lance can apply
    /// stop checks before polling it. Any bounded prefetch must be owned by the
    /// stream and cancelled when it is dropped.
    ///
    /// Batches follow `context.probes` order, including empty batches. Each batch
    /// must preserve the built-in partition search's candidate budget, filtering,
    /// bounds and tie semantics. A cumulative global heap cannot replace these batches:
    /// their row counts drive the caller's stopping decision.
    ///
    /// Lance owns stop checks and calls [`PartitionSearchControl::record_batch`]
    /// exactly once for each accepted batch. The strategy must bound prefetch and
    /// cancel outstanding work when its stream is dropped. Both the stream and
    /// its batches must use [`VECTOR_RESULT_SCHEMA`]. Each batch has at most
    /// `k * refine_factor` rows (an unset refinement factor means 1).
    ///
    /// `None` leaves the search unexecuted and selects the built-in partition
    /// stream. Errors propagate without retrying on another strategy. Existing
    /// implementations can retain that fallback by omitting this method:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::RecordBatch;
    /// use lance::index::vector::ivf::v2::rabit_scan::{RabitScan, RabitScanContext};
    /// use lance_core::Result;
    ///
    /// #[derive(Debug)]
    /// struct BuiltInScan;
    /// #[async_trait::async_trait]
    /// impl RabitScan for BuiltInScan {
    ///     async fn search(&self, _: Arc<RabitScanContext>) -> Result<Option<RecordBatch>> {
    ///         Ok(None)
    ///     }
    /// }
    /// ```
    fn search_partition_stream(
        &self,
        _context: Arc<RabitScanContext>,
    ) -> Result<Option<SendableRecordBatchStream>> {
        Ok(None)
    }
}

/// Keep the streaming strategy's result accounting at the same boundary as
/// built-in partition search. In particular, empty batches still represent a
/// searched partition, and an incomplete stream must not silently lose rows.
pub(super) fn controlled_partition_stream(
    source: SendableRecordBatchStream,
    expected_batches: usize,
    candidate_limit: usize,
    control: Arc<dyn PartitionSearchControl>,
) -> Result<SendableRecordBatchStream> {
    if source.schema().as_ref() != VECTOR_RESULT_SCHEMA.as_ref() {
        return Err(Error::internal(format!(
            "RaBitQ partition stream schema must be {:?}, got {:?}",
            VECTOR_RESULT_SCHEMA.as_ref(),
            source.schema()
        )));
    }
    let batches = stream::unfold(Some((source, 0usize)), move |state| {
        let control = control.clone();
        async move {
            let (mut source, emitted) = state?;
            if control.should_stop() {
                return None;
            }
            // An error from a poll already in flight remains an error even if
            // another segment satisfies the shared budget during that poll.
            let next = match source.next().await {
                Some(Ok(batch)) => Some(batch),
                Some(Err(error)) => return Some((Err(error), None)),
                None => None,
            };
            if control.should_stop() {
                return None;
            }
            let batch = match next {
                Some(batch) => batch,
                None if emitted == expected_batches => return None,
                None => {
                    let error = Error::internal(format!(
                        "RaBitQ partition stream ended after {emitted} batches; expected {expected_batches}"
                    ));
                    return Some((Err(error.into()), None));
                }
            };
            let invalid = if emitted == expected_batches {
                Some(format!(
                    "RaBitQ partition stream produced more than {expected_batches} batches"
                ))
            } else if batch.schema().as_ref() != VECTOR_RESULT_SCHEMA.as_ref() {
                Some(format!(
                    "RaBitQ partition batch {emitted} has an unexpected schema: {:?}",
                    batch.schema()
                ))
            } else if batch.num_rows() > candidate_limit {
                Some(format!(
                    "RaBitQ partition batch {emitted} has {} rows, exceeding candidate limit {candidate_limit}",
                    batch.num_rows()
                ))
            } else {
                None
            };
            if let Some(message) = invalid {
                return Some((Err(Error::internal(message).into()), None));
            }
            control.record_batch(&batch);
            // Release prefetched work as soon as this batch meets the budget,
            // without waiting for the consumer to ask for another batch.
            let state = if control.should_stop() {
                None
            } else {
                Some((source, emitted + 1))
            };
            Some((Ok(batch), state))
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        VECTOR_RESULT_SCHEMA.clone(),
        batches,
    )))
}

/// Native preparation and scoring inputs for one query.
///
/// The index and its Arrow buffers are shared; constructing this context does
/// not copy partitions. Strategy calls operate on partitions or row batches,
/// keeping the inner scoring loop concrete and free of virtual calls.
#[derive(Clone)]
pub struct RabitScanContext {
    index: Arc<RabitIndex>,
    raw_query: Option<Arc<RabitRawQueryContext>>,
    /// Original query, before setting each partition's centroid distance.
    pub query: Query,
    /// Ordered partition identifiers, paired with `centroid_distances`.
    pub partitions: Arc<UInt32Array>,
    /// Query-to-centroid distances in `partitions` order.
    pub centroid_distances: Arc<Float32Array>,
    /// The range of partition ranks this invocation must score.
    pub probes: Range<usize>,
    /// Query filter, including deletions. Wait for readiness before preparation.
    pub pre_filter: Arc<dyn PreFilter>,
    /// The calling query's metrics, including its I/O accounting.
    pub metrics: Arc<dyn MetricsCollector>,
}

/// A prepared full partition whose ownership bounds its buffer lifetime.
pub struct RabitPreparedPartition(PreparedRabitPartition);

enum PreparedRabitPartition {
    Ordinary(PreparedPartitionSearch<FlatIndex, RabitQuantizer>),
    Planes {
        sign: Box<RabitSignPartition>,
        ex: RecordBatch,
    },
}

/// Candidate and tie semantics used when preparing and scoring gathered rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RabitScanScope {
    /// A single heap shared across the query's probes.
    Query,
    /// An independent native candidate batch for each controlled probe.
    Partition,
}

/// A sign-only partition with native lower bounds and filter acceptance.
pub struct RabitSignPartition {
    scope: RabitScanScope,
    entry: Arc<PartitionEntry<FlatIndex, RabitQuantizer>>,
    partition_id: usize,
    dist_q_c: f32,
    _in_flight: PreparedPartitionGuard,
    /// Binary inner products and lower bounds in partition row order.
    pub stage: SignStage,
    /// Accepted physical partition offsets, excluding FRI tombstones as well
    /// as rows rejected by the query prefilter. `None` accepts every row.
    pub accept: Option<Vec<bool>>,
}

impl RabitSignPartition {
    /// Number of rows in the original partition, including filtered rows.
    pub fn num_rows(&self) -> usize {
        self.entry.storage.len()
    }
}

impl RabitScanContext {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        index: Arc<RabitIndex>,
        raw_query: Option<Arc<RabitRawQueryContext>>,
        query: Query,
        partitions: Arc<UInt32Array>,
        centroid_distances: Arc<Float32Array>,
        probes: Range<usize>,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Self {
        Self {
            index,
            raw_query,
            query,
            partitions,
            centroid_distances,
            probes,
            pre_filter,
            metrics,
        }
    }

    /// Storage primitives for reading whole planes or selected rows.
    pub fn storage(&self) -> &IvfQuantizationStorage<RabitQuantizer> {
        &self.index.storage
    }

    /// Cache namespace for immutable planes and code-only entries of this file.
    /// It includes the storage location and is independent of FRI translation.
    pub fn cache(&self) -> &WeakLanceCache {
        &self.index.raw_cache
    }

    /// Whether native full scoring can use raw-query sign bounds.
    pub fn has_raw_query_bounds(&self) -> bool {
        !self.index.use_query_residual && self.raw_query.is_some()
    }

    /// Prepare whole ex codes with the same address view as candidate gathers.
    ///
    /// With the plane cache layout, current native Full/Accurate storage keeps
    /// encoded rows in file order; FRI only translates the sign view's addresses.
    /// Partition cache layouts, other encodings and Fast mode retain their
    /// ordinary partition reader so preparation uses the same entries as prewarm.
    /// Wait for the query filter before preparing either representation.
    ///
    /// ```
    /// # use std::collections::BinaryHeap;
    /// # use lance::index::vector::ivf::v2::rabit_scan::RabitScanContext;
    /// # async fn score(context: &RabitScanContext, partition: usize, distance: f32) -> lance::Result<()> {
    /// context.pre_filter.wait_for_ready().await?;
    /// let prepared = context.prepare_partition(partition, distance).await?;
    /// let mut heap = BinaryHeap::new();
    /// context.score_partition(prepared, &mut heap)?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn prepare_partition(
        &self,
        partition_id: usize,
        dist_q_c: f32,
    ) -> Result<RabitPreparedPartition> {
        if self.index.uses_rabit_plane_cache()
            && self.storage().supports_candidate_reads()
            && self.has_raw_query_bounds()
            && self.query.approx_mode != ApproxMode::Fast
        {
            let _prepare_timer =
                IndexTimer::new(self.metrics.as_ref(), IndexTiming::PartitionPrepare);
            let context = Arc::new(self.clone());
            scan_stats::counters().stage1_dispatches.incr();
            let sign = context.prepare_sign_partition(partition_id, dist_q_c);
            let ex =
                self.storage()
                    .load_plane(partition_id, 1, self.cache(), self.metrics.io_stats());
            let (sign, ex) = if sequential_plane_loads()? || self.cache().plane_admission_gated() {
                (sign.await?, ex.await?)
            } else {
                tokio::try_join!(sign, ex)?
            };
            let sign = sign.ok_or_else(|| {
                Error::internal(format!(
                    "current native RaBitQ partition {partition_id} has no full-scoring sign bounds"
                ))
            })?;
            return Ok(RabitPreparedPartition(PreparedRabitPartition::Planes {
                sign: Box::new(sign),
                ex,
            }));
        }
        let mut query = self.query.clone();
        query.dist_q_c = dist_q_c;
        self.index
            .prepare_partition(
                partition_id,
                Arc::new(query),
                dist_q_c,
                self.pre_filter.clone(),
                self.metrics.as_ref(),
                self.raw_query.clone(),
                false,
            )
            .await
            .map(|prepared| RabitPreparedPartition(PreparedRabitPartition::Ordinary(prepared)))
    }

    /// Read a sign plane and calculate its native bounds on the CPU pool.
    ///
    /// `None` means this partition has no usable sign bounds and must be
    /// prepared whole. The caller must first wait for `pre_filter` readiness.
    pub async fn prepare_sign_partition(
        self: &Arc<Self>,
        partition_id: usize,
        dist_q_c: f32,
    ) -> Result<Option<RabitSignPartition>> {
        self.prepare_sign_partition_with_scope(partition_id, dist_q_c, RabitScanScope::Query)
            .await
    }

    /// Prepare bounds with the scoring contract of a query or partition heap.
    /// Partition scope preserves scalar filter precision and native local tie order.
    /// Its bounds are negative infinity: gather every accepted row, since native
    /// partition search does not use query-wide approximate distance pruning.
    /// The caller must wait for the query filter before preparing either scope.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::index::vector::ivf::v2::rabit_scan::{RabitScanContext, RabitScanScope};
    /// # async fn prepare(context: Arc<RabitScanContext>) -> lance::Result<()> {
    /// context.pre_filter.wait_for_ready().await?;
    /// let sign = context.prepare_sign_partition_with_scope(0, 0.0, RabitScanScope::Partition).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn prepare_sign_partition_with_scope(
        self: &Arc<Self>,
        partition_id: usize,
        dist_q_c: f32,
        scope: RabitScanScope,
    ) -> Result<Option<RabitSignPartition>> {
        let loading = Instant::now();
        let io_stats = self.metrics.io_stats();
        let (index, sign) = tokio::try_join!(
            self.index.load_sub_index(partition_id, io_stats.clone()),
            self.storage()
                .load_sign_stage(partition_id, self.cache(), io_stats),
        )?;
        scan_stats::counters().sign_load_ns.add_elapsed(loading);
        let mut entry = PartitionEntry::new(index, sign);
        entry.cache_whole_partition = false;
        let entry = Arc::new(entry);
        let pre_filter = RabitIndex::prefilter_for_partition(
            &self.index.index_cache,
            partition_id,
            &entry,
            self.pre_filter.clone(),
        )
        .await?;
        let context = self.clone();
        let in_flight = self.index.prepared_partitions.track();
        lance_core::utils::tokio::spawn_cpu(move || {
            let computing = Instant::now();
            let storage = &entry.storage;
            let residual = RabitIndex::query_context_for_scratch(
                false,
                false,
                partition_id,
                None,
                rotated_partition_centroid_slice(
                    context.index.rq_search_cache.as_deref(),
                    partition_id,
                ),
                context.raw_query.as_deref(),
            )?;
            let mut stage = SignStage::default();
            let gated = context.index.scratch_pool.with_scratch(|scratch| {
                let calc = storage.dist_calculator_with_scratch(
                    context.query.key.clone(),
                    dist_q_c,
                    residual,
                    &mut scratch.query_f32,
                    DistanceCalculatorOptions {
                        approx_mode: context.query.approx_mode,
                    },
                );
                if scope == RabitScanScope::Partition && !pre_filter.is_empty() {
                    calc.filtered_partition_sign_stage_with_scratch(
                        &mut stage,
                        &mut scratch.u16,
                        &mut scratch.u8,
                        &mut scratch.u32,
                    )
                } else {
                    calc.full_sign_stage_with_scratch(
                        &mut stage,
                        &mut scratch.u16,
                        &mut scratch.u8,
                        &mut scratch.u32,
                    )
                }
                .is_ok()
            });
            if !gated {
                scan_stats::counters().stage1_cpu_ns.add_elapsed(computing);
                return Ok(None);
            }
            // FRI deletions must be excluded even when there is no query
            // filter: their slots remain so ex gathers keep file offsets.
            let needs_mask = !pre_filter.is_empty()
                || storage
                    .row_ids()
                    .any(|&row_id| row_id == RowAddress::TOMBSTONE_ROW);
            let accept = needs_mask.then(|| {
                let mask = pre_filter.mask();
                storage
                    .row_ids()
                    .map(|&row_id| row_id != RowAddress::TOMBSTONE_ROW && mask.selected(row_id))
                    .collect()
            });
            scan_stats::counters().stage1_cpu_ns.add_elapsed(computing);
            if scope == RabitScanScope::Partition {
                // The query heap's approximate pruning can skip intermediate
                // native candidates and change the partition batch or its ties.
                stage.lower_bounds.fill(f32::NEG_INFINITY);
                context.metrics.record_comparisons(storage.len());
            }
            Ok(Some(RabitSignPartition {
                scope,
                entry,
                partition_id,
                dist_q_c,
                _in_flight: in_flight,
                stage,
                accept,
            }))
        })
        .await
    }

    /// Execute native partition search when sign-bound pruning is unavailable.
    /// The returned batch preserves the built-in partition candidate order.
    ///
    /// ```
    /// # use lance::index::vector::ivf::v2::rabit_scan::RabitScanContext;
    /// # async fn fallback(context: &RabitScanContext) -> lance::Result<()> {
    /// let batch = context.search_partition(0, 0.0).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_partition(
        &self,
        partition_id: usize,
        dist_q_c: f32,
    ) -> Result<RecordBatch> {
        let mut query = self.query.clone();
        query.dist_q_c = dist_q_c;
        self.index
            .search_in_partition(
                partition_id,
                &query,
                self.pre_filter.clone(),
                self.metrics.as_ref(),
            )
            .await
    }

    /// Accumulate a prepared partition into a query-local heap using native scoring.
    pub fn score_partition(
        &self,
        prepared: RabitPreparedPartition,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
    ) -> Result<()> {
        match prepared.0 {
            PreparedRabitPartition::Ordinary(prepared) => {
                self.index.scratch_pool.with_scratch(|scratch| {
                    prepared.accumulate_into(heap, scratch, self.metrics.as_ref())
                })
            }
            PreparedRabitPartition::Planes { sign, ex } => {
                let _search_timer = IndexTimer::new(self.metrics.as_ref(), IndexTiming::Search);
                let refine_factor = self.query.refine_factor.unwrap_or(1) as usize;
                let candidate_count = self.query.k.checked_mul(refine_factor).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "RaBitQ candidate count overflow: k={}, refine_factor={refine_factor}",
                        self.query.k
                    ))
                })?;
                if candidate_count == 0 {
                    return Ok(());
                }
                let mut survivors = Vec::with_capacity(candidate_count.min(sign.num_rows()));
                let mut stage1 = StagePruneCounts::default();
                select_full_survivors(
                    &sign.stage.lower_bounds,
                    sign.accept.as_deref(),
                    self.query.upper_bound,
                    heap.peek()
                        .filter(|_| heap.len() >= candidate_count)
                        .map(|node| node.dist.0),
                    &mut survivors,
                    &mut stage1,
                );
                let RabitExLayout::Native {
                    rotated_dim,
                    num_bits,
                    ..
                } = self.storage().rabit_ex_layout()?;
                let ex_rows = ExRows::from_native_batch(&ex, rotated_dim, num_bits)?;
                self.score_ex_survivors(
                    &sign,
                    &survivors,
                    stage1,
                    ExIndex::Identity,
                    ex_rows,
                    heap,
                    |_| {},
                )?;
                Ok(())
            }
        }
    }

    /// Score native gathered rows with the same full-precision
    /// kernel, filters and live heap bounds as the built-in scan.
    #[allow(clippy::too_many_arguments)]
    pub fn score_rabit_survivors(
        &self,
        sign: &RabitSignPartition,
        survivors: &[u32],
        stage1: StagePruneCounts,
        gathered: &GatheredRabitEx,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        on_first_full: impl FnMut(f32),
    ) -> Result<RabitPruneCounters> {
        match gathered {
            GatheredRabitEx::Native { batch, index, .. } => {
                let RabitExLayout::Native {
                    rotated_dim,
                    num_bits,
                    ..
                } = self.storage().rabit_ex_layout()?;
                let ex_rows = ExRows::from_native_batch(batch, rotated_dim, num_bits)?;
                self.score_ex_survivors(
                    sign,
                    survivors,
                    stage1,
                    *index,
                    ex_rows,
                    heap,
                    on_first_full,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score_ex_survivors(
        &self,
        sign: &RabitSignPartition,
        survivors: &[u32],
        stage1: StagePruneCounts,
        index: ExIndex,
        ex_rows: ExRows<'_>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        on_first_full: impl FnMut(f32),
    ) -> Result<RabitPruneCounters> {
        let storage = &sign.entry.storage;
        if sign.scope == RabitScanScope::Query {
            self.metrics.record_comparisons(storage.len());
        }
        let expected_rows = match index {
            ExIndex::Identity => storage.len(),
            ExIndex::Compact => survivors.len(),
        };
        if ex_rows.len() != expected_rows {
            return Err(Error::internal(format!(
                "RaBitQ gather of partition {} returned {} ex rows, expected {expected_rows}",
                sign.partition_id,
                ex_rows.len(),
            )));
        }
        if survivors.windows(2).any(|rows| rows[0] >= rows[1])
            || survivors
                .last()
                .is_some_and(|&row| row as usize >= storage.len())
        {
            return Err(Error::invalid_input(format!(
                "survivors of partition {} must be sorted, unique and below its {} rows",
                sign.partition_id,
                storage.len()
            )));
        }
        let rows = survivors
            .iter()
            .enumerate()
            .map(|(position, &offset)| SurvivorRow {
                ex_index: match index {
                    ExIndex::Identity => offset,
                    ExIndex::Compact => position as u32,
                },
                row_id: storage.row_id(offset),
                binary_ip: sign.stage.binary_ips[offset as usize],
                lower_bound: sign.stage.lower_bounds[offset as usize],
            })
            .collect::<Vec<_>>();
        let residual = RabitIndex::query_context_for_scratch(
            self.index.use_query_residual,
            self.index.use_residual_scratch,
            sign.partition_id,
            None,
            rotated_partition_centroid_slice(
                self.index.rq_search_cache.as_deref(),
                sign.partition_id,
            ),
            self.raw_query.as_deref(),
        )?;
        self.index.scratch_pool.with_scratch(|scratch| {
            let calc = storage.dist_calculator_with_ex_rows(
                self.query.key.clone(),
                sign.dist_q_c,
                residual,
                &mut scratch.query_f32,
                DistanceCalculatorOptions {
                    approx_mode: self.query.approx_mode,
                },
                ex_rows,
            )?;
            if sign.scope == RabitScanScope::Partition {
                return Ok(calc.accumulate_partition_survivor_rows(
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.query.lower_bound,
                    self.query.upper_bound,
                    &rows,
                    stage1,
                    heap,
                    on_first_full,
                ));
            }
            Ok(calc.accumulate_survivor_rows(
                self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                self.query.lower_bound,
                self.query.upper_bound,
                &rows,
                stage1,
                heap,
                on_first_full,
            ))
        })
    }

    /// Convert the final native heap to the standard vector result schema.
    pub fn finish(&self, heap: BinaryHeap<OrderedNode<u64>>) -> Result<RecordBatch> {
        global_heap_to_batch(heap, self.metrics.as_ref())
    }
}
