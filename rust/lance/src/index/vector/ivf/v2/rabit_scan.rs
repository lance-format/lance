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
use lance_core::cache::WeakLanceCache;
use lance_core::{Error, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::Query;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::scan_stats;
use lance_index::vector::bq::storage::{
    ExRows, RabitPruneCounters, SignStage, StagePruneCounts, SurvivorRow,
};
use lance_index::vector::flat::index::FlatIndex;
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::storage::{
    DistanceCalculatorOptions, ExIndex, GatheredRabitEx, IndexFileKey, IvfQuantizationStorage,
    OriginLatencyClass, RabitExLayout, RabitRawQueryContext, VectorStore,
};

use super::{
    IVFIndex, PartitionEntry, PreparedPartitionGuard, PreparedPartitionSearch,
    rotated_partition_centroid_slice,
};

type RabitIndex = IVFIndex<FlatIndex, RabitQuantizer>;

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
}

/// Native preparation and scoring inputs for one query.
///
/// The index and its Arrow buffers are shared; constructing this context does
/// not copy partitions. Strategy calls operate on partitions or row batches,
/// keeping the inner scoring loop concrete and free of virtual calls.
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
pub struct RabitPreparedPartition(PreparedPartitionSearch<FlatIndex, RabitQuantizer>);

/// A sign-only partition with native lower bounds and filter acceptance.
pub struct RabitSignPartition {
    entry: Arc<PartitionEntry<FlatIndex, RabitQuantizer>>,
    partition_id: usize,
    dist_q_c: f32,
    _in_flight: PreparedPartitionGuard,
    /// Binary inner products and lower bounds in partition row order.
    pub stage: SignStage,
    /// Accepted partition offsets; `None` accepts every row.
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

    /// This index's cache namespace; entries do not retain the strategy.
    pub fn cache(&self) -> &WeakLanceCache {
        &self.index.index_cache
    }

    /// Whether native full scoring can use raw-query sign bounds.
    pub fn has_raw_query_bounds(&self) -> bool {
        !self.index.use_query_residual && self.raw_query.is_some()
    }

    /// Prepare a whole partition with the ordinary full-scan implementation.
    pub async fn prepare_partition(
        &self,
        partition_id: usize,
        dist_q_c: f32,
    ) -> Result<RabitPreparedPartition> {
        let mut query = self.query.clone();
        query.dist_q_c = dist_q_c;
        self.index
            .prepare_partition_without_prefilter_wait(
                partition_id,
                &query,
                self.pre_filter.clone(),
                self.metrics.as_ref(),
                self.raw_query.clone(),
            )
            .await
            .map(RabitPreparedPartition)
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
            self.cache(),
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
                calc.full_sign_stage_with_scratch(
                    &mut stage,
                    &mut scratch.u16,
                    &mut scratch.u8,
                    &mut scratch.u32,
                )
                .is_ok()
            });
            if !gated {
                scan_stats::counters().stage1_cpu_ns.add_elapsed(computing);
                return Ok(None);
            }
            let accept = (!pre_filter.is_empty()).then(|| {
                let mask = pre_filter.mask();
                storage
                    .row_ids()
                    .map(|&row_id| mask.selected(row_id))
                    .collect()
            });
            scan_stats::counters().stage1_cpu_ns.add_elapsed(computing);
            Ok(Some(RabitSignPartition {
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

    /// Accumulate a prepared partition into a query-local heap using native scoring.
    pub fn score_partition(
        &self,
        prepared: RabitPreparedPartition,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
    ) -> Result<()> {
        self.index.scratch_pool.with_scratch(|scratch| {
            RabitIndex::accumulate_prepared_partition_search(
                self.index.use_query_residual,
                self.index.use_residual_scratch,
                prepared.0,
                heap,
                scratch,
                self.metrics.as_ref(),
            )
        })
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
        self.metrics.record_comparisons(storage.len());
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
        RabitIndex::global_heap_to_batch(heap, self.metrics.as_ref())
    }
}
