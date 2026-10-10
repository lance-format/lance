// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use crate::vector::bq::ex_dot::blocked_ex_code_bytes;
use crate::vector::bq::layered::{PlaneBatch, PlaneKey, RQLayout};
use crate::vector::bq::layered_stats;
use crate::vector::quantizer::QuantizerStorage;
use arrow::compute::concat_batches;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt8Array, UInt32Array, cast::AsArray,
    types::UInt8Type,
};
use arrow_schema::SchemaRef;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::cache::WeakLanceCache;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_index_core::remapping::{BatchRowIdRemapper, remap_row_ids_preserving_layout_async};
use lance_io::ReadBatchParams;
use lance_io::scheduler::IoStats;
use lance_io::spill::SpillStore;
use lance_linalg::distance::DistanceType;
use prost::Message;
use std::{
    any::Any,
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet},
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::{Arc, LazyLock, Mutex, OnceLock},
    time::Instant,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crossbeam_queue::ArrayQueue;

use crate::frag_reuse::{FragReuseIndex, FragReuseIndexHandle};
use crate::scalar::RowIdRemapper;
use crate::{
    pb,
    vector::{
        ivf::storage::{IVF_METADATA_KEY, IvfModel},
        quantizer::Quantization,
    },
};

use super::graph::OrderedFloat;
use super::graph::OrderedNode;
use super::pairwise::{
    EncodedPartition, PairwisePartition, PairwiseScorer, PairwiseSpillWriter, list_values,
};
use super::quantizer::{Quantizer, QuantizerMetadata};
use super::{ApproxMode, DISTANCE_TYPE_KEY};

/// Coalesce source-index reads independently of the scoring vector batch size.
const PAIRWISE_READ_BATCH_SIZE: usize = 8192;
// Bound each scoring/spill batch in code space; very wide vectors still need
// at least one aligned group of 32 rows for packed quantizers.
const PAIRWISE_VECTOR_BATCH_BYTES: usize = 16 * 1024 * 1024;

/// Stage consecutive `batch_size` row ranges of `source` concurrently on the
/// CPU pool, yielding them in storage order.
fn stage_pairwise_batches(
    scorer: Arc<PairwiseScorer>,
    source: RecordBatch,
    batch_size: usize,
    remapper: Option<Arc<dyn RowIdRemapper>>,
) -> impl Stream<Item = Result<RecordBatch>> {
    let source = Arc::new(source);
    let num_rows = source.num_rows();
    stream::iter((0..num_rows).step_by(batch_size))
        .map(move |start| {
            let rows = start..start.saturating_add(batch_size).min(num_rows);
            let (source, scorer, remapper) = (source.clone(), scorer.clone(), remapper.clone());
            spawn_cpu(move || scorer.stage(&source, rows, remapper.as_deref()))
        })
        .buffered(get_num_compute_intensive_cpus())
}

async fn spawn_prewarm_materialization<R, F>(materialize: F) -> Result<R>
where
    R: Send + 'static,
    F: FnOnce() -> Result<R> + Send + 'static,
{
    // `spawn_cpu` work is non-cancellable. If the caller-owned cache loader is
    // dropped, this pure CPU tail may finish, but its result is dropped with
    // this future and therefore cannot be inserted into the cache. A later
    // cache lookup remains responsible for retrying the load.
    spawn_cpu(materialize).await
}

fn compact_prewarm_batches(batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    let schema = batches
        .first()
        .ok_or_else(|| Error::internal("prewarm partition has no storage batches"))?
        .schema();
    if batches.len() == 1 {
        let batch = batches.into_iter().next().ok_or_else(|| {
            Error::internal("prewarm partition storage batch unexpectedly missing")
        })?;
        if batch.num_rows() == 0 {
            Ok(batch)
        } else {
            Ok(batch.shrink_to_fit()?)
        }
    } else {
        // Concatenation allocates compact output buffers already; do not
        // deep-copy them a second time with `shrink_to_fit`.
        Ok(concat_batches(&schema, batches.iter())?)
    }
}

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;

    // return the distances of all rows
    // k_hint is a hint that can be used for optimization
    fn distance_all(&self, k_hint: usize) -> Vec<f32>;

    // Write the distances of all rows into caller-owned scratch buffers.
    fn distance_all_with_scratch(
        &self,
        k_hint: usize,
        dists: &mut Vec<f32>,
        _u16_scratch: &mut Vec<u16>,
        _u8_scratch: &mut Vec<u8>,
        _u32_scratch: &mut Vec<u32>,
    ) {
        *dists = self.distance_all(k_hint);
    }

    fn prefetch(&self, _id: u32) {}

    /// Whether [`Self::accumulate_topk_with_scratch`] can replace scoring every
    /// row with [`Self::distance_all`] and pushing each score into a top-k heap.
    ///
    /// When this returns true, `accumulate_topk_with_scratch` into an empty heap
    /// must, for any `lower_bound` and `upper_bound`, select the same rows (up
    /// to ties at the k-th distance), with bit-identical distances, as pushing
    /// every `distance_all` score that lies in `[lower_bound, upper_bound)`, or
    /// every score when both bounds are `None`. Since the accumulator treats a
    /// missing bound as `f32::MIN` or `f32::MAX`, every score must lie in
    /// `[f32::MIN, f32::MAX)`.
    fn has_exact_topk_scan(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_topk_with_scratch(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        row_id: impl Fn(u32) -> u64,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        dists: &mut Vec<f32>,
        u16_scratch: &mut Vec<u16>,
        u8_scratch: &mut Vec<u8>,
        u32_scratch: &mut Vec<u32>,
    ) {
        if k == 0 {
            return;
        }

        self.distance_all_with_scratch(k, dists, u16_scratch, u8_scratch, u32_scratch);
        accumulate_distances_into_heap(k, lower_bound, upper_bound, row_id, res, dists);
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_filtered_topk_with_scratch(
        &self,
        k: usize,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        row_ids: impl Iterator<Item = (u32, u64)>,
        accept_row: impl Fn(u64) -> bool,
        res: &mut BinaryHeap<OrderedNode<u64>>,
        _dists: &mut Vec<f32>,
        _u16_scratch: &mut Vec<u16>,
        _u8_scratch: &mut Vec<u8>,
        _u32_scratch: &mut Vec<u32>,
    ) {
        if k == 0 {
            return;
        }

        let lower_bound = lower_bound.unwrap_or(f32::MIN).into();
        let upper_bound = upper_bound.unwrap_or(f32::MAX).into();
        let mut max_dist = res.peek().map(|node| node.dist);

        for (id, row_id) in row_ids {
            if !accept_row(row_id) {
                continue;
            }
            let dist = OrderedFloat(self.distance(id));
            if dist < lower_bound || dist >= upper_bound {
                continue;
            }
            if res.len() < k {
                res.push(OrderedNode::new(row_id, dist));
                if res.len() == k {
                    max_dist = res.peek().map(|node| node.dist);
                }
            } else if max_dist.is_some_and(|max_dist| max_dist > dist) {
                res.pop();
                res.push(OrderedNode::new(row_id, dist));
                max_dist = res.peek().map(|node| node.dist);
            }
        }
    }
}

/// Push rows `0..dists.len()` whose distance lies in `[lower_bound,
/// upper_bound)` into the top-`k` heap `res`.
pub(crate) fn accumulate_distances_into_heap(
    k: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    row_id: impl Fn(u32) -> u64,
    res: &mut BinaryHeap<OrderedNode<u64>>,
    dists: &[f32],
) {
    let lower_bound = lower_bound.unwrap_or(f32::MIN).into();
    let upper_bound = upper_bound.unwrap_or(f32::MAX).into();
    let mut max_dist = res.peek().map(|node| node.dist);

    for (id, dist) in dists.iter().copied().enumerate() {
        let dist = OrderedFloat(dist);
        if dist < lower_bound || dist >= upper_bound {
            continue;
        }
        if res.len() < k {
            res.push(OrderedNode::new(row_id(id as u32), dist));
            if res.len() == k {
                max_dist = res.peek().map(|node| node.dist);
            }
        } else if max_dist.is_some_and(|max_dist| max_dist > dist) {
            res.pop();
            res.push(OrderedNode::new(row_id(id as u32), dist));
            max_dist = res.peek().map(|node| node.dist);
        }
    }
}

pub const STORAGE_METADATA_KEY: &str = "storage_metadata";

#[derive(Debug)]
pub struct QueryScratch {
    pub distances: Vec<f32>,
    pub query_f32: Vec<f32>,
    pub u16: Vec<u16>,
    pub u8: Vec<u8>,
    pub u32: Vec<u32>,
}

impl QueryScratch {
    pub const fn new() -> Self {
        Self {
            distances: Vec::new(),
            query_f32: Vec::new(),
            u16: Vec::new(),
            u8: Vec::new(),
            u32: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: QueryScratchCapacity) -> Self {
        Self {
            distances: vec![0.0; capacity.distances],
            query_f32: vec![0.0; capacity.query_f32],
            u16: vec![0; capacity.u16],
            u8: vec![0; capacity.u8],
            u32: vec![0; capacity.u32],
        }
    }
}

impl Default for QueryScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSizeOf for QueryScratch {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.distances.capacity() * size_of::<f32>()
            + self.query_f32.capacity() * size_of::<f32>()
            + self.u16.capacity() * size_of::<u16>()
            + self.u8.capacity() * size_of::<u8>()
            + self.u32.capacity() * size_of::<u32>()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QueryScratchCapacity {
    pub distances: usize,
    pub query_f32: usize,
    pub u16: usize,
    pub u8: usize,
    pub u32: usize,
}

impl QueryScratchCapacity {
    pub const fn new(distances: usize, query_f32: usize, u16: usize, u8: usize) -> Self {
        Self::new_with_u32(distances, query_f32, u16, u8, 0)
    }

    pub const fn new_with_u32(
        distances: usize,
        query_f32: usize,
        u16: usize,
        u8: usize,
        u32: usize,
    ) -> Self {
        Self {
            distances,
            query_f32,
            u16,
            u8,
            u32,
        }
    }

    fn deep_size_bytes(&self) -> usize {
        self.distances * size_of::<f32>()
            + self.query_f32 * size_of::<f32>()
            + self.u16 * size_of::<u16>()
            + self.u8 * size_of::<u8>()
            + self.u32 * size_of::<u32>()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistanceCalculatorOptions {
    pub approx_mode: ApproxMode,
    pub rq_precision: super::bq::layered::RQPrecision,
}

#[derive(Debug)]
pub struct RabitRawQueryContext {
    pub code_dim: usize,
    pub ex_bits: u8,
    pub rotated_query: Vec<f32>,
    pub dist_table: Vec<f32>,
    /// The rotated query zero-padded to a 64-dim multiple for the ex-dot
    /// kernels; empty when `code_dim` is already aligned (the kernels then
    /// read `rotated_query` directly).
    pub ex_query: Vec<f32>,
    pub sum_q: f32,
}

#[derive(Clone, Copy)]
pub enum QueryResidual<'a> {
    Centroid(&'a dyn arrow_array::Array),
    RabitRawQuery {
        rotated_centroid: Option<&'a [f32]>,
        query: Option<&'a RabitRawQueryContext>,
    },
}

#[derive(Debug)]
pub struct QueryScratchPool {
    scratches: ArrayQueue<QueryScratch>,
    scratch_capacity: QueryScratchCapacity,
}

impl QueryScratchPool {
    pub fn new(size: usize) -> Self {
        Self::with_capacity(size, QueryScratchCapacity::default())
    }

    pub fn with_capacity(size: usize, capacity: QueryScratchCapacity) -> Self {
        let size = size.max(1);
        let scratches = ArrayQueue::new(size);
        for _ in 0..size {
            scratches
                .push(QueryScratch::with_capacity(capacity))
                .expect("query scratch pool should have spare capacity during initialization");
        }
        Self {
            scratches,
            scratch_capacity: capacity,
        }
    }

    pub fn scratch(&self) -> QueryScratchGuard<'_> {
        let (scratch, pooled) = if let Some(scratch) = self.scratches.pop() {
            (scratch, true)
        } else {
            (QueryScratch::with_capacity(self.scratch_capacity), false)
        };
        QueryScratchGuard {
            pool: self,
            scratch: Some(scratch),
            pooled,
        }
    }

    pub fn with_scratch<T>(&self, f: impl FnOnce(&mut QueryScratch) -> T) -> T {
        let mut scratch = self.scratch();
        f(&mut scratch)
    }
}

pub struct QueryScratchGuard<'a> {
    pool: &'a QueryScratchPool,
    scratch: Option<QueryScratch>,
    pooled: bool,
}

impl Deref for QueryScratchGuard<'_> {
    type Target = QueryScratch;

    fn deref(&self) -> &Self::Target {
        self.scratch
            .as_ref()
            .expect("query scratch guard should hold scratch")
    }
}

impl DerefMut for QueryScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch
            .as_mut()
            .expect("query scratch guard should hold scratch")
    }
}

impl Drop for QueryScratchGuard<'_> {
    fn drop(&mut self) {
        if !self.pooled {
            return;
        }
        if let Some(scratch) = self.scratch.take() {
            match self.pool.scratches.push(scratch) {
                Ok(()) => {}
                Err(_) => unreachable!("query scratch pool should not exceed its capacity"),
            }
        }
    }
}

impl DeepSizeOf for QueryScratchPool {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let mut total = self.scratches.capacity() * size_of::<QueryScratch>();
        let mut scratches = Vec::new();
        while let Some(scratch) = self.scratches.pop() {
            total += scratch.deep_size_of_children(context);
            scratches.push(scratch);
        }
        let checked_out = self.scratches.capacity().saturating_sub(scratches.len());
        total += checked_out * self.scratch_capacity.deep_size_bytes();
        for scratch in scratches {
            let _ = self.scratches.push(scratch);
        }
        total
    }
}

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory or on-disk, raw vector or quantized vectors.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait VectorStore: Send + Sync + Sized + Clone {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send>;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return [DistanceType].
    fn distance_type(&self) -> DistanceType;

    /// Get the lance ROW ID from one vector.
    fn row_id(&self, id: u32) -> u64;

    fn row_ids(&self) -> impl Iterator<Item = &u64>;

    /// Append Raw [RecordBatch] into the Storage.
    /// The storage implement will perform quantization if necessary.
    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self>;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calculator can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef, dist_q_c: f32) -> Self::DistanceCalculator<'_>;

    /// Create a [DistCalculator], reusing caller-owned scratch for query-time
    /// precomputed state when the storage supports it.
    fn dist_calculator_with_scratch<'a>(
        &'a self,
        query: ArrayRef,
        dist_q_c: f32,
        _residual: Option<QueryResidual<'a>>,
        _f32_scratch: &'a mut Vec<f32>,
        _options: DistanceCalculatorOptions,
    ) -> Self::DistanceCalculator<'a> {
        self.dist_calculator(query, dist_q_c)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;

    fn dist_between(&self, u: u32, v: u32) -> f32 {
        let dist_cal_u = self.dist_calculator_from_id(u);
        dist_cal_u.distance(v)
    }

    fn prefers_candidate(&self, candidate: &OrderedNode, selected: &[OrderedNode]) -> bool {
        let dist_cal_candidate = self.dist_calculator_from_id(candidate.id);
        selected
            .iter()
            .all(|other| candidate.dist < OrderedFloat(dist_cal_candidate.distance(other.id)))
    }
}

pub struct StorageBuilder<Q: Quantization> {
    vector_column: String,
    distance_type: DistanceType,
    quantizer: Q,

    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(
        vector_column: String,
        distance_type: DistanceType,
        quantizer: Q,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::new_with_remapper(vector_column, distance_type, quantizer, frag_reuse_index)
    }

    #[doc(hidden)]
    pub fn new_with_remapper(
        vector_column: String,
        distance_type: DistanceType,
        quantizer: Q,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        Ok(Self {
            vector_column,
            distance_type,
            quantizer,
            frag_reuse_index,
        })
    }

    pub fn build(&self, batches: Vec<RecordBatch>) -> Result<Q::Storage> {
        let mut batch = concat_batches(batches[0].schema_ref(), batches.iter())?;

        if batch.column_by_name(self.quantizer.column()).is_none() {
            let vectors = batch
                .column_by_name(&self.vector_column)
                .ok_or(Error::index(format!(
                    "Vector column {} not found in batch",
                    self.vector_column
                )))?;
            let codes = self.quantizer.quantize(vectors)?;
            batch = batch.drop_column(&self.vector_column)?.try_with_column(
                arrow_schema::Field::new(self.quantizer.column(), codes.data_type().clone(), true),
                codes,
            )?;
        }

        debug_assert!(batch.column_by_name(ROW_ID).is_some());
        debug_assert!(batch.column_by_name(self.quantizer.column()).is_some());

        Q::Storage::try_from_batch_with_remapper(
            batch,
            &self.quantizer.metadata(None),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }
}

/// Avoid loading whole cold planes for isolated candidate reads. Repeated reads
/// covering half a partition amortize promotion, with a cooldown after admission pressure.
const PLANE_PROMOTION_READS: usize = 3;
const PLANE_PROMOTION_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Debug, Default)]
struct PlaneAccess {
    reads: usize,
    rows: usize,
    last_attempt: Option<std::time::Instant>,
}
impl DeepSizeOf for PlaneAccess {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        0
    }
}

/// Runtime promotion history shared by reconstructions of the same cached index.
/// This contains no readers or object-store handles and is not persisted to disk.
#[derive(Debug, Clone, Default)]
pub struct PlaneAccessTracker {
    /// Cascade candidate reads, see [`PLANE_PROMOTION_READS`].
    accesses: Arc<Mutex<HashMap<(usize, u8), PlaneAccess>>>,
    /// Background promotions of the lazy full-precision scan.
    lazy: Arc<LazyPromotionState>,
}

impl DeepSizeOf for PlaneAccessTracker {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.accesses.deep_size_of_children(context) + self.lazy.deep_size_of_children(context)
    }
}

#[derive(Debug, Default)]
struct LazyPromotionHistory {
    /// Sparse gathers of each non-resident `(partition, plane)` since its last promotion.
    sparse_reads: HashMap<(usize, u8), u32>,
    in_flight: HashSet<(usize, u8)>,
}

#[derive(Debug, Default)]
struct LazyPromotionState {
    history: Mutex<LazyPromotionHistory>,
    /// Bounds concurrent promotions; sized by the first promoting query's config.
    permits: OnceLock<Arc<Semaphore>>,
}

impl DeepSizeOf for LazyPromotionState {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        history.sparse_reads.capacity() * size_of::<((usize, u8), u32)>()
            + history.in_flight.capacity() * size_of::<(usize, u8)>()
    }
}

/// A started background promotion of one ex plane. Dropping it, when the
/// promotion finishes or is abandoned, frees its in-flight slot and permit.
#[derive(Debug)]
pub struct LazyPromotionTicket {
    pub partition: usize,
    pub plane: u8,
    state: Arc<LazyPromotionState>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for LazyPromotionTicket {
    fn drop(&mut self) {
        self.state
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .in_flight
            .remove(&(self.partition, self.plane));
        layered_stats::promotion_finished();
    }
}

/// Enables the lazy layered full-precision scan (`0` or `1`).
pub const LAZY_FULL_ENV: &str = "LANCE_RQ_LAZY_FULL";
/// Staleness window W: a gather may be issued while up to W earlier probes are unscored.
pub const LAZY_WINDOW_ENV: &str = "LANCE_RQ_LAZY_WINDOW";
/// Ex-plane gather policy: `cost`, `sparse` or `whole`.
pub const LAZY_DENSE_ENV: &str = "LANCE_RQ_LAZY_DENSE";
/// Most coalesced row runs a sparse gather may issue per plane.
pub const LAZY_MAX_RUNS_ENV: &str = "LANCE_RQ_LAZY_MAX_RUNS";
/// Aligned sparse bytes, as a fraction of the plane, from which a whole read is cheaper.
pub const LAZY_DENSE_BYTES_FRACTION_ENV: &str = "LANCE_RQ_LAZY_DENSE_BYTES_FRACTION";
/// Background promotion policy: `off` or `bg:N` (promote after N sparse
/// gathers). Backends that gate plane admission never promote.
pub const LAZY_PROMOTE_ENV: &str = "LANCE_RQ_LAZY_PROMOTE";
/// Most background promotions running at once.
pub const LAZY_PROMOTE_INFLIGHT_ENV: &str = "LANCE_RQ_LAZY_PROMOTE_INFLIGHT";
/// Largest survivor count a scoring batch scores on the query task instead of the CPU pool.
pub const LAZY_INLINE_ROWS_ENV: &str = "LANCE_RQ_LAZY_INLINE_ROWS";
/// Whether a gather whose window gate opens before the heap holds `k` rows
/// may be issued then, selecting every accepted row as the eager load does
/// (`0` or `1`, default `1`). It is issued early when waiting would not save
/// reads, because the gathers issued once the heap fills are expected to be
/// dense anyway (large `k`), or when the heap is still not full after the
/// probes that hold `k` rows were scored (filters), where waiting would issue
/// the gathers one probe at a time. `0` is an ablation knob: such gathers
/// wait for the heap to fill or for every earlier probe to be scored. With
/// [`LAZY_DENSE_TO_EAGER_ENV`] on, the large-`k` probes of the queries it
/// routes are scored eagerly instead and never gathered.
pub const LAZY_EAGER_BEFORE_FULL_ENV: &str = "LANCE_RQ_LAZY_EAGER_BEFORE_FULL";
/// Whether a probe that `k` and the partition sizes predict to be gathered
/// whole is loaded and scored by the eager scan instead (`0` or `1`, default
/// `1`): a probe scored before the probes ahead of it can hold `k` rows,
/// whose gather selects every row, and every probe when the gathers issued
/// once the heap fills are expected to read whole planes (large `k`). The
/// eager load reads the partition's planes together (all at once on backends
/// that do not gate plane admission), with as many probes in flight as the
/// eager scan prepares, instead of the sign plane at staging and the ex
/// planes a round trip later within the gather window. Only queries without
/// a prefilter (deletions included) or upper distance bound are routed: a
/// gather with an infinite threshold selects only the accepted rows below
/// the bound, which the probe's sign plane tells. The `sparse` gather policy
/// ([`LAZY_DENSE_ENV`]) never reads a whole plane and routes no probe. `0` is
/// an ablation knob: such probes take the lazy pipeline.
pub const LAZY_DENSE_TO_EAGER_ENV: &str = "LANCE_RQ_LAZY_DENSE_TO_EAGER";
/// Most coalesced row runs a sparse gather reads from the origin file when the
/// persistent tier does not hold the plane; with more, it loads (and admits)
/// the whole plane instead. Unlimited by default, and ignored by backends that
/// gate plane admission, which would not admit the plane.
pub const LAZY_ORIGIN_MAX_RUNS_ENV: &str = "LANCE_RQ_LAZY_ORIGIN_MAX_RUNS";
/// Benchmark knob (`0` or `1`, default `0`): `1` loads a layered partition's
/// sign plane before its ex planes on backends that do not gate plane
/// admission too, as gated backends always do, instead of loading all three
/// planes at once. Read once per process.
pub const SEQUENTIAL_PLANE_LOADS_ENV: &str = "LANCE_RQ_SEQUENTIAL_PLANE_LOADS";

const DEFAULT_LAZY_WINDOW: usize = 16;
const DEFAULT_LAZY_MAX_RUNS: usize = 16;
const DEFAULT_LAZY_DENSE_BYTES_FRACTION: f64 = 0.5;
const DEFAULT_LAZY_PROMOTE_READS: u32 = 1;
const DEFAULT_LAZY_PROMOTE_INFLIGHT: usize = 4;
/// About the rows one ~100µs CPU dispatch would score.
const DEFAULT_LAZY_INLINE_ROWS: usize = 512;

/// [`SEQUENTIAL_PLANE_LOADS_ENV`], read once per process.
static SEQUENTIAL_PLANE_LOADS: LazyLock<std::result::Result<bool, String>> = LazyLock::new(|| {
    sequential_plane_loads_from(std::env::var(SEQUENTIAL_PLANE_LOADS_ENV).ok().as_deref())
        .map_err(|err| err.to_string())
});

/// Whether layered partition loads read the sign plane before the ex planes on
/// every backend ([`SEQUENTIAL_PLANE_LOADS_ENV`]). The variable is read once
/// per process; an invalid value fails here and every layered partition load.
/// Call it at startup to fail before serving.
pub fn sequential_plane_loads() -> Result<bool> {
    SEQUENTIAL_PLANE_LOADS.clone().map_err(Error::invalid_input)
}

fn sequential_plane_loads_from(value: Option<&str>) -> Result<bool> {
    value.map_or(Ok(false), |value| {
        parse_flag(SEQUENTIAL_PLANE_LOADS_ENV, value)
    })
}

/// Parse a `0`/`1` (or `false`/`true`) environment flag.
fn parse_flag(name: &str, value: &str) -> Result<bool> {
    match value.trim() {
        "1" | "true" => Ok(true),
        "0" | "false" => Ok(false),
        _ => Err(Error::invalid_input(format!(
            "{name}={value:?} is invalid, expected 0 or 1"
        ))),
    }
}

/// Page size the gather cost model aligns row runs to, matching the
/// persistent cache's aligned reads.
const LAZY_GATHER_PAGE_BYTES: usize = 4096;
/// Rows closer than this share one read, as in the plane cache codec.
const LAZY_GATHER_COALESCE_GAP_BYTES: usize = 4096;
/// Estimated cache envelope and plane header ahead of the first row. It only
/// shifts page boundaries in the cost estimate.
const LAZY_GATHER_BODY_OFFSET_BYTES: usize = 64;
/// Bytes a cached ex-plane row stores besides its codes: add and scale factors.
const EX_PLANE_FACTOR_BYTES: usize = 2 * size_of::<f32>();

/// How the lazy scan chooses between reading selected ex-plane rows and the whole plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseGatherMode {
    /// Read the whole plane when the row runs are too many or cover too much
    /// of it (see [`plan_plane_gather`]).
    Cost,
    /// Always read selected rows; no probe is routed to the eager scan
    /// ([`LAZY_DENSE_TO_EAGER_ENV`]).
    Sparse,
    /// Always read the whole plane.
    Whole,
}

/// Whether sparse gathers promote whole planes into RAM in the background.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyPromotion {
    /// Only whole-plane gathers admit planes.
    Off,
    /// Promote a plane after this many sparse gathers of it while not
    /// resident, on backends that do not gate plane admission.
    Background { reads: u32 },
}

/// Settings of the lazy layered full-precision scan, which bounds every row
/// from the sign plane and reads the ex planes of the survivors only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayeredLazyConfig {
    pub enabled: bool,
    /// Probes a gather may run ahead of scoring; see [`LAZY_WINDOW_ENV`].
    pub window: usize,
    pub dense: DenseGatherMode,
    pub max_runs: usize,
    pub dense_bytes_fraction: f64,
    pub promote: LazyPromotion,
    pub promote_inflight: usize,
    pub inline_rows: usize,
    /// See [`LAZY_EAGER_BEFORE_FULL_ENV`].
    pub eager_before_full: bool,
    /// See [`LAZY_ORIGIN_MAX_RUNS_ENV`]; `usize::MAX` never falls back.
    pub origin_max_runs: usize,
    /// See [`LAZY_DENSE_TO_EAGER_ENV`].
    pub dense_to_eager: bool,
}

impl Default for LayeredLazyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window: DEFAULT_LAZY_WINDOW,
            dense: DenseGatherMode::Cost,
            max_runs: DEFAULT_LAZY_MAX_RUNS,
            dense_bytes_fraction: DEFAULT_LAZY_DENSE_BYTES_FRACTION,
            promote: LazyPromotion::Background {
                reads: DEFAULT_LAZY_PROMOTE_READS,
            },
            promote_inflight: DEFAULT_LAZY_PROMOTE_INFLIGHT,
            inline_rows: DEFAULT_LAZY_INLINE_ROWS,
            eager_before_full: true,
            origin_max_runs: usize::MAX,
            dense_to_eager: true,
        }
    }
}

impl LayeredLazyConfig {
    /// Read the `LANCE_RQ_LAZY_*` variables; unset variables keep their defaults.
    /// The other knobs are read only when [`LAZY_FULL_ENV`] enables the scan,
    /// so with it off they are ignored exactly as before the scan existed.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let invalid = |name: &str, value: &str, expected: &str| {
            Error::invalid_input(format!("{name}={value:?} is invalid, expected {expected}"))
        };
        let count = |name: &str, default: usize, min: usize| -> Result<usize> {
            let Some(value) = lookup(name) else {
                return Ok(default);
            };
            value
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|parsed| *parsed >= min)
                .ok_or_else(|| invalid(name, &value, &format!("an integer >= {min}")))
        };
        let mut config = Self::default();
        if let Some(value) = lookup(LAZY_FULL_ENV) {
            config.enabled = parse_flag(LAZY_FULL_ENV, &value)?;
        }
        if !config.enabled {
            return Ok(config);
        }
        config.window = count(LAZY_WINDOW_ENV, config.window, 0)?;
        if let Some(value) = lookup(LAZY_DENSE_ENV) {
            config.dense = match value.trim() {
                "cost" => DenseGatherMode::Cost,
                "sparse" => DenseGatherMode::Sparse,
                "whole" => DenseGatherMode::Whole,
                _ => return Err(invalid(LAZY_DENSE_ENV, &value, "cost, sparse or whole")),
            };
        }
        config.max_runs = count(LAZY_MAX_RUNS_ENV, config.max_runs, 0)?;
        if let Some(value) = lookup(LAZY_DENSE_BYTES_FRACTION_ENV) {
            config.dense_bytes_fraction = value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|fraction| fraction.is_finite() && *fraction >= 0.0)
                .ok_or_else(|| {
                    invalid(
                        LAZY_DENSE_BYTES_FRACTION_ENV,
                        &value,
                        "a finite number >= 0",
                    )
                })?;
        }
        if let Some(value) = lookup(LAZY_PROMOTE_ENV) {
            config.promote = match value.trim() {
                "off" => LazyPromotion::Off,
                other => other
                    .strip_prefix("bg:")
                    .and_then(|reads| reads.parse::<u32>().ok())
                    .filter(|reads| *reads > 0)
                    .map(|reads| LazyPromotion::Background { reads })
                    .ok_or_else(|| invalid(LAZY_PROMOTE_ENV, &value, "off or bg:N with N >= 1"))?,
            };
        }
        config.promote_inflight = count(LAZY_PROMOTE_INFLIGHT_ENV, config.promote_inflight, 1)?;
        config.inline_rows = count(LAZY_INLINE_ROWS_ENV, config.inline_rows, 0)?;
        if let Some(value) = lookup(LAZY_EAGER_BEFORE_FULL_ENV) {
            config.eager_before_full = parse_flag(LAZY_EAGER_BEFORE_FULL_ENV, &value)?;
        }
        config.origin_max_runs = count(LAZY_ORIGIN_MAX_RUNS_ENV, config.origin_max_runs, 0)?;
        if let Some(value) = lookup(LAZY_DENSE_TO_EAGER_ENV) {
            config.dense_to_eager = parse_flag(LAZY_DENSE_TO_EAGER_ENV, &value)?;
        }
        Ok(config)
    }
}

/// Where the lazy scan reads one ex plane of a partition from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneSource {
    /// The RAM-resident entry, read like a native memory hit. Falls back to
    /// [`Self::Whole`] if the entry was evicted since it was planned.
    Resident,
    /// The whole plane, admitted like the eager full-precision load.
    Whole,
    /// Selected rows of the persistent entry (or the origin file), without
    /// RAM admission.
    Sparse,
}

/// Sources for the high (index 0) and low (index 1) ex planes of one gather.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherPlan {
    pub planes: [PlaneSource; 2],
    /// Per plane, whether a [`PlaneSource::Sparse`] read that the persistent
    /// tier cannot serve loads the whole plane (see
    /// [`origin_reads_whole_plane`]) instead of reading the rows from the
    /// origin file. Backends that gate plane admission ignore it: they would
    /// read the whole plane without admitting it, so every query would pay
    /// for it again.
    pub origin_whole: [bool; 2],
}

/// Row geometry of a layered index's ex planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayeredExLayout {
    pub rotated_dim: usize,
    pub num_bits: u8,
    /// Persistent-cache bytes per row of the high and low planes (codes plus factors).
    pub row_bytes: [usize; 2],
}

/// Choose a sparse or whole read of one non-resident ex plane for sorted,
/// unique `rows` of a partition with `partition_rows` rows.
///
/// Rows are grouped into runs with the plane cache codec's gap rule and each
/// run is rounded out to whole pages. The plane is read whole when that takes
/// more than `max_runs` runs, or when the pages cover at least
/// `dense_bytes_fraction` of the plane.
pub fn plan_plane_gather(
    rows: &[u32],
    partition_rows: usize,
    row_bytes: usize,
    config: &LayeredLazyConfig,
) -> PlaneSource {
    match config.dense {
        DenseGatherMode::Sparse => return PlaneSource::Sparse,
        DenseGatherMode::Whole => return PlaneSource::Whole,
        DenseGatherMode::Cost => {}
    }
    let (runs, pages) = sparse_read_extent(rows, row_bytes);
    let plane_bytes = LAZY_GATHER_BODY_OFFSET_BYTES + partition_rows * row_bytes;
    if runs > config.max_runs
        || (pages * LAZY_GATHER_PAGE_BYTES) as f64
            >= config.dense_bytes_fraction * plane_bytes as f64
    {
        PlaneSource::Whole
    } else {
        PlaneSource::Sparse
    }
}

/// Whether a sparse read of sorted, unique `rows` that the persistent tier
/// cannot serve should load the whole plane instead of reading the rows from
/// the origin file: when the rows take more than
/// [`LayeredLazyConfig::origin_max_runs`] runs (grouped as in
/// [`plan_plane_gather`]).
pub fn origin_reads_whole_plane(
    rows: &[u32],
    row_bytes: usize,
    config: &LayeredLazyConfig,
) -> bool {
    config.origin_max_runs != usize::MAX
        && sparse_read_extent(rows, row_bytes).0 > config.origin_max_runs
}

/// Row runs and pages of a sparse read of sorted, unique `rows`: rows closer
/// than the plane cache codec's gap share a run, and every run is rounded out
/// to whole pages.
fn sparse_read_extent(rows: &[u32], row_bytes: usize) -> (usize, usize) {
    let gap_rows = LAZY_GATHER_COALESCE_GAP_BYTES / row_bytes.max(1);
    let page =
        |row: usize| (LAZY_GATHER_BODY_OFFSET_BYTES + row * row_bytes) / LAZY_GATHER_PAGE_BYTES;
    let page_end = |row: usize| {
        (LAZY_GATHER_BODY_OFFSET_BYTES + row * row_bytes).div_ceil(LAZY_GATHER_PAGE_BYTES)
    };
    let mut runs = 0usize;
    let mut pages = 0usize;
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && (rows[end] - rows[end - 1] - 1) as usize <= gap_rows {
            end += 1;
        }
        pages += page_end(rows[end - 1] as usize + 1) - page(rows[start] as usize);
        runs += 1;
        start = end;
    }
    (runs, pages)
}

/// How stage-2 survivors index a [`GatheredEx`]'s batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExIndex {
    /// Both planes are whole: a survivor's ex index is its partition offset.
    Identity,
    /// The batches hold the gathered rows only, in gather order.
    Compact,
}

/// The ex-plane rows of one partition's survivors.
#[derive(Debug)]
pub struct GatheredEx {
    /// High-plane batch (codes and high factors).
    pub high: RecordBatch,
    /// Low-plane batch (codes and the full ex factors).
    pub low: RecordBatch,
    pub index: ExIndex,
    /// The source that served each plane, `None` when no rows were needed.
    pub sources: [Option<PlaneSource>; 2],
    /// Planes whose sparse rows came from the origin file.
    pub origin_row_reads: usize,
    /// Background promotions the caller must run (and then drop).
    pub promotions: Vec<LazyPromotionTicket>,
}

/// One ex plane fetched for a gather.
struct FetchedPlane {
    batch: RecordBatch,
    source: PlaneSource,
    from_origin: bool,
    promotion: Option<LazyPromotionTicket>,
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Q::Metadata,

    ivf: IvfModel,
    /// Legacy synchronous remapper (index_version 0). Mutually exclusive with
    /// `batch_remapper`; both `None` means no translation is needed.
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    /// Asynchronous batch remapper (tagged histories).
    batch_remapper: Option<Arc<dyn BatchRowIdRemapper>>,
    plane_access: PlaneAccessTracker,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.plane_access.deep_size_of_children(context)
    }
}

impl<Q: Quantization> IvfQuantizationStorage<Q> {
    /// Open a Loader.
    ///
    ///
    pub async fn try_new(
        reader: FileReader,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::try_new_with_remapper(reader, frag_reuse_index).await
    }

    #[doc(hidden)]
    pub async fn try_new_with_remapper(
        reader: FileReader,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let schema = reader.schema();

        let distance_type = DistanceType::try_from(
            schema
                .metadata
                .get(DISTANCE_TYPE_KEY)
                .ok_or(Error::index(format!("{} not found", DISTANCE_TYPE_KEY)))?
                .as_str(),
        )?;

        let ivf_pos = schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::index(format!("{} not found", IVF_METADATA_KEY)))?
            .parse()
            .map_err(|e| Error::index(format!("Failed to decode IVF metadata: {}", e)))?;
        // Parsed before the reads below because it holds the position of the
        // quantizer buffer, which is what lets that read start alongside the
        // IVF one.
        let mut metadata_strs: Vec<String> = serde_json::from_str(
            schema
                .metadata
                .get(STORAGE_METADATA_KEY)
                .ok_or(Error::index(format!("{} not found", STORAGE_METADATA_KEY)))?
                .as_str(),
        )?;
        debug_assert_eq!(metadata_strs.len(), 1);
        // for now the metadata is the same for all partitions, so we just store one
        let metadata_str = metadata_strs
            .pop()
            .ok_or(Error::index("metadata is empty".to_string()))?;
        let mut metadata: Q::Metadata = serde_json::from_str(&metadata_str)?;
        let quantizer_buffer_pos = metadata.buffer_index();

        // Both positions come from the schema metadata, so the reads do not
        // depend on each other: issue them together instead of waiting for the
        // IVF protobuf before asking for the quantizer buffer.
        let (ivf_bytes, quantizer_bytes) =
            futures::try_join!(reader.read_global_buffer(ivf_pos), async {
                match quantizer_buffer_pos {
                    Some(pos) => reader.read_global_buffer(pos).await.map(Some),
                    None => Ok(None),
                }
            })?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_bytes)?)?;

        // we store large metadata (e.g. PQ codebook) in global buffer,
        // and the schema metadata just contains a pointer to the buffer
        if let Some(bytes) = quantizer_bytes {
            metadata.parse_buffer(bytes)?;
        }

        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
            batch_remapper: None,
            plane_access: Default::default(),
        })
    }

    /// Construct from pre-parsed metadata, skipping global buffer reads.
    /// Used when reconstructing from a disk cache.
    pub fn from_cached(
        reader: FileReader,
        ivf: IvfModel,
        metadata: Q::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::from_cached_with_remapper(reader, ivf, metadata, distance_type, frag_reuse_index)
    }

    #[doc(hidden)]
    pub fn from_cached_with_remapper(
        reader: FileReader,
        ivf: IvfModel,
        metadata: Q::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Self {
        Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
            batch_remapper: None,
            plane_access: Default::default(),
        }
    }

    /// Set the batch row-ID remapper used when decoding each partition.
    ///
    /// Only tagged fragment-reuse histories use this; the legacy remapper is
    /// supplied through the constructors instead.
    pub fn with_row_id_remapping(mut self, remapping: Arc<dyn BatchRowIdRemapper>) -> Self {
        self.frag_reuse_index = None;
        self.batch_remapper = Some(remapping);
        debug_assert!(self.frag_reuse_index.is_none() || self.batch_remapper.is_none());
        self
    }

    /// Promotion history to retain in the index's cached runtime state.
    pub fn plane_access_tracker(&self) -> &PlaneAccessTracker {
        &self.plane_access
    }

    /// Reuse promotion history when binding new readers to a cached index.
    pub fn with_plane_access_tracker(mut self, tracker: PlaneAccessTracker) -> Self {
        self.plane_access = tracker;
        self
    }

    pub fn reader(&self) -> &FileReader {
        &self.reader
    }

    pub fn ivf(&self) -> &IvfModel {
        &self.ivf
    }

    pub fn num_rows(&self) -> u64 {
        self.reader.num_rows()
    }

    pub fn partition_size(&self, part_id: usize) -> usize {
        self.ivf.partition_size(part_id)
    }

    pub fn quantizer(&self) -> Result<Quantizer> {
        let metadata = self.metadata();
        Q::from_metadata(metadata, self.distance_type)
    }

    pub fn metadata(&self) -> &Q::Metadata {
        &self.metadata
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.reader.schema().as_ref().into())
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    /// Load a partition's quantization storage, optionally measuring the exact
    /// I/O it performs into `io_stats`.
    ///
    /// When `io_stats` is `Some`, the partition is read through a reader whose
    /// scheduler also records into the sink (a cheap clone that shares all
    /// cached metadata, so no file is re-opened).  When `None`, the normal
    /// uninstrumented reader is used.
    pub async fn load_partition(
        &self,
        part_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batch = if range.is_empty() {
            let schema = self.reader.schema();
            let arrow_schema = arrow_schema::Schema::from(schema.as_ref());
            RecordBatch::new_empty(Arc::new(arrow_schema))
        } else {
            let reader = match &io_stats {
                Some(io_stats) => Cow::Owned(self.reader.with_io_stats(io_stats.recorder())),
                None => Cow::Borrowed(&self.reader),
            };
            let batches = reader
                .read_stream(
                    ReadBatchParams::Range(range),
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            let schema = Arc::new(self.reader.schema().as_ref().into());
            concat_batches(&schema, batches.iter())?
        };
        if let Some(remapping) = &self.batch_remapper {
            // Tagged asynchronous path.
            lance_index_core::remapping::check_batch_remapping_entry()?;
            let row_id_idx = batch.schema().index_of(ROW_ID)?;
            let (batch, remapper) =
                remap_row_ids_preserving_layout_async(remapping.as_ref(), batch, row_id_idx)
                    .await?;
            return Q::Storage::try_from_batch_with_remapper(
                batch,
                self.metadata(),
                self.distance_type,
                Some(remapper),
            );
        }
        // Legacy synchronous remapping path.
        Q::Storage::try_from_batch_with_remapper(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }

    /// Warm every plane entry. Backend admission enforces its byte budget.
    ///
    /// A backend that gates lower planes on their sign plane is warmed plane
    /// by plane: all sign planes first, then the high and low planes of
    /// partitions whose sign plane stayed resident. Other backends admit plane
    /// entries like any entry, so they are warmed partition by partition,
    /// which loads (and persists) every plane whatever fits in RAM, as the
    /// native partition prewarm does.
    pub async fn prewarm_planes(&self, cache: &WeakLanceCache) -> Result<()> {
        if !cache.plane_admission_gated() {
            for part_id in 0..self.num_partitions() {
                for plane in 0..=2 {
                    cache
                        .get_or_insert_with_key(
                            PlaneKey {
                                partition: part_id,
                                plane,
                            },
                            || async {
                                Ok(PlaneBatch(
                                    self.read_plane(part_id, plane, None, None).await?,
                                ))
                            },
                        )
                        .await?;
                }
            }
            return Ok(());
        }
        for plane in 0..=2 {
            for part_id in 0..self.num_partitions() {
                if plane > 0
                    && cache
                        .get_resident_with_key(&PlaneKey {
                            partition: part_id,
                            plane: 0,
                        })
                        .await
                        .is_none()
                {
                    continue;
                }
                cache
                    .get_or_insert_with_key(
                        PlaneKey {
                            partition: part_id,
                            plane,
                        },
                        || async {
                            Ok(PlaneBatch(
                                self.read_plane(part_id, plane, None, None).await?,
                            ))
                        },
                    )
                    .await?;
            }
        }
        Ok(())
    }

    /// Whether this loader owns an opt-in layered RaBitQ index.
    pub fn is_layered_rq(&self) -> bool {
        matches!(self.quantizer(), Ok(Quantizer::Rabit(rq)) if rq.metadata_ref().layered)
    }

    /// Read independent plane entries through the existing persistent cache.
    pub async fn load_partition_at_precision(
        &self,
        part_id: usize,
        precision: super::bq::layered::RQPrecision,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        use super::bq::layered::RQPrecision;
        if !self.is_layered_rq() {
            if precision != RQPrecision::Full {
                return Err(Error::invalid_input(
                    "rq_precision requires a layered IVF_RQ index",
                ));
            }
            return self.load_partition(part_id, io_stats).await;
        }
        let last = match precision {
            RQPrecision::Sign => 0,
            RQPrecision::High => 1,
            RQPrecision::Full => 2,
        };
        let load_plane = |plane| self.load_plane_entry(part_id, plane, cache, io_stats.clone());
        let planes = if sequential_plane_loads()? || cache.plane_admission_gated() {
            // Admit the sign dependency first. The high and low reads can then
            // overlap without changing admission policy or assembled column order.
            let sign = load_plane(0).await?;
            let ex = futures::future::try_join_all((1..=last).map(load_plane)).await?;
            std::iter::once(sign).chain(ex).collect()
        } else {
            // Admission does not depend on the sign plane's residency, so a
            // missed partition reads its planes in one round trip.
            // `try_join_all` returns them in plane order.
            futures::future::try_join_all((0..=last).map(load_plane)).await?
        };
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for batch in planes {
            fields.extend(batch.0.schema().fields().iter().cloned());
            columns.extend(batch.0.columns().iter().cloned());
        }
        let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)?;
        let (batch, remapper) = if let Some(remapping) = &self.batch_remapper {
            lance_index_core::remapping::check_batch_remapping_entry()?;
            let row_id_idx = batch.schema().index_of(ROW_ID)?;
            let (batch, remapper) =
                remap_row_ids_preserving_layout_async(remapping.as_ref(), batch, row_id_idx)
                    .await?;
            (batch, Some(remapper))
        } else {
            (batch, self.frag_reuse_index.clone())
        };
        Q::Storage::try_from_batch_at_precision(
            batch,
            self.metadata(),
            self.distance_type,
            remapper,
            precision,
        )
    }

    /// Remappers can remove physical rows, so candidate offsets require an identity mapping.
    pub fn supports_candidate_reads(&self) -> bool {
        self.frag_reuse_index.is_none() && self.batch_remapper.is_none()
    }

    /// Load one whole plane entry of a layered partition.
    ///
    /// A backend that gates lower planes on their sign plane admits a lower
    /// plane only while the partition's sign plane is resident; otherwise the
    /// plane is read without admission, from the persistent entry or the
    /// origin file. Other backends admit every plane through their ordinary
    /// policy, as native partitions are.
    pub async fn load_plane_entry(
        &self,
        part_id: usize,
        plane: u8,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<Arc<PlaneBatch>> {
        let key = PlaneKey {
            partition: part_id,
            plane,
        };
        if !cache.plane_admission_gated() {
            return cache
                .get_or_insert_with_key(key, || async {
                    Ok(PlaneBatch(
                        self.read_plane(part_id, plane, None, io_stats.clone())
                            .await?,
                    ))
                })
                .await;
        }
        let sign_resident = plane == 0
            || cache
                .get_resident_with_key(&PlaneKey {
                    partition: part_id,
                    plane: 0,
                })
                .await
                .is_some();
        let batch = if sign_resident {
            cache
                .get_or_insert_with_key_hit(key, || async {
                    Ok(PlaneBatch(
                        self.read_plane(part_id, plane, None, io_stats.clone())
                            .await?,
                    ))
                })
                .await?
                .0
        } else if let Some(batch) = cache.get_without_promotion_with_key(&key).await {
            batch
        } else {
            // An oversized sign plane falls back to partition streaming; do
            // not admit smaller ex entries without their sign dependency.
            Arc::new(PlaneBatch(
                self.read_plane(part_id, plane, None, io_stats.clone())
                    .await?,
            ))
        };
        Ok(batch)
    }

    /// Ex-plane geometry of a layered index.
    pub fn layered_ex_layout(&self) -> Result<LayeredExLayout> {
        let Quantizer::Rabit(rq) = self.quantizer()? else {
            return Err(Error::invalid_input(
                "ex-plane layout requires a layered IVF_RQ index",
            ));
        };
        let metadata = rq.metadata_ref();
        if !metadata.layered {
            return Err(Error::invalid_input(
                "ex-plane layout requires a layered IVF_RQ index",
            ));
        }
        let layout = RQLayout::try_new(metadata.num_bits)?;
        let rotated_dim = metadata.rotated_dim();
        Ok(LayeredExLayout {
            rotated_dim,
            num_bits: metadata.num_bits,
            row_bytes: [layout.high_bits, layout.low_bits]
                .map(|bits| blocked_ex_code_bytes(rotated_dim, bits) + EX_PLANE_FACTOR_BYTES),
        })
    }

    /// Stage-1 storage of the lazy full-precision scan: the partition's sign
    /// plane alone, loaded (and admitted) like the eager load's sign plane.
    pub async fn load_sign_stage(
        &self,
        part_id: usize,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        if !self.supports_candidate_reads() {
            return Err(Error::invalid_input(
                "the lazy full-precision scan requires an index without a row-id remapper",
            ));
        }
        let sign = self.load_plane_entry(part_id, 0, cache, io_stats).await?;
        Q::Storage::try_from_sign_plane_for_full(
            sign.0.clone(),
            self.metadata(),
            self.distance_type,
        )
    }

    /// Fetch the high and low ex-plane rows at the sorted, unique partition
    /// offsets `rows`, each plane from the source `plan` names.
    ///
    /// When both planes end up whole the batches are the whole planes and
    /// survivors keep their offsets as ex index ([`ExIndex::Identity`]);
    /// otherwise whole planes are cut down to `rows` so both batches hold the
    /// gathered rows in order ([`ExIndex::Compact`]). No rows means no reads.
    #[allow(clippy::too_many_arguments)]
    pub async fn gather_ex_rows(
        &self,
        part_id: usize,
        rows: &[u32],
        plan: GatherPlan,
        config: &LayeredLazyConfig,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<GatheredEx> {
        let partition_rows = self.partition_size(part_id);
        if rows.windows(2).any(|pair| pair[0] >= pair[1])
            || rows
                .last()
                .is_some_and(|&row| row as usize >= partition_rows)
        {
            return Err(Error::invalid_input(format!(
                "gathered offsets of partition {part_id} must be sorted, unique and below its {partition_rows} rows"
            )));
        }
        if rows.is_empty() {
            let (high, low) = futures::try_join!(
                self.read_plane(part_id, 1, Some(Vec::new()), None),
                self.read_plane(part_id, 2, Some(Vec::new()), None),
            )?;
            return Ok(GatheredEx {
                high,
                low,
                index: ExIndex::Compact,
                sources: [None; 2],
                origin_row_reads: 0,
                promotions: Vec::new(),
            });
        }
        let (high, low) = futures::try_join!(
            self.fetch_ex_plane(
                part_id,
                1,
                rows,
                plan.planes[0],
                plan.origin_whole[0],
                config,
                cache,
                io_stats.clone()
            ),
            self.fetch_ex_plane(
                part_id,
                2,
                rows,
                plan.planes[1],
                plan.origin_whole[1],
                config,
                cache,
                io_stats
            ),
        )?;
        let index = if high.source != PlaneSource::Sparse && low.source != PlaneSource::Sparse {
            ExIndex::Identity
        } else {
            ExIndex::Compact
        };
        let indices = UInt32Array::from(rows.to_vec());
        let compact = |plane: &FetchedPlane| -> Result<RecordBatch> {
            if index == ExIndex::Compact && plane.source != PlaneSource::Sparse {
                Ok(plane.batch.take(&indices)?)
            } else {
                Ok(plane.batch.clone())
            }
        };
        Ok(GatheredEx {
            high: compact(&high)?,
            low: compact(&low)?,
            index,
            sources: [Some(high.source), Some(low.source)],
            origin_row_reads: usize::from(high.from_origin) + usize::from(low.from_origin),
            promotions: [high.promotion, low.promotion]
                .into_iter()
                .flatten()
                .collect(),
        })
    }

    /// Fetch one ex plane from `source`; `origin_whole` is the plan's
    /// [`GatherPlan::origin_whole`] for the plane.
    #[allow(clippy::too_many_arguments)]
    async fn fetch_ex_plane(
        &self,
        part_id: usize,
        plane: u8,
        rows: &[u32],
        source: PlaneSource,
        origin_whole: bool,
        config: &LayeredLazyConfig,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<FetchedPlane> {
        let stats = layered_stats::counters();
        let started = Instant::now();
        let key = PlaneKey {
            partition: part_id,
            plane,
        };
        if source == PlaneSource::Resident
            && let Some(batch) = cache.get_resident_with_key(&key).await
        {
            stats.fetch_resident_ns.add_elapsed(started);
            return Ok(FetchedPlane {
                batch: batch.0.clone(),
                source: PlaneSource::Resident,
                from_origin: false,
                promotion: None,
            });
        }
        if source == PlaneSource::Sparse {
            let persisted = cache.get_rows_with_key(&key, rows).await;
            if persisted.is_some() || !origin_whole || cache.plane_admission_gated() {
                let (batch, from_origin) = match persisted {
                    Some(batch) => (batch.0.clone(), false),
                    None => (
                        self.read_plane(part_id, plane, Some(rows.to_vec()), io_stats)
                            .await?,
                        true,
                    ),
                };
                if batch.num_rows() != rows.len() {
                    return Err(Error::internal(format!(
                        "sparse gather of partition {part_id} plane {plane} returned {} rows for {} offsets",
                        batch.num_rows(),
                        rows.len()
                    )));
                }
                stats.fetch_sparse_ns.add_elapsed(started);
                return Ok(FetchedPlane {
                    batch,
                    source: PlaneSource::Sparse,
                    from_origin,
                    promotion: self.request_lazy_promotion(part_id, plane, config, cache),
                });
            }
            // Too many origin row runs: load the whole plane instead, which
            // this backend admits like the eager load, so later queries find
            // it cached.
            stats.origin_whole_fallbacks.incr();
        }
        let batch = self
            .load_plane_entry(part_id, plane, cache, io_stats)
            .await?;
        stats.fetch_whole_ns.add_elapsed(started);
        Ok(FetchedPlane {
            batch: batch.0.clone(),
            source: PlaneSource::Whole,
            from_origin: false,
            promotion: None,
        })
    }

    /// Record a sparse gather of a non-resident plane and, once the policy's
    /// read count is reached, start a promotion unless one for the plane is
    /// already running or every in-flight permit is taken.
    ///
    /// A gated backend admits ex planes only behind their resident sign
    /// plane and evicts the lowest-priority plane first, so a promoted plane
    /// would mostly be read whole and dropped again; promotions run only on
    /// backends that admit planes through their ordinary policy.
    fn request_lazy_promotion(
        &self,
        part_id: usize,
        plane: u8,
        config: &LayeredLazyConfig,
        cache: &WeakLanceCache,
    ) -> Option<LazyPromotionTicket> {
        let LazyPromotion::Background { reads } = config.promote else {
            return None;
        };
        if cache.plane_admission_gated() {
            return None;
        }
        let stats = layered_stats::counters();
        let state = &self.plane_access.lazy;
        let key = (part_id, plane);
        let mut history = state.history.lock().unwrap_or_else(|e| e.into_inner());
        if history.in_flight.contains(&key) {
            stats.promotions_deduped.incr();
            return None;
        }
        let count = history.sparse_reads.entry(key).or_default();
        *count = count.saturating_add(1);
        if *count < reads {
            return None;
        }
        let permits = state
            .permits
            .get_or_init(|| Arc::new(Semaphore::new(config.promote_inflight)));
        let Ok(permit) = permits.clone().try_acquire_owned() else {
            stats.promotions_skipped.incr();
            return None;
        };
        history.sparse_reads.remove(&key);
        history.in_flight.insert(key);
        stats.promotions_issued.incr();
        layered_stats::promotion_started();
        Some(LazyPromotionTicket {
            partition: part_id,
            plane,
            state: state.clone(),
            _permit: permit,
        })
    }

    /// Assemble a full-precision store from candidate rows only.
    pub async fn load_candidates(
        &self,
        part_id: usize,
        rows: Vec<u32>,
        cache: &lance_core::cache::WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        use super::bq::layered::RQPrecision;
        use super::bq::storage::{RABIT_CODE_COLUMN, take_packed_codes};
        use arrow_array::cast::AsArray;
        if !self.supports_candidate_reads() {
            return Err(Error::invalid_input(
                "candidate reads require an index without a row-id remapper",
            ));
        }
        if rows.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::invalid_input(
                "candidate offsets must be sorted and unique",
            ));
        }
        let indices = arrow_array::UInt32Array::from(rows.clone());
        // The three planes are independent. Overlap their cache/origin reads,
        // while preserving plane order when assembling the full row schema.
        let batches = futures::future::try_join_all((0..=2).map(|plane| {
            let rows = &rows;
            let indices = &indices;
            let io_stats = io_stats.clone();
            async move {
                let key = PlaneKey {
                    partition: part_id,
                    plane,
                };
                let resident = cache.get_resident_with_key(&key).await;
                let was_resident = resident.is_some();
                let cached = if plane == 0 && resident.is_none() {
                    cache.get_without_promotion_with_key(&key).await
                } else {
                    resident
                };
                let selected_cached = if plane > 0 && cached.is_none() {
                    cache.get_rows_with_key(&key, rows).await
                } else {
                    None
                };
                let batch = if plane == 0 {
                    // Sign codes are transposed across rows. Gather their physical
                    // offsets directly, then pack only the selected rows.
                    let raw = if let Some(value) = cached {
                        value.0.clone()
                    } else {
                        self.read_plane(part_id, plane, None, io_stats.clone())
                            .await?
                    };
                    let codes = raw
                        .column_by_name(RABIT_CODE_COLUMN)
                        .ok_or_else(|| Error::invalid_input("missing sign codes"))?;
                    let selected_codes = take_packed_codes(codes.as_fixed_size_list(), rows)?;
                    // The transposed sign column cannot be gathered with Arrow
                    // take. Avoid copying it only to replace that copy immediately.
                    let field = raw.schema().field_with_name(RABIT_CODE_COLUMN)?.clone();
                    raw.drop_column(RABIT_CODE_COLUMN)?
                        .take(indices)?
                        .try_with_column(field, Arc::new(selected_codes))?
                } else if let Some(value) = cached {
                    value.0.take(indices)?
                } else if let Some(value) = selected_cached {
                    value.0.clone()
                } else {
                    self.read_plane(part_id, plane, Some(rows.clone()), io_stats.clone())
                        .await?
                };
                if plane > 0
                    && !was_resident
                    && cache
                        .get_resident_with_key(&PlaneKey {
                            partition: part_id,
                            plane: 0,
                        })
                        .await
                        .is_some()
                {
                    let promote = {
                        let mut accesses = self
                            .plane_access
                            .accesses
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        let access = accesses.entry((part_id, plane)).or_default();
                        access.reads = access.reads.saturating_add(1);
                        access.rows = access.rows.saturating_add(rows.len());
                        let cooled = access
                            .last_attempt
                            .is_none_or(|last| last.elapsed() >= PLANE_PROMOTION_COOLDOWN);
                        if cooled
                            && access.reads >= PLANE_PROMOTION_READS
                            && access.rows >= self.partition_size(part_id).div_ceil(2)
                        {
                            access.reads = 0;
                            access.rows = 0;
                            access.last_attempt = Some(std::time::Instant::now());
                            true
                        } else {
                            false
                        }
                    };
                    if promote {
                        cache
                            .get_or_insert_with_key(
                                PlaneKey {
                                    partition: part_id,
                                    plane,
                                },
                                || async {
                                    Ok(super::bq::layered::PlaneBatch(
                                        self.read_plane(part_id, plane, None, io_stats.clone())
                                            .await?,
                                    ))
                                },
                            )
                            .await?;
                    }
                }
                Ok::<_, Error>(batch)
            }
        }))
        .await?;
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for batch in batches {
            fields.extend(batch.schema().fields().iter().cloned());
            columns.extend(batch.columns().iter().cloned());
        }
        let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)?;
        Q::Storage::try_from_batch_at_precision(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
            RQPrecision::Full,
        )
    }

    /// Read only selected rows of a plane; sorted indices are coalesced by the reader.
    pub async fn read_plane(
        &self,
        part_id: usize,
        plane: u8,
        rows: Option<Vec<u32>>,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        let projection = lance_file::versions::reader_projection_from_column_names(
            self.reader.metadata().version(),
            self.reader.schema(),
            super::bq::layered::plane_columns(plane),
        )?;
        let schema = Arc::new(arrow_schema::Schema::from(projection.schema.as_ref()));
        let range = self.ivf.row_range(part_id);
        if range.is_empty() || rows.as_ref().is_some_and(Vec::is_empty) {
            return Ok(RecordBatch::new_empty(schema));
        }
        let params = if let Some(rows) = rows {
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::new();
            for row in rows {
                if row as usize >= range.len() {
                    return Err(Error::invalid_input(
                        "candidate offset exceeds partition length",
                    ));
                }
                let start = range.start as u64 + u64::from(row);
                if let Some(last) = ranges.last_mut()
                    && last.end == start
                {
                    last.end += 1;
                } else {
                    ranges.push(start..start + 1);
                }
            }
            ReadBatchParams::Ranges(ranges.into())
        } else {
            ReadBatchParams::Range(range)
        };
        let reader = match &io_stats {
            Some(stats) => Cow::Owned(self.reader.with_io_stats(stats.recorder())),
            None => Cow::Borrowed(&self.reader),
        };
        let batches = reader
            .read_stream_projected(
                params,
                u32::MAX,
                1,
                projection,
                FilterExpression::no_filter(),
            )
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&schema, batches.iter()).map_err(Into::into)
    }

    /// Read only `columns` of the given rows; the batch keeps file schema order.
    async fn read_vector_columns(
        &self,
        params: ReadBatchParams,
        columns: &[&str],
    ) -> Result<RecordBatch> {
        let projection = lance_file::versions::reader_projection_from_column_names(
            self.reader.version(),
            self.reader.schema(),
            columns,
        )?;
        let schema = Arc::new(projection.schema.as_ref().into());
        let batches = self
            .reader
            .read_stream_projected(
                params,
                u32::MAX,
                1,
                projection,
                FilterExpression::no_filter(),
            )
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        Ok(concat_batches(&schema, batches.iter())?)
    }

    async fn read_vector_range(&self, range: std::ops::Range<usize>) -> Result<RecordBatch> {
        let batches = self
            .reader
            .read_stream(
                ReadBatchParams::Range(range),
                u32::MAX,
                1,
                FilterExpression::no_filter(),
            )
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let schema = Arc::new(self.reader.schema().as_ref().into());
        Ok(concat_batches(&schema, batches.iter())?)
    }

    /// Stage a partition's codes once for native pair scoring, in storage order.
    ///
    /// `batch_size` is a maximum; wide staged rows use smaller power-of-two
    /// batches targeting 16 MiB, with a minimum packed group of 32 rows.
    /// Partitions whose staged size exceeds `memory_limit` are written to
    /// `spill_store` one batch at a time. Row IDs are remapped during staging.
    pub async fn prepare_pairwise_partition(
        &self,
        partition_id: usize,
        batch_size: usize,
        memory_limit: usize,
        centroid: ArrayRef,
        spill_store: &dyn SpillStore,
    ) -> Result<PairwisePartition> {
        if batch_size == 0 || !batch_size.is_multiple_of(32) {
            return Err(Error::invalid_input(format!(
                "pairwise batch_size={batch_size} must be a positive multiple of 32"
            )));
        }
        if partition_id >= self.num_partitions() {
            return Err(Error::invalid_input(format!(
                "partition_id={partition_id} out of range 0..{}",
                self.num_partitions()
            )));
        }
        let num_rows = self.partition_size(partition_id);
        let quantizer = self.quantizer()?;
        let scorer = {
            let quantizer = quantizer.clone();
            let metric = self.distance_type;
            let schema = self.schema();
            Arc::new(
                spawn_cpu(move || PairwiseScorer::new(&quantizer, centroid, metric, &schema))
                    .await?,
            )
        };
        let row_bytes = scorer.row_bytes();
        let rows = PAIRWISE_VECTOR_BATCH_BYTES / row_bytes;
        let batch_size = if rows < batch_size {
            1usize << rows.max(32).ilog2()
        } else {
            batch_size
        };
        let fits = row_bytes
            .checked_mul(num_rows)
            .and_then(|bytes| bytes.checked_add(4096))
            .is_some_and(|bytes| bytes <= memory_limit);
        let encoded = if num_rows == 0 {
            EncodedPartition::Memory(Vec::new())
        } else if fits {
            let source = self
                .read_pairwise_codes(partition_id, 0..num_rows, &quantizer)
                .await?;
            EncodedPartition::Memory(
                stage_pairwise_batches(
                    scorer.clone(),
                    source,
                    batch_size,
                    self.frag_reuse_index.clone(),
                )
                .try_collect()
                .await?,
            )
        } else {
            let mut writer = PairwiseSpillWriter::new(spill_store).await?;
            // Align source reads to whole vector batches so each spill range
            // still corresponds to exactly one staged batch during replay.
            let read_batch_size = PAIRWISE_READ_BATCH_SIZE.div_ceil(batch_size) * batch_size;
            for start in (0..num_rows).step_by(read_batch_size) {
                let end = start.saturating_add(read_batch_size).min(num_rows);
                let source = self
                    .read_pairwise_codes(partition_id, start..end, &quantizer)
                    .await?;
                let mut batches = stage_pairwise_batches(
                    scorer.clone(),
                    source,
                    batch_size,
                    self.frag_reuse_index.clone(),
                );
                while let Some(batch) = batches.try_next().await? {
                    writer.write(batch).await?;
                }
            }
            EncodedPartition::Spilled(writer.finish().await?)
        };
        Ok(PairwisePartition {
            encoded,
            scorer,
            batch_size,
            num_rows,
        })
    }

    /// Read a row range of a partition's index file. PQ codes are returned
    /// column-major over the range (the on-disk transposed layout), which the
    /// PQ scorer stages without transposing.
    async fn read_pairwise_codes(
        &self,
        partition_id: usize,
        range: std::ops::Range<usize>,
        quantizer: &Quantizer,
    ) -> Result<RecordBatch> {
        if partition_id >= self.num_partitions() {
            return Err(Error::invalid_input(format!(
                "partition_id={partition_id} out of range 0..{}",
                self.num_partitions()
            )));
        }
        let partition = self.ivf.row_range(partition_id);
        if range.start > range.end || range.end > partition.len() || range.is_empty() {
            return Err(Error::invalid_input(format!(
                "vector range {range:?} outside partition {partition_id} of size {}",
                partition.len()
            )));
        }
        let start = range.start;
        let rows = partition.start + start..partition.start + range.end;
        let column = quantizer.column();
        let transposed_pq =
            matches!(quantizer, Quantizer::Product(_)) && self.metadata.is_transposed();
        if !transposed_pq || range.len() == partition.len() {
            let batch = self.read_vector_range(rows).await?;
            if !matches!(quantizer, Quantizer::Product(_)) || transposed_pq {
                // Non-PQ codes are row-major, and a whole partition's
                // flattened transposed PQ rows are already column-major.
                return Ok(batch);
            }
            return spawn_cpu(move || {
                let codes = batch
                    .column_by_name(column)
                    .and_then(|codes| codes.as_fixed_size_list_opt())
                    .ok_or_else(|| Error::invalid_input(format!("missing {column}")))?;
                let code_bytes = codes.value_length() as usize;
                let values = codes
                    .values()
                    .as_primitive_opt::<UInt8Type>()
                    .ok_or_else(|| Error::invalid_input(format!("{column} must hold u8 codes")))?;
                let values = super::pq::storage::transpose(values, batch.num_rows(), code_bytes);
                let codes = FixedSizeListArray::try_new_from_values(values, code_bytes as i32)?;
                Ok(batch.replace_column_by_name(column, Arc::new(codes))?)
            })
            .await;
        }
        let schema = arrow_schema::Schema::from(self.reader.schema().as_ref());
        let code_index = schema.index_of(column)?;
        let code_nullable = schema.field(code_index).is_nullable();
        let code_bytes = match schema.field(code_index).data_type() {
            arrow_schema::DataType::FixedSizeList(_, size) => *size as usize,
            other => {
                return Err(Error::invalid_input(format!(
                    "{column} must be a fixed-size list, got {other}"
                )));
            }
        };
        // The file rows at this range hold unrelated code bytes, so read only
        // the other columns (row IDs) there.
        let other_columns = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .filter(|name| *name != column)
            .collect::<Vec<_>>();
        let batch = self
            .read_vector_columns(ReadBatchParams::Range(rows), &other_columns)
            .await?;
        // The flattened column-major matrix spans the whole partition: each
        // code byte's values for this range are one contiguous span. Coalesce
        // the file rows holding all spans into a single ranged read.
        let spans = (0..code_bytes)
            .map(|byte| {
                let first = byte * partition.len() + start;
                first..first + range.len()
            })
            .collect::<Vec<_>>();
        let mut row_ranges: Vec<std::ops::Range<usize>> = Vec::new();
        let mut span_ranges = Vec::with_capacity(spans.len());
        for span in &spans {
            let rows = span.start / code_bytes..span.end.div_ceil(code_bytes);
            match row_ranges.last_mut() {
                Some(last) if last.end >= rows.start => last.end = last.end.max(rows.end),
                _ => row_ranges.push(rows),
            }
            span_ranges.push(row_ranges.len() - 1);
        }
        let encoded = self
            .read_vector_columns(
                ReadBatchParams::Ranges(
                    row_ranges
                        .iter()
                        .map(|rows| {
                            (partition.start + rows.start) as u64
                                ..(partition.start + rows.end) as u64
                        })
                        .collect::<Vec<_>>()
                        .into(),
                ),
                &[column],
            )
            .await?;
        spawn_cpu(move || {
            let values = list_values::<UInt8Type>(&encoded, column)?;
            let mut range_offsets = Vec::with_capacity(row_ranges.len());
            let mut offset = 0;
            for rows in &row_ranges {
                range_offsets.push(offset);
                offset += rows.len();
            }
            let mut codes = Vec::with_capacity(spans.len() * range.len());
            for (span, index) in spans.into_iter().zip(span_ranges) {
                let row = range_offsets[index] + span.start / code_bytes - row_ranges[index].start;
                let first = row * code_bytes + span.start % code_bytes;
                codes.extend_from_slice(&values[first..first + span.len()]);
            }
            let codes = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(codes),
                code_bytes as i32,
            )?;
            let field = arrow_schema::Field::new(column, codes.data_type().clone(), code_nullable);
            Ok(batch.try_with_column_at(code_index, field, Arc::new(codes))?)
        })
        .await
    }

    /// Materialize a compact partition for the parallel prewarm path.
    ///
    /// The input may be a slice of a larger contiguous read. Deep-copying its
    /// visible rows before constructing storage prevents a cached partition
    /// from retaining the entire prewarm window's Arrow buffers.
    #[doc(hidden)]
    pub async fn materialize_partition_for_prewarm(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<Q::Storage>
    where
        Q::Metadata: 'static,
        Q::Storage: 'static,
    {
        let metadata = self.metadata.clone();
        let distance_type = self.distance_type;
        if let Some(remapping) = &self.batch_remapper {
            // Tagged asynchronous path.
            lance_index_core::remapping::check_batch_remapping_entry()?;
            let batch =
                spawn_prewarm_materialization(move || compact_prewarm_batches(batches)).await?;
            let row_id_idx = batch.schema().index_of(ROW_ID)?;
            // Row-map IO must finish outside the CPU-only materialization pool.
            let (batch, remapper) =
                remap_row_ids_preserving_layout_async(remapping.as_ref(), batch, row_id_idx)
                    .await?;
            return spawn_prewarm_materialization(move || {
                Q::Storage::try_from_batch_with_remapper(
                    batch,
                    &metadata,
                    distance_type,
                    Some(remapper),
                )
            })
            .await;
        }
        // Legacy synchronous remapping path.
        let frag_reuse_index = self.frag_reuse_index.clone();
        spawn_prewarm_materialization(move || {
            let batch = compact_prewarm_batches(batches)?;
            Q::Storage::try_from_batch_with_remapper(
                batch,
                &metadata,
                distance_type,
                frag_reuse_index,
            )
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DenseGatherMode, LAZY_DENSE_BYTES_FRACTION_ENV, LAZY_DENSE_ENV, LAZY_DENSE_TO_EAGER_ENV,
        LAZY_EAGER_BEFORE_FULL_ENV, LAZY_FULL_ENV, LAZY_INLINE_ROWS_ENV, LAZY_MAX_RUNS_ENV,
        LAZY_ORIGIN_MAX_RUNS_ENV, LAZY_PROMOTE_ENV, LAZY_PROMOTE_INFLIGHT_ENV, LAZY_WINDOW_ENV,
        LayeredLazyConfig, LazyPromotion, PlaneSource, QueryScratchCapacity, QueryScratchPool,
        SEQUENTIAL_PLANE_LOADS_ENV, compact_prewarm_batches, origin_reads_whole_plane,
        plan_plane_gather, sequential_plane_loads, sequential_plane_loads_from,
        spawn_prewarm_materialization,
    };
    use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
    use lance_core::Error;
    use lance_core::deepsize::DeepSizeOf;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_prewarm_materialization_uses_cpu_pool() {
        let thread_name =
            spawn_prewarm_materialization(|| Ok(std::thread::current().name().map(str::to_owned)))
                .await
                .unwrap();
        assert_eq!(thread_name.as_deref(), Some("lance-cpu"));
    }

    #[test]
    fn test_prewarm_storage_batches_own_compact_buffers() {
        let parent = RecordBatch::try_from_iter([(
            "value",
            Arc::new(UInt64Array::from_iter_values(0..100)) as ArrayRef,
        )])
        .unwrap();
        let parent_array = parent
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let parent_ptr = parent_array.values().as_ptr();
        let parent_size = parent_array.get_array_memory_size();

        let compact = compact_prewarm_batches(vec![parent.slice(10, 10)]).unwrap();
        let compact_array = compact
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_ne!(compact_array.values().as_ptr(), parent_ptr);
        assert!(compact_array.get_array_memory_size() < parent_size);
        assert_eq!(compact_array.values(), &(10..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_query_scratch_pool_reuses_buffers() {
        let pool = QueryScratchPool::new(1);
        let first_ptrs = pool.with_scratch(|scratch| {
            scratch.query_f32.clear();
            scratch.query_f32.resize(16, 1.0);
            scratch.distances.clear();
            scratch.distances.resize(8, 2.0);
            scratch.u16.clear();
            scratch.u16.resize(4, 3);
            scratch.u8.clear();
            scratch.u8.resize(2, 4);
            scratch.u32.clear();
            scratch.u32.resize(3, 5);
            (
                scratch.query_f32.as_ptr(),
                scratch.distances.as_ptr(),
                scratch.u16.as_ptr(),
                scratch.u8.as_ptr(),
                scratch.u32.as_ptr(),
            )
        });

        let second_ptrs = pool.with_scratch(|scratch| {
            assert_eq!(scratch.query_f32.len(), 16);
            assert!(scratch.query_f32.iter().all(|value| *value == 1.0));
            assert_eq!(scratch.distances.len(), 8);
            assert!(scratch.distances.iter().all(|value| *value == 2.0));
            assert_eq!(scratch.u16.len(), 4);
            assert!(scratch.u16.iter().all(|value| *value == 3));
            assert_eq!(scratch.u8.len(), 2);
            assert!(scratch.u8.iter().all(|value| *value == 4));
            assert_eq!(scratch.u32.len(), 3);
            assert!(scratch.u32.iter().all(|value| *value == 5));
            (
                scratch.query_f32.as_ptr(),
                scratch.distances.as_ptr(),
                scratch.u16.as_ptr(),
                scratch.u8.as_ptr(),
                scratch.u32.as_ptr(),
            )
        });

        assert_eq!(first_ptrs, second_ptrs);
    }

    #[test]
    fn test_query_scratch_pool_is_pool_owned() {
        let first_pool = QueryScratchPool::new(1);
        let second_pool = QueryScratchPool::new(1);

        let first_ptr = first_pool.with_scratch(|scratch| {
            scratch.query_f32.resize(16, 1.0);
            scratch.query_f32.as_ptr()
        });
        let second_ptr = second_pool.with_scratch(|scratch| {
            scratch.query_f32.resize(16, 1.0);
            scratch.query_f32.as_ptr()
        });

        assert_ne!(first_ptr, second_ptr);
    }

    #[test]
    fn test_query_scratch_pool_uses_temporary_scratch_when_empty() {
        let pool =
            QueryScratchPool::with_capacity(1, QueryScratchCapacity::new_with_u32(8, 16, 4, 2, 3));
        let pooled = pool.scratch();
        assert!(pooled.pooled);

        let temporary = pool.scratch();
        assert!(!temporary.pooled);
        assert_eq!(temporary.distances.len(), 8);
        assert_eq!(temporary.query_f32.len(), 16);
        assert_eq!(temporary.u16.len(), 4);
        assert_eq!(temporary.u8.len(), 2);
        assert_eq!(temporary.u32.len(), 3);
    }

    #[test]
    fn test_query_scratch_pool_deep_size_includes_buffer_capacity() {
        let empty_size = QueryScratchPool::new(1).deep_size_of();
        let pool =
            QueryScratchPool::with_capacity(1, QueryScratchCapacity::new_with_u32(8, 16, 4, 2, 3));

        assert!(pool.deep_size_of() > empty_size);

        let idle_size = pool.deep_size_of();
        let _checked_out = pool.scratch();

        assert_eq!(pool.deep_size_of(), idle_size);
    }

    #[test]
    fn test_layered_lazy_config_parses_every_knob() {
        let env = HashMap::from([
            (LAZY_FULL_ENV, "1"),
            (LAZY_WINDOW_ENV, "4"),
            (LAZY_DENSE_ENV, "sparse"),
            (LAZY_MAX_RUNS_ENV, "0"),
            (LAZY_DENSE_BYTES_FRACTION_ENV, "0.25"),
            (LAZY_PROMOTE_ENV, "bg:2"),
            (LAZY_PROMOTE_INFLIGHT_ENV, "8"),
            (LAZY_INLINE_ROWS_ENV, "0"),
            (LAZY_EAGER_BEFORE_FULL_ENV, "0"),
            (LAZY_ORIGIN_MAX_RUNS_ENV, "5"),
            (LAZY_DENSE_TO_EAGER_ENV, "0"),
        ]);
        let config =
            LayeredLazyConfig::from_lookup(|name| env.get(name).map(|value| value.to_string()))
                .unwrap();
        assert_eq!(
            config,
            LayeredLazyConfig {
                enabled: true,
                window: 4,
                dense: DenseGatherMode::Sparse,
                max_runs: 0,
                dense_bytes_fraction: 0.25,
                promote: LazyPromotion::Background { reads: 2 },
                promote_inflight: 8,
                inline_rows: 0,
                eager_before_full: false,
                origin_max_runs: 5,
                dense_to_eager: false,
            }
        );
        assert_eq!(
            LayeredLazyConfig::from_lookup(|_| None).unwrap(),
            LayeredLazyConfig::default()
        );
        assert!(!LayeredLazyConfig::default().enabled);
        assert!(LayeredLazyConfig::default().eager_before_full);
        assert_eq!(LayeredLazyConfig::default().origin_max_runs, usize::MAX);
        assert!(LayeredLazyConfig::default().dense_to_eager);
        // The origin run cap does not depend on the eager-before-full switch,
        // and dense probes go to the eager scan unless switched off.
        let env = HashMap::from([(LAZY_FULL_ENV, "1"), (LAZY_ORIGIN_MAX_RUNS_ENV, "2")]);
        let config =
            LayeredLazyConfig::from_lookup(|name| env.get(name).map(|value| value.to_string()))
                .unwrap();
        assert_eq!(
            (
                config.eager_before_full,
                config.origin_max_runs,
                config.dense_to_eager
            ),
            (true, 2, true)
        );
        let env = HashMap::from([(LAZY_FULL_ENV, "1"), (LAZY_DENSE_TO_EAGER_ENV, "false")]);
        let config =
            LayeredLazyConfig::from_lookup(|name| env.get(name).map(|value| value.to_string()))
                .unwrap();
        assert_eq!(
            config,
            LayeredLazyConfig {
                enabled: true,
                dense_to_eager: false,
                ..Default::default()
            }
        );
        let error =
            LayeredLazyConfig::from_lookup(|key| (key == LAZY_FULL_ENV).then(|| "yes".into()))
                .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(LAZY_FULL_ENV), "{error}");
        for (name, value) in [
            (LAZY_DENSE_ENV, "dense"),
            (LAZY_PROMOTE_ENV, "bg:0"),
            (LAZY_PROMOTE_INFLIGHT_ENV, "0"),
            (LAZY_DENSE_BYTES_FRACTION_ENV, "-1"),
            (LAZY_WINDOW_ENV, "many"),
            (LAZY_EAGER_BEFORE_FULL_ENV, "on"),
            (LAZY_ORIGIN_MAX_RUNS_ENV, "-1"),
            (LAZY_DENSE_TO_EAGER_ENV, "2"),
        ] {
            let enabled = |key: &str| -> Option<String> {
                if key == LAZY_FULL_ENV {
                    Some("1".into())
                } else {
                    (key == name).then(|| value.into())
                }
            };
            let error = LayeredLazyConfig::from_lookup(enabled).unwrap_err();
            assert!(
                matches!(error, Error::InvalidInput { .. }),
                "{name}={value}"
            );
            assert!(error.to_string().contains(name), "{error}");
            // With the scan off (unset or 0) the other knobs are ignored.
            for full in [None, Some("0")] {
                let disabled = |key: &str| -> Option<String> {
                    if key == LAZY_FULL_ENV {
                        full.map(String::from)
                    } else {
                        (key == name).then(|| value.into())
                    }
                };
                assert_eq!(
                    LayeredLazyConfig::from_lookup(disabled).unwrap(),
                    LayeredLazyConfig::default(),
                    "{name}={value} with {LAZY_FULL_ENV}={full:?}"
                );
            }
        }
    }

    #[test]
    fn test_plan_plane_gather_cost_rule() {
        // 520-byte rows: rows at most 7 apart share one run.
        let row_bytes = 520;
        let config = LayeredLazyConfig {
            max_runs: 2,
            dense_bytes_fraction: 0.5,
            ..Default::default()
        };
        assert_eq!(
            plan_plane_gather(&[10, 17, 24], 2000, row_bytes, &config),
            PlaneSource::Sparse
        );
        assert_eq!(
            plan_plane_gather(&[10, 100, 1000], 2000, row_bytes, &config),
            PlaneSource::Whole,
            "three runs exceed max_runs"
        );
        assert_eq!(
            plan_plane_gather(&(0..1200).collect::<Vec<_>>(), 2000, row_bytes, &config),
            PlaneSource::Whole,
            "one run covering most of the plane"
        );
        for (dense, expected) in [
            (DenseGatherMode::Sparse, PlaneSource::Sparse),
            (DenseGatherMode::Whole, PlaneSource::Whole),
        ] {
            let config = LayeredLazyConfig { dense, ..config };
            assert_eq!(
                plan_plane_gather(&[10, 100, 1000], 2000, row_bytes, &config),
                expected
            );
            assert_eq!(plan_plane_gather(&[10], 2000, row_bytes, &config), expected);
        }
    }

    #[test]
    fn test_origin_reads_whole_plane_counts_row_runs() {
        // 520-byte rows: rows at most 7 apart share one run.
        let row_bytes = 520;
        let two_runs = [10, 17, 100];
        let three_runs = [10, 100, 1000];
        // Unlimited by default: origin rows are read whatever their runs.
        let unlimited = LayeredLazyConfig::default();
        assert!(!origin_reads_whole_plane(
            &three_runs,
            row_bytes,
            &unlimited
        ));
        let capped = LayeredLazyConfig {
            origin_max_runs: 2,
            ..unlimited
        };
        assert!(!origin_reads_whole_plane(&two_runs, row_bytes, &capped));
        assert!(origin_reads_whole_plane(&three_runs, row_bytes, &capped));
        // The cap applies whether gathers before the heap fills are deferred.
        let deferred = LayeredLazyConfig {
            eager_before_full: false,
            ..capped
        };
        assert!(origin_reads_whole_plane(&three_runs, row_bytes, &deferred));
        let none = LayeredLazyConfig {
            origin_max_runs: 0,
            ..unlimited
        };
        assert!(origin_reads_whole_plane(&[10], row_bytes, &none));
        assert!(!origin_reads_whole_plane(&[], row_bytes, &none));
    }

    #[test]
    fn test_sequential_plane_loads_flag() {
        // The accessor reports the process environment's value.
        let process = std::env::var(SEQUENTIAL_PLANE_LOADS_ENV).ok();
        assert_eq!(
            sequential_plane_loads().ok(),
            sequential_plane_loads_from(process.as_deref()).ok()
        );
        assert!(!sequential_plane_loads_from(None).unwrap());
        assert!(!sequential_plane_loads_from(Some("0")).unwrap());
        assert!(sequential_plane_loads_from(Some("1")).unwrap());
        assert!(sequential_plane_loads_from(Some(" true ")).unwrap());
        let error = sequential_plane_loads_from(Some("yes")).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains(SEQUENTIAL_PLANE_LOADS_ENV),
            "{error}"
        );
    }

    #[test]
    fn test_query_scratch_pool_initializes_buffer_capacity() {
        let pool =
            QueryScratchPool::with_capacity(1, QueryScratchCapacity::new_with_u32(8, 16, 4, 2, 3));

        pool.with_scratch(|scratch| {
            assert_eq!(scratch.distances.len(), 8);
            assert_eq!(scratch.distances.capacity(), 8);
            assert_eq!(scratch.query_f32.len(), 16);
            assert_eq!(scratch.query_f32.capacity(), 16);
            assert_eq!(scratch.u16.len(), 4);
            assert_eq!(scratch.u16.capacity(), 4);
            assert_eq!(scratch.u8.len(), 2);
            assert_eq!(scratch.u8.capacity(), 2);
            assert_eq!(scratch.u32.len(), 3);
            assert_eq!(scratch.u32.capacity(), 3);
        });
    }
}
