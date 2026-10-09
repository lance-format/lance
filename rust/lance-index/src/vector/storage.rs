// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use crate::vector::bq::ex_dot::blocked_ex_code_bytes;
use crate::vector::bq::layered::{
    EntryColumns, PlaneBatch, PlaneKey, RQLayout, SIGN_BOUNDS_PLANE, SignBounds, plane_columns,
    plane_entry_columns,
};
use crate::vector::bq::layered_stats;
use crate::vector::bq::partition_codes::{PartitionCodes, PartitionCodesKey};
use crate::vector::bq::plane_rows::{PACKED_COLUMNS, PlaneRowsSpec};
use crate::vector::bq::resident::{ResidentColumnStore, ResidentLoadTrigger, is_resident};
use crate::vector::bq::storage::{RQRowLayout, normalize_entry_codes};
use crate::vector::bq::transform::ERROR_FACTORS_COLUMN;
use crate::vector::quantizer::{QuantizationMetadata, QuantizationType, QuantizerStorage};
use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::cache::{CacheTier, WeakLanceCache, pinned_partition_cap};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, ReaderProjection};
use lance_io::ReadBatchParams;
use lance_io::object_store::{DEFAULT_CLOUD_IO_PARALLELISM, ObjectStore};
use lance_io::scheduler::IoStats;
use lance_linalg::distance::DistanceType;
use prost::Message;
use std::{
    any::Any,
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::{Arc, LazyLock, Mutex, OnceLock, Weak},
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

use super::exact_buffers::exact_batch;
use super::graph::OrderedFloat;
use super::graph::OrderedNode;
use super::quantizer::{Quantizer, QuantizerMetadata};
use super::{ApproxMode, DISTANCE_TYPE_KEY};

pub use crate::vector::bq::resident::{
    ResidentColumns, ResidentColumnsEntry, ResidentColumnsKey, ResidentPreopen,
    resident_columns_bytes, resident_store_charge, resident_store_count, resident_store_is_live,
    resident_store_leases, resident_store_preopen_lease,
};

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

/// `columns` of `batch`, in that order, sharing its arrays.
fn project_columns(batch: &RecordBatch, columns: &[&str]) -> Result<RecordBatch> {
    let schema = batch.schema();
    let indices = columns
        .iter()
        .map(|column| schema.index_of(column))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(batch.project(&indices)?)
}

/// A prewarm's rows of one partition, slices of a read of several, copied
/// once into buffers of their own, so that the entry keeps neither the
/// whole read nor more than its values (see [`exact_batch`]).
fn compact_prewarm_batches(batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    let schema = batches
        .first()
        .ok_or_else(|| Error::internal("prewarm partition has no storage batches"))?
        .schema();
    if let [batch] = batches.as_slice()
        && batch.num_rows() == 0
    {
        return Ok(batch.clone());
    }
    exact_batch(&schema, &batches, true)
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

/// Identifies an index file across the process: the index's UUID, and the
/// prefix of the object store that holds the file (its scheme and bucket, or
/// whatever else tells the store apart; see [`ObjectStore::store_prefix`])
/// with the file's path in that store. Every open of the file binds to the
/// runtime handles registered under it, so that indexes opened at once, a
/// re-open while an older index still runs, and a state read back from a
/// persistent cache tier share them: the resident store
/// ([`ResidentColumns::in_index_cache`]) and the lazy scan's far gather
/// permits ([`IvfQuantizationStorage::with_index_file`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexFileKey(Arc<IndexFileLocation>);

#[derive(Debug, PartialEq, Eq, Hash)]
struct IndexFileLocation {
    index_uuid: String,
    store_prefix: String,
    path: String,
}

impl IndexFileKey {
    /// The file at `path` in the object store with prefix `store_prefix`,
    /// of the index with UUID `index_uuid`.
    pub fn new(index_uuid: &str, store_prefix: &str, path: &str) -> Self {
        Self(Arc::new(IndexFileLocation {
            index_uuid: index_uuid.to_owned(),
            store_prefix: store_prefix.to_owned(),
            path: path.to_owned(),
        }))
    }

    /// The UUID of the index the file belongs to.
    pub(crate) fn index_uuid(&self) -> &str {
        &self.0.index_uuid
    }

    /// The prefix of the object store that holds the file.
    pub(crate) fn store_prefix(&self) -> &str {
        &self.0.store_prefix
    }

    /// The file's path in its object store.
    pub(crate) fn path(&self) -> &str {
        &self.0.path
    }
}

/// Values shared process-wide by key and held weakly: a value lives while
/// anything outside the registry holds it.
pub(crate) type WeakRegistry<K, V> = Mutex<HashMap<K, Weak<V>>>;

/// The value `registry` holds for `key` while anything else still holds it,
/// else the one `create` makes, registered for the next caller. Entries of
/// dropped values are pruned whenever a value is registered.
pub(crate) fn shared_by_key<K: Eq + Hash + Clone, V>(
    registry: &LazyLock<WeakRegistry<K, V>>,
    key: &K,
    create: impl FnOnce() -> Arc<V>,
) -> Arc<V> {
    let mut values = registry.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(value) = values.get(key).and_then(Weak::upgrade) {
        return value;
    }
    let value = create();
    values.retain(|_, held| held.strong_count() > 0);
    values.insert(key.clone(), Arc::downgrade(&value));
    value
}

/// Runtime promotion history shared by reconstructions of the same cached index.
/// This contains no readers or object-store handles and is not persisted to disk.
#[derive(Debug, Clone, Default)]
pub struct PlaneAccessTracker {
    /// Cascade candidate reads, see [`PLANE_PROMOTION_READS`].
    accesses: Arc<Mutex<HashMap<(usize, u8), PlaneAccess>>>,
    /// Background promotions and far gather permits of the lazy
    /// full-precision scan.
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
    /// Bounds the lazy scan's gathers in flight beyond the ordinary window
    /// across queries; see [`IvfQuantizationStorage::lazy_far_permits`].
    far_permits: Mutex<Option<LazyFarPermits>>,
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

/// The lazy far gather pools of the open index files, by file and pool size;
/// see [`LazyFarPermits::for_file`].
static FAR_PERMIT_POOLS: LazyLock<WeakRegistry<(IndexFileKey, usize), Semaphore>> =
    LazyLock::new(Default::default);

/// The permits of an index's lazy gathers issued beyond the ordinary window
/// ([`LAZY_FAR_WINDOW_ENV`]), shared by the index's queries and by the
/// reconstructions of its cached state, and by every open of its file once
/// bound to it ([`IvfQuantizationStorage::with_index_file`]). A gather holds
/// its permit until its reads return; dropping the permit, also when the
/// gather is cancelled, returns it to the pool.
#[derive(Debug, Clone)]
pub struct LazyFarPermits {
    semaphore: Arc<Semaphore>,
    /// Permits the pool holds while none is taken.
    size: usize,
}

impl LazyFarPermits {
    /// A pool of `size` permits, at most [`Semaphore::MAX_PERMITS`].
    pub fn try_new(size: usize) -> Result<Self> {
        if size > Semaphore::MAX_PERMITS {
            return Err(Error::invalid_input(format!(
                "a lazy far gather pool of {size} permits exceeds the {} a pool can hold",
                Semaphore::MAX_PERMITS
            )));
        }
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(size)),
            size,
        })
    }

    /// The pool of `size` permits that every open of index file `file` in
    /// the process shares: the live one while an index, a cached state or a
    /// taken permit holds it, else a new one. Keyed by size too, so that an
    /// index whose config sets another size (tests replace it) never draws
    /// from a pool of the old size.
    fn for_file(file: &IndexFileKey, size: usize) -> Result<Self> {
        let new = Self::try_new(size)?;
        let semaphore = shared_by_key(&FAR_PERMIT_POOLS, &(file.clone(), size), || new.semaphore);
        Ok(Self { semaphore, size })
    }

    /// A permit, if one is free now.
    pub fn try_acquire(&self) -> Option<OwnedSemaphorePermit> {
        self.semaphore.clone().try_acquire_owned().ok()
    }

    /// Wait until a permit is free and take it. Dropping the future before
    /// then takes none.
    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit> {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::internal("the lazy far gather permit pool was closed"))
    }

    /// Permits the pool holds while none is taken.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Permits taken now, i.e. gathers issued beyond the ordinary window
    /// whose reads have not returned.
    pub fn in_flight(&self) -> usize {
        // Permits only return to the pool when dropped, so at most `size` are free.
        self.size - self.semaphore.available_permits()
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
/// Background promotion policy: `off` (default), `auto` or `bg:N` (promote
/// after N sparse gathers). `auto` is `off` for an
/// [`OriginLatencyClass::High`] origin and `bg:1` otherwise, resolved when an
/// index opens (see [`LazyPromotion::resolve`]). The default is `off` on every
/// origin because the V10 workload measured no gain from promotions on local
/// NVMe either, at the cost of extra plane reads. Backends that gate plane
/// admission never promote.
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
/// wait for the heap to fill or for every earlier probe to be scored. The
/// large-`k` probes that [`LAZY_DENSE_TO_EAGER_ENV`] routes are scored
/// eagerly instead and never gathered.
pub const LAZY_EAGER_BEFORE_FULL_ENV: &str = "LANCE_RQ_LAZY_EAGER_BEFORE_FULL";
/// Which probes that `k` and the partition sizes predict to be gathered
/// whole are loaded and scored by the eager scan instead: `off`, `origin`
/// (default) or `all` (see [`DenseToEager`]; the flag's former `0` and `1`
/// mean `off` and `all`). A probe is predicted dense when it is scored
/// before the probes ahead of it can hold `k` rows, so its gather selects
/// every row, and every probe is when the gathers issued once the heap fills
/// are expected to read whole planes (large `k`). The eager load reads the
/// partition's planes together (all at once on backends that do not gate
/// plane admission), with as many probes in flight as the eager scan
/// prepares, instead of the sign plane at staging and the ex planes a round
/// trip later within the gather window. Only queries without a prefilter
/// (deletions included) or upper distance bound are routed: a gather with an
/// infinite threshold selects only the accepted rows below the bound, which
/// the probe's sign plane tells. The `sparse` gather policy
/// ([`LAZY_DENSE_ENV`]) never reads a whole plane and routes no probe.
pub const LAZY_DENSE_TO_EAGER_ENV: &str = "LANCE_RQ_LAZY_DENSE_TO_EAGER";
/// Most coalesced row runs a sparse gather reads from the origin file when the
/// persistent tier does not hold the plane; with more, it loads (and admits)
/// the whole plane instead. Unlimited by default, and ignored by backends that
/// gate plane admission, which would not admit the plane.
pub const LAZY_ORIGIN_MAX_RUNS_ENV: &str = "LANCE_RQ_LAZY_ORIGIN_MAX_RUNS";
/// Largest gap in bytes between the row runs of a sparse gather that reads the
/// origin file, because the persistent tier does not hold the plane, that one
/// request still spans: `auto` (default) or a byte count. Each run is cut back
/// out of the request before decoding, so only the requests and the bytes
/// read change, not the rows returned. `auto` is
/// [`HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES`] for an [`OriginLatencyClass::High`]
/// origin and the object store's block size, the gap of every other read,
/// otherwise (see [`LazyOriginGap`]).
pub const LAZY_ORIGIN_GAP_BYTES_ENV: &str = "LANCE_RQ_LAZY_ORIGIN_GAP_BYTES";
/// The gap `auto` gives the sparse origin reads of an
/// [`OriginLatencyClass::High`] origin. S3 latency per request is about flat
/// from 60 KiB to 700 KiB, so merging runs up to 256 KiB apart makes about a
/// third fewer S3 requests than the 64 KiB block size. Lazy scans on S3 were
/// faster with it than with 1 MiB, which makes fewer requests but reads more
/// bytes between the runs.
pub const HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES: u64 = 256 * 1024;
/// Staleness window of the gathers of probes that read an ex plane from an
/// [`OriginLatencyClass::High`] origin, because no cache tier held their high
/// or low plane when they were staged (default 64): such a gather may be
/// issued while up to this many earlier probes are unscored, instead of
/// [`LAZY_WINDOW_ENV`]'s. A gather issued beyond the ordinary window holds a
/// permit of the index's pool ([`LAZY_FAR_INFLIGHT_ENV`]) until its reads
/// return; when none is free it waits for a permit or for the ordinary
/// window, whichever comes first, so it is never issued later than without
/// this window. A value no larger than [`LAZY_WINDOW_ENV`]'s turns it off, and
/// it never applies to a low-latency origin. Results are the same either way.
pub const LAZY_FAR_WINDOW_ENV: &str = "LANCE_RQ_LAZY_FAR_WINDOW";
/// Most gathers of one index, over all its queries, issued beyond the
/// ordinary window under [`LAZY_FAR_WINDOW_ENV`] whose reads have not
/// returned (default 64, at least 1). The pool is sized once per index.
pub const LAZY_FAR_INFLIGHT_ENV: &str = "LANCE_RQ_LAZY_FAR_INFLIGHT";
/// When the lazy scan publishes the heap's threshold to the gathers once a
/// lazy probe's survivors first fill the heap to `k` rows: `off` (default)
/// only once the probe is scored whole, as every other probe publishes, or
/// `on` as soon as they do, partway through scoring the probe. A gather
/// waiting for the threshold selects its survivors against the first one
/// published; a mid-probe threshold is the `k`-th best of the rows scored so
/// far, looser than the probe's final one, so such a gather reads more rows.
/// On S3 at 32 concurrent queries, the gathers of probe ranks 1-15 selected
/// about 80x (k=10) and 30x (k=100) as many rows with `on` as with `off`,
/// which served 1.49x and 1.32x the queries per second; at one query in
/// flight the two did not differ. Results are the same either way: the
/// heap's top never increases, so a later threshold still covers every row
/// stage 2, pruning on the live heap, scores.
pub const LAZY_PARTIAL_PUBLISH_ENV: &str = "LANCE_RQ_LAZY_PARTIAL_PUBLISH";
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
/// Every probe of a 64-probe query. On S3, a window of 64 for every gather
/// cut mean latency to 0.69–0.81x of the ordinary window's at k=100 and
/// k=1000, while 32 gave MS MARCO no gain.
const DEFAULT_LAZY_FAR_WINDOW: usize = 64;
/// As many as Lance's default I/O parallelism for cloud object stores.
const DEFAULT_LAZY_FAR_INFLIGHT: usize = DEFAULT_CLOUD_IO_PARALLELISM;

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

/// Latency class of an IVF_RQ index's origin reads, those that no cache tier
/// serves: `auto` (default), `low` or `high`. `auto` takes the class the
/// session opening the index declares, and without one the class of its
/// object store: high for a cloud object store ([`ObjectStore::is_cloud`]) and
/// low for local files and memory. A process whose object store is wrapped in
/// a local cache that also caches index files declares `low` on its session
/// instead, since the wrapper hides that from the store's scheme; `low` or
/// `high` here overrides every session. Read once per process; the class is
/// resolved when an index opens, see [`OriginLatencyClass::resolve_with_hint`].
pub const ORIGIN_LATENCY_ENV: &str = "LANCE_RQ_ORIGIN_LATENCY";

/// How slow an index's origin reads are. Reader policies that trade extra
/// bytes or background work for fewer origin requests default to on only for
/// [`Self::High`]. Whether an index keeps its small columns resident does
/// not depend on the class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum OriginLatencyClass {
    /// Local files and memory.
    #[default]
    Low,
    /// Cloud object stores, where every request costs a network round trip.
    High,
}

impl OriginLatencyClass {
    /// The class `auto` resolves to for an index read from `object_store`.
    pub fn of_store(object_store: &ObjectStore) -> Self {
        if object_store.is_cloud() {
            Self::High
        } else {
            Self::Low
        }
    }

    /// The class of an index read from `object_store` under `setting`, the
    /// value of [`ORIGIN_LATENCY_ENV`] with `None` for `auto`, when no session
    /// declares one.
    pub fn resolve(setting: Option<Self>, object_store: &ObjectStore) -> Self {
        Self::resolve_with_hint(setting, None, object_store)
    }

    /// The class of an index read from `object_store` under `setting`, the
    /// value of [`ORIGIN_LATENCY_ENV`] with `None` for `auto`, opened by a
    /// session that declares `hint` for the indexes it opens (`None` when it
    /// declares none). An explicit setting wins over the hint, and the hint
    /// over the store's own class ([`Self::of_store`]): only the serving
    /// process knows whether a wrapper around the store serves index reads
    /// from a local cache.
    pub fn resolve_with_hint(
        setting: Option<Self>,
        hint: Option<Self>,
        object_store: &ObjectStore,
    ) -> Self {
        setting
            .or(hint)
            .unwrap_or_else(|| Self::of_store(object_store))
    }

    /// The knob's spelling of the class: `low` or `high`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
        }
    }

    /// Whether reading planes that the index cache holds in `tiers` makes a
    /// request to a slow origin: this class is [`Self::High`] and some plane
    /// is [`CacheTier::Absent`]. A lazy probe whose ex planes are such is
    /// counted in `s3_bound_probes`.
    pub fn reads_slow_origin(self, tiers: &[CacheTier]) -> bool {
        self == Self::High && tiers.contains(&CacheTier::Absent)
    }
}

impl std::fmt::Display for OriginLatencyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// [`ORIGIN_LATENCY_ENV`], read once per process.
static ORIGIN_LATENCY: LazyLock<std::result::Result<Option<OriginLatencyClass>, String>> =
    LazyLock::new(|| {
        origin_latency_from(std::env::var(ORIGIN_LATENCY_ENV).ok().as_deref())
            .map_err(|err| err.to_string())
    });

/// The class [`ORIGIN_LATENCY_ENV`] sets, `None` for `auto`. The variable is
/// read once per process; an invalid value fails here and every IVF_RQ index
/// open.
pub fn origin_latency_setting() -> Result<Option<OriginLatencyClass>> {
    ORIGIN_LATENCY.clone().map_err(Error::invalid_input)
}

fn origin_latency_from(value: Option<&str>) -> Result<Option<OriginLatencyClass>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.trim() {
        "auto" => Ok(None),
        "low" => Ok(Some(OriginLatencyClass::Low)),
        "high" => Ok(Some(OriginLatencyClass::High)),
        _ => Err(Error::invalid_input(format!(
            "{ORIGIN_LATENCY_ENV}={value:?} is invalid, expected auto, low or high"
        ))),
    }
}

/// Where a layered index's cache keeps the estimator bounds columns:
/// `lazy` (default) in their own plane entry, read only by High precision and
/// by full precision on a file without error factors, or `eager` in the sign
/// plane, which every scan reads (see [`SignBounds`]). Results are the same
/// either way. Read once per process; the placement is resolved when a
/// layered index opens.
pub const SIGN_BOUNDS_ENV: &str = "LANCE_RQ_SIGN_BOUNDS";

/// [`SIGN_BOUNDS_ENV`], read once per process.
static SIGN_BOUNDS: LazyLock<std::result::Result<SignBounds, String>> = LazyLock::new(|| {
    sign_bounds_from(std::env::var(SIGN_BOUNDS_ENV).ok().as_deref()).map_err(|err| err.to_string())
});

/// The bounds placement [`SIGN_BOUNDS_ENV`] sets. The variable is read once
/// per process; an invalid value fails here and every layered IVF_RQ index
/// open.
pub fn sign_bounds_setting() -> Result<SignBounds> {
    SIGN_BOUNDS.clone().map_err(Error::invalid_input)
}

fn sign_bounds_from(value: Option<&str>) -> Result<SignBounds> {
    let Some(value) = value else {
        return Ok(SignBounds::default());
    };
    match value.trim() {
        "lazy" => Ok(SignBounds::Lazy),
        "eager" => Ok(SignBounds::Eager),
        _ => Err(Error::invalid_input(format!(
            "{SIGN_BOUNDS_ENV}={value:?} is invalid, expected lazy or eager"
        ))),
    }
}

/// Whether an IVF_RQ index, native or layered, keeps the small columns of its
/// storage file (row ids and factors) in memory, so that partition and plane
/// reads fetch only the code and bounds columns from the file: `auto`
/// (default), `on` or `off`. `auto` is on wherever the opening index's cache
/// can pin the store, whatever the origin (see
/// [`ResidentColumnsSetting::admits`]). An index of the file loads the store
/// when it opens, unless a live index or the index cache already holds it.
/// The store is an entry of the index cache, charged in its budget and
/// leased, so kept in RAM, while an index of the file is live (see
/// [`ResidentColumns`]); [`resident_columns_bytes`] gives its size without
/// I/O. Results are the same either way. `on` or `off` here overrides the
/// setting of the session opening an index, which `auto` defers to (see
/// [`ResidentColumnsSetting::resolve_with_session`]). Read once per process;
/// resolved when an index opens.
pub const RESIDENT_COLUMNS_ENV: &str = "LANCE_RQ_RESIDENT_COLUMNS";

/// The size of an IVF_RQ storage file's resident store, known without
/// reading it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentStoreSize {
    /// Bytes of values the store holds ([`resident_columns_bytes`]).
    pub bytes: u64,
    /// Bytes an index cache charges for the loaded store
    /// ([`resident_store_charge`]).
    pub charge: u64,
}

/// Whether a resident store an index cache charges `charge` bytes for fits
/// the cache's pinned cap, so that a lease pins it: the cap
/// ([`lance_core::cache::pinned_partition_cap`]) of a partition as large as
/// the largest entry the cache admits, `max_entry_bytes`
/// ([`lance_core::cache::CacheBackend::max_entry_bytes`]; `None`: no limit
/// below its capacity). A store past the cap would overflow on every lease
/// and stay evictable while in use; one that takes most of a cache shard
/// would also leave the planes too little room.
pub fn resident_store_fits(charge: u64, max_entry_bytes: Option<u64>) -> bool {
    max_entry_bytes.is_none_or(|max| charge <= pinned_partition_cap(max))
}

/// The value of [`RESIDENT_COLUMNS_ENV`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ResidentColumnsSetting {
    /// On wherever the opening index charges the store in an index cache
    /// that can pin it (one with a pin budget) and its charge fits that
    /// cache's pinned cap ([`resident_store_fits`]), whatever the origin;
    /// off otherwise.
    #[default]
    Auto,
    /// On whatever the store's size or cache. An oversize store, or one in
    /// a cache that cannot pin, stays evictable while in use. An index
    /// opened without an index cache shares an uncharged store with the
    /// file's live indexes.
    On,
    /// Off: reads fetch every column they return from the file.
    Off,
}

impl ResidentColumnsSetting {
    /// The setting an index opens with: `env`, the value of
    /// [`RESIDENT_COLUMNS_ENV`], when it is `on` or `off`, and otherwise
    /// `session`, what the opening session sets for the IVF_RQ indexes it
    /// opens (`auto` unless it sets one).
    pub fn resolve_with_session(env: Self, session: Self) -> Self {
        match env {
            Self::Auto => session,
            Self::On | Self::Off => env,
        }
    }

    /// Whether an index keeps its small columns resident in a store of
    /// `store`'s size, charged in an index cache whose largest admissible
    /// entry is `max_entry_bytes` and that has a pin budget when
    /// `has_pin_budget` ([`lance_core::cache::PinnedStats::cap_bytes`] above
    /// zero): never when `off`, and only a store with some columns
    /// otherwise; `auto` also requires the pin budget and the bytes the
    /// cache charges for the store to fit its pinned cap
    /// ([`resident_store_fits`]), so that a lease pins it. `on` and `off`
    /// ignore `has_pin_budget`.
    pub fn admits(
        self,
        store: ResidentStoreSize,
        max_entry_bytes: Option<u64>,
        has_pin_budget: bool,
    ) -> bool {
        match self {
            Self::Off => false,
            Self::On => store.bytes > 0,
            Self::Auto => {
                store.bytes > 0
                    && has_pin_budget
                    && resident_store_fits(store.charge, max_entry_bytes)
            }
        }
    }

    /// The knob's spelling of the setting: `auto`, `on` or `off`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

impl std::fmt::Display for ResidentColumnsSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// [`RESIDENT_COLUMNS_ENV`], read once per process.
static RESIDENT_COLUMNS: LazyLock<std::result::Result<ResidentColumnsSetting, String>> =
    LazyLock::new(|| {
        resident_columns_from(std::env::var(RESIDENT_COLUMNS_ENV).ok().as_deref())
            .map_err(|err| err.to_string())
    });

/// The setting of [`RESIDENT_COLUMNS_ENV`]. The variable is read once per
/// process; an invalid value fails here and every IVF_RQ index open.
pub fn resident_columns_setting() -> Result<ResidentColumnsSetting> {
    RESIDENT_COLUMNS.clone().map_err(Error::invalid_input)
}

fn resident_columns_from(value: Option<&str>) -> Result<ResidentColumnsSetting> {
    let Some(value) = value else {
        return Ok(ResidentColumnsSetting::default());
    };
    match value.trim() {
        "auto" => Ok(ResidentColumnsSetting::Auto),
        "on" => Ok(ResidentColumnsSetting::On),
        "off" => Ok(ResidentColumnsSetting::Off),
        _ => Err(Error::invalid_input(format!(
            "{RESIDENT_COLUMNS_ENV}={value:?} is invalid, expected auto, on or off"
        ))),
    }
}

/// How long an IVF_RQ index file's resident store is kept in RAM: `index`
/// (default) or `process`. Read once per process; an invalid value fails
/// every IVF_RQ index open.
pub const RESIDENT_LIFETIME_ENV: &str = "LANCE_RQ_RESIDENT_LIFETIME";

/// The value of [`RESIDENT_LIFETIME_ENV`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ResidentLifetime {
    /// Leased by the live indexes of the file: an idle store is an ordinary
    /// entry of the index cache, evicted under pressure and loaded again on
    /// its next use.
    #[default]
    Index,
    /// Also leased once for the life of the process, so the index cache
    /// keeps it pinned (within its pinned cap) once it loaded. Still charged
    /// in the cache budget.
    Process,
}

impl ResidentLifetime {
    /// The knob's spelling of the lifetime: `index` or `process`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Process => "process",
        }
    }
}

impl std::fmt::Display for ResidentLifetime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// [`RESIDENT_LIFETIME_ENV`], read once per process.
static RESIDENT_LIFETIME: LazyLock<std::result::Result<ResidentLifetime, String>> =
    LazyLock::new(|| {
        resident_lifetime_from(std::env::var(RESIDENT_LIFETIME_ENV).ok().as_deref())
            .map_err(|err| err.to_string())
    });

/// The setting of [`RESIDENT_LIFETIME_ENV`].
pub fn resident_lifetime_setting() -> Result<ResidentLifetime> {
    RESIDENT_LIFETIME.clone().map_err(Error::invalid_input)
}

fn resident_lifetime_from(value: Option<&str>) -> Result<ResidentLifetime> {
    let Some(value) = value else {
        return Ok(ResidentLifetime::default());
    };
    match value.trim() {
        "index" => Ok(ResidentLifetime::Index),
        "process" => Ok(ResidentLifetime::Process),
        _ => Err(Error::invalid_input(format!(
            "{RESIDENT_LIFETIME_ENV}={value:?} is invalid, expected index or process"
        ))),
    }
}

/// What an IVF_RQ index's cache entries hold while its small columns are
/// resident (see [`RESIDENT_COLUMNS_ENV`]): `codes` (default) or `all`. With
/// `codes`, a layered index's sign, high and low plane entries and a native
/// flat index's partition entries ([`PartitionCodes`]) hold only the columns
/// reads fetch from the file, and every read attaches copies of the resident
/// rows, so the cache keeps no second copy of the row ids and factors. An
/// index without a resident store keeps `all`, whatever the setting (see
/// [`EntryColumns::resolve`]). Results are the same either way. Read once
/// per process; resolved when an index opens.
pub const ENTRY_COLUMNS_ENV: &str = "LANCE_RQ_ENTRY_COLUMNS";

/// The setting of [`ENTRY_COLUMNS_ENV`] when it is unset.
pub const DEFAULT_ENTRY_COLUMNS: EntryColumns = EntryColumns::Codes;

/// [`ENTRY_COLUMNS_ENV`], read once per process.
static ENTRY_COLUMNS: LazyLock<std::result::Result<EntryColumns, String>> = LazyLock::new(|| {
    entry_columns_from(std::env::var(ENTRY_COLUMNS_ENV).ok().as_deref())
        .map_err(|err| err.to_string())
});

/// The setting of [`ENTRY_COLUMNS_ENV`]. The variable is read once per
/// process; an invalid value fails here and every IVF_RQ index open.
pub fn entry_columns_setting() -> Result<EntryColumns> {
    ENTRY_COLUMNS.clone().map_err(Error::invalid_input)
}

fn entry_columns_from(value: Option<&str>) -> Result<EntryColumns> {
    let Some(value) = value else {
        return Ok(DEFAULT_ENTRY_COLUMNS);
    };
    match value.trim() {
        "codes" => Ok(EntryColumns::Codes),
        "all" => Ok(EntryColumns::All),
        _ => Err(Error::invalid_input(format!(
            "{ENTRY_COLUMNS_ENV}={value:?} is invalid, expected codes or all"
        ))),
    }
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LazyPromotion {
    /// [`Self::Off`] for an [`OriginLatencyClass::High`] origin and `bg:1`
    /// otherwise; see [`Self::resolve`]. An index resolves it when it opens,
    /// so a scan never sees it (and would promote nothing).
    Auto,
    /// Only whole-plane gathers admit planes.
    #[default]
    Off,
    /// Promote a plane after this many sparse gathers of it while not
    /// resident, on backends that do not gate plane admission.
    Background { reads: u32 },
}

impl LazyPromotion {
    /// The policy of an index whose origin is `class`: [`Self::Auto`]
    /// resolves, an explicit policy applies whatever the origin.
    ///
    /// A promotion reads the whole plane again, from the origin when no
    /// cache tier holds it, and its admission evicts sign planes that later
    /// queries then miss. On S3 that costs more origin requests than the
    /// promoted planes save: without promotions the lazy scan of MS MARCO at
    /// k=100 made 175 instead of 235 requests per query and read 12.0
    /// instead of 21.6 MiB.
    pub fn resolve(self, class: OriginLatencyClass) -> Self {
        match (self, class) {
            (Self::Auto, OriginLatencyClass::High) => Self::Off,
            (Self::Auto, OriginLatencyClass::Low) => Self::Background {
                reads: DEFAULT_LAZY_PROMOTE_READS,
            },
            (policy, _) => policy,
        }
    }
}

/// The knob's spelling of the policy: `auto`, `off` or `bg:N`.
impl std::fmt::Display for LazyPromotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Off => f.write_str("off"),
            Self::Background { reads } => write!(f, "bg:{reads}"),
        }
    }
}

/// The value of [`LAZY_DENSE_TO_EAGER_ENV`]: which probes predicted to be
/// gathered whole, of non-empty partitions whose high and low planes are not
/// both resident, the eager scan loads and scores instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DenseToEager {
    /// None: every such probe takes the lazy pipeline.
    Off,
    /// Those whose high or low plane no cache tier holds on an
    /// [`OriginLatencyClass::High`] origin, where the eager load saves a
    /// round trip to the origin (see
    /// [`OriginLatencyClass::reads_slow_origin`]). Where the planes are
    /// local, loading them eagerly saves little and delays scoring: routing
    /// every such probe on NVMe raised the p99 of Coyo k=10000 queries from
    /// 68.4 to 90.5 ms.
    #[default]
    Origin,
    /// Every one, wherever its planes are.
    All,
}

impl DenseToEager {
    /// Whether a predicted-dense probe of a non-empty partition whose high
    /// and low planes are in `tiers`, not both resident, is loaded and
    /// scored by the eager scan on an origin of `class`.
    pub fn routes(self, class: OriginLatencyClass, tiers: &[CacheTier]) -> bool {
        match self {
            Self::Off => false,
            Self::Origin => class.reads_slow_origin(tiers),
            Self::All => true,
        }
    }

    /// The knob's spelling of the mode: `off`, `origin` or `all`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Origin => "origin",
            Self::All => "all",
        }
    }
}

impl std::fmt::Display for DenseToEager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The value of [`LAZY_ORIGIN_GAP_BYTES_ENV`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LazyOriginGap {
    /// [`HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES`] for an
    /// [`OriginLatencyClass::High`] origin, where every request is a round
    /// trip; the object store's block size otherwise.
    #[default]
    Auto,
    /// This many bytes, whatever the origin.
    Bytes(u64),
}

impl LazyOriginGap {
    /// The gap within which the sparse origin reads of an index whose origin
    /// is `class` merge row runs into one request, or `None` for the object
    /// store's block size, which every other read of the file uses.
    pub fn coalesce_gap(self, class: OriginLatencyClass) -> Option<u64> {
        match (self, class) {
            (Self::Bytes(bytes), _) => Some(bytes),
            (Self::Auto, OriginLatencyClass::High) => Some(HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES),
            (Self::Auto, OriginLatencyClass::Low) => None,
        }
    }

    /// The gap in bytes for an origin of `class` whose object store merges
    /// ranges within `block_size` bytes by default.
    pub fn resolve(self, class: OriginLatencyClass, block_size: u64) -> u64 {
        self.coalesce_gap(class).unwrap_or(block_size)
    }
}

/// Settings of the lazy layered full-precision scan, which bounds every row
/// from the sign plane and reads the ex planes of the survivors only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayeredLazyConfig {
    pub enabled: bool,
    /// Probes a gather may run ahead of scoring; see [`LAZY_WINDOW_ENV`],
    /// and [`Self::far_window`] for gathers that read a slow origin.
    pub window: usize,
    pub dense: DenseGatherMode,
    pub max_runs: usize,
    pub dense_bytes_fraction: f64,
    /// See [`LAZY_PROMOTE_ENV`]; an index holds it resolved for its origin.
    pub promote: LazyPromotion,
    pub promote_inflight: usize,
    pub inline_rows: usize,
    /// See [`LAZY_EAGER_BEFORE_FULL_ENV`].
    pub eager_before_full: bool,
    /// See [`LAZY_ORIGIN_MAX_RUNS_ENV`]; `usize::MAX` never falls back.
    pub origin_max_runs: usize,
    /// See [`LAZY_ORIGIN_GAP_BYTES_ENV`].
    pub origin_gap: LazyOriginGap,
    /// See [`LAZY_DENSE_TO_EAGER_ENV`].
    pub dense_to_eager: DenseToEager,
    /// See [`LAZY_FAR_WINDOW_ENV`] and [`Self::active_far_window`].
    pub far_window: usize,
    /// See [`LAZY_FAR_INFLIGHT_ENV`].
    pub far_inflight: usize,
    /// See [`LAZY_PARTIAL_PUBLISH_ENV`]: `true` publishes the threshold as
    /// soon as a probe's survivors fill the heap.
    pub partial_publish: bool,
}

impl Default for LayeredLazyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            window: DEFAULT_LAZY_WINDOW,
            dense: DenseGatherMode::Cost,
            max_runs: DEFAULT_LAZY_MAX_RUNS,
            dense_bytes_fraction: DEFAULT_LAZY_DENSE_BYTES_FRACTION,
            promote: LazyPromotion::Off,
            promote_inflight: DEFAULT_LAZY_PROMOTE_INFLIGHT,
            inline_rows: DEFAULT_LAZY_INLINE_ROWS,
            eager_before_full: true,
            origin_max_runs: usize::MAX,
            origin_gap: LazyOriginGap::Auto,
            dense_to_eager: DenseToEager::Origin,
            far_window: DEFAULT_LAZY_FAR_WINDOW,
            far_inflight: DEFAULT_LAZY_FAR_INFLIGHT,
            partial_publish: false,
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
                "auto" => LazyPromotion::Auto,
                "off" => LazyPromotion::Off,
                other => other
                    .strip_prefix("bg:")
                    .and_then(|reads| reads.parse::<u32>().ok())
                    .filter(|reads| *reads > 0)
                    .map(|reads| LazyPromotion::Background { reads })
                    .ok_or_else(|| {
                        invalid(LAZY_PROMOTE_ENV, &value, "auto, off or bg:N with N >= 1")
                    })?,
            };
        }
        config.promote_inflight = count(LAZY_PROMOTE_INFLIGHT_ENV, config.promote_inflight, 1)?;
        config.inline_rows = count(LAZY_INLINE_ROWS_ENV, config.inline_rows, 0)?;
        if let Some(value) = lookup(LAZY_EAGER_BEFORE_FULL_ENV) {
            config.eager_before_full = parse_flag(LAZY_EAGER_BEFORE_FULL_ENV, &value)?;
        }
        config.origin_max_runs = count(LAZY_ORIGIN_MAX_RUNS_ENV, config.origin_max_runs, 0)?;
        if let Some(value) = lookup(LAZY_ORIGIN_GAP_BYTES_ENV) {
            config.origin_gap = match value.trim() {
                "auto" => LazyOriginGap::Auto,
                bytes => bytes
                    .parse::<u64>()
                    .map(LazyOriginGap::Bytes)
                    .map_err(|_| {
                        invalid(LAZY_ORIGIN_GAP_BYTES_ENV, &value, "auto or a byte count")
                    })?,
            };
        }
        if let Some(value) = lookup(LAZY_DENSE_TO_EAGER_ENV) {
            // `0`, `1` and their `false`/`true` spellings are the values of
            // the former on/off flag.
            config.dense_to_eager = match value.trim() {
                "off" | "0" | "false" => DenseToEager::Off,
                "origin" => DenseToEager::Origin,
                "all" | "1" | "true" => DenseToEager::All,
                _ => {
                    return Err(invalid(
                        LAZY_DENSE_TO_EAGER_ENV,
                        &value,
                        "off, origin or all",
                    ));
                }
            };
        }
        config.far_window = count(LAZY_FAR_WINDOW_ENV, config.far_window, 0)?;
        if let Some(value) = lookup(LAZY_FAR_INFLIGHT_ENV) {
            // A permit pool holds at most `Semaphore::MAX_PERMITS`.
            config.far_inflight = value
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|permits| (1..=Semaphore::MAX_PERMITS).contains(permits))
                .ok_or_else(|| {
                    invalid(
                        LAZY_FAR_INFLIGHT_ENV,
                        &value,
                        &format!("an integer from 1 to {}", Semaphore::MAX_PERMITS),
                    )
                })?;
        }
        if let Some(value) = lookup(LAZY_PARTIAL_PUBLISH_ENV) {
            config.partial_publish = match value.trim() {
                "on" => true,
                "off" => false,
                _ => return Err(invalid(LAZY_PARTIAL_PUBLISH_ENV, &value, "on or off")),
            };
        }
        Ok(config)
    }

    /// The staleness window of the gathers of probes that read an ex plane
    /// from an origin of `class`, [`Self::far_window`] when it applies, or
    /// `None` when they keep [`Self::window`]: the origin is not
    /// [`OriginLatencyClass::High`], or the far window is no wider.
    pub fn active_far_window(&self, class: OriginLatencyClass) -> Option<usize> {
        (class == OriginLatencyClass::High && self.far_window > self.window)
            .then_some(self.far_window)
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

/// The file rows of the partition rows `range`: all of them, or the sorted
/// partition offsets `rows`.
fn partition_file_rows(range: &std::ops::Range<usize>, rows: Option<&[u32]>) -> UInt64Array {
    let start = range.start as u64;
    match rows {
        Some(rows) => UInt64Array::from_iter_values(rows.iter().map(|&row| start + u64::from(row))),
        None => UInt64Array::from_iter_values(start..range.end as u64),
    }
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

/// Planes of a layered partition that have a cache entry: sign, high, low
/// and [`SIGN_BOUNDS_PLANE`].
const ENTRY_PLANES: usize = SIGN_BOUNDS_PLANE as usize + 1;

/// The schemas a storage attaches code-only entries to, built once per
/// storage rather than on every read of an entry.
#[derive(Debug, Default)]
struct AttachSchemas {
    /// A plane's projection of the file, by bounds placement (lazy, eager)
    /// and plane.
    planes: [[OnceLock<SchemaRef>; ENTRY_PLANES]; 2],
    /// A native partition's columns, which a storage built from its
    /// code-only entry holds.
    partition: OnceLock<SchemaRef>,
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Q::Metadata,

    ivf: IvfModel,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    plane_access: PlaneAccessTracker,
    /// See [`Self::origin_latency`].
    origin_latency: OriginLatencyClass,
    /// See [`Self::sign_bounds`].
    sign_bounds: SignBounds,
    /// See [`Self::resident_columns`].
    resident_columns: ResidentColumns,
    /// See [`Self::resident_columns_enabled`].
    resident_columns_enabled: bool,
    /// What the index asks its cache entries to hold; see
    /// [`Self::entry_columns`].
    entry_columns: EntryColumns,
    /// The file's metadata with its sign codes marked packed, which
    /// storages built from code-only entries take, built on first use; `None`
    /// when the file stores them packed. See [`Self::entry_metadata`].
    packed_metadata: OnceLock<Option<Q::Metadata>>,
    /// The schemas code-only entries are attached to, built on first use:
    /// every plane's projection of the file, by bounds placement, and a
    /// native partition's; see [`Self::plane_schema`].
    attach_schemas: AttachSchemas,
    /// The file the storage reads, when bound to the runtime handles every
    /// open of it shares; see [`Self::with_index_file`]. `None` keeps the
    /// far gather pools to the plane access tracker.
    index_file: Option<IndexFileKey>,
    /// The file's plane-row layout, when its metadata declares one: built
    /// and validated when the storage opens the file, or on first use when it
    /// is reconstructed from a cache; see [`Self::plane_rows`].
    plane_rows: OnceLock<Arc<PlaneRowsSpec>>,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // The resident store is charged as its own entry of the index cache
        // (`ResidentColumnsEntry`), not with every storage reading it.
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
        let ivf_bytes = reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_bytes)?)?;

        let mut metadata: Vec<String> = serde_json::from_str(
            schema
                .metadata
                .get(STORAGE_METADATA_KEY)
                .ok_or(Error::index(format!("{} not found", STORAGE_METADATA_KEY)))?
                .as_str(),
        )?;
        debug_assert_eq!(metadata.len(), 1);
        // for now the metadata is the same for all partitions, so we just store one
        let metadata = metadata
            .pop()
            .ok_or(Error::index("metadata is empty".to_string()))?;
        let mut metadata: Q::Metadata = serde_json::from_str(&metadata)?;
        // we store large metadata (e.g. PQ codebook) in global buffer,
        // and the schema metadata just contains a pointer to the buffer
        if let Some(pos) = metadata.buffer_index() {
            let bytes = reader.read_global_buffer(pos).await?;
            metadata.parse_buffer(bytes)?;
        }
        // A file whose schema does not match its declared row layout is an
        // invalid index; it fails here rather than on a later read.
        let plane_rows = OnceLock::new();
        match metadata.row_layout() {
            RQRowLayout::Columns => {
                if let Some(name) = PACKED_COLUMNS
                    .iter()
                    .find(|name| schema.field(name).is_some())
                {
                    return Err(Error::index(format!(
                        "invalid IVF_RQ file: its metadata declares the column layout, but it has plane-row column {name}"
                    )));
                }
            }
            RQRowLayout::PlaneRows => {
                let spec = Self::plane_rows_spec(&metadata, distance_type, &reader)?;
                plane_rows.get_or_init(|| Arc::new(spec));
            }
        }

        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
            plane_access: Default::default(),
            origin_latency: OriginLatencyClass::default(),
            sign_bounds: SignBounds::default(),
            resident_columns: ResidentColumns::default(),
            resident_columns_enabled: false,
            entry_columns: EntryColumns::default(),
            packed_metadata: OnceLock::new(),
            attach_schemas: AttachSchemas::default(),
            index_file: None,
            plane_rows,
        })
    }

    /// The plane-row layout of `reader`'s file, an IVF_RQ storage with
    /// `metadata`, checked against the file's schema.
    fn plane_rows_spec(
        metadata: &Q::Metadata,
        distance_type: DistanceType,
        reader: &FileReader,
    ) -> Result<PlaneRowsSpec> {
        let Quantizer::Rabit(rq) = Q::from_metadata(metadata, distance_type)? else {
            return Err(Error::index(
                "invalid index file: only an IVF_RQ file can store plane rows",
            ));
        };
        let schema = arrow_schema::Schema::from(reader.schema().as_ref());
        PlaneRowsSpec::for_file(rq.metadata_ref(), &schema)
    }

    /// How the file stores each row's fields, from its metadata.
    pub fn row_layout(&self) -> RQRowLayout {
        self.metadata.row_layout()
    }

    /// The file's plane-row layout, `None` for the column layout. Reads of a
    /// plane-row file go through it: they read the packed columns that hold
    /// the fields they want and unpack them into those fields, so every read
    /// returns what it returns from a column-layout file.
    pub fn plane_rows(&self) -> Result<Option<&Arc<PlaneRowsSpec>>> {
        if self.row_layout() == RQRowLayout::Columns {
            return Ok(None);
        }
        if let Some(spec) = self.plane_rows.get() {
            return Ok(Some(spec));
        }
        let spec = Self::plane_rows_spec(&self.metadata, self.distance_type, &self.reader)?;
        Ok(Some(self.plane_rows.get_or_init(|| Arc::new(spec))))
    }

    /// The fields reads return: the file's schema, or on a plane-row file
    /// the column layout's fields its packed columns hold, then the columns
    /// it carries.
    pub fn logical_schema(&self) -> Result<SchemaRef> {
        Ok(match self.plane_rows()? {
            Some(spec) => spec.logical_schema().clone(),
            None => self.schema(),
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
            plane_access: Default::default(),
            origin_latency: OriginLatencyClass::default(),
            sign_bounds: SignBounds::default(),
            resident_columns: ResidentColumns::default(),
            resident_columns_enabled: false,
            entry_columns: EntryColumns::default(),
            packed_metadata: OnceLock::new(),
            attach_schemas: AttachSchemas::default(),
            index_file: None,
            plane_rows: OnceLock::new(),
        }
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

    /// Set the latency class of reads from this storage's file, resolved
    /// when the index opened.
    pub fn with_origin_latency(mut self, class: OriginLatencyClass) -> Self {
        self.origin_latency = class;
        self
    }

    /// The latency class of reads from this storage's file, see
    /// [`ORIGIN_LATENCY_ENV`]. [`OriginLatencyClass::Low`] unless the index
    /// set it with [`Self::with_origin_latency`].
    pub fn origin_latency(&self) -> OriginLatencyClass {
        self.origin_latency
    }

    /// Set where the index cache keeps a layered index's bounds columns,
    /// resolved when the index opened.
    pub fn with_sign_bounds(mut self, sign_bounds: SignBounds) -> Self {
        self.sign_bounds = sign_bounds;
        self
    }

    /// Where the index cache keeps a layered index's bounds columns, see
    /// [`SIGN_BOUNDS_ENV`]. [`SignBounds::Lazy`] unless the index set it with
    /// [`Self::with_sign_bounds`].
    pub fn sign_bounds(&self) -> SignBounds {
        self.sign_bounds
    }

    /// This storage's handle on the resident store of its file, which leases
    /// the store while the storage lives. It loads only while
    /// [`Self::resident_columns_enabled`], when the index opens
    /// ([`Self::load_resident_store`]).
    pub fn resident_columns(&self) -> &ResidentColumns {
        &self.resident_columns
    }

    /// Read the small columns through `store`, the handle an open binds to
    /// the store every live index of the file shares
    /// ([`ResidentColumns::in_index_cache`]), so that the store loads once.
    pub fn with_resident_columns(mut self, store: ResidentColumns) -> Self {
        self.resident_columns = store;
        self
    }

    /// Bind the lazy scan's far gathers to the permit pools that every open
    /// of index file `file` in the process shares, one per pool size, so that
    /// indexes of the file opened at once, a re-open and a state read back
    /// from a persistent cache tier bound their far gathers together. An
    /// index binds when it opens and when it is reconstructed.
    pub fn with_index_file(mut self, file: IndexFileKey) -> Self {
        self.index_file = Some(file);
        self
    }

    /// Set whether reads take the file's small columns from the resident
    /// store, resolved when the index opened. A plane-row file keeps no
    /// small columns apart, so it never reads through the store.
    pub fn with_resident_columns_enabled(mut self, enabled: bool) -> Self {
        self.resident_columns_enabled = enabled && self.row_layout() == RQRowLayout::Columns;
        self
    }

    /// Whether reads take the file's small columns from the resident store,
    /// see [`RESIDENT_COLUMNS_ENV`], and fetch only the others from the file.
    /// `false` unless the index set it with
    /// [`Self::with_resident_columns_enabled`].
    pub fn resident_columns_enabled(&self) -> bool {
        self.resident_columns_enabled
    }

    /// Load the resident store through this storage's handle, admitted to
    /// and leased in the index cache the handle is bound to, unless it is
    /// loaded already: an index loads its store when it opens, so that no
    /// read loads it. The load's I/O is added to `io_stats`. Nothing to load
    /// unless [`Self::resident_columns_enabled`], which a plane-row file
    /// never is.
    pub async fn load_resident_store(&self, io_stats: Option<&IoStats>) -> Result<()> {
        if !self.resident_columns_enabled {
            return Ok(());
        }
        self.resident_columns
            .get_or_load(&self.reader, io_stats, ResidentLoadTrigger::Open)
            .await
            .map(|_| ())
    }

    /// Bytes the resident store holds for this storage's file, loaded or
    /// not; see [`resident_columns_bytes`]. Zero for a plane-row file, whose
    /// small columns are packed with the codes of their plane.
    pub fn resident_columns_bytes(&self) -> u64 {
        if self.row_layout() == RQRowLayout::PlaneRows {
            return 0;
        }
        resident_columns_bytes(&self.schema(), self.num_rows())
    }

    /// The size of the resident store of this storage's file, loaded or
    /// not, and what an index cache charges for it; zero for a plane-row
    /// file. It reads nothing.
    pub fn resident_store_size(&self) -> Result<ResidentStoreSize> {
        let bytes = self.resident_columns_bytes();
        if bytes == 0 {
            return Ok(ResidentStoreSize::default());
        }
        Ok(ResidentStoreSize {
            bytes,
            charge: resident_store_charge(&self.reader)?,
        })
    }

    /// Ask the cache entries to hold `entry_columns`, resolved when the
    /// index opened ([`ENTRY_COLUMNS_ENV`]); see [`Self::entry_columns`].
    pub fn with_entry_columns(mut self, entry_columns: EntryColumns) -> Self {
        self.entry_columns = entry_columns;
        self
    }

    /// What the cache entries of this storage hold: [`EntryColumns::Codes`]
    /// when the index asked for it and reads take the small columns from
    /// the resident store ([`Self::resident_columns_enabled`]), which every
    /// read of a code-only entry needs; [`EntryColumns::All`] otherwise.
    pub fn entry_columns(&self) -> EntryColumns {
        self.entry_columns.resolve(self.resident_columns_enabled)
    }

    /// The cache key of plane `plane` of `partition` under this storage's
    /// bounds placement and entry columns.
    pub fn plane_key(&self, partition: usize, plane: u8) -> PlaneKey {
        PlaneKey {
            partition,
            plane,
            sign_bounds: self.sign_bounds,
            entry_columns: self.entry_columns(),
        }
    }

    /// The metadata a storage built from this storage's cache entries takes.
    /// Code-only entries keep their codes packed and blocked (see
    /// [`Self::normalize_entry`]), so a storage built from them takes the
    /// file's metadata with its sign codes marked packed and rewrites nothing;
    /// full entries keep the codes as the file stores them.
    fn entry_metadata(&self) -> Result<&Q::Metadata> {
        if self.entry_columns() == EntryColumns::All {
            return Ok(&self.metadata);
        }
        let packed = match self.packed_metadata.get() {
            Some(packed) => packed,
            None => {
                let packed = match self.quantizer()? {
                    Quantizer::Rabit(rq) if !rq.metadata_ref().packed => {
                        let quantizer = Q::try_from(Quantizer::Rabit(rq))?;
                        Some(quantizer.metadata(Some(QuantizationMetadata {
                            transposed: true,
                            ..Default::default()
                        })))
                    }
                    _ => None,
                };
                self.packed_metadata.get_or_init(|| packed)
            }
        };
        Ok(packed.as_ref().unwrap_or(&self.metadata))
    }

    /// `batch`, the columns of a code-only entry read from the file, with its
    /// codes as the entry keeps them: packed and blocked, once here rather
    /// than on every construction of a storage from the entry.
    fn normalize_entry(&self, batch: RecordBatch) -> Result<RecordBatch> {
        match self.quantizer()? {
            Quantizer::Rabit(rq) => normalize_entry_codes(batch, rq.metadata_ref()),
            _ => Ok(batch),
        }
    }

    /// The tier the index cache would serve plane `plane` of `partition`
    /// from, checked without reading the plane or counting as an access.
    /// [`CacheTier::Absent`] means a read goes to this storage's file, whose
    /// latency is [`Self::origin_latency`].
    pub async fn plane_tier(
        &self,
        partition: usize,
        plane: u8,
        cache: &WeakLanceCache,
    ) -> CacheTier {
        cache
            .peek_tier_with_key(&self.plane_key(partition, plane))
            .await
    }

    /// The permits of the lazy scan's gathers issued beyond the ordinary
    /// window ([`LAZY_FAR_WINDOW_ENV`]), `config.far_inflight` of them,
    /// shared by every query of this index and by the reconstructions of its
    /// cached state, and, once bound to its file ([`Self::with_index_file`]),
    /// by every open of the file in the process. The plane access tracker
    /// keeps the pool, so a cached state holds it. An index's config is
    /// fixed when it opens, so it keeps one pool; a config of another size
    /// (tests replace it) takes another pool, and gathers holding the old
    /// one's permits return them there.
    pub fn lazy_far_permits(&self, config: &LayeredLazyConfig) -> Result<LazyFarPermits> {
        let size = config.far_inflight;
        let mut pool = self
            .plane_access
            .lazy
            .far_permits
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match pool.as_ref() {
            Some(permits) if permits.size() == size => Ok(permits.clone()),
            _ => {
                let permits = match &self.index_file {
                    Some(file) => LazyFarPermits::for_file(file, size)?,
                    None => LazyFarPermits::try_new(size)?,
                };
                Ok(pool.insert(permits).clone())
            }
        }
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

    /// The file's schema: on a plane-row file, its packed columns; reads
    /// return [`Self::logical_schema`]'s fields.
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
    ///
    /// With [`Self::resident_columns_enabled`], only the code and bounds
    /// columns are read from the file. A plane-row file's partition is read
    /// as its packed columns and unpacked.
    pub async fn load_partition(
        &self,
        part_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        let batch = self
            .read_partition_batch(part_id, io_stats.as_ref())
            .await?;
        Q::Storage::try_from_batch_with_remapper(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }

    /// Every row of partition `part_id` as [`Self::logical_schema`]'s
    /// columns, read through a reader that also records into `io_stats`:
    /// what [`Self::load_partition`] builds the partition's storage from.
    pub async fn read_partition_batch(
        &self,
        part_id: usize,
        io_stats: Option<&IoStats>,
    ) -> Result<RecordBatch> {
        let range = self.ivf.row_range(part_id);
        let schema = self.logical_schema()?;
        Ok(if range.is_empty() {
            RecordBatch::new_empty(schema)
        } else if let Some(spec) = self.plane_rows()? {
            let batches = self
                .file_reader(io_stats, None)
                .read_stream(
                    ReadBatchParams::Range(range),
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            spec.unpack(&batches, &schema)?
        } else if let Some(store) = self.resident_store(&schema, io_stats).await? {
            let params = ReadBatchParams::Range(range.clone());
            let reader = self.file_reader(io_stats, None);
            self.read_with_resident_columns(store, schema, range, None, params, &reader)
                .await?
        } else {
            let reader = self.file_reader(io_stats, None);
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
            if Q::quantization_type() == QuantizationType::Rabit {
                // In buffers of exactly its values, as a plane-row read
                // unpacks them, so the partition entries of both layouts
                // weigh the same.
                exact_batch(&schema, &batches, false)?
            } else {
                concat_batches(&schema, batches.iter())?
            }
        })
    }

    /// The file columns a native partition's code-only entry
    /// ([`PartitionCodes`]) holds, in the file's order: those the resident
    /// store does not keep, the codes. A plane-row file names the columns of
    /// the column layout, which its reads unpack.
    fn partition_code_columns(&self) -> Result<Vec<String>> {
        Ok(self
            .logical_schema()?
            .fields()
            .iter()
            .filter(|field| !is_resident(field))
            .map(|field| field.name().clone())
            .collect())
    }

    /// [`Self::partition_code_columns`] as a projection of a column-layout
    /// file.
    pub fn partition_codes_projection(&self) -> Result<ReaderProjection> {
        let columns = self.partition_code_columns()?;
        let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
        lance_file::versions::reader_projection_from_column_names(
            self.reader.metadata().version(),
            self.reader.schema(),
            &columns,
        )
    }

    /// Read the code-only entry of native partition `part_id` from the file,
    /// through a reader that also records into `io_stats`: its code columns
    /// alone, which an object store serves in one request each. A plane-row
    /// file, which caches whole partitions, gives the same columns cut from
    /// its unpacked partition.
    pub async fn read_partition_codes(
        &self,
        part_id: usize,
        io_stats: Option<&IoStats>,
    ) -> Result<PartitionCodes> {
        let columns = self.partition_code_columns()?;
        let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
        let batch = if self.plane_rows()?.is_some() {
            // The code columns are packed with the partition's others.
            let partition = self.read_partition_batch(part_id, io_stats).await?;
            project_columns(&partition, &columns)?
        } else {
            self.read_file_columns(self.ivf.row_range(part_id), &columns, io_stats)
                .await?
        };
        Ok(PartitionCodes(self.normalize_entry(batch)?))
    }

    /// The code-only entry of a native partition from `batches`, its rows of
    /// the [`Self::partition_codes_projection`] columns that a prewarm read
    /// of several partitions decoded: copied into buffers of its own, so the
    /// entry does not keep the whole read.
    pub fn partition_codes_from_batches(
        &self,
        batches: Vec<RecordBatch>,
    ) -> Result<PartitionCodes> {
        Ok(PartitionCodes(
            self.normalize_entry(compact_prewarm_batches(batches)?)?,
        ))
    }

    /// The columns a storage built from a native partition's code-only
    /// entry of `entry_schema` holds: the file's in its order, then the
    /// entry's columns the file lacks. Ex codes the file stores sequentially
    /// are kept blocked, where a storage built from the file lays them out
    /// (see `load_blocked_ex_codes`).
    fn partition_schema(&self, entry_schema: &arrow_schema::Schema) -> SchemaRef {
        let file_schema = self.schema();
        let fields: Vec<_> = file_schema
            .fields()
            .iter()
            .filter(|field| {
                is_resident(field) || entry_schema.field_with_name(field.name()).is_ok()
            })
            .chain(
                entry_schema
                    .fields()
                    .iter()
                    .filter(|field| file_schema.field_with_name(field.name()).is_err()),
            )
            .cloned()
            .collect();
        Arc::new(arrow_schema::Schema::new_with_metadata(
            fields,
            file_schema.metadata().clone(),
        ))
    }

    /// The storage of native partition `part_id` built from `codes`, its
    /// code-only entry: the columns the resident store keeps are copies of
    /// its rows, the others the entry's, in the order a storage built from
    /// a read of the file holds them, so the storage is that one, bit for
    /// bit and byte for byte. The index loaded the store when it opened; a
    /// storage built outside an index open loads it on its first build, as a
    /// fallback, and adds the load's I/O to `load_stats`.
    pub async fn partition_from_codes(
        &self,
        part_id: usize,
        codes: &PartitionCodes,
        load_stats: Option<&IoStats>,
    ) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let entry = &codes.0;
        if entry.num_rows() != range.len() {
            return Err(Error::internal(format!(
                "code-only entry of partition {part_id} holds {} rows of its {}",
                entry.num_rows(),
                range.len()
            )));
        }
        // The entries of a file all hold the same columns, so the schema
        // built for the first one serves every other.
        let schema = self
            .attach_schemas
            .partition
            .get_or_init(|| self.partition_schema(&entry.schema()))
            .clone();
        let batch = if range.is_empty() {
            // As a read of no rows, which loads no store.
            RecordBatch::new_empty(schema)
        } else {
            let store = self
                .resident_store(&schema, load_stats)
                .await?
                .ok_or_else(|| {
                    Error::internal(format!(
                        "the code-only entry of partition {part_id} needs the resident store"
                    ))
                })?;
            store.attach(schema, Some(entry), &partition_file_rows(&range, None))?
        };
        Q::Storage::try_from_batch_with_remapper(
            batch,
            self.entry_metadata()?,
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }

    /// The storage of native partition `part_id` read through its code-only
    /// entry in `cache` ([`PartitionCodesKey`]), loaded from the file on a
    /// miss and admitted when `write_cache`, and whether the entry was a hit.
    /// Every read attaches the resident columns ([`Self::partition_from_codes`]),
    /// which the index loaded when it opened, so a hit reads nothing from the
    /// file.
    pub async fn load_partition_cached(
        &self,
        part_id: usize,
        cache: &WeakLanceCache,
        write_cache: bool,
        io_stats: Option<IoStats>,
    ) -> Result<(Q::Storage, bool)> {
        let key = PartitionCodesKey { partition: part_id };
        let (codes, hit) = if write_cache {
            cache
                .get_or_insert_with_key_hit(key, || async {
                    self.read_partition_codes(part_id, io_stats.as_ref()).await
                })
                .await?
        } else if let Some(codes) = cache.get_with_key(&key).await {
            (codes, true)
        } else {
            let codes = self
                .read_partition_codes(part_id, io_stats.as_ref())
                .await?;
            (Arc::new(codes), false)
        };
        let storage = self
            .partition_from_codes(part_id, &codes, io_stats.as_ref())
            .await?;
        Ok((storage, hit))
    }

    /// Warm every plane entry a full-precision scan reads. Backend admission
    /// enforces its byte budget.
    ///
    /// A backend that gates lower planes on their sign plane is warmed plane
    /// by plane: all sign planes first, then the other planes of partitions
    /// whose sign plane stayed resident. Other backends admit plane entries
    /// like any entry, so they are warmed partition by partition, which loads
    /// (and persists) every plane whatever fits in RAM, as the native
    /// partition prewarm does.
    pub async fn prewarm_planes(&self, cache: &WeakLanceCache) -> Result<()> {
        use super::bq::layered::RQPrecision;
        let planes: Vec<u8> = std::iter::once(0)
            .chain(self.planes_after_sign(RQPrecision::Full)?)
            .collect();
        if !cache.plane_admission_gated() {
            for part_id in 0..self.num_partitions() {
                for &plane in &planes {
                    cache
                        .get_or_insert_with_key(self.plane_key(part_id, plane), || async {
                            Ok(PlaneBatch(
                                self.read_plane_entry(part_id, plane, None).await?,
                            ))
                        })
                        .await?;
                }
            }
            return Ok(());
        }
        for &plane in &planes {
            for part_id in 0..self.num_partitions() {
                if plane > 0
                    && cache
                        .get_resident_with_key(&self.plane_key(part_id, 0))
                        .await
                        .is_none()
                {
                    continue;
                }
                cache
                    .get_or_insert_with_key(self.plane_key(part_id, plane), || async {
                        Ok(PlaneBatch(
                            self.read_plane_entry(part_id, plane, None).await?,
                        ))
                    })
                    .await?;
            }
        }
        Ok(())
    }

    /// Whether this loader owns an opt-in layered RaBitQ index.
    pub fn is_layered_rq(&self) -> bool {
        matches!(self.quantizer(), Ok(Quantizer::Rabit(rq)) if rq.metadata_ref().layered)
    }

    /// The planes a layered partition load at `precision` reads besides the
    /// sign plane, in the order the eager bounds placement assembles its
    /// columns: the bounds plane when the scan prunes with the bounds and the
    /// sign plane leaves them out, then the ex planes the precision scores.
    /// High precision prunes with the high bounds; full precision prunes
    /// with the error factors, or with the full bounds on a file without them.
    fn planes_after_sign(&self, precision: super::bq::layered::RQPrecision) -> Result<Vec<u8>> {
        use super::bq::layered::RQPrecision;
        let has_error_factors = match self.plane_rows()? {
            Some(spec) => spec
                .logical_schema()
                .field_with_name(ERROR_FACTORS_COLUMN)
                .is_ok(),
            None => self.reader.schema().field(ERROR_FACTORS_COLUMN).is_some(),
        };
        let (reads_bounds, last_ex_plane) = match precision {
            RQPrecision::Sign => (false, 0),
            RQPrecision::High => (true, 1),
            RQPrecision::Full => (!has_error_factors, 2),
        };
        let mut planes = Vec::with_capacity(usize::from(last_ex_plane) + 1);
        if reads_bounds && self.sign_bounds == SignBounds::Lazy {
            planes.push(SIGN_BOUNDS_PLANE);
        }
        planes.extend(1..=last_ex_plane);
        Ok(planes)
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
        let after_sign = self.planes_after_sign(precision)?;
        let load_plane = |plane| self.load_plane(part_id, plane, cache, io_stats.clone());
        let planes = if sequential_plane_loads()? || cache.plane_admission_gated() {
            // Admit the sign dependency first. The other reads can then
            // overlap without changing admission policy or assembled column order.
            let sign = load_plane(0).await?;
            let rest =
                futures::future::try_join_all(after_sign.iter().copied().map(load_plane)).await?;
            std::iter::once(sign).chain(rest).collect()
        } else {
            // Admission does not depend on the sign plane's residency, so a
            // missed partition reads its planes in one round trip.
            // `try_join_all` returns them in the order requested.
            futures::future::try_join_all(std::iter::once(0).chain(after_sign).map(load_plane))
                .await?
        };
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for batch in planes {
            fields.extend(batch.schema().fields().iter().cloned());
            columns.extend(batch.columns().iter().cloned());
        }
        let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns)?;
        Q::Storage::try_from_batch_at_precision(
            batch,
            self.entry_metadata()?,
            self.distance_type,
            self.frag_reuse_index.clone(),
            precision,
        )
    }

    /// Remappers can remove physical rows, so candidate offsets require an identity mapping.
    pub fn supports_candidate_reads(&self) -> bool {
        self.frag_reuse_index.is_none()
    }

    /// Load one whole plane entry of a layered partition, which holds the
    /// plane or, under [`EntryColumns::Codes`], its file columns alone (see
    /// [`Self::read_plane_entry`]); [`Self::load_plane`] attaches the
    /// resident columns.
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
        let key = self.plane_key(part_id, plane);
        if !cache.plane_admission_gated() {
            return cache
                .get_or_insert_with_key(key, || async {
                    Ok(PlaneBatch(
                        self.read_plane_entry(part_id, plane, io_stats.clone())
                            .await?,
                    ))
                })
                .await;
        }
        let sign_resident = plane == 0
            || cache
                .get_resident_with_key(&self.plane_key(part_id, 0))
                .await
                .is_some();
        let batch = if sign_resident {
            cache
                .get_or_insert_with_key_hit(key, || async {
                    Ok(PlaneBatch(
                        self.read_plane_entry(part_id, plane, io_stats.clone())
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
                self.read_plane_entry(part_id, plane, io_stats.clone())
                    .await?,
            ))
        };
        Ok(batch)
    }

    /// One whole plane of a layered partition, read through the cache: its
    /// entry ([`Self::load_plane_entry`]) with the resident columns attached
    /// ([`Self::attach_resident`]), so the batch is the plane a read of the
    /// file returns whatever the entry holds.
    pub async fn load_plane(
        &self,
        part_id: usize,
        plane: u8,
        cache: &WeakLanceCache,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        let entry = self
            .load_plane_entry(part_id, plane, cache, io_stats.clone())
            .await?;
        self.attach_resident(part_id, plane, &entry.0, None, io_stats.as_ref())
            .await
    }

    /// What the cache entry of plane `plane` of `part_id` holds, read from
    /// the file: the whole plane ([`Self::read_plane`]) or, under
    /// [`EntryColumns::Codes`], only its file columns
    /// ([`plane_entry_columns`]), with the sign codes packed. Every loader
    /// of a plane entry reads it here.
    pub async fn read_plane_entry(
        &self,
        part_id: usize,
        plane: u8,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        self.read_plane_entry_with(part_id, plane, self.entry_columns(), io_stats)
            .await
    }

    /// What a cache entry of plane `plane` of `part_id` holding
    /// `entry_columns` holds, read from the file whatever this storage's own
    /// entries hold ([`Self::entry_columns`]); see [`Self::read_plane_entry`].
    /// A plane-row file, which caches whole planes, gives the same columns
    /// cut from its unpacked plane.
    pub async fn read_plane_entry_with(
        &self,
        part_id: usize,
        plane: u8,
        entry_columns: EntryColumns,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        if entry_columns == EntryColumns::All {
            return self.read_plane(part_id, plane, None, io_stats).await;
        }
        let columns = plane_entry_columns(plane, self.sign_bounds, entry_columns);
        if columns.is_empty() {
            return Err(Error::invalid_input(format!(
                "a layered partition has no plane {plane} with {} bounds",
                self.sign_bounds
            )));
        }
        let batch = if self.plane_rows()?.is_some() {
            // The entry's columns are packed with the plane's others.
            let plane = self.read_plane(part_id, plane, None, io_stats).await?;
            project_columns(&plane, &columns)?
        } else {
            let range = self.ivf.row_range(part_id);
            self.read_file_columns(range, &columns, io_stats.as_ref())
                .await?
        };
        self.normalize_entry(batch)
    }

    /// Plane `plane`'s projection of the file under this storage's bounds
    /// placement, as a read of the plane returns it.
    fn plane_schema(&self, plane: u8) -> Result<SchemaRef> {
        let build = || -> Result<SchemaRef> {
            let projection = lance_file::versions::reader_projection_from_column_names(
                self.reader.metadata().version(),
                self.reader.schema(),
                plane_columns(plane, self.sign_bounds),
            )?;
            Ok(Arc::new(arrow_schema::Schema::from(
                projection.schema.as_ref(),
            )))
        };
        let placement = usize::from(self.sign_bounds == SignBounds::Eager);
        let Some(slot) = self.attach_schemas.planes[placement].get(usize::from(plane)) else {
            return build();
        };
        if let Some(schema) = slot.get() {
            return Ok(schema.clone());
        }
        let schema = build()?;
        Ok(slot.get_or_init(|| schema).clone())
    }

    /// Plane `plane` of partition `part_id` at the sorted partition offsets
    /// `rows` (every row when `None`), assembled from `entry`, a cache entry
    /// of the plane at those rows. A full entry is the plane. A code-only
    /// entry ([`EntryColumns::Codes`]) gets copies of the resident store's
    /// rows at the rows' file offsets, in the plane's column order, so the
    /// batch is the one, bit for bit and byte for byte, that a read of the
    /// file returns (see [`Self::read_plane`]). The index loaded the store
    /// when it opened; a storage built outside an index open loads it on its
    /// first attach, as a fallback, and adds the load's I/O to `load_stats`.
    pub async fn attach_resident(
        &self,
        part_id: usize,
        plane: u8,
        entry: &RecordBatch,
        rows: Option<&[u32]>,
        load_stats: Option<&IoStats>,
    ) -> Result<RecordBatch> {
        if self.entry_columns() == EntryColumns::All {
            return Ok(entry.clone());
        }
        let range = self.ivf.row_range(part_id);
        let expected_rows = rows.map_or(range.len(), <[u32]>::len);
        if entry.num_rows() != expected_rows {
            return Err(Error::internal(format!(
                "code-only entry of partition {part_id} plane {plane} holds {} rows, the read needs {expected_rows}",
                entry.num_rows()
            )));
        }
        let schema = self.plane_schema(plane)?;
        if expected_rows == 0 {
            // As a read of no rows, which loads no store.
            return Ok(RecordBatch::new_empty(schema));
        }
        let Some(store) = self.resident_store(&schema, load_stats).await? else {
            // A plane without resident columns, the bounds plane, is its
            // entry.
            return Ok(entry.clone());
        };
        store
            .attach(schema, Some(entry), &partition_file_rows(&range, rows))
            .map_err(|error| {
                Error::internal(format!(
                    "attaching the resident columns of partition {part_id} plane {plane}: {error}"
                ))
            })
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
        let sign = self.load_plane(part_id, 0, cache, io_stats).await?;
        Q::Storage::try_from_sign_plane_for_full(sign, self.entry_metadata()?, self.distance_type)
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
        let key = self.plane_key(part_id, plane);
        if source == PlaneSource::Resident
            && let Some(entry) = cache.get_resident_with_key(&key).await
        {
            let batch = self
                .attach_resident(part_id, plane, &entry.0, None, io_stats.as_ref())
                .await?;
            stats.fetch_resident_ns.add_elapsed(started);
            return Ok(FetchedPlane {
                batch,
                source: PlaneSource::Resident,
                from_origin: false,
                promotion: None,
            });
        }
        if source == PlaneSource::Sparse {
            let persisted = cache.get_rows_with_key(&key, rows).await;
            if persisted.is_some() || !origin_whole || cache.plane_admission_gated() {
                let (batch, from_origin) = match persisted {
                    Some(entry) => (
                        self.attach_resident(
                            part_id,
                            plane,
                            &entry.0,
                            Some(rows),
                            io_stats.as_ref(),
                        )
                        .await?,
                        false,
                    ),
                    // An origin read returns every column of the plane.
                    None => (
                        self.read_origin_rows(part_id, plane, rows, config, io_stats.as_ref())
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
        let batch = self.load_plane(part_id, plane, cache, io_stats).await?;
        stats.fetch_whole_ns.add_elapsed(started);
        Ok(FetchedPlane {
            batch,
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
        if rows.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::invalid_input(
                "candidate offsets must be sorted and unique",
            ));
        }
        let indices = arrow_array::UInt32Array::from(rows.clone());
        let code_only = self.entry_columns() == EntryColumns::Codes;
        // The planes are independent. Overlap their cache/origin reads,
        // while preserving plane order when assembling the full row schema.
        let planes = std::iter::once(0).chain(self.planes_after_sign(RQPrecision::Full)?);
        let batches = futures::future::try_join_all(planes.map(|plane| {
            let rows = &rows;
            let indices = &indices;
            let io_stats = io_stats.clone();
            async move {
                let key = self.plane_key(part_id, plane);
                let resident = cache.get_resident_with_key(&key).await;
                let was_resident = resident.is_some();
                // Sign codes and bounds keep a raw body, which has no row
                // directory, so their persisted entries are read whole.
                let row_addressable = plane != 0 && plane != SIGN_BOUNDS_PLANE;
                let cached = if !row_addressable && resident.is_none() {
                    cache.get_without_promotion_with_key(&key).await
                } else {
                    resident
                };
                let selected_cached = if row_addressable && cached.is_none() {
                    cache.get_rows_with_key(&key, rows).await
                } else {
                    None
                };
                // An entry's rows, attached to the resident columns when the
                // entry holds its codes alone.
                let attach = |entry: RecordBatch| {
                    let io_stats = io_stats.clone();
                    async move {
                        self.attach_resident(
                            part_id,
                            plane,
                            &entry,
                            Some(rows.as_slice()),
                            io_stats.as_ref(),
                        )
                        .await
                    }
                };
                let batch = if plane == 0 {
                    // Sign codes are transposed across rows. Gather their physical
                    // offsets directly, then pack only the selected rows.
                    let raw = if let Some(value) = cached {
                        value.0.clone()
                    } else {
                        self.read_plane_entry(part_id, plane, io_stats.clone())
                            .await?
                    };
                    let codes = raw
                        .column_by_name(RABIT_CODE_COLUMN)
                        .ok_or_else(|| Error::invalid_input("missing sign codes"))?;
                    let selected_codes = take_packed_codes(codes.as_fixed_size_list(), rows)?;
                    if code_only {
                        // The entry's other columns, the bounds, at the rows.
                        let columns = raw
                            .schema()
                            .fields()
                            .iter()
                            .zip(raw.columns())
                            .map(|(field, column)| {
                                if field.name() == RABIT_CODE_COLUMN {
                                    Ok(Arc::new(selected_codes.clone()) as ArrayRef)
                                } else {
                                    Ok(arrow_select::take::take(column, indices, None)?)
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;
                        attach(RecordBatch::try_new(raw.schema(), columns)?).await?
                    } else {
                        // The transposed sign column cannot be gathered with Arrow
                        // take. Avoid copying it only to replace that copy immediately.
                        let field = raw.schema().field_with_name(RABIT_CODE_COLUMN)?.clone();
                        raw.drop_column(RABIT_CODE_COLUMN)?
                            .take(indices)?
                            .try_with_column(field, Arc::new(selected_codes))?
                    }
                } else if let Some(value) = cached {
                    attach(value.0.take(indices)?).await?
                } else if let Some(value) = selected_cached {
                    attach(value.0.clone()).await?
                } else {
                    // An origin read returns every column of the plane.
                    self.read_plane(part_id, plane, Some(rows.clone()), io_stats.clone())
                        .await?
                };
                if plane > 0
                    && !was_resident
                    && cache
                        .get_resident_with_key(&self.plane_key(part_id, 0))
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
                            .get_or_insert_with_key(self.plane_key(part_id, plane), || async {
                                Ok(PlaneBatch(
                                    self.read_plane_entry(part_id, plane, io_stats.clone())
                                        .await?,
                                ))
                            })
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
            self.entry_metadata()?,
            self.distance_type,
            self.frag_reuse_index.clone(),
            RQPrecision::Full,
        )
    }

    /// Read only selected rows of a plane; sorted indices are coalesced by the reader.
    /// With [`Self::resident_columns_enabled`], only the plane's code (or
    /// bounds) column is read from the file.
    pub async fn read_plane(
        &self,
        part_id: usize,
        plane: u8,
        rows: Option<Vec<u32>>,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        self.read_plane_with_coalesce_gap(part_id, plane, rows, io_stats, None)
            .await
    }

    /// [`Self::read_plane`], reading the file through
    /// [`FileReader::with_coalesce_gap`] when `coalesce_gap` is set: the
    /// ranges of the selected rows that are at most that many bytes apart
    /// share one request, instead of those within the object store's block
    /// size. The batch is the same for every gap; only the requests and the
    /// bytes read differ.
    pub async fn read_plane_with_coalesce_gap(
        &self,
        part_id: usize,
        plane: u8,
        rows: Option<Vec<u32>>,
        io_stats: Option<IoStats>,
        coalesce_gap: Option<u64>,
    ) -> Result<RecordBatch> {
        self.read_plane_with_stats(
            part_id,
            plane,
            rows,
            coalesce_gap,
            io_stats.as_ref(),
            io_stats.as_ref(),
        )
        .await
    }

    /// Read the sorted partition offsets `rows` of ex plane `plane` from the
    /// origin file for a sparse gather, merging row runs within `config`'s
    /// origin gap for the index's origin, and count the read's requests and
    /// bytes after coalescing and the rows it gathers. A first-use load of the
    /// resident store that the read starts is not counted; its I/O, like the
    /// read's, is added to `io_stats`.
    async fn read_origin_rows(
        &self,
        part_id: usize,
        plane: u8,
        rows: &[u32],
        config: &LayeredLazyConfig,
        io_stats: Option<&IoStats>,
    ) -> Result<RecordBatch> {
        let origin = IoStats::new();
        let batch = self
            .read_plane_with_stats(
                part_id,
                plane,
                Some(rows.to_vec()),
                config.origin_gap.coalesce_gap(self.origin_latency),
                Some(&origin),
                io_stats,
            )
            .await;
        let read = origin.snapshot();
        let stats = layered_stats::counters();
        stats.origin_sparse_requests.add(read.iops);
        stats.origin_sparse_bytes.add(read.bytes_read);
        stats.origin_sparse_rows.add(rows.len() as u64);
        if let Some(io_stats) = io_stats {
            io_stats.add_scan_stats(&read);
        }
        batch
    }

    /// [`Self::read_plane_with_coalesce_gap`], recording the plane's reads
    /// from the file into `read_stats` and a first-use load of the resident
    /// store into `load_stats`, so that a caller can count them apart.
    async fn read_plane_with_stats(
        &self,
        part_id: usize,
        plane: u8,
        rows: Option<Vec<u32>>,
        coalesce_gap: Option<u64>,
        read_stats: Option<&IoStats>,
        load_stats: Option<&IoStats>,
    ) -> Result<RecordBatch> {
        let columns = super::bq::layered::plane_columns(plane, self.sign_bounds);
        if columns.is_empty() {
            return Err(Error::invalid_input(format!(
                "a layered partition has no plane {plane} with {} bounds",
                self.sign_bounds
            )));
        }
        let (projection, schema) = self.read_projection(columns)?;
        let range = self.ivf.row_range(part_id);
        if range.is_empty() || rows.as_ref().is_some_and(Vec::is_empty) {
            return Ok(RecordBatch::new_empty(schema));
        }
        let params = if let Some(rows) = &rows {
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::new();
            for &row in rows {
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
            ReadBatchParams::Range(range.clone())
        };
        let reader = self.file_reader(read_stats, coalesce_gap);
        if let Some(store) = self.resident_store(&schema, load_stats).await? {
            return self
                .read_with_resident_columns(store, schema, range, rows.as_deref(), params, &reader)
                .await;
        }
        self.read_projected(&reader, params, projection, &schema, rows.is_none())
            .await
    }

    /// How a read of the file's `columns` reads them, and the schema of the
    /// batch it returns: the columns themselves, or on a plane-row file the
    /// packed columns that hold them, which the read unpacks into them.
    fn read_projection(&self, columns: &[&str]) -> Result<(ReaderProjection, SchemaRef)> {
        let plane_rows = self.plane_rows()?;
        let file_columns = match plane_rows {
            Some(spec) => spec.packed_for(columns)?,
            None => columns.to_vec(),
        };
        let projection = lance_file::versions::reader_projection_from_column_names(
            self.reader.metadata().version(),
            self.reader.schema(),
            &file_columns,
        )?;
        let schema = match plane_rows {
            Some(spec) => spec.projection(columns)?,
            None => Arc::new(arrow_schema::Schema::from(projection.schema.as_ref())),
        };
        Ok((projection, schema))
    }

    /// Read `projection` ([`Self::read_projection`]) at `params` through
    /// `reader` into one batch of `schema`, unpacking plane rows. A read of
    /// every row of a partition (`whole`), which cache entries hold, returns
    /// buffers of exactly its values, as an unpacked plane-row read does; a
    /// read of selected rows keeps the decoder's buffers.
    async fn read_projected(
        &self,
        reader: &FileReader,
        params: ReadBatchParams,
        projection: ReaderProjection,
        schema: &SchemaRef,
        whole: bool,
    ) -> Result<RecordBatch> {
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
        match self.plane_rows()? {
            Some(spec) => spec.unpack(&batches, schema),
            None if whole => exact_batch(schema, &batches, false),
            None => concat_batches(schema, batches.iter()).map_err(Into::into),
        }
    }

    /// This storage's file reader, also recording its I/O into `io_stats`,
    /// and merging requested ranges at most `coalesce_gap` bytes apart
    /// instead of those within the object store's block size.
    fn file_reader(
        &self,
        io_stats: Option<&IoStats>,
        coalesce_gap: Option<u64>,
    ) -> Cow<'_, FileReader> {
        let reader = match io_stats {
            Some(stats) => Cow::Owned(self.reader.with_io_stats(stats.recorder())),
            None => Cow::Borrowed(&self.reader),
        };
        match coalesce_gap {
            Some(gap) => Cow::Owned(reader.with_coalesce_gap(gap)),
            None => reader,
        }
    }

    /// The resident store, when [`Self::resident_columns_enabled`] and it
    /// keeps some of `schema`'s columns. The index loaded it when it opened
    /// ([`Self::load_resident_store`]); a read loads it only as a fallback,
    /// for a storage built outside an index open, which counts
    /// `resident_columns_read_loads` and adds the load's I/O to that read's
    /// `io_stats`.
    async fn resident_store(
        &self,
        schema: &arrow_schema::Schema,
        io_stats: Option<&IoStats>,
    ) -> Result<Option<&ResidentColumnStore>> {
        let reads_resident = schema.fields().iter().any(|field| is_resident(field));
        if !self.resident_columns_enabled || !reads_resident {
            return Ok(None);
        }
        self.resident_columns
            .get_or_load(&self.reader, io_stats, ResidentLoadTrigger::Read)
            .await
            .map(Some)
    }

    /// Read `schema`'s columns at the file rows `range` of a partition,
    /// either every row or the sorted partition offsets `rows`, as `params`
    /// selects them. Columns `store` keeps are copies of its rows, and only
    /// the others are read from the file, through `reader`, so the batch is
    /// the one a read of every column from the file returns.
    async fn read_with_resident_columns(
        &self,
        store: &ResidentColumnStore,
        schema: SchemaRef,
        range: std::ops::Range<usize>,
        rows: Option<&[u32]>,
        params: ReadBatchParams,
        reader: &FileReader,
    ) -> Result<RecordBatch> {
        if range.end as u64 > store.num_rows() {
            return Err(Error::invalid_input(format!(
                "partition rows {range:?} exceed the {} rows of the storage file",
                store.num_rows()
            )));
        }
        let file_columns: Vec<&str> = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .filter(|name| store.column(name).is_none())
            .collect();
        let file_batch = if file_columns.is_empty() {
            None
        } else {
            let projection = lance_file::versions::reader_projection_from_column_names(
                self.reader.metadata().version(),
                self.reader.schema(),
                &file_columns,
            )?;
            let file_schema = Arc::new(arrow_schema::Schema::from(projection.schema.as_ref()));
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
            Some(if rows.is_none() {
                // A whole read, which entries hold: see `read_projected`.
                exact_batch(&file_schema, &batches, false)?
            } else {
                concat_batches(&file_schema, batches.iter())?
            })
        };
        store.attach(
            schema,
            file_batch.as_ref(),
            &partition_file_rows(&range, rows),
        )
    }

    /// `columns` of the file at every row of the partition rows `range`,
    /// read through a reader that also records into `io_stats`: what a
    /// code-only entry holds before its codes are normalized.
    async fn read_file_columns(
        &self,
        range: std::ops::Range<usize>,
        columns: &[&str],
        io_stats: Option<&IoStats>,
    ) -> Result<RecordBatch> {
        let (projection, schema) = self.read_projection(columns)?;
        if range.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        let reader = self.file_reader(io_stats, None);
        self.read_projected(
            &reader,
            ReadBatchParams::Range(range),
            projection,
            &schema,
            true,
        )
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
        let frag_reuse_index = self.frag_reuse_index.clone();
        let plane_rows = self.plane_rows()?.cloned();
        spawn_prewarm_materialization(move || {
            // Unpacking copies the rows into buffers of their own too.
            let batch = match plane_rows {
                Some(spec) => spec.unpack(&batches, spec.logical_schema())?,
                None => compact_prewarm_batches(batches)?,
            };
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
        DEFAULT_ENTRY_COLUMNS, DenseGatherMode, DenseToEager, ENTRY_COLUMNS_ENV, EntryColumns,
        HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES, IndexFileKey, LAZY_DENSE_BYTES_FRACTION_ENV,
        LAZY_DENSE_ENV, LAZY_DENSE_TO_EAGER_ENV, LAZY_EAGER_BEFORE_FULL_ENV, LAZY_FAR_INFLIGHT_ENV,
        LAZY_FAR_WINDOW_ENV, LAZY_FULL_ENV, LAZY_INLINE_ROWS_ENV, LAZY_MAX_RUNS_ENV,
        LAZY_ORIGIN_GAP_BYTES_ENV, LAZY_ORIGIN_MAX_RUNS_ENV, LAZY_PARTIAL_PUBLISH_ENV,
        LAZY_PROMOTE_ENV, LAZY_PROMOTE_INFLIGHT_ENV, LAZY_WINDOW_ENV, LayeredLazyConfig,
        LazyFarPermits, LazyOriginGap, LazyPromotion, ORIGIN_LATENCY_ENV, OriginLatencyClass,
        PlaneSource, QueryScratchCapacity, QueryScratchPool, RESIDENT_COLUMNS_ENV,
        RESIDENT_LIFETIME_ENV, ResidentColumnsSetting, ResidentLifetime, ResidentStoreSize,
        SEQUENTIAL_PLANE_LOADS_ENV, SIGN_BOUNDS_ENV, SignBounds, compact_prewarm_batches,
        entry_columns_from, entry_columns_setting, origin_latency_from, origin_latency_setting,
        origin_reads_whole_plane, plan_plane_gather, resident_columns_from,
        resident_columns_setting, resident_lifetime_from, resident_lifetime_setting,
        resident_store_fits, sequential_plane_loads, sequential_plane_loads_from, sign_bounds_from,
        sign_bounds_setting, spawn_prewarm_materialization,
    };
    use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
    use futures::FutureExt;
    use lance_core::Error;
    use lance_core::cache::CacheTier;
    use lance_core::deepsize::DeepSizeOf;
    use lance_io::object_store::ObjectStore;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

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
        // In a buffer of exactly its values, as a plane-row read unpacks
        // them, which the pieces of a read join into as well.
        let exact_bytes = 10 * size_of::<u64>();
        assert_eq!(compact_array.values().inner().capacity(), exact_bytes);
        let joined =
            compact_prewarm_batches(vec![parent.slice(10, 3), parent.slice(13, 7)]).unwrap();
        assert_eq!(joined, compact);
        assert_eq!(
            joined.column(0).to_data().buffers()[0].capacity(),
            exact_bytes
        );
        // An exact batch is still copied, so an entry never shares a read.
        let copied = compact_prewarm_batches(vec![compact.clone()]).unwrap();
        assert_eq!(copied, compact);
        assert_ne!(
            copied.column(0).to_data().buffers()[0].as_ptr(),
            compact_array.values().as_ptr() as *const u8
        );
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
            (LAZY_ORIGIN_GAP_BYTES_ENV, " 65536 "),
            (LAZY_DENSE_TO_EAGER_ENV, "0"),
            (LAZY_FAR_WINDOW_ENV, "8"),
            (LAZY_FAR_INFLIGHT_ENV, " 2 "),
            (LAZY_PARTIAL_PUBLISH_ENV, " on "),
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
                origin_gap: LazyOriginGap::Bytes(65536),
                dense_to_eager: DenseToEager::Off,
                far_window: 8,
                far_inflight: 2,
                partial_publish: true,
            }
        );
        assert_eq!(
            LayeredLazyConfig::from_lookup(|_| None).unwrap(),
            LayeredLazyConfig::default()
        );
        assert!(!LayeredLazyConfig::default().enabled);
        assert!(LayeredLazyConfig::default().eager_before_full);
        assert_eq!(LayeredLazyConfig::default().origin_max_runs, usize::MAX);
        assert_eq!(LayeredLazyConfig::default().promote, LazyPromotion::Off);
        assert_eq!(
            LayeredLazyConfig::default().dense_to_eager,
            DenseToEager::Origin
        );
        assert!(!LayeredLazyConfig::default().partial_publish);
        let env = HashMap::from([(LAZY_FULL_ENV, "1"), (LAZY_PARTIAL_PUBLISH_ENV, "off")]);
        assert_eq!(
            LayeredLazyConfig::from_lookup(|name| env.get(name).map(|value| value.to_string()))
                .unwrap(),
            LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            }
        );
        // The origin run cap does not depend on the eager-before-full switch,
        // and dense probes whose planes are on a slow origin go to the eager
        // scan unless set otherwise.
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
            (true, 2, DenseToEager::Origin)
        );
        let env = HashMap::from([(LAZY_FULL_ENV, "1"), (LAZY_DENSE_TO_EAGER_ENV, "false")]);
        let config =
            LayeredLazyConfig::from_lookup(|name| env.get(name).map(|value| value.to_string()))
                .unwrap();
        assert_eq!(
            config,
            LayeredLazyConfig {
                enabled: true,
                dense_to_eager: DenseToEager::Off,
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
            (LAZY_PROMOTE_ENV, "on"),
            (LAZY_PROMOTE_INFLIGHT_ENV, "0"),
            (LAZY_DENSE_BYTES_FRACTION_ENV, "-1"),
            (LAZY_WINDOW_ENV, "many"),
            (LAZY_EAGER_BEFORE_FULL_ENV, "on"),
            (LAZY_ORIGIN_MAX_RUNS_ENV, "-1"),
            (LAZY_ORIGIN_GAP_BYTES_ENV, "1MiB"),
            (LAZY_ORIGIN_GAP_BYTES_ENV, "-1"),
            (LAZY_DENSE_TO_EAGER_ENV, "2"),
            (LAZY_DENSE_TO_EAGER_ENV, "on"),
            (LAZY_FAR_WINDOW_ENV, "wide"),
            (LAZY_FAR_INFLIGHT_ENV, "0"),
            (LAZY_PARTIAL_PUBLISH_ENV, "1"),
            (LAZY_PARTIAL_PUBLISH_ENV, "partial"),
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
    fn test_lazy_origin_gap_knob() {
        let parse = |gap: &str| {
            LayeredLazyConfig::from_lookup(|key| match key {
                LAZY_FULL_ENV => Some("1".to_string()),
                LAZY_ORIGIN_GAP_BYTES_ENV => Some(gap.to_string()),
                _ => None,
            })
            .map(|config| config.origin_gap)
        };
        let defaults = LayeredLazyConfig::default();
        assert_eq!(defaults.origin_gap, LazyOriginGap::Auto);
        assert_eq!(parse(" auto ").unwrap(), LazyOriginGap::Auto);
        for bytes in [0, 64 * 1024, 256 * 1024, 1024 * 1024, u64::MAX] {
            assert_eq!(
                parse(&bytes.to_string()).unwrap(),
                LazyOriginGap::Bytes(bytes)
            );
        }
        for value in ["max", "AUTO", "64k", "256k", "-1", ""] {
            let error = parse(value).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(LAZY_ORIGIN_GAP_BYTES_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
        // `auto` widens the gap only where every request is a round trip and
        // keeps the object store's block size elsewhere; a byte count is the
        // gap whatever the origin.
        let block_size = 4096;
        let classes = [OriginLatencyClass::Low, OriginLatencyClass::High];
        assert_eq!(
            classes.map(|class| LazyOriginGap::Auto.coalesce_gap(class)),
            [None, Some(HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES)]
        );
        assert_eq!(
            classes.map(|class| LazyOriginGap::Auto.resolve(class, block_size)),
            [block_size, HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES]
        );
        for bytes in [0, 64 * 1024, 256 * 1024, 1024 * 1024, u64::MAX] {
            let gap = LazyOriginGap::Bytes(bytes);
            assert_eq!(
                classes.map(|class| gap.coalesce_gap(class)),
                [Some(bytes); 2]
            );
            assert_eq!(
                classes.map(|class| gap.resolve(class, block_size)),
                [bytes; 2]
            );
        }
    }

    #[test]
    fn test_lazy_promote_knob() {
        let parse = |promote: &str| {
            LayeredLazyConfig::from_lookup(|key| match key {
                LAZY_FULL_ENV => Some("1".to_string()),
                LAZY_PROMOTE_ENV => Some(promote.to_string()),
                _ => None,
            })
            .map(|config| config.promote)
        };
        assert_eq!(LayeredLazyConfig::default().promote, LazyPromotion::Off);
        for (value, expected) in [
            (" auto ", LazyPromotion::Auto),
            ("off", LazyPromotion::Off),
            ("bg:1", LazyPromotion::Background { reads: 1 }),
            ("bg:3", LazyPromotion::Background { reads: 3 }),
        ] {
            assert_eq!(parse(value).unwrap(), expected, "{value:?}");
        }
        for value in ["AUTO", "on", "bg:", "bg:0", "bg:-1", ""] {
            let error = parse(value).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(LAZY_PROMOTE_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
        // `auto` promotes after one sparse gather unless every origin read
        // is a round trip; an explicit policy applies whatever the origin.
        let classes = [OriginLatencyClass::Low, OriginLatencyClass::High];
        assert_eq!(
            classes.map(|class| LazyPromotion::Auto.resolve(class)),
            [LazyPromotion::Background { reads: 1 }, LazyPromotion::Off]
        );
        for policy in [LazyPromotion::Off, LazyPromotion::Background { reads: 2 }] {
            assert_eq!(classes.map(|class| policy.resolve(class)), [policy; 2]);
        }
        for (policy, spelling) in [
            (LazyPromotion::Auto, "auto"),
            (LazyPromotion::Off, "off"),
            (LazyPromotion::Background { reads: 2 }, "bg:2"),
        ] {
            assert_eq!(policy.to_string(), spelling);
            assert_eq!(parse(spelling).unwrap(), policy);
        }
    }

    #[test]
    fn test_lazy_dense_to_eager_knob() {
        let parse = |mode: &str| {
            LayeredLazyConfig::from_lookup(|key| match key {
                LAZY_FULL_ENV => Some("1".to_string()),
                LAZY_DENSE_TO_EAGER_ENV => Some(mode.to_string()),
                _ => None,
            })
            .map(|config| config.dense_to_eager)
        };
        for (value, expected) in [
            ("off", DenseToEager::Off),
            (" origin ", DenseToEager::Origin),
            ("all", DenseToEager::All),
            // The former flag's values keep their meaning.
            ("0", DenseToEager::Off),
            ("false", DenseToEager::Off),
            ("1", DenseToEager::All),
            ("true", DenseToEager::All),
        ] {
            assert_eq!(parse(value).unwrap(), expected, "{value:?}");
        }
        for mode in [DenseToEager::Off, DenseToEager::Origin, DenseToEager::All] {
            assert_eq!(mode.to_string(), mode.as_str());
            assert_eq!(parse(mode.as_str()).unwrap(), mode);
        }
        for value in ["2", "ORIGIN", "on", "auto", ""] {
            let error = parse(value).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(LAZY_DENSE_TO_EAGER_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }

        // Tiers of the high and low planes of probes whose planes are not
        // both resident: some plane on no cache tier, or every plane local.
        let (resident, local, absent) = (CacheTier::Resident, CacheTier::Local, CacheTier::Absent);
        let (low, high) = (OriginLatencyClass::Low, OriginLatencyClass::High);
        let slow = [[resident, absent], [absent, local], [absent, absent]];
        let local_only = [[resident, local], [local, resident], [local, local]];
        for tiers in slow {
            assert!(high.reads_slow_origin(&tiers), "{tiers:?}");
            assert!(!low.reads_slow_origin(&tiers), "{tiers:?}");
            assert!(DenseToEager::Origin.routes(high, &tiers), "{tiers:?}");
            assert!(!DenseToEager::Origin.routes(low, &tiers), "{tiers:?}");
        }
        for tiers in local_only {
            for class in [low, high] {
                let context = format!("{class} {tiers:?}");
                assert!(!class.reads_slow_origin(&tiers), "{context}");
                assert!(!DenseToEager::Origin.routes(class, &tiers), "{context}");
            }
        }
        for tiers in slow.iter().chain(&local_only) {
            for class in [low, high] {
                let context = format!("{class} {tiers:?}");
                assert!(!DenseToEager::Off.routes(class, tiers), "{context}");
                assert!(DenseToEager::All.routes(class, tiers), "{context}");
            }
        }
    }

    #[test]
    fn test_lazy_far_window_knob() {
        let parse = |window: Option<&str>, inflight: Option<&str>| {
            LayeredLazyConfig::from_lookup(|key| match key {
                LAZY_FULL_ENV => Some("1".to_string()),
                LAZY_FAR_WINDOW_ENV => window.map(String::from),
                LAZY_FAR_INFLIGHT_ENV => inflight.map(String::from),
                _ => None,
            })
        };
        let defaults = LayeredLazyConfig::default();
        assert_eq!((defaults.far_window, defaults.far_inflight), (64, 64));
        let config = parse(Some(" 0 "), Some("1")).unwrap();
        assert_eq!((config.far_window, config.far_inflight), (0, 1));
        let most = Semaphore::MAX_PERMITS.to_string();
        assert_eq!(
            parse(None, Some(most.as_str())).unwrap().far_inflight,
            Semaphore::MAX_PERMITS
        );
        // A pool cannot hold more permits, and a pool without any would
        // never issue a gather beyond the ordinary window.
        let too_many = (Semaphore::MAX_PERMITS + 1).to_string();
        for (name, window, inflight) in [
            (LAZY_FAR_WINDOW_ENV, Some("wide"), None),
            (LAZY_FAR_WINDOW_ENV, Some("-1"), None),
            (LAZY_FAR_INFLIGHT_ENV, None, Some("0")),
            (LAZY_FAR_INFLIGHT_ENV, None, Some(too_many.as_str())),
            (LAZY_FAR_INFLIGHT_ENV, None, Some("")),
        ] {
            let error = parse(window, inflight).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            let value = window.or(inflight).unwrap();
            assert!(message.contains(name), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }

        // The far window applies only where every origin read is a round
        // trip, and only when wider than the ordinary window.
        let (low, high) = (OriginLatencyClass::Low, OriginLatencyClass::High);
        assert_eq!(defaults.active_far_window(low), None);
        assert_eq!(defaults.active_far_window(high), Some(64));
        for (window, far_window, expected) in [
            (16, 17, Some(17)),
            (16, 16, None),
            (16, 8, None),
            (0, usize::MAX, Some(usize::MAX)),
        ] {
            let config = LayeredLazyConfig {
                window,
                far_window,
                ..defaults
            };
            let context = format!("window={window} far_window={far_window}");
            assert_eq!(config.active_far_window(high), expected, "{context}");
            assert_eq!(config.active_far_window(low), None, "{context}");
        }
    }

    #[tokio::test]
    async fn test_lazy_far_permits_count_permits_in_flight() {
        let error = LazyFarPermits::try_new(Semaphore::MAX_PERMITS + 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let permits = LazyFarPermits::try_new(2).unwrap();
        let shared = permits.clone();
        assert_eq!((permits.size(), permits.in_flight()), (2, 0));
        let first = permits.try_acquire().unwrap();
        let second = shared.try_acquire().unwrap();
        // Clones share one pool.
        assert!(permits.try_acquire().is_none());
        assert_eq!(shared.in_flight(), 2);
        // A wait abandoned before a permit frees up takes none.
        assert!(permits.acquire().now_or_never().is_none());
        drop(first);
        assert_eq!(permits.in_flight(), 1);
        let third = permits.acquire().await.unwrap();
        assert_eq!(shared.in_flight(), 2);
        drop((second, third));
        assert_eq!(permits.in_flight(), 0);
    }

    /// Every open of an index file draws its far gathers from one pool per
    /// size, which lives while an open or a taken permit holds it. The same
    /// path in another bucket is another file.
    #[test]
    fn test_lazy_far_permits_shared_per_index_file() {
        let path = "t.lance/_indices/far-permits-test/auxiliary.idx";
        let file = IndexFileKey::new("far-permits-test", "s3$bucket", path);
        let pool = LazyFarPermits::for_file(&file, 2).unwrap();
        let held = pool.try_acquire().unwrap();
        let shared = LazyFarPermits::for_file(&file, 2).unwrap();
        assert_eq!((shared.size(), shared.in_flight()), (2, 1));
        assert_eq!(LazyFarPermits::for_file(&file, 3).unwrap().in_flight(), 0);
        let other = IndexFileKey::new("far-permits-test", "s3$other", path);
        assert_eq!(LazyFarPermits::for_file(&other, 2).unwrap().in_flight(), 0);
        // The taken permit keeps the pool once no open holds it.
        drop((pool, shared));
        assert_eq!(LazyFarPermits::for_file(&file, 2).unwrap().in_flight(), 1);
        drop(held);
        assert_eq!(LazyFarPermits::for_file(&file, 2).unwrap().in_flight(), 0);
        let error = LazyFarPermits::for_file(&file, Semaphore::MAX_PERMITS + 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
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
    fn test_origin_latency_knob() {
        // The accessor reports the process environment's value.
        let process = std::env::var(ORIGIN_LATENCY_ENV).ok();
        assert_eq!(
            origin_latency_setting().ok(),
            origin_latency_from(process.as_deref()).ok()
        );
        assert_eq!(origin_latency_from(None).unwrap(), None);
        assert_eq!(origin_latency_from(Some("auto")).unwrap(), None);
        assert_eq!(
            origin_latency_from(Some("low")).unwrap(),
            Some(OriginLatencyClass::Low)
        );
        assert_eq!(
            origin_latency_from(Some(" high ")).unwrap(),
            Some(OriginLatencyClass::High)
        );
        for class in [OriginLatencyClass::Low, OriginLatencyClass::High] {
            assert_eq!(
                origin_latency_from(Some(class.as_str())).unwrap(),
                Some(class)
            );
            assert_eq!(class.to_string(), class.as_str());
        }
        assert_eq!(OriginLatencyClass::default(), OriginLatencyClass::Low);
        for value in ["cloud", "HIGH", "1", ""] {
            let error = origin_latency_from(Some(value)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(ORIGIN_LATENCY_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
    }

    /// An explicit setting wins over a session's hint, and the hint over the
    /// class of the store, low for local files and memory; without a hint the
    /// class resolves as [`OriginLatencyClass::resolve`] resolves it. Lance's
    /// IVF tests cover a cloud store, which needs a URL to build.
    #[test]
    fn test_origin_latency_resolve_with_hint() {
        let classes = [
            None,
            Some(OriginLatencyClass::Low),
            Some(OriginLatencyClass::High),
        ];
        for store in [ObjectStore::local(), ObjectStore::memory()] {
            let scheme = store.scheme().to_string();
            let auto = OriginLatencyClass::of_store(&store);
            assert_eq!(auto, OriginLatencyClass::Low, "{scheme}");
            for setting in classes {
                assert_eq!(
                    OriginLatencyClass::resolve_with_hint(setting, None, &store),
                    OriginLatencyClass::resolve(setting, &store),
                    "{scheme} setting={setting:?}"
                );
                for hint in classes {
                    let expected = match (setting, hint) {
                        (Some(class), _) | (None, Some(class)) => class,
                        (None, None) => auto,
                    };
                    assert_eq!(
                        OriginLatencyClass::resolve_with_hint(setting, hint, &store),
                        expected,
                        "{scheme} setting={setting:?} hint={hint:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_sign_bounds_knob() {
        // The accessor reports the process environment's value.
        let process = std::env::var(SIGN_BOUNDS_ENV).ok();
        assert_eq!(
            sign_bounds_setting().ok(),
            sign_bounds_from(process.as_deref()).ok()
        );
        assert_eq!(sign_bounds_from(None).unwrap(), SignBounds::Lazy);
        assert_eq!(
            sign_bounds_from(Some(" eager ")).unwrap(),
            SignBounds::Eager
        );
        for sign_bounds in [SignBounds::Lazy, SignBounds::Eager] {
            assert_eq!(
                sign_bounds_from(Some(sign_bounds.as_str())).unwrap(),
                sign_bounds
            );
            assert_eq!(sign_bounds.to_string(), sign_bounds.as_str());
        }
        assert_eq!(SignBounds::default(), SignBounds::Lazy);
        for value in ["auto", "EAGER", "1", ""] {
            let error = sign_bounds_from(Some(value)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(SIGN_BOUNDS_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
    }

    #[test]
    fn test_resident_columns_knob() {
        // The accessor reports the process environment's value.
        let process = std::env::var(RESIDENT_COLUMNS_ENV).ok();
        assert_eq!(
            resident_columns_setting().ok(),
            resident_columns_from(process.as_deref()).ok()
        );
        assert_eq!(
            resident_columns_from(None).unwrap(),
            ResidentColumnsSetting::Auto
        );
        assert_eq!(
            resident_columns_from(Some(" on ")).unwrap(),
            ResidentColumnsSetting::On
        );
        let settings = [
            ResidentColumnsSetting::Auto,
            ResidentColumnsSetting::On,
            ResidentColumnsSetting::Off,
        ];
        for setting in settings {
            assert_eq!(
                resident_columns_from(Some(setting.as_str())).unwrap(),
                setting
            );
            assert_eq!(setting.to_string(), setting.as_str());
        }
        assert_eq!(
            ResidentColumnsSetting::default(),
            ResidentColumnsSetting::Auto
        );
        // Whatever the origin, `auto` keeps a store wherever the cache pins
        // it; `on` keeps one and `off` none.
        let store = ResidentStoreSize {
            bytes: 100,
            charge: 200,
        };
        for (setting, expected) in [
            (ResidentColumnsSetting::Auto, true),
            (ResidentColumnsSetting::On, true),
            (ResidentColumnsSetting::Off, false),
        ] {
            assert_eq!(setting.admits(store, Some(1 << 20), true), expected);
        }
        for value in ["1", "ON", "true", ""] {
            let error = resident_columns_from(Some(value)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(RESIDENT_COLUMNS_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
    }

    /// The environment's `on` or `off` wins over every session setting; its
    /// `auto` defers to the session's.
    #[test]
    fn resident_columns_resolve_with_session() {
        use ResidentColumnsSetting::{Auto, Off, On};
        for (env, session, expected) in [
            (Auto, Auto, Auto),
            (Auto, On, On),
            (Auto, Off, Off),
            (On, Auto, On),
            (On, On, On),
            (On, Off, On),
            (Off, Auto, Off),
            (Off, On, Off),
            (Off, Off, Off),
        ] {
            assert_eq!(
                ResidentColumnsSetting::resolve_with_session(env, session),
                expected,
                "env={env} session={session}"
            );
        }
    }

    /// `codes` by default; an index keeps code-only entries only with a
    /// resident store.
    #[test]
    fn test_entry_columns_knob() {
        let process = std::env::var(ENTRY_COLUMNS_ENV).ok();
        assert_eq!(
            entry_columns_setting().ok(),
            entry_columns_from(process.as_deref()).ok()
        );
        assert_eq!(DEFAULT_ENTRY_COLUMNS, EntryColumns::Codes);
        assert_eq!(entry_columns_from(None).unwrap(), EntryColumns::Codes);
        assert_eq!(
            entry_columns_from(Some(" all ")).unwrap(),
            EntryColumns::All
        );
        for setting in [EntryColumns::All, EntryColumns::Codes] {
            assert_eq!(entry_columns_from(Some(setting.as_str())).unwrap(), setting);
        }
        for (setting, resident, expected) in [
            (EntryColumns::Codes, true, EntryColumns::Codes),
            (EntryColumns::Codes, false, EntryColumns::All),
            (EntryColumns::All, true, EntryColumns::All),
            (EntryColumns::All, false, EntryColumns::All),
        ] {
            assert_eq!(setting.resolve(resident), expected, "{setting} {resident}");
        }
        for value in ["code", "ALL", "1", ""] {
            let error = entry_columns_from(Some(value)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(ENTRY_COLUMNS_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
    }

    /// `off` never keeps a store; `on` keeps any store with columns, however
    /// large and whatever the cache; `auto` keeps one only in a cache with a
    /// pin budget, when its charge fits the pinned cap of the cache's
    /// largest admissible entry, half of it, whatever its values' bytes, and
    /// any size on a cache without that limit.
    #[test]
    fn admits_follows_setting_charge_and_pinned_cap() {
        const MAX_ENTRY: u64 = (1 << 20) + 1;
        let cap = lance_core::cache::pinned_partition_cap(MAX_ENTRY);
        assert_eq!(cap, MAX_ENTRY / 2);
        let size = |bytes, charge| ResidentStoreSize { bytes, charge };
        let cases = [
            (size(0, 0), Some(MAX_ENTRY)),
            (size(1, 100), Some(MAX_ENTRY)),
            (size(cap - 100, cap), Some(MAX_ENTRY)),
            // Values at the cap, charged past it.
            (size(cap, cap + 100), Some(MAX_ENTRY)),
            (size(cap - 100, cap + 1), Some(MAX_ENTRY)),
            (size(MAX_ENTRY + 1, MAX_ENTRY + 100), Some(MAX_ENTRY)),
            (size(1, 100), Some(0)),
            (size(u64::MAX, u64::MAX), None),
            (size(0, 0), None),
        ];
        for (setting, expected) in [
            (
                ResidentColumnsSetting::Auto,
                [false, true, true, false, false, false, false, true, false],
            ),
            (
                ResidentColumnsSetting::On,
                [false, true, true, true, true, true, true, true, false],
            ),
            (ResidentColumnsSetting::Off, [false; 9]),
        ] {
            let admitted = cases.map(|(store, max)| setting.admits(store, max, true));
            assert_eq!(admitted, expected, "{setting}");
            // Without a pin budget a lease cannot pin the store, so `auto`
            // keeps none whatever the fit; `on` and `off` ignore the budget.
            let unpinnable = cases.map(|(store, max)| setting.admits(store, max, false));
            let expected = match setting {
                ResidentColumnsSetting::Auto => [false; 9],
                _ => expected,
            };
            assert_eq!(unpinnable, expected, "{setting} without a pin budget");
        }
        assert!(resident_store_fits(cap, Some(MAX_ENTRY)));
        assert!(!resident_store_fits(cap + 1, Some(MAX_ENTRY)));
        assert!(resident_store_fits(u64::MAX, None));
    }

    #[test]
    fn resident_lifetime_parses_and_rejects_unknown() {
        let process = std::env::var(RESIDENT_LIFETIME_ENV).ok();
        assert_eq!(
            resident_lifetime_setting().ok(),
            resident_lifetime_from(process.as_deref()).ok()
        );
        assert_eq!(
            resident_lifetime_from(None).unwrap(),
            ResidentLifetime::Index
        );
        assert_eq!(ResidentLifetime::default(), ResidentLifetime::Index);
        for lifetime in [ResidentLifetime::Index, ResidentLifetime::Process] {
            assert_eq!(
                resident_lifetime_from(Some(lifetime.as_str())).unwrap(),
                lifetime
            );
            assert_eq!(lifetime.to_string(), lifetime.as_str());
        }
        assert_eq!(
            resident_lifetime_from(Some(" process ")).unwrap(),
            ResidentLifetime::Process
        );
        for value in ["query", "PROCESS", "1", ""] {
            let error = resident_lifetime_from(Some(value)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let message = error.to_string();
            assert!(message.contains(RESIDENT_LIFETIME_ENV), "{error}");
            assert!(message.contains(&format!("{value:?}")), "{error}");
        }
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
