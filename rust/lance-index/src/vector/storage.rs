// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use crate::vector::quantizer::QuantizerStorage;
use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{Field, Schema, SchemaRef};
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, ReaderProjection};
use lance_io::ReadBatchParams;
use lance_io::scheduler::IoStats;
use lance_linalg::distance::DistanceType;
use prost::Message;
use std::{
    any::Any,
    borrow::Cow,
    collections::{BinaryHeap, HashSet},
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::Arc,
};

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
use super::quantizer::{Quantizer, QuantizerMetadata};
use super::{ApproxMode, DISTANCE_TYPE_KEY};

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

/// Schema-metadata key recording the *source dataset* field ids of a storage file's
/// covering ("included") columns, comma separated in physical schema order.
///
/// Arrow fields carry no Lance field id, so a storage file's own schema cannot say
/// which logical column a covering column came from -- and the ids in the file's Lance
/// schema are file-local, assigned when the writer was built. Without this, a
/// distributed merge comparing shards by name and type would accept two shards whose
/// covering columns share a name and type but belong to different fields (a column
/// dropped and re-added between shard builds) and concatenate them as one.
///
/// Absent when the storage has no covering columns.
pub const COVERING_FIELD_IDS_KEY: &str = "covering_field_ids";

/// Full-precision vectors carried in index storage so `refine` can re-rank without
/// taking them from the base table.
pub const REFINE_VECTOR_COLUMN: &str = "__refine_vector";

/// Names the build pipeline uses in flight that must never be a user column.
///
/// [`REFINE_VECTOR_COLUMN`] carries the copy of the indexed column past the transform
/// chain, which consumes a column of that column's own name. The copy takes the real name
/// back before it is written, so this never appears in a stored schema -- but a *user*
/// covering column of the same name would collide with the copy mid-flight, so the name is
/// reserved. Reservation is all this is for: it is chained into `RESERVED_STORAGE_COLUMNS`
/// at index-create time and rejected there.
pub const DEFERRED_INTERNAL_COLUMNS: &[&str] = &[REFINE_VECTOR_COLUMN];

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
/// Indices of the covering ("included") columns in a vector-storage schema: every
/// field whose name is not one of `internal`. Each storage lists its own non-covering
/// column names (the row id, its quantization code columns, and any legacy bookkeeping
/// column it carries) and shares this filter, so covering detection stays a per-storage
/// data decision rather than duplicated iteration logic. See
/// [`VectorStore::covering_field_indices`] for why the set cannot be inferred generically.
pub(crate) fn covering_field_indices_excluding(
    schema: &arrow_schema::Schema,
    internal: &[&str],
) -> Vec<usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| !internal.contains(&f.name().as_str()))
        .map(|(i, _)| i)
        .collect()
}

/// Resolve the covering columns a physical storage schema can safely serve.
///
/// Arrow fields do not carry Lance field ids, so the physical columns are usable only
/// when the storage also stamps one source id for each column. Invalid or incomplete
/// capability metadata deliberately resolves to no columns: the manifest declaration is
/// still valid, but readers must fetch those values from the base table instead.
fn physical_covering_fields_from_schema(schema: &Schema, internal: &[&str]) -> Vec<(i32, Field)> {
    let covering_indices = covering_field_indices_excluding(schema, internal);
    if covering_indices.is_empty() {
        return Vec::new();
    }

    let Some(encoded_ids) = schema.metadata().get(COVERING_FIELD_IDS_KEY) else {
        // Every return of `Vec::new()` below withdraws the whole covering payload, and the
        // query then reads those columns from the base table with correct results and no
        // error -- so without a line here the operator cannot tell a covered index that
        // stopped eliding takes from one that never did.
        log::debug!(
            "Index storage carries {} covering column(s) but no `{}` stamp, so none can be \
             served; their values will come from a base-table read.",
            covering_indices.len(),
            COVERING_FIELD_IDS_KEY
        );
        return Vec::new();
    };
    let Ok(source_ids) = encoded_ids
        .split(',')
        .map(str::trim)
        .map(str::parse::<i32>)
        .collect::<std::result::Result<Vec<_>, _>>()
    else {
        return Vec::new();
    };
    if source_ids.len() != covering_indices.len()
        || source_ids.iter().collect::<HashSet<_>>().len() != source_ids.len()
    {
        log::debug!(
            "Index storage's `{}` stamp ({:?}) does not pair one distinct source id with each \
             of its {} covering column(s), so none can be served; their values will come from \
             a base-table read.",
            COVERING_FIELD_IDS_KEY,
            source_ids,
            covering_indices.len()
        );
        return Vec::new();
    }

    let covering_names = covering_indices
        .iter()
        .map(|index| schema.field(*index).name())
        .collect::<HashSet<_>>();
    if covering_names.len() != covering_indices.len() {
        log::debug!(
            "Index storage has duplicate covering column names, so a value cannot be bound to \
             one source id; none can be served and their values will come from a base-table \
             read."
        );
        return Vec::new();
    }

    source_ids
        .into_iter()
        .zip(covering_indices)
        .map(|(source_id, index)| (source_id, schema.field(index).clone()))
        .collect()
}

/// Remap `batch`'s row ids through `frag_reuse_index`, locating the row id column
/// by name rather than assuming it is at a fixed position. A covering storage batch
/// is built from a scan projection of `[<covering...>, vector]` with `_rowid`
/// appended by the scan, so `_rowid` lands wherever the covering columns end --
/// never reliably at a fixed index once any are configured.
pub(crate) fn remap_row_ids_by_name(
    batch: RecordBatch,
    frag_reuse_index: &dyn RowIdRemapper,
) -> Result<RecordBatch> {
    let row_id_idx = batch
        .schema()
        .index_of(ROW_ID)
        .map_err(|_| Error::schema(format!("column {} not found", ROW_ID)))?;
    frag_reuse_index.remap_row_ids_record_batch(batch, row_id_idx)
}

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

    /// This storage's own column names: the row id, its quantization code
    /// columns, and any legacy bookkeeping column it carries. Everything else in
    /// the schema is a covering ("included") column.
    ///
    /// Required rather than defaulted on purpose: only the storage knows which of
    /// its columns are code columns, and a wrong answer here silently mislabels
    /// covering data (e.g. it would mistake RaBitQ's code columns for covering
    /// ones). Making it a required associated const means a new storage cannot
    /// forget to answer.
    const INTERNAL_COLUMNS: &'static [&'static str];

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

    /// Field indices of the "included"/covering columns: the extra columns
    /// stored alongside the row id and quantization code so a covered query can
    /// skip the take from the base table. Derived from [`Self::INTERNAL_COLUMNS`].
    fn covering_field_indices(&self) -> Vec<usize> {
        covering_field_indices_excluding(self.schema().as_ref(), Self::INTERNAL_COLUMNS)
    }

    /// `[_rowid, <included cols...>]` for the whole storage. Captured while a
    /// partition is loaded during search so covering columns can be emitted with
    /// the result without a separate take. Returns `None` if there are no
    /// included columns (ordinary index — nothing to cover).
    fn covering_batch(&self) -> Result<Option<RecordBatch>> {
        let included = self.covering_field_indices();
        if included.is_empty() {
            return Ok(None);
        }
        let schema = self.schema().clone();
        let row_id_idx = schema.index_of(ROW_ID)?;
        let mut indices = Vec::with_capacity(included.len() + 1);
        indices.push(row_id_idx);
        indices.extend(included);
        // Project each batch BEFORE concatenating. Concatenating the full partition first
        // would copy every quantization-code column only to discard it on the next line --
        // and those columns dominate the partition (HNSW partitions target 1 << 20 rows), so
        // on a multi-probe query that copy is hundreds of MB of pure waste on the ANN hot
        // path. Only `[_rowid, <included...>]` is ever read from the result.
        let projected: Vec<RecordBatch> = self
            .to_batches()?
            .map(|batch| batch.project(&indices))
            .collect::<std::result::Result<_, _>>()?;
        let projected_schema = Arc::new(schema.project(&indices)?);
        Ok(Some(concat_batches(&projected_schema, projected.iter())?))
    }

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
        // Batches can come from different sources (existing partitions loaded
        // from disk vs freshly shuffled/split/reassigned data) that carry the
        // same columns in a different order -- notably when the index has
        // included/covering columns. `concat_batches` matches columns by
        // position, so align every batch to the first batch's column order (by
        // name). No-op when all batches already share a schema.
        let schema = batches[0].schema();
        let aligned: Vec<RecordBatch> = batches
            .iter()
            .map(|b| {
                if b.schema() == schema {
                    return Ok(b.clone());
                }
                // `project_by_schema` reorders by name and errors on a missing
                // column, but it cannot see extras -- and a batch with MORE
                // columns must error rather than have them silently dropped
                // (e.g. covering columns the index metadata still advertises).
                // Equal count + every expected column present = same set.
                if b.num_columns() != schema.fields().len() {
                    let names = |s: &arrow_schema::Schema| {
                        s.fields()
                            .iter()
                            .map(|f| f.name().as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    return Err(Error::index(format!(
                        "mismatched columns while merging vector storage batches: \
                         expected [{}], got [{}]",
                        names(&schema),
                        names(&b.schema()),
                    )));
                }
                Ok(b.project_by_schema(schema.as_ref())?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut batch = concat_batches(&schema, aligned.iter())?;

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

/// How [`IvfQuantizationStorage::take_covering`] reads a partition's covering values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoveringRead {
    /// [`ReadBatchParams::Indices`] -- read only the survivors' rows.
    Scattered,
    /// [`ReadBatchParams::Range`] -- read the partition's whole row range.
    Sequential,
}

/// Survivors at or above this percentage of a partition's rows make the scattered read
/// degenerate: `Indices` over most of a range costs more than the sequential read of that
/// range it replaced, because the scattered form gives up contiguity while still touching
/// nearly every page.
///
/// # Derivation
///
/// Measured, not guessed. At production partition sizes (~8,192 rows) and `k <= 100` the
/// survivors of one probe are **<= 1.2%** of a partition; on the 2,048-row fixture
/// partitions with `k = 10` it is 0.49%. Ten percent therefore sits roughly an order of
/// magnitude above the worst observed common case, so the fallback costs nothing in
/// practice while still bounding the degenerate case at "no worse than the read this
/// replaced". The design's §1 warns explicitly against assuming random access always
/// wins; this is that guard, deliberately conservative in the direction that cannot
/// regress.
pub const COVERING_SCATTERED_READ_MAX_PERCENT: usize = 10;

/// Which read shape to use for `survivors` covering rows out of `partition_rows`.
///
/// A zero-row partition has nothing to read either way; it is reported as sequential so
/// callers have a single empty-range path.
pub fn covering_read_for(survivors: usize, partition_rows: usize) -> CoveringRead {
    if partition_rows == 0
        || survivors.saturating_mul(100)
            >= partition_rows.saturating_mul(COVERING_SCATTERED_READ_MAX_PERCENT)
    {
        CoveringRead::Sequential
    } else {
        CoveringRead::Scattered
    }
}

/// Which columns a partition load reads out of the storage file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionColumns {
    /// Only the storage's own columns ([`VectorStore::INTERNAL_COLUMNS`]): the row id,
    /// the quantization codes and any per-row factor columns. Covering columns are left
    /// unread, which is what makes the loaded partition independent of the query that
    /// loaded it -- and therefore sound to share under a partition-id-only cache key.
    Internal,
    /// Every column the file carries, covering columns included. Required by the paths
    /// that rewrite a partition into a new index file: a covering column that was never
    /// read cannot be written back.
    All,
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Q::Metadata,

    ivf: IvfModel,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    /// Lazily-computed covering schema (see [`Self::covering_schema`]). The schema is
    /// fixed for the storage's lifetime and is consulted on every search, so it is
    /// resolved once rather than per query. Constructing it builds an empty `Q::Storage`:
    /// cheap for a current-format index (zero rows, and the codebook is cloned, not
    /// decoded), but the legacy `codebook_tensor` path does decode, so this is not free
    /// for every index.
    covering_schema: std::sync::OnceLock<Option<SchemaRef>>,
    /// Lazily-computed internal-column projection (see [`Self::internal_projection`]).
    /// Fixed for the storage's lifetime and consulted on every partition load, so it is
    /// resolved once rather than per probe.
    internal_projection: std::sync::OnceLock<ReaderProjection>,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.ivf.deep_size_of_children(context)
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

        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
            covering_schema: std::sync::OnceLock::new(),
            internal_projection: std::sync::OnceLock::new(),
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
            covering_schema: std::sync::OnceLock::new(),
            internal_projection: std::sync::OnceLock::new(),
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

    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.reader.schema().as_ref().into())
    }

    /// The `[_rowid, <included cols...>]` schema for this index's covering
    /// columns, or `None` if the index has no covering columns. Derived from an
    /// empty storage instance (no I/O) so callers can emit the correct covered
    /// output schema even when zero partitions are searched. Cached because it is queried
    /// per search. The cache is best-effort under concurrency: racing cold callers may each
    /// construct one and all but the first are discarded. That is deliberate -- the empty
    /// construction is cheap on the current format, and `OnceLock::get_or_init` cannot
    /// carry the `Result` this returns.
    pub fn covering_schema(&self) -> Result<Option<SchemaRef>> {
        if let Some(cached) = self.covering_schema.get() {
            return Ok(cached.clone());
        }
        let arrow_schema = arrow_schema::Schema::from(self.reader.schema().as_ref());
        let empty = RecordBatch::new_empty(Arc::new(arrow_schema));
        // No remapper: the batch is empty, so there are no row ids to remap and the
        // remapper cannot influence the schema this derives. Passing one would also be
        // wrong now that the field holds a `dyn RowIdRemapper` -- the default
        // `try_from_batch_with_remapper` rejects a remapper outright for every storage
        // that does not override it, which would fail this call on an SQ/RQ/FLAT index
        // whose dataset happens to carry a fragment-reuse index.
        let storage = Q::Storage::try_from_batch(empty, self.metadata(), self.distance_type, None)?;
        let computed = storage.covering_batch()?.map(|b| b.schema());
        Ok(self.covering_schema.get_or_init(|| computed).clone())
    }

    /// Physical covering columns this storage can safely serve, paired with their source
    /// dataset field ids and returned in storage order.
    ///
    /// A missing or malformed source-id stamp makes the physical covering payload
    /// unavailable rather than making the index unreadable. Query planning will then use
    /// the ordinary base-table take for the declared columns.
    pub fn physical_covering_fields(&self) -> Vec<(i32, Field)> {
        let schema = Schema::from(self.reader.schema().as_ref());
        physical_covering_fields_from_schema(&schema, Q::Storage::INTERNAL_COLUMNS)
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    /// The read projection selecting only this storage's own columns.
    ///
    /// [`VectorStore::INTERNAL_COLUMNS`] is the union over every build a storage type
    /// has ever produced -- the legacy `__ivf_part_id`, RaBitQ's factor columns that
    /// exist only above a `num_bits` threshold -- so it is intersected with this file's
    /// schema first. Projecting the const directly would reject any file that legitimately
    /// omits one of those columns, which is most of them.
    fn internal_projection(&self) -> Result<ReaderProjection> {
        if let Some(cached) = self.internal_projection.get() {
            return Ok(cached.clone());
        }
        let schema = self.reader.schema();
        let names: Vec<&str> = Q::Storage::INTERNAL_COLUMNS
            .iter()
            .copied()
            .filter(|name| schema.field(name).is_some())
            .collect();
        let projection = lance_file::versions::reader_projection_from_column_names(
            self.reader.metadata().version(),
            schema,
            &names,
        )?;
        Ok(self.internal_projection.get_or_init(|| projection).clone())
    }

    /// Load a partition's quantization storage, reading the columns named by
    /// `columns` and optionally measuring the exact I/O it performs into `io_stats`.
    ///
    /// [`PartitionColumns::Internal`] leaves the covering columns on disk. Search
    /// wants that: covering is needed for at most `k` survivors while the codes are
    /// scanned for every row on every probe, and an entry that holds no covering is
    /// the same entry for every query -- which is what makes caching it under a
    /// partition id alone correct rather than accidentally correct.
    ///
    /// When `io_stats` is `Some`, the partition is read through a reader whose
    /// scheduler also records into the sink (a cheap clone that shares all
    /// cached metadata, so no file is re-opened).  When `None`, the normal
    /// uninstrumented reader is used.
    pub async fn load_partition(
        &self,
        part_id: usize,
        columns: PartitionColumns,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        let projection = match columns {
            PartitionColumns::Internal => Some(self.internal_projection()?),
            PartitionColumns::All => None,
        };
        // The batch schema must describe exactly what was read: `concat_batches`
        // indexes each batch by this schema's field positions without comparing the
        // two, so the full file schema over a projected read would index past the
        // last column.
        let schema: SchemaRef = Arc::new(match &projection {
            Some(projection) => projection.schema.as_ref().into(),
            None => self.reader.schema().as_ref().into(),
        });
        let range = self.ivf.row_range(part_id);
        let batch = if range.is_empty() {
            RecordBatch::new_empty(schema)
        } else {
            let reader = match &io_stats {
                Some(io_stats) => Cow::Owned(self.reader.with_io_stats(io_stats.recorder())),
                None => Cow::Borrowed(&self.reader),
            };
            let params = ReadBatchParams::Range(range);
            let stream = match projection {
                Some(projection) => {
                    reader
                        .read_stream_projected(
                            params,
                            u32::MAX,
                            1,
                            projection,
                            FilterExpression::no_filter(),
                        )
                        .await?
                }
                None => {
                    reader
                        .read_stream(params, u32::MAX, 1, FilterExpression::no_filter())
                        .await?
                }
            };
            let batches = stream.try_collect::<Vec<_>>().await?;
            concat_batches(&schema, batches.iter())?
        };
        Q::Storage::try_from_batch_with_remapper(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }

    /// This storage's covering ("included") column names in storage order, narrowed to
    /// `wanted`.
    ///
    /// `wanted` is [`Query::covering_projection`], whose three states are all meaningful:
    /// `None` keeps every physical column, `Some(&[])` keeps none, and `Some(cols)` keeps
    /// the intersection. Execution planners pass an explicit, storage-verified `Some`;
    /// `None` is only the raw-index behavior before capability resolution. The result is
    /// empty for an ordinary index and for a query that needs no covering column -- in
    /// both cases the caller skips the covering read entirely.
    ///
    /// Order follows storage, never `wanted`: that is the order the gathered batch comes
    /// back in. The planner verifies that this physical order is compatible with its
    /// declaration-ordered output schema before enabling covering.
    ///
    /// [`Query::covering_projection`]: crate::vector::Query::covering_projection
    pub fn covering_columns(&self, wanted: Option<&[String]>) -> Result<Vec<String>> {
        let Some(schema) = self.covering_schema()? else {
            return Ok(Vec::new());
        };
        Ok(schema
            .fields()
            .iter()
            .map(|field| field.name())
            .filter(|name| name.as_str() != ROW_ID)
            .filter(|name| wanted.is_none_or(|wanted| wanted.contains(name)))
            .cloned()
            .collect())
    }

    /// The read projection [`Self::take_covering`] uses: `[_rowid, <columns...>]`.
    fn covering_projection(&self, columns: &[String]) -> Result<ReaderProjection> {
        let mut names: Vec<&str> = Vec::with_capacity(columns.len() + 1);
        names.push(ROW_ID);
        names.extend(columns.iter().map(String::as_str));
        lance_file::versions::reader_projection_from_column_names(
            self.reader.metadata().version(),
            self.reader.schema(),
            &names,
        )
    }

    /// The `[_rowid, <columns...>]` schema [`Self::take_covering`] returns.
    ///
    /// Callers declare their covered output schema from this rather than from
    /// [`Self::covering_schema`] so that the schema they promise and the batches they
    /// emit come from a single derivation. The two agree today, but they are computed
    /// from different inputs (a projected file schema against an empty `Q::Storage`) and
    /// a divergence would surface as a stream that mixes batch schemas.
    pub fn covering_read_schema(&self, columns: &[String]) -> Result<SchemaRef> {
        Ok(Arc::new(
            self.covering_projection(columns)?.schema.as_ref().into(),
        ))
    }

    /// Read `[_rowid, <columns...>]` for one partition's search survivors.
    ///
    /// This is the covering read the search path pays after scoring has settled the heap,
    /// so it is proportional to `k` rather than to the partition size -- which is why the
    /// partition entry the search caches carries no covering at all (see
    /// [`PartitionColumns::Internal`]) and why this result is deliberately **not** cached:
    /// it is a different set of rows for every query.
    ///
    /// `positions` are offsets into the partition's row range, strictly ascending and
    /// deduplicated (the reader's take path requires that). `None` means the caller could
    /// not derive them -- a deferred fragment-reuse remap drops rows from the loaded
    /// partition, so its row order no longer matches the file's -- and the whole range is
    /// read instead, with the caller matching by row id.
    ///
    /// Row ids come back with the values, so the caller can align covering values to its
    /// survivors **by row id** rather than by position, and a survivor whose row the
    /// gather did not return is an error rather than a silent null.
    pub async fn take_covering(
        &self,
        part_id: usize,
        positions: Option<&[u32]>,
        columns: &[String],
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        let projection = self.covering_projection(columns)?;
        // As in `load_partition`: `concat_batches` indexes each batch by this schema's
        // field positions without comparing the two, so it must describe the projected
        // read rather than the whole file.
        let schema: SchemaRef = Arc::new(projection.schema.as_ref().into());
        let range = self.ivf.row_range(part_id);
        if range.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        // `ReadBatchParams::Indices` addresses rows with a `u32`, so a partition ending
        // beyond that cannot be expressed as a scattered read at all. Falling back keeps
        // such a file readable (just slower) instead of failing the query.
        let addressable = range.end <= u32::MAX as usize;
        let read = match positions {
            Some(positions) if addressable => covering_read_for(positions.len(), range.len()),
            _ => CoveringRead::Sequential,
        };
        let params = match read {
            CoveringRead::Sequential => ReadBatchParams::Range(range.clone()),
            CoveringRead::Scattered => ReadBatchParams::Indices(UInt32Array::from_iter_values(
                // `addressable` bounds the sum, but saturate anyway: a caller passing a
                // position past the partition must reach the reader's own bounds check as
                // an error, not overflow here.
                positions
                    .unwrap_or_default()
                    .iter()
                    .map(|position| (range.start as u32).saturating_add(*position)),
            )),
        };
        let reader = match &io_stats {
            Some(io_stats) => Cow::Owned(self.reader.with_io_stats(io_stats.recorder())),
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
        let batch = concat_batches(&schema, batches.iter())?;
        match &self.frag_reuse_index {
            // The loaded partition's row ids were remapped through the same index when it
            // was built from this file, so the gathered ids must be too or nothing matches.
            Some(frag_reuse_index) => remap_row_ids_by_name(batch, frag_reuse_index.as_ref()),
            None => Ok(batch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        COVERING_FIELD_IDS_KEY, COVERING_SCATTERED_READ_MAX_PERCENT, CoveringRead,
        QueryScratchCapacity, QueryScratchPool, covering_read_for,
        physical_covering_fields_from_schema,
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::deepsize::DeepSizeOf;
    use std::collections::HashMap;

    const TEST_INTERNAL_COLUMNS: &[&str] = &["_rowid", "__pq_code"];

    fn storage_schema(covering_fields: Vec<Field>, source_ids: Option<&str>) -> Schema {
        let mut fields = vec![
            Field::new("_rowid", DataType::UInt64, false),
            Field::new("__pq_code", DataType::Binary, false),
        ];
        fields.extend(covering_fields);
        let metadata = source_ids
            .map(|ids| HashMap::from([(COVERING_FIELD_IDS_KEY.to_string(), ids.to_string())]))
            .unwrap_or_default();
        Schema::new_with_metadata(fields, metadata)
    }

    #[test]
    fn test_physical_covering_fields_require_valid_source_ids() {
        let covering = vec![
            Field::new("price", DataType::Int64, true),
            Field::new("payload", DataType::Utf8, false),
        ];
        let schema = storage_schema(covering.clone(), Some("17, 23"));

        let resolved = physical_covering_fields_from_schema(&schema, TEST_INTERNAL_COLUMNS);

        assert_eq!(
            resolved,
            vec![(17, covering[0].clone()), (23, covering[1].clone())]
        );
    }

    #[test]
    fn test_physical_covering_fields_treat_invalid_capability_as_absent() {
        let covering = vec![
            Field::new("price", DataType::Int64, true),
            Field::new("payload", DataType::Utf8, false),
        ];
        for source_ids in [None, Some(""), Some("17"), Some("17,nope"), Some("17,17")] {
            let schema = storage_schema(covering.clone(), source_ids);
            assert!(
                physical_covering_fields_from_schema(&schema, TEST_INTERNAL_COLUMNS).is_empty(),
                "source ids {source_ids:?} must not prove a physical covering capability"
            );
        }

        let duplicate_names = storage_schema(
            vec![
                Field::new("price", DataType::Int64, true),
                Field::new("price", DataType::Int64, true),
            ],
            Some("17,23"),
        );
        assert!(
            physical_covering_fields_from_schema(&duplicate_names, TEST_INTERNAL_COLUMNS)
                .is_empty()
        );
    }

    #[test]
    fn test_physical_covering_fields_ignore_orphaned_source_id_metadata() {
        let schema = storage_schema(Vec::new(), Some("17"));
        assert!(physical_covering_fields_from_schema(&schema, TEST_INTERNAL_COLUMNS).is_empty());
    }

    /// The scattered read is the point of the survivor gather, so it must be chosen across
    /// the whole range of survivor fractions a real query produces -- and abandoned once
    /// `Indices` would scatter over most of the partition anyway, where the sequential read
    /// it replaced is cheaper.
    ///
    /// The inputs are the measured ones: at production partition sizes (~8,192 rows) the
    /// survivors of one probe are <= 1.2% at `k <= 100`, and 0.49% on the 2,048-row fixture
    /// partitions at `k = 10`. Both must land on `Scattered` with room to spare, or the
    /// threshold is not conservative -- it is in the way.
    ///
    /// The two rows either side of 10% are what make this a threshold test rather than a
    /// "scattered usually wins" test: moving `COVERING_SCATTERED_READ_MAX_PERCENT` in
    /// either direction fails one of them.
    #[test]
    fn test_covering_read_falls_back_to_a_sequential_read_near_the_threshold() {
        assert_eq!(COVERING_SCATTERED_READ_MAX_PERCENT, 10);

        // k = 10 on a 2,048-row fixture partition: 0.49%.
        assert_eq!(covering_read_for(10, 2048), CoveringRead::Scattered);
        // k = 100 on an 8,192-row production partition: 1.2%, the measured worst case.
        assert_eq!(covering_read_for(100, 8192), CoveringRead::Scattered);
        // Either side of the threshold: 819/8192 = 9.998%, 820/8192 = 10.01%.
        assert_eq!(covering_read_for(819, 8192), CoveringRead::Scattered);
        assert_eq!(covering_read_for(820, 8192), CoveringRead::Sequential);
        // The degenerate case the threshold exists for.
        assert_eq!(covering_read_for(8192, 8192), CoveringRead::Sequential);
        // An empty partition has nothing to read either way; one path, not two.
        assert_eq!(covering_read_for(0, 0), CoveringRead::Sequential);
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
