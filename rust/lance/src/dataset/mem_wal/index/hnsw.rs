// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory HNSW index for vector similarity search.
//!
//! This is the MemWAL adapter around the local HNSW graph. Vectors are
//! retained by reference from the writer's Arrow batches, inserts are
//! published under a multi-reader / single-writer contract, and flush
//! snapshots are emitted in Lance's on-disk HNSW + FLAT storage format.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use lance_core::{Error, Result};
use lance_index::vector::hnsw::{HNSW, builder::HnswBuildParams};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;

use super::super::hnsw::{ArrowFixedSizeListVectorStore, BuildParams, HnswGraph, SearchParams};
use super::super::memtable::batch_store::StoredBatch;
use super::plugin::{
    FlushContext, FlushOutcome, MemIndex, MemIndexBuildContext, MemIndexPlugin, ResolveContext,
    ResolvedIndex,
};
use super::query::{MemMatches, MemQuery, RankedMatch, SearchContext, VectorMemQuery};
use crate::index::vector::details::vector_index_details_default;
use lance_index::IndexType;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering};
use lance_table::format::IndexMetadata;

pub use super::RowPosition;

const MEM_HNSW_DIM_PLACEHOLDER: usize = 0;

/// Write-optimized default HNSW build parameters for MemTable indexes.
///
/// MemTable HNSW graphs are rebuilt on every flush, so build speed matters more
/// than for a static base-table index. A dbpedia-1M (1536-d) sweep showed
/// `num_edges = 16, ef_construction = 100` is the best fast-write/good-recall
/// point: ~27% faster to flush than the generic `HnswBuildParams::default()`
/// (`num_edges = 20, ef_construction = 150`) with an equal-or-better recall
/// ceiling (~0.95 at ef=256, the SQ8-quantization limit). Lower `num_edges`
/// (e.g. 12) flushes faster still but drops the recall ceiling below 0.95.
pub fn mem_wal_hnsw_default() -> HnswBuildParams {
    HnswBuildParams::default()
        .num_edges(16)
        .ef_construction(100)
}

/// In-memory HNSW index queryable while building.
pub struct HnswMemIndex {
    field_id: i32,
    /// The covered column, held as a slice because that is the shape
    /// [`MemIndex::columns`] returns.
    columns: Vec<String>,
    distance_type: DistanceType,
    /// Vector dimension (lazy-initialized on first insert).
    dim: AtomicUsize,
    /// Capacity (max vectors) for both the HNSW graph and vector store.
    capacity: usize,
    /// Maximum number of Arrow batches retained by reference.
    max_batches: usize,
    build_params: HnswBuildParams,
    state: OnceLock<HnswState>,
}

struct HnswState {
    storage: Arc<ArrowFixedSizeListVectorStore>,
    graph: HnswGraph,
}

impl std::fmt::Debug for HnswMemIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswMemIndex")
            .field("field_id", &self.field_id)
            .field("column", &self.column())
            .field("distance_type", &self.distance_type)
            .field("dim", &self.dim.load(Ordering::Acquire))
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .finish()
    }
}

impl HnswMemIndex {
    pub fn with_capacity(
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        build_params: HnswBuildParams,
        capacity: usize,
        max_batches: usize,
    ) -> Self {
        Self {
            field_id,
            columns: vec![column],
            distance_type,
            dim: AtomicUsize::new(MEM_HNSW_DIM_PLACEHOLDER),
            capacity,
            max_batches,
            build_params,
            state: OnceLock::new(),
        }
    }

    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    pub fn column_name(&self) -> &str {
        &self.columns[0]
    }

    /// The covered column.
    fn column(&self) -> &str {
        &self.columns[0]
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn build_params(&self) -> &HnswBuildParams {
        &self.build_params
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Vector dimension. Returns 0 before the first insert.
    pub fn dim(&self) -> usize {
        self.dim.load(Ordering::Acquire)
    }

    pub fn len(&self) -> usize {
        self.state.get().map(|s| s.graph.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upper bound on heap bytes held — or already committed — by this index.
    ///
    /// Sized by `capacity` (the writer's `max_memtable_rows`) rather than by
    /// rows inserted: the graph and lookup slabs are pre-allocated in full on
    /// the first insert, so an idle vector memtable costs the same as a full
    /// one.
    ///
    /// Non-zero *before* that first insert too. The allocation is settled the
    /// moment the index exists — only `dim` is still unknown, and no term
    /// depends on it — so reporting zero until the row that triggers it would
    /// hide the largest allocation in a vector memtable from the admission
    /// controller that runs just ahead of it. Until then this is the reserved
    /// estimate; from the first insert on it is the graph's own measurement.
    pub(crate) fn resident_bytes(&self) -> usize {
        match self.state.get() {
            Some(s) => s.graph.resident_bytes() + s.storage.resident_bytes(),
            None => {
                HnswGraph::reserved_bytes(self.capacity, &build_params_of(&self.build_params))
                    + ArrowFixedSizeListVectorStore::reserved_bytes(self.capacity, self.max_batches)
            }
        }
    }

    fn ensure_state(&self, dim: usize) -> Result<&HnswState> {
        if let Some(state) = self.state.get() {
            if state.storage.dim() != dim {
                return Err(Error::invalid_input(format!(
                    "HNSW index column '{}' dimension changed: expected {}, got {}",
                    self.column(),
                    state.storage.dim(),
                    dim
                )));
            }
            return Ok(state);
        }

        let state = HnswState {
            storage: Arc::new(ArrowFixedSizeListVectorStore::try_new(
                self.capacity,
                self.max_batches,
                dim,
                self.distance_type,
            )?),
            graph: HnswGraph::try_new(self.capacity, to_lance_hnsw_params(&self.build_params)?)?,
        };
        self.dim.store(dim, Ordering::Release);

        if self.state.set(state).is_err() {
            // Another writer initialized first. The shard writer has a
            // single-writer contract, but handle the race defensively.
            let Some(state) = self.state.get() else {
                return Err(Error::internal(
                    "HNSW state initialization raced but no state was installed",
                ));
            };
            if state.storage.dim() != dim {
                return Err(Error::invalid_input(format!(
                    "HNSW index column '{}' dimension changed: expected {}, got {}",
                    self.column(),
                    state.storage.dim(),
                    dim
                )));
            }
            return Ok(state);
        }

        self.state
            .get()
            .ok_or_else(|| Error::internal("HNSW state was not installed after initialization"))
    }

    /// Insert vectors from a single batch.
    ///
    /// The vector column is appended by reference; the Arrow values buffer is
    /// not copied. The graph then indexes the dense id range assigned to this
    /// batch.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let (col_idx, _) = batch
            .schema()
            .column_with_name(self.column())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "HNSW index column '{}' is not in the inserted batch schema",
                    self.column()
                ))
            })?;
        let column = batch.column(col_idx);
        let fsl_ref = column.as_fixed_size_list_opt().ok_or_else(|| {
            Error::invalid_input(format!(
                "Column '{}' is not a FixedSizeList, got {:?}",
                self.column(),
                column.data_type()
            ))
        })?;
        if fsl_ref.is_empty() {
            return Ok(());
        }
        if fsl_ref.values().as_primitive_opt::<Float32Type>().is_none() {
            return Err(Error::invalid_input(format!(
                "Column '{}' must be FixedSizeList<Float32>, got values type {:?}",
                self.column(),
                fsl_ref.values().data_type()
            )));
        }
        // Null-vector rows (e.g. tombstones) are skipped inside `append_batch`,
        // which yields an empty range for an all-null batch — handled below.
        let dim = fsl_ref.value_length() as usize;
        let state = self.ensure_state(dim)?;
        let vectors = Arc::new(fsl_ref.clone());
        let id_range = state.storage.append_batch(vectors, row_offset)?;
        if id_range.is_empty() {
            return Ok(());
        }
        let snapshot = state.storage.snapshot();
        state.graph.insert_batch(id_range, &snapshot)
    }

    /// Insert vectors from multiple batches.
    pub fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        let mut combined_range: Option<std::ops::Range<u32>> = None;
        let mut state: Option<&HnswState> = None;

        for stored in batches {
            let (col_idx, _) = stored
                .data
                .schema()
                .column_with_name(self.column())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "HNSW index column '{}' is not in the inserted batch schema",
                        self.column()
                    ))
                })?;
            let column = stored.data.column(col_idx);
            let fsl_ref = column.as_fixed_size_list_opt().ok_or_else(|| {
                Error::invalid_input(format!(
                    "Column '{}' is not a FixedSizeList, got {:?}",
                    self.column(),
                    column.data_type()
                ))
            })?;
            if fsl_ref.is_empty() {
                continue;
            }
            if fsl_ref.values().as_primitive_opt::<Float32Type>().is_none() {
                return Err(Error::invalid_input(format!(
                    "Column '{}' must be FixedSizeList<Float32>, got values type {:?}",
                    self.column(),
                    fsl_ref.values().data_type()
                )));
            }
            // Null-vector rows (e.g. tombstones) are skipped inside
            // `append_batch`; an all-null batch yields an empty range, handled
            // by the `id_range.is_empty()` continue below.
            let dim = fsl_ref.value_length() as usize;
            let current_state = match state {
                Some(state) => {
                    if state.storage.dim() != dim {
                        return Err(Error::invalid_input(format!(
                            "HNSW index column '{}' dimension changed: expected {}, got {}",
                            self.column(),
                            state.storage.dim(),
                            dim
                        )));
                    }
                    state
                }
                None => self.ensure_state(dim)?,
            };
            state = Some(current_state);

            let vectors = Arc::new(fsl_ref.clone());
            let id_range = current_state
                .storage
                .append_batch(vectors, stored.row_offset)?;
            if id_range.is_empty() {
                continue;
            }

            match &mut combined_range {
                Some(range) if range.end == id_range.start => {
                    range.end = id_range.end;
                }
                Some(range) => {
                    return Err(Error::internal(format!(
                        "non-contiguous HNSW vector id range while inserting batches: existing={:?}, next={:?}",
                        range, id_range
                    )));
                }
                None => {
                    combined_range = Some(id_range);
                }
            }
        }

        if let (Some(state), Some(id_range)) = (state, combined_range) {
            let snapshot = state.storage.snapshot();
            state.graph.insert_batch(id_range, &snapshot)?;
        }
        Ok(())
    }

    /// Search for nearest neighbors of `query` with MVCC visibility.
    ///
    /// Distances are exact because the in-memory graph is backed by FLAT
    /// vectors. Rows with positions greater than `max_row_position` are
    /// filtered after graph search.
    pub fn search(
        &self,
        query: &FixedSizeListArray,
        k: usize,
        ef: Option<usize>,
        max_row_position: RowPosition,
    ) -> Result<Vec<(f32, RowPosition)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != 1 {
            return Err(Error::invalid_input(format!(
                "Query must have exactly 1 vector, got {}",
                query.len()
            )));
        }
        if query.null_count() > 0 {
            return Err(Error::invalid_input("HNSW query vector must not be null"));
        }
        let Some(state) = self.state.get() else {
            return Ok(Vec::new());
        };
        if query.value_length() as usize != state.storage.dim() {
            return Err(Error::invalid_input(format!(
                "HNSW query dimension mismatch: expected {}, got {}",
                state.storage.dim(),
                query.value_length()
            )));
        }
        let query_values = query.value(0);
        let Some(query_values) = query_values.as_primitive_opt::<Float32Type>() else {
            return Err(Error::invalid_input(format!(
                "HNSW query must contain Float32 values, got {:?}",
                query_values.data_type()
            )));
        };

        let ef_actual = ef.unwrap_or(k.max(64)).max(k);
        let snapshot = state.storage.snapshot();
        let mut out: Vec<_> = state
            .graph
            .search(
                query_values.values(),
                SearchParams::new(ef_actual, ef_actual),
                &snapshot,
            )?
            .into_iter()
            .filter_map(|result| {
                if result.row_id <= max_row_position && result.distance.is_finite() {
                    Some((result.distance, result.row_id))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        Ok(out)
    }

    /// Snapshot the in-memory HNSW into the Lance on-disk representation:
    /// returns the graph + the FLAT vector storage record batch.
    pub fn to_lance_hnsw(&self, total_rows: Option<u64>) -> Result<Option<(HNSW, RecordBatch)>> {
        let Some(state) = self.state.get() else {
            return Ok(None);
        };
        if state.graph.is_empty() {
            return Ok(None);
        }
        // Bound the graph by storage, and only in that direction. A graph past
        // storage names rows the batch has no vector for, which is the defect
        // this fixes. Storage past the graph is left whole on purpose: those
        // rows are unreachable by traversal either way, but `HNSW::search`
        // brute-forces the storage domain under a narrow prefilter
        // (`flat_search`), so dropping them would lose results this export
        // previously returned. Closing that gap means finishing index
        // application before export, not trimming storage to match.
        let storage_batch = state.storage.to_record_batch(total_rows)?;
        let hnsw_batch = state
            .graph
            .to_lance_hnsw_batch(Some(storage_batch.num_rows()))?;
        let hnsw = HNSW::load(hnsw_batch)?;
        Ok(Some((hnsw, storage_batch)))
    }
}

fn to_lance_hnsw_params(params: &HnswBuildParams) -> Result<BuildParams> {
    let params = build_params_of(params);
    // Validate by constructing a tiny graph with these params. This keeps
    // invalid builder options as boundary errors instead of delayed panics.
    HnswGraph::try_new(1, params.clone())?;
    Ok(params)
}

/// The same field-for-field translation without the validating build, so sizing
/// questions can be answered off a config that has not been accepted yet.
fn build_params_of(params: &HnswBuildParams) -> BuildParams {
    BuildParams {
        max_level: params.max_level,
        m: params.m,
        ef_construction: params.ef_construction,
        prefetch_distance: params.prefetch_distance,
        ..BuildParams::default()
    }
}

#[async_trait::async_trait]
impl super::plugin::MemIndex for HnswMemIndex {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    /// A graph's metric is baked into its structure, so a search asking for a
    /// different one is declined and brute-forced instead.
    fn can_answer(&self, query: &dyn MemQuery) -> bool {
        query
            .as_any()
            .downcast_ref::<VectorMemQuery>()
            .is_some_and(|query| {
                query
                    .distance_type
                    .is_none_or(|wanted| wanted == self.distance_type())
            })
    }

    fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        Self::insert(self, batch, row_offset)
    }

    // One graph insertion over every batch handed over, not one per batch.
    fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        Self::insert_batches(self, batches)
    }

    fn resident_bytes(&self) -> usize {
        Self::resident_bytes(self)
    }

    fn search(&self, query: &dyn MemQuery, ctx: &SearchContext) -> Result<Option<MemMatches>> {
        if !self.can_answer(query) {
            return Ok(None);
        }
        let Some(query) = query.as_any().downcast_ref::<VectorMemQuery>() else {
            return Ok(None);
        };
        let neighbours = Self::search(self, &query.vector, query.k, query.ef, ctx.max_visible)?;
        Ok(Some(MemMatches::ranked(
            neighbours
                .into_iter()
                .map(|(distance, position)| RankedMatch::new(position, distance))
                .collect(),
        )))
    }

    /// A vector index does not train from a value stream: its flush writes the
    /// graph and its storage directly.
    async fn flush(&self, ctx: &FlushContext<'_>) -> Result<FlushOutcome> {
        let generation = ctx.generation(self.column_name())?;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch as ArrowRecordBatch};
        use arrow_schema::Schema as ArrowSchema;
        use lance_arrow::FixedSizeListArrayExt;
        use lance_core::ROW_ID;
        use lance_file::versions as file_versions;
        use lance_file::writer::FileWriterOptions;
        use lance_index::pb;
        use lance_index::vector::DISTANCE_TYPE_KEY;
        use lance_index::vector::SQ_CODE_COLUMN;
        use lance_index::vector::hnsw::HNSW;
        use lance_index::vector::ivf::storage::IVF_METADATA_KEY;
        use lance_index::vector::sq::ScalarQuantizer;
        use lance_index::vector::storage::STORAGE_METADATA_KEY;
        use lance_index::vector::v3::subindex::IvfSubIndex;
        use lance_index::{
            INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY,
            IndexMetadata as IndexMetaSchema,
        };
        use prost::Message;
        use std::ops::Range;
        use std::sync::Arc;

        // Write the index files at the base dataset's storage version (matches
        // the flushed data fragments; 2.2 avoids the v2.1 miniblock chunk cap).
        let storage_version = generation.storage_version;

        let index_uuid = uuid::Uuid::new_v4();
        let index_dir = generation
            .path
            .clone()
            .join("_indices")
            .join(index_uuid.to_string());

        let distance_type = self.distance_type();
        let dim = self.dim();
        if dim == 0 {
            // No vector was ever inserted (e.g. an all-tombstone generation):
            // skip the index, keep the data flush.
            return Ok(FlushOutcome::Skip);
        }
        // Forward-written data: HNSW row ids line up 1:1 with the data file, so
        // no position reversal (pass `None`).
        let Some((hnsw, flat_storage_batch)) = self.to_lance_hnsw(None)? else {
            // Every vector in the generation is null → empty graph; skip the
            // index rather than failing the flush.
            return Ok(FlushOutcome::Skip);
        };

        // Train SQ8 on the full memtable in one pass: learn global min/max
        // from every flushed vector, then quantize all rows in one shot.
        let row_id_col = flat_storage_batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::invalid_input("_rowid missing from HNSW storage batch"))?
            .clone();
        let flat_col = flat_storage_batch
            .column_by_name(lance_index::vector::flat::storage::FLAT_COLUMN)
            .ok_or_else(|| Error::invalid_input("flat column missing from HNSW storage batch"))?
            .clone();
        let flat_fsl = flat_col.as_fixed_size_list();
        let mut sq = ScalarQuantizer::new(8, dim);
        let bounds: Range<f64> = sq.update_bounds::<Float32Type>(flat_fsl)?;
        let sq_codes = sq.transform::<Float32Type>(flat_fsl as &dyn arrow_array::Array)?;

        let storage_schema = ArrowSchema::new(vec![
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(
                SQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::UInt8,
                        true,
                    )),
                    dim as i32,
                ),
                true,
            ),
        ]);
        let storage_batch = ArrowRecordBatch::try_new(
            Arc::new(storage_schema.clone()),
            vec![row_id_col, sq_codes],
        )?;

        // Single-partition IVF for both the storage and graph files. We need
        // *some* centroid because the on-disk read path routes every query
        // through `IvfModel::find_partitions` before HNSW search; that call
        // unwraps `centroids`. With one partition the centroid value is
        // irrelevant for routing — every query goes to partition 0 — so use
        // a zero vector.
        let zero_centroid_values = Float32Array::from(vec![0.0f32; dim]);
        let zero_centroid_fsl =
            FixedSizeListArray::try_new_from_values(zero_centroid_values, dim as i32)?;
        let mut storage_ivf =
            lance_index::vector::ivf::storage::IvfModel::new(zero_centroid_fsl.clone(), None);
        storage_ivf.add_partition(storage_batch.num_rows() as u32);

        let storage_path = index_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        let mut storage_writer = file_versions::create_writer(
            storage_version,
            generation.object_store.create(&storage_path).await?,
            (&storage_schema).try_into()?,
            FileWriterOptions::default(),
        )?;
        storage_writer.write_batch(&storage_batch).await?;

        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());

        // The reader needs the SQ metadata in two forms: a single
        // ScalarQuantizationMetadata under SQ_METADATA_KEY (whole-file path),
        // and a JSON array of per-partition ScalarQuantizationMetadata strings
        // under STORAGE_METADATA_KEY (per-partition path). With one partition
        // we serialize the same value twice.
        let sq_meta = lance_index::vector::sq::storage::ScalarQuantizationMetadata {
            dim,
            num_bits: 8,
            bounds,
        };
        let sq_meta_json = serde_json::to_string(&sq_meta)?;
        storage_writer.add_schema_metadata(
            STORAGE_METADATA_KEY,
            serde_json::to_string(&[&sq_meta_json])?,
        );
        storage_writer.add_schema_metadata(
            lance_index::vector::sq::storage::SQ_METADATA_KEY,
            sq_meta_json,
        );
        storage_writer.finish().await?;

        // Write the HNSW graph batch to index.idx. The graph file uses the
        // same single-partition IVF model with zero centroid for the same
        // reason as the storage file.
        let hnsw_batch = hnsw.to_batch()?;
        let hnsw_metadata_json = hnsw_batch
            .schema_ref()
            .metadata()
            .get(lance_index::vector::hnsw::builder::HNSW_METADATA_KEY)
            .cloned()
            .unwrap_or_default();
        // Force fullzip structural encoding for the graph's List<u32>/List<f32>
        // columns. The HNSW graph has dense level-0 neighbor lists followed by
        // many empty higher-level lists; the v2.x miniblock List codec decodes
        // the row count incorrectly for that shape at scale (the locally-sparse
        // empty block is not captured by the global levels-per-value average),
        // and at 2.1 it also overflows the 32 KiB miniblock cap. Fullzip
        // round-trips it correctly.
        let fullzip_meta = std::collections::HashMap::from([(
            lance_encoding::constants::STRUCTURAL_ENCODING_META_KEY.to_string(),
            lance_encoding::constants::STRUCTURAL_ENCODING_FULLZIP.to_string(),
        )]);
        let index_schema: ArrowSchema = {
            let base = HNSW::schema();
            let fields = base
                .fields()
                .iter()
                .map(|f| {
                    if matches!(f.data_type(), arrow_schema::DataType::List(_)) {
                        Arc::new(f.as_ref().clone().with_metadata(fullzip_meta.clone()))
                    } else {
                        f.clone()
                    }
                })
                .collect::<Vec<_>>();
            ArrowSchema::new(fields)
        };
        let index_path = index_dir.clone().join(INDEX_FILE_NAME);
        let mut index_writer = file_versions::create_writer(
            storage_version,
            generation.object_store.create(&index_path).await?,
            (&index_schema).try_into()?,
            FileWriterOptions::default(),
        )?;
        index_writer.write_batch(&hnsw_batch).await?;

        let mut index_ivf =
            lance_index::vector::ivf::storage::IvfModel::new(zero_centroid_fsl, None);
        index_ivf.add_partition(hnsw_batch.num_rows() as u32);
        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        // The on-disk type string matches Lance's index loader vocabulary —
        // an HNSW sub-index over SQ8-quantized vector storage, registered
        // under the same name as the standard IVF_HNSW_SQ path even though
        // our IVF layer is a single-partition placeholder.
        let index_metadata = IndexMetaSchema {
            index_type: "IVF_HNSW_SQ".to_string(),
            distance_type: distance_type.to_string(),
        };
        index_writer.add_schema_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            serde_json::to_string(&index_metadata)?,
        );
        let ivf_buffer_pos = index_writer
            .add_global_buffer(index_ivf_pb.encode_to_vec().into())
            .await?;
        index_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        // Per-partition HNSW metadata: a JSON array with one entry.
        index_writer.add_schema_metadata(
            HNSW::metadata_key(),
            serde_json::to_string(&[hnsw_metadata_json])?,
        );
        index_writer.finish().await?;

        // Packed the same way index creation does; hand-building the `Any` here
        // produced a `type.googleapis.com/` url no other writer in lance emits.
        let index_details = Some(Arc::new(vector_index_details_default()));
        // The generation is committed by now, so the index records itself
        // against the fragments and version it actually covers.
        let index_meta = IndexMetadata {
            uuid: index_uuid,
            name: generation.name.to_string(),
            fields: vec![self.field_id],
            covering_fields: vec![],
            dataset_version: generation.dataset.version().version,
            fragment_bitmap: Some(generation.dataset.fragment_bitmap.as_ref().clone()),
            index_details,
            base_id: None,
            created_at: Some(chrono::Utc::now()),
            index_version: 1,
            files: None,
        };

        Ok(FlushOutcome::Wrote(Box::new(index_meta)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float32Array, Int32Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;

    fn make_batch(start_id: i32, n: usize, dim: usize) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + n as i32).collect();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for &id in &ids {
            for d in 0..dim {
                flat.push((id as f32 * 0.01) + (d as f32 * 0.001));
            }
        }
        batch_of(ids, flat, dim)
    }

    /// `ids` with their vectors, `dim` values each, laid out row after row.
    fn batch_of(ids: Vec<i32>, flat: Vec<f32>, dim: usize) -> RecordBatch {
        let inner = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new_from_values(inner, dim as i32).unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids)), Arc::new(fsl)]).unwrap()
    }

    /// The graph is pre-allocated from `capacity`, so its footprint is settled
    /// before any row arrives and barely moves as rows do. A memory budget that
    /// samples only row bytes would miss all of it, and one that waited for the
    /// first insert would miss the allocation that insert triggers.
    #[test]
    fn test_resident_bytes_is_preallocated_not_proportional_to_rows() {
        let dim = 8;
        let capacity = 4_000;
        let index = || {
            HnswMemIndex::with_capacity(
                1,
                "vector".to_string(),
                DistanceType::L2,
                HnswBuildParams::default().num_edges(16).ef_construction(64),
                capacity,
                64,
            )
        };

        let untouched = index().resident_bytes();

        let sparse = index();
        sparse.insert(&make_batch(0, 1, dim), 0).unwrap();
        let one_row = sparse.resident_bytes();

        let full = index();
        full.insert(&make_batch(0, capacity, dim), 0).unwrap();
        let all_rows = full.resident_bytes();

        // One row already pays for the whole graph: well over a KB per slot of
        // capacity, and within a small factor of the fully-populated index.
        assert!(
            one_row > capacity * 128,
            "one row should commit the pre-allocated graph, got {one_row} for capacity {capacity}"
        );
        assert!(
            all_rows < one_row * 2,
            "a full index ({all_rows}) should not dwarf a one-row index ({one_row})"
        );

        // The charge is visible before the row that commits it, and close
        // enough to the real thing to admit against. The reservation walks the
        // level ladder in expectation where the graph samples it, so allow a
        // 25% band either way rather than demanding equality.
        assert!(
            untouched.abs_diff(one_row) * 4 < one_row,
            "reserved {untouched} should track the built graph {one_row} before the first insert"
        );
    }

    #[test]
    fn test_index_insert_and_search() {
        let dim = 8;
        let n = 200;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            n,
            64,
        );

        let batch = make_batch(0, n, dim);
        index.insert(&batch, 0).unwrap();
        assert_eq!(index.len(), n);

        let fsl = batch.column_by_name("vector").unwrap().as_fixed_size_list();
        let query_inner =
            Float32Array::from(fsl.value(5).as_primitive::<Float32Type>().values().to_vec());
        let query = FixedSizeListArray::try_new_from_values(query_inner, dim as i32).unwrap();

        let results = index.search(&query, 5, Some(32), u64::MAX).unwrap();
        assert!(!results.is_empty());
        let (best_dist, best_pos) = results[0];
        assert!(
            best_dist < 1e-4,
            "expected near-zero distance, got {}",
            best_dist
        );
        assert_eq!(best_pos, 5);
    }

    /// A dot index ranks the larger product first and reports `1 - dot`, the
    /// distance the base table's dot index reports.
    #[test]
    fn test_dot_index_ranks_the_larger_product_first() {
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::Dot,
            HnswBuildParams::default(),
            2,
            64,
        );
        let batch = batch_of(vec![0, 1], vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0], 3);
        index.insert(&batch, 0).unwrap();

        let query =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 1.0, 1.0]), 3)
                .unwrap();
        // Products 3 and 30.
        assert_eq!(
            index.search(&query, 2, None, u64::MAX).unwrap(),
            vec![(-29.0, 1), (-2.0, 0)]
        );
    }

    #[test]
    fn test_index_insert_batches_combines_hnsw_insert_range() {
        let dim = 8;
        let n = 200;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            n,
            64,
        );

        let first = make_batch(0, 75, dim);
        let second = make_batch(75, 125, dim);
        let stored = vec![
            StoredBatch::new(first, 0, 0),
            StoredBatch::new(second.clone(), 75, 1),
        ];
        index.insert_batches(&stored).unwrap();
        assert_eq!(index.len(), n);

        let fsl = second
            .column_by_name("vector")
            .unwrap()
            .as_fixed_size_list();
        let query_inner =
            Float32Array::from(fsl.value(7).as_primitive::<Float32Type>().values().to_vec());
        let query = FixedSizeListArray::try_new_from_values(query_inner, dim as i32).unwrap();

        let results = index.search(&query, 5, Some(32), u64::MAX).unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().any(|&(dist, pos)| pos == 82 && dist < 1e-4),
            "expected exact row position 82 in top-5 candidates, got {:?}",
            results
        );
    }

    #[test]
    fn test_index_visibility_filter() {
        let dim = 8;
        let n = 50;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            n,
            64,
        );
        let batch = make_batch(0, n, dim);
        index.insert(&batch, 0).unwrap();

        let fsl = batch.column_by_name("vector").unwrap().as_fixed_size_list();
        let query_inner = Float32Array::from(
            fsl.value(40)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec(),
        );
        let query = FixedSizeListArray::try_new_from_values(query_inner, dim as i32).unwrap();

        let results = index.search(&query, 5, Some(32), 10).unwrap();
        for (_, pos) in &results {
            assert!(*pos <= 10);
        }
    }

    #[test]
    fn test_index_empty_search() {
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default(),
            16,
            16,
        );
        let inner = Float32Array::from(vec![0.0; 4]);
        let query = FixedSizeListArray::try_new_from_values(inner, 4).unwrap();
        let results = index.search(&query, 5, None, u64::MAX).unwrap();
        assert!(results.is_empty());
    }

    /// Storage leading the graph must keep its rows.
    ///
    /// `insert_batches` appends storage before it builds and publishes the
    /// graph, so storage can lead. Those rows are unreachable by traversal
    /// either way, but `HNSW::search` brute-forces the storage domain under a
    /// narrow prefilter, so trimming storage to the graph would drop results
    /// this export used to return. The graph still may not exceed storage.
    #[test]
    fn to_lance_hnsw_keeps_storage_rows_the_graph_has_not_reached() {
        let dim = 8;
        let n = 32;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(8).ef_construction(32),
            n * 2,
            4,
        );
        index.insert(&make_batch(0, n, dim), 0).unwrap();

        // Reproduce the interval: storage takes the next batch, the graph does
        // not see it yet.
        let state = index.state.get().expect("state is initialized");
        let extra = make_batch(n as i32, n, dim);
        let vectors = extra
            .column_by_name("vector")
            .unwrap()
            .as_fixed_size_list_opt()
            .unwrap()
            .clone();
        state
            .storage
            .append_batch(Arc::new(vectors), n as u64)
            .unwrap();
        assert!(
            state.storage.committed_len() > state.graph.len(),
            "the test needs storage ahead of the graph"
        );

        let Some((hnsw, storage_batch)) = index.to_lance_hnsw(None).unwrap() else {
            panic!("expected HNSW snapshot");
        };
        assert_eq!(
            storage_batch.num_rows(),
            n * 2,
            "storage keeps every committed row; a narrow prefilter scans them"
        );
        assert_eq!(hnsw.len(), n, "the graph covers only what it indexed");
        assert!(
            hnsw.len() <= storage_batch.num_rows(),
            "the graph must never name a row storage has no vector for"
        );
    }

    #[test]
    fn test_to_lance_hnsw_reverses_row_ids() {
        let dim = 8;
        let n = 32;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(8).ef_construction(32),
            n,
            4,
        );
        let batch = make_batch(0, n, dim);
        index.insert(&batch, 10).unwrap();

        let Some((hnsw, storage_batch)) = index.to_lance_hnsw(Some(100)).unwrap() else {
            panic!("expected HNSW snapshot");
        };
        assert_eq!(hnsw.len(), n);
        let row_ids = storage_batch
            .column_by_name(lance_core::ROW_ID)
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(row_ids.value(0), 89);
        assert_eq!(row_ids.value(n - 1), 58);
    }

    #[test]
    fn test_index_concurrent_insert_and_search() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering as StdOrdering};
        use std::thread;

        let dim = 16;
        let n = 500;
        let index = StdArc::new(HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(8).ef_construction(32),
            n,
            256,
        ));

        let initial = make_batch(-1, 1, dim);
        index.insert(&initial, 0).unwrap();

        let stop = StdArc::new(AtomicBool::new(false));
        let mut reader_handles = Vec::new();
        for _ in 0..4 {
            let index = index.clone();
            let stop = stop.clone();
            reader_handles.push(thread::spawn(move || {
                let inner = Float32Array::from(vec![0.5_f32; dim]);
                let query = FixedSizeListArray::try_new_from_values(inner, dim as i32).unwrap();
                let mut iters = 0u64;
                while !stop.load(StdOrdering::Relaxed) {
                    let _ = index.search(&query, 5, Some(32), u64::MAX).unwrap();
                    iters += 1;
                }
                iters
            }));
        }

        let writer_index = index.clone();
        let writer_handle = thread::spawn(move || {
            for i in 1..(n / 5) {
                let batch = make_batch(i as i32 * 5, 5, dim);
                let row_offset = (i as u64) * 5 + 1;
                writer_index.insert(&batch, row_offset).unwrap();
            }
        });

        writer_handle.join().unwrap();
        stop.store(true, StdOrdering::Release);
        let mut total_reader_iters = 0u64;
        for h in reader_handles {
            total_reader_iters += h.join().unwrap();
        }

        assert!(index.len() > 1);
        assert!(total_reader_iters > 0);
    }

    /// Build a 3-row batch whose middle vector row is null at the list level.
    fn batch_with_null_middle(dim: usize) -> RecordBatch {
        let mut values: Vec<f32> = Vec::new();
        values.extend(std::iter::repeat_n(1.0f32, dim)); // row 0
        values.extend(std::iter::repeat_n(0.0f32, dim)); // row 1 (null placeholder)
        values.extend(std::iter::repeat_n(3.0f32, dim)); // row 2
        let inner = Arc::new(Float32Array::from(values)) as arrow_array::ArrayRef;
        let nulls = arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::from(vec![
            true, false, true,
        ]));
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            inner,
            Some(nulls),
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(fsl)],
        )
        .unwrap()
    }

    #[test]
    fn test_index_skips_null_vector_row() {
        // A null vector row gets no graph node; the surviving rows keep their
        // original positions (the tombstone-enabling fix).
        let dim = 4;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default().num_edges(16).ef_construction(64),
            16,
            16,
        );
        index.insert(&batch_with_null_middle(dim), 0).unwrap();
        assert_eq!(index.len(), 2, "the null row is skipped");

        let query = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![3.0f32; dim]),
            dim as i32,
        )
        .unwrap();
        let results = index.search(&query, 2, Some(16), u64::MAX).unwrap();
        assert!(!results.is_empty());
        let (best_dist, best_pos) = results[0];
        assert!(best_dist < 1e-4, "got {}", best_dist);
        assert_eq!(
            best_pos, 2,
            "row 2 resolves to its original offset, not 1, after the skip"
        );
        assert!(
            results.iter().all(|(_, pos)| *pos != 1),
            "the skipped null row must never be returned"
        );
    }

    #[test]
    fn test_index_all_null_batch_adds_no_nodes() {
        // An all-null batch (e.g. an all-tombstone memtable) inserts cleanly and
        // adds no nodes.
        let dim = 4;
        let index = HnswMemIndex::with_capacity(
            1,
            "vector".to_string(),
            DistanceType::L2,
            HnswBuildParams::default(),
            8,
            8,
        );
        let inner = Arc::new(Float32Array::from(vec![0.0f32; dim * 2])) as arrow_array::ArrayRef;
        let nulls =
            arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::from(vec![false, false]));
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            inner,
            Some(nulls),
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0, 1])), Arc::new(fsl)],
        )
        .unwrap();
        index.insert(&batch, 0).unwrap();
        assert_eq!(index.len(), 0, "no nodes for an all-null batch");
    }
}

/// What an in-memory HNSW index needs before it takes a row.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswParams {
    /// The metric the base index was built with, so the memtable and the base
    /// table produce comparable distances.
    pub distance_type: DistanceType,
    /// Graph parameters, defaulted unless the writer overrode them.
    pub build_params: HnswBuildParams,
}

/// Declares the built-in HNSW memtable index.
#[derive(Debug, Default)]
pub struct HnswMemIndexPlugin;

#[async_trait::async_trait]
impl MemIndexPlugin for HnswMemIndexPlugin {
    fn name(&self) -> &str {
        "Hnsw"
    }

    fn details_message(&self) -> &str {
        "VectorIndexDetails"
    }

    fn flush_index_type(&self) -> IndexType {
        IndexType::IvfHnswSq
    }

    fn training_criteria(&self) -> TrainingCriteria {
        // Never consulted: the flush writes the graph itself rather than
        // handing rows to a trainer.
        TrainingCriteria::new(TrainingOrdering::None)
    }

    async fn resolve(&self, ctx: &ResolveContext<'_>) -> Result<ResolvedIndex> {
        use crate::index::DatasetIndexInternalExt;
        use lance_index::metrics::NoOpMetricsCollector;

        let [column] = ctx.columns else {
            return Err(Error::invalid_input(format!(
                "vector index '{}' must cover exactly one column",
                ctx.name
            )));
        };

        // Inherit the base table's metric so the in-memory index and the base
        // index produce comparable distances. The recorded details state it,
        // and for an index that covers nothing they are the only source: it
        // carries its settings with no file to open. Opening the index is the
        // fallback for an entry whose details do not decode. Surface the
        // failure rather than defaulting to L2 — flushed `IVF_HNSW_SQ` files
        // bake the metric into their metadata, so a wrong default would be
        // durable corruption.
        let recorded = ctx
            .index_meta
            .index_details
            .as_deref()
            .and_then(crate::index::vector::details::vector_params_from_details)
            .map(|params| params.metric_type);
        let distance_type = match recorded {
            Some(distance_type) => distance_type,
            None => ctx
                .dataset
                .open_vector_index(column, &ctx.index_meta.uuid, &NoOpMetricsCollector)
                .await
                .map_err(|e| {
                    Error::invalid_input(format!(
                        "Failed to open base vector index '{}' to inherit distance type: {}",
                        ctx.name, e
                    ))
                })?
                .metric_type(),
        };

        Ok(ResolvedIndex::with_params(
            ctx.columns.to_vec(),
            HnswParams {
                distance_type,
                build_params: ctx
                    .overrides::<HnswBuildParams>()?
                    .cloned()
                    .unwrap_or_else(mem_wal_hnsw_default),
            },
        ))
    }

    fn validate(&self, ctx: &MemIndexBuildContext<'_>) -> Result<()> {
        use arrow_schema::DataType;

        let (column, _) = ctx.single_column()?;
        ctx.check_top_level_columns()?;
        let field = ctx
            .schema
            .field(column)
            .expect("check_columns_resolve accepted the column");
        match field.data_type() {
            DataType::FixedSizeList(item, dim) => {
                if item.data_type() != &DataType::Float32 {
                    return Err(Error::invalid_input(format!(
                        "HNSW index '{}' requires a FixedSizeList<Float32> column; column \
                         '{column}' has item type {:?}",
                        ctx.name,
                        item.data_type()
                    )));
                }
                // `HnswMemIndex.dim` is a placeholder until the first batch
                // pins it, so a zero-width vector would otherwise only surface
                // at insert time — on already-durable data.
                if dim <= 0 {
                    return Err(Error::invalid_input(format!(
                        "HNSW index '{}' requires a vector dimension > 0; column '{column}' has \
                         dimension {dim}",
                        ctx.name,
                    )));
                }
                Ok(())
            }
            other => Err(Error::invalid_input(format!(
                "HNSW index '{}' requires a FixedSizeList<Float32> column; column '{column}' is \
                 {other:?}",
                ctx.name
            ))),
        }
    }

    fn create(&self, ctx: &MemIndexBuildContext<'_>) -> Result<Arc<dyn MemIndex>> {
        let (column, field_id) = ctx.single_column()?;
        let params = ctx.params::<HnswParams>()?;
        Ok(Arc::new(HnswMemIndex::with_capacity(
            field_id,
            column.to_string(),
            params.distance_type,
            params.build_params.clone(),
            ctx.capacity_rows,
            ctx.capacity_batches,
        )))
    }
}
