// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{borrow::Cow, sync::Arc};

use super::index::FlatMetadata;
use crate::frag_reuse::{FragReuseIndex, FragReuseIndexHandle};
use crate::scalar::RowIdRemapper;
use crate::vector::PART_ID_COLUMN;
use crate::vector::quantizer::QuantizerStorage;
use crate::vector::storage::{DistCalculator, VectorStore, remap_row_ids_by_name};
use crate::vector::utils::do_prefetch;
use arrow::array::AsArray;
use arrow::compute::concat_batches;
use arrow::datatypes::{Float16Type, Float64Type, UInt8Type};
use arrow_array::ArrowPrimitiveType;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array,
    types::{Float32Type, UInt64Type},
};
use arrow_schema::{DataType, SchemaRef};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_linalg::distance::hamming::hamming;
use lance_linalg::distance::{Cosine, DistanceType, Dot, L2, Normalize, norm_l2_fsl};

pub const FLAT_COLUMN: &str = "flat";

/// Flat storage's internal (non-covering) column names: the row id, the flat vector
/// column, and the legacy `__ivf_part_id` column (left in pre-#3606 single-partition
/// builds; unlike PQ, flat storage does not drop it at construction). Every other
/// column in a flat storage schema is a user-declared covering ("included") column.
/// The single authority for this set -- the distributed shard merger classifies from
/// it too, so a new internal column added here is excluded everywhere at once.
pub const FLAT_INTERNAL_COLUMNS: &[&str] = &[ROW_ID, FLAT_COLUMN, PART_ID_COLUMN];

/// Per-vector L2 norms cached for Cosine distance, so `cosine_with_norms` can
/// skip recomputing each stored vector's norm per comparison. `None` for other
/// metrics and for value types without a norm kernel.
fn cosine_norms_cache(
    vectors: &FixedSizeListArray,
    distance_type: DistanceType,
) -> Option<Arc<Float32Array>> {
    if distance_type != DistanceType::Cosine {
        return None;
    }
    norm_l2_fsl(vectors).ok().map(Arc::new)
}

/// All data are stored in memory
#[derive(Debug, Clone)]
pub struct FlatFloatStorage {
    metadata: FlatMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    pub(super) row_ids: Arc<UInt64Array>,
    vectors: Arc<FixedSizeListArray>,
    /// Per-vector L2 norms for Cosine. `None` for other metrics.
    norms: Option<Arc<Float32Array>>,
}

impl DeepSizeOf for FlatFloatStorage {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        let mut size = self.batch.get_array_memory_size();
        if let Some(norms) = &self.norms {
            size += norms.get_array_memory_size();
        }
        size
    }
}

#[async_trait::async_trait]
impl QuantizerStorage for FlatFloatStorage {
    type Metadata = FlatMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::try_from_batch_with_remapper(batch, metadata, distance_type, frag_reuse_index)
    }

    fn try_from_batch_with_remapper(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let batch = if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            remap_row_ids_by_name(batch, frag_reuse_index_ref.as_ref())?
        } else {
            batch
        };

        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        let norms = cosine_norms_cache(&vectors, distance_type);
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            vectors,
            norms,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        _: &V1FileReader,
        _: std::ops::Range<usize>,
        _: DistanceType,
        _: &Self::Metadata,
        _: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

impl FlatFloatStorage {
    // used for only testing
    pub fn new(vectors: FixedSizeListArray, distance_type: DistanceType) -> Self {
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64));
        let vectors = Arc::new(vectors);

        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids.clone() as ArrayRef, true),
            (FLAT_COLUMN, vectors.clone() as ArrayRef, true),
        ])
        .unwrap();

        let norms = cosine_norms_cache(&vectors, distance_type);
        Self {
            metadata: FlatMetadata {
                dim: vectors.value_length() as usize,
            },
            batch,
            distance_type,
            row_ids,
            vectors,
            norms,
        }
    }

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.value(id as usize)
    }
}

impl VectorStore for FlatFloatStorage {
    type DistanceCalculator<'a> = FlatFloatDistanceCalc<'a>;

    /// Flat storage is `[_rowid, flat, <included cols...>]`.
    const INTERNAL_COLUMNS: &'static [&'static str] = FLAT_INTERNAL_COLUMNS;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok([self.batch.clone()].into_iter())
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch])?;
        let mut storage = self.clone();
        storage.row_ids = Arc::new(
            new_batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        storage.vectors = Arc::new(
            new_batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        storage.norms = cosine_norms_cache(&storage.vectors, storage.distance_type);
        storage.batch = new_batch;
        Ok(storage)
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.values()[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        let norms = self.norms.as_ref().map(|n| n.values().as_ref());
        Self::DistanceCalculator::new(self.vectors.as_ref(), query, self.distance_type, norms)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        let norms = self.norms.as_ref().map(|n| n.values().as_ref());
        Self::DistanceCalculator::new_from_id(self.vectors.as_ref(), id, self.distance_type, norms)
    }
}

/// All data are stored in memory
#[derive(Debug, Clone)]
pub struct FlatBinStorage {
    metadata: FlatMetadata,
    batch: RecordBatch,
    distance_type: DistanceType,

    // helper fields
    pub(super) row_ids: Arc<UInt64Array>,
    vectors: Arc<FixedSizeListArray>,
}

impl DeepSizeOf for FlatBinStorage {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
    }
}

#[async_trait::async_trait]
impl QuantizerStorage for FlatBinStorage {
    type Metadata = FlatMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::try_from_batch_with_remapper(batch, metadata, distance_type, frag_reuse_index)
    }

    fn try_from_batch_with_remapper(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let batch = if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            remap_row_ids_by_name(batch, frag_reuse_index_ref.as_ref())?
        } else {
            batch
        };

        let row_ids = Arc::new(
            batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        let vectors = Arc::new(
            batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        Ok(Self {
            metadata: metadata.clone(),
            batch,
            distance_type,
            row_ids,
            vectors,
        })
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    async fn load_partition(
        _: &V1FileReader,
        _: std::ops::Range<usize>,
        _: DistanceType,
        _: &Self::Metadata,
        _: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        unimplemented!("Flat will be used in new index builder which doesn't require this")
    }
}

impl FlatBinStorage {
    // used for only testing
    pub fn new(vectors: FixedSizeListArray, distance_type: DistanceType) -> Self {
        let row_ids = Arc::new(UInt64Array::from_iter_values(0..vectors.len() as u64));
        let vectors = Arc::new(vectors);

        let batch = RecordBatch::try_from_iter_with_nullable(vec![
            (ROW_ID, row_ids.clone() as ArrayRef, true),
            (FLAT_COLUMN, vectors.clone() as ArrayRef, true),
        ])
        .unwrap();

        Self {
            metadata: FlatMetadata {
                dim: vectors.value_length() as usize,
            },
            batch,
            distance_type,
            row_ids,
            vectors,
        }
    }

    pub fn vector(&self, id: u32) -> ArrayRef {
        self.vectors.value(id as usize)
    }
}

impl VectorStore for FlatBinStorage {
    type DistanceCalculator<'a> = FlatDistanceCal<'a, UInt8Type>;

    /// Flat storage is `[_rowid, flat, <included cols...>]`.
    const INTERNAL_COLUMNS: &'static [&'static str] = FLAT_INTERNAL_COLUMNS;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok([self.batch.clone()].into_iter())
    }

    fn append_batch(&self, batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        // TODO: use chunked storage
        let new_batch = concat_batches(&batch.schema(), vec![&self.batch, &batch])?;
        let mut storage = self.clone();
        storage.row_ids = Arc::new(
            new_batch
                .column_by_name(ROW_ID)
                .ok_or(Error::schema(format!("column {} not found", ROW_ID)))?
                .as_primitive::<UInt64Type>()
                .clone(),
        );
        storage.vectors = Arc::new(
            new_batch
                .column_by_name(FLAT_COLUMN)
                .ok_or(Error::schema("column flat not found".to_string()))?
                .as_fixed_size_list()
                .clone(),
        );
        storage.batch = new_batch;
        Ok(storage)
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        self.row_ids.values()[id as usize]
    }

    fn row_ids(&self) -> impl Iterator<Item = &u64> {
        self.row_ids.values().iter()
    }

    fn dist_calculator(&self, query: ArrayRef, _dist_q_c: f32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new_binary(self.vectors.as_ref(), query, self.distance_type)
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        Self::DistanceCalculator::new_binary_from_id(self.vectors.as_ref(), id, self.distance_type)
    }
}

pub struct FlatDistanceCal<'a, T: ArrowPrimitiveType> {
    vectors: &'a [T::Native],
    query: Cow<'a, [T::Native]>,
    dimension: usize,
    query_norm: Option<f32>,
    vector_norms: Option<&'a [f32]>,
    #[allow(clippy::type_complexity)]
    distance_fn: fn(&[T::Native], &[T::Native]) -> f32,
}

impl<'a, T> FlatDistanceCal<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: L2 + Cosine + Dot,
{
    fn new(
        vectors: &'a FixedSizeListArray,
        query: ArrayRef,
        distance_type: DistanceType,
        vector_norms: Option<&'a [f32]>,
    ) -> Self {
        debug_assert!(
            vector_norms.is_none_or(|norms| norms.len() == vectors.len()),
            "expected one cached norm per vector"
        );
        // Gained significant performance improvement by using strong typed primitive slice.
        let flat_array = vectors.values().as_primitive::<T>();
        let dimension = vectors.value_length() as usize;
        let query: Cow<'a, [T::Native]> = Cow::Owned(query.as_primitive::<T>().values().to_vec());
        // Only cache the query norm alongside the stored norms.
        let query_norm = (distance_type == DistanceType::Cosine && vector_norms.is_some())
            .then(|| T::Native::norm_l2(query.as_ref()));
        Self {
            vectors: flat_array.values(),
            query,
            dimension,
            query_norm,
            vector_norms,
            distance_fn: distance_type.func(),
        }
    }

    fn new_from_id(
        vectors: &'a FixedSizeListArray,
        id: u32,
        distance_type: DistanceType,
        vector_norms: Option<&'a [f32]>,
    ) -> Self {
        debug_assert!(
            vector_norms.is_none_or(|norms| norms.len() == vectors.len()),
            "expected one cached norm per vector"
        );
        let flat_array = vectors.values().as_primitive::<T>();
        let dimension = vectors.value_length() as usize;
        let vectors = flat_array.values();
        let id = id as usize;
        let query: Cow<'a, [T::Native]> =
            Cow::Borrowed(&vectors[dimension * id..dimension * (id + 1)]);
        // The query is stored vector `id`, so reuse its cached norm.
        let query_norm = match (distance_type, vector_norms) {
            (DistanceType::Cosine, Some(norms)) => Some(norms[id]),
            _ => None,
        };
        Self {
            vectors,
            query,
            dimension,
            query_norm,
            vector_norms,
            distance_fn: distance_type.func(),
        }
    }
}

impl<'a> FlatDistanceCal<'a, UInt8Type> {
    fn new_binary(
        vectors: &'a FixedSizeListArray,
        query: ArrayRef,
        _distance_type: DistanceType,
    ) -> Self {
        // Gained significant performance improvement by using strong typed primitive slice.
        // TODO: to support other data types other than `f32`, make FlatDistanceCal a generic struct.
        let flat_array = vectors.values().as_primitive::<UInt8Type>();
        let dimension = vectors.value_length() as usize;
        Self {
            vectors: flat_array.values(),
            query: Cow::Owned(query.as_primitive::<UInt8Type>().values().to_vec()),
            dimension,
            query_norm: None,
            vector_norms: None,
            distance_fn: hamming,
        }
    }

    fn new_binary_from_id(
        vectors: &'a FixedSizeListArray,
        id: u32,
        _distance_type: DistanceType,
    ) -> Self {
        let flat_array = vectors.values().as_primitive::<UInt8Type>();
        let dimension = vectors.value_length() as usize;
        let vectors = flat_array.values();
        let id = id as usize;
        Self {
            vectors,
            query: Cow::Borrowed(&vectors[dimension * id..dimension * (id + 1)]),
            dimension,
            query_norm: None,
            vector_norms: None,
            distance_fn: hamming,
        }
    }
}

impl<T: ArrowPrimitiveType> FlatDistanceCal<'_, T> {
    #[inline]
    fn get_vector(&self, id: u32) -> &[T::Native] {
        &self.vectors[self.dimension * id as usize..self.dimension * (id + 1) as usize]
    }
}

impl<T: ArrowPrimitiveType> DistCalculator for FlatDistanceCal<'_, T>
where
    T::Native: Cosine,
{
    #[inline]
    fn distance(&self, id: u32) -> f32 {
        let query = self.query.as_ref();
        let vector = self.get_vector(id);
        match (self.query_norm, self.vector_norms) {
            (Some(x_norm), Some(norms)) => {
                T::Native::cosine_with_norms(query, x_norm, norms[id as usize], vector)
            }
            _ => (self.distance_fn)(query, vector),
        }
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        let query = self.query.as_ref();
        match (self.query_norm, self.vector_norms) {
            (Some(x_norm), Some(norms)) => {
                debug_assert_eq!(
                    norms.len(),
                    self.vectors.len() / self.dimension,
                    "cached norms must cover every vector, otherwise `zip` silently truncates"
                );
                self.vectors
                    .chunks_exact(self.dimension)
                    .zip(norms)
                    .map(|(vector, &y_norm)| {
                        T::Native::cosine_with_norms(query, x_norm, y_norm, vector)
                    })
                    .collect()
            }
            _ => self
                .vectors
                .chunks_exact(self.dimension)
                .map(|vector| (self.distance_fn)(query, vector))
                .collect(),
        }
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        let vector = self.get_vector(id);
        do_prefetch(vector.as_ptr_range())
    }
}

pub enum FlatFloatDistanceCalc<'a> {
    Float16(FlatDistanceCal<'a, Float16Type>),
    Float32(FlatDistanceCal<'a, Float32Type>),
    Float64(FlatDistanceCal<'a, Float64Type>),
}

impl<'a> FlatFloatDistanceCalc<'a> {
    fn new(
        vectors: &'a FixedSizeListArray,
        query: ArrayRef,
        distance_type: DistanceType,
        vector_norms: Option<&'a [f32]>,
    ) -> Self {
        match vectors.value_type() {
            DataType::Float16 => Self::Float16(FlatDistanceCal::<Float16Type>::new(
                vectors,
                query,
                distance_type,
                vector_norms,
            )),
            DataType::Float32 => Self::Float32(FlatDistanceCal::<Float32Type>::new(
                vectors,
                query,
                distance_type,
                vector_norms,
            )),
            DataType::Float64 => Self::Float64(FlatDistanceCal::<Float64Type>::new(
                vectors,
                query,
                distance_type,
                vector_norms,
            )),
            dt => panic!("flat float storage does not support data type {dt}"),
        }
    }

    fn new_from_id(
        vectors: &'a FixedSizeListArray,
        id: u32,
        distance_type: DistanceType,
        vector_norms: Option<&'a [f32]>,
    ) -> Self {
        match vectors.value_type() {
            DataType::Float16 => Self::Float16(FlatDistanceCal::<Float16Type>::new_from_id(
                vectors,
                id,
                distance_type,
                vector_norms,
            )),
            DataType::Float32 => Self::Float32(FlatDistanceCal::<Float32Type>::new_from_id(
                vectors,
                id,
                distance_type,
                vector_norms,
            )),
            DataType::Float64 => Self::Float64(FlatDistanceCal::<Float64Type>::new_from_id(
                vectors,
                id,
                distance_type,
                vector_norms,
            )),
            dt => panic!("flat float storage does not support data type {dt}"),
        }
    }
}

impl DistCalculator for FlatFloatDistanceCalc<'_> {
    fn distance(&self, id: u32) -> f32 {
        match self {
            Self::Float16(calc) => calc.distance(id),
            Self::Float32(calc) => calc.distance(id),
            Self::Float64(calc) => calc.distance(id),
        }
    }

    fn distance_all(&self, k_hint: usize) -> Vec<f32> {
        match self {
            Self::Float16(calc) => calc.distance_all(k_hint),
            Self::Float32(calc) => calc.distance_all(k_hint),
            Self::Float64(calc) => calc.distance_all(k_hint),
        }
    }

    fn prefetch(&self, id: u32) {
        match self {
            Self::Float16(calc) => calc.prefetch(id),
            Self::Float32(calc) => calc.prefetch(id),
            Self::Float64(calc) => calc.prefetch(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float16Array, Float32Array, Float64Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    fn make_f16_storage() -> FlatFloatStorage {
        let values = Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(6.0),
        ]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    fn make_f64_storage() -> FlatFloatStorage {
        let values = Float64Array::from(vec![1.0, 2.0, 4.0, 6.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    // Pre-#3606 `num_partitions=1` builds left `__ivf_part_id` in the flat
    // storage file. It must never be classified as a covering column --
    // otherwise search emits a column the exec's declared schema (from
    // `IndexMetadata.covering_fields`, empty for those indexes) lacks,
    // desyncing the two and failing every query on that legacy index.
    #[test]
    fn test_flat_covering_excludes_legacy_part_id_column() {
        use crate::vector::PART_ID_COLUMN;
        const DIM: i32 = 8;
        const N: usize = 8;

        let part_id_batch = |vectors: ArrayRef| {
            RecordBatch::try_from_iter_with_nullable(vec![
                (
                    ROW_ID,
                    Arc::new(UInt64Array::from_iter_values(0..N as u64)) as ArrayRef,
                    true,
                ),
                (FLAT_COLUMN, vectors, true),
                (
                    PART_ID_COLUMN,
                    Arc::new(arrow_array::UInt32Array::from_iter_values(
                        (0..N).map(|_| 0),
                    )) as ArrayRef,
                    true,
                ),
            ])
            .unwrap()
        };
        let meta = FlatMetadata { dim: DIM as usize };

        let floats =
            arrow_array::Float32Array::from_iter_values((0..N * DIM as usize).map(|v| v as f32));
        let float_storage = FlatFloatStorage::try_from_batch(
            part_id_batch(Arc::new(
                FixedSizeListArray::try_new_from_values(floats, DIM).unwrap(),
            )),
            &meta,
            DistanceType::L2,
            None,
        )
        .unwrap();
        assert!(
            float_storage.covering_field_indices().is_empty(),
            "legacy __ivf_part_id must not be classified as covering (float storage)"
        );

        let codes =
            arrow_array::UInt8Array::from_iter_values((0..N * DIM as usize).map(|v| v as u8));
        let bin_storage = FlatBinStorage::try_from_batch(
            part_id_batch(Arc::new(
                FixedSizeListArray::try_new_from_values(codes, DIM).unwrap(),
            )),
            &meta,
            DistanceType::L2,
            None,
        )
        .unwrap();
        assert!(
            bin_storage.covering_field_indices().is_empty(),
            "legacy __ivf_part_id must not be classified as covering (binary storage)"
        );
    }

    #[test]
    fn test_flat_float_storage_distance_f16() {
        let storage = make_f16_storage();
        let query: ArrayRef = Arc::new(Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ]));

        let calc = storage.dist_calculator(query, 0.0);
        let distances = calc.distance_all(2);

        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 25.0).abs() < 1e-4);
    }

    #[test]
    fn test_flat_float_storage_distance_f64() {
        let storage = make_f64_storage();
        let query: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));

        let calc = storage.dist_calculator(query, 0.0);
        let distances = calc.distance_all(2);

        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 25.0).abs() < 1e-6);
    }

    fn make_flat_test_batch(vectors: FixedSizeListArray, first_row_id: u64) -> RecordBatch {
        let num_rows = vectors.len() as u64;
        RecordBatch::try_from_iter(vec![
            (
                ROW_ID,
                Arc::new(UInt64Array::from_iter_values(
                    first_row_id..first_row_id + num_rows,
                )) as ArrayRef,
            ),
            (FLAT_COLUMN, Arc::new(vectors) as ArrayRef),
        ])
        .unwrap()
    }

    /// Assert that the cached-norm Cosine path agrees with the uncached
    /// `Cosine::cosine` reference for both `distance` and `distance_all`.
    fn assert_cosine_matches_uncached<T>(vectors: FixedSizeListArray, query: ArrayRef)
    where
        T: ArrowPrimitiveType,
        T::Native: L2 + Cosine + Dot,
    {
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<T>().values().to_vec();
        let query_values = query.as_primitive::<T>().values().to_vec();

        let storage = FlatFloatStorage::new(vectors, DistanceType::Cosine);
        let calc = storage.dist_calculator(query, 0.0);
        let all = calc.distance_all(storage.len());
        assert_eq!(all.len(), storage.len());

        for (id, vector) in values.chunks_exact(dim).enumerate() {
            let expected = T::Native::cosine(&query_values, vector);
            assert!(
                (all[id] - expected).abs() < 1e-5,
                "distance_all[{id}]: {} vs uncached {expected}",
                all[id]
            );
            assert!(
                (calc.distance(id as u32) - expected).abs() < 1e-5,
                "distance({id}): {} vs uncached {expected}",
                calc.distance(id as u32)
            );
        }
    }

    #[test]
    fn test_cosine_cached_norms_match_uncached_cosine() {
        // Caching the stored vectors' norms must not change the distances that
        // `Cosine::cosine` would compute inline, for any supported value type.
        let f32_vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            4,
        )
        .unwrap();
        assert_cosine_matches_uncached::<Float32Type>(
            f32_vectors,
            Arc::new(Float32Array::from(vec![0.5, 0.5, 0.5, 0.5])),
        );

        let f16_values: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let f16_vectors =
            FixedSizeListArray::try_new_from_values(Float16Array::from(f16_values), 4).unwrap();
        assert_cosine_matches_uncached::<Float16Type>(
            f16_vectors,
            Arc::new(Float16Array::from(vec![f16::from_f32(0.5); 4])),
        );

        let f64_vectors = FixedSizeListArray::try_new_from_values(
            Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            4,
        )
        .unwrap();
        assert_cosine_matches_uncached::<Float64Type>(
            f64_vectors,
            Arc::new(Float64Array::from(vec![0.5, 0.5, 0.5, 0.5])),
        );
    }

    #[rstest]
    #[case::l2(DistanceType::L2, false)]
    #[case::dot(DistanceType::Dot, false)]
    #[case::cosine(DistanceType::Cosine, true)]
    fn test_norms_cached_only_for_cosine(
        #[case] distance_type: DistanceType,
        #[case] expect_norms: bool,
    ) {
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 4).unwrap();
        let batch = make_flat_test_batch(vectors.clone(), 0);
        let metadata = FlatMetadata { dim: 4 };

        let loaded =
            FlatFloatStorage::try_from_batch(batch, &metadata, distance_type, None).unwrap();
        assert_eq!(loaded.distance_type(), distance_type);
        assert_eq!(loaded.norms.is_some(), expect_norms);

        // `new` is on the build path (see `HnswBuilder::build`) and must agree.
        let built = FlatFloatStorage::new(vectors, distance_type);
        assert_eq!(built.norms.is_some(), expect_norms);
    }

    #[test]
    fn test_append_batch_recomputes_norms() {
        // A stale norms cache would silently truncate `distance_all` via `zip`,
        // so appending must extend it to cover the new rows.
        let head = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]),
            4,
        )
        .unwrap();
        let storage = FlatFloatStorage::new(head, DistanceType::Cosine);
        assert_eq!(storage.norms.as_ref().unwrap().len(), 1);

        let tail = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            4,
        )
        .unwrap();
        let appended = storage
            .append_batch(make_flat_test_batch(tail, 1), FLAT_COLUMN)
            .unwrap();

        assert_eq!(appended.len(), 3);
        let norms = appended.norms.as_ref().expect("norms kept after append");
        assert_eq!(norms.len(), 3);
        for (id, expected) in [
            (0, 30.0f32.sqrt()),
            (1, 174.0f32.sqrt()),
            (2, 446.0f32.sqrt()),
        ] {
            assert!(
                (norms.value(id) - expected).abs() < 1e-4,
                "norms[{id}]: {} vs {expected}",
                norms.value(id)
            );
        }

        let query: ArrayRef = Arc::new(Float32Array::from(vec![0.5, 0.5, 0.5, 0.5]));
        assert_eq!(
            appended.dist_calculator(query, 0.0).distance_all(3).len(),
            3
        );
    }

    #[test]
    fn test_dist_calculator_from_id_reuses_cached_norms() {
        // HNSW builds graphs through `dist_calculator_from_id`; the query is a
        // stored vector, so its norm comes from the cache instead of `norm_l2`.
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 4).unwrap();
        let storage = FlatFloatStorage::new(vectors, DistanceType::Cosine);

        let calc = storage.dist_calculator_from_id(1);
        // Self-distance of a vector against itself is ~0 under cosine.
        assert!(calc.distance(1).abs() < 1e-6, "got {}", calc.distance(1));

        let expected = f32::cosine(&[5.0, 6.0, 7.0, 8.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!(
            (calc.distance(0) - expected).abs() < 1e-6,
            "{} vs {expected}",
            calc.distance(0)
        );
    }

    #[test]
    fn normalized_f16_cosine_keeps_self_match_at_zero_lower_bound() {
        let values = Float16Array::from(vec![
            f16::from_f32(7.0),
            f16::from_f32(47.0),
            f16::from_f32(13.0),
        ]);
        let raw = FixedSizeListArray::try_new_from_values(values, 3).unwrap();
        let normalized = lance_linalg::kernels::normalize_fsl(&raw).unwrap();
        let query = normalized.value(0);
        let batch = make_flat_test_batch(normalized, 0);
        let storage = FlatFloatStorage::try_from_batch(
            batch,
            &FlatMetadata { dim: 3 },
            DistanceType::Cosine,
            None,
        )
        .unwrap();

        let distance = storage.dist_calculator(query, 0.0).distance(0);
        assert!(
            distance >= 0.0,
            "lower_bound=0 would drop self-match: {distance}"
        );
    }

    #[test]
    fn test_deep_size_accounts_for_cached_norms() {
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 4).unwrap();

        let cosine = FlatFloatStorage::new(vectors.clone(), DistanceType::Cosine);
        let l2 = FlatFloatStorage::new(vectors, DistanceType::L2);
        assert!(
            cosine.deep_size_of() > l2.deep_size_of(),
            "cosine storage must report the cached norms: {} vs {}",
            cosine.deep_size_of(),
            l2.deep_size_of()
        );
    }
}
