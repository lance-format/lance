// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Product Quantization storage
//!
//! Used as storage backend for Graph based algorithms.

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::{
    cmp::min,
    collections::BinaryHeap,
    sync::{Arc, OnceLock},
};

use arrow::datatypes::{self, UInt8Type};
use arrow_array::{ArrayRef, ArrowPrimitiveType, PrimitiveArray};
use arrow_array::{
    FixedSizeListArray, RecordBatch, UInt8Array, UInt64Array,
    cast::AsArray,
    types::{Float32Type, UInt64Type},
};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, ROW_ID, Result};
use lance_file::versions::v1::{
    reader::FileReader as V1FileReader, writer::FileWriter as V1FileWriter,
};
use lance_io::{object_store::ObjectStore, utils::read_message};
use lance_linalg::distance::{Cosine, DistanceType, Dot, L2};
use lance_linalg::simd::dist_table::{
    filter_4bit_dist_table_transposed, sum_4bit_dist_table_transposed,
};
use lance_table::utils::LanceIteratorExtension;
use lance_table::{format::SelfDescribingFileReader, io::manifest::ManifestDescribing};
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};

use super::ProductQuantizer;
use super::distance::{
    bounded_4bit_scores, build_distance_table_dot, build_distance_table_l2, compute_pq_distance,
    compute_pq_distance_4bit_row, compute_pq_distance_4bit_rows, quantize_4bit_distance_table,
};
use crate::frag_reuse::{FragReuseIndex, FragReuseIndexHandle};
use crate::scalar::RowIdRemapper;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::{
    INDEX_METADATA_SCHEMA_KEY, IndexMetadata, pb,
    vector::{
        PQ_CODE_COLUMN,
        pq::transform::PQTransformer,
        quantizer::{QuantizerMetadata, QuantizerStorage},
        storage::{DistCalculator, VectorStore, accumulate_distances_into_heap},
        transform::Transformer,
    },
};

pub const PQ_METADATA_KEY: &str = "lance:pq";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizationMetadata {
    pub codebook_position: usize,
    pub nbits: u32,
    pub num_sub_vectors: usize,
    pub dimension: usize,

    #[serde(skip)]
    pub codebook: Option<FixedSizeListArray>,

    // empty for v1 format
    // used for v3 format
    // deprecated in later version
    pub codebook_tensor: Vec<u8>,
    pub transposed: bool,
}

impl DeepSizeOf for ProductQuantizationMetadata {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.codebook
            .as_ref()
            .map(|codebook| (codebook as &dyn arrow_array::Array).deep_size_of_children(context))
            .unwrap_or(0)
    }
}

impl PartialEq for ProductQuantizationMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.num_sub_vectors == other.num_sub_vectors
            && self.nbits == other.nbits
            && self.dimension == other.dimension
            && self.codebook == other.codebook
    }
}

#[async_trait]
impl QuantizerMetadata for ProductQuantizationMetadata {
    fn is_transposed(&self) -> bool {
        self.transposed
    }

    fn buffer_index(&self) -> Option<u32> {
        if self.codebook_position > 0 {
            // the global buffer index starts from 1
            Some(self.codebook_position as u32)
        } else {
            None
        }
    }

    fn set_buffer_index(&mut self, index: u32) {
        self.codebook_position = index as usize;
    }

    fn parse_buffer(&mut self, bytes: Bytes) -> Result<()> {
        debug_assert!(!bytes.is_empty());
        debug_assert!(self.codebook.is_none());
        let codebook_tensor: pb::Tensor = pb::Tensor::decode(bytes)?;
        self.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(())
    }

    fn extra_metadata(&self) -> Result<Option<Bytes>> {
        if let Some(codebook) = &self.codebook {
            let codebook_tensor: pb::Tensor = pb::Tensor::try_from(codebook)?;
            let mut bytes = BytesMut::new();
            codebook_tensor.encode(&mut bytes)?;
            Ok(Some(bytes.freeze()))
        } else if !self.codebook_tensor.is_empty() {
            // Legacy format: codebook is stored inline in the metadata JSON.
            // Return it as-is; it's already a protobuf-encoded Tensor that
            // parse_buffer() can handle.
            Ok(Some(Bytes::from(self.codebook_tensor.clone())))
        } else {
            Ok(None)
        }
    }

    async fn load(reader: &V1FileReader) -> Result<Self> {
        let metadata = reader
            .schema()
            .metadata
            .get(PQ_METADATA_KEY)
            .ok_or(Error::index(format!(
                "Reading PQ storage: metadata key {} not found",
                PQ_METADATA_KEY
            )))?;
        let mut metadata: Self = serde_json::from_str(metadata)
            .map_err(|_| Error::index(format!("Failed to parse PQ metadata: {}", metadata)))?;

        debug_assert!(metadata.codebook.is_none());
        debug_assert!(metadata.codebook_tensor.is_empty());

        let codebook_tensor: pb::Tensor =
            read_message(reader.object_reader.as_ref(), metadata.codebook_position).await?;
        metadata.codebook = Some(FixedSizeListArray::try_from(&codebook_tensor)?);
        Ok(metadata)
    }
}

/// Product Quantization Storage
///
/// It stores PQ code, as well as the row ID to the original vectors.
///
/// It is possible to store additional metadata to accelerate filtering later.
#[derive(Clone, Debug)]
pub struct ProductQuantizationStorage {
    metadata: ProductQuantizationMetadata,
    distance_type: DistanceType,
    batch: RecordBatch,

    // For easy access
    pq_code: Arc<UInt8Array>,
    row_ids: Arc<UInt64Array>,
    pairwise_distance_table: Arc<OnceLock<Vec<f32>>>,
}

impl DeepSizeOf for ProductQuantizationStorage {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.batch.deep_size_of_children(context)
            + self
                .metadata
                .codebook
                .as_ref()
                .map(|codebook| {
                    (codebook as &dyn arrow_array::Array).deep_size_of_children(context)
                })
                .unwrap_or(0)
            + self
                .pairwise_distance_table
                .get()
                .map(|table| table.deep_size_of_children(context))
                .unwrap_or(0)
    }
}

impl PartialEq for ProductQuantizationStorage {
    fn eq(&self, other: &Self) -> bool {
        self.distance_type == other.distance_type
            && self.metadata.eq(&other.metadata)
            && self.batch.columns().eq(other.batch.columns())
    }
}

impl ProductQuantizationStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        codebook: FixedSizeListArray,
        batch: RecordBatch,
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        distance_type: DistanceType,
        transposed: bool,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let frag_reuse_index = frag_reuse_index
            .map(|index| Arc::new(FragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        Self::new_with_remapper(
            codebook,
            batch,
            num_bits,
            num_sub_vectors,
            dimension,
            distance_type,
            transposed,
            frag_reuse_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_remapper(
        codebook: FixedSizeListArray,
        mut batch: RecordBatch,
        num_bits: u32,
        num_sub_vectors: usize,
        dimension: usize,
        distance_type: DistanceType,
        transposed: bool,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        if batch.num_columns() != 2 {
            log::warn!(
                "PQ storage should have 2 columns, but got {} columns: {}",
                batch.num_columns(),
                batch.schema(),
            );
            batch = batch.project(&[
                batch.schema().index_of(ROW_ID)?,
                batch.schema().index_of(PQ_CODE_COLUMN)?,
            ])?;
        }

        let Some(row_ids) = batch.column_by_name(ROW_ID) else {
            return Err(Error::index(
                "Row ID column not found from PQ storage".to_string(),
            ));
        };
        let mut row_ids: Arc<UInt64Array> = row_ids
            .as_primitive_opt::<UInt64Type>()
            .ok_or(Error::index(
                "Row ID column is not of type UInt64".to_string(),
            ))?
            .clone()
            .into();

        if !transposed {
            let num_sub_vectors_in_byte = if num_bits == 4 {
                num_sub_vectors / 2
            } else {
                num_sub_vectors
            };
            let pq_col = batch[PQ_CODE_COLUMN].as_fixed_size_list();
            let transposed_code = transpose(
                pq_col.values().as_primitive::<UInt8Type>(),
                row_ids.len(),
                num_sub_vectors_in_byte,
            );
            let pq_code_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                transposed_code,
                num_sub_vectors_in_byte as i32,
            )?);
            batch = batch.replace_column_by_name(PQ_CODE_COLUMN, pq_code_fsl)?;
        }

        let mut pq_code: Arc<UInt8Array> = batch[PQ_CODE_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive()
            .clone()
            .into();

        if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            let transposed_codes = pq_code.values();
            let mut new_row_ids = Vec::with_capacity(row_ids.len());
            let mut new_codes = Vec::with_capacity(row_ids.len() * num_sub_vectors);

            let row_ids_values = row_ids.values();
            for (i, row_id) in row_ids_values.iter().enumerate() {
                if let Some(mapped_value) = frag_reuse_index_ref.remap_row_id(*row_id) {
                    new_row_ids.push(mapped_value);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        num_bits,
                        num_sub_vectors,
                        i as u32,
                    ));
                }
            }

            let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
            let new_codes = UInt8Array::from(new_codes);
            batch = if new_row_ids.is_empty() {
                RecordBatch::new_empty(batch.schema())
            } else {
                let num_bytes_in_code = new_codes.len() / new_row_ids.len();
                let new_transposed_codes =
                    transpose(&new_codes, new_row_ids.len(), num_bytes_in_code);
                let codes_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                    new_transposed_codes,
                    num_bytes_in_code as i32,
                )?);
                RecordBatch::try_new(batch.schema(), vec![new_row_ids, codes_fsl])?
            };
            pq_code = batch[PQ_CODE_COLUMN]
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .clone()
                .into();
            // Refresh the stored row ids from the remapped batch. Without this
            // the storage keeps the pre-remap (compacted-away) addresses while
            // its codes are remapped, so search returns stale row ids and the
            // take fails with "fragment ... does not exist".
            row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone().into();
        }

        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };
        let metadata = ProductQuantizationMetadata {
            codebook_position: 0,
            nbits: num_bits,
            num_sub_vectors,
            dimension,
            codebook: Some(codebook),
            codebook_tensor: Vec::new(), // empty for v1 format
            transposed: true,
        };
        Ok(Self {
            metadata,
            distance_type,
            batch,
            pq_code,
            row_ids,
            pairwise_distance_table: Arc::new(OnceLock::new()),
        })
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Build a PQ storage from ProductQuantizer and a RecordBatch.
    ///
    /// Parameters
    /// ----------
    /// quantizer: ProductQuantizer
    ///    The quantizer used to transform the vectors.
    /// batch: RecordBatch
    ///   The batch of vectors to be transformed.
    /// vector_col: &str
    ///   The name of the column containing the vectors.
    pub async fn build(
        quantizer: ProductQuantizer,
        batch: &RecordBatch,
        vector_col: &str,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let codebook = quantizer.codebook.clone();
        let num_bits = quantizer.num_bits;
        let dimension = quantizer.dimension;
        let num_sub_vectors = quantizer.num_sub_vectors;
        let metric_type = quantizer.distance_type;
        let transform = PQTransformer::new(quantizer, vector_col, PQ_CODE_COLUMN);
        let batch = transform.transform(batch)?;
        Self::new(
            codebook,
            batch,
            num_bits,
            num_sub_vectors,
            dimension,
            metric_type,
            false,
            frag_reuse_index,
        )
    }

    pub fn codebook(&self) -> &FixedSizeListArray {
        self.metadata.codebook.as_ref().unwrap()
    }

    /// Load full PQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// object_store: &ObjectStore
    ///   The object store to load the storage from.
    /// path: &Path
    ///  The path to the storage.
    ///
    /// Returns
    /// --------
    /// Self
    ///
    /// Currently it loads everything in memory.
    /// TODO: support lazy loading later.
    pub async fn load(
        object_store: &ObjectStore,
        path: &Path,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let reader = V1FileReader::try_new_self_described(object_store, path, None).await?;
        let schema = reader.schema();

        let metadata_str = schema
            .metadata
            .get(INDEX_METADATA_SCHEMA_KEY)
            .ok_or(Error::index(format!(
                "Reading PQ storage: index key {} not found",
                INDEX_METADATA_SCHEMA_KEY
            )))?;
        let index_metadata: IndexMetadata = serde_json::from_str(metadata_str).map_err(|_| {
            Error::index(format!("Failed to parse index metadata: {}", metadata_str))
        })?;
        let distance_type: DistanceType =
            DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let metadata = ProductQuantizationMetadata::load(&reader).await?;
        Self::load_partition(
            &reader,
            0..reader.len(),
            distance_type,
            &metadata,
            frag_reuse_index,
        )
        .await
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    pub fn get_row_ids(&self, ids: &[u32]) -> Vec<u64> {
        ids.iter()
            .map(|&id| self.row_ids.value(id as usize))
            .collect()
    }

    fn pairwise_distance_table(&self) -> &[f32] {
        self.pairwise_distance_table
            .get_or_init(|| {
                let codebook = self.metadata.codebook.as_ref().unwrap();
                match codebook.value_type() {
                    DataType::Float16 => build_pairwise_distance_table(
                        codebook
                            .values()
                            .as_primitive::<datatypes::Float16Type>()
                            .values(),
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        self.metadata.dimension,
                        self.distance_type,
                    ),
                    DataType::Float32 => build_pairwise_distance_table(
                        codebook
                            .values()
                            .as_primitive::<datatypes::Float32Type>()
                            .values(),
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        self.metadata.dimension,
                        self.distance_type,
                    ),
                    DataType::Float64 => build_pairwise_distance_table(
                        codebook
                            .values()
                            .as_primitive::<datatypes::Float64Type>()
                            .values(),
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        self.metadata.dimension,
                        self.distance_type,
                    ),
                    _ => unimplemented!("Unsupported data type: {:?}", codebook.value_type()),
                }
            })
            .as_slice()
    }

    /// Write the PQ storage as a Lance partition to disk,
    /// and returns the number of rows written.
    ///
    pub async fn write_partition(
        &self,
        writer: &mut V1FileWriter<ManifestDescribing>,
    ) -> Result<usize> {
        let batch_size: usize = 10240; // TODO: make it configurable
        for offset in (0..self.batch.num_rows()).step_by(batch_size) {
            let length = min(batch_size, self.batch.num_rows() - offset);
            let slice = self.batch.slice(offset, length);
            writer.write(&[slice]).await?;
        }
        Ok(self.batch.num_rows())
    }
}

pub fn transpose<T: ArrowPrimitiveType>(
    original: &PrimitiveArray<T>,
    num_rows: usize,
    num_columns: usize,
) -> PrimitiveArray<T>
where
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    if original.is_empty() {
        return original.clone();
    }

    let mut transposed_codes = vec![T::default_value(); original.len()];
    for (vec_idx, codes) in original.values().chunks_exact(num_columns).enumerate() {
        for (sub_vec_idx, code) in codes.iter().enumerate() {
            transposed_codes[sub_vec_idx * num_rows + vec_idx] = *code;
        }
    }

    transposed_codes.into()
}

#[async_trait]
impl QuantizerStorage for ProductQuantizationStorage {
    type Metadata = ProductQuantizationMetadata;

    fn try_from_batch(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };

        // now it supports only Float32Type
        let codebook = match &metadata.codebook {
            Some(codebook) => codebook.clone(),
            None => {
                // legacy format would contains codebook tensor but not codebook
                debug_assert!(!metadata.codebook_tensor.is_empty());
                let codebook_tensor = pb::Tensor::decode(metadata.codebook_tensor.as_slice())?;
                FixedSizeListArray::try_from(&codebook_tensor)?
            }
        };

        Self::new(
            codebook,
            batch,
            metadata.nbits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
            frag_reuse_index,
        )
    }

    fn try_from_batch_with_remapper(
        batch: RecordBatch,
        metadata: &Self::Metadata,
        distance_type: DistanceType,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let distance_type = match distance_type {
            DistanceType::Cosine => DistanceType::L2,
            _ => distance_type,
        };
        let codebook = match &metadata.codebook {
            Some(codebook) => codebook.clone(),
            None => {
                debug_assert!(!metadata.codebook_tensor.is_empty());
                let codebook_tensor = pb::Tensor::decode(metadata.codebook_tensor.as_slice())?;
                FixedSizeListArray::try_from(&codebook_tensor)?
            }
        };
        Self::new_with_remapper(
            codebook,
            batch,
            metadata.nbits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
            frag_reuse_index,
        )
    }

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    // we can't use the default implementation of remap,
    // because PQ Storage transposed the PQ codes
    fn remap(&self, mapping: &RowAddrRemap) -> Result<Self> {
        let transposed_codes = self.pq_code.values();
        let mut new_row_ids = Vec::with_capacity(self.len());
        let mut new_codes = Vec::with_capacity(self.len() * self.metadata.num_sub_vectors);

        let row_ids = self.row_ids.values();
        for (i, row_id) in row_ids.iter().enumerate() {
            match mapping.get(*row_id) {
                Some(Some(new_id)) => {
                    new_row_ids.push(new_id);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        i as u32,
                    ));
                }
                Some(None) => {}
                None => {
                    new_row_ids.push(*row_id);
                    new_codes.extend(get_pq_code(
                        transposed_codes,
                        self.metadata.nbits,
                        self.metadata.num_sub_vectors,
                        i as u32,
                    ));
                }
            }
        }

        let new_row_ids = Arc::new(UInt64Array::from(new_row_ids));
        let new_codes = UInt8Array::from(new_codes);
        let batch = if new_row_ids.is_empty() {
            RecordBatch::new_empty(self.schema())
        } else {
            let num_bytes_in_code = new_codes.len() / new_row_ids.len();
            let new_transposed_codes = transpose(&new_codes, new_row_ids.len(), num_bytes_in_code);
            let codes_fsl = Arc::new(FixedSizeListArray::try_new_from_values(
                new_transposed_codes,
                num_bytes_in_code as i32,
            )?);
            RecordBatch::try_new(self.schema(), vec![new_row_ids.clone(), codes_fsl])?
        };
        let transposed_codes = batch[PQ_CODE_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive::<UInt8Type>()
            .clone();

        Ok(Self {
            metadata: self.metadata.clone(),
            distance_type: self.distance_type,
            batch,
            pq_code: Arc::new(transposed_codes),
            row_ids: new_row_ids,
            pairwise_distance_table: self.pairwise_distance_table.clone(),
        })
    }

    /// Load a partition of PQ storage from disk.
    ///
    /// Parameters
    /// ----------
    /// - *reader: &V1FileReader
    async fn load_partition(
        reader: &V1FileReader,
        range: std::ops::Range<usize>,
        distance_type: DistanceType,
        metadata: &Self::Metadata,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        // Hard coded to float32 for now
        let codebook = metadata
            .codebook
            .as_ref()
            .ok_or(Error::index(
                "Codebook not found in PQ metadata".to_string(),
            ))?
            .values()
            .as_primitive::<Float32Type>()
            .clone();

        let codebook =
            FixedSizeListArray::try_new_from_values(codebook, metadata.dimension as i32)?;

        let schema = reader.schema();
        let batch = reader.read_range(range, schema).await?;

        Self::new(
            codebook,
            batch,
            metadata.nbits,
            metadata.num_sub_vectors,
            metadata.dimension,
            distance_type,
            metadata.transposed,
            frag_reuse_index,
        )
    }
}

impl VectorStore for ProductQuantizationStorage {
    type DistanceCalculator<'a> = PQDistCalculator;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch>> {
        Ok(std::iter::once(self.batch.clone()))
    }

    fn append_batch(&self, _batch: RecordBatch, _vector_column: &str) -> Result<Self> {
        unimplemented!()
    }

    fn schema(&self) -> &SchemaRef {
        self.batch.schema_ref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
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
        let codebook = self.metadata.codebook.as_ref().unwrap();
        match codebook.value_type() {
            DataType::Float16 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float16Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float16Type>().values(),
                self.distance_type,
            ),
            DataType::Float32 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float32Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float32Type>().values(),
                self.distance_type,
            ),
            DataType::Float64 => PQDistCalculator::new(
                codebook
                    .values()
                    .as_primitive::<datatypes::Float64Type>()
                    .values(),
                self.metadata.nbits,
                self.metadata.num_sub_vectors,
                self.pq_code.clone(),
                query.as_primitive::<datatypes::Float64Type>().values(),
                self.distance_type,
            ),
            _ => unimplemented!("Unsupported data type: {:?}", codebook.value_type()),
        }
    }

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_> {
        let codes = get_pq_code(
            self.pq_code.values(),
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            id,
        );
        PQDistCalculator::new_from_codes(
            self.pairwise_distance_table(),
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            self.pq_code.clone(),
            codes,
            self.distance_type,
        )
    }

    fn dist_between(&self, u: u32, v: u32) -> f32 {
        // this is a fast way to compute distance between two vectors in the same storage.
        // it doesn't construct the distance table.
        let pq_codes = self.pq_code.values();
        let u_codes = get_pq_code(
            pq_codes,
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            u,
        );
        let v_codes = get_pq_code(
            pq_codes,
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            v,
        );
        pq_code_distance(
            self.pairwise_distance_table(),
            self.metadata.nbits,
            self.metadata.num_sub_vectors,
            u_codes,
            v_codes,
            self.distance_type,
        )
    }

    fn prefers_candidate(&self, candidate: &OrderedNode, selected: &[OrderedNode]) -> bool {
        selected
            .iter()
            .all(|other| candidate.dist < OrderedFloat(self.dist_between(candidate.id, other.id)))
    }
}

/// Distance calculator backed by PQ code.
pub struct PQDistCalculator {
    distance_table: Vec<f32>,
    pq_code: Arc<UInt8Array>,
    num_sub_vectors: usize,
    num_bits: u32,
    distance_type: DistanceType,
}

impl PQDistCalculator {
    fn new<T: L2 + Dot>(
        codebook: &[T],
        num_bits: u32,
        num_sub_vectors: usize,
        pq_code: Arc<UInt8Array>,
        query: &[T],
        distance_type: DistanceType,
    ) -> Self {
        let distance_table = match distance_type {
            DistanceType::L2 | DistanceType::Cosine => {
                build_distance_table_l2(codebook, num_bits, num_sub_vectors, query)
            }
            DistanceType::Dot => {
                build_distance_table_dot(codebook, num_bits, num_sub_vectors, query)
            }
            _ => unimplemented!("DistanceType is not supported: {:?}", distance_type),
        };
        Self {
            distance_table,
            num_sub_vectors,
            pq_code,
            num_bits,
            distance_type,
        }
    }

    fn new_from_codes(
        pairwise_distance_table: &[f32],
        num_bits: u32,
        num_sub_vectors: usize,
        pq_code: Arc<UInt8Array>,
        query_codes: impl Iterator<Item = u8>,
        distance_type: DistanceType,
    ) -> Self {
        let distance_table = distance_table_from_codes(
            pairwise_distance_table,
            num_bits,
            num_sub_vectors,
            query_codes,
        );
        Self {
            distance_table,
            num_sub_vectors,
            pq_code,
            num_bits,
            distance_type,
        }
    }

    fn get_pq_code(&self, id: u32) -> impl Iterator<Item = usize> + '_ {
        get_pq_code(
            self.pq_code.values(),
            self.num_bits,
            self.num_sub_vectors,
            id,
        )
        .map(|v| v as usize)
    }
}

impl DistCalculator for PQDistCalculator {
    fn distance(&self, id: u32) -> f32 {
        let dist = if self.num_bits == 4 {
            compute_pq_distance_4bit_row(&self.distance_table, self.pq_code.values(), id as usize)
        } else {
            let num_centroids = 2_usize.pow(self.num_bits);
            self.get_pq_code(id)
                .enumerate()
                .map(|(i, c)| self.distance_table[i * num_centroids + c])
                .sum()
        };

        if self.distance_type == DistanceType::Dot {
            dist - (self.num_sub_vectors as f32 - 1.0)
        } else {
            dist
        }
    }

    fn has_exact_topk_scan(&self) -> bool {
        self.num_bits == 4
            && matches!(self.distance_type, DistanceType::L2 | DistanceType::Dot)
            && bounded_4bit_scores(&self.distance_table)
    }

    fn distance_all(&self, _k_hint: usize) -> Vec<f32> {
        match self.distance_type {
            DistanceType::L2 => compute_pq_distance(
                &self.distance_table,
                self.num_bits,
                self.num_sub_vectors,
                self.pq_code.values(),
            ),
            DistanceType::Cosine => {
                // it seems we implemented cosine distance at some version,
                // but from now on, we should use normalized L2 distance.
                debug_assert!(
                    false,
                    "cosine distance should be converted to normalized L2 distance"
                );
                // L2 over normalized vectors:  ||x - y|| = x^2 + y^2 - 2 * xy = 1 + 1 - 2 * xy = 2 * (1 - xy)
                // Cosine distance: 1 - |xy| / (||x|| * ||y||) = 1 - xy / (x^2 * y^2) = 1 - xy / (1 * 1) = 1 - xy
                // Therefore, Cosine = L2 / 2
                let l2_dists = compute_pq_distance(
                    &self.distance_table,
                    self.num_bits,
                    self.num_sub_vectors,
                    self.pq_code.values(),
                );
                l2_dists.into_iter().map(|v| v / 2.0).collect()
            }
            DistanceType::Dot => {
                let dot_dists = compute_pq_distance(
                    &self.distance_table,
                    self.num_bits,
                    self.num_sub_vectors,
                    self.pq_code.values(),
                );
                let diff = self.num_sub_vectors as f32 - 1.0;
                dot_dists.into_iter().map(|v| v - diff).collect()
            }
            _ => unimplemented!("distance type is not supported: {:?}", self.distance_type),
        }
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
        quantized_dists: &mut Vec<u16>,
        quantized_table: &mut Vec<u8>,
        candidates: &mut Vec<u32>,
    ) {
        // `OrderedFloat` ranks a positive NaN above every score, so such a
        // lower bound admits no row.
        if k == 0 || lower_bound.is_some_and(|bound| bound.is_nan() && bound.is_sign_positive()) {
            return;
        }
        let code_len = self.num_sub_vectors / 2;
        let table = (self.num_bits == 4
            && matches!(self.distance_type, DistanceType::L2 | DistanceType::Dot)
            // When every row fits in the heap, every row is scored anyway.
            && self.pq_code.len() / code_len > k.saturating_sub(res.len()))
        .then(|| quantize_4bit_distance_table(&self.distance_table, quantized_table))
        .flatten();
        let Some(table) = table else {
            *dists = self.distance_all(k);
            accumulate_distances_into_heap(k, lower_bound, upper_bound, row_id, res, dists);
            return;
        };

        // A row whose quantized sum is `q` scores within a bounded error of `q`
        // steps above an offset. Rows whose sums exceed `cutoff` cannot score
        // below the current k-th best, so only the others are scored exactly;
        // the heap sees the same distances as a per-row scan.
        let score_diff = if self.distance_type == DistanceType::Dot {
            self.num_sub_vectors as f32 - 1.0
        } else {
            0.0
        };
        let lower_bound = OrderedFloat(lower_bound.unwrap_or(f32::MIN));
        let upper_bound = OrderedFloat(upper_bound.unwrap_or(f32::MAX));
        let mut threshold = upper_bound;
        if res.len() >= k
            && let Some(node) = res.peek()
        {
            threshold = threshold.min(node.dist);
        }
        let Some(mut cutoff) = table.sum_cutoff(threshold.0, score_diff) else {
            return;
        };

        let num_rows = self.pq_code.len() / code_len;
        let stored_all_sums = if res.len() >= k && lower_bound <= OrderedFloat(f32::MIN) {
            // A full heap already bounds the k-th best distance, so the kernel
            // reports only the rows within `cutoff`, and only their sums are
            // stored, at their rows, for the sort and the rescoring below; the
            // other rows' entries are never read. The cutoff stays fixed during
            // the scan and only the rescoring lowers it.
            //
            // A heap inherited from a farther partition can be loose enough to
            // admit most rows, which costs more to push and sort than storing
            // every sum and narrowing them by the k-th sum bound. Once more
            // rows than the budget are admitted, the scan switches to that:
            // the rows already passed over are marked above any cutoff and
            // every later row's sum is stored.
            let budget = fused_candidate_budget(num_rows, k);
            if quantized_dists.len() < num_rows {
                quantized_dists.resize(num_rows, 0);
            }
            candidates.clear();
            let mut store_all = false;
            filter_4bit_dist_table_transposed(
                num_rows,
                code_len,
                self.pq_code.values(),
                quantized_table,
                cutoff,
                |start, mut mask, sums| {
                    if store_all {
                        if mask == u64::MAX {
                            quantized_dists[start..start + sums.len()].copy_from_slice(sums);
                        } else {
                            while mask != 0 {
                                let i = mask.trailing_zeros() as usize;
                                quantized_dists[start + i] = sums[i];
                                mask &= mask - 1;
                            }
                        }
                        return u16::MAX;
                    }
                    while mask != 0 {
                        let i = mask.trailing_zeros() as usize;
                        quantized_dists[start + i] = sums[i];
                        candidates.push((start + i) as u32);
                        mask &= mask - 1;
                    }
                    if candidates.len() <= budget {
                        return cutoff;
                    }
                    store_all = true;
                    // Candidates hold every admitted row so far, ascending, so
                    // the rows between them are the ones passed over.
                    let mut next_row = 0;
                    for &row in candidates.iter() {
                        quantized_dists[next_row..row as usize].fill(u16::MAX);
                        next_row = row as usize + 1;
                    }
                    let passed = (start + sums.len()).min(num_rows);
                    quantized_dists[next_row..passed].fill(u16::MAX);
                    u16::MAX
                },
            );
            store_all
        } else {
            // Before the heap is full only the upper bound limits the cutoff,
            // so a filter would report most rows. Storing every sum instead
            // lets the k-th sum bound below narrow the candidates first. A
            // search with a lower bound takes this path as well.
            quantized_dists.resize(num_rows, 0);
            sum_4bit_dist_table_transposed(
                num_rows,
                code_len,
                self.pq_code.values(),
                quantized_table,
                quantized_dists,
            );
            true
        };
        if stored_all_sums {
            // Without a lower bound, k rows with small sums bound the k-th best
            // distance of this partition before any row is scored. A row marked
            // `u16::MAX` only raises its block's minimum, so the bound holds.
            if lower_bound <= OrderedFloat(f32::MIN)
                && let Some(sum) =
                    kth_smallest_sum_bound(&quantized_dists[..num_rows], k, candidates)
            {
                threshold = threshold.min(OrderedFloat(table.kth_bound_distance(sum, score_diff)));
                let Some(sum) = table.sum_cutoff(threshold.0, score_diff) else {
                    return;
                };
                cutoff = cutoff.min(sum);
            }
            candidates.clear();
            collect_sums_at_most(&quantized_dists[..num_rows], cutoff, candidates);
        }

        // Score candidates from the smallest sum up, so the heap settles on its
        // final rows early and the remaining candidates can be cut off at once.
        let candidates = sort_ids_by_sum(candidates, quantized_dists);
        // Candidates are scored a chunk at a time so their add chains overlap,
        // then fed to the heap one by one in the same order as before. A chunk
        // may score up to `RESCORE_CHUNK - 1` rows past the cut-off; those
        // distances are dropped unused.
        const RESCORE_CHUNK: usize = 16;
        let dot_offset =
            (self.distance_type == DistanceType::Dot).then_some(self.num_sub_vectors as f32 - 1.0);
        'rescore: for chunk in candidates.chunks(RESCORE_CHUNK) {
            if quantized_dists[chunk[0] as usize] > cutoff {
                break;
            }
            dists.resize(chunk.len(), 0.0);
            compute_pq_distance_4bit_rows(
                &self.distance_table,
                self.pq_code.values(),
                chunk,
                dists,
            );
            if let Some(dot_offset) = dot_offset {
                // The same post-op as `distance()`.
                dists.iter_mut().for_each(|dist| *dist -= dot_offset);
            }
            for (&id, &dist) in chunk.iter().zip(dists.iter()) {
                if quantized_dists[id as usize] > cutoff {
                    break 'rescore;
                }
                let dist = OrderedFloat(dist);
                if dist < lower_bound || dist >= upper_bound {
                    continue;
                }
                if res.len() < k {
                    res.push(OrderedNode::new(row_id(id), dist));
                } else if let Some(mut top) = res.peek_mut()
                    && top.dist > dist
                {
                    *top = OrderedNode::new(row_id(id), dist);
                } else {
                    continue;
                }
                if res.len() >= k
                    && let Some(node) = res.peek()
                {
                    match table.sum_cutoff(node.dist.0, score_diff) {
                        Some(sum) => cutoff = cutoff.min(sum),
                        None => break 'rescore,
                    }
                }
            }
        }
    }
}

/// Append the ids of the rows whose sum is at most `cutoff`, in ascending
/// order.
///
/// Comparing a fixed-size block of sums into a bitmask vectorizes, where a
/// per-row branch would mispredict on a sparse selection.
fn collect_sums_at_most(sums: &[u16], cutoff: u16, ids: &mut Vec<u32>) {
    const BLOCK: usize = 64;
    let mask = |block: &[u16]| {
        block.iter().enumerate().fold(0u64, |mask, (i, &sum)| {
            mask | (u64::from(sum <= cutoff) << i)
        })
    };
    let (blocks, tail) = sums.as_chunks::<BLOCK>();
    let masks = blocks
        .iter()
        .map(|block| mask(block))
        .chain(std::iter::once(mask(tail)));
    for (block_idx, mut mask) in masks.enumerate() {
        let start = (block_idx * BLOCK) as u32;
        while mask != 0 {
            ids.push(start + mask.trailing_zeros());
            mask &= mask - 1;
        }
    }
}

/// Sort `ids` by their sums and return the sorted ids, which may be stored
/// past the input in `ids`.
///
/// Candidate sums usually span a narrow window, where a counting sort replaces
/// `n log n` indirect comparisons with a few linear passes.
fn sort_ids_by_sum<'a>(ids: &'a mut Vec<u32>, sums: &[u16]) -> &'a [u32] {
    /// Sum range always sorted by counting. Its 16 KiB of u32 buckets stay in
    /// L1 and cost about as much as a small comparison sort, and it covers the
    /// whole sum span of 16 sub-vectors (`16 * 255` units). Wider ranges use
    /// counting only up to twice the candidate count, which keeps the sort
    /// linear in the candidates.
    const MIN_COUNTING_RANGE: usize = 4096;
    let num_ids = ids.len();
    let Some((min_sum, max_sum)) =
        ids.iter()
            .map(|&id| sums[id as usize])
            .fold(None, |range, sum| {
                let (min, max) = range.unwrap_or((sum, sum));
                Some((min.min(sum), max.max(sum)))
            })
    else {
        return ids;
    };
    let range = usize::from(max_sum - min_sum) + 1;
    if range > MIN_COUNTING_RANGE.max(2 * num_ids) {
        ids.sort_unstable_by_key(|&id| sums[id as usize]);
        return ids;
    }

    ids.resize(2 * num_ids + range, 0);
    let (unsorted, rest) = ids.split_at_mut(num_ids);
    let (starts, sorted) = rest.split_at_mut(range);
    for &id in unsorted.iter() {
        starts[usize::from(sums[id as usize] - min_sum)] += 1;
    }
    let mut start = 0;
    for count in starts.iter_mut() {
        (*count, start) = (start, start + *count);
    }
    for &id in unsorted.iter() {
        let start = &mut starts[usize::from(sums[id as usize] - min_sum)];
        sorted[*start as usize] = id;
        *start += 1;
    }
    sorted
}

/// How many rows the fused filter of a full-heap scan may admit before it
/// switches to storing every sum.
///
/// The store path's extra passes cost well under a nanosecond per row, while
/// each admitted row costs a few to push and sort, more when a wide sum range
/// sends the sort to comparisons. Switching is not free either: the rows after
/// the switch are stored through the filter's callback, which is slower than
/// the plain store kernel. On the x86 `pq4_topk_prefilled` bench, a heap
/// admitting all 1024 rows (m=96, k=100) ran the fused path at 0.55x the store
/// path and 0.92x with this budget, while budgets of `num_rows / 32` and
/// `num_rows / 64` switched more often and lowered the bench's geomean against
/// the store path from 1.03x to 1.01x and 1.00x.
///
/// A heap that admits a few percent of a large partition stays within the
/// budget yet can cost more than the store path, whose k-th sum bound comes
/// from the partition's own rows: with m=64, rows=16384, k=10 and a heap at
/// twice the k-th distance, the filter admits about 1000 rows where that bound
/// keeps about 90, and the scan runs at 0.87x the store path. No budget fixes
/// this case alone. Checking the budget against the rows passed so far
/// (`k + passed_rows / 8` after each batch) never switched here and matched
/// this budget within noise, `passed_rows / 16` still did not switch, and
/// `passed_rows / 32` raised the case to 0.93x but slowed k=100 heaps at 1.2
/// and 2.0 times the k-th distance to 0.85x-0.88x and the geomean to 1.03x.
fn fused_candidate_budget(num_rows: usize, k: usize) -> usize {
    k + num_rows / 8
}

/// An upper bound of the k-th smallest value in `sums`, or `None` if there are
/// fewer than `k` sums. Each of about `4 * k` contiguous blocks contributes its
/// minimum, a distinct row, so the k-th smallest block minimum is reached by at
/// least `k` rows while costing only a vectorizable pass and a small selection.
fn kth_smallest_sum_bound(sums: &[u16], k: usize, block_mins: &mut Vec<u32>) -> Option<u16> {
    if k == 0 || sums.len() < k {
        return None;
    }
    let block_len = (sums.len() / (4 * k)).max(1);
    block_mins.clear();
    block_mins.extend(
        sums.chunks(block_len)
            .map(|block| block.iter().copied().min().unwrap_or(u16::MAX) as u32),
    );
    debug_assert!(
        block_mins.len() >= k,
        "{} blocks for k={k}",
        block_mins.len()
    );
    let (_, kth, _) = block_mins.select_nth_unstable(k - 1);
    Some(*kth as u16)
}

pub(crate) fn build_pairwise_distance_table<T: L2 + Cosine + Dot>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    distance_type: DistanceType,
) -> Vec<f32> {
    let num_centroids = 2_usize.pow(num_bits);
    let sub_vector_width = dimension / num_sub_vectors;
    let mut result = Vec::with_capacity(num_sub_vectors * num_centroids * num_centroids);
    let distance_fn = distance_type.func();
    for sub_vector_idx in 0..num_sub_vectors {
        let sub_vector_offset = sub_vector_idx * num_centroids * sub_vector_width;
        let centroids =
            &codebook[sub_vector_offset..sub_vector_offset + num_centroids * sub_vector_width];
        for query_centroid_idx in 0..num_centroids {
            let query_offset = query_centroid_idx * sub_vector_width;
            let query = &centroids[query_offset..query_offset + sub_vector_width];
            for centroid_idx in 0..num_centroids {
                let centroid_offset = centroid_idx * sub_vector_width;
                let centroid = &centroids[centroid_offset..centroid_offset + sub_vector_width];
                result.push(distance_fn(query, centroid));
            }
        }
    }
    result
}

fn distance_table_from_codes(
    pairwise_distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    query_codes: impl Iterator<Item = u8>,
) -> Vec<f32> {
    let num_centroids = 2_usize.pow(num_bits);
    let mut distance_table = Vec::with_capacity(num_sub_vectors * num_centroids);
    if num_bits == 4 {
        for (byte_idx, query_code) in query_codes.enumerate() {
            let current_idx = (query_code & 0x0F) as usize;
            let current_sub_vector_idx = 2 * byte_idx;
            extend_pairwise_distance_row(
                &mut distance_table,
                pairwise_distance_table,
                num_centroids,
                current_sub_vector_idx,
                current_idx,
            );

            let next_idx = (query_code >> 4) as usize;
            let next_sub_vector_idx = current_sub_vector_idx + 1;
            extend_pairwise_distance_row(
                &mut distance_table,
                pairwise_distance_table,
                num_centroids,
                next_sub_vector_idx,
                next_idx,
            );
        }
    } else {
        for (sub_vector_idx, query_code) in query_codes.enumerate() {
            extend_pairwise_distance_row(
                &mut distance_table,
                pairwise_distance_table,
                num_centroids,
                sub_vector_idx,
                query_code as usize,
            );
        }
    }
    distance_table
}

fn extend_pairwise_distance_row(
    distance_table: &mut Vec<f32>,
    pairwise_distance_table: &[f32],
    num_centroids: usize,
    sub_vector_idx: usize,
    query_centroid_idx: usize,
) {
    let start = (sub_vector_idx * num_centroids + query_centroid_idx) * num_centroids;
    distance_table.extend_from_slice(&pairwise_distance_table[start..start + num_centroids]);
}

fn pq_code_distance(
    pairwise_distance_table: &[f32],
    num_bits: u32,
    num_sub_vectors: usize,
    lhs_codes: impl Iterator<Item = u8>,
    rhs_codes: impl Iterator<Item = u8>,
    distance_type: DistanceType,
) -> f32 {
    let num_centroids = 2_usize.pow(num_bits);
    let dist = if num_bits == 4 {
        lhs_codes
            .zip(rhs_codes)
            .enumerate()
            .map(|(byte_idx, (lhs, rhs))| {
                let current_sub_vector_idx = 2 * byte_idx;
                pairwise_distance(
                    pairwise_distance_table,
                    num_centroids,
                    current_sub_vector_idx,
                    (lhs & 0x0F) as usize,
                    (rhs & 0x0F) as usize,
                ) + pairwise_distance(
                    pairwise_distance_table,
                    num_centroids,
                    current_sub_vector_idx + 1,
                    (lhs >> 4) as usize,
                    (rhs >> 4) as usize,
                )
            })
            .sum()
    } else {
        lhs_codes
            .zip(rhs_codes)
            .enumerate()
            .map(|(sub_vector_idx, (lhs, rhs))| {
                pairwise_distance(
                    pairwise_distance_table,
                    num_centroids,
                    sub_vector_idx,
                    lhs as usize,
                    rhs as usize,
                )
            })
            .sum()
    };

    if distance_type == DistanceType::Dot {
        dist - (num_sub_vectors as f32 - 1.0)
    } else {
        dist
    }
}

fn pairwise_distance(
    pairwise_distance_table: &[f32],
    num_centroids: usize,
    sub_vector_idx: usize,
    query_centroid_idx: usize,
    centroid_idx: usize,
) -> f32 {
    pairwise_distance_table
        [(sub_vector_idx * num_centroids + query_centroid_idx) * num_centroids + centroid_idx]
}

fn get_pq_code(
    pq_code: &[u8],
    num_bits: u32,
    num_sub_vectors: usize,
    id: u32,
) -> impl Iterator<Item = u8> + '_ {
    let num_bytes = if num_bits == 4 {
        num_sub_vectors / 2
    } else {
        num_sub_vectors
    };

    let num_vectors = pq_code.len() / num_bytes;
    pq_code
        .iter()
        .skip(id as usize)
        .step_by(num_vectors)
        .copied()
        .exact_size(num_bytes)
}

#[cfg(test)]
fn get_centroids<T: Clone>(
    codebook: &[T],
    num_bits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    codes: impl Iterator<Item = u8>,
) -> Vec<T> {
    // codebook[i][j] is the j-th centroid of the i-th sub-vector.
    // the codebook is stored as a flat array, codebook[i * num_centroids + j] = codebook[i][j]

    if num_bits == 4 {
        return get_centroids_4bit(codebook, num_sub_vectors, dimension, codes);
    }

    let num_centroids: usize = 2_usize.pow(8);
    let sub_vector_width = dimension / num_sub_vectors;
    let mut centroids = Vec::with_capacity(dimension);
    for (sub_vec_idx, centroid_idx) in codes.enumerate() {
        let centroid_idx = centroid_idx as usize;
        let centroid = &codebook[sub_vec_idx * num_centroids * sub_vector_width
            + centroid_idx * sub_vector_width
            ..sub_vec_idx * num_centroids * sub_vector_width
                + (centroid_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(centroid);
    }
    centroids
}

#[cfg(test)]
fn get_centroids_4bit<T: Clone>(
    codebook: &[T],
    num_sub_vectors: usize,
    dimension: usize,
    codes: impl Iterator<Item = u8>,
) -> Vec<T> {
    let num_centroids: usize = 16;
    let sub_vector_width = dimension / num_sub_vectors;
    let mut centroids = Vec::with_capacity(dimension);
    for (sub_vec_idx, centroid_idx) in codes.into_iter().enumerate() {
        let current_idx = (centroid_idx & 0x0F) as usize;
        let offset = 2 * sub_vec_idx * num_centroids * sub_vector_width;
        let current_centroid = &codebook[offset + current_idx * sub_vector_width
            ..offset + (current_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(current_centroid);

        let next_idx = (centroid_idx >> 4) as usize;
        let offset = (2 * sub_vec_idx + 1) * num_centroids * sub_vector_width;
        let next_centroid = &codebook
            [offset + next_idx * sub_vector_width..offset + (next_idx + 1) * sub_vector_width];
        centroids.extend_from_slice(next_centroid);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use crate::vector::storage::StorageBuilder;

    use super::*;

    use arrow_array::{Float32Array, UInt32Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID_FIELD;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    const DIM: usize = 32;
    const TOTAL: usize = 512;
    const NUM_SUB_VECTORS: usize = 16;

    async fn create_pq_storage() -> ProductQuantizationStorage {
        let codebook = Float32Array::from_iter_values((0..256 * DIM).map(|_| rand::random()));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, DistanceType::Dot);

        let schema = ArrowSchema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);
        let vectors = Float32Array::from_iter_values((0..TOTAL * DIM).map(|_| rand::random()));
        let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(fsl), Arc::new(row_ids)]).unwrap();

        StorageBuilder::new("vec".to_owned(), pq.distance_type, pq, None)
            .unwrap()
            .build(vec![batch])
            .unwrap()
    }

    async fn create_pq_storage_with_extra_column() -> ProductQuantizationStorage {
        let codebook = Float32Array::from_iter_values((0..256 * DIM).map(|_| rand::random()));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, DistanceType::Dot);

        let schema = ArrowSchema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
            Field::new("extra", DataType::UInt32, true),
        ]);
        let vectors = Float32Array::from_iter_values((0..TOTAL * DIM).map(|_| rand::random()));
        let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
        let extra_column = UInt32Array::from_iter_values((0..TOTAL).map(|v| v as u32));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(
            schema.into(),
            vec![Arc::new(fsl), Arc::new(row_ids), Arc::new(extra_column)],
        )
        .unwrap();

        StorageBuilder::new("vec".to_owned(), pq.distance_type, pq, None)
            .unwrap()
            .build(vec![batch])
            .unwrap()
    }

    #[tokio::test]
    async fn test_build_pq_storage() {
        let storage = create_pq_storage().await;
        assert_eq!(storage.len(), TOTAL);
        assert_eq!(storage.metadata.num_sub_vectors, NUM_SUB_VECTORS);
        assert_eq!(
            storage.metadata.codebook.as_ref().unwrap().values().len(),
            256 * DIM
        );
        assert_eq!(storage.pq_code.len(), TOTAL * NUM_SUB_VECTORS);
        assert_eq!(storage.row_ids.len(), TOTAL);
    }

    fn create_4bit_pq_storage(
        rows: usize,
        distance_type: DistanceType,
    ) -> ProductQuantizationStorage {
        let mut rng = StdRng::seed_from_u64(rows as u64);
        let codebook = Float32Array::from_iter_values((0..16 * DIM).map(|_| rng.random()));
        let codebook = FixedSizeListArray::try_new_from_values(codebook, DIM as i32).unwrap();
        let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 4, DIM, codebook, distance_type);
        let schema = ArrowSchema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Field::new_list_field(DataType::Float32, true).into(),
                    DIM as i32,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);
        let vectors = Float32Array::from_iter_values((0..rows * DIM).map(|_| rng.random()));
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let row_ids = UInt64Array::from_iter_values(0..rows as u64);
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(fsl), Arc::new(row_ids)]).unwrap();
        StorageBuilder::new("vec".to_owned(), distance_type, pq, None)
            .unwrap()
            .build(vec![batch])
            .unwrap()
    }

    /// The 4-bit bulk scan must select the same top-k, with the same
    /// distances, as scoring every row with `distance()`, from an empty heap or
    /// one already holding another partition's rows, with or without bounds.
    #[rstest]
    fn test_4bit_accumulate_topk_matches_per_row(
        #[values(DistanceType::L2, DistanceType::Dot)] distance_type: DistanceType,
        #[values((5, 10), (1000, 10), (5000, 100))] rows_and_k: (usize, usize),
        #[values(false, true)] prefilled: bool,
        #[values(false, true)] bounded: bool,
    ) {
        let (rows, k) = rows_and_k;
        let storage = create_4bit_pq_storage(rows, distance_type);
        let mut rng = StdRng::seed_from_u64(7);
        let query: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..DIM).map(|_| rng.random::<f32>()),
        ));
        let calc = storage.dist_calculator(query, 0.0);
        let exact = (0..rows as u32)
            .map(|id| calc.distance(id))
            .collect::<Vec<_>>();
        let mut sorted = exact.clone();
        sorted.sort_by(f32::total_cmp);
        let (lower_bound, upper_bound) = if bounded {
            (Some(sorted[rows / 5]), Some(sorted[rows * 4 / 5]))
        } else {
            (None, None)
        };
        let mut heap = BinaryHeap::new();
        if prefilled {
            for (i, dist) in sorted.iter().step_by(3).take(k).enumerate() {
                heap.push(OrderedNode::new(u64::MAX - i as u64, OrderedFloat(*dist)));
            }
        }
        let row_id = |id: u32| id as u64 + 1_000_000;
        let mut expected = heap.clone();
        accumulate_distances_into_heap(k, lower_bound, upper_bound, row_id, &mut expected, &exact);

        let mut actual = heap;
        calc.accumulate_topk_with_scratch(
            k,
            lower_bound,
            upper_bound,
            row_id,
            &mut actual,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        // Rows tied at the k-th distance may be kept in either order, so
        // compare the selected distances and check each row's own distance.
        let dists = |heap: &BinaryHeap<OrderedNode<u64>>| {
            let mut dists = heap
                .iter()
                .map(|node| node.dist.0.to_bits())
                .collect::<Vec<_>>();
            dists.sort_unstable();
            dists
        };
        assert_eq!(dists(&actual), dists(&expected));
        for node in &actual {
            // Prefilled nodes carry ids far above every row's.
            if let Some(id) = node
                .id
                .checked_sub(1_000_000)
                .filter(|&id| id < rows as u64)
            {
                assert_eq!(node.dist.0.to_bits(), exact[id as usize].to_bits());
            }
        }
    }

    #[tokio::test]
    async fn test_distance_all() {
        let storage = create_pq_storage().await;
        let query = Arc::new(Float32Array::from_iter_values((0..DIM).map(|v| v as f32)));
        let dist_calc = storage.dist_calculator(query, 0.0);
        let expected = (0..storage.len())
            .map(|id| dist_calc.distance(id as u32))
            .collect::<Vec<_>>();
        let distances = dist_calc.distance_all(100);
        assert_eq!(distances, expected);
    }

    #[tokio::test]
    async fn test_dist_between() {
        let mut rng = rand::rng();
        let storage = create_pq_storage().await;
        let u = rng.random_range(0..storage.len() as u32);
        let v = rng.random_range(0..storage.len() as u32);
        let dist1 = storage.dist_between(u, v);
        let dist2 = storage.dist_between(v, u);
        assert_eq!(dist1, dist2);
    }

    #[tokio::test]
    async fn test_dist_calculator_from_id_matches_reconstructed_distance() {
        let mut rng = rand::rng();
        let storage = create_pq_storage().await;
        let u = rng.random_range(0..storage.len() as u32);
        let v = rng.random_range(0..storage.len() as u32);
        let codebook = storage
            .metadata
            .codebook
            .as_ref()
            .unwrap()
            .values()
            .as_primitive::<datatypes::Float32Type>();
        let pq_codes = storage.pq_code.values();
        let qu = get_centroids(
            codebook.values(),
            storage.metadata.nbits,
            storage.metadata.num_sub_vectors,
            storage.metadata.dimension,
            get_pq_code(
                pq_codes,
                storage.metadata.nbits,
                storage.metadata.num_sub_vectors,
                u,
            ),
        );
        let qv = get_centroids(
            codebook.values(),
            storage.metadata.nbits,
            storage.metadata.num_sub_vectors,
            storage.metadata.dimension,
            get_pq_code(
                pq_codes,
                storage.metadata.nbits,
                storage.metadata.num_sub_vectors,
                v,
            ),
        );
        let expected = storage.distance_type.func()(&qu, &qv);
        let dist_calc = storage.dist_calculator_from_id(u);
        assert!((dist_calc.distance(v) - expected).abs() < 1e-4);
        assert!((storage.dist_between(u, v) - expected).abs() < 1e-4);
    }

    // The first half of the rows is rewritten in order into frag 1; the second
    // half is deleted. remap must behave the same in either RowAddrRemap mode.
    fn pq_remap_compact() -> RowAddrRemap {
        use lance_core::utils::row_addr_remap::GroupInput;
        use roaring::RoaringTreemap;
        RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter((0..TOTAL / 2).map(|i| i as u64)),
            old_frag_ids: vec![0],
            new_frags: vec![(1, (TOTAL / 2) as u32)],
        }])
        .unwrap()
    }

    fn pq_remap_explicit() -> RowAddrRemap {
        RowAddrRemap::direct(
            (0..TOTAL / 2)
                .map(|i| (i as u64, Some((1u64 << 32) | i as u64)))
                .chain((TOTAL / 2..TOTAL).map(|i| (i as u64, None)))
                .collect(),
        )
    }

    #[tokio::test]
    async fn test_async_remap_preserves_transposed_codes() {
        #[derive(Debug)]
        struct Mapping(RowAddrRemap);
        #[async_trait::async_trait]
        impl lance_index_core::remapping::BatchRowIdRemapper for Mapping {
            async fn remap_row_ids(&self, ids: &[u64]) -> Result<Vec<Option<u64>>> {
                Ok(ids
                    .iter()
                    .map(|id| self.0.get(*id).unwrap_or(Some(*id)))
                    .collect())
            }
        }
        let storage = create_pq_storage().await;
        let mapping = pq_remap_compact();
        let expected = storage.remap(&mapping).unwrap();
        let remapping = Mapping(mapping);
        let row_id_idx = storage.batch.schema().index_of(ROW_ID).unwrap();
        let (batch, remapper) = lance_index_core::remapping::remap_row_ids_preserving_layout_async(
            &remapping,
            storage.batch.clone(),
            row_id_idx,
        )
        .await
        .unwrap();
        assert_eq!(
            batch.column_by_name(PQ_CODE_COLUMN),
            storage.batch.column_by_name(PQ_CODE_COLUMN)
        );
        let mut metadata = storage.metadata.clone();
        metadata.transposed = true;
        let actual = ProductQuantizationStorage::try_from_batch_with_remapper(
            batch,
            &metadata,
            storage.distance_type,
            Some(remapper),
        )
        .unwrap();
        assert_eq!(actual.row_ids, expected.row_ids);
        assert_eq!(actual.pq_code, expected.pq_code);
    }

    #[rstest]
    #[case(pq_remap_compact())]
    #[case(pq_remap_explicit())]
    #[tokio::test]
    async fn test_remap_with_extra_column(#[case] remap: RowAddrRemap) {
        let storage = create_pq_storage_with_extra_column().await;
        let new_storage = storage.remap(&remap).unwrap();
        assert_eq!(new_storage.len(), TOTAL / 2);
        assert_eq!(new_storage.row_ids.len(), TOTAL / 2);
        for (i, row_id) in new_storage.row_ids().enumerate() {
            // Rewritten row i lands at offset i of frag 1.
            assert_eq!(*row_id, (1u64 << 32) | i as u64);
        }
        assert_eq!(new_storage.batch.num_columns(), 2);
        assert!(new_storage.batch.column_by_name(ROW_ID).is_some());
        assert!(new_storage.batch.column_by_name(PQ_CODE_COLUMN).is_some());
    }
}
