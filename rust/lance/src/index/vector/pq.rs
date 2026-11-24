// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index::scalar::RowAddrTranslator;
use lance_index::scalar::RowAddrTranslatorRef;
use std::any::Any;
use std::sync::Arc;

use arrow::{
    array::{ArrayData, make_array},
    compute::concat,
};
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, UInt8Array, UInt64Array,
    cast::{AsArray, as_primitive_array},
    new_empty_array,
};
use arrow_array::{ArrayRef, Float32Array, UInt32Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_arrow::{BufferExt, DataTypeExt, FixedSizeListArrayExt};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::frag_reuse::CompactFragReuseIndex;
use lance_index::metrics::MetricsCollector;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::storage::{ProductQuantizationStorage, transpose};
use lance_index::vector::quantizer::{Quantization, QuantizationType, Quantizer};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    Index, IndexType,
    vector::{Query, pq::ProductQuantizer},
};
use lance_io::traits::Reader;
use lance_linalg::distance::{DistanceType, MetricType};
use log::{info, warn};
use roaring::RoaringBitmap;
use serde_json::json;
use tracing::{Level, instrument, span};
// Re-export
pub use lance_index::vector::pq::PQBuildParams;
use lance_linalg::kernels::normalize_fsl_owned;

use super::VectorIndex;
use crate::Dataset;
use crate::index::prefilter::PreFilter;
use crate::index::vector::utils::maybe_sample_training_data;
use crate::io::exec::knn::KNN_INDEX_SCHEMA;
use crate::{Error, Result};

/// Product Quantization Index.
///
#[derive(Clone)]
pub struct PQIndex {
    /// Product quantizer.
    pub pq: ProductQuantizer,

    /// PQ code
    /// the PQ codes are stored in a transposed way,
    /// call `Self::get_pq_codes` to get the PQ code for a specific vector.
    pub code: Option<Arc<UInt8Array>>,

    /// ROW Id used to refer to the actual row in dataset.
    pub row_ids: Option<Arc<UInt64Array>>,

    /// Metric type.
    metric_type: MetricType,

    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,

    /// Bytes of PQ code each row holds, in the file and in memory. Equal to
    /// [`packed_code_bytes`] unless [`stores_packed_codes`] says the file holds
    /// another layout; such an index is read and remapped at one byte per
    /// sub-vector, as earlier releases did, and refuses search and merging.
    code_bytes_per_vector: usize,
}

/// Bytes of PQ code per row in the layout writers since 0.20 use: two 4-bit
/// codes per byte, since they reject an odd sub-vector count at 4 bits, and
/// one byte per sub-vector otherwise.
fn packed_code_bytes(pq: &ProductQuantizer) -> usize {
    if pq.num_bits == 4 && pq.num_sub_vectors.is_multiple_of(2) {
        pq.num_sub_vectors / 2
    } else {
        pq.num_sub_vectors
    }
}

/// Whether a legacy IVF_PQ file stores [`packed_code_bytes`] per row, which
/// only 4-bit files with an even sub-vector count can fail.
///
/// A legacy file holds its partitions back to back from offset 0, each one's
/// codes followed by its `u64` row ids, with the index protobuf right after
/// them at `metadata_offset`. A 4-bit file fails this when a release before
/// 0.20 wrote it at one byte per sub-vector, or when a release before this fix
/// remapped or merged a packed file: those read pages at that width.
pub(crate) fn stores_packed_codes(
    pq: &ProductQuantizer,
    num_rows: usize,
    metadata_offset: usize,
) -> bool {
    let row_bytes = packed_code_bytes(pq) + std::mem::size_of::<u64>();
    pq.num_sub_vectors == packed_code_bytes(pq)
        || num_rows.checked_mul(row_bytes) == Some(metadata_offset)
}

async fn read_legacy_index_values(
    reader: &dyn Reader,
    data_type: &DataType,
    offset: usize,
    length: usize,
) -> Result<ArrayRef> {
    if length == 0 {
        return Ok(new_empty_array(data_type));
    }

    let byte_length = length
        .checked_mul(data_type.byte_width())
        .ok_or_else(|| Error::index("legacy IVF page byte length overflow".to_string()))?;
    let end = offset
        .checked_add(byte_length)
        .ok_or_else(|| Error::index("legacy IVF page offset overflow".to_string()))?;
    let bytes = reader.get_range(offset..end).await?;
    let buffer = if bytes.len() < byte_length {
        arrow_buffer::Buffer::copy_bytes_bytes(bytes, byte_length)
    } else {
        arrow_buffer::Buffer::from_bytes_bytes(bytes, data_type.byte_width() as u64)
    };
    let data = ArrayData::builder(data_type.clone())
        .len(length)
        .null_count(0)
        .add_buffer(buffer)
        .build()?;
    Ok(make_array(data))
}

impl DeepSizeOf for PQIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.pq.deep_size_of_children(context)
            + self
                .code
                .as_ref()
                .map(|code| {
                    if context.mark_seen(Arc::as_ptr(code) as *const () as usize) {
                        (code.as_ref() as &dyn arrow_array::Array).deep_size_of_children(context)
                    } else {
                        0
                    }
                })
                .unwrap_or(0)
            + self
                .row_ids
                .as_ref()
                .map(|row_ids| {
                    if context.mark_seen(Arc::as_ptr(row_ids) as *const () as usize) {
                        (row_ids.as_ref() as &dyn arrow_array::Array).deep_size_of_children(context)
                    } else {
                        0
                    }
                })
                .unwrap_or(0)
    }
}

impl std::fmt::Debug for PQIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQ(m={}, nbits={}, {})",
            self.pq.code_dim(),
            self.pq.num_bits,
            self.metric_type
        )
    }
}

impl PQIndex {
    /// Load a PQ index (page) from the disk.
    pub(crate) fn new(
        pq: ProductQuantizer,
        metric_type: MetricType,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    ) -> Self {
        let code_bytes_per_vector = packed_code_bytes(&pq);
        Self {
            code: None,
            row_ids: None,
            pq,
            metric_type,
            frag_reuse_index,
            code_bytes_per_vector,
        }
    }

    /// Reads the index file at one byte per sub-vector, for a file that does
    /// not store the packed layout, see [`stores_packed_codes`].
    pub(crate) fn with_unpacked_codes(mut self) -> Self {
        self.code_bytes_per_vector = self.pq.num_sub_vectors;
        self
    }

    /// Filter the row id and PQ code arrays based on the pre-filter.
    fn filter_arrays(
        pre_filter: &dyn PreFilter,
        code: Arc<UInt8Array>,
        row_ids: Arc<UInt64Array>,
        _num_sub_vectors: i32,
    ) -> Result<(Arc<UInt8Array>, Arc<UInt64Array>)> {
        let num_vectors = row_ids.len();
        if num_vectors == 0 {
            warn!("Filtering on empty PQ code array");
            return Ok((code, row_ids));
        }
        let indices_to_keep = pre_filter.filter_row_ids(Box::new(row_ids.values().iter()));
        let indices_to_keep = UInt64Array::from(indices_to_keep);

        let row_ids = take(row_ids.as_ref(), &indices_to_keep, None)?;
        let row_ids = Arc::new(as_primitive_array(&row_ids).clone());

        let code = code
            .values()
            .chunks_exact(num_vectors)
            .flat_map(|c| {
                let mut filtered = Vec::with_capacity(indices_to_keep.len());
                for idx in indices_to_keep.values() {
                    filtered.push(c[*idx as usize]);
                }
                filtered
            })
            .collect();

        Ok((Arc::new(code), row_ids))
    }

    /// Bytes of PQ code each row of `code` holds, and the width remapping
    /// writes back, see [`stores_packed_codes`].
    pub(crate) fn code_bytes_per_vector(&self) -> usize {
        self.code_bytes_per_vector
    }

    /// Fails for a 4-bit index that does not store two codes per byte, which
    /// search and merging cannot read, see [`stores_packed_codes`].
    pub(crate) fn check_codes_readable(&self) -> Result<()> {
        if self.pq.num_bits != 4 || self.code_bytes_per_vector * 2 == self.pq.num_sub_vectors {
            return Ok(());
        }
        Err(Error::not_supported(format!(
            "legacy IVF_PQ index stores {} bytes per row for {} 4-bit sub-vectors, not \
             the packed layout Lance 0.20 and later write: a release before 0.20 wrote \
             it, or an earlier release's compaction or index merge rewrote it from \
             misread codes. Rebuild the index: optimize it with retrain set, or drop and \
             recreate it. 4-bit PQ needs an even number of sub-vectors, so an index with \
             an odd count must be recreated with other parameters.",
            self.code_bytes_per_vector, self.pq.num_sub_vectors
        )))
    }

    fn get_pq_codes(transposed_codes: &UInt8Array, vec_idx: usize, num_vectors: usize) -> Vec<u8> {
        transposed_codes
            .values()
            .iter()
            .skip(vec_idx)
            .step_by(num_vectors)
            .cloned()
            .collect()
    }
}

#[async_trait]
impl Index for PQIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    async fn prewarm(&self) -> Result<()> {
        // Nothing is lazily loaded in PQ index, so we can return immediately.
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "index_type": "PQ",
            "nbits": self.pq.num_bits,
            "num_sub_vectors": self.pq.code_dim(),
            "dimension": self.pq.dimension,
            "metric_type": self.metric_type.to_string(),
        }))
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        if let Some(row_ids) = &self.row_ids {
            let mut frag_ids = row_ids
                .values()
                .iter()
                .map(|&row_id| RowAddress::from(row_id).fragment_id())
                .collect::<Vec<_>>();
            frag_ids.sort();
            frag_ids.dedup();
            Ok(RoaringBitmap::from_sorted_iter(frag_ids).unwrap())
        } else {
            Err(Error::index(
                "PQIndex::calculate_included_frags: PQ is not initialized".to_string(),
            ))
        }
    }
}

impl PQIndex {
    /// The one remap implementation behind the legacy `remap` and
    /// `remap_streaming`: this page's addresses are the unit of translation.
    async fn remap_with(&mut self, mapping: RowAddrTranslatorRef<'_>) -> Result<()> {
        let num_vectors = self.row_ids.as_ref().unwrap().len();
        // One page's addresses are the unit of translation.
        let mapping = mapping
            .resolve(self.row_ids.as_ref().unwrap().values().iter().copied())
            .await?;
        let row_ids = self.row_ids.as_ref().unwrap().values().iter();
        let transposed_codes = self.code.as_ref().unwrap();
        let remapped = row_ids
            .enumerate()
            .filter_map(|(vec_idx, old_row_id)| {
                let new_row_id = mapping.get(*old_row_id);
                // If the row id is not in the mapping then this row is not remapped and we keep as is
                let new_row_id = new_row_id.unwrap_or(Some(*old_row_id));
                new_row_id.map(|new_row_id| {
                    (
                        new_row_id,
                        Self::get_pq_codes(transposed_codes, vec_idx, num_vectors),
                    )
                })
            })
            .collect::<Vec<_>>();

        self.row_ids = Some(Arc::new(UInt64Array::from_iter_values(
            remapped.iter().map(|(row_id, _)| *row_id),
        )));

        let pq_codes =
            UInt8Array::from_iter_values(remapped.into_iter().flat_map(|(_, code)| code));
        let transposed_codes = transpose(
            &pq_codes,
            self.row_ids.as_ref().unwrap().len(),
            self.code_bytes_per_vector(),
        );
        self.code = Some(Arc::new(transposed_codes));
        Ok(())
    }
}

#[async_trait]
impl VectorIndex for PQIndex {
    /// Search top-k nearest neighbors for `key` within one PQ partition.
    ///
    #[instrument(level = "debug", skip_all, name = "PQIndex::search")]
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        if self.code.is_none() || self.row_ids.is_none() {
            return Err(Error::index(
                "PQIndex::search: PQ is not initialized".to_string(),
            ));
        }
        self.check_codes_readable()?;
        pre_filter.wait_for_ready().await?;

        let code = self.code.as_ref().unwrap().clone();
        let row_ids = self.row_ids.as_ref().unwrap().clone();

        metrics.record_comparisons(row_ids.len());

        let pq = self.pq.clone();
        let query = query.clone();
        let num_sub_vectors = self.pq.code_dim() as i32;
        let search = move || {
            let (code, row_ids) = if pre_filter.is_empty() {
                Ok((code, row_ids))
            } else {
                Self::filter_arrays(pre_filter.as_ref(), code, row_ids, num_sub_vectors)
            }?;

            // Pre-compute distance table for each sub-vector.
            let distances = pq.compute_distances(query.key.as_ref(), &code)?;

            debug_assert_eq!(distances.len(), row_ids.len());

            let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
            if query.lower_bound.is_none() && query.upper_bound.is_none() {
                let indices = sort_to_indices(&distances, None, Some(limit))?;
                let distances = take(&distances, &indices, None)?;
                let row_ids = take(row_ids.as_ref(), &indices, None)?;
                Ok(RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![distances, row_ids],
                )?)
            } else {
                let indices = sort_to_indices(&distances, None, None)?;
                let mut dists = Vec::with_capacity(limit);
                let mut ids = Vec::with_capacity(limit);
                for idx in indices.values().iter() {
                    let dist = distances.value(*idx as usize);
                    let id = row_ids.value(*idx as usize);
                    if query.lower_bound.is_some_and(|lb| dist < lb) {
                        continue;
                    }
                    if query.upper_bound.is_some_and(|ub| dist >= ub) {
                        break;
                    }

                    dists.push(dist);
                    ids.push(id);

                    if dists.len() >= limit {
                        break;
                    }
                }
                let dists = Arc::new(Float32Array::from(dists));
                let ids = Arc::new(UInt64Array::from(ids));
                Ok(RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![dists, ids],
                )?)
            }
        };
        spawn_cpu(search).await
    }

    fn find_partitions(&self, _: &Query) -> Result<(UInt32Array, Float32Array)> {
        unimplemented!("only for IVF")
    }

    fn total_partitions(&self) -> usize {
        1
    }

    async fn search_in_partition(
        &self,
        _: usize,
        _: &Query,
        _: Arc<dyn PreFilter>,
        _: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        unimplemented!("only for IVF")
    }

    fn is_loadable(&self) -> bool {
        true
    }

    fn use_residual(&self) -> bool {
        ProductQuantizer::use_residual(self.metric_type)
    }

    /// Load a PQ index (page) from the disk.
    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let code_bytes_per_vector = self.code_bytes_per_vector;
        let pq_code_length = code_bytes_per_vector * length;
        let pq_codes =
            read_legacy_index_values(reader.as_ref(), &DataType::UInt8, offset, pq_code_length)
                .await?;

        let row_id_offset = offset + pq_code_length /* *1 */;
        let row_ids =
            read_legacy_index_values(reader.as_ref(), &DataType::UInt64, row_id_offset, length)
                .await?;

        let pq_codes = transpose(
            pq_codes.as_primitive(),
            row_ids.len(),
            code_bytes_per_vector,
        );

        let (primitive_row_ids, transposed_pq_codes) =
            if let Some(frag_reuse_index_ref) = self.frag_reuse_index.as_ref() {
                let num_vectors = row_ids.len();
                let row_ids = row_ids.as_primitive::<UInt64Type>().values().iter();
                let (remapped_row_ids, remapped_pq_codes): (Vec<u64>, Vec<Vec<u8>>) = row_ids
                    .enumerate()
                    .filter_map(|(vec_idx, old_row_id)| {
                        let new_row_id = frag_reuse_index_ref.remap_row_id(*old_row_id);
                        new_row_id.map(|new_row_id| {
                            (
                                new_row_id,
                                Self::get_pq_codes(&pq_codes, vec_idx, num_vectors),
                            )
                        })
                    })
                    .unzip();
                let transposed_codes = transpose(
                    &UInt8Array::from_iter_values(remapped_pq_codes.into_iter().flatten()),
                    remapped_row_ids.len(),
                    code_bytes_per_vector,
                );
                (
                    Arc::new(UInt64Array::from_iter_values(remapped_row_ids)),
                    Arc::new(transposed_codes),
                )
            } else {
                (Arc::new(row_ids.as_primitive().clone()), Arc::new(pq_codes))
            };

        Ok(Box::new(Self {
            code: Some(transposed_pq_codes),
            row_ids: Some(primitive_row_ids),
            pq: self.pq.clone(),
            metric_type: self.metric_type,
            frag_reuse_index: self.frag_reuse_index.clone(),
            code_bytes_per_vector,
        }))
    }

    async fn to_batch_stream(&self, with_vector: bool) -> Result<SendableRecordBatchStream> {
        let row_ids = self.row_ids.clone().ok_or(Error::index(
            "PQIndex::to_batch_stream: row ids not loaded for PQ".to_string(),
        ))?;

        let num_rows = row_ids.len();
        let mut fields = vec![ROW_ID_FIELD.clone()];
        let mut columns: Vec<ArrayRef> = vec![row_ids];
        if with_vector {
            let transposed_codes = self.code.clone().ok_or(Error::index(
                "PQIndex::to_batch_stream: PQ codes not loaded for PQ".to_string(),
            ))?;
            let code_bytes_per_vector = self.code_bytes_per_vector;
            let original_codes = transpose(&transposed_codes, code_bytes_per_vector, num_rows);
            fields.push(Field::new(
                self.pq.column(),
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    code_bytes_per_vector as i32,
                ),
                true,
            ));
            columns.push(Arc::new(FixedSizeListArray::try_new_from_values(
                original_codes,
                code_bytes_per_vector as i32,
            )?));
        }

        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        let stream = RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::once(futures::future::ready(Ok(batch))),
        );
        Ok(Box::pin(stream))
    }

    fn num_rows(&self) -> u64 {
        self.row_ids
            .as_ref()
            .map_or(0, |row_ids| row_ids.len() as u64)
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &u64> + '_> {
        match self.row_ids.as_ref() {
            Some(row_ids) => Box::new(row_ids.values().iter()),
            None => Box::new(std::iter::empty()),
        }
    }

    async fn remap(&mut self, mapping: &RowAddrRemap) -> Result<()> {
        self.remap_with(mapping.into()).await
    }

    async fn remap_streaming(&mut self, translator: &RowAddrTranslator) -> Result<()> {
        self.remap_with(translator.as_ref()).await
    }

    fn ivf_model(&self) -> &IvfModel {
        unimplemented!("only for IVF")
    }

    fn quantizer(&self) -> Quantizer {
        Quantizer::Product(self.pq.clone())
    }

    fn partition_size(&self, _: usize) -> usize {
        unimplemented!("only for IVF")
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        (SubIndexType::Flat, QuantizationType::Product)
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
    }
}

/// Train Product Quantizer model.
///
/// Parameters:
/// - `dataset`: The dataset to train the PQ model.
/// - `column`: The column name of the dataset.
/// - `dim`: The dimension of the vectors.
/// - `metric_type`: The metric type of the vectors.
/// - `params`: The parameters to train the PQ model.
/// - `ivf`: Optional IVF model for computing residuals. The model must contain nonempty centroids
///   when training a new codebook.
pub async fn build_pq_model(
    dataset: &Dataset,
    column: &str,
    dim: usize,
    metric_type: MetricType,
    params: &PQBuildParams,
    ivf: Option<&IvfModel>,
) -> Result<ProductQuantizer> {
    build_pq_model_in_fragments(dataset, column, dim, metric_type, params, ivf, None).await
}

pub async fn build_pq_model_in_fragments(
    dataset: &Dataset,
    column: &str,
    dim: usize,
    metric_type: MetricType,
    params: &PQBuildParams,
    ivf: Option<&IvfModel>,
    fragment_ids: Option<&[u32]>,
) -> Result<ProductQuantizer> {
    if let Some(codebook) = &params.codebook {
        let (_, value_type) =
            crate::index::vector::utils::get_vector_type(dataset.schema(), column)?;
        lance_index::vector::pq::validate_codebook_value_type(codebook.data_type(), &value_type)?;
        lance_index::vector::pq::validate_supplied_codebook(
            codebook.len(),
            dim,
            params.num_sub_vectors,
            params.num_bits,
        )?;
        let dt = if metric_type == MetricType::Cosine {
            info!("Normalize training data for PQ training: Cosine");
            MetricType::L2
        } else {
            metric_type
        };

        return match codebook.data_type() {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(ProductQuantizer::new(
                params.num_sub_vectors,
                params.num_bits as u32,
                dim,
                FixedSizeListArray::try_new_from_values(
                    codebook.slice(0, codebook.len()),
                    dim as i32,
                )?,
                dt,
            )),
            _ => Err(Error::index(format!(
                "Wrong codebook data type: {:?}",
                codebook.data_type()
            ))),
        };
    }

    let ivf_centroids = ivf
        .map(|model| {
            model
                .centroids_array()
                .filter(|centroids| !centroids.values().is_empty())
                .ok_or_else(|| {
                    Error::invalid_input("PQ residual training requires nonempty IVF centroids")
                })
        })
        .transpose()?;

    info!(
        "Start to train PQ code: PQ{}, bits={}",
        params.num_sub_vectors, params.num_bits
    );
    // 2^num_bits panics on an unrepresentable num_bits, so it stays below the
    // supplied-codebook branch, which rejects that input instead.
    let num_codes = lance_index::vector::pq::num_centroids(params.num_bits as u32);
    let expected_sample_size = num_codes * params.sample_rate;
    info!(
        "Loading training data for PQ. Sample size: {}",
        expected_sample_size
    );
    let start = std::time::Instant::now();
    let mut training_data =
        maybe_sample_training_data(dataset, column, expected_sample_size, fragment_ids).await?;
    info!(
        "Finished loading training data in {:02} seconds",
        start.elapsed().as_secs_f32()
    );
    assert_eq!(training_data.logical_null_count(), 0);

    info!(
        "starting to compute partitions for PQ training, sample size: {}",
        training_data.len()
    );

    if metric_type == MetricType::Cosine {
        info!("Normalize training data for PQ training: Cosine");
        training_data = normalize_fsl_owned(training_data)?;
    }

    let training_data = if let Some(centroids) = ivf_centroids {
        // Compute residual for PQ training.
        //
        // TODO: consolidate IVF models to `lance_index`.
        let ivf2 = lance_index::vector::ivf::new_ivf_transformer(
            centroids.clone(),
            MetricType::L2,
            vec![],
        );
        span!(Level::INFO, "compute residual for PQ training")
            .in_scope(|| ivf2.compute_residual(&training_data))?
    } else {
        training_data
    };

    if training_data.len() < num_codes {
        warn!(
            "Skip PQ training: only {} rows available, needs >= {}",
            training_data.len(),
            num_codes
        );
        return Err(Error::unprocessable(format!(
            "Not enough rows to train PQ. Requires {num_codes} rows but only {available} available",
            available = training_data.len()
        )));
    }

    info!("Start train PQ: params={:#?}", params);
    let pq = ProductQuantizer::build(&training_data, DistanceType::L2, params)?;
    info!("Trained PQ in: {} seconds", start.elapsed().as_secs_f32());
    Ok(pq)
}

pub(crate) fn build_pq_storage(
    distance_type: DistanceType,
    row_ids: Arc<dyn Array>,
    code_array: Vec<Arc<dyn Array>>,
    pq: ProductQuantizer,
) -> Result<ProductQuantizationStorage> {
    let pq_arrs = code_array.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let pq_column = concat(&pq_arrs)?;
    std::mem::drop(code_array);

    let pq_batch = RecordBatch::try_from_iter_with_nullable(vec![
        (ROW_ID, row_ids, true),
        (pq.column(), pq_column, false),
    ])?;
    let pq_store = ProductQuantizationStorage::new(
        pq.codebook.clone(),
        pq_batch,
        pq.num_bits,
        pq.code_dim(),
        pq.dimension,
        distance_type,
        false,
        // TODO: support auto-remap with frag_reuse_index for HNSW
        None,
    )?;

    Ok(pq_store)
}
#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::{ops::Range, sync::Mutex};

    use arrow::datatypes::Float32Type;
    use arrow_array::RecordBatchIterator;
    use arrow_array::types::{Int64Type, UInt8Type};
    use arrow_schema::{Field, Schema};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::{TempDir, TempObjFile, TempStrDir};
    use lance_io::object_store::ObjectStore;
    use lance_io::traits::Writer;
    use lance_io::utils::{read_last_block, read_metadata_offset, read_version};
    use lance_linalg::kernels::normalize_fsl;
    use lance_table::format::IndexMetadata;
    use object_store::path::Path;
    use tokio::io::AsyncWriteExt;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::DatasetIndexExt;
    use crate::index::vector::ivf::build_ivf_model;
    use crate::utils::test::copy_test_data_to_tmp;
    use lance_index::INDEX_FILE_NAME;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::vector::DEFAULT_QUERY_PARALLELISM;
    use lance_index::vector::PQ_CODE_COLUMN;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_select::RowAddrMask;
    use lance_testing::datagen::{
        generate_random_array_with_range, generate_random_array_with_seed,
    };

    const DIM: usize = 128;

    #[tokio::test]
    async fn empty_legacy_ivf_pq_partition_does_not_read_object_store() {
        let object_store = ObjectStore::memory();
        let reader = object_store
            .open(&Path::from("missing-index"))
            .await
            .unwrap();
        let codebook_values = Float32Array::from_iter_values((0..256).map(|value| value as f32));
        let codebook = FixedSizeListArray::try_new_from_values(codebook_values, 1).unwrap();
        let index = PQIndex::new(
            ProductQuantizer::new(1, 8, 1, codebook, DistanceType::L2),
            MetricType::L2,
            None,
        );

        let loaded = index.load(reader.into(), 1, 0).await.unwrap();

        assert_eq!(loaded.num_rows(), 0);
    }

    #[tokio::test]
    async fn legacy_index_values_are_read_from_the_requested_offset() {
        let path = TempObjFile::default();
        let object_store = ObjectStore::local();
        let mut writer = object_store.create(&path).await.unwrap();
        writer.write_all(&[0xFF]).await.unwrap();
        writer.write_all(&11_u64.to_le_bytes()).await.unwrap();
        writer.write_all(&12_u64.to_le_bytes()).await.unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let reader = object_store.open(&path).await.unwrap();
        let values = read_legacy_index_values(reader.as_ref(), &DataType::UInt64, 1, 2)
            .await
            .unwrap();
        assert_eq!(
            values.as_primitive::<UInt64Type>(),
            &UInt64Array::from_iter_values([11, 12])
        );
    }

    async fn generate_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range::<Float32Type>(1000 * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(
            Schema::new(vec![Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            )])
            .with_metadata(metadata),
        );
        let fsl = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![fsl.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        (Dataset::write(batches, test_uri, None).await.unwrap(), fsl)
    }

    #[tokio::test]
    async fn test_build_pq_model_l2() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (dataset, _) = generate_dataset(test_uri, 100.0..120.0).await;

        let centroids = generate_random_array_with_range::<Float32Type>(4 * DIM, -1.0..1.0);
        let fsl = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf = IvfModel::new(fsl, None);
        let params = PQBuildParams {
            max_iters: 2,
            sample_rate: 4,
            ..PQBuildParams::new(16, 8)
        };
        let pq = build_pq_model(&dataset, "vector", DIM, MetricType::L2, &params, Some(&ivf))
            .await
            .unwrap();

        assert_eq!(pq.code_dim(), 16);
        assert_eq!(pq.num_bits, 8);
        assert_eq!(pq.dimension, DIM);

        let codebook = pq.codebook;
        assert_eq!(codebook.len(), 256);
        codebook
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((99.0..121.0).contains(v));
            });
    }

    #[tokio::test]
    async fn test_build_pq_model_rejects_empty_ivf_centroids() {
        let (dataset, vectors) = generate_dataset("memory://", 100.0..120.0).await;
        let ivf = IvfModel::new(vectors.as_ref().slice(0, 0), None);
        let params = PQBuildParams::new(16, 8);
        let err = build_pq_model(&dataset, "vector", DIM, MetricType::L2, &params, Some(&ivf))
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(
            err.to_string().contains("nonempty IVF centroids"),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn test_build_pq_model_cosine() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (dataset, vectors) = generate_dataset(test_uri, 100.0..120.0).await;

        let ivf_params = IvfBuildParams::new(4);
        let ivf = build_ivf_model(
            &dataset,
            "vector",
            DIM,
            MetricType::Cosine,
            &ivf_params,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();
        let params = PQBuildParams {
            max_iters: 2,
            sample_rate: 4,
            ..PQBuildParams::new(16, 8)
        };
        let pq = build_pq_model(
            &dataset,
            "vector",
            DIM,
            MetricType::Cosine,
            &params,
            Some(&ivf),
        )
        .await
        .unwrap();

        assert_eq!(pq.code_dim(), 16);
        assert_eq!(pq.num_bits, 8);
        assert_eq!(pq.dimension, DIM);

        let codebook = pq.codebook.clone();
        assert_eq!(codebook.len(), 256);
        codebook
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((-1.0..1.0).contains(v));
            });

        let vectors = normalize_fsl(&vectors).unwrap();
        let row = vectors.slice(0, 1);

        let ivf2 = lance_index::vector::ivf::new_ivf_transformer(
            ivf.centroids.clone().unwrap(),
            MetricType::L2,
            vec![],
        );

        let residual_query = ivf2.compute_residual(&row).unwrap();
        let pq_code = pq.quantize(&residual_query).unwrap();
        let distances = pq
            .compute_distances(
                &residual_query.value(0),
                pq_code.as_fixed_size_list().values().as_primitive(),
            )
            .unwrap();
        assert!(
            distances.values().iter().all(|&d| d <= 0.001),
            "distances: {:?}",
            distances
        );
    }

    #[tokio::test]
    async fn test_build_pq_model_insufficient_rows_returns_prereq() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let dim = 16;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        )]));

        let vectors = generate_random_array_with_seed::<Float32Type>(dim * 10, [11u8; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, dim as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let params = PQBuildParams::new(16, 8);
        let err = build_pq_model(&dataset, "vector", dim, MetricType::L2, &params, None)
            .await
            .unwrap_err();

        assert!(matches!(err, Error::Unprocessable { .. }));
    }

    /// The supplied codebook is checked before 2^num_bits is derived, so a
    /// num_bits that does not fit reports the input rather than overflowing.
    #[tokio::test]
    async fn test_build_pq_model_rejects_unrepresentable_num_bits() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let dim = 16;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        )]));

        let vectors = generate_random_array_with_seed::<Float32Type>(dim * 10, [11u8; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, dim as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let codebook = Arc::new(generate_random_array_with_seed::<Float32Type>(
            dim, [12u8; 32],
        ));
        let params = PQBuildParams::with_codebook(4, usize::BITS as usize, codebook);
        let err = build_pq_model(&dataset, "vector", dim, MetricType::L2, &params, None)
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(err.to_string().contains("not representable"), "got {err}");
    }

    /// A legacy IVF_PQ index with `num_bits=4` over 512 rows in four
    /// fragments, written by pylance 0.24.1, which packs two codes per byte.
    /// See `test_data/v0.24.1/datagen.py`.
    const LEGACY_4BIT_PACKED: &str = "v0.24.1/legacy_ivf_pq_4bit";
    const LEGACY_4BIT_PARTITIONS: usize = 2;

    async fn open_legacy_4bit_fixture(fixture: &str) -> (TempDir, Dataset) {
        let test_dir = copy_test_data_to_tmp(fixture).unwrap();
        let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
        let footer = index_file_footer(&dataset).await;
        assert_eq!(
            read_version(&footer).unwrap(),
            (0, 1),
            "the fixture must remain a legacy IVF_PQ index file"
        );
        (test_dir, dataset)
    }

    async fn index_file_footer(dataset: &Dataset) -> bytes::Bytes {
        let indices = dataset.load_indices().await.unwrap();
        let index = indices.iter().find(|i| i.name == "vector_idx").unwrap();
        let path = dataset
            .indices_dir()
            .join(index.uuid.to_string())
            .join(INDEX_FILE_NAME);
        let reader = dataset.object_store.open(&path).await.unwrap();
        read_last_block(reader.as_ref()).await.unwrap()
    }

    /// Maps the `id` of every row in the index to its PQ code, read back
    /// through the index partitions.
    async fn legacy_4bit_codes_by_id(dataset: &Dataset) -> HashMap<i64, Vec<u8>> {
        let rows = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let ids_by_row_id: HashMap<u64, i64> = rows[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .zip(
                rows["id"]
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect();

        let mut codes = HashMap::new();
        for partition in 0..LEGACY_4BIT_PARTITIONS {
            let stream = dataset
                .read_index_partition("vector_idx", partition, true)
                .await
                .unwrap();
            for batch in stream.try_collect::<Vec<_>>().await.unwrap() {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let partition_codes = batch[PQ_CODE_COLUMN].as_fixed_size_list();
                assert_eq!(partition_codes.value_length(), 2, "4 sub-vectors x 4 bits");
                for (i, row_id) in row_ids.values().iter().enumerate() {
                    let id = *ids_by_row_id.get(row_id).unwrap_or_else(|| {
                        panic!("partition {partition}: row id {row_id} is not a live row")
                    });
                    let code = partition_codes
                        .value(i)
                        .as_primitive::<UInt8Type>()
                        .values()
                        .to_vec();
                    assert!(codes.insert(id, code).is_none(), "id {id} indexed twice");
                }
            }
        }
        assert_eq!(codes.len(), ids_by_row_id.len());
        codes
    }

    /// Queries every 37th row with its own vector and expects the row in its
    /// top 10.
    async fn assert_legacy_4bit_rows_find_themselves(dataset: &Dataset) {
        let rows = dataset
            .scan()
            .project(&["id", "vector"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = rows["id"].as_primitive::<Int64Type>();
        let vectors = rows["vector"].as_fixed_size_list();
        for row in (0..rows.num_rows()).step_by(37) {
            let results = dataset
                .scan()
                .project(&["id"])
                .unwrap()
                .nearest("vector", vectors.value(row).as_ref(), 10)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let found = results["id"].as_primitive::<Int64Type>().values();
            assert!(
                found.contains(&ids.value(row)),
                "id {} not in its own top 10: {found:?}",
                ids.value(row)
            );
        }
    }

    #[tokio::test]
    async fn test_legacy_4bit_ivf_pq_search() {
        let (_test_dir, dataset) = open_legacy_4bit_fixture(LEGACY_4BIT_PACKED).await;

        assert_eq!(legacy_4bit_codes_by_id(&dataset).await.len(), 512);
        assert_legacy_4bit_rows_find_themselves(&dataset).await;
    }

    /// Immediate remap rewrites the index file in the packed layout; deferred
    /// remap keeps the file and translates each page as it loads.
    #[rstest::rstest]
    #[case::immediate_remap(false)]
    #[case::deferred_remap(true)]
    #[tokio::test]
    async fn test_legacy_4bit_ivf_pq_compaction_keeps_codes(#[case] defer_index_remap: bool) {
        let (_test_dir, mut dataset) = open_legacy_4bit_fixture(LEGACY_4BIT_PACKED).await;
        let mut expected = legacy_4bit_codes_by_id(&dataset).await;
        let index_uuid = |indices: &[IndexMetadata]| {
            indices
                .iter()
                .find(|index| index.name == "vector_idx")
                .unwrap()
                .uuid
        };
        let old_uuid = index_uuid(&dataset.load_indices().await.unwrap());

        dataset.delete("id % 7 = 3").await.unwrap();
        let options = CompactionOptions {
            defer_index_remap,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!((metrics.fragments_removed, metrics.fragments_added), (4, 1));

        let new_uuid = index_uuid(&dataset.load_indices().await.unwrap());
        assert_eq!(new_uuid == old_uuid, defer_index_remap);
        expected.retain(|id, _| id % 7 != 3);
        assert_eq!(legacy_4bit_codes_by_id(&dataset).await, expected);
        assert_legacy_4bit_rows_find_themselves(&dataset).await;
        // Deferred remap keeps the file written for all 512 rows.
        let file_rows = if defer_index_remap {
            512
        } else {
            expected.len()
        };
        assert_eq!(
            read_metadata_offset(&index_file_footer(&dataset).await).unwrap(),
            file_rows * (2 + 8),
            "the index file packs two 4-bit codes per byte"
        );
    }

    /// Two legacy 4-bit files that store one byte per sub-vector. pylance
    /// 0.19.2 wrote the first so. The second is the 0.24.1 fixture after a
    /// compaction on 14.0.3, which read each packed page at one byte per
    /// sub-vector and wrote the misread codes and row ids back. See
    /// `datagen.py` next to each. Search and merging ask for a rebuild,
    /// compaction keeps the stored bytes as 14.0.3 did, and retrain rebuilds.
    #[rstest::rstest]
    #[case::written_before_0_20("v0.19.2/legacy_ivf_pq_4bit_unpacked")]
    #[case::misremapped("v14.0.3/legacy_ivf_pq_4bit_misremapped")]
    #[tokio::test]
    async fn test_legacy_4bit_ivf_pq_unpacked_index_asks_for_rebuild(#[case] fixture: &str) {
        async fn assert_search_asks_for_rebuild(dataset: &Dataset) {
            let query = Float32Array::from(vec![0.5_f32; 16]);
            let err = dataset
                .scan()
                .nearest("vector", &query, 10)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap_err();
            assert!(err.to_string().contains("Rebuild the index"), "got {err}");
        }

        let (_test_dir, mut dataset) = open_legacy_4bit_fixture(fixture).await;
        assert_search_asks_for_rebuild(&dataset).await;

        dataset.delete("id < 300").await.unwrap();
        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert_eq!(metrics.fragments_added, 1);
        let mut index_rows = 0;
        for partition in 0..LEGACY_4BIT_PARTITIONS {
            let batches = dataset
                .read_index_partition("vector_idx", partition, false)
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            index_rows += batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        }
        assert_eq!(
            read_metadata_offset(&index_file_footer(&dataset).await).unwrap(),
            index_rows * (4 + 8),
            "the remapped index keeps one stored byte per sub-vector"
        );
        assert_search_asks_for_rebuild(&dataset).await;

        let schema = Arc::new(Schema::from(dataset.schema()));
        let vectors = generate_random_array_with_seed::<Float32Type>(64 * 16, [9u8; 32]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int64Array::from_iter_values(512..576)),
                Arc::new(FixedSizeListArray::try_new_from_values(vectors, 16).unwrap()),
            ],
        )
        .unwrap();
        dataset
            .append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
            .await
            .unwrap();
        let err = dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "got {err:?}");
        assert!(err.to_string().contains("Rebuild the index"), "got {err}");

        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();
        assert_legacy_4bit_rows_find_themselves(&dataset).await;
    }

    /// Writers since 0.20 reject an odd sub-vector count at 4 bits, so such a
    /// file can only come from an earlier release, which stored one byte per
    /// code. The page reads at that width and refuses search.
    #[tokio::test]
    async fn test_legacy_4bit_odd_sub_vector_page_keeps_one_byte_per_code() {
        let path = TempObjFile::default();
        let object_store = ObjectStore::local();
        let mut writer = object_store.create(&path).await.unwrap();
        writer.write_all(&[1, 2, 3, 4, 5, 6]).await.unwrap();
        writer.write_all(&11_u64.to_le_bytes()).await.unwrap();
        writer.write_all(&12_u64.to_le_bytes()).await.unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let codebook = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values((0..16 * 3).map(|value| value as f32)),
            3,
        )
        .unwrap();
        let index = PQIndex::new(
            ProductQuantizer::new(3, 4, 3, codebook, DistanceType::L2),
            MetricType::L2,
            None,
        );
        let reader = object_store.open(&path).await.unwrap();
        let page = index.load(reader.into(), 0, 2).await.unwrap();

        assert_eq!(page.row_ids().copied().collect::<Vec<_>>(), vec![11, 12]);
        let batches = page
            .to_batch_stream(true)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            batches[0][PQ_CODE_COLUMN]
                .as_fixed_size_list()
                .value_length(),
            3
        );
        let err = page
            .as_any()
            .downcast_ref::<PQIndex>()
            .unwrap()
            .check_codes_readable()
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "got {err:?}");
    }

    struct TestPreFilter {
        row_ids: Vec<u64>,
        is_empty_threads: Option<Arc<Mutex<Vec<String>>>>,
    }

    impl TestPreFilter {
        fn new(row_ids: Vec<u64>) -> Self {
            Self {
                row_ids,
                is_empty_threads: None,
            }
        }

        fn with_thread_capture(row_ids: Vec<u64>, threads: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                row_ids,
                is_empty_threads: Some(threads),
            }
        }
    }

    #[async_trait]
    impl PreFilter for TestPreFilter {
        async fn wait_for_ready(&self) -> Result<()> {
            Ok(())
        }

        fn is_empty(&self) -> bool {
            if let Some(threads) = &self.is_empty_threads {
                threads.lock().unwrap().push(
                    std::thread::current()
                        .name()
                        .unwrap_or("unknown")
                        .to_string(),
                );
            }
            self.row_ids.is_empty()
        }

        fn mask(&self) -> Arc<RowAddrMask> {
            RowAddrMask::all_rows().into()
        }

        fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
            row_ids
                .filter(|&row_id| self.row_ids.contains(row_id))
                .cloned()
                .collect()
        }
    }

    #[test]
    fn test_filter_on_empty_pq_code() {
        let pre_filter = TestPreFilter::new(vec![1, 3, 5, 7, 9]);
        let code = Arc::new(UInt8Array::from(Vec::<u8>::new()));
        let row_ids = Arc::new(UInt64Array::from(Vec::<u64>::new()));

        let (code, row_ids) = PQIndex::filter_arrays(&pre_filter, code, row_ids, 16).unwrap();
        assert!(code.values().is_empty());
        assert!(row_ids.is_empty());
    }

    #[tokio::test]
    async fn test_pq_search_runs_on_cpu_thread_in_sequential_mode() {
        let codebook_values = Float32Array::from_iter_values((0..256).map(|value| value as f32));
        let codebook = FixedSizeListArray::try_new_from_values(codebook_values, 1).unwrap();
        let index = PQIndex {
            pq: ProductQuantizer::new(1, 8, 1, codebook, DistanceType::L2),
            code: Some(Arc::new(UInt8Array::from(vec![0, 1, 0]))),
            row_ids: Some(Arc::new(UInt64Array::from(vec![10, 11, 12]))),
            metric_type: MetricType::L2,
            frag_reuse_index: None,
            code_bytes_per_vector: 1,
        };
        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0])) as ArrayRef,
            k: 2,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            rq_graph_scoring: Default::default(),
        };
        let is_empty_threads = Arc::new(Mutex::new(Vec::new()));
        let pre_filter = Arc::new(TestPreFilter::with_thread_capture(
            vec![],
            is_empty_threads.clone(),
        ));

        let batch = index
            .search(&query, pre_filter, &NoOpMetricsCollector)
            .await
            .unwrap();

        assert_eq!(batch.num_rows(), 2);
        let is_empty_threads = is_empty_threads.lock().unwrap();
        assert!(
            !is_empty_threads.is_empty(),
            "expected PQ search closure to evaluate the prefilter"
        );
        assert!(
            is_empty_threads
                .iter()
                .all(|name| name.contains("lance-cpu")),
            "expected PQ search closure to run on a lance-cpu thread, got {is_empty_threads:?}"
        );
    }
}
