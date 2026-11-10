// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WARP multivector search over a dataset's vector index segments.
//!
//! [`warp_select`] chooses each query token's partitions and missing similarity
//! from the IVF centroids alone; [`warp_score`] then scores the tokens of those
//! partitions, and [`top_k_documents`] reduces them to document scores.
//! [`WarpExec`] runs the three phases as one execution node.

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{
    Array, ArrayRef, Float32Array, RecordBatch, UInt64Array,
    cast::AsArray,
    types::{Float32Type, UInt64Type},
};
use arrow_schema::{DataType, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties};
use futures::{StreamExt, TryFutureExt, TryStreamExt, future::try_join_all, stream};
use lance_core::ROW_ID;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::prefilter::PreFilter;
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::warp::{
    Stride, centroid_similarities, missing_similarity, probe_partitions, top_k_documents,
};
use lance_index::vector::{DIST_COL, Query, VectorIndex};
use lance_linalg::distance::{DistanceType, dot_f32};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::{AnnIndexMetrics, KNN_INDEX_SCHEMA, build_dataset_prefilter, prefilter_for_segment};
use crate::dataset::Dataset;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::prefilter::DatasetPreFilter;
use crate::index::vector::ivf::IVFIndex;
use crate::io::exec::utils::{PreFilterMasks, PreFilterSource};
use crate::{Error, Result};

/// WARPselect output for one index segment.
#[derive(Debug)]
pub struct SegmentSelection {
    pub metadata: IndexMetadata,
    pub index: Arc<dyn VectorIndex>,
    /// `Γ[i][k] = <c_k, q_i>` for query token `i` and partition `k`.
    pub centroid_similarities: Vec<Vec<f32>>,
    /// `P_i`: partitions query token `i` probes in this segment, best first.
    pub probed_partitions: Vec<Vec<u32>>,
}

/// WARPselect output for one multivector query.
#[derive(Debug)]
pub struct WarpSelection {
    pub column: String,
    /// The unit query tokens `q_i`.
    pub query_tokens: Vec<Vec<f32>>,
    /// The vector column's float type, which partition search expects queries in.
    pub value_type: DataType,
    pub segments: Vec<SegmentSelection>,
    /// `m_i`: the similarity imputed for documents query token `i` never
    /// observes, estimated over the partitions of every segment.
    pub missing_similarities: Vec<f32>,
}

/// Per-query WARP search parameters.
#[derive(Debug, Clone, Copy)]
pub struct WarpParams {
    /// Partitions each query token probes per index segment (`n_probe`).
    pub nprobe: usize,
    /// The rank `t'` whose estimated similarity is imputed for unobserved
    /// documents.
    pub imputation_rank: usize,
}

/// Run WARPselect for a flattened multivector query against index segments.
///
/// Each segment is probed with its own `nprobe` partitions, as IVF search
/// treats delta segments. `m_i` instead walks the partitions of all segments
/// together, because it estimates the `imputation_rank`-th most similar token
/// in the whole indexed corpus.
///
/// Only centroids and partition sizes are read; they live in the IVF model
/// loaded (and cached) when a segment is opened, so no partition data is
/// fetched from storage.
pub async fn warp_select(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    query_tokens: &dyn Array,
    dim: usize,
    params: WarpParams,
    metrics: &dyn MetricsCollector,
) -> Result<WarpSelection> {
    let WarpParams {
        nprobe,
        imputation_rank,
    } = params;
    if nprobe == 0 {
        return Err(Error::invalid_input(
            "WARP nprobes must be at least 1, got 0",
        ));
    }
    if imputation_rank == 0 {
        return Err(Error::invalid_input(
            "WARP imputation_rank must be at least 1, got 0",
        ));
    }
    if dim == 0 || !query_tokens.len().is_multiple_of(dim) || query_tokens.is_empty() {
        return Err(Error::invalid_input(format!(
            "WARP query on column {column} has {} values, which is not a positive multiple \
             of the vector dimension {dim}",
            query_tokens.len()
        )));
    }

    if segments.is_empty() {
        return Err(Error::invalid_input(format!(
            "WARP search on column {column} needs at least one index segment"
        )));
    }

    let value_type = query_tokens.data_type().clone();
    // Multivector indices are cosine-only, so compare against unit query tokens.
    let query_tokens = cast(query_tokens, &DataType::Float32)?;
    let query_tokens = query_tokens
        .as_primitive::<Float32Type>()
        .values()
        .chunks_exact(dim)
        .map(|token| {
            let norm = dot_f32(token, token).sqrt();
            if norm > 0.0 {
                token.iter().map(|value| value / norm).collect()
            } else {
                token.to_vec()
            }
        })
        .collect::<Vec<Vec<f32>>>();

    let indices = try_join_all(
        segments
            .iter()
            .map(|segment| dataset.open_vector_index(column, &segment.uuid, metrics)),
    )
    .await?;
    let column = column.to_string();
    let segments = segments.to_vec();

    spawn_cpu(move || {
        let segments = indices
            .into_iter()
            .zip(segments)
            .map(|(index, metadata)| {
                let ivf = index.ivf_model();
                let centroids = ivf.centroids_array().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "WARP search on column {column} needs IVF centroids, but index segment \
                         {} has none",
                        metadata.uuid
                    ))
                })?;
                let centroid_similarities = query_tokens
                    .iter()
                    .map(|token| centroid_similarities(token, centroids))
                    .collect::<Result<Vec<_>>>()?;
                let probed_partitions = centroid_similarities
                    .iter()
                    .map(|similarities| probe_partitions(similarities, nprobe))
                    .collect();
                Ok(SegmentSelection {
                    metadata,
                    index,
                    centroid_similarities,
                    probed_partitions,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // The centroid model's `lengths` are not populated for current index
        // files; the partition storage holds the real sizes.
        let partition_sizes = segments
            .iter()
            .flat_map(|segment| {
                (0..segment.index.total_partitions())
                    .map(|partition| segment.index.partition_size(partition) as u32)
            })
            .collect::<Vec<_>>();
        let missing_similarities = (0..query_tokens.len())
            .map(|token| {
                let similarities = segments
                    .iter()
                    .flat_map(|segment| segment.centroid_similarities[token].iter().copied())
                    .collect::<Vec<_>>();
                missing_similarity(&similarities, &partition_sizes, imputation_rank)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(WarpSelection {
            column,
            query_tokens,
            value_type,
            segments,
            missing_similarities,
        })
    })
    .await
}

/// Score every token in each query token's probed partitions: `<t̂, q_i>` for
/// each `t` in a partition of `P_i`, where `t̂` is the token as the index
/// stores it.
///
/// Returns, per query token `i`, one [`Stride`] per probed (segment, partition)
/// pair, in no particular order.
///
/// Partitions are searched partition by partition, all query tokens probing
/// one partition back to back: concurrent loads of a partition are coalesced
/// and later ones hit the index cache, so each probed partition is read from
/// storage once per query rather than once per query token.
///
/// Scoring runs through each segment's own partition search, which already
/// knows its quantizer: FLAT compares raw vectors, while PQ, SQ and RQ score
/// their codes (against `q_i - c_k` for the residual quantizers PQ and RQ). So
/// WARP needs no per-quantizer kernel. Partition search returns at most `k`
/// rows, so `k` is set to the partition size to keep every token.
///
/// `pre_filter` removes deleted and filtered rows; rows a segment no longer
/// owns are dropped per segment, as in ANN search.
pub async fn warp_score(
    dataset: &Arc<Dataset>,
    selection: &WarpSelection,
    pre_filter: Arc<DatasetPreFilter>,
    metrics: &dyn MetricsCollector,
) -> Result<Vec<Vec<Stride>>> {
    let segment_filters = try_join_all(selection.segments.iter().map(|segment| {
        prefilter_for_segment(dataset.clone(), &segment.metadata, pre_filter.clone())
    }))
    .await?;
    let conversions = selection
        .segments
        .iter()
        .map(|segment| cosine_similarity_conversion(segment.index.as_ref(), segment.metadata.uuid))
        .collect::<Result<Vec<_>>>()?;

    let mut probes = selection
        .segments
        .iter()
        .enumerate()
        .flat_map(|(segment_id, segment)| {
            segment
                .probed_partitions
                .iter()
                .enumerate()
                .flat_map(move |(token, partitions)| {
                    partitions
                        .iter()
                        .map(move |&partition| (token, segment_id, partition as usize))
                })
        })
        .collect::<Vec<_>>();
    probes.sort_unstable_by_key(|&(token, segment_id, partition)| (segment_id, partition, token));

    let scored = stream::iter(probes)
        .map(|(token, segment_id, partition)| {
            let segment = &selection.segments[segment_id];
            let pre_filter = segment_filters[segment_id].clone();
            let to_similarity = conversions[segment_id];
            async move {
                let stride = score_stride(
                    selection,
                    segment,
                    token,
                    partition,
                    pre_filter,
                    to_similarity,
                    metrics,
                )
                .await?;
                Result::Ok((token, stride))
            }
        })
        .buffer_unordered(get_num_compute_intensive_cpus())
        .try_collect::<Vec<_>>()
        .await?;

    let mut strides = vec![Vec::new(); selection.query_tokens.len()];
    for (token, stride) in scored {
        strides[token].push(stride);
    }
    Ok(strides)
}

/// Map a segment's partition search distance to cosine similarity, or reject
/// a segment WARP cannot score.
///
/// HNSW sub-indices are rejected because their graph search only reaches the
/// tokens connected to its starting node, so even with `k` and `ef` at the
/// partition size it can silently skip tokens of a probed partition.
///
/// Multivector indices are cosine, but only FLAT reports cosine distance
/// `1 - <t, q>`. PQ, SQ and RQ store normalized vectors and report squared L2
/// distance `|q - t̂|^2 = 1 + |t̂|^2 - 2<t̂, q>`, which is `2 - 2<t̂, q>` when the
/// reconstruction keeps unit norm.
fn cosine_similarity_conversion(index: &dyn VectorIndex, uuid: Uuid) -> Result<fn(f32) -> f32> {
    if index.as_any().is::<IVFIndex>() {
        return Err(Error::not_supported(format!(
            "WARP search does not support the legacy IVF_PQ index format of segment {uuid}; \
             rebuild the index or use MultivectorAlgorithm::Xtr"
        )));
    }
    if index.metric_type() != DistanceType::Cosine {
        return Err(Error::invalid_input(format!(
            "WARP search needs a cosine index, but segment {uuid} uses {}",
            index.metric_type()
        )));
    }
    let (sub_index, quantization) = index.sub_index_type();
    if !matches!(sub_index, SubIndexType::Flat) {
        return Err(Error::not_supported(format!(
            "WARP search does not support the {sub_index} sub-index of segment {uuid}, \
             which may skip tokens of a partition; use MultivectorAlgorithm::Xtr"
        )));
    }
    match quantization {
        QuantizationType::Flat => Ok(|distance| 1.0 - distance),
        QuantizationType::Product | QuantizationType::Scalar | QuantizationType::Rabit => {
            Ok(|distance| 1.0 - distance / 2.0)
        }
        QuantizationType::FlatBin => Err(Error::not_supported(format!(
            "WARP search does not support the binary FLAT quantization of segment {uuid}"
        ))),
    }
}

/// Whether WARP can search every one of `segments`, i.e. whether
/// [`warp_score`] would not reject any of them as unsupported.
pub async fn warp_supports_segments(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
) -> Result<bool> {
    for segment in segments {
        let index = dataset
            .open_vector_index(column, &segment.uuid, &NoOpMetricsCollector)
            .await?;
        match cosine_similarity_conversion(index.as_ref(), segment.uuid) {
            Ok(_) => {}
            Err(Error::NotSupported { .. }) => return Ok(false),
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

async fn score_stride(
    selection: &WarpSelection,
    segment: &SegmentSelection,
    token: usize,
    partition: usize,
    pre_filter: Arc<dyn PreFilter>,
    to_similarity: fn(f32) -> f32,
    metrics: &dyn MetricsCollector,
) -> Result<Stride> {
    let index = segment.index.as_ref();
    let partition_size = index.partition_size(partition);
    if partition_size == 0 {
        return Ok(Stride::default());
    }
    let query_token = &selection.query_tokens[token];
    let centroid = index.ivf_model().centroid(partition).ok_or_else(|| {
        Error::index(format!(
            "partition {partition} of segment {} has no centroid",
            segment.metadata.uuid
        ))
    })?;
    let centroid = cast(&centroid, &DataType::Float32)?;
    let centroid = centroid.as_primitive::<Float32Type>().values();
    // RQ needs `|q_i - c_k|^2`, the distance IVF probing would have reported.
    let dist_q_c =
        1.0 + dot_f32(centroid, centroid) - 2.0 * segment.centroid_similarities[token][partition];
    let key: ArrayRef = Arc::new(Float32Array::from(query_token.clone()));
    let query = Query {
        column: selection.column.clone(),
        key: cast(&key, &selection.value_type)?,
        k: partition_size,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 1,
        maximum_nprobes: None,
        search_effort: 0.5,
        ef: None,
        refine_factor: None,
        metric_type: Some(DistanceType::Cosine),
        use_index: true,
        query_parallelism: 0,
        dist_q_c,
        approx_mode: Default::default(),
    };
    let batch = index
        .search_in_partition(partition, &query, pre_filter, metrics)
        .await?;

    let column = |name: &str| {
        batch.column_by_name(name).ok_or_else(|| {
            Error::internal(format!(
                "partition search of segment {} returned no {name} column",
                segment.metadata.uuid
            ))
        })
    };
    let row_ids = column(ROW_ID)?.as_primitive::<UInt64Type>().values();
    let distances = column(DIST_COL)?.as_primitive::<Float32Type>().values();
    let mut scored = row_ids
        .iter()
        .zip(distances.iter())
        .map(|(&row_id, &distance)| (row_id, to_similarity(distance)))
        .collect::<Vec<_>>();
    scored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    scored.dedup_by_key(|(row_id, _)| *row_id);
    let (row_ids, similarities) = scored.into_iter().unzip();
    Ok(Stride {
        row_ids,
        similarities,
    })
}

/// Executes WARP multivector search: WARPselect ([`warp_select`]), scoring of
/// the probed partitions ([`warp_score`]) and the reduction to the top `k`
/// ([`top_k_documents`]).
///
/// Emits one batch of [`KNN_INDEX_SCHEMA`], `k * refine_factor` rows sorted by
/// ascending distance, where a document's distance is the number of query
/// tokens minus its WARP score, as XTR reports it.
#[derive(Debug)]
pub struct WarpExec {
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    /// The multivector query; `key` holds the query tokens concatenated.
    query: Query,
    dim: usize,
    params: WarpParams,
    prefilter_source: PreFilterSource,
    /// Rows whose index entries are stale due to a newer data overlay.
    overlay_block: Option<RowAddrMask>,
    /// External row-address mask ANDed into the prefilter.
    external_mask: Option<Arc<RowAddrMask>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl WarpExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        query: Query,
        dim: usize,
        params: WarpParams,
        prefilter_source: PreFilterSource,
        overlay_block: Option<RowAddrMask>,
        external_mask: Option<Arc<RowAddrMask>>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            indices,
            query,
            dim,
            params,
            prefilter_source,
            overlay_block,
            external_mask,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for WarpExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let WarpParams {
            nprobe,
            imputation_rank,
        } = self.params;
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "Warp: k={}, nprobes={nprobe}, imputation_rank={imputation_rank}, deltas={}",
                self.query.k,
                self.indices.len()
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "Warp\nk={}\nnprobes={nprobe}\nimputation_rank={imputation_rank}\ndeltas={}",
                self.query.k,
                self.indices.len()
            ),
        }
    }
}

impl ExecutionPlan for WarpExec {
    fn name(&self) -> &str {
        "WarpExec"
    }

    fn schema(&self) -> SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) | PreFilterSource::ScalarIndexQuery(src) => {
                vec![src]
            }
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != self.children().len() {
            return Err(DataFusionError::Internal(format!(
                "WarpExec has {} children but got {} replacements",
                self.children().len(),
                children.len()
            )));
        }
        let prefilter_source = match (&self.prefilter_source, children.pop()) {
            (PreFilterSource::FilteredRowIds(_), Some(src)) => PreFilterSource::FilteredRowIds(src),
            (PreFilterSource::ScalarIndexQuery(_), Some(src)) => {
                PreFilterSource::ScalarIndexQuery(src)
            }
            (source, _) => source.clone(),
        };
        Ok(Arc::new(Self::new(
            self.dataset.clone(),
            self.indices.clone(),
            self.query.clone(),
            self.dim,
            self.params,
            prefilter_source,
            self.overlay_block.clone(),
            self.external_mask.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let pre_filter = build_dataset_prefilter(
            self.dataset.clone(),
            &self.indices,
            &self.prefilter_source,
            partition,
            context,
            PreFilterMasks {
                overlay_block: self.overlay_block.clone(),
                external_mask: self.external_mask.clone(),
            },
            &self.metrics,
        )?;
        let dataset = self.dataset.clone();
        let indices = self.indices.clone();
        let query = self.query.clone();
        let dim = self.dim;
        let params = self.params;
        let metrics = AnnIndexMetrics::new(&self.metrics, partition);

        let search = async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let selection = warp_select(
                &dataset,
                &query.column,
                &indices,
                query.key.as_ref(),
                dim,
                params,
                &metrics.index_metrics,
            )
            .await?;
            let strides =
                warp_score(&dataset, &selection, pre_filter, &metrics.index_metrics).await?;
            metrics
                .partitions_searched
                .add(strides.iter().map(Vec::len).sum());
            metrics.index_metrics.flush_io();

            let num_query_tokens = selection.missing_similarities.len() as f32;
            let missing_similarities = selection.missing_similarities;
            let k = query.k * query.refine_factor.unwrap_or(1) as usize;
            let top =
                spawn_cpu(move || top_k_documents(&strides, &missing_similarities, k)).await?;
            let (row_ids, distances): (Vec<_>, Vec<_>) = top
                .into_iter()
                .map(|(row_id, score)| (row_id, num_query_tokens - score))
                .unzip();
            let batch = RecordBatch::try_new(
                KNN_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(Float32Array::from(distances)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )?;
            metrics.baseline_metrics.record_output(batch.num_rows());
            metrics.baseline_metrics.done();
            Result::Ok(batch)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream::once(search.map_err(DataFusionError::from)),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::{HashMap, HashSet};

    use arrow_array::{
        FixedSizeListArray, Int32Array, ListArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::vector::IndexFileVersion;
    use crate::index::vector::VectorIndexParams;

    const DIM: usize = 8;
    const TOKENS_PER_ROW: usize = 4;

    fn multivector_batch(first_id: i32, num_rows: usize, rng: &mut StdRng) -> RecordBatch {
        let values = Float32Array::from_iter_values(
            (0..num_rows * TOKENS_PER_ROW * DIM).map(|_| rng.random_range(-1.0..1.0)),
        );
        let tokens = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();
        let field = Arc::new(Field::new("item", tokens.data_type().clone(), true));
        let rows = ListArray::new(
            field,
            OffsetBuffer::from_lengths(std::iter::repeat_n(TOKENS_PER_ROW, num_rows)),
            Arc::new(tokens),
            None,
        );
        let ids = Int32Array::from_iter_values(first_id..first_id + num_rows as i32);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("vector", rows.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(rows)]).unwrap()
    }

    async fn write(dataset: Option<&mut Dataset>, batch: RecordBatch) -> Option<Dataset> {
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let params = WriteParams {
            max_rows_per_file: 100,
            mode: WriteMode::Append,
            ..Default::default()
        };
        match dataset {
            Some(dataset) => {
                dataset.append(reader, Some(params)).await.unwrap();
                None
            }
            None => Some(
                Dataset::write(reader, "memory://", Some(params))
                    .await
                    .unwrap(),
            ),
        }
    }

    /// 500 rows over several fragments, indexed as two segments: the first 300
    /// rows by `params`, the other 200 by an appended delta segment.
    async fn two_segment_dataset(
        params: &VectorIndexParams,
        rng: &mut StdRng,
    ) -> (Dataset, Vec<IndexMetadata>) {
        let mut dataset = write(None, multivector_batch(0, 300, rng)).await.unwrap();
        dataset
            .create_index(&["vector"], IndexType::Vector, None, params, true)
            .await
            .unwrap();
        write(Some(&mut dataset), multivector_batch(300, 200, rng)).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert!(dataset.get_fragments().len() > 1);
        let segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(segments.len(), 2);
        (dataset, segments)
    }

    /// Partitions are probed per segment, but `m_i` walks every segment's
    /// partitions together with their sizes.
    #[tokio::test]
    async fn test_warp_select_across_segments() {
        let mut rng = StdRng::seed_from_u64(42);
        let (dataset, segments) = two_segment_dataset(
            &VectorIndexParams::ivf_flat(4, DistanceType::Cosine),
            &mut rng,
        )
        .await;

        let num_query_tokens = 3;
        let query = Float32Array::from_iter_values(
            (0..num_query_tokens * DIM).map(|_| rng.random_range(-1.0..1.0)),
        );
        let imputation_rank = 700;
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                nprobe: 2,
                imputation_rank,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        assert_eq!(selection.segments.len(), 2);
        let total_tokens = selection
            .segments
            .iter()
            .flat_map(|segment| {
                (0..segment.index.total_partitions())
                    .map(|partition| segment.index.partition_size(partition))
            })
            .sum::<usize>();
        assert_eq!(total_tokens, 500 * TOKENS_PER_ROW);

        for (token, query_token) in query.values().chunks_exact(DIM).enumerate() {
            let norm = dot_f32(query_token, query_token).sqrt();
            let unit = query_token.iter().map(|v| v / norm).collect::<Vec<_>>();

            let mut walk = Vec::new();
            for segment in &selection.segments {
                let ivf = segment.index.ivf_model();
                let expected =
                    centroid_similarities(&unit, ivf.centroids_array().unwrap()).unwrap();
                assert_eq!(segment.centroid_similarities[token], expected);
                assert_eq!(
                    segment.probed_partitions[token],
                    probe_partitions(&expected, 2)
                );
                walk.extend(
                    expected
                        .into_iter()
                        .enumerate()
                        .map(|(partition, similarity)| {
                            (similarity, segment.index.partition_size(partition))
                        }),
                );
            }
            // Expected m_i from a full sort of the combined partitions.
            walk.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut population = 0;
            let expected = walk
                .iter()
                .find(|(_, size)| {
                    population += *size;
                    population >= imputation_rank
                })
                .unwrap()
                .0;
            assert_eq!(selection.missing_similarities[token], expected);
        }
    }

    #[rstest]
    #[case::zero_nprobe(0, 10, 8, 16)]
    #[case::zero_imputation_rank(2, 0, 8, 16)]
    #[case::ragged_query(2, 10, 8, 15)]
    #[case::no_segments(2, 10, 8, 16)]
    #[tokio::test]
    async fn test_warp_select_rejects_invalid_input(
        #[case] nprobe: usize,
        #[case] imputation_rank: usize,
        #[case] dim: usize,
        #[case] query_len: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(7);
        let dataset = write(None, multivector_batch(0, 10, &mut rng))
            .await
            .unwrap();
        let query = Float32Array::from(vec![0.5; query_len]);
        let err = warp_select(
            &dataset,
            "vector",
            &[],
            &query,
            dim,
            WarpParams {
                nprobe,
                imputation_rank,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    fn ivf(num_partitions: usize) -> IvfBuildParams {
        IvfBuildParams::new(num_partitions)
    }

    /// Phase 2 scores every surviving token of every probed partition through
    /// each quantizer's own partition search, reported as cosine similarity.
    ///
    /// `max_mean_error` bounds how far a scored row is, on average, from the
    /// exact cosine similarity of its closest-matching token: exact for FLAT,
    /// quantization error otherwise.
    #[rstest]
    #[case::ivf_flat(VectorIndexParams::ivf_flat(4, DistanceType::Cosine), 1e-5)]
    #[case::ivf_pq(VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 50), 0.05)]
    #[case::ivf_sq(
        VectorIndexParams::with_ivf_sq_params(
            DistanceType::Cosine,
            ivf(4),
            SQBuildParams::default()
        ),
        0.01
    )]
    #[case::ivf_rq(VectorIndexParams::ivf_rq(4, 4, DistanceType::Cosine), 0.05)]
    #[tokio::test]
    async fn test_warp_score_quantizations(
        #[case] params: VectorIndexParams,
        #[case] max_mean_error: f32,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let (mut dataset, segments) = two_segment_dataset(&params, &mut rng).await;
        dataset.delete("id % 10 = 0").await.unwrap();
        let dataset = Arc::new(dataset);

        let query =
            Float32Array::from_iter_values((0..2 * DIM).map(|_| rng.random_range(-1.0..1.0)));
        // Probing every partition must reach every live row.
        let nprobe = 4;
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                nprobe,
                imputation_rank: 100,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let pre_filter = Arc::new(DatasetPreFilter::new(dataset.clone(), &segments, None));
        let strides = warp_score(&dataset, &selection, pre_filter, &NoOpMetricsCollector)
            .await
            .unwrap();

        let live = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let tokens_by_row = live[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .zip(live["vector"].as_list::<i32>().iter())
            .map(|(&row_id, tokens)| {
                let tokens = tokens.unwrap();
                let values = tokens
                    .as_fixed_size_list()
                    .values()
                    .as_primitive::<Float32Type>();
                (
                    row_id,
                    values
                        .values()
                        .chunks_exact(DIM)
                        .map(<[f32]>::to_vec)
                        .collect(),
                )
            })
            .collect::<HashMap<u64, Vec<Vec<f32>>>>();
        assert_eq!(tokens_by_row.len(), 450);

        assert_eq!(strides.len(), 2);
        for (query_token, token_strides) in selection.query_tokens.iter().zip(&strides) {
            assert_eq!(token_strides.len(), segments.len() * nprobe);
            let mut best_by_row = HashMap::<u64, f32>::new();
            let mut total_error = 0.0;
            for stride in token_strides {
                assert!(stride.row_ids.is_sorted_by(|a, b| a < b));
                assert_eq!(stride.row_ids.len(), stride.similarities.len());
                for (row_id, &similarity) in stride.row_ids.iter().zip(&stride.similarities) {
                    let best = best_by_row.entry(*row_id).or_insert(f32::MIN);
                    *best = best.max(similarity);
                    let tokens = &tokens_by_row[row_id];
                    total_error += tokens
                        .iter()
                        .map(|token| {
                            let cosine = dot_f32(token, query_token) / dot_f32(token, token).sqrt();
                            (similarity - cosine).abs()
                        })
                        .fold(f32::MAX, f32::min);
                }
            }
            let observed = best_by_row.keys().copied().collect::<HashSet<_>>();
            let expected = tokens_by_row.keys().copied().collect::<HashSet<_>>();
            assert_eq!(observed, expected);
            if max_mean_error <= 1e-5 {
                // Exact scores: each stride keeps a row's best token, so the
                // best over all strides is the row's best token overall.
                for (row_id, best) in &best_by_row {
                    let exact = tokens_by_row[row_id]
                        .iter()
                        .map(|token| dot_f32(token, query_token) / dot_f32(token, token).sqrt())
                        .fold(f32::MIN, f32::max);
                    assert!(
                        (best - exact).abs() <= 1e-5,
                        "row {row_id}: {best} vs {exact}"
                    );
                }
            }
            let num_scored = token_strides.iter().map(|s| s.row_ids.len()).sum::<usize>();
            let mean_error = total_error / num_scored as f32;
            assert!(mean_error <= max_mean_error, "mean error {mean_error}");
        }
    }

    fn legacy_ivf_pq() -> VectorIndexParams {
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 50);
        params.version(IndexFileVersion::Legacy);
        params
    }

    #[rstest]
    #[case::legacy_ivf_pq(legacy_ivf_pq(), "legacy IVF_PQ")]
    #[case::ivf_hnsw_flat(
        VectorIndexParams::ivf_hnsw(DistanceType::Cosine, ivf(4), HnswBuildParams::default()),
        "HNSW sub-index"
    )]
    #[case::ivf_hnsw_sq(
        VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::Cosine,
            ivf(4),
            HnswBuildParams::default(),
            SQBuildParams::default()
        ),
        "HNSW sub-index"
    )]
    #[tokio::test]
    async fn test_warp_score_rejects_unsupported_indices(
        #[case] params: VectorIndexParams,
        #[case] message: &str,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut dataset = write(None, multivector_batch(0, 300, &mut rng))
            .await
            .unwrap();
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
        let dataset = Arc::new(dataset);
        let query = Float32Array::from(vec![0.5; DIM]);
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                nprobe: 1,
                imputation_rank: 1,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let pre_filter = Arc::new(DatasetPreFilter::new(dataset.clone(), &segments, None));
        let err = warp_score(&dataset, &selection, pre_filter, &NoOpMetricsCollector)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
        assert!(err.to_string().contains(message), "{err}");
    }
}
