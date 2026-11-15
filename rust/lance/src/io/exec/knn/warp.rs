// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WARP multivector search over a dataset's vector index segments.
//!
//! [`warp_select`] chooses each query token's partitions and missing similarity
//! from the IVF centroids alone; [`warp_score`] then scores the tokens of those
//! partitions, and [`warp_top_k`] reduces them to document scores.
//! [`WarpExec`] runs the three phases as one execution node.

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array, cast::AsArray,
    types::Float32Type,
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
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt, TryFutureExt, future::try_join_all, stream};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_index::metrics::{LocalMetricsCollector, MetricsCollector, NoOpMetricsCollector};
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::warp::{
    Stride, centroid_similarities, default_imputation_rank, merge_strides, merge_top_k,
    missing_similarity, probe_partitions, split_row_ids, top_k_documents,
};
use lance_index::vector::{ApproxMode, PreparedPartitionRows, Query, VectorIndex};
use lance_linalg::distance::{DistanceType, dot_f32};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::adaptive_probe::interpolate_budget;
use super::{AnnIndexMetrics, KNN_INDEX_SCHEMA, prefilter_for_segment};
use crate::dataset::Dataset;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::prefilter::DatasetPreFilter;
use crate::index::vector::ivf::IVFIndex;
use crate::index::vector::ivf::v2::GLOBAL_TOPK_CHUNK_BYTES;
use crate::io::exec::utils::{PreFilterMasks, PreFilterSource, build_prefilter};
use crate::{Error, Result};

/// WARPselect output for one index segment.
#[derive(Debug)]
pub struct SegmentSelection {
    pub metadata: IndexMetadata,
    pub index: Arc<dyn VectorIndex>,
    /// `Γ[i][k] = <c_k, q_i>` for query token `i` and partition `k`.
    pub centroid_similarities: Vec<Vec<f32>>,
    /// `|c_k|^2` for partition `k`.
    pub centroid_norms: Arc<[f32]>,
    /// `P_i`: partitions query token `i` probes in this segment, best first.
    pub probed_partitions: Vec<Vec<u32>>,
}

/// WARPselect output for one multivector query.
#[derive(Debug)]
pub struct WarpSelection {
    pub column: String,
    /// The unit query tokens `q_i`.
    pub query_tokens: Vec<Vec<f32>>,
    /// The unit query tokens as partition search expects them, in the vector
    /// column's float type.
    pub query_keys: Vec<ArrayRef>,
    pub segments: Vec<SegmentSelection>,
    /// `m_i`: the similarity imputed for documents query token `i` never
    /// observes, estimated over the partitions of every segment.
    pub missing_similarities: Vec<f32>,
    /// How the probed partitions' tokens are scored.
    pub approx_mode: ApproxMode,
}

/// Per-query WARP search parameters.
#[derive(Debug, Clone, Copy)]
pub struct WarpParams {
    /// Partitions each query token probes per index segment (`n_probe`) at
    /// the default search effort, before the probe bounds apply.
    pub nprobe: usize,
    /// The fewest partitions each query token probes per segment, as in
    /// [`Query::minimum_nprobes`].
    pub minimum_nprobes: usize,
    /// The most partitions each query token probes per segment, as in
    /// [`Query::maximum_nprobes`]; unset allows all partitions.
    pub maximum_nprobes: Option<usize>,
    /// The search effort in [0, 1], as in [`Query::search_effort`].
    pub search_effort: f64,
    /// The rank `t'` whose estimated similarity is imputed for unobserved
    /// documents; unset uses [`default_imputation_rank`] over the searched
    /// segments.
    pub imputation_rank: Option<usize>,
    /// The number of documents requested, which sets the default `t'` of
    /// indices with many partitions.
    pub k: usize,
    /// How the partitions' quantized tokens are scored, as in
    /// [`Query::approx_mode`].
    pub approx_mode: ApproxMode,
}

impl WarpParams {
    /// Probe `nprobe` partitions per query token and segment at the default
    /// effort, with no other bounds, and impute the `imputation_rank`-th
    /// similarity.
    #[cfg(test)]
    pub fn new(nprobe: usize, imputation_rank: usize) -> Self {
        Self {
            nprobe,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            imputation_rank: Some(imputation_rank),
            k: 10,
            approx_mode: ApproxMode::default(),
        }
    }

    /// The partitions each query token probes in a segment of
    /// `num_partitions`.
    ///
    /// As in IVF search, effort 0.5 keeps the default budget (`nprobe`), 0
    /// probes the minimum and 1 every partition up to the maximum, with
    /// geometric interpolation in between. The minimum and maximum always
    /// bound the result.
    fn segment_nprobe(&self, num_partitions: usize) -> usize {
        let upper = self
            .maximum_nprobes
            .map_or(num_partitions, |maximum| maximum.min(num_partitions));
        if upper == 0 {
            return 0;
        }
        let lower = self.minimum_nprobes.clamp(1, upper);
        let default = self.nprobe.clamp(lower, upper);
        let effort = self.search_effort;
        if effort == 0.5 {
            default
        } else if effort < 0.5 {
            interpolate_budget(effort, (0.0, lower), (0.5, default))
        } else {
            interpolate_budget(effort, (0.5, default), (1.0, upper))
        }
    }
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
        search_effort,
        approx_mode,
        ..
    } = params;
    if nprobe == 0 || params.maximum_nprobes == Some(0) {
        return Err(Error::invalid_input(
            "WARP nprobes must be at least 1, got 0",
        ));
    }
    if !search_effort.is_finite() || !(0.0..=1.0).contains(&search_effort) {
        return Err(Error::invalid_input(format!(
            "search_effort must be finite and in [0, 1], got {search_effort}"
        )));
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
    let query_keys = query_tokens
        .iter()
        .map(|token| {
            let key: ArrayRef = Arc::new(Float32Array::from(token.clone()));
            Ok(cast(&key, &value_type)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let indices = try_join_all(
        segments
            .iter()
            .map(|segment| dataset.open_vector_index(column, &segment.uuid, metrics)),
    )
    .await?;
    // Each segment's centroids, cast to f32 once, and their squared norms.
    let centroids = try_join_all(indices.iter().zip(segments).map(|(index, metadata)| {
        let centroids = index.ivf_model().centroids_array().cloned();
        let uuid = metadata.uuid;
        let column = column.to_string();
        spawn_cpu(move || {
            let centroids = centroids.ok_or_else(|| {
                Error::invalid_input(format!(
                    "WARP search on column {column} needs IVF centroids, but index segment \
                     {uuid} has none"
                ))
            })?;
            let dim = centroids.value_length();
            let values = cast(centroids.values().as_ref(), &DataType::Float32)?;
            let values = values.as_primitive::<Float32Type>();
            let start = centroids.offset() * dim as usize;
            let values = values.slice(start, centroids.len() * dim as usize);
            let norms = values
                .values()
                .chunks_exact(dim as usize)
                .map(|centroid| dot_f32(centroid, centroid))
                .collect::<Arc<[f32]>>();
            let centroids = FixedSizeListArray::try_new_from_values(values, dim)?;
            Result::Ok((centroids, norms))
        })
    }))
    .await?;
    // The centroid model's `lengths` are not populated for current index
    // files; the partition storage holds the real sizes.
    let partition_sizes = indices
        .iter()
        .flat_map(|index| {
            (0..index.total_partitions()).map(|partition| index.partition_size(partition) as u32)
        })
        .collect::<Arc<[u32]>>();
    let imputation_rank = imputation_rank.unwrap_or_else(|| {
        let num_tokens = partition_sizes.iter().map(|&size| size as u64).sum();
        default_imputation_rank(num_tokens, partition_sizes.len(), params.k)
    });
    let nprobes = indices
        .iter()
        .map(|index| params.segment_nprobe(index.total_partitions()))
        .collect::<Arc<[usize]>>();

    // Each query token is selected independently, so tokens run in parallel.
    let shared_centroids = Arc::new(
        centroids
            .iter()
            .map(|(centroids, _)| centroids.clone())
            .collect::<Vec<_>>(),
    );
    let per_token = try_join_all(query_tokens.iter().cloned().map(|token| {
        let centroids = shared_centroids.clone();
        let partition_sizes = partition_sizes.clone();
        let nprobes = nprobes.clone();
        spawn_cpu(move || {
            let similarities = centroids
                .iter()
                .map(|centroids| centroid_similarities(&token, centroids))
                .collect::<Result<Vec<_>>>()?;
            let probed = similarities
                .iter()
                .zip(nprobes.iter())
                .map(|(similarities, &nprobe)| probe_partitions(similarities, nprobe))
                .collect::<Vec<_>>();
            let missing =
                missing_similarity(&similarities.concat(), &partition_sizes, imputation_rank)?;
            Result::Ok((similarities, probed, missing))
        })
    }))
    .await?;

    let mut segments = indices
        .into_iter()
        .zip(segments.iter().cloned())
        .zip(centroids)
        .map(
            |((index, metadata), (_, centroid_norms))| SegmentSelection {
                metadata,
                index,
                centroid_similarities: Vec::with_capacity(query_tokens.len()),
                centroid_norms,
                probed_partitions: Vec::with_capacity(query_tokens.len()),
            },
        )
        .collect::<Vec<_>>();
    let mut missing_similarities = Vec::with_capacity(query_tokens.len());
    for (similarities, probed, missing) in per_token {
        for ((segment, similarities), probed) in segments.iter_mut().zip(similarities).zip(probed) {
            segment.centroid_similarities.push(similarities);
            segment.probed_partitions.push(probed);
        }
        missing_similarities.push(missing);
    }

    Ok(WarpSelection {
        column: column.to_string(),
        query_tokens,
        query_keys,
        segments,
        missing_similarities,
        approx_mode,
    })
}

/// Score every token in each query token's probed partitions: `<t̂, q_i>` for
/// each `t` in a partition of `P_i`, where `t̂` is the token as the index
/// stores it.
///
/// Returns, per query token `i`, one [`Stride`] per probed (segment, partition)
/// pair, in no particular order.
///
/// Each probed partition is loaded and scanned once, for every query token
/// that probes it, through [`VectorIndex::score_prepared_partition_rows`], which scores
/// the partition with its own quantizer: FLAT compares raw vectors, while PQ,
/// SQ and RQ score their codes (against `q_i - c_k` for the residual
/// quantizers PQ and RQ). So WARP needs no per-quantizer kernel.
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
                        .map(move |&partition| (segment_id, partition as usize, token))
                })
        })
        .collect::<Vec<_>>();
    probes.sort_unstable();
    let mut partitions = Vec::<(usize, usize, Vec<usize>)>::new();
    for (segment_id, partition, token) in probes {
        match partitions.last_mut() {
            Some((last_segment, last_partition, tokens))
                if (*last_segment, *last_partition) == (segment_id, partition) =>
            {
                tokens.push(token);
            }
            _ => partitions.push((segment_id, partition, vec![token])),
        }
    }

    // Partitions are prepared (loaded, pre-filter awaited) concurrently, then
    // scored in batches, one CPU task per batch: one task per partition would
    // contend on the CPU pool's queue. A prepared partition pins its storage
    // until scored, so batches are capped in partitions and bytes, and the
    // in-flight batches share the byte budget of one global top-k chunk:
    // what a query pins does not grow with the partitions it probes.
    let num_cpus = get_num_compute_intensive_cpus();
    let partitions_per_task = partitions
        .len()
        .div_ceil(2 * num_cpus)
        .clamp(1, MAX_PARTITIONS_PER_TASK);
    let bytes_per_task = (*GLOBAL_TOPK_CHUNK_BYTES / num_cpus).max(1);
    let prepared = stream::iter(partitions)
        .map(|(segment_id, partition, tokens)| {
            let segment = &selection.segments[segment_id];
            let pre_filter = segment_filters[segment_id].clone();
            let to_similarity = conversions[segment_id];
            async move {
                let prepared = if segment.index.partition_size(partition) == 0 {
                    None
                } else {
                    let queries = tokens
                        .iter()
                        .map(|&token| partition_query(selection, segment, token, partition))
                        .collect::<Vec<_>>();
                    Some(
                        segment
                            .index
                            .prepare_partition_rows(partition, &queries, pre_filter, metrics)
                            .await?,
                    )
                };
                Result::Ok(PreparedPartition {
                    index: segment.index.clone(),
                    to_similarity,
                    tokens,
                    prepared,
                })
            }
        })
        .buffer_unordered(num_cpus)
        // Boxing erases the closure types, which otherwise make the compiler
        // demand the stream be generic over the borrowed selection's lifetimes.
        .boxed();
    let scored = score_in_batches(
        prepared,
        BatchLimits {
            max_tasks: num_cpus,
            max_items: partitions_per_task,
            max_bytes: bytes_per_task,
        },
        PreparedPartition::size_bytes,
        score_batch,
    )
    .await?;

    let mut strides = vec![Vec::new(); selection.query_tokens.len()];
    for (task_strides, task_metrics) in scored {
        task_metrics.dump_into(metrics);
        for (token, stride) in task_strides {
            strides[token].push(stride);
        }
    }
    Ok(strides)
}

/// Most probed partitions one CPU task scores: enough to amortize dispatching
/// the task, few enough to keep the partitions in-flight tasks pin bounded.
const MAX_PARTITIONS_PER_TASK: usize = 16;

/// A probed partition ready to score on the CPU pool.
struct PreparedPartition {
    index: Arc<dyn VectorIndex>,
    to_similarity: fn(f32) -> f32,
    /// The query tokens probing the partition.
    tokens: Vec<usize>,
    /// The prepared scan; `None` for an empty partition.
    prepared: Option<PreparedPartitionRows>,
}

impl PreparedPartition {
    fn size_bytes(&self) -> usize {
        self.prepared
            .as_ref()
            .map_or(0, |prepared| prepared.size_bytes)
    }
}

/// How [`score_in_batches`] groups items into tasks.
#[derive(Debug, Clone, Copy)]
struct BatchLimits {
    /// Most batches being scored at once.
    max_tasks: usize,
    /// Most items in one batch.
    max_items: usize,
    /// A batch stops growing once its items reach this many bytes.
    max_bytes: usize,
}

/// Group the items of `prepared` into batches and `score` each, keeping at
/// most `limits.max_tasks` batches in flight. The items alive at once are
/// bounded by the batches in flight, the batch being assembled and the items
/// `prepared` itself buffers, however many items there are.
async fn score_in_batches<T, R, Fut>(
    mut prepared: impl Stream<Item = Result<T>> + Unpin,
    limits: BatchLimits,
    size_bytes: fn(&T) -> usize,
    score: fn(Vec<T>) -> Fut,
) -> Result<Vec<R>>
where
    Fut: Future<Output = Result<R>>,
{
    let mut tasks = FuturesUnordered::new();
    let mut scored = Vec::new();
    let mut is_prepared_exhausted = false;
    loop {
        while !is_prepared_exhausted && tasks.len() < limits.max_tasks.max(1) {
            match next_batch(&mut prepared, limits, size_bytes).await {
                Some(batch) => tasks.push(score(batch?)),
                None => is_prepared_exhausted = true,
            }
        }
        match tasks.next().await {
            Some(result) => scored.push(result?),
            None => return Ok(scored),
        }
    }
}

/// Take items until the batch reaches `limits.max_items` items or
/// `limits.max_bytes` bytes; `None` once `prepared` is exhausted.
async fn next_batch<T>(
    prepared: &mut (impl Stream<Item = Result<T>> + Unpin),
    limits: BatchLimits,
    size_bytes: fn(&T) -> usize,
) -> Option<Result<Vec<T>>> {
    let mut batch = Vec::new();
    let mut bytes = 0;
    while batch.len() < limits.max_items.max(1) && bytes < limits.max_bytes {
        match prepared.next().await {
            Some(Ok(item)) => {
                bytes += size_bytes(&item);
                batch.push(item);
            }
            Some(Err(err)) => return Some(Err(err)),
            None => break,
        }
    }
    (!batch.is_empty()).then_some(Ok(batch))
}

/// Score a batch of prepared partitions in one CPU task, as
/// `(query token, stride)` pairs.
fn score_batch(
    partitions: Vec<PreparedPartition>,
) -> impl Future<Output = Result<(Vec<(usize, Stride)>, LocalMetricsCollector)>> {
    spawn_cpu(move || {
        let task_metrics = LocalMetricsCollector::default();
        let mut strides = Vec::new();
        for PreparedPartition {
            index,
            to_similarity,
            tokens,
            prepared,
        } in partitions
        {
            let Some(prepared) = prepared else {
                strides.extend(tokens.into_iter().map(|token| (token, Stride::default())));
                continue;
            };
            let scores = index.score_prepared_partition_rows(prepared, &task_metrics)?;
            for (token, mut similarities) in tokens.into_iter().zip(scores.distances) {
                similarities
                    .iter_mut()
                    .for_each(|distance| *distance = to_similarity(*distance));
                let stride = Stride {
                    row_ids: scores.row_ids.clone(),
                    similarities,
                };
                strides.push((token, stride));
            }
        }
        Ok((strides, task_metrics))
    })
}

/// The query that scores query token `token` against `partition`.
fn partition_query(
    selection: &WarpSelection,
    segment: &SegmentSelection,
    token: usize,
    partition: usize,
) -> Query {
    // RQ needs `|q_i - c_k|^2`, the distance IVF probing would have reported.
    let dist_q_c = 1.0 + segment.centroid_norms[partition]
        - 2.0 * segment.centroid_similarities[token][partition];
    Query {
        column: selection.column.clone(),
        key: selection.query_keys[token].clone(),
        k: 0,
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
        approx_mode: selection.approx_mode,
    }
}

/// The bounds of a query's distance range: inclusive lower, exclusive upper.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistanceRange {
    pub lower: Option<f32>,
    pub upper: Option<f32>,
}

impl DistanceRange {
    fn contains(&self, distance: f32) -> bool {
        self.lower.is_none_or(|lower| distance >= lower)
            && self.upper.is_none_or(|upper| distance < upper)
    }
}

/// The `k` best documents of a WARP query from its strides, best first, as
/// `(row_id, WARP score)`.
///
/// Only documents whose distance (the number of query tokens minus the score,
/// as [`WarpExec`] reports it) lies in `distance_range` compete for the top k.
///
/// Each query token's strides are merged on the CPU pool in parallel; the sum
/// across tokens is then split by row id range, also in parallel.
pub async fn warp_top_k(
    strides: Vec<Vec<Stride>>,
    missing_similarities: Vec<f32>,
    distance_range: DistanceRange,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    // Below this many scored rows per range, splitting costs more than it saves.
    const MIN_ROWS_PER_RANGE: usize = 8192;
    let token_scores = try_join_all(
        strides
            .into_iter()
            .map(|token_strides| spawn_cpu(move || merge_strides(&token_strides))),
    )
    .await?;
    let num_rows = token_scores
        .iter()
        .map(|scores| scores.row_ids.len())
        .sum::<usize>();
    let max_ranges = (num_rows / MIN_ROWS_PER_RANGE).clamp(1, get_num_compute_intensive_cpus());
    let ranges = split_row_ids(&token_scores, max_ranges);
    let token_scores = Arc::new(token_scores);
    let missing_similarities = Arc::new(missing_similarities);
    let tops = try_join_all(ranges.into_iter().map(|rows| {
        let token_scores = token_scores.clone();
        let missing_similarities = missing_similarities.clone();
        spawn_cpu(move || {
            let num_query_tokens = missing_similarities.len() as f32;
            let is_eligible = |score| distance_range.contains(num_query_tokens - score);
            top_k_documents(&token_scores, &missing_similarities, rows, is_eligible, k)
        })
    }))
    .await?;
    Ok(merge_top_k(tops, k))
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
             rebuild the index"
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
             which may skip tokens of a partition"
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

/// Executes WARP multivector search: WARPselect ([`warp_select`]), scoring of
/// the probed partitions ([`warp_score`]) and the reduction to the top `k`
/// ([`warp_top_k`]).
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
    /// External row-address mask, combined with the prefilter by logical AND.
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
            minimum_nprobes,
            search_effort,
            imputation_rank,
            ..
        } = self.params;
        let imputation_rank =
            imputation_rank.map_or_else(|| "default".to_string(), |rank| rank.to_string());
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "Warp: k={}, nprobes={nprobe}, imputation_rank={imputation_rank}, deltas={}, \
                 minimum_nprobes={minimum_nprobes}, search_effort={search_effort}",
                self.query.k,
                self.indices.len()
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "Warp\nk={}\nnprobes={nprobe}\nimputation_rank={imputation_rank}\ndeltas={}\n\
                 minimum_nprobes={minimum_nprobes}\nsearch_effort={search_effort}",
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
        let pre_filter = build_prefilter(
            context,
            partition,
            &self.prefilter_source,
            self.dataset.clone(),
            &self.indices,
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
            let distance_range = DistanceRange {
                lower: query.lower_bound,
                upper: query.upper_bound,
            };
            let top = warp_top_k(strides, missing_similarities, distance_range, k).await?;
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

    use arrow_array::types::UInt64Type;
    use lance_core::ROW_ID;
    use lance_index::vector::DIST_COL;

    use std::collections::{HashMap, HashSet};

    use arrow_array::{
        FixedSizeListArray, Int32Array, ListArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{Field, Schema};
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
            WarpParams::new(2, imputation_rank),
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

        for (token, query_token) in query.values().as_chunks::<DIM>().0.iter().enumerate() {
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

        // The default t' scales with the tokens of every segment; this corpus
        // of 2,000 tokens is below its first step, so 0 is imputed.
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                imputation_rank: None,
                ..WarpParams::new(2, 1)
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        assert_eq!(selection.missing_similarities, vec![0.0; num_query_tokens]);
    }

    #[rstest]
    #[case::zero_nprobe(0, 8, 16)]
    #[case::ragged_query(2, 8, 15)]
    #[case::no_segments(2, 8, 16)]
    #[tokio::test]
    async fn test_warp_select_rejects_invalid_input(
        #[case] nprobe: usize,
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
            WarpParams::new(nprobe, 10),
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
            WarpParams::new(nprobe, 100),
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
                        .as_chunks::<DIM>()
                        .0
                        .iter()
                        .map(|token| token.to_vec())
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
            WarpParams::new(1, 1),
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

    /// Scoring as WARP did before partitions were scanned directly: a top-k
    /// partition search over every row of the partition per (token, partition)
    /// probe, sorted by row id and reduced to each row's best token.
    async fn reference_strides(
        dataset: &Arc<Dataset>,
        selection: &WarpSelection,
        pre_filter: Arc<DatasetPreFilter>,
    ) -> Vec<Vec<Stride>> {
        let mut strides = vec![Vec::new(); selection.query_tokens.len()];
        for segment in &selection.segments {
            let pre_filter =
                prefilter_for_segment(dataset.clone(), &segment.metadata, pre_filter.clone())
                    .await
                    .unwrap();
            let to_similarity =
                cosine_similarity_conversion(segment.index.as_ref(), segment.metadata.uuid)
                    .unwrap();
            for (token, partitions) in segment.probed_partitions.iter().enumerate() {
                for &partition in partitions {
                    let partition = partition as usize;
                    let partition_size = segment.index.partition_size(partition);
                    if partition_size == 0 {
                        strides[token].push(Stride::default());
                        continue;
                    }
                    let mut query = partition_query(selection, segment, token, partition);
                    query.k = partition_size;
                    let batch = segment
                        .index
                        .search_in_partition(
                            partition,
                            &query,
                            pre_filter.clone(),
                            &NoOpMetricsCollector,
                        )
                        .await
                        .unwrap();
                    let mut scored = batch[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .zip(batch[DIST_COL].as_primitive::<Float32Type>().values())
                        .map(|(&row_id, &distance)| (row_id, to_similarity(distance)))
                        .collect::<Vec<_>>();
                    scored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
                    scored.dedup_by_key(|(row_id, _)| *row_id);
                    let (row_ids, similarities): (Vec<_>, Vec<_>) = scored.into_iter().unzip();
                    strides[token].push(Stride {
                        row_ids: row_ids.into(),
                        similarities,
                    });
                }
            }
        }
        strides
    }

    /// Scores every observed document by the definition, adding query tokens
    /// in ascending order, and sorts them all.
    fn reference_top_k(strides: &[Vec<Stride>], missing: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut best = HashMap::<u64, Vec<Option<f32>>>::new();
        for (token, token_strides) in strides.iter().enumerate() {
            for stride in token_strides {
                for (&row_id, &similarity) in stride.row_ids.iter().zip(&stride.similarities) {
                    let slot = &mut best
                        .entry(row_id)
                        .or_insert_with(|| vec![None; strides.len()])[token];
                    *slot = Some(slot.map_or(similarity, |current| current.max(similarity)));
                }
            }
        }
        let mut scored = best
            .into_iter()
            .map(|(row_id, per_token)| {
                let score = per_token
                    .iter()
                    .zip(missing)
                    .map(|(best, &missing)| best.unwrap_or(missing))
                    .fold(0.0, |score, similarity| score + similarity);
                (row_id, score)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Scanning each probed partition once for all its query tokens, and the
    /// parallel reduction, return the same top k as scoring every probe with a
    /// partition search and reducing by the definition.
    ///
    /// The query's approximation mode reaches the scoring of RQ partitions.
    #[rstest]
    #[case::ivf_flat(
        VectorIndexParams::ivf_flat(4, DistanceType::Cosine),
        ApproxMode::Normal
    )]
    #[case::ivf_pq(
        VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 50),
        ApproxMode::Normal
    )]
    #[case::ivf_sq(
        VectorIndexParams::with_ivf_sq_params(
            DistanceType::Cosine,
            ivf(4),
            SQBuildParams::default()
        ),
        ApproxMode::Normal
    )]
    #[case::ivf_rq(
        VectorIndexParams::ivf_rq(4, 4, DistanceType::Cosine),
        ApproxMode::Normal
    )]
    #[case::ivf_rq_fast(
        VectorIndexParams::ivf_rq(4, 4, DistanceType::Cosine),
        ApproxMode::Fast
    )]
    #[case::ivf_rq_accurate(
        VectorIndexParams::ivf_rq(4, 4, DistanceType::Cosine),
        ApproxMode::Accurate
    )]
    #[tokio::test]
    async fn test_warp_matches_partition_search_reference(
        #[case] params: VectorIndexParams,
        #[case] approx_mode: ApproxMode,
    ) {
        let mut rng = StdRng::seed_from_u64(7);
        let (mut dataset, segments) = two_segment_dataset(&params, &mut rng).await;
        dataset.delete("id % 10 = 0").await.unwrap();
        let dataset = Arc::new(dataset);
        let query =
            Float32Array::from_iter_values((0..3 * DIM).map(|_| rng.random_range(-1.0..1.0)));
        // Probing two of four partitions leaves rows unobserved by some tokens.
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                approx_mode,
                ..WarpParams::new(2, 100)
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let pre_filter = Arc::new(DatasetPreFilter::new(dataset.clone(), &segments, None));
        let k = 50;

        let strides = warp_score(
            &dataset,
            &selection,
            pre_filter.clone(),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let reference = reference_strides(&dataset, &selection, pre_filter.clone()).await;
        for (token, (actual, expected)) in strides.iter().zip(&reference).enumerate() {
            assert_eq!(
                merge_strides(actual).unwrap(),
                merge_strides(expected).unwrap(),
                "token {token}"
            );
        }

        let expected = reference_top_k(&reference, &selection.missing_similarities, k);
        assert_eq!(expected.len(), k);
        let top = warp_top_k(
            strides.clone(),
            selection.missing_similarities.clone(),
            DistanceRange::default(),
            k,
        )
        .await
        .unwrap();
        assert_eq!(top, expected);

        assert_eq!(
            partition_query(&selection, &selection.segments[0], 0, 0).approx_mode,
            approx_mode
        );
        // The deletions route IVF_RQ through per-row scoring, where only Fast
        // changes distances; Accurate only affects unfiltered batch scoring.
        if approx_mode == ApproxMode::Fast {
            let mut normal = selection;
            normal.approx_mode = ApproxMode::Normal;
            let normal_strides = warp_score(&dataset, &normal, pre_filter, &NoOpMetricsCollector)
                .await
                .unwrap();
            // Strides arrive in completion order, so compare them merged.
            let merged = |strides: &[Vec<Stride>]| {
                strides
                    .iter()
                    .map(|token_strides| merge_strides(token_strides).unwrap())
                    .collect::<Vec<_>>()
            };
            assert_ne!(merged(&strides), merged(&normal_strides));
        }
    }

    #[rstest]
    #[case::default(32, 1, None, 0.5, 100, 32)]
    #[case::minimum_raises(32, 64, None, 0.5, 100, 64)]
    #[case::minimum_capped_by_partitions(32, 64, None, 0.5, 40, 40)]
    #[case::fixed(4, 4, Some(4), 0.5, 100, 4)]
    #[case::fewer_partitions(32, 1, None, 0.5, 10, 10)]
    #[case::no_partitions(32, 1, None, 0.5, 0, 0)]
    #[case::least_effort(32, 1, None, 0.0, 100, 1)]
    #[case::least_effort_keeps_minimum(32, 8, None, 0.0, 100, 8)]
    #[case::full_effort(32, 1, None, 1.0, 100, 100)]
    #[case::full_effort_keeps_maximum(50, 1, Some(50), 1.0, 100, 50)]
    // 32 * (100 / 32)^0.5 = 56.6, rounded up.
    #[case::interpolated_effort(32, 1, None, 0.75, 100, 57)]
    fn test_segment_nprobe(
        #[case] nprobe: usize,
        #[case] minimum_nprobes: usize,
        #[case] maximum_nprobes: Option<usize>,
        #[case] search_effort: f64,
        #[case] num_partitions: usize,
        #[case] expected: usize,
    ) {
        let params = WarpParams {
            minimum_nprobes,
            maximum_nprobes,
            search_effort,
            ..WarpParams::new(nprobe, 1)
        };
        assert_eq!(params.segment_nprobe(num_partitions), expected);
    }

    /// The scanner's probe controls reach WARP, which multivector search uses
    /// by default.
    #[rstest]
    #[case::default(None, None, None, 32)]
    #[case::nprobes(Some(4), None, None, 4)]
    #[case::minimum_nprobes(None, Some(36), None, 36)]
    #[case::full_effort(None, None, Some(1.0), 40)]
    #[case::least_effort(None, None, Some(0.0), 1)]
    #[tokio::test]
    async fn test_warp_probe_controls(
        #[case] nprobes: Option<usize>,
        #[case] minimum_nprobes: Option<usize>,
        #[case] search_effort: Option<f64>,
        #[case] expected: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut dataset = write(None, multivector_batch(0, 700, &mut rng))
            .await
            .unwrap();
        let num_partitions = 40;
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_flat(num_partitions, DistanceType::Cosine),
                true,
            )
            .await
            .unwrap();
        // One query token, so each probed partition is searched once.
        let query = Float32Array::from(vec![0.5; DIM]);
        let mut scanner = dataset.scan();
        scanner.nearest("vector", &query, 10).unwrap();
        if let Some(nprobes) = nprobes {
            scanner.nprobes(nprobes);
        }
        if let Some(minimum_nprobes) = minimum_nprobes {
            scanner.minimum_nprobes(minimum_nprobes);
        }
        if let Some(search_effort) = search_effort {
            scanner.search_effort(search_effort).unwrap();
        }
        let plan = scanner.analyze_plan().await.unwrap();
        let warp = plan
            .lines()
            .find(|line| line.contains("Warp:"))
            .unwrap_or_else(|| panic!("{plan}"));
        assert!(
            warp.contains(&format!("partitions_searched={expected},")),
            "{warp}"
        );
    }

    /// Counts the items alive, like the storage prepared partitions pin.
    struct Pinned {
        live: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Pinned {
        fn new(
            live: &Arc<std::sync::atomic::AtomicUsize>,
            peak: &Arc<std::sync::atomic::AtomicUsize>,
        ) -> Self {
            let now = live.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            peak.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            Self { live: live.clone() }
        }
    }

    impl Drop for Pinned {
        fn drop(&mut self) {
            self.live.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    async fn score_pinned(batch: Vec<Pinned>) -> Result<usize> {
        tokio::task::yield_now().await;
        Ok(batch.len())
    }

    /// However many partitions a query probes, scoring pins at most the
    /// prepared items buffered upstream, the batch being assembled and every
    /// in-flight batch; and every item is scored exactly once.
    #[rstest]
    #[case::by_items(BatchLimits { max_tasks: 3, max_items: 4, max_bytes: usize::MAX }, 1, 4)]
    #[case::by_bytes(BatchLimits { max_tasks: 2, max_items: 100, max_bytes: 10 }, 4, 3)]
    #[case::one_task(BatchLimits { max_tasks: 1, max_items: 5, max_bytes: usize::MAX }, 1, 5)]
    #[tokio::test]
    async fn test_score_in_batches_bounds_pinned_items(
        #[case] limits: BatchLimits,
        #[case] item_bytes: usize,
        #[case] max_batch: usize,
    ) {
        const NUM_ITEMS: usize = 1000;
        const PREPARE_WINDOW: usize = 2;
        let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prepared = stream::iter(0..NUM_ITEMS)
            .map(|_| {
                let (live, peak) = (live.clone(), peak.clone());
                async move { Result::Ok(Pinned::new(&live, &peak)) }
            })
            .buffer_unordered(PREPARE_WINDOW)
            .boxed();
        let size_bytes: fn(&Pinned) -> usize = match item_bytes {
            1 => |_| 1,
            _ => |_| 4,
        };
        let scored = score_in_batches(prepared, limits, size_bytes, score_pinned)
            .await
            .unwrap();

        assert_eq!(scored.iter().sum::<usize>(), NUM_ITEMS);
        assert!(scored.iter().all(|&len| (1..=max_batch).contains(&len)));
        let bound = PREPARE_WINDOW + (limits.max_tasks + 1) * max_batch;
        let peak = peak.load(std::sync::atomic::Ordering::SeqCst);
        assert!(peak <= bound, "peak {peak}, bound {bound}");
        assert_eq!(live.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// Partition scans count toward the index's prepared-partition tracker
    /// and are all released once scored.
    #[tokio::test]
    async fn test_warp_score_releases_prepared_partitions() {
        let mut rng = StdRng::seed_from_u64(3);
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
            ivf(4),
            lance_index::vector::pq::PQBuildParams::new(4, 8),
        );
        let (dataset, segments) = two_segment_dataset(&params, &mut rng).await;
        let dataset = Arc::new(dataset);
        let query = Float32Array::from_iter_values((0..DIM).map(|_| rng.random_range(-1.0..1.0)));
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams::new(4, 1),
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        let pre_filter = Arc::new(DatasetPreFilter::new(dataset.clone(), &segments, None));
        warp_score(&dataset, &selection, pre_filter, &NoOpMetricsCollector)
            .await
            .unwrap();
        for segment in &selection.segments {
            let tracker = segment
                .index
                .as_any()
                .downcast_ref::<crate::index::vector::ivf::v2::IvfPq>()
                .expect("IVF_PQ index")
                .prepared_partitions();
            assert!(tracker.peak() >= 1);
            assert_eq!(tracker.in_flight(), 0);
        }
    }

    /// A distance range bounds WARP's documents before top-k selection, with
    /// and without refinement, returning what exact search returns.
    #[rstest]
    #[case::upper(None, Some(0.000001), None)]
    #[case::lower(Some(0.8), Some(1.2), None)]
    #[case::lower_refined(Some(0.8), Some(1.2), Some(2))]
    #[tokio::test]
    async fn test_warp_distance_range(
        #[case] lower: Option<f32>,
        #[case] upper: Option<f32>,
        #[case] refine: Option<u32>,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let (dataset, _) = two_segment_dataset(
            &VectorIndexParams::ivf_flat(4, DistanceType::Cosine),
            &mut rng,
        )
        .await;
        let query = Float32Array::from(vec![0.5; DIM]);
        let expected = dataset
            .scan()
            .nearest("vector", &query, 10)
            .unwrap()
            .use_index(false)
            .distance_range(lower, upper)
            .try_into_batch()
            .await
            .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", &query, 10)
            .unwrap()
            .nprobes(4)
            .distance_range(lower, upper);
        if let Some(factor) = refine {
            scanner.refine(factor);
        }
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(plan.contains("Warp:"), "{plan}");
        let result = scanner.try_into_batch().await.unwrap();
        let distances = result[DIST_COL].as_primitive::<Float32Type>();
        assert_eq!(result.num_rows(), expected.num_rows());
        assert!(
            distances
                .values()
                .iter()
                .all(|&d| lower.is_none_or(|l| d >= l) && upper.is_none_or(|u| d < u)),
            "out-of-range distances: {:?}",
            distances.values()
        );
    }
}
