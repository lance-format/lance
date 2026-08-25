// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental cache execution for reusable IVF candidate pools.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{Float32Array, RecordBatch, UInt32Array, UInt64Array};
use futures::{StreamExt, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::{Error, ROW_ID, Result};
use lance_index::IndexType;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::{DIST_COL, PartitionSearchResult, Query, VectorIndex};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;

use crate::dataset::Dataset;
use crate::index::DatasetIndexExt;
use crate::session::index_caches::{
    CachedVectorCandidate, VectorResultsCacheEntry, VectorResultsCacheIdentity,
};

use super::knn::KNN_INDEX_SCHEMA;

/// Maximum reusable candidate pool retained for one exact query/search shape.
///
/// This is deliberately fixed while the feature is experimental. It is part of
/// the cache identity, so changing it cannot reuse entries created under another
/// limit.
pub(super) const RESULTS_CACHE_CANDIDATE_LIMIT: usize = 1000;

const VECTOR_RESULTS_CACHE_ENV: &str = "LANCE_EXPERIMENTAL_VECTOR_RESULTS_CACHE";

pub(super) struct ResultsCacheSearch {
    pub batch: RecordBatch,
    /// True for both a warm entry and a concurrent load coalesced behind its owner.
    pub was_hit: bool,
}

pub(super) struct ResultsCacheSearchParams<'a> {
    pub cache: &'a LanceCache,
    pub identity: VectorResultsCacheIdentity,
    pub index: Arc<dyn VectorIndex>,
    pub query: &'a Query,
    pub partitions: Arc<UInt32Array>,
    pub centroid_distances: Arc<Float32Array>,
    pub prefilter: Arc<dyn PreFilter>,
    pub segment_mask: Option<Arc<RowAddrMask>>,
    pub metrics: Arc<dyn MetricsCollector>,
    pub parallelism: usize,
}

pub(super) fn is_enabled() -> bool {
    std::env::var(VECTOR_RESULTS_CACHE_ENV)
        .ok()
        .is_some_and(|value| value == "1")
}

pub(super) async fn identity_for_query(
    dataset: &Dataset,
    index_metadata: &IndexMetadata,
    index: &dyn VectorIndex,
    query: &Query,
    partition_ids: &[u32],
    has_prefilter: bool,
    has_overlay: bool,
) -> Option<VectorResultsCacheIdentity> {
    let (sub_index_type, quantization_type) = match index.index_type() {
        IndexType::IvfRq => (SubIndexType::Flat, QuantizationType::Rabit),
        IndexType::IvfSq => (SubIndexType::Flat, QuantizationType::Scalar),
        _ => return None,
    };
    let index_store = dataset.object_store_for_index(index_metadata).await.ok()?;
    let fragment_reuse_uuid = dataset
        .load_indices()
        .await
        .ok()?
        .iter()
        .find(|metadata| metadata.name == FRAG_REUSE_INDEX_NAME)
        .map(|metadata| metadata.uuid);
    let manifest_location = dataset.manifest_location();
    let manifest_path = manifest_location.path.as_ref();
    let manifest_etag = manifest_location.e_tag.as_deref().unwrap_or_default();
    let manifest_size = manifest_location
        .size
        .map_or_else(|| "none".to_owned(), |size| format!("some:{size}"));
    let dataset_read_identity = format!(
        "{}:{}/{}:{}/{}:{}",
        manifest_path.len(),
        manifest_path,
        manifest_etag.len(),
        manifest_etag,
        manifest_size.len(),
        manifest_size
    );

    VectorResultsCacheIdentity::try_new(
        &index_store.store_prefix,
        &dataset_read_identity,
        dataset.version_id(),
        index_metadata,
        fragment_reuse_uuid,
        query,
        index.metric_type(),
        sub_index_type,
        quantization_type,
        partition_ids,
        RESULTS_CACHE_CANDIDATE_LIMIT,
        has_prefilter,
        has_overlay,
    )
    .ok()
}

pub(super) async fn search(params: ResultsCacheSearchParams<'_>) -> Result<ResultsCacheSearch> {
    let ResultsCacheSearchParams {
        cache,
        identity,
        index,
        query,
        partitions,
        centroid_distances,
        prefilter,
        segment_mask,
        metrics,
        parallelism,
    } = params;
    let mut populated_batch = None;
    let populated_batch_slot = &mut populated_batch;
    let loader_identity = identity.clone();
    let loader_index = index.clone();
    let loader_partitions = partitions.clone();
    let loader_centroid_distances = centroid_distances.clone();
    let loader_prefilter = prefilter.clone();
    let loader_segment_mask = segment_mask.clone();
    let loader_metrics = metrics.clone();
    let cache_lookup = cache
        .get_or_insert_with_key_hit(identity.clone(), move || async move {
            let (candidates, batch) = populate_candidates(
                loader_index,
                query,
                loader_partitions,
                loader_centroid_distances,
                loader_prefilter,
                loader_segment_mask,
                loader_metrics,
                parallelism,
                loader_identity.result_limit(),
            )
            .await?;
            *populated_batch_slot = Some(batch);
            VectorResultsCacheEntry::try_new(loader_identity, candidates)
        })
        .await;

    let (entry, was_cached) = match cache_lookup {
        Ok(result) => result,
        Err(error) => {
            if let Some(batch) = populated_batch {
                // The candidate search succeeded, so an invalid cache entry must not
                // discard its query result or cause the ordinary path to repeat work.
                log::debug!("not storing invalid vector results cache entry: {error}");
                return Ok(ResultsCacheSearch {
                    batch,
                    was_hit: false,
                });
            }
            return Err(error);
        }
    };

    if !was_cached {
        let batch = populated_batch.ok_or_else(|| {
            Error::internal(
                "vector results cache loader completed without its candidate-search batch",
            )
        })?;
        return Ok(ResultsCacheSearch {
            batch,
            was_hit: false,
        });
    }

    if entry.is_compatible_with(&identity) {
        match replay_candidates(
            index.clone(),
            query,
            entry.candidates(),
            identity.result_limit(),
            metrics.clone(),
            parallelism,
        )
        .await
        {
            Ok(batch) => {
                return Ok(ResultsCacheSearch {
                    batch,
                    was_hit: true,
                });
            }
            Err(error) => {
                // A stale or malformed in-memory entry must never fail the query.
                // Re-run the candidate-producing path and replace it.
                log::debug!("ignoring unusable vector results cache entry: {error}");
            }
        }
    }

    // An unusable existing entry cannot enter get-or-insert's loader, so replace it
    // directly. Normal cold misses are single-flighted by the lookup above.
    let (candidates, batch) = populate_candidates(
        index,
        query,
        partitions,
        centroid_distances,
        prefilter,
        segment_mask,
        metrics,
        parallelism,
        identity.result_limit(),
    )
    .await?;
    match VectorResultsCacheEntry::try_new(identity.clone(), candidates) {
        Ok(entry) => cache.insert_with_key(&identity, Arc::new(entry)).await,
        Err(error) => {
            // Cache population is best-effort and cannot change query success.
            log::debug!("not storing invalid vector results cache entry: {error}");
        }
    }
    Ok(ResultsCacheSearch {
        batch,
        was_hit: false,
    })
}

#[allow(clippy::too_many_arguments)]
async fn populate_candidates(
    index: Arc<dyn VectorIndex>,
    query: &Query,
    partitions: Arc<UInt32Array>,
    centroid_distances: Arc<Float32Array>,
    prefilter: Arc<dyn PreFilter>,
    segment_mask: Option<Arc<RowAddrMask>>,
    metrics: Arc<dyn MetricsCollector>,
    parallelism: usize,
    result_limit: usize,
) -> Result<(Vec<CachedVectorCandidate>, RecordBatch)> {
    if partitions.len() != centroid_distances.len() {
        return Err(Error::invalid_input(format!(
            "partition count {} does not match centroid distance count {} for vector results cache",
            partitions.len(),
            centroid_distances.len()
        )));
    }
    let accumulated = futures::stream::iter(0..partitions.len())
        .map(|partition_index| {
            let index = index.clone();
            let prefilter = prefilter.clone();
            let metrics = metrics.clone();
            let mut partition_query = query.clone();
            let partition_id = partitions.value(partition_index);
            partition_query.dist_q_c = centroid_distances.value(partition_index);
            async move {
                index
                    .search_in_partition_with_candidates(
                        partition_id as usize,
                        &partition_query,
                        prefilter,
                        metrics.as_ref(),
                        RESULTS_CACHE_CANDIDATE_LIMIT,
                    )
                    .await
            }
        })
        .buffered(parallelism.max(1))
        .try_fold(
            SearchAccumulator::new(RESULTS_CACHE_CANDIDATE_LIMIT, result_limit),
            move |mut accumulated, partition_result| {
                let segment_mask = segment_mask.clone();
                async move {
                    accumulated.add(partition_result, segment_mask.as_deref())?;
                    Ok(accumulated)
                }
            },
        )
        .await?;
    accumulated.finish()
}

struct SearchAccumulator {
    candidate_limit: usize,
    result_limit: usize,
    candidates: BinaryHeap<OrderedNode<CachedVectorCandidate>>,
    results: BinaryHeap<OrderedNode<u64>>,
}

impl SearchAccumulator {
    fn new(candidate_limit: usize, result_limit: usize) -> Self {
        Self {
            candidate_limit,
            result_limit,
            candidates: BinaryHeap::with_capacity(candidate_limit),
            results: BinaryHeap::with_capacity(result_limit),
        }
    }

    fn add(
        &mut self,
        search_result: PartitionSearchResult,
        segment_mask: Option<&RowAddrMask>,
    ) -> Result<()> {
        if search_result.candidates.len() != search_result.batch.num_rows() {
            return Err(Error::internal(format!(
                "partition search returned {} candidate identities for {} rows",
                search_result.candidates.len(),
                search_result.batch.num_rows()
            )));
        }
        let row_ids = search_result
            .batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("partition search result has no row-id column"))?
            .as_primitive::<UInt64Type>();
        let distances = search_result
            .batch
            .column_by_name(DIST_COL)
            .ok_or_else(|| Error::internal("partition search result has no distance column"))?
            .as_primitive::<Float32Type>();
        if row_ids.len() != distances.len() {
            return Err(Error::internal(format!(
                "partition search returned {} row ids and {} distances",
                row_ids.len(),
                distances.len()
            )));
        }
        for ((candidate, &row_id), &distance) in search_result
            .candidates
            .iter()
            .zip(row_ids.values())
            .zip(distances.values())
        {
            if distance.is_nan() {
                continue;
            }
            if segment_mask.is_some_and(|mask| !mask.selected(row_id)) {
                continue;
            }
            let candidate =
                CachedVectorCandidate::new(candidate.partition_id, candidate.offset_in_partition);
            push_top_candidate(
                &mut self.candidates,
                self.candidate_limit,
                candidate,
                distance,
            );
            push_top_candidate(&mut self.results, self.result_limit, row_id, distance);
        }
        Ok(())
    }

    fn finish(self) -> Result<(Vec<CachedVectorCandidate>, RecordBatch)> {
        let mut ordered_candidates = self.candidates.into_vec();
        ordered_candidates.sort_by(|left, right| {
            left.dist
                .cmp(&right.dist)
                .then_with(|| left.id.partition_id().cmp(&right.id.partition_id()))
                .then_with(|| {
                    left.id
                        .offset_in_partition()
                        .cmp(&right.id.offset_in_partition())
                })
        });
        let candidates = ordered_candidates.into_iter().map(|node| node.id).collect();
        Ok((candidates, batch_from_heap(self.results)?))
    }
}

fn push_top_candidate<T: Eq>(
    heap: &mut BinaryHeap<OrderedNode<T>>,
    limit: usize,
    id: T,
    distance: f32,
) {
    if limit == 0 {
        return;
    }
    let node = OrderedNode::new(id, OrderedFloat(distance));
    if heap.len() < limit {
        heap.push(node);
    } else if heap
        .peek()
        .is_some_and(|farthest| farthest.dist > node.dist)
    {
        heap.pop();
        heap.push(node);
    }
}

async fn replay_candidates(
    index: Arc<dyn VectorIndex>,
    query: &Query,
    candidates: &[CachedVectorCandidate],
    result_limit: usize,
    metrics: Arc<dyn MetricsCollector>,
    parallelism: usize,
) -> Result<RecordBatch> {
    let mut offsets_by_partition = HashMap::<u32, Vec<u32>>::new();
    for candidate in candidates {
        offsets_by_partition
            .entry(candidate.partition_id())
            .or_default()
            .push(candidate.offset_in_partition());
    }

    let scored_partitions = futures::stream::iter(offsets_by_partition)
        .map(|(partition_id, offsets)| {
            let index = index.clone();
            let query = query.clone();
            let metrics = metrics.clone();
            async move {
                let batch = index
                    .score_partition_candidates(
                        partition_id as usize,
                        &query,
                        &offsets,
                        metrics.as_ref(),
                    )
                    .await?;
                Result::Ok((partition_id, offsets.len(), batch))
            }
        })
        .buffered(parallelism.max(1))
        .try_collect::<Vec<_>>()
        .await?;

    let mut top_results = BinaryHeap::with_capacity(result_limit);
    for (partition_id, expected_rows, batch) in scored_partitions {
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("candidate scoring result has no row-id column"))?
            .as_primitive::<UInt64Type>();
        let distances = batch
            .column_by_name(DIST_COL)
            .ok_or_else(|| Error::internal("candidate scoring result has no distance column"))?
            .as_primitive::<Float32Type>();
        if row_ids.len() != expected_rows || distances.len() != expected_rows {
            return Err(Error::internal(format!(
                "candidate scoring for partition {partition_id} returned {} row ids and {} distances for {} offsets",
                row_ids.len(),
                distances.len(),
                expected_rows
            )));
        }
        for (&row_id, &distance) in row_ids.values().iter().zip(distances.values()) {
            if distance.is_nan() {
                continue;
            }
            push_top_candidate(&mut top_results, result_limit, row_id, distance);
        }
    }
    batch_from_heap(top_results)
}

fn batch_from_heap(heap: BinaryHeap<OrderedNode<u64>>) -> Result<RecordBatch> {
    let mut ordered = heap.into_vec();
    ordered.sort_by(|left, right| {
        left.dist
            .cmp(&right.dist)
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut row_ids = Vec::with_capacity(ordered.len());
    let mut distances = Vec::with_capacity(ordered.len());
    for result in ordered {
        row_ids.push(result.id);
        distances.push(result.dist.0);
    }
    Ok(RecordBatch::try_new(
        KNN_INDEX_SCHEMA.clone(),
        vec![
            Arc::new(Float32Array::from(distances)),
            Arc::new(UInt64Array::from(row_ids)),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use lance_core::cache::QuickCacheBackend;
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::row_addr_remap::RowAddrRemap;
    use lance_index::Index;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::prefilter::NoFilter;
    use lance_index::vector::ivf::storage::IvfModel;
    use lance_index::vector::quantizer::Quantizer;
    use lance_index::vector::v3::subindex::SubIndexType;
    use lance_index::vector::{ApproxMode, DEFAULT_QUERY_PARALLELISM, PartitionSearchCandidate};
    use lance_io::traits::Reader;
    use roaring::RoaringBitmap;
    use uuid::Uuid;

    use super::*;

    #[derive(Debug, DeepSizeOf)]
    struct TestCandidateIndex {
        miss_partition_searches: AtomicUsize,
        miss_search_yields: AtomicUsize,
        block_miss_searches: AtomicUsize,
        hit_partition_scores: AtomicUsize,
        active_hit_partition_scores: AtomicUsize,
        max_active_hit_partition_scores: AtomicUsize,
        return_malformed_hit_shape: AtomicUsize,
        row_ids: Vec<u64>,
    }

    impl TestCandidateIndex {
        fn new() -> Self {
            Self {
                miss_partition_searches: AtomicUsize::new(0),
                miss_search_yields: AtomicUsize::new(0),
                block_miss_searches: AtomicUsize::new(0),
                hit_partition_scores: AtomicUsize::new(0),
                active_hit_partition_scores: AtomicUsize::new(0),
                max_active_hit_partition_scores: AtomicUsize::new(0),
                return_malformed_hit_shape: AtomicUsize::new(0),
                row_ids: Vec::new(),
            }
        }
    }

    #[async_trait]
    impl Index for TestCandidateIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn index_type(&self) -> IndexType {
            IndexType::IvfRq
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait]
    impl VectorIndex for TestCandidateIndex {
        async fn search(
            &self,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            Err(Error::not_supported("test whole-index search"))
        }

        fn find_partitions(&self, _query: &Query) -> Result<(UInt32Array, Float32Array)> {
            Ok((
                UInt32Array::from(vec![0, 1]),
                Float32Array::from(vec![0.0, 0.0]),
            ))
        }

        fn total_partitions(&self) -> usize {
            2
        }

        async fn search_in_partition(
            &self,
            _partition_id: usize,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            Err(Error::not_supported("test partition search"))
        }

        async fn search_in_partition_with_candidates(
            &self,
            partition_id: usize,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
            _candidate_limit: usize,
        ) -> Result<PartitionSearchResult> {
            self.miss_partition_searches.fetch_add(1, Ordering::Relaxed);
            for _ in 0..self.miss_search_yields.load(Ordering::Relaxed) {
                tokio::task::yield_now().await;
            }
            while self.block_miss_searches.load(Ordering::Relaxed) != 0 {
                tokio::task::yield_now().await;
            }
            let (row_ids, distances) = match partition_id {
                0 => (vec![0, 1], vec![0.4, 0.2]),
                1 => (vec![10, 11], vec![0.3, 0.1]),
                _ => {
                    return Err(Error::invalid_input(format!(
                        "unexpected test partition {partition_id}"
                    )));
                }
            };
            let candidates = (0..row_ids.len())
                .map(|offset| PartitionSearchCandidate {
                    partition_id: partition_id as u32,
                    offset_in_partition: offset as u32,
                })
                .collect();
            let batch = RecordBatch::try_new(
                KNN_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(Float32Array::from(distances)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )?;
            Ok(PartitionSearchResult { batch, candidates })
        }

        async fn score_partition_candidates(
            &self,
            partition_id: usize,
            _query: &Query,
            offsets_in_partition: &[u32],
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            self.hit_partition_scores.fetch_add(1, Ordering::Relaxed);
            let active = self
                .active_hit_partition_scores
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            self.max_active_hit_partition_scores
                .fetch_max(active, Ordering::Relaxed);
            tokio::task::yield_now().await;
            let mut scored = offsets_in_partition
                .iter()
                .map(|offset| match (partition_id, *offset) {
                    (0, 0) => Ok((0, 0.05)),
                    (0, 1) => Ok((1, 0.8)),
                    (1, 0) => Ok((10, 0.6)),
                    (1, 1) => Ok((11, 0.1)),
                    _ => Err(Error::invalid_input(format!(
                        "unexpected test candidate ({partition_id}, {offset})"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            if self.return_malformed_hit_shape.load(Ordering::Relaxed) != 0 {
                scored.pop();
            }
            self.active_hit_partition_scores
                .fetch_sub(1, Ordering::Relaxed);
            let (row_ids, distances): (Vec<_>, Vec<_>) = scored.into_iter().unzip();
            Ok(RecordBatch::try_new(
                KNN_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(Float32Array::from(distances)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )?)
        }

        fn is_loadable(&self) -> bool {
            false
        }

        fn use_residual(&self) -> bool {
            false
        }

        async fn load(
            &self,
            _reader: Arc<dyn Reader>,
            _offset: usize,
            _length: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            Err(Error::not_supported("test index load"))
        }

        fn num_rows(&self) -> u64 {
            0
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
            Box::new(self.row_ids.iter())
        }

        async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
            Ok(())
        }

        async fn to_batch_stream(
            &self,
            _with_vector: bool,
        ) -> Result<datafusion::execution::SendableRecordBatchStream> {
            Err(Error::not_supported("test batch stream"))
        }

        fn ivf_model(&self) -> &IvfModel {
            unreachable!("test cache path does not inspect the IVF model")
        }

        fn quantizer(&self) -> Quantizer {
            unreachable!("test cache path does not inspect the quantizer")
        }

        fn partition_size(&self, _part_id: usize) -> usize {
            2
        }

        fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
            (SubIndexType::Flat, QuantizationType::Rabit)
        }

        fn metric_type(&self) -> lance_linalg::distance::DistanceType {
            lance_linalg::distance::DistanceType::L2
        }
    }

    fn test_query() -> Query {
        Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0, 1.0])),
            k: 2,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 2,
            maximum_nprobes: Some(2),
            ef: None,
            refine_factor: None,
            metric_type: Some(lance_linalg::distance::DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: ApproxMode::Accurate,
        }
    }

    fn test_identity(query: &Query) -> VectorResultsCacheIdentity {
        let metadata = IndexMetadata {
            uuid: Uuid::from_u128(1),
            fields: vec![0],
            covering_fields: vec![],
            name: "vector_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some([0_u32].into_iter().collect()),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.table.VectorIndexDetails".to_string(),
                value: vec![1],
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        };
        VectorResultsCacheIdentity::try_new(
            "memory",
            "manifest-1",
            1,
            &metadata,
            None,
            query,
            lance_linalg::distance::DistanceType::L2,
            SubIndexType::Flat,
            QuantizationType::Rabit,
            &[0, 1],
            RESULTS_CACHE_CANDIDATE_LIMIT,
            false,
            false,
        )
        .unwrap()
    }

    fn row_ids(batch: &RecordBatch) -> Vec<u64> {
        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
    }

    #[test]
    fn accumulator_filters_nan_and_orders_ties_with_aligned_rows() {
        let mut accumulator = SearchAccumulator::new(4, 4);
        accumulator
            .add(
                PartitionSearchResult {
                    batch: RecordBatch::try_new(
                        KNN_INDEX_SCHEMA.clone(),
                        vec![
                            Arc::new(Float32Array::from(vec![0.2, f32::NAN, 0.2])),
                            Arc::new(UInt64Array::from(vec![30, 20, 10])),
                        ],
                    )
                    .unwrap(),
                    candidates: vec![
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 2,
                        },
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 1,
                        },
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 0,
                        },
                    ],
                },
                None,
            )
            .unwrap();

        let (candidates, batch) = accumulator.finish().unwrap();
        assert_eq!(
            candidates,
            vec![
                CachedVectorCandidate::new(0, 0),
                CachedVectorCandidate::new(0, 2),
            ]
        );
        assert_eq!(row_ids(&batch), vec![10, 30]);
        assert_eq!(
            batch[DIST_COL].as_primitive::<Float32Type>().values(),
            &[0.2, 0.2]
        );
    }

    #[test]
    fn accumulator_filters_candidates_outside_segment() {
        let mut accumulator = SearchAccumulator::new(4, 4);
        let segment_mask =
            RowAddrMask::from_allowed(lance_select::RowAddrTreeMap::from_iter([10_u64, 30]));
        accumulator
            .add(
                PartitionSearchResult {
                    batch: RecordBatch::try_new(
                        KNN_INDEX_SCHEMA.clone(),
                        vec![
                            Arc::new(Float32Array::from(vec![0.3, 0.2, 0.1])),
                            Arc::new(UInt64Array::from(vec![30, 20, 10])),
                        ],
                    )
                    .unwrap(),
                    candidates: vec![
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 2,
                        },
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 1,
                        },
                        PartitionSearchCandidate {
                            partition_id: 0,
                            offset_in_partition: 0,
                        },
                    ],
                },
                Some(&segment_mask),
            )
            .unwrap();

        let (candidates, batch) = accumulator.finish().unwrap();
        assert_eq!(
            candidates,
            vec![
                CachedVectorCandidate::new(0, 0),
                CachedVectorCandidate::new(0, 2),
            ]
        );
        assert_eq!(row_ids(&batch), vec![10, 30]);
    }

    #[tokio::test]
    async fn miss_populates_candidates_and_hit_rescores_them() {
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(64 * 1024)));
        let index = Arc::new(TestCandidateIndex::new());
        let query = test_query();
        let identity = test_identity(&query);
        let partitions = Arc::new(UInt32Array::from(vec![0, 1]));
        let centroid_distances = Arc::new(Float32Array::from(vec![0.0, 0.0]));

        let miss = search(ResultsCacheSearchParams {
            cache: &cache,
            identity: identity.clone(),
            index: index.clone(),
            query: &query,
            partitions: partitions.clone(),
            centroid_distances: centroid_distances.clone(),
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        })
        .await
        .unwrap();
        assert!(!miss.was_hit);
        assert_eq!(row_ids(&miss.batch), vec![11, 1]);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 2);
        assert_eq!(index.hit_partition_scores.load(Ordering::Relaxed), 0);

        let hit = search(ResultsCacheSearchParams {
            cache: &cache,
            identity,
            index: index.clone(),
            query: &query,
            partitions,
            centroid_distances,
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        })
        .await
        .unwrap();
        assert!(hit.was_hit);
        assert_eq!(row_ids(&hit.batch), vec![0, 11]);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 2);
        assert_eq!(index.hit_partition_scores.load(Ordering::Relaxed), 2);
        assert_eq!(
            index
                .max_active_hit_partition_scores
                .load(Ordering::Relaxed),
            2
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_cold_searches_populate_candidates_once() {
        const SEARCHES: usize = 4;

        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(64 * 1024)));
        let index = Arc::new(TestCandidateIndex::new());
        index.miss_search_yields.store(16, Ordering::Relaxed);
        let query = test_query();
        let identity = test_identity(&query);
        let partitions = Arc::new(UInt32Array::from(vec![0, 1]));
        let centroid_distances = Arc::new(Float32Array::from(vec![0.0, 0.0]));

        let results = futures::future::join_all((0..SEARCHES).map(|_| {
            search(ResultsCacheSearchParams {
                cache: &cache,
                identity: identity.clone(),
                index: index.clone(),
                query: &query,
                partitions: partitions.clone(),
                centroid_distances: centroid_distances.clone(),
                prefilter: Arc::new(NoFilter),
                segment_mask: None,
                metrics: Arc::new(NoOpMetricsCollector),
                parallelism: 2,
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .unwrap();

        assert_eq!(results.iter().filter(|result| !result.was_hit).count(), 1);
        assert_eq!(results.iter().filter(|result| result.was_hit).count(), 3);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 2);
        assert_eq!(index.hit_partition_scores.load(Ordering::Relaxed), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_cold_search_retries_after_population_is_cancelled() {
        let cache = Arc::new(LanceCache::with_backend(Arc::new(
            QuickCacheBackend::with_capacity(64 * 1024),
        )));
        let index = Arc::new(TestCandidateIndex::new());
        index.block_miss_searches.store(1, Ordering::Relaxed);

        let owner = {
            let cache = cache.clone();
            let index = index.clone();
            tokio::spawn(async move {
                let query = test_query();
                let identity = test_identity(&query);
                search(ResultsCacheSearchParams {
                    cache: cache.as_ref(),
                    identity,
                    index,
                    query: &query,
                    partitions: Arc::new(UInt32Array::from(vec![0, 1])),
                    centroid_distances: Arc::new(Float32Array::from(vec![0.0, 0.0])),
                    prefilter: Arc::new(NoFilter),
                    segment_mask: None,
                    metrics: Arc::new(NoOpMetricsCollector),
                    parallelism: 2,
                })
                .await
            })
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while index.miss_partition_searches.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("population owner did not start both partition searches");

        let contender = {
            let cache = cache.clone();
            let index = index.clone();
            tokio::spawn(async move {
                let query = test_query();
                let identity = test_identity(&query);
                search(ResultsCacheSearchParams {
                    cache: cache.as_ref(),
                    identity,
                    index,
                    query: &query,
                    partitions: Arc::new(UInt32Array::from(vec![0, 1])),
                    centroid_distances: Arc::new(Float32Array::from(vec![0.0, 0.0])),
                    prefilter: Arc::new(NoFilter),
                    segment_mask: None,
                    metrics: Arc::new(NoOpMetricsCollector),
                    parallelism: 2,
                })
                .await
            })
        };
        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 2);
        assert!(!contender.is_finished());

        owner.abort();
        assert!(matches!(owner.await, Err(error) if error.is_cancelled()));
        index.block_miss_searches.store(0, Ordering::Relaxed);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), contender)
            .await
            .expect("contender remained parked after population cancellation")
            .unwrap()
            .unwrap();
        assert!(!result.was_hit);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 4);
    }

    #[tokio::test]
    async fn different_queries_with_same_partitions_do_not_share_candidates() {
        let cache = LanceCache::with_capacity(64 * 1024);
        let index = Arc::new(TestCandidateIndex::new());
        let first_query = test_query();
        let first_identity = test_identity(&first_query);
        let partitions = Arc::new(UInt32Array::from(vec![0, 1]));
        let centroid_distances = Arc::new(Float32Array::from(vec![0.0, 0.0]));

        let first = search(ResultsCacheSearchParams {
            cache: &cache,
            identity: first_identity,
            index: index.clone(),
            query: &first_query,
            partitions: partitions.clone(),
            centroid_distances: centroid_distances.clone(),
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        })
        .await
        .unwrap();
        assert!(!first.was_hit);

        let mut second_query = first_query.clone();
        second_query.key = Arc::new(Float32Array::from(vec![1.0, 0.0]));
        let second_identity = test_identity(&second_query);
        let second = search(ResultsCacheSearchParams {
            cache: &cache,
            identity: second_identity.clone(),
            index: index.clone(),
            query: &second_query,
            partitions: partitions.clone(),
            centroid_distances: centroid_distances.clone(),
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        })
        .await
        .unwrap();
        assert!(!second.was_hit);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 4);
        assert_eq!(index.hit_partition_scores.load(Ordering::Relaxed), 0);

        let repeated_second = search(ResultsCacheSearchParams {
            cache: &cache,
            identity: second_identity,
            index: index.clone(),
            query: &second_query,
            partitions,
            centroid_distances,
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        })
        .await
        .unwrap();
        assert!(repeated_second.was_hit);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 4);
        assert_eq!(index.hit_partition_scores.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn results_cache_malformed_hit_shape_falls_back_and_replaces_entry() {
        let cache = LanceCache::with_capacity(64 * 1024);
        let index = Arc::new(TestCandidateIndex::new());
        let query = test_query();
        let identity = test_identity(&query);
        let partitions = Arc::new(UInt32Array::from(vec![0, 1]));
        let centroid_distances = Arc::new(Float32Array::from(vec![0.0, 0.0]));
        let params = || ResultsCacheSearchParams {
            cache: &cache,
            identity: identity.clone(),
            index: index.clone(),
            query: &query,
            partitions: partitions.clone(),
            centroid_distances: centroid_distances.clone(),
            prefilter: Arc::new(NoFilter),
            segment_mask: None,
            metrics: Arc::new(NoOpMetricsCollector),
            parallelism: 2,
        };

        let cold = search(params()).await.unwrap();
        assert!(!cold.was_hit);
        assert_eq!(row_ids(&cold.batch), vec![11, 1]);

        index.return_malformed_hit_shape.store(1, Ordering::Relaxed);
        let fallback = search(params()).await.unwrap();
        assert!(!fallback.was_hit);
        assert_eq!(row_ids(&fallback.batch), vec![11, 1]);
        assert_eq!(index.miss_partition_searches.load(Ordering::Relaxed), 4);

        index.return_malformed_hit_shape.store(0, Ordering::Relaxed);
        let replaced = search(params()).await.unwrap();
        assert!(replaced.was_hit);
        assert_eq!(row_ids(&replaced.batch), vec![0, 11]);
    }
}
