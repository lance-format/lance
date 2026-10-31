// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{Float32Array, RecordBatch, UInt32Array};
use async_trait::async_trait;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, stream};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::scalar::RowAddrTranslator;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pairwise::PairwisePartition;
use lance_index::vector::quantizer::{QuantizationType, Quantizer};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::vamana::VamanaBuildParams;
use lance_index::vector::{PartitionSearchControl, Query, VECTOR_RESULT_SCHEMA, VectorIndex};
use lance_index::{Index, IndexType};
use lance_io::scheduler::{IoStats, ScanStats};
use lance_io::spill::SpillStore;
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;

use super::rescore::rescore;
use crate::dataset::Dataset;
use crate::index::resolve_index_column;
use crate::{Error, Result};

/// `index` as queries search it when its partitions hold Vamana graphs,
/// otherwise `index` itself. `column` may name the index instead, as when
/// it is prewarmed, and is resolved as the opening of its file resolves it.
pub fn wrap_for_search(
    dataset: &Dataset,
    index_meta: &IndexMetadata,
    column: &str,
    index: Arc<dyn VectorIndex>,
) -> Result<Arc<dyn VectorIndex>> {
    if !matches!(index.sub_index_type().0, SubIndexType::Vamana) {
        return Ok(index);
    }
    let (column, _) = resolve_index_column(dataset.schema(), index_meta, column)?;
    Ok(Arc::new(VamanaIvfIndex {
        inner: index,
        dataset: Arc::new(dataset.clone()),
        column,
        recorded_params: super::details::recorded_params(index_meta),
    }))
}

/// An IVF_VAMANA_* index as a query searches it: the IVF index whose
/// partitions hold Vamana graphs, answering with exact distances.
///
/// A walk ranks a partition's rows by their codes. Over scalar or product
/// codes the nearest of them are scored again from the dataset's own vectors
/// (see [`rescore`]), unless a refine will score them again anyway; flat
/// codes are exact already. Everything but search goes to the IVF index, and
/// `as_any` returns it, so maintenance finds the index it builds from.
///
/// Lance opens a vector index for every query, so this keeps nothing a query
/// would rather find cached, and is never cached itself: it holds the dataset
/// that opened it, whose session holds the cache.
#[derive(Debug)]
pub struct VamanaIvfIndex {
    inner: Arc<dyn VectorIndex>,
    dataset: Arc<Dataset>,
    column: String,
    /// The graph parameters the segment's details record.
    recorded_params: Option<VamanaBuildParams>,
}

/// How many rows each stage of a query keeps.
#[derive(Debug, Clone, Copy)]
struct Sizes {
    /// The rows a partition, or a range of partitions searched together,
    /// answers with: `k × refine_factor`, as from any IVF index.
    results: usize,
    /// The walk's nearest candidates a query looks at again, at their exact
    /// distances unless the codes are exact: `k × refine_factor`, or `2k`
    /// without a refine factor.
    shortlist: usize,
    /// `L`, the candidates each walk keeps: `ef`, or half as many again as
    /// the shortlist, and never fewer than it.
    search_list_size: usize,
}

impl Sizes {
    fn of(query: &Query) -> Result<Self> {
        let times_k = |factor: u32| {
            query.k.checked_mul(factor as usize).ok_or_else(|| {
                Error::invalid_input(format!(
                    "k {} times refine factor {factor} overflows",
                    query.k
                ))
            })
        };
        let results = times_k(query.refine_factor.unwrap_or(1))?;
        let shortlist = times_k(query.refine_factor.unwrap_or(2))?;
        if let Some(ef) = query.ef
            && ef < results
        {
            return Err(Error::index(format!(
                "ef must be greater than or equal to k ({ef} < {results})"
            )));
        }
        let search_list_size = query
            .ef
            .unwrap_or(shortlist.saturating_add(shortlist / 2))
            .max(shortlist);
        Ok(Self {
            results,
            shortlist,
            search_list_size,
        })
    }
}

/// A bound on a query's distances in the units a walk over `codes` measures
/// in: scalar and product codes measure cosine as the squared L2 distance
/// between unit vectors, which is twice the cosine distance.
fn walk_bound(
    bound: Option<f32>,
    codes: QuantizationType,
    distance_type: DistanceType,
) -> Option<f32> {
    if codes != QuantizationType::Flat && distance_type == DistanceType::Cosine {
        bound.map(|bound| bound * 2.0)
    } else {
        bound
    }
}

impl VamanaIvfIndex {
    fn codes(&self) -> QuantizationType {
        self.inner.sub_index_type().1
    }

    /// Whether `query`'s candidates are scored again here: their codes are
    /// approximate, and no refine reads their vectors again after the index.
    fn rescores(&self, query: &Query) -> bool {
        self.codes() != QuantizationType::Flat && query.refine_factor.is_none()
    }

    /// The query each partition's walk answers.
    fn partition_query(&self, query: &Query, sizes: &Sizes) -> Query {
        let mut partition_query = query.clone();
        partition_query.k = if self.rescores(query) {
            sizes.shortlist
        } else {
            sizes.results
        };
        partition_query.refine_factor = None;
        partition_query.ef = Some(sizes.search_list_size);
        // A refined query arrives with its bounds on the codes' scale already:
        // the scanner scales them for the index and refines at exact distances.
        if query.refine_factor.is_none() {
            let distance_type = self.inner.metric_type();
            partition_query.lower_bound =
                walk_bound(query.lower_bound, self.codes(), distance_type);
            partition_query.upper_bound =
                walk_bound(query.upper_bound, self.codes(), distance_type);
        }
        partition_query
    }

    /// The query's results among the nearest candidates of `found` together,
    /// at their exact distances, with the reads counted in `io_stats`.
    async fn rescore_shortlist(
        &self,
        query: &Query,
        sizes: &Sizes,
        found: &[RecordBatch],
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        let found = concat_batches(&VECTOR_RESULT_SCHEMA, found)?;
        let distances = found.column(0).as_primitive::<Float32Type>().values();
        let row_ids = found.column(1).as_primitive::<UInt64Type>().values();
        let mut shortlist = (0..found.num_rows()).collect::<Vec<_>>();
        shortlist.sort_by(|a, b| {
            distances[*a]
                .total_cmp(&distances[*b])
                .then(row_ids[*a].cmp(&row_ids[*b]))
        });
        shortlist.truncate(sizes.shortlist);
        rescore(
            &self.dataset,
            &self.column,
            self.inner.metric_type(),
            query,
            shortlist.into_iter().map(|row| row_ids[row]).collect(),
            sizes.results,
            io_stats,
        )
        .await
    }

    fn inner_mut(&mut self) -> Result<&mut dyn VectorIndex> {
        Arc::get_mut(&mut self.inner)
            .map(|inner| inner as &mut dyn VectorIndex)
            .ok_or_else(|| {
                Error::internal("an IVF_VAMANA index shared by queries cannot be remapped")
            })
    }
}

fn check_partition_range(
    partitions: &UInt32Array,
    q_c_dists: &Float32Array,
    start_idx: usize,
    end_idx: usize,
) -> Result<()> {
    if partitions.len() != q_c_dists.len() {
        return Err(Error::invalid_input(format!(
            "partition count {} does not match centroid distance count {}",
            partitions.len(),
            q_c_dists.len()
        )));
    }
    if start_idx > end_idx || end_idx > partitions.len() {
        return Err(Error::invalid_input(format!(
            "invalid partition search range [{start_idx}, {end_idx}) for {} partitions",
            partitions.len()
        )));
    }
    Ok(())
}

impl DeepSizeOf for VamanaIvfIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.inner.deep_size_of_children(context)
    }
}

#[async_trait]
impl Index for VamanaIvfIndex {
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    /// The IVF index's statistics. An index whose graphs are all empty
    /// records its graph parameters only in its details, and reports those,
    /// so that the segments built after it from its statistics keep them.
    fn statistics(&self) -> Result<serde_json::Value> {
        let mut statistics = self.inner.statistics()?;
        if let Some(params) = &self.recorded_params
            && let Some(sub_index) = statistics
                .get_mut("sub_index")
                .and_then(serde_json::Value::as_object_mut)
            && !sub_index.contains_key("params")
        {
            // Through text, as a partition's parameters come: `alpha` then
            // reads alike from either.
            let params = serde_json::from_str(&serde_json::to_string(params)?)?;
            sub_index.insert("params".to_string(), params);
        }
        Ok(statistics)
    }

    async fn prewarm(&self) -> Result<()> {
        self.inner.prewarm().await
    }

    fn index_type(&self) -> IndexType {
        self.inner.index_type()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        self.inner.calculate_included_frags().await
    }
}

/// Batch and prepared partition searches stay unsupported, as their trait
/// defaults say: neither would score the candidates again.
#[async_trait]
impl VectorIndex for VamanaIvfIndex {
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        self.inner.search(query, pre_filter, metrics).await
    }

    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
        self.inner.find_partitions(query)
    }

    fn total_partitions(&self) -> usize {
        self.inner.total_partitions()
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let sizes = Sizes::of(query)?;
        let found = self
            .inner
            .search_in_partition(
                partition_id,
                &self.partition_query(query, &sizes),
                pre_filter,
                metrics,
            )
            .await?;
        if !self.rescores(query) {
            return Ok(found);
        }
        self.rescore_shortlist(query, &sizes, &[found], metrics.io_stats())
            .await
    }

    /// Partitions are searched one after another, and a range of them is
    /// scored again in one read.
    fn auto_query_parallelism(&self, _cpu_pool_size: usize) -> usize {
        1
    }

    async fn search_partitions(
        self: Arc<Self>,
        query: Query,
        partitions: Arc<UInt32Array>,
        q_c_dists: Arc<Float32Array>,
        start_idx: usize,
        end_idx: usize,
        pre_filter: Arc<dyn PreFilter>,
        control: Option<Arc<dyn PartitionSearchControl>>,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Result<SendableRecordBatchStream> {
        check_partition_range(&partitions, &q_c_dists, start_idx, end_idx)?;
        // As HNSW refuses a short list only in a partition it searches.
        if start_idx == end_idx {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
                stream::empty(),
            )));
        }
        let sizes = Sizes::of(&query)?;
        if !self.rescores(&query) {
            return self
                .inner
                .clone()
                .search_partitions(
                    self.partition_query(&query, &sizes),
                    partitions,
                    q_c_dists,
                    start_idx,
                    end_idx,
                    pre_filter,
                    control,
                    metrics,
                )
                .await;
        }
        // A control stops the search on the rows the partitions answer with,
        // so each partition answers with its own rescored rows.
        if let Some(control) = control {
            let searched = stream::try_unfold(start_idx, move |idx| {
                let index = self.clone();
                let partitions = partitions.clone();
                let q_c_dists = q_c_dists.clone();
                let mut query = query.clone();
                let pre_filter = pre_filter.clone();
                let control = control.clone();
                let metrics = metrics.clone();
                async move {
                    if idx >= end_idx || control.should_stop() {
                        return Ok(None);
                    }
                    query.dist_q_c = q_c_dists.value(idx);
                    let batch = index
                        .search_in_partition(
                            partitions.value(idx) as usize,
                            &query,
                            pre_filter,
                            metrics.as_ref(),
                        )
                        .await?;
                    control.record_batch(&batch);
                    Ok::<_, Error>(Some((batch, idx + 1)))
                }
            });
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
                searched.map_err(DataFusionError::from),
            )));
        }
        let io_stats = metrics.io_stats();
        let found = self
            .inner
            .clone()
            .search_partitions(
                self.partition_query(&query, &sizes),
                partitions,
                q_c_dists,
                start_idx,
                end_idx,
                pre_filter,
                None,
                metrics,
            )
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let nearest = self
            .rescore_shortlist(&query, &sizes, &found, io_stats)
            .await?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            VECTOR_RESULT_SCHEMA.clone(),
            stream::once(async move { Ok(nearest) }),
        )))
    }

    fn is_loadable(&self) -> bool {
        self.inner.is_loadable()
    }

    fn use_residual(&self) -> bool {
        self.inner.use_residual()
    }

    async fn load(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        self.inner.load(reader, offset, length).await
    }

    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        self.inner
            .load_partition(reader, offset, length, partition_id)
            .await
    }

    async fn partition_reader(
        &self,
        partition_id: usize,
        with_vector: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        self.inner
            .partition_reader(partition_id, with_vector, metrics)
            .await
    }

    fn supports_pairwise_vectors(&self) -> bool {
        self.inner.supports_pairwise_vectors()
    }

    async fn prepare_pairwise_partition(
        &self,
        partition_id: usize,
        batch_size: usize,
        memory_limit: usize,
        spill_store: &dyn SpillStore,
    ) -> Result<PairwisePartition> {
        self.inner
            .prepare_pairwise_partition(partition_id, batch_size, memory_limit, spill_store)
            .await
    }

    async fn to_batch_stream(&self, with_vector: bool) -> Result<SendableRecordBatchStream> {
        self.inner.to_batch_stream(with_vector).await
    }

    fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
        self.inner.row_ids()
    }

    async fn remap(&mut self, mapping: &RowAddrRemap) -> Result<()> {
        self.inner_mut()?.remap(mapping).await
    }

    fn stored_fragments(&self) -> Option<RoaringBitmap> {
        self.inner.stored_fragments()
    }

    async fn remap_streaming(&mut self, translator: &RowAddrTranslator) -> Result<()> {
        self.inner_mut()?.remap_streaming(translator).await
    }

    fn metric_type(&self) -> DistanceType {
        self.inner.metric_type()
    }

    fn ivf_model(&self) -> &IvfModel {
        self.inner.ivf_model()
    }

    fn quantizer(&self) -> Quantizer {
        self.inner.quantizer()
    }

    fn partition_size(&self, part_id: usize) -> usize {
        self.inner.partition_size(part_id)
    }

    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        self.inner.sub_index_type()
    }

    fn open_io_stats(&self) -> Option<ScanStats> {
        self.inner.open_io_stats()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Float32Array;
    use lance_index::vector::DEFAULT_QUERY_PARALLELISM;
    use rstest::rstest;

    use super::*;

    fn query(k: usize, refine_factor: Option<u32>, ef: Option<usize>) -> Query {
        Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0f32; 4])),
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef,
            refine_factor,
            metric_type: None,
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            adaptive_stop: None,
        }
    }

    /// `(results, shortlist, search_list_size)` for `k`, `refine_factor` and
    /// `ef`: the shortlist is twice `k` without a refine factor, and the list
    /// is half as long again unless `ef` sets it, and never shorter than the
    /// shortlist.
    #[rstest]
    #[case::defaults(10, None, None, (10, 20, 30))]
    #[case::refined(10, Some(3), None, (30, 30, 45))]
    #[case::short_ef_widened(10, None, Some(15), (10, 20, 20))]
    #[case::long_ef_kept(10, None, Some(100), (10, 20, 100))]
    #[case::ef_equal_to_the_results(10, Some(2), Some(20), (20, 20, 20))]
    fn sizes_follow_k_refine_factor_and_ef(
        #[case] k: usize,
        #[case] refine_factor: Option<u32>,
        #[case] ef: Option<usize>,
        #[case] expected: (usize, usize, usize),
    ) {
        let sizes = Sizes::of(&query(k, refine_factor, ef)).unwrap();
        assert_eq!(
            (sizes.results, sizes.shortlist, sizes.search_list_size),
            expected
        );
    }

    /// As over HNSW, `ef` below `k × refine_factor` is refused, and so is a
    /// `k × refine_factor` that does not fit.
    #[rstest]
    #[case::ef_below_k(
        query(10, None, Some(9)),
        "ef must be greater than or equal to k (9 < 10)",
        |error: &Error| matches!(error, Error::Index { .. })
    )]
    #[case::ef_below_the_refined_k(
        query(10, Some(2), Some(15)),
        "ef must be greater than or equal to k (15 < 20)",
        |error: &Error| matches!(error, Error::Index { .. })
    )]
    #[case::overflow(
        query(usize::MAX, Some(2), None),
        "overflows",
        |error: &Error| matches!(error, Error::InvalidInput { .. })
    )]
    fn sizes_refuse_what_no_walk_can_answer(
        #[case] query: Query,
        #[case] reason: &str,
        #[case] is_kind: fn(&Error) -> bool,
    ) {
        let error = Sizes::of(&query).unwrap_err();
        assert!(is_kind(&error), "{error:?}");
        assert!(error.to_string().contains(reason), "{error}");
    }

    #[rstest]
    #[case::flat_cosine(QuantizationType::Flat, DistanceType::Cosine, 0.25)]
    #[case::scalar_cosine(QuantizationType::Scalar, DistanceType::Cosine, 0.5)]
    #[case::product_cosine(QuantizationType::Product, DistanceType::Cosine, 0.5)]
    #[case::flat_l2(QuantizationType::Flat, DistanceType::L2, 0.25)]
    #[case::scalar_l2(QuantizationType::Scalar, DistanceType::L2, 0.25)]
    #[case::product_l2(QuantizationType::Product, DistanceType::L2, 0.25)]
    fn a_bound_is_doubled_where_codes_measure_cosine_as_l2(
        #[case] codes: QuantizationType,
        #[case] distance_type: DistanceType,
        #[case] expected: f32,
    ) {
        assert_eq!(walk_bound(Some(0.25), codes, distance_type), Some(expected));
        assert_eq!(walk_bound(None, codes, distance_type), None);
    }
}
