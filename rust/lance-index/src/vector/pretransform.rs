// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query pre-transforming vector index decorator.

use std::any::Any;
use std::fmt::Debug;
use std::mem::size_of_val;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt32Array};
use async_trait::async_trait;
use datafusion_execution::SendableRecordBatchStream;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_io::traits::Reader;
use lance_linalg::distance::DistanceType;
use roaring::RoaringBitmap;

use super::ivf::storage::IvfModel;
use super::pca::PcaModel;
use super::quantizer::{QuantizationType, Quantizer};
use super::v3::subindex::SubIndexType;
use super::{PartitionSearchControl, PreparedPartitionSearchHandle, Query, VectorIndex};
use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;
use crate::{Index, IndexType};

/// A vector index decorator that transforms query vectors before delegation.
///
/// The transform is applied exactly once at each outer query entry point.
/// Prepared partition execution delegates directly because its query was
/// transformed during preparation.
#[derive(Debug)]
pub struct PreTransformIndex {
    inner: Arc<dyn VectorIndex>,
    model: Arc<PcaModel>,
}

impl PreTransformIndex {
    /// Creates a query decorator backed by a trained PCA model.
    pub fn new_pca(inner: Arc<dyn VectorIndex>, model: Arc<PcaModel>) -> Self {
        Self { inner, model }
    }

    fn transformed_query(&self, query: &Query) -> Result<Query> {
        if query.lower_bound.is_some() || query.upper_bound.is_some() {
            return Err(Error::not_supported(
                "PCA vector indices do not support lower_bound or upper_bound queries".to_string(),
            ));
        }
        let mut transformed = query.clone();
        transformed.key = self.model.transform_query(query.key.as_ref())?;
        Ok(transformed)
    }
}

impl DeepSizeOf for PreTransformIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let inner_size = self.inner.deep_size_of_children(context);
        let model_size = if context.mark_seen(Arc::as_ptr(&self.model) as *const () as usize) {
            size_of_val(self.model.as_ref()) + self.model.deep_size_of_children(context)
        } else {
            0
        };
        inner_size + model_size
    }
}

#[async_trait]
impl Index for PreTransformIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.inner.statistics()
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

#[async_trait]
impl VectorIndex for PreTransformIndex {
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let query = self.transformed_query(query)?;
        self.inner.search(&query, pre_filter, metrics).await
    }

    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
        let query = self.transformed_query(query)?;
        self.inner.find_partitions(&query)
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
        let query = self.transformed_query(query)?;
        self.inner
            .search_in_partition(partition_id, &query, pre_filter, metrics)
            .await
    }

    async fn prepare_partition_search(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<PreparedPartitionSearchHandle> {
        let query = self.transformed_query(query)?;
        self.inner
            .prepare_partition_search(partition_id, &query, pre_filter, metrics)
            .await
    }

    fn search_prepared_partition(
        &self,
        prepared: PreparedPartitionSearchHandle,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        self.inner.search_prepared_partition(prepared, metrics)
    }

    fn supports_prepared_partition_search(&self) -> bool {
        self.inner.supports_prepared_partition_search()
    }

    fn auto_query_parallelism(&self, cpu_pool_size: usize) -> usize {
        self.inner.auto_query_parallelism(cpu_pool_size)
    }

    fn prepare_partition_query(
        self: Arc<Self>,
        query: &Query,
    ) -> Result<Option<(Arc<dyn VectorIndex>, Query)>> {
        let query = self.transformed_query(query)?;
        Ok(Some((self.inner.clone(), query)))
    }

    #[allow(clippy::too_many_arguments)]
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
        let query = self.transformed_query(&query)?;
        self.inner
            .clone()
            .search_partitions(
                query, partitions, q_c_dists, start_idx, end_idx, pre_filter, control, metrics,
            )
            .await
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
        let inner = self.inner.load(reader, offset, length).await?;
        Ok(Box::new(Self::new_pca(
            Arc::from(inner),
            self.model.clone(),
        )))
    }

    async fn load_partition(
        &self,
        reader: Arc<dyn Reader>,
        offset: usize,
        length: usize,
        partition_id: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        let inner = self
            .inner
            .load_partition(reader, offset, length, partition_id)
            .await?;
        Ok(Box::new(Self::new_pca(
            Arc::from(inner),
            self.model.clone(),
        )))
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
        let inner = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            Error::index(
                "cannot remap PreTransformIndex because its wrapped index is shared".to_string(),
            )
        })?;
        inner.remap(mapping).await
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

    fn supports_pairwise_vectors(&self) -> bool {
        // Pairwise scoring compares stored codes. Those codes were written
        // after the PCA projection, so the inner index can score them directly.
        self.inner.supports_pairwise_vectors()
    }

    async fn prepare_pairwise_partition(
        &self,
        partition_id: usize,
        batch_size: usize,
        memory_limit: usize,
        spill_store: &dyn lance_io::spill::SpillStore,
    ) -> Result<super::pairwise::PairwisePartition> {
        self.inner
            .prepare_pairwise_partition(partition_id, batch_size, memory_limit, spill_store)
            .await
    }

    fn stored_fragments(&self) -> Option<RoaringBitmap> {
        self.inner.stored_fragments()
    }

    fn open_io_stats(&self) -> Option<lance_io::scheduler::ScanStats> {
        self.inner.open_io_stats()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Float32Type;
    use arrow_array::{Array, FixedSizeListArray, Float32Array};
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{StreamExt, stream};
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    use super::super::VECTOR_RESULT_SCHEMA;
    use super::super::flat::index::FlatQuantizer;
    use super::super::pca::{PcaModel, PcaParams};
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;

    fn test_model() -> Arc<PcaModel> {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![2.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, -1.0]),
            2,
        )
        .expect("test vectors must be valid");
        Arc::new(PcaModel::train(&vectors, &PcaParams::new(1)).expect("test PCA model must train"))
    }

    #[derive(Debug, Default)]
    struct MockCalls {
        search: usize,
        find_partitions: usize,
        search_partitions: usize,
        search_in_partition: usize,
        prepare: usize,
        execute_prepared: usize,
    }

    #[derive(Debug)]
    struct MockVectorIndex {
        calls: Mutex<MockCalls>,
        queries: Mutex<Vec<Query>>,
        ivf: IvfModel,
        row_ids: Vec<u64>,
    }

    impl MockVectorIndex {
        fn new() -> Self {
            Self {
                calls: Mutex::new(MockCalls::default()),
                queries: Mutex::new(Vec::new()),
                ivf: IvfModel::empty(),
                row_ids: vec![7, 9],
            }
        }

        fn record_query(&self, query: &Query) {
            self.queries.lock().unwrap().push(query.clone());
        }

        fn result_batch() -> RecordBatch {
            RecordBatch::new_empty(VECTOR_RESULT_SCHEMA.clone())
        }

        fn empty_stream() -> SendableRecordBatchStream {
            Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
                stream::empty(),
            ))
        }
    }

    impl DeepSizeOf for MockVectorIndex {
        fn deep_size_of_children(&self, context: &mut Context) -> usize {
            self.ivf.deep_size_of_children(context) + self.row_ids.deep_size_of_children(context)
        }
    }

    #[async_trait]
    impl Index for MockVectorIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({"mock": true}))
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn index_type(&self) -> IndexType {
            IndexType::IvfFlat
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::from_iter([3, 5]))
        }
    }

    #[async_trait]
    impl VectorIndex for MockVectorIndex {
        async fn search(
            &self,
            query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            self.calls.lock().unwrap().search += 1;
            self.record_query(query);
            Ok(Self::result_batch())
        }

        fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
            self.calls.lock().unwrap().find_partitions += 1;
            self.record_query(query);
            Ok((
                UInt32Array::from(vec![2, 4]),
                Float32Array::from(vec![0.2, 0.4]),
            ))
        }

        fn total_partitions(&self) -> usize {
            6
        }

        async fn search_in_partition(
            &self,
            _partition_id: usize,
            query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            self.calls.lock().unwrap().search_in_partition += 1;
            self.record_query(query);
            Ok(Self::result_batch())
        }

        async fn prepare_partition_search(
            &self,
            _partition_id: usize,
            query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn MetricsCollector,
        ) -> Result<PreparedPartitionSearchHandle> {
            self.calls.lock().unwrap().prepare += 1;
            self.record_query(query);
            Ok(Box::new(query.clone()))
        }

        fn search_prepared_partition(
            &self,
            prepared: PreparedPartitionSearchHandle,
            _metrics: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            self.calls.lock().unwrap().execute_prepared += 1;
            let query = prepared.downcast::<Query>().map_err(|_| {
                Error::invalid_input("test prepared handle did not contain a Query")
            })?;
            self.record_query(&query);
            Ok(Self::result_batch())
        }

        fn supports_prepared_partition_search(&self) -> bool {
            true
        }

        fn auto_query_parallelism(&self, cpu_pool_size: usize) -> usize {
            cpu_pool_size.saturating_sub(1)
        }

        #[allow(clippy::too_many_arguments)]
        async fn search_partitions(
            self: Arc<Self>,
            query: Query,
            _partitions: Arc<UInt32Array>,
            _q_c_dists: Arc<Float32Array>,
            _start_idx: usize,
            _end_idx: usize,
            _pre_filter: Arc<dyn PreFilter>,
            _control: Option<Arc<dyn PartitionSearchControl>>,
            _metrics: Arc<dyn MetricsCollector>,
        ) -> Result<SendableRecordBatchStream> {
            self.calls.lock().unwrap().search_partitions += 1;
            self.record_query(&query);
            Ok(Self::empty_stream())
        }

        fn is_loadable(&self) -> bool {
            true
        }

        fn use_residual(&self) -> bool {
            true
        }

        async fn load(
            &self,
            _reader: Arc<dyn Reader>,
            _offset: usize,
            _length: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            Err(Error::index("mock load is not configured"))
        }

        async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
            Ok(Self::empty_stream())
        }

        fn num_rows(&self) -> u64 {
            self.row_ids.len() as u64
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
            Box::new(self.row_ids.iter())
        }

        async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
            Ok(())
        }

        fn metric_type(&self) -> DistanceType {
            DistanceType::Cosine
        }

        fn ivf_model(&self) -> &IvfModel {
            &self.ivf
        }

        fn quantizer(&self) -> Quantizer {
            Quantizer::Flat(FlatQuantizer::new(2, DistanceType::Cosine))
        }

        fn partition_size(&self, part_id: usize) -> usize {
            part_id + 10
        }

        fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
            (SubIndexType::Flat, QuantizationType::Flat)
        }
    }

    fn query() -> Query {
        Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![1.0, 2.0])),
            k: 17,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 2,
            maximum_nprobes: Some(8),
            search_effort: 0.5,
            ef: Some(31),
            refine_factor: Some(3),
            metric_type: Some(DistanceType::Dot),
            use_index: true,
            query_parallelism: 4,
            dist_q_c: 1.25,
            approx_mode: super::super::ApproxMode::Accurate,
        }
    }

    fn decorated() -> (Arc<PreTransformIndex>, Arc<MockVectorIndex>, Arc<PcaModel>) {
        let inner = Arc::new(MockVectorIndex::new());
        let model = test_model();
        let index = Arc::new(PreTransformIndex::new_pca(inner.clone(), model.clone()));
        (index, inner, model)
    }

    fn assert_transformed_query(actual: &Query, original: &Query, model: &PcaModel) {
        let expected = model
            .transform_query(original.key.as_ref())
            .expect("test query must project");
        assert_eq!(
            actual.key.as_primitive::<Float32Type>().values(),
            expected.as_primitive::<Float32Type>().values()
        );
        assert_eq!(actual.column, original.column);
        assert_eq!(actual.k, original.k);
        assert_eq!(actual.lower_bound, original.lower_bound);
        assert_eq!(actual.upper_bound, original.upper_bound);
        assert_eq!(actual.minimum_nprobes, original.minimum_nprobes);
        assert_eq!(actual.maximum_nprobes, original.maximum_nprobes);
        assert_eq!(actual.search_effort, original.search_effort);
        assert_eq!(actual.ef, original.ef);
        assert_eq!(actual.refine_factor, original.refine_factor);
        assert_eq!(actual.metric_type, original.metric_type);
        assert_eq!(actual.use_index, original.use_index);
        assert_eq!(actual.query_parallelism, original.query_parallelism);
        assert_eq!(actual.dist_q_c, original.dist_q_c);
        assert_eq!(actual.approx_mode, original.approx_mode);
    }

    #[tokio::test]
    async fn transforms_find_and_search_once() {
        let (index, inner, model) = decorated();
        let query = query();

        index.find_partitions(&query).unwrap();
        assert_transformed_query(&inner.queries.lock().unwrap()[0], &query, &model);

        index
            .search(&query, Arc::new(NoFilter), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_transformed_query(&inner.queries.lock().unwrap()[1], &query, &model);
    }

    #[tokio::test]
    async fn search_partitions_delegates_override_after_one_transform() {
        let (index, inner, model) = decorated();
        let query = query();
        let stream = index
            .search_partitions(
                query.clone(),
                Arc::new(UInt32Array::from(vec![0, 1, 2])),
                Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])),
                0,
                3,
                Arc::new(NoFilter),
                None,
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap();
        let batches = stream.collect::<Vec<_>>().await;

        assert!(batches.is_empty());
        let calls = inner.calls.lock().unwrap();
        assert_eq!(calls.search_partitions, 1);
        assert_eq!(calls.search_in_partition, 0);
        assert_transformed_query(&inner.queries.lock().unwrap()[0], &query, &model);
    }

    #[tokio::test]
    async fn direct_partition_search_transforms_once() {
        let (index, inner, model) = decorated();
        let query = query();

        index
            .search_in_partition(3, &query, Arc::new(NoFilter), &NoOpMetricsCollector)
            .await
            .unwrap();

        assert_eq!(inner.calls.lock().unwrap().search_in_partition, 1);
        assert_transformed_query(&inner.queries.lock().unwrap()[0], &query, &model);
    }

    #[tokio::test]
    async fn prepared_search_only_transforms_during_prepare() {
        let (index, inner, model) = decorated();
        let query = query();

        let prepared = index
            .prepare_partition_search(3, &query, Arc::new(NoFilter), &NoOpMetricsCollector)
            .await
            .unwrap();
        index
            .search_prepared_partition(prepared, &NoOpMetricsCollector)
            .unwrap();

        let calls = inner.calls.lock().unwrap();
        assert_eq!(calls.prepare, 1);
        assert_eq!(calls.execute_prepared, 1);
        let queries = inner.queries.lock().unwrap();
        assert_transformed_query(&queries[0], &query, &model);
        assert_transformed_query(&queries[1], &query, &model);
    }

    #[tokio::test]
    async fn delegates_index_and_vector_capabilities() {
        let (index, inner, _) = decorated();

        assert_eq!(
            index.statistics().unwrap(),
            serde_json::json!({"mock": true})
        );
        index.prewarm().await.unwrap();
        assert_eq!(index.index_type(), IndexType::IvfFlat);
        assert_eq!(
            index
                .calculate_included_frags()
                .await
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![3, 5]
        );
        assert_eq!(index.total_partitions(), 6);
        assert!(index.is_loadable());
        assert!(index.use_residual());
        assert_eq!(index.num_rows(), 2);
        assert_eq!(index.row_ids().copied().collect::<Vec<_>>(), vec![7, 9]);
        assert_eq!(index.metric_type(), DistanceType::Cosine);
        assert!(std::ptr::eq(index.ivf_model(), inner.ivf_model()));
        assert_eq!(index.quantizer().code_dim(), 2);
        assert_eq!(index.partition_size(4), 14);
        assert!(matches!(
            index.sub_index_type(),
            (SubIndexType::Flat, QuantizationType::Flat)
        ));
        assert!(index.supports_prepared_partition_search());
        assert_eq!(index.auto_query_parallelism(8), 7);
        assert!(index.open_io_stats().is_none());

        let batch_stream = index.to_batch_stream(true).await.unwrap();
        assert!(batch_stream.collect::<Vec<_>>().await.is_empty());
    }

    #[tokio::test]
    async fn remap_rejects_shared_inner_without_panicking() {
        let inner = Arc::new(MockVectorIndex::new());
        let mut index = PreTransformIndex::new_pca(inner.clone(), test_model());

        let error = index.remap(&RowAddrRemap::empty()).await.unwrap_err();

        assert!(matches!(error, Error::Index { .. }));
        assert!(
            error.to_string().contains("wrapped index is shared"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn prepared_partition_query_is_reused_by_parallel_searches() {
        let (index, inner, model) = decorated();
        let original = query();
        let (search_index, prepared_query) = index
            .prepare_partition_query(&original)
            .expect("partition query preparation should succeed")
            .expect("decorator should prepare the partition query");

        for partition_id in 0..3 {
            search_index
                .search_in_partition(
                    partition_id,
                    &prepared_query,
                    Arc::new(NoFilter),
                    &NoOpMetricsCollector,
                )
                .await
                .expect("partition search should succeed");
        }

        assert_eq!(inner.calls.lock().unwrap().search_in_partition, 3);
        for actual in inner.queries.lock().unwrap().iter() {
            assert_transformed_query(actual, &original, &model);
        }
    }

    #[test]
    fn deep_size_deduplicates_shared_inner_and_transform() {
        let (index, _, _) = decorated();
        let mut context = Context::new();

        let first = index.deep_size_of_children(&mut context);
        let second = index.deep_size_of_children(&mut context);

        assert!(first > 0);
        assert_eq!(second, 0);
    }

    #[rstest]
    #[case::lower(Some(0.1), None)]
    #[case::upper(None, Some(0.5))]
    #[case::both(Some(0.1), Some(0.5))]
    fn pca_query_rejects_distance_bounds(
        #[case] lower_bound: Option<f32>,
        #[case] upper_bound: Option<f32>,
    ) {
        let (index, _, _) = decorated();
        let mut original = query();
        original.lower_bound = lower_bound;
        original.upper_bound = upper_bound;
        let error = index.transformed_query(&original).unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("PCA vector indices"));
        assert!(error.to_string().contains("lower_bound or upper_bound"));
    }

    #[test]
    fn transforming_a_query_preserves_the_original_for_refine() {
        let (index, _, model) = decorated();
        let original = query();

        let transformed = index
            .transformed_query(&original)
            .expect("query transform should succeed");

        assert_eq!(
            original
                .key
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("original key must remain Float32")
                .values(),
            &[1.0, 2.0]
        );
        assert_transformed_query(&transformed, &original, &model);
    }
}
