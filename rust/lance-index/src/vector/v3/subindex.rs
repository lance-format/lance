// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

use crate::metrics::MetricsCollector;
use crate::vector::graph::OrderedNode;
use crate::vector::storage::{QueryResidual, QueryScratch, VectorStore};
use crate::vector::{flat, hnsw};
use crate::{
    prefilter::PreFilter,
    vector::{PartitionSearchResult, Query},
};
/// A sub index for IVF index
pub trait IvfSubIndex: Send + Sync + Debug + DeepSizeOf {
    type QueryParams: Send + Sync + for<'a> From<&'a Query>;
    type BuildParams: Clone + Send + Sync;

    /// Serialize per-partition metadata for a partition the build left empty.
    ///
    /// Most sub-indexes store no per-partition rows and return `None`, which
    /// the writer stores as an empty string. Sub-indexes that still parse
    /// per-partition metadata for empty partitions (an empty `HNSW` graph must
    /// round-trip through `read_sub_index_batch`) return the serialized
    /// metadata here, carrying the segment's build parameters so an empty
    /// partition cannot misrepresent the parameters readers derive from the
    /// segment's metadata.
    fn empty_partition_metadata(_build_params: Option<&Self::BuildParams>) -> Option<String> {
        None
    }

    /// Load the sub index from a record batch with a single row
    fn load(data: RecordBatch) -> Result<Self>
    where
        Self: Sized;

    fn name() -> &'static str;

    fn metadata_key() -> &'static str;

    /// Return the schema of the sub index
    fn schema() -> arrow_schema::SchemaRef;

    /// The subset of [`Self::schema`] that [`Self::load`] actually reads.
    ///
    /// Index files always carry the full `schema()`, so narrowing the read is
    /// purely a storage optimization: it keeps write-only columns from being
    /// fetched, without changing what is written. `None` reads every column.
    fn read_columns() -> Option<&'static [&'static str]> {
        None
    }

    /// Build the sub-index query params for one IVF partition.
    ///
    /// `query_key` is the partition-processed key and `dist_q_c` is the
    /// distance from the query to that partition's centroid. Callers share one
    /// [`Query`] across probes and pass both per-partition values here. The
    /// default goes through `From<&Query>` and therefore still clones the query
    /// once per probe; implementations that can build their params directly
    /// should override it to avoid that clone.
    fn query_params(query: &Query, query_key: &ArrayRef, dist_q_c: f32) -> Self::QueryParams {
        query_params_from_processed(query, query_key, dist_q_c)
    }

    /// Search the sub index for nearest neighbors.
    /// # Arguments:
    /// * `query` - The query vector
    /// * `k` - The number of nearest neighbors to return
    /// * `params` - The query parameters
    /// * `prefilter` - The prefilter object indicating which vectors to skip
    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch>;

    /// Search the sub-index, reusing scratch buffers owned by the caller.
    #[allow(clippy::too_many_arguments)]
    fn search_with_scratch(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
        _residual: Option<QueryResidual<'_>>,
        _scratch: &mut QueryScratch,
    ) -> Result<RecordBatch> {
        self.search(query, k, params, storage, prefilter, metrics)
    }

    /// Search the sub-index while preserving each result's partition-local
    /// storage offset. Implementations that cannot expose stable offsets return
    /// [`Error::NotSupported`].
    #[allow(clippy::too_many_arguments)]
    fn search_with_candidates_with_scratch(
        &self,
        _query: ArrayRef,
        _candidate_limit: usize,
        _params: Self::QueryParams,
        _storage: &impl VectorStore,
        _prefilter: Arc<dyn PreFilter>,
        _metrics: &dyn MetricsCollector,
        _partition_id: u32,
        _residual: Option<QueryResidual<'_>>,
        _scratch: &mut QueryScratch,
    ) -> Result<PartitionSearchResult> {
        Err(Error::not_supported(
            "partition-local candidate identities are not supported for this sub-index",
        ))
    }

    /// Return true if this sub-index can accumulate candidates into a caller-owned heap.
    fn supports_global_topk_heap() -> bool {
        false
    }

    /// Search this partition and accumulate candidates into a caller-owned top-k heap.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_topk(
        &self,
        _query: ArrayRef,
        _k: usize,
        _params: Self::QueryParams,
        _storage: &impl VectorStore,
        _prefilter: Arc<dyn PreFilter>,
        _heap: &mut BinaryHeap<OrderedNode<u64>>,
        _metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        unimplemented!("global top-k heap search is not supported for this sub-index")
    }

    /// Search this partition and accumulate candidates into a caller-owned top-k heap,
    /// reusing scratch buffers owned by the caller.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_topk_with_scratch(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        _residual: Option<QueryResidual<'_>>,
        _scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        self.accumulate_topk(query, k, params, storage, prefilter, heap, metrics)
    }

    /// Given a vector storage, containing all the data for the IVF partition, build the sub index.
    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self>
    where
        Self: Sized;

    fn remap(&self, mapping: &RowAddrRemap, store: &impl VectorStore) -> Result<Self>
    where
        Self: Sized;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;
}

fn query_params_from_processed<P>(query: &Query, query_key: &ArrayRef, dist_q_c: f32) -> P
where
    P: for<'a> From<&'a Query>,
{
    let mut query = query.clone();
    query.key = query_key.clone();
    query.dist_q_c = dist_q_c;
    P::from(&query)
}

#[derive(Debug, Clone, Copy)]
pub enum SubIndexType {
    Flat,
    Hnsw,
}

impl std::fmt::Display for SubIndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flat => write!(f, "{}", flat::index::FlatIndex::name()),
            Self::Hnsw => write!(f, "{}", hnsw::builder::HNSW::name()),
        }
    }
}

impl TryFrom<&str> for SubIndexType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "FLAT" => Ok(Self::Flat),
            "HNSW" => Ok(Self::Hnsw),
            _ => Err(Error::index(format!("unknown sub index type {}", value))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Float32Array;
    use lance_linalg::distance::DistanceType;

    use super::*;
    use crate::vector::ApproxMode;

    #[derive(Debug)]
    struct CapturedParams {
        key: ArrayRef,
        dist_q_c: f32,
        k: usize,
    }

    impl From<&Query> for CapturedParams {
        fn from(query: &Query) -> Self {
            Self {
                key: query.key.clone(),
                dist_q_c: query.dist_q_c,
                k: query.k,
            }
        }
    }

    #[test]
    fn default_query_params_receive_the_processed_key() {
        let original_key = Arc::new(Float32Array::from(vec![1.0, 2.0])) as ArrayRef;
        let processed_key = Arc::new(Float32Array::from(vec![3.0, 4.0])) as ArrayRef;
        let query = Query {
            column: "vector".to_owned(),
            key: original_key.clone(),
            k: 7,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: 0,
            dist_q_c: 0.0,
            approx_mode: ApproxMode::Normal,
        };

        let params: CapturedParams = query_params_from_processed(&query, &processed_key, 9.5);

        assert!(Arc::ptr_eq(&params.key, &processed_key));
        assert!(!Arc::ptr_eq(&params.key, &original_key));
        assert_eq!(params.dist_q_c, 9.5);
        assert_eq!(params.k, 7);
        assert!(Arc::ptr_eq(&query.key, &original_key));
        assert_eq!(query.dist_q_c, 0.0);
    }
}
