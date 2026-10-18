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
use crate::{prefilter::PreFilter, vector::Query};
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

    // fork(vamana)
    /// Whether [`Self::index_vectors_incremental`] may extend a base of
    /// `vertices` rows into a merged partition of `num_rows` rows whose first
    /// `matched` rows are the base's; `false` when it builds the partition
    /// anew whatever the base holds. More matched rows or fewer merged ones
    /// never turn it false, so that a merge holding a bound on each need not
    /// read a base it would not extend. By default nothing is extended.
    fn may_extend(_vertices: usize, _matched: usize, _num_rows: usize) -> bool {
        false
    }

    /// Build the sub index of `storage`, a partition an index merge assembled
    /// from the rows `base` was built over and new rows, by extending `base`.
    ///
    /// The first `base_rows` positions of `storage` are rows of `base`'s
    /// partition, in their order there, some of them filtered out. Past them a
    /// row id proves nothing: a rewritten row comes back with a new vector under
    /// its old stable row id, or at its old address when rewritten in place.
    /// By default the sub index is built anew.
    fn index_vectors_incremental(
        _base: &Self,
        storage: &impl VectorStore,
        _base_rows: usize,
        params: Self::BuildParams,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Self::index_vectors(storage, params)
    }

    fn remap(&self, mapping: &RowAddrRemap, store: &impl VectorStore) -> Result<Self>
    where
        Self: Sized;

    /// Encode the sub index into a record batch
    fn to_batch(&self) -> Result<RecordBatch>;
}

#[derive(Debug, Clone, Copy)]
pub enum SubIndexType {
    Flat,
    Hnsw,
    Vamana, // fork(vamana)
}

impl std::fmt::Display for SubIndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Flat => write!(f, "{}", flat::index::FlatIndex::name()),
            Self::Hnsw => write!(f, "{}", hnsw::builder::HNSW::name()),
            Self::Vamana => write!(f, "{}", crate::vector::vamana::Vamana::name()), // fork(vamana)
        }
    }
}

impl TryFrom<&str> for SubIndexType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "FLAT" => Ok(Self::Flat),
            "HNSW" => Ok(Self::Hnsw),
            "VAMANA" => Ok(Self::Vamana), // fork(vamana)
            _ => Err(Error::index(format!("unknown sub index type {}", value))),
        }
    }
}
