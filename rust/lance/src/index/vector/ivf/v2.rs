// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::marker::PhantomData;
use std::{
    any::Any,
    borrow::Cow,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::{
        Arc, LazyLock, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::index::vector::{
    IndexFileVersion,
    builder::index_type_string,
    utils::{gather_covering_columns_by_row_id, row_id_take_indices},
};
use crate::index::{PreFilter, vector::VectorIndex};
use arrow::compute::concat_batches;
use arrow_arith::numeric::sub;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt32Array, UInt64Array, cast::AsArray};
use arrow_schema::{DataType, Field};
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::future::BoxFuture;
use futures::prelude::stream::{self, TryStreamExt};
use futures::{StreamExt, TryFutureExt};
use lance_arrow::RecordBatchExt;
use lance_core::cache::{
    CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
    KeyBuilder, LanceCache, WeakLanceCache,
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::{IO_TYPE_LOAD_VECTOR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{CachedFileMetadata, FileReader, FileReaderOptions, ReaderProjection};
use lance_index::cache_pb::IvfStateHeader;
use lance_index::frag_reuse::{CompactFragReuseIndex, CompactFragReuseIndexHandle};
use lance_index::metrics::{LocalMetricsCollector, MetricsCollector, NoOpMetricsCollector};
use lance_index::prefilter::NoFilter;
use lance_index::scalar::RowIdRemapper;
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::{blocked_ex_code_bytes, padded_query_len};
use lance_index::vector::bq::rabit_ex_bits;
use lance_index::vector::bq::storage::{RabitQueryEstimator, SEGMENT_NUM_CODES};
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{
    QuantizationType, Quantizer, QuantizerMetadata, QuantizerStorage,
};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::storage::{
    QueryResidual, QueryScratch, QueryScratchCapacity, QueryScratchPool, RabitRawQueryContext,
    VectorStore,
};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, Index, IndexType, pb,
    vector::{
        DISTANCE_TYPE_KEY, PartitionSearchControl, PreparedPartitionSearchHandle, Query,
        VECTOR_RESULT_SCHEMA,
        ivf::storage::IVF_METADATA_KEY,
        quantizer::Quantization,
        storage::{IvfQuantizationStorage, PartitionColumns},
        v3::subindex::IvfSubIndex,
    },
};
use lance_index::{INDEX_METADATA_SCHEMA_KEY, IndexMetadata};
use lance_io::local::to_local_path;
use lance_io::scheduler::{IoStats, ScanStats, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_io::{
    ReadBatchParams, object_store::ObjectStore, scheduler::ScanScheduler, traits::Reader,
};
use lance_linalg::distance::DistanceType;
use lance_select::RowAddrTreeMap;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, instrument};
use uuid::Uuid;

use super::{IvfIndexPartitionStatistics, IvfIndexStatistics, maybe_centroids_for_stats};

pub(crate) type RabitSearchCacheCell = Arc<Mutex<Option<Option<Arc<RabitSearchCache>>>>>;

/// Serializable state of an IVF index, sufficient to reconstruct the index
/// without re-reading global buffers from object storage.
///
/// Serializable, type-specific state of an IVF index.
///
/// Generic over `Q` so that the parsed quantizer metadata (`Q::Metadata`) can
/// be stored directly, avoiding repeated JSON round-trips on reconstruction.
/// Produced by [`IVFIndex::to_state_entry`] and wrapped in [`IvfStateEntryBox`]
/// for storage in the index cache.
#[derive(Debug, Clone)]
pub(crate) struct IvfIndexState<Q: Quantization> {
    pub(crate) index_file_path: String,
    pub(crate) uuid: String,
    pub(crate) ivf: IvfModel,
    /// IvfModel for the auxiliary/storage file (quantizer row layout).
    /// The index and aux files have independent row layouts, so we must store
    /// both to avoid using wrong row offsets during reconstruction.
    pub(crate) aux_ivf: IvfModel,
    pub(crate) distance_type: DistanceType,
    pub(crate) sub_index_metadata: Vec<String>,
    /// Parsed quantizer metadata — stored directly to avoid JSON re-parsing on
    /// every warm-path reconstruction.
    pub(crate) metadata: <Q::Storage as QuantizerStorage>::Metadata,
    pub(crate) sub_index_type: SubIndexType,
    pub(crate) quantization_type: QuantizationType,
    /// File sizes for the index and auxiliary files, used to avoid HEAD requests
    /// when reconstructing from cache.
    pub(crate) index_file_size: u64,
    pub(crate) aux_file_size: u64,
    /// Runtime-only cache, intentionally excluded from the CacheCodec wire format.
    pub(crate) rq_search_cache: RabitSearchCacheCell,
}

/// Number of prepared partitions handed to a single `spawn_cpu` dispatch on the
/// streaming search path.
///
/// The streaming path deliberately avoids per-partition CPU-task fan-out (a measured
/// 14-30% latency win, see #6475). Searching a batch of partitions per `spawn_cpu`
/// keeps most of that benefit — the per-dispatch overhead is paid once per
/// `STREAMING_SEARCH_BATCH_SIZE` partitions instead of once per partition — while
/// keeping the channel `recv`/`send` in async code so no CPU-pool thread ever parks on
/// a channel (which can deadlock the pool on small hosts, see #7642). `should_stop` is
/// still checked per partition, so early-stop granularity is unchanged.
///
/// This is a tunable knob: larger batches amortize dispatch overhead further and keep
/// more work on a single CPU thread, at the cost of more prepared partitions held in
/// memory at once. The batch is an upper bound: the search loop greedily drains
/// whatever is already prepared rather than waiting for a full batch, so a slow
/// producer yields small batches (matching the old search-as-it-arrives latency) and
/// only a fast producer fills whole ones. Override with the
/// `LANCE_IVF_STREAMING_SEARCH_BATCH_SIZE` environment variable.
pub(crate) const DEFAULT_STREAMING_SEARCH_BATCH_SIZE: usize = 16;

pub(crate) static STREAMING_SEARCH_BATCH_SIZE: LazyLock<usize> = LazyLock::new(|| {
    let batch_size = std::env::var("LANCE_IVF_STREAMING_SEARCH_BATCH_SIZE")
        .map(|value| {
            value
                .parse()
                .expect("failed to parse LANCE_IVF_STREAMING_SEARCH_BATCH_SIZE")
        })
        .unwrap_or(DEFAULT_STREAMING_SEARCH_BATCH_SIZE);
    assert!(
        batch_size > 0,
        "LANCE_IVF_STREAMING_SEARCH_BATCH_SIZE must be greater than 0, got {batch_size}"
    );
    batch_size
});

struct PreparedPartitionSearch<S: IvfSubIndex, Q: Quantization> {
    query: Query,
    pre_filter: Arc<dyn PreFilter>,
    partition_id: usize,
    /// Rows this partition occupies in the storage file. Compared against the loaded
    /// partition's row count to decide whether a storage position is also a file
    /// position (see [`CoveringGather::WholeRange`]).
    partition_rows: usize,
    partition_centroid: Option<ArrayRef>,
    rq_search_cache: Option<Arc<RabitSearchCache>>,
    raw_query_context: Option<Arc<RabitRawQueryContext>>,
    part_entry: Arc<PartitionEntry<S, Q>>,
    _marker: PhantomData<(S, Q)>,
}

/// The covering ("included") columns one query needs from one index, resolved once per
/// search.
///
/// Absent (`None` at the call sites that hold this) for an ordinary index **and** for a
/// query whose covering projection names none of the declared columns: both mean the same
/// thing to the search path -- emit `[_distance, _rowid]` and do no covering read at all.
/// Collapsing the second case into "project an already-read batch down to nothing" is
/// exactly the regression [`Query::covering_projection`] documents.
struct QueryCovering {
    /// `[_rowid, <included...>]`, in index declaration order. Both the declared stream
    /// schema and the emitted batches are built from this.
    schema: arrow_schema::SchemaRef,
    /// The same included columns by name and in the same order, for the storage read.
    columns: Vec<String>,
}

/// What the covering gather must read for one partition's search survivors.
enum CoveringGather {
    /// The query needs no covering column from this index.
    NotNeeded,
    /// Read exactly these positions within the partition's row range. Strictly ascending
    /// and deduplicated, which is what the reader's take path requires.
    Positions(Vec<u32>),
    /// Positions are not derivable: a deferred fragment-reuse remap drops rows from the
    /// loaded partition, so a storage position is no longer the file position it was read
    /// from. Read the partition's whole range instead and match by row id.
    WholeRange,
}

/// Where one heap survivor's covering values live, recorded while its partition is still
/// loaded so the gather after the heap settles is a bounded read rather than a re-search.
#[derive(Debug, Clone, Copy)]
struct CoveringLocation {
    partition_id: usize,
    /// Position within the partition's row range, or `None` when positions are not
    /// derivable for that partition (see [`CoveringGather::WholeRange`]).
    position: Option<u32>,
}

#[derive(Debug)]
pub(crate) struct RabitSearchCache {
    rotated_centroids: Vec<f32>,
    code_dim: usize,
}

pub(crate) fn empty_rabit_search_cache_cell() -> RabitSearchCacheCell {
    Arc::new(Mutex::new(None))
}

fn rabit_search_cache_cell(cache: Option<Arc<RabitSearchCache>>) -> RabitSearchCacheCell {
    Arc::new(Mutex::new(Some(cache)))
}

fn rotated_partition_centroid_slice(
    cache: Option<&RabitSearchCache>,
    partition_id: usize,
) -> Option<&[f32]> {
    let cache = cache?;
    let start = partition_id.checked_mul(cache.code_dim)?;
    let end = start.checked_add(cache.code_dim)?;
    cache.rotated_centroids.get(start..end)
}

/// `f32` scratch needed for the ex-bit query state: a zero-padded query copy
/// when the rotated dim is not a multiple of the 64-dim kernel block (the
/// FastScan ex LUT is built directly from the query, with no f32 table).
fn rabit_ex_scratch_len(dim: usize, num_bits: u8) -> usize {
    let multi_bit = rabit_ex_bits(num_bits)
        .map(|ex_bits| ex_bits > 0)
        .unwrap_or(true);
    if !multi_bit || dim.is_multiple_of(64) {
        0
    } else {
        padded_query_len(dim)
    }
}

fn rabit_u8_scratch_len(dim: usize, num_bits: u8) -> usize {
    let binary_dist_table_len = dim * 4;
    let ex_dist_table_len = rabit_ex_bits(num_bits)
        .ok()
        .and_then(|ex_bits| match ex_bits {
            2 | 4 | 8 => Some(blocked_ex_code_bytes(dim, ex_bits)),
            _ => None,
        })
        .map(|ex_code_len| ex_code_len * 2 * SEGMENT_NUM_CODES)
        .unwrap_or_default();
    binary_dist_table_len.max(ex_dist_table_len)
}

fn rabit_query_scratch_capacity(
    dim: usize,
    max_partition_len: usize,
    num_bits: u8,
) -> QueryScratchCapacity {
    let dist_table_len = dim * 4;
    let ex_scratch_len = rabit_ex_scratch_len(dim, num_bits);
    let u8_scratch_len = rabit_u8_scratch_len(dim, num_bits);

    QueryScratchCapacity::new(
        max_partition_len,
        dim + dist_table_len + ex_scratch_len,
        max_partition_len.max(dist_table_len),
        u8_scratch_len,
    )
}

impl<Q: Quantization> DeepSizeOf for IvfIndexState<Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.index_file_path.deep_size_of_children(context)
            + self.uuid.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.aux_ivf.deep_size_of_children(context)
            + self.sub_index_metadata.deep_size_of_children(context)
            + self.metadata.deep_size_of_children(context)
            + self
                .rq_search_cache
                .lock()
                .ok()
                .and_then(|cache| cache.as_ref().and_then(|cache| cache.as_ref().cloned()))
                .map(|cache| cache.rotated_centroids.len() * std::mem::size_of::<f32>())
                .unwrap_or_default()
    }
}

/// Object-safe interface for a type-erased `IvfIndexState<Q>`.
///
/// Stored as `Arc<dyn IvfStateEntry>` inside [`IvfStateEntryBox`], which is
/// the concrete type held in the index cache. Splitting the trait from the
/// wrapper lets the cache infrastructure work with a sized type while the
/// hot paths call `reconstruct` without knowing `Q`.
pub(crate) trait IvfStateEntry: DeepSizeOf + Send + Sync + 'static {
    fn serialize_state(&self, w: &mut CacheEntryWriter<'_>) -> Result<()>;

    fn reconstruct<'a>(
        &'a self,
        object_store: Arc<ObjectStore>,
        file_metadata_cache: &'a LanceCache,
        index_cache: LanceCache,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    ) -> BoxFuture<'a, Result<Arc<dyn VectorIndex>>>;
}

/// Sized wrapper around `Arc<dyn IvfStateEntry>` for use as a cache value.
///
/// `IvfStateEntryBox` is the `CacheKey::ValueType` for `IvfIndexStateCacheKey`.
/// `CacheCodecImpl` on this type holds the full deserialization dispatch
/// (matching on `quantization_type`) so callers never need to branch on
/// index type after a cache hit.
pub(crate) struct IvfStateEntryBox(pub(crate) Arc<dyn IvfStateEntry>);

impl DeepSizeOf for IvfStateEntryBox {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// Wire format:
/// ```text
/// HEADER   : IvfStateHeader proto (paths, types, quantizer metadata JSON)
/// RAW_BLOB : IVF model protobuf
/// RAW_BLOB : quantizer extra-metadata buffer (may be empty)
/// RAW_BLOB : auxiliary IVF model protobuf
/// ```
impl CacheCodecImpl for IvfStateEntryBox {
    const TYPE_ID: &'static str = "lance.vector.ivf.IvfState";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        self.0.serialize_state(w)
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        // Parse the common header, then dispatch on quantization_type to
        // construct the right IvfIndexState<Q>.
        let header: IvfStateHeader = r.read_header()?;

        let ivf_bytes = r.read_raw()?;
        let ivf = IvfModel::try_from(
            pb::Ivf::decode(ivf_bytes.as_ref())
                .map_err(|e| lance_core::Error::io(format!("IvfIndexState IVF decode: {e}")))?,
        )?;

        let extra_bytes = r.read_raw()?;

        let aux_ivf_bytes = r.read_raw()?;
        let aux_ivf =
            IvfModel::try_from(pb::Ivf::decode(aux_ivf_bytes.as_ref()).map_err(|e| {
                lance_core::Error::io(format!("IvfIndexState aux IVF decode: {e}"))
            })?)?;

        let distance_type = DistanceType::try_from(header.distance_type.as_str())?;
        let sub_index_type = SubIndexType::try_from(header.sub_index_type.as_str())?;
        let quantization_type = header.quantization_type.parse::<QuantizationType>()?;

        // Helper: parse Q::Metadata from the JSON+extra_bytes in the header,
        // then build an IvfStateEntryBox wrapping IvfIndexState<Q>.
        fn make_entry<Q: Quantization + 'static>(
            header: IvfStateHeader,
            ivf: IvfModel,
            aux_ivf: IvfModel,
            extra_bytes: bytes::Bytes,
            distance_type: DistanceType,
            sub_index_type: SubIndexType,
            quantization_type: QuantizationType,
        ) -> Result<IvfStateEntryBox>
        where
            <Q::Storage as QuantizerStorage>::Metadata:
                serde::de::DeserializeOwned + QuantizerMetadata,
        {
            let mut metadata: <Q::Storage as QuantizerStorage>::Metadata =
                serde_json::from_str(&header.quantizer_metadata_json)
                    .map_err(|e| lance_core::Error::io(format!("IvfIndexState metadata: {e}")))?;
            if !extra_bytes.is_empty() {
                metadata.parse_buffer(extra_bytes)?;
            }
            Ok(IvfStateEntryBox(Arc::new(IvfIndexState::<Q> {
                index_file_path: header.index_file_path,
                uuid: header.uuid,
                ivf,
                aux_ivf,
                distance_type,
                sub_index_metadata: header.sub_index_metadata,
                metadata,
                sub_index_type,
                quantization_type,
                index_file_size: header.index_file_size,
                aux_file_size: header.aux_file_size,
                rq_search_cache: empty_rabit_search_cache_cell(),
            })))
        }

        match quantization_type {
            QuantizationType::Flat => make_entry::<FlatQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::FlatBin => make_entry::<FlatBinQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Product => make_entry::<ProductQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Scalar => make_entry::<ScalarQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Rabit => make_entry::<RabitQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
        }
    }
}

impl<Q: Quantization + 'static> IvfStateEntry for IvfIndexState<Q> {
    fn serialize_state(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let quantizer_metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState metadata: {e}")))?;
        let extra = self.metadata.extra_metadata()?;
        let extra = extra.as_deref().unwrap_or(&[]);

        let header = IvfStateHeader {
            index_file_path: self.index_file_path.clone(),
            uuid: self.uuid.to_string(),
            distance_type: self.distance_type.to_string(),
            sub_index_metadata: self.sub_index_metadata.clone(),
            sub_index_type: self.sub_index_type.to_string(),
            quantization_type: self.quantization_type.to_string(),
            quantizer_metadata_json,
            index_file_size: self.index_file_size,
            aux_file_size: self.aux_file_size,
        };
        let ivf_bytes = pb::Ivf::try_from(&self.ivf)?.encode_to_vec();
        let aux_ivf_bytes = pb::Ivf::try_from(&self.aux_ivf)?.encode_to_vec();

        w.write_header(&header)?;
        w.write_raw(&ivf_bytes)?;
        w.write_raw(extra)?;
        w.write_raw(&aux_ivf_bytes)?;
        Ok(())
    }

    fn reconstruct<'a>(
        &'a self,
        object_store: Arc<ObjectStore>,
        file_metadata_cache: &'a LanceCache,
        index_cache: LanceCache,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    ) -> BoxFuture<'a, Result<Arc<dyn VectorIndex>>> {
        Box::pin(async move {
            match self.sub_index_type {
                SubIndexType::Flat => {
                    reconstruct_typed::<FlatIndex, Q>(
                        self,
                        object_store,
                        file_metadata_cache,
                        index_cache,
                        frag_reuse_index,
                    )
                    .await
                }
                SubIndexType::Hnsw => {
                    reconstruct_typed::<HNSW, Q>(
                        self,
                        object_store,
                        file_metadata_cache,
                        index_cache,
                        frag_reuse_index,
                    )
                    .await
                }
            }
        })
    }
}

struct FileMetadataCacheKey;

impl CacheKey for FileMetadataCacheKey {
    type ValueType = CachedFileMetadata;
    fn type_name() -> &'static str {
        "CachedFileMetadata"
    }
    fn key(&self) -> std::borrow::Cow<'_, str> {
        "".into()
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.ivf-file-metadata-key", 1)
    }

    fn write_key(&self, _builder: &mut KeyBuilder) {}
}

/// Open a FileReader, reusing cached file metadata if available.
async fn open_reader_cached(
    scheduler: &Arc<ScanScheduler>,
    path: &Path,
    cache: &LanceCache,
    known_file_size: u64,
) -> Result<FileReader> {
    let file_cache = cache.with_key_prefix(path.as_ref());
    // CachedFileSize::new(0) == CachedFileSize::unknown(); passing the raw
    // hint directly is safe — the type already encodes 0 as "unknown".
    let cached_size = CachedFileSize::new(known_file_size);

    if let Some(cached_meta) = file_cache.get_with_key(&FileMetadataCacheKey).await {
        let file_scheduler = scheduler.open_file(path, &cached_size).await?;
        let encodings_io = Arc::new(LanceEncodingsIo::new(file_scheduler));
        FileReader::try_open_with_file_metadata(
            encodings_io,
            path.clone(),
            None,
            Arc::<DecoderPlugins>::default(),
            cached_meta,
            cache,
            FileReaderOptions::default(),
        )
        .await
    } else {
        let file_scheduler = scheduler.open_file(path, &cached_size).await?;
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            cache,
            FileReaderOptions::default(),
        )
        .await?;
        // File metadata is store-free, so it outlives the reader opened here:
        // cache it to spare later reconstructions the footer read.
        file_cache
            .insert_with_key(&FileMetadataCacheKey, reader.metadata().clone())
            .await;
        Ok(reader)
    }
}

#[derive(Debug)]
pub struct PartitionEntry<S: IvfSubIndex, Q: Quantization> {
    pub index: S,
    pub storage: Q::Storage,
    partition_rows: OnceLock<Arc<RowAddrTreeMap>>,
    partition_rows_accounted: AtomicBool,
}

impl<S: IvfSubIndex, Q: Quantization> PartitionEntry<S, Q> {
    pub(super) fn new(index: S, storage: Q::Storage) -> Self {
        Self {
            index,
            storage,
            partition_rows: OnceLock::new(),
            partition_rows_accounted: AtomicBool::new(false),
        }
    }

    fn partition_rows(&self) -> Arc<RowAddrTreeMap> {
        self.partition_rows
            .get_or_init(|| Arc::new(self.storage.row_ids().collect()))
            .clone()
    }
}

impl<S: IvfSubIndex, Q: Quantization> DeepSizeOf for PartitionEntry<S, Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.index.deep_size_of_children(context)
            + self.storage.deep_size_of_children(context)
            + self
                .partition_rows
                .get()
                .map(|rows| rows.deep_size_of_children(context))
                .unwrap_or_default()
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> VectorIndexCacheEntry
    for PartitionEntry<S, Q>
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Cache key for IVF partitions
#[derive(Debug, Clone)]
pub struct IVFPartitionKey<S: IvfSubIndex, Q: Quantization> {
    pub partition_id: usize,
    _marker: PhantomData<(S, Q)>,
}

impl<S: IvfSubIndex, Q: Quantization> IVFPartitionKey<S, Q> {
    pub fn new(partition_id: usize) -> Self {
        Self {
            partition_id,
            _marker: PhantomData,
        }
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> CacheKey for IVFPartitionKey<S, Q> {
    type ValueType = PartitionEntry<S, Q>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("ivf-{}", self.partition_id).into()
    }

    fn type_name() -> &'static str {
        "IVFPartition"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.ivf-partition-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(S::name());
        builder.write_variant(match Q::quantization_type() {
            QuantizationType::Flat => 0,
            QuantizationType::FlatBin => 1,
            QuantizationType::Product => 2,
            QuantizationType::Scalar => 3,
            QuantizationType::Rabit => 4,
        });
        builder.write_u64(self.partition_id as u64);
    }

    fn codec() -> Option<CacheCodec> {
        super::partition_serde::partition_entry_codec::<S, Q>()
    }
}

/// IVF Index.
#[derive(Debug)]
pub struct IVFIndex<S: IvfSubIndex + 'static, Q: Quantization + 'static> {
    /// Local display path (via `to_local_path`), used for statistics.
    uri: String,
    /// Object-store path to the index file (forward-slash separated).
    /// Used by `cacheable_state()` for cross-platform reconstruction.
    index_path: String,
    uuid: Uuid,

    /// Ivf model
    ivf: IvfModel,

    reader: FileReader,
    /// Narrowed read of the index file, when the sub-index declares that
    /// [`IvfSubIndex::load`] consumes only part of what it writes. `None` reads
    /// every column. Built once here because it is fallible and only depends on
    /// the file schema.
    read_projection: Option<ReaderProjection>,
    sub_index_metadata: Vec<String>,
    storage: IvfQuantizationStorage<Q>,

    distance_type: DistanceType,

    index_cache: WeakLanceCache,

    io_parallelism: usize,
    /// Cumulative I/O performed while opening this index (file footers, IVF
    /// centroids, quantization metadata).  Captured once in `try_new`; exposed
    /// via [`VectorIndex::open_io_stats`] so the opening query can attribute the
    /// one-time open cost to its plan metrics.
    open_io_stats: ScanStats,
    scratch_pool: Arc<QueryScratchPool>,
    use_query_residual: bool,
    use_residual_scratch: bool,
    rq_search_cache: Option<Arc<RabitSearchCache>>,

    _marker: PhantomData<(S, Q)>,
}

impl<S: IvfSubIndex, Q: Quantization> DeepSizeOf for IVFIndex<S, Q> {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // `Uuid` is a fixed 16-byte struct with no heap children, so contributes 0.
        self.uri.deep_size_of_children(context)
            + self.index_path.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.sub_index_metadata.deep_size_of_children(context)
            + self.storage.deep_size_of_children(context)
            + self.scratch_pool.deep_size_of_children(context)
            + self
                .rq_search_cache
                .as_ref()
                .map(|cache| cache.rotated_centroids.len() * std::mem::size_of::<f32>())
                .unwrap_or_default()
        // Skipping session since it is a weak ref
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization> IVFIndex<S, Q> {
    fn read_projection(reader: &FileReader) -> Result<Option<ReaderProjection>> {
        S::read_columns()
            .map(|columns| {
                lance_file::versions::reader_projection_from_column_names(
                    reader.metadata().version(),
                    reader.schema(),
                    columns,
                )
            })
            .transpose()
    }

    async fn cache_partition_rows(
        index_cache: &WeakLanceCache,
        partition_id: usize,
        partition: &Arc<PartitionEntry<S, Q>>,
    ) -> Result<Arc<RowAddrTreeMap>> {
        let rows = partition.partition_rows();
        if !partition.partition_rows_accounted.load(Ordering::Acquire) {
            let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);
            if index_cache
                .insert_with_key(&cache_key, partition.clone())
                .await
            {
                partition
                    .partition_rows_accounted
                    .store(true, Ordering::Release);
            }
        }
        Ok(rows)
    }

    async fn prefilter_for_partition(
        index_cache: &WeakLanceCache,
        partition_id: usize,
        partition: &Arc<PartitionEntry<S, Q>>,
        pre_filter: Arc<dyn PreFilter>,
    ) -> Result<Arc<dyn PreFilter>> {
        if pre_filter.is_empty() {
            return Ok(Arc::new(NoFilter));
        }
        if !pre_filter.needs_partition_row_ids() {
            return Ok(pre_filter);
        }
        let rows = Self::cache_partition_rows(index_cache, partition_id, partition).await?;
        if pre_filter.is_empty_for(rows.as_ref()) {
            Ok(Arc::new(NoFilter))
        } else {
            Ok(pre_filter)
        }
    }

    fn use_query_residual(
        storage: &IvfQuantizationStorage<Q>,
        distance_type: DistanceType,
    ) -> bool {
        if Q::quantization_type() == QuantizationType::Rabit
            && let Ok(Quantizer::Rabit(rq)) = storage.quantizer()
        {
            return rq.metadata_ref().query_estimator == RabitQueryEstimator::ResidualQuery;
        }
        Q::use_residual(distance_type)
    }

    fn build_rq_search_cache(
        ivf: &IvfModel,
        storage: &IvfQuantizationStorage<Q>,
    ) -> Result<Option<Arc<RabitSearchCache>>> {
        if Q::quantization_type() != QuantizationType::Rabit {
            return Ok(None);
        }
        let Quantizer::Rabit(rq) = storage.quantizer()? else {
            return Ok(None);
        };
        if rq.metadata_ref().query_estimator != RabitQueryEstimator::RawQuery {
            return Ok(None);
        }
        let centroids = ivf
            .centroids_array()
            .ok_or_else(|| Error::index("IVF_RQ raw-query search requires centroids"))?;
        let rotated_centroids = rq.rotate_fsl_to_f32(centroids)?;
        Ok(Some(Arc::new(RabitSearchCache {
            rotated_centroids,
            code_dim: rq.code_dim(),
        })))
    }

    fn rq_search_cache_from_state(
        state: &IvfIndexState<Q>,
        storage: &IvfQuantizationStorage<Q>,
    ) -> Result<Option<Arc<RabitSearchCache>>> {
        let mut cache = state
            .rq_search_cache
            .lock()
            .map_err(|_| Error::internal("RQ search cache lock was poisoned".to_string()))?;
        if let Some(cache) = cache.as_ref() {
            return Ok(cache.clone());
        }
        let built = Self::build_rq_search_cache(&state.ivf, storage)?;
        *cache = Some(built.clone());
        Ok(built)
    }

    fn prepare_rq_raw_query_context(
        &self,
        query: &ArrayRef,
    ) -> Result<Option<Arc<RabitRawQueryContext>>> {
        if Q::quantization_type() != QuantizationType::Rabit || self.use_query_residual {
            return Ok(None);
        }
        let Quantizer::Rabit(rq) = self.storage.quantizer()? else {
            return Ok(None);
        };
        if rq.metadata_ref().query_estimator != RabitQueryEstimator::RawQuery {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            rq.metadata_ref()
                .prepare_raw_query_context(query.as_ref())?,
        )))
    }

    async fn prepare_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
        raw_query_context: Option<Arc<RabitRawQueryContext>>,
    ) -> Result<PreparedPartitionSearch<S, Q>> {
        let (part_entry, ()) = tokio::try_join!(
            self.load_partition(partition_id, true, metrics),
            pre_filter.wait_for_ready(),
        )?;
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &part_entry, pre_filter)
                .await?;
        Ok(PreparedPartitionSearch {
            query: query.clone(),
            pre_filter,
            partition_id,
            partition_rows: self.storage.partition_size(partition_id),
            partition_centroid: self.ivf.centroid(partition_id),
            rq_search_cache: self.rq_search_cache.clone(),
            raw_query_context,
            part_entry,
            _marker: PhantomData,
        })
    }

    async fn prepare_partition_without_prefilter_wait(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
        raw_query_context: Option<Arc<RabitRawQueryContext>>,
    ) -> Result<PreparedPartitionSearch<S, Q>> {
        let part_entry = self.load_partition(partition_id, true, metrics).await?;
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &part_entry, pre_filter)
                .await?;
        Ok(PreparedPartitionSearch {
            query: query.clone(),
            pre_filter,
            partition_id,
            partition_rows: self.storage.partition_size(partition_id),
            partition_centroid: self.ivf.centroid(partition_id),
            rq_search_cache: self.rq_search_cache.clone(),
            raw_query_context,
            part_entry,
            _marker: PhantomData,
        })
    }

    /// The CPU half of a prepared partition search: score the partition and locate its
    /// survivors' covering rows. Reading those rows is I/O and happens in the caller's
    /// async context (see [`IVFIndex::append_covering`]) -- this runs on the CPU pool,
    /// where awaiting is not an option.
    fn run_prepared_partition_search(
        use_query_residual: bool,
        use_residual_scratch: bool,
        prepared: PreparedPartitionSearch<S, Q>,
        want_covering: bool,
        metrics: &dyn MetricsCollector,
        scratch: &mut QueryScratch,
    ) -> Result<(RecordBatch, CoveringGather)> {
        let PreparedPartitionSearch {
            query,
            pre_filter,
            partition_id,
            partition_rows,
            partition_centroid,
            rq_search_cache,
            raw_query_context,
            part_entry,
            _marker: _,
        } = prepared;
        let rotated_partition_centroid =
            rotated_partition_centroid_slice(rq_search_cache.as_deref(), partition_id);
        let residual = Self::query_context_for_scratch(
            use_query_residual,
            use_residual_scratch,
            partition_id,
            partition_centroid.as_ref(),
            rotated_partition_centroid,
            raw_query_context.as_deref(),
        )?;
        let query = Self::preprocess_partition_query_owned(
            use_query_residual,
            use_residual_scratch,
            partition_id,
            partition_centroid.as_ref(),
            query,
        )?;
        let param = (&query).into();
        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let k = query.k * refine_factor;
        let batch = part_entry.index.search_with_scratch(
            query.key,
            k,
            param,
            &part_entry.storage,
            pre_filter,
            metrics,
            residual,
            scratch,
        )?;
        let gather = match want_covering {
            true => Self::survivor_positions(&batch, &part_entry.storage, partition_rows)?,
            false => CoveringGather::NotNeeded,
        };
        Ok((batch, gather))
    }

    /// Locate the rows of `batch` (a `[_distance, _rowid]` partition result) within the
    /// partition's row range in the storage file.
    ///
    /// The scan walks the partition's row ids once against the (`<= k`)-sized survivor set
    /// with an early exit, rather than building a partition-sized side map per probe.
    /// Positions come out ascending and deduplicated because each storage row is visited
    /// at most once, which is what the reader's take path requires.
    fn survivor_positions(
        batch: &RecordBatch,
        storage: &Q::Storage,
        partition_rows: usize,
    ) -> Result<CoveringGather> {
        // A deferred fragment-reuse remap filters rows out of the partition as it is
        // loaded, so from that point a storage position is no longer the file position it
        // was read from and cannot address the file.
        if storage.len() != partition_rows {
            return Ok(CoveringGather::WholeRange);
        }
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("search result missing row id".to_string()))?
            .as_primitive::<arrow_array::types::UInt64Type>();
        let mut needed: HashSet<u64> = row_ids.values().iter().copied().collect();
        let mut positions = Vec::with_capacity(needed.len());
        Self::locate_survivors(storage, &mut needed, |_, position| positions.push(position));
        Ok(CoveringGather::Positions(positions))
    }

    /// Scan a partition's row ids once against the (`<= k`)-sized `needed` set with an
    /// early exit, calling `on_hit(row_id, storage_position)` for each located row.
    ///
    /// The one shared core of [`Self::survivor_positions`] (per-partition search) and
    /// the locator inside `accumulate_prepared_partition_search` (global-heap search).
    /// The two callers differ in how they react to a storage/partition misalignment --
    /// early whole-range fallback vs. scan-anyway with `position: None` -- but the scan
    /// itself must stay identical: a future change to when positions are valid (e.g. a
    /// new remap variant) goes through here, so the two paths cannot silently diverge
    /// on the deferred-remap case. Positions come out ascending and deduplicated
    /// because each storage row is visited at most once.
    fn locate_survivors(
        storage: &Q::Storage,
        needed: &mut HashSet<u64>,
        mut on_hit: impl FnMut(u64, u32),
    ) {
        for (position, row_id) in storage.row_ids().enumerate() {
            if needed.remove(row_id) {
                on_hit(*row_id, position as u32);
                if needed.is_empty() {
                    break;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_prepared_partition_search(
        use_query_residual: bool,
        use_residual_scratch: bool,
        prepared: PreparedPartitionSearch<S, Q>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        covering_locations: &mut HashMap<u64, CoveringLocation>,
        want_covering: bool,
        scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        let PreparedPartitionSearch {
            query,
            pre_filter,
            partition_id,
            partition_rows,
            partition_centroid,
            rq_search_cache,
            raw_query_context,
            part_entry,
            _marker: _,
        } = prepared;
        let rotated_partition_centroid =
            rotated_partition_centroid_slice(rq_search_cache.as_deref(), partition_id);
        let residual = Self::query_context_for_scratch(
            use_query_residual,
            use_residual_scratch,
            partition_id,
            partition_centroid.as_ref(),
            rotated_partition_centroid,
            raw_query_context.as_deref(),
        )?;
        let query = Self::preprocess_partition_query_owned(
            use_query_residual,
            use_residual_scratch,
            partition_id,
            partition_centroid.as_ref(),
            query,
        )?;
        let param = (&query).into();
        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let k = query.k * refine_factor;
        // Accumulate this partition's contribution to the top-k heap first, so the heap
        // membership consulted below already reflects it.
        part_entry.index.accumulate_topk_with_scratch(
            query.key,
            k,
            param,
            &part_entry.storage,
            pre_filter,
            heap,
            residual,
            scratch,
            metrics,
        )?;
        // Keep covering O(k): record only where the current heap survivors' covering rows
        // live, then prune the map to the heap's membership so rows this partition evicted
        // drop their entry. Nothing is read here -- the read happens once, after the heap
        // has settled, for the survivors that actually made it (see
        // `gather_survivor_covering`); buffering every probed partition's covering to serve
        // k winners is exactly the O(nprobe * partition_size) cost this phase removes.
        // Every heap survivor is located: a row is a survivor only if it was in the top-k
        // when its own partition was processed (the heap never re-admits an evicted row),
        // so it was located then -- or it is new from this partition.
        // The lookup scans the partition's row ids against the (<= k)-sized needed set
        // with an early exit, instead of building a partition-sized side map per probe.
        if want_covering {
            let heap_ids: HashSet<u64> = heap.iter().map(|node| node.id).collect();
            let mut needed: HashSet<u64> = heap_ids
                .iter()
                .filter(|id| !covering_locations.contains_key(id))
                .copied()
                .collect();
            if !needed.is_empty() {
                // A deferred fragment-reuse remap filters rows out of the partition as it
                // is loaded, so from that point a storage position no longer addresses the
                // file and this partition must be re-read as a whole range.
                let aligned = part_entry.storage.len() == partition_rows;
                Self::locate_survivors(&part_entry.storage, &mut needed, |row_id, position| {
                    covering_locations.insert(
                        row_id,
                        CoveringLocation {
                            partition_id,
                            position: aligned.then_some(position),
                        },
                    );
                });
            }
            covering_locations.retain(|id, _| heap_ids.contains(id));
            // Invariant: the map locates exactly the heap's distinct row ids -- never more
            // (that would break the O(k) bound) and never fewer (a survivor would be
            // missing its covering at emit time).
            debug_assert_eq!(
                covering_locations.len(),
                heap_ids.len(),
                "covering locations must track exactly the heap's survivors (O(k))"
            );
        }
        Ok(())
    }

    fn query_context_for_scratch<'a>(
        use_query_residual: bool,
        use_residual_scratch: bool,
        partition_id: usize,
        partition_centroid: Option<&'a ArrayRef>,
        rotated_partition_centroid: Option<&'a [f32]>,
        raw_query_context: Option<&'a RabitRawQueryContext>,
    ) -> Result<Option<QueryResidual<'a>>> {
        if use_residual_scratch {
            let partition_centroid = partition_centroid.ok_or_else(|| {
                Error::index(format!("partition centroid {partition_id} does not exist"))
            })?;
            Ok(Some(QueryResidual::Centroid(partition_centroid.as_ref())))
        } else if !use_query_residual
            && (rotated_partition_centroid.is_some() || raw_query_context.is_some())
        {
            Ok(Some(QueryResidual::RabitRawQuery {
                rotated_centroid: rotated_partition_centroid,
                query: raw_query_context,
            }))
        } else {
            Ok(None)
        }
    }

    fn global_heap_to_batch(
        heap: BinaryHeap<OrderedNode<u64>>,
        // `[_rowid, <included cols...>]` covering the heap survivors and nothing else,
        // gathered after the heap settled by `gather_survivor_covering`. `None` when the
        // gather was not run (no survivor was located).
        covering: Option<&RecordBatch>,
        // `[_rowid, <included cols...>]` schema for the covering columns this query needs,
        // or None when the index has none or the query needs none. This is a per-query
        // constant, so it -- not whether any covering was gathered -- decides whether to
        // emit the wider covered schema. That keeps the emitted schema equal to the schema
        // the exec declares even when zero partitions were searched (heap empty).
        covering_schema: Option<&arrow_schema::Schema>,
    ) -> Result<RecordBatch> {
        let (row_ids, dists): (Vec<_>, Vec<_>) = heap.into_iter().map(|r| (r.id, r.dist.0)).unzip();
        let dist_arr: ArrayRef = Arc::new(Float32Array::from(dists));
        let row_id_arr: ArrayRef = Arc::new(UInt64Array::from(row_ids));
        let Some(covering_schema) = covering_schema else {
            // Ordinary index: `[_distance, _rowid]`.
            return Ok(RecordBatch::try_new(
                VECTOR_RESULT_SCHEMA.clone(),
                vec![dist_arr, row_id_arr],
            )?);
        };
        // Covered index: emit `[_distance, _rowid, <included cols...>]`.
        let mut fields: Vec<arrow_schema::FieldRef> = VECTOR_RESULT_SCHEMA.fields().to_vec();
        let mut columns: Vec<ArrayRef> = vec![dist_arr, row_id_arr.clone()];
        if row_id_arr.is_empty() {
            // No survivors (heap empty / zero partitions searched). Emit the covering
            // columns as empty arrays so the schema still matches the declared covered
            // schema.
            for field in covering_schema.fields() {
                if field.name() == ROW_ID {
                    continue;
                }
                fields.push(field.clone());
                columns.push(arrow_array::new_null_array(field.data_type(), 0));
            }
        } else {
            // Survivors but nothing gathered for them. Reported rather than filled with
            // nulls: a null fill is indistinguishable from genuine nulls once a covering
            // column is nullable, so the query would return the right rows with silently
            // wrong values.
            let covering = covering.ok_or_else(|| {
                Error::index(format!(
                    "index declares covering columns {:?} but none were gathered for the \
                     {} result rows",
                    covering_schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .filter(|name| *name != ROW_ID)
                        .collect::<Vec<_>>(),
                    row_id_arr.len(),
                ))
            })?;
            // Align the gathered values to the final row ids by row id, never by
            // position: the gather returns each partition's rows in file order, which is
            // not the heap's order. A survivor the gather did not return is an error
            // there, so this cannot silently drop a row.
            let row_id_u64 = row_id_arr.as_primitive::<arrow_array::types::UInt64Type>();
            let included = gather_covering_columns_by_row_id(covering, row_id_u64)?;
            for (field, array) in included {
                fields.push(field);
                columns.push(array);
            }
        }
        Ok(RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(fields)),
            columns,
        )?)
    }

    /// The schema search results carry: `[_distance, _rowid]` plus the covering
    /// ("included") columns of `covering_schema`, in storage order. Used to declare
    /// stream schemas that stay consistent with the (possibly widened) batches.
    fn covered_result_schema(
        covering_schema: Option<&arrow_schema::Schema>,
    ) -> arrow_schema::SchemaRef {
        let Some(covering_schema) = covering_schema else {
            return VECTOR_RESULT_SCHEMA.clone();
        };
        let mut fields: Vec<arrow_schema::FieldRef> = VECTOR_RESULT_SCHEMA.fields().to_vec();
        for field in covering_schema.fields() {
            if field.name() == ROW_ID {
                continue;
            }
            fields.push(field.clone());
        }
        Arc::new(arrow_schema::Schema::new(fields))
    }

    /// The covering columns `query` needs from this index, or `None` when there are none
    /// to emit.
    ///
    /// `None` covers two cases that are the same instruction to the search path -- do no
    /// covering work at all: an ordinary index, and a covered index whose covering
    /// projection this query narrows to nothing. See [`Query::covering_projection`] for
    /// why the second must not degrade into "read everything and project it away".
    fn query_covering(&self, query: &Query) -> Result<Option<QueryCovering>> {
        let columns = self
            .storage
            .covering_columns(query.covering_projection.as_deref())?;
        if columns.is_empty() {
            return Ok(None);
        }
        Ok(Some(QueryCovering {
            schema: self.storage.covering_read_schema(&columns)?,
            columns,
        }))
    }

    /// Widen a `[_distance, _rowid]` partition result with its survivors' covering values,
    /// read from the storage file by position.
    ///
    /// This is the per-partition emit site. It runs in the caller's async context, after
    /// the CPU phase has settled which rows survived, so the read is proportional to `k`
    /// rather than to the partition size -- and it is deliberately not cached, being a
    /// different set of rows for every query.
    async fn append_covering(
        &self,
        partition_id: usize,
        batch: RecordBatch,
        gather: CoveringGather,
        covering: Option<&QueryCovering>,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        let positions = match &gather {
            CoveringGather::NotNeeded => return Ok(batch),
            CoveringGather::Positions(positions) => Some(positions.as_slice()),
            CoveringGather::WholeRange => None,
        };
        let Some(covering) = covering else {
            return Ok(batch);
        };
        let row_ids = batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("search result missing row id".to_string()))?
            .as_primitive::<arrow_array::types::UInt64Type>();
        let mut fields: Vec<arrow_schema::FieldRef> = batch.schema().fields().to_vec();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        if row_ids.is_empty() {
            // Nothing survived this partition, so there is nothing to read. Still emit the
            // covering columns (empty) so every batch on the stream carries the schema the
            // exec declared.
            for field in covering.schema.fields() {
                if field.name() == ROW_ID {
                    continue;
                }
                columns.push(arrow_array::new_null_array(field.data_type(), 0));
                fields.push(field.clone());
            }
        } else {
            let gathered = self
                .storage
                .take_covering(partition_id, positions, &covering.columns, io_stats)
                .await?;
            // By row id, never by position: the gather returns the partition's rows in
            // file order while the search result is in distance order. A survivor the
            // gather did not return is an error there rather than a silent null.
            for (field, array) in gather_covering_columns_by_row_id(&gathered, row_ids)? {
                fields.push(field);
                columns.push(array);
            }
        }
        Ok(RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(fields)),
            columns,
        )?)
    }

    /// Read `[_rowid, <included...>]` for the heap's survivors: one bounded read per
    /// partition that still owns one, after the heap has settled.
    ///
    /// Each partition's covering is narrowed to its own survivors before the next
    /// partition is read, so what the loop accumulates is `O(survivors)` and only one
    /// partition's covering is ever alive at a time. That matters on the whole-range
    /// fallback (a pending fragment-reuse remap puts every partition on it): keeping each
    /// partition's whole covering to concatenate at the end would make peak memory
    /// `partitions x partition size`, per concurrent query and outside the shared,
    /// evictable partition cache -- the opposite of what bounding this read is for.
    ///
    /// Returns `None` when no survivor was located, which for a covered query means the
    /// heap is empty -- [`Self::global_heap_to_batch`] turns "survivors but nothing
    /// gathered" into an error rather than a null fill.
    async fn gather_survivor_covering(
        &self,
        locations: &HashMap<u64, CoveringLocation>,
        covering: &QueryCovering,
        io_stats: Option<IoStats>,
    ) -> Result<Option<RecordBatch>> {
        // Group by partition so each contributing partition is read once, not once per
        // survivor. `None` positions mark a partition whose positions are not derivable.
        let mut by_partition: HashMap<usize, (Option<Vec<u32>>, Vec<u64>)> = HashMap::new();
        for (row_id, location) in locations {
            let (positions, row_ids) = by_partition
                .entry(location.partition_id)
                .or_insert_with(|| (Some(Vec::new()), Vec::new()));
            row_ids.push(*row_id);
            match (positions.as_mut(), location.position) {
                (Some(positions), Some(position)) => positions.push(position),
                _ => *positions = None,
            }
        }
        let mut batches = Vec::with_capacity(by_partition.len());
        for (partition_id, (mut positions, row_ids)) in by_partition {
            if let Some(positions) = positions.as_mut() {
                // The reader's take path requires strictly ascending indices; survivors
                // arrive in heap order, which is neither sorted nor partition-local.
                positions.sort_unstable();
                positions.dedup();
            }
            let gathered = self
                .storage
                .take_covering(
                    partition_id,
                    positions.as_deref(),
                    &covering.columns,
                    io_stats.clone(),
                )
                .await?;
            // By row id, never by position: on the fallback `gathered` is the partition's
            // whole range in file order, and even on the scattered read the caller's
            // survivors are in heap order. A survivor the read did not return is an error
            // here rather than a row quietly dropped from the result.
            let survivors = UInt64Array::from(row_ids);
            let take_idx = row_id_take_indices(&gathered, &survivors)?;
            batches.push(take_record_batch(&gathered, &take_idx)?);
        }
        if batches.is_empty() {
            return Ok(None);
        }
        Ok(Some(concat_batches(&covering.schema, batches.iter())?))
    }

    fn preprocess_partition_query(
        use_query_residual: bool,
        use_residual_scratch: bool,
        partition_id: usize,
        partition_centroid: Option<&ArrayRef>,
        query: &Query,
    ) -> Result<Query> {
        Self::preprocess_partition_query_owned(
            use_query_residual,
            use_residual_scratch,
            partition_id,
            partition_centroid,
            query.clone(),
        )
    }

    fn preprocess_partition_query_owned(
        use_query_residual: bool,
        use_residual_scratch: bool,
        partition_id: usize,
        partition_centroid: Option<&ArrayRef>,
        mut query: Query,
    ) -> Result<Query> {
        if use_query_residual {
            let partition_centroid = partition_centroid.ok_or_else(|| {
                Error::index(format!("partition centroid {partition_id} does not exist"))
            })?;
            if use_residual_scratch {
                return Ok(query);
            }
            let residual_key = sub(&query.key, partition_centroid)?;
            query.key = residual_key;
        }
        Ok(query)
    }

    fn query_scratch_capacity(
        ivf: &IvfModel,
        storage: &IvfQuantizationStorage<Q>,
    ) -> QueryScratchCapacity {
        if Q::quantization_type() != QuantizationType::Rabit {
            return QueryScratchCapacity::default();
        }

        let dim = ivf.dimension();
        let max_partition_len = ivf.lengths.iter().copied().max().unwrap_or_default() as usize;
        let num_bits = match storage.quantizer() {
            Ok(Quantizer::Rabit(rq)) => rq.metadata_ref().num_bits,
            _ => 9,
        };

        rabit_query_scratch_capacity(dim, max_partition_len, num_bits)
    }

    fn use_residual_scratch(ivf: &IvfModel, use_query_residual: bool) -> bool {
        Q::quantization_type() == QuantizationType::Rabit
            && use_query_residual
            && ivf
                .centroids_array()
                .map(|centroids| centroids.value_type() == DataType::Float32)
                .unwrap_or(false)
    }

    fn query_scratch_pool(ivf: &IvfModel, storage: &IvfQuantizationStorage<Q>) -> QueryScratchPool {
        QueryScratchPool::with_capacity(
            get_num_compute_intensive_cpus(),
            Self::query_scratch_capacity(ivf, storage),
        )
    }

    /// Create a new IVF index.
    pub(crate) async fn try_new(
        object_store: Arc<ObjectStore>,
        index_dir: Path,
        uuid: Uuid,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
        file_metadata_cache: &LanceCache,
        index_cache: LanceCache,
        file_sizes: HashMap<String, u64>,
    ) -> Result<Self> {
        let io_parallelism = object_store.io_parallelism();
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

        let uuid_str = uuid.to_string();
        let uri = index_dir
            .clone()
            .join(uuid_str.as_str())
            .join(INDEX_FILE_NAME);
        let cached_size = file_sizes
            .get(INDEX_FILE_NAME)
            .map(|&size| CachedFileSize::new(size))
            .unwrap_or_else(CachedFileSize::unknown);
        let index_reader = FileReader::try_open(
            scheduler.open_file(&uri, &cached_size).await?,
            None,
            Arc::<DecoderPlugins>::default(),
            file_metadata_cache,
            FileReaderOptions::default(),
        )
        .await?;
        let index_metadata: IndexMetadata = serde_json::from_str(
            index_reader
                .schema()
                .metadata
                .get(INDEX_METADATA_SCHEMA_KEY)
                .ok_or(Error::index(format!("{} not found", DISTANCE_TYPE_KEY)))?
                .as_str(),
        )?;
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let ivf_pos = index_reader
            .schema()
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::index(format!("{} not found", IVF_METADATA_KEY)))?
            .parse()
            .map_err(|e| Error::index(format!("Failed to decode IVF position: {}", e)))?;
        let ivf_pb_bytes = index_reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_pb_bytes)?)?;

        let sub_index_metadata = index_reader
            .schema()
            .metadata
            .get(S::metadata_key())
            .ok_or(Error::index(format!("{} not found", S::metadata_key())))?;
        let sub_index_metadata: Vec<String> = serde_json::from_str(sub_index_metadata)?;

        let aux_cached_size = file_sizes
            .get(INDEX_AUXILIARY_FILE_NAME)
            .map(|&size| CachedFileSize::new(size))
            .unwrap_or_else(CachedFileSize::unknown);
        let storage_reader = FileReader::try_open(
            scheduler
                .open_file(
                    &index_dir
                        .clone()
                        .join(uuid_str.as_str())
                        .join(INDEX_AUXILIARY_FILE_NAME),
                    &aux_cached_size,
                )
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            file_metadata_cache,
            FileReaderOptions::default(),
        )
        .await?;
        let frag_reuse_index = frag_reuse_index
            .clone()
            .map(|index| Arc::new(CompactFragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
        let storage =
            IvfQuantizationStorage::try_new_with_remapper(storage_reader, frag_reuse_index).await?;

        // Cache file metadata so reconstructions from IvfIndexState can skip
        // footer reads.
        file_metadata_cache
            .with_key_prefix(uri.as_ref())
            .insert_with_key(&FileMetadataCacheKey, index_reader.metadata().clone())
            .await;
        let aux_path = index_dir
            .clone()
            .join(uuid_str.as_str())
            .join(INDEX_AUXILIARY_FILE_NAME);
        file_metadata_cache
            .with_key_prefix(aux_path.as_ref())
            .insert_with_key(&FileMetadataCacheKey, storage.reader().metadata().clone())
            .await;

        let scratch_pool = Arc::new(Self::query_scratch_pool(&ivf, &storage));
        let use_query_residual = Self::use_query_residual(&storage, distance_type);
        let use_residual_scratch = Self::use_residual_scratch(&ivf, use_query_residual);
        let rq_search_cache = Self::build_rq_search_cache(&ivf, &storage)?;

        // The scheduler is freshly created above and, at this point, has served
        // only the open-time reads (file footers, IVF centroids, quantization
        // metadata) -- partition reads happen later, during queries.  So its
        // cumulative stats are exactly the one-time index-open I/O.
        let open_io_stats = scheduler.stats();

        let read_projection = Self::read_projection(&index_reader)?;
        Ok(Self {
            uri: to_local_path(&uri),
            index_path: uri.as_ref().to_string(),
            uuid,
            scratch_pool,
            use_query_residual,
            use_residual_scratch,
            rq_search_cache,
            ivf,
            reader: index_reader,
            read_projection,
            storage,
            sub_index_metadata,
            distance_type,
            index_cache: WeakLanceCache::from(&index_cache),
            io_parallelism,
            open_io_stats,
            _marker: PhantomData,
        })
    }

    /// Reconstruct an IVFIndex from pre-parsed state without any I/O.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_cached_state(
        uri: String,
        index_path: String,
        uuid: Uuid,
        ivf: IvfModel,
        reader: FileReader,
        storage: IvfQuantizationStorage<Q>,
        sub_index_metadata: Vec<String>,
        distance_type: DistanceType,
        index_cache: LanceCache,
        io_parallelism: usize,
        rq_search_cache: Option<Arc<RabitSearchCache>>,
    ) -> Result<Self> {
        let scratch_pool = Arc::new(Self::query_scratch_pool(&ivf, &storage));
        let use_query_residual = Self::use_query_residual(&storage, distance_type);
        let use_residual_scratch = Self::use_residual_scratch(&ivf, use_query_residual);
        let read_projection = Self::read_projection(&reader)?;
        Ok(Self {
            uri,
            index_path,
            uuid,
            scratch_pool,
            use_query_residual,
            use_residual_scratch,
            rq_search_cache,
            ivf,
            reader,
            read_projection,
            storage,
            sub_index_metadata,
            distance_type,
            index_cache: WeakLanceCache::from(&index_cache),
            io_parallelism,
            // Reconstruction from cached state re-opens readers on its own path;
            // the open-time I/O is not attributed here (it is a one-time cost,
            // and the first open via `try_new` already accounts for it).
            open_io_stats: ScanStats::default(),
            _marker: PhantomData,
        })
    }

    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::index(format!(
                "partition id {} is out of range of {} partitions",
                partition_id,
                self.ivf.num_partitions()
            )));
        }

        let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);

        if write_cache {
            let result = self
                .index_cache
                .get_or_insert_with_key_hit(cache_key, || async {
                    info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
                    metrics.record_part_load();
                    // Internal columns only: this entry is cached under a partition id
                    // alone, so it must not depend on which covering columns the loading
                    // query wanted.
                    self.load_partition_entry(
                        partition_id,
                        PartitionColumns::Internal,
                        metrics.io_stats(),
                    )
                    .await
                })
                .await;
            match &result {
                Ok((_, true)) => metrics.record_index_cache_hit(),
                _ => metrics.record_index_cache_miss(),
            }
            let (entry, _) = result?;
            Ok(entry)
        } else {
            if let Some(part_idx) = self.index_cache.get_with_key(&cache_key).await {
                metrics.record_index_cache_hit();
                return Ok(part_idx);
            }
            metrics.record_index_cache_miss();
            info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
            metrics.record_part_load();
            Ok(Arc::new(
                self.load_partition_entry(
                    partition_id,
                    PartitionColumns::Internal,
                    metrics.io_stats(),
                )
                .await?,
            ))
        }
    }

    /// Load a partition entry carrying every column its storage file holds, bypassing
    /// the partition cache in both directions.
    ///
    /// Rewrite paths need this: the entry they load is written straight back out, so a
    /// covering column left unread is a covering column the rewritten index no longer
    /// has. The cache is bypassed rather than consulted because its key
    /// ([`IVFPartitionKey`]) has no covering component -- a hit would hand back the
    /// codes-only entry the search path stores, and an insert would hand a covering-laden
    /// entry to the search path.
    pub(crate) async fn load_partition_entry_with_covering(
        &self,
        partition_id: usize,
    ) -> Result<PartitionEntry<S, Q>> {
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::index(format!(
                "partition id {} is out of range of {} partitions",
                partition_id,
                self.ivf.num_partitions()
            )));
        }
        self.load_partition_entry(partition_id, PartitionColumns::All, None)
            .await
    }

    async fn load_partition_entry(
        &self,
        partition_id: usize,
        columns: PartitionColumns,
        io_stats: Option<IoStats>,
    ) -> Result<PartitionEntry<S, Q>> {
        // `concat_batches` indexes the batches by this schema's field positions
        // without comparing the two, so the schema has to describe exactly what
        // was read: the full file schema over a projected read would index past
        // the last column.
        let schema = Arc::new(match &self.read_projection {
            Some(projection) => projection.schema.as_ref().into(),
            None => self.reader.schema().as_ref().into(),
        });
        let batch = match self.reader.metadata().num_rows {
            0 => RecordBatch::new_empty(schema),
            _ => {
                let row_range = self.ivf.row_range(partition_id);
                if row_range.is_empty() {
                    RecordBatch::new_empty(schema)
                } else {
                    // When I/O is being measured, read through a reader whose
                    // scheduler also records into the per-query sink (a cheap
                    // clone sharing all cached metadata, no file re-open).
                    // Otherwise borrow the shared reader as-is, with no clone.
                    let reader = match &io_stats {
                        Some(io_stats) => {
                            Cow::Owned(self.reader.with_io_stats(io_stats.recorder()))
                        }
                        None => Cow::Borrowed(&self.reader),
                    };
                    let params = ReadBatchParams::Range(row_range);
                    let stream = match &self.read_projection {
                        Some(projection) => {
                            reader
                                .read_stream_projected(
                                    params,
                                    u32::MAX,
                                    1,
                                    projection.clone(),
                                    FilterExpression::no_filter(),
                                )
                                .await?
                        }
                        None => {
                            reader
                                .read_stream(params, u32::MAX, 1, FilterExpression::no_filter())
                                .await?
                        }
                    };
                    let batches = stream.try_collect::<Vec<_>>().await?;
                    concat_batches(&schema, batches.iter())?
                }
            }
        };
        let batch = batch.add_metadata(
            S::metadata_key().to_owned(),
            self.sub_index_metadata[partition_id].clone(),
        )?;
        let idx = S::load(batch)?;
        let storage = self
            .load_partition_storage(partition_id, columns, io_stats)
            .await?;
        Ok(PartitionEntry::new(idx, storage))
    }

    pub async fn load_partition_storage(
        &self,
        partition_id: usize,
        columns: PartitionColumns,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        self.storage
            .load_partition(partition_id, columns, io_stats)
            .await
    }

    /// Names of this index's covering ("included") columns, in storage order, or
    /// empty if the index has none. Used by remap/rebuild paths to re-project the
    /// covering columns so they survive into the rewritten storage.
    pub(crate) fn covering_column_names(&self) -> Result<Vec<String>> {
        Ok(self
            .storage
            .covering_schema()?
            .map(|schema| {
                schema
                    .fields()
                    .iter()
                    .filter(|f| f.name() != ROW_ID)
                    .map(|f| f.name().to_string())
                    .collect()
            })
            .unwrap_or_default())
    }

    /// preprocess the query vector given the partition id.
    ///
    /// Internal API with no stability guarantees.
    #[instrument(level = "debug", skip(self))]
    pub fn preprocess_query(&self, partition_id: usize, query: &Query) -> Result<Query> {
        Self::preprocess_partition_query(
            self.use_query_residual,
            self.use_residual_scratch,
            partition_id,
            self.ivf.centroid(partition_id).as_ref(),
            query,
        )
    }

    /// Export the index state needed for reconstruction from a disk cache.
    pub(crate) fn to_state_entry(&self) -> IvfStateEntryBox {
        let (sub_index_type, quantization_type) = self.sub_index_type();
        IvfStateEntryBox(Arc::new(IvfIndexState::<Q> {
            index_file_path: self.index_path.clone(),
            uuid: self.uuid.to_string(),
            ivf: self.ivf.clone(),
            aux_ivf: self.storage.ivf().clone(),
            distance_type: self.distance_type,
            sub_index_metadata: self.sub_index_metadata.clone(),
            metadata: self.storage.metadata().clone(),
            sub_index_type,
            quantization_type,
            index_file_size: self.reader.metadata().file_size(),
            aux_file_size: self.storage.reader().metadata().file_size(),
            rq_search_cache: rabit_search_cache_cell(self.rq_search_cache.clone()),
        }))
    }
}

#[async_trait]
impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> Index for IVFIndex<S, Q> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        futures::stream::iter(0..self.ivf.num_partitions())
            .map(Ok)
            .try_for_each_concurrent(Some(self.io_parallelism), |part_id| {
                self.load_partition(part_id, true, &NoOpMetricsCollector)
                    .map_ok(|_| ())
            })
            .await
    }

    fn index_type(&self) -> IndexType {
        match self.sub_index_type() {
            (SubIndexType::Flat, QuantizationType::Flat)
            | (SubIndexType::Flat, QuantizationType::FlatBin) => IndexType::IvfFlat,
            (SubIndexType::Flat, QuantizationType::Product) => IndexType::IvfPq,
            (SubIndexType::Flat, QuantizationType::Scalar) => IndexType::IvfSq,
            (SubIndexType::Flat, QuantizationType::Rabit) => IndexType::IvfRq,
            (SubIndexType::Hnsw, QuantizationType::Product) => IndexType::IvfHnswPq,
            (SubIndexType::Hnsw, QuantizationType::Scalar) => IndexType::IvfHnswSq,
            (SubIndexType::Hnsw, QuantizationType::Flat)
            | (SubIndexType::Hnsw, QuantizationType::FlatBin) => IndexType::IvfHnswFlat,
            (sub_index_type, quantization_type) => {
                unimplemented!(
                    "unsupported index type: {}, {}",
                    sub_index_type,
                    quantization_type
                )
            }
        }
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = (0..self.ivf.num_partitions())
            .map(|part_id| IvfIndexPartitionStatistics {
                size: self.storage.partition_size(part_id) as u32,
            })
            .collect::<Vec<_>>();

        let centroid_vecs = maybe_centroids_for_stats(self.ivf.centroids.as_ref().unwrap())?;

        let (sub_index_type, quantization_type) = self.sub_index_type();
        let index_type = index_type_string(sub_index_type, quantization_type);
        let mut sub_index_stats: serde_json::Map<String, serde_json::Value> =
            if let Some(metadata) = self.sub_index_metadata.iter().find(|m| !m.is_empty()) {
                serde_json::from_str(metadata)?
            } else {
                serde_json::map::Map::new()
            };
        let mut store_stats = serde_json::to_value(self.storage.metadata())?;
        let store_stats = store_stats.as_object_mut().ok_or(Error::internal(
            "failed to get storage metadata".to_string(),
        ))?;

        sub_index_stats.append(store_stats);
        if S::name() == "FLAT" {
            let qt_label = match Q::quantization_type() {
                // FlatBin is the Hamming variant of Flat; report as "FLAT".
                QuantizationType::FlatBin => "FLAT".to_string(),
                other => other.to_string(),
            };
            sub_index_stats.insert("index_type".to_string(), qt_label.into());
        } else {
            sub_index_stats.insert("index_type".to_string(), S::name().into());
        }

        let sub_index_distance_type = if matches!(Q::quantization_type(), QuantizationType::Product)
            && self.distance_type == DistanceType::Cosine
        {
            DistanceType::L2
        } else {
            self.distance_type
        };
        sub_index_stats.insert(
            "metric_type".to_string(),
            sub_index_distance_type.to_string().into(),
        );

        // we need to drop some stats from the metadata
        sub_index_stats.remove("codebook_position");
        sub_index_stats.remove("codebook");
        sub_index_stats.remove("codebook_tensor");

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type,
            uuid: self.uuid.to_string(),
            uri: self.uri.clone(),
            metric_type: self.distance_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: serde_json::Value::Object(sub_index_stats),
            partitions: partitions_statistics,
            centroids: centroid_vecs,
            loss: self.ivf.loss(),
            index_file_version: IndexFileVersion::V3,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!(
            "this method is only needed for migrating older manifests, not for this new index"
        )
    }
}

#[async_trait]
impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> VectorIndex for IVFIndex<S, Q> {
    async fn search(
        &self,
        _query: &Query,
        _pre_filter: Arc<dyn PreFilter>,
        _metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        unimplemented!(
            "IVFIndex not currently used as sub-index and top-level indices do partition-aware search"
        )
    }

    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
        let dt = if self.distance_type == DistanceType::Cosine {
            DistanceType::L2
        } else {
            self.distance_type
        };

        let max_nprobes = query.maximum_nprobes.unwrap_or(self.ivf.num_partitions());

        self.ivf.find_partitions(&query.key, max_nprobes, dt)
    }

    fn total_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    fn physical_covering_fields(&self) -> Result<Vec<(i32, Field)>> {
        Ok(self.storage.physical_covering_fields())
    }

    #[instrument(level = "debug", skip(self, pre_filter, metrics))]
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let part_entry = self.load_partition(partition_id, true, metrics).await?;
        pre_filter.wait_for_ready().await?;
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &part_entry, pre_filter)
                .await?;

        let partition_centroid = self.ivf.centroid(partition_id);
        let rq_search_cache = self.rq_search_cache.clone();
        let raw_query_context = self.prepare_rq_raw_query_context(&query.key)?;
        let query = Self::preprocess_partition_query(
            self.use_query_residual,
            self.use_residual_scratch,
            partition_id,
            partition_centroid.as_ref(),
            query,
        )?;
        let scratch_pool = self.scratch_pool.clone();
        let use_query_residual = self.use_query_residual;
        let use_residual_scratch = self.use_residual_scratch;
        let covering = self.query_covering(&query)?;
        let want_covering = covering.is_some();
        let partition_rows = self.storage.partition_size(partition_id);
        let (batch, gather, local_metrics) = spawn_cpu(move || {
            let param = (&query).into();
            let refine_factor = query.refine_factor.unwrap_or(1) as usize;
            let k = query.k * refine_factor;
            let local_metrics = LocalMetricsCollector::default();
            let rotated_partition_centroid =
                rotated_partition_centroid_slice(rq_search_cache.as_deref(), partition_id);
            let residual = Self::query_context_for_scratch(
                use_query_residual,
                use_residual_scratch,
                partition_id,
                partition_centroid.as_ref(),
                rotated_partition_centroid,
                raw_query_context.as_deref(),
            )?;
            let batch = scratch_pool.with_scratch(|scratch| {
                part_entry.index.search_with_scratch(
                    query.key,
                    k,
                    param,
                    &part_entry.storage,
                    pre_filter,
                    &local_metrics,
                    residual,
                    scratch,
                )
            })?;
            // Locate the survivors' covering rows while the partition is still loaded;
            // reading them is I/O and cannot happen on the CPU pool.
            let gather = match want_covering {
                true => Self::survivor_positions(&batch, &part_entry.storage, partition_rows)?,
                false => CoveringGather::NotNeeded,
            };
            Result::Ok((batch, gather, local_metrics))
        })
        .await?;

        local_metrics.dump_into(metrics);

        self.append_covering(
            partition_id,
            batch,
            gather,
            covering.as_ref(),
            metrics.io_stats(),
        )
        .await
    }

    async fn prepare_partition_search(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<PreparedPartitionSearchHandle> {
        let raw_query_context = self.prepare_rq_raw_query_context(&query.key)?;
        Ok(Box::new(
            self.prepare_partition(partition_id, query, pre_filter, metrics, raw_query_context)
                .await?,
        ))
    }

    /// The synchronous phase of a prepared partition search.
    ///
    /// A covered query cannot be served here: since phase 8 the covering values are read
    /// from the storage file once the survivors are known, and this entry point exists
    /// precisely to run on the CPU pool where that read cannot be awaited. Callers that
    /// need covering columns use [`VectorIndex::search_in_partition`] or
    /// [`VectorIndex::search_partitions`], both of which own an async context; this
    /// reports the mismatch rather than returning rows with their covering columns
    /// silently missing.
    fn search_prepared_partition(
        &self,
        prepared: PreparedPartitionSearchHandle,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let prepared = prepared
            .downcast::<PreparedPartitionSearch<S, Q>>()
            .map_err(|_| Error::internal("failed to downcast prepared partition search"))?;
        if let Some(covering) = self.query_covering(&prepared.query)? {
            return Err(Error::index(format!(
                "search_prepared_partition cannot materialize the covering columns {:?} \
                 this query needs: the covering values are read from index storage after \
                 scoring, and this is the synchronous phase of a prepared search. Use \
                 search_in_partition or search_partitions instead.",
                covering.columns,
            )));
        }
        let (batch, _) = self.scratch_pool.with_scratch(|scratch| {
            Self::run_prepared_partition_search(
                self.use_query_residual,
                self.use_residual_scratch,
                *prepared,
                false,
                metrics,
                scratch,
            )
        })?;
        Ok(batch)
    }

    /// False for a covered index, because [`Self::search_prepared_partition`] rejects the
    /// queries such an index exists to serve (the covering values are read after scoring,
    /// which is I/O, and that method is the synchronous phase).
    ///
    /// The flag carries no query, so it answers for the index as a whole: a covered index
    /// that a particular query narrows to no covering column would in fact be servable
    /// there, but advertising `true` and then failing would strand a dispatcher after it
    /// had already paid for the partition load. Saying `false` lets it choose
    /// [`VectorIndex::search_in_partition`] or [`VectorIndex::search_partitions`] up
    /// front. An unreadable covering schema answers `false` for the same reason.
    fn supports_prepared_partition_search(&self) -> bool {
        matches!(self.storage.covering_schema(), Ok(None))
    }

    fn auto_query_parallelism(&self, cpu_pool_size: usize) -> usize {
        if S::supports_global_topk_heap() {
            1
        } else {
            cpu_pool_size.max(1)
        }
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

        let prepare_parallelism = get_num_compute_intensive_cpus().max(1);
        let raw_query_context = self.prepare_rq_raw_query_context(&query.key)?;
        // Resolved once per search, before `query` is moved into the prepare stream.
        let covering = self.query_covering(&query)?;
        let want_covering = covering.is_some();

        if control.is_none() && S::supports_global_topk_heap() {
            let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
            pre_filter.wait_for_ready().await?;
            let prepare_index = self.clone();
            let prepare_metrics = metrics.clone();
            let prepare_raw_query_context = raw_query_context.clone();
            let prepared = stream::iter(start_idx..end_idx)
                .map(move |idx| {
                    let part_id = partitions.value(idx);
                    let mut query = query.clone();
                    query.dist_q_c = q_c_dists.value(idx);
                    let index = prepare_index.clone();
                    let pre_filter = pre_filter.clone();
                    let metrics = prepare_metrics.clone();
                    let raw_query_context = prepare_raw_query_context.clone();
                    async move {
                        index
                            .prepare_partition_without_prefilter_wait(
                                part_id as usize,
                                &query,
                                pre_filter,
                                metrics.as_ref(),
                                raw_query_context,
                            )
                            .await
                    }
                })
                .buffered(prepare_parallelism)
                .try_collect::<Vec<_>>()
                .await?;

            let use_query_residual = self.use_query_residual;
            let use_residual_scratch = self.use_residual_scratch;
            let search_metrics = metrics.clone();
            let scratch_pool = self.scratch_pool.clone();
            let (heap, covering_locations) = spawn_cpu(
                move || -> DataFusionResult<(
                    BinaryHeap<OrderedNode<u64>>,
                    HashMap<u64, CoveringLocation>,
                )> {
                    let mut heap = BinaryHeap::with_capacity(heap_capacity);
                    // O(k) side map locating the heap survivors' covering rows, recorded
                    // in-flight while each partition is loaded (empty for ordinary
                    // indexes). Nothing is read until the heap has settled.
                    let mut covering_locations: HashMap<u64, CoveringLocation> = HashMap::new();
                    scratch_pool.with_scratch(|scratch| -> DataFusionResult<()> {
                        for prepared in prepared {
                            Self::accumulate_prepared_partition_search(
                                use_query_residual,
                                use_residual_scratch,
                                prepared,
                                &mut heap,
                                &mut covering_locations,
                                want_covering,
                                scratch,
                                search_metrics.as_ref(),
                            )
                            .map_err(DataFusionError::from)?;
                        }
                        Ok(())
                    })?;
                    Ok((heap, covering_locations))
                },
            )
            .await?;

            // The gather is I/O, so it runs here rather than on the CPU pool: one bounded
            // read per contributing partition, for the survivors only.
            let gathered = match covering.as_ref() {
                Some(covering) => {
                    self.gather_survivor_covering(&covering_locations, covering, metrics.io_stats())
                        .await?
                }
                None => None,
            };
            let batch = Self::global_heap_to_batch(
                heap,
                gathered.as_ref(),
                covering.as_ref().map(|covering| covering.schema.as_ref()),
            )?;

            // Schema may be wider than VECTOR_RESULT_SCHEMA when covering columns
            // are emitted; take it from the produced batch so they stay consistent.
            let result_schema = batch.schema();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                result_schema,
                stream::once(async move { Ok(batch) }),
            )));
        }

        // The prepared channel holds a full search batch so that partitions prepared
        // while the previous batch is being searched are ready for the next greedy
        // drain, instead of serializing producer and consumer through a single slot.
        let (prepared_tx, mut prepared_rx) =
            mpsc::channel::<Result<PreparedPartitionSearch<S, Q>>>(*STREAMING_SEARCH_BATCH_SIZE);
        let (batch_tx, batch_rx) = mpsc::channel::<DataFusionResult<RecordBatch>>(1);

        let prepare_index = self.clone();
        let prepare_metrics = metrics.clone();
        let prepare_raw_query_context = raw_query_context.clone();
        tokio::spawn(async move {
            let prepare_stream = stream::iter(start_idx..end_idx)
                .map(move |idx| {
                    let part_id = partitions.value(idx);
                    let mut query = query.clone();
                    query.dist_q_c = q_c_dists.value(idx);
                    let index = prepare_index.clone();
                    let pre_filter = pre_filter.clone();
                    let metrics = prepare_metrics.clone();
                    let raw_query_context = prepare_raw_query_context.clone();
                    async move {
                        index
                            .prepare_partition(
                                part_id as usize,
                                &query,
                                pre_filter,
                                metrics.as_ref(),
                                raw_query_context,
                            )
                            .await
                    }
                })
                .buffered(prepare_parallelism);

            futures::pin_mut!(prepare_stream);
            while let Some(prepared) = prepare_stream.next().await {
                let has_error = prepared.is_err();
                if prepared_tx.send(prepared).await.is_err() || has_error {
                    break;
                }
            }
        });

        let use_query_residual = self.use_query_residual;
        let use_residual_scratch = self.use_residual_scratch;
        let search_metrics = metrics.clone();
        let search_control = control.clone();
        let scratch_pool = self.scratch_pool.clone();
        // The covering gather runs in the async half of the search loop, so it needs the
        // index and the metrics sink there as well as inside the CPU closure.
        let gather_index = self.clone();
        let gather_metrics = metrics.clone();
        // The per-partition batches are widened with the covering columns this query needs
        // (`append_covering`), so declare the matching schema -- the global-heap branch
        // above already emits its batch's own (covered) schema. Resolved before the search
        // loop takes ownership of the covering set.
        let result_schema =
            Self::covered_result_schema(covering.as_ref().map(|covering| covering.schema.as_ref()));
        // Search prepared partitions in batches. Each batch is searched in a single
        // `spawn_cpu` dispatch (amortizing the per-dispatch overhead the single-worker
        // design in #6475 avoided), but the channel `recv`/`send` stay in async code so
        // no CPU-pool thread ever parks on a channel — parking one can deadlock the pool
        // on small hosts (#7642). `should_stop` is checked per partition, so early-stop
        // granularity is unchanged.
        //
        // Batches are formed greedily: wait for one prepared partition, then drain
        // whatever else is already prepared, up to the batch size. Waiting for a full
        // batch instead would delay the first search (and the early-stop feedback it
        // produces) behind up to a whole batch of prepare I/O, which is significant
        // when prepare parallelism is low.
        tokio::spawn(async move {
            loop {
                // Stop pulling as soon as the search is done — or the receiver of our
                // results is gone — so the producer stops preparing partitions we
                // would never search.
                if search_control
                    .as_ref()
                    .is_some_and(|control| control.should_stop())
                    || batch_tx.is_closed()
                {
                    return;
                }

                let mut prepared_batch = Vec::with_capacity(*STREAMING_SEARCH_BATCH_SIZE);
                let mut prepare_error = None;
                let mut producer_done = false;
                match prepared_rx.recv().await {
                    Some(Ok(prepared)) => prepared_batch.push(prepared),
                    Some(Err(err)) => prepare_error = Some(DataFusionError::from(err)),
                    None => producer_done = true,
                }
                while prepare_error.is_none()
                    && !producer_done
                    && prepared_batch.len() < *STREAMING_SEARCH_BATCH_SIZE
                {
                    match prepared_rx.try_recv() {
                        Ok(Ok(prepared)) => prepared_batch.push(prepared),
                        Ok(Err(err)) => {
                            prepare_error = Some(DataFusionError::from(err));
                        }
                        // Nothing else is prepared yet; search what we have rather
                        // than waiting for more.
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            producer_done = true;
                        }
                    }
                }

                if !prepared_batch.is_empty() {
                    let scratch_pool = scratch_pool.clone();
                    let search_metrics = search_metrics.clone();
                    let search_control = search_control.clone();
                    // `is_closed` is synchronously callable, so a sender clone lets the
                    // CPU loop notice a dropped receiver between partitions instead of
                    // searching out the whole batch for a cancelled query. (A `select!`
                    // on `closed()` would not help here: `spawn_cpu` closures are not
                    // cancellable, so abandoning the await leaves the work running.)
                    let cancel_probe = batch_tx.clone();
                    let search_output = spawn_cpu(move || {
                        // Each output carries the partition it came from and where its
                        // survivors sit in that partition, so the covering read below can
                        // be a bounded take rather than a whole-partition scan. The read
                        // itself is I/O and stays out of this CPU closure.
                        let mut outputs: Vec<
                            DataFusionResult<(usize, RecordBatch, CoveringGather)>,
                        > = Vec::with_capacity(prepared_batch.len());
                        // `stopped` means the whole search should end (an error, an
                        // early-stop signal, or cancellation), not just this batch.
                        let mut stopped = false;
                        scratch_pool.with_scratch(|scratch| {
                            for prepared in prepared_batch {
                                if search_control
                                    .as_ref()
                                    .is_some_and(|control| control.should_stop())
                                    || cancel_probe.is_closed()
                                {
                                    stopped = true;
                                    break;
                                }
                                let partition_id = prepared.partition_id;
                                match Self::run_prepared_partition_search(
                                    use_query_residual,
                                    use_residual_scratch,
                                    prepared,
                                    want_covering,
                                    search_metrics.as_ref(),
                                    scratch,
                                )
                                .map_err(DataFusionError::from)
                                {
                                    Ok((batch, gather)) => {
                                        if let Some(control) = search_control.as_ref() {
                                            control.record_batch(&batch);
                                        }
                                        outputs.push(Ok((partition_id, batch, gather)));
                                    }
                                    Err(err) => {
                                        outputs.push(Err(err));
                                        stopped = true;
                                        break;
                                    }
                                }
                            }
                        });
                        Ok::<_, DataFusionError>((outputs, stopped))
                    })
                    .await;

                    let (outputs, stopped) = match search_output {
                        Ok(output) => output,
                        // Defensive: the closure always returns Ok (search errors are
                        // captured per partition in `outputs`), so this arm should be
                        // unreachable. Forward and stop rather than drop silently.
                        Err(err) => {
                            let _ = batch_tx.send(Err(err)).await;
                            return;
                        }
                    };
                    for output in outputs {
                        let output = match output {
                            Ok((partition_id, batch, gather)) => gather_index
                                .append_covering(
                                    partition_id,
                                    batch,
                                    gather,
                                    covering.as_ref(),
                                    gather_metrics.io_stats(),
                                )
                                .await
                                .map_err(DataFusionError::from),
                            Err(err) => Err(err),
                        };
                        if batch_tx.send(output).await.is_err() {
                            return;
                        }
                    }
                    if stopped {
                        return;
                    }
                }

                if let Some(err) = prepare_error {
                    let _ = batch_tx.send(Err(err)).await;
                    return;
                }
                if producer_done {
                    return;
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            result_schema,
            ReceiverStream::new(batch_rx),
        )))
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
        Err(Error::index("Flat index does not support load".to_string()))
    }

    async fn partition_reader(
        &self,
        partition_id: usize,
        with_vector: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        let partition = self.load_partition(partition_id, false, metrics).await?;
        let store = &partition.storage;
        let schema = if with_vector {
            store.schema().clone()
        } else {
            let schema = store.schema();
            let row_id_idx = schema.index_of(ROW_ID)?;
            Arc::new(store.schema().project(&[row_id_idx])?)
        };

        let batches = store
            .to_batches()?
            .map(|b| {
                let batch = b.project_by_schema(&schema)?;
                Ok(batch)
            })
            .collect::<Vec<_>>();
        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
        Ok(Box::pin(stream))
    }

    async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
        unimplemented!("this method is for only sub index");
    }

    fn num_rows(&self) -> u64 {
        self.storage.num_rows()
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
        todo!("this method is for only IVF_HNSW_* index");
    }

    async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
        Err(Error::index(
            "Remapping IVF in this way not supported".to_string(),
        ))
    }

    fn ivf_model(&self) -> &IvfModel {
        &self.ivf
    }

    fn quantizer(&self) -> Quantizer {
        self.storage.quantizer().unwrap()
    }

    fn partition_size(&self, part_id: usize) -> usize {
        self.storage.partition_size(part_id)
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        (S::name().try_into().unwrap(), Q::quantization_type())
    }

    fn metric_type(&self) -> DistanceType {
        self.distance_type
    }

    fn open_io_stats(&self) -> Option<ScanStats> {
        Some(self.open_io_stats)
    }
}

pub type IvfFlatIndex = IVFIndex<FlatIndex, FlatQuantizer>;
pub type IvfPq = IVFIndex<FlatIndex, ProductQuantizer>;
pub type IvfHnswSqIndex = IVFIndex<HNSW, ScalarQuantizer>;
pub type IvfHnswPqIndex = IVFIndex<HNSW, ProductQuantizer>;

async fn reconstruct_typed<S: IvfSubIndex + 'static, Q: Quantization + 'static>(
    state: &IvfIndexState<Q>,
    object_store: Arc<ObjectStore>,
    file_metadata_cache: &LanceCache,
    index_cache: LanceCache,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
) -> Result<Arc<dyn VectorIndex>> {
    let io_parallelism = object_store.io_parallelism();

    let index_path = Path::parse(&state.index_file_path)
        .map_err(|e| Error::io(format!("invalid index path: {e}")))?;

    // Derive aux path from the index path's parent directory.
    let mut parts: Vec<_> = index_path.parts().collect();
    parts.pop();
    let dir: Path = parts.into_iter().collect();
    let aux_path = dir.clone().join(INDEX_AUXILIARY_FILE_NAME);

    // Readers carry a scheduler bound to an object store, so they cannot be
    // shared across dataset opens. Reuse only portable file metadata and bind
    // fresh readers to the object store supplied for this reconstruction.
    let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
    let scheduler = ScanScheduler::new(object_store, scheduler_config);
    let index_reader = open_reader_cached(
        &scheduler,
        &index_path,
        file_metadata_cache,
        state.index_file_size,
    )
    .await?;
    let aux_reader = open_reader_cached(
        &scheduler,
        &aux_path,
        file_metadata_cache,
        state.aux_file_size,
    )
    .await?;

    let frag_reuse_index = frag_reuse_index
        .map(|index| Arc::new(CompactFragReuseIndexHandle(index)) as Arc<dyn RowIdRemapper>);
    let storage = IvfQuantizationStorage::from_cached_with_remapper(
        aux_reader,
        state.aux_ivf.clone(),
        state.metadata.clone(),
        state.distance_type,
        frag_reuse_index,
    );
    let rq_search_cache = IVFIndex::<S, Q>::rq_search_cache_from_state(state, &storage)?;

    let parsed_uuid = Uuid::parse_str(&state.uuid)
        .map_err(|e| Error::index(format!("Invalid UUID in IvfIndexState: {e}")))?;
    let index = IVFIndex::<S, Q>::from_cached_state(
        to_local_path(&index_path),
        index_path.to_string(),
        parsed_uuid,
        state.ivf.clone(),
        index_reader,
        storage,
        state.sub_index_metadata.clone(),
        state.distance_type,
        index_cache,
        io_parallelism,
        rq_search_cache,
    )?;
    Ok(Arc::new(index))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::iter::repeat_n;
    use std::{
        ops::Range,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use all_asserts::{assert_ge, assert_lt};
    use arrow::datatypes::{Float64Type, UInt8Type, UInt64Type};
    use arrow::{array::AsArray, datatypes::Float32Type};
    use arrow_array::{
        Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Int64Array,
        ListArray, PrimitiveArray, RecordBatch, RecordBatchIterator, UInt64Array,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use itertools::Itertools;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::bq::{
        RQBuildParams, RQRotationType,
        ex_dot::{blocked_ex_code_bytes, padded_query_len},
        storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RabitQuantizationMetadata, RabitQueryEstimator},
        transform::{EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN},
    };
    use lance_index::vector::storage::{PartitionColumns, VectorStore};
    use lance_index::vector::v3::subindex::IvfSubIndex;

    use crate::dataset::{
        InsertBuilder, NewColumnTransform, UpdateBuilder, WriteMode, WriteParams,
    };
    use crate::index::DatasetIndexExt;
    use crate::index::DatasetIndexInternalExt;
    use crate::index::vector::ivf::v2::{
        CoveringLocation, IVFPartitionKey, IvfFlatIndex, IvfHnswSqIndex, IvfPq, IvfStateEntryBox,
        PartitionEntry,
    };
    use crate::index::vector::utils::gather_covering_columns_by_row_id;
    use crate::utils::test::copy_test_data_to_tmp;
    use crate::{
        Dataset,
        index::vector::{VectorIndex, VectorIndexParams},
    };
    use crate::{
        dataset::optimize::{CompactionOptions, compact_files},
        index::vector::IndexFileVersion,
    };
    use lance_core::cache::{CacheBackend, CacheCodecImpl, LanceCache, WeakLanceCache};
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ID, Result};
    use lance_datagen::{Dimension, RowCount, Seed, array, gen_batch};
    use lance_encoding::decoder::DecoderPlugins;
    use lance_file::reader::{FileReader, FileReaderOptions};
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::prefilter::PreFilter;
    use lance_index::progress::IndexBuildProgress;
    use lance_index::vector::DIST_COL;
    use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::hnsw::HNSW;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
    use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
    use lance_index::vector::quantizer::QuantizerMetadata;
    use lance_index::vector::sq::ScalarQuantizer;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::{
        pq::storage::ProductQuantizationMetadata,
        sq::storage::{SQ_METADATA_KEY, ScalarQuantizationMetadata},
        storage::STORAGE_METADATA_KEY,
    };
    use lance_index::{INDEX_AUXILIARY_FILE_NAME, metrics::NoOpMetricsCollector};
    use lance_io::{
        object_store::{ObjectStore, ObjectStoreParams, StorageOptionsAccessor},
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use lance_linalg::distance::{DistanceType, multivec_distance};
    use lance_linalg::kernels::normalize_fsl;
    use lance_select::{RowAddrMask, RowAddrTreeMap};
    use lance_table::format::IndexMetadata;
    use lance_testing::datagen::{generate_random_array, generate_random_array_with_range};
    use rand::distr::{Distribution, StandardUniform, uniform::SampleUniform};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use uuid::Uuid;

    const NUM_ROWS: usize = 512;
    const DIM: usize = 32;
    // 8-bit PQ needs at least 256 training vectors; 320 leaves a stable margin
    // while 20 neighbors provide a useful recall oracle.
    const PQ_MATRIX_NUM_ROWS: usize = 320;
    const PQ_MATRIX_K: usize = 20;
    // An 8-bit PQ codebook has 256 centroids, so this is the smallest valid
    // training fixture shared by the 8-bit and 4-bit runtime cases.
    const LIGHTWEIGHT_PQ_ROWS: usize = 256;
    const LIGHTWEIGHT_PQ_PARTITIONS: usize = 2;
    const LIGHTWEIGHT_PQ_SUB_VECTORS: usize = 4;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

    struct PartitionCoverageTestFilter {
        needs_partition_rows: bool,
    }

    #[async_trait::async_trait]
    impl PreFilter for PartitionCoverageTestFilter {
        async fn wait_for_ready(&self) -> Result<()> {
            Ok(())
        }

        fn is_empty(&self) -> bool {
            false
        }

        fn needs_partition_row_ids(&self) -> bool {
            self.needs_partition_rows
        }

        fn is_empty_for(&self, _rows: &RowAddrTreeMap) -> bool {
            true
        }

        fn mask(&self) -> Arc<RowAddrMask> {
            Arc::new(RowAddrMask::all_rows())
        }

        fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
            row_ids.enumerate().map(|(index, _)| index as u64).collect()
        }
    }

    #[tokio::test]
    async fn test_partition_coverage_is_only_built_for_capable_filters() {
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0_f32; 16]), 4)
                .unwrap();
        let entry = Arc::new(PartitionEntry::<FlatIndex, FlatQuantizer>::new(
            FlatIndex::default(),
            FlatFloatStorage::new(vectors, DistanceType::L2),
        ));
        let cache = LanceCache::with_capacity(1 << 20);
        cache
            .insert_with_key(
                &IVFPartitionKey::<FlatIndex, FlatQuantizer>::new(0),
                entry.clone(),
            )
            .await;
        let weak_cache = WeakLanceCache::from(&cache);
        let size_without_coverage = entry.deep_size_of();
        let cache_weight_without_coverage = cache.size_bytes().await;

        let ordinary_filter: Arc<dyn PreFilter> = Arc::new(PartitionCoverageTestFilter {
            needs_partition_rows: false,
        });
        let returned = super::IVFIndex::<FlatIndex, FlatQuantizer>::prefilter_for_partition(
            &weak_cache,
            0,
            &entry,
            ordinary_filter.clone(),
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(&returned, &ordinary_filter));
        assert!(entry.partition_rows.get().is_none());
        assert_eq!(cache.size_bytes().await, cache_weight_without_coverage);

        let segment_filter: Arc<dyn PreFilter> = Arc::new(PartitionCoverageTestFilter {
            needs_partition_rows: true,
        });
        let returned = super::IVFIndex::<FlatIndex, FlatQuantizer>::prefilter_for_partition(
            &weak_cache,
            0,
            &entry,
            segment_filter,
        )
        .await
        .unwrap();
        assert!(returned.is_empty());

        let first_rows = entry.partition_rows();
        let second_rows = entry.partition_rows();
        assert!(Arc::ptr_eq(&first_rows, &second_rows));
        assert!(entry.deep_size_of() > size_without_coverage);
        let cache_weight_with_coverage = cache.size_bytes().await;
        assert!(cache_weight_with_coverage > cache_weight_without_coverage);
        assert!(cache_weight_with_coverage >= entry.deep_size_of());
        assert!(entry.partition_rows_accounted.load(Ordering::Acquire));
    }

    #[test]
    fn test_rotated_partition_centroid_slice_borrows_cache() {
        let cache = super::RabitSearchCache {
            rotated_centroids: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            code_dim: 2,
        };

        let centroid = super::rotated_partition_centroid_slice(Some(&cache), 1).unwrap();

        assert_eq!(centroid, &[3.0, 4.0]);
        assert_eq!(centroid.as_ptr(), cache.rotated_centroids[2..].as_ptr());
        assert!(super::rotated_partition_centroid_slice(Some(&cache), 3).is_none());
        assert!(super::rotated_partition_centroid_slice(None, 0).is_none());
    }

    #[test]
    fn test_rabit_ex_scratch_len_uses_num_bits() {
        // Block-aligned dims read the rotated query in place.
        let dim = 960;
        for num_bits in [1, 3, 5, 7, 9] {
            assert_eq!(super::rabit_ex_scratch_len(dim, num_bits), 0);
        }

        // Unaligned multi-bit queries add one padded query copy.
        let dim = 968;
        assert_eq!(super::rabit_ex_scratch_len(dim, 1), 0);
        assert_eq!(super::rabit_ex_scratch_len(dim, 7), padded_query_len(dim));
    }

    #[test]
    fn test_rabit_u8_scratch_len_includes_ex_fastscan_tables() {
        let dim = 960;

        assert_eq!(super::rabit_u8_scratch_len(dim, 1), dim * 4);
        assert_eq!(super::rabit_u8_scratch_len(dim, 3), dim * 8);
        assert_eq!(super::rabit_u8_scratch_len(dim, 5), dim * 16);
        assert_eq!(super::rabit_u8_scratch_len(dim, 7), dim * 4);
        assert_eq!(super::rabit_u8_scratch_len(dim, 9), dim * 32);
    }

    #[test]
    fn test_rabit_query_scratch_capacity_does_not_preallocate_u32() {
        let dim = 960;
        let max_partition_len = 4096;

        let capacity = super::rabit_query_scratch_capacity(dim, max_partition_len, 5);

        assert_eq!(capacity.distances, max_partition_len);
        assert_eq!(capacity.query_f32, dim + dim * 4);
        assert_eq!(capacity.u16, max_partition_len);
        assert_eq!(capacity.u8, dim * 16);
        assert_eq!(capacity.u32, 0);
    }

    /// Vector values laid out as `num_clusters` well-separated clusters: row `r` lands in
    /// cluster `r % num_clusters`, centred at `cluster * 1000.0` with a tiny per-row offset
    /// so no two rows coincide.
    ///
    /// **The separation is load-bearing, not cosmetic.** `early_pruning` RAISES
    /// `minimum_nprobes` to the number of centroids within `dists[0] * factor` (7.0 for
    /// `k` in 2..=10, 81.0 for `k >= 11`). With evenly-spread vectors every centroid
    /// survives pruning, so `minimum_nprobes` reaches `maximum_nprobes`, `late_search`
    /// returns at its first guard, and the no-rows shortcut -- and therefore the covered
    /// recovery path -- is never reached. A test targeting either then passes vacuously
    /// against a live bug, which has happened repeatedly on this feature. With these
    /// clusters a query at the origin is ~10^6 times closer to cluster 0 than to any
    /// other, so pruning yields 1 and unsearched partitions remain.
    ///
    /// See also `generate_clustered_batch`, the pre-existing generator used by the
    /// non-covering partition-split tests; it emits a different schema and is not a
    /// drop-in substitute here.
    fn clustered_vector_values(
        rows: impl IntoIterator<Item = usize>,
        dim: i32,
        num_clusters: usize,
    ) -> Vec<f32> {
        rows.into_iter()
            .flat_map(|r| {
                let center = (r % num_clusters) as f32 * 1000.0;
                (0..dim as usize).map(move |d| center + (r * dim as usize + d) as f32 * 1e-3)
            })
            .collect()
    }

    async fn generate_test_dataset<T: ArrowPrimitiveType>(
        test_uri: &str,
        range: Range<T::Native>,
    ) -> (Dataset, Arc<FixedSizeListArray>)
    where
        T::Native: SampleUniform,
    {
        let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, false);
        let vectors = batch.column_by_name("vector").unwrap().clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dataset, Arc::new(vectors.as_fixed_size_list().clone()))
    }

    async fn generate_multivec_test_dataset<T: ArrowPrimitiveType>(
        test_uri: &str,
        range: Range<T::Native>,
    ) -> (Dataset, Arc<ListArray>)
    where
        T::Native: SampleUniform,
    {
        let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, true);
        let vectors = batch.column_by_name("vector").unwrap().clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, Arc::new(vectors.as_list::<i32>().clone()))
    }

    async fn append_dataset<T: ArrowPrimitiveType>(
        dataset: &mut Dataset,
        num_rows: usize,
        range: Range<T::Native>,
    ) -> ArrayRef
    where
        T::Native: SampleUniform,
    {
        let is_multivector = matches!(
            dataset.schema().field("vector").unwrap().data_type(),
            DataType::List(_)
        );
        let row_count = dataset.count_all_rows().await.unwrap();
        let (batch, schema) =
            generate_batch::<T>(num_rows, Some(row_count as u64), range, is_multivector);
        let vectors = batch["vector"].clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();
        vectors
    }

    async fn open_rq_aux_reader(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
        index_uuid: &str,
    ) -> FileReader {
        let index_path = dataset
            .indices_dir()
            .join(index_uuid)
            .join(INDEX_AUXILIARY_FILE_NAME);
        let file_scheduler = scheduler
            .open_file(&index_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap()
    }

    async fn get_rq_metadata(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
        index_uuid: &str,
    ) -> RabitQuantizationMetadata {
        let reader = open_rq_aux_reader(dataset, scheduler, index_uuid).await;
        let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
        let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
        serde_json::from_str(&metadata_entries[0]).unwrap()
    }

    async fn get_sq_metadata(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
        index_uuid: &str,
    ) -> ScalarQuantizationMetadata {
        let index_path = dataset
            .indices_dir()
            .join(index_uuid)
            .join(INDEX_AUXILIARY_FILE_NAME);
        let file_scheduler = scheduler
            .open_file(&index_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        if let Some(metadata) = reader.schema().metadata.get(SQ_METADATA_KEY) {
            serde_json::from_str(metadata).unwrap()
        } else {
            let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
            let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
            serde_json::from_str(&metadata_entries[0]).unwrap()
        }
    }

    async fn assert_rq_rotation_type(dataset: &Dataset, expected: RQRotationType) {
        let obj_store = Arc::new(ObjectStore::local());
        let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty(), "Expected at least one vector index");
        for index in indices.iter() {
            let rq_meta =
                get_rq_metadata(dataset, scheduler.clone(), &index.uuid.to_string()).await;
            assert_eq!(
                rq_meta.rotation_type, expected,
                "RQ rotation type mismatch for index {}",
                index.uuid
            );
        }
    }

    fn generate_batch<T: ArrowPrimitiveType>(
        num_rows: usize,
        start_id: Option<u64>,
        range: Range<T::Native>,
        is_multivector: bool,
    ) -> (RecordBatch, SchemaRef)
    where
        T::Native: SampleUniform,
    {
        const VECTOR_NUM_PER_ROW: usize = 3;
        let start_id = start_id.unwrap_or(0);
        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + num_rows as u64,
        ));
        let total_floats = match is_multivector {
            true => num_rows * VECTOR_NUM_PER_ROW * DIM,
            false => num_rows * DIM,
        };
        let vectors = generate_random_array_with_range::<T>(total_floats, range);
        let data_type = vectors.data_type().clone();
        let mut fields = vec![Field::new("id", DataType::UInt64, false)];
        let mut arrays: Vec<ArrayRef> = vec![ids];
        let mut fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        if fsl.value_type() != DataType::UInt8 {
            fsl = normalize_fsl(&fsl).unwrap();
        }
        if is_multivector {
            let vector_field = Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
                true,
            ));
            fields.push(Field::new(
                "vector",
                DataType::List(vector_field.clone()),
                true,
            ));
            let array = Arc::new(ListArray::new(
                vector_field,
                OffsetBuffer::from_lengths(std::iter::repeat_n(VECTOR_NUM_PER_ROW, num_rows)),
                Arc::new(fsl),
                None,
            ));
            arrays.push(array);
        } else {
            fields.push(Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
                true,
            ));
            let array = Arc::new(fsl);
            arrays.push(array);
        }
        let schema: Arc<_> = Schema::new(fields).into();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        (batch, schema)
    }

    fn generate_clustered_batch(
        rows_per_partition: usize,
        offsets: [f32; 2],
    ) -> (RecordBatch, SchemaRef) {
        let num_partitions = offsets.len();
        let total_rows = rows_per_partition * num_partitions;
        let mut ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * DIM);
        let mut rng = StdRng::seed_from_u64(42);
        for (cluster_idx, offset) in offsets.iter().enumerate() {
            for row in 0..rows_per_partition {
                ids.push((cluster_idx * rows_per_partition + row) as u64);
                for dim in 0..DIM {
                    let base = if dim == 0 { *offset } else { 0.0 };
                    let noise = (rng.random::<f32>() - 0.5) * 0.02;
                    values.push(base + noise);
                }
            }
        }
        let ids = Arc::new(UInt64Array::from(ids));
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap(),
        );
        let schema: Arc<_> = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ])
        .into();
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        (batch, schema)
    }

    fn generate_clustered_multivec_batch(
        cluster_sizes: &[usize],
        centroids: &[(f32, f32)],
        vectors_per_row: usize,
        start_id: u64,
    ) -> (RecordBatch, SchemaRef) {
        assert_eq!(
            cluster_sizes.len(),
            centroids.len(),
            "cluster sizes and centroids must match"
        );
        const ITEM_FIELD_NAME: &str = "item";
        let total_rows: usize = cluster_sizes.iter().sum();
        let mut ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * vectors_per_row * DIM);
        let mut rng = StdRng::seed_from_u64(12345);
        let mut current_id = start_id;
        for (&rows, &(x, y)) in cluster_sizes.iter().zip(centroids.iter()) {
            for _ in 0..rows {
                ids.push(current_id);
                current_id += 1;
                for _ in 0..vectors_per_row {
                    for dim in 0..DIM {
                        let base = match dim {
                            0 => x,
                            1 => y,
                            _ => 0.0,
                        };
                        let noise = (rng.random::<f32>() - 0.5) * 0.02;
                        values.push(base + noise);
                    }
                }
            }
        }
        let ids_array = Arc::new(UInt64Array::from(ids));
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap();
        let vector_field = Arc::new(Field::new(
            ITEM_FIELD_NAME,
            DataType::FixedSizeList(
                Arc::new(Field::new(ITEM_FIELD_NAME, DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        ));
        let offsets_buffer =
            OffsetBuffer::from_lengths(std::iter::repeat_n(vectors_per_row, total_rows));
        let list_array = Arc::new(ListArray::new(
            vector_field.clone(),
            offsets_buffer,
            Arc::new(vectors),
            None,
        ));
        let schema: Arc<_> = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", DataType::List(vector_field), false),
        ])
        .into();
        let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, list_array]).unwrap();
        (batch, schema)
    }

    fn build_centroids_for_offsets(offsets: &[f32]) -> Arc<FixedSizeListArray> {
        let mut centroid_values = Vec::with_capacity(offsets.len() * DIM);
        for &offset in offsets {
            for dim in 0..DIM {
                centroid_values.push(if dim == 0 { offset } else { 0.0 });
            }
        }
        Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        )
    }

    fn build_centroids_2d(centroids: &[(f32, f32)]) -> Arc<FixedSizeListArray> {
        let mut values = Vec::with_capacity(centroids.len() * DIM);
        for &(x, y) in centroids {
            for dim in 0..DIM {
                values.push(match dim {
                    0 => x,
                    1 => y,
                    _ => 0.0,
                });
            }
        }
        Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap(),
        )
    }

    fn make_fragment_offset_batches(
        rows_per_fragment: usize,
        offsets: &[f32],
    ) -> (Arc<Schema>, Vec<RecordBatch>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                false,
            ),
        ]));

        let mut next_id = 0_u64;
        let batches = offsets
            .iter()
            .map(|offset| {
                let ids = Arc::new(UInt64Array::from_iter_values(
                    next_id..next_id + rows_per_fragment as u64,
                ));
                next_id += rows_per_fragment as u64;

                let mut values = Vec::with_capacity(rows_per_fragment * DIM);
                for _ in 0..rows_per_fragment {
                    for dim in 0..DIM {
                        values.push(*offset + dim as f32);
                    }
                }

                let vectors = Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                        .unwrap(),
                );
                RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap()
            })
            .collect();

        (schema, batches)
    }

    struct VectorIndexTestContext {
        stats_json: String,
        stats: serde_json::Value,
        index: Arc<dyn VectorIndex>,
    }

    impl VectorIndexTestContext {
        fn stats(&self) -> &serde_json::Value {
            &self.stats
        }

        fn stats_json(&self) -> &str {
            &self.stats_json
        }

        fn num_partitions(&self) -> usize {
            self.stats()["indices"][0]["num_partitions"]
                .as_u64()
                .expect("num_partitions should be present") as usize
        }

        fn ivf(&self) -> &IvfPq {
            self.index
                .as_any()
                .downcast_ref::<IvfPq>()
                .expect("expected IvfPq index")
        }

        fn ivf_flat(&self) -> &IvfFlatIndex {
            self.index
                .as_any()
                .downcast_ref::<IvfFlatIndex>()
                .expect("expected IvfFlat index")
        }
    }

    fn lightweight_pq_params() -> PQBuildParams {
        PQBuildParams {
            num_sub_vectors: LIGHTWEIGHT_PQ_SUB_VECTORS,
            num_bits: 4,
            max_iters: 2,
            sample_rate: 16,
            ..Default::default()
        }
    }

    fn lightweight_pq_params_with_bits(num_bits: usize) -> PQBuildParams {
        let num_sub_vectors = if num_bits == 4 {
            // M4 is only a 2-byte code, so random KMeans/HNSW can leave recall near
            // the threshold. M32 restores the original 4-bit test capacity.
            DIM
        } else {
            LIGHTWEIGHT_PQ_SUB_VECTORS
        };
        PQBuildParams {
            num_sub_vectors,
            num_bits,
            max_iters: 2,
            sample_rate: 16,
            ..Default::default()
        }
    }

    fn lightweight_hnsw_params() -> HnswBuildParams {
        HnswBuildParams::default()
            .max_level(2)
            .num_edges(4)
            .ef_construction(16)
    }

    fn make_seeded_vector_batch(num_rows: usize) -> (RecordBatch, SchemaRef) {
        let batch = lance_datagen::gen_batch()
            .with_seed(lance_datagen::Seed::from(42))
            .col("id", lance_datagen::array::step::<UInt64Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>((DIM as u32).into()),
            )
            .into_batch_rows(lance_datagen::RowCount::from(num_rows as u64))
            .unwrap();
        let schema = batch.schema();
        (batch, schema)
    }

    async fn search_lightweight_pq_index(
        dataset: &Dataset,
        query: &dyn Array,
        k: usize,
        num_partitions: usize,
        refine_factor: u32,
        ef: usize,
    ) -> RecordBatch {
        dataset
            .scan()
            .nearest("vector", query, k)
            .unwrap()
            .minimum_nprobes(num_partitions)
            .ef(ef)
            .refine(refine_factor)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap()
    }

    async fn assert_lightweight_pq_index(
        distance_type: DistanceType,
        num_bits: usize,
        use_hnsw: bool,
    ) {
        const INDEX_NAME: &str = "test_index";
        const K: usize = 10;

        let test_dir = TempStrDir::default();
        let (batch, schema) = make_seeded_vector_batch(LIGHTWEIGHT_PQ_ROWS);
        let vectors = batch["vector"].as_fixed_size_list().clone();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
            .await
            .unwrap();

        let mut ivf_params = IvfBuildParams::new(LIGHTWEIGHT_PQ_PARTITIONS);
        ivf_params.max_iters = 2;
        ivf_params.sample_rate = 16;
        let pq_params = lightweight_pq_params_with_bits(num_bits);
        let expected_num_sub_vectors = pq_params.num_sub_vectors;
        let params = if use_hnsw {
            VectorIndexParams::with_ivf_hnsw_pq_params(
                distance_type,
                ivf_params,
                lightweight_hnsw_params(),
                pq_params,
            )
        } else {
            VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params)
        };
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats_json = dataset.index_statistics(INDEX_NAME).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
        let expected_index_type = if use_hnsw { "IVF_HNSW_PQ" } else { "IVF_PQ" };
        let expected_sub_index = if use_hnsw { "HNSW" } else { "PQ" };
        assert_eq!(stats["index_type"], expected_index_type);
        assert_eq!(
            stats["indices"][0]["num_partitions"],
            LIGHTWEIGHT_PQ_PARTITIONS
        );
        assert_eq!(
            stats["indices"][0]["sub_index"]["index_type"],
            expected_sub_index
        );
        assert_eq!(stats["indices"][0]["sub_index"]["nbits"], num_bits);
        assert_eq!(
            stats["indices"][0]["sub_index"]["num_sub_vectors"],
            expected_num_sub_vectors
        );
        if use_hnsw {
            let hnsw_params = &stats["indices"][0]["sub_index"]["params"];
            assert_eq!(hnsw_params["max_level"], 2);
            assert_eq!(hnsw_params["m"], 4);
            assert_eq!(hnsw_params["ef_construction"], 16);
        }

        let query = vectors.value(0);
        let ground_truth = ground_truth(&dataset, "vector", query.as_ref(), K, distance_type).await;
        let before_reopen = search_lightweight_pq_index(
            &dataset,
            query.as_ref(),
            K,
            LIGHTWEIGHT_PQ_PARTITIONS,
            4,
            64,
        )
        .await;
        assert_eq!(before_reopen.num_rows(), K);
        let row_ids = before_reopen[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), K);
        let distances = before_reopen[DIST_COL]
            .as_primitive::<Float32Type>()
            .values();
        assert!(distances.iter().all(|distance| distance.is_finite()));
        assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
        let recall = row_ids
            .iter()
            .filter(|row_id| ground_truth.contains(row_id))
            .count() as f32
            / K as f32;
        assert_ge!(recall, 0.5, "recall: {recall}");

        drop(dataset);
        let reopened = Dataset::open(test_dir.as_str()).await.unwrap();
        let reopened_stats: serde_json::Value =
            serde_json::from_str(&reopened.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(reopened_stats, stats);
        assert_eq!(
            search_lightweight_pq_index(
                &reopened,
                query.as_ref(),
                K,
                LIGHTWEIGHT_PQ_PARTITIONS,
                4,
                64,
            )
            .await,
            before_reopen
        );
    }

    async fn load_vector_index_context(
        dataset: &Dataset,
        column: &str,
        index_name: &str,
    ) -> VectorIndexTestContext {
        let stats_json = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
        let uuid_str = stats["indices"][0]["uuid"]
            .as_str()
            .expect("Index uuid should be present");
        let uuid = Uuid::parse_str(uuid_str).expect("uuid in stats should be a valid UUID");
        let index = dataset
            .open_vector_index(column, &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        VectorIndexTestContext {
            stats_json,
            stats,
            index,
        }
    }

    /// End-to-end test for index-included ("covering") columns : an
    /// `covering_columns` request on the build params must survive the full
    /// build -> shuffle -> persist -> reopen path and land in the IVF_PQ
    /// partition storage, so covered queries can avoid a take.
    #[tokio::test]
    async fn test_ivf_pq_covering_columns_roundtrip() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        // `generate_test_dataset` produces an `id` (UInt64) column besides `vector`.
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let storage = ctx
            .ivf()
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap();
        assert!(
            storage.batch().column_by_name("id").is_some(),
            "included column 'id' should be co-located in IVF_PQ partition storage"
        );

        // The covering columns are also declared generically on IndexMetadata
        // (by field id), so the read path can discover them for any index type.
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let indices = dataset.load_indices().await.unwrap();
        let idx = indices
            .iter()
            .find(|i| i.name == INDEX_NAME)
            .expect("index should exist");
        assert_eq!(
            idx.covering_fields,
            vec![id_field_id],
            "IndexMetadata.covering_fields should record the covered column's field id"
        );

        // The REAL producer must raise the reader+writer fence, not just the
        // doctored-metadata commit that upstream's
        // `test_covering_commit_fences_the_table_with_a_feature_flag` exercises: a
        // pre-covering build that opened this dataset would select the index by
        // membership of `fields` and answer a query on the carried column with
        // an index keyed on another one.
        use lance_table::feature_flags::FLAG_COVERED_INDEX_METADATA;
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_COVERED_INDEX_METADATA,
            0,
            "a covered create must raise the reader fence"
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_COVERED_INDEX_METADATA,
            0,
            "a covered create must raise the writer fence"
        );
    }

    /// A multivector column stores one entry per sub-vector (all sharing the source
    /// row id). The covering build must preserve every sub-vector -- the stale-row
    /// dedup must not collapse legitimate repeated row ids, or recall silently drops.
    #[tokio::test]
    async fn test_ivf_pq_covered_multivector_preserves_all_subvectors() {
        const INDEX_NAME: &str = "vector_idx";
        const SUBVECS_PER_ROW: usize = 3;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) =
            generate_multivec_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        // Multivector requires cosine. Cover the `id` column.
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let mut total_stored = 0usize;
        let mut has_id = false;
        for p in 0..ctx.num_partitions() {
            let storage = ctx
                .ivf()
                .load_partition_storage(p, PartitionColumns::All, None)
                .await
                .unwrap();
            total_stored += storage.batch().num_rows();
            has_id |= storage.batch().column_by_name("id").is_some();
        }
        assert!(
            has_id,
            "multivector covered storage should carry covering column 'id'"
        );
        assert_eq!(
            total_stored,
            NUM_ROWS * SUBVECS_PER_ROW,
            "covering multivector build must keep every sub-vector, not collapse per row id"
        );
    }

    /// Read-side payoff for multivector: a covered projection is carried through
    /// `MultivectorScoringExec` (which re-groups sub-vectors back to rows), so no
    /// `TakeExec` against the base table is needed.
    #[tokio::test]
    async fn test_ivf_pq_covered_multivector_projection_skips_take() {
        use arrow_array::types::UInt64Type;

        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let (mut dataset, vectors) =
            generate_multivec_test_dataset::<Float32Type>(test_dir.as_str(), 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let query = vectors.value(0);
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 10).unwrap();
        scan.minimum_nprobes(4);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered multivector projection ['id'] should skip TakeExec; plan was:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' must be emitted through multivector scoring")
            .as_primitive::<UInt64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        assert!(batch.num_rows() > 0, "query should return rows");
        // Single-fragment, step-id dataset => id == row offset == _rowid, so a correctly
        // carried covered value equals the row id for every returned row (payload
        // re-attached to the right row through scoring, not misaligned or stale).
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i),
                row_ids.value(i),
                "covered 'id' must stay row-aligned through multivector scoring"
            );
        }
    }

    /// A dotted/nested include path cannot be covered (the covering gather projects only
    /// top-level names); reject it at build time.
    #[tokio::test]
    async fn test_ivf_pq_rejects_dotted_include_column() {
        let test_dir = TempStrDir::default();
        let (mut dataset, _) =
            generate_test_dataset::<Float32Type>(test_dir.as_str(), 0.0..1.0).await;
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id.sub".to_string()]);
        let err = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .expect_err("a dotted include column must be rejected");
        assert!(
            err.to_string().contains("nested/dotted"),
            "expected a nested/dotted rejection, got: {err}"
        );
    }

    /// Covering names that collide with index storage (the indexed vector column itself,
    /// a reserved storage column, or a duplicate) must be rejected at build time.
    #[tokio::test]
    async fn test_ivf_rejects_reserved_and_duplicate_covering_columns() {
        let test_dir = TempStrDir::default();
        let (mut dataset, _) =
            generate_test_dataset::<Float32Type>(test_dir.as_str(), 0.0..1.0).await;
        let build = |cols: Vec<String>| {
            let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
            params.covering_columns(cols);
            params
        };

        // The indexed vector column itself.
        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &build(vec!["vector".to_string()]),
                true,
            )
            .await
            .expect_err("covering the indexed vector column must be rejected");
        assert!(
            err.to_string().contains("indexed vector column itself"),
            "got: {err}"
        );

        // Reserved storage/transform column names. The reserved check runs before the
        // schema-existence check, so these are rejected even though the dataset lacks them.
        // The RaBitQ code/factor columns and the IVF partition transform's `__centroid_dist`
        // are internal to the build pipeline: covering one would advertise it in
        // `covering_fields` while the storage's `covering_field_indices` drops it, so a
        // covered query would declare a column storage never emits.
        for internal in [
            "__pq_code",
            "__sq_code",
            "__ivf_part_id",
            "__centroid_dist",
            "__ex_codes",
            "__blocked_ex_codes",
            "__add_factors",
            "__scale_factors",
            "__error_factors",
            "__add_factors_ex",
            "__scale_factors_ex",
        ] {
            let err = dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    None,
                    &build(vec![internal.to_string()]),
                    true,
                )
                .await
                .expect_err("covering a reserved storage/transform name must be rejected");
            assert!(
                err.to_string().contains("reserved index storage"),
                "internal name '{internal}' must be reserved, got: {err}"
            );
        }

        // Duplicate covering columns.
        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &build(vec!["id".to_string(), "id".to_string()]),
                true,
            )
            .await
            .expect_err("duplicate covering columns must be rejected");
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    /// A blob column stores out-of-line descriptors, not inline data, so it cannot be
    /// covered by a vector index; reject it at build time.
    #[tokio::test]
    async fn test_ivf_flat_rejects_blob_include_column() {
        use arrow_array::{
            Float32Array, Int32Array, LargeBinaryArray, RecordBatch, RecordBatchIterator,
        };
        use lance_arrow::BLOB_META_KEY;
        use lance_file::version::LanceFileVersion;
        use std::collections::HashMap;

        let dim = 4i32;
        let n = 32usize;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
            Field::new("blobs", DataType::LargeBinary, true).with_metadata(HashMap::from([(
                BLOB_META_KEY.to_string(),
                "true".to_string(),
            )])),
        ]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..n as i32 * dim).map(|v| v as f32).collect::<Vec<_>>()),
            dim,
        )
        .unwrap();
        let blobs: Vec<Option<&[u8]>> = (0..n).map(|_| Some(b"x".as_slice())).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())),
                Arc::new(vectors),
                Arc::new(LargeBinaryArray::from(blobs)),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://c1-blob-include",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["blobs".to_string()]);
        let err = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .expect_err("a blob include column must be rejected");
        assert!(
            err.to_string().contains("blob column"),
            "expected a blob rejection, got: {err}"
        );
    }

    /// Covering must never change QUERY RESULTS: for any query, a covered index
    /// returns exactly the rows and distances a non-covered index returns (covering
    /// only changes how projected columns are fetched). Exercises both the late-search
    /// shortcut config (min < max nprobes: prefilter-matched rows without index
    /// entries come back with INFINITY distance) and the all-partitions-searched
    /// corner (min == max: such rows are NOT returned at all).
    #[tokio::test]
    async fn test_covered_prefilter_results_match_non_covered() {
        use arrow_array::types::UInt64Type;
        use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
        use arrow_buffer::NullBuffer;

        const INDEX: &str = "vec_idx";
        let dim = 4i32;
        let n = 4096usize;
        let n_null = 3usize; // ids 0..3 have NULL vectors (no index entry)

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));
        let ids: Vec<i32> = (0..n as i32).collect();
        // The prefilter matches ids 0..5: three null-vector rows plus two real ones.
        let cats: Vec<&str> = (0..n)
            .map(|i| if i < 5 { "picked" } else { "rest" })
            .collect();
        // Two well-separated clusters (around 0.0 and 100.0). The real picked rows
        // (ids 3, 4) live in cluster B; the query below probes cluster A first, so the
        // late search / shortcut path is genuinely exercised at min < max nprobes.
        let values: Vec<f32> = (0..n)
            .flat_map(|r| {
                let center = if r % 2 == 0 { 0.0f32 } else { 100.0 };
                (0..dim as usize)
                    .map(move |d| center + ((r * dim as usize + d) % 400) as f32 * 1e-3)
            })
            .collect();
        let validity: Vec<bool> = (0..n).map(|i| i >= n_null).collect();
        let vector = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(Float32Array::from(values)),
            Some(NullBuffer::from(validity)),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(cats)),
                Arc::new(vector),
            ],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://covered_noncovered_parity",
            None,
        )
        .await
        .unwrap();

        // (rowid -> distance) results for the prefiltered query under the given probe
        // configuration and index variant.
        async fn run(dataset: &Dataset, min_nprobes: usize, max_nprobes: usize) -> Vec<(u64, f32)> {
            let q = Float32Array::from(vec![0.0f32; 4]);
            let mut scan = dataset.scan();
            scan.nearest("vector", &q, 10).unwrap();
            scan.minimum_nprobes(min_nprobes);
            scan.maximum_nprobes(max_nprobes);
            scan.filter("category = 'picked'").unwrap();
            scan.prefilter(true);
            scan.with_row_id();
            scan.project(&["id"]).unwrap();
            let plan = scan.explain_plan(false).await.unwrap();
            assert!(
                plan.contains("ANNSubIndex"),
                "the parity comparison requires the index path; plan:\n{plan}"
            );
            let batch = scan.try_into_batch().await.unwrap();
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            let dists = batch["_distance"].as_primitive::<Float32Type>();
            let mut out: Vec<(u64, f32)> = row_ids
                .values()
                .iter()
                .zip(dists.values().iter())
                .map(|(r, d)| (*r, *d))
                .collect();
            out.sort_by_key(|(r, _)| *r);
            out
        }

        for (min_nprobes, max_nprobes) in [(1usize, 2usize), (2, 2)] {
            let centroids = || {
                let vals: Vec<f32> = [0.0f32, 100.0]
                    .iter()
                    .flat_map(|c| std::iter::repeat_n(*c, dim as usize))
                    .collect();
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(vals), dim).unwrap(),
                )
            };
            let plain_params = VectorIndexParams::with_ivf_flat_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(2, centroids()).unwrap(),
            );
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some(INDEX.to_string()),
                    &plain_params,
                    true,
                )
                .await
                .unwrap();
            let plain = run(&dataset, min_nprobes, max_nprobes).await;

            let mut covered_params = VectorIndexParams::with_ivf_flat_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(2, centroids()).unwrap(),
            );
            covered_params.covering_columns(vec!["id".to_string()]);
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some(INDEX.to_string()),
                    &covered_params,
                    true,
                )
                .await
                .unwrap();
            let covered = run(&dataset, min_nprobes, max_nprobes).await;

            assert_eq!(
                covered, plain,
                "covered results must be identical to non-covered \
                 (min_nprobes={min_nprobes}, max_nprobes={max_nprobes})"
            );
        }
    }

    /// A row whose vector is null has no index entry, so the covered search never returns
    /// it -- but a bounded, selective prefilter that admits it still expects it in the
    /// results (parity with a non-covered scan). The covered path must recover it, with
    /// its covering column fetched from the base table.
    #[tokio::test]
    async fn test_ivf_covered_recovers_null_vector_prefilter_rows() {
        use arrow_array::types::{Int32Type, UInt64Type};
        use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
        use arrow_buffer::NullBuffer;

        const INDEX: &str = "vec_idx";
        let dim = 4i32;
        let n = 40usize;
        let n_rare = 3usize; // category "rare" rows, all with NULL vectors (no index entry)

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        let ids: Vec<i32> = (0..n as i32).collect();
        let cats: Vec<&str> = (0..n)
            .map(|i| if i < n_rare { "rare" } else { "common" })
            .collect();
        // Two well-separated clusters so early pruning keeps minimum_nprobes at 1:
        // the recovery path only fires where the non-covered shortcut would (an
        // all-partitions-searched query returns only found rows on both).
        let values: Vec<f32> = clustered_vector_values(0..n, dim, 2);
        let validity: Vec<bool> = (0..n).map(|i| i >= n_rare).collect();
        let vector = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(Float32Array::from(values)),
            Some(NullBuffer::from(validity)),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(cats)),
                Arc::new(vector),
            ],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://covered_null_vec_recover",
            None,
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        let scalar = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(&["category"], IndexType::BTree, None, &scalar, false)
            .await
            .unwrap();

        // The prefilter admits only the "rare" rows -- all null-vector, so the search
        // returns nothing; every result row must come from the recovery path.
        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 10).unwrap();
        // Recovery matches the non-covered shortcut, which needs unsearched
        // partitions to exist (min < max nprobes); all-partitions-searched queries
        // return only found rows on both covered and non-covered indexes.
        scan.minimum_nprobes(1);
        scan.filter("category = 'rare'").unwrap();
        scan.prefilter(true);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch.num_rows(),
            n_rare,
            "all null-vector prefilter rows must be recovered (covered parity with non-covered)"
        );
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' must be emitted")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        for i in 0..ids.len() {
            // Row-aligned covered value fetched from the base table for the recovered row.
            assert_eq!(ids.value(i) as u64, row_ids.value(i));
            assert!(
                (ids.value(i) as usize) < n_rare,
                "only the rare rows are admitted"
            );
        }
    }

    /// The covered exec emits from two paths in one stream: the search path (rows with an
    /// index entry, covering columns read back from index storage) and the recovery path
    /// (null-vector rows, covering columns taken from the base table with the exec's declared
    /// dataset-typed schema). A prefilter admitting BOTH kinds forces both paths into a single
    /// result, so this guards that their batch schemas stay compatible -- if the index
    /// round-trip ever changed a covered field's type or nullability, the concatenation here
    /// would fail. Covered column is non-nullable to make a nullability drift observable.
    #[tokio::test]
    async fn test_ivf_covered_mixed_search_and_recovery_share_schema() {
        use arrow_array::types::{Int32Type, UInt64Type};
        use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
        use arrow_buffer::NullBuffer;

        const INDEX: &str = "vec_idx";
        let dim = 4i32;
        let n = 40usize;
        let n_null = 3usize; // rows 0..3 have NULL vectors (no index entry -> recovery path)
        let admit_below = 5i32; // prefilter admits ids 0..5: null rows 0,1,2 + indexed rows 3,4

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        let ids: Vec<i32> = (0..n as i32).collect();
        let cats: Vec<&str> = (0..n).map(|_| "x").collect();
        // Two well-separated clusters so early pruning keeps minimum_nprobes at 1
        // (see test_ivf_covered_recovers_null_vector_prefilter_rows).
        let values: Vec<f32> = clustered_vector_values(0..n, dim, 2);
        let validity: Vec<bool> = (0..n).map(|i| i >= n_null).collect();
        let vector = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(Float32Array::from(values)),
            Some(NullBuffer::from(validity)),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(cats)),
                Arc::new(vector),
            ],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://covered_mixed_recover",
            None,
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        // A btree on id makes `id < admit_below` a bounded, selective prefilter.
        let scalar = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(&["id"], IndexType::BTree, None, &scalar, false)
            .await
            .unwrap();

        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 10).unwrap();
        // Recovery matches the non-covered shortcut, which needs unsearched
        // partitions to exist (min < max nprobes); all-partitions-searched queries
        // return only found rows on both covered and non-covered indexes.
        scan.minimum_nprobes(1);
        scan.filter(&format!("id < {admit_below}")).unwrap();
        scan.prefilter(true);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        // Concatenating the search-path batches (ids 3,4) with the recovery batch (ids 0,1,2)
        // succeeds only if both paths carry a compatible schema for the covered `id` column.
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch.num_rows(),
            admit_below as usize,
            "both indexed and null-vector admitted rows must be returned"
        );
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' must be emitted")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        let mut got: Vec<i32> = Vec::with_capacity(ids.len());
        for i in 0..ids.len() {
            // id == row offset == _rowid, so a row-aligned covered value equals its row id.
            assert_eq!(ids.value(i) as u64, row_ids.value(i));
            got.push(ids.value(i));
        }
        got.sort_unstable();
        assert_eq!(
            got,
            (0..admit_below).collect::<Vec<_>>(),
            "search-path and recovery-path rows must together cover every admitted id"
        );
    }

    /// On a STABLE-ROW-ID dataset the covered null-vector recovery must resolve stable row
    /// ids through the row-id index before taking covering payload -- feeding them to an
    /// address-space take (`frag = id >> 32`, `offset = id`) reads the wrong physical row or
    /// errors once stable id != physical address. A second fragment makes the two diverge:
    /// its rows have addresses `(1 << 32) | offset` but small monotonic stable ids.
    #[tokio::test]
    async fn test_ivf_covered_recovers_null_vector_stable_row_ids() {
        use arrow_array::types::{Int32Type, UInt64Type};
        use arrow_array::{Int32Array, RecordBatchIterator};
        use arrow_buffer::NullBuffer;

        const INDEX: &str = "vec_idx";
        let dim = 4i32;
        let frag0 = 10usize; // fragment 0: ids 0..10, all indexed
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        let make_batch = |ids: Vec<i32>, null_rows: &[i32]| {
            // Two well-separated clusters. Recovery is only reachable while
            // `minimum_nprobes < maximum_nprobes`, and `early_pruning` raises the minimum to
            // the number of centroids within `dists[0] * 7.0` (k = 10). Points strung along
            // a line leave those centroids only ~7.5x apart -- a margin thin enough that a
            // different kmeans outcome silently disables the path this test exists to cover.
            let values: Vec<f32> =
                clustered_vector_values(ids.iter().map(|id| *id as usize), dim, 2);
            let validity: Vec<bool> = ids.iter().map(|id| !null_rows.contains(id)).collect();
            let vector = FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim,
                Arc::new(Float32Array::from(values)),
                Some(NullBuffer::from(validity)),
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(vector)],
            )
            .unwrap()
        };

        // Fragment 0: ids 0..10, all with vectors.
        let batch0 = make_batch((0..frag0 as i32).collect(), &[]);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch0)], schema.clone()),
            "memory://covered_stable_null_recover",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: frag0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Fragment 1 (append): ids 10..14; 10 and 11 have NULL vectors (no index entry).
        // Their stable ids (10, 11) are far smaller than their physical addresses
        // ((1 << 32) | 0/1), so an address-space take of id 10 lands at fragment 0 offset 10,
        // which is out of range (fragment 0 has offsets 0..9) -- the bug.
        let batch1 = make_batch((frag0 as i32..frag0 as i32 + 4).collect(), &[10, 11]);
        dataset
            .append(RecordBatchIterator::new([Ok(batch1)], schema.clone()), None)
            .await
            .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        // A btree on id makes `id >= 10` a bounded, selective prefilter.
        let scalar = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(&["id"], IndexType::BTree, None, &scalar, false)
            .await
            .unwrap();

        // Admit ids 10..14: 12,13 come from the search path, 10,11 from the recovery path.
        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 10).unwrap();
        // Recovery matches the non-covered shortcut, which needs unsearched
        // partitions to exist (min < max nprobes); all-partitions-searched queries
        // return only found rows on both covered and non-covered indexes.
        scan.minimum_nprobes(1);
        scan.filter("id >= 10").unwrap();
        scan.prefilter(true);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 4, "all admitted rows must be returned");
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' must be emitted")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        for i in 0..ids.len() {
            // id == stable row id == _rowid, so a correctly-resolved covered value matches.
            assert_eq!(
                ids.value(i) as u64,
                row_ids.value(i),
                "covered id must be the row's true value, not an address-space misread"
            );
        }
        let mut got: Vec<i32> = ids.values().to_vec();
        got.sort_unstable();
        assert_eq!(got, vec![10, 11, 12, 13]);
    }

    /// The covered recovery takes its payload with `MissingRowPolicy::Ignore`, which silently
    /// returns fewer rows than requested when an id no longer resolves. A stale prefilter can
    /// list such ids: a scalar index built over a fragment that a later delete removed keeps
    /// emitting that fragment's rows, and `create_deletion_mask_impl` produces no mask at all
    /// when every fragment in the VECTOR index's bitmap is intact. Pairing the requested ids
    /// with the returned payload then fails with "all columns in a record batch must have the
    /// same length" -- a query the same dataset answers fine without covering.
    ///
    /// Scope: this covers the *unresolvable-id* shortfall only -- the fragment is gone, so
    /// `get_row_addrs` drops the ids before the read. The other way a take can come up short --
    /// an id that resolves to a live address whose row is tombstoned -- is not exercised here,
    /// and is not reachable.
    ///
    /// Note those are two different masks, despite the shared name. The one above is
    /// `create_deletion_mask_impl`'s, scoped to the vector index's bitmap, and it can indeed be
    /// absent. The one that makes tombstoned rows unreachable is `DatasetPreFilter`'s
    /// `deleted_ids`, which is intersected in separately and spans every fragment in the manifest,
    /// so
    /// it holds no matter which `PreFilterSource` produced the ids.
    #[tokio::test]
    async fn test_covered_recovery_tolerates_unresolvable_prefilter_ids() {
        use arrow_array::{Int32Array, RecordBatchIterator};

        let dim = 4i32;
        let frag0 = 16usize;
        let extra = 3usize;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));
        // Two separated clusters so `early_pruning` leaves an unsearched partition and the
        // late-search shortcut (which arms the covered recovery) is reachable.
        let make_batch = |ids: Vec<i32>| {
            let values: Vec<f32> =
                clustered_vector_values(ids.iter().map(|id| *id as usize), dim, 2);
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(
                        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim)
                            .unwrap(),
                    ),
                ],
            )
            .unwrap()
        };

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(
                [Ok(make_batch((0..frag0 as i32).collect()))],
                schema.clone(),
            ),
            "memory://covered_recovery_stale_prefilter",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: frag0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Covered index over fragment 0 only.
        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Fragment 1, then a scalar index spanning BOTH fragments.
        let ids: Vec<i32> = (frag0 as i32..(frag0 + extra) as i32).collect();
        dataset
            .append(
                RecordBatchIterator::new([Ok(make_batch(ids))], schema.clone()),
                None,
            )
            .await
            .unwrap();
        let scalar = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(&["id"], IndexType::BTree, None, &scalar, false)
            .await
            .unwrap();

        // Drop fragment 1 entirely. The BTree still lists its ids, so the prefilter admits
        // rows the row-id index can no longer resolve.
        dataset.delete("id >= 16").await.unwrap();

        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 10).unwrap();
        scan.minimum_nprobes(1);
        scan.filter("id >= 15").unwrap();
        scan.prefilter(true);
        scan.project(&["id"]).unwrap();

        let batch = scan
            .try_into_batch()
            .await
            .expect("covered query must tolerate prefilter ids that no longer resolve");
        let ids = batch["id"].as_primitive::<arrow_array::types::Int32Type>();
        let got: Vec<i32> = ids.values().to_vec();
        assert_eq!(
            got,
            vec![15],
            "only the surviving admitted row may come back; ids from the deleted fragment \
             must be dropped, not paired with mismatched payload"
        );
    }

    /// The covered end-of-stream recovery re-emits every prefilter row the search did not
    /// emit. `DatasetPreFilter` produces a bounded ALLOW LIST from deletions alone on a
    /// stable-row-id dataset, so that recovery fires on queries carrying NO filter at all
    /// (`PreFilterSource::None`). Tracking the emitted row ids must therefore not be gated
    /// on a prefilter source being present -- otherwise the "already emitted" set is empty
    /// and every live row is emitted a second time with INFINITY distance.
    #[tokio::test]
    async fn test_covered_unfiltered_query_does_not_duplicate_rows() {
        use arrow_array::types::{Int32Type, UInt64Type};
        use arrow_array::{Int32Array, RecordBatchIterator};

        const INDEX: &str = "vec_idx";
        let dim = 4i32;
        let n = 12usize;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        // Two well-separated clusters with the query sitting on top of the first: the second
        // centroid is orders of magnitude farther, so `early_pruning` leaves minimum_nprobes
        // at 1. An unsearched partition is what makes the late-search no-rows shortcut (and
        // hence the covered recovery) reachable -- with evenly spread vectors early pruning
        // raises minimum_nprobes to cover every partition and the shortcut never fires.
        let ids: Vec<i32> = (0..n as i32).collect();
        let values: Vec<f32> = clustered_vector_values(0..n, dim, 2);
        let vector = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim,
            Arc::new(Float32Array::from(values)),
            None,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids)), Arc::new(vector)],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://covered_unfiltered_no_dupes",
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // A deletion makes the stable-row-id deletion mask a bounded ALLOW LIST, so
        // `max_len()` is Some(live) even though the query below carries no filter.
        dataset.delete("id = 1").await.unwrap();
        let live = n - 1;

        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        // k > live so the "fewer than k prefilter matches" shortcut condition holds.
        scan.nearest("vector", &q, live + 9).unwrap();
        scan.minimum_nprobes(1);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let batch = scan.try_into_batch().await.unwrap();

        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        let mut seen: Vec<u64> = row_ids.values().to_vec();
        seen.sort_unstable();
        let mut distinct = seen.clone();
        distinct.dedup();
        assert_eq!(
            seen.len(),
            distinct.len(),
            "covered unfiltered query returned DUPLICATE row ids: {seen:?}"
        );
        assert_eq!(
            batch.num_rows(),
            live,
            "covered unfiltered query must return each live row exactly once"
        );

        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' must be emitted")
            .as_primitive::<Int32Type>();
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i) as u64,
                row_ids.value(i),
                "covered id must stay row-aligned"
            );
        }
        let mut got: Vec<i32> = ids.values().to_vec();
        got.sort_unstable();
        assert_eq!(
            got,
            (0..n as i32).filter(|id| *id != 1).collect::<Vec<_>>(),
            "every live row must appear exactly once"
        );
    }

    /// A covered index whose `covering_fields` a pre-covering writer cleared (prost drops
    /// unknown proto field 11 on re-serialization; `FLAG_COVERED_INDEX_METADATA` fences
    /// this off only best-effort, since released clients consult writer flags solely on
    /// the append/overwrite path) -- the degraded state -- must stay MAINTAINABLE, not just
    /// readable. Its auxiliary storage still physically carries the payload while freshly
    /// scanned rows do not, so combining the two used to fail `StorageBuilder::build`'s width
    /// check ("mismatched columns while merging vector storage batches: expected
    /// [_rowid, __pq_code, id], got [_rowid, __pq_code]"), leaving an index that could never
    /// be merge-optimized again -- recoverable only by dropping and rebuilding it.
    /// `OptimizeOptions::append()` never hit this because it does not load existing storage.
    #[tokio::test]
    async fn test_degraded_covered_index_can_still_be_optimized() {
        use crate::dataset::WriteDestination;
        use crate::dataset::transaction::Operation;
        use arrow_array::{Int32Array, RecordBatchIterator};

        const DIMS: usize = 16;
        const TOTAL: usize = 256;
        const NPART: usize = 4;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let make = |lo: i32, hi: i32| {
            let ids: Vec<i32> = (lo..hi).collect();
            let values: Vec<f32> =
                clustered_vector_values(ids.iter().map(|r| *r as usize), DIMS as i32, NPART);
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        DIMS as i32,
                    ),
                    true,
                ),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            Float32Array::from(values),
                            DIMS as i32,
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
            (schema, batch)
        };

        let (schema, batch) = make(0, TOTAL as i32);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            test_uri,
            None,
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_pq(NPART, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        // Degrade: clear the declaration, leave the payload in the auxiliary file.
        let mut cleared = dataset.load_indices_by_name("vector_idx").await.unwrap()[0].clone();
        assert!(!cleared.covering_fields.is_empty());
        cleared.covering_fields = Vec::new();
        let read_version = dataset.manifest.version;
        let mut dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::CreateIndex {
                new_indices: vec![cleared],
                removed_indices: vec![],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        // Fresh rows land in an unindexed fragment; the merge below must combine them with
        // the wider existing storage.
        let (schema2, batch2) = make(TOTAL as i32, TOTAL as i32 + 128);
        dataset
            .append(RecordBatchIterator::new([Ok(batch2)], schema2), None)
            .await
            .unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .expect("a metadata-degraded covered index must still merge-optimize");

        // The rebuilt storage is narrowed to the declaration, so it no longer carries the
        // undeclared payload -- and the index still answers queries.
        let ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        let storage = ctx
            .ivf()
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap();
        assert!(
            storage.batch().column_by_name("id").is_none(),
            "merged storage must drop payload the metadata no longer declares"
        );

        let q = Float32Array::from(vec![0.0f32; DIMS]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 5).unwrap();
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 5);
        assert!(batch.column_by_name("id").is_some());
    }

    /// `row_id` is an ordinary user column name -- `_rowid` is the reserved one -- so it is
    /// coverable. The `row_id` -> `_rowid` rename exists only for precomputed shuffle buffers
    /// (which cannot name a column `_rowid`), but it used to run on the ordinary scan path
    /// too. Once covering started projecting user columns into that scan, covering a column
    /// named `row_id` renamed it on top of the real `_rowid` from `with_row_id()` and index
    /// creation died with `Duplicate field name "_rowid" in schema`.
    #[tokio::test]
    async fn test_covering_column_named_row_id_is_supported() {
        use arrow_array::types::Int32Type;
        use arrow_array::{Int32Array, RecordBatchIterator};

        let dim = 4i32;
        let n = 128usize;

        let schema = Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));
        let values: Vec<f32> = (0..n * dim as usize).map(|i| i as f32).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://covering_named_row_id",
            None,
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["row_id".to_string()]);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .expect("a covering column named `row_id` must not collide with the virtual _rowid");

        // And the covered payload is served correctly, not confused with the virtual column.
        let q = Float32Array::from(vec![0.0f32; dim as usize]);
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 5).unwrap();
        scan.project(&["row_id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 5);
        let ids = batch["row_id"].as_primitive::<Int32Type>();
        let mut got: Vec<i32> = ids.values().to_vec();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![0, 1, 2, 3, 4],
            "covered `row_id` payload must be the user's values"
        );
    }

    /// A covered ("included") struct's payload schema is fixed at index build time. Growing
    /// that struct via `add_columns` (which commits as `Operation::Merge`) -- even an
    /// AllNulls, metadata-only child that writes no data file -- would leave the index
    /// emitting the old struct type while covered queries declare the new one, an Arrow type
    /// mismatch. The commit boundary must reject it (drop the index first), mirroring the
    /// `Project` drop/alter guard.
    #[tokio::test]
    async fn test_add_columns_child_to_covered_struct_is_rejected() {
        use arrow_array::{Int32Array, RecordBatchIterator, StructArray};
        use arrow_schema::Fields;
        use lance_file::version::LanceFileVersion;

        use crate::dataset::NewColumnTransform;

        let n = 64usize;
        let dim = 4i32;
        let meta_fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("meta", DataType::Struct(meta_fields.clone()), true),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
        ]));

        let a = Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>()));
        let meta = Arc::new(StructArray::new(meta_fields, vec![a], None));
        let values: Vec<f32> = (0..n * dim as usize).map(|i| i as f32 + 1.0).collect();
        let vector = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim).unwrap(),
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![meta, vector]).unwrap();

        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema.clone()),
            "memory://covered_struct_add_child",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let mut params = VectorIndexParams::ivf_flat(2, DistanceType::L2);
        params.covering_columns(vec!["meta".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Add child `meta.b` -- merges into the covered struct, changing its type.
        let add =
            NewColumnTransform::AllNulls(Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "meta",
                DataType::Struct(Fields::from(vec![Field::new("b", DataType::Int32, true)])),
                true,
            )])));
        let err = dataset
            .add_columns(add, None, None)
            .await
            .expect_err("growing a covered struct's subtree must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("included") && msg.contains("vec_idx"),
            "expected a covered-field rejection naming the index, got: {err}"
        );
    }

    /// Read-side payoff for every vector index type: a query projecting only a covered
    /// column is satisfied from the index -- no `TakeExec` against the base table --
    /// with row-aligned values and sane recall.
    ///
    /// The fixture offsets `id` away from `_rowid` on purpose. `generate_test_dataset`
    /// starts ids at 0, so `id == _rowid` for every row -- and under that fixture the
    /// row-alignment assertion below is satisfied by any defect that serves the row id
    /// where the covered value belongs, or the reverse. That is exactly the confusion a
    /// storage batch ordered `[covering..., _rowid, code]` invites, and it is silent: a
    /// `UInt64` covering column substituted for `_rowid` downcasts cleanly. The offset
    /// separates the two columns for all seven quantizer families.
    #[rstest]
    #[case::pq(VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2))]
    #[case::sq(VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        SQBuildParams::default()
    ))]
    // 16 sub-vectors, not the 4 the plain `pq` case uses: at DIM=32 that is 2 dimensions
    // per codebook instead of 8. HNSW *builds* its graph from PQ distances, so a coarse
    // quantizer degrades construction and search together, and on this uniform-random
    // fixture the true top-10 is full of near-ties -- enough that rounding differences
    // between NEON and AVX reorder it, so the recall gate read 0.4 on aarch64 while
    // passing on x86. Plain `pq` scans partitions exhaustively and is unaffected, which
    // is why only this case needs the extra fidelity. Same reasoning as the `rq` case.
    #[case::hnsw_pq(VectorIndexParams::with_ivf_hnsw_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default(),
        PQBuildParams::new(16, 8)
    ))]
    #[case::hnsw_sq(VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default(),
        SQBuildParams::default()
    ))]
    // RQ uses 5 bits: 1-bit RaBitQ quantization is too coarse to clear the recall
    // gate on this random data without a refine (which would re-add the take).
    #[case::rq(VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        RQBuildParams::with_rotation_type(5, RQRotationType::Fast)
    ))]
    #[case::flat(VectorIndexParams::ivf_flat(4, DistanceType::L2))]
    #[case::hnsw_flat(VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default()
    ))]
    #[tokio::test]
    async fn test_covered_projection_skips_take(#[case] mut params: VectorIndexParams) {
        const INDEX_NAME: &str = "vector_idx";
        // Disjoint from the row id range (0..NUM_ROWS), so no covered `id` can coincide
        // with any row id.
        const ID_OFFSET: u64 = 1_000_000;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (batch, schema) =
            generate_batch::<Float32Type>(NUM_ROWS, Some(ID_OFFSET), 0.0..1.0, false);
        let vectors = Arc::new(
            batch
                .column_by_name("vector")
                .unwrap()
                .as_fixed_size_list()
                .clone(),
        );
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        // The HNSW search beam defaults to `k + k / 2` = 15, which on this uniform-random
        // fixture leaves the recall gate below a coin flip for the coarsest quantizer:
        // `hnsw_pq` returned 0.4 on ~2.5% of runs. The gate is a sanity check on take
        // elision, not a measurement of beam width, so give the graph a beam wide enough
        // that the assertion tests what it names. Ignored by the non-HNSW cases.
        scan.ef(100);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should not require a TakeExec; plan was:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' column must be emitted")
            .as_primitive::<UInt64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        assert_eq!(ids.len(), 10, "should return k=10 rows");
        // Single-fragment, step-id dataset => id == ID_OFFSET + row offset == ID_OFFSET +
        // _rowid, so a correctly covered id is the row id plus the offset for every
        // returned row (row-aligned, not stale). The offset also means a row id sourced
        // from the covering column instead of `_rowid` cannot satisfy this.
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i),
                row_ids.value(i) + ID_OFFSET,
                "covered id must match the row's true id (row {i})"
            );
        }

        // Take elision is worthless if the covered search returns the wrong neighbors,
        // so also gate on recall against brute-force ground truth. All 4 partitions are
        // probed, so 0.5 sits far below any quantizer's real recall on this data.
        let returned: HashSet<u64> = row_ids.values().iter().copied().collect();
        let truth = ground_truth(&dataset, "vector", q, 10, DistanceType::L2).await;
        let recall = truth.intersection(&returned).count() as f32 / truth.len() as f32;
        assert!(
            recall >= 0.5,
            "covered recall {recall} < 0.5 (returned {returned:?}, truth {truth:?})"
        );
    }

    /// The most likely real query shape: a projection mixing a covered column
    /// with an uncovered one. `filtered_read.rs` re-subtracts the projection
    /// against the covered stream's schema, so `id` must not be re-fetched
    /// while `extra` -- a column the index does not carry -- still goes
    /// through a `TakeExec`. Every other covered test projects only-covered
    /// or only-uncovered columns and would not catch a regression here.
    #[tokio::test]
    async fn test_ivf_pq_covered_partial_projection() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        // 16 sub-vectors, not 4: at DIM=32 that is 2 dimensions per codebook instead of
        // 8. This fixture is uniform-random, so the true top-10 is full of near-ties, and
        // at 4 sub-vectors the PQ error is large enough that NEON-vs-AVX rounding reorders
        // them -- the recall gate below read 0.4 on aarch64 while passing on x86. The gate
        // is a sanity check on take elision, not a measurement of quantizer accuracy.
        let mut params = VectorIndexParams::ivf_pq(4, 8, 16, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // A column the index does not cover; the take path must still fetch it.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "extra".to_string(),
                    "CAST(id AS BIGINT) + 1000".to_string(),
                )]),
                None,
                None,
            )
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id", "extra"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("LanceRead"),
            "uncovered column 'extra' must still require a TakeExec; plan was:\n{plan}"
        );
        assert!(
            plan.contains("projection=[extra]"),
            "the take must fetch only the uncovered column, not re-fetch the \
             already-covered 'id'; plan was:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10, "k=10 should return 10 rows");
        let ids = batch["id"].as_primitive::<UInt64Type>();
        let extras = batch["extra"].as_primitive::<arrow_array::types::Int64Type>();
        for (id, extra) in ids.values().iter().zip(extras.values().iter()) {
            assert_eq!(
                *extra,
                *id as i64 + 1000,
                "uncovered 'extra' must match the base table, not a stale/misaligned value"
            );
        }

        // Take elision only matters if the covered result is correct.
        let returned: HashSet<u64> = batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect();
        let truth = ground_truth(&dataset, "vector", q, 10, DistanceType::L2).await;
        let recall = truth.intersection(&returned).count() as f32 / truth.len() as f32;
        assert!(
            recall >= 0.5,
            "covered IVF_PQ recall {recall} < 0.5 (returned {returned:?}, truth {truth:?})"
        );
    }

    /// A covering declaration is not proof that every segment already stores the payload.
    /// This models a transitional distributed build: both segments carry the same logical
    /// declaration, but the new segment was produced by a writer that emitted only the
    /// vector-index storage. Commit must preserve that declaration, and planning must see
    /// the missing physical capability and fetch `id` from the base table for all results.
    #[tokio::test]
    async fn test_declared_but_physically_absent_covering_field_falls_back_to_take() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut covered_params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        covered_params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &covered_params,
                true,
            )
            .await
            .unwrap();

        let original_indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(original_indices.len(), 1);
        let declared_fields = original_indices[0].fields.clone();
        let declared_covering_fields = original_indices[0].covering_fields.clone();

        let original_fragment_ids: HashSet<u32> = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id() as u32)
            .collect();
        // PQ training on the new segment needs >= 256 rows (2^8 codes).
        append_dataset::<Float32Type>(&mut dataset, 300, 0.0..1.0).await;
        let new_fragment_ids: Vec<u32> = dataset
            .get_fragments()
            .into_iter()
            .map(|f| f.id() as u32)
            .filter(|id| !original_fragment_ids.contains(id))
            .collect();
        assert!(!new_fragment_ids.is_empty(), "append should add fragments");

        // Build a segment for just the new fragments, without covering.
        let uncovered_params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        let mut transitional_segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &uncovered_params)
            .name(INDEX_NAME.to_string())
            .fragments(new_fragment_ids)
            .replace(true)
            .execute_uncommitted()
            .await
            .unwrap();
        assert!(
            transitional_segment.covering_fields.is_empty(),
            "the new segment must genuinely be uncovered for this repro"
        );

        // The logical declaration is intentionally independent of the payload written by
        // this segment. Do not rewrite its physical auxiliary storage.
        transitional_segment.fields = declared_fields;
        transitional_segment.covering_fields = declared_covering_fields;
        dataset
            .commit_existing_index_segments(INDEX_NAME, "vector", vec![transitional_segment])
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("LanceRead"),
            "one segment cannot physically serve the declared 'id', so the query must use a \
             base-table take; plan was:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        let ids = batch["id"].as_primitive::<UInt64Type>();
        let projection = crate::dataset::ProjectionRequest::from_columns(["id"], dataset.schema());
        let truth = dataset
            .take_rows(row_ids.values(), projection)
            .await
            .unwrap();
        let truth_ids = truth["id"].as_primitive::<UInt64Type>();
        assert_eq!(
            ids, truth_ids,
            "fallback values must come from the base table"
        );
    }

    /// A declaration may be wider than the physical payload without disabling the part
    /// storage can prove. The index carries `id`; `tag` is added to the declaration only.
    /// Planning must keep `id` covered and take exactly `tag` from the base table.
    #[tokio::test]
    async fn test_physical_covering_subset_serves_only_proven_columns() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(
                    "tag".to_string(),
                    "CAST(id AS BIGINT) + 1000".to_string(),
                )]),
                None,
                None,
            )
            .await
            .unwrap();

        let id_field_id = dataset.schema().field("id").unwrap().id;
        let tag_field_id = dataset.schema().field("tag").unwrap().id;
        let mut params = VectorIndexParams::ivf_flat(4, DistanceType::L2);
        params.covering_columns(vec!["id".to_string()]);
        let mut segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name(INDEX_NAME.to_string())
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(segment.covering_fields, vec![id_field_id]);

        // Widen only the logical dependency. The auxiliary storage remains physically
        // capable of serving `id` and has no `tag` column.
        segment.fields.push(tag_field_id);
        segment.covering_fields.push(tag_field_id);
        dataset
            .commit_existing_index_segments(INDEX_NAME, "vector", vec![segment])
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.project(&["id", "tag"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("LanceRead") && plan.contains("projection=[tag]"),
            "only the physically absent 'tag' should use a base-table take; plan was:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"].as_primitive::<UInt64Type>();
        let tags = batch["tag"].as_primitive::<arrow_array::types::Int64Type>();
        for (id, tag) in ids.values().iter().zip(tags.values()) {
            assert_eq!(*tag, *id as i64 + 1000);
        }
    }

    /// Same payoff but forcing the BATCH/late-search path: `k` larger than one
    /// partition makes the initial `minimum_nprobes` sweep under-fill, so
    /// `late_search` (the run_prepared batch path) fires. Covering must hold there too.
    ///
    /// Four well-separated clusters, not uniform-random data: on uniform-random
    /// data at this `k` (500, i.e. `k > 10`), `early_pruning`'s generous 81x
    /// factor (`knn.rs`) makes the pruning heuristic select every partition
    /// regardless of the caller's `minimum_nprobes(1)`, so `adjust_probes` bumps
    /// `minimum_nprobes` up to `maximum_nprobes` and `late_search` returns empty
    /// immediately (`max_nprobes <= min_nprobes`) -- confirmed by instrumenting
    /// `late_search` on the prior uniform-random version of this test, which
    /// logged `min_nprobes=4 max_nprobes=4` despite the explicit `min=1,max=4`
    /// request. With separated clusters the nearest centroid is far closer than
    /// the rest, so the heuristic keeps `minimum_nprobes` at 1 and `late_search`
    /// genuinely runs.
    #[tokio::test]
    async fn test_ivf_pq_covered_projection_batch_path() {
        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = NUM_ROWS / NUM_CLUSTERS;
        let offsets = [0.0f32, 1000.0, 2000.0, 3000.0];

        let mut rng = StdRng::seed_from_u64(7);
        let mut ids = Vec::with_capacity(NUM_ROWS);
        let mut values = Vec::with_capacity(NUM_ROWS * DIM);
        for (cluster_idx, offset) in offsets.iter().enumerate() {
            for row in 0..ROWS_PER_CLUSTER {
                ids.push((cluster_idx * ROWS_PER_CLUSTER + row) as u64);
                for dim in 0..DIM {
                    let base = if dim == 0 { *offset } else { 0.0 };
                    let noise = (rng.random::<f32>() - 0.5) * 0.02;
                    values.push(base + noise);
                }
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(NUM_CLUSTERS, centroids).unwrap();
        let mut params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams {
                num_bits: 8,
                num_sub_vectors: 4,
                max_iters: 2,
                ..Default::default()
            },
        );
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // A query at cluster 0's exact center: its nearest centroid is a near-zero
        // distance away, the rest are ~1000+ away, so `early_pruning` keeps
        // `minimum_nprobes` at 1 rather than escalating it to `maximum_nprobes`.
        let q = Float32Array::from(vec![0.0f32; DIM]);
        let mut scan = dataset.scan();
        // k >> one partition's rows (NUM_ROWS=512 / 4 parts) forces late expansion.
        scan.nearest("vector", &q, 500).unwrap();
        scan.minimum_nprobes(1);
        scan.maximum_nprobes(NUM_CLUSTERS);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should not require a TakeExec (batch path); plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert!(
            batch.num_rows() > ROWS_PER_CLUSTER,
            "late search should have expanded beyond the nearest partition's {ROWS_PER_CLUSTER} rows, got {}",
            batch.num_rows()
        );
        // Row alignment, not just presence: ids are assigned in write order over a single
        // fragment, so a correctly covered `id` equals its row id on every returned row.
        let ids = batch["id"].as_primitive::<UInt64Type>();
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i),
                row_ids.value(i),
                "covered payload must stay row-aligned through the batch path"
            );
        }
    }

    /// A covered query combined with a selective prefilter (scalar index +
    /// `prefilter`) must still skip the TakeExec and return correct, filtered
    /// rows. Exercises the covering read path under a scalar-index prefilter.
    #[tokio::test]
    async fn test_ivf_pq_covered_with_scalar_prefilter() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // A scalar index on `id` turns the prefilter into a bounded AllowList.
        let scalar_params = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_btree".to_string()),
                &scalar_params,
                true,
            )
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.filter("id < 5").unwrap();
        scan.prefilter(true);
        scan.project(&["id"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should not require a TakeExec even with a prefilter; plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert!(batch.column_by_name("id").is_some());
        // Every id in [0, 5) is present in the dataset, so the prefilter must
        // return exactly 5 rows -- not zero, which would make the loop below
        // vacuously true regardless of whether filtering actually worked.
        assert_eq!(batch.num_rows(), 5);
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>();
        for v in ids.values() {
            assert!(
                *v < 5,
                "all returned rows must satisfy the prefilter id < 5"
            );
        }
    }

    /// A covered index must keep its covering columns through optimize/merge.
    /// After appending rows and merging them into the index, the merged
    /// auxiliary storage must still carry the included column, and a covered
    /// projection must still skip the take. The covering columns are threaded
    /// through the incremental optimize pipeline -- the unindexed-fragment
    /// shuffle, the partition-split reshuffle, and the partition-join
    /// reassignment -- as passenger columns (re-gathered by row id).
    #[rstest]
    #[case::pq(VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2))]
    #[case::sq(VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        SQBuildParams::default()
    ))]
    #[case::hnsw_pq(VectorIndexParams::with_ivf_hnsw_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default(),
        PQBuildParams::new(4, 8)
    ))]
    #[case::hnsw_sq(VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default(),
        SQBuildParams::default()
    ))]
    #[case::rq(VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        RQBuildParams::with_rotation_type(1, RQRotationType::Fast)
    ))]
    #[case::flat(VectorIndexParams::ivf_flat(4, DistanceType::L2))]
    #[case::hnsw_flat(VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default()
    ))]
    #[tokio::test]
    async fn test_covered_survives_optimize(#[case] mut params: VectorIndexParams) {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Append rows (unindexed fragments) and merge them into the index.
        append_dataset::<Float32Type>(&mut dataset, NUM_ROWS, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        // A covered projection must still be answered from the index with no take after
        // the merge. This is the real proof that the appended rows kept their covering
        // column: if the merge had dropped it, `try_into_batch` would error on the
        // schema mismatch (the exec declares `id` from `covering_fields`, but the
        // storage could not emit it).
        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.with_row_id();
        scan.project(&["id"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should skip the take after optimize; plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10);
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' column must be emitted")
            .as_primitive::<arrow_array::types::UInt64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<arrow_array::types::UInt64Type>();

        // Ground truth via an independent (non-index, base-table take) path.
        // The dataset now spans two fragments (the original write plus the
        // appended delta), so id no longer equals _rowid by construction the
        // way it does for the single-fragment `test_covered_projection_skips_take`
        // -- but the covered value must still equal the row's *true* id. This
        // catches a merge that carries the column but scrambles which row
        // its values ended up attached to (e.g. an off-by-one row-id gather).
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let projection = crate::dataset::ProjectionRequest::from_columns(["id"], dataset.schema());
        let truth = dataset.take_rows(&row_id_vec, projection).await.unwrap();
        let truth_ids = truth
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>();
        for i in 0..ids.len() {
            assert_eq!(
                ids.value(i),
                truth_ids.value(i),
                "row {i} (row_id {}): covered id {} != true id {} -- merge scrambled the \
                 covering column",
                row_ids.value(i),
                ids.value(i),
                truth_ids.value(i)
            );
        }
    }

    /// Covering must survive a *retrain* optimize. Retrain rebuilds the storage from
    /// scratch, so the covering column must be re-materialized -- otherwise the fresh
    /// files omit it while the committed metadata still advertises `covering_fields`,
    /// and a covered projection expects a column the storage cannot emit.
    #[tokio::test]
    async fn test_ivf_pq_covered_survives_retrain() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        append_dataset::<Float32Type>(&mut dataset, NUM_ROWS, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();

        // The retrained storage must re-materialize the covering column.
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let storage = ctx
            .ivf()
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap();
        assert!(
            storage.batch().column_by_name("id").is_some(),
            "retrained storage should re-materialize covering column 'id'"
        );

        // And a covered projection is still answered from the index (no take).
        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.project(&["id"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should skip the take after retrain; plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert!(batch.column_by_name("id").is_some());
    }

    /// The streaming partition-search branch (used by HNSW sub-indexes and controlled
    /// late searches) must declare the covered schema on the stream it returns, matching
    /// the widened batches it emits -- the global-heap branch already does. A stream
    /// whose declared schema disagrees with its batches is a latent hazard for any
    /// consumer that trusts the declaration.
    #[tokio::test]
    async fn test_covered_search_partitions_stream_declares_covering_schema() {
        use arrow_array::UInt32Array;
        use futures::TryStreamExt;
        use lance_index::prefilter::NoFilter;
        use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};

        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            IvfBuildParams::new(4),
            HnswBuildParams::default(),
        );
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let query = Query {
            column: "vector".to_string(),
            key: vectors.value(0),
            k: 5,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 4,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: None,
        };
        let partitions = Arc::new(UInt32Array::from(vec![0u32, 1, 2, 3]));
        let dists = Arc::new(Float32Array::from(vec![0.0f32; 4]));
        let stream = ctx
            .index
            .clone()
            .search_partitions(
                query,
                partitions,
                dists,
                0,
                4,
                Arc::new(NoFilter),
                None,
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap();

        let declared = stream.schema();
        assert!(
            declared.column_with_name("id").is_some(),
            "the covered stream must declare the covering column; declared: {declared:?}"
        );
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert!(!batches.is_empty());
        for batch in &batches {
            assert_eq!(
                batch.schema(),
                declared,
                "emitted batches must match the declared stream schema"
            );
        }
    }

    /// Index storage that physically carries covering columns while the manifest
    /// declares none (`covering_fields` empty) must still be queryable: the extra
    /// columns are dropped from search batches (with a warning) instead of emitting
    /// batches wider than the plan's declared `[_distance, _rowid]` schema. This is
    /// the legacy-tolerance contract: a stable-format index file with an unexpected
    /// extra column used to be projected down with a warning, never a query failure.
    #[tokio::test]
    async fn test_undeclared_storage_covering_is_dropped_not_fatal() {
        use crate::dataset::WriteDestination;
        use crate::dataset::transaction::Operation;

        const DIMS: usize = 16;
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;
        const TOTAL: usize = NUM_CLUSTERS * ROWS_PER_CLUSTER;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Well-separated clusters, not uniform-random vectors: a query near one
        // cluster must land far outside the early-pruning heuristic's threshold
        // (`early_pruning`/`adjust_probes` in `io/exec/knn.rs`) for every other
        // cluster's centroid, so `late_search` gets a real range of partitions
        // left to search instead of the heuristic folding all of them into the
        // early, synchronous phase. Uniform-random vectors near the dataset's
        // overall centroid are roughly equidistant from every partition centroid,
        // so every partition is searched up front and `late_search` never runs --
        // silently defeating the "hard case" below (a defect recorded once
        // already on this project: a covered-ANN test with too few effectively-
        // reachable partitions passes without ever exercising the path it claims
        // to cover).
        let mut rng = StdRng::seed_from_u64(7);
        let mut ids = Vec::with_capacity(TOTAL);
        let mut values = Vec::with_capacity(TOTAL * DIMS);
        for cluster in 0..NUM_CLUSTERS {
            let center = (cluster * 1000) as f32;
            for row in 0..ROWS_PER_CLUSTER {
                ids.push((cluster * ROWS_PER_CLUSTER + row) as i32);
                for dim in 0..DIMS {
                    let base = if dim == 0 { center } else { 0.0 };
                    values.push(base + (rng.random::<f32>() - 0.5) * 0.02);
                }
            }
        }
        let ids_arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(ids));
        let vectors: ArrayRef = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIMS as i32)
                .unwrap(),
        );
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids_arr, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let mut params = VectorIndexParams::ivf_pq(NUM_CLUSTERS, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        // Re-commit the index metadata with the covering declaration cleared while the
        // storage keeps the payload column -- the state a pre-covering writer (which
        // drops the unknown `covering_fields` proto field) or a legacy index file with
        // a stray extra column would produce.
        let mut cleared = dataset.load_indices_by_name("vector_idx").await.unwrap()[0].clone();
        assert!(!cleared.covering_fields.is_empty());
        cleared.covering_fields = Vec::new();
        let read_version = dataset.manifest.version;
        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::CreateIndex {
                new_indices: vec![cleared],
                removed_indices: vec![],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();

        // Query the LAST cluster's center: its rows' ids (`3*ROWS_PER_CLUSTER..`)
        // never satisfy the `id < 3` filter used below, so the "hard case" prefilter
        // match (ids 0/1/2, in the first cluster) is guaranteed to be entirely
        // unfound by the early search of this, the nearest, cluster/partition --
        // exercising the not-found shortcut instead of incidentally short-circuiting
        // because the early search already found every matching row.
        let mut q_values = vec![0.0f32; DIMS];
        q_values[0] = ((NUM_CLUSTERS - 1) * 1000) as f32;
        let q = Float32Array::from(q_values);

        // The index is no longer covering, so the query takes `id` from the base
        // table; the storage's undeclared payload column must be silently dropped
        // from the search batches, not fail the query.
        let mut scan = dataset.scan();
        scan.nearest("vector", &q, 5).unwrap();
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 5);
        assert!(batch.column_by_name("id").is_some());

        // The hard case: a bounded selective prefilter over unsearched partitions makes
        // `late_search` emit its 2-column not-found shortcut batch into the same stream as
        // the search batches. Unnarrowed, those search batches still carry the undeclared
        // payload column, so the widths disagree where DataFusion's TopK concatenates them
        // -- `interleave_record_batch` reads every batch through the FIRST batch's schema
        // and panics with an out-of-bounds column index.
        // Repeated because the unnarrowed failure is order-dependent: it only panics when
        // TopK happens to store a wide batch first, which is roughly 4 runs in 5. One
        // iteration would make this test pass intermittently against a real regression.
        for attempt in 0..5 {
            let mut scan = dataset.scan();
            scan.nearest("vector", &q, 5).unwrap();
            scan.minimum_nprobes(1);
            scan.maximum_nprobes(NUM_CLUSTERS);
            scan.filter("id < 3").unwrap();
            scan.prefilter(true);
            scan.project(&["id"]).unwrap();
            let batch = scan.try_into_batch().await.unwrap();
            assert_eq!(
                batch.num_rows(),
                3,
                "all prefilter-matched rows must come back (attempt {attempt})"
            );
            let ids = batch["id"].as_primitive::<arrow_array::types::Int32Type>();
            let mut got: Vec<i32> = ids.values().to_vec();
            got.sort_unstable();
            assert_eq!(
                got,
                vec![0, 1, 2],
                "covered payload must be taken from the base table once the declaration is \
                 gone (attempt {attempt})"
            );
        }
    }

    /// A covered index that searches zero partitions (empty heap)
    /// must still emit the covered schema `[_distance, _rowid, <included>]`, not
    /// the bare `[_distance, _rowid]` -- otherwise the produced batch mismatches
    /// the wider schema the exec declares from `covering_fields`. The decision is
    /// driven by the stable per-index covering schema, not the gathered `covering` batch.
    #[test]
    fn test_global_heap_to_batch_covered_empty_emits_covered_schema() {
        use arrow_schema::{DataType, Field, Schema};
        let covering = Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("id", DataType::UInt64, true),
        ]);
        // Empty heap and nothing gathered, but the query DOES want covering columns.
        let batch =
            IvfPq::global_heap_to_batch(std::collections::BinaryHeap::new(), None, Some(&covering))
                .unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(
            batch.num_columns(),
            3,
            "covered empty result must keep the covering column"
        );
        assert!(batch.column_by_name("id").is_some());

        // Ordinary (non-covered) index: bare `[_distance, _rowid]`.
        let plain =
            IvfPq::global_heap_to_batch(std::collections::BinaryHeap::new(), None, None).unwrap();
        assert_eq!(plain.num_columns(), 2);
    }

    /// A result row id that is absent from the covering buffer is an invariant
    /// break (every heap winner comes from a searched partition whose covering
    /// batch was captured). It must surface as an error -- never as row 0's
    /// covering values silently attached to an unrelated row.
    #[test]
    fn test_gather_covering_by_rowid_errors_on_missing_rowid() {
        use arrow_schema::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new("id", DataType::UInt64, true),
        ]));
        let source = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(UInt64Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        // Present row ids gather fine.
        let ok =
            gather_covering_columns_by_row_id(&source, &UInt64Array::from(vec![3, 1])).unwrap();
        assert_eq!(ok.len(), 1);
        let (_, values) = &ok[0];
        let values = values.as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(values.values(), &[30, 10]);

        // Row id 99 is absent from the covering source: must be an error.
        let err = gather_covering_columns_by_row_id(&source, &UInt64Array::from(vec![2, 99]))
            .expect_err("missing row id must error, not return unrelated values");
        assert!(err.to_string().contains("99"), "got: {err}");
    }

    /// A covered index must keep its covering columns through compaction, which
    /// rewrites fragments and REMAPS the index's row ids. Delete rows to force a
    /// compaction+remap; the remapped storage must still carry the covering
    /// column and covered projections must still skip the take.
    #[tokio::test]
    async fn test_ivf_pq_covered_survives_compaction() {
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Delete rows so compaction rewrites fragments and remaps the index.
        dataset.delete("id < 100").await.unwrap();
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        // The remapped partition storage must still carry the covering column.
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let storage = ctx
            .ivf()
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap();
        assert!(
            storage.batch().column_by_name("id").is_some(),
            "remapped storage should keep covering column 'id'"
        );

        // Covered projection still skips the take, and deleted rows are gone.
        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.project(&["id"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should skip the take after compaction; plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        // Plenty of non-deleted rows remain for k=10 to fill; guard against an
        // empty result, which would make the loop below vacuously true.
        assert_eq!(batch.num_rows(), 10);
        let ids = batch
            .column_by_name("id")
            .expect("covered projection should return id")
            .as_primitive::<arrow_array::types::UInt64Type>();
        for v in ids.values() {
            assert!(*v >= 100, "deleted rows (id < 100) must not be returned");
        }
    }

    /// A covered index with TWO covering columns of different arrow types must
    /// survive compaction (which remaps the index's row ids). Every other
    /// covering test in this suite uses a single `UInt64` covering column
    /// (`id`), which hid two bugs in the default `QuantizerStorage::remap`
    /// (used by flat, HNSW_FLAT and SQ, unlike PQ/RQ which override it):
    /// it read the row id column by *position* (`column(0)`), which is only
    /// ever correct by coincidence for a single leading `UInt64` covering
    /// column. With a non-`UInt64` covering column first, the downcast
    /// panics; with a `UInt64` one first (as `id` always was), it silently
    /// overwrites the storage's real row ids with the covering column's
    /// values. Covering column order here is deliberately `[tag, id]` so
    /// `tag` (`Int32`) lands first in the storage batch, reproducing the
    /// panicking variant, while `id` (`UInt64`) still covers the silent one.
    #[rstest]
    #[case::flat(VectorIndexParams::ivf_flat(4, DistanceType::L2))]
    #[case::hnsw_flat(VectorIndexParams::ivf_hnsw(
        DistanceType::L2,
        IvfBuildParams::new(4),
        HnswBuildParams::default()
    ))]
    #[case::sq(VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        SQBuildParams::default()
    ))]
    #[tokio::test]
    async fn test_covered_survives_compaction_multiple_covering_columns(
        #[case] mut params: VectorIndexParams,
    ) {
        use arrow_array::Int32Array;
        use arrow_array::types::Int32Type;

        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let ids = UInt64Array::from_iter_values(0..NUM_ROWS as u64);
        // Distinct from `id` (negative, offset) so a bug that swaps or
        // misaligns the two covering columns cannot pass by coincidence.
        let tags = Int32Array::from_iter_values((0..NUM_ROWS as i32).map(|i| -i - 1));
        let vectors = generate_random_array_with_range::<Float32Type>(NUM_ROWS * DIM, 0.0..1.0);
        let fsl =
            normalize_fsl(&FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap())
                .unwrap();
        let query_vector = fsl.value(0);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("tag", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(tags), Arc::new(fsl)],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        params.covering_columns(vec!["tag".to_string(), "id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Delete rows so compaction rewrites fragments and remaps the index.
        dataset.delete("id < 100").await.unwrap();
        compact_after_deletions(&mut dataset).await;

        let q = query_vector.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["tag", "id"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['tag', 'id'] should skip the take after compaction; plan:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        // Plenty of non-deleted rows remain for k=10 to fill; guard against an
        // empty result, which would make the checks below vacuously true.
        assert_eq!(batch.num_rows(), 10);

        let returned_ids = batch
            .column_by_name("id")
            .expect("covered projection should return id")
            .as_primitive::<UInt64Type>();
        let returned_tags = batch
            .column_by_name("tag")
            .expect("covered projection should return tag")
            .as_primitive::<Int32Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        for v in returned_ids.values() {
            assert!(*v >= 100, "deleted rows (id < 100) must not be returned");
        }

        // Ground truth via an independent (non-index, base-table take) path,
        // for BOTH covering columns: a corrupted or misaligned remap of
        // either column must be caught, not just detected by its presence.
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let projection =
            crate::dataset::ProjectionRequest::from_columns(["id", "tag"], dataset.schema());
        let truth = dataset.take_rows(&row_id_vec, projection).await.unwrap();
        let truth_ids = truth
            .column_by_name("id")
            .unwrap()
            .as_primitive::<UInt64Type>();
        let truth_tags = truth
            .column_by_name("tag")
            .unwrap()
            .as_primitive::<Int32Type>();
        for i in 0..returned_ids.len() {
            assert_eq!(
                returned_ids.value(i),
                truth_ids.value(i),
                "row {i} (row_id {}): covered id != true id after compaction \
                 (multi-covering-column case)",
                row_ids.value(i)
            );
            assert_eq!(
                returned_tags.value(i),
                truth_tags.value(i),
                "row {i} (row_id {}): covered tag != true tag after compaction \
                 (multi-covering-column case)",
                row_ids.value(i)
            );
        }
    }

    /// A current writer stores its physical covering columns in declaration order. The
    /// reader now verifies this independently and falls back if a transitional segment
    /// differs, but a newly built index should retain the optimization for every declared
    /// column rather than relying on that safety net.
    ///
    /// Declaration order here is `[payload, price]`: the reverse of both the dataset's
    /// field-id order (`id`, `price`, `payload`, `vector`) and the projection order the
    /// query asks for. An implementation that sorted by field id, walked the schema, or
    /// echoed the request order would produce `[price, payload]` and fail.
    ///
    /// The two covering columns carry different Arrow types and values that are
    /// deliberately disjoint from the row ids and from each other (`price` is negative,
    /// `payload` is a string): a fixture where a covering value equals its row id makes a
    /// positional-for-by-name substitution invisible.
    #[tokio::test]
    async fn test_covering_declaration_and_storage_agree_on_order() {
        use crate::index::covering::effective_covering;
        use arrow_array::{Int32Array, StringArray, types::Int32Type};
        use lance_index::vector::storage::VectorStore;

        const INDEX_NAME: &str = "vector_idx";
        const DIMS: usize = 16;
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;
        const TOTAL: usize = NUM_CLUSTERS * ROWS_PER_CLUSTER;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Well-separated clusters, one per partition, so the ANN path is genuinely
        // exercised rather than degenerating into a single-partition scan.
        let mut flat = Vec::with_capacity(TOTAL * DIMS);
        for row in 0..TOTAL {
            let center = (row / ROWS_PER_CLUSTER) as f32 * 50.0;
            for d in 0..DIMS {
                flat.push(center + (row % ROWS_PER_CLUSTER) as f32 * 0.001 + d as f32 * 0.0001);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), DIMS as i32).unwrap(),
        );
        let ids = Arc::new(UInt64Array::from_iter_values(0..TOTAL as u64));
        let prices = Arc::new(Int32Array::from_iter_values(
            (0..TOTAL as i32).map(|i| -i - 7),
        ));
        let payloads = Arc::new(StringArray::from_iter_values(
            (0..TOTAL).map(|i| format!("p{}", i * 3 + 11)),
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("price", DataType::Int32, false),
            Field::new("payload", DataType::Utf8, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![ids, prices, payloads, vectors.clone()])
                .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        let mut params = VectorIndexParams::ivf_flat(NUM_CLUSTERS, DistanceType::L2);
        params.covering_columns(vec!["payload".to_string(), "price".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // 1. The manifest records the declaration in the order it was requested.
        let price_id = dataset.schema().field("price").unwrap().id;
        let payload_id = dataset.schema().field("payload").unwrap().id;
        assert!(
            price_id < payload_id,
            "the fixture needs declaration order to differ from field-id order"
        );
        let index_meta = dataset
            .load_indices_by_name(INDEX_NAME)
            .await
            .unwrap()
            .into_iter()
            .next()
            .expect("index metadata");
        assert_eq!(index_meta.covering_fields, vec![payload_id, price_id]);

        // 2. The read path's manifest-side resolution follows that order, whichever order
        //    the query requests its columns in.
        let requested = ["price".to_string(), "payload".to_string()];
        let declared = effective_covering(
            &index_meta.covering_fields,
            Some(&requested),
            dataset.schema(),
        )
        .unwrap();
        let declared_names: Vec<&str> = declared.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(declared_names, vec!["payload", "price"]);

        // 3. The storage-side resolution -- `covering_field_indices`, derived from each
        //    storage's `INTERNAL_COLUMNS` -- lists the same columns in the same order.
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let storage = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex")
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap();
        let storage_schema = storage.schema().clone();
        let storage_names: Vec<&str> = storage
            .covering_field_indices()
            .into_iter()
            .map(|i| storage_schema.field(i).name().as_str())
            .collect();
        assert_eq!(
            storage_names, declared_names,
            "storage order must equal declaration order; if these diverge the covered \
             values are emitted under the wrong names"
        );

        // 4. End to end: the covered values must match the base table, matched BY NAME.
        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(NUM_CLUSTERS);
        scan.with_row_id();
        scan.project(&["price", "payload"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "the covered query must go through the index; plan was:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10);

        let row_ids: Vec<u64> = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let truth = dataset
            .take_rows(
                &row_ids,
                crate::dataset::ProjectionRequest::from_columns(
                    ["price", "payload"],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let got_prices = batch
            .column_by_name("price")
            .unwrap()
            .as_primitive::<Int32Type>();
        let truth_prices = truth
            .column_by_name("price")
            .unwrap()
            .as_primitive::<Int32Type>();
        let got_payloads = batch.column_by_name("payload").unwrap().as_string::<i32>();
        let truth_payloads = truth.column_by_name("payload").unwrap().as_string::<i32>();
        for (i, row_id) in row_ids.iter().enumerate() {
            assert_eq!(
                got_prices.value(i),
                truth_prices.value(i),
                "row {i} (row_id {row_id}): covered price landed under the wrong name"
            );
            assert_eq!(
                got_payloads.value(i),
                truth_payloads.value(i),
                "row {i} (row_id {row_id}): covered payload landed under the wrong name"
            );
        }
    }

    /// Counts the cache hit/miss signal `IVFIndex::load_partition` reports, which is how
    /// the tests below observe whether two searches shared one partition entry.
    #[derive(Default)]
    struct CacheCountingMetrics {
        hits: AtomicUsize,
        misses: AtomicUsize,
    }

    impl lance_index::metrics::MetricsCollector for CacheCountingMetrics {
        fn record_parts_loaded(&self, _num_parts: usize) {}
        fn record_index_loads(&self, _num_indexes: usize) {}
        fn record_comparisons(&self, _num_comparisons: usize) {}
        fn record_index_cache_hits(&self, num_hits: usize) {
            self.hits.fetch_add(num_hits, Ordering::Relaxed);
        }
        fn record_index_cache_misses(&self, num_misses: usize) {
            self.misses.fetch_add(num_misses, Ordering::Relaxed);
        }
    }

    /// A covered IVF_FLAT fixture with one well-separated cluster per partition and two
    /// covering columns of different Arrow types. The covering values are deliberately
    /// disjoint from the row ids (`price` is negative, `payload` is a string) -- a fixture
    /// where a covering value equals its row id hides a positional-for-by-name
    /// substitution. Row id equals row offset, so the caller can compute ground truth
    /// against `vectors` directly.
    async fn covered_flat_fixture(
        uri: &str,
        index_name: &str,
        num_clusters: usize,
        rows_per_cluster: usize,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        use arrow_array::{Int32Array, StringArray};

        const DIMS: usize = 16;
        let total = num_clusters * rows_per_cluster;

        let mut flat = Vec::with_capacity(total * DIMS);
        for row in 0..total {
            let center = (row / rows_per_cluster) as f32 * 50.0;
            for d in 0..DIMS {
                flat.push(center + (row % rows_per_cluster) as f32 * 0.001 + d as f32 * 0.0001);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), DIMS as i32).unwrap(),
        );
        let prices = Arc::new(Int32Array::from_iter_values(
            (0..total as i32).map(|i| -i - 7),
        ));
        let payloads = Arc::new(StringArray::from_iter_values(
            (0..total).map(|i| format!("p{}", i * 3 + 11)),
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Int32, false),
            Field::new("payload", DataType::Utf8, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![prices, payloads, vectors.clone()]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, uri, None).await.unwrap();

        let mut params = VectorIndexParams::ivf_flat(num_clusters, DistanceType::L2);
        params.covering_columns(vec!["payload".to_string(), "price".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(index_name.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        (dataset, vectors)
    }

    /// A partition loaded for search carries only the storage's own columns, so the entry
    /// it is cached under -- `IVFPartitionKey { partition_id }`, which has no covering
    /// component -- is the same entry for every query. That is what makes the key correct
    /// rather than accidentally correct: the naive narrowing (read whichever covering
    /// columns *this* query asked for) would let the first query to touch a partition
    /// decide what every later query finds there.
    ///
    /// Two searches asking for disjoint covering subsets run against the same index, in
    /// order. The second must reuse the entry the first populated, that shared entry must
    /// carry no covering column at all -- an entry holding `payload` but not `price` is
    /// exactly the unsound state ruled out here -- and both must return the partition's
    /// true nearest neighbours.
    ///
    /// The line under test is `PartitionColumns::Internal` in `load_partition_entry`.
    /// Switching it to `PartitionColumns::All` makes the covering assertions below fail.
    #[tokio::test]
    async fn test_partition_cache_entry_is_independent_of_the_query_covering_subset() {
        use lance_index::prefilter::NoFilter;
        use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;
        const K: usize = 10;

        let test_dir = TempStrDir::default();
        let (dataset, vectors) = covered_flat_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        let query_key = vectors.value(0);
        let query_for = |covering: &[&str]| Query {
            column: "vector".to_string(),
            key: query_key.clone(),
            k: K,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: NUM_CLUSTERS,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: Some(covering.iter().map(|c| c.to_string()).collect()),
        };

        let payload_metrics = CacheCountingMetrics::default();
        let payload_result = index
            .search_in_partition(
                0,
                &query_for(&["payload"]),
                Arc::new(NoFilter),
                &payload_metrics,
            )
            .await
            .unwrap();
        assert_eq!(
            (
                payload_metrics.misses.load(Ordering::Relaxed),
                payload_metrics.hits.load(Ordering::Relaxed)
            ),
            (1, 0),
            "the first search must populate the entry rather than find one already there"
        );

        let price_metrics = CacheCountingMetrics::default();
        let price_result = index
            .search_in_partition(
                0,
                &query_for(&["price"]),
                Arc::new(NoFilter),
                &price_metrics,
            )
            .await
            .unwrap();
        assert_eq!(
            (
                price_metrics.hits.load(Ordering::Relaxed),
                price_metrics.misses.load(Ordering::Relaxed)
            ),
            (1, 0),
            "the second search must reuse the first one's entry -- if it loaded its own, \
             the entries are query-specific and nothing below proves the key is sound"
        );

        // The one entry the two searches shared holds no covering column at all, so there
        // is no subset for one query to have fixed on another's behalf.
        let entry = index
            .load_partition(0, true, &NoOpMetricsCollector)
            .await
            .unwrap();
        let part = entry.as_ref();
        assert!(
            part.storage.covering_field_indices().is_empty(),
            "a cached partition must carry no covering column; it carries {:?}",
            part.storage.schema()
        );
        assert!(
            part.storage.covering_batch().unwrap().is_none(),
            "covering_batch() on a codes-only partition must report none"
        );

        // Both searches answer correctly over that shared entry. Ground truth is the
        // partition's own rows ranked by distance to the query, computed from the base
        // table rather than from the index.
        let partition_row_ids: Vec<u64> = index
            .load_partition_storage(0, PartitionColumns::All, None)
            .await
            .unwrap()
            .row_ids()
            .copied()
            .collect();
        assert!(
            partition_row_ids.len() > K,
            "partition 0 must hold more rows than k, or the ranking below is vacuous"
        );
        let query_values = query_key.as_primitive::<Float32Type>().values().to_vec();
        let mut truth: Vec<(u64, f32)> = partition_row_ids
            .iter()
            .map(|row_id| {
                let vector = vectors.value(*row_id as usize);
                let distance = vector
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .zip(query_values.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>();
                (*row_id, distance)
            })
            .collect();
        truth.sort_by(|a, b| a.1.total_cmp(&b.1));
        let expected: Vec<u64> = truth.iter().take(K).map(|(row_id, _)| *row_id).collect();

        for (label, result) in [("payload", &payload_result), ("price", &price_result)] {
            let mut got: Vec<u64> = result
                .column_by_name(ROW_ID)
                .expect("search result carries row ids")
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec();
            got.sort_unstable();
            let mut want = expected.clone();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "the {label} query's neighbours must be the partition's true nearest k"
            );
            assert!(result.column_by_name(DIST_COL).is_some());
        }

        // The covering half of "both get right answers": each query receives exactly the
        // column it projected, gathered for its survivors out of a shared entry that holds
        // neither. `price` is `-row_id - 7` and `payload` is `p{row_id * 3 + 11}`, so a
        // value taken from the wrong row -- or from the wrong column -- cannot coincide.
        for (label, result, other) in [
            ("payload", &payload_result, "price"),
            ("price", &price_result, "payload"),
        ] {
            assert!(
                result.column_by_name(other).is_none(),
                "the {label} query must not materialize {other}: a covering column the \
                 query never reads is the cost this narrowing exists to avoid, and it is \
                 invisible in results"
            );
            let row_ids = result
                .column_by_name(ROW_ID)
                .expect("search result carries row ids")
                .as_primitive::<UInt64Type>();
            let covering = result
                .column_by_name(label)
                .unwrap_or_else(|| panic!("the {label} query must materialize {label}"));
            for i in 0..row_ids.len() {
                let row_id = row_ids.value(i);
                match label {
                    "price" => assert_eq!(
                        covering
                            .as_primitive::<arrow_array::types::Int32Type>()
                            .value(i),
                        -(row_id as i32) - 7,
                        "row {i} (row_id {row_id}): gathered price belongs to another row"
                    ),
                    _ => assert_eq!(
                        covering.as_string::<i32>().value(i),
                        format!("p{}", row_id * 3 + 11),
                        "row {i} (row_id {row_id}): gathered payload belongs to another row"
                    ),
                }
            }
        }
    }

    /// A covered IVF_FLAT fixture built for the survivor gather: one well-separated cluster
    /// per partition, a **nullable** covering column, and covering values disjoint from the
    /// row ids.
    ///
    /// Both of those properties are load-bearing and no other covered fixture in this suite
    /// has them. Every other one declares its covering columns non-nullable, which is why a
    /// covered query that returned a silent NULL for every survivor stayed invisible until
    /// it was probed deliberately -- non-nullable columns turn it into a loud arrow error
    /// instead. And a fixture whose covering value equals its row id cannot tell a by-name
    /// gather from a positional one.
    ///
    /// `payload` is deliberately wide (~1 KB/row). The gather exists for the regime where
    /// the covering payload dwarfs the quantization code; at a narrow width the scattered
    /// and sequential reads cost almost the same and nothing can be measured about them.
    ///
    /// Returns the dataset and its vectors; row id equals row offset, so ground truth can
    /// be computed against `vectors` directly.
    async fn covered_gather_fixture(
        uri: &str,
        index_name: &str,
        num_clusters: usize,
        rows_per_cluster: usize,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        use arrow_array::{Int32Array, StringArray};

        const DIMS: usize = 16;
        let total = num_clusters * rows_per_cluster;

        let mut flat = Vec::with_capacity(total * DIMS);
        for row in 0..total {
            let center = (row / rows_per_cluster) as f32 * 50.0;
            for d in 0..DIMS {
                flat.push(center + (row % rows_per_cluster) as f32 * 0.001 + d as f32 * 0.0001);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), DIMS as i32).unwrap(),
        );
        // Decreasing, and offset by 1000, so `tag` is never equal to its row id.
        let tags = Arc::new(Int32Array::from_iter_values(
            (0..total as i32).map(|i| 1000 - i),
        ));
        // Every third row is NULL. With k = 10 the result always contains both a NULL and
        // a non-NULL note, so neither case is vacuous.
        let notes = Arc::new(StringArray::from_iter((0..total).map(|i| match i % 3 {
            0 => None,
            _ => Some(format!("n{}", i * 7 + 3)),
        })));
        let payloads = Arc::new(StringArray::from_iter_values(
            (0..total).map(|i| format!("{i:04}").repeat(256)),
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("tag", DataType::Int32, false),
            Field::new("note", DataType::Utf8, true),
            Field::new("payload", DataType::Utf8, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![tags, notes, payloads, vectors.clone()])
                .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, uri, None).await.unwrap();

        let mut params = VectorIndexParams::ivf_flat(num_clusters, DistanceType::L2);
        params.covering_columns(vec![
            "note".to_string(),
            "tag".to_string(),
            "payload".to_string(),
        ]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(index_name.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        (dataset, vectors)
    }

    /// The gather reads the survivors' rows and nothing else -- until they stop being a
    /// small fraction of the partition, at which point `Indices` degenerates into a
    /// scattered read of nearly everything and the sequential read it replaced is cheaper.
    ///
    /// Both branches are exercised here against the same real index, and the assertion
    /// distinguishes them directly rather than by cost: the scattered read returns exactly
    /// the rows asked for, the sequential fallback returns the partition's whole range
    /// (the caller aligns by row id either way, so both are correct). A test that passed on
    /// either branch would prove nothing about the threshold, so the two row counts are
    /// asserted separately and the byte counts are asserted on top.
    ///
    /// The line under test is `covering_read_for`'s comparison against
    /// `COVERING_SCATTERED_READ_MAX_PERCENT`. Neutralise the *comparison*, not the constant:
    /// `few` and `many` are derived from the same constant, so moving it drags both sides of
    /// the threshold with it and the behavioural assertions stay satisfied. Collapsing
    /// `covering_read_for` to `CoveringRead::Sequential` fails this test, and so does
    /// collapsing it to `CoveringRead::Scattered` for every non-empty partition.
    #[tokio::test]
    async fn test_covering_gather_reads_only_the_survivors_rows_until_the_threshold() {
        use lance_index::vector::storage::COVERING_SCATTERED_READ_MAX_PERCENT;
        use lance_io::scheduler::IoStats;

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;

        let test_dir = TempStrDir::default();
        let (dataset, _) = covered_gather_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        // The largest partition, not partition 0: IVF centroid initialisation is unseeded,
        // so any individual partition can come out empty and `partition_size(0) == 0` makes
        // both sides of the threshold the same set. Over `NUM_CLUSTERS` partitions the
        // largest necessarily holds at least `ROWS_PER_CLUSTER`, which keeps the guard below
        // satisfied for every assignment rather than for the lucky ones.
        let partition_id = (0..index.storage.num_partitions())
            .max_by_key(|partition| index.storage.partition_size(*partition))
            .expect("the index must hold at least one partition");
        let partition_rows = index.storage.partition_size(partition_id);
        assert!(
            partition_rows >= 20,
            "partition {partition_id} holds {partition_rows} rows; the threshold is a \
             percentage, so a tiny partition makes both sides of it the same set"
        );
        // Derive both sides from the actual partition size: cluster-to-partition
        // assignment is not stable across IVF training changes, so fixed counts go
        // stale. `few` sits strictly below the threshold, `many` at or above it.
        let threshold_rows = (partition_rows * COVERING_SCATTERED_READ_MAX_PERCENT).div_ceil(100);
        let few: Vec<u32> = (0..threshold_rows.saturating_sub(1).max(1) as u32).collect();
        let many: Vec<u32> = (0..(threshold_rows + 1).min(partition_rows) as u32).collect();
        assert!(
            many.len() * 100 >= partition_rows * COVERING_SCATTERED_READ_MAX_PERCENT
                && few.len() * 100 < partition_rows * COVERING_SCATTERED_READ_MAX_PERCENT,
            "the fixture must put `few` below the threshold and `many` above it for a \
             {partition_rows}-row partition"
        );
        let columns = vec!["payload".to_string()];

        let read = async |positions: Option<&[u32]>| {
            let io_stats = IoStats::new();
            let batch = index
                .storage
                .take_covering(partition_id, positions, &columns, Some(io_stats.clone()))
                .await
                .unwrap();
            (batch, io_stats.snapshot().bytes_read)
        };

        let (few_batch, few_bytes) = read(Some(&few)).await;
        let (many_batch, many_bytes) = read(Some(&many)).await;
        let (whole_batch, whole_bytes) = read(None).await;

        assert_eq!(
            few_batch.num_rows(),
            few.len(),
            "below the threshold the gather must read only the survivors' rows"
        );
        assert_eq!(
            many_batch.num_rows(),
            partition_rows,
            "above the threshold the gather must fall back to the partition's whole range"
        );
        assert_eq!(
            whole_batch.num_rows(),
            partition_rows,
            "positions the caller could not derive read the whole range"
        );

        assert!(whole_bytes > 0, "the fixture must actually perform I/O");
        assert_eq!(
            many_bytes, whole_bytes,
            "the fallback is the same read as the whole-range one: {many_bytes} vs \
             {whole_bytes} bytes"
        );
        assert!(
            few_bytes * 4 < whole_bytes,
            "the scattered read must be materially cheaper than the range it replaces: \
             {few_bytes} vs {whole_bytes} bytes for {} of {partition_rows} rows",
            few.len()
        );

        // Neither branch is a stub: both return the right values for the rows asked for,
        // matched by row id rather than by position.
        let whole_row_ids = whole_batch
            .column_by_name(ROW_ID)
            .expect("gathered batch carries row ids")
            .as_primitive::<UInt64Type>();
        let whole_payloads = whole_batch["payload"].as_string::<i32>();
        for batch in [&few_batch, &many_batch] {
            let row_ids = batch
                .column_by_name(ROW_ID)
                .expect("gathered batch carries row ids")
                .as_primitive::<UInt64Type>();
            let payloads = batch["payload"].as_string::<i32>();
            for (i, position) in few.iter().map(|p| *p as usize).enumerate() {
                assert_eq!(row_ids.value(i), whole_row_ids.value(position));
                assert_eq!(
                    payloads.value(i),
                    whole_payloads.value(position),
                    "gathered payload does not belong to the row it was asked for"
                );
                assert_eq!(
                    payloads.value(i),
                    format!("{:04}", row_ids.value(i)).repeat(256),
                    "gathered payload does not match the base table"
                );
            }
        }
    }

    /// The multi-partition gather must hold `O(survivors)`, not one whole covering batch
    /// per contributing partition -- even when every partition takes the whole-range
    /// fallback, which is what a pending fragment-reuse index forces for all of them.
    ///
    /// This is a memory bound, so it is asserted against bytes measured in the same test
    /// rather than a magic number: one partition's whole-range covering read is the
    /// positive control, and the gather's result over *four* such partitions must stay
    /// well under it. Row counts back it up -- a gather that accumulated whole partitions
    /// returns every row of every one of them, not the four survivors.
    ///
    /// The line under test is the narrowing in `gather_survivor_covering`
    /// (`take_record_batch` on each partition's batch before the next partition is read).
    /// Pushing the untouched `gathered` batch instead fails both assertions here:
    /// `4 * partition_rows` rows instead of 4, and a result larger than a whole partition.
    #[tokio::test]
    async fn test_covering_gather_holds_only_the_survivors_when_the_read_falls_back() {
        use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 256;

        let test_dir = TempStrDir::default();
        let (dataset, vectors) = covered_gather_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        let query = Query {
            column: "vector".to_string(),
            key: vectors.value(0),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: NUM_CLUSTERS,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: Some(Arc::from(vec!["payload".to_string()])),
        };
        let covering = index
            .query_covering(&query)
            .unwrap()
            .expect("the fixture declares covering columns");

        // One survivor per partition, each with `position: None` -- the state a deferred
        // fragment-reuse remap leaves every partition in, and the only one where the
        // gather reads more than the survivors' own rows.
        let mut locations: HashMap<u64, CoveringLocation> = HashMap::new();
        let mut expected_payloads: HashMap<u64, String> = HashMap::new();
        let mut whole_partition_bytes = 0;
        for partition_id in 0..NUM_CLUSTERS {
            let whole = index
                .storage
                .take_covering(partition_id, None, &covering.columns, None)
                .await
                .unwrap();
            assert_eq!(
                whole.num_rows(),
                index.storage.partition_size(partition_id),
                "the control must read the partition's whole range"
            );
            whole_partition_bytes = whole_partition_bytes.max(whole.get_array_memory_size());
            let row_id = whole
                .column_by_name(ROW_ID)
                .expect("gathered batch carries row ids")
                .as_primitive::<UInt64Type>()
                .value(0);
            expected_payloads.insert(row_id, whole["payload"].as_string::<i32>().value(0).into());
            locations.insert(
                row_id,
                CoveringLocation {
                    partition_id,
                    position: None,
                },
            );
        }
        assert_eq!(locations.len(), NUM_CLUSTERS, "one survivor per partition");

        let gathered = index
            .gather_survivor_covering(&locations, &covering, None)
            .await
            .unwrap()
            .expect("survivors were located, so the gather must return their covering");

        assert_eq!(
            gathered.num_rows(),
            locations.len(),
            "the gather must keep the survivors' rows only; holding every contributing \
             partition's whole covering would return {} rows",
            (0..NUM_CLUSTERS)
                .map(|p| index.storage.partition_size(p))
                .sum::<usize>()
        );
        assert!(
            whole_partition_bytes > 0,
            "the fixture must actually hold covering bytes"
        );
        assert!(
            gathered.get_array_memory_size() * 4 < whole_partition_bytes,
            "the gather must hold far less than a single partition's covering: {} bytes \
             for {} survivors vs {whole_partition_bytes} bytes for one partition",
            gathered.get_array_memory_size(),
            locations.len(),
        );

        // Bounded, and still correct: every survivor's own value, matched by row id.
        let row_ids = gathered
            .column_by_name(ROW_ID)
            .expect("gathered batch carries row ids")
            .as_primitive::<UInt64Type>();
        let payloads = gathered["payload"].as_string::<i32>();
        for (i, row_id) in row_ids.values().iter().enumerate() {
            assert_eq!(
                payloads.value(i),
                expected_payloads[row_id],
                "gathered payload does not belong to the row it was asked for"
            );
            assert_eq!(
                payloads.value(i),
                format!("{row_id:04}").repeat(256),
                "gathered payload does not match the base table"
            );
        }
        assert_eq!(
            row_ids.values().iter().copied().collect::<HashSet<_>>(),
            locations.keys().copied().collect::<HashSet<_>>(),
            "the gather must return exactly the survivors it was asked for"
        );
    }

    /// A covered query must return the covering values the index actually holds -- for a
    /// **nullable** covering column too, where a null fill is indistinguishable from the
    /// truth.
    ///
    /// This is the shape that hid a silent bug: when every covering fixture declares its
    /// columns non-nullable, an implementation that emits nulls for the survivors fails
    /// loudly on all of them and looks like a schema problem. A nullable column turns the
    /// same defect into ten rows of silently wrong values, and only a comparison against
    /// the base table catches it. `tag` is `1000 - offset` and `note` is `n{offset*7+3}`,
    /// so neither equals its row id and a positional gather cannot coincide with a
    /// by-name one.
    ///
    /// Both emit sites are covered: `query_parallelism = 1` merges partitions through the
    /// global top-k heap (`global_heap_to_batch`), `> 1` searches each partition
    /// separately (`append_covering`). Restoring only one of them fails one case.
    ///
    /// The absence of a `LanceRead` is what makes this an index measurement rather than a
    /// scan measurement: a covering column is semantically transparent, so a plan that
    /// re-fetched it from the base table would return byte-identical results.
    #[rstest]
    #[case::global_heap(1)]
    #[case::per_partition(4)]
    #[tokio::test]
    async fn test_covered_query_returns_nullable_covering_values_from_the_index(
        #[case] query_parallelism: i32,
    ) {
        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;
        const K: usize = 10;

        let test_dir = TempStrDir::default();
        let (dataset, vectors) = covered_gather_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;

        let query_key = vectors.value(0);
        let mut scan = dataset.scan();
        scan.nearest("vector", query_key.as_primitive::<Float32Type>(), K)
            .unwrap();
        scan.minimum_nprobes(NUM_CLUSTERS);
        scan.query_parallelism(query_parallelism);
        scan.with_row_id();
        scan.project(&["tag", "note"]).unwrap();

        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "a projection of covered columns only must not fetch from the base table, or \
             this test measures the scan rather than the index; plan:\n{plan}"
        );
        assert!(
            !plan.contains("payload"),
            "the query reads neither `payload` nor anything derived from it, so the index \
             must not be asked to materialize it; plan:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), K, "k = {K} must return {K} rows");
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        let tags = batch["tag"].as_primitive::<arrow_array::types::Int32Type>();
        let notes = batch["note"].as_string::<i32>();

        let mut nulls = 0;
        let mut values = 0;
        for i in 0..batch.num_rows() {
            let offset = row_ids.value(i) as usize;
            assert_eq!(
                tags.value(i),
                1000 - offset as i32,
                "row {i} (row_id {offset}): covered tag belongs to another row"
            );
            match offset % 3 {
                0 => {
                    assert!(
                        notes.is_null(i),
                        "row {i} (row_id {offset}): covered note must be NULL, got {:?}",
                        notes.value(i)
                    );
                    nulls += 1;
                }
                _ => {
                    assert!(
                        !notes.is_null(i),
                        "row {i} (row_id {offset}): covered note must not be NULL -- a null \
                         fill for every survivor is exactly the silent failure this fixture \
                         exists to catch"
                    );
                    assert_eq!(
                        notes.value(i),
                        format!("n{}", offset * 7 + 3),
                        "row {i} (row_id {offset}): covered note belongs to another row"
                    );
                    values += 1;
                }
            }
        }
        assert!(
            nulls > 0 && values > 0,
            "the result must contain both NULL and non-NULL notes, or one of the two cases \
             above is vacuous (got {nulls} nulls, {values} values)"
        );
    }

    /// `search_prepared_partition` is the synchronous phase of a prepared search: it runs
    /// on the CPU pool, where the covering gather -- which is I/O -- cannot be awaited.
    /// A query that needs covering columns must be told so, not handed rows whose covering
    /// columns are quietly absent; the caller has async entry points for exactly this.
    ///
    /// The same call with `covering_projection = Some(&[])` must succeed and return the
    /// bare `[_distance, _rowid]`. That is the state the projection narrowing exists for
    /// and the one that silently degrades if it is folded into `None`: here it is the
    /// difference between an error and a result.
    ///
    /// `supports_prepared_partition_search` must say so up front. A dispatcher that trusts
    /// the flag pays for the partition load before it ever reaches the rejection, so a
    /// covered index answering `true` there is a trap. The plain index built below is the
    /// control: without it, `false` would be satisfied by an index that simply never
    /// supported the entry point.
    #[tokio::test]
    async fn test_prepared_partition_search_rejects_a_query_that_needs_covering() {
        use lance_index::prefilter::NoFilter;
        use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;

        let test_dir = TempStrDir::default();
        let (dataset, vectors) = covered_gather_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        let query_for = |covering: Option<Vec<String>>| Query {
            column: "vector".to_string(),
            key: vectors.value(0),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: NUM_CLUSTERS,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: covering.map(Arc::from),
        };

        let prepared = index
            .prepare_partition_search(
                0,
                &query_for(Some(vec!["tag".to_string()])),
                Arc::new(NoFilter),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let err = index
            .search_prepared_partition(prepared, &NoOpMetricsCollector)
            .expect_err("a covered query must not be silently served without its covering");
        assert!(
            err.to_string().contains("search_prepared_partition"),
            "the error must name the entry point that cannot serve it, got: {err}"
        );

        let prepared = index
            .prepare_partition_search(
                0,
                &query_for(Some(Vec::new())),
                Arc::new(NoFilter),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let batch = index
            .search_prepared_partition(prepared, &NoOpMetricsCollector)
            .expect("a query needing no covering column is served here as before");
        assert_eq!(
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec![DIST_COL, ROW_ID],
            "`Some(&[])` means no covering work at all, not covering projected away"
        );
        assert!(batch.num_rows() > 0, "the partition must be non-empty");

        // The capability flag must not advertise what the rejection above denies. It
        // carries no query, so it answers for the index: a covered index says no.
        assert!(
            !index.supports_prepared_partition_search(),
            "a covered index must not advertise a prepared search it rejects"
        );

        // Control: the same index shape without covering columns still supports it.
        let plain_dir = TempStrDir::default();
        let plain_schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            vectors.data_type().clone(),
            false,
        )]));
        let plain_batch =
            RecordBatch::try_new(plain_schema.clone(), vec![vectors.clone()]).unwrap();
        let mut plain_dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(plain_batch)], plain_schema),
            plain_dir.as_str(),
            None,
        )
        .await
        .unwrap();
        plain_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &VectorIndexParams::ivf_flat(NUM_CLUSTERS, DistanceType::L2),
                true,
            )
            .await
            .unwrap();
        let plain_ctx = load_vector_index_context(&plain_dataset, "vector", INDEX_NAME).await;
        let plain_index = plain_ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");
        assert!(
            plain_index.supports_prepared_partition_search(),
            "an ordinary index still supports the prepared search; without this the \
             covered assertion above would pass on an index that never supported it"
        );
    }

    /// A partition read with only its internal columns must still be a well-formed
    /// storage: `try_from_batch` does not require covering columns, and `covering_batch()`
    /// on the result reports none rather than erroring or fabricating an empty batch.
    ///
    /// The same partition read with `PartitionColumns::All` is the control. Without it,
    /// "reports none" would pass just as well on an index that never had covering columns,
    /// and the row-id comparison would have nothing to catch a narrowed read that silently
    /// shifted rows.
    #[tokio::test]
    async fn test_codes_only_partition_storage_is_well_formed() {
        use lance_index::vector::flat::storage::FLAT_COLUMN;

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;

        let test_dir = TempStrDir::default();
        let (dataset, _) = covered_flat_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;
        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        // Whichever partition k-means actually filled, not partition 0 unconditionally:
        // which centroid wins which rows depends on initialisation, and on aarch64
        // partition 0 came out empty and tripped the vacuity guard. Any non-empty
        // partition exercises the same storage invariants, so this asserts on the
        // storage layout rather than on the clustering.
        let mut chosen = None;
        for candidate in 0..index.ivf_model().num_partitions() {
            let storage = index
                .load_partition_storage(candidate, PartitionColumns::All, None)
                .await
                .unwrap();
            if !storage.is_empty() {
                chosen = Some((candidate, storage));
                break;
            }
        }
        let (partition_id, all) =
            chosen.expect("some partition must be non-empty or every assertion below is vacuous");
        let internal = index
            .load_partition_storage(partition_id, PartitionColumns::Internal, None)
            .await
            .unwrap();
        let covering_names = |storage: &lance_index::vector::flat::storage::FlatFloatStorage| {
            let schema = storage.schema().clone();
            storage
                .covering_field_indices()
                .into_iter()
                .map(|i| schema.field(i).name().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            covering_names(&all),
            vec!["payload".to_string(), "price".to_string()],
            "the control read must carry both covering columns, or the narrowed arm below \
             proves nothing"
        );
        assert!(all.covering_batch().unwrap().is_some());

        assert!(
            covering_names(&internal).is_empty(),
            "a codes-only read must carry no covering column"
        );
        assert!(
            internal.covering_batch().unwrap().is_none(),
            "covering_batch() on a codes-only storage reports none rather than erroring \
             or emitting an empty batch"
        );

        // Well-formed: the storage's own columns are all there, and the rows are the same
        // rows in the same order. Row ids are compared through `row_ids()`, which resolves
        // `_rowid` by name -- the covered layout puts covering columns before it.
        assert_eq!(internal.len(), all.len());
        assert_eq!(
            internal.row_ids().copied().collect::<Vec<_>>(),
            all.row_ids().copied().collect::<Vec<_>>(),
            "narrowing the read must not disturb which rows the partition holds"
        );
        assert!(internal.schema().column_with_name(ROW_ID).is_some());
        assert!(internal.schema().column_with_name(FLAT_COLUMN).is_some());
        assert_eq!(
            internal.schema().fields().len(),
            2,
            "flat storage's internal columns are exactly [{ROW_ID}, {FLAT_COLUMN}]; schema \
             was {:?}",
            internal.schema()
        );
    }

    /// The rebuild path reads a partition with `PartitionColumns::All` on purpose: it
    /// re-writes those batches into the index being built, so a covering column left unread
    /// is a covering column the merged index no longer has -- silently, since nothing
    /// downstream of the rebuild asks for it again.
    ///
    /// Asserted on the merged STORAGE rather than through a query, so it holds independently
    /// of what the search path currently does with covering. The line under test is
    /// `PartitionColumns::All` in `IvfIndexBuilder::take_partition_batches`; switching it to
    /// `Internal` drops the covering columns from the merged index and fails this test.
    #[tokio::test]
    async fn test_merge_optimize_preserves_covering_in_storage() {
        use arrow_array::{Int32Array, StringArray, types::Int32Type};

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;
        const INDEXED: usize = NUM_CLUSTERS * ROWS_PER_CLUSTER;
        const APPENDED: usize = 128;
        const DIMS: usize = 16;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) =
            covered_flat_fixture(test_uri, INDEX_NAME, NUM_CLUSTERS, ROWS_PER_CLUSTER).await;

        // Append unindexed rows, then merge them in. The merge is what re-reads the
        // existing partitions and re-writes them into the new index file.
        let mut flat = Vec::with_capacity(APPENDED * DIMS);
        for row in 0..APPENDED {
            for d in 0..DIMS {
                flat.push(row as f32 * 0.01 + d as f32 * 0.0001);
            }
        }
        let appended_vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), DIMS as i32).unwrap(),
        );
        let schema = dataset.schema().into();
        let appended = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    (INDEXED..INDEXED + APPENDED).map(|i| -(i as i32) - 7),
                )),
                Arc::new(StringArray::from_iter_values(
                    (INDEXED..INDEXED + APPENDED).map(|i| format!("p{}", i * 3 + 11)),
                )),
                appended_vectors,
            ],
        )
        .unwrap();
        let appended_schema = appended.schema();
        dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(appended)], appended_schema),
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        let mut covered: HashMap<u64, (String, i32)> = HashMap::new();
        for partition in 0..index.ivf_model().num_partitions() {
            let storage = index
                .load_partition_storage(partition, PartitionColumns::All, None)
                .await
                .unwrap();
            for batch in storage.to_batches().unwrap() {
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .expect("storage batch carries row ids")
                    .as_primitive::<UInt64Type>();
                let payloads = batch
                    .column_by_name("payload")
                    .expect("merged storage must keep covering column 'payload'")
                    .as_string::<i32>();
                let prices = batch
                    .column_by_name("price")
                    .expect("merged storage must keep covering column 'price'")
                    .as_primitive::<Int32Type>();
                for i in 0..batch.num_rows() {
                    covered.insert(
                        row_ids.value(i),
                        (payloads.value(i).to_string(), prices.value(i)),
                    );
                }
            }
        }
        assert_eq!(
            covered.len(),
            INDEXED + APPENDED,
            "the merge must index every row, or the value check below misses the rows it dropped"
        );

        // Values, not just presence: a merge that re-attached covering by position rather
        // than by row id keeps the column and corrupts it.
        let row_ids: Vec<u64> = covered.keys().copied().collect();
        let truth = dataset
            .take_rows(
                &row_ids,
                crate::dataset::ProjectionRequest::from_columns(
                    ["payload", "price"],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let truth_payloads = truth.column_by_name("payload").unwrap().as_string::<i32>();
        let truth_prices = truth
            .column_by_name("price")
            .unwrap()
            .as_primitive::<Int32Type>();
        for (i, row_id) in row_ids.iter().enumerate() {
            let (payload, price) = &covered[row_id];
            assert_eq!(
                (payload.as_str(), *price),
                (truth_payloads.value(i), truth_prices.value(i)),
                "row_id {row_id}: merged covering value does not match the base table"
            );
        }
    }

    /// The remap path rewrites a partition into a new index file exactly as the merge
    /// path does, so it has the same requirement: it must read every column, not just
    /// the internal ones. It reaches the storage through the partition *entry* rather
    /// than through `load_partition_storage`, which is why the merge test above does not
    /// cover it -- and why a codes-only entry silently strips the covering columns from
    /// every compacted covered index.
    ///
    /// Asserted on the remapped STORAGE, not through a query, so it holds independently
    /// of what the search path currently does with covering (`test_ivf_pq_covered_survives_compaction`
    /// asserts the same property through a query and is ignored until the survivors-only
    /// gather lands). The line under test is `load_partition_entry_with_covering` in
    /// `IvfIndexBuilder::remap`; the cached `load_partition` reads internal columns only
    /// and fails this test.
    #[tokio::test]
    async fn test_compaction_remap_preserves_covering_in_storage() {
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        use arrow_array::types::Int32Type;

        const INDEX_NAME: &str = "vector_idx";
        const NUM_CLUSTERS: usize = 4;
        const ROWS_PER_CLUSTER: usize = 64;

        let test_dir = TempStrDir::default();
        let (mut dataset, _) = covered_flat_fixture(
            test_dir.as_str(),
            INDEX_NAME,
            NUM_CLUSTERS,
            ROWS_PER_CLUSTER,
        )
        .await;

        // Delete enough rows to clear the materialize-deletions threshold, so compaction
        // rewrites the fragment and remaps the index rather than no-opping.
        dataset.delete("price > -60").await.unwrap();
        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert!(
            metrics.files_removed > 0,
            "compaction must actually rewrite fragments, or the remap under test never runs"
        );

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let index = ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlatIndex");

        let mut covered: HashMap<u64, (String, i32)> = HashMap::new();
        for partition in 0..index.ivf_model().num_partitions() {
            let storage = index
                .load_partition_storage(partition, PartitionColumns::All, None)
                .await
                .unwrap();
            for batch in storage.to_batches().unwrap() {
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .expect("storage batch carries row ids")
                    .as_primitive::<UInt64Type>();
                let payloads = batch
                    .column_by_name("payload")
                    .expect("remapped storage must keep covering column 'payload'")
                    .as_string::<i32>();
                let prices = batch
                    .column_by_name("price")
                    .expect("remapped storage must keep covering column 'price'")
                    .as_primitive::<Int32Type>();
                for i in 0..batch.num_rows() {
                    covered.insert(
                        row_ids.value(i),
                        (payloads.value(i).to_string(), prices.value(i)),
                    );
                }
            }
        }
        assert!(
            !covered.is_empty(),
            "the remapped index must still hold rows, or the value check below is vacuous"
        );

        // Values, not just presence: a remap that re-attached covering by position rather
        // than by row id keeps the column and corrupts it, and the row ids themselves are
        // the post-compaction addresses.
        let row_ids: Vec<u64> = covered.keys().copied().collect();
        let truth = dataset
            .take_rows(
                &row_ids,
                crate::dataset::ProjectionRequest::from_columns(
                    ["payload", "price"],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        let truth_payloads = truth.column_by_name("payload").unwrap().as_string::<i32>();
        let truth_prices = truth
            .column_by_name("price")
            .unwrap()
            .as_primitive::<Int32Type>();
        for (i, row_id) in row_ids.iter().enumerate() {
            let (payload, price) = &covered[row_id];
            assert_eq!(
                (payload.as_str(), *price),
                (truth_payloads.value(i), truth_prices.value(i)),
                "row_id {row_id}: remapped covering value does not match the base table"
            );
        }
    }

    /// A covered FLAT/SQ index must survive a *deferred* remap: `compact_files`
    /// with `defer_index_remap: true` rewrites fragments but leaves the index's
    /// row ids to be remapped later via a fragment-reuse index (FRI) instead of
    /// remapping them inline. `IvfQuantizationStorage::load_partition` passes
    /// that FRI on every subsequent load (search, optimize, another build), so
    /// a covered FLAT/SQ storage batch -- `[_rowid, code, <covering...>]`, three
    /// or more columns -- must not be rejected as though it could only ever be
    /// the two-column `(value, row_id)` shape a plain scalar index has.
    ///
    /// PQ and RQ are excluded: PQ remaps inline via `rebuild_storage_batch` and
    /// RQ via its own `remap`, so neither ever reaches the shared FRI path this
    /// guards. The HNSW_FLAT/HNSW_SQ variants are also excluded here: HNSW's
    /// graph independently fails to survive `defer_index_remap` even without
    /// any covering columns (pre-existing, unrelated to covering -- see the
    /// P0 fix report), so they would fail this test for a reason this fix
    /// does not address.
    ///
    /// This is also the only test that reaches [`CoveringGather::WholeRange`]: the FRI
    /// drops rows as the partition is loaded, so a storage position no longer addresses
    /// the file and the survivor gather must re-read the whole range and match by row id.
    /// Removing that alignment check fails here with `row id ... missing from covering
    /// source` -- loudly, because the gather is matched by row id rather than by position.
    #[rstest]
    #[case::flat(VectorIndexParams::ivf_flat(4, DistanceType::L2))]
    #[case::sq(VectorIndexParams::with_ivf_sq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        SQBuildParams::default()
    ))]
    #[tokio::test]
    async fn test_covered_survives_deferred_remap_frag_reuse(
        #[case] mut params: VectorIndexParams,
    ) {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Delete rows and defer the index remap: this fragment-reuse index now
        // covers the covered vector index built above, so every subsequent load
        // of its storage carries it.
        dataset.delete("id < 100").await.unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // A search must load the (now FRI-tagged) covered storage without
        // rejecting its three-or-more-column shape, and still return correct,
        // row-aligned covering values for the surviving rows.
        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10);
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' column must be emitted")
            .as_primitive::<UInt64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();

        // Ground truth via an independent (non-index, base-table take) path,
        // keyed by the post-compaction row id the scan returned. Compaction
        // reassigns physical row addresses, so `id == _rowid` no longer holds
        // here the way it does pre-compaction -- but the covered value must
        // still equal the row's true id.
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let projection = crate::dataset::ProjectionRequest::from_columns(["id"], dataset.schema());
        let truth = dataset.take_rows(&row_id_vec, projection).await.unwrap();
        let truth_ids = truth
            .column_by_name("id")
            .unwrap()
            .as_primitive::<UInt64Type>();
        for i in 0..ids.len() {
            assert!(
                ids.value(i) >= 100,
                "deleted rows (id < 100) must not be returned"
            );
            assert_eq!(
                ids.value(i),
                truth_ids.value(i),
                "row {i} (row_id {}): covered id != true id after deferred-remap compaction",
                row_ids.value(i)
            );
        }

        // Building a SECOND covered index while a fragment-reuse index already
        // exists on the dataset must also succeed: `StorageBuilder::build`
        // threads the FRI into `Q::Storage::try_from_batch` for every new
        // build, indexed or not.
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx_2".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // ...and the index it produces must be *correct*, not merely buildable. That
        // build is where a covered storage batch reaches the fragment-reuse remap in
        // its build-time column order -- `[<covering...>, _rowid, code]`, row id NOT
        // at index 0. The search earlier in this test loads the written file instead,
        // whose `[_rowid, code, <covering...>]` order makes a positional row-id lookup
        // right by accident. So without querying this second index, nothing pins
        // `remap_row_ids_by_name` locating the row id by name: a positional variant
        // remaps the covering column, leaves the real row ids unremapped, and no
        // assertion anywhere observes it.
        dataset.drop_index(INDEX_NAME).await.unwrap();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id"]).unwrap();
        // Pin the ANN path: a flat-KNN fallback would satisfy every assertion below
        // without the remapped index being read at all.
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "the covered query must go through the second index; plan was:\n{plan}"
        );
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id'] should skip the base-table take; plan was:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10);
        let ids = batch
            .column_by_name("id")
            .expect("covered 'id' column must be emitted")
            .as_primitive::<UInt64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let projection = crate::dataset::ProjectionRequest::from_columns(["id"], dataset.schema());
        let truth = dataset.take_rows(&row_id_vec, projection).await.unwrap();
        let truth_ids = truth
            .column_by_name("id")
            .unwrap()
            .as_primitive::<UInt64Type>();
        for i in 0..ids.len() {
            assert!(
                ids.value(i) >= 100,
                "deleted rows (id < 100) must not be returned by the second index"
            );
            assert_eq!(
                ids.value(i),
                truth_ids.value(i),
                "row {i} (row_id {}): covered id != true id for the index built \
                 while a fragment-reuse index was already present",
                row_ids.value(i)
            );
        }
    }

    /// Fix 2: under query parallelism > 1 the parallel search branch (which uses
    /// `search_in_partition`, not the heap merge) must also emit covering
    /// columns. On a multi-core runner this exercises the parallel path; if the
    /// session resolves parallelism to 1 it still passes via the sequential path.
    #[tokio::test]
    async fn test_ivf_pq_covered_projection_parallel() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.minimum_nprobes(4); // search several partitions...
        scan.query_parallelism(4); // ...in parallel, forcing the parallel branch
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert!(batch.column_by_name("id").is_some());
        assert_eq!(batch.num_rows(), 10, "k=10 should return 10 rows");
    }

    /// Fix 3: a covered index with unindexed fragments (rows appended but not
    /// optimized) must not panic on a non-fast-search query. The flat search
    /// over the unindexed rows now projects the covering columns so the union
    /// with the index result succeeds.
    #[tokio::test]
    async fn test_ivf_pq_covered_with_unindexed_fragments() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.covering_columns(vec!["id".to_string()]);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Append rows WITHOUT optimizing -> unindexed fragments; a (non-fast)
        // query then goes through the index + flat-search combine path.
        append_dataset::<Float32Type>(&mut dataset, 100, 0.0..1.0).await;

        let q = vectors.value(0);
        let q = q.as_primitive::<Float32Type>();
        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert!(batch.column_by_name("id").is_some());
        assert_eq!(batch.num_rows(), 10, "k=10 should return 10 rows");
    }

    async fn shrink_smallest_partition(
        dataset: &mut Dataset,
        index_name: &str,
        expected_after_join: usize,
        next_id: &mut u64,
    ) -> (usize, usize, usize) {
        const ROWS_TO_APPEND_FOR_JOIN: usize = 32;
        let row_count_before = dataset.count_all_rows().await.unwrap();
        let index_ctx = load_vector_index_context(dataset, "vector", index_name).await;
        let partitions = index_ctx.stats()["indices"][0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        let (partition_idx, _size) = partitions
            .iter()
            .enumerate()
            .filter_map(|(idx, part)| part["size"].as_u64().map(|size| (idx, size)))
            .filter(|(_, size)| *size > 1)
            .min_by_key(|(_, size)| *size)
            .expect("should have at least one partition with joinable rows");

        let row_ids = load_partition_row_ids(index_ctx.ivf(), partition_idx).await;
        assert!(
            row_ids.len() > 1,
            "Partition {} should have removable rows",
            partition_idx
        );

        let rows = dataset
            .take_rows(&row_ids, dataset.schema().clone())
            .await
            .unwrap();
        let ids = rows["id"].as_primitive::<UInt64Type>().values();
        let template_values = rows["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        delete_ids(dataset, &ids[1..]).await;
        compact_after_deletions(dataset).await;

        append_constant_vector_with_start_id(
            dataset,
            ROWS_TO_APPEND_FOR_JOIN,
            &template_values,
            next_id,
        )
        .await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let post_ctx = load_vector_index_context(dataset, "vector", index_name).await;
        let post_partitions = post_ctx.num_partitions();
        assert_eq!(
            post_partitions,
            expected_after_join,
            "Expected partitions to be at most {} after join, got stats: {}",
            expected_after_join,
            post_ctx.stats_json()
        );

        let row_count_after = dataset.count_all_rows().await.unwrap();
        debug_assert!(
            row_count_before + ROWS_TO_APPEND_FOR_JOIN >= row_count_after,
            "row count should not increase after delete + append"
        );
        let deleted_rows = row_count_before + ROWS_TO_APPEND_FOR_JOIN - row_count_after;

        (deleted_rows, ROWS_TO_APPEND_FOR_JOIN, post_partitions)
    }

    async fn append_constant_vector_with_start_id(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        next_id: &mut u64,
    ) {
        append_constant_vector_batch(dataset, rows, template, *next_id, None).await;
        *next_id += rows as u64;
    }

    async fn append_partition_templates(
        dataset: &mut Dataset,
        rows_per_template: usize,
        templates: &[Vec<f32>],
    ) {
        assert!(
            !templates.is_empty(),
            "at least one template is required for append"
        );
        for template in templates {
            assert_eq!(
                template.len(),
                DIM,
                "Template vector should have {} dimensions",
                DIM
            );
        }

        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        let total_rows = rows_per_template * templates.len();
        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + total_rows as u64,
        ));
        let mut appended_values = Vec::with_capacity(total_rows * DIM);
        for template in templates {
            for _ in 0..rows_per_template {
                appended_values.extend_from_slice(template);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(appended_values),
                DIM as i32,
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(batches, None).await.unwrap();
    }

    async fn append_constant_vector_with_params(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        write_params: Option<WriteParams>,
    ) {
        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        append_constant_vector_batch(dataset, rows, template, start_id, write_params).await;
    }

    async fn append_constant_vector_batch(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        start_id: u64,
        write_params: Option<WriteParams>,
    ) {
        assert_eq!(
            template.len(),
            DIM,
            "Template vector should have {} dimensions",
            DIM
        );

        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + rows as u64,
        ));
        let mut appended_values = Vec::with_capacity(rows * DIM);
        for _ in 0..rows {
            appended_values.extend_from_slice(template);
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(appended_values),
                DIM as i32,
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let params = write_params.map(|mut params| {
            params.mode = WriteMode::Append;
            params
        });
        dataset.append(batches, params).await.unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_and_verify_append_phase(
        dataset: &mut Dataset,
        index_name: &str,
        template: &[f32],
        next_id: &mut u64,
        rows_to_append: usize,
        expected_partitions: usize,
        expected_total_rows: usize,
        expected_index_count: usize,
        expect_split: bool,
    ) {
        append_constant_vector_with_start_id(dataset, rows_to_append, template, next_id).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let stats_json = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();

        let indices = stats["indices"]
            .as_array()
            .expect("indices array should exist");
        if expect_split {
            assert_eq!(
                indices.len(),
                expected_index_count,
                "Expected {} index entries after split, got {}, stats: {}",
                expected_index_count,
                indices.len(),
                stats
            );
        } else {
            assert!(
                indices.len() >= expected_index_count,
                "Expected at least {} index entries after append, got {}, stats: {}",
                expected_index_count,
                indices.len(),
                stats
            );
        }
        assert!(
            stats["num_indices"].as_u64().unwrap() as usize >= expected_index_count,
            "num_indices should be at least {}, stats: {}",
            expected_index_count,
            stats
        );
        assert_eq!(
            stats["num_indexed_rows"].as_u64().unwrap() as usize,
            expected_total_rows,
            "Total indexed rows mismatch after append"
        );

        let base_index = indices
            .iter()
            .max_by_key(|entry| entry["num_partitions"].as_u64().unwrap_or(0))
            .expect("at least one index entry should exist");
        assert_eq!(
            base_index["num_partitions"].as_u64().unwrap() as usize,
            expected_partitions,
            "Partition count mismatch after append"
        );

        if expected_index_count == 1 {
            let partitions = base_index["partitions"]
                .as_array()
                .expect("partitions should exist");
            assert_eq!(
                partitions.len(),
                expected_partitions,
                "Expected {} partitions, found {}",
                expected_partitions,
                partitions.len()
            );
            let partition_sizes: Vec<usize> = partitions
                .iter()
                .map(|part| part["size"].as_u64().unwrap() as usize)
                .collect();
            let total_partition_rows: usize = partition_sizes.iter().sum();
            assert_eq!(
                total_partition_rows, expected_total_rows,
                "Partition sizes should sum to total rows: {:?}",
                partition_sizes
            );
        } else {
            assert!(
                !expect_split,
                "Split should result in a single merged index"
            );
        }

        assert_eq!(
            dataset.count_all_rows().await.unwrap(),
            expected_total_rows,
            "Dataset row count mismatch after append"
        );
    }

    async fn load_partition_row_ids(index: &IvfPq, partition_idx: usize) -> Vec<u64> {
        index
            .storage
            .load_partition(partition_idx, PartitionColumns::Internal, None)
            .await
            .unwrap()
            .row_ids()
            .copied()
            .collect()
    }

    async fn load_flat_partition_row_ids(index: &IvfFlatIndex, partition_idx: usize) -> Vec<u64> {
        index
            .storage
            .load_partition(partition_idx, PartitionColumns::Internal, None)
            .await
            .unwrap()
            .row_ids()
            .copied()
            .collect()
    }

    async fn delete_ids(dataset: &mut Dataset, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let predicate = ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        dataset
            .delete(&format!("id in ({})", predicate))
            .await
            .unwrap();
    }

    async fn compact_after_deletions(dataset: &mut Dataset) {
        compact_files(
            dataset,
            CompactionOptions {
                materialize_deletions_threshold: 0.0,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    }

    async fn ground_truth(
        dataset: &Dataset,
        column: &str,
        query: &dyn Array,
        k: usize,
        distance_type: DistanceType,
    ) -> HashSet<u64> {
        let batch = dataset
            .scan()
            .with_row_id()
            .nearest(column, query, k)
            .unwrap()
            .distance_metric(distance_type)
            .use_index(false)
            .try_into_batch()
            .await
            .unwrap();
        batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect()
    }

    fn multivec_ground_truth(
        vectors: &ListArray,
        query: &dyn Array,
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u64)> {
        let query = if let Some(list_array) = query.as_list_opt::<i32>() {
            list_array.values().clone()
        } else {
            query.as_fixed_size_list().values().clone()
        };
        multivec_distance(&query, vectors, distance_type)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, dist)| (dist, i as u64))
            .sorted_by(|a, b| a.0.total_cmp(&b.0))
            .take(k)
            .collect()
    }

    const TWO_FRAG_NUM_ROWS: usize = 2000;
    const TWO_FRAG_DIM: usize = 128;
    const TWO_FRAG_NUM_PARTITIONS: usize = 4;
    const TWO_FRAG_NUM_SUBVECTORS: usize = 16;
    const TWO_FRAG_NUM_BITS: usize = 8;
    const TWO_FRAG_SAMPLE_RATE: usize = 7;
    const TWO_FRAG_MAX_ITERS: u32 = 20;

    fn make_two_fragment_batches() -> (Arc<Schema>, Vec<RecordBatch>) {
        let ids = Arc::new(UInt64Array::from_iter_values(0..TWO_FRAG_NUM_ROWS as u64));

        let values = generate_random_array_with_range(TWO_FRAG_NUM_ROWS * TWO_FRAG_DIM, 0.0..1.0);
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(values),
                TWO_FRAG_DIM as i32,
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();

        (schema, vec![batch])
    }

    /// The covering tests' own vector geometry, kept separate from the shared `TWO_FRAG_*`
    /// fixture. Global codebook training costs `dim * iters * samples * centroids` and these
    /// tests pay it once per case while building and merging several indexes, which is what put
    /// them over the one-second local-unit-test budget. What they assert -- that covered values
    /// survive every build, merge and lifecycle step and stay row-aligned with the base table --
    /// is independent of quantization resolution, so the vector shrinks instead. `dim /
    /// num_sub_vectors` stays 8, as the shared fixture has it, and the codebook stays 8-bit: that
    /// is the production default and the width these tests exist to cover.
    ///
    /// The shared fixture keeps its own dimensions: its tests assert *exact* single-vs-split
    /// top-K equality over uniform-random vectors, which a coarser codebook breaks by making
    /// distances tie.
    const COVERED_DIM: usize = 32;
    const COVERED_NUM_SUBVECTORS: usize = 4;
    const COVERED_MAX_ITERS: u32 = 4;

    /// Like `make_two_fragment_batches`, but with two covering columns: a non-null `id` and a
    /// **nullable** `payload` (every 3rd value null). Covered tests use this so null covering
    /// values are exercised through the per-segment build and the cross-shard merge.
    fn make_covered_test_batches() -> (Arc<Schema>, Vec<RecordBatch>) {
        let ids = Arc::new(UInt64Array::from_iter_values(0..TWO_FRAG_NUM_ROWS as u64));
        let payload = Arc::new(UInt64Array::from_iter(
            (0..TWO_FRAG_NUM_ROWS as u64).map(|v| if v % 3 == 0 { None } else { Some(v + 1000) }),
        ));

        // Clustered vectors: uniform-random data is pathological for IVF_PQ recall (the curse of
        // dimensionality), which would make a recall gate flaky. Instead pack points into
        // well-separated clusters of <= k points each (centers 4.0 apart per dim -> ~23 in L2 at
        // `COVERED_DIM`, tiny within-cluster jitter), so a query drawn from a cluster has its whole
        // cluster as the unambiguous nearest neighbors and IVF_PQ recall is reliably high.
        const CLUSTER_SIZE: usize = 8;
        let mut flat = Vec::with_capacity(TWO_FRAG_NUM_ROWS * COVERED_DIM);
        for row in 0..TWO_FRAG_NUM_ROWS {
            let center = (row / CLUSTER_SIZE) as f32 * 4.0;
            let within = (row % CLUSTER_SIZE) as f32;
            for d in 0..COVERED_DIM {
                flat.push(center + within * 0.002 + d as f32 * 0.00001);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), COVERED_DIM as i32)
                .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("payload", DataType::UInt64, true),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, payload, vectors]).unwrap();

        (schema, vec![batch])
    }

    /// Assert `refine` is still served out of index storage rather than a base-table take.
    ///
    /// The index under test must cover `payload`, so the projection is served from the index
    /// too and the only thing that could still read the base table is refine's own fetch of
    /// full-precision vectors. Works for every index type, unlike inspecting partition
    /// storage, which needs the concrete index struct.
    async fn assert_refine_served_from_index(dataset: &Dataset, query: &dyn Array, what: &str) {
        let mut scan = dataset.scan();
        scan.nearest("vector", query, 10).unwrap();
        scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
        scan.refine(2);
        scan.project(&["payload"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "{what}: refine must still be served from the index; plan:\n{plan}"
        );
        let rows = scan.try_into_batch().await.unwrap().num_rows();
        assert_eq!(
            rows, 10,
            "{what}: the query must still return its k neighbours"
        );
    }

    /// Append rows in the schema `make_covered_test_batches` writes, laid out over the same
    /// clusters, so every partition ends up holding existing *and* freshly indexed rows.
    async fn append_covered_rows(dataset: &mut Dataset, rows: usize) {
        const CLUSTER_SIZE: usize = 8;
        let start = dataset.count_all_rows().await.unwrap() as u64;
        let ids = Arc::new(UInt64Array::from_iter_values(start..start + rows as u64));
        let payload = Arc::new(UInt64Array::from_iter(
            (0..rows as u64).map(|v| if v % 3 == 0 { None } else { Some(v + 5000) }),
        ));
        let mut flat = Vec::with_capacity(rows * COVERED_DIM);
        for row in 0..rows {
            let center = (row / CLUSTER_SIZE) as f32 * 4.0;
            let within = (row % CLUSTER_SIZE) as f32;
            for d in 0..COVERED_DIM {
                flat.push(center + within * 0.002 + d as f32 * 0.00001);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(flat), COVERED_DIM as i32)
                .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("payload", DataType::UInt64, true),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, payload, vectors]).unwrap();
        dataset
            .append(RecordBatchIterator::new(vec![Ok(batch)], schema), None)
            .await
            .unwrap();
    }

    async fn write_dataset_from_batches(
        test_uri: &str,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Dataset {
        write_dataset_from_batches_with_max_rows(test_uri, schema, batches, 500).await
    }

    async fn write_dataset_from_batches_with_max_rows(
        test_uri: &str,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
        max_rows_per_file: usize,
    ) -> Dataset {
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

        let write_params = WriteParams {
            max_rows_per_file,
            mode: WriteMode::Overwrite,
            ..Default::default()
        };

        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap()
    }

    async fn prepare_global_ivf_pq(
        dataset: &Dataset,
        vector_column: &str,
    ) -> (IvfBuildParams, PQBuildParams) {
        prepare_ivf_pq(
            dataset,
            vector_column,
            TWO_FRAG_DIM,
            TWO_FRAG_NUM_PARTITIONS,
            TWO_FRAG_NUM_SUBVECTORS,
            TWO_FRAG_NUM_BITS,
            TWO_FRAG_MAX_ITERS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn prepare_ivf_pq(
        dataset: &Dataset,
        vector_column: &str,
        expected_dimension: usize,
        num_partitions: usize,
        num_sub_vectors: usize,
        num_bits: usize,
        max_iters: u32,
        sample_rate: usize,
    ) -> (IvfBuildParams, PQBuildParams) {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();

        let dim = vectors.value_length() as usize;
        assert_eq!(dim, expected_dimension, "unexpected vector dimension");

        let values = vectors.values().as_primitive::<Float32Type>();

        let kmeans_params = KMeansParams::new(None, max_iters, 1, DistanceType::L2);
        let kmeans =
            train_kmeans::<Float32Type>(values, kmeans_params, dim, num_partitions, sample_rate)
                .unwrap();

        let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
        let centroids_fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(num_partitions, centroids_fsl).unwrap();
        ivf_params.max_iters = max_iters as usize;
        ivf_params.sample_rate = sample_rate;

        let mut pq_train_params = PQBuildParams::new(num_sub_vectors, num_bits);
        pq_train_params.max_iters = max_iters as usize;
        pq_train_params.sample_rate = sample_rate;

        let pq = pq_train_params.build(vectors, DistanceType::L2).unwrap();
        let codebook_flat = pq.codebook.values().as_primitive::<Float32Type>().clone();
        let pq_codebook: ArrayRef = Arc::new(codebook_flat);
        let mut pq_params = PQBuildParams::with_codebook(num_sub_vectors, num_bits, pq_codebook);
        pq_params.max_iters = max_iters as usize;
        pq_params.sample_rate = sample_rate;

        (ivf_params, pq_params)
    }

    /// `prepare_global_ivf_pq` for the covering fixture: same shared training path, but over
    /// `COVERED_DIM` vectors and the covering codebook.
    async fn prepare_covered_ivf_pq(
        dataset: &Dataset,
        vector_column: &str,
    ) -> (IvfBuildParams, PQBuildParams) {
        prepare_ivf_pq(
            dataset,
            vector_column,
            COVERED_DIM,
            TWO_FRAG_NUM_PARTITIONS,
            COVERED_NUM_SUBVECTORS,
            TWO_FRAG_NUM_BITS,
            COVERED_MAX_ITERS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .await
    }

    /// `prepare_global_ivf` for the covering fixture. Distributed builds need every shard to
    /// share one set of centroids, so the covered non-PQ cases pre-train them here rather than
    /// letting each shard train its own.
    async fn prepare_covered_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();

        let dim = vectors.value_length() as usize;
        assert_eq!(dim, COVERED_DIM, "unexpected vector dimension");

        let values = vectors.values().as_primitive::<Float32Type>();
        let kmeans_params = KMeansParams::new(None, COVERED_MAX_ITERS, 1, DistanceType::L2);
        let kmeans = train_kmeans::<Float32Type>(
            values,
            kmeans_params,
            dim,
            TWO_FRAG_NUM_PARTITIONS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .unwrap();

        let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
        let centroids_fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(TWO_FRAG_NUM_PARTITIONS, centroids_fsl).unwrap();
        ivf_params.max_iters = COVERED_MAX_ITERS as usize;
        ivf_params.sample_rate = TWO_FRAG_SAMPLE_RATE;
        ivf_params
    }

    /// Params for `index_type` over the covering fixture, with IVF (and PQ, where the type
    /// needs it) trained from `dataset`. Shared by the covering and carried-vector lifecycle
    /// tests so each of them parametrizes over the same quantizer families.
    async fn params_for_index_type(dataset: &Dataset, index_type: &str) -> VectorIndexParams {
        match index_type {
            "IVF_PQ" => {
                let (ivf_params, pq_params) = prepare_covered_ivf_pq(dataset, "vector").await;
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
            }
            "IVF_SQ" => VectorIndexParams::with_ivf_sq_params(
                DistanceType::L2,
                prepare_covered_ivf(dataset, "vector").await,
                SQBuildParams::default(),
            ),
            "IVF_RQ" => VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                prepare_covered_ivf(dataset, "vector").await,
                RQBuildParams::new(1),
            ),
            "IVF_FLAT" => VectorIndexParams::with_ivf_flat_params(
                DistanceType::L2,
                prepare_covered_ivf(dataset, "vector").await,
            ),
            "IVF_HNSW_PQ" => {
                let (ivf_params, pq_params) = prepare_covered_ivf_pq(dataset, "vector").await;
                VectorIndexParams::with_ivf_hnsw_pq_params(
                    DistanceType::L2,
                    ivf_params,
                    lightweight_hnsw_params(),
                    pq_params,
                )
            }
            "IVF_HNSW_FLAT" => VectorIndexParams::ivf_hnsw(
                DistanceType::L2,
                prepare_covered_ivf(dataset, "vector").await,
                lightweight_hnsw_params(),
            ),
            "IVF_HNSW_SQ" => VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::L2,
                prepare_covered_ivf(dataset, "vector").await,
                lightweight_hnsw_params(),
                SQBuildParams::default(),
            ),
            other => panic!("unexpected index type {other}"),
        }
    }

    async fn prepare_global_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();

        let dim = vectors.value_length() as usize;
        assert_eq!(dim, TWO_FRAG_DIM, "unexpected vector dimension");

        let values = vectors.values().as_primitive::<Float32Type>();
        let kmeans_params = KMeansParams::new(None, TWO_FRAG_MAX_ITERS, 1, DistanceType::L2);
        let kmeans = train_kmeans::<Float32Type>(
            values,
            kmeans_params,
            dim,
            TWO_FRAG_NUM_PARTITIONS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .unwrap();

        let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
        let centroids_fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(TWO_FRAG_NUM_PARTITIONS, centroids_fsl).unwrap();
        ivf_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
        ivf_params.sample_rate = TWO_FRAG_SAMPLE_RATE;
        ivf_params
    }

    async fn build_segments_for_fragment_groups(
        dataset: &mut Dataset,
        fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
        params: &VectorIndexParams,
        index_name: &str,
    ) -> Vec<IndexMetadata> {
        let mut segments = Vec::new();

        for fragments in fragment_groups {
            let mut builder = dataset.create_index_builder(&["vector"], IndexType::Vector, params);
            builder = builder.name(index_name.to_string()).fragments(fragments);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }

        segments
    }

    async fn build_ivfpq_for_fragment_groups(
        dataset: &mut Dataset,
        fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
        ivf_params: &IvfBuildParams,
        pq_params: &PQBuildParams,
        index_name: &str,
    ) {
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params.clone(),
            pq_params.clone(),
        );

        let segments =
            build_segments_for_fragment_groups(dataset, fragment_groups, &params, index_name).await;
        let committed_segments =
            build_distributed_segments(dataset, segments, params.index_type(), index_name).await;
        assert!(!committed_segments.is_empty());
    }

    fn assert_centroids_equal(reference: &serde_json::Value, candidate: &serde_json::Value) {
        let centroids_a = reference["centroids"]
            .as_array()
            .expect("centroids should be an array");
        let centroids_b = candidate["centroids"]
            .as_array()
            .expect("centroids should be an array");
        assert_eq!(
            centroids_a.len(),
            centroids_b.len(),
            "num centroids mismatch",
        );
        for (row_a, row_b) in centroids_a.iter().zip(centroids_b.iter()) {
            let row_a = row_a
                .as_array()
                .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_a));
            let row_b = row_b
                .as_array()
                .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_b));
            assert_eq!(row_a.len(), row_b.len(), "centroid dim mismatch");
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                let fa = va.as_f64().expect("centroid must be numeric") as f32;
                let fb = vb.as_f64().expect("centroid must be numeric") as f32;
                assert!(
                    (fa - fb).abs() <= 1e-4,
                    "centroid mismatch: {} vs {}",
                    fa,
                    fb
                );
            }
        }
    }

    fn sum_partition_sizes(indices: &[serde_json::Value]) -> Vec<u64> {
        let mut totals = Vec::new();
        for index in indices {
            let partitions = index["partitions"]
                .as_array()
                .expect("partitions should be an array");
            if totals.is_empty() {
                totals.resize(partitions.len(), 0);
            } else {
                assert_eq!(totals.len(), partitions.len(), "num partitions mismatch");
            }
            for (total, partition) in totals.iter_mut().zip(partitions.iter()) {
                *total += partition["size"].as_u64().expect("partition size");
            }
        }
        totals
    }

    fn assert_ivf_layout_compatible(stats_a: &serde_json::Value, stats_b: &serde_json::Value) {
        let indices_a = stats_a["indices"]
            .as_array()
            .expect("indices should be an array");
        let indices_b = stats_b["indices"]
            .as_array()
            .expect("indices should be an array");
        assert!(
            !indices_a.is_empty() && !indices_b.is_empty(),
            "indices should not be empty",
        );

        let reference = &indices_a[0];
        for index in indices_a.iter().skip(1).chain(indices_b.iter()) {
            assert_centroids_equal(reference, index);
        }

        let sizes_a = sum_partition_sizes(indices_a);
        let sizes_b = sum_partition_sizes(indices_b);
        assert_eq!(sizes_a, sizes_b, "aggregated partition sizes mismatch");
    }

    /// Commit caller-defined segment groups as one logical index.
    async fn build_distributed_segments(
        dataset: &mut Dataset,
        segments: Vec<IndexMetadata>,
        _index_type: IndexType,
        index_name: &str,
    ) -> Vec<IndexMetadata> {
        dataset
            .commit_existing_index_segments(index_name, "vector", segments.clone())
            .await
            .unwrap();
        segments
    }

    #[tokio::test]
    async fn test_ivfpq_recall_performance_on_two_frags_single_vs_split() {
        const INDEX_NAME: &str = "vector_idx";

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        let (schema, batches) = make_two_fragment_batches();

        let ds_single_uri = format!("{}/single", base_uri);
        let ds_split_uri = format!("{}/split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        let fragments_single = ds_single.get_fragments();
        assert!(
            fragments_single.len() >= 2,
            "expected at least 2 fragments in ds_single, got {}",
            fragments_single.len()
        );
        let fragments_split = ds_split.get_fragments();
        assert!(
            fragments_split.len() >= 2,
            "expected at least 2 fragments in ds_split, got {}",
            fragments_split.len()
        );

        let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;

        let group_single = vec![
            fragments_single[0].id() as u32,
            fragments_single[1].id() as u32,
        ];
        build_ivfpq_for_fragment_groups(
            &mut ds_single,
            vec![group_single],
            &ivf_params,
            &pq_params,
            INDEX_NAME,
        )
        .await;

        let group0 = vec![fragments_split[0].id() as u32];
        let group1 = vec![fragments_split[1].id() as u32];
        build_ivfpq_for_fragment_groups(
            &mut ds_split,
            vec![group0, group1],
            &ivf_params,
            &pq_params,
            INDEX_NAME,
        )
        .await;

        let stats_single_json = ds_single.index_statistics(INDEX_NAME).await.unwrap();
        let stats_split_json = ds_split.index_statistics(INDEX_NAME).await.unwrap();
        let stats_single: serde_json::Value = serde_json::from_str(&stats_single_json).unwrap();
        let stats_split: serde_json::Value = serde_json::from_str(&stats_split_json).unwrap();
        assert_ivf_layout_compatible(&stats_single, &stats_split);
        assert_eq!(
            stats_single["num_indexed_rows"],
            stats_split["num_indexed_rows"]
        );

        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                let row_ids = result[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<u64>>();
                ids_per_query.push(row_ids);
            }
            ids_per_query
        }

        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;

        assert_eq!(
            ids_single, ids_split,
            "single vs split index returned different Top-K row ids",
        );
    }

    #[rstest]
    #[case::ivf_flat(IndexType::IvfFlat)]
    #[case::ivf_pq(IndexType::IvfPq)]
    #[case::ivf_sq(IndexType::IvfSq)]
    #[case::ivf_rq(IndexType::IvfRq)]
    #[tokio::test]
    async fn test_distributed_vector_build_commits_multiple_segments_and_preserves_query_results(
        #[case] index_type: IndexType,
    ) {
        const INDEX_NAME: &str = "vector_idx";
        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        // Generate the data once, then write it twice to two independent dataset URIs.
        let (schema, batches) = make_two_fragment_batches();

        let ds_single_uri = format!("{}/single", base_uri);
        let ds_split_uri = format!("{}/split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        // Ensure we have at least 2 fragments.
        let fragments_single = ds_single.get_fragments();
        assert!(
            fragments_single.len() >= 2,
            "expected at least 2 fragments in ds_single, got {}",
            fragments_single.len()
        );
        let fragments_split = ds_split.get_fragments();
        assert!(
            fragments_split.len() >= 2,
            "expected at least 2 fragments in ds_split, got {}",
            fragments_split.len()
        );

        let distributed_params = match index_type {
            IndexType::IvfFlat => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
            }
            IndexType::IvfPq => {
                let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
            }
            IndexType::IvfSq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_sq_params(
                    DistanceType::L2,
                    ivf_params,
                    SQBuildParams::default(),
                )
            }
            IndexType::IvfRq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_rq_params(
                    DistanceType::L2,
                    ivf_params,
                    RQBuildParams::with_rotation_type(1, RQRotationType::Fast),
                )
            }
            other => panic!("unsupported test index type: {}", other),
        };

        ds_single
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &distributed_params,
                true,
            )
            .await
            .unwrap();

        let fragment_groups = fragments_split
            .iter()
            .map(|fragment| vec![fragment.id() as u32])
            .collect::<Vec<_>>();
        let expected_segment_count = fragment_groups.len();
        let segments = build_segments_for_fragment_groups(
            &mut ds_split,
            fragment_groups,
            &distributed_params,
            INDEX_NAME,
        )
        .await;
        let segments =
            build_distributed_segments(&mut ds_split, segments, index_type, INDEX_NAME).await;
        assert_eq!(segments.len(), expected_segment_count);
        for segment in &segments {
            let segment_index = ds_split
                .indices_dir()
                .clone()
                .join(segment.uuid.to_string())
                .join(crate::index::INDEX_FILE_NAME);
            assert!(
                ds_split
                    .object_store
                    .as_ref()
                    .exists(&segment_index)
                    .await
                    .unwrap(),
                "segment file should exist at {}",
                segment_index
            );
        }

        let committed_segments = ds_split.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(committed_segments.len(), expected_segment_count);
        for committed in committed_segments {
            let covered_fragments = committed
                .fragment_bitmap
                .as_ref()
                .expect("distributed segment should have fragment coverage");
            assert_eq!(covered_fragments.len(), 1);
        }

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                let row_ids = result[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<u64>>();
                ids_per_query.push(row_ids);
            }
            ids_per_query
        }

        // Collect a deterministic query set from ds_single.
        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;

        if index_type == IndexType::IvfRq {
            for row_ids in &ids_split {
                assert_eq!(
                    row_ids.len(),
                    K,
                    "distributed IVF_RQ query should still return exactly {K} row ids",
                );
            }
        } else {
            assert_eq!(
                ids_single, ids_split,
                "single vs segmented distributed index returned different Top-K row ids",
            );
        }
    }

    #[rstest]
    #[case::ivf_flat(IndexType::IvfFlat)]
    #[case::ivf_pq(IndexType::IvfPq)]
    #[case::ivf_sq(IndexType::IvfSq)]
    #[tokio::test]
    async fn test_distributed_vector_grouped_build_allows_concurrent_group_execution(
        #[case] index_type: IndexType,
    ) {
        const INDEX_NAME: &str = "grouped_idx";
        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        let (schema, batches) = make_two_fragment_batches();
        let ds_single_uri = format!("{}/grouped_single", base_uri);
        let ds_split_uri = format!("{}/grouped_split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        let distributed_params = match index_type {
            IndexType::IvfFlat => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
            }
            IndexType::IvfPq => {
                let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
            }
            IndexType::IvfSq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_sq_params(
                    DistanceType::L2,
                    ivf_params,
                    SQBuildParams::default(),
                )
            }
            other => panic!("unsupported test index type: {}", other),
        };

        ds_single
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &distributed_params,
                true,
            )
            .await
            .unwrap();

        let fragment_groups = ds_split
            .get_fragments()
            .into_iter()
            .map(|fragment| vec![fragment.id() as u32])
            .collect::<Vec<_>>();
        let segments = build_segments_for_fragment_groups(
            &mut ds_split,
            fragment_groups,
            &distributed_params,
            INDEX_NAME,
        )
        .await;

        assert!(segments.len() >= 4);
        let grouped_inputs = segments
            .chunks(2)
            .map(|group| group.to_vec())
            .collect::<Vec<_>>();
        let mut expected_fragment_coverage = grouped_inputs
            .iter()
            .map(|group| {
                group
                    .iter()
                    .flat_map(|partial| {
                        partial
                            .fragment_bitmap
                            .as_ref()
                            .expect("partial shard should have fragment coverage")
                            .iter()
                    })
                    .sorted()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        expected_fragment_coverage.sort();

        let grouped_segments = futures::future::try_join_all(
            grouped_inputs
                .into_iter()
                .map(|group| ds_split.merge_existing_index_segments(group)),
        )
        .await
        .unwrap();
        let grouped_segments =
            build_distributed_segments(&mut ds_split, grouped_segments, index_type, INDEX_NAME)
                .await;
        assert_eq!(grouped_segments.len(), expected_fragment_coverage.len());
        let mut actual_fragment_coverage = grouped_segments
            .iter()
            .map(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .expect("segment should have fragment coverage")
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        actual_fragment_coverage.sort();
        assert_eq!(
            actual_fragment_coverage, expected_fragment_coverage,
            "built segment coverage should equal the union of its source partial shards",
        );

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                ids_per_query.push(
                    result[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .copied()
                        .collect(),
                );
            }
            ids_per_query
        }

        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;
        if matches!(index_type, IndexType::IvfSq) {
            for (single, split) in ids_single.iter().zip(ids_split.iter()) {
                assert_eq!(single.len(), split.len());
                let overlap = single
                    .iter()
                    .filter(|row_id| split.contains(row_id))
                    .count();
                assert!(
                    overlap >= K / 3,
                    "single vs segmented distributed SQ index returned too little top-k overlap",
                );
            }
        } else {
            assert_eq!(ids_single, ids_split);
        }
    }

    #[tokio::test]
    async fn test_distributed_vector_plan_rejects_overlapping_fragment_coverage() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_two_fragment_batches();
        let dataset_uri = format!("{}/overlap_fragments", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragment = dataset.get_fragments()[0].id() as u32;
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
        );
        let mut segments = Vec::new();

        for _ in 0..2 {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        let err = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("overlapping fragment coverage"));
    }

    #[tokio::test]
    async fn test_distributed_vector_build_supports_hnsw_variants() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_two_fragment_batches();
        let dataset_uri = format!("{}/distributed_hnsw_supported", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
        );
        let mut segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    /// A distributed (sharded, precomputed-IVF) build must support covering ("included")
    /// columns: each shard writes the covered columns into its own segment's storage, the
    /// segment metadata advertises them as the trailing entries of `fields`, and a covered
    /// projection over the committed index skips the base-table take while returning
    /// correct, row-aligned covered values.
    #[tokio::test]
    async fn test_distributed_vector_build_supports_covering_columns() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_covered_test_batches();
        let dataset_uri = format!("{}/distributed_covered", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let payload_field_id = dataset.schema().field("payload").unwrap().id;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["id".to_string(), "payload".to_string()]);

        // One covered segment per fragment, over ALL fragments, so the committed index gives
        // full coverage -- no uncovered flat delta-scan can mask a covering defect. Each shard
        // reads the covering columns from its own fragment subset and writes them into its
        // segment's storage.
        let mut segments = Vec::new();
        for fragment in fragments.iter() {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            assert_eq!(
                segment.covering_fields,
                vec![id_field_id, payload_field_id],
                "distributed covered segment must record both covered columns' field ids"
            );
            // Carried columns are a trailing subset of `fields` with the keyed vector field
            // first; `IndexMetadata::validate_covering_fields` rejects any other shape.
            assert_eq!(
                segment.fields,
                vec![vector_field_id, id_field_id, payload_field_id],
                "covered fields must be the trailing entries of the segment's fields"
            );
            segments.push(segment);
        }

        dataset
            .commit_existing_index_segments("vec_idx", "vector", segments)
            .await
            .unwrap();

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let q = q.as_primitive::<Float32Type>();

        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id", "payload"]).unwrap();

        // The base-table take renders as `LanceRead` in the explained plan, so that -- not
        // the string "Take" -- is what its absence has to be asserted on. A flat-KNN
        // fallback has no take either, so pin the ANN path first or the assertion below
        // could pass without the index being read at all.
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "the covered query must go through the committed index; plan was:\n{plan}"
        );
        assert!(
            !plan.contains("LanceRead"),
            "covered projection ['id','payload'] should skip the base-table take; plan was:\n{plan}"
        );

        let covered = scan.try_into_batch().await.unwrap();
        let row_ids = covered
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        assert!(!row_ids.is_empty());

        // Every covered column -- including the nullable `payload` with its nulls -- must be
        // row-aligned with an independent base-table take. Full array equality checks values
        // AND validity, so a dropped or shifted null slot fails here; the null count is
        // asserted first so that claim cannot rest on an all-valid result set.
        assert!(
            covered
                .column_by_name("payload")
                .expect("covered 'payload' column")
                .null_count()
                > 0,
            "the returned rows must include a null covered 'payload'"
        );
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let base = dataset
            .take_rows(
                &row_id_vec,
                crate::dataset::ProjectionRequest::from_columns(
                    ["id", "payload"],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        for col in ["id", "payload"] {
            assert_eq!(
                covered.column_by_name(col).expect("covered column"),
                base.column_by_name(col).expect("base-table column"),
                "covered '{col}' must match the base-table value (incl. nulls) for each row"
            );
        }

        // Take elision is worthless if the covered search returns the wrong neighbors, so also
        // gate on recall against brute-force ground truth (all 4 partitions probed).
        let returned: HashSet<u64> = row_id_vec.iter().copied().collect();
        let truth = ground_truth(&dataset, "vector", q, 10, DistanceType::L2).await;
        let recall = truth.intersection(&returned).count() as f32 / truth.len() as f32;
        assert!(
            recall >= 0.5,
            "covered distributed build recall {recall} < 0.5 (returned {returned:?}, truth {truth:?})"
        );
    }

    /// The cross-shard segment *merger* must also carry covering columns: merging covered
    /// shards into one unified auxiliary index must retain the covered payload (not drop it at
    /// the merger's rebuilt output schema), so a covered projection over the merged index still
    /// skips the take and returns correct values. Parametrized over quantizer family so each
    /// type's internal-name list (which the merger uses to detect the covering columns) is
    /// exercised on the covered-merge path, including IVF_HNSW_PQ for the HNSW-variant lists.
    // Note: IVF_RQ (RaBitQ) is intentionally excluded -- distributed RQ *merge* fails
    // independently of covering (shards train different `fast_rotation_signs`, so the merger's
    // structural-equality check rejects them); covered RQ works on the per-segment-commit path.
    /// The HNSW arms build with `lightweight_hnsw_params()`: this test asserts that the merger
    /// carries covering columns through each quantizer's internal-name list, which graph quality
    /// has no bearing on, and a full-size graph would push the cases past the one-second budget.
    /// `hnsw_flat` and `hnsw_sq` are not redundant with `flat`/`sq`: they are the only cases that
    /// reach the `IvfHnswFlat` / `IvfHnswSq` arms of `covering_fields_from_shard_schema`. Without
    /// them, mapping `IvfHnswSq` to `FLAT_INTERNAL_COLUMNS` would classify `__sq_code` as a
    /// covering column -- a duplicate field in the writer schema -- with no test failing.
    #[rstest]
    #[case::pq("IVF_PQ")]
    #[case::sq("IVF_SQ")]
    #[case::flat("IVF_FLAT")]
    #[case::hnsw_pq("IVF_HNSW_PQ")]
    #[case::hnsw_flat("IVF_HNSW_FLAT")]
    #[case::hnsw_sq("IVF_HNSW_SQ")]
    #[tokio::test]
    async fn test_distributed_vector_merge_supports_covering_columns(#[case] index_type: &str) {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_covered_test_batches();
        let dataset_uri = format!("{}/distributed_covered_merge_{index_type}", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let payload_field_id = dataset.schema().field("payload").unwrap().id;

        let mut params = params_for_index_type(&dataset, index_type).await;
        params.covering_columns(vec!["id".to_string(), "payload".to_string()]);

        // All fragments, so the merged index gives full coverage (no uncovered delta scan).
        let mut segments = Vec::new();
        for fragment in fragments.iter() {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        // Merge the covered shards into one unified segment, then commit it.
        let merged = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap();
        assert_eq!(
            merged.covering_fields,
            vec![id_field_id, payload_field_id],
            "merged covered segment must retain both covered columns' field ids"
        );
        dataset
            .commit_existing_index_segments("vec_idx", "vector", vec![merged])
            .await
            .unwrap();

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let q = q.as_primitive::<Float32Type>();

        let mut scan = dataset.scan();
        scan.nearest("vector", q, 10).unwrap();
        scan.nprobes(4);
        scan.with_row_id();
        scan.project(&["id", "payload"]).unwrap();

        // As above: pin the ANN path, then assert the take (`LanceRead`) is gone.
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "the covered query must go through the merged index; plan was:\n{plan}"
        );
        assert!(
            !plan.contains("LanceRead"),
            "covered projection over the merged index should skip the base-table take; \
             plan was:\n{plan}"
        );

        let covered = scan.try_into_batch().await.unwrap();
        let row_ids = covered
            .column_by_name(ROW_ID)
            .expect("row id column")
            .as_primitive::<UInt64Type>();
        assert!(!row_ids.is_empty());

        // Both covered columns (incl. the nullable `payload` with its nulls) must be
        // row-aligned with an independent base-table take over the merged index. Full array
        // equality checks validity too, and the null count is asserted first so that claim
        // cannot rest on an all-valid result set.
        assert!(
            covered
                .column_by_name("payload")
                .expect("covered 'payload' column")
                .null_count()
                > 0,
            "the returned rows must include a null covered 'payload'"
        );
        let row_id_vec: Vec<u64> = row_ids.values().to_vec();
        let base = dataset
            .take_rows(
                &row_id_vec,
                crate::dataset::ProjectionRequest::from_columns(
                    ["id", "payload"],
                    dataset.schema(),
                ),
            )
            .await
            .unwrap();
        for col in ["id", "payload"] {
            assert_eq!(
                covered.column_by_name(col).expect("covered column"),
                base.column_by_name(col).expect("base-table column"),
                "covered '{col}' over the merged index must match the base-table value for each row"
            );
        }

        // Gate on recall too, so a merge that returns the wrong neighbors is caught.
        let returned: HashSet<u64> = row_id_vec.iter().copied().collect();
        let truth = ground_truth(&dataset, "vector", q, 10, DistanceType::L2).await;
        let recall = truth.intersection(&returned).count() as f32 / truth.len() as f32;
        assert!(
            recall >= 0.5,
            "covered merged {index_type} recall {recall} < 0.5 (returned {returned:?}, truth {truth:?})"
        );
    }

    /// Distributed builds must carry refine vectors too, on both shapes: committing one
    /// segment per shard, and merging the shards into one unified segment first.
    ///
    /// The merge shape is the one with its own machinery: the cross-shard merger classifies
    /// a shard's carried columns by *excluding* the storage's internal names, so carried
    /// vectors -- which sit under the indexed column's own name at rest -- are only picked
    /// up because that name is not internal. `payload` is covered alongside so the
    /// projection is served from the index and any base-table read the plan shows can only
    /// be refine's own.
    #[rstest]
    #[case::per_segment_commit(false)]
    #[case::cross_shard_merge(true)]
    #[tokio::test]
    async fn test_distributed_build_carries_refine_vectors(#[case] merge_shards: bool) {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let uri = format!(
            "{}/distributed_refine_{}",
            test_dir.as_str(),
            if merge_shards { "merged" } else { "segments" }
        );
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 2,
            "need several shards to distribute over"
        );
        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let payload_field_id = dataset.schema().field("payload").unwrap().id;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);

        // One shard per fragment, over every fragment, so the committed index gives full
        // coverage and no uncovered delta scan can mask a defect.
        let mut segments = Vec::new();
        for fragment in fragments.iter() {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vec_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            assert_eq!(
                segment.covering_fields,
                vec![payload_field_id, vector_field_id],
                "each shard must declare the covering column and the carried vectors"
            );
            segments.push(segment);
        }

        let to_commit = match merge_shards {
            true => {
                let merged = dataset
                    .merge_existing_index_segments(segments)
                    .await
                    .unwrap();
                assert_eq!(
                    merged.covering_fields,
                    vec![payload_field_id, vector_field_id],
                    "the unified segment must retain the carried vectors"
                );
                vec![merged]
            }
            false => segments,
        };
        dataset
            .commit_existing_index_segments("vec_idx", "vector", to_commit)
            .await
            .unwrap();

        assert_refine_served_from_index(&dataset, query.as_ref(), "distributed").await;
    }

    /// A merged covered segment must itself be usable as an input to a later merge --
    /// the hierarchical/incremental distributed workflow. The merger requires every
    /// covered shard to carry its source field ids, so the merged output has to carry
    /// them too; otherwise the merger rejects its own product and tells the user to
    /// rebuild a shard that only the merger can produce.
    #[tokio::test]
    async fn test_distributed_covered_merge_output_can_be_merged_again() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_covered_test_batches();
        let dataset_uri = format!("{}/distributed_covered_remerge", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 3,
            "need >= 3 fragments so the second-stage merge has more than one input"
        );
        // IVF_FLAT: a merged *PQ* segment stores transposed codes and the merger requires
        // row-major shards, so PQ cannot be re-merged for reasons unrelated to covering.
        // FLAT and SQ can, which is where the missing stamp actually bit.
        let mut params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_covered_ivf(&dataset, "vector").await,
        );
        params.covering_columns(vec!["payload".to_string()]);

        let mut segments = Vec::new();
        for fragment in fragments.iter() {
            segments.push(
                dataset
                    .create_index_builder(&["vector"], IndexType::Vector, &params)
                    .name("vec_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }

        // Stage 1: merge the first two shards.
        let rest = segments.split_off(2);
        let merged = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap();
        // Stage 2: feed that merged segment back in alongside the remaining shards. This
        // is the hierarchical workflow the missing stamp used to make impossible.
        let mut stage_two = vec![merged];
        stage_two.extend(rest);
        let remerged = dataset
            .merge_existing_index_segments(stage_two)
            .await
            .expect("a merged covered segment must be mergeable again");
        assert_eq!(
            remerged.covering_fields,
            vec![dataset.schema().field("payload").unwrap().id],
            "the re-merged segment must still declare the covered field"
        );
        dataset
            .commit_existing_index_segments("vec_idx", "vector", vec![remerged])
            .await
            .unwrap();
    }

    /// Same covering column *name and type* across shards, but two different logical
    /// fields: `payload` is dropped and re-added between the two shard builds, so it
    /// comes back with a fresh field id. Name+type comparison accepts this and
    /// `concat_batches` would then stack shard A's values for the dropped field under
    /// the re-added one -- covered queries over shard A's rows serving the old column.
    ///
    /// Carried columns live in `fields`, so the two shards disagree on `fields` as well as
    /// on `covering_fields`, and the segment-level check rejects them before the auxiliary
    /// merger runs. The merger's own `COVERING_FIELD_IDS_KEY` comparison guards the same
    /// case one layer down, for drivers calling `merge_partial_vector_auxiliary_files`
    /// directly; what this test pins is that the dataset API never lets the corruption
    /// through.
    #[tokio::test]
    async fn test_distributed_vector_merge_rejects_covering_of_different_field_ids() {
        use crate::dataset::NewColumnTransform;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        let payload = Arc::new(UInt64Array::from_iter_values(0..TWO_FRAG_NUM_ROWS as u64));
        let values = generate_random_array_with_range(TWO_FRAG_NUM_ROWS * COVERED_DIM, 0.0..1.0);
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), COVERED_DIM as i32)
                .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("payload", DataType::UInt64, true),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![payload, vectors]).unwrap();
        let dataset_uri = format!("{}/distributed_covered_merge_field_id", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, vec![batch]).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);

        // Shard A covers `payload` as it exists now.
        let segment_a = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vec_idx".to_string())
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        let original_field_id = dataset.schema().field("payload").unwrap().id;

        // Drop and re-add `payload`: same name, same type, new field id. The re-add is
        // metadata-only (AllNulls), so no data file changes and nothing else notices.
        dataset.drop_columns(&["payload"]).await.unwrap();
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(arrow_schema::Schema::new(vec![
                    Field::new("payload", DataType::UInt64, true),
                ]))),
                None,
                None,
            )
            .await
            .unwrap();
        let new_field_id = dataset.schema().field("payload").unwrap().id;
        assert_ne!(
            original_field_id, new_field_id,
            "re-adding the column must produce a new field id, or this test proves nothing"
        );

        let fragments = dataset.get_fragments();
        let segment_b = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vec_idx".to_string())
            .fragments(vec![fragments[1].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();

        let err = dataset
            .merge_existing_index_segments(vec![segment_a, segment_b])
            .await
            .expect_err("merging shards covering different field ids must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("identical fields"),
            "error should describe the covered-field disagreement; was: {msg}"
        );
    }

    #[tokio::test]
    async fn test_optimize_refuses_to_restamp_rebound_covering() {
        use crate::dataset::NewColumnTransform;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let dataset_uri = format!("{}/optimize_refuses_restamp", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let all_fragments: Vec<u32> = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect();

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);

        let mut segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vec_idx".to_string())
            .fragments(all_fragments)
            .execute_uncommitted()
            .await
            .unwrap();
        let original_field_id = dataset.schema().field("payload").unwrap().id;

        // Drop and re-add `payload`: same name, same type, new field id. The re-add is
        // metadata-only (AllNulls), so no data file changes and no fragment looks stale.
        dataset.drop_columns(&["payload"]).await.unwrap();
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(arrow_schema::Schema::new(vec![
                    Field::new("payload", DataType::UInt64, true),
                ]))),
                None,
                None,
            )
            .await
            .unwrap();
        let new_field_id = dataset.schema().field("payload").unwrap().id;
        assert_ne!(
            original_field_id, new_field_id,
            "re-adding the column must produce a new field id, or this test proves nothing"
        );

        // The segment keeps its true build version, so the staleness pass runs and passes;
        // only the id rebind is wrong.
        for id in segment.fields.iter_mut() {
            if *id == original_field_id {
                *id = new_field_id;
            }
        }
        for id in segment.covering_fields.iter_mut() {
            if *id == original_field_id {
                *id = new_field_id;
            }
        }

        dataset
            .commit_existing_index_segments("vec_idx", "vector", vec![segment])
            .await
            .unwrap();

        // Before optimize the mismatch is visible and the read path does the right thing:
        // the stamped source id disagrees with the declaration, so payload comes from the
        // base table and is all null.
        let mut scan = dataset.scan();
        scan.nearest("vector", query.as_ref(), 10).unwrap();
        scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
        scan.project(&["payload"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let payload = batch["payload"].as_primitive::<UInt64Type>();
        assert_eq!(
            payload.null_count(),
            payload.len(),
            "precondition: the rebound declaration must fall back to the base table"
        );

        // A rebuild copies that payload by name and would re-derive the stamp from the
        // CURRENT schema, making the declaration and the stamp agree -- which would start
        // serving the old field's values and elide the base-table read. Refuse instead.
        let err = dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .expect_err("optimize must not re-stamp a covering payload onto a different field id");
        let msg = err.to_string();
        assert!(
            msg.contains("covering values stamped with source field ids")
                && msg.contains(&original_field_id.to_string())
                && msg.contains(&new_field_id.to_string()),
            "error must name both the stored and the current ids; got: {msg}"
        );

        // And the index is left in the state the read path already handles.
        let mut scan = dataset.scan();
        scan.nearest("vector", query.as_ref(), 10).unwrap();
        scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
        scan.project(&["payload"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let payload = batch["payload"].as_primitive::<UInt64Type>();
        assert_eq!(
            payload.null_count(),
            payload.len(),
            "after the refusal the payload must still come from the base table"
        );
    }

    /// Refine vectors are as wide as the vector column itself, so reading them with every
    /// partition probed would undo the bound that reading covering per survivor exists to
    /// give: codes are scanned for every row on every probe, refine values are needed for
    /// at most `k`. They are storage-owned but deliberately NOT part of the per-partition
    /// read set -- the distinction `DEFERRED_INTERNAL_COLUMNS` exists to draw.
    #[tokio::test]
    async fn test_refine_vectors_are_not_read_with_every_partition() {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/refine_not_per_partition", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", "vec_idx").await;
        let storage = ctx
            .ivf()
            .load_partition_storage(0, PartitionColumns::Internal, None)
            .await
            .unwrap();
        let loaded: Vec<String> = storage
            .batch()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert!(
            !loaded.iter().any(|n| n == "vector"),
            "a per-partition load must leave the carried vectors on disk; loaded {loaded:?}"
        );

        // They are covering, so they are read for survivors instead -- which is the whole
        // point: codes are scanned for every row on every probe, carried vectors are needed
        // for at most `k`.
        let physical: Vec<String> = ctx
            .ivf()
            .physical_covering_fields()
            .unwrap()
            .iter()
            .map(|(_, f)| f.name().clone())
            .collect();
        assert_eq!(
            physical,
            vec!["vector".to_string()],
            "carried vectors must be servable as covering, just not per partition"
        );
    }

    /// Covering columns and refine vectors must coexist. Covering is classified by
    /// name-exclusion against the storage's own internal set and then matched against the
    /// `covering_field_ids` stamp by *arity*; a refine column counted as covering makes the
    /// two disagree, and `physical_covering_fields_from_schema` answers "no covering at all"
    /// on a mismatch -- silently withdrawing a payload the index really does carry.
    #[tokio::test]
    async fn test_refine_vectors_do_not_withdraw_covering_columns() {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/refine_plus_covering", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", "vec_idx").await;
        let physical = ctx.ivf().physical_covering_fields().unwrap();
        let names: Vec<String> = physical.iter().map(|(_, f)| f.name().clone()).collect();
        assert_eq!(
            names,
            vec!["payload".to_string(), "vector".to_string()],
            "both the declared covering column and the carried vectors must be servable"
        );
    }

    /// The point of carrying refine vectors: `refine` re-ranks from index storage instead
    /// of taking full-precision vectors from the base table.
    ///
    /// Asserted against a control rather than on its own. "No base-table read" is only
    /// meaningful if the same query *does* read the base table without the option -- a bare
    /// absence assertion passes for any number of unrelated reasons.
    #[tokio::test]
    async fn test_refine_reads_vectors_from_the_index_not_the_base_table() {
        async fn plan_for(store_vectors: bool, dir: &TempStrDir, name: &str) -> (String, usize) {
            let (schema, batches) = make_covered_test_batches();
            let uri = format!("{}/{}", dir.as_str(), name);
            let query = batches[0]["vector"].as_fixed_size_list().value(0);
            let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

            let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
            let mut params =
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
            // `payload` is covered so the index can serve the user's projection; the only
            // thing that could still need the base table is refine's own vector fetch.
            params.covering_columns(vec!["payload".to_string()]);
            params.store_vectors_for_refine(store_vectors);
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some("vec_idx".into()),
                    &params,
                    true,
                )
                .await
                .unwrap();

            let mut scan = dataset.scan();
            scan.nearest("vector", query.as_ref(), 10).unwrap();
            scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
            scan.refine(2);
            // Project only `payload`: the user never asks for the vector column, so nothing
            // but refine itself needs it. This is the shape the narrowing would strip.
            scan.project(&["payload"]).unwrap();
            let plan = scan.explain_plan(true).await.unwrap();
            let rows = scan.try_into_batch().await.unwrap().num_rows();
            (plan, rows)
        }

        let test_dir = TempStrDir::default();
        let (plain_plan, plain_rows) = plan_for(false, &test_dir, "refine_plain").await;
        let (carried_plan, carried_rows) = plan_for(true, &test_dir, "refine_carried").await;

        // Control: without the option, refine must fetch vectors from the base table. The
        // covered `payload` is served from the index in both variants, so a base-table read
        // here can only be refine's.
        assert!(
            plain_plan.contains("LanceRead"),
            "control: refine without carried vectors must read the base table; plan:\n{plain_plan}"
        );
        assert!(
            !carried_plan.contains("LanceRead"),
            "refine must be served from the index when vectors are carried; plan:\n{carried_plan}"
        );
        assert_eq!(
            plain_rows, carried_rows,
            "carrying vectors must not change the result count"
        );
    }

    /// Carrying vectors buys nothing on a multivector index, so it is refused rather than
    /// silently doubling the index.
    ///
    /// The refine step runs *after* `MultivectorScoringExec` has re-grouped sub-vectors back
    /// to rows, so it scores the row-shaped `List` column. Index storage holds one
    /// `FixedSizeList` per sub-vector -- the shape the search itself needs -- which cannot
    /// substitute for it. A second copy of the widest column in the table, readable by
    /// nothing, is worth an error rather than a surprise.
    #[tokio::test]
    async fn test_store_vectors_for_refine_is_rejected_on_multivector() {
        let test_dir = TempStrDir::default();
        let (mut dataset, _) =
            generate_multivec_test_dataset::<Float32Type>(test_dir.as_str(), 0.0..1.0).await;

        // Multivector requires cosine.
        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::Cosine, 2);
        params.store_vectors_for_refine(true);

        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err("carrying vectors on a multivector index must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains("multivector"),
            "the error must name the option and why it does not apply; got: {msg}"
        );
    }

    /// Only the V3 covering-aware storages keep an extra column row-aligned with the code.
    /// A legacy build ignores the option outright, but `CreateIndexBuilder` still records
    /// the vector in `covering_fields`, so the index would advertise a payload its storage
    /// never wrote and every query would quietly fall back to the base table.
    #[tokio::test]
    async fn test_store_vectors_for_refine_requires_v3() {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/refine_requires_v3", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.store_vectors_for_refine(true);
        params.version(IndexFileVersion::Legacy);

        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err("a legacy build cannot carry refine vectors");
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains("V3"),
            "the error must name the option and the version it needs; got: {msg}"
        );
    }

    /// A flat quantizer is an identity copy of the vector column, and the flat IVF
    /// transformer adds no residual step, so under L2 the carried copy is byte-identical to
    /// the `FLAT_COLUMN` the index already stores -- and `refine` re-ranking flat distances
    /// is a no-op, since those distances are already exact. Carrying vectors there doubles
    /// the index for nothing, so it joins the other unproductive configurations the
    /// validator refuses rather than being silently accepted.
    #[rstest]
    #[case::flat("IVF_FLAT")]
    #[case::hnsw_flat("IVF_HNSW_FLAT")]
    #[tokio::test]
    async fn test_store_vectors_for_refine_rejects_flat_quantizers(#[case] index_type: &str) {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/refine_rejects_{index_type}", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let mut params = params_for_index_type(&dataset, index_type).await;
        params.store_vectors_for_refine(true);

        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err(
                "a flat quantizer already stores the vectors; carrying them again \
                         must be refused",
            );
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains("flat"),
            "the error must name the option and the quantizer; got: {msg}"
        );
    }

    /// Carried vectors are stored as a top-level covering column and read back through the
    /// same top-level-only resolution every other covering column uses, so a nested
    /// indexed column has nowhere to put them. Plain indexing of such a column still works,
    /// so the refusal has to be the option's, not the column's.
    #[tokio::test]
    async fn test_store_vectors_for_refine_rejects_nested_column() {
        use arrow_array::StructArray;

        const ROWS: usize = 512;
        const NDIM: usize = 32;
        let test_dir = TempStrDir::default();

        let values = generate_random_array_with_range::<Float32Type>(ROWS * NDIM, 0.0..1.0);
        let vectors =
            Arc::new(FixedSizeListArray::try_new_from_values(values, NDIM as i32).unwrap());
        let inner = Field::new("embedding", vectors.data_type().clone(), false);
        let data = Arc::new(StructArray::from(vec![(
            Arc::new(inner),
            vectors as ArrayRef,
        )]));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            data.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![data]).unwrap();
        let uri = format!("{}/refine_nested_column", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, vec![batch]).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        dataset
            .create_index(
                &["data.embedding"],
                IndexType::Vector,
                Some("plain_idx".into()),
                &params,
                true,
            )
            .await
            .expect("precondition: a nested vector column indexes normally");

        params.store_vectors_for_refine(true);
        let err = dataset
            .create_index(
                &["data.embedding"],
                IndexType::Vector,
                Some("refine_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err("carrying vectors for a nested indexed column must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains("data.embedding"),
            "the error must name the option and the column; got: {msg}"
        );
    }

    /// The carried copy travels under a reserved internal name until the transform chain
    /// has consumed the indexed column. An indexed column already using that name collides
    /// with the copy mid-flight, and the classifier that decides what storage carries as
    /// covering payload never counts it either.
    #[tokio::test]
    async fn test_store_vectors_for_refine_rejects_reserved_column_name() {
        use lance_index::vector::storage::REFINE_VECTOR_COLUMN;

        const ROWS: usize = 512;
        const NDIM: usize = 32;
        let test_dir = TempStrDir::default();

        let values = generate_random_array_with_range::<Float32Type>(ROWS * NDIM, 0.0..1.0);
        let vectors =
            Arc::new(FixedSizeListArray::try_new_from_values(values, NDIM as i32).unwrap());
        let schema = Arc::new(Schema::new(vec![Field::new(
            REFINE_VECTOR_COLUMN,
            vectors.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![vectors]).unwrap();
        let uri = format!("{}/refine_reserved_name", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, vec![batch]).await;

        let mut params = VectorIndexParams::ivf_pq(4, 8, 4, DistanceType::L2, 2);
        params.store_vectors_for_refine(true);
        let err = dataset
            .create_index(
                &[REFINE_VECTOR_COLUMN],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err("an indexed column using the scratch name must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains(REFINE_VECTOR_COLUMN),
            "the error must name the option and the reserved name; got: {msg}"
        );
    }

    /// Precomputed shuffle buffers arrive already quantized and without the raw vector
    /// column, so there is nothing left to copy aside by the time the build reads them --
    /// exactly the reason covering columns are refused with them.
    #[tokio::test]
    async fn test_store_vectors_for_refine_rejects_precomputed_shuffle_buffers() {
        use object_store::path::Path;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/refine_precomputed_buffers", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (mut ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        ivf_params.precomputed_shuffle_buffers =
            Some((Path::from("shuffle/data"), vec!["part0".to_string()]));
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.store_vectors_for_refine(true);

        let err = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .expect_err("carrying vectors from precomputed buffers must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("store_vectors_for_refine") && msg.contains("precomputed shuffle"),
            "the error must name the option and the incompatible source; got: {msg}"
        );
    }

    /// Optimize can split an oversized partition, which re-streams the affected rows
    /// through the transform chain on a different path from the initial build. That path
    /// must carry the vectors too, or the rebuilt partitions lose them while the declared
    /// storage schema still names the column.
    #[tokio::test]
    async fn test_partition_split_preserves_carried_vectors() {
        const INDEX_NAME: &str = "vector_idx";
        // IVF_PQ splits above MAX_PARTITION_SIZE_FACTOR * 8192 = 32_768 rows.
        const BASE_ROWS: usize = 512;
        const APPEND_ROWS: usize = 33_000;
        // Two clusters, but only one is grown past the split threshold.
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let (batch, schema) = generate_clustered_batch(BASE_ROWS, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_dir.as_str(),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let pq_params = PQBuildParams::new(4, 8);
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Grow only the first cluster, so exactly one partition crosses the threshold.
        let mut template = vec![0.0; DIM];
        template[0] = offsets[0];
        append_partition_templates(&mut dataset, APPEND_ROWS, &[template]).await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert!(
            ctx.num_partitions() > 2,
            "precondition: the partition must actually have split, stats: {}",
            ctx.stats_json()
        );
        for p in 0..ctx.num_partitions() {
            let storage = ctx
                .ivf()
                .load_partition_storage(p, PartitionColumns::All, None)
                .await
                .unwrap();
            assert!(
                storage.batch().column_by_name("vector").is_some(),
                "partition {p} lost its carried vectors across the split"
            );
        }
    }

    /// An explicit merge hands `StorageBuilder::build` two sources at once: batches read
    /// back from existing storage, where the carried vectors sit under the indexed column's
    /// own name, and freshly shuffled batches, where they are still under the in-flight
    /// scratch name. The two must agree on one name, or the merge fails outright.
    ///
    /// The split test above cannot catch this: it grows a single cluster, so the affected
    /// partition is served entirely by the split shuffle reader and the untouched
    /// partitions receive no fresh rows -- the two sources never meet in one build.
    #[rstest]
    #[case::pq("IVF_PQ")]
    #[case::sq("IVF_SQ")]
    #[case::rq("IVF_RQ")]
    #[case::hnsw_pq("IVF_HNSW_PQ")]
    #[case::hnsw_sq("IVF_HNSW_SQ")]
    #[tokio::test]
    async fn test_merge_keeps_carried_vectors(#[case] index_type: &str) {
        const INDEX_NAME: &str = "vec_idx";
        const APPENDED_ROWS: usize = 800;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let uri = format!(
            "{}/merge_keeps_carried_vectors_{index_type}",
            test_dir.as_str()
        );
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let mut params = params_for_index_type(&dataset, index_type).await;
        // `payload` is covered so the projection is served from the index too, which is what
        // lets the plan assertion attribute any base-table read to refine alone.
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        append_covered_rows(&mut dataset, APPENDED_ROWS).await;
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        // Both preconditions matter: one segment means the appended rows were folded into
        // the segment the merge read from, rather than left as a fresh delta -- which is the
        // only arrangement that puts stored and freshly shuffled batches in one storage
        // build, the thing this test exists to exercise.
        assert_eq!(
            ctx.stats()["num_indices"].as_u64().unwrap(),
            1,
            "precondition: the merge must leave a single segment; stats: {}",
            ctx.stats_json()
        );
        assert_eq!(
            ctx.stats()["num_indexed_rows"].as_u64().unwrap() as usize,
            TWO_FRAG_NUM_ROWS + APPENDED_ROWS,
            "precondition: the merged segment must cover every row; stats: {}",
            ctx.stats_json()
        );
        assert_refine_served_from_index(&dataset, query.as_ref(), index_type).await;
    }

    /// The counterpart to the split test above. Joining an undersized partition takes its
    /// rows from the *base table* and reassigns them, on a path that never runs the
    /// transform chain and so never sees the carried vectors. Those reassigned rows still
    /// meet the untouched partitions' stored copies in one storage build, so they have to
    /// arrive carrying the vectors too.
    #[rstest]
    #[case::pq("IVF_PQ")]
    #[case::sq("IVF_SQ")]
    #[case::rq("IVF_RQ")]
    #[case::hnsw_pq("IVF_HNSW_PQ")]
    #[case::hnsw_sq("IVF_HNSW_SQ")]
    #[tokio::test]
    async fn test_partition_join_keeps_carried_vectors(#[case] index_type: &str) {
        const INDEX_NAME: &str = "vec_idx";
        const APPENDED_ROWS: usize = 200;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let uri = format!(
            "{}/join_keeps_carried_vectors_{index_type}",
            test_dir.as_str()
        );
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let mut params = params_for_index_type(&dataset, index_type).await;
        // `payload` is covered so the projection is served from the index too, which is what
        // lets the plan assertion attribute any base-table read to refine alone.
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Every partition holds far fewer than `MIN_PARTITION_SIZE_PERCENT` of IVF_PQ's
        // target partition size, so the optimize below joins one away.
        append_covered_rows(&mut dataset, APPENDED_ROWS).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert!(
            ctx.num_partitions() < TWO_FRAG_NUM_PARTITIONS,
            "precondition: a partition must actually have been joined away, stats: {}",
            ctx.stats_json()
        );
        assert_eq!(
            ctx.stats()["num_indexed_rows"].as_u64().unwrap() as usize,
            TWO_FRAG_NUM_ROWS + APPENDED_ROWS,
            "the reassigned rows must all survive the join; stats: {}",
            ctx.stats_json()
        );
        assert_refine_served_from_index(&dataset, query.as_ref(), index_type).await;
    }

    /// A retrain rebuilds every segment from scratch and re-declares the covering payload
    /// from committed metadata, where carried refine vectors appear as the indexed column
    /// itself. That declaration travels back through `VectorIndexParams`, whose covering
    /// setter is the user-facing one and rejects the indexed column outright, so it has to
    /// be split back out into the flag before validation ever sees it.
    #[rstest]
    #[case::pq("IVF_PQ")]
    #[case::sq("IVF_SQ")]
    #[case::rq("IVF_RQ")]
    #[case::hnsw_pq("IVF_HNSW_PQ")]
    #[case::hnsw_sq("IVF_HNSW_SQ")]
    #[tokio::test]
    async fn test_retrain_keeps_carried_vectors(#[case] index_type: &str) {
        const INDEX_NAME: &str = "vec_idx";
        const APPENDED_ROWS: usize = 200;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let uri = format!(
            "{}/retrain_keeps_carried_vectors_{index_type}",
            test_dir.as_str()
        );
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let mut params = params_for_index_type(&dataset, index_type).await;
        // `payload` is covered so the projection is served from the index too, which is what
        // lets the plan assertion attribute any base-table read to refine alone.
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        append_covered_rows(&mut dataset, APPENDED_ROWS).await;
        dataset
            .optimize_indices(&OptimizeOptions::retrain())
            .await
            .unwrap();

        let ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            ctx.stats()["num_indexed_rows"].as_u64().unwrap() as usize,
            TWO_FRAG_NUM_ROWS + APPENDED_ROWS,
            "a retrain must rebuild over the whole dataset; stats: {}",
            ctx.stats_json()
        );

        // The declaration has to survive too: dropping it would leave storage carrying a
        // payload no query can reach.
        let indices = dataset.load_indices().await.unwrap();
        let index = indices
            .iter()
            .find(|idx| idx.name == INDEX_NAME)
            .expect("index must still exist");
        let payload_field_id = dataset.schema().field("payload").unwrap().id;
        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        assert_eq!(
            index.covering_fields,
            vec![payload_field_id, vector_field_id],
            "the rebuilt segment must still declare both the covering column and the \
             carried vectors"
        );

        assert_refine_served_from_index(&dataset, query.as_ref(), index_type).await;
    }

    /// The carried copy must be the user's RAW vectors -- the whole design rests on the copy
    /// being taken *before* the IVF/quantizer transform chain, which rewrites the indexed
    /// column in place (residual under L2 and cosine, normalized first under cosine).
    ///
    /// Every other refine test asserts plan shape and row count, which a copy taken one step
    /// too late would satisfy perfectly while `refine` re-ranked against residuals. This one
    /// reads the values back and compares them to the source data. Because `id` is covered
    /// too, the whole projection is served from index storage -- the absent `LanceRead` is
    /// what proves the vectors came from the index rather than the base table, so the
    /// equality below is an assertion about what the *index* holds.
    ///
    /// Cosine is the case with no positive coverage at all until now, and the one the design
    /// note calls irrecoverable: normalization discards magnitude, so if the copy were taken
    /// after it, these vectors would come back unit-length. Recall is asserted only for L2 --
    /// this fixture's rows are near-constant across dimensions, so every row points in
    /// almost the same direction and cosine ranking on it would be meaningless.
    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    #[tokio::test]
    async fn test_carried_vectors_are_the_raw_source_values(#[case] distance_type: DistanceType) {
        const INDEX_NAME: &str = "vec_idx";
        const K: usize = 10;
        const CLUSTER_SIZE: u64 = 8;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let source = batches[0]["vector"].as_fixed_size_list().clone();
        let query = source.value(0);
        let uri = format!("{}/carried_raw_{distance_type}", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
        // `id` covered so the projection below needs no base-table take either; without it
        // the plan reads the base table for `id` and proves nothing about the vectors.
        params.covering_columns(vec!["id".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.nearest("vector", query.as_ref(), K).unwrap();
        scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
        scan.refine(2);
        scan.project(&["vector", "id"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            !plan.contains("LanceRead"),
            "precondition: the projection must be served from the index, or the values \
             below say nothing about what the index carries; plan:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch.num_rows(),
            K,
            "the query must return its k neighbours"
        );
        let ids = batch["id"].as_primitive::<UInt64Type>();
        let got = batch["vector"].as_fixed_size_list();
        for row in 0..batch.num_rows() {
            let id = ids.value(row) as usize;
            let expected = source.value(id);
            let expected = expected.as_primitive::<Float32Type>();
            let actual = got.value(row);
            let actual = actual.as_primitive::<Float32Type>();
            assert_eq!(
                actual.values(),
                expected.values(),
                "carried vector for id {id} is not the source vector -- the copy was taken \
                 after the transform chain (residual, or normalized under cosine)"
            );
        }

        if distance_type == DistanceType::L2 {
            // The query is row 0, whose true neighbours are its own cluster.
            let hits = (0..batch.num_rows())
                .filter(|row| ids.value(*row) < CLUSTER_SIZE)
                .count();
            assert!(
                hits * 2 >= CLUSTER_SIZE as usize,
                "recall below 0.5: only {hits} of the query's {CLUSTER_SIZE}-row cluster came \
                 back in the top {K}"
            );
        }
    }

    /// Renaming an indexed column stays legal wherever the index stores no copy of it
    /// *by name*, and the renamed index must still survive both maintenance paths.
    ///
    /// This is the counterpart to
    /// `test_alter_of_indexed_column_refused_with_carried_refine`: that one pins the refusal
    /// for a refine-carrying index, this one pins that the refusal did not spread to indexes
    /// it must not cover. A plain index stores only `_rowid`/codes/part-id, and a covering
    /// index adds the *covered* columns -- in neither case is the indexed column present
    /// under its own name, so a rename leaves nothing stale and the field id the index is
    /// keyed on survives it.
    #[rstest]
    #[case::plain(false)]
    #[case::covering_payload(true)]
    #[tokio::test]
    async fn test_rename_indexed_column_survives_maintenance(#[case] covered: bool) {
        use crate::dataset::ColumnAlteration;

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/rename_survives_{covered}", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        if covered {
            params.covering_columns(vec!["payload".to_string()]);
        }
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("ix".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        dataset
            .alter_columns(&[ColumnAlteration::new("vector".into()).rename("embedding".into())])
            .await
            .expect("renaming an indexed column stays legal when no copy is stored by name");

        // The index must still be usable, and -- the part the refine bug broke -- both
        // maintenance paths must still run against it.
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .expect("optimize must still work after the rename");
        dataset.delete("id < 16").await.unwrap();
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .expect("compaction must still work after the rename");

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "the index must survive the rename");
        assert_eq!(
            indices[0].keyed_field(),
            Some(dataset.schema().field("embedding").unwrap().id),
            "the index must stay keyed on the renamed column's (unchanged) field id"
        );
    }

    /// An inline remap (`compact_files` without `defer_index_remap`) rebuilds the index
    /// through `new_remapper`, which reads its covering set back out of *storage* -- where
    /// the carried vectors sit under the indexed column's own name, indistinguishable from
    /// an ordinary covering column. That set has to be split back into covering columns
    /// proper plus the carried-refine flag, or the rebuilt index disagrees with every other
    /// build path about what it is carrying.
    #[tokio::test]
    async fn test_remap_preserves_carried_vectors() {
        const INDEX_NAME: &str = "vec_idx";

        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let uri = format!("{}/remap_keeps_carried_vectors", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);
        params.store_vectors_for_refine(true);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Deleting rows gives compaction something to rewrite, which is what triggers the
        // remap of the index built above.
        dataset.delete("id < 16").await.unwrap();
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let index = indices
            .iter()
            .find(|idx| idx.name == INDEX_NAME)
            .expect("the index must survive compaction");
        let payload_id = dataset.schema().field("payload").unwrap().id;
        let vector_id = dataset.schema().field("vector").unwrap().id;
        assert_eq!(
            index.covering_fields,
            vec![payload_id, vector_id],
            "the remapped index must still declare the covering column and the carried \
             vectors, in that order"
        );

        assert_refine_served_from_index(&dataset, query.as_ref(), "remap").await;
    }

    /// PQ keeps only lossy codes, so `refine` re-ranks by taking full-precision vectors
    /// from the base table. Storing them in the index removes that take. They are carried
    /// as a storage-internal column, NOT via `covering_columns`: the transform chain
    /// rewrites the indexed column in place (residual, and normalization under cosine), so
    /// a user-visible copy under the column's own name would hand back values that are not
    /// the user's vectors -- and under cosine the magnitude is gone for good.
    #[tokio::test]
    async fn test_refine_vectors_are_stored_in_index_storage() {
        let test_dir = TempStrDir::default();
        let (schema, batches) = make_covered_test_batches();
        let uri = format!("{}/store_refine_vectors", test_dir.as_str());
        let mut dataset = write_dataset_from_batches(&uri, schema, batches).await;

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.store_vectors_for_refine(true);

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let index = dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // Physically present under the indexed column's own name -- the copy is taken
        // before the transform chain, so its values are that column's values -- and stamped
        // with that column's source id, which is what makes it servable rather than merely
        // present.
        let vector_field_id = dataset.schema().field("vector").unwrap().id;
        let physical = index.physical_covering_fields().unwrap();
        assert!(
            physical
                .iter()
                .any(|(id, field)| *id == vector_field_id && field.name() == "vector"),
            "index storage must carry the vector column stamped with its source field id; \
             got {physical:?}"
        );

        // And declared, so the search emits it and `refine` can find it by name.
        assert_eq!(
            indices[0].covering_fields,
            vec![vector_field_id],
            "carried vectors must be declared covering, or nothing will read them"
        );
    }

    /// The single-segment counterpart to the merge test above, and the case a name-and-type
    /// check cannot see. A driver rebinds a segment built for a dropped `payload` onto a
    /// re-added field with the same name and type but a fresh id.
    ///
    /// The declaration/payload contract permits the commit. At read time, however, the
    /// storage's stamped source id must prevent those old values from being served under the
    /// new field: planning falls back to a base-table take, whose re-added column is all null.
    #[tokio::test]
    async fn test_rebound_covering_field_id_falls_back_to_base_table() {
        use crate::dataset::NewColumnTransform;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_covered_test_batches();
        let query = batches[0]["vector"].as_fixed_size_list().value(0);
        let dataset_uri = format!("{}/commit_rebind_covering_field_id", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let all_fragments: Vec<u32> = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect();

        let (ivf_params, pq_params) = prepare_covered_ivf_pq(&dataset, "vector").await;
        let mut params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        params.covering_columns(vec!["payload".to_string()]);

        let mut segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vec_idx".to_string())
            .fragments(all_fragments)
            .execute_uncommitted()
            .await
            .unwrap();
        let original_field_id = dataset.schema().field("payload").unwrap().id;

        // Drop and re-add `payload`: same name, same type, new field id. The re-add is
        // metadata-only (AllNulls), so no data file changes and no fragment looks stale.
        dataset.drop_columns(&["payload"]).await.unwrap();
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(arrow_schema::Schema::new(vec![
                    Field::new("payload", DataType::UInt64, true),
                ]))),
                None,
                None,
            )
            .await
            .unwrap();
        let new_field_id = dataset.schema().field("payload").unwrap().id;
        assert_ne!(
            original_field_id, new_field_id,
            "re-adding the column must produce a new field id, or this test proves nothing"
        );

        // The segment keeps its true build version, so the staleness pass runs and passes;
        // only the id rebind is wrong.
        for id in segment.fields.iter_mut() {
            if *id == original_field_id {
                *id = new_field_id;
            }
        }
        for id in segment.covering_fields.iter_mut() {
            if *id == original_field_id {
                *id = new_field_id;
            }
        }

        dataset
            .commit_existing_index_segments("vec_idx", "vector", vec![segment])
            .await
            .unwrap();

        let mut scan = dataset.scan();
        scan.nearest("vector", query.as_ref(), 10).unwrap();
        scan.nprobes(TWO_FRAG_NUM_PARTITIONS);
        scan.project(&["payload"]).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("LanceRead"),
            "storage was built from field id {original_field_id}, not the declaration's \
             rebound id {new_field_id}, so payload must use a base-table take; plan:\n{plan}"
        );

        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 10);
        let payload = batch["payload"].as_primitive::<UInt64Type>();
        assert_eq!(
            payload.null_count(),
            payload.len(),
            "the re-added all-null payload must come from the base table, not stale index values"
        );
    }

    #[rstest]
    #[case::flat("IVF_HNSW_FLAT")]
    #[case::pq("IVF_HNSW_PQ")]
    #[case::sq("IVF_HNSW_SQ")]
    #[tokio::test]
    async fn test_merge_existing_hnsw_segments_rebuilds_graph(#[case] expected_index_type: &str) {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches, max_rows_per_file) = if expected_index_type == "IVF_HNSW_PQ" {
            let (batch, schema) = make_seeded_vector_batch(LIGHTWEIGHT_PQ_ROWS * 2);
            (schema, vec![batch], LIGHTWEIGHT_PQ_ROWS)
        } else {
            let (schema, batches) = make_two_fragment_batches();
            (schema, batches, 500)
        };
        let dataset_uri = format!("{}/merge_hnsw_rebuilds_graph", base_uri);
        let mut dataset = write_dataset_from_batches_with_max_rows(
            &dataset_uri,
            schema,
            batches,
            max_rows_per_file,
        )
        .await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = match expected_index_type {
            "IVF_HNSW_FLAT" => VectorIndexParams::ivf_hnsw(
                DistanceType::L2,
                prepare_global_ivf(&dataset, "vector").await,
                HnswBuildParams::default(),
            ),
            "IVF_HNSW_PQ" => {
                let (ivf_params, pq_params) = prepare_ivf_pq(
                    &dataset,
                    "vector",
                    DIM,
                    LIGHTWEIGHT_PQ_PARTITIONS,
                    LIGHTWEIGHT_PQ_SUB_VECTORS,
                    8,
                    2,
                    16,
                )
                .await;
                VectorIndexParams::with_ivf_hnsw_pq_params(
                    DistanceType::L2,
                    ivf_params,
                    lightweight_hnsw_params(),
                    pq_params,
                )
            }
            "IVF_HNSW_SQ" => VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::L2,
                prepare_global_ivf(&dataset, "vector").await,
                HnswBuildParams::default(),
                SQBuildParams::default(),
            ),
            other => panic!("unexpected HNSW index type {other}"),
        };
        let mut segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        let merged = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments("vector_idx", "vector", vec![merged])
            .await
            .unwrap();

        let stats = dataset.index_statistics("vector_idx").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(stats["index_type"].as_str().unwrap(), expected_index_type);
        assert_eq!(
            stats["indices"][0]["sub_index"]["index_type"]
                .as_str()
                .unwrap(),
            "HNSW"
        );

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_merge_existing_hnsw_segments_rejects_mismatched_build_params() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_two_fragment_batches();
        let dataset_uri = format!("{}/merge_hnsw_rejects_mismatched_params", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);

        let ivf_params = prepare_global_ivf(&dataset, "vector").await;
        let default_params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            ivf_params.clone(),
            HnswBuildParams::default(),
        );
        let custom_params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            ivf_params,
            HnswBuildParams::default().num_edges(16),
        );

        let first_segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &default_params)
            .name("vector_idx".to_string())
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        let second_segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &custom_params)
            .name("vector_idx".to_string())
            .fragments(vec![fragments[1].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();

        let error = dataset
            .merge_existing_index_segments(vec![first_segment, second_segment])
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("HNSW build parameters mismatch while merging index segments"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_merge_index_metadata_reports_progress() {
        const INDEX_NAME: &str = "vector_idx";

        let test_dir = TempStrDir::default();
        let dataset_uri = format!("{}/progress", test_dir.as_str());
        let (schema, batches) = make_two_fragment_batches();
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 2,
            "expected at least 2 fragments, got {}",
            fragments.len()
        );
        let expected_rows = fragments[0].physical_rows().await.unwrap() as u64
            + fragments[1].physical_rows().await.unwrap() as u64;
        let (ivf_params, pq_params) = prepare_global_ivf_pq(&dataset, "vector").await;
        let params = VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        let mut segments = Vec::new();
        for fragment in fragments.iter().take(2) {
            segments.push(
                dataset
                    .create_index_builder(&["vector"], IndexType::Vector, &params)
                    .name(INDEX_NAME.to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }

        let progress = Arc::new(RecordingProgress::default());
        let merged_segment = crate::index::vector::ivf::merge_segments_with_progress(
            dataset.object_store.as_ref(),
            &dataset.indices_dir(),
            segments,
            progress.clone(),
        )
        .await
        .unwrap();
        dataset
            .commit_existing_index_segments(INDEX_NAME, "vector", vec![merged_segment])
            .await
            .unwrap();

        let events = progress.recorded_events();
        let tags = events
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let merge_total = events
            .iter()
            .find_map(|(kind, stage, value)| {
                if kind == "start" && stage == "merge_partitions" {
                    Some(*value)
                } else {
                    None
                }
            })
            .expect("missing merge_partitions start total");
        let merged_rows = events
            .iter()
            .filter_map(|(kind, stage, value)| {
                if kind == "progress" && stage == "merge_partitions" {
                    Some(*value)
                } else {
                    None
                }
            })
            .next_back()
            .unwrap_or_default();
        let read_start = tags
            .iter()
            .position(|e| e == "start:read_shard_metadata")
            .expect("missing read_shard_metadata start");
        let read_complete = tags
            .iter()
            .position(|e| e == "complete:read_shard_metadata")
            .expect("missing read_shard_metadata complete");
        let merge_start = tags
            .iter()
            .position(|e| e == "start:merge_partitions")
            .expect("missing merge_partitions start");
        let merge_complete = tags
            .iter()
            .position(|e| e == "complete:merge_partitions")
            .expect("missing merge_partitions complete");
        let aux_start = tags
            .iter()
            .position(|e| e == "start:write_auxiliary_index")
            .expect("missing write_auxiliary_index start");
        let aux_complete = tags
            .iter()
            .position(|e| e == "complete:write_auxiliary_index")
            .expect("missing write_auxiliary_index complete");
        let root_start = tags
            .iter()
            .position(|e| e == "start:write_root_index")
            .expect("missing write_root_index start");
        let root_complete = tags
            .iter()
            .position(|e| e == "complete:write_root_index")
            .expect("missing write_root_index complete");

        assert!(read_start < read_complete);
        assert!(read_complete < merge_start);
        assert!(merge_start < merge_complete);
        assert!(merge_complete < aux_start);
        assert!(aux_start < aux_complete);
        assert!(aux_complete < root_start);
        assert!(root_start < root_complete);
        assert_eq!(
            merge_total, expected_rows,
            "expected merge_partitions total rows to match dataset rows"
        );
        assert_eq!(
            merged_rows, expected_rows,
            "expected merge_partitions completed rows to match dataset rows"
        );
        assert!(
            tags.iter().any(|e| e == "progress:write_root_index"),
            "expected write_root_index progress callbacks"
        );
    }

    #[tokio::test]
    async fn test_distributed_ivf_sq_worker_training_respects_fragment_filter() {
        const ROWS_PER_FRAGMENT: usize = 64;
        const FRAGMENT_OFFSETS: [f32; 2] = [0.0, 1000.0];

        let test_dir = TempStrDir::default();
        let dataset_uri = format!("{}/distributed_sq_fragment_filter", test_dir.as_str());
        let (schema, batches) = make_fragment_offset_batches(ROWS_PER_FRAGMENT, &FRAGMENT_OFFSETS);
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), FRAGMENT_OFFSETS.len());

        let ivf_params =
            IvfBuildParams::try_with_centroids(2, build_centroids_for_offsets(&FRAGMENT_OFFSETS))
                .unwrap();
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            ivf_params,
            SQBuildParams::default(),
        );

        let segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("sq_fragment_filter".to_string())
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();

        let scheduler = ScanScheduler::new(
            Arc::new(dataset.object_store.as_ref().clone()),
            SchedulerConfig::default_for_testing(),
        );
        let sq_meta = get_sq_metadata(&dataset, scheduler, &segment.uuid.to_string()).await;

        assert_eq!(sq_meta.bounds.start, 0.0);
        assert_eq!(sq_meta.bounds.end, (DIM - 1) as f64);
        assert_lt!(sq_meta.bounds.end, FRAGMENT_OFFSETS[1] as f64);
    }

    async fn test_index(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
    ) {
        match params.metric_type {
            DistanceType::Hamming => {
                test_index_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4, dataset)
                    .await;
            }
            _ => {
                test_index_impl::<Float32Type>(
                    params.clone(),
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                    dataset.clone(),
                )
                .await;

                if dataset.is_none() {
                    test_index_impl::<Float64Type>(
                        params,
                        nlist,
                        recall_requirement,
                        0.0..1.0,
                        dataset,
                    )
                    .await;
                }
            }
        }
    }

    fn pq_matrix_batch<T>() -> RecordBatch
    where
        T: ArrowPrimitiveType + 'static,
        T::Native: Copy + 'static,
        PrimitiveArray<T>: From<Vec<T::Native>> + 'static,
        StandardUniform: Distribution<T::Native>,
    {
        gen_batch()
            .with_seed(Seed(42))
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::rand_vec::<T>(Dimension::from(DIM as u32)))
            .into_batch_rows(RowCount::from(PQ_MATRIX_NUM_ROWS as u64))
            .unwrap()
    }

    fn pq_matrix_params(
        nlist: usize,
        distance_type: DistanceType,
        version: IndexFileVersion,
    ) -> VectorIndexParams {
        let mut ivf_params = IvfBuildParams::new(nlist);
        ivf_params.max_iters = 2;
        ivf_params.sample_rate = PQ_MATRIX_NUM_ROWS;
        let pq_params = PQBuildParams {
            num_sub_vectors: 4,
            num_bits: 8,
            max_iters: 2,
            sample_rate: 1,
            ..Default::default()
        };
        let mut params =
            VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
        params.version(version);
        params
    }

    async fn test_pq_matrix_case(
        nlist: usize,
        distance_type: DistanceType,
        version: IndexFileVersion,
    ) {
        const INDEX_NAME: &str = "pq_matrix";

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let batch = pq_matrix_batch::<Float32Type>();
        let schema = batch.schema();
        let query = batch["vector"].as_fixed_size_list().value(0);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        let params = pq_matrix_params(nlist, distance_type, version.clone());
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats["index_type"], "IVF_PQ");
        let indices = stats["indices"].as_array().unwrap();
        assert_eq!(indices.len(), 1);
        let index = &indices[0];
        assert_eq!(index["index_type"], "IVF_PQ");
        assert_eq!(index["metric_type"], distance_type.to_string());
        assert_eq!(index["num_partitions"], nlist);
        assert_eq!(index["sub_index"]["index_type"], "PQ");
        assert_eq!(
            index["index_file_version"],
            match version {
                IndexFileVersion::Legacy => "Legacy",
                IndexFileVersion::V3 => "V3",
            }
        );

        drop(dataset);
        let dataset = Dataset::open(test_uri).await.unwrap();
        let ground_truth = ground_truth(
            &dataset,
            "vector",
            query.as_ref(),
            PQ_MATRIX_K,
            distance_type,
        )
        .await;
        let result = dataset
            .scan()
            .nearest("vector", query.as_primitive::<Float32Type>(), PQ_MATRIX_K)
            .unwrap()
            .nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(result.num_rows(), PQ_MATRIX_K);
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(
            row_ids.iter().copied().collect::<HashSet<_>>().len(),
            PQ_MATRIX_K
        );
        let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
        assert!(distances.iter().all(|distance| distance.is_finite()));
        assert!(
            distances.windows(2).all(|pair| pair[0] <= pair[1]),
            "distances are not sorted: {distances:?}"
        );
        let recall = row_ids
            .iter()
            .filter(|row_id| ground_truth.contains(row_id))
            .count() as f32
            / PQ_MATRIX_K as f32;
        assert_ge!(recall, 0.5, "recall: {recall}, row_ids: {row_ids:?}");
    }

    async fn test_index_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
        dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = match dataset {
            Some((dataset, vectors)) => (dataset, vectors),
            None => generate_test_dataset::<T>(test_uri, range).await,
        };

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        test_recall::<T>(
            params.clone(),
            nlist,
            recall_requirement,
            vector_column,
            &dataset,
            vectors.clone(),
        )
        .await;

        if params.stages.len() > 1
            && matches!(params.version, IndexFileVersion::V3)
            && params.index_type() == IndexType::IvfPq
        {
            let indices = dataset.load_indices().await.unwrap();
            assert_eq!(indices.len(), 1);
            let old_meta = indices[0].clone();
            rewrite_pq_storage(&mut dataset, &old_meta).await.unwrap();
            // do the test again
            test_recall::<T>(
                params,
                nlist,
                recall_requirement,
                vector_column,
                &dataset,
                vectors.clone(),
            )
            .await;
        }
    }

    async fn test_remap(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
        match params.metric_type {
            DistanceType::Hamming => {
                Box::pin(test_remap_impl::<UInt8Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0..4,
                ))
                .await;
            }
            _ => {
                let index_type = params.index_type();
                Box::pin(test_remap_impl::<Float32Type>(
                    params.clone(),
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                ))
                .await;
                if matches!(index_type, IndexType::IvfFlat | IndexType::IvfHnswFlat) {
                    Box::pin(test_remap_impl::<Float64Type>(
                        params,
                        nlist,
                        recall_requirement,
                        0.0..1.0,
                    ))
                    .await;
                }
            }
        }
    }

    async fn test_remap_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        // let recall_requirement = recall_requirement * 0.99;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query = vectors.value(0);
        // delete half rows to trigger compact
        let half_rows = NUM_ROWS / 2;
        dataset
            .delete(&format!("id < {}", half_rows))
            .await
            .unwrap();
        // update the other half rows
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where(&format!("id >= {} and id<{}", half_rows, half_rows + 50))
            .unwrap()
            .set("id", &format!("{}+id", NUM_ROWS))
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = Dataset::open(update_result.new_dataset.uri())
            .await
            .unwrap();
        let num_rows = dataset.count_rows(None).await.unwrap();
        assert_eq!(num_rows, half_rows);
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        // query again, the result should not include the deleted row
        let result = dataset.scan().try_into_batch().await.unwrap();
        let ids = result["id"].as_primitive::<UInt64Type>();
        assert_eq!(ids.len(), half_rows);
        ids.values().iter().for_each(|id| {
            assert!(*id >= half_rows as u64 + 50);
        });

        // make sure we can still hit the recall
        let gt = ground_truth(&dataset, vector_column, &query, 100, params.metric_type).await;
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .minimum_nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = results[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = row_ids.intersection(&gt).count() as f32 / 100.0;
        // 100 can't be exactly expressed as a float, so we need to use a tolerance
        assert_ge!(
            recall,
            recall_requirement - f32::EPSILON,
            "num_rows: {}, intersection: {}, recall: {}",
            row_ids.len(),
            row_ids.intersection(&gt).count(),
            recall
        );

        // delete so that only one row left, to trigger remap and there must be some empty partitions
        let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
        dataset.delete("id > 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .minimum_nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
    }

    async fn test_delete_all_rows(params: VectorIndexParams) {
        match params.metric_type {
            DistanceType::Hamming => {
                test_delete_all_rows_impl::<UInt8Type>(params, 0..4).await;
            }
            _ => {
                test_delete_all_rows_impl::<Float32Type>(params, 0.0..1.0).await;
            }
        }
    }

    async fn test_delete_all_rows_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("id >= 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

        // optimize after delete all rows
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let query = vectors.value(0);
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 0);

        // compact after delete all rows
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("id >= 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 0);
    }

    #[tokio::test]
    async fn test_flat_knn() {
        test_distance_range(None, 4).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 1.0)]
    #[case(4, DistanceType::Cosine, 1.0)]
    #[case(4, DistanceType::Dot, 1.0)]
    #[case(4, DistanceType::Hamming, 0.9)]
    #[tokio::test]
    async fn test_build_ivf_flat(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let params = VectorIndexParams::ivf_flat(nlist, distance_type);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_remap(params.clone(), nlist, recall_requirement).await;
        test_delete_all_rows(params).await;
    }

    #[rstest]
    #[case::l2(4, DistanceType::L2)]
    #[case::cosine(4, DistanceType::Cosine)]
    #[case::dot(4, DistanceType::Dot)]
    #[tokio::test]
    async fn test_build_ivf_pq(#[case] nlist: usize, #[case] distance_type: DistanceType) {
        test_pq_matrix_case(nlist, distance_type, IndexFileVersion::Legacy).await;
    }

    #[rstest]
    #[case::l2_nlist1(1, DistanceType::L2)]
    #[case::cosine_nlist1(1, DistanceType::Cosine)]
    #[case::dot_nlist1(1, DistanceType::Dot)]
    #[case::l2_nlist4(4, DistanceType::L2)]
    #[case::cosine_nlist4(4, DistanceType::Cosine)]
    #[case::dot_nlist4(4, DistanceType::Dot)]
    #[tokio::test]
    async fn test_build_ivf_pq_v3(#[case] nlist: usize, #[case] distance_type: DistanceType) {
        test_pq_matrix_case(nlist, distance_type, IndexFileVersion::V3).await;
    }

    #[rstest]
    #[case::legacy(IndexFileVersion::Legacy)]
    #[case::v3(IndexFileVersion::V3)]
    #[tokio::test]
    async fn test_ivf_pq_distance_range(#[case] version: IndexFileVersion) {
        let params = pq_matrix_params(1, DistanceType::L2, version);
        test_distance_range(Some(params), 1).await;
    }

    #[rstest]
    #[case::legacy(IndexFileVersion::Legacy)]
    #[case::v3(IndexFileVersion::V3)]
    #[tokio::test]
    async fn test_ivf_pq_f64_smoke(#[case] version: IndexFileVersion) {
        let test_dir = TempStrDir::default();
        let batch = pq_matrix_batch::<Float64Type>();
        let schema = batch.schema();
        let vectors = Arc::new(batch["vector"].as_fixed_size_list().clone());
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
            .await
            .unwrap();
        let params = pq_matrix_params(1, DistanceType::L2, version);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        test_recall::<Float64Type>(params, 1, 0.5, "vector", &dataset, vectors).await;
    }

    #[tokio::test]
    async fn test_legacy_ivf_pq_cosine_multivec_smoke() {
        let params = pq_matrix_params(1, DistanceType::Cosine, IndexFileVersion::Legacy);
        test_index_multivec_impl::<Float32Type>(params, 1, 0.5, 0.0..1.0).await;
    }

    #[tokio::test]
    async fn test_ivf_pq_delete_all_rows_lifecycle() {
        let params = pq_matrix_params(1, DistanceType::L2, IndexFileVersion::V3);
        test_delete_all_rows(params).await;
    }

    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    #[case::dot(DistanceType::Dot)]
    #[tokio::test]
    async fn test_build_ivf_pq_4bit(#[case] distance_type: DistanceType) {
        assert_lightweight_pq_index(distance_type, 4, false).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.85)]
    #[case(4, DistanceType::Cosine, 0.85)]
    #[case(4, DistanceType::Dot, 0.75)]
    #[tokio::test]
    async fn test_build_ivf_sq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_sq_params(distance_type, ivf_params, sq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist, recall_requirement).await;
    }

    #[tokio::test]
    async fn test_build_ivf_sq_dot_with_negative_values() {
        let nlist = 4;
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let params =
            VectorIndexParams::with_ivf_sq_params(DistanceType::Dot, ivf_params, sq_params);

        test_index_impl::<Float32Type>(params, nlist, 0.75, -1.0..1.0, None).await;
    }

    // These queries probe every partition, so recall here measures RaBitQ quantization
    // error alone. At 1 bit per dimension it averages ~0.67 on this uniformly random,
    // L2-normalized data, and each build draws a fresh random rotation, so no bar worth
    // asserting sits clear of the spread. 5 bits lifts recall to ~0.97; its `ex_bits = 4`
    // also covers a FastScan ex-code kernel that the multi-bit test below never reaches.
    #[rstest]
    #[case(1, DistanceType::L2, 0.9)]
    #[case(1, DistanceType::Cosine, 0.9)]
    #[case(1, DistanceType::Dot, 0.9)]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.9)]
    #[tokio::test]
    async fn test_build_ivf_rq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
        #[values(RQRotationType::Fast, RQRotationType::Matrix)] rotation_type: RQRotationType,
    ) {
        let _ = env_logger::try_init();
        let ivf_params = IvfBuildParams::new(nlist);
        let rq_params = RQBuildParams::with_rotation_type(5, rotation_type);
        let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params.clone(), nlist, recall_requirement).await;
    }

    #[rstest]
    #[case::l2(DistanceType::L2, 9)]
    #[case::cosine(DistanceType::Cosine, 9)]
    // ex_bits=3 and ex_bits=5 have no FastScan support and use the bit-plane
    // repack, so these searches go through the exact ex-dot rerank kernels
    // end to end.
    #[case::l2_plane_repack_3bit(DistanceType::L2, 4)]
    #[case::l2_plane_repack_5bit(DistanceType::L2, 6)]
    #[tokio::test]
    async fn test_build_ivf_rq_multi_bit_persists_split_codes_and_searches(
        #[case] distance_type: DistanceType,
        #[case] num_bits: u8,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(4);
        let rq_params = RQBuildParams::with_rotation_type(num_bits, RQRotationType::Fast);
        let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        let obj_store = Arc::new(ObjectStore::local());
        let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
        let index_uuid = indices[0].uuid.to_string();
        let rq_meta = get_rq_metadata(&dataset, scheduler.clone(), &index_uuid).await;
        assert_eq!(rq_meta.num_bits, num_bits);
        assert_eq!(rq_meta.query_estimator, RabitQueryEstimator::RawQuery);

        let reader = open_rq_aux_reader(&dataset, scheduler, &index_uuid).await;
        let schema = reader.schema();
        let ex_field = schema.field(RABIT_BLOCKED_EX_CODE_COLUMN).unwrap();
        let DataType::FixedSizeList(_, ex_code_bytes) = ex_field.data_type() else {
            panic!("RQ ex-code field should be FixedSizeList");
        };
        let expected_ex_code_bytes =
            blocked_ex_code_bytes(rq_meta.rotated_dim(), num_bits - 1) as i32;
        assert_eq!(ex_code_bytes, expected_ex_code_bytes);
        assert!(schema.field(EX_ADD_FACTORS_COLUMN).is_some());
        assert!(schema.field(EX_SCALE_FACTORS_COLUMN).is_some());

        test_recall::<Float32Type>(params, 4, 0.5, "vector", &dataset, vectors).await;
    }

    #[rstest]
    #[case::fast(RQRotationType::Fast)]
    #[case::matrix(RQRotationType::Matrix)]
    #[tokio::test]
    async fn test_ivf_rq_rotation_type_after_optimize(#[case] rotation_type: RQRotationType) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(4);
        let rq_params = RQBuildParams::with_rotation_type(1, rotation_type);
        let params = VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf_params, rq_params);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        assert_rq_rotation_type(&dataset, rotation_type).await;

        append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            2,
            "Expected append optimize to create one delta index"
        );
        assert_rq_rotation_type(&dataset, rotation_type).await;

        dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();
        let indices_after_merge = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_merge.len(),
            1,
            "Expected merge optimize to merge indices into one"
        );
        assert_rq_rotation_type(&dataset, rotation_type).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[case(4, DistanceType::Hamming, 0.9)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_flat(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::ivf_hnsw(distance_type, ivf_params, hnsw_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist, recall_requirement).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_sq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            sq_params,
        );
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_delete_all_rows(params.clone()).await;
        test_remap(params, nlist, recall_requirement).await;
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_sq_dot_with_negative_values() {
        let nlist = 4;
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::Dot,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        test_index_impl::<Float32Type>(params, nlist, 0.75, -1.0..1.0, None).await;
    }

    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    #[case::dot(DistanceType::Dot)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_pq(#[case] distance_type: DistanceType) {
        assert_lightweight_pq_index(distance_type, 8, true).await;
    }

    #[rstest]
    #[case::l2(DistanceType::L2)]
    #[case::cosine(DistanceType::Cosine)]
    #[case::dot(DistanceType::Dot)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_pq_4bit(#[case] distance_type: DistanceType) {
        assert_lightweight_pq_index(distance_type, 4, true).await;
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_pq_multivec() {
        const NUM_ROWS: usize = 64;
        const K: usize = 10;

        let test_dir = TempStrDir::default();
        let batch = lance_datagen::gen_batch()
            .with_seed(lance_datagen::Seed::from(42))
            .col("id", lance_datagen::array::step::<UInt64Type>())
            .col(
                "vector",
                lance_datagen::array::cycle_vec_var(
                    lance_datagen::array::rand_vec::<Float32Type>((DIM as u32).into()),
                    3_u32.into(),
                    4_u32.into(),
                ),
            )
            .into_batch_rows(lance_datagen::RowCount::from(NUM_ROWS as u64))
            .unwrap();
        let vectors = batch["vector"].as_list::<i32>().clone();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
            .await
            .unwrap();

        let mut ivf_params = IvfBuildParams::new(1);
        ivf_params.max_iters = 2;
        ivf_params.sample_rate = 16;
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            DistanceType::Cosine,
            ivf_params,
            lightweight_hnsw_params(),
            lightweight_pq_params(),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query = vectors.value(0);
        // Three vectors per query amplify the internal candidate k. This
        // bounded budget covers all 64 * 3 vector entries in the fixture.
        let result = search_lightweight_pq_index(&dataset, query.as_ref(), K, 1, 2, 256).await;
        assert_eq!(result.num_rows(), K);
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), K);
        let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
        assert!(distances.iter().all(|distance| distance.is_finite()));
        assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));

        let ground_truth = multivec_ground_truth(&vectors, query.as_ref(), K, DistanceType::Cosine)
            .into_iter()
            .map(|(_, row_id)| row_id)
            .collect::<HashSet<_>>();
        let recall = row_ids
            .iter()
            .filter(|row_id| ground_truth.contains(row_id))
            .count() as f32
            / K as f32;
        assert_ge!(recall, 0.5, "recall: {recall}");
    }

    // `lance-index` keeps these crate-private; spelling them out here also pins
    // the on-disk names, which are part of the index file contract.
    const HNSW_VECTOR_ID_COL: &str = "__vector_id";
    const HNSW_NEIGHBORS_COL: &str = "__neighbors";

    async fn build_ivf_hnsw_sq(test_uri: &str, nlist: usize) -> Dataset {
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            IvfBuildParams::new(nlist),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        dataset
    }

    async fn open_ivf_hnsw_sq(dataset: &Dataset) -> Arc<dyn VectorIndex> {
        let indices = dataset.load_indices().await.unwrap();
        dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap()
    }

    async fn assert_hnsw_columns(dataset: &Dataset, context: &str) {
        let index = open_ivf_hnsw_sq(dataset).await;
        let hnsw = index
            .as_any()
            .downcast_ref::<IvfHnswSqIndex>()
            .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");

        let written = hnsw
            .reader
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            written,
            vec![HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL, DIST_COL],
            "{context}: the written index file must keep every column"
        );

        // Every partition, not just the first: a projection that applied
        // unevenly would leave the partition cache holding mixed schemas.
        for partition_id in 0..hnsw.ivf.num_partitions() {
            let entry = hnsw
                .load_partition_entry(partition_id, PartitionColumns::Internal, None)
                .await
                .unwrap();
            let loaded = entry.index.to_batch().unwrap();
            let loaded_schema = loaded.schema();
            let read = loaded_schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                read,
                vec![HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL],
                "{context}: partition {partition_id} materialized the write-only distance column"
            );
        }
    }

    /// The index file keeps all three HNSW columns while a loaded partition
    /// carries only the two the graph reads. Both halves matter: shrinking the
    /// written schema would panic readers older than v8.0.0, and widening the
    /// read back would undo the saving.
    ///
    /// Re-checked after a delta merge, because that is the one path that writes
    /// a new index file while an already-projected index is open.
    #[tokio::test]
    async fn test_hnsw_partition_load_reads_only_graph_columns() {
        let test_dir = TempStrDir::default();
        let mut dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
        assert_hnsw_columns(&dataset, "fresh index").await;

        append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();
        assert_hnsw_columns(&dataset, "after delta merge").await;
    }

    /// The saving is the point of the change, so pin it: reading a real
    /// partition range through the declared projection must move strictly fewer
    /// bytes than the full-schema read the index used to perform.
    #[tokio::test]
    async fn test_hnsw_read_projection_moves_fewer_bytes() {
        use futures::TryStreamExt as _;

        let test_dir = TempStrDir::default();
        let dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
        let index = open_ivf_hnsw_sq(&dataset).await;
        let hnsw = index
            .as_any()
            .downcast_ref::<IvfHnswSqIndex>()
            .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");

        let projection = hnsw
            .read_projection
            .as_ref()
            .expect("HNSW declares a read projection");
        assert_eq!(projection.schema.fields.len(), 2);

        let row_range = hnsw.ivf.row_range(0);
        assert!(!row_range.is_empty(), "partition 0 should hold rows");
        let store = dataset.object_store.as_ref();

        let read_bytes_for = async |projection: lance_file::reader::ReaderProjection| {
            let _ = store.io_stats_incremental();
            hnsw.reader
                .read_stream_projected(
                    lance_io::ReadBatchParams::Range(row_range.clone()),
                    u32::MAX,
                    1,
                    projection,
                    lance_encoding::decoder::FilterExpression::no_filter(),
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            store.io_stats_incremental().read_bytes
        };

        // Projected first on purpose: it then pays any first-touch metadata
        // cost, so the comparison understates rather than flatters the saving.
        let projected_bytes = read_bytes_for(projection.clone()).await;
        let full_bytes = read_bytes_for(lance_file::versions::reader_projection_from_whole_schema(
            hnsw.reader.schema(),
            hnsw.reader.metadata().version(),
        ))
        .await;

        assert!(
            projected_bytes > 0,
            "the projected read still has to fetch the graph"
        );
        assert_lt!(projected_bytes, full_bytes);
    }

    /// The projection selects columns by field id, and `__neighbors` and
    /// `_distance` are both 4-byte-item lists whose child fields share a name,
    /// so a wrong column index would reinterpret distances as neighbor ids with
    /// no type error to catch it. Compare the columns themselves, not just names.
    #[tokio::test]
    async fn test_hnsw_projected_read_matches_full_read() {
        use futures::TryStreamExt as _;

        let test_dir = TempStrDir::default();
        let dataset = build_ivf_hnsw_sq(test_dir.as_str(), 4).await;
        let index = open_ivf_hnsw_sq(&dataset).await;
        let hnsw = index
            .as_any()
            .downcast_ref::<IvfHnswSqIndex>()
            .expect("IVF_HNSW_SQ should open as IvfHnswSqIndex");
        let projection = hnsw
            .read_projection
            .as_ref()
            .expect("HNSW declares a read projection");

        let read_range = async |proj: lance_file::reader::ReaderProjection,
                                range: std::ops::Range<usize>| {
            let batches = hnsw
                .reader
                .read_stream_projected(
                    lance_io::ReadBatchParams::Range(range),
                    u32::MAX,
                    1,
                    proj,
                    lance_encoding::decoder::FilterExpression::no_filter(),
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
        };

        let mut compared = 0;
        for partition_id in 0..hnsw.ivf.num_partitions() {
            let range = hnsw.ivf.row_range(partition_id);
            if range.is_empty() {
                continue;
            }
            let full = read_range(
                lance_file::versions::reader_projection_from_whole_schema(
                    hnsw.reader.schema(),
                    hnsw.reader.metadata().version(),
                ),
                range.clone(),
            )
            .await;
            let projected = read_range(projection.clone(), range).await;

            assert_eq!(projected.num_columns(), 2);
            assert_eq!(projected.num_rows(), full.num_rows());
            for name in [HNSW_VECTOR_ID_COL, HNSW_NEIGHBORS_COL] {
                assert_eq!(
                    projected.column_by_name(name).unwrap(),
                    full.column_by_name(name).unwrap(),
                    "partition {partition_id}: {name} differs between the projected and full read"
                );
            }
            compared += 1;
        }
        assert!(compared > 0, "no non-empty partition was compared");
    }

    async fn test_index_multivec(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
        // we introduce XTR for performance, which would reduce the recall a little bit
        let recall_requirement = recall_requirement * 0.9;
        match params.metric_type {
            DistanceType::Hamming => {
                test_index_multivec_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4)
                    .await;
            }
            _ => {
                test_index_multivec_impl::<Float32Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                )
                .await;
            }
        }
    }

    async fn test_index_multivec_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, vectors) = generate_multivec_test_dataset::<T>(test_uri, range).await;

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_index".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let query = vectors.value(0);
        let k = 100;

        let result = dataset
            .scan()
            .nearest("vector", &query, k)
            .unwrap()
            .minimum_nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        assert_eq!(row_ids.len(), k);
        assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), k);
        let dists = result[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let results = dists.into_iter().zip(row_ids.clone()).collect::<Vec<_>>();
        let row_ids = row_ids.into_iter().collect::<HashSet<_>>();

        let gt = multivec_ground_truth(&vectors, &query, k, params.metric_type);
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = row_ids.intersection(&gt_set).count() as f32 / 100.0;
        assert!(
            recall >= recall_requirement,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_migrate_v1_to_v3() {
        // only test the case of IVF_PQ
        // because only IVF_PQ is supported in v1
        let nlist = 4;
        let recall_requirement = 0.9;
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let v1_params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::Cosine, ivf_params, pq_params)
                .version(crate::index::vector::IndexFileVersion::Legacy)
                .clone();

        let v3_params = v1_params
            .clone()
            .version(crate::index::vector::IndexFileVersion::V3)
            .clone();

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        test_index(
            v1_params,
            nlist,
            recall_requirement,
            Some((dataset.clone(), vectors.clone())),
        )
        .await;
        dataset.checkout_latest().await.unwrap();
        // retest with v3 params on the same dataset
        test_index(
            v3_params,
            nlist,
            recall_requirement,
            Some((dataset.clone(), vectors)),
        )
        .await;

        dataset.checkout_latest().await.unwrap();
        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 1); // v1 index should be replaced by v3 index
        let index = dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let v3_index = index.as_any().downcast_ref::<super::IvfPq>();
        assert!(v3_index.is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn test_index_stats(
        #[values(
            (VectorIndexParams::ivf_flat(4, DistanceType::Hamming), IndexType::IvfFlat),
            (VectorIndexParams::ivf_pq(4, 8, 8, DistanceType::L2, 10), IndexType::IvfPq),
            (VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::Cosine,
                IvfBuildParams::new(4),
                Default::default(),
                Default::default()
            ), IndexType::IvfHnswSq),
        )]
        index: (VectorIndexParams, IndexType),
    ) {
        let (params, index_type) = index;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let nlist = 4;
        let (mut dataset, _) = match params.metric_type {
            DistanceType::Hamming => generate_test_dataset::<UInt8Type>(test_uri, 0..2).await,
            _ => generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await,
        };
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_index".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats = dataset.index_statistics("test_index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

        assert_eq!(
            stats["index_type"].as_str().unwrap(),
            index_type.to_string()
        );
        for index in stats["indices"].as_array().unwrap() {
            assert_eq!(
                index["index_type"].as_str().unwrap(),
                index_type.to_string()
            );
            assert_eq!(
                index["num_partitions"].as_number().unwrap(),
                &serde_json::Number::from(nlist)
            );

            let sub_index = match index_type {
                IndexType::IvfHnswPq | IndexType::IvfHnswSq => "HNSW",
                IndexType::IvfPq => "PQ",
                _ => "FLAT",
            };
            assert_eq!(
                index["sub_index"]["index_type"].as_str().unwrap(),
                sub_index
            );
        }
    }

    #[tokio::test]
    async fn test_index_stats_empty_partition() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let num_rows = 32;
        let num_partitions = num_rows + 2;
        let mut vector_values = vec![0.0; num_rows * DIM];
        for row in 0..num_rows {
            vector_values[row * DIM + row] = 1.0;
        }
        let one_hot_vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(vector_values.clone()),
                DIM as i32,
            )
            .unwrap(),
        );
        let batch = gen_batch()
            .col("id", array::step::<UInt64Type>())
            .col("vector", array::jitter_centroids(one_hot_vectors, 0.0))
            .into_batch_rows(RowCount::from(num_rows as u64))
            .unwrap();
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        // Keep partition 0 empty: stats previously failed when the first partition was empty.
        let mut centroid_values = Vec::with_capacity(num_partitions * DIM);
        centroid_values.extend(std::iter::repeat_n(2.0, DIM));
        centroid_values.extend(vector_values);
        centroid_values.extend(std::iter::repeat_n(-2.0, DIM));
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        );
        let ivf_params = IvfBuildParams::try_with_centroids(num_partitions, centroids).unwrap();
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default()
            .max_level(1)
            .num_edges(4)
            .ef_construction(4);
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_index".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats = dataset.index_statistics("test_index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

        assert_eq!(stats["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
        let indices = stats["indices"].as_array().unwrap();
        assert_eq!(indices.len(), 1);
        let index = &indices[0];
        assert_eq!(index["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
        assert_eq!(
            index["num_partitions"].as_number().unwrap(),
            &serde_json::Number::from(num_partitions)
        );
        assert_eq!(index["sub_index"]["index_type"].as_str().unwrap(), "HNSW");
        let partition_sizes = index["partitions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|partition| partition["size"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(partition_sizes.len(), num_partitions);
        assert_eq!(partition_sizes.iter().sum::<u64>(), num_rows as u64);
        assert_eq!(partition_sizes[0], 0);
        assert!(partition_sizes.contains(&0));
    }

    async fn test_distance_range(params: Option<VectorIndexParams>, nlist: usize) {
        match params.as_ref().map_or(DistanceType::L2, |p| p.metric_type) {
            DistanceType::Hamming => {
                test_distance_range_impl::<UInt8Type>(params, nlist, 0..255).await;
            }
            _ => {
                test_distance_range_impl::<Float32Type>(params, nlist, 0.0..1.0).await;
            }
        }
    }

    async fn test_distance_range_impl<T: ArrowPrimitiveType>(
        params: Option<VectorIndexParams>,
        nlist: usize,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range).await;

        let vector_column = "vector";
        let dist_type = params.as_ref().map_or(DistanceType::L2, |p| p.metric_type);
        if let Some(params) = params {
            dataset
                .create_index(&[vector_column], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
        }

        let query = vectors.value(0);
        let k = 10;
        let result = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), k);
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        let dists = result[DIST_COL].as_primitive::<Float32Type>().values();

        let part_idx = k / 2;
        let part_dist = dists[part_idx];

        let left_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), part_idx)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(None, Some(part_dist))
            .try_into_batch()
            .await
            .unwrap();
        let right_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k - part_idx)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(Some(part_dist), None)
            .try_into_batch()
            .await
            .unwrap();
        // don't verify the number of results and row ids for hamming distance,
        // because there are many vectors with the same distance
        if dist_type != DistanceType::Hamming {
            // Tolerate a single tied pair at the partition boundary. When
            // dists[part_idx - 1] == part_dist, the strict-less left filter
            // excludes both tied vectors and the inclusive right filter
            // includes both, shifting one row from left to right and dropping
            // row_ids[k - 1] off right_res's limit. Observed for Dot distance
            // on ARM where SIMD FMA yields tied float32 dot products that x86
            // does not. The distance-value assertions below still cover
            // partition correctness in both cases.
            let boundary_tie = part_idx > 0 && dists[part_idx - 1] == part_dist;
            let left_row_ids = left_res[ROW_ID].as_primitive::<UInt64Type>().values();
            let right_row_ids = right_res[ROW_ID].as_primitive::<UInt64Type>().values();
            if boundary_tie {
                assert_eq!(left_res.num_rows(), part_idx - 1);
                for i in 0..(part_idx - 1) {
                    assert_eq!(left_row_ids[i], row_ids[i]);
                }
                assert_eq!(right_res.num_rows(), k - part_idx);
                // right_row_ids[0..2] are the two tied vectors in tiebreaker
                // order; their identity is not pinned. right_row_ids[i] for
                // i >= 2 aligns with row_ids[part_idx + i - 1] because the
                // tie shifts one vector from left to right.
                for i in 2..(k - part_idx) {
                    assert_eq!(right_row_ids[i], row_ids[part_idx + i - 1]);
                }
            } else {
                assert_eq!(left_res.num_rows(), part_idx);
                assert_eq!(right_res.num_rows(), k - part_idx);
                row_ids.iter().enumerate().for_each(|(i, id)| {
                    if i < part_idx {
                        assert_eq!(left_row_ids[i], *id,);
                    } else {
                        assert_eq!(right_row_ids[i - part_idx], *id,);
                    }
                });
            }
        }
        let left_dists = left_res[DIST_COL].as_primitive::<Float32Type>().values();
        let right_dists = right_res[DIST_COL].as_primitive::<Float32Type>().values();
        left_dists.iter().for_each(|d| {
            assert!(d < &part_dist);
        });
        right_dists.iter().for_each(|d| {
            assert!(d >= &part_dist);
        });

        let exclude_last_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(dists.first().copied(), dists.last().copied())
            .try_into_batch()
            .await
            .unwrap();
        if dist_type != DistanceType::Hamming {
            let excluded_count = dists.iter().filter(|d| *d == dists.last().unwrap()).count();
            assert_eq!(exclude_last_res.num_rows(), k - excluded_count);
            let res_row_ids = exclude_last_res[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values();
            row_ids.iter().enumerate().for_each(|(i, id)| {
                if i < k - excluded_count {
                    assert_eq!(res_row_ids[i], *id);
                }
            });
        }
        let res_dists = exclude_last_res[DIST_COL]
            .as_primitive::<Float32Type>()
            .values();
        res_dists.iter().for_each(|d| {
            assert_ge!(*d, dists[0]);
            assert_lt!(*d, dists[k - 1]);
        });
    }

    #[tokio::test]
    async fn test_index_with_zero_vectors() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (batch, schema) = generate_batch::<Float32Type>(256, None, 0.0..1.0, false);
        let vector_field = schema.field(1).clone();
        let zero_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(vec![0.0; DIM]),
                        DIM as i32,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![batch, zero_batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let vector_column = vector_field.name();
        let params = VectorIndexParams::ivf_pq(4, 8, DIM / 8, DistanceType::Cosine, 50);
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
    }

    async fn test_recall<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        vector_column: &str,
        dataset: &Dataset,
        vectors: Arc<FixedSizeListArray>,
    ) {
        let query = vectors.value(0);
        let k = 100;
        let result = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let row_ids = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let dists = result[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let results = dists.into_iter().zip(row_ids).collect::<Vec<_>>();
        let row_ids = results.iter().map(|(_, id)| *id).collect::<HashSet<_>>();
        assert!(row_ids.len() == k);

        let gt = ground_truth(dataset, vector_column, &query, k, params.metric_type).await;

        let recall = row_ids.intersection(&gt).count() as f32 / k as f32;
        assert!(
            recall >= recall_requirement,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    /// Rewrite the auxiliary storage file to the legacy PQ format (codebook
    /// embedded in schema metadata rather than stored as a global buffer), then
    /// commit a `CreateIndex` transaction so the manifest records the correct
    /// new file size.
    /// Rewrite the auxiliary PQ storage file with the codebook inlined into
    /// schema metadata (legacy format). Uses a new UUID to avoid cache key
    /// collisions with the original index.
    async fn rewrite_pq_storage(dataset: &mut Dataset, old_meta: &IndexMetadata) -> Result<()> {
        use crate::dataset::transaction::{Operation, Transaction};

        let obj_store = Arc::new(ObjectStore::local());
        let old_dir = dataset.indices_dir().join(old_meta.uuid.to_string());
        let new_uuid = uuid::Uuid::new_v4();
        let new_dir = dataset.indices_dir().join(new_uuid.to_string());

        // Copy the main index file to the new directory unchanged.
        obj_store
            .copy(
                &old_dir.clone().join(super::INDEX_FILE_NAME),
                &new_dir.clone().join(super::INDEX_FILE_NAME),
            )
            .await?;

        // Read the original auxiliary file.
        let old_aux_path = old_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        let scheduler =
            ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());
        let reader = FileReader::try_open(
            scheduler
                .open_file(&old_aux_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;

        // Rewrite auxiliary file with PQ codebook inlined into schema metadata.
        let mut metadata = reader.schema().metadata.clone();
        let projection = lance_file::versions::reader_projection_from_whole_schema(
            reader.schema(),
            reader.metadata().version(),
        );
        let batches = reader
            .read_stream_projected(
                lance_io::ReadBatchParams::RangeFull,
                u32::MAX,
                u32::MAX,
                projection,
                lance_encoding::decoder::FilterExpression::no_filter(),
            )
            .await?;
        use futures::TryStreamExt as _;
        let batches = batches.try_collect::<Vec<_>>().await?;
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
        let new_aux_path = new_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        let mut writer = lance_file::versions::create_writer(
            reader.metadata().version(),
            obj_store.create(&new_aux_path).await?,
            batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&batch).await?;
        writer
            .add_global_buffer(reader.read_global_buffer(1).await?)
            .await?;
        let codebook = reader.read_global_buffer(2).await?;
        let pq_metadata: Vec<String> = serde_json::from_str(&metadata[STORAGE_METADATA_KEY])?;
        let mut pq_metadata: ProductQuantizationMetadata = serde_json::from_str(&pq_metadata[0])?;
        pq_metadata.codebook_position = 0;
        pq_metadata.codebook_tensor = codebook.to_vec();
        let pq_metadata = serde_json::to_string(&pq_metadata)?;
        metadata.insert(
            STORAGE_METADATA_KEY.to_owned(),
            serde_json::to_string(&vec![pq_metadata])?,
        );
        for (key, value) in metadata {
            writer.add_schema_metadata(key, value);
        }
        writer.finish().await?;

        // Build new IndexMetadata with the new UUID and file sizes.
        let new_files =
            lance_table::format::list_index_files_with_sizes(&obj_store, &new_dir).await?;
        let mut new_meta = old_meta.clone();
        new_meta.uuid = new_uuid;
        new_meta.files = Some(new_files);

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_meta],
                removed_indices: vec![old_meta.clone()],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_legacy_non_divisible_pq_search() {
        const DIM: usize = 64;
        const PERSISTED_DIM: usize = 56;

        let test_dir = copy_test_data_to_tmp("v0.10.15/non_divisible_pq").unwrap();
        let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
        let query = Float32Array::from(
            (1..=DIM)
                .map(|value| value as f32 + if value <= PERSISTED_DIM { 1.0 } else { 1_000.0 })
                .collect::<Vec<_>>(),
        );

        let result = dataset
            .scan()
            .nearest("vector", &query, 1)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(result.num_rows(), 1);
        assert_eq!(
            result[DIST_COL].as_primitive::<Float32Type>().values(),
            &[PERSISTED_DIM as f32]
        );
    }

    #[tokio::test]
    async fn test_pq_storage_backwards_compat() {
        let test_dir = copy_test_data_to_tmp("v0.27.1/pq_in_schema").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        // Just make sure we can query the index.
        let dataset = Dataset::open(test_uri).await.unwrap();
        let query_vec = Float32Array::from(vec![0_f32; 32]);
        let search_result = dataset
            .scan()
            .nearest("vec", &query_vec, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(search_result.num_rows(), 5);

        let obj_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());

        async fn get_pq_metadata(
            dataset: &Dataset,
            scheduler: Arc<ScanScheduler>,
        ) -> ProductQuantizationMetadata {
            let index = dataset.load_indices().await.unwrap();
            let index_path = dataset.indices_dir().join(index[0].uuid.to_string());
            let file_scheduler = scheduler
                .open_file(
                    &index_path.clone().join(INDEX_AUXILIARY_FILE_NAME),
                    &CachedFileSize::unknown(),
                )
                .await
                .unwrap();
            let reader = FileReader::try_open(
                file_scheduler,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await
            .unwrap();
            let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
            serde_json::from_str(&serde_json::from_str::<Vec<String>>(metadata).unwrap()[0])
                .unwrap()
        }
        let pq_meta: ProductQuantizationMetadata =
            get_pq_metadata(&dataset, scheduler.clone()).await;
        assert!(pq_meta.buffer_index().is_none());

        // If we add data and optimize indices, then we start using the global
        // buffer for the PQ index.
        let new_data = RecordBatch::try_new(
            Arc::new(Schema::from(dataset.schema())),
            vec![
                Arc::new(Int64Array::from(vec![0])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 32]), 32)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let mut dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![new_data])
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        let pq_meta: ProductQuantizationMetadata =
            get_pq_metadata(&dataset, scheduler.clone()).await;
        assert!(pq_meta.buffer_index().is_some());
    }

    #[tokio::test]
    async fn test_optimize_with_empty_partition() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let num_rows = dataset.count_all_rows().await.unwrap();
        let nlist = num_rows + 2;
        let centroids = generate_random_array(nlist * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params =
            IvfBuildParams::try_with_centroids(nlist, Arc::new(ivf_centroids)).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
            ivf_params,
            PQBuildParams::default(),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        append_dataset::<Float32Type>(&mut dataset, 1, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_compaction_remaps_second_delta_with_shared_partition_topology() {
        const INDEX_NAME: &str = "vector_idx";
        const BASE_ROWS_PER_PARTITION: usize = 2_200;
        const SMALL_APPEND_ROWS: usize = 64;
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            lightweight_pq_params(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let template_batch = dataset
            .take_rows(&[0], dataset.schema().clone())
            .await
            .unwrap();
        let template_values = template_batch["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let mut append_params = WriteParams {
            max_rows_per_file: 32,
            max_rows_per_group: 32,
            ..Default::default()
        };
        append_params.mode = WriteMode::Append;
        append_constant_vector_with_params(
            &mut dataset,
            SMALL_APPEND_ROWS,
            &template_values,
            Some(append_params),
        )
        .await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let stats_before: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats_before["num_indices"].as_u64().unwrap(), 2);
        let partitions_before: Vec<usize> = stats_before["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(partitions_before.len(), 2);
        let base_partition_count = partitions_before
            .iter()
            .copied()
            .max()
            .expect("expected at least one partition count");
        assert!(base_partition_count >= 2);
        assert!(
            partitions_before
                .iter()
                .all(|count| *count == base_partition_count)
        );

        let indices_meta = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(indices_meta.len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 5_000,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stats_after_compaction: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats_after_compaction["num_indices"].as_u64().unwrap(), 2);
        let mut partitions_after: Vec<usize> = stats_after_compaction["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
            .collect();
        partitions_after.sort_unstable();
        assert_eq!(
            partitions_after,
            vec![base_partition_count, base_partition_count]
        );
    }

    #[tokio::test]
    async fn test_spfresh_join_split() {
        const INDEX_NAME: &str = "vector_idx";
        const NLIST: usize = 2;
        const NO_SPLIT_APPEND_ROWS: usize = 32;
        // The joined base and no-split delta contain 2,265 rows. This append
        // takes the single IVF-PQ partition one row past its 32,768-row limit.
        const SPLIT_APPEND_ROWS: usize = 30_504;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let cluster_sizes = [100, 2_200];
        let total_rows: usize = cluster_sizes.iter().sum();

        let mut centroid_values = Vec::new();
        for i in 0..NLIST {
            for j in 0..DIM {
                centroid_values.push(if j == 0 { (i as f32) * 10.0 } else { 0.0 });
            }
        }
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        );

        let mut ids = Vec::new();
        let mut vector_values = Vec::new();
        let mut current_id = 0u64;
        for (cluster_idx, &size) in cluster_sizes.iter().enumerate() {
            let centroid_base = (cluster_idx as f32) * 10.0;
            for _ in 0..size {
                ids.push(current_id);
                current_id += 1;
                for j in 0..DIM {
                    vector_values.push(if j == 0 {
                        centroid_base + (current_id % 100) as f32 * 0.005
                    } else {
                        (current_id % 50) as f32 * 0.01
                    });
                }
            }
        }

        let ids_array = Arc::new(UInt64Array::from(ids.clone()));
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vector_values), DIM as i32)
                .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let ivf_params = IvfBuildParams::try_with_centroids(NLIST, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            lightweight_pq_params(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let template_id = cluster_sizes[0] as u64;
        let template_batch = dataset
            .take_rows(&[template_id], dataset.schema().clone())
            .await
            .unwrap();
        let template_values = template_batch["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        assert_eq!(
            template_values.len(),
            DIM,
            "Template vector should match DIM"
        );

        let mut next_id = total_rows as u64;
        let mut expected_rows = total_rows;

        let (deleted_rows, appended_rows, actual_partitions) =
            shrink_smallest_partition(&mut dataset, INDEX_NAME, 1, &mut next_id).await;
        expected_rows = expected_rows - deleted_rows + appended_rows;
        assert_eq!(actual_partitions, 1);
        assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            &mut next_id,
            NO_SPLIT_APPEND_ROWS,
            1,
            expected_rows + NO_SPLIT_APPEND_ROWS,
            2,
            false,
        )
        .await;
        expected_rows += NO_SPLIT_APPEND_ROWS;

        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            &mut next_id,
            SPLIT_APPEND_ROWS,
            2,
            expected_rows + SPLIT_APPEND_ROWS,
            1,
            true,
        )
        .await;
    }

    #[tokio::test]
    async fn test_partition_split_on_append_multivec() {
        const INDEX_NAME: &str = "vector_idx";
        const VECTORS_PER_ROW: usize = 3;
        // 512 base rows and this append flatten to 33,036 vectors, just over
        // the 32,768-vector IVF-PQ split threshold.
        const APPEND_ROWS: usize = 10_500;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, _) =
            generate_multivec_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
            IvfBuildParams::new(1),
            lightweight_pq_params(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let initial_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(initial_ctx.num_partitions(), 1);

        append_dataset::<Float32Type>(&mut dataset, APPEND_ROWS, 0.0..0.05).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let expected_rows = NUM_ROWS + APPEND_ROWS;
        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            2,
            "Expected one oversized multivector partition to split, stats: {}",
            final_ctx.stats_json()
        );
        let partitions = final_ctx.stats()["indices"][0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        assert_eq!(partitions.len(), 2);
        assert_eq!(
            partitions
                .iter()
                .map(|partition| partition["size"].as_u64().unwrap() as usize)
                .sum::<usize>(),
            expected_rows * VECTORS_PER_ROW
        );
        assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

        let query_batch = dataset
            .scan()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let query = query_batch["vector"].as_list::<i32>().value(0);
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", &query, 10)
            .unwrap()
            .distance_metric(DistanceType::Cosine)
            .try_into_batch()
            .await
            .unwrap();
        let mut row_ids = HashSet::new();
        for row_id in results[ROW_ID].as_primitive::<UInt64Type>().values() {
            assert!(row_ids.insert(*row_id), "duplicate row id {row_id}");
        }
    }

    #[tokio::test]
    async fn test_split_multiple_partitions_in_one_optimize() {
        const INDEX_NAME: &str = "vector_idx";
        const BASE_ROWS_PER_PARTITION: usize = 512;
        // Each IVF-FLAT partition reaches 16,512 rows, just over its 16,384-row
        // split threshold.
        const APPEND_ROWS_PER_PARTITION: usize = 16_000;
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let initial_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(initial_ctx.num_partitions(), 2);
        let templates = offsets
            .iter()
            .map(|offset| {
                let mut template = vec![0.0; DIM];
                template[0] = *offset;
                template
            })
            .collect::<Vec<_>>();

        append_partition_templates(&mut dataset, APPEND_ROWS_PER_PARTITION, &templates).await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            4,
            "Expected both original partitions to split in one optimize, stats: {}",
            final_ctx.stats_json()
        );

        let indices = final_ctx.stats()["indices"]
            .as_array()
            .expect("indices should be present");
        assert_eq!(
            indices.len(),
            1,
            "Expected split optimize to merge into one index, stats: {}",
            final_ctx.stats_json()
        );

        let partitions = indices[0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        assert_eq!(partitions.len(), 4);
        let expected_rows = 2 * BASE_ROWS_PER_PARTITION + 2 * APPEND_ROWS_PER_PARTITION;
        let total_partition_rows = partitions
            .iter()
            .map(|part| part["size"].as_u64().unwrap() as usize)
            .sum::<usize>();
        assert_eq!(total_partition_rows, expected_rows);
        assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

        let mut indexed_row_ids = HashSet::with_capacity(expected_rows);
        for partition_idx in 0..final_ctx.num_partitions() {
            for row_id in load_flat_partition_row_ids(final_ctx.ivf_flat(), partition_idx).await {
                assert!(
                    indexed_row_ids.insert(row_id),
                    "row id {row_id} appeared in multiple partitions"
                );
            }
        }
        assert_eq!(indexed_row_ids.len(), expected_rows);
        let live_row_ids = dataset.scan().with_row_id().try_into_batch().await.unwrap()[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(indexed_row_ids, live_row_ids);

        let nearest = dataset
            .scan()
            .with_row_id()
            .nearest("vector", &Float32Array::from(templates[0].clone()), 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = nearest[ROW_ID].as_primitive::<UInt64Type>();
        let mut seen = HashSet::new();
        for row_id in ids.values() {
            assert!(seen.insert(*row_id), "Duplicate row id found: {}", row_id);
        }
    }

    #[tokio::test]
    async fn test_join_partition_on_delete_multivec() {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        const MULTIVEC_PER_ROW: usize = 3;
        const APPEND_ROWS: usize = 32;
        let cluster_sizes = [800, 800, 400];
        // Multivector indices require cosine distance. Unit centroids in three
        // distinct directions avoid the collinear assignment in the old fixture.
        let centroids = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)];
        let total_rows = cluster_sizes.iter().sum::<usize>();
        let mut dataset = {
            let (batch, schema) =
                generate_clustered_multivec_batch(&cluster_sizes, &centroids, MULTIVEC_PER_ROW, 0);
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            Dataset::write(
                batches,
                test_uri,
                Some(WriteParams {
                    mode: crate::dataset::WriteMode::Overwrite,
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
        };

        let ivf_params =
            IvfBuildParams::try_with_centroids(centroids.len(), build_centroids_2d(&centroids))
                .unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
            ivf_params,
            lightweight_pq_params(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let index_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(index_ctx.num_partitions(), 3);

        let mut logical_row_ids = {
            let ivf = index_ctx.ivf();
            let mut smallest: Option<HashSet<u64>> = None;
            for i in 0..ivf.ivf.num_partitions() {
                let partition_row_ids = load_partition_row_ids(ivf, i)
                    .await
                    .into_iter()
                    .collect::<HashSet<_>>();
                if partition_row_ids.is_empty() {
                    continue;
                }

                let is_better = smallest
                    .as_ref()
                    .map(|existing| partition_row_ids.len() < existing.len())
                    .unwrap_or(true);
                if is_better {
                    smallest = Some(partition_row_ids);
                }
            }
            smallest
                .expect("expected a non-empty partition")
                .into_iter()
                .collect::<Vec<_>>()
        };
        logical_row_ids.sort_unstable();
        assert_eq!(logical_row_ids.len(), cluster_sizes[2]);
        let retained_id = logical_row_ids[0];
        delete_ids(&mut dataset, &logical_row_ids[1..]).await;
        compact_after_deletions(&mut dataset).await;

        let (append_batch, append_schema) = generate_clustered_multivec_batch(
            &[APPEND_ROWS],
            &centroids[2..],
            MULTIVEC_PER_ROW,
            total_rows as u64,
        );
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(append_batch)], append_schema),
                None,
            )
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            2,
            "Expected the reduced multivector partition to join, stats: {}",
            final_ctx.stats_json()
        );
        assert_eq!(final_ctx.stats()["num_indices"].as_u64().unwrap(), 1);
        let expected_rows = total_rows - cluster_sizes[2] + 1 + APPEND_ROWS;
        assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

        let sample_row = dataset
            .scan()
            .with_row_id()
            .filter(&format!("id = {retained_id}"))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(sample_row.num_rows(), 1);
        let retained_row_id = sample_row[ROW_ID].as_primitive::<UInt64Type>().value(0);
        let mut indexed_row_id_counts = HashMap::new();
        for partition_idx in 0..final_ctx.num_partitions() {
            for row_id in load_partition_row_ids(final_ctx.ivf(), partition_idx).await {
                *indexed_row_id_counts.entry(row_id).or_insert(0usize) += 1;
            }
        }
        assert_eq!(
            indexed_row_id_counts.values().sum::<usize>(),
            expected_rows * MULTIVEC_PER_ROW
        );
        assert_eq!(
            indexed_row_id_counts.get(&retained_row_id),
            Some(&MULTIVEC_PER_ROW),
            "all vectors for the retained logical row should survive the join"
        );
        assert!(
            indexed_row_id_counts
                .values()
                .all(|count| *count == MULTIVEC_PER_ROW),
            "each logical row should have exactly {MULTIVEC_PER_ROW} indexed vectors"
        );
        let live_row_ids = dataset.scan().with_row_id().try_into_batch().await.unwrap()[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(live_row_ids.len(), expected_rows);
        assert_eq!(
            indexed_row_id_counts
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            live_row_ids
        );
    }

    async fn row_ids_matching(dataset: &Dataset, predicate: &str) -> HashSet<u64> {
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.filter(predicate).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect()
    }

    struct OptimizeAfterDelete {
        deleted_row_ids: HashSet<u64>,
        index_row_ids: HashSet<u64>,
        num_partitions_after: usize,
        stats_json: String,
    }

    /// Shared scenario for the issue-7701 regressions: stable-row-id dataset,
    /// IVF_FLAT index, scattered delete, optimize. Asserts the invariants both
    /// partition adjustments must hold -- no live row lost, no id that never
    /// existed -- and returns the state for the mode-specific assertions.
    async fn optimize_after_delete(
        total_rows: usize,
        nlist: usize,
        delete_predicate: &str,
        keep_predicate: &str,
    ) -> OptimizeAfterDelete {
        const INDEX_NAME: &str = "vector_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_batch::<Float32Type>(total_rows, None, 0.0..1.0, false);
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = VectorIndexParams::ivf_flat(nlist, DistanceType::L2);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let deleted_row_ids = row_ids_matching(&dataset, delete_predicate).await;
        let live_row_ids = row_ids_matching(&dataset, keep_predicate).await;
        dataset.delete(delete_predicate).await.unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        let num_partitions_after = final_ctx.num_partitions();
        let stats_json = final_ctx.stats_json().to_string();
        let flat = final_ctx
            .index
            .as_any()
            .downcast_ref::<IvfFlatIndex>()
            .expect("expected IvfFlat index");
        let mut index_row_ids = HashSet::new();
        for part in 0..flat.ivf.num_partitions() {
            index_row_ids.extend(load_flat_partition_row_ids(flat, part).await);
        }

        for row_id in &live_row_ids {
            assert!(
                index_row_ids.contains(row_id),
                "live row id {} missing from index after optimize",
                row_id
            );
        }
        for row_id in &index_row_ids {
            assert!(
                live_row_ids.contains(row_id) || deleted_row_ids.contains(row_id),
                "unexpected row id {} in index after optimize",
                row_id
            );
        }

        OptimizeAfterDelete {
            deleted_row_ids,
            index_row_ids,
            num_partitions_after,
            stats_json,
        }
    }

    #[tokio::test]
    async fn test_optimize_join_after_delete_with_stable_row_ids() {
        // Regression test for https://github.com/lance-format/lance/issues/7701:
        // every partition (400 rows / 4) is under the IVF_FLAT join threshold,
        // so optimize joins the smallest after a scattered delete.
        let run = optimize_after_delete(400, 4, "id % 3 = 0", "id % 3 != 0").await;

        assert_eq!(
            run.num_partitions_after, 3,
            "optimize should have joined the smallest partition, got stats: {}",
            run.stats_json
        );

        // The join reads every partition's stored rows through the merge filter, so
        // deleted ids are dropped index-wide and not just from the joined partition.
        for row_id in &run.deleted_row_ids {
            assert!(
                !run.index_row_ids.contains(row_id),
                "deleted row id {} still in index after join",
                row_id
            );
        }
    }

    #[tokio::test]
    async fn test_optimize_split_after_delete_with_stable_row_ids() {
        // Regression test for https://github.com/lance-format/lance/issues/7701:
        // one partition holds more than 4x the IVF_FLAT target, so optimize
        // splits it after a scattered delete. This path reaches
        // filter_deleted_ids through reshuffle_partitions, unlike the join
        // path's take_vectors.
        let run = optimize_after_delete(20_000, 1, "id % 5 = 0", "id % 5 != 0").await;

        assert!(
            run.num_partitions_after > 1,
            "optimize should have split the oversized partition, got stats: {}",
            run.stats_json
        );

        // The split rebuilds the whole partition from live rows: no deleted
        // ids remain.
        for row_id in &run.deleted_row_ids {
            assert!(
                !run.index_row_ids.contains(row_id),
                "deleted row id {} still in index after split",
                row_id
            );
        }
    }

    #[tokio::test]
    async fn test_prewarm_ivf_pq() {
        use lance_io::assert_io_eq;

        const INDEX_NAME: &str = "my_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::new(4, 4),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        append_dataset::<Float32Type>(&mut dataset, 8, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        // Reopen to avoid carrying index state in memory from index creation.
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(indices.len(), 2, "expected two index deltas");
        let unique_uuids: HashSet<_> = indices.iter().map(|meta| meta.uuid).collect();
        assert_eq!(unique_uuids.len(), 2, "expected two unique index UUIDs");

        // Reset IO stats after index creation.
        dataset.object_store.as_ref().io_stats_incremental();

        // Prewarm should perform IO to load all index deltas into cache.
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert!(
            stats.read_iops > 0,
            "prewarm should have read from disk, but read_iops was 0"
        );

        // Query should not perform IO after prewarming all deltas.
        let q = vectors.value(0);
        dataset
            .scan()
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap()
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "query should not perform IO after prewarm"
        );

        // Second prewarm should not need IO (already cached).
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
    }

    /// Index-cache backend that can drop partition entries on demand.
    ///
    /// Used to simulate cache invalidation after a credential rotation:
    /// partition keys are opaque digests, so the test supplies the exact
    /// [`InternalCacheKey`] set to bypass once the index identity is known
    /// (see [`ivf_partition_cache_keys`]).
    #[derive(Debug)]
    struct PartitionBypassCacheBackend {
        inner: lance_core::cache::MokaCacheBackend,
        partition_keys: std::sync::Mutex<HashSet<lance_core::cache::InternalCacheKey>>,
        bypass_partitions: AtomicBool,
        partition_hits: AtomicUsize,
    }

    impl PartitionBypassCacheBackend {
        fn new() -> Self {
            Self {
                inner: lance_core::cache::MokaCacheBackend::with_capacity(256 * 1024 * 1024),
                partition_keys: std::sync::Mutex::new(HashSet::new()),
                bypass_partitions: AtomicBool::new(false),
                partition_hits: AtomicUsize::new(0),
            }
        }

        fn set_partition_keys(&self, partition_keys: HashSet<lance_core::cache::InternalCacheKey>) {
            *self.partition_keys.lock().unwrap() = partition_keys;
        }

        fn is_partition(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
            self.partition_keys.lock().unwrap().contains(key)
        }

        fn set_bypass_partitions(&self, bypass_partitions: bool) {
            self.bypass_partitions
                .store(bypass_partitions, Ordering::Relaxed);
        }

        fn should_bypass(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
            self.bypass_partitions.load(Ordering::Relaxed) && self.is_partition(key)
        }

        /// Whether the backend currently holds an entry for `key`.
        async fn contains(&self, key: &lance_core::cache::InternalCacheKey) -> bool {
            self.inner.get(key, None).await.is_some()
        }

        fn partition_hits(&self) -> usize {
            self.partition_hits.load(Ordering::Relaxed)
        }
    }

    /// Derive the internal cache keys of the IVF partition entries for an
    /// index, replicating the namespace path
    /// `dataset URI -> index UUID -> frag-reuse UUID` used when opening the
    /// index. V3 partitions use [`IVFPartitionKey`]; legacy (v0.1/v0.2)
    /// indices use `LegacyIVFPartitionKey`.
    fn ivf_partition_cache_keys(
        dataset_uri: &str,
        uuid: &uuid::Uuid,
        fri_uuid: Option<&uuid::Uuid>,
        num_partitions: usize,
        index_version: &IndexFileVersion,
    ) -> HashSet<lance_core::cache::InternalCacheKey> {
        use lance_core::cache::{CacheKey, CacheNamespace, KeyBuilder, UnsizedCacheKey};

        let mut namespace = CacheNamespace::root().child(dataset_uri);
        namespace = namespace.child(uuid.as_hyphenated().to_string().as_str());
        if let Some(fri_uuid) = fri_uuid {
            namespace = namespace.child(fri_uuid.as_hyphenated().to_string().as_str());
        }

        (0..num_partitions)
            .map(|partition_id| {
                if matches!(index_version, IndexFileVersion::V3) {
                    let cache_key =
                        IVFPartitionKey::<FlatIndex, ProductQuantizer>::new(partition_id);
                    let mut builder = KeyBuilder::new(
                        namespace,
                        IVFPartitionKey::<FlatIndex, ProductQuantizer>::stable_type_id(),
                        IVFPartitionKey::<FlatIndex, ProductQuantizer>::schema(),
                    );
                    cache_key.write_key(&mut builder);
                    builder.finish()
                } else {
                    let cache_key =
                        crate::index::vector::ivf::LegacyIVFPartitionKey::new(partition_id);
                    let mut builder = KeyBuilder::new(
                        namespace,
                        crate::index::vector::ivf::LegacyIVFPartitionKey::stable_type_id(),
                        crate::index::vector::ivf::LegacyIVFPartitionKey::schema(),
                    );
                    cache_key.write_key(&mut builder);
                    builder.finish()
                }
            })
            .collect()
    }

    #[async_trait::async_trait]
    impl lance_core::cache::CacheBackend for PartitionBypassCacheBackend {
        async fn get(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            codec: Option<lance_core::cache::CacheCodec>,
        ) -> Option<lance_core::cache::CacheEntry> {
            if self.should_bypass(key) {
                None
            } else {
                let entry = self.inner.get(key, codec).await;
                if entry.is_some() && self.is_partition(key) {
                    self.partition_hits.fetch_add(1, Ordering::Relaxed);
                }
                entry
            }
        }

        async fn insert(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            entry: lance_core::cache::CacheEntry,
            size_bytes: usize,
            codec: Option<lance_core::cache::CacheCodec>,
        ) {
            if !self.should_bypass(key) {
                self.inner.insert(key, entry, size_bytes, codec).await;
            }
        }

        async fn get_or_insert<'a>(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            loader: std::pin::Pin<
                Box<
                    dyn futures::Future<Output = Result<(lance_core::cache::CacheEntry, usize)>>
                        + Send
                        + 'a,
                >,
            >,
            codec: Option<lance_core::cache::CacheCodec>,
        ) -> Result<(lance_core::cache::CacheEntry, bool)> {
            if self.should_bypass(key) {
                let (entry, _) = loader.await?;
                Ok((entry, false))
            } else {
                let result = self.inner.get_or_insert(key, loader, codec).await;
                if result.as_ref().is_ok_and(|(_, is_cache_hit)| *is_cache_hit)
                    && self.is_partition(key)
                {
                    self.partition_hits.fetch_add(1, Ordering::Relaxed);
                }
                result
            }
        }

        async fn clear(&self) {
            self.inner.clear().await;
        }

        async fn num_entries(&self) -> usize {
            self.inner.num_entries().await
        }

        async fn size_bytes(&self) -> usize {
            self.inner.size_bytes().await
        }

        fn approx_num_entries(&self) -> usize {
            self.inner.approx_num_entries()
        }

        fn approx_size_bytes(&self) -> usize {
            self.inner.approx_size_bytes()
        }
    }

    /// Integration test: create a vector index, prewarm it through a
    /// serializing cache backend, then query. Verifies that entries are
    /// serialized to bytes and that queries produce correct results after
    /// deserialization.
    #[rstest]
    #[case::ivf_pq(
        VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        ),
        <PartitionEntry<FlatIndex, ProductQuantizer> as CacheCodecImpl>::TYPE_ID
    )]
    #[case::ivf_hnsw_sq(
        VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        ),
        <PartitionEntry<HNSW, ScalarQuantizer> as CacheCodecImpl>::TYPE_ID
    )]
    #[tokio::test]
    async fn test_prewarm_and_query_with_serializing_backend(
        #[case] params: VectorIndexParams,
        #[case] partition_type_id: &'static str,
    ) {
        use crate::utils::test::serializing_cache::SerializingCacheBackend;
        use lance_io::assert_io_eq;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create dataset with vector index using default cache
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("serde_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let q = Float32Array::from_iter_values(repeat_n(0.5, DIM));
        let expected = ground_truth(&dataset, "vector", &q, 10, DistanceType::L2).await;

        // Re-open with the serializing backend
        let backend = Arc::new(SerializingCacheBackend::new());
        let session = Arc::new(crate::session::Session::with_index_cache_backend(
            backend.clone(),
            128 * 1024 * 1024,
            Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
        ));
        let dataset = crate::DatasetBuilder::from_uri(test_uri)
            .with_session(session)
            .load()
            .await
            .unwrap();

        // Prewarm — this should serialize entries into the backend
        dataset.prewarm_index("serde_idx").await.unwrap();
        let serialized = backend.serialized_entry_count().await;
        let state_type_id = IvfStateEntryBox::TYPE_ID;
        let state_inserts = backend.serialized_insert_count(state_type_id).await;
        let partition_inserts = backend.serialized_insert_count(partition_type_id).await;
        let passthrough = backend.l1_entry_count().await;
        assert!(
            serialized > 0,
            "prewarm should have serialized entries into the backend"
        );
        assert_eq!(
            passthrough, 0,
            "all index cache entries should have codecs (nothing in passthrough), \
             but found {passthrough} passthrough entries"
        );

        drop(dataset);
        let backend = Arc::new(backend.restart());
        assert_eq!(
            backend.l1_entry_count().await,
            0,
            "restarting must discard the in-memory L1"
        );
        assert_eq!(
            backend.serialized_entry_count().await,
            serialized,
            "restarting must retain the serialized IVF state and partitions"
        );
        let session = Arc::new(crate::session::Session::with_index_cache_backend(
            backend.clone(),
            128 * 1024 * 1024,
            Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
        ));
        let dataset = crate::DatasetBuilder::from_uri(test_uri)
            .with_session(session)
            .load()
            .await
            .unwrap();

        // Query — the recreated backend will deserialize entries from bytes.
        // All index entries are in serialized form, so every cache hit involves
        // a deserialization round-trip.
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", &q, 10)
            .unwrap()
            .nprobes(4)
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            backend.serialized_insert_count(state_type_id).await,
            state_inserts,
            "the first restarted query must reuse the serialized IVF state"
        );
        assert_eq!(
            backend.serialized_insert_count(partition_type_id).await,
            partition_inserts,
            "the first restarted query must reuse every serialized IVF partition"
        );
        assert_eq!(results.num_rows(), 10, "should return 10 nearest neighbors");

        // Verify distances are sorted (ascending for L2)
        let distances: Vec<f32> = results
            .column_by_name("_distance")
            .unwrap()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        for w in distances.windows(2) {
            assert!(w[1] >= w[0], "distances should be sorted ascending");
        }

        let row_ids = results[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = row_ids.intersection(&expected).count() as f32 / expected.len() as f32;
        assert_ge!(
            recall,
            0.5,
            "serialized IVF query recall is below threshold: {recall}"
        );

        dataset.object_store.as_ref().io_stats_incremental();
        dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .nprobes(4)
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "warmed IVF query should not perform IO after backend restart"
        );
    }

    #[rstest]
    #[case::v3(IndexFileVersion::V3)]
    #[case::legacy(IndexFileVersion::Legacy)]
    #[tokio::test]
    async fn test_vector_cache_uses_current_object_store(#[case] index_version: IndexFileVersion) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        append_dataset::<Float32Type>(&mut dataset, NUM_ROWS, 0.0..1.0).await;
        assert_eq!(dataset.get_fragments().len(), 2);

        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        )
        .version(index_version.clone())
        .clone();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("credential_rotation_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();
        let index_meta = dataset
            .load_indices_by_name("credential_rotation_idx")
            .await
            .unwrap()
            .pop()
            .unwrap();
        let query = vectors.value(0);
        let ground_truth = ground_truth(&dataset, "vector", &query, 20, DistanceType::L2).await;

        let cache_backend = Arc::new(PartitionBypassCacheBackend::new());
        let session = Arc::new(crate::session::Session::with_index_cache_backend(
            cache_backend.clone(),
            128 * 1024 * 1024,
            Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
        ));
        let dataset = crate::DatasetBuilder::from_uri(test_uri)
            .with_session(session)
            .load()
            .await
            .unwrap();

        let store_params_a = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(
                    "credential_generation".to_owned(),
                    "secret-generation-a".to_owned(),
                )]),
            ))),
            ..Default::default()
        };
        let (store_a, _) = ObjectStore::from_uri_and_params(
            dataset.session().store_registry(),
            dataset.uri(),
            &store_params_a,
        )
        .await
        .unwrap();
        let dataset_a = dataset.with_object_store(store_a.clone(), Some(store_params_a));

        let store_params_b = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(
                    "credential_generation".to_owned(),
                    "secret-generation-b".to_owned(),
                )]),
            ))),
            ..Default::default()
        };
        let (store_b, _) = ObjectStore::from_uri_and_params(
            dataset.session().store_registry(),
            dataset.uri(),
            &store_params_b,
        )
        .await
        .unwrap();
        assert!(!Arc::ptr_eq(&store_a, &store_b));
        let dataset_b = dataset.with_object_store(store_b.clone(), Some(store_params_b));

        let _ = store_a.io_stats_incremental();
        let _ = store_b.io_stats_incremental();

        dataset_a
            .scan()
            .nearest("vector", &query, 20)
            .unwrap()
            .minimum_nprobes(4)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let frag_reuse_uuid = dataset_a.frag_reuse_index_uuid().await;
        let state_cache_key =
            crate::index::IvfIndexStateCacheKey::new(&index_meta.uuid, frag_reuse_uuid.as_ref());
        let cached_state = if matches!(index_version, IndexFileVersion::V3) {
            Some(
                dataset_a
                    .index_cache
                    .get_with_key(&state_cache_key)
                    .await
                    .expect("V3 IVF state should be cached"),
            )
        } else {
            None
        };
        let index_path_fragment = format!("_indices/{}", index_meta.uuid);
        let first_store_stats = store_a.io_stats_incremental();
        assert!(
            first_store_stats
                .requests
                .iter()
                .any(|request| request.path.as_ref().contains(&index_path_fragment)),
            "the first query should read the index through the first object store: {first_store_stats:#?}"
        );
        let partition_keys = ivf_partition_cache_keys(
            dataset.uri(),
            &index_meta.uuid,
            frag_reuse_uuid.as_ref(),
            4,
            &index_version,
        );
        cache_backend.set_partition_keys(partition_keys.clone());
        for partition_key in &partition_keys {
            assert!(
                cache_backend.contains(partition_key).await,
                "the first query should populate portable partition entries"
            );
        }
        let index_entries_after_a = dataset.session().index_cache_stats().await.num_entries;
        let metadata_entries_after_a = dataset.session().metadata_cache_stats().await.num_entries;
        let _ = store_b.io_stats_incremental();

        cache_backend.set_bypass_partitions(true);
        let results = dataset_b
            .scan()
            .nearest("vector", &query, 20)
            .unwrap()
            .minimum_nprobes(4)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = results[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = row_ids.intersection(&ground_truth).count() as f32 / 20.0;
        assert_ge!(recall, 0.5);

        let old_store_stats = store_a.io_stats_incremental();
        let old_store_index_reads = old_store_stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains(&index_path_fragment))
            .count();
        let new_store_stats = store_b.io_stats_incremental();
        let new_store_index_reads = new_store_stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains(&index_path_fragment))
            .count();
        if matches!(index_version, IndexFileVersion::V3) {
            assert_eq!(
                old_store_index_reads, 0,
                "the new dataset query must not use readers bound to the old object store: {old_store_stats:#?}"
            );
            assert!(
                new_store_index_reads > 0,
                "the new dataset query should read the index through the new object store: {new_store_stats:#?}"
            );
        } else {
            // Legacy live indices are shared across dataset opens: their
            // readers stay bound to the object store that first populated the
            // cache, so the second dataset keeps reading through the old store.
            assert!(
                old_store_index_reads > 0,
                "the cached legacy index should keep reading through the original object store: {old_store_stats:#?}"
            );
            assert_eq!(
                new_store_index_reads, 0,
                "the cached legacy index must not reopen through the new object store: {new_store_stats:#?}"
            );
        }
        if let Some(cached_state) = cached_state {
            let state_after_rotation = dataset_b
                .index_cache
                .get_with_key(&state_cache_key)
                .await
                .expect("V3 IVF state should remain cached after rotation");
            assert!(
                Arc::ptr_eq(&cached_state, &state_after_rotation),
                "store-free IVF state should be reused across object-store generations"
            );
        }

        // Re-query through the first dataset: V3 portable state is rebound to
        // the store supplied by each reconstruction, while the cached legacy
        // index keeps reading through its original store. Either way the second
        // store must not be touched.
        let _ = store_a.io_stats_incremental();
        let _ = store_b.io_stats_incremental();
        dataset_a
            .scan()
            .nearest("vector", &query, 20)
            .unwrap()
            .minimum_nprobes(4)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let store_a_stats = store_a.io_stats_incremental();
        let store_a_index_reads = store_a_stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains(&index_path_fragment))
            .count();
        let store_b_stats = store_b.io_stats_incremental();
        let store_b_index_reads = store_b_stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains(&index_path_fragment))
            .count();
        assert!(
            store_a_index_reads > 0,
            "re-querying the first dataset should read the index through its object store: {store_a_stats:#?}"
        );
        assert_eq!(
            store_b_index_reads, 0,
            "re-querying the first dataset must not use the second object store: {store_b_stats:#?}"
        );

        // Cache keys are opaque digests, so they cannot embed credential
        // material by construction. What rotation must not do is mint new
        // entries: the same portable state, partitions, and file metadata
        // serve both object-store generations.
        let index_entries_after_rotation = dataset.session().index_cache_stats().await.num_entries;
        let metadata_entries_after_rotation =
            dataset.session().metadata_cache_stats().await.num_entries;
        assert_eq!(
            index_entries_after_rotation, index_entries_after_a,
            "credential rotation must not create new index cache entries"
        );
        assert_eq!(
            metadata_entries_after_rotation, metadata_entries_after_a,
            "credential rotation must not create new metadata cache entries"
        );

        cache_backend.set_bypass_partitions(false);
        let partition_hits_before = cache_backend.partition_hits();
        dataset_b
            .scan()
            .nearest("vector", &query, 20)
            .unwrap()
            .minimum_nprobes(4)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            cache_backend.partition_hits() > partition_hits_before,
            "the second store should reuse portable partitions populated by the first"
        );
    }

    #[tokio::test]
    async fn test_shallow_clone_ivf_rq_uses_resolved_index_directory() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let clone_uri = format!("{}/clone", test_dir.as_str());
        let (mut source, vectors) =
            generate_test_dataset::<Float32Type>(&source_uri, 0.0..1.0).await;
        append_dataset::<Float32Type>(&mut source, NUM_ROWS, 0.0..1.0).await;
        assert_eq!(source.get_fragments().len(), 2);

        let params = VectorIndexParams::ivf_rq(4, 5, DistanceType::L2);
        source
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("ivf_rq_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let query = vectors.value(0);
        let ground_truth = ground_truth(&source, "vector", &query, 20, DistanceType::L2).await;
        source
            .tags()
            .create("with_ivf_rq", source.version().version)
            .await
            .unwrap();
        let cloned = source
            .shallow_clone(&clone_uri, "with_ivf_rq", None)
            .await
            .unwrap();

        let index_meta = cloned
            .load_indices_by_name("ivf_rq_idx")
            .await
            .unwrap()
            .pop()
            .unwrap();
        assert!(
            index_meta.base_id.is_some(),
            "a shallow-cloned index should reference its source base"
        );
        assert_eq!(
            cloned.indice_files_dir(&index_meta).unwrap(),
            source.indices_dir(),
            "the cloned index should resolve its path through the source base"
        );
        assert_ne!(
            cloned.indice_files_dir(&index_meta).unwrap(),
            cloned.indices_dir(),
            "the cloned index should not use the clone's primary index directory"
        );

        let cloned = crate::DatasetBuilder::from_uri(&clone_uri)
            .with_session(Arc::new(crate::session::Session::default()))
            .load()
            .await
            .unwrap();

        let results = cloned
            .scan()
            .nearest("vector", &query, 20)
            .unwrap()
            .minimum_nprobes(4)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = results[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = row_ids.intersection(&ground_truth).count() as f32 / 20.0;
        assert_ge!(recall, 0.5);
    }
}
