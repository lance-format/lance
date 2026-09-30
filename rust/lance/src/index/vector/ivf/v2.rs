// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::marker::PhantomData;
use std::{
    any::Any,
    borrow::Cow,
    collections::{BinaryHeap, HashMap},
    ops::Range,
    sync::{
        Arc, LazyLock, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::index::vector::{IndexFileVersion, builder::index_type_string};
use crate::index::{PreFilter, vector::VectorIndex};
use arrow::compute::concat_batches;
use arrow_arith::numeric::sub;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::future::BoxFuture;
use futures::prelude::stream::{self, TryStreamExt};
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::cache::{
    CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
    CacheLease, KeyBuilder, LanceCache, WeakLanceCache,
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
use lance_index::metrics::{LocalMetricsCollector, MetricsCollector};
use lance_index::prefilter::NoFilter;
use lance_index::scalar::RowIdRemapper;
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::{blocked_ex_code_bytes, padded_query_len};
use lance_index::vector::bq::layered::{EntryColumns, SignBounds};
use lance_index::vector::bq::layered_stats;
use lance_index::vector::bq::partition_codes::PartitionCodesKey;
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
    IndexFileKey, LayeredLazyConfig, OriginLatencyClass, PlaneAccessTracker, QueryResidual,
    QueryScratch, QueryScratchCapacity, QueryScratchPool, RabitRawQueryContext, ResidentColumns,
    ResidentColumnsSetting, VectorStore, entry_columns_setting, origin_latency_setting,
    resident_columns_setting, resident_lifetime_setting, resident_store_fits, sign_bounds_setting,
};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, Index, IndexType, pb,
    vector::{
        DISTANCE_TYPE_KEY, PartitionSearchControl, PreparedPartitionSearchHandle, Query,
        VECTOR_RESULT_SCHEMA, ivf::storage::IVF_METADATA_KEY, quantizer::Quantization,
        storage::IvfQuantizationStorage, v3::subindex::IvfSubIndex,
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

mod lazy_full;

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
    /// Runtime-only counters survive reader reconstruction between queries.
    /// Its far gather pools are the ones every open of the index file
    /// shares, bound by reconstructions (see `IndexFileKey`), so a state
    /// read back from a persistent tier with a fresh tracker shares them too.
    /// The state holds no resident store: every reconstruction binds its
    /// storage to the store the index cache charges (see
    /// [`IvfOpenContext::file_cache`]), so a cached state pins nothing.
    pub(crate) plane_access: PlaneAccessTracker,
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

/// Prepared partition storage (bytes) handed to a single `spawn_cpu` dispatch on the
/// global-top-k search path.
///
/// That path scores every probed partition into one heap, so it streams prepared
/// partitions through scoring in chunks; the chunk is what bounds resident partition
/// memory (together with the prepare window) independently of `nprobes`, see
/// [`PreparedPartitionTracker`]. Chunking by bytes rather than by partition count
/// keeps both the memory bound and the dispatch overhead predictable whatever the
/// partition size: a dispatch costs two thread hops (~100µs), so on a warm cache
/// the 16-partition streaming batch would spend ~40% of a 1024-probe query on
/// dispatch overhead for ~800 KiB partitions, while a fixed large partition count
/// would pin GiBs for the multi-MiB partitions of a billion-row RQ index. 64 MiB is
/// a few milliseconds of scoring per dispatch across those sizes. Override with the
/// `LANCE_IVF_GLOBAL_TOPK_CHUNK_BYTES` environment variable.
pub(crate) const DEFAULT_GLOBAL_TOPK_CHUNK_BYTES: usize = 64 * 1024 * 1024;

pub(crate) static GLOBAL_TOPK_CHUNK_BYTES: LazyLock<usize> = LazyLock::new(|| {
    let chunk_bytes = std::env::var("LANCE_IVF_GLOBAL_TOPK_CHUNK_BYTES")
        .map(|value| {
            value
                .parse()
                .expect("failed to parse LANCE_IVF_GLOBAL_TOPK_CHUNK_BYTES")
        })
        .unwrap_or(DEFAULT_GLOBAL_TOPK_CHUNK_BYTES);
    assert!(
        chunk_bytes > 0,
        "LANCE_IVF_GLOBAL_TOPK_CHUNK_BYTES must be greater than 0, got {chunk_bytes}"
    );
    chunk_bytes
});

/// Upper bound on partitions per global-top-k scoring chunk, so that tiny partitions
/// (far below [`GLOBAL_TOPK_CHUNK_BYTES`]) still leave the resident-partition bound
/// expressible as a partition count: at most the prepare window plus two chunks.
pub(crate) const GLOBAL_TOPK_CHUNK_MAX_PARTITIONS: usize = 128;

/// Largest global-top-k heap converted to the result batch on the async task
/// instead of a `spawn_cpu` dispatch; see the use site for the rationale.
const GLOBAL_TOPK_INLINE_HEAP_LEN: usize = 4096;

/// Lazy layered full-precision scan settings from the `LANCE_RQ_LAZY_*`
/// environment, read once per process. An invalid value fails opening a
/// layered index rather than silently changing its scan; with the scan off
/// only `LANCE_RQ_LAZY_FULL` itself is read.
static LAYERED_LAZY_CONFIG: LazyLock<std::result::Result<LayeredLazyConfig, String>> =
    LazyLock::new(|| LayeredLazyConfig::from_env().map_err(|err| err.to_string()));

const IVF_PREWARM_WINDOW_SIZE_ENV: &str = "LANCE_IVF_PREWARM_WINDOW_SIZE_BYTES";
/// Default encoded-byte target of one prewarm read window.
///
/// Measured on one 1B-row IVF_RQ 1-bit segment (147 GB) read from S3 on
/// r8i.8xlarge with 192 GiB of index cache: 16 MiB windows filled the cache at
/// 500-526 MB/s regardless of how many windows were in flight (30, 128 or 512)
/// or of `LANCE_IO_THREADS` (64, 256, 1024), while 64 MiB windows reached
/// 935 MB/s with the same peak RSS (+1 GB) because the reader issues fewer,
/// larger object-store requests (3.96 MB vs 2.47 MB average).
const DEFAULT_IVF_PREWARM_WINDOW_SIZE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PartitionWindow {
    partitions: Range<usize>,
    estimated_encoded_bytes: u64,
}

#[derive(Clone, Copy)]
struct PrewarmFileLayout<'a> {
    ivf: &'a IvfModel,
    encoded_bytes: u64,
    num_rows: u64,
}

fn parse_prewarm_window_size_bytes(value: Option<&str>) -> Result<u64> {
    let Some(value) = value else {
        return Ok(DEFAULT_IVF_PREWARM_WINDOW_SIZE_BYTES);
    };
    let size = value.parse::<u64>().map_err(|error| {
        Error::invalid_input(format!(
            "{IVF_PREWARM_WINDOW_SIZE_ENV} must be a positive byte count, got {value:?}: {error}"
        ))
    })?;
    if size == 0 {
        return Err(Error::invalid_input(format!(
            "{IVF_PREWARM_WINDOW_SIZE_ENV} must be a positive byte count, got {value:?}"
        )));
    }
    Ok(size)
}

fn prewarm_window_size_bytes() -> Result<u64> {
    let value = std::env::var(IVF_PREWARM_WINDOW_SIZE_ENV).ok();
    parse_prewarm_window_size_bytes(value.as_deref())
}

fn estimate_encoded_bytes(num_data_bytes: u64, num_rows: u64, row_count: usize) -> u64 {
    if row_count == 0 || num_rows == 0 || num_data_bytes == 0 {
        return 0;
    }
    let numerator = u128::from(num_data_bytes) * row_count as u128;
    numerator
        .div_ceil(u128::from(num_rows))
        .min(u64::MAX as u128) as u64
}

/// Plan adjacent partition windows by a metadata-estimated encoded-I/O target.
///
/// The target is measured in bytes, not partition count. It is not a hard
/// decoded-memory cap: compression/page skew can make a window larger, and a
/// single oversized partition is deliberately admitted as a singleton.
fn plan_partition_windows(
    index: PrewarmFileLayout<'_>,
    storage: PrewarmFileLayout<'_>,
    target_bytes: u64,
    max_partitions: usize,
) -> Result<Vec<PartitionWindow>> {
    if target_bytes == 0 {
        return Err(Error::invalid_input(
            "IVF prewarm window target must be positive",
        ));
    }
    if index.ivf.num_partitions() != storage.ivf.num_partitions() {
        return Err(Error::index(format!(
            "IVF index has {} partitions but auxiliary storage has {}",
            index.ivf.num_partitions(),
            storage.ivf.num_partitions()
        )));
    }
    if max_partitions == 0 {
        return Err(Error::invalid_input(
            "IVF prewarm window partition cap must be positive",
        ));
    }

    let mut windows = Vec::new();
    let mut start = 0;
    let mut window_bytes = 0_u64;
    for partition_id in 0..index.ivf.num_partitions() {
        let partition_bytes = estimate_encoded_bytes(
            index.encoded_bytes,
            index.num_rows,
            index.ivf.partition_size(partition_id),
        )
        .checked_add(estimate_encoded_bytes(
            storage.encoded_bytes,
            storage.num_rows,
            storage.ivf.partition_size(partition_id),
        ))
        .ok_or_else(|| Error::index("IVF prewarm partition byte estimate overflowed u64"))?;

        let exceeds_target = window_bytes
            .checked_add(partition_bytes)
            .is_none_or(|bytes| bytes > target_bytes);
        let is_discontinuous = partition_id > start
            && (index.ivf.row_range(partition_id - 1).end
                != index.ivf.row_range(partition_id).start
                || storage.ivf.row_range(partition_id - 1).end
                    != storage.ivf.row_range(partition_id).start);
        let reached_partition_cap = partition_id - start >= max_partitions;
        if partition_id > start && (exceeds_target || is_discontinuous || reached_partition_cap) {
            windows.push(PartitionWindow {
                partitions: start..partition_id,
                estimated_encoded_bytes: window_bytes,
            });
            start = partition_id;
            window_bytes = 0;
        }
        window_bytes = window_bytes
            .checked_add(partition_bytes)
            .ok_or_else(|| Error::index("IVF prewarm window byte estimate overflowed u64"))?;
        if window_bytes > target_bytes {
            windows.push(PartitionWindow {
                partitions: start..partition_id + 1,
                estimated_encoded_bytes: window_bytes,
            });
            start = partition_id + 1;
            window_bytes = 0;
        }
    }
    if start < index.ivf.num_partitions() {
        windows.push(PartitionWindow {
            partitions: start..index.ivf.num_partitions(),
            estimated_encoded_bytes: window_bytes,
        });
    }
    Ok(windows)
}

fn split_window_batches(
    schema: &arrow_schema::SchemaRef,
    partition_lengths: &[usize],
    batches: Vec<RecordBatch>,
) -> Result<Vec<Vec<RecordBatch>>> {
    let expected_rows = partition_lengths
        .iter()
        .try_fold(0_usize, |total, length| {
            total
                .checked_add(*length)
                .ok_or_else(|| Error::index("IVF prewarm window row count overflowed usize"))
        })?;
    let actual_rows = batches.iter().try_fold(0_usize, |total, batch| {
        total
            .checked_add(batch.num_rows())
            .ok_or_else(|| Error::index("IVF prewarm decoded row count overflowed usize"))
    })?;
    if actual_rows != expected_rows {
        return Err(Error::index(format!(
            "IVF prewarm window decoded {actual_rows} rows, expected {expected_rows}"
        )));
    }

    let mut output = Vec::with_capacity(partition_lengths.len());
    let mut batch_id = 0;
    let mut batch_offset = 0;
    for &partition_length in partition_lengths {
        if partition_length == 0 {
            output.push(vec![RecordBatch::new_empty(schema.clone())]);
            continue;
        }
        let mut remaining = partition_length;
        let mut slices = Vec::new();
        while remaining > 0 {
            let batch = batches.get(batch_id).ok_or_else(|| {
                Error::index("IVF prewarm window ended before its partition boundary")
            })?;
            let available = batch.num_rows() - batch_offset;
            let slice_length = available.min(remaining);
            slices.push(batch.slice(batch_offset, slice_length));
            batch_offset += slice_length;
            remaining -= slice_length;
            if batch_offset == batch.num_rows() {
                batch_id += 1;
                batch_offset = 0;
            }
        }
        output.push(slices);
    }
    Ok(output)
}

fn compact_partition_batches(batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    let schema = batches
        .first()
        .ok_or_else(|| Error::internal("IVF prewarm partition has no decoded batches"))?
        .schema();
    if batches.len() == 1 {
        let batch = batches
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("IVF prewarm partition batch unexpectedly missing"))?;
        if batch.num_rows() == 0 {
            Ok(batch)
        } else {
            Ok(batch.shrink_to_fit()?)
        }
    } else {
        Ok(concat_batches(&schema, batches.iter())?)
    }
}

struct PartitionPrewarmBatches {
    index: Vec<RecordBatch>,
    storage: Vec<RecordBatch>,
}

async fn read_partition_window_batches(
    reader: &FileReader,
    projection: Option<&ReaderProjection>,
    schema: &arrow_schema::SchemaRef,
    ivf: &IvfModel,
    partitions: Range<usize>,
    io_stats: Option<IoStats>,
) -> Result<Vec<Vec<RecordBatch>>> {
    if partitions.is_empty() {
        return Ok(Vec::new());
    }
    let partition_lengths = if reader.num_rows() == 0 {
        vec![0; partitions.len()]
    } else {
        partitions
            .clone()
            .map(|partition_id| ivf.partition_size(partition_id))
            .collect::<Vec<_>>()
    };
    let row_start = if reader.num_rows() == 0 {
        0
    } else {
        ivf.row_range(partitions.start).start
    };
    let row_end = if reader.num_rows() == 0 {
        0
    } else {
        ivf.row_range(partitions.end - 1).end
    };
    let batches = if row_start == row_end {
        Vec::new()
    } else {
        let reader = match &io_stats {
            Some(io_stats) => Cow::Owned(reader.with_io_stats(io_stats.recorder())),
            None => Cow::Borrowed(reader),
        };
        let params = ReadBatchParams::Range(row_start..row_end);
        let stream = match projection {
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
        stream.try_collect::<Vec<_>>().await?
    };
    split_window_batches(schema, &partition_lengths, batches)
}

/// Number of prewarm windows kept in flight. Raising this beyond the CPU pool
/// size measured no throughput gain (see `DEFAULT_IVF_PREWARM_WINDOW_SIZE_BYTES`);
/// the window size is the lever.
fn prewarm_parallelism(io_parallelism: usize, cpu_parallelism: usize) -> usize {
    io_parallelism.max(1).min(cpu_parallelism.max(1))
}

struct PreparedPartitionSearch<S: IvfSubIndex, Q: Quantization> {
    query: Query,
    pre_filter: Arc<dyn PreFilter>,
    partition_id: usize,
    partition_centroid: Option<ArrayRef>,
    rq_search_cache: Option<Arc<RabitSearchCache>>,
    raw_query_context: Option<Arc<RabitRawQueryContext>>,
    part_entry: Arc<PartitionEntry<S, Q>>,
    /// Released together with `part_entry`, so the tracker's live count is the
    /// number of partitions whose storage is pinned by in-flight searches.
    _in_flight: PreparedPartitionGuard,
    _marker: PhantomData<(S, Q)>,
}

/// Live count and high-water mark of [`PreparedPartitionSearch`] values for one
/// index.
///
/// A prepared partition holds a strong reference to its [`PartitionEntry`] — the
/// partition's whole quantized storage — until it has been scored and dropped, so
/// the index cache cannot evict it in the meantime. The count is therefore the
/// partition memory a query holds *outside* the cache's budget. Every search path
/// must keep it bounded by its prepare window and scoring chunk size rather than
/// by `nprobes`: a query that probes thousands of partitions of a large RQ index
/// would otherwise pin hundreds of GiB before scoring the first one.
#[derive(Debug, Default)]
pub(crate) struct PreparedPartitionTracker {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

impl PreparedPartitionTracker {
    fn track(self: &Arc<Self>) -> PreparedPartitionGuard {
        let now = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(now, Ordering::Relaxed);
        PreparedPartitionGuard(self.clone())
    }

    /// Prepared partitions currently alive.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Most prepared partitions alive at once over the index's lifetime.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

struct PreparedPartitionGuard(Arc<PreparedPartitionTracker>);

impl Drop for PreparedPartitionGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
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
            + self.plane_access.deep_size_of_children(context)
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
        context: IvfOpenContext,
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
                plane_access: PlaneAccessTracker::default(),
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
        context: IvfOpenContext,
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
                        context,
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
                        context,
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
    cache_whole_partition: bool,
    /// Memoized size of the immutable parts (this struct, the sub-index and the
    /// quantized storage): every query that prepares this partition needs its
    /// size to budget scoring chunks, and the walk over the storage's arrays
    /// costs a few microseconds per partition — measurable on a warm
    /// probe-everything query — while these never change once loaded.
    immutable_bytes: OnceLock<usize>,
    /// Memoized size of `partition_rows`, set when the coverage is built. Kept
    /// apart from `immutable_bytes` because the coverage can appear after an
    /// earlier query already memoized the size; folding it in later keeps
    /// [`Self::size_bytes`] equal to [`DeepSizeOf::deep_size_of`].
    coverage_bytes: OnceLock<usize>,
}

impl<S: IvfSubIndex, Q: Quantization> PartitionEntry<S, Q> {
    pub(super) fn new(index: S, storage: Q::Storage) -> Self {
        Self {
            index,
            storage,
            partition_rows: OnceLock::new(),
            partition_rows_accounted: AtomicBool::new(false),
            cache_whole_partition: true,
            immutable_bytes: OnceLock::new(),
            coverage_bytes: OnceLock::new(),
        }
    }

    /// Bytes this entry pins in memory, equal to [`DeepSizeOf::deep_size_of`]
    /// but memoized: the immutable parts are computed on first use and the
    /// lazily built partition coverage is added once it exists.
    fn size_bytes(&self) -> usize {
        let immutable = *self.immutable_bytes.get_or_init(|| {
            let mut context = lance_core::deepsize::Context::new();
            std::mem::size_of::<Self>()
                + self.index.deep_size_of_children(&mut context)
                + self.storage.deep_size_of_children(&mut context)
        });
        immutable + self.coverage_bytes.get().copied().unwrap_or_default()
    }

    fn partition_rows(&self) -> Arc<RowAddrTreeMap> {
        let rows = self
            .partition_rows
            .get_or_init(|| Arc::new(self.storage.row_ids().collect()));
        self.coverage_bytes
            .get_or_init(|| rows.deep_size_of_children(&mut lance_core::deepsize::Context::new()));
        rows.clone()
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

/// What the session opening an IVF index declares about it, passed to
/// [`IVFIndex::try_new`] and to the reconstruction from a cached
/// [`IvfIndexState`]. The default declares nothing and charges no cache.
#[derive(Debug, Clone, Default)]
pub(crate) struct IvfOpenContext {
    /// The origin latency class the session declares for the IVF_RQ indexes
    /// it opens (`Session::index_origin_latency`), `None` when it declares
    /// none; see [`IVFIndex::origin_latency_at_open`].
    pub(crate) origin_latency_hint: Option<OriginLatencyClass>,
    /// The index's namespace of the index cache without a fragment reuse
    /// segment, where an IVF_RQ index with resident small columns charges
    /// and leases their store (see [`ResidentColumns::in_index_cache`]), so
    /// that a new fragment reuse index keeps the store. `None` keeps the
    /// store charged nowhere, shared with the file's other live indexes.
    pub(crate) file_cache: Option<LanceCache>,
    /// A lease on the index's resident store that the opener took before
    /// its first index-cache access (see
    /// [`lance_index::vector::storage::resident_store_preopen_lease`]), kept
    /// by an index that keeps its small columns resident and dropped by any
    /// other.
    pub(crate) resident_lease: Option<CacheLease>,
}

/// The key of the auxiliary (storage) file of index `uuid` in `index_dir`
/// on `object_store`, which IVF_RQ runtime handles, the resident store and
/// the lazy scan's far gather pools, are shared by.
pub(crate) fn aux_file_key(
    object_store: &ObjectStore,
    index_dir: &Path,
    uuid: &Uuid,
) -> IndexFileKey {
    let uuid = uuid.to_string();
    let aux_path = index_dir
        .clone()
        .join(uuid.as_str())
        .join(INDEX_AUXILIARY_FILE_NAME);
    IndexFileKey::new(&uuid, &object_store.store_prefix, aux_path.as_ref())
}

/// Warn, once per process, that `LANCE_RQ_RESIDENT_COLUMNS=on` keeps a
/// resident store of `bytes` that the index cache, admitting entries up to
/// `max_entry_bytes`, cannot hold: it stays loaded while an index of the
/// file is live, but charged nowhere and loaded again after each drop.
fn warn_resident_store_oversize(bytes: u64, max_entry_bytes: Option<u64>) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "LANCE_RQ_RESIDENT_COLUMNS=on keeps a resident store of {bytes} bytes, more than \
             the {max_entry_bytes:?} bytes the index cache admits per entry"
        );
    });
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
    prepared_partitions: Arc<PreparedPartitionTracker>,
    /// Whether the storage is a layered IVF_RQ index, fixed when opened.
    layered_rq: bool,
    /// Lazy full-precision scan settings of a layered index; see
    /// [`LayeredLazyConfig`]. Locked only to copy it once per query.
    layered_lazy: Mutex<LayeredLazyConfig>,
    /// Latency class of reads from the index's files, resolved when the
    /// index opens; the storage carries the same class.
    origin_latency: OriginLatencyClass,
    /// Block size of the object store the index's files are read from, the
    /// gap within which reads merge requested ranges unless a lazy sparse
    /// origin read sets its own; see [`Self::lazy_origin_gap_bytes`].
    origin_block_size: u64,
    /// Staging parallelism of the lazy scan when non-zero; see
    /// `set_lazy_prepare_parallelism_for_test`.
    #[cfg(test)]
    lazy_prepare_parallelism_for_test: AtomicUsize,

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
        if partition.cache_whole_partition
            && !partition.partition_rows_accounted.load(Ordering::Acquire)
        {
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
            self.load_initial_partition(partition_id, query, metrics),
            pre_filter.wait_for_ready(),
        )?;
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &part_entry, pre_filter)
                .await?;
        Ok(PreparedPartitionSearch {
            query: query.clone(),
            pre_filter,
            partition_id,
            partition_centroid: self.ivf.centroid(partition_id),
            rq_search_cache: self.rq_search_cache.clone(),
            raw_query_context,
            part_entry,
            _in_flight: self.prepared_partitions.track(),
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
        let part_entry = self
            .load_initial_partition(partition_id, query, metrics)
            .await?;
        let pre_filter =
            Self::prefilter_for_partition(&self.index_cache, partition_id, &part_entry, pre_filter)
                .await?;
        Ok(PreparedPartitionSearch {
            query: query.clone(),
            pre_filter,
            partition_id,
            partition_centroid: self.ivf.centroid(partition_id),
            rq_search_cache: self.rq_search_cache.clone(),
            raw_query_context,
            part_entry,
            _in_flight: self.prepared_partitions.track(),
            _marker: PhantomData,
        })
    }

    fn run_prepared_partition_search(
        use_query_residual: bool,
        use_residual_scratch: bool,
        prepared: PreparedPartitionSearch<S, Q>,
        metrics: &dyn MetricsCollector,
        scratch: &mut QueryScratch,
    ) -> Result<RecordBatch> {
        let PreparedPartitionSearch {
            query,
            pre_filter,
            partition_id,
            partition_centroid,
            rq_search_cache,
            raw_query_context,
            part_entry,
            _in_flight: _,
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
        Ok(batch)
    }

    /// Pull prepared partitions off `prepared` until their pinned storage reaches
    /// `chunk_bytes` or the chunk holds [`GLOBAL_TOPK_CHUNK_MAX_PARTITIONS`],
    /// always taking at least one. `None` once the stream is exhausted; a failed
    /// prepare ends the chunk (and the search) immediately.
    async fn next_scoring_chunk<St>(
        prepared: &mut St,
        chunk_bytes: usize,
    ) -> Option<Result<Vec<PreparedPartitionSearch<S, Q>>>>
    where
        St: Stream<Item = Result<PreparedPartitionSearch<S, Q>>> + Unpin,
    {
        let mut chunk = Vec::new();
        let mut bytes = 0;
        while bytes < chunk_bytes && chunk.len() < GLOBAL_TOPK_CHUNK_MAX_PARTITIONS {
            match prepared.next().await {
                Some(Ok(prepared)) => {
                    bytes += prepared.part_entry.size_bytes();
                    chunk.push(prepared);
                }
                Some(Err(err)) => return Some(Err(err)),
                None => break,
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(Ok(chunk))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_prepared_partition_search(
        use_query_residual: bool,
        use_residual_scratch: bool,
        prepared: PreparedPartitionSearch<S, Q>,
        heap: &mut BinaryHeap<OrderedNode<u64>>,
        scratch: &mut QueryScratch,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        let PreparedPartitionSearch {
            query,
            pre_filter,
            partition_id,
            partition_centroid,
            rq_search_cache,
            raw_query_context,
            part_entry,
            _in_flight: _,
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
        )
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

    fn global_heap_to_batch(heap: BinaryHeap<OrderedNode<u64>>) -> Result<RecordBatch> {
        let (row_ids, dists): (Vec<_>, Vec<_>) = heap.into_iter().map(|r| (r.id, r.dist.0)).unzip();
        Ok(RecordBatch::try_new(
            VECTOR_RESULT_SCHEMA.clone(),
            vec![
                Arc::new(Float32Array::from(dists)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )?)
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn try_new(
        object_store: Arc<ObjectStore>,
        index_dir: Path,
        uuid: Uuid,
        frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
        file_metadata_cache: &LanceCache,
        index_cache: LanceCache,
        file_sizes: HashMap<String, u64>,
        context: IvfOpenContext,
    ) -> Result<Self> {
        let io_parallelism = object_store.io_parallelism();
        let origin_latency =
            Self::origin_latency_at_open(&object_store, context.origin_latency_hint)?;
        let origin_block_size = object_store.block_size() as u64;
        let uuid_str = uuid.to_string();
        let aux_path = index_dir
            .clone()
            .join(uuid_str.as_str())
            .join(INDEX_AUXILIARY_FILE_NAME);
        let index_file = aux_file_key(&object_store, &index_dir, &uuid);
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

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
            scheduler.open_file(&aux_path, &aux_cached_size).await?,
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
        let resident_columns = Self::resident_columns_at_open(
            origin_latency,
            storage.resident_columns_bytes(),
            context.file_cache.as_ref(),
        )?;
        let resident_store =
            Self::resident_store_at_open(resident_columns, &index_file, context).await?;
        let storage = storage
            .with_origin_latency(origin_latency)
            .with_resident_columns(resident_store)
            .with_resident_columns_enabled(resident_columns)
            .with_entry_columns(Self::entry_columns_at_open()?)
            .with_index_file(index_file);

        // Cache file metadata so reconstructions from IvfIndexState can skip
        // footer reads.
        file_metadata_cache
            .with_key_prefix(uri.as_ref())
            .insert_with_key(&FileMetadataCacheKey, index_reader.metadata().clone())
            .await;
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
        let layered_rq =
            Q::quantization_type() == QuantizationType::Rabit && storage.is_layered_rq();
        let layered_lazy = Self::layered_lazy_config_at_open(layered_rq, origin_latency)?;
        let storage = storage.with_sign_bounds(Self::sign_bounds_at_open(layered_rq)?);
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
            prepared_partitions: Arc::default(),
            layered_rq,
            layered_lazy: Mutex::new(layered_lazy),
            origin_latency,
            origin_block_size,
            #[cfg(test)]
            lazy_prepare_parallelism_for_test: AtomicUsize::new(0),
            _marker: PhantomData,
        })
    }

    /// Reconstruct an IVFIndex from pre-parsed state without any I/O, over
    /// a storage already bound to its resident store.
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
        origin_latency: OriginLatencyClass,
        origin_block_size: u64,
    ) -> Result<Self> {
        let storage = storage.with_origin_latency(origin_latency);
        let scratch_pool = Arc::new(Self::query_scratch_pool(&ivf, &storage));
        let use_query_residual = Self::use_query_residual(&storage, distance_type);
        let use_residual_scratch = Self::use_residual_scratch(&ivf, use_query_residual);
        let read_projection = Self::read_projection(&reader)?;
        let layered_rq =
            Q::quantization_type() == QuantizationType::Rabit && storage.is_layered_rq();
        let layered_lazy = Self::layered_lazy_config_at_open(layered_rq, origin_latency)?;
        let storage = storage.with_sign_bounds(Self::sign_bounds_at_open(layered_rq)?);
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
            prepared_partitions: Arc::default(),
            layered_rq,
            layered_lazy: Mutex::new(layered_lazy),
            origin_latency,
            origin_block_size,
            #[cfg(test)]
            lazy_prepare_parallelism_for_test: AtomicUsize::new(0),
            _marker: PhantomData,
        })
    }

    /// The origin latency class of an index opened on `object_store` by a
    /// session that declares `hint`. Only IVF_RQ indexes read (and validate)
    /// `LANCE_RQ_ORIGIN_LATENCY` and take the hint, the setting first (see
    /// [`OriginLatencyClass::resolve_with_hint`]); any other index takes the
    /// class of its store, since no reader policy of theirs depends on it.
    fn origin_latency_at_open(
        object_store: &ObjectStore,
        hint: Option<OriginLatencyClass>,
    ) -> Result<OriginLatencyClass> {
        if Q::quantization_type() != QuantizationType::Rabit {
            return Ok(OriginLatencyClass::of_store(object_store));
        }
        Ok(OriginLatencyClass::resolve_with_hint(
            origin_latency_setting()?,
            hint,
            object_store,
        ))
    }

    /// Latency class of reads from the index's files, resolved when the index
    /// opened; see [`OriginLatencyClass`].
    pub fn origin_latency(&self) -> OriginLatencyClass {
        self.origin_latency
    }

    /// The gap in bytes within which the lazy scan's sparse reads from the
    /// index's files merge row runs into one request: the index's
    /// `LANCE_RQ_LAZY_ORIGIN_GAP_BYTES` resolved for its origin latency class
    /// and object store; see [`lance_index::vector::storage::LazyOriginGap`].
    pub fn lazy_origin_gap_bytes(&self) -> u64 {
        let origin_gap = self
            .layered_lazy
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .origin_gap;
        origin_gap.resolve(self.origin_latency, self.origin_block_size)
    }

    /// The index's lazy scan settings: `LANCE_RQ_LAZY_*` read when it
    /// opened, with the background promotion policy resolved for its origin
    /// latency class; see [`LayeredLazyConfig`].
    pub fn layered_lazy_config(&self) -> LayeredLazyConfig {
        *self
            .layered_lazy
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    }

    /// Replace the origin latency class resolved at open, and what the
    /// index resolves from it when it opens (residency and the lazy scan
    /// settings), so tests can open both classes over one store in one
    /// process.
    #[cfg(test)]
    pub(crate) fn with_origin_latency_for_test(
        mut self,
        origin_latency: OriginLatencyClass,
    ) -> Result<Self> {
        let resident_columns = Self::resident_columns_at_open(
            origin_latency,
            self.storage.resident_columns_bytes(),
            None,
        )?;
        let layered_lazy = Self::layered_lazy_config_at_open(self.layered_rq, origin_latency)?;
        self.origin_latency = origin_latency;
        self.storage = self
            .storage
            .with_origin_latency(origin_latency)
            .with_resident_columns_enabled(resident_columns);
        self.layered_lazy = Mutex::new(layered_lazy);
        Ok(self)
    }

    /// Whether an index opened with origin latency `class` keeps its small
    /// columns, a store of `bytes`, resident in `file_cache`: when the
    /// setting resolves for the class and admits the store for the cache's
    /// largest entry (see [`ResidentColumnsSetting::admits`]). Only IVF_RQ
    /// indexes read (and validate) `LANCE_RQ_RESIDENT_COLUMNS` and
    /// `LANCE_RQ_RESIDENT_LIFETIME`: the resident columns are told apart from
    /// the RaBitQ code columns, so other storages keep none.
    fn resident_columns_at_open(
        class: OriginLatencyClass,
        bytes: u64,
        file_cache: Option<&LanceCache>,
    ) -> Result<bool> {
        if Q::quantization_type() != QuantizationType::Rabit {
            return Ok(false);
        }
        resident_lifetime_setting()?;
        let setting = resident_columns_setting()?;
        if !setting.resolve(class) {
            return Ok(false);
        }
        let max_entry_bytes = file_cache.and_then(LanceCache::max_entry_bytes);
        if bytes > 0 && !resident_store_fits(bytes, max_entry_bytes) {
            layered_stats::counters().resident_columns_oversize.incr();
            if setting == ResidentColumnsSetting::On
                && max_entry_bytes.is_some_and(|max| bytes > max)
            {
                warn_resident_store_oversize(bytes, max_entry_bytes);
            }
        }
        Ok(setting.admits(bytes, max_entry_bytes))
    }

    /// The resident store handle of an index opening on index file `file`:
    /// when the index keeps its small columns `resident`, the store every
    /// live index of the file shares, charged and leased in the context's
    /// file cache with the opener's pre-open lease, so that indexes opened at
    /// once and re-opens load it once; else a store of its own, which loads
    /// only if a test switches residency on after the index opened, and the
    /// pre-open lease is dropped.
    async fn resident_store_at_open(
        resident: bool,
        file: &IndexFileKey,
        context: IvfOpenContext,
    ) -> Result<ResidentColumns> {
        if !resident {
            return Ok(ResidentColumns::default());
        }
        Ok(match &context.file_cache {
            Some(cache) => {
                ResidentColumns::in_index_cache(
                    cache,
                    file,
                    context.resident_lease,
                    resident_lifetime_setting()?,
                )
                .await
            }
            None => ResidentColumns::shared(file),
        })
    }

    /// Whether reads keep the index's small columns resident and fetch only
    /// its code and bounds columns from the file, resolved when the index
    /// opened; see `LANCE_RQ_RESIDENT_COLUMNS`.
    pub fn resident_columns_enabled(&self) -> bool {
        self.storage.resident_columns_enabled()
    }

    /// Bytes the index's resident columns take, whether or not it keeps
    /// them; see [`lance_index::vector::storage::resident_columns_bytes`].
    pub fn resident_columns_bytes(&self) -> u64 {
        self.storage.resident_columns_bytes()
    }

    /// Replace whether reads keep the small columns resident, so tests can
    /// open both in one process.
    #[cfg(test)]
    pub(crate) fn with_resident_columns_for_test(mut self, enabled: bool) -> Self {
        self.storage = self.storage.with_resident_columns_enabled(enabled);
        self
    }

    /// The entry columns an opening index asks its storage for: an IVF_RQ
    /// index reads (and validates) `LANCE_RQ_ENTRY_COLUMNS`, and its storage
    /// resolves them against residency (see
    /// [`IvfQuantizationStorage::entry_columns`]); any other index keeps
    /// full entries.
    fn entry_columns_at_open() -> Result<EntryColumns> {
        if Q::quantization_type() != QuantizationType::Rabit {
            return Ok(EntryColumns::All);
        }
        entry_columns_setting()
    }

    /// What the index's cache entries hold, resolved when it opened; see
    /// `LANCE_RQ_ENTRY_COLUMNS`. [`EntryColumns::Codes`] only for an index
    /// whose small columns are resident: a layered index's sign, high and
    /// low plane entries, or a native flat index's partition entries, which
    /// are then [`PartitionCodes`](lance_index::vector::bq::partition_codes::PartitionCodes).
    /// A native index with a graph caches whole partitions.
    pub fn entry_columns(&self) -> EntryColumns {
        if self.layered_rq || self.caches_partition_codes() {
            self.storage.entry_columns()
        } else {
            EntryColumns::All
        }
    }

    /// Whether the index caches its partitions as code-only entries: a
    /// native IVF_RQ index with a flat sub-index, which holds nothing but
    /// the storage, and code-only entries (see [`Self::entry_columns`]).
    fn caches_partition_codes(&self) -> bool {
        Q::quantization_type() == QuantizationType::Rabit
            && !self.layered_rq
            && S::name() == <FlatIndex as IvfSubIndex>::name()
            && self.storage.entry_columns() == EntryColumns::Codes
    }

    /// Replace the entry columns the index asks for, so tests can open both
    /// in one process; the storage still resolves them against residency.
    #[cfg(test)]
    pub(crate) fn with_entry_columns_for_test(mut self, entry_columns: EntryColumns) -> Self {
        self.storage = self.storage.with_entry_columns(entry_columns);
        self
    }

    /// Where a newly opened index's cache keeps the bounds columns. Only
    /// layered indexes read (and validate) `LANCE_RQ_SIGN_BOUNDS`.
    fn sign_bounds_at_open(layered_rq: bool) -> Result<SignBounds> {
        if layered_rq {
            sign_bounds_setting()
        } else {
            Ok(SignBounds::default())
        }
    }

    /// Where the index cache keeps a layered index's bounds columns,
    /// resolved when the index opened; see [`SignBounds`].
    pub fn sign_bounds(&self) -> SignBounds {
        self.storage.sign_bounds()
    }

    /// Whether the index is a layered IVF_RQ index, the only kind that reads
    /// `LANCE_RQ_SIGN_BOUNDS` and `LANCE_RQ_LAZY_*`; any other index keeps
    /// their defaults.
    pub fn is_layered_rq(&self) -> bool {
        self.layered_rq
    }

    /// Replace where the index cache keeps the bounds columns, so tests can
    /// open both placements in one process.
    #[cfg(test)]
    pub(crate) fn with_sign_bounds_for_test(mut self, sign_bounds: SignBounds) -> Self {
        self.storage = self.storage.with_sign_bounds(sign_bounds);
        self
    }

    /// Prepared-partition accounting for this index instance; see
    /// [`PreparedPartitionTracker`].
    #[cfg(test)]
    pub(crate) fn prepared_partitions(&self) -> &PreparedPartitionTracker {
        &self.prepared_partitions
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

        if self.caches_partition_codes() {
            return self
                .load_code_only_partition(partition_id, write_cache, metrics)
                .await;
        }

        let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);

        if write_cache {
            let result = self
                .index_cache
                .get_or_insert_with_key_hit(cache_key, || async {
                    info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
                    metrics.record_part_load();
                    self.load_partition_entry(partition_id, metrics.io_stats())
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
                self.load_partition_entry(partition_id, metrics.io_stats())
                    .await?,
            ))
        }
    }

    /// A native flat partition read through its code-only entry, which is
    /// cached (admitted when `write_cache`) while the partition entry built
    /// from it, with copies of the resident rows, is built again on every
    /// read; see [`IvfQuantizationStorage::load_partition_cached`].
    async fn load_code_only_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        let io_stats = metrics.io_stats();
        let (storage, hit) = self
            .storage
            .load_partition_cached(
                partition_id,
                &self.index_cache,
                write_cache,
                io_stats.clone(),
            )
            .await?;
        if hit {
            metrics.record_index_cache_hit();
        } else {
            metrics.record_index_cache_miss();
            info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
            metrics.record_part_load();
        }
        // A flat sub-index holds no rows, so building it reads nothing.
        let index = self.load_sub_index(partition_id, io_stats).await?;
        let mut entry = PartitionEntry::new(index, storage);
        entry.cache_whole_partition = false;
        Ok(Arc::new(entry))
    }

    /// Cut candidates across all probed partitions before fetching lower planes.
    pub async fn quantized_candidates(
        &self,
        query: &Query,
        partitions: &UInt32Array,
        centroid_dists: &Float32Array,
        range: Range<usize>,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<lance_index::vector::bq::layered::QuantizedCandidate>> {
        use lance_index::vector::storage::DistanceCalculatorOptions;
        let start = range.start;
        let end = range.end;
        self.initial_precision(query)?;
        if !self.storage.is_layered_rq() || !self.storage.supports_candidate_reads() {
            return Err(Error::invalid_input(
                "quantized candidates require a layered index without a row-id remapper",
            ));
        }
        if partitions.len() != centroid_dists.len() || start > end || end > partitions.len() {
            return Err(Error::invalid_input(
                "invalid quantized candidate partition range",
            ));
        }
        let factor = query
            .rq_cascade_factor
            .ok_or_else(|| Error::internal("missing cascade factor"))?;
        let k = query
            .k
            .checked_mul(query.refine_factor.unwrap_or(1) as usize)
            .ok_or_else(|| Error::invalid_input("cascade k overflow"))?;
        let limit = k
            .checked_mul(factor as usize)
            .ok_or_else(|| Error::invalid_input("cascade candidate count overflow"))?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        pre_filter.wait_for_ready().await?;
        let raw_query = self.prepare_rq_raw_query_context(&query.key)?;
        let mut coarse_heap: BinaryHeap<
            OrderedNode<(usize, u32, u64, lance_index::vector::graph::OrderedFloat)>,
        > = BinaryHeap::with_capacity(limit);
        let mut distances = HashMap::new();
        let load_parallelism = self
            .io_parallelism
            .max(1)
            .min(get_num_compute_intensive_cpus().max(1));
        // Keep the probe order for heap ties while overlapping independent
        // partition reads. The window bounds how many loaded stores stay pinned.
        let mut prepared = stream::iter(start..end)
            .map(|index| async move {
                let part_id = partitions.value(index) as usize;
                let mut local_query = query.clone();
                local_query.dist_q_c = centroid_dists.value(index);
                let entry = self
                    .load_initial_partition(part_id, &local_query, metrics)
                    .await?;
                Result::Ok((part_id, local_query, entry))
            })
            .buffered(load_parallelism);
        let mut pending = prepared.try_next().await?;
        while let Some((part_id, local_query, entry)) = pending {
            distances.insert(part_id, local_query.dist_q_c);
            let filter = pre_filter.clone();
            let raw_query = raw_query.clone();
            let cache = self.rq_search_cache.clone();
            let scoring = spawn_cpu(move || -> Result<_> {
                let mut scratch = Vec::new();
                let context = QueryResidual::RabitRawQuery {
                    rotated_centroid: rotated_partition_centroid_slice(cache.as_deref(), part_id),
                    query: raw_query.as_deref(),
                };
                let storage = entry
                    .storage
                    .as_any()
                    .downcast_ref::<lance_index::vector::bq::storage::RabitQuantizationStorage>()
                    .ok_or_else(|| Error::internal("layered candidate storage is not RaBitQ"))?;
                let calc = storage.dist_calculator_with_scratch(
                    local_query.key,
                    local_query.dist_q_c,
                    Some(context),
                    &mut scratch,
                    DistanceCalculatorOptions {
                        approx_mode: local_query.approx_mode,
                        ..Default::default()
                    },
                );
                let binary_inner_products = calc.binary_inner_products();
                for row in filter.filter_row_ids(Box::new(entry.storage.row_ids())) {
                    if coarse_heap.len() == limit
                        && calc
                            .lower_bound_with_binary_inner_product(
                                row as u32,
                                binary_inner_products[row as usize],
                            )
                            .is_some_and(|bound| {
                                coarse_heap.peek().is_some_and(|top| bound >= top.dist.0)
                            })
                    {
                        continue;
                    }
                    let node = OrderedNode::new(
                        (
                            part_id,
                            row as u32,
                            entry.storage.row_id(row as u32),
                            binary_inner_products[row as usize].into(),
                        ),
                        calc.distance_with_binary_inner_product(
                            row as u32,
                            binary_inner_products[row as usize],
                        )
                        .into(),
                    );
                    if coarse_heap.len() < limit {
                        coarse_heap.push(node);
                    } else if coarse_heap.peek().is_some_and(|top| node.dist < top.dist)
                        && let Some(mut top) = coarse_heap.peek_mut()
                    {
                        *top = node;
                    }
                }
                Ok(coarse_heap)
            });
            let (scored, next) = tokio::join!(scoring, prepared.try_next());
            coarse_heap = scored?;
            pending = next?;
        }
        Ok(coarse_heap
            .into_sorted_vec()
            .into_iter()
            .map(
                |node| lance_index::vector::bq::layered::QuantizedCandidate {
                    index_uuid: self.uuid,
                    partition_id: node.id.0,
                    row_offset: node.id.1,
                    row_id: node.id.2,
                    distance: node.dist.0,
                    centroid_distance: distances[&node.id.0],
                    binary_inner_product: node.id.3.0,
                },
            )
            .collect())
    }

    /// Rerank a global cut of quantized candidates with the original query, without data-file reads.
    pub async fn rerank_quantized_candidates(
        &self,
        query: &Query,
        input: Vec<lance_index::vector::bq::layered::QuantizedCandidate>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        use lance_index::vector::storage::DistanceCalculatorOptions;
        let k = query
            .k
            .checked_mul(query.refine_factor.unwrap_or(1) as usize)
            .ok_or_else(|| Error::invalid_input("candidate rerank k overflow"))?;
        if k == 0 {
            return Ok(RecordBatch::new_empty(VECTOR_RESULT_SCHEMA.clone()));
        }
        let raw_query = self.prepare_rq_raw_query_context(&query.key)?;
        let mut candidates: std::collections::BTreeMap<
            usize,
            Vec<lance_index::vector::bq::layered::QuantizedCandidate>,
        > = std::collections::BTreeMap::new();
        let mut distances = HashMap::new();
        for candidate in input {
            if candidate.index_uuid != self.uuid
                || candidate.partition_id >= self.ivf.num_partitions()
            {
                return Err(Error::invalid_input(
                    "quantized candidate belongs to a different index or invalid partition",
                ));
            }
            distances.insert(candidate.partition_id, candidate.centroid_distance);
            candidates
                .entry(candidate.partition_id)
                .or_default()
                .push(candidate);
        }
        let mut full_heap: BinaryHeap<OrderedNode<u64>> = BinaryHeap::with_capacity(k);
        let load_parallelism = self
            .io_parallelism
            .max(1)
            .min(get_num_compute_intensive_cpus().max(1));
        // Match the dense scan's probe order, including centroid-distance ties.
        // Statistical pruning must see the same evolving top-k threshold when
        // the candidate cut retains every row. Physical partition order is not
        // probe order. Recompute the small centroid ordering for public rerank
        // callers, whose candidate list need not retain its original ordering.
        let mut candidates: Vec<_> = candidates.into_iter().collect();
        if candidates.len() > 1 {
            let (probes, _) = self.find_partitions(query)?;
            let mut ranks = vec![usize::MAX; self.ivf.num_partitions()];
            for (rank, &partition) in probes.values().iter().enumerate() {
                ranks[partition as usize] = rank;
            }
            candidates.sort_by_key(|(part_id, _)| ranks[*part_id]);
        }
        let mut prepared = stream::iter(candidates)
            .map(|(part_id, mut rows)| async move {
                rows.sort_unstable_by_key(|row| row.row_offset);
                if rows
                    .windows(2)
                    .any(|pair| pair[0].row_offset == pair[1].row_offset)
                {
                    return Err(Error::invalid_input("duplicate quantized candidate offset"));
                }
                let offsets = rows.iter().map(|row| row.row_offset).collect();
                let storage = self
                    .storage
                    .load_candidates(part_id, offsets, &self.index_cache, metrics.io_stats())
                    .await?;
                Result::Ok((part_id, rows, storage))
            })
            .buffered(load_parallelism);
        let mut pending = prepared.try_next().await?;
        while let Some((part_id, rows, storage)) = pending {
            let dist_q_c = distances[&part_id];
            let key = query.key.clone();
            let cache = self.rq_search_cache.clone();
            let raw_query = raw_query.clone();
            let approx_mode = query.approx_mode;
            let lower_bound = query.lower_bound;
            let upper_bound = query.upper_bound;
            let scoring = spawn_cpu(move || -> Result<_> {
                let mut scratch = Vec::new();
                let context = QueryResidual::RabitRawQuery {
                    rotated_centroid: rotated_partition_centroid_slice(cache.as_deref(), part_id),
                    query: raw_query.as_deref(),
                };
                let rq_storage = storage
                    .as_any()
                    .downcast_ref::<lance_index::vector::bq::storage::RabitQuantizationStorage>()
                    .ok_or_else(|| Error::internal("layered rerank storage is not RaBitQ"))?;
                let calc = rq_storage.dist_calculator_with_scratch(
                    key,
                    dist_q_c,
                    Some(context),
                    &mut scratch,
                    DistanceCalculatorOptions {
                        approx_mode,
                        ..Default::default()
                    },
                );
                for (row, candidate) in rows.iter().enumerate() {
                    if storage.row_id(row as u32) != candidate.row_id {
                        return Err(Error::invalid_input(
                            "candidate row id does not match its physical offset",
                        ));
                    }
                    if let Some(bound) = calc.lower_bound_with_binary_inner_product(
                        row as u32,
                        candidate.binary_inner_product,
                    ) && (upper_bound.is_some_and(|upper| bound >= upper)
                        || (full_heap.len() == k
                            && full_heap.peek().is_some_and(|top| bound >= top.dist.0)))
                    {
                        continue;
                    }
                    let distance = calc.distance_with_binary_inner_product(
                        row as u32,
                        candidate.binary_inner_product,
                    );
                    if lower_bound.is_some_and(|bound| distance < bound)
                        || upper_bound.is_some_and(|bound| distance >= bound)
                    {
                        continue;
                    }
                    let node = OrderedNode::new(storage.row_id(row as u32), distance.into());
                    if full_heap.len() < k {
                        full_heap.push(node);
                    } else if full_heap.peek().is_some_and(|top| node.dist < top.dist) {
                        // Match dense top-k's replacement order for equal scores.
                        full_heap.pop();
                        full_heap.push(node);
                    }
                }
                Ok(full_heap)
            });
            let (scored, next) = tokio::join!(scoring, prepared.try_next());
            full_heap = scored?;
            pending = next?;
        }
        spawn_cpu(move || Self::global_heap_to_batch(full_heap)).await
    }

    async fn cascade_partitions(
        &self,
        query: &Query,
        partitions: &UInt32Array,
        centroid_dists: &Float32Array,
        range: Range<usize>,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let candidates = self
            .quantized_candidates(
                query,
                partitions,
                centroid_dists,
                range,
                pre_filter,
                metrics,
            )
            .await?;
        self.rerank_quantized_candidates(query, candidates, metrics)
            .await
    }

    async fn load_initial_partition(
        &self,
        partition_id: usize,
        query: &Query,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        use lance_index::vector::bq::layered::RQPrecision;
        let mut precision = self.initial_precision(query)?;
        if query.rq_cascade_factor.is_some()
            && precision == RQPrecision::High
            && self.storage.supports_candidate_reads()
            && self
                .index_cache
                .get_resident_with_key(&self.storage.plane_key(partition_id, 1))
                .await
                .is_none()
        {
            precision = RQPrecision::Sign;
        }
        self.load_query_partition(partition_id, precision, metrics)
            .await
    }

    fn initial_precision(
        &self,
        query: &Query,
    ) -> Result<lance_index::vector::bq::layered::RQPrecision> {
        use lance_index::vector::bq::layered::RQPrecision;
        if let Some(factor) = query.rq_cascade_factor {
            if factor == 0
                || query.rq_precision != RQPrecision::Full
                || !self.storage.is_layered_rq()
                || query.approx_mode == lance_index::vector::ApproxMode::Fast
            {
                return Err(Error::invalid_input(
                    "rq_cascade_factor requires a positive factor, full precision and normal/accurate mode on a layered IVF_RQ index",
                ));
            }
            if self.storage.supports_candidate_reads()
                && query.lower_bound.is_none()
                && query.upper_bound.is_none()
            {
                return Ok(RQPrecision::High);
            }
        }
        Ok(query.rq_precision)
    }

    async fn load_query_partition(
        &self,
        partition_id: usize,
        precision: lance_index::vector::bq::layered::RQPrecision,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        if !self.storage.is_layered_rq() {
            if precision != lance_index::vector::bq::layered::RQPrecision::Full {
                return Err(Error::invalid_input(
                    "rq_precision requires a layered IVF_RQ index",
                ));
            }
            return self.load_partition(partition_id, true, metrics).await;
        }
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::invalid_input("partition id out of range"));
        }
        let mut entry = self
            .load_partition_entry_at_precision(partition_id, metrics.io_stats(), Some(precision))
            .await?;
        entry.cache_whole_partition = false;
        Ok(Arc::new(entry))
    }

    async fn load_partition_entry(
        &self,
        partition_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<PartitionEntry<S, Q>> {
        self.load_partition_entry_at_precision(partition_id, io_stats, None)
            .await
    }

    async fn load_partition_entry_at_precision(
        &self,
        partition_id: usize,
        io_stats: Option<IoStats>,
        precision: Option<lance_index::vector::bq::layered::RQPrecision>,
    ) -> Result<PartitionEntry<S, Q>> {
        let idx = self.load_sub_index(partition_id, io_stats.clone()).await?;
        let storage = if let Some(precision) = precision {
            Box::pin(self.storage.load_partition_at_precision(
                partition_id,
                precision,
                &self.index_cache,
                io_stats,
            ))
            .await?
        } else {
            self.load_partition_storage(partition_id, io_stats).await?
        };
        Ok(PartitionEntry::new(idx, storage))
    }

    async fn load_sub_index(&self, partition_id: usize, io_stats: Option<IoStats>) -> Result<S> {
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
        S::load(batch)
    }

    async fn materialize_prewarm_partition(
        &self,
        partition_id: usize,
        batches: PartitionPrewarmBatches,
    ) -> Result<PartitionEntry<S, Q>>
    where
        Q::Metadata: 'static,
        Q::Storage: 'static,
    {
        let sub_index_metadata = self.sub_index_metadata[partition_id].clone();
        if batches.index.iter().all(|batch| batch.num_rows() == 0) {
            // IVF-Flat/RQ stores no per-partition sub-index rows. Keep this
            // trivial construction inline instead of dispatching one CPU task
            // per partition (hundreds of thousands on large indexes).
            let batch = compact_partition_batches(batches.index)?
                .add_metadata(S::metadata_key().to_owned(), sub_index_metadata)?;
            let index = S::load(batch)?;
            let storage = self
                .storage
                .materialize_partition_for_prewarm(batches.storage)
                .await?;
            return Ok(PartitionEntry::new(index, storage));
        }
        let index = spawn_cpu(move || {
            let batch = compact_partition_batches(batches.index)?
                .add_metadata(S::metadata_key().to_owned(), sub_index_metadata)?;
            S::load(batch)
        });
        let storage = self
            .storage
            .materialize_partition_for_prewarm(batches.storage);
        let (index, storage) = tokio::try_join!(index, storage)?;
        Ok(PartitionEntry::new(index, storage))
    }

    async fn prewarm_partition_window(&self, partitions: Range<usize>) -> Result<()>
    where
        Q::Metadata: 'static,
        Q::Storage: 'static,
    {
        if self.caches_partition_codes() {
            return self.prewarm_partition_codes_window(partitions).await;
        }
        let index_schema = Arc::new(match &self.read_projection {
            Some(projection) => projection.schema.as_ref().into(),
            None => self.reader.schema().as_ref().into(),
        });
        let storage_schema = Arc::new(self.storage.reader().schema().as_ref().into());
        let mut partition_id = partitions.start;
        while partition_id < partitions.end {
            let leader_key = IVFPartitionKey::<S, Q>::new(partition_id);
            if self.index_cache.get_with_key(&leader_key).await.is_some() {
                partition_id += 1;
                continue;
            }

            // Stop at a cached hole. This prevents a warm partition from being
            // included in a larger contiguous read merely because later
            // partitions are cold.
            let mut run_end = partition_id + 1;
            while run_end < partitions.end {
                let key = IVFPartitionKey::<S, Q>::new(run_end);
                if self.index_cache.get_with_key(&key).await.is_some() {
                    break;
                }
                run_end += 1;
            }
            let run = partition_id..run_end;
            let (_, was_cached) = self
                .index_cache
                .get_or_insert_with_key_hit(leader_key, || async {
                    let (index_batches, storage_batches) = tokio::try_join!(
                        read_partition_window_batches(
                            &self.reader,
                            self.read_projection.as_ref(),
                            &index_schema,
                            &self.ivf,
                            run.clone(),
                            None,
                        ),
                        read_partition_window_batches(
                            self.storage.reader(),
                            None,
                            &storage_schema,
                            self.storage.ivf(),
                            run.clone(),
                            None,
                        )
                    )?;
                    if index_batches.len() != run.len() || storage_batches.len() != run.len() {
                        return Err(Error::internal(format!(
                            "IVF prewarm run {:?} produced {} index and {} storage partitions",
                            run,
                            index_batches.len(),
                            storage_batches.len()
                        )));
                    }

                    let mut payloads = index_batches
                        .into_iter()
                        .zip(storage_batches)
                        .map(|(index, storage)| PartitionPrewarmBatches { index, storage });
                    let leader_batches = payloads.next().ok_or_else(|| {
                        Error::internal(format!(
                            "IVF prewarm run {:?} did not produce its leader partition",
                            run
                        ))
                    })?;
                    let mut follower_loads = FuturesUnordered::new();
                    for (offset, batches) in payloads.enumerate() {
                        let follower_id = run.start + offset + 1;
                        follower_loads.push(async move {
                            let key = IVFPartitionKey::<S, Q>::new(follower_id);
                            self.index_cache
                                .get_or_insert_with_key(key, || async move {
                                    self.materialize_prewarm_partition(follower_id, batches)
                                        .await
                                })
                                .await
                                .map(|_| ())
                        });
                    }
                    let mut first_error = None;
                    while let Some(result) = follower_loads.next().await {
                        if let Err(error) = result
                            && first_error.is_none()
                        {
                            first_error = Some(error);
                        }
                    }
                    if let Some(error) = first_error {
                        return Err(error);
                    }
                    self.materialize_prewarm_partition(partition_id, leader_batches)
                        .await
                })
                .await?;
            partition_id = if was_cached {
                // A query or another prewarm owned this key. It may not have
                // populated the remainder of our planned window, so resume at
                // the next partition instead of skipping the full run.
                partition_id + 1
            } else {
                run_end
            };
        }
        Ok(())
    }

    /// [`Self::prewarm_partition_window`] of an index that caches its
    /// partitions as code-only entries: one read of the window's code
    /// columns, cut into the entries of its partitions. The flat sub-index
    /// holds nothing to warm, and the resident store loads on first use.
    async fn prewarm_partition_codes_window(&self, partitions: Range<usize>) -> Result<()> {
        let projection = self.storage.partition_codes_projection()?;
        let schema: arrow_schema::SchemaRef = Arc::new(projection.schema.as_ref().into());
        let mut partition_id = partitions.start;
        while partition_id < partitions.end {
            let cached = |partition| async move {
                let key = PartitionCodesKey { partition };
                self.index_cache.get_with_key(&key).await.is_some()
            };
            if cached(partition_id).await {
                partition_id += 1;
                continue;
            }
            // Stop at a cached hole, as the whole-partition prewarm does.
            let mut run_end = partition_id + 1;
            while run_end < partitions.end && !cached(run_end).await {
                run_end += 1;
            }
            let run = partition_id..run_end;
            let batches = read_partition_window_batches(
                self.storage.reader(),
                Some(&projection),
                &schema,
                self.storage.ivf(),
                run.clone(),
                None,
            )
            .await?;
            if batches.len() != run.len() {
                return Err(Error::internal(format!(
                    "IVF prewarm run {run:?} produced {} partitions of codes",
                    batches.len()
                )));
            }
            for (partition, batches) in run.zip(batches) {
                let codes = self.storage.partition_codes_from_batches(batches)?;
                self.index_cache
                    .get_or_insert_with_key(
                        PartitionCodesKey { partition },
                        || async move { Ok(codes) },
                    )
                    .await?;
            }
            partition_id = run_end;
        }
        Ok(())
    }

    pub async fn load_partition_storage(
        &self,
        partition_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<Q::Storage> {
        self.storage.load_partition(partition_id, io_stats).await
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
            plane_access: self.storage.plane_access_tracker().clone(),
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
        if self.storage.is_layered_rq() {
            return self.storage.prewarm_planes(&self.index_cache).await;
        }
        let cpu_parallelism = get_num_compute_intensive_cpus();
        let target_bytes = prewarm_window_size_bytes()?;
        let parallelism = prewarm_parallelism(self.io_parallelism, cpu_parallelism);
        let max_partitions = cpu_parallelism.saturating_mul(2).max(1);
        let windows = plan_partition_windows(
            PrewarmFileLayout {
                ivf: &self.ivf,
                encoded_bytes: self.reader.metadata().num_data_bytes,
                num_rows: self.reader.num_rows(),
            },
            PrewarmFileLayout {
                ivf: self.storage.ivf(),
                encoded_bytes: self.storage.reader().metadata().num_data_bytes,
                num_rows: self.storage.reader().num_rows(),
            },
            target_bytes,
            max_partitions,
        )?;
        let planned_bytes: u64 = windows.iter().map(|w| w.estimated_encoded_bytes).sum();
        info!(
            uuid = %self.uuid,
            windows = windows.len(),
            planned_bytes,
            window_bytes = target_bytes,
            parallelism,
            io_parallelism = self.io_parallelism,
            "prewarming IVF partitions in byte windows"
        );
        let started = std::time::Instant::now();
        stream::iter(windows)
            .map(Ok)
            .try_for_each_concurrent(Some(parallelism), |window| async move {
                self.prewarm_partition_window(window.partitions).await
            })
            .await?;
        let elapsed = started.elapsed();
        info!(
            uuid = %self.uuid,
            elapsed_ms = elapsed.as_millis() as u64,
            planned_mb_per_s = planned_bytes as f64 / 1e6 / elapsed.as_secs_f64().max(1e-9),
            "prewarmed IVF partitions"
        );
        Ok(())
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

    #[instrument(level = "debug", skip(self, pre_filter, metrics))]
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        self.initial_precision(query)?;
        if query.rq_cascade_factor.is_some()
            && self.storage.supports_candidate_reads()
            && query.lower_bound.is_none()
            && query.upper_bound.is_none()
        {
            return self
                .cascade_partitions(
                    query,
                    &UInt32Array::from(vec![partition_id as u32]),
                    &Float32Array::from(vec![query.dist_q_c]),
                    0..1,
                    pre_filter,
                    metrics,
                )
                .await;
        }
        let part_entry = self
            .load_initial_partition(partition_id, query, metrics)
            .await?;
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
        let (batch, local_metrics) = spawn_cpu(move || {
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
            Result::Ok((batch, local_metrics))
        })
        .await?;

        local_metrics.dump_into(metrics);

        Ok(batch)
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

    fn search_prepared_partition(
        &self,
        prepared: PreparedPartitionSearchHandle,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let prepared = prepared
            .downcast::<PreparedPartitionSearch<S, Q>>()
            .map_err(|_| Error::internal("failed to downcast prepared partition search"))?;
        self.scratch_pool.with_scratch(|scratch| {
            Self::run_prepared_partition_search(
                self.use_query_residual,
                self.use_residual_scratch,
                *prepared,
                metrics,
                scratch,
            )
        })
    }

    fn supports_prepared_partition_search(&self) -> bool {
        true
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

        self.initial_precision(&query)?;
        if query.rq_cascade_factor.is_some()
            && self.storage.supports_candidate_reads()
            && query.lower_bound.is_none()
            && query.upper_bound.is_none()
        {
            let batch = self
                .cascade_partitions(
                    &query,
                    &partitions,
                    &q_c_dists,
                    start_idx..end_idx,
                    pre_filter,
                    metrics.as_ref(),
                )
                .await?;
            if let Some(control) = control {
                control.record_batch(&batch);
            }
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
                stream::once(async { Ok(batch) }),
            )));
        }

        let prepare_parallelism = get_num_compute_intensive_cpus().max(1);
        let raw_query_context = self.prepare_rq_raw_query_context(&query.key)?;

        if control.is_none() && S::supports_global_topk_heap() {
            if let Some(config) = self.lazy_full_config(&query, raw_query_context.is_some()) {
                if !self
                    .all_ex_planes_resident(&partitions, start_idx..end_idx)
                    .await
                {
                    let batch = Box::pin(self.clone().search_partitions_lazy_full(
                        query,
                        partitions,
                        q_c_dists,
                        start_idx..end_idx,
                        pre_filter,
                        metrics,
                        raw_query_context,
                        config,
                    ))
                    .await?;
                    return Ok(Box::pin(RecordBatchStreamAdapter::new(
                        VECTOR_RESULT_SCHEMA.clone(),
                        stream::once(async move { Ok(batch) }),
                    )));
                }
                layered_stats::counters().lazy_all_resident_skips.incr();
            }
            let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
            pre_filter.wait_for_ready().await?;
            let prepare_index = self.clone();
            let prepare_metrics = metrics.clone();
            let prepare_raw_query_context = raw_query_context.clone();
            // Stream prepared partitions through scoring in chunks rather than
            // collecting all of them first. A prepared partition pins its whole
            // quantized storage, so collecting `nprobes` of them before scoring
            // makes peak memory scale with `nprobes` (a 4096-probe query over a
            // large RQ index pinned hundreds of GiB per index segment). Chunking
            // bounds resident partitions to the prepare window plus two chunks:
            // one being assembled by `next_scoring_chunk` while another is scored.
            // `buffered` preserves the probe order, so the heap accumulates
            // partitions in the same order as before (which decides which of
            // several rows tied at the k-th distance the capped heap keeps).
            let mut prepared = stream::iter(start_idx..end_idx)
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
                .fuse();
            let chunk_bytes = *GLOBAL_TOPK_CHUNK_BYTES;

            let use_query_residual = self.use_query_residual;
            let use_residual_scratch = self.use_residual_scratch;
            let mut heap = BinaryHeap::with_capacity(heap_capacity);
            // Score each chunk on the CPU pool while the next chunk prepares (the
            // same overlap `search_partitions_batch` uses): `spawn_cpu` starts the
            // scoring immediately and the async task, never a CPU-pool thread, does
            // the waiting (#7642). The heap is threaded through each dispatch so
            // scoring stays sequential, and a scored chunk is dropped before the
            // next one is scored.
            let mut pending = Self::next_scoring_chunk(&mut prepared, chunk_bytes).await;
            while let Some(chunk) = pending {
                let chunk = chunk?;
                let search_metrics = metrics.clone();
                let scratch_pool = self.scratch_pool.clone();
                let score = spawn_cpu(move || -> Result<BinaryHeap<OrderedNode<u64>>> {
                    scratch_pool.with_scratch(|scratch| -> Result<()> {
                        for prepared in chunk {
                            Self::accumulate_prepared_partition_search(
                                use_query_residual,
                                use_residual_scratch,
                                prepared,
                                &mut heap,
                                scratch,
                                search_metrics.as_ref(),
                            )?;
                        }
                        Ok(())
                    })?;
                    Ok(heap)
                });
                let (scored, next) =
                    futures::join!(score, Self::next_scoring_chunk(&mut prepared, chunk_bytes));
                heap = scored?;
                pending = next;
            }

            // Turning the heap into the result batch is a sort of `k * refine_factor`
            // entries. Below a few thousand that is well under the ~100µs where a
            // `spawn_cpu` dispatch pays for itself, so do it inline and keep the
            // common small-k query at one dispatch (as before the chunked scoring).
            let batch = if heap.len() <= GLOBAL_TOPK_INLINE_HEAP_LEN {
                Self::global_heap_to_batch(heap)?
            } else {
                spawn_cpu(move || Self::global_heap_to_batch(heap)).await?
            };

            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                VECTOR_RESULT_SCHEMA.clone(),
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
                        let mut outputs: Vec<DataFusionResult<RecordBatch>> =
                            Vec::with_capacity(prepared_batch.len());
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
                                match Self::run_prepared_partition_search(
                                    use_query_residual,
                                    use_residual_scratch,
                                    prepared,
                                    search_metrics.as_ref(),
                                    scratch,
                                )
                                .map_err(DataFusionError::from)
                                {
                                    Ok(batch) => {
                                        if let Some(control) = search_control.as_ref() {
                                            control.record_batch(&batch);
                                        }
                                        outputs.push(Ok(batch));
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
            VECTOR_RESULT_SCHEMA.clone(),
            ReceiverStream::new(batch_rx),
        )))
    }

    fn supports_batch_partition_search(&self) -> bool {
        S::supports_global_topk_heap()
    }

    async fn search_partitions_batch(
        self: Arc<Self>,
        query: Query,
        partitions_per_query: Vec<Arc<UInt32Array>>,
        q_c_dists_per_query: Vec<Arc<Float32Array>>,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Result<Vec<RecordBatch>> {
        if !S::supports_global_topk_heap() {
            return Err(Error::not_supported(
                "batch partition search requires a global top-k heap sub-index",
            ));
        }
        let query_count = partitions_per_query.len();
        if q_c_dists_per_query.len() != query_count {
            return Err(Error::invalid_input(format!(
                "batch partition search: {query_count} query partition lists but {} distance lists",
                q_c_dists_per_query.len()
            )));
        }
        if query_count == 0 {
            return Ok(Vec::new());
        }
        if !query.key.len().is_multiple_of(query_count) {
            return Err(Error::invalid_input(format!(
                "batch partition search: query key length {} is not divisible by query count {query_count}",
                query.key.len()
            )));
        }
        let dim = query.key.len() / query_count;
        self.initial_precision(&query)?;
        if query.rq_cascade_factor.is_some()
            && self.storage.supports_candidate_reads()
            && query.lower_bound.is_none()
            && query.upper_bound.is_none()
        {
            let mut results = Vec::with_capacity(query_count);
            for query_idx in 0..query_count {
                let mut single = query.clone();
                single.key = query.key.slice(query_idx * dim, dim);
                results.push(
                    self.cascade_partitions(
                        &single,
                        &partitions_per_query[query_idx],
                        &q_c_dists_per_query[query_idx],
                        0..partitions_per_query[query_idx].len(),
                        pre_filter.clone(),
                        metrics.as_ref(),
                    )
                    .await?,
                );
            }
            return Ok(results);
        }

        // Per-query immutable search state: the query vector slice and the
        // optional Rabit raw-query context both depend only on the query vector,
        // so compute them once up front rather than per probed partition.
        let mut base_queries = Vec::with_capacity(query_count);
        let mut raw_query_contexts = Vec::with_capacity(query_count);
        for query_index in 0..query_count {
            if partitions_per_query[query_index].len() != q_c_dists_per_query[query_index].len() {
                return Err(Error::invalid_input(format!(
                    "batch partition search: query {query_index} has {} partitions but {} distances",
                    partitions_per_query[query_index].len(),
                    q_c_dists_per_query[query_index].len()
                )));
            }
            let mut single_query = query.clone();
            single_query.key = query.key.slice(query_index * dim, dim);
            raw_query_contexts.push(self.prepare_rq_raw_query_context(&single_query.key)?);
            base_queries.push(single_query);
        }
        // Shared across every chunk's scoring dispatch below, so wrap once in an
        // `Arc` instead of cloning the whole `Vec` per chunk.
        let base_queries = Arc::new(base_queries);
        let raw_query_contexts = Arc::new(raw_query_contexts);

        // Invert the per-query partition lists so each distinct partition is
        // loaded once and scored against every query that probes it.
        let mut assignments: HashMap<u32, Vec<(usize, f32)>> = HashMap::new();
        for (query_index, (parts, dists)) in partitions_per_query
            .iter()
            .zip(q_c_dists_per_query.iter())
            .enumerate()
        {
            for (part_id, dist_q_c) in parts.values().iter().zip(dists.values().iter()) {
                assignments
                    .entry(*part_id)
                    .or_default()
                    .push((query_index, *dist_q_c));
            }
        }

        pre_filter.wait_for_ready().await?;

        // Score partitions in a deterministic order. `assignments` is a HashMap,
        // so its iteration order (and hence the order partitions accumulate into
        // each per-query heap) is otherwise arbitrary. When several rows tie at
        // the k-th distance, which one the capped heap keeps depends on insertion
        // order, so a stable partition order is what makes the selected top-k
        // deterministic across runs. (Any of the tied rows is an equally valid
        // k-th neighbor, so this does not affect recall.)
        let mut assignment_list: Vec<(u32, Vec<(usize, f32)>)> = assignments.into_iter().collect();
        assignment_list.sort_by_key(|(part_id, _)| *part_id);

        // Load each distinct partition's storage exactly once (the shared I/O
        // that batch search exists to save), but *stream* the loaded partitions
        // through scoring in chunks rather than materializing them all. A wide
        // batch probes up to `min(query_count * nprobes, num_partitions)` distinct
        // partitions, so collecting every loaded partition before scoring would
        // make peak memory scale with the batch width — up to the whole index.
        // Streaming bounds resident partition storage to the load window plus one
        // chunk. `buffered` preserves the sorted load order above, so scoring order
        // (and thus the k-th-distance tie-break) stays deterministic.
        let load_parallelism = get_num_compute_intensive_cpus().max(1);
        let load_index = self.clone();
        let load_metrics = metrics.clone();
        let precision = query.rq_precision;
        let mut loaded_chunks = stream::iter(assignment_list)
            .map(move |(part_id, probing_queries)| {
                let index = load_index.clone();
                let metrics = load_metrics.clone();
                async move {
                    let part_entry = index
                        .load_query_partition(part_id as usize, precision, metrics.as_ref())
                        .await?;
                    Result::Ok((part_id as usize, part_entry, probing_queries))
                }
            })
            .buffered(load_parallelism)
            .chunks(*STREAMING_SEARCH_BATCH_SIZE);

        let use_query_residual = self.use_query_residual;
        let use_residual_scratch = self.use_residual_scratch;
        let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
        let mut heaps: Vec<BinaryHeap<OrderedNode<u64>>> = (0..query_count)
            .map(|_| BinaryHeap::with_capacity(heap_capacity))
            .collect();

        // Score each chunk on the CPU pool while the next chunk loads. `spawn_cpu`
        // dispatches the scoring immediately and only touches CPU-bound state, so
        // `join!`-ing it with the next `loaded_chunks` pull keeps partition I/O in
        // flight during scoring: the async task, never a CPU-pool thread, does the
        // waiting (#7642), and the load stream is not paused (the pairing `spawn_cpu`'s
        // docs recommend with `buffered`). Scoring stays sequential across chunks —
        // each mutates the same per-query heaps — so a step costs about
        // max(load, score) rather than their sum, and a scored chunk's storage is
        // dropped before the next is scored, keeping peak memory bounded.
        let mut pending = loaded_chunks.next().await;
        while let Some(chunk) = pending {
            let chunk = chunk.into_iter().collect::<Result<Vec<_>>>()?;
            let index = self.clone();
            let pre_filter = pre_filter.clone();
            let base_queries = base_queries.clone();
            let raw_query_contexts = raw_query_contexts.clone();
            let scratch_pool = self.scratch_pool.clone();
            let search_metrics = metrics.clone();
            let score = spawn_cpu(move || -> Result<Vec<BinaryHeap<OrderedNode<u64>>>> {
                scratch_pool.with_scratch(|scratch| -> Result<()> {
                    for (part_id, part_entry, probing_queries) in &chunk {
                        let partition_centroid = index.ivf.centroid(*part_id);
                        for (query_index, dist_q_c) in probing_queries {
                            let mut single_query = base_queries[*query_index].clone();
                            single_query.dist_q_c = *dist_q_c;
                            let prepared = PreparedPartitionSearch::<S, Q> {
                                query: single_query,
                                pre_filter: pre_filter.clone(),
                                partition_id: *part_id,
                                partition_centroid: partition_centroid.clone(),
                                rq_search_cache: index.rq_search_cache.clone(),
                                raw_query_context: raw_query_contexts[*query_index].clone(),
                                part_entry: part_entry.clone(),
                                _in_flight: index.prepared_partitions.track(),
                                _marker: PhantomData,
                            };
                            Self::accumulate_prepared_partition_search(
                                use_query_residual,
                                use_residual_scratch,
                                prepared,
                                &mut heaps[*query_index],
                                scratch,
                                search_metrics.as_ref(),
                            )?;
                        }
                    }
                    Ok(())
                })?;
                Ok(heaps)
            });
            // Load the next chunk while this one is scored on the CPU pool.
            let (scored, next) = futures::join!(score, loaded_chunks.next());
            heaps = scored?;
            pending = next;
        }

        heaps
            .into_iter()
            .map(Self::global_heap_to_batch)
            .collect::<Result<Vec<_>>>()
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
    context: IvfOpenContext,
) -> Result<Arc<dyn VectorIndex>> {
    let io_parallelism = object_store.io_parallelism();
    let origin_latency =
        IVFIndex::<S, Q>::origin_latency_at_open(&object_store, context.origin_latency_hint)?;
    let origin_block_size = object_store.block_size() as u64;

    let index_path = Path::parse(&state.index_file_path)
        .map_err(|e| Error::io(format!("invalid index path: {e}")))?;

    // Derive aux path from the index path's parent directory.
    let mut parts: Vec<_> = index_path.parts().collect();
    parts.pop();
    let dir: Path = parts.into_iter().collect();
    let aux_path = dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
    let index_file = IndexFileKey::new(&state.uuid, &object_store.store_prefix, aux_path.as_ref());

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
    )
    .with_plane_access_tracker(state.plane_access.clone());
    // Bind the store as an open does: the state holds none, so a state read
    // back from a persistent tier binds to the one the index cache holds or
    // the file's live indexes share, and any state pins nothing.
    let resident = IVFIndex::<S, Q>::resident_columns_at_open(
        origin_latency,
        storage.resident_columns_bytes(),
        context.file_cache.as_ref(),
    )?;
    let resident_columns =
        IVFIndex::<S, Q>::resident_store_at_open(resident, &index_file, context).await?;
    let storage = storage
        .with_resident_columns(resident_columns)
        .with_resident_columns_enabled(resident)
        .with_entry_columns(IVFIndex::<S, Q>::entry_columns_at_open()?)
        .with_index_file(index_file);
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
        origin_latency,
        origin_block_size,
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
    use lance_index::vector::bq::builder::RabitQuantizer;
    use lance_index::vector::bq::{
        RQBuildParams, RQRotationType,
        ex_dot::{blocked_ex_code_bytes, padded_query_len},
        storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RabitQuantizationMetadata, RabitQueryEstimator},
        transform::{EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN},
    };
    use lance_index::vector::ivf::storage::IvfModel;
    use lance_index::vector::storage::{DistCalculator, IvfQuantizationStorage, VectorStore};
    use lance_index::vector::v3::subindex::IvfSubIndex;

    use crate::dataset::{InsertBuilder, UpdateBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::DatasetIndexInternalExt;
    use crate::index::vector::ivf::v2::{
        IVFPartitionKey, IvfFlatIndex, IvfHnswSqIndex, IvfOpenContext, IvfPq, IvfStateEntryBox,
        PartitionEntry,
    };
    use crate::utils::test::copy_test_data_to_tmp;
    use crate::{
        Dataset,
        index::vector::{VectorIndex, VectorIndexParams},
    };
    use crate::{
        dataset::optimize::{CompactionOptions, compact_files},
        index::vector::IndexFileVersion,
    };
    use futures::TryStreamExt;
    use lance_core::cache::{CacheBackend, CacheCodecImpl, LanceCache, WeakLanceCache};
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::utils::tokio::get_num_compute_intensive_cpus;
    use lance_core::{ROW_ID, Result};
    use lance_datagen::{Dimension, RowCount, Seed, array, gen_batch};
    use lance_encoding::decoder::DecoderPlugins;
    use lance_file::reader::{FileReader, FileReaderOptions};
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::prefilter::{NoFilter, PreFilter};
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
    use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, Query};
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

    #[test]
    fn test_prewarm_parallelism_is_bounded_by_io_and_cpu() {
        assert_eq!(super::prewarm_parallelism(8, 4), 4);
        assert_eq!(super::prewarm_parallelism(2, 4), 2);
        assert_eq!(super::prewarm_parallelism(0, 0), 1);
    }

    #[test]
    fn test_prewarm_window_size_config() {
        assert_eq!(
            super::parse_prewarm_window_size_bytes(None).unwrap(),
            64 * 1024 * 1024
        );
        assert_eq!(
            super::parse_prewarm_window_size_bytes(Some("1048576")).unwrap(),
            1_048_576
        );
        for value in ["0", "not-a-byte-count"] {
            let error = super::parse_prewarm_window_size_bytes(Some(value)).unwrap_err();
            assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
            let message = error.to_string();
            assert!(message.contains("LANCE_IVF_PREWARM_WINDOW_SIZE_BYTES"));
            assert!(message.contains(value));
        }
    }

    fn ivf_with_lengths(lengths: &[u32]) -> IvfModel {
        let mut ivf = IvfModel::empty();
        for &length in lengths {
            ivf.add_partition(length);
        }
        ivf
    }

    fn prewarm_layout(
        ivf: &IvfModel,
        encoded_bytes: u64,
        num_rows: u64,
    ) -> super::PrewarmFileLayout<'_> {
        super::PrewarmFileLayout {
            ivf,
            encoded_bytes,
            num_rows,
        }
    }

    #[test]
    fn test_plan_prewarm_windows_uses_combined_encoded_bytes() {
        let ivf = ivf_with_lengths(&[2, 0, 4, 20, 0, 2]);
        let windows = super::plan_partition_windows(
            prewarm_layout(&ivf, 112, 28),
            prewarm_layout(&ivf, 168, 28),
            50,
            100,
        )
        .unwrap();

        assert_eq!(
            windows,
            vec![
                super::PartitionWindow {
                    partitions: 0..2,
                    estimated_encoded_bytes: 20,
                },
                super::PartitionWindow {
                    partitions: 2..3,
                    estimated_encoded_bytes: 40,
                },
                super::PartitionWindow {
                    partitions: 3..4,
                    estimated_encoded_bytes: 200,
                },
                super::PartitionWindow {
                    partitions: 4..6,
                    estimated_encoded_bytes: 20,
                },
            ]
        );
        assert_eq!(windows.first().unwrap().partitions.start, 0);
        assert_eq!(windows.last().unwrap().partitions.end, ivf.num_partitions());
        for pair in windows.windows(2) {
            assert_eq!(pair[0].partitions.end, pair[1].partitions.start);
        }
    }

    #[test]
    fn test_plan_prewarm_windows_caps_empty_partitions_and_splits_gaps() {
        let empty_ivf = ivf_with_lengths(&[0, 0, 0, 0, 0]);
        let windows = super::plan_partition_windows(
            prewarm_layout(&empty_ivf, 0, 0),
            prewarm_layout(&empty_ivf, 0, 0),
            1024,
            2,
        )
        .unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| window.partitions.clone())
                .collect::<Vec<_>>(),
            vec![0..2, 2..4, 4..5]
        );

        let index_ivf = ivf_with_lengths(&[2, 2, 2]);
        let mut storage_ivf = IvfModel::empty();
        storage_ivf.add_partition_with_offset(0, 2);
        storage_ivf.add_partition_with_offset(20, 2);
        storage_ivf.add_partition_with_offset(22, 2);
        let windows = super::plan_partition_windows(
            prewarm_layout(&index_ivf, 6, 6),
            prewarm_layout(&storage_ivf, 6, 6),
            1024,
            100,
        )
        .unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| window.partitions.clone())
                .collect::<Vec<_>>(),
            vec![0..1, 1..3]
        );
    }

    #[test]
    fn test_split_prewarm_window_compacts_partition_buffers() {
        let parent = RecordBatch::try_from_iter([(
            "value",
            Arc::new(UInt64Array::from_iter_values(0..100)) as ArrayRef,
        )])
        .unwrap();
        let parent_ptr = parent["value"]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ptr();
        let parent_size = parent["value"].get_array_memory_size();
        let mut partitions =
            super::split_window_batches(&parent.schema(), &[10, 0, 90], vec![parent]).unwrap();

        let shared_ptr = partitions[0][0]["value"]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ptr();
        assert_eq!(shared_ptr, parent_ptr);
        let compact = super::compact_partition_batches(partitions.remove(0)).unwrap();
        let compact_ptr = compact["value"]
            .as_primitive::<UInt64Type>()
            .values()
            .as_ptr();
        assert_ne!(compact_ptr, parent_ptr);
        assert_lt!(compact["value"].get_array_memory_size(), parent_size);
        assert_eq!(compact.num_rows(), 10);
        assert_eq!(partitions[0][0].num_rows(), 0);
        assert_eq!(partitions[1][0].num_rows(), 90);
    }

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
        // Memoize the chunk-budget size before the coverage exists; it must
        // still pick the coverage up once a later filter builds it.
        assert_eq!(entry.size_bytes(), entry.as_ref().deep_size_of());
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
        assert_eq!(entry.size_bytes(), entry.as_ref().deep_size_of());
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

    /// Rows of `vectors_per_row` vectors around their cluster's centroid. The
    /// `straddling_row`, if any, spreads its vectors over the centroids instead,
    /// one per vector, so it belongs to several partitions at once.
    fn generate_clustered_multivec_batch(
        cluster_sizes: &[usize],
        centroids: &[(f32, f32)],
        vectors_per_row: usize,
        start_id: u64,
        straddling_row: Option<u64>,
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
                let row_id = current_id;
                ids.push(row_id);
                current_id += 1;
                for vector_idx in 0..vectors_per_row {
                    let (x, y) = match straddling_row {
                        Some(id) if id == row_id => centroids[vector_idx % centroids.len()],
                        _ => (x, y),
                    };
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
        distance_type: DistanceType,
    ) -> RecordBatch {
        dataset
            .scan()
            .nearest("vector", query, k)
            .unwrap()
            .distance_metric(distance_type)
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

        // A single k=10 query of 4-bit PQ + Dot sits near the 0.5 bar: Windows
        // CI has landed at 0.4 (4/10). Average over several queries so one
        // platform-dependent ranking does not fail the contract.
        const NUM_QUERIES: usize = 10;
        // 4-bit PQ ranking is coarser; rescoring more candidates is cheap on
        // this 256-row fixture and keeps mean recall clear of 0.5.
        let refine_factor = if num_bits == 4 { 8 } else { 4 };
        // HNSW requires ef >= k * refine. IVF_PQ ignores ef.
        let ef = 64.max(K * refine_factor as usize);
        let mut hits = 0usize;
        let query = vectors.value(0);
        let before_reopen = search_lightweight_pq_index(
            &dataset,
            query.as_ref(),
            K,
            LIGHTWEIGHT_PQ_PARTITIONS,
            refine_factor,
            ef,
            distance_type,
        )
        .await;
        for query_idx in 0..NUM_QUERIES {
            let query = vectors.value(query_idx);
            let ground_truth =
                ground_truth(&dataset, "vector", query.as_ref(), K, distance_type).await;
            let result = if query_idx == 0 {
                before_reopen.clone()
            } else {
                search_lightweight_pq_index(
                    &dataset,
                    query.as_ref(),
                    K,
                    LIGHTWEIGHT_PQ_PARTITIONS,
                    refine_factor,
                    ef,
                    distance_type,
                )
                .await
            };
            assert_eq!(result.num_rows(), K);
            let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
            assert_eq!(row_ids.iter().copied().collect::<HashSet<_>>().len(), K);
            let distances = result[DIST_COL].as_primitive::<Float32Type>().values();
            assert!(distances.iter().all(|distance| distance.is_finite()));
            assert!(distances.windows(2).all(|pair| pair[0] <= pair[1]));
            hits += row_ids
                .iter()
                .filter(|row_id| ground_truth.contains(row_id))
                .count();
        }
        let recall = hits as f32 / (NUM_QUERIES * K) as f32;
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
                refine_factor,
                ef,
                distance_type,
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

        append_template_vector_with_start_id(
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

    async fn append_template_vector_with_start_id(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        next_id: &mut u64,
    ) {
        append_template_vector_batch(dataset, rows, template, *next_id, None).await;
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
        // A tiny per-row drift keeps the appended rows distinct (identical rows
        // cannot be split by clustering) without moving them off their template's
        // partition.
        let mut appended_values = Vec::with_capacity(total_rows * DIM);
        for template in templates {
            for row in 0..rows_per_template {
                let mut values = template.clone();
                values[0] += row as f32 * 0.0001;
                appended_values.extend_from_slice(&values);
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

    async fn append_template_vector_with_params(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        write_params: Option<WriteParams>,
    ) {
        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        append_template_vector_batch(dataset, rows, template, start_id, write_params).await;
    }

    async fn append_template_vector_batch(
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
        // The same tiny per-row drift as `append_partition_templates`: a split
        // into `ceil(rows / target)` pieces needs that many distinct rows, and
        // identical rows would leave some pieces empty at random.
        let mut appended_values = Vec::with_capacity(rows * DIM);
        for row in 0..rows {
            let mut values = template.to_vec();
            values[0] += row as f32 * 0.0001;
            appended_values.extend_from_slice(&values);
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
        append_template_vector_with_start_id(dataset, rows_to_append, template, next_id).await;
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
            .load_partition(partition_idx, None)
            .await
            .unwrap()
            .row_ids()
            .copied()
            .collect()
    }

    async fn load_flat_partition_row_ids(index: &IvfFlatIndex, partition_idx: usize) -> Vec<u64> {
        index
            .storage
            .load_partition(partition_idx, None)
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

    #[tokio::test]
    async fn test_layered_plane_promotion_survives_query_reconstruction() {
        use lance_index::vector::bq::layered::{EntryColumns, PlaneKey, SignBounds};

        let dir = TempStrDir::default();
        let (mut dataset, vectors) =
            generate_test_dataset::<Float32Type>(dir.as_str(), 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_rq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            RQBuildParams::new(7).with_layered(true),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        let cold = crate::DatasetBuilder::from_uri(dir.as_str())
            .with_session(Arc::new(crate::session::Session::new(
                64 * 1024 * 1024,
                1024 * 1024,
                Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
            )))
            .load()
            .await
            .unwrap();
        let index_id = cold.load_indices().await.unwrap()[0].uuid;
        let cache = cold.index_cache.for_index(&index_id, None);
        let query = vectors.value(0);
        let num_rows = cold.count_rows(None).await.unwrap();
        for pass in 0..3 {
            let result = cold
                .scan()
                .nearest("vector", query.as_ref(), num_rows)
                .unwrap()
                .nprobes(4)
                .rq_cascade_factor(1)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(result.num_rows(), num_rows);
            for partition in 0..4 {
                for plane in [1, 2] {
                    // Ex plane keys do not depend on the bounds placement.
                    // A local index keeps no resident store, so its entries
                    // hold every column.
                    let key = PlaneKey {
                        partition,
                        plane,
                        sign_bounds: SignBounds::default(),
                        entry_columns: EntryColumns::All,
                    };
                    let resident = cache.get_resident_with_key(&key).await;
                    assert_eq!(
                        resident.is_some(),
                        pass == 2,
                        "pass={pass}, partition={partition}, plane={plane}"
                    );
                }
            }
        }
    }

    /// Prewarm and full-precision loads of a layered index over a two-tier
    /// cache, with and without sign-gated plane admission.
    mod layered_plane_gating {
        use super::*;

        use bytes::Bytes;
        use lance_core::cache::{
            CacheCodec, CacheEntry, CacheTier, InternalCacheKey, QuickCacheBackend,
        };
        use lance_index::vector::ApproxMode;
        use lance_index::vector::bq::layered::{PlaneBatch, RQPrecision};
        use lance_io::assert_io_eq;
        use lance_io::object_store::ObjectStoreRegistry;

        use crate::session::Session;

        const PARTITIONS: usize = 8;
        const PLANES: usize = 3;
        const BITS: u8 = 7;
        const K: usize = 10;
        /// Holds every entry, so the planes' total size can be measured.
        const LARGE_RAM_BYTES: usize = 256 * 1024 * 1024;
        const METADATA_CACHE_BYTES: usize = 64 * 1024 * 1024;
        /// The small RAM tier holds 1 / this of the plane bytes.
        const SMALL_RAM_DIVISOR: usize = 10;

        type DiskEntries = HashMap<InternalCacheKey, (Bytes, CacheCodec, usize)>;

        /// RAM tier plus an unbounded serialized "disk" tier, like a tiered
        /// cache: loads are written through and disk hits are admitted to
        /// RAM. `gated` is what the backend reports for plane admission.
        struct PlaneTierTestBackend {
            ram: QuickCacheBackend,
            disk: std::sync::Mutex<DiskEntries>,
            gated: bool,
        }

        impl std::fmt::Debug for PlaneTierTestBackend {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("PlaneTierTestBackend")
                    .field("gated", &self.gated)
                    .finish_non_exhaustive()
            }
        }

        impl PlaneTierTestBackend {
            fn new(ram_bytes: usize, gated: bool) -> Self {
                Self {
                    ram: QuickCacheBackend::with_capacity(ram_bytes),
                    disk: Default::default(),
                    gated,
                }
            }

            fn read_disk(&self, key: &InternalCacheKey) -> Option<(CacheEntry, usize)> {
                let (bytes, codec, size) = self.disk.lock().unwrap().get(key).cloned()?;
                codec.deserialize(&bytes).hit().map(|entry| (entry, size))
            }

            /// Persisted plane entries and their accounted bytes.
            fn persisted_planes(&self) -> (usize, usize) {
                self.disk
                    .lock()
                    .unwrap()
                    .values()
                    .filter(|(_, codec, _)| {
                        codec.type_id() == <PlaneBatch as CacheCodecImpl>::TYPE_ID
                    })
                    .fold((0, 0), |(entries, bytes), (_, _, size)| {
                        (entries + 1, bytes + size)
                    })
            }
        }

        #[async_trait::async_trait]
        impl CacheBackend for PlaneTierTestBackend {
            async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
                self.ram.get_resident(key).await
            }

            /// RAM as `peek_resident` reports it, then the serialized tier.
            async fn peek_tier(&self, key: &InternalCacheKey) -> CacheTier {
                if self.peek_resident(key).await {
                    CacheTier::Resident
                } else if self.disk.lock().unwrap().contains_key(key) {
                    CacheTier::Local
                } else {
                    CacheTier::Absent
                }
            }

            fn plane_admission_gated(&self) -> bool {
                self.gated
            }

            async fn get_without_promotion(
                &self,
                key: &InternalCacheKey,
                _codec: Option<CacheCodec>,
            ) -> Option<CacheEntry> {
                match self.ram.get_resident(key).await {
                    Some(entry) => Some(entry),
                    None => self.read_disk(key).map(|(entry, _)| entry),
                }
            }

            async fn get(
                &self,
                key: &InternalCacheKey,
                _codec: Option<CacheCodec>,
            ) -> Option<CacheEntry> {
                if let Some(entry) = self.ram.get(key, None).await {
                    return Some(entry);
                }
                let (entry, size) = self.read_disk(key)?;
                self.ram.insert(key, entry.clone(), size, None).await;
                Some(entry)
            }

            async fn insert(
                &self,
                key: &InternalCacheKey,
                entry: CacheEntry,
                size_bytes: usize,
                codec: Option<CacheCodec>,
            ) {
                if let Some(codec) = codec {
                    let mut bytes = Vec::new();
                    codec.serialize(&entry, &mut bytes).unwrap();
                    self.disk
                        .lock()
                        .unwrap()
                        .insert(*key, (Bytes::from(bytes), codec, size_bytes));
                }
                self.ram.insert(key, entry, size_bytes, None).await;
            }

            async fn get_or_insert<'a>(
                &self,
                key: &InternalCacheKey,
                loader: std::pin::Pin<
                    Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
                >,
                codec: Option<CacheCodec>,
            ) -> Result<(CacheEntry, bool)> {
                if let Some(entry) = self.get(key, codec).await {
                    return Ok((entry, true));
                }
                let (entry, size) = loader.await?;
                self.insert(key, entry.clone(), size, codec).await;
                Ok((entry, false))
            }

            async fn clear(&self) {
                self.ram.clear().await;
                self.disk.lock().unwrap().clear();
            }

            async fn num_entries(&self) -> usize {
                self.ram.num_entries().await
            }

            async fn size_bytes(&self) -> usize {
                self.ram.size_bytes().await
            }
        }

        async fn open_prewarmed(
            uri: &str,
            backend: Arc<PlaneTierTestBackend>,
        ) -> (Dataset, Arc<dyn VectorIndex>) {
            let session = Session::with_index_cache_backend(
                backend,
                METADATA_CACHE_BYTES,
                Arc::new(ObjectStoreRegistry::default()),
            );
            let dataset = crate::DatasetBuilder::from_uri(uri)
                .with_session(Arc::new(session))
                .load()
                .await
                .unwrap();
            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let index = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            index.prewarm().await.unwrap();
            (dataset, index)
        }

        fn full_query(key: ArrayRef) -> Query {
            Query {
                column: "vector".to_string(),
                key,
                k: K,
                lower_bound: None,
                upper_bound: None,
                minimum_nprobes: PARTITIONS,
                maximum_nprobes: Some(PARTITIONS),
                ef: None,
                refine_factor: None,
                metric_type: None,
                use_index: true,
                query_parallelism: DEFAULT_QUERY_PARALLELISM,
                dist_q_c: 0.0,
                approx_mode: ApproxMode::Normal,
                rq_precision: RQPrecision::Full,
                rq_cascade_factor: None,
            }
        }

        /// Row ids and distance bits of every probed partition, in result order.
        async fn search_all_partitions(
            index: &Arc<dyn VectorIndex>,
            query: &Query,
        ) -> (Vec<u64>, Vec<u32>) {
            let (partitions, dists) = index.find_partitions(query).unwrap();
            let probes = partitions.len();
            let batches = index
                .clone()
                .search_partitions(
                    query.clone(),
                    Arc::new(partitions),
                    Arc::new(dists),
                    0,
                    probes,
                    Arc::new(NoFilter),
                    None,
                    Arc::new(NoOpMetricsCollector),
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut row_ids = Vec::new();
            let mut distance_bits = Vec::new();
            for batch in &batches {
                row_ids.extend(batch[ROW_ID].as_primitive::<UInt64Type>().values());
                distance_bits.extend(
                    batch[DIST_COL]
                        .as_primitive::<Float32Type>()
                        .values()
                        .iter()
                        .map(|dist| dist.to_bits()),
                );
            }
            (row_ids, distance_bits)
        }

        /// A backend that admits planes like any entry is warmed partition by
        /// partition, so every plane is persisted although RAM holds only a
        /// fraction of them, and warm full-precision queries read nothing from
        /// the index file. A sign-gated backend of the same size persists only
        /// the lower planes of partitions whose sign plane stayed resident.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_ungated_prewarm_persists_every_plane() {
            let dir = TempStrDir::default();
            let (mut dataset, vectors) =
                generate_test_dataset::<Float32Type>(dir.as_str(), 0.0..1.0).await;
            let params = VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                IvfBuildParams::new(PARTITIONS),
                RQBuildParams::new(BITS).with_layered(true),
            );
            dataset
                .create_index(&["vector"], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
            let query = full_query(vectors.value(0));

            let resident = Arc::new(PlaneTierTestBackend::new(LARGE_RAM_BYTES, false));
            let (_, index) = open_prewarmed(dir.as_str(), resident.clone()).await;
            let (entries, plane_bytes) = resident.persisted_planes();
            assert_eq!(entries, PLANES * PARTITIONS);
            let expected = search_all_partitions(&index, &query).await;
            assert_eq!(expected.0.len(), K);

            let small_ram = plane_bytes / SMALL_RAM_DIVISOR;
            let ungated = Arc::new(PlaneTierTestBackend::new(small_ram, false));
            let (dataset, index) = open_prewarmed(dir.as_str(), ungated.clone()).await;
            assert_eq!(ungated.persisted_planes(), (entries, plane_bytes));
            assert!(ungated.ram.size_bytes().await <= small_ram);
            assert_eq!(search_all_partitions(&index, &query).await, expected);
            dataset.object_store.as_ref().io_stats_incremental();
            for _ in 0..2 {
                assert_eq!(search_all_partitions(&index, &query).await, expected);
            }
            let io = dataset.object_store.as_ref().io_stats_incremental();
            assert_io_eq!(io, read_iops, 0, "warm layered queries read no plane");

            let gated = Arc::new(PlaneTierTestBackend::new(small_ram, true));
            let (_, index) = open_prewarmed(dir.as_str(), gated.clone()).await;
            let (gated_entries, _) = gated.persisted_planes();
            assert!(
                (PARTITIONS..PLANES * PARTITIONS).contains(&gated_entries),
                "gated prewarm persisted {gated_entries} plane entries"
            );
            assert_eq!(search_all_partitions(&index, &query).await, expected);
        }
    }

    #[rstest]
    #[case(5)]
    #[case(7)]
    #[case(9)]
    #[tokio::test]
    async fn test_layered_rq_create_search_cascade_and_remap(
        #[case] bits: u8,
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)]
        distance_type: DistanceType,
    ) {
        use lance_index::vector::bq::layered::RQPrecision;
        let dir = TempStrDir::default();
        let (mut dataset, vectors) =
            generate_test_dataset::<Float32Type>(dir.as_str(), 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_rq_params(
            distance_type,
            IvfBuildParams::new(4),
            RQBuildParams::new(bits).with_layered(true),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        let query = vectors.value(0);
        let gt = ground_truth(&dataset, "vector", &query, 100, distance_type).await;
        let mut full = None;
        for precision in [RQPrecision::Sign, RQPrecision::High, RQPrecision::Full] {
            let result = dataset
                .scan()
                .nearest("vector", query.as_ref(), 100)
                .unwrap()
                .nprobes(4)
                .rq_precision(precision)
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap();
            let ids = result[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            assert!(
                ids.intersection(&gt).count() >= 50,
                "precision={precision:?}"
            );
            if precision == RQPrecision::Full {
                full = Some(result);
            }
        }
        let cold = crate::DatasetBuilder::from_uri(dir.as_str())
            .with_session(Arc::new(crate::session::Session::new(
                0,
                1024 * 1024,
                Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
            )))
            .load()
            .await
            .unwrap();
        let cascade = cold
            .scan()
            .nearest("vector", query.as_ref(), 100)
            .unwrap()
            .nprobes(4)
            .rq_cascade_factor(100)
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let full = full.unwrap();
        assert_eq!(cascade[ROW_ID].as_ref(), full[ROW_ID].as_ref());
        assert_eq!(cascade[DIST_COL].as_ref(), full[DIST_COL].as_ref());
        let upper = full[DIST_COL].as_primitive::<Float32Type>().value(50);
        let mut range_results = Vec::new();
        for cascade_factor in [None, Some(4)] {
            let mut scan = cold.scan();
            scan.nearest("vector", query.as_ref(), 100)
                .unwrap()
                .nprobes(4)
                .distance_range(None, Some(upper))
                .with_row_id();
            if let Some(factor) = cascade_factor {
                scan.rq_cascade_factor(factor).unwrap();
            }
            range_results.push(scan.try_into_batch().await.unwrap());
        }
        assert_eq!(
            range_results[0][ROW_ID].as_ref(),
            range_results[1][ROW_ID].as_ref()
        );
        assert_eq!(
            range_results[0][DIST_COL].as_ref(),
            range_results[1][DIST_COL].as_ref()
        );

        let index = &dataset.load_indices().await.unwrap()[0];
        let scheduler = ScanScheduler::new(
            Arc::new(ObjectStore::local()),
            SchedulerConfig::default_for_testing(),
        );
        let reader = open_rq_aux_reader(&dataset, scheduler, &index.uuid.to_string()).await;
        let loader = IvfQuantizationStorage::<RabitQuantizer>::try_new(reader, None)
            .await
            .unwrap();
        let source = loader.load_partition(0, None).await.unwrap();
        let offsets: Vec<u32> = (0..source.len() as u32).step_by(3).take(5).collect();
        assert!(offsets.len() > 1);
        let no_cache = LanceCache::no_cache();
        let selected = loader
            .load_candidates(0, offsets.clone(), &WeakLanceCache::from(&no_cache), None)
            .await
            .unwrap();
        let source_calc = source.dist_calculator(query.clone(), 0.0);
        let selected_calc = selected.dist_calculator(query.clone(), 0.0);
        for (row, offset) in offsets.into_iter().enumerate() {
            assert_eq!(selected.row_id(row as u32), source.row_id(offset));
            assert_eq!(
                selected_calc.distance(row as u32).to_bits(),
                source_calc.distance(offset).to_bits()
            );
        }
        append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let scheduler = ScanScheduler::new(
            Arc::new(ObjectStore::local()),
            SchedulerConfig::default_for_testing(),
        );
        for index in indices.iter() {
            assert!(
                get_rq_metadata(&dataset, scheduler.clone(), &index.uuid.to_string())
                    .await
                    .layered
            );
        }
        test_remap(params, 4, 0.5).await;
    }

    /// The lazy layered full-precision scan must return exactly the eager
    /// scan's batch (row ids, distance bits and heap order) for every cache
    /// state, filter, bound and scan setting.
    mod layered_lazy {
        use super::*;
        use std::sync::LazyLock;
        use std::sync::atomic::AtomicU64;
        use std::time::Duration;

        use arrow::compute::concat_batches;
        use arrow_array::UInt32Array;
        use bytes::Bytes;
        use lance_arrow::RecordBatchExt;
        use lance_core::cache::{
            CacheCodec, CacheEntry, CachePin, CacheTier, InternalCacheKey, PinnedEntryLoader,
            PinnedStats, PinnedValue, QuickCacheBackend,
        };
        use lance_encoding::decoder::FilterExpression;
        use lance_file::version::LanceFileVersion;
        use lance_file::writer::FileWriterOptions;
        use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
        use lance_index::metrics::LocalMetricsCollector;
        use lance_index::vector::bq::layered::{
            EntryColumns, FULL_BOUNDS_COLUMN, PlaneBatch, RQPrecision, SIGN_BOUNDS_PLANE,
            SignBounds, plane_columns, plane_entry_columns,
        };
        use lance_index::vector::bq::layered_stats::{self, LayeredLazyStats, RANK_BUCKETS};
        use lance_index::vector::bq::partition_codes::{PartitionCodes, PartitionCodesKey};
        use lance_index::vector::bq::storage::{
            RABIT_CODE_COLUMN, RabitPruneStatsSnapshot, rabit_prune_stats_snapshot,
        };
        use lance_index::vector::storage::{
            DenseGatherMode, DenseToEager, GatherPlan, HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES,
            IndexFileKey, LayeredLazyConfig, LazyOriginGap, LazyPromotion, OriginLatencyClass,
            PlaneSource, ResidentColumnsKey, ResidentColumnsSetting, ResidentLifetime,
            entry_columns_setting, origin_latency_setting, resident_columns_setting,
            resident_lifetime_setting, resident_store_is_live, sign_bounds_setting,
        };
        use lance_index::vector::{ApproxMode, PartitionSearchControl, VECTOR_RESULT_SCHEMA};
        use lance_io::ReadBatchParams;
        use lance_io::assert_io_eq;
        use lance_io::object_store::ObjectStoreRegistry;
        use lance_io::scheduler::IoStats;
        use rand::seq::SliceRandom;

        use crate::session::Session;

        type IvfRq = super::super::IVFIndex<FlatIndex, RabitQuantizer>;

        const LAZY_DIM: usize = 128;
        const LAZY_PARTITIONS: usize = 64;
        const LAZY_ROWS: usize = 20_000;
        /// `(partition, rows)` of partitions that are empty or smaller than
        /// one 32-row FastScan block, or straddle its boundary.
        const LAZY_SMALL_PARTITIONS: [(usize, usize); 8] = [
            (3, 0),
            (11, 0),
            (7, 1),
            (13, 5),
            (19, 17),
            (23, 31),
            (29, 32),
            (31, 33),
        ];
        /// About a tenth of the index's plane bytes, so plane entries churn.
        const LAZY_SMALL_CACHE_BYTES: usize = 512 * 1024;
        /// A few plane entries, so nearly every plane a query reads is admitted anew.
        const LAZY_TINY_CACHE_BYTES: usize = 64 * 1024;
        const LAZY_LARGE_CACHE_BYTES: usize = 256 * 1024 * 1024;
        const LAZY_METADATA_CACHE_BYTES: usize = 64 * 1024 * 1024;

        /// These tests assert deltas of process-wide counters, so they run one at a time.
        static LAZY_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(Default::default);

        /// RAM tier plus a serialized "disk" tier, like a tiered cache: disk
        /// hits are admitted to RAM, loads are written through, and selected
        /// rows are read from the serialized entry.
        struct TieredPlaneTestBackend {
            ram: QuickCacheBackend,
            disk: std::sync::Mutex<HashMap<InternalCacheKey, (Bytes, CacheCodec, usize)>>,
            gated: bool,
            serve_rows: AtomicBool,
            /// Report the plane entries of empty partitions as not resident,
            /// as if the tier had evicted them.
            hide_empty_planes: AtomicBool,
            /// When set, every RAM admission through `get_or_insert` waits
            /// for a slot in this bounded channel, like a tiered cache whose
            /// admissions queue their evictions for the disk.
            spill: std::sync::Mutex<Option<tokio::sync::mpsc::Sender<()>>>,
            /// When set, plane loads through `get_or_insert` are recorded
            /// here, and each yields once after it starts so that loads its
            /// caller issues together overlap.
            plane_loads: std::sync::Mutex<Option<Vec<PlaneLoad>>>,
            /// How long every row read waits before it is served or misses,
            /// like a slow tier, so that sparse gathers stay in flight.
            row_read_delay: std::sync::Mutex<Duration>,
            /// Keys of the cached index states written through, so that a
            /// test can evict the states alone from RAM.
            state_keys: std::sync::Mutex<HashSet<InternalCacheKey>>,
            /// Keys whose RAM copy `get` treats as evicted until it admits
            /// the persistent copy again.
            evicted_from_ram: std::sync::Mutex<HashSet<InternalCacheKey>>,
            /// Bytes of churn the next admission of an entry evicted with
            /// `evict_states_from_ram` runs first, as another query's planes
            /// admitted meanwhile would.
            churn_on_admission: AtomicUsize,
            /// Admissions of evicted entries that ran churn first.
            churned_admissions: AtomicUsize,
            /// Source of the keys churned entries take.
            churn_keys: AtomicU64,
        }

        /// The loader a backend's `get_or_insert` receives.
        type EntryLoader<'a> = std::pin::Pin<
            Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
        >;

        /// A plane load through the backend's `get_or_insert`, by plane.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum PlaneLoad {
            Started(u8),
            Finished(u8),
        }

        impl std::fmt::Debug for TieredPlaneTestBackend {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TieredPlaneTestBackend")
                    .field("gated", &self.gated)
                    .finish_non_exhaustive()
            }
        }

        impl TieredPlaneTestBackend {
            fn new(ram_bytes: usize, gated: bool) -> Self {
                Self {
                    ram: QuickCacheBackend::with_capacity(ram_bytes),
                    disk: Default::default(),
                    gated,
                    serve_rows: AtomicBool::new(true),
                    hide_empty_planes: AtomicBool::new(false),
                    spill: Default::default(),
                    plane_loads: Default::default(),
                    row_read_delay: Default::default(),
                    state_keys: Default::default(),
                    evicted_from_ram: Default::default(),
                    churn_on_admission: AtomicUsize::new(0),
                    churned_admissions: AtomicUsize::new(0),
                    churn_keys: AtomicU64::new(0),
                }
            }

            /// Evict the cached index states from RAM alone, keeping their
            /// persistent copies: the next open reads its state back.
            fn evict_states_from_ram(&self) {
                let states = self.state_keys.lock().unwrap().clone();
                self.evicted_from_ram.lock().unwrap().extend(states);
            }

            /// Run `bytes` of churn before the next admission of an entry
            /// evicted with `evict_states_from_ram`.
            fn churn_on_next_admission(&self, bytes: usize) {
                self.churn_on_admission.store(bytes, Ordering::SeqCst);
            }

            /// Admit `bytes` of entries to RAM, each read back so that it is
            /// promoted to the hot ring and pushes older entries out, as the
            /// planes of another index's queries do.
            async fn churn_ram(&self, bytes: usize) {
                const CHURN_ENTRY_BYTES: usize = 16 * 1024;
                for _ in 0..bytes / CHURN_ENTRY_BYTES {
                    let id = self.churn_keys.fetch_add(1, Ordering::Relaxed);
                    let mut key = [0xC5; 16];
                    key[..8].copy_from_slice(&id.to_le_bytes());
                    let key = InternalCacheKey::from_bytes(key);
                    let entry = Arc::new(vec![0u8; CHURN_ENTRY_BYTES]);
                    self.ram.insert(&key, entry, CHURN_ENTRY_BYTES, None).await;
                    self.ram.get(&key, None).await;
                }
            }

            fn set_row_read_delay(&self, delay: Duration) {
                *self.row_read_delay.lock().unwrap() = delay;
            }

            fn record_plane_loads(&self) {
                *self.plane_loads.lock().unwrap() = Some(Vec::new());
            }

            /// The plane loads recorded since `record_plane_loads`, which
            /// stops recording.
            fn take_plane_loads(&self) -> Vec<PlaneLoad> {
                self.plane_loads.lock().unwrap().take().unwrap_or_default()
            }

            /// Record `load` when recording; returns whether it did.
            fn record_plane_load(&self, load: PlaneLoad) -> bool {
                match self.plane_loads.lock().unwrap().as_mut() {
                    Some(loads) => {
                        loads.push(load);
                        true
                    }
                    None => false,
                }
            }

            async fn get_or_insert_entry(
                &self,
                key: &InternalCacheKey,
                loader: EntryLoader<'_>,
                codec: Option<CacheCodec>,
            ) -> Result<(CacheEntry, bool)> {
                if let Some(entry) = self.ram.get(key, None).await {
                    return Ok((entry, true));
                }
                let admitted = match self.read_disk(key) {
                    Some((entry, size)) => {
                        self.ram.insert(key, entry.clone(), size, None).await;
                        (entry, true)
                    }
                    None => {
                        let (entry, size) = loader.await?;
                        self.insert(key, entry.clone(), size, codec).await;
                        (entry, false)
                    }
                };
                let spill = self.spill.lock().unwrap().clone();
                if let Some(spill) = spill {
                    // A closed channel only means the test stopped draining it.
                    let _ = spill.send(()).await;
                }
                Ok(admitted)
            }

            fn set_spill(&self, spill: Option<tokio::sync::mpsc::Sender<()>>) {
                *self.spill.lock().unwrap() = spill;
            }

            fn persist(
                &self,
                key: &InternalCacheKey,
                entry: &CacheEntry,
                size: usize,
                codec: Option<CacheCodec>,
            ) {
                let Some(codec) = codec else {
                    return;
                };
                let mut bytes = Vec::new();
                codec.serialize(entry, &mut bytes).unwrap();
                self.disk
                    .lock()
                    .unwrap()
                    .insert(*key, (Bytes::from(bytes), codec, size));
            }

            fn read_disk(&self, key: &InternalCacheKey) -> Option<(CacheEntry, usize)> {
                let (bytes, codec, size) = self.disk.lock().unwrap().get(key).cloned()?;
                codec.deserialize(&bytes).hit().map(|entry| (entry, size))
            }

            fn persisted_entries(&self, type_id: &str) -> usize {
                self.disk
                    .lock()
                    .unwrap()
                    .values()
                    .filter(|(_, codec, _)| codec.type_id() == type_id)
                    .count()
            }
        }

        #[async_trait::async_trait]
        impl CacheBackend for TieredPlaneTestBackend {
            async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
                let entry = self.ram.get_resident(key).await?;
                let empty_plane = entry
                    .downcast_ref::<PlaneBatch>()
                    .is_some_and(|plane| plane.0.num_rows() == 0);
                if empty_plane && self.hide_empty_planes.load(Ordering::Relaxed) {
                    return None;
                }
                Some(entry)
            }

            /// RAM as `peek_resident` reports it, hidden empty planes
            /// included, then the serialized tier.
            async fn peek_tier(&self, key: &InternalCacheKey) -> CacheTier {
                if self.peek_resident(key).await {
                    CacheTier::Resident
                } else if self.disk.lock().unwrap().contains_key(key) {
                    CacheTier::Local
                } else {
                    CacheTier::Absent
                }
            }

            fn plane_admission_gated(&self) -> bool {
                self.gated
            }

            async fn get_without_promotion(
                &self,
                key: &InternalCacheKey,
                _codec: Option<CacheCodec>,
            ) -> Option<CacheEntry> {
                match self.ram.get_resident(key).await {
                    Some(entry) => Some(entry),
                    None => self.read_disk(key).map(|(entry, _)| entry),
                }
            }

            async fn get_rows(
                &self,
                key: &InternalCacheKey,
                rows: &[u32],
                _codec: Option<CacheCodec>,
            ) -> Option<CacheEntry> {
                let delay = *self.row_read_delay.lock().unwrap();
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                if !self.serve_rows.load(Ordering::Relaxed) {
                    return None;
                }
                let (bytes, codec, _) = self.disk.lock().unwrap().get(key).cloned()?;
                let reader = |range: Range<usize>| -> Result<Bytes> { Ok(bytes.slice(range)) };
                codec.deserialize_rows(&reader, rows).hit()
            }

            async fn get(
                &self,
                key: &InternalCacheKey,
                _codec: Option<CacheCodec>,
            ) -> Option<CacheEntry> {
                let evicted = self.evicted_from_ram.lock().unwrap().remove(key);
                if !evicted && let Some(entry) = self.ram.get(key, None).await {
                    return Some(entry);
                }
                let (entry, size) = self.read_disk(key)?;
                let churn = if evicted {
                    self.churn_on_admission.swap(0, Ordering::SeqCst)
                } else {
                    0
                };
                if churn > 0 {
                    self.churn_ram(churn).await;
                    self.churned_admissions.fetch_add(1, Ordering::SeqCst);
                }
                self.ram.insert(key, entry.clone(), size, None).await;
                Some(entry)
            }

            async fn insert(
                &self,
                key: &InternalCacheKey,
                entry: CacheEntry,
                size_bytes: usize,
                codec: Option<CacheCodec>,
            ) {
                if entry.downcast_ref::<IvfStateEntryBox>().is_some() {
                    self.state_keys.lock().unwrap().insert(*key);
                }
                self.persist(key, &entry, size_bytes, codec);
                self.ram.insert(key, entry, size_bytes, None).await;
            }

            /// Pinned entries live in RAM only, as in a tiered cache.
            async fn get_leased(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
                self.ram.get_leased(key).await
            }

            async fn insert_pinned(
                &self,
                key: &InternalCacheKey,
                entry: CacheEntry,
                size_bytes: usize,
                pin: &Arc<CachePin>,
            ) {
                self.ram.insert_pinned(key, entry, size_bytes, pin).await;
            }

            async fn get_or_insert_pinned<'a>(
                &self,
                key: &InternalCacheKey,
                loader: PinnedEntryLoader<'a>,
            ) -> Result<(CacheEntry, bool)> {
                self.ram.get_or_insert_pinned(key, loader).await
            }

            fn max_entry_bytes(&self) -> Option<u64> {
                self.ram.max_entry_bytes()
            }

            fn pinned_stats(&self) -> PinnedStats {
                self.ram.pinned_stats()
            }

            async fn get_or_insert<'a>(
                &self,
                key: &InternalCacheKey,
                loader: std::pin::Pin<
                    Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
                >,
                codec: Option<CacheCodec>,
            ) -> Result<(CacheEntry, bool)> {
                let plane = codec
                    .and_then(|codec| codec.plane_tag())
                    .filter(|&plane| self.record_plane_load(PlaneLoad::Started(plane)));
                if plane.is_some() {
                    tokio::task::yield_now().await;
                }
                let result = self.get_or_insert_entry(key, loader, codec).await;
                if let Some(plane) = plane {
                    self.record_plane_load(PlaneLoad::Finished(plane));
                }
                result
            }

            async fn clear(&self) {
                self.ram.clear().await;
                self.disk.lock().unwrap().clear();
            }

            async fn num_entries(&self) -> usize {
                self.ram.num_entries().await
            }

            async fn size_bytes(&self) -> usize {
                self.ram.size_bytes().await
            }
        }

        /// Prefilter over a row-address mask. `collapses` makes the scan ask
        /// for partition coverage and replace the filter with `NoFilter`.
        struct LazyTestFilter {
            mask: Arc<RowAddrMask>,
            collapses: bool,
        }

        #[async_trait::async_trait]
        impl PreFilter for LazyTestFilter {
            async fn wait_for_ready(&self) -> Result<()> {
                Ok(())
            }

            fn is_empty(&self) -> bool {
                false
            }

            fn needs_partition_row_ids(&self) -> bool {
                self.collapses
            }

            fn is_empty_for(&self, _rows: &RowAddrTreeMap) -> bool {
                self.collapses
            }

            fn mask(&self) -> Arc<RowAddrMask> {
                self.mask.clone()
            }

            fn filter_row_ids<'a>(
                &self,
                row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>,
            ) -> Vec<u64> {
                self.mask.selected_indices(row_ids)
            }
        }

        /// A prefilter whose scalar-index load failed.
        struct FailingFilter;

        #[async_trait::async_trait]
        impl PreFilter for FailingFilter {
            async fn wait_for_ready(&self) -> Result<()> {
                Err(lance_core::Error::internal("prefilter failed to load"))
            }

            fn is_empty(&self) -> bool {
                false
            }

            fn mask(&self) -> Arc<RowAddrMask> {
                Arc::new(RowAddrMask::all_rows())
            }

            fn filter_row_ids<'a>(
                &self,
                row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>,
            ) -> Vec<u64> {
                row_ids.copied().collect()
            }
        }

        struct NeverStop;

        impl PartitionSearchControl for NeverStop {
            fn should_stop(&self) -> bool {
                false
            }
        }

        fn lazy_test_filters() -> Vec<(&'static str, Arc<dyn PreFilter>)> {
            let rows = |keep: fn(u64) -> bool| -> RowAddrTreeMap {
                (0..LAZY_ROWS as u64)
                    .filter(|&row| keep(row))
                    .collect::<Vec<_>>()
                    .iter()
                    .collect()
            };
            let filter = |mask: RowAddrMask, collapses: bool| -> Arc<dyn PreFilter> {
                Arc::new(LazyTestFilter {
                    mask: Arc::new(mask),
                    collapses,
                })
            };
            vec![
                ("none", Arc::new(NoFilter) as Arc<dyn PreFilter>),
                (
                    "sparse",
                    filter(RowAddrMask::from_allowed(rows(|row| row % 20 == 3)), false),
                ),
                (
                    "dense",
                    filter(RowAddrMask::from_block(rows(|row| row % 20 == 7)), false),
                ),
                (
                    "alternating",
                    filter(RowAddrMask::from_allowed(rows(|row| row % 2 == 0)), false),
                ),
                (
                    "single",
                    filter(RowAddrMask::from_allowed(rows(|row| row == 42)), false),
                ),
                (
                    "empty",
                    filter(RowAddrMask::from_allowed(RowAddrTreeMap::default()), false),
                ),
                ("collapse", filter(RowAddrMask::all_rows(), true)),
            ]
        }

        /// Lazy settings the parity tests rotate through, each with the origin
        /// latency class of the index it runs on. On a low-latency origin the
        /// default dense-to-eager routing (`origin`) routes no probe, so some
        /// settings route every predicted-dense probe (`all`), and the far
        /// window never applies. Promotions are off unless set, so some
        /// settings that gather sparse rows promote a plane after one read.
        /// The count of settings is prime, so over the parity test's cases
        /// the rotation pairs every setting with every `k`, filter,
        /// approximation mode and bounds choice.
        fn lazy_test_configs() -> Vec<(OriginLatencyClass, LayeredLazyConfig)> {
            let enabled = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };
            let promote_once = LazyPromotion::Background { reads: 1 };
            let low = [
                // The cost policy's defaults, whose sparse gathers promote.
                LayeredLazyConfig {
                    promote: promote_once,
                    ..enabled
                },
                // Sparse origin reads merge every run of a plane's column page.
                LayeredLazyConfig {
                    window: 0,
                    dense: DenseGatherMode::Sparse,
                    promote: LazyPromotion::Off,
                    inline_rows: 0,
                    origin_gap: LazyOriginGap::Bytes(u64::MAX),
                    ..enabled
                },
                // Whole gathers predict every probe dense, which would
                // otherwise send every probe to the eager scan.
                LayeredLazyConfig {
                    window: 1,
                    dense: DenseGatherMode::Whole,
                    promote: LazyPromotion::Background { reads: 2 },
                    inline_rows: usize::MAX,
                    dense_to_eager: DenseToEager::Off,
                    ..enabled
                },
                LayeredLazyConfig {
                    window: 8,
                    max_runs: 0,
                    dense_bytes_fraction: 0.1,
                    dense_to_eager: DenseToEager::All,
                    ..enabled
                },
                // Sparse origin reads merge only touching runs.
                LayeredLazyConfig {
                    window: 64,
                    max_runs: 2,
                    promote: LazyPromotion::Off,
                    inline_rows: 0,
                    origin_gap: LazyOriginGap::Bytes(0),
                    ..enabled
                },
                // Every gather is sparse, and promotes.
                LayeredLazyConfig {
                    max_runs: usize::MAX,
                    dense_bytes_fraction: f64::INFINITY,
                    dense_to_eager: DenseToEager::All,
                    promote: promote_once,
                    ..enabled
                },
                LayeredLazyConfig {
                    window: 4,
                    eager_before_full: false,
                    dense_to_eager: DenseToEager::All,
                    ..enabled
                },
                // Sparse gathers, which promote the planes they read.
                LayeredLazyConfig {
                    dense: DenseGatherMode::Sparse,
                    origin_max_runs: 0,
                    promote: promote_once,
                    ..enabled
                },
                LayeredLazyConfig {
                    dense_to_eager: DenseToEager::Off,
                    ..enabled
                },
            ];
            // On a high-latency origin a probe whose high or low plane no
            // cache tier holds is gathered within the far window.
            let high = [
                // The defaults: a far window of 64 probes with 64 permits,
                // and predicted-dense probes on the slow origin routed to the
                // eager scan.
                enabled,
                // Far gathers one at a time, beyond a window of one probe.
                LayeredLazyConfig {
                    window: 1,
                    far_window: LAZY_PARTITIONS,
                    far_inflight: 1,
                    ..enabled
                },
                // Far gathers as soon as the heap fills, beyond no window.
                LayeredLazyConfig {
                    window: 0,
                    eager_before_full: false,
                    dense_to_eager: DenseToEager::Off,
                    far_window: usize::MAX,
                    ..enabled
                },
                // The defaults with promotions after one sparse read: the far
                // window and the wide origin gap while promotions run.
                LayeredLazyConfig {
                    promote: promote_once,
                    ..enabled
                },
            ];
            let low = low
                .into_iter()
                .map(|config| (OriginLatencyClass::Low, config));
            let high = high
                .into_iter()
                .map(|config| (OriginLatencyClass::High, config));
            low.chain(high).collect()
        }

        /// Clustered vectors around seeded centroids, in shuffled row order,
        /// with the partition sizes of [`LAZY_SMALL_PARTITIONS`].
        fn lazy_test_data() -> (RecordBatch, Arc<FixedSizeListArray>) {
            let mut rng = StdRng::seed_from_u64(0x1a2e_7ed5);
            let centroids: Vec<f32> = (0..LAZY_PARTITIONS * LAZY_DIM)
                .map(|_| rng.random_range(-1.0f32..1.0))
                .collect();
            let small_rows: usize = LAZY_SMALL_PARTITIONS.iter().map(|(_, rows)| rows).sum();
            let regular_rows =
                (LAZY_ROWS - small_rows) / (LAZY_PARTITIONS - LAZY_SMALL_PARTITIONS.len());
            let mut sizes: Vec<usize> = (0..LAZY_PARTITIONS)
                .map(|partition| {
                    LAZY_SMALL_PARTITIONS
                        .iter()
                        .find(|(small, _)| *small == partition)
                        .map_or(regular_rows, |(_, rows)| *rows)
                })
                .collect();
            sizes[0] += LAZY_ROWS - sizes.iter().sum::<usize>();
            let mut partitions: Vec<usize> = sizes
                .iter()
                .enumerate()
                .flat_map(|(partition, &rows)| repeat_n(partition, rows))
                .collect();
            partitions.shuffle(&mut rng);
            let mut values = Vec::with_capacity(LAZY_ROWS * LAZY_DIM);
            for partition in partitions {
                for dim in 0..LAZY_DIM {
                    values.push(
                        centroids[partition * LAZY_DIM + dim] + rng.random_range(-0.05f32..0.05),
                    );
                }
            }
            let vectors = FixedSizeListArray::try_new_from_values(
                Float32Array::from(values),
                LAZY_DIM as i32,
            )
            .unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("vector", vectors.data_type().clone(), true),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from_iter_values(0..LAZY_ROWS as u64)),
                    Arc::new(vectors),
                ],
            )
            .unwrap();
            let centroids = FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroids),
                LAZY_DIM as i32,
            )
            .unwrap();
            (batch, Arc::new(centroids))
        }

        async fn write_lazy_test_dataset(
            uri: &str,
            bits: u8,
            distance_type: DistanceType,
        ) -> (Dataset, RecordBatch) {
            write_rq_test_dataset(uri, bits, distance_type, true).await
        }

        /// The lazy test data with a layered or a native IVF_RQ index.
        async fn write_rq_test_dataset(
            uri: &str,
            bits: u8,
            distance_type: DistanceType,
            layered: bool,
        ) -> (Dataset, RecordBatch) {
            let (batch, centroids) = lazy_test_data();
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let mut dataset = Dataset::write(reader, uri, None).await.unwrap();
            let params = VectorIndexParams::with_ivf_rq_params(
                distance_type,
                IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
                RQBuildParams::new(bits).with_layered(layered),
            );
            dataset
                .create_index(&["vector"], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
            (dataset, batch)
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum LazyTestCache {
            /// No capacity: every plane comes from the origin file.
            Origin,
            /// Lance's default backend at about a tenth of the planes, prewarmed.
            Small,
            /// Everything resident after prewarm.
            Resident,
            /// A tiered backend without sign gating, prewarmed to disk.
            Ungated,
            /// A tiered backend with sign gating at about a tenth of the planes.
            Gated,
            /// A tiered backend without sign gating that holds every plane.
            TieredResident,
            /// A tiered backend without sign gating whose RAM holds only a
            /// few planes, prewarmed to disk.
            Tiny,
            /// An empty tiered backend without sign gating that can hold every plane.
            ColdUngated,
            /// An empty tiered backend with sign gating that can hold every plane.
            ColdGated,
        }

        async fn open_lazy_test_index(
            uri: &str,
            cache: LazyTestCache,
        ) -> (
            Dataset,
            Arc<dyn VectorIndex>,
            Option<Arc<TieredPlaneTestBackend>>,
        ) {
            let (dataset, tiered) = open_lazy_test_dataset(uri, cache).await;
            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let index = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            if cache.is_warmed() {
                index.prewarm().await.unwrap();
            }
            (dataset, index, tiered)
        }

        impl LazyTestCache {
            /// Whether the tests warm an index opened over this cache.
            fn is_warmed(self) -> bool {
                !matches!(self, Self::Origin | Self::ColdUngated | Self::ColdGated)
            }
        }

        /// The dataset at `uri` over a new session whose index cache is
        /// `cache`, and the tiered backend of that cache, if it is one.
        async fn open_lazy_test_dataset(
            uri: &str,
            cache: LazyTestCache,
        ) -> (Dataset, Option<Arc<TieredPlaneTestBackend>>) {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let (session, tiered) = match cache {
                LazyTestCache::Origin => {
                    (Session::new(0, LAZY_METADATA_CACHE_BYTES, registry), None)
                }
                LazyTestCache::Small => (
                    Session::new(LAZY_SMALL_CACHE_BYTES, LAZY_METADATA_CACHE_BYTES, registry),
                    None,
                ),
                LazyTestCache::Resident => (
                    Session::new(LAZY_LARGE_CACHE_BYTES, LAZY_METADATA_CACHE_BYTES, registry),
                    None,
                ),
                LazyTestCache::Ungated
                | LazyTestCache::Gated
                | LazyTestCache::TieredResident
                | LazyTestCache::Tiny
                | LazyTestCache::ColdUngated
                | LazyTestCache::ColdGated => {
                    let (ram_bytes, gated) = match cache {
                        LazyTestCache::Gated => (LAZY_SMALL_CACHE_BYTES, true),
                        LazyTestCache::TieredResident | LazyTestCache::ColdUngated => {
                            (LAZY_LARGE_CACHE_BYTES, false)
                        }
                        LazyTestCache::ColdGated => (LAZY_LARGE_CACHE_BYTES, true),
                        LazyTestCache::Tiny => (LAZY_TINY_CACHE_BYTES, false),
                        _ => (LAZY_SMALL_CACHE_BYTES, false),
                    };
                    let backend = Arc::new(TieredPlaneTestBackend::new(ram_bytes, gated));
                    (
                        Session::with_index_cache_backend(
                            backend.clone(),
                            LAZY_METADATA_CACHE_BYTES,
                            registry,
                        ),
                        Some(backend),
                    )
                }
            };
            let dataset = crate::DatasetBuilder::from_uri(uri)
                .with_session(Arc::new(session))
                .load()
                .await
                .unwrap();
            (dataset, tiered)
        }

        fn lazy_index(index: &Arc<dyn VectorIndex>) -> &IvfRq {
            index
                .as_any()
                .downcast_ref::<IvfRq>()
                .expect("layered IVF_RQ index")
        }

        fn lazy_test_query(key: ArrayRef, k: usize, nprobes: usize) -> Query {
            Query {
                rq_cascade_factor: None,
                rq_precision: RQPrecision::Full,
                column: "vector".to_string(),
                key,
                k,
                lower_bound: None,
                upper_bound: None,
                minimum_nprobes: nprobes,
                maximum_nprobes: Some(nprobes),
                ef: None,
                refine_factor: None,
                metric_type: None,
                use_index: true,
                query_parallelism: DEFAULT_QUERY_PARALLELISM,
                dist_q_c: 0.0,
                approx_mode: ApproxMode::Normal,
            }
        }

        async fn search_global(
            index: &Arc<dyn VectorIndex>,
            query: &Query,
            filter: Arc<dyn PreFilter>,
        ) -> Result<RecordBatch> {
            let (partitions, dists) = index.find_partitions(query)?;
            let probes = partitions.len();
            let batches = index
                .clone()
                .search_partitions(
                    query.clone(),
                    Arc::new(partitions),
                    Arc::new(dists),
                    0,
                    probes,
                    filter,
                    None,
                    Arc::new(NoOpMetricsCollector),
                )
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            Ok(concat_batches(&VECTOR_RESULT_SCHEMA, batches.iter())?)
        }

        /// Row ids and distance bits in heap order.
        fn result_bits(batch: &RecordBatch) -> (Vec<u64>, Vec<u32>) {
            (
                batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec(),
                batch[DIST_COL]
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .map(|dist| dist.to_bits())
                    .collect(),
            )
        }

        fn prune_delta(
            after: RabitPruneStatsSnapshot,
            before: RabitPruneStatsSnapshot,
        ) -> [u64; 7] {
            [
                after.calls - before.calls,
                after.candidates - before.candidates,
                after.pruned_upper_bound - before.pruned_upper_bound,
                after.pruned_heap - before.pruned_heap,
                after.exact - before.exact,
                after.exact_rejected - before.exact_rejected,
                after.bypass_calls - before.bypass_calls,
            ]
        }

        /// Run `query` eagerly and with `config`, assert identical batches and
        /// (when `LANCE_RQ_PRUNE_STATS` is on) identical prune counters, and
        /// return the lazy run's counters.
        async fn assert_lazy_matches_eager(
            index: &Arc<dyn VectorIndex>,
            query: &Query,
            filter: &Arc<dyn PreFilter>,
            config: LayeredLazyConfig,
            context: &str,
        ) -> LayeredLazyStats {
            let ivf = lazy_index(index);
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let before = rabit_prune_stats_snapshot();
            let eager = search_global(index, query, filter.clone()).await.unwrap();
            let after_eager = rabit_prune_stats_snapshot();
            layered_stats::snapshot_and_reset();
            ivf.set_layered_lazy_config_for_test(config);
            let lazy = search_global(index, query, filter.clone()).await.unwrap();
            let after_lazy = rabit_prune_stats_snapshot();
            let stats = layered_stats::snapshot_and_reset();
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            assert_eq!(result_bits(&lazy), result_bits(&eager), "{context}");
            if before.enabled {
                assert_eq!(
                    prune_delta(after_lazy, after_eager),
                    prune_delta(after_eager, before),
                    "prune counters: {context}"
                );
            }
            assert_eq!(stats.needed_not_fetched, 0, "{context}");
            assert_lazy_chain_timing(&stats, context);
            stats
        }

        /// Check the release-chain timings of `stats`, taken over lazy scans
        /// that all completed. An eager scan times nothing; every lazy scan
        /// times its first probe's scoring once; a wait is timed only when it
        /// is counted, and the waits of a gather are part of its gate wait.
        /// Within one scan, the first probe's scoring is published no later
        /// than the heap first fills, and when a gather waited for the
        /// threshold or its turn, no gather that waits on scoring was issued
        /// before that publish.
        fn assert_lazy_chain_timing(stats: &LayeredLazyStats, context: &str) {
            let timings = [
                (
                    "rank 0 scored",
                    stats.lazy_rank0_scored_queries,
                    stats.lazy_rank0_scored_ns,
                ),
                (
                    "first gather issue",
                    stats.lazy_first_gather_issue_queries,
                    stats.lazy_first_gather_issue_ns,
                ),
                (
                    "window wait",
                    stats.lazy_window_waits,
                    stats.lazy_window_wait_ns,
                ),
                (
                    "release wait",
                    stats.deferred_issues,
                    stats.lazy_release_wait_ns,
                ),
                (
                    "far permit wait",
                    stats.far_permit_waits,
                    stats.far_permit_wait_ns,
                ),
            ];
            for (timing, count, ns) in timings {
                if stats.lazy_queries == 0 {
                    assert_eq!((count, ns), (0, 0), "{timing}: {context}");
                }
                if count == 0 {
                    assert_eq!(ns, 0, "{timing}: {context}");
                }
            }
            assert_eq!(
                stats.lazy_rank0_scored_queries, stats.lazy_queries,
                "{context}"
            );
            assert!(
                stats.lazy_first_gather_issue_queries <= stats.lazy_queries,
                "{context}"
            );
            let waits =
                stats.lazy_window_wait_ns + stats.lazy_release_wait_ns + stats.far_permit_wait_ns;
            assert!(
                waits <= stats.gate_wait_ns,
                "waits {waits} ns beyond the gate wait {} ns: {context}",
                stats.gate_wait_ns
            );
            if stats.lazy_queries == 1 {
                if stats.time_to_first_full_ns > 0 {
                    assert!(
                        stats.lazy_rank0_scored_ns <= stats.time_to_first_full_ns,
                        "{stats:?} {context}"
                    );
                }
                if stats.deferred_issues > 0 {
                    assert_eq!(stats.lazy_first_gather_issue_queries, 1, "{context}");
                    assert!(
                        stats.lazy_rank0_scored_ns <= stats.lazy_first_gather_issue_ns,
                        "{stats:?} {context}"
                    );
                }
            }
        }

        /// Deltas of the promotion gauge are not reset by snapshots; wait
        /// until background promotions have finished.
        async fn wait_for_promotions() {
            for _ in 0..500 {
                if layered_stats::promotions_in_flight() == 0 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!(
                "{} lazy promotions still running",
                layered_stats::promotions_in_flight()
            );
        }

        fn with_bounds(query: &Query, eager: &RecordBatch) -> Option<Query> {
            let mut dists = eager[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .to_vec();
            if dists.len() < 4 {
                return None;
            }
            dists.sort_by(f32::total_cmp);
            let mut bounded = query.clone();
            bounded.lower_bound = Some(dists[dists.len() / 10]);
            bounded.upper_bound = Some(dists[dists.len() * 3 / 5]);
            Some(bounded)
        }

        #[rstest]
        #[case::rq5_l2(5, DistanceType::L2)]
        #[case::rq7_l2(7, DistanceType::L2)]
        #[case::rq9_l2(9, DistanceType::L2)]
        #[case::rq7_cosine(7, DistanceType::Cosine)]
        #[case::rq7_dot(7, DistanceType::Dot)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_full_matches_eager(
            #[case] bits: u8,
            #[case] distance_type: DistanceType,
        ) {
            const STAGING_STEPS: usize = 8;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), bits, distance_type).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let mut rng = StdRng::seed_from_u64(u64::from(bits));
            let random: ArrayRef = Arc::new(Float32Array::from_iter_values(
                (0..LAZY_DIM).map(|_| rng.random_range(-1.0f32..1.0)),
            ));
            let keys = [vectors.value(0), vectors.value(777), random];
            let filters = lazy_test_filters();
            let configs = lazy_test_configs();
            let mut case = 0usize;
            // Single scans in which a gather waited for the threshold or its
            // turn, whose release chain `assert_lazy_chain_timing` orders.
            let mut deferred_scans = 0usize;
            let mut count_deferred = |stats: &LayeredLazyStats| {
                deferred_scans += usize::from(stats.lazy_queries == 1 && stats.deferred_issues > 0);
            };
            for cache in [
                LazyTestCache::Origin,
                LazyTestCache::Small,
                LazyTestCache::Resident,
                LazyTestCache::Ungated,
            ] {
                let (dataset, opened, _) = open_lazy_test_index(dir.as_str(), cache).await;
                let lengths = &lazy_index(&opened).storage.ivf().lengths;
                assert!(lengths.contains(&0), "the fixture needs an empty partition");
                assert!(
                    lengths.iter().any(|&rows| rows > 0 && rows < 32),
                    "the fixture needs a partition below one FastScan block"
                );
                // Each setting runs on an index pinned to its origin latency
                // class, over the caches the dataset's index warmed.
                let (store, index_dir) = index_files(&dataset).await;
                let open = |class| {
                    open_with_origin_latency(&dataset, store.clone(), index_dir.clone(), class)
                };
                let low = open(OriginLatencyClass::Low).await;
                let high = open(OriginLatencyClass::High).await;
                // Without a cache every plane of the high-latency twin is on
                // the slow origin, so its gathers may run ahead of scoring
                // within the far window. Staging several probes at once,
                // whatever the host's cores, makes some do so, as asserted
                // below.
                let origin_far = cache == LazyTestCache::Origin;
                if origin_far {
                    lazy_index(&high).set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
                }
                let mut far_early_issues = 0;
                let pinned = |class: OriginLatencyClass| match class {
                    OriginLatencyClass::Low => &low,
                    OriginLatencyClass::High => &high,
                };
                for key in &keys {
                    for nprobes in [1, 4, LAZY_PARTITIONS] {
                        for k in [1, 10, 100, 5000, LAZY_ROWS + 1] {
                            case += 1;
                            let (filter_name, filter) = &filters[case % filters.len()];
                            let (class, config) = configs[case % configs.len()];
                            let index = pinned(class);
                            let mut query = lazy_test_query(key.clone(), k, nprobes);
                            if case.is_multiple_of(3) {
                                query.approx_mode = ApproxMode::Accurate;
                            }
                            let context = format!(
                                "bits={bits} {distance_type:?} cache={cache:?} nprobes={nprobes} k={k} approx={:?} filter={filter_name} class={class} config={config:?}",
                                query.approx_mode
                            );
                            let stats =
                                assert_lazy_matches_eager(index, &query, filter, config, &context)
                                    .await;
                            count_deferred(&stats);
                            match cache {
                                LazyTestCache::Resident => {
                                    assert_eq!(stats.lazy_all_resident_skips, 1, "{context}");
                                    assert_eq!(stats.lazy_queries, 0, "{context}");
                                }
                                LazyTestCache::Origin => {
                                    assert_eq!(stats.lazy_queries, 1, "{context}");
                                }
                                LazyTestCache::Ungated => {
                                    assert_eq!(stats.origin_row_reads, 0, "{context}");
                                }
                                LazyTestCache::Small
                                | LazyTestCache::Gated
                                | LazyTestCache::TieredResident
                                | LazyTestCache::Tiny
                                | LazyTestCache::ColdUngated
                                | LazyTestCache::ColdGated => {}
                            }
                            if class == OriginLatencyClass::High {
                                far_early_issues += stats.far_early_issues;
                            }
                            if case % 2 == 1 {
                                let eager =
                                    search_global(index, &query, filter.clone()).await.unwrap();
                                if let Some(bounded) = with_bounds(&query, &eager) {
                                    let stats = assert_lazy_matches_eager(
                                        index,
                                        &bounded,
                                        filter,
                                        config,
                                        &format!("{context} bounds"),
                                    )
                                    .await;
                                    count_deferred(&stats);
                                    if class == OriginLatencyClass::High {
                                        far_early_issues += stats.far_early_issues;
                                    }
                                }
                            }
                        }
                    }
                }
                // Every setting against every filter on the widest probe set.
                let query = lazy_test_query(keys[0].clone(), 100, LAZY_PARTITIONS);
                for &(class, config) in &configs {
                    let index = pinned(class);
                    for (filter_name, filter) in &filters {
                        let context = format!(
                            "bits={bits} {distance_type:?} cache={cache:?} filter={filter_name} class={class} config={config:?}"
                        );
                        let stats =
                            assert_lazy_matches_eager(index, &query, filter, config, &context)
                                .await;
                        count_deferred(&stats);
                        if class == OriginLatencyClass::High {
                            far_early_issues += stats.far_early_issues;
                        }
                    }
                }
                if origin_far {
                    assert!(
                        far_early_issues > 0,
                        "bits={bits} {distance_type:?}: no gather ran ahead of scoring on the high-latency origin"
                    );
                }
                wait_for_promotions().await;
            }
            assert!(
                deferred_scans > 0,
                "bits={bits} {distance_type:?}: no scan deferred a gather, so no release chain was ordered"
            );
        }

        /// An ungated backend is warmed partition by partition, so every plane
        /// is persisted even when RAM holds only part of the sign planes, and
        /// warm queries read nothing from the origin file, whether the first
        /// probe (the only one this query gathers rows of: its threshold is
        /// +inf) is gathered sparsely, as the sparse gather policy always
        /// does, or loaded by the eager scan, as the cost policy routes it
        /// when every predicted-dense probe is routed.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_ungated_prewarm_persists_every_plane() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Ungated).await;
            let tiered = tiered.unwrap();
            assert_eq!(
                tiered.persisted_entries(<PlaneBatch as CacheCodecImpl>::TYPE_ID),
                3 * LAZY_PARTITIONS
            );
            let key = batch["vector"].as_fixed_size_list().value(5);
            let query = lazy_test_query(key, 100, LAZY_PARTITIONS);
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            search_global(&index, &query, filter.clone()).await.unwrap();
            // The sparse run goes first: the routed eager load admits the
            // first probe's planes, after which it would gather nothing.
            for dense in [DenseGatherMode::Sparse, DenseGatherMode::Cost] {
                let context = format!("ungated prewarm dense={dense:?}");
                dataset.object_store.as_ref().io_stats_incremental();
                let stats = assert_lazy_matches_eager(
                    &index,
                    &query,
                    &filter,
                    LayeredLazyConfig {
                        enabled: true,
                        dense,
                        promote: LazyPromotion::Off,
                        dense_to_eager: DenseToEager::All,
                        ..Default::default()
                    },
                    &context,
                )
                .await;
                let io = dataset.object_store.as_ref().io_stats_incremental();
                assert_io_eq!(
                    io,
                    read_iops,
                    0,
                    "warm layered queries read no origin plane: {context}"
                );
                assert_eq!(stats.origin_row_reads, 0, "{context}");
                let sparse_planes: u64 = stats.high_sparse.iter().chain(&stats.low_sparse).sum();
                let eager_loads: u64 = stats.dense_to_eager.iter().sum();
                if dense == DenseGatherMode::Sparse {
                    assert!(eager_loads == 0 && sparse_planes > 0, "{context} {stats:?}");
                } else {
                    assert!(eager_loads > 0, "{context} {stats:?}");
                }
            }
        }

        /// Queries the lazy scan cannot serve run the eager scan and are
        /// counted by reason.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_ineligible_queries_fall_back() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let key = batch["vector"].as_fixed_size_list().value(9);
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let config = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };

            let mut fast = lazy_test_query(key.clone(), 10, 8);
            fast.approx_mode = ApproxMode::Fast;
            let stats = assert_lazy_matches_eager(&index, &fast, &filter, config, "fast").await;
            assert_eq!((stats.ineligible_fast, stats.lazy_queries), (1, 0));

            let base = lazy_test_query(key.clone(), 100, 8);
            let eager = search_global(&index, &base, filter.clone()).await.unwrap();
            let mut cascade = with_bounds(&base, &eager).unwrap();
            cascade.rq_cascade_factor = Some(4);
            let stats =
                assert_lazy_matches_eager(&index, &cascade, &filter, config, "cascade").await;
            assert_eq!((stats.ineligible_cascade, stats.lazy_queries), (1, 0));

            let mut refine = lazy_test_query(key.clone(), 10, 8);
            refine.refine_factor = Some(2);
            let stats = assert_lazy_matches_eager(&index, &refine, &filter, config, "refine").await;
            assert_eq!((stats.ineligible_refine, stats.lazy_queries), (1, 0));

            let mut high = lazy_test_query(key.clone(), 10, 8);
            high.rq_precision = RQPrecision::High;
            let stats = assert_lazy_matches_eager(&index, &high, &filter, config, "high").await;
            assert_eq!((stats.ineligible_precision, stats.lazy_queries), (1, 0));

            // `k == 0` runs the eager scan, which still waits for the
            // prefilter and so reports its failure.
            let zero = lazy_test_query(key, 0, 8);
            let stats = assert_lazy_matches_eager(&index, &zero, &filter, config, "k=0").await;
            assert_eq!((stats.ineligible_k_zero, stats.lazy_queries), (1, 0));
            let failing: Arc<dyn PreFilter> = Arc::new(FailingFilter);
            let ivf = lazy_index(&index);
            for lazy in [LayeredLazyConfig::default(), config] {
                ivf.set_layered_lazy_config_for_test(lazy);
                let error = search_global(&index, &zero, failing.clone())
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("prefilter failed"), "{error}");
            }
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
        }

        /// A gated backend admits ex planes only behind their resident sign
        /// plane, so a background promotion would read a whole plane only to
        /// drop it; sparse gathers never promote there.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_gated_backend_never_promotes() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Gated).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let config = LayeredLazyConfig {
                enabled: true,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Background { reads: 1 },
                ..Default::default()
            };
            let mut sparse_planes = 0;
            for row in 0..4 {
                let query = lazy_test_query(vectors.value(row * 101), 100, LAZY_PARTITIONS);
                // Repeat each query so a plane is gathered sparsely more than once.
                for repeat in 0..2 {
                    let context = format!("gated row={row} repeat={repeat}");
                    let stats =
                        assert_lazy_matches_eager(&index, &query, &filter, config, &context).await;
                    assert_eq!(stats.lazy_queries, 1, "{context}");
                    assert_eq!(
                        (
                            stats.promotions_issued,
                            stats.promotions_deduped,
                            stats.promotions_skipped,
                            stats.promotion_bytes
                        ),
                        (0, 0, 0, 0),
                        "{context}"
                    );
                    sparse_planes += stats
                        .high_sparse
                        .iter()
                        .chain(&stats.low_sparse)
                        .sum::<u64>();
                }
            }
            assert_eq!(layered_stats::promotions_in_flight(), 0);
            assert!(sparse_planes > 0, "the gated setup must gather sparsely");
        }

        /// Empty partitions have no ex rows, so their plane entries, which the
        /// lazy scan never touches, do not keep a query whose other ex planes
        /// are all resident off the eager scan.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_empty_partitions_count_as_resident() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::TieredResident).await;
            let tiered = tiered.unwrap();
            let ivf = lazy_index(&index);
            let empty = LAZY_SMALL_PARTITIONS
                .iter()
                .find(|(_, rows)| *rows == 0)
                .map(|(partition, _)| *partition)
                .unwrap();
            assert_eq!(ivf.storage.partition_size(empty), 0);
            let high = ivf.storage.plane_key(empty, 1);
            assert!(ivf.index_cache.peek_resident_with_key(&high).await);
            tiered.hide_empty_planes.store(true, Ordering::Relaxed);
            assert!(!ivf.index_cache.peek_resident_with_key(&high).await);

            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let config = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };
            let query = lazy_test_query(
                batch["vector"].as_fixed_size_list().value(11),
                100,
                LAZY_PARTITIONS,
            );
            let stats =
                assert_lazy_matches_eager(&index, &query, &filter, config, "hidden empty planes")
                    .await;
            assert_eq!(
                (stats.lazy_all_resident_skips, stats.lazy_queries),
                (1, 0),
                "{stats:?}"
            );
        }

        /// `auto` makes cloud object stores class high and local files and
        /// memory class low, unless the session declares a class; an explicit
        /// class applies to every store, whatever the session declares.
        #[test]
        fn test_origin_latency_class_resolves_by_store() {
            let s3 = ObjectStore::new(
                Arc::new(object_store::memory::InMemory::new()),
                url::Url::parse("s3://bucket/").unwrap(),
                None,
                None,
                false,
                true,
                1,
                lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
                None,
            );
            for (store, auto) in [
                (ObjectStore::local(), OriginLatencyClass::Low),
                (ObjectStore::memory(), OriginLatencyClass::Low),
                (s3, OriginLatencyClass::High),
            ] {
                let scheme = store.scheme().to_string();
                assert_eq!(OriginLatencyClass::of_store(&store), auto, "{scheme}");
                assert_eq!(OriginLatencyClass::resolve(None, &store), auto, "{scheme}");
                assert_eq!(
                    OriginLatencyClass::resolve_with_hint(None, None, &store),
                    auto,
                    "{scheme}"
                );
                for class in [OriginLatencyClass::Low, OriginLatencyClass::High] {
                    assert_eq!(
                        OriginLatencyClass::resolve(Some(class), &store),
                        class,
                        "{scheme}"
                    );
                    assert_eq!(
                        OriginLatencyClass::resolve_with_hint(None, Some(class), &store),
                        class,
                        "{scheme} hint={class}"
                    );
                    for hint in [OriginLatencyClass::Low, OriginLatencyClass::High] {
                        assert_eq!(
                            OriginLatencyClass::resolve_with_hint(Some(class), Some(hint), &store),
                            class,
                            "{scheme} setting={class} hint={hint}"
                        );
                    }
                }
            }
        }

        /// The class is resolved when the index opens and carried into its
        /// storage, and through a reconstruction from the cached state. The
        /// test dataset is a local file, so the class is what the environment
        /// sets, or low.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_origin_latency_survives_reconstruction() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let expected = OriginLatencyClass::resolve(
                origin_latency_setting().unwrap(),
                &ObjectStore::local(),
            );
            let ivf = lazy_index(&index);
            assert_eq!(ivf.origin_latency(), expected);
            assert_eq!(ivf.storage.origin_latency(), expected);

            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let reconstructed = lazy_index(&reopened);
            assert_eq!(reconstructed.origin_latency(), expected);
            assert_eq!(reconstructed.storage.origin_latency(), expected);
        }

        /// A session's origin latency hint is the class of the IVF_RQ indexes
        /// it opens, when they open and when they are reconstructed from the
        /// cached state, and residency, the lazy origin gap and the promotion
        /// policy follow it. Without a hint an index takes its store's class,
        /// low for the test's local files, and an IVF_PQ index the session
        /// opens keeps its store's class whatever the hint. Results do not
        /// depend on the class. A class the environment sets overrides every
        /// hint, so the hint takes effect only where it sets none.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_session_origin_latency_hint_reaches_rq_index() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let rq_dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(rq_dir.as_str(), 7, DistanceType::L2).await;
            let pq_dir = TempStrDir::default();
            let (mut pq_dataset, _) =
                generate_test_dataset::<Float32Type>(pq_dir.as_str(), 0.0..1.0).await;
            let pq_params = VectorIndexParams::with_ivf_pq_params(
                DistanceType::L2,
                IvfBuildParams::new(LIGHTWEIGHT_PQ_PARTITIONS),
                lightweight_pq_params(),
            );
            pq_dataset
                .create_index(&["vector"], IndexType::Vector, None, &pq_params, true)
                .await
                .unwrap();

            let setting = origin_latency_setting().unwrap();
            let lazy_setting = LayeredLazyConfig::from_env().unwrap();
            let resident_setting = resident_columns_setting().unwrap();
            let entry_setting = entry_columns_setting().unwrap();
            let vectors = batch["vector"].as_fixed_size_list();
            let queries = [
                lazy_test_query(vectors.value(0), 10, 8),
                lazy_test_query(vectors.value(777), 100, LAZY_PARTITIONS),
            ];
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let lazy = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };
            let mut first_results = vec![None; queries.len()];
            for hint in [
                Some(OriginLatencyClass::High),
                Some(OriginLatencyClass::Low),
                None,
            ] {
                let session = Arc::new(
                    Session::new(
                        LAZY_LARGE_CACHE_BYTES,
                        LAZY_METADATA_CACHE_BYTES,
                        Arc::new(ObjectStoreRegistry::default()),
                    )
                    .with_index_origin_latency(hint),
                );
                let dataset = crate::DatasetBuilder::from_uri(rq_dir.as_str())
                    .with_session(session.clone())
                    .load()
                    .await
                    .unwrap();
                let store_class = OriginLatencyClass::of_store(&dataset.object_store);
                assert_eq!(store_class, OriginLatencyClass::Low);
                let expected = setting.or(hint).unwrap_or(store_class);
                let block_size = dataset.object_store.block_size() as u64;
                let uuid = dataset.load_indices().await.unwrap()[0].uuid;
                let opened = dataset
                    .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
                let state_key =
                    crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
                assert!(
                    dataset.index_cache.get_with_key(&state_key).await.is_some(),
                    "the reopen must reconstruct from the cached state"
                );
                let reconstructed = dataset
                    .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                for (path, index) in [("open", &opened), ("reconstruct", &reconstructed)] {
                    let context = format!("hint={hint:?} {path}");
                    let ivf = lazy_index(index);
                    assert_eq!(ivf.origin_latency(), expected, "{context}");
                    assert_eq!(ivf.storage.origin_latency(), expected, "{context}");
                    assert_eq!(
                        ivf.resident_columns_enabled(),
                        resident_setting.resolve(expected),
                        "{context}"
                    );
                    // Code-only entries follow residency, so the hint.
                    assert_eq!(
                        ivf.entry_columns(),
                        entry_setting.resolve(ivf.resident_columns_enabled()),
                        "{context}"
                    );
                    assert_eq!(
                        ivf.lazy_origin_gap_bytes(),
                        lazy_setting.origin_gap.resolve(expected, block_size),
                        "{context}"
                    );
                    assert_eq!(
                        ivf.layered_lazy_config().promote,
                        lazy_setting.promote.resolve(expected),
                        "{context}"
                    );
                    for (position, query) in queries.iter().enumerate() {
                        let context = format!("{context} query={position}");
                        assert_lazy_matches_eager(index, query, &filter, lazy, &context).await;
                        let eager = search_global(index, query, filter.clone()).await.unwrap();
                        let bits = result_bits(&eager);
                        let first = first_results[position].get_or_insert_with(|| bits.clone());
                        assert_eq!(&bits, first, "{context}");
                    }
                }

                let pq = crate::DatasetBuilder::from_uri(pq_dir.as_str())
                    .with_session(session)
                    .load()
                    .await
                    .unwrap();
                let pq_class = OriginLatencyClass::of_store(&pq.object_store);
                let pq_uuid = pq.load_indices().await.unwrap()[0].uuid;
                let pq_frag_reuse_uuid = pq.frag_reuse_index_uuid().await;
                let pq_state_key =
                    crate::index::IvfIndexStateCacheKey::new(&pq_uuid, pq_frag_reuse_uuid.as_ref());
                for path in ["open", "reconstruct"] {
                    let cached = pq.index_cache.get_with_key(&pq_state_key).await.is_some();
                    assert_eq!(cached, path == "reconstruct", "hint={hint:?} {path}");
                    let index = pq
                        .open_vector_index("vector", &pq_uuid, &NoOpMetricsCollector)
                        .await
                        .unwrap();
                    let ivf_pq = index
                        .as_any()
                        .downcast_ref::<IvfPq>()
                        .expect("IVF_PQ index");
                    assert_eq!(ivf_pq.origin_latency(), pq_class, "hint={hint:?} {path}");
                    assert_eq!(ivf_pq.entry_columns(), EntryColumns::All);
                }
            }
            wait_for_promotions().await;
        }

        /// A plane's tier follows where the backend holds it: RAM, then only
        /// the serialized tier, then nowhere.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_plane_tier_follows_cache_tiers() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            let ivf = lazy_index(&index);
            let cache = &ivf.index_cache;
            let partition = (0..LAZY_PARTITIONS)
                .find(|&partition| ivf.storage.partition_size(partition) > 0)
                .unwrap();
            let high = ivf.storage.plane_tier(partition, 1, cache).await;
            assert_eq!(high, CacheTier::Absent);

            ivf.storage
                .load_plane_entry(partition, 1, cache, None)
                .await
                .unwrap();
            let high = ivf.storage.plane_tier(partition, 1, cache).await;
            let low = ivf.storage.plane_tier(partition, 2, cache).await;
            assert_eq!((high, low), (CacheTier::Resident, CacheTier::Absent));

            // A tier peek of a persisted plane does not admit it to RAM.
            tiered.ram.clear().await;
            let high = ivf.storage.plane_tier(partition, 1, cache).await;
            assert_eq!(high, CacheTier::Local);
            assert_eq!(tiered.ram.num_entries().await, 0);

            tiered.clear().await;
            let high = ivf.storage.plane_tier(partition, 1, cache).await;
            assert_eq!(high, CacheTier::Absent);
        }

        /// Open the index of `dataset` over the dataset's caches, keeping its
        /// bounds columns where `sign_bounds` places them.
        async fn open_with_sign_bounds(
            dataset: &Dataset,
            sign_bounds: SignBounds,
        ) -> Arc<dyn VectorIndex> {
            let index = dataset.load_indices().await.unwrap()[0].clone();
            let ivf = IvfRq::try_new(
                dataset.object_store.clone(),
                dataset.indice_files_dir(&index).unwrap(),
                index.uuid,
                None,
                &dataset.metadata_cache,
                dataset.index_cache.for_index(&index.uuid, None),
                index.file_size_map(),
                IvfOpenContext::default(),
            )
            .await
            .unwrap()
            .with_sign_bounds_for_test(sign_bounds);
            Arc::new(ivf)
        }

        /// Every precision in both approximation modes, the cascade, and the
        /// lazy scan over every partition, labelled for assertion messages.
        fn sign_bounds_test_queries(key: &ArrayRef) -> Vec<(String, Query, LayeredLazyConfig)> {
            let eager_scan = LayeredLazyConfig::default();
            let mut queries = Vec::new();
            for rq_precision in [RQPrecision::Sign, RQPrecision::High, RQPrecision::Full] {
                for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                    let mut query = lazy_test_query(key.clone(), 100, 8);
                    query.rq_precision = rq_precision;
                    query.approx_mode = approx_mode;
                    let label = format!("precision={rq_precision:?} approx={approx_mode:?}");
                    queries.push((label, query, eager_scan));
                }
            }
            let mut cascade = lazy_test_query(key.clone(), 100, 8);
            cascade.rq_cascade_factor = Some(4);
            queries.push(("cascade".to_string(), cascade, eager_scan));
            let lazy_scan = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };
            let every_probe = lazy_test_query(key.clone(), 100, LAZY_PARTITIONS);
            queries.push(("lazy scan".to_string(), every_probe, lazy_scan));
            queries
        }

        /// Prewarm both placements into their shared cache, where each finds
        /// its own sign plane: only the eager placement's holds the bounds.
        async fn assert_sign_planes_coexist(placements: &[Arc<dyn VectorIndex>]) {
            for index in placements {
                index.prewarm().await.unwrap();
            }
            for index in placements {
                let ivf = lazy_index(index);
                let partition = (0..LAZY_PARTITIONS)
                    .find(|&partition| ivf.storage.partition_size(partition) > 0)
                    .unwrap();
                let sign = ivf
                    .index_cache
                    .get_resident_with_key(&ivf.storage.plane_key(partition, 0))
                    .await
                    .expect("prewarmed sign plane");
                let bounds_in_sign = sign.0.column_by_name(FULL_BOUNDS_COLUMN).is_some();
                assert_eq!(bounds_in_sign, ivf.sign_bounds() == SignBounds::Eager);
            }
        }

        /// Keeping the bounds columns in their own plane changes only which
        /// entries hold them: every precision, the cascade and the lazy scan
        /// return the batch of the sign plane with bounds, bit for bit, from
        /// the origin file and from one cache that holds both placements.
        #[rstest]
        #[case::rq5_l2(5, DistanceType::L2)]
        #[case::rq7_cosine(7, DistanceType::Cosine)]
        #[case::rq9_dot(9, DistanceType::Dot)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_sign_bounds_placements_match(
            #[case] bits: u8,
            #[case] distance_type: DistanceType,
        ) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), bits, distance_type).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let keys = [vectors.value(0), vectors.value(777)];
            for cache in [LazyTestCache::Origin, LazyTestCache::TieredResident] {
                let (dataset, _, _) = open_lazy_test_index(dir.as_str(), cache).await;
                let placements = [
                    open_with_sign_bounds(&dataset, SignBounds::Eager).await,
                    open_with_sign_bounds(&dataset, SignBounds::Lazy).await,
                ];
                if cache == LazyTestCache::TieredResident {
                    assert_sign_planes_coexist(&placements).await;
                }
                for key in &keys {
                    for (label, query, scan) in sign_bounds_test_queries(key) {
                        let context =
                            format!("bits={bits} {distance_type:?} cache={cache:?} {label}");
                        let mut results = Vec::new();
                        for index in &placements {
                            let ivf = lazy_index(index);
                            ivf.set_layered_lazy_config_for_test(scan);
                            let result = search_global(index, &query, Arc::new(NoFilter)).await;
                            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                            results.push(result_bits(&result.unwrap()));
                        }
                        assert!(!results[0].0.is_empty(), "{context}");
                        assert_eq!(results[1], results[0], "{context}");
                    }
                }
            }
        }

        /// The bounds plane is read only by the precisions that prune with
        /// the bounds, and only when the sign plane leaves them out. Full
        /// precision prunes with the error factors every layered file has.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_bounds_plane_loads_only_when_read() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, _, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            for (sign_bounds, high) in [
                (SignBounds::Lazy, vec![0, 1, SIGN_BOUNDS_PLANE]),
                (SignBounds::Eager, vec![0, 1]),
            ] {
                let index = open_with_sign_bounds(&dataset, sign_bounds).await;
                let ivf = lazy_index(&index);
                let partition = (0..LAZY_PARTITIONS)
                    .find(|&partition| ivf.storage.partition_size(partition) > 0)
                    .unwrap();
                for (precision, expected) in [
                    (RQPrecision::Sign, vec![0]),
                    (RQPrecision::High, high.clone()),
                    (RQPrecision::Full, vec![0, 1, 2]),
                ] {
                    tiered.clear().await;
                    tiered.record_plane_loads();
                    ivf.storage
                        .load_partition_at_precision(partition, precision, &ivf.index_cache, None)
                        .await
                        .unwrap();
                    let mut loaded: Vec<u8> = tiered
                        .take_plane_loads()
                        .into_iter()
                        .filter_map(|load| match load {
                            PlaneLoad::Started(plane) => Some(plane),
                            PlaneLoad::Finished(_) => None,
                        })
                        .collect();
                    loaded.sort_unstable();
                    assert_eq!(loaded, expected, "{sign_bounds} {precision:?}");
                }
            }
        }

        /// The bounds placement is what the environment sets, resolved when
        /// the index opens and again when it is reconstructed from the
        /// cached state.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_sign_bounds_survive_reconstruction() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let expected = sign_bounds_setting().unwrap();
            let ivf = lazy_index(&index);
            assert!(ivf.is_layered_rq());
            assert_eq!(ivf.sign_bounds(), expected);
            assert_eq!(ivf.storage.sign_bounds(), expected);

            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert!(lazy_index(&reopened).is_layered_rq());
            assert_eq!(lazy_index(&reopened).sign_bounds(), expected);
        }

        /// A row-id remapper can drop physical rows, so offsets are not row
        /// positions and the lazy scan must not run.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_remapped_index_is_ineligible() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (batch, schema) = generate_batch::<Float32Type>(1000, None, 0.0..1.0, false);
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema);
            let mut dataset = Dataset::write(
                reader,
                dir.as_str(),
                Some(WriteParams {
                    max_rows_per_file: 200,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            let params = VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                IvfBuildParams::new(4),
                RQBuildParams::new(7).with_layered(true),
            );
            dataset
                .create_index(&["vector"], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
            dataset.delete("id % 7 = 0").await.unwrap();
            compact_files(
                &mut dataset,
                CompactionOptions {
                    target_rows_per_fragment: 500,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            let uuid = dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .find(|index| index.name != FRAG_REUSE_INDEX_NAME)
                .unwrap()
                .uuid;
            let index = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert!(!lazy_index(&index).storage.supports_candidate_reads());
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(1), 20, 4);
            let stats = assert_lazy_matches_eager(
                &index,
                &query,
                &(Arc::new(NoFilter) as Arc<dyn PreFilter>),
                LayeredLazyConfig {
                    enabled: true,
                    ..Default::default()
                },
                "remapper",
            )
            .await;
            assert_eq!((stats.ineligible_remapper, stats.lazy_queries), (1, 0));
        }

        /// Per-partition, streaming, batch and cascade searches never use the
        /// lazy scan.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_leaves_other_search_paths_unchanged() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let ivf = lazy_index(&index);
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(3), 50, 8);
            let (partitions, dists) = index.find_partitions(&query).unwrap();
            let partitions = Arc::new(partitions);
            let dists = Arc::new(dists);
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let mut cascade = query.clone();
            cascade.rq_cascade_factor = Some(4);

            let mut outputs = Vec::new();
            for config in [
                LayeredLazyConfig::default(),
                LayeredLazyConfig {
                    enabled: true,
                    ..Default::default()
                },
            ] {
                ivf.set_layered_lazy_config_for_test(config);
                layered_stats::snapshot_and_reset();
                let in_partition = index
                    .search_in_partition(
                        partitions.value(0) as usize,
                        &query,
                        filter.clone(),
                        &NoOpMetricsCollector,
                    )
                    .await
                    .unwrap();
                let streamed = index
                    .clone()
                    .search_partitions(
                        query.clone(),
                        partitions.clone(),
                        dists.clone(),
                        0,
                        partitions.len(),
                        filter.clone(),
                        Some(Arc::new(NeverStop)),
                        Arc::new(NoOpMetricsCollector),
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                let batched = index
                    .clone()
                    .search_partitions_batch(
                        query.clone(),
                        vec![partitions.clone()],
                        vec![dists.clone()],
                        filter.clone(),
                        Arc::new(NoOpMetricsCollector),
                    )
                    .await
                    .unwrap();
                let cascaded = search_global(&index, &cascade, filter.clone())
                    .await
                    .unwrap();
                let stats = layered_stats::snapshot_and_reset();
                assert_eq!(stats.lazy_queries, 0);
                assert_eq!(stats.lazy_all_resident_skips, 0);
                assert_lazy_chain_timing(&stats, "other search paths");
                let mut output = vec![result_bits(&in_partition), result_bits(&cascaded)];
                output.extend(streamed.iter().map(result_bits));
                output.extend(batched.iter().map(result_bits));
                outputs.push(output);
            }
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            assert_eq!(outputs[0], outputs[1]);
        }

        /// Concurrent queries share the promotion tracker, a cleared cache
        /// and a missing persistent entry fall back to the origin file, and a
        /// dropped query releases what it holds.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_lifecycle() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Ungated).await;
            let tiered = tiered.unwrap();
            let ivf = lazy_index(&index);
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let config = LayeredLazyConfig {
                enabled: true,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Background { reads: 1 },
                // Gathers the persistent tier cannot serve read origin rows.
                origin_max_runs: usize::MAX,
                ..Default::default()
            };
            let queries: Vec<Query> = (0..4)
                .map(|row| lazy_test_query(vectors.value(row * 101), 10, LAZY_PARTITIONS))
                .collect();

            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::new();
            for query in &queries {
                expected.push(result_bits(
                    &search_global(&index, query, filter.clone()).await.unwrap(),
                ));
            }

            // Identical queries in parallel promote each plane at most once at a time.
            ivf.set_layered_lazy_config_for_test(config);
            layered_stats::snapshot_and_reset();
            let concurrent = futures::future::try_join_all(
                queries
                    .iter()
                    .chain(queries.iter())
                    .map(|query| search_global(&index, query, filter.clone())),
            )
            .await
            .unwrap();
            for (position, result) in concurrent.iter().enumerate() {
                assert_eq!(result_bits(result), expected[position % queries.len()]);
            }
            wait_for_promotions().await;
            let stats = layered_stats::snapshot_and_reset();
            assert!(stats.promotions_issued > 0);
            // A promoted plane the small RAM tier evicted again is not completed.
            assert!(stats.promotions_completed > 0, "{stats:?}");
            assert!(
                stats.promotions_completed <= stats.promotions_issued,
                "{stats:?}"
            );

            // Without persistent rows the gathers read the origin file. The
            // promotions above may have left every plane these queries need
            // resident, which would skip the gathers, so empty the RAM tier.
            tiered.ram.clear().await;
            tiered.serve_rows.store(false, Ordering::Relaxed);
            for (query, expected) in queries.iter().zip(&expected) {
                let result = search_global(&index, query, filter.clone()).await.unwrap();
                assert_eq!(&result_bits(&result), expected);
            }
            wait_for_promotions().await;
            let stats = layered_stats::snapshot_and_reset();
            assert!(stats.origin_row_reads > 0, "{stats:?}");
            tiered.serve_rows.store(true, Ordering::Relaxed);

            // Clearing the cache between queries loses every plane entry.
            tiered.clear().await;
            for (query, expected) in queries.iter().zip(&expected) {
                let result = search_global(&index, query, filter.clone()).await.unwrap();
                assert_eq!(&result_bits(&result), expected);
            }
            wait_for_promotions().await;
            let stats = layered_stats::snapshot_and_reset();
            assert!(stats.origin_row_reads > 0, "{stats:?}");

            // A dropped query stops its producer and releases its partitions.
            for timeout in [1, 5, 50] {
                let _ = tokio::time::timeout(
                    Duration::from_micros(timeout),
                    search_global(&index, &queries[0], filter.clone()),
                )
                .await;
            }
            wait_for_promotions().await;
            for _ in 0..500 {
                if ivf.prepared_partitions().in_flight() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(ivf.prepared_partitions().in_flight(), 0);
            let result = search_global(&index, &queries[0], filter.clone())
                .await
                .unwrap();
            assert_eq!(result_bits(&result), expected[0]);
            wait_for_promotions().await;
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            drop(dataset);
        }

        /// A staged probe the producer is not polling must not hold a slot of
        /// a fair cache resource that the head gather waits for.
        ///
        /// Every RAM admission of the backend waits for a slot in a bounded
        /// spill channel drained slowly, like a tiered cache whose admissions
        /// queue their evictions for the disk. Staging steps queue there for
        /// sign planes ahead of the gathers' whole ex planes. With one step
        /// window the gather buffer holds three probes, and while its head is
        /// pending the producer stops polling the staging buffer. When the
        /// steps were plain futures in those buffers, the channel handed the
        /// next freed slot to its oldest waiter, a staging step nobody polled,
        /// which never used it: the head gather and every later admission then
        /// waited forever. Steps now run as tasks and complete on their own,
        /// whether gathers before the heap fills are deferred or issued at
        /// once, and when staging loads predicted-dense probes for the eager
        /// scan instead (every probe here, as whole gathers predict them dense).
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_unpolled_staging_cannot_hold_spill_slot() {
            const PROBES: usize = 32;
            const QUERIES: usize = 16;
            // Staged probes in flight; the deadlock needs staging steps
            // queued on the spill channel behind the ones that fill the
            // gather buffer, whatever the host's core count.
            const STAGING_STEPS: usize = 8;
            const QUERY_TIMEOUT: Duration = Duration::from_secs(10);
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Tiny).await;
            let tiered = tiered.unwrap();
            let ivf = lazy_index(&index);
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            // A quarter of the probed rows: the first probes are gathered
            // before the heap fills, the later ones gate on its threshold.
            let k = LAZY_ROWS * PROBES / LAZY_PARTITIONS / 4;
            let queries: Vec<Query> = (0..QUERIES)
                .map(|row| lazy_test_query(vectors.value(row * 211), k, PROBES))
                .collect();

            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::with_capacity(QUERIES);
            for query in &queries {
                expected.push(result_bits(
                    &search_global(&index, query, filter.clone()).await.unwrap(),
                ));
            }

            let (spill_tx, mut spill_rx) = tokio::sync::mpsc::channel::<()>(1);
            let drainer = tokio::spawn(async move {
                while spill_rx.recv().await.is_some() {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            });
            tiered.set_spill(Some(spill_tx));
            ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
            for (eager_before_full, dense_to_eager) in [
                (false, DenseToEager::Off),
                (true, DenseToEager::Off),
                (true, DenseToEager::All),
            ] {
                ivf.set_layered_lazy_config_for_test(LayeredLazyConfig {
                    enabled: true,
                    window: 1,
                    dense: DenseGatherMode::Whole,
                    promote: LazyPromotion::Off,
                    eager_before_full,
                    dense_to_eager,
                    ..Default::default()
                });
                let setting = format!(
                    "eager_before_full={eager_before_full} dense_to_eager={dense_to_eager}"
                );
                layered_stats::snapshot_and_reset();
                for (position, (query, expected)) in queries.iter().zip(&expected).enumerate() {
                    let result = tokio::time::timeout(
                        QUERY_TIMEOUT,
                        search_global(&index, query, filter.clone()),
                    )
                    .await
                    .unwrap_or_else(|_| {
                        panic!(
                            "lazy query {position} ({setting}) did not finish within {QUERY_TIMEOUT:?}"
                        )
                    })
                    .unwrap();
                    assert_eq!(
                        &result_bits(&result),
                        expected,
                        "query {position} {setting}"
                    );
                }
                let stats = layered_stats::snapshot_and_reset();
                assert_eq!(stats.lazy_queries, QUERIES as u64, "{setting} {stats:?}");
                assert_eq!(stats.needed_not_fetched, 0, "{setting} {stats:?}");
                let whole_planes: u64 = stats.high_whole.iter().chain(&stats.low_whole).sum();
                let eager_loads: u64 = stats.dense_to_eager.iter().sum();
                if dense_to_eager == DenseToEager::All {
                    assert!(
                        eager_loads > 0 && whole_planes == 0,
                        "staging must load every probe eagerly: {setting} {stats:?}"
                    );
                } else {
                    assert!(
                        eager_loads == 0 && whole_planes > 0,
                        "gathers must admit whole ex planes: {setting} {stats:?}"
                    );
                }
            }
            tiered.set_spill(None);
            drainer.abort();
            ivf.set_lazy_prepare_parallelism_for_test(0);
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
        }

        /// Sums of the lazy counters that tell routing, issue and origin
        /// policies apart.
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
        struct IssueTotals {
            dense_to_eager: [u64; RANK_BUCKETS],
            certain_dense: [u64; RANK_BUCKETS],
            lazy_probes: u64,
            empty: u64,
            eager_before_full: u64,
            deferred_issues: u64,
            serial_waits: u64,
            origin_whole_fallbacks: u64,
            origin_row_reads: u64,
            sparse_planes: u64,
            whole_planes: u64,
            rank0_scored_queries: u64,
            first_gather_issue_queries: u64,
            release_wait_ns: u64,
        }

        impl IssueTotals {
            fn add(&mut self, stats: &LayeredLazyStats) {
                for (totals, counts) in [
                    (&mut self.dense_to_eager, stats.dense_to_eager),
                    (&mut self.certain_dense, stats.certain_dense),
                ] {
                    for (total, count) in totals.iter_mut().zip(counts) {
                        *total += count;
                    }
                }
                self.lazy_probes += stats.lazy_probes.iter().sum::<u64>();
                self.empty += stats.empty.iter().sum::<u64>();
                self.eager_before_full += stats.eager_before_full;
                self.deferred_issues += stats.deferred_issues;
                self.serial_waits += stats.serial_waits;
                self.origin_whole_fallbacks += stats.origin_whole_fallbacks;
                self.origin_row_reads += stats.origin_row_reads;
                self.sparse_planes += stats
                    .high_sparse
                    .iter()
                    .chain(&stats.low_sparse)
                    .sum::<u64>();
                self.whole_planes += stats.high_whole.iter().chain(&stats.low_whole).sum::<u64>();
                self.rank0_scored_queries += stats.lazy_rank0_scored_queries;
                self.first_gather_issue_queries += stats.lazy_first_gather_issue_queries;
                self.release_wait_ns += stats.lazy_release_wait_ns;
            }
        }

        /// Issuing gathers before the heap fills, and loading whole planes
        /// instead of many origin row runs, change only what is read: results
        /// match the eager scan with either switched on or off, including `k`
        /// beyond the rows of the first probes and backends that serve no
        /// persistent rows. Backends that gate plane admission (Lance's
        /// default cache included) never fall back to whole planes. Dense
        /// probes stay in the lazy pipeline here, so every probe is gathered.
        #[rstest]
        #[case::origin(LazyTestCache::Origin, true)]
        #[case::small(LazyTestCache::Small, true)]
        #[case::gated(LazyTestCache::Gated, true)]
        #[case::tiny_without_rows(LazyTestCache::Tiny, false)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_eager_before_full_matches_eager(
            #[case] cache: LazyTestCache,
            #[case] serve_rows: bool,
        ) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, tiered) = open_lazy_test_index(dir.as_str(), cache).await;
            if let Some(tiered) = &tiered {
                tiered.serve_rows.store(serve_rows, Ordering::Relaxed);
            }
            let origin_only = tiered.is_none() || !serve_rows;
            let gated = lazy_index(&index).index_cache.plane_admission_gated();
            let vectors = batch["vector"].as_fixed_size_list();
            let filters = lazy_test_filters();
            let mut totals: HashMap<(bool, usize), IssueTotals> = HashMap::new();
            let mut case = 0usize;
            for row in [0, 777] {
                for nprobes in [8, LAZY_PARTITIONS] {
                    for k in [10, 1000, 5000, LAZY_ROWS + 1] {
                        for eager_before_full in [true, false] {
                            for origin_max_runs in [0, 2, usize::MAX] {
                                for dense in [DenseGatherMode::Cost, DenseGatherMode::Sparse] {
                                    case += 1;
                                    let (filter_name, filter) = &filters[case % filters.len()];
                                    let config = LayeredLazyConfig {
                                        enabled: true,
                                        dense,
                                        eager_before_full,
                                        origin_max_runs,
                                        dense_to_eager: DenseToEager::Off,
                                        ..Default::default()
                                    };
                                    let query = lazy_test_query(vectors.value(row), k, nprobes);
                                    let context = format!(
                                        "cache={cache:?} serve_rows={serve_rows} row={row} nprobes={nprobes} k={k} filter={filter_name} config={config:?}"
                                    );
                                    let stats = assert_lazy_matches_eager(
                                        &index, &query, filter, config, &context,
                                    )
                                    .await;
                                    if !eager_before_full {
                                        assert_eq!(stats.eager_before_full, 0, "{context}");
                                    }
                                    if origin_max_runs == usize::MAX || gated {
                                        assert_eq!(stats.origin_whole_fallbacks, 0, "{context}");
                                    }
                                    totals
                                        .entry((eager_before_full, origin_max_runs))
                                        .or_default()
                                        .add(&stats);
                                }
                            }
                        }
                    }
                }
            }
            wait_for_promotions().await;
            let on_uncapped = totals[&(true, usize::MAX)];
            assert!(on_uncapped.eager_before_full > 0, "{totals:?}");
            if origin_only && !gated {
                // The origin run cap applies with either issue policy.
                for eager_before_full in [true, false] {
                    let capped = totals[&(eager_before_full, 0)];
                    assert!(capped.origin_whole_fallbacks > 0, "{totals:?}");
                    assert_eq!(capped.origin_row_reads, 0, "{totals:?}");
                    assert!(
                        totals[&(eager_before_full, usize::MAX)].origin_row_reads > 0,
                        "{totals:?}"
                    );
                }
            }
        }

        /// Loading and scoring predicted-dense probes with the eager scan
        /// changes only how they are read: results match the eager scan with
        /// the routing off, on for every such probe (`all`) or on for those
        /// that read a plane from a high-latency origin (`origin`) of either
        /// class, for `k` within the first probe and beyond the rows of the
        /// first probes, with prefilters that keep most or few rows and with
        /// distance bounds, whether the planes come from the origin, an
        /// ungated persistent tier or a tier that gates admission. Only
        /// queries without a prefilter or upper bound are routed: those drop
        /// rows of every probe before its gather, even at `k` beyond every
        /// probed row, so their probes stay in the lazy pipeline. `origin`
        /// routes no probe on a low-latency origin, and on a high-latency one
        /// only probes with a plane on no cache tier: every probe `all`
        /// routes when nothing is cached, none once the persistent tier holds
        /// every plane.
        #[rstest]
        #[case::origin(LazyTestCache::Origin)]
        #[case::ungated(LazyTestCache::Ungated)]
        #[case::gated(LazyTestCache::Gated)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_dense_to_eager_matches_eager(#[case] cache: LazyTestCache) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) = open_lazy_test_index(dir.as_str(), cache).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let filters: Vec<_> = lazy_test_filters()
                .into_iter()
                .filter(|(name, _)| matches!(*name, "none" | "sparse" | "dense"))
                .collect();
            let mut cases = Vec::new();
            for row in [0, 777] {
                for nprobes in [1, 8, LAZY_PARTITIONS] {
                    for k in [10, 1000, LAZY_ROWS + 1] {
                        for (filter_name, filter) in &filters {
                            let query = lazy_test_query(vectors.value(row), k, nprobes);
                            let eager =
                                search_global(&index, &query, filter.clone()).await.unwrap();
                            let bounded = with_bounds(&query, &eager);
                            for query in std::iter::once(query).chain(bounded) {
                                let label = format!(
                                    "cache={cache:?} row={row} nprobes={nprobes} k={k} filter={filter_name} bounds={}",
                                    query.upper_bound.is_some()
                                );
                                cases.push((label, *filter_name == "none", query, filter.clone()));
                            }
                        }
                    }
                }
            }

            let (store, index_dir) = index_files(&dataset).await;
            let (low, high) = (OriginLatencyClass::Low, OriginLatencyClass::High);
            let low_index =
                open_with_origin_latency(&dataset, store.clone(), index_dir.clone(), low).await;
            let high_index = open_with_origin_latency(&dataset, store, index_dir, high).await;
            // `off` and `all` do not depend on the class.
            let settings = [
                (DenseToEager::Off, low),
                (DenseToEager::All, low),
                (DenseToEager::Origin, low),
                (DenseToEager::Origin, high),
            ];
            let mut routed = [0u64; 4];
            for (label, unfiltered, query, filter) in &cases {
                let mut routed_by_all = None;
                for (position, &(mode, class)) in settings.iter().enumerate() {
                    let index = match class {
                        OriginLatencyClass::Low => &low_index,
                        OriginLatencyClass::High => &high_index,
                    };
                    let config = LayeredLazyConfig {
                        enabled: true,
                        dense_to_eager: mode,
                        ..Default::default()
                    };
                    let context = format!("{label} dense_to_eager={mode} class={class}");
                    let stats =
                        assert_lazy_matches_eager(index, query, filter, config, &context).await;
                    let context = format!("{context} {stats:?}");
                    let probes_routed: u64 = stats.dense_to_eager.iter().sum();
                    let slow_probes: u64 = stats.s3_bound_probes.iter().sum();
                    if !unfiltered || query.upper_bound.is_some() {
                        assert_eq!(probes_routed, 0, "{context}");
                    }
                    if class == low {
                        assert_eq!(slow_probes, 0, "{context}");
                    }
                    // Every probe `origin` routes reads a plane from the origin.
                    if mode == DenseToEager::Origin {
                        let mut buckets = stats.dense_to_eager.iter().zip(&stats.s3_bound_probes);
                        assert!(buckets.all(|(routed, slow)| routed <= slow), "{context}");
                    }
                    if mode == DenseToEager::All {
                        routed_by_all = Some(stats.dense_to_eager);
                    } else if class == high && stats.lazy_queries == 1 {
                        match cache {
                            // Nothing is cached: every staged probe reads the origin.
                            LazyTestCache::Origin => {
                                let empty_probes: u64 = stats.empty.iter().sum();
                                assert_eq!(
                                    slow_probes + empty_probes,
                                    query.minimum_nprobes as u64,
                                    "{context}"
                                );
                                assert_eq!(Some(stats.dense_to_eager), routed_by_all, "{context}");
                            }
                            // Prewarm persisted every plane.
                            LazyTestCache::Ungated => {
                                assert_eq!((probes_routed, slow_probes), (0, 0), "{context}");
                            }
                            _ => {}
                        }
                    }
                    routed[position] += probes_routed;
                }
            }
            wait_for_promotions().await;
            let [off, all, origin_low, origin_high] = routed;
            assert_eq!(off, 0, "switched off, no probe is routed");
            assert!(all > 0, "{routed:?}");
            assert_eq!(origin_low, 0, "a low-latency origin routes no probe");
            match cache {
                LazyTestCache::Origin => assert_eq!(origin_high, all, "{routed:?}"),
                LazyTestCache::Ungated => assert_eq!(origin_high, 0, "{routed:?}"),
                _ => {}
            }
        }

        /// `count` queries for `k` rows over `nprobes` probes, keyed by rows
        /// of `vectors`, whose first probe alone holds more than `2 * k` rows:
        /// it fills the heap, and the threshold then keeps under half of the
        /// rows scored, so later gathers are not expected to be dense.
        fn queries_filled_by_first_probe(
            index: &Arc<dyn VectorIndex>,
            vectors: &FixedSizeListArray,
            k: usize,
            nprobes: usize,
            count: usize,
        ) -> Vec<Query> {
            let queries: Vec<Query> = (0..vectors.len())
                .step_by(211)
                .map(|row| lazy_test_query(vectors.value(row), k, nprobes))
                .filter(|query| {
                    let (partitions, _) = index.find_partitions(query).unwrap();
                    lazy_index(index)
                        .storage
                        .partition_size(partitions.value(0) as usize)
                        > 2 * k
                })
                .take(count)
                .collect();
            assert_eq!(queries.len(), count);
            queries
        }

        /// Lazy settings without promotions, with the given issue and routing
        /// policies.
        fn issue_policy(
            eager_before_full: bool,
            dense_to_eager: DenseToEager,
        ) -> LayeredLazyConfig {
            LayeredLazyConfig {
                enabled: true,
                promote: LazyPromotion::Off,
                eager_before_full,
                dense_to_eager,
                ..Default::default()
            }
        }

        /// Issue counters of the lazy scan under each of `configs`, summed
        /// over `queries`.
        async fn lazy_issue_totals<const N: usize>(
            index: &Arc<dyn VectorIndex>,
            queries: &[Query],
            filter: &Arc<dyn PreFilter>,
            configs: [LayeredLazyConfig; N],
        ) -> [IssueTotals; N] {
            let mut totals = [IssueTotals::default(); N];
            for (config, totals) in configs.into_iter().zip(&mut totals) {
                for (position, query) in queries.iter().enumerate() {
                    let context = format!(
                        "eager_before_full={} dense_to_eager={} query={position} k={}",
                        config.eager_before_full, config.dense_to_eager, query.k
                    );
                    let stats =
                        assert_lazy_matches_eager(index, query, filter, config, &context).await;
                    assert_eq!(stats.lazy_queries, 1, "{context}");
                    totals.add(&stats);
                }
            }
            totals
        }

        /// Which probes the eager scan loads, and whether lazy gathers may be
        /// issued before the heap fills, depend on `k`, not on how staging
        /// races the first probes.
        ///
        /// When `k` spans several probes, the gathers issued once the heap
        /// fills would read whole planes anyway, so with every predicted-dense
        /// probe routed (`all`) every probe is loaded and scored by the eager
        /// scan and no ex plane is gathered whole. With the routing off, the
        /// gathers read the planes whole and are issued at once, and none
        /// waits for the heap or its turn.
        ///
        /// When the first probe alone fills the heap (small `k`), only that
        /// probe, scored with an infinite threshold, goes to the eager scan,
        /// exactly the probe the lazy pipeline otherwise gathers as certain
        /// dense; later gathers wait for its finite threshold, as the
        /// eager-before-full ablation always does.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_issue_policy_depends_on_k() {
            const PROBES: usize = 32;
            const QUERIES: usize = 8;
            const STAGING_STEPS: usize = 8;
            const SMALL_K: usize = 10;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let ivf = lazy_index(&index);
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            // A quarter of the probed rows, several probes' worth.
            let large_k = LAZY_ROWS * PROBES / LAZY_PARTITIONS / 4;
            let queries = |k: usize| -> Vec<Query> {
                (0..QUERIES)
                    .map(|row| lazy_test_query(vectors.value(row * 211), k, PROBES))
                    .collect()
            };
            let small = queries_filled_by_first_probe(&index, vectors, SMALL_K, PROBES, QUERIES);
            let configs = [
                issue_policy(true, DenseToEager::All),
                issue_policy(true, DenseToEager::Off),
                issue_policy(false, DenseToEager::Off),
            ];
            ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
            let [large_routed, large_on, large_off] =
                lazy_issue_totals(&index, &queries(large_k), &filter, configs).await;
            let [small_routed, small_on, small_off] =
                lazy_issue_totals(&index, &small, &filter, configs).await;
            ivf.set_lazy_prepare_parallelism_for_test(0);

            // Nothing is resident, so every probe of a non-empty partition
            // goes to the eager scan, in every rank bucket the probes reach.
            let probes = (QUERIES * PROBES) as u64;
            let routed: u64 = large_routed.dense_to_eager.iter().sum();
            assert_eq!(routed + large_routed.empty, probes, "{large_routed:?}");
            assert!(
                large_routed.dense_to_eager[..4]
                    .iter()
                    .all(|&count| count > 0),
                "{large_routed:?}"
            );
            assert_eq!(large_routed.dense_to_eager[4], 0, "{large_routed:?}");
            assert_eq!(
                (
                    large_routed.lazy_probes,
                    large_routed.whole_planes,
                    large_routed.eager_before_full,
                    large_routed.deferred_issues,
                ),
                (0, 0, 0, 0),
                "{large_routed:?}"
            );
            assert_eq!(large_routed.certain_dense, [0; RANK_BUCKETS]);
            for large in [large_on, large_off] {
                assert_eq!(large.dense_to_eager, [0; RANK_BUCKETS], "{large:?}");
                assert_eq!(large.lazy_probes + large.empty, probes, "{large:?}");
                assert!(large.whole_planes > 0, "{large:?}");
                assert!(large.certain_dense[0] > 0, "{large:?}");
            }
            assert_eq!(
                (large_on.deferred_issues, large_on.serial_waits),
                (0, 0),
                "{large_on:?}"
            );
            assert!(large_on.eager_before_full > 0, "{large_on:?}");
            assert_eq!(large_off.eager_before_full, 0, "{large_off:?}");
            assert!(large_off.deferred_issues > 0, "{large_off:?}");

            let mut first_probe = [0; RANK_BUCKETS];
            first_probe[0] = QUERIES as u64;
            assert_eq!(small_routed.dense_to_eager, first_probe, "{small_routed:?}");
            assert_eq!(small_routed.certain_dense, [0; RANK_BUCKETS]);
            assert!(small_routed.lazy_probes > 0, "{small_routed:?}");
            for small in [small_on, small_off] {
                assert_eq!(small.dense_to_eager, [0; RANK_BUCKETS], "{small:?}");
                assert_eq!(small.certain_dense, first_probe, "{small:?}");
                assert!(small.deferred_issues > 0, "{small:?}");
            }
            for small in [small_routed, small_on, small_off] {
                assert_eq!(
                    (small.eager_before_full, small.serial_waits),
                    (0, 0),
                    "{small:?}"
                );
            }

            // Every scan times its first probe's scoring, and a wait for the
            // threshold or a turn is timed exactly when one happens. Each
            // small-`k` query gathers later probes, whose issue waits on the
            // first probe's threshold (see `assert_lazy_chain_timing`),
            // whether that probe is routed or gathered as certain dense; the
            // routed large-`k` queries gather nothing.
            let queries = QUERIES as u64;
            for totals in [
                large_routed,
                large_on,
                large_off,
                small_routed,
                small_on,
                small_off,
            ] {
                assert_eq!(totals.rank0_scored_queries, queries, "{totals:?}");
                assert_eq!(
                    totals.release_wait_ns > 0,
                    totals.deferred_issues > 0,
                    "{totals:?}"
                );
            }
            assert_eq!(
                large_routed.first_gather_issue_queries, 0,
                "{large_routed:?}"
            );
            for small in [small_routed, small_on, small_off] {
                assert_eq!(small.first_gather_issue_queries, queries, "{small:?}");
            }
        }

        /// A selective prefilter keeps the heap from filling after the probes
        /// that hold `k` rows. Deferred gathers are then issued at once
        /// instead of one probe at a time, with identical results; the
        /// ablation waits for each probe's turn.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_eager_before_full_when_filters_delay_the_heap() {
            const PROBES: usize = 16;
            const QUERIES: usize = 8;
            const STAGING_STEPS: usize = 8;
            const K: usize = 100;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let ivf = lazy_index(&index);
            let vectors = batch["vector"].as_fixed_size_list();
            let (_, filter) = lazy_test_filters()
                .into_iter()
                .find(|(name, _)| *name == "sparse")
                .unwrap();
            // The first probe alone holds `k` rows, too few of which the
            // filter accepts.
            let queries = queries_filled_by_first_probe(&index, vectors, K, PROBES, QUERIES);
            ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
            let configs = [
                issue_policy(true, DenseToEager::Off),
                issue_policy(false, DenseToEager::Off),
            ];
            let [on, off] = lazy_issue_totals(&index, &queries, &filter, configs).await;
            ivf.set_lazy_prepare_parallelism_for_test(0);
            assert!(on.eager_before_full > 0, "{on:?}");
            assert!(on.serial_waits < off.serial_waits, "{on:?} {off:?}");
            assert_eq!(off.eager_before_full, 0, "{off:?}");
        }

        /// A sparse gather the persistent tier cannot serve loads the whole
        /// plane once its origin row runs exceed the cap, whether gathers
        /// before the heap fills are deferred or not, with identical results.
        /// Without a cap, or on a backend that gates plane admission (which
        /// may not admit the plane; Lance's default cache is one), it reads
        /// the rows.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_origin_runs_fall_back_to_whole_plane() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            for cache in [
                LazyTestCache::Origin,
                LazyTestCache::Ungated,
                LazyTestCache::Gated,
            ] {
                let (_dataset, index, tiered) = open_lazy_test_index(dir.as_str(), cache).await;
                if let Some(tiered) = &tiered {
                    // Every sparse gather that misses RAM goes to the origin.
                    tiered.serve_rows.store(false, Ordering::Relaxed);
                }
                let gated = lazy_index(&index).index_cache.plane_admission_gated();
                assert_eq!(gated, cache != LazyTestCache::Ungated, "{cache:?}");
                for eager_before_full in [true, false] {
                    for origin_max_runs in [0, usize::MAX] {
                        let config = LayeredLazyConfig {
                            enabled: true,
                            dense: DenseGatherMode::Sparse,
                            promote: LazyPromotion::Off,
                            eager_before_full,
                            origin_max_runs,
                            ..Default::default()
                        };
                        let mut totals = IssueTotals::default();
                        for row in 0..4 {
                            let query =
                                lazy_test_query(vectors.value(row * 101), 100, LAZY_PARTITIONS);
                            let context = format!("{cache:?} {config:?} row={row}");
                            totals.add(
                                &assert_lazy_matches_eager(
                                    &index, &query, &filter, config, &context,
                                )
                                .await,
                            );
                        }
                        let context = format!("{cache:?} {config:?} {totals:?}");
                        if origin_max_runs == 0 && !gated {
                            // Every origin gather takes at least one run.
                            assert!(totals.origin_whole_fallbacks > 0, "{context}");
                            assert_eq!(
                                (totals.origin_row_reads, totals.sparse_planes),
                                (0, 0),
                                "{context}"
                            );
                        } else {
                            assert_eq!(totals.origin_whole_fallbacks, 0, "{context}");
                            assert!(totals.origin_row_reads > 0, "{context}");
                        }
                    }
                }
            }
        }

        /// A backend that admits planes like any entry loads a missed
        /// partition's three planes at once, so it pays one round trip; a
        /// gated backend admits the sign plane before it starts the ex planes,
        /// whose admission depends on it. Results are identical.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_plane_loads_overlap_unless_gated() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let key = batch["vector"].as_fixed_size_list().value(17);
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let single = lazy_test_query(key.clone(), 10, 1);
            let every = lazy_test_query(key, 100, LAZY_PARTITIONS);
            let (_resident_dataset, resident, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let expected = [
                result_bits(
                    &search_global(&resident, &single, filter.clone())
                        .await
                        .unwrap(),
                ),
                result_bits(
                    &search_global(&resident, &every, filter.clone())
                        .await
                        .unwrap(),
                ),
            ];
            for cache in [LazyTestCache::ColdUngated, LazyTestCache::ColdGated] {
                let (_dataset, index, tiered) = open_lazy_test_index(dir.as_str(), cache).await;
                let tiered = tiered.unwrap();
                tiered.record_plane_loads();
                let result = search_global(&index, &single, filter.clone())
                    .await
                    .unwrap();
                let loads = tiered.take_plane_loads();
                assert_eq!(result_bits(&result), expected[0], "{cache:?}");
                assert_eq!(loads.len(), 6, "{cache:?} {loads:?}");
                let position =
                    |load: PlaneLoad| loads.iter().position(|&seen| seen == load).unwrap();
                let sign_loaded = position(PlaneLoad::Finished(0));
                for plane in [1, 2] {
                    let started = position(PlaneLoad::Started(plane));
                    if cache == LazyTestCache::ColdGated {
                        assert!(sign_loaded < started, "{cache:?} {loads:?}");
                    } else {
                        assert!(started < sign_loaded, "{cache:?} {loads:?}");
                    }
                }
                let result = search_global(&index, &every, filter.clone()).await.unwrap();
                assert_eq!(result_bits(&result), expected[1], "{cache:?}");
            }
        }

        /// Rows per page of a storage file that [`rewrite_rq_storage`] cuts
        /// into small pages: prime, so page ends fall inside partitions.
        const RESIDENT_TEST_PAGE_ROWS: usize = 97;
        /// First reads issued at once against a storage whose resident
        /// columns have not loaded.
        const RESIDENT_TEST_CONCURRENT_READS: usize = 32;

        /// A reader of `source`'s storage file rewritten as a `version` file:
        /// in pages of `page_rows` rows when set, so that partitions straddle
        /// pages, or else in the writer's default pages, as the index writer
        /// lays it out.
        async fn rewrite_rq_storage(
            source: &IvfQuantizationStorage<RabitQuantizer>,
            version: LanceFileVersion,
            page_rows: Option<usize>,
        ) -> FileReader {
            let schema = source.schema();
            let batches = source
                .reader()
                .read_stream(
                    ReadBatchParams::RangeFull,
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let batch = concat_batches(&schema, batches.iter()).unwrap();
            let store = Arc::new(ObjectStore::memory());
            let path = object_store::path::Path::from("resident/storage.lance");
            // Without a cache, every write flushes a page of each column.
            let options = FileWriterOptions {
                data_cache_bytes: page_rows.map(|_| 1),
                ..Default::default()
            };
            let mut writer = lance_file::versions::create_writer(
                version.resolve(),
                store.create(&path).await.unwrap(),
                lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
                options,
            )
            .unwrap();
            let rows = batch.num_rows();
            let step = page_rows.unwrap_or(rows).max(1);
            for offset in (0..rows).step_by(step) {
                let slice = batch.slice(offset, step.min(rows - offset));
                writer.write_batch(&slice).await.unwrap();
            }
            writer.finish().await.unwrap();
            let scheduler = ScanScheduler::new(store, SchedulerConfig::default_for_testing());
            let file = scheduler
                .open_file(&path, &CachedFileSize::unknown())
                .await
                .unwrap();
            FileReader::try_open(
                file,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await
            .unwrap()
        }

        /// A storage over `reader`, a file of `source`'s rows, that takes the
        /// small columns from its resident store when `resident`.
        fn rq_storage_over(
            source: &IvfQuantizationStorage<RabitQuantizer>,
            reader: &FileReader,
            resident: bool,
        ) -> IvfQuantizationStorage<RabitQuantizer> {
            IvfQuantizationStorage::<RabitQuantizer>::from_cached_with_remapper(
                reader.clone(),
                source.ivf().clone(),
                source.metadata().clone(),
                source.distance_type(),
                None,
            )
            .with_sign_bounds(source.sign_bounds())
            .with_resident_columns_enabled(resident)
        }

        /// Whether the file rows `range` span more than one page of the first
        /// column of `reader`'s file.
        fn straddles_page(reader: &FileReader, range: Range<usize>) -> bool {
            let pages = &reader.metadata().column_infos[0].page_infos;
            let mut end = 0;
            pages.iter().any(|page| {
                end += page.num_rows as usize;
                range.start < end && end < range.end
            })
        }

        /// Ascending offsets into a partition of `rows` rows, some in runs and
        /// some alone, as a sparse gather reads them.
        fn resident_test_rows(rows: usize) -> Vec<u32> {
            (0..rows as u32)
                .filter(|row| row % 5 != 3 && row % 7 != 1)
                .collect()
        }

        /// Open the index of `dataset` over the dataset's caches, with its
        /// small columns resident or read from the file.
        async fn open_with_resident_columns(
            dataset: &Dataset,
            resident: bool,
        ) -> Arc<dyn VectorIndex> {
            let index = dataset.load_indices().await.unwrap()[0].clone();
            let ivf = IvfRq::try_new(
                dataset.object_store.clone(),
                dataset.indice_files_dir(&index).unwrap(),
                index.uuid,
                None,
                &dataset.metadata_cache,
                dataset.index_cache.for_index(&index.uuid, None),
                index.file_size_map(),
                IvfOpenContext::default(),
            )
            .await
            .unwrap()
            .with_resident_columns_for_test(resident);
            Arc::new(ivf)
        }

        /// Reads that take the small columns from the resident store return
        /// the batches that reads of the file return, bit for bit: whole
        /// partitions and planes, and planes at selected rows, of every
        /// partition, empty ones included, in the writer's pages and in pages
        /// that partitions straddle, in either file version. They are charged
        /// the same bytes, except that selected rows of a v2.0 file may be
        /// charged less.
        #[rstest]
        #[case::native_v2_0(false, LanceFileVersion::V2_0)]
        #[case::native_v2_2(false, LanceFileVersion::V2_2)]
        #[case::layered_v2_0(true, LanceFileVersion::V2_0)]
        #[case::layered_v2_2(true, LanceFileVersion::V2_2)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_columns_reads_match_file_reads(
            #[case] layered: bool,
            #[case] version: LanceFileVersion,
        ) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let source = &lazy_index(&index).storage;
            // A native file has the sign plane's columns too.
            let mut planes = vec![0];
            if layered {
                planes.extend([1, 2]);
                if source.sign_bounds() == SignBounds::Lazy {
                    planes.push(SIGN_BOUNDS_PLANE);
                }
            }
            for page_rows in [None, Some(RESIDENT_TEST_PAGE_ROWS)] {
                let reader = rewrite_rq_storage(source, version, page_rows).await;
                let file = rq_storage_over(source, &reader, false);
                let resident = rq_storage_over(source, &reader, true);
                let straddled = (0..source.num_partitions())
                    .any(|partition| straddles_page(&reader, source.ivf().row_range(partition)));
                assert_eq!(straddled, page_rows.is_some(), "pages={page_rows:?}");
                for partition in 0..source.num_partitions() {
                    let context = format!(
                        "layered={layered} {version:?} pages={page_rows:?} partition={partition}"
                    );
                    let expected = file.load_partition(partition, None).await.unwrap();
                    let actual = resident.load_partition(partition, None).await.unwrap();
                    assert_eq!(
                        actual.to_batches().unwrap().collect::<Vec<_>>(),
                        expected.to_batches().unwrap().collect::<Vec<_>>(),
                        "{context}"
                    );
                    assert_eq!(actual.deep_size_of(), expected.deep_size_of(), "{context}");
                    let rows = resident_test_rows(source.partition_size(partition));
                    for &plane in &planes {
                        for rows in [None, Some(rows.clone())] {
                            let sparse = rows.is_some();
                            let context = format!("{context} plane={plane} sparse={sparse}");
                            let expected = file
                                .read_plane(partition, plane, rows.clone(), None)
                                .await
                                .unwrap();
                            let actual = resident
                                .read_plane(partition, plane, rows, None)
                                .await
                                .unwrap();
                            assert_eq!(actual, expected, "{context}");
                            let (actual_bytes, expected_bytes) =
                                (actual.deep_size_of(), expected.deep_size_of());
                            if sparse && version == LanceFileVersion::V2_0 {
                                // A v2.0 read of several row runs in a page copies a
                                // fixed-width column's runs into a buffer that reserves
                                // the bytes after the first run only, so it reallocates
                                // to about twice their bytes, while the resident copies
                                // hold exactly theirs. The two reads lay out every
                                // other column, and pages, the same way. Only whole
                                // planes are cached as plane entries, where the charge
                                // must match; selected rows are scored and dropped.
                                assert!(
                                    actual_bytes <= expected_bytes,
                                    "{context}: {actual_bytes} > {expected_bytes}"
                                );
                            } else {
                                assert_eq!(actual_bytes, expected_bytes, "{context}");
                            }
                        }
                    }
                }
                assert_eq!(
                    resident.resident_columns().loaded_bytes(),
                    Some(source.resident_columns_bytes())
                );
            }
        }

        /// With its small columns resident, a storage reads only the code and
        /// bounds columns from the file. In the writer's pages, as the
        /// benchmark indexes are laid out, a native partition miss makes 2
        /// requests instead of 8, and a layered whole plane 1. The read that
        /// loads the store counts the load's requests too.
        #[rstest]
        #[case::native(false)]
        #[case::layered(true)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_columns_cut_origin_requests(#[case] layered: bool) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            assert_eq!(lazy_index(&index).is_layered_rq(), layered);
            let source = &lazy_index(&index).storage;
            let reader = rewrite_rq_storage(source, LanceFileVersion::V2_0, None).await;
            // Lazy bounds whatever `LANCE_RQ_SIGN_BOUNDS` sets: a whole sign
            // plane then reads its code column alone from the file, where
            // eager bounds would add the two bounds columns, never resident.
            let file = rq_storage_over(source, &reader, false).with_sign_bounds(SignBounds::Lazy);
            let resident =
                rq_storage_over(source, &reader, true).with_sign_bounds(SignBounds::Lazy);
            let schema = source.schema();
            // The code and bounds columns are those without a fixed width.
            let file_columns = schema
                .fields()
                .iter()
                .filter(|field| field.data_type().primitive_width().is_none())
                .count() as u64;
            let columns = schema.fields().len() as u64;
            if !layered {
                assert_eq!((columns, file_columns), (8, 2));
            }
            let partition = (0..source.num_partitions())
                .find(|&partition| source.partition_size(partition) > 1)
                .unwrap();
            let requests = |stats: &IoStats| stats.snapshot().iops;

            layered_stats::snapshot_and_reset();
            let first = IoStats::new();
            resident
                .load_partition(partition, Some(first.clone()))
                .await
                .unwrap();
            let load = layered_stats::snapshot_and_reset();
            assert!(load.resident_columns_load_requests > 0, "{load:?}");
            assert_eq!(
                requests(&first),
                load.resident_columns_load_requests + file_columns
            );

            for (storage, expected) in [(&file, columns), (&resident, file_columns)] {
                let stats = IoStats::new();
                storage
                    .load_partition(partition, Some(stats.clone()))
                    .await
                    .unwrap();
                let enabled = storage.resident_columns_enabled();
                assert_eq!(requests(&stats), expected, "resident={enabled}");
            }
            if layered {
                let rows = resident_test_rows(source.partition_size(partition));
                for plane in [0, 1, 2] {
                    let columns = plane_columns(plane, SignBounds::Lazy).len() as u64;
                    let mut sparse = Vec::new();
                    for (storage, expected) in [(&file, columns), (&resident, 1)] {
                        let enabled = storage.resident_columns_enabled();
                        let whole = IoStats::new();
                        storage
                            .read_plane(partition, plane, None, Some(whole.clone()))
                            .await
                            .unwrap();
                        assert_eq!(
                            requests(&whole),
                            expected,
                            "plane={plane} resident={enabled}"
                        );
                        let stats = IoStats::new();
                        storage
                            .read_plane(partition, plane, Some(rows.clone()), Some(stats.clone()))
                            .await
                            .unwrap();
                        sparse.push(requests(&stats));
                    }
                    // A sparse read fetches the row runs of the code column alone.
                    assert!(sparse[1] < sparse[0], "plane={plane} {sparse:?}");
                }
            }
            // Only the first read loaded the store.
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_bytes, 0, "{stats:?}");
        }

        /// First reads issued at once, as by queries that miss together, load
        /// the resident store once, and a storage that shares the store, as a
        /// reconstruction of the index does, does not load it again. The load
        /// holds the bytes that `resident_columns_bytes` gives without I/O.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_columns_load_once() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let source = &lazy_index(&index).storage;
            // Row ids and the add, scale and error factors of the sign codes
            // and the add and scale factors of both ex planes.
            let per_row: u64 = source
                .schema()
                .fields()
                .iter()
                .map(|field| match field.data_type() {
                    DataType::UInt64 => 8,
                    DataType::Float32 => 4,
                    _ => 0,
                })
                .sum();
            assert_eq!(per_row, 36);
            let bytes = source.resident_columns_bytes();
            assert_eq!(bytes, per_row * LAZY_ROWS as u64);

            let storage = Arc::new(rq_storage_over(source, source.reader(), true));
            layered_stats::snapshot_and_reset();
            let reads: Vec<_> = (0..RESIDENT_TEST_CONCURRENT_READS)
                .map(|read| {
                    let storage = storage.clone();
                    tokio::spawn(async move {
                        let partition = read % LAZY_PARTITIONS;
                        if read % 2 == 0 {
                            storage
                                .read_plane(partition, 0, None, None)
                                .await
                                .map(|plane| plane.num_rows())
                        } else {
                            storage
                                .load_partition(partition, None)
                                .await
                                .map(|loaded| loaded.len())
                        }
                    })
                })
                .collect();
            for read in reads {
                read.await.unwrap().unwrap();
            }
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_bytes, bytes, "{stats:?}");
            // The arrays hold at least their values.
            assert!(stats.resident_columns_alloc_bytes >= bytes, "{stats:?}");
            assert!(stats.resident_columns_load_requests > 0, "{stats:?}");
            assert!(stats.resident_columns_load_bytes > 0, "{stats:?}");
            assert_eq!(storage.resident_columns().loaded_bytes(), Some(bytes));

            let shared = rq_storage_over(source, source.reader(), true)
                .with_resident_columns(storage.resident_columns().clone());
            let partition = (0..LAZY_PARTITIONS)
                .find(|&partition| source.partition_size(partition) > 0)
                .unwrap();
            shared.load_partition(partition, None).await.unwrap();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_bytes, 0, "{stats:?}");
        }

        /// Searches return the same batches whether the small columns are
        /// resident or read from the origin file: every precision, the
        /// cascade and the lazy scan of a layered index, and a native index in
        /// both approximation modes.
        #[rstest]
        #[case::native(false)]
        #[case::layered(true)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_columns_search_parity(#[case] layered: bool) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) =
                write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let (dataset, _, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let file = open_with_resident_columns(&dataset, false).await;
            let resident = open_with_resident_columns(&dataset, true).await;
            layered_stats::snapshot_and_reset();
            for key in [vectors.value(0), vectors.value(777)] {
                let queries = if layered {
                    sign_bounds_test_queries(&key)
                } else {
                    [ApproxMode::Normal, ApproxMode::Accurate]
                        .into_iter()
                        .map(|approx_mode| {
                            let mut query = lazy_test_query(key.clone(), 100, 8);
                            query.approx_mode = approx_mode;
                            let label = format!("approx={approx_mode:?}");
                            (label, query, LayeredLazyConfig::default())
                        })
                        .collect()
                };
                for (label, query, scan) in queries {
                    let mut results = Vec::new();
                    for index in [&file, &resident] {
                        let ivf = lazy_index(index);
                        ivf.set_layered_lazy_config_for_test(scan);
                        let result = search_global(index, &query, Arc::new(NoFilter)).await;
                        ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                        results.push(result_bits(&result.unwrap()));
                    }
                    let context = format!("layered={layered} {label}");
                    assert!(!results[0].0.is_empty(), "{context}");
                    assert_eq!(results[1], results[0], "{context}");
                }
            }
            let stats = layered_stats::snapshot_and_reset();
            let bytes = lazy_index(&resident).resident_columns_bytes();
            assert_eq!(stats.resident_columns_bytes, bytes, "{stats:?}");
            let file_store = lazy_index(&file).storage.resident_columns();
            assert_eq!(file_store.loaded_bytes(), None);
        }

        /// Whether the small columns are resident is what the environment
        /// resolves for the index's origin and index cache, when the index
        /// opens and when it is reconstructed from the cached state, and a
        /// reconstruction while the index lives shares its store.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_columns_survive_reconstruction() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let ivf = lazy_index(&index);
            let bytes = ivf.resident_columns_bytes();
            let setting = resident_columns_setting().unwrap();
            let expected = setting.resolve(ivf.origin_latency())
                && setting.admits(bytes, dataset.index_cache.max_entry_bytes());
            assert_eq!(ivf.resident_columns_enabled(), expected);
            assert_eq!(ivf.storage.resident_columns_enabled(), expected);
            // Load the store through the index's handle.
            let partition = (0..LAZY_PARTITIONS)
                .find(|&partition| ivf.storage.partition_size(partition) > 0)
                .unwrap();
            rq_storage_over(&ivf.storage, ivf.storage.reader(), true)
                .with_resident_columns(ivf.storage.resident_columns().clone())
                .load_partition(partition, None)
                .await
                .unwrap();
            assert_eq!(ivf.storage.resident_columns().loaded_bytes(), Some(bytes));

            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let reopened = lazy_index(&reopened);
            assert_eq!(reopened.resident_columns_enabled(), expected);
            // A resident store is the one the live index loaded; an index
            // that keeps none resident reads the file.
            let store = reopened.storage.resident_columns();
            assert_eq!(
                store.loaded_bytes(),
                expected.then_some(bytes),
                "shared store"
            );
        }

        /// Scheme of the stores [`CloudTestStoreProvider`] makes.
        const CLOUD_TEST_SCHEME: &str = "rqcloud";

        /// In-memory stores over one backend, under a scheme that `auto`
        /// takes for a cloud object store: a dataset written through a
        /// registry holding the provider opens as class high, from any
        /// session over that registry.
        #[derive(Debug)]
        struct CloudTestStoreProvider(Arc<object_store::memory::InMemory>);

        #[async_trait::async_trait]
        impl lance_io::object_store::ObjectStoreProvider for CloudTestStoreProvider {
            async fn new_store(
                &self,
                base_path: url::Url,
                _params: &ObjectStoreParams,
            ) -> Result<ObjectStore> {
                Ok(ObjectStore::new(
                    self.0.clone(),
                    base_path,
                    None,
                    None,
                    false,
                    true,
                    lance_io::object_store::DEFAULT_CLOUD_IO_PARALLELISM,
                    lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
                    None,
                ))
            }
        }

        /// A registry that resolves [`CLOUD_TEST_SCHEME`] URIs to a new
        /// in-memory backend.
        fn cloud_test_registry() -> Arc<ObjectStoreRegistry> {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let backend = Arc::new(object_store::memory::InMemory::new());
            registry.insert(CLOUD_TEST_SCHEME, Arc::new(CloudTestStoreProvider(backend)));
            registry
        }

        /// The lazy test data with a layered index, written through
        /// `registry` to a new dataset under [`CLOUD_TEST_SCHEME`]: the
        /// dataset's URI and the data.
        async fn write_cloud_lazy_test_dataset(
            registry: Arc<ObjectStoreRegistry>,
        ) -> (String, RecordBatch) {
            let uri = format!("{CLOUD_TEST_SCHEME}://bucket/{}", Uuid::new_v4());
            let (batch, centroids) = lazy_test_data();
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let params = WriteParams {
                session: Some(Arc::new(Session::new(
                    0,
                    LAZY_METADATA_CACHE_BYTES,
                    registry,
                ))),
                ..Default::default()
            };
            let mut dataset = Dataset::write(reader, uri.as_str(), Some(params))
                .await
                .unwrap();
            let params = VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
                RQBuildParams::new(7).with_layered(true),
            );
            dataset
                .create_index(&["vector"], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
            (uri, batch)
        }

        /// The dataset at `uri` over `session`.
        async fn open_with_session(uri: &str, session: Session) -> Dataset {
            crate::DatasetBuilder::from_uri(uri)
                .with_session(Arc::new(session))
                .load()
                .await
                .unwrap()
        }

        /// Open the index of `dataset` with UUID `uuid`, as a query does, and
        /// search it for `query` with the lazy scan settings `config`.
        async fn open_and_search(
            dataset: &Dataset,
            uuid: &Uuid,
            query: &Query,
            config: LayeredLazyConfig,
        ) -> Result<Arc<dyn VectorIndex>> {
            let index = dataset
                .open_vector_index("vector", uuid, &NoOpMetricsCollector)
                .await?;
            lazy_index(&index).set_layered_lazy_config_for_test(config);
            search_global(&index, query, Arc::new(NoFilter)).await?;
            Ok(index)
        }

        /// Indexes of one file opened at once on a fresh session, as a
        /// benchmark's workers open a cold index, bind to one resident store
        /// and one far gather pool: their first searches load the store once,
        /// the index cache charges and pins it while they live, and a permit
        /// taken through one index is taken from every index's pool. The file
        /// is on a store that `auto` takes for a cloud object store, so the
        /// index opens as class high and keeps its small columns resident,
        /// unless the environment sets otherwise; an index cache with no room
        /// admits no store, so there `auto` keeps them in the file.
        #[rstest]
        #[case::charged(LAZY_LARGE_CACHE_BYTES)]
        #[case::zero_capacity(0)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_concurrent_cold_opens_load_once(#[case] cache_bytes: usize) {
            const OPENS: usize = 8;
            const FAR_INFLIGHT: usize = 2;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let registry = cloud_test_registry();
            let (uri, batch) = write_cloud_lazy_test_dataset(registry.clone()).await;
            let session = Session::new(cache_bytes, LAZY_METADATA_CACHE_BYTES, registry);
            let dataset = open_with_session(&uri, session).await;
            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let key = batch["vector"].as_fixed_size_list().value(0);
            let query = lazy_test_query(key, 10, 8);
            let config = LayeredLazyConfig {
                enabled: true,
                far_inflight: FAR_INFLIGHT,
                ..Default::default()
            };

            layered_stats::snapshot_and_reset();
            let indexes = futures::future::try_join_all(
                (0..OPENS).map(|_| open_and_search(&dataset, &uuid, &query, config)),
            )
            .await
            .unwrap();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.lazy_queries, OPENS as u64, "{stats:?}");
            let first = lazy_index(&indexes[0]);
            if origin_latency_setting().unwrap().is_none() {
                assert_eq!(first.origin_latency(), OriginLatencyClass::High);
            }
            let setting = resident_columns_setting().unwrap();
            let bytes = first.resident_columns_bytes();
            let max_entry_bytes = dataset.index_cache.max_entry_bytes();
            assert_eq!(max_entry_bytes, Some(cache_bytes as u64));
            let resident = first.resident_columns_enabled();
            assert_eq!(
                resident,
                setting.resolve(first.origin_latency()) && setting.admits(bytes, max_entry_bytes)
            );
            if cache_bytes == 0 && setting == ResidentColumnsSetting::Auto {
                assert!(!resident);
                assert!(stats.resident_columns_oversize > 0, "{stats:?}");
            }
            let loaded = resident.then_some(bytes);
            // One load, however many indexes read the file at once.
            assert_eq!(
                stats.resident_columns_loads,
                u64::from(resident),
                "{stats:?}"
            );
            assert_eq!(
                stats.resident_columns_bytes,
                loaded.unwrap_or(0),
                "{stats:?}"
            );
            assert_eq!(
                stats.resident_columns_load_requests > 0,
                resident,
                "{stats:?}"
            );
            for index in &indexes {
                let store = lazy_index(index).storage.resident_columns();
                assert_eq!(store.loaded_bytes(), loaded);
            }
            // The live indexes pin the one store the cache charges.
            let pinned = dataset.index_cache.pinned_stats();
            let charged = resident && cache_bytes > 0;
            assert_eq!(pinned.pinned_entries, u64::from(charged), "{pinned:?}");
            assert_eq!(pinned.overflow, 0, "{pinned:?}");

            let pools: Vec<_> = indexes
                .iter()
                .map(|index| lazy_index(index).storage.lazy_far_permits(&config).unwrap())
                .collect();
            let held = pools[0].try_acquire().unwrap();
            for pool in &pools {
                assert_eq!((pool.size(), pool.in_flight()), (FAR_INFLIGHT, 1));
            }
            drop(held);
            assert!(pools.iter().all(|pool| pool.in_flight() == 0));
        }

        /// A cached state lost while an older index of its file still runs
        /// rebinds to that index's resident store and far gather pool, both
        /// when the files are opened anew and when the state is read back
        /// from the persistent tier, so neither loads the store again, and
        /// the open admits the store the cache lost again. The cache then
        /// holds it idle, though no state holds it: it stays loaded once
        /// every index of the file is dropped.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_lost_state_rebinds_index_file_handles() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let registry = cloud_test_registry();
            let (uri, batch) = write_cloud_lazy_test_dataset(registry.clone()).await;
            let tiered = Arc::new(TieredPlaneTestBackend::new(LAZY_LARGE_CACHE_BYTES, false));
            let session = Session::with_index_cache_backend(
                tiered.clone(),
                LAZY_METADATA_CACHE_BYTES,
                registry,
            );
            let dataset = open_with_session(&uri, session).await;
            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            let key = batch["vector"].as_fixed_size_list().value(0);
            let query = lazy_test_query(key, 10, 8);
            let config = LayeredLazyConfig::default();
            let far = LayeredLazyConfig {
                far_inflight: 2,
                ..config
            };

            layered_stats::snapshot_and_reset();
            let old = open_and_search(&dataset, &uuid, &query, config)
                .await
                .unwrap();
            let ivf = lazy_index(&old);
            let loaded = ivf
                .resident_columns_enabled()
                .then(|| ivf.resident_columns_bytes());
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(
                stats.resident_columns_bytes,
                loaded.unwrap_or(0),
                "{stats:?}"
            );
            assert_eq!(ivf.storage.resident_columns().loaded_bytes(), loaded);
            let pool = ivf.storage.lazy_far_permits(&far).unwrap();
            let held = pool.try_acquire().unwrap();

            // Evicted from every tier: the next open reads the files anew.
            tiered.clear().await;
            assert!(dataset.index_cache.get_with_key(&state_key).await.is_none());
            let reopened = open_and_search(&dataset, &uuid, &query, config)
                .await
                .unwrap();
            // Evicted from RAM only: the next open reads the state back.
            assert_eq!(tiered.persisted_entries(IvfStateEntryBox::TYPE_ID), 1);
            tiered.ram.clear().await;
            let read_back = open_and_search(&dataset, &uuid, &query, config)
                .await
                .unwrap();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_bytes, 0, "{stats:?}");
            // Each open found the store gone from the cleared RAM.
            let recharges = if loaded.is_some() { 2 } else { 0 };
            assert_eq!(stats.resident_columns_recharges, recharges, "{stats:?}");
            for (label, index) in [("reopened", &reopened), ("read back", &read_back)] {
                let storage = &lazy_index(index).storage;
                assert_eq!(storage.resident_columns().loaded_bytes(), loaded, "{label}");
                let permits = storage.lazy_far_permits(&far).unwrap();
                assert_eq!((permits.size(), permits.in_flight()), (2, 1), "{label}");
            }
            drop(held);

            drop((old, reopened, read_back, pool));
            let again = open_and_search(&dataset, &uuid, &query, config)
                .await
                .unwrap();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_bytes, 0, "{stats:?}");
            let store = lazy_index(&again).storage.resident_columns();
            assert_eq!(store.loaded_bytes(), loaded);
        }

        /// RAM of the tiered backend the resident store lifecycle tests use:
        /// room for the lazy test index's store under `auto`, which takes at
        /// most half of it, and for a query's planes, and small enough that
        /// churn evicts idle entries quickly.
        const RESIDENT_TEST_RAM_BYTES: usize = 4 * 1024 * 1024;
        /// Bytes of the lazy test index's resident store: row ids and seven
        /// four-byte factors per row.
        const LAZY_RESIDENT_BYTES: u64 = 36 * LAZY_ROWS as u64;
        /// Room a store's cache entry may take beyond its arrays' buffers.
        const RESIDENT_ENTRY_OVERHEAD_BYTES: u64 = 64 * 1024;

        /// Whether the lazy test index opened as class high, as on S3, over
        /// an index cache admitting entries up to `max_entry_bytes`, keeps
        /// its small columns resident as the environment resolves it, for
        /// the default `index` lifetime. The resident store lifecycle tests
        /// have nothing to check when the environment keeps the columns in
        /// the file, and check the `process` lifetime elsewhere.
        fn resident_as_high(max_entry_bytes: Option<u64>) -> bool {
            let class = origin_latency_setting()
                .unwrap()
                .unwrap_or(OriginLatencyClass::High);
            let setting = resident_columns_setting().unwrap();
            let index_lifetime = resident_lifetime_setting().unwrap() == ResidentLifetime::Index;
            index_lifetime
                && setting.resolve(class)
                && setting.admits(LAZY_RESIDENT_BYTES, max_entry_bytes)
        }

        /// The layered lazy test index at `uri`, opened by a session that
        /// declares class high over the index cache `backend`: the dataset,
        /// the index's UUID and the key of its storage file.
        struct ResidentTestIndex {
            dataset: Dataset,
            uuid: Uuid,
            file: IndexFileKey,
        }

        impl ResidentTestIndex {
            async fn open(uri: &str, backend: Arc<dyn CacheBackend>) -> Self {
                let session = Session::with_index_cache_backend(
                    backend,
                    LAZY_METADATA_CACHE_BYTES,
                    Arc::new(ObjectStoreRegistry::default()),
                )
                .with_index_origin_latency(Some(OriginLatencyClass::High));
                let dataset = open_with_session(uri, session).await;
                let index = dataset.load_indices().await.unwrap()[0].clone();
                let index_dir = dataset.indice_files_dir(&index).unwrap();
                let file =
                    super::super::aux_file_key(&dataset.object_store, &index_dir, &index.uuid);
                Self {
                    dataset,
                    uuid: index.uuid,
                    file,
                }
            }

            async fn open_index(&self) -> Arc<dyn VectorIndex> {
                self.dataset
                    .open_vector_index("vector", &self.uuid, &NoOpMetricsCollector)
                    .await
                    .unwrap()
            }

            /// Open the index as a query does, search it and drop it.
            async fn search(&self, query: &Query) -> (Vec<u64>, Vec<u32>) {
                let index = self.open_index().await;
                let result = search_global(&index, query, Arc::new(NoFilter))
                    .await
                    .unwrap();
                result_bits(&result)
            }

            /// The index's namespace of the index cache that charges the store.
            fn file_cache(&self) -> LanceCache {
                self.dataset.index_cache.for_index(&self.uuid, None)
            }

            /// Whether RAM holds the store, checked without an access.
            async fn store_cached(&self) -> bool {
                self.file_cache()
                    .peek_resident_with_key(&ResidentColumnsKey::new(&self.file))
                    .await
            }
        }

        /// Polls of a condition that tasks a search spawned settle, which may
        /// hold its index for a moment after the search returned.
        const DRAIN_POLLS: usize = 500;
        const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

        /// Wait until no loaded store of `file` is alive.
        async fn wait_until_store_freed(file: &IndexFileKey) -> bool {
            for _ in 0..DRAIN_POLLS {
                if !resident_store_is_live(file) {
                    return true;
                }
                tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
            }
            false
        }

        /// Wait until `cache` holds no leased entry.
        async fn wait_until_unleased(cache: &LanceCache) -> PinnedStats {
            for _ in 0..DRAIN_POLLS {
                let stats = cache.pinned_stats();
                if stats.leased_entries == 0 {
                    return stats;
                }
                tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
            }
            cache.pinned_stats()
        }

        /// The store is an entry of the index's namespace of the index
        /// cache, charged what its arrays allocate, and pinned while an
        /// index of the file lives; no storage charges it again. Once the
        /// index drops, the store stays cached, idle and unpinned.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_store_charged_in_index_cache() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

            layered_stats::snapshot_and_reset();
            let index = test.open_index().await;
            search_global(&index, &query, Arc::new(NoFilter))
                .await
                .unwrap();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(
                stats.resident_columns_bytes, LAZY_RESIDENT_BYTES,
                "{stats:?}"
            );
            let alloc = stats.resident_columns_alloc_bytes;
            let entry = test
                .file_cache()
                .get_resident_with_key(&ResidentColumnsKey::new(&test.file))
                .await
                .expect("the store is cached");
            let charged = entry.deep_size_of() as u64;
            assert!(charged >= alloc, "{charged} < {alloc}");
            assert!(
                charged <= alloc + RESIDENT_ENTRY_OVERHEAD_BYTES,
                "{charged}"
            );
            let ivf = lazy_index(&index);
            assert!((ivf.storage.deep_size_of() as u64) < LAZY_RESIDENT_BYTES);
            let lease = ivf.storage.resident_columns().lease().expect("leased");
            assert!(lease.is_pinned());
            let pinned = test.dataset.index_cache.pinned_stats();
            assert_eq!((pinned.pinned_entries, pinned.leased_entries), (1, 1));
            assert!(pinned.pinned_bytes >= charged, "{pinned:?}");

            drop((entry, index));
            let pinned = wait_until_unleased(&test.dataset.index_cache).await;
            assert_eq!((pinned.pinned_entries, pinned.leased_entries), (0, 0));
            assert!(test.store_cached().await);
        }

        /// A cached state read back from a persistent tier is admitted to RAM
        /// before the index binds its store, and that admission can evict:
        /// the open leases the store first. With the state on the persistent
        /// tier alone, the store in RAM idle and no index live, ten times the
        /// RAM of churn during the state's admission neither evicts nor
        /// reloads the store.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_state_promotion_from_nvme_keeps_store() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

            layered_stats::snapshot_and_reset();
            let expected = test.search(&query).await;
            assert!(test.store_cached().await);
            backend.evict_states_from_ram();
            backend.churn_on_next_admission(10 * RESIDENT_TEST_RAM_BYTES);
            let actual = test.search(&query).await;
            assert_eq!(actual, expected);
            assert_eq!(backend.churned_admissions.load(Ordering::SeqCst), 1);
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(stats.resident_store_evictions, 0, "{stats:?}");
            assert_eq!(stats.resident_columns_preopen_leases, 1, "{stats:?}");
            assert_eq!(stats.pinned_overflow, 0, "{stats:?}");
            assert!(test.store_cached().await);
        }

        /// The V12 regression: a cached state evicted while no index of its
        /// file is live leaves the store cached, since the state holds none,
        /// and the next open reads the state back and binds the same store.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_state_eviction_keeps_store() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 100, 16);

            layered_stats::snapshot_and_reset();
            let expected = test.search(&query).await;
            for _ in 0..3 {
                backend.evict_states_from_ram();
                assert_eq!(test.search(&query).await, expected);
            }
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(stats.resident_store_evictions, 0, "{stats:?}");
            assert_eq!(stats.resident_columns_recharges, 0, "{stats:?}");
            assert_eq!(stats.resident_columns_preopen_leases, 3, "{stats:?}");
        }

        /// No cached value holds a lease: once every query finished and its
        /// index dropped, however many ran at once, nothing is leased or
        /// pinned, and the store stays cached, idle, with the cached state.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_cached_state_holds_no_lease() {
            const CONCURRENT_QUERIES: usize = 4;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let lazy = LayeredLazyConfig {
                enabled: true,
                ..Default::default()
            };
            let queries: Vec<_> = (0..CONCURRENT_QUERIES)
                .map(|query| lazy_test_query(vectors.value(query * 97), 10, 8))
                .collect();
            futures::future::try_join_all(
                queries
                    .iter()
                    .map(|query| open_and_search(&test.dataset, &test.uuid, query, lazy)),
            )
            .await
            .unwrap();
            test.search(&queries[0]).await;

            let pinned = wait_until_unleased(&test.dataset.index_cache).await;
            assert_eq!((pinned.pinned_entries, pinned.leased_entries), (0, 0));
            let entry = test
                .file_cache()
                .get_resident_with_key(&ResidentColumnsKey::new(&test.file))
                .await
                .expect("the idle store stays cached");
            assert_eq!(entry.cache_pin().holders(), 0);
            let frag_reuse_uuid = test.dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&test.uuid, frag_reuse_uuid.as_ref());
            assert!(
                test.dataset
                    .index_cache
                    .get_with_key(&state_key)
                    .await
                    .is_some()
            );
        }

        /// An idle store is an ordinary entry: another index's churn evicts
        /// it once no index of its file is live, freeing it, and the next
        /// query that reads the file loads it once more, with the same
        /// results.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_idle_store_evicted_by_other_index_reloads_once() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

            layered_stats::snapshot_and_reset();
            let expected = test.search(&query).await;
            backend.churn_ram(10 * RESIDENT_TEST_RAM_BYTES).await;
            assert!(!test.store_cached().await);
            assert!(wait_until_store_freed(&test.file).await);
            // The churn pushed the index's planes off the persistent tier
            // too, so the next query reads them from the file.
            backend.disk.lock().unwrap().clear();
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(stats.resident_store_evictions, 1, "{stats:?}");

            assert_eq!(test.search(&query).await, expected);
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(stats.resident_columns_preopen_leases, 0, "{stats:?}");
            assert!(test.store_cached().await);
        }

        /// A state read back from the persistent tier after RAM lost both it
        /// and the store, while an older index of the file still runs, binds
        /// that index's store through the registry and admits it again,
        /// leased: no load.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_state_from_persistent_tier_binds_resident_store() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

            let old = test.open_index().await;
            let expected = result_bits(
                &search_global(&old, &query, Arc::new(NoFilter))
                    .await
                    .unwrap(),
            );
            backend.ram.clear().await;
            assert!(!test.store_cached().await);
            layered_stats::snapshot_and_reset();
            let read_back = test.open_index().await;
            let actual = search_global(&read_back, &query, Arc::new(NoFilter))
                .await
                .unwrap();
            assert_eq!(result_bits(&actual), expected);
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 0, "{stats:?}");
            assert_eq!(stats.resident_columns_registry_reuses, 1, "{stats:?}");
            assert_eq!(stats.resident_columns_recharges, 1, "{stats:?}");
            assert!(test.store_cached().await);
            let lease = lazy_index(&read_back)
                .storage
                .resident_columns()
                .lease()
                .expect("leased");
            assert!(lease.is_pinned());
            assert_eq!(lease.pin().holders(), 2);
        }

        /// The store is charged in the index's namespace without a fragment
        /// reuse segment, so opens under two fragment reuse namespaces, as
        /// before and after a compaction with deferred remapping, share one
        /// cached store and load it once.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_store_survives_frag_reuse_change() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
            let index = test.dataset.load_indices().await.unwrap()[0].clone();

            layered_stats::snapshot_and_reset();
            let mut results = Vec::new();
            for frag_reuse_uuid in [Uuid::new_v4(), Uuid::new_v4()] {
                let context = IvfOpenContext {
                    origin_latency_hint: Some(OriginLatencyClass::High),
                    file_cache: Some(test.file_cache()),
                    resident_lease: None,
                };
                let opened: Arc<dyn VectorIndex> = Arc::new(
                    IvfRq::try_new(
                        test.dataset.object_store.clone(),
                        test.dataset.indice_files_dir(&index).unwrap(),
                        index.uuid,
                        None,
                        &test.dataset.metadata_cache,
                        test.dataset
                            .index_cache
                            .for_index(&index.uuid, Some(&frag_reuse_uuid)),
                        index.file_size_map(),
                        context,
                    )
                    .await
                    .unwrap(),
                );
                let result = search_global(&opened, &query, Arc::new(NoFilter))
                    .await
                    .unwrap();
                results.push(result_bits(&result));
            }
            assert_eq!(results[0], results[1]);
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(stats.resident_columns_binds, 2, "{stats:?}");
            assert!(test.store_cached().await);
        }

        /// The store lives no longer than its session: once every index, the
        /// dataset and the session are dropped, it is freed.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_resident_store_freed_on_session_drop() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            if !resident_as_high(backend.max_entry_bytes()) {
                return;
            }
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let test = ResidentTestIndex::open(dir.as_str(), backend).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
            test.search(&query).await;
            assert!(resident_store_is_live(&test.file));
            let file = test.file.clone();
            drop(test);
            assert!(wait_until_store_freed(&file).await);
        }

        /// A store that would take more than half of the index cache's
        /// largest admissible entry stays in the file under `auto`, which
        /// counts it, with the same results as a resident store.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_oversize_store_resolves_off_under_auto() {
            const SMALL_RAM_BYTES: usize = 1024 * 1024;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let small = Arc::new(TieredPlaneTestBackend::new(SMALL_RAM_BYTES, false));
            let large = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
            let setting = resident_columns_setting().unwrap();
            if setting != ResidentColumnsSetting::Auto || !resident_as_high(large.max_entry_bytes())
            {
                return;
            }
            assert!(LAZY_RESIDENT_BYTES > small.max_entry_bytes().unwrap() / 2);
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 100, 16);
            let mut results = Vec::new();
            for (backend, resident) in [(small, false), (large, true)] {
                let test = ResidentTestIndex::open(dir.as_str(), backend).await;
                layered_stats::snapshot_and_reset();
                let index = test.open_index().await;
                assert_eq!(lazy_index(&index).resident_columns_enabled(), resident);
                let result = search_global(&index, &query, Arc::new(NoFilter))
                    .await
                    .unwrap();
                results.push(result_bits(&result));
                let stats = layered_stats::snapshot_and_reset();
                assert_eq!(
                    stats.resident_columns_oversize,
                    u64::from(!resident),
                    "{stats:?}"
                );
                assert_eq!(
                    stats.resident_columns_loads,
                    u64::from(resident),
                    "{stats:?}"
                );
            }
            assert_eq!(results[0], results[1]);
        }

        /// Open the index of the dataset at `uri` over a new session whose
        /// index cache is `cache`, as if its origin were of `class`, with its
        /// small columns resident or read from the file and asking for
        /// `entry_columns`; warm it when the tests warm that cache.
        async fn open_entry_columns_test_index(
            uri: &str,
            cache: LazyTestCache,
            class: OriginLatencyClass,
            resident: bool,
            entry_columns: EntryColumns,
        ) -> (
            Dataset,
            Arc<dyn VectorIndex>,
            Option<Arc<TieredPlaneTestBackend>>,
        ) {
            let (dataset, tiered) = open_lazy_test_dataset(uri, cache).await;
            let index = open_entry_columns_index(&dataset, class, resident, entry_columns).await;
            if cache.is_warmed() {
                index.prewarm().await.unwrap();
            }
            (dataset, index, tiered)
        }

        /// Open the index of `dataset` over the dataset's caches, as
        /// [`open_entry_columns_test_index`] does.
        async fn open_entry_columns_index(
            dataset: &Dataset,
            class: OriginLatencyClass,
            resident: bool,
            entry_columns: EntryColumns,
        ) -> Arc<dyn VectorIndex> {
            let (store, index_dir) = index_files(dataset).await;
            let ivf = open_rq_index(dataset, store, index_dir)
                .await
                .with_origin_latency_for_test(class)
                .unwrap()
                .with_resident_columns_for_test(resident)
                .with_entry_columns_for_test(entry_columns);
            Arc::new(ivf)
        }

        /// Bytes the columns of `full` that `entry` does not hold add to the
        /// charge of `full`: each array and its buffers.
        fn bytes_beyond(full: &RecordBatch, entry: &RecordBatch) -> usize {
            full.schema()
                .fields()
                .iter()
                .zip(full.columns())
                .filter(|(field, _)| entry.column_by_name(field.name()).is_none())
                .map(|(_, column)| {
                    column.deep_size_of_children(&mut lance_core::deepsize::Context::new())
                })
                .sum()
        }

        fn column_names(batch: &RecordBatch) -> Vec<String> {
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect()
        }

        /// A code-only entry holds the file columns of its plane or native
        /// partition, and is charged the bytes of a read of the file but the
        /// resident columns'. Attaching the resident columns to it, whole or
        /// at selected rows, gives the batch a read of the file gives, with or
        /// without the resident store, bit for bit, and whole planes and
        /// partitions byte for byte: every partition, empty ones included, in
        /// the writer's pages and in pages that partitions straddle, in either
        /// file version. An entry of other rows is an error.
        #[rstest]
        #[case::native_v2_0(false, LanceFileVersion::V2_0)]
        #[case::native_v2_2(false, LanceFileVersion::V2_2)]
        #[case::layered_v2_0(true, LanceFileVersion::V2_0)]
        #[case::layered_v2_2(true, LanceFileVersion::V2_2)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_code_only_entries_match_file_reads(
            #[case] layered: bool,
            #[case] version: LanceFileVersion,
        ) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let source = &lazy_index(&index).storage;
            let sign_bounds = source.sign_bounds();
            // A native file has the sign plane's columns too.
            let mut planes = vec![0];
            if layered {
                planes.extend([1, 2]);
                if sign_bounds == SignBounds::Lazy {
                    planes.push(SIGN_BOUNDS_PLANE);
                }
            }
            for page_rows in [None, Some(RESIDENT_TEST_PAGE_ROWS)] {
                let reader = rewrite_rq_storage(source, version, page_rows).await;
                let file = rq_storage_over(source, &reader, false);
                let resident = rq_storage_over(source, &reader, true);
                let codes =
                    rq_storage_over(source, &reader, true).with_entry_columns(EntryColumns::Codes);
                assert_eq!(resident.entry_columns(), EntryColumns::All);
                assert_eq!(codes.entry_columns(), EntryColumns::Codes);
                // Code-only entries need the resident store.
                let unresident =
                    rq_storage_over(source, &reader, false).with_entry_columns(EntryColumns::Codes);
                assert_eq!(unresident.entry_columns(), EntryColumns::All);
                for partition in 0..source.num_partitions() {
                    let context = format!(
                        "layered={layered} {version:?} pages={page_rows:?} partition={partition}"
                    );
                    let size = source.partition_size(partition);
                    if !layered {
                        let expected = file.load_partition(partition, None).await.unwrap();
                        let entry = codes.read_partition_codes(partition, None).await.unwrap();
                        assert_eq!(
                            column_names(&entry.0),
                            [RABIT_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_COLUMN],
                            "{context}"
                        );
                        let actual = codes
                            .partition_from_codes(partition, &entry, None)
                            .await
                            .unwrap();
                        assert_eq!(
                            actual.to_batches().unwrap().collect::<Vec<_>>(),
                            expected.to_batches().unwrap().collect::<Vec<_>>(),
                            "{context}"
                        );
                        assert_eq!(actual.deep_size_of(), expected.deep_size_of(), "{context}");
                        // The entry is charged what the partition built from
                        // it is but the copies of the resident rows.
                        let actual_batch = actual.to_batches().unwrap().next().unwrap();
                        assert_eq!(
                            entry.0.deep_size_of(),
                            actual_batch.deep_size_of() - bytes_beyond(&actual_batch, &entry.0),
                            "{context}"
                        );
                    }
                    let rows = resident_test_rows(size);
                    let every_row: Vec<u32> = (0..size as u32).collect();
                    for &plane in &planes {
                        let context = format!("{context} plane={plane}");
                        let full = file.read_plane(partition, plane, None, None).await.unwrap();
                        assert_eq!(
                            resident
                                .read_plane(partition, plane, None, None)
                                .await
                                .unwrap(),
                            full,
                            "{context}"
                        );
                        let entry = codes
                            .read_plane_entry(partition, plane, None)
                            .await
                            .unwrap();
                        assert_eq!(
                            column_names(&entry),
                            plane_entry_columns(plane, sign_bounds, EntryColumns::Codes),
                            "{context}"
                        );
                        let whole = codes
                            .attach_resident(partition, plane, &entry, None, None)
                            .await
                            .unwrap();
                        assert_eq!(whole, full, "{context}");
                        assert_eq!(whole.deep_size_of(), full.deep_size_of(), "{context}");
                        // The entry is charged what the plane is but the
                        // copies of the resident rows.
                        assert_eq!(
                            entry.deep_size_of(),
                            whole.deep_size_of() - bytes_beyond(&whole, &entry),
                            "{context}"
                        );
                        for selected in [&rows, &every_row] {
                            let expected = file
                                .read_plane(partition, plane, Some(selected.clone()), None)
                                .await
                                .unwrap();
                            let entry_rows =
                                entry.take(&UInt32Array::from(selected.clone())).unwrap();
                            let actual = codes
                                .attach_resident(
                                    partition,
                                    plane,
                                    &entry_rows,
                                    Some(selected),
                                    None,
                                )
                                .await
                                .unwrap();
                            assert_eq!(actual, expected, "{context} rows={}", selected.len());
                        }
                        if !rows.is_empty() && rows.len() < size {
                            let error = codes
                                .attach_resident(partition, plane, &entry, Some(&rows), None)
                                .await
                                .unwrap_err();
                            assert!(
                                matches!(error, lance_core::Error::Internal { .. }),
                                "{context}: {error}"
                            );
                            assert!(
                                error
                                    .to_string()
                                    .contains(&format!("partition {partition} plane {plane}")),
                                "{context}: {error}"
                            );
                        }
                    }
                }
            }
        }

        /// Searches return the same batches, and assemble partitions of the
        /// same sizes, whether the index caches code-only or full entries,
        /// with its small columns resident or not: every precision, the
        /// cascade and the lazy scan under every setting of the parity tests
        /// of a layered index, reading its planes from the origin file or
        /// from a tiered cache that serves selected rows of persisted
        /// entries, and a native index in both approximation modes and
        /// under every prefilter.
        #[rstest]
        #[case::native(false)]
        #[case::layered(true)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_code_only_entries_search_parity(#[case] layered: bool) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) =
                write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let key = batch["vector"].as_fixed_size_list().value(777);
            let mut queries: Vec<(String, Query, LayeredLazyConfig, Arc<dyn PreFilter>)> =
                Vec::new();
            if layered {
                for (label, query, config) in sign_bounds_test_queries(&key) {
                    queries.push((label, query, config, Arc::new(NoFilter)));
                }
                for (_, config) in lazy_test_configs() {
                    let query = lazy_test_query(key.clone(), 100, LAZY_PARTITIONS);
                    queries.push((format!("{config:?}"), query, config, Arc::new(NoFilter)));
                }
            } else {
                for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                    for (filter_name, filter) in lazy_test_filters() {
                        let mut query = lazy_test_query(key.clone(), 100, 8);
                        query.approx_mode = approx_mode;
                        let label = format!("approx={approx_mode:?} filter={filter_name}");
                        queries.push((label, query, LayeredLazyConfig::default(), filter));
                    }
                }
            }
            let variants = [
                (EntryColumns::All, false),
                (EntryColumns::All, true),
                (EntryColumns::Codes, false),
                (EntryColumns::Codes, true),
            ];
            for cache in [LazyTestCache::Origin, LazyTestCache::Ungated] {
                let mut expected: Option<Vec<(Vec<u64>, Vec<u32>)>> = None;
                let mut expected_sizes: Option<Vec<usize>> = None;
                for (entry_columns, resident) in variants {
                    let context = format!(
                        "layered={layered} cache={cache:?} entries={entry_columns} resident={resident}"
                    );
                    layered_stats::snapshot_and_reset();
                    let (_dataset, index, _) = open_entry_columns_test_index(
                        dir.as_str(),
                        cache,
                        OriginLatencyClass::High,
                        resident,
                        entry_columns,
                    )
                    .await;
                    let ivf = lazy_index(&index);
                    assert_eq!(
                        ivf.entry_columns(),
                        entry_columns.resolve(resident),
                        "{context}"
                    );
                    let mut results = Vec::new();
                    for (label, query, config, filter) in &queries {
                        ivf.set_layered_lazy_config_for_test(*config);
                        let result = search_global(&index, query, filter.clone()).await;
                        ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                        let result =
                            result.unwrap_or_else(|error| panic!("{context} {label}: {error}"));
                        results.push(result_bits(&result));
                    }
                    assert!(results.iter().any(|(ids, _)| !ids.is_empty()), "{context}");
                    wait_for_promotions().await;
                    let stats = layered_stats::snapshot_and_reset();
                    if entry_columns.resolve(resident) == EntryColumns::Codes {
                        assert!(stats.resident_attach_calls > 0, "{context} {stats:?}");
                    }
                    // The tiered cache persisted every plane, so the lazy
                    // scan gathers selected rows from the persistent tier and
                    // none from the origin file, and those rows, attached to
                    // the resident columns, are the rows a read of the file
                    // returns.
                    if layered && cache == LazyTestCache::Ungated {
                        assert_eq!(stats.origin_row_reads, 0, "{context} {stats:?}");
                        let partition = (0..LAZY_PARTITIONS)
                            .find(|&partition| ivf.storage.partition_size(partition) > 1)
                            .unwrap();
                        let rows = resident_test_rows(ivf.storage.partition_size(partition));
                        let plan = GatherPlan {
                            planes: [PlaneSource::Sparse; 2],
                            origin_whole: [false; 2],
                        };
                        let gathered = ivf
                            .storage
                            .gather_ex_rows(
                                partition,
                                &rows,
                                plan,
                                &LayeredLazyConfig::default(),
                                &ivf.index_cache,
                                None,
                            )
                            .await
                            .unwrap();
                        assert_eq!(gathered.sources, [Some(PlaneSource::Sparse); 2]);
                        assert_eq!(gathered.origin_row_reads, 0, "{context}");
                        for (plane, batch) in [(1, &gathered.high), (2, &gathered.low)] {
                            let expected = ivf
                                .storage
                                .read_plane(partition, plane, Some(rows.clone()), None)
                                .await
                                .unwrap();
                            // A row read of a full entry decodes a schema
                            // without the file's metadata; the columns match.
                            assert_eq!(
                                column_names(batch),
                                column_names(&expected),
                                "{context} plane={plane}"
                            );
                            assert_eq!(
                                batch.columns(),
                                expected.columns(),
                                "{context} plane={plane}"
                            );
                        }
                    }
                    match &expected {
                        None => expected = Some(results),
                        Some(expected) => {
                            for ((label, ..), (actual, expected)) in
                                queries.iter().zip(results.iter().zip(expected))
                            {
                                assert_eq!(actual, expected, "{context} {label}");
                            }
                        }
                    }
                    // A partition is assembled of the same arrays from the
                    // origin file whatever the entries hold.
                    if cache == LazyTestCache::Origin {
                        let mut sizes = Vec::with_capacity(LAZY_PARTITIONS);
                        for partition in 0..LAZY_PARTITIONS {
                            let entry = if layered {
                                ivf.load_query_partition(
                                    partition,
                                    RQPrecision::Full,
                                    &NoOpMetricsCollector,
                                )
                                .await
                            } else {
                                ivf.load_partition(partition, true, &NoOpMetricsCollector)
                                    .await
                            };
                            sizes.push(entry.unwrap().size_bytes());
                        }
                        match &expected_sizes {
                            None => expected_sizes = Some(sizes),
                            Some(expected) => assert_eq!(&sizes, expected, "{context}"),
                        }
                    }
                }
            }
        }

        /// With its entries cached, a storage with code-only entries reads
        /// nothing from the file; a miss reads the code columns alone, one
        /// request per column in the writer's pages: 1 for a layered sign,
        /// high or low plane, 2 for a native partition. Reading the entries
        /// loads no store, and the first attaches, issued at once, load it
        /// once.
        #[rstest]
        #[case::native(false)]
        #[case::layered(true)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_code_only_entries_cut_origin_requests(#[case] layered: bool) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let source = &lazy_index(&index).storage;
            let reader = rewrite_rq_storage(source, LanceFileVersion::V2_0, None).await;
            // Lazy bounds whatever `LANCE_RQ_SIGN_BOUNDS` sets: a sign plane
            // entry then holds the code column alone.
            let codes = Arc::new(
                rq_storage_over(source, &reader, true)
                    .with_sign_bounds(SignBounds::Lazy)
                    .with_entry_columns(EntryColumns::Codes),
            );
            let planes: &[u8] = if layered { &[0, 1, 2] } else { &[] };
            let warm = LanceCache::with_capacity(LAZY_LARGE_CACHE_BYTES);
            let requests = |stats: &IoStats| stats.snapshot().iops;

            layered_stats::snapshot_and_reset();
            for partition in 0..LAZY_PARTITIONS {
                if layered {
                    for &plane in planes {
                        codes
                            .load_plane_entry(partition, plane, &WeakLanceCache::from(&warm), None)
                            .await
                            .unwrap();
                    }
                } else {
                    let codes = &codes;
                    warm.get_or_insert_with_key(PartitionCodesKey { partition }, || async move {
                        codes.read_partition_codes(partition, None).await
                    })
                    .await
                    .unwrap();
                }
            }
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 0, "{stats:?}");

            let reads: Vec<_> = (0..RESIDENT_TEST_CONCURRENT_READS)
                .map(|read| {
                    let codes = codes.clone();
                    let cache = WeakLanceCache::from(&warm);
                    tokio::spawn(async move {
                        let partition = read % LAZY_PARTITIONS;
                        if layered {
                            let plane = (read % 3) as u8;
                            codes
                                .load_plane(partition, plane, &cache, None)
                                .await
                                .map(|plane| (plane.num_rows(), true))
                        } else {
                            codes
                                .load_partition_cached(partition, &cache, true, None)
                                .await
                                .map(|(storage, hit)| (storage.len(), hit))
                        }
                    })
                })
                .collect();
            for read in reads {
                let (_, hit) = read.await.unwrap().unwrap();
                assert!(hit);
            }
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 1, "{stats:?}");
            assert_eq!(
                stats.resident_columns_bytes,
                source.resident_columns_bytes(),
                "{stats:?}"
            );
            assert!(stats.resident_attach_calls > 0, "{stats:?}");

            let partition = (0..LAZY_PARTITIONS)
                .find(|&partition| source.partition_size(partition) > 1)
                .unwrap();
            let cold = LanceCache::with_capacity(LAZY_LARGE_CACHE_BYTES);
            for (cache, hit) in [(&warm, true), (&cold, false)] {
                let cache = WeakLanceCache::from(cache);
                if layered {
                    for &plane in planes {
                        let stats = IoStats::new();
                        codes
                            .load_plane(partition, plane, &cache, Some(stats.clone()))
                            .await
                            .unwrap();
                        let expected = if hit { 0 } else { 1 };
                        assert_eq!(requests(&stats), expected, "plane={plane} hit={hit}");
                    }
                } else {
                    let stats = IoStats::new();
                    let (_, was_hit) = codes
                        .load_partition_cached(partition, &cache, true, Some(stats.clone()))
                        .await
                        .unwrap();
                    assert_eq!(was_hit, hit);
                    let expected = if hit { 0 } else { 2 };
                    assert_eq!(requests(&stats), expected, "hit={hit}");
                }
            }
            let stats = layered_stats::snapshot_and_reset();
            assert_eq!(stats.resident_columns_loads, 0, "{stats:?}");
        }

        /// Code-only and full entries of one index persist under keys of
        /// their own: an index of either kind finds none of the other's
        /// entries in a cache the other warmed, and reads its own, every one
        /// of which decodes, with the same results.
        #[rstest]
        #[case::native(false)]
        #[case::layered(true)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_code_only_entries_persist_apart_from_full_entries(#[case] layered: bool) {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) =
                write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
            let key = batch["vector"].as_fixed_size_list().value(0);
            let query = lazy_test_query(key, 100, LAZY_PARTITIONS);
            let partition_entry_type =
                <PartitionEntry<FlatIndex, RabitQuantizer> as CacheCodecImpl>::TYPE_ID;
            for order in [
                [EntryColumns::All, EntryColumns::Codes],
                [EntryColumns::Codes, EntryColumns::All],
            ] {
                let context = format!("layered={layered} order={order:?}");
                let (dataset, tiered) =
                    open_lazy_test_dataset(dir.as_str(), LazyTestCache::ColdUngated).await;
                let tiered = tiered.unwrap();
                let persisted = |entry_columns| match (layered, entry_columns) {
                    (true, _) => tiered.persisted_entries(<PlaneBatch as CacheCodecImpl>::TYPE_ID),
                    (false, EntryColumns::All) => tiered.persisted_entries(partition_entry_type),
                    (false, EntryColumns::Codes) => {
                        tiered.persisted_entries(<PartitionCodes as CacheCodecImpl>::TYPE_ID)
                    }
                };
                let mut indexes = Vec::new();
                for entry_columns in order {
                    let index = open_entry_columns_index(
                        &dataset,
                        OriginLatencyClass::High,
                        true,
                        entry_columns,
                    )
                    .await;
                    assert_eq!(lazy_index(&index).entry_columns(), entry_columns);
                    indexes.push(index);
                }
                indexes[0].prewarm().await.unwrap();
                let first_entries = persisted(order[0]);
                assert!(first_entries > 0, "{context}");
                // The second kind finds none of the first kind's entries.
                let second = lazy_index(&indexes[1]);
                for partition in 0..LAZY_PARTITIONS {
                    let tiers = if layered {
                        let mut tiers = Vec::new();
                        for plane in [0, 1, 2] {
                            tiers.push(
                                second
                                    .storage
                                    .plane_tier(partition, plane, &second.index_cache)
                                    .await,
                            );
                        }
                        tiers
                    } else if order[1] == EntryColumns::Codes {
                        vec![
                            second
                                .index_cache
                                .peek_tier_with_key(&PartitionCodesKey { partition })
                                .await,
                        ]
                    } else {
                        vec![
                            second
                                .index_cache
                                .peek_tier_with_key(
                                    &IVFPartitionKey::<FlatIndex, RabitQuantizer>::new(partition),
                                )
                                .await,
                        ]
                    };
                    assert!(
                        tiers.iter().all(|&tier| tier == CacheTier::Absent),
                        "{context} partition={partition} {tiers:?}"
                    );
                }
                indexes[1].prewarm().await.unwrap();
                if layered {
                    // Both kinds hold the sign, high and low planes.
                    assert_eq!(persisted(order[1]), 2 * first_entries, "{context}");
                } else {
                    assert_eq!(persisted(order[1]), LAZY_PARTITIONS, "{context}");
                    assert_eq!(first_entries, LAZY_PARTITIONS, "{context}");
                }
                for (bytes, codec, _) in tiered.disk.lock().unwrap().values() {
                    assert!(
                        codec.deserialize(bytes).hit().is_some(),
                        "{context}: a persisted {} entry does not decode",
                        codec.type_id()
                    );
                }
                // Each kind reads its own entries back from the persistent
                // tier, as it wrote them.
                tiered.ram.clear().await;
                let mut results = Vec::new();
                for (index, entry_columns) in indexes.iter().zip(order) {
                    let ivf = lazy_index(index);
                    let partition = (0..LAZY_PARTITIONS)
                        .find(|&partition| ivf.storage.partition_size(partition) > 0)
                        .unwrap();
                    let (held, expected): (Vec<String>, Vec<&str>) = if layered {
                        let entry = ivf
                            .index_cache
                            .get_with_key(&ivf.storage.plane_key(partition, 1))
                            .await
                            .unwrap();
                        let sign_bounds = ivf.sign_bounds();
                        (
                            column_names(&entry.0),
                            plane_entry_columns(1, sign_bounds, entry_columns),
                        )
                    } else if entry_columns == EntryColumns::Codes {
                        let entry = ivf
                            .index_cache
                            .get_with_key(&PartitionCodesKey { partition })
                            .await
                            .unwrap();
                        (
                            column_names(&entry.0),
                            vec![RABIT_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_COLUMN],
                        )
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    assert_eq!(held, expected, "{context} {entry_columns}");
                    let result = search_global(index, &query, Arc::new(NoFilter))
                        .await
                        .unwrap();
                    results.push(result_bits(&result));
                }
                assert_eq!(results[0], results[1], "{context}");
            }
        }

        /// A native index with code-only entries warms them, one per
        /// partition and no whole partition, and a search that follows reads
        /// every partition from them: no cache miss and no partition load,
        /// with the results of full entries.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_native_prewarm_inserts_partition_codes() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, false).await;
            let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(3), 100, 16);
            let partition_entry_type =
                <PartitionEntry<FlatIndex, RabitQuantizer> as CacheCodecImpl>::TYPE_ID;
            let mut results = Vec::new();
            for entry_columns in [EntryColumns::Codes, EntryColumns::All] {
                let (_dataset, index, tiered) = open_entry_columns_test_index(
                    dir.as_str(),
                    LazyTestCache::TieredResident,
                    OriginLatencyClass::High,
                    true,
                    entry_columns,
                )
                .await;
                let tiered = tiered.unwrap();
                let codes = tiered.persisted_entries(<PartitionCodes as CacheCodecImpl>::TYPE_ID);
                let whole = tiered.persisted_entries(partition_entry_type);
                match entry_columns {
                    EntryColumns::Codes => assert_eq!((codes, whole), (LAZY_PARTITIONS, 0)),
                    EntryColumns::All => assert_eq!((codes, whole), (0, LAZY_PARTITIONS)),
                }
                let metrics = Arc::new(LocalMetricsCollector::default());
                let (partitions, dists) = index.find_partitions(&query).unwrap();
                let probes = partitions.len();
                let batches = index
                    .clone()
                    .search_partitions(
                        query.clone(),
                        Arc::new(partitions),
                        Arc::new(dists),
                        0,
                        probes,
                        Arc::new(NoFilter),
                        None,
                        metrics.clone(),
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap();
                let result = concat_batches(&VECTOR_RESULT_SCHEMA, batches.iter()).unwrap();
                assert_eq!(metrics.index_cache_misses(), 0, "{entry_columns}");
                assert!(metrics.index_cache_hits() >= probes, "{entry_columns}");
                assert_eq!(
                    metrics.parts_loaded.load(Ordering::Relaxed),
                    0,
                    "{entry_columns}"
                );
                results.push(result_bits(&result));
            }
            assert_eq!(results[0], results[1]);
        }

        /// A native index whose file stores its sign codes unpacked keeps
        /// them packed in its code-only entries: every partition built from a
        /// cached entry uses the entry's codes as they are, and searches
        /// return the results of full entries, which pack the codes on every
        /// construction.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_code_only_entries_keep_unpacked_codes_packed() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (batch, centroids) = lazy_test_data();
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let mut dataset = Dataset::write(reader, dir.as_str(), None).await.unwrap();
            let mut params = VectorIndexParams::with_ivf_rq_params(
                DistanceType::L2,
                IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
                RQBuildParams::new(7),
            );
            params.skip_transpose(true);
            dataset
                .create_index(&["vector"], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
            let vectors = batch["vector"].as_fixed_size_list();
            let mut results = Vec::new();
            for entry_columns in [EntryColumns::All, EntryColumns::Codes] {
                let (_dataset, index, _) = open_entry_columns_test_index(
                    dir.as_str(),
                    LazyTestCache::ColdUngated,
                    OriginLatencyClass::High,
                    true,
                    entry_columns,
                )
                .await;
                let ivf = lazy_index(&index);
                assert!(!ivf.storage.metadata().packed);
                let mut bits = Vec::new();
                for key in [vectors.value(0), vectors.value(777)] {
                    for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                        let mut query = lazy_test_query(key.clone(), 100, 8);
                        query.approx_mode = approx_mode;
                        let result = search_global(&index, &query, Arc::new(NoFilter))
                            .await
                            .unwrap();
                        bits.push(result_bits(&result));
                    }
                }
                results.push(bits);
                if entry_columns == EntryColumns::Codes {
                    for partition in 0..LAZY_PARTITIONS {
                        if ivf.storage.partition_size(partition) == 0 {
                            continue;
                        }
                        let loaded = ivf
                            .load_partition(partition, true, &NoOpMetricsCollector)
                            .await
                            .unwrap();
                        let cached = ivf
                            .index_cache
                            .get_with_key(&PartitionCodesKey { partition })
                            .await
                            .unwrap();
                        let storage_batch = loaded.storage.to_batches().unwrap().next().unwrap();
                        let values = |batch: &RecordBatch| {
                            batch[RABIT_CODE_COLUMN]
                                .as_fixed_size_list()
                                .values()
                                .to_data()
                                .buffers()[0]
                                .as_ptr()
                        };
                        assert_eq!(
                            values(&storage_batch),
                            values(&cached.0),
                            "partition {partition} rewrote its cached codes"
                        );
                    }
                }
            }
            assert_eq!(results[0], results[1]);
        }

        /// Gaps that coalesced origin reads are checked at, in ascending
        /// order so that neighbours compare a narrower gap with a wider one:
        /// touching ranges only, V11's S3 block size, a narrower gap for a
        /// high-latency origin, the gap `auto` gives one, and every range of
        /// a column page.
        const COALESCE_TEST_GAPS: [u64; 5] = [
            0,
            64 * 1024,
            256 * 1024,
            HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES,
            u64::MAX,
        ];
        /// Spacing of the single rows of a sparse selection.
        const COALESCE_TEST_ROW_STEP: usize = 41;

        /// Sparse plane reads return the same batch at every coalescing gap:
        /// every plane of every partition, empty ones included, with the
        /// small columns read from the file or resident, in the writer's
        /// pages and in pages that partitions straddle, in either file
        /// version. In a v2.0 file, as the benchmark indexes are laid out, a
        /// wider gap makes no more requests, gap 0 makes more than a gap that
        /// covers the partition, and that makes no more than a whole read,
        /// which is one request per column page.
        #[rstest]
        #[case::v2_0(LanceFileVersion::V2_0)]
        #[case::v2_2(LanceFileVersion::V2_2)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_read_plane_matches_across_coalesce_gaps(
            #[case] version: LanceFileVersion,
        ) {
            assert!(COALESCE_TEST_GAPS.is_sorted(), "{COALESCE_TEST_GAPS:?}");
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (_dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let source = &lazy_index(&index).storage;
            let mut planes = vec![0, 1, 2];
            if source.sign_bounds() == SignBounds::Lazy {
                planes.push(SIGN_BOUNDS_PLANE);
            }
            // A v2.0 file reads each column page with one request per range
            // group and no page initialization, so its requests are exact.
            let counts_requests = version == LanceFileVersion::V2_0;
            let requests = |stats: &IoStats| stats.snapshot().iops;
            for (page_rows, resident) in [
                (None, false),
                (Some(RESIDENT_TEST_PAGE_ROWS), false),
                (Some(RESIDENT_TEST_PAGE_ROWS), true),
            ] {
                let reader = rewrite_rq_storage(source, version, page_rows).await;
                let storage = rq_storage_over(source, &reader, resident);
                let straddled = (0..source.num_partitions())
                    .any(|partition| straddles_page(&reader, source.ivf().row_range(partition)));
                assert_eq!(straddled, page_rows.is_some(), "pages={page_rows:?}");
                for &plane in &planes {
                    let layout = format!(
                        "{version:?} pages={page_rows:?} resident={resident} plane={plane}"
                    );
                    // Requests at gap 0 and at the covering gap, summed over
                    // the partitions and row selections.
                    let (mut separate, mut covering) = (0, 0);
                    for partition in 0..source.num_partitions() {
                        let partition_rows = source.partition_size(partition);
                        let selections: [Vec<u32>; 2] = [
                            resident_test_rows(partition_rows),
                            (0..partition_rows as u32)
                                .step_by(COALESCE_TEST_ROW_STEP)
                                .collect(),
                        ];
                        let whole = IoStats::new();
                        if counts_requests {
                            storage
                                .read_plane(partition, plane, None, Some(whole.clone()))
                                .await
                                .unwrap();
                        }
                        for rows in selections {
                            let context =
                                format!("{layout} partition={partition} rows={}", rows.len());
                            let expected = storage
                                .read_plane(partition, plane, Some(rows.clone()), None)
                                .await
                                .unwrap();
                            let mut gap_requests = Vec::with_capacity(COALESCE_TEST_GAPS.len());
                            for gap in COALESCE_TEST_GAPS {
                                let stats = IoStats::new();
                                let actual = storage
                                    .read_plane_with_coalesce_gap(
                                        partition,
                                        plane,
                                        Some(rows.clone()),
                                        Some(stats.clone()),
                                        Some(gap),
                                    )
                                    .await
                                    .unwrap();
                                assert_eq!(actual, expected, "{context} gap={gap}");
                                gap_requests.push(requests(&stats));
                            }
                            if counts_requests {
                                let context = format!(
                                    "{context} requests={gap_requests:?} whole={}",
                                    requests(&whole)
                                );
                                assert!(
                                    gap_requests.windows(2).all(|pair| pair[1] <= pair[0]),
                                    "{context}"
                                );
                                let last = *gap_requests.last().unwrap();
                                assert!(last <= requests(&whole), "{context}");
                                separate += gap_requests[0];
                                covering += last;
                            }
                        }
                    }
                    if counts_requests {
                        assert!(separate > covering, "{layout}: {separate} vs {covering}");
                    }
                }
            }
        }

        /// The lazy scan returns the eager results whatever gap its sparse
        /// reads of the origin file merge row runs within, whether it reads
        /// every plane by selected rows or chooses by cost. Every gap counts
        /// the same origin rows, and reading them, a wider gap makes no more
        /// origin requests, for no fewer bytes; `auto` on a high-latency
        /// origin reads as [`HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES`] does.
        #[rstest]
        #[case::rq7_l2(7, DistanceType::L2)]
        #[case::rq9_dot(9, DistanceType::Dot)]
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_origin_gap_parity(
            #[case] bits: u8,
            #[case] distance_type: DistanceType,
        ) {
            assert!(COALESCE_TEST_GAPS.is_sorted(), "{COALESCE_TEST_GAPS:?}");
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), bits, distance_type).await;
            let vectors = batch["vector"].as_fixed_size_list();
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
            let queries: Vec<Query> = [vectors.value(0), vectors.value(777)]
                .into_iter()
                .flat_map(|key| {
                    [(4, 10), (16, 100), (LAZY_PARTITIONS, 1000)]
                        .map(|(nprobes, k)| lazy_test_query(key.clone(), k, nprobes))
                })
                .collect();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let gaps = std::iter::once(LazyOriginGap::Auto)
                .chain(COALESCE_TEST_GAPS.map(LazyOriginGap::Bytes));
            for origin_gap in gaps {
                for dense in [DenseGatherMode::Sparse, DenseGatherMode::Cost] {
                    let config = LayeredLazyConfig {
                        enabled: true,
                        dense,
                        promote: LazyPromotion::Off,
                        origin_gap,
                        ..Default::default()
                    };
                    let label = format!("{distance_type:?} bits={bits} {origin_gap:?} {dense:?}");
                    let mut origin_row_reads = 0;
                    for query in &queries {
                        let context =
                            format!("{label} nprobes={} k={}", query.minimum_nprobes, query.k);
                        let stats =
                            assert_lazy_matches_eager(&index, query, &filter, config, &context)
                                .await;
                        origin_row_reads += stats.origin_row_reads;
                    }
                    // Nothing is cached, so every sparse read goes to the origin.
                    if dense == DenseGatherMode::Sparse {
                        assert!(origin_row_reads > 0, "{label}");
                    }
                }
            }

            let label = format!("{distance_type:?} bits={bits}");
            let mut reads = Vec::with_capacity(COALESCE_TEST_GAPS.len());
            for gap in COALESCE_TEST_GAPS {
                reads.push(
                    origin_sparse_reads(&index, &queries, LazyOriginGap::Bytes(gap), &label).await,
                );
            }
            let requests = reads.iter().map(|read| read.requests).collect::<Vec<_>>();
            let bytes = reads.iter().map(|read| read.bytes).collect::<Vec<_>>();
            let rows = reads.iter().map(|read| read.rows).collect::<Vec<_>>();
            // Every gap gathers the same rows, so the rows counted are the
            // denominator of each gap's byte amplification.
            let context = format!("{label} requests={requests:?} bytes={bytes:?} rows={rows:?}");
            assert!(rows[0] > 0, "{context}");
            assert!(
                rows.iter().all(|&gap_rows| gap_rows == rows[0]),
                "{context}"
            );
            // Gap 0 merges only touching row runs, and `u64::MAX` every run
            // of a read.
            assert!(
                requests.windows(2).all(|pair| pair[1] <= pair[0]),
                "{context}"
            );
            assert!(bytes.windows(2).all(|pair| pair[1] >= pair[0]), "{context}");
            // A wider gap merges row runs that are apart, reading the bytes
            // between them, so it makes fewer requests exactly when it reads
            // more bytes. Every sparse read of the rq9 Dot data is one run per
            // column, so only the rq7 L2 data must merge.
            let last = requests.len() - 1;
            assert_eq!(
                requests[0] > requests[last],
                bytes[0] < bytes[last],
                "{context}"
            );
            if distance_type == DistanceType::L2 {
                assert!(requests[0] > requests[last], "{context}");
            }

            let (store, index_dir) = index_files(&dataset).await;
            let high =
                open_with_origin_latency(&dataset, store, index_dir, OriginLatencyClass::High)
                    .await;
            let auto = origin_sparse_reads(&high, &queries, LazyOriginGap::Auto, &label).await;
            let fixed = origin_sparse_reads(
                &high,
                &queries,
                LazyOriginGap::Bytes(HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES),
                &label,
            )
            .await;
            assert!(auto.requests > 0, "{label}: {auto:?}");
            assert_eq!(auto, fixed, "{label}");
            assert_eq!(auto.rows, rows[0], "{label}: {auto:?}");
        }

        /// What the lazy scan's sparse reads of the origin file added up to.
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
        struct OriginSparseReads {
            /// Requests after coalescing and splitting.
            requests: u64,
            /// Bytes read, the gaps between row runs included.
            bytes: u64,
            /// Rows gathered.
            rows: u64,
        }

        /// The origin requests, bytes and rows of the lazy scan's sparse
        /// reads over `queries` on `index` with `origin_gap`, each query's
        /// results checked against the eager scan's. Every lazy probe is
        /// gathered by selected rows once every earlier probe was scored, so
        /// the rows each gather reads are the same for every gap. Nothing is
        /// cached, so the origin serves each survivor's row of both ex planes.
        async fn origin_sparse_reads(
            index: &Arc<dyn VectorIndex>,
            queries: &[Query],
            origin_gap: LazyOriginGap,
            label: &str,
        ) -> OriginSparseReads {
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let config = LayeredLazyConfig {
                enabled: true,
                window: 0,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Off,
                origin_gap,
                dense_to_eager: DenseToEager::Off,
                far_window: 0,
                ..Default::default()
            };
            // A gather reads its survivors' rows of the high and the low plane.
            const EX_PLANES: u64 = 2;
            let mut reads = OriginSparseReads::default();
            for query in queries {
                let context = format!(
                    "{label} {origin_gap:?} nprobes={} k={}",
                    query.minimum_nprobes, query.k
                );
                let stats =
                    assert_lazy_matches_eager(index, query, &filter, config, &context).await;
                let survivors = stats.rows_fetched.iter().sum::<u64>();
                assert_eq!(
                    stats.origin_sparse_rows,
                    EX_PLANES * survivors,
                    "{context}: {stats:?}"
                );
                reads.requests += stats.origin_sparse_requests;
                reads.bytes += stats.origin_sparse_bytes;
                reads.rows += stats.origin_sparse_rows;
            }
            reads
        }

        /// The lazy origin gap is the environment's setting resolved for the
        /// index's origin, when the index opens and when it is reconstructed
        /// from the cached state; a replaced setting resolves the same way.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_origin_gap_resolves_at_open() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let ivf = lazy_index(&index);
            let block_size = dataset.object_store.block_size() as u64;
            let setting = LayeredLazyConfig::from_env().unwrap().origin_gap;
            let expected = setting.resolve(ivf.origin_latency(), block_size);
            assert_eq!(ivf.lazy_origin_gap_bytes(), expected);

            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(lazy_index(&reopened).lazy_origin_gap_bytes(), expected);

            let auto = match ivf.origin_latency() {
                OriginLatencyClass::High => HIGH_LATENCY_LAZY_ORIGIN_GAP_BYTES,
                OriginLatencyClass::Low => block_size,
            };
            for (origin_gap, expected) in [
                (LazyOriginGap::Auto, auto),
                (LazyOriginGap::Bytes(0), 0),
                (LazyOriginGap::Bytes(u64::MAX), u64::MAX),
            ] {
                ivf.set_layered_lazy_config_for_test(LayeredLazyConfig {
                    origin_gap,
                    ..Default::default()
                });
                assert_eq!(ivf.lazy_origin_gap_bytes(), expected, "{origin_gap:?}");
            }
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
        }

        /// The object store holding the files of `dataset`'s index, and their
        /// directory.
        async fn index_files(dataset: &Dataset) -> (Arc<ObjectStore>, object_store::path::Path) {
            let index = dataset.load_indices().await.unwrap()[0].clone();
            (
                dataset.object_store.clone(),
                dataset.indice_files_dir(&index).unwrap(),
            )
        }

        /// A copy of the files of `dataset`'s index in an in-memory store
        /// under an `s3` URL, which `auto` makes a high-latency origin, and
        /// their directory there.
        async fn cloud_index_files(
            dataset: &Dataset,
        ) -> (Arc<ObjectStore>, object_store::path::Path) {
            let (local, local_dir) = index_files(dataset).await;
            let uuid = dataset.load_indices().await.unwrap()[0].uuid.to_string();
            let store = Arc::new(ObjectStore::new(
                Arc::new(object_store::memory::InMemory::new()),
                url::Url::parse("s3://bucket/").unwrap(),
                None,
                None,
                false,
                true,
                lance_io::object_store::DEFAULT_CLOUD_IO_PARALLELISM,
                lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
                None,
            ));
            let dir = object_store::path::Path::from("indices");
            for file in [lance_index::INDEX_FILE_NAME, INDEX_AUXILIARY_FILE_NAME] {
                let bytes = local
                    .read_one_all(&local_dir.clone().join(uuid.as_str()).join(file))
                    .await
                    .unwrap();
                store
                    .put(&dir.clone().join(uuid.as_str()).join(file), &bytes)
                    .await
                    .unwrap();
            }
            (store, dir)
        }

        /// Open the index of `dataset` from its files in `index_dir` of
        /// `store`, over the dataset's caches.
        async fn open_rq_index(
            dataset: &Dataset,
            store: Arc<ObjectStore>,
            index_dir: object_store::path::Path,
        ) -> IvfRq {
            let index = dataset.load_indices().await.unwrap()[0].clone();
            IvfRq::try_new(
                store,
                index_dir,
                index.uuid,
                None,
                &dataset.metadata_cache,
                dataset.index_cache.for_index(&index.uuid, None),
                index.file_size_map(),
                IvfOpenContext::default(),
            )
            .await
            .unwrap()
        }

        /// [`open_rq_index`], as if the index's origin were of `class`.
        async fn open_with_origin_latency(
            dataset: &Dataset,
            store: Arc<ObjectStore>,
            index_dir: object_store::path::Path,
            class: OriginLatencyClass,
        ) -> Arc<dyn VectorIndex> {
            let ivf = open_rq_index(dataset, store, index_dir)
                .await
                .with_origin_latency_for_test(class)
                .unwrap();
            Arc::new(ivf)
        }

        /// Hold the high and low planes of every non-empty partition where
        /// `tiers` places them: on the persistent tier only for
        /// [`CacheTier::Local`], nowhere for [`CacheTier::Absent`]. Nothing
        /// else stays cached.
        async fn lay_out_ex_planes(
            index: &Arc<dyn VectorIndex>,
            tiered: &TieredPlaneTestBackend,
            tiers: impl Fn(usize) -> [CacheTier; 2],
        ) {
            let ivf = lazy_index(index);
            tiered.clear().await;
            for partition in 0..LAZY_PARTITIONS {
                if ivf.storage.partition_size(partition) == 0 {
                    continue;
                }
                for (plane, tier) in [1u8, 2].into_iter().zip(tiers(partition)) {
                    if tier == CacheTier::Local {
                        ivf.storage
                            .load_plane_entry(partition, plane, &ivf.index_cache, None)
                            .await
                            .unwrap();
                    }
                }
            }
            tiered.ram.clear().await;
        }

        /// Background promotions are off unless set, and `auto` is off for an
        /// index whose origin latency class is high and after one sparse read
        /// otherwise. The index resolves the environment's policy when it
        /// opens and when it is reconstructed from the cached state, and
        /// resolves a replaced class or setting the same way.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_promotion_resolves_at_open() {
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, index, _) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
            let setting = LayeredLazyConfig::from_env().unwrap().promote;
            let ivf = lazy_index(&index);
            let expected = setting.resolve(ivf.origin_latency());
            assert_ne!(expected, LazyPromotion::Auto);
            assert_eq!(ivf.layered_lazy_config().promote, expected);

            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let reconstructed = lazy_index(&reopened);
            assert_eq!(reconstructed.layered_lazy_config().promote, expected);

            let (store, index_dir) = index_files(&dataset).await;
            for class in [OriginLatencyClass::Low, OriginLatencyClass::High] {
                let auto = match class {
                    OriginLatencyClass::Low => LazyPromotion::Background { reads: 1 },
                    OriginLatencyClass::High => LazyPromotion::Off,
                };
                let index =
                    open_with_origin_latency(&dataset, store.clone(), index_dir.clone(), class)
                        .await;
                let ivf = lazy_index(&index);
                assert_eq!(ivf.origin_latency(), class);
                assert_eq!(ivf.storage.origin_latency(), class);
                let promote = ivf.layered_lazy_config().promote;
                assert_eq!(promote, setting.resolve(class), "{class}");
                for promote in [
                    LazyPromotion::Auto,
                    LazyPromotion::Off,
                    LazyPromotion::Background { reads: 2 },
                ] {
                    ivf.set_layered_lazy_config_for_test(LayeredLazyConfig {
                        promote,
                        ..Default::default()
                    });
                    let expected = if promote == LazyPromotion::Auto {
                        auto
                    } else {
                        promote
                    };
                    let resolved = ivf.layered_lazy_config().promote;
                    assert_eq!(resolved, expected, "{class} {promote}");
                }
            }
        }

        /// With `auto` promotions, an index read from a cloud object store
        /// gathers sparse rows from it without ever promoting a plane: no read
        /// reaches the store once a query has returned, and results are the
        /// eager scan's. The same gathers on a local file, whose class is
        /// low, promote planes after one read. Both indexes share one ungated
        /// backend that would admit promoted planes, emptied before every
        /// query so that each sparse gather reads the origin.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_promotion_follows_origin_latency() {
            const QUERIES: usize = 3;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, _, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            let (cloud_store, cloud_dir) = cloud_index_files(&dataset).await;
            let cloud = open_rq_index(&dataset, cloud_store.clone(), cloud_dir).await;
            // `auto` makes the cloud store's class high unless the environment
            // sets it; pin it high for the promotions below.
            let class =
                OriginLatencyClass::resolve(origin_latency_setting().unwrap(), &cloud_store);
            assert_eq!(cloud.origin_latency(), class);
            let cloud = cloud
                .with_origin_latency_for_test(OriginLatencyClass::High)
                .unwrap();
            let cloud: Arc<dyn VectorIndex> = Arc::new(cloud);
            let (local_store, local_dir) = index_files(&dataset).await;
            let local = open_with_origin_latency(
                &dataset,
                local_store.clone(),
                local_dir,
                OriginLatencyClass::Low,
            )
            .await;

            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let queries: Vec<Query> = (0..QUERIES)
                .map(|row| lazy_test_query(vectors.value(row * 101), 100, LAZY_PARTITIONS))
                .collect();
            lazy_index(&local).set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::with_capacity(QUERIES);
            for query in &queries {
                let eager = search_global(&local, query, filter.clone()).await.unwrap();
                expected.push(result_bits(&eager));
            }
            // Promotions are off unless set; `auto` resolves by class.
            let config = LayeredLazyConfig {
                enabled: true,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Auto,
                ..Default::default()
            };
            for (index, store, class) in [
                (&cloud, &cloud_store, OriginLatencyClass::High),
                (&local, &local_store, OriginLatencyClass::Low),
            ] {
                let ivf = lazy_index(index);
                ivf.set_layered_lazy_config_for_test(config);
                let promote = ivf.layered_lazy_config().promote;
                assert_eq!(promote, LazyPromotion::Auto.resolve(class), "{class}");
                layered_stats::snapshot_and_reset();
                let mut late_reads = 0;
                for (query, expected) in queries.iter().zip(&expected) {
                    tiered.clear().await;
                    let result = search_global(index, query, filter.clone()).await.unwrap();
                    assert_eq!(&result_bits(&result), expected, "{class}");
                    store.io_stats_incremental();
                    wait_for_promotions().await;
                    late_reads += store.io_stats_incremental().read_iops;
                }
                let stats = layered_stats::snapshot_and_reset();
                ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                let context = format!("{class} {stats:?}");
                assert_eq!(stats.lazy_queries, QUERIES as u64, "{context}");
                assert_eq!(stats.needed_not_fetched, 0, "{context}");
                assert!(stats.origin_row_reads > 0, "{context}");
                match class {
                    OriginLatencyClass::High => {
                        assert_eq!(
                            (stats.promotions_issued, stats.promotion_bytes),
                            (0, 0),
                            "{context}"
                        );
                        assert_eq!(late_reads, 0, "{context}");
                    }
                    OriginLatencyClass::Low => {
                        assert!(stats.promotions_issued > 0, "{context}");
                        assert!(stats.promotion_bytes > 0, "{context}");
                    }
                }
            }
        }

        /// Under `origin`, staging routes a predicted-dense probe only when
        /// the index's origin is high latency and no cache tier holds its
        /// high or low plane; `all` routes it wherever its planes are, and
        /// `off` never does. Before every query the partitions hold both ex
        /// planes on the persistent tier, only the high plane there, or
        /// neither anywhere, and nothing in RAM. One query's `k` exceeds every
        /// probed row, so every probe is predicted dense; the others' first
        /// probe alone fills the heap, so only that one is.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_dense_to_eager_follows_plane_tiers() {
            const SMALL_K: usize = 10;
            const SMALL_K_QUERIES: usize = 4;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, _, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            let (store, index_dir) = index_files(&dataset).await;
            let (low, high) = (OriginLatencyClass::Low, OriginLatencyClass::High);
            let low_index =
                open_with_origin_latency(&dataset, store.clone(), index_dir.clone(), low).await;
            let high_index = open_with_origin_latency(&dataset, store, index_dir, high).await;
            let (local, absent) = (CacheTier::Local, CacheTier::Absent);
            let layout = |partition: usize| match partition % 3 {
                0 => [local, local],
                1 => [local, absent],
                _ => [absent, absent],
            };

            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let every_row = lazy_test_query(vectors.value(0), LAZY_ROWS + 1, LAZY_PARTITIONS);
            let mut queries = vec![every_row];
            queries.extend(queries_filled_by_first_probe(
                &low_index,
                vectors,
                SMALL_K,
                LAZY_PARTITIONS,
                SMALL_K_QUERIES,
            ));
            lazy_index(&low_index).set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::with_capacity(queries.len());
            for query in &queries {
                let eager = search_global(&low_index, query, filter.clone())
                    .await
                    .unwrap();
                expected.push(result_bits(&eager));
            }

            // The layout is what the tier checks of a high-latency index see.
            // Each index lays out its own entries: the high-latency one's
            // hold their codes alone when its small columns are resident.
            lay_out_ex_planes(&high_index, &tiered, layout).await;
            let ivf = lazy_index(&high_index);
            let cache = &ivf.index_cache;
            for partition in 0..LAZY_PARTITIONS {
                if ivf.storage.partition_size(partition) > 0 {
                    let tiers = [
                        ivf.storage.plane_tier(partition, 1, cache).await,
                        ivf.storage.plane_tier(partition, 2, cache).await,
                    ];
                    assert_eq!(tiers, layout(partition), "partition {partition}");
                }
            }

            for mode in [DenseToEager::Off, DenseToEager::Origin, DenseToEager::All] {
                for (index, class) in [(&low_index, low), (&high_index, high)] {
                    let ivf = lazy_index(index);
                    for (position, (query, eager)) in queries.iter().zip(&expected).enumerate() {
                        lay_out_ex_planes(index, &tiered, layout).await;
                        ivf.set_layered_lazy_config_for_test(LayeredLazyConfig {
                            enabled: true,
                            promote: LazyPromotion::Off,
                            dense_to_eager: mode,
                            ..Default::default()
                        });
                        layered_stats::snapshot_and_reset();
                        let result = search_global(index, query, filter.clone()).await.unwrap();
                        let stats = layered_stats::snapshot_and_reset();
                        ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                        let context = format!("{mode} {class} query={position} {stats:?}");
                        assert_eq!(&result_bits(&result), eager, "{context}");
                        assert_eq!(stats.lazy_queries, 1, "{context}");
                        assert_eq!(stats.needed_not_fetched, 0, "{context}");

                        let (partitions, _) = index.find_partitions(query).unwrap();
                        let mut routed = [0; RANK_BUCKETS];
                        let mut slow = [0; RANK_BUCKETS];
                        for (rank, &partition) in partitions.values().iter().enumerate() {
                            let partition = partition as usize;
                            if ivf.storage.partition_size(partition) == 0 {
                                continue;
                            }
                            // A plane on no cache tier is a slow read on a slow origin only.
                            let reads_slow_origin = class == high && !partition.is_multiple_of(3);
                            let routes = match mode {
                                DenseToEager::Off => false,
                                DenseToEager::Origin => reads_slow_origin,
                                DenseToEager::All => true,
                            };
                            let predicted_dense = query.k > LAZY_ROWS || rank == 0;
                            let bucket = layered_stats::rank_bucket(rank);
                            routed[bucket] += u64::from(predicted_dense && routes);
                            slow[bucket] += u64::from(reads_slow_origin);
                        }
                        assert_eq!(stats.dense_to_eager, routed, "{context}");
                        assert_eq!(stats.s3_bound_probes, slow, "{context}");
                    }
                }
            }
        }

        /// Only a probe whose high or low plane no cache tier holds, on a
        /// high-latency origin, may be gathered beyond the ordinary window,
        /// and only while the far window is wider. Every other gather is
        /// issued within the ordinary window: on a low-latency origin, with
        /// every ex plane on the persistent tier, or with the far window no
        /// wider. Row reads are slowed so that scoring lags the staging of
        /// later probes, which leaves far gathers room to run ahead. On the
        /// slow origin far gathers also run ahead for larger `k`, under a
        /// prefilter and distance bounds, whether gathers wait for the heap to
        /// fill or not, and while predicted-dense probes are routed to the
        /// eager scan. Results are the eager scan's in every setting.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_far_window_by_tier() {
            const PROBES: usize = 32;
            const SMALL_K: usize = 10;
            /// Heaps that about one probe fills, and that several probes fill.
            const WIDE_KS: [usize; 2] = [100, 1000];
            const QUERIES: usize = 2;
            const STAGING_STEPS: usize = 8;
            const WINDOW: usize = 1;
            const ROW_READ_DELAY: Duration = Duration::from_millis(5);
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, _, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            let (store, index_dir) = index_files(&dataset).await;
            let (low, high) = (OriginLatencyClass::Low, OriginLatencyClass::High);
            let low_index =
                open_with_origin_latency(&dataset, store.clone(), index_dir.clone(), low).await;
            let high_index = open_with_origin_latency(&dataset, store, index_dir, high).await;
            let (local, absent) = (CacheTier::Local, CacheTier::Absent);
            let mixed = |partition: usize| match partition % 3 {
                0 => [local, local],
                1 => [local, absent],
                _ => [absent, absent],
            };

            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let queries =
                queries_filled_by_first_probe(&low_index, vectors, SMALL_K, PROBES, QUERIES);
            lazy_index(&low_index).set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::with_capacity(QUERIES);
            for query in &queries {
                let eager = search_global(&low_index, query, filter.clone())
                    .await
                    .unwrap();
                expected.push(result_bits(&eager));
            }
            // The same queries with heaps that fill after one probe or
            // several, without and with a prefilter, and without and with
            // distance bounds, which keep the heap from filling.
            let (prefilter_name, prefilter) = lazy_test_filters()
                .into_iter()
                .find(|(name, _)| *name == "alternating")
                .unwrap();
            let mut wide = Vec::new();
            for query in &queries {
                for k in WIDE_KS {
                    let mut query = query.clone();
                    query.k = k;
                    for (filter_name, filter) in [
                        ("none", filter.clone()),
                        (prefilter_name, prefilter.clone()),
                    ] {
                        let eager = search_global(&low_index, &query, filter.clone())
                            .await
                            .unwrap();
                        let bounded = with_bounds(&query, &eager).unwrap();
                        let bounded_eager = search_global(&low_index, &bounded, filter.clone())
                            .await
                            .unwrap();
                        wide.push((
                            query.clone(),
                            filter_name,
                            filter.clone(),
                            result_bits(&eager),
                        ));
                        wide.push((bounded, filter_name, filter, result_bits(&bounded_eager)));
                    }
                }
            }

            let far = LayeredLazyConfig {
                enabled: true,
                window: WINDOW,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Off,
                far_window: PROBES,
                ..Default::default()
            };
            let far_off = LayeredLazyConfig {
                far_window: WINDOW,
                ..far
            };
            tiered.set_row_read_delay(ROW_READ_DELAY);
            // (setting, index, whether every ex plane is on the persistent
            // tier, config, whether far gathers may run ahead)
            for (setting, index, all_local, config, runs_ahead) in [
                ("low", &low_index, false, far, false),
                ("high, planes local", &high_index, true, far, false),
                ("high, far window off", &high_index, false, far_off, false),
                ("high", &high_index, false, far, true),
            ] {
                let ivf = lazy_index(index);
                ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
                let mut far_early_issues = 0;
                let mut staleness_max = 0;
                for (position, (query, eager)) in queries.iter().zip(&expected).enumerate() {
                    if all_local {
                        lay_out_ex_planes(index, &tiered, |_| [local, local]).await;
                    } else {
                        lay_out_ex_planes(index, &tiered, mixed).await;
                    }
                    ivf.set_layered_lazy_config_for_test(config);
                    layered_stats::snapshot_and_reset();
                    let result = search_global(index, query, filter.clone()).await.unwrap();
                    let stats = layered_stats::snapshot_and_reset();
                    ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                    let context = format!("{setting} query={position} {stats:?}");
                    assert_eq!(&result_bits(&result), eager, "{context}");
                    assert_eq!(stats.lazy_queries, 1, "{context}");
                    assert_eq!(stats.needed_not_fetched, 0, "{context}");
                    assert_lazy_chain_timing(&stats, &context);
                    // A probe on the slow origin issues at most one far gather.
                    let slow_probes: u64 = stats.s3_bound_probes.iter().sum();
                    assert!(stats.far_early_issues <= slow_probes, "{context}");
                    if !runs_ahead {
                        assert_eq!(
                            (stats.far_early_issues, stats.far_permit_waits),
                            (0, 0),
                            "{context}"
                        );
                        assert!(stats.staleness_max <= WINDOW as u64, "{context}");
                    }
                    far_early_issues += stats.far_early_issues;
                    staleness_max = staleness_max.max(stats.staleness_max);
                }
                ivf.set_lazy_prepare_parallelism_for_test(0);
                if runs_ahead {
                    assert!(far_early_issues > 0, "{setting}");
                    assert!(staleness_max > WINDOW as u64, "{setting}: {staleness_max}");
                }
            }

            // The wider queries run ahead on the slow origin too: with gathers
            // issued before the heap fills or held until it does, and with the
            // predicted-dense probes whose planes are on the slow origin
            // loaded by the eager scan. Under the cost policy every probe of an
            // unfiltered, unbounded `k = 1000` query is predicted dense, so
            // that setting routes some; the sparse policy routes none.
            // (setting, config, whether it routes probes to the eager scan)
            let wide_settings = [
                ("sparse", far, false),
                (
                    "sparse, held until the heap fills",
                    LayeredLazyConfig {
                        eager_before_full: false,
                        ..far
                    },
                    false,
                ),
                (
                    "cost, dense probes on the slow origin routed",
                    LayeredLazyConfig {
                        dense: DenseGatherMode::Cost,
                        dense_to_eager: DenseToEager::Origin,
                        ..far
                    },
                    true,
                ),
            ];
            let ivf = lazy_index(&high_index);
            ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
            for (setting, config, routes) in wide_settings {
                let mut far_early_issues = 0;
                let mut routed = 0;
                for (position, (query, filter_name, filter, eager)) in wide.iter().enumerate() {
                    lay_out_ex_planes(&high_index, &tiered, mixed).await;
                    ivf.set_layered_lazy_config_for_test(config);
                    layered_stats::snapshot_and_reset();
                    let result = search_global(&high_index, query, filter.clone())
                        .await
                        .unwrap();
                    let stats = layered_stats::snapshot_and_reset();
                    ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
                    let context = format!(
                        "{setting} query={position} k={} filter={filter_name} bounds={} {stats:?}",
                        query.k,
                        query.upper_bound.is_some()
                    );
                    assert_eq!(&result_bits(&result), eager, "{context}");
                    assert_eq!(stats.lazy_queries, 1, "{context}");
                    assert_eq!(stats.needed_not_fetched, 0, "{context}");
                    assert_lazy_chain_timing(&stats, &context);
                    let slow_probes: u64 = stats.s3_bound_probes.iter().sum();
                    assert!(stats.far_early_issues <= slow_probes, "{context}");
                    far_early_issues += stats.far_early_issues;
                    routed += stats.dense_to_eager.iter().sum::<u64>();
                }
                assert!(far_early_issues > 0, "{setting}");
                assert_eq!(routed > 0, routes, "{setting}: {routed} probes routed");
            }
            ivf.set_lazy_prepare_parallelism_for_test(0);
            tiered.set_row_read_delay(Duration::ZERO);
        }

        /// Gathers beyond the ordinary window share their index's permits
        /// across queries: concurrent queries hold at most one at a time with
        /// one permit and at most four with four, and every permit is back
        /// once the queries finish or are dropped mid-scan. The pool lives in
        /// the state the cached index is reconstructed from. Nothing is
        /// cached, so every probe reads its ex planes from the high-latency
        /// origin, and row reads are slowed so that gathers overlap.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn test_layered_lazy_far_permits_bound_in_flight() {
            const PROBES: usize = 32;
            const SMALL_K: usize = 10;
            const QUERIES: usize = 3;
            const STAGING_STEPS: usize = 8;
            const ROW_READ_DELAY: Duration = Duration::from_millis(5);
            // Rows of the dropped query: a quarter of the rows, so that the
            // heap fills only after several probes, with a threshold that
            // leaves the far gathers after them rows to read.
            const DROPPED_QUERY_K: usize = LAZY_ROWS / 4;
            let _serial = LAZY_TEST_LOCK.lock().await;
            let dir = TempStrDir::default();
            let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
            let (dataset, opened, tiered) =
                open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
            let tiered = tiered.unwrap();
            let one_permit = LayeredLazyConfig {
                enabled: true,
                window: 0,
                dense: DenseGatherMode::Sparse,
                promote: LazyPromotion::Off,
                far_window: PROBES,
                far_inflight: 1,
                ..Default::default()
            };

            // A reconstruction of the index shares its pool.
            let pool = lazy_index(&opened)
                .storage
                .lazy_far_permits(&one_permit)
                .unwrap();
            let held = pool.try_acquire().unwrap();
            let uuid = dataset.load_indices().await.unwrap()[0].uuid;
            let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
            let state_key =
                crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
            assert!(
                dataset.index_cache.get_with_key(&state_key).await.is_some(),
                "the reopen must reconstruct from the cached state"
            );
            let reopened = dataset
                .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let shared = lazy_index(&reopened)
                .storage
                .lazy_far_permits(&one_permit)
                .unwrap();
            assert_eq!((shared.size(), shared.in_flight()), (1, 1));
            assert!(shared.try_acquire().is_none());
            drop(held);
            assert_eq!(shared.in_flight(), 0);

            let (store, index_dir) = index_files(&dataset).await;
            let high = OriginLatencyClass::High;
            let index = open_with_origin_latency(&dataset, store, index_dir, high).await;
            let ivf = lazy_index(&index);
            let vectors = batch["vector"].as_fixed_size_list();
            let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
            let queries = queries_filled_by_first_probe(&index, vectors, SMALL_K, PROBES, QUERIES);
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
            let mut expected = Vec::with_capacity(QUERIES);
            for query in &queries {
                let eager = search_global(&index, query, filter.clone()).await.unwrap();
                expected.push(result_bits(&eager));
            }

            tiered.set_row_read_delay(ROW_READ_DELAY);
            ivf.set_lazy_prepare_parallelism_for_test(STAGING_STEPS);
            for far_inflight in [1, 4] {
                let config = LayeredLazyConfig {
                    far_inflight,
                    ..one_permit
                };
                let permits = ivf.storage.lazy_far_permits(&config).unwrap();
                ivf.set_layered_lazy_config_for_test(config);
                tiered.clear().await;
                layered_stats::snapshot_and_reset();
                let results = futures::future::try_join_all(
                    queries
                        .iter()
                        .map(|query| search_global(&index, query, filter.clone())),
                )
                .await
                .unwrap();
                let stats = layered_stats::snapshot_and_reset();
                let context = format!("far_inflight={far_inflight} {stats:?}");
                for (result, expected) in results.iter().zip(&expected) {
                    assert_eq!(&result_bits(result), expected, "{context}");
                }
                assert_eq!(stats.lazy_queries, QUERIES as u64, "{context}");
                assert_eq!(stats.needed_not_fetched, 0, "{context}");
                assert_lazy_chain_timing(&stats, &context);
                assert!(stats.far_early_issues > 0, "{context}");
                // Observed as each far gather takes its permit.
                let most = far_inflight as u64;
                assert!((1..=most).contains(&stats.far_in_flight_max), "{context}");
                if far_inflight == 1 {
                    // The queries' far gathers queue for the one permit, and
                    // the queue is timed.
                    assert!(stats.far_permit_waits > 0, "{context}");
                    assert!(stats.far_permit_wait_ns > 0, "{context}");
                } else {
                    assert!(stats.far_in_flight_max > 1, "{context}");
                }
                assert_eq!(permits.in_flight(), 0, "{context}");
            }

            // A query dropped while a far gather holds a permit returns it.
            // A permit is seen only while its gather reads rows, for the row
            // read delay: a far gather whose rows the threshold prunes holds
            // it for no time. The first probe of a query with `SMALL_K` fills
            // the heap, and whether its threshold leaves any far gather rows
            // depends on how far staging ran ahead of scoring, so this query
            // takes a `k` that leaves the far gathers rows to read.
            let mut dropped = queries[0].clone();
            dropped.k = DROPPED_QUERY_K;
            let permits = ivf.storage.lazy_far_permits(&one_permit).unwrap();
            ivf.set_layered_lazy_config_for_test(one_permit);
            tiered.clear().await;
            let permit_taken = async {
                while permits.in_flight() == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };
            let finished = tokio::select! {
                result = search_global(&index, &dropped, filter.clone()) => Some(result),
                () = permit_taken => None,
            };
            assert!(
                finished.is_none(),
                "the query finished before a far gather took a permit"
            );
            for _ in 0..500 {
                if permits.in_flight() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(
                permits.in_flight(),
                0,
                "a dropped query must return its permits"
            );
            tiered.set_row_read_delay(Duration::ZERO);
            ivf.set_lazy_prepare_parallelism_for_test(0);
            ivf.set_layered_lazy_config_for_test(LayeredLazyConfig::default());
        }
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
        // Multivector search ranks the final candidates by PQ-approximated
        // scores before refining, so the M4 code of `lightweight_pq_params`
        // (unseeded KMeans) drops recall below the threshold in a few percent
        // of runs. The 4-bit M32 code keeps recall at 1.0.
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            DistanceType::Cosine,
            ivf_params,
            lightweight_hnsw_params(),
            lightweight_pq_params_with_bits(4),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query = vectors.value(0);
        // Three vectors per query amplify the internal candidate k. This
        // bounded budget covers all 64 * 3 vector entries in the fixture.
        let result = search_lightweight_pq_index(
            &dataset,
            query.as_ref(),
            K,
            1,
            2,
            256,
            DistanceType::Cosine,
        )
        .await;
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
            let entry = hnsw.load_partition_entry(partition_id, None).await.unwrap();
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

    /// The global-top-k path (flat sub-index, no early-stop control) must not
    /// pin every probed partition before scoring: a prepared partition holds the
    /// partition's whole quantized storage, so with a high explicit `nprobes`
    /// that made peak memory scale with `nprobes` instead of with the prepare
    /// window. Probe every partition of an index with more partitions than the
    /// window and check the in-flight high-water mark stays within it.
    #[tokio::test]
    async fn test_global_topk_search_bounds_in_flight_prepared_partitions() {
        const INDEX_NAME: &str = "vector_idx";
        const K: usize = 10;
        const ROWS_PER_PARTITION: usize = 16;

        // Resident partitions are bounded by the prepare window plus two scoring
        // chunks (one being scored, one being assembled). The partitions here are
        // far smaller than the chunk byte budget, so the partition cap is what
        // ends a chunk. Size the index so that the old collect-everything
        // behavior would clearly exceed the bound.
        let in_flight_bound =
            get_num_compute_intensive_cpus().max(1) + 2 * super::GLOBAL_TOPK_CHUNK_MAX_PARTITIONS;
        let num_partitions = 2 * in_flight_bound;

        let test_dir = TempStrDir::default();
        let (batch, schema) = make_seeded_vector_batch(num_partitions * ROWS_PER_PARTITION);
        let query_vector = batch["vector"].as_fixed_size_list().value(0);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, test_dir.as_str(), None)
            .await
            .unwrap();
        let mut ivf_params = IvfBuildParams::new(num_partitions);
        ivf_params.max_iters = 2;
        ivf_params.sample_rate = 16;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            lightweight_pq_params(),
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

        let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        let index = dataset
            .open_vector_index("vector", &indices[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let tracker = index
            .as_any()
            .downcast_ref::<super::IvfPq>()
            .expect("IVF_PQ index")
            .prepared_partitions();

        let query = Query {
            rq_cascade_factor: None,
            rq_precision: Default::default(),
            column: "vector".to_string(),
            key: query_vector,
            k: K,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: num_partitions,
            maximum_nprobes: Some(num_partitions),
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        };
        let (partitions, q_c_dists) = index.find_partitions(&query).unwrap();
        assert_eq!(partitions.len(), num_partitions);
        let results = index
            .clone()
            .search_partitions(
                query,
                Arc::new(partitions),
                Arc::new(q_c_dists),
                0,
                num_partitions,
                Arc::new(NoFilter),
                None,
                Arc::new(NoOpMetricsCollector),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let num_results: usize = results.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(num_results, K);
        assert_eq!(
            tracker.in_flight(),
            0,
            "every prepared partition must be released once the search completes"
        );
        assert!(
            tracker.peak() >= 1,
            "the search must have gone through prepared partitions"
        );
        assert!(
            tracker.peak() <= in_flight_bound,
            "peak in-flight prepared partitions {} exceeds the bound {} (probed {} partitions)",
            tracker.peak(),
            in_flight_bound,
            num_partitions
        );
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
        append_template_vector_with_params(
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

        // The oversized partition is split straight to the target size in one
        // optimize: ceil(rows / target) pieces instead of a single halving.
        let split_rows = expected_rows + SPLIT_APPEND_ROWS;
        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            &mut next_id,
            SPLIT_APPEND_ROWS,
            split_rows.div_ceil(IndexType::IvfPq.target_partition_size()),
            split_rows,
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
        // Every vector of a row counts towards the partition size, and the split
        // goes straight to the target size: ceil(vectors / target) pieces.
        let expected_partitions =
            (expected_rows * VECTORS_PER_ROW).div_ceil(IndexType::IvfPq.target_partition_size());
        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            expected_partitions,
            "Expected the oversized multivector partition to split into {expected_partitions}, stats: {}",
            final_ctx.stats_json()
        );
        let partitions = final_ctx.stats()["indices"][0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        assert_eq!(partitions.len(), expected_partitions);
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

        // Both partitions hold BASE + APPEND rows and are split straight to the
        // target size in one optimize: ceil(rows / target) pieces each.
        let pieces_per_partition = (BASE_ROWS_PER_PARTITION + APPEND_ROWS_PER_PARTITION)
            .div_ceil(IndexType::IvfFlat.target_partition_size());
        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            2 * pieces_per_partition,
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
        assert_eq!(partitions.len(), 2 * pieces_per_partition);
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
        // Row 1600, the one retained below, has one vector in each partition.
        // Joining the partition that holds one of them must reassign only that
        // vector, not re-add the two the other partitions keep.
        let mut dataset = {
            let (batch, schema) = generate_clustered_multivec_batch(
                &cluster_sizes,
                &centroids,
                MULTIVEC_PER_ROW,
                0,
                Some(1600),
            );
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
            None,
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
        // so one optimize joins all of them but the largest after a scattered delete.
        let run = optimize_after_delete(400, 4, "id % 3 = 0", "id % 3 != 0").await;

        assert_eq!(
            run.num_partitions_after, 1,
            "optimize should have joined every undersized partition but one, got stats: {}",
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

    #[rstest]
    #[case::ivf_pq(VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        PQBuildParams::new(4, 4),
    ))]
    #[case::ivf_rq(VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::new(4),
        RQBuildParams::with_rotation_type(5, RQRotationType::Fast),
    ))]
    #[tokio::test]
    async fn test_prewarm_vector_index(#[case] params: VectorIndexParams) {
        use lance_io::assert_io_eq;

        const INDEX_NAME: &str = "my_idx";
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

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

        // Concurrent prewarms should single-flight each partition through the
        // cache loader and leave a complete warm cache for both callers.
        let (first, second) = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            tokio::join!(
                dataset.prewarm_index(INDEX_NAME),
                dataset.prewarm_index(INDEX_NAME)
            )
        })
        .await
        .expect("concurrent prewarms deadlocked");
        first.unwrap();
        second.unwrap();
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
