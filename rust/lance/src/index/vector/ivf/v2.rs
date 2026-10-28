// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index::scalar::RowAddrTranslator;
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

use crate::index::frag_reuse::ResolvedRemapping;
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
use lance_index::metrics::{IndexTimer, IndexTiming, LocalMetricsCollector, MetricsCollector};
use lance_index::prefilter::NoFilter;
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::{blocked_ex_code_bytes, padded_query_len};
use lance_index::vector::bq::partition_codes::PartitionCodesKey;
use lance_index::vector::bq::planes::EntryColumns;
use lance_index::vector::bq::rabit_ex_bits;
use lance_index::vector::bq::scan_stats;
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
    IndexFileKey, OriginLatencyClass, QueryResidual, QueryScratch, QueryScratchCapacity,
    QueryScratchPool, RabitRawQueryContext, ResidentAttach, ResidentColumns,
    ResidentColumnsSetting, ResidentStoreSize, VectorStore, entry_columns_setting,
    origin_latency_setting, resident_columns_setting, resident_lifetime_setting,
    resident_store_fits, shares_resident_store,
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
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, instrument};
use uuid::Uuid;

use super::{IvfIndexPartitionStatistics, IvfIndexStatistics, maybe_centroids_for_stats};

pub mod rabit_scan;

use rabit_scan::{RabitCacheLayout, RabitScan, RabitScanContext, RabitScanFactory};

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
        frag_reuse_index: Option<ResolvedRemapping>,
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
        frag_reuse_index: Option<ResolvedRemapping>,
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
    /// Whether [`IVFIndex::cache_partition_rows`] may cache the entry whole.
    /// An entry built for one read, from a code-only entry or of a native
    /// partition's planes, sets it false: its resident columns may be views
    /// of the resident store, charged their rows' bytes (which the store's
    /// cache entry already charges) yet keeping the whole store column
    /// alive, so such an entry must never be cached.
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

    /// Whether the entry's storage holds memory of a resident store, which
    /// no cached entry may (see [`shares_resident_store`]); the loaders of
    /// partition entries check it in debug builds.
    fn shares_resident_store(&self) -> bool {
        self.storage
            .to_batches()
            .is_ok_and(|mut batches| batches.any(|batch| shares_resident_store(&batch)))
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
    /// Session policy only; deliberately absent from cached index state.
    pub(crate) rabit_scan_factory: Option<Arc<dyn RabitScanFactory>>,
    /// The origin latency class the session declares for the IVF_RQ indexes
    /// it opens (`Session::index_origin_latency`), `None` when it declares
    /// none; see [`IVFIndex::origin_latency_at_open`].
    pub(crate) origin_latency_hint: Option<OriginLatencyClass>,
    /// Whether the session keeps the small columns of the IVF_RQ indexes it
    /// opens resident (`Session::index_resident_columns`), `auto` unless it
    /// sets one; `LANCE_RQ_RESIDENT_COLUMNS` set to `on` or `off` overrides
    /// it. See [`IVFIndex::resident_columns_at_open`].
    pub(crate) resident_columns: ResidentColumnsSetting,
    /// The index's namespace of the index cache without a fragment reuse
    /// segment, where an IVF_RQ index with resident small columns charges
    /// their store (see [`ResidentColumns::in_index_cache`]), so
    /// that a new fragment reuse index keeps the store. `None` keeps the
    /// store charged nowhere, shared with the file's other live indexes.
    pub(crate) file_cache: Option<LanceCache>,
    /// A strong reference acquired before opening the index can evict the
    /// cached store. Dropped if the index does not use resident columns.
    pub(crate) resident_columns_handle: Option<ResidentColumns>,
    /// The opening query's I/O stats, which a load of the resident store in
    /// a reconstruction from a cached [`IvfIndexState`] adds to; `None` when
    /// the opener records none. [`IVFIndex::try_new`] ignores it: its own
    /// scheduler records the load with the rest of the open's I/O
    /// ([`VectorIndex::open_io_stats`]).
    pub(crate) io_stats: Option<IoStats>,
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
/// resident store that the index cache charges `charge` bytes for, more than
/// its largest admissible entry, `max_entry_bytes`. Index handles keep this
/// allocation alive independently of cache capacity.
fn warn_resident_store_oversize(charge: u64, max_entry_bytes: Option<u64>) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        log::warn!(
            "LANCE_RQ_RESIDENT_COLUMNS=on keeps a resident store the index cache charges \
             {charge} bytes for, exceeding its {max_entry_bytes:?} byte entry limit; \
             live index handles retain this memory outside the cache"
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
    rabit_scan: Option<Arc<dyn RabitScan>>,
    /// Latency class of reads from the index's files, resolved when the
    /// index opens; the storage carries the same class.
    origin_latency: OriginLatencyClass,
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
            debug_assert!(
                !partition.shares_resident_store(),
                "partition {partition_id} would be cached with memory of a resident store"
            );
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
        let _stage_timer = IndexTimer::new(metrics, IndexTiming::PartitionPrepare);
        let (part_entry, ()) =
            tokio::try_join!(self.load_query_partition(partition_id, metrics), async {
                let _wait_timer = IndexTimer::new(metrics, IndexTiming::PrefilterWait);
                pre_filter.wait_for_ready().await
            },)?;
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
        let _stage_timer = IndexTimer::new(metrics, IndexTiming::PartitionPrepare);
        let part_entry = self.load_query_partition(partition_id, metrics).await?;
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
        let _stage_timer = IndexTimer::new(metrics, IndexTiming::Search);
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
        let _stage_timer = IndexTimer::new(metrics, IndexTiming::Search);
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

    fn global_heap_to_batch(
        heap: BinaryHeap<OrderedNode<u64>>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let _result_timer = IndexTimer::new(metrics, IndexTiming::ResultMaterialize);
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
        frag_reuse_index: Option<ResolvedRemapping>,
        file_metadata_cache: &LanceCache,
        index_cache: LanceCache,
        file_sizes: HashMap<String, u64>,
        context: IvfOpenContext,
    ) -> Result<Self> {
        let io_parallelism = object_store.io_parallelism();
        let origin_latency =
            Self::origin_latency_at_open(&object_store, context.origin_latency_hint)?;
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
        let storage = match frag_reuse_index {
            Some(ResolvedRemapping::V0(remapper)) => {
                IvfQuantizationStorage::try_new_with_remapper(storage_reader, Some(remapper))
                    .await?
            }
            Some(ResolvedRemapping::V1Translate { remapper, .. }) => {
                IvfQuantizationStorage::try_new_with_remapper(storage_reader, None)
                    .await?
                    .with_row_id_remapping(remapper)
            }
            // An untouched segment on a tagged dataset decodes the original
            // way, with no remapper.
            Some(ResolvedRemapping::V1Identity) | None => {
                IvfQuantizationStorage::try_new_with_remapper(storage_reader, None).await?
            }
        };
        let resident_columns = Self::resident_columns_at_open(
            &storage,
            context.file_cache.as_ref(),
            context.resident_columns,
        )?;
        let rabit_scan = if storage.supports_rabit_planes() {
            context
                .rabit_scan_factory
                .as_ref()
                .map(|factory| factory.create(&index_file, origin_latency))
                .transpose()?
        } else {
            None
        };
        let resident_store =
            Self::resident_store_at_open(resident_columns, &index_file, context).await?;
        let storage = storage
            .with_origin_latency(origin_latency)
            .with_resident_columns(resident_store)
            .with_resident_columns_enabled(resident_columns)
            .with_entry_columns(Self::entry_columns_at_open()?);
        // Load the store before any read needs it. The scheduler records the
        // load's requests with the rest of the open's I/O, so the load adds
        // them to no other stats.
        storage.load_resident_store(None).await?;

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
        // metadata, and the resident store's load) -- partition reads happen
        // later, during queries.  So its cumulative stats are exactly the
        // one-time index-open I/O.
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
            prepared_partitions: Arc::default(),
            rabit_scan,
            origin_latency,
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
    ) -> Result<Self> {
        let storage = storage.with_origin_latency(origin_latency);
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
            prepared_partitions: Arc::default(),
            rabit_scan: None,
            origin_latency,
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

    /// Replace the origin latency class so tests can exercise both classes
    /// over one store in one process.
    #[cfg(test)]
    pub(crate) fn with_origin_latency_for_test(
        mut self,
        origin_latency: OriginLatencyClass,
    ) -> Result<Self> {
        self.origin_latency = origin_latency;
        self.storage = self.storage.with_origin_latency(origin_latency);
        Ok(self)
    }

    /// Whether an index keeps the small columns of `storage`'s file
    /// resident in `file_cache`, whatever its origin latency class: when the
    /// setting admits the store (see [`ResidentColumnsSetting::admits`]).
    /// The setting is `LANCE_RQ_RESIDENT_COLUMNS` when it is `on` or `off`,
    /// and otherwise `session`, the opening session's
    /// ([`ResidentColumnsSetting::resolve_with_session`]). `off` keeps none
    /// and sizes nothing. `auto` keeps one when its charged bytes fit the
    /// cache entry limit; without a cache, only `on` keeps one, shared and
    /// uncharged. A store that does not fit counts
    /// `resident_columns_oversize`. Only IVF_RQ
    /// indexes read (and validate) `LANCE_RQ_RESIDENT_COLUMNS` and
    /// `LANCE_RQ_RESIDENT_LIFETIME`: the resident columns are told apart
    /// from the RaBitQ code columns, so other storages keep none.
    fn resident_columns_at_open(
        storage: &IvfQuantizationStorage<Q>,
        file_cache: Option<&LanceCache>,
        session: ResidentColumnsSetting,
    ) -> Result<bool> {
        if !storage.supports_rabit_planes() {
            return Ok(false);
        }
        resident_lifetime_setting()?;
        let setting =
            ResidentColumnsSetting::resolve_with_session(resident_columns_setting()?, session);
        if setting == ResidentColumnsSetting::Off {
            return Ok(false);
        }
        let store = storage.resident_store_size()?;
        let Some(cache) = file_cache else {
            return Ok(setting == ResidentColumnsSetting::On && store.bytes > 0);
        };
        let max_entry_bytes = cache.max_entry_bytes();
        if store.bytes > 0 && !resident_store_fits(store.charge, max_entry_bytes) {
            scan_stats::counters().resident_columns_oversize.incr();
            if setting == ResidentColumnsSetting::On {
                warn_resident_store_oversize(store.charge, max_entry_bytes);
            }
        }
        Ok(setting.admits(store, max_entry_bytes))
    }

    /// The resident store handle of an index opening on index file `file`:
    /// when the index keeps its small columns `resident`, the store every
    /// live index of the file shares and offered to the ordinary file cache.
    /// The opener's strong reference bridges eviction during index opening.
    /// Otherwise use an independent store that loads only if residency is
    /// enabled later.
    /// The opening index loads the bound store next, unless it is loaded already.
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
                    context.resident_columns_handle,
                    resident_lifetime_setting()?,
                )
                .await
            }
            None => ResidentColumns::shared(file),
        })
    }

    /// Whether reads keep the index's small columns resident and fetch only
    /// its code columns from the file, resolved when the index
    /// opened; see `LANCE_RQ_RESIDENT_COLUMNS`.
    pub fn resident_columns_enabled(&self) -> bool {
        self.storage.resident_columns_enabled()
    }

    /// Bytes the index's resident columns take, whether or not it keeps
    /// them; see [`lance_index::vector::storage::resident_columns_bytes`].
    /// Zero for a storage layout that does not support resident columns.
    pub fn resident_columns_bytes(&self) -> u64 {
        self.storage.resident_columns_bytes()
    }

    /// The size of the index's resident store, whether or not it keeps
    /// one, and what an index cache charges for it once loaded, which the
    /// open compared with the cache's entry limit; see
    /// [`ResidentColumnsSetting::admits`]. Zero for unsupported storage layouts.
    pub fn resident_store_size(&self) -> Result<ResidentStoreSize> {
        self.storage.resident_store_size()
    }

    /// Replace whether reads keep the small columns resident, so tests can
    /// open both in one process. Enabling loads the store through the
    /// index's handle unless it is loaded, as an open that keeps the columns
    /// resident does: a store of the index's own when the open kept none.
    /// Disabling also gives the index a store handle of its own, so that it
    /// never shares the store the open bound.
    #[cfg(test)]
    pub(crate) async fn with_resident_columns_for_test(mut self, enabled: bool) -> Result<Self> {
        if !enabled {
            self.storage = self
                .storage
                .with_resident_columns(ResidentColumns::default());
        }
        self.storage = self.storage.with_resident_columns_enabled(enabled);
        self.storage.load_resident_store(None).await?;
        Ok(self)
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

    /// How reads attach the resident rows to the whole partitions and
    /// planes they never cache; see
    /// [`IvfQuantizationStorage::resident_attach`].
    pub fn resident_attach(&self) -> ResidentAttach {
        self.storage.resident_attach()
    }

    /// Replace how reads attach the resident rows, so tests can run both
    /// modes in one process.
    #[cfg(test)]
    pub(crate) fn with_resident_attach_for_test(mut self, resident_attach: ResidentAttach) -> Self {
        self.storage = self.storage.with_resident_attach(resident_attach);
        self
    }

    /// What the index's cache entries hold, resolved when it opened; see
    /// `LANCE_RQ_ENTRY_COLUMNS`. [`EntryColumns::Codes`] only for an index
    /// whose small columns are resident: its sign and ex plane entries,
    /// or a native flat index's partition entries, which
    /// are then [`PartitionCodes`](lance_index::vector::bq::partition_codes::PartitionCodes).
    /// A native index with a graph caches whole partitions.
    pub fn entry_columns(&self) -> EntryColumns {
        if self.uses_rabit_plane_cache() || self.caches_partition_codes() {
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
        self.load_partition_attached(partition_id, write_cache, self.resident_attach(), metrics)
            .await
    }

    /// [`Self::load_partition`], with the resident rows of a code-only
    /// partition attached as `attach` says; see
    /// [`IvfQuantizationStorage::partition_from_codes`].
    async fn load_partition_attached(
        &self,
        partition_id: usize,
        write_cache: bool,
        attach: ResidentAttach,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        let _stage_timer = IndexTimer::new(metrics, IndexTiming::PartitionLoad);
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::index(format!(
                "partition id {} is out of range of {} partitions",
                partition_id,
                self.ivf.num_partitions()
            )));
        }

        if self.caches_partition_codes() {
            return self
                .load_code_only_partition(partition_id, write_cache, attach, metrics)
                .await;
        }

        let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);

        if write_cache {
            let result = self
                .index_cache
                .get_or_insert_with_key_hit(cache_key, || async {
                    info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
                    metrics.record_part_load();
                    let entry = self
                        .load_partition_entry(partition_id, metrics.io_stats())
                        .await?;
                    debug_assert!(
                        !entry.shares_resident_store(),
                        "partition {partition_id} would be cached with memory of a resident store"
                    );
                    Ok(entry)
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
    /// from it, with the resident rows attached as `attach` says, is built
    /// again on every read and never cached; see
    /// [`IvfQuantizationStorage::load_partition_cached`].
    async fn load_code_only_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        attach: ResidentAttach,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        let io_stats = metrics.io_stats();
        let (storage, hit) = self
            .storage
            .load_partition_cached(
                partition_id,
                &self.index_cache,
                write_cache,
                attach,
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

    async fn load_query_partition(
        &self,
        partition_id: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        if !self.uses_rabit_plane_cache() {
            return self.load_partition(partition_id, true, metrics).await;
        }
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::invalid_input(format!(
                "partition id {partition_id} is out of range of {} partitions",
                self.ivf.num_partitions()
            )));
        }
        let mut entry = self
            .load_plane_partition_entry(partition_id, metrics.io_stats())
            .await?;
        entry.cache_whole_partition = false;
        Ok(Arc::new(entry))
    }

    /// A strategy's native plane cache must also serve prewarm and eager
    /// fallback; otherwise the scan would bypass already-warmed partitions.
    fn uses_rabit_plane_cache(&self) -> bool {
        self.rabit_scan
            .as_ref()
            .is_some_and(|scan| scan.cache_layout() == RabitCacheLayout::Planes)
    }

    async fn load_partition_entry(
        &self,
        partition_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<PartitionEntry<S, Q>> {
        let batch = self
            .read_sub_index_batch(
                partition_id,
                self.read_projection.as_ref(),
                io_stats.clone(),
            )
            .await?;
        let idx = S::load(batch)?;
        let storage = self.load_partition_storage(partition_id, io_stats).await?;
        Ok(PartitionEntry::new(idx, storage))
    }

    async fn load_plane_partition_entry(
        &self,
        partition_id: usize,
        io_stats: Option<IoStats>,
    ) -> Result<PartitionEntry<S, Q>> {
        let index = self.load_sub_index(partition_id, io_stats.clone()).await?;
        let storage = self
            .storage
            .load_rabit_partition_planes(partition_id, &self.index_cache, io_stats)
            .await?;
        Ok(PartitionEntry::new(index, storage))
    }

    async fn load_sub_index(&self, partition_id: usize, io_stats: Option<IoStats>) -> Result<S> {
        let batch = self
            .read_sub_index_batch(partition_id, self.read_projection.as_ref(), io_stats)
            .await?;
        S::load(batch)
    }

    /// Read the serialized sub-index of a partition, with its metadata attached.
    ///
    /// `projection` narrows the read to the columns search needs; `None` reads
    /// every column the writer emitted, which a caller rewriting the partition
    /// has to carry through. Bypasses the partition cache.
    pub(crate) async fn read_sub_index_batch(
        &self,
        partition_id: usize,
        projection: Option<&ReaderProjection>,
        io_stats: Option<IoStats>,
    ) -> Result<RecordBatch> {
        // `concat_batches` indexes the batches by this schema's field positions
        // without comparing the two, so the schema has to describe exactly what
        // was read: the full file schema over a projected read would index past
        // the last column.
        let schema = Arc::new(match projection {
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
                    let batches = stream.try_collect::<Vec<_>>().await?;
                    concat_batches(&schema, batches.iter())?
                }
            }
        };
        Ok(batch.add_metadata(
            S::metadata_key().to_owned(),
            self.sub_index_metadata[partition_id].clone(),
        )?)
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
                                    let entry = self
                                        .materialize_prewarm_partition(follower_id, batches)
                                        .await?;
                                    debug_assert!(
                                        !entry.shares_resident_store(),
                                        "prewarmed partition {follower_id} holds memory of a resident store"
                                    );
                                    Ok(entry)
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
                    let entry = self
                        .materialize_prewarm_partition(partition_id, leader_batches)
                        .await?;
                    debug_assert!(
                        !entry.shares_resident_store(),
                        "prewarmed partition {partition_id} holds memory of a resident store"
                    );
                    Ok(entry)
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
    /// holds nothing to warm, and the index loaded the resident store when
    /// it opened.
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
                debug_assert!(
                    !shares_resident_store(&codes.0),
                    "prewarmed codes of partition {partition} hold memory of a resident store"
                );
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

    /// [`Index::prewarm`] of an index that is not native: its partitions,
    /// in byte windows ([`IVF_PREWARM_WINDOW_SIZE_ENV`]) read with bounded
    /// parallelism.
    async fn prewarm_partitions(&self) -> Result<()> {
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
        // Every IVF index loaded its resident store when it opened, and its
        // handle keeps it for the index's life, so prewarm has none to load.
        debug_assert!(
            !self.storage.resident_columns_enabled()
                || self.storage.resident_columns().loaded_bytes().is_some(),
            "IVF_RQ index {} prewarms with a resident store its open did not load",
            self.uuid
        );
        if self.uses_rabit_plane_cache() {
            self.storage.prewarm_planes(&self.index_cache).await?;
        } else {
            self.prewarm_partitions().await?;
        }
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
        let (part_entry, pre_filter) = {
            // Match split preparation without counting CPU queueing or search.
            let _prepare_timer = IndexTimer::new(metrics, IndexTiming::PartitionPrepare);
            let part_entry = self.load_query_partition(partition_id, metrics).await?;
            {
                let _wait_timer = IndexTimer::new(metrics, IndexTiming::PrefilterWait);
                pre_filter.wait_for_ready().await?;
            }
            let pre_filter = Self::prefilter_for_partition(
                &self.index_cache,
                partition_id,
                &part_entry,
                pre_filter,
            )
            .await?;
            (part_entry, pre_filter)
        };

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
        let queued = Instant::now();
        let (batch, local_metrics) = spawn_cpu(move || {
            let local_metrics = LocalMetricsCollector::default();
            local_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
            let search_timer = IndexTimer::new(&local_metrics, IndexTiming::Search);
            let param = (&query).into();
            let refine_factor = query.refine_factor.unwrap_or(1) as usize;
            let k = query.k * refine_factor;
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
            drop(search_timer);
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

        let prepare_parallelism = get_num_compute_intensive_cpus().max(1);
        let raw_query_context = self.prepare_rq_raw_query_context(&query.key)?;

        if control.is_none() && S::supports_global_topk_heap() {
            if let Some(strategy) = &self.rabit_scan {
                let erased: Arc<dyn std::any::Any + Send + Sync> = self.clone();
                if let Ok(index) = erased.downcast::<IVFIndex<FlatIndex, RabitQuantizer>>() {
                    let context = Arc::new(RabitScanContext::new(
                        index,
                        raw_query_context.clone(),
                        query.clone(),
                        partitions.clone(),
                        q_c_dists.clone(),
                        start_idx..end_idx,
                        pre_filter.clone(),
                        metrics.clone(),
                    ));
                    if let Some(batch) = strategy.search(context).await? {
                        return Ok(Box::pin(RecordBatchStreamAdapter::new(
                            VECTOR_RESULT_SCHEMA.clone(),
                            stream::once(async move { Ok(batch) }),
                        )));
                    }
                }
            }
            let heap_capacity = query.k * query.refine_factor.unwrap_or(1) as usize;
            {
                let _wait_timer = IndexTimer::new(metrics.as_ref(), IndexTiming::PrefilterWait);
                pre_filter.wait_for_ready().await?;
            }
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
                let queued = Instant::now();
                let score = spawn_cpu(move || -> Result<BinaryHeap<OrderedNode<u64>>> {
                    search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
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
                Self::global_heap_to_batch(heap, metrics.as_ref())?
            } else {
                let result_metrics = metrics.clone();
                let queued = Instant::now();
                spawn_cpu(move || {
                    result_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                    Self::global_heap_to_batch(heap, result_metrics.as_ref())
                })
                .await?
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
                    let queued = Instant::now();
                    let search_output = spawn_cpu(move || {
                        search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
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

        {
            let _wait_timer = IndexTimer::new(metrics.as_ref(), IndexTiming::PrefilterWait);
            pre_filter.wait_for_ready().await?;
        }

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
        let load_pre_filter = pre_filter;
        let mut loaded_chunks = stream::iter(assignment_list)
            .map(move |(part_id, probing_queries)| {
                let index = load_index.clone();
                let metrics = load_metrics.clone();
                let pre_filter = load_pre_filter.clone();
                async move {
                    let part_id = part_id as usize;
                    let part_entry = index
                        .load_query_partition(part_id, metrics.as_ref())
                        .await?;
                    let pre_filter = Self::prefilter_for_partition(
                        &index.index_cache,
                        part_id,
                        &part_entry,
                        pre_filter,
                    )
                    .await?;
                    Result::Ok((part_id, part_entry, probing_queries, pre_filter))
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
            let base_queries = base_queries.clone();
            let raw_query_contexts = raw_query_contexts.clone();
            let scratch_pool = self.scratch_pool.clone();
            let search_metrics = metrics.clone();
            let queued = Instant::now();
            let score = spawn_cpu(move || -> Result<Vec<BinaryHeap<OrderedNode<u64>>>> {
                search_metrics.record_timing(IndexTiming::CpuQueueWait, queued.elapsed());
                scratch_pool.with_scratch(|scratch| -> Result<()> {
                    for (part_id, part_entry, probing_queries, pre_filter) in &chunk {
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
            .map(|heap| Self::global_heap_to_batch(heap, metrics.as_ref()))
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
        // The stream leaves the index: its caller may hold the batches after
        // the index, and its lease on the resident store, are gone, where a
        // view of the store would keep a whole store column alive, uncharged.
        let partition = self
            .load_partition_attached(partition_id, false, ResidentAttach::Copy, metrics)
            .await?;
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

    fn supports_pairwise_vectors(&self) -> bool {
        true
    }

    async fn prepare_pairwise_partition(
        &self,
        partition_id: usize,
        batch_size: usize,
        memory_limit: usize,
        spill_store: &dyn lance_io::spill::SpillStore,
    ) -> Result<lance_index::vector::pairwise::PairwisePartition> {
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::invalid_input(format!(
                "partition_id={partition_id} out of range 0..{}",
                self.ivf.num_partitions()
            )));
        }
        let centroid = self.ivf.centroid(partition_id).ok_or_else(|| {
            Error::invalid_input(format!("partition_id={partition_id} has no centroid"))
        })?;
        self.storage
            .prepare_pairwise_partition(
                partition_id,
                batch_size,
                memory_limit,
                centroid,
                spill_store,
            )
            .await
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

    async fn remap_streaming(&mut self, _translator: &RowAddrTranslator) -> Result<()> {
        // No mapping to materialize for an index that cannot be remapped.
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
    frag_reuse_index: Option<ResolvedRemapping>,
    context: IvfOpenContext,
) -> Result<Arc<dyn VectorIndex>> {
    let io_parallelism = object_store.io_parallelism();
    let origin_latency =
        IVFIndex::<S, Q>::origin_latency_at_open(&object_store, context.origin_latency_hint)?;

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

    let make_storage = |legacy| {
        IvfQuantizationStorage::from_cached_with_remapper(
            aux_reader,
            state.aux_ivf.clone(),
            state.metadata.clone(),
            state.distance_type,
            legacy,
        )
    };
    let storage = match frag_reuse_index {
        Some(ResolvedRemapping::V0(remapper)) => make_storage(Some(remapper)),
        Some(ResolvedRemapping::V1Translate { remapper, .. }) => {
            make_storage(None).with_row_id_remapping(remapper)
        }
        Some(ResolvedRemapping::V1Identity) | None => make_storage(None),
    };
    // Bind the store as an open does: the state holds none, so a state read
    // back from a persistent tier binds to the one the index cache holds or
    // the file's live indexes share, and any state pins nothing.
    let resident = IVFIndex::<S, Q>::resident_columns_at_open(
        &storage,
        context.file_cache.as_ref(),
        context.resident_columns,
    )?;
    let io_stats = context.io_stats.clone();
    let rabit_scan = if storage.supports_rabit_planes() {
        context
            .rabit_scan_factory
            .as_ref()
            .map(|factory| factory.create(&index_file, origin_latency))
            .transpose()?
    } else {
        None
    };
    let resident_columns =
        IVFIndex::<S, Q>::resident_store_at_open(resident, &index_file, context).await?;
    let storage = storage
        .with_resident_columns(resident_columns)
        .with_resident_columns_enabled(resident)
        .with_entry_columns(IVFIndex::<S, Q>::entry_columns_at_open()?);
    // Load the store, unless a live index or the cache holds it, as an open
    // does. A reconstruction reports no open I/O, so the load's requests go
    // to the opening query's stats.
    storage.load_resident_store(io_stats.as_ref()).await?;
    let rq_search_cache = IVFIndex::<S, Q>::rq_search_cache_from_state(state, &storage)?;

    let parsed_uuid = Uuid::parse_str(&state.uuid)
        .map_err(|e| Error::index(format!("Invalid UUID in IvfIndexState: {e}")))?;
    let mut index = IVFIndex::<S, Q>::from_cached_state(
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
    )?;
    index.rabit_scan = rabit_scan;
    Ok(Arc::new(index))
}

#[cfg(test)]
mod tests;
