// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row map decoding for stable-partition rewrites.
//!
//! The file holds one nullable u16 label per physical source row in scan
//! order. Labels select a destination fragment; nulls mark source deletions.
//! A global buffer stores cumulative counts for blocks of rows, so lookups
//! need only read the labels of the blocks they touch.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt16Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use arrow_schema::DataType;
use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, WeakLanceCache};
use lance_core::datatypes::Schema;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::stable_partition::{
    CountsMatrix, DEFAULT_BLOCK_ROWS, SweepTranslator, translate_in_block,
};
use lance_core::{Error, Result};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{CachedFileMetadata, FileReader, FileReaderOptions, FilterExpression};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

/// The single column of a row map file.
pub const LABEL_COLUMN: &str = "label";

/// Namespace prefix for cached row-map label chunks in the index cache.
pub const ROW_MAP_CACHE_PREFIX: &str = "row-map-blocks";

/// Number of logical row-map blocks per cache chunk (the cache unit).
///
/// A block holds [`DEFAULT_BLOCK_ROWS`] (64K) rows; one label is a u16, so a
/// full block is ~128 KiB decoded. Grouping 32 blocks makes a ~4 MiB chunk:
/// this keeps the number of cache entries small (~48 at 100M rows, ~477 at 1B)
/// while turning what would be many small block reads into one big sequential
/// range read per chunk on a cold miss. Whole-file caching (one 2 GiB blob at
/// 1B rows) would blow the LRU; per-block caching (thousands of entries) would
/// bloat the entry count. This is the middle ground.
pub const ROW_MAP_CACHE_CHUNK_BLOCKS: usize = 32;

/// Decoded size of one full chunk: the smallest index-cache capacity at which
/// a chunk entry can be retained at all. A cache that is known to be smaller
/// must not be used for chunks, since every block request would then read a
/// whole chunk only to discard it.
pub const ROW_MAP_CACHE_CHUNK_BYTES: usize =
    ROW_MAP_CACHE_CHUNK_BLOCKS * DEFAULT_BLOCK_ROWS as usize * std::mem::size_of::<u16>();

/// Cached decoded labels for one chunk (`ROW_MAP_CACHE_CHUNK_BLOCKS` contiguous
/// blocks) of a row map file. The concrete `UInt16Array` does not implement
/// `DeepSizeOf`, so this newtype delegates to the `dyn Array` impl for weight
/// accounting in the index cache.
#[derive(Debug, Clone)]
pub struct RowMapChunk {
    labels: UInt16Array,
}

/// Labels a reader holds across the blocks of one request (see
/// [`RowMapReader::block_labels_reusing`]): the contiguous block range they
/// cover and the labels themselves, a cached chunk or a one-off span read.
#[derive(Debug, Clone)]
pub struct HeldLabels {
    blocks: Range<usize>,
    labels: Arc<RowMapChunk>,
}

impl DeepSizeOf for RowMapChunk {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        (&self.labels as &dyn Array).deep_size_of_children(context)
    }
}

/// Index-cache key for one row-map label chunk. Keyed by a stable per-map
/// identity plus the chunk index so distinct row maps never collide and the
/// entry survives across queries and dataset snapshots.
#[derive(Clone)]
pub struct RowMapChunkKey {
    /// Stable identity of the row map (e.g. the transition fingerprint).
    pub row_map_id: [u8; 32],
    /// Chunk index = block / [`ROW_MAP_CACHE_CHUNK_BLOCKS`].
    pub chunk: usize,
}

impl CacheKey for RowMapChunkKey {
    type ValueType = RowMapChunk;

    fn key(&self) -> Cow<'_, str> {
        format!("{:x?}:{}", self.row_map_id, self.chunk).into()
    }

    fn type_name() -> &'static str {
        "RowMapChunk"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-row-map-chunk", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(&self.row_map_id);
        builder.write_u64(self.chunk as u64);
    }
}

/// A handle that routes row-map label reads through the shared index cache.
///
/// Cloned cheaply; carries the stable per-map identity and the (already
/// namespaced) cache. Absence of this handle on a [`RowMapReader`] means the
/// reader reads labels directly, byte-identical to the pre-cache behavior.
#[derive(Clone)]
pub struct RowMapBlockCache {
    cache: WeakLanceCache,
    row_map_id: [u8; 32],
}

impl RowMapBlockCache {
    /// Wrap a cache for row-map chunk storage. The cache is namespaced with
    /// [`ROW_MAP_CACHE_PREFIX`] so chunk entries never collide with other
    /// index-cache values.
    pub fn new(cache: WeakLanceCache, row_map_id: [u8; 32]) -> Self {
        Self {
            cache: cache.with_key_prefix(ROW_MAP_CACHE_PREFIX),
            row_map_id,
        }
    }
}

/// Schema-metadata key holding the global buffer index of the encoded counts
/// matrix.
pub const COUNTS_BUFFER_INDEX_KEY: &str = "lance:stable_partition:counts_buffer_index";

/// Reads the schema, counts buffer, and requested label ranges of a row map.
/// The native implementation uses a Lance file; other storage adapters can
/// supply the same metadata and Arrow batches.
#[async_trait]
pub trait RowMapFile: Send + Sync {
    /// The persisted schema, including the counts buffer's index.
    fn schema(&self) -> &Schema;
    /// The number of physical source rows in the file.
    fn num_rows(&self) -> u64;
    /// Read the global buffer at the index recorded by the schema.
    async fn read_global_buffer(&self, index: u32) -> Result<Bytes>;
    /// Read the label column over one range of physical source rows.
    async fn read_range(&self, range: Range<usize>) -> Result<RecordBatch>;
    /// Read sorted, nonoverlapping row ranges into one batch in range order.
    async fn read_ranges(&self, ranges: &[Range<usize>]) -> Result<RecordBatch>;
}

#[async_trait]
impl RowMapFile for FileReader {
    fn schema(&self) -> &Schema {
        self.schema()
    }

    fn num_rows(&self) -> u64 {
        self.num_rows()
    }

    async fn read_global_buffer(&self, index: u32) -> Result<Bytes> {
        self.read_global_buffer(index).await
    }

    async fn read_range(&self, range: Range<usize>) -> Result<RecordBatch> {
        read_label_batches(self, ReadBatchParams::Range(range), u32::MAX).await
    }

    async fn read_ranges(&self, ranges: &[Range<usize>]) -> Result<RecordBatch> {
        let total_rows = ranges.iter().map(|range| range.len() as u64).sum::<u64>();
        let batch_size = u32::try_from(total_rows).unwrap_or(u32::MAX).max(1);
        let ranges = ranges
            .iter()
            .map(|range| range.start as u64..range.end as u64)
            .collect();
        read_label_batches(self, ReadBatchParams::Ranges(ranges), batch_size).await
    }
}

async fn read_label_batches(
    reader: &FileReader,
    params: ReadBatchParams,
    batch_size: u32,
) -> Result<RecordBatch> {
    // Opening validates that the entire schema is the single label column.
    let batches: Vec<_> = reader
        .read_stream(params, batch_size, 16, FilterExpression::no_filter())
        .await?
        .try_collect()
        .await?;
    match batches.as_slice() {
        [] => Ok(RecordBatch::new_empty(Arc::new(
            reader.schema().as_ref().into(),
        ))),
        [batch] => Ok(batch.clone()),
        [first, ..] => Ok(arrow::compute::concat_batches(
            first.schema_ref(),
            &batches,
        )?),
    }
}

struct RowMapFileMetadataKey<'a>(&'a Path);

impl CacheKey for RowMapFileMetadataKey<'_> {
    type ValueType = CachedFileMetadata;

    fn key(&self) -> Cow<'_, str> {
        self.0.as_ref().into()
    }

    fn type_name() -> &'static str {
        "RowMapFileMetadata"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.table.fri-row-map-file-metadata", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.0.as_ref());
    }
}

pub(super) async fn open_file(
    store: Arc<ObjectStore>,
    path: &Path,
    file_size: &CachedFileSize,
    metadata_cache: &LanceCache,
) -> Result<FileReader> {
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    let file_scheduler = scheduler.open_file(path, file_size).await?;

    // A map is immutable. The footer is reused across validation and cached
    // mappings, even when a mapping reader has been evicted.
    let metadata = metadata_cache
        .get_or_insert_with_key(RowMapFileMetadataKey(path), || {
            FileReader::read_all_metadata(&file_scheduler)
        })
        .await?;
    let options = FileReaderOptions::default();
    let io = LanceEncodingsIo::new(file_scheduler).with_read_chunk_size(options.read_chunk_size);
    FileReader::try_open_with_file_metadata(
        Arc::new(io),
        path.clone(),
        None,
        Arc::default(),
        metadata,
        metadata_cache,
        options,
    )
    .await
}

fn label_column(batch: &RecordBatch) -> Result<&UInt16Array> {
    batch
        .column_by_name(LABEL_COLUMN)
        .and_then(|column| column.as_primitive_opt::<UInt16Type>())
        .ok_or_else(|| {
            Error::corrupt_file_named(
                "row_map",
                "row map read returned a batch without a u16 label column",
            )
        })
}

/// Reader over a row map file: one tail read at open, block reads on demand.
pub struct RowMapReader {
    reader: Arc<dyn RowMapFile>,
    counts: CountsMatrix,
    /// When present, block reads are served from (and populated into) the shared
    /// index cache in ~4 MiB chunks. When absent, reads go straight to the file.
    block_cache: Option<RowMapBlockCache>,
}

impl std::fmt::Debug for RowMapReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowMapReader")
            .field("counts", &self.counts)
            .finish_non_exhaustive()
    }
}

impl RowMapReader {
    /// Open a row map file: decodes and validates the counts matrix from its
    /// global buffer without touching any label data. A corrupt counts buffer
    /// must fail here rather than translate rows to wrong addresses.
    pub async fn open(reader: Arc<dyn RowMapFile>) -> Result<Self> {
        Self::open_with_cache(reader, None).await
    }

    /// Open a row map file, optionally routing label reads through the shared
    /// index cache. With `block_cache = None` this is identical to [`open`] and
    /// every label read hits the file; with a cache, reads are served from and
    /// populated into ~4 MiB chunks.
    ///
    /// [`open`]: Self::open
    pub async fn open_with_cache(
        reader: Arc<dyn RowMapFile>,
        block_cache: Option<RowMapBlockCache>,
    ) -> Result<Self> {
        let schema = reader.schema();
        let [label_field] = schema.fields.as_slice() else {
            return Err(Error::corrupt_file_named(
                "row_map",
                format!(
                    "row map file must hold exactly one column, found {}",
                    schema.fields.len()
                ),
            ));
        };
        if label_field.name != LABEL_COLUMN
            || label_field.data_type() != DataType::UInt16
            || !label_field.nullable
        {
            return Err(Error::corrupt_file_named(
                "row_map",
                format!(
                    "row map file must hold a single nullable u16 {LABEL_COLUMN} column, found {} of type {}",
                    label_field.name,
                    label_field.data_type()
                ),
            ));
        }
        let buffer_index = schema
            .metadata
            .get(COUNTS_BUFFER_INDEX_KEY)
            .and_then(|index| index.parse::<u32>().ok())
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    "row_map",
                    format!("row map file is missing the {COUNTS_BUFFER_INDEX_KEY} metadata"),
                )
            })?;
        let counts = CountsMatrix::decode(&reader.read_global_buffer(buffer_index).await?)?;
        counts.validate()?;
        if reader.num_rows() != counts.total_rows() {
            return Err(Error::corrupt_file_named(
                "row_map",
                format!(
                    "row map file holds {} labels but its counts describe {} source rows",
                    reader.num_rows(),
                    counts.total_rows()
                ),
            ));
        }
        Ok(Self {
            reader,
            counts,
            block_cache,
        })
    }

    /// The validated counts matrix decoded at open: per-destination totals
    /// and per-block counter bases without any label IO.
    pub fn counts(&self) -> &CountsMatrix {
        &self.counts
    }

    /// The decoded labels of one block.
    ///
    /// With a block cache attached, the block's enclosing ~4 MiB chunk is loaded
    /// (or served from cache) with one range read, and the block's labels are
    /// sliced out. Without a cache, exactly the block's rows are read, identical
    /// to the pre-cache behavior.
    pub async fn block_labels(&self, block: usize) -> Result<UInt16Array> {
        self.block_labels_reusing(block, block, &mut None).await
    }

    /// [`Self::block_labels`] for a caller that visits blocks in order and
    /// wants the labels it just loaded to serve the next blocks. `last_block`
    /// is the end of the run of adjacent blocks, starting at `block`, that
    /// the caller will request next (every block in between included); `held`
    /// carries the loaded labels between calls, and a block inside the held
    /// span never touches the cache or the file. With a block cache the span
    /// is the block's chunk, so a request performs one read per chunk whether
    /// or not the cache retains it. Without a block cache the span is the
    /// run `block..=last_block`, bounded to a chunk's worth, so a point
    /// lookup reads exactly its block, a run of adjacent blocks is read in
    /// one range read, and a sparse request never reads a block it does not
    /// touch (its caller passes each run's end, not the request's last
    /// block).
    pub async fn block_labels_reusing(
        &self,
        block: usize,
        last_block: usize,
        held: &mut Option<HeldLabels>,
    ) -> Result<UInt16Array> {
        let span = match held {
            Some(span) if span.blocks.contains(&block) => span,
            _ => {
                let loaded = match &self.block_cache {
                    Some(block_cache) => {
                        let chunk_idx = block / ROW_MAP_CACHE_CHUNK_BLOCKS;
                        let first = chunk_idx * ROW_MAP_CACHE_CHUNK_BLOCKS;
                        HeldLabels {
                            blocks: first
                                ..(first + ROW_MAP_CACHE_CHUNK_BLOCKS)
                                    .min(self.counts.num_blocks()),
                            labels: self.chunk_labels(block_cache, chunk_idx).await?,
                        }
                    }
                    None => {
                        let end = last_block
                            .max(block)
                            .min(block + ROW_MAP_CACHE_CHUNK_BLOCKS - 1)
                            .min(self.counts.num_blocks() - 1);
                        let rows =
                            self.counts.block_range(block).start..self.counts.block_range(end).end;
                        HeldLabels {
                            blocks: block..end + 1,
                            labels: Arc::new(RowMapChunk {
                                labels: self
                                    .read_labels(rows.start as usize..rows.end as usize)
                                    .await?,
                            }),
                        }
                    }
                };
                held.insert(loaded)
            }
        };
        // Slice the requested block out of the held labels.
        let span_start = self.counts.block_range(span.blocks.start).start;
        let block_range = self.counts.block_range(block);
        let offset = (block_range.start - span_start) as usize;
        let len = (block_range.end - block_range.start) as usize;
        Ok(span.labels.labels.slice(offset, len))
    }

    /// Read a contiguous row range of labels straight from the file.
    async fn read_labels(&self, range: Range<usize>) -> Result<UInt16Array> {
        let batch = self.reader.read_range(range).await?;
        Ok(label_column(&batch)?.clone())
    }

    /// The row range covered by one chunk, clamped to the total row count (the
    /// last chunk may hold fewer than a full complement of blocks).
    fn chunk_row_range(&self, chunk_idx: usize) -> Range<u64> {
        let first_block = chunk_idx * ROW_MAP_CACHE_CHUNK_BLOCKS;
        let last_block =
            ((chunk_idx + 1) * ROW_MAP_CACHE_CHUNK_BLOCKS - 1).min(self.counts.num_blocks() - 1);
        self.counts.block_range(first_block).start..self.counts.block_range(last_block).end
    }

    /// Load one chunk of labels, serving from the index cache when present and
    /// otherwise reading the chunk's full row range with one sequential read.
    /// Concurrent loads of the same chunk are deduplicated by the cache.
    async fn chunk_labels(
        &self,
        block_cache: &RowMapBlockCache,
        chunk_idx: usize,
    ) -> Result<Arc<RowMapChunk>> {
        let key = RowMapChunkKey {
            row_map_id: block_cache.row_map_id,
            chunk: chunk_idx,
        };
        let range = self.chunk_row_range(chunk_idx);
        let reader = self.reader.clone();
        block_cache
            .cache
            .get_or_insert_with_key(key, || async move {
                let batch = reader
                    .read_range(range.start as usize..range.end as usize)
                    .await?;
                Ok(RowMapChunk {
                    labels: label_column(&batch)?.clone(),
                })
            })
            .await
    }

    /// Translate one physical source row (concatenated scan order). Returns
    /// `None` when the row was deleted at the source, otherwise
    /// `(destination index, destination row offset)`. Reads one block.
    pub async fn translate(&self, row: u64) -> Result<Option<(u16, u32)>> {
        self.check_row(row)?;
        let block = self.counts.block_of(row);
        let labels = self.block_labels(block).await?;
        let pos = (row - self.counts.block_range(block).start) as usize;
        translate_in_block(&self.counts, block, labels.values(), labels.nulls(), pos)
    }

    /// Translate a batch of physical source rows with one coalesced read of
    /// the touched blocks.
    pub async fn translate_many(&self, rows: &[u64]) -> Result<Vec<Option<(u16, u32)>>> {
        for &row in rows {
            self.check_row(row)?;
        }
        let mut blocks: Vec<usize> = rows.iter().map(|&row| self.counts.block_of(row)).collect();
        blocks.sort_unstable();
        blocks.dedup();
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let ranges: Vec<Range<usize>> = blocks
            .iter()
            .map(|&block| {
                let range = self.counts.block_range(block);
                range.start as usize..range.end as usize
            })
            .collect();
        // One coalesced read; the result concatenates the requested block
        // ranges in order.
        let batch = self.reader.read_ranges(&ranges).await?;
        let labels = label_column(&batch)?;
        let mut block_starts = Vec::with_capacity(blocks.len());
        let mut start = 0usize;
        for range in &ranges {
            block_starts.push(start);
            start += range.end - range.start;
        }
        rows.iter()
            .map(|&row| {
                let block = self.counts.block_of(row);
                let slot = blocks.binary_search(&block).map_err(|_| {
                    Error::internal(format!("row-map block {block} missing from the request"))
                })?;
                let range = &ranges[slot];
                let block_labels = labels.slice(block_starts[slot], range.end - range.start);
                let pos = (row - range.start as u64) as usize;
                translate_in_block(
                    &self.counts,
                    block,
                    block_labels.values(),
                    block_labels.nulls(),
                    pos,
                )
            })
            .collect()
    }

    /// Translate every physical source row in `rows` in order, block by
    /// block, calling `f(row, translation)` for each. O(1) per row after the
    /// per-block read.
    pub async fn sweep(
        &self,
        rows: Range<u64>,
        mut f: impl FnMut(u64, Option<(u16, u32)>) -> Result<()>,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.check_row(rows.start)?;
        self.check_row(rows.end - 1)?;
        let start_block = self.counts.block_of(rows.start);
        let end_block = self.counts.block_of(rows.end - 1);
        // Blocks are read one at a time on purpose: a sweep is bulk work and
        // bounded memory matters more than per-block latency. Overlapping the
        // next block's read with translation belongs to the read-integration
        // follow-up.
        // The sweep counters start at a block boundary; rows before
        // `rows.start` in the first block are translated and discarded.
        let mut translator = SweepTranslator::new(&self.counts, start_block)?;
        // Consecutive blocks reuse the labels loaded for the first of them, so
        // the sweep pays one read per chunk's worth of blocks even when the
        // cache cannot retain anything.
        let mut held = None;
        for block in start_block..=end_block {
            let labels = self
                .block_labels_reusing(block, end_block, &mut held)
                .await?;
            let block_start = self.counts.block_range(block).start;
            let validity = labels.nulls();
            for (i, &value) in labels.values().iter().enumerate() {
                let label = validity
                    .is_none_or(|validity| validity.is_valid(i))
                    .then_some(value);
                let translated = translator.advance(label)?;
                let row = block_start + i as u64;
                if rows.contains(&row) {
                    f(row, translated)?;
                }
            }
        }
        Ok(())
    }

    fn check_row(&self, row: u64) -> Result<()> {
        if row >= self.counts.total_rows() {
            return Err(Error::invalid_input(format!(
                "source row {row} is outside the {} rows of this row map",
                self.counts.total_rows()
            )));
        }
        Ok(())
    }
}
