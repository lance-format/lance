// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row map file IO for stable-partition (reordered) rewrites.
//!
//! A reordered rewrite persists its per-row destination labels as one Lance
//! file with a single nullable u16 column, one row per physical source row in
//! concatenated scan order:
//!
//! * value — the index of the row's destination fragment in the rewrite's
//!   ordered destination list,
//! * NULL — the row was deleted at the source; the rewrite moved it nowhere.
//!
//! The per-block cumulative counts
//! ([`CountsMatrix`]) are
//! serialized into a global buffer of the same file, so opening costs one tail
//! read plus an in-memory consistency check of the counts; no label IO. Point translations read one block of labels; sweeps
//! stream blocks in order. The translation arithmetic itself lives in
//! [`lance_core::utils::stable_partition`].
//!
//! The writer is fed labels of *live* rows only, in source scan order, and
//! interleaves the NULL rows itself from the source deletion vectors: the
//! rewrite job scans with deletions applied, so it never sees a deleted row.
//!
//! Correctness rests on the stable-partition ordering contract spelled out
//! in [`lance_core::utils::stable_partition`]: labels arrive in source
//! physical-row order, each destination is written in that same order and
//! never re-sorted, and the destination list is fixed for the rewrite. A
//! job that routes rows through parallel writers must restore that order
//! per destination before feeding this writer.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::builder::UInt16Builder;
use arrow_array::{RecordBatch, UInt16Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use bytes::Bytes;
use lance_core::datatypes::Schema;
use lance_core::utils::stable_partition::{CountsMatrix, CountsMatrixBuilder, DEFAULT_BLOCK_ROWS};
use lance_core::{Error, Result};
use lance_table::system_index::frag_reuse::row_map::{self, COUNTS_BUFFER_INDEX_KEY, RowMapFile};
use roaring::RoaringBitmap;

use crate::scalar::{IndexFile, IndexReader, IndexWriter};

pub use row_map::{
    HeldLabels, LABEL_COLUMN, ROW_MAP_CACHE_CHUNK_BLOCKS, ROW_MAP_CACHE_CHUNK_BYTES,
    ROW_MAP_CACHE_PREFIX, RowMapBlockCache, RowMapChunk, RowMapChunkKey,
};

fn label_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![Field::new(
        LABEL_COLUMN,
        DataType::UInt16,
        true,
    )]))
}

/// The physical layout of one source fragment, in scan order.
#[derive(Debug, Clone)]
pub struct SourceRows {
    /// Physical rows of the fragment, including deleted rows.
    pub physical_rows: u64,
    /// Deleted row offsets within the fragment at the rewrite's read version.
    pub deleted: Option<RoaringBitmap>,
}

/// Streaming writer for a row map file.
///
/// Feed the destination labels of live source rows in scan order via
/// [`Self::append_labels`]; deleted rows become NULLs automatically. The
/// caller creates the underlying [`IndexWriter`] (choosing store and file
/// name) via `IndexStore::new_index_file` with [`RowMapWriter::schema`].
pub struct RowMapWriter {
    writer: Box<dyn IndexWriter>,
    counts: CountsMatrixBuilder,
    sources: Vec<SourceRows>,
    src_idx: usize,
    src_offset: u64,
    pending: UInt16Builder,
    pending_rows: u32,
    block_rows: u32,
}

impl RowMapWriter {
    /// The Arrow schema of a row map file, for `IndexStore::new_index_file`.
    pub fn schema() -> Arc<ArrowSchema> {
        label_schema()
    }

    /// Create a writer over `sources` (the rewrite's source fragments in
    /// scan order) targeting `num_destinations` ordered destinations, with
    /// the default 64K-row block size.
    pub fn try_new(
        writer: Box<dyn IndexWriter>,
        sources: Vec<SourceRows>,
        num_destinations: u32,
    ) -> Result<Self> {
        Self::try_new_with_block_rows(writer, sources, num_destinations, DEFAULT_BLOCK_ROWS)
    }

    /// [`Self::try_new`] with an explicit block size, the counts granularity
    /// and label IO unit; tests use small blocks to exercise boundaries.
    pub fn try_new_with_block_rows(
        writer: Box<dyn IndexWriter>,
        sources: Vec<SourceRows>,
        num_destinations: u32,
        block_rows: u32,
    ) -> Result<Self> {
        for (i, source) in sources.iter().enumerate() {
            if source.physical_rows > u64::from(u32::MAX) {
                return Err(Error::invalid_input(format!(
                    "source fragment {i} claims {} physical rows, beyond the row-address offset range",
                    source.physical_rows
                )));
            }
            if let Some(deleted) = &source.deleted
                && deleted
                    .max()
                    .is_some_and(|max| u64::from(max) >= source.physical_rows)
            {
                return Err(Error::invalid_input(format!(
                    "source fragment {i} has a deleted row offset outside its {} physical rows",
                    source.physical_rows
                )));
            }
        }
        Ok(Self {
            writer,
            counts: CountsMatrixBuilder::try_new(num_destinations, block_rows)?,
            sources,
            src_idx: 0,
            src_offset: 0,
            pending: UInt16Builder::with_capacity(block_rows as usize),
            pending_rows: 0,
            block_rows,
        })
    }

    /// Record one physical source row and flush a full block of labels.
    async fn emit(&mut self, label: Option<u16>) -> Result<()> {
        self.counts.push(label)?;
        match label {
            Some(label) => self.pending.append_value(label),
            None => self.pending.append_null(),
        }
        self.pending_rows += 1;
        if self.pending_rows == self.block_rows {
            self.flush().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let labels = self.pending.finish();
        let batch = RecordBatch::try_new(label_schema(), vec![Arc::new(labels)])?;
        self.writer.write_record_batch(batch).await?;
        self.pending_rows = 0;
        Ok(())
    }

    /// Emit NULLs for deleted rows at the physical cursor, stopping at the
    /// next live row (or the end of the sources).
    async fn skip_deleted(&mut self) -> Result<bool> {
        while let Some(source) = self.sources.get(self.src_idx) {
            if self.src_offset >= source.physical_rows {
                self.src_idx += 1;
                self.src_offset = 0;
                continue;
            }
            let deleted = source
                .deleted
                .as_ref()
                .is_some_and(|deleted| deleted.contains(self.src_offset as u32));
            if !deleted {
                return Ok(true);
            }
            self.emit(None).await?;
            self.src_offset += 1;
        }
        Ok(false)
    }

    /// Append the destination labels of the next live source rows, in scan
    /// order.
    pub async fn append_labels(&mut self, labels: &[u16]) -> Result<()> {
        for &label in labels {
            if !self.skip_deleted().await? {
                return Err(Error::invalid_input(
                    "more labels than live rows in the source fragments",
                ));
            }
            self.emit(Some(label)).await?;
            self.src_offset += 1;
        }
        Ok(())
    }

    /// Finish the file: drain trailing deleted rows, write the counts matrix
    /// into a global buffer, and close the writer.
    pub async fn finish(mut self) -> Result<(IndexFile, CountsMatrix)> {
        if self.skip_deleted().await? {
            return Err(Error::invalid_input(
                "fewer labels than live rows in the source fragments",
            ));
        }
        self.flush().await?;
        let counts = self.counts.finish();
        let buffer_index = self
            .writer
            .add_global_buffer(Bytes::from(counts.encode()))
            .await?;
        let file = self
            .writer
            .finish_with_metadata(
                [(
                    COUNTS_BUFFER_INDEX_KEY.to_string(),
                    buffer_index.to_string(),
                )]
                .into(),
            )
            .await?;
        Ok((file, counts))
    }
}

struct IndexRowMapFile(Arc<dyn IndexReader>);

#[async_trait]
impl RowMapFile for IndexRowMapFile {
    fn schema(&self) -> &Schema {
        self.0.schema()
    }

    fn num_rows(&self) -> u64 {
        self.0.num_rows() as u64
    }

    async fn read_global_buffer(&self, index: u32) -> Result<Bytes> {
        self.0.read_global_buffer(index).await
    }

    async fn read_range(&self, range: Range<usize>) -> Result<RecordBatch> {
        self.0.read_range(range, Some(&[LABEL_COLUMN])).await
    }

    async fn read_ranges(&self, ranges: &[Range<usize>]) -> Result<RecordBatch> {
        self.0.read_ranges(ranges, Some(&[LABEL_COLUMN])).await
    }
}

/// Reader over a row map stored in an [`IndexReader`].
/// Decoding and translation are shared with the Lance file reader.
pub struct RowMapReader(pub(super) row_map::RowMapReader);

impl std::fmt::Debug for RowMapReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl RowMapReader {
    /// Read and validate the row map's counts buffer without reading labels.
    pub async fn open(reader: Arc<dyn IndexReader>) -> Result<Self> {
        Self::open_with_cache(reader, None).await
    }

    /// Open a row map, optionally caching decoded label chunks on demand.
    pub async fn open_with_cache(
        reader: Arc<dyn IndexReader>,
        block_cache: Option<RowMapBlockCache>,
    ) -> Result<Self> {
        let reader = Arc::new(IndexRowMapFile(reader));
        row_map::RowMapReader::open_with_cache(reader, block_cache)
            .await
            .map(Self)
    }

    /// The validated block counts and per-destination totals.
    pub fn counts(&self) -> &CountsMatrix {
        self.0.counts()
    }

    /// Decode the labels of a block.
    pub async fn block_labels(&self, block: usize) -> Result<UInt16Array> {
        self.0.block_labels(block).await
    }

    /// Decode a block, holding its chunk or adjacent run for subsequent calls.
    ///
    /// See [the table reader](row_map::RowMapReader::block_labels_reusing) for
    /// the meaning of `last_block` and `held`.
    pub async fn block_labels_reusing(
        &self,
        block: usize,
        last_block: usize,
        held: &mut Option<HeldLabels>,
    ) -> Result<UInt16Array> {
        self.0.block_labels_reusing(block, last_block, held).await
    }

    /// Translate one physical source row into its destination and row offset.
    /// Returns `None` when the row was deleted at the source.
    pub async fn translate(&self, row: u64) -> Result<Option<(u16, u32)>> {
        self.0.translate(row).await
    }

    /// Translate physical source rows in input order, coalescing label reads.
    pub async fn translate_many(&self, rows: &[u64]) -> Result<Vec<Option<(u16, u32)>>> {
        self.0.translate_many(rows).await
    }

    /// Visit a range of source rows in order, reading one label span at a time.
    pub async fn sweep(
        &self,
        rows: Range<u64>,
        f: impl FnMut(u64, Option<(u16, u32)>) -> Result<()>,
    ) -> Result<()> {
        self.0.sweep(rows, f).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{IndexStore, lance_format::LanceIndexStore};
    use futures::FutureExt;
    use lance_core::cache::WeakLanceCache;
    use lance_core::utils::tempfile::TempDir;
    use lance_io::object_store::ObjectStore;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    fn test_store(tempdir: &TempDir) -> Arc<dyn IndexStore> {
        let test_path = tempdir.obj_path();
        let (object_store, test_path) = ObjectStore::from_uri(test_path.as_ref())
            .now_or_never()
            .unwrap()
            .unwrap();
        let cache = Arc::new(lance_core::cache::LanceCache::with_capacity(
            128 * 1024 * 1024,
        ));
        Arc::new(LanceIndexStore::new(object_store, test_path, cache))
    }

    struct TestMap {
        sources: Vec<SourceRows>,
        /// Live-row labels in scan order: what the rewrite job feeds.
        live_labels: Vec<u16>,
        /// Per physical source row: the expected translation.
        expected: Vec<Option<(u16, u32)>>,
    }

    /// Build sources with deterministic deletions and labels, plus the
    /// reference translation computed with per-destination counters.
    fn make_test_map(source_rows: &[u64], num_destinations: u32, seed: u64) -> TestMap {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sources = Vec::new();
        let mut live_labels = Vec::new();
        let mut expected = Vec::new();
        let mut counters = HashMap::new();
        for &physical_rows in source_rows {
            let mut deleted = RoaringBitmap::new();
            for offset in 0..physical_rows {
                // ~1/5 of rows deleted.
                if rng.random_ratio(1, 5) {
                    deleted.insert(offset as u32);
                }
            }
            for offset in 0..physical_rows {
                if deleted.contains(offset as u32) {
                    expected.push(None);
                } else {
                    let label = rng.random_range(0..num_destinations) as u16;
                    let counter = counters.entry(label).or_insert(0u32);
                    live_labels.push(label);
                    expected.push(Some((label, *counter)));
                    *counter += 1;
                }
            }
            sources.push(SourceRows {
                physical_rows,
                deleted: (!deleted.is_empty()).then_some(deleted),
            });
        }
        TestMap {
            sources,
            live_labels,
            expected,
        }
    }

    async fn write_map(
        store: &Arc<dyn IndexStore>,
        map: &TestMap,
        num_destinations: u32,
        block_rows: u32,
        labels_per_append: usize,
    ) -> (IndexFile, CountsMatrix) {
        let writer = store
            .new_index_file("row_map.lance", RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            map.sources.clone(),
            num_destinations,
            block_rows,
        )
        .unwrap();
        for chunk in map.live_labels.chunks(labels_per_append) {
            writer.append_labels(chunk).await.unwrap();
        }
        writer.finish().await.unwrap()
    }

    #[tokio::test]
    async fn test_row_map_round_trip() {
        let num_destinations = 4u32;
        let map = make_test_map(&[10, 7, 5], num_destinations, 7);
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        // block_rows=8 forces multiple blocks with a short final block, and
        // appends of 3 labels cross block boundaries mid-call.
        let (file, written_counts) = write_map(&store, &map, num_destinations, 8, 3).await;
        assert!(file.size_bytes > 0);
        written_counts.validate().unwrap();

        let reader = RowMapReader::open(store.open_index_file("row_map.lance").await.unwrap())
            .await
            .unwrap();
        let counts = reader.counts();
        assert_eq!(counts, &written_counts);
        assert_eq!(counts.total_rows(), 22);
        assert_eq!(
            counts.total_live_rows(),
            map.live_labels.len() as u64,
            "every live row is labeled, every deleted row is not"
        );
        // Conservation: per-destination totals match the reference counters.
        for label in 0..num_destinations as u16 {
            let expected_total = map
                .expected
                .iter()
                .flatten()
                .filter(|(l, _)| *l == label)
                .count() as u32;
            assert_eq!(counts.total(label), expected_total);
        }

        // Point lookups: every physical source row.
        for (row, &expected) in map.expected.iter().enumerate() {
            assert_eq!(
                reader.translate(row as u64).await.unwrap(),
                expected,
                "row {row}"
            );
        }
        assert!(reader.translate(22).await.is_err());

        // Batch lookup in arbitrary order with one coalesced read.
        let rows: Vec<u64> = vec![21, 0, 13, 13, 7, 20];
        let translated = reader.translate_many(&rows).await.unwrap();
        for (&row, translated) in rows.iter().zip(&translated) {
            assert_eq!(translated, &map.expected[row as usize], "row {row}");
        }
        assert!(reader.translate_many(&[3, 22]).await.is_err());

        // Sweeps: the full range and one starting mid-block.
        for start in [0u64, 11] {
            let mut swept = Vec::new();
            reader
                .sweep(start..22, |row, translated| {
                    swept.push((row, translated));
                    Ok(())
                })
                .await
                .unwrap();
            let expected: Vec<_> = (start..22)
                .map(|row| (row, map.expected[row as usize]))
                .collect();
            assert_eq!(swept, expected, "sweep from {start}");
        }
    }

    #[tokio::test]
    async fn test_row_map_larger_multi_block() {
        // 1000 rows in blocks of 64 across unevenly-sized sources.
        let num_destinations = 5u32;
        let map = make_test_map(&[400, 1, 599], num_destinations, 42);
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let (_, _) = write_map(&store, &map, num_destinations, 64, 100).await;
        let reader = RowMapReader::open(store.open_index_file("row_map.lance").await.unwrap())
            .await
            .unwrap();
        let rows: Vec<u64> = (0..1000).collect();
        assert_eq!(
            reader.translate_many(&rows).await.unwrap(),
            map.expected,
            "batch translation of every row"
        );
        let mut swept = Vec::new();
        reader
            .sweep(0..1000, |_, translated| {
                swept.push(translated);
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(swept, map.expected);
    }

    #[tokio::test]
    async fn test_block_cache_matches_uncached() {
        // Enough rows that a chunk (ROW_MAP_CACHE_CHUNK_BLOCKS blocks) is not the
        // whole file: with block_rows=8 and 32 blocks/chunk a chunk is 256 rows,
        // so ~5000 rows spans multiple chunks with a short final chunk.
        let num_destinations = 6u32;
        let map = make_test_map(&[2000, 1, 2999], num_destinations, 99);
        let total = map.expected.len() as u64;
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let (_, _) = write_map(&store, &map, num_destinations, 8, 137).await;

        let cache = lance_core::cache::LanceCache::with_capacity(64 * 1024 * 1024);
        let block_cache = RowMapBlockCache::new(WeakLanceCache::from(&cache), [7u8; 32]);
        let reader = RowMapReader::open_with_cache(
            store.open_index_file("row_map.lance").await.unwrap(),
            Some(block_cache),
        )
        .await
        .unwrap();

        // Every physical row translates identically to the reference.
        let rows: Vec<u64> = (0..total).collect();
        assert_eq!(
            reader.translate_many(&rows).await.unwrap(),
            map.expected,
            "cached batch translation of every row"
        );
        // Point lookups (repeat some rows to exercise cache hits).
        for row in [0u64, 255, 256, 257, total - 1, 256, 0] {
            assert_eq!(
                reader.translate(row).await.unwrap(),
                map.expected[row as usize],
                "cached point lookup row {row}"
            );
        }
        // A full sweep goes through the same chunked block reads.
        let mut swept = Vec::new();
        reader
            .sweep(0..total, |_, translated| {
                swept.push(translated);
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(swept, map.expected, "cached sweep");

        // Cache-off reader over the same file must produce identical output.
        let uncached = RowMapReader::open(store.open_index_file("row_map.lance").await.unwrap())
            .await
            .unwrap();
        assert_eq!(
            uncached.translate_many(&rows).await.unwrap(),
            reader.translate_many(&rows).await.unwrap(),
            "cache-off path is byte-identical to cache-on translations"
        );
    }

    /// Counts the label reads an [`IndexReader`] serves, and how many rows
    /// they asked for, so a test can pin the number of file reads a request
    /// performs.
    struct CountingReader {
        inner: Arc<dyn IndexReader>,
        reads: std::sync::atomic::AtomicUsize,
        rows: std::sync::atomic::AtomicUsize,
    }

    impl CountingReader {
        fn new(inner: Arc<dyn IndexReader>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                reads: Default::default(),
                rows: Default::default(),
            })
        }

        /// `(reads, rows)` since the last call.
        fn take(&self) -> (usize, usize) {
            use std::sync::atomic::Ordering::Relaxed;
            (self.reads.swap(0, Relaxed), self.rows.swap(0, Relaxed))
        }
    }

    #[async_trait::async_trait]
    impl IndexReader for CountingReader {
        async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
            self.inner.read_record_batch(n, batch_size).await
        }

        async fn read_global_buffer(&self, index: u32) -> Result<Bytes> {
            self.inner.read_global_buffer(index).await
        }

        async fn read_range(
            &self,
            range: Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            use std::sync::atomic::Ordering::Relaxed;
            self.reads.fetch_add(1, Relaxed);
            self.rows.fetch_add(range.len(), Relaxed);
            self.inner.read_range(range, projection).await
        }

        async fn read_ranges(
            &self,
            ranges: &[Range<usize>],
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            use std::sync::atomic::Ordering::Relaxed;
            self.reads.fetch_add(1, Relaxed);
            self.rows
                .fetch_add(ranges.iter().map(Range::len).sum(), Relaxed);
            self.inner.read_ranges(ranges, projection).await
        }

        async fn num_batches(&self, batch_size: u64) -> u32 {
            self.inner.num_batches(batch_size).await
        }

        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }

        fn schema(&self) -> &lance_core::datatypes::Schema {
            self.inner.schema()
        }
    }

    #[tokio::test]
    async fn test_request_reads_each_chunk_once_without_cache_retention() {
        // Three full chunks of 32 blocks (block_rows=8 -> 256 rows per chunk)
        // and a short fourth chunk.
        let num_destinations = 3u32;
        let map = make_test_map(&[500, 300], num_destinations, 17);
        let total = map.expected.len() as u64;
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let (_, _) = write_map(&store, &map, num_destinations, 8, 101).await;
        let counting = CountingReader::new(store.open_index_file("row_map.lance").await.unwrap());
        let num_blocks = 100usize;
        let num_chunks = num_blocks.div_ceil(ROW_MAP_CACHE_CHUNK_BLOCKS);

        // A zero-capacity cache runs every loader and retains nothing: the
        // block cache is attached, but no chunk survives past its request.
        let cache = lance_core::cache::LanceCache::with_capacity(0);
        let block_cache = RowMapBlockCache::new(WeakLanceCache::from(&cache), [3u8; 32]);
        let reader = RowMapReader::open_with_cache(counting.clone(), Some(block_cache))
            .await
            .unwrap();
        assert_eq!(reader.counts().num_blocks(), num_blocks);
        counting.take();

        // Block by block with held labels: one whole-chunk read per chunk.
        let mut held = None;
        let mut labels = Vec::new();
        for block in 0..num_blocks {
            labels.push(
                reader
                    .block_labels_reusing(block, num_blocks - 1, &mut held)
                    .await
                    .unwrap(),
            );
        }
        let (reads, rows) = counting.take();
        assert_eq!(reads, num_chunks, "one read per chunk with a held chunk");
        assert_eq!(rows as u64, total, "each row read exactly once");

        // A sweep visits blocks in order and must do the same.
        let mut swept = Vec::new();
        reader
            .sweep(0..total, |_, translated| {
                swept.push(translated);
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(swept, map.expected);
        let (reads, rows) = counting.take();
        assert_eq!(reads, num_chunks, "a sweep reads each chunk once");
        assert_eq!(rows as u64, total);

        // Independent block requests cannot reuse anything: without retention
        // every one of them pays a whole-chunk read. This is the amplification
        // the held chunk avoids for a request, and the reason the reader is
        // opened without a block cache when the cache is known to be smaller
        // than a chunk.
        for (block, expected) in labels.iter().enumerate() {
            assert_eq!(
                &reader.block_labels(block).await.unwrap(),
                expected,
                "block {block}"
            );
        }
        let (reads, rows) = counting.take();
        assert_eq!(reads, num_blocks);
        assert!(rows as u64 > total * (ROW_MAP_CACHE_CHUNK_BLOCKS as u64 / 2));

        // Without a block cache each block request reads exactly its rows.
        let plain = RowMapReader::open(counting.clone()).await.unwrap();
        counting.take();
        for (block, expected) in labels.iter().enumerate() {
            assert_eq!(&plain.block_labels(block).await.unwrap(), expected);
        }
        assert_eq!(counting.take(), (num_blocks, total as usize));

        // Without a block cache a run of adjacent blocks is read in one
        // read, bounded to a chunk's worth of blocks, and a sparse request
        // reads exactly its blocks: two blocks far apart cost two block
        // reads, not the span between them.
        let mut held = None;
        assert_eq!(
            &plain.block_labels_reusing(0, 0, &mut held).await.unwrap(),
            &labels[0]
        );
        assert_eq!(
            &plain.block_labels_reusing(99, 99, &mut held).await.unwrap(),
            &labels[99]
        );
        assert_eq!(
            counting.take(),
            (2, 16),
            "two touched blocks, two block reads"
        );
        let mut held = None;
        for (block, expected) in labels.iter().enumerate().take(21).skip(5) {
            assert_eq!(
                &plain
                    .block_labels_reusing(block, 20, &mut held)
                    .await
                    .unwrap(),
                expected
            );
        }
        assert_eq!(counting.take(), (1, 16 * 8), "blocks 5..=20 in one read");
        let mut held = None;
        for (block, expected) in labels.iter().enumerate() {
            assert_eq!(
                &plain
                    .block_labels_reusing(block, num_blocks - 1, &mut held)
                    .await
                    .unwrap(),
                expected
            );
        }
        assert_eq!(
            counting.take(),
            (num_chunks, total as usize),
            "a whole-map request reads a chunk's worth of blocks at a time"
        );
        let mut swept = Vec::new();
        plain
            .sweep(0..total, |_, translated| {
                swept.push(translated);
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(swept, map.expected);
        assert_eq!(counting.take(), (num_chunks, total as usize));
    }

    #[tokio::test]
    async fn test_writer_label_count_mismatches() {
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let sources = vec![SourceRows {
            physical_rows: 4,
            deleted: Some(RoaringBitmap::from_iter([1u32])),
        }];

        // Too many labels: 4 physical - 1 deleted = 3 live rows.
        let writer = store
            .new_index_file("too_many.lance", RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer =
            RowMapWriter::try_new_with_block_rows(writer, sources.clone(), 2, 8).unwrap();
        assert!(writer.append_labels(&[0, 1, 0, 1]).await.is_err());

        // Too few labels are caught at finish.
        let writer = store
            .new_index_file("too_few.lance", RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer =
            RowMapWriter::try_new_with_block_rows(writer, sources.clone(), 2, 8).unwrap();
        writer.append_labels(&[0, 1]).await.unwrap();
        assert!(writer.finish().await.is_err());

        // A deleted offset outside the fragment's physical rows is rejected
        // up front.
        let writer = store
            .new_index_file("bad_dv.lance", RowMapWriter::schema())
            .await
            .unwrap();
        assert!(
            RowMapWriter::try_new_with_block_rows(
                writer,
                vec![SourceRows {
                    physical_rows: 4,
                    deleted: Some(RoaringBitmap::from_iter([4u32])),
                }],
                2,
                8,
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn test_open_rejects_wrong_schema() {
        // Opening a file that is not a row map fails with an error instead of
        // panicking on the column cast.
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "foo",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let mut writer = store
            .new_index_file("not_a_row_map.lance", schema)
            .await
            .unwrap();
        writer.finish().await.unwrap();
        let reader = store.open_index_file("not_a_row_map.lance").await.unwrap();
        assert!(RowMapReader::open(reader).await.is_err());
    }

    #[tokio::test]
    async fn test_null_edge_cases() {
        // Deterministic NULL/empty shapes: a fully-deleted source, a
        // zero-physical-row source, and a deleted tail that only finish()
        // drains. 7 physical rows, 2 live.
        let sources = vec![
            SourceRows {
                physical_rows: 4,
                deleted: Some(RoaringBitmap::from_iter(0u32..4)),
            },
            SourceRows {
                physical_rows: 0,
                deleted: None,
            },
            SourceRows {
                physical_rows: 3,
                deleted: Some(RoaringBitmap::from_iter([2u32])),
            },
        ];
        let tempdir = TempDir::default();
        let store = test_store(&tempdir);
        let writer = store
            .new_index_file("row_map.lance", RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(writer, sources, 2, 4).unwrap();
        writer.append_labels(&[1, 0]).await.unwrap();
        let (_, counts) = writer.finish().await.unwrap();
        assert_eq!(counts.total_rows(), 7);
        assert_eq!((counts.total(0), counts.total(1)), (1, 1));

        let reader = RowMapReader::open(store.open_index_file("row_map.lance").await.unwrap())
            .await
            .unwrap();
        let expected = [
            None,
            None,
            None,
            None,            // source 1: fully deleted
            Some((1u16, 0)), // source 3 row 0
            Some((0u16, 0)), // source 3 row 1
            None,            // source 3 row 2: the deleted tail
        ];
        for (row, &expected) in expected.iter().enumerate() {
            assert_eq!(reader.translate(row as u64).await.unwrap(), expected);
        }
        // Empty inputs are empty outputs.
        assert!(reader.translate_many(&[]).await.unwrap().is_empty());
        reader
            .sweep(3..3, |_, _| panic!("empty sweep must visit nothing"))
            .await
            .unwrap();
    }
}
