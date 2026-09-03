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
use arrow_array::cast::AsArray;
use arrow_array::types::UInt16Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance_core::utils::stable_partition::{
    CountsMatrix, CountsMatrixBuilder, DEFAULT_BLOCK_ROWS, SweepTranslator, translate_in_block,
};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use crate::scalar::{IndexFile, IndexReader, IndexWriter};

/// The single column of a row map file.
pub const LABEL_COLUMN: &str = "label";

/// Schema-metadata key holding the global buffer index of the encoded counts
/// matrix.
const COUNTS_BUFFER_INDEX_KEY: &str = "lance:stable_partition:counts_buffer_index";

fn label_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![Field::new(
        LABEL_COLUMN,
        DataType::UInt16,
        true,
    )]))
}

fn label_column(batch: &RecordBatch) -> Result<&UInt16Array> {
    batch
        .columns()
        .first()
        .and_then(|column| column.as_primitive_opt::<UInt16Type>())
        .ok_or_else(|| {
            Error::corrupt_file_named(
                "row_map",
                "row map read returned a batch without a u16 label column",
            )
        })
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

/// Reader over a row map file: one tail read at open, block reads on demand.
pub struct RowMapReader {
    reader: Arc<dyn IndexReader>,
    counts: CountsMatrix,
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
    pub async fn open(reader: Arc<dyn IndexReader>) -> Result<Self> {
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
            || label_field.data_type() != arrow_schema::DataType::UInt16
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
        if reader.num_rows() as u64 != counts.total_rows() {
            return Err(Error::corrupt_file_named(
                "row_map",
                format!(
                    "row map file holds {} labels but its counts describe {} source rows",
                    reader.num_rows(),
                    counts.total_rows()
                ),
            ));
        }
        Ok(Self { reader, counts })
    }

    /// The validated counts matrix decoded at open: per-destination totals
    /// and per-block counter bases without any label IO.
    pub fn counts(&self) -> &CountsMatrix {
        &self.counts
    }

    /// The decoded labels of one block.
    pub async fn block_labels(&self, block: usize) -> Result<UInt16Array> {
        let range = self.counts.block_range(block);
        let batch = self
            .reader
            .read_range(
                range.start as usize..range.end as usize,
                Some(&[LABEL_COLUMN]),
            )
            .await?;
        Ok(label_column(&batch)?.clone())
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
        let batch = self
            .reader
            .read_ranges(&ranges, Some(&[LABEL_COLUMN]))
            .await?;
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
                let slot = blocks.binary_search(&block).unwrap();
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
        for block in start_block..=end_block {
            let labels = self.block_labels(block).await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::{IndexStore, lance_format::LanceIndexStore};
    use futures::FutureExt;
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
