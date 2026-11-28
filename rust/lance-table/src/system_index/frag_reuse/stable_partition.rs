// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition mapping semantics, independent of dataset lineage traversal.

use super::row_map::{RowMapBlockCache, RowMapReader, open_file};
use crate::format::pb::fragment_reuse_index_details::FragmentDigest;
use async_trait::async_trait;
use futures::future::BoxFuture;
use lance_core::cache::LanceCache;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::fragment_reuse::MappingReader;
use lance_core::utils::stable_partition::CountsMatrix;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use roaring::RoaringBitmap;
use std::sync::Arc;
use std::{collections::HashMap, future::Future};
use tokio::sync::OnceCell;

/// Immutable file name within the mapping's independently owned directory.
pub const MAPPING_FILE: &str = "stable_partition.lance";

/// A mapping reader that opens labels only when addresses need translation.
pub struct StablePartitionMapping {
    open_row_map: Box<dyn Fn() -> BoxFuture<'static, Result<RowMapReader>> + Send + Sync>,
    reader: OnceCell<RowMapReader>,
    sources: HashMap<u32, (u64, u64)>,
    destinations: Vec<FragmentDigest>,
    total_rows: u64,
}

impl std::fmt::Debug for StablePartitionMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StablePartitionMapping")
            .field("sources", &self.sources)
            .field("destinations", &self.destinations)
            .finish_non_exhaustive()
    }
}

impl StablePartitionMapping {
    /// Bind a resolved immutable Lance file to the transition's source scan
    /// order and destination label order. The caller selects the base and
    /// supplies the recorded file size (zero means unknown).
    ///
    /// The file is opened when the first valid, nonempty request arrives.
    /// With `block_cache`, label reads can reuse cached decoded chunks.
    pub fn try_from_file(
        store: Arc<ObjectStore>,
        path: Path,
        file_size: u64,
        metadata_cache: LanceCache,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
        block_cache: Option<RowMapBlockCache>,
    ) -> Result<Self> {
        Self::try_new(sources, destinations, move || {
            let store = store.clone();
            let path = path.clone();
            let metadata_cache = metadata_cache.clone();
            let block_cache = block_cache.clone();
            async move {
                let file_size = CachedFileSize::new(file_size);
                let file = open_file(store, &path, &file_size, &metadata_cache).await?;
                RowMapReader::open_with_cache(Arc::new(file), block_cache).await
            }
        })
    }

    /// Validate transition digests and defer opening the row map until the
    /// first valid, nonempty translation. A failed open can be retried.
    pub fn try_new<Fut>(
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
        open_row_map: impl Fn() -> Fut + Send + Sync + 'static,
    ) -> Result<Self>
    where
        Fut: Future<Output = Result<RowMapReader>> + Send + 'static,
    {
        let mut total_rows = 0_u64;
        let mut offsets = HashMap::with_capacity(sources.len());
        for source in sources {
            if source.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                || source.num_deleted_rows > source.physical_rows
                || source.physical_rows > u32::MAX as u64
                || offsets
                    .insert(source.id as u32, (total_rows, source.physical_rows))
                    .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "invalid or duplicate source fragment {} with {} rows",
                    source.id, source.physical_rows
                )));
            }
            total_rows = total_rows
                .checked_add(source.physical_rows)
                .ok_or_else(|| Error::invalid_input("source physical row count overflow"))?;
        }
        let destination_fragments: RoaringBitmap =
            destinations.iter().map(|f| f.id as u32).collect();
        if destinations.len() > u16::MAX as usize
            || destination_fragments.len() != destinations.len() as u64
            || destinations.iter().any(|f| {
                f.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                    || f.physical_rows > u32::MAX as u64
                    || f.num_deleted_rows != 0
            })
        {
            return Err(Error::invalid_input(
                "invalid stable-partition destination layout",
            ));
        }
        Ok(Self {
            open_row_map: Box::new(move || Box::pin(open_row_map())),
            reader: OnceCell::new(),
            sources: offsets,
            destinations,
            total_rows,
        })
    }

    /// Open (once) the row-map reader, validating its dimensions against the
    /// transition digests. Reuses the cached reader on later calls; the reader
    /// carries the block cache so its label reads are chunked when enabled.
    async fn open_reader(&self) -> Result<&RowMapReader> {
        self.reader
            .get_or_try_init(|| async {
                let reader = (self.open_row_map)().await?;
                let counts = reader.counts();
                if counts.total_rows() != self.total_rows
                    || counts.num_destinations() as usize != self.destinations.len()
                {
                    return Err(corrupt("row-map dimensions differ from transition digests"));
                }
                for (label, destination) in self.destinations.iter().enumerate() {
                    if u64::from(counts.total(label as u16)) != destination.physical_rows {
                        return Err(corrupt(format!(
                            "row-map total differs for destination {}",
                            destination.id
                        )));
                    }
                }
                Ok(reader)
            })
            .await
    }
}

impl DeepSizeOf for StablePartitionMapping {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.sources.capacity() * (std::mem::size_of::<(u32, (u64, u64))>() + 1)
            + self.destinations.capacity() * std::mem::size_of::<FragmentDigest>()
            + self
                .reader
                .get()
                .map_or(0, |reader| reader.counts().deep_size_of())
        // The session accounts for the shared caches.
    }
}

#[async_trait]
impl MappingReader for StablePartitionMapping {
    async fn remap_row_id(&self, row_id: u64) -> Result<Option<u64>> {
        // Reuse the block reader for a single address too.
        let mapped = self.remap_row_ids(&[row_id]).await?;
        mapped
            .into_iter()
            .next()
            .ok_or_else(|| Error::internal("stable-partition mapping omitted the requested row"))
    }

    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        let mut output = vec![None; row_ids.len()];
        let mut requests = Vec::with_capacity(row_ids.len());
        for (position, &row_id) in row_ids.iter().enumerate() {
            let address = RowAddress::from(row_id);
            let &(base, rows) = self.sources.get(&address.fragment_id()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "address {address} is outside stable-partition sources"
                ))
            })?;
            if u64::from(address.row_offset()) >= rows {
                return Err(Error::invalid_input(format!(
                    "address {address} exceeds source length {rows}"
                )));
            }
            requests.push((base + u64::from(address.row_offset()), position));
        }
        if requests.is_empty() {
            return Ok(output);
        }
        let reader = self.open_reader().await?;
        requests.sort_unstable_by_key(|&(row, _)| row);
        let mut remaining = requests.as_slice();
        // Blocks are visited in row order, so consecutive blocks reuse the
        // labels loaded for the first of them: one read per cache chunk, or
        // without a chunk cache one read per run of adjacent blocks the
        // request touches (bounded to a chunk's worth). A sparse request
        // never reads a block it does not touch.
        let mut held = None;
        // The last block of the run of adjacent blocks the request is
        // currently walking. It is discovered once, when a run starts, and
        // reused for every block of the run, so the lookahead over a run of
        // B blocks costs B steps in total rather than B*(B-1)/2.
        let mut run_end: Option<usize> = None;
        while let Some(&(first, _)) = remaining.first() {
            let counts = reader.counts();
            let block = counts.block_of(first);
            let range = counts.block_range(block);
            let end = remaining.partition_point(|&(row, _)| row < range.end);
            let (batch, rest) = remaining.split_at(end);
            let run_last = match run_end {
                Some(last) if block <= last => last,
                _ => *run_end.insert(contiguous_run_end(counts, block, rest)),
            };
            let labels = reader
                .block_labels_reusing(block, run_last, &mut held)
                .await?;
            if labels.len() as u64 != range.end - range.start {
                return Err(corrupt(format!(
                    "row-map block {block} has an unexpected label count"
                )));
            }
            // One sweep per touched block: bounded label memory and no
            // repeated prefix scans for dense index pages or duplicates.
            let mut counters = counts.counters_at_block(block);
            let mut requested = batch.iter().peekable();
            for (offset, label) in labels.iter().enumerate() {
                let translated = if let Some(label) = label {
                    let counter = counters
                        .get_mut(label as usize)
                        .ok_or_else(|| corrupt(format!("invalid row-map label {label}")))?;
                    let destination_offset = *counter;
                    *counter = counter
                        .checked_add(1)
                        .ok_or_else(|| corrupt("row-map count overflow"))?;
                    Some(
                        RowAddress::new_from_parts(
                            self.destinations[label as usize].id as u32,
                            destination_offset,
                        )
                        .into(),
                    )
                } else {
                    None
                };
                let row = range.start + offset as u64;
                while let Some(&&(requested_row, position)) = requested.peek() {
                    if requested_row != row {
                        break;
                    }
                    output[position] = translated;
                    requested.next();
                }
            }
            if counters != counts.counters_at_block(block + 1) {
                return Err(corrupt(format!(
                    "row-map labels disagree with counts in block {block}"
                )));
            }
            remaining = rest;
        }

        Ok(output)
    }
}

/// The last block of the run of adjacent blocks that starts at `block` and
/// continues through `ahead`, the sorted requests after `block`'s own: the
/// run ends at the first block `ahead` skips. Each step advances past one
/// whole block of requests, so a run of B blocks costs B - 1 steps plus at
/// most one step to find the gap after it; the caller asks once per run.
fn contiguous_run_end(counts: &CountsMatrix, block: usize, ahead: &[(u64, usize)]) -> usize {
    let mut run_end = block;
    let mut cursor = 0;
    while cursor < ahead.len() {
        #[cfg(test)]
        tests::LOOKAHEAD_STEPS.with(|steps| steps.set(steps.get() + 1));
        let next = counts.block_of(ahead[cursor].0);
        if next != run_end + 1 {
            break;
        }
        run_end = next;
        let next_range = counts.block_range(next);
        cursor += ahead[cursor..].partition_point(|&(row, _)| row < next_range.end);
    }
    run_end
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named(MAPPING_FILE, message)
}

#[cfg(test)]
mod tests {
    use super::super::row_map::{COUNTS_BUFFER_INDEX_KEY, LABEL_COLUMN, RowMapFile};
    use super::*;
    use arrow_array::{RecordBatch, UInt16Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use bytes::Bytes;
    use lance_core::datatypes::Schema;
    use lance_core::utils::stable_partition::CountsMatrixBuilder;
    use lance_file::reader::FileReader;
    use lance_file::version::ConcreteFileVersion;
    use lance_file::versions::create_writer;
    use lance_file::writer::FileWriterOptions;
    use std::ops::Range;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    thread_local! {
        /// Lookahead steps [`contiguous_run_end`] has taken on this thread
        /// (the tests run on the current-thread runtime).
        pub(super) static LOOKAHEAD_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// The lookahead steps taken since the last call.
    fn take_lookahead_steps() -> usize {
        LOOKAHEAD_STEPS.with(|steps| steps.replace(0))
    }

    #[tokio::test]
    async fn opens_the_bound_lance_file_on_first_translation() {
        let (store, directory) = ObjectStore::from_uri("memory://").await.unwrap();
        let path = directory.join(MAPPING_FILE);
        let metadata_cache = LanceCache::with_capacity(1024 * 1024);
        let mapping = StablePartitionMapping::try_from_file(
            store.clone(),
            path.clone(),
            0,
            metadata_cache.clone(),
            vec![FragmentDigest {
                num_deleted_rows: 1,
                ..digest(1, 5)
            }],
            vec![digest(2, 2), digest(3, 2)],
            None,
        )
        .unwrap();
        // The file does not exist yet.
        assert!(mapping.remap_row_ids(&[]).await.unwrap().is_empty());

        let labels = vec![Some(1), None, Some(0), Some(1), Some(0)];
        let size = write_labels(&store, &path, labels, 2).await;
        let addr = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));
        assert_eq!(
            mapping
                .remap_row_ids(&[addr(1, 4), addr(1, 1), addr(1, 0)])
                .await
                .unwrap(),
            vec![Some(addr(2, 1)), None, Some(addr(3, 0))]
        );

        let file = open_file(store, &path, &CachedFileSize::new(size), &metadata_cache)
            .await
            .unwrap();
        let reader = RowMapReader::open(Arc::new(file)).await.unwrap();
        assert_eq!(
            reader.translate_many(&[4, 1, 0, 4]).await.unwrap(),
            vec![Some((0, 1)), None, Some((1, 0)), Some((0, 1))]
        );
    }

    /// Counts the label reads the row-map reader performs and the rows they
    /// asked for.
    struct CountingReader {
        inner: FileReader,
        reads: AtomicUsize,
        rows: AtomicUsize,
    }

    impl CountingReader {
        fn new(inner: FileReader) -> Arc<Self> {
            Arc::new(Self {
                inner,
                reads: Default::default(),
                rows: Default::default(),
            })
        }

        /// `(reads, rows)` since the last call.
        fn take(&self) -> (usize, usize) {
            (self.reads.swap(0, Relaxed), self.rows.swap(0, Relaxed))
        }
    }

    #[async_trait]
    impl RowMapFile for CountingReader {
        fn schema(&self) -> &Schema {
            self.inner.schema()
        }

        fn num_rows(&self) -> u64 {
            self.inner.num_rows()
        }

        async fn read_global_buffer(&self, index: u32) -> Result<Bytes> {
            self.inner.read_global_buffer(index).await
        }

        async fn read_range(&self, range: Range<usize>) -> Result<RecordBatch> {
            self.reads.fetch_add(1, Relaxed);
            self.rows.fetch_add(range.len(), Relaxed);
            self.inner.read_range(range).await
        }

        async fn read_ranges(&self, ranges: &[Range<usize>]) -> Result<RecordBatch> {
            self.reads.fetch_add(1, Relaxed);
            self.rows
                .fetch_add(ranges.iter().map(|range| range.len()).sum(), Relaxed);
            self.inner.read_ranges(ranges).await
        }
    }

    async fn write_labels(
        store: &ObjectStore,
        path: &Path,
        labels: Vec<Option<u16>>,
        block_rows: u32,
    ) -> u64 {
        let mut counts = CountsMatrixBuilder::try_new(2, block_rows).unwrap();
        for &label in &labels {
            counts.push(label).unwrap();
        }

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            LABEL_COLUMN,
            DataType::UInt16,
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt16Array::from(labels))])
            .unwrap();
        let mut writer = create_writer(
            ConcreteFileVersion::V2_0,
            store.create(path).await.unwrap(),
            Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        let buffer_index = writer
            .add_global_buffer(Bytes::from(counts.finish().encode()))
            .await
            .unwrap();
        writer.add_schema_metadata(COUNTS_BUFFER_INDEX_KEY, buffer_index.to_string());
        writer.finish().await.unwrap().size_bytes
    }

    async fn counted_file(labels: Vec<Option<u16>>, block_rows: u32) -> Arc<CountingReader> {
        let (store, directory) = ObjectStore::from_uri("memory://").await.unwrap();
        let path = directory.join(MAPPING_FILE);
        let size = write_labels(&store, &path, labels, block_rows).await;
        let file = open_file(
            store,
            &path,
            &CachedFileSize::new(size),
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        CountingReader::new(file)
    }

    fn digest(id: u64, rows: u64) -> FragmentDigest {
        FragmentDigest {
            id,
            physical_rows: rows,
            num_deleted_rows: 0,
        }
    }

    /// Without a block cache a batch reads exactly the blocks it touches: a
    /// run of adjacent blocks in one read, and nothing between two blocks
    /// far apart (the request's last block is not a span to read up to).
    #[tokio::test]
    async fn sparse_batch_reads_only_touched_blocks_without_cache() {
        // One source of 800 rows in 100 blocks of 8, two destinations of 400.
        let labels = (0..800).map(|row| Some(row % 2)).collect();
        let counting = counted_file(labels, 8).await;
        let mapping = StablePartitionMapping::try_new(
            vec![digest(1, 800)],
            vec![digest(2, 400), digest(3, 400)],
            {
                let counting = counting.clone();
                move || RowMapReader::open(counting.clone())
            },
        )
        .unwrap();
        let addr = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));

        // Blocks 0 and 99: two block reads of eight rows, not the span.
        assert_eq!(
            mapping
                .remap_row_ids(&[addr(1, 0), addr(1, 799)])
                .await
                .unwrap(),
            vec![Some(addr(2, 0)), Some(addr(3, 399))]
        );
        assert_eq!(counting.take(), (2, 16));
        // An adjacent run (blocks 5, 6, 7) is read once.
        mapping
            .remap_row_ids(&[addr(1, 40), addr(1, 48), addr(1, 56)])
            .await
            .unwrap();
        assert_eq!(counting.take(), (1, 24));
        // A run and then a gap: two reads, only the touched blocks.
        mapping
            .remap_row_ids(&[addr(1, 0), addr(1, 8), addr(1, 80)])
            .await
            .unwrap();
        assert_eq!(counting.take(), (2, 24));
    }

    /// A long run of consecutive blocks (several chunks) discovers its
    /// boundary once: the lookahead is linear in the number of blocks, the
    /// run is read a chunk at a time, and the batch agrees with per-row
    /// translation (duplicates, a deleted row and request order included).
    #[tokio::test]
    async fn long_consecutive_run_lookahead_is_linear() {
        const BLOCKS: usize = 300;
        const BLOCK_ROWS: u64 = 8;
        const ROWS: u64 = BLOCKS as u64 * BLOCK_ROWS;
        const DELETED: u32 = 5;
        // One source of 2400 rows in 300 blocks of 8, one row deleted.
        let mut live_labels = (0..ROWS as u16 - 1).map(|row| row % 2);
        let labels = (0..ROWS as u32)
            .map(|row| {
                if row == DELETED {
                    None
                } else {
                    live_labels.next()
                }
            })
            .collect();
        let counting = counted_file(labels, BLOCK_ROWS as u32).await;
        let mapping = StablePartitionMapping::try_new(
            vec![FragmentDigest {
                num_deleted_rows: 1,
                ..digest(1, ROWS)
            }],
            vec![digest(2, 1200), digest(3, 1199)],
            {
                let counting = counting.clone();
                move || RowMapReader::open(counting.clone())
            },
        )
        .unwrap();
        take_lookahead_steps();
        let addr = |offset| u64::from(RowAddress::new_from_parts(1, offset));

        // One row per block, every block, in reverse order, plus the deleted
        // row and a duplicate.
        let mut request: Vec<u64> = (0..BLOCKS as u32)
            .rev()
            .map(|block| addr(block * BLOCK_ROWS as u32 + block % BLOCK_ROWS as u32))
            .collect();
        request.push(addr(DELETED));
        request.push(addr(7 * BLOCK_ROWS as u32 + 7));
        let batch = mapping.remap_row_ids(&request).await.unwrap();
        // Linear lookahead: one step per block of the run (the old per-block
        // rediscovery took B * (B - 1) / 2 = 44850 steps for B = 300).
        let steps = take_lookahead_steps();
        assert_eq!(steps, BLOCKS - 1);
        assert!(steps <= 2 * BLOCKS);
        // The run is read a chunk (32 blocks) at a time, every block once.
        assert_eq!(counting.take(), (BLOCKS.div_ceil(32), ROWS as usize));

        assert_eq!(batch.len(), request.len());
        assert_eq!(batch[BLOCKS], None, "the deleted row stays deleted");
        assert_eq!(batch[BLOCKS + 1], batch[BLOCKS - 1 - 7], "duplicate");
        for (row_id, translated) in request.iter().zip(&batch) {
            assert_eq!(mapping.remap_row_id(*row_id).await.unwrap(), *translated);
        }
        // Request order is kept: the first entry is block 299's row 2395,
        // live row 2394 (label 0), the 1198th row of destination 2.
        assert_eq!(
            batch[0],
            Some(u64::from(RowAddress::new_from_parts(2, 1197)))
        );
        assert_eq!(
            batch[BLOCKS - 1],
            Some(u64::from(RowAddress::new_from_parts(2, 0)))
        );
    }
}
