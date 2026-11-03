// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition mapping semantics, independent of dataset lineage traversal.

use super::row_map::{RowMapBlockCache, RowMapReader};
use crate::scalar::IndexStore;
use async_trait::async_trait;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::fragment_reuse::MappingReader;
use lance_core::utils::stable_partition::CountsMatrix;
use lance_core::{Error, Result};
use lance_table::format::pb::fragment_reuse_index_details::FragmentDigest;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Immutable file name within the mapping's independently owned directory.
pub const MAPPING_FILE: &str = "stable_partition.lance";

/// A mapping reader that opens labels only when addresses need translation.
pub struct StablePartitionMapping {
    store: Arc<dyn IndexStore>,
    reader: OnceCell<RowMapReader>,
    sources: HashMap<u32, (u64, u64)>,
    destinations: Vec<FragmentDigest>,
    total_rows: u64,
    /// When present, the row-map reader routes its label reads through the
    /// shared index cache in ~4 MiB chunks. `None` reads labels directly.
    block_cache: Option<RowMapBlockCache>,
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
    /// Bind the file store to source scan order and destination label order.
    /// The store resolves the dataset base; this reader does not interpret manifests.
    ///
    /// `block_cache`, when supplied, routes the row map's label reads through the
    /// shared index cache in ~4 MiB chunks. Pass `None` (see [`try_new`]) to read
    /// labels directly, which is byte-identical to the pre-cache behavior.
    ///
    /// [`try_new`]: Self::try_new
    pub fn try_new_with_cache(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
        block_cache: Option<RowMapBlockCache>,
    ) -> Result<Self> {
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
            store,
            reader: OnceCell::new(),
            sources: offsets,
            destinations,
            total_rows,
            block_cache,
        })
    }

    /// Convenience constructor with no block cache: reads labels directly.
    pub fn try_new(
        store: Arc<dyn IndexStore>,
        sources: Vec<FragmentDigest>,
        destinations: Vec<FragmentDigest>,
    ) -> Result<Self> {
        Self::try_new_with_cache(store, sources, destinations, None)
    }

    /// Open (once) the row-map reader, validating its dimensions against the
    /// transition digests. Reuses the cached reader on later calls; the reader
    /// carries the block cache so its label reads are chunked when enabled.
    async fn open_reader(&self) -> Result<&RowMapReader> {
        self.reader
            .get_or_try_init(|| async {
                let reader = RowMapReader::open_with_cache(
                    self.store.open_index_file(MAPPING_FILE).await?,
                    self.block_cache.clone(),
                )
                .await?;
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
        // The IndexStore and file metadata belong to the session's metadata cache.
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
    use super::*;
    use crate::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow_array::RecordBatch;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempDir;
    use lance_index_core::scalar::IndexReader;
    use lance_io::object_store::ObjectStore;

    thread_local! {
        /// Lookahead steps [`contiguous_run_end`] has taken on this thread
        /// (the tests run on the current-thread runtime).
        pub(super) static LOOKAHEAD_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// The lookahead steps taken since the last call.
    fn take_lookahead_steps() -> usize {
        LOOKAHEAD_STEPS.with(|steps| steps.replace(0))
    }

    /// Counts the label reads the row-map reader performs and the rows they
    /// asked for.
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

        async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
            self.inner.read_global_buffer(index).await
        }

        async fn read_range(
            &self,
            range: std::ops::Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            use std::sync::atomic::Ordering::Relaxed;
            self.reads.fetch_add(1, Relaxed);
            self.rows.fetch_add(range.len(), Relaxed);
            self.inner.read_range(range, projection).await
        }

        async fn read_ranges(
            &self,
            ranges: &[std::ops::Range<usize>],
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            use std::sync::atomic::Ordering::Relaxed;
            self.reads.fetch_add(1, Relaxed);
            self.rows
                .fetch_add(ranges.iter().map(|range| range.len()).sum(), Relaxed);
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

    /// Without a block cache a batch reads exactly the blocks it touches: a
    /// run of adjacent blocks in one read, and nothing between two blocks
    /// far apart (the request's last block is not a span to read up to).
    #[tokio::test]
    async fn sparse_batch_reads_only_touched_blocks_without_cache() {
        let directory = TempDir::default();
        let (object_store, path) = ObjectStore::from_uri(directory.obj_path().as_ref())
            .await
            .unwrap();
        let store: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::no_cache()),
        ));
        // One source of 800 rows in 100 blocks of 8, two destinations of 400.
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: 800,
                deleted: None,
            }],
            2,
            8,
        )
        .unwrap();
        let labels: Vec<u16> = (0..800u16).map(|i| i % 2).collect();
        writer.append_labels(&labels).await.unwrap();
        writer.finish().await.unwrap();
        let counting = CountingReader::new(store.open_index_file(MAPPING_FILE).await.unwrap());
        let digest = |id, rows| FragmentDigest {
            id,
            physical_rows: rows,
            num_deleted_rows: 0,
        };
        let mapping = StablePartitionMapping::try_new(
            store.clone(),
            vec![digest(1, 800)],
            vec![digest(2, 400), digest(3, 400)],
        )
        .unwrap();
        assert!(
            mapping
                .reader
                .set(RowMapReader::open(counting.clone()).await.unwrap())
                .is_ok()
        );
        counting.take();
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
        let directory = TempDir::default();
        let (object_store, path) = ObjectStore::from_uri(directory.obj_path().as_ref())
            .await
            .unwrap();
        let store: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::no_cache()),
        ));
        // One source of 2400 rows in 300 blocks of 8 (nine full chunks of 32
        // blocks and a partial tenth), one row deleted, two destinations.
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: ROWS,
                deleted: Some([DELETED].into_iter().collect()),
            }],
            2,
            BLOCK_ROWS as u32,
        )
        .unwrap();
        let labels: Vec<u16> = (0..ROWS as u16 - 1).map(|i| i % 2).collect();
        writer.append_labels(&labels).await.unwrap();
        writer.finish().await.unwrap();
        let counting = CountingReader::new(store.open_index_file(MAPPING_FILE).await.unwrap());
        let digest = |id, rows| FragmentDigest {
            id,
            physical_rows: rows,
            num_deleted_rows: 0,
        };
        let mapping = StablePartitionMapping::try_new(
            store.clone(),
            vec![digest(1, ROWS)],
            vec![digest(2, 1200), digest(3, 1199)],
        )
        .unwrap();
        assert!(
            mapping
                .reader
                .set(RowMapReader::open(counting.clone()).await.unwrap())
                .is_ok()
        );
        counting.take();
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

    #[tokio::test]
    async fn mapping_single_and_batch_translation_are_lazy() {
        let directory = TempDir::default();
        let (object_store, path) = ObjectStore::from_uri(directory.obj_path().as_ref())
            .await
            .unwrap();
        let store = Arc::new(LanceIndexStore::new(
            object_store,
            path,
            Arc::new(LanceCache::with_capacity(1024 * 1024)),
        ));
        let source = FragmentDigest {
            id: 1,
            physical_rows: 5,
            num_deleted_rows: 1,
        };
        let destinations = vec![
            FragmentDigest {
                id: 2,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
            FragmentDigest {
                id: 3,
                physical_rows: 2,
                num_deleted_rows: 0,
            },
        ];
        let mapping =
            StablePartitionMapping::try_new(store.clone(), vec![source], destinations).unwrap();
        assert!(mapping.remap_row_ids(&[]).await.unwrap().is_empty());
        assert!(mapping.reader.get().is_none());
        // Empty input must not open the file.
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(
            writer,
            vec![SourceRows {
                physical_rows: 5,
                deleted: Some([1].into_iter().collect()),
            }],
            2,
            2,
        )
        .unwrap();
        writer.append_labels(&[1, 0, 1, 0]).await.unwrap();
        writer.finish().await.unwrap();
        let addr = |fragment, offset| u64::from(RowAddress::new_from_parts(fragment, offset));
        assert_eq!(
            mapping
                .remap_row_ids(&[addr(1, 4), addr(1, 1), addr(1, 0), addr(1, 0), addr(1, 2)])
                .await
                .unwrap(),
            vec![
                Some(addr(2, 1)),
                None,
                Some(addr(3, 0)),
                Some(addr(3, 0)),
                Some(addr(2, 0))
            ]
        );
        assert!(mapping.reader.get().is_some());
        assert_eq!(
            mapping.remap_row_id(addr(1, 4)).await.unwrap(),
            Some(addr(2, 1))
        );
        assert_eq!(mapping.remap_row_id(addr(1, 1)).await.unwrap(), None);
        for (row_id, message) in [
            (addr(9, 0), "outside stable-partition sources"),
            (addr(1, 5), "exceeds source length"),
        ] {
            for error in [
                mapping.remap_row_id(row_id).await.unwrap_err(),
                mapping.remap_row_ids(&[row_id]).await.unwrap_err(),
            ] {
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains(message));
            }
        }
    }
}
