// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::{FutureExt, future::BoxFuture};
use lance_core::utils::tokio::spawn;
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;

use super::reader::DEFAULT_READ_CHUNK_SIZE;

/// Bytes each task copies when a range read in chunks is put back together:
/// small enough that a task holds its runtime thread for well under a
/// millisecond.
const REASSEMBLY_COPY_BYTES: usize = 1024 * 1024;

/// Copies a range's chunks into one buffer, in pieces on the current runtime.
///
/// The copy runs when the decoder awaits the read, and the decoder awaits its
/// pages one by one, so on a single core it bounded scans of wide rows (2 GB/s
/// for 1,024-dim f32 vectors in memory, mostly faulting in the new buffer).
/// It stays in the awaited future rather than in a task of its own because
/// the read's backpressure reservation is refunded when its chunks are taken:
/// copying ahead of the decoder would let scans read without bound. The
/// pieces skip the CPU pool, where they would queue behind the batches an
/// index build is computing on while that computation waits for this data.
async fn concat_chunks(chunks: Vec<Bytes>) -> lance_core::Result<Bytes> {
    let total_size = chunks.iter().map(Bytes::len).sum();
    if tokio::runtime::Handle::try_current().is_err() {
        let mut combined = Vec::with_capacity(total_size);
        for chunk in &chunks {
            combined.extend_from_slice(chunk);
        }
        return Ok(Bytes::from(combined));
    }
    let mut unfilled = BytesMut::with_capacity(total_size);
    let mut copies = Vec::with_capacity(total_size.div_ceil(REASSEMBLY_COPY_BYTES) + chunks.len());
    for chunk in &chunks {
        for start in (0..chunk.len()).step_by(REASSEMBLY_COPY_BYTES) {
            let src = chunk.slice(start..chunk.len().min(start + REASSEMBLY_COPY_BYTES));
            // Each piece owns its own stretch of the buffer's capacity, so the
            // pieces fill in parallel and rejoin without copying.
            let rest = unfilled.split_off(src.len());
            let mut piece = std::mem::replace(&mut unfilled, rest);
            copies.push(spawn(async move {
                piece.extend_from_slice(&src);
                piece
            }));
        }
    }
    let mut combined = BytesMut::new();
    for piece in futures::future::try_join_all(copies).await? {
        combined.unsplit(piece);
    }
    Ok(combined.freeze())
}

#[derive(Debug)]
pub struct LanceEncodingsIo {
    scheduler: FileScheduler,
    /// Size of chunks when reading large pages
    read_chunk_size: u64,
}

impl LanceEncodingsIo {
    pub fn new(scheduler: FileScheduler) -> Self {
        Self {
            scheduler,
            read_chunk_size: DEFAULT_READ_CHUNK_SIZE,
        }
    }

    pub fn with_read_chunk_size(mut self, read_chunk_size: u64) -> Self {
        self.read_chunk_size = read_chunk_size;
        self
    }
}

impl EncodingsIo for LanceEncodingsIo {
    fn io_buffer_size(&self) -> Option<u64> {
        Some(self.scheduler.io_buffer_size())
    }

    fn coalesce_gap(&self) -> u64 {
        self.scheduler.block_size()
    }

    fn with_bypass_backpressure(&self) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            scheduler: self.scheduler.with_bypass_backpressure(),
            read_chunk_size: self.read_chunk_size,
        }))
    }

    fn with_io_stats(
        &self,
        stats: Arc<dyn lance_core::utils::io_stats::IoStatsRecorder>,
    ) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            scheduler: self.scheduler.with_io_stats(stats),
            read_chunk_size: self.read_chunk_size,
        }))
    }

    fn with_coalesce_gap(&self, gap: u64) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            scheduler: self.scheduler.with_block_size(gap),
            read_chunk_size: self.read_chunk_size,
        }))
    }

    fn submit_request(
        &self,
        ranges: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        if let Some(invalid) = ranges.iter().find(|range| range.start > range.end) {
            return futures::future::ready(Err(lance_core::Error::invalid_input(format!(
                "Invalid read range {}..{}: start is after end",
                invalid.start, invalid.end
            ))))
            .boxed();
        }
        let mut split_ranges = Vec::new();
        let mut split_indices = Vec::new(); // Track which original range each split came from
        // Large ranges (above read_chunk_size) will be split into
        // multiple reads.  Empty ranges will skip the I/O layer
        // entirely.  If we have either of these we will need to
        // reassemble our results, inserting empties and merging parts
        let mut needs_reassembly = false;

        // Split large ranges into smaller chunks
        //
        // TODO: consider read_chunk_size before submitting requests.
        for (idx, range) in ranges.iter().enumerate() {
            if range.start == range.end {
                // EncodingsIo requires one result per input range. Zero-length
                // ranges schedule no I/O, so their empty results are restored
                // after the non-empty requests complete.
                needs_reassembly = true;
                continue;
            }
            let range_size = range.end - range.start;

            if range_size > self.read_chunk_size {
                needs_reassembly = true;
                let num_chunks = range_size.div_ceil(self.read_chunk_size);
                let chunk_size = range_size / num_chunks;

                for i in 0..num_chunks {
                    let start = range.start + i * chunk_size;
                    let end = if i == num_chunks - 1 {
                        range.end // Last chunk gets any remaining bytes
                    } else {
                        start + chunk_size
                    };
                    split_ranges.push(start..end);
                    split_indices.push(idx);
                }
            } else {
                split_ranges.push(range.clone());
                split_indices.push(idx);
            }
        }

        let fut = self.scheduler.submit_request(split_ranges, priority);

        async move {
            let split_results = fut.await?;

            if split_results.len() != split_indices.len() {
                return Err(lance_core::Error::internal(format!(
                    "Encoding I/O returned {} results for {} requested range chunks",
                    split_results.len(),
                    split_indices.len()
                )));
            }
            if !needs_reassembly {
                return Ok(split_results);
            }

            let mut results = vec![Vec::new(); ranges.len()];

            for (split_result, orig_idx) in split_results.into_iter().zip(split_indices) {
                results[orig_idx].push(split_result);
            }

            let mut reassembled = Vec::with_capacity(ranges.len());
            for (range, chunks) in ranges.iter().zip(results) {
                if chunks.is_empty() {
                    if range.start == range.end {
                        reassembled.push(bytes::Bytes::new());
                        continue;
                    }
                    return Err(lance_core::Error::internal(format!(
                        "Encoding I/O returned no data for non-empty range {}..{}",
                        range.start, range.end
                    )));
                }
                if chunks.len() == 1 {
                    reassembled.push(chunks[0].clone());
                    continue;
                }

                reassembled.push(concat_chunks(chunks).await?);
            }
            Ok(reassembled)
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use lance_io::utils::CachedFileSize;

    use super::*;
    use crate::testing::FsFixture;

    #[tokio::test]
    async fn test_split_range_reassembles() {
        // Neither the range nor its chunks are whole multiples of the chunk or
        // copy piece size, so every split boundary is exercised.
        let data: Vec<u8> = (0..13 * 1024 * 1024 + 7).map(|i| (i % 251) as u8).collect();
        let fs = FsFixture::default();
        fs.object_store.put(&fs.tmp_path, &data).await.unwrap();
        let scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let io = LanceEncodingsIo::new(scheduler).with_read_chunk_size(5 * 1024 * 1024);

        let ranges = vec![0..10, 10..10, 10..data.len() as u64 - 3];
        let results = io.submit_request(ranges.clone(), 0).await.unwrap();

        assert_eq!(results.len(), ranges.len());
        for (range, bytes) in ranges.iter().zip(&results) {
            assert_eq!(
                bytes.as_ref(),
                &data[range.start as usize..range.end as usize],
                "range {range:?}"
            );
        }
    }
}
