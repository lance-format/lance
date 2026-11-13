// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading fragments under a session's read memory budget.
//!
//! A budgeted scan splits each fragment's ranges into chunks and reserves a
//! chunk's pages and decoded size before requesting the pages. Each output
//! batch takes its share of that reservation until the scan emits it; the
//! rest ends once the chunk is decoded. Chunks must be admitted in the order the
//! scan consumes them; see [`lance_io::read_budget`].
//!
//! A chunk's decoded size is first estimated from page metadata, which knows
//! what dictionary pages decode to but not what compressed ones do. Until a
//! scan has decoded a batch, it assumes `UNMEASURED_DECODED_PER_RAW`: its
//! chunks read that much less from disk and reserve the same memory as later
//! ones, so a scan reads ahead at once without risking the limit on data that
//! decodes larger than it is on disk. From its first decoded batch on, a scan
//! reserves what its batches decoded to per byte read: raised at once by a
//! larger batch, lowered slowly by smaller ones.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use datafusion::physical_plan::metrics::Time;
use lance_core::Result;
use lance_file::reader::ReadEstimate;
use lance_io::read_budget::{Admission, BudgetReader, Reservation};

use crate::dataset::fragment::FragmentReader;

/// How far one smaller batch moves a scan's learned decoded size toward its
/// own. A larger batch replaces it at once: under-reserving is what overruns
/// the limit, over-reserving only reads less ahead.
const DECODED_FALL_WEIGHT: f64 = 0.25;

/// Decoded bytes per byte on disk a scan assumes until it has decoded a batch,
/// unless the metadata knows more: compressed pages decode to a few times their
/// size on disk. Chunks are sized by it, so a first chunk reads less but
/// reserves no more than a later one.
const UNMEASURED_DECODED_PER_RAW: f64 = 4.0;

/// Ranges of one fragment that a budgeted scan admits and reads together.
pub(super) struct ReadChunk {
    pub ranges: Vec<Range<u64>>,
    /// What the ranges load
    pub estimate: ReadEstimate,
}

impl ReadChunk {
    pub fn rows(&self) -> u64 {
        self.ranges.iter().map(|r| r.end - r.start).sum()
    }
}

/// Splits sorted `ranges` into consecutive groups of `rows_per_chunk` rows (the
/// last group may be smaller).
pub(super) fn split_ranges(ranges: &[Range<u64>], rows_per_chunk: u64) -> Vec<Vec<Range<u64>>> {
    let mut chunks = Vec::new();
    let mut chunk = Vec::new();
    let mut chunk_rows = 0;
    for range in ranges {
        let mut start = range.start;
        while start < range.end {
            let end = range.end.min(start + rows_per_chunk - chunk_rows);
            chunk.push(start..end);
            chunk_rows += end - start;
            start = end;
            if chunk_rows == rows_per_chunk {
                chunks.push(std::mem::take(&mut chunk));
                chunk_rows = 0;
            }
        }
    }
    if !chunk.is_empty() {
        chunks.push(chunk);
    }
    chunks
}

/// Splits `ranges` of `reader` into chunks of about `chunk_bytes` on disk, or
/// of whole batches when `keep_batch_size` is set.
pub(super) async fn plan_chunks(
    reader: &FragmentReader,
    ranges: &[Range<u64>],
    batch_size: u32,
    keep_batch_size: bool,
    chunk_bytes: u64,
) -> Result<Vec<ReadChunk>> {
    let estimate = reader.estimate_read(ranges.to_vec().into()).await?;
    let bytes = estimate.bytes;
    let rows: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    let batch_size = (batch_size as u64).max(1);
    let rows_per_chunk = rows.div_ceil(bytes.div_ceil(chunk_bytes).max(1));
    // Whole batches avoid short batches at chunk ends, but a chunk smaller than
    // one batch is decoded as its own smaller batch rather than rounded up past
    // the target.
    let rows_per_chunk = if rows_per_chunk >= batch_size || keep_batch_size {
        rows_per_chunk.next_multiple_of(batch_size)
    } else {
        rows_per_chunk
    };
    if rows_per_chunk >= rows {
        return Ok(vec![ReadChunk {
            ranges: ranges.to_vec(),
            estimate,
        }]);
    }
    futures::future::try_join_all(split_ranges(ranges, rows_per_chunk).into_iter().map(
        |ranges| async move {
            let estimate = reader.estimate_read(ranges.clone().into()).await?;
            Ok(ReadChunk { ranges, estimate })
        },
    ))
    .await
}

/// Decoded bytes per byte read, learned from each chunk's first batch and
/// shared by all of a scan's chunks.
struct DecodedSize {
    /// NaN until a batch has been decoded.
    decoded_per_raw: AtomicU64,
}

impl DecodedSize {
    fn new() -> Self {
        Self {
            decoded_per_raw: AtomicU64::new(f64::NAN.to_bits()),
        }
    }

    fn learned(&self) -> Option<f64> {
        let ratio = f64::from_bits(self.decoded_per_raw.load(Ordering::Relaxed));
        (!ratio.is_nan()).then_some(ratio)
    }

    fn per_raw(&self) -> f64 {
        self.learned().unwrap_or(UNMEASURED_DECODED_PER_RAW)
    }

    /// What `estimate` decodes to: as learned once a batch is. Before that,
    /// the assumed ratio, but only up to a chunk of `chunk_bytes` in all: a
    /// unit larger than a chunk, such as a take's whole batch, would otherwise
    /// reserve several times its size. The metadata's figure when larger.
    fn decoded(&self, estimate: ReadEstimate, chunk_bytes: u64) -> u64 {
        match self.learned() {
            Some(ratio) => (estimate.bytes as f64 * ratio) as u64,
            None => ((estimate.bytes as f64 * UNMEASURED_DECODED_PER_RAW) as u64)
                .min(chunk_bytes.saturating_sub(estimate.bytes))
                .max(estimate.decoded_bytes),
        }
    }

    fn learn(&self, observed: f64) {
        let _ = self
            .decoded_per_raw
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                let current = f64::from_bits(bits);
                let next = if current.is_nan() || observed >= current {
                    observed
                } else {
                    current + (observed - current) * DECODED_FALL_WEIGHT
                };
                Some(next.to_bits())
            });
    }
}

/// One scan's use of the read memory budget.
pub struct ScanBudget {
    reader: BudgetReader,
    /// Reserve the whole read as one unit instead of chunk by chunk, for
    /// consumers that collect all of it before using any.
    whole_read: bool,
    decoded: Arc<DecodedSize>,
    /// Where the scan's reads add the time they waited for the budget.
    wait_time: Time,
}

impl ScanBudget {
    pub fn new(reader: BudgetReader, wait_time: Time) -> Self {
        Self {
            reader,
            whole_read: false,
            decoded: Arc::new(DecodedSize::new()),
            wait_time,
        }
    }

    /// A budget that reserves a whole read at once: its consumer collects all
    /// of the read before using any, so reserving it in chunks would make it
    /// wait for itself.
    pub fn whole_read(reader: BudgetReader, wait_time: Time) -> Self {
        Self {
            whole_read: true,
            ..Self::new(reader, wait_time)
        }
    }

    /// The same scan's budget for another stream of its reads, such as a
    /// partition, which is consumed independently and so needs its own reader.
    /// What the scan learns about decoded sizes is shared.
    pub fn sibling(&self) -> Self {
        Self {
            reader: self.reader.sibling(),
            whole_read: self.whole_read,
            decoded: self.decoded.clone(),
            wait_time: self.wait_time.clone(),
        }
    }

    pub fn is_whole_read(&self) -> bool {
        self.whole_read
    }

    /// The size on disk the scan splits fragments into.
    pub fn chunk_size(&self) -> ChunkSize {
        ChunkSize {
            chunk_bytes: self.reader.chunk_bytes(),
            decoded: self.decoded.clone(),
        }
    }

    /// The next chunk's place in line. Call in the order the scan consumes
    /// its chunks.
    pub fn admission(&self) -> ChunkAdmission {
        ChunkAdmission {
            admission: self.reader.admission(),
            decoded: self.decoded.clone(),
            chunk_bytes: self.reader.chunk_bytes(),
            wait_time: self.wait_time.clone(),
        }
    }
}

/// The size on disk of a scan's chunks, chosen so that a chunk's pages and
/// their decoded data, which a chunk reserves together, are about one chunk of
/// the budget. Follows the decoded size the scan learns.
#[derive(Clone)]
pub struct ChunkSize {
    chunk_bytes: u64,
    decoded: Arc<DecodedSize>,
}

impl ChunkSize {
    pub fn bytes(&self) -> u64 {
        ((self.chunk_bytes as f64 / (1.0 + self.decoded.per_raw())) as u64).max(1)
    }
}

pub struct ChunkAdmission {
    admission: Admission,
    decoded: Arc<DecodedSize>,
    chunk_bytes: u64,
    wait_time: Time,
}

impl ChunkAdmission {
    /// Waits until `rows` loading `estimate` may be read and reserves their
    /// pages and decoded data, which are in memory at once.
    pub async fn reserve(self, rows: u64, estimate: ReadEstimate) -> Arc<ChunkBudget> {
        let decoded = self.decoded.decoded(estimate, self.chunk_bytes);
        let reservation = {
            let _waiting = self.wait_time.timer();
            self.admission.reserve(estimate.bytes + decoded).await
        };
        Arc::new(ChunkBudget {
            reservation,
            raw_bytes: estimate.bytes,
            rows,
            decoded: self.decoded,
            measured: Mutex::new(false),
        })
    }
}

/// The memory reserved for decoding one chunk: its pages and decoded data,
/// released once the chunk is decoded. Its output batches move their share
/// out with [`ChunkBudget::charge_output`] and keep it until emitted.
pub struct ChunkBudget {
    reservation: Reservation,
    raw_bytes: u64,
    rows: u64,
    decoded: Arc<DecodedSize>,
    /// Held while the first batch resizes the reservation, so no batch moves
    /// its share out of a reservation that is about to be replaced.
    measured: Mutex<bool>,
}

impl ChunkBudget {
    /// Charges an output batch of the chunk until the batch is dropped, moving
    /// it out of the chunk's reservation so it is not counted twice. It is
    /// already in memory, so this never waits.
    pub fn charge_output(&self, bytes: u64) -> Reservation {
        self.reservation.split_off(bytes)
    }

    /// Sizes the reservation from the chunk's first decoded batch, which shows
    /// how much the chunk's data grows when decoded.
    pub fn observe_decoded(&self, batch: &RecordBatch) {
        if batch.num_rows() == 0 || self.raw_bytes == 0 {
            return;
        }
        let mut measured = self.measured.lock().unwrap();
        if std::mem::replace(&mut *measured, true) {
            return;
        }
        let decoded =
            batch.get_array_memory_size() as f64 * self.rows as f64 / batch.num_rows() as f64;
        self.reservation.resize(self.raw_bytes + decoded as u64);
        self.decoded.learn(decoded / self.raw_bytes as f64);
    }
}

#[cfg(test)]
mod tests {
    use lance_io::read_budget::{QueryId, ReadMemoryBudget};

    use super::*;

    fn estimate(bytes: u64, decoded_bytes: u64) -> ReadEstimate {
        ReadEstimate {
            bytes,
            decoded_bytes,
        }
    }

    /// Before a batch is decoded, chunks reserve the assumed decoded size, or
    /// the metadata's when larger, and are admitted without waiting on each
    /// other; after, what batches decoded to per byte read. Output moves out
    /// of its chunk's reservation and is charged until dropped.
    #[tokio::test]
    async fn test_chunk_budget_learns_decoded_size() {
        let budget = Arc::new(ReadMemoryBudget::new(1 << 30));
        let scan = ScanBudget::new(budget.reader(QueryId::unique()), Time::new());
        let unmeasured = 1000 + ((1000.0 * UNMEASURED_DECODED_PER_RAW) as u64).max(3000);
        let chunk = scan.admission().reserve(100, estimate(1000, 3000)).await;
        let unused = scan.admission().reserve(100, estimate(1000, 3000)).await;
        assert_eq!(budget.reserved_bytes(), 2 * unmeasured);
        drop(unused);
        let batch =
            arrow_array::record_batch!(("x", Int32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).unwrap();
        let decoded = (batch.get_array_memory_size() as f64 * 10.0) as u64;
        chunk.observe_decoded(&batch);
        assert_eq!(budget.reserved_bytes(), 1000 + decoded);
        let next = scan.admission().reserve(100, estimate(1000, 3000)).await;
        assert_eq!(budget.reserved_bytes(), 2 * (1000 + decoded));
        // Output moves out of its chunk's reservation, not on top of it.
        let output = chunk.charge_output(500);
        assert_eq!(budget.reserved_bytes(), 2 * (1000 + decoded));
        drop(chunk);
        assert_eq!(budget.reserved_bytes(), 1000 + decoded + 500);
        drop((output, next));
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// A larger decoded size replaces the learned one at once; a smaller one
    /// moves it only part of the way.
    #[tokio::test]
    async fn test_decoded_size_rises_at_once_and_falls_slowly() {
        let budget = Arc::new(ReadMemoryBudget::new(1 << 30));
        let scan = ScanBudget::new(budget.reader(QueryId::unique()), Time::new());
        let learned = |scan: &ScanBudget| scan.decoded.per_raw();
        scan.decoded.learn(4.0);
        assert_eq!(learned(&scan), 4.0);
        scan.decoded.learn(8.0);
        assert_eq!(learned(&scan), 8.0);
        scan.decoded.learn(0.0);
        assert_eq!(learned(&scan), 8.0 * (1.0 - DECODED_FALL_WEIGHT));
    }

    #[test]
    fn test_split_ranges() {
        assert_eq!(
            split_ranges(&[0..10], 4),
            vec![vec![0..4], vec![4..8], vec![8..10]]
        );
        // A chunk can take rows from several ranges, and a range can span chunks.
        assert_eq!(
            split_ranges(&[0..3, 5..6, 10..15], 4),
            vec![vec![0..3, 5..6], vec![10..14], vec![14..15]]
        );
        assert_eq!(split_ranges(&[0..4], 4), vec![vec![0..4]]);
        assert!(split_ranges(&[], 4).is_empty());
    }
}
