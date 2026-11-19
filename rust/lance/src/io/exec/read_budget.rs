// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading fragments under a session's read memory budget.
//!
//! A budgeted scan splits each fragment's ranges into chunks and reserves a
//! chunk's pages and decoded size before requesting the pages. Each output
//! batch takes its share of that reservation until the scan emits it; the
//! rest is released once the chunk is decoded. Chunks are admitted in the
//! order the scan consumes them.
//!
//! Page metadata gives the decoded size of fixed-width and dictionary pages.
//! For the other bytes, a scan assumes `UNMEASURED_DECODED_PER_RAW` until it
//! has decoded a batch, then uses what its batches decoded to.

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
/// own. A larger batch replaces it at once, since under-reserving can overrun
/// the limit.
const DECODED_FALL_WEIGHT: f64 = 0.25;

/// Decoded bytes per byte on disk a scan assumes for pages whose decoded size
/// the metadata does not give, until it has decoded a batch. Compressed pages
/// decode to a few times their size on disk.
const UNMEASURED_DECODED_PER_RAW: f64 = 4.0;

/// Ranges of one fragment that a budgeted scan admits and reads together.
pub(super) struct ReadChunk {
    pub ranges: Vec<Range<u64>>,
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

/// Splits `ranges` of `reader` into chunks of about one `chunk_size` of
/// memory each, or of whole batches when `keep_batch_size` is set.
pub(super) async fn plan_chunks(
    reader: &FragmentReader,
    ranges: &[Range<u64>],
    batch_size: u32,
    keep_batch_size: bool,
    chunk_size: &ChunkSize,
) -> Result<Vec<ReadChunk>> {
    let estimate = reader.estimate_read(ranges.to_vec().into()).await?;
    let rows: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    let batch_size = (batch_size as u64).max(1);
    let chunks = (chunk_size.memory(estimate) / chunk_size.chunk_bytes as f64).ceil();
    let rows_per_chunk = rows.div_ceil((chunks as u64).max(1));
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

/// What the bytes whose decoded size the metadata does not give decode to,
/// per byte read, learned from a scan's decoded batches.
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

    /// What `estimate` decodes to: the metadata's figure where it knows, and
    /// the learned or assumed ratio for the rest.
    fn expected(&self, estimate: ReadEstimate) -> f64 {
        known_decoded(estimate) + estimate.unknown_bytes as f64 * self.per_raw()
    }

    /// What to reserve for `estimate`'s decoded data. Until a batch is
    /// decoded, the assumed ratio may raise it only up to `chunk_bytes` in
    /// all, so a unit larger than a chunk, such as a take's whole batch, does
    /// not reserve several times its size.
    fn decoded(&self, estimate: ReadEstimate, chunk_bytes: u64) -> u64 {
        if self.learned().is_some() {
            return self.expected(estimate).round() as u64;
        }
        let known = known_decoded(estimate);
        let unknown = estimate.unknown_bytes as f64;
        let room = chunk_bytes as f64 - estimate.bytes as f64 - known;
        (known
            + (unknown * UNMEASURED_DECODED_PER_RAW)
                .min(room)
                .max(unknown)) as u64
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

/// What the metadata knows `estimate` decodes to.
fn known_decoded(estimate: ReadEstimate) -> f64 {
    estimate
        .decoded_bytes
        .saturating_sub(estimate.unknown_bytes) as f64
}

/// One scan's use of the read memory budget.
pub struct ScanBudget {
    reader: BudgetReader,
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

    /// The scan's budget for another independently consumed stream, such as a
    /// partition. It shares what the scan learns about decoded sizes.
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

    /// A reader for blob payloads the scan reads after their descriptors.
    pub fn payload_reader(&self) -> BudgetReader {
        self.reader.sibling()
    }

    /// The memory per chunk the scan splits fragments into.
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
            reader: self.reader.clone(),
            decoded: self.decoded.clone(),
            chunk_bytes: self.reader.chunk_bytes(),
            wait_time: self.wait_time.clone(),
        }
    }
}

/// The memory, pages plus decoded data, that a scan's chunks aim for.
#[derive(Clone)]
pub struct ChunkSize {
    pub chunk_bytes: u64,
    decoded: Arc<DecodedSize>,
}

impl ChunkSize {
    /// The memory a read of `estimate` is expected to take: its pages and
    /// their decoded data.
    pub fn memory(&self, estimate: ReadEstimate) -> f64 {
        estimate.bytes as f64 + self.decoded.expected(estimate)
    }
}

pub struct ChunkAdmission {
    admission: Admission,
    reader: BudgetReader,
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
            reader: self.reader,
            estimate,
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
    reader: BudgetReader,
    estimate: ReadEstimate,
    rows: u64,
    decoded: Arc<DecodedSize>,
    /// Held while the first batch resizes the reservation, so no batch moves
    /// its share out of a reservation that is about to be replaced.
    measured: Mutex<bool>,
}

impl ChunkBudget {
    /// Charges an output batch of the chunk until the batch is dropped, taken
    /// from the chunk's decoded share so it is not counted twice. Never waits.
    pub fn charge_output(&self, bytes: u64) -> Reservation {
        self.reservation.split_off(bytes, self.estimate.bytes)
    }

    /// A reader for the blob payloads of the chunk's batches, which are
    /// admitted after the chunk itself.
    pub fn payload_reader(&self) -> BudgetReader {
        self.reader.sibling()
    }

    /// Resizes the reservation from the chunk's first decoded batch, and
    /// teaches the scan's decoded size from every batch.
    pub fn observe_decoded(&self, batch: &RecordBatch) {
        // Files without page metadata (v1) estimate nothing to correct.
        if batch.num_rows() == 0 || self.estimate.bytes == 0 {
            return;
        }
        let decoded =
            batch.get_array_memory_size() as f64 * self.rows as f64 / batch.num_rows() as f64;
        {
            let mut measured = self.measured.lock().unwrap();
            if !std::mem::replace(&mut *measured, true) {
                self.reservation
                    .resize(self.estimate.bytes + decoded as u64);
            }
        }
        if self.estimate.unknown_bytes > 0 {
            let unknown = (decoded - known_decoded(self.estimate)).max(0.0);
            self.decoded
                .learn(unknown / self.estimate.unknown_bytes as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use lance_io::read_budget::{QueryId, ReadMemoryBudget};

    use super::*;

    /// `bytes` on disk, of which `unknown` have a decoded size the metadata
    /// does not give, and the rest decode to `known`.
    fn estimate(bytes: u64, known: u64, unknown: u64) -> ReadEstimate {
        ReadEstimate {
            bytes,
            decoded_bytes: known + unknown,
            unknown_bytes: unknown,
        }
    }

    fn scan_budget() -> (Arc<ReadMemoryBudget>, ScanBudget) {
        let budget = Arc::new(ReadMemoryBudget::new(1 << 30));
        let scan = ScanBudget::new(budget.reader(QueryId::unique()), Time::new());
        (budget, scan)
    }

    /// Before a batch is decoded, chunks reserve what the metadata knows plus
    /// the assumed ratio for the rest; after, what batches decoded to per
    /// unknown byte read.
    #[tokio::test]
    async fn test_chunk_budget_learns_decoded_size() {
        let (budget, scan) = scan_budget();
        let unmeasured = 1000 + 100 + (500.0 * UNMEASURED_DECODED_PER_RAW) as u64;
        let chunk = scan
            .admission()
            .reserve(100, estimate(1000, 100, 500))
            .await;
        let unused = scan
            .admission()
            .reserve(100, estimate(1000, 100, 500))
            .await;
        assert_eq!(budget.reserved_bytes(), 2 * unmeasured);
        drop(unused);
        // Exactly what the metadata says when it knows every byte, and nothing
        // is learned from it.
        let known = scan.admission().reserve(100, estimate(1000, 100, 0)).await;
        assert_eq!(budget.reserved_bytes(), unmeasured + 1100);
        drop(known);
        let batch =
            arrow_array::record_batch!(("x", Int32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).unwrap();
        let decoded = (batch.get_array_memory_size() as f64 * 10.0) as u64;
        chunk.observe_decoded(&batch);
        assert_eq!(budget.reserved_bytes(), 1000 + decoded);
        // The next chunk keeps the known 100 and learns the rest per unknown
        // byte.
        let per_unknown = (decoded as f64 - 100.0) / 500.0;
        assert_eq!(scan.decoded.learned(), Some(per_unknown));
        let _next = scan
            .admission()
            .reserve(100, estimate(1000, 100, 500))
            .await;
        assert_eq!(budget.reserved_bytes(), 2 * (1000 + decoded));
    }

    /// Output is taken from its chunk's decoded share and charged until
    /// dropped; past that share it is charged on top, since the chunk still
    /// holds its pages.
    #[rstest::rstest]
    #[case::within_the_decoded_share(400, 1500)]
    #[case::past_the_decoded_share(2000, 3000)]
    #[tokio::test]
    async fn test_charge_output(#[case] output: u64, #[case] reserved: u64) {
        let (budget, scan) = scan_budget();
        let chunk = scan.admission().reserve(100, estimate(1000, 500, 0)).await;
        let output = chunk.charge_output(output);
        assert_eq!(budget.reserved_bytes(), reserved);
        drop(chunk);
        assert_eq!(budget.reserved_bytes(), output.bytes());
        drop(output);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// A larger decoded size replaces the learned one at once; a smaller one
    /// moves it only part of the way.
    #[test]
    fn test_decoded_size_rises_at_once_and_falls_slowly() {
        let decoded = DecodedSize::new();
        assert_eq!(decoded.per_raw(), UNMEASURED_DECODED_PER_RAW);
        decoded.learn(4.0);
        assert_eq!(decoded.per_raw(), 4.0);
        decoded.learn(8.0);
        assert_eq!(decoded.per_raw(), 8.0);
        decoded.learn(0.0);
        assert_eq!(decoded.per_raw(), 8.0 * (1.0 - DECODED_FALL_WEIGHT));
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
