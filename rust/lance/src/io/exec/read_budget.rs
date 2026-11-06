// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading fragments under a session's read memory budget.
//!
//! A budgeted scan splits each fragment's ranges into chunks and reserves a
//! chunk's pages and decoded size before requesting the pages. That
//! reservation ends once the chunk is decoded; its output batches are then
//! charged until the scan emits them. Chunks must be admitted in the order the
//! scan consumes them; see [`lance_io::read_budget`].

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_io::read_budget::{Admission, BudgetReader, Reservation};

use crate::dataset::fragment::FragmentReader;

/// Under a read memory budget, a scan reads in chunks of about this fraction of
/// the limit, so that many scans can read ahead at once and the one overdraft
/// is small.
const READ_CHUNKS_PER_BUDGET: u64 = 16;

/// The size reads are split into under a budget of `limit` bytes.
pub fn chunk_bytes(limit: u64) -> u64 {
    (limit / READ_CHUNKS_PER_BUDGET).max(1)
}

/// Ranges of one fragment that a budgeted scan admits and reads together.
pub(super) struct ReadChunk {
    pub ranges: Vec<Range<u64>>,
    /// Bytes on disk the ranges load
    pub bytes: u64,
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

/// Splits `ranges` of `reader` into chunks of about `chunk_bytes` on disk.
pub(super) async fn plan_chunks(
    reader: &FragmentReader,
    ranges: &[Range<u64>],
    batch_size: u32,
    chunk_bytes: u64,
) -> Result<Vec<ReadChunk>> {
    let bytes = reader.estimate_read_bytes(ranges.to_vec().into()).await?;
    let rows: u64 = ranges.iter().map(|r| r.end - r.start).sum();
    let batch_size = (batch_size as u64).max(1);
    let rows_per_chunk = rows.div_ceil(bytes.div_ceil(chunk_bytes).max(1));
    // Whole batches avoid short batches at chunk ends, but a chunk smaller than
    // one batch is decoded as its own smaller batch rather than rounded up past
    // the target.
    let rows_per_chunk = if rows_per_chunk >= batch_size {
        rows_per_chunk.next_multiple_of(batch_size)
    } else {
        rows_per_chunk
    };
    if rows_per_chunk >= rows {
        return Ok(vec![ReadChunk {
            ranges: ranges.to_vec(),
            bytes,
        }]);
    }
    futures::future::try_join_all(split_ranges(ranges, rows_per_chunk).into_iter().map(
        |ranges| async move {
            let bytes = reader.estimate_read_bytes(ranges.clone().into()).await?;
            Ok(ReadChunk { ranges, bytes })
        },
    ))
    .await
}

/// One scan's use of the read memory budget.
pub struct ScanBudget {
    reader: BudgetReader,
    /// Reserve the whole read as one unit instead of chunk by chunk, for
    /// consumers that collect all of it before using any.
    whole_read: bool,
    /// Decoded bytes per byte read, learned from the scan's first batches.
    decoded_per_raw: Arc<AtomicU64>,
}

impl ScanBudget {
    pub fn new(reader: BudgetReader) -> Self {
        Self {
            reader,
            whole_read: false,
            decoded_per_raw: Arc::new(AtomicU64::new(1.0_f64.to_bits())),
        }
    }

    /// A budget that reserves a whole read at once: its consumer collects all
    /// of the read before using any, so reserving it in chunks would make it
    /// wait for itself.
    pub fn whole_read(reader: BudgetReader) -> Self {
        Self {
            whole_read: true,
            ..Self::new(reader)
        }
    }

    /// The same scan's budget for another partition, which consumes its
    /// chunks independently and so needs its own reader.
    pub fn for_partition(&self) -> Self {
        Self {
            reader: self.reader.sibling(),
            whole_read: self.whole_read,
            decoded_per_raw: self.decoded_per_raw.clone(),
        }
    }

    pub fn is_whole_read(&self) -> bool {
        self.whole_read
    }

    /// The size the scan splits fragments into.
    pub fn chunk_bytes(&self) -> u64 {
        chunk_bytes(self.reader.limit())
    }

    /// The next chunk's place in line. Call in the order the scan consumes
    /// its chunks.
    pub fn admission(&self) -> ChunkAdmission {
        ChunkAdmission {
            admission: self.reader.admission(),
            decoded_per_raw: self.decoded_per_raw.clone(),
        }
    }
}

pub struct ChunkAdmission {
    admission: Admission,
    decoded_per_raw: Arc<AtomicU64>,
}

impl ChunkAdmission {
    /// Waits until `rows` loading `bytes` on disk may be read and reserves their
    /// pages and decoded data, which are in memory at once.
    pub async fn reserve(self, rows: u64, bytes: u64) -> Arc<ChunkBudget> {
        let decoded = bytes as f64 * f64::from_bits(self.decoded_per_raw.load(Ordering::Relaxed));
        Arc::new(ChunkBudget {
            reservation: self.admission.reserve(bytes + decoded as u64).await,
            raw_bytes: bytes,
            rows,
            decoded_per_raw: self.decoded_per_raw,
            measured: AtomicBool::new(false),
        })
    }
}

/// The memory reserved for decoding one chunk: its pages and decoded data,
/// released once the chunk is decoded. What the scan then still holds is its
/// output, charged batch by batch with [`ChunkBudget::charge_output`].
pub struct ChunkBudget {
    reservation: Reservation,
    raw_bytes: u64,
    rows: u64,
    decoded_per_raw: Arc<AtomicU64>,
    measured: AtomicBool,
}

impl ChunkBudget {
    /// Charges an output batch of the chunk until the batch is dropped. It is
    /// already in memory, so this never waits.
    pub fn charge_output(&self, bytes: u64) -> Reservation {
        self.reservation.reader().charge(bytes)
    }

    /// Sizes the reservation from the chunk's first decoded batch, which shows
    /// how much the chunk's data grows when decoded.
    pub fn observe_decoded(&self, batch: &RecordBatch) {
        if batch.num_rows() == 0
            || self.raw_bytes == 0
            || self.measured.swap(true, Ordering::Relaxed)
        {
            return;
        }
        let decoded =
            batch.get_array_memory_size() as f64 * self.rows as f64 / batch.num_rows() as f64;
        self.decoded_per_raw.store(
            (decoded / self.raw_bytes as f64).to_bits(),
            Ordering::Relaxed,
        );
        self.reservation.resize(self.raw_bytes + decoded as u64);
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::ExecutionPlan;
    use futures::TryStreamExt;
    use lance_core::ROW_ID;
    use lance_core::datatypes::OnMissing;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{ByteCount, array, gen_batch};
    use lance_io::read_budget::{QueryId, ReadMemoryBudget};
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::ProjectionRequest;
    use crate::dataset::builder::DatasetBuilder;
    use crate::io::exec::filtered_read::{
        FilteredReadExec, FilteredReadOptions, FilteredReadThreadingMode,
    };
    use crate::io::exec::{LanceScanConfig, LanceScanExec, TakeExec};
    use crate::session::Session;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    /// Small batches split each fragment into many chunks.
    const BATCH_SIZE: usize = 64;

    async fn open(path: &str, limit: Option<u64>) -> Arc<Dataset> {
        let mut session = Session::default();
        if let Some(limit) = limit {
            session = session.with_read_memory_limit(limit);
        }
        let dataset = DatasetBuilder::from_uri(path)
            .with_session(Arc::new(session))
            .load()
            .await
            .unwrap();
        Arc::new(dataset)
    }

    fn sorted_ids(batches: &[RecordBatch]) -> Vec<i32> {
        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| batch["id"].as_primitive::<Int32Type>().values().to_vec())
            .collect();
        ids.sort_unstable();
        ids
    }

    async fn filtered_scan(dataset: &Dataset) -> Vec<i32> {
        let mut scanner = dataset.scan();
        scanner
            .filter("id % 3 = 0")
            .unwrap()
            .project(&["id", "payload"])
            .unwrap()
            .batch_size(BATCH_SIZE);
        let batches: Vec<_> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        sorted_ids(&batches)
    }

    async fn full_fragment_scan(dataset: &Arc<Dataset>) -> Vec<i32> {
        let scan = LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig {
                batch_size: BATCH_SIZE,
                ..Default::default()
            },
        );
        let batches: Vec<_> = scan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        sorted_ids(&batches)
    }

    /// Scans the ids with their row addresses, then takes each row's payload.
    async fn take_payloads(dataset: &Arc<Dataset>) -> Vec<i32> {
        let scan = LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().project(&["id"]).unwrap()),
            LanceScanConfig {
                batch_size: BATCH_SIZE,
                with_row_address: true,
                ..Default::default()
            },
        );
        let payload = dataset
            .empty_projection()
            .union_column("payload", OnMissing::Error)
            .unwrap();
        let take = TakeExec::try_new(dataset.clone(), Arc::new(scan), payload)
            .unwrap()
            .unwrap();
        let batches: Vec<RecordBatch> = take
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        for batch in &batches {
            let payload = batch["payload"].as_binary::<i64>();
            assert_eq!(payload.null_count(), 0);
            assert!(payload.iter().all(|value| value.unwrap().len() == 2048));
        }
        sorted_ids(&batches)
    }

    /// Takes rows by position; `id` equals the position, so the ids must come
    /// back exactly as requested, in order and with duplicates.
    async fn check_dataset_take(dataset: &Dataset) {
        let contiguous: Vec<u64> = (100..300).collect();
        let sorted = vec![5, 1500, 3000, 4999, 7999];
        let unsorted = vec![7000, 3, 3, 2500, 10, 6999];
        for rows in [contiguous, sorted, unsorted] {
            let batch = dataset
                .take(
                    &rows,
                    ProjectionRequest::from_columns(["id", "payload"], dataset.schema()),
                )
                .await
                .unwrap();
            let ids: Vec<u64> = batch["id"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .map(|id| *id as u64)
                .collect();
            assert_eq!(ids, rows);
            assert_eq!(batch["payload"].null_count(), 0);
        }
    }

    /// Budgeted scans return exactly the rows of unbudgeted ones, whether every
    /// chunk fits, chunks wait for each other, or every chunk overdraws.
    #[rstest]
    #[case::every_chunk_overdraws(1)]
    #[case::chunks_wait(64 * 1024)]
    #[case::chunks_fit(16 * 1024 * 1024)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_budgeted_scans_return_every_row(#[case] limit: u64) {
        let dir = TempStrDir::default();
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("payload", array::rand_fixedbin(ByteCount::from(2048), true))
            .into_dataset(&dir, FragmentCount::from(8), FragmentRowCount::from(1000))
            .await
            .unwrap();
        let unbudgeted = open(&dir, None).await;
        let budgeted = open(&dir, Some(limit)).await;

        let expected = filtered_scan(&unbudgeted).await;
        assert_eq!(filtered_scan(&budgeted).await, expected);
        assert_eq!(
            full_fragment_scan(&budgeted).await,
            full_fragment_scan(&unbudgeted).await
        );
        assert_eq!(
            take_payloads(&budgeted).await,
            take_payloads(&unbudgeted).await
        );
        check_dataset_take(&budgeted).await;
        let limited = budgeted
            .scan()
            .limit(Some(10), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(limited.num_rows(), 10);
        // Concurrent scans share the budget.
        let concurrent = futures::future::join_all((0..8).map(|_| filtered_scan(&budgeted))).await;
        assert!(concurrent.iter().all(|ids| *ids == expected));
        let taken = take_payloads(&unbudgeted).await;
        let concurrent = futures::future::join_all((0..8).map(|_| take_payloads(&budgeted))).await;
        assert!(concurrent.iter().all(|ids| *ids == taken));
        futures::future::join_all((0..8).map(|_| check_dataset_take(&budgeted))).await;

        let budget = budgeted.session.read_budget().unwrap();
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// Partitions admit their chunks independently: a partition holding a
    /// chunk it has not emitted never blocks another partition, even while
    /// another query fills the budget.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_budgeted_partitions_never_wait_for_each_other() {
        let dir = TempStrDir::default();
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .into_dataset(&dir, FragmentCount::from(8), FragmentRowCount::from(1000))
            .await
            .unwrap();
        let limit = 16 * 1024;
        let dataset = open(&dir, Some(limit)).await;
        let budget = dataset.session.read_budget().unwrap();
        let other_query = budget.reader(QueryId::unique()).charge(limit);

        // Chunks of hundreds of rows, emitted 16 at a time.
        let mut options = FilteredReadOptions::basic_full_read(&dataset).with_batch_size(16);
        options.threading_mode = FilteredReadThreadingMode::MultiplePartitions(2);
        let read = FilteredReadExec::try_new(dataset.clone(), options, None).unwrap();
        let context = Arc::new(TaskContext::default());
        let mut stalled = read.execute(0, context.clone()).unwrap();
        let mut batches = vec![stalled.try_next().await.unwrap().unwrap()];
        let drained = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            read.execute(1, context).unwrap().try_collect::<Vec<_>>(),
        )
        .await
        .expect("a partition waited for one that holds memory")
        .unwrap();
        batches.extend(drained);
        batches.extend(stalled.try_collect::<Vec<_>>().await.unwrap());
        assert_eq!(sorted_ids(&batches), (0..8000).collect::<Vec<_>>());
        drop(other_query);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// Rows and null row ids from a whole-fragment scan that keeps deleted
    /// rows with a null row id.
    async fn scan_with_deletions_null(dataset: &Arc<Dataset>) -> (usize, usize) {
        let scan = LanceScanExec::new(
            dataset.clone(),
            dataset.fragments().clone(),
            None,
            Arc::new(dataset.schema().clone()),
            LanceScanConfig {
                batch_size: BATCH_SIZE,
                with_row_id: true,
                with_make_deletions_null: true,
                ..Default::default()
            },
        );
        let batches: Vec<RecordBatch> = scan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches.iter().fold((0, 0), |(rows, nulls), batch| {
            (rows + batch.num_rows(), nulls + batch[ROW_ID].null_count())
        })
    }

    /// A budgeted whole-fragment scan reads every physical row, as an
    /// unbudgeted one does, so deleted rows come back with null row ids.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_budgeted_scan_keeps_deleted_rows() {
        let dir = TempStrDir::default();
        let mut dataset = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("payload", array::rand_fixedbin(ByteCount::from(2048), true))
            .into_dataset(&dir, FragmentCount::from(4), FragmentRowCount::from(1000))
            .await
            .unwrap();
        dataset.delete("id % 5 = 0").await.unwrap();
        assert_eq!(
            scan_with_deletions_null(&open(&dir, None).await).await,
            (4000, 800)
        );
        assert_eq!(
            scan_with_deletions_null(&open(&dir, Some(64 * 1024)).await).await,
            (4000, 800)
        );
    }

    /// `Dataset::take` never waits, so a caller can take while it holds a
    /// paused scan, here holding the overdraft.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_take_while_a_scan_is_paused() {
        let dir = TempStrDir::default();
        gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("payload", array::rand_fixedbin(ByteCount::from(2048), true))
            .into_dataset(&dir, FragmentCount::from(8), FragmentRowCount::from(1000))
            .await
            .unwrap();
        let dataset = open(&dir, Some(1)).await;
        let mut scanner = dataset.scan();
        scanner.batch_size(BATCH_SIZE);
        let mut paused = scanner.try_into_stream().await.unwrap();
        let mut rows = paused.try_next().await.unwrap().unwrap().num_rows();
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            check_dataset_take(&dataset),
        )
        .await
        .expect("a take waited for a paused scan");
        while let Some(batch) = paused.try_next().await.unwrap() {
            rows += batch.num_rows();
        }
        assert_eq!(rows, 8000);
        assert_eq!(dataset.session.read_budget().unwrap().reserved_bytes(), 0);
    }

    /// A chunk reserves its pages and their decoded size, learns the decoded
    /// size from its first batch for itself and later chunks, and its output
    /// is charged until dropped.
    #[tokio::test]
    async fn test_chunk_budget_learns_decoded_size() {
        let budget = Arc::new(ReadMemoryBudget::new(1 << 30));
        let scan = ScanBudget::new(budget.reader(QueryId::unique()));
        // Before any batch is decoded, decoded data is assumed as large as its pages.
        let chunk = scan.admission().reserve(100, 1000).await;
        assert_eq!(budget.reserved_bytes(), 2000);
        let batch =
            arrow_array::record_batch!(("x", Int32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).unwrap();
        let decoded = (batch.get_array_memory_size() as f64 * 10.0) as u64;
        chunk.observe_decoded(&batch);
        assert_eq!(budget.reserved_bytes(), 1000 + decoded);
        let next = scan.admission().reserve(100, 1000).await;
        assert_eq!(budget.reserved_bytes(), 2 * (1000 + decoded));
        let output = chunk.charge_output(500);
        drop(chunk);
        assert_eq!(budget.reserved_bytes(), 1000 + decoded + 500);
        drop((output, next));
        assert_eq!(budget.reserved_bytes(), 0);
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
