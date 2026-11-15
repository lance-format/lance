// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use futures::{FutureExt, future::BoxFuture};
use lance_core::{
    Result,
    utils::{allocation::ReservedBuffer, tokio::spawn_cpu},
};

use super::*;
use crate::reader::{FullZipScanOutcome, ReaderProjection};

fn batches() -> Vec<RecordBatch> {
    let meta = HashMap::from([(
        STRUCTURAL_ENCODING_META_KEY.to_string(),
        STRUCTURAL_ENCODING_FULLZIP.to_string(),
    )]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("label", DataType::Utf8, false).with_metadata(meta.clone()),
        Field::new("frames", array(0, 1).data_type().clone(), true).with_metadata(meta),
    ]));
    (0..16)
        .map(|page| {
            let frames = array(page, 1);
            let labels: ArrayRef = Arc::new(StringArray::from_iter_values(
                (0..frames.len()).map(|row| format!("p{page:02}r{row}")),
            ));
            RecordBatch::try_new(schema.clone(), vec![labels, frames]).unwrap()
        })
        .collect()
}

async fn project(fs: &FsFixture, reader: &FileReader, fields: &[usize]) -> FileReader {
    let mut schema = reader.base_projection().schema.as_ref().clone();
    schema.fields = fields
        .iter()
        .map(|index| schema.fields[*index].clone())
        .collect();
    FileReader::try_open_with_file_metadata(
        reader.core.scheduler.clone(),
        fs.tmp_path.as_ref().clone(),
        Some(ReaderProjection {
            schema: Arc::new(schema),
            column_indices: fields.iter().map(|index| *index as u32).collect(),
        }),
        Arc::<DecoderPlugins>::default(),
        reader.metadata().clone(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

fn assert_batch(batch: &RecordBatch, rows: &Range<u64>, expected: &RecordBatch) {
    assert_eq!(batch.num_rows() as u64, rows.end - rows.start);
    let expected = expected.slice(rows.start as usize, batch.num_rows());
    assert_eq!(batch.schema().fields(), expected.schema().fields());
    for (actual, expected) in batch.columns().iter().zip(expected.columns()) {
        assert_eq!(actual.to_data(), expected.to_data());
    }
}

// This independently composes the public per-column API. It is the bounded
// reference for the compound cursor's progress, not an unadmitted fallback.
async fn sequential_row(reader: &FileReader, bytes: usize) -> Option<RecordBatch> {
    let budget = AllocationBudget::new(bytes);
    let mut work = budget.try_reserve(bytes).unwrap();
    let schema = Arc::new(Schema::from(reader.schema().as_ref()));
    let mut arrays = Vec::with_capacity(schema.fields().len());
    for (column, field) in reader.metadata().column_infos.iter().zip(schema.fields()) {
        let page = &column.page_infos[0];
        let decoder = Arc::new(
            FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap(),
        );
        let (remaining, output) = decoder
            .try_read_array_range_on_cpu(0..1, field.data_type().clone(), work, |range, output| {
                reader
                    .core
                    .scheduler
                    .submit_single_into(range, page.priority, output)
            })
            .await
            .unwrap();
        work = remaining;
        let Some(output) = output else {
            drop((arrays, work));
            assert_eq!(budget.reserved(), 0);
            return None;
        };
        arrays.push(output);
    }
    let batch = RecordBatch::try_new(schema, arrays).unwrap();
    drop(work);
    Some(batch)
}

#[rstest]
#[tokio::test]
async fn batch_dispatch_preserves_feasible_single_row(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(2, 4)] columns: usize,
) {
    let source = batches().remove(0);
    let schema = Arc::new(Schema::new(
        (0..columns)
            .map(|i| {
                source
                    .schema()
                    .field(1)
                    .clone()
                    .with_name(format!("frames{i}"))
            })
            .collect::<Vec<_>>(),
    ));
    let expected = RecordBatch::try_new(
        schema,
        (0..columns).map(|_| source.column(1).clone()).collect(),
    )
    .unwrap();
    let (_fs, reader) = fixture_batches(version, false, vec![expected.clone()], 1).await;
    // Find a feasible workspace through the public per-column operation, without
    // deriving a byte formula from either implementation. Each probe releases
    // its output before the next; this does not measure timing or process RSS.
    let (mut low, mut high) = (0, 8192);
    let batch = sequential_row(&reader, high).await.unwrap();
    assert_batch(&batch, &(0..1), &expected);
    drop(batch);
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if let Some(batch) = sequential_row(&reader, mid).await {
            assert_batch(&batch, &(0..1), &expected);
            high = mid;
        } else {
            low = mid;
        }
    }
    let budget = AllocationBudget::new(high);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), high, 1)
        .await
        .unwrap();
    let outcome = scan.try_next().await.unwrap();
    let FullZipScanOutcome::Batch { rows, batch } = outcome else {
        panic!("{columns} columns fit sequentially within {high} bytes, got {outcome:?}")
    };
    assert_eq!(rows, 0..1);
    assert_batch(&batch, &rows, &expected);
    drop(batch);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scan.position(), 1);
}

#[rstest]
#[tokio::test]
async fn multi_column_prefixes_align_pages_and_keep_one_budget(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] reverse: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input).unwrap();
    // The narrow label column buffers more rows per page than the nested payload.
    let (fs, reader) = fixture_batches(version, lite, input, 2048).await;
    let layouts: Vec<Vec<u64>> = reader
        .metadata()
        .column_infos
        .iter()
        .map(|column| column.page_infos.iter().map(|page| page.num_rows).collect())
        .collect();
    assert_ne!(
        layouts[0], layouts[1],
        "fixture must cross different column page boundaries"
    );
    let order = if reverse { [1, 0] } else { [0, 1] };
    let reader = project(&fs, &reader, &order).await;
    let expected = expected.project(&order).unwrap();
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(1024);
    let end = expected.num_rows() as u64 - 1;
    let mut scan = reader
        .open_admitted_fullzip_scan(1..end, budget.clone(), 1024, 64)
        .await
        .unwrap();
    assert_eq!(stats.snapshot().iops, 0);
    let mut cursor = 1;
    let mut saw_smaller_prefix = false;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, cursor);
                assert_batch(&batch, &rows, &expected);
                saw_smaller_prefix |= rows.end - rows.start < 3;
                cursor = rows.end;
                // A single retained column alias still owns its physical charge.
                let held = batch.column(0).slice(0, 1);
                drop(batch);
                assert!(budget.reserved() > 0 && budget.reserved() <= 1024);
                let reads = stats.snapshot().iops;
                if cursor < end {
                    assert!(matches!(
                        scan.try_next().await.unwrap(),
                        FullZipScanOutcome::NeedsCapacity
                    ));
                    assert_eq!(scan.position(), cursor);
                }
                assert_eq!(stats.snapshot().iops, reads);
                drop(held);
                assert_eq!(budget.reserved(), 0);
            }
            FullZipScanOutcome::NeedsCapacity => panic!("one complete row must fit after drain"),
            FullZipScanOutcome::Finished => break,
        }
    }
    assert_eq!(cursor, end);
    assert!(saw_smaller_prefix);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn later_column_refusal_releases_earlier_output(#[values(false, true)] lite: bool) {
    let (fs, reader) = fixture_batches(ConcreteFileVersion::V2_2, lite, batches(), 1).await;
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let mut first = None;
    let ty = Schema::from(reader.schema().as_ref());
    for (index, column) in reader.metadata().column_infos.iter().enumerate() {
        let page = &column.page_infos[0];
        let decoder =
            FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
        let output = decoder
            .try_read_array_range_in(
                0..7,
                ty.field(index).data_type(),
                &mut work,
                |range, buffer| {
                    reader
                        .core
                        .scheduler
                        .submit_single_into(range, page.priority, buffer)
                },
            )
            .await
            .unwrap();
        if index == 0 {
            first = Some(output.expect("first column must complete"));
        } else {
            assert!(
                output.is_none(),
                "later column must refuse within the same workspace"
            );
        }
    }
    drop(work);
    assert!(
        budget.reserved() > 0,
        "earlier output remains owned until the compound attempt drops it"
    );
    drop(first);
    assert_eq!(budget.reserved(), 0);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 1024, 7)
        .await
        .unwrap();
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("compound retry must complete a smaller aligned prefix")
    };
    assert!(rows.end > 0 && rows.end < 7);
    assert_batch(&batch, &rows, &batches()[0]);
    drop((batch, scan, fs));
    assert_eq!(budget.reserved(), 0);
}

#[derive(Debug)]
struct LaterColumnIo {
    inner: Arc<dyn EncodingsIo>,
    second_buffers: Vec<Range<u64>>,
    // 0: normal, 1: gated, 2: error. Applies only after the first column is complete.
    mode: Arc<AtomicUsize>,
    second_reads: Arc<AtomicUsize>,
    entered: Arc<tokio::sync::Notify>,
}

impl EncodingsIo for LaterColumnIo {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<bytes::Bytes>>> {
        self.inner.submit_request(ranges, priority)
    }

    fn submit_single_into(
        &self,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        let is_second = self
            .second_buffers
            .iter()
            // A compact whole-page read can span both payload and index.
            .any(|buffer| range.start < buffer.end && buffer.start < range.end);
        let inner = self.inner.clone();
        let mode = self.mode.clone();
        let entered = self.entered.clone();
        let second_reads = self.second_reads.clone();
        async move {
            if is_second {
                second_reads.fetch_add(1, Ordering::Relaxed);
                entered.notify_one();
                match mode.load(Ordering::Relaxed) {
                    1 => std::future::pending::<()>().await,
                    2 => return Err(Error::io("injected later-column failure")),
                    _ => (),
                }
            }
            inner.submit_single_into(range, priority, output).await
        }
        .boxed()
    }
}

#[rstest]
#[tokio::test]
async fn later_column_cancellation_and_error_leave_no_partial_batch(
    #[values(false, true)] lite: bool,
    #[values(3, 7)] max_rows: u64,
) {
    let (_fs, reader) = fixture_batches(ConcreteFileVersion::V2_2, lite, batches(), 1).await;
    let mode = Arc::new(AtomicUsize::new(1));
    let second_reads = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let second_buffers = reader.metadata().column_infos[1]
        .page_infos
        .iter()
        .flat_map(|page| {
            page.buffer_offsets_and_sizes
                .iter()
                .map(|(offset, len)| *offset..*offset + *len)
        })
        .collect();
    let reader = reader.with_scheduler(Arc::new(LaterColumnIo {
        inner: reader.core.scheduler.clone(),
        second_buffers,
        mode: mode.clone(),
        second_reads: second_reads.clone(),
        entered: entered.clone(),
    }));
    let budget = AllocationBudget::new(4096);
    // Independent credit must survive a failed compound attempt, regardless of
    // whether earlier columns are still prepared input or retained raw output.
    let peer = budget.try_reserve(1024).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 3072, max_rows)
        .await
        .unwrap();
    {
        let future = scan.try_next();
        tokio::pin!(future);
        tokio::select! {
            _ = entered.notified() => (),
            result = &mut future => panic!("gated later column completed: {result:?}"),
        }
        // Earlier input, the pending read and remaining workspace stay charged
        // independently of the peer. No intermediate refund is required.
        let active = budget.reserved();
        assert!(active > peer.len() && active <= 4096);
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(
        budget.reserved(),
        peer.len(),
        "cancellation must preserve unrelated credit"
    );
    mode.store(2, Ordering::Relaxed);
    second_reads.store(0, Ordering::Relaxed);
    let error = scan.try_next().await.unwrap_err();
    assert!(matches!(error, Error::IO { .. }));
    assert!(error.to_string().contains("later-column failure"));
    assert_eq!(
        second_reads.load(Ordering::Relaxed),
        1,
        "errors must not trigger a smaller retry"
    );
    assert_eq!(scan.position(), 0);
    assert_eq!(budget.reserved(), peer.len());
    mode.store(0, Ordering::Relaxed);
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("retry must return original complete rows")
    };
    assert_eq!(rows, 0..max_rows);
    assert_batch(&batch, &rows, &batches()[0]);
    drop(batch);
    assert_eq!(budget.reserved(), peer.len());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn queued_batch_owns_all_columns_and_credit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values("delivered", "cancelled", "dropped", "undelivered", "error")] outcome: &str,
) {
    let (_fs, mut reader) = fixture_batches(version, lite, batches(), 1).await;
    if outcome == "error" {
        Arc::make_mut(&mut reader.core.base_projection.schema).fields[1].nullable = false;
    }
    let budget = AllocationBudget::new(4096);
    let peer = budget.try_reserve(1024).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 3072, 3)
        .await
        .unwrap();
    let queued = RefCell::new(None);
    let entered = tokio::sync::Notify::new();
    let dispatches = AtomicUsize::new(0);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let mut receiver = Some(receiver);
    let mut future = Box::pin(scan.try_next_with(|job| {
        dispatches.fetch_add(1, Ordering::SeqCst);
        *queued.borrow_mut() = Some(job);
        entered.notify_one();
        let receiver = receiver.take().expect("one complete batch fits");
        async move { receiver.await.unwrap() }
    }));
    tokio::select! {
        _ = entered.notified() => (),
        result = &mut future => panic!("queued batch completed: {result:?}"),
    }
    let job = queued.borrow_mut().take().unwrap();
    assert_eq!(dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(budget.reserved(), 4096);
    assert!(budget.try_reserve(1).is_none());
    match outcome {
        "delivered" => {
            sender.send(spawn_cpu(move || job.decode()).await).unwrap();
            let FullZipScanOutcome::Batch { rows, batch } = future.await.unwrap() else {
                panic!("both columns must be delivered in one batch")
            };
            assert_eq!(rows, 0..3);
            assert_eq!(scan.position(), 3);
            assert_batch(&batch, &rows, &batches()[0]);
            let alias = batch.column(1).slice(0, 1);
            drop(batch);
            assert!(budget.reserved() > peer.len());
            drop(alias);
        }
        "undelivered" => {
            sender.send(spawn_cpu(move || job.decode()).await).unwrap();
            assert!(budget.reserved() > peer.len());
            drop(future);
            assert_eq!(scan.position(), 0);
        }
        _ => {
            drop(future);
            assert_eq!(scan.position(), 0);
            assert_eq!(budget.reserved(), 4096, "queued job still owns input");
            if outcome == "dropped" {
                drop(job);
            } else {
                let result = spawn_cpu(move || job.decode()).await;
                if outcome == "error" {
                    let error = result.unwrap_err();
                    assert!(matches!(error, Error::Arrow { .. }));
                    assert!(error.to_string().contains("non-nullable"));
                } else {
                    let batch = result.unwrap().unwrap();
                    assert_batch(&batch, &(0..3), &batches()[0]);
                    drop(batch);
                }
            }
        }
    }
    assert_eq!(budget.reserved(), peer.len());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn multi_column_finite_peers_progress_under_shared_credit(#[values(false, true)] lite: bool) {
    let (_fs, reader) = fixture_batches(ConcreteFileVersion::V2_2, lite, batches(), 1).await;
    let budget = AllocationBudget::new(2048);
    let mut a = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 1024, 7)
        .await
        .unwrap();
    let mut b = reader
        .open_admitted_fullzip_scan(7..14, budget.clone(), 1024, 7)
        .await
        .unwrap();
    let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        futures::join!(a.try_next(), b.try_next())
    })
    .await
    .unwrap();
    for (result, expected, start) in [(a, &batches()[0], 0), (b, &batches()[1], 7)] {
        let FullZipScanOutcome::Batch { rows, batch } = result.unwrap() else {
            panic!("independent peer must make progress")
        };
        assert_eq!(rows.start, start);
        assert_batch(&batch, &(0..rows.end - start), expected);
        assert!(budget.reserved() <= 2048);
    }
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn exact_array_range_rejects_invalid_input_before_io() {
    let (_fs, reader) = fixture_batches(ConcreteFileVersion::V2_2, false, batches(), 1).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let page = &reader.metadata().column_infos[0].page_infos[0];
    let decoder = FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    for rows in [0..0, 0..8, Range { start: 5, end: 4 }] {
        let error = decoder
            .try_read_array_range_in(rows, &DataType::Utf8, &mut work, |range, buffer| {
                reader
                    .core
                    .scheduler
                    .submit_single_into(range, page.priority, buffer)
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("nonempty range within the page"));
    }
    let error = decoder
        .try_read_array_range_in(0..1, &DataType::Int32, &mut work, |range, buffer| {
            reader
                .core
                .scheduler
                .submit_single_into(range, page.priority, buffer)
        })
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("binary/string"));
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(work.len(), 1024);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn later_column_batch_error_refunds_all_columns(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
) {
    let (_fs, mut reader) = fixture_batches(version, false, batches(), 1).await;
    Arc::make_mut(&mut reader.core.base_projection.schema).fields[1].nullable = false;
    let budget = AllocationBudget::new(4096);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 7)
        .await
        .unwrap();
    for _ in 0..2 {
        let error = scan.try_next().await.unwrap_err();
        assert!(matches!(error, Error::Arrow { .. }));
        assert!(error.to_string().contains("frames") && error.to_string().contains("non-nullable"));
        assert_eq!(scan.position(), 0);
        assert_eq!(budget.reserved(), 0);
    }
}
