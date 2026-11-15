// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use futures::{FutureExt, future::BoxFuture};
use lance_core::{
    Result,
    utils::{allocation::ReservedBuffer, deletion::DeletionVector, tokio::spawn_cpu},
};

use super::*;
use crate::reader::{FullZipScanOutcome, ReaderProjection};

mod row_ids;
mod row_markers;
mod row_versions;

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

fn assert_address_batch(
    batch: &RecordBatch,
    rows: &Range<u64>,
    expected: &RecordBatch,
    fragment: u32,
) {
    let selected = expected.slice(rows.start as usize, (rows.end - rows.start) as usize);
    let addresses: ArrayRef =
        Arc::new(arrow_array::UInt64Array::from_iter(rows.clone().map(
            |row| (![1, 4].contains(&row)).then_some(u64::from(fragment) * (1_u64 << 32) + row),
        )));
    let mut fields = selected.schema().fields().to_vec();
    fields.extend([
        Arc::new(lance_core::ROW_ID_FIELD.clone()),
        Arc::new(lance_core::ROW_ADDR_FIELD.clone()),
    ]);
    let mut arrays = selected.columns().to_vec();
    arrays.extend([addresses.clone(), addresses]);
    fields.extend([
        Arc::new(lance_core::ROW_LAST_UPDATED_AT_VERSION_FIELD.clone()),
        Arc::new(lance_core::ROW_CREATED_AT_VERSION_FIELD.clone()),
    ]);
    let versions: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![1; selected.num_rows()]));
    arrays.extend([versions.clone(), versions]);
    assert!(Arc::ptr_eq(
        &batch[lance_core::ROW_LAST_UPDATED_AT_VERSION],
        &batch[lance_core::ROW_CREATED_AT_VERSION]
    ));
    let expected = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    assert_batch(batch, &(0..rows.end - rows.start), &expected);
    assert!(Arc::ptr_eq(
        &batch[lance_core::ROW_ID],
        &batch[lance_core::ROW_ADDR]
    ));
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
#[case::exact(false, false, 0)]
#[case::partial(true, false, 9)]
#[case::reversed_partial(true, true, 9)]
#[tokio::test]
async fn later_column_cancellation_and_error_leave_no_partial_batch(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(3, 7)] max_rows: u64,
    #[case] partial: bool,
    #[case] reverse: bool,
    #[case] start: u64,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input).unwrap();
    let (fs, reader) = fixture_batches(version, lite, input, 1).await;
    let order = if reverse { [1, 0] } else { [0, 1] };
    let reader = project(&fs, &reader, &order).await;
    let expected = expected.project(&order).unwrap();
    let mode = Arc::new(AtomicUsize::new(1));
    let second_reads = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let second_buffers = reader.metadata().column_infos[order[1]]
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
    // Another file's final Arrow alias, not just an unallocated reservation,
    // must survive cancellation/error of a partially admitted compound read.
    let (_peer_fs, peer_reader) = fixture_arrays(version, lite, vec![array(17, 1)]).await;
    let mut peer_scan = peer_reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), 4096, 1)
        .await
        .unwrap();
    let FullZipScanOutcome::Batch { batch, .. } = peer_scan.try_next().await.unwrap() else {
        panic!("peer must return its complete first row")
    };
    let peer = batch.column(0).slice(0, 1);
    drop((batch, peer_scan, peer_reader));
    let peer_bytes = budget.reserved();
    assert!(peer_bytes > 0 && peer_bytes < 4096);
    let workspace = if partial { 4096 } else { 4096 - peer_bytes };
    let end = start + 11;
    let mut scan = reader
        .open_admitted_fullzip_scan(start..end, budget.clone(), workspace, max_rows)
        .await
        .unwrap()
        .with_partial_workspace(partial);
    for selection in [start..end, end + 4..end + 15] {
        if scan.position() != selection.start {
            scan.advance_to_range(selection.clone()).unwrap();
        }
        let start = selection.start;
        let end = selection.end;
        mode.store(1, Ordering::Relaxed);
        // Completed reads can leave one notification permit. It must not stand
        // in for observing the next gated read after advancing the selection.
        let _ = entered.notified().now_or_never();
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
            assert!(active > peer_bytes && active <= 4096);
        }
        assert_eq!(scan.position(), start);
        assert_eq!(
            budget.reserved(),
            peer_bytes,
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
        assert_eq!(scan.position(), start);
        assert_eq!(budget.reserved(), peer_bytes);
        mode.store(0, Ordering::Relaxed);
        let mut next = start;
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert_eq!(rows.start, next);
                    assert!(rows.end > next && rows.end <= end);
                    assert_batch(&batch, &rows, &expected);
                    next = rows.end;
                    assert_eq!(scan.position(), next);
                    assert!(budget.reserved() <= 4096);
                    drop(batch);
                    assert_eq!(budget.reserved(), peer_bytes);
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => {
                    panic!("one row must fit beside retained peer")
                }
            }
        }
        assert_eq!(next, end);
    }
    assert_eq!(peer.to_data(), array(17, 1).slice(0, 1).to_data());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn queued_batch_owns_all_columns_and_credit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] partial: bool,
    #[values("delivered", "cancelled", "dropped", "undelivered", "error")] outcome: &str,
) {
    let (_fs, mut reader) = fixture_batches(version, lite, batches(), 1).await;
    if outcome == "error" {
        Arc::make_mut(&mut reader.core.base_projection.schema).fields[1].nullable = false;
    }
    let budget = AllocationBudget::new(4096);
    let peer = budget.try_reserve(1024).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), if partial { 4096 } else { 3072 }, 3)
        .await
        .unwrap()
        .with_partial_workspace(partial)
        .with_row_created_at_version(None)
        .unwrap()
        .with_row_id(7, None)
        .unwrap()
        .with_row_last_updated_at_version(None)
        .unwrap()
        .with_row_address(7)
        .unwrap()
        .with_deleted_rows_as_nulls(Arc::new([1_u32, 4].into_iter().collect()))
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
            assert_address_batch(&batch, &rows, &batches()[0], 7);
            let alias = batch[lance_core::ROW_ID].slice(0, 1);
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
                    assert_address_batch(&batch, &(0..3), &batches()[0], 7);
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
async fn partial_workspace_progresses_while_another_file_retains_output(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] partial: bool,
) {
    let (_fs_a, reader_a) = fixture_arrays(version, lite, vec![array(17, 1)]).await;
    let arrays = [array(31, 1), array(32, 1)];
    let expected = arrow_select::concat::concat(&[arrays[0].as_ref(), arrays[1].as_ref()]).unwrap();
    let (_fs_b, reader_b) = fixture_arrays(version, lite, arrays.to_vec()).await;
    let stats = IoStats::new();
    let reader_b = reader_b.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(4096);
    let mut a = reader_a
        .open_admitted_fullzip_scan(0..1, budget.clone(), 4096, 1)
        .await
        .unwrap();
    let FullZipScanOutcome::Batch { batch, .. } = a.try_next().await.unwrap() else {
        panic!("first file must complete one row")
    };
    assert_eq!(
        batch.column(0).to_data(),
        array(17, 1).slice(0, 1).to_data()
    );
    let mut held = Some(batch.column(0).slice(0, 1));
    drop((batch, a, reader_a));
    let held_bytes = budget.reserved();
    assert!(held_bytes > 0 && held_bytes < 4095);
    let mut b = reader_b
        .open_admitted_fullzip_scan(0..expected.len() as u64, budget.clone(), 4096, 4)
        .await
        .unwrap()
        .with_partial_workspace(partial);
    // Exhaustion must remain a finite refusal with no partial output or cursor
    // advancement, even when the smaller workspace cannot hold a minimal row.
    let blocker = budget.try_reserve(4095 - held_bytes).unwrap();
    assert!(matches!(
        b.try_next().await.unwrap(),
        FullZipScanOutcome::NeedsCapacity
    ));
    assert_eq!(b.position(), 0);
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4095);
    drop(blocker);
    if !partial {
        // Preserve default exact-workspace semantics for existing callers.
        assert!(matches!(
            b.try_next().await.unwrap(),
            FullZipScanOutcome::NeedsCapacity
        ));
        assert_eq!(b.position(), 0);
        assert_eq!(stats.snapshot().iops, 0);
        drop(held.take());
    }
    let retained = budget.reserved();
    let mut next = 0;
    loop {
        match b.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, next);
                assert!(rows.end > rows.start && rows.end <= expected.len() as u64);
                assert_eq!(
                    batch.column(0).to_data(),
                    expected.slice(next as usize, batch.num_rows()).to_data()
                );
                next = rows.end;
                assert!(budget.reserved() <= 4096);
                drop(batch);
                assert_eq!(budget.reserved(), retained);
            }
            FullZipScanOutcome::Finished => break,
            FullZipScanOutcome::NeedsCapacity => {
                panic!("second file's row must fit available credit")
            }
        }
    }
    assert_eq!(next, expected.len() as u64);
    assert!(stats.snapshot().iops > 0);
    if partial {
        assert_eq!(budget.reserved(), held_bytes);
        assert!(
            held.is_some(),
            "the slow first consumer must still own its output"
        );
    }
    drop((held, b));
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

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn finite_files_wait_for_shared_capacity_and_drain_forward_selections(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
) {
    let budget = AllocationBudget::new(4096);
    let blocker = budget.try_reserve(4096).unwrap();
    let mut fixtures = Vec::new();
    let mut expected_files = Vec::new();
    for file in 0..3 {
        let input = batches()
            .into_iter()
            .enumerate()
            .map(|(page, batch)| {
                let labels: ArrayRef = Arc::new(StringArray::from_iter_values(
                    (0..batch.num_rows()).map(|row| format!("f{file}p{page}r{row}")),
                ));
                RecordBatch::try_new(batch.schema(), vec![labels, array(page + file * 31, 1)])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        expected_files
            .push(arrow_select::concat::concat_batches(&input[0].schema(), &input).unwrap());
        fixtures.push(fixture_batches(version, lite, input, 1).await);
    }
    let mut scans = Vec::new();
    for (fragment, (_, reader)) in fixtures.iter().enumerate() {
        scans.push(
            reader
                .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 3)
                .await
                .unwrap()
                .with_row_id(fragment as u32, None)
                .unwrap()
                .with_row_address(fragment as u32)
                .unwrap(),
        );
    }
    let waits = AtomicUsize::new(0);
    let drains =
        scans
            .iter_mut()
            .zip(&expected_files)
            .enumerate()
            .map(|(fragment, (scan, expected))| {
                let budget = &budget;
                let waits = &waits;
                async move {
                    let mut delivered = 0;
                    for selection in [0..7, 11..20] {
                        if scan.position() != selection.start {
                            scan.advance_to_range(selection.clone()).unwrap();
                        }
                        let mut next = selection.start;
                        while next < selection.end {
                            waits.fetch_add(1, Ordering::Relaxed);
                            budget.wait_for_capacity(4096).await.unwrap();
                            match scan.try_next().await.unwrap() {
                                FullZipScanOutcome::Batch { rows, batch } => {
                                    assert_eq!(rows.start, next);
                                    assert!(rows.end > next && rows.end <= selection.end);
                                    assert_batch(&batch.project(&[0, 1]).unwrap(), &rows, expected);
                                    let expected_ids = arrow_array::UInt64Array::from_iter_values(
                                        rows.clone()
                                            .map(|row| (fragment as u64) * (1_u64 << 32) + row),
                                    );
                                    assert_eq!(
                                        batch[lance_core::ROW_ID].to_data(),
                                        expected_ids.to_data()
                                    );
                                    assert!(Arc::ptr_eq(
                                        &batch[lance_core::ROW_ID],
                                        &batch[lance_core::ROW_ADDR]
                                    ));
                                    delivered += batch.num_rows();
                                    next = rows.end;
                                    let alias = batch[lance_core::ROW_ID].slice(0, 1);
                                    drop(batch);
                                    assert!(budget.reserved() > 0 && budget.reserved() <= 4096);
                                    // The independent consumer owns the only credit that
                                    // can unblock another file. It drains without waiting
                                    // for that file or occupying a CPU worker.
                                    tokio::task::yield_now().await;
                                    drop(alias);
                                }
                                FullZipScanOutcome::NeedsCapacity => {
                                    // These fixtures fit a full workspace. A competing
                                    // admission may win after the wait; it must release
                                    // capacity before this caller retries. A decode
                                    // refusal with an empty pool would require replan.
                                    assert!(budget.reserved() > 0, "full workspace must fit a row");
                                }
                                FullZipScanOutcome::Finished => {
                                    panic!("selected rows are still pending")
                                }
                            }
                        }
                        assert!(matches!(
                            scan.try_next().await.unwrap(),
                            FullZipScanOutcome::Finished
                        ));
                    }
                    delivered
                }
            });
    let mut drains = Box::pin(futures::future::join_all(drains));
    assert!(drains.as_mut().now_or_never().is_none());
    assert_eq!(
        waits.load(Ordering::Relaxed),
        3,
        "all files must be waiting"
    );
    let mut cancelled = Box::pin(budget.wait_for_capacity(4096));
    assert!(cancelled.as_mut().now_or_never().is_none());
    drop(cancelled);
    assert_eq!(budget.reserved(), 4096);
    drop(blocker);
    let delivered = tokio::time::timeout(std::time::Duration::from_secs(5), drains)
        .await
        .unwrap();
    assert_eq!(delivered, [16, 16, 16]);
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

#[rstest]
#[tokio::test]
async fn merged_files_share_workspace_across_boundaries_and_cancellation(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] partial: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input[..6]).unwrap();
    // Independent physical files with unequal page boundaries. The merged cursor
    // must preserve row alignment, not zip batches with different row counts.
    let labels = [0..4, 4..15, 15..42].map(|rows| {
        expected
            .slice(rows.start, rows.len())
            .project(&[0])
            .unwrap()
    });
    let frames = [0..9, 9..21, 21..42].map(|rows| {
        expected
            .slice(rows.start, rows.len())
            .project(&[1])
            .unwrap()
    });
    let (_left_fs, left) = fixture_batches(version, lite, labels.into(), 1).await;
    let (_right_fs, right) = fixture_batches(version, !lite, frames.into(), 1).await;
    let left_stats = IoStats::new();
    let right_stats = IoStats::new();
    let left = left.with_io_stats(left_stats.recorder());
    let right = right.with_io_stats(right_stats.recorder());
    let mode = Arc::new(AtomicUsize::new(1));
    let entered = Arc::new(tokio::sync::Notify::new());
    let reads = Arc::new(AtomicUsize::new(0));
    let right = right.with_scheduler(Arc::new(LaterColumnIo {
        inner: right.core.scheduler.clone(),
        second_buffers: right.metadata().column_infos[0]
            .page_infos
            .iter()
            .flat_map(|page| {
                page.buffer_offsets_and_sizes
                    .iter()
                    .map(|(offset, len)| *offset..*offset + *len)
            })
            .collect(),
        mode: mode.clone(),
        second_reads: reads.clone(),
        entered: entered.clone(),
    }));
    let budget = AllocationBudget::new(4096);
    // Retain real payload from an unrelated file throughout cancellation/error.
    let (_peer_fs, peer_reader) = fixture_arrays(version, lite, vec![array(29, 1)]).await;
    let mut peer_scan = peer_reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), 4096, 1)
        .await
        .unwrap();
    let FullZipScanOutcome::Batch { batch, .. } = peer_scan.try_next().await.unwrap() else {
        panic!("peer must fit")
    };
    let peer = batch.column(0).slice(0, 1);
    drop((batch, peer_scan));
    let retained = budget.reserved();
    assert!(retained > 0 && retained < 4096);
    let workspace = if partial { 4096 } else { 4096 - retained };
    let a = left
        .open_admitted_fullzip_scan(3..13, budget.clone(), workspace, 8)
        .await
        .unwrap()
        .with_partial_workspace(partial);
    let b = right
        .open_admitted_fullzip_scan(3..13, budget.clone(), workspace, 8)
        .await
        .unwrap()
        .with_partial_workspace(partial);
    let mut scan = a.try_merge(b).unwrap();
    assert_eq!(left_stats.snapshot().iops, 0);
    assert_eq!(right_stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), retained);
    for selection in [3..13, 17..35] {
        if scan.position() != selection.start {
            scan.advance_to_range(selection.clone()).unwrap();
        }
        mode.store(1, Ordering::Relaxed);
        let _ = entered.notified().now_or_never();
        {
            let future = scan.try_next();
            tokio::pin!(future);
            tokio::select! {
                _ = entered.notified() => (),
                result = &mut future => panic!("later file must be gated: {result:?}"),
            }
            assert!(left_stats.snapshot().iops > 0);
            let active = budget.reserved();
            assert!(active > retained && active <= 4096);
        }
        assert_eq!(scan.position(), selection.start);
        assert_eq!(budget.reserved(), retained);
        mode.store(2, Ordering::Relaxed);
        reads.store(0, Ordering::Relaxed);
        let error = scan.try_next().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("later-column failure"));
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        assert_eq!(scan.position(), selection.start);
        assert_eq!(budget.reserved(), retained);
        mode.store(0, Ordering::Relaxed);
        let mut next = selection.start;
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert_eq!(rows.start, next);
                    assert!(rows.end > next && rows.end <= selection.end);
                    for boundary in [4, 9, 15, 21, 42] {
                        assert!(rows.end <= boundary || rows.start >= boundary);
                    }
                    assert_batch(&batch, &rows, &expected);
                    next = rows.end;
                    let alias = batch.column(1).slice(0, 1);
                    drop(batch);
                    assert!(budget.reserved() > retained && budget.reserved() <= 4096);
                    tokio::task::yield_now().await;
                    drop(alias);
                    assert_eq!(budget.reserved(), retained);
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => panic!("compound row must fit beside peer"),
            }
        }
        assert_eq!(next, selection.end);
    }
    assert_eq!(peer.to_data(), array(29, 1).slice(0, 1).to_data());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::budget("budget", "same budget")]
#[case::workspace("workspace", "workspace/partial/max_rows")]
#[case::partial("partial", "workspace/partial/max_rows")]
#[case::max_rows("max_rows", "workspace/partial/max_rows")]
#[case::position("position", "aligned physical")]
#[case::end("end", "aligned physical")]
#[case::length("length", "aligned physical")]
#[case::duplicate("duplicate", "disjoint field names")]
#[tokio::test]
async fn merged_files_reject_incompatible_cursors_before_io(
    #[case] invalid: &str,
    #[case] message: &str,
) {
    let input = batches();
    let (fs, reader) = fixture_batches(ConcreteFileVersion::V2_3, false, input.clone(), 1).await;
    let (short_fs, shorter) =
        fixture_batches(ConcreteFileVersion::V2_3, true, input[..2].to_vec(), 1).await;
    let left = project(&fs, &reader, &[0]).await;
    let right = match invalid {
        "duplicate" => project(&fs, &reader, &[0]).await,
        "length" => project(&short_fs, &shorter, &[1]).await,
        _ => project(&fs, &reader, &[1]).await,
    };
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let other_budget = if invalid == "budget" {
        AllocationBudget::new(4096)
    } else {
        budget.clone()
    };
    let a = left
        .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 4)
        .await
        .unwrap();
    let range = match invalid {
        "position" => 1..7,
        "end" => 0..6,
        _ => 0..7,
    };
    let b = right
        .open_admitted_fullzip_scan(
            range,
            other_budget.clone(),
            if invalid == "workspace" { 2048 } else { 4096 },
            if invalid == "max_rows" { 8 } else { 4 },
        )
        .await
        .unwrap()
        .with_partial_workspace(invalid == "partial");
    let error = a.try_merge(b).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(other_budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn deleted_rows_preserve_physical_selection_and_compound_retry(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] bitmap: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input[..5]).unwrap();
    let labels = input[..5]
        .iter()
        .map(|batch| batch.project(&[0]).unwrap())
        .collect();
    let frames = input[..5]
        .iter()
        .map(|batch| batch.project(&[1]).unwrap())
        .collect();
    let (_left_fs, left) = fixture_batches(version, lite, labels, 1).await;
    let (_right_fs, right) = fixture_batches(version, !lite, frames, 1).await;
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let mode = Arc::new(AtomicUsize::new(1));
    let entered = Arc::new(tokio::sync::Notify::new());
    let reads = Arc::new(AtomicUsize::new(0));
    let right = right.with_scheduler(Arc::new(LaterColumnIo {
        inner: right.core.scheduler.clone(),
        second_buffers: right.metadata().column_infos[0]
            .page_infos
            .iter()
            .flat_map(|page| {
                page.buffer_offsets_and_sizes
                    .iter()
                    .map(|(offset, len)| *offset..*offset + *len)
            })
            .collect(),
        mode: mode.clone(),
        second_reads: reads.clone(),
        entered: entered.clone(),
    }));
    let offsets = [
        0, 1, 2, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 26, 27, 33, 34,
    ];
    let deleted = Arc::new(if bitmap {
        DeletionVector::Bitmap(offsets.into_iter().collect())
    } else {
        DeletionVector::Set(offsets.into_iter().collect())
    });
    let budget = AllocationBudget::new(4096);
    let a = left
        .open_admitted_fullzip_scan(0..13, budget.clone(), 4096, 8)
        .await
        .unwrap()
        .with_deleted_rows(deleted.clone())
        .unwrap();
    let b = right
        .open_admitted_fullzip_scan(0..13, budget.clone(), 4096, 8)
        .await
        .unwrap()
        .with_deleted_rows(deleted.clone())
        .unwrap();
    let mut scan = a.try_merge(b).unwrap();
    assert_eq!(stats.snapshot().iops, 0);
    let held = budget.try_reserve(4096).unwrap();
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::NeedsCapacity
    ));
    assert_eq!(
        scan.position(),
        0,
        "refusal must not commit leading deletions"
    );
    assert_eq!(stats.snapshot().iops, 0);
    drop(held);
    {
        let future = scan.try_next();
        tokio::pin!(future);
        tokio::select! {
            _ = entered.notified() => (),
            result = &mut future => panic!("later file must wait: {result:?}"),
        }
        assert!(budget.reserved() > 0 && budget.reserved() <= 4096);
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(budget.reserved(), 0);
    mode.store(2, Ordering::Relaxed);
    reads.store(0, Ordering::Relaxed);
    let error = scan.try_next().await.unwrap_err();
    assert!(matches!(error, Error::IO { .. }));
    assert!(error.to_string().contains("later-column failure"));
    assert_eq!(reads.load(Ordering::Relaxed), 1);
    assert_eq!(scan.position(), 0);
    assert_eq!(budget.reserved(), 0);
    mode.store(0, Ordering::Relaxed);
    for selection in [0..13, 14..21, 22..35] {
        if scan.position() != selection.start {
            scan.advance_to_range(selection.clone()).unwrap();
        }
        let expected_rows = selection
            .clone()
            .filter(|row| !offsets.contains(&(*row as u32)))
            .collect::<Vec<_>>();
        let mut actual_rows = Vec::new();
        let all_deleted = expected_rows.is_empty();
        let before = stats.snapshot().iops;
        // A completely deleted interval must finish even if all credit is held.
        let held = all_deleted.then(|| budget.try_reserve(4096).unwrap());
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert!(!all_deleted);
                    assert!(rows.start >= selection.start && rows.end <= selection.end);
                    assert_batch(&batch, &rows, &expected);
                    actual_rows.extend(rows.clone());
                    assert_eq!(scan.position(), rows.end);
                    let alias = batch.column(1).slice(0, 1);
                    drop(batch);
                    assert!(budget.reserved() > 0 && budget.reserved() <= 4096);
                    let before = stats.snapshot().iops;
                    if rows.end < selection.end
                        && (rows.end..selection.end).any(|row| !offsets.contains(&(row as u32)))
                    {
                        assert!(matches!(
                            scan.try_next().await.unwrap(),
                            FullZipScanOutcome::NeedsCapacity
                        ));
                        assert_eq!(scan.position(), rows.end);
                        assert_eq!(stats.snapshot().iops, before);
                    }
                    drop(alias);
                    assert_eq!(budget.reserved(), 0);
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => panic!("live row must fit after output drain"),
            }
        }
        assert_eq!(actual_rows, expected_rows);
        assert_eq!(scan.position(), selection.end);
        if all_deleted {
            assert_eq!(stats.snapshot().iops, before);
            assert_eq!(budget.reserved(), 4096);
        }
        drop(held);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::outside("outside", "offsets below")]
#[case::after_attempt("after_attempt", "before the first read")]
#[case::different_snapshot("different_snapshot", "same shared deletion")]
#[case::missing_snapshot("missing_snapshot", "same shared deletion")]
#[tokio::test]
async fn deleted_rows_reject_invalid_or_unaligned_snapshots(
    #[case] invalid: &str,
    #[case] message: &str,
) {
    let (fs, reader) = fixture_batches(ConcreteFileVersion::V2_3, false, batches(), 1).await;
    let left = project(&fs, &reader, &[0]).await;
    let right = project(&fs, &reader, &[1]).await;
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let deleted = Arc::new([0_u32, 2].into_iter().collect::<DeletionVector>());
    let mut a = left
        .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 4)
        .await
        .unwrap();
    let error = match invalid {
        "outside" => a
            .with_deleted_rows(Arc::new([reader.num_rows() as u32].into_iter().collect()))
            .unwrap_err(),
        "after_attempt" => {
            assert!(matches!(
                a.try_next().await.unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            assert_eq!(a.position(), 0);
            a.with_deleted_rows(deleted).unwrap_err()
        }
        _ => {
            let a = a.with_deleted_rows(deleted.clone()).unwrap();
            let mut b = right
                .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 4)
                .await
                .unwrap();
            if invalid == "different_snapshot" {
                // Even equal values cannot establish that both callers supplied
                // the same immutable fragment snapshot.
                b = b
                    .with_deleted_rows(Arc::new(deleted.as_ref().clone()))
                    .unwrap();
            }
            a.try_merge(b).unwrap_err()
        }
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn logical_ranges_preserve_live_counts_and_physical_rows(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] bitmap: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input[..5]).unwrap();
    let labels = input[..5]
        .iter()
        .map(|batch| batch.project(&[0]).unwrap())
        .collect();
    let frames = input[..5]
        .iter()
        .map(|batch| batch.project(&[1]).unwrap())
        .collect();
    let (_left_fs, left) = fixture_batches(version, lite, labels, 1).await;
    let (_right_fs, right) = fixture_batches(version, !lite, frames, 1).await;
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let offsets = [
        0, 1, 2, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 26, 27, 33, 34,
    ];
    let deleted = Arc::new(if bitmap {
        DeletionVector::Bitmap(offsets.into_iter().collect())
    } else {
        DeletionVector::Set(offsets.into_iter().collect())
    });
    let live = (0..expected.num_rows() as u64)
        .filter(|row| !offsets.contains(&(*row as u32)))
        .collect::<Vec<_>>();
    let count = live.len() as u64;
    let budget = AllocationBudget::new(4096);
    let a = left
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 8)
        .await
        .unwrap();
    let b = right
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 8)
        .await
        .unwrap();
    let mut scan = a.try_merge(b).unwrap().with_deleted_rows(deleted).unwrap();
    for selection in [0..0, 1..4, 4..6, 8..count, count..count] {
        let reads = stats.snapshot().iops;
        scan.advance_to_live_range(selection.clone()).unwrap();
        assert_eq!(stats.snapshot().iops, reads, "mapping uses metadata only");
        let expected_rows = &live[selection.start as usize..selection.end as usize];
        let initial = scan.position();
        let mut actual_rows = Vec::new();
        if !selection.is_empty() {
            let error = scan
                .advance_to_live_range(selection.end..selection.end)
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("must be drained"));
            assert_eq!(scan.position(), initial);
        }
        let held = budget.try_reserve(4096).unwrap();
        let outcome = scan.try_next().await.unwrap();
        if selection.is_empty() {
            assert!(matches!(outcome, FullZipScanOutcome::Finished));
            assert_eq!(stats.snapshot().iops, reads);
            drop(held);
            continue;
        }
        assert!(matches!(outcome, FullZipScanOutcome::NeedsCapacity));
        assert_eq!(scan.position(), initial);
        assert_eq!(stats.snapshot().iops, reads);
        drop(held);
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert_batch(&batch, &rows, &expected);
                    actual_rows.extend(rows.clone());
                    assert_eq!(scan.position(), rows.end);
                    let alias = batch.column(1).slice(0, 1);
                    drop(batch);
                    assert!(budget.reserved() > 0 && budget.reserved() <= 4096);
                    if actual_rows.len() < expected_rows.len() {
                        let reads = stats.snapshot().iops;
                        assert!(matches!(
                            scan.try_next().await.unwrap(),
                            FullZipScanOutcome::NeedsCapacity
                        ));
                        assert_eq!(scan.position(), rows.end);
                        assert_eq!(stats.snapshot().iops, reads);
                    }
                    drop(alias);
                    assert_eq!(budget.reserved(), 0);
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => {
                    panic!("selected live row must fit after drain")
                }
            }
        }
        assert_eq!(actual_rows, expected_rows);
        assert_eq!(actual_rows.len() as u64, selection.end - selection.start);
        // Reject a backward request without disturbing the drained cursor.
        let position = scan.position();
        let error = scan.advance_to_live_range(0..1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("start at or after"));
        assert_eq!(scan.position(), position);
    }
    assert_eq!(scan.position(), expected.num_rows() as u64);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn logical_empty_eof_and_invalid_ranges_preserve_snapshot(
    #[values(false, true)] all_deleted: bool,
) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_3, false, 1, 2).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let deleted = Arc::new(if all_deleted {
        (0..reader.num_rows() as u32).collect::<DeletionVector>()
    } else {
        DeletionVector::NoDeletions
    });
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 4)
        .await
        .unwrap()
        .with_deleted_rows(deleted.clone())
        .unwrap();
    let live_count = if all_deleted { 0 } else { reader.num_rows() };
    for invalid in [0..live_count + 1, Range { start: 1, end: 0 }] {
        let error = scan.advance_to_live_range(invalid).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("live rows"));
        assert_eq!(scan.position(), 0);
        assert_eq!(stats.snapshot().iops, 0);
    }
    // Failed mapping must not freeze the snapshot; successful mapping must.
    let mut scan = scan.with_deleted_rows(deleted.clone()).unwrap();
    scan.advance_to_live_range(live_count..live_count).unwrap();
    assert_eq!(scan.position(), reader.num_rows());
    assert_eq!(stats.snapshot().iops, 0);
    let error = scan.with_deleted_rows(deleted).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("logical selection"));
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
