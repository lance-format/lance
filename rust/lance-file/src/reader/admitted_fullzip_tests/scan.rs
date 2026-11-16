// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use futures::{FutureExt, future::BoxFuture};
use lance_core::{Result, utils::allocation::ReservedBuffer};

use super::*;
use crate::reader::{FullZipScanOutcome, ReaderProjection};

#[derive(Debug)]
pub(super) struct RangeIo {
    pub(super) inner: Arc<dyn EncodingsIo>,
    pub(super) reads: Arc<Mutex<Vec<Range<u64>>>>,
}

impl EncodingsIo for RangeIo {
    fn submit_request(
        &self,
        _ranges: Vec<Range<u64>>,
        _priority: u64,
    ) -> BoxFuture<'static, Result<Vec<bytes::Bytes>>> {
        async { Err(Error::io("unexpected unadmitted payload read")) }.boxed()
    }

    fn submit_single_into(
        &self,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        self.reads.lock().unwrap().push(range.clone());
        self.inner.submit_single_into(range, priority, output)
    }
}

#[rstest]
#[tokio::test]
async fn admitted_scan_advances_without_reading_skipped_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
) {
    // Repeated binary values use dense FullZip in v2.3. All-null values instead
    // produce an unsupported layout in both versions, independently of the
    // cursor's validation: structural ConstantLayout or planar sparse structure.
    let unsupported: ArrayRef = Arc::new(LargeBinaryArray::new_null(8));
    let (_fs, reader) = fixture_arrays(
        version,
        lite,
        vec![array(0, 0), array(1, 0), array(2, 0), unsupported],
    )
    .await;
    match &reader.metadata().column_infos[0].page_infos[3].encoding {
        PageEncoding::Structural(layout) => assert!(matches!(
            &layout.layout,
            Some(lance_encoding::format::metadata::page_layout::Layout::ConstantLayout(_))
        )),
        PageEncoding::Planar(layout) => assert!(layout.sparse().is_some()),
        PageEncoding::Legacy(_) => panic!("fixture must use a current-format layout"),
    }
    let skipped_buffers = reader.metadata().column_infos[0].page_infos[1]
        .buffer_offsets_and_sizes
        .iter()
        .map(|(offset, len)| *offset..*offset + *len)
        .collect::<Vec<_>>();
    assert!(
        skipped_buffers.iter().any(|range| !range.is_empty()),
        "skipped page must have physical input to exclude"
    );
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..4, budget.clone(), 8192, 8)
        .await
        .unwrap();
    assert!(reads.lock().unwrap().is_empty());
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("first range must fit")
    };
    assert_eq!(rows, 0..4);
    assert_eq!(batch.column(0).to_data(), array(0, 0).slice(0, 4).to_data());
    let held = batch.column(0).slice(0, 1);
    drop(batch);
    let held_bytes = budget.reserved();
    let count = reads.lock().unwrap().len();
    assert!(held_bytes > 0);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    // A bad next selection must not alter a drained cursor or retained output.
    let error = scan.advance_to_range(24..25).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("FullZip"));
    assert_eq!(scan.position(), 4);
    assert_eq!(budget.reserved(), held_bytes);
    scan.advance_to_range(17..22).unwrap();
    assert_eq!(scan.position(), 17);
    assert_eq!(reads.lock().unwrap().len(), count);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::NeedsCapacity
    ));
    assert_eq!(scan.position(), 17);
    assert_eq!(reads.lock().unwrap().len(), count);
    drop(held);
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("second selected range must fit after drain")
    };
    assert_eq!(rows, 17..22);
    assert_eq!(batch.column(0).to_data(), array(2, 0).slice(1, 5).to_data());
    drop(batch);
    assert_eq!(budget.reserved(), 0);
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    for read in reads.lock().unwrap().iter() {
        assert!(
            skipped_buffers
                .iter()
                .all(|gap| read.end <= gap.start || gap.end <= read.start)
        );
    }
}

#[tokio::test]
async fn admitted_scan_rejects_invalid_advance_without_changing_selection() {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, false, 0, 1).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(8192);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), 8192, 1)
        .await
        .unwrap();
    let error = scan.advance_to_range(3..4).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("must be drained"));
    assert_eq!(scan.position(), 0);
    assert_eq!(stats.snapshot().iops, 0);
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("invalid advance must leave the original row selected")
    };
    assert_eq!(rows, 0..1);
    assert_eq!(batch.column(0).to_data(), array(0, 0).slice(0, 1).to_data());
    drop(batch);
    let reads = stats.snapshot().iops;
    for rows in [0..2, Range { start: 4, end: 3 }, 7..9, u64::MAX..u64::MAX] {
        let error = scan.advance_to_range(rows).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("FullZip next range"));
        assert_eq!(scan.position(), 1);
    }
    for empty in [1, 5, 8] {
        scan.advance_to_range(empty..empty).unwrap();
        assert_eq!(scan.position(), empty);
        assert!(matches!(
            scan.try_next().await.unwrap(),
            FullZipScanOutcome::Finished
        ));
    }
    assert_eq!(stats.snapshot().iops, reads);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn admitted_scan_crosses_pages_without_read_ahead(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(0, 1, 2)] depth: usize,
) {
    let (_fs, reader) = fixture(version, lite, depth, 17).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(8192);
    let end = reader.num_rows() - 1;
    let mut scan = reader
        .open_admitted_fullzip_scan(1..end, budget.clone(), 8192, 3)
        .await
        .unwrap();
    assert_eq!(stats.snapshot().iops, 0, "open must not read any payload");
    assert_eq!(scan.position(), 1);
    drop(scan.try_next());
    assert_eq!(stats.snapshot().iops, 0, "unpolled demand must not read");
    let page_rows = array(0, depth).len() as u64;
    let mut next = 1;
    let mut batches = 0;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, next);
                assert!(rows.end > rows.start && rows.end - rows.start <= 3);
                assert!(rows.end <= end);
                let page = (rows.start / page_rows) as usize;
                let offset = (rows.start % page_rows) as usize;
                assert_eq!(
                    batch.column(0).to_data(),
                    array(page, depth).slice(offset, batch.num_rows()).to_data()
                );
                next = rows.end;
                assert_eq!(scan.position(), next);
                let held = batch.slice(0, 1);
                drop(batch);
                let before = stats.snapshot().iops;
                if next < end {
                    assert!(matches!(
                        scan.try_next().await.unwrap(),
                        FullZipScanOutcome::NeedsCapacity
                    ));
                    assert_eq!(scan.position(), next);
                }
                assert_eq!(
                    stats.snapshot().iops,
                    before,
                    "retained output must not cause speculative input"
                );
                drop(held);
                assert_eq!(budget.reserved(), 0);
                batches += 1;
            }
            FullZipScanOutcome::NeedsCapacity => {
                panic!("complete single operation fits after drain")
            }
            FullZipScanOutcome::Finished => break,
        }
    }
    assert_eq!(next, end);
    assert!(batches > 17);
    let before = stats.snapshot().iops;
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    assert_eq!(stats.snapshot().iops, before);
}

#[tokio::test]
async fn admitted_scan_empty_invalid_and_unsupported_selection() {
    let constant: ArrayRef = Arc::new(LargeBinaryArray::from_vec(vec![b"same"; 8]));
    let (_fs, reader) = fixture_arrays(
        ConcreteFileVersion::V2_2,
        false,
        vec![array(0, 0), constant],
    )
    .await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(8192);
    for (rows, bytes, max_rows) in [
        (0..17, 8192, 3),
        (Range { start: 9, end: 8 }, 8192, 3),
        (0..1, 0, 3),
        (0..1, 8192, 0),
    ] {
        let error = reader
            .open_admitted_fullzip_scan(rows, budget.clone(), bytes, max_rows)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("FullZip"));
    }
    let error = reader
        .open_admitted_fullzip_scan(0..16, budget.clone(), 8192, 3)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("FullZip"));
    assert_eq!(
        stats.snapshot().iops,
        0,
        "later unsupported page must fail before any output"
    );
    for position in [0, 1, 8, 9, 16] {
        let mut empty = reader
            .open_admitted_fullzip_scan(position..position, budget.clone(), 8192, 3)
            .await
            .unwrap();
        assert!(matches!(
            empty.try_next().await.unwrap(),
            FullZipScanOutcome::Finished
        ));
        assert_eq!(empty.position(), position);
    }
    assert_eq!(stats.snapshot().iops, 0);
    // An unsupported page outside the selection is irrelevant to this read.
    let mut valid = reader
        .open_admitted_fullzip_scan(0..8, budget.clone(), 8192, 8)
        .await
        .unwrap();
    let FullZipScanOutcome::Batch { rows, batch } = valid.try_next().await.unwrap() else {
        panic!("expected full first page")
    };
    assert_eq!(rows, 0..8);
    assert_eq!(batch.column(0).to_data(), array(0, 0).to_data());
    drop(batch);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn admitted_scan_respects_nonzero_projection_and_schema(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("other", DataType::LargeBinary, true),
        Field::new("payload", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_FULLZIP.to_string(),
        )])),
    ]));
    let batches = (0..2)
        .map(|page| {
            // Deliberately unsupported outside the selected projection.
            let other = Arc::new(LargeBinaryArray::from_vec(vec![b"constant"; 8]));
            RecordBatch::try_new(schema.clone(), vec![other, array(page, 0)]).unwrap()
        })
        .collect();
    let (fs, reader) = fixture_batches(version, false, batches, 1).await;
    let budget = AllocationBudget::new(8192);
    let error = reader
        .open_admitted_fullzip_scan(0..16, budget.clone(), 8192, 8)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("dense FullZip page"));
    let mut schema = reader.core.base_projection.schema.as_ref().clone();
    schema.fields.remove(0);
    schema.fields[0].name = "renamed".to_string();
    let selected = FileReader::try_open_with_file_metadata(
        reader.core.scheduler.clone(),
        fs.tmp_path.as_ref().clone(),
        Some(ReaderProjection {
            schema: Arc::new(schema),
            column_indices: vec![1],
        }),
        Arc::<DecoderPlugins>::default(),
        reader.metadata().clone(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    let mut scan = selected
        .open_admitted_fullzip_scan(7..10, budget.clone(), 8192, 8)
        .await
        .unwrap();
    for (rows, expected) in [
        (7..8, array(0, 0).slice(7, 1)),
        (8..10, array(1, 0).slice(0, 2)),
    ] {
        let FullZipScanOutcome::Batch {
            rows: actual,
            batch,
        } = scan.try_next().await.unwrap()
        else {
            panic!("expected projected batch")
        };
        assert_eq!(actual, rows);
        assert_eq!(batch.schema().field(0).name(), "renamed");
        assert_eq!(batch.column(0).to_data(), expected.to_data());
        drop(batch);
        assert_eq!(budget.reserved(), 0);
    }
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    let mut structural = selected.clone();
    structural.core.options.decoder_config.list_structure = Arc::new(
        std::collections::BTreeMap::from([(1, lance_encoding::decoder::ListReadMode::Length)]),
    );
    let error = structural
        .open_admitted_fullzip_scan(0..1, budget, 8192, 1)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("full-value reads"));
}

#[rstest]
#[tokio::test]
async fn admitted_scan_batch_error_keeps_cursor_and_refunds(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
) {
    let (fs, reader) = fixture(version, false, 0, 1).await;
    let mut projection = reader.base_projection().clone();
    Arc::make_mut(&mut projection.schema).fields[0].nullable = false;
    let selected = FileReader::try_open_with_file_metadata(
        reader.core.scheduler.clone(),
        fs.tmp_path.as_ref().clone(),
        Some(projection),
        Arc::<DecoderPlugins>::default(),
        reader.metadata().clone(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    let budget = AllocationBudget::new(8192);
    let mut scan = selected
        .open_admitted_fullzip_scan(0..4, budget.clone(), 8192, 4)
        .await
        .unwrap();
    // The typed array succeeds; only the final batch rejects the null in row 1.
    // Neither the initial attempt nor a retry may consume this failed prefix.
    for _ in 0..2 {
        let error = scan.try_next().await.unwrap_err();
        assert!(matches!(error, Error::Arrow { .. }));
        assert!(error.to_string().contains("non-nullable"));
        assert_eq!(scan.position(), 0);
        assert_eq!(budget.reserved(), 0);
    }
}

#[derive(Debug)]
pub(super) struct GateIo {
    pub(super) inner: Arc<dyn EncodingsIo>,
    pub(super) block: Arc<AtomicBool>,
    pub(super) fail: Arc<AtomicBool>,
    pub(super) entered: Arc<tokio::sync::Notify>,
}

impl EncodingsIo for GateIo {
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
        let inner = self.inner.clone();
        let block = self.block.clone();
        let fail = self.fail.clone();
        let entered = self.entered.clone();
        async move {
            entered.notify_one();
            if block.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            if fail.load(Ordering::Relaxed) {
                return Err(Error::io("injected admitted scan read failure"));
            }
            inner.submit_single_into(range, priority, output).await
        }
        .boxed()
    }
}

#[rstest]
#[tokio::test]
async fn admitted_scan_cancellation_and_error_do_not_consume_rows(
    #[values(false, true)] lite: bool,
) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 1, 2).await;
    let block = Arc::new(AtomicBool::new(true));
    let fail = Arc::new(AtomicBool::new(false));
    let entered = Arc::new(tokio::sync::Notify::new());
    let reader = reader.with_scheduler(Arc::new(GateIo {
        inner: reader.core.scheduler.clone(),
        block: block.clone(),
        fail: fail.clone(),
        entered: entered.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut scan = reader
        .open_admitted_fullzip_scan(1..10, budget.clone(), 8192, 3)
        .await
        .unwrap();
    {
        let future = scan.try_next();
        tokio::pin!(future);
        tokio::select! {
            _ = entered.notified() => (),
            result = &mut future => panic!("gated read completed: {result:?}"),
        }
        assert!(budget.reserved() > 0);
    }
    assert_eq!(scan.position(), 1);
    assert_eq!(budget.reserved(), 0);
    block.store(false, Ordering::Relaxed);
    fail.store(true, Ordering::Relaxed);
    let error = scan.try_next().await.unwrap_err();
    assert!(matches!(error, Error::IO { .. }));
    assert!(
        error
            .to_string()
            .contains("injected admitted scan read failure")
    );
    assert_eq!(scan.position(), 1);
    assert_eq!(budget.reserved(), 0);
    fail.store(false, Ordering::Relaxed);
    let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
        panic!("retry must return same rows")
    };
    assert_eq!(rows, 1..4);
    assert_eq!(batch.column(0).to_data(), array(0, 1).slice(1, 3).to_data());
    drop((batch, scan));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn admitted_scan_finite_peers_share_one_budget(#[values(false, true)] lite: bool) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 2, 3).await;
    let budget = AllocationBudget::new(8192);
    let mut a = reader
        .open_admitted_fullzip_scan(0..4, budget.clone(), 4096, 4)
        .await
        .unwrap();
    let mut b = reader
        .open_admitted_fullzip_scan(4..8, budget.clone(), 4096, 4)
        .await
        .unwrap();
    let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        futures::join!(a.try_next(), b.try_next())
    })
    .await
    .unwrap();
    for (result, expected, start) in [(a, array(0, 2), 0), (b, array(1, 2), 4)] {
        let FullZipScanOutcome::Batch { rows, batch } = result.unwrap() else {
            panic!("peer must make progress")
        };
        assert_eq!(rows.start, start);
        assert_eq!(
            batch.column(0).to_data(),
            expected.slice(0, batch.num_rows()).to_data()
        );
        assert!(budget.reserved() <= 8192);
    }
    assert_eq!(budget.reserved(), 0);
}
