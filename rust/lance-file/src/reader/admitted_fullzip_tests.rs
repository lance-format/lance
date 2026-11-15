// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise raw page composition without opting ordinary FileReader scans into it.

use std::{collections::HashMap, ops::Range, sync::Arc};

use arrow_array::{Array, ArrayRef, LargeBinaryArray, ListArray, RecordBatch, RecordBatchIterator};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_core::{Error, cache::LanceCache, utils::allocation::AllocationBudget};
use lance_encoding::{
    BufferScheduler, EncodingsIo,
    compression::DefaultDecompressionStrategy,
    constants::{STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY},
    decoder::{DecodedPage, DecoderPlugins, FilterExpression, PageEncoding, PageInfo},
    encodings::logical::primitive::{DecodedFullZipPrefix, FullZipScheduler},
    repdef::CompositeRepDefUnraveler,
};
use lance_io::{
    ReadBatchParams,
    scheduler::{IoStats, ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use rstest::rstest;

use super::{FileReader, FileReaderOptions};
use crate::{
    testing::{FsFixture, write_lance_file},
    version::ConcreteFileVersion,
    writer::FileWriterOptions,
};

fn array(page: usize, depth: usize) -> ArrayRef {
    let values = (0..8)
        .map(|row| match row {
            1 | 6 => None,
            2 => Some(Vec::new()),
            _ => Some(
                (0..128 + row)
                    .map(|i| (page * 17 + row * 29 + i * 31) as u8)
                    .collect::<Vec<_>>(),
            ),
        })
        .collect::<Vec<_>>();
    let mut result: ArrayRef = Arc::new(LargeBinaryArray::from_iter(
        values.iter().map(|value| value.as_deref()),
    ));
    for layer in 0..depth {
        let (offsets, valid) = if layer == 0 {
            (
                vec![0_i32, 2, 2, 2, 5, 6, 6, 8],
                vec![true, true, false, true, true, false, true],
            )
        } else {
            (vec![0_i32, 2, 2, 3, 7], vec![true, true, false, true])
        };
        result = Arc::new(ListArray::new(
            Arc::new(Field::new("item", result.data_type().clone(), true)),
            OffsetBuffer::new(offsets.into()),
            result,
            Some(NullBuffer::from(valid)),
        ));
    }
    result
}

async fn fixture(
    version: ConcreteFileVersion,
    lite: bool,
    depth: usize,
    pages: usize,
) -> (FsFixture, FileReader) {
    let mut fs = FsFixture::default();
    fs.scheduler = ScanScheduler::new(
        fs.object_store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 1024,
            use_lite_scheduler: Some(lite),
        },
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", array(0, depth).data_type().clone(), true).with_metadata(
            HashMap::from([(
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            )]),
        ),
    ]));
    let batches = (0..pages)
        .map(|page| RecordBatch::try_new(schema.clone(), vec![array(page, depth)]))
        .collect::<Vec<_>>();
    write_lance_file(
        RecordBatchIterator::new(batches, schema),
        &fs,
        version,
        FileWriterOptions {
            data_cache_bytes: Some(1),
            ..Default::default()
        },
    )
    .await;
    let scheduler = fs
        .scheduler
        .open_file(&fs.tmp_path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &Arc::new(LanceCache::no_cache()),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    // Keep the temporary file alive through all dependent scheduled reads.
    (fs, reader)
}

async fn prefix(
    reader: &FileReader,
    page: &PageInfo,
    rows: Range<u64>,
    budget: &Arc<AllocationBudget>,
    workspace: usize,
) -> Option<DecodedFullZipPrefix> {
    FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {})
        .unwrap()
        .try_read_decode_prefix_in(rows, budget, workspace, |range, output| {
            reader
                .core
                .scheduler
                .submit_single_into(range, page.priority, output)
        })
        .await
        .unwrap()
}

fn materialize(page: DecodedPage, depth: usize, budget: &Arc<AllocationBudget>) -> ArrayRef {
    // Reserve remaining shared credit once; reconstruction cannot grow the budget.
    let mut work = budget.try_reserve(8192 - budget.reserved()).unwrap();
    let leaf = page
        .data
        .try_to_arrow_in(&DataType::LargeBinary, &mut work)
        .unwrap()
        .unwrap();
    let mut levels = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let valid = levels
        .try_unravel_validity_in(leaf.len(), &mut work)
        .unwrap()
        .unwrap();
    let mut result = arrow_array::make_array(leaf.into_builder().nulls(valid).build().unwrap());
    for _ in 0..depth {
        let (offsets, valid) = levels
            .try_unravel_offsets_in::<i32>(&mut work)
            .unwrap()
            .unwrap();
        result = Arc::new(ListArray::new(
            Arc::new(Field::new("item", result.data_type().clone(), true)),
            offsets,
            result,
            valid,
        ));
    }
    result
}

#[rstest]
#[tokio::test]
async fn admitted_file_prefixes_preserve_rows_and_owners(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(0, 1, 2)] depth: usize,
    #[values(3, 17)] num_pages: usize,
) {
    let (_fs, reader) = fixture(version, lite, depth, num_pages).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(
        pages.len(),
        num_pages,
        "fixture must have separate physical pages"
    );
    let budget = AllocationBudget::new(8192);
    let mut file_row = 0;
    let mut decoded_payload = 0;
    for (index, page) in pages.iter().enumerate() {
        let expected = array(index, depth);
        assert_eq!(page.num_rows as usize, expected.len());
        assert_eq!(page.priority, file_row);
        // Start inside every page, including later pages whose file row ID is nonzero.
        let mut row = 1;
        while row < page.num_rows {
            let result = prefix(&reader, page, row..page.num_rows, &budget, 8192)
                .await
                .unwrap();
            assert_eq!(result.rows.start, row);
            assert!(result.rows.end > row && result.rows.end <= page.num_rows);
            let count = (result.rows.end - row) as usize;
            let output = materialize(result.page, depth, &budget);
            assert_eq!(
                output.to_data(),
                expected.slice(row as usize, count).to_data()
            );
            let mut leaf = output.as_ref();
            for _ in 0..depth {
                leaf = leaf
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .values()
                    .as_ref();
            }
            decoded_payload += leaf
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(<[u8]>::len)
                .sum::<usize>();
            assert!(budget.reserved() > 0 && budget.reserved() <= 8192);
            let held = output.slice(0, 1);
            drop(output);
            // A last Arrow alias keeps its entire backing owner charged. Refusal
            // must happen before any new index or data read, and must not mean EOF.
            let before = stats.snapshot().iops;
            assert!(
                prefix(&reader, page, row..page.num_rows, &budget, 8192)
                    .await
                    .is_none()
            );
            assert_eq!(stats.snapshot().iops, before);
            drop(held);
            assert_eq!(budget.reserved(), 0);
            row = result.rows.end;
        }
        // Ordinary FileReader ranges keep their projection, plugin and decode path.
        let ordinary = reader
            .read_stream(
                ReadBatchParams::Range(file_row as usize..(file_row + page.num_rows) as usize),
                page.num_rows as u32,
                1,
                FilterExpression::no_filter(),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(ordinary.len(), 1);
        assert_eq!(ordinary[0].column(0).to_data(), expected.to_data());
        file_row += page.num_rows;
    }
    assert_eq!(file_row, reader.num_rows());
    if num_pages == 17 {
        assert!(
            decoded_payload > 8192,
            "cumulative output must exceed the shared budget"
        );
    }
}

#[rstest]
#[tokio::test]
async fn admitted_file_peers_share_credit_and_resume(#[values(false, true)] lite: bool) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 0, 3).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    let budget = AllocationBudget::new(8192);
    let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        futures::join!(
            prefix(&reader, &pages[0], 0..pages[0].num_rows, &budget, 4096),
            prefix(&reader, &pages[1], 0..pages[1].num_rows, &budget, 4096),
        )
    })
    .await
    .expect("finite peers must progress through scheduled dependent index/data reads");
    let a = a.unwrap();
    let b = b.unwrap();
    let a_rows = a.rows.clone();
    let b_rows = b.rows.clone();
    let a = materialize(a.page, 0, &budget);
    let b = materialize(b.page, 0, &budget);
    assert_eq!(
        a.to_data(),
        array(0, 0).slice(0, a_rows.end as usize).to_data()
    );
    assert_eq!(
        b.to_data(),
        array(1, 0).slice(0, b_rows.end as usize).to_data()
    );
    assert!(
        prefix(&reader, &pages[2], 0..1, &budget, 8192)
            .await
            .is_none()
    );
    drop((a, b));
    assert_eq!(budget.reserved(), 0);
    let next = prefix(&reader, &pages[2], 0..1, &budget, 8192)
        .await
        .unwrap();
    assert_eq!(
        materialize(next.page, 0, &budget).to_data(),
        array(2, 0).slice(0, 1).to_data()
    );
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn admitted_file_adapter_never_falls_back_to_unowned_reads() {
    let budget = AllocationBudget::new(16);
    let io = BufferScheduler::new(bytes::Bytes::from_static(b"test"));
    let buffer = budget.try_reserve(4).unwrap().allocate().unwrap();
    let error = io.submit_single_into(0..4, 0, buffer).await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("does not support admitted reads")
    );
    assert_eq!(budget.reserved(), 0);
    // The same service still supports ordinary reads.
    assert_eq!(io.submit_single(0..4, 0).await.unwrap().as_ref(), b"test");
    let page = PageInfo {
        num_rows: 1,
        priority: 0,
        encoding: PageEncoding::Structural(Default::default()),
        buffer_offsets_and_sizes: Arc::from([]),
    };
    let error =
        FullZipScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("expected a dense FullZip page"));
}
