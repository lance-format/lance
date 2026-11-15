// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise raw page composition without opting ordinary FileReader scans into it.

use std::{collections::HashMap, ops::Range, sync::Arc};

use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, LargeListArray, ListArray, RecordBatch, RecordBatchIterator,
    StringArray,
};
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

mod multi;
mod scan;

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
    fixture_arrays(
        version,
        lite,
        (0..pages).map(|page| array(page, depth)).collect(),
    )
    .await
}

async fn fixture_arrays(
    version: ConcreteFileVersion,
    lite: bool,
    arrays: Vec<ArrayRef>,
) -> (FsFixture, FileReader) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", arrays[0].data_type().clone(), true).with_metadata(HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            ),
        ])),
    ]));
    let batches = arrays
        .into_iter()
        .map(|array| RecordBatch::try_new(schema.clone(), vec![array]))
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    fixture_batches(version, lite, batches, 1).await
}

async fn fixture_batches(
    version: ConcreteFileVersion,
    lite: bool,
    batches: Vec<RecordBatch>,
    data_cache_bytes: u64,
) -> (FsFixture, FileReader) {
    let mut fs = FsFixture::default();
    fs.scheduler = ScanScheduler::new(
        fs.object_store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 1024,
            use_lite_scheduler: Some(lite),
        },
    );
    let schema = batches[0].schema();
    write_lance_file(
        RecordBatchIterator::new(batches.into_iter().map(Ok), schema),
        &fs,
        version,
        FileWriterOptions {
            data_cache_bytes: Some(data_cache_bytes),
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

#[rstest]
#[tokio::test]
async fn admitted_file_array_prefix_keeps_reconstruction_credit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(0, 1, 2)] depth: usize,
) {
    let (_fs, reader) = fixture(version, lite, depth, 3).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    let budget = AllocationBudget::new(8192);
    for (index, page) in pages.iter().enumerate() {
        let expected = array(index, depth);
        let decoder =
            FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
        let mut row = 0;
        while row < page.num_rows {
            let result = decoder
                .try_read_array_prefix_in(
                    row..page.num_rows,
                    expected.data_type(),
                    &budget,
                    8192,
                    |range, output| {
                        reader
                            .core
                            .scheduler
                            .submit_single_into(range, page.priority, output)
                    },
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(result.rows.start, row);
            assert!(result.rows.end > row);
            assert_eq!(
                result.array.to_data(),
                expected
                    .slice(row as usize, (result.rows.end - row) as usize)
                    .to_data()
            );
            row = result.rows.end;
            let held = result.array.slice(0, 1);
            drop(result);
            assert!(budget.reserved() > 0 && budget.reserved() <= 8192);
            assert!(budget.try_reserve(8192).is_none());
            drop(held);
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[rstest]
#[tokio::test]
async fn admitted_file_reconstruction_refuses_after_successful_decode(
    #[values(false, true)] lite: bool,
) {
    // Many empty/null large lists have tiny value data but large final offsets.
    // This distinguishes reconstruction refusal from the existing codec refusal.
    // Distinct visible leaves avoid the writer's earlier Constant-page selection.
    let mut offsets = vec![0_i64; 65];
    offsets[64] = 2;
    let expected: ArrayRef = Arc::new(LargeListArray::new(
        Arc::new(Field::new("item", DataType::LargeBinary, true)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(LargeBinaryArray::from_vec(vec![b"leaf", b"tail"])),
        Some(NullBuffer::from(
            (0..64)
                .map(|row| row % 3 != 0 || row == 63)
                .collect::<Vec<_>>(),
        )),
    ));
    let (_fs, reader) =
        fixture_arrays(ConcreteFileVersion::V2_2, lite, vec![expected.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    assert!(
        matches!(&page.encoding, PageEncoding::Structural(layout)
            if matches!(&layout.layout, Some(lance_encoding::format::metadata::page_layout::Layout::FullZipLayout(_)))),
        "fixture must exercise the writer's actual FullZip layout"
    );
    let budget = AllocationBudget::new(640);
    let raw = prefix(&reader, page, 0..64, &budget, 640).await.unwrap();
    assert_eq!(
        raw.rows,
        0..64,
        "fixture must first prove full raw decode fits"
    );
    drop(raw);
    assert_eq!(budget.reserved(), 0);
    let decoder = FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
    let mut row = 0;
    let mut prefixes = 0;
    while row < 64 {
        let result = decoder
            .try_read_array_prefix_in(
                row..64,
                expected.data_type(),
                &budget,
                640,
                |range, output| {
                    reader
                        .core
                        .scheduler
                        .submit_single_into(range, page.priority, output)
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.rows.start, row);
        assert!(result.rows.end > row);
        if row == 0 {
            assert!(
                result.rows.end < 64,
                "whole-array reconstruction must refuse the first candidate"
            );
        }
        assert_eq!(
            result.array.to_data(),
            expected
                .slice(row as usize, (result.rows.end - row) as usize)
                .to_data()
        );
        assert!(budget.reserved() <= 640);
        row = result.rows.end;
        drop(result);
        assert_eq!(budget.reserved(), 0);
        prefixes += 1;
    }
    assert!(prefixes > 1);
}

#[rstest]
#[tokio::test]
async fn admitted_file_array_peers_finish_without_reacquiring_credit(
    #[values(false, true)] lite: bool,
) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 2, 2).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    let a = FullZipScheduler::try_from_page(&pages[0], &DefaultDecompressionStrategy {}).unwrap();
    let b = FullZipScheduler::try_from_page(&pages[1], &DefaultDecompressionStrategy {}).unwrap();
    let expected_a = array(0, 2);
    let expected_b = array(1, 2);
    let budget = AllocationBudget::new(8192);
    let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        futures::join!(
            a.try_read_array_prefix_in(
                0..pages[0].num_rows,
                expected_a.data_type(),
                &budget,
                4096,
                |range, output| reader.core.scheduler.submit_single_into(
                    range,
                    pages[0].priority,
                    output
                )
            ),
            b.try_read_array_prefix_in(
                0..pages[1].num_rows,
                expected_b.data_type(),
                &budget,
                4096,
                |range, output| reader.core.scheduler.submit_single_into(
                    range,
                    pages[1].priority,
                    output
                )
            ),
        )
    })
    .await
    .expect("independent peers must finish within their original reservations");
    let a = a.unwrap().unwrap();
    let b = b.unwrap().unwrap();
    assert_eq!(
        a.array.to_data(),
        expected_a.slice(0, a.rows.end as usize).to_data()
    );
    assert_eq!(
        b.array.to_data(),
        expected_b.slice(0, b.rows.end as usize).to_data()
    );
    assert!(budget.reserved() <= 8192);
    drop((a, b));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn admitted_file_array_schema_errors_do_not_read_or_retry() {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, false, 1, 1).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    let decoder = FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(8192);
    let mut reads = 0;
    for data_type in [
        DataType::Int32,
        DataType::LargeBinary,
        DataType::List(Arc::new(Field::new("item", DataType::Binary, true))),
    ] {
        let error = decoder
            .try_read_array_prefix_in(0..page.num_rows, &data_type, &budget, 8192, |_, _| {
                reads += 1;
                std::future::ready(Err(Error::io("unexpected read")))
            })
            .await
            .err()
            .unwrap();
        assert!(matches!(
            error,
            Error::NotSupported { .. } | Error::InvalidInput { .. }
        ));
        assert!(error.to_string().contains("FullZip"));
        assert_eq!(budget.reserved(), 0);
        assert_eq!(reads, 0);
    }
}

#[rstest]
#[tokio::test]
async fn admitted_file_array_binary_and_string_shapes(
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    leaf_type: DataType,
    #[values(false, true)] lite: bool,
) {
    let leaf = StringArray::from(vec![Some("one"), None, Some(""), Some("你好")]);
    let leaf = arrow_cast::cast(&leaf, &leaf_type).unwrap();
    let inner: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("text", leaf_type, true)),
        OffsetBuffer::new(vec![0_i32, 2, 2, 4].into()),
        leaf,
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let expected: ArrayRef = Arc::new(LargeListArray::new(
        Arc::new(Field::new("lists", inner.data_type().clone(), true)),
        OffsetBuffer::new(vec![0_i64, 0, 1, 3].into()),
        inner,
        None,
    ));
    let (_fs, reader) =
        fixture_arrays(ConcreteFileVersion::V2_2, lite, vec![expected.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    let decoder = FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(1024);
    let result = decoder
        .try_read_array_prefix_in(
            0..3,
            expected.data_type(),
            &budget,
            1024,
            |range, output| {
                reader
                    .core
                    .scheduler
                    .submit_single_into(range, page.priority, output)
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.rows, 0..3);
    assert_eq!(result.array.to_data(), expected.to_data());
    drop(result);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn admitted_file_array_materialization_error_is_not_a_smaller_prefix() {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, false, 1, 1).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let page = &reader.metadata().column_infos[0].page_infos[0];
    let decoder = FullZipScheduler::try_from_page(page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(8192);
    let data_type = DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, false)));
    let error = decoder
        .try_read_array_prefix_in(
            0..page.num_rows,
            &data_type,
            &budget,
            8192,
            |range, output| {
                reader
                    .core
                    .scheduler
                    .submit_single_into(range, page.priority, output)
            },
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::Arrow { .. }));
    assert!(error.to_string().contains("cannot contain nulls"));
    assert_eq!(
        stats.snapshot().iops,
        2,
        "one combined index read plus payload; no smaller retry"
    );
    assert_eq!(budget.reserved(), 0);
}
