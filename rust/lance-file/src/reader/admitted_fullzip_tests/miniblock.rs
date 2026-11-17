// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::scan::RangeIo;
use super::*;
use crate::reader::FullZipScanOutcome;
use arrow_array::{BinaryArray, LargeStringArray, UInt32Array};
use lance_encoding::constants::STRUCTURAL_ENCODING_MINIBLOCK;
use std::sync::Mutex;

fn nested_values(deep: bool) -> ArrayRef {
    use arrow_array::builder::{LargeBinaryBuilder, LargeListBuilder, ListBuilder};

    fn append(builder: &mut LargeListBuilder<LargeBinaryBuilder>, row: usize) {
        if row % 7 == 3 {
            builder.append(false);
            return;
        }
        let count = if row.is_multiple_of(11) {
            2048
        } else if row % 5 == 2 {
            0
        } else {
            row % 9 + 1
        };
        for item in 0..count {
            if item % 17 == 4 {
                builder.values().append_null();
            } else {
                builder
                    .values()
                    .append_value(format!("row-{row:08x}-item-{item:08x}"));
            }
        }
        builder.append(true);
    }

    let mut inner = LargeListBuilder::new(LargeBinaryBuilder::new());
    if deep {
        let mut outer = ListBuilder::new(inner);
        for row in 0..24 {
            if row % 7 == 3 {
                outer.append(false);
                continue;
            }
            if row % 5 != 2 {
                append(outer.values(), row * 2);
                append(outer.values(), row * 2 + 1);
            }
            outer.append(true);
        }
        Arc::new(outer.finish())
    } else {
        for row in 0..24 {
            append(&mut inner, row);
        }
        Arc::new(inner.finish())
    }
}

#[rstest]
#[tokio::test]
async fn actual_nested_miniblock_scan_and_take(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] deep: bool,
) {
    let expected = nested_values(deep);
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", expected.data_type().clone(), true).with_metadata(HashMap::from([(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
        )])),
    ]));
    let batches = (0..2)
        .map(|i| RecordBatch::try_new(schema.clone(), vec![expected.slice(i * 12, 12)]).unwrap())
        .collect();
    let (_fs, reader) = fixture_batches(version, lite, batches, 1).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(pages.len(), 2);
    let mut has_continuation = false;
    for page in pages.iter() {
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => layout.clone(),
            PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
            _ => panic!("dense writer fixture"),
        };
        let Some(Layout::MiniBlockLayout(layout)) = layout.layout else {
            panic!("stored MiniBlock required");
        };
        assert_eq!(layout.repetition_index_depth, 1);
        assert!(layout.dictionary.is_none());
        let (start, len) = page.buffer_offsets_and_sizes[2];
        let index = reader
            .core
            .scheduler
            .submit_request(vec![start..start + len], 0)
            .await
            .unwrap();
        has_continuation |= index[0]
            .chunks_exact(16)
            .any(|pair| u64::from_le_bytes(pair[8..].try_into().unwrap()) > 0);
    }
    assert!(
        has_continuation,
        "must exercise an actual list crossing chunk boundaries"
    );
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    const WORKSPACE: usize = 1024 * 1024;
    let budget = AllocationBudget::new(WORKSPACE);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..24, budget.clone(), WORKSPACE, 5)
        .await
        .unwrap();
    assert!(reads.lock().unwrap().is_empty());
    let mut position = 0;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, position);
                assert_eq!(
                    batch.column(0).to_data(),
                    expected
                        .slice(position as usize, batch.num_rows())
                        .to_data()
                );
                position = rows.end;
                assert!(budget.reserved() <= WORKSPACE);
                drop(batch);
                assert_eq!(budget.reserved(), 0);
            }
            FullZipScanOutcome::Finished => break,
            FullZipScanOutcome::NeedsCapacity => panic!("fixture prefix must fit"),
        }
    }
    assert_eq!(position, 24);
    let indices = [0, 3, 11, 12, 23];
    let want = arrow_select::take::take(
        expected.as_ref(),
        &UInt32Array::from(indices.to_vec()),
        None,
    )
    .unwrap();
    let mut work = budget.try_reserve(WORKSPACE).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let batch = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(batch.column(0).to_data(), want.to_data());
    let alias = batch.column(0).to_data().buffers()[0].slice_with_length(0, 1);
    drop((work, batch));
    assert!(budget.reserved() > 1);
    assert!(budget.try_reserve(WORKSPACE).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(WORKSPACE).is_some());

    // Both operations admit before dispatch and finish with one CPU worker in CI.
    let shared = AllocationBudget::new(2 * WORKSPACE);
    let mut first = reader
        .open_admitted_fullzip_scan(0..1, shared.clone(), WORKSPACE, 1)
        .await
        .unwrap();
    let mut second = reader
        .open_admitted_fullzip_scan(0..1, shared.clone(), WORKSPACE, 1)
        .await
        .unwrap();
    let (first, second) = tokio::join!(first.try_next(), second.try_next());
    for result in [first, second] {
        let FullZipScanOutcome::Batch { batch, .. } = result.unwrap() else {
            panic!("admitted row must finish");
        };
        assert_eq!(batch.column(0).to_data(), expected.slice(0, 1).to_data());
    }
    assert_eq!(shared.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn actual_flat_miniblock_scan_and_take(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    ty: DataType,
) {
    let values = (0..2048)
        .map(|i| format!("row-{i:08x}-{}", "abcdefghijk".repeat(i % 7 + 1)))
        .collect::<Vec<_>>();
    let expected: ArrayRef = match ty {
        DataType::Binary => Arc::new(BinaryArray::from_iter_values(
            values.iter().map(|s| s.as_bytes()),
        )),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter_values(
            values.iter().map(|s| s.as_bytes()),
        )),
        DataType::Utf8 => Arc::new(StringArray::from_iter_values(&values)),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from_iter_values(&values)),
        _ => unreachable!(),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", ty, false).with_metadata(HashMap::from([(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_MINIBLOCK.to_string(),
        )])),
    ]));
    let batches = (0..2)
        .map(|i| {
            RecordBatch::try_new(schema.clone(), vec![expected.slice(i * 1024, 1024)]).unwrap()
        })
        .collect();
    let (_fs, reader) = fixture_batches(version, lite, batches, 1).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(pages.len(), 2);
    for page in pages.iter() {
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => layout.clone(),
            PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
            _ => panic!("dense writer fixture"),
        };
        let Some(Layout::MiniBlockLayout(layout)) = layout.layout else {
            panic!("fixture must store MiniBlock");
        };
        assert_eq!(layout.repetition_index_depth, 0);
        assert!(layout.dictionary.is_none());
        assert!(
            page.buffer_offsets_and_sizes[0].1 > if layout.has_large_chunk { 4 } else { 2 },
            "must cross actual chunks"
        );
    }
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut scan = reader
        .open_admitted_fullzip_scan(13..2033, budget.clone(), 65536, 37)
        .await
        .unwrap();
    assert!(reads.lock().unwrap().is_empty());
    let mut position = 13;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, position);
                assert_eq!(
                    batch.column(0).to_data(),
                    expected
                        .slice(position as usize, batch.num_rows())
                        .to_data()
                );
                position = rows.end;
                assert!(budget.reserved() <= 65536);
                drop(batch);
                assert_eq!(budget.reserved(), 0);
            }
            FullZipScanOutcome::Finished => break,
            FullZipScanOutcome::NeedsCapacity => panic!("small selected prefix must fit"),
        }
    }
    assert_eq!(position, 2033);
    let indices = [0, 1023, 1024, 2047];
    let want = arrow_select::take::take(
        expected.as_ref(),
        &UInt32Array::from(indices.to_vec()),
        None,
    )
    .unwrap();
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let batch = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(batch.column(0).to_data(), want.to_data());
    let alias = batch.column(0).to_data().buffers()[0].slice_with_length(0, 1);
    drop((work, batch));
    assert!(budget.reserved() > 1);
    assert!(budget.try_reserve(65536).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(65536).is_some());
}
