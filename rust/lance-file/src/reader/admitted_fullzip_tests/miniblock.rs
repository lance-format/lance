// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::scan::RangeIo;
use super::*;
use crate::reader::FullZipScanOutcome;
use arrow_array::{BinaryArray, LargeStringArray, UInt32Array};
use lance_encoding::constants::STRUCTURAL_ENCODING_MINIBLOCK;
use std::sync::Mutex;

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
