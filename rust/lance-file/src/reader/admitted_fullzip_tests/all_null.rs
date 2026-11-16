// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::reader::FullZipScanOutcome;
use arrow_array::{UInt32Array, new_null_array};

#[rstest]
#[tokio::test]
async fn null_file_drains_without_payload_reads(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    data_type: DataType,
) {
    let source = new_null_array(&data_type, 9);
    let (_fs, reader) = fixture_arrays(version, lite, vec![source.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    assert!(page.buffer_offsets_and_sizes.is_empty());
    match &page.encoding {
        PageEncoding::Structural(layout) => {
            assert!(matches!(layout.layout, Some(Layout::ConstantLayout(_))))
        }
        PageEncoding::Planar(layout) => assert!(layout.sparse().is_some()),
        PageEncoding::Legacy(_) => panic!("expected current writer"),
    }
    let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(super::scan::RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(128);
    let mut scan = reader
        .open_admitted_fullzip_scan(1..8, budget.clone(), 128, 3)
        .await
        .unwrap();
    for (start, end) in [(1, 4), (4, 7), (7, 8)] {
        let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
            panic!("expected null batch")
        };
        assert_eq!(rows, start..end);
        assert_eq!(
            batch.column(0).to_data(),
            source
                .slice(start as usize, (end - start) as usize)
                .to_data()
        );
        let alias = batch.column(0).to_data().buffers()[0].slice_with_length(0, 1);
        drop(batch);
        assert!(budget.reserved() > 1);
        if end < 8 {
            assert!(matches!(
                scan.try_next().await.unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            assert_eq!(scan.position(), end);
        }
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
    assert!(matches!(
        scan.try_next().await.unwrap(),
        FullZipScanOutcome::Finished
    ));
    assert!(reads.lock().unwrap().is_empty());
    // Output shortage cannot advance a real file cursor or silently omit rows.
    let mut short = reader
        .open_admitted_fullzip_scan(0..1, budget.clone(), 1, 1)
        .await
        .unwrap();
    assert!(matches!(
        short.try_next().await.unwrap(),
        FullZipScanOutcome::NeedsCapacity
    ));
    assert_eq!(short.position(), 0);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_combines_null_and_value_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
) {
    let pages = [
        array(0, 0),
        new_null_array(&DataType::LargeBinary, 8),
        array(2, 0),
    ];
    let joined =
        arrow_select::concat::concat(&pages.iter().map(AsRef::as_ref).collect::<Vec<_>>()).unwrap();
    let indices = vec![0, 7, 8, 10, 15, 16, 23];
    let expected =
        arrow_select::take::take(joined.as_ref(), &UInt32Array::from(indices.clone()), None)
            .unwrap();
    let (_fs, reader) = fixture_arrays(version, lite, pages.to_vec()).await;
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let prepared = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = prepared.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    let nulls = output.column(0).nulls().unwrap().clone();
    drop((output, work));
    assert!(budget.reserved() > 0);
    drop(nulls);
    assert_eq!(budget.reserved(), 0);
}
