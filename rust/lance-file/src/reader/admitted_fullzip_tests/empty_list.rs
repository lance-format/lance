// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::reader::FullZipScanOutcome;
use arrow_array::{UInt32Array, new_empty_array};
use lance_encoding::format::pb23;

fn empty_lists(large: bool, depth: usize, leaf: DataType, valid: Vec<bool>) -> ArrayRef {
    let child_type = (1..depth).fold(leaf, |child, _| {
        DataType::List(Arc::new(Field::new("item", child, false)))
    });
    let field = Arc::new(Field::new("item", child_type.clone(), false));
    let child = new_empty_array(&child_type);
    if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(vec![0_i64; valid.len() + 1].into()),
            child,
            Some(NullBuffer::from(valid)),
        ))
    } else {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(vec![0_i32; valid.len() + 1].into()),
            child,
            Some(NullBuffer::from(valid)),
        ))
    }
}

#[rstest]
#[tokio::test]
async fn current_writer_empty_lists_drain_without_payload_reads(
    #[values(false, true)] lite: bool,
    #[values(false, true)] large: bool,
    #[values(1, 3)] depth: usize,
    #[values(DataType::LargeBinary, DataType::Utf8)] leaf: DataType,
) {
    let source = empty_lists(
        large,
        depth,
        leaf,
        vec![true, false, true, true, false, true, true, false, true],
    );
    let (_fs, reader) = fixture_arrays(ConcreteFileVersion::V2_3, lite, vec![source.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    assert!(page.buffer_offsets_and_sizes.is_empty());
    let PageEncoding::Planar(layout) = &page.encoding else {
        panic!("expected V2_3 Planar page")
    };
    assert!(matches!(
        layout.values.as_ref().unwrap().codec,
        Some(pb23::value_plane::Codec::None(_))
    ));
    let Some(pb23::sparse_structural_layer::Layer::List(list)) =
        &layout.sparse().unwrap().layers[0].layer
    else {
        panic!("expected List structure")
    };
    assert_eq!(list.num_child_slots, 0);
    let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(super::scan::RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(256);
    let mut scan = reader
        .open_admitted_fullzip_scan(1..8, budget.clone(), 256, 3)
        .await
        .unwrap();
    for (start, end) in [(1, 4), (4, 7), (7, 8)] {
        let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
            panic!("expected empty/null List batch")
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
    assert!(reads.lock().unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn sparse_take_composes_empty_and_nonempty_lists(#[values(false, true)] lite: bool) {
    let populated = array(0, 1);
    let DataType::List(field) = populated.data_type() else {
        unreachable!()
    };
    let empty: ArrayRef = Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0_i32; 4].into()),
        new_empty_array(field.data_type()),
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let pages = [populated.clone(), empty, populated];
    let joined =
        arrow_select::concat::concat(&pages.iter().map(AsRef::as_ref).collect::<Vec<_>>()).unwrap();
    let indices = vec![0, 6, 7, 8, 9, 10, 16];
    let expected =
        arrow_select::take::take(joined.as_ref(), &UInt32Array::from(indices.clone()), None)
            .unwrap();
    let (_fs, reader) = fixture_arrays(ConcreteFileVersion::V2_3, lite, pages.to_vec()).await;
    let budget = AllocationBudget::new(65536);
    let mut short = budget.try_reserve(1).unwrap();
    assert!(
        reader
            .prepare_admitted_fullzip_take_in(&indices, &mut short)
            .await
            .unwrap()
            .is_none()
    );
    drop(short);
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(65536).unwrap();
    let prepared = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = prepared.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    let alias = output.column(0).nulls().unwrap().clone();
    drop((output, work));
    assert!(budget.reserved() > 0);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}
