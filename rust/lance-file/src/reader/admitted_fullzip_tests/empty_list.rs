// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::reader::FullZipScanOutcome;
use arrow_array::{UInt32Array, new_empty_array, new_null_array};
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

fn wrap_null_children(child: ArrayRef, large: bool, offsets: &[i32], valid: Vec<bool>) -> ArrayRef {
    let field = Arc::new(Field::new("item", child.data_type().clone(), true));
    if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(
                offsets
                    .iter()
                    .map(|v| i64::from(*v))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            child,
            Some(NullBuffer::from(valid)),
        ))
    } else {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(offsets.to_vec().into()),
            child,
            Some(NullBuffer::from(valid)),
        ))
    }
}

#[rstest]
#[tokio::test]
async fn current_writer_null_children_preserve_selected_structure_without_io(
    #[values(false, true)] lite: bool,
    #[values(false, true)] large: bool,
    #[values(false, true)] nested: bool,
    #[values(DataType::LargeBinary, DataType::Utf8)] leaf: DataType,
) {
    let children = if nested {
        wrap_null_children(
            new_null_array(&leaf, 6),
            !large,
            &[0, 0, 1, 1, 1, 3, 3, 3, 6, 6, 6],
            vec![
                true, true, false, true, true, false, true, true, false, true,
            ],
        )
    } else {
        new_null_array(&leaf, 10)
    };
    let source = wrap_null_children(
        children,
        large,
        &[0, 2, 2, 4, 4, 6, 6, 8, 8, 10],
        vec![true, false, true, true, true, false, true, true, true],
    );
    let (_fs, reader) = fixture_arrays(ConcreteFileVersion::V2_3, lite, vec![source.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    assert!(
        page.buffer_offsets_and_sizes.is_empty(),
        "fixture must exercise metadata-only positions/counts"
    );
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
    assert_eq!(list.num_child_slots, 10);
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
            panic!("expected List batch")
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
    // Noncontiguous ranges join under the same workspace, including a null and an empty parent.
    let indices = vec![0, 1, 3, 4, 8];
    let expected =
        arrow_select::take::take(source.as_ref(), &UInt32Array::from(indices.clone()), None)
            .unwrap();
    let take_budget = AllocationBudget::new(4096);
    let mut work = take_budget.try_reserve(4096).unwrap();
    let prepared = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = prepared.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    drop((output, work));
    assert_eq!(take_budget.reserved(), 0);
    assert!(reads.lock().unwrap().is_empty());
}

#[rstest]
#[case::counts(false, false, false)]
#[case::positions(true, false, false)]
#[case::positions_and_validity(true, true, false)]
#[case::rle_counts(false, false, true)]
#[tokio::test]
async fn current_writer_payload_counts_use_bounded_input_windows(
    #[values(false, true)] lite: bool,
    #[values(false, true)] large: bool,
    #[case] payload_positions: bool,
    #[case] payload_validity: bool,
    #[case] rle_counts: bool,
) {
    // Distinct non-monotonic counts exercise direct payload codecs; repeated
    // stretches exercise RLE. Inspect the stored descriptor below, not a hint.
    let mut offsets = vec![0_i32];
    let valid = (0..2051)
        .map(|i| !payload_validity || i % 13 != (i / 17) % 13)
        .collect::<Vec<_>>();
    for i in 0..2051 {
        let count = if !valid[i as usize] || (payload_positions && i % 7 == (i / 11) % 7) {
            0
        } else if rle_counts {
            1 + ((i / 64) * 17 % 31)
        } else {
            1 + (i * 1031 % 2053)
        };
        offsets.push(offsets.last().unwrap() + count);
    }
    let source = wrap_null_children(
        new_null_array(&DataType::Binary, *offsets.last().unwrap() as usize),
        large,
        &offsets,
        valid,
    );
    let (_fs, reader) = fixture_arrays(ConcreteFileVersion::V2_3, lite, vec![source.clone()]).await;
    let page = &reader.metadata().column_infos[0].page_infos[0];
    assert_eq!(
        page.buffer_offsets_and_sizes.len(),
        1 + usize::from(payload_positions) + usize::from(payload_validity),
        "fixture must contain payload counts and the requested position/validity streams"
    );
    if rle_counts {
        let lance_encoding::decoder::PageEncoding::Planar(layout) = &page.encoding else {
            unreachable!()
        };
        let Some(pb23::sparse_structural_layer::Layer::List(list)) =
            &layout.sparse().unwrap().layers[0].layer
        else {
            unreachable!()
        };
        assert!(
            matches!(
                list.counts.as_ref().unwrap().compression,
                Some(lance_encoding::format::metadata::compressive_encoding::Compression::Rle(_))
            ),
            "fixture must use stored RLE counts"
        );
    }
    let reads = Arc::new(std::sync::Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(super::scan::RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut scan = reader
        .open_admitted_fullzip_scan(2047..2051, budget.clone(), 65536, 2)
        .await
        .unwrap();
    for (start, end) in [(2047, 2049), (2049, 2051)] {
        let FullZipScanOutcome::Batch { rows, batch } = scan.try_next().await.unwrap() else {
            panic!("expected selected List batch")
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
        if end < 2051 {
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
    let selected = [0, 1024, 2050];
    let mut work = budget.try_reserve(65536).unwrap();
    let prepared = reader
        .prepare_admitted_fullzip_take_in(&selected, &mut work)
        .await
        .unwrap()
        .unwrap();
    let batch = prepared.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(
        batch.column(0).to_data(),
        arrow_select::take::take(source.as_ref(), &UInt32Array::from(selected.to_vec()), None)
            .unwrap()
            .to_data()
    );
    drop((batch, work));
    assert_eq!(budget.reserved(), 0);
    let reads = reads.lock().unwrap();
    assert!(reads.len() > 2);
    assert!(reads.iter().all(|range| range.end - range.start <= 8192));
}
