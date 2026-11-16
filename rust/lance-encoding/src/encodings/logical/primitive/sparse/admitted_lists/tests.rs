// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::{LargeListArray, ListArray, UInt32Array, new_empty_array, new_null_array};
use arrow_buffer::OffsetBuffer;
use arrow_schema::Field;
use lance_core::utils::allocation::{AllocationBudget, ReservedBuffer};
use rstest::rstest;

use super::*;
use crate::{
    compression::DefaultDecompressionStrategy,
    encodings::{logical::primitive::AdmittedBytePageScheduler, physical::range::RangeEncoder},
    format::EncodingUtils,
};

fn positions(is_null: bool, start: u64, step: u64, count: u64) -> pb23::SparseValiditySet {
    pb23::SparseValiditySet {
        meaning: if is_null {
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32
        } else {
            pb23::sparse_validity_set::Meaning::SparseValidityValidPositions as i32
        },
        num_positions: count,
        positions: Some(RangeEncoder::new(64, start, step).encoding(count).unwrap()),
    }
}

fn fixture(depth: usize, validity: pb23::SparseValiditySet) -> PageInfo {
    let mut layers = (0..depth)
        .map(|index| pb23::SparseStructuralLayer {
            layer: Some(pb23::sparse_structural_layer::Layer::List(
                pb23::SparseListLayer {
                    num_child_slots: 0,
                    num_non_empty_positions: 0,
                    non_empty_positions: Some(EncodingUtils::constant(None)),
                    counts: Some(EncodingUtils::constant(None)),
                    validity: Some(if index == 0 {
                        validity.clone()
                    } else {
                        positions(true, 0, 1, 0)
                    }),
                },
            )),
        })
        .collect::<Vec<_>>();
    layers.push(pb23::SparseStructuralLayer {
        layer: Some(pb23::sparse_structural_layer::Layer::Validity(positions(
            true, 0, 1, 0,
        ))),
    });
    PageInfo {
        num_rows: 9,
        priority: 0,
        encoding: PageEncoding::Planar(pb23::PlanarLayout {
            structure: Some(pb23::StructuralPlane {
                codec: Some(pb23::structural_plane::Codec::Sparse(
                    pb23::SparseStructure { layers },
                )),
            }),
            values: Some(pb23::ValuePlane {
                codec: Some(pb23::value_plane::Codec::None(pb23::NoValues {})),
            }),
        }),
        buffer_offsets_and_sizes: Arc::from([]),
    }
}

fn shape(leaf: DataType, depth: usize, large: bool) -> DataType {
    (0..depth).fold(leaf, |child, index| {
        let field = Arc::new(
            Field::new(format!("child{index}"), child, false)
                .with_metadata(HashMap::from([("source".into(), "empty-list".into())])),
        );
        if large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        }
    })
}

fn expected(data_type: &DataType, valid: Vec<bool>) -> ArrayRef {
    let nulls = Some(NullBuffer::from(valid.clone()));
    match data_type {
        DataType::List(field) => Arc::new(ListArray::new(
            field.clone(),
            OffsetBuffer::new(vec![0_i32; valid.len() + 1].into()),
            new_empty_array(field.data_type()),
            nulls,
        )),
        DataType::LargeList(field) => Arc::new(LargeListArray::new(
            field.clone(),
            OffsetBuffer::new(vec![0_i64; valid.len() + 1].into()),
            new_empty_array(field.data_type()),
            nulls,
        )),
        _ => unreachable!(),
    }
}

async fn unexpected_read(_: Range<u64>, _: ReservedBuffer) -> Result<ReservedBuffer> {
    Err(Error::io("empty List must not read payload"))
}

#[rstest]
#[tokio::test]
async fn empty_and_null_lists_preserve_shape_and_physical_owners(
    #[values(false, true)] large: bool,
    #[values(1, 3)] depth: usize,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    leaf: DataType,
) {
    let width = if large { 8 } else { 4 };
    let leaf_width = if matches!(leaf, DataType::Binary | DataType::Utf8) {
        4
    } else {
        8
    };
    let data_type = shape(leaf, depth, large);
    for is_null in [false, true] {
        // Empty sets, singletons and arithmetic sequences exercise both meanings.
        for selected in [vec![], vec![4], vec![2, 4, 6], (0..9).collect()] {
            let (start, step) = match selected.as_slice() {
                [] => (0, 1),
                [one] => (*one, 1),
                [first, second, ..] => (*first, second - first),
            };
            let page = AdmittedBytePageScheduler::try_from_page(
                &fixture(
                    depth,
                    positions(is_null, start, step, selected.len() as u64),
                ),
                &DefaultDecompressionStrategy::default(),
            )
            .unwrap();
            for range in [0..2, 2..7, 7..9] {
                let valid = range
                    .clone()
                    .map(|row| selected.contains(&row) != is_null)
                    .collect::<Vec<_>>();
                let len = valid.len();
                let own = (len + 1) * width + usize::from(valid.contains(&false));
                let total = own + (depth - 1) * width + leaf_width;
                let budget = AllocationBudget::new(total);
                let mut short = budget.try_reserve(total - 1).unwrap();
                let input = page
                    .prepare_array_range_in(
                        range.clone(),
                        data_type.clone(),
                        &mut short,
                        unexpected_read,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert!(input.decode_in(&mut short).unwrap().is_none());
                assert_eq!(short.len(), total - 1, "refusal must allocate nothing");
                drop(short);
                assert_eq!(budget.reserved(), 0);
                let mut work = budget.try_reserve(total).unwrap();
                let input = page
                    .prepare_array_range_in(
                        range.clone(),
                        data_type.clone(),
                        &mut work,
                        unexpected_read,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(work.len(), total, "no payload allocation during prepare");
                let output = input.decode_in(&mut work).unwrap().unwrap();
                assert_eq!(output.to_data(), expected(&data_type, valid).to_data());
                assert_eq!(work.len(), 0);
                let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
                drop((output, work));
                assert_eq!(
                    budget.reserved(),
                    own,
                    "one offset byte retains the parent bitmap too"
                );
                assert!(budget.try_reserve(total).is_none());
                drop(alias);
                assert_eq!(budget.reserved(), 0);
                assert!(budget.try_reserve(total).is_some());
            }
        }
    }
}

#[rstest]
#[case::missing_leaf(0, "missing leaf")]
#[case::nonempty_child(1, "count sum")]
#[case::missing_counts(2, "missing unsigned")]
#[case::invalid_meaning(3, "validity meaning")]
#[case::out_of_range(4, "outside domain")]
#[case::duplicate_positions(5, "strictly increasing")]
#[case::unexpected_payload(6, "unexpected buffers")]
#[case::nonempty_leaf(7, "exceeds domain")]
fn malformed_structure_does_not_become_an_empty_list(#[case] kind: u8, #[case] message: &str) {
    let mut page = fixture(2, positions(true, 2, 2, 3));
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    if kind == 0 {
        structure.layers.pop();
    }
    if kind == 7 {
        structure.layers[2].layer = Some(pb23::sparse_structural_layer::Layer::Validity(
            positions(true, 0, 1, 1),
        ));
    }
    let index = usize::from(kind == 1);
    let Some(pb23::sparse_structural_layer::Layer::List(list)) =
        structure.layers[index].layer.as_mut()
    else {
        unreachable!()
    };
    match kind {
        1 => list.num_child_slots = 1,
        2 => list.counts = None,
        3 => list.validity.as_mut().unwrap().meaning = 999,
        4 => list.validity = Some(positions(true, 9, 1, 1)),
        5 => {
            let validity = list.validity.as_mut().unwrap();
            validity.num_positions = 2;
            validity.positions = Some(EncodingUtils::constant(Some(
                2_u64.to_le_bytes().to_vec().into(),
            )));
        }
        6 => page.buffer_offsets_and_sizes = Arc::from([(0, 1)]),
        _ => (),
    }
    let error = MetadataListScheduler::try_from_page(&page).unwrap_err();
    if matches!(kind, 3 | 4 | 5 | 7) {
        assert!(matches!(error, Error::InvalidInput { .. }));
    } else {
        assert!(matches!(error, Error::CorruptFile { .. }));
    }
    assert!(error.to_string().contains(message), "{error}");
}

#[tokio::test]
async fn validates_selection_before_capacity_and_cancels_without_owners() {
    let page = AdmittedBytePageScheduler::try_from_page(
        &fixture(1, positions(true, 2, 2, 3)),
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let data_type = shape(DataType::LargeBinary, 1, false);
    for range in [0..0, 0..10, Range { start: 3, end: 2 }] {
        let error = page
            .prepare_array_range_in(range, data_type.clone(), &mut work, unexpected_read)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("selection is outside"));
    }
    for invalid in [
        DataType::Binary,
        shape(DataType::Int32, 1, false),
        shape(DataType::Binary, 2, true),
    ] {
        let error = page
            .prepare_array_range_in(0..1, invalid, &mut work, unexpected_read)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("sparse List"));
    }
    let input = page
        .prepare_array_range_in(0..1, data_type, &mut work, unexpected_read)
        .await
        .unwrap()
        .unwrap();
    drop(input);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn payload_positions_are_not_assumed_empty() {
    let mut page = fixture(1, positions(true, 2, 2, 3));
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    let Some(pb23::sparse_structural_layer::Layer::List(list)) = structure.layers[0].layer.as_mut()
    else {
        unreachable!()
    };
    list.validity.as_mut().unwrap().positions = Some(EncodingUtils::flat(64, None));
    page.buffer_offsets_and_sizes = Arc::from([(0, 24)]);
    let error = MetadataListScheduler::try_from_page(&page).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("metadata-only positions"));
}

#[test]
fn overflowing_validity_descriptor_is_rejected_before_range_arithmetic() {
    let mut validity = positions(true, 2, 2, 3);
    validity.positions = Some(EncodingUtils::range(64, u64::MAX - 1, 2));
    let error = MetadataListScheduler::try_from_page(&fixture(1, validity)).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("overflows"));
}

fn list_layer(
    domain: u64,
    children: u64,
    start: u64,
    step: u64,
    count: u64,
    counts: CompressiveEncoding,
    nulls: pb23::SparseValiditySet,
) -> pb23::SparseStructuralLayer {
    assert!(nulls.num_positions <= domain);
    pb23::SparseStructuralLayer {
        layer: Some(pb23::sparse_structural_layer::Layer::List(
            pb23::SparseListLayer {
                num_child_slots: children,
                non_empty_positions: Some(
                    RangeEncoder::new(64, start, step).encoding(count).unwrap(),
                ),
                num_non_empty_positions: count,
                counts: Some(counts),
                validity: Some(nulls),
            },
        )),
    }
}

fn populated_fixture(depth: usize, is_null: bool) -> PageInfo {
    let mut page = fixture(depth, positions(true, 0, 1, 0));
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    // Nonempty parents have two children. Null-position polarity also retains valid empty slots at 3,7.
    structure.layers[0] = list_layer(
        9,
        10,
        0,
        2,
        5,
        EncodingUtils::constant(Some(2_u64.to_le_bytes().to_vec().into())),
        if is_null {
            positions(true, 1, 4, 2)
        } else {
            positions(false, 0, 2, 5)
        },
    );
    let children = if depth == 2 {
        // Nested nonempty slots have 1,2,3 children; the two polarities exercise both empty and null gaps.
        structure.layers[1] = list_layer(
            10,
            6,
            1,
            3,
            3,
            RangeEncoder::new(64, 1, 1).encoding(3).unwrap(),
            if is_null {
                positions(true, 2, 3, 3)
            } else {
                positions(false, 1, 3, 3)
            },
        );
        6
    } else {
        10
    };
    structure.layers[depth].layer = Some(pb23::sparse_structural_layer::Layer::Validity(
        positions(is_null, 0, 1, if is_null { children } else { 0 }),
    ));
    page
}

fn wrap(child: ArrayRef, large: bool, offsets: &[i32], valid: Vec<bool>) -> ArrayRef {
    let field = Arc::new(
        Field::new("item", child.data_type().clone(), true)
            .with_metadata(HashMap::from([("semantic".into(), "null leaves".into())])),
    );
    if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(
                offsets
                    .iter()
                    .map(|n| i64::from(*n))
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

fn populated_source(leaf: &DataType, large: bool, depth: usize, is_null: bool) -> ArrayRef {
    let children = if depth == 2 {
        wrap(
            new_null_array(leaf, 6),
            !large,
            &[0, 0, 1, 1, 1, 3, 3, 3, 6, 6, 6],
            if is_null {
                vec![
                    true, true, false, true, true, false, true, true, false, true,
                ]
            } else {
                vec![
                    false, true, false, false, true, false, false, true, false, false,
                ]
            },
        )
    } else {
        new_null_array(leaf, 10)
    };
    wrap(
        children,
        large,
        &[0, 2, 2, 4, 4, 6, 6, 8, 8, 10],
        if is_null {
            vec![true, false, true, true, true, false, true, true, true]
        } else {
            vec![true, false, true, false, true, false, true, false, true]
        },
    )
}

// Public Arrow take supplies a compact independent oracle. Only live logical
// buffers matter here; Arrow allocator capacity is not our physical ownership.
fn buffer_bytes(data: &ArrayData) -> usize {
    data.buffers().iter().map(Buffer::len).sum::<usize>()
        + if data.null_count() > 0 {
            data.len().div_ceil(8)
        } else {
            0
        }
}

fn tree_bytes(data: &ArrayData) -> usize {
    buffer_bytes(data) + data.child_data().iter().map(tree_bytes).sum::<usize>()
}

#[rstest]
#[tokio::test]
async fn selected_nonempty_null_lists_fit_one_workspace(
    #[values(false, true)] large: bool,
    #[values(1, 2)] depth: usize,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    leaf: DataType,
) {
    for is_null in [false, true] {
        let source = populated_source(&leaf, large, depth, is_null);
        let scheduler = MetadataListScheduler::try_from_page(&populated_fixture(depth, is_null))
            .unwrap()
            .unwrap();
        for range in [0..2, 1..2, 3..4, 2..7, 7..9] {
            let indices =
                UInt32Array::from((range.start as u32..range.end as u32).collect::<Vec<_>>());
            let expected = arrow_select::take::take(source.as_ref(), &indices, None).unwrap();
            let bytes = tree_bytes(&expected.to_data());
            let own = buffer_bytes(&expected.to_data());
            let budget = AllocationBudget::new(bytes);
            let mut short = budget.try_reserve(bytes - 1).unwrap();
            let input = scheduler
                .prepare(range.clone(), source.data_type().clone())
                .unwrap();
            assert!(input.decode_in(&mut short).unwrap().is_none());
            assert_eq!(short.len(), bytes - 1);
            drop(short);
            assert_eq!(budget.reserved(), 0);
            let mut work = budget.try_reserve(bytes).unwrap();
            let output = scheduler
                .prepare(range.clone(), source.data_type().clone())
                .unwrap()
                .decode_in(&mut work)
                .unwrap()
                .unwrap();
            assert_eq!(output.to_data(), expected.to_data());
            assert_eq!(work.len(), 0);
            let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
            drop((output, work));
            assert_eq!(budget.reserved(), own);
            assert!(budget.try_reserve(bytes).is_none());
            drop(alias);
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[rstest]
#[case::count_sum(0, "count sum")]
#[case::zero_count(1, "positive")]
#[case::null_parent_with_children(2, "null List parent")]
#[case::visible_leaf(3, "all children null")]
fn invalid_nonempty_structure_is_rejected(#[case] kind: u8, #[case] message: &str) {
    let mut page = populated_fixture(1, false);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    if kind == 3 {
        structure.layers[1].layer = Some(pb23::sparse_structural_layer::Layer::Validity(
            positions(false, 0, 1, 1),
        ));
    }
    let Some(pb23::sparse_structural_layer::Layer::List(list)) = structure.layers[0].layer.as_mut()
    else {
        unreachable!()
    };
    match kind {
        0 => list.num_child_slots = 9,
        1 => {
            list.counts = Some(EncodingUtils::constant(Some(
                0_u64.to_le_bytes().to_vec().into(),
            )))
        }
        2 => list.validity = Some(positions(true, 0, 2, 5)),
        _ => (),
    }
    let error = MetadataListScheduler::try_from_page(&page).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(message), "{error}");
}

#[tokio::test]
async fn tiny_selection_does_not_materialize_the_page_domain() {
    let mut page = populated_fixture(1, false);
    page.num_rows = u64::MAX;
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        layout.structure.as_mut().unwrap().codec.as_mut()
    else {
        unreachable!()
    };
    structure.layers[0] = list_layer(
        u64::MAX,
        u64::MAX,
        0,
        1,
        u64::MAX,
        EncodingUtils::constant(Some(1_u64.to_le_bytes().to_vec().into())),
        positions(true, 0, 1, 0),
    );
    let scheduler = MetadataListScheduler::try_from_page(&page)
        .unwrap()
        .unwrap();
    let field = Arc::new(Field::new("item", DataType::Binary, true));
    let budget = AllocationBudget::new(17); // two List offsets, two leaf offsets, one leaf bitmap
    let mut work = budget.try_reserve(17).unwrap();
    let output = scheduler
        .prepare(u64::MAX - 1..u64::MAX, DataType::List(field.clone()))
        .unwrap()
        .decode_in(&mut work)
        .unwrap()
        .unwrap();
    let expected = ListArray::new(
        field,
        OffsetBuffer::new(vec![0_i32, 1].into()),
        new_null_array(&DataType::Binary, 1),
        None,
    );
    assert_eq!(output.to_data(), arrow_array::Array::to_data(&expected));
    assert_eq!(work.len(), 0);
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn arithmetic_counts_validate_sum_and_prefixes_without_materialization() {
    for count in [0_u64, 1, 2, 7] {
        for first in [1_u64, 3, 9] {
            for step in [0_u64, 1, 3] {
                let encoding = if step == 0 {
                    EncodingUtils::constant(
                        (count > 0).then(|| first.to_le_bytes().to_vec().into()),
                    )
                } else {
                    RangeEncoder::new(64, first, step).encoding(count).unwrap()
                };
                let counts = Counts::parse(Some(&encoding), count).unwrap();
                let values = (0..count).map(|i| first + i * step).collect::<Vec<_>>();
                for prefix in 0..=count {
                    assert_eq!(
                        counts.prefix(prefix).unwrap(),
                        values[..prefix as usize].iter().sum::<u64>()
                    );
                }
            }
        }
    }
    let overflow = EncodingUtils::constant(Some(u64::MAX.to_le_bytes().to_vec().into()));
    let error = Counts::parse(Some(&overflow), 2).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("sum overflows"));
    let payload = EncodingUtils::flat(64, None);
    let error = Counts::parse(Some(&payload), 3).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("metadata-only counts"));
}
