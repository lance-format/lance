// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{BinaryArray, Int32Array, ListArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::Field;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

fn leaf<T: ByteArrayType>() -> ArrayRef {
    let offsets = [0, 4, 5, 11, 13, 13, 16, 19].map(|value| T::Offset::from_usize(value).unwrap());
    Arc::new(
        GenericByteArray::<T>::try_new(
            OffsetBuffer::new(offsets.to_vec().into()),
            Buffer::from("dropahiddenéxyzend".as_bytes()),
            Some(NullBuffer::from(vec![
                true, true, false, true, true, true, true,
            ])),
        )
        .unwrap()
        .slice(1, 5),
    )
}

fn wrap<O: OffsetSizeTrait>(child: ArrayRef) -> ArrayRef {
    let offsets = if child.len() == 5 {
        [0, 1, 3, 3, 5]
    } else {
        [0, 0, 1, 2, 3]
    };
    let field = Arc::new(
        Field::new("frames", child.data_type().clone(), true).with_metadata(
            std::collections::HashMap::from([("unit".into(), "bytes".into())]),
        ),
    );
    Arc::new(
        GenericListArray::<O>::try_new(
            field,
            OffsetBuffer::new(
                offsets
                    .map(|value| O::from_usize(value).unwrap())
                    .to_vec()
                    .into(),
            ),
            child,
            // The last null list has nonempty physical children; they are hidden.
            Some(NullBuffer::from(vec![true, true, true, false])),
        )
        .unwrap()
        .slice(1, 3),
    )
}

#[rstest]
fn nested_matches_arrow(
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    data_type: DataType,
    #[values(false, true)] large_outer: bool,
    #[values(1, 3)] depth: usize,
) {
    let mut source = match data_type {
        DataType::Binary => leaf::<GenericBinaryType<i32>>(),
        DataType::LargeBinary => leaf::<GenericBinaryType<i64>>(),
        DataType::Utf8 => leaf::<GenericStringType<i32>>(),
        DataType::LargeUtf8 => leaf::<GenericStringType<i64>>(),
        _ => unreachable!(),
    };
    for level in 0..depth {
        source = if (level % 2 == 0) == large_outer {
            wrap::<i64>(source)
        } else {
            wrap::<i32>(source)
        };
    }
    let second = source.slice(0, 2);
    let sources = [source.as_ref(), second.as_ref()];
    let selections: &[&[(usize, usize)]] = &[
        &[(0, 2), (1, 0), (0, 1), (0, 0), (1, 1), (0, 2), (1, 0)],
        &[(0, 2), (0, 2)],
        &[(0, 1)],
        &[],
    ];
    for indices in selections {
        let expected = arrow_select::interleave::interleave(&sources, indices).unwrap();
        let budget = AllocationBudget::new(4096);
        let mut workspace = budget.try_reserve(4096).unwrap();
        let actual = try_interleave_nested_binary_in(&sources, indices, &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(actual.data_type(), source.data_type());
        assert_eq!(actual.to_data(), expected.to_data());
        drop((actual, workspace));
        assert_eq!(budget.reserved(), 0);
    }
}

fn small_list() -> ListArray {
    let values = Arc::new(BinaryArray::from(vec![
        b"a".as_slice(),
        b"hidden".as_slice(),
        b"bcd".as_slice(),
    ]));
    ListArray::try_new(
        Arc::new(Field::new("frame", DataType::Binary, false)),
        OffsetBuffer::new(vec![0, 1, 2, 2, 3].into()),
        values,
        Some(NullBuffer::from(vec![true, false, true, true])),
    )
    .unwrap()
}

const ALL_ROWS: [(usize, usize); 4] = [(0, 0), (0, 1), (0, 2), (0, 3)];

#[rstest]
#[case::zero(0)]
#[case::parent_only(21)]
#[case::one_byte_short(36)]
fn refuses_complete_tree_before_allocating(#[case] available: usize) {
    let source = small_list();
    // Parent: five i32 offsets + validity = 21. Selected child: three i32
    // offsets + "abcd" = 16. Hidden child bytes are not part of the output.
    let budget = AllocationBudget::new(37);
    let mut workspace = budget.try_reserve(available).unwrap();
    assert!(
        try_interleave_nested_binary_in(&[&source], &ALL_ROWS, &mut workspace)
            .unwrap()
            .is_none()
    );
    assert_eq!(workspace.len(), available);
    assert_eq!(budget.reserved(), available);
    drop(workspace);
    let mut workspace = budget.try_reserve(37).unwrap();
    let output = try_interleave_nested_binary_in(&[&source], &ALL_ROWS, &mut workspace)
        .unwrap()
        .unwrap();
    let expected = arrow_select::interleave::interleave(&[&source], &ALL_ROWS).unwrap();
    assert_eq!(output.to_data(), expected.to_data());
    assert_eq!(workspace.len(), 0);
    assert_eq!(budget.reserved(), 37);
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::parent_offsets(0, 21)]
#[case::parent_validity(1, 21)]
#[case::child_offsets(2, 16)]
#[case::child_values(3, 16)]
fn aliases_retain_their_physical_node(#[case] held: usize, #[case] charge: usize) {
    let source = small_list();
    let budget = AllocationBudget::new(74);
    let mut workspace = budget.try_reserve(74).unwrap();
    let input = try_interleave_nested_binary_in(&[&source], &ALL_ROWS, &mut workspace)
        .unwrap()
        .unwrap();
    let output = try_interleave_nested_binary_in(&[input.as_ref()], &ALL_ROWS, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(workspace.len(), 0);
    let data = output.to_data();
    let alias = match held {
        0 => data.buffers()[0].slice_with_length(0, 1),
        1 => data.nulls().unwrap().buffer().slice_with_length(0, 1),
        2 | 3 => data.child_data()[0].buffers()[held - 2].slice_with_length(0, 1),
        _ => unreachable!(),
    };
    drop((data, output, workspace));
    assert_eq!(budget.reserved(), 37 + charge);
    drop(input);
    assert_eq!(budget.reserved(), charge);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn descendant_consumer_release_allows_next_operation() {
    let source = small_list();
    let budget = AllocationBudget::new(37);
    let mut workspace = budget.try_reserve(37).unwrap();
    let output = try_interleave_nested_binary_in(&[&source], &ALL_ROWS, &mut workspace)
        .unwrap()
        .unwrap();
    let alias = output.to_data().child_data()[0].buffers()[1].clone();
    drop((output, workspace));
    assert_eq!(budget.reserved(), 16);
    let next = budget.wait_for_capacity(37);
    futures::pin_mut!(next);
    assert!(futures::poll!(&mut next).is_pending());
    drop(alias);
    assert!(matches!(
        futures::poll!(&mut next),
        std::task::Poll::Ready(Ok(()))
    ));
    let mut workspace = budget.try_reserve(37).unwrap();
    let output = try_interleave_nested_binary_in(&[&source], &ALL_ROWS, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(
        output.to_data(),
        arrow_select::interleave::interleave(&[&source], &ALL_ROWS)
            .unwrap()
            .to_data()
    );
    drop((output, workspace));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::invalid_source(0, "source 1 exceeds 1 sources")]
#[case::invalid_row(1, "row 4 exceeds source 0 length 4")]
#[case::mismatched_fields(2, "expected source type")]
#[case::unsupported_empty_selection(3, "does not support Int32")]
fn errors_precede_capacity_refusal(#[case] case: usize, #[case] detail: &str) {
    let source = small_list();
    let wrong = ListArray::try_new(
        Arc::new(Field::new("other", DataType::Int32, false)),
        OffsetBuffer::new(vec![0, 1].into()),
        Arc::new(Int32Array::from(vec![1])),
        None,
    )
    .unwrap();
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let result = match case {
        0 => try_interleave_nested_binary_in(&[&source], &[(1, 0)], &mut workspace),
        1 => try_interleave_nested_binary_in(&[&source], &[(0, 4)], &mut workspace),
        2 => try_interleave_nested_binary_in(&[&source, &wrong], &[], &mut workspace),
        3 => try_interleave_nested_binary_in(&[&wrong], &[], &mut workspace),
        _ => unreachable!(),
    };
    let error = result.unwrap_err();
    if case == 3 {
        assert!(matches!(error, Error::NotSupported { .. }));
    } else {
        assert!(matches!(error, Error::InvalidInput { .. }));
    }
    assert!(error.to_string().contains(detail), "{error}");
    assert_eq!(budget.reserved(), 0);
    assert_eq!(workspace.len(), 0);
}

#[rstest]
#[case::flat(0)]
#[case::list(1)]
#[case::deep(3)]
fn concat_matches_arrow_and_retains_owners(#[case] depth: usize) {
    let mut array = leaf::<GenericBinaryType<i64>>();
    for _ in 0..depth {
        array = wrap::<i32>(array);
    }
    let sliced = array.slice(1, array.len() - 1);
    let empty = array.slice(0, 0);
    let sources = [sliced.as_ref(), empty.as_ref(), array.as_ref()];
    let expected = arrow_select::concat::concat(&sources).unwrap();
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let output = try_concat_nested_binary_in(&sources, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(output.to_data(), expected.to_data());
    let charge = 4096 - work.len();
    let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
    drop((output, work));
    assert!(budget.reserved() > 1);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    let mut short = budget.try_reserve(charge - 1).unwrap();
    assert!(
        try_concat_nested_binary_in(&sources, &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), charge - 1);
    drop(short);
    let mut exact = budget.try_reserve(charge).unwrap();
    let output = try_concat_nested_binary_in(&sources, &mut exact)
        .unwrap()
        .unwrap();
    assert_eq!(output.to_data(), expected.to_data());
    assert!(exact.is_empty());
    drop((output, exact));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concat_validates_sources_without_credit() {
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let bytes = leaf::<GenericBinaryType<i32>>();
    let strings = leaf::<GenericStringType<i32>>();
    let unsupported = Int32Array::from(vec![1]);
    for sources in [vec![], vec![bytes.as_ref(), strings.as_ref()]] {
        let error = try_concat_nested_binary_in(&sources, &mut work).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("interleave"));
    }
    let error = try_concat_nested_binary_in(&[&unsupported], &mut work).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("Int32"));
}
