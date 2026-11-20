// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{BinaryArray, Int32Array};
use arrow_buffer::ArrowNativeType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;

fn fixture<T: ByteArrayType>() -> ArrayRef {
    // The null value has a nonempty physical segment. The slice also starts
    // inside both offset and validity buffers, rather than at their origin.
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

fn sources(data_type: &DataType) -> [ArrayRef; 2] {
    let array = match data_type {
        DataType::Binary => fixture::<GenericBinaryType<i32>>(),
        DataType::LargeBinary => fixture::<GenericBinaryType<i64>>(),
        DataType::Utf8 => fixture::<GenericStringType<i32>>(),
        DataType::LargeUtf8 => fixture::<GenericStringType<i64>>(),
        _ => unreachable!(),
    };
    let second = array.slice(2, 3);
    [array, second]
}

#[rstest]
#[case::binary(DataType::Binary)]
#[case::large_binary(DataType::LargeBinary)]
#[case::utf8(DataType::Utf8)]
#[case::large_utf8(DataType::LargeUtf8)]
fn matches_arrow_for_sliced_sources(#[case] data_type: DataType) {
    let arrays = sources(&data_type);
    let arrays = [arrays[0].as_ref(), arrays[1].as_ref()];
    let selections: &[&[(usize, usize)]] = &[
        &[(1, 2), (0, 1), (0, 2), (1, 1), (0, 0), (1, 2), (0, 1)],
        &[(0, 1), (0, 1)],
        &[(0, 3), (1, 1)],
        &[],
    ];
    for indices in selections {
        let expected = arrow_select::interleave::interleave(&arrays, indices).unwrap();
        let budget = AllocationBudget::new(1024);
        let mut workspace = budget.try_reserve(1024).unwrap();
        let actual = try_interleave_binary_in(&arrays, indices, &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(actual.to_data(), expected.to_data());
        drop((workspace, actual));
        assert_eq!(budget.reserved(), 0);
    }
}

#[test]
fn binary_preserves_non_utf8_bytes() {
    let source = BinaryArray::from(vec![Some(&[0xff, 0x00][..]), None, Some(&[][..])]);
    let indices = [(0, 1), (0, 0), (0, 2), (0, 0)];
    let expected = arrow_select::interleave::interleave(&[&source], &indices).unwrap();
    let budget = AllocationBudget::new(64);
    let mut workspace = budget.try_reserve(64).unwrap();
    let actual = try_interleave_binary_in(&[&source], &indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(actual.to_data(), expected.to_data());
}

#[rstest]
#[case::offsets(0)]
#[case::values(1)]
#[case::validity(2)]
fn retains_one_physical_charge_until_last_alias(#[case] held_buffer: usize) {
    let source = BinaryArray::from(vec![Some(b"a".as_slice()), None, Some(b"bcd".as_slice())]);
    let indices = [(0, 0), (0, 1), (0, 2)];
    // Four i32 offsets, four payload bytes, one validity byte: 21 bytes.
    let budget = AllocationBudget::new(42);
    let mut workspace = budget.try_reserve(42).unwrap();
    let input = try_interleave_binary_in(&[&source], &indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(workspace.len(), 21);
    let output = try_interleave_binary_in(&[input.as_ref()], &indices, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(workspace.len(), 0);
    assert!(
        try_interleave_binary_in(&[input.as_ref()], &indices, &mut workspace)
            .unwrap()
            .is_none()
    );
    let data = output.to_data();
    let alias = match held_buffer {
        0 | 1 => data.buffers()[held_buffer].slice_with_length(0, 1),
        2 => data.nulls().unwrap().buffer().slice_with_length(0, 1),
        _ => unreachable!(),
    };
    drop((data, output, workspace));
    assert_eq!(budget.reserved(), 42);
    drop(input);
    assert_eq!(budget.reserved(), 21);
    assert!(budget.try_reserve(22).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(42).is_some());
}

#[test]
fn capacity_refusal_does_not_consume_workspace() {
    let source = BinaryArray::from(vec![Some(b"a".as_slice()), None, Some(b"bcd".as_slice())]);
    let budget = AllocationBudget::new(21);
    let mut short = budget.try_reserve(20).unwrap();
    assert!(
        try_interleave_binary_in(&[&source], &[(0, 0), (0, 1), (0, 2)], &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), 20);
    assert_eq!(budget.reserved(), 20);
    drop(short);
    let mut exact = budget.try_reserve(21).unwrap();
    let output = try_interleave_binary_in(&[&source], &[(0, 0), (0, 1), (0, 2)], &mut exact)
        .unwrap()
        .unwrap();
    assert_eq!(exact.len(), 0);
    assert_eq!(budget.reserved(), 21);
    drop((exact, output));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn consumer_alias_blocks_then_releases_next_operation() {
    let source = BinaryArray::from(vec![b"a".as_slice()]);
    let budget = AllocationBudget::new(9);
    let mut workspace = budget.try_reserve(9).unwrap();
    let output = try_interleave_binary_in(&[&source], &[(0, 0)], &mut workspace)
        .unwrap()
        .unwrap();
    let held_value = output.to_data().buffers()[1].clone();
    drop((workspace, output));
    // The next operation owns no credit/input needed to release this pool.
    // A cancelled waiter must not consume the eventual consumer-release wake.
    {
        let cancelled = budget.wait_for_capacity(9);
        futures::pin_mut!(cancelled);
        assert!(futures::poll!(&mut cancelled).is_pending());
    }
    let next = budget.wait_for_capacity(9);
    futures::pin_mut!(next);
    assert!(futures::poll!(&mut next).is_pending());
    assert!(budget.try_reserve(9).is_none());
    drop(held_value);
    assert!(matches!(
        futures::poll!(&mut next),
        std::task::Poll::Ready(Ok(()))
    ));
    let mut workspace = budget.try_reserve(9).unwrap();
    let output = try_interleave_binary_in(&[&source], &[(0, 0)], &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(output.to_data(), source.to_data());
    drop((workspace, output));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::empty_sources(0, "sources must not be empty")]
#[case::source_index(1, "source 1 exceeds 1 sources")]
#[case::row_index(2, "row 1 exceeds source 0 length 1")]
#[case::mixed_unselected_type(3, "expected source type Binary, got Int32")]
#[case::unsupported(4, "does not support Int32")]
fn validates_before_capacity_refusal(#[case] case: usize, #[case] detail: &str) {
    let source = BinaryArray::from(vec![b"a".as_slice()]);
    let wrong = Int32Array::from(vec![1]);
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let result = match case {
        0 => try_interleave_binary_in(&[], &[], &mut workspace),
        1 => try_interleave_binary_in(&[&source], &[(1, 0)], &mut workspace),
        2 => try_interleave_binary_in(&[&source], &[(0, 1)], &mut workspace),
        3 => try_interleave_binary_in(&[&source, &wrong], &[], &mut workspace),
        4 => try_interleave_binary_in(&[&wrong], &[], &mut workspace),
        _ => unreachable!(),
    };
    let error = result.unwrap_err();
    if case == 4 {
        assert!(matches!(error, Error::NotSupported { .. }));
    } else {
        assert!(matches!(error, Error::InvalidInput { .. }));
    }
    assert!(error.to_string().contains(detail), "{error}");
    assert_eq!(workspace.len(), 0);
    assert_eq!(budget.reserved(), 0);
}
