// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, FixedSizeListArray, LargeBinaryArray, make_array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, bit_util};
use arrow_schema::{DataType, Field};
use lance_core::{Error, utils::allocation::AllocationBudget};
use rstest::rstest;

use super::*;

fn fixed(data: Vec<u8>, width: u64, rows: u64) -> DataBlock {
    DataBlock::FixedWidth(FixedWidthDataBlock {
        data: data.into(),
        bits_per_value: width,
        num_values: rows,
        block_info: Default::default(),
    })
}

fn binary(width: u8, offsets: &[i64], bytes: &[u8], is_unaligned: bool) -> DataBlock {
    let mut encoded = vec![0_u8; usize::from(is_unaligned)];
    for offset in offsets {
        if width == 32 {
            encoded.extend_from_slice(&(*offset as i32).to_ne_bytes());
        } else {
            encoded.extend_from_slice(&offset.to_ne_bytes());
        }
    }
    let len = encoded.len() - usize::from(is_unaligned);
    let encoded = LanceBuffer::from(encoded).slice_with_length(usize::from(is_unaligned), len);
    DataBlock::VariableWidth(VariableWidthBlock {
        data: bytes.to_vec().into(),
        offsets: encoded,
        bits_per_offset: width,
        num_values: offsets.len() as u64 - 1,
        block_info: Default::default(),
    })
}

#[rstest]
#[case::boolean(1)]
#[case::three_bits(3)]
#[case::seven_bits(7)]
#[case::byte(8)]
#[case::fixed_binary(24)]
#[case::wide(64)]
fn concat_discards_each_input_padding(#[case] width: u64) {
    let mut bits = Vec::new();
    let inputs = [3, 0, 5, 9].map(|rows| {
        let count = (rows * width) as usize;
        let selected = (0..count)
            .map(|index| (bits.len() + index) % 5 < 2)
            .collect::<Vec<_>>();
        bits.extend_from_slice(&selected);
        // Deliberately set padding bits and add a trailing byte: neither belongs
        // to the logical output, even across non-byte-aligned chunk boundaries.
        let mut data = vec![0xff; count.div_ceil(8) + 1];
        for (index, bit) in selected.into_iter().enumerate() {
            if !bit {
                bit_util::unset_bit(&mut data, index);
            }
        }
        fixed(data, width, rows)
    });
    let expected = BooleanBuffer::from(bits);
    let bytes = expected.values().len();
    let budget = AllocationBudget::new(bytes);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = DataBlock::try_concat_in(&inputs, &mut workspace)
        .unwrap()
        .unwrap();
    let DataBlock::FixedWidth(output) = output else {
        panic!("fixed output required")
    };
    assert_eq!(output.num_values, 17);
    assert_eq!(output.bits_per_value, width);
    assert_eq!(output.data.as_ref(), expected.values());
    assert_eq!(output.data.into_buffer().capacity(), bytes);
    assert!(workspace.is_empty());
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::binary(32)]
#[case::large_binary(64)]
fn concat_normalizes_sliced_and_unaligned_offsets(
    #[case] width: u8,
    #[values(false, true)] is_unaligned: bool,
) {
    let inputs = [
        binary(width, &[2, 4, 4, 5], b"xxabctail", is_unaligned),
        binary(width, &[1], b"unused", is_unaligned),
        binary(width, &[1, 4, 5], b"ydefgtail", is_unaligned),
    ];
    let expected_values: [&[u8]; 5] = [b"ab", b"", b"c", b"def", b"g"];
    let expected: Arc<dyn Array> = if width == 32 {
        Arc::new(BinaryArray::from_vec(expected_values.to_vec()))
    } else {
        Arc::new(LargeBinaryArray::from_vec(expected_values.to_vec()))
    };
    let bytes = 7 + 6 * usize::from(width / 8);
    let budget = AllocationBudget::new(bytes);
    let mut short = budget.try_reserve(bytes - 1).unwrap();
    for _ in 0..2 {
        assert!(
            DataBlock::try_concat_in(&inputs, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), bytes - 1);
        assert_eq!(budget.reserved(), bytes - 1);
        assert!(budget.try_reserve(2).is_none());
    }
    drop(short);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = DataBlock::try_concat_in(&inputs, &mut workspace)
        .unwrap()
        .unwrap();
    let array = make_array(
        output
            .try_to_arrow_in(expected.data_type(), &mut workspace)
            .unwrap()
            .unwrap(),
    );
    assert_eq!(array.to_data(), expected.to_data());
    assert_eq!(budget.reserved(), bytes);
    drop(output);
    assert_eq!(budget.reserved(), bytes);
    drop(array);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::nullable_boolean(0)]
#[case::nullable_vectors(1)]
#[case::deep_nullable_vectors(2)]
fn concat_preflights_all_nullable_children(#[case] depth: usize) {
    // Every level crosses a byte boundary between input pieces. Sliced Arrow
    // inputs exercise independent normalization, including nullable child bits.
    let rows = 13;
    let leaf_len = rows * 3_usize.pow(depth as u32);
    let mut expected: Arc<dyn Array> = Arc::new(BooleanArray::from(
        (0..leaf_len)
            .map(|i| if i % 5 == 0 { None } else { Some(i % 3 == 0) })
            .collect::<Vec<_>>(),
    ));
    let mut bytes = 2 * leaf_len.div_ceil(8);
    for _ in 0..depth {
        let len = expected.len() / 3;
        let field = Arc::new(Field::new("item", expected.data_type().clone(), true));
        let nulls = NullBuffer::from((0..len).map(|i| i % 7 != 0).collect::<Vec<_>>());
        expected = Arc::new(FixedSizeListArray::new(field, 3, expected, Some(nulls)));
        bytes += len.div_ceil(8);
    }
    let inputs = [(0, 3), (3, 0), (3, 5), (8, 5)]
        .map(|(start, len)| DataBlock::from_decoded_array(expected.slice(start, len)));
    let budget = AllocationBudget::new(bytes);
    let mut short = budget.try_reserve(bytes - 1).unwrap();
    assert!(
        DataBlock::try_concat_in(&inputs, &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), bytes - 1);
    assert_eq!(budget.reserved(), bytes - 1);
    drop(short);
    let mut workspace = budget.try_reserve(bytes).unwrap();
    let output = DataBlock::try_concat_in(&inputs, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(workspace.len(), 0);
    let actual = output
        .try_to_arrow_in(expected.data_type(), &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(actual, expected.to_data());
    assert_eq!(actual.get_buffer_memory_size(), bytes);
    drop(output);
    assert_eq!(budget.reserved(), bytes);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concat_promotes_valid_chunks_without_losing_nulls() {
    let inputs = [
        DataBlock::from_array(BooleanArray::from(vec![true, false, true])),
        DataBlock::from_array(BooleanArray::from(vec![None, Some(false), Some(true)])),
        DataBlock::from_array(BooleanArray::from(vec![false, true, false])),
    ];
    let expected = BooleanArray::from(vec![
        Some(true),
        Some(false),
        Some(true),
        None,
        Some(false),
        Some(true),
        Some(false),
        Some(true),
        Some(false),
    ]);
    let budget = AllocationBudget::new(4);
    let mut workspace = budget.try_reserve(4).unwrap();
    let output = DataBlock::try_concat_in(&inputs, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(
        output
            .try_to_arrow_in(&DataType::Boolean, &mut workspace)
            .unwrap()
            .unwrap(),
        expected.to_data()
    );
    assert!(workspace.is_empty());
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concat_keeps_physical_input_and_output_owners_charged() {
    let budget = AllocationBudget::new(68);
    let mut workspace = budget.try_reserve(68).unwrap();
    let mut input = workspace.try_allocate(64).unwrap().unwrap();
    input.as_mut()[..4].copy_from_slice(&[10, 20, 30, 40]);
    let owner = Buffer::from(input.freeze());
    let blocks = [0, 2].map(|start| {
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: owner.slice_with_length(start, 2).into(),
            bits_per_value: 8,
            num_values: 2,
            block_info: Default::default(),
        })
    });
    let output = DataBlock::try_concat_in(&blocks, &mut workspace)
        .unwrap()
        .unwrap();
    let array = make_array(
        output
            .try_to_arrow_in(&DataType::UInt8, &mut workspace)
            .unwrap()
            .unwrap(),
    );
    assert_eq!(array.to_data().buffers()[0].as_slice(), &[10, 20, 30, 40]);
    let last_input = owner.slice_with_length(63, 1);
    drop(owner);
    drop(blocks);
    assert_eq!(budget.reserved(), 68);
    assert!(budget.try_reserve(1).is_none());
    drop(last_input);
    assert_eq!(budget.reserved(), 4);
    // Another consumer can proceed while the first output is still retained.
    let second = budget.try_reserve(64).unwrap();
    let last_output = array.slice(3, 1);
    drop(array);
    drop(output);
    drop(workspace);
    assert_eq!(budget.reserved(), 68);
    assert!(budget.try_reserve(1).is_none());
    drop(last_output);
    assert_eq!(budget.reserved(), 64);
    drop(second);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::short_fixed(0, "fixed-width buffer is too short")]
#[case::short_validity(1, "validity buffer is too short")]
#[case::zero_dimension(2, "dimension must be positive")]
#[case::partial_child(3, "not divisible by dimension")]
#[case::offset_order(4, "offsets are not monotonic")]
#[case::negative_offset(5, "negative or unaddressable offset")]
#[case::offset_bounds(6, "offset exceeds value buffer")]
#[case::short_offsets(7, "offset buffer is too short")]
#[case::width_mismatch(8, "mismatched fixed widths")]
#[case::width_overflow(9, "output size overflows")]
fn concat_rejects_malformed_later_input_before_allocating(
    #[case] kind: usize,
    #[case] message: &str,
) {
    let (first, last) = match kind {
        0 => (fixed(vec![1], 8, 1), fixed(vec![], 8, 1)),
        1 => {
            let wrap = |nulls: Vec<u8>| {
                DataBlock::Nullable(NullableDataBlock {
                    data: Box::new(fixed(vec![1], 8, 1)),
                    nulls: nulls.into(),
                    block_info: Default::default(),
                })
            };
            (wrap(vec![1]), wrap(vec![]))
        }
        2 | 3 => {
            let wrap = |dimension| {
                DataBlock::FixedSizeList(FixedSizeListBlock {
                    child: Box::new(fixed(vec![1], 8, 1)),
                    dimension,
                })
            };
            // Put the invalid subtree under Nullable so count extraction cannot
            // call unchecked DataBlock::num_values before complete validation.
            let nullable = |data| {
                DataBlock::Nullable(NullableDataBlock {
                    data: Box::new(data),
                    nulls: vec![1].into(),
                    block_info: Default::default(),
                })
            };
            (
                nullable(wrap(1)),
                nullable(wrap(if kind == 2 { 0 } else { 2 })),
            )
        }
        4..=7 => {
            let first = binary(32, &[0, 1], b"a", false);
            let mut last = binary(
                32,
                match kind {
                    4 => &[0, 2, 1],
                    5 => &[-1, 0],
                    _ => &[0, 2],
                },
                b"a",
                true,
            );
            if kind == 7 {
                let DataBlock::VariableWidth(block) = &mut last else {
                    unreachable!()
                };
                block.num_values = 2;
            }
            (first, last)
        }
        8 => (fixed(vec![1], 8, 1), fixed(vec![1, 2], 16, 1)),
        9 => (fixed(vec![], u64::MAX, 0), fixed(vec![], u64::MAX, 2)),
        _ => unreachable!(),
    };
    let budget = AllocationBudget::new(1024);
    let mut workspace = budget.try_reserve(1024).unwrap();
    let error = DataBlock::try_concat_in(&[first, last], &mut workspace).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(workspace.len(), 1024);
    assert_eq!(budget.reserved(), 1024);
    assert!(budget.try_reserve(1).is_none());
}

#[test]
fn concat_empty_inputs_keep_zero_credit_and_typed_alignment() {
    let budget = AllocationBudget::new(0);
    let mut workspace = budget.try_reserve(0).unwrap();
    let inputs = [
        DataBlock::Empty(),
        DataBlock::AllNull(AllNullDataBlock { num_values: 0 }),
    ];
    assert!(matches!(
        DataBlock::try_concat_in(&inputs, &mut workspace)
            .unwrap()
            .unwrap(),
        DataBlock::Empty()
    ));
    let output = DataBlock::try_concat_in(&[fixed(vec![], 64, 0)], &mut workspace)
        .unwrap()
        .unwrap();
    let arrow = output
        .try_to_arrow_in(&DataType::Float64, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(arrow.len(), 0);
    assert_eq!(arrow.buffers()[0].as_ptr().align_offset(ALIGNMENT), 0);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concat_canonical_empty_is_neutral_between_typed_chunks() {
    let inputs = [
        fixed(vec![10], 8, 1),
        DataBlock::AllNull(AllNullDataBlock { num_values: 0 }),
        DataBlock::Empty(),
        fixed(vec![20], 8, 1),
    ];
    let budget = AllocationBudget::new(2);
    let mut workspace = budget.try_reserve(2).unwrap();
    let output = DataBlock::try_concat_in(&inputs, &mut workspace)
        .unwrap()
        .unwrap();
    let array = output
        .try_to_arrow_in(&DataType::UInt8, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(array.buffers()[0].as_slice(), &[10, 20]);
    drop(output);
    drop(array);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concat_unsupported_shape_never_falls_back() {
    let block = DataBlock::Struct(crate::data::StructDataBlock {
        children: vec![fixed(vec![1], 8, 1)],
        validity: None,
        block_info: Default::default(),
    });
    let budget = AllocationBudget::new(64);
    let mut workspace = budget.try_reserve(64).unwrap();
    let error = DataBlock::try_concat_in(&[block], &mut workspace).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("admitted concatenation requires")
    );
    assert_eq!(workspace.len(), 64);
    assert_eq!(budget.reserved(), 64);
}
