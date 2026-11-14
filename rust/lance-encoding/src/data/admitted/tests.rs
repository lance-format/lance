// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, StructArray, make_array, new_null_array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, bit_util};
use arrow_schema::{DataType, Field};
use lance_core::{Error, utils::allocation::AllocationBudget};
use rstest::rstest;

use crate::{
    buffer::LanceBuffer,
    data::{
        AllNullDataBlock, BlockInfo, DataBlock, FixedSizeListBlock, FixedWidthDataBlock,
        NullableDataBlock, StructDataBlock, VariableWidthBlock,
    },
};

fn field(data_type: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", data_type, true))
}

fn null_type(kind: usize) -> DataType {
    match kind {
        0 => DataType::Null,
        1 => DataType::Boolean,
        2 => DataType::Float32,
        3 => DataType::Decimal128(20, 4),
        4 => DataType::FixedSizeBinary(3),
        5 => DataType::Utf8,
        6 => DataType::LargeBinary,
        7 => DataType::List(field(DataType::LargeBinary)),
        8 => DataType::LargeList(field(DataType::List(field(DataType::LargeBinary)))),
        9 => DataType::FixedSizeList(
            field(DataType::FixedSizeList(field(DataType::Boolean), 3)),
            2,
        ),
        10 => DataType::Struct(
            vec![
                field(DataType::FixedSizeList(field(DataType::Float32), 3)),
                Arc::new(Field::new("text", DataType::LargeUtf8, false)),
            ]
            .into(),
        ),
        11 => DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3),
        12 => DataType::FixedSizeList(
            Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Boolean, false)), 3),
                false,
            )),
            2,
        ),
        _ => unreachable!(),
    }
}

#[rstest]
#[case::null(0)]
#[case::boolean(1)]
#[case::float(2)]
#[case::decimal(3)]
#[case::fixed_binary(4)]
#[case::text(5)]
#[case::large_binary(6)]
#[case::list_large_binary(7)]
#[case::nested_lists(8)]
#[case::nested_vectors(9)]
#[case::struct_with_nonnullable_child(10)]
#[case::nonnull_vector_child(11)]
#[case::nested_nonnull_vector_children(12)]
fn typed_nulls_admit_the_complete_arrow_payload(
    #[case] kind: usize,
    #[values(0, 1, 513)] len: usize,
) {
    let data_type = null_type(kind);
    // Arrow independently constructs the expected layout, including empty offsets
    // and children. Its exact null-array allocation sizes are the required payload.
    let expected = new_null_array(&data_type, len);
    let bytes = expected.to_data().get_buffer_memory_size();
    // Parent-mask validation uses one temporary child bitmap. Nested validations
    // run sequentially; price the largest bitmap, not their sum.
    let scratch = match kind {
        11 => (len * 3).div_ceil(8),
        12 => (len * 6).div_ceil(8),
        _ => 0,
    };
    let required = bytes + scratch;
    let block = DataBlock::AllNull(AllNullDataBlock {
        num_values: len as u64,
    });
    if required > 0 {
        let budget = AllocationBudget::new(required - 1);
        let mut work = budget.try_reserve(required - 1).unwrap();
        assert!(
            block
                .try_to_arrow_in(&data_type, &mut work)
                .unwrap()
                .is_none()
        );
        assert_eq!(work.len(), required - 1);
        assert_eq!(budget.reserved(), required - 1);
    }
    let budget = AllocationBudget::new(required);
    let mut work = budget.try_reserve(required).unwrap();
    let actual = block
        .try_to_arrow_in(&data_type, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), scratch);
    assert_eq!(actual, expected.to_data());
    assert_eq!(actual.get_buffer_memory_size(), bytes);
    assert_eq!(budget.reserved(), required);
    let last = actual.slice(0, usize::from(len > 0));
    drop(actual);
    drop(work);
    assert_eq!(budget.reserved(), bytes);
    drop(last);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn pinned_arrow_validation_scratch_has_the_priced_layout(
    #[values(0, 1, 513, 1024)] len: usize,
    #[values(1, 3, 18)] dimension: usize,
) {
    let parent =
        NullBuffer::from((0..len + 3).map(|i| i % 3 != 0).collect::<Vec<_>>()).slice(3, len);
    let expanded = parent.expand(dimension);
    assert_eq!(
        expanded.inner().inner().capacity(),
        (len * dimension).div_ceil(8)
    );
    for i in 0..len * dimension {
        assert_eq!(expanded.is_null(i), parent.is_null(i / dimension));
    }
}

#[rstest]
fn byte_booleans_charge_only_the_packed_output(#[values(0, 1, 9, 513)] len: usize) {
    let output_bytes = len.div_ceil(8);
    let budget = AllocationBudget::new(len + output_bytes);
    let mut work = budget.try_reserve(len + output_bytes).unwrap();
    let mut input = work.try_allocate(len).unwrap().unwrap();
    for (i, byte) in input.as_mut().iter_mut().enumerate() {
        *byte = [0, 1, 2, 255][i % 4];
    }
    let block = DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: 8,
        num_values: len as u64,
        data: LanceBuffer::from(Buffer::from(input.freeze())),
        block_info: BlockInfo::new(),
    });
    if output_bytes > 0 {
        let mut short = work.try_split(output_bytes - 1).unwrap();
        assert!(
            block
                .try_to_arrow_in(&DataType::Boolean, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), output_bytes - 1);
        drop(short);
        drop(work);
        work = budget.try_reserve(output_bytes).unwrap();
    }
    let actual = block
        .try_to_arrow_in(&DataType::Boolean, &mut work)
        .unwrap()
        .unwrap();
    let expected = BooleanArray::from_iter((0..len).map(|i| Some(i % 4 != 0)));
    assert_eq!(actual, expected.to_data());
    assert_eq!(budget.reserved(), len + output_bytes);
    assert!(work.is_empty());
    drop(block);
    assert_eq!(budget.reserved(), output_bytes);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn nested_conversion_admits_all_children_and_preserves_sliced_validity() {
    // One physical input contains expanded booleans, leaf validity and a sliced
    // parent validity. All three aliases must retain a single physical charge.
    let input_bytes = 23;
    let new_bytes = 3 + 7 * 8 + 1; // packed 18 booleans + six null LargeBinary rows
    let budget = AllocationBudget::new(input_bytes + new_bytes);
    let mut work = budget.try_reserve(input_bytes + new_bytes).unwrap();
    let mut input = work.try_allocate(input_bytes).unwrap().unwrap();
    for i in 0..18 {
        input.as_mut()[i] = if i % 3 == 0 { 0 } else { 7 };
        if i % 4 != 1 {
            bit_util::set_bit(&mut input.as_mut()[18..21], i);
        }
    }
    for i in 0..11 {
        if i % 3 != 0 {
            bit_util::set_bit(&mut input.as_mut()[21..23], i);
        }
    }
    let source = input.freeze();
    let booleans = DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: 8,
        num_values: 18,
        data: LanceBuffer::from(Buffer::from(source.slice(..18))),
        block_info: BlockInfo::new(),
    });
    let list = DataBlock::FixedSizeList(FixedSizeListBlock {
        dimension: 3,
        child: Box::new(DataBlock::Nullable(NullableDataBlock {
            data: Box::new(booleans),
            block_info: BlockInfo::new(),
            nulls: LanceBuffer::from(Buffer::from(source.slice(18..21))),
        })),
    });
    let block = DataBlock::Struct(StructDataBlock {
        children: vec![list, DataBlock::AllNull(AllNullDataBlock { num_values: 6 })],
        block_info: BlockInfo::new(),
        validity: Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from(source.slice(21..23)),
            5,
            6,
        ))),
    });
    let fields = vec![
        field(DataType::FixedSizeList(field(DataType::Boolean), 3)),
        Arc::new(Field::new("payload", DataType::LargeBinary, true)),
    ]
    .into();
    let data_type = DataType::Struct(fields);
    let mut short = work.try_split(new_bytes - 1).unwrap();
    assert!(
        block
            .try_to_arrow_in(&data_type, &mut short)
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), new_bytes - 1);
    assert_eq!(budget.reserved(), input_bytes + new_bytes);
    drop(short);
    drop(work);
    work = budget.try_reserve(new_bytes).unwrap();
    let actual = block
        .try_to_arrow_in(&data_type, &mut work)
        .unwrap()
        .unwrap();
    let expected_values =
        BooleanArray::from_iter((0..18).map(|i| (i % 4 != 1).then_some(i % 3 != 0)));
    let expected_list =
        FixedSizeListArray::new(field(DataType::Boolean), 3, Arc::new(expected_values), None);
    let DataType::Struct(fields) = data_type else {
        unreachable!()
    };
    let expected = StructArray::new(
        fields,
        vec![
            Arc::new(expected_list),
            new_null_array(&DataType::LargeBinary, 6),
        ],
        Some(NullBuffer::from(
            (5..11).map(|i| i % 3 != 0).collect::<Vec<_>>(),
        )),
    );
    assert_eq!(actual, expected.to_data());
    assert!(work.is_empty());
    let actual = make_array(actual);
    let last = actual.slice(0, 1);
    drop(actual);
    drop(block);
    drop(source);
    drop(work);
    assert_eq!(budget.reserved(), input_bytes + new_bytes);
    drop(last);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn compatible_variable_layout_is_zero_copy() {
    let budget = AllocationBudget::new(27);
    let mut work = budget.try_reserve(27).unwrap();
    let mut input = work.try_allocate(27).unwrap().unwrap();
    for (chunk, value) in input.as_mut()[..24]
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip([0_i64, 1, 3])
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    input.as_mut()[24..].copy_from_slice(b"abc");
    let source = input.freeze();
    let block = DataBlock::VariableWidth(VariableWidthBlock {
        bits_per_offset: 64,
        num_values: 2,
        block_info: BlockInfo::new(),
        offsets: LanceBuffer::from(Buffer::from(source.slice(..24))),
        data: LanceBuffer::from(Buffer::from(source.slice(24..))),
    });
    let actual = block
        .try_to_arrow_in(&DataType::LargeUtf8, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    assert_eq!(actual.buffers()[0].as_ptr(), source.as_ptr());
    assert_eq!(actual.buffers()[1].as_ptr(), source.slice(24..).as_ptr());
    let expected = arrow_array::LargeStringArray::from(vec!["a", "bc"]);
    assert_eq!(actual, expected.to_data());
    drop(block);
    drop(source);
    assert_eq!(budget.reserved(), 27);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::negative_dimension(DataType::FixedSizeList(field(DataType::Float32), -1), 1, "dimension")]
#[case::overflow(DataType::Float64, u64::MAX, "overflows")]
#[case::unsupported_nested(DataType::Struct(vec![field(DataType::LargeBinary), field(DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)))].into()), 4, "no admitted null")]
fn invalid_or_unsupported_null_layout_preserves_credit(
    #[case] data_type: DataType,
    #[case] len: u64,
    #[case] message: &str,
) {
    let block = DataBlock::AllNull(AllNullDataBlock { num_values: len });
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let error = block.try_to_arrow_in(&data_type, &mut work).unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::NotSupported { .. }
    ));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(work.len(), 1024);
    assert_eq!(budget.reserved(), 1024);
}

#[test]
fn late_invalid_child_is_detected_before_allocating_earlier_child() {
    let block = DataBlock::Struct(StructDataBlock {
        children: vec![
            DataBlock::AllNull(AllNullDataBlock { num_values: 9 }),
            DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: 8,
                num_values: 9,
                block_info: BlockInfo::new(),
            }),
        ],
        block_info: BlockInfo::new(),
        validity: None,
    });
    let data_type =
        DataType::Struct(vec![field(DataType::Float32), field(DataType::Boolean)].into());
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let error = block.try_to_arrow_in(&data_type, &mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("expanded Boolean buffer"));
    assert_eq!(work.len(), 1024);
    assert_eq!(budget.reserved(), 1024);
}
