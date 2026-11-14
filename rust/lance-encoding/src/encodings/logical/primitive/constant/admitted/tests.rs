// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::VecDeque, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, Float32Array,
    LargeBinaryArray, LargeStringArray, StringArray, StructArray, UInt64Array, make_array,
    new_null_array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::DataType;
use lance_core::{
    Error,
    utils::allocation::{AllocationBudget, Reservation},
};
use rstest::rstest;

use super::super::{ConstantPageDecoder, DecodeConstantTask};
use crate::{
    decoder::{DecodePageTask, StructuralPageDecoder},
    repdef::{CompositeRepDefUnraveler, DefinitionInterpretation},
};

fn task(scalar: ArrayRef, count: u64) -> Box<dyn DecodePageTask> {
    let mut decoder = ConstantPageDecoder {
        ranges: VecDeque::from([5..5 + count]),
        scalar,
        rep: None,
        def: None,
        def_meaning: Arc::from([DefinitionInterpretation::AllValidItem]),
        max_rep: 0,
        max_visible_def: 0,
        cursor_row: 0,
        cursor_level: 0,
        num_rows: count,
    };
    decoder.drain(count).unwrap()
}

// Make every scalar buffer (and its validity) alias one admitted physical owner,
// with unused gaps/trailing bytes. Visible Arrow capacities cannot price that owner.
fn owned_scalar(
    source: &ArrayRef,
    headroom: usize,
) -> (ArrayRef, Arc<AllocationBudget>, Reservation, usize) {
    let data = source.to_data();
    let mut inputs = data.buffers().to_vec();
    if let Some(nulls) = data.nulls() {
        inputs.push(nulls.inner().inner().clone());
    }
    let mut cursor = 64_usize;
    let mut starts = Vec::new();
    for input in &inputs {
        cursor = cursor.next_multiple_of(64);
        starts.push(cursor);
        cursor += input.len();
    }
    let physical = cursor + 31;
    let budget = AllocationBudget::new(physical + headroom);
    let mut work = budget.try_reserve(physical + headroom).unwrap();
    let mut owner = work.try_allocate(physical).unwrap().unwrap();
    for (input, start) in inputs.iter().zip(&starts) {
        owner.as_mut()[*start..*start + input.len()].copy_from_slice(input);
    }
    let owner = owner.freeze();
    let mut buffers = inputs
        .iter()
        .zip(starts)
        .map(|(input, start)| Buffer::from(owner.slice(start..start + input.len())))
        .collect::<Vec<_>>();
    let mut builder = data.clone().into_builder();
    if let Some(nulls) = data.nulls() {
        let buffer = buffers.pop().unwrap();
        builder = builder.nulls(Some(NullBuffer::new(BooleanBuffer::new(
            buffer,
            nulls.offset(),
            nulls.len(),
        ))));
    }
    let result = make_array(builder.buffers(buffers).build().unwrap());
    (result, budget, work, physical)
}

fn fixture(kind: usize) -> ArrayRef {
    let data: ArrayRef = match kind {
        0 => Arc::new(Float32Array::from(vec![-1.0, 3.5, 7.0])),
        1 => Arc::new(BooleanArray::from(vec![false, true, false])),
        2 => Arc::new(BooleanArray::from(vec![true, false, true])),
        3 => Arc::new(BinaryArray::from(vec![
            b"before".as_slice(),
            b"\x00\x80\xff",
            b"after",
        ])),
        4 => Arc::new(LargeBinaryArray::from(vec![
            b"before".as_slice(),
            b"\x00\x80\xff",
            b"after",
        ])),
        5 => Arc::new(StringArray::from(vec!["before", "a界", "after"])),
        6 => Arc::new(LargeStringArray::from(vec!["before", "a界", "after"])),
        7 => Arc::new(BinaryArray::from(vec![b"before".as_slice(), b"", b"after"])),
        8 => new_null_array(&DataType::Float32, 3),
        9 => new_null_array(&DataType::LargeBinary, 3),
        10 => Arc::new(
            FixedSizeBinaryArray::try_from_iter([b"one", b"two", b"six"].into_iter()).unwrap(),
        ),
        _ => unreachable!(),
    };
    data.slice(1, 1)
}

#[rstest]
#[case::float(0)]
#[case::true_bits(1)]
#[case::false_bits(2)]
#[case::binary(3)]
#[case::large_binary(4)]
#[case::utf8(5)]
#[case::large_utf8(6)]
#[case::empty_binary(7)]
#[case::null_float(8)]
#[case::null_binary(9)]
#[case::fixed_binary(10)]
fn selected_constant_rows_use_expanded_admission(
    #[case] kind: usize,
    #[values(0, 1, 19)] count: usize,
) {
    let original = fixture(kind);
    let expected =
        arrow_select::take::take(original.as_ref(), &UInt64Array::from(vec![0; count]), None)
            .unwrap();
    // Price the logical destination independently of the new implementation.
    let output = match kind {
        0 => count * 4,
        1 | 2 => count.div_ceil(8),
        3 => (count + 1) * 4 + count * 3,
        4 => (count + 1) * 8 + count * 3,
        5 => (count + 1) * 4 + count * "a界".len(),
        6 => (count + 1) * 8 + count * "a界".len(),
        7 => (count + 1) * 4,
        8 => count * 4 + count.div_ceil(8),
        9 => (count + 1) * 8 + count.div_ceil(8),
        10 => count * 3,
        _ => unreachable!(),
    };
    let (scalar, budget, mut work, input) = owned_scalar(&original, output);
    let task = task(scalar.clone(), count as u64);
    if output > 0 {
        let mut short = work.try_split(output - 1).unwrap();
        let result = task.try_decode_in(&mut short).unwrap();
        if original.is_null(0) {
            // Nulls remain structural until the typed stage, which must also refuse.
            assert!(
                result
                    .unwrap()
                    .data
                    .try_to_arrow_in(original.data_type(), &mut short)
                    .unwrap()
                    .is_none()
            );
        } else {
            assert!(result.is_none());
        }
        assert_eq!(short.len(), output - 1);
        assert_eq!(budget.reserved(), input + output);
        drop(short);
        drop(work);
        work = budget.try_reserve(output).unwrap();
    }
    let decoded = task.try_decode_in(&mut work).unwrap().unwrap();
    let actual = decoded
        .data
        .try_to_arrow_in(original.data_type(), &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(actual, expected.to_data());
    assert!(work.is_empty());
    let mut repdef = CompositeRepDefUnraveler::new(vec![decoded.repdef]);
    assert!(repdef.unravel_validity(count).unwrap().is_none());
    let last = actual.slice(0, usize::from(count > 0));
    drop(actual);
    drop(decoded.data);
    drop(task);
    drop(scalar);
    drop(work);
    assert_eq!(budget.reserved(), output);
    drop(last);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn repeated_large_binary_refuses_oversized_batch_and_makes_finite_progress() {
    let bytes = vec![0x5a_u8; 4096];
    let original: ArrayRef = Arc::new(LargeBinaryArray::from(vec![bytes.as_slice()]));
    let target = 16 * 1024;
    let (scalar, budget, mut work, input) = owned_scalar(&original, target);
    let oversized = task(scalar.clone(), 300);
    assert!(oversized.try_decode_in(&mut work).unwrap().is_none());
    assert_eq!(work.len(), target);
    assert_eq!(budget.reserved(), input + target);
    drop(work);

    // One slow consumer retains a complete one-row output. Another request drains
    // all 300 rows in two-row operations under the same finite shared budget.
    let one_row = 4096 + 16;
    let two_rows = 8192 + 24;
    let mut slow_work = budget.try_reserve(one_row).unwrap();
    let slow = task(scalar.clone(), 1)
        .try_decode_in(&mut slow_work)
        .unwrap()
        .unwrap();
    let slow_output = slow
        .data
        .try_to_arrow_in(original.data_type(), &mut slow_work)
        .unwrap()
        .unwrap();
    drop(slow);
    drop(slow_work);
    let mut decoder = ConstantPageDecoder {
        ranges: VecDeque::from([0..300]),
        scalar: scalar.clone(),
        rep: None,
        def: None,
        def_meaning: Arc::from([DefinitionInterpretation::AllValidItem]),
        max_rep: 0,
        max_visible_def: 0,
        cursor_row: 0,
        cursor_level: 0,
        num_rows: 300,
    };
    let mut consumed = 0;
    for _ in 0..150 {
        let current = decoder.drain(2).unwrap();
        let mut work = budget.try_reserve(two_rows).unwrap();
        assert!(budget.try_reserve(one_row).is_none());
        let decoded = current.try_decode_in(&mut work).unwrap().unwrap();
        let array = make_array(
            decoded
                .data
                .try_to_arrow_in(original.data_type(), &mut work)
                .unwrap()
                .unwrap(),
        );
        let array = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(array.len(), 2);
        for value in array.iter() {
            assert_eq!(value.unwrap(), bytes);
        }
        consumed += array.len();
        assert!(budget.reserved() <= input + target);
    }
    assert_eq!(consumed, 300);
    assert_eq!(budget.reserved(), input + one_row);
    drop(slow_output);
    drop(oversized);
    drop(decoder);
    drop(scalar);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn offset_overflow_requests_a_smaller_batch_without_allocating() {
    let bytes = vec![42_u8; 32768];
    let original: ArrayRef = Arc::new(BinaryArray::from(vec![bytes.as_slice()]));
    let (scalar, budget, mut work, input) = owned_scalar(&original, 0);
    let task = task(scalar, 65536); // 2 GiB cannot fit an i32 Binary offset.
    assert!(task.try_decode_in(&mut work).unwrap().is_none());
    assert_eq!(budget.reserved(), input);
    assert!(work.is_empty());
}

#[test]
fn repdef_requires_its_own_admitted_path_and_keeps_ordinary_decode() {
    let scalar = fixture(0);
    let task = DecodeConstantTask {
        scalar,
        rep: None,
        def: Some(ScalarBuffer::from(vec![0_u16, 1, 0])),
        level_slices: vec![0..3],
        visible_items_total: 0,
        def_meaning: Arc::from([DefinitionInterpretation::NullableItem]),
        max_visible_def: 1,
    };
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let error = task.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("rep/def reconstruction"));
    assert_eq!(work.len(), 64);
    let output = Box::new(task)
        .decode()
        .unwrap()
        .data
        .into_arrow(DataType::Float32, true)
        .unwrap();
    assert_eq!(
        output,
        Float32Array::from(vec![Some(3.5), None, Some(3.5)]).to_data()
    );
}

#[test]
fn unsupported_page_task_never_falls_back() {
    #[derive(Debug)]
    struct Unsupported;
    impl DecodePageTask for Unsupported {
        fn decode(self: Box<Self>) -> lance_core::Result<crate::decoder::DecodedPage> {
            panic!("unadmitted decoder must not run")
        }
    }
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let error = Unsupported.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no admitted decoder"));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn nonnull_complex_constant_requires_an_admitted_codec() {
    let scalar: ArrayRef = Arc::new(StructArray::new_empty_fields(1, None));
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let error = task(scalar, 1).try_decode_in(&mut work).err().unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no admitted expansion"));
    assert_eq!(work.len(), 64);
}
