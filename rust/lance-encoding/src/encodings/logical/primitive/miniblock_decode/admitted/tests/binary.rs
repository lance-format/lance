// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, BinaryArray, LargeBinaryArray};

use super::multiple::select;
use super::*;
use crate::encodings::physical::binary::BinaryMiniBlockDecompressor;

mod fsst;
mod metadata;
mod payload;

fn binary_fixture(bits: u8, large: bool, nested: bool, available: usize) -> Fixture {
    let width = usize::from(bits / 8);
    let start = 5 * width;
    let mut payload = Vec::new();
    for position in [0, 1, 1, 3, 4] {
        payload.extend_from_slice(&((start + position) as u64).to_le_bytes()[..width]);
    }
    payload.extend_from_slice(b"abcd");
    payload.resize(payload.len().next_multiple_of(width), 0);
    framed_fixture(
        &[&payload],
        BinaryMiniBlockDecompressor::new(bits).unwrap(),
        large,
        nested,
        available,
    )
}

fn framed_fixture(
    payloads: &[&[u8]],
    decoder: impl MiniBlockDecompressor + 'static,
    large: bool,
    nested: bool,
    available: usize,
) -> Fixture {
    let rep = nested.then_some([1_u16, 0, 1, 1, 1, 0]);
    let def = nested.then_some([0_u16, 1, 2, 3, 0, 0]);
    let Fixture { mut task, .. } = fixture(
        rep.as_ref().map(|v| &v[..]),
        def.as_ref().map(|v| &v[..]),
        Codec::Flat,
        large,
        0,
    );
    let count = if nested { 6_u16 } else { 4 };
    let mut frame = count.to_le_bytes().to_vec();
    if nested {
        frame.extend_from_slice(&12_u16.to_le_bytes());
        frame.extend_from_slice(&12_u16.to_le_bytes());
    }
    for payload in payloads {
        if large {
            frame.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        } else {
            frame.extend_from_slice(&u16::try_from(payload.len()).unwrap().to_le_bytes());
        }
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    for levels in rep.iter().chain(def.iter()) {
        frame.extend_from_slice(bytemuck::cast_slice(levels));
        frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    for payload in payloads {
        frame.extend_from_slice(payload);
        frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    let physical = frame.len() + 41;
    let budget = AllocationBudget::new(physical + available);
    let mut work = budget.try_reserve(physical + available).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + frame.len()].copy_from_slice(&frame);
    task.instructions[0].1.data =
        LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, frame.len()));
    task.instructions[0].1.byte_range = 0..frame.len() as u64;
    task.value_decompressor = Arc::new(decoder);
    task.num_buffers = payloads.len() as u64;
    Fixture {
        task,
        budget,
        work,
        physical,
    }
}

#[rstest]
fn binary_values_preserve_task_overlap_and_list_semantics(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
    #[values(false, true)] nested: bool,
) {
    let width = usize::from(bits / 8);
    let selected: &[&[u8]] = if multiple || nested {
        &[b"a", b"", b"bc", b"d"]
    } else {
        &[b"", b"bc"]
    };
    let output_bytes =
        (selected.len() + 1) * width + selected.iter().map(|v| v.len()).sum::<usize>();
    let required = if multiple {
        11 * width + 8 + if nested { 24 } else { 0 }
    } else {
        output_bytes + if nested { 48 } else { 0 }
    };
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = binary_fixture(bits, large, nested, required);
    select(
        &mut task,
        if multiple {
            &[(0, 2), (2, 2)]
        } else if nested {
            &[(0, 4)]
        } else {
            &[(1, 2)]
        },
    );
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), required - 1);
        assert_eq!(budget.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let array = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    let expected = if bits == 32 {
        BinaryArray::from_vec(selected.to_vec()).to_data()
    } else {
        LargeBinaryArray::from_vec(selected.to_vec()).to_data()
    };
    assert_eq!(array, expected);
    assert_eq!(array.get_buffer_memory_size(), output_bytes);
    assert_eq!(work.len(), if nested && !multiple { 24 } else { 0 });
    drop(task);
    drop(work);
    assert_eq!(
        budget.reserved(),
        output_bytes + if nested { 24 } else { 0 }
    );
    if nested {
        let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
        // Reconstruction is a separate operation with its own complete headroom.
        let mut reconstruction = budget.try_reserve(64).unwrap();
        let valid = structure
            .try_unravel_validity_in(4, &mut reconstruction)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(valid.iter().collect::<Vec<_>>(), [true, false, true, true]);
        let (offsets, validity) = structure
            .try_unravel_offsets_in::<i32>(&mut reconstruction)
            .unwrap()
            .unwrap();
        assert_eq!(offsets.as_ref(), [0, 2, 2, 2, 4]);
        assert_eq!(
            validity.unwrap().iter().collect::<Vec<_>>(),
            [true, false, true, true]
        );
    } else {
        drop(page.repdef);
    }
    drop(page.data);
    assert_eq!(budget.reserved(), output_bytes);
    drop(array);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn empty_binary_selections_still_charge_one_offset(#[values(32, 64)] bits: u8) {
    let width = usize::from(bits / 8);
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = binary_fixture(bits, false, false, 3 * width);
    select(&mut task, &[(1, 0), (3, 0)]);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let data = page.data.as_variable_width_ref().unwrap();
    assert_eq!(data.num_values, 0);
    assert!(data.data.is_empty());
    assert_eq!(data.offsets.as_ref(), vec![0; width]);
    assert!(work.is_empty());
    drop(task);
    drop(work);
    assert_eq!(budget.reserved(), width);
    drop(page);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn malformed_later_binary_chunk_keeps_transactional_credit(#[values(false, true)] nested: bool) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = binary_fixture(32, false, nested, 1024);
    select(&mut task, &[(0, 2), (2, 2)]);
    let original = &task.instructions[1].1.data;
    let mut bytes = original.to_vec();
    let payload_start = if nested { 40 } else { 8 };
    bytes[payload_start + 16..payload_start + 20].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut corrupt = work.try_allocate(bytes.len()).unwrap().unwrap();
    corrupt.as_mut().copy_from_slice(&bytes);
    task.instructions[1].1.data = LanceBuffer::from(Buffer::from(corrupt.freeze()));
    let available = work.len();
    let reserved = budget.reserved();
    for _ in 0..2 {
        let error = task.try_decode_in(&mut work).err().unwrap();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("bounds"), "{error}");
        assert_eq!(work.len(), available);
        assert_eq!(budget.reserved(), reserved);
    }
}

#[test]
fn absent_binary_instruction_does_not_return_an_untyped_empty() {
    let Fixture {
        mut task, mut work, ..
    } = binary_fixture(32, false, false, 16);
    task.instructions.clear();
    let error = task.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("explicit empty range"));
    assert_eq!(work.len(), 16);
}
