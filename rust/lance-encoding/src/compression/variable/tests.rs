// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;
use crate::{
    compression::{
        DecompressionStrategy, DefaultDecompressionStrategy, VariablePerValueDecompressor,
    },
    encodings::{
        logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock},
        physical::{binary::VariableEncoder, fsst::test_utils::charge_per_value},
    },
};

fn fixture(
    bits: u8,
    values: &[Vec<u8>],
) -> (Box<dyn VariablePerValueDecompressor>, VariableWidthBlock) {
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let block = if bits == 32 {
        DataBlock::from_array(BinaryArray::from_vec(refs))
    } else {
        DataBlock::from_array(LargeBinaryArray::from_vec(refs))
    };
    let (encoded, description) = VariableEncoder::default().compress(block).unwrap();
    let PerValueDataBlock::Variable(input) = encoded else {
        panic!("expected raw variable values");
    };
    let codec = DefaultDecompressionStrategy::default()
        .create_variable_per_value_decompressor(&description)
        .unwrap();
    (codec, input)
}

#[rstest]
fn raw_per_value_detaches_shared_input_with_exact_overlap(
    #[values(32, 64)] bits: u8,
    #[values(0..4, 1..3, 1..2, 4..4)] range: Range<usize>,
    #[values(0, 1)] displacement: usize,
    #[values(0, 4096)] trailing: usize,
) {
    let values = [
        vec![b'a'; 257],
        vec![],
        b"\0\xfftail".to_vec(),
        b"end".to_vec(),
    ];
    let selected = &values[range.clone()];
    let (codec, mut input) = fixture(bits, &values);
    let width = usize::from(bits / 8);
    let start = values[..range.start].iter().map(Vec::len).sum::<usize>();
    let payload = selected.iter().map(Vec::len).sum::<usize>();
    input.data = input.data.slice_with_length(start, payload);
    let mut offsets = vec![0; width];
    let mut end = 0_u64;
    for value in selected {
        end += value.len() as u64;
        offsets.extend_from_slice(&end.to_le_bytes()[..width]);
    }
    input.offsets = LanceBuffer::from(offsets);
    input.num_values = selected.len() as u64;
    let mut bytes = input.data.as_ref().to_vec();
    bytes.resize(payload + trailing, 255);
    input.data = LanceBuffer::from(bytes);
    let needed = (selected.len() + 1) * width + payload;
    let physical = displacement + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical + needed);
    let mut work = pool.try_reserve(physical + needed).unwrap();
    let (input, sibling, actual) = charge_per_value(input, displacement, &mut work);
    assert_eq!(actual, physical);
    assert_eq!(
        codec.plan_decode(&input).unwrap(),
        PerValueDecodeBudget {
            output_bytes: needed,
            workspace_bytes: needed,
        }
    );
    let mut short = work.try_split(needed - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_in(&input, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), needed - 1);
        assert_eq!(pool.reserved(), physical + needed);
    }
    drop((work, short));
    let mut work = pool.try_reserve(needed).unwrap();
    let output = codec.try_decompress_in(&input, &mut work).unwrap().unwrap();
    assert!(work.is_empty());
    let refs = selected.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let array = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(array, expected);
    assert_eq!(array.get_buffer_memory_size(), needed);
    drop((input, output, work));
    assert_eq!(pool.reserved(), physical + needed);
    drop(sibling);
    assert_eq!(pool.reserved(), needed);
    drop(array);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn raw_per_value_rejects_complete_invalid_offsets_before_admission(
    #[values(32, 64)] bits: u8,
    #[values("decreasing", "signed", "nonzero", "outside", "truncated", "extra")] corruption: &str,
) {
    let (codec, mut input) = fixture(bits, &[b"ab".to_vec(), b"c".to_vec()]);
    let width = usize::from(bits / 8);
    let mut offsets = input.offsets.as_ref().to_vec();
    match corruption {
        "decreasing" => offsets[2 * width..3 * width].fill(0),
        "signed" => offsets[3 * width - 1] = 128,
        "nonzero" => offsets[0] = 1,
        "outside" => offsets[2 * width] = 4,
        "truncated" => {
            offsets.pop();
        }
        "extra" => offsets.extend_from_slice(&0_u64.to_le_bytes()[..width]),
        _ => unreachable!(),
    }
    input.offsets = LanceBuffer::from(offsets);
    let pool = AllocationBudget::new(0);
    let mut work = pool.try_reserve(0).unwrap();
    for _ in 0..2 {
        let error = codec.try_decompress_in(&input, &mut work).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("offset"));
        assert_eq!(work.len(), 0);
        assert_eq!(pool.reserved(), 0);
    }
    let mut callbacks = 0;
    let error = codec
        .visit_binary_values(&input, &mut |_| {
            callbacks += 1;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("offset"));
    assert_eq!(callbacks, 0);
}

#[rstest]
#[case::width(16, 0, "width")]
#[case::count(64, u64::MAX, "count")]
fn raw_per_value_invalid_geometry_keeps_credit(
    #[case] bits: u8,
    #[case] count: u64,
    #[case] message: &str,
) {
    let (codec, mut input) = fixture(64, &[b"x".to_vec()]);
    input.bits_per_offset = bits;
    input.num_values = count;
    let pool = AllocationBudget::new(16);
    let mut work = pool.try_reserve(16).unwrap();
    let error = codec.try_decompress_in(&input, &mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message));
    assert_eq!(work.len(), 16);
}

#[rstest]
fn raw_per_value_finite_consumers_progress_after_final_alias(#[values(32, 64)] bits: u8) {
    let (codec, input) = fixture(bits, &[b"payload".to_vec(), vec![]]);
    let needed = 3 * usize::from(bits / 8) + 7;
    let physical = 1 + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical + 2 * needed);
    let mut input_work = pool.try_reserve(physical).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut input_work);
    drop((input_work, sibling));
    let consume = || {
        let mut work = pool.try_reserve(needed).unwrap();
        codec.try_decompress_in(&input, &mut work).unwrap().unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let a = scope.spawn(consume);
        let b = scope.spawn(consume);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(pool.try_reserve(1).is_none());
    let DataBlock::VariableWidth(first) = first else {
        panic!("expected raw output")
    };
    let alias = first.data.slice_with_length(0, 1);
    drop(first);
    assert!(pool.try_reserve(needed).is_none());
    drop(alias);
    let third = consume();
    drop((second, third, input));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn raw_per_value_ordinary_decode_stays_zero_copy(#[values(32, 64)] bits: u8) {
    let (codec, input) = fixture(bits, &[b"payload".to_vec()]);
    let data_ptr = input.data.as_ptr();
    let offset_ptr = input.offsets.as_ptr();
    assert!(!codec.allocates_output());
    let DataBlock::VariableWidth(output) = codec.decompress(input).unwrap() else {
        panic!("expected raw output")
    };
    assert_eq!(output.data.as_ptr(), data_ptr);
    assert_eq!(output.offsets.as_ptr(), offset_ptr);
}
