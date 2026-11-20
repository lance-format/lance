// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;
use crate::{
    compression::{BlockDecompressor, MiniBlockDecompressor},
    compression_config::CompressionFieldParams,
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
        physical::binary::BinaryMiniBlockEncoder,
    },
    format::metadata::compressive_encoding::Compression,
};

mod general;
mod metadata;
mod payload;

fn frame(values: &[&[u8]], width: usize) -> Vec<u8> {
    let mut pos = (values.len() + 1) * width;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(pos as u64).to_le_bytes()[..width]);
    for value in values {
        pos += value.len();
        bytes.extend_from_slice(&(pos as u64).to_le_bytes()[..width]);
    }
    for value in values {
        bytes.extend_from_slice(value);
    }
    bytes.resize(bytes.len().next_multiple_of(width), 0);
    bytes
}

fn expected(values: &[&[u8]], bits: u8) -> arrow_data::ArrayData {
    if bits == 32 {
        BinaryArray::from_vec(values.to_vec()).to_data()
    } else {
        LargeBinaryArray::from_vec(values.to_vec()).to_data()
    }
}

#[rstest]
#[case::full(0..5)]
#[case::middle(1..4)]
#[case::empty_item(2..3)]
#[case::empty_selection(3..3)]
#[case::tail(4..5)]
fn binary_ranges_charge_selected_payload_and_offsets(
    #[case] range: Range<u64>,
    #[values(32, 64)] bits: u8,
    #[values(0, 1)] displacement: usize,
) {
    let values: &[&[u8]] = &[b"alpha", &[0, 255, 4], b"", b"z", b"tail"];
    let width = usize::from(bits / 8);
    let selected = &values[range.start as usize..range.end as usize];
    let bytes = (selected.len() + 1) * width + selected.iter().map(|v| v.len()).sum::<usize>();
    let encoded = frame(values, width);
    let physical = encoded.len() + 129;
    let pool = AllocationBudget::new(physical + bytes);
    let mut credit = pool.try_reserve(physical).unwrap();
    let mut input = credit.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[displacement..displacement + encoded.len()].copy_from_slice(&encoded);
    let backing = Buffer::from(input.freeze());
    let sibling = backing.slice_with_length(physical - 1, 1);
    let data = [LanceBuffer::from(
        backing.slice_with_length(displacement, encoded.len()),
    )];
    drop(backing);
    let codec = BinaryMiniBlockDecompressor::new(bits).unwrap();
    let plan = codec
        .plan_range(&data, values.len() as u64, range.clone())
        .unwrap();
    assert_eq!(plan.output_bytes, bytes);
    assert_eq!(plan.workspace_bytes, bytes);
    let mut short = pool.try_reserve(bytes - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_range_in(&data, 5, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), bytes - 1);
    }
    drop(short);
    let mut work = pool.try_reserve(bytes).unwrap();
    let output = codec
        .try_decompress_range_in(&data, 5, range, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let array = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(array, expected(selected, bits));
    assert_eq!(array.get_buffer_memory_size(), bytes);
    drop(output);
    drop(data);
    assert_eq!(pool.reserved(), physical + bytes);
    drop(sibling);
    assert_eq!(pool.reserved(), bytes);
    drop(array);
    drop(work);
    drop(credit);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::arity(0, "one buffer")]
#[case::truncated_word(1, "offset table")]
#[case::truncated_table(2, "offset table")]
#[case::overlap(3, "overlap")]
#[case::late_decrease(4, "decrease")]
#[case::late_bounds(5, "bounds")]
#[case::count_overflow(6, "overflows")]
#[case::selection_bounds(7, "range")]
#[case::selection_reversed(8, "range")]
fn malformed_binary_plan_preserves_workspace(
    #[case] fault: u8,
    #[case] message: &str,
    #[values(32, 64)] bits: u8,
) {
    let width = usize::from(bits / 8);
    let mut bytes = frame(&[b"a", b"bc", b"def"], width);
    let mut count = 3;
    let mut range = 0..1;
    match fault {
        1 => {
            bytes.pop();
        }
        2 => bytes.truncate(width * 2),
        3 => bytes[..width].fill(0),
        4 => bytes[3 * width..4 * width].fill(0),
        5 => bytes[3 * width..4 * width].copy_from_slice(&999_u64.to_le_bytes()[..width]),
        6 => count = u64::MAX,
        7 => range = 0..4,
        8 => range = Range { start: 2, end: 1 },
        _ => {}
    }
    let data = if fault == 0 {
        vec![]
    } else {
        vec![LanceBuffer::from(bytes)]
    };
    let codec = BinaryMiniBlockDecompressor::new(bits).unwrap();
    let pool = AllocationBudget::new(1024);
    let mut work = pool.try_reserve(1024).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, count, range.clone(), &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(work.len(), 1024);
        assert_eq!(pool.reserved(), 1024);
    }
}

#[rstest]
fn prefix_count_does_not_mistake_remaining_offsets_for_values(#[values(32, 64)] bits: u8) {
    let data = [LanceBuffer::from(frame(
        &[b"a", b"bc", b"unused"],
        usize::from(bits / 8),
    ))];
    let codec = BinaryMiniBlockDecompressor::new(bits).unwrap();
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    let output = codec
        .try_decompress_range_in(&data, 2, 1..2, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(output.as_variable_width_ref().unwrap().data.as_ref(), b"bc");
}

#[rstest]
fn current_writer_flat_fallback_round_trip(#[values(32, 64)] bits: u8) {
    let values: &[&[u8]] = &[b"", b"one", b"x", b"four"];
    let array: arrow_array::ArrayRef = if bits == 32 {
        std::sync::Arc::new(BinaryArray::from_vec(values.to_vec()))
    } else {
        std::sync::Arc::new(LargeBinaryArray::from_vec(values.to_vec()))
    };
    let encoder =
        BinaryMiniBlockEncoder::with_generic_offsets(None, CompressionFieldParams::default());
    let (encoded, description) = encoder
        .compress(
            MiniBlockCompressionContext::new(0, true, true),
            DataBlock::from(array),
        )
        .unwrap();
    let Some(Compression::Variable(variable)) = description.compression.as_ref() else {
        panic!("expected Variable")
    };
    assert!(matches!(
        variable.offsets.as_ref().unwrap().compression,
        Some(Compression::Flat(_))
    ));
    assert_eq!(encoded.chunks.len(), 1);
    let length = encoded.chunks[0].buffer_sizes[0] as usize;
    let data = [encoded.data[0].slice_with_length(0, length)];
    let codec = BinaryMiniBlockDecompressor::from_variable(variable).unwrap();
    let pool = AllocationBudget::new(128);
    let mut work = pool.try_reserve(128).unwrap();
    let block = codec
        .try_decompress_range_in(&data, 4, 1..4, &mut work)
        .unwrap()
        .unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = block.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected(&values[1..], bits));
}

#[derive(Debug)]
struct ForbiddenOffsets;
impl BlockDecompressor for ForbiddenOffsets {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary offset decoder must not run during admission")
    }
}

#[test]
fn unsupported_offsets_do_not_fall_back() {
    let codec = BinaryMiniBlockDecompressor {
        layout: BinaryMiniBlockLayout::Generic {
            bits_per_offset: 32,
            offsets: Box::new(ForbiddenOffsets),
            offsets_have_payload: true,
        },
    };
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    let error = codec
        .try_decompress_range_in(
            &[LanceBuffer::empty(), LanceBuffer::empty()],
            0,
            0..0,
            &mut work,
        )
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("unsigned payload visitor"));
    assert_eq!(work.len(), 64);
}
