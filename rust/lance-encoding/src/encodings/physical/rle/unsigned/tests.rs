// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::OutOfLineBitpacking;
use crate::{
    compression::{BlockCompressor, MiniBlockDecompressor, create_fixed_width_block_decompressor},
    compression_config::CompressionFieldParams,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{
        binary::BinaryMiniBlockDecompressor,
        constant::ConstantEncoder,
        range::RangeEncoder,
        rle::BlockRleEncoder,
        sequence::{BlockCost, select_block_codec},
        value::ValueEncoder,
    },
    format::{
        EncodingUtils,
        metadata::{CompressiveEncoding, compressive_encoding::Compression},
    },
};
use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

#[derive(Clone, Copy, Debug)]
enum Children {
    Flat,
    RangeValues,
    RangeLengths,
    ConstantLengths,
    MetadataOnly,
    #[cfg(feature = "bitpacking")]
    PackedValues,
    #[cfg(feature = "bitpacking")]
    PackedLengths,
}

fn fixed(bits: u64, values: &[u64]) -> FixedWidthDataBlock {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes()[..(bits / 8) as usize]);
    }
    FixedWidthDataBlock {
        bits_per_value: bits,
        num_values: values.len() as u64,
        data: LanceBuffer::from(bytes),
        block_info: BlockInfo::new(),
    }
}

fn fixture(bits: u64, children: Children) -> (Option<LanceBuffer>, CompressiveEncoding, Vec<u64>) {
    let mut values: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut lengths: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut run_values = [0, 3, 7];
    let mut run_lengths = [2, 1, 3];
    match children {
        Children::Flat => {}
        Children::RangeValues | Children::MetadataOnly => {
            run_values = [0, 2, 4];
            values = Box::new(RangeEncoder::new(bits, 0, 2));
            if matches!(children, Children::MetadataOnly) {
                run_lengths = [2, 2, 2];
                lengths = Box::new(ConstantEncoder::new(32, 2));
            }
        }
        Children::RangeLengths => {
            run_lengths = [1, 2, 3];
            lengths = Box::new(RangeEncoder::new(32, 1, 1));
        }
        Children::ConstantLengths => {
            run_lengths = [2, 2, 2];
            lengths = Box::new(ConstantEncoder::new(32, 2));
        }
        #[cfg(feature = "bitpacking")]
        Children::PackedValues => values = Box::new(OutOfLineBitpacking::new(3, bits)),
        #[cfg(feature = "bitpacking")]
        Children::PackedLengths => lengths = Box::new(OutOfLineBitpacking::new(2, 32)),
    }
    let expected = run_values
        .into_iter()
        .zip(run_lengths)
        .flat_map(|(v, n)| std::iter::repeat_n(v, n))
        .collect::<Vec<_>>();
    let (data, encoding) = BlockRleEncoder::try_new(bits, values, lengths)
        .unwrap()
        .compress(DataBlock::FixedWidth(fixed(bits, &expected)))
        .unwrap();
    (data, encoding, expected)
}

#[rstest]
#[case::flat(Children::Flat)]
#[case::range_values(Children::RangeValues)]
#[case::range_lengths(Children::RangeLengths)]
#[case::constant_lengths(Children::ConstantLengths)]
#[case::metadata_only(Children::MetadataOnly)]
fn rle_unsigned_visit_matches_encoder(
    #[case] children: Children,
    #[values(32, 64)] bits: u64,
    #[values(0, 1)] displacement: usize,
) {
    check_visitor(bits, children, displacement);
}

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::packed_values(Children::PackedValues)]
#[case::packed_lengths(Children::PackedLengths)]
fn rle_unsigned_visit_packed_children(
    #[case] children: Children,
    #[values(32, 64)] bits: u64,
    #[values(0, 1)] displacement: usize,
) {
    check_visitor(bits, children, displacement);
}

fn check_visitor(bits: u64, children: Children, displacement: usize) {
    let (data, encoding, expected) = fixture(bits, children);
    check_encoded(bits, data, encoding, expected, displacement);
}

fn check_encoded(
    bits: u64,
    data: Option<LanceBuffer>,
    encoding: CompressiveEncoding,
    expected: Vec<u64>,
    displacement: usize,
) {
    let data = data.map(|data| {
        let mut bytes = vec![255; displacement];
        bytes.extend_from_slice(&data);
        LanceBuffer::from(bytes).slice_with_length(displacement, data.len())
    });
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(data.as_ref(), bits, expected.len() as u64, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected);
    let mut calls = 0;
    let error = codec
        .visit_unsigned(data.as_ref(), bits, expected.len() as u64, &mut |_| {
            calls += 1;
            Err(Error::invalid_input("consumer stopped"))
        })
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("consumer stopped"));
    assert_eq!(calls, 1);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn rle_unsigned_packed_children_cross_block(
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_packed_lengths: bool,
) {
    let expected = (0..1051_u64)
        .flat_map(|i| std::iter::repeat_n(i % 8, (1 + i % 4) as usize))
        .collect::<Vec<_>>();
    let (values, lengths): (Box<dyn BlockCompressor>, Box<dyn BlockCompressor>) =
        if is_packed_lengths {
            (
                Box::<ValueEncoder>::default(),
                Box::new(OutOfLineBitpacking::new(3, 32)),
            )
        } else {
            (
                Box::new(OutOfLineBitpacking::new(3, bits)),
                Box::<ValueEncoder>::default(),
            )
        };
    let (data, encoding) = BlockRleEncoder::try_new(bits, values, lengths)
        .unwrap()
        .compress(DataBlock::FixedWidth(fixed(bits, &expected)))
        .unwrap();
    check_encoded(bits, data, encoding, expected, 1);
}

#[rstest]
#[case::flat(Children::Flat)]
#[case::metadata(Children::MetadataOnly)]
#[case::range_values(Children::RangeValues)]
fn rle_unsigned_binary_keeps_selected_output_and_owner(
    #[case] children: Children,
    #[values(32, 64)] bits: u64,
    #[values(false, true)] empty: bool,
) {
    let (input, encoding, offsets) = fixture(bits, children);
    let all = b"abcdefgh";
    let values = &all[..*offsets.last().unwrap() as usize];
    let range = if empty { 2..2 } else { 1..4 };
    let selected = offsets[range.start as usize..=range.end as usize]
        .windows(2)
        .map(|v| &values[v[0] as usize..v[1] as usize])
        .collect::<Vec<_>>();
    let required = (selected.len() + 1) * (bits / 8) as usize
        + selected.iter().map(|v| v.len()).sum::<usize>();
    let payload_len = input.as_ref().map_or(0, |v| v.len());
    let physical = payload_len + values.len() + 33;
    let pool = AllocationBudget::new(physical + required);
    let mut work = pool.try_reserve(physical + required).unwrap();
    let mut owner = work.try_allocate(physical).unwrap().unwrap();
    if let Some(input) = &input {
        owner.as_mut()[1..1 + payload_len].copy_from_slice(input);
    }
    owner.as_mut()[1 + payload_len..1 + payload_len + values.len()].copy_from_slice(values);
    let owner = Buffer::from(owner.freeze());
    let sibling = owner.slice_with_length(physical - 1, 1);
    let mut data = Vec::new();
    if input.is_some() {
        data.push(LanceBuffer::from(owner.slice_with_length(1, payload_len)));
    }
    data.push(LanceBuffer::from(
        owner.slice_with_length(1 + payload_len, values.len()),
    ));
    drop(owner);
    let Some(Compression::Variable(variable)) = EncodingUtils::variable(encoding, None).compression
    else {
        panic!("expected Variable")
    };
    let codec = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
    let count = offsets.len() as u64 - 1;
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_range_in(&data, count, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), required - 1);
        assert_eq!(pool.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = pool.try_reserve(required).unwrap();
    let output = codec
        .try_decompress_range_in(&data, count, range, &mut work)
        .unwrap()
        .unwrap();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(selected).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(selected).to_data(),
        )
    };
    let actual = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.get_buffer_memory_size(), required);
    assert!(work.is_empty());
    drop(data);
    drop(output);
    drop(work);
    assert_eq!(pool.reserved(), physical + required);
    drop(sibling);
    assert_eq!(pool.reserved(), required);
    drop(actual);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::zero_last_length(0)]
#[case::excess_last_length(1)]
#[case::short_last_length(2)]
#[case::truncated_length(3)]
#[case::extra_length(4)]
#[case::bad_header(5)]
fn rle_unsigned_rejects_corrupt_lengths_before_visits(
    #[case] fault: u8,
    #[values(32, 64)] bits: u64,
) {
    let (data, encoding, expected) = fixture(bits, Children::Flat);
    let mut data = data.unwrap().to_vec();
    let len = data.len();
    match fault {
        0 => data[len - 4..].copy_from_slice(&0_u32.to_le_bytes()),
        1 => data[len - 4..].copy_from_slice(&4_u32.to_le_bytes()),
        2 => data[len - 4..].copy_from_slice(&2_u32.to_le_bytes()),
        3 => {
            data.pop();
        }
        4 => data.extend_from_slice(&1_u32.to_le_bytes()),
        _ => data[..8].copy_from_slice(&u64::MAX.to_le_bytes()),
    }
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut calls = 0;
    let error = codec
        .visit_unsigned(
            Some(&LanceBuffer::from(data)),
            bits,
            expected.len() as u64,
            &mut |_| {
                calls += 1;
                Ok(())
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::CorruptFile { .. }
    ));
    assert!(error.to_string().contains("RLE") || error.to_string().contains("payload"));
    assert_eq!(calls, 0);
}

#[rstest]
fn rle_unsigned_current_selector_round_trip(#[values(32, 64)] bits: u64) {
    let member = (0..32_u64)
        .flat_map(|v| std::iter::repeat_n(v, 8))
        .collect::<Vec<_>>();
    let mut both = member.clone();
    both.extend_from_slice(&member);
    let data = fixed(bits, &both);
    let selected = select_block_codec(
        &data,
        &[0..member.len(), member.len()..both.len()],
        &CompressionFieldParams::default(),
        BlockCost::default(),
        true,
    )
    .unwrap();
    assert!(matches!(
        selected.expected_encoding().compression,
        Some(Compression::Rle(_))
    ));
    let (payload, encoding) = selected.compress(fixed(bits, &member)).unwrap();
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(payload.as_ref(), bits, member.len() as u64, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, member);
}

#[rstest]
#[case::late_decrease(0)]
#[case::late_out_of_bounds(1)]
#[case::late_zero_length(2)]
fn rle_unsigned_binary_rejects_late_corruption_before_admission(
    #[case] fault: u8,
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_empty: bool,
) {
    let (input, encoding, offsets) = fixture(bits, Children::Flat);
    let mut input = input.unwrap().to_vec();
    let width = (bits / 8) as usize;
    if fault == 2 {
        let len = input.len();
        input[len - 4..].fill(0);
    } else {
        let value = if fault == 0 { 1_u64 } else { 8_u64 };
        input[8 + 2 * width..8 + 3 * width].copy_from_slice(&value.to_le_bytes()[..width]);
    }
    let data = [
        LanceBuffer::from(input),
        LanceBuffer::from(b"abcdefg".to_vec()),
    ];
    let Some(Compression::Variable(variable)) = EncodingUtils::variable(encoding, None).compression
    else {
        panic!("expected Variable")
    };
    let codec = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(
                &data,
                offsets.len() as u64 - 1,
                if is_empty { 0..0 } else { 0..1 },
                &mut work,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(error.to_string().contains("RLE") || error.to_string().contains("offset"));
        assert_eq!(work.len(), 64);
        assert_eq!(pool.reserved(), 64);
    }
}

#[rstest]
fn rle_unsigned_validates_width_and_cardinality(
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_empty: bool,
) {
    let (input, encoding, values) = fixture(bits, Children::Flat);
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut calls = 0;
    let error = codec
        .visit_unsigned(
            input.as_ref(),
            if is_empty { bits } else { 16 },
            if is_empty { 0 } else { values.len() as u64 },
            &mut |_| {
                calls += 1;
                Ok(())
            },
        )
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("expected nonempty"));
    assert_eq!(calls, 0);
}
