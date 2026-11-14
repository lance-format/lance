// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::{
        BlockCompressor, BlockDecompressor, MiniBlockDecompressor,
        create_fixed_width_block_decompressor,
    },
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{
        binary::BinaryMiniBlockDecompressor,
        constant::{ConstantBlockDecompressor, ConstantEncoder},
        dictionary::BlockDictionaryEncoder,
        range::RangeEncoder,
        value::ValueEncoder,
    },
    format::{
        EncodingUtils,
        metadata::{CompressiveEncoding, compressive_encoding::Compression},
    },
};
#[cfg(feature = "bitpacking")]
use crate::{
    compression_config::CompressionFieldParams,
    encodings::physical::{
        bitpacking::OutOfLineBitpacking,
        sequence::{BlockCost, select_block_codec},
    },
};
use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum Children {
    Flat,
    MetadataItems,
    MetadataIndices,
    MetadataOnly,
    ConstantIndices,
    #[cfg(feature = "bitpacking")]
    PackedIndices,
    #[cfg(feature = "bitpacking")]
    PackedItems,
    #[cfg(feature = "bitpacking")]
    PackedBoth,
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
    // Item order is deliberately different from logical offset order.
    let mut items = vec![4, 0, 1, 3];
    let mut expected = vec![0, 1, 1, 3, 4];
    let mut indices: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut values: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    match children {
        Children::Flat => {}
        Children::MetadataItems => {
            items = vec![0, 1, 2, 3];
            expected = vec![0, 1, 1, 2, 3];
            values = Box::new(RangeEncoder::new(bits, 0, 1));
        }
        Children::MetadataIndices => {
            items = vec![0, 1, 3, 4];
            expected = items.clone();
            indices = Box::new(RangeEncoder::new(32, 0, 1));
        }
        Children::MetadataOnly => {
            items = vec![0, 1, 2, 3];
            expected = items.clone();
            indices = Box::new(RangeEncoder::new(32, 0, 1));
            values = Box::new(RangeEncoder::new(bits, 0, 1));
        }
        Children::ConstantIndices => {
            items = vec![0, 3, 5, 7];
            expected = vec![0; 5];
            indices = Box::new(ConstantEncoder::new(32, 0));
        }
        #[cfg(feature = "bitpacking")]
        Children::PackedIndices => indices = Box::new(OutOfLineBitpacking::new(2, 32)),
        #[cfg(feature = "bitpacking")]
        Children::PackedItems => values = Box::new(OutOfLineBitpacking::new(3, bits)),
        #[cfg(feature = "bitpacking")]
        Children::PackedBoth => {
            indices = Box::new(OutOfLineBitpacking::new(2, 32));
            values = Box::new(OutOfLineBitpacking::new(3, bits));
        }
    }
    let (data, encoding) = BlockDictionaryEncoder::try_new(bits, Arc::from(items), indices, values)
        .unwrap()
        .compress(DataBlock::FixedWidth(fixed(bits, &expected)))
        .unwrap();
    (data, encoding, expected)
}

#[rstest]
#[case::flat(Children::Flat)]
#[case::metadata_items(Children::MetadataItems)]
#[case::metadata_indices(Children::MetadataIndices)]
#[case::metadata_only(Children::MetadataOnly)]
#[case::unused_items(Children::ConstantIndices)]
fn dictionary_unsigned_binary_keeps_only_selected_output(
    #[case] children: Children,
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_empty: bool,
) {
    check_binary(bits, children, is_empty);
}

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::indices(Children::PackedIndices)]
#[case::items(Children::PackedItems)]
#[case::both(Children::PackedBoth)]
fn dictionary_unsigned_packed_binary_keeps_only_selected_output(
    #[case] children: Children,
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_empty: bool,
) {
    check_binary(bits, children, is_empty);
}

fn check_binary(bits: u64, children: Children, is_empty: bool) {
    let (input, encoding, offsets) = fixture(bits, children);
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(input.as_ref(), bits, offsets.len() as u64, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, offsets);
    let mut calls = 0;
    let error = codec
        .visit_unsigned(input.as_ref(), bits, offsets.len() as u64, &mut |_| {
            calls += 1;
            Err(Error::invalid_input("consumer stopped"))
        })
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("consumer stopped"));
    assert_eq!(calls, 1);
    let values = &b"abcdefgh"[..*offsets.last().unwrap() as usize];
    let range = if is_empty { 2..2 } else { 1..3 };
    let selected = offsets[range.start as usize..=range.end as usize]
        .windows(2)
        .map(|v| &values[v[0] as usize..v[1] as usize])
        .collect::<Vec<_>>();
    let required = (selected.len() + 1) * (bits / 8) as usize
        + selected.iter().map(|v| v.len()).sum::<usize>();
    let input_len = input.as_ref().map_or(0, |v| v.len());
    let physical = input_len + values.len() + 43;
    let pool = AllocationBudget::new(physical + required);
    let mut work = pool.try_reserve(physical + required).unwrap();
    let mut owner = work.try_allocate(physical).unwrap().unwrap();
    if let Some(input) = &input {
        owner.as_mut()[1..1 + input_len].copy_from_slice(input);
    }
    owner.as_mut()[1 + input_len..1 + input_len + values.len()].copy_from_slice(values);
    let owner = Buffer::from(owner.freeze());
    let sibling = owner.slice_with_length(physical - 1, 1);
    let mut data = Vec::new();
    if input.is_some() {
        data.push(LanceBuffer::from(owner.slice_with_length(1, input_len)));
    }
    data.push(LanceBuffer::from(
        owner.slice_with_length(1 + input_len, values.len()),
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
#[case::late_invalid_index(0)]
#[case::late_decreasing_offset(1)]
#[case::truncated_items(2)]
#[case::extra_items(3)]
#[case::header_overflow(4)]
fn dictionary_unsigned_binary_rejects_corruption_before_admission(
    #[case] fault: u8,
    #[values(32, 64)] bits: u64,
    #[values(false, true)] is_empty: bool,
) {
    let (input, encoding, offsets) = fixture(bits, Children::Flat);
    let mut input = input.unwrap().to_vec();
    let width = (bits / 8) as usize;
    match fault {
        0 => input[32..36].copy_from_slice(&4_u32.to_le_bytes()),
        // The last offset references item zero, whose value follows all indices.
        1 => input[36..36 + width].copy_from_slice(&2_u64.to_le_bytes()[..width]),
        2 => {
            input.pop();
            input[8..16].copy_from_slice(&((4 * width - 1) as u64).to_le_bytes());
        }
        3 => {
            input.extend_from_slice(&0_u64.to_le_bytes()[..width]);
            input[8..16].copy_from_slice(&((5 * width) as u64).to_le_bytes());
        }
        _ => input[..8].copy_from_slice(&u64::MAX.to_le_bytes()),
    }
    let data = [
        LanceBuffer::from(input),
        LanceBuffer::from(b"abcd".to_vec()),
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
        assert!(
            error.to_string().contains("Dictionary")
                || error.to_string().contains("offset")
                || error.to_string().contains("payload")
        );
        assert_eq!(work.len(), 64);
        assert_eq!(pool.reserved(), 64);
    }
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn dictionary_unsigned_packed_children_cross_blocks(#[values(32, 64)] bits: u64) {
    let items = (0..1051_u64).map(|i| i * 37 % 2048).collect::<Vec<_>>();
    let expected = (0..2057)
        .map(|i| items[i * 131 % items.len()])
        .collect::<Vec<_>>();
    let (payload, encoding) = BlockDictionaryEncoder::try_new(
        bits,
        Arc::from(items),
        Box::new(OutOfLineBitpacking::new(11, 32)),
        Box::new(OutOfLineBitpacking::new(11, bits)),
    )
    .unwrap()
    .compress(DataBlock::FixedWidth(fixed(bits, &expected)))
    .unwrap();
    let payload = payload.unwrap();
    let mut input = vec![255];
    input.extend_from_slice(&payload);
    let input = LanceBuffer::from(input).slice_with_length(1, payload.len());
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(Some(&input), bits, expected.len() as u64, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn dictionary_unsigned_current_selector_round_trip(#[values(32, 64)] bits: u64) {
    let expected = (0..1024)
        .map(|i| [0, 9000, 1000, 5000, 9000, 0, 5000, 1000][i % 8])
        .collect::<Vec<_>>();
    let data = fixed(bits, &expected);
    let selected = select_block_codec(
        &data,
        &[0..512, 512..1024],
        &CompressionFieldParams::default(),
        BlockCost::default(),
        false,
    )
    .unwrap();
    assert!(matches!(
        selected.expected_encoding().compression,
        Some(Compression::Dictionary(_))
    ));
    let (payload, encoding) = selected.compress(fixed(bits, &expected[..512])).unwrap();
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut actual = Vec::new();
    codec
        .visit_unsigned(payload.as_ref(), bits, 512, &mut |v| {
            actual.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(actual, expected[..512]);
}

#[derive(Debug)]
struct VisitingItemsWithoutLookup;

impl BlockDecompressor for VisitingItemsWithoutLookup {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("allocating fallback must not run")
    }
    fn requires_payload(&self) -> bool {
        false
    }
    fn visit_unsigned(
        &self,
        _: Option<&LanceBuffer>,
        _: u64,
        _: u64,
        visitor: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        visitor(0)?;
        visitor(1)
    }
}

#[test]
fn dictionary_unsigned_requires_lookup_before_visiting_indices() {
    let codec = BlockDictionaryDecompressor::try_new(
        32,
        2,
        Box::new(ConstantBlockDecompressor::new(32, Some(0))),
        Box::new(VisitingItemsWithoutLookup),
    )
    .unwrap();
    let mut calls = 0;
    let error = codec
        .visit_unsigned(None, 32, 5, &mut |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("metadata plan"));
    assert_eq!(calls, 0);
}

#[rstest]
fn dictionary_unsigned_empty_indices_validate_items(#[values(32, 64)] bits: u64) {
    let (payload, encoding) = BlockDictionaryEncoder::try_new(
        bits,
        Arc::from([3, 1]),
        Box::<ValueEncoder>::default(),
        Box::<ValueEncoder>::default(),
    )
    .unwrap()
    .compress(DataBlock::FixedWidth(fixed(bits, &[])))
    .unwrap();
    let codec = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
    let mut calls = 0;
    codec
        .visit_unsigned(payload.as_ref(), bits, 0, &mut |_| {
            calls += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(calls, 0);
    let mut input = payload.unwrap().to_vec();
    input.pop();
    let error = codec
        .visit_unsigned(Some(&LanceBuffer::from(input)), bits, 0, &mut |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("Dictionary"));
    assert_eq!(calls, 0);
}
