// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(not(feature = "bitpacking"))]

use arrow_buffer::Buffer;
use lance_core::{Error, utils::allocation::AllocationBudget};
use lance_encoding::{
    buffer::LanceBuffer,
    compression::{
        DecompressionStrategy, DefaultDecompressionStrategy, MiniBlockDecompressor,
        try_bitpacking_block,
    },
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::alp::AlpMiniBlockDecompressor,
    format::{EncodingUtils, metadata},
};
use rstest::rstest;

#[test]
fn disabled_bitpacking_is_not_selected() {
    let data = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(vec![1_u16; 4]),
        bits_per_value: 16,
        num_values: 4,
        block_info: BlockInfo::new(),
    });
    // An unavailable codec must not require its selection statistics.
    assert!(try_bitpacking_block(&data).is_none());
}

#[rstest]
#[case::inline(EncodingUtils::inline_bitpacking(16, None))]
#[case::out_of_line(EncodingUtils::out_of_line_bitpacking(16, EncodingUtils::flat(1, None)))]
fn disabled_bitpacking_decoder_reports_unsupported(
    #[case] encoding: metadata::CompressiveEncoding,
) {
    let error = DefaultDecompressionStrategy::default()
        .create_block_decompressor(&encoding)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("not built with bitpacking support")
    );
}

#[rstest]
fn disabled_alp_preserves_empty_and_buffer_validation(#[values(32, 64)] bits: u32) {
    let codec = AlpMiniBlockDecompressor::from_description(&metadata::Alp {
        bits_per_value: bits,
    })
    .unwrap();
    let empty = codec.decompress(vec![], 0).unwrap();
    let empty = empty.as_fixed_width().unwrap();
    assert_eq!(empty.num_values, 0);
    assert_eq!(empty.bits_per_value, u64::from(bits));

    let error = codec
        .decompress(vec![LanceBuffer::empty(); 3], 1)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("expects 4 buffers, got 3"));

    let error = codec
        .decompress(vec![LanceBuffer::empty(); 4], 1)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("ALP requires bitpacking support")
    );
}

#[test]
fn flat_level_admission_remains_available_without_optional_codecs() {
    let codec = DefaultDecompressionStrategy::default()
        .create_block_decompressor(&EncodingUtils::flat(16, None))
        .unwrap();
    let budget = AllocationBudget::new(12);
    let mut work = budget.try_reserve(12).unwrap();
    let mut input = work.try_allocate(6).unwrap().unwrap();
    input
        .as_mut()
        .copy_from_slice(bytemuck::cast_slice(&[1_u16, 0, 1]));
    let input = LanceBuffer::from(Buffer::from(input.freeze()));
    let plan = codec.plan_levels(&input, 3).unwrap();
    let output = plan.try_decode_in(&mut work).unwrap().unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u16>(output.as_ref()), &[1, 0, 1]);
    drop(input);
    drop(work);
    assert_eq!(budget.reserved(), 6);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn rle_level_admission_requires_no_optional_codec(#[values(8, 16, 32)] width: u64) {
    let codec = DefaultDecompressionStrategy::default()
        .create_block_decompressor(&EncodingUtils::rle(
            EncodingUtils::flat(16, None),
            EncodingUtils::flat(width, None),
        ))
        .unwrap();
    let mut bytes = 4_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[1, 0, 0, 0]);
    for length in [2_u32, 1] {
        bytes.extend_from_slice(&length.to_le_bytes()[..(width / 8) as usize]);
    }
    let budget = AllocationBudget::new(bytes.len() + 6);
    let mut work = budget.try_reserve(bytes.len() + 6).unwrap();
    let mut input = work.try_allocate(bytes.len()).unwrap().unwrap();
    input.as_mut().copy_from_slice(&bytes);
    let input = LanceBuffer::from(Buffer::from(input.freeze()));
    let output = codec
        .plan_levels(&input, 3)
        .unwrap()
        .try_decode_in(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u16>(output.as_ref()), &[1, 1, 0]);
    assert_eq!(work.len(), 0);
    drop(input);
    drop(work);
    assert_eq!(budget.reserved(), 6);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}
