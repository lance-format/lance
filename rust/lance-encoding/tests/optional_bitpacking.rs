// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![cfg(not(feature = "bitpacking"))]

use arrow_buffer::Buffer;
use lance_core::{Error, utils::allocation::AllocationBudget};
use lance_encoding::{
    buffer::LanceBuffer,
    compression::{DecompressionStrategy, DefaultDecompressionStrategy, try_bitpacking_block},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    format::{EncodingUtils, metadata::CompressiveEncoding},
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
fn disabled_bitpacking_decoder_reports_unsupported(#[case] encoding: CompressiveEncoding) {
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
