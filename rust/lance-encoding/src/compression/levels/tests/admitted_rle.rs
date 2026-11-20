// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::{BlockCompressor, DecompressionStrategy, DefaultDecompressionStrategy},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::rle::{RleChildDecompressor, RleDecompressor, RleEncoder, RunLengthWidth},
    format::EncodingUtils,
};

fn frame(values: &[u16], lengths: &[u32], width: usize) -> Vec<u8> {
    let mut data = ((values.len() * 2) as u64).to_le_bytes().to_vec();
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    for length in lengths {
        data.extend_from_slice(&length.to_le_bytes()[..width]);
    }
    data
}

fn codec(width: usize) -> Box<dyn BlockDecompressor> {
    DefaultDecompressionStrategy::default()
        .create_block_decompressor(&EncodingUtils::rle(
            EncodingUtils::flat(16, None),
            EncodingUtils::flat((width * 8) as u64, None),
        ))
        .unwrap()
}

#[rstest]
fn direct_expansion_preserves_values_and_owners(
    #[values(1, 2, 4)] width: usize,
    #[values(false, true)] unaligned: bool,
    #[values(0, 1, 255)] first_run: u32,
) {
    let (values, lengths, expected) = if first_run == 0 {
        (vec![], vec![], vec![])
    } else {
        let mut expected = vec![0x1234; first_run as usize];
        expected.extend_from_slice(&[0, 0, u16::MAX]);
        (vec![0x1234, 0, u16::MAX], vec![first_run, 2, 1], expected)
    };
    check(
        codec(width).as_ref(),
        &frame(&values, &lengths, width),
        &expected,
        unaligned,
        0,
    );
}

#[rstest]
#[case::u16(2, 256)]
#[case::u32(4, 65_536)]
fn wide_lengths_do_not_truncate(#[case] width: usize, #[case] length: u32) {
    check(
        codec(width).as_ref(),
        &frame(&[7], &[length], width),
        &vec![7; length as usize],
        true,
        0,
    );
}

#[rstest]
fn current_writer_frame_uses_admitted_reader(
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)] width: RunLengthWidth,
) {
    let mut expected = vec![2_u16; 300];
    expected.extend_from_slice(&[0, 1, 1]);
    let block = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(expected.clone()),
        bits_per_value: 16,
        num_values: expected.len() as u64,
        block_info: BlockInfo::new(),
    });
    let (encoded, description) = RleEncoder::with_run_length_width(width)
        .compress(block)
        .unwrap();
    let codec = DefaultDecompressionStrategy::default()
        .create_block_decompressor(&description)
        .unwrap();
    check(codec.as_ref(), &encoded.unwrap(), &expected, true, 0);
}

#[rstest]
#[case::short_header(vec![0; 7], 1, 0, "8-byte header")]
#[case::overflow_header(u64::MAX.to_le_bytes().to_vec(), 1, 1, "frame length")]
#[case::short_values([4_u64.to_le_bytes().as_slice(), &[1, 0]].concat(), 1, 1, "frame length")]
#[case::odd_values([1_u64.to_le_bytes().as_slice(), &[1, 1]].concat(), 1, 1, "incomplete u16")]
#[case::missing_length(frame(&[1], &[], 1), 1, 1, "run-length bytes")]
#[case::short_length(frame(&[1], &[1], 1), 2, 1, "run-length bytes")]
#[case::extra_length(frame(&[1], &[1, 1], 2), 2, 1, "run-length bytes")]
#[case::zero_length(frame(&[1], &[0], 4), 4, 1, "length is zero")]
#[case::underfull(frame(&[1], &[1], 4), 4, 2, "cover 1 values, expected 2")]
#[case::overfull(frame(&[1], &[2], 4), 4, 1, "exceed expected count")]
#[case::empty_nonzero(frame(&[], &[], 1), 1, 1, "cover 0 values, expected 1")]
#[case::run_sum(frame(&[1, 2], &[u32::MAX, 1], 4), 4, u32::MAX as u64, "exceed expected count")]
#[case::output_overflow(frame(&[], &[], 1), 1, u64::MAX, "output length overflows")]
fn malformed_frame_is_rejected_without_expansion(
    #[case] bytes: Vec<u8>,
    #[case] width: usize,
    #[case] count: u64,
    #[case] message: &str,
) {
    let input = LanceBuffer::from(bytes);
    let error = codec(width).plan_levels(&input, count).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
}

#[test]
fn wrong_value_width_is_rejected() {
    let error = RleDecompressor::new(32)
        .plan_levels(&LanceBuffer::empty(), 0)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("require 16-bit values"));
}

#[derive(Debug)]
struct UnadmittedChild;

impl BlockDecompressor for UnadmittedChild {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> lance_core::Result<DataBlock> {
        panic!("ordinary child decoder must not run during admission");
    }
}

#[rstest]
fn compressed_child_is_refused_before_decode(#[values(false, true)] compressed_values: bool) {
    let (values, lengths) = if compressed_values {
        (
            RleChildDecompressor::block(16, Box::new(UnadmittedChild), true),
            RleChildDecompressor::flat(8),
        )
    } else {
        (
            RleChildDecompressor::flat(16),
            RleChildDecompressor::block(8, Box::new(UnadmittedChild), true),
        )
    };
    let codec = RleDecompressor::with_child_decompressors(16, RunLengthWidth::U8, values, lengths);
    let error = codec.plan_levels(&LanceBuffer::empty(), 0).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("require flat"));
}
