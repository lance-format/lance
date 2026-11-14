// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::{BlockCompressor, DecompressionStrategy, DefaultDecompressionStrategy},
    encodings::physical::bitpacking::OutOfLineBitpacking,
    format::{EncodingUtils, metadata::CompressiveEncoding},
};

fn child(values: &[u64], bits: u64, packed: bool) -> (LanceBuffer, CompressiveEncoding) {
    let bytes = LanceBuffer::from(words(values, (bits / 8) as usize));
    if !packed {
        return (bytes, EncodingUtils::flat(bits, None));
    }
    let (payload, encoding) = BlockCompressor::compress(
        &OutOfLineBitpacking::new(3, bits),
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: bytes,
            bits_per_value: bits,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        }),
    )
    .unwrap();
    (payload.unwrap(), encoding)
}

fn fixture(
    values: &[u64],
    lengths: &[u64],
    bits: u64,
    length_width: RunLengthWidth,
    packed_values: bool,
) -> (Box<dyn MiniBlockDecompressor>, Vec<LanceBuffer>) {
    let (values, value_encoding) = child(values, bits, packed_values);
    let (lengths, length_encoding) = child(lengths, length_width.bits_per_value(), !packed_values);
    let strategy = DefaultDecompressionStrategy::default();
    let codec = strategy
        .create_miniblock_decompressor(
            &EncodingUtils::rle(value_encoding, length_encoding),
            &strategy,
        )
        .unwrap();
    (codec, vec![values, lengths])
}

#[rstest]
fn borrowed_children_keep_exact_budget_and_shared_owner(
    #[values(8, 16, 32, 64)] bits: u64,
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)]
    length_width: RunLengthWidth,
    #[values(false, true)] packed_values: bool,
    #[values(0, 1)] displacement: usize,
) {
    let values: Vec<_> = (0..1043).map(|i| i % 7).collect();
    let lengths: Vec<_> = (0..1043).map(|i| i % 3 + 1).collect();
    let expected: Vec<_> = values
        .iter()
        .zip(&lengths)
        .flat_map(|(&value, &length)| std::iter::repeat_n(value, length as usize))
        .collect();
    let count = expected.len() as u64;
    let (codec, data) = fixture(&values, &lengths, bits, length_width, packed_values);
    for range in [0..0, 1..7, count - 3..count, 0..count] {
        let output_bytes = (range.end - range.start) as usize * (bits / 8) as usize;
        let second = displacement + data[0].len() + 3;
        let physical = second + data[1].len() + 19;
        let budget = AllocationBudget::new(physical + output_bytes);
        let mut input = budget.try_reserve(physical).unwrap();
        let mut backing = input.try_allocate(physical).unwrap().unwrap();
        backing.as_mut()[displacement..displacement + data[0].len()].copy_from_slice(&data[0]);
        backing.as_mut()[second..second + data[1].len()].copy_from_slice(&data[1]);
        let backing = Buffer::from(backing.freeze());
        let sibling = backing.slice_with_length(physical - 1, 1);
        let charged = [
            LanceBuffer::from(backing.slice_with_length(displacement, data[0].len())),
            LanceBuffer::from(backing.slice_with_length(second, data[1].len())),
        ];
        drop(backing);
        drop(input);
        let plan = codec.plan_range(&charged, count, range.clone()).unwrap();
        assert_eq!(plan.output_bytes, output_bytes);
        assert_eq!(plan.workspace_bytes, output_bytes);
        if output_bytes > 0 {
            let mut short = budget.try_reserve(output_bytes - 1).unwrap();
            for _ in 0..2 {
                assert!(
                    codec
                        .try_decompress_range_in(&charged, count, range.clone(), &mut short)
                        .unwrap()
                        .is_none()
                );
                assert_eq!(short.len(), output_bytes - 1);
                assert_eq!(budget.reserved(), physical + output_bytes - 1);
            }
        }
        let mut work = budget.try_reserve(output_bytes).unwrap();
        let decoded = codec
            .try_decompress_range_in(&charged, count, range.clone(), &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(
            decoded.as_fixed_width_ref().unwrap().data.as_ref(),
            words(
                &expected[range.start as usize..range.end as usize],
                (bits / 8) as usize
            )
        );
        assert!(work.is_empty());
        drop(charged);
        assert_eq!(budget.reserved(), physical + output_bytes);
        drop(sibling);
        assert_eq!(budget.reserved(), output_bytes);
        drop(decoded);
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::truncated(0)]
#[case::extra_byte(1)]
#[case::late_zero_length(2)]
#[case::partial_flat_sibling(3)]
fn malformed_children_precede_empty_or_prefix_output(
    #[case] fault: u8,
    #[values(false, true)] packed_values: bool,
    #[values(false, true)] empty: bool,
) {
    let values = vec![7; 1043];
    let mut lengths = vec![2; 1043];
    if fault == 2 {
        lengths[1042] = 0;
    }
    let (codec, mut data) = fixture(&values, &lengths, 32, RunLengthWidth::U8, packed_values);
    let packed = usize::from(!packed_values);
    match fault {
        0 => data[packed] = data[packed].slice_with_length(0, data[packed].len() - 1),
        1 => {
            let mut bytes = data[packed].to_vec();
            bytes.push(0);
            data[packed] = LanceBuffer::from(bytes);
        }
        3 => {
            // Choose a partial Flat word even when the Flat side is the u8 lengths.
            // In that case dropping a whole entry makes the packed cardinality invalid.
            let flat = 1 - packed;
            data[flat] = data[flat].slice_with_length(0, data[flat].len() - 1);
        }
        _ => {}
    }
    let budget = AllocationBudget::new(4);
    let mut work = budget.try_reserve(4).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, 2086, 0..u64::from(!empty), &mut work)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(
            error.to_string().contains(if fault == 2 {
                "zero run length"
            } else if fault == 3 && !packed_values {
                "partial value"
            } else {
                "payload length"
            }),
            "{error}"
        );
        assert_eq!(work.len(), 4);
        assert_eq!(budget.reserved(), 4);
    }
}
