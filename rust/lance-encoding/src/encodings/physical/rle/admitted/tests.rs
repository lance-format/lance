// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::Buffer;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::format::metadata::compressive_encoding::Compression;
use crate::{
    compression::{
        BlockDecompressor, DecompressionStrategy, DefaultDecompressionStrategy,
        MiniBlockDecompressor,
    },
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
        physical::rle::{RleChildDecompressor, RleEncoder, RunLengthWidth},
    },
};

fn words(values: &[u64], width: usize) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes().into_iter().take(width))
        .collect()
}

#[rstest]
fn selected_ranges_keep_only_admitted_output(
    #[values(8, 16, 32, 64)] bits: u64,
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)]
    length_width: RunLengthWidth,
    #[values(0, 1)] prefix: usize,
) {
    let width = (bits / 8) as usize;
    let maximum = u64::MAX >> (64 - bits);
    let values = words(&[1, 7, maximum], width);
    let lengths = words(&[2, 3, 1], length_width.bytes_per_value());
    let expected = [1, 1, 7, 7, 7, maximum];
    for range in [0..6, 1..4, 2..2, 5..6] {
        let bytes = (range.end - range.start) as usize * width;
        let physical = prefix + values.len() + 3 + lengths.len() + 17;
        let budget = AllocationBudget::new(physical + bytes);
        let mut input_credit = budget.try_reserve(physical).unwrap();
        let mut backing = input_credit.try_allocate(physical).unwrap().unwrap();
        backing.as_mut()[prefix..prefix + values.len()].copy_from_slice(&values);
        let lengths_start = prefix + values.len() + 3;
        backing.as_mut()[lengths_start..lengths_start + lengths.len()].copy_from_slice(&lengths);
        let backing = Buffer::from(backing.freeze());
        let data = vec![
            LanceBuffer::from(backing.slice_with_length(prefix, values.len())),
            LanceBuffer::from(backing.slice_with_length(lengths_start, lengths.len())),
        ];
        drop(backing);
        drop(input_credit);
        let codec = RleDecompressor::with_run_length_width(bits, length_width);
        let plan = codec.plan_range(&data, 6, range.clone()).unwrap();
        assert_eq!(plan.output_bytes, bytes);
        assert_eq!(plan.workspace_bytes, bytes);
        assert_eq!(
            codec
                .admitted_output_layout()
                .unwrap()
                .payload_bytes(range.end - range.start)
                .unwrap(),
            bytes
        );
        if bytes > 0 {
            let mut short = budget.try_reserve(bytes - 1).unwrap();
            for _ in 0..2 {
                assert!(
                    codec
                        .try_decompress_range_in(&data, 6, range.clone(), &mut short)
                        .unwrap()
                        .is_none()
                );
                assert_eq!(short.len(), bytes - 1);
                assert_eq!(budget.reserved(), physical + bytes - 1);
            }
        }
        let mut work = budget.try_reserve(bytes).unwrap();
        let decoded = codec
            .try_decompress_range_in(&data, 6, range.clone(), &mut work)
            .unwrap()
            .unwrap();
        let block = decoded.as_fixed_width_ref().unwrap();
        assert_eq!(block.num_values, range.end - range.start);
        assert_eq!(
            block.data.as_ref(),
            words(&expected[range.start as usize..range.end as usize], width)
        );
        assert!(work.is_empty());
        let alias = block.data.clone();
        drop(data);
        drop(work);
        drop(decoded);
        assert_eq!(budget.reserved(), bytes);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
fn tiny_tail_does_not_allocate_the_full_expansion(
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)]
    length_width: RunLengthWidth,
) {
    let count = length_width.max_run_length();
    let data = vec![
        LanceBuffer::from(u64::MAX.to_le_bytes().to_vec()),
        LanceBuffer::from(words(&[count], length_width.bytes_per_value())),
    ];
    let codec = RleDecompressor::with_run_length_width(64, length_width);
    let budget = AllocationBudget::new(24);
    let mut work = budget.try_reserve(24).unwrap();
    let plan = codec.plan_range(&data, count, count - 3..count).unwrap();
    assert_eq!(plan.workspace_bytes, 24);
    let block = codec
        .try_decompress_range_in(&data, count, count - 3..count, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(
        block.as_fixed_width_ref().unwrap().data.as_ref(),
        &[255; 24]
    );
    assert_eq!(budget.reserved(), 24);
    drop(block);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn canonical_empty_and_chunk_boundary_selection_keep_reader_semantics() {
    let codec = RleDecompressor::new(32);
    let budget = AllocationBudget::new(12);
    let mut work = budget.try_reserve(12).unwrap();
    let empty = codec
        .try_decompress_range_in(&[], 0, 0..0, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(empty.as_fixed_width_ref().unwrap().num_values, 0);
    assert_eq!(work.len(), 12);
    let data = vec![
        LanceBuffer::from(words(&[1, 7], 4)),
        LanceBuffer::from(vec![2, 8]),
    ];
    let decoded = codec
        .try_decompress_range_in(&data, 4, 1..4, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(
        decoded.as_fixed_width_ref().unwrap().data.as_ref(),
        words(&[1, 7, 7], 4)
    );
    let ordinary = codec.decompress_range(data, 4, 1..4).unwrap();
    assert_eq!(
        decoded.as_fixed_width_ref().unwrap().data.as_ref(),
        ordinary.as_fixed_width_ref().unwrap().data.as_ref()
    );
}

#[rstest]
#[case::buffer_count(0, "exactly 2")]
#[case::partial_value(1, "Invalid buffer sizes")]
#[case::cardinality(2, "Inconsistent RLE buffers")]
#[case::partial_length(3, "Invalid buffer sizes")]
#[case::zero_length(4, "zero run length")]
#[case::too_few_rows(5, "produced 5 values")]
#[case::past_end(6, "selection")]
#[case::reversed(7, "selection")]
#[case::unsupported_width(8, "require 8, 16, 32 or 64 bits")]
#[case::child_width(9, "child widths disagree")]
fn malformed_later_input_fails_before_any_output(#[case] fault: u8, #[case] message: &str) {
    let mut codec = RleDecompressor::with_run_length_width(32, RunLengthWidth::U16);
    let mut data = vec![
        LanceBuffer::from(words(&[1, 7, 9], 4)),
        LanceBuffer::from(words(&[2, 3, 1], 2)),
    ];
    let mut range = 0..1; // The selected prefix is valid; later corruption must still fail.
    match fault {
        0 => {
            data.pop();
        }
        1 => data[0] = data[0].slice_with_length(0, 11),
        2 => data[1] = data[1].slice_with_length(0, 4),
        3 => data[1] = data[1].slice_with_length(0, 5),
        4 => data[1] = LanceBuffer::from(words(&[2, 3, 0], 2)),
        5 => data[1] = LanceBuffer::from(words(&[2, 2, 1], 2)),
        6 => range = 0..7,
        7 => range = Range { start: 4, end: 2 },
        8 => codec.bits_per_value = 3,
        9 => codec.values = RleChildDecompressor::flat(16),
        _ => unreachable!(),
    }
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, 6, range.clone(), &mut work)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(work.len(), 64);
        assert_eq!(budget.reserved(), 64);
    }
}

#[test]
fn overflowing_output_geometry_preserves_credit() {
    let codec = RleDecompressor::new(64);
    let budget = AllocationBudget::new(8);
    let mut work = budget.try_reserve(8).unwrap();
    let error = codec
        .try_decompress_range_in(&[], u64::MAX, 0..u64::MAX, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("overflows"));
    assert_eq!(work.len(), 8);
    assert_eq!(budget.reserved(), 8);
}

#[derive(Debug)]
struct ForbiddenChild;

impl BlockDecompressor for ForbiddenChild {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary child decoder must not run during admission")
    }
}

#[rstest]
fn compressed_children_refuse_before_ordinary_decode(#[values(false, true)] value_child: bool) {
    let mut codec = RleDecompressor::new(32);
    let child = RleChildDecompressor::block(
        if value_child { 32 } else { 8 },
        Box::new(ForbiddenChild),
        false,
    );
    if value_child {
        codec.values = child;
    } else {
        codec.run_lengths = child;
    }
    let budget = AllocationBudget::new(4);
    let mut work = budget.try_reserve(4).unwrap();
    let data = vec![
        LanceBuffer::from(words(&[7], 4)),
        LanceBuffer::from(vec![1]),
    ];
    let error = codec
        .try_decompress_range_in(&data, 1, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no unsigned payload visitor"));
    assert_eq!(work.len(), 4);
}

#[derive(Debug, Default)]
struct ReplayFailure(AtomicUsize);

impl BlockDecompressor for ReplayFailure {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary child decoder must not run during admission")
    }

    fn visit_unsigned(
        &self,
        _: Option<&LanceBuffer>,
        _: u64,
        _: u64,
        visitor: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        visitor(7)?;
        if self.0.fetch_add(1, Ordering::Relaxed) % 2 == 1 {
            return Err(Error::invalid_input("child failed during replay"));
        }
        visitor(9)
    }
}

#[test]
fn replay_error_recycles_exclusive_output() {
    let codec = RleDecompressor::with_child_decompressors(
        32,
        RunLengthWidth::U8,
        RleChildDecompressor::block(32, Box::<ReplayFailure>::default(), true),
        RleChildDecompressor::flat(8),
    );
    let data = [LanceBuffer::empty(), LanceBuffer::from(vec![1, 1])];
    let budget = AllocationBudget::new(8);
    let mut work = budget.try_reserve(8).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, 2, 0..2, &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("child failed during replay"));
        assert_eq!(work.len(), 8);
        assert_eq!(budget.reserved(), 8);
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[cfg(feature = "bitpacking")]
mod packed;

#[rstest]
fn current_writer_chunks_preserve_selected_rows(
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)]
    length_width: RunLengthWidth,
) {
    check_writer_chunks(length_width, false);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn current_writer_packed_chunks_preserve_selected_rows(
    #[values(RunLengthWidth::U8, RunLengthWidth::U16, RunLengthWidth::U32)]
    length_width: RunLengthWidth,
) {
    check_writer_chunks(length_width, true);
}

fn check_writer_chunks(length_width: RunLengthWidth, packed: bool) {
    let runs = if packed { 2048 } else { 40 };
    let expected: Vec<u32> = (0..runs)
        .flat_map(|r| std::iter::repeat_n(r % 7, if packed { 4 } else { 9 }))
        .collect();
    let input = DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: 32,
        num_values: expected.len() as u64,
        data: LanceBuffer::reinterpret_vec(expected.clone()),
        block_info: BlockInfo::new(),
    });
    let encoder = RleEncoder::with_child_encoding(length_width, None, None, packed);
    let (encoded, encoding) = MiniBlockCompressor::compress(
        &encoder,
        MiniBlockCompressionContext::new(0, true, true),
        input,
    )
    .unwrap();
    if packed {
        let Some(Compression::Rle(rle)) = encoding.compression.as_ref() else {
            panic!("not RLE");
        };
        assert!([&rle.values, &rle.run_lengths].iter().any(|child| matches!(
            child.as_ref().unwrap().compression.as_ref().unwrap(),
            Compression::OutOfLineBitpacking(_)
        )));
    }
    let strategy = DefaultDecompressionStrategy::default();
    let codec = strategy
        .create_miniblock_decompressor(&encoding, &strategy)
        .unwrap();
    let mut offsets = [0, 0];
    let mut processed = 0;
    for chunk in &encoded.chunks {
        let count = chunk.num_values(processed, encoded.num_values);
        let data: Vec<_> = chunk
            .buffer_sizes
            .iter()
            .enumerate()
            .map(|(i, size)| {
                let value = encoded.data[i].slice_with_length(offsets[i], *size as usize);
                offsets[i] += *size as usize;
                value
            })
            .collect();
        let range = count.min(1)..count;
        let bytes = (range.end - range.start) as usize * 4;
        let budget = AllocationBudget::new(bytes);
        let mut work = budget.try_reserve(bytes).unwrap();
        let decoded = codec
            .try_decompress_range_in(&data, count, range.clone(), &mut work)
            .unwrap()
            .unwrap();
        let expected =
            &expected[(processed + range.start) as usize..(processed + range.end) as usize];
        assert_eq!(
            decoded.as_fixed_width_ref().unwrap().data.as_ref(),
            bytemuck::cast_slice::<u32, u8>(expected)
        );
        processed += count;
    }
    assert_eq!(processed, expected.len() as u64);
}
