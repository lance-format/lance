// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::{CompressedBufferEncoder, CompressionConfig};
use super::*;
use crate::compression::VariablePerValueDecompressor;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

fn codec(scheme: CompressionScheme) -> CompressedBufferEncoder {
    CompressedBufferEncoder::try_new(CompressionConfig::new(scheme, None)).unwrap()
}

fn input(frames: &[Vec<u8>], bits: u8, displacement: usize) -> VariableWidthBlock {
    let mut bytes = vec![0; displacement];
    let mut offsets = vec![0; displacement];
    let width = usize::from(bits / 8);
    offsets.extend_from_slice(&0_u64.to_le_bytes()[..width]);
    for frame in frames {
        bytes.extend_from_slice(frame);
        let offset = bytes.len() - displacement;
        offsets.extend_from_slice(&(offset as u64).to_le_bytes()[..width]);
    }
    VariableWidthBlock {
        data: LanceBuffer::from(bytes)
            .slice_with_length(displacement, frames.iter().map(Vec::len).sum()),
        offsets: LanceBuffer::from(offsets)
            .slice_with_length(displacement, (frames.len() + 1) * width),
        bits_per_offset: bits,
        num_values: frames.len() as u64,
        block_info: BlockInfo::new(),
    }
}

#[rstest]
#[case::none(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
#[cfg_attr(feature = "zstd", case::zstd(CompressionScheme::Zstd))]
fn general_per_value_owner_and_refusal(
    #[case] scheme: CompressionScheme,
    #[values(32, 64)] bits: u8,
    #[values(0, 1)] displacement: usize,
) {
    let codec = codec(scheme);
    let expected = [vec![7; 32776], vec![], vec![1, 2, 3, 4], vec![]];
    let frames: Vec<_> = expected
        .iter()
        .map(|value| {
            let mut frame = Vec::new();
            codec.compressor.compress(value, &mut frame).unwrap();
            frame
        })
        .collect();
    let mut frames = frames;
    // Null slot has no frame, while an empty nonnull value has a real empty frame.
    frames[3].clear();
    let data = input(&frames, bits, displacement);
    let plan = codec.plan_decode(&data).unwrap();
    let expected_values = expected.concat();
    assert_eq!(
        plan.output_bytes,
        expected_values.len() + 5 * usize::from(bits / 8)
    );
    let pool = AllocationBudget::new(plan.workspace_bytes);
    let mut short = pool.try_reserve(plan.workspace_bytes - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_in(&data, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), plan.workspace_bytes - 1);
    }
    drop(short);
    let mut work = pool.try_reserve(plan.workspace_bytes).unwrap();
    let result = codec
        .try_decompress_in(&data, &mut work)
        .unwrap()
        .unwrap()
        .as_variable_width()
        .unwrap();
    assert_eq!(result.data.as_ref(), expected_values);
    let mut actual = Vec::new();
    variable::visit(&result, &mut |value| {
        actual.push(value.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(work.len(), plan.workspace_bytes - plan.output_bytes);
    let alias = result.data.slice_with_length(1, 1);
    drop(result);
    drop(work);
    assert_eq!(pool.reserved(), expected_values.len());
    assert_eq!(alias.as_ref(), &[7]);
    drop(alias);
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "zstd")]
#[rstest]
#[case::prefix_mismatch(0)]
#[case::truncated(1)]
#[case::checksum(2)]
fn general_per_value_zstd_corruption_refunds(#[case] kind: u8) {
    let codec = codec(CompressionScheme::Zstd);
    // Independent writer with checksum makes corruption detectable even when
    // the damaged byte belongs to a literal and does not change decoded length.
    let mut encoder = ::zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
    encoder.include_checksum(true).unwrap();
    encoder.set_pledged_src_size(Some(4)).unwrap();
    std::io::Write::write_all(&mut encoder, &[1, 2, 3, 4]).unwrap();
    let mut frame = 4_u64.to_le_bytes().to_vec();
    frame.extend_from_slice(&encoder.finish().unwrap());
    match kind {
        0 => frame[0] = 5,
        1 => {
            frame.pop();
        }
        2 => {
            *frame.last_mut().unwrap() ^= 1;
        }
        _ => unreachable!(),
    }
    let data = input(&[frame], 64, 1);
    let pool = AllocationBudget::new(1024 * 1024);
    let mut work = pool.try_reserve(1024 * 1024).unwrap();
    for _ in 0..2 {
        assert!(codec.try_decompress_in(&data, &mut work).is_err());
        assert_eq!(work.len(), 1024 * 1024);
        assert_eq!(pool.reserved(), 1024 * 1024);
    }
}

#[cfg(feature = "zstd")]
#[test]
fn general_per_value_raw_zstd_has_bounded_context() {
    let raw = ::zstd::bulk::compress(&[9; 4096], 0).unwrap();
    let data = input(&[raw], 32, 0);
    let codec = codec(CompressionScheme::Zstd);
    let plan = codec.plan_decode(&data).unwrap();
    assert!(plan.workspace_bytes > plan.output_bytes);
    let pool = AllocationBudget::new(plan.workspace_bytes);
    let mut work = pool.try_reserve(plan.workspace_bytes).unwrap();
    let output = codec
        .try_decompress_in(&data, &mut work)
        .unwrap()
        .unwrap()
        .as_variable_width()
        .unwrap();
    assert_eq!(output.data.as_ref(), &[9; 4096]);
    drop(work);
    assert_eq!(pool.reserved(), 4096 + 8);
    drop(output);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn general_per_value_checks_all_offsets_before_refusal() {
    let mut data = input(&[vec![1], vec![2]], 32, 0);
    data.offsets = LanceBuffer::reinterpret_vec(vec![0_u32, 1, 3]);
    let pool = AllocationBudget::new(0);
    let mut work = pool.try_reserve(0).unwrap();
    assert!(
        codec(CompressionScheme::None)
            .try_decompress_in(&data, &mut work)
            .is_err()
    );
    assert_eq!(pool.reserved(), 0);
}
