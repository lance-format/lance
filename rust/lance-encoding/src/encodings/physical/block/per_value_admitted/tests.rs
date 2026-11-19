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
#[case::short_decoded_output(3)]
#[case::insufficient_declared_output(4)]
fn general_per_value_zstd_corruption_refunds(#[case] kind: u8) {
    let codec = codec(CompressionScheme::Zstd);
    // Independent writer with checksum makes corruption detectable even when
    // the damaged byte belongs to a literal and does not change decoded length.
    let mut encoder = ::zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
    encoder.include_checksum(true).unwrap();
    encoder.set_pledged_src_size(Some(4)).unwrap();
    if kind >= 3 {
        // Without frame content size, only decoding can disprove the prefix.
        encoder.include_contentsize(false).unwrap();
    }
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
        3 => frame[0] = 5,
        4 => frame[0] = 3,
        _ => unreachable!(),
    }
    // Checksum failure occurs after both the previous value and this frame's
    // output have been written. Neither output may escape the failed attempt.
    let previous = ::zstd::bulk::compress(&[17; 4097], 0).unwrap();
    let data = input(&[previous, frame], 64, 1);
    let pool = AllocationBudget::new(1024 * 1024);
    let mut work = pool.try_reserve(1024 * 1024).unwrap();
    for _ in 0..2 {
        let error = codec.try_decompress_in(&data, &mut work).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(match kind {
            0 => "prefix disagrees",
            1 => "Src size is incorrect",
            2 => "checksum",
            3 => "initializer produced 4 bytes, expected 5",
            4 => "Destination buffer is too small",
            _ => unreachable!(),
        }));
        assert_eq!(work.len(), 1024 * 1024);
        assert_eq!(pool.reserved(), 1024 * 1024);
    }
}

#[cfg(feature = "zstd")]
#[rstest]
fn mixed_zstd_frames_match_independent_decoder(
    #[values(32, 64)] bits: u8,
    #[values(0, 1)] displacement: usize,
) {
    let expected: Vec<Vec<u8>> = [0, 1, 262_145, 63, 0, 131_073, 64, 65, 32777, 0]
        .into_iter()
        .enumerate()
        .map(|(frame, len)| {
            (0..len)
                .map(|i| ((i * (frame + 1) * 31) ^ (i >> (frame % 7))) as u8)
                .collect()
        })
        .collect();
    let mut frames = Vec::new();
    for (i, value) in expected.iter().enumerate() {
        // Vary frame sizes, entropy and compression levels across one context,
        // including multi-block frames followed by small and empty frames.
        let compressed = ::zstd::bulk::compress(value, if i % 2 == 0 { 1 } else { 7 }).unwrap();
        assert_eq!(
            ::zstd::bulk::decompress(&compressed, value.len()).unwrap(),
            *value
        );
        let mut frame = if i % 2 == 0 {
            (value.len() as u64).to_le_bytes().to_vec()
        } else {
            Vec::new()
        };
        frame.extend_from_slice(&compressed);
        frames.push(frame);
    }
    // A null slot has no frame; the first empty slot keeps its real Zstd frame.
    frames.last_mut().unwrap().clear();
    let codec = codec(CompressionScheme::Zstd);
    let data = input(&frames, bits, displacement);
    let plan = codec.plan_decode(&data).unwrap();
    let pool = AllocationBudget::new(plan.workspace_bytes);
    let mut work = pool.try_reserve(plan.workspace_bytes).unwrap();
    let output = codec
        .try_decompress_in(&data, &mut work)
        .unwrap()
        .unwrap()
        .as_variable_width()
        .unwrap();
    let mut actual = Vec::new();
    variable::visit(&output, &mut |value| {
        actual.push(value.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(work.len(), plan.workspace_bytes - plan.output_bytes);
    let last = expected.concat().last().copied().unwrap();
    let alias = output.data.slice_with_length(output.data.len() - 1, 1);
    drop((output, work));
    assert_eq!(
        pool.reserved(),
        expected.iter().map(Vec::len).sum::<usize>()
    );
    assert_eq!(alias.as_ref(), &[last]);
    drop(alias);
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "zstd")]
#[test]
fn zstd_frame_cannot_use_previous_output_as_dictionary() {
    // A raw dictionary has no dictionary ID. Only independent frame state, not
    // an ID mismatch, prevents this frame from referring to preceding output.
    let mut state = 0x1234_5678_u32;
    let dictionary: Vec<u8> = (0..32768)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let value = &dictionary[dictionary.len() - 8192..];
    let mut compressor = ::zstd::bulk::Compressor::with_dictionary(3, &dictionary).unwrap();
    let dependent = compressor.compress(value).unwrap();
    assert!(::zstd::bulk::decompress(&dependent, value.len()).is_err());
    let mut independent = ::zstd::bulk::Decompressor::with_dictionary(&dictionary).unwrap();
    assert_eq!(
        independent.decompress(&dependent, value.len()).unwrap(),
        value
    );

    let preceding = ::zstd::bulk::compress(&dictionary, 1).unwrap();
    let data = input(&[preceding, dependent], 64, 1);
    let codec = codec(CompressionScheme::Zstd);
    let plan = codec.plan_decode(&data).unwrap();
    let pool = AllocationBudget::new(plan.workspace_bytes);
    let mut work = pool.try_reserve(plan.workspace_bytes).unwrap();
    let error = codec.try_decompress_in(&data, &mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert_eq!(work.len(), plan.workspace_bytes);

    // A subsequent independent attempt reinitializes its recycled storage.
    let valid = input(&[::zstd::bulk::compress(value, 1).unwrap()], 64, 1);
    let output = codec
        .try_decompress_in(&valid, &mut work)
        .unwrap()
        .unwrap()
        .as_variable_width()
        .unwrap();
    assert_eq!(output.data.as_ref(), value);
    drop((output, work));
    assert_eq!(pool.reserved(), 0);
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
