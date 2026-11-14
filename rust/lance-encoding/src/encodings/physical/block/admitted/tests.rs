// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::{
    BufferCompressor, CompressionConfig, CompressionScheme, GeneralBufferCompressor,
};
use super::*;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

fn codec(scheme: CompressionScheme) -> Box<dyn BufferCompressor> {
    GeneralBufferCompressor::get_compressor(CompressionConfig::new(scheme, None)).unwrap()
}

fn check_owned_output(scheme: CompressionScheme, len: usize, displacement: usize, shared: bool) {
    let codec = codec(scheme);
    let expected: Vec<_> = (0..len).map(|i| (i % 7) as u8).collect();
    let mut encoded = Vec::new();
    codec.compress(&expected, &mut encoded).unwrap();
    let physical = displacement + encoded.len() + 19;
    let budget = AllocationBudget::new(physical + len);
    let mut input_work = budget.try_reserve(physical).unwrap();
    let mut input = input_work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[displacement..displacement + encoded.len()].copy_from_slice(&encoded);
    let input = input.freeze();
    let sibling = input.slice(physical - 1..);
    let input = input.slice(displacement..displacement + encoded.len());
    drop(input_work);
    if len > 0 {
        let mut short = budget.try_reserve(len - 1).unwrap();
        for _ in 0..2 {
            assert!(
                codec
                    .try_decompress_exact_in(&input, len, &mut short)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(short.len(), len - 1);
            assert_eq!(budget.reserved(), physical + len - 1);
        }
    }
    let mut work = budget.try_reserve(len).unwrap();
    let decoded = codec
        .try_decompress_exact_in(&input, len, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(decoded.as_ref(), expected);
    assert_eq!(decoded.capacity(), len);
    assert_eq!(
        decoded
            .as_ref()
            .as_ptr()
            .align_offset(arrow_buffer::alloc::ALIGNMENT),
        0
    );
    assert!(work.is_empty());
    drop(input);
    assert_eq!(budget.reserved(), physical + len);
    drop(sibling);
    assert_eq!(budget.reserved(), len);
    if shared {
        let bytes = decoded.freeze();
        let last = bytes.slice(..len.min(1));
        drop(bytes);
        drop(work);
        assert_eq!(budget.reserved(), len);
        drop(last);
    } else {
        decoded.recycle(&mut work).unwrap();
        assert_eq!(work.len(), len);
        drop(work);
    }
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn uncompressed_output_tracks_its_owner(
    #[values(0, 1, 63, 65, 4096)] len: usize,
    #[values(0, 1)] displacement: usize,
    #[values(false, true)] shared: bool,
) {
    check_owned_output(CompressionScheme::None, len, displacement, shared);
}

#[cfg(feature = "lz4")]
#[rstest]
fn lz4_output_tracks_its_owner(
    #[values(0, 1, 65, 4096, 65536)] len: usize,
    #[values(0, 1)] displacement: usize,
    #[values(false, true)] shared: bool,
) {
    check_owned_output(CompressionScheme::Lz4, len, displacement, shared);
}

#[rstest]
fn uncompressed_size_error_preserves_credit(#[values(0, 2, 4, usize::MAX)] expected: usize) {
    let budget = AllocationBudget::new(8);
    let mut work = budget.try_reserve(8).unwrap();
    let error = codec(CompressionScheme::None)
        .try_decompress_exact_in(&[1, 2, 3], expected, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("Uncompressed payload has 3 bytes")
    );
    assert_eq!(work.len(), 8);
    assert_eq!(budget.reserved(), 8);
}

#[cfg(feature = "lz4")]
#[rstest]
#[case::short_prefix(vec![3, 0], 3, "length prefix")]
#[case::wrong_prefix(vec![4, 0, 0, 0, 0x30, 1, 2, 3], 3, "declares 4")]
#[case::short_output(vec![4, 0, 0, 0, 0x30, 1, 2, 3], 4, "produced 3")]
#[case::long_output(vec![2, 0, 0, 0, 0x30, 1, 2, 3], 2, "decompression error")]
#[case::truncated_literal(vec![3, 0, 0, 0, 0x30, 1, 2], 3, "decompression error")]
#[case::invalid_offset(vec![8, 0, 0, 0, 0x10, 1, 0, 0], 8, "decompression error")]
#[case::trailing_bytes(vec![3, 0, 0, 0, 0x30, 1, 2, 3, 0], 3, "decompression error")]
#[case::missing_empty_token(vec![0, 0, 0, 0], 0, "decompression error")]
#[case::corrupt_expansion(vec![255, 255, 255, 255, 0], u32::MAX as usize, "size bounds")]
fn lz4_errors_recycle_partial_output(
    #[case] input: Vec<u8>,
    #[case] expected: usize,
    #[case] message: &str,
) {
    let codec = codec(CompressionScheme::Lz4);
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_exact_in(&input, expected, &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(work.len(), 16);
        assert_eq!(budget.reserved(), 16);
    }
}

#[cfg(feature = "zstd")]
#[test]
fn unaccounted_zstd_context_never_falls_back_to_ordinary_decode() {
    let codec = codec(CompressionScheme::Zstd);
    let mut encoded = Vec::new();
    codec.compress(&[1, 2, 3], &mut encoded).unwrap();
    let budget = AllocationBudget::new(3);
    let mut work = budget.try_reserve(3).unwrap();
    let error = codec
        .try_decompress_exact_in(&encoded, 3, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no admitted exact decoder"));
    assert_eq!(work.len(), 3);
    assert_eq!(budget.reserved(), 3);
}

#[cfg(feature = "lz4")]
#[test]
fn scratch_and_selected_output_require_overlapping_credit() {
    // A complete literal frame, independent of the compressor under test.
    let input = [8, 0, 0, 0, 0x80, 0, 1, 2, 3, 4, 5, 6, 7];
    let budget = AllocationBudget::new(10);
    let mut work = budget.try_reserve(10).unwrap();
    let scratch = codec(CompressionScheme::Lz4)
        .try_decompress_exact_in(&input, 8, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 2);
    let plain = codec(CompressionScheme::None);
    assert!(
        plain
            .try_decompress_exact_in(&scratch.as_ref()[1..4], 3, &mut work)
            .unwrap()
            .is_none()
    );
    assert_eq!(work.len(), 2);
    let output = plain
        .try_decompress_exact_in(&scratch.as_ref()[3..5], 2, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(output.as_ref(), &[3, 4]);
    assert_eq!(budget.reserved(), 10);
    scratch.recycle(&mut work).unwrap();
    assert_eq!(work.len(), 8);
    let alias = output.freeze().slice(1..2);
    drop(work);
    assert_eq!(budget.reserved(), 2);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[cfg(feature = "lz4")]
#[test]
fn overlapping_decodes_resume_after_final_owner_drops() {
    let input = [4, 0, 0, 0, 0x40, 1, 2, 3, 4];
    let codec = codec(CompressionScheme::Lz4);
    let budget = AllocationBudget::new(8);
    let mut first = budget.try_reserve(4).unwrap();
    let first_output = codec
        .try_decompress_exact_in(&input, 4, &mut first)
        .unwrap()
        .unwrap()
        .freeze();
    let mut second = budget.try_reserve(4).unwrap();
    let second_output = codec
        .try_decompress_exact_in(&input, 4, &mut second)
        .unwrap()
        .unwrap();
    let last_alias = first_output.slice(3..);
    drop(first_output);
    assert!(budget.try_reserve(4).is_none());
    drop(last_alias);
    let mut third = budget.try_reserve(4).unwrap();
    let third_output = codec
        .try_decompress_exact_in(&input, 4, &mut third)
        .unwrap()
        .unwrap();
    assert_eq!(third_output.as_ref(), &[1, 2, 3, 4]);
    assert_eq!(budget.reserved(), 8);
    drop((first, second, third, second_output, third_output));
    assert_eq!(budget.reserved(), 0);
}
