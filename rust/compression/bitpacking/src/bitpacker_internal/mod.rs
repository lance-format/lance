// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Lance-owned u32 SIMD bitpacking kernels.
//
// This is adapted from the MIT-licensed `bitpacking` crate so Lance can keep the
// hot FTS posting-list bitpacking implementation inside lance-bitpacking while
// preserving byte compatibility with the existing 4x format.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::use_self)]

#[macro_use]
mod macros;

mod bitpacker4x;
mod bitpacker8x;

pub use bitpacker4x::BitPacker4x;
pub use bitpacker8x::BitPacker8x;

pub(crate) trait Available {
    fn available() -> bool;
}

pub(crate) trait UnsafeBitPacker {
    const BLOCK_LEN: usize;

    unsafe fn compress(decompressed: &[u32], compressed: &mut [u8], num_bits: u8) -> usize;

    unsafe fn compress_sorted(
        initial: u32,
        decompressed: &[u32],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize;

    unsafe fn compress_strictly_sorted(
        initial: Option<u32>,
        decompressed: &[u32],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize;

    unsafe fn decompress(compressed: &[u8], decompressed: &mut [u32], num_bits: u8) -> usize;

    unsafe fn decompress_sorted(
        initial: u32,
        compressed: &[u8],
        decompressed: &mut [u32],
        num_bits: u8,
    ) -> usize;

    unsafe fn decompress_strictly_sorted(
        initial: Option<u32>,
        compressed: &[u8],
        decompressed: &mut [u32],
        num_bits: u8,
    ) -> usize;

    unsafe fn num_bits(decompressed: &[u32]) -> u8;

    unsafe fn num_bits_sorted(initial: u32, decompressed: &[u32]) -> u8;

    unsafe fn num_bits_strictly_sorted(initial: Option<u32>, decompressed: &[u32]) -> u8;
}

/// Block bitpacker for fixed-size `u32` blocks.
///
/// Implementations own runtime SIMD dispatch and use caller-provided buffers.
/// Packed bytes are stable for a given implementation and bit width.
pub trait BitPacker: Sized + Clone + Copy {
    /// Number of `u32` values in one physical block.
    const BLOCK_LEN: usize;

    /// Select the best supported implementation for the current CPU.
    ///
    /// Lance uses SIMD backends when available and falls back to a scalar
    /// backend otherwise, matching the existing allocation-free call shape used
    /// by the upstream `bitpacking` crate.
    fn new() -> Self;

    /// Compress one full block of raw values into `compressed`.
    fn compress(&self, decompressed: &[u32], compressed: &mut [u8], num_bits: u8) -> usize;

    /// Delta-compress one full non-decreasing block into `compressed`.
    fn compress_sorted(
        &self,
        initial: u32,
        decompressed: &[u32],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize;

    /// Delta-compress one full strictly increasing block into `compressed`.
    fn compress_strictly_sorted(
        &self,
        initial: Option<u32>,
        decompressed: &[u32],
        compressed: &mut [u8],
        num_bits: u8,
    ) -> usize;

    /// Decompress one raw block into `decompressed`.
    fn decompress(&self, compressed: &[u8], decompressed: &mut [u32], num_bits: u8) -> usize;

    /// Decompress one delta-compressed non-decreasing block.
    fn decompress_sorted(
        &self,
        initial: u32,
        compressed: &[u8],
        decompressed: &mut [u32],
        num_bits: u8,
    ) -> usize;

    /// Decompress one delta-compressed strictly increasing block.
    fn decompress_strictly_sorted(
        &self,
        initial: Option<u32>,
        compressed: &[u8],
        decompressed: &mut [u32],
        num_bits: u8,
    ) -> usize;

    /// Return the minimum bit width needed to represent a full raw block.
    fn num_bits(&self, decompressed: &[u32]) -> u8;

    /// Return the minimum bit width needed to represent deltas in a full block.
    fn num_bits_sorted(&self, initial: u32, decompressed: &[u32]) -> u8;

    /// Return the minimum bit width needed to represent strict deltas in a full block.
    fn num_bits_strictly_sorted(&self, initial: Option<u32>, decompressed: &[u32]) -> u8;

    /// Return the byte size of one compressed block at `num_bits`.
    #[must_use]
    fn compressed_block_size(num_bits: u8) -> usize {
        Self::BLOCK_LEN * num_bits as usize / 8
    }
}

#[inline]
fn most_significant_bit(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use rstest::rstest;

    use super::{BitPacker, BitPacker4x, BitPacker8x};

    /// A call that breaks one of the bounds a kernel checks.
    #[derive(Clone, Copy, Debug)]
    enum BadCall {
        CompressShortInput(u8),
        CompressShortOutput(u8),
        CompressSortedShortOutput,
        CompressStrictlySortedShortOutput,
        CompressWideNumBits,
        DecompressShortInput(u8),
        DecompressShortOutput,
        DecompressSortedShortOutput,
        DecompressStrictlySortedShortOutput,
        DecompressWideNumBits,
        NumBitsShortInput,
        NumBitsSortedShortInput,
        NumBitsStrictlySortedShortInput,
    }

    fn make_bad_call<B: BitPacker>(packer: B, bad_call: BadCall) {
        // Each short buffer is a prefix of one twice as long, so a kernel that lost its check
        // reads or writes the spare tail and returns instead of corrupting the heap.
        let block = B::BLOCK_LEN;
        let input = vec![0u32; 2 * block];
        let mut packed = vec![0u8; 2 * B::compressed_block_size(32)];
        let mut output = vec![0u32; 2 * block];
        let short_packed = |num_bits| B::compressed_block_size(num_bits) - 1;
        match bad_call {
            BadCall::CompressShortInput(num_bits) => {
                packer.compress(&input[..block - 1], &mut packed, num_bits);
            }
            BadCall::CompressShortOutput(num_bits) => {
                let packed = &mut packed[..short_packed(num_bits)];
                packer.compress(&input[..block], packed, num_bits);
            }
            BadCall::CompressSortedShortOutput => {
                packer.compress_sorted(0, &input[..block], &mut packed[..short_packed(5)], 5);
            }
            BadCall::CompressStrictlySortedShortOutput => {
                let packed = &mut packed[..short_packed(5)];
                packer.compress_strictly_sorted(None, &input[..block], packed, 5);
            }
            BadCall::CompressWideNumBits => {
                packer.compress(&input[..block], &mut packed, 33);
            }
            BadCall::DecompressShortInput(num_bits) => {
                packer.decompress(&packed[..short_packed(num_bits)], &mut output, num_bits);
            }
            BadCall::DecompressShortOutput => {
                packer.decompress(&packed, &mut output[..block - 1], 5);
            }
            BadCall::DecompressSortedShortOutput => {
                packer.decompress_sorted(0, &packed, &mut output[..block - 1], 5);
            }
            BadCall::DecompressStrictlySortedShortOutput => {
                packer.decompress_strictly_sorted(None, &packed, &mut output[..block - 1], 5);
            }
            BadCall::DecompressWideNumBits => {
                packer.decompress(&packed, &mut output, 33);
            }
            BadCall::NumBitsShortInput => {
                packer.num_bits(&input[..block - 1]);
            }
            BadCall::NumBitsSortedShortInput => {
                packer.num_bits_sorted(0, &input[..block - 1]);
            }
            BadCall::NumBitsStrictlySortedShortInput => {
                packer.num_bits_strictly_sorted(None, &input[..block - 1]);
            }
        }
    }

    fn assert_rejected<B: BitPacker>(backend: &str, packer: B, bad_call: BadCall, expected: &str) {
        let Err(payload) = catch_unwind(AssertUnwindSafe(|| make_bad_call(packer, bad_call)))
        else {
            panic!("{bad_call:?} on {backend} did not panic");
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            message.contains(expected),
            "{bad_call:?} on {backend} panicked with {message:?}, expected {expected:?}"
        );
    }

    #[rstest]
    #[case::compress_short_input(BadCall::CompressShortInput(5), "Input block too small")]
    #[case::compress_short_input_32_bits(BadCall::CompressShortInput(32), "Input block too small")]
    #[case::compress_short_output(BadCall::CompressShortOutput(5), "Output array too small")]
    #[case::compress_short_output_32_bits(
        BadCall::CompressShortOutput(32),
        "Output array too small"
    )]
    #[case::compress_sorted_short_output(
        BadCall::CompressSortedShortOutput,
        "Output array too small"
    )]
    #[case::compress_strictly_sorted_short_output(
        BadCall::CompressStrictlySortedShortOutput,
        "Output array too small"
    )]
    #[case::compress_wide_num_bits(BadCall::CompressWideNumBits, "Num bits must be <= 32")]
    #[case::decompress_short_input(
        BadCall::DecompressShortInput(5),
        "Compressed array seems too small"
    )]
    #[case::decompress_short_input_32_bits(
        BadCall::DecompressShortInput(32),
        "Compressed array seems too small"
    )]
    #[case::decompress_short_output(
        BadCall::DecompressShortOutput,
        "The output array is not large enough"
    )]
    #[case::decompress_sorted_short_output(
        BadCall::DecompressSortedShortOutput,
        "The output array is not large enough"
    )]
    #[case::decompress_strictly_sorted_short_output(
        BadCall::DecompressStrictlySortedShortOutput,
        "The output array is not large enough"
    )]
    #[case::decompress_wide_num_bits(BadCall::DecompressWideNumBits, "Num bits must be <= 32")]
    #[case::num_bits_short_input(BadCall::NumBitsShortInput, "`decompressed`'s len is not")]
    #[case::num_bits_sorted_short_input(
        BadCall::NumBitsSortedShortInput,
        "`decompressed`'s len is not"
    )]
    #[case::num_bits_strictly_sorted_short_input(
        BadCall::NumBitsStrictlySortedShortInput,
        "`decompressed`'s len is not"
    )]
    fn kernels_reject_out_of_bounds_calls(#[case] bad_call: BadCall, #[case] expected: &str) {
        let backends_4x = [
            ("4x scalar", Some(BitPacker4x::new_scalar())),
            ("4x SSE3", BitPacker4x::new_sse()),
            ("4x NEON", BitPacker4x::new_neon()),
        ];
        for (backend, packer) in backends_4x {
            if let Some(packer) = packer {
                assert_rejected(backend, packer, bad_call, expected);
            }
        }
        let backends_8x = [
            ("8x scalar", Some(BitPacker8x::new_scalar())),
            ("8x AVX2", BitPacker8x::new_avx2()),
            ("8x NEON", BitPacker8x::new_neon()),
        ];
        for (backend, packer) in backends_8x {
            if let Some(packer) = packer {
                assert_rejected(backend, packer, bad_call, expected);
            }
        }
    }
}
