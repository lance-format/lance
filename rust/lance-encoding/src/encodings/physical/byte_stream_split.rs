// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Byte Stream Split (BSS) Miniblock Format
//!
//! Byte Stream Split is a data transformation technique that improves compression
//! by reorganizing multi-byte values to group bytes from the same position together.
//! This is particularly effective for data where some byte positions have low entropy.
//!
//! ## How It Works
//!
//! BSS splits multi-byte values by byte position, creating separate streams
//! for each byte position across all values. This transformation is most beneficial
//! when certain byte positions have low entropy (e.g., high-order bytes that are
//! mostly zeros, sign-extended bytes, or floating-point sign/exponent bytes that
//! cluster around common values).
//!
//! ### Example
//!
//! Input data (f32): `[1.0, 2.0, 3.0, 4.0]`
//!
//! In little-endian bytes:
//! - 1.0 = `[00, 00, 80, 3F]`
//! - 2.0 = `[00, 00, 00, 40]`
//! - 3.0 = `[00, 00, 40, 40]`
//! - 4.0 = `[00, 00, 80, 40]`
//!
//! After BSS transformation:
//! - Byte stream 0: `[00, 00, 00, 00]` (all first bytes)
//! - Byte stream 1: `[00, 00, 00, 00]` (all second bytes)
//! - Byte stream 2: `[80, 00, 40, 80]` (all third bytes)
//! - Byte stream 3: `[3F, 40, 40, 40]` (all fourth bytes)
//!
//! Output: `[00, 00, 00, 00, 00, 00, 00, 00, 80, 00, 40, 80, 3F, 40, 40, 40]`
//!
//! ## Compression Benefits
//!
//! BSS itself doesn't compress data - it reorders it. The compression benefit
//! comes when BSS is combined with general-purpose compression (e.g., LZ4):
//!
//! 1. **Timestamps**: Sequential timestamps have similar high-order bytes
//! 2. **Sensor data**: Readings often vary in a small range, sharing exponent bits
//! 3. **Financial data**: Prices may cluster around certain values
//!
//! ## Supported Types
//!
//! - 32-bit floating point (f32)
//! - 64-bit floating point (f64)
//!
//! ## Chunk Handling
//!
//! - Maximum chunk size depends on data type:
//!   - f32: 1024 values (4KB per chunk)
//!   - f64: 512 values (4KB per chunk)
//! - All chunks share a single global buffer
//! - Non-last chunks always contain power-of-2 values

use std::fmt::Debug;

use arrow_array::{cast::AsArray, types::UInt64Type};
use lance_core::Result;
#[cfg(target_arch = "x86_64")]
use lance_core::utils::cpu::CpuFeatures;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::compression_config::BssMode;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext, MiniBlockCompressor,
};
use crate::format::EncodingUtils;
use crate::format::metadata::CompressiveEncoding;
use crate::statistics::{GetStat, Stat};

/// Byte Stream Split encoder for floating point values
///
/// This encoding splits floating point values by byte position and stores
/// each byte stream separately. This improves compression ratios for
/// floating point data with similar patterns.
#[derive(Debug, Clone)]
pub struct ByteStreamSplitEncoder {
    bits_per_value: usize,
}

impl ByteStreamSplitEncoder {
    pub fn new(bits_per_value: usize) -> Self {
        assert!(
            bits_per_value == 32 || bits_per_value == 64,
            "ByteStreamSplit only supports 32-bit (f32) or 64-bit (f64) values"
        );
        Self { bits_per_value }
    }

    fn bytes_per_value(&self) -> usize {
        self.bits_per_value / 8
    }

    fn max_chunk_size(&self) -> usize {
        // For ByteStreamSplit, total bytes = bytes_per_value * chunk_size
        // MAX_MINIBLOCK_BYTES = 8186
        // For f32 (4 bytes): 8186 / 4 = 2046, so max chunk = 1024 (power of 2)
        // For f64 (8 bytes): 8186 / 8 = 1023, so max chunk = 512 (power of 2)
        match self.bits_per_value {
            32 => 1024,
            64 => 512,
            _ => unreachable!("ByteStreamSplit only supports 32 or 64 bit values"),
        }
    }
}

impl MiniBlockCompressor for ByteStreamSplitEncoder {
    fn compress(
        &self,
        _context: MiniBlockCompressionContext,
        page: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match page {
            DataBlock::FixedWidth(data) => {
                let num_values = data.num_values;
                let bytes_per_value = self.bytes_per_value();

                if num_values == 0 {
                    return Ok((
                        MiniBlockCompressed {
                            data: vec![],
                            chunks: vec![],
                            num_values: 0,
                        },
                        EncodingUtils::byte_stream_split(EncodingUtils::flat(
                            self.bits_per_value as u64,
                            None,
                        )),
                    ));
                }

                let num_values_usize = usize::try_from(num_values).map_err(|_| {
                    lance_core::Error::invalid_input_source(
                        format!("ByteStreamSplit value count {num_values} overflows usize").into(),
                    )
                })?;
                let total_size =
                    num_values_usize
                        .checked_mul(bytes_per_value)
                        .ok_or_else(|| {
                            lance_core::Error::invalid_input_source(
                                format!(
                                    "ByteStreamSplit byte length overflows usize for {num_values} values"
                                )
                                .into(),
                            )
                        })?;
                let data_slice = data.data.as_ref();
                if data_slice.len() != total_size {
                    return Err(lance_core::Error::invalid_input_source(
                        format!(
                            "Expected {} bytes for ByteStreamSplit compression, but got {}",
                            total_size,
                            data_slice.len()
                        )
                        .into(),
                    ));
                }

                let max_chunk_size = self.max_chunk_size();
                let mut global_buffer: Vec<u8> = Vec::with_capacity(total_size);
                let dst_base = global_buffer.as_mut_ptr();
                let mut chunks = Vec::with_capacity(num_values_usize.div_ceil(max_chunk_size));
                let mut processed_values = 0usize;

                while processed_values < num_values_usize {
                    let chunk_size = (num_values_usize - processed_values).min(max_chunk_size);
                    let chunk_offset = processed_values * bytes_per_value;
                    let chunk_bytes = chunk_size * bytes_per_value;

                    // SAFETY: `chunk_offset + chunk_bytes <= total_size == global_buffer.capacity()`
                    // and `data_slice` has length `total_size`. Each iteration writes every byte
                    // in `dst_base.add(chunk_offset)..dst_base.add(chunk_offset + chunk_bytes)`.
                    unsafe {
                        split_streams(
                            data_slice.as_ptr().add(chunk_offset),
                            dst_base.add(chunk_offset),
                            chunk_size,
                            bytes_per_value,
                        );
                    }

                    let log_num_values = if processed_values + chunk_size == num_values_usize {
                        0 // Last chunk
                    } else {
                        chunk_size.ilog2() as u8
                    };

                    debug_assert!(chunk_bytes > 0);
                    chunks.push(MiniBlockChunk {
                        buffer_sizes: vec![chunk_bytes as u32],
                        log_num_values,
                    });

                    processed_values += chunk_size;
                }

                // SAFETY: The loop above partitioned `0..num_values_usize` without gaps and
                // `split_streams` initialized all `chunk_size * bytes_per_value` bytes of each
                // chunk, covering `0..total_size`.
                unsafe {
                    global_buffer.set_len(total_size);
                }

                let data_buffers = vec![LanceBuffer::from(global_buffer)];

                // TODO: Should support underlying compression
                let encoding = EncodingUtils::byte_stream_split(EncodingUtils::flat(
                    self.bits_per_value as u64,
                    None,
                ));

                Ok((
                    MiniBlockCompressed {
                        data: data_buffers,
                        chunks,
                        num_values,
                    },
                    encoding,
                ))
            }
            _ => Err(lance_core::Error::invalid_input_source(
                "ByteStreamSplit encoding only supports FixedWidth data blocks".into(),
            )),
        }
    }
}

/// Byte Stream Split decompressor
#[derive(Debug)]
pub struct ByteStreamSplitDecompressor {
    bits_per_value: usize,
}

impl ByteStreamSplitDecompressor {
    pub fn new(bits_per_value: usize) -> Self {
        assert!(
            bits_per_value == 32 || bits_per_value == 64,
            "ByteStreamSplit only supports 32-bit (f32) or 64-bit (f64) values"
        );
        Self { bits_per_value }
    }

    fn bytes_per_value(&self) -> usize {
        self.bits_per_value / 8
    }
}

impl MiniBlockDecompressor for ByteStreamSplitDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: self.bits_per_value as u64,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        let bytes_per_value = self.bytes_per_value();
        let num_values_usize = usize::try_from(num_values).map_err(|_| {
            lance_core::Error::invalid_input_source(
                format!("ByteStreamSplit value count {num_values} overflows usize").into(),
            )
        })?;
        let total_bytes = num_values_usize
            .checked_mul(bytes_per_value)
            .ok_or_else(|| {
                lance_core::Error::invalid_input_source(
                    format!("ByteStreamSplit byte length overflows usize for {num_values} values")
                        .into(),
                )
            })?;

        if data.len() != 1 {
            return Err(lance_core::Error::invalid_input_source(
                format!(
                    "ByteStreamSplit decompression expects 1 buffer, but got {}",
                    data.len()
                )
                .into(),
            ));
        }

        let input_buffer = &data[0];

        if input_buffer.len() != total_bytes {
            return Err(lance_core::Error::invalid_input_source(
                format!(
                    "Expected {} bytes for decompression, but got {}",
                    total_bytes,
                    input_buffer.len()
                )
                .into(),
            ));
        }

        let mut output: Vec<u8> = Vec::with_capacity(total_bytes);
        // SAFETY: `input_buffer` has length `total_bytes == num_values_usize * bytes_per_value`
        // and `output` has capacity `total_bytes`. `unsplit_streams` writes every byte in
        // `0..total_bytes` before `set_len`.
        unsafe {
            unsplit_streams(
                input_buffer.as_ptr(),
                output.as_mut_ptr(),
                num_values_usize,
                bytes_per_value,
            );
            output.set_len(total_bytes);
        }

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(output),
            bits_per_value: self.bits_per_value as u64,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }

    fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
        num_values.checked_mul(self.bytes_per_value() as u64)
    }
}

/// Splits `num_values` elements of `bytes_per_value` (`4` or `8`) bytes from `src` into
/// `bytes_per_value` contiguous byte streams in `dst`.
///
/// # Safety
///
/// `src` must be valid for reads of `num_values * bytes_per_value` bytes, `dst` must be
/// valid for non-overlapping writes of `num_values * bytes_per_value` bytes, and
/// `bytes_per_value` must be `4` or `8`.
#[inline]
unsafe fn split_streams(src: *const u8, dst: *mut u8, num_values: usize, bytes_per_value: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if CpuFeatures::host().contains(CpuFeatures::AVX2) {
            match bytes_per_value {
                4 => return x86_64_avx2::split_w4(src, dst, num_values),
                8 => return x86_64_avx2::split_w8(src, dst, num_values),
                _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match bytes_per_value {
            4 => return aarch64_neon::split_w4(src, dst, num_values),
            8 => return aarch64_neon::split_w8(src, dst, num_values),
            _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
        }
    }
    match bytes_per_value {
        4 => split_scalar_range::<4>(src, dst, num_values, 0),
        8 => split_scalar_range::<8>(src, dst, num_values, 0),
        _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
    }
}

/// Reassembles `num_values` elements of `bytes_per_value` (`4` or `8`) bytes into `dst`
/// from `bytes_per_value` contiguous byte streams in `src`.
///
/// # Safety
///
/// `src` must be valid for reads of `num_values * bytes_per_value` bytes, `dst` must be
/// valid for non-overlapping writes of `num_values * bytes_per_value` bytes, and
/// `bytes_per_value` must be `4` or `8`.
#[inline]
unsafe fn unsplit_streams(src: *const u8, dst: *mut u8, num_values: usize, bytes_per_value: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if CpuFeatures::host().contains(CpuFeatures::AVX2) {
            match bytes_per_value {
                4 => return x86_64_avx2::unsplit_w4(src, dst, num_values),
                8 => return x86_64_avx2::unsplit_w8(src, dst, num_values),
                _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match bytes_per_value {
            4 => return aarch64_neon::unsplit_w4(src, dst, num_values),
            8 => return aarch64_neon::unsplit_w8(src, dst, num_values),
            _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
        }
    }
    match bytes_per_value {
        4 => unsplit_scalar_range::<4>(src, dst, num_values, 0),
        8 => unsplit_scalar_range::<8>(src, dst, num_values, 0),
        _ => unreachable!("ByteStreamSplit only supports 4 or 8 byte values"),
    }
}

#[inline]
unsafe fn split_scalar_range<const WIDTH: usize>(
    src: *const u8,
    dst: *mut u8,
    num_values: usize,
    start: usize,
) {
    for i in start..num_values {
        let src_val = src.add(i * WIDTH);
        for j in 0..WIDTH {
            *dst.add(j * num_values + i) = *src_val.add(j);
        }
    }
}

#[inline]
unsafe fn unsplit_scalar_range<const WIDTH: usize>(
    src: *const u8,
    dst: *mut u8,
    num_values: usize,
    start: usize,
) {
    for i in start..num_values {
        let dst_val = dst.add(i * WIDTH);
        for j in 0..WIDTH {
            *dst_val.add(j) = *src.add(j * num_values + i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64_avx2 {
    use std::arch::x86_64::*;

    use super::{split_scalar_range, unsplit_scalar_range};

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn split_w4(src: *const u8, dst: *mut u8, num_values: usize) {
        let shuf = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6,
            10, 14, 3, 7, 11, 15,
        );
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        let dst0 = dst;
        let dst1 = dst.add(num_values);
        let dst2 = dst.add(num_values * 2);
        let dst3 = dst.add(num_values * 3);

        let mut i = 0usize;
        while i + 32 <= num_values {
            let base = src.add(i * 4);
            let v0 = _mm256_loadu_si256(base.cast::<__m256i>());
            let v1 = _mm256_loadu_si256(base.add(32).cast::<__m256i>());
            let v2 = _mm256_loadu_si256(base.add(64).cast::<__m256i>());
            let v3 = _mm256_loadu_si256(base.add(96).cast::<__m256i>());

            // Group 4 bytes per stream inside each 128-bit lane, then pack low/high lanes
            // into four 64-bit chunks `[s0(8B), s1(8B), s2(8B), s3(8B)]`.
            let p0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v0, shuf), perm);
            let p1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v1, shuf), perm);
            let p2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v2, shuf), perm);
            let p3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v3, shuf), perm);

            let u01_lo = _mm256_unpacklo_epi64(p0, p1);
            let u01_hi = _mm256_unpackhi_epi64(p0, p1);
            let u23_lo = _mm256_unpacklo_epi64(p2, p3);
            let u23_hi = _mm256_unpackhi_epi64(p2, p3);

            let s0 = _mm256_permute2x128_si256(u01_lo, u23_lo, 0x20);
            let s1 = _mm256_permute2x128_si256(u01_hi, u23_hi, 0x20);
            let s2 = _mm256_permute2x128_si256(u01_lo, u23_lo, 0x31);
            let s3 = _mm256_permute2x128_si256(u01_hi, u23_hi, 0x31);

            _mm256_storeu_si256(dst0.add(i).cast::<__m256i>(), s0);
            _mm256_storeu_si256(dst1.add(i).cast::<__m256i>(), s1);
            _mm256_storeu_si256(dst2.add(i).cast::<__m256i>(), s2);
            _mm256_storeu_si256(dst3.add(i).cast::<__m256i>(), s3);

            i += 32;
        }

        split_scalar_range::<4>(src, dst, num_values, i);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn unsplit_w4(src: *const u8, dst: *mut u8, num_values: usize) {
        let src0 = src;
        let src1 = src.add(num_values);
        let src2 = src.add(num_values * 2);
        let src3 = src.add(num_values * 3);

        let mut i = 0usize;
        while i + 32 <= num_values {
            let s0 = _mm256_loadu_si256(src0.add(i).cast::<__m256i>());
            let s1 = _mm256_loadu_si256(src1.add(i).cast::<__m256i>());
            let s2 = _mm256_loadu_si256(src2.add(i).cast::<__m256i>());
            let s3 = _mm256_loadu_si256(src3.add(i).cast::<__m256i>());

            let lo_01 = _mm256_unpacklo_epi8(s0, s1);
            let hi_01 = _mm256_unpackhi_epi8(s0, s1);
            let lo_23 = _mm256_unpacklo_epi8(s2, s3);
            let hi_23 = _mm256_unpackhi_epi8(s2, s3);

            let v_0_16 = _mm256_unpacklo_epi16(lo_01, lo_23);
            let v_4_20 = _mm256_unpackhi_epi16(lo_01, lo_23);
            let v_8_24 = _mm256_unpacklo_epi16(hi_01, hi_23);
            let v_12_28 = _mm256_unpackhi_epi16(hi_01, hi_23);

            let out_base = dst.add(i * 4);
            _mm256_storeu_si256(
                out_base.cast::<__m256i>(),
                _mm256_permute2x128_si256(v_0_16, v_4_20, 0x20),
            );
            _mm256_storeu_si256(
                out_base.add(32).cast::<__m256i>(),
                _mm256_permute2x128_si256(v_8_24, v_12_28, 0x20),
            );
            _mm256_storeu_si256(
                out_base.add(64).cast::<__m256i>(),
                _mm256_permute2x128_si256(v_0_16, v_4_20, 0x31),
            );
            _mm256_storeu_si256(
                out_base.add(96).cast::<__m256i>(),
                _mm256_permute2x128_si256(v_8_24, v_12_28, 0x31),
            );

            i += 32;
        }

        unsplit_scalar_range::<4>(src, dst, num_values, i);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn split_w8(src: *const u8, dst: *mut u8, num_values: usize) {
        let shuf8 = _mm256_setr_epi8(
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12,
            5, 13, 6, 14, 7, 15,
        );
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        let mut i = 0usize;
        while i + 16 <= num_values {
            let base = src.add(i * 8);
            let v0 = _mm256_loadu_si256(base.cast::<__m256i>());
            let v1 = _mm256_loadu_si256(base.add(32).cast::<__m256i>());
            let v2 = _mm256_loadu_si256(base.add(64).cast::<__m256i>());
            let v3 = _mm256_loadu_si256(base.add(96).cast::<__m256i>());

            let t0 = _mm256_shuffle_epi8(v0, shuf8);
            let t1 = _mm256_shuffle_epi8(v1, shuf8);
            let t2 = _mm256_shuffle_epi8(v2, shuf8);
            let t3 = _mm256_shuffle_epi8(v3, shuf8);

            let a0 = _mm256_permute2x128_si256(t0, t1, 0x20);
            let a1 = _mm256_permute2x128_si256(t0, t1, 0x31);
            let b_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(a0, a1), perm);
            let b_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(a0, a1), perm);

            let c0 = _mm256_permute2x128_si256(t2, t3, 0x20);
            let c1 = _mm256_permute2x128_si256(t2, t3, 0x31);
            let d_lo = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi16(c0, c1), perm);
            let d_hi = _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi16(c0, c1), perm);

            let s02 = _mm256_unpacklo_epi64(b_lo, d_lo);
            let s13 = _mm256_unpackhi_epi64(b_lo, d_lo);
            let s46 = _mm256_unpacklo_epi64(b_hi, d_hi);
            let s57 = _mm256_unpackhi_epi64(b_hi, d_hi);

            _mm_storeu_si128(dst.add(i).cast::<__m128i>(), _mm256_castsi256_si128(s02));
            _mm_storeu_si128(
                dst.add(num_values + i).cast::<__m128i>(),
                _mm256_castsi256_si128(s13),
            );
            _mm_storeu_si128(
                dst.add(num_values * 2 + i).cast::<__m128i>(),
                _mm256_extracti128_si256::<1>(s02),
            );
            _mm_storeu_si128(
                dst.add(num_values * 3 + i).cast::<__m128i>(),
                _mm256_extracti128_si256::<1>(s13),
            );
            _mm_storeu_si128(
                dst.add(num_values * 4 + i).cast::<__m128i>(),
                _mm256_castsi256_si128(s46),
            );
            _mm_storeu_si128(
                dst.add(num_values * 5 + i).cast::<__m128i>(),
                _mm256_castsi256_si128(s57),
            );
            _mm_storeu_si128(
                dst.add(num_values * 6 + i).cast::<__m128i>(),
                _mm256_extracti128_si256::<1>(s46),
            );
            _mm_storeu_si128(
                dst.add(num_values * 7 + i).cast::<__m128i>(),
                _mm256_extracti128_si256::<1>(s57),
            );

            i += 16;
        }

        split_scalar_range::<8>(src, dst, num_values, i);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_interleaved_w8_group(
        a01: __m256i,
        a23: __m256i,
        a45: __m256i,
        a67: __m256i,
        out_base: *mut u8,
    ) {
        let q03_lo = _mm256_unpacklo_epi16(a01, a23);
        let q03_hi = _mm256_unpackhi_epi16(a01, a23);
        let q47_lo = _mm256_unpacklo_epi16(a45, a67);
        let q47_hi = _mm256_unpackhi_epi16(a45, a67);

        let r0 = _mm256_unpacklo_epi32(q03_lo, q47_lo);
        let r1 = _mm256_unpackhi_epi32(q03_lo, q47_lo);
        let r2 = _mm256_unpacklo_epi32(q03_hi, q47_hi);
        let r3 = _mm256_unpackhi_epi32(q03_hi, q47_hi);

        _mm256_storeu_si256(
            out_base.cast::<__m256i>(),
            _mm256_permute2x128_si256(r0, r1, 0x20),
        );
        _mm256_storeu_si256(
            out_base.add(32).cast::<__m256i>(),
            _mm256_permute2x128_si256(r2, r3, 0x20),
        );
        _mm256_storeu_si256(
            out_base.add(128).cast::<__m256i>(),
            _mm256_permute2x128_si256(r0, r1, 0x31),
        );
        _mm256_storeu_si256(
            out_base.add(160).cast::<__m256i>(),
            _mm256_permute2x128_si256(r2, r3, 0x31),
        );
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn unsplit_w8(src: *const u8, dst: *mut u8, num_values: usize) {
        let mut i = 0usize;
        while i + 32 <= num_values {
            let s0 = _mm256_loadu_si256(src.add(i).cast::<__m256i>());
            let s1 = _mm256_loadu_si256(src.add(num_values + i).cast::<__m256i>());
            let s2 = _mm256_loadu_si256(src.add(num_values * 2 + i).cast::<__m256i>());
            let s3 = _mm256_loadu_si256(src.add(num_values * 3 + i).cast::<__m256i>());
            let s4 = _mm256_loadu_si256(src.add(num_values * 4 + i).cast::<__m256i>());
            let s5 = _mm256_loadu_si256(src.add(num_values * 5 + i).cast::<__m256i>());
            let s6 = _mm256_loadu_si256(src.add(num_values * 6 + i).cast::<__m256i>());
            let s7 = _mm256_loadu_si256(src.add(num_values * 7 + i).cast::<__m256i>());

            let p01_lo = _mm256_unpacklo_epi8(s0, s1);
            let p01_hi = _mm256_unpackhi_epi8(s0, s1);
            let p23_lo = _mm256_unpacklo_epi8(s2, s3);
            let p23_hi = _mm256_unpackhi_epi8(s2, s3);
            let p45_lo = _mm256_unpacklo_epi8(s4, s5);
            let p45_hi = _mm256_unpackhi_epi8(s4, s5);
            let p67_lo = _mm256_unpacklo_epi8(s6, s7);
            let p67_hi = _mm256_unpackhi_epi8(s6, s7);

            let out_base = dst.add(i * 8);
            store_interleaved_w8_group(p01_lo, p23_lo, p45_lo, p67_lo, out_base);
            store_interleaved_w8_group(p01_hi, p23_hi, p45_hi, p67_hi, out_base.add(64));

            i += 32;
        }

        unsplit_scalar_range::<8>(src, dst, num_values, i);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
    use std::arch::aarch64::*;

    use super::{split_scalar_range, unsplit_scalar_range};

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn split_w4(src: *const u8, dst: *mut u8, num_values: usize) {
        let mut i = 0usize;
        while i + 16 <= num_values {
            let streams = vld4q_u8(src.add(i * 4));
            vst1q_u8(dst.add(i), streams.0);
            vst1q_u8(dst.add(num_values + i), streams.1);
            vst1q_u8(dst.add(num_values * 2 + i), streams.2);
            vst1q_u8(dst.add(num_values * 3 + i), streams.3);
            i += 16;
        }
        split_scalar_range::<4>(src, dst, num_values, i);
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn unsplit_w4(src: *const u8, dst: *mut u8, num_values: usize) {
        let mut i = 0usize;
        while i + 16 <= num_values {
            let streams = uint8x16x4_t(
                vld1q_u8(src.add(i)),
                vld1q_u8(src.add(num_values + i)),
                vld1q_u8(src.add(num_values * 2 + i)),
                vld1q_u8(src.add(num_values * 3 + i)),
            );
            vst4q_u8(dst.add(i * 4), streams);
            i += 16;
        }
        unsplit_scalar_range::<4>(src, dst, num_values, i);
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn split_w8(src: *const u8, dst: *mut u8, num_values: usize) {
        let mut i = 0usize;
        while i + 16 <= num_values {
            let lo = vld4q_u8(src.add(i * 8));
            let hi = vld4q_u8(src.add(i * 8 + 64));

            vst1q_u8(dst.add(i), vuzp1q_u8(lo.0, hi.0));
            vst1q_u8(dst.add(num_values + i), vuzp1q_u8(lo.1, hi.1));
            vst1q_u8(dst.add(num_values * 2 + i), vuzp1q_u8(lo.2, hi.2));
            vst1q_u8(dst.add(num_values * 3 + i), vuzp1q_u8(lo.3, hi.3));
            vst1q_u8(dst.add(num_values * 4 + i), vuzp2q_u8(lo.0, hi.0));
            vst1q_u8(dst.add(num_values * 5 + i), vuzp2q_u8(lo.1, hi.1));
            vst1q_u8(dst.add(num_values * 6 + i), vuzp2q_u8(lo.2, hi.2));
            vst1q_u8(dst.add(num_values * 7 + i), vuzp2q_u8(lo.3, hi.3));
            i += 16;
        }
        split_scalar_range::<8>(src, dst, num_values, i);
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn unsplit_w8(src: *const u8, dst: *mut u8, num_values: usize) {
        let mut i = 0usize;
        while i + 16 <= num_values {
            let s0 = vld1q_u8(src.add(i));
            let s1 = vld1q_u8(src.add(num_values + i));
            let s2 = vld1q_u8(src.add(num_values * 2 + i));
            let s3 = vld1q_u8(src.add(num_values * 3 + i));
            let s4 = vld1q_u8(src.add(num_values * 4 + i));
            let s5 = vld1q_u8(src.add(num_values * 5 + i));
            let s6 = vld1q_u8(src.add(num_values * 6 + i));
            let s7 = vld1q_u8(src.add(num_values * 7 + i));

            let lo = uint8x16x4_t(
                vzip1q_u8(s0, s4),
                vzip1q_u8(s1, s5),
                vzip1q_u8(s2, s6),
                vzip1q_u8(s3, s7),
            );
            let hi = uint8x16x4_t(
                vzip2q_u8(s0, s4),
                vzip2q_u8(s1, s5),
                vzip2q_u8(s2, s6),
                vzip2q_u8(s3, s7),
            );

            vst4q_u8(dst.add(i * 8), lo);
            vst4q_u8(dst.add(i * 8 + 64), hi);
            i += 16;
        }
        unsplit_scalar_range::<8>(src, dst, num_values, i);
    }
}

/// Determine if BSS should be used based on mode and data characteristics
pub fn should_use_bss(data: &FixedWidthDataBlock, mode: BssMode) -> bool {
    // Only support 32-bit and 64-bit values
    // BSS is most effective for these common types (floats, timestamps, etc.)
    // 16-bit values have limited benefit with only 2 streams
    if data.bits_per_value != 32 && data.bits_per_value != 64 {
        return false;
    }

    let sensitivity = mode.to_sensitivity();

    // Fast paths
    if sensitivity <= 0.0 {
        return false;
    }
    if sensitivity >= 1.0 {
        return true;
    }

    // Auto mode: check byte position entropy
    evaluate_entropy_for_bss(data, sensitivity)
}

/// Evaluate if BSS should be used based on byte position entropy
fn evaluate_entropy_for_bss(data: &FixedWidthDataBlock, sensitivity: f32) -> bool {
    // Get the precomputed entropy statistics
    let Some(entropy_stat) = data.get_stat(Stat::BytePositionEntropy) else {
        return false; // No entropy data available
    };

    let entropies = entropy_stat.as_primitive::<UInt64Type>();
    if entropies.is_empty() {
        return false;
    }

    // Calculate average entropy across all byte positions
    let sum: u64 = entropies.values().iter().sum();
    let avg_entropy = sum as f64 / entropies.len() as f64 / 1000.0; // Scale back from integer

    // Entropy threshold based on sensitivity
    // sensitivity = 0.5 (default auto) -> threshold = 4.0 bits
    // sensitivity = 0.0 (off) -> threshold = 0.0 (never use)
    // sensitivity = 1.0 (on) -> threshold = 8.0 (always use)
    let entropy_threshold = sensitivity as f64 * 8.0;

    // Use BSS if average entropy is below threshold
    // Lower entropy means more repetitive byte patterns
    avg_entropy < entropy_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn test_round_trip_f32() {
        let encoder = ByteStreamSplitEncoder::new(32);
        let decompressor = ByteStreamSplitDecompressor::new(32);

        let values: Vec<f32> = vec![
            1.0,
            2.5,
            -3.7,
            4.2,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        let decompressed = decompressor
            .decompress(compressed.data, values.len() as u64)
            .unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        let result_bytes = decompressed_fixed.data.as_ref();
        let result_values: Vec<f32> = result_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();

        assert_eq!(values, result_values);
    }

    #[test]
    fn test_round_trip_f64() {
        let encoder = ByteStreamSplitEncoder::new(64);
        let decompressor = ByteStreamSplitDecompressor::new(64);

        let values: Vec<f64> = vec![
            1.0,
            2.5,
            -3.7,
            4.2,
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        let decompressed = decompressor
            .decompress(compressed.data, values.len() as u64)
            .unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        let result_bytes = decompressed_fixed.data.as_ref();
        let result_values: Vec<f64> = result_bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk))
            .collect();

        assert_eq!(values, result_values);
    }

    #[test]
    fn test_empty_data() {
        let encoder = ByteStreamSplitEncoder::new(32);
        let decompressor = ByteStreamSplitDecompressor::new(32);

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::empty(),
            bits_per_value: 32,
            num_values: 0,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        let decompressed = decompressor.decompress(compressed.data, 0).unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        assert_eq!(decompressed_fixed.num_values, 0);
        assert!(decompressed_fixed.data.is_empty());
    }

    #[rstest]
    #[case::w4(4, 32)]
    #[case::w8(8, 64)]
    fn test_simd_matches_reference_across_boundary_lengths(
        #[case] width: usize,
        #[case] bits_per_value: usize,
    ) {
        let encoder = ByteStreamSplitEncoder::new(bits_per_value);
        let decompressor = ByteStreamSplitDecompressor::new(bits_per_value);

        for num_values in (0..=70).chain([511, 512, 513, 1023, 1024, 1025, 2117]) {
            let total_bytes = num_values * width;
            let input: Vec<u8> = (0..total_bytes)
                .map(|idx| ((idx.wrapping_mul(131).wrapping_add(17)) ^ (idx >> 3)) as u8)
                .collect();

            // Verify single-chunk split & unsplit byte-for-byte against scalar reference
            if num_values > 0 && num_values <= encoder.max_chunk_size() {
                let mut expected_split = vec![0u8; total_bytes];
                // SAFETY: `input` and `expected_split` both contain `num_values * width` bytes.
                // Each call passes the matching `num_values`, so `split_scalar_range` reads those
                // values and writes every byte of the corresponding `WIDTH` interleaved output in
                // bounds; the inactive width processes zero values.
                unsafe {
                    split_scalar_range::<4>(
                        input.as_ptr(),
                        expected_split.as_mut_ptr(),
                        if width == 4 { num_values } else { 0 },
                        0,
                    );
                    if width == 8 {
                        split_scalar_range::<8>(
                            input.as_ptr(),
                            expected_split.as_mut_ptr(),
                            num_values,
                            0,
                        );
                    }
                }

                let (compressed, _) = encoder
                    .compress(
                        MiniBlockCompressionContext::new(0, true, true),
                        DataBlock::FixedWidth(FixedWidthDataBlock {
                            data: LanceBuffer::from(input.clone()),
                            bits_per_value: bits_per_value as u64,
                            num_values: num_values as u64,
                            block_info: BlockInfo::new(),
                        }),
                    )
                    .unwrap();
                assert_eq!(compressed.data[0].as_ref(), expected_split.as_slice());

                let decoded = decompressor
                    .decompress(compressed.data, num_values as u64)
                    .unwrap();
                assert_eq!(
                    decoded.as_fixed_width_ref().unwrap().data.as_ref(),
                    input.as_slice()
                );
            } else {
                // Multi-chunk encoder round-trip across chunks + unaligned tail
                let (compressed, _) = encoder
                    .compress(
                        MiniBlockCompressionContext::new(0, true, true),
                        DataBlock::FixedWidth(FixedWidthDataBlock {
                            data: LanceBuffer::from(input.clone()),
                            bits_per_value: bits_per_value as u64,
                            num_values: num_values as u64,
                            block_info: BlockInfo::new(),
                        }),
                    )
                    .unwrap();

                let mut rebuilt = Vec::with_capacity(total_bytes);
                let mut offset = 0usize;
                let mut remaining = num_values;
                for chunk in &compressed.chunks {
                    let chunk_bytes = chunk.buffer_sizes[0] as usize;
                    let chunk_vals = if chunk.log_num_values == 0 {
                        remaining
                    } else {
                        1usize << chunk.log_num_values
                    };
                    let slice = compressed.data[0].slice_with_length(offset, chunk_bytes);
                    let decoded = decompressor
                        .decompress(vec![slice], chunk_vals as u64)
                        .unwrap();
                    rebuilt.extend_from_slice(decoded.as_fixed_width_ref().unwrap().data.as_ref());
                    offset += chunk_bytes;
                    remaining -= chunk_vals;
                }
                assert_eq!(rebuilt, input);
            }
        }
    }

    #[test]
    fn test_invalid_buffer_size_rejected() {
        let encoder = ByteStreamSplitEncoder::new(32);
        let decompressor = ByteStreamSplitDecompressor::new(32);

        let encode_err = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::from(vec![0u8; 10]),
                    bits_per_value: 32,
                    num_values: 4,
                    block_info: BlockInfo::new(),
                }),
            )
            .unwrap_err();
        assert!(matches!(encode_err, lance_core::Error::InvalidInput { .. }));
        assert!(encode_err.to_string().contains("Expected 16 bytes"));

        let decode_err = decompressor
            .decompress(vec![LanceBuffer::from(vec![0u8; 10])], 4)
            .unwrap_err();
        assert!(matches!(decode_err, lance_core::Error::InvalidInput { .. }));
        assert!(decode_err.to_string().contains("Expected 16 bytes"));
    }
}
