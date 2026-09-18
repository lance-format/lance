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
//! - Maximum chunk size is the smaller of the byte budget and
//!   `LANCE_MINIBLOCK_MAX_VALUES` rounded down to a power of two:
//!   - f32: 1024 values (4KB per chunk) at the default configuration
//!   - f64: 512 values (4KB per chunk) at the default configuration
//! - All chunks share a single global buffer
//! - Non-last chunks always contain power-of-2 values

use std::fmt::Debug;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::compression_config::BssMode;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_VALUES, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
    MiniBlockCompressor,
};
use crate::format::ProtobufUtils21;
use crate::format::pb21::CompressiveEncoding;
use crate::statistics::{GetStat, Stat};
use arrow_array::{cast::AsArray, types::UInt64Type};
use lance_core::Result;

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
        Self::chunk_size_for(self.bits_per_value, *MAX_MINIBLOCK_VALUES)
    }

    /// Largest chunk this encoder may emit, given the configured value cap.
    ///
    /// Split out from [`Self::max_chunk_size`] so the rounding is testable without
    /// touching the process-wide `LANCE_MINIBLOCK_MAX_VALUES` lazy static.
    fn chunk_size_for(bits_per_value: usize, configured_max_values: u64) -> usize {
        // For ByteStreamSplit, total bytes = bytes_per_value * chunk_size
        // MAX_MINIBLOCK_BYTES = 8186
        // For f32 (4 bytes): 8186 / 4 = 2046, so max chunk = 1024 (power of 2)
        // For f64 (8 bytes): 8186 / 8 = 1023, so max chunk = 512 (power of 2)
        let by_bytes = match bits_per_value {
            32 => 1024,
            64 => 512,
            _ => unreachable!("ByteStreamSplit only supports 32 or 64 bit values"),
        };
        // `LANCE_MINIBLOCK_MAX_VALUES` is documented as an upper bound on the values in
        // any mini-block chunk, so honor it here too. It has to be rounded down to a
        // power of two: `compress` records a non-final chunk's length as
        // `chunk_size.ilog2()`, and a chunk size that is not a power of two would
        // declare fewer values than were written. Two is the floor, because a non-final
        // chunk carrying `log_num_values == 0` is rejected on read.
        let configured = usize::try_from(configured_max_values)
            .unwrap_or(usize::MAX)
            .max(2);
        by_bytes.min(1usize << configured.ilog2())
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
                        ProtobufUtils21::byte_stream_split(ProtobufUtils21::flat(
                            self.bits_per_value as u64,
                            None,
                        )),
                    ));
                }

                let total_size = num_values as usize * bytes_per_value;
                let mut global_buffer = vec![0u8; total_size];

                let mut chunks = Vec::new();
                let data_slice = data.data.as_ref();
                let mut processed_values = 0usize;
                let max_chunk_size = self.max_chunk_size();

                while processed_values < num_values as usize {
                    let chunk_size = (num_values as usize - processed_values).min(max_chunk_size);
                    let chunk_offset = processed_values * bytes_per_value;

                    // Create chunk-local byte streams
                    for i in 0..chunk_size {
                        let src_offset = (processed_values + i) * bytes_per_value;
                        for j in 0..bytes_per_value {
                            // Store in chunk-local byte stream format
                            let dst_offset = chunk_offset + j * chunk_size + i;
                            global_buffer[dst_offset] = data_slice[src_offset + j];
                        }
                    }

                    let chunk_bytes = chunk_size * bytes_per_value;
                    let log_num_values = if processed_values + chunk_size == num_values as usize {
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

                let data_buffers = vec![LanceBuffer::from(global_buffer)];

                // TODO: Should support underlying compression
                let encoding = ProtobufUtils21::byte_stream_split(ProtobufUtils21::flat(
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
        let total_bytes = num_values as usize * bytes_per_value;

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

        let mut output = vec![0u8; total_bytes];

        // Input buffer contains chunk-local byte streams
        for i in 0..num_values as usize {
            for j in 0..bytes_per_value {
                let src_offset = j * num_values as usize + i;
                output[i * bytes_per_value + j] = input_buffer[src_offset];
            }
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

    #[test]
    fn test_round_trip_f32() {
        let encoder = ByteStreamSplitEncoder::new(32);
        let decompressor = ByteStreamSplitDecompressor::new(32);

        // Test data
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

        // Compress
        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        // Decompress
        let decompressed = decompressor
            .decompress(compressed.data, values.len() as u64)
            .unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        // Verify
        let result_bytes = decompressed_fixed.data.as_ref();
        let result_values: Vec<f32> = result_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        assert_eq!(values, result_values);
    }

    #[test]
    fn test_round_trip_f64() {
        let encoder = ByteStreamSplitEncoder::new(64);
        let decompressor = ByteStreamSplitDecompressor::new(64);

        // Test data
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

        // Compress
        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        // Decompress
        let decompressed = decompressor
            .decompress(compressed.data, values.len() as u64)
            .unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        // Verify
        let result_bytes = decompressed_fixed.data.as_ref();
        let result_values: Vec<f64> = result_bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
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

        // Compress empty data
        let (compressed, _encoding) = encoder
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        // Decompress empty data
        let decompressed = decompressor.decompress(compressed.data, 0).unwrap();
        let DataBlock::FixedWidth(decompressed_fixed) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        assert_eq!(decompressed_fixed.num_values, 0);
        assert_eq!(decompressed_fixed.data.len(), 0);
    }

    /// The round trips above fit in one chunk, so nothing checked a non-final chunk's
    /// `log_num_values` against the length actually written. That is the invariant
    /// `chunk_size_for` rounds to a power of two to preserve.
    #[rstest::rstest]
    #[case::f32(32)]
    #[case::f64(64)]
    fn test_non_final_chunks_declare_their_length(#[case] bits_per_value: usize) {
        let bytes_per_value = bits_per_value / 8;
        let max_chunk =
            ByteStreamSplitEncoder::chunk_size_for(bits_per_value, *MAX_MINIBLOCK_VALUES);
        let num_values = max_chunk * 2 + 5;

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(vec![7u8; num_values * bytes_per_value]),
            bits_per_value: bits_per_value as u64,
            num_values: num_values as u64,
            block_info: BlockInfo::new(),
        });
        let (compressed, _encoding) = ByteStreamSplitEncoder::new(bits_per_value)
            .compress(MiniBlockCompressionContext::new(0, true, true), data_block)
            .unwrap();

        assert_eq!(compressed.num_values, num_values as u64);
        assert_eq!(compressed.chunks.len(), 3);
        let (last, non_final) = compressed.chunks.split_last().unwrap();
        for chunk in non_final {
            assert_eq!(
                1usize << chunk.log_num_values,
                max_chunk,
                "a non-final chunk must declare the {max_chunk} values it holds"
            );
            assert_eq!(chunk.buffer_sizes[0] as usize, max_chunk * bytes_per_value);
        }
        // Zero is the sentinel for the trailing chunk, whose length the reader derives.
        assert_eq!(last.log_num_values, 0);
        assert_eq!(last.buffer_sizes[0] as usize, 5 * bytes_per_value);
    }

    /// `compress` stores a non-final chunk's length as `chunk_size.ilog2()`, so the
    /// configured cap has to reach this encoder as a power of two no smaller than two.
    #[rstest::rstest]
    // The default cap is above the byte budget, so the byte budget still decides.
    #[case::default_cap(4096, 1024, 512)]
    // The maximum cap likewise leaves the byte budget binding.
    #[case::max_cap(32768, 1024, 512)]
    // A cap below the byte budget takes over.
    #[case::below_byte_budget(256, 256, 256)]
    // A cap that is not a power of two rounds down rather than truncating ilog2.
    #[case::rounds_down(300, 256, 256)]
    // The env parser clamps to a minimum of one; two is the smallest usable chunk.
    #[case::floor_of_two(1, 2, 2)]
    fn test_chunk_size_honors_configured_value_cap(
        #[case] configured: u64,
        #[case] expected_f32: usize,
        #[case] expected_f64: usize,
    ) {
        assert_eq!(
            ByteStreamSplitEncoder::chunk_size_for(32, configured),
            expected_f32
        );
        assert_eq!(
            ByteStreamSplitEncoder::chunk_size_for(64, configured),
            expected_f64
        );
    }
}
