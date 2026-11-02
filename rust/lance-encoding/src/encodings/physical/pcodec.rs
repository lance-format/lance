// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pcodec (pco) mini-block compression for floating-point values and
//! fixed-size lists of them.
//!
//! A fixed-size list of floats is usually a small vector sampled over time
//! (a pose, a joint state, an IMU reading). Neighbouring rows are similar per
//! list position, not across positions, so each chunk is transposed to
//! item-major order before compression: all values of position 0, then all of
//! position 1, and so on. The transposed chunk is written as one pco
//! standalone file, which picks its own delta order and binning per chunk.
//!
//! ```text
//! rows (n values, d items each)      chunk payload (pco input, n * d items)
//! [a0 b0 c0]                         [a0 a1 .. an-1  b0 b1 .. bn-1  c0 ..]
//! [a1 b1 c1]          ──transpose──▶
//! ...
//! ```
//!
//! Every chunk holds a power-of-two number of rows and decodes on its own, so
//! a random take of one row decodes one chunk.

use std::fmt::Debug;

use arrow_buffer::ArrowNativeType;
use lance_core::{Error, Result};
use pco::ChunkConfig;
use pco::data_types::Number;
use pco::standalone::{simple_compress_into, simple_decompress_into};

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_VALUES, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
    MiniBlockCompressor,
};
use crate::encodings::physical::value::{ValueDecompressor, ValueEncoder};
use crate::format::EncodingUtils;
use crate::format::metadata::{self, CompressiveEncoding, compressive_encoding::Compression};

/// Upper bound on the uncompressed bytes of one chunk.
///
/// pco keeps per-chunk metadata (bins, delta order) for the whole chunk, so it
/// needs a few hundred values per list position to amortize it. On recorded
/// robot trajectories (f32 and f64 vectors of 3 to 64 items), 32 KiB chunks
/// compress within 1% of 256-row chunks per stream, while the usual 8 KiB
/// mini-block budget costs up to 7% more bytes.
const TARGET_CHUNK_BYTES: u64 = 32 * 1024;

/// The highest pco compression level; pco 1.0 does not export it.
const MAX_PCODEC_LEVEL: usize = 12;

/// Compresses float32/float64 mini-blocks with pco.
#[derive(Debug, Clone)]
pub struct PcodecMiniBlockEncoder {
    config: ChunkConfig,
}

impl PcodecMiniBlockEncoder {
    /// `level` is the pco compression level (0 to 12).
    pub fn try_new(level: Option<i32>) -> Result<Self> {
        let level = match level {
            None => pco::DEFAULT_COMPRESSION_LEVEL,
            Some(level) => usize::try_from(level)
                .ok()
                .filter(|level| *level <= MAX_PCODEC_LEVEL)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "pcodec compression level must be between 0 and {}, got {level}",
                        MAX_PCODEC_LEVEL
                    ))
                })?,
        };
        Ok(Self {
            config: ChunkConfig::default().with_compression_level(level),
        })
    }
}

impl MiniBlockCompressor for PcodecMiniBlockEncoder {
    fn compress(
        &self,
        _context: MiniBlockCompressionContext,
        page: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        let (values, data) = match page {
            DataBlock::FixedWidth(data) => (EncodingUtils::flat(data.bits_per_value, None), data),
            DataBlock::FixedSizeList(fsl) => {
                let values = ValueEncoder::fsl_to_encoding(&fsl);
                let data = fsl.try_into_flat().ok_or_else(|| {
                    Error::invalid_input("pcodec does not support fixed-size lists with null items")
                })?;
                (values, data)
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "pcodec cannot compress a {} data block",
                    other.name()
                )));
            }
        };
        let shape = ValueShape::try_from_values(&values)?;
        let compressed = match shape.bits_per_item {
            32 => compress_page::<f32>(&data, shape.dimension, &self.config)?,
            64 => compress_page::<f64>(&data, shape.dimension, &self.config)?,
            _ => unreachable!("ValueShape only accepts 32 and 64-bit items"),
        };
        Ok((compressed, EncodingUtils::pcodec(values)))
    }
}

/// Number of rows per chunk: the largest power of two whose uncompressed size
/// fits [`TARGET_CHUNK_BYTES`], at least two rows and at most
/// [`MAX_MINIBLOCK_VALUES`]. A non-final chunk cannot hold one row because
/// `log_num_values == 0` marks the final chunk.
fn rows_per_chunk(bytes_per_row: u64) -> u64 {
    let mut rows = 2;
    while rows * 2 * bytes_per_row <= TARGET_CHUNK_BYTES && rows * 2 <= *MAX_MINIBLOCK_VALUES {
        rows *= 2;
    }
    rows
}

fn compress_page<T: Number + ArrowNativeType>(
    data: &FixedWidthDataBlock,
    dimension: usize,
    config: &ChunkConfig,
) -> Result<MiniBlockCompressed> {
    let items = data.data.borrow_to_typed_slice::<T>();
    let items = items.as_ref();
    let num_rows = data.num_values as usize;
    debug_assert_eq!(items.len(), num_rows * dimension);

    let chunk_rows = rows_per_chunk((dimension * size_of::<T>()) as u64) as usize;
    let mut output = Vec::with_capacity(size_of_val(items) / 2);
    let mut chunks = Vec::with_capacity(num_rows.div_ceil(chunk_rows));
    let mut transposed = Vec::with_capacity(chunk_rows * dimension);
    for (chunk_idx, row_items) in items.chunks(chunk_rows * dimension).enumerate() {
        let rows = row_items.len() / dimension;
        let chunk_items = if dimension == 1 {
            row_items
        } else {
            transposed.clear();
            for item in 0..dimension {
                transposed.extend(row_items.iter().skip(item).step_by(dimension));
            }
            transposed.as_slice()
        };
        let start = output.len();
        output = simple_compress_into(chunk_items, config, output)
            .map_err(|err| Error::invalid_input(format!("pcodec compression failed: {err}")))?;
        let size = u32::try_from(output.len() - start).map_err(|_| {
            Error::invalid_input(format!(
                "pcodec chunk {chunk_idx} is {} bytes, which exceeds the mini-block chunk limit",
                output.len() - start
            ))
        })?;
        let is_last = (chunk_idx + 1) * chunk_rows >= num_rows;
        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![size],
            // The last chunk's row count is implied by the page row count.
            log_num_values: if is_last { 0 } else { rows.ilog2() as u8 },
        });
    }
    Ok(MiniBlockCompressed {
        data: vec![LanceBuffer::from(output)],
        chunks,
        num_values: data.num_values,
    })
}

/// The item width and list dimension described by a pcodec `values` encoding.
#[derive(Debug, Clone, Copy)]
struct ValueShape {
    bits_per_item: u64,
    /// Items per value; the product of all fixed-size-list dimensions.
    dimension: usize,
}

impl ValueShape {
    fn try_from_values(values: &CompressiveEncoding) -> Result<Self> {
        let mut dimension = 1_usize;
        let mut encoding = values;
        loop {
            match encoding.compression.as_ref() {
                Some(Compression::Flat(flat)) => {
                    if !matches!(flat.bits_per_value, 32 | 64) || flat.data.is_some() {
                        return Err(Error::invalid_input(format!(
                            "pcodec requires plain 32 or 64-bit Flat items, got {} bits{}",
                            flat.bits_per_value,
                            if flat.data.is_some() {
                                " with buffer compression"
                            } else {
                                ""
                            }
                        )));
                    }
                    return Ok(Self {
                        bits_per_item: flat.bits_per_value,
                        dimension,
                    });
                }
                Some(Compression::FixedSizeList(fsl)) => {
                    if fsl.has_validity {
                        return Err(Error::invalid_input(
                            "pcodec does not support fixed-size lists with item validity",
                        ));
                    }
                    dimension = usize::try_from(fsl.items_per_value)
                        .ok()
                        .filter(|dim| *dim > 0)
                        .and_then(|dim| dimension.checked_mul(dim))
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "pcodec fixed-size list has invalid dimension {}",
                                fsl.items_per_value
                            ))
                        })?;
                    encoding = fsl.values.as_deref().ok_or_else(|| {
                        Error::invalid_input("pcodec fixed-size list is missing its values")
                    })?;
                }
                other => {
                    return Err(Error::invalid_input(format!(
                        "pcodec values must be Flat or FixedSizeList, got {other:?}"
                    )));
                }
            }
        }
    }
}

/// Decompresses pcodec mini-block chunks back into their Flat or FixedSizeList layout.
#[derive(Debug)]
pub struct PcodecMiniBlockDecompressor {
    shape: ValueShape,
    /// Rebuilds the fixed-size-list layers from the row-major items.
    values: ValueDecompressor,
}

impl PcodecMiniBlockDecompressor {
    pub fn from_description(description: &metadata::Pcodec) -> Result<Self> {
        let values = description
            .values
            .as_deref()
            .ok_or_else(|| Error::invalid_input("pcodec encoding is missing its values"))?;
        let shape = ValueShape::try_from_values(values)?;
        let values = match values.compression.as_ref() {
            Some(Compression::Flat(flat)) => ValueDecompressor::from_flat(flat),
            Some(Compression::FixedSizeList(fsl)) => ValueDecompressor::from_fsl(fsl)?,
            _ => unreachable!("ValueShape only accepts Flat and FixedSizeList values"),
        };
        Ok(Self { shape, values })
    }

    fn decompress_items<T: Number + ArrowNativeType>(
        &self,
        compressed: &[u8],
        num_values: u64,
    ) -> Result<LanceBuffer> {
        let dimension = self.shape.dimension;
        let num_items = usize::try_from(num_values)
            .ok()
            .and_then(|rows| rows.checked_mul(dimension))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "pcodec chunk of {num_values} values with dimension {dimension} overflows"
                ))
            })?;
        let mut items = vec![T::default(); num_items];
        let progress = simple_decompress_into(compressed, &mut items).map_err(|err| {
            Error::corrupt_file_named(
                "pcodec chunk",
                format!("pcodec decompression failed: {err}"),
            )
        })?;
        if progress.n_processed != num_items || !progress.finished {
            return Err(Error::corrupt_file_named(
                "pcodec chunk",
                format!(
                    "pcodec chunk decoded {} items{}, expected {num_items}",
                    progress.n_processed,
                    if progress.finished {
                        ""
                    } else {
                        " with data left over"
                    }
                ),
            ));
        }
        if dimension == 1 {
            return Ok(LanceBuffer::reinterpret_vec(items));
        }
        let rows = num_items / dimension;
        let mut row_major = vec![T::default(); num_items];
        for (item, column) in items.chunks_exact(rows).enumerate() {
            for (row, value) in column.iter().enumerate() {
                row_major[row * dimension + item] = *value;
            }
        }
        Ok(LanceBuffer::reinterpret_vec(row_major))
    }
}

impl MiniBlockDecompressor for PcodecMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let [compressed] = data.as_slice() else {
            return Err(Error::invalid_input(format!(
                "pcodec mini-block has {} buffers, expected 1",
                data.len()
            )));
        };
        let items = match self.shape.bits_per_item {
            32 => self.decompress_items::<f32>(compressed, num_values)?,
            64 => self.decompress_items::<f64>(compressed, num_values)?,
            _ => unreachable!("ValueShape only accepts 32 and 64-bit items"),
        };
        MiniBlockDecompressor::decompress(&self.values, vec![items], num_values)
    }

    fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
        num_values
            .checked_mul(self.shape.dimension as u64)?
            .checked_mul(self.shape.bits_per_item / 8)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::Field;
    use rstest::rstest;

    use super::*;
    use crate::compression::try_pcodec_miniblock;
    use crate::compression_config::CompressionFieldParams;
    use crate::data::{BlockInfo, FixedSizeListBlock};

    /// Smooth values per list position plus NaN payloads, signed zeros,
    /// infinities and subnormals, as little-endian bytes.
    fn sample_bytes(bits_per_item: u64, num_items: usize) -> Vec<u8> {
        let special = [
            f64::from_bits(0x7ff8_dead_beef_0001),
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 4.0,
        ];
        let values = (0..num_items).map(|i| match i % 97 {
            j if j < special.len() => special[j],
            _ => (i as f64 * 0.37).sin() * 1e3,
        });
        match bits_per_item {
            32 => values
                .flat_map(|v| match v.to_bits() {
                    // f64 -> f32 drops the payload, so write an f32 NaN payload directly.
                    0x7ff8_dead_beef_0001 => 0x7fc0_beef_u32.to_le_bytes(),
                    _ => (v as f32).to_bits().to_le_bytes(),
                })
                .collect(),
            _ => values.flat_map(|v| v.to_bits().to_le_bytes()).collect(),
        }
    }

    fn page(bits_per_item: u64, dims: &[i32], num_rows: usize) -> DataBlock {
        let dimension = dims.iter().product::<i32>() as u64;
        let mut data_type = match bits_per_item {
            32 => DataType::Float32,
            _ => DataType::Float64,
        };
        for dim in dims.iter().rev() {
            data_type =
                DataType::FixedSizeList(Arc::new(ArrowField::new("item", data_type, true)), *dim);
        }
        let flat = FixedWidthDataBlock {
            data: LanceBuffer::from(sample_bytes(bits_per_item, num_rows * dimension as usize)),
            bits_per_value: bits_per_item * dimension,
            num_values: num_rows as u64,
            block_info: BlockInfo::new(),
        };
        FixedSizeListBlock::from_flat(flat, &data_type)
    }

    fn into_bytes(block: DataBlock) -> Vec<u8> {
        match block {
            DataBlock::FixedWidth(data) => data.data.to_vec(),
            DataBlock::FixedSizeList(fsl) => fsl.try_into_flat().unwrap().data.to_vec(),
            other => panic!("unexpected block {}", other.name()),
        }
    }

    fn compress(page: DataBlock) -> (MiniBlockCompressed, CompressiveEncoding) {
        PcodecMiniBlockEncoder::try_new(None)
            .unwrap()
            .compress(MiniBlockCompressionContext::new(0, true, true), page)
            .unwrap()
    }

    fn decompress_chunks(
        compressed: &MiniBlockCompressed,
        encoding: &CompressiveEncoding,
    ) -> Vec<u8> {
        let Some(Compression::Pcodec(description)) = encoding.compression.as_ref() else {
            panic!("expected a pcodec encoding, got {encoding:?}");
        };
        let decompressor = PcodecMiniBlockDecompressor::from_description(description).unwrap();
        let mut decoded = Vec::new();
        let (mut rows_before, mut offset) = (0, 0);
        for chunk in &compressed.chunks {
            let rows = chunk.num_values(rows_before, compressed.num_values);
            let size = chunk.buffer_sizes[0] as usize;
            let buffer = compressed.data[0].slice_with_length(offset, size);
            decoded.extend(into_bytes(
                decompressor.decompress(vec![buffer], rows).unwrap(),
            ));
            rows_before += rows;
            offset += size;
        }
        decoded
    }

    #[rstest]
    #[case::scalar_f32(32, vec![], 5000)]
    #[case::scalar_f64(64, vec![], 5000)]
    #[case::vector_f32(32, vec![3], 2049)]
    #[case::nested_f64(64, vec![4, 4], 1000)]
    #[case::single_row(64, vec![7], 1)]
    fn test_round_trip_is_bit_exact(
        #[case] bits_per_item: u64,
        #[case] dims: Vec<i32>,
        #[case] num_rows: usize,
    ) {
        let page = page(bits_per_item, &dims, num_rows);
        let expected = into_bytes(page.clone());
        let (compressed, encoding) = compress(page);
        assert_eq!(compressed.num_values, num_rows as u64);
        assert_eq!(decompress_chunks(&compressed, &encoding), expected);
    }

    #[test]
    fn test_chunks_hold_a_power_of_two_rows_within_budget() {
        assert_eq!(rows_per_chunk(4), *MAX_MINIBLOCK_VALUES);
        assert_eq!(rows_per_chunk(3 * 4), 2048);
        assert_eq!(rows_per_chunk(64 * 8), 64);
        assert_eq!(rows_per_chunk(TARGET_CHUNK_BYTES + 1), 2);

        let (compressed, _) = compress(page(64, &[64], 130));
        let log_rows: Vec<u8> = compressed.chunks.iter().map(|c| c.log_num_values).collect();
        // Two full 64-row chunks, then the remainder, whose size is implied.
        assert_eq!(log_rows, vec![6, 6, 0]);
    }

    #[test]
    fn test_rejects_invalid_level() {
        for level in [-1, 13] {
            let err = PcodecMiniBlockEncoder::try_new(Some(level)).unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(err.to_string().contains("between 0 and 12"), "{err}");
        }
        assert!(PcodecMiniBlockEncoder::try_new(Some(0)).is_ok());
    }

    #[test]
    fn test_rejects_corrupt_chunks() {
        let (compressed, encoding) = compress(page(64, &[3], 100));
        let Some(Compression::Pcodec(description)) = encoding.compression.as_ref() else {
            panic!("expected a pcodec encoding");
        };
        let decompressor = PcodecMiniBlockDecompressor::from_description(description).unwrap();
        let chunk = compressed.data[0].clone();

        let truncated = chunk.slice_with_length(0, chunk.len() / 2);
        let err = decompressor.decompress(vec![truncated], 100).unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err}");

        let err = decompressor.decompress(vec![chunk], 99).unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err}");
        assert!(err.to_string().contains("expected 297"), "{err}");
    }

    #[test]
    fn test_rejects_unsupported_descriptors() {
        let nullable_items =
            EncodingUtils::pcodec(EncodingUtils::fsl(3, true, EncodingUtils::flat(32, None)));
        let half_floats = EncodingUtils::pcodec(EncodingUtils::flat(16, None));
        for encoding in [nullable_items, half_floats] {
            let Some(Compression::Pcodec(description)) = encoding.compression.as_ref() else {
                unreachable!()
            };
            let err = PcodecMiniBlockDecompressor::from_description(description).unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        }
    }

    #[test]
    fn test_selector_rejects_non_float_fields() {
        let params = CompressionFieldParams {
            compression: Some("pcodec".to_string()),
            ..Default::default()
        };
        let field = Field::try_from(&ArrowField::new("ids", DataType::Int32, false)).unwrap();
        let err = try_pcodec_miniblock(&field, &page(32, &[], 4), &params).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(
            err.to_string().contains("field 'ids' has type Int32"),
            "{err}"
        );

        let field = Field::try_from(&ArrowField::new("x", DataType::Float32, false)).unwrap();
        assert!(
            try_pcodec_miniblock(&field, &page(32, &[], 4), &params)
                .unwrap()
                .is_some()
        );
    }
}
