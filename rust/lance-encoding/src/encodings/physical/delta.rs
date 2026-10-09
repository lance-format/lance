// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Delta encoding for non-decreasing unsigned block sequences.

use super::{checked_fixed_values, try_vec_with_capacity};
use crate::{
    buffer::LanceBuffer,
    compression::{BlockCompressor, BlockDecompressor, validate_delta_child_encoding},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};
use lance_core::{Error, Result};

#[derive(Debug)]
/// Encodes adjacent differences with a concrete child block codec.
pub struct DeltaEncoder {
    bits_per_value: u64,
    base: u64,
    child: Box<dyn BlockCompressor>,
}

impl DeltaEncoder {
    /// Creates a delta codec with the first value and child difference codec.
    pub fn new(bits_per_value: u64, base: u64, child: Box<dyn BlockCompressor>) -> Self {
        Self {
            bits_per_value,
            base,
            child,
        }
    }
}

impl BlockCompressor for DeltaEncoder {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Delta encoding requires a fixed-width data block",
            ));
        };
        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(format!(
                "Delta codec expects {}-bit values, got {}",
                self.bits_per_value, data.bits_per_value
            )));
        }
        let deltas = encode_deltas(data, self.base)?;
        let (payload, child_encoding) = self.child.compress(DataBlock::FixedWidth(deltas))?;
        validate_delta_child_encoding(&child_encoding, self.bits_per_value)?;
        Ok((
            payload,
            ProtobufUtils21::delta(self.bits_per_value, self.base, child_encoding),
        ))
    }
}

fn encode_deltas(data: FixedWidthDataBlock, expected_base: u64) -> Result<FixedWidthDataBlock> {
    if !matches!(data.bits_per_value, 32 | 64) {
        return Err(Error::invalid_input(format!(
            "Delta only supports 32 or 64-bit values, got {}",
            data.bits_per_value
        )));
    }
    if data.num_values < 2 {
        return Err(Error::invalid_input(format!(
            "Delta requires at least 2 values, got {}",
            data.num_values
        )));
    }

    let deltas = match data.bits_per_value {
        32 => {
            let values = checked_fixed_values::<u32>(&data, "Delta input")?;
            if u64::from(values[0]) != expected_base {
                return Err(Error::invalid_input(format!(
                    "Delta base mismatch: codec expects {expected_base}, input starts with {}",
                    values[0]
                )));
            }
            let mut deltas = try_vec_with_capacity::<u32>(data.num_values - 1, "Delta output")?;
            for (index, pair) in values.windows(2).enumerate() {
                deltas.push(pair[1].checked_sub(pair[0]).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta input decreases at index {}: {} -> {}",
                        index + 1,
                        pair[0],
                        pair[1]
                    ))
                })?);
            }
            LanceBuffer::reinterpret_vec(deltas)
        }
        64 => {
            let values = checked_fixed_values::<u64>(&data, "Delta input")?;
            if values[0] != expected_base {
                return Err(Error::invalid_input(format!(
                    "Delta base mismatch: codec expects {expected_base}, input starts with {}",
                    values[0]
                )));
            }
            let mut deltas = try_vec_with_capacity::<u64>(data.num_values - 1, "Delta output")?;
            for (index, pair) in values.windows(2).enumerate() {
                deltas.push(pair[1].checked_sub(pair[0]).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta input decreases at index {}: {} -> {}",
                        index + 1,
                        pair[0],
                        pair[1]
                    ))
                })?);
            }
            LanceBuffer::reinterpret_vec(deltas)
        }
        _ => unreachable!("delta width was validated above"),
    };
    Ok(FixedWidthDataBlock {
        bits_per_value: data.bits_per_value,
        data: deltas,
        num_values: data.num_values - 1,
        block_info: BlockInfo::default(),
    })
}

#[derive(Debug)]
pub(crate) struct DeltaDecompressor {
    bits_per_value: u64,
    base: u64,
    child: Box<dyn BlockDecompressor>,
}

impl DeltaDecompressor {
    pub(crate) fn new(bits_per_value: u64, base: u64, child: Box<dyn BlockDecompressor>) -> Self {
        Self {
            bits_per_value,
            base,
            child,
        }
    }
}

impl BlockDecompressor for DeltaDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        validate_delta_header(self.bits_per_value, self.base, num_values)?;
        let child = self.child.decompress(data, num_values - 1)?;
        reconstruct_deltas(child, self.bits_per_value, self.base, num_values)
    }

    fn requires_payload(&self) -> bool {
        self.child.requires_payload()
    }
}

pub(crate) fn validate_delta_header(bits_per_value: u64, base: u64, num_values: u64) -> Result<()> {
    if !matches!(bits_per_value, 32 | 64) {
        return Err(Error::invalid_input(format!(
            "Delta only supports 32 or 64-bit values, got {bits_per_value}"
        )));
    }
    if bits_per_value == 32 && base > u32::MAX as u64 {
        return Err(Error::invalid_input(format!(
            "Delta base {base} exceeds u32::MAX"
        )));
    }
    if num_values < 2 {
        return Err(Error::invalid_input(format!(
            "Delta requires at least 2 values, got {num_values}"
        )));
    }
    Ok(())
}

fn reconstruct_deltas(
    child: DataBlock,
    bits_per_value: u64,
    base: u64,
    num_values: u64,
) -> Result<DataBlock> {
    let DataBlock::FixedWidth(child) = child else {
        return Err(Error::invalid_input(
            "Delta child decoded to a non fixed-width block",
        ));
    };
    if child.bits_per_value != bits_per_value || child.num_values != num_values - 1 {
        return Err(Error::invalid_input(format!(
            "Delta child decoded {} {}-bit values, expected {} {}-bit values",
            child.num_values,
            child.bits_per_value,
            num_values - 1,
            bits_per_value
        )));
    }

    let data = match bits_per_value {
        32 => {
            let deltas = checked_fixed_values::<u32>(&child, "Delta child")?;
            let mut values = try_vec_with_capacity::<u32>(num_values, "Delta output")?;
            let mut current = u32::try_from(base)
                .map_err(|_| Error::invalid_input(format!("Delta base {base} exceeds u32::MAX")))?;
            values.push(current);
            for (index, delta) in deltas.iter().enumerate() {
                current = current.checked_add(*delta).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta prefix sum overflows u32 at index {}",
                        index + 1
                    ))
                })?;
                values.push(current);
            }
            LanceBuffer::reinterpret_vec(values)
        }
        64 => {
            let deltas = checked_fixed_values::<u64>(&child, "Delta child")?;
            let mut values = try_vec_with_capacity::<u64>(num_values, "Delta output")?;
            let mut current = base;
            values.push(current);
            for (index, delta) in deltas.iter().enumerate() {
                current = current.checked_add(*delta).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Delta prefix sum overflows u64 at index {}",
                        index + 1
                    ))
                })?;
                values.push(current);
            }
            LanceBuffer::reinterpret_vec(values)
        }
        _ => unreachable!("delta header was validated before reconstruction"),
    };

    Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value,
        data,
        num_values,
        block_info: BlockInfo::default(),
    }))
}

#[cfg(test)]
mod tests {
    use crate::compression::{DecompressionStrategy, DefaultDecompressionStrategy};
    use crate::encodings::physical::{range::RangeEncoder, rle::RleEncoder, value::ValueEncoder};

    use super::*;

    #[test]
    fn delta_round_trip_with_payload() {
        let input = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(vec![5_u64, 5, 9, 12]),
            num_values: 4,
            block_info: BlockInfo::default(),
        });
        let encoder = DeltaEncoder::new(64, 5, Box::new(ValueEncoder::default()));
        let (payload, encoding) = encoder.compress(input).unwrap();
        let crate::format::pb21::compressive_encoding::Compression::Delta(delta) =
            encoding.compression.as_ref().unwrap()
        else {
            panic!("expected Delta encoding");
        };
        let child = crate::compression::DefaultDecompressionStrategy::default()
            .create_block_decompressor(delta.deltas.as_deref().unwrap())
            .unwrap();
        let decoded = DeltaDecompressor::new(64, 5, child)
            .decompress(payload, 4)
            .unwrap()
            .as_fixed_width()
            .unwrap();
        assert_eq!(
            decoded.data.borrow_to_typed_slice::<u64>().as_ref(),
            &[5, 5, 9, 12]
        );
    }

    #[test]
    fn delta_propagates_metadata_only_child() {
        let input = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(vec![0_u32, 1, 3, 6]),
            num_values: 4,
            block_info: BlockInfo::default(),
        });
        let encoder = DeltaEncoder::new(32, 0, Box::new(RangeEncoder::new(32, 1, 1)));
        let (payload, encoding) = encoder.compress(input).unwrap();
        assert!(payload.is_none());

        let decoder = crate::compression::DefaultDecompressionStrategy::default()
            .create_block_decompressor(&encoding)
            .unwrap();
        assert!(!decoder.requires_payload());
        let decoded = decoder
            .decompress(None, 4)
            .unwrap()
            .as_fixed_width()
            .unwrap();
        assert_eq!(
            decoded.data.borrow_to_typed_slice::<u32>().as_ref(),
            &[0, 1, 3, 6]
        );
    }

    #[test]
    fn delta_rejects_decreasing_input() {
        let input = FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(vec![2_u32, 1]),
            num_values: 2,
            block_info: BlockInfo::default(),
        };
        let error = encode_deltas(input, 2).unwrap_err();
        assert!(error.to_string().contains("decreases"));
    }

    #[test]
    fn delta_rejects_unsupported_rle_child_before_emitting_encoding() {
        let input = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(vec![10_u32, 10, 12, 12, 15]),
            num_values: 5,
            block_info: BlockInfo::default(),
        });
        let encoder = DeltaEncoder::new(32, 10, Box::new(RleEncoder::new()));

        let error = encoder.compress(input).unwrap_err();

        assert!(matches!(&error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("does not support a rle child"));
    }

    #[test]
    fn range_and_delta_return_errors_in_miniblock_positions() {
        let range = ProtobufUtils21::range(32, 0, 1);
        let delta = ProtobufUtils21::delta(32, 0, range.clone());
        let strategy = DefaultDecompressionStrategy::default();

        for encoding in [&range, &delta] {
            let Err(error) = strategy.create_miniblock_decompressor(encoding, &strategy) else {
                panic!("expected block-only encoding to be rejected in a mini-block position");
            };
            assert!(matches!(&error, Error::NotSupported { .. }));
            assert!(
                error
                    .to_string()
                    .contains("only supported in block positions")
            );
        }
    }
}
