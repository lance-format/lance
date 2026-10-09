// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routines for compressing and decompressing constant-encoded data

use crate::{
    buffer::LanceBuffer,
    compression::{
        BlockCompressor, BlockDecompressor, FixedPerValueDecompressor, require_no_block_payload,
    },
    data::{AllNullDataBlock, ConstantDataBlock, DataBlock, FixedWidthDataBlock},
    encodings::physical::{checked_fixed_values, try_vec_with_capacity},
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};

use lance_core::{Error, Result};

/// Metadata-only compressor for a repeated unsigned `u32` or `u64` value.
#[derive(Debug)]
pub struct ConstantEncoder {
    bits_per_value: u64,
    value: u64,
}

impl ConstantEncoder {
    pub fn new(bits_per_value: u64, value: u64) -> Self {
        Self {
            bits_per_value,
            value,
        }
    }

    /// Describes a known constant sequence without materializing its values.
    /// Zero values have no scalar; non-empty sequences store one typed scalar.
    pub(crate) fn encoding(&self, num_values: u64) -> Result<CompressiveEncoding> {
        if !matches!(self.bits_per_value, 32 | 64) {
            return Err(Error::invalid_input(format!(
                "Constant block compression only supports 32 or 64-bit values, got {}",
                self.bits_per_value
            )));
        }
        if num_values == 0 {
            return Ok(ProtobufUtils21::constant(None));
        }
        let scalar = if self.bits_per_value == 32 {
            let value = u32::try_from(self.value).map_err(|_| {
                Error::invalid_input(format!("Constant value {} exceeds u32::MAX", self.value))
            })?;
            bytes::Bytes::copy_from_slice(&value.to_le_bytes())
        } else {
            bytes::Bytes::copy_from_slice(&self.value.to_le_bytes())
        };
        Ok(ProtobufUtils21::constant(Some(scalar)))
    }
}

impl BlockCompressor for ConstantEncoder {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Constant block compression requires fixed-width data",
            ));
        };
        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(format!(
                "Constant codec expects {}-bit values, got {}",
                self.bits_per_value, data.bits_per_value
            )));
        }
        let encoding = self.encoding(data.num_values)?;
        match self.bits_per_value {
            32 => {
                if checked_fixed_values::<u32>(&data, "Constant input")?
                    .iter()
                    .any(|candidate| u64::from(*candidate) != self.value)
                {
                    return Err(Error::invalid_input(
                        "Constant input contains a different value",
                    ));
                }
            }
            64 => {
                if checked_fixed_values::<u64>(&data, "Constant input")?
                    .iter()
                    .any(|candidate| *candidate != self.value)
                {
                    return Err(Error::invalid_input(
                        "Constant input contains a different value",
                    ));
                }
            }
            _ => unreachable!("constant width was validated above"),
        };
        Ok((None, encoding))
    }
}

/// Materializes a metadata-only constant as a typed fixed-width block.
#[derive(Debug)]
pub(crate) struct ConstantBlockDecompressor {
    bits_per_value: u64,
    value: Option<u64>,
}

impl ConstantBlockDecompressor {
    pub(crate) fn new(bits_per_value: u64, value: Option<u64>) -> Self {
        Self {
            bits_per_value,
            value,
        }
    }
}

impl BlockDecompressor for ConstantBlockDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        require_no_block_payload(data, "Constant")?;
        if self.value.is_none() != (num_values == 0) {
            return Err(Error::invalid_input(
                "Constant scalar must be absent exactly when cardinality is zero",
            ));
        }
        let value = self.value.unwrap_or(0);
        let output_len = usize::try_from(num_values)
            .map_err(|_| Error::invalid_input("Constant output cardinality does not fit usize"))?;
        let data = match self.bits_per_value {
            32 => {
                let value = u32::try_from(value).map_err(|_| {
                    Error::invalid_input(format!("Constant value {value} exceeds u32::MAX"))
                })?;
                let mut values = try_vec_with_capacity::<u32>(num_values, "Constant output")?;
                values.resize(output_len, value);
                LanceBuffer::reinterpret_vec(values)
            }
            64 => {
                let mut values = try_vec_with_capacity::<u64>(num_values, "Constant output")?;
                values.resize(output_len, value);
                LanceBuffer::reinterpret_vec(values)
            }
            bits_per_value => {
                return Err(Error::invalid_input(format!(
                    "Constant block decompression only supports 32 or 64-bit values, got {bits_per_value}"
                )));
            }
        };
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            data,
            num_values,
            block_info: Default::default(),
        }))
    }

    fn requires_payload(&self) -> bool {
        false
    }
}

/// A decompressor for constant-encoded data
#[derive(Debug)]
pub struct ConstantDecompressor {
    scalar: Option<LanceBuffer>,
}

impl ConstantDecompressor {
    pub fn new(scalar: Option<LanceBuffer>) -> Self {
        Self { scalar }
    }
}

impl BlockDecompressor for ConstantDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        require_no_block_payload(data, "Constant")?;
        if let Some(scalar) = self.scalar.clone() {
            Ok(DataBlock::Constant(ConstantDataBlock {
                data: scalar,
                num_values,
            }))
        } else {
            Ok(DataBlock::AllNull(AllNullDataBlock { num_values }))
        }
    }

    fn requires_payload(&self) -> bool {
        false
    }
}

impl FixedPerValueDecompressor for ConstantDecompressor {
    fn decompress(&self, _data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock> {
        if let Some(scalar) = self.scalar.clone() {
            Ok(DataBlock::Constant(ConstantDataBlock {
                data: scalar,
                num_values,
            }))
        } else {
            Ok(DataBlock::AllNull(AllNullDataBlock { num_values }))
        }
    }

    fn bits_per_value(&self) -> u64 {
        self.scalar
            .as_ref()
            .map(|s| s.len() as u64 * 8)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case(32)]
    #[case(64)]
    fn constant_metadata_round_trip(#[case] bits: u64) {
        let encoder = ConstantEncoder::new(bits, 7);
        for count in [0, 3] {
            let expected = if bits == 32 {
                LanceBuffer::reinterpret_vec(vec![7_u32; count as usize])
            } else {
                LanceBuffer::reinterpret_vec(vec![7_u64; count as usize])
            };
            let (payload, encoding) = encoder
                .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: expected.clone(),
                    bits_per_value: bits,
                    num_values: count,
                    block_info: Default::default(),
                }))
                .unwrap();
            assert!(payload.is_none());
            let decoder = crate::encodings::physical::sequence::SequenceDecoder::try_new(
                &encoding, bits, count,
            )
            .unwrap();
            let decoded = decoder.decode(payload).unwrap().as_fixed_width().unwrap();
            assert_eq!(decoded.num_values, count);
            assert_eq!(decoded.data.as_ref(), expected.as_ref());
        }
        assert_eq!(
            encoder.encoding(u64::MAX).unwrap(),
            encoder.encoding(1).unwrap()
        );
    }

    #[test]
    fn constant_metadata_validates_width_and_scalar() {
        for encoder in [
            ConstantEncoder::new(16, 1),
            ConstantEncoder::new(32, u64::MAX),
        ] {
            let error = encoder.encoding(1).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("Constant"));
        }
    }

    #[test]
    fn block_constant_requires_no_payload() {
        let decompressor = ConstantDecompressor::new(None);

        assert!(!decompressor.requires_payload());
        assert!(matches!(
            BlockDecompressor::decompress(&decompressor, None, 3).unwrap(),
            DataBlock::AllNull(AllNullDataBlock { num_values: 3 })
        ));
        assert!(
            BlockDecompressor::decompress(&decompressor, Some(LanceBuffer::empty()), 3).is_err()
        );
    }
}
