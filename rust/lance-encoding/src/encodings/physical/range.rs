// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata-only arithmetic range encoding for unsigned block sequences.

use super::{checked_fixed_values, constant::ConstantEncoder, try_vec_with_capacity};
use crate::{
    buffer::LanceBuffer,
    compression::{BlockCompressor, BlockDecompressor, require_no_block_payload},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};
use lance_core::{Error, Result};

pub(crate) fn checked_range_last(
    bits_per_value: u64,
    start: u64,
    step: u64,
    num_values: u64,
) -> Result<u64> {
    if !matches!(bits_per_value, 32 | 64) {
        return Err(Error::invalid_input(format!(
            "Range only supports 32 or 64-bit values, got {bits_per_value}"
        )));
    }
    if step == 0 {
        return Err(Error::invalid_input("Range step must be positive"));
    }
    if num_values < 2 {
        return Err(Error::invalid_input(format!(
            "Range requires at least 2 values, got {num_values}"
        )));
    }

    let distance = step.checked_mul(num_values - 1).ok_or_else(|| {
        Error::invalid_input(format!(
            "Range step multiplication overflows: step={step}, num_values={num_values}"
        ))
    })?;
    let last = start.checked_add(distance).ok_or_else(|| {
        Error::invalid_input(format!(
            "Range final value overflows: start={start}, step={step}, num_values={num_values}"
        ))
    })?;
    if bits_per_value == 32 && last > u32::MAX as u64 {
        return Err(Error::invalid_input(format!(
            "Range final value {last} exceeds u32::MAX"
        )));
    }
    Ok(last)
}

#[derive(Debug)]
/// Encodes a validated arithmetic sequence without a payload buffer.
pub struct RangeEncoder {
    bits_per_value: u64,
    start: u64,
    step: u64,
}

impl RangeEncoder {
    /// Creates a range codec for `u32` or `u64` values.
    pub fn new(bits_per_value: u64, start: u64, step: u64) -> Self {
        Self {
            bits_per_value,
            start,
            step,
        }
    }

    /// Describes a known arithmetic sequence without materializing its values.
    /// Empty and singleton sequences use Constant; longer ranges are overflow-checked.
    pub(crate) fn encoding(&self, num_values: u64) -> Result<CompressiveEncoding> {
        if num_values < 2 {
            return ConstantEncoder::new(self.bits_per_value, self.start).encoding(num_values);
        }
        checked_range_last(self.bits_per_value, self.start, self.step, num_values)?;
        Ok(ProtobufUtils21::range(
            self.bits_per_value,
            self.start,
            self.step,
        ))
    }
}

impl BlockCompressor for RangeEncoder {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Range encoding requires a fixed-width data block",
            ));
        };
        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(format!(
                "Range codec expects {}-bit values, got {}",
                self.bits_per_value, data.bits_per_value
            )));
        }
        let encoding = self.encoding(data.num_values)?;

        match self.bits_per_value {
            32 => validate_values(
                checked_fixed_values::<u32>(&data, "Range input")?
                    .iter()
                    .map(|value| u64::from(*value)),
                self.start,
                self.step,
            )?,
            64 => validate_values(
                checked_fixed_values::<u64>(&data, "Range input")?
                    .iter()
                    .copied(),
                self.start,
                self.step,
            )?,
            _ => unreachable!("range width was validated above"),
        }
        Ok((None, encoding))
    }
}

fn validate_values(values: impl Iterator<Item = u64>, start: u64, step: u64) -> Result<()> {
    for (index, value) in values.enumerate() {
        let expected = step
            .checked_mul(index as u64)
            .and_then(|distance| start.checked_add(distance))
            .ok_or_else(|| Error::invalid_input("Range value calculation overflows"))?;
        if value != expected {
            return Err(Error::invalid_input(format!(
                "Range input mismatch at index {index}: expected {expected}, got {value}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct RangeDecompressor {
    bits_per_value: u64,
    start: u64,
    step: u64,
}

impl RangeDecompressor {
    pub(crate) fn new(bits_per_value: u64, start: u64, step: u64) -> Self {
        Self {
            bits_per_value,
            start,
            step,
        }
    }
}

impl BlockDecompressor for RangeDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        require_no_block_payload(data, "Range")?;
        checked_range_last(self.bits_per_value, self.start, self.step, num_values)?;
        materialize_range(self.bits_per_value, self.start, self.step, num_values)
    }

    fn requires_payload(&self) -> bool {
        false
    }
}

fn materialize_range(
    bits_per_value: u64,
    start: u64,
    step: u64,
    num_values: u64,
) -> Result<DataBlock> {
    let data = match bits_per_value {
        32 => {
            let mut current = u32::try_from(start).map_err(|_| {
                Error::invalid_input(format!("Range value {start} exceeds u32::MAX"))
            })?;
            let step = u32::try_from(step)
                .map_err(|_| Error::invalid_input(format!("Range step {step} exceeds u32::MAX")))?;
            let mut values = try_vec_with_capacity::<u32>(num_values, "Range output")?;
            for index in 0..num_values {
                values.push(current);
                if index + 1 < num_values {
                    current += step;
                }
            }
            LanceBuffer::reinterpret_vec(values)
        }
        64 => {
            let mut current = start;
            let mut values = try_vec_with_capacity::<u64>(num_values, "Range output")?;
            for index in 0..num_values {
                values.push(current);
                if index + 1 < num_values {
                    current += step;
                }
            }
            LanceBuffer::reinterpret_vec(values)
        }
        _ => unreachable!("range width was validated before materialization"),
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
    use super::*;

    #[rstest::rstest]
    #[case(32)]
    #[case(64)]
    fn range_metadata_normalizes_short_sequences(#[case] bits: u64) {
        let encoder = RangeEncoder::new(bits, 3, 5);
        for count in 0..=2 {
            let data = if bits == 32 {
                LanceBuffer::reinterpret_vec(
                    (0..count).map(|i| (3 + 5 * i) as u32).collect::<Vec<_>>(),
                )
            } else {
                LanceBuffer::reinterpret_vec((0..count).map(|i| 3 + 5 * i).collect::<Vec<u64>>())
            };
            let (payload, encoding) = encoder
                .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: data.clone(),
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
            assert_eq!(decoded.data.as_ref(), data.as_ref());
        }
    }

    #[test]
    fn range_metadata_checks_bounds_without_materializing() {
        let count = 1_000_000_000_000;
        let encoding = RangeEncoder::new(64, 3, 5).encoding(count).unwrap();
        let decoder =
            crate::encodings::physical::sequence::SequenceDecoder::try_new(&encoding, 64, count)
                .unwrap();
        assert_eq!(
            decoder.metadata(),
            Some(
                crate::encodings::physical::sequence::SequenceMetadata::Range { start: 3, step: 5 }
            )
        );
        for encoder in [
            RangeEncoder::new(32, u32::MAX as u64, 1),
            RangeEncoder::new(64, u64::MAX, 1),
            RangeEncoder::new(64, 0, 0),
        ] {
            let error = encoder.encoding(2).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("Range"));
        }
    }
    #[test]
    fn range_round_trip_u32() {
        let input = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(vec![3_u32, 8, 13, 18]),
            num_values: 4,
            block_info: BlockInfo::default(),
        });
        let (payload, encoding) = RangeEncoder::new(32, 3, 5).compress(input).unwrap();
        assert!(payload.is_none());
        assert_eq!(encoding, ProtobufUtils21::range(32, 3, 5));

        let decoded = RangeDecompressor::new(32, 3, 5)
            .decompress(payload, 4)
            .unwrap()
            .as_fixed_width()
            .unwrap();
        assert_eq!(
            decoded.data.borrow_to_typed_slice::<u32>().as_ref(),
            &[3, 8, 13, 18]
        );
    }

    #[test]
    fn range_rejects_overflow() {
        let error = checked_range_last(32, u32::MAX as u64, 1, 2).unwrap_err();
        assert!(error.to_string().contains("exceeds u32::MAX"));
    }
}
