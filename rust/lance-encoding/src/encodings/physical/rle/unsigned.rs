// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Visit generic RLE without decoded child arrays or alignment copies.

use super::{
    BlockRleDecompressor, BlockRleRunCount, infer_generic_run_count, split_generic_rle_payload,
};
use crate::buffer::LanceBuffer;
use lance_core::{Error, Result};

pub(super) fn visit(
    codec: &BlockRleDecompressor,
    data: Option<&LanceBuffer>,
    bits: u64,
    count: u64,
    visitor: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    if bits != codec.bits_per_value || count == 0 {
        return Err(Error::invalid_input(format!(
            "unsigned RLE visitor expected nonempty {}-bit values, got {count} {bits}-bit values",
            codec.bits_per_value
        )));
    }
    // Cloning/slicing LanceBuffer retains the same charged backing owner.
    let (values, lengths) = split_generic_rle_payload(
        data.cloned(),
        codec.values.requires_payload(),
        codec.run_lengths.requires_payload(),
    )?;
    let runs = infer_generic_run_count(
        codec.run_count,
        values.as_ref(),
        lengths.as_ref(),
        bits,
        count,
    )?;

    // Validate every run before expanding any values, including late bad lengths.
    let mut seen = 0;
    let mut total = 0_u64;
    codec
        .run_lengths
        .visit_unsigned(lengths.as_ref(), 32, runs, &mut |length| {
            if seen >= runs || length == 0 {
                return Err(Error::invalid_input(format!(
                    "RLE run {seen} has invalid length {length} or exceeds {runs} runs"
                )));
            }
            total = total
                .checked_add(length)
                .filter(|total| *total <= count)
                .ok_or_else(|| Error::invalid_input("RLE run length sum exceeds value count"))?;
            seen += 1;
            Ok(())
        })?;
    if seen != runs || total != count {
        return Err(Error::invalid_input(format!(
            "RLE run lengths describe {seen} runs / {total} values, expected {runs} / {count}"
        )));
    }

    // The grammar guarantees a metadata or Flat child, so one side is directly
    // addressable while the other streams. Never buffer one child to zip them.
    let mut index = 0;
    let mut emitted = 0_u64;
    let mut emit = |value, length| {
        emitted = emitted
            .checked_add(length)
            .filter(|total| *total <= count)
            .ok_or_else(|| Error::invalid_input("RLE replay exceeds value count"))?;
        for _ in 0..length {
            visitor(value)?;
        }
        Ok(())
    };
    match codec.run_count {
        BlockRleRunCount::FlatValues => {
            codec
                .run_lengths
                .visit_unsigned(lengths.as_ref(), 32, runs, &mut |length| {
                    let value = flat_value(values.as_ref(), bits, index)?;
                    index += 1;
                    emit(value, length)
                })?;
        }
        source => {
            codec
                .values
                .visit_unsigned(values.as_ref(), bits, runs, &mut |value| {
                    if index >= runs {
                        return Err(Error::invalid_input("RLE values exceed run count"));
                    }
                    let length = match source {
                        BlockRleRunCount::ConstantRunLength(length) => length,
                        // The first length pass validated this range at 32 bits.
                        BlockRleRunCount::RangeRunLengths { start, step } => start + index * step,
                        BlockRleRunCount::FlatRunLengths => {
                            flat_value(lengths.as_ref(), 32, index)?
                        }
                        BlockRleRunCount::FlatValues => {
                            return Err(Error::internal("RLE run source changed during visit"));
                        }
                    };
                    index += 1;
                    emit(value, length)
                })?;
        }
    }
    if index != runs || emitted != count {
        return Err(Error::invalid_input(format!(
            "RLE replay visited {index} runs / {emitted} values, expected {runs} / {count}"
        )));
    }
    Ok(())
}

fn flat_value(data: Option<&LanceBuffer>, bits: u64, index: u64) -> Result<u64> {
    let data = data.ok_or_else(|| Error::invalid_input("RLE Flat child is missing its payload"))?;
    let width = (bits / 8) as usize;
    let start = usize::try_from(index)
        .ok()
        .and_then(|index| index.checked_mul(width))
        .ok_or_else(|| Error::invalid_input("RLE Flat child position overflows"))?;
    let bytes = data
        .get(start..)
        .and_then(|tail| tail.get(..width))
        .ok_or_else(|| Error::invalid_input("RLE Flat child has too few values"))?;
    let mut value = [0; 8];
    value[..width].copy_from_slice(bytes);
    Ok(u64::from_le_bytes(value))
}

#[cfg(test)]
mod tests;
