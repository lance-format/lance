// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat-child RLE ranges with no decoded child or alignment scratch.

use std::ops::Range;

use arrow_buffer::{Buffer, alloc::ALIGNMENT};
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::RleDecompressor;
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecodeBudget, MiniBlockOutputLayout},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

pub(super) fn layout(codec: &RleDecompressor) -> Result<MiniBlockOutputLayout> {
    if !matches!(codec.bits_per_value, 8 | 16 | 32 | 64) {
        return Err(Error::invalid_input(format!(
            "admitted RLE values require 8, 16, 32 or 64 bits, got {}",
            codec.bits_per_value
        )));
    }
    if !codec.values.is_identity() || !codec.run_lengths.is_identity() {
        return Err(Error::not_supported(
            "admitted RLE values require flat value and run-length children",
        ));
    }
    if codec.values.bits_per_value() != codec.bits_per_value
        || codec.run_lengths.bits_per_value() != codec.run_length_width.bits_per_value()
    {
        return Err(Error::invalid_input(
            "RLE child widths disagree with value/run-length metadata",
        ));
    }
    Ok(MiniBlockOutputLayout::FixedWidth {
        bits_per_value: codec.bits_per_value,
    })
}

pub(super) fn plan(
    codec: &RleDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
) -> Result<MiniBlockDecodeBudget> {
    let layout = layout(codec)?;
    if range.start > range.end || range.end > num_values {
        return Err(Error::corrupt_file_named(
            "rle",
            format!("RLE mini-block selection {range:?} exceeds {num_values} values in chunk"),
        ));
    }
    let output_bytes = layout.payload_bytes(range.end - range.start)?;
    std::alloc::Layout::from_size_align(output_bytes, ALIGNMENT)
        .map_err(|_| Error::invalid_input("RLE output exceeds addressable allocation size"))?;
    let budget = MiniBlockDecodeBudget {
        output_bytes,
        workspace_bytes: output_bytes,
    };
    // The ordinary MiniBlock reader accepts a canonical empty chunk with no buffers.
    if num_values == 0 {
        return Ok(budget);
    }
    let [values, lengths] = data else {
        return Err(Error::invalid_input(format!(
            "RLE decompressor expects exactly 2 buffers, got {}",
            data.len()
        )));
    };
    let length_bytes = codec.validate_buffer_sizes(
        values,
        lengths,
        (codec.bits_per_value / 8) as usize,
        "admitted values",
    )?;
    let mut position = 0_u64;
    for bytes in lengths.chunks_exact(length_bytes) {
        let length = codec.run_length_width.read_length(bytes);
        if length == 0 {
            return Err(Error::invalid_input(
                "RLE decoding encountered a zero run length",
            ));
        }
        let remaining = num_values - position;
        position += length.min(remaining);
        // Preserve the ordinary MiniBlock reader's chunk-boundary semantics.
        // Block-format structural RLE has a stricter exact-sum contract.
        if length > remaining {
            break;
        }
    }
    if position != num_values {
        return Err(Error::invalid_input(format!(
            "RLE decoding produced {position} values, expected {num_values}"
        )));
    }
    Ok(budget)
}

pub(super) fn decode(
    codec: &RleDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let budget = plan(codec, data, num_values, range.clone())?;
    let Some(mut output) = workspace.try_allocate(budget.workspace_bytes)? else {
        return Ok(None);
    };
    if budget.output_bytes > 0 {
        let width = (codec.bits_per_value / 8) as usize;
        let length_width = codec.run_length_width.bytes_per_value();
        let mut position = 0_u64;
        let mut written = 0;
        for (value, length) in data[0]
            .chunks_exact(width)
            .zip(data[1].chunks_exact(length_width))
        {
            let length = codec.run_length_width.read_length(length);
            let remaining = num_values - position;
            let end = position + length.min(remaining);
            let selected = end.min(range.end).saturating_sub(position.max(range.start));
            // The selected count and complete input were checked before admission.
            let bytes = selected as usize * width;
            for destination in output.as_mut()[written..written + bytes].chunks_exact_mut(width) {
                destination.copy_from_slice(value);
            }
            written += bytes;
            position = end;
            if length > remaining || position >= range.end {
                break;
            }
        }
        debug_assert_eq!(written, budget.output_bytes);
    }
    Ok(Some(DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: codec.bits_per_value,
        num_values: range.end - range.start,
        data: LanceBuffer::from(Buffer::from(output.freeze())),
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests;
