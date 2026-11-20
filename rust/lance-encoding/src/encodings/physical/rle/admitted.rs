// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! RLE ranges with borrowed child traversal and no decoded or alignment scratch.

use std::ops::Range;

use arrow_buffer::{Buffer, alloc::ALIGNMENT};
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::RleDecompressor;
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecodeBudget, MiniBlockOutputLayout},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

mod children;

pub(super) fn layout(codec: &RleDecompressor) -> Result<MiniBlockOutputLayout> {
    if !matches!(codec.bits_per_value, 8 | 16 | 32 | 64) {
        return Err(Error::invalid_input(format!(
            "admitted RLE values require 8, 16, 32 or 64 bits, got {}",
            codec.bits_per_value
        )));
    }
    if !codec.values.is_identity() && !codec.run_lengths.is_identity() {
        return Err(Error::not_supported(
            "admitted RLE values require at least one flat child to determine the run count",
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
    let mut position = 0_u64;
    let mut clamped = false;
    children::visit(codec, data, &mut |_, length| {
        if clamped {
            return Ok(());
        }
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
            clamped = true;
        }
        Ok(())
    })?;
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
        let mut position = 0_u64;
        let mut written = 0_usize;
        let copied = children::visit(codec, data, &mut |value, length| {
            if position >= range.end {
                return Ok(());
            }
            let remaining = num_values - position;
            let end = position + length.min(remaining);
            let selected = end.min(range.end).saturating_sub(position.max(range.start));
            // The selected count and complete input were checked before admission.
            let bytes = usize::try_from(selected)
                .ok()
                .and_then(|n| n.checked_mul(width))
                .and_then(|n| written.checked_add(n))
                .ok_or_else(|| Error::internal("RLE replay output size overflowed"))?;
            let destination = output
                .as_mut()
                .get_mut(written..bytes)
                .ok_or_else(|| Error::internal("RLE replay exceeded selected output size"))?;
            for destination in destination.chunks_exact_mut(width) {
                destination.copy_from_slice(&value.to_le_bytes()[..width]);
            }
            written = bytes;
            position = end;
            Ok(())
        })
        .and_then(|()| {
            if written != budget.output_bytes {
                return Err(Error::internal("RLE replay changed selected output size"));
            }
            Ok(())
        });
        if let Err(error) = copied {
            output.recycle(workspace)?;
            return Err(error);
        }
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
