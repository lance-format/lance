// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Price FSST expansion from borrowed values before allocating selected output.

use std::ops::Range;

use arrow_buffer::Buffer;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{FsstMiniBlockDecompressor, get_fsst_decompressor, map_fsst_error};
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecodeBudget, MiniBlockOutputLayout},
    data::{BlockInfo, DataBlock, VariableWidthBlock},
};

pub(super) fn layout(codec: &FsstMiniBlockDecompressor) -> Result<MiniBlockOutputLayout> {
    get_fsst_decompressor(&codec.decompressor)?;
    match codec.inner_decompressor.admitted_output_layout()? {
        shape @ MiniBlockOutputLayout::VariableWidth {
            bits_per_offset: 32 | 64,
        } => Ok(shape),
        _ => Err(Error::not_supported(
            "FSST admission requires binary values",
        )),
    }
}

fn width(codec: &FsstMiniBlockDecompressor) -> Result<usize> {
    let MiniBlockOutputLayout::VariableWidth { bits_per_offset } = layout(codec)? else {
        return Err(Error::internal("FSST admitted layout is not binary"));
    };
    Ok(usize::from(bits_per_offset / 8))
}

pub(super) fn plan(
    codec: &FsstMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
) -> Result<MiniBlockDecodeBudget> {
    if range.start > range.end || range.end > num_values {
        return Err(Error::corrupt_file_named(
            "FSST",
            "selection exceeds value count",
        ));
    }
    let width = width(codec)?;
    let decoder = get_fsst_decompressor(&codec.decompressor)?;
    let mut bytes = 0_usize;
    let mut row = 0_u64;
    codec
        .inner_decompressor
        .visit_binary_values(data, num_values, &mut |value| {
            // Validate even values outside the selection, including an empty range.
            let size = decoder
                .decompressed_value_len(value)
                .map_err(map_fsst_error)?;
            if range.contains(&row) {
                bytes = bytes
                    .checked_add(size)
                    .ok_or_else(|| Error::invalid_input("FSST output size overflows"))?;
            }
            row = row
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("FSST value count overflows"))?;
            Ok(())
        })?;
    if row != num_values {
        return Err(Error::corrupt_file_named(
            "FSST",
            "inner value count disagrees with chunk",
        ));
    }
    let offsets = (range.end - range.start)
        .checked_add(1)
        .and_then(|n| n.checked_mul(width as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid_input("FSST output offset size overflows"))?;
    let output_bytes = offsets
        .checked_add(bytes)
        .ok_or_else(|| Error::invalid_input("FSST output size overflows"))?;
    layout(codec)?.validate_range_bytes(range.end - range.start, output_bytes)?;
    Ok(MiniBlockDecodeBudget {
        output_bytes,
        workspace_bytes: output_bytes,
    })
}

pub(super) fn decode(
    codec: &FsstMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let budget = plan(codec, data, num_values, range.clone())?;
    if budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let width = width(codec)?;
    let count = range.end - range.start;
    let decoder = get_fsst_decompressor(&codec.decompressor)?;
    // Planning checked both products against usize and Arrow's signed offsets.
    let offset_bytes = (count as usize + 1) * width;
    let mut offsets = workspace
        .try_allocate(offset_bytes)?
        .ok_or_else(|| Error::internal("FSST offsets exceeded complete admission"))?;
    let mut values = match workspace.try_allocate(budget.output_bytes - offset_bytes) {
        Ok(Some(values)) => values,
        other => {
            offsets.recycle(workspace)?;
            return other.map(|_| None);
        }
    };
    offsets.as_mut()[..width].fill(0);
    let mut row = 0_u64;
    let mut written = 0_usize;
    let result = codec
        .inner_decompressor
        .visit_binary_values(data, num_values, &mut |value| {
            if range.contains(&row) {
                let size = decoder
                    .decompressed_value_len(value)
                    .map_err(map_fsst_error)?;
                let end = written
                    .checked_add(size)
                    .filter(|end| *end <= values.as_ref().len())
                    .ok_or_else(|| {
                        Error::corrupt_file_named("FSST", "replay exceeds planned output")
                    })?;
                decoder
                    .decompress_value_into(value, &mut values.as_mut()[written..end])
                    .map_err(map_fsst_error)?;
                written = end;
                let offset = (row - range.start + 1) as usize * width;
                offsets.as_mut()[offset..offset + width]
                    .copy_from_slice(&(written as u64).to_le_bytes()[..width]);
            }
            row = row
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("FSST value count overflows"))?;
            Ok(())
        })
        .and_then(|()| {
            if row == num_values && written == values.as_ref().len() {
                Ok(())
            } else {
                Err(Error::corrupt_file_named(
                    "FSST",
                    "replay disagrees with planned output",
                ))
            }
        });
    if let Err(error) = result {
        offsets.recycle(workspace)?;
        values.recycle(workspace)?;
        return Err(error);
    }
    Ok(Some(DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::from(Buffer::from(values.freeze())),
        offsets: LanceBuffer::from(Buffer::from(offsets.freeze())),
        bits_per_offset: (width * 8) as u8,
        num_values: count,
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests;
