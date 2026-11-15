// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact FSST output admission after FullZip has separated values and offsets.

use arrow_buffer::Buffer;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{FsstPerValueDecompressor, get_fsst_decompressor, map_fsst_error};
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockOutputLayout, PerValueDecodeBudget},
    data::{BlockInfo, DataBlock, VariableWidthBlock},
};

pub(super) fn plan(
    codec: &FsstPerValueDecompressor,
    data: &VariableWidthBlock,
) -> Result<PerValueDecodeBudget> {
    if !matches!(data.bits_per_offset, 32 | 64) {
        return Err(Error::invalid_input("FSST offset width must be 32 or 64"));
    }
    let decoder = get_fsst_decompressor(&codec.decompressor)?;
    let mut bytes = 0_usize;
    let mut count = 0_u64;
    codec
        .inner_decompressor
        .visit_binary_values(data, &mut |value| {
            bytes = bytes
                .checked_add(
                    decoder
                        .decompressed_value_len(value)
                        .map_err(map_fsst_error)?,
                )
                .ok_or_else(|| Error::invalid_input("FSST expanded payload size overflows"))?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("FSST value count overflows"))?;
            Ok(())
        })?;
    if count != data.num_values {
        return Err(Error::corrupt_file_named(
            "FSST",
            "inner value count disagrees with block",
        ));
    }
    let offsets = count
        .checked_add(1)
        .and_then(|n| n.checked_mul(u64::from(data.bits_per_offset / 8)))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid_input("FSST offset allocation size overflows"))?;
    let output_bytes = bytes
        .checked_add(offsets)
        .ok_or_else(|| Error::invalid_input("FSST output allocation size overflows"))?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: data.bits_per_offset,
    }
    .validate_range_bytes(count, output_bytes)?;
    Ok(PerValueDecodeBudget {
        output_bytes,
        workspace_bytes: output_bytes,
    })
}

pub(super) fn decode(
    codec: &FsstPerValueDecompressor,
    data: &VariableWidthBlock,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let budget = plan(codec, data)?;
    if budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let decoder = get_fsst_decompressor(&codec.decompressor)?;
    let width = usize::from(data.bits_per_offset / 8);
    // The plan checked both products and Arrow's signed offset limit.
    let offset_bytes = (data.num_values as usize + 1) * width;
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
        .visit_binary_values(data, &mut |value| {
            if row >= data.num_values {
                return Err(Error::corrupt_file_named(
                    "FSST",
                    "replay exceeds planned value count",
                ));
            }
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
            row += 1;
            let offset = row as usize * width;
            offsets.as_mut()[offset..offset + width]
                .copy_from_slice(&(written as u64).to_le_bytes()[..width]);
            Ok(())
        })
        .and_then(|()| {
            if row == data.num_values && written == values.as_ref().len() {
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
        bits_per_offset: data.bits_per_offset,
        num_values: data.num_values,
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests;
