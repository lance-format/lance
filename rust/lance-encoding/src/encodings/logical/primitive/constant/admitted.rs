// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Complete expanded-value admission for flat constant pages without rep/def buffers.

use arrow_array::{Array, cast::AsArray};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use crate::{
    buffer::LanceBuffer,
    data::{AllNullDataBlock, BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
};

fn allocate(workspace: &mut Reservation, bytes: usize) -> Result<ReservedBuffer> {
    workspace
        .try_allocate(bytes)?
        .ok_or_else(|| Error::internal("constant output exceeded its complete admission"))
}

fn fixed(values: &[u8], count: usize, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
    let Some(bytes) = values.len().checked_mul(count) else {
        return Ok(None);
    };
    if bytes > workspace.len() {
        return Ok(None);
    }
    let mut output = allocate(workspace, bytes)?;
    if !values.is_empty() {
        for slot in output.as_mut().chunks_exact_mut(values.len()) {
            slot.copy_from_slice(values);
        }
    }
    Ok(Some(DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::from(Buffer::from(output.freeze())),
        num_values: count as u64,
        bits_per_value: (values.len() as u64) * 8,
        block_info: BlockInfo::new(),
    })))
}

fn binary(
    value: &[u8],
    count: usize,
    wide: bool,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let Some(bytes) = value.len().checked_mul(count) else {
        return Ok(None);
    };
    let max_offset = if wide {
        i64::MAX as u64
    } else {
        i32::MAX as u64
    };
    // A batch that cannot fit the existing offset type must be split, not silently
    // truncated or treated as a total-result cap. No output has been allocated yet.
    if bytes as u64 > max_offset {
        return Ok(None);
    }
    let offset_width = if wide { 8 } else { 4 };
    let Some(offset_bytes) = count
        .checked_add(1)
        .and_then(|n| n.checked_mul(offset_width))
    else {
        return Ok(None);
    };
    let Some(total) = bytes.checked_add(offset_bytes) else {
        return Ok(None);
    };
    if total > workspace.len() {
        return Ok(None);
    }
    let mut data = allocate(workspace, bytes)?;
    let mut offsets = allocate(workspace, offset_bytes)?;
    if !value.is_empty() {
        for slot in data.as_mut().chunks_exact_mut(value.len()) {
            slot.copy_from_slice(value);
        }
    }
    if wide {
        for (i, slot) in offsets
            .as_mut()
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .enumerate()
        {
            slot.copy_from_slice(&((i * value.len()) as i64).to_le_bytes());
        }
    } else {
        for (i, slot) in offsets
            .as_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .enumerate()
        {
            slot.copy_from_slice(&((i * value.len()) as i32).to_le_bytes());
        }
    }
    Ok(Some(DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::from(Buffer::from(data.freeze())),
        offsets: LanceBuffer::from(Buffer::from(offsets.freeze())),
        bits_per_offset: (offset_width * 8) as u8,
        num_values: count as u64,
        block_info: BlockInfo::new(),
    })))
}

pub(super) fn repeat(
    scalar: &dyn Array,
    count: u64,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    if scalar.len() != 1 {
        return Err(Error::invalid_input(
            "constant page needs exactly one scalar",
        ));
    }
    if scalar.is_null(0) {
        // Preserve the structural sentinel. Typed materialization of its values,
        // offsets and validity must use DataBlock::try_to_arrow_in afterward.
        return Ok(Some(DataBlock::AllNull(AllNullDataBlock {
            num_values: count,
        })));
    }
    let count = usize::try_from(count)
        .map_err(|_| Error::invalid_input("constant output row count exceeds usize"))?;
    let data_type = scalar.data_type();
    let width = match data_type {
        DataType::FixedSizeBinary(width) => Some(
            usize::try_from(*width)
                .map_err(|_| Error::invalid_input("negative constant byte width"))?,
        ),
        _ => data_type.primitive_width(),
    };
    if let Some(width) = width {
        let source = scalar.to_data();
        let start = source
            .offset()
            .checked_mul(width)
            .ok_or_else(|| Error::invalid_input("constant scalar offset overflows"))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Error::invalid_input("constant scalar width overflows"))?;
        let value = source
            .buffers()
            .first()
            .and_then(|buffer| buffer.get(start..end))
            .ok_or_else(|| {
                Error::invalid_input("constant scalar buffer is shorter than its typed layout")
            })?;
        return fixed(value, count, workspace);
    }
    match data_type {
        DataType::Boolean => {
            let bytes = count.div_ceil(8);
            if bytes > workspace.len() {
                return Ok(None);
            }
            let mut output = allocate(workspace, bytes)?;
            if scalar.as_boolean().value(0) {
                output.as_mut().fill(255);
                let remainder = count % 8;
                if remainder != 0 {
                    output.as_mut()[bytes - 1] = (1 << remainder) - 1;
                }
            }
            Ok(Some(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(Buffer::from(output.freeze())),
                num_values: count as u64,
                bits_per_value: 1,
                block_info: BlockInfo::new(),
            })))
        }
        DataType::Binary => binary(scalar.as_binary::<i32>().value(0), count, false, workspace),
        DataType::LargeBinary => binary(scalar.as_binary::<i64>().value(0), count, true, workspace),
        DataType::Utf8 => binary(
            scalar.as_string::<i32>().value(0).as_bytes(),
            count,
            false,
            workspace,
        ),
        DataType::LargeUtf8 => binary(
            scalar.as_string::<i64>().value(0).as_bytes(),
            count,
            true,
            workspace,
        ),
        other => Err(Error::not_supported(format!(
            "constant page has no admitted expansion for {other}"
        ))),
    }
}

#[cfg(test)]
mod tests;
