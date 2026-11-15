// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Borrow or admit already unzipped raw values without alignment scratch.

use arrow_buffer::Buffer;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{MiniBlockOutputLayout, PerValueDecodeBudget};
use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, VariableWidthBlock},
};

fn offset<const WIDTH: usize>(bytes: &[u8]) -> u64 {
    let mut word = [0_u8; 8];
    word[..WIDTH].copy_from_slice(bytes);
    u64::from_le_bytes(word)
}

// Keep the signed limit and byte width constant inside the traversal. Byte
// chunks also support deliberately unaligned slices without an alignment copy.
fn validate_offsets<const WIDTH: usize>(bytes: &[u8], payload_len: usize) -> Result<usize> {
    let max = if WIDTH == 4 {
        i32::MAX as u64
    } else {
        i64::MAX as u64
    };
    let mut previous = 0;
    for (i, bytes) in bytes.chunks_exact(WIDTH).enumerate() {
        let next = offset::<WIDTH>(bytes);
        if (i == 0 && next != 0) || next < previous || next > max || next > payload_len as u64 {
            return Err(Error::corrupt_file_named(
                "per-value binary",
                "offsets must start at zero, remain monotonic and fit the signed input bounds",
            ));
        }
        previous = next;
    }
    Ok(previous as usize)
}

// Return offset width and logical payload length only after the
// entire table is validated. In particular, an invalid tail cannot be hidden
// by an empty workspace or by a callback failing on an earlier value.
fn validate(data: &VariableWidthBlock) -> Result<(usize, usize)> {
    let width = match data.bits_per_offset {
        32 => 4,
        64 => 8,
        _ => return Err(Error::invalid_input("binary offset width must be 32 or 64")),
    };
    let count = usize::try_from(data.num_values)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| Error::invalid_input("binary offset count overflows"))?;
    let bytes = count
        .checked_mul(width)
        .ok_or_else(|| Error::invalid_input("binary offset size overflows"))?;
    if data.offsets.len() != bytes {
        return Err(Error::corrupt_file_named(
            "per-value binary",
            "offset table size disagrees with value count",
        ));
    }
    let payload = if width == 4 {
        validate_offsets::<4>(&data.offsets, data.data.len())?
    } else {
        validate_offsets::<8>(&data.offsets, data.data.len())?
    };
    Ok((width, payload))
}

pub fn visit(
    data: &VariableWidthBlock,
    visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let (width, _) = validate(data)?;
    if width == 4 {
        visit_values::<4>(data, visitor)
    } else {
        visit_values::<8>(data, visitor)
    }
}

fn visit_values<const WIDTH: usize>(
    data: &VariableWidthBlock,
    visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    // Full validation precedes callbacks. Reuse the previous endpoint rather
    // than decoding each interior offset again for the following value.
    let mut start = 0;
    for bytes in data.offsets.chunks_exact(WIDTH).skip(1) {
        let end = offset::<WIDTH>(bytes) as usize;
        visitor(&data.data[start..end])?;
        start = end;
    }
    Ok(())
}

pub fn plan(data: &VariableWidthBlock) -> Result<PerValueDecodeBudget> {
    let (_, payload_bytes) = validate(data)?;
    let output_bytes = data
        .offsets
        .len()
        .checked_add(payload_bytes)
        .ok_or_else(|| Error::invalid_input("binary output allocation size overflows"))?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: data.bits_per_offset,
    }
    .validate_range_bytes(data.num_values, output_bytes)?;
    Ok(PerValueDecodeBudget {
        output_bytes,
        workspace_bytes: output_bytes,
    })
}

pub fn decode(data: &VariableWidthBlock, workspace: &mut Reservation) -> Result<Option<DataBlock>> {
    let budget = plan(data)?;
    if budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let mut offsets = workspace
        .try_allocate(data.offsets.len())?
        .ok_or_else(|| Error::internal("binary offsets exceeded complete admission"))?;
    let payload_bytes = budget.output_bytes - data.offsets.len();
    let mut values = match workspace.try_allocate(payload_bytes) {
        Ok(Some(values)) => values,
        other => {
            offsets.recycle(workspace)?;
            return other.map(|_| None);
        }
    };
    // The caller keeps the complete input owner charged during these copies.
    // Owning exact output allocations detaches partial results from coalesced
    // input backing and trailing unused bytes; no alignment copy is needed.
    offsets.as_mut().copy_from_slice(&data.offsets);
    values.as_mut().copy_from_slice(&data.data[..payload_bytes]);
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
