// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Flat and metadata-only offsets with no implicit alignment copy or full expansion.

use std::ops::Range;

use arrow_buffer::Buffer;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{BinaryMiniBlockDecompressor, BinaryMiniBlockLayout};
use crate::{
    buffer::LanceBuffer,
    compression::{MiniBlockDecodeBudget, MiniBlockOutputLayout},
    data::{BlockInfo, DataBlock, VariableWidthBlock},
    encodings::physical::{range::checked_range_last, sequence::SequenceMetadata},
};

fn width(codec: &BinaryMiniBlockDecompressor) -> Result<u8> {
    match codec.layout {
        BinaryMiniBlockLayout::Legacy { bits_per_offset } => Ok(bits_per_offset),
        BinaryMiniBlockLayout::Generic {
            bits_per_offset,
            offsets_have_payload: false,
            ..
        } => Ok(bits_per_offset),
        BinaryMiniBlockLayout::Generic { .. } => Err(Error::not_supported(
            "admitted binary ranges require a separate compressed-offset plan",
        )),
    }
}

pub(super) fn layout(codec: &BinaryMiniBlockDecompressor) -> Result<MiniBlockOutputLayout> {
    Ok(MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: width(codec)?,
    })
}

// Read byte words, avoiding borrow_to_typed_slice's uncharged alignment copy.
fn offset(data: &[u8], index: usize, width: usize) -> u64 {
    let mut word = [0_u8; 8];
    word[..width].copy_from_slice(&data[index * width..(index + 1) * width]);
    u64::from_le_bytes(word)
}

pub(super) struct RangePlan {
    pub(super) budget: MiniBlockDecodeBudget,
    values: Range<usize>,
    offsets: Range<u64>,
    offset_bytes: usize,
    width: usize,
    // A validated arithmetic sequence; None means offsets reside in input bytes.
    metadata: Option<(u64, u64)>,
}

fn finish_plan(
    codec: &BinaryMiniBlockDecompressor,
    range: Range<u64>,
    values: Range<usize>,
    metadata: Option<(u64, u64)>,
) -> Result<RangePlan> {
    let width = usize::from(width(codec)? / 8);
    let offset_bytes = (range.end - range.start)
        .checked_add(1)
        .and_then(|n| n.checked_mul(width as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid_input("binary range offset size overflows"))?;
    let output_bytes = offset_bytes
        .checked_add(values.len())
        .ok_or_else(|| Error::invalid_input("binary range output size overflows"))?;
    let offset_end = range
        .end
        .checked_add(1)
        .ok_or_else(|| Error::invalid_input("binary range offset count overflows"))?;
    layout(codec)?.validate_range_bytes(range.end - range.start, output_bytes)?;
    Ok(RangePlan {
        budget: MiniBlockDecodeBudget {
            output_bytes,
            workspace_bytes: output_bytes,
        },
        values,
        offsets: range.start..offset_end,
        offset_bytes,
        width,
        metadata,
    })
}

pub(super) fn plan(
    codec: &BinaryMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
) -> Result<RangePlan> {
    let bits = width(codec)?;
    if range.start > range.end || range.end > num_values {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "selected range exceeds value count",
        ));
    }
    if let BinaryMiniBlockLayout::Generic { offsets, .. } = &codec.layout {
        let [values] = data else {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "metadata offsets require one value buffer",
            ));
        };
        let count = num_values.checked_add(1).ok_or_else(|| {
            Error::corrupt_file_named("binary mini-block", "offset count overflows")
        })?;
        let (start, step, last) = match offsets.plan_unsigned_metadata(u64::from(bits), count)? {
            SequenceMetadata::Empty => {
                return Err(Error::corrupt_file_named(
                    "binary mini-block",
                    "metadata offsets must include a leading offset",
                ));
            }
            SequenceMetadata::Constant(value) => (value, 0, value),
            SequenceMetadata::Range { start, step } => (
                start,
                step,
                checked_range_last(u64::from(bits), start, step, count)?,
            ),
        };
        let max = if bits == 32 {
            i32::MAX as u64
        } else {
            i64::MAX as u64
        };
        if start != 0 || last != values.len() as u64 || last > max {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "metadata offsets must start at zero and end at the signed value-buffer length",
            ));
        }
        // The complete sequence is checked; evaluating either selected endpoint is safe.
        let selected = (start + step * range.start) as usize..(start + step * range.end) as usize;
        return finish_plan(codec, range, selected, Some((start, step)));
    }
    let [input] = data else {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "flat-offset chunk requires one buffer",
        ));
    };
    let width = usize::from(bits / 8);
    let count = usize::try_from(num_values)
        .ok()
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| Error::corrupt_file_named("binary mini-block", "offset count overflows"))?;
    let table_bytes = count.checked_mul(width).ok_or_else(|| {
        Error::corrupt_file_named("binary mini-block", "offset table size overflows")
    })?;
    if !input.len().is_multiple_of(width) || input.len() < table_bytes {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "truncated or incorrectly sized offset table",
        ));
    }
    let first = offset(input, 0, width);
    if first < table_bytes as u64 {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "values overlap offset table",
        ));
    }
    let mut previous = first;
    for i in 0..count {
        let next = offset(input, i, width);
        if next < previous || next > input.len() as u64 {
            return Err(Error::corrupt_file_named(
                "binary mini-block",
                "offsets decrease or exceed input bounds",
            ));
        }
        previous = next;
    }
    let offsets = range.start as usize..range.end as usize + 1;
    let values = offset(input, offsets.start, width) as usize
        ..offset(input, offsets.end - 1, width) as usize;
    finish_plan(codec, range, values, None)
}

pub(super) fn decode(
    codec: &BinaryMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let plan = plan(codec, data, num_values, range.clone())?;
    if plan.budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let mut offsets = workspace
        .try_allocate(plan.offset_bytes)?
        .ok_or_else(|| Error::internal("binary offsets exceeded complete admission"))?;
    let mut values = workspace
        .try_allocate(plan.values.len())?
        .ok_or_else(|| Error::internal("binary values exceeded complete admission"))?;
    for (dst, i) in offsets
        .as_mut()
        .chunks_exact_mut(plan.width)
        .zip(plan.offsets)
    {
        let original = match plan.metadata {
            Some((start, step)) => start + step * i,
            None => offset(&data[0], i as usize, plan.width),
        };
        let value = original - plan.values.start as u64;
        dst.copy_from_slice(&value.to_le_bytes()[..plan.width]);
    }
    values.as_mut().copy_from_slice(&data[0][plan.values]);
    Ok(Some(DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::from(Buffer::from(values.freeze())),
        offsets: LanceBuffer::from(Buffer::from(offsets.freeze())),
        bits_per_offset: (plan.width * 8) as u8,
        num_values: range.end - range.start,
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests;
