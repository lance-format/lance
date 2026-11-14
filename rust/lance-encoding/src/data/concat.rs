// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan the complete output tree before publishing any shared payload owners.

use std::alloc::Layout;

use arrow_buffer::{Buffer, alloc::ALIGNMENT, bit_util};
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{
    AllNullDataBlock, DataBlock, FixedSizeListBlock, FixedWidthDataBlock, NullableDataBlock,
    VariableWidthBlock,
};
use crate::buffer::LanceBuffer;

fn invalid(detail: &str) -> Error {
    Error::invalid_input(format!("admitted concatenation: {detail}"))
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("output size overflows"))
}

fn mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid("output size overflows"))
}

fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("value count exceeds addressable size"))
}

// Reading bytes avoids allocating an alignment copy for sliced offset buffers.
fn offset(bytes: &[u8], width: usize, index: usize) -> Result<usize> {
    let start = mul(index, width)?;
    let bytes = bytes
        .get(start..add(start, width)?)
        .ok_or_else(|| invalid("offset buffer is too short"))?;
    let value = match width {
        4 => i32::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| invalid("invalid offset width"))?,
        ) as i64,
        8 => i64::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| invalid("invalid offset width"))?,
        ),
        _ => return Err(invalid("offset width must be 32 or 64 bits")),
    };
    usize::try_from(value).map_err(|_| invalid("negative or unaddressable offset"))
}

fn row_count(block: &DataBlock) -> Result<usize> {
    match block {
        DataBlock::Empty() => Ok(0),
        DataBlock::AllNull(block) => size(block.num_values),
        DataBlock::FixedWidth(block) => size(block.num_values),
        DataBlock::VariableWidth(block) => size(block.num_values),
        DataBlock::Nullable(block) => row_count(&block.data),
        DataBlock::FixedSizeList(block) => {
            let dimension = size(block.dimension)?;
            if dimension == 0 {
                return Err(invalid("fixed-size-list dimension must be positive"));
            }
            let count = row_count(&block.child)?;
            if !count.is_multiple_of(dimension) {
                return Err(invalid(
                    "fixed-size-list child count is not divisible by dimension",
                ));
            }
            Ok(count / dimension)
        }
        _ => Err(Error::not_supported(
            "admitted concatenation does not support this child layout",
        )),
    }
}

enum Plan<'a> {
    Empty,
    AllNull(usize),
    Fixed {
        parts: Vec<(&'a [u8], usize)>,
        width: u64,
        rows: usize,
        bytes: usize,
    },
    Variable {
        parts: Vec<(&'a VariableWidthBlock, usize, usize)>,
        width: usize,
        rows: usize,
        bytes: usize,
        offsets_bytes: usize,
    },
    Nullable {
        parts: Vec<(Option<&'a [u8]>, usize)>,
        child: Box<Self>,
        rows: usize,
    },
    List {
        child: Box<Self>,
        dimension: u64,
    },
}

impl<'a> Plan<'a> {
    fn new(blocks: Vec<&'a DataBlock>) -> Result<Self> {
        // Zero-value canonical results carry no physical type. Preserve typed
        // empty blocks, but allow the decoder's canonical empty between chunks.
        let blocks = blocks
            .into_iter()
            .filter(|block| {
                !matches!(
                    block,
                    DataBlock::Empty() | DataBlock::AllNull(AllNullDataBlock { num_values: 0 })
                )
            })
            .collect::<Vec<_>>();
        let Some(first) = blocks.first() else {
            return Ok(Self::Empty);
        };
        if blocks
            .iter()
            .any(|block| matches!(block, DataBlock::Nullable(_)))
        {
            let mut parts = Vec::with_capacity(blocks.len());
            let mut children = Vec::with_capacity(blocks.len());
            let mut rows = 0;
            for block in blocks {
                let (child, nulls) = match block {
                    DataBlock::Nullable(block) => (block.data.as_ref(), Some(block.nulls.as_ref())),
                    block => (block, None),
                };
                let count = row_count(child)?;
                if nulls.is_some_and(|nulls| count.div_ceil(8) > nulls.len()) {
                    return Err(invalid("validity buffer is too short"));
                }
                rows = add(rows, count)?;
                parts.push((nulls, count));
                children.push(child);
            }
            let child = Box::new(Self::new(children)?);
            return Ok(Self::Nullable { parts, child, rows });
        }
        match first {
            DataBlock::AllNull(_) => {
                let rows = blocks.iter().try_fold(0, |rows, block| {
                    let DataBlock::AllNull(block) = block else {
                        return Err(invalid("mismatched block layouts"));
                    };
                    add(rows, size(block.num_values)?)
                })?;
                Ok(Self::AllNull(rows))
            }
            DataBlock::FixedWidth(first) => {
                let width = first.bits_per_value;
                if width == 0 {
                    return Err(invalid("fixed width must be positive"));
                }
                let mut parts = Vec::with_capacity(blocks.len());
                let mut rows = 0;
                for block in blocks {
                    let DataBlock::FixedWidth(block) = block else {
                        return Err(invalid("mismatched block layouts"));
                    };
                    if block.bits_per_value != width {
                        return Err(invalid("mismatched fixed widths"));
                    }
                    let count = size(block.num_values)?;
                    let bits = mul(count, size(width)?)?;
                    if bits.div_ceil(8) > block.data.len() {
                        return Err(invalid("fixed-width buffer is too short"));
                    }
                    rows = add(rows, count)?;
                    parts.push((block.data.as_ref(), bits));
                }
                let bytes = mul(rows, size(width)?)?.div_ceil(8);
                Ok(Self::Fixed {
                    parts,
                    width,
                    rows,
                    bytes,
                })
            }
            DataBlock::VariableWidth(first) => {
                let width = match first.bits_per_offset {
                    32 => 4,
                    64 => 8,
                    _ => return Err(invalid("offset width must be 32 or 64 bits")),
                };
                let mut parts = Vec::with_capacity(blocks.len());
                let mut rows = 0;
                let mut bytes = 0;
                for block in blocks {
                    let DataBlock::VariableWidth(block) = block else {
                        return Err(invalid("mismatched block layouts"));
                    };
                    if usize::from(block.bits_per_offset) != width * 8 {
                        return Err(invalid("mismatched offset widths"));
                    }
                    let count = size(block.num_values)?;
                    let needed = mul(add(count, 1)?, width)?;
                    if needed > block.offsets.len() {
                        return Err(invalid("offset buffer is too short"));
                    }
                    let start = offset(&block.offsets, width, 0)?;
                    let mut end = start;
                    for index in 1..=count {
                        let next = offset(&block.offsets, width, index)?;
                        if next < end {
                            return Err(invalid("offsets are not monotonic"));
                        }
                        end = next;
                    }
                    if end > block.data.len() {
                        return Err(invalid("offset exceeds value buffer"));
                    }
                    rows = add(rows, count)?;
                    bytes = add(bytes, end - start)?;
                    parts.push((block, start, end));
                }
                let limit = if width == 4 {
                    i32::MAX as u64
                } else {
                    i64::MAX as u64
                };
                if bytes as u64 > limit {
                    return Err(invalid(
                        "concatenated offsets exceed signed offset width; split the batch",
                    ));
                }
                let offsets_bytes = mul(add(rows, 1)?, width)?;
                Ok(Self::Variable {
                    parts,
                    width,
                    rows,
                    bytes,
                    offsets_bytes,
                })
            }
            DataBlock::FixedSizeList(first) => {
                let dimension = first.dimension;
                if dimension == 0 {
                    return Err(invalid("fixed-size-list dimension must be positive"));
                }
                let mut children = Vec::with_capacity(blocks.len());
                for block in blocks {
                    let DataBlock::FixedSizeList(block) = block else {
                        return Err(invalid("mismatched block layouts"));
                    };
                    if block.dimension != dimension {
                        return Err(invalid("mismatched fixed-size-list dimensions"));
                    }
                    let count = row_count(&block.child)?;
                    if !(count as u64).is_multiple_of(dimension) {
                        return Err(invalid(
                            "fixed-size-list child count is not divisible by dimension",
                        ));
                    }
                    children.push(block.child.as_ref());
                }
                let child = Box::new(Self::new(children)?);
                Ok(Self::List { child, dimension })
            }
            _ => Err(Error::not_supported(
                "admitted concatenation requires fixed/variable-width, nullable or fixed-size-list blocks",
            )),
        }
    }

    fn workspace_bytes(&self) -> Result<usize> {
        let allocation = |bytes| {
            Layout::from_size_align(bytes, ALIGNMENT)
                .map(|_| bytes)
                .map_err(|_| invalid("payload exceeds addressable allocation size"))
        };
        match self {
            Self::Empty | Self::AllNull(_) => Ok(0),
            Self::Fixed { bytes, .. } => allocation(*bytes),
            Self::Variable {
                bytes,
                offsets_bytes,
                ..
            } => add(allocation(*bytes)?, allocation(*offsets_bytes)?),
            Self::Nullable { child, rows, .. } => {
                add(allocation(rows.div_ceil(8))?, child.workspace_bytes()?)
            }
            Self::List { child, .. } => child.workspace_bytes(),
        }
    }

    fn materialize(self, workspace: &mut Reservation) -> Result<DataBlock> {
        match self {
            Self::Empty => Ok(DataBlock::Empty()),
            Self::AllNull(rows) => Ok(DataBlock::AllNull(AllNullDataBlock {
                num_values: rows as u64,
            })),
            Self::Fixed {
                parts,
                width,
                rows,
                bytes,
            } => {
                let mut output = allocate(workspace, bytes)?;
                concatenate_bits(&parts, output.as_mut());
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: shared(output),
                    bits_per_value: width,
                    num_values: rows as u64,
                    block_info: Default::default(),
                }))
            }
            Self::Variable {
                parts,
                width,
                rows,
                bytes,
                offsets_bytes,
            } => {
                let mut data = allocate(workspace, bytes)?;
                let mut offsets = allocate(workspace, offsets_bytes)?;
                let mut base = 0;
                let mut output_index = 1;
                for (block, start, end) in parts {
                    let len = end - start;
                    data.as_mut()[base..base + len].copy_from_slice(&block.data[start..end]);
                    for index in 1..=block.num_values as usize {
                        let next = base + (offset(&block.offsets, width, index)? - start);
                        let destination =
                            &mut offsets.as_mut()[output_index * width..(output_index + 1) * width];
                        if width == 4 {
                            destination.copy_from_slice(&(next as i32).to_ne_bytes());
                        } else {
                            destination.copy_from_slice(&(next as i64).to_ne_bytes());
                        }
                        output_index += 1;
                    }
                    base += len;
                }
                Ok(DataBlock::VariableWidth(VariableWidthBlock {
                    data: shared(data),
                    offsets: shared(offsets),
                    bits_per_offset: (width * 8) as u8,
                    num_values: rows as u64,
                    block_info: Default::default(),
                }))
            }
            Self::Nullable { parts, child, rows } => {
                let mut nulls = allocate(workspace, rows.div_ceil(8))?;
                let mut cursor = 0;
                for (input, count) in parts {
                    if let Some(input) = input {
                        for index in 0..count {
                            if bit_util::get_bit(input, index) {
                                bit_util::set_bit(nulls.as_mut(), cursor + index);
                            }
                        }
                    } else {
                        for index in 0..count {
                            bit_util::set_bit(nulls.as_mut(), cursor + index);
                        }
                    }
                    cursor += count;
                }
                Ok(DataBlock::Nullable(NullableDataBlock {
                    data: Box::new(child.materialize(workspace)?),
                    nulls: shared(nulls),
                    block_info: Default::default(),
                }))
            }
            Self::List {
                child, dimension, ..
            } => Ok(DataBlock::FixedSizeList(FixedSizeListBlock {
                child: Box::new(child.materialize(workspace)?),
                dimension,
            })),
        }
    }
}

fn allocate(workspace: &mut Reservation, bytes: usize) -> Result<ReservedBuffer> {
    workspace
        .try_allocate(bytes)?
        .ok_or_else(|| Error::internal("concatenation exceeded its complete workspace plan"))
}

fn shared(buffer: ReservedBuffer) -> LanceBuffer {
    LanceBuffer::from(Buffer::from(buffer.freeze()))
}

fn concatenate_bits(parts: &[(&[u8], usize)], output: &mut [u8]) {
    let mut cursor: usize = 0;
    for (input, bits) in parts {
        if cursor.is_multiple_of(8) && bits.is_multiple_of(8) {
            output[cursor / 8..(cursor + bits) / 8].copy_from_slice(&input[..bits / 8]);
        } else {
            // Padding at a chunk boundary is not part of the selected values.
            for index in 0..*bits {
                if bit_util::get_bit(input, index) {
                    bit_util::set_bit(output, cursor + index);
                }
            }
        }
        cursor += bits;
    }
}

impl DataBlock {
    /// Concatenate compatible physical blocks after admitting the entire output tree.
    ///
    /// Supports fixed/variable-width blocks, nullable children and fixed-size lists.
    /// Canonical empty blocks are neutral. Binary offsets are normalized; bitmap
    /// padding is discarded at each input boundary. Other layouts return an error.
    /// Every input is validated before any output allocation. Insufficient credit
    /// returns `None`, leaving inputs and workspace unchanged for a smaller batch.
    ///
    /// Input physical owners must already be charged separately; they remain alive
    /// during output allocation. This accounts payload, including input/output
    /// overlap, but excludes plan metadata and allocator overhead. It does not make
    /// earlier decoding transactional: callers must preflight that stage separately.
    ///
    /// ```
    /// use lance_core::utils::allocation::AllocationBudget;
    /// use lance_encoding::data::{DataBlock, FixedWidthDataBlock};
    /// let inputs = [1_u8, 2].map(|value| DataBlock::FixedWidth(FixedWidthDataBlock {
    ///     data: vec![value].into(), bits_per_value: 8, num_values: 1,
    ///     block_info: Default::default(),
    /// }));
    /// let budget = AllocationBudget::new(2);
    /// let mut workspace = budget.try_reserve(2).unwrap();
    /// let output = DataBlock::try_concat_in(&inputs, &mut workspace).unwrap().unwrap();
    /// assert_eq!(output.num_values(), 2);
    /// assert!(workspace.is_empty());
    /// drop(output);
    /// assert_eq!(budget.reserved(), 0); // The example's input Vecs are not charged.
    /// ```
    pub fn try_concat_in(blocks: &[Self], workspace: &mut Reservation) -> Result<Option<Self>> {
        let plan = Plan::new(blocks.iter().collect())?;
        if plan.workspace_bytes()? > workspace.len() {
            return Ok(None);
        }
        plan.materialize(workspace).map(Some)
    }
}

#[cfg(test)]
mod tests;
