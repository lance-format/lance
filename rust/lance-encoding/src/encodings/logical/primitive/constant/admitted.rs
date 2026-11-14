// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Complete selected-level and expanded-value admission for constant page tasks.

use std::ops::Range;

use arrow_array::{Array, cast::AsArray};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::DataType;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::DecodeConstantTask;
use crate::{
    buffer::LanceBuffer,
    data::{AllNullDataBlock, BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    decoder::DecodedPage,
    repdef::{RepDefUnraveler, max_visible_level},
};

fn allocate(workspace: &mut Reservation, bytes: usize) -> Result<ReservedBuffer> {
    workspace
        .try_allocate(bytes)?
        .ok_or_else(|| Error::internal("constant output exceeded its complete admission"))
}

enum Scalar<'a> {
    Null,
    Fixed(Buffer),
    Boolean(bool),
    Binary {
        value: &'a [u8],
        offset_width: usize,
    },
}

struct Expansion<'a> {
    scalar: Scalar<'a>,
    count: usize,
    data_bytes: usize,
    offset_bytes: usize,
    is_all_valid: bool,
}

impl<'a> Expansion<'a> {
    fn plan(scalar: &'a dyn Array, count: usize, valid_count: usize) -> Result<Option<Self>> {
        if scalar.len() != 1 {
            return Err(Error::invalid_input(
                "constant page needs exactly one scalar",
            ));
        }
        let data_type = scalar.data_type();
        let width = match data_type {
            DataType::FixedSizeBinary(width) => Some(
                usize::try_from(*width)
                    .map_err(|_| Error::invalid_input("negative constant byte width"))?,
            ),
            _ => data_type.primitive_width(),
        };
        let scalar = if scalar.is_null(0) {
            Scalar::Null
        } else if let Some(width) = width {
            let source = scalar.to_data();
            let start = source
                .offset()
                .checked_mul(width)
                .ok_or_else(|| Error::invalid_input("constant scalar offset overflows"))?;
            let end = start
                .checked_add(width)
                .ok_or_else(|| Error::invalid_input("constant scalar width overflows"))?;
            let buffer = source
                .buffers()
                .first()
                .filter(|b| end <= b.len())
                .ok_or_else(|| {
                    Error::invalid_input("constant scalar buffer is shorter than its typed layout")
                })?;
            Scalar::Fixed(buffer.slice_with_length(start, width))
        } else {
            match data_type {
                DataType::Boolean => Scalar::Boolean(scalar.as_boolean().value(0)),
                DataType::Binary => Scalar::Binary {
                    value: scalar.as_binary::<i32>().value(0),
                    offset_width: 4,
                },
                DataType::LargeBinary => Scalar::Binary {
                    value: scalar.as_binary::<i64>().value(0),
                    offset_width: 8,
                },
                DataType::Utf8 => Scalar::Binary {
                    value: scalar.as_string::<i32>().value(0).as_bytes(),
                    offset_width: 4,
                },
                DataType::LargeUtf8 => Scalar::Binary {
                    value: scalar.as_string::<i64>().value(0).as_bytes(),
                    offset_width: 8,
                },
                other => {
                    return Err(Error::not_supported(format!(
                        "constant page has no admitted expansion for {other}"
                    )));
                }
            }
        };
        let (data_bytes, offset_bytes) = match &scalar {
            Scalar::Null => (0, 0),
            Scalar::Fixed(value) => {
                let Some(bytes) = value.len().checked_mul(count) else {
                    return Ok(None);
                };
                (bytes, 0)
            }
            Scalar::Boolean(_) => (count.div_ceil(8), 0),
            Scalar::Binary {
                value,
                offset_width,
            } => {
                // Null slots repeat the offset, not the potentially huge scalar.
                let Some(bytes) = value.len().checked_mul(valid_count) else {
                    return Ok(None);
                };
                let limit = if *offset_width == 8 {
                    i64::MAX as u64
                } else {
                    i32::MAX as u64
                };
                if bytes as u64 > limit {
                    return Ok(None);
                }
                let Some(offsets) = count
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(*offset_width))
                else {
                    return Ok(None);
                };
                (bytes, offsets)
            }
        };
        Ok(Some(Self {
            scalar,
            count,
            data_bytes,
            offset_bytes,
            is_all_valid: valid_count == count,
        }))
    }

    fn materialize(
        self,
        visible: impl Iterator<Item = bool>,
        workspace: &mut Reservation,
    ) -> Result<DataBlock> {
        let Self {
            scalar,
            count,
            data_bytes,
            offset_bytes,
            is_all_valid,
        } = self;
        match scalar {
            Scalar::Null => Ok(DataBlock::AllNull(AllNullDataBlock {
                num_values: count as u64,
            })),
            Scalar::Fixed(value) => {
                let mut data = allocate(workspace, data_bytes)?;
                if is_all_valid && !value.is_empty() {
                    for slot in data.as_mut().chunks_exact_mut(value.len()) {
                        slot.copy_from_slice(&value);
                    }
                } else if !value.is_empty() {
                    for (slot, valid) in data.as_mut().chunks_exact_mut(value.len()).zip(visible) {
                        if valid {
                            slot.copy_from_slice(&value);
                        }
                    }
                }
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::from(Buffer::from(data.freeze())),
                    num_values: count as u64,
                    bits_per_value: value.len() as u64 * 8,
                    block_info: BlockInfo::new(),
                }))
            }
            Scalar::Boolean(value) => {
                let mut data = allocate(workspace, data_bytes)?;
                if value && is_all_valid {
                    data.as_mut().fill(255);
                    if !count.is_multiple_of(8) {
                        data.as_mut()[data_bytes - 1] = (1 << (count % 8)) - 1;
                    }
                } else if value {
                    for (i, valid) in visible.enumerate() {
                        if valid {
                            data.as_mut()[i / 8] |= 1 << (i % 8);
                        }
                    }
                }
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::from(Buffer::from(data.freeze())),
                    num_values: count as u64,
                    bits_per_value: 1,
                    block_info: BlockInfo::new(),
                }))
            }
            Scalar::Binary {
                value,
                offset_width,
            } => {
                let mut data = allocate(workspace, data_bytes)?;
                let mut offsets = allocate(workspace, offset_bytes)?;
                if !value.is_empty() {
                    for slot in data.as_mut().chunks_exact_mut(value.len()) {
                        slot.copy_from_slice(value);
                    }
                }
                let mut position = 0;
                if offset_width == 8 {
                    for (slot, valid) in offsets
                        .as_mut()
                        .as_chunks_mut::<8>()
                        .0
                        .iter_mut()
                        .skip(1)
                        .zip(visible)
                    {
                        if valid {
                            position += value.len();
                        }
                        slot.copy_from_slice(&(position as i64).to_le_bytes());
                    }
                } else {
                    for (slot, valid) in offsets
                        .as_mut()
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .skip(1)
                        .zip(visible)
                    {
                        if valid {
                            position += value.len();
                        }
                        slot.copy_from_slice(&(position as i32).to_le_bytes());
                    }
                }
                Ok(DataBlock::VariableWidth(VariableWidthBlock {
                    data: LanceBuffer::from(Buffer::from(data.freeze())),
                    offsets: LanceBuffer::from(Buffer::from(offsets.freeze())),
                    bits_per_offset: (offset_width * 8) as u8,
                    num_values: count as u64,
                    block_info: BlockInfo::new(),
                }))
            }
        }
    }
}

fn selected_levels<'a>(
    source: &'a Option<ScalarBuffer<u16>>,
    range: &Range<usize>,
) -> Result<Option<&'a [u16]>> {
    source
        .as_ref()
        .map(|v| {
            v.get(range.clone())
                .ok_or_else(|| Error::invalid_input("constant level selection exceeds its input"))
        })
        .transpose()
}

fn validate_selection(task: &DecodeConstantTask) -> Result<usize> {
    RepDefUnraveler::validate_dense_levels(None, None, &task.def_meaning)?;
    if task.max_visible_def != max_visible_level(&task.def_meaning) {
        return Err(Error::invalid_input(
            "constant visible level disagrees with its schema",
        ));
    }
    let mut total = 0_usize;
    for range in &task.level_slices {
        if range.start > range.end {
            return Err(Error::invalid_input("constant level selection is reversed"));
        }
        RepDefUnraveler::validate_dense_levels(
            selected_levels(&task.rep, range)?,
            selected_levels(&task.def, range)?,
            &task.def_meaning,
        )?;
        total = total
            .checked_add(range.len())
            .ok_or_else(|| Error::invalid_input("constant level selection length overflows"))?;
    }
    if task.def.is_none() && total as u64 != task.visible_items_total {
        return Err(Error::invalid_input(
            "constant selected level and item counts differ",
        ));
    }
    Ok(total)
}

fn visible_values(task: &DecodeConstantTask) -> impl Iterator<Item = bool> + '_ {
    task.level_slices
        .iter()
        .flat_map(|range| range.clone())
        .filter_map(move |i| {
            task.def.as_ref().map_or(Some(true), |def| {
                (def[i] <= task.max_visible_def).then_some(def[i] == 0)
            })
        })
}

fn copy_levels(
    task: &DecodeConstantTask,
    source: &Option<ScalarBuffer<u16>>,
    bytes: usize,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    source
        .as_ref()
        .map(|source| {
            let mut output = allocate(workspace, bytes)?;
            let mut position = 0;
            for range in &task.level_slices {
                let values = bytemuck::cast_slice(&source[range.clone()]);
                output.as_mut()[position..position + values.len()].copy_from_slice(values);
                position += values.len();
            }
            Ok(output)
        })
        .transpose()
}

pub(super) fn decode(
    task: &DecodeConstantTask,
    workspace: &mut Reservation,
) -> Result<Option<DecodedPage>> {
    let level_count = validate_selection(task)?;
    let (count, valid_count) = if task.def.is_some() {
        (
            visible_values(task).count(),
            visible_values(task).filter(|valid| *valid).count(),
        )
    } else {
        (level_count, level_count)
    };
    let Some(expansion) = Expansion::plan(task.scalar.as_ref(), count, valid_count)? else {
        return Ok(None);
    };
    let Some(level_bytes) = level_count.checked_mul(2) else {
        return Ok(None);
    };
    let streams = usize::from(task.rep.is_some()) + usize::from(task.def.is_some());
    let Some(required) = level_bytes
        .checked_mul(streams)
        .and_then(|n| n.checked_add(expansion.data_bytes))
        .and_then(|n| n.checked_add(expansion.offset_bytes))
    else {
        return Ok(None);
    };
    if required > workspace.len() {
        return Ok(None);
    }
    // One preflight includes selected levels and expanded values. There is no
    // take-index Vec, concat scratch, or second level copy in this path.
    let rep = copy_levels(task, &task.rep, level_bytes, workspace)?;
    let def = copy_levels(task, &task.def, level_bytes, workspace)?;
    let data = expansion.materialize(visible_values(task), workspace)?;
    let repdef =
        RepDefUnraveler::from_admitted_levels(rep, def, task.def_meaning.clone(), count as u64)?;
    Ok(Some(DecodedPage { data, repdef }))
}

#[cfg(test)]
mod tests;
