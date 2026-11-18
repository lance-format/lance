// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Materialize raw flat bytes without transient dense definition levels.

use arrow_array::{ArrayRef, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, alloc::ALIGNMENT};
use arrow_data::ArrayData;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{super::unzip_admitted::visit, FullZipArrayInput};
use crate::compression::MiniBlockOutputLayout;

struct Plan {
    rows: usize,
    offsets: usize,
    values: usize,
    nulls: usize,
    validity: usize,
    total: usize,
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::invalid_input("flat FullZip output size overflows"))
}

pub(super) fn decode(
    input: &FullZipArrayInput,
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let mut rows = 0;
    let mut values = 0;
    let mut nulls = 0;
    // Reuse the complete framing/schema checks, even when no output credit is
    // available. Shape validation at entry guarantees one leaf and no lists.
    visit(
        &input.scheduler,
        std::slice::from_ref(&input.input),
        input.count,
        0..input.count,
        |_, def, visible, value| {
            if !visible {
                return Err(Error::corrupt_file_named(
                    "FullZip",
                    "flat byte array contains an invisible item",
                ));
            }
            rows = add(rows, 1)?;
            values = add(values, value.len())?;
            nulls = add(nulls, usize::from(def != 0))?;
            Ok(())
        },
    )?;
    let width = usize::from(input.scheduler.bits_per_offset / 8);
    let offsets = add(rows, 1)?
        .checked_mul(width)
        .ok_or_else(|| Error::invalid_input("flat FullZip offsets size overflows"))?;
    let payload = add(offsets, values)?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: input.scheduler.bits_per_offset,
    }
    .validate_range_bytes(input.count, payload)?;
    let validity = if nulls == 0 { 0 } else { rows.div_ceil(8) };
    let total = add(payload, validity)?;
    std::alloc::Layout::from_size_align(total, ALIGNMENT)
        .map_err(|_| Error::invalid_input("flat FullZip output exceeds addressable size"))?;
    let plan = Plan {
        rows,
        offsets,
        values,
        nulls,
        validity,
        total,
    };
    match (width, nulls != 0) {
        (4, false) => materialize::<4, false>(input, plan, workspace),
        (4, true) => materialize::<4, true>(input, plan, workspace),
        (8, false) => materialize::<8, false>(input, plan, workspace),
        (8, true) => materialize::<8, true>(input, plan, workspace),
        _ => Err(Error::internal(
            "validated flat FullZip offset width changed",
        )),
    }
}

fn materialize<const WIDTH: usize, const HAS_NULLS: bool>(
    input: &FullZipArrayInput,
    plan: Plan,
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let Some(mut output) = workspace.try_allocate_writer(plan.total)? else {
        return Ok(None);
    };
    // Keep offsets and validity in the initialized prefix; values are appended
    // exactly once. Byte values need no padding or offset alignment.
    output.extend_zeros(plan.offsets + plan.validity)?;
    let mut row = 0;
    let mut written = 0;
    let mut nulls = 0;
    visit(
        &input.scheduler,
        std::slice::from_ref(&input.input),
        input.count,
        0..input.count,
        |_, def, _, value| {
            output.extend_from_slice(value)?;
            written += value.len();
            if HAS_NULLS {
                if def == 0 {
                    output.initialized_mut()[plan.offsets + row / 8] |= 1 << (row % 8);
                } else {
                    nulls += 1;
                }
            }
            row += 1;
            let offset = &mut output.initialized_mut()[row * WIDTH..(row + 1) * WIDTH];
            // WIDTH is monomorphized; preflight checked signed representability.
            if WIDTH == 4 {
                offset.copy_from_slice(&(written as i32).to_ne_bytes());
            } else {
                offset.copy_from_slice(&(written as i64).to_ne_bytes());
            }
            Ok(())
        },
    )?;
    if row != plan.rows || written != plan.values || nulls != plan.nulls {
        return Err(Error::internal(
            "flat FullZip replay disagrees with preflight",
        ));
    }
    let storage = Buffer::from(output.finish()?);
    let nulls = HAS_NULLS.then(|| {
        NullBuffer::new(BooleanBuffer::new(
            storage.slice_with_length(plan.offsets, plan.validity),
            0,
            plan.rows,
        ))
    });
    // Arrow validates offsets and UTF-8 without allocating a second payload.
    // All three slices retain the single complete charged output owner.
    Ok(Some(make_array(
        ArrayData::builder(input.data_type.clone())
            .len(plan.rows)
            .add_buffer(storage.slice_with_length(0, plan.offsets))
            .add_buffer(storage.slice_with_length(plan.offsets + plan.validity, plan.values))
            .nulls(nulls)
            .build()?,
    )))
}
