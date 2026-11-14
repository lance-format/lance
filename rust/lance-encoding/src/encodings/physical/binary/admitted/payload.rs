// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

pub(super) fn plan(
    codec: &BinaryMiniBlockDecompressor,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
) -> Result<RangePlan> {
    let [input, values] = data else {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "encoded offsets require two buffers",
        ));
    };
    let BinaryMiniBlockLayout::Generic {
        offsets,
        bits_per_offset,
        ..
    } = &codec.layout
    else {
        return Err(Error::internal("encoded offset layout changed"));
    };
    let count = num_values
        .checked_add(1)
        .ok_or_else(|| Error::corrupt_file_named("binary mini-block", "offset count overflows"))?;
    let max = if *bits_per_offset == 32 {
        i32::MAX as u64
    } else {
        i64::MAX as u64
    };
    let mut position = 0;
    let mut previous = 0;
    let mut selected = 0..0;
    offsets.visit_unsigned(
        Some(input),
        u64::from(*bits_per_offset),
        count,
        &mut |value| {
            if position >= count
                || (position == 0 && value != 0)
                || value < previous
                || value > max
                || value > values.len() as u64
            {
                return Err(Error::corrupt_file_named(
                    "binary mini-block",
                    "invalid encoded offset value or cardinality",
                ));
            }
            if position == range.start {
                selected.start = value as usize;
            }
            if position == range.end {
                selected.end = value as usize;
            }
            previous = value;
            position += 1;
            Ok(())
        },
    )?;
    if position != count || previous != values.len() as u64 {
        return Err(Error::corrupt_file_named(
            "binary mini-block",
            "encoded offsets must end at the value-buffer length",
        ));
    }
    finish_plan(codec, range, selected, None)
}

pub(super) fn write(
    codec: &BinaryMiniBlockDecompressor,
    input: &LanceBuffer,
    num_values: u64,
    plan: &RangePlan,
    output: &mut [u8],
) -> Result<()> {
    let BinaryMiniBlockLayout::Generic {
        offsets,
        bits_per_offset,
        ..
    } = &codec.layout
    else {
        return Err(Error::internal("encoded offset layout changed"));
    };
    let count = num_values
        .checked_add(1)
        .ok_or_else(|| Error::invalid_input("encoded offset count overflows"))?;
    let mut position = 0;
    // Revisit the same immutable input after admission. Only the selected offsets
    // are materialized; Delta keeps a scalar prefix sum, never a decoded array.
    offsets.visit_unsigned(
        Some(input),
        u64::from(*bits_per_offset),
        count,
        &mut |value| {
            if position >= count {
                return Err(Error::internal(
                    "encoded offset visitor changed cardinality",
                ));
            }
            if plan.offsets.contains(&position) {
                let normalized = value
                    .checked_sub(plan.values.start as u64)
                    .ok_or_else(|| Error::internal("encoded offset visitor changed values"))?;
                let start = (position - plan.offsets.start) as usize * plan.width;
                output[start..start + plan.width]
                    .copy_from_slice(&normalized.to_le_bytes()[..plan.width]);
            }
            position += 1;
            Ok(())
        },
    )?;
    if position != count {
        return Err(Error::internal(
            "encoded offset visitor changed cardinality",
        ));
    }
    Ok(())
}
