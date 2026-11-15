// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Borrow already unzipped values without materializing aligned offsets.

use lance_core::{Error, Result};

use crate::data::VariableWidthBlock;

fn offset(bytes: &[u8], index: usize, width: usize) -> u64 {
    let mut word = [0_u8; 8];
    word[..width].copy_from_slice(&bytes[index * width..(index + 1) * width]);
    u64::from_le_bytes(word)
}

pub(crate) fn visit(
    data: &VariableWidthBlock,
    visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
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
    let max = if width == 4 {
        i32::MAX as u64
    } else {
        i64::MAX as u64
    };
    let mut previous = 0;
    for i in 0..count {
        let next = offset(&data.offsets, i, width);
        if (i == 0 && next != 0) || next < previous || next > max || next > data.data.len() as u64 {
            return Err(Error::corrupt_file_named(
                "per-value binary",
                "offsets must start at zero, remain monotonic and fit the signed input bounds",
            ));
        }
        previous = next;
    }
    for i in 0..count - 1 {
        let start = offset(&data.offsets, i, width) as usize;
        let end = offset(&data.offsets, i + 1, width) as usize;
        visitor(&data.data[start..end])?;
    }
    Ok(())
}
