// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::{ArrowNativeType, ScalarBuffer};
use lance_core::{Error, Result};

use crate::data::FixedWidthDataBlock;

pub mod binary;
#[cfg(feature = "bitpacking")]
pub mod bitpacking;
pub mod block;
pub mod byte_stream_split;
pub mod constant;
pub mod delta;
pub mod dictionary;
pub mod fsst;
pub mod general;
pub mod packed;
pub mod range;
pub mod rle;
pub mod value;

pub(crate) fn try_vec_with_capacity<T>(num_values: u64, label: &str) -> Result<Vec<T>> {
    let capacity = usize::try_from(num_values)
        .map_err(|_| Error::invalid_input(format!("{label} cardinality does not fit usize")))?;
    let output_bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::invalid_input(format!("{label} byte length overflows usize")))?;
    if output_bytes > isize::MAX as usize {
        return Err(Error::invalid_input(format!(
            "{label} byte length {output_bytes} exceeds isize::MAX"
        )));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        Error::invalid_input(format!(
            "{label} could not reserve {capacity} values ({output_bytes} bytes): {error}"
        ))
    })?;
    Ok(values)
}

pub(crate) fn checked_fixed_values<T: ArrowNativeType>(
    data: &FixedWidthDataBlock,
    label: &str,
) -> Result<ScalarBuffer<T>> {
    let expected = usize::try_from(data.num_values)
        .ok()
        .and_then(|len| len.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| Error::invalid_input(format!("{label} byte length overflows usize")))?;
    if expected > isize::MAX as usize {
        return Err(Error::invalid_input(format!(
            "{label} byte length {expected} exceeds isize::MAX"
        )));
    }
    if data.data.len() != expected {
        return Err(Error::invalid_input(format!(
            "{label} has {} bytes, expected {expected}",
            data.data.len()
        )));
    }
    Ok(data.data.borrow_to_typed_slice::<T>())
}
