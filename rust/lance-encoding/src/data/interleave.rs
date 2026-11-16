// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Byte-array assembly with one admitted physical output owner.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, GenericByteArray, OffsetSizeTrait,
    cast::AsArray,
    types::{ByteArrayType, GenericBinaryType, GenericStringType},
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer, bit_util};
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

#[cfg(test)]
mod tests;

/// Interleave Binary, LargeBinary, Utf8 or LargeUtf8 into an admitted allocation.
///
/// Sources must be nonempty and have identical supported types. Each index is
/// `(source, row)`; order and duplicates are preserved. Null values do not copy
/// their inaccessible payload. Invalid types/indices are errors even when the
/// workspace is exhausted. `Ok(None)` means insufficient workspace, not EOF or
/// permission to drop rows. It leaves the workspace unchanged and allocates no
/// output payload. This function never waits or grows the workspace.
///
/// Offsets, values and optional validity share one exact physical allocation.
/// Every output buffer alias retains its full charge until the last alias drops.
/// Reserve input/output overlap before dependent reads: inputs and routing
/// metadata are borrowed, and this function does not admit them retroactively.
/// Array/Arc headers and allocator overhead are outside this payload budget.
pub fn try_interleave_binary_in(
    arrays: &[&dyn Array],
    indices: &[(usize, usize)],
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let first = arrays
        .first()
        .ok_or_else(|| invalid("sources must not be empty"))?;
    match first.data_type() {
        DataType::Binary => interleave::<GenericBinaryType<i32>>(arrays, indices, workspace),
        DataType::LargeBinary => interleave::<GenericBinaryType<i64>>(arrays, indices, workspace),
        DataType::Utf8 => interleave::<GenericStringType<i32>>(arrays, indices, workspace),
        DataType::LargeUtf8 => interleave::<GenericStringType<i64>>(arrays, indices, workspace),
        data_type => Err(Error::not_supported(format!(
            "admitted byte interleave does not support {data_type}"
        ))),
    }
}

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("admitted byte interleave: {detail}"))
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("output size overflows"))
}

fn source<T: ByteArrayType>(array: &dyn Array) -> Result<&GenericByteArray<T>> {
    array.as_bytes_opt::<T>().ok_or_else(|| {
        invalid(format!(
            "expected source type {}, got {}",
            T::DATA_TYPE,
            array.data_type()
        ))
    })
}

fn interleave<T: ByteArrayType>(
    arrays: &[&dyn Array],
    indices: &[(usize, usize)],
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    // Validate even unselected sources without allocating a typed reference list.
    for array in arrays {
        source::<T>(*array)?;
    }
    let mut value_bytes = 0;
    let mut has_nulls = false;
    for &(array_index, row) in indices {
        let array = arrays.get(array_index).ok_or_else(|| {
            invalid(format!(
                "source {array_index} exceeds {} sources",
                arrays.len()
            ))
        })?;
        let array = source::<T>(*array)?;
        if row >= array.len() {
            return Err(invalid(format!(
                "row {row} exceeds source {array_index} length {}",
                array.len()
            )));
        }
        if array.is_null(row) {
            has_nulls = true;
        } else {
            let value: &[u8] = array.value(row).as_ref();
            value_bytes = add(value_bytes, value.len())?;
        }
    }
    if value_bytes > T::Offset::MAX_OFFSET {
        return Err(invalid(format!(
            "{value_bytes} value bytes exceed {} offset capacity",
            T::DATA_TYPE
        )));
    }
    let offset_count = add(indices.len(), 1)?;
    let offset_width = std::mem::size_of::<T::Offset>();
    let offset_bytes = offset_count
        .checked_mul(offset_width)
        .ok_or_else(|| invalid("offset size overflows"))?;
    let validity_start = add(offset_bytes, value_bytes)?;
    let validity_bytes = if has_nulls {
        indices.len().div_ceil(8)
    } else {
        0
    };
    let total_bytes = add(validity_start, validity_bytes)?;
    let Some(mut output) = workspace.try_allocate(total_bytes)? else {
        return Ok(None);
    };

    let bytes = output.as_mut();
    let mut value_end = 0;
    for (index, &(array_index, row)) in indices.iter().enumerate() {
        let array = source::<T>(arrays[array_index])?;
        if array.is_valid(row) {
            let value: &[u8] = array.value(row).as_ref();
            // The first pass checked the sum and every selected source/row.
            let next_end = value_end + value.len();
            bytes[offset_bytes + value_end..offset_bytes + next_end].copy_from_slice(value);
            value_end = next_end;
            if has_nulls {
                bit_util::set_bit(&mut bytes[validity_start..], index);
            }
        }
        let offset = &mut bytes[(index + 1) * offset_width..(index + 2) * offset_width];
        // ByteArrayType is sealed to i32/i64 offsets. The first pass verified
        // representability; byte writes avoid an unaligned mutable cast.
        if T::Offset::IS_LARGE {
            offset.copy_from_slice(&(value_end as i64).to_ne_bytes());
        } else {
            offset.copy_from_slice(&(value_end as i32).to_ne_bytes());
        }
    }

    let output = Buffer::from(output.freeze());
    // The reserved base is Arrow-aligned, offsets start at zero, and the fully
    // initialized monotonic offsets fit the checked layout and value payload.
    let offsets = OffsetBuffer::new(ScalarBuffer::<T::Offset>::new(
        output.slice_with_length(0, offset_bytes),
        0,
        offset_count,
    ));
    let values = output.slice_with_length(offset_bytes, value_bytes);
    let nulls = has_nulls.then(|| {
        NullBuffer::new(BooleanBuffer::new(
            output.slice_with_length(validity_start, validity_bytes),
            0,
            indices.len(),
        ))
    });
    Ok(Some(Arc::new(GenericByteArray::<T>::try_new(
        offsets, values, nulls,
    )?)))
}
