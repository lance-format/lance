// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Byte-array assembly with one admitted physical output owner.

use std::alloc::Layout;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, GenericByteArray, OffsetSizeTrait,
    cast::AsArray,
    types::{ByteArrayType, GenericBinaryType, GenericStringType},
};
use arrow_buffer::{
    BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer, alloc::ALIGNMENT, bit_util,
};
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

mod nested;
#[cfg(test)]
mod tests;

pub use nested::try_interleave_nested_binary_in;

// Recreate a selection for each planning/materialization pass. Nested lists
// expand child ranges lazily instead of allocating one routing entry per child.
trait Selection {
    fn iter(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_>;
}

struct RootSelection<'a>(&'a [(usize, usize)]);

impl Selection for RootSelection<'_> {
    fn iter(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        Box::new(self.0.iter().copied())
    }
}

struct BytePlan {
    rows: usize,
    offset_width: usize,
    offset_bytes: usize,
    value_bytes: usize,
    validity_bytes: usize,
    total_bytes: usize,
}

impl BytePlan {
    fn new<O: OffsetSizeTrait>(rows: usize, value_bytes: usize, has_nulls: bool) -> Result<Self> {
        let offset_count = add(rows, 1)?;
        let offset_width = std::mem::size_of::<O>();
        let offset_bytes = offset_count
            .checked_mul(offset_width)
            .ok_or_else(|| invalid("offset size overflows"))?;
        let validity_bytes = if has_nulls { rows.div_ceil(8) } else { 0 };
        let total_bytes = add(add(offset_bytes, value_bytes)?, validity_bytes)?;
        Layout::from_size_align(total_bytes, ALIGNMENT)
            .map_err(|_| invalid("output exceeds addressable allocation size"))?;
        Ok(Self {
            rows,
            offset_width,
            offset_bytes,
            value_bytes,
            validity_bytes,
            total_bytes,
        })
    }

    fn finish<O: OffsetSizeTrait>(
        &self,
        output: Buffer,
    ) -> (OffsetBuffer<O>, Buffer, Option<NullBuffer>) {
        // The reserved base is Arrow-aligned, and the plan checked the layout.
        // Both materializers write initialized, monotonic, representable offsets.
        let offsets = OffsetBuffer::new(ScalarBuffer::<O>::new(
            output.slice_with_length(0, self.offset_bytes),
            0,
            self.rows + 1,
        ));
        let values = output.slice_with_length(self.offset_bytes, self.value_bytes);
        let nulls = (self.validity_bytes != 0).then(|| {
            NullBuffer::new(BooleanBuffer::new(
                output.slice_with_length(self.offset_bytes + self.value_bytes, self.validity_bytes),
                0,
                self.rows,
            ))
        });
        (offsets, values, nulls)
    }

    fn write_offset(&self, bytes: &mut [u8], index: usize, value: usize) {
        let offset = &mut bytes[(index + 1) * self.offset_width..(index + 2) * self.offset_width];
        // Arrow's sealed offsets are i32/i64; planning checked representability.
        if self.offset_width == 8 {
            offset.copy_from_slice(&(value as i64).to_ne_bytes());
        } else {
            offset.copy_from_slice(&(value as i32).to_ne_bytes());
        }
    }
}

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
/// Planning/iterator metadata, Array/Arc headers and allocator overhead are
/// outside this payload budget.
pub fn try_interleave_binary_in(
    arrays: &[&dyn Array],
    indices: &[(usize, usize)],
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let first = arrays
        .first()
        .ok_or_else(|| invalid("sources must not be empty"))?;
    let selection = RootSelection(indices);
    match first.data_type() {
        DataType::Binary => interleave::<GenericBinaryType<i32>>(arrays, &selection, workspace),
        DataType::LargeBinary => {
            interleave::<GenericBinaryType<i64>>(arrays, &selection, workspace)
        }
        DataType::Utf8 => interleave::<GenericStringType<i32>>(arrays, &selection, workspace),
        DataType::LargeUtf8 => interleave::<GenericStringType<i64>>(arrays, &selection, workspace),
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
    selection: &dyn Selection,
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let plan = plan_bytes::<T>(arrays, selection)?;
    if plan.total_bytes > workspace.len() {
        return Ok(None);
    }
    build_bytes::<T>(arrays, selection, &plan, workspace).map(Some)
}

fn plan_bytes<T: ByteArrayType>(
    arrays: &[&dyn Array],
    selection: &dyn Selection,
) -> Result<BytePlan> {
    // Validate even unselected sources without allocating a typed reference list.
    for array in arrays {
        source::<T>(*array)?;
    }
    let mut value_bytes = 0;
    let mut has_nulls = false;
    let mut rows = 0;
    for (array_index, row) in selection.iter() {
        rows = add(rows, 1)?;
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
    BytePlan::new::<T::Offset>(rows, value_bytes, has_nulls)
}

fn build_bytes<T: ByteArrayType>(
    arrays: &[&dyn Array],
    selection: &dyn Selection,
    plan: &BytePlan,
    workspace: &mut Reservation,
) -> Result<ArrayRef> {
    let mut output = workspace
        .try_allocate(plan.total_bytes)?
        .ok_or_else(|| Error::internal("admitted interleave exceeded its complete output plan"))?;

    let bytes = output.as_mut();
    let mut value_end = 0;
    for (index, (array_index, row)) in selection.iter().enumerate() {
        let array = source::<T>(arrays[array_index])?;
        if array.is_valid(row) {
            let value: &[u8] = array.value(row).as_ref();
            // The first pass checked the sum and every selected source/row.
            let next_end = value_end + value.len();
            bytes[plan.offset_bytes + value_end..plan.offset_bytes + next_end]
                .copy_from_slice(value);
            value_end = next_end;
            if plan.validity_bytes != 0 {
                bit_util::set_bit(&mut bytes[plan.offset_bytes + plan.value_bytes..], index);
            }
        }
        plan.write_offset(bytes, index, value_end);
    }

    let (offsets, values, nulls) = plan.finish::<T::Offset>(Buffer::from(output.freeze()));
    Ok(Arc::new(GenericByteArray::<T>::try_new(
        offsets, values, nulls,
    )?))
}
