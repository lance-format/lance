// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rebuild a struct spine without unadmitted validity unions.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray, cast::AsArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, bit_util};
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

mod validity;
pub use validity::{StructValiditySelection, try_replace_struct_validities_in};

/// Replace one child reached by a nonempty path of struct child indices.
///
/// The replacement must have the target child's type and row count. Unchanged
/// children and unchanged validity buffers are shared, including sliced offsets.
/// A required child's new nulls extend its parent mask, recursively through any
/// required struct ancestors; nullable children do not change parent validity.
/// This is structural masking, not resolution of overlay version precedence.
///
/// The full path is validated and its new bitmaps are sized before allocating.
/// Insufficient workspace returns `Ok(None)` without changing the reservation or
/// input arrays. No stage waits for additional credit while retaining inputs.
/// Each new bitmap owns its exact charge through the last Arrow alias. Sources,
/// replacement and reused masks must already be accounted by the caller; this
/// function does not admit their storage retroactively. Planning/Array/Arc metadata
/// and allocator overhead are outside this payload budget.
///
/// Required children of type Null, Dictionary, Union or RunEndEncoded are rejected:
/// Arrow's constructor may materialize their logical nulls outside the workspace.
/// Other unsupported paths (for example descending through a list) are errors;
/// there is no fallback to an ordinary uncharged splice.
///
/// ```
/// use arrow_array::{ArrayRef, StructArray};
/// use lance_core::{Result, utils::allocation::Reservation};
/// use lance_encoding::data::try_splice_struct_in;
/// fn replace_first(base: &StructArray, replacement: ArrayRef,
///                  workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
///     try_splice_struct_in(base, &[0], replacement, workspace)
/// }
/// ```
pub fn try_splice_struct_in(
    source: &StructArray,
    path: &[usize],
    replacement: ArrayRef,
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let plan = Plan::new(source, path, replacement.as_ref())?;
    if plan.total_bytes > workspace.len() {
        return Ok(None);
    }
    plan.build(replacement, workspace).map(Some)
}

struct Plan<'a> {
    source: &'a StructArray,
    index: usize,
    child: Option<Box<Self>>,
    validity_bytes: usize,
    total_bytes: usize,
}

impl<'a> Plan<'a> {
    fn new(source: &'a StructArray, path: &[usize], replacement: &dyn Array) -> Result<Self> {
        let Some((&index, rest)) = path.split_first() else {
            return Err(Error::invalid_input(
                "struct splice requires a nonempty child path",
            ));
        };
        let field = source.fields().get(index).ok_or_else(|| {
            Error::invalid_input(format!(
                "struct splice child {index} exceeds {} fields",
                source.num_columns()
            ))
        })?;
        check_required_fields(source)?;
        let child = if rest.is_empty() {
            if replacement.data_type() != field.data_type() || replacement.len() != source.len() {
                return Err(Error::invalid_input(format!(
                    "struct splice replacement has type {} and {} rows; expected {} and {} rows",
                    replacement.data_type(),
                    replacement.len(),
                    field.data_type(),
                    source.len()
                )));
            }
            None
        } else {
            let child = source.column(index).as_struct_opt().ok_or_else(|| {
                Error::invalid_input(format!("struct splice child {index} is not a Struct"))
            })?;
            Some(Box::new(Self::new(child, rest, replacement)?))
        };
        let mut plan = Self {
            source,
            index,
            child,
            validity_bytes: 0,
            total_bytes: 0,
        };
        // Existing required siblings are already masked by the source validity.
        // Only the replaced path can introduce newly unmasked child nulls.
        if (0..source.len()).any(|row| source.is_valid(row) && !plan.is_valid(row, replacement)) {
            plan.validity_bytes = source.len().div_ceil(8);
        }
        plan.total_bytes = plan
            .validity_bytes
            .checked_add(plan.child.as_ref().map_or(0, |child| child.total_bytes))
            .ok_or_else(|| Error::invalid_input("struct splice validity size overflows"))?;
        Ok(plan)
    }

    fn is_valid(&self, row: usize, replacement: &dyn Array) -> bool {
        self.source.is_valid(row)
            && (self.source.fields()[self.index].is_nullable()
                || self.child.as_ref().map_or_else(
                    || replacement.is_valid(row),
                    |child| child.is_valid(row, replacement),
                ))
    }

    fn build(&self, replacement: ArrayRef, workspace: &mut Reservation) -> Result<ArrayRef> {
        // Produce masks before moving replacement into a deeper rebuilt child.
        // The complete plan includes all these simultaneously retained bitmaps.
        let nulls = if self.validity_bytes == 0 {
            self.source.nulls().cloned()
        } else {
            let mut output = workspace
                .try_allocate(self.validity_bytes)?
                .ok_or_else(|| {
                    Error::internal("struct splice exceeded its complete validity plan")
                })?;
            for row in 0..self.source.len() {
                if self.is_valid(row, replacement.as_ref()) {
                    bit_util::set_bit(output.as_mut(), row);
                }
            }
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from(output.freeze()),
                0,
                self.source.len(),
            )))
        };
        let child = match &self.child {
            Some(child) => child.build(replacement, workspace)?,
            None => replacement,
        };
        let mut columns = self.source.columns().to_vec();
        columns[self.index] = child;
        Ok(Arc::new(StructArray::try_new_with_length(
            self.source.fields().clone(),
            columns,
            nulls,
            self.source.len(),
        )?))
    }
}

fn check_required_fields(source: &StructArray) -> Result<()> {
    for field in source.fields() {
        if !field.is_nullable()
            && matches!(
                field.data_type(),
                DataType::Null
                    | DataType::Dictionary(_, _)
                    | DataType::Union(_, _)
                    | DataType::RunEndEncoded(_, _)
            )
        {
            return Err(Error::not_supported(format!(
                "admitted struct splice cannot materialize logical nulls for required field '{}' of type {}",
                field.name(),
                field.data_type()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
