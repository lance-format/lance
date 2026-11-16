// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// Borrowed validity routing for one struct in a reconstructed output tree.
///
/// `path` uses struct child indices; an empty path selects the root. `indices`
/// contains one `(source, row)` pair per output row. Only the selected struct's
/// validity is read: source children and field types need not match the target.
/// Callers resolve overlay precedence before supplying this routing. Sources and
/// routing storage must already be accounted by the caller.
///
/// See [`try_replace_struct_validities_in`] for a complete example.
pub struct StructValiditySelection<'a> {
    /// Unique path from the root to a struct whose validity is replaced.
    pub path: &'a [usize],
    /// Struct arrays supplying already-accounted validity buffers.
    pub sources: &'a [&'a StructArray],
    /// Selected source and row for each output row, in output order.
    pub indices: &'a [(usize, usize)],
}

/// Replace several struct validities under one payload reservation.
///
/// Selected masks are authoritative, including null-to-valid transitions. The
/// complete resulting tree must mask every required child's nulls; inconsistent
/// selections are errors before any allocation, even with no workspace credit.
/// Unselected ancestors preserve their old validity and mask new required-child
/// nulls. Updates are simultaneous and independent of their input order, avoiding
/// intermediate invalid parents while restoring a required descendant.
///
/// All paths, source indices and complete bitmap sizes are checked first.
/// Insufficient capacity returns `Ok(None)` without mutation or allocation.
/// Unchanged masks/children stay shared; all-valid masks need no buffer. Each new
/// exact bitmap allocation retains its charge through the last alias. No stage
/// waits for more credit. Input/routing storage, planning/Array/Arc metadata and
/// allocator overhead are not newly admitted here. Required logical-null types
/// have the same restrictions as [`super::try_splice_struct_in`].
///
/// This does not fetch overlay inputs or determine newest-file precedence.
///
/// ```
/// use arrow_array::{ArrayRef, StructArray};
/// use lance_core::{Result, utils::allocation::Reservation};
/// use lance_encoding::data::{StructValiditySelection, try_replace_struct_validities_in};
/// fn restore(base: &StructArray, donor: &StructArray, indices: &[(usize, usize)],
///            workspace: &mut Reservation) -> Result<Option<ArrayRef>> {
///     let sources = [donor];
///     let updates = [StructValiditySelection { path: &[], sources: &sources, indices }];
///     try_replace_struct_validities_in(base, &updates, workspace)
/// }
/// ```
pub fn try_replace_struct_validities_in(
    source: &StructArray,
    updates: &[StructValiditySelection<'_>],
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    for (index, update) in updates.iter().enumerate() {
        if updates[..index]
            .iter()
            .any(|prior| prior.path == update.path)
        {
            return Err(Error::invalid_input(format!(
                "duplicate struct validity path {:?}",
                update.path
            )));
        }
        let mut target = source;
        for &child in update.path {
            target = target
                .columns()
                .get(child)
                .and_then(|array| array.as_struct_opt())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "struct validity path {:?} cannot descend through child {child}",
                        update.path
                    ))
                })?;
        }
        if update.indices.len() != target.len() {
            return Err(Error::invalid_input(format!(
                "struct validity path {:?} has {} selections, expected {} rows",
                update.path,
                update.indices.len(),
                target.len()
            )));
        }
        for &(input, row) in update.indices {
            if update
                .sources
                .get(input)
                .is_none_or(|array| row >= array.len())
            {
                return Err(Error::invalid_input(format!(
                    "struct validity path {:?} selects invalid source {input} row {row}",
                    update.path
                )));
            }
        }
    }
    let plan = ValidityPlan::new(source, &[], updates)?;
    if plan.total_bytes > workspace.len() {
        return Ok(None);
    }
    plan.build(workspace).map(Some)
}

enum Mask {
    Reuse,
    AllValid,
    Allocate(usize),
}

struct ValidityPlan<'a> {
    source: &'a StructArray,
    selection: Option<&'a StructValiditySelection<'a>>,
    children: Vec<(usize, Self)>,
    mask: Mask,
    total_bytes: usize,
}

impl<'a> ValidityPlan<'a> {
    fn new(
        source: &'a StructArray,
        path: &[usize],
        updates: &'a [StructValiditySelection<'a>],
    ) -> Result<Self> {
        check_required_fields(source)?;
        let mut children = Vec::new();
        for (index, child) in source.columns().iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(index);
            if updates
                .iter()
                .any(|update| update.path.starts_with(&child_path))
            {
                let child = child.as_struct_opt().ok_or_else(|| {
                    Error::internal("validated struct validity path changed type")
                })?;
                children.push((index, Self::new(child, &child_path, updates)?));
            }
        }
        let mut plan = Self {
            source,
            selection: updates.iter().find(|update| update.path == path),
            children,
            mask: Mask::Reuse,
            total_bytes: 0,
        };
        let mut is_unchanged = true;
        let mut is_all_valid = true;
        for row in 0..source.len() {
            let is_valid = plan.is_valid(row);
            is_unchanged &= is_valid == source.is_valid(row);
            is_all_valid &= is_valid;
            if is_valid {
                for (index, field) in source.fields().iter().enumerate() {
                    if !field.is_nullable() && !plan.child_is_valid(index, row) {
                        return Err(Error::invalid_input(format!(
                            "struct validity path {path:?} exposes required child '{}' null at row {row}",
                            field.name()
                        )));
                    }
                }
            }
        }
        plan.mask = if is_unchanged {
            Mask::Reuse
        } else if is_all_valid {
            Mask::AllValid
        } else {
            Mask::Allocate(source.len().div_ceil(8))
        };
        plan.total_bytes = match plan.mask {
            Mask::Allocate(bytes) => bytes,
            _ => 0,
        };
        for (_, child) in &plan.children {
            plan.total_bytes = plan
                .total_bytes
                .checked_add(child.total_bytes)
                .ok_or_else(|| Error::invalid_input("struct validity bitmap size overflows"))?;
        }
        Ok(plan)
    }

    fn child_is_valid(&self, index: usize, row: usize) -> bool {
        self.children
            .iter()
            .find(|(child, _)| *child == index)
            .map_or_else(
                || self.source.column(index).is_valid(row),
                |(_, child)| child.is_valid(row),
            )
    }

    fn is_valid(&self, row: usize) -> bool {
        match self.selection {
            Some(selection) => {
                let (input, position) = selection.indices[row];
                selection.sources[input].is_valid(position)
            }
            None => {
                self.source.is_valid(row)
                    && self.children.iter().all(|(index, child)| {
                        self.source.fields()[*index].is_nullable() || child.is_valid(row)
                    })
            }
        }
    }

    fn build(&self, workspace: &mut Reservation) -> Result<ArrayRef> {
        let nulls = match self.mask {
            Mask::Reuse => self.source.nulls().cloned(),
            Mask::AllValid => None,
            Mask::Allocate(bytes) => {
                let mut output = workspace.try_allocate(bytes)?.ok_or_else(|| {
                    Error::internal("struct validity exceeded its complete bitmap plan")
                })?;
                for row in 0..self.source.len() {
                    if self.is_valid(row) {
                        bit_util::set_bit(output.as_mut(), row);
                    }
                }
                Some(NullBuffer::new(BooleanBuffer::new(
                    Buffer::from(output.freeze()),
                    0,
                    self.source.len(),
                )))
            }
        };
        let mut columns = self.source.columns().to_vec();
        for (index, child) in &self.children {
            columns[*index] = child.build(workspace)?;
        }
        Ok(Arc::new(StructArray::try_new_with_length(
            self.source.fields().clone(),
            columns,
            nulls,
            self.source.len(),
        )?))
    }
}

#[cfg(test)]
mod tests;
