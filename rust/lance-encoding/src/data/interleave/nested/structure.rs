// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::StructArray;

use super::*;

#[cfg(test)]
mod tests;

pub(super) struct StructPlan {
    rows: usize,
    validity_bytes: usize,
    children: Vec<Plan>,
    pub(super) total_bytes: usize,
}

fn struct_sources<'a>(arrays: &[&'a dyn Array]) -> Result<Vec<&'a StructArray>> {
    arrays
        .iter()
        .map(|array| {
            array.as_struct_opt().ok_or_else(|| {
                invalid(format!("expected Struct source, got {}", array.data_type()))
            })
        })
        .collect()
}

pub(super) fn plan_struct(arrays: &[&dyn Array], selection: &dyn Selection) -> Result<StructPlan> {
    let structs = struct_sources(arrays)?;
    let mut rows = 0;
    let mut has_nulls = false;
    // Validate the selection even for structs with no fields. Their child passes
    // cannot establish the parent row domain or its validity allocation.
    for (source, row) in selection.iter() {
        let array = structs
            .get(source)
            .ok_or_else(|| invalid(format!("source {source} exceeds {} sources", structs.len())))?;
        if row >= array.len() {
            return Err(invalid(format!(
                "row {row} exceeds source {source} length {}",
                array.len()
            )));
        }
        rows = add(rows, 1)?;
        has_nulls |= array.is_null(row);
    }
    let validity_bytes = if has_nulls { rows.div_ceil(8) } else { 0 };
    Layout::from_size_align(validity_bytes, ALIGNMENT)
        .map_err(|_| invalid("struct validity exceeds addressable allocation size"))?;
    let mut total_bytes = validity_bytes;
    let mut children = Vec::with_capacity(structs[0].num_columns());
    for column in 0..structs[0].num_columns() {
        let arrays: Vec<&dyn Array> = structs
            .iter()
            .map(|array| array.column(column).as_ref())
            .collect();
        let child = plan_tree(&arrays, selection)?;
        total_bytes = add(total_bytes, child.bytes())?;
        children.push(child);
    }
    Ok(StructPlan {
        rows,
        validity_bytes,
        children,
        total_bytes,
    })
}

pub(super) fn build_struct(
    arrays: &[&dyn Array],
    selection: &dyn Selection,
    plan: &StructPlan,
    workspace: &mut Reservation,
) -> Result<ArrayRef> {
    let structs = struct_sources(arrays)?;
    let mut children = Vec::with_capacity(plan.children.len());
    for (column, child_plan) in plan.children.iter().enumerate() {
        let arrays: Vec<&dyn Array> = structs
            .iter()
            .map(|array| array.column(column).as_ref())
            .collect();
        children.push(build_tree(&arrays, selection, child_plan, workspace)?);
    }
    let nulls = if plan.validity_bytes == 0 {
        None
    } else {
        let mut output = workspace
            .try_allocate(plan.validity_bytes)?
            .ok_or_else(|| {
                Error::internal("admitted struct interleave exceeded its complete output plan")
            })?;
        for (index, (source, row)) in selection.iter().enumerate() {
            if structs[source].is_valid(row) {
                bit_util::set_bit(output.as_mut(), index);
            }
        }
        Some(NullBuffer::new(BooleanBuffer::new(
            Buffer::from(output.freeze()),
            0,
            plan.rows,
        )))
    };
    // Byte/list/struct logical_nulls only clones the existing bitmap. Arrow's
    // non-nullable-child check compares bit chunks without allocating a mask.
    Ok(Arc::new(StructArray::try_new_with_length(
        structs[0].fields().clone(),
        children,
        nulls,
        plan.rows,
    )?))
}
