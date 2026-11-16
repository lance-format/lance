// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Nested list assembly without an expanded child routing table.

use arrow_array::GenericListArray;
use arrow_buffer::ArrowNativeType;
use arrow_schema::FieldRef;

use super::*;

#[cfg(test)]
mod tests;

/// Interleave byte arrays with any number of List/LargeList wrappers.
///
/// Sources have the same Arrow type, with Binary/LargeBinary/Utf8/LargeUtf8
/// leaves. Each index is `(source, row)`; order, duplicates, list fields and
/// null/empty distinctions are preserved. Children hidden by null lists are
/// omitted. Unsupported shapes and invalid input are errors, with no ordinary
/// interleave fallback. Empty selections still validate the entire type tree.
///
/// The complete output tree is planned before any payload allocation. `Ok(None)`
/// leaves the workspace unchanged and means drain/replan, never EOF. No stage
/// waits for credit or grows the workspace while holding input. Each list level
/// owns admitted offsets/validity, and each leaf owns admitted offsets/values/
/// validity. Buffer aliases retain their corresponding full physical charges.
///
/// Child selections are regenerated from borrowed list ranges on each pass,
/// without an index table proportional to the number of child elements. Planning
/// metadata scales with source count and nesting depth; it, Array/Arc headers,
/// input/routing storage and allocator overhead are outside this payload budget.
/// The caller must reserve the complete input/output overlap before reads.
pub fn try_interleave_nested_binary_in(
    arrays: &[&dyn Array],
    indices: &[(usize, usize)],
    workspace: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let selection = RootSelection(indices);
    let plan = plan_tree(arrays, &selection)?;
    if plan.bytes() > workspace.len() {
        return Ok(None);
    }
    build_tree(arrays, &selection, &plan, workspace).map(Some)
}

enum Plan {
    Bytes(BytePlan),
    List {
        own: BytePlan,
        child: Box<Plan>,
        total_bytes: usize,
    },
}

impl Plan {
    fn bytes(&self) -> usize {
        match self {
            Self::Bytes(plan) => plan.total_bytes,
            Self::List { total_bytes, .. } => *total_bytes,
        }
    }
}

fn list_sources<'a, O: OffsetSizeTrait>(
    arrays: &[&'a dyn Array],
) -> Result<Vec<&'a GenericListArray<O>>> {
    arrays
        .iter()
        .map(|array| {
            array.as_list_opt::<O>().ok_or_else(|| {
                invalid(format!(
                    "expected {}List source, got {}",
                    O::PREFIX,
                    array.data_type()
                ))
            })
        })
        .collect()
}

struct Children<'a, O: OffsetSizeTrait> {
    parents: &'a dyn Selection,
    arrays: &'a [&'a GenericListArray<O>],
}

impl<O: OffsetSizeTrait> Selection for Children<'_, O> {
    fn iter(&self) -> Box<dyn Iterator<Item = (usize, usize)> + '_> {
        Box::new(self.parents.iter().flat_map(|(source, row)| {
            // The parent planning pass validated these source/row indices.
            let array = self.arrays[source];
            let range = if array.is_null(row) {
                0..0
            } else {
                let offsets = array.value_offsets();
                offsets[row].as_usize()..offsets[row + 1].as_usize()
            };
            range.map(move |row| (source, row))
        }))
    }
}

fn plan_tree(arrays: &[&dyn Array], selection: &dyn Selection) -> Result<Plan> {
    let first = arrays
        .first()
        .ok_or_else(|| invalid("sources must not be empty"))?;
    for array in arrays {
        if array.data_type() != first.data_type() {
            return Err(invalid(format!(
                "expected source type {}, got {}",
                first.data_type(),
                array.data_type()
            )));
        }
    }
    match first.data_type() {
        DataType::Binary => {
            plan_bytes::<GenericBinaryType<i32>>(arrays, selection).map(Plan::Bytes)
        }
        DataType::LargeBinary => {
            plan_bytes::<GenericBinaryType<i64>>(arrays, selection).map(Plan::Bytes)
        }
        DataType::Utf8 => plan_bytes::<GenericStringType<i32>>(arrays, selection).map(Plan::Bytes),
        DataType::LargeUtf8 => {
            plan_bytes::<GenericStringType<i64>>(arrays, selection).map(Plan::Bytes)
        }
        DataType::List(_) => plan_list::<i32>(arrays, selection),
        DataType::LargeList(_) => plan_list::<i64>(arrays, selection),
        data_type => Err(Error::not_supported(format!(
            "admitted nested byte interleave does not support {data_type}"
        ))),
    }
}

fn plan_list<O: OffsetSizeTrait>(arrays: &[&dyn Array], selection: &dyn Selection) -> Result<Plan> {
    let lists = list_sources::<O>(arrays)?;
    let mut rows = 0;
    let mut child_rows = 0;
    let mut has_nulls = false;
    for (source, row) in selection.iter() {
        let array = lists
            .get(source)
            .ok_or_else(|| invalid(format!("source {source} exceeds {} sources", lists.len())))?;
        if row >= array.len() {
            return Err(invalid(format!(
                "row {row} exceeds source {source} length {}",
                array.len()
            )));
        }
        rows = add(rows, 1)?;
        if array.is_null(row) {
            has_nulls = true;
        } else {
            let offsets = array.value_offsets();
            child_rows = add(
                child_rows,
                offsets[row + 1].as_usize() - offsets[row].as_usize(),
            )?;
        }
    }
    if child_rows > O::MAX_OFFSET {
        return Err(invalid(format!(
            "{child_rows} child rows exceed {}List offset capacity",
            O::PREFIX
        )));
    }
    let own = BytePlan::new::<O>(rows, 0, has_nulls)?;
    let child_arrays: Vec<&dyn Array> = lists.iter().map(|array| array.values().as_ref()).collect();
    let selection = Children {
        parents: selection,
        arrays: &lists,
    };
    let child = Box::new(plan_tree(&child_arrays, &selection)?);
    let total_bytes = add(own.total_bytes, child.bytes())?;
    Ok(Plan::List {
        own,
        child,
        total_bytes,
    })
}

fn build_tree(
    arrays: &[&dyn Array],
    selection: &dyn Selection,
    plan: &Plan,
    workspace: &mut Reservation,
) -> Result<ArrayRef> {
    match (arrays[0].data_type(), plan) {
        (DataType::Binary, Plan::Bytes(plan)) => {
            build_bytes::<GenericBinaryType<i32>>(arrays, selection, plan, workspace)
        }
        (DataType::LargeBinary, Plan::Bytes(plan)) => {
            build_bytes::<GenericBinaryType<i64>>(arrays, selection, plan, workspace)
        }
        (DataType::Utf8, Plan::Bytes(plan)) => {
            build_bytes::<GenericStringType<i32>>(arrays, selection, plan, workspace)
        }
        (DataType::LargeUtf8, Plan::Bytes(plan)) => {
            build_bytes::<GenericStringType<i64>>(arrays, selection, plan, workspace)
        }
        (DataType::List(field), Plan::List { own, child, .. }) => {
            build_list::<i32>(arrays, selection, own, child, field, workspace)
        }
        (DataType::LargeList(field), Plan::List { own, child, .. }) => {
            build_list::<i64>(arrays, selection, own, child, field, workspace)
        }
        _ => Err(Error::internal(
            "admitted interleave type differs from its complete plan",
        )),
    }
}

fn build_list<O: OffsetSizeTrait>(
    arrays: &[&dyn Array],
    selection: &dyn Selection,
    own: &BytePlan,
    child_plan: &Plan,
    field: &FieldRef,
    workspace: &mut Reservation,
) -> Result<ArrayRef> {
    let lists = list_sources::<O>(arrays)?;
    let child_arrays: Vec<&dyn Array> = lists.iter().map(|array| array.values().as_ref()).collect();
    let children = Children {
        parents: selection,
        arrays: &lists,
    };
    let child = build_tree(&child_arrays, &children, child_plan, workspace)?;
    let mut output = workspace.try_allocate(own.total_bytes)?.ok_or_else(|| {
        Error::internal("admitted list interleave exceeded its complete output plan")
    })?;
    let bytes = output.as_mut();
    let mut child_end = 0;
    for (index, (source, row)) in selection.iter().enumerate() {
        let array = lists[source];
        if array.is_valid(row) {
            let offsets = array.value_offsets();
            // The complete planning pass checked this cumulative child count
            // and its representability in the parent's offset type.
            child_end += offsets[row + 1].as_usize() - offsets[row].as_usize();
            if own.validity_bytes != 0 {
                bit_util::set_bit(&mut bytes[own.offset_bytes..], index);
            }
        }
        own.write_offset(bytes, index, child_end);
    }
    let (offsets, _, nulls) = own.finish::<O>(Buffer::from(output.freeze()));
    Ok(Arc::new(GenericListArray::<O>::try_new(
        field.clone(),
        offsets,
        child,
        nulls,
    )?))
}
