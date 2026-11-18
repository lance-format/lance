// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Materialize a single raw byte-list layer without dense rep/def allocations.

use arrow_array::{ArrayRef, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, alloc::ALIGNMENT};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{super::unzip_admitted::visit, FullZipArrayInput};
use crate::{compression::MiniBlockOutputLayout, repdef::DefinitionInterpretation as Meaning};

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::invalid_input("raw FullZip list output size overflows"))
}

#[derive(Default, PartialEq)]
struct Counts {
    rows: usize,
    children: usize,
    values: usize,
    parent_nulls: usize,
    child_nulls: usize,
}

// A hidden parent has exactly one marker and cannot acquire continuation items.
// This also rejects corruption before checking output credit.
#[derive(Default)]
struct State {
    counts: Counts,
    hidden_parent: bool,
}

impl State {
    fn push(
        &mut self,
        rep: u16,
        def: u16,
        visible: bool,
        value: &[u8],
        null: Option<u16>,
    ) -> Result<()> {
        if rep == 1 {
            self.counts.rows = add(self.counts.rows, 1)?;
            self.counts.parent_nulls =
                add(self.counts.parent_nulls, usize::from(null == Some(def)))?;
            self.hidden_parent = !visible;
        } else if self.counts.rows == 0 || self.hidden_parent || !visible {
            return Err(Error::corrupt_file_named(
                "FullZip",
                "list continuation has no visible parent",
            ));
        }
        if visible {
            self.counts.children = add(self.counts.children, 1)?;
            self.counts.values = add(self.counts.values, value.len())?;
            self.counts.child_nulls = add(self.counts.child_nulls, usize::from(def != 0))?;
        }
        Ok(())
    }
}

struct Layout {
    offsets: usize,
    values: usize,
    validity: usize,
    total: usize,
}

impl Layout {
    fn new(rows: usize, width: usize, values: usize, nulls: usize) -> Result<Self> {
        let offsets = add(rows, 1)?
            .checked_mul(width)
            .ok_or_else(|| Error::invalid_input("raw FullZip list offsets overflow"))?;
        let validity = if nulls == 0 { 0 } else { rows.div_ceil(8) };
        let total = add(add(offsets, values)?, validity)?;
        std::alloc::Layout::from_size_align(total, ALIGNMENT).map_err(|_| {
            Error::invalid_input("raw FullZip list output exceeds addressable size")
        })?;
        Ok(Self {
            offsets,
            values,
            validity,
            total,
        })
    }

    fn nulls(&self, storage: &Buffer, rows: usize) -> Option<NullBuffer> {
        (self.validity != 0).then(|| {
            NullBuffer::new(BooleanBuffer::new(
                storage.slice_with_length(self.offsets + self.values, self.validity),
                0,
                rows,
            ))
        })
    }

    fn set_valid(&self, storage: &mut [u8], row: usize) {
        if self.validity != 0 {
            storage[self.offsets + self.values + row / 8] |= 1 << (row % 8);
        }
    }
}

fn offset<const WIDTH: usize>(storage: &mut [u8], row: usize, value: usize) {
    // Preflight proves the signed offset limit and complete buffer size.
    let dest = &mut storage[row * WIDTH..(row + 1) * WIDTH];
    if WIDTH == 4 {
        dest.copy_from_slice(&(value as i32).to_ne_bytes());
    } else {
        dest.copy_from_slice(&(value as i64).to_ne_bytes());
    }
}

pub(super) fn decode(
    input: &FullZipArrayInput,
    work: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let (field, parent_width) = match &input.data_type {
        DataType::List(field) => (field, 4),
        DataType::LargeList(field) => (field, 8),
        _ => {
            return Err(Error::internal(
                "raw FullZip list decoder requires one list",
            ));
        }
    };
    let meanings = &input.scheduler.details.def_meaning;
    let [leaf, parent] = meanings.as_ref() else {
        return Err(Error::invalid_input(
            "raw FullZip list requires exactly two structural layers",
        ));
    };
    let null = match parent {
        Meaning::NullableList | Meaning::NullableAndEmptyableList => {
            Some(leaf.num_def_levels() + 1)
        }
        Meaning::AllValidList | Meaning::EmptyableList => None,
        _ => return Err(Error::invalid_input("raw FullZip parent must be a list")),
    };
    let mut state = State::default();
    visit(
        &input.scheduler,
        std::slice::from_ref(&input.input),
        input.count,
        0..input.count,
        |rep, def, visible, value| state.push(rep, def, visible, value, null),
    )?;
    let counts = state.counts;
    if counts.rows as u64 != input.count {
        return Err(Error::corrupt_file_named(
            "FullZip",
            "raw list row count differs from selected rows",
        ));
    }
    if parent_width == 4 && counts.children > i32::MAX as usize
        || parent_width == 8 && counts.children as u64 > i64::MAX as u64
    {
        return Err(Error::invalid_input(
            "raw FullZip list child count exceeds signed offsets",
        ));
    }
    let child_width = usize::from(input.scheduler.bits_per_offset / 8);
    let parent = Layout::new(counts.rows, parent_width, 0, counts.parent_nulls)?;
    let child = Layout::new(
        counts.children,
        child_width,
        counts.values,
        counts.child_nulls,
    )?;
    MiniBlockOutputLayout::VariableWidth {
        bits_per_offset: input.scheduler.bits_per_offset,
    }
    .validate_range_bytes(counts.children as u64, add(child.offsets, child.values)?)?;
    if add(parent.total, child.total)? > work.len() {
        return Ok(None);
    }
    match (parent_width, child_width) {
        (4, 4) => materialize::<4, 4>(input, field.data_type(), counts, parent, child, null, work),
        (4, 8) => materialize::<4, 8>(input, field.data_type(), counts, parent, child, null, work),
        (8, 4) => materialize::<8, 4>(input, field.data_type(), counts, parent, child, null, work),
        (8, 8) => materialize::<8, 8>(input, field.data_type(), counts, parent, child, null, work),
        _ => Err(Error::internal(
            "validated raw FullZip offset width changed",
        )),
    }
}

fn materialize<const PARENT: usize, const CHILD: usize>(
    input: &FullZipArrayInput,
    leaf: &DataType,
    counts: Counts,
    parent: Layout,
    child: Layout,
    null: Option<u16>,
    work: &mut Reservation,
) -> Result<Option<ArrayRef>> {
    let Some(mut lists) = work.try_allocate(parent.total)? else {
        return Ok(None);
    };
    let Some(mut values) = work.try_allocate(child.total)? else {
        lists.recycle(work)?;
        return Ok(None);
    };
    let mut state = State::default();
    let replay = visit(
        &input.scheduler,
        std::slice::from_ref(&input.input),
        input.count,
        0..input.count,
        |rep, def, visible, value| {
            let before = &state.counts;
            if rep == 1 {
                offset::<PARENT>(lists.as_mut(), before.rows, before.children);
                if null != Some(def) {
                    parent.set_valid(lists.as_mut(), before.rows);
                }
            }
            if visible {
                values.as_mut()
                    [child.offsets + before.values..child.offsets + before.values + value.len()]
                    .copy_from_slice(value);
                offset::<CHILD>(
                    values.as_mut(),
                    before.children + 1,
                    before.values + value.len(),
                );
                if def == 0 {
                    child.set_valid(values.as_mut(), before.children);
                }
            }
            state.push(rep, def, visible, value, null)
        },
    );
    if let Err(error) = replay {
        lists.recycle(work)?;
        values.recycle(work)?;
        return Err(error);
    }
    if state.counts != counts {
        lists.recycle(work)?;
        values.recycle(work)?;
        return Err(Error::internal(
            "raw FullZip list replay disagrees with preflight",
        ));
    }
    offset::<PARENT>(lists.as_mut(), counts.rows, counts.children);
    // Separate aligned final owners avoid padding or alignment copies. Every
    // parent/child buffer alias retains its complete node's physical allocation.
    let values = Buffer::from(values.freeze());
    let child_data = ArrayData::builder(leaf.clone())
        .len(counts.children)
        .add_buffer(values.slice_with_length(0, child.offsets))
        .add_buffer(values.slice_with_length(child.offsets, child.values))
        .nulls(child.nulls(&values, counts.children))
        .build()?;
    let lists = Buffer::from(lists.freeze());
    Ok(Some(make_array(
        ArrayData::builder(input.data_type.clone())
            .len(counts.rows)
            .add_buffer(lists.slice_with_length(0, parent.offsets))
            .add_child_data(child_data)
            .nulls(parent.nulls(&lists, counts.rows))
            .build()?,
    )))
}
