// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::OffsetSizeTrait;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::{CompositeRepDefUnraveler, DefinitionInterpretation as Meaning, RepDefUnraveler};

type OffsetOutput<T> = (OffsetBuffer<T>, Option<NullBuffer>);

struct LayerPlan {
    null: u16,
    empty: u16,
    upper_null: u16,
    max_visible: u16,
    next_def: u16,
    next_rep: u16,
    lists: usize,
}

impl LayerPlan {
    fn new(source: &RepDefUnraveler) -> Result<Self> {
        if source.sparse.is_some() {
            return Err(Error::not_supported(
                "sparse offsets need their own admitted path",
            ));
        }
        let remaining = source
            .def_meaning
            .get(source.current_layer..)
            .ok_or_else(|| Error::invalid_input("offsets requested beyond structural layers"))?;
        let valid = source.current_def_cmp;
        let add = |n| {
            valid
                .checked_add(n)
                .ok_or_else(|| Error::invalid_input("list definition level overflows u16"))
        };
        let (null, empty, next_def) = match remaining.first() {
            Some(Meaning::NullableList) => (add(1)?, 0, add(1)?),
            Some(Meaning::EmptyableList) => (0, add(1)?, add(1)?),
            Some(Meaning::NullableAndEmptyableList) => (add(1)?, add(2)?, add(2)?),
            Some(Meaning::AllValidList) => (0, 0, valid),
            _ => return Err(Error::invalid_input("offsets require a list layer")),
        };
        let next_rep = source
            .current_rep_cmp
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("list repetition level overflows u16"))?;
        let upper_null = valid.max(null).max(empty);
        let mut max_visible = upper_null;
        for meaning in &remaining[1..] {
            match meaning {
                Meaning::NullableItem => {
                    max_visible = max_visible.checked_add(1).ok_or_else(|| {
                        Error::invalid_input("parent definition level overflows u16")
                    })?
                }
                Meaning::AllValidItem => {}
                _ => break,
            }
        }
        let rep = source
            .rep_levels
            .as_ref()
            .ok_or_else(|| Error::invalid_input("list offsets require repetition levels"))?;
        let remaining_lists = remaining.iter().filter(|m| m.is_list()).count();
        if rep.iter().any(|&v| v as usize > remaining_lists)
            || source.def_levels.as_ref().is_some_and(|def| {
                rep.len() != def.len()
                    || def
                        .iter()
                        .any(|&v| v as usize >= source.levels_to_rep.len())
            })
        {
            return Err(Error::invalid_input(
                "list levels do not match the remaining structure",
            ));
        }
        Ok(Self {
            null,
            empty,
            upper_null,
            max_visible,
            next_def,
            next_rep,
            lists: 0,
        })
    }

    // None is invisible at this layer (or a continuation); a present bool is
    // the validity of a new list. The second value says whether a child exists.
    fn event(&self, rep: u16, def: u16) -> (Option<bool>, bool) {
        if rep == 0 {
            (None, true)
        } else if def == 0 {
            (Some(true), true)
        } else if def > self.max_visible {
            (None, false)
        } else if def == self.null || def > self.upper_null {
            (Some(false), false)
        } else if def == self.empty {
            (Some(true), false)
        } else {
            (Some(true), true)
        }
    }

    fn events<'a>(
        &'a self,
        source: &'a RepDefUnraveler,
    ) -> impl Iterator<Item = (Option<bool>, bool)> + 'a {
        source
            .rep_levels
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &rep)| self.event(rep, source.def_levels.as_ref().map_or(0, |def| def[i])))
    }

    fn advance(&self, source: &mut RepDefUnraveler) {
        let reps = source.rep_levels.as_mut().unwrap();
        let mut write = 0;
        for read in 0..reps.len() {
            if reps[read] != 0 {
                reps[write] = reps[read] - 1;
                if let Some(defs) = &mut source.def_levels {
                    defs[write] = defs[read];
                }
                write += 1;
            }
        }
        reps.truncate(write);
        if let Some(defs) = &mut source.def_levels {
            defs.truncate(write);
        }
        source.num_items = self.lists as u64;
        source.current_layer += 1;
        source.current_def_cmp = self.next_def;
        source.current_rep_cmp = self.next_rep;
    }
}

// One entry per page/sibling, never per row. Payload output and comparisons are
// scanned without temporary offset/index arrays. This metadata is not charged.
struct OffsetPlan {
    layers: Vec<LayerPlan>,
    comparisons: Vec<Self>,
    lists: usize,
    children: usize,
    has_bitmap: bool,
}

impl OffsetPlan {
    fn new(source: &CompositeRepDefUnraveler) -> Result<Self> {
        let mut plan = Self {
            layers: Vec::with_capacity(source.unravelers.len()),
            comparisons: Vec::with_capacity(source.comparisons.len()),
            lists: 0,
            children: 0,
            has_bitmap: false,
        };
        for page in &source.unravelers {
            let mut layer = LayerPlan::new(page)?;
            let mut lists = 0_usize;
            for (start, child) in layer.events(page) {
                lists = lists
                    .checked_add(usize::from(start.is_some()))
                    .ok_or_else(|| Error::invalid_input("list count overflows usize"))?;
                plan.children = plan
                    .children
                    .checked_add(usize::from(child))
                    .ok_or_else(|| Error::invalid_input("list child count overflows usize"))?;
            }
            layer.lists = lists;
            plan.lists = plan
                .lists
                .checked_add(lists)
                .ok_or_else(|| Error::invalid_input("combined list count overflows usize"))?;
            plan.has_bitmap |= !page.is_all_valid();
            plan.layers.push(layer);
        }
        for sibling in &source.comparisons {
            let other = Self::new(sibling)?;
            if plan.children != other.children || !plan.starts(source).eq(other.starts(sibling)) {
                return Err(Error::invalid_input(
                    "structural sibling fields have incompatible list metadata",
                ));
            }
            plan.comparisons.push(other);
        }
        Ok(plan)
    }

    fn starts<'a>(
        &'a self,
        source: &'a CompositeRepDefUnraveler,
    ) -> impl Iterator<Item = (usize, bool)> + 'a {
        let mut offset = 0;
        self.layers
            .iter()
            .zip(&source.unravelers)
            .flat_map(|(layer, page)| layer.events(page))
            .filter_map(move |(start, child)| {
                let result = start.map(|valid| (offset, !self.has_bitmap || valid));
                offset += usize::from(child); // Total was checked during planning.
                result
            })
    }

    fn advance(&self, source: &mut CompositeRepDefUnraveler) {
        for (layer, page) in self.layers.iter().zip(&mut source.unravelers) {
            layer.advance(page);
        }
        for (plan, sibling) in self.comparisons.iter().zip(&mut source.comparisons) {
            plan.advance(sibling);
        }
    }
}

impl CompositeRepDefUnraveler {
    /// Reconstruct dense List/LargeList offsets and validity in admitted owners.
    /// All page/sibling checks precede allocation and cursor changes. `None`
    /// means insufficient credit or an offset width too small for this selection;
    /// the caller may replan, with input/workspace unchanged. Malformed metadata
    /// is an error. Continuations across page boundaries retain their semantics.
    ///
    /// Input level allocations must already be charged. Logical compaction does
    /// not release their physical credit. Page/schema-sized planning metadata is
    /// excluded; this does not make the ordinary decode path memory bounded.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::utils::allocation::AllocationBudget;
    /// use lance_encoding::repdef::{CompositeRepDefUnraveler, DefinitionInterpretation, RepDefUnraveler};
    /// let budget = AllocationBudget::new(18); // 6 level bytes + 12 offset bytes
    /// let mut work = budget.try_reserve(18).unwrap();
    /// let levels = RepDefUnraveler::try_new_in(
    ///     Some(&[1, 0, 1]), None,
    ///     Arc::from([DefinitionInterpretation::AllValidList]), 3, &mut work,
    /// ).unwrap().unwrap();
    /// let mut source = CompositeRepDefUnraveler::new(vec![levels]);
    /// let (offsets, validity) = source.try_unravel_offsets_in::<i32>(&mut work).unwrap().unwrap();
    /// assert_eq!(offsets.as_ref(), &[0, 2, 3]);
    /// assert!(validity.is_none());
    /// drop(source);
    /// drop(work);
    /// assert_eq!(budget.reserved(), 12);
    /// drop(offsets);
    /// assert_eq!(budget.reserved(), 0);
    /// ```
    pub fn try_unravel_offsets_in<T: OffsetSizeTrait + bytemuck::Pod>(
        &mut self,
        workspace: &mut Reservation,
    ) -> Result<Option<OffsetOutput<T>>> {
        let plan = OffsetPlan::new(self)?;
        if T::from_usize(plan.children).is_none() {
            return Ok(None);
        }
        let count = plan
            .lists
            .checked_add(1)
            .ok_or_else(|| Error::invalid_input("list offset count overflows usize"))?;
        let offset_bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::invalid_input("list offset payload overflows usize"))?;
        let validity_bytes = if plan.has_bitmap {
            plan.lists.div_ceil(8)
        } else {
            0
        };
        let total = offset_bytes
            .checked_add(validity_bytes)
            .ok_or_else(|| Error::invalid_input("list output payload overflows usize"))?;
        // Check both layouts before either reservation can be split.
        for bytes in [offset_bytes, validity_bytes] {
            std::alloc::Layout::from_size_align(bytes, arrow_buffer::alloc::ALIGNMENT).map_err(
                |_| Error::invalid_input("list output exceeds addressable allocation size"),
            )?;
        }
        if total > workspace.len() {
            return Ok(None);
        }
        let mut offsets = workspace.try_allocate(offset_bytes)?.unwrap();
        let mut validity = if plan.has_bitmap {
            Some(workspace.try_allocate(validity_bytes)?.unwrap())
        } else {
            None
        };
        let typed = bytemuck::cast_slice_mut::<u8, T>(offsets.as_mut());
        for (i, (offset, valid)) in plan.starts(self).enumerate() {
            typed[i] = T::from_usize(offset).unwrap();
            if valid && let Some(bits) = &mut validity {
                bits.as_mut()[i / 8] |= 1 << (i % 8);
            }
        }
        typed[plan.lists] = T::from_usize(plan.children).unwrap();
        let offsets =
            OffsetBuffer::new(ScalarBuffer::new(Buffer::from(offsets.freeze()), 0, count));
        let validity = validity.map(|bits| {
            NullBuffer::new(BooleanBuffer::new(
                Buffer::from(bits.freeze()),
                0,
                plan.lists,
            ))
        });
        plan.advance(self);
        Ok(Some((offsets, validity)))
    }
}

#[cfg(test)]
mod tests;
