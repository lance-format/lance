// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Owned dense levels and validity output with explicit payload admission.

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{CompositeRepDefUnraveler, DefinitionInterpretation, LevelBuffer, RepDefUnraveler};

mod compressed;
mod offsets;

#[derive(Debug)]
pub(super) enum Levels {
    Ordinary(LevelBuffer),
    Admitted { storage: ReservedBuffer, len: usize },
}

impl Levels {
    fn from_owned(storage: ReservedBuffer) -> Result<Self> {
        let bytes = storage.as_ref().len();
        if !bytes.is_multiple_of(2) {
            return Err(Error::invalid_input(
                "dense level storage has a partial u16",
            ));
        }
        Ok(Self::Admitted {
            storage,
            len: bytes / 2,
        })
    }

    pub(super) fn truncate(&mut self, len: usize) {
        match self {
            Self::Ordinary(values) => values.truncate(len),
            // Logical compaction must not refund the still-live physical allocation.
            Self::Admitted { len: current, .. } => *current = (*current).min(len),
        }
    }

    fn copy_in(values: &[u16], workspace: &mut Reservation) -> Result<Self> {
        let bytes = std::mem::size_of_val(values);
        let mut storage = workspace
            .try_allocate(bytes)?
            .ok_or_else(|| Error::internal("dense levels exceeded their complete admission"))?;
        storage
            .as_mut()
            .copy_from_slice(bytemuck::cast_slice(values));
        Ok(Self::Admitted {
            storage,
            len: values.len(),
        })
    }
}

impl Deref for Levels {
    type Target = [u16];

    fn deref(&self) -> &[u16] {
        match self {
            Self::Ordinary(values) => values,
            // ReservedBuffer has Arrow alignment and exactly 2 * input.len bytes.
            // Its logical length can only shrink; no resizing interface is exposed.
            Self::Admitted { storage, len } => &bytemuck::cast_slice(storage.as_ref())[..*len],
        }
    }
}

impl DerefMut for Levels {
    fn deref_mut(&mut self) -> &mut [u16] {
        match self {
            Self::Ordinary(values) => values,
            Self::Admitted { storage, len } => {
                &mut bytemuck::cast_slice_mut(storage.as_mut())[..*len]
            }
        }
    }
}

impl RepDefUnraveler {
    pub(crate) fn validate_dense_levels(
        rep: Option<&[u16]>,
        def: Option<&[u16]>,
        meanings: &[DefinitionInterpretation],
    ) -> Result<()> {
        if meanings.is_empty() {
            return Err(Error::invalid_input(
                "dense levels require a structural interpretation",
            ));
        }
        if rep
            .zip(def)
            .is_some_and(|(rep, def)| rep.len() != def.len())
        {
            return Err(Error::invalid_input(
                "repetition and definition lengths differ",
            ));
        }
        // Bound schema counters before the ordinary metadata constructor uses u16.
        let max_rep = meanings.iter().filter(|m| m.is_list()).count();
        let max_def = meanings.iter().try_fold(0_usize, |total, meaning| {
            let added = match meaning {
                DefinitionInterpretation::AllValidItem | DefinitionInterpretation::AllValidList => {
                    0
                }
                DefinitionInterpretation::NullableAndEmptyableList => 2,
                _ => 1,
            };
            total
                .checked_add(added)
                .ok_or_else(|| Error::invalid_input("definition depth overflows"))
        })?;
        if max_rep > u16::MAX as usize || max_def > u16::MAX as usize {
            return Err(Error::invalid_input("structural depth exceeds u16 levels"));
        }
        if rep.is_some_and(|v| v.iter().any(|&x| x as usize > max_rep))
            || def.is_some_and(|v| v.iter().any(|&x| x as usize > max_def))
        {
            return Err(Error::invalid_input(
                "level exceeds its structural interpretation",
            ));
        }
        Ok(())
    }

    /// Copy both dense level arrays into one admitted workspace. Refusal preserves
    /// the workspace and borrowed input. Input storage must already be accounted.
    /// Charges remain until the unraveler drops, even after logical compaction.
    /// Schema-sized metadata is excluded. Output bitmaps/offsets need separate
    /// admission; this constructor does not make ordinary unravel methods bounded.
    pub fn try_new_in(
        rep: Option<&[u16]>,
        def: Option<&[u16]>,
        meanings: Arc<[DefinitionInterpretation]>,
        num_items: u64,
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        Self::validate_dense_levels(rep, def, &meanings)?;
        let bytes = rep
            .map_or(0, std::mem::size_of_val)
            .checked_add(def.map_or(0, std::mem::size_of_val))
            .ok_or_else(|| Error::invalid_input("dense level allocation size overflows"))?;
        if bytes > workspace.len() {
            return Ok(None);
        }
        let mut result = Self::new(None, None, meanings, num_items);
        result.rep_levels = rep.map(|v| Levels::copy_in(v, workspace)).transpose()?;
        result.def_levels = def.map(|v| Levels::copy_in(v, workspace)).transpose()?;
        Ok(Some(result))
    }

    /// Adopt already admitted level allocations without copying them again.
    /// Validation errors drop the supplied owners and release their charges.
    pub(crate) fn from_admitted_levels(
        rep: Option<ReservedBuffer>,
        def: Option<ReservedBuffer>,
        meanings: Arc<[DefinitionInterpretation]>,
        num_items: u64,
    ) -> Result<Self> {
        let rep = rep.map(Levels::from_owned).transpose()?;
        let def = def.map(Levels::from_owned).transpose()?;
        Self::validate_dense_levels(rep.as_deref(), def.as_deref(), &meanings)?;
        let mut result = Self::new(None, None, meanings, num_items);
        result.rep_levels = rep;
        result.def_levels = def;
        Ok(result)
    }
}

// Iterates the current dense validity layer without a temporary bitmap or index
// array. Construction is preceded by validation, so level lookups are in bounds.
struct ValidityBits<'a> {
    source: &'a RepDefUnraveler,
    next: usize,
}

impl<'a> ValidityBits<'a> {
    fn new(source: &'a RepDefUnraveler) -> Self {
        Self { source, next: 0 }
    }
}

impl Iterator for ValidityBits<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.source.is_all_valid() {
            if self.next < self.source.num_items as usize {
                self.next += 1;
                return Some(true);
            }
            return None;
        }
        let defs = self.source.def_levels.as_ref()?;
        while let Some(&level) = defs.get(self.next) {
            self.next += 1;
            if self.source.levels_to_rep[level as usize] <= self.source.current_rep_cmp {
                return Some(level <= self.source.current_def_cmp);
            }
        }
        None
    }
}

impl CompositeRepDefUnraveler {
    fn validate_admitted_validity(&self, num_values: usize) -> Result<bool> {
        let mut count = 0_usize;
        let mut has_bitmap = false;
        for source in &self.unravelers {
            if source.sparse.is_some() {
                return Err(Error::not_supported(
                    "sparse validity needs its own admitted path",
                ));
            }
            let meaning = source
                .def_meaning
                .get(source.current_layer)
                .ok_or_else(|| {
                    Error::invalid_input("validity requested beyond structural layers")
                })?;
            if meaning.is_list() {
                return Err(Error::invalid_input(
                    "unravel list offsets before its next validity layer",
                ));
            }
            let num_items = usize::try_from(source.num_items)
                .map_err(|_| Error::invalid_input("validity length exceeds usize"))?;
            if source.def_levels.as_ref().is_some_and(|levels| {
                levels
                    .iter()
                    .any(|&level| level as usize >= source.levels_to_rep.len())
            }) {
                return Err(Error::invalid_input(
                    "definition level exceeds validity interpretation",
                ));
            }
            if !source.is_all_valid() && source.current_def_cmp == u16::MAX {
                return Err(Error::invalid_input("validity comparison overflows u16"));
            }
            let source_count = if source.is_all_valid() {
                num_items
            } else {
                ValidityBits::new(source).count()
            };
            count = count
                .checked_add(source_count)
                .ok_or_else(|| Error::invalid_input("combined validity length overflows"))?;
            has_bitmap |= !source.is_all_valid();
        }
        if count != num_values {
            return Err(Error::invalid_input(format!(
                "validity produced {count} values, expected {num_values}"
            )));
        }
        for comparison in &self.comparisons {
            comparison.validate_admitted_validity(num_values)?;
            if !self
                .unravelers
                .iter()
                .flat_map(ValidityBits::new)
                .eq(comparison.unravelers.iter().flat_map(ValidityBits::new))
            {
                return Err(Error::invalid_input(
                    "structural sibling fields have incompatible validity metadata",
                ));
            }
        }
        Ok(has_bitmap)
    }

    fn advance_admitted_validity(&mut self) {
        for source in &mut self.unravelers {
            if !source.is_all_valid() {
                source.current_def_cmp += 1;
            }
            source.current_layer += 1;
        }
        for comparison in &mut self.comparisons {
            comparison.advance_admitted_validity();
        }
    }

    /// Materialize dense item/struct validity in admitted storage. Outer `None`
    /// means insufficient credit and leaves all cursors/workspace unchanged;
    /// `Some(None)` means the layer is all valid. All sibling comparisons and row
    /// counts are checked before allocation or cursor advancement. Source level
    /// allocations must already be charged, e.g. by `RepDefUnraveler::try_new_in`.
    /// List offsets have their own `try_unravel_offsets_in` admission. Sparse
    /// plans are not covered. Ordinary methods retain their behavior/allocations.
    pub fn try_unravel_validity_in(
        &mut self,
        num_values: usize,
        workspace: &mut Reservation,
    ) -> Result<Option<Option<NullBuffer>>> {
        let has_bitmap = self.validate_admitted_validity(num_values)?;
        let validity = if has_bitmap {
            let Some(mut buffer) = workspace.try_allocate(num_values.div_ceil(8))? else {
                return Ok(None);
            };
            for (i, valid) in self
                .unravelers
                .iter()
                .flat_map(ValidityBits::new)
                .enumerate()
            {
                if valid {
                    buffer.as_mut()[i / 8] |= 1 << (i % 8);
                }
            }
            Some(NullBuffer::new(BooleanBuffer::new(
                Buffer::from(buffer.freeze()),
                0,
                num_values,
            )))
        } else {
            None
        };
        self.advance_admitted_validity();
        Ok(Some(validity))
    }
}

#[cfg(test)]
mod tests;
