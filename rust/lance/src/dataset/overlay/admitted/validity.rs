// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared-ancestor validity precedence across independently routed fields.

use lance_encoding::data::StructValiditySelection;

use super::*;

/// One atomic field's already-admitted inputs for a shared struct ancestor.
///
/// All inputs in a resolution must describe the same selected output rows and
/// the same ancestor. Sources may have different projected children: only their
/// struct validity is inspected. The caller establishes file/row identity and
/// accounts for the original routing and fetched validity owners.
pub struct OverlayStructValidityInput<'a> {
    /// Atomic field routing; source zero contributes no overlay validity.
    pub routing: &'a AdmittedOverlayRouting,
    /// Global newest-first file ranks corresponding to `sources` (lower is newer).
    /// Ranks must be strictly increasing within this input, matching its routing.
    /// The same rank across inputs denotes the same file.
    pub file_order: &'a [usize],
    /// Fetched ancestor arrays, ordered like this field's overlay sources, with
    /// exactly the sorted unique rows requested by its routing.
    pub sources: &'a [&'a StructArray],
}

/// Final admitted validity selection for one shared struct ancestor.
///
/// The row table holds one `(source, row)` pair per output row under one physical
/// owner. No overlapping per-row candidate or priority table is built. Source
/// references and per-field descriptors remain metadata outside that charge.
/// This is opt-in; ordinary overlay reads and performance remain unchanged.
pub struct AdmittedStructValidityRouting<'a> {
    storage: Buffer,
    rows: usize,
    sources: Vec<&'a StructArray>,
}

impl<'a> AdmittedStructValidityRouting<'a> {
    /// Resolve global file precedence without allocating an uncharged row table.
    ///
    /// The newest file selected by any contributing field wins, even if its
    /// struct is null. Uncovered rows use `base` validity; supply the original
    /// base ancestor, before child splicing may have masked it. Equal file ranks
    /// keep the first input, matching ordinary overlay resolution. Callers must
    /// establish that equal ranks really refer to the same file/ancestor.
    /// All row/source counts and file orders are checked before allocation, even
    /// without credit. `None` leaves the workspace unchanged and means replan,
    /// never missing validity or successful partial output. No stage waits for
    /// additional credit. Repeated field probes are not performance-qualified.
    ///
    /// ```
    /// use arrow_array::{ArrayRef, StructArray};
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use lance::dataset::{AdmittedStructValidityRouting, OverlayStructValidityInput};
    /// use lance_encoding::data::try_replace_struct_validities_in;
    /// fn resolve(base: &StructArray, children: &StructArray,
    ///     inputs: &[OverlayStructValidityInput<'_>], work: &mut Reservation)
    ///     -> Result<Option<ArrayRef>> {
    ///     let Some(routing) = AdmittedStructValidityRouting::try_new(base, inputs, work)?
    ///         else { return Ok(None) };
    ///     try_replace_struct_validities_in(children, &[routing.selection(&[])], work)
    /// }
    /// ```
    pub fn try_new(
        base: &'a StructArray,
        inputs: &[OverlayStructValidityInput<'a>],
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        let rows = base.len();
        let bytes = checked_index_bytes(rows)?;
        let mut source_count = 1_usize;
        for input in inputs {
            if input.routing.rows != rows
                || input.sources.len() != input.routing.offsets.len()
                || input.file_order.len() != input.sources.len()
            {
                return Err(Error::invalid_input(
                    "struct validity input row/source counts differ from routing or base",
                ));
            }
            if input.file_order.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(Error::invalid_input(
                    "struct validity file order must be strictly newest-first",
                ));
            }
            for (array, range) in input.sources.iter().zip(&input.routing.offsets) {
                if array.len() != range.len() / size_of::<u32>() {
                    return Err(Error::invalid_input(
                        "struct validity fetched row count differs from routed offsets",
                    ));
                }
            }
            source_count = source_count
                .checked_add(input.sources.len())
                .ok_or_else(|| Error::invalid_input("struct validity source count overflows"))?;
        }
        let Some(mut storage) = workspace.try_allocate(bytes)? else {
            return Ok(None);
        };
        let buffer = storage.as_mut();
        if !buffer.as_ptr().cast::<(usize, usize)>().is_aligned() {
            return Err(Error::internal(
                "struct validity routing is not tuple aligned",
            ));
        }
        // SAFETY: checked_index_bytes rejects padding and overflow. The reserved
        // zeroed buffer is tuple-aligned and has exactly rows tuples; all bit
        // patterns are valid usizes. This exclusive view never escapes.
        let indices = unsafe {
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<(usize, usize)>(), rows)
        };
        for (row, output) in indices.iter_mut().enumerate() {
            *output = (0, row);
            let mut newest = None;
            let mut source_start = 1;
            for input in inputs {
                let (source, position) = input.routing.indices()[row];
                if let Some(overlay) = source.checked_sub(1) {
                    let file = input.file_order[overlay];
                    if newest.is_none_or(|current| file < current) {
                        newest = Some(file);
                        *output = (source_start + overlay, position);
                    }
                }
                // The complete sum was checked before allocation. Individual
                // source/position bounds follow the immutable routing constructor.
                source_start += input.sources.len();
            }
        }
        let mut sources = Vec::with_capacity(source_count);
        sources.push(base);
        sources.extend(
            inputs
                .iter()
                .flat_map(|input| input.sources.iter().copied()),
        );
        Ok(Some(Self {
            storage: Buffer::from(storage.freeze()),
            rows,
            sources,
        }))
    }

    /// Borrow the final selection for simultaneous, admitted whole-tree restoration.
    ///
    /// `path` is a struct-child index path in the reconstructed target tree. Use
    /// this together with the selections for all affected ancestors in one
    /// `try_replace_struct_validities_in` call; do not restore ancestors in an
    /// arbitrary sequence that can invalidate still-unrestored required children.
    pub fn selection<'b>(&'b self, path: &'b [usize]) -> StructValiditySelection<'b> {
        let indices = if self.rows == 0 {
            &[]
        } else {
            // SAFETY: try_new initializes this exact immutable tuple buffer; the
            // borrowed selection cannot outlive its charged physical owner.
            unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast(), self.rows) }
        };
        StructValiditySelection {
            path,
            sources: &self.sources,
            indices,
        }
    }
}

#[cfg(test)]
mod tests;
