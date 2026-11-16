// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    mem::{align_of, size_of},
    ops::Range,
};

use arrow_buffer::{Buffer, ScalarBuffer};
use lance_core::utils::allocation::Reservation;
use lance_encoding::data::try_interleave_nested_binary_in;

use super::*;

mod merge;
pub use merge::PreparedOverlayBatch;
mod validity;
pub use validity::{AdmittedStructValidityRouting, OverlayStructValidityInput};

/// Newest-first overlay routing with one admitted physical owner.
///
/// The final interleave indices and sorted unique fetch offsets share one exact
/// allocation. Duplicate fetch slots are compacted in place; their backing stays
/// charged, without an overlapping set/hash table. Every exported offset buffer
/// retains the complete allocation until its last alias drops. Coverages, input
/// selection and per-overlay descriptors are borrowed/metadata and are outside
/// this row-storage charge. This does not admit file reads retroactively.
///
/// Routing probes coverage for each selected row and each planning/materializing
/// pass. It is opt-in and not performance-qualified; ordinary routing is unchanged.
pub struct AdmittedOverlayRouting {
    storage: Buffer,
    rows: usize,
    offsets: Vec<Range<usize>>,
    any_overlay: bool,
}

impl AdmittedOverlayRouting {
    /// Plan and allocate all row storage, or return `None` without allocation.
    ///
    /// Source zero is the base; subsequent sources follow `coverages` newest-first.
    /// Null/descending selections are errors even with insufficient workspace.
    /// Output order and duplicate rows are preserved; each overlay's fetch offsets
    /// are sorted and unique. Inputs and dependent output must fit the same admitted
    /// operation: this method never grows or waits for more credit.
    ///
    /// ```
    /// use lance::dataset::{AdmittedOverlayRouting, OverlayBatchSelection};
    /// use lance_core::{Result, utils::allocation::Reservation};
    /// use roaring::RoaringBitmap;
    /// fn route(workspace: &mut Reservation) -> Result<Option<AdmittedOverlayRouting>> {
    ///     let coverage: RoaringBitmap = [1, 3].into_iter().collect();
    ///     AdmittedOverlayRouting::try_new(
    ///         &OverlayBatchSelection::Range(0..4), &[&coverage], workspace)
    /// }
    /// ```
    pub fn try_new(
        selection: &OverlayBatchSelection,
        coverages: &[&RoaringBitmap],
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        let rows = match selection {
            OverlayBatchSelection::Range(range) => range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| Error::invalid_input("overlay routing range is descending"))?
                as usize,
            OverlayBatchSelection::Indices(indices) => {
                if indices.null_count() != 0 {
                    return Err(Error::invalid_input(
                        "overlay routing selection contains nulls",
                    ));
                }
                indices.len()
            }
        };
        let index_bytes = checked_index_bytes(rows)?;
        let mut counts = vec![0_usize; coverages.len()];
        for row in selected_rows(selection) {
            if let Some((overlay, _)) = winner(row, coverages) {
                counts[overlay] += 1;
            }
        }
        let mut total = index_bytes;
        let mut offsets = Vec::with_capacity(counts.len());
        for count in counts {
            let end = count
                .checked_mul(size_of::<u32>())
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| Error::invalid_input("overlay routing fetch size overflows"))?;
            offsets.push(total..end);
            total = end;
        }
        let Some(mut storage) = workspace.try_allocate(total)? else {
            return Ok(None);
        };
        let bytes = storage.as_mut();
        if !bytes.as_ptr().cast::<(usize, usize)>().is_aligned() {
            return Err(Error::internal(
                "reserved overlay routing is not tuple aligned",
            ));
        }
        // SAFETY: split bounds are checked, the reserved base is tuple-aligned,
        // and index_bytes is a multiple of tuple size. Zeroed bytes represent
        // valid usize pairs. Each typed view exclusively borrows its own region.
        let (indices, fetch_bytes) = bytes.split_at_mut(index_bytes);
        let indices = unsafe {
            std::slice::from_raw_parts_mut(indices.as_mut_ptr().cast::<(usize, usize)>(), rows)
        };
        let mut used = vec![0_usize; offsets.len()];
        for (position, row) in selected_rows(selection).enumerate() {
            if let Some((overlay, offset)) = winner(row, coverages) {
                let start = offsets[overlay].start - index_bytes + used[overlay] * size_of::<u32>();
                fetch_bytes[start..start + size_of::<u32>()].copy_from_slice(&offset.to_ne_bytes());
                used[overlay] += 1;
                indices[position] = (overlay + 1, offset as usize);
            } else {
                indices[position] = (0, position);
            }
        }
        for range in &mut offsets {
            let values = &mut fetch_bytes[range.start - index_bytes..range.end - index_bytes];
            // SAFETY: tuple-size boundaries and each u32-sized region preserve
            // u32 alignment. All bits of a u32 are valid; no references escape.
            let values = unsafe {
                std::slice::from_raw_parts_mut(
                    values.as_mut_ptr().cast::<u32>(),
                    values.len() / size_of::<u32>(),
                )
            };
            values.sort_unstable();
            let mut unique = 0;
            for read in 0..values.len() {
                if unique == 0 || values[read] != values[unique - 1] {
                    values[unique] = values[read];
                    unique += 1;
                }
            }
            range.end = range.start + unique * size_of::<u32>();
        }
        for (source, position) in indices {
            if *source == 0 {
                continue;
            }
            let range = &offsets[*source - 1];
            let values = &fetch_bytes[range.start - index_bytes..range.end - index_bytes];
            // SAFETY: same initialized/aligned region as above, now borrowed
            // immutably while the disjoint indices region is updated.
            let values = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u32>(),
                    values.len() / size_of::<u32>(),
                )
            };
            *position = values.binary_search(&(*position as u32)).map_err(|_| {
                Error::internal("overlay fetch offset missing from planned routing")
            })?;
        }
        let any_overlay = offsets.iter().any(|range| !range.is_empty());
        Ok(Some(Self {
            storage: Buffer::from(storage.freeze()),
            rows,
            offsets,
            any_overlay,
        }))
    }

    /// Borrow final `(source, fetched-position)` indices without another row table.
    pub fn indices(&self) -> &[(usize, usize)] {
        if self.rows == 0 {
            return &[];
        }
        // SAFETY: try_new initialized this aligned prefix as usize tuples. The
        // immutable Buffer retains its owner and cannot be mutated through aliases.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast(), self.rows) }
    }

    /// Borrow one overlay's sorted unique fetch offsets with shared ownership.
    /// An empty result requests no values. The index excludes the base source.
    pub fn fetch_offsets(&self, overlay: usize) -> Result<ScalarBuffer<u32>> {
        let range = self.offsets.get(overlay).ok_or_else(|| {
            Error::invalid_input(format!("overlay routing has no overlay {overlay}"))
        })?;
        if self.storage.is_empty() {
            return Ok(ScalarBuffer::from(Vec::<u32>::new()));
        }
        Ok(ScalarBuffer::new(
            self.storage.slice_with_length(range.start, range.len()),
            0,
            range.len() / size_of::<u32>(),
        ))
    }

    /// Interleave already-admitted base/fetched arrays using the original workspace.
    ///
    /// `fetched` is newest-first and has exactly the rows requested by
    /// [`Self::fetch_offsets`]. Covered nulls win; there is no value fall-through.
    /// An uncovered batch shares the base; otherwise the admitted nested byte
    /// interleave type restrictions apply. Metadata and borrowed inputs are not
    /// admitted by this method. `None` requires draining/replanning the operation.
    pub fn try_interleave_in(
        &self,
        base: &ArrayRef,
        fetched: &[ArrayRef],
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayRef>> {
        if base.len() != self.rows || fetched.len() != self.offsets.len() {
            return Err(Error::invalid_input(
                "overlay routing base/fetched lengths do not match plan",
            ));
        }
        for (values, range) in fetched.iter().zip(&self.offsets) {
            if values.len() != range.len() / size_of::<u32>() {
                return Err(Error::invalid_input(
                    "overlay fetched row count does not match planned offsets",
                ));
            }
        }
        if !self.any_overlay {
            return Ok(Some(base.clone()));
        }
        let mut sources: Vec<&dyn Array> = Vec::with_capacity(fetched.len() + 1);
        sources.push(base.as_ref());
        sources.extend(fetched.iter().map(|array| array.as_ref()));
        try_interleave_nested_binary_in(&sources, self.indices(), workspace)
    }
}

// The output is also exposed as initialized bytes. Reject tuple layouts with
// padding before publishing them; the owner supplies Arrow-aligned storage.
fn checked_index_bytes(rows: usize) -> Result<usize> {
    if size_of::<(usize, usize)>() != 2 * size_of::<usize>()
        || !align_of::<(usize, usize)>().is_multiple_of(align_of::<u32>())
    {
        return Err(Error::not_supported(
            "overlay routing requires unpadded aligned usize pairs",
        ));
    }
    rows.checked_mul(size_of::<(usize, usize)>())
        .ok_or_else(|| Error::invalid_input("overlay routing index size overflows"))
}

fn selected_rows(selection: &OverlayBatchSelection) -> Box<dyn Iterator<Item = u32> + '_> {
    match selection {
        OverlayBatchSelection::Range(range) => Box::new(range.clone()),
        OverlayBatchSelection::Indices(indices) => Box::new(indices.values().iter().copied()),
    }
}

fn winner(row: u32, coverages: &[&RoaringBitmap]) -> Option<(usize, u32)> {
    coverages.iter().enumerate().find_map(|(index, coverage)| {
        coverage
            .contains(row)
            .then(|| (index, (coverage.rank(row) - 1) as u32))
    })
}

#[cfg(test)]
mod tests;
