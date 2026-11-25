// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;
use std::sync::Arc;

use lance_core::utils::address::RowAddress;
use lance_select::{RowAddrMask, RowAddrSelection};
use roaring::RoaringBitmap;

const MAX_DENSE_EXPANSION: u64 = 16;
const MAX_SMALL_DENSE_BYTES: u64 = 128 * 1024;
const EXTRA_DENSE_BUDGET: u64 = 4 * 1024 * 1024;
const MAX_PREPARATION_VALUES_PER_PROBE_ROW: u64 = 16;

pub struct PreparedRowAddrMask {
    allow: bool,
    roaring: Option<Arc<RowAddrMask>>,
    fragments: BTreeMap<u32, FragmentSelection>,
}

enum FragmentSelection {
    Full,
    Range { min: u32, max: u32 },
    Sparse(RoaringBitmap),
    Dense { base: u32, words: Vec<u64> },
}

impl FragmentSelection {
    fn prepare(selection: &RowAddrSelection, extra_dense_budget: &mut u64) -> Self {
        let bitmap = match selection {
            RowAddrSelection::Full => return Self::Full,
            RowAddrSelection::Partial(bitmap) if bitmap.is_full() => return Self::Full,
            RowAddrSelection::Partial(bitmap) => bitmap,
        };
        let Some((min, max)) = bitmap.min().zip(bitmap.max()) else {
            return Self::Sparse(bitmap.clone());
        };
        if bitmap.len() == u64::from(max) - u64::from(min) + 1 {
            return Self::Range { min, max };
        }
        let base = min / 64 * 64;
        let num_words = (u64::from(max) - u64::from(base)) / 64 + 1;
        let dense_bytes = num_words * 8;
        if dense_bytes > bitmap.serialized_size() as u64 * MAX_DENSE_EXPANSION {
            if dense_bytes > MAX_SMALL_DENSE_BYTES
                || dense_bytes > *extra_dense_budget
                || bitmap.len() > (u64::from(max) - u64::from(min)).div_ceil(2)
            {
                return Self::Sparse(bitmap.clone());
            }
            *extra_dense_budget -= dense_bytes;
        }
        let mut words = vec![0; num_words as usize];
        for row in bitmap {
            words[((row - base) / 64) as usize] |= 1u64 << (row % 64);
        }
        Self::Dense { base, words }
    }

    fn contains(&self, row: u32) -> bool {
        match self {
            Self::Full => true,
            Self::Range { min, max } => (*min..=*max).contains(&row),
            Self::Sparse(bitmap) => bitmap.contains(row),
            Self::Dense { base, words } => row
                .checked_sub(*base)
                .and_then(|offset| words.get(offset as usize / 64))
                .is_some_and(|word| word & (1u64 << (row % 64)) != 0),
        }
    }
}

impl PreparedRowAddrMask {
    pub(crate) fn for_probe_rows(mask: Arc<RowAddrMask>, probe_rows: usize) -> Self {
        let rows = match mask.as_ref() {
            RowAddrMask::AllowList(rows) | RowAddrMask::BlockList(rows) => rows,
        };
        let budget = (probe_rows as u64).saturating_mul(MAX_PREPARATION_VALUES_PER_PROBE_ROW);
        let mut values = 0_u64;
        for (_, selection) in rows.iter() {
            values = values.saturating_add(match selection {
                RowAddrSelection::Full => 1,
                RowAddrSelection::Partial(bitmap) => bitmap.len().saturating_add(1),
            });
            if values > budget {
                return Self {
                    allow: false,
                    roaring: Some(mask),
                    fragments: BTreeMap::new(),
                };
            }
        }
        Self::new(&mask)
    }

    #[cfg(test)]
    pub(crate) fn original_mask(&self) -> Option<&Arc<RowAddrMask>> {
        self.roaring.as_ref()
    }

    fn new(mask: &RowAddrMask) -> Self {
        let mut mask = mask.clone();
        let (rows, allow) = match &mut mask {
            RowAddrMask::AllowList(rows) => (rows, true),
            RowAddrMask::BlockList(rows) => (rows, false),
        };
        rows.optimize_for_lookup();
        let mut extra_dense_budget = EXTRA_DENSE_BUDGET;
        Self {
            allow,
            roaring: None,
            fragments: rows
                .iter()
                .map(|(&fragment, selection)| {
                    (
                        fragment,
                        FragmentSelection::prepare(selection, &mut extra_dense_budget),
                    )
                })
                .collect(),
        }
    }

    fn selection<'a>(
        &self,
        rows: impl Iterator<Item = &'a u64>,
        reject_tombstone: bool,
    ) -> impl Iterator<Item = bool> {
        let mut fragment = None;
        let mut selection = None;
        rows.map(move |&row| {
            if reject_tombstone && row == RowAddress::TOMBSTONE_ROW {
                return false;
            }
            let address = RowAddress::new_from_u64(row);
            let fragment_id = address.fragment_id();
            if fragment != Some(fragment_id) {
                fragment = Some(fragment_id);
                selection = self.fragments.get(&fragment_id);
            }
            selection.is_some_and(|selection| selection.contains(address.row_offset()))
                == self.allow
        })
    }

    pub(crate) fn acceptance<'a>(&self, rows: impl Iterator<Item = &'a u64>) -> Vec<bool> {
        if let Some(mask) = &self.roaring {
            return rows
                .map(|&row| row != RowAddress::TOMBSTONE_ROW && mask.selected(row))
                .collect();
        }
        self.selection(rows, true).collect()
    }

    pub(crate) fn selected_indices<'a>(&self, rows: impl Iterator<Item = &'a u64>) -> Vec<u64> {
        if let Some(mask) = &self.roaring {
            return mask.selected_indices(rows);
        }
        self.selection(rows, false)
            .enumerate()
            .filter_map(|(offset, selected)| selected.then_some(offset as u64))
            .collect()
    }

    pub(crate) fn selected_offsets_into(&self, rows: &[u64], out: &mut Vec<u32>) {
        if let Some(mask) = &self.roaring {
            mask.selected_offsets_into(rows.iter(), out);
            return;
        }
        out.clear();
        let mut start = 0;
        while start < rows.len() {
            let fragment = RowAddress::new_from_u64(rows[start]).fragment_id();
            let end = start
                + 1
                + rows[start + 1..]
                    .iter()
                    .position(|&row| RowAddress::new_from_u64(row).fragment_id() != fragment)
                    .unwrap_or(rows.len() - start - 1);
            let run = &rows[start..end];
            match self.fragments.get(&fragment) {
                None => {
                    if !self.allow {
                        out.extend((start..end).map(|offset| offset as u32));
                    }
                }
                Some(FragmentSelection::Full) => {
                    if self.allow {
                        out.extend((start..end).map(|offset| offset as u32));
                    }
                }
                Some(FragmentSelection::Range { min, max }) => {
                    self.extend_offsets(run, start, out, |row| (*min..=*max).contains(&row));
                }
                Some(FragmentSelection::Sparse(bitmap)) => {
                    self.extend_offsets(run, start, out, |row| bitmap.contains(row));
                }
                Some(FragmentSelection::Dense { base, words }) => {
                    self.extend_offsets(run, start, out, |row| {
                        row.checked_sub(*base)
                            .and_then(|offset| words.get(offset as usize / 64))
                            .is_some_and(|word| word & (1u64 << (row % 64)) != 0)
                    });
                }
            }
            start = end;
        }
    }

    fn extend_offsets(
        &self,
        run: &[u64],
        start: usize,
        out: &mut Vec<u32>,
        contains: impl Fn(u32) -> bool,
    ) {
        if self.allow {
            for (offset, &row) in run.iter().enumerate() {
                if contains(RowAddress::new_from_u64(row).row_offset()) {
                    out.push((start + offset) as u32);
                }
            }
        } else {
            for (offset, &row) in run.iter().enumerate() {
                if !contains(RowAddress::new_from_u64(row).row_offset()) {
                    out.push((start + offset) as u32);
                }
            }
        }
    }
}

#[cfg(test)]
pub mod tests;
