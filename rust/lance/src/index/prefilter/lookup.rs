// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;

use lance_core::utils::address::RowAddress;
use lance_select::{RowAddrMask, RowAddrSelection};
use roaring::RoaringBitmap;

const MAX_DENSE_EXPANSION: u64 = 16;
const MAX_SMALL_DENSE_BYTES: u64 = 128 * 1024;
const EXTRA_DENSE_BUDGET: u64 = 4 * 1024 * 1024;

pub(crate) struct PreparedRowAddrMask {
    allow: bool,
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
                || bitmap.len() > (u64::from(max) - u64::from(min) + 1) / 2
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
    pub(crate) fn new(mask: &RowAddrMask) -> Self {
        let mut mask = mask.clone();
        let (rows, allow) = match &mut mask {
            RowAddrMask::AllowList(rows) => (rows, true),
            RowAddrMask::BlockList(rows) => (rows, false),
        };
        rows.optimize_for_lookup();
        let mut extra_dense_budget = EXTRA_DENSE_BUDGET;
        Self {
            allow,
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
        self.selection(rows, true).collect()
    }

    pub(crate) fn selected_indices<'a>(&self, rows: impl Iterator<Item = &'a u64>) -> Vec<u64> {
        self.selection(rows, false)
            .enumerate()
            .filter_map(|(offset, selected)| selected.then_some(offset as u64))
            .collect()
    }

    pub(crate) fn selected_offsets_into(&self, rows: &[u64], out: &mut Vec<u32>) {
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
mod tests {
    use lance_core::utils::address::RowAddress;
    use lance_select::{RowAddrMask, RowAddrTreeMap};
    use roaring::RoaringBitmap;

    use super::{EXTRA_DENSE_BUDGET, FragmentSelection, PreparedRowAddrMask};

    #[test]
    fn prepared_mask_preserves_compact_dense_ranges() {
        let mut range = RoaringBitmap::new();
        range.insert_range(0..1_000_000);
        range.optimize();
        let mut holes = range.clone();
        holes.remove(1);
        holes.remove(999998);
        holes.optimize();
        let mut rows = RowAddrTreeMap::new();
        rows.insert_bitmap(5, range);
        rows.insert_bitmap(6, holes);
        let input = (5..=6)
            .flat_map(|fragment| {
                [0, 1, 2, 999998, 999999, 1000000]
                    .map(move |row| u64::from(RowAddress::new_from_parts(fragment, row)))
            })
            .collect::<Vec<_>>();
        for mask in [
            RowAddrMask::AllowList(rows.clone()),
            RowAddrMask::BlockList(rows),
        ] {
            let prepared = PreparedRowAddrMask::new(&mask);
            assert!(prepared.fragments.values().all(|selection| matches!(
                selection,
                FragmentSelection::Sparse(_) | FragmentSelection::Range { .. }
            )));
            assert_eq!(
                prepared.selected_indices(input.iter()),
                mask.selected_indices(input.iter())
            );
            let mut offsets = vec![u32::MAX];
            prepared.selected_offsets_into(&input, &mut offsets);
            assert_eq!(
                offsets
                    .iter()
                    .map(|&offset| u64::from(offset))
                    .collect::<Vec<_>>(),
                mask.selected_indices(input.iter())
            );
            prepared.selected_offsets_into(&[], &mut offsets);
            assert!(offsets.is_empty());
        }
    }

    #[test]
    fn prepared_mask_bounds_extra_dense_allocations() {
        let mut rows = RowAddrTreeMap::new();
        for fragment in 0..40 {
            rows.insert_bitmap(fragment, RoaringBitmap::from_iter([0, 999999]));
        }
        let input = (0..40)
            .flat_map(|fragment| {
                [0, 1, 999998, 999999, 1000000]
                    .map(move |row| u64::from(RowAddress::new_from_parts(fragment, row)))
            })
            .collect::<Vec<_>>();
        for mask in [
            RowAddrMask::AllowList(rows.clone()),
            RowAddrMask::BlockList(rows),
        ] {
            let prepared = PreparedRowAddrMask::new(&mask);
            let dense_bytes: usize = prepared
                .fragments
                .values()
                .map(|selection| match selection {
                    FragmentSelection::Dense { words, .. } => words.len() * 8,
                    _ => 0,
                })
                .sum();
            assert!(dense_bytes > 0 && dense_bytes as u64 <= EXTRA_DENSE_BUDGET);
            assert!(
                prepared
                    .fragments
                    .values()
                    .any(|selection| matches!(selection, FragmentSelection::Sparse(_)))
            );
            assert!(matches!(
                prepared.fragments.get(&0),
                Some(FragmentSelection::Dense { .. })
            ));
            assert_eq!(
                prepared.selected_indices(input.iter()),
                mask.selected_indices(input.iter())
            );
            let mut offsets = vec![u32::MAX];
            prepared.selected_offsets_into(&input, &mut offsets);
            assert_eq!(
                offsets
                    .iter()
                    .map(|&offset| u64::from(offset))
                    .collect::<Vec<_>>(),
                mask.selected_indices(input.iter())
            );
            prepared.selected_offsets_into(&[], &mut offsets);
            assert!(offsets.is_empty());
        }
    }

    #[test]
    fn prepared_mask_preserves_membership() {
        let mut rows = RowAddrTreeMap::new();
        let mut runs = RoaringBitmap::from_iter((0..32768).filter(|row| row % 64 < 32));
        runs.optimize();
        rows.insert_bitmap(7, runs);
        rows.insert_fragment(8);
        rows.insert_bitmap(9, RoaringBitmap::from_iter([0, u32::MAX - 1]));
        rows.insert_bitmap(10, RoaringBitmap::full());
        rows.insert_bitmap(11, RoaringBitmap::from_iter([65, 67, 128, 192]));
        let mut input = (0..33000)
            .map(|row| u64::from(RowAddress::new_from_parts(7, row)))
            .collect::<Vec<_>>();
        input.extend(
            [
                0,
                1,
                63,
                64,
                65,
                66,
                67,
                127,
                128,
                191,
                192,
                193,
                u32::MAX - 1,
                u32::MAX,
            ]
            .into_iter()
            .flat_map(|row| {
                (8..=12).map(move |fragment| u64::from(RowAddress::new_from_parts(fragment, row)))
            }),
        );
        input.extend([input[0], RowAddress::TOMBSTONE_ROW]);
        for mask in [
            RowAddrMask::AllowList(rows.clone()),
            RowAddrMask::BlockList(rows),
        ] {
            let prepared = PreparedRowAddrMask::new(&mask);
            assert!(matches!(
                prepared.fragments.get(&7),
                Some(FragmentSelection::Dense { .. })
            ));
            assert!(matches!(
                prepared.fragments.get(&9),
                Some(FragmentSelection::Sparse(_))
            ));
            assert!(matches!(
                prepared.fragments.get(&10),
                Some(FragmentSelection::Full)
            ));
            let expected = input
                .iter()
                .map(|&row| row != RowAddress::TOMBSTONE_ROW && mask.selected(row))
                .collect::<Vec<_>>();
            assert_eq!(prepared.acceptance(input.iter()), expected);
            assert_eq!(
                prepared.selected_indices(input.iter()),
                mask.selected_indices(input.iter())
            );
            let mut offsets = vec![u32::MAX];
            prepared.selected_offsets_into(&input, &mut offsets);
            assert_eq!(
                offsets
                    .iter()
                    .map(|&offset| u64::from(offset))
                    .collect::<Vec<_>>(),
                mask.selected_indices(input.iter())
            );
            prepared.selected_offsets_into(&[], &mut offsets);
            assert!(offsets.is_empty());
        }
    }

    #[test]
    fn prepared_mask_preserves_empty_masks_and_long_ranges() {
        let mut rows = RowAddrTreeMap::new();
        let mut range = RoaringBitmap::new();
        range.insert_range(0..10_000_000);
        range.optimize();
        rows.insert_bitmap(3, range);
        let prepared = PreparedRowAddrMask::new(&RowAddrMask::AllowList(rows));
        assert!(matches!(
            prepared.fragments.get(&3),
            Some(FragmentSelection::Range { .. })
        ));
        let input = [
            0,
            u64::from(RowAddress::new_from_parts(3, 5)),
            RowAddress::TOMBSTONE_ROW,
        ];
        for mask in [RowAddrMask::all_rows(), RowAddrMask::allow_nothing()] {
            let prepared = PreparedRowAddrMask::new(&mask);
            let expected = input
                .iter()
                .map(|&row| row != RowAddress::TOMBSTONE_ROW && mask.selected(row))
                .collect::<Vec<_>>();
            assert_eq!(prepared.acceptance(input.iter()), expected);
            assert_eq!(
                prepared.selected_indices(input.iter()),
                mask.selected_indices(input.iter())
            );
            let mut offsets = vec![u32::MAX];
            prepared.selected_offsets_into(&input, &mut offsets);
            assert_eq!(
                offsets
                    .iter()
                    .map(|&offset| u64::from(offset))
                    .collect::<Vec<_>>(),
                mask.selected_indices(input.iter())
            );
            prepared.selected_offsets_into(&[], &mut offsets);
            assert!(offsets.is_empty());
        }
    }
}
