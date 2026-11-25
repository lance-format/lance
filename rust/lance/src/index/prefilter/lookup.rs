// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;

use lance_core::utils::address::RowAddress;
use lance_select::{RowAddrMask, RowAddrSelection};
use roaring::RoaringBitmap;

const MAX_DENSE_EXPANSION: u64 = 16;

pub(crate) struct PreparedRowAddrMask {
    allow: bool,
    fragments: BTreeMap<u32, FragmentSelection>,
}

enum FragmentSelection {
    Full,
    Sparse(RoaringBitmap),
    Dense { base: u32, words: Vec<u64> },
}

impl FragmentSelection {
    fn prepare(selection: &RowAddrSelection) -> Self {
        let bitmap = match selection {
            RowAddrSelection::Full => return Self::Full,
            RowAddrSelection::Partial(bitmap) if bitmap.is_full() => return Self::Full,
            RowAddrSelection::Partial(bitmap) => bitmap,
        };
        let Some((min, max)) = bitmap.min().zip(bitmap.max()) else {
            return Self::Sparse(bitmap.clone());
        };
        let base = min / 64 * 64;
        let num_words = (u64::from(max) - u64::from(base)) / 64 + 1;
        if num_words * 8 > bitmap.serialized_size() as u64 * MAX_DENSE_EXPANSION {
            return Self::Sparse(bitmap.clone());
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
        Self {
            allow,
            fragments: rows
                .iter()
                .map(|(&fragment, selection)| (fragment, FragmentSelection::prepare(selection)))
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
}

#[cfg(test)]
mod tests {
    use lance_core::utils::address::RowAddress;
    use lance_select::{RowAddrMask, RowAddrTreeMap};
    use roaring::RoaringBitmap;

    use super::{FragmentSelection, PreparedRowAddrMask};

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
            Some(FragmentSelection::Sparse(_))
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
        }
    }
}
