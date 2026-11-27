// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

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
pub fn broad_mask_and_probes() -> (Arc<RowAddrMask>, Vec<Vec<u64>>) {
    let mut bitmap = RoaringBitmap::from_iter((0..1_000_000).filter(|row| row % 64 < 32));
    bitmap.optimize();
    let mut rows = RowAddrTreeMap::new();
    for fragment in 0..128 {
        rows.insert_bitmap(fragment, bitmap.clone());
    }
    let probes = (0..16_u64)
        .map(|partition| {
            let mut rows = (0..70_000_u64)
                .map(|row| {
                    let sample = ((row * 16 + partition) * 911) % 128_000_000;
                    ((sample / 1_000_000) << 32) | (sample % 1_000_000)
                })
                .collect::<Vec<_>>();
            rows.sort_unstable();
            rows
        })
        .collect();
    (Arc::new(RowAddrMask::from_allowed(rows)), probes)
}

#[test]
fn prepared_mask_keeps_large_domains_in_roaring() {
    let (allow, probes) = broad_mask_and_probes();
    let RowAddrMask::AllowList(rows) = allow.as_ref() else {
        unreachable!()
    };
    for mask in [
        allow.clone(),
        Arc::new(RowAddrMask::from_block(rows.clone())),
    ] {
        let lookup = PreparedRowAddrMask::for_probe_rows(mask.clone(), probes[0].len());
        assert!(Arc::ptr_eq(lookup.original_mask().unwrap(), &mask));
        assert!(lookup.fragments.is_empty());
        let mut offsets = Vec::new();
        for probe in &probes {
            lookup.selected_offsets_into(probe, &mut offsets);
            assert_eq!(
                offsets
                    .iter()
                    .map(|&row| u64::from(row))
                    .collect::<Vec<_>>(),
                mask.selected_indices(probe.iter())
            );
            assert_eq!(
                lookup.selected_indices(probe.iter()),
                mask.selected_indices(probe.iter())
            );
            assert_eq!(
                lookup.acceptance(probe.iter()),
                probe
                    .iter()
                    .map(|&row| row != RowAddress::TOMBSTONE_ROW && mask.selected(row))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(lookup.acceptance([u64::MAX].iter()), vec![false]);
    }
}
