// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The memory a batch remap over a long chain holds while it runs, measured
//! through the global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::{GroupInputWithLayout, RowAddrRemap};
use roaring::RoaringTreemap;
use rstest::rstest;

struct MeasuringAllocator;

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static LIVE_BYTES: Cell<isize> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<isize> = const { Cell::new(0) };
}

fn record(bytes: isize) {
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        let live = LIVE_BYTES.get() + bytes;
        LIVE_BYTES.set(live);
        PEAK_BYTES.set(PEAK_BYTES.get().max(live));
    }
}

unsafe impl GlobalAlloc for MeasuringAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if new_ptr == ptr {
            record(new_size as isize - layout.size() as isize);
        } else if !new_ptr.is_null() {
            // A block that moves is held twice while it is copied.
            record(new_size as isize);
            record(-(layout.size() as isize));
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: MeasuringAllocator = MeasuringAllocator;

// Keep in sync with REMAP_BATCH_ROWS.
const PASS_ROWS: usize = 1 << 16;

/// The most a batch remap may hold at once while tracking `rows` rows that pass
/// through `fragments` fragments. Per row: a list slot with up to 4x spare
/// (16 B), a moved-row buffer slot with up to 2x spare (8 B), and 4 B while a
/// list is copied to grow. Per fragment: a map entry, with the table's spare
/// slots and its old table while it grows, and a list's last spare slots (240 B).
fn scratch_bound(rows: usize, fragments: usize) -> usize {
    28 * rows + 240 * fragments + 1024
}

fn remap_measuring_peak(remap: &RowAddrRemap, row_addrs: &mut [Option<u64>]) -> usize {
    LIVE_BYTES.set(0);
    PEAK_BYTES.set(0);
    MEASURING.set(true);
    remap.remap_in_place(row_addrs);
    MEASURING.set(false);
    assert_eq!(LIVE_BYTES.get(), 0, "the remap kept memory after returning");
    PEAK_BYTES.get() as usize
}

fn addr(fragment: u32, offset: u32) -> u64 {
    RowAddress::new_from_parts(fragment, offset).into()
}

#[derive(Default)]
struct Chain {
    steps: Vec<RowAddrRemap>,
    maps: Vec<HashMap<u64, Option<u64>>>,
}

impl Chain {
    fn direct(&mut self, map: HashMap<u64, Option<u64>>) {
        self.steps.push(RowAddrRemap::direct(map.clone()));
        self.maps.push(map);
    }

    /// Offsets in `rows..` are outside the layout and remain unaffected.
    fn compact(&mut self, fragment: u32, rows: u32, new_fragment: u32) {
        let mut rewritten = RoaringTreemap::new();
        rewritten.insert_range(addr(fragment, 0)..addr(fragment, rows));
        self.steps.push(
            RowAddrRemap::compact_with_layout([GroupInputWithLayout {
                rewritten_old_row_addrs: rewritten,
                old_frags: vec![(fragment, rows)],
                new_frags: vec![(new_fragment, rows)],
            }])
            .unwrap(),
        );
        self.maps.push(
            (0..rows)
                .map(|offset| (addr(fragment, offset), Some(addr(new_fragment, offset))))
                .collect(),
        );
    }

    fn check(self, mut batch: Vec<Option<u64>>) {
        let mut fragments = HashSet::new();
        let expected = batch
            .iter()
            .map(|&row_addr| {
                let mut current = row_addr?;
                fragments.insert(current >> 32);
                for map in &self.maps {
                    match map.get(&current) {
                        None => {}
                        Some(None) => return None,
                        Some(&Some(mapped)) => {
                            current = mapped;
                            fragments.insert(current >> 32);
                        }
                    }
                }
                Some(current)
            })
            .collect::<Vec<_>>();
        let chain = RowAddrRemap::chained(self.steps);
        for (row_addr, answer) in batch.iter().zip(&expected) {
            if let Some(old_addr) = *row_addr {
                assert_eq!(
                    chain.get(old_addr).unwrap_or(Some(old_addr)),
                    *answer,
                    "address {old_addr:#x}"
                );
            }
        }
        let peak = remap_measuring_peak(&chain, &mut batch);
        assert_eq!(batch, expected);
        let bound = scratch_bound(batch.len().min(PASS_ROWS), fragments.len());
        assert!(
            peak <= bound,
            "held {peak} B, over the {bound} B bound for {} rows in {} fragments",
            batch.len(),
            fragments.len()
        );
    }
}

/// Each step moves all but one row of the fragment it names to a fresh
/// fragment, so the batch spreads over one fragment more per step, each holding
/// the row left behind.
#[rstest]
fn split_scratch_is_bounded_by_the_batch(
    #[values(false, true)] mixed: bool,
    #[values(32, 128, 511)] steps: u32,
) {
    const ROWS: u32 = 512;
    let mut chain = Chain::default();
    for step in 0..steps {
        // Offset `moving` stays: outside a compact step's layout, or missing
        // from a direct step's partial map.
        let moving = ROWS - 1 - step;
        if mixed && step % 2 == 0 {
            chain.compact(step, moving, step + 1);
        } else {
            chain.direct(
                (0..moving)
                    .map(|offset| (addr(step, offset), Some(addr(step + 1, offset))))
                    .collect(),
            );
        }
    }
    chain.check((0..ROWS).map(|offset| Some(addr(0, offset))).collect());
}

/// One row in each of 4,096 fragments, so the map of lists outweighs the lists.
#[test]
fn many_fragment_scratch_is_bounded_by_the_batch() {
    const FRAGMENTS: u32 = 4_096;
    let mut chain = Chain::default();
    for step in 0..64 {
        if step % 8 == 7 {
            chain.compact(3_000 + step, 1, FRAGMENTS + step);
            continue;
        }
        chain.direct(HashMap::from([
            (addr(step, 0), Some(addr(FRAGMENTS + step, 0))),
            (addr(1_000 + step, 0), None),
            (addr(2_000 + step, 0), Some(addr(2_000 + step, 0))),
            // Named, but not in the batch.
            (addr(3_000 + step, 1), None),
        ]));
    }
    chain.check(
        (0..FRAGMENTS)
            .map(|fragment| (fragment % 100 != 99).then(|| addr(fragment, 0)))
            .collect(),
    );
}

/// Nine families of 8,192 rows split as above, one compaction group each,
/// across two passes, with deleted rows and rows no step names.
#[test]
fn split_scratch_across_passes_is_bounded_by_a_pass() {
    const FAMILIES: u32 = 9;
    const FAMILY_ROWS: u32 = 8_192;
    const STEPS: u32 = 32;
    const SPLITS: u32 = STEPS / 2;
    let fragment = |family: u32, splits: u32| family * 1_000 + splits;
    let chain = RowAddrRemap::chained((0..STEPS).map(|step| {
        let splits = step / 2;
        let moving = FAMILY_ROWS - 1 - splits;
        RowAddrRemap::compact_with_layout((step % 2..FAMILIES).step_by(2).map(|family| {
            let mut rewritten = RoaringTreemap::new();
            rewritten.insert_range(
                addr(fragment(family, splits), 0)..addr(fragment(family, splits), moving),
            );
            GroupInputWithLayout {
                rewritten_old_row_addrs: rewritten,
                old_frags: vec![(fragment(family, splits), moving)],
                new_frags: vec![(fragment(family, splits + 1), moving)],
            }
        }))
        .unwrap()
    }));

    let mut batch = (0..FAMILIES)
        .flat_map(|family| {
            (0..FAMILY_ROWS).map(move |offset| {
                (offset % 1_000 != 999).then(|| addr(fragment(family, 0), offset))
            })
        })
        .chain((0..2_000).map(|offset| Some(addr(999_999, offset))))
        .collect::<Vec<_>>();
    assert!(batch.len() > PASS_ROWS);
    let expected = batch
        .iter()
        .map(|row_addr| {
            let old_addr = (*row_addr)?;
            let (old_fragment, offset) = ((old_addr >> 32) as u32, old_addr as u32);
            if old_fragment == 999_999 {
                return Some(old_addr);
            }
            let splits = (FAMILY_ROWS - 1 - offset).min(SPLITS);
            Some(addr(fragment(old_fragment / 1_000, splits), offset))
        })
        .collect::<Vec<_>>();

    let peak = remap_measuring_peak(&chain, &mut batch);
    assert_eq!(batch, expected);
    let bound = scratch_bound(PASS_ROWS, (FAMILIES * (SPLITS + 1) + 1) as usize);
    assert!(peak <= bound, "held {peak} B, over the {bound} B bound");
}
