// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CPU-only equality benchmarks. Fixture construction is outside measurement.
//! Allocation counters report requested bytes during one equality call, including
//! temporary storage; they do not measure copied bytes or process RSS.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    io::Write,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_select::{NullableRowAddrSet, RowAddrTreeMap};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: Every allocation operation forwards the original layout and pointer
// to System; the counters only observe requested allocation sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

fn fixture(
    fragments: u32,
    stride: usize,
    with_nulls: bool,
) -> (NullableRowAddrSet, NullableRowAddrSet) {
    let mut selected = RowAddrTreeMap::new();
    let mut nulls = RowAddrTreeMap::new();
    for fragment in 0..fragments {
        selected.insert_bitmap(fragment, (0..1_u32 << 20).step_by(stride).collect());
        if with_nulls {
            nulls.insert_bitmap(fragment, (0..1_u32 << 20).step_by(stride * 4).collect());
        }
    }
    let left = NullableRowAddrSet::new(selected.clone(), nulls.clone());
    let right = NullableRowAddrSet::new(selected - &nulls, nulls);
    (left, right)
}

fn bench_nullable_equality(c: &mut Criterion) {
    let mut group = c.benchmark_group("nullable_equality");
    for fragments in [16, 128] {
        for (density, stride) in [("dense", 2), ("sparse", 4096)] {
            for with_nulls in [false, true] {
                let (left, right) = fixture(fragments, stride, with_nulls);
                let label = format!(
                    "{density}_{}",
                    if with_nulls { "nulls" } else { "no_nulls" }
                );
                ALLOCATIONS.store(0, Ordering::Relaxed);
                ALLOCATED_BYTES.store(0, Ordering::Relaxed);
                COUNTING.store(true, Ordering::Relaxed);
                let equal = black_box(&left) == black_box(&right);
                COUNTING.store(false, Ordering::Relaxed);
                assert!(equal);
                let record = format!(
                    "allocation,{label},{fragments},{},{}\n",
                    ALLOCATIONS.load(Ordering::Relaxed),
                    ALLOCATED_BYTES.load(Ordering::Relaxed)
                );
                std::io::stdout().write_all(record.as_bytes()).unwrap();
                group.bench_with_input(BenchmarkId::new(label, fragments), &fragments, |b, _| {
                    b.iter(|| black_box(black_box(&left) == black_box(&right)));
                });
            }
        }
    }
    // Unequal NULLs should reject without traversing/cloning selected rows.
    let (left, _) = fixture(128, 2, false);
    let right = left.clone().with_nulls(RowAddrTreeMap::from_iter([0]));
    group.bench_function("unequal_nulls", |b| {
        b.iter(|| black_box(black_box(&left) == black_box(&right)))
    });
    group.finish();
}

criterion_group!(benches, bench_nullable_equality);
criterion_main!(benches);
