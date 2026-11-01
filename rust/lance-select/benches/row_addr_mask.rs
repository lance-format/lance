// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmarks for `RowAddrMask` / `RowAddrTreeMap`.
//!
//! These benchmarks are deliberately structured to expose the row-cardinality
//! scaling weakness of the current per-row bitmap representation. Producers
//! (e.g. scalar-index `search` implementations) and consumers (e.g.
//! `mask_to_offset_ranges`) are frequently range-shaped, but every operation
//! must round-trip through `Partial(RoaringBitmap)` and therefore costs O(N)
//! in the number of rows, not O(R) in the number of distinct ranges.
//!
//! Each benchmark varies the number of rows while keeping the number of
//! ranges fixed at 1. A range-aware representation should make these
//! near-constant time; today they are linear in N.
//! The `owned_union` group separately measures copying versus moving bitmap
//! storage when merging maps with disjoint or identical fragment IDs.
//!
//! Run with `cargo bench -p lance-select --bench row_addr_mask`.

use std::ops::Range;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance_select::{NullableRowAddrMask, NullableRowAddrSet, RowAddrMask, RowAddrTreeMap};

/// Row counts we sweep across. Chosen to cover the realistic range of
/// matches a zonemap produces for an `IS NULL`-like predicate on a single
/// fragment: a few thousand rows up through tens of millions.
const ROW_COUNTS: &[u64] = &[10_000, 100_000, 1_000_000, 10_000_000];

fn make_range_mask(num_rows: u64) -> RowAddrTreeMap {
    // Build a mask covering a single contiguous run in fragment 0.
    // This is the exact shape a scalar-index search produces when it
    // determines a contiguous chunk of zones matches.
    let mut map = RowAddrTreeMap::new();
    map.insert_range(0..num_rows);
    map
}

/// Producer cost: building a mask from one contiguous Range.
///
/// Today this is O(N) — every bit gets inserted into a roaring bitmap.
/// With a range-aware representation it would be O(1) (push a single run).
fn bench_insert_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_range_single_run");
    for &n in ROW_COUNTS {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut map = RowAddrTreeMap::new();
                map.insert_range(0..n);
                std::hint::black_box(map);
            });
        });
    }
    group.finish();
}

/// Consumer cost: iterating every row address in a dense mask.
///
/// `into_addr_iter` walks set bits one at a time. For a contiguous run
/// of N rows this is O(N) — even though the rows are trivially
/// representable as a single Range. This is what `mask_to_offset_ranges`
/// does after intersecting with a source segment: it pays per-row
/// iteration cost only to immediately collapse the addresses back into
/// ranges via `GroupingIterator`.
fn bench_iter_addrs(c: &mut Criterion) {
    let mut group = c.benchmark_group("into_addr_iter_single_run");
    for &n in ROW_COUNTS {
        let map = make_range_mask(n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let count: u64 = map.clone().into_addr_iter().count() as u64;
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

/// Best-achievable iteration over the same data.
///
/// `Iter::next_range` walks the bitmap's run containers in O(num_runs).
/// For a single contiguous run this should be ~constant time — the
/// public `RowAddrMask` API gives no way to surface that today, so the
/// performance is currently inaccessible to callers. Comparing this to
/// `into_addr_iter_single_run` quantifies the speedup a range-aware
/// representation could deliver to consumers.
fn bench_iter_runs(c: &mut Criterion) {
    let mut group = c.benchmark_group("next_range_iter_single_run");
    for &n in ROW_COUNTS {
        // Use the same underlying roaring bitmap shape that `make_range_mask`
        // produces internally (one fragment, one contiguous run).
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert_range(0..(n as u32));
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut iter = bitmap.iter();
                let mut runs: u64 = 0;
                while iter.next_range().is_some() {
                    runs += 1;
                }
                std::hint::black_box(runs);
            });
        });
    }
    group.finish();
}

/// Range-aware consumer cost via the public `RowAddrTreeMap::iter_runs`
/// API. The map is built the ordinary way (`insert_range` → Partial
/// bitmap); `iter_runs` walks the bitmap's run containers via
/// `Iter::next_range`. Compare against `into_addr_iter_single_run` to see
/// the consumer-side speedup callers get without changing the underlying
/// representation.
fn bench_iter_runs_partial(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter_runs_partial_single_run");
    for &n in ROW_COUNTS {
        let map = make_range_mask(n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut runs: u64 = 0;
                for _ in map.iter_runs() {
                    runs += 1;
                }
                std::hint::black_box(runs);
            });
        });
    }
    group.finish();
}

/// Set intersection of two range-shaped masks.
///
/// Both inputs are single contiguous runs that overlap in their middle
/// half (so the output is itself a single contiguous run). With per-row
/// bitmaps this is O(N) — the entire bitmap participates in the AND.
/// With ranges it would be O(1).
fn bench_intersect_ranges(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_two_runs");
    for &n in ROW_COUNTS {
        let lhs = make_range_mask(n);
        let rhs_range = (n / 4)..(3 * n / 4);
        let mut rhs = RowAddrTreeMap::new();
        rhs.insert_range(rhs_range);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut tmp = lhs.clone();
                tmp &= &rhs;
                std::hint::black_box(tmp);
            });
        });
    }
    group.finish();
}

/// Full round trip: build a source range bitmap, AND with a mask, iterate
/// each surviving bit. This is the exact slow path of
/// `mask_to_offset_ranges` in `lance-table/src/rowids.rs:387`. Profiling
/// a 10M-row zonemap `IS NULL` query showed this consuming ~55% of the
/// hot-loop time (~495 ms of 889 ms). The benchmark separates the
/// per-row producer/consumer cost from the rest of the scan pipeline so
/// it can be tracked in isolation.
fn bench_range_to_ranges_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("mask_to_offset_ranges_inner_loop");
    for &n in ROW_COUNTS {
        // The mask selects the back half of a 2N-row fragment.
        let mask_range = n..(2 * n);
        let mask = RowAddrMask::AllowList(RowAddrTreeMap::from(mask_range));
        // The source segment covers the whole fragment.
        let src: Range<u64> = 0..(2 * n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                // Mimic the slow path: materialize source range, AND with mask,
                // iterate to count survivors (a stand-in for whatever the
                // consumer actually does — e.g. GroupingIterator).
                let mut ids = RowAddrTreeMap::from(src.clone());
                ids.mask(&mask);
                let count = ids.into_addr_iter().count();
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

/// Same end-to-end shape as `mask_to_offset_ranges_inner_loop`, but the
/// final per-bit walk is replaced by `iter_runs`. Quantifies the speedup
/// the consumer side gets purely from switching iteration APIs — no
/// representation change.
fn bench_range_to_ranges_round_trip_runs(c: &mut Criterion) {
    let mut group = c.benchmark_group("mask_to_offset_ranges_inner_loop_runs");
    for &n in ROW_COUNTS {
        let mask_range = n..(2 * n);
        let mask = RowAddrMask::AllowList(RowAddrTreeMap::from(mask_range));
        let src: Range<u64> = 0..(2 * n);
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut ids = RowAddrTreeMap::from(src.clone());
                ids.mask(&mask);
                let count: u64 = ids
                    .iter_runs()
                    .map(|(_, r)| (*r.end() as u64) - (*r.start() as u64) + 1)
                    .sum();
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

/// Many small runs vs one big run with the same total cardinality.
///
/// A range-aware representation should be O(num_runs), so the
/// `single_run` case should be ~K times faster than the `K_runs` case.
/// Today they are essentially equal: the cost is dictated by the number
/// of rows, not the number of runs.
fn bench_runs_vs_rows(c: &mut Criterion) {
    let total_rows: u64 = 1_000_000;
    let mut group = c.benchmark_group("insert_runs_constant_cardinality");

    group.throughput(Throughput::Elements(total_rows));
    group.bench_function("single_run_1M", |b| {
        b.iter(|| {
            let mut map = RowAddrTreeMap::new();
            map.insert_range(0..total_rows);
            std::hint::black_box(map);
        });
    });

    for k in [10u64, 100, 1_000, 10_000] {
        let run_size = total_rows / k;
        // Stride between runs is 2 * run_size so the bitmap is half full.
        let stride = run_size * 2;
        group.bench_function(format!("{k}_runs_1M_total"), |b| {
            b.iter(|| {
                let mut map = RowAddrTreeMap::new();
                for i in 0..k {
                    let start = i * stride;
                    map.insert_range(start..(start + run_size));
                }
                std::hint::black_box(map);
            });
        });
    }
    group.finish();
}

const UNION_INPUT_COUNT: usize = 4;
const UNION_ROWS_PER_FRAGMENT: u32 = 1 << 20;
const UNION_FRAGMENT_COUNTS: &[u32] = &[32, 128];

#[derive(Clone, Copy)]
enum UnionTopology {
    Disjoint,
    Identical,
}

impl UnionTopology {
    fn name(self) -> &'static str {
        match self {
            Self::Disjoint => "disjoint",
            Self::Identical => "identical_fragment_ids",
        }
    }
}

struct UnionInputs {
    accumulator: RowAddrTreeMap,
    remaining: Vec<RowAddrTreeMap>,
}

struct NullableUnionInputs {
    accumulator: NullableRowAddrMask,
    remaining: Vec<NullableRowAddrMask>,
}

/// Build one 50%-dense bitmap, then clone it into each fragment. With 1M
/// offsets, this is 512K selected rows in 16 bitset containers (about 128 KiB
/// of bitmap payload per fragment). The four input maps therefore hold about
/// 16/64 MiB of bitmap payload at 32/128 fragments per map. This intentionally
/// uses alternating offsets instead of ranges, which Roaring would compress
/// into run containers.
fn make_union_maps(
    representative_bitmap: &roaring::RoaringBitmap,
    fragment_count: u32,
    topology: UnionTopology,
) -> Vec<RowAddrTreeMap> {
    (0..UNION_INPUT_COUNT)
        .map(|input_index| {
            let first_fragment = match topology {
                UnionTopology::Disjoint => input_index as u32 * fragment_count,
                UnionTopology::Identical => 0,
            };
            let mut map = RowAddrTreeMap::new();
            for fragment_offset in 0..fragment_count {
                map.insert_bitmap(
                    first_fragment + fragment_offset,
                    representative_bitmap.clone(),
                );
            }
            map
        })
        .collect()
}

/// Clone and split a fixture during `iter_batched` setup, outside the timed
/// routine. Every union method receives the same four-map shape and contents.
fn clone_union_inputs(maps: &[RowAddrTreeMap]) -> UnionInputs {
    UnionInputs {
        accumulator: maps[0].clone(),
        remaining: maps[1..].to_vec(),
    }
}

fn clone_nullable_union_inputs(maps: &[RowAddrTreeMap]) -> NullableUnionInputs {
    let as_allow_list = |map: &RowAddrTreeMap| {
        NullableRowAddrMask::AllowList(NullableRowAddrSet::new(map.clone(), RowAddrTreeMap::new()))
    };
    NullableUnionInputs {
        accumulator: as_allow_list(&maps[0]),
        remaining: maps[1..].iter().map(as_allow_list).collect(),
    }
}

/// Compare owned `|=` folding with borrowed `|=` and `Extend<Self>` controls.
/// The fixture holds four maps, each with 32 or 128 fragments. Disjoint cases
/// exercise insertion of new fragment bitmaps; identical cases merge the same
/// fragment IDs and bitmap contents repeatedly. `iter_batched` clones inputs
/// in setup (`PerIteration`), outside the timed routine. Results are returned
/// through `black_box` so result destruction is outside the timed routine; the
/// borrowed control also returns its source maps to keep their destruction out
/// of the timed routine.
fn bench_owned_union(c: &mut Criterion) {
    let mut representative_bitmap = roaring::RoaringBitmap::new();
    for row_offset in (0..UNION_ROWS_PER_FRAGMENT).step_by(2) {
        representative_bitmap.insert(row_offset);
    }

    let mut group = c.benchmark_group("owned_union");
    for &fragment_count in UNION_FRAGMENT_COUNTS {
        for topology in [UnionTopology::Disjoint, UnionTopology::Identical] {
            let maps = make_union_maps(&representative_bitmap, fragment_count, topology);
            let case = format!("{}_{}fragments_per_input", topology.name(), fragment_count);

            group.bench_function(BenchmarkId::new("owned_bitor_assign", &case), |b| {
                b.iter_batched(
                    || clone_union_inputs(&maps),
                    |inputs| {
                        let UnionInputs {
                            mut accumulator,
                            remaining,
                        } = inputs;
                        for rhs in remaining {
                            accumulator |= rhs;
                        }
                        std::hint::black_box(accumulator)
                    },
                    BatchSize::PerIteration,
                );
            });

            group.bench_function(BenchmarkId::new("borrowed_bitor_assign", &case), |b| {
                b.iter_batched(
                    || clone_union_inputs(&maps),
                    |inputs| {
                        let UnionInputs {
                            mut accumulator,
                            remaining,
                        } = inputs;
                        for rhs in &remaining {
                            accumulator |= rhs;
                        }
                        std::hint::black_box((accumulator, remaining))
                    },
                    BatchSize::PerIteration,
                );
            });

            group.bench_function(BenchmarkId::new("extend_self", &case), |b| {
                b.iter_batched(
                    || clone_union_inputs(&maps),
                    |inputs| {
                        let UnionInputs {
                            mut accumulator,
                            remaining,
                        } = inputs;
                        accumulator.extend(remaining);
                        std::hint::black_box(accumulator)
                    },
                    BatchSize::PerIteration,
                );
            });

            if matches!(topology, UnionTopology::Disjoint) {
                group.bench_function(BenchmarkId::new("nullable_allow_list_bitor", &case), |b| {
                    b.iter_batched(
                        || clone_nullable_union_inputs(&maps),
                        |inputs| {
                            let NullableUnionInputs {
                                mut accumulator,
                                remaining,
                            } = inputs;
                            for rhs in remaining {
                                accumulator = accumulator | rhs;
                            }
                            std::hint::black_box(accumulator)
                        },
                        BatchSize::PerIteration,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_range,
    bench_iter_addrs,
    bench_iter_runs,
    bench_iter_runs_partial,
    bench_intersect_ranges,
    bench_range_to_ranges_round_trip,
    bench_range_to_ranges_round_trip_runs,
    bench_runs_vs_rows,
    bench_owned_union,
);
criterion_main!(benches);
