// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_select::{
    NullableIndexExprResult, NullableRowAddrMask, NullableRowAddrSet, RowAddrTreeMap,
};
use roaring::RoaringBitmap;

fn fixture(fragments: u32, track_nulls: bool) -> [NullableIndexExprResult; 3] {
    let mut rows = [
        RoaringBitmap::new(),
        RoaringBitmap::new(),
        RoaringBitmap::new(),
    ];
    let mut nulls = RoaringBitmap::new();
    for row in 0..1_000_000u32 {
        let score = row.wrapping_mul(0x9e37_79b9).rotate_left(13) % 101;
        let label = row.wrapping_mul(0x85eb_ca6b).rotate_left(7) % 10;
        let tags = row.wrapping_mul(0xc2b2_ae35).rotate_left(19) % 100;
        if label == 0 {
            rows[0].insert(row);
        }
        if score >= 50 {
            rows[1].insert(row);
        }
        if tags < 43 {
            rows[2].insert(row);
        }
        if track_nulls && tags >= 95 {
            nulls.insert(row);
        }
    }
    std::array::from_fn(|column| {
        let mut selected = RowAddrTreeMap::new();
        let mut null_rows = RowAddrTreeMap::new();
        for fragment in 0..fragments {
            selected.insert_bitmap(fragment, rows[column].clone());
            if column == 2 && track_nulls {
                null_rows.insert_bitmap(fragment, nulls.clone());
            }
        }
        NullableIndexExprResult::exact(NullableRowAddrMask::AllowList(NullableRowAddrSet::new(
            selected, null_rows,
        )))
    })
}

fn bench_combined_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_filter_masks");
    for fragments in [16, 100] {
        for track_nulls in [false, true] {
            let masks = fixture(fragments, track_nulls);
            group.bench_with_input(
                BenchmarkId::new(format!("nulls_{track_nulls}"), fragments),
                &masks,
                |b, masks| {
                    b.iter_batched(
                        || masks.clone(),
                        |[label, score, tags]| black_box(((label & score) & tags).drop_nulls()),
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
    group.finish();

    let mut group = c.benchmark_group("exact_drop_nulls");
    for track_nulls in [false, true] {
        let masks = fixture(100, track_nulls);
        group.bench_with_input(
            BenchmarkId::from_parameter(track_nulls),
            &masks[2],
            |b, mask| {
                b.iter_batched(
                    || mask.clone(),
                    |mask| black_box(mask.drop_nulls()),
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_combined_filter);
criterion_main!(benches);
