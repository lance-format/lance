// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance_linalg::distance::{Cosine, norm_l2::norm_l2};

fn bench_cosine_range(c: &mut Criterion) {
    const DIMENSION: usize = 128;
    const ROWS: usize = 4096;

    let mut query = vec![0.0_f32; DIMENSION];
    query[0] = 1.0;
    let mut batch = vec![0.0; DIMENSION * ROWS];
    for (i, row) in batch.as_chunks_mut::<DIMENSION>().0.iter_mut().enumerate() {
        row[1 + i % (DIMENSION - 1)] = 1.0;
    }
    let mut distances = vec![0.0; ROWS];
    let mut group = c.benchmark_group("cosine_f32_orthogonal");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function("batch", |b| {
        b.iter(|| {
            f32::cosine_batch_into(
                black_box(&query),
                black_box(&batch),
                DIMENSION,
                black_box(&mut distances),
            );
            black_box(&distances);
        });
    });
    group.bench_function("cached_norms", |b| {
        b.iter(|| {
            for (distance, row) in distances.iter_mut().zip(batch.as_chunks::<DIMENSION>().0) {
                *distance = f32::cosine_with_norms(black_box(&query), 1.0, 1.0, black_box(row));
            }
            black_box(&distances);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("cosine_f64_cached_norms");
    for dimension in [128, 768] {
        let x: Vec<f64> = (0..dimension).map(|i| (i % 17 + 1) as f64 / 19.0).collect();
        let y: Vec<f64> = (0..dimension).map(|i| (i % 13 + 1) as f64 / 15.0).collect();
        let (x_norm, y_norm) = (norm_l2(&x), norm_l2(&y));
        group.bench_with_input(
            BenchmarkId::from_parameter(dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    black_box(f64::cosine_with_norms(
                        black_box(&x),
                        black_box(x_norm),
                        black_box(y_norm),
                        black_box(&y),
                    ));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_cosine_range);
criterion_main!(benches);
