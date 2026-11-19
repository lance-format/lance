// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance_linalg::distance::{Cosine, norm_l2::norm_l2};
use lance_linalg::kernels::normalize;

// Benchmark-only alternatives for the design comparison in #11269. Independent
// partial sums allow SIMD. Separate dot and norm reductions avoid LLVM packing
// unrelated products together instead of vectorizing across dimensions.
#[inline]
fn dot_widened(x: &[f32], y: &[f32]) -> f64 {
    let (x_chunks, x_remainder) = x.as_chunks::<16>();
    let (y_chunks, y_remainder) = y.as_chunks::<16>();
    let mut sums = [0.0; 16];
    for (x, y) in x_chunks.iter().zip(y_chunks) {
        for i in 0..16 {
            sums[i] = f64::from(x[i]).mul_add(f64::from(y[i]), sums[i]);
        }
    }
    sums.iter().sum::<f64>()
        + x_remainder
            .iter()
            .zip(y_remainder)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum::<f64>()
}

fn cosine_widened(query: &[f32], query_norm: f64, row: &[f32]) -> f32 {
    let product = dot_widened(query, row);
    let row_norm = dot_widened(row, row).sqrt();
    (1.0 - product / query_norm / row_norm) as f32
}

fn bench_widening_comparison(c: &mut Criterion) {
    const ROWS: usize = 4096;
    for dimension in [7, 128, 768] {
        for (workload, scale) in [
            ("ordinary", 1.0),
            ("large", 1e20),
            ("tiny", 1e-30),
            ("mixed", 1.0),
        ] {
            let query: Vec<f32> = (0..dimension)
                .map(|i| (i as f32 % 17.0 - 8.0) / 9.0 * scale)
                .collect();
            let mut batch: Vec<f32> = (0..ROWS * dimension)
                .map(|i| {
                    let row = i / dimension;
                    let scale = if workload == "mixed" && row.is_multiple_of(64) {
                        1e20
                    } else {
                        scale
                    };
                    ((i % dimension * 7 + row * 13) % 31) as f32 / 16.0 * scale - 0.9375 * scale
                })
                .collect();
            batch[..dimension].copy_from_slice(&query);
            let mut distances = vec![0.0; ROWS];
            let mut widened_distances = vec![0.0; ROWS];
            let mut normalized = vec![0.0; batch.len()];
            let mut widened_normalized = vec![0.0; batch.len()];

            // Check both candidates on the measured inputs outside the timed loop,
            // including self matches, non-multiple-of-16 tails, and extreme ranges.
            f32::cosine_batch_into(&query, &batch, dimension, &mut distances);
            let query_norm = dot_widened(&query, &query).sqrt();
            for (distance, row) in widened_distances
                .iter_mut()
                .zip(batch.chunks_exact(dimension))
            {
                *distance = cosine_widened(&query, query_norm, row);
            }
            assert!(distances[0].abs() < 2e-6);
            assert!(widened_distances[0].abs() < 2e-6);
            assert!(
                distances
                    .iter()
                    .zip(&widened_distances)
                    .all(|(a, b)| (a - b).abs() < 2e-6)
            );
            for ((row, output), widened_output) in batch
                .chunks_exact(dimension)
                .zip(normalized.chunks_exact_mut(dimension))
                .zip(widened_normalized.chunks_exact_mut(dimension))
            {
                for (output, value) in output.iter_mut().zip(normalize(row).0) {
                    *output = value;
                }
                let norm = dot_widened(row, row).sqrt();
                for (output, &value) in widened_output.iter_mut().zip(row) {
                    *output = (f64::from(value) / norm) as f32;
                }
                assert!((dot_widened(widened_output, widened_output) - 1.0).abs() < 2e-6);
            }
            assert!(
                normalized
                    .iter()
                    .zip(&widened_normalized)
                    .all(|(a, b)| (a - b).abs() < 2e-6)
            );

            let mut group = c.benchmark_group(format!("cosine_f32_widening/{workload}"));
            group.throughput(Throughput::Elements(ROWS as u64));
            group.bench_function(BenchmarkId::new("fallback", dimension), |b| {
                b.iter(|| {
                    f32::cosine_batch_into(
                        black_box(&query),
                        black_box(&batch),
                        dimension,
                        black_box(&mut distances),
                    );
                    black_box(&distances);
                });
            });
            group.bench_function(BenchmarkId::new("widen", dimension), |b| {
                b.iter(|| {
                    let query = black_box(&query);
                    let query_norm = dot_widened(query, query).sqrt();
                    for (distance, row) in black_box(&mut widened_distances)
                        .iter_mut()
                        .zip(black_box(&batch).chunks_exact(dimension))
                    {
                        *distance = cosine_widened(query, query_norm, row);
                    }
                    black_box(&widened_distances);
                });
            });
            group.finish();

            let mut group = c.benchmark_group(format!("normalize_f32_widening/{workload}"));
            group.throughput(Throughput::Elements(ROWS as u64));
            group.bench_function(BenchmarkId::new("fallback", dimension), |b| {
                b.iter(|| {
                    for (row, output) in black_box(&batch)
                        .chunks_exact(dimension)
                        .zip(black_box(&mut normalized).chunks_exact_mut(dimension))
                    {
                        for (output, value) in output.iter_mut().zip(normalize(row).0) {
                            *output = value;
                        }
                    }
                    black_box(&normalized);
                });
            });
            group.bench_function(BenchmarkId::new("widen", dimension), |b| {
                b.iter(|| {
                    for (row, output) in black_box(&batch)
                        .chunks_exact(dimension)
                        .zip(black_box(&mut widened_normalized).chunks_exact_mut(dimension))
                    {
                        let norm = dot_widened(row, row).sqrt();
                        for (output, &value) in output.iter_mut().zip(row) {
                            *output = (f64::from(value) / norm) as f32;
                        }
                    }
                    black_box(&widened_normalized);
                });
            });
            group.finish();
        }
    }
}

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

criterion_group!(benches, bench_cosine_range, bench_widening_comparison);
criterion_main!(benches);
