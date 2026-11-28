// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{
    FixedSizeListArray, Float32Array,
    types::{Float16Type, Float32Type, Float64Type},
};
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray, bfloat16::BFloat16Type};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::cosine::{Cosine, cosine_distance_batch_into};
use lance_linalg::distance::cosine_u8::{cosine_u8, cosine_u8_scalar};
use num_traits::Float;

#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

use lance_testing::datagen::generate_random_array_with_seed;

fn cosine_scalar<T: Float>(x: &[T], y: &[T], dim: usize) -> Vec<T> {
    y.chunks_exact(dim)
        .map(|vec| {
            let mut dot = T::zero();
            let mut x_norm = T::zero();
            let mut y_norm = T::zero();

            for (&xi, &yi) in x.iter().zip(vec.iter()) {
                dot = dot + xi * yi;
                x_norm = x_norm + xi * xi;
                y_norm = y_norm + yi * yi;
            }
            dot / (x_norm * y_norm).sqrt()
        })
        .collect()
}

fn run_bench<T: ArrowFloatType>(c: &mut Criterion)
where
    T::Native: Cosine,
{
    const DIMENSION: usize = 1024;
    const TOTAL: usize = 512 * 1024;

    let type_name = std::any::type_name::<T::Native>();
    let key = generate_random_array_with_seed::<T>(DIMENSION, [0; 32]);
    let target = generate_random_array_with_seed::<T>(TOTAL * DIMENSION, [42; 32]);

    c.bench_function(format!("Cosine({}, scalar)", type_name).as_str(), |b| {
        b.iter(|| {
            black_box(cosine_scalar(key.as_slice(), target.as_slice(), DIMENSION));
        })
    });

    c.bench_function(
        format!("Cosine({}, auto-vectorized)", type_name).as_str(),
        |b| {
            let mut distances = vec![0.0; TOTAL];
            b.iter(|| {
                cosine_distance_batch_into(
                    key.as_slice(),
                    target.as_slice(),
                    DIMENSION,
                    &mut distances,
                );
                black_box(&mut distances);
            })
        },
    );
}

fn bench_distance(c: &mut Criterion) {
    run_bench::<BFloat16Type>(c);
    run_bench::<Float16Type>(c);
    run_bench::<Float32Type>(c);
    run_bench::<Float64Type>(c);

    let key: Float32Array = generate_random_array_with_seed::<Float32Type>(8, [0; 32]);
    let target = generate_random_array_with_seed::<Float32Type>(1024 * 1024 * 8, [42; 32]);

    c.bench_function("Cosine(simd,f32x8) rng seed", |b| {
        let mut distances = vec![0.0; target.len() / 8];
        b.iter(|| {
            cosine_distance_batch_into(key.values(), target.values(), 8, &mut distances);
            black_box(&mut distances);
        })
    });

    let target = FixedSizeListArray::try_new_from_values(target, 8).unwrap();
    c.bench_function("Cosine(f32x8, arrow) rng seed", |b| {
        b.iter(|| black_box(DistanceType::Cosine.batch(&key, &target).unwrap()))
    });

    // u8 cosine benchmarks
    {
        use rand::Rng;
        use std::iter::repeat_with;

        const DIMENSION: usize = 1024;
        const TOTAL: usize = 1024 * 1024;
        let mut rng = rand::rng();
        // Slice 64-byte-aligned views out of padded buffers: zmm loads split cache
        // lines on misaligned inputs, so results depended on heap placement.
        let key_buf: Vec<u8> = repeat_with(|| rng.random()).take(DIMENSION + 63).collect();
        let key_u8 = &key_buf[key_buf.as_ptr().align_offset(64)..][..DIMENSION];
        let target_buf: Vec<u8> = repeat_with(|| rng.random())
            .take(TOTAL * DIMENSION + 63)
            .collect();
        let target_u8 = &target_buf[target_buf.as_ptr().align_offset(64)..][..TOTAL * DIMENSION];

        c.bench_function("Cosine(u8, scalar)", |b| {
            b.iter(|| {
                black_box(
                    target_u8
                        .as_chunks::<DIMENSION>()
                        .0
                        .iter()
                        .map(|tgt| cosine_u8_scalar(key_u8, tgt))
                        .fold(0.0, |acc: f32, v| acc + v),
                );
            });
        });

        c.bench_function("Cosine(u8, SIMD)", |b| {
            b.iter(|| {
                black_box(
                    target_u8
                        .as_chunks::<DIMENSION>()
                        .0
                        .iter()
                        .map(|tgt| cosine_u8(key_u8, tgt))
                        .fold(0.0, |acc: f32, v| acc + v),
                );
            });
        });
    }
}

/// `cosine_with_norms` with precomputed norms, as used by flat storage. Keeps
/// ~256 KiB of 64-byte-aligned targets in cache so it measures the kernel.
fn bench_cosine_with_norms(c: &mut Criterion) {
    for dim in [8, 16, 32, 64, 128, 256, 768, 1024, 1536] {
        let num_vectors = 64 * 1024 / dim;
        let aligned = |len: usize, seed: [u8; 32]| {
            let values = generate_random_array_with_seed::<Float32Type>(len + 15, seed);
            let values = values.values().to_vec();
            let offset = values.as_ptr().align_offset(64);
            (values, offset)
        };
        let (key_buf, key_offset) = aligned(dim, [0; 32]);
        let (target_buf, target_offset) = aligned(num_vectors * dim, [42; 32]);
        let x = &key_buf[key_offset..][..dim];
        let target = &target_buf[target_offset..][..num_vectors * dim];
        let x_norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y_norms = target
            .chunks_exact(dim)
            .map(|y| y.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect::<Vec<_>>();
        c.bench_function(&format!("Cosine(f32, with norms, dim={dim})"), |b| {
            b.iter(|| {
                for (y, &y_norm) in target.chunks_exact(black_box(dim)).zip(&y_norms) {
                    black_box(f32::cosine_with_norms(x, x_norm, y_norm, y));
                }
            });
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_cosine_with_norms, bench_distance);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_cosine_with_norms, bench_distance);
criterion_main!(benches);
