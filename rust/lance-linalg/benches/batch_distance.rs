// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use lance_linalg::distance::{Cosine, Dot, L2};

const TOTAL_VALUES: usize = 1024 * 1024;

fn bench_batch_distance(c: &mut Criterion) {
    // 8 and 16 are common PQ sub-vector widths, 128 a common IVF centroid
    // width. Cosine is measured at 8 only.
    for dimension in [8, 16, 32, 128] {
        let key = (0..dimension)
            .map(|index| index as f32 * 0.125 + 0.25)
            .collect::<Vec<_>>();
        let batch = (0..TOTAL_VALUES)
            .map(|index| (index % 31) as f32 * 0.03125 - 0.5)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; TOTAL_VALUES / dimension];

        // Callers pass the dimension at runtime. A constant would let LLVM unroll
        // the loop only where the kernel inlines into this closure, so the
        // comparison would measure inlining rather than the kernel.
        c.bench_function(&format!("Batch distance dot f32 dim {dimension}"), |b| {
            b.iter(|| {
                <f32 as Dot>::dot_batch_into(
                    black_box(&key),
                    black_box(&batch),
                    black_box(dimension),
                    &mut output,
                );
                black_box(&mut output);
            })
        });
        c.bench_function(&format!("Batch distance l2 f32 dim {dimension}"), |b| {
            b.iter(|| {
                <f32 as L2>::l2_batch_into(
                    black_box(&key),
                    black_box(&batch),
                    black_box(dimension),
                    &mut output,
                );
                black_box(&mut output);
            })
        });
        if dimension == 8 {
            c.bench_function("Batch distance cosine f32 dim 8", |b| {
                b.iter(|| {
                    <f32 as Cosine>::cosine_batch_into(
                        black_box(&key),
                        black_box(&batch),
                        black_box(dimension),
                        &mut output,
                    );
                    black_box(&mut output);
                })
            });
        }
    }
}

fn bench_time() -> Duration {
    let seconds = option_env!("TARGET_TIME")
        .unwrap_or("5")
        .parse()
        .expect("TARGET_TIME must be an integer number of seconds");
    Duration::from_secs(seconds)
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10)
        .measurement_time(bench_time());
    targets = bench_batch_distance
);
criterion_main!(benches);
