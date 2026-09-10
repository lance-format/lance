// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::OnceLock;

use arrow::array::AsArray;
use arrow::datatypes::Float32Type;
use arrow_array::FixedSizeListArray;
use criterion::{Criterion, criterion_group, criterion_main};

use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::utils::SimpleIndex;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

use lance_index::vector::kmeans::{
    KMeans, KMeansAlgo, KMeansAlgoFloat, KMeansParams, compute_partitions_arrow_array,
};
use lance_index::vector::pq::PQBuildParams;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array;

fn bench_train(c: &mut Criterion) {
    let params = [
        // (64 * 1024, 8),      // training PQ
        // (64 * 1024, 128),    // training IVF with small vectors (1M rows)
        // (64 * 1024, 1024),   // training IVF with large vectors (1M rows)
        // (256 * 1024, 1024),  // hit the threshold for using HNSW to speed up
        // (256 * 2048, 1024),  // hit the threshold for using HNSW to speed up
        // (256 * 4096, 1024),  // hit the threshold for using HNSW to speed up
        (256 * 16384, 1024), // hit the threshold for using HNSW to speed up
    ];
    for (n, dimension) in params {
        let k = n / 256;
        let data: OnceLock<FixedSizeListArray> = OnceLock::new();
        let centroids: OnceLock<FixedSizeListArray> = OnceLock::new();

        c.bench_function(&format!("train_{}d_{}k", dimension, k), |b| {
            let params = KMeansParams::default().with_hierarchical_k(0);
            b.iter(|| {
                let data = data.get_or_init(|| {
                    let values = generate_random_array(n * dimension as usize);
                    FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                });
                KMeans::new_with_params(data, k, &params).ok().unwrap();
            })
        });

        if k > 256 {
            for hierarchical_k in [4, 8, 16, 24, 32] {
                let params = KMeansParams::default().with_hierarchical_k(hierarchical_k);
                c.bench_function(
                    &format!(
                        "train_{}d_{}k_hierarchical_{}",
                        dimension, k, hierarchical_k
                    ),
                    |b| {
                        b.iter(|| {
                            let data = data.get_or_init(|| {
                                let values = generate_random_array(n * dimension as usize);
                                FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                            });
                            KMeans::new_with_params(data, k, &params).ok().unwrap()
                        });
                    },
                );
            }
        }

        let mut group = c.benchmark_group(format!("compute_membership_{}d_{}k", dimension, k));

        group.bench_function("flat", |b| {
            b.iter(|| {
                let data = data.get_or_init(|| {
                    let values = generate_random_array(n * dimension as usize);
                    FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                });
                let centroids = centroids.get_or_init(|| {
                    let values = generate_random_array(k * dimension as usize);
                    FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                });
                compute_partitions_arrow_array(centroids, data, DistanceType::L2)
            })
        });

        if k * dimension as usize >= 1_000_000 {
            let index: OnceLock<SimpleIndex> = OnceLock::new();
            group.bench_function("with_index", |b| {
                b.iter(|| {
                    let data = data.get_or_init(|| {
                        let values = generate_random_array(n * dimension as usize);
                        FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                    });
                    let centroids = centroids.get_or_init(|| {
                        let values = generate_random_array(k * dimension as usize);
                        FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                    });
                    let index = index.get_or_init(|| {
                        SimpleIndex::may_train_index(
                            centroids.values().clone(),
                            dimension as usize,
                            DistanceType::L2,
                        )
                        .unwrap()
                        .unwrap()
                    });
                    KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
                        centroids.values().as_primitive::<Float32Type>().values(),
                        data.values().as_primitive::<Float32Type>().values(),
                        dimension as usize,
                        DistanceType::L2,
                        0.0,
                        None,
                        Some(index),
                    )
                })
            });
        }
    }
}

fn bench_pq_build(c: &mut Criterion) {
    let (dimension, num_sub_vectors) = (128, 16);
    let mut group = c.benchmark_group(format!(
        "pq_build_sampled_{}d_{}m",
        dimension, num_sub_vectors
    ));
    for num_bits in [4, 8] {
        let data = OnceLock::new();
        let params = PQBuildParams::new(num_sub_vectors, num_bits);
        group.bench_function(format!("{}bit", num_bits), |b| {
            b.iter(|| {
                let data = data.get_or_init(|| {
                    let n = 256 * (1 << num_bits);
                    let values = generate_random_array(n * dimension as usize);
                    FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                });
                params.build(data, DistanceType::L2).unwrap()
            })
        });
    }
    group.finish();

    // Callers may pass more rows than the kmeans sample cap
    // (`sample_rate * num_centroids`). The old sub-vector division copied the
    // whole input while training only read a prefix.
    let mut group = c.benchmark_group(format!(
        "pq_build_oversampled_{}d_{}m",
        dimension, num_sub_vectors
    ));
    for num_bits in [4, 8] {
        let data = OnceLock::new();
        let params = PQBuildParams::new(num_sub_vectors, num_bits);
        group.bench_function(format!("{}bit", num_bits), |b| {
            b.iter(|| {
                let data = data.get_or_init(|| {
                    let n = 4 * 1024 * 1024;
                    let values = generate_random_array(n * dimension as usize);
                    FixedSizeListArray::try_new_from_values(values, dimension).unwrap()
                });
                params.build(data, DistanceType::L2).unwrap()
            })
        });
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
    .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_train, bench_pq_build);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_train, bench_pq_build);
criterion_main!(benches);
