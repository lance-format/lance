// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, types::Float32Type};
use lance_arrow::FixedSizeListArrayExt;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

use lance_index::vector::ivf::IvfTransformer;
use lance_index::vector::utils::SimpleIndex;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;
use std::sync::Arc;

fn bench_partitions(c: &mut Criterion) {
    const DIMENSION: usize = 1536;
    const SEED: [u8; 32] = [42; 32];

    let query: Float32Array = generate_random_array_with_seed::<Float32Type>(DIMENSION, SEED);

    for num_centroids in &[10240, 65536] {
        let centroids =
            generate_random_array_with_seed::<Float32Type>(num_centroids * DIMENSION, SEED);
        let fsl = FixedSizeListArray::try_new_from_values(centroids, DIMENSION as i32).unwrap();

        for k in &[1, 10, 50] {
            let ivf = IvfTransformer::new(fsl.clone(), DistanceType::L2, vec![]);
            c.bench_function(format!("IVF{},k={},L2", num_centroids, k).as_str(), |b| {
                b.iter(|| {
                    let _ = ivf.find_partitions(&query, *k);
                })
            });

            let ivf = IvfTransformer::new(fsl.clone(), DistanceType::Cosine, vec![]);
            c.bench_function(
                format!("IVF{},k={},Cosine", num_centroids, k).as_str(),
                |b| {
                    b.iter(|| {
                        let _ = ivf.find_partitions(&query, *k);
                    })
                },
            );
        }

        let ivf = IvfTransformer::new(fsl.clone(), DistanceType::L2, vec![]);
        let batch = generate_random_array_with_seed::<Float32Type>(DIMENSION * 4096, SEED);
        let fsl = FixedSizeListArray::try_new_from_values(batch, DIMENSION as i32).unwrap();
        c.bench_function(
            format!("compute_partitions: IVF{},L2,n={}", num_centroids, 4096).as_str(),
            |b| b.iter(|| ivf.compute_partitions(&fsl)),
        );
    }
}

fn bench_centroid_routing(c: &mut Criterion) {
    const SEED: [u8; 32] = [42; 32];

    // This is the sweep proposed in #8775. Filter by dimension or partition count on the
    // Criterion command line for a shorter local run.
    for dimension in [128, 768, 1024] {
        let query = generate_random_array_with_seed::<Float32Type>(dimension, SEED);
        let query_ref: ArrayRef = Arc::new(query.clone());

        for num_centroids in [256, 1024, 4096, 16384] {
            let centroid_values =
                generate_random_array_with_seed::<Float32Type>(num_centroids * dimension, SEED);
            let fsl =
                FixedSizeListArray::try_new_from_values(centroid_values.clone(), dimension as i32)
                    .unwrap();
            let exact = IvfTransformer::new(fsl, DistanceType::L2, vec![]);
            let hnsw = SimpleIndex::try_new_centroid_index(
                Arc::new(centroid_values),
                dimension,
                DistanceType::L2,
            )
            .unwrap()
            .unwrap();

            let mut group = c.benchmark_group(format!(
                "centroid_routing/dim={dimension}/partitions={num_centroids}"
            ));
            for nprobes in [4, 16, 64] {
                group.bench_with_input(
                    BenchmarkId::new("exact", nprobes),
                    &nprobes,
                    |b, &nprobes| b.iter(|| exact.find_partitions(&query, nprobes)),
                );
                for ef_multiplier in [1, 2, 4, 8] {
                    let centroid_ef = nprobes * ef_multiplier;
                    group.bench_with_input(
                        BenchmarkId::new(format!("hnsw/ef={centroid_ef}"), nprobes),
                        &(nprobes, centroid_ef),
                        |b, &(nprobes, centroid_ef)| {
                            b.iter(|| hnsw.search(query_ref.clone(), nprobes, centroid_ef))
                        },
                    );
                }
            }
            group.finish();
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_partitions, bench_centroid_routing);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_partitions, bench_centroid_routing);

criterion_main!(benches);
