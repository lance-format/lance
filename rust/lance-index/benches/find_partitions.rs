// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::Float32Array;
use arrow_array::{FixedSizeListArray, types::Float32Type};
use lance_arrow::FixedSizeListArrayExt;

use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

use lance_index::vector::ivf::IvfTransformer;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;

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

/// Partition search at common embedding widths, where the query-to-centroid
/// distances rather than the top-k selection dominate.
fn bench_partitions_by_dimension(c: &mut Criterion) {
    const NUM_CENTROIDS: usize = 10240;
    const NPROBES: usize = 20;
    const SEED: [u8; 32] = [42; 32];

    for dimension in [128, 768] {
        let query: Float32Array = generate_random_array_with_seed::<Float32Type>(dimension, SEED);
        let centroids =
            generate_random_array_with_seed::<Float32Type>(NUM_CENTROIDS * dimension, SEED);
        let fsl = FixedSizeListArray::try_new_from_values(centroids, dimension as i32).unwrap();

        for distance_type in [DistanceType::L2, DistanceType::Dot] {
            let ivf = IvfTransformer::new(fsl.clone(), distance_type, vec![]);
            c.bench_function(
                format!("IVF{NUM_CENTROIDS},dim={dimension},k={NPROBES},{distance_type}").as_str(),
                |b| {
                    b.iter(|| {
                        let _ = ivf.find_partitions(&query, NPROBES);
                    })
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_partitions, bench_partitions_by_dimension);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_partitions, bench_partitions_by_dimension);

criterion_main!(benches);
