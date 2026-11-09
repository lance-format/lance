// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of 4-bit PQ partition scans through the top-k accumulator, from
//! an empty heap as the first IVF partition sees it and from a full heap as
//! later partitions do.

use std::collections::BinaryHeap;
use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID_FIELD;
use lance_index::vector::graph::{OrderedFloat, OrderedNode};
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::pq::storage::ProductQuantizationStorage;
use lance_index::vector::storage::{DistCalculator, StorageBuilder, VectorStore};
use lance_linalg::distance::DistanceType;
use rand::{Rng, SeedableRng, rngs::StdRng};

const NUM_QUERIES: usize = 32;

/// Rows scattered around a few cluster centers, like the residuals of one IVF
/// partition, and queries near random rows.
fn build_case(
    rows: usize,
    dim: usize,
    num_sub_vectors: usize,
) -> (ProductQuantizationStorage, Vec<ArrayRef>) {
    let mut rng = StdRng::seed_from_u64(42);
    let centers = (0..16 * dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect::<Vec<_>>();
    let values = (0..rows)
        .flat_map(|_| {
            let center = rng.random_range(0..16) * dim;
            (0..dim)
                .map(|i| centers[center + i] + rng.random_range(-0.5f32..0.5))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values.clone()), dim as i32)
            .unwrap();
    let mut params = PQBuildParams::new(num_sub_vectors, 4);
    params.max_iters = 10;
    let pq = params.build(&vectors, DistanceType::L2).unwrap();
    let schema = Schema::new(vec![
        Field::new("vec", vectors.data_type().clone(), true),
        ROW_ID_FIELD.clone(),
    ]);
    let batch = RecordBatch::try_new(
        schema.into(),
        vec![
            Arc::new(vectors),
            Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
        ],
    )
    .unwrap();
    let storage = StorageBuilder::new("vec".to_owned(), DistanceType::L2, pq, None)
        .unwrap()
        .build(vec![batch])
        .unwrap();
    let queries = (0..NUM_QUERIES)
        .map(|_| {
            let row = rng.random_range(0..rows) * dim;
            let query = values[row..row + dim]
                .iter()
                .map(|v| v + rng.random_range(-0.2f32..0.2))
                .collect::<Float32Array>();
            Arc::new(query) as ArrayRef
        })
        .collect();
    (storage, queries)
}

fn bench_pq4_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq4_topk");
    for (dim, num_sub_vectors) in [(128, 64), (768, 96)] {
        for rows in [1024, 16384] {
            let (storage, queries) = build_case(rows, dim, num_sub_vectors);
            for k in [10, 100] {
                // A full heap holding this partition's k-th best distance.
                let prefills = queries
                    .iter()
                    .map(|query| {
                        let calc = storage.dist_calculator(query.clone(), 0.0);
                        let mut dists = (0..rows as u32)
                            .map(|id| calc.distance(id))
                            .collect::<Vec<_>>();
                        dists.sort_by(f32::total_cmp);
                        (0..k)
                            .map(|i| {
                                OrderedNode::new(u64::MAX - i as u64, OrderedFloat(dists[k - 1]))
                            })
                            .collect::<BinaryHeap<_>>()
                    })
                    .collect::<Vec<_>>();
                for prefilled in [false, true] {
                    let id = format!(
                        "dim={dim},m={num_sub_vectors},rows={rows},k={k},prefilled={prefilled}"
                    );
                    let (mut dists, mut u16s, mut u8s, mut u32s) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    group.bench_function(BenchmarkId::from_parameter(id), |b| {
                        b.iter(|| {
                            for (query, prefill) in queries.iter().zip(&prefills) {
                                let mut heap = if prefilled {
                                    prefill.clone()
                                } else {
                                    BinaryHeap::with_capacity(k)
                                };
                                storage
                                    .dist_calculator(query.clone(), 0.0)
                                    .accumulate_topk_with_scratch(
                                        k,
                                        None,
                                        None,
                                        |id| id as u64,
                                        &mut heap,
                                        &mut dists,
                                        &mut u16s,
                                        &mut u8s,
                                        &mut u32s,
                                    );
                                black_box(heap.len());
                            }
                        })
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_pq4_topk
);
criterion_main!(benches);
