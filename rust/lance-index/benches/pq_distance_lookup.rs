// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of PQ distance lookup for individual candidate vectors.

use std::{hint::black_box, sync::Arc};

use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID_FIELD;
use lance_index::vector::{
    pq::ProductQuantizer,
    storage::{DistCalculator, StorageBuilder, VectorStore},
};
use lance_linalg::distance::DistanceType;
use rand::{Rng, SeedableRng, rngs::SmallRng};

const DIM: usize = 512;
const NUM_SUB_VECTORS: usize = 64;
const TOTAL: usize = 65_536;
const IDS_PER_ITER: usize = 16_384;

fn random_f32_array(len: usize, seed: u64) -> Float32Array {
    let mut rng = SmallRng::seed_from_u64(seed);
    Float32Array::from_iter_values((0..len).map(|_| rng.random_range(-1.0f32..1.0f32)))
}

fn create_storage() -> lance_index::vector::pq::storage::ProductQuantizationStorage {
    let distance_type = DistanceType::Dot;
    let codebook_values = random_f32_array(256 * DIM, 88);
    let codebook = FixedSizeListArray::try_new_from_values(codebook_values, DIM as i32).unwrap();
    let pq = ProductQuantizer::new(NUM_SUB_VECTORS, 8, DIM, codebook, distance_type);

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(
            "vec",
            DataType::FixedSizeList(
                Field::new_list_field(DataType::Float32, true).into(),
                DIM as i32,
            ),
            true,
        ),
        ROW_ID_FIELD.clone(),
    ]));
    let vectors = random_f32_array(TOTAL * DIM, 3);
    let row_ids = UInt64Array::from_iter_values((0..TOTAL).map(|v| v as u64));
    let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
    let batch = RecordBatch::try_new(schema, vec![Arc::new(fsl), Arc::new(row_ids)]).unwrap();

    StorageBuilder::new("vec".to_owned(), distance_type, pq, None)
        .unwrap()
        .build(vec![batch])
        .unwrap()
}

fn distance_lookup(c: &mut Criterion) {
    let storage = create_storage();
    let query = Arc::new(random_f32_array(DIM, 32));
    let dist_calc = storage.dist_calculator(query, 0.0);
    let ids = (0..IDS_PER_ITER)
        .map(|i| ((i * 31) % TOTAL) as u32)
        .collect::<Vec<_>>();

    c.bench_function(
        "pq_distance_lookup: 8bit,total=65536,pq=64,dim=512,ids=16384",
        |b| {
            b.iter(|| {
                let mut total = 0.0;
                for id in ids.iter().copied() {
                    total += dist_calc.distance(black_box(id));
                }
                black_box(total);
            })
        },
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(30);
    targets = distance_lookup
);
criterion_main!(benches);
