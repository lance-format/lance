// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Throughput of native duplicate-pair tile scoring: one 32-row anchor block
//! against a later 8192-row staged batch (an off-diagonal tile) of 1536-d
//! codes, with a threshold that keeps almost no pair, so the time is the
//! kernel plus the filter loop.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::ex_dot::blocked_ex_code_bytes;
use lance_index::vector::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN};
use lance_index::vector::bq::transform::{EX_SCALE_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use lance_index::vector::flat::index::FlatQuantizer;
use lance_index::vector::flat::storage::FLAT_COLUMN;
use lance_index::vector::pairwise::PairwisePartition;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::Quantizer;
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::{PQ_CODE_COLUMN, SQ_CODE_COLUMN};
use lance_linalg::distance::DistanceType;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 1536;
const ANCHOR_ROWS: usize = 32;
const CANDIDATE_ROWS: usize = 8192;
const PQ_SUB_VECTORS: usize = 96;
/// Far below the distance of any two random rows, so almost nothing is emitted.
const THRESHOLD: f32 = 1e-4;

fn fsl(values: ArrayRef, width: usize) -> ArrayRef {
    Arc::new(FixedSizeListArray::try_new_from_values(values, width as i32).unwrap())
}

fn batch(rows: usize, first_id: u64, columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let mut fields = vec![Field::new(ROW_ID, DataType::UInt64, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from_iter_values(
        first_id..first_id + rows as u64,
    ))];
    for (name, array) in columns {
        fields.push(Field::new(name, array.data_type().clone(), false));
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn random_f32(rng: &mut SmallRng, len: usize) -> ArrayRef {
    Arc::new(Float32Array::from_iter_values(
        (0..len).map(|_| rng.random_range(-1.0f32..1.0)),
    ))
}

fn random_u8(rng: &mut SmallRng, len: usize) -> ArrayRef {
    Arc::new(UInt8Array::from_iter_values((0..len).map(|_| rng.random())))
}

/// An anchor batch and a candidate batch of source index rows.
fn sources(
    rng: &mut SmallRng,
    mut columns: impl FnMut(&mut SmallRng, usize) -> Vec<(&'static str, ArrayRef)>,
) -> [RecordBatch; 2] {
    let anchor = batch(ANCHOR_ROWS, 0, columns(rng, ANCHOR_ROWS));
    let candidates = batch(
        CANDIDATE_ROWS,
        ANCHOR_ROWS as u64,
        columns(rng, CANDIDATE_ROWS),
    );
    [anchor, candidates]
}

fn flat(rng: &mut SmallRng, metric: DistanceType) -> (Quantizer, [RecordBatch; 2]) {
    let sources = sources(rng, |rng, rows| {
        vec![(FLAT_COLUMN, fsl(random_f32(rng, rows * DIM), DIM))]
    });
    (Quantizer::Flat(FlatQuantizer::new(DIM, metric)), sources)
}

fn pq(rng: &mut SmallRng, num_bits: u32, metric: DistanceType) -> (Quantizer, [RecordBatch; 2]) {
    let codebook =
        FixedSizeListArray::try_new_from_values(random_f32(rng, (1 << num_bits) * DIM), DIM as i32)
            .unwrap();
    let pq = ProductQuantizer::new(PQ_SUB_VECTORS, num_bits, DIM, codebook, metric);
    let code_bytes = PQ_SUB_VECTORS * num_bits as usize / 8;
    // Random bytes are equally valid column-major and row-major codes.
    let sources = sources(rng, |rng, rows| {
        vec![(
            PQ_CODE_COLUMN,
            fsl(random_u8(rng, rows * code_bytes), code_bytes),
        )]
    });
    (Quantizer::Product(pq), sources)
}

fn sq(rng: &mut SmallRng) -> (Quantizer, [RecordBatch; 2]) {
    let sources = sources(rng, |rng, rows| {
        vec![(SQ_CODE_COLUMN, fsl(random_u8(rng, rows * DIM), DIM))]
    });
    (
        Quantizer::Scalar(ScalarQuantizer::with_bounds(8, DIM, -1.0..1.0)),
        sources,
    )
}

fn rq(rng: &mut SmallRng, bits: u8) -> (Quantizer, [RecordBatch; 2]) {
    let rq =
        RabitQuantizer::new_with_rotation::<Float32Type>(bits, DIM as i32, RQRotationType::Fast);
    let sources = sources(rng, |rng, rows| {
        // Stored l2/cosine scale factors are `-2s`.
        let scales: ArrayRef = Arc::new(Float32Array::from_iter_values(
            (0..rows).map(|_| rng.random_range(-0.2f32..-0.01)),
        ));
        let mut columns = vec![(
            RABIT_CODE_COLUMN,
            fsl(random_u8(rng, rows * DIM / 8), DIM / 8),
        )];
        if bits == 1 {
            columns.push((SCALE_FACTORS_COLUMN, scales));
        } else {
            let width = blocked_ex_code_bytes(DIM, bits - 1);
            columns.push((
                RABIT_BLOCKED_EX_CODE_COLUMN,
                fsl(random_u8(rng, rows * width), width),
            ));
            columns.push((EX_SCALE_FACTORS_COLUMN, scales));
        }
        columns
    });
    (Quantizer::Rabit(rq), sources)
}

fn bench_pairwise_tile(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let cases = vec![
        (
            "flat_l2",
            DistanceType::L2,
            flat(&mut rng, DistanceType::L2),
        ),
        (
            "flat_cosine",
            DistanceType::Cosine,
            flat(&mut rng, DistanceType::Cosine),
        ),
        (
            "pq8_cosine",
            DistanceType::Cosine,
            pq(&mut rng, 8, DistanceType::Cosine),
        ),
        (
            "pq4_cosine",
            DistanceType::Cosine,
            pq(&mut rng, 4, DistanceType::Cosine),
        ),
        ("sq_cosine", DistanceType::Cosine, sq(&mut rng)),
        ("rq1_cosine", DistanceType::Cosine, rq(&mut rng, 1)),
        ("rq5_cosine", DistanceType::Cosine, rq(&mut rng, 5)),
        ("rq8_cosine", DistanceType::Cosine, rq(&mut rng, 8)),
    ];
    let mut group = c.benchmark_group("pairwise_tile");
    group.throughput(Throughput::Elements((ANCHOR_ROWS * CANDIDATE_ROWS) as u64));
    for (name, metric, (quantizer, sources)) in cases {
        let centroid = random_f32(&mut rng, DIM);
        let partition =
            PairwisePartition::stage_in_memory(&quantizer, centroid, metric, &sources).unwrap();
        let (anchor, candidates) = runtime.block_on(async {
            (
                partition.read_vectors(0).await.unwrap(),
                partition.read_vectors(1).await.unwrap(),
            )
        });
        let hits = partition
            .score_block(&anchor, 0..ANCHOR_ROWS, &candidates, THRESHOLD)
            .unwrap();
        assert!(
            hits.distances.len() < ANCHOR_ROWS,
            "{name} emits too many pairs"
        );
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    partition
                        .score_block(&anchor, 0..ANCHOR_ROWS, &candidates, THRESHOLD)
                        .unwrap(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .sample_size(20);
    targets = bench_pairwise_tile);

criterion_main!(benches);
