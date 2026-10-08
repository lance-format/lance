// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF_HNSW_RQ kernel + graph-build benches. Numbers are not asserted.
//!
//! - `from_id` / `dist_between`: production pair path used while inserting
//!   into the HNSW graph (`CachedSymPair`, not a raw Arrow re-lookup).
//! - `hnsw_rq_build`: `HNSW::index_vectors` over Residual RQ storage.
//! - The other groups time walk / rerank / pack kernels.
//!
//! Index-level Flat `IVF_RQ` vs `IVF_HNSW_RQ` (build, search, recall) is
//! `lance --bench ivf_rq_hnsw`.
//!
//! ```text
//! cargo bench -p lance-index --bench sym_dist -- --quick --noplot
//! # Criterion takes one filter. Graph pair / graph-build subsets:
//! cargo bench -p lance-index --bench sym_dist -- --quick --noplot from_id
//! cargo bench -p lance-index --bench sym_dist -- --quick --noplot hnsw_rq_build
//! ```

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt8Type};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RabitQuantizationStorage, RabitQueryEstimator,
};
use lance_index::vector::bq::sym::{
    SymFactors, const_scaling_factor_3ex, mask_ip_x0_q, prepare_residual_warmup_query, sym_dist,
    warmup_ip_x0_q,
};
use lance_index::vector::bq::transform::{
    RQTransformer, SYM_BIN_CODES_COLUMN, SYM_GAMMA_COLUMN, SYM_IP_CENT_COLUMN, SYM_RHO_COLUMN,
    SYM_UNORM_COLUMN,
};
use lance_index::vector::graph::OrderedNode;
use lance_index::vector::hnsw::builder::{HNSW, HnswBuildParams};
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::storage::{DistCalculator, VectorStore};
use lance_index::vector::transform::Transformer;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{CENTROID_DIST_COLUMN, PART_ID_COLUMN};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::l2::l2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const WARMUP_DIMS: [usize; 10] = [64, 96, 104, 128, 192, 200, 256, 384, 512, 768];
const QUERY_DIM: usize = 960;
const NUM_BASE: usize = 4096;
const QUERY_BITS: [u8; 2] = [5, 9];
const GRAPH_DIM: usize = 128;
const GRAPH_N: usize = 2048;

struct SymBenchData {
    rq: RabitQuantizer,
    transformer: RQTransformer,
    batch: RecordBatch,
    encoded: RecordBatch,
}

fn encode_sym_batch(residuals: &[f32], dim: usize, num_bits: u8) -> SymBenchData {
    let n = residuals.len() / dim;
    let residual_fsl =
        FixedSizeListArray::try_new_from_values(Float32Array::from(residuals.to_vec()), dim as i32)
            .unwrap();
    let mut params = RQBuildParams::with_rotation_type(num_bits, RQRotationType::Fast);
    // Production IVF_HNSW_RQ shape: RawQuery F_* + `__sym_*` columns.
    params.with_sym_columns = true;
    let rq = RabitQuantizer::build(&residual_fsl, DistanceType::L2, &params).unwrap();
    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32; dim]), dim as i32)
            .unwrap();
    let transformer =
        RQTransformer::new(rq.clone(), DistanceType::L2, centroids, "vector").unwrap();
    let centroid_dists: Vec<f32> = residuals
        .chunks_exact(dim)
        .map(|row| row.iter().map(|value| value * value).sum())
        .collect();
    let batch = RecordBatch::try_from_iter(vec![
        ("vector", Arc::new(residual_fsl) as ArrayRef),
        (
            PART_ID_COLUMN,
            Arc::new(UInt32Array::from(vec![0u32; n])) as ArrayRef,
        ),
        (
            CENTROID_DIST_COLUMN,
            Arc::new(Float32Array::from(centroid_dists)) as ArrayRef,
        ),
    ])
    .unwrap();
    let encoded = transformer.transform(&batch).unwrap();
    SymBenchData {
        rq,
        transformer,
        batch,
        encoded,
    }
}

fn encode_rq_storage(
    residuals: &[f32],
    dim: usize,
    num_bits: u8,
    walk: bool,
) -> RabitQuantizationStorage {
    let n = residuals.len() / dim;
    let data = encode_sym_batch(residuals, dim, num_bits);
    let with_row_ids = data
        .encoded
        .try_with_column(
            lance_core::ROW_ID_FIELD.clone(),
            Arc::new(UInt64Array::from_iter_values(0..n as u64)) as ArrayRef,
        )
        .unwrap();
    let mut metadata = data.rq.metadata(None);
    metadata.query_estimator = RabitQueryEstimator::RawQuery;
    // Walk storage keeps `with_sym_columns` so `dist_calculator()` is
    // binary-only (1-bit warmup). Raw storage clears the flag so
    // `dist_calculator()` uses the exact kernel; `__sym_*` stay in the
    // batch for mask_ip.
    metadata.with_sym_columns = walk;
    RabitQuantizationStorage::try_from_batch(with_row_ids, &metadata, DistanceType::L2, None)
        .unwrap()
}

fn random_query(dim: usize, seed: u64) -> (ArrayRef, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let dist_q_c = values.iter().map(|value| value * value).sum();
    (Arc::new(Float32Array::from(values)) as ArrayRef, dist_q_c)
}

fn scan_ids(calc: &impl DistCalculator, n: usize) -> f32 {
    let mut acc = 0.0f32;
    for id in 0..n as u32 {
        acc += calc.distance(id);
    }
    acc
}

fn scan_ids_perm(calc: &impl DistCalculator, ids: &[u32]) -> f32 {
    let mut acc = 0.0f32;
    for &id in ids {
        acc += calc.distance(id);
    }
    acc
}

fn shuffled_ids(n: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..n as u32).collect();
    let mut rng = StdRng::seed_from_u64(20260917);
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        ids.swap(i, j);
    }
    ids
}

fn pack_bins(residuals: &[f32], dim: usize) -> (Vec<u8>, usize) {
    let data = encode_sym_batch(residuals, dim, 5);
    let bins = data.encoded[SYM_BIN_CODES_COLUMN].as_fixed_size_list();
    let width = bins.value_length() as usize;
    let values = bins.values().as_primitive::<UInt8Type>().values();
    (values.to_vec(), width)
}

fn warmup_scan(bins: &[u8], width: usize, planes: &[u64], delta: f32, vl: f32) -> f32 {
    let mut acc = 0.0f32;
    for row in bins.chunks_exact(width) {
        acc += warmup_ip_x0_q(row, planes, delta, vl);
    }
    acc
}

fn mask_ip_scan(bins: &[u8], width: usize, query: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for row in bins.chunks_exact(width) {
        acc += mask_ip_x0_q(query, row);
    }
    acc
}

fn bench_warmup(c: &mut Criterion) {
    for dim in WARMUP_DIMS {
        let mut rng = StdRng::seed_from_u64(20260914 + dim as u64);
        let residuals: Vec<f32> = (0..NUM_BASE * dim)
            .map(|_| rng.random_range(-2.0f32..2.0))
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let warmup = prepare_residual_warmup_query(&query, const_scaling_factor_3ex(dim));
        let (bins, width) = pack_bins(&residuals, dim);

        let mut group = c.benchmark_group(format!("warmup_scan_dim{dim}"));
        group.throughput(Throughput::Elements(NUM_BASE as u64));
        group.bench_function("dispatch", |b| {
            b.iter(|| {
                black_box(warmup_scan(
                    &bins,
                    width,
                    &warmup.planes,
                    warmup.delta,
                    warmup.vl,
                ))
            })
        });
        group.finish();
    }
}

/// Pair `sym_dist` (HNSW_RQ graph build / prune).
fn bench_sym_dist_pair(c: &mut Criterion) {
    for dim in WARMUP_DIMS {
        for num_bits in QUERY_BITS {
            let mut rng = StdRng::seed_from_u64(20260914 + dim as u64);
            let residuals: Vec<f32> = (0..NUM_BASE * dim)
                .map(|_| rng.random_range(-2.0f32..2.0))
                .collect();
            let data = encode_sym_batch(&residuals, dim, num_bits);
            let encoded = &data.encoded;
            let bin = encoded[SYM_BIN_CODES_COLUMN].as_fixed_size_list();
            let bin_stride = bin.value_length() as usize;
            let bins = bin.values().as_primitive::<UInt8Type>().values();
            let ex = encoded[RABIT_BLOCKED_EX_CODE_COLUMN].as_fixed_size_list();
            let ex_stride = ex.value_length() as usize;
            let exs = ex.values().as_primitive::<UInt8Type>().values();
            let f32_col = |name: &str| encoded[name].as_primitive::<Float32Type>().values();
            let (rho, gamma, unorm, ip_cent) = (
                f32_col(SYM_RHO_COLUMN),
                f32_col(SYM_GAMMA_COLUMN),
                f32_col(SYM_UNORM_COLUMN),
                f32_col(SYM_IP_CENT_COLUMN),
            );
            let factors: Vec<SymFactors> = (0..NUM_BASE)
                .map(|i| SymFactors {
                    rho: rho[i],
                    gamma: gamma[i],
                    unorm: unorm[i],
                    ip_cent: ip_cent[i],
                })
                .collect();

            let mut group = c.benchmark_group(format!("sym_dist_pair_dim{dim}_b{num_bits}"));
            group.throughput(Throughput::Elements((NUM_BASE - 1) as u64));
            group.bench_function("dispatch", |b| {
                b.iter(|| {
                    let mut acc = 0.0f32;
                    for i in 0..NUM_BASE - 1 {
                        acc += sym_dist(
                            &factors[i],
                            &bins[i * bin_stride..(i + 1) * bin_stride],
                            &exs[i * ex_stride..(i + 1) * ex_stride],
                            &factors[i + 1],
                            &bins[(i + 1) * bin_stride..(i + 2) * bin_stride],
                            &exs[(i + 1) * ex_stride..(i + 2) * ex_stride],
                            dim,
                            num_bits,
                            DistanceType::L2,
                        );
                    }
                    black_box(acc)
                })
            });
            group.finish();
        }
    }
}

fn bench_l2_pair(c: &mut Criterion) {
    for dim in WARMUP_DIMS {
        let mut rng = StdRng::seed_from_u64(20260914 + dim as u64);
        let residuals: Vec<f32> = (0..NUM_BASE * dim)
            .map(|_| rng.random_range(-2.0f32..2.0))
            .collect();
        let mut group = c.benchmark_group(format!("l2_pair_dim{dim}"));
        group.throughput(Throughput::Elements((NUM_BASE - 1) as u64));
        group.bench_function("dispatch", |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for i in 0..NUM_BASE - 1 {
                    acc += l2(
                        &residuals[i * dim..(i + 1) * dim],
                        &residuals[(i + 1) * dim..(i + 2) * dim],
                    );
                }
                black_box(acc)
            })
        });
        group.finish();
    }
}

/// HNSW search `distance(id)`: Flat L2 vs Residual 1-bit walk vs RawQuery 5/9.
fn bench_query_scan(c: &mut Criterion) {
    let dim = QUERY_DIM;
    let mut rng = StdRng::seed_from_u64(20260915);
    let residuals: Vec<f32> = (0..NUM_BASE * dim)
        .map(|_| rng.random_range(-2.0f32..2.0))
        .collect();
    let (query, dist_q_c) = random_query(dim, 20260916);
    let query_f32 = query.as_primitive::<Float32Type>().values().to_vec();

    let mut group = c.benchmark_group(format!("query_scan_dim{dim}"));
    group.throughput(Throughput::Elements(NUM_BASE as u64));

    group.bench_function("flat_l2", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for row in residuals.chunks_exact(dim) {
                acc += l2(&query_f32, row);
            }
            black_box(acc)
        })
    });

    let (bins, width) = pack_bins(&residuals, dim);
    let warmup = prepare_residual_warmup_query(&query_f32, const_scaling_factor_3ex(dim));
    group.bench_function("popcount_1bit_warmup", |b| {
        b.iter(|| {
            black_box(warmup_scan(
                &bins,
                width,
                &warmup.planes,
                warmup.delta,
                warmup.vl,
            ))
        })
    });
    group.bench_function("mask_ip_x0_q", |b| {
        b.iter(|| black_box(mask_ip_scan(&bins, width, &query_f32)))
    });

    let perm = shuffled_ids(NUM_BASE);
    for num_bits in QUERY_BITS {
        let walk_store = encode_rq_storage(&residuals, dim, num_bits, true);
        let raw_store = encode_rq_storage(&residuals, dim, num_bits, false);
        let residual_calc = walk_store.dist_calculator(query.clone(), dist_q_c);
        let raw_calc = raw_store.dist_calculator(query.clone(), dist_q_c);
        group.bench_function(format!("hnsw_walk_residual_b{num_bits}"), |b| {
            b.iter(|| black_box(scan_ids(&residual_calc, NUM_BASE)))
        });
        group.bench_function(format!("hnsw_rerank_raw_b{num_bits}"), |b| {
            b.iter(|| black_box(scan_ids(&raw_calc, NUM_BASE)))
        });
        group.bench_function(format!("dist_calculator_setup_residual_b{num_bits}"), |b| {
            b.iter(|| black_box(walk_store.dist_calculator(query.clone(), dist_q_c)))
        });
        group.bench_function(format!("dist_calculator_setup_raw_b{num_bits}"), |b| {
            b.iter(|| black_box(raw_store.dist_calculator(query.clone(), dist_q_c)))
        });
        group.bench_function(format!("hnsw_rerank_raw_b{num_bits}_gather"), |b| {
            b.iter(|| black_box(scan_ids_perm(&raw_calc, &perm)))
        });
        let walk_dists: Vec<OrderedNode> = (0..NUM_BASE as u32)
            .map(|id| OrderedNode::new(id, residual_calc.distance(id).into()))
            .collect();
        group.bench_function(format!("hnsw_rerank_storage_b{num_bits}"), |b| {
            b.iter(|| {
                let mut results = walk_dists.clone();
                walk_store.rerank(query.clone(), dist_q_c, NUM_BASE, &mut results);
                black_box(results)
            })
        });
    }

    group.bench_function("flat_l2_gather", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &id in &perm {
                let start = id as usize * dim;
                acc += l2(&query_f32, &residuals[start..start + dim]);
            }
            black_box(acc)
        })
    });
    group.finish();
}

/// Production graph-build pair path: `from_id` + `dist_between`.
fn bench_from_id_pair(c: &mut Criterion) {
    for dim in [104usize, 128, QUERY_DIM] {
        let mut rng = StdRng::seed_from_u64(20260914 + dim as u64);
        let residuals: Vec<f32> = (0..NUM_BASE * dim)
            .map(|_| rng.random_range(-2.0f32..2.0))
            .collect();
        let storage = encode_rq_storage(&residuals, dim, 5, true);
        let n = storage.len();

        let mut group = c.benchmark_group(format!("from_id_pair_dim{dim}_b5"));
        group.throughput(Throughput::Elements((n - 1) as u64));
        group.bench_function("from_id", |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for i in 0..n - 1 {
                    let calc = storage.dist_calculator_from_id(i as u32);
                    acc += calc.distance((i + 1) as u32);
                }
                black_box(acc)
            })
        });
        group.bench_function("dist_between", |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for i in 0..n - 1 {
                    acc += storage.dist_between(i as u32, (i + 1) as u32);
                }
                black_box(acc)
            })
        });
        group.finish();
    }
}

/// Time `HNSW::index_vectors` on Residual RQ storage (the IVF_HNSW_RQ graph).
fn bench_hnsw_rq_build(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(20260918);
    let residuals: Vec<f32> = (0..GRAPH_N * GRAPH_DIM)
        .map(|_| rng.random_range(-2.0f32..2.0))
        .collect();
    let storage = encode_rq_storage(&residuals, GRAPH_DIM, 5, true);
    let params = HnswBuildParams::default()
        .max_level(4)
        .num_edges(8)
        .ef_construction(32);

    let mut group = c.benchmark_group(format!("hnsw_rq_build_n{GRAPH_N}_dim{GRAPH_DIM}_b5"));
    group.sample_size(10);
    group.bench_function("index_vectors", |b| {
        b.iter(|| black_box(HNSW::index_vectors(&storage, params.clone()).unwrap()))
    });
    group.finish();
}

/// Multi-bit encode, where `RQTransformer` also attaches the five sym columns.
/// `num_bits=1` is the pre-sym baseline.
fn bench_encode_sym_columns(c: &mut Criterion) {
    let dim = 512usize;
    let mut rng = StdRng::seed_from_u64(20260914);
    let residuals: Vec<f32> = (0..NUM_BASE * dim)
        .map(|_| rng.random_range(-2.0f32..2.0))
        .collect();
    for num_bits in [1u8, 5] {
        let data = encode_sym_batch(&residuals, dim, num_bits);
        let mut group = c.benchmark_group(format!("encode_dim{dim}"));
        group.throughput(Throughput::Elements(NUM_BASE as u64));
        group.bench_function(format!("num_bits={num_bits}"), |b| {
            b.iter(|| black_box(data.transformer.transform(black_box(&data.batch)).unwrap()))
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_warmup,
    bench_sym_dist_pair,
    bench_l2_pair,
    bench_query_scan,
    bench_from_id_pair,
    bench_hnsw_rq_build,
    bench_encode_sym_columns
);
criterion_main!(benches);
