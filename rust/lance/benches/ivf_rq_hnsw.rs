// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Same-data Flat `IVF_RQ` vs `IVF_HNSW_RQ`.
//!
//! Setup prints build seconds and recall@k (HNSW vs IVF_RQ, both vs exact L2).
//! Criterion then times search only. Recall is not asserted.
//!
//! ```text
//! cargo bench -p lance --bench ivf_rq_hnsw -- --quick --noplot
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::{Dataset, dataset::WriteMode};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_index::IndexType;
use lance_index::vector::bq::{RQBuildParams, RQRotationType};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_linalg::distance::MetricType;
use lance_linalg::distance::l2::l2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;
const N: usize = 4_096;
const NQ: usize = 32;
const K: usize = 10;
const NLIST: usize = 8;
const NUM_BITS: u8 = 5;
const NPROBE: usize = 4;
const EF: usize = 64;

struct BenchData {
    vectors: Vec<f32>,
    queries: Vec<Vec<f32>>,
    exact: Vec<Vec<u64>>,
}

fn random_data() -> BenchData {
    let mut rng = StdRng::seed_from_u64(20260916);
    let vectors: Vec<f32> = (0..N * DIM)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let queries: Vec<Vec<f32>> = (0..NQ)
        .map(|_| (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let exact = queries
        .iter()
        .map(|query| exact_topk(&vectors, query, K))
        .collect();
    BenchData {
        vectors,
        queries,
        exact,
    }
}

fn exact_topk(base: &[f32], query: &[f32], k: usize) -> Vec<u64> {
    let n = base.len() / DIM;
    let mut scored: Vec<(f32, u64)> = (0..n)
        .map(|i| (l2(&base[i * DIM..(i + 1) * DIM], query), i as u64))
        .collect();
    scored.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
    scored[..k].sort_by(|a, b| a.0.total_cmp(&b.0));
    scored[..k].iter().map(|(_, id)| *id).collect()
}

fn recall_at_k(got: &[u64], want: &[u64]) -> f64 {
    let hits = got.iter().filter(|id| want.contains(id)).count();
    hits as f64 / want.len().max(1) as f64
}

fn write_dataset(rt: &Runtime, uri: &str, vectors: &[f32]) -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            DIM as i32,
        ),
        false,
    )]));
    let fsl =
        FixedSizeListArray::try_new_from_values(Float32Array::from(vectors.to_vec()), DIM as i32)
            .unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
    let reader = RecordBatchIterator::new(std::iter::once(Ok(batch)), schema);
    rt.block_on(Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: N,
            mode: WriteMode::Create,
            ..Default::default()
        }),
    ))
    .unwrap()
}

fn ivf_params() -> IvfBuildParams {
    IvfBuildParams {
        num_partitions: Some(NLIST),
        ..Default::default()
    }
}

fn rq_params() -> RQBuildParams {
    RQBuildParams::with_rotation_type(NUM_BITS, RQRotationType::Fast)
}

fn search_ids(rt: &Runtime, dataset: &Dataset, query: &[f32]) -> Vec<u64> {
    let q = Float32Array::from(query.to_vec());
    let batch = rt
        .block_on(async {
            dataset
                .scan()
                .nearest("vector", &q, K)
                .unwrap()
                .minimum_nprobes(NPROBE)
                .ef(EF)
                .fast_search()
                .try_into_batch()
                .await
        })
        .unwrap();
    batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
}

fn mean_recall(rt: &Runtime, dataset: &Dataset, queries: &[Vec<f32>], truth: &[Vec<u64>]) -> f64 {
    queries
        .iter()
        .zip(truth)
        .map(|(query, want)| recall_at_k(&search_ids(rt, dataset, query), want))
        .sum::<f64>()
        / queries.len() as f64
}

fn overlap_recall(rt: &Runtime, left: &Dataset, right: &Dataset, queries: &[Vec<f32>]) -> f64 {
    queries
        .iter()
        .map(|query| {
            let a = search_ids(rt, left, query);
            let b = search_ids(rt, right, query);
            recall_at_k(&a, &b)
        })
        .sum::<f64>()
        / queries.len() as f64
}

fn bench_ivf_rq_hnsw(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let data = random_data();
    let tmp = tempfile::tempdir().unwrap();
    let rq_uri = tmp.path().join("ivf_rq").to_string_lossy().into_owned();
    let hnsw_uri = tmp
        .path()
        .join("ivf_hnsw_rq")
        .to_string_lossy()
        .into_owned();

    let mut rq_ds = write_dataset(&rt, &rq_uri, &data.vectors);
    let mut hnsw_ds = write_dataset(&rt, &hnsw_uri, &data.vectors);

    let rq_index = VectorIndexParams::with_ivf_rq_params(MetricType::L2, ivf_params(), rq_params());
    let hnsw_index = VectorIndexParams::with_ivf_hnsw_rq_params(
        MetricType::L2,
        ivf_params(),
        HnswBuildParams::default()
            .max_level(5)
            .num_edges(16)
            .ef_construction(64),
        rq_params(),
    );

    let t0 = Instant::now();
    rt.block_on(rq_ds.create_index(
        &["vector"],
        IndexType::Vector,
        Some("ivf_rq".to_string()),
        &rq_index,
        false,
    ))
    .unwrap();
    let rq_build = t0.elapsed();

    let t0 = Instant::now();
    rt.block_on(hnsw_ds.create_index(
        &["vector"],
        IndexType::Vector,
        Some("ivf_hnsw_rq".to_string()),
        &hnsw_index,
        false,
    ))
    .unwrap();
    let hnsw_build = t0.elapsed();

    let rq_recall = mean_recall(&rt, &rq_ds, &data.queries, &data.exact);
    let hnsw_recall = mean_recall(&rt, &hnsw_ds, &data.queries, &data.exact);
    let hnsw_vs_rq = overlap_recall(&rt, &hnsw_ds, &rq_ds, &data.queries);

    println!(
        "task9 n={N} dim={DIM} nlist={NLIST} bits={NUM_BITS} nq={NQ} k={K} nprobe={NPROBE} ef={EF}"
    );
    println!(
        "IVF_RQ      build={:.3}s  recall@{K} vs exact={rq_recall:.4}",
        rq_build.as_secs_f64()
    );
    println!(
        "IVF_HNSW_RQ build={:.3}s  recall@{K} vs exact={hnsw_recall:.4}  vs IVF_RQ={hnsw_vs_rq:.4}",
        hnsw_build.as_secs_f64()
    );

    let mut group = c.benchmark_group("ivf_rq_hnsw_search");
    group.throughput(Throughput::Elements(NQ as u64));
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    group.bench_function("ivf_rq", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            for query in &data.queries {
                rows += search_ids(&rt, &rq_ds, query).len();
            }
            std::hint::black_box(rows)
        })
    });
    group.bench_function("ivf_hnsw_rq", |b| {
        b.iter(|| {
            let mut rows = 0usize;
            for query in &data.queries {
                rows += search_ids(&rt, &hnsw_ds, query).len();
            }
            std::hint::black_box(rows)
        })
    });
    group.finish();
}

criterion_group!(benches, bench_ivf_rq_hnsw);
criterion_main!(benches);
