// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare hierarchical k-means with Lloyd nodes (Lance's default) against
//! hierarchical k-means with e-means nodes on DINO10B-style `.bvecs` data
//! (per row: a little-endian `u32` dimension, then one `u8` per component).
//!
//! With `--pca` and `--dim`, every vector is centered and projected onto its
//! leading `dim` principal components (from `pca.py`) as it loads, which
//! varies the dimension while keeping the data real.
//!
//! Trains through `train_kmeans` with the parameters Lance's IVF build uses
//! (50 Lloyd iterations, balance factor 1, fan-out 16, `sample_rate * k`
//! training rows), then reports as one JSON line:
//!
//! - training wall time and process CPU time,
//! - validation MSE over the held-out queries and training MSE over a prefix
//!   of the training rows, with the per-query distances optionally dumped for
//!   a paired comparison between runs,
//! - with the `kmeans-profile` feature, the time of every training phase and
//!   the busy time of every kind of work (see `vector::kmeans::profile`),
//! - with the feature, the number of busy threads sampled every millisecond of
//!   training, which shows when threads sit idle,
//! - the cost of an empty rayon fork-join on this machine, and the speedup an
//!   embarrassingly parallel distance computation reaches on all threads: the
//!   ceiling set by the hardware (for example efficiency cores) rather than by
//!   the trainer.
//!
//! Run:
//!   cargo run --release -p lance-index --features kmeans-profile --example emeans_bench -- \
//!     --base base_8M.bvecs --queries queries_clean.bvecs --rows 4194304 --k 16384 \
//!     --node-algorithm emeans --threads 18 --seed 1 --out results.jsonl

#![allow(clippy::print_stdout)]

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use lance_index::vector::kmeans::emeans::{BatchSize, EMeansParams};
use lance_index::vector::kmeans::profile;
use lance_index::vector::kmeans::{
    HierarchicalNodeAlgorithm, KMeansAlgoFloat, KMeansParams, compute_partitions_with_dists,
    train_kmeans,
};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::l2::l2_distance_batch_into;
use lance_linalg::kernels::argmin_value_float;
use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use serde_json::json;

struct Args {
    base: PathBuf,
    queries: PathBuf,
    rows: usize,
    k: usize,
    sample_rate: usize,
    hk: usize,
    max_iters: u32,
    balance_factor: f32,
    node_algorithm: String,
    epochs: u32,
    batch_scale: f64,
    pca: Option<PathBuf>,
    dim: Option<usize>,
    threads: usize,
    seed: u64,
    num_queries: usize,
    train_eval_rows: usize,
    label: String,
    out: Option<PathBuf>,
    dump_query_dists: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut args = Args {
        base: PathBuf::new(),
        queries: PathBuf::new(),
        rows: 1 << 22,
        k: 1 << 14,
        sample_rate: 256,
        hk: 16,
        max_iters: 50,
        balance_factor: 1.0,
        node_algorithm: "lloyd".to_string(),
        epochs: EMeansParams::default().epochs,
        batch_scale: 1.0,
        pca: None,
        dim: None,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        seed: 1,
        num_queries: 100_000,
        train_eval_rows: 100_000,
        label: String::new(),
        out: None,
        dump_query_dists: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        let [flag, value] = pair else {
            panic!("flag {:?} has no value", pair[0]);
        };
        match flag.as_str() {
            "--base" => args.base = PathBuf::from(value),
            "--queries" => args.queries = PathBuf::from(value),
            "--rows" => args.rows = value.parse().unwrap(),
            "--k" => args.k = value.parse().unwrap(),
            "--sample-rate" => args.sample_rate = value.parse().unwrap(),
            "--hk" => args.hk = value.parse().unwrap(),
            "--max-iters" => args.max_iters = value.parse().unwrap(),
            "--balance-factor" => args.balance_factor = value.parse().unwrap(),
            "--node-algorithm" => args.node_algorithm = value.clone(),
            "--epochs" => args.epochs = value.parse().unwrap(),
            "--batch-scale" => args.batch_scale = value.parse().unwrap(),
            "--pca" => args.pca = Some(PathBuf::from(value)),
            "--dim" => args.dim = Some(value.parse().unwrap()),
            "--threads" => args.threads = value.parse().unwrap(),
            "--seed" => args.seed = value.parse().unwrap(),
            "--num-queries" => args.num_queries = value.parse().unwrap(),
            "--train-eval-rows" => args.train_eval_rows = value.parse().unwrap(),
            "--label" => args.label = value.clone(),
            "--out" => args.out = Some(PathBuf::from(value)),
            "--dump-query-dists" => args.dump_query_dists = Some(PathBuf::from(value)),
            other => panic!("unknown flag {other}"),
        }
    }
    if args.label.is_empty() {
        args.label = format!("{}-t{}-s{}", args.node_algorithm, args.threads, args.seed);
    }
    args
}

/// Centering and projection onto the leading principal components.
struct Projection {
    /// Mean of the source vectors, `source_dim` values.
    mean: Vec<f32>,
    /// The leading `dim` principal components, one row of `source_dim` values
    /// each, in decreasing order of variance.
    components: Vec<f32>,
    dim: usize,
}

impl Projection {
    /// Read the leading `dim` components of a `pca.py` file: the mean, then
    /// every component, all `source_dim` little-endian `f32` values long.
    fn read(path: &PathBuf, source_dim: usize, dim: usize) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(
            values.len(),
            (source_dim + 1) * source_dim,
            "{path:?} does not hold a {source_dim}-dimensional PCA"
        );
        assert!(
            dim <= source_dim,
            "--dim {dim} exceeds the source dimension {source_dim}"
        );
        Self {
            mean: values[..source_dim].to_vec(),
            components: values[source_dim..source_dim * (dim + 1)].to_vec(),
            dim,
        }
    }
}

/// Rows read and converted at a time.
const READ_CHUNK_ROWS: usize = 65_536;
/// Rows projected by one task: one GEMM of this many rows.
const PROJECT_BLOCK_ROWS: usize = 256;

/// Read the first `rows` vectors of a `.bvecs` file as `f32`, projected when
/// `projection` is given. Returns the values and their dimension.
fn read_bvecs(path: &PathBuf, rows: usize, projection: Option<&Projection>) -> (Vec<f32>, usize) {
    let mut file = BufReader::with_capacity(64 << 20, File::open(path).unwrap());
    let mut header = [0u8; 4];
    file.read_exact(&mut header).unwrap();
    let source_dim = u32::from_le_bytes(header) as usize;
    file.seek_relative(-4).unwrap();
    let dim = projection.map_or(source_dim, |p| p.dim);
    let row_bytes = 4 + source_dim;
    let mut values = vec![0f32; rows * dim];
    let mut bytes = vec![0u8; READ_CHUNK_ROWS * row_bytes];
    for (chunk_index, out) in values.chunks_mut(READ_CHUNK_ROWS * dim).enumerate() {
        let chunk_rows = out.len() / dim;
        let bytes = &mut bytes[..chunk_rows * row_bytes];
        file.read_exact(bytes).unwrap();
        for (i, row) in bytes.chunks_exact(row_bytes).enumerate() {
            let header = u32::from_le_bytes(row[..4].try_into().unwrap()) as usize;
            assert_eq!(
                header,
                source_dim,
                "row {} header",
                chunk_index * READ_CHUNK_ROWS + i
            );
        }
        let bytes = &*bytes;
        match projection {
            None => out
                .par_chunks_mut(source_dim)
                .zip(bytes.par_chunks_exact(row_bytes))
                .for_each(|(out, row)| {
                    for (v, &b) in out.iter_mut().zip(&row[4..]) {
                        *v = f32::from(b);
                    }
                }),
            Some(projection) => out
                .par_chunks_mut(PROJECT_BLOCK_ROWS * dim)
                .zip(bytes.par_chunks(PROJECT_BLOCK_ROWS * row_bytes))
                .for_each(|(out, block)| {
                    let block_rows = out.len() / dim;
                    let mut centered = Vec::with_capacity(block_rows * source_dim);
                    for row in block.chunks_exact(row_bytes) {
                        centered.extend(
                            row[4..]
                                .iter()
                                .zip(&projection.mean)
                                .map(|(&b, &m)| f32::from(b) - m),
                        );
                    }
                    general_mat_mul(
                        1.0,
                        &ArrayView2::from_shape((block_rows, source_dim), &centered).unwrap(),
                        &ArrayView2::from_shape((dim, source_dim), &projection.components)
                            .unwrap()
                            .t(),
                        0.0,
                        &mut ArrayViewMut2::from_shape((block_rows, dim), out).unwrap(),
                    );
                }),
        }
    }
    (values, dim)
}

/// User plus system CPU time of this process, in seconds. Includes the time
/// idle rayon threads spin while looking for work.
fn process_cpu_secs() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid, writable `rusage` for getrusage to fill.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(ret, 0, "getrusage failed");
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    secs(usage.ru_utime) + secs(usage.ru_stime)
}

/// Squared L2 distance of every vector to its nearest centroid.
fn nearest_distances(centroids: &Float32Array, vectors: &[f32], dim: usize) -> Vec<f32> {
    let (_, dists) = compute_partitions_with_dists::<Float32Type, KMeansAlgoFloat<Float32Type>>(
        centroids,
        &Float32Array::from(vectors.to_vec()),
        dim,
        DistanceType::L2,
    );
    dists.into_iter().map(|d| d.unwrap_or(f32::NAN)).collect()
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64
}

/// Wall seconds of one empty fork-join of `tasks` tasks on the global pool,
/// the median over repetitions. Prices a parallel region's dispatch and
/// collection with no work to hide them behind.
fn empty_region_secs(tasks: usize) -> f64 {
    let items: Vec<u64> = (0..tasks as u64).collect();
    let mut samples: Vec<f64> = (0..2_000)
        .map(|_| {
            let start = Instant::now();
            let sum: u64 = items
                .par_iter()
                .with_max_len(1)
                .map(|&i| std::hint::black_box(i))
                .sum();
            std::hint::black_box(sum);
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// How long one sample of the busy-thread count lasts.
const BUSY_SAMPLE_PERIOD: Duration = Duration::from_millis(1);

/// Samples [`profile::busy_threads`] until stopped. Returns the samples.
fn sample_busy_threads(stop: Arc<AtomicBool>) -> std::thread::JoinHandle<Vec<u16>> {
    std::thread::spawn(move || {
        let mut samples = Vec::new();
        while !stop.load(Ordering::Relaxed) {
            samples.push(profile::busy_threads() as u16);
            std::thread::sleep(BUSY_SAMPLE_PERIOD);
        }
        samples
    })
}

/// Speedup of the global pool over one thread on independent nearest-centroid
/// searches (`dim` components, 16 centroids, 512 vectors per task): no
/// sharing, no sequential part and negligible scheduling, so what it loses to
/// perfect scaling is the hardware's doing.
fn ceiling_speedup(dim: usize) -> f64 {
    let vectors: Vec<f32> = (0..65_536 * dim).map(|i| (i % 251) as f32).collect();
    let centroids: Vec<f32> = (0..16 * dim).map(|i| (i % 241) as f32).collect();
    let search = || {
        vectors
            .par_chunks(512 * dim)
            .map(|chunk| {
                let mut distances = [0.0f32; 16];
                chunk
                    .chunks_exact(dim)
                    .map(|vector| {
                        l2_distance_batch_into(vector, &centroids, dim, &mut distances);
                        argmin_value_float(distances.iter().copied()).map_or(0, |(c, _)| c as u64)
                    })
                    .sum::<u64>()
            })
            .sum::<u64>()
    };
    let best_of = |run: &dyn Fn() -> u64| {
        (0..5)
            .map(|_| {
                let start = Instant::now();
                std::hint::black_box(run());
                start.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| best_of(&search));
    single / best_of(&search)
}

fn main() {
    let args = parse_args();
    // The projection runs one GEMM per rayon task; keep matrixmultiply from
    // handing each to its own small pool. Training is unaffected: e-means
    // tiles stay below the size at which matrixmultiply threads at all.
    // SAFETY: no other thread exists yet to read the environment.
    unsafe { std::env::set_var("MATMUL_NUM_THREADS", "1") };
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    let load_start = Instant::now();
    let projection = args.pca.as_ref().map(|path| {
        let mut header = [0u8; 4];
        File::open(&args.base)
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        let source_dim = u32::from_le_bytes(header) as usize;
        Projection::read(path, source_dim, args.dim.unwrap_or(source_dim))
    });
    let (base, dim) = read_bvecs(&args.base, args.rows, projection.as_ref());
    let (queries, query_dim) = read_bvecs(&args.queries, args.num_queries, projection.as_ref());
    assert_eq!(
        dim, query_dim,
        "query dimension differs from base dimension"
    );
    println!(
        "loaded {} x {dim} base and {} queries in {:.1}s",
        args.rows,
        args.num_queries,
        load_start.elapsed().as_secs_f64()
    );

    let node_algorithm = match args.node_algorithm.as_str() {
        "lloyd" => HierarchicalNodeAlgorithm::Lloyd,
        "emeans" => HierarchicalNodeAlgorithm::EMeans(EMeansParams {
            epochs: args.epochs,
            batch_size: BatchSize::Law {
                scale: args.batch_scale,
            },
            ..Default::default()
        }),
        other => panic!("unknown --node-algorithm {other}"),
    };
    let params = KMeansParams::new(None, args.max_iters, 1, DistanceType::L2)
        .with_balance_factor(args.balance_factor)
        .with_hierarchical_k(args.hk)
        .with_hierarchical_node_algorithm(node_algorithm)
        .with_seed(args.seed);
    let training = Float32Array::from(base);

    profile::reset();
    let stop_sampling = Arc::new(AtomicBool::new(false));
    let sampler = sample_busy_threads(stop_sampling.clone());
    let cpu_start = process_cpu_secs();
    let train_start = Instant::now();
    let kmeans =
        train_kmeans::<Float32Type>(&training, params, dim, args.k, args.sample_rate).unwrap();
    let train_secs = train_start.elapsed().as_secs_f64();
    let train_cpu_secs = process_cpu_secs() - cpu_start;
    let profile = profile::snapshot();
    stop_sampling.store(true, Ordering::Relaxed);
    let busy_samples = sampler.join().unwrap();
    println!(
        "{}: trained {} centroids in {train_secs:.1}s ({train_cpu_secs:.1}s CPU)",
        args.label, args.k
    );

    let centroids = kmeans.centroids.as_primitive::<Float32Type>().clone();
    let query_dists = nearest_distances(&centroids, &queries, dim);
    let train_eval_rows = args.train_eval_rows.min(args.rows);
    let train_dists =
        nearest_distances(&centroids, &training.values()[..train_eval_rows * dim], dim);
    let val_mse = mean(&query_dists);
    let train_mse = mean(&train_dists);
    println!(
        "{}: validation MSE {val_mse:.2}, training MSE {train_mse:.2}",
        args.label
    );
    if let Some(path) = &args.dump_query_dists {
        let bytes: Vec<u8> = query_dists.iter().flat_map(|d| d.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }

    // Share of samples at each busy-thread count, and a timeline averaged over
    // at most 2,000 windows.
    let mut busy_histogram = vec![0u64; args.threads + 1];
    for &busy in &busy_samples {
        busy_histogram[(busy as usize).min(args.threads)] += 1;
    }
    let window = busy_samples.len().div_ceil(2_000).max(1);
    let busy_timeline: Vec<f64> = busy_samples
        .chunks(window)
        .map(|chunk| chunk.iter().map(|&b| f64::from(b)).sum::<f64>() / chunk.len() as f64)
        .collect();
    let empty_region = json!({
        "tasks_1": empty_region_secs(1),
        "tasks_8": empty_region_secs(8),
        "tasks_64": empty_region_secs(64),
    });
    let phases: serde_json::Map<String, serde_json::Value> = profile
        .phases
        .iter()
        .map(|(phase, total)| {
            (
                phase.name().to_string(),
                json!({ "secs": total.secs, "calls": total.calls, "first_secs": total.first_secs }),
            )
        })
        .collect();
    let work: serde_json::Map<String, serde_json::Value> = profile
        .work
        .iter()
        .map(|(work, total)| {
            (
                work.name().to_string(),
                json!({ "secs": total.secs, "calls": total.calls, "first_secs": total.first_secs }),
            )
        })
        .collect();
    let result = json!({
        "label": args.label,
        "node_algorithm": args.node_algorithm,
        "epochs": args.epochs,
        "batch_scale": args.batch_scale,
        "pca": args.pca.is_some(),
        "rows": args.rows,
        "k": args.k,
        "dim": dim,
        "hk": args.hk,
        "max_iters": args.max_iters,
        "balance_factor": args.balance_factor,
        "sample_rate": args.sample_rate,
        "threads": args.threads,
        "seed": args.seed,
        "train_secs": train_secs,
        "train_cpu_secs": train_cpu_secs,
        "val_mse": val_mse,
        "train_mse": train_mse,
        "num_queries": args.num_queries,
        "train_eval_rows": train_eval_rows,
        "profiled": cfg!(feature = "kmeans-profile"),
        "phases": phases,
        "work": work,
        "empty_region_secs": empty_region,
        "ceiling_speedup": ceiling_speedup(dim),
        "busy_histogram": busy_histogram,
        "busy_timeline": busy_timeline,
        "busy_timeline_window_secs": train_secs / busy_timeline.len().max(1) as f64,
    });
    let line = serde_json::to_string(&result).unwrap();
    println!("{line}");
    if let Some(out) = &args.out {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .unwrap();
        writeln!(file, "{line}").unwrap();
    }
}
