// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Measure IVF k-means training time against clustering quality on a big-ann
//! style `.fbin` dataset (`u32` row count, `u32` dimension, then row-major
//! `f32` vectors), such as the 10M subsets of DEEP1B or Text-to-Image1B.
//!
//! Trains on a seeded sample of `sample_rate * k` rows, assigns the whole base
//! to the trained centroids with the exact flat assignment, and reports:
//!
//! - training and assignment wall time,
//! - WCSS (mean squared L2 distance to the assigned centroid) on the base and
//!   on the training sample,
//! - partition-size balance: coefficient of variation, percentiles relative to
//!   the mean size, empty and oversized (> 4x mean) partitions,
//! - per-query scan cost (rows in the `nprobes` nearest partitions) and
//!   recall@10 of the probed partitions against exact ground truth.
//!
//! Run:
//!   cargo run --release -p lance-index --example kmeans_quality -- \
//!     --base /data/T2I/base.10M.fbin --queries /data/T2I/query.public.100K.fbin \
//!     --k 2441 --mode hierarchical --hk 16 --refine-iters 2 --seed 1 --out results.jsonl

#![allow(clippy::print_stdout)]

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use lance_index::vector::kmeans::{
    KMeansAlgoFloat, KMeansParams, compute_partitions_with_dists, kmeans_find_partitions,
    train_kmeans,
};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::l2::l2_distance_batch;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

const TOP_K: usize = 10;

struct Args {
    base: PathBuf,
    queries: PathBuf,
    rows: Option<usize>,
    k: usize,
    sample_rate: usize,
    seed: u64,
    hk: usize,
    max_iters: u32,
    mode: String,
    refine_iters: u32,
    balance_factor: f32,
    nprobes: usize,
    num_queries: usize,
    threads: Option<usize>,
    label: String,
    out: Option<PathBuf>,
    skip_assign: bool,
    stream: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        base: PathBuf::new(),
        queries: PathBuf::new(),
        rows: None,
        k: 2441,
        sample_rate: 256,
        seed: 1,
        hk: 16,
        max_iters: 50,
        mode: "hierarchical".to_string(),
        refine_iters: 0,
        balance_factor: 1.0,
        nprobes: 20,
        num_queries: 1000,
        threads: None,
        label: String::new(),
        out: None,
        skip_assign: false,
        stream: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < argv.len() {
        let value = &argv[i + 1];
        match argv[i].as_str() {
            "--base" => args.base = PathBuf::from(value),
            "--queries" => args.queries = PathBuf::from(value),
            "--rows" => args.rows = Some(value.parse().unwrap()),
            "--k" => args.k = value.parse().unwrap(),
            "--sample-rate" => args.sample_rate = value.parse().unwrap(),
            "--seed" => args.seed = value.parse().unwrap(),
            "--hk" => args.hk = value.parse().unwrap(),
            "--max-iters" => args.max_iters = value.parse().unwrap(),
            "--mode" => args.mode = value.clone(),
            "--refine-iters" => args.refine_iters = value.parse().unwrap(),
            "--balance-factor" => args.balance_factor = value.parse().unwrap(),
            "--nprobes" => args.nprobes = value.parse().unwrap(),
            "--num-queries" => args.num_queries = value.parse().unwrap(),
            "--threads" => args.threads = Some(value.parse().unwrap()),
            "--label" => args.label = value.clone(),
            "--out" => args.out = Some(PathBuf::from(value)),
            "--skip-assign" => {
                args.skip_assign = value == "true";
            }
            "--stream" => {
                args.stream = value == "true";
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(args.base.exists(), "--base {:?} does not exist", args.base);
    assert!(
        args.queries.exists(),
        "--queries {:?} does not exist",
        args.queries
    );
    args
}

/// Rows of the base, either held in memory or streamed from the `.fbin` file in
/// chunks so that a base larger than memory can still be assigned and scored.
enum Base {
    Memory {
        values: Vec<f32>,
        rows: usize,
        dim: usize,
    },
    File {
        path: PathBuf,
        rows: usize,
        dim: usize,
    },
}

const STREAM_CHUNK_ROWS: usize = 200_000;

impl Base {
    fn rows(&self) -> usize {
        match self {
            Self::Memory { rows, .. } | Self::File { rows, .. } => *rows,
        }
    }
    fn dim(&self) -> usize {
        match self {
            Self::Memory { dim, .. } | Self::File { dim, .. } => *dim,
        }
    }
    /// Call `f(start_row, values)` for consecutive chunks covering every row.
    fn for_each_chunk(&self, mut f: impl FnMut(usize, &[f32])) {
        match self {
            Self::Memory { values, rows, dim } => {
                let mut start = 0;
                while start < *rows {
                    let take = STREAM_CHUNK_ROWS.min(rows - start);
                    f(start, &values[start * dim..(start + take) * dim]);
                    start += take;
                }
            }
            Self::File { path, rows, dim } => {
                let mut file = BufReader::with_capacity(8 << 20, File::open(path).unwrap());
                file.seek(SeekFrom::Start(8)).unwrap();
                let mut bytes = vec![0u8; STREAM_CHUNK_ROWS * dim * 4];
                let mut values = vec![0f32; STREAM_CHUNK_ROWS * dim];
                let mut start = 0;
                while start < *rows {
                    let take = STREAM_CHUNK_ROWS.min(rows - start);
                    file.read_exact(&mut bytes[..take * dim * 4]).unwrap();
                    for (v, b) in values[..take * dim].iter_mut().zip(bytes.chunks_exact(4)) {
                        *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    }
                    f(start, &values[..take * dim]);
                    start += take;
                }
            }
        }
    }
    /// Gather the given rows, returned in the order of `rows`.
    fn gather(&self, rows: &[usize]) -> Vec<f32> {
        let dim = self.dim();
        let mut out = vec![0f32; rows.len() * dim];
        let mut order: Vec<(usize, usize)> = rows
            .iter()
            .copied()
            .enumerate()
            .map(|(pos, row)| (row, pos))
            .collect();
        order.sort_unstable();
        let mut next = 0;
        self.for_each_chunk(|start, values| {
            let end = start + values.len() / dim;
            while next < order.len() && order[next].0 < end {
                let (row, pos) = order[next];
                out[pos * dim..(pos + 1) * dim]
                    .copy_from_slice(&values[(row - start) * dim..(row - start + 1) * dim]);
                next += 1;
            }
        });
        out
    }
}

fn fbin_header(path: &Path) -> (usize, usize) {
    let mut file = File::open(path).unwrap();
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();
    (
        u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize,
        u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize,
    )
}

/// Read up to `limit` rows of an `.fbin` file. Returns the flat values, the
/// number of rows read and the dimension.
fn read_fbin(path: &Path, limit: Option<usize>) -> (Vec<f32>, usize, usize) {
    let mut file = BufReader::with_capacity(1 << 20, File::open(path).unwrap());
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();
    let n = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let rows = limit.map_or(n, |limit| limit.min(n));
    let mut values = Vec::with_capacity(rows * d);
    let mut chunk = vec![0u8; 64 << 20];
    let mut remaining = rows * d * 4;
    while remaining > 0 {
        let take = remaining.min(chunk.len());
        file.read_exact(&mut chunk[..take]).unwrap();
        values.extend(
            chunk[..take]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        remaining -= take;
    }
    (values, rows, d)
}

/// Exact top-10 neighbours of every query over the base, cached next to the
/// query file since they do not depend on the clustering.
fn ground_truth(args: &Args, base: &Base, rows: usize, d: usize, queries: &[f32]) -> Vec<u32> {
    let cache = args.queries.with_extension(format!(
        "gt{}.rows{}.top{}.bin",
        args.num_queries, rows, TOP_K
    ));
    if let Ok(mut file) = File::open(&cache) {
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        if bytes.len() == args.num_queries * TOP_K * 4 {
            return bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
        }
    }
    let start = Instant::now();
    let num_queries = queries.len() / d;
    let mut best: Vec<Vec<(f32, u32)>> = vec![Vec::with_capacity(TOP_K + 1); num_queries];
    base.for_each_chunk(|chunk_start, values| {
        queries
            .par_chunks(d)
            .zip(best.par_iter_mut())
            .for_each(|(query, best)| {
                for (i, dist) in l2_distance_batch(query, values, d).enumerate() {
                    if best.len() < TOP_K || dist < best[TOP_K - 1].0 {
                        let pos = best.partition_point(|(other, _)| *other <= dist);
                        best.insert(pos, (dist, (chunk_start + i) as u32));
                        best.truncate(TOP_K);
                    }
                }
            });
    });
    let gt: Vec<u32> = best
        .into_iter()
        .flat_map(|best| best.into_iter().map(|(_, row)| row))
        .collect();
    println!(
        "ground truth for {} queries over {} rows in {:.1}s",
        args.num_queries,
        rows,
        start.elapsed().as_secs_f64()
    );
    let bytes: Vec<u8> = gt.iter().flat_map(|row| row.to_le_bytes()).collect();
    std::fs::write(&cache, bytes).unwrap();
    gt
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn main() {
    let args = parse_args();
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let load_start = Instant::now();
    let base = if args.stream {
        let (rows, dim) = fbin_header(&args.base);
        let rows = args.rows.map_or(rows, |limit| limit.min(rows));
        Base::File {
            path: args.base.clone(),
            rows,
            dim,
        }
    } else {
        let (values, rows, dim) = read_fbin(&args.base, args.rows);
        Base::Memory { values, rows, dim }
    };
    let (rows, d) = (base.rows(), base.dim());
    let (queries, _, qd) = read_fbin(&args.queries, Some(args.num_queries));
    assert_eq!(d, qd, "query dimension differs from base dimension");
    println!(
        "loaded {} x {} base and {} queries in {:.1}s",
        rows,
        d,
        args.num_queries,
        load_start.elapsed().as_secs_f64()
    );

    // Seeded training sample in random order: the trainer takes prefixes of it
    // when it sub-samples, so the order must not carry structure.
    let sample_size = (args.sample_rate * args.k).min(rows);
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let sample_indices = rand::seq::index::sample(&mut rng, rows, sample_size).into_vec();
    let sample = Float32Array::from(base.gather(&sample_indices));

    let hierarchical = match args.mode.as_str() {
        "flat" => false,
        "hierarchical" => true,
        other => panic!("unknown --mode {other}"),
    };
    let params = KMeansParams::new(None, args.max_iters, 1, DistanceType::L2)
        .with_balance_factor(args.balance_factor)
        .with_refine_iters(args.refine_iters)
        .with_seed(args.seed)
        .with_hierarchical_k(if hierarchical { args.hk } else { 1 });

    let train_start = Instant::now();
    let kmeans = train_kmeans::<Float32Type>(&sample, params, d, args.k, args.sample_rate).unwrap();
    let train_secs = train_start.elapsed().as_secs_f64();
    let centroids = kmeans.centroids.as_primitive::<Float32Type>().clone();
    let k = centroids.len() / d;
    println!(
        "trained {} centroids ({} {}, hk={}, refine={}) in {:.1}s",
        k, args.mode, args.k, args.hk, args.refine_iters, train_secs
    );
    if args.skip_assign {
        let result = serde_json::json!({
            "label": args.label, "k": args.k, "mode": args.mode, "hk": args.hk,
            "refine_iters": args.refine_iters, "seed": args.seed, "rows": rows, "dim": d,
            "train_secs": train_secs, "train_only": true,
        });
        println!("{}", serde_json::to_string(&result).unwrap());
        if let Some(out) = &args.out {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(out)
                .unwrap();
            writeln!(file, "{}", serde_json::to_string(&result).unwrap()).unwrap();
        }
        return;
    }

    let assign_start = Instant::now();
    let mut membership: Vec<Option<u32>> = Vec::with_capacity(rows);
    let mut dists: Vec<Option<f32>> = Vec::with_capacity(rows);
    base.for_each_chunk(|_, values| {
        let chunk = Float32Array::from(values.to_vec());
        let (chunk_membership, chunk_dists) = compute_partitions_with_dists::<
            Float32Type,
            KMeansAlgoFloat<Float32Type>,
        >(&centroids, &chunk, d, DistanceType::L2);
        membership.extend(chunk_membership);
        dists.extend(chunk_dists);
    });
    let assign_secs = assign_start.elapsed().as_secs_f64();
    let centroids = centroids.values();

    let mut sizes = vec![0usize; k];
    let mut assigned = 0usize;
    let mut wcss = 0.0f64;
    for (cluster, dist) in membership.iter().zip(dists.iter()) {
        if let (Some(cluster), Some(dist)) = (cluster, dist) {
            sizes[*cluster as usize] += 1;
            assigned += 1;
            wcss += *dist as f64;
        }
    }
    let wcss_base = wcss / assigned as f64;
    let wcss_sample = sample_indices
        .iter()
        .filter_map(|&row| dists[row].map(|dist| dist as f64))
        .sum::<f64>()
        / sample_indices.len() as f64;

    let mean_size = assigned as f64 / k as f64;
    let mut sorted: Vec<f64> = sizes.iter().map(|&s| s as f64 / mean_size).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cv = (sorted.iter().map(|s| (s - 1.0).powi(2)).sum::<f64>() / k as f64).sqrt();
    let empty = sizes.iter().filter(|&&s| s == 0).count();
    let tiny = sorted.iter().filter(|&&s| s < 0.25).count();
    let oversized: Vec<usize> = sizes
        .iter()
        .copied()
        .filter(|&s| s as f64 > 4.0 * mean_size)
        .collect();
    let rows_in_oversized = oversized.iter().sum::<usize>() as f64 / assigned as f64;

    let gt = ground_truth(&args, &base, rows, d, queries.as_slice());
    let max_probes = (args.nprobes * 2).min(k);
    let probed: Vec<Vec<u32>> = queries
        .as_slice()
        .par_chunks(d)
        .map(|query| {
            let (ids, _) =
                kmeans_find_partitions(centroids, query, max_probes, DistanceType::L2).unwrap();
            ids.values().to_vec()
        })
        .collect();
    let mut query_metrics = serde_json::Map::new();
    for nprobes in [args.nprobes / 2, args.nprobes, args.nprobes * 2] {
        let nprobes = nprobes.clamp(1, max_probes);
        let mut scanned: Vec<f64> = Vec::with_capacity(probed.len());
        let mut recall = 0.0f64;
        for (q, ids) in probed.iter().enumerate() {
            let ids = &ids[..nprobes];
            scanned.push(ids.iter().map(|&id| sizes[id as usize] as f64).sum::<f64>());
            let probed_set: HashSet<u32> = ids.iter().copied().collect();
            let hits = gt[q * TOP_K..(q + 1) * TOP_K]
                .iter()
                .filter(|&&row| membership[row as usize].is_some_and(|c| probed_set.contains(&c)))
                .count();
            recall += hits as f64 / TOP_K as f64;
        }
        recall /= probed.len() as f64;
        scanned.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = nprobes as f64 * mean_size;
        query_metrics.insert(
            format!("nprobes_{nprobes}"),
            serde_json::json!({
                "recall_at_10": recall,
                "scan_rows_mean": scanned.iter().sum::<f64>() / scanned.len() as f64,
                "scan_rows_p50": percentile(&scanned, 0.5),
                "scan_rows_p90": percentile(&scanned, 0.9),
                "scan_rows_p99": percentile(&scanned, 0.99),
                "scan_rows_max": scanned.last().copied().unwrap_or(0.0),
                "scan_p99_over_expected": percentile(&scanned, 0.99) / expected,
                "scan_max_over_expected": scanned.last().copied().unwrap_or(0.0) / expected,
            }),
        );
    }

    let result = serde_json::json!({
        "label": args.label,
        "base": args.base,
        "rows": rows,
        "dim": d,
        "k": args.k,
        "k_trained": k,
        "sample_rate": args.sample_rate,
        "sample_size": sample_size,
        "seed": args.seed,
        "mode": args.mode,
        "hk": args.hk,
        "max_iters": args.max_iters,
        "refine_iters": args.refine_iters,
        "balance_factor": args.balance_factor,
        "threads": args.threads.unwrap_or_else(rayon::current_num_threads),
        "stream": args.stream,
        "train_secs": train_secs,
        "assign_secs": assign_secs,
        "train_loss": kmeans.loss,
        "wcss_base": wcss_base,
        "wcss_sample": wcss_sample,
        "size_mean": mean_size,
        "size_cv": cv,
        "size_p50_over_mean": percentile(&sorted, 0.5),
        "size_p90_over_mean": percentile(&sorted, 0.9),
        "size_p99_over_mean": percentile(&sorted, 0.99),
        "size_max_over_mean": sorted.last().copied().unwrap_or(0.0),
        "empty": empty,
        "tiny_under_quarter_mean": tiny,
        "oversized_over_4x_mean": oversized.len(),
        "rows_in_oversized": rows_in_oversized,
        "queries": query_metrics,
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
    if let Some(out) = &args.out {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .unwrap();
        writeln!(file, "{}", serde_json::to_string(&result).unwrap()).unwrap();
    }
}
