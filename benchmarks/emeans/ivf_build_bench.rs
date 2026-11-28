// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Times full IVF_PQ index builds on DINO10B and checks their recall.
//!
//! `write` turns the first rows of a DINO10B `.bvecs` file, optionally
//! projected onto its leading principal components, into a Lance dataset with
//! an `id` column, and stores held-out queries with their exact 100 nearest
//! rows. `build` builds an IVF_PQ index on that dataset, records when each
//! build stage starts and ends, measures recall@10 against the exact
//! neighbours and drops the index, so every build starts from the same
//! dataset.
//!
//! The file builds as an example of the `lance` crate both on this branch and
//! on `main`: copy it to `rust/lance/examples/`, and for `main` delete the
//! lines between the `kmeans_algorithm` markers, which name a field `main`
//! lacks.

#![allow(clippy::print_stdout)]

use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, IndexBuildProgress};
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::vector::ivf::IvfBuildParams;
// kmeans_algorithm: begin
use lance_index::vector::kmeans::HierarchicalNodeAlgorithm;
// kmeans_algorithm: end
use lance_index::vector::pq::PQBuildParams;
use lance_linalg::distance::MetricType;
use lance_linalg::distance::dot::dot;
use lance_linalg::distance::l2::l2_distance_batch_into;
use rayon::prelude::*;
use serde_json::json;

const INDEX_NAME: &str = "vector_idx";
/// Exact neighbours stored per query.
const GT_K: usize = 100;
/// Neighbours a recall query asks for.
const RECALL_K: usize = 10;
/// Rows read and converted at a time.
const READ_CHUNK_ROWS: usize = 65_536;
/// Rows per batch handed to the dataset writer.
const WRITE_BATCH_ROWS: usize = 65_536;
/// Values of the base rows that one exact-neighbour task scores every query
/// against, sized to stay in a core's L2 cache.
const GT_BLOCK_VALUES: usize = 65_536;
/// Log lines kept per build.
const MAX_LOG_LINES: usize = 10_000;
/// Recall queries in flight at once.
const QUERY_CONCURRENCY: usize = 16;

/// `--flag value` pairs after the command.
struct Flags(Vec<(String, String)>);

impl Flags {
    fn parse(argv: &[String], allowed: &[&str]) -> Self {
        let pairs = argv
            .chunks(2)
            .map(|pair| {
                let [flag, value] = pair else {
                    panic!("flag {:?} has no value", pair[0]);
                };
                let name = flag
                    .strip_prefix("--")
                    .unwrap_or_else(|| panic!("expected a --flag, got {flag:?}"));
                assert!(
                    allowed.contains(&name),
                    "unknown flag --{name}; expected one of {allowed:?}"
                );
                (name.to_string(), value.clone())
            })
            .collect();
        Self(pairs)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(flag, _)| flag == name)
            .map(|(_, value)| value.as_str())
    }

    fn required(&self, name: &str) -> &str {
        self.get(name)
            .unwrap_or_else(|| panic!("--{name} is required"))
    }

    fn parse_or<T: FromStr>(&self, name: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.get(name).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|e| panic!("--{name} {value}: {e:?}"))
        })
    }

    fn list<T: FromStr>(&self, name: &str, default: &str) -> Vec<T>
    where
        T::Err: std::fmt::Debug,
    {
        self.get(name)
            .unwrap_or(default)
            .split(',')
            .map(|value| {
                value
                    .parse()
                    .unwrap_or_else(|e| panic!("--{name} {value}: {e:?}"))
            })
            .collect()
    }
}

/// Centering and projection onto the leading principal components.
struct Projection {
    /// Mean of the source vectors, `source_dim` values.
    mean: Vec<f32>,
    /// The leading `dim` principal components, one row of `source_dim` values
    /// each.
    components: Vec<f32>,
    dim: usize,
}

impl Projection {
    /// Read the leading `dim` components of a `pca.py` file: the mean, then
    /// every component, all `source_dim` little-endian `f32` values long.
    fn read(path: &Path, source_dim: usize, dim: usize) -> Self {
        let values = read_f32s(path);
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

    fn project(&self, row: &[u8], centered: &mut Vec<f32>, out: &mut [f32]) {
        centered.clear();
        centered.extend(row.iter().zip(&self.mean).map(|(&b, &m)| f32::from(b) - m));
        for (value, component) in out
            .iter_mut()
            .zip(self.components.chunks_exact(self.mean.len()))
        {
            *value = dot(&centered[..], component);
        }
    }
}

fn read_f32s(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn source_dim_of(path: &Path) -> usize {
    let mut header = [0u8; 4];
    File::open(path).unwrap().read_exact(&mut header).unwrap();
    u32::from_le_bytes(header) as usize
}

/// Read the first `rows` vectors of a `.bvecs` file as `f32`, projected when
/// `projection` is given. Returns the values and their dimension.
fn read_bvecs(path: &Path, rows: usize, projection: Option<&Projection>) -> (Vec<f32>, usize) {
    let source_dim = source_dim_of(path);
    let mut file = BufReader::with_capacity(64 << 20, File::open(path).unwrap());
    let dim = projection.map_or(source_dim, |p| p.dim);
    let row_bytes = 4 + source_dim;
    let mut values = vec![0f32; rows * dim];
    let mut bytes = vec![0u8; READ_CHUNK_ROWS * row_bytes];
    for (chunk_index, out) in values.chunks_mut(READ_CHUNK_ROWS * dim).enumerate() {
        let chunk_rows = out.len() / dim;
        let bytes = &mut bytes[..chunk_rows * row_bytes];
        file.read_exact(bytes).unwrap_or_else(|e| {
            panic!("{path:?} holds fewer than {rows} rows: {e}");
        });
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
        out.par_chunks_mut(dim)
            .zip(bytes.par_chunks_exact(row_bytes))
            .for_each_init(Vec::new, |centered, (out, row)| match projection {
                None => {
                    for (v, &b) in out.iter_mut().zip(&row[4..]) {
                        *v = f32::from(b);
                    }
                }
                Some(projection) => projection.project(&row[4..], centered, out),
            });
    }
    (values, dim)
}

/// Pushes `entry` if it is among the `GT_K` smallest seen.
fn push_bounded(heap: &mut BinaryHeap<(u32, u32)>, entry: (u32, u32)) {
    if heap.len() < GT_K {
        heap.push(entry);
    } else if entry < *heap.peek().unwrap() {
        heap.pop();
        heap.push(entry);
    }
}

/// The `GT_K` rows nearest to every query under L2, nearest first. A heap
/// orders `(distance bits, row)` pairs, which sort like the distances since
/// squared distances are never negative.
fn exact_neighbours(values: &[f32], queries: &[f32], dim: usize) -> Vec<u32> {
    let num_queries = queries.len() / dim;
    let block_rows = (GT_BLOCK_VALUES / dim).max(1);
    let empty_heaps = || vec![BinaryHeap::with_capacity(GT_K + 1); num_queries];
    let heaps = values
        .par_chunks(block_rows * dim)
        .enumerate()
        .fold(
            || (empty_heaps(), vec![0f32; block_rows]),
            |(mut heaps, mut distances), (block_index, block)| {
                let first_row = block_index * block_rows;
                let distances_in_block = &mut distances[..block.len() / dim];
                for (heap, query) in heaps.iter_mut().zip(queries.chunks_exact(dim)) {
                    l2_distance_batch_into(query, block, dim, distances_in_block);
                    for (offset, &distance) in distances_in_block.iter().enumerate() {
                        push_bounded(heap, (distance.to_bits(), (first_row + offset) as u32));
                    }
                }
                (heaps, distances)
            },
        )
        .map(|(heaps, _)| heaps)
        .reduce(empty_heaps, |mut merged, heaps| {
            for (merged, heap) in merged.iter_mut().zip(heaps) {
                for entry in heap {
                    push_bounded(merged, entry);
                }
            }
            merged
        });
    heaps
        .into_iter()
        .flat_map(|heap| heap.into_sorted_vec().into_iter().map(|(_, row)| row))
        .collect()
}

fn write_le<T: Copy>(path: &Path, values: &[T], to_bytes: impl Fn(T) -> [u8; 4]) {
    let bytes: Vec<u8> = values.iter().flat_map(|&v| to_bytes(v)).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

async fn write(flags: &Flags) {
    let base = PathBuf::from(flags.required("base"));
    let queries_path = PathBuf::from(flags.required("queries"));
    let rows: usize = flags.parse_or("rows", 1 << 20);
    let num_queries: usize = flags.parse_or("num-queries", 1_000);
    let dir = PathBuf::from(flags.required("dir"));
    let projection = flags.get("pca").map(|path| {
        let source_dim = source_dim_of(&base);
        Projection::read(
            Path::new(path),
            source_dim,
            flags.parse_or("dim", source_dim),
        )
    });
    std::fs::create_dir_all(&dir).unwrap();

    let start = Instant::now();
    let (values, dim) = read_bvecs(&base, rows, projection.as_ref());
    let (queries, _) = read_bvecs(&queries_path, num_queries, projection.as_ref());
    println!(
        "loaded {rows} x {dim} rows and {num_queries} queries in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let neighbours = exact_neighbours(&values, &queries, dim);
    println!(
        "found the exact neighbours in {:.1}s",
        start.elapsed().as_secs_f64()
    );
    write_le(&dir.join("queries.f32"), &queries, f32::to_le_bytes);
    write_le(&dir.join("neighbours.u32"), &neighbours, u32::to_le_bytes);

    let start = Instant::now();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));
    let values = Arc::new(values);
    let batch_schema = schema.clone();
    let batches = (0..rows.div_ceil(WRITE_BATCH_ROWS)).map(move |batch_index| {
        let first = batch_index * WRITE_BATCH_ROWS;
        let end = (first + WRITE_BATCH_ROWS).min(rows);
        let ids = UInt64Array::from_iter_values(first as u64..end as u64);
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(values[first * dim..end * dim].to_vec()),
            dim as i32,
        )?;
        RecordBatch::try_new(batch_schema.clone(), vec![Arc::new(ids), Arc::new(vectors)])
    });
    let reader = RecordBatchIterator::new(batches, schema);
    let uri = dir.join("data.lance");
    let dataset = Dataset::write(reader, uri.to_str().unwrap(), Some(WriteParams::default()))
        .await
        .unwrap();
    println!(
        "wrote {} rows in {} fragments in {:.1}s",
        dataset.count_rows(None).await.unwrap(),
        dataset.get_fragments().len(),
        start.elapsed().as_secs_f64()
    );
    let meta = json!({
        "rows": rows,
        "dim": dim,
        "num_queries": num_queries,
        "gt_k": GT_K,
        "pca": projection.is_some(),
    });
    std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
}

/// When each build stage starts and completes.
#[derive(Debug, Default)]
struct StageClock {
    events: Mutex<Vec<(String, &'static str, Instant)>>,
}

impl StageClock {
    fn record(&self, stage: &str, event: &'static str) {
        self.events
            .lock()
            .unwrap()
            .push((stage.to_string(), event, Instant::now()));
    }
}

#[async_trait::async_trait]
impl IndexBuildProgress for StageClock {
    async fn stage_start(
        &self,
        stage: &str,
        _total: Option<u64>,
        _unit: &str,
    ) -> lance::Result<()> {
        self.record(stage, "start");
        Ok(())
    }

    async fn stage_progress(&self, _stage: &str, _completed: u64) -> lance::Result<()> {
        Ok(())
    }

    async fn stage_complete(&self, stage: &str) -> lance::Result<()> {
        self.record(stage, "complete");
        Ok(())
    }
}

/// Keeps Lance's info log lines with the time they were logged, which splits
/// the training stages into sampling and training.
struct BuildLog {
    lines: Mutex<Vec<(Instant, String, String)>>,
}

impl log::Log for BuildLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info && metadata.target().starts_with("lance")
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let message = record.args().to_string();
        // Every k-means node logs its iterations; they would crowd the stage
        // markers out of the kept lines.
        if !message.starts_with("KMeans training: ") {
            let mut lines = self.lines.lock().unwrap();
            if lines.len() < MAX_LOG_LINES {
                lines.push((Instant::now(), record.target().to_string(), message));
            }
        }
    }

    fn flush(&self) {}
}

static BUILD_LOG: BuildLog = BuildLog {
    lines: Mutex::new(Vec::new()),
};

/// User plus system CPU time of this process, in seconds, and its peak
/// resident set size in bytes.
fn process_usage() -> (f64, u64) {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid, writable `rusage` for getrusage to fill.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(ret, 0, "getrusage failed");
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    // Linux reports the peak in KiB, macOS in bytes.
    let rss_unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
    (
        secs(usage.ru_utime) + secs(usage.ru_stime),
        usage.ru_maxrss as u64 * rss_unit,
    )
}

/// `CLOCK_MONOTONIC` in seconds, the clock `perf record -k mono` stamps
/// samples with, so a profile can be split at the build's stage boundaries.
fn monotonic_secs() -> f64 {
    let mut now: libc::timespec = unsafe { std::mem::zeroed() };
    // SAFETY: `now` is a valid, writable `timespec` for clock_gettime to fill.
    let ret = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    assert_eq!(ret, 0, "clock_gettime failed");
    now.tv_sec as f64 + now.tv_nsec as f64 * 1e-9
}

/// Recall@10 of the index over the stored queries with `nprobes` partitions
/// and, when `refine` is positive, that refine factor.
async fn recall(
    dataset: &Dataset,
    queries: &[f32],
    neighbours: &[u32],
    dim: usize,
    nprobes: usize,
    refine: u32,
) -> f64 {
    let num_queries = queries.len() / dim;
    let hits: usize = futures::stream::iter(0..num_queries)
        .map(|query_index| async move {
            let query = Float32Array::from(queries[query_index * dim..][..dim].to_vec());
            let mut scan = dataset.scan();
            scan.nearest("vector", &query, RECALL_K)
                .unwrap()
                .nprobes(nprobes);
            if refine > 0 {
                scan.refine(refine);
            }
            scan.project(&["id"]).unwrap();
            let batch = scan.try_into_batch().await.unwrap();
            let truth = &neighbours[query_index * GT_K..][..RECALL_K];
            batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .filter(|&&id| truth.contains(&(id as u32)))
                .count()
        })
        .buffer_unordered(QUERY_CONCURRENCY)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .sum();
    hits as f64 / (num_queries * RECALL_K) as f64
}

async fn build(flags: &Flags) {
    let dir = PathBuf::from(flags.required("dir"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let dim = meta["dim"].as_u64().unwrap() as usize;
    let rows = meta["rows"].as_u64().unwrap() as usize;
    let k: usize = flags.parse_or("k", 4096);
    let num_sub_vectors: usize = flags.parse_or("m", dim / 8);
    let sample_rate: usize = flags.parse_or("sample-rate", 256);
    let variant = flags.required("variant");
    let label = flags.get("label").unwrap_or(variant).to_string();

    let mut ivf = IvfBuildParams::new(k);
    ivf.sample_rate = sample_rate;
    // kmeans_algorithm: begin
    ivf.kmeans_algorithm = match variant {
        "lloyd" => Some(HierarchicalNodeAlgorithm::Lloyd),
        // The default: e-means for L2.
        "emeans" => None,
        other => panic!("--variant {other}: this build runs lloyd or emeans"),
    };
    // kmeans_algorithm: end
    let pq = PQBuildParams {
        num_sub_vectors,
        num_bits: 8,
        ..Default::default()
    };
    let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf, pq);

    let uri = dir.join("data.lance");
    let mut dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();
    let clock = Arc::new(StageClock::default());
    BUILD_LOG.lines.lock().unwrap().clear();
    let (cpu_start, _) = process_usage();
    let start = Instant::now();
    println!("{label}: build starts at monotonic {:.6}", monotonic_secs());
    dataset
        .create_index_builder(&["vector"], IndexType::Vector, &params)
        .name(INDEX_NAME.to_string())
        .replace(true)
        .progress(clock.clone())
        .await
        .unwrap();
    let build_secs = start.elapsed().as_secs_f64();
    let (cpu_end, peak_rss) = process_usage();
    println!("{label}: built IVF{k},PQ{num_sub_vectors} on {rows} x {dim} in {build_secs:.1}s");

    let since_start = |at: Instant| at.saturating_duration_since(start).as_secs_f64();
    let stages: Vec<serde_json::Value> = clock
        .events
        .lock()
        .unwrap()
        .iter()
        .map(|(stage, event, at)| json!([stage, event, since_start(*at)]))
        .collect();
    let log: Vec<serde_json::Value> = BUILD_LOG
        .lines
        .lock()
        .unwrap()
        .iter()
        .map(|(at, target, message)| json!([since_start(*at), target, message]))
        .collect();

    let queries = read_f32s(&dir.join("queries.f32"));
    let neighbours: Vec<u32> = std::fs::read(dir.join("neighbours.u32"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let mut recalls = Vec::new();
    for nprobes in flags.list::<usize>("nprobes", "16,64") {
        for refine in flags.list::<u32>("refine", "0") {
            let start = Instant::now();
            let recall = recall(&dataset, &queries, &neighbours, dim, nprobes, refine).await;
            println!(
                "{label}: recall@{RECALL_K} {recall:.4} at nprobes {nprobes}, refine {refine}"
            );
            recalls.push(json!({
                "nprobes": nprobes,
                "refine": refine,
                "recall": recall,
                "secs": start.elapsed().as_secs_f64(),
            }));
        }
    }

    dataset.drop_index(INDEX_NAME).await.unwrap();
    dataset
        .cleanup_old_versions(chrono::Duration::zero(), Some(true), None)
        .await
        .unwrap();

    let result = json!({
        "label": label,
        "variant": variant,
        "rows": rows,
        "dim": dim,
        "k": k,
        "num_sub_vectors": num_sub_vectors,
        "sample_rate": sample_rate,
        "threads": std::thread::available_parallelism().map_or(1, |n| n.get()),
        "build_secs": build_secs,
        "build_cpu_secs": cpu_end - cpu_start,
        "peak_rss_bytes": peak_rss,
        "stages": stages,
        "log": log,
        "recall": recalls,
    });
    let line = serde_json::to_string(&result).unwrap();
    if let Some(out) = flags.get("out") {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .unwrap();
        writeln!(file, "{line}").unwrap();
    }
}

/// Times one read path on its own: a scan of the vector column, or the
/// training sample IVF training draws.
async fn probe(flags: &Flags) {
    let dir = PathBuf::from(flags.required("dir"));
    let dataset = Dataset::open(dir.join("data.lance").to_str().unwrap())
        .await
        .unwrap();
    let what = flags.required("what");
    let start = Instant::now();
    let (rows, bytes) = match what {
        "scan" => {
            let mut scan = dataset.scan();
            scan.project(&["vector"]).unwrap();
            if let Some(readahead) = flags.get("batch-readahead") {
                scan.batch_readahead(readahead.parse().unwrap());
            }
            if flags.parse_or("in-order", false) {
                scan.scan_in_order(true);
            }
            if flags.parse_or("row-id", false) {
                scan.with_row_id();
            }
            let mut stream = scan.try_into_stream().await.unwrap();
            let (mut rows, mut bytes) = (0usize, 0usize);
            while let Some(batch) = stream.next().await {
                let batch = batch.unwrap();
                rows += batch.num_rows();
                bytes += batch.get_array_memory_size();
            }
            (rows, bytes)
        }
        "sample" => {
            let rows: usize = flags.parse_or("rows", 1 << 20);
            let sample = lance::index::vector::utils::maybe_sample_training_data(
                &dataset, "vector", rows, None,
            )
            .await
            .unwrap();
            (sample.len(), sample.get_array_memory_size())
        }
        other => panic!("--what {other}: expected scan or sample"),
    };
    let secs = start.elapsed().as_secs_f64();
    println!(
        "probe {what}: {rows} rows, {:.1} GB in {secs:.2}s ({:.2} GB/s)",
        bytes as f64 / 1e9,
        bytes as f64 / 1e9 / secs
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = argv.split_first() else {
        panic!("usage: ivf_build_bench write|build --flag value ...");
    };
    log::set_logger(&BUILD_LOG).unwrap();
    log::set_max_level(log::LevelFilter::Info);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    match command.as_str() {
        "write" => runtime.block_on(write(&Flags::parse(
            rest,
            &[
                "base",
                "queries",
                "rows",
                "num-queries",
                "dir",
                "pca",
                "dim",
            ],
        ))),
        "build" => runtime.block_on(build(&Flags::parse(
            rest,
            &[
                "dir",
                "k",
                "m",
                "sample-rate",
                "variant",
                "label",
                "nprobes",
                "refine",
                "out",
            ],
        ))),
        "probe" => runtime.block_on(probe(&Flags::parse(
            rest,
            &[
                "dir",
                "what",
                "rows",
                "batch-readahead",
                "in-order",
                "row-id",
            ],
        ))),
        other => panic!("unknown command {other}; expected write, build or probe"),
    }
}
