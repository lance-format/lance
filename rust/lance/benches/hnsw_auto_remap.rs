// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end cost of serving an IVF_HNSW index through deferred compaction.
//!
//! For each deletion rate the benchmark restores one shared base dataset,
//! deletes rows, and measures the index in four states:
//!
//! - `pre`: deletions applied, not compacted (deleted rows are prefiltered).
//! - `auto`: after a deferred-remap compaction; partitions are realigned with
//!   the fragment-reuse index as they load (the auto-remap path).
//! - `phys`: after the physical remap, which repairs the graph once on disk.
//! - `rebuild`: after rebuilding the index from scratch.
//!
//! Each state reports the cold prewarm time of every partition and the cold
//! first-query time (each with a fresh session), warm query latency, and
//! recall@k against exact search. Pass `--uri s3://...` to run against an
//! object store; the default is a local temporary directory.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::types::Int64Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator, UInt32Array,
    cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use lance::dataset::optimize::remapping::remap_column_index;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::session::Session;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::DistanceType;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const INDEX_NAME: &str = "vec_idx";
/// Deletions select rows whose `del` value, uniform in `[0, DEL_BUCKETS)`,
/// falls below the deletion rate in basis points.
const DEL_BUCKETS: u32 = 10_000;

struct Config {
    uri: String,
    rows: usize,
    dim: usize,
    clusters: usize,
    partitions: usize,
    fragment_rows: usize,
    delete_basis_points: Vec<u32>,
    queries: usize,
    k: usize,
    nprobes: usize,
    ef: usize,
    cold_repeats: usize,
    skip_rebuild: bool,
}

fn parse_config() -> Config {
    let mut config = Config {
        uri: std::env::temp_dir()
            .join("lance-hnsw-auto-remap-bench")
            .to_string_lossy()
            .into_owned(),
        rows: 1_000_000,
        dim: 128,
        clusters: 1_000,
        partitions: 256,
        fragment_rows: 10_000,
        delete_basis_points: vec![0, 100, 1_000, 3_000],
        queries: 200,
        k: 10,
        nprobes: 16,
        ef: 100,
        cold_repeats: 3,
        skip_rebuild: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{arg} requires a value"))
        };
        match arg.as_str() {
            // `cargo bench` passes this marker even to a custom harness.
            "--bench" => {}
            "--uri" => config.uri = value(),
            "--rows" => config.rows = value().parse().unwrap(),
            "--dim" => config.dim = value().parse().unwrap(),
            "--clusters" => config.clusters = value().parse().unwrap(),
            "--partitions" => config.partitions = value().parse().unwrap(),
            "--fragment-rows" => config.fragment_rows = value().parse().unwrap(),
            "--delete-bps" => {
                config.delete_basis_points =
                    value().split(',').map(|v| v.parse().unwrap()).collect()
            }
            "--queries" => config.queries = value().parse().unwrap(),
            "--k" => config.k = value().parse().unwrap(),
            "--nprobes" => config.nprobes = value().parse().unwrap(),
            "--ef" => config.ef = value().parse().unwrap(),
            "--cold-repeats" => config.cold_repeats = value().parse().unwrap(),
            "--skip-rebuild" => config.skip_rebuild = true,
            other => panic!("unknown argument: {other}"),
        }
    }
    config
}

/// Gaussian clusters, so the HNSW graph has the neighborhood structure of
/// real embeddings instead of uniform noise.
struct VectorGen {
    centers: Vec<f32>,
    dim: usize,
}

/// Standard deviation of each cluster around its center.
const CLUSTER_STDDEV: f32 = 0.1;

/// A normal sample by Box-Muller, to avoid a distribution dependency.
fn gaussian(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.random_range(f32::EPSILON..1.0);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

impl VectorGen {
    fn new(dim: usize, clusters: usize, rng: &mut SmallRng) -> Self {
        let centers = (0..dim * clusters)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        Self { centers, dim }
    }

    fn fill(&self, rows: usize, rng: &mut SmallRng) -> Vec<f32> {
        let clusters = self.centers.len() / self.dim;
        let mut values = Vec::with_capacity(rows * self.dim);
        for _ in 0..rows {
            let center = rng.random_range(0..clusters) * self.dim;
            values.extend(
                self.centers[center..center + self.dim]
                    .iter()
                    .map(|c| c + CLUSTER_STDDEV * gaussian(rng)),
            );
        }
        values
    }
}

fn index_params(config: &Config) -> VectorIndexParams {
    VectorIndexParams::with_ivf_hnsw_sq_params(
        DistanceType::L2,
        IvfBuildParams {
            num_partitions: Some(config.partitions),
            ..Default::default()
        },
        HnswBuildParams::default(),
        SQBuildParams::default(),
    )
}

async fn open_cold(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap()
}

/// Write the base dataset and its index, or reuse them from an earlier run.
/// Returns the base version every scenario restores.
async fn prepare_base(config: &Config, uri: &str) -> u64 {
    if let Ok(dataset) = Dataset::open(uri).await
        && dataset.count_rows(None).await.unwrap() == config.rows
        && dataset
            .load_index_by_name(INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    {
        let version = dataset.version().version;
        println!("reusing base dataset {uri} at version {version}");
        // The first version with the index is the base; later versions are
        // scenario leftovers.
        for v in dataset.versions().await.unwrap() {
            let at = dataset.checkout_version(v.version).await.unwrap();
            if at.load_index_by_name(INDEX_NAME).await.unwrap().is_some() {
                return v.version;
            }
        }
        unreachable!("an indexed version exists");
    }

    let mut rng = SmallRng::seed_from_u64(7);
    let generator = VectorGen::new(config.dim, config.clusters, &mut rng);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("del", DataType::UInt32, false),
        Field::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                config.dim as i32,
            ),
            false,
        ),
    ]));
    let batch_rows = config.fragment_rows.min(10_000);
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < config.rows {
        let rows = batch_rows.min(config.rows - start);
        let ids = Int64Array::from_iter_values(start as i64..(start + rows) as i64);
        let del =
            UInt32Array::from_iter_values((0..rows).map(|_| rng.random_range(0..DEL_BUCKETS)));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(generator.fill(rows, &mut rng)),
            config.dim as i32,
        )
        .unwrap();
        batches.push(Ok(RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(del), Arc::new(vectors)],
        )
        .unwrap()));
        start += rows;
    }
    let started = Instant::now();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: config.fragment_rows,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    println!(
        "wrote {} rows in {} fragments in {:.1}s",
        config.rows,
        dataset.get_fragments().len(),
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some(INDEX_NAME.into()),
            &index_params(config),
            false,
        )
        .await
        .unwrap();
    println!(
        "built IVF_HNSW_SQ ({} partitions) in {:.1}s",
        config.partitions,
        started.elapsed().as_secs_f64()
    );
    dataset.version().version
}

fn query_vectors(config: &Config) -> Vec<Float32Array> {
    // Same cluster centers as the data (same seed), different noise draws.
    let mut rng = SmallRng::seed_from_u64(7);
    let generator = VectorGen::new(config.dim, config.clusters, &mut rng);
    let mut rng = SmallRng::seed_from_u64(11);
    generator
        .fill(config.queries, &mut rng)
        .chunks(config.dim)
        .map(|v| Float32Array::from(v.to_vec()))
        .collect()
}

async fn knn(
    dataset: &Dataset,
    query: &Float32Array,
    config: &Config,
    use_index: bool,
) -> Vec<i64> {
    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", query, config.k)
        .unwrap()
        .nprobes(config.nprobes)
        .ef(config.ef)
        .use_index(use_index);
    scanner.project(&["id"]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch["id"].as_primitive::<Int64Type>().values().to_vec()
}

struct StateResult {
    state: &'static str,
    transition: Option<Duration>,
    cold_prewarm: Duration,
    cold_first_query: Duration,
    warm_p50: Duration,
    warm_p99: Duration,
    recall: f64,
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values[values.len() / 2]
}

async fn measure_state(
    state: &'static str,
    transition: Option<Duration>,
    uri: &str,
    config: &Config,
    queries: &[Float32Array],
    truth: &[Vec<i64>],
) -> StateResult {
    let mut prewarms = Vec::with_capacity(config.cold_repeats);
    let mut first_queries = Vec::with_capacity(config.cold_repeats);
    for _ in 0..config.cold_repeats {
        let dataset = open_cold(uri).await;
        let started = Instant::now();
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        prewarms.push(started.elapsed());

        let dataset = open_cold(uri).await;
        // Load the index metadata first so the timing covers partition loads.
        dataset.load_indices().await.unwrap();
        let started = Instant::now();
        knn(&dataset, &queries[0], config, true).await;
        first_queries.push(started.elapsed());
    }

    let dataset = open_cold(uri).await;
    dataset.prewarm_index(INDEX_NAME).await.unwrap();
    let mut latencies = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    for (query, expected) in queries.iter().zip(truth) {
        let started = Instant::now();
        let ids = knn(&dataset, query, config, true).await;
        latencies.push(started.elapsed());
        hits += ids.iter().filter(|id| expected.contains(id)).count();
    }
    latencies.sort();
    StateResult {
        state,
        transition,
        cold_prewarm: median(prewarms),
        cold_first_query: median(first_queries),
        warm_p50: latencies[latencies.len() / 2],
        warm_p99: latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)],
        recall: hits as f64 / (truth.len() * config.k) as f64,
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

async fn run_scenario(
    config: &Config,
    uri: &str,
    base_version: u64,
    delete_bps: u32,
    queries: &[Float32Array],
) -> Vec<StateResult> {
    let mut dataset = Dataset::open(uri)
        .await
        .unwrap()
        .checkout_version(base_version)
        .await
        .unwrap();
    dataset.restore().await.unwrap();
    if delete_bps > 0 {
        dataset
            .delete(&format!("del < {delete_bps}"))
            .await
            .unwrap();
    }
    let survivors = dataset.count_rows(None).await.unwrap();
    println!(
        "\n== delete {:.2}% ({} survivors) ==",
        delete_bps as f64 / 100.0,
        survivors
    );

    let started = Instant::now();
    let mut truth = Vec::with_capacity(queries.len());
    for query in queries {
        truth.push(knn(&dataset, query, config, false).await);
    }
    println!(
        "exact ground truth in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    let mut results = Vec::new();
    results.push(measure_state("pre", None, uri, config, queries, &truth).await);

    let started = Instant::now();
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 250_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let compaction = started.elapsed();
    assert!(metrics.fragments_removed > 0, "compaction removed nothing");
    results.push(measure_state("auto", Some(compaction), uri, config, queries, &truth).await);

    let started = Instant::now();
    remap_column_index(&mut dataset, &["vec"], Some(INDEX_NAME.into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let remap = started.elapsed();
    results.push(measure_state("phys", Some(remap), uri, config, queries, &truth).await);

    if !config.skip_rebuild {
        let started = Instant::now();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some(INDEX_NAME.into()),
                &index_params(config),
                true,
            )
            .await
            .unwrap();
        let rebuild = started.elapsed();
        results.push(measure_state("rebuild", Some(rebuild), uri, config, queries, &truth).await);
    }

    println!(
        "{:<8} {:>12} {:>13} {:>14} {:>10} {:>10} {:>8}",
        "state", "transition_s", "cold_prewarm", "cold_1st_query", "warm_p50", "warm_p99", "recall"
    );
    for r in &results {
        println!(
            "{:<8} {:>12} {:>11.1}ms {:>12.1}ms {:>8.2}ms {:>8.2}ms {:>8.4}",
            r.state,
            r.transition
                .map_or("-".to_string(), |d| format!("{:.2}", d.as_secs_f64())),
            ms(r.cold_prewarm),
            ms(r.cold_first_query),
            ms(r.warm_p50),
            ms(r.warm_p99),
            r.recall
        );
        println!(
            "JSON {{\"delete_bps\":{delete_bps},\"state\":\"{}\",\"transition_s\":{},\"cold_prewarm_ms\":{:.2},\"cold_first_query_ms\":{:.2},\"warm_p50_ms\":{:.3},\"warm_p99_ms\":{:.3},\"recall\":{:.4}}}",
            r.state,
            r.transition
                .map_or("null".to_string(), |d| format!("{:.3}", d.as_secs_f64())),
            ms(r.cold_prewarm),
            ms(r.cold_first_query),
            ms(r.warm_p50),
            ms(r.warm_p99),
            r.recall
        );
    }
    results
}

#[tokio::main]
async fn main() {
    // `RUST_LOG=warn` surfaces graph-rebuild fallbacks during the remap.
    env_logger::init();
    let config = parse_config();
    let uri = format!("{}/base", config.uri.trim_end_matches('/'));
    println!(
        "rows={} dim={} partitions={} fragment_rows={} queries={} k={} nprobes={} ef={} uri={uri}",
        config.rows,
        config.dim,
        config.partitions,
        config.fragment_rows,
        config.queries,
        config.k,
        config.nprobes,
        config.ef
    );
    let base_version = prepare_base(&config, &uri).await;
    let queries = query_vectors(&config);
    for &delete_bps in &config.delete_basis_points {
        run_scenario(&config, &uri, base_version, delete_bps, &queries).await;
    }
}
