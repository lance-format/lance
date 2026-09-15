// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SPBENCH read benchmark: vector-search QPS/latency against one dataset
//! state produced by the `sp_bench` fixture driver (tests/sp_bench.rs).
//!
//! Public-API only: `Dataset::open` / `checkout_version` / `Scanner`, plus
//! the io-stats facility on `ObjectStore` (`Dataset::object_store(None)` +
//! `io_stats_incremental`) for the cold-query `_fri` IO report.
//!
//! Environment:
//!   SPBENCH_URI            dataset path (required)
//!   SPBENCH_VERSION        dataset version to check out (required; S1/S2/S3/S4)
//!   SPBENCH_CONCURRENCY    comma list, default "1,16,64,256"
//!   SPBENCH_DURATION_SECS  measured seconds per level, default 30
//!   SPBENCH_WARMUP_SECS    warmup seconds per level, default 5
//!   SPBENCH_TOPK           default 10
//!   SPBENCH_NPROBES        default 32
//!   SPBENCH_QUERIES        distinct query vectors, default 1024 (same seeded
//!                          scheme as the fixture's ingest/query generator)
//!   SPBENCH_INDEX_CACHE_MB session index-cache capacity in MiB; 0/unset keeps
//!                          the 6 GiB default (STEP 0 diagnostic lever)
//!
//! Run:
//!   SPBENCH_URI=/nvme/spbench.lance SPBENCH_VERSION=<S3> \
//!   cargo run --release -p lance --example sp_read_bench

// The machine-readable `SPBENCH_READ <key>=<value>` stdout lines are the
// benchmark's output contract; downstream tooling parses them.
#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arrow_schema::DataType;
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Must match tests/sp_bench.rs so queries are in-distribution.
const QUERY_SEED: u64 = 0x5BEC_0002;

/// Open the fixture at `version`, honoring the `SPBENCH_INDEX_CACHE_MB` knob.
///
/// The stable-partition mapping and its row-map live in the session index
/// cache, so shrinking that cache is the lever the STEP 0 diagnostic uses to
/// tell block-level row-map re-reads (constant-high `_fri` regardless of cache
/// size) apart from whole-mapping eviction (`_fri` low with a big cache, high
/// with a tiny one). Unset / `0` keeps the 6 GiB default.
async fn open_fixture(uri: &str, version: u64) -> lance::Result<Dataset> {
    let mut builder = DatasetBuilder::from_uri(uri).with_version(version);
    if let Some(mb) = std::env::var("SPBENCH_INDEX_CACHE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|mb| *mb > 0)
    {
        builder = builder.with_index_cache_size_bytes(mb * 1024 * 1024);
    }
    builder.load().await
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer, got {v:?}"))
        })
        .unwrap_or(default)
}

fn fill_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let mut norm = 0f32;
    for x in v.iter_mut() {
        *x = rng.random_range(-1f32..1f32);
        norm += *x * *x;
    }
    let norm = norm.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

fn query_pool(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    (0..n).map(|_| fill_unit_vector(&mut rng, dim)).collect()
}

fn rss_report() -> String {
    #[cfg(target_os = "linux")]
    {
        let mut rss_kb = 0u64;
        let mut hwm_kb = 0u64;
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    rss_kb = rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                    hwm_kb = rest
                        .trim()
                        .trim_end_matches(" kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                }
            }
        }
        format!("rss_kb={rss_kb} peak_rss_kb={hwm_kb}")
    }
    #[cfg(not(target_os = "linux"))]
    {
        // macOS: only the peak is cheaply available (ru_maxrss, bytes).
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc == 0 {
            format!("peak_rss_kb={}", usage.ru_maxrss / 1024)
        } else {
            "peak_rss_kb=unavailable".to_string()
        }
    }
}

struct QueryCfg {
    topk: usize,
    nprobes: usize,
    empty_projection: bool,
}

async fn run_query(dataset: &Dataset, query: &[f32], cfg: &QueryCfg) -> lance::Result<usize> {
    let q = arrow_array::Float32Array::from(query.to_vec());
    let mut scan = dataset.scan();
    scan.nearest("vec", &q, cfg.topk)?;
    scan.nprobes(cfg.nprobes);
    scan.with_row_id();
    if cfg.empty_projection {
        scan.project::<&str>(&[])?;
    } else {
        scan.project(&["id"])?;
    }
    let batch = scan.try_into_batch().await?;
    Ok(batch.num_rows())
}

#[allow(clippy::too_many_arguments)]
async fn run_level(
    dataset: Dataset,
    queries: Arc<Vec<Vec<f32>>>,
    cfg: Arc<QueryCfg>,
    concurrency: usize,
    warmup: Duration,
    duration: Duration,
) -> (f64, u64, u64, u64, usize) {
    let counter = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let measure_start = start + warmup;
    let end = measure_start + duration;
    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let dataset = dataset.clone();
        let queries = queries.clone();
        let counter = counter.clone();
        let cfg = cfg.clone();
        handles.push(tokio::spawn(async move {
            let mut latencies_us: Vec<u64> = Vec::new();
            loop {
                let issued_at = Instant::now();
                if issued_at >= end {
                    break;
                }
                let qi = counter.fetch_add(1, Ordering::Relaxed) % queries.len();
                let rows = run_query(&dataset, &queries[qi], &cfg)
                    .await
                    .expect("query failed");
                assert!(rows > 0, "topk query returned no rows");
                if issued_at >= measure_start {
                    latencies_us.push(issued_at.elapsed().as_micros() as u64);
                }
            }
            latencies_us
        }));
    }
    let mut latencies: Vec<u64> = Vec::new();
    for handle in handles {
        latencies.extend(handle.await.expect("worker panicked"));
    }
    latencies.sort_unstable();
    let n = latencies.len();
    let pct = |p: f64| -> u64 {
        if n == 0 {
            return 0;
        }
        latencies[(((n as f64) * p) as usize).min(n - 1)]
    };
    let qps = n as f64 / duration.as_secs_f64();
    (qps, pct(0.50), pct(0.95), pct(0.99), n)
}

#[tokio::main]
async fn main() {
    let uri = std::env::var("SPBENCH_URI").expect("SPBENCH_URI is required");
    let version: u64 = std::env::var("SPBENCH_VERSION")
        .expect("SPBENCH_VERSION is required (one of the fixture's S versions)")
        .parse()
        .expect("SPBENCH_VERSION must be an integer");
    let concurrency: Vec<usize> = std::env::var("SPBENCH_CONCURRENCY")
        .unwrap_or_else(|_| "1,16,64,256".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("bad SPBENCH_CONCURRENCY entry"))
        .collect();
    let duration = Duration::from_secs(env_usize("SPBENCH_DURATION_SECS", 30) as u64);
    let warmup = Duration::from_secs(env_usize("SPBENCH_WARMUP_SECS", 5) as u64);
    let topk = env_usize("SPBENCH_TOPK", 10);
    let nprobes = env_usize("SPBENCH_NPROBES", 32);
    let num_queries = env_usize("SPBENCH_QUERIES", 1024);
    let index_cache_mb =
        std::env::var("SPBENCH_INDEX_CACHE_MB").unwrap_or_else(|_| "0".to_string());
    println!("SPBENCH_READ config index_cache_mb={index_cache_mb}");

    let dataset = open_fixture(&uri, version)
        .await
        .expect("failed to open dataset at SPBENCH_VERSION");

    // Discover the vector dimension from the schema.
    let field = dataset
        .schema()
        .field("vec")
        .expect("dataset has no `vec` column");
    let dim = match &field.data_type() {
        DataType::FixedSizeList(_, dim) => *dim as usize,
        other => panic!("`vec` must be a FixedSizeList, got {other:?}"),
    };
    let queries = Arc::new(query_pool(dim, num_queries));

    // The goal is to keep data-page IO out of the measurement: row id only,
    // no take of data columns. Verify the scanner accepts an empty physical
    // projection alongside `with_row_id`; fall back to the `id` column.
    let mut cfg = QueryCfg {
        topk,
        nprobes,
        empty_projection: true,
    };
    if run_query(&dataset, &queries[0], &cfg).await.is_err() {
        cfg.empty_projection = false;
        run_query(&dataset, &queries[0], &cfg)
            .await
            .expect("fallback projection [id] failed too");
    }
    println!(
        "SPBENCH_READ version={version} projection={}",
        if cfg.empty_projection {
            "rowid_only"
        } else {
            "id_fallback"
        }
    );
    let cfg = Arc::new(cfg);

    // COLD measurement: fresh open (fresh session caches), single query.
    {
        let cold = open_fixture(&uri, version).await.expect("cold open failed");
        let store = cold
            .object_store(None)
            .await
            .expect("object store unavailable");
        store.io_stats_incremental(); // reset
        let t = Instant::now();
        run_query(&cold, &queries[0], &cfg)
            .await
            .expect("cold query failed");
        let cold_us = t.elapsed().as_micros();
        let stats = store.io_stats_incremental();
        let fri: Vec<_> = stats
            .requests
            .iter()
            .filter(|request| request.path.as_ref().contains("_fri"))
            .collect();
        let fri_bytes: u64 = fri
            .iter()
            .map(|request| {
                request
                    .range
                    .as_ref()
                    .map(|range| range.end - range.start)
                    .unwrap_or(0)
            })
            .sum();
        println!(
            "SPBENCH_READ cold version={version} first_query_us={cold_us} \
             read_iops={} read_bytes={} fri_reads={} fri_read_bytes={fri_bytes}",
            stats.read_iops,
            stats.read_bytes,
            fri.len(),
        );
    }

    for conc in concurrency {
        let (qps, p50, p95, p99, n) = run_level(
            dataset.clone(),
            queries.clone(),
            cfg.clone(),
            conc,
            warmup,
            duration,
        )
        .await;
        println!(
            "SPBENCH_READ version={version} conc={conc} qps={qps:.1} \
             p50_us={p50} p95_us={p95} p99_us={p99} queries={n} {}",
            rss_report()
        );
    }
}
