// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remapping one IVF_HNSW_SQ partition after compaction: carrying the graph
//! over ([`remap_graph_batch`]) versus rebuilding it ([`HNSW::remap`], the
//! previous behavior).
//!
//! Production remap copies the graph when no row is deleted. When rows are
//! deleted, `repair` keeps the surviving edges and reconnects each node that
//! lost a neighbor with a construction-ef beam search and the builder's
//! neighbor heuristic, instead of rebuilding. `reuse` is the unrepaired edge
//! drop, kept so its recall cost stays visible.
//!
//! For each partition size and deleted fraction this reports wall time, CPU
//! time and peak heap growth of both paths, recall@K of both graphs against
//! exact search on the surviving vectors, the level-0 degree left after
//! dropping edges, and whether search on an unchanged graph still returns the
//! same neighbors under the remapped row ids. The quantized storage `remap` is
//! shared by both paths and reported separately.
//!
//! It is not a criterion bench: a rebuild of a large partition takes minutes,
//! and the recall and memory figures are the point. Configure with
//! `HNSW_REMAP_SIZES` (comma separated, default `10000,100000,500000`),
//! `HNSW_REMAP_DIM` (default 128), `HNSW_REMAP_DELETED` (percent, default
//! `0,1,10,25,50`) and `HNSW_REMAP_REPEATS` (default 3).
//!
//! Vectors are uniform random by default, which has the full intrinsic
//! dimension and is much harder for HNSW than real embeddings. Setting
//! `HNSW_REMAP_LATENT_DIM` instead draws them from a random linear map of a
//! uniform latent space of that dimension, plus small noise.

// The results are printed for the human running the bench.
#![allow(clippy::print_stdout)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use arrow::array::AsArray;
use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, UInt64Array, types::Float32Type, types::UInt32Type,
};
use arrow_schema::{Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID_FIELD;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::create_writer;
use lance_file::writer::FileWriterOptions;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::prefilter::NoFilter;
use lance_index::vector::hnsw::builder::{HNSW_METADATA_KEY, HnswBuildParams, HnswQueryParams};
use lance_index::vector::hnsw::remap::{remap_graph_batch, remap_graph_repair};
use lance_index::vector::hnsw::{HNSW, HnswMetadata};
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::sq::{
    ScalarQuantizer, builder::SQBuildParams, storage::ScalarQuantizationStorage,
};
use lance_index::vector::storage::{StorageBuilder, VectorStore};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;
use object_store::path::Path;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const K: usize = 10;
const NUM_QUERIES: usize = 200;
/// `k + k / 2` is the ef a query gets when it sets none; 64 and 256 are tuned queries.
const EFS: [usize; 3] = [K + K / 2, 64, 256];
/// Remapped rows move to a new fragment, as they do in compaction.
const NEW_FRAGMENT: u64 = 1 << 32;

struct PeakAlloc;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

fn cpu_time() -> Duration {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage fills the struct it is given.
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

struct Measured<T> {
    value: T,
    wall: Duration,
    cpu: Duration,
    peak_bytes: usize,
}

fn measure<T>(f: impl FnOnce() -> T) -> Measured<T> {
    let base = ALLOCATED.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let cpu = cpu_time();
    let start = Instant::now();
    let value = f();
    let wall = start.elapsed();
    let cpu = cpu_time() - cpu;
    let peak_bytes = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    Measured {
        value,
        wall,
        cpu,
        peak_bytes,
    }
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

fn generate_vectors(
    num_vectors: usize,
    dim: usize,
    latent_dim: Option<usize>,
    seed: u8,
) -> FixedSizeListArray {
    let Some(latent_dim) = latent_dim else {
        let data = generate_random_array_with_seed::<Float32Type>(num_vectors * dim, [seed; 32]);
        return FixedSizeListArray::try_new_from_values(data, dim as i32).unwrap();
    };
    // The same map for data and queries, so both lie near one subspace.
    let mut map_rng = StdRng::seed_from_u64(0);
    let map = (0..latent_dim * dim)
        .map(|_| map_rng.random_range(-1.0f32..1.0))
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut data = Vec::with_capacity(num_vectors * dim);
    let mut latent = vec![0.0f32; latent_dim];
    for _ in 0..num_vectors {
        latent.iter_mut().for_each(|x| *x = rng.random());
        for d in 0..dim {
            let x: f32 = (0..latent_dim).map(|l| latent[l] * map[l * dim + d]).sum();
            data.push(x + rng.random_range(-0.01f32..0.01));
        }
    }
    FixedSizeListArray::try_new_from_values(arrow_array::Float32Array::from(data), dim as i32)
        .unwrap()
}

fn sq_storage(vectors: &FixedSizeListArray, row_ids: UInt64Array) -> ScalarQuantizationStorage {
    let quantizer = <ScalarQuantizer as Quantization>::build(
        vectors,
        DistanceType::L2,
        &SQBuildParams::default(),
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("vector", vectors.data_type().clone(), true),
        ROW_ID_FIELD.clone(),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(vectors.clone()), Arc::new(row_ids)]).unwrap();
    StorageBuilder::new("vector".to_owned(), DistanceType::L2, quantizer, None)
        .unwrap()
        .build(vec![batch])
        .unwrap()
}

/// Exact top-K row ids of each query among `vectors` (row id = position).
fn ground_truth(vectors: &FixedSizeListArray, queries: &FixedSizeListArray) -> Vec<HashSet<u64>> {
    let dim = vectors.value_length() as usize;
    let data = vectors.values().as_primitive::<Float32Type>().values();
    let queries_data = queries.values().as_primitive::<Float32Type>().values();
    (0..queries.len())
        .into_par_iter()
        .map(|q| {
            let query = &queries_data[q * dim..(q + 1) * dim];
            let mut dists = data
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i as u64)
                })
                .collect::<Vec<_>>();
            dists.select_nth_unstable_by(K, |a, b| a.0.total_cmp(&b.0));
            dists[..K].iter().map(|(_, i)| *i).collect()
        })
        .collect()
}

/// Search every query and return the row ids each one found.
fn search(
    hnsw: &HNSW,
    storage: &ScalarQuantizationStorage,
    queries: &FixedSizeListArray,
    ef: usize,
) -> Vec<Vec<u64>> {
    (0..queries.len())
        .map(|q| {
            let params = HnswQueryParams {
                ef,
                lower_bound: None,
                upper_bound: None,
                dist_q_c: 0.0,
                use_acorn: false,
            };
            let batch = hnsw
                .search(
                    queries.value(q),
                    K,
                    params,
                    storage,
                    Arc::new(NoFilter),
                    &NoOpMetricsCollector,
                )
                .unwrap();
            batch["_rowid"]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

/// Recall@K where ground truth is in position space and results are row ids.
fn recall(results: &[Vec<u64>], truth: &[HashSet<u64>], to_position: impl Fn(u64) -> u64) -> f64 {
    let hits: usize = results
        .iter()
        .zip(truth)
        .map(|(found, truth)| {
            found
                .iter()
                .filter(|row_id| truth.contains(&to_position(**row_id)))
                .count()
        })
        .sum();
    hits as f64 / (results.len() * K) as f64
}

struct GraphStats {
    rows: usize,
    level_members: usize,
    edges: usize,
    level0_mean_degree: f64,
    level0_isolated: usize,
    max_neighbor_id: Option<u32>,
}

fn graph_stats(batch: &RecordBatch) -> GraphStats {
    let metadata: HnswMetadata =
        serde_json::from_str(&batch.schema_ref().metadata()[HNSW_METADATA_KEY]).unwrap();
    let hnsw = HNSW::load(batch.clone()).unwrap();
    let neighbors = batch["__neighbors"].as_list::<i32>();
    let level0 = metadata.level_offsets[1];
    let degrees = (0..level0).map(|row| neighbors.value_length(row) as usize);
    let values = neighbors.values().as_primitive::<UInt32Type>();
    GraphStats {
        rows: batch.num_rows(),
        level_members: (0..hnsw.max_level() as usize)
            .map(|level| hnsw.num_nodes(level))
            .sum(),
        edges: values.len(),
        level0_mean_degree: degrees.clone().sum::<usize>() as f64 / level0.max(1) as f64,
        level0_isolated: degrees.filter(|d| *d == 0).count(),
        max_neighbor_id: values.values().iter().max().copied(),
    }
}

fn print_measured<T>(case: &str, path: &str, runs: &[Measured<T>]) {
    let mut walls = runs
        .iter()
        .map(|r| r.wall.as_secs_f64())
        .collect::<Vec<_>>();
    walls.sort_by(f64::total_cmp);
    let median = walls[walls.len() / 2];
    let raw = runs
        .iter()
        .map(|r| {
            format!(
                "{:.4}s/cpu {:.3}s/peak {:.1}MiB",
                r.wall.as_secs_f64(),
                r.cpu.as_secs_f64(),
                r.peak_bytes as f64 / (1 << 20) as f64
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    println!("time {case} path={path} median_wall_s={median:.4} runs=[{raw}]");
}

fn main() {
    let sizes = env_list("HNSW_REMAP_SIZES", "10000,100000,500000");
    let dim = env_list("HNSW_REMAP_DIM", "128")[0];
    let deleted_percents = env_list("HNSW_REMAP_DELETED", "0,1,10,25,50");
    let repeats = env_list("HNSW_REMAP_REPEATS", "3")[0];
    let latent_dim = std::env::var("HNSW_REMAP_LATENT_DIM")
        .ok()
        .map(|v| v.parse::<usize>().unwrap());
    let params = HnswBuildParams::default();
    println!(
        "config dim={dim} latent_dim={latent_dim:?} k={K} queries={NUM_QUERIES} efs={EFS:?} \
         repeats={repeats} rayon_threads={} m={} ef_construction={} max_level={}",
        rayon::current_num_threads(),
        params.m,
        params.ef_construction,
        params.max_level
    );
    let rt = tokio::runtime::Runtime::new().unwrap();

    for &size in &sizes {
        let vectors = generate_vectors(size, dim, latent_dim, 42);
        let queries = generate_vectors(NUM_QUERIES, dim, latent_dim, 7);
        let storage = sq_storage(&vectors, UInt64Array::from_iter_values(0..size as u64));

        let built = measure(|| HNSW::index_vectors(&storage, params.clone()).unwrap());
        println!(
            "build n={size} wall_s={:.3} cpu_s={:.3}",
            built.wall.as_secs_f64(),
            built.cpu.as_secs_f64()
        );
        let original = HNSW::load(built.value.to_batch().unwrap()).unwrap();
        // The batch the remap reads: every column, as written to the index file.
        let graph = original.to_batch().unwrap();
        let original_stats = graph_stats(&graph);
        println!(
            "graph n={size} rows={} level_members={} edges={} level0_mean_degree={:.2} \
             max_neighbor_id={:?}",
            original_stats.rows,
            original_stats.level_members,
            original_stats.edges,
            original_stats.level0_mean_degree,
            original_stats.max_neighbor_id
        );
        let truth_before = ground_truth(&vectors, &queries);
        let before = EFS
            .iter()
            .map(|&ef| search(&original, &storage, &queries, ef))
            .collect::<Vec<_>>();

        for &percent in &deleted_percents {
            let case = format!("n={size} deleted={percent}%");
            let mut rng = StdRng::seed_from_u64(percent as u64);
            let mut kept = Vec::with_capacity(size);
            let mapping = (0..size as u64)
                .map(|row_id| {
                    if rng.random_range(0..100) < percent {
                        (row_id, None)
                    } else {
                        kept.push(row_id as u32);
                        (row_id, Some(NEW_FRAGMENT + kept.len() as u64 - 1))
                    }
                })
                .collect::<HashMap<_, _>>();
            let mapping = RowAddrRemap::direct(mapping);

            let storage_remap = (0..repeats)
                .map(|_| measure(|| storage.remap(&mapping).unwrap()))
                .collect::<Vec<_>>();
            print_measured(&case, "storage_remap", &storage_remap);
            let new_storage = storage.remap(&mapping).unwrap();

            let reuse = (0..repeats)
                .map(|_| {
                    measure(|| {
                        let new_ids = storage
                            .row_ids()
                            .scan(0u32, |next, row_id| {
                                Some(match mapping.get(*row_id) {
                                    Some(None) => None,
                                    _ => {
                                        *next += 1;
                                        Some(*next - 1)
                                    }
                                })
                            })
                            .collect::<Vec<_>>();
                        HNSW::load(remap_graph_batch(&graph, &new_ids).unwrap()).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            print_measured(&case, "reuse", &reuse);
            let repair = (0..repeats)
                .map(|_| {
                    measure(|| {
                        let new_ids = storage
                            .row_ids()
                            .scan(0u32, |next, row_id| {
                                Some(match mapping.get(*row_id) {
                                    Some(None) => None,
                                    _ => {
                                        *next += 1;
                                        Some(*next - 1)
                                    }
                                })
                            })
                            .collect::<Vec<_>>();
                        HNSW::load(remap_graph_repair(&graph, &new_ids, &new_storage).unwrap())
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            print_measured(&case, "repair", &repair);
            let rebuild = (0..repeats)
                .map(|_| {
                    measure(|| {
                        HNSW::load(graph.clone())
                            .unwrap()
                            .remap(&mapping, &new_storage)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            print_measured(&case, "rebuild", &rebuild);

            let reused = &reuse[0].value;
            let repaired = &repair[0].value;
            let rebuilt = &rebuild[0].value;
            let reused_batch = reused.to_batch().unwrap();
            let repaired_batch = repaired.to_batch().unwrap();
            let stats = graph_stats(&reused_batch);
            let repair_stats = graph_stats(&repaired_batch);
            let rebuilt_stats = graph_stats(&rebuilt.to_batch().unwrap());
            assert!(
                stats
                    .max_neighbor_id
                    .is_none_or(|id| (id as usize) < kept.len()),
                "{case}: an edge points past the remapped graph"
            );
            assert_eq!(stats.rows, stats.level_members);
            println!(
                "graph {case} path=reuse rows={} level_members={} edges={} \
                 level0_mean_degree={:.2} level0_isolated={} max_neighbor_id={:?}",
                stats.rows,
                stats.level_members,
                stats.edges,
                stats.level0_mean_degree,
                stats.level0_isolated,
                stats.max_neighbor_id
            );
            println!(
                "graph {case} path=repair rows={} level_members={} edges={} \
                 level0_mean_degree={:.2} level0_isolated={} max_neighbor_id={:?}",
                repair_stats.rows,
                repair_stats.level_members,
                repair_stats.edges,
                repair_stats.level0_mean_degree,
                repair_stats.level0_isolated,
                repair_stats.max_neighbor_id
            );
            println!(
                "graph {case} path=rebuild rows={} level_members={} edges={} \
                 level0_mean_degree={:.2} level0_isolated={}",
                rebuilt_stats.rows,
                rebuilt_stats.level_members,
                rebuilt_stats.edges,
                rebuilt_stats.level0_mean_degree,
                rebuilt_stats.level0_isolated
            );

            let kept_vectors = arrow::compute::take(
                &vectors,
                &arrow_array::UInt32Array::from(kept.clone()),
                None,
            )
            .unwrap();
            let truth_after = ground_truth(kept_vectors.as_fixed_size_list(), &queries);
            let to_position = |row_id: u64| row_id - NEW_FRAGMENT;
            let new_row_ids = new_storage.row_ids().copied().collect::<HashSet<_>>();
            for (i, &ef) in EFS.iter().enumerate() {
                let after_reuse = search(reused, &new_storage, &queries, ef);
                let after_repair = search(repaired, &new_storage, &queries, ef);
                // Each rebuild is a different graph (insertion runs in parallel),
                // so their spread is the noise to judge the reuse delta against.
                let rebuild_recalls = rebuild
                    .iter()
                    .map(|run| {
                        let found = search(&run.value, &new_storage, &queries, ef);
                        recall(&found, &truth_after, to_position)
                    })
                    .collect::<Vec<_>>();
                let rebuild_median = {
                    let mut sorted = rebuild_recalls.clone();
                    sorted.sort_by(f64::total_cmp);
                    sorted[sorted.len() / 2]
                };
                assert!(
                    after_reuse
                        .iter()
                        .flatten()
                        .all(|row_id| new_row_ids.contains(row_id)),
                    "{case}: search returned a row id outside the remapped storage"
                );
                let same_as_before = if percent == 0 {
                    let remapped_before = before[i]
                        .iter()
                        .map(|ids| {
                            ids.iter()
                                .map(|id| mapping.get(*id).flatten().unwrap())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    format!(" identical_to_pre_remap={}", remapped_before == after_reuse)
                } else {
                    String::new()
                };
                println!(
                    "recall {case} ef={ef} pre_remap={:.4} reuse={:.4} repair={:.4} \
                     rebuild_median={:.4} rebuild_runs={:?}{same_as_before}",
                    recall(&before[i], &truth_before, |id| id),
                    recall(&after_reuse, &truth_after, to_position),
                    recall(&after_repair, &truth_after, to_position),
                    rebuild_median,
                    rebuild_recalls
                        .iter()
                        .map(|r| format!("{r:.4}"))
                        .collect::<Vec<_>>(),
                );
            }

            if percent == deleted_percents[deleted_percents.len() - 1] {
                // The remapped graph goes through the same writer as the index file.
                let written = rt.block_on(async {
                    let store = ObjectStore::memory();
                    let path = Path::from("hnsw_remap_graph.lance");
                    let schema =
                        lance_core::datatypes::Schema::try_from(HNSW::schema().as_ref()).unwrap();
                    let mut writer = create_writer(
                        ConcreteFileVersion::V2_1,
                        store.create(&path).await.unwrap(),
                        schema,
                        FileWriterOptions::default(),
                    )
                    .unwrap();
                    writer.write_batch(&reused_batch).await.unwrap();
                    writer.finish().await.unwrap();
                    store.size(&path).await.unwrap()
                });
                println!(
                    "write {case} format=2.1 rows={} bytes={written}",
                    reused_batch.num_rows()
                );
            }
        }
    }
}
