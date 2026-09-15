// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SPBENCH fixture driver: builds the dataset states for the stable-partition
//! remap read benchmark and times every maintenance stage.
//!
//! States produced (versions printed as `SPBENCH S1=.. S2=.. S3=..`):
//!   S1 = segmented IVF_RQ(1-bit) index over the source fragments
//!   S2 = after a synthetic stable partition ALL src -> dest fragments
//!   S3 = after the tagged remap job (free-case restamp for fully-covered
//!        segments; partially-covered segments are skipped)
//!   S4 = (optional, `SPBENCH_FREECASE=1`) restamp path on a second dataset
//!        with SEGMENTS=1, at `<uri>_freecase`
//!
//! Run locally (small defaults):
//!   cargo test -q -p lance --test sp_bench -- --ignored --nocapture
//!
//! Rig-scale example:
//!   SPBENCH_URI=/nvme/spbench.lance SPBENCH_ROWS=100000000 \
//!   SPBENCH_SRC_FRAGS=100 SPBENCH_DEST_FRAGS=100 SPBENCH_DIM=256 \
//!   SPBENCH_SEGMENTS=4 SPBENCH_IVF_PARTITIONS=4096 SPBENCH_FREECASE=1 \
//!   cargo test --release -p lance --test sp_bench -- --ignored --nocapture

// The machine-readable `SPBENCH <key>=<value>` stdout lines are the harness's
// output contract; downstream tooling parses them.
#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use lance::Dataset;
use lance::dataset::transaction::{
    FragReuseUpdate, FragmentReuseRewrite, Operation, RewriteGroup, Transaction,
};
use lance::dataset::write::CommitBuilder;
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_core::cache::LanceCache;
use lance_index::IndexType;
use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
use lance_index::frag_reuse::stable_partition::MAPPING_FILE;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::IndexStore;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_linalg::distance::DistanceType;
use lance_table::format::Fragment;
use lance_table::format::pb::fragment_reuse_index_details::{
    FragmentDigest, StablePartition, Transition, transition,
};
use object_store::path::Path as ObjPath;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

const INGEST_SEED: u64 = 0x5BEC_0001;
const QUERY_SEED: u64 = 0x5BEC_0002;
const BUCKET_SEED: u64 = 0x5BEC_0003;
const GEN_CHUNK_ROWS: usize = 65_536;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer, got {v:?}"))
        })
        .unwrap_or(default)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn bucket_of(id: u64, dest_frags: usize) -> usize {
    (splitmix64(id ^ BUCKET_SEED) % dest_frags as u64) as usize
}

fn table_schema(dim: usize) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
        Field::new("id", DataType::UInt64, false),
    ]))
}

/// Random unit vectors: uniform in [-1, 1) per component, then normalized.
/// The read bench (`examples/sp_read_bench.rs`) generates its query pool with
/// the SAME scheme and QUERY_SEED so queries are in-distribution.
fn fill_unit_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    let mut values = vec![0f32; n * dim];
    for row in 0..n {
        let slice = &mut values[row * dim..(row + 1) * dim];
        let mut norm = 0f32;
        for v in slice.iter_mut() {
            *v = rng.random_range(-1f32..1f32);
            norm += *v * *v;
        }
        let norm = norm.sqrt().max(1e-12);
        for v in slice.iter_mut() {
            *v /= norm;
        }
    }
    values
}

fn make_batch(
    schema: &Arc<ArrowSchema>,
    rng: &mut StdRng,
    start_id: u64,
    n: usize,
    dim: usize,
) -> RecordBatch {
    let values = fill_unit_vectors(rng, n, dim);
    let vec_arr = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(Float32Array::from(values)),
        None,
    )
    .unwrap();
    let ids = UInt64Array::from_iter_values(start_id..start_id + n as u64);
    RecordBatch::try_new(schema.clone(), vec![Arc::new(vec_arr), Arc::new(ids)]).unwrap()
}

fn query_pool(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(QUERY_SEED);
    (0..n)
        .map(|_| fill_unit_vectors(&mut rng, 1, dim))
        .collect()
}

fn vmhwm_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest.trim().trim_end_matches(" kB").trim().parse().ok();
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_bytes(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

#[derive(Clone)]
struct Config {
    rows: usize,
    src_frags: usize,
    dest_frags: usize,
    dim: usize,
    segments: usize,
    ivf_partitions: usize,
    topk: usize,
    nprobes: usize,
    /// SPBENCH_INDEX_TYPE: "ivf_rq_1bit" (default), "ivf_rq_8bit", "ivf_pq".
    /// Controls both the built index and the from-scratch rebuild baseline so
    /// remap-vs-rebuild is measured for the same index kind.
    index_type: String,
}

/// Vector index params for the configured index type. Both the segmented build
/// and the rebuild baseline call this so they stay identical.
fn index_params(cfg: &Config) -> VectorIndexParams {
    match cfg.index_type.as_str() {
        "ivf_rq_1bit" => VectorIndexParams::ivf_rq(cfg.ivf_partitions, 1, DistanceType::L2),
        "ivf_rq_8bit" => VectorIndexParams::ivf_rq(cfg.ivf_partitions, 8, DistanceType::L2),
        "ivf_pq" => VectorIndexParams::ivf_pq(
            cfg.ivf_partitions,
            8,
            (cfg.dim / 8).max(1),
            DistanceType::L2,
            50,
        ),
        // IVF_HNSW_SQ: graph-building index, the most expensive to rebuild —
        // the case where remap is most likely to beat rebuild.
        "ivf_hnsw_sq" => VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            lance_index::vector::ivf::IvfBuildParams::new(cfg.ivf_partitions),
            lance_index::vector::hnsw::builder::HnswBuildParams::default(),
            lance_index::vector::sq::builder::SQBuildParams::default(),
        ),
        other => panic!(
            "unknown SPBENCH_INDEX_TYPE {other:?}; use ivf_rq_1bit | ivf_rq_8bit | ivf_pq | ivf_hnsw_sq"
        ),
    }
}

/// INGEST + INDEX interleaved the natural OSS way: ingest one slice of whole
/// fragments, `create_index` on the first slice, then per following slice
/// `optimize_indices(OptimizeOptions::append())` (num_indices_to_merge = 0),
/// which appends a new delta segment covering exactly the new fragments.
async fn ingest_and_index(uri: &str, cfg: &Config, prefix: &str) -> (Dataset, u64) {
    let rows_per_frag = cfg.rows / cfg.src_frags;
    let frags_per_seg = cfg.src_frags / cfg.segments;
    let schema = table_schema(cfg.dim);
    let mut rng = StdRng::seed_from_u64(INGEST_SEED);
    let mut next_id: u64 = 0;
    let mut dataset: Option<Dataset> = None;
    let mut ingest_secs = 0f64;
    let mut index_secs = 0f64;

    for seg in 0..cfg.segments {
        let t = Instant::now();
        for frag_in_seg in 0..frags_per_seg {
            let mut batches = Vec::with_capacity(rows_per_frag.div_ceil(GEN_CHUNK_ROWS));
            let mut remaining = rows_per_frag;
            while remaining > 0 {
                let n = remaining.min(GEN_CHUNK_ROWS);
                batches.push(make_batch(&schema, &mut rng, next_id, n, cfg.dim));
                next_id += n as u64;
                remaining -= n;
            }
            let params = WriteParams {
                mode: if seg == 0 && frag_in_seg == 0 {
                    WriteMode::Overwrite
                } else {
                    WriteMode::Append
                },
                max_rows_per_file: rows_per_frag,
                ..Default::default()
            };
            dataset = Some(match dataset.take() {
                None => {
                    let reader =
                        RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
                    Dataset::write(reader, uri, Some(params)).await.unwrap()
                }
                Some(ds) => InsertBuilder::new(Arc::new(ds))
                    .with_params(&params)
                    .execute(batches)
                    .await
                    .unwrap(),
            });
        }
        ingest_secs += t.elapsed().as_secs_f64();

        let ds = dataset.as_mut().unwrap();
        let t = Instant::now();
        if seg == 0 {
            let params = index_params(cfg);
            ds.create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();
        } else {
            ds.optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
        }
        let seg_secs = t.elapsed().as_secs_f64();
        index_secs += seg_secs;
        println!("SPBENCH {prefix}index_seg{seg}_secs={seg_secs:.3}");
    }

    let dataset = dataset.unwrap();
    let seg_count = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == "vec_idx")
        .count();
    assert_eq!(
        seg_count, cfg.segments,
        "expected {} index segments, found {seg_count}",
        cfg.segments
    );
    println!("SPBENCH {prefix}ingest_secs={ingest_secs:.3}");
    println!("SPBENCH {prefix}index_secs={index_secs:.3}");
    println!("SPBENCH {prefix}index_variant=IVF_RQ_1bit");
    let s1 = dataset.version().version;
    (dataset, s1)
}

/// Synthetic stable partition of ALL source fragments into `dest_frags`
/// bucket-sorted destinations (seeded hash on `id`), committed through the
/// branch's SP commit path. This is the scaled analogue of the in-crate
/// `frag_reuse_reader::tests::prepare_partition` /
/// `commit_stable_partition` test helpers: destination data files really
/// contain the reordered rows, the row map is written under `_fri/<uuid>/`,
/// and the Rewrite commit appends the transition to the reuse ledger.
async fn stable_partition_all(dataset: Dataset, dataset_dir: &str, cfg: &Config) -> Dataset {
    let src_ids: Vec<u64> = {
        let mut ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
        ids.sort_unstable();
        ids
    };
    let source_fragments: Vec<Fragment> = src_ids
        .iter()
        .map(|id| {
            dataset
                .fragments()
                .iter()
                .find(|f| f.id == *id)
                .unwrap()
                .clone()
        })
        .collect();
    let max_src_id = *src_ids.last().unwrap();

    // Reserve enough fragment ids that the explicit destination ids
    // (max_src_id+1 ..) stay unique against any later append.
    let read_version = dataset.version().version;
    let dataset = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::ReserveFragments {
                num_fragments: (cfg.dest_frags * 2) as u32,
            },
            None,
        ))
        .await
        .unwrap();

    // One ordered pass over the sources: label every row with its destination
    // bucket and stage the bucket-sorted sub-batches.
    let mut labels: Vec<u16> = Vec::with_capacity(cfg.rows);
    let mut buckets: Vec<Vec<RecordBatch>> = vec![Vec::new(); cfg.dest_frags];
    {
        let mut scan = dataset.scan();
        scan.with_fragments(source_fragments.clone());
        scan.scan_in_order(true);
        scan.batch_size(16_384);
        let mut stream = scan.try_into_stream().await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let ids = batch["id"].as_primitive::<UInt64Type>();
            let mut positions: Vec<Vec<u32>> = vec![Vec::new(); cfg.dest_frags];
            for (i, id) in ids.values().iter().enumerate() {
                let b = bucket_of(*id, cfg.dest_frags);
                labels.push(b as u16);
                positions[b].push(i as u32);
            }
            for (b, pos) in positions.into_iter().enumerate() {
                if pos.is_empty() {
                    continue;
                }
                let idx = UInt32Array::from(pos);
                let cols = batch
                    .columns()
                    .iter()
                    .map(|c| arrow::compute::take(c, &idx, None).unwrap())
                    .collect();
                buckets[b].push(RecordBatch::try_new(batch.schema(), cols).unwrap());
            }
        }
    }
    assert_eq!(labels.len(), cfg.rows, "sources must scan every row");

    // Write each destination as ONE fragment (uncommitted appends), a few in
    // flight at a time; every destination receives rows from every source.
    let ds_arc = Arc::new(dataset.clone());
    let futs = buckets.into_iter().enumerate().map(|(b, batches)| {
        let ds = ds_arc.clone();
        async move {
            let rows: usize = batches.iter().map(|x| x.num_rows()).sum();
            assert!(
                rows > 0,
                "destination bucket {b} is empty; raise SPBENCH_ROWS"
            );
            let txn = InsertBuilder::new(ds)
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: rows,
                    ..Default::default()
                })
                .execute_uncommitted(batches)
                .await
                .unwrap();
            let Operation::Append { fragments } = txn.operation else {
                unreachable!("uncommitted append must yield an Append operation")
            };
            assert_eq!(fragments.len(), 1, "bucket {b} must land in one fragment");
            fragments.into_iter().next().unwrap()
        }
    });
    let mut destinations: Vec<Fragment> = futures::stream::iter(futs)
        .buffered(8)
        .collect::<Vec<_>>()
        .await;
    for (b, fragment) in destinations.iter_mut().enumerate() {
        fragment.id = max_src_id + 1 + b as u64;
    }

    // Row map for the transition, written under `_fri/<uuid>/` exactly like
    // the reader-test helper (labels in source scan order). For an object-store
    // URI (az://container/path) the object path is the URL path component only
    // (the store is already rooted at the container); for a local path use the
    // absolute path directly.
    let map_id = Uuid::new_v4();
    let base = if dataset_dir.contains("://") {
        let url = url::Url::parse(dataset_dir).expect("dataset_dir is a valid URI");
        ObjPath::from(url.path().trim_start_matches('/'))
    } else {
        ObjPath::from_absolute_path(dataset_dir).unwrap()
    };
    let store = LanceIndexStore::with_format_version(
        dataset.object_store(None).await.unwrap(),
        base.join("_fri").join(map_id.to_string()),
        Arc::new(LanceCache::with_capacity(1024 * 1024)),
        lance_file::version::ConcreteFileVersion::V2_1,
    );
    let writer = store
        .new_index_file(MAPPING_FILE, RowMapWriter::schema())
        .await
        .unwrap();
    let source_rows: Vec<SourceRows> = source_fragments
        .iter()
        .map(|f| SourceRows {
            physical_rows: f.physical_rows.unwrap() as u64,
            deleted: None,
        })
        .collect();
    let mut writer = RowMapWriter::try_new(writer, source_rows, cfg.dest_frags as u32).unwrap();
    writer.append_labels(&labels).await.unwrap();
    let (file, _) = writer.finish().await.unwrap();

    let transition = Transition {
        sources: source_fragments
            .iter()
            .map(|f| FragmentDigest {
                id: f.id,
                physical_rows: f.physical_rows.unwrap() as u64,
                num_deleted_rows: 0,
            })
            .collect(),
        destinations: destinations
            .iter()
            .map(|f| FragmentDigest {
                id: f.id,
                physical_rows: f.physical_rows.unwrap() as u64,
                num_deleted_rows: 0,
            })
            .collect(),
        mapping: Some(transition::Mapping::StablePartition(StablePartition {
            map_id: map_id.to_string(),
            map_size_bytes: file.size_bytes,
            base_id: None,
        })),
    };

    let read_version = dataset.version().version;
    CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: source_fragments,
                    new_fragments: destinations,
                }],
                rewritten_indices: vec![],
                frag_reuse: Some(FragReuseUpdate::AppendTransitions(
                    FragmentReuseRewrite::new(vec![transition]),
                )),
            },
            None,
        ))
        .await
        .unwrap()
}

/// Drive the tagged remap over every segment of `vec_idx` through the public
/// entry point (`remap_column_index`, the function the A11/A15 tests drive).
/// Each call remaps the first not-yet-remapped segment and re-appends it at
/// the end of the manifest index list, so looping until the version stops
/// advancing walks all segments; the terminating call is the idempotent
/// no-op (the A13 gate).
async fn remap_all_segments(dataset: &mut Dataset, prefix: &str) -> (usize, f64, f64) {
    let mut seg = 0usize;
    let mut total_secs = 0f64;
    let noop_secs;
    loop {
        let version_before = dataset.version().version;
        let t = Instant::now();
        lance::dataset::optimize::remapping::remap_column_index(
            dataset,
            &["vec"],
            Some("vec_idx".into()),
        )
        .await
        .unwrap();
        let secs = t.elapsed().as_secs_f64();
        if dataset.version().version == version_before {
            noop_secs = secs;
            break;
        }
        println!("SPBENCH {prefix}remap_seg{seg}_secs={secs:.3}");
        total_secs += secs;
        seg += 1;
        assert!(seg <= 1024, "remap loop failed to converge");
    }
    (seg, total_secs, noop_secs)
}

async fn topk_ids(dataset: &Dataset, query: &[f32], cfg: &Config) -> Vec<u64> {
    let q = Float32Array::from(query.to_vec());
    let mut scan = dataset.scan();
    scan.nearest("vec", &q, cfg.topk).unwrap();
    scan.nprobes(cfg.nprobes);
    scan.project(&["id"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut ids: Vec<u64> = batch["id"]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect();
    ids.sort_unstable();
    ids
}

/// Fresh-session query at `version`, returning the number of `_fri` object
/// reads it performed (A12-style check; needs lance-io's test-util tracking,
/// which the dev-dependency graph enables).
async fn fri_reads_for_query(uri: &str, version: u64, query: &[f32], cfg: &Config) -> usize {
    let dataset = Dataset::open(uri)
        .await
        .unwrap()
        .checkout_version(version)
        .await
        .unwrap();
    let store = dataset.object_store(None).await.unwrap();
    store.io_stats_incremental(); // reset
    let _ = topk_ids(&dataset, query, cfg).await;
    let stats = store.io_stats_incremental();
    stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains("_fri"))
        .count()
}

/// Full pipeline on one dataset; returns (S1, S2, S3, dataset at S3-head).
async fn run_fixture(
    uri: &str,
    dataset_dir: &str,
    cfg: &Config,
    prefix: &str,
    cloud: bool,
) -> (u64, u64, u64, Dataset) {
    assert_eq!(
        cfg.rows % cfg.src_frags,
        0,
        "SPBENCH_ROWS must divide evenly into SPBENCH_SRC_FRAGS"
    );
    assert_eq!(
        cfg.src_frags % cfg.segments,
        0,
        "SPBENCH_SRC_FRAGS must divide evenly into SPBENCH_SEGMENTS"
    );
    assert!(
        cfg.dest_frags <= u16::MAX as usize,
        "row-map labels are u16"
    );

    // Stage 1+2: INGEST + INDEX (interleaved; timings reported per stage).
    let (dataset, s1) = ingest_and_index(uri, cfg, prefix).await;
    println!("SPBENCH {prefix}s1={s1}");

    // Stage 3: SP REWRITE.
    let t = Instant::now();
    let dataset = stable_partition_all(dataset, dataset_dir, cfg).await;
    let sp_secs = t.elapsed().as_secs_f64();
    let s2 = dataset.version().version;
    println!("SPBENCH {prefix}sp_secs={sp_secs:.3}");
    println!(
        "SPBENCH {prefix}sp_rows_per_s={:.0}",
        cfg.rows as f64 / sp_secs.max(1e-9)
    );
    println!("SPBENCH {prefix}s2={s2}");

    // Stage 4: REMAP (tagged remap job; the free-case restamp when the
    // segment fully covers the partition, a clean skip when it only partially
    // covers it).
    if let Some(kb) = vmhwm_kb() {
        println!("SPBENCH {prefix}vmhwm_before_remap_kb={kb}");
    } else {
        println!("SPBENCH {prefix}vmhwm_before_remap_kb=unavailable_on_macos");
    }
    let mut dataset = dataset;
    let (segments_remapped, remap_secs, noop_secs) = remap_all_segments(&mut dataset, prefix).await;
    let s3 = dataset.version().version;
    assert_eq!(
        segments_remapped, cfg.segments,
        "every index segment must be remapped exactly once"
    );
    println!("SPBENCH {prefix}remap_total_secs={remap_secs:.3}");
    println!(
        "SPBENCH {prefix}remap_rows_per_s={:.0}",
        cfg.rows as f64 / remap_secs.max(1e-9)
    );
    println!("SPBENCH {prefix}remap_noop_secs={noop_secs:.6}");
    println!("SPBENCH {prefix}s3={s3}");
    if let Some(kb) = vmhwm_kb() {
        println!("SPBENCH {prefix}vmhwm_after_remap_kb={kb}");
    }

    // Correctness gate: the same topk query at S2 and S3 must return
    // IDENTICAL rows. S2 already returns translated addresses (through the
    // row map), so comparing the `id` values is layout-independent and must
    // match exactly.
    let query = &query_pool(cfg.dim, 1)[0];
    let at_s2 = dataset.checkout_version(s2).await.unwrap();
    let at_s3 = dataset.checkout_version(s3).await.unwrap();
    let ids_s2 = topk_ids(&at_s2, query, cfg).await;
    let ids_s3 = topk_ids(&at_s3, query, cfg).await;
    assert_eq!(
        ids_s2, ids_s3,
        "topk rows must be identical before and after the remap"
    );
    assert_eq!(ids_s2.len(), cfg.topk);
    println!("SPBENCH {prefix}gate=ok");

    // A12-style IO check on fresh sessions: queries at S2 need the row map,
    // queries at S3 must not touch it. Skipped in cloud mode (the check keys on
    // local-fs path walking); correctness is still covered by the gate above.
    if cloud {
        println!("SPBENCH {prefix}fri_reads_query_skipped_cloud");
    } else {
        let fri_s2 = fri_reads_for_query(uri, s2, query, cfg).await;
        let fri_s3 = fri_reads_for_query(uri, s3, query, cfg).await;
        println!("SPBENCH {prefix}fri_reads_query_s2={fri_s2}");
        println!("SPBENCH {prefix}fri_reads_query_s3={fri_s3}");
        assert_eq!(
            fri_s3, 0,
            "queries after the remap must not read the row map"
        );
    }

    (s1, s2, s3, dataset)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "SPBENCH harness; run explicitly with --ignored --nocapture"]
async fn sp_bench() {
    let rows = env_usize("SPBENCH_ROWS", 200_000);
    let cfg = Config {
        rows,
        src_frags: env_usize("SPBENCH_SRC_FRAGS", 20),
        dest_frags: env_usize("SPBENCH_DEST_FRAGS", 20),
        dim: env_usize("SPBENCH_DIM", 64),
        segments: env_usize("SPBENCH_SEGMENTS", 4),
        ivf_partitions: env_usize("SPBENCH_IVF_PARTITIONS", (rows / 1000).clamp(16, 4096)),
        topk: env_usize("SPBENCH_TOPK", 10),
        nprobes: env_usize("SPBENCH_NPROBES", 32),
        index_type: std::env::var("SPBENCH_INDEX_TYPE")
            .unwrap_or_else(|_| "ivf_rq_1bit".to_string()),
    };
    // SPBENCH_CLOUD=1 allows an object-store URI (e.g. az://...). The SP driver
    // already writes the row map through the dataset object store, so cloud
    // backing works; the local-fs `_fri` read-count check is skipped in this mode.
    let cloud = std::env::var("SPBENCH_CLOUD").as_deref() == Ok("1");
    let tmp = tempfile::tempdir().unwrap();
    let dataset_dir = std::env::var("SPBENCH_URI").unwrap_or_else(|_| {
        tmp.path()
            .join("spbench.lance")
            .to_str()
            .unwrap()
            .to_string()
    });
    assert!(
        cloud || !dataset_dir.contains("://") || dataset_dir.starts_with("file://"),
        "SPBENCH_URI must be a local path unless SPBENCH_CLOUD=1; the harness \
         writes the row map and walks _fri via the local filesystem"
    );
    let dataset_dir = dataset_dir
        .strip_prefix("file://")
        .unwrap_or(&dataset_dir)
        .to_string();
    println!("SPBENCH uri={dataset_dir}");
    println!(
        "SPBENCH rows={} src_frags={} dest_frags={} dim={} segments={} ivf_partitions={}",
        cfg.rows, cfg.src_frags, cfg.dest_frags, cfg.dim, cfg.segments, cfg.ivf_partitions
    );

    let (s1, s2, s3, mut dataset) =
        Box::pin(run_fixture(&dataset_dir, &dataset_dir, &cfg, "", cloud)).await;

    // Stage 5 (optional): free-case leg on a second dataset with SEGMENTS=1;
    // its post-remap version is S4.
    let s4 = if std::env::var("SPBENCH_FREECASE").as_deref() == Ok("1") {
        let freecase_dir = format!("{dataset_dir}_freecase");
        println!("SPBENCH freecase_uri={freecase_dir}");
        let freecase_cfg = Config {
            segments: 1,
            ..cfg.clone()
        };
        let (_, _, freecase_s3, _) = Box::pin(run_fixture(
            &freecase_dir,
            &freecase_dir,
            &freecase_cfg,
            "freecase_",
            cloud,
        ))
        .await;
        Some(freecase_s3)
    } else {
        None
    };

    // Stage 6: REBUILD BASELINE, last so S1-S3 stay undisturbed (this commits
    // a new version past S3).
    let t = Instant::now();
    let params = index_params(&cfg);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let rebuild_secs = t.elapsed().as_secs_f64();
    let rebuild_version = dataset.version().version;
    println!("SPBENCH rebuild_secs={rebuild_secs:.3}");
    println!("SPBENCH rebuild_version={rebuild_version} (S1-S3 are earlier versions)");

    // Stage 7: ACCOUNTING.
    let fri_bytes = dir_bytes(&std::path::Path::new(&dataset_dir).join("_fri"));
    println!("SPBENCH fri_bytes={fri_bytes}");

    // Stage 8: summary for the read bench.
    match s4 {
        Some(s4) => println!("SPBENCH S1={s1} S2={s2} S3={s3} S4={s4}"),
        None => println!("SPBENCH S1={s1} S2={s2} S3={s3}"),
    }
}
