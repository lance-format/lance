// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What it costs to keep a fragment's stable row id sequence in the manifest,
//! and what changes when the sequence spills to a hidden data file column.
//!
//! The workload is the one that produced the numbers in
//! <https://github.com/lance-format/lance/issues/8621>: a stable-row-id table
//! that has had rows deleted and then compacted, so the deletions become holes
//! in the row id sequences and the sequences stop being plain ranges. From
//! there the benchmark measures, per arm, both the case for the design and the
//! costs it adds:
//!
//! 1. manifest bytes -- the claim is that this stops growing with the table
//! 2. cold dataset open -- every reader pays the manifest decode
//! 3. commit latency for one small append -- every writer rewrites the manifest
//! 4. reading a sequence back, split into loading one fragment's sequence, the
//!    dataset-wide row id index build that a query pays before it can resolve
//!    an id, and the `take` itself once that index exists -- the read cost the
//!    design adds
//! 5. compaction wall time and bytes written -- the write cost it adds
//!
//! The two arms differ only in `CompactionOptions::inline_row_ids_max_bytes`:
//! `Some(usize::MAX)` never spills, which is today's behavior, and `None` takes
//! the format's 200 KiB inline budget, which is what this change makes the
//! default.
//!
//! ## Running
//!
//! Spilled row ids are an unstable feature, so a release build -- which is what
//! `cargo bench` produces -- has to opt in:
//!
//! ```bash
//! LANCE_ENABLE_UNSTABLE_SPILLED_ROW_IDS=1 cargo bench --bench rowid_spill
//! ```
//!
//! ## Configuration
//!
//! - `BENCH_FRAGMENTS`: fragments to write (default 8).
//! - `BENCH_ROWS_PER_FRAGMENT`: rows in each (default 1,000,000).
//! - `BENCH_DELETE_PERCENT`: percentage of rows deleted before compaction
//!   (default 30). Deletions are what turn the sequences into something the
//!   run encoding cannot compress.
//! - `BENCH_APPENDS`: appends timed for the commit-latency figure (default 5).

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::{get_row_id_index, load_row_id_sequence};
use lance::dataset::{Dataset, ProjectionRequest, WriteMode, WriteParams};
use lance::session::Session;
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::feature_flags::ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV;
use lance_table::format::{RowDatasetVersionMeta, RowIdMeta};
use tokio::runtime::Runtime;

const DEFAULT_FRAGMENTS: usize = 8;
const DEFAULT_ROWS_PER_FRAGMENT: usize = 1_000_000;
const DEFAULT_DELETE_PERCENT: usize = 30;
const DEFAULT_APPENDS: usize = 5;

/// Repeats for the cold-open figure, which is sub-millisecond on local disk.
const OPEN_SAMPLES: usize = 10;

/// Repeats for the two cold sequence-read figures.
const READ_SAMPLES: usize = 3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[derive(Clone, Copy)]
struct Config {
    fragments: usize,
    rows_per_fragment: usize,
    delete_percent: usize,
    appends: usize,
}

impl Config {
    fn from_env() -> Self {
        Self {
            fragments: env_usize("BENCH_FRAGMENTS", DEFAULT_FRAGMENTS),
            rows_per_fragment: env_usize("BENCH_ROWS_PER_FRAGMENT", DEFAULT_ROWS_PER_FRAGMENT),
            delete_percent: env_usize("BENCH_DELETE_PERCENT", DEFAULT_DELETE_PERCENT),
            appends: env_usize("BENCH_APPENDS", DEFAULT_APPENDS),
        }
    }
}

struct ArmResult {
    compaction: Duration,
    data_bytes: u64,
    manifest_bytes: u64,
    inline_row_id_bytes: u64,
    inline_row_version_bytes: u64,
    open: Duration,
    commit: Duration,
    load_sequence: Duration,
    index_build: Duration,
    take_warm: Duration,
    spilled_fragments: usize,
    total_fragments: usize,
}

/// Encoded bytes each fragment keeps inline in the manifest, split by which of
/// the three per-row sequence families they belong to. The row version families
/// are the other two columns section 5.3 moves; this change only moves row ids,
/// so the split says how much of the manifest is left behind.
fn inline_sequence_bytes(dataset: &Dataset) -> (u64, u64) {
    let mut row_ids = 0;
    let mut versions = 0;
    for fragment in dataset.manifest.fragments.iter() {
        if let Some(RowIdMeta::Inline(data)) = &fragment.row_id_meta {
            row_ids += data.len() as u64;
        }
        for meta in [
            &fragment.created_at_version_meta,
            &fragment.last_updated_at_version_meta,
        ]
        .into_iter()
        .flatten()
        {
            if let RowDatasetVersionMeta::Inline(data) = meta {
                versions += data.len() as u64;
            }
        }
    }
    (row_ids, versions)
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn batch(schema: Arc<ArrowSchema>, start: i64, len: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start..(start + len as i64));
    let values = Int64Array::from_iter_values((start..(start + len as i64)).map(|i| i * 3));
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)]).unwrap()
}

/// A fresh session with no caches, so each measurement pays the real decode
/// rather than reading back what the previous one memoized.
fn cold_session() -> Arc<Session> {
    Arc::new(Session::new(0, 0, Arc::new(ObjectStoreRegistry::default())))
}

async fn open_cold(uri: &str) -> Dataset {
    lance::dataset::builder::DatasetBuilder::from_uri(uri)
        .with_session(cold_session())
        .load()
        .await
        .unwrap()
}

/// Write the table, then delete a slice of every fragment so that compaction
/// has holes to materialize.
async fn build_base(uri: &str, config: Config) -> Dataset {
    let schema = schema();
    let mut dataset = None;
    for fragment in 0..config.fragments {
        let start = (fragment * config.rows_per_fragment) as i64;
        let data = batch(schema.clone(), start, config.rows_per_fragment);
        let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
        dataset = Some(
            Dataset::write(
                reader,
                uri,
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: config.rows_per_fragment,
                    mode: if fragment == 0 {
                        WriteMode::Create
                    } else {
                        WriteMode::Append
                    },
                    skip_auto_cleanup: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
    }

    let mut dataset = dataset.unwrap();
    // Spread the deletions across every fragment rather than truncating a
    // prefix: a hole every few rows is what stops the sequence from encoding
    // as a range.
    dataset
        .delete(&format!("id % 100 < {}", config.delete_percent))
        .await
        .unwrap();
    dataset
}

fn dir_bytes(path: &str) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            total += dir_bytes(&entry.path().to_string_lossy());
        } else {
            total += metadata.len();
        }
    }
    total
}

/// Size of the manifest for exactly `version`, which is the blob a commit at
/// that version rewrote and every reader of it downloads.
///
/// Both manifest naming schemes live under `_versions/`: V1 names the file after
/// the version, V2 after `u64::MAX - version`, zero padded.
fn manifest_bytes(dir: &str, version: u64) -> u64 {
    let versions = format!("{dir}/_versions");
    let candidates = [
        format!("{versions}/{version}.manifest"),
        format!("{versions}/{:020}.manifest", u64::MAX - version),
    ];
    candidates
        .iter()
        .find_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

async fn run_arm(dir: &str, config: Config, inline_row_ids_max_bytes: Option<usize>) -> ArmResult {
    let uri = dir.to_string();
    let mut dataset = build_base(&uri, config).await;

    let started = Instant::now();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: config.rows_per_fragment,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.0,
            inline_row_ids_max_bytes,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let compaction = started.elapsed();

    let total_fragments = dataset.get_fragments().len();
    let spilled_fragments = dataset
        .get_fragments()
        .iter()
        .filter(|fragment| {
            fragment
                .metadata()
                .row_id_meta
                .as_ref()
                .is_some_and(|meta| meta.column_file().is_some())
        })
        .count();

    let manifest_bytes = manifest_bytes(dir, dataset.version().version);
    let (inline_row_id_bytes, inline_row_version_bytes) = inline_sequence_bytes(&dataset);
    // Captured before the appends below, and while the pre-compaction files are
    // still on disk in both arms. Both arms wrote identical user data, so the
    // difference between them is exactly what the spilled columns cost.
    let data_bytes = dir_bytes(&format!("{dir}/data"));

    // A cold open is sub-millisecond on a local filesystem, so a single sample
    // is mostly noise. Averaged, since this is one of the figures the design is
    // argued on.
    let started = Instant::now();
    for _ in 0..OPEN_SAMPLES {
        let _ = open_cold(&uri).await;
    }
    let open = started.elapsed() / OPEN_SAMPLES as u32;

    // One fragment's sequence, read cold. This is the work the design moves out
    // of the manifest decode and into a data file read, so it is measured on its
    // own rather than only through the query that depends on it. The last
    // fragment, so resolving its ids cannot short-circuit on the first one.
    let mut probe_id = 0;
    let mut load_total = Duration::ZERO;
    for _ in 0..READ_SAMPLES {
        // Opened outside the timed region: this row is the sequence read, not
        // the manifest decode that `open` already reports.
        let cold = open_cold(&uri).await;
        let last = cold.get_fragments().pop().unwrap();
        let started = Instant::now();
        let sequence = load_row_id_sequence(&cold, last.metadata()).await.unwrap();
        load_total += started.elapsed();
        probe_id = sequence.iter().next_back().unwrap();
    }
    let load_sequence = load_total / READ_SAMPLES as u32;

    // The dataset-wide row id index, which is what any query that resolves a row
    // id pays for before it can touch data.
    let mut index_total = Duration::ZERO;
    for _ in 0..READ_SAMPLES {
        let cold = open_cold(&uri).await;
        let started = Instant::now();
        let _ = get_row_id_index(&cold).await.unwrap();
        index_total += started.elapsed();
    }
    let index_build = index_total / READ_SAMPLES as u32;

    // Once the index exists the two arms follow the same path, so this row is a
    // control: it should not move. It needs a session that actually caches --
    // `open_cold` gives the caches zero capacity, so on that dataset `take_rows`
    // would rebuild the index and re-measure `index_build`. Holding `index`
    // alive across the take keeps it from being evicted.
    let warm = Dataset::open(&uri).await.unwrap();
    let index = get_row_id_index(&warm).await.unwrap();
    let projection = ProjectionRequest::from_columns(["value"], warm.schema());
    let started = Instant::now();
    warm.take_rows(&[probe_id], projection).await.unwrap();
    let take_warm = started.elapsed();
    drop(index);

    let schema = schema();
    let mut commit_total = Duration::ZERO;
    for append in 0..config.appends {
        let data = batch(schema.clone(), 1_000_000_000 + append as i64 * 10, 10);
        let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
        let started = Instant::now();
        Dataset::write(
            reader,
            &uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                skip_auto_cleanup: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        commit_total += started.elapsed();
    }

    ArmResult {
        compaction,
        data_bytes,
        manifest_bytes,
        inline_row_id_bytes,
        inline_row_version_bytes,
        open,
        commit: commit_total / config.appends as u32,
        load_sequence,
        index_build,
        take_warm,
        spilled_fragments,
        total_fragments,
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn bench_rowid_spill(_c: &mut Criterion) {
    if std::env::var_os(ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV).is_none() && !cfg!(debug_assertions) {
        panic!(
            "set {ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV}=1 to run this benchmark: spilled row ids \
             are an unstable feature and a release build refuses the dataset without it"
        );
    }

    let config = Config::from_env();
    let runtime = Runtime::new().unwrap();

    println!("=== Row id sequence placement ===");
    println!(
        "{} fragments x {} rows, {}% deleted then compacted",
        config.fragments, config.rows_per_fragment, config.delete_percent
    );
    println!();

    // `usize::MAX` reproduces the behavior on main, where a compacted fragment's
    // sequence always stays inline however large it grows. `None` takes the
    // format's documented 200 KiB inline budget, which is what this change makes
    // the default.
    let inline_dir = tempfile::tempdir().unwrap();
    let inline = runtime.block_on(run_arm(
        &inline_dir.path().to_string_lossy(),
        config,
        Some(usize::MAX),
    ));

    let spill_dir = tempfile::tempdir().unwrap();
    let spilled = runtime.block_on(run_arm(&spill_dir.path().to_string_lossy(), config, None));

    println!(
        "{:<28} {:>14} {:>14} {:>10}",
        "metric", "inline", "spilled", "ratio"
    );
    let row = |name: &str, inline: f64, spilled: f64, unit: &str| {
        println!(
            "{:<28} {:>12.2}{unit:<2} {:>12.2}{unit:<2} {:>9.2}x",
            name,
            inline,
            spilled,
            if spilled == 0.0 {
                f64::INFINITY
            } else {
                inline / spilled
            }
        );
    };
    row(
        "manifest size",
        mib(inline.manifest_bytes),
        mib(spilled.manifest_bytes),
        "M",
    );
    row(
        "  of which row ids",
        mib(inline.inline_row_id_bytes),
        mib(spilled.inline_row_id_bytes),
        "M",
    );
    row(
        "  of which row versions",
        mib(inline.inline_row_version_bytes),
        mib(spilled.inline_row_version_bytes),
        "M",
    );
    row(
        "cold dataset open",
        inline.open.as_secs_f64() * 1e3,
        spilled.open.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "append commit (mean)",
        inline.commit.as_secs_f64() * 1e3,
        spilled.commit.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "load one sequence (cold)",
        inline.load_sequence.as_secs_f64() * 1e3,
        spilled.load_sequence.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "row id index build (cold)",
        inline.index_build.as_secs_f64() * 1e3,
        spilled.index_build.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "take by row id (index built)",
        inline.take_warm.as_secs_f64() * 1e3,
        spilled.take_warm.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "compaction",
        inline.compaction.as_secs_f64() * 1e3,
        spilled.compaction.as_secs_f64() * 1e3,
        "ms",
    );
    row(
        "data files on disk",
        mib(inline.data_bytes),
        mib(spilled.data_bytes),
        "M",
    );
    println!();
    // The commit that creates the fragments writes them into the manifest file
    // twice: once in the fragment list, once in the inline transaction section
    // (`Manifest::transaction_section`). So a byte of inline sequence costs two
    // bytes of manifest at that version, and one byte at every version after.
    println!(
        "manifest holds each sequence twice at the creating commit, so the inline arm's \
         {:.2}M of row ids accounts for {:.2}M of the {:.2}M manifest delta",
        mib(inline.inline_row_id_bytes),
        mib(2 * inline.inline_row_id_bytes),
        mib(inline.manifest_bytes.saturating_sub(spilled.manifest_bytes)),
    );
    println!(
        "spilled fragments: inline arm {}/{}, spilled arm {}/{}",
        inline.spilled_fragments,
        inline.total_fragments,
        spilled.spilled_fragments,
        spilled.total_fragments
    );
}

criterion_group!(benches, bench_rowid_spill);
criterion_main!(benches);
