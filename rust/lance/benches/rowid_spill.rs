// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What it costs to keep a fragment's stable row id sequence in the manifest,
//! and what changes when the sequence spills to a hidden data file column.
//!
//! Two workloads, because they sit at opposite ends of what the run encoding
//! can do with a sequence:
//!
//! - `deleted`: rows deleted and then compacted, the workload behind the
//!   numbers in <https://github.com/lance-format/lance/issues/8621>. The
//!   deletions become holes, so the sequence encodes as a range plus a bitmap
//!   and costs a fraction of a byte per row.
//! - `shuffled`: every row rewritten in random order, which is what a
//!   reclustering pass leaves behind. Each fragment's rows now come from all
//!   over the table, so there is no run structure left and the sequence falls
//!   back to `U64Segment::Array`, at four bytes per row on the wire. This is
//!   the worst case for keeping sequences inline.
//!
//! Each workload runs two arms, and within a workload the arms differ only in
//! `CompactionOptions::inline_row_ids_max_bytes`: `Some(usize::MAX)` never
//! spills, which is today's behavior, and `None` takes the format's 200 KiB
//! inline budget, which is what this change makes the default. Both arms
//! measure the case for the design and the costs it adds:
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
//! - `BENCH_DELETE_PERCENT`: percentage of rows deleted before compaction in
//!   the `deleted` workload (default 30). Deletions are what turn the sequences
//!   into something the run encoding cannot compress into a plain range.
//! - `BENCH_APPENDS`: appends timed for the commit-latency figure (default 5).
//! - `BENCH_SCENARIOS`: comma-separated subset of `deleted,shuffled` to run
//!   (default both).

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::{get_row_id_index, load_row_id_sequence};
use lance::dataset::{
    CommitBuilder, Dataset, InsertBuilder, ProjectionRequest, WriteMode, WriteParams,
};
use lance::session::Session;
use lance_core::ROW_ID;
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::feature_flags::ENABLE_UNSTABLE_SPILLED_ROW_IDS_ENV;
use lance_table::format::{RowDatasetVersionMeta, RowIdMeta};
use lance_table::rowids::{RowIdSequence, write_row_ids};
use lance_table::transaction::{Operation, RewriteGroup, TransactionBuilder};
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

/// How the table is left before compaction, which decides what the row id
/// sequences look like and so what keeping them inline costs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    /// A slice of every fragment deleted, then compacted. The surviving ids are
    /// still ascending, so the sequence encodes as a range plus a bitmap of
    /// holes.
    Deleted,
    /// Every row rewritten in a random order. The ids in a fragment are no
    /// longer ascending or contiguous, so the encoding degrades to a bitpacked
    /// array of absolute values.
    Shuffled,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::Deleted => "deleted",
            Self::Shuffled => "shuffled",
        }
    }
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
    transaction_bytes: u64,
    inline_row_id_bytes: u64,
    inline_row_version_bytes: u64,
    rows: u64,
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

/// `value` is a pure function of `id` so that a `take` by row id can be checked
/// against the id it resolved to, which is what proves a spilled sequence maps
/// rows to the same places the inline one did.
fn value_of(id: i64) -> i64 {
    id * 3
}

fn batch(schema: Arc<ArrowSchema>, start: i64, len: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start..(start + len as i64));
    let values = Int64Array::from_iter_values((start..(start + len as i64)).map(value_of));
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

/// Write the table one fragment per commit, so the fragments come from
/// different versions the way an incrementally loaded table's would.
async fn write_table(uri: &str, config: Config) -> Dataset {
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
    dataset.unwrap()
}

/// Every row of the table paired with its stable row id.
async fn read_rows_with_ids(dataset: &Dataset) -> Vec<(u64, i64)> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    scanner.project(&["id"]).unwrap();
    let mut stream = scanner.try_into_stream().await.unwrap();

    let mut rows = Vec::with_capacity(dataset.count_rows(None).await.unwrap());
    while let Some(batch) = stream.try_next().await.unwrap() {
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>();
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        rows.extend(
            row_ids
                .values()
                .iter()
                .copied()
                .zip(ids.values().iter().copied()),
        );
    }
    rows
}

/// Fisher-Yates driven by SplitMix64, so the permutation is the same on every
/// run and both arms of a scenario shuffle identically. Written out rather than
/// depending on `rand`, which the `lance` crate does not otherwise use.
fn shuffle<T>(items: &mut [T]) {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        items.swap(i, (next() % (i as u64 + 1)) as usize);
    }
}

/// Rewrite the whole table in a random row order, keeping every row's stable
/// row id.
///
/// This is the shape a reclustering pass leaves behind, and it goes through the
/// same `Operation::Rewrite` a compaction commits: the new fragments carry the
/// permuted sequences, which is how such a pass would have to preserve row ids.
/// The fragments are written at half the target size so that the compaction
/// that follows has neighbors to merge, and therefore rechunks -- and in the
/// spilled arm spills -- every sequence.
async fn shuffle_rewrite(dataset: Dataset, config: Config) -> Dataset {
    let mut rows = read_rows_with_ids(&dataset).await;
    shuffle(&mut rows);

    let rows_per_file = config.rows_per_fragment / 2;
    let arrow_schema = schema();
    let shuffled_ids: Vec<u64> = rows.iter().map(|(row_id, _)| *row_id).collect();
    let batches: Vec<RecordBatch> = rows
        .chunks(rows_per_file)
        .map(|chunk| {
            let ids = Int64Array::from_iter_values(chunk.iter().map(|(_, id)| *id));
            let values = Int64Array::from_iter_values(chunk.iter().map(|(_, id)| value_of(*id)));
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(ids), Arc::new(values)])
                .unwrap()
        })
        .collect();
    drop(rows);

    let dataset = Arc::new(dataset);
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), arrow_schema);
    let uncommitted = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            enable_stable_row_ids: true,
            max_rows_per_file: rows_per_file,
            skip_auto_cleanup: true,
            ..Default::default()
        })
        .execute_uncommitted_stream(reader)
        .await
        .unwrap();

    let mut new_fragments = match uncommitted.operation {
        Operation::Append { fragments } => fragments,
        other => panic!("uncommitted write produced {other:?}, expected an append"),
    };

    // The uncommitted write handed each new fragment a fresh range of row ids.
    // Replace them with the ids the rows actually carry, sliced in the order
    // the fragments were written.
    let mut offset = 0;
    for fragment in new_fragments.iter_mut() {
        let rows_in_fragment = fragment.physical_rows.unwrap();
        let sequence = RowIdSequence::from(&shuffled_ids[offset..offset + rows_in_fragment]);
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
        offset += rows_in_fragment;
    }
    assert_eq!(
        offset,
        shuffled_ids.len(),
        "the new fragments hold {offset} rows but the table has {}",
        shuffled_ids.len()
    );

    let transaction = TransactionBuilder::new(
        dataset.version().version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: dataset.manifest.fragments.as_ref().clone(),
                new_fragments,
            }],
            rewritten_indices: Vec::new(),
            frag_reuse_index: None,
        },
    )
    .build();

    CommitBuilder::new(dataset)
        .with_skip_auto_cleanup(true)
        .execute(transaction)
        .await
        .unwrap()
}

/// The table as the scenario leaves it, ready for the compaction that decides
/// where each sequence lands.
async fn build_base(uri: &str, config: Config, scenario: Scenario) -> Dataset {
    let mut dataset = write_table(uri, config).await;
    match scenario {
        Scenario::Deleted => {
            // Spread the deletions across every fragment rather than truncating
            // a prefix: a hole every few rows is what stops the sequence from
            // encoding as a range.
            dataset
                .delete(&format!("id % 100 < {}", config.delete_percent))
                .await
                .unwrap();
            dataset
        }
        Scenario::Shuffled => shuffle_rewrite(dataset, config).await,
    }
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

async fn run_arm(
    dir: &str,
    config: Config,
    scenario: Scenario,
    inline_row_ids_max_bytes: Option<usize>,
) -> ArmResult {
    let uri = dir.to_string();
    let mut dataset = build_base(&uri, config, scenario).await;

    // A commit writes the whole transaction to its own file under
    // `_transactions/` as well as putting the fragment list in the manifest, so
    // the compaction's transaction blob is a write cost the manifest row does
    // not show.
    let transactions_before = dir_bytes(&format!("{dir}/_transactions"));

    let started = Instant::now();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: config.rows_per_fragment,
            // Only bites in the `deleted` scenario; the `shuffled` one has no
            // deletions and is picked up by the below-target rule instead.
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
    let transaction_bytes =
        dir_bytes(&format!("{dir}/_transactions")).saturating_sub(transactions_before);

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

    let rows = dataset.count_rows(None).await.unwrap() as u64;
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
    let taken = warm.take_rows(&[probe_id], projection).await.unwrap();
    let take_warm = started.elapsed();
    drop(index);

    // Row ids were handed out in `id` order, so the row a spilled sequence
    // resolves has to be the one whose `value` matches. Cheap, and it is the
    // only thing separating a fast answer from a correct one.
    let value = taken
        .column_by_name("value")
        .unwrap()
        .as_primitive::<Int64Type>()
        .value(0);
    assert_eq!(
        value,
        value_of(probe_id as i64),
        "row id {probe_id} resolved to a row holding {value}"
    );

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
        transaction_bytes,
        inline_row_id_bytes,
        inline_row_version_bytes,
        rows,
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

fn report(scenario: Scenario, config: Config, inline: &ArmResult, spilled: &ArmResult) {
    println!();
    match scenario {
        Scenario::Deleted => println!(
            "--- {}: {} fragments x {} rows, {}% deleted then compacted ---",
            scenario.name(),
            config.fragments,
            config.rows_per_fragment,
            config.delete_percent
        ),
        Scenario::Shuffled => println!(
            "--- {}: {} fragments x {} rows, rewritten in random order then compacted ---",
            scenario.name(),
            config.fragments,
            config.rows_per_fragment
        ),
    }
    println!();

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
        "compaction transaction file",
        mib(inline.transaction_bytes),
        mib(spilled.transaction_bytes),
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
    // Bytes per row is what makes the two scenarios comparable: it is the
    // encoding's cost for one id, and it is what decides whether a sequence
    // belongs in the manifest at all.
    println!(
        "{} rows: {:.2} B/row inline in the manifest, {:.2} B/row in the spilled column",
        inline.rows,
        inline.inline_row_id_bytes as f64 / inline.rows as f64,
        spilled.data_bytes.saturating_sub(inline.data_bytes) as f64 / spilled.rows as f64,
    );
    // A commit puts the fragment list in the manifest and writes the whole
    // transaction to its own file, so an inline sequence is written twice per
    // commit. It is written a third time when the transaction serializes under
    // `MAX_INLINE_TRANSACTION_BYTES` (20 MiB), because it is then copied into
    // the manifest as well (`Manifest::transaction_section`) -- which is why
    // the small-sequence scenario shows a manifest delta of twice its row ids
    // and the large one does not.
    println!(
        "commit blobs for the compaction: inline {:.2}M manifest + {:.2}M transaction file, \
         spilled {:.2}M + {:.2}M; row ids inline are {:.2}M",
        mib(inline.manifest_bytes),
        mib(inline.transaction_bytes),
        mib(spilled.manifest_bytes),
        mib(spilled.transaction_bytes),
        mib(inline.inline_row_id_bytes),
    );
    println!(
        "spilled fragments: inline arm {}/{}, spilled arm {}/{}",
        inline.spilled_fragments,
        inline.total_fragments,
        spilled.spilled_fragments,
        spilled.total_fragments
    );
}

fn scenarios_from_env() -> Vec<Scenario> {
    let all = [Scenario::Deleted, Scenario::Shuffled];
    let Ok(requested) = std::env::var("BENCH_SCENARIOS") else {
        return all.to_vec();
    };
    let selected: Vec<Scenario> = all
        .into_iter()
        .filter(|scenario| {
            requested
                .split(',')
                .any(|name| name.trim() == scenario.name())
        })
        .collect();
    assert!(
        !selected.is_empty(),
        "BENCH_SCENARIOS={requested} selected none of: deleted, shuffled"
    );
    selected
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

    for scenario in scenarios_from_env() {
        // `usize::MAX` reproduces the behavior on main, where a compacted
        // fragment's sequence always stays inline however large it grows.
        // `None` takes the format's documented 200 KiB inline budget, which is
        // what this change makes the default.
        let inline_dir = tempfile::tempdir().unwrap();
        let inline = runtime.block_on(run_arm(
            &inline_dir.path().to_string_lossy(),
            config,
            scenario,
            Some(usize::MAX),
        ));

        let spill_dir = tempfile::tempdir().unwrap();
        let spilled = runtime.block_on(run_arm(
            &spill_dir.path().to_string_lossy(),
            config,
            scenario,
            None,
        ));

        report(scenario, config, &inline, &spilled);
    }
}

criterion_group!(benches, bench_rowid_spill);
criterion_main!(benches);
