// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What it costs to keep a fragment's row lineage -- its stable row id
//! sequence and its created-at and last-updated-at version sequences -- in the
//! manifest, and what changes when those sequences spill to hidden columns of
//! the fragment's data file.
//!
//! Two workloads, because they sit at opposite ends of what the run encoding
//! can do with a sequence:
//!
//! - `deleted`: rows deleted and then compacted, the workload behind the
//!   numbers in <https://github.com/lance-format/lance/issues/8621>. The
//!   deletions become holes, so the row id sequence encodes as a range plus a
//!   bitmap and costs a fraction of a byte per row. A compacted fragment merges
//!   a few whole fragments, so its version sequences stay a few runs long.
//! - `shuffled`: every row rewritten in random order, which is what a
//!   reclustering pass leaves behind. Each fragment's rows now come from all
//!   over the table, so there is no run structure left: the row id sequence
//!   falls back to `U64Segment::Array`, at four bytes per row on the wire, and
//!   the rows' created-at versions interleave, so most rows start a new version
//!   run. This is the worst case for keeping sequences inline. At the defaults
//!   the inline arm carries about 200 MB of version runs in its manifest, which
//!   makes that arm much slower to open and to commit to.
//!
//! Each workload runs two arms, and within a workload the arms differ only in
//! the table config: the inline arm never opts in, which is today's behavior,
//! and the spilled arm sets `lance.row_lineage.spill=true` and takes the
//! format's 200 KiB inline budget unless `BENCH_INLINE_MAX_BYTES` overrides it.
//! Both arms measure the case for the design and the costs it adds:
//!
//! 1. manifest bytes -- the claim is that this stops growing with the table.
//!    Taken from the first append after the compaction, which is the manifest
//!    every later reader decodes and every later commit rewrites. The
//!    compaction's own manifest file can also hold a copy of its transaction
//!    that no reader downloads, so it is reported on its own row
//! 2. cold dataset open -- every reader pays the manifest decode
//! 3. commit latency for one small append -- every writer rewrites the
//!    manifest. Only the commit is timed; the append's data file is written
//!    before the timer starts
//! 4. reading a sequence back, split into loading one fragment's sequence and
//!    the dataset-wide row id index build that a query pays before it can
//!    resolve an id -- the read cost the design adds. The `take` that follows
//!    once the index and the fragment's sequence are cached is a control: it
//!    should cost the same in both arms
//! 5. compaction wall time and the bytes it wrote -- the write cost it adds
//!
//! Timings are reported as the median and the minimum of their samples.
//!
//! The benchmark also checks its premise: the spilled arm must have spilled
//! something and the inline arm nothing, and after the compaction a full scan
//! compares every row's row id and versions with the ones the workload handed
//! out.
//!
//! ## Running
//!
//! Spilled row lineage is an unstable feature, so a release build -- which is
//! what `cargo bench` produces -- has to opt in:
//!
//! ```bash
//! LANCE_ENABLE_UNSTABLE_SPILLED_ROW_LINEAGE=1 cargo bench --bench rowid_spill
//! # A quick run. At this size the deleted table fits the default inline
//! # budget, so the budget is set to zero to keep something spilling.
//! BENCH_FRAGMENTS=2 BENCH_ROWS_PER_FRAGMENT=200000 BENCH_INLINE_MAX_BYTES=0 \
//!   LANCE_ENABLE_UNSTABLE_SPILLED_ROW_LINEAGE=1 cargo bench --bench rowid_spill
//! ```
//!
//! ## Configuration
//!
//! - `BENCH_FRAGMENTS`: fragments to write (default 8).
//! - `BENCH_ROWS_PER_FRAGMENT`: rows in each (default 1,000,000).
//! - `BENCH_DELETE_PERCENT`: percentage of rows deleted before compaction in
//!   the `deleted` workload, 1 to 99 (default 30). Deletions are what turn the
//!   sequences into something the run encoding cannot compress into a plain
//!   range.
//! - `BENCH_APPENDS`: appends timed for the commit-latency figure (default 5).
//! - `BENCH_INLINE_MAX_BYTES`: the spilled arm's inline budget in bytes,
//!   written to `lance.row_lineage.inline_max_bytes` (default: the format's
//!   200 KiB). A small table's sequences can all fit the default budget, which
//!   would leave the spilled arm nothing to measure and stops the benchmark,
//!   so a quick run sets this to 0.
//! - `BENCH_SCENARIOS`: comma-separated subset of `deleted,shuffled` to run
//!   (default both).
//!
//! A value that does not parse or is out of range stops the benchmark before
//! any data is written. A spilled arm that spills nothing stops it after that
//! arm's compaction.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::{
    DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES, INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY,
    SPILL_ROW_LINEAGE_CONFIG_KEY, get_row_id_index, load_row_id_sequence,
};
use lance::dataset::{
    CommitBuilder, Dataset, InsertBuilder, ProjectionRequest, WriteMode, WriteParams,
};
use lance::session::Session;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::feature_flags::ENABLE_UNSTABLE_SPILLED_ROW_LINEAGE_ENV;
use lance_table::format::{RowDatasetVersionMeta, RowDatasetVersionSequence, RowIdMeta};
use lance_table::rowids::version::write_dataset_versions;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use lance_table::transaction::{Operation, RewriteGroup, TransactionBuilder};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use tokio::runtime::Runtime;

const DEFAULT_FRAGMENTS: usize = 8;
const DEFAULT_ROWS_PER_FRAGMENT: usize = 1_000_000;
const DEFAULT_DELETE_PERCENT: usize = 30;
const DEFAULT_APPENDS: usize = 5;

/// Repeats for the cold-open figure, which is sub-millisecond on local disk
/// for a small manifest.
const OPEN_SAMPLES: usize = 10;

/// Repeats for the two cold sequence-read figures.
const READ_SAMPLES: usize = 10;

/// Row ids taken, one per call, for the take figure.
const TAKE_PROBES: usize = 16;

/// The value of `name`, or `None` when it is unset. A value that does not
/// parse stops the benchmark: falling back to the default would quietly run,
/// say, `BENCH_ROWS_PER_FRAGMENT=1e5` at a million rows.
fn env_usize(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok()?;
    match value.parse() {
        Ok(parsed) => Some(parsed),
        Err(error) => panic!("{name}={value:?} is not a non-negative integer: {error}"),
    }
}

/// How the table is left before compaction, which decides what the lineage
/// sequences look like and so what keeping them inline costs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    /// A slice of every fragment deleted, then compacted. The surviving ids are
    /// still ascending, so the sequence encodes as a range plus a bitmap of
    /// holes.
    Deleted,
    /// Every row rewritten in a random order. The ids in a fragment are no
    /// longer ascending or contiguous, so the encoding degrades to a bitpacked
    /// array of absolute values, and neighboring rows no longer share a
    /// created-at version.
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
    /// The spilled arm's inline budget, or `None` for the format's default.
    inline_max_bytes: Option<usize>,
}

impl Config {
    fn from_env() -> Self {
        Self {
            fragments: env_usize("BENCH_FRAGMENTS").unwrap_or(DEFAULT_FRAGMENTS),
            rows_per_fragment: env_usize("BENCH_ROWS_PER_FRAGMENT")
                .unwrap_or(DEFAULT_ROWS_PER_FRAGMENT),
            delete_percent: env_usize("BENCH_DELETE_PERCENT").unwrap_or(DEFAULT_DELETE_PERCENT),
            appends: env_usize("BENCH_APPENDS").unwrap_or(DEFAULT_APPENDS),
            inline_max_bytes: env_usize("BENCH_INLINE_MAX_BYTES"),
        }
    }

    /// Reject a configuration that would fail, or measure nothing, only after
    /// minutes of setup.
    fn validate(self, scenarios: &[Scenario]) {
        assert!(
            self.fragments >= 1,
            "BENCH_FRAGMENTS={} must be at least 1",
            self.fragments
        );
        assert!(
            self.appends >= 1,
            "BENCH_APPENDS={} must be at least 1: the commit figures and the steady-state \
             manifest both come from the appends",
            self.appends
        );
        if scenarios.contains(&Scenario::Shuffled) {
            assert!(
                self.rows_per_fragment >= 2,
                "BENCH_ROWS_PER_FRAGMENT={} must be at least 2 for the shuffled scenario, which \
                 rewrites the table in fragments of half that size",
                self.rows_per_fragment
            );
        }
        if scenarios.contains(&Scenario::Deleted) {
            assert!(
                (1..=99).contains(&self.delete_percent),
                "BENCH_DELETE_PERCENT={} must be between 1 and 99 for the deleted scenario: 0 \
                 leaves the compaction nothing to do and 100 deletes the whole table",
                self.delete_percent
            );
            // A row survives when `id % 100 >= delete_percent`, so with at most 99
            // percent deleted the row with id 99 always does; a smaller table can
            // lose every row.
            assert!(
                self.fragments.saturating_mul(self.rows_per_fragment) >= 100,
                "BENCH_FRAGMENTS x BENCH_ROWS_PER_FRAGMENT = {} x {} must reach 100 rows for the \
                 deleted scenario, or the deletions can remove every row",
                self.fragments,
                self.rows_per_fragment
            );
        }
    }
}

/// Fragments whose sequence of each lineage family left the manifest. The
/// families are planned on their own sizes, so a workload can spill one and
/// keep another inline.
struct SpilledFragments {
    row_ids: usize,
    created_at: usize,
    last_updated_at: usize,
}

impl SpilledFragments {
    fn any(&self) -> bool {
        self.row_ids + self.created_at + self.last_updated_at > 0
    }
}

struct ArmResult {
    compaction: Duration,
    compaction_output_bytes: u64,
    manifest_bytes: u64,
    compaction_manifest_bytes: u64,
    transaction_bytes: u64,
    inline_row_id_bytes: u64,
    inline_row_version_bytes: u64,
    rows: u64,
    open: Vec<Duration>,
    commit: Vec<Duration>,
    load_sequence: Vec<Duration>,
    index_build: Vec<Duration>,
    take: Vec<Duration>,
    spilled_fragments: SpilledFragments,
    total_fragments: usize,
}

/// Encoded bytes each fragment keeps inline in the manifest, split by which of
/// the three per-row sequence families they belong to. The row version families
/// are spilled on their own size, so the split shows which sequences a
/// workload actually moves.
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

/// The version `write_table` created the row with `id` at: the first fragment
/// is created at version 1 and each later one appended at the next version.
fn created_at_version(id: u64, config: Config) -> u64 {
    id / config.rows_per_fragment as u64 + 1
}

/// Rewrite the whole table in a random row order, keeping every row's stable
/// row id and the version that created it.
///
/// This is the shape a reclustering pass leaves behind, and it goes through the
/// same `Operation::Rewrite` a compaction commits: the new fragments carry the
/// permuted sequences, which is how such a pass would have to preserve row
/// lineage. The fragments are written at half the target size so that the
/// compaction that follows has neighbors to merge, and therefore rechunks --
/// and in the spilled arm spills -- every sequence.
///
/// Nothing is read back from the table: `write_table` gives the row with `id`
/// the row id `id`, so the permutation of the ids is also the permutation of
/// the row ids, and the created-at version follows from the id too.
async fn shuffle_rewrite(dataset: Dataset, config: Config) -> Dataset {
    let total_rows = config.fragments * config.rows_per_fragment;
    assert_eq!(
        dataset.manifest.next_row_id, total_rows as u64,
        "write_table handed out {} row ids for {total_rows} rows, so the row ids no longer \
         equal the ids",
        dataset.manifest.next_row_id
    );

    let mut ids: Vec<u64> = (0..total_rows as u64).collect();
    // A fixed seed, so both arms of a scenario rewrite the same permutation.
    ids.shuffle(&mut SmallRng::seed_from_u64(0x2545_F491_4F6C_DD1D));
    let ids = Arc::new(ids);

    let rows_per_file = config.rows_per_fragment / 2;
    let arrow_schema = schema();
    // Batches are built as the writer asks for them, so the permuted table is
    // never held in memory next to the ids. The writer needs a reader that owns
    // what it reads, hence the shared vector.
    let batches = {
        let ids = ids.clone();
        let schema = arrow_schema.clone();
        (0..total_rows).step_by(rows_per_file).map(move |start| {
            let chunk = &ids[start..(start + rows_per_file).min(total_rows)];
            let id_column = Int64Array::from_iter_values(chunk.iter().map(|id| *id as i64));
            let values = Int64Array::from_iter_values(chunk.iter().map(|id| value_of(*id as i64)));
            RecordBatch::try_new(schema.clone(), vec![Arc::new(id_column), Arc::new(values)])
        })
    };

    let dataset = Arc::new(dataset);
    let reader = RecordBatchIterator::new(batches, arrow_schema);
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

    // An uncommitted append carries no lineage: an append is handed its row ids
    // and versions when it commits, and these fragments are committed by a
    // rewrite instead. Give them the lineage the rows already have, sliced in
    // the order the fragments were written. No row has been updated, so a row
    // was last updated at the version that created it.
    let mut offset = 0;
    for fragment in new_fragments.iter_mut() {
        let rows_in_fragment = fragment.physical_rows.unwrap();
        let fragment_ids = &ids[offset..offset + rows_in_fragment];
        let sequence = RowIdSequence::from(fragment_ids);
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));

        let created = fragment_ids
            .iter()
            .map(|id| created_at_version(*id, config))
            .collect::<Vec<_>>();
        let versions: Arc<[u8]> =
            write_dataset_versions(&RowDatasetVersionSequence::from_versions(&created)).into();
        fragment.created_at_version_meta = Some(RowDatasetVersionMeta::Inline(versions.clone()));
        fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Inline(versions));
        offset += rows_in_fragment;
    }
    assert_eq!(
        offset, total_rows,
        "the new fragments hold {offset} rows but the table has {total_rows}"
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

/// Scan every row's lineage and compare it with what the workload handed out:
/// the row with `id` has the row id `id`, was created at
/// [`created_at_version`], and has not been updated since. A spilled column of
/// the right length but with misaligned values gets past the reader's own
/// checks, so this is what separates a fast answer from a correct one.
///
/// Returns the number of rows checked.
async fn check_lineage(dataset: &Dataset, config: Config) -> u64 {
    let mut scanner = dataset.scan();
    scanner
        .project(&[
            "id",
            ROW_ID,
            ROW_CREATED_AT_VERSION,
            ROW_LAST_UPDATED_AT_VERSION,
        ])
        .unwrap();
    let mut stream = scanner.try_into_stream().await.unwrap();

    let mut checked = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>();
        let created = batch
            .column_by_name(ROW_CREATED_AT_VERSION)
            .unwrap()
            .as_primitive::<UInt64Type>();
        let last_updated = batch
            .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
            .unwrap()
            .as_primitive::<UInt64Type>();
        assert_eq!(
            created.null_count() + last_updated.null_count(),
            0,
            "the scan returned rows without a created-at or last-updated-at version"
        );
        for row in 0..batch.num_rows() {
            let id = ids.value(row) as u64;
            let version = created_at_version(id, config);
            let lineage = (
                row_ids.value(row),
                created.value(row),
                last_updated.value(row),
            );
            assert_eq!(
                lineage,
                (id, version, version),
                "the row with id {id} came back with (row id, created at, last updated at) = \
                 {lineage:?}"
            );
        }
        checked += batch.num_rows() as u64;
    }
    checked
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

async fn run_arm(dir: &str, config: Config, scenario: Scenario, spill: bool) -> ArmResult {
    let uri = dir.to_string();
    let mut dataset = build_base(&uri, config, scenario).await;
    if spill {
        let inline_budget = config.inline_max_bytes.map(|bytes| bytes.to_string());
        let mut table_config = vec![(SPILL_ROW_LINEAGE_CONFIG_KEY, "true")];
        if let Some(bytes) = &inline_budget {
            table_config.push((INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, bytes.as_str()));
        }
        dataset.update_config(table_config).await.unwrap();
    }

    // A commit writes the whole transaction to its own file under
    // `_transactions/` as well as putting the fragment list in the manifest, so
    // the compaction's transaction blob is a write cost the manifest row does
    // not show. The data directory is measured the same way, so that only the
    // files the compaction wrote count, not the ones the base table left
    // behind.
    let transactions_before = dir_bytes(&format!("{dir}/_transactions"));
    let data_before = dir_bytes(&format!("{dir}/data"));

    let started = Instant::now();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: config.rows_per_fragment,
            // Only bites in the `deleted` scenario; the `shuffled` one has no
            // deletions and is picked up by the below-target rule instead.
            materialize_deletions: true,
            materialize_deletions_threshold: 0.0,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let compaction = started.elapsed();
    let transaction_bytes =
        dir_bytes(&format!("{dir}/_transactions")).saturating_sub(transactions_before);
    // Both arms wrote identical user data, so the difference between them is
    // what the spilled lineage columns cost.
    let compaction_output_bytes = dir_bytes(&format!("{dir}/data")).saturating_sub(data_before);

    let fragments = dataset.manifest.fragments.as_slice();
    let total_fragments = fragments.len();
    let spilled_fragments = SpilledFragments {
        row_ids: fragments
            .iter()
            .filter(|fragment| matches!(fragment.row_id_meta, Some(RowIdMeta::Column)))
            .count(),
        created_at: fragments
            .iter()
            .filter(|fragment| {
                matches!(
                    fragment.created_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                )
            })
            .count(),
        last_updated_at: fragments
            .iter()
            .filter(|fragment| {
                matches!(
                    fragment.last_updated_at_version_meta,
                    Some(RowDatasetVersionMeta::Column)
                )
            })
            .count(),
    };
    if spill {
        // Spilling is planned per compaction task, so some outputs staying
        // inline is a legitimate result. None spilling is not: the table would
        // compare the inline layout with itself.
        assert!(
            spilled_fragments.any(),
            "the spilled arm spilled nothing: every compacted sequence of the {} scenario fits \
             the {} byte inline budget at BENCH_FRAGMENTS={} BENCH_ROWS_PER_FRAGMENT={}; lower \
             the budget ({INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY}) with BENCH_INLINE_MAX_BYTES",
            scenario.name(),
            config
                .inline_max_bytes
                .unwrap_or(DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES),
            config.fragments,
            config.rows_per_fragment,
        );
    } else {
        assert!(
            !spilled_fragments.any(),
            "the inline arm never set {SPILL_ROW_LINEAGE_CONFIG_KEY}, yet it spilled row ids in \
             {} fragments, created-at versions in {} and last-updated-at versions in {}",
            spilled_fragments.row_ids,
            spilled_fragments.created_at,
            spilled_fragments.last_updated_at,
        );
    }

    let rows = dataset.count_rows(None).await.unwrap() as u64;
    let compaction_manifest_bytes = dataset
        .manifest_location()
        .size
        .expect("a commit records the size of the manifest it wrote");
    let (inline_row_id_bytes, inline_row_version_bytes) = inline_sequence_bytes(&dataset);

    // A cold open of a small manifest is sub-millisecond on a local
    // filesystem, so a single sample is mostly noise.
    let mut open = Vec::with_capacity(OPEN_SAMPLES);
    for _ in 0..OPEN_SAMPLES {
        let started = Instant::now();
        let _opened = open_cold(&uri).await;
        open.push(started.elapsed());
    }

    // One fragment's sequence, read cold. This is the work the design moves out
    // of the manifest decode and into a data file read, so it is measured on its
    // own rather than only through the query that depends on it.
    let mut load_sequence = Vec::with_capacity(READ_SAMPLES);
    for _ in 0..READ_SAMPLES {
        // Opened outside the timed region: this row is the sequence read, not
        // the manifest decode that `open` already reports.
        let cold = open_cold(&uri).await;
        let last = cold
            .get_fragments()
            .pop()
            .expect("compaction left no fragments");
        let started = Instant::now();
        load_row_id_sequence(&cold, last.metadata()).await.unwrap();
        load_sequence.push(started.elapsed());
    }

    // The dataset-wide row id index, which is what any query that resolves a row
    // id pays for before it can touch data.
    let mut index_build = Vec::with_capacity(READ_SAMPLES);
    for _ in 0..READ_SAMPLES {
        let cold = open_cold(&uri).await;
        let started = Instant::now();
        get_row_id_index(&cold).await.unwrap();
        index_build.push(started.elapsed());
    }

    // The take itself, once everything it depends on is cached, which makes
    // this row a control. It needs a session that actually caches --
    // `open_cold` gives the caches zero capacity, so on that dataset
    // `take_rows` would rebuild the index and re-measure `index_build`. The
    // index build bypasses the per-fragment sequence cache, yet opening the
    // target fragment for the take goes through it, so the target's sequence
    // is loaded before the timer too: otherwise the spilled arm's take would
    // re-read the hidden column `load_sequence` already measures. One untimed
    // take then opens the data file and caches its metadata.
    let warm = Dataset::open(&uri).await.unwrap();
    get_row_id_index(&warm).await.unwrap();
    let target = warm
        .get_fragments()
        .pop()
        .expect("compaction left no fragments");
    let sequence = load_row_id_sequence(&warm, target.metadata())
        .await
        .unwrap();
    let probes = (0..TAKE_PROBES)
        .map(|probe| {
            let offset = probe * sequence.len() as usize / TAKE_PROBES;
            sequence
                .get(offset)
                .expect("the offset is within the sequence")
        })
        .collect::<Vec<_>>();
    let projection = ProjectionRequest::from_columns(["value"], warm.schema());
    warm.take_rows(&probes[..1], projection.clone())
        .await
        .unwrap();
    let mut take = Vec::with_capacity(TAKE_PROBES);
    for probe_id in probes {
        let request = projection.clone();
        let started = Instant::now();
        let taken = warm.take_rows(&[probe_id], request).await.unwrap();
        take.push(started.elapsed());

        // A spot check of the index path: row ids were handed out in `id`
        // order, so the row a sequence resolves has to be the one whose `value`
        // matches.
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
    }

    // The full check, after the timed reads so that it cannot warm what they
    // measure, and before the appends, whose rows break `_rowid == id`.
    let checked = check_lineage(&dataset, config).await;
    assert_eq!(
        checked, rows,
        "the lineage scan returned {checked} rows, but the table has {rows}"
    );

    // Only the commit is timed. The dataset is opened once, so the manifest
    // decode that `open` reports is not counted again, and each append's data
    // file is written before the timer starts.
    let schema = schema();
    let append_params = WriteParams {
        mode: WriteMode::Append,
        skip_auto_cleanup: true,
        ..Default::default()
    };
    let mut current = Arc::new(Dataset::open(&uri).await.unwrap());
    let mut commit = Vec::with_capacity(config.appends);
    let mut manifest_bytes = 0;
    for append in 0..config.appends {
        let data = batch(schema.clone(), 1_000_000_000 + append as i64 * 10, 10);
        let transaction = InsertBuilder::new(current.clone())
            .with_params(&append_params)
            .execute_uncommitted(vec![data])
            .await
            .unwrap();
        let started = Instant::now();
        let committed = CommitBuilder::new(current.clone())
            .with_skip_auto_cleanup(true)
            .execute(transaction)
            .await
            .unwrap();
        commit.push(started.elapsed());
        if append == 0 {
            // The compacted fragment list plus one small fragment and a small
            // inline transaction: the manifest every reader decodes from now on
            // and every later commit rewrites.
            manifest_bytes = committed
                .manifest_location()
                .size
                .expect("a commit records the size of the manifest it wrote");
        }
        current = Arc::new(committed);
    }

    ArmResult {
        compaction,
        compaction_output_bytes,
        manifest_bytes,
        compaction_manifest_bytes,
        transaction_bytes,
        inline_row_id_bytes,
        inline_row_version_bytes,
        rows,
        open,
        commit,
        load_sequence,
        index_build,
        take,
        spilled_fragments,
        total_fragments,
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The middle sample, or the mean of the two middle ones.
fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2
    } else {
        sorted[middle]
    }
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
    println!(
        "spilled arm: {SPILL_ROW_LINEAGE_CONFIG_KEY}=true with a {} byte inline budget",
        config
            .inline_max_bytes
            .unwrap_or(DEFAULT_INLINE_ROW_LINEAGE_MAX_BYTES)
    );
    println!();

    println!(
        "{:<40} {:>14} {:>14} {:>10}",
        "metric", "inline", "spilled", "ratio"
    );
    let row = |name: &str, inline: f64, spilled: f64, unit: &str| {
        let ratio = if spilled == 0.0 {
            f64::INFINITY
        } else {
            inline / spilled
        };
        println!("{name:<40} {inline:>12.2}{unit:<2} {spilled:>12.2}{unit:<2} {ratio:>9.2}x");
    };
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
    // The median is the figure to quote; the minimum under it shows how much
    // of the median is noise.
    let timing = |name: &str, inline: &[Duration], spilled: &[Duration]| {
        let fastest = |samples: &[Duration]| samples.iter().min().copied().unwrap_or_default();
        row(
            &format!("{name} (median)"),
            ms(median(inline)),
            ms(median(spilled)),
            "ms",
        );
        row("  min", ms(fastest(inline)), ms(fastest(spilled)), "ms");
    };

    row(
        "manifest size (steady state)",
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
    // When the compaction's transaction serializes to at most
    // `MAX_INLINE_TRANSACTION_BYTES` (20 MiB), a copy of it is written into the
    // manifest file ahead of the manifest itself (`Manifest::transaction_section`),
    // on top of the file under `_transactions/`. Readers decode only the
    // manifest, so the copy is a one-off write cost, which is why this file is
    // reported apart from the steady-state size above.
    row(
        "compaction manifest (incl. inline txn)",
        mib(inline.compaction_manifest_bytes),
        mib(spilled.compaction_manifest_bytes),
        "M",
    );
    row(
        "compaction transaction file",
        mib(inline.transaction_bytes),
        mib(spilled.transaction_bytes),
        "M",
    );
    timing("cold dataset open", &inline.open, &spilled.open);
    timing("append commit", &inline.commit, &spilled.commit);
    timing(
        "load one sequence (cold)",
        &inline.load_sequence,
        &spilled.load_sequence,
    );
    timing(
        "row id index build (cold)",
        &inline.index_build,
        &spilled.index_build,
    );
    timing("take by row id (control)", &inline.take, &spilled.take);
    row(
        "compaction",
        ms(inline.compaction),
        ms(spilled.compaction),
        "ms",
    );
    row(
        "compaction output (data files)",
        mib(inline.compaction_output_bytes),
        mib(spilled.compaction_output_bytes),
        "M",
    );

    println!();
    // Bytes per row is what makes the two scenarios comparable: it is the
    // encoding's cost for one row, and it is what decides whether a sequence
    // belongs in the manifest at all.
    let spilled_column_bytes = spilled
        .compaction_output_bytes
        .saturating_sub(inline.compaction_output_bytes);
    println!(
        "{} rows: the inline arm keeps {:.2} B/row of row ids and {:.2} B/row of row versions \
         in the manifest; the spilled lineage columns add {:.2} B/row to the compaction output",
        inline.rows,
        inline.inline_row_id_bytes as f64 / inline.rows as f64,
        inline.inline_row_version_bytes as f64 / inline.rows as f64,
        spilled_column_bytes as f64 / spilled.rows as f64,
    );
    println!(
        "spilled arm, fragments with a spilled sequence: row ids {}/{}, created at {}/{}, last \
         updated at {}/{}",
        spilled.spilled_fragments.row_ids,
        spilled.total_fragments,
        spilled.spilled_fragments.created_at,
        spilled.total_fragments,
        spilled.spilled_fragments.last_updated_at,
        spilled.total_fragments,
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
    if std::env::var_os(ENABLE_UNSTABLE_SPILLED_ROW_LINEAGE_ENV).is_none()
        && !cfg!(debug_assertions)
    {
        panic!(
            "set {ENABLE_UNSTABLE_SPILLED_ROW_LINEAGE_ENV}=1 to run this benchmark: spilled row lineage \
             are an unstable feature and a release build refuses the dataset without it"
        );
    }

    let config = Config::from_env();
    let scenarios = scenarios_from_env();
    config.validate(&scenarios);
    let runtime = Runtime::new().unwrap();

    println!("=== Row lineage placement ===");

    for scenario in scenarios {
        // A table that has not opted in reproduces the behavior on main, where
        // a compacted fragment's sequences always stay inline however large
        // they grow. Each arm's table is deleted before the next arm builds its
        // own, so the two never take up disk at the same time.
        let run = |spill: bool| {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().to_string_lossy().into_owned();
            runtime.block_on(run_arm(&uri, config, scenario, spill))
        };
        let inline = run(false);
        let spilled = run(true);

        report(scenario, config, &inline, &spilled);
    }
}

criterion_group!(benches, bench_rowid_spill);
criterion_main!(benches);
