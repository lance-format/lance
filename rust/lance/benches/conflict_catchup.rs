// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a long-running job (compaction, index build) pays to commit after
//! `V` versions landed while it ran, on a table of `N` fragments.
//!
//! The job reads version `R`, the history moves on by `V` seeded concurrent
//! commits (appends, row and whole-fragment deletes, compactions, index
//! optimizes; all named operations, half action sets, or all action sets),
//! and the job commits through [`CommitBuilder`] from a process of its own:
//! a fresh session with the default caches. Its first attempt loses a race
//! to one more commit, so it also pays for a retry. The store is the local
//! file system, so wall time leaves out object store latency; the reads it
//! would pay for are counted instead.
//!
//! Reported per case, beside criterion's wall time for the whole commit:
//!
//! - `peak_heap`: the most heap the commit held above where it started.
//! - reads and bytes, by kind: manifests (`_versions/`), transaction files,
//!   deletion files, index files.
//! - time inside the catch-up (`catch_up`; on main and before it was
//!   streamed, `load_and_sort_new_transactions` then the footprint
//!   bookkeeping, `record_committed_history`), the conflict check
//!   (`check_txns`) and the finish (`finish_with_tagged_rewrite`), summed over
//!   attempts, wall (busy) per span.
//! - `retry`: from the lost race to the commit landing.
//!
//! The history is synthetic: every fragment points at the one data file the
//! table was created with, and nothing reads data. Fixtures are built once
//! per (V, N, mix) under `CATCHUP_DIR` and reused across runs; each
//! iteration removes what the job wrote.
//!
//! Small sizes are enough. The catch-up costs about 0.6-0.9 us per
//! version x fragment, so its time and reads grow linearly in V x N. Peak heap
//! grows with N alone, unless something holds a manifest per version. What a
//! regression adds is a ratio or a per-version slope, and V=100 already shows
//! it. Measured on main, V=1000 x N=300k took 272 s against 6.9 s at V=1000 x
//! N=10k: the same cost per version x fragment, to within a third.
//!
//! ```bash
//! # V=100, N=10k, every mix and job: under a second a commit
//! cargo bench -p lance --profile release-with-debug --bench conflict_catchup
//! # the slope in V and in N, a few seconds a commit
//! CATCHUP_VERSIONS=1000 CATCHUP_ITERATIONS=3 \
//!   cargo bench -p lance --profile release-with-debug --bench conflict_catchup
//! CATCHUP_FRAGMENTS=100000 CATCHUP_ITERATIONS=3 \
//!   cargo bench -p lance --profile release-with-debug --bench conflict_catchup
//! ```
//!
//! Configuration:
//!
//! - `CATCHUP_VERSIONS`, `CATCHUP_FRAGMENTS`: comma-separated sweeps
//!   (defaults 100 and 10000). A (V, N) pair whose fixture would exceed
//!   `CATCHUP_MAX_FRAGMENT_VERSIONS` (V x N, default 3e8, about 25 GB of
//!   manifests) is skipped.
//! - `CATCHUP_MIXES`: any of `legacy,half,action` (default all).
//! - `CATCHUP_JOBS`: job names as printed (default all).
//! - `CATCHUP_CACHE`: when set, the job's session has the default metadata
//!   cache instead of none.
//! - `CATCHUP_ITERATIONS`: run each case this many times outside criterion,
//!   for cases too slow for its ten samples. Index coverage jobs at large
//!   V x N hold a manifest per version today: mind the memory.
//! - `CATCHUP_DIR`: where fixtures live (default `$TMPDIR/lance-conflict-catchup`).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, RewriteGroup, Transaction};
use lance::dataset::{CommitBuilder, Dataset, InsertBuilder};
use lance::index::DatasetIndexExt;
use lance::session::Session;
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_io::utils::tracking_store::{IOTracker, IoStats};
use lance_select::RowAddrTreeMap;
use lance_table::format::{DataFile, Fragment, IndexMetadata};
use lance_table::io::deletion::write_deletion_file;
use lance_table::transaction::action::{
    Action, AddDataFile, AddFragment, AddFragmentParts, AddIndexSegment, AddIndexSegmentParts,
    AdjustIndexCoverage, AdjustIndexCoverageParts, CompositeOperation, FieldRef, Ref,
    RemoveFragment, SetDeletionFile, UserStep,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;
use tokio::runtime::Runtime;
use tracing::span;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracking_allocator::{AllocationGroupId, AllocationRegistry, AllocationTracker, Allocator};
use uuid::Uuid;

const ROWS_PER_FRAGMENT: u32 = 16;
/// The fragments a delete job deletes the last row of; concurrent deletes
/// take other rows of them, so the job merges deletion files.
const DELETE_FRAGMENTS: Range<u64> = 0..4;
/// The fragments a compaction job removes; nothing concurrent touches them.
const COMPACT_FRAGMENTS: Range<u64> = 4..8;
const PHASES: [&str; 5] = [
    "catch_up",
    "load_and_sort_new_transactions",
    "record_committed_history",
    "check_txns",
    "finish_with_tagged_rewrite",
];

// ---------------------------------------------------------------------------
// Peak heap

#[global_allocator]
static ALLOCATOR: Allocator<std::alloc::System> = Allocator::system();

static HEAP: AtomicI64 = AtomicI64::new(0);
static PEAK_HEAP: AtomicI64 = AtomicI64::new(0);

struct HeapTracker;

impl AllocationTracker for HeapTracker {
    fn allocated(&self, _: usize, object_size: usize, _: usize, _: AllocationGroupId) {
        let now = HEAP.fetch_add(object_size as i64, Ordering::Relaxed) + object_size as i64;
        PEAK_HEAP.fetch_max(now, Ordering::Relaxed);
    }

    fn deallocated(
        &self,
        _: usize,
        object_size: usize,
        _: usize,
        _: AllocationGroupId,
        _: AllocationGroupId,
    ) {
        HEAP.fetch_sub(object_size as i64, Ordering::Relaxed);
    }
}

/// Restart the peak at the current heap, returning that baseline.
fn reset_peak_heap() -> i64 {
    let now = HEAP.load(Ordering::Relaxed);
    PEAK_HEAP.store(now, Ordering::Relaxed);
    now
}

// ---------------------------------------------------------------------------
// Time per phase

#[derive(Debug, Default, Clone, Copy)]
struct PhaseTime {
    wall: Duration,
    busy: Duration,
}

static PHASE_TIMES: LazyLock<Mutex<HashMap<&'static str, PhaseTime>>> =
    LazyLock::new(Default::default);

struct SpanTiming {
    opened: Instant,
    entered: Option<Instant>,
    busy: Duration,
}

/// Sums each phase span's wall time (open to close) and busy time (polled).
struct PhaseLayer;

impl<S> Layer<S> for PhaseLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanTiming {
                opened: Instant::now(),
                entered: None,
                busy: Duration::ZERO,
            });
        }
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(timing) = span.extensions_mut().get_mut::<SpanTiming>()
        {
            timing.entered = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id)
            && let Some(timing) = span.extensions_mut().get_mut::<SpanTiming>()
            && let Some(entered) = timing.entered.take()
        {
            timing.busy += entered.elapsed();
        }
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id)
            && let Some(timing) = span.extensions_mut().remove::<SpanTiming>()
        {
            let mut times = PHASE_TIMES.lock().unwrap();
            let phase = times.entry(span.name()).or_default();
            phase.wall += timing.opened.elapsed();
            phase.busy += timing.busy;
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mix {
    Legacy,
    Half,
    Action,
}

impl Mix {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Half => "half",
            Self::Action => "action",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Job {
    LegacyAppend,
    LegacyDelete,
    LegacyCreateIndex,
    ActionAppend,
    ActionDelete,
    ActionCompaction,
    ActionAddIndexSegment,
    ActionAdjustIndexCoverage,
}

const JOBS: [Job; 8] = [
    Job::LegacyAppend,
    Job::LegacyDelete,
    Job::LegacyCreateIndex,
    Job::ActionAppend,
    Job::ActionDelete,
    Job::ActionCompaction,
    Job::ActionAddIndexSegment,
    Job::ActionAdjustIndexCoverage,
];

impl Job {
    fn name(self) -> &'static str {
        match self {
            Self::LegacyAppend => "legacy_append",
            Self::LegacyDelete => "legacy_delete",
            Self::LegacyCreateIndex => "legacy_create_index",
            Self::ActionAppend => "action_append",
            Self::ActionDelete => "action_delete",
            Self::ActionCompaction => "action_compaction",
            Self::ActionAddIndexSegment => "action_add_index_segment",
            Self::ActionAdjustIndexCoverage => "action_adjust_index_coverage",
        }
    }

    fn deletes(self) -> bool {
        matches!(self, Self::LegacyDelete | Self::ActionDelete)
    }
}

struct Config {
    versions: Vec<u64>,
    fragments: Vec<u64>,
    mixes: Vec<Mix>,
    jobs: Vec<Job>,
    max_fragment_versions: u64,
    /// Run each case this many times outside criterion, for cases too slow
    /// for its ten samples.
    iterations: Option<u64>,
    dir: PathBuf,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

impl Config {
    fn from_env() -> Self {
        let numbers = |name, default| {
            env_list(name, default)
                .iter()
                .map(|n| n.parse::<f64>().unwrap() as u64)
                .collect::<Vec<_>>()
        };
        let mixes = env_list("CATCHUP_MIXES", "legacy,half,action")
            .iter()
            .map(|name| {
                [Mix::Legacy, Mix::Half, Mix::Action]
                    .into_iter()
                    .find(|mix| mix.name() == name)
                    .unwrap_or_else(|| panic!("unknown mix {name}"))
            })
            .collect();
        let job_names = env_list("CATCHUP_JOBS", "");
        let jobs = JOBS
            .into_iter()
            .filter(|job| job_names.is_empty() || job_names.iter().any(|name| name == job.name()))
            .collect();
        Self {
            versions: numbers("CATCHUP_VERSIONS", "100"),
            fragments: numbers("CATCHUP_FRAGMENTS", "10000"),
            mixes,
            jobs,
            max_fragment_versions: numbers("CATCHUP_MAX_FRAGMENT_VERSIONS", "3e8")[0],
            iterations: std::env::var("CATCHUP_ITERATIONS")
                .ok()
                .map(|n| n.parse().unwrap()),
            dir: std::env::var("CATCHUP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::env::temp_dir().join("lance-conflict-catchup")),
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture

fn action_set(read_version: u64, actions: Vec<Action>) -> Transaction {
    Transaction::new_from_version(
        read_version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "catchup", actions,
        )])),
    )
}

fn data_file(template: &Fragment) -> DataFile {
    let mut file = template.files[0].clone();
    file.fields = Default::default();
    file
}

fn added_fragment(template: &Fragment, rows: u64, data_change: bool) -> [Action; 2] {
    [
        Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: Ref::Local(0),
            physical_rows: rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change,
        })),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                data_file(template),
                vec![FieldRef::Committed(0), FieldRef::Committed(1)],
                None,
                data_change,
            )
            .unwrap(),
        ),
    ]
}

fn segment(
    name: &str,
    field: i32,
    fragments: impl IntoIterator<Item = u64>,
    version: u64,
) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        fields: vec![field],
        covering_fields: vec![],
        dataset_version: version,
        fragment_bitmap: Some(fragments.into_iter().map(|id| id as u32).collect()),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: Some(vec![]),
        clustering_id: None,
        metadata: Default::default(),
    }
}

fn add_index_segment(segment: IndexMetadata, fragments: impl IntoIterator<Item = u64>) -> Action {
    Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
            uuid: segment.uuid,
            name: segment.name,
            fields: segment
                .fields
                .into_iter()
                .map(FieldRef::Committed)
                .collect(),
            covering_fields: vec![],
            index_details: None,
            index_version: 0,
            covered_fragments: Some(fragments.into_iter().map(Ref::Committed).collect()),
            files: vec![],
            base: None,
            created_at: None,
            dataset_version: Some(segment.dataset_version),
            data_change: false,
        })
        .unwrap(),
    )
}

/// What a built fixture records about itself.
#[derive(Debug, Clone)]
struct Fixture {
    uri: String,
    read_version: u64,
    latest_version: u64,
    fragments: u64,
    template: Fragment,
    /// The segment of `idx_a` at the read version.
    segment: IndexMetadata,
    /// Every file of the fixture, relative to its directory. Anything else
    /// is the job's, removed after each iteration.
    files: HashSet<PathBuf>,
    /// The race winner's files, relative to the fixture directory, kept in
    /// `race/` until the job's first attempt writes its manifest.
    winner: Vec<PathBuf>,
}

struct Builder {
    writer: Dataset,
    template: Fragment,
    fragments: u64,
    deleted: HashMap<u64, RoaringBitmap>,
    unindexed: Vec<u64>,
    rng: SmallRng,
}

impl Builder {
    async fn commit(&mut self, transaction: Transaction) {
        self.writer = CommitBuilder::new(Arc::new(self.writer.clone()))
            .with_experimental_composite_operations(true)
            .with_skip_auto_cleanup(true)
            .execute(transaction)
            .await
            .unwrap();
    }

    fn latest_fragment(&self) -> u64 {
        self.writer.manifest.max_fragment_id().unwrap()
    }

    fn pick(&mut self, candidates: &[u64]) -> Option<u64> {
        (!candidates.is_empty()).then(|| candidates[self.rng.random_range(0..candidates.len())])
    }

    async fn append(&mut self, as_action: bool) {
        let version = self.writer.manifest.version;
        let transaction = if as_action {
            action_set(
                version,
                added_fragment(&self.template, ROWS_PER_FRAGMENT as u64, true).into(),
            )
        } else {
            Transaction::new_from_version(
                version,
                Operation::Append {
                    fragments: vec![self.template.clone()],
                },
            )
        };
        self.commit(transaction).await;
        let appended = self.latest_fragment();
        self.unindexed.push(appended);
    }

    async fn delete_row(&mut self, fragment: Option<u64>, as_action: bool) -> bool {
        let candidates: Vec<u64> = self
            .writer
            .fragments()
            .iter()
            .map(|f| f.id)
            .filter(|id| !COMPACT_FRAGMENTS.contains(id))
            .collect();
        let Some(fragment) = fragment.or_else(|| self.pick(&candidates)) else {
            return false;
        };
        let deleted = self.deleted.entry(fragment).or_default();
        let Some(row) = (0..ROWS_PER_FRAGMENT - 1).find(|row| !deleted.contains(*row)) else {
            return false;
        };
        deleted.insert(row);
        let deletions = DeletionVector::from(deleted.clone());
        let version = self.writer.manifest.version;
        let file = write_deletion_file(
            &self.writer.branch_location().path,
            fragment,
            version,
            &deletions,
            self.writer.object_store(None).await.unwrap().as_ref(),
            0,
        )
        .await
        .unwrap()
        .unwrap();
        let transaction = if as_action {
            action_set(
                version,
                vec![Action::SetDeletionFile(
                    SetDeletionFile::try_new_unchecked(fragment, file, None, true).unwrap(),
                )],
            )
        } else {
            let mut updated = fragment_by_id(&self.writer, fragment).unwrap().clone();
            updated.deletion_file = Some(file);
            Transaction::new_from_version(
                version,
                Operation::Delete {
                    updated_fragments: vec![updated],
                    deleted_fragment_ids: vec![],
                    predicate: "catchup".into(),
                },
            )
        };
        self.commit(transaction).await;
        true
    }

    async fn delete_fragment(&mut self, as_action: bool) -> bool {
        let candidates: Vec<u64> = self
            .writer
            .fragments()
            .iter()
            .map(|f| f.id)
            .filter(|id| *id >= COMPACT_FRAGMENTS.end)
            .collect();
        let Some(fragment) = self.pick(&candidates) else {
            return false;
        };
        let version = self.writer.manifest.version;
        let transaction = if as_action {
            action_set(
                version,
                vec![Action::RemoveFragment(RemoveFragment::new_unchecked(
                    Ref::Committed(fragment),
                    true,
                ))],
            )
        } else {
            Transaction::new_from_version(
                version,
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![fragment],
                    predicate: "catchup".into(),
                },
            )
        };
        self.commit(transaction).await;
        self.unindexed.retain(|id| *id != fragment);
        true
    }

    async fn compact(&mut self, as_action: bool) -> bool {
        let sources: Vec<Fragment> = self
            .unindexed
            .iter()
            .filter(|id| !self.deleted.contains_key(id))
            .filter_map(|id| fragment_by_id(&self.writer, *id).cloned())
            .take(2)
            .collect();
        if sources.len() < 2 {
            return false;
        }
        let version = self.writer.manifest.version;
        let rows = 2 * ROWS_PER_FRAGMENT as u64;
        let transaction = if as_action {
            let mut actions: Vec<Action> = sources
                .iter()
                .map(|f| {
                    Action::RemoveFragment(RemoveFragment::new_unchecked(
                        Ref::Committed(f.id),
                        false,
                    ))
                })
                .collect();
            actions.extend(added_fragment(&self.template, rows, false));
            action_set(version, actions)
        } else {
            let mut compacted = self.template.clone();
            compacted.physical_rows = Some(rows as usize);
            Transaction::new_from_version(
                version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: sources.clone(),
                        new_fragments: vec![compacted],
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                },
            )
        };
        self.commit(transaction).await;
        self.unindexed
            .retain(|id| !sources.iter().any(|f| f.id == *id));
        let compacted = self.latest_fragment();
        self.unindexed.push(compacted);
        true
    }

    async fn optimize(&mut self, as_action: bool) -> bool {
        let fragments: Vec<u64> = std::mem::take(&mut self.unindexed)
            .into_iter()
            .filter(|id| fragment_by_id(&self.writer, *id).is_some())
            .collect();
        if fragments.is_empty() {
            return false;
        }
        let version = self.writer.manifest.version;
        let segment = segment("opt", 1, fragments.iter().copied(), version);
        let transaction = if as_action {
            action_set(version, vec![add_index_segment(segment, fragments)])
        } else {
            Transaction::new_from_version(
                version,
                Operation::CreateIndex {
                    new_indices: vec![segment],
                    removed_indices: vec![],
                },
            )
        };
        self.commit(transaction).await;
        true
    }

    async fn advance(&mut self, count: u64, mix: Mix) {
        for step in 0..count {
            let as_action = match mix {
                Mix::Legacy => false,
                Mix::Half => self.rng.random_bool(0.5),
                Mix::Action => true,
            };
            // Every so often a delete job's fragments lose a row too, so it
            // has deletion files to merge.
            if step % 8 == 0 {
                let fragment = DELETE_FRAGMENTS.start
                    + (step / 8) % (DELETE_FRAGMENTS.end - DELETE_FRAGMENTS.start);
                self.delete_row(Some(fragment), as_action).await;
                continue;
            }
            let committed = match self.rng.random_range(0..100) {
                0..35 => {
                    self.append(as_action).await;
                    true
                }
                35..70 => self.delete_row(None, as_action).await,
                70..75 => self.delete_fragment(as_action).await,
                75..85 => self.compact(as_action).await,
                _ => self.optimize(as_action).await,
            };
            if !committed {
                self.append(as_action).await;
            }
        }
    }
}

/// The fragment `id` of `dataset`, whose fragments are sorted by id.
fn fragment_by_id(dataset: &Dataset, id: u64) -> Option<&Fragment> {
    let fragments = dataset.manifest.fragments.as_ref();
    fragments
        .binary_search_by_key(&id, |fragment| fragment.id)
        .ok()
        .map(|index| &fragments[index])
}

fn copy_into(from: &FsPath, to: &FsPath, file: &FsPath) {
    std::fs::create_dir_all(to.join(file).parent().unwrap()).unwrap();
    std::fs::copy(from.join(file), to.join(file)).unwrap();
}

fn list_files(root: &FsPath) -> HashSet<PathBuf> {
    fn walk(root: &FsPath, dir: &FsPath, files: &mut HashSet<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, files);
            } else {
                files.insert(path.strip_prefix(root).unwrap().to_path_buf());
            }
        }
    }
    let mut files = HashSet::new();
    walk(root, root, &mut files);
    files
}

async fn build_fixture(dir: &FsPath, versions: u64, fragments: u64, mix: Mix) -> Fixture {
    let table = dir.join("table");
    let uri = table.to_str().unwrap().to_string();
    let data = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..ROWS_PER_FRAGMENT as i32)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(
                0,
                ROWS_PER_FRAGMENT as usize,
            ))),
        ],
    )
    .unwrap();
    let writer = InsertBuilder::new(uri.as_str())
        .execute(vec![data])
        .await
        .unwrap();
    let mut template = writer.fragments()[0].clone();
    template.id = 0;
    let mut builder = Builder {
        writer,
        template: template.clone(),
        fragments,
        deleted: HashMap::new(),
        unindexed: Vec::new(),
        rng: SmallRng::seed_from_u64(42),
    };
    let version = builder.writer.manifest.version;
    builder
        .commit(Transaction::new_from_version(
            version,
            Operation::Append {
                fragments: vec![template; fragments as usize - 1],
            },
        ))
        .await;
    let version = builder.writer.manifest.version;
    let idx_a = segment("idx_a", 0, 0..fragments / 2, version);
    builder
        .commit(Transaction::new_from_version(
            version,
            Operation::CreateIndex {
                new_indices: vec![idx_a.clone()],
                removed_indices: vec![],
            },
        ))
        .await;
    let read_version = builder.writer.manifest.version;
    let started = Instant::now();
    builder.advance(versions, mix).await;
    eprintln!(
        "built {versions} versions over {fragments} fragments ({}) in {:.1?}",
        mix.name(),
        started.elapsed()
    );
    let latest_version = builder.writer.manifest.version;

    // What a commit overwrites rather than adds (the version hint), kept
    // to restore after every iteration.
    let before = list_files(&table);
    for file in &before {
        if file.starts_with("_versions") && file.extension().is_none_or(|ext| ext != "manifest") {
            copy_into(&table, &dir.join("pristine"), file);
        }
    }
    // The race winner: one more append, moved aside until the job races it.
    builder.append(false).await;
    let mut winner = Vec::new();
    for file in list_files(&table) {
        let pristine = dir.join("pristine").join(&file);
        let is_new = !before.contains(&file);
        if is_new
            || (pristine.exists()
                && std::fs::read(&pristine).unwrap() != std::fs::read(table.join(&file)).unwrap())
        {
            copy_into(&table, &dir.join("race"), &file);
            if is_new {
                std::fs::remove_file(table.join(&file)).unwrap();
            } else {
                copy_into(&dir.join("pristine"), &table, &file);
            }
            winner.push(file);
        }
    }
    let fixture = Fixture {
        uri,
        read_version,
        latest_version,
        fragments: builder.fragments,
        template: builder.template.clone(),
        segment: idx_a,
        files: list_files(&table),
        winner,
    };
    std::fs::write(
        dir.join("READY"),
        format!("{read_version} {latest_version} {}", fixture.segment.uuid),
    )
    .unwrap();
    fixture
}

async fn load_fixture(dir: &FsPath, versions: u64, fragments: u64, mix: Mix) -> Fixture {
    let ready = dir.join("READY");
    let Ok(contents) = std::fs::read_to_string(&ready) else {
        if dir.exists() {
            std::fs::remove_dir_all(dir).unwrap();
        }
        std::fs::create_dir_all(dir).unwrap();
        return build_fixture(dir, versions, fragments, mix).await;
    };
    let fields: Vec<&str> = contents.split_whitespace().collect();
    let (read_version, latest_version) = (fields[0].parse().unwrap(), fields[1].parse().unwrap());
    let table = dir.join("table");
    let uri = table.to_str().unwrap().to_string();
    let read = DatasetBuilder::from_uri(&uri)
        .with_version(read_version)
        .load()
        .await
        .unwrap();
    let mut template = read.fragments()[0].clone();
    template.id = 0;
    template.deletion_file = None;
    let indices = read.load_indices().await.unwrap();
    let segment = indices
        .iter()
        .find(|index| index.uuid.to_string() == fields[2])
        .unwrap()
        .clone();
    Fixture {
        uri,
        read_version,
        latest_version,
        fragments,
        template,
        segment,
        files: list_files(&table),
        winner: list_files(&dir.join("race")).into_iter().collect(),
    }
}

// ---------------------------------------------------------------------------
// The job

/// The job, staged at the read version, and the rows it deletes.
async fn stage(
    job: Job,
    fixture: &Fixture,
    dataset: &Dataset,
) -> (Transaction, Option<RowAddrTreeMap>) {
    let read_version = fixture.read_version;
    let mut deletion_files = Vec::new();
    if job.deletes() {
        for fragment in DELETE_FRAGMENTS {
            let deletions = DeletionVector::from(RoaringBitmap::from_iter([ROWS_PER_FRAGMENT - 1]));
            let file = write_deletion_file(
                &dataset.branch_location().path,
                fragment,
                read_version,
                &deletions,
                dataset.object_store(None).await.unwrap().as_ref(),
                0,
            )
            .await
            .unwrap()
            .unwrap();
            deletion_files.push((fragment, file));
        }
    }
    let deleted_rows = || {
        RowAddrTreeMap::from_iter(
            DELETE_FRAGMENTS.map(|fragment| (fragment << 32) | (ROWS_PER_FRAGMENT - 1) as u64),
        )
    };
    let new_segment = || segment("job_idx", 1, 0..fixture.fragments, read_version);
    let template = &fixture.template;
    match job {
        Job::LegacyAppend => (
            Transaction::new_from_version(
                read_version,
                Operation::Append {
                    fragments: vec![template.clone()],
                },
            ),
            None,
        ),
        Job::ActionAppend => (
            action_set(
                read_version,
                added_fragment(template, ROWS_PER_FRAGMENT as u64, true).into(),
            ),
            None,
        ),
        Job::LegacyDelete => {
            let updated_fragments = deletion_files
                .into_iter()
                .map(|(id, file)| {
                    let mut fragment = template.clone();
                    fragment.id = id;
                    fragment.deletion_file = Some(file);
                    fragment
                })
                .collect();
            (
                Transaction::new_from_version(
                    read_version,
                    Operation::Delete {
                        updated_fragments,
                        deleted_fragment_ids: vec![],
                        predicate: "job".into(),
                    },
                ),
                Some(deleted_rows()),
            )
        }
        Job::ActionDelete => (
            action_set(
                read_version,
                deletion_files
                    .into_iter()
                    .map(|(id, file)| {
                        Action::SetDeletionFile(
                            SetDeletionFile::try_new_unchecked(id, file, None, true).unwrap(),
                        )
                    })
                    .collect(),
            ),
            Some(deleted_rows()),
        ),
        Job::ActionCompaction => {
            let mut actions: Vec<Action> = COMPACT_FRAGMENTS
                .map(|id| {
                    Action::RemoveFragment(RemoveFragment::new_unchecked(Ref::Committed(id), false))
                })
                .collect();
            let rows = (COMPACT_FRAGMENTS.end - COMPACT_FRAGMENTS.start) * ROWS_PER_FRAGMENT as u64;
            actions.extend(added_fragment(template, rows, false));
            (action_set(read_version, actions), None)
        }
        Job::LegacyCreateIndex => (
            Transaction::new_from_version(
                read_version,
                Operation::CreateIndex {
                    new_indices: vec![new_segment()],
                    removed_indices: vec![],
                },
            ),
            None,
        ),
        Job::ActionAddIndexSegment => (
            action_set(
                read_version,
                vec![add_index_segment(new_segment(), 0..fixture.fragments)],
            ),
            None,
        ),
        Job::ActionAdjustIndexCoverage => (
            action_set(
                read_version,
                vec![Action::AdjustIndexCoverage(
                    AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
                        uuid: fixture.segment.uuid,
                        name: fixture.segment.name.clone(),
                        add_fragments: (fixture.fragments / 2..fixture.fragments)
                            .map(Ref::Committed)
                            .collect(),
                        remove_fragments: vec![],
                        fields: vec![FieldRef::Committed(0)],
                        covering_fields: vec![],
                    })
                    .unwrap(),
                )],
            ),
            None,
        ),
    }
}

/// The job's store: counted, and losing the race on its first manifest
/// write.
#[derive(Debug)]
struct JobStore {
    tracker: IOTracker,
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl WrappingObjectStore for JobStore {
    fn wrap(
        &self,
        store_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        self.tracker.wrap(
            store_prefix,
            Arc::new(ProxyObjectStore::new(original, self.policy.clone())),
        )
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Reads {
    count: u64,
    bytes: u64,
}

#[derive(Debug, Default, Clone)]
struct Measurement {
    total: Duration,
    retry: Duration,
    peak_heap: i64,
    reads: [Reads; 5],
    phases: HashMap<&'static str, PhaseTime>,
}

const READ_KINDS: [&str; 5] = ["manifest", "transaction", "deletion", "index", "other"];

fn reads_by_kind(stats: &IoStats) -> [Reads; 5] {
    let mut reads = [Reads::default(); 5];
    for request in &stats.requests {
        let path = request.path.as_ref();
        if request.method == "list" && path.contains("/_indices/") {
            reads[3].count += 1;
        }
        if !matches!(request.method, "get_opts" | "get_ranges") {
            continue;
        }
        let kind = if path.contains("/_versions/") {
            0
        } else if path.contains("/_transactions/") {
            1
        } else if path.contains("/_deletions/") {
            2
        } else if path.contains("/_indices/") {
            3
        } else {
            4
        };
        reads[kind].count += 1;
        let bytes = request
            .range
            .as_ref()
            .map_or(0, |range| range.end - range.start);
        reads[kind].bytes += bytes;
    }
    reads
}

struct Case {
    job: Job,
    fixture: Fixture,
    dir: PathBuf,
}

impl Case {
    /// Remove what the last iteration's job wrote.
    fn reset(&self) {
        let table = self.dir.join("table");
        for file in list_files(&table).difference(&self.fixture.files) {
            std::fs::remove_file(table.join(file)).unwrap();
        }
        let pristine = self.dir.join("pristine");
        if pristine.exists() {
            for file in list_files(&pristine) {
                copy_into(&pristine, &table, &file);
            }
        }
    }

    async fn run(&self) -> Measurement {
        self.reset();
        let tracker = IOTracker::default();
        let race_lost: Arc<Mutex<Option<Instant>>> = Arc::default();
        let mut policy = ProxyObjectStorePolicy::new();
        {
            let winner = Mutex::new(Some(self.fixture.winner.clone()));
            let race_lost = race_lost.clone();
            let (dir, table) = (self.dir.join("race"), self.dir.join("table"));
            policy.set_before_policy(
                "lose_race",
                Arc::new(move |method, path| {
                    let is_manifest_write = matches!(
                        method,
                        "put" | "copy_if_not_exists" | "rename_if_not_exists"
                    ) && path.as_ref().contains("/_versions/");
                    if is_manifest_write && let Some(files) = winner.lock().unwrap().take() {
                        for file in files {
                            std::fs::copy(dir.join(&file), table.join(&file)).unwrap();
                        }
                        *race_lost.lock().unwrap() = Some(Instant::now());
                    }
                    Ok(())
                }),
            );
        }
        let store_params = ObjectStoreParams {
            object_store_wrapper: Some(Arc::new(JobStore {
                tracker: tracker.clone(),
                policy: Arc::new(Mutex::new(policy)),
            })),
            ..Default::default()
        };
        // A process of its own, with the default caches.
        // Through the object store API, not the local file shortcut, so
        // every read is counted.
        let uri = format!("file-object-store://{}", self.fixture.uri);
        // By default no metadata cache, as on a table whose manifests
        // outgrow it: at 300k fragments a decoded manifest is ~250 MB.
        let session = match std::env::var("CATCHUP_CACHE") {
            Ok(_) => Session::default(),
            Err(_) => Session::new(0, 0, Default::default()),
        };
        let dataset = DatasetBuilder::from_uri(&uri)
            .with_session(Arc::new(session))
            .with_version(self.fixture.read_version)
            .with_store_params(store_params)
            .load()
            .await
            .unwrap();
        let (transaction, affected_rows) = stage(self.job, &self.fixture, &dataset).await;
        let mut builder = CommitBuilder::new(Arc::new(dataset))
            .with_experimental_composite_operations(true)
            .with_skip_auto_cleanup(true);
        if let Some(rows) = affected_rows {
            builder = builder.with_affected_rows(rows);
        }

        tracker.incremental_stats();
        PHASE_TIMES.lock().unwrap().clear();
        let baseline = reset_peak_heap();
        let started = Instant::now();
        let committed = builder.execute(transaction).await.unwrap();
        let total = started.elapsed();
        let peak_heap = PEAK_HEAP.load(Ordering::Relaxed) - baseline;
        assert_eq!(committed.manifest.version, self.fixture.latest_version + 2);
        let retry = race_lost
            .lock()
            .unwrap()
            .map_or(Duration::ZERO, |lost| lost.elapsed());
        Measurement {
            total,
            retry,
            peak_heap,
            reads: reads_by_kind(&tracker.incremental_stats()),
            phases: PHASE_TIMES.lock().unwrap().clone(),
        }
    }
}

fn report(name: &str, measurements: &[Measurement]) {
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
    let mb = |bytes: f64| bytes / (1 << 20) as f64;
    let mut line = format!(
        "CATCHUP {name}: total_ms={:.1} retry_ms={:.1} peak_heap_mb={:.1}",
        median(measurements.iter().map(|m| ms(m.total)).collect()),
        median(measurements.iter().map(|m| ms(m.retry)).collect()),
        median(
            measurements
                .iter()
                .map(|m| mb(m.peak_heap as f64))
                .collect()
        ),
    );
    for (kind, label) in READ_KINDS.iter().enumerate() {
        let last = &measurements[measurements.len() - 1].reads[kind];
        line.push_str(&format!(
            " {label}_reads={} {label}_mb={:.1}",
            last.count,
            mb(last.bytes as f64)
        ));
    }
    for phase in PHASES {
        let wall = median(
            measurements
                .iter()
                .map(|m| m.phases.get(phase).map_or(0.0, |t| ms(t.wall)))
                .collect(),
        );
        let busy = median(
            measurements
                .iter()
                .map(|m| m.phases.get(phase).map_or(0.0, |t| ms(t.busy)))
                .collect(),
        );
        line.push_str(&format!(" {phase}_ms={wall:.1}/{busy:.1}"));
    }
    println!("{line}");
}

fn bench_catchup(c: &mut Criterion) {
    AllocationRegistry::set_global_tracker(HeapTracker).unwrap();
    AllocationRegistry::enable_tracking();
    let filter = tracing_subscriber::filter::filter_fn(|metadata| {
        metadata.is_span() && PHASES.contains(&metadata.name())
    });
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(PhaseLayer.with_filter(filter)),
    )
    .unwrap();

    let config = Config::from_env();
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("conflict_catchup");
    group.sample_size(10);
    for &fragments in &config.fragments {
        for &versions in &config.versions {
            if versions * fragments > config.max_fragment_versions {
                eprintln!(
                    "skipping V={versions} N={fragments}: over CATCHUP_MAX_FRAGMENT_VERSIONS"
                );
                continue;
            }
            for &mix in &config.mixes {
                let dir = config
                    .dir
                    .join(format!("v{versions}-n{fragments}-{}", mix.name()));
                let fixture = runtime.block_on(load_fixture(&dir, versions, fragments, mix));
                for &job in &config.jobs {
                    let case = Case {
                        job,
                        fixture: fixture.clone(),
                        dir: dir.clone(),
                    };
                    let name = format!("{}/v{versions}/n{fragments}/{}", job.name(), mix.name());
                    let measurements = Mutex::new(Vec::new());
                    if let Some(iterations) = config.iterations {
                        for _ in 0..iterations {
                            let measurement = runtime.block_on(case.run());
                            measurements.lock().unwrap().push(measurement);
                        }
                        report(&name, &measurements.lock().unwrap());
                        case.reset();
                        continue;
                    }
                    group.bench_function(&name, |b| {
                        b.iter_custom(|iterations| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iterations {
                                let measurement = runtime.block_on(case.run());
                                total += measurement.total;
                                measurements.lock().unwrap().push(measurement);
                            }
                            total
                        })
                    });
                    report(&name, &measurements.lock().unwrap());
                    case.reset();
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(1)).measurement_time(Duration::from_secs(1));
    targets = bench_catchup
);
criterion_main!(benches);
