// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What a long-running job pays to catch up with the versions committed while
//! it ran: a job staged at version `R` commits over `V` concurrent commits,
//! through the public commit path, from a process of its own (a fresh session
//! with no metadata cache, as on a table whose manifests outgrow it).
//!
//! The history is synthetic and never reads data: fragments share the data
//! file of the one fragment the table was created with, deletion files are
//! real but tiny, and index segments are metadata only. The bounds are on
//! what the catch-up reads and holds, counted rather than timed, so they are
//! exact at any size; the sizes here keep each case well under a second.

use std::ops::Range;
use std::sync::Mutex;

use bytes::Bytes;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry, WrappingObjectStore};
use lance_io::utils::tracking_store::{IOTracker, IoStats};
use lance_table::transaction::action::{
    Action as TxnAction, AddDataFile, AddFragment, AddFragmentParts, AddIndexSegment,
    AddIndexSegmentParts, AdjustIndexCoverage, AdjustIndexCoverageParts, CompositeOperation,
    FieldRef, Ref as ActionRef, RemoveFragment, SetDeletionFile, UserStep,
};
use object_store::path::Path;
use object_store::{ObjectStore as _, ObjectStoreExt as _, PutPayload};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;

use super::*;
use crate::dataset::builder::DatasetBuilder;
use crate::session::Session;

/// Fragments at the read version.
const FRAGMENTS: u64 = 1_000;
/// Versions committed while the job ran.
const VERSIONS: u64 = 40;
const ROWS_PER_FRAGMENT: u32 = 16;
/// The fragments a job touches: a compaction removes them all, a delete
/// deletes the last row of the first `JOB_DELETES`.
const JOB_FRAGMENTS: Range<u64> = 0..8;
const JOB_DELETES: u64 = 4;
/// The reads of one manifest: its tail, the rest, the transaction and the
/// index sections. What a named job over named commits, as before action
/// sets, reads per version.
const READS_PER_MANIFEST: u64 = 4;
/// Index metadata reads, which do not grow with the versions caught up on.
const INDEX_READS: u64 = 8;
/// The manifests a retry loads besides the race winner's.
const RETRY_SLACK: u64 = 3;

/// The job staged at the read version.
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

impl Job {
    fn deletes(self) -> bool {
        matches!(self, Self::LegacyDelete | Self::ActionDelete)
    }
}

/// The form the concurrent commits take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mix {
    Legacy,
    Half,
    Action,
}

/// A table with a history of concurrent commits, and the handle they were
/// committed through.
struct History {
    writer: Dataset,
    read_version: u64,
    template: Fragment,
    /// The segment every fixture has at the read version, over the first half
    /// of the fragments, which [`Job::ActionAdjustIndexCoverage`] widens.
    segment: IndexMetadata,
    /// The rows deleted so far, by fragment, so no delete reads the file it
    /// extends.
    deleted: HashMap<u64, RoaringBitmap>,
    /// The fragments appended since the last index optimize.
    unindexed: Vec<u64>,
    rng: SmallRng,
}

fn data_file(template: &Fragment) -> DataFile {
    let mut file = template.files[0].clone();
    file.fields = Default::default();
    file
}

fn action_set(read_version: u64, actions: Vec<TxnAction>) -> Transaction {
    Transaction::new_from_version(
        read_version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "catchup", actions,
        )])),
    )
}

/// An `AddFragment` + `AddDataFile` of a fragment backed by `template`'s data
/// file, as `token`.
fn added_fragment(template: &Fragment, token: u32, rows: u64, data_change: bool) -> [TxnAction; 2] {
    [
        TxnAction::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
            id: ActionRef::Local(token),
            physical_rows: rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change,
        })),
        TxnAction::AddDataFile(
            AddDataFile::try_new_unchecked(
                ActionRef::Local(token),
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

impl History {
    /// A table of `FRAGMENTS` fragments and one index segment, at the version
    /// the job reads.
    async fn new(uri: &str) -> Self {
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
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
        let writer = InsertBuilder::new(uri).execute(vec![data]).await.unwrap();
        let mut template = writer.fragments()[0].clone();
        template.id = 0;
        let mut history = Self {
            writer,
            read_version: 0,
            template: template.clone(),
            segment: segment("idx_a", 0, 0..FRAGMENTS / 2, 0),
            deleted: HashMap::new(),
            unindexed: Vec::new(),
            rng: SmallRng::seed_from_u64(42),
        };
        history
            .commit(
                Transaction::new_from_version(
                    history.writer.manifest.version,
                    Operation::Append {
                        fragments: vec![template; FRAGMENTS as usize - 1],
                    },
                ),
                None,
            )
            .await;
        let mut segment = history.segment.clone();
        segment.dataset_version = history.writer.manifest.version;
        history
            .commit(
                Transaction::new_from_version(
                    history.writer.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![segment.clone()],
                        removed_indices: vec![],
                    },
                ),
                None,
            )
            .await;
        history.segment = segment;
        history.read_version = history.writer.manifest.version;
        history
    }

    async fn commit(&mut self, transaction: Transaction, affected_rows: Option<RowAddrTreeMap>) {
        let mut builder = CommitBuilder::new(Arc::new(self.writer.clone()))
            .with_experimental_composite_operations(true);
        if let Some(rows) = affected_rows {
            builder = builder.with_affected_rows(rows);
        }
        self.writer = builder.execute(transaction).await.unwrap();
    }

    fn live(&self) -> Vec<u64> {
        self.writer.fragments().iter().map(|f| f.id).collect()
    }

    /// Whether the concurrent commits may change `fragment`: never the job's
    /// fragments, except a delete job's, whose rows they delete too.
    fn may_touch(&self, fragment: u64, job: Job) -> bool {
        !JOB_FRAGMENTS.contains(&fragment) || (job.deletes() && fragment < JOB_DELETES)
    }

    /// Commit `count` seeded concurrent commits, a mix of appends, row and
    /// whole-fragment deletes, compactions of fragments appended since the
    /// job read, and index optimizes over those.
    async fn advance(&mut self, count: u64, job: Job, mix: Mix) {
        for step in 0..count {
            let as_action = match mix {
                Mix::Legacy => false,
                Mix::Half => self.rng.random_bool(0.5),
                Mix::Action => true,
            };
            // Every so often a delete job's fragments lose a row too, so it
            // has deletion files to merge.
            if job.deletes() && step % 8 == 0 {
                let fragment = (step / 8) % JOB_DELETES;
                self.delete_row(job, Some(fragment), as_action).await;
                continue;
            }
            let roll = self.rng.random_range(0..100);
            let committed = match roll {
                0..35 => {
                    self.append(as_action).await;
                    true
                }
                35..70 => self.delete_row(job, None, as_action).await,
                70..75 => self.delete_fragment(job, as_action).await,
                75..85 => self.compact(as_action).await,
                _ => self.optimize(as_action).await,
            };
            if !committed {
                self.append(as_action).await;
            }
        }
    }

    async fn append(&mut self, as_action: bool) {
        let read_version = self.writer.manifest.version;
        let transaction = if as_action {
            action_set(
                read_version,
                added_fragment(&self.template, 0, ROWS_PER_FRAGMENT as u64, true).into(),
            )
        } else {
            Transaction::new_from_version(
                read_version,
                Operation::Append {
                    fragments: vec![self.template.clone()],
                },
            )
        };
        self.commit(transaction, None).await;
        let appended = self.writer.manifest.max_fragment_id().unwrap();
        self.unindexed.push(appended);
    }

    /// Delete one more row of a fragment the job allows, keeping the last row
    /// of every fragment for a delete job.
    async fn delete_row(&mut self, job: Job, fragment: Option<u64>, as_action: bool) -> bool {
        let candidates: Vec<u64> = self
            .live()
            .into_iter()
            .filter(|fragment| self.may_touch(*fragment, job))
            .collect();
        let fragment =
            fragment.unwrap_or_else(|| candidates[self.rng.random_range(0..candidates.len())]);
        let deleted = self.deleted.entry(fragment).or_default();
        let Some(row) = (0..ROWS_PER_FRAGMENT - 1).find(|row| !deleted.contains(*row)) else {
            return false;
        };
        deleted.insert(row);
        let deletions = DeletionVector::from(deleted.clone());
        let deletion_file = write_deletion_file(
            &self.writer.base,
            fragment,
            self.writer.manifest.version,
            &deletions,
            self.writer.object_store.as_ref(),
            0,
        )
        .await
        .unwrap()
        .unwrap();
        let read_version = self.writer.manifest.version;
        let rows = RowAddrTreeMap::from_iter([(fragment << 32) | row as u64]);
        let transaction = if as_action {
            action_set(
                read_version,
                vec![TxnAction::SetDeletionFile(
                    SetDeletionFile::try_new_unchecked(fragment, deletion_file, None, true)
                        .unwrap(),
                )],
            )
        } else {
            let mut updated = self
                .writer
                .manifest
                .fragment_by_id(fragment)
                .unwrap()
                .clone();
            updated.deletion_file = Some(deletion_file);
            Transaction::new_from_version(
                read_version,
                Operation::Delete {
                    updated_fragments: vec![updated],
                    deleted_fragment_ids: vec![],
                    predicate: "catchup".into(),
                },
            )
        };
        self.commit(transaction, Some(rows)).await;
        true
    }

    async fn delete_fragment(&mut self, job: Job, as_action: bool) -> bool {
        let candidates: Vec<u64> = self
            .live()
            .into_iter()
            .filter(|fragment| self.may_touch(*fragment, job) && *fragment >= JOB_DELETES)
            .collect();
        let fragment = candidates[self.rng.random_range(0..candidates.len())];
        let read_version = self.writer.manifest.version;
        let transaction = if as_action {
            action_set(
                read_version,
                vec![TxnAction::RemoveFragment(RemoveFragment::new_unchecked(
                    ActionRef::Committed(fragment),
                    true,
                ))],
            )
        } else {
            Transaction::new_from_version(
                read_version,
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![fragment],
                    predicate: "catchup".into(),
                },
            )
        };
        self.commit(transaction, None).await;
        self.unindexed.retain(|unindexed| *unindexed != fragment);
        true
    }

    /// Compact two fragments appended since the job read into one, without
    /// recording the move: no index covers them.
    async fn compact(&mut self, as_action: bool) -> bool {
        let live = self.live();
        let sources: Vec<Fragment> = self
            .unindexed
            .iter()
            .filter(|id| live.contains(id) && !self.deleted.contains_key(id))
            .take(2)
            .map(|id| self.writer.manifest.fragment_by_id(*id).unwrap().clone())
            .collect();
        if sources.len() < 2 {
            return false;
        }
        let read_version = self.writer.manifest.version;
        let rows = 2 * ROWS_PER_FRAGMENT as u64;
        let transaction = if as_action {
            let mut actions: Vec<TxnAction> = sources
                .iter()
                .map(|fragment| {
                    TxnAction::RemoveFragment(RemoveFragment::new_unchecked(
                        ActionRef::Committed(fragment.id),
                        false,
                    ))
                })
                .collect();
            actions.extend(added_fragment(&self.template, 0, rows, false));
            action_set(read_version, actions)
        } else {
            let mut compacted = self.template.clone();
            compacted.physical_rows = Some(rows as usize);
            Transaction::new_from_version(
                read_version,
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
        self.commit(transaction, None).await;
        self.unindexed
            .retain(|id| !sources.iter().any(|source| source.id == *id));
        let compacted = self.writer.manifest.max_fragment_id().unwrap();
        self.unindexed.push(compacted);
        let fragments = self.writer.fragments();
        assert!(
            fragments.windows(2).all(|pair| pair[0].id < pair[1].id),
            "a compaction left the fragments out of id order"
        );
        true
    }

    /// Index the fragments appended since the last optimize, as a new segment
    /// of a delta index.
    async fn optimize(&mut self, as_action: bool) -> bool {
        let live = self.live();
        let fragments: Vec<u64> = std::mem::take(&mut self.unindexed)
            .into_iter()
            .filter(|id| live.contains(id))
            .collect();
        if fragments.is_empty() {
            return false;
        }
        let read_version = self.writer.manifest.version;
        let segment = segment("opt", 1, fragments.iter().copied(), read_version);
        let transaction = if as_action {
            action_set(
                read_version,
                vec![TxnAction::AddIndexSegment(
                    AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                        uuid: segment.uuid,
                        name: segment.name,
                        fields: vec![FieldRef::Committed(1)],
                        covering_fields: vec![],
                        index_details: None,
                        index_version: 0,
                        covered_fragments: Some(
                            fragments.into_iter().map(ActionRef::Committed).collect(),
                        ),
                        files: vec![],
                        base: None,
                        created_at: None,
                        dataset_version: Some(read_version),
                        data_change: false,
                    })
                    .unwrap(),
                )],
            )
        } else {
            Transaction::new_from_version(
                read_version,
                Operation::CreateIndex {
                    new_indices: vec![segment],
                    removed_indices: vec![],
                },
            )
        };
        self.commit(transaction, None).await;
        true
    }

    /// `job`, staged at the read version, and the rows it deletes.
    async fn stage(&self, job: Job) -> (Transaction, Option<RowAddrTreeMap>) {
        let read_version = self.read_version;
        let deleted_rows = || {
            RowAddrTreeMap::from_iter(
                (0..JOB_DELETES).map(|fragment| (fragment << 32) | (ROWS_PER_FRAGMENT - 1) as u64),
            )
        };
        let mut deletion_files = Vec::new();
        if job.deletes() {
            for fragment in 0..JOB_DELETES {
                let deletions =
                    DeletionVector::from(RoaringBitmap::from_iter([ROWS_PER_FRAGMENT - 1]));
                let file = write_deletion_file(
                    &self.writer.base,
                    fragment,
                    read_version,
                    &deletions,
                    self.writer.object_store.as_ref(),
                    0,
                )
                .await
                .unwrap()
                .unwrap();
                deletion_files.push((fragment, file));
            }
        }
        let new_segment = || segment("job_idx", 1, 0..FRAGMENTS, read_version);
        match job {
            Job::LegacyAppend => (
                Transaction::new_from_version(
                    read_version,
                    Operation::Append {
                        fragments: vec![self.template.clone()],
                    },
                ),
                None,
            ),
            Job::ActionAppend => (
                action_set(
                    read_version,
                    added_fragment(&self.template, 0, ROWS_PER_FRAGMENT as u64, true).into(),
                ),
                None,
            ),
            Job::LegacyDelete => {
                let updated_fragments = deletion_files
                    .into_iter()
                    .map(|(id, file)| {
                        let mut fragment = self.template.clone();
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
                            TxnAction::SetDeletionFile(
                                SetDeletionFile::try_new_unchecked(id, file, None, true).unwrap(),
                            )
                        })
                        .collect(),
                ),
                Some(deleted_rows()),
            ),
            Job::ActionCompaction => {
                let mut actions: Vec<TxnAction> = JOB_FRAGMENTS
                    .map(|id| {
                        TxnAction::RemoveFragment(RemoveFragment::new_unchecked(
                            ActionRef::Committed(id),
                            false,
                        ))
                    })
                    .collect();
                let rows = (JOB_FRAGMENTS.end - JOB_FRAGMENTS.start) * ROWS_PER_FRAGMENT as u64;
                actions.extend(added_fragment(&self.template, 0, rows, false));
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
            Job::ActionAddIndexSegment => {
                let segment = new_segment();
                (
                    action_set(
                        read_version,
                        vec![TxnAction::AddIndexSegment(
                            AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
                                uuid: segment.uuid,
                                name: segment.name,
                                fields: vec![FieldRef::Committed(1)],
                                covering_fields: vec![],
                                index_details: None,
                                index_version: 0,
                                covered_fragments: Some(
                                    (0..FRAGMENTS).map(ActionRef::Committed).collect(),
                                ),
                                files: vec![],
                                base: None,
                                created_at: None,
                                dataset_version: Some(read_version),
                                data_change: false,
                            })
                            .unwrap(),
                        )],
                    ),
                    None,
                )
            }
            Job::ActionAdjustIndexCoverage => (
                action_set(
                    read_version,
                    vec![TxnAction::AdjustIndexCoverage(
                        AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
                            uuid: self.segment.uuid,
                            name: self.segment.name.clone(),
                            add_fragments: (FRAGMENTS / 2..FRAGMENTS)
                                .map(ActionRef::Committed)
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

    /// Commit one more append, and take it back out of the store: the files
    /// it wrote, to be put back as a writer that wins the race against the
    /// job's commit.
    async fn race_winner(&mut self) -> Vec<(Path, Bytes)> {
        let store = self.writer.object_store.inner.clone();
        let base = self.writer.base.clone();
        let list = || async {
            store
                .list(Some(&base))
                .map_ok(|meta| (meta.location, meta.e_tag))
                .try_collect::<HashMap<_, _>>()
                .await
                .unwrap()
        };
        let before = list().await;
        self.append(true).await;
        let mut files = Vec::new();
        for (path, e_tag) in list().await {
            match before.get(&path) {
                None => {
                    let bytes = store.get(&path).await.unwrap().bytes().await.unwrap();
                    store.delete(&path).await.unwrap();
                    files.push((path, bytes));
                }
                Some(was) => assert_eq!(was, &e_tag, "a commit rewrote {path}"),
            }
        }
        files
    }
}

/// What the job's commit read, held and repeated.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Cost {
    /// Manifests read through, by reads that reached the end of the file. A
    /// manifest is read from its end, so each load is one such read.
    manifest_loads: u64,
    /// Every read of a manifest, the index section and partial reads included.
    manifest_reads: u64,
    deletion_reads: u64,
    index_reads: u64,
    peak_history: usize,
    window_checks: usize,
}

impl Cost {
    fn of(stats: &IoStats, sizes: &HashMap<Path, u64>, observed: probe::Observed) -> Self {
        let mut cost = Self {
            peak_history: observed.peak_history,
            window_checks: observed.window_checks,
            ..Default::default()
        };
        for request in &stats.requests {
            let path = request.path.as_ref();
            if request.method == "list" && path.contains("/_indices/") {
                cost.index_reads += 1;
            }
            if !matches!(request.method, "get_opts" | "get_ranges") {
                continue;
            }
            if path.contains("/_versions/") {
                cost.manifest_reads += 1;
                let reaches_end = request.range.as_ref().is_none_or(|range| {
                    sizes
                        .get(&request.path)
                        .is_some_and(|size| range.end == *size)
                });
                cost.manifest_loads += u64::from(reaches_end);
            } else if path.contains("/_deletions/") {
                cost.deletion_reads += 1;
            } else if path.contains("/_indices/") {
                cost.index_reads += 1;
            }
        }
        cost
    }
}

/// Puts the race winner's files in place just before the job's first
/// manifest write, so its first attempt loses and the retry catches up.
struct Race {
    winner: Mutex<Option<Vec<(Path, Bytes)>>>,
    store: Arc<dyn object_store::ObjectStore>,
    tracker: IOTracker,
    /// What the first attempt read and observed, taken when it lost.
    first_attempt: Mutex<Option<(IoStats, probe::Observed)>>,
}

/// The job's store: counted by `tracker`, and losing `race` if there is one.
#[derive(Debug)]
struct JobStore {
    tracker: IOTracker,
    policy: Arc<std::sync::Mutex<ProxyObjectStorePolicy>>,
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

impl std::fmt::Debug for Race {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Race").finish_non_exhaustive()
    }
}

/// The cost of committing `job` over `VERSIONS` commits of `mix`, and, with
/// `lose_race`, of the retry after losing its first attempt to one more.
async fn catch_up(job: Job, mix: Mix, lose_race: bool) -> (Cost, Option<Cost>) {
    let uri = format!("shared-memory://catchup-{}/table", Uuid::new_v4());
    let mut history = History::new(&uri).await;
    let (transaction, affected_rows) = history.stage(job).await;
    history.advance(VERSIONS, job, mix).await;
    let winner = if lose_race {
        Some(history.race_winner().await)
    } else {
        None
    };

    let tracker = IOTracker::default();
    let race = Arc::new(Race {
        winner: Mutex::new(winner),
        store: history.writer.object_store.inner.clone(),
        tracker: tracker.clone(),
        first_attempt: Mutex::new(None),
    });
    let mut policy = ProxyObjectStorePolicy::new();
    let on_write = race.clone();
    policy.set_before_policy(
        "lose_race",
        Arc::new(move |method, path| {
            let is_manifest_write = matches!(
                method,
                "put" | "copy_if_not_exists" | "rename_if_not_exists"
            ) && path.as_ref().contains("/_versions/");
            if is_manifest_write && let Some(files) = on_write.winner.lock().unwrap().take() {
                *on_write.first_attempt.lock().unwrap() =
                    Some((on_write.tracker.incremental_stats(), probe::take()));
                for (path, bytes) in files {
                    futures::executor::block_on(on_write.store.put(&path, PutPayload::from(bytes)))
                        .unwrap();
                }
            }
            Ok(())
        }),
    );
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(JobStore {
            tracker: tracker.clone(),
            policy: Arc::new(std::sync::Mutex::new(policy)),
        })),
        ..Default::default()
    };
    // A process of its own, whose metadata cache holds none of the manifests.
    let session = Arc::new(Session::new(0, 0, Arc::new(ObjectStoreRegistry::default())));
    let job_dataset = DatasetBuilder::from_uri(&uri)
        .with_session(session)
        .with_version(history.read_version)
        .with_store_params(store_params)
        .load()
        .await
        .unwrap();
    tracker.incremental_stats();
    probe::take();

    let mut builder = CommitBuilder::new(Arc::new(job_dataset))
        .with_experimental_composite_operations(true)
        .with_skip_auto_cleanup(true);
    if let Some(rows) = affected_rows {
        builder = builder.with_affected_rows(rows);
    }
    let committed = builder.execute(transaction).await.unwrap();
    let expected = history.read_version + VERSIONS + 1 + u64::from(lose_race);
    assert_eq!(committed.manifest.version, expected);

    let sizes = race
        .store
        .list(Some(&history.writer.base))
        .map_ok(|meta| (meta.location, meta.size))
        .try_collect::<HashMap<_, _>>()
        .await
        .unwrap();
    let last = Cost::of(&tracker.incremental_stats(), &sizes, probe::take());
    let first = race.first_attempt.lock().unwrap().take();
    match first {
        Some((stats, observed)) => (Cost::of(&stats, &sizes, observed), Some(last)),
        None => {
            assert!(!lose_race, "the job never wrote its manifest");
            (last, None)
        }
    }
}

/// A job catching up on `VERSIONS` commits loads each manifest once, reads
/// each deletion file it merges at most twice (as it read it and as it is
/// now) and the index metadata a bounded number of times, holds no manifest
/// per version, and checks the window once.
#[rstest]
#[tokio::test]
async fn catch_up_reads_each_version_once(
    #[values(
        Job::LegacyAppend,
        Job::LegacyDelete,
        Job::LegacyCreateIndex,
        Job::ActionAppend,
        Job::ActionDelete,
        Job::ActionCompaction,
        Job::ActionAddIndexSegment,
        Job::ActionAdjustIndexCoverage
    )]
    job: Job,
    #[values(Mix::Legacy, Mix::Half, Mix::Action)] mix: Mix,
) {
    let (cost, _) = catch_up(job, mix, false).await;
    let merged = if job.deletes() { JOB_DELETES } else { 0 };
    let bounds = Cost {
        manifest_loads: VERSIONS + 1,
        manifest_reads: READS_PER_MANIFEST * (VERSIONS + 1) + 2,
        deletion_reads: 2 * merged,
        index_reads: INDEX_READS,
        peak_history: 1,
        window_checks: 1,
    };
    assert_within(&cost, &bounds, &format!("{job:?} over {mix:?}"));
}

/// The retry after a lost race catches up on the one commit that won it,
/// whatever the first attempt caught up on.
#[rstest]
#[tokio::test]
async fn retry_reads_only_the_race_winner(
    #[values(
        Job::LegacyDelete,
        Job::LegacyCreateIndex,
        Job::ActionDelete,
        Job::ActionCompaction,
        Job::ActionAddIndexSegment
    )]
    job: Job,
) {
    let (_, retry) = catch_up(job, Mix::Half, true).await;
    let retry = retry.unwrap();
    let merged = if job.deletes() { JOB_DELETES } else { 0 };
    let bounds = Cost {
        manifest_loads: 1 + RETRY_SLACK,
        manifest_reads: READS_PER_MANIFEST * (1 + RETRY_SLACK) + 2,
        deletion_reads: 2 * merged,
        index_reads: INDEX_READS,
        peak_history: 1,
        window_checks: 1,
    };
    assert_within(&retry, &bounds, &format!("retry of {job:?}"));
}

#[track_caller]
fn assert_within(cost: &Cost, bounds: &Cost, case: &str) {
    let over = [
        ("manifest_loads", cost.manifest_loads, bounds.manifest_loads),
        ("manifest_reads", cost.manifest_reads, bounds.manifest_reads),
        ("deletion_reads", cost.deletion_reads, bounds.deletion_reads),
        ("index_reads", cost.index_reads, bounds.index_reads),
        (
            "peak_history",
            cost.peak_history as u64,
            bounds.peak_history as u64,
        ),
        (
            "window_checks",
            cost.window_checks as u64,
            bounds.window_checks as u64,
        ),
    ]
    .into_iter()
    .filter(|(_, actual, bound)| actual > bound)
    .map(|(name, actual, bound)| format!("{name} {actual} > {bound}"))
    .collect::<Vec<_>>();
    assert!(over.is_empty(), "{case}: {} ({cost:?})", over.join(", "));
}
