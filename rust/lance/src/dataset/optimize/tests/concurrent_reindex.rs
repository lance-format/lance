// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An index built while other processes write to and compact the table, then
//! committed through conflict resolution. On a table whose fragment reuse index
//! is untagged, a deferred compaction does not conflict with the build; changes
//! that leave its coverage stale withdraw that coverage instead. Every commit is
//! checked by comparing index lookups with a scan.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use futures::future::BoxFuture;

use super::*;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};
use crate::session::Session;

const ROWS_PER_FRAGMENT: i32 = 6;

/// `fragments` fragments of `id`, `val = id * 10` and `spare`, with a committed
/// index on `id`, so a deferred compaction records its groups. An update of
/// `val` leaves `spare` alone, so it rewrites the column in place.
async fn indexed_table(uri: &str, fragments: i32) -> Dataset {
    let rows = fragments * ROWS_PER_FRAGMENT;
    let batch = record_batch!(
        ("id", Int32, (0..rows).collect::<Vec<_>>()),
        (
            "val",
            Int32,
            (0..rows).map(|id| id * 10).collect::<Vec<_>>()
        ),
        ("spare", Int32, vec![42; rows as usize])
    )
    .unwrap();
    let params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        max_rows_per_group: ROWS_PER_FRAGMENT as usize,
        data_storage_version: Some(LanceFileVersion::Stable),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, uri, Some(params)).await.unwrap();
    build_index(&mut dataset, "id", "id_idx", false)
        .await
        .unwrap();
    dataset
}

/// The table opened in a fresh session, as another process would.
async fn open_in_new_session(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(Session::default()))
        .load()
        .await
        .unwrap()
}

async fn build_index(dataset: &mut Dataset, column: &str, name: &str, replace: bool) -> Result<()> {
    dataset
        .create_index(
            &[column],
            IndexType::BTree,
            Some(name.into()),
            &ScalarIndexParams::default(),
            replace,
        )
        .await
        .map(|_| ())
}

/// Compacts into fragments of `fragments_per_group` inputs, deferring the remap.
fn deferred_compaction(fragments_per_group: i32) -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: (fragments_per_group * ROWS_PER_FRAGMENT) as usize,
        defer_index_remap: true,
        ..Default::default()
    }
}

/// [`deferred_compaction`] that also rewrites every fragment carrying an
/// overlay, folding its values into the base data.
fn folding_compaction(fragments_per_group: i32) -> CompactionOptions {
    CompactionOptions {
        max_overlays_per_fragment: Some(0),
        ..deferred_compaction(fragments_per_group)
    }
}

/// Sets `val` of each id, rewriting the column in place.
async fn update_in_place(dataset: Dataset, ids: Vec<i32>, vals: Vec<i32>) {
    let patch = record_batch!(("id", Int32, ids), ("val", Int32, vals)).unwrap();
    let schema = patch.schema();
    let mut merge = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()]).unwrap();
    merge
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns);
    merge
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
        .await
        .unwrap();
}

/// Overlays `val` of the first row of `fragment` with `value`.
async fn overlay_val(dataset: Dataset, fragment: u64, value: i32) -> Dataset {
    let val = dataset.schema().field("val").unwrap().id;
    commit_overlay(
        dataset,
        fragment,
        &[val],
        OverlayCoverage::dense(bitmap([0])),
        vec![i32_array([Some(value)])],
    )
    .await
}

async fn clean_up_versions(uri: &str, versions: Vec<u64>) {
    let policy = CleanupPolicyBuilder::default()
        .versions(versions.clone())
        .unwrap()
        .build();
    let dataset = open_in_new_session(uri).await;
    dataset.cleanup_with_policy(policy).await.unwrap();
    let retained = dataset.version_refs().await.unwrap();
    assert!(
        versions
            .iter()
            .all(|gone| retained.iter().all(|v| v.version != *gone))
    );
}

async fn live_fragments(uri: &str) -> RoaringBitmap {
    let dataset = open_in_new_session(uri).await;
    dataset.fragments().iter().map(|f| f.id as u32).collect()
}

/// The fragment that holds the row with `id`.
async fn fragment_of(uri: &str, id: i32) -> u32 {
    let dataset = open_in_new_session(uri).await;
    let mut scan = dataset.scan();
    scan.filter(&format!("id = {id}"))
        .unwrap()
        .with_row_address()
        .project(&["id"])
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    (batch[lance_core::ROW_ADDR]
        .as_primitive::<UInt64Type>()
        .value(0)
        >> 32) as u32
}

/// The fragments `name` covers, by their current ids.
async fn coverage(uri: &str, name: &str) -> RoaringBitmap {
    let dataset = open_in_new_session(uri).await;
    let index = dataset.load_index_by_name(name).await.unwrap().unwrap();
    index.fragment_bitmap.unwrap()
}

/// Checks that `val_idx` exists and that looking up each of `values`, and a few
/// ranges, through it finds what a scan finds.
async fn assert_lookups_match_scan(uri: &str, values: impl IntoIterator<Item = i32>) {
    let dataset = open_in_new_session(uri).await;
    assert!(
        dataset
            .load_index_by_name("val_idx")
            .await
            .unwrap()
            .is_some()
    );
    let mut scan = dataset.scan();
    scan.use_scalar_index(false).project(&["val"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut scanned = HashMap::<i32, usize>::new();
    for val in batch["val"].as_primitive::<Int32Type>().values() {
        *scanned.entry(*val).or_default() += 1;
    }
    for value in values {
        let found = dataset
            .count_rows(Some(format!("val = {value}")))
            .await
            .unwrap();
        let want = scanned.get(&value).copied().unwrap_or(0);
        assert_eq!(found, want, "val = {value}");
    }
    for bound in [0, 100, 1_000, 1_000_000, 3_000_000] {
        let found = dataset
            .count_rows(Some(format!("val < {bound}")))
            .await
            .unwrap();
        let want = scanned
            .iter()
            .filter(|(value, _)| **value < bound)
            .map(|(_, count)| count)
            .sum::<usize>();
        assert_eq!(found, want, "val < {bound}");
    }
}

fn assert_retryable(result: Result<()>) {
    let error = result.unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "expected RetryableCommitConflict, got: {error:?}"
    );
}

type Hook = BoxFuture<'static, ()>;

/// Runs a hook during the first index commit attempt, so that attempt loses
/// its race and the commit retries.
#[derive(Default)]
struct BeforeFirstIndexCommit {
    hook: Mutex<Option<Hook>>,
    attempts: AtomicUsize,
}

impl std::fmt::Debug for BeforeFirstIndexCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeforeFirstIndexCommit")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl CommitHandler for BeforeFirstIndexCommit {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &object_store::path::Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let is_index = transaction
            .as_ref()
            .and_then(|transaction| transaction.as_pb().operation.as_ref())
            .is_some_and(|operation| matches!(operation, PbOperation::CreateIndex(_)));
        if is_index {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            let hook = self.hook.lock().unwrap().take();
            if let Some(hook) = hook {
                hook.await;
            }
        }
        ConditionalPutCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

/// An index builder in another process whose first commit attempt runs `hook`.
async fn builder_racing(uri: &str, hook: Hook) -> (Dataset, Arc<BeforeFirstIndexCommit>) {
    let race = Arc::new(BeforeFirstIndexCommit {
        hook: Mutex::new(Some(hook)),
        ..Default::default()
    });
    let builder = DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(Session::default()))
        .with_commit_handler(race.clone())
        .load()
        .await
        .unwrap();
    (builder, race)
}

/// Original values of the first `rows` ids, plus `extra`.
fn probes(rows: i32, extra: impl IntoIterator<Item = i32>) -> Vec<i32> {
    (0..rows).map(|id| id * 10).chain(extra).collect()
}

/// The index lands past a deferred compaction, in this process or another,
/// covering the compacted fragments under their old ids.
#[rstest]
#[case::other_process(false)]
#[case::same_process(true)]
#[tokio::test]
async fn test_index_commits_past_a_deferred_compaction(#[case] same_process: bool) {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 4).await;
    let mut builder = if same_process {
        table.clone()
    } else {
        open_in_new_session(uri).await
    };
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    assert_eq!(coverage(uri, "val_idx").await, live_fragments(uri).await);
    assert_lookups_match_scan(uri, probes(24, [])).await;
}

/// A chain of deferred compactions is followed to the fragments the index
/// covers.
#[tokio::test]
async fn test_index_commits_past_chained_deferred_compactions() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 4).await;
    let mut builder = open_in_new_session(uri).await;
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();
    compact_files(&mut table, deferred_compaction(4), None)
        .await
        .unwrap();
    assert_eq!(table.fragments().len(), 1);

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    assert_eq!(coverage(uri, "val_idx").await, live_fragments(uri).await);
    assert_lookups_match_scan(uri, probes(24, [])).await;
}

/// A group mixing a fragment the index covers with one written after its
/// build is left out of its coverage, and those rows are scanned.
#[tokio::test]
async fn test_index_leaves_out_a_partly_covered_group() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 3).await;
    let mut builder = open_in_new_session(uri).await;
    let batch = record_batch!(
        ("id", Int32, (18..24).collect::<Vec<_>>()),
        ("val", Int32, (18..24).map(|id| id * 10).collect::<Vec<_>>()),
        ("spare", Int32, vec![42; 6])
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    table.append(reader, None).await.unwrap();
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    // Fragments 0 and 1 form a covered group; 2 and the appended 3 do not.
    let covered = fragment_of(uri, 0).await;
    assert_ne!(covered, fragment_of(uri, 18).await);
    assert_eq!(
        coverage(uri, "val_idx").await,
        RoaringBitmap::from_iter([covered])
    );
    assert_lookups_match_scan(uri, probes(24, [])).await;
}

/// A compaction that remapped indexes inline leaves nothing to translate an
/// index built before it, so the index retries.
#[tokio::test]
async fn test_index_retries_past_an_eager_compaction() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 4).await;
    // A deferred compaction first, so the table has a fragment reuse index.
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();
    let mut builder = open_in_new_session(uri).await;
    let eager = CompactionOptions {
        defer_index_remap: false,
        ..deferred_compaction(4)
    };
    compact_files(&mut table, eager, None).await.unwrap();

    assert_retryable(build_index(&mut builder, "val", "val_idx", false).await);
}

/// Once the remap job trims a compaction's record, an index from another
/// process has nothing to translate its old coverage and retries.
#[tokio::test]
async fn test_index_retries_after_a_trimmed_compaction() {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 4).await;
    let mut builder = open_in_new_session(uri).await;
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();
    remapping::remap_column_index(&mut table, &["id"], Some("id_idx".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut table).await.unwrap();
    assert!(
        table
            .frag_reuse_index()
            .await
            .unwrap()
            .is_none_or(|fri| fri.details.versions.is_empty())
    );

    assert_retryable(build_index(&mut builder, "val", "val_idx", false).await);
}

/// An in-place update on a compaction's output withdraws the index's coverage
/// of the fragments that output came from, and only those.
#[rstest]
#[case::other_process(false)]
#[case::same_process(true)]
#[tokio::test]
async fn test_index_withdraws_a_compaction_output_updated_in_place(#[case] same_process: bool) {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 4).await;
    let mut builder = if same_process {
        table.clone()
    } else {
        open_in_new_session(uri).await
    };
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();
    // Row 0 is in fragment 0, compacted with fragment 1.
    update_in_place(table, vec![0], vec![999]).await;

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    let mut untouched = live_fragments(uri).await;
    untouched.remove(fragment_of(uri, 0).await);
    assert_eq!(coverage(uri, "val_idx").await, untouched);
    assert_lookups_match_scan(uri, probes(24, [999])).await;
}

#[derive(Clone, Copy, Debug)]
enum Fold {
    /// The overlay is folded in by the compaction that reads it.
    ByTheCompaction,
    /// It lands on a compaction's output and a second compaction folds it in.
    ByALaterCompaction,
    /// It stays an overlay, which masks the index's stale values at query time.
    Never,
}

/// An overlay newer than the index, folded into a compaction's output, leaves
/// nothing to mask the stale values, so the group is withdrawn.
#[rstest]
#[case::by_the_compaction(Fold::ByTheCompaction, false)]
#[case::by_the_compaction_same_process(Fold::ByTheCompaction, true)]
#[case::by_a_later_compaction(Fold::ByALaterCompaction, false)]
#[case::never(Fold::Never, false)]
#[tokio::test]
async fn test_index_withdraws_a_folded_overlay(#[case] fold: Fold, #[case] same_process: bool) {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 2).await;
    let mut builder = if same_process {
        table.clone()
    } else {
        open_in_new_session(uri).await
    };
    if matches!(fold, Fold::ByALaterCompaction | Fold::Never) {
        compact_files(&mut table, deferred_compaction(2), None)
            .await
            .unwrap();
    }
    let fragment = table.fragments()[0].id;
    let mut table = overlay_val(table, fragment, 999).await;
    if !matches!(fold, Fold::Never) {
        compact_files(&mut table, folding_compaction(2), None)
            .await
            .unwrap();
        assert!(table.fragments().iter().all(|f| f.overlays.is_empty()));
    }

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    assert_lookups_match_scan(uri, probes(12, [999])).await;
}

#[derive(Clone, Copy, Debug)]
enum Racing {
    UpdateInPlace,
    FoldOverlay,
    /// An in-place update, then cleanup of the compaction's version.
    UpdateAndCleanUpCompaction,
    /// The remap job trims the compaction's record.
    RemapAndTrim,
}

/// The first index commit attempt loses a race to a change; the retry, which
/// only checks what committed since, still accounts for the compaction the
/// first attempt let through.
#[rstest]
#[case::update_in_place(Racing::UpdateInPlace)]
#[case::fold_overlay(Racing::FoldOverlay)]
#[case::update_and_clean_up_compaction(Racing::UpdateAndCleanUpCompaction)]
#[case::remap_and_trim(Racing::RemapAndTrim)]
#[tokio::test]
async fn test_index_retry_accounts_for_a_compaction_already_checked(#[case] racing: Racing) {
    let dir = TempStrDir::default();
    let uri = dir.as_str().to_string();
    let table = indexed_table(&uri, 2).await;
    // Keeps the build's version through the cleanup.
    table
        .tags()
        .create("index-build", table.manifest.version)
        .await
        .unwrap();
    let hook_uri = uri.clone();
    let hook: Hook = Box::pin(async move {
        let mut table = open_in_new_session(&hook_uri).await;
        let compaction = table.manifest.version;
        match racing {
            Racing::UpdateInPlace => update_in_place(table, vec![0], vec![999]).await,
            Racing::FoldOverlay => {
                let fragment = table.fragments()[0].id;
                let mut table = overlay_val(table, fragment, 999).await;
                compact_files(&mut table, folding_compaction(1), None)
                    .await
                    .unwrap();
            }
            Racing::UpdateAndCleanUpCompaction => {
                update_in_place(table, vec![0], vec![999]).await;
                clean_up_versions(&hook_uri, vec![compaction]).await;
            }
            Racing::RemapAndTrim => {
                remapping::remap_column_index(&mut table, &["id"], Some("id_idx".into()))
                    .await
                    .unwrap();
                cleanup_frag_reuse_index(&mut table).await.unwrap();
            }
        }
    });
    let (mut builder, race) = builder_racing(&uri, hook).await;
    let mut table = open_in_new_session(&uri).await;
    compact_files(&mut table, deferred_compaction(2), None)
        .await
        .unwrap();

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    assert_eq!(race.attempts.load(Ordering::SeqCst), 2);
    assert_lookups_match_scan(&uri, probes(12, [999])).await;
}

/// A retry keeps what the first attempt pruned for an update, even once the
/// update's version is cleaned up. No compaction is involved.
#[tokio::test]
async fn test_index_retry_keeps_pruning_for_a_cleaned_up_update() {
    let dir = TempStrDir::default();
    let uri = dir.as_str().to_string();
    let table = indexed_table(&uri, 2).await;
    table
        .tags()
        .create("index-build", table.manifest.version)
        .await
        .unwrap();
    let hook_uri = uri.clone();
    let hook: Hook = Box::pin(async move {
        let mut table = open_in_new_session(&hook_uri).await;
        let update = table.manifest.version;
        table.update_config([("racing", "true")]).await.unwrap();
        clean_up_versions(&hook_uri, vec![update]).await;
    });
    let (mut builder, race) = builder_racing(&uri, hook).await;
    update_in_place(open_in_new_session(&uri).await, vec![0], vec![999]).await;

    build_index(&mut builder, "val", "val_idx", false)
        .await
        .unwrap();

    assert_eq!(race.attempts.load(Ordering::SeqCst), 2);
    assert_lookups_match_scan(&uri, probes(12, [999])).await;
}

/// A cleaned-up version in the window can hide a change to a compaction's
/// output, so an index covering a recorded compaction's sources refuses it.
/// An index covering none, or a table with no fragment reuse index, commits.
#[rstest]
#[case::hides_a_covered_compaction(true, true, false)]
#[case::compaction_before_the_build(true, false, true)]
#[case::no_fragment_reuse_index(false, false, true)]
#[tokio::test]
async fn test_index_over_a_cleaned_up_version(
    #[case] compact: bool,
    #[case] compaction_in_window: bool,
    #[case] commits: bool,
) {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 2).await;
    if compact && !compaction_in_window {
        compact_files(&mut table, deferred_compaction(2), None)
            .await
            .unwrap();
    }
    let read_version = table.manifest.version;
    table
        .tags()
        .create("index-build", read_version)
        .await
        .unwrap();
    let mut builder = open_in_new_session(uri).await;
    if compaction_in_window {
        compact_files(&mut table, deferred_compaction(2), None)
            .await
            .unwrap();
    } else {
        table.update_config([("first", "true")]).await.unwrap();
    }
    update_in_place(table, vec![0], vec![999]).await;
    clean_up_versions(uri, vec![read_version + 1]).await;

    let result = build_index(&mut builder, "val", "val_idx", false).await;
    if commits {
        result.unwrap();
        assert_lookups_match_scan(uri, probes(12, [999])).await;
    } else {
        assert_retryable(result);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Append,
    Delete,
    UpdateInPlace,
    Overlay,
    Compact,
    CompactFoldingOverlays,
    AddColumn,
    /// The remap job: remap `id_idx`, then trim the fragment reuse index.
    RemapAndTrim,
}

/// Writes from other processes. Appended ids count up with `val = id * 10`,
/// updates set `val` from 1_000_001 and overlays from 2_000_001.
struct Workload {
    uri: String,
    rng: Mutex<u64>,
    next_id: AtomicI32,
    updates: AtomicI32,
    overlays: AtomicI32,
}

impl Workload {
    fn new(uri: &str, seed: u64, rows: i32) -> Self {
        Self {
            uri: uri.to_string(),
            rng: Mutex::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
            next_id: rows.into(),
            updates: 0.into(),
            overlays: 0.into(),
        }
    }

    fn next(&self, bound: u64) -> u64 {
        let mut rng = self.rng.lock().unwrap();
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng % bound
    }

    /// `len` operations drawn from `kinds`.
    fn draw(&self, kinds: &[Op], len: usize) -> Vec<Op> {
        (0..len)
            .map(|_| kinds[self.next(kinds.len() as u64) as usize])
            .collect()
    }

    async fn run(&self, ops: Vec<Op>) {
        for op in ops {
            let mut table = open_in_new_session(&self.uri).await;
            match op {
                Op::Append => {
                    let first = self.next_id.fetch_add(ROWS_PER_FRAGMENT, Ordering::SeqCst);
                    let ids = (first..first + ROWS_PER_FRAGMENT).collect::<Vec<_>>();
                    let rows = ids.len();
                    let schema = Arc::new(ArrowSchema::from(table.schema()));
                    let columns = schema
                        .fields()
                        .iter()
                        .map(|field| match field.name().as_str() {
                            "id" => Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
                            "val" => {
                                Arc::new(Int32Array::from_iter_values(ids.iter().map(|id| id * 10)))
                            }
                            "spare" => Arc::new(Int32Array::from(vec![42; rows])),
                            _ => arrow_array::new_null_array(field.data_type(), rows),
                        })
                        .collect();
                    let batch = RecordBatch::try_new(schema, columns).unwrap();
                    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
                    table.append(reader, None).await.unwrap();
                }
                Op::Delete => {
                    let filter = format!("id % 11 = {}", self.next(11));
                    table.delete(&filter).await.unwrap();
                }
                Op::UpdateInPlace => {
                    let mut ids = Vec::new();
                    while ids.len() < 3 {
                        let id = self.next(self.next_id.load(Ordering::SeqCst) as u64) as i32;
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                    let vals = ids
                        .iter()
                        .map(|_| 1_000_001 + self.updates.fetch_add(1, Ordering::SeqCst))
                        .collect();
                    update_in_place(table, ids, vals).await;
                }
                Op::Overlay => {
                    let fragments = table.fragments();
                    let fragment = fragments[self.next(fragments.len() as u64) as usize].id;
                    let value = 2_000_001 + self.overlays.fetch_add(1, Ordering::SeqCst);
                    overlay_val(table, fragment, value).await;
                }
                Op::Compact | Op::CompactFoldingOverlays => {
                    let fragments_per_group = 2 + self.next(3) as i32;
                    let options = if op == Op::Compact {
                        deferred_compaction(fragments_per_group)
                    } else {
                        folding_compaction(fragments_per_group)
                    };
                    compact_files(&mut table, options, None).await.unwrap();
                }
                Op::AddColumn => {
                    let name = format!("extra_{}", table.manifest.version);
                    table
                        .add_columns(
                            crate::dataset::NewColumnTransform::SqlExpressions(vec![(
                                name,
                                "CAST(NULL AS INT)".into(),
                            )]),
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                }
                Op::RemapAndTrim => {
                    if table.frag_reuse_index().await.unwrap().is_some() {
                        remapping::remap_column_index(&mut table, &["id"], Some("id_idx".into()))
                            .await
                            .unwrap();
                        cleanup_frag_reuse_index(&mut table).await.unwrap();
                    }
                }
            }
        }
    }

    fn probes(&self) -> Vec<i32> {
        let updates = self.updates.load(Ordering::SeqCst);
        let overlays = self.overlays.load(Ordering::SeqCst);
        probes(self.next_id.load(Ordering::SeqCst), [])
            .into_iter()
            .chain((1..=updates).map(|n| 1_000_000 + n))
            .chain((1..=overlays).map(|n| 2_000_000 + n))
            .collect()
    }
}

/// Builds `val_idx` in another process while a seeded workload drawn from
/// `kinds` runs, part of it during the first commit attempt. Returns whether
/// the index committed; when it did, every lookup matches a scan.
async fn index_under_workload(seed: u64, kinds: &[Op], rebuild: bool) -> bool {
    let dir = TempStrDir::default();
    let uri = dir.as_str();
    let mut table = indexed_table(uri, 12).await;
    if rebuild {
        build_index(&mut table, "val", "val_idx", false)
            .await
            .unwrap();
    }
    let workload = Arc::new(Workload::new(uri, seed, 12 * ROWS_PER_FRAGMENT));
    let before = workload.draw(kinds, 6);
    let during = workload.draw(kinds, 3);
    let racing = workload.clone();
    let (mut builder, _) =
        builder_racing(uri, Box::pin(async move { racing.run(during).await })).await;

    Box::pin(workload.run(before.clone())).await;
    match build_index(&mut builder, "val", "val_idx", rebuild).await {
        Ok(()) => {
            assert_lookups_match_scan(uri, workload.probes()).await;
            true
        }
        Err(Error::RetryableCommitConflict { .. }) => false,
        Err(error) => panic!("seed {seed} {before:?}: {error}"),
    }
}

/// Full rebuilds while other processes write, overlay, add columns and run
/// deferred compactions, before and during the commit: every rebuild commits.
#[tokio::test]
async fn test_rebuild_commits_under_writes_and_deferred_compactions() {
    use Op::*;
    let kinds = [
        Append,
        Delete,
        UpdateInPlace,
        Overlay,
        Compact,
        Compact,
        CompactFoldingOverlays,
        AddColumn,
    ];
    for seed in 0..16 {
        assert!(
            Box::pin(index_under_workload(seed, &kinds, true)).await,
            "seed {seed}: the rebuild retried"
        );
    }
}

/// With the remap job trimming records, a new index may retry, but whenever it
/// commits every lookup matches a scan.
#[tokio::test]
async fn test_index_under_the_remap_job_is_correct_or_retries() {
    use Op::*;
    let kinds = [
        Append,
        Delete,
        UpdateInPlace,
        Overlay,
        Compact,
        Compact,
        CompactFoldingOverlays,
        RemapAndTrim,
    ];
    let mut committed = 0;
    for seed in 0..16 {
        if Box::pin(index_under_workload(seed, &kinds, false)).await {
            committed += 1;
        }
    }
    assert!(committed >= 8, "only {committed} of 16 seeds committed");
}
