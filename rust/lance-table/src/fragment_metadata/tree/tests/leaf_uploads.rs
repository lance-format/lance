// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::*;
use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as OSResult,
};

const FRAGMENTS: u64 = 2048;

fn small_leaves() -> FragmentTreeConfig {
    FragmentTreeConfig {
        max_node_bytes: 4 * 1024,
        max_leaf_bytes: 16 * 1024,
        semantic_buffer_bytes: 2 * 1024,
        ..FragmentTreeConfig::default()
    }
}

/// Holds each leaf PUT for a delay chosen by its issue order and counts how
/// many overlap. A PUT counts as completed only after the inner store wrote it.
/// Also counts node and root PUTs that start while a leaf PUT is in flight.
#[derive(Clone, Debug)]
struct LeafPutProbe(Arc<LeafPutCounts>);

#[derive(Debug)]
struct LeafPutCounts {
    hold: fn(usize) -> Duration,
    issued: AtomicUsize,
    completed: AtomicUsize,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    parents: AtomicUsize,
    parents_during_leaf_puts: AtomicUsize,
}

impl LeafPutCounts {
    fn note_parent_put(&self, location: &Path) {
        if !location.as_ref().contains("_bt/") {
            return;
        }
        self.parents.fetch_add(1, Ordering::SeqCst);
        if self.in_flight.load(Ordering::SeqCst) > 0 {
            self.parents_during_leaf_puts.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl LeafPutProbe {
    fn holding(hold: fn(usize) -> Duration) -> Self {
        Self(Arc::new(LeafPutCounts {
            hold,
            issued: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            parents: AtomicUsize::new(0),
            parents_during_leaf_puts: AtomicUsize::new(0),
        }))
    }

    fn wrap(&self, store: &ObjectStore) -> Arc<ObjectStore> {
        let mut store = store.clone();
        store.apply_wrapper(self);
        Arc::new(store)
    }

    fn peak(&self) -> usize {
        self.0.peak.load(Ordering::SeqCst)
    }

    /// Checked when a write call returns: every leaf it issued is stored.
    fn assert_every_put_landed(&self) {
        let issued = self.0.issued.load(Ordering::SeqCst);
        assert!(issued > 0, "the write must issue leaf PUTs");
        assert_eq!(self.0.completed.load(Ordering::SeqCst), issued);
        assert_eq!(self.0.in_flight.load(Ordering::SeqCst), 0);
    }

    /// A parent may name a leaf only after that leaf's PUT has completed.
    fn assert_no_parent_put_overlapped_a_leaf_put(&self) {
        assert!(
            self.0.parents.load(Ordering::SeqCst) > 0,
            "the write must issue node or root PUTs"
        );
        assert_eq!(
            self.0.parents_during_leaf_puts.load(Ordering::SeqCst),
            0,
            "a node or root PUT started while a leaf PUT was in flight"
        );
    }
}

impl WrappingObjectStore for LeafPutProbe {
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }

    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OSObjectStore>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(LeafPutStore {
            target: original,
            probe: self.clone(),
        })
    }
}

#[derive(Debug)]
struct LeafPutStore {
    target: Arc<dyn OSObjectStore>,
    probe: LeafPutProbe,
}

impl fmt::Display for LeafPutStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LeafPutStore({})", self.target)
    }
}

#[async_trait]
impl OSObjectStore for LeafPutStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        let counts = &self.probe.0;
        if !location.as_ref().contains("_bt/leaf/") {
            counts.note_parent_put(location);
            return self.target.put_opts(location, bytes, opts).await;
        }
        let order = counts.issued.fetch_add(1, Ordering::SeqCst);
        let in_flight = counts.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        counts.peak.fetch_max(in_flight, Ordering::SeqCst);
        tokio::time::sleep((counts.hold)(order)).await;
        let written = self.target.put_opts(location, bytes, opts).await;
        counts.completed.fetch_add(1, Ordering::SeqCst);
        counts.in_flight.fetch_sub(1, Ordering::SeqCst);
        written
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.probe.0.note_parent_put(location);
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.target.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.target.copy_opts(from, to, opts).await
    }
}

/// Every path that writes several leaves in one call.
#[derive(Clone, Copy, Debug)]
enum LeafWrite {
    Bootstrap,
    Commit(LeafCommit),
}

#[derive(Clone, Copy, Debug)]
enum LeafCommit {
    BulkMerge,
    BulkSplitOfOneRange,
    BufferedAppendIntoOneChild,
    BufferedSiblingDrains,
}

/// A height-two fixture and the fewest keys any of its leaves holds.
async fn height_two_fixture() -> (Fixture, u64) {
    let fixture = Fixture::new(FRAGMENTS, small_leaves(), SnapshotPolicy::default()).await;
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.height >= 2, "{shape:?}");
    assert!(shape.leaf_keys.len() >= 16, "{shape:?}");
    let smallest_leaf = shape.leaf_keys.iter().copied().min().unwrap();
    (fixture, smallest_leaf)
}

fn widened(ids: std::ops::Range<u64>, files: u32) -> Vec<Fragment> {
    ids.map(|id| make_fragment_with_files(id, files)).collect()
}

/// The upserts `commit` makes, the fragments they leave, and whether they
/// commit in bulk.
fn leaf_commit(
    commit: LeafCommit,
    smallest_leaf: u64,
) -> (Vec<pb::FragmentAction>, Vec<Fragment>, bool) {
    let (changed, bulk) = match commit {
        LeafCommit::BulkMerge => (widened(0..FRAGMENTS, 2), true),
        // Ids below the smallest leaf's key count all sit in the first leaf,
        // and widening them splits that one range several times.
        LeafCommit::BulkSplitOfOneRange => (widened(0..smallest_leaf, 16), true),
        LeafCommit::BufferedAppendIntoOneChild => (
            (FRAGMENTS..FRAGMENTS + FRAGMENTS / 4)
                .map(make_fragment)
                .collect(),
            false,
        ),
        LeafCommit::BufferedSiblingDrains => (widened(0..FRAGMENTS, 4), false),
    };
    let actions = changed.iter().map(action::upsert_fragment).collect();
    let mut expected: BTreeMap<u64, Fragment> =
        (0..FRAGMENTS).map(|id| (id, make_fragment(id))).collect();
    expected.extend(changed.into_iter().map(|fragment| (fragment.id, fragment)));
    (actions, expected.into_values().collect(), bulk)
}

#[rstest]
#[case::bootstrap(LeafWrite::Bootstrap)]
#[case::bulk_merge(LeafWrite::Commit(LeafCommit::BulkMerge))]
#[case::bulk_split_of_one_range(LeafWrite::Commit(LeafCommit::BulkSplitOfOneRange))]
#[case::buffered_append_into_one_child(LeafWrite::Commit(LeafCommit::BufferedAppendIntoOneChild))]
#[case::buffered_sibling_drains(LeafWrite::Commit(LeafCommit::BufferedSiblingDrains))]
#[tokio::test]
async fn leaf_puts_overlap_within_the_store_window(#[case] write: LeafWrite) {
    let policy = SnapshotPolicy::default();
    let (mut fixture, smallest_leaf) = height_two_fixture().await;
    let probe = LeafPutProbe::holding(|_| Duration::from_millis(5));
    let store = probe.wrap(&fixture.store);
    let expected = match write {
        LeafWrite::Bootstrap => {
            let mut fragments: Vec<_> = (0..FRAGMENTS).map(make_fragment).collect();
            let (_, snapshot, _) = FragmentTree::bootstrap_snapshot(
                store,
                fixture.base.clone(),
                fixture.scheduler.clone(),
                Arc::new(LanceCache::with_capacity(0)),
                fixture.config.clone(),
                &mut fragments,
                1,
                policy,
            )
            .await
            .unwrap();
            probe.assert_every_put_landed();
            fixture.snapshot = snapshot;
            fragments
        }
        LeafWrite::Commit(commit) => {
            let (actions, expected, bulk) = leaf_commit(commit, smallest_leaf);
            fixture.tree = fixture.tree.with_object_store(store);
            fixture.commit(actions, policy, bulk).await;
            probe.assert_every_put_landed();
            assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
            fixture.tree.verify_watermarks().await.unwrap();
            expected
        }
    };
    assert!(
        probe.peak() >= 2,
        "leaf PUTs must overlap: {}",
        probe.peak()
    );
    assert!(
        probe.peak() <= MAX_CONCURRENT_LEAF_DRAINS,
        "leaf PUTs must stay within the store window: {}",
        probe.peak()
    );
    // Sibling drains may write one child's parent while another child's
    // leaves are in flight, since that parent never names those leaves.
    if !matches!(write, LeafWrite::Commit(LeafCommit::BufferedSiblingDrains)) {
        probe.assert_no_parent_put_overlapped_a_leaf_put();
    }
    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    assert_eq!(reopened.materialize().await.unwrap(), expected);
}

#[tokio::test]
async fn bulk_keeps_key_order_when_later_leaf_puts_finish_first() {
    let policy = SnapshotPolicy::default();
    let (mut fixture, smallest_leaf) = height_two_fixture().await;
    let probe = LeafPutProbe::holding(|order| Duration::from_millis(16 - order.min(15) as u64));
    let (actions, expected, bulk) = leaf_commit(LeafCommit::BulkMerge, smallest_leaf);
    fixture.tree = fixture.tree.with_object_store(probe.wrap(&fixture.store));
    fixture.commit(actions, policy, bulk).await;
    probe.assert_every_put_landed();
    probe.assert_no_parent_put_overlapped_a_leaf_put();
    assert!(
        probe.peak() >= 2,
        "leaf PUTs must overlap: {}",
        probe.peak()
    );
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.leaf_keys.len() >= 16, "{shape:?}");
    fixture.tree.verify_watermarks().await.unwrap();
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    let reopened = fixture.open(&fixture.snapshot, 2).await;
    assert_eq!(reopened.materialize().await.unwrap(), expected);
}

#[rstest]
#[case::failed_put(FailWhen::Before, 2)]
#[case::lost_put_response(FailWhen::After, 2)]
#[case::failed_put_past_window(FailWhen::Before, 10)]
#[case::lost_put_response_past_window(FailWhen::After, 10)]
#[tokio::test]
async fn bulk_leaf_put_failure_keeps_the_previous_version(
    #[case] when: FailWhen,
    #[case] nth: usize,
) {
    let policy = SnapshotPolicy::default();
    let (mut fixture, smallest_leaf) = height_two_fixture().await;
    let original = fixture.tree.materialize().await.unwrap();
    let previous = fixture.snapshot.clone();
    let (actions, expected, _) = leaf_commit(LeafCommit::BulkMerge, smallest_leaf);
    let ids: Vec<_> = actions.iter().filter_map(action::target_frag_id).collect();
    let touched = fixture.tree.resolve_touched_for_bulk(&ids).await.unwrap();
    let failpoint = FailpointController::default();
    failpoint.arm(Failpoint {
        on: FailOn::Put,
        when,
        path_contains: "_bt/leaf/".into(),
        nth,
    });
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&failpoint);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let error = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions.clone()),
            &touched,
            &previous,
            policy,
            true,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IO { .. }), "{error}");
    assert!(error.to_string().contains("failpoint"), "{error}");
    assert!(failpoint.tripped());
    assert_eq!(fixture.tree.version(), 1);
    assert_eq!(fixture.tree.root_buffer_len(), 0);
    assert_eq!(fixture.tree.materialize().await.unwrap(), original);

    failpoint.disarm();
    let (snapshot, _) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions),
            &touched,
            &previous,
            policy,
            true,
        )
        .await
        .unwrap();
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    assert_eq!(
        fixture
            .open(&snapshot, 2)
            .await
            .materialize()
            .await
            .unwrap(),
        expected
    );
    assert_eq!(
        fixture
            .open(&previous, 1)
            .await
            .materialize()
            .await
            .unwrap(),
        original
    );
}
