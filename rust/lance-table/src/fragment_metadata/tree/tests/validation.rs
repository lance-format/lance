// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::{DeletionFile, DeletionFileType};
use crate::fragment_metadata::root_path;
use object_store::ObjectStoreExt;

fn object_path(fixture: &Fixture, path: &str) -> Path {
    fixture
        .base
        .parts()
        .chain(Path::from(path).parts())
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Warmth {
    SharedLeafCache,
    ResidentInventory,
    /// The records a resident bootstrap keeps, from input with metadata size
    /// hints.
    HintedBootstrap,
}

/// A shared leaf cache admits a leaf on its second decode.
#[rstest]
#[case::shared_leaf_cache(Warmth::SharedLeafCache)]
#[case::resident_inventory(Warmth::ResidentInventory)]
#[case::hinted_bootstrap(Warmth::HintedBootstrap)]
#[tokio::test]
async fn validate_reads_every_object_that_caches_and_residency_answer_for(#[case] warmth: Warmth) {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let fixture = Fixture::new(64, config, policy).await;
    assert!(fixture.tree.height() >= 2);
    let expected: Vec<_> = (0..64).map(make_fragment).collect();
    let warm = match warmth {
        Warmth::SharedLeafCache => {
            let leaves = LanceCache::with_capacity(64 * 1024 * 1024);
            let first = reopen_tree(
                &fixture.tree,
                fixture.context().with_leaf_cache(leaves.clone()),
            )
            .await;
            let warm = reopen_tree(&fixture.tree, fixture.context().with_leaf_cache(leaves)).await;
            assert_eq!(first.materialize().await.unwrap(), expected);
            assert_eq!(warm.materialize().await.unwrap(), expected);
            warm
        }
        Warmth::ResidentInventory => fixture.tree.clone().hydrate().await.unwrap(),
        Warmth::HintedBootstrap => {
            let mut hinted = expected.clone();
            for fragment in &mut hinted {
                fragment.files[0].file_metadata_size_bytes = std::num::NonZero::new(512);
            }
            // Leaves have no slot for the hint yet, so the tree reports it as
            // unknown.
            FragmentTree::bootstrap_resident_snapshot(
                fixture.context(),
                fixture.config.clone(),
                hinted,
                1,
                policy,
            )
            .await
            .unwrap()
            .0
        }
    };
    fixture.io.incremental_stats();
    assert_eq!(warm.materialize().await.unwrap(), expected);
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);

    let Some(pb::fragment_tree::Root::RootUuid(uuid)) = &warm.descriptor.root else {
        panic!("expected an external root");
    };
    let node_paths = warm.node_paths().await.unwrap();
    assert!(node_paths.iter().any(|path| path.starts_with("_bt/node/")));
    let expected_gets: BTreeMap<_, _> = std::iter::once(root_path(uuid).unwrap())
        .chain(node_paths.iter().cloned())
        .map(|path| (object_path(&fixture, &path), 1))
        .collect();
    fixture.io.incremental_stats();
    warm.validate().await.unwrap();
    let mut gets = BTreeMap::new();
    for request in fixture.io.incremental_stats().requests {
        if request.method.starts_with("get") {
            *gets.entry(request.path).or_default() += 1;
        }
    }
    assert_eq!(gets, expected_gets);

    let leaf = node_paths
        .iter()
        .find(|path| path.starts_with("_bt/leaf/"))
        .unwrap();
    fixture
        .store
        .inner
        .delete(&object_path(&fixture, leaf))
        .await
        .unwrap();
    assert_eq!(warm.materialize().await.unwrap(), expected);
    let error = warm.validate().await.unwrap_err();
    assert!(error.is_not_found(), "{error:?}");
    assert!(error.to_string().contains(leaf.as_str()), "{error}");
}

#[derive(Clone, Copy, Debug)]
enum RootFrontier {
    /// An inline root under a descriptor whose frontier is above the root's.
    InlineRoot,
    /// An external root whose suffix holds a sequence below the interior's.
    SuffixWindow,
}

/// Interiors below the root were written before it, so the format bounds
/// their sequences by the root's frontier. An interior action inside the
/// suffix window claims to follow suffix actions committed after the root.
#[rstest]
#[case::inline_root(
    RootFrontier::InlineRoot,
    "sequence number 3 is duplicated or outside [1, 3)"
)]
#[case::suffix_window(
    RootFrontier::SuffixWindow,
    "sequence number 3 is duplicated or outside [1, 2)"
)]
#[tokio::test]
async fn reads_commits_and_validate_bound_interiors_by_the_root_frontier(
    #[case] root: RootFrontier,
    #[case] expected: &str,
) {
    let mut fixture =
        Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let left =
        write_leaf_interior(&fixture.tree, &[&[0], &[1]], vec![buffered_file(1, 0, 3)]).await;
    let right = write_leaf_interior(&fixture.tree, &[&[2], &[3]], Vec::new()).await;
    let descriptor = match root {
        RootFrontier::InlineRoot => {
            route_root_to(&mut fixture.tree, vec![left, right], Vec::new(), 3);
            let mut descriptor = fixture.tree.descriptor.as_ref().clone();
            descriptor.next_action_sequence = 4;
            descriptor
        }
        RootFrontier::SuffixWindow => {
            let root = pb::FragmentTreeRoot {
                children: vec![left, right],
                buffer: Vec::new(),
                next_action_sequence: 2,
            };
            let (uuid, _) = fixture.tree.store.write_root_base(&root).await.unwrap();
            pb::FragmentTree {
                root: Some(pb::fragment_tree::Root::RootUuid(uuid)),
                mutations_since_root: vec![buffered_file(1, 1, 2)],
                next_action_sequence: 4,
            }
        }
    };
    let tree = fixture.open(&descriptor, 1).await;

    let read = tree.materialize().await.err();
    let audit = tree.validate().await.err();
    let commit = tree
        .begin_commit(CommitMode::Bulk)
        .prepare(
            ValidatedCommit::fragment_actions(Vec::new()),
            SnapshotPolicy::default(),
        )
        .await
        .err();
    for error in [&read, &audit, &commit] {
        assert!(
            matches!(error, Some(Error::CorruptFile { .. }))
                && error
                    .as_ref()
                    .is_some_and(|error| error.to_string().contains(expected)),
            "materialize gives {read:?}, validate gives {audit:?}, a bulk commit gives {commit:?}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum SquashedHistory {
    StaleAction,
    CancellingDeltas,
}

/// Opening squashes the whole buffer of a root over leaves, keeping the
/// newest sequence of each fragment and the sum of its deltas.
#[rstest]
#[case::stale_action(SquashedHistory::StaleAction, "is at or below the watermark 2")]
#[case::cancelling_deltas(
    SquashedHistory::CancellingDeltas,
    "stores fragment, row, and visible deltas 1 1 1 but the record changed by 0 0 0"
)]
#[tokio::test]
async fn validate_rejects_history_that_opening_squashes_away(
    #[case] history: SquashedHistory,
    #[case] expected_error: &str,
) {
    let mut fixture =
        Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let fragments: Vec<_> = (0..4).map(make_fragment).collect();
    let leaf = fixture
        .tree
        .store
        .write_leaf(&fragments, 2)
        .await
        .unwrap()
        .child_ref;
    let tagged =
        |action_sequence: u64, action: pb::FragmentAction, delta: i64| pb::FragmentTreeMutation {
            action_sequence,
            action: Some(action),
            fragment_count_delta: delta,
            total_rows_delta: delta,
            visible_rows_delta: delta,
        };
    let buffer = match history {
        SquashedHistory::StaleAction => vec![
            tagged(2, action::clear_deletion_file(0), 0),
            tagged(3, action::clear_deletion_file(0), 0),
        ],
        SquashedHistory::CancellingDeltas => vec![
            tagged(3, action::upsert_fragment(&fragments[0]).unwrap(), 1),
            tagged(4, action::upsert_fragment(&fragments[0]).unwrap(), -1),
        ],
    };
    route_root_to(&mut fixture.tree, vec![leaf], buffer, 5);
    let tree = fixture.open(&fixture.tree.descriptor, 1).await;
    assert_eq!(tree.root_buffer_len(), 1);
    assert_eq!(tree.materialize().await.unwrap(), fragments);

    let error = tree.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains(expected_error), "{error}");
}

#[tokio::test]
async fn validate_replays_a_root_buffer_and_suffix_that_opening_squashes() {
    let checkpoint = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let suffix = SnapshotPolicy {
        max_suffix_bytes: 1024,
        ..checkpoint
    };
    let mut fixture = Fixture::new(4, FragmentTreeConfig::default(), checkpoint).await;
    assert_eq!(fixture.tree.height(), 1);
    let deletion_file = DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(1),
        base_id: None,
    };
    fixture
        .commit(
            vec![action::add_deletion_file(2, &deletion_file).unwrap()],
            checkpoint,
            false,
        )
        .await;
    assert_eq!(fixture.tree.root_buffer_len(), 1);
    fixture
        .commit(vec![action::clear_deletion_file(2)], suffix, false)
        .await;
    let Some(pb::fragment_tree::Root::RootUuid(uuid)) = &fixture.tree.descriptor.root else {
        panic!("expected an external root");
    };
    let root = fixture.tree.store.read_root_base(uuid).await.unwrap();
    let persisted: Vec<_> = root
        .buffer
        .iter()
        .chain(&fixture.tree.descriptor.mutations_since_root)
        .map(content::action_key)
        .collect();
    assert_eq!(persisted, [2, 2]);

    let reopened = fixture.open(&fixture.tree.descriptor, 3).await;
    assert_eq!(reopened.root_buffer_len(), 1);
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (0..4).map(make_fragment).collect::<Vec<_>>()
    );
    reopened.validate().await.unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Unallocated {
    StoredRecord,
    BufferedAction,
}

/// Only ids below `next_fragment_id` are allocated, so no read serves a
/// record or action that names a higher one, which the next append would
/// allocate again.
#[rstest]
#[case::stored(
    Unallocated::StoredRecord,
    "stores fragment 9 at or above the end 4 of its range"
)]
#[case::buffered(
    Unallocated::BufferedAction,
    "buffers actions for fragments 9 through 9, outside its range [0, 4)"
)]
#[tokio::test]
async fn reads_and_validate_reject_ids_at_or_above_next_fragment_id(
    #[case] placement: Unallocated,
    #[case] expected: &str,
) {
    let mut fixture =
        Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let unallocated = make_fragment(9);
    let children = match placement {
        Unallocated::StoredRecord => {
            // The second leaf's fence is below the frontier, so only its
            // contents name the unallocated id.
            let mut leaves = Vec::new();
            for ids in [vec![0, 1], vec![2, 3, 9]] {
                let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
                let written = fixture.tree.store.write_leaf(&fragments, 0).await.unwrap();
                leaves.push(written.child_ref);
            }
            leaves
        }
        Unallocated::BufferedAction => {
            let upsert = pb::FragmentTreeMutation {
                action_sequence: 1,
                action: Some(action::upsert_fragment(&unallocated).unwrap()),
                fragment_count_delta: 1,
                total_rows_delta: 1,
                visible_rows_delta: 1,
            };
            vec![write_leaf_interior(&fixture.tree, &[&[0, 1], &[2, 3]], vec![upsert]).await]
        }
    };
    route_root_to(&mut fixture.tree, children, Vec::new(), 2);
    let tree = Arc::new(fixture.open(&fixture.tree.descriptor, 1).await);

    let scanned = tree.clone().scan(0).try_collect::<Vec<_>>().await.err();
    let resolved = tree.resolve_fragment(9).await.err();
    let audited = tree.validate().await.err();
    for error in [&scanned, &resolved, &audited] {
        assert!(
            matches!(error, Some(Error::CorruptFile { .. }))
                && error
                    .as_ref()
                    .is_some_and(|error| error.to_string().contains(expected)),
            "scan gives {scanned:?}, resolve gives {resolved:?}, validate gives {audited:?}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum MisstatedCount {
    Fragments,
    PhysicalRows,
    VisibleRows,
}

#[rstest]
#[case::fragments(
    MisstatedCount::Fragments,
    "counts 5 fragments, 4 physical rows and 4 visible rows"
)]
#[case::physical_rows(
    MisstatedCount::PhysicalRows,
    "counts 4 fragments, 5 physical rows and 4 visible rows"
)]
#[case::visible_rows(
    MisstatedCount::VisibleRows,
    "counts 4 fragments, 4 physical rows and 3 visible rows"
)]
#[tokio::test]
async fn validate_rejects_counts_that_disagree_with_persisted_records(
    #[case] count: MisstatedCount,
    #[case] counted: &str,
) {
    let fixture = Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut tree = fixture.tree.clone();
    match count {
        MisstatedCount::Fragments => tree.total_fragments += 1,
        MisstatedCount::PhysicalRows => tree.total_rows += 1,
        MisstatedCount::VisibleRows => tree.visible_rows -= 1,
    }
    assert_eq!(
        tree.materialize().await.unwrap(),
        (0..4).map(make_fragment).collect::<Vec<_>>()
    );

    let error = tree.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    let expected = format!(
        "version 1 {counted}, but its persisted records hold 4 fragments, 4 physical rows \
         and 4 visible rows"
    );
    assert!(error.to_string().contains(&expected), "{error}");
}

#[derive(Clone, Copy, Debug)]
enum ResidentDifference {
    ChangedFile,
    EndsEarly,
    RunsLong,
}

/// The counts still match storage when the inventory ends early or runs long.
#[rstest]
#[case::changed_file(ResidentDifference::ChangedFile, 2)]
#[case::ends_early(ResidentDifference::EndsEarly, 3)]
#[case::runs_long(ResidentDifference::RunsLong, 4)]
#[tokio::test]
async fn validate_rejects_a_resident_inventory_that_differs_from_storage(
    #[case] difference: ResidentDifference,
    #[case] first_different_id: u64,
) {
    let fixture = Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut tree = fixture.tree.clone().hydrate().await.unwrap();
    let mut served: Vec<_> = (0..4).map(make_fragment).collect();
    match difference {
        ResidentDifference::ChangedFile => served[2].files[0] = make_replacement_data_file(2, 0),
        ResidentDifference::EndsEarly => {
            served.pop();
        }
        ResidentDifference::RunsLong => served.push(make_fragment(4)),
    }
    tree.resident = Some(Arc::new(served.clone()));
    assert_eq!(tree.materialize().await.unwrap(), served);

    let error = tree.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    let expected = format!(
        "version 1 serves fragment {first_different_id} differently from its persisted records"
    );
    assert!(error.to_string().contains(&expected), "{error}");
}

#[derive(Clone, Copy, Debug)]
enum Alias {
    SiblingLeaf,
    SiblingInterior,
    OwnSubtree,
}

#[rstest]
#[case::sibling_leaf(Alias::SiblingLeaf)]
#[case::sibling_interior(Alias::SiblingInterior)]
#[case::own_subtree(Alias::OwnSubtree)]
#[tokio::test]
async fn validate_rejects_a_node_reached_from_two_positions(#[case] alias: Alias) {
    let fixture = Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let (tree, expected) = match alias {
        Alias::SiblingLeaf => {
            let source = source_store(&fixture);
            let leaf = source
                .write_leaf(&[make_fragment(0)], 0)
                .await
                .unwrap()
                .child_ref;
            let expected = format!("Leaf {} fence 1 is above its first fragment 0", leaf.path);
            let tree = open_with_sibling_alias(&fixture, &source, leaf, 1).await;
            (tree, expected)
        }
        Alias::SiblingInterior => {
            let source = source_store(&fixture);
            let mut leaves = Vec::new();
            for id in 0..2 {
                let leaf = source.write_leaf(&[make_fragment(id)], 0).await.unwrap();
                leaves.push(leaf.child_ref);
            }
            let expected = format!("Invalid child reference at position 0: {:?}", leaves[0]);
            let interior = source.write_internal(leaves, Vec::new()).await.unwrap();
            let tree = open_with_sibling_alias(&fixture, &source, interior.child_ref, 2).await;
            (tree, expected)
        }
        Alias::OwnSubtree => open_with_own_subtree_alias(&fixture).await,
    };

    let error = tree.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains(&expected), "{error}");
}

fn source_store(fixture: &Fixture) -> NodeStore {
    NodeStore::new(
        fixture.store.clone(),
        Path::from("source"),
        fixture.scheduler.clone(),
    )
}

/// A point read follows only the reference that owns its id.
async fn open_with_sibling_alias(
    fixture: &Fixture,
    source: &NodeStore,
    node: pb::FragmentTreeChild,
    alias_fence: u64,
) -> FragmentTree {
    let alias = pb::FragmentTreeChild {
        base_id: Some(2),
        min_key: alias_fence,
        ..node.clone()
    };
    let owner = pb::FragmentTreeChild {
        base_id: Some(1),
        ..node
    };
    let bases = HashMap::from([
        (1, (fixture.store.clone(), source.base().clone())),
        (2, (fixture.store.clone(), source.base().clone())),
    ]);
    let descriptor = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children: vec![owner, alias],
            buffer: Vec::new(),
            next_action_sequence: 1,
        })),
        mutations_since_root: Vec::new(),
        next_action_sequence: 1,
    };
    let tree = FragmentTree::open_snapshot(
        fixture.context().with_foreign_bases(bases),
        &descriptor,
        1,
        fixture.config.clone(),
        fixture.tree.next_fragment_id(),
    )
    .await
    .unwrap();
    assert_eq!(tree.resolve_fragment(0).await.unwrap().unwrap().id, 0);
    tree
}

/// The self reference copies the other child's counts, and the interior
/// buffers removals of that child's fragments. The reference differs from the
/// interior's contents in height alone.
async fn open_with_own_subtree_alias(fixture: &Fixture) -> (FragmentTree, String) {
    let path = "_bt/node/own-subtree.node".to_string();
    let sibling = write_leaf_interior(&fixture.tree, &[&[2], &[3]], Vec::new()).await;
    let removals = (2..4)
        .map(|id| pb::FragmentTreeMutation {
            action_sequence: id - 1,
            action: Some(action::remove_fragment(id)),
            fragment_count_delta: -1,
            total_rows_delta: -1,
            visible_rows_delta: -1,
        })
        .collect();
    let own = pb::FragmentTreeChild {
        path: path.clone(),
        min_key: 0,
        ..sibling.clone()
    };
    let mut interior = pb::FragmentTreeNode {
        children: vec![own, sibling],
        buffer: removals,
    };
    // The reference names the object that holds it, so its size counts itself.
    while interior.children[0].object_size != interior.encoded_len() as u64 {
        interior.children[0].object_size = interior.encoded_len() as u64;
    }
    fixture
        .store
        .put(&object_path(fixture, &path), &interior.encode_to_vec())
        .await
        .unwrap();
    let reference = node::internal_ref(
        path.clone(),
        &interior.children,
        &interior.buffer,
        interior.children[0].object_size,
    )
    .unwrap();
    let descriptor = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children: vec![reference],
            buffer: Vec::new(),
            next_action_sequence: 3,
        })),
        mutations_since_root: Vec::new(),
        next_action_sequence: 3,
    };
    let tree = fixture.open(&descriptor, 1).await;
    let expected = format!("Internal node {path} contents differ from its parent reference");
    (tree, expected)
}

/// Replay converts buffered files without validating them.
#[tokio::test]
async fn validate_rejects_a_buffered_file_that_fails_file_validation() {
    let fixture = Fixture::new(2, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut file = make_backfill_data_file(0, 0);
    file.fields = Arc::from([2, 3]);
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    root.buffer.push(stored(
        root.next_action_sequence,
        action::add_data_file(0, &file),
    ));
    root.next_action_sequence += 1;
    snapshot.next_action_sequence = root.next_action_sequence;
    let tree = fixture.open(&snapshot, 1).await;
    let mut expected: Vec<_> = (0..2).map(make_fragment).collect();
    expected[0].files.push(file);
    assert_eq!(tree.materialize().await.unwrap(), expected);

    let error = tree.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("contained fewer column_indices than fields"),
        "{error}"
    );
}

#[derive(Clone, Copy, Debug)]
enum AbsentTarget {
    Record,
    File,
}

/// A stored edit of state its subtree does not hold is corrupt metadata on
/// every read path, validate included.
#[rstest]
#[case::record(AbsentTarget::Record, "frag_id=2")]
#[case::file(AbsentTarget::File, "missing.lance")]
#[tokio::test]
async fn stored_edit_of_absent_state_is_reported_as_corrupt(
    #[case] target: AbsentTarget,
    #[case] named: &str,
) {
    let mut fixture =
        Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let edit = match target {
        AbsentTarget::Record => buffered_file(2, 0, 1),
        AbsentTarget::File => stored(
            1,
            action::replace_data_file(1, "missing.lance", &make_replacement_data_file(1, 0)),
        ),
    };
    let interior = write_leaf_interior(&fixture.tree, &[&[0, 1], &[3]], vec![edit]).await;
    route_root_to(&mut fixture.tree, vec![interior], Vec::new(), 2);
    let tree = fixture.open(&fixture.tree.descriptor, 1).await;

    let read = tree.materialize().await.err();
    let audit = tree.validate().await.err();
    for error in [&read, &audit] {
        assert!(
            matches!(error, Some(Error::CorruptFile { .. }))
                && error
                    .as_ref()
                    .is_some_and(|error| error.to_string().contains(named)),
            "materialize gives {read:?}, validate gives {audit:?}"
        );
    }
}

/// Mutations may be serialized in any order, so a suffix that lists a
/// removal before the append it undoes still opens.
#[tokio::test]
async fn suffix_totals_do_not_depend_on_serialization_order() {
    let fixture = Fixture::new(1, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &fixture.tree.descriptor.root else {
        panic!("expected inline root");
    };
    let next = root.next_action_sequence;
    let (uuid, _) = fixture.tree.store.write_root_base(root).await.unwrap();
    let mutation = |action_sequence: u64, action: pb::FragmentAction, count: i64, rows: i64| {
        pb::FragmentTreeMutation {
            action_sequence,
            action: Some(action),
            fragment_count_delta: count,
            total_rows_delta: rows,
            visible_rows_delta: rows,
        }
    };
    let snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::RootUuid(uuid)),
        mutations_since_root: vec![
            mutation(next + 1, action::remove_fragment(1), -1, -5),
            mutation(
                next,
                action::upsert_fragment(&make_fragment(1).with_physical_rows(5)).unwrap(),
                1,
                5,
            ),
        ],
        next_action_sequence: next + 2,
    };
    let tree =
        FragmentTree::open_snapshot(fixture.context(), &snapshot, 2, fixture.config.clone(), 2)
            .await
            .unwrap();
    assert_eq!(tree.count_rows(), 1);
    assert_eq!(tree.materialize().await.unwrap(), vec![make_fragment(0)]);
}
