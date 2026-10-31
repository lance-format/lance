// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
use crate::format::{DeletionFile, DeletionFileType};
use crate::fragment_metadata::root_path;
use object_store::ObjectStoreExt;

#[rstest]
#[case::different_bases(false)]
#[case::independent_memory_stores(true)]
#[tokio::test]
async fn traversal_uses_resolved_owners(#[case] independent_stores: bool) {
    let mut fixture =
        Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut children = Vec::new();
    let mut bases = HashMap::new();
    for id in 0..2 {
        let (object_store, base) = if independent_stores {
            (Arc::new(ObjectStore::memory()), Path::default())
        } else {
            (fixture.store.clone(), Path::from(format!("source-{id}")))
        };
        let source = NodeStore::new(
            object_store.clone(),
            base.clone(),
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing()),
        );
        let mut child = source
            .write_leaf(&[make_fragment(id)], 0)
            .await
            .unwrap()
            .child_ref;
        let bytes = object_store
            .inner
            .get(&Path::from(format!("{base}/{}", child.path)))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        child.path = "_bt/leaf/shared-name.lance".into();
        object_store
            .put(&Path::from(format!("{base}/{}", child.path)), &bytes)
            .await
            .unwrap();
        child.base_id = Some(id as u32 + 1);
        bases.insert(id as u32 + 1, (object_store, base));
        children.push(child);
    }
    let mut local = fixture
        .tree
        .store
        .write_leaf(&[make_fragment(2)], 0)
        .await
        .unwrap()
        .child_ref;
    local.base_id = Some(4);
    bases.insert(4, (fixture.store.clone(), fixture.base.clone()));
    children.push(local);
    crate::fragment_metadata::validation::children(&children, 1, 0).unwrap();
    fixture.tree = reopen_tree(
        &fixture.tree,
        fixture.context().with_foreign_bases(bases.clone()),
    )
    .await;
    fixture.tree.children = children.clone();
    assert_eq!(fixture.tree.node_paths().await.unwrap().len(), 3);
    // Ownership follows the resolved location, so a same name leaf under
    // another base is not this owner's, and the base id that names the
    // fixture's own dataset lists only the local leaf.
    for child in &children {
        let owner = child.base_id.unwrap();
        let (store, base) = &bases[&owner];
        assert_eq!(
            fixture
                .tree
                .object_paths_owned_by(store, base)
                .await
                .unwrap(),
            vec![child.path.clone()],
            "owner {owner}"
        );
    }

    bases.insert(3, bases[&1].clone());
    fixture.tree = reopen_tree(&fixture.tree, fixture.context().with_foreign_bases(bases)).await;
    fixture.tree.children = children.clone();
    let mut alias = children[0].clone();
    alias.base_id = Some(3);
    fixture.tree.children.push(alias);
    let error = fixture.tree.node_paths().await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("repeats resolved node"),
        "{error}"
    );
}

/// Leaves at one relative path under two bases hold different records, so
/// the summary of one is never reused for the other.
#[rstest]
#[case::different_bases(false)]
#[case::independent_memory_stores(true)]
#[tokio::test]
async fn summaries_keep_same_name_leaves_of_other_bases_apart(#[case] independent_stores: bool) {
    let fixture = Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut leaves = LeafSummaries::default();
    for (base_id, files) in [(1, 1), (2, 3)] {
        let (object_store, base) = if independent_stores {
            (Arc::new(ObjectStore::memory()), Path::default())
        } else {
            (
                fixture.store.clone(),
                Path::from(format!("source-{base_id}")),
            )
        };
        let source = NodeStore::new(
            object_store.clone(),
            base.clone(),
            ScanScheduler::new(object_store.clone(), SchedulerConfig::default_for_testing()),
        );
        let mut child = source
            .write_leaf(&[make_fragment_with_files(0, files)], 0)
            .await
            .unwrap()
            .child_ref;
        let bytes = object_store
            .inner
            .get(&Path::from(format!("{base}/{}", child.path)))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        child.path = "_bt/leaf/shared-name.lance".into();
        object_store
            .put(&Path::from(format!("{base}/{}", child.path)), &bytes)
            .await
            .unwrap();
        child.base_id = Some(base_id);
        let bases = HashMap::from([(base_id, (object_store, base))]);
        let mut tree =
            reopen_tree(&fixture.tree, fixture.context().with_foreign_bases(bases)).await;
        tree.children = vec![child];
        tree.next_fragment_id = 1;
        let summary = tree.summary(&mut leaves).await.unwrap();
        assert_eq!(
            summary,
            ManifestSummary::of(&tree.materialize().await.unwrap())
        );
        assert_eq!(summary.total_data_files, u64::from(files));
    }
}

#[tokio::test]
async fn traversal_inherits_owner_through_interior_nodes() {
    let mut fixture =
        Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut roots = Vec::new();
    let mut bases = HashMap::new();
    for owner in 1..=2 {
        let base = Path::from(format!("source-{owner}"));
        let source = NodeStore::new(
            fixture.store.clone(),
            base.clone(),
            fixture.scheduler.clone(),
        );
        let mut leaves = Vec::new();
        for offset in 0..2 {
            leaves.push(
                source
                    .write_leaf(&[make_fragment((owner - 1) * 2 + offset)], 0)
                    .await
                    .unwrap()
                    .child_ref,
            );
        }
        let mut child = source
            .write_internal(leaves, Vec::new())
            .await
            .unwrap()
            .child_ref;
        let bytes = fixture
            .store
            .inner
            .get(&Path::from(format!("{base}/{}", child.path)))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        child.path = "_bt/node/shared.node".into();
        fixture
            .store
            .put(&Path::from(format!("{base}/{}", child.path)), &bytes)
            .await
            .unwrap();
        child.base_id = Some(owner as u32);
        bases.insert(owner as u32, (fixture.store.clone(), base));
        roots.push(child);
    }
    fixture.tree = reopen_tree(
        &fixture.tree,
        fixture.context().with_foreign_bases(bases.clone()),
    )
    .await;
    fixture.tree.children = roots;
    fixture.tree.next_fragment_id = 4;
    assert_eq!(fixture.tree.node_paths().await.unwrap().len(), 6);
    assert!(
        fixture
            .tree
            .object_paths_owned_by(&fixture.store, &fixture.base)
            .await
            .unwrap()
            .is_empty()
    );
    for child in &fixture.tree.children {
        let node = fixture.tree.store.read_internal(child).await.unwrap();
        assert!(
            node.children
                .iter()
                .all(|leaf| leaf.base_id == child.base_id)
        );
        // Each owner's view holds its own interior node and the leaves under
        // it, even though both interior nodes share one relative path.
        let (store, base) = &bases[&child.base_id.unwrap()];
        let mut owned = vec![child.path.clone()];
        owned.extend(node.children.iter().map(|leaf| leaf.path.clone()));
        owned.sort();
        assert_eq!(
            fixture
                .tree
                .object_paths_owned_by(store, base)
                .await
                .unwrap(),
            owned
        );
    }
}

/// A branch root folds the source's suffix, gives every inherited reference
/// the source's base id and follows the branch's publication policy. The
/// branch owns only its root, and through the source's location it keeps
/// every source node and never the source's root.
#[rstest]
#[case::inline(SnapshotPolicy::default())]
#[case::external(SnapshotPolicy { inline_root_bytes: 0, ..SnapshotPolicy::default() })]
#[tokio::test]
async fn branch_snapshot_folds_the_suffix_and_owns_only_its_root(
    #[case] branch_policy: SnapshotPolicy,
) {
    let checkpoint = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let mut source = Fixture::new(64, FragmentTreeConfig::default(), checkpoint).await;
    source
        .commit(vec![action::remove_fragment(0)], checkpoint, false)
        .await;
    let deletion = DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Bitmap,
        num_deleted_rows: Some(1),
        base_id: None,
    };
    source
        .commit(
            vec![
                action::add_data_file(1, &make_backfill_data_file(1, 9)),
                action::add_deletion_file(2, &deletion).unwrap(),
            ],
            SnapshotPolicy {
                max_suffix_bytes: 1024,
                ..checkpoint
            },
            false,
        )
        .await;
    assert!(!source.tree.descriptor.mutations_since_root.is_empty());
    let opened = source.open(&source.tree.descriptor, 3).await;
    let mut expected = opened.materialize().await.unwrap();
    for fragment in &mut expected {
        for file in fragment.referenced_lance_files_mut() {
            file.base_id.get_or_insert(7);
        }
        if let Some(deletion) = &mut fragment.deletion_file {
            deletion.base_id.get_or_insert(7);
        }
    }
    let branch_base = Path::from("branch");
    let branch_context = TreeContext::new(
        source.store.clone(),
        branch_base.clone(),
        source.scheduler.clone(),
    );

    let prepared = opened
        .clone()
        .branch_snapshot(branch_context.clone(), 7, branch_policy)
        .await
        .unwrap();
    assert!(prepared.resident_fragments().is_none());
    let descriptor = prepared.descriptor().as_ref();
    assert!(descriptor.mutations_since_root.is_empty());
    // A resident source stays resident, stamped as a branch read stamps it.
    let resident = opened
        .clone()
        .hydrate()
        .await
        .unwrap()
        .branch_snapshot(branch_context.clone(), 7, branch_policy)
        .await
        .unwrap();
    assert_eq!(
        resident
            .resident_fragments()
            .map(|fragments| fragments.as_slice()),
        Some(expected.as_slice())
    );
    let external = matches!(descriptor.root, Some(pb::fragment_tree::Root::RootUuid(_)));
    assert_eq!(external, branch_policy.inline_root_bytes == 0);
    let branch = FragmentTree::open_snapshot(
        branch_context.with_foreign_bases(HashMap::from([(
            7,
            (source.store.clone(), source.base.clone()),
        )])),
        descriptor,
        opened.version(),
        source.config.clone(),
        opened.next_fragment_id(),
    )
    .await
    .unwrap();
    assert!(branch.children.iter().all(|child| child.base_id == Some(7)));
    assert_eq!(branch.materialize().await.unwrap(), expected);
    branch.validate().await.unwrap();
    let own = branch
        .object_paths_owned_by(&source.store, &branch_base)
        .await
        .unwrap();
    assert_eq!(own.len(), usize::from(external));
    assert!(
        own.iter().all(|path| path.starts_with("_bt/root/")),
        "{own:?}"
    );
    assert_eq!(
        branch
            .object_paths_owned_by(&source.store, &source.base)
            .await
            .unwrap(),
        opened.node_paths().await.unwrap(),
    );
}

#[tokio::test]
async fn branch_snapshot_applies_the_root_hard_capacity() {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        ..SnapshotPolicy::default()
    };
    let fixture = Fixture::new(64, FragmentTreeConfig::default(), policy).await;
    let root_bytes = fixture.tree.compacted_root().encoded_len() as u64;
    // The source root fits exactly. Stamping inherited base ids grows it.
    let config = FragmentTreeConfig {
        max_node_bytes: root_bytes,
        max_leaf_bytes: root_bytes,
        semantic_buffer_bytes: root_bytes,
        hard_capacity_bytes: root_bytes,
        ..FragmentTreeConfig::default()
    };
    let source = FragmentTree::open_snapshot(
        fixture.context(),
        &fixture.tree.descriptor,
        1,
        config,
        fixture.tree.next_fragment_id(),
    )
    .await
    .unwrap();
    let branch = TreeContext::new(
        fixture.store.clone(),
        Path::from("branch"),
        fixture.scheduler.clone(),
    );

    let Err(error) = source.branch_snapshot(branch, 7, policy).await else {
        panic!("a branch root over hard_capacity_bytes must be refused");
    };
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("hard_capacity_bytes"), "{error}");
}

/// A snapshot lists its external root only at its own dataset, and never a
/// root that a later commit attempt prepared from it.
#[rstest]
#[case::inline(SnapshotPolicy::default())]
#[case::external(SnapshotPolicy { inline_root_bytes: 0, max_suffix_bytes: 0 })]
#[tokio::test]
async fn owned_objects_list_the_published_root_only_at_its_own_dataset(
    #[case] policy: SnapshotPolicy,
) {
    let fixture = Fixture::new(64, FragmentTreeConfig::default(), policy).await;
    let published = fixture.open(&fixture.tree.descriptor, 1).await;
    let mut expected = published.node_paths().await.unwrap();
    if let Some(pb::fragment_tree::Root::RootUuid(uuid)) = &fixture.tree.descriptor.root {
        expected.push(root_path(uuid).unwrap());
    }
    expected.sort();
    let (prepared, _) = published
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::remove_fragment(0)]),
            SnapshotPolicy {
                inline_root_bytes: 0,
                max_suffix_bytes: 0,
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        prepared.descriptor.root,
        Some(pb::fragment_tree::Root::RootUuid(_))
    ));

    assert_eq!(
        published
            .object_paths_owned_by(&fixture.store, &fixture.base)
            .await
            .unwrap(),
        expected,
    );
    assert!(
        published
            .object_paths_owned_by(&fixture.store, &Path::from("elsewhere"))
            .await
            .unwrap()
            .is_empty()
    );
}

#[rstest]
#[case::empty(0)]
#[case::nonempty(3)]
#[tokio::test]
async fn inventory_retention_is_explicit_including_empty_trees(#[case] count: u64) {
    let policy = SnapshotPolicy::default();
    let fixture = Fixture::new(count, FragmentTreeConfig::default(), policy).await;
    let expected: Vec<_> = (0..count).map(make_fragment).collect();
    assert!(fixture.tree.resident_fragments().is_none());
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    assert!(fixture.tree.resident_fragments().is_none());
    let eager = fixture.tree.clone().hydrate().await.unwrap();
    assert_eq!(eager.resident_fragments().unwrap().as_ref(), &expected);
    assert!(fixture.tree.resident_fragments().is_none());
    let (owned, _) = FragmentTree::bootstrap_resident_snapshot(
        fixture.context(),
        fixture.config.clone(),
        expected.clone(),
        1,
        policy,
    )
    .await
    .unwrap();
    assert_eq!(owned.resident_fragments().unwrap().as_ref(), &expected);
    assert_eq!(
        fixture
            .open(&fixture.tree.descriptor, 1)
            .await
            .resident_fragments(),
        None
    );
}

#[rstest]
#[case::incremental(CommitMode::Incremental)]
#[case::bulk(CommitMode::Bulk)]
#[tokio::test]
async fn resident_prepare_matches_reopened_records(#[case] mode: CommitMode) {
    let policy = SnapshotPolicy::default();
    let fixture = Fixture::new(3, FragmentTreeConfig::default(), policy).await;
    let source = fixture
        .tree
        .clone()
        .hydrate()
        .await
        .unwrap()
        .into_published(1, fixture.config.clone(), fixture.context())
        .unwrap();
    let original = source.resident_fragments().unwrap().clone();
    fixture.io.incremental_stats();
    let mut attempt = source.begin_commit(mode);
    assert_eq!(
        attempt
            .resolve_touched(&[0, 1, 3])
            .await
            .unwrap()
            .fragments
            .len(),
        2
    );
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    assert!(!attempt.has_validation_reads());
    let mut file = make_backfill_data_file(0, 1);
    file.file_metadata_size_bytes = std::num::NonZero::new(512);
    let second_file = make_backfill_data_file(2, 1);
    let mut added = make_fragment(3);
    added.physical_rows = Some(5);
    let mut overlay_file = make_backfill_data_file(3, 1);
    overlay_file.file_metadata_size_bytes = std::num::NonZero::new(256);
    added.overlays.push(DataOverlayFile {
        data_file: overlay_file,
        coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        committed_version: 1,
    });
    let (prepared, _) = attempt
        .prepare(
            ValidatedCommit::fragment_actions(vec![
                action::add_data_file(0, &file),
                action::add_data_file(2, &second_file),
                action::remove_fragment(1),
                action::upsert_fragment(&added).unwrap(),
            ]),
            policy,
        )
        .await
        .unwrap();
    // Leaves have no slot for the hint yet, so the tree reports it as unknown.
    file.file_metadata_size_bytes = None;
    added.overlays[0].data_file.file_metadata_size_bytes = None;
    let mut expected = original.as_ref().clone();
    expected[0].files.push(file);
    expected[2].files.push(second_file);
    expected.remove(1);
    expected.push(added);
    let retained = prepared.resident_fragments().unwrap();
    assert_eq!(retained.as_ref(), &expected);
    prepared.validate().await.unwrap();
    for fragment in retained.iter() {
        for (file, first) in fragment.files.iter().zip(&retained[0].files) {
            assert!(Arc::ptr_eq(&file.fields, &first.fields));
            assert!(Arc::ptr_eq(&file.column_indices, &first.column_indices));
        }
    }
    let reopened = FragmentTree::open_snapshot(
        fixture.context(),
        prepared.descriptor(),
        2,
        fixture.config.clone(),
        prepared.next_fragment_id(),
    )
    .await
    .unwrap();
    assert_eq!(reopened.materialize().await.unwrap(), expected);
    assert!(Arc::ptr_eq(source.resident_fragments().unwrap(), &original));
    assert_eq!(source.count_fragments(), 3);
    assert_eq!(source.version(), 1);
    assert_eq!(prepared.version(), 1);
    let published = prepared
        .into_published(2, fixture.config.clone(), fixture.context())
        .unwrap();
    assert_eq!(published.version(), 2);
    assert_eq!(published.resident_fragments().unwrap().as_ref(), &expected);
}

#[tokio::test]
async fn eager_noop_shares_inventory_and_lazy_result_discards_it() {
    let policy = SnapshotPolicy::default();
    let fixture = Fixture::new(3, FragmentTreeConfig::default(), policy).await;
    let source = fixture.tree.clone().hydrate().await.unwrap();
    let original = source.resident_fragments().unwrap().clone();
    fixture.io.incremental_stats();
    let (retained, _) = source
        .begin_commit(CommitMode::Bulk)
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(
        &original,
        retained.resident_fragments().unwrap()
    ));
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    let mut attempt = source
        .begin_commit(CommitMode::Bulk)
        .with_inventory_retention(InventoryRetention::Discard);
    attempt.resolve_all().await.unwrap();
    assert!(!attempt.has_validation_reads());
    let (unloaded, _) = attempt
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .unwrap();
    assert!(unloaded.resident_fragments().is_none());
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    assert!(Arc::ptr_eq(&original, source.resident_fragments().unwrap()));
}

#[tokio::test]
async fn bulk_retention_starts_only_when_validation_reads_tree_objects() {
    let fixture = Fixture::new(3, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut attempt = fixture.tree.begin_commit(CommitMode::Bulk);
    assert!(!attempt.has_validation_reads());
    attempt.resolve_touched(&[]).await.unwrap();
    attempt.resolve_touched(&[3]).await.unwrap();
    assert!(!attempt.has_validation_reads());
    attempt.resolve_touched(&[0]).await.unwrap();
    assert!(attempt.has_validation_reads());
    assert!(!fixture.tree.store.has_validation_reads());
    let empty = Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut attempt = empty.tree.begin_commit(CommitMode::Bulk);
    attempt.resolve_all().await.unwrap();
    assert!(!attempt.has_validation_reads());
}

#[tokio::test]
async fn retained_checkpoint_and_reopened_snapshot_replay_the_same_records() {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let fixture = Fixture::new(3, FragmentTreeConfig::default(), policy).await;
    let source = fixture.tree.clone().hydrate().await.unwrap();
    let (prepared, stats) = source
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::add_data_file(
                0,
                &make_backfill_data_file(0, 1),
            )]),
            policy,
        )
        .await
        .unwrap();
    assert_eq!(stats.checkpoints, 1);
    let reopened = fixture.open(&prepared.descriptor, 2).await;
    let next = ValidatedCommit::fragment_actions(vec![action::add_data_file(
        0,
        &make_backfill_data_file(0, 2),
    )]);
    let suffix_policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: usize::MAX,
    };
    let (retained, _) = prepared
        .begin_commit(CommitMode::Incremental)
        .prepare(next.clone(), suffix_policy)
        .await
        .unwrap();
    let (fresh, _) = reopened
        .begin_commit(CommitMode::Incremental)
        .prepare(next, suffix_policy)
        .await
        .unwrap();
    assert_eq!(
        retained.materialize().await.unwrap(),
        fresh.materialize().await.unwrap()
    );
}
