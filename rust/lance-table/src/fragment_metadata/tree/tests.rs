// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::ManifestSummary;
use crate::fragment_metadata::support::{
    make_backfill_data_file, make_fragment, make_fragment_with_files, make_replacement_data_file,
};
use crate::fragment_metadata::{CommitMode, TreeCommit, TreeContext, action};
use lance_core::cache::LanceCache;
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::failpoint::{FailOn, FailWhen, Failpoint, FailpointController};
use lance_io::utils::tracking_store::IOTracker;
use object_store::ObjectStoreExt;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use std::collections::HashMap;

mod interior_cache;
mod leaf_uploads;
mod ownership;
mod sequences;
mod validation;

/// Singleton repair must preserve ancestor-buffered actions without advancing
/// the leaf watermark past them.
#[tokio::test]
async fn singleton_repair_leaves_the_callers_pending_actions_unmaterialized() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig {
        max_node_bytes: 4096,
        max_leaf_bytes: 64 * 1024,
        semantic_buffer_bytes: 2048,
        ..FragmentTreeConfig::default()
    };
    // Draining `narrow` leaves a singleton. A pending bucket under `wide`
    // crosses the drain threshold only after the two directories merge.
    const NARROW_LEAF: u64 = 64;
    const WIDE_LEAF: u64 = 224;
    let total = 2 * NARROW_LEAF + 3 * WIDE_LEAF;
    let mut fixture = Fixture::new(total, config.clone(), policy).await;
    let mut expected: BTreeMap<u64, Fragment> = fixture
        .tree
        .materialize()
        .await
        .unwrap()
        .into_iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();
    let watermark = fixture.tree.next_action_sequence - 1;
    let mut leaves = Vec::new();
    let mut start = 0;
    for count in [NARROW_LEAF, NARROW_LEAF, WIDE_LEAF, WIDE_LEAF, WIDE_LEAF] {
        let fragments: Vec<Fragment> = (start..start + count)
            .map(|id| expected[&id].clone())
            .collect();
        let written = fixture
            .tree
            .store
            .write_leaves(&fragments, None, watermark, &config)
            .await
            .unwrap();
        assert_eq!(written.len(), 1, "each range must encode as one leaf");
        leaves.push(written.into_iter().next().unwrap().child_ref);
        start += count;
    }
    let (narrow_leaves, wide_leaves) = leaves.split_at(2);
    for leaf in narrow_leaves {
        assert!(
            node::is_underflow(leaf, &config),
            "narrow leaves must underflow so they coalesce: {leaf:?}"
        );
    }
    let mut pending = Vec::new();
    let mut next_action_sequence = fixture.tree.next_action_sequence;
    let mut pending_files = BTreeMap::new();
    let merged_children: Vec<_> = std::iter::once(narrow_leaves[0].clone())
        .chain(wide_leaves.iter().cloned())
        .collect();
    for leaf in wide_leaves {
        let mut bucket = Vec::new();
        let mut id = leaf.min_key;
        // A small margin covers the routing bytes the coalesced leaf changes.
        while node::internal_logical_bytes(&[], &bucket)
            < flush::amortization_gate(flush::fair_share(&merged_children, &config), leaf) + 64
        {
            let file = make_backfill_data_file(id, 1);
            bucket.push(pb::FragmentTreeMutation {
                action_sequence: next_action_sequence,
                action: Some(action::add_data_file(id, &file)),
                ..Default::default()
            });
            pending_files.insert(id, file);
            next_action_sequence += 1;
            id += 1;
        }
        assert!(
            node::internal_logical_bytes(&[], &bucket)
                < flush::amortization_gate(flush::fair_share(wide_leaves, &config), leaf),
            "the bucket must sit below the gate of its own node and above the merged node's"
        );
        pending.extend(bucket);
    }
    let narrow = fixture
        .tree
        .store
        .write_internal(narrow_leaves.to_vec(), Vec::new())
        .await
        .unwrap()
        .child_ref;
    let wide = fixture
        .tree
        .store
        .write_internal(wide_leaves.to_vec(), pending.clone())
        .await
        .unwrap()
        .child_ref;
    assert!(
        !node::internal_overflows(wide_leaves, &pending, &config)
            && node::buffer_pressured(&pending, &config),
        "the wide node must be pressured yet unable to drain any child"
    );
    fixture.tree.children = vec![narrow, wide];
    fixture.tree.buffer.clear();
    fixture.tree.buffer_index.take();
    fixture.tree.next_action_sequence = next_action_sequence;
    fixture.tree.store.next_action_sequence = next_action_sequence;
    fixture.tree.descriptor = Arc::new(pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(
            fixture.tree.compacted_root(),
        )),
        mutations_since_root: Vec::new(),
        next_action_sequence,
    });
    fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;

    // One residual action under `wide` stays in the root because it is below
    // the root's gate for that child. Enough actions under `narrow` pressure
    // the root, drain into `narrow`, rewrite both of its leaves, and leave
    // them small enough to coalesce into one.
    let residual = 2 * NARROW_LEAF + 2 * WIDE_LEAF + WIDE_LEAF / 2;
    let mut commit_files = vec![(residual, make_backfill_data_file(residual, 2))];
    let mut actions = vec![action::add_data_file(residual, &commit_files[0].1)];
    while node::internal_logical_bytes(&[], &pending_for(&actions)) < config.semantic_buffer_bytes {
        let id = actions.len() as u64 - 1;
        let file = make_backfill_data_file(id, 2);
        actions.push(action::add_data_file(id, &file));
        commit_files.push((id, file));
    }
    let narrow_actions = actions.len() as u64 - 1;
    assert!(
        narrow_actions < 2 * NARROW_LEAF,
        "the pressuring actions must all route under the narrow node"
    );
    for (id, file) in pending_files
        .iter()
        .chain(commit_files.iter().map(|(id, file)| (id, file)))
    {
        expected.get_mut(id).unwrap().files.push(file.clone());
    }
    let stats = fixture.commit(actions, policy, false).await;
    assert_eq!(
        stats.messages_materialized, narrow_actions,
        "only the actions drained into the narrow node may reach a leaf"
    );
    assert!(stats.merges >= 1, "the singleton must have been repaired");

    let reader = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reader.validate().await.unwrap();
    let shape = reader.shape_report().await.unwrap();
    assert_eq!(
        shape.root_buffer_len + shape.node_buffer_lens.iter().sum::<u64>(),
        pending.len() as u64 + 1,
        "the wide node's pending actions and the residual must still be buffered"
    );
    let resolved = reader.resolve_fragment(residual).await.unwrap().unwrap();
    assert_eq!(resolved, expected[&residual]);
    assert_eq!(
        reader.materialize().await.unwrap(),
        expected.into_values().collect::<Vec<_>>()
    );
}

/// No reader opens a root whose only child is an interior with one child, so
/// a root that cannot adopt such a child must not publish it.
#[rstest]
#[case::incremental(CommitMode::Incremental)]
#[case::bulk(CommitMode::Bulk)]
#[tokio::test]
async fn root_shrink_that_overflows_never_publishes_a_single_child_interior(
    #[case] mode: CommitMode,
) {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let config = FragmentTreeConfig::new(16 * 1024, 16).with_max_leaf_bytes(64 * 1024);
    let mut fixture = Fixture::new(0, config.clone(), policy).await;
    let mut expected: Vec<Fragment> = (0..5).map(make_fragment).collect();
    let mut next_action_sequence = 1;
    let mut buffered_files = |id: u64, bytes: u64| {
        let mut buffer = Vec::new();
        while node::internal_logical_bytes(&[], &buffer) < bytes {
            let file = make_backfill_data_file(id, buffer.len() as u32);
            buffer.push(stored(
                next_action_sequence,
                action::add_data_file(id, &file),
            ));
            expected[id as usize].files.push(file);
            next_action_sequence += 1;
        }
        buffer
    };
    let interior_buffer = buffered_files(2, 8 * 1024);
    let root_buffer = buffered_files(3, 9 * 1024);
    let buffered_actions = (interior_buffer.len() + root_buffer.len()) as u64;
    let nodes = &fixture.tree.store;
    let mut leaves = Vec::new();
    for id in 0..4 {
        let leaf = nodes
            .write_leaf(&[make_fragment(id)], 0)
            .await
            .unwrap()
            .child_ref;
        assert!(node::is_underflow(&leaf, &config), "{leaf:?}");
        leaves.push(leaf);
    }
    assert!(
        leaves.iter().map(|leaf| leaf.object_size).sum::<u64>() <= config.leaf_coalesce_ceiling(),
        "the four leaves must coalesce into one"
    );
    let narrow = nodes
        .write_internal(leaves[..2].to_vec(), Vec::new())
        .await
        .unwrap()
        .child_ref;
    let wide = nodes
        .write_internal(leaves[2..].to_vec(), interior_buffer.clone())
        .await
        .unwrap()
        .child_ref;
    assert!(
        node::is_underflow(&narrow, &config)
            && narrow.object_size + wide.object_size <= config.coalesce_ceiling(),
        "the two interiors must coalesce into one"
    );
    let children = vec![narrow, wide];
    assert!(
        !node::internal_overflows(&children, &root_buffer, &config)
            && !node::buffer_pressured(&root_buffer, &config),
        "the root must not drain"
    );
    // Any leaf reference stands in for the coalesced leaf.
    assert!(
        node::internal_overflows(
            &leaves[..1],
            &[root_buffer.clone(), interior_buffer].concat(),
            &config
        ),
        "the root must not fit the coalesced interior's buffer beside its own"
    );
    let snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children,
            buffer: root_buffer,
            next_action_sequence,
        })),
        mutations_since_root: Vec::new(),
        next_action_sequence,
    };
    fixture.tree = FragmentTree::open_snapshot(fixture.context(), &snapshot, 1, config, 4)
        .await
        .unwrap();

    fixture.io.incremental_stats();
    let (tree, stats) = fixture
        .tree
        .begin_commit(mode)
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&expected[4]).unwrap()]),
            policy,
        )
        .await
        .unwrap();
    assert_eq!(
        stats.tree_write_bytes,
        fixture.io.stats().written_bytes,
        "{stats:?}"
    );
    // Only a bulk rewrite drains both buffers, so in incremental mode this
    // shows that the root collapse retried the commit in bulk.
    assert_eq!(
        stats.messages_materialized,
        buffered_actions + 1,
        "{stats:?}"
    );
    fixture.tree = tree;
    fixture.tree.validate().await.unwrap();
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    assert_eq!(reopened.materialize().await.unwrap(), expected);
    assert_eq!(reopened.height(), 1);
}

async fn reopen_tree(tree: &FragmentTree, context: TreeContext) -> FragmentTree {
    FragmentTree::open_snapshot(
        context,
        &tree.descriptor,
        tree.version(),
        tree.config.clone(),
        tree.next_fragment_id(),
    )
    .await
    .unwrap()
}

fn commit_mode(bulk: bool) -> CommitMode {
    if bulk {
        CommitMode::Bulk
    } else {
        CommitMode::Incremental
    }
}

fn pending_for(actions: &[pb::FragmentAction]) -> Vec<pb::FragmentTreeMutation> {
    actions
        .iter()
        .map(|action| pb::FragmentTreeMutation {
            action: Some(action.clone()),
            ..Default::default()
        })
        .collect()
}

struct Fixture {
    tree: FragmentTree,
    config: FragmentTreeConfig,
    store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    io: IOTracker,
}

#[rstest]
#[case::zero_sequence("zero_sequence", "next_action_sequence")]
#[case::zero_object_size("zero_object_size", "child reference")]
#[case::unallocated_root_target(
    "root_target",
    "Root buffer names fragment 2, at or above next_fragment_id 2"
)]
#[case::unallocated_suffix_target(
    "suffix_target",
    "mutations_since_root at version 1 names fragment 2, at or above next_fragment_id 2"
)]
#[tokio::test]
async fn snapshot_rejects_invalid_root_and_child_metadata(
    #[case] corruption: &str,
    #[case] expected: &str,
) {
    let fixture = Fixture::new(2, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    let unallocated = stored(root.next_action_sequence, action::clear_deletion_file(2));
    match corruption {
        "zero_sequence" => root.next_action_sequence = 0,
        "zero_object_size" => root.children[0].object_size = 0,
        "root_target" => {
            root.buffer.push(unallocated);
            root.next_action_sequence += 1;
            snapshot.next_action_sequence = root.next_action_sequence;
        }
        _ => {
            let (uuid, _) = fixture.tree.store.write_root_base(root).await.unwrap();
            snapshot.root = Some(pb::fragment_tree::Root::RootUuid(uuid));
            snapshot.mutations_since_root = vec![unallocated];
            snapshot.next_action_sequence += 1;
        }
    }
    let result = FragmentTree::open_snapshot(
        TreeContext::new(fixture.store, fixture.base, fixture.scheduler),
        &snapshot,
        1,
        fixture.config.clone(),
        fixture.tree.next_fragment_id(),
    )
    .await;
    let error = result
        .err()
        .expect("invalid snapshot must be rejected at open");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains(expected), "{error}");
}

impl Fixture {
    async fn new(count: u64, config: FragmentTreeConfig, policy: SnapshotPolicy) -> Self {
        let io = IOTracker::default();
        let (store, base) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory://",
            &ObjectStoreParams {
                object_store_wrapper: Some(Arc::new(io.clone())),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::default_for_testing());
        let (tree, _) = FragmentTree::bootstrap_snapshot(
            TreeContext::new(store.clone(), base.clone(), scheduler.clone()),
            config.clone(),
            &mut (0..count).map(make_fragment).collect::<Vec<_>>(),
            1,
            policy,
        )
        .await
        .unwrap();
        Self {
            tree,
            config,
            store,
            base,
            scheduler,
            io,
        }
    }

    fn context(&self) -> TreeContext {
        TreeContext::new(
            self.store.clone(),
            self.base.clone(),
            self.scheduler.clone(),
        )
    }

    async fn open(&self, snapshot: &pb::FragmentTree, version: u64) -> FragmentTree {
        FragmentTree::open_snapshot(
            self.context(),
            snapshot,
            version,
            self.config.clone(),
            self.tree.next_fragment_id(),
        )
        .await
        .unwrap()
    }

    async fn commit(
        &mut self,
        actions: Vec<pb::FragmentAction>,
        policy: SnapshotPolicy,
        bulk: bool,
    ) -> CommitStats {
        let ids: Vec<_> = actions.iter().filter_map(action::target_frag_id).collect();
        let mut transaction = self.tree.begin_commit(commit_mode(bulk));
        transaction.resolve_touched(&ids).await.unwrap();
        let (tree, stats) = transaction
            .prepare(ValidatedCommit::fragment_actions(actions), policy)
            .await
            .unwrap();
        self.tree = tree;
        stats
    }
}

#[rstest]
#[case::fragments("Fragments")]
#[case::physical_rows("physical rows")]
#[case::visible_rows("visible rows")]
#[tokio::test]
async fn snapshot_counts_are_checked_against_children_and_pending_changes(#[case] count: &str) {
    let checkpoint = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    // Removing under 1/16 of a leaf stays buffered, which this test needs.
    let mut fixture = Fixture::new(64, FragmentTreeConfig::default(), checkpoint).await;
    fixture
        .commit(vec![action::remove_fragment(0)], checkpoint, false)
        .await;
    assert_eq!(fixture.tree.root_buffer_len(), 1);
    fixture
        .commit(
            vec![action::remove_fragment(1)],
            SnapshotPolicy {
                max_suffix_bytes: 1024,
                ..checkpoint
            },
            false,
        )
        .await;
    assert_eq!(fixture.tree.descriptor.mutations_since_root.len(), 1);
    let reopened = fixture.open(&fixture.tree.descriptor, 3).await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (2..64).map(make_fragment).collect::<Vec<_>>()
    );

    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    match count {
        "Fragments" => snapshot.mutations_since_root[0].fragment_count_delta = -999,
        "physical rows" => snapshot.mutations_since_root[0].total_rows_delta = -999,
        "visible rows" => snapshot.mutations_since_root[0].visible_rows_delta = -999,
        _ => unreachable!(),
    }
    let error = FragmentTree::open_snapshot(
        TreeContext::new(fixture.store, fixture.base, fixture.scheduler),
        &snapshot,
        3,
        fixture.config.clone(),
        fixture.tree.next_fragment_id(),
    )
    .await
    .err()
    .expect("snapshot totals must agree with the tree");
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains(count), "{error}");
}

#[rstest]
#[case::buffered(17, false)]
#[case::bulk(29, true)]
#[tokio::test]
async fn mutations_survive_routing_growth_and_historical_reads(
    #[case] seed: u64,
    #[case] bulk: bool,
) {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 512,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let mut fixture = Fixture::new(48, config, policy).await;
    let initial = fixture.tree.descriptor.clone();
    let mut expected: BTreeMap<_, _> = (0..48).map(|id| (id, make_fragment(id))).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut maximum_height = fixture.tree.height();
    for round in 0..12 {
        let mut changes = Vec::new();
        for offset in 0..4 {
            let id = 48 + round * 4 + offset;
            let fragment = make_fragment(id);
            changes.push(action::upsert_fragment(&fragment).unwrap());
            expected.insert(id, fragment);
        }
        let id = rng.random_range(0..48);
        let mut fragment = make_fragment(id);
        fragment.files[0].path = format!("round-{round}-{id}.lance");
        changes.push(action::upsert_fragment(&fragment).unwrap());
        expected.insert(id, fragment);
        fixture.commit(changes, policy, bulk).await;
        maximum_height = maximum_height.max(fixture.tree.height());
        assert_eq!(
            fixture.tree.materialize().await.unwrap(),
            expected.values().cloned().collect::<Vec<_>>()
        );
        fixture.tree.validate().await.unwrap();
        let reopened = fixture.open(&fixture.tree.descriptor, round + 2).await;
        assert_eq!(
            reopened.materialize().await.unwrap(),
            fixture.tree.materialize().await.unwrap()
        );
    }
    assert!(
        maximum_height >= 2,
        "the fixture must cross an interior level"
    );
    let remove = expected
        .keys()
        .copied()
        .filter(|id| *id != 7)
        .map(action::remove_fragment)
        .collect();
    fixture.commit(remove, policy, true).await;
    assert_eq!(fixture.tree.height(), 1);
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        vec![expected[&7].clone()]
    );
    let historical = fixture.open(&initial, 1).await;
    assert_eq!(
        historical.materialize().await.unwrap(),
        (0..48).map(make_fragment).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn bulk_coalescing_preserves_changes_across_three_leaves() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default();
    let mut fixture = Fixture::new(12, config, policy).await;
    let mut expected: Vec<_> = (0..12).map(make_fragment).collect();
    let mut children = Vec::new();
    for fragments in expected.chunks(4) {
        let written = fixture.tree.store.write_leaf(fragments, 0).await.unwrap();
        assert!(written.io_bytes <= fixture.config.leaf_merge_floor());
        children.push(written.child_ref);
    }
    fixture.tree.children = children;
    Arc::make_mut(&mut fixture.tree.descriptor).root = Some(pb::fragment_tree::Root::InlineRoot(
        fixture.tree.compacted_root(),
    ));
    let mut actions = Vec::new();
    for fragment in expected.iter_mut().step_by(4) {
        fragment.files.push(make_backfill_data_file(fragment.id, 2));
        actions.push(action::upsert_fragment(fragment).unwrap());
    }

    let stats = fixture.commit(actions, policy, true).await;
    assert_eq!(
        (stats.merges, stats.flushes, stats.messages_materialized),
        (2, 3, 3)
    );
    assert_eq!(fixture.tree.root_child_count(), 1);
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    fixture.tree.validate().await.unwrap();
    assert_eq!(
        fixture
            .open(&fixture.tree.descriptor, 2)
            .await
            .materialize()
            .await
            .unwrap(),
        expected
    );
}

#[tokio::test]
async fn bulk_reuses_validation_leaf_reads_and_coalesces_across_parents() {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let mut fixture = Fixture::new(64, config, policy).await;
    let leaves = fixture.tree.shape_report().await.unwrap().leaf_keys.len();
    assert!(fixture.tree.height() >= 2);
    fixture.io.incremental_stats();
    fixture
        .commit((1..64).map(action::remove_fragment).collect(), policy, true)
        .await;
    let measured = fixture.io.incremental_stats();
    let reads = measured
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with("get_") && request.path.as_ref().contains("_bt/leaf/")
        })
        .count();
    assert_eq!(
        reads, leaves,
        "validation and materialization must share leaf reads: {measured:?}"
    );
    assert_eq!(fixture.tree.root_child_count(), 1);
    assert_eq!(
        measured.write_iops, 2,
        "one replacement leaf and one root base"
    );
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        vec![make_fragment(0)]
    );
}

/// A bulk commit on a height-2 tree whose leaves all sit above the merge floor,
/// so materialization reads only the leaves that own a replaced data file.
async fn bulk_fixture_without_coalescing() -> (Fixture, SnapshotPolicy, Vec<u64>) {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let merge_floor = config.leaf_merge_floor();
    let fixture = Fixture::new(64, config, policy).await;
    let report = fixture.tree.shape_report().await.unwrap();
    assert!(fixture.tree.height() >= 2);
    assert!(
        report.leaf_bytes.iter().all(|bytes| *bytes > merge_floor),
        "{report:?}"
    );
    (fixture, policy, report.leaf_keys)
}

/// Commit one replaced data file per id in bulk and check the result. Returns
/// the leaf GETs of the commit itself, excluding the verifying read.
async fn commit_replacements(
    fixture: &mut Fixture,
    policy: SnapshotPolicy,
    ids: &[u64],
    transaction: TreeCommit,
) -> usize {
    let actions = ids
        .iter()
        .map(|id| {
            action::replace_data_file(
                *id,
                &make_fragment(*id).files[0].path,
                &make_replacement_data_file(*id, 0),
            )
        })
        .collect();
    let (tree, _) = transaction
        .prepare(ValidatedCommit::fragment_actions(actions), policy)
        .await
        .unwrap();
    fixture.tree = tree;
    let commit_leaf_gets = leaf_gets(&fixture.io.incremental_stats());
    let expected: Vec<_> = (0..64)
        .map(|id| {
            let mut fragment = make_fragment(id);
            if ids.contains(&id) {
                fragment.files[0] = make_replacement_data_file(id, 0);
            }
            fragment
        })
        .collect();
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    commit_leaf_gets
}

#[rstest]
#[case::same_leaf("same_leaf", 1)]
#[case::disjoint_leaves("disjoint_leaves", 2)]
#[tokio::test]
async fn repeated_bulk_resolves_share_retained_leaf_reads(
    #[case] placement: &str,
    #[case] covering_leaves: usize,
) {
    let (mut fixture, policy, leaf_keys) = bulk_fixture_without_coalescing().await;
    assert!(leaf_keys[0] >= 2, "{leaf_keys:?}");
    let second = match placement {
        "same_leaf" => 1,
        "disjoint_leaves" => leaf_keys[0],
        _ => unreachable!(),
    };
    fixture.io.incremental_stats();
    // Native validation resolves the touched set, then probes an index witness
    // through a second resolve before the same bulk materialization.
    let mut transaction = fixture.tree.begin_commit(CommitMode::Bulk);
    transaction.resolve_touched(&[0]).await.unwrap();
    transaction.resolve_touched(&[second]).await.unwrap();
    assert_eq!(leaf_gets(&fixture.io.incremental_stats()), covering_leaves);
    assert_eq!(
        commit_replacements(&mut fixture, policy, &[0, second], transaction).await,
        0,
        "materialization must reread every leaf from retention"
    );
}

#[rstest]
#[case::before_prepare(true)]
#[case::during_prepare(false)]
#[tokio::test]
async fn bulk_commit_reads_each_changed_leaf_once(#[case] is_resolved: bool) {
    let (mut fixture, policy, leaf_keys) = bulk_fixture_without_coalescing().await;
    let ids = [0, leaf_keys[0]];
    fixture.io.incremental_stats();
    let mut transaction = fixture.tree.begin_commit(CommitMode::Bulk);
    if is_resolved {
        transaction.resolve_touched(&ids).await.unwrap();
    }
    let validation_leaf_gets = leaf_gets(&fixture.io.incremental_stats());
    let commit_leaf_gets = commit_replacements(&mut fixture, policy, &ids, transaction).await;
    assert_eq!(validation_leaf_gets + commit_leaf_gets, ids.len());
}

#[rstest]
#[case::compacted(false, true)]
#[case::buffered(true, true)]
#[case::near_capacity(false, false)]
#[tokio::test]
async fn root_contracts_multiple_children_with_room_to_grow(
    #[case] buffered: bool,
    #[case] should_contract: bool,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(4, FragmentTreeConfig::default(), policy).await;
    let mut leaves = Vec::new();
    for id in 0..4 {
        let written = fixture
            .tree
            .store
            .write_leaves(&[make_fragment(id)], None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    let mut expected: Vec<_> = (0..4).map(make_fragment).collect();
    let mut child_buffer = Vec::new();
    if buffered {
        let old_path = expected[0].files[0].path.clone();
        expected[0].files[0].path = "first-replacement.lance".into();
        child_buffer.push(pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::replace_data_file(
                0,
                &old_path,
                &expected[0].files[0],
            )),
            ..Default::default()
        });
        let old_path = expected[0].files[0].path.clone();
        expected[0].files[0].path = "second-replacement.lance".into();
        fixture.tree.buffer.push(pb::FragmentTreeMutation {
            action_sequence: 2,
            action: Some(action::replace_data_file(
                0,
                &old_path,
                &expected[0].files[0],
            )),
            ..Default::default()
        });
        fixture.tree.next_action_sequence = 3;
        fixture.tree.store.next_action_sequence = 3;
    }
    let left = fixture
        .tree
        .store
        .write_internal(leaves[..2].to_vec(), child_buffer)
        .await
        .unwrap();
    let right = fixture
        .tree
        .store
        .write_internal(leaves[2..].to_vec(), Vec::new())
        .await
        .unwrap();
    fixture.tree.children = vec![left.child_ref, right.child_ref];
    let collapsed_bytes = fixture
        .tree
        .children
        .iter()
        .map(|child| child.object_size)
        .sum::<u64>()
        + node::internal_logical_bytes(&[], &fixture.tree.buffer);
    fixture.tree.config.max_node_bytes = if should_contract {
        collapsed_bytes * 2
    } else {
        collapsed_bytes * 10 / 9
    };
    let mut historical = fixture.tree.descriptor.as_ref().clone();
    historical.next_action_sequence = fixture.tree.next_action_sequence;
    historical.root = Some(pb::fragment_tree::Root::InlineRoot(
        fixture.tree.compacted_root(),
    ));
    assert_eq!(fixture.tree.height(), 2);
    assert_eq!(
        fixture.tree.resolve_fragment(0).await.unwrap(),
        Some(expected[0].clone())
    );

    fixture.io.incremental_stats();
    let rewrite =
        rewrite::RewriteNodes::new(&fixture.tree.buffer, fixture.tree.next_action_sequence);
    fixture.tree.maybe_shrink_root(&rewrite).await.unwrap();
    let io = fixture.io.incremental_stats();
    assert_eq!(fixture.tree.height(), if should_contract { 1 } else { 2 });
    assert_eq!(io.write_iops, 0, "contraction must reuse the leaves");
    assert!(
        io.requests
            .iter()
            .all(|request| !request.path.as_ref().contains("_bt/leaf/"))
    );
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    assert_eq!(
        fixture.tree.resolve_fragment(0).await.unwrap(),
        Some(expected[0].clone())
    );
    let mut snapshot = historical.clone();
    snapshot.root = Some(pb::fragment_tree::Root::InlineRoot(
        fixture.tree.compacted_root(),
    ));
    let reopened = fixture.open(&snapshot, 1).await;
    reopened.validate().await.unwrap();
    assert_eq!(reopened.materialize().await.unwrap(), expected);
    let old = fixture.open(&historical, 1).await;
    assert_eq!(old.height(), 2);
    assert_eq!(old.materialize().await.unwrap(), expected);
}

/// A buffered commit on a tree whose root routes through interiors reads
/// nothing when the joined level could not fit the fanout limit. The child
/// references already record each interior's fanout.
#[tokio::test]
async fn small_commit_under_a_routed_root_reads_no_node() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_max_leaf_bytes(1);
    let mut fixture = Fixture::new(300, config, policy).await;
    assert!(fixture.tree.height() >= 3, "{}", fixture.tree.height());
    let actions = vec![action::upsert_fragment(&make_fragment(300)).unwrap()];
    let mut transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    transaction.resolve_touched(&[300]).await.unwrap();
    fixture.io.incremental_stats();
    let (tree, _) = transaction
        .prepare(ValidatedCommit::fragment_actions(actions), policy)
        .await
        .unwrap();
    fixture.tree = tree;
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (0..301).map(make_fragment).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn bootstrap_packs_compressed_batches_within_the_leaf_target() {
    let config = FragmentTreeConfig::default();
    let fixture = Fixture::new(0, config.clone(), SnapshotPolicy::default()).await;
    let fragments: Vec<_> = (5_000_000..5_000_216)
        .map(|id| make_fragment_with_files(id, 128))
        .collect();
    let encoded = fixture.tree.store.encode_leaf(&fragments).await.unwrap();
    assert!(node::leaf_logical_bytes(&fragments) > config.max_leaf_bytes * 2);
    assert!(encoded.len() as u64 <= config.max_leaf_bytes);
    fixture.io.incremental_stats();
    let store = fixture.tree.store.context().into_store(&config, 1);
    let (tree, stats) = FragmentTree::build(
        store,
        config.clone(),
        &fragments,
        1,
        SnapshotPolicy::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        stats.num_leaves, 1,
        "the complete encoding fits in one leaf"
    );
    assert_eq!(
        fixture.io.incremental_stats().write_iops,
        1,
        "do not publish candidate leaves"
    );
    assert!(
        tree.leaf_object_sizes()
            .iter()
            .all(|size| *size <= config.max_leaf_bytes)
    );
    assert_eq!(tree.materialize().await.unwrap(), fragments);
}

#[derive(Debug, Clone, Copy)]
enum ScanReadahead {
    /// Leaf reads in flight.
    Fixed(usize),
    Ramped,
}

impl ScanReadahead {
    fn scan(
        self,
        tree: Arc<FragmentTree>,
        lower_bound: u64,
    ) -> BoxStream<'static, Result<Fragment>> {
        match self {
            Self::Fixed(leaves) => {
                tree.scan_with_readahead(lower_bound, NonZeroUsize::new(leaves).unwrap())
            }
            Self::Ramped => tree.scan(lower_bound),
        }
    }
}

/// A height-two tree of 64 fragments whose first eight carry buffered
/// actions, and the fragments it must stream.
async fn deep_stream_fixture() -> (Fixture, Vec<Fragment>) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let mut fixture = Fixture::new(64, config, policy).await;
    let mut expected: Vec<_> = (0..64).map(make_fragment).collect();
    let mut actions = Vec::new();
    for fragment in expected.iter_mut().take(8) {
        let file = crate::fragment_metadata::support::make_backfill_data_file(fragment.id, 0);
        actions.push(action::add_data_file(fragment.id, &file));
        fragment.files.push(file);
    }
    actions.push(action::remove_fragment(63));
    actions.push(action::upsert_fragment(&make_fragment(64)).unwrap());
    expected[63] = make_fragment(64);
    fixture.commit(actions, policy, false).await;
    assert!(fixture.tree.height() >= 2);
    (fixture, expected)
}

/// The fixture's tree behind a store whose GETs wait `latency`, with a
/// tracker of the requests it completes.
fn delayed_tree(
    tree: FragmentTree,
    latency: std::time::Duration,
) -> (Arc<FragmentTree>, IOTracker) {
    let delayed = FailpointController::default();
    delayed.set_get_latency(latency);
    let io = IOTracker::default();
    let mut store = tree.store.object_store.as_ref().clone();
    store.apply_wrapper(&delayed);
    store.apply_wrapper(&io);
    (Arc::new(tree.with_object_store(Arc::new(store))), io)
}

/// Keys per leaf in walk order. The shape report lists leaves in stack order.
async fn leaf_keys_in_walk_order(tree: &FragmentTree) -> Vec<u64> {
    let mut keys = Vec::new();
    let mut stack: Vec<_> = tree.children.iter().rev().cloned().collect();
    while let Some(child) = stack.pop() {
        if child.height == 0 {
            keys.push(child.num_keys);
            continue;
        }
        let node = tree.store.read_internal(&child).await.unwrap();
        stack.extend(node.children.into_iter().rev());
    }
    keys
}

fn leaf_requests(stats: &lance_io::utils::tracking_store::IoStats) -> Vec<String> {
    stats
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with("get_") && request.path.as_ref().contains("_bt/leaf/")
        })
        .map(|request| request.path.to_string())
        .collect()
}

#[rstest]
#[case::serial(ScanReadahead::Fixed(1), 0)]
#[case::fixed_from_start(ScanReadahead::Fixed(4), 0)]
#[case::ramped_from_start(ScanReadahead::Ramped, 0)]
#[case::ramped_from_middle(ScanReadahead::Ramped, 32)]
#[case::fixed_from_middle(ScanReadahead::Fixed(4), 32)]
#[tokio::test]
async fn scan_honors_readahead_and_preserves_order(
    #[case] readahead: ScanReadahead,
    #[case] lower_bound: u64,
) {
    let (fixture, mut expected) = deep_stream_fixture().await;
    expected.retain(|fragment| fragment.id >= lower_bound);
    let leaf_count = fixture.tree.shape_report().await.unwrap().leaf_keys.len();
    let (tree, io) = delayed_tree(fixture.tree, std::time::Duration::from_millis(2));
    let actual: Vec<_> = readahead
        .scan(tree, lower_bound)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(actual, expected);
    let measured = io.incremental_stats();
    let leaf_reads = leaf_requests(&measured);
    let distinct: HashSet<_> = leaf_reads.iter().collect();
    assert_eq!(
        distinct.len(),
        leaf_reads.len(),
        "each leaf must be read once"
    );
    if lower_bound == 0 {
        assert_eq!(leaf_reads.len(), leaf_count, "every leaf must be read");
    } else {
        assert!(
            (3..leaf_count).contains(&leaf_reads.len()),
            "routing must skip the leaves below the bound and leave several to overlap: {} of {leaf_count}",
            leaf_reads.len()
        );
    }
    if matches!(readahead, ScanReadahead::Fixed(1)) {
        assert_eq!(measured.num_stages, measured.read_iops);
    } else {
        assert!(
            measured.num_stages < measured.read_iops,
            "deep leaf reads must overlap: {measured:?}"
        );
    }
}

#[tokio::test]
async fn resident_scan_from_a_bound_reads_nothing_and_matches_a_lazy_scan() {
    let (fixture, expected) = deep_stream_fixture().await;
    let resident = Arc::new(fixture.tree.clone().hydrate().await.unwrap());
    let lazy = Arc::new(fixture.tree);
    // Fragment 63 was removed, so the second bound falls in a hole.
    for lower_bound in [32, 63] {
        let scanned: Vec<_> = lazy.clone().scan(lower_bound).try_collect().await.unwrap();
        let from_bound: Vec<_> = expected
            .iter()
            .filter(|fragment| fragment.id >= lower_bound)
            .cloned()
            .collect();
        assert_eq!(scanned, from_bound, "lazy scan from {lower_bound}");
        fixture.io.incremental_stats();
        let answered: Vec<_> = resident
            .clone()
            .scan(lower_bound)
            .try_collect()
            .await
            .unwrap();
        assert_eq!(fixture.io.incremental_stats().read_iops, 0);
        assert_eq!(answered, scanned, "resident scan from {lower_bound}");
    }
}

/// A LIMIT scan that ends inside the first leaf must not pay for a second
/// leaf, so the ramped window admits more only after the first is drained.
#[rstest]
#[case::first_fragment("first_fragment")]
#[case::all_but_last_of_first_leaf("all_but_last")]
#[case::whole_first_leaf("whole_leaf")]
#[tokio::test]
async fn scan_reads_one_leaf_until_the_first_leaf_is_drained(#[case] taken: &str) {
    // No commit, so the first leaf keeps its bootstrap fill.
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let fixture = Fixture::new(64, config, SnapshotPolicy::default()).await;
    assert!(fixture.tree.height() >= 2);
    let expected: Vec<_> = (0..64).map(make_fragment).collect();
    let first_leaf = leaf_keys_in_walk_order(&fixture.tree).await[0] as usize;
    assert!(first_leaf > 2, "{first_leaf}");
    let taken = match taken {
        "first_fragment" => 1,
        "all_but_last" => first_leaf - 1,
        _ => first_leaf,
    };
    let latency = std::time::Duration::from_millis(2);
    let (tree, io) = delayed_tree(fixture.tree, latency);
    let actual: Vec<_> = tree.scan(0).take(taken).try_collect().await.unwrap();
    assert_eq!(actual, expected[..taken]);
    // A read admitted before the drop would complete within one latency.
    tokio::time::sleep(latency * 4).await;
    assert_eq!(leaf_requests(&io.incremental_stats()).len(), 1);
}

/// A routing error found while the window reads ahead must wait until the
/// fragments of every earlier leaf are yielded.
#[rstest]
#[case::fixed_1(ScanReadahead::Fixed(1))]
#[case::fixed_4(ScanReadahead::Fixed(4))]
#[case::ramped(ScanReadahead::Ramped)]
#[tokio::test]
async fn scan_yields_earlier_leaves_before_a_later_routing_error(#[case] readahead: ScanReadahead) {
    let mut fixture =
        Fixture::new(5, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let tree = &mut fixture.tree;
    let mut interiors = Vec::new();
    // The middle interior owns [8, 12) but fences a child at 20.
    for leaves in [
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
        vec![vec![8, 9], vec![20]],
        vec![vec![12, 13], vec![14, 15]],
    ] {
        let mut children = Vec::new();
        for ids in leaves {
            let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
            children.push(
                tree.store
                    .write_leaf(&fragments, 0)
                    .await
                    .unwrap()
                    .child_ref,
            );
        }
        interiors.push(
            tree.store
                .write_internal(children, Vec::new())
                .await
                .unwrap()
                .child_ref,
        );
    }
    tree.children = interiors;
    tree.next_fragment_id = 21;
    tree.buffer.clear();
    tree.buffer_index.take();
    let mut stream = readahead.scan(Arc::new(fixture.tree), 0);
    let mut yielded = Vec::new();
    let error = loop {
        match stream.try_next().await {
            Ok(Some(fragment)) => yielded.push(fragment.id),
            Ok(None) => panic!("the walk must fail at the middle interior"),
            Err(error) => break error,
        }
    };
    assert_eq!(yielded, (0..8).collect::<Vec<_>>());
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
    assert!(stream.try_next().await.unwrap().is_none());
}

/// Dropping a stream aborts the leaf reads its window admitted, so they are
/// never decoded or sighted by the leaf cache. A GET already handed to the
/// scheduler still completes, so each admitted read wastes one GET.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_scan_admits_no_unread_leaf() {
    let (mut fixture, _) = deep_stream_fixture().await;
    let leaf_keys = leaf_keys_in_walk_order(&fixture.tree).await;
    let leaf_count = leaf_keys.len();
    assert!(leaf_count >= 7, "{leaf_keys:?}");
    fixture.tree = reopen_tree(
        &fixture.tree,
        fixture
            .tree
            .store
            .context()
            .with_leaf_cache(LanceCache::with_capacity(16 * 1024 * 1024)),
    )
    .await;
    let latency = std::time::Duration::from_millis(20);
    let (tree, io) = delayed_tree(fixture.tree, latency);
    // Draining two leaves widens the window to four. Taking the first
    // fragment of the third admits up to three more leaves.
    const CONSUMED_LEAVES: usize = 3;
    const MAX_ADMITTED_UNREAD: usize = 3;
    let taken = (leaf_keys[0] + leaf_keys[1] + 1) as usize;
    let mut stream = tree.clone().scan(0);
    for _ in 0..taken {
        stream.try_next().await.unwrap().unwrap();
    }
    assert_eq!(
        leaf_requests(&io.incremental_stats()).len(),
        CONSUMED_LEAVES
    );
    // Let the admitted reads hand their GETs to the scheduler, so the drop
    // aborts reads in flight rather than tasks that never started.
    tokio::time::sleep(latency / 4).await;
    drop(stream);
    tokio::time::sleep(latency * 2).await;
    // Routing admits the leaves it reaches without an interior GET, so the
    // count depends on the layout but must be at least one.
    let wasted = leaf_requests(&io.incremental_stats()).len();
    assert!(
        (1..=MAX_ADMITTED_UNREAD).contains(&wasted),
        "{wasted} leaf GETs after drop, the window must have admitted unread leaves"
    );
    // A full walk admits every leaf sighted before it, so a second walk
    // reads exactly the leaves nobody sighted before the first.
    for _ in 0..2 {
        io.incremental_stats();
        tree.clone()
            .scan_with_readahead(0, NonZeroUsize::new(8).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
    }
    assert_eq!(
        leaf_requests(&io.incremental_stats()).len(),
        leaf_count - CONSUMED_LEAVES,
        "only the consumed leaves may have been sighted"
    );
}

/// Where the materialized tree keeps its committed actions, and what the
/// store has already seen of its nodes.
#[derive(Clone, Copy, Debug)]
enum MaterializeReads {
    /// Bulk commits drained every action into its leaf. No cache.
    Drained,
    /// Buffered commits left actions in interior buffers. No cache.
    InteriorActions,
    /// As `InteriorActions`, through a session cache that has read every
    /// node once, so interiors are cached and leaves only sighted.
    WarmCache,
    /// As `InteriorActions`, retained on scratch for a bulk commit.
    BulkValidation,
}

fn node_gets(stats: &lance_io::utils::tracking_store::IoStats, kind: &str) -> usize {
    stats
        .requests
        .iter()
        .filter(|request| request.path.as_ref().contains(kind) && request.method.contains("get"))
        .count()
}

#[rstest]
#[case::drained(MaterializeReads::Drained)]
#[case::interior_actions(MaterializeReads::InteriorActions)]
#[case::warm_cache(MaterializeReads::WarmCache)]
#[case::bulk_validation(MaterializeReads::BulkValidation)]
#[tokio::test]
async fn materialize_overlaps_leaf_gets_and_reads_each_node_once(#[case] reads: MaterializeReads) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(512, u32::MAX).with_max_leaf_bytes(6 * 1024);
    let mut fixture = Fixture::new(64, config, policy).await;
    let mut expected: Vec<_> = (0..64).map(make_fragment).collect();
    let bulk = matches!(reads, MaterializeReads::Drained);
    for round in 0..8u32 {
        let mut actions = Vec::new();
        // Eight distinct ids below 63 per round, spread over every leaf.
        for slot in 0..8u64 {
            let id = (u64::from(round) * 3 + slot * 8) % 63;
            let file = make_backfill_data_file(id, round);
            actions.push(action::add_data_file(id, &file));
            expected[id as usize].files.push(file);
        }
        fixture.commit(actions, policy, bulk).await;
    }
    fixture
        .commit(
            vec![
                action::remove_fragment(63),
                action::upsert_fragment(&make_fragment(64)).unwrap(),
            ],
            policy,
            bulk,
        )
        .await;
    expected[63] = make_fragment(64);
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.height >= 2, "{shape:?}");
    let interior_actions: u64 = shape.node_buffer_lens.iter().sum();
    if bulk {
        assert_eq!(interior_actions + shape.root_buffer_len, 0, "{shape:?}");
    } else {
        assert!(interior_actions > 0, "{shape:?}");
    }
    let (leaves, interiors) = (shape.leaf_keys.len(), shape.node_bytes.len());
    let delayed = FailpointController::default();
    delayed.set_get_latency(std::time::Duration::from_millis(2));
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    store.apply_wrapper(&io);
    let mut tree = fixture.tree.with_object_store(Arc::new(store));
    let mut transaction = None;
    match reads {
        MaterializeReads::WarmCache => {
            tree = reopen_tree(
                &tree,
                tree.store
                    .context()
                    .with_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024)),
            )
            .await;
            assert_eq!(tree.materialize().await.unwrap(), expected);
        }
        MaterializeReads::BulkValidation => {
            transaction = Some(tree.begin_commit(CommitMode::Bulk));
        }
        MaterializeReads::Drained | MaterializeReads::InteriorActions => {}
    }
    io.incremental_stats();
    let actual = if let Some(transaction) = transaction.as_mut() {
        transaction
            .resolve_all()
            .await
            .unwrap()
            .fragments
            .values()
            .cloned()
            .collect()
    } else {
        tree.materialize().await.unwrap()
    };
    assert_eq!(actual, expected);
    let measured = io.incremental_stats();
    assert_eq!(node_gets(&measured, "_bt/leaf/"), leaves, "{measured:?}");
    let expected_interior_gets = match reads {
        MaterializeReads::WarmCache => 0,
        _ => interiors,
    };
    assert_eq!(node_gets(&measured, "_bt/node/"), expected_interior_gets);
    assert!(
        measured.num_stages < measured.read_iops,
        "leaf GETs must overlap: {measured:?}"
    );

    // A second cache read admits leaves. The bulk attempt retains its complete
    // inventory. Both can answer another full read without fetching leaves.
    if matches!(
        reads,
        MaterializeReads::WarmCache | MaterializeReads::BulkValidation
    ) {
        let actual = if let Some(transaction) = transaction.as_mut() {
            transaction
                .resolve_all()
                .await
                .unwrap()
                .fragments
                .values()
                .cloned()
                .collect()
        } else {
            tree.materialize().await.unwrap()
        };
        assert_eq!(actual, expected);
        assert_eq!(node_gets(&io.incremental_stats(), "_bt/leaf/"), 0);
    }
}

/// Three interiors of two single-fragment leaves each, written directly so a
/// test can damage one leaf, served through a session cache. Cache keys name
/// the store, so every read goes through the one returned GET delay.
async fn two_level_tree(
    fixture: &Fixture,
) -> (
    FragmentTree,
    Vec<Vec<pb::FragmentTreeChild>>,
    FailpointController,
) {
    let delayed = FailpointController::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    let mut tree = fixture.tree.clone().with_object_store(Arc::new(store));
    tree = reopen_tree(
        &tree,
        tree.store
            .context()
            .with_leaf_cache(LanceCache::with_capacity(1024 * 1024)),
    )
    .await;
    let mut interiors = Vec::new();
    for ids in [[0, 5], [10, 20], [30, 40]] {
        let mut leaves = Vec::new();
        for id in ids {
            leaves.push(
                tree.store
                    .write_leaf(&[make_fragment(id)], 0)
                    .await
                    .unwrap()
                    .child_ref,
            );
        }
        interiors.push(leaves);
    }
    tree.children = write_interiors(&tree, &interiors).await;
    tree.next_fragment_id = 41;
    tree.buffer.clear();
    tree.buffer_index.take();
    (tree, interiors, delayed)
}

async fn write_interiors(
    tree: &FragmentTree,
    interiors: &[Vec<pb::FragmentTreeChild>],
) -> Vec<pb::FragmentTreeChild> {
    let mut children = Vec::new();
    for leaves in interiors {
        children.push(
            tree.store
                .write_internal(leaves.clone(), Vec::new())
                .await
                .unwrap()
                .child_ref,
        );
    }
    children
}

#[derive(Clone, Copy, Debug)]
enum EarlierLeaf {
    Intact,
    /// Its reference overstates its keys, found only once it is decoded.
    Corrupt,
}

/// The last leaf's object is gone, so its GET fails while earlier GETs are
/// still in flight. Its error surfaces only after every earlier leaf has
/// been decoded, and an earlier failure wins.
#[rstest]
#[case::intact_leaves_before_missing_leaf(EarlierLeaf::Intact)]
#[case::corrupt_leaf_before_missing_leaf(EarlierLeaf::Corrupt)]
#[tokio::test]
async fn materialize_reports_a_failed_leaf_get_in_key_order(#[case] earlier: EarlierLeaf) {
    let fixture = Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let (mut tree, mut interiors, delayed) = two_level_tree(&fixture).await;
    let expected: Vec<_> = [0, 5, 10, 20, 30, 40].map(make_fragment).into();
    // Every leaf is sighted once, so its next decode admits it.
    assert_eq!(tree.materialize().await.unwrap(), expected);
    let missing = interiors[2][1].path.clone();
    let path: Path = fixture
        .base
        .parts()
        .chain(Path::from(missing.as_str()).parts())
        .collect();
    fixture.store.inner.delete(&path).await.unwrap();
    let corrupt = interiors[0][0].path.clone();
    if matches!(earlier, EarlierLeaf::Corrupt) {
        interiors[0][0].num_keys += 1;
        tree.children = write_interiors(&tree, &interiors).await;
    }
    delayed.set_get_latency(std::time::Duration::from_millis(2));

    let error = tree.materialize().await.unwrap_err();
    match earlier {
        EarlierLeaf::Corrupt => {
            assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
            assert!(
                error.to_string().contains(&corrupt)
                    && error.to_string().contains("contents differ"),
                "{error}"
            );
        }
        EarlierLeaf::Intact => {
            assert!(error.to_string().contains(&missing), "{error}");
            fixture.io.incremental_stats();
            let earlier_leaves = interiors.concat();
            for leaf in &earlier_leaves[..earlier_leaves.len() - 1] {
                tree.store.read_leaf(leaf).await.unwrap();
            }
            assert_eq!(
                node_gets(&fixture.io.incremental_stats(), "_bt/leaf/"),
                0,
                "every earlier leaf was decoded a second time, and so admitted"
            );
        }
    }
}

/// GETs still in flight when the call is dropped are cancelled and admit
/// nothing, since only the calling task decodes and admits leaves.
#[tokio::test]
async fn dropped_materialize_cancels_its_gets_and_admits_no_leaf() {
    let fixture = Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let (tree, interiors, delayed) = two_level_tree(&fixture).await;
    // Leaves are sighted and interiors cached, so the next walk reaches
    // every leaf without waiting on a GET, and the next decode admits.
    let expected = tree.materialize().await.unwrap();
    delayed.set_get_latency(std::time::Duration::from_millis(50));
    fixture.io.incremental_stats();
    let dropped =
        tokio::time::timeout(std::time::Duration::from_millis(5), tree.materialize()).await;
    assert!(
        dropped.is_err(),
        "the call must still be waiting on its GETs"
    );
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    // The tracker sits below the delay, so it sees a GET only once the GET
    // outlives its delay, which a detached task would do by now.
    assert_eq!(
        node_gets(&fixture.io.incremental_stats(), "_bt/leaf/"),
        0,
        "a dropped call must cancel its in-flight GETs"
    );
    delayed.set_get_latency(std::time::Duration::ZERO);

    fixture.io.incremental_stats();
    assert_eq!(tree.materialize().await.unwrap(), expected);
    assert_eq!(
        node_gets(&fixture.io.incremental_stats(), "_bt/leaf/"),
        interiors.concat().len(),
        "no leaf may be admitted by a dropped call"
    );
}

/// Each interior buffer is valid alone. Replay spans every decoded buffer,
/// so the sequence both of them use is still rejected.
#[tokio::test]
async fn materialize_rejects_a_sequence_repeated_in_sibling_subtrees() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    let tree = &mut fixture.tree;
    let mut children = Vec::new();
    for ids in [[0, 1], [2, 3]] {
        let mut leaves = Vec::new();
        for id in ids {
            leaves.push(
                tree.store
                    .write_leaf(&[make_fragment(id)], 0)
                    .await
                    .unwrap()
                    .child_ref,
            );
        }
        let buffered = pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::add_data_file(
                ids[0],
                &make_backfill_data_file(ids[0], 0),
            )),
            ..Default::default()
        };
        children.push(
            tree.store
                .write_internal(leaves, vec![buffered])
                .await
                .unwrap()
                .child_ref,
        );
    }
    tree.children = children;
    tree.next_fragment_id = 4;
    tree.buffer.clear();
    tree.buffer_index.take();
    tree.next_action_sequence = 2;
    tree.store.next_action_sequence = 2;
    let error = tree.materialize().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("another buffer read by this operation"),
        "{error}"
    );
}

#[derive(Debug, Clone, Copy)]
enum PendingDeletions {
    None,
    InRoot,
    InInteriors,
}

#[rstest]
#[case::no_buffer(512, 0, 0, PendingDeletions::None)]
#[case::root_buffer(512, 1, 2, PendingDeletions::InRoot)]
#[case::interior_buffer(2048, 40, 16, PendingDeletions::InInteriors)]
#[tokio::test]
async fn leaf_stream_marks_leaves_with_routed_actions_as_replayed(
    #[case] fragment_count: u64,
    #[case] rounds: u64,
    #[case] deletions_per_round: u64,
    #[case] pending: PendingDeletions,
) {
    let config = FragmentTreeConfig {
        max_node_bytes: 4 * 1024,
        max_leaf_bytes: 16 * 1024,
        semantic_buffer_bytes: 2 * 1024,
        ..FragmentTreeConfig::default()
    };
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(fragment_count, config, policy).await;
    for round in 1..=rounds {
        // Deletions keep every id, so a leaf with a routed action still holds
        // the id the action names.
        let actions = (0..deletions_per_round)
            .map(|slot| {
                action::add_deletion_file(
                    (round * 16 + slot * 127) % fragment_count,
                    &crate::format::DeletionFile {
                        read_version: round,
                        id: round,
                        file_type: crate::format::DeletionFileType::Bitmap,
                        num_deleted_rows: Some(1),
                        base_id: None,
                    },
                )
                .unwrap()
            })
            .collect();
        fixture.commit(actions, policy, false).await;
    }
    let shape = fixture.tree.shape_report().await.unwrap();
    let interior_buffered: u64 = shape.node_buffer_lens.iter().sum();
    assert!(shape.leaf_keys.len() > 1, "{shape:?}");
    match pending {
        PendingDeletions::None => {
            assert_eq!((shape.root_buffer_len, interior_buffered), (0, 0));
        }
        PendingDeletions::InRoot => {
            assert!(shape.root_buffer_len > 0, "{shape:?}");
            assert_eq!(interior_buffered, 0);
        }
        PendingDeletions::InInteriors => assert!(interior_buffered > 0, "{shape:?}"),
    }

    let tree = Arc::new(
        fixture
            .open(&fixture.tree.descriptor, fixture.tree.version())
            .await,
    );
    let buffered = tree.buffered_action_keys().await.unwrap();
    let leaves: Vec<LeafFragments> = tree
        .clone()
        .leaf_stream(&single_flight_cache(), |_| true)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(leaves.len(), shape.leaf_keys.len());
    let mut locations = BTreeSet::new();
    let mut replayed = 0;
    for leaf in &leaves {
        match leaf {
            LeafFragments::Stored {
                location,
                fragments,
            } => {
                assert!(
                    fragments
                        .iter()
                        .all(|fragment| !buffered.contains(&fragment.id))
                );
                assert!(locations.insert(location.clone()), "{location:?} repeats");
            }
            LeafFragments::Replayed(fragments) => {
                assert!(
                    fragments
                        .iter()
                        .any(|fragment| buffered.contains(&fragment.id))
                );
                replayed += 1;
            }
        }
    }
    match pending {
        PendingDeletions::None => assert_eq!(replayed, 0),
        PendingDeletions::InRoot => {
            assert!(replayed > 0 && replayed < leaves.len(), "{replayed}");
        }
        PendingDeletions::InInteriors => assert!(replayed > 0),
    }
    let flattened: Vec<Fragment> = leaves
        .iter()
        .flat_map(|leaf| match leaf {
            LeafFragments::Stored { fragments, .. } => fragments.to_vec(),
            LeafFragments::Replayed(fragments) => fragments.clone(),
        })
        .collect();
    let streamed: Vec<Fragment> = tree
        .clone()
        .scan_with_readahead(0, NonZeroUsize::MIN)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(flattened, streamed);
    assert_eq!(flattened, fixture.tree.materialize().await.unwrap());

    // A walk that skips every recorded location reads only replayed leaves.
    fixture.io.incremental_stats();
    let unrecorded: Vec<LeafFragments> = tree
        .leaf_stream(&single_flight_cache(), |location| {
            !locations.contains(location)
        })
        .try_collect()
        .await
        .unwrap();
    assert_eq!(unrecorded.len(), replayed);
    assert_eq!(
        leaf_requests(&fixture.io.incremental_stats()).len(),
        replayed
    );
}

/// A childless root keeps its whole table in buffered actions, which no stored
/// leaf holds, so its walk is one replayed leaf in id order.
#[tokio::test]
async fn leaf_stream_yields_a_childless_root_as_one_replayed_leaf() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    let actions = [3, 1, 2]
        .into_iter()
        .map(|id| action::upsert_fragment(&make_fragment(id)).unwrap())
        .collect();
    fixture.commit(actions, policy, false).await;
    assert!(fixture.tree.children.is_empty());

    let tree = Arc::new(
        fixture
            .open(&fixture.tree.descriptor, fixture.tree.version())
            .await,
    );
    let leaves: Vec<LeafFragments> = tree
        .clone()
        .leaf_stream(&single_flight_cache(), |_| true)
        .try_collect()
        .await
        .unwrap();
    let summary = tree.summary(&mut LeafSummaries::default()).await.unwrap();
    let streamed: Vec<Fragment> = tree
        .scan_with_readahead(0, NonZeroUsize::MIN)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(
        streamed
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(summary, ManifestSummary::of(&streamed));
    assert_eq!(leaves, [LeafFragments::Replayed(streamed)]);
}

/// Versions share leaves, so one pass over them reads a stored leaf once and
/// replays a leaf with routed actions again only when its actions change.
#[tokio::test]
async fn summaries_read_each_stored_leaf_once_and_replay_only_changed_actions() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(300, multi_leaf_fixture_config(), policy).await;
    let leaf_count = fixture.tree.shape_report().await.unwrap().leaf_keys.len();
    assert!(leaf_count >= 3, "{leaf_count}");
    let deletion = |id: u64, version: u64| {
        action::add_deletion_file(
            id,
            &crate::format::DeletionFile {
                read_version: version,
                id: version,
                file_type: crate::format::DeletionFileType::Bitmap,
                num_deleted_rows: Some(1),
                base_id: None,
            },
        )
        .unwrap()
    };
    let mut versions = vec![fixture.tree.clone()];
    // Each commit changes the actions routed to one leaf, and the root buffer
    // holds them all, so no leaf is rewritten.
    for actions in [
        vec![deletion(0, 2)],
        vec![action::upsert_fragment(&make_fragment(300)).unwrap()],
        vec![deletion(1, 4)],
    ] {
        fixture.commit(actions, policy, false).await;
        versions.push(fixture.tree.clone());
    }
    let shape = fixture.tree.shape_report().await.unwrap();
    assert_eq!(shape.leaf_keys.len(), leaf_count);
    assert_eq!(shape.root_buffer_len, 3);

    let mut leaves = LeafSummaries::default();
    let mut reads = Vec::new();
    for version in &versions {
        let tree = FragmentTree::open_snapshot(
            fixture.context(),
            &version.descriptor,
            version.version(),
            fixture.config.clone(),
            version.next_fragment_id(),
        )
        .await
        .unwrap();
        let expected = ManifestSummary::of(&tree.materialize().await.unwrap());
        fixture.io.incremental_stats();
        assert_eq!(tree.summary(&mut leaves).await.unwrap(), expected);
        reads.push(leaf_requests(&fixture.io.incremental_stats()).len());
    }
    // The second deletion lands in the first deletion's leaf, while the
    // appended leaf repeats its actions.
    assert_eq!(reads, [leaf_count, 1, 1, 1]);
}

#[tokio::test]
async fn childless_root_scan_honors_its_lower_bound_and_ends_after_a_failed_replay() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    let actions = [3, 1, 2]
        .into_iter()
        .map(|id| action::upsert_fragment(&make_fragment(id)).unwrap())
        .collect();
    fixture.commit(actions, policy, false).await;
    assert!(fixture.tree.children.is_empty());
    let from_two: Vec<u64> = Arc::new(fixture.tree.clone())
        .scan(2)
        .map_ok(|fragment| fragment.id)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(from_two, [2, 3]);

    let mut tree = fixture.tree;
    tree.buffer[0].fragment_count_delta = 0;
    tree.buffer_index.take();
    let mut stream = Arc::new(tree).scan(0);
    let error = stream
        .try_next()
        .await
        .expect_err("a misstated delta must not replay");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("deltas"), "{error}");
    assert!(stream.try_next().await.unwrap().is_none());
}

fn single_flight_cache() -> SingleFlightLeafCache {
    SingleFlightLeafCache::new(LanceCache::with_capacity(64 * 1024 * 1024))
}

/// `count` handles on the fixture's current version that share one store
/// whose GETs take `latency`, and the tracker of that store.
async fn delayed_handles(
    fixture: &Fixture,
    count: usize,
    latency: std::time::Duration,
) -> (Vec<Arc<FragmentTree>>, IOTracker) {
    let delayed = FailpointController::default();
    delayed.set_get_latency(latency);
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    store.apply_wrapper(&io);
    let store = Arc::new(store);
    let mut handles = Vec::with_capacity(count);
    for _ in 0..count {
        let tree = fixture
            .open(&fixture.tree.descriptor, fixture.tree.version())
            .await;
        handles.push(Arc::new(tree.with_object_store(store.clone())));
    }
    (handles, io)
}

async fn walk_leaves(
    tree: Arc<FragmentTree>,
    leaves: &SingleFlightLeafCache,
) -> Result<Vec<Fragment>> {
    let walked: Vec<LeafFragments> = tree.leaf_stream(leaves, |_| true).try_collect().await?;
    Ok(walked
        .iter()
        .flat_map(|leaf| match leaf {
            LeafFragments::Stored { fragments, .. } => fragments.to_vec(),
            LeafFragments::Replayed(fragments) => fragments.clone(),
        })
        .collect())
}

async fn wait_until(reached: impl Fn() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !reached() {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

fn multi_leaf_fixture_config() -> FragmentTreeConfig {
    FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024)
}

/// Walks that reach a leaf together wait on one read of it, and that read
/// admits the leaf because a second walk wanted it, so a later walk reads
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_leaf_streams_read_each_leaf_once() {
    let fixture = Fixture::new(300, multi_leaf_fixture_config(), SnapshotPolicy::default()).await;
    let expected = fixture.tree.materialize().await.unwrap();
    let (handles, io) = delayed_handles(&fixture, 4, std::time::Duration::from_millis(20)).await;
    assert!(handles[0].root_child_count() >= 2);
    let leaves = single_flight_cache();
    io.incremental_stats();

    let walked = futures::future::try_join_all(
        handles
            .iter()
            .map(|tree| walk_leaves(tree.clone(), &leaves)),
    )
    .await
    .unwrap();
    assert!(walked.iter().all(|fragments| *fragments == expected));
    let reads = leaf_requests(&io.incremental_stats());
    let distinct: BTreeSet<&String> = reads.iter().collect();
    assert!(distinct.len() >= 2, "{reads:?}");
    assert_eq!(reads.len(), distinct.len(), "{reads:?}");
    assert_eq!(leaves.reads_in_flight(), 0);

    assert_eq!(
        walk_leaves(handles[0].clone(), &leaves).await.unwrap(),
        expected
    );
    assert!(leaf_requests(&io.incremental_stats()).is_empty());
}

/// A walk that reads a leaf alone only sights it, as a plain cached read
/// does, so the second walk reads it again and admits it.
#[tokio::test]
async fn lone_leaf_stream_admits_a_leaf_on_its_second_read() {
    let fixture = Fixture::new(300, multi_leaf_fixture_config(), SnapshotPolicy::default()).await;
    let expected = fixture.tree.materialize().await.unwrap();
    let (handles, io) = delayed_handles(&fixture, 1, std::time::Duration::ZERO).await;
    let leaves = single_flight_cache();
    io.incremental_stats();

    let mut reads_per_walk = Vec::new();
    for _ in 0..3 {
        assert_eq!(
            walk_leaves(handles[0].clone(), &leaves).await.unwrap(),
            expected
        );
        reads_per_walk.push(leaf_requests(&io.incremental_stats()).len());
    }
    assert!(reads_per_walk[0] >= 2, "{reads_per_walk:?}");
    assert_eq!(
        reads_per_walk,
        [reads_per_walk[0], reads_per_walk[0], 0],
        "sighted, admitted, cached"
    );
    assert_eq!(leaves.reads_in_flight(), 0);
}

/// A failed read is never shared. Each walk that was waiting on it then leads
/// its own read of the missing leaf, so every walk reports the store's
/// not-found error naming the leaf and none stays waiting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_leaf_read_gives_every_waiting_walk_its_own_error() {
    let fixture = Fixture::new(300, multi_leaf_fixture_config(), SnapshotPolicy::default()).await;
    let (handles, _io) = delayed_handles(&fixture, 4, std::time::Duration::from_millis(20)).await;
    let mut walk = handles[0].start_walk_from(0);
    let (leaf, _, _) = handles[0].walk_next_leaf(&mut walk).await.unwrap().unwrap();
    let path: Path = fixture
        .base
        .parts()
        .chain(Path::from(leaf.path.as_str()).parts())
        .collect();
    fixture.store.inner.delete(&path).await.unwrap();
    let leaves = single_flight_cache();
    let spawn_walk = |tree: &Arc<FragmentTree>| {
        let tree = tree.clone();
        let leaves = leaves.clone();
        tokio::spawn(async move { walk_leaves(tree, &leaves).await })
    };

    let lead = spawn_walk(&handles[0]);
    wait_until(|| leaves.reads_in_flight() == 1).await;
    let followers: Vec<_> = handles[1..].iter().map(spawn_walk).collect();
    // A failed read wakes every walk still waiting on it. One of them leads
    // the next read and the rest wait on that one.
    for waiting in (0..=3).rev() {
        wait_until(|| (leaves.reads_in_flight(), leaves.waiting_readers()) == (1, waiting)).await;
    }
    for spawned in std::iter::once(lead).chain(followers) {
        let error = spawned.await.unwrap().unwrap_err();
        assert!(
            !matches!(error, Error::Cloned { .. } | Error::Shared { .. }),
            "{error:?}"
        );
        assert!(error.is_not_found(), "{error:?}");
        assert!(error.to_string().contains(&leaf.path), "{error}");
    }
    assert_eq!((leaves.reads_in_flight(), leaves.waiting_readers()), (0, 0));
}

/// Aborting the walk that leads a read wakes the walk waiting on it, which
/// then reads the leaf itself and returns its records.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_lead_read_leaves_no_walk_waiting() {
    let fixture = Fixture::new(300, multi_leaf_fixture_config(), SnapshotPolicy::default()).await;
    let latency = std::time::Duration::from_millis(50);
    let (handles, _io) = delayed_handles(&fixture, 1, latency).await;
    let tree = handles[0].clone();
    let mut walk = tree.start_walk_from(0);
    let (child, _, end) = tree.walk_next_leaf(&mut walk).await.unwrap().unwrap();
    let expected = tree.store.read_leaf_in_range(&child, end).await.unwrap();
    let leaves = single_flight_cache();
    let read = |leaves: SingleFlightLeafCache| {
        let store = tree.store.clone();
        let child = child.clone();
        tokio::spawn(async move {
            store
                .read_leaf_single_flight(&leaves, &child, end)
                .await
                .map(|read| read.into_vec())
        })
    };

    let lead = read(leaves.clone());
    wait_until(|| leaves.reads_in_flight() == 1).await;
    let follower = read(leaves.clone());
    wait_until(|| leaves.waiting_readers() == 1).await;
    lead.abort();
    let fragments = tokio::time::timeout(latency * 4, follower)
        .await
        .expect("the waiting walk must not hang")
        .unwrap()
        .unwrap();
    assert_eq!(fragments, expected);
    assert_eq!(leaves.reads_in_flight(), 0);
}

/// Two inherited bases can resolve to one stored leaf, yet its records belong
/// to the base the reference names, as a flat manifest's data files carry
/// their own base id. Reads through the two bases that miss together lead
/// separate reads, and every read returns records stamped with its own base.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aliased_base_ids_never_share_a_single_flight_read() {
    let fixture = Fixture::new(300, multi_leaf_fixture_config(), SnapshotPolicy::default()).await;
    let (handles, io) = delayed_handles(&fixture, 1, std::time::Duration::from_millis(20)).await;
    let tree = handles[0].clone();
    let mut walk = tree.start_walk_from(0);
    let (local, _, end) = tree.walk_next_leaf(&mut walk).await.unwrap().unwrap();
    let unstamped = tree.store.read_leaf_in_range(&local, end).await.unwrap();
    let owned_by = |base_id: u32| {
        let mut fragments = unstamped.clone();
        for file in fragments
            .iter_mut()
            .flat_map(|fragment| &mut fragment.files)
        {
            file.base_id = Some(base_id);
        }
        fragments
    };
    let mut store = tree.store.clone();
    let owner = (store.object_store.clone(), store.base().clone());
    store.set_foreign_bases(HashMap::from([(1, owner.clone()), (2, owner)]));
    let aliases = [1, 2].map(|base_id| pb::FragmentTreeChild {
        base_id: Some(base_id),
        ..local.clone()
    });
    assert_eq!(
        store.child_location(&aliases[0]).unwrap(),
        store.child_location(&aliases[1]).unwrap()
    );
    let leaves = single_flight_cache();
    let read = |child: pb::FragmentTreeChild| {
        let store = store.clone();
        let leaves = leaves.clone();
        tokio::spawn(async move {
            store
                .read_leaf_single_flight(&leaves, &child, end)
                .await
                .map(|read| read.into_vec())
        })
    };

    io.incremental_stats();
    let mut gets_per_pass = Vec::new();
    for pass in 0..3 {
        let reads = aliases.clone().map(&read);
        if pass == 0 {
            wait_until(|| leaves.reads_in_flight() + leaves.waiting_readers() == 2).await;
            assert_eq!((leaves.reads_in_flight(), leaves.waiting_readers()), (2, 0));
        }
        for (handle, base_id) in reads.into_iter().zip([1, 2]) {
            assert_eq!(
                handle.await.unwrap().unwrap(),
                owned_by(base_id),
                "pass {pass}"
            );
        }
        gets_per_pass.push(leaf_requests(&io.incremental_stats()).len());
    }
    assert_eq!(gets_per_pass, [2, 2, 0], "sighted, admitted, cached");
    assert_eq!((leaves.reads_in_flight(), leaves.waiting_readers()), (0, 0));
}

#[tokio::test]
async fn failed_prepare_preserves_frontiers_and_buffer() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(1024, u32::MAX)
        .with_max_leaf_bytes(4096)
        .with_hard_capacity_bytes(8192);
    let mut fixture = Fixture::new(2, config, policy).await;
    let error = fixture
        .tree
        .begin_commit(CommitMode::Bulk)
        .prepare(
            ValidatedCommit::fragment_actions(vec![
                action::upsert_fragment(&make_fragment_with_files(9, 256)).unwrap(),
            ]),
            policy,
        )
        .await
        .err()
        .expect("preparation must fail");
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("hard_capacity_bytes"), "{error}");
    assert_eq!(fixture.tree.version(), 1);
    assert_eq!(fixture.tree.next_fragment_id(), 2);
    assert_eq!(fixture.tree.root_buffer_len(), 0);
    fixture
        .commit(
            vec![action::upsert_fragment(&make_fragment(2)).unwrap()],
            policy,
            false,
        )
        .await;
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        (0..3).map(make_fragment).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::unloaded(false)]
#[case::resident(true)]
#[tokio::test]
async fn dropped_prepare_leaves_the_tree_unchanged(#[case] resident: bool) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(1, config, policy).await;
    if resident {
        fixture.tree = fixture.tree.hydrate().await.unwrap();
    }
    let original_resident = fixture.tree.resident.clone();
    let original = fixture.tree.materialize().await.unwrap();
    let original_descriptor = fixture.tree.descriptor.clone();
    assert_eq!(fixture.tree.children.len(), 1);
    let gate = flush::amortization_gate(
        flush::fair_share(&fixture.tree.children, &fixture.tree.config),
        &fixture.tree.children[0],
    );
    let mut file = make_backfill_data_file(0, 1);
    // Semantic pressure alone can leave a small edit buffered. Cross the leaf's
    // drain threshold so preparation must read it after taking its routing state.
    file.path = format!("data/{}.lance", "x".repeat(gate as usize));
    let action = action::add_data_file(0, &file);
    let tagged = [stored(fixture.tree.next_action_sequence, action.clone())];
    assert!(node::buffer_pressured(&tagged, &fixture.tree.config));
    assert!(node::internal_logical_bytes(&[], &tagged) >= gate);
    let delayed = FailpointController::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let mut transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    transaction.resolve_touched(&[0]).await.unwrap();
    delayed.set_get_latency(std::time::Duration::from_secs(60));
    let mut pending = Box::pin(transaction.prepare(
        ValidatedCommit::fragment_actions(vec![action.clone()]),
        policy,
    ));
    // Validation is complete. Suspend after the attempt starts rewriting its nodes.
    assert!(futures::poll!(pending.as_mut()).is_pending());
    drop(pending);
    fixture.tree = fixture.tree.with_object_store(fixture.store.clone());
    assert_eq!(fixture.tree.version(), 1);
    assert_eq!(fixture.tree.descriptor, original_descriptor);
    assert_eq!(fixture.tree.materialize().await.unwrap(), original);
    if let Some(original_resident) = &original_resident {
        assert!(Arc::ptr_eq(
            original_resident,
            fixture.tree.resident_fragments().unwrap()
        ));
    }

    let transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    let (prepared, stats) = transaction
        .prepare(ValidatedCommit::fragment_actions(vec![action]), policy)
        .await
        .unwrap();
    assert_eq!(fixture.tree.descriptor, original_descriptor);
    assert_eq!(fixture.tree.materialize().await.unwrap(), original);
    assert_eq!(prepared.version(), fixture.tree.version());
    assert!(
        stats.flushes > 0,
        "the retry must drain the same leaf: {stats:?}"
    );
    let mut expected = original;
    expected[0].files.push(file);
    let reopened = fixture.open(&prepared.descriptor, 2).await;
    assert_eq!(reopened.materialize().await.unwrap(), expected);
}

#[rstest]
#[case::cleared_selection(false)]
#[case::stale_selection(true)]
#[tokio::test]
async fn modifying_resolved_selection_does_not_change_commit_deltas(
    #[case] stale: bool,
    #[values(false, true)] remove: bool,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(2, FragmentTreeConfig::default(), policy).await;
    let old = fixture
        .tree
        .begin_commit(CommitMode::Incremental)
        .resolve_touched(&[0])
        .await
        .unwrap()
        .clone();
    let mut current = make_fragment(0);
    current.physical_rows = Some(5);
    fixture
        .commit(
            vec![action::upsert_fragment(&current).unwrap()],
            policy,
            false,
        )
        .await;
    let source = fixture.tree.materialize().await.unwrap();
    let descriptor = fixture.tree.descriptor.clone();
    let mut transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    let mut selected = transaction.resolve_touched(&[0]).await.unwrap().clone();
    assert_eq!(selected.get(0), Some(&current));
    selected.ids.clear();
    if stale {
        selected.fragments = old.fragments;
        assert_eq!(selected.get(0).unwrap().physical_rows, Some(1));
    } else {
        selected.fragments.clear();
        assert!(selected.get(0).is_none());
    }
    let mut replacement = make_fragment(0);
    replacement.physical_rows = Some(7);
    let change = if remove {
        action::remove_fragment(0)
    } else {
        action::upsert_fragment(&replacement).unwrap()
    };
    let (prepared, _) = transaction
        .prepare(ValidatedCommit::fragment_actions(vec![change]), policy)
        .await
        .unwrap();
    let expected = if remove {
        vec![make_fragment(1)]
    } else {
        vec![replacement, make_fragment(1)]
    };
    assert_eq!(prepared.count_fragments(), expected.len() as u64);
    assert_eq!(prepared.count_rows(), if remove { 1 } else { 8 });
    assert_eq!(prepared.count_visible_rows(), if remove { 1 } else { 8 });
    assert_eq!(prepared.materialize().await.unwrap(), expected);
    assert_eq!(
        fixture
            .open(&prepared.descriptor, 2)
            .await
            .materialize()
            .await
            .unwrap(),
        expected
    );
    assert_eq!(fixture.tree.descriptor, descriptor);
    assert_eq!(fixture.tree.materialize().await.unwrap(), source);
}

#[tokio::test]
async fn prepare_resolves_existing_targets_without_a_prior_resolve() {
    let policy = SnapshotPolicy::default();
    let fixture = Fixture::new(2, FragmentTreeConfig::default(), policy).await;
    let mut replacement = make_fragment(0);
    replacement.physical_rows = Some(7);
    let (prepared, _) = fixture
        .tree
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&replacement).unwrap()]),
            policy,
        )
        .await
        .unwrap();
    assert_eq!(prepared.count_fragments(), 2);
    assert_eq!(prepared.count_rows(), 8);
    assert_eq!(prepared.count_visible_rows(), 8);
    assert_eq!(
        prepared.materialize().await.unwrap(),
        vec![replacement, make_fragment(1)]
    );
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        vec![make_fragment(0), make_fragment(1)]
    );
}

#[tokio::test]
async fn complete_inventory_proves_reserved_absence_without_another_read() {
    let policy = SnapshotPolicy::default();
    let fixture = Fixture::new(2, FragmentTreeConfig::default(), policy).await;
    let (reserved, _) = fixture
        .tree
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit {
                next_fragment_id: Some(8),
                fragment_actions: Vec::new(),
            },
            policy,
        )
        .await
        .unwrap();
    let mut transaction = reserved.begin_commit(CommitMode::Incremental);
    let inventory = transaction.resolve_all().await.unwrap();
    assert_eq!(inventory.fragments.len(), 2);
    assert!(inventory.get(5).is_none());
    fixture.io.incremental_stats();
    let (prepared, _) = transaction
        .prepare(
            ValidatedCommit::fragment_actions(vec![
                action::upsert_fragment(&make_fragment(5)).unwrap(),
            ]),
            policy,
        )
        .await
        .unwrap();
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    assert_eq!(prepared.next_fragment_id(), 8);
    assert_eq!(prepared.count_fragments(), 3);
    assert_eq!(prepared.count_rows(), 3);
    assert_eq!(prepared.count_visible_rows(), 3);
    assert_eq!(
        prepared.materialize().await.unwrap(),
        vec![make_fragment(0), make_fragment(1), make_fragment(5)]
    );
    assert_eq!(reserved.count_fragments(), 2);
}

fn stored(action_sequence: u64, action: pb::FragmentAction) -> pb::FragmentTreeMutation {
    pb::FragmentTreeMutation {
        action_sequence,
        action: Some(action),
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum OuterBuffer {
    Root,
    Interior,
}

#[derive(Clone, Copy, Debug)]
enum LaterMutation {
    ClearDeletionFile,
    Upsert,
}

// The spec permits sequences 1 and 3 in one buffer, with 2 in a deeper buffer.
#[rstest]
#[case::root_reset_then_clear(OuterBuffer::Root, LaterMutation::ClearDeletionFile)]
#[case::root_edit_between_resets(OuterBuffer::Root, LaterMutation::Upsert)]
#[case::interior_reset_then_clear(OuterBuffer::Interior, LaterMutation::ClearDeletionFile)]
#[tokio::test]
async fn interleaved_stored_history_replays_in_sequence_order(
    #[case] outer: OuterBuffer,
    #[case] later: LaterMutation,
) {
    let policy = SnapshotPolicy::default();
    // Fanout 4 and 4 KiB leaves keep these small nodes above the merge
    // floors, so drains reach every level instead of collapsing the tree.
    let config = FragmentTreeConfig::new(16 * 1024, 4)
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(0, config.clone(), policy).await;
    let nodes = &fixture.tree.store;
    let left = nodes.write_leaf(&[make_fragment(0)], 0).await.unwrap();
    let right = nodes.write_leaf(&[make_fragment(1)], 0).await.unwrap();
    let added = make_backfill_data_file(0, 1);
    let inner = nodes
        .write_internal(
            vec![left.child_ref, right.child_ref],
            vec![stored(2, action::add_data_file(0, &added))],
        )
        .await
        .unwrap();
    let mut sibling = Vec::new();
    for id in [2, 3] {
        sibling.push(
            nodes
                .write_leaf(&[make_fragment(id)], 0)
                .await
                .unwrap()
                .child_ref,
        );
    }
    let sibling = nodes.write_internal(sibling, Vec::new()).await.unwrap();
    let mut first = make_fragment(0);
    first.files[0] = make_replacement_data_file(0, 0);
    let (last, expected) = match later {
        LaterMutation::Upsert => {
            let mut second = make_fragment(0);
            second.files[0] = make_replacement_data_file(0, 1);
            (action::upsert_fragment(&second).unwrap(), second)
        }
        LaterMutation::ClearDeletionFile => {
            let mut expected = first.clone();
            expected.files.push(added);
            (action::clear_deletion_file(0), expected)
        }
    };
    let outer_buffer = vec![
        stored(1, action::upsert_fragment(&first).unwrap()),
        stored(3, last),
    ];
    let (children, buffer) = match outer {
        OuterBuffer::Interior => {
            let outer = nodes
                .write_internal(vec![inner.child_ref, sibling.child_ref], outer_buffer)
                .await
                .unwrap();
            (vec![outer.child_ref], Vec::new())
        }
        OuterBuffer::Root => (vec![inner.child_ref, sibling.child_ref], outer_buffer),
    };
    let snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children,
            buffer,
            next_action_sequence: 4,
        })),
        mutations_since_root: vec![],
        next_action_sequence: 4,
    };
    fixture.tree = FragmentTree::open_snapshot(
        TreeContext::new(
            fixture.store.clone(),
            fixture.base.clone(),
            fixture.scheduler.clone(),
        ),
        &snapshot,
        1,
        config,
        4,
    )
    .await
    .unwrap();
    assert_eq!(
        fixture.tree.resolve_fragment(0).await.unwrap(),
        Some(expected.clone())
    );

    let appended = make_fragment(4);
    let (tree, stats) = fixture
        .tree
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&appended).unwrap()]),
            policy,
        )
        .await
        .unwrap();
    fixture.tree = tree;
    assert!(stats.flushes > 0, "{stats:?}");
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        [
            vec![expected],
            (1..4).map(make_fragment).collect(),
            vec![appended]
        ]
        .concat()
    );
}

#[tokio::test]
async fn fresh_opens_fold_repeated_edits_above_leaves() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(2, FragmentTreeConfig::default(), policy).await;
    assert_eq!(fixture.tree.height(), 1);
    let mut replaced = make_fragment(0);
    for round in 0..3 {
        replaced.files[0] = make_replacement_data_file(0, round);
        fixture
            .commit(
                vec![action::upsert_fragment(&replaced).unwrap()],
                policy,
                false,
            )
            .await;
        fixture.tree = fixture
            .open(&fixture.tree.descriptor, fixture.tree.version())
            .await;
        assert_eq!(fixture.tree.root_buffer_len(), 1);
    }
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        vec![replaced, make_fragment(1)]
    );
}

// Replacing fragment 0 twice must not count as replacing both fragments in the leaf.
#[tokio::test]
async fn repeated_replacements_of_one_fragment_cover_it_once() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let fixture = Fixture::new(0, config.clone(), policy).await;
    let leaf = fixture
        .tree
        .store
        .write_leaf(&[make_fragment(0), make_fragment(1)], 0)
        .await
        .unwrap();
    // Wide records make the drain worth its leaf rewrite.
    let mut first = make_fragment_with_files(0, 32);
    first.files[0] = make_replacement_data_file(0, 0);
    let snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children: vec![leaf.child_ref],
            buffer: vec![stored(1, action::upsert_fragment(&first).unwrap())],
            next_action_sequence: 2,
        })),
        mutations_since_root: vec![],
        next_action_sequence: 2,
    };
    let tree = FragmentTree::open_snapshot(
        TreeContext::new(
            fixture.store.clone(),
            fixture.base.clone(),
            fixture.scheduler.clone(),
        ),
        &snapshot,
        1,
        config.clone(),
        2,
    )
    .await
    .unwrap();
    let mut second = first.clone();
    second.files[0] = make_replacement_data_file(0, 1);
    let mut transaction = tree.begin_commit(CommitMode::Incremental);
    transaction.resolve_touched(&[0]).await.unwrap();
    let (prepared, stats) = transaction
        .prepare(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&second).unwrap()]),
            policy,
        )
        .await
        .unwrap();
    let tree = prepared;
    assert!(stats.flushes > 0, "{stats:?}");
    assert_eq!(
        tree.materialize().await.unwrap(),
        vec![second, make_fragment(1)]
    );
}

#[rstest]
#[case::rewrite(false, None)]
#[case::split_and_remove(true, None)]
#[case::failed_put(false, Some((FailWhen::Before, 2)))]
#[case::lost_put_response(false, Some((FailWhen::After, 2)))]
// Past the store's leaf PUT window, so earlier PUTs are still in flight.
#[case::failed_put_past_window(true, Some((FailWhen::Before, 10)))]
#[case::lost_put_response_past_window(true, Some((FailWhen::After, 10)))]
#[tokio::test]
async fn sibling_leaf_drains_overlap_and_preserve_snapshots(
    #[case] change_shape: bool,
    #[case] failure: Option<(FailWhen, usize)>,
) {
    let policy = SnapshotPolicy::default();
    // The scenario needs every leaf directly under the root, so the fanout
    // cap is lifted; the default cap would split this wide directory.
    let config = FragmentTreeConfig {
        max_children_per_node: u32::MAX,
        ..FragmentTreeConfig::new(16 * 1024, u32::MAX)
            .with_max_leaf_bytes(6 * 1024)
            .with_semantic_buffer_bytes(1)
    };
    let mut fixture = Fixture::new(128, config, policy).await;
    assert_eq!(fixture.tree.height(), 1);
    assert!(fixture.tree.children.len() > 1);
    let original = fixture.tree.materialize().await.unwrap();
    let previous = fixture.tree.descriptor.clone();
    let mut expected = Vec::new();
    let mut actions = Vec::new();
    for fragment in &original {
        let child = node::child_index_for(&fixture.tree.children, fragment.id);
        // Each leaf keeps its first record, so every drain reads its leaf and
        // the reads can overlap.
        if fragment.id == fixture.tree.children[child].min_key {
            expected.push(fragment.clone());
            continue;
        }
        if change_shape && child.is_multiple_of(2) {
            actions.push(action::remove_fragment(fragment.id));
        } else {
            let replacement =
                make_fragment_with_files(fragment.id, if change_shape { 16 } else { 2 });
            actions.push(action::upsert_fragment(&replacement).unwrap());
            expected.push(replacement);
        }
    }
    let ids: Vec<_> = original.iter().map(|fragment| fragment.id).collect();
    let delayed = FailpointController::default();
    delayed.set_get_latency(std::time::Duration::from_millis(2));
    if let Some((when, nth)) = failure {
        delayed.arm(Failpoint {
            on: FailOn::Put,
            when,
            path_contains: "_bt/leaf/".into(),
            nth,
        });
    }
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    store.apply_wrapper(&io);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let mut transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    transaction.resolve_touched(&ids).await.unwrap();
    let result = transaction
        .prepare(ValidatedCommit::fragment_actions(actions.clone()), policy)
        .await;
    if failure.is_some() {
        let error = result.err().expect("preparation must fail");
        assert!(matches!(error, Error::IO { .. }), "{error}");
        assert!(delayed.tripped());
        assert!(error.to_string().contains("failpoint"), "{error}");
        assert_eq!(fixture.tree.version(), 1);
        assert_eq!(fixture.tree.root_buffer_len(), 0);
        assert_eq!(fixture.tree.materialize().await.unwrap(), original);
        assert_eq!(fixture.tree.descriptor, previous);
        delayed.disarm();
        let (prepared, _) = fixture
            .tree
            .begin_commit(CommitMode::Incremental)
            .prepare(ValidatedCommit::fragment_actions(actions), policy)
            .await
            .unwrap();
        fixture.tree = prepared;
    } else {
        let (prepared, stats) = result.unwrap();
        fixture.tree = prepared;
        assert!(stats.flushes > 1);
        if change_shape {
            assert!(stats.splits > 0);
        }
        let measured = io.incremental_stats();
        assert!(
            measured.num_stages < measured.read_iops + measured.write_iops,
            "sibling drains must overlap: {measured:?}"
        );
    }
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
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

#[rstest]
#[case::buffered_inline(false, usize::MAX)]
#[case::buffered_external(false, 0)]
#[case::bulk_external(true, 0)]
#[tokio::test]
async fn aliased_file_actions_survive_pressure_and_checkpoint(
    #[case] bulk: bool,
    #[case] inline_root_bytes: usize,
) {
    let policy = SnapshotPolicy {
        inline_root_bytes,
        max_suffix_bytes: 256,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX)
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(32, config, policy).await;
    assert!(fixture.tree.height() >= 2);
    let mut expected: BTreeMap<_, _> = (0..32).map(|id| (id, make_fragment(id))).collect();
    let mut fragment = make_fragment_with_files(7, 2);
    fragment.files[0].path = "A".into();
    fragment.files[1].path = "B".into();
    fixture
        .commit(
            vec![action::upsert_fragment(&fragment).unwrap()],
            policy,
            bulk,
        )
        .await;
    expected.insert(7, fragment.clone());
    let historical = fixture.tree.descriptor.clone();
    let mut materialized = 0;
    for (from, to) in [("B", "A"), ("A", "C")] {
        let mut replacement = fragment.files[0].clone();
        replacement.path = to.into();
        // Replacement mapping and version are deliberately different: the
        // action changes the first matching slot's location fields only.
        replacement.fields = Arc::from([99]);
        replacement.column_indices = Arc::from([9]);
        replacement.file_major_version = 99;
        let stats = fixture
            .commit(
                vec![action::replace_data_file(7, from, &replacement)],
                policy,
                bulk,
            )
            .await;
        materialized += stats.messages_materialized;
        let matched = fragment
            .files
            .iter_mut()
            .find(|file| file.path == from)
            .unwrap();
        matched.path = to.into();
        matched.file_size_bytes = replacement.file_size_bytes;
        matched.base_id = replacement.base_id;
        expected.insert(7, fragment.clone());
        let reopened = fixture
            .open(&fixture.tree.descriptor, fixture.tree.version())
            .await;
        assert_eq!(
            reopened.resolve_fragment(7).await.unwrap(),
            Some(fragment.clone())
        );
        assert_eq!(
            reopened.materialize().await.unwrap(),
            expected.values().cloned().collect::<Vec<_>>()
        );
        assert_eq!(
            reopened
                .resolve_fragments(&RoaringBitmap::from_iter([7]))
                .await
                .unwrap(),
            vec![fragment.clone()]
        );
        assert_eq!(
            reopened.fragment_at_row_offset(7).await.unwrap(),
            OffsetResolution::Found(Box::new(fragment.clone()), 0)
        );
        fixture.tree.validate().await.unwrap();
    }
    assert_eq!(
        fragment
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>(),
        ["C", "A"]
    );
    if !bulk {
        assert_eq!(
            materialized, 0,
            "a chain below the amortization gate stays buffered"
        );
    }
    fixture.commit(Vec::new(), policy, true).await;
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        expected.values().cloned().collect::<Vec<_>>()
    );
    let old = fixture.open(&historical, 2).await;
    let old_fragment = old.resolve_fragment(7).await.unwrap().unwrap();
    assert_eq!(
        old_fragment
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>(),
        ["A", "B"]
    );
}

#[tokio::test]
async fn point_scatter_and_lower_bound_reads_use_the_same_state() {
    let mut fixture = Fixture::new(
        64,
        FragmentTreeConfig::new(1024, u32::MAX).with_max_leaf_bytes(4096),
        SnapshotPolicy::default(),
    )
    .await;
    fixture
        .commit(
            vec![
                action::remove_fragment(7),
                action::upsert_fragment(&make_fragment(90)).unwrap(),
            ],
            SnapshotPolicy::default(),
            false,
        )
        .await;
    let bitmap = RoaringBitmap::from_iter([2, 7, 8, 63, 90]);
    let selected = fixture.tree.resolve_fragments(&bitmap).await.unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![2, 8, 63, 90]
    );
    let tree = Arc::new(fixture.tree);
    let from: Vec<_> = tree.clone().scan(62).try_collect().await.unwrap();
    assert_eq!(
        from.iter().map(|fragment| fragment.id).collect::<Vec<_>>(),
        vec![62, 63, 90]
    );
    assert_eq!(tree.resolve_fragment(7).await.unwrap(), None);
}

#[derive(Clone, Copy, Debug)]
enum PendingActions {
    None,
    Root,
    Interior,
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    Dense,
    Sparse,
}

#[derive(Clone, Copy, Debug)]
enum LeafCache {
    Disabled,
    Warm,
}

fn leaf_gets(stats: &lance_io::utils::tracking_store::IoStats) -> usize {
    stats
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with("get_") && request.path.as_ref().contains("_bt/leaf/")
        })
        .count()
}

/// Set reads concatenate per-leaf results instead of merging them into one
/// map, so each leaf must keep its IDs ascending and replay its own routed
/// actions, whether the leaf comes from storage or the session cache.
#[rstest]
#[case::no_actions_dense(PendingActions::None, Selection::Dense)]
#[case::no_actions_sparse(PendingActions::None, Selection::Sparse)]
#[case::root_actions_dense(PendingActions::Root, Selection::Dense)]
#[case::interior_actions_dense(PendingActions::Interior, Selection::Dense)]
#[tokio::test]
async fn resolve_fragments_matches_materialize_in_id_order(
    #[case] pending: PendingActions,
    #[case] selection: Selection,
    #[values(LeafCache::Disabled, LeafCache::Warm)] cache: LeafCache,
) {
    let mut fixture = match pending {
        PendingActions::Interior => interior_cache::buffered_fixture().await,
        PendingActions::None | PendingActions::Root => {
            Fixture::new(
                512,
                FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024),
                SnapshotPolicy::default(),
            )
            .await
        }
    };
    let past_last = fixture.tree.next_fragment_id() + 5;
    if !matches!(pending, PendingActions::None) {
        // The upsert lands above the last leaf's last stored key, inside the
        // routed range of that leaf.
        fixture
            .commit(
                vec![
                    action::remove_fragment(7),
                    action::upsert_fragment(&make_fragment(past_last)).unwrap(),
                ],
                SnapshotPolicy::default(),
                false,
            )
            .await;
    }
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.leaf_keys.len() >= 4, "{shape:?}");
    let interior_actions: u64 = shape.node_buffer_lens.iter().sum();
    match pending {
        PendingActions::None => {
            assert_eq!((shape.root_buffer_len, interior_actions), (0, 0))
        }
        PendingActions::Root => assert!(shape.root_buffer_len > 0, "{shape:?}"),
        PendingActions::Interior => assert!(interior_actions > 0, "{shape:?}"),
    }

    let all = fixture.tree.materialize().await.unwrap();
    let bitmap: RoaringBitmap = match selection {
        Selection::Dense => (0..=past_last as u32).collect(),
        Selection::Sparse => (0..=past_last as u32).step_by(7).collect(),
    };
    let expected: Vec<Fragment> = all
        .into_iter()
        .filter(|fragment| bitmap.contains(fragment.id as u32))
        .collect();
    assert!(expected.len() > 1);
    assert_eq!(
        expected.last().unwrap().id == past_last,
        !matches!(pending, PendingActions::None)
    );
    assert_eq!(
        expected.iter().any(|fragment| fragment.id == 7),
        matches!(pending, PendingActions::None)
    );

    let mut tree = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    if let LeafCache::Warm = cache {
        tree = reopen_tree(
            &tree,
            tree.store
                .context()
                .with_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024)),
        )
        .await;
        // A leaf is admitted on its second whole read, so the third one hits.
        for _ in 0..2 {
            assert_eq!(tree.resolve_fragments(&bitmap).await.unwrap(), expected);
        }
    }
    fixture.io.incremental_stats();
    let resolved = tree.resolve_fragments(&bitmap).await.unwrap();
    let leaf_reads = leaf_gets(&fixture.io.incremental_stats());
    match cache {
        LeafCache::Disabled => assert!(leaf_reads > 0),
        LeafCache::Warm => assert_eq!(leaf_reads, 0),
    }
    assert_eq!(resolved, expected);
    assert!(resolved.windows(2).all(|pair| pair[0].id < pair[1].id));
}

#[tokio::test]
async fn scattered_batches_wait_for_amortization_and_replay_in_order() {
    const ROUNDS: u32 = 24;
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default()
        .with_max_leaf_bytes(16 * 1024)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(512, config.clone(), policy).await;
    assert_eq!(fixture.tree.height(), 1);
    let leaves = fixture.tree.root_child_count();
    assert!(leaves >= 8, "{leaves} leaves");
    let original = fixture.tree.materialize().await.unwrap();
    let historical = fixture.tree.descriptor.clone();
    let mut expected: BTreeMap<u64, Fragment> = original
        .iter()
        .map(|fragment| (fragment.id, fragment.clone()))
        .collect();
    let mut targets = Vec::with_capacity(leaves);
    for child in 0..leaves {
        let id = original
            .iter()
            .map(|fragment| fragment.id)
            .find(|id| node::child_index_for(&fixture.tree.children, *id) == child)
            .unwrap();
        targets.push(id);
    }
    let chained = targets[leaves / 2];
    let mut flushes = 0;
    let mut deferred_rounds = 0;
    for round in 0..ROUNDS {
        let mut actions = Vec::new();
        for &id in &targets {
            let file = make_backfill_data_file(id, round);
            actions.push(action::add_data_file(id, &file));
            expected.get_mut(&id).unwrap().files.push(file);
        }
        let current = expected[&chained].files[0].path.clone();
        let replacement = make_replacement_data_file(chained, round);
        actions.push(action::replace_data_file(chained, &current, &replacement));
        let slot = &mut expected.get_mut(&chained).unwrap().files[0];
        slot.path = replacement.path.clone();
        slot.file_size_bytes = replacement.file_size_bytes;
        slot.base_id = replacement.base_id;

        let stats = fixture.commit(actions, policy, false).await;
        flushes += stats.flushes;
        deferred_rounds += u64::from(stats.flushes == 0);
        assert!(
            !node::internal_overflows(&fixture.tree.children, &fixture.tree.buffer, &config),
            "round {round}: the root must stay within its structural budget"
        );
        assert_eq!(
            fixture.tree.materialize().await.unwrap(),
            expected.values().cloned().collect::<Vec<_>>(),
            "round {round}"
        );
        fixture.tree.validate().await.unwrap();
    }
    assert!(
        deferred_rounds > 0 && flushes >= leaves as u64,
        "every leaf drained at least once and some rounds only buffered: \
         flushes={flushes} deferred_rounds={deferred_rounds}"
    );
    assert!(
        flushes <= (leaves as u64) * u64::from(ROUNDS) / 4,
        "sub-gate batches must not rewrite a leaf every round: flushes={flushes} leaves={leaves}"
    );
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        expected.values().cloned().collect::<Vec<_>>()
    );
    let chained_fragment = reopened.resolve_fragment(chained).await.unwrap().unwrap();
    assert_eq!(
        chained_fragment.files[0].path,
        make_replacement_data_file(chained, ROUNDS - 1).path
    );
    assert_eq!(
        chained_fragment.files[0].fields,
        original[0].files[0].fields
    );
    assert_eq!(chained_fragment.files.len(), 1 + ROUNDS as usize);
    assert_eq!(
        fixture
            .open(&historical, 1)
            .await
            .materialize()
            .await
            .unwrap(),
        original
    );
}

#[rstest]
#[case::inline_root_with_suffix("inline_root_with_suffix")]
#[case::external_root_suffix_zero("external_root_suffix_zero")]
#[case::root_buffer_zero("root_buffer_zero")]
#[case::root_next_zero("root_next_zero")]
#[tokio::test]
async fn sequence_zero_is_rejected_on_open(#[case] shape: &str) {
    let fixture = Fixture::new(4, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let pb::fragment_tree::Root::InlineRoot(root) = fixture.tree.descriptor.root.clone().unwrap()
    else {
        panic!("expected inline root");
    };
    assert_eq!(root.next_action_sequence, 1, "bootstrap starts at 1");
    assert_eq!(root.children[0].materialized_through_action_sequence, 0);
    let zero = pb::FragmentTreeMutation {
        action_sequence: 0,
        action: Some(action::remove_fragment(1)),
        fragment_count_delta: -1,
        total_rows_delta: -1,
        visible_rows_delta: -1,
    };
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    match shape {
        "inline_root_with_suffix" => {
            snapshot.mutations_since_root = vec![zero];
            snapshot.next_action_sequence = 1;
        }
        "external_root_suffix_zero" => {
            let store = NodeStore::new(
                fixture.store.clone(),
                fixture.base.clone(),
                fixture.scheduler.clone(),
            );
            let (path, _) = store.write_root_base(&root).await.unwrap();
            snapshot.root = Some(pb::fragment_tree::Root::RootUuid(path));
            snapshot.mutations_since_root = vec![zero];
            snapshot.next_action_sequence = 1;
        }
        "root_buffer_zero" => {
            let mut root = root;
            root.buffer = vec![zero];
            snapshot.root = Some(pb::fragment_tree::Root::InlineRoot(root));
        }
        _ => {
            let mut root = root;
            root.next_action_sequence = 0;
            snapshot.root = Some(pb::fragment_tree::Root::InlineRoot(root));
            snapshot.next_action_sequence = 0;
        }
    }
    let error = FragmentTree::open_snapshot(
        TreeContext::new(fixture.store, fixture.base, fixture.scheduler),
        &snapshot,
        1,
        fixture.config.clone(),
        fixture.tree.next_fragment_id(),
    )
    .await
    .err()
    .expect("sequence 0 must not open");
    assert!(
        matches!(
            error,
            Error::CorruptFile { .. } | Error::InvalidInput { .. }
        ),
        "{error}"
    );
}

#[tokio::test]
async fn first_fences_follow_the_parent_not_the_first_stored_id() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(512, u32::MAX)
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(64, config, policy).await;
    assert!(fixture.tree.height() >= 2);
    assert_eq!(fixture.tree.children[0].min_key, 0);
    let actions: Vec<_> = (0..6).map(action::remove_fragment).collect();
    fixture.commit(actions, policy, false).await;
    fixture.commit(Vec::new(), policy, true).await;
    assert_eq!(fixture.tree.children[0].min_key, 0);
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    let fragments = reopened.materialize().await.unwrap();
    assert_eq!(fragments[0].id, 6);
    reopened.validate().await.unwrap();
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    if let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root {
        root.children[0].min_key = 6;
    } else {
        panic!("expected inline root");
    }
    let error = FragmentTree::open_snapshot(
        TreeContext::new(
            fixture.store.clone(),
            fixture.base.clone(),
            fixture.scheduler.clone(),
        ),
        &snapshot,
        fixture.tree.version(),
        fixture.config.clone(),
        fixture.tree.next_fragment_id(),
    )
    .await
    .err()
    .expect("a first fence above the parent's must not open");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// Removing everything below the last routing child empties the leading
/// subtrees in one drain. The surviving interior inherits the root's lower
/// bound, so the fence stored on its own first child must move with it.
#[tokio::test]
async fn draining_leading_subtrees_moves_the_surviving_interior_fence() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(512, u32::MAX)
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(512, config, policy).await;
    assert!(fixture.tree.height() >= 2);
    let survivor = fixture.tree.children.last().unwrap();
    assert!(survivor.height >= 1 && survivor.min_key > 0);
    let first_survivor = survivor.min_key;
    let actions: Vec<_> = (0..first_survivor).map(action::remove_fragment).collect();
    fixture.commit(actions, policy, false).await;

    assert_eq!(fixture.tree.children[0].min_key, 0);
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    assert!(reopened.buffered_action_keys().await.unwrap().is_empty());
    let ids: Vec<_> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    assert_eq!(ids, (first_survivor..512).collect::<Vec<_>>());
}

/// A create buffered for a leaf that later loses every stored fragment lands
/// in that leaf's replacement, never in a sibling whose watermark already
/// covers the create.
#[tokio::test]
async fn emptied_leaf_keeps_its_buffered_creates_under_its_own_watermark() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024);
    let mut fixture = Fixture::new(256, config, policy).await;
    assert_eq!(fixture.tree.height(), 1);
    assert!(fixture.tree.root_child_count() >= 2);
    let last = fixture.tree.children.last().unwrap().min_key;
    let second = fixture.tree.children[1].min_key;
    let fresh = fixture.tree.next_fragment_id();
    fixture
        .commit(
            vec![action::upsert_fragment(&make_fragment(fresh)).unwrap()],
            policy,
            false,
        )
        .await;
    assert_eq!(fixture.tree.root_buffer_len(), 1);
    // Removing a quarter of the first leaf drains only that leaf.
    let removed = second / 4;
    fixture
        .commit(
            (0..removed).map(action::remove_fragment).collect(),
            policy,
            false,
        )
        .await;
    assert_eq!(fixture.tree.root_buffer_len(), 1);
    fixture
        .commit(
            (last..fresh).map(action::remove_fragment).collect(),
            policy,
            false,
        )
        .await;

    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    let ids: Vec<_> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    let expected: Vec<_> = (removed..last).chain([fresh]).collect();
    assert_eq!(ids, expected);
}

/// One commit shrinks the first routing child to a single leaf while appends
/// split the last child past the fanout, so the root must split too. The
/// shrunken child is repaired before the root splits, or a split piece would
/// persist a single-child interior that no reader accepts.
#[tokio::test]
async fn contraction_and_root_split_in_one_commit_keep_nodes_readable() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(64 * 1024, 4)
        .with_max_leaf_bytes(16 * 1024)
        .with_semantic_buffer_bytes(1);
    let mut full_root = None;
    for count in (64..4096).step_by(32) {
        let fixture = Fixture::new(count, config.clone(), policy).await;
        if fixture.tree.height() == 2 && fixture.tree.root_child_count() == 4 {
            full_root = Some(fixture);
            break;
        }
    }
    let mut fixture = full_root.expect("some table size gives a two-level tree with a full root");
    let first = fixture
        .tree
        .store
        .read_internal(&fixture.tree.children[0])
        .await
        .unwrap();
    assert!(
        first.children.len() >= 2,
        "the first child needs several leaves"
    );
    let last_leaf_start = first.children.last().unwrap().min_key;
    let fresh = fixture.tree.next_fragment_id();
    let appended = fresh;
    let mut actions: Vec<_> = (0..last_leaf_start).map(action::remove_fragment).collect();
    actions.extend(
        (fresh..fresh + appended).map(|id| action::upsert_fragment(&make_fragment(id)).unwrap()),
    );
    fixture.commit(actions, policy, false).await;

    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    let ids: Vec<_> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    let expected: Vec<_> = (last_leaf_start..fresh + appended).collect();
    assert_eq!(ids, expected);
}

/// A point lookup decodes one row of its leaf, yet returns exactly what a
/// whole-leaf read holds, for every id and for an id past the last one, with
/// buffered actions replayed over it. Two whole reads admit every leaf; later
/// point reads use that cache without reading any leaf again.
#[rstest]
#[case::several_leaves(300, 1, FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024))]
#[case::one_leaf(300, 1, FragmentTreeConfig::default().with_max_leaf_bytes(4 * 1024 * 1024))]
#[case::interior_buffers(2048, 40, FragmentTreeConfig {
    max_node_bytes: 4 * 1024,
    max_leaf_bytes: 16 * 1024,
    semantic_buffer_bytes: 2 * 1024,
    ..FragmentTreeConfig::default()
})]
#[tokio::test]
async fn point_reads_match_whole_leaf_reads(
    #[case] count: u64,
    #[case] rounds: u64,
    #[case] config: FragmentTreeConfig,
) {
    let mut fixture = Fixture::new(count, config, SnapshotPolicy::default()).await;
    for round in 1..=rounds {
        let actions = (0..16)
            .map(|slot| {
                let id = (round * 16 + slot * 127) % count;
                action::add_data_file(id, &make_backfill_data_file(id, round as u32))
            })
            .collect();
        fixture
            .commit(actions, SnapshotPolicy::default(), false)
            .await;
    }
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.root_buffer_len > 0, "{shape:?}");
    assert!(
        shape.height < 2 || shape.node_buffer_lens.iter().sum::<u64>() > 0,
        "{shape:?}"
    );
    let whole = fixture.tree.materialize().await.unwrap();
    let mut cached = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    cached = reopen_tree(
        &cached,
        cached
            .store
            .context()
            .with_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024)),
    )
    .await;
    fixture.io.incremental_stats();
    for _ in 0..2 {
        assert_eq!(cached.materialize().await.unwrap(), whole);
    }
    for fragment in &whole {
        for _ in 0..3 {
            let point = cached.resolve_fragment(fragment.id).await.unwrap();
            assert_eq!(point.as_ref(), Some(fragment));
        }
    }
    let leaf_reads = fixture
        .io
        .incremental_stats()
        .requests
        .into_iter()
        .filter(|request| {
            request.method.starts_with("get_") && request.path.as_ref().contains("_bt/leaf/")
        })
        .count();
    assert_eq!(leaf_reads, 2 * shape.leaf_keys.len(), "{shape:?}");
    for fragment in &whole {
        assert_eq!(
            fixture
                .tree
                .resolve_fragment(fragment.id)
                .await
                .unwrap()
                .as_ref(),
            Some(fragment)
        );
    }
    assert!(cached.resolve_fragment(count).await.unwrap().is_none());
    assert!(
        fixture
            .tree
            .resolve_fragment(count)
            .await
            .unwrap()
            .is_none()
    );
}

/// A batch that overwrites every record of every leaf determines the new
/// leaves, so the drain reads each old leaf once, for the record headers that
/// check the stored deltas, and never decodes its files.
#[tokio::test]
async fn wholly_replaced_leaves_rebuild_from_record_headers() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default()
        .with_max_leaf_bytes(16 * 1024)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(256, config, policy).await;
    let leaves = fixture.tree.root_child_count();
    assert!(leaves >= 2);
    assert_eq!(fixture.tree.height(), 1);
    let expected: Vec<_> = (0..256).map(|id| make_fragment_with_files(id, 2)).collect();
    let actions: Vec<_> = expected
        .iter()
        .map(|fragment| action::upsert_fragment(fragment).unwrap())
        .collect();
    let ids: Vec<_> = (0..256).collect();
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&io);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let mut transaction = fixture.tree.begin_commit(CommitMode::Incremental);
    transaction.resolve_touched(&ids).await.unwrap();
    io.incremental_stats();
    let (prepared, _) = transaction
        .prepare(ValidatedCommit::fragment_actions(actions), policy)
        .await
        .unwrap();

    let leaf_reads: Vec<_> = io
        .incremental_stats()
        .requests
        .into_iter()
        .filter(|request| {
            request.method.starts_with("get") && request.path.as_ref().contains("_bt/leaf/")
        })
        .collect();
    assert_eq!(leaf_reads.len(), leaves, "{leaf_reads:?}");
    fixture.tree = prepared;
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    assert_eq!(reopened.materialize().await.unwrap(), expected);
}

/// Check removals against the stored record headers before rebuilding or
/// retiring a leaf.
#[rstest]
#[case::rebuilt_leaf(true)]
#[case::dropped_leaf(false)]
#[tokio::test]
async fn drain_rejects_a_removal_of_a_fragment_the_leaf_lacks(#[case] creates: bool) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(8, config, policy).await;
    fixture
        .commit(vec![action::remove_fragment(7)], policy, true)
        .await;
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].num_keys, 7);
    // Fragments 0 to 5 are removed, and so is fragment 7, which the leaf no
    // longer holds. Fragment 6 is never named.
    let mut actions: Vec<_> = (0..6)
        .chain([7])
        .map(|id| (action::remove_fragment(id), -1))
        .collect();
    if creates {
        actions.push((action::upsert_fragment(&make_fragment(8)).unwrap(), 1));
    }
    for (action, delta) in actions {
        root.buffer.push(pb::FragmentTreeMutation {
            action_sequence: root.next_action_sequence,
            action: Some(action),
            fragment_count_delta: delta,
            total_rows_delta: delta,
            visible_rows_delta: delta,
        });
        root.next_action_sequence += 1;
    }
    snapshot.next_action_sequence = root.next_action_sequence;
    let tree = FragmentTree::open_snapshot(
        TreeContext::new(
            fixture.store.clone(),
            fixture.base.clone(),
            fixture.scheduler.clone(),
        ),
        &snapshot,
        fixture.tree.version(),
        fixture.config.clone(),
        9,
    )
    .await
    .unwrap();
    let error = tree
        .begin_commit(CommitMode::Incremental)
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .err()
        .expect("a removal of an absent fragment must not retire the leaf");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// Check removals against the stored records before retiring an entire subtree.
#[tokio::test]
async fn drain_rejects_a_removal_that_would_drop_a_subtree_it_does_not_cover() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(512, u32::MAX)
        .with_max_leaf_bytes(6 * 1024)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(64, config, policy).await;
    fixture
        .commit(vec![action::remove_fragment(0)], policy, true)
        .await;
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    assert!(root.children[0].height > 0);
    let end = root.children[1].min_key;
    assert_eq!(root.children[0].num_keys, end - 1);
    // Every stored fragment of the first subtree is removed except the last,
    // and fragment 0, which the subtree no longer holds, is removed too.
    for id in (0..end - 1).collect::<Vec<_>>() {
        root.buffer.push(pb::FragmentTreeMutation {
            action_sequence: root.next_action_sequence,
            action: Some(action::remove_fragment(id)),
            fragment_count_delta: -1,
            total_rows_delta: -1,
            visible_rows_delta: -1,
        });
        root.next_action_sequence += 1;
    }
    snapshot.next_action_sequence = root.next_action_sequence;
    let tree = fixture.open(&snapshot, fixture.tree.version()).await;
    let error = tree
        .begin_commit(CommitMode::Incremental)
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .err()
        .expect("a removal of an absent fragment must not retire the subtree");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// A drain whose batch claims to replace every stored record still checks
/// each stored delta against the record headers. A buffer that counts an
/// insert as a replacement would otherwise drop a fragment nothing targets.
#[tokio::test]
async fn drain_rejects_a_replacement_that_misstates_its_delta() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(8, config, policy).await;
    fixture
        .commit(vec![action::remove_fragment(7)], policy, true)
        .await;
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    assert_eq!(root.children.len(), 1);
    assert_eq!(root.children[0].num_keys, 7);
    // Six stored fragments are replaced, and fragment 7, which the leaf no
    // longer holds, is inserted with the zero delta of a replacement.
    for id in (0..6).chain([7]) {
        root.buffer.push(pb::FragmentTreeMutation {
            action_sequence: root.next_action_sequence,
            action: Some(action::upsert_fragment(&make_fragment_with_files(id, 2)).unwrap()),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        });
        root.next_action_sequence += 1;
    }
    snapshot.next_action_sequence = root.next_action_sequence;
    let tree = fixture.open(&snapshot, fixture.tree.version()).await;
    let error = tree
        .begin_commit(CommitMode::Incremental)
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .err()
        .expect("a misstated replacement must not rebuild the leaf");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// Removing all but one fragment from a tree three or more levels deep leaves
/// single-child interiors that no sibling can absorb. The commit rebuilds in
/// bulk rather than persisting that chain.
#[tokio::test]
async fn collapsing_to_one_leaf_rebuilds_instead_of_persisting_a_chain() {
    let policy = SnapshotPolicy {
        inline_root_bytes: 0,
        max_suffix_bytes: 0,
    };
    let config = FragmentTreeConfig::new(512, u32::MAX)
        .with_max_leaf_bytes(1024)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(512, config, policy).await;
    assert!(
        fixture.tree.height() >= 3,
        "height {}",
        fixture.tree.height()
    );
    let survivor = 300;
    let actions: Vec<_> = (0..512)
        .filter(|id| *id != survivor)
        .map(action::remove_fragment)
        .collect();
    fixture.io.incremental_stats();
    let stats = fixture.commit(actions, policy, false).await;
    assert_eq!(
        stats.tree_write_bytes,
        fixture.io.stats().written_bytes,
        "{stats:?}"
    );

    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    assert_eq!(reopened.height(), 1);
    let ids: Vec<_> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    assert_eq!(ids, vec![survivor]);
}

/// Regrowth from a childless root lifts fresh leaves into interior nodes once
/// they exceed the fanout. An interior node stores its children's fences and a
/// reader checks the first stored fence against the parent's entry, so the
/// first leaf must carry the root's lower bound before the lift. A first live
/// id above zero is what exposes the difference.
#[tokio::test]
async fn childless_root_regrowth_that_splits_the_root_reopens() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default()
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(0, config, policy).await;
    assert!(fixture.tree.children.is_empty());
    let ids: Vec<u64> = (100..120).collect();
    let actions: Vec<_> = ids
        .iter()
        .map(|id| action::upsert_fragment(&make_fragment(*id)).unwrap())
        .collect();
    fixture.commit(actions, policy, false).await;
    assert!(
        fixture.tree.height() >= 2,
        "twenty singleton leaves must split the root at fanout {}",
        fixture.config.max_children_per_node
    );
    assert_eq!(fixture.tree.children[0].min_key, 0);
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    reopened.validate().await.unwrap();
    let regrown: Vec<u64> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    assert_eq!(regrown, ids);
}

/// Reads and validation reject a buffered action at or below its leaf's
/// watermark, including a point read whose newest action replaces the whole
/// record.
#[rstest]
#[case::point("point")]
#[case::point_after_reset("point_after_reset")]
#[case::offset("offset")]
#[case::fixed_readahead_scan("fixed_readahead_scan")]
#[case::serial_scan("serial_scan")]
#[case::ramped_scan("ramped_scan")]
#[case::resolve_fragments("resolve_fragments")]
#[case::resolve_touched("resolve_touched")]
#[case::materialize("materialize")]
#[case::validate("validate")]
#[tokio::test]
async fn mutation_at_or_below_its_leaf_watermark_is_rejected(#[case] read: &str) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default()
        .with_max_leaf_bytes(4096)
        .with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(8, config, policy).await;
    let target = 4;
    fixture
        .commit(
            vec![action::upsert_fragment(&make_fragment_with_files(target, 4)).unwrap()],
            policy,
            true,
        )
        .await;
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    let leaf = &root.children[node::child_index_for(&root.children, target)];
    assert!(leaf.materialized_through_action_sequence >= 1);
    let stale = if read == "point_after_reset" {
        action::upsert_fragment(&make_fragment_with_files(target, 4)).unwrap()
    } else {
        action::clear_deletion_file(target)
    };
    root.buffer.push(pb::FragmentTreeMutation {
        action_sequence: leaf.materialized_through_action_sequence,
        action: Some(stale),
        fragment_count_delta: 0,
        total_rows_delta: 0,
        visible_rows_delta: 0,
    });
    let tree = fixture.open(&snapshot, fixture.tree.version()).await;
    let error = match read {
        "point" | "point_after_reset" => tree.resolve_fragment(target).await.err(),
        "resolve_fragments" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([target as u32]))
            .await
            .err(),
        "resolve_touched" => tree
            .begin_commit(CommitMode::Incremental)
            .resolve_touched(&[target])
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "validate" => tree.validate().await.err(),
        "fixed_readahead_scan" => Arc::new(tree)
            .scan_with_readahead(0, NonZeroUsize::new(4).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "serial_scan" => Arc::new(tree)
            .scan_with_readahead(0, NonZeroUsize::MIN)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "ramped_scan" => Arc::new(tree).scan(0).try_collect::<Vec<_>>().await.err(),
        _ => tree.fragment_at_row_offset(target).await.err(),
    }
    .expect("a mutation at the watermark must not replay");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("watermark"), "{error}");
}

/// With a node budget that fits exactly two child references per split piece,
/// greedy packing leaves a one-child piece at the right edge. No split may
/// persist it, since the format rejects single-child interiors on read.
#[rstest]
#[case::bootstrap("bootstrap")]
#[case::bulk_rebuild("bulk")]
#[case::two_level_lift("buffered")]
#[case::interior_drain("interior")]
#[tokio::test]
async fn splits_never_persist_a_single_child_interior(#[case] path: &str) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(300, 16).with_max_leaf_bytes(1);
    let (count, added) = match path {
        "bootstrap" => (7, 0),
        "bulk" => (4, 1),
        "interior" => (8, 7),
        _ => (4, 7),
    };
    let mut fixture = Fixture::new(count, config, policy).await;
    if path == "interior" {
        assert!(fixture.tree.height() >= 2);
    }
    if added > 0 {
        let actions = (count..count + added)
            .map(|id| action::upsert_fragment(&make_fragment(id)).unwrap())
            .collect();
        let stats = fixture.commit(actions, policy, path == "bulk").await;
        if path == "interior" {
            assert!(stats.max_flush_depth > 0);
            assert!(stats.splits > 0);
        }
    }
    let reopened = fixture
        .open(&fixture.tree.descriptor, fixture.tree.version())
        .await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (0..count + added).map(make_fragment).collect::<Vec<_>>()
    );
    reopened.validate().await.unwrap();
    assert!(
        reopened
            .shape_report()
            .await
            .unwrap()
            .node_fanouts
            .iter()
            .all(|fanout| (2..=16).contains(fanout))
    );
}

/// A node budget whose split piece holds one child reference cannot build
/// routing. Bootstrap and a root lift refuse it instead of publishing
/// single-child interiors that no reader can open.
#[rstest]
#[case::bootstrap("bootstrap")]
#[case::root_lift("lift")]
#[tokio::test]
async fn budget_that_cannot_pair_child_references_is_refused(#[case] path: &str) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(200, 16).with_max_leaf_bytes(1);
    let error = if path == "bootstrap" {
        let (store, base) = ObjectStore::from_uri("memory://").await.unwrap();
        let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::default_for_testing());
        FragmentTree::bootstrap_snapshot(
            TreeContext::new(store, base, scheduler),
            config,
            &mut (0..4).map(make_fragment).collect::<Vec<_>>(),
            1,
            policy,
        )
        .await
        .err()
    } else {
        let fixture = Fixture::new(2, config, policy).await;
        let actions: Vec<_> = (2..4)
            .map(|id| action::upsert_fragment(&make_fragment(id)).unwrap())
            .collect();
        fixture
            .tree
            .begin_commit(CommitMode::Incremental)
            .prepare(ValidatedCommit::fragment_actions(actions), policy)
            .await
            .err()
    }
    .expect("a budget that cannot pair child references must be refused");
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
}

/// A drain rejects a leaf that stores a fragment at or above the end of the
/// range its parent assigned. Rewriting it would otherwise publish a piece
/// fenced past its right sibling, or keep a fragment point reads never find.
#[tokio::test]
async fn drain_rejects_a_leaf_storing_keys_past_its_range_end() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(1, config, policy).await;
    let mut leaves = Vec::new();
    for ids in [vec![0], vec![60, 150], vec![100, 120]] {
        let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
        let written = fixture
            .tree
            .store
            .write_leaves(&fragments, None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    // The middle leaf owns [50, 100) but stores 150, which its sibling's
    // range covers.
    leaves[1].min_key = 50;
    let tree = &mut fixture.tree;
    tree.children = leaves;
    // Enough inserts into the middle leaf's range that draining it pays.
    tree.buffer = (61..91)
        .zip(1..)
        .map(|(id, action_sequence)| pb::FragmentTreeMutation {
            action_sequence,
            action: Some(action::upsert_fragment(&make_fragment(id)).unwrap()),
            fragment_count_delta: 1,
            total_rows_delta: 1,
            visible_rows_delta: 1,
        })
        .collect();
    tree.buffer_index.take();
    tree.total_fragments = 35;
    tree.total_rows = 35;
    tree.visible_rows = 35;
    tree.next_action_sequence = 31;
    tree.store.next_action_sequence = 31;
    let error = tree
        .rewrite_tree(CommitMode::Incremental)
        .await
        .expect_err("a leaf storing keys past its range must not be rewritten");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

#[rstest]
#[case::point("point")]
#[case::set("set")]
#[case::materialize("materialize")]
#[case::fixed_readahead_scan("fixed_readahead_scan")]
#[case::ramped_scan("ramped_scan")]
#[case::row_offset("row_offset")]
#[case::bulk("bulk")]
#[case::coalesce("coalesce")]
#[case::validate("validate")]
#[tokio::test]
async fn reads_reject_a_leaf_storing_keys_past_its_range_end(
    #[case] read: &str,
    #[values(false, true)] deep: bool,
    #[values("cold", "cached", "retained")] reads: &str,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(5, FragmentTreeConfig::default(), policy).await;
    let mut leaves = Vec::new();
    for ids in [vec![0], vec![60, 150], vec![100], vec![120]] {
        let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
        let written = fixture
            .tree
            .store
            .write_leaves(&fragments, None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    // The middle leaf owns [50, 100) but stores fragment 150.
    leaves[1].min_key = 50;
    if reads == "cached" {
        fixture.tree = reopen_tree(
            &fixture.tree,
            fixture
                .context()
                .with_leaf_cache(LanceCache::with_capacity(1024 * 1024)),
        )
        .await;
    }
    let tree = &mut fixture.tree;
    if reads == "retained" {
        tree.store.retain_validation_reads().unwrap();
    }
    if reads != "cold" {
        let mut wide = tree.clone();
        wide.children = leaves[..2].to_vec();
        wide.total_fragments = 3;
        wide.total_rows = 3;
        wide.visible_rows = 3;
        wide.next_fragment_id = 151;
        for _ in 0..2 {
            let fragments = wide
                .resolve_fragments(&RoaringBitmap::from_iter([60]))
                .await
                .unwrap();
            assert_eq!(fragments, vec![make_fragment(60)]);
        }
        fixture.io.incremental_stats();
        wide.resolve_fragments(&RoaringBitmap::from_iter([60]))
            .await
            .unwrap();
        assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    }
    tree.children = if deep {
        let left = tree
            .store
            .write_internal(leaves[..2].to_vec(), Vec::new())
            .await
            .unwrap();
        let right = tree
            .store
            .write_internal(leaves[2..].to_vec(), Vec::new())
            .await
            .unwrap();
        vec![left.child_ref, right.child_ref]
    } else {
        leaves
    };
    tree.next_fragment_id = 151;
    tree.buffer.clear();
    tree.buffer_index.take();
    if read == "bulk" {
        tree.buffer.push(pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::clear_deletion_file(60)),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        });
        tree.next_action_sequence = 2;
        tree.store.next_action_sequence = 2;
    }
    let error = match read {
        "point" => tree.resolve_fragment(60).await.err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([60]))
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "fixed_readahead_scan" => Arc::new(fixture.tree)
            .scan_with_readahead(0, NonZeroUsize::new(4).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "ramped_scan" => Arc::new(fixture.tree)
            .scan(0)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "row_offset" => tree.fragment_at_row_offset(1).await.err(),
        "validate" => {
            describe_in_memory_root(tree);
            tree.validate().await.err()
        }
        _ => tree.rewrite_tree(commit_mode(read == "bulk")).await.err(),
    }
    .expect("a leaf storing keys past its range must not be read");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

#[rstest]
#[case::point("point")]
#[case::set("set")]
#[case::materialize("materialize")]
#[case::fixed_readahead_scan("fixed_readahead_scan")]
#[case::row_offset("row_offset")]
#[case::bulk("bulk")]
#[case::coalesce("coalesce")]
#[case::validate("validate")]
#[tokio::test]
async fn reads_reject_child_fences_past_the_parent_range(
    #[case] read: &str,
    #[values(false, true)] cached: bool,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(5, FragmentTreeConfig::default(), policy).await;
    if cached {
        fixture.tree = reopen_tree(
            &fixture.tree,
            fixture
                .context()
                .with_leaf_cache(LanceCache::with_capacity(1024 * 1024)),
        )
        .await;
    }
    let tree = &mut fixture.tree;
    let mut leaves = Vec::new();
    for ids in [vec![0, 110], vec![150], vec![100], vec![120]] {
        let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
        leaves.push(
            tree.store
                .write_leaf(&fragments, 0)
                .await
                .unwrap()
                .child_ref,
        );
    }
    let left = tree
        .store
        .write_internal(leaves[..2].to_vec(), Vec::new())
        .await
        .unwrap();
    let right = tree
        .store
        .write_internal(leaves[2..].to_vec(), Vec::new())
        .await
        .unwrap();
    tree.children = vec![left.child_ref, right.child_ref];
    tree.next_fragment_id = 151;
    tree.buffer.clear();
    tree.buffer_index.take();
    if cached {
        let mut wide = tree.clone();
        wide.children.truncate(1);
        assert_eq!(wide.resolve_fragment(0).await.unwrap().unwrap().id, 0);
    }
    if read == "bulk" {
        tree.buffer.push(pb::FragmentTreeMutation {
            action_sequence: 1,
            action: Some(action::clear_deletion_file(0)),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        });
        tree.next_action_sequence = 2;
        tree.store.next_action_sequence = 2;
    }
    // The left interior ends at 100, so its child fence at 150 is invalid.
    let error = match read {
        "point" => tree.resolve_fragment(0).await.err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([0]))
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "fixed_readahead_scan" => Arc::new(fixture.tree)
            .scan_with_readahead(0, NonZeroUsize::new(4).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "row_offset" => tree.fragment_at_row_offset(0).await.err(),
        "validate" => {
            describe_in_memory_root(tree);
            tree.validate().await.err()
        }
        _ => tree.rewrite_tree(commit_mode(read == "bulk")).await.err(),
    }
    .expect("a child fence must stay within its parent's range");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

/// Writes one leaf per id group, each materialized through sequence 0, under
/// a new interior that buffers `buffer`.
async fn write_leaf_interior(
    tree: &FragmentTree,
    leaf_ids: &[&[u64]],
    buffer: Vec<pb::FragmentTreeMutation>,
) -> pb::FragmentTreeChild {
    let mut leaves = Vec::new();
    for ids in leaf_ids {
        let fragments: Vec<_> = ids.iter().copied().map(make_fragment).collect();
        leaves.push(
            tree.store
                .write_leaf(&fragments, 0)
                .await
                .unwrap()
                .child_ref,
        );
    }
    tree.store
        .write_internal(leaves, buffer)
        .await
        .unwrap()
        .child_ref
}

fn buffered_file(id: u64, col: u32, action_sequence: u64) -> pb::FragmentTreeMutation {
    pb::FragmentTreeMutation {
        action_sequence,
        action: Some(action::add_data_file(id, &make_backfill_data_file(id, col))),
        fragment_count_delta: 0,
        total_rows_delta: 0,
        visible_rows_delta: 0,
    }
}

fn route_root_to(
    tree: &mut FragmentTree,
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
    next_action_sequence: u64,
) {
    tree.children = children;
    tree.buffer = buffer;
    tree.buffer_index.take();
    tree.next_action_sequence = next_action_sequence;
    tree.store.next_action_sequence = next_action_sequence;
    describe_in_memory_root(tree);
}

/// Point the descriptor at the in-memory root so `validate` audits the
/// routing a test wrote.
fn describe_in_memory_root(tree: &mut FragmentTree) {
    tree.descriptor = Arc::new(pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(tree.compacted_root())),
        mutations_since_root: Vec::new(),
        next_action_sequence: tree.next_action_sequence,
    });
}

/// Sibling interiors under one parent are independent once the parent is
/// verified, so a set resolution reads them together while every covering
/// node is still read once and interior buffers still replay.
#[tokio::test]
async fn set_resolution_overlaps_sibling_interior_reads() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    // Eight level-1 interiors of two leaves each, [4j, 4j + 1] and
    // [4j + 2, 4j + 3], under two level-2 interiors of four.
    let mut level_one = Vec::new();
    for j in 0..8u64 {
        let first = 4 * j;
        level_one.push(
            write_leaf_interior(
                &fixture.tree,
                &[&[first, first + 1], &[first + 2, first + 3]],
                vec![buffered_file(first, 0, j + 1)],
            )
            .await,
        );
    }
    let mut level_two = Vec::new();
    for group in level_one.chunks(4) {
        level_two.push(
            fixture
                .tree
                .store
                .write_internal(group.to_vec(), Vec::new())
                .await
                .unwrap()
                .child_ref,
        );
    }
    route_root_to(&mut fixture.tree, level_two, Vec::new(), 9);
    fixture.tree.next_fragment_id = 32;
    assert_eq!(fixture.tree.height(), 3);

    let delayed = FailpointController::default();
    delayed.set_get_latency(std::time::Duration::from_millis(2));
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    store.apply_wrapper(&io);
    let tree = fixture.tree.with_object_store(Arc::new(store));
    // One id per level-1 interior, each with a pending action in that interior.
    let requested = RoaringBitmap::from_iter((0..8u32).map(|j| 4 * j));
    let resolved = tree.resolve_fragments(&requested).await.unwrap();

    let expected: Vec<_> = (0..8u64)
        .map(|j| {
            let mut fragment = make_fragment(4 * j);
            fragment.files.push(make_backfill_data_file(4 * j, 0));
            fragment
        })
        .collect();
    assert_eq!(resolved, expected);
    let measured = io.incremental_stats();
    let paths = |class: &str| -> Vec<String> {
        measured
            .requests
            .iter()
            .filter(|request| {
                request.path.as_ref().contains(class) && request.method.contains("get")
            })
            .map(|request| request.path.to_string())
            .collect()
    };
    let nodes = paths("_bt/node/");
    let leaves = paths("_bt/leaf/");
    assert_eq!(nodes.len(), 10, "every covering interior: {nodes:?}");
    assert_eq!(leaves.len(), 8, "one covering leaf per level-1 interior");
    assert_eq!(nodes.iter().collect::<HashSet<_>>().len(), nodes.len());
    assert_eq!(leaves.iter().collect::<HashSet<_>>().len(), leaves.len());
    assert_eq!(measured.read_iops, 18);
    // A serial discovery walk alone takes one stage per interior.
    assert!(
        measured.num_stages < nodes.len() as u64,
        "sibling interior reads must overlap: {measured:?}"
    );
}

/// Where the corrupt interior sits relative to the missing right sibling.
#[derive(Debug, Clone, Copy)]
enum CorruptPlacement {
    LeftSibling,
    /// The right sibling's read fails before the walk reaches the corrupt
    /// node below a valid left sibling, so surfacing errors in read order
    /// would report the missing right sibling instead.
    LeftDescendant,
}

/// Concurrent sibling reads still report the error a serial walk meets
/// first, not the first read to fail.
#[rstest]
#[tokio::test]
async fn set_resolution_reports_the_first_corrupt_interior_in_walk_order(
    #[values(CorruptPlacement::LeftSibling, CorruptPlacement::LeftDescendant)]
    placement: CorruptPlacement,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    // The corrupt interior ends at 100, so its child fence at 150 is invalid.
    let corrupt = write_leaf_interior(&fixture.tree, &[&[0, 110], &[150]], Vec::new()).await;
    let valid = write_leaf_interior(&fixture.tree, &[&[100], &[120]], Vec::new()).await;
    let (left, right) = match placement {
        CorruptPlacement::LeftSibling => (corrupt, valid),
        CorruptPlacement::LeftDescendant => {
            let store = &fixture.tree.store;
            let left = store.write_internal(vec![corrupt], Vec::new()).await;
            let right = store.write_internal(vec![valid], Vec::new()).await;
            (left.unwrap().child_ref, right.unwrap().child_ref)
        }
    };
    let location = Path::from(format!("{}/{}", fixture.base, right.path));
    fixture.store.inner.delete(&location).await.unwrap();
    route_root_to(&mut fixture.tree, vec![left, right], Vec::new(), 1);
    fixture.tree.next_fragment_id = 151;

    let error = fixture
        .tree
        .resolve_fragments(&RoaringBitmap::from_iter([0, 120]))
        .await
        .expect_err("a child fence must stay within its parent's range");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

/// The duplicate sequence check covers the actions routed to one leaf from
/// the root and every interior on its path.
#[tokio::test]
async fn set_resolution_rejects_a_duplicate_sequence_on_one_replay_path() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(0, FragmentTreeConfig::default(), policy).await;
    let left = write_leaf_interior(
        &fixture.tree,
        &[&[0, 1], &[2]],
        vec![buffered_file(0, 0, 1)],
    )
    .await;
    let right = write_leaf_interior(&fixture.tree, &[&[100], &[120]], Vec::new()).await;
    route_root_to(
        &mut fixture.tree,
        vec![left, right],
        vec![buffered_file(0, 1, 1)],
        2,
    );
    fixture.tree.next_fragment_id = 121;

    let error = fixture
        .tree
        .resolve_fragments(&RoaringBitmap::from_iter([0, 120]))
        .await
        .expect_err("two actions on one replay path must not share a sequence");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("another buffer read by this operation"),
        "{error}"
    );
}

/// Reads, rewrites and validation reject an interior action targeting a
/// sibling's range, including point and set reads that name none of the
/// forged targets and a rewrite with nothing to drain.
#[rstest]
#[case::full_read("materialize")]
#[case::fixed_readahead_scan("fixed_readahead_scan")]
#[case::row_offset("row_offset")]
#[case::drain("drain")]
#[case::bulk_drain("bulk")]
#[case::coalesce("coalesce")]
#[case::point("point")]
#[case::point_below_range("point_below_range")]
#[case::cached_point("cached_point")]
#[case::set("set")]
#[case::touched("touched")]
#[case::validate("validate")]
#[tokio::test]
async fn interior_action_outside_its_range_is_rejected(#[case] read: &str) {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(5, config, policy).await;
    let mut leaves = Vec::new();
    for ids in [vec![0], vec![1, 2], vec![3], vec![4]] {
        let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
        let written = fixture
            .tree
            .store
            .write_leaves(&fragments, None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    // The left interior owns [0, 3) but buffers an insert of fragment 3,
    // which the right interior holds.
    let mut duplicate = make_fragment(3);
    duplicate.files[0].path = "duplicate.lance".into();
    let misrouted = pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action::upsert_fragment(&duplicate).unwrap()),
        fragment_count_delta: 1,
        total_rows_delta: 1,
        visible_rows_delta: 1,
    };
    let left = fixture
        .tree
        .store
        .write_internal(leaves[..2].to_vec(), vec![misrouted])
        .await
        .unwrap();
    // The right interior owns [3, 2^32) but buffers a file for fragment 2,
    // which the left interior holds.
    let right = fixture
        .tree
        .store
        .write_internal(leaves[2..].to_vec(), vec![buffered_file(2, 0, 2)])
        .await
        .unwrap();
    if read == "cached_point" {
        fixture.tree = reopen_tree(
            &fixture.tree,
            fixture
                .context()
                .with_leaf_cache(LanceCache::with_capacity(1024 * 1024)),
        )
        .await;
    }
    let tree = &mut fixture.tree;
    tree.children = vec![left.child_ref, right.child_ref];
    tree.buffer.clear();
    tree.buffer_index.take();
    tree.total_fragments = 6;
    tree.total_rows = 6;
    tree.visible_rows = 6;
    tree.next_action_sequence = 3;
    tree.store.next_action_sequence = 3;
    if matches!(read, "drain" | "bulk") {
        tree.buffer.push(pb::FragmentTreeMutation {
            action_sequence: 3,
            action: Some(action::remove_fragment(2)),
            fragment_count_delta: -1,
            total_rows_delta: -1,
            visible_rows_delta: -1,
        });
        tree.next_action_sequence = 4;
        tree.store.next_action_sequence = 4;
    }
    let error = match read {
        "fixed_readahead_scan" => Arc::new(fixture.tree)
            .scan_with_readahead(0, NonZeroUsize::new(4).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "row_offset" => tree.fragment_at_row_offset(0).await.err(),
        "point" => tree.resolve_fragment(0).await.err(),
        "point_below_range" => tree.resolve_fragment(4).await.err(),
        "cached_point" => tree.resolve_fragment(0).await.err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([0]))
            .await
            .err(),
        "touched" => tree
            .begin_commit(CommitMode::Incremental)
            .resolve_touched(&[0])
            .await
            .err(),
        "validate" => {
            describe_in_memory_root(tree);
            tree.validate().await.err()
        }
        _ => tree.rewrite_tree(commit_mode(read == "bulk")).await.err(),
    }
    .expect("an interior action outside its range must be rejected");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("outside"), "{error}");
}

/// Where a forged tree repeats action sequence 1. Each buffer is valid alone.
#[derive(Clone, Copy, Debug)]
enum RepeatedSequence {
    /// The left and right interiors each buffer sequence 1.
    SiblingInteriors,
    /// The root and the left interior each buffer sequence 1.
    RootAndInterior,
}

/// Leaves `[0]` and `[1, 2]` under a left interior, `[3]` and `[4]` under a
/// right one. Sequence 1 clears the absent deletion file of fragment 1 in the
/// left interior, and of fragment 3 in the buffer `placement` names.
async fn tree_repeating_a_sequence(placement: RepeatedSequence) -> FragmentTree {
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let fixture = Fixture::new(5, config, SnapshotPolicy::default()).await;
    let mut leaves = Vec::new();
    for ids in [vec![0], vec![1, 2], vec![3], vec![4]] {
        let fragments: Vec<_> = ids.into_iter().map(make_fragment).collect();
        let written = fixture
            .tree
            .store
            .write_leaves(&fragments, None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    let clear = |id| pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action::clear_deletion_file(id)),
        ..Default::default()
    };
    let (right_buffer, root_buffer) = match placement {
        RepeatedSequence::SiblingInteriors => (vec![clear(3)], Vec::new()),
        RepeatedSequence::RootAndInterior => (Vec::new(), vec![clear(3)]),
    };
    let left = fixture
        .tree
        .store
        .write_internal(leaves[..2].to_vec(), vec![clear(1)])
        .await
        .unwrap();
    let right = fixture
        .tree
        .store
        .write_internal(leaves[2..].to_vec(), right_buffer)
        .await
        .unwrap();
    let mut tree = fixture.tree;
    route_root_to(
        &mut tree,
        vec![left.child_ref, right.child_ref],
        root_buffer,
        2,
    );
    tree
}

/// One sequence namespace spans the tree, so an operation that decodes two
/// buffers holding the same sequence must reject the tree, even when each
/// buffer routes its copy to a different leaf.
#[rstest]
#[case::materialize("materialize", RepeatedSequence::SiblingInteriors)]
#[case::fixed_readahead_scan("fixed_readahead_scan", RepeatedSequence::SiblingInteriors)]
#[case::set("set", RepeatedSequence::SiblingInteriors)]
#[case::bulk("bulk", RepeatedSequence::SiblingInteriors)]
#[case::drain("drain", RepeatedSequence::SiblingInteriors)]
#[case::validate("validate", RepeatedSequence::SiblingInteriors)]
#[case::bounded_scan_below_root("bounded_scan", RepeatedSequence::RootAndInterior)]
#[case::set_below_root("set", RepeatedSequence::RootAndInterior)]
#[case::point_below_root("point", RepeatedSequence::RootAndInterior)]
#[case::row_offset_below_root("row_offset", RepeatedSequence::RootAndInterior)]
#[case::bulk_below_root("bulk", RepeatedSequence::RootAndInterior)]
#[case::drain_below_root("drain", RepeatedSequence::RootAndInterior)]
#[case::validate_below_root("validate", RepeatedSequence::RootAndInterior)]
#[tokio::test]
async fn reads_reject_a_sequence_repeated_across_decoded_buffers(
    #[case] read: &str,
    #[case] placement: RepeatedSequence,
) {
    let mut tree = tree_repeating_a_sequence(placement).await;
    if matches!(read, "bulk" | "drain") {
        tree.buffer.push(pb::FragmentTreeMutation {
            action_sequence: 2,
            action: Some(action::remove_fragment(2)),
            fragment_count_delta: -1,
            total_rows_delta: -1,
            visible_rows_delta: -1,
        });
        tree.buffer_index.take();
        tree.next_action_sequence = 3;
        tree.store.next_action_sequence = 3;
    }
    let error = match read {
        "materialize" => tree.materialize().await.err(),
        // Starting at fragment 1 still descends the left interior, so the
        // lower bound path decodes both copies.
        "bounded_scan" => Arc::new(tree).scan(1).try_collect::<Vec<_>>().await.err(),
        "fixed_readahead_scan" => Arc::new(tree)
            .scan_with_readahead(0, NonZeroUsize::new(4).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([1, 3]))
            .await
            .err(),
        "point" => tree.resolve_fragment(1).await.err(),
        "row_offset" => tree.fragment_at_row_offset(1).await.err(),
        "bulk" | "drain" => tree.rewrite_tree(commit_mode(read == "bulk")).await.err(),
        "validate" => tree.validate().await.err(),
        other => unreachable!("no read named {other}"),
    }
    .expect("a sequence repeated across decoded buffers must be rejected");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    let expected = "another buffer read by this operation";
    assert!(error.to_string().contains(expected), "{error}");
}

#[tokio::test]
async fn point_checks_unrequested_sequences_after_an_empty_interior() {
    let tree = tree_repeating_a_sequence(RepeatedSequence::RootAndInterior).await;
    assert_eq!(
        tree.resolve_fragment(4).await.unwrap(),
        Some(make_fragment(4))
    );

    // The first point visits the empty right buffer. The left buffer repeats
    // a root sequence for fragment 1, even though the next point requests 0.
    let error = tree
        .resolve_fragment(0)
        .await
        .expect_err("an unrequested action must still be checked against root sequences");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    let expected = "another buffer read by this operation";
    assert!(error.to_string().contains(expected), "{error}");
}

#[tokio::test]
async fn buffered_rewrite_preserves_sequences_moved_between_nodes() {
    let mut tree = tree_repeating_a_sequence(RepeatedSequence::RootAndInterior).await;
    tree.buffer[0].action_sequence = 2;
    tree.buffer.push(pb::FragmentTreeMutation {
        action_sequence: 3,
        action: Some(action::remove_fragment(2)),
        fragment_count_delta: -1,
        total_rows_delta: -1,
        visible_rows_delta: -1,
    });
    tree.next_action_sequence = 4;
    tree.store.next_action_sequence = 4;
    tree.total_fragments = 4;
    tree.total_rows = 4;
    tree.visible_rows = 4;
    tree.buffer_index.take();

    let acc = tree.rewrite_tree(CommitMode::Incremental).await.unwrap();
    assert!(acc.materialized > 0);
    assert!(acc.merges > 0);
    assert_eq!(
        tree.materialize().await.unwrap(),
        [0, 1, 3, 4]
            .into_iter()
            .map(make_fragment)
            .collect::<Vec<_>>()
    );
    describe_in_memory_root(&mut tree);
    tree.validate().await.unwrap();
}

#[tokio::test]
async fn buffered_rewrite_admits_each_source_once_but_rechecks_its_range() {
    let tree = tree_repeating_a_sequence(RepeatedSequence::SiblingInteriors).await;
    let rewrite = rewrite::RewriteNodes::new(&tree.buffer, tree.next_action_sequence);
    let child = &tree.children[0];
    let end = tree.children[1].min_key;
    let decoded = rewrite
        .read_internal(&tree.store, child, end)
        .await
        .unwrap();
    let rewritten = rewrite
        .write_internal(&tree.store, decoded.children, decoded.buffer)
        .await
        .unwrap();
    for source in [child, &rewritten.child_ref] {
        let decoded = rewrite
            .read_internal(&tree.store, source, end)
            .await
            .unwrap();
        assert_eq!(decoded.buffer.len(), 1);
        assert_eq!(decoded.buffer[0].action_sequence, 1);
    }
    let error = rewrite
        .read_internal(&tree.store, child, 1)
        .await
        .expect_err("an admitted source must still fit its current parent range");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

#[tokio::test]
async fn buffered_rewrite_does_not_read_untouched_subtrees_to_check_sequences() {
    let mut tree = tree_repeating_a_sequence(RepeatedSequence::SiblingInteriors).await;
    // Both interiors are full, and joining their four children cannot fit
    // the root. An unpressured rewrite has no reason to read either buffer.
    tree.config.max_children_per_node = 2;
    for child in &tree.children {
        tree.store
            .object_store
            .inner
            .delete(&tree.store.base().clone().join(child.path.as_str()))
            .await
            .unwrap();
    }
    let children = tree.children.clone();
    let acc = tree.rewrite_tree(CommitMode::Incremental).await.unwrap();
    assert_eq!(acc.io_bytes, 0);
    assert_eq!(tree.children, children);
    assert!(tree.buffer.is_empty());
}

/// Uniqueness across subtrees an operation does not read is a writer
/// invariant. Readers must not fetch more to check it.
#[rstest]
#[case::point("point", RepeatedSequence::SiblingInteriors, vec![1])]
#[case::set("set", RepeatedSequence::SiblingInteriors, vec![1])]
#[case::row_offset("row_offset", RepeatedSequence::SiblingInteriors, vec![1])]
#[case::bounded_scan_below_root("bounded_scan", RepeatedSequence::RootAndInterior, vec![3, 4])]
#[tokio::test]
async fn reads_leave_sequences_in_unread_subtrees_to_the_writer(
    #[case] read: &str,
    #[case] placement: RepeatedSequence,
    #[case] expected_ids: Vec<u64>,
) {
    let tree = tree_repeating_a_sequence(placement).await;
    let fragments = match read {
        "point" => vec![tree.resolve_fragment(1).await.unwrap().unwrap()],
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([1]))
            .await
            .unwrap(),
        "row_offset" => match tree.fragment_at_row_offset(1).await.unwrap() {
            OffsetResolution::Found(fragment, 0) => vec![*fragment],
            other => panic!("row 1 must be the first row of fragment 1: {other:?}"),
        },
        // Starting at fragment 3 skips the left interior and its copy.
        "bounded_scan" => Arc::new(tree)
            .scan(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap(),
        other => unreachable!("no read named {other}"),
    };
    let expected: Vec<_> = expected_ids.into_iter().map(make_fragment).collect();
    assert_eq!(fragments, expected);
}

/// A read or validation checks a buffered replacement's deltas against the
/// stored record, including a record a set read selects from a cached leaf.
#[rstest]
#[case::point("point")]
#[case::serial_scan("serial_scan")]
#[case::set_cold("set_cold")]
#[case::set_cached("set_cached")]
#[case::validate("validate")]
#[tokio::test]
async fn buffered_reset_with_a_misstated_delta_is_rejected(#[case] read: &str) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(8, FragmentTreeConfig::default(), policy).await;
    let target = 3;
    let ids = RoaringBitmap::from_iter([target as u32]);
    let leaves = LanceCache::with_capacity(1024 * 1024);
    if read == "set_cached" {
        fixture.tree = reopen_tree(
            &fixture.tree,
            fixture.tree.store.context().with_leaf_cache(leaves.clone()),
        )
        .await;
        for _ in 0..2 {
            let fragments = fixture.tree.resolve_fragments(&ids).await.unwrap();
            assert_eq!(fragments, vec![make_fragment(target)]);
        }
    }
    let mut snapshot = fixture.tree.descriptor.as_ref().clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    assert!(!root.children.is_empty());
    // The stored fragment is replaced by an identical record, so the true
    // deltas are zero. The buffer counts it as an insert.
    root.buffer.push(pb::FragmentTreeMutation {
        action_sequence: root.next_action_sequence,
        action: Some(action::upsert_fragment(&make_fragment(target)).unwrap()),
        fragment_count_delta: 1,
        total_rows_delta: 0,
        visible_rows_delta: 0,
    });
    root.next_action_sequence += 1;
    snapshot.next_action_sequence = root.next_action_sequence;
    let mut tree = fixture.open(&snapshot, fixture.tree.version()).await;
    if read == "set_cached" {
        tree = reopen_tree(&tree, tree.store.context().with_leaf_cache(leaves)).await;
    }
    fixture.io.incremental_stats();
    let error = match read {
        "point" => tree.resolve_fragment(target).await.err(),
        "set_cold" | "set_cached" => tree.resolve_fragments(&ids).await.err(),
        "validate" => tree.validate().await.err(),
        _ => Arc::new(tree)
            .scan_with_readahead(0, NonZeroUsize::MIN)
            .try_collect::<Vec<_>>()
            .await
            .err(),
    }
    .expect("a misstated delta must be rejected");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    if read == "set_cached" {
        assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    }
}

/// A set read returns only the requested records whether its covering leaves
/// are decoded, sighted, admitted, served from the leaf cache or retained for
/// validation, and fetches each covering leaf once unless it is cached.
#[rstest]
#[case::no_cache("no_cache")]
#[case::first_sighting("first_sighting")]
#[case::admission("admission")]
#[case::cache_hit("cache_hit")]
#[case::retained("retained")]
#[tokio::test]
async fn set_reads_return_requested_records_in_every_cache_state(#[case] state: &str) {
    let config = FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024);
    let mut fixture = Fixture::new(300, config, SnapshotPolicy::default()).await;
    assert_eq!(fixture.tree.height(), 1);
    // 5000 is past every stored ID, so it routes to the last leaf and is absent.
    let ids = RoaringBitmap::from_iter([1, 100, 200, 299, 5000]);
    let children = &fixture.tree.children;
    let covering: std::collections::BTreeSet<usize> = ids
        .iter()
        .map(|id| children.partition_point(|child| child.min_key <= u64::from(id)) - 1)
        .collect();
    assert!(covering.len() >= 3, "{covering:?}");
    let expected: Vec<_> = ids
        .iter()
        .map(u64::from)
        .filter(|id| *id < 300)
        .map(make_fragment)
        .collect();
    let prior_reads = match state {
        "no_cache" | "first_sighting" => 0,
        "admission" => 1,
        _ => 2,
    };
    if state != "no_cache" {
        fixture.tree = reopen_tree(
            &fixture.tree,
            fixture
                .tree
                .store
                .context()
                .with_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024)),
        )
        .await;
    }
    for _ in 0..prior_reads {
        assert_eq!(
            fixture.tree.resolve_fragments(&ids).await.unwrap(),
            expected
        );
    }
    if state == "retained" {
        fixture.tree.store.retain_validation_reads().unwrap();
    }
    fixture.io.incremental_stats();
    assert_eq!(
        fixture.tree.resolve_fragments(&ids).await.unwrap(),
        expected
    );
    let leaf_gets = fixture
        .io
        .incremental_stats()
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with("get") && request.path.as_ref().contains("_bt/leaf/")
        })
        .count();
    let expected_gets = if state == "cache_hit" {
        0
    } else {
        covering.len()
    };
    assert_eq!(leaf_gets, expected_gets);
}

/// A sibling reference is checked against its object only when read, so
/// grouping siblings to merge must not trust its fanout. A sibling whose
/// fanout overflows the sum fails the fit check and stays unread, and a
/// whole read reports it.
#[tokio::test]
async fn merge_grouping_leaves_a_sibling_with_overflowing_fanout_unread() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::default();
    let fixture = Fixture::new(0, config.clone(), policy).await;
    let mut interiors = Vec::new();
    for ids in [[0, 1], [2, 3]] {
        interiors
            .push(write_leaf_interior(&fixture.tree, &[&[ids[0]], &[ids[1]]], Vec::new()).await);
    }
    assert!(node::is_underflow(&interiors[0], &config));
    interiors[1].num_children = u32::MAX;
    let snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(pb::FragmentTreeRoot {
            children: interiors,
            buffer: Vec::new(),
            next_action_sequence: 1,
        })),
        mutations_since_root: Vec::new(),
        next_action_sequence: 1,
    };
    let tree = FragmentTree::open_snapshot(fixture.context(), &snapshot, 1, config, 4)
        .await
        .unwrap();
    let (prepared, _) = tree
        .begin_commit(CommitMode::Incremental)
        .prepare(ValidatedCommit::fragment_actions(Vec::new()), policy)
        .await
        .unwrap();
    let error = prepared.validate().await.unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("contents differ from its parent reference"),
        "{error}"
    );
}

#[derive(Clone, Copy, Debug)]
enum UnreadableRecord {
    UnknownDeletionFileType,
    OverlayWithoutCoverage,
}

/// Opening decodes every buffered upsert, so prepare must not buffer a
/// record that decoding rejects.
#[rstest]
#[case::unknown_deletion_file_type(
    UnreadableRecord::UnknownDeletionFileType,
    "Unknown deletion file type"
)]
#[case::overlay_without_coverage(UnreadableRecord::OverlayWithoutCoverage, "missing its coverage")]
#[tokio::test]
async fn prepare_rejects_a_last_upsert_that_readers_reject(
    #[case] defect: UnreadableRecord,
    #[case] expected: &str,
) {
    let fixture = Fixture::new(10, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut record = pb::DataFragment::from(&make_fragment(10));
    match defect {
        UnreadableRecord::UnknownDeletionFileType => {
            record.deletion_file = Some(pb::DeletionFile {
                file_type: 7,
                ..Default::default()
            });
        }
        UnreadableRecord::OverlayWithoutCoverage => {
            record.overlays.push(pb::DataOverlayFile {
                data_file: Some(record.files[0].clone()),
                coverage: None,
                committed_version: 1,
            });
        }
    }
    let upsert = pb::FragmentAction {
        action: Some(pb::fragment_action::Action::UpsertFragment(record)),
    };
    let error = fixture
        .tree
        .begin_commit(CommitMode::Incremental)
        .prepare(
            ValidatedCommit::fragment_actions(vec![upsert]),
            SnapshotPolicy::default(),
        )
        .await
        .err()
        .expect("prepare must not buffer a record readers reject");
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains(expected), "{error}");
}
