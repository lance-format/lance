// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::fragment_metadata::action;
use crate::fragment_metadata::support::{
    make_backfill_data_file, make_fragment, make_fragment_with_files, make_replacement_data_file,
};
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::SchedulerConfig;
use lance_io::utils::failpoint::{FailOn, FailWhen, Failpoint, FailpointController};
use lance_io::utils::tracking_store::IOTracker;
use object_store::ObjectStoreExt;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstest::rstest;

mod interior_cache;
mod leaf_uploads;
mod ownership;

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
    fixture.snapshot = pb::FragmentTree {
        root: Some(pb::fragment_tree::Root::InlineRoot(
            fixture.tree.compacted_root(),
        )),
        mutations_since_root: Vec::new(),
        next_action_sequence,
    };
    fixture.tree.snapshot = Some(Box::new(fixture.snapshot.clone()));
    fixture
        .open(&fixture.snapshot, fixture.tree.version())
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reader.verify_watermarks().await.unwrap();
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
    snapshot: pb::FragmentTree,
    store: Arc<ObjectStore>,
    base: Path,
    scheduler: Arc<ScanScheduler>,
    io: IOTracker,
}

#[rstest]
#[case::zero_sequence(false)]
#[case::zero_object_size(true)]
#[tokio::test]
async fn snapshot_rejects_invalid_root_and_child_metadata(#[case] is_child_invalid: bool) {
    let fixture = Fixture::new(2, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let mut snapshot = fixture.snapshot.clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    if is_child_invalid {
        root.children[0].object_size = 0;
    } else {
        root.next_action_sequence = 0;
    }
    let result = FragmentTree::open_snapshot(
        fixture.store,
        fixture.base,
        fixture.scheduler,
        Arc::new(LanceCache::with_capacity(0)),
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
    if is_child_invalid {
        assert!(
            error.to_string().contains("object_size") || error.to_string().contains("child"),
            "{error}"
        );
    } else {
        assert!(
            error.to_string().contains("next_action_sequence"),
            "{error}"
        );
    }
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
        let (tree, snapshot, _) = FragmentTree::bootstrap_snapshot(
            store.clone(),
            base.clone(),
            scheduler.clone(),
            Arc::new(LanceCache::with_capacity(0)),
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
            snapshot,
            store,
            base,
            scheduler,
            io,
        }
    }

    async fn open(&self, snapshot: &pb::FragmentTree, version: u64) -> FragmentTree {
        FragmentTree::open_snapshot(
            self.store.clone(),
            self.base.clone(),
            self.scheduler.clone(),
            Arc::new(LanceCache::with_capacity(0)),
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
        let touched = if bulk {
            self.tree.resolve_touched_for_bulk(&ids).await.unwrap()
        } else {
            self.tree.resolve_touched(&ids).await.unwrap()
        };
        let (snapshot, stats) = self
            .tree
            .prepare_snapshot(
                ValidatedCommit::fragment_actions(actions),
                &touched,
                &self.snapshot,
                policy,
                bulk,
            )
            .await
            .unwrap();
        self.snapshot = snapshot;
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
    assert_eq!(fixture.snapshot.mutations_since_root.len(), 1);
    let reopened = fixture.open(&fixture.snapshot, 3).await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (2..64).map(make_fragment).collect::<Vec<_>>()
    );

    let mut snapshot = fixture.snapshot.clone();
    match count {
        "Fragments" => snapshot.mutations_since_root[0].fragment_count_delta = -999,
        "physical rows" => snapshot.mutations_since_root[0].total_rows_delta = -999,
        "visible rows" => snapshot.mutations_since_root[0].visible_rows_delta = -999,
        _ => unreachable!(),
    }
    let error = FragmentTree::open_snapshot(
        fixture.store,
        fixture.base,
        fixture.scheduler,
        Arc::new(LanceCache::with_capacity(0)),
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
    let initial = fixture.snapshot.clone();
    let mut expected: BTreeMap<_, _> = (0..48).map(|id| (id, make_fragment(id))).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut maximum_height = fixture.tree.height();
    for round in 0..12 {
        let mut changes = Vec::new();
        for offset in 0..4 {
            let id = 48 + round * 4 + offset;
            let fragment = make_fragment(id);
            changes.push(action::upsert_fragment(&fragment));
            expected.insert(id, fragment);
        }
        let id = rng.random_range(0..48);
        let mut fragment = make_fragment(id);
        fragment.files[0].path = format!("round-{round}-{id}.lance");
        changes.push(action::upsert_fragment(&fragment));
        expected.insert(id, fragment);
        fixture.commit(changes, policy, bulk).await;
        maximum_height = maximum_height.max(fixture.tree.height());
        assert_eq!(
            fixture.tree.materialize().await.unwrap(),
            expected.values().cloned().collect::<Vec<_>>()
        );
        fixture.tree.verify_watermarks().await.unwrap();
        let reopened = fixture.open(&fixture.snapshot, round + 2).await;
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
    fixture.snapshot.root = Some(pb::fragment_tree::Root::InlineRoot(
        fixture.tree.compacted_root(),
    ));
    fixture.tree.snapshot = Some(Box::new(fixture.snapshot.clone()));
    let mut actions = Vec::new();
    for fragment in expected.iter_mut().step_by(4) {
        fragment.files.push(make_backfill_data_file(fragment.id, 2));
        actions.push(action::upsert_fragment(fragment));
    }

    let stats = fixture.commit(actions, policy, true).await;
    assert_eq!(stats.merges, 2);
    assert_eq!(fixture.tree.root_child_count(), 1);
    assert_eq!(fixture.tree.materialize().await.unwrap(), expected);
    fixture.tree.verify_watermarks().await.unwrap();
    assert_eq!(
        fixture
            .open(&fixture.snapshot, 2)
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
    touched: &TouchedFragments,
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
    let (snapshot, _) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions),
            touched,
            &fixture.snapshot,
            policy,
            true,
        )
        .await
        .unwrap();
    fixture.snapshot = snapshot;
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
    let mut touched = fixture.tree.resolve_touched_for_bulk(&[0]).await.unwrap();
    let witness = fixture
        .tree
        .resolve_touched_for_bulk(&[second])
        .await
        .unwrap();
    touched.ids.extend(witness.ids);
    touched.fragments.extend(witness.fragments);
    assert_eq!(leaf_gets(&fixture.io.incremental_stats()), covering_leaves);
    assert_eq!(
        commit_replacements(&mut fixture, policy, &[0, second], &touched).await,
        0,
        "materialization must reread every leaf from retention"
    );
}

#[rstest]
#[case::retained("retained")]
#[case::unretained("unretained")]
#[tokio::test]
async fn bulk_commit_without_retention_reads_each_changed_leaf_once(#[case] validation: &str) {
    let (mut fixture, policy, leaf_keys) = bulk_fixture_without_coalescing().await;
    let ids = [0, leaf_keys[0]];
    fixture.io.incremental_stats();
    let touched = match validation {
        "retained" => fixture.tree.resolve_touched_for_bulk(&ids).await.unwrap(),
        // An eager writer validates against its resident fragment list and
        // reads no leaf before materialization.
        "unretained" => TouchedFragments {
            ids: ids.iter().copied().collect(),
            fragments: ids.iter().map(|id| (*id, make_fragment(*id))).collect(),
        },
        _ => unreachable!(),
    };
    let validation_leaf_gets = leaf_gets(&fixture.io.incremental_stats());
    let commit_leaf_gets = commit_replacements(&mut fixture, policy, &ids, &touched).await;
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
    let mut historical = fixture.snapshot.clone();
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
    let rewrite = rewrite::RewriteNodes::new(&fixture.tree.buffer);
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
    fixture.tree.verify_watermarks().await.unwrap();
    let mut snapshot = historical.clone();
    snapshot.root = Some(pb::fragment_tree::Root::InlineRoot(
        fixture.tree.compacted_root(),
    ));
    let reopened = fixture.open(&snapshot, 1).await;
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
    let actions = vec![action::upsert_fragment(&make_fragment(300))];
    let touched = fixture.tree.resolve_touched(&[300]).await.unwrap();
    fixture.io.incremental_stats();
    let (snapshot, _) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions),
            &touched,
            &fixture.snapshot.clone(),
            policy,
            false,
        )
        .await
        .unwrap();
    assert_eq!(fixture.io.incremental_stats().read_iops, 0);
    let reopened = fixture.open(&snapshot, fixture.tree.version()).await;
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
    let (tree, stats) = FragmentTree::build(fixture.tree.store, config.clone(), &fragments)
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

/// Which fragment stream a test drains.
#[derive(Debug, Clone, Copy)]
enum LeafWindowKind {
    /// `fragment_stream_with_prefetch`, the eager load's stream.
    Fixed(usize),
    /// `fragment_stream_from`, the lazy scanner's stream.
    Ramped,
}

impl LeafWindowKind {
    fn stream(
        self,
        tree: Arc<FragmentTree>,
        lower_bound: u64,
    ) -> BoxStream<'static, Result<Fragment>> {
        match self {
            Self::Fixed(prefetch) => {
                assert_eq!(lower_bound, 0, "a fixed window always starts at zero");
                tree.fragment_stream_with_prefetch(prefetch)
            }
            Self::Ramped => tree.fragment_stream_from(lower_bound),
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
    actions.push(action::upsert_fragment(&make_fragment(64)));
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
#[case::serial(LeafWindowKind::Fixed(1), 0)]
#[case::prefetched(LeafWindowKind::Fixed(4), 0)]
#[case::ramped_from_start(LeafWindowKind::Ramped, 0)]
#[case::ramped_from_middle(LeafWindowKind::Ramped, 32)]
#[tokio::test]
async fn deep_stream_honors_prefetch_and_preserves_order(
    #[case] window: LeafWindowKind,
    #[case] lower_bound: u64,
) {
    let (fixture, mut expected) = deep_stream_fixture().await;
    expected.retain(|fragment| fragment.id >= lower_bound);
    let leaf_count = fixture.tree.shape_report().await.unwrap().leaf_keys.len();
    let (tree, io) = delayed_tree(fixture.tree, std::time::Duration::from_millis(2));
    let actual: Vec<_> = window
        .stream(tree, lower_bound)
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
    if matches!(window, LeafWindowKind::Fixed(1)) {
        assert_eq!(measured.num_stages, measured.read_iops);
    } else {
        assert!(
            measured.num_stages < measured.read_iops,
            "deep leaf reads must overlap: {measured:?}"
        );
    }
}

/// A LIMIT scan that ends inside the first leaf must not pay for a second
/// leaf, so the ramped window admits more only after the first is drained.
#[rstest]
#[case::first_fragment("first_fragment")]
#[case::all_but_last_of_first_leaf("all_but_last")]
#[case::whole_first_leaf("whole_leaf")]
#[tokio::test]
async fn fragment_stream_from_reads_one_leaf_until_the_first_leaf_is_drained(#[case] taken: &str) {
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
    let actual: Vec<_> = tree
        .fragment_stream_from(0)
        .take(taken)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(actual, expected[..taken]);
    // A read admitted before the drop would complete within one latency.
    tokio::time::sleep(latency * 4).await;
    assert_eq!(leaf_requests(&io.incremental_stats()).len(), 1);
}

/// A routing error found while the window reads ahead must wait until the
/// fragments of every earlier leaf are yielded.
#[rstest]
#[case::fixed_1(LeafWindowKind::Fixed(1))]
#[case::fixed_4(LeafWindowKind::Fixed(4))]
#[case::ramped(LeafWindowKind::Ramped)]
#[tokio::test]
async fn fragment_stream_yields_earlier_leaves_before_a_later_routing_error(
    #[case] window: LeafWindowKind,
) {
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
    let mut stream = window.stream(Arc::new(fixture.tree), 0);
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
async fn dropped_fragment_stream_admits_no_unread_leaf() {
    let (mut fixture, _) = deep_stream_fixture().await;
    let leaf_keys = leaf_keys_in_walk_order(&fixture.tree).await;
    let leaf_count = leaf_keys.len();
    assert!(leaf_count >= 7, "{leaf_keys:?}");
    fixture
        .tree
        .set_leaf_cache(LanceCache::with_capacity(16 * 1024 * 1024));
    let latency = std::time::Duration::from_millis(20);
    let (tree, io) = delayed_tree(fixture.tree, latency);
    // Draining two leaves widens the window to four. Taking the first
    // fragment of the third admits up to three more leaves.
    const CONSUMED_LEAVES: usize = 3;
    const MAX_ADMITTED_UNREAD: usize = 3;
    let taken = (leaf_keys[0] + leaf_keys[1] + 1) as usize;
    let mut stream = tree.clone().fragment_stream_from(0);
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
            .fragment_stream_with_prefetch(8)
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
                action::upsert_fragment(&make_fragment(64)),
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
    match reads {
        MaterializeReads::WarmCache => {
            tree.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
            assert_eq!(tree.materialize().await.unwrap(), expected);
        }
        MaterializeReads::BulkValidation => {
            tree.resolve_touched_for_bulk(&[]).await.unwrap();
        }
        MaterializeReads::Drained | MaterializeReads::InteriorActions => {}
    }
    io.incremental_stats();
    assert_eq!(tree.materialize().await.unwrap(), expected);
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

    // A leaf decoded twice through the cache is admitted, and scratch keeps
    // every leaf a bulk commit read, so a further pass fetches no leaf.
    if matches!(
        reads,
        MaterializeReads::WarmCache | MaterializeReads::BulkValidation
    ) {
        assert_eq!(tree.materialize().await.unwrap(), expected);
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
    tree.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
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

#[tokio::test]
async fn failed_prepare_restores_frontiers_and_buffer() {
    let policy = SnapshotPolicy::default();
    let config = FragmentTreeConfig::new(1024, u32::MAX)
        .with_max_leaf_bytes(4096)
        .with_hard_capacity_bytes(8192);
    let mut fixture = Fixture::new(2, config, policy).await;
    let touched = fixture.tree.resolve_touched(&[9]).await.unwrap();
    let error = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(
                &make_fragment_with_files(9, 256),
            )]),
            &touched,
            &fixture.snapshot,
            policy,
            true,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("hard_capacity_bytes"), "{error}");
    assert_eq!(fixture.tree.version(), 1);
    assert_eq!(fixture.tree.next_fragment_id(), 2);
    assert_eq!(fixture.tree.root_buffer_len(), 0);
    fixture
        .commit(
            vec![action::upsert_fragment(&make_fragment(2))],
            policy,
            false,
        )
        .await;
    assert_eq!(
        fixture.tree.materialize().await.unwrap(),
        (0..3).map(make_fragment).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn dropped_prepare_leaves_the_tree_unchanged() {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(1, FragmentTreeConfig::default(), policy).await;
    let original = fixture.tree.materialize().await.unwrap();
    let touched = fixture.tree.resolve_touched(&[0]).await.unwrap();
    let delayed = FailpointController::default();
    delayed.set_get_latency(std::time::Duration::from_secs(60));
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&delayed);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let mut pending = Box::pin(fixture.tree.prepare_snapshot(
        ValidatedCommit::fragment_actions(vec![action::add_data_file(
            0,
            &make_backfill_data_file(0, 1),
        )]),
        &touched,
        &fixture.snapshot,
        policy,
        true,
    ));
    // Suspend at the first leaf GET, after the rewrite takes the routing.
    assert!(futures::poll!(pending.as_mut()).is_pending());
    drop(pending);
    fixture.tree = fixture.tree.with_object_store(fixture.store.clone());
    assert_eq!(fixture.tree.version(), 1);
    assert_eq!(fixture.tree.materialize().await.unwrap(), original);

    let appended = make_fragment(1);
    let (snapshot, _) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&appended)]),
            &TouchedFragments::default(),
            &fixture.snapshot,
            policy,
            false,
        )
        .await
        .unwrap();
    let reopened = fixture.open(&snapshot, fixture.tree.version()).await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        vec![original[0].clone(), appended]
    );
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
            (action::upsert_fragment(&second), second)
        }
        LaterMutation::ClearDeletionFile => {
            let mut expected = first.clone();
            expected.files.push(added);
            (action::clear_deletion_file(0), expected)
        }
    };
    let outer_buffer = vec![stored(1, action::upsert_fragment(&first)), stored(3, last)];
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
        fixture.store.clone(),
        fixture.base.clone(),
        fixture.scheduler.clone(),
        Arc::new(LanceCache::with_capacity(0)),
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
    let (published, stats) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&appended)]),
            &TouchedFragments::default(),
            &snapshot,
            policy,
            false,
        )
        .await
        .unwrap();
    assert!(stats.flushes > 0, "{stats:?}");
    let reopened = fixture.open(&published, fixture.tree.version()).await;
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
            .commit(vec![action::upsert_fragment(&replaced)], policy, false)
            .await;
        fixture.tree = fixture
            .open(&fixture.snapshot, fixture.tree.version())
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
            buffer: vec![stored(1, action::upsert_fragment(&first))],
            next_action_sequence: 2,
        })),
        mutations_since_root: vec![],
        next_action_sequence: 2,
    };
    let mut tree = FragmentTree::open_snapshot(
        fixture.store.clone(),
        fixture.base.clone(),
        fixture.scheduler.clone(),
        Arc::new(LanceCache::with_capacity(0)),
        &snapshot,
        1,
        config.clone(),
        2,
    )
    .await
    .unwrap();
    let mut second = first.clone();
    second.files[0] = make_replacement_data_file(0, 1);
    let touched = tree.resolve_touched(&[0]).await.unwrap();
    let (_, stats) = tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(vec![action::upsert_fragment(&second)]),
            &touched,
            &snapshot,
            policy,
            false,
        )
        .await
        .unwrap();
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
    let previous = fixture.snapshot.clone();
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
            actions.push(action::upsert_fragment(&replacement));
            expected.push(replacement);
        }
    }
    let ids: Vec<_> = original.iter().map(|fragment| fragment.id).collect();
    let touched = fixture.tree.resolve_touched(&ids).await.unwrap();
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
    let result = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions.clone()),
            &touched,
            &previous,
            policy,
            false,
        )
        .await;
    if failure.is_some() {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::IO { .. }), "{error}");
        assert!(delayed.tripped());
        assert!(error.to_string().contains("failpoint"), "{error}");
        assert_eq!(fixture.tree.version(), 1);
        assert_eq!(fixture.tree.root_buffer_len(), 0);
        assert_eq!(fixture.tree.materialize().await.unwrap(), original);
        delayed.disarm();
        fixture
            .tree
            .prepare_snapshot(
                ValidatedCommit::fragment_actions(actions),
                &touched,
                &previous,
                policy,
                false,
            )
            .await
            .unwrap();
    } else {
        let (_, stats) = result.unwrap();
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
        .commit(vec![action::upsert_fragment(&fragment)], policy, bulk)
        .await;
    expected.insert(7, fragment.clone());
    let historical = fixture.snapshot.clone();
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
            .open(&fixture.snapshot, fixture.tree.version())
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
        fixture.tree.verify_watermarks().await.unwrap();
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
                action::upsert_fragment(&make_fragment(90)),
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
    let from: Vec<_> = tree
        .clone()
        .fragment_stream_from(62)
        .try_collect()
        .await
        .unwrap();
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
                    action::upsert_fragment(&make_fragment(past_last)),
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    if let LeafCache::Warm = cache {
        tree.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
        // A leaf is admitted on its second whole read, so the third one hits.
        for _ in 0..2 {
            assert_eq!(
                tree.resolve_fragments_concurrent(&bitmap, 4).await.unwrap(),
                expected
            );
        }
    }
    fixture.io.incremental_stats();
    let resolved = tree.resolve_fragments_concurrent(&bitmap, 4).await.unwrap();
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
    let historical = fixture.snapshot.clone();
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
        fixture.tree.verify_watermarks().await.unwrap();
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
        .open(&fixture.snapshot, fixture.tree.version())
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
    let pb::fragment_tree::Root::InlineRoot(root) = fixture.snapshot.root.clone().unwrap() else {
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
    let mut snapshot = fixture.snapshot.clone();
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
                Arc::new(LanceCache::with_capacity(0)),
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
        fixture.store,
        fixture.base,
        fixture.scheduler,
        Arc::new(LanceCache::with_capacity(0)),
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    let fragments = reopened.materialize().await.unwrap();
    assert_eq!(fragments[0].id, 6);
    reopened.verify_watermarks().await.unwrap();
    let mut snapshot = fixture.snapshot.clone();
    if let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root {
        root.children[0].min_key = 6;
    } else {
        panic!("expected inline root");
    }
    let error = FragmentTree::open_snapshot(
        fixture.store.clone(),
        fixture.base.clone(),
        fixture.scheduler.clone(),
        Arc::new(LanceCache::with_capacity(0)),
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_reachable().await.unwrap();
    reopened.verify_watermarks().await.unwrap();
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
            vec![action::upsert_fragment(&make_fragment(fresh))],
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_watermarks().await.unwrap();
    reopened.verify_reachable().await.unwrap();
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
    actions.extend((fresh..fresh + appended).map(|id| action::upsert_fragment(&make_fragment(id))));
    fixture.commit(actions, policy, false).await;

    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_reachable().await.unwrap();
    reopened.verify_watermarks().await.unwrap();
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
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    cached.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
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

/// Trees that share a leaf cache stop reading a leaf from storage once it has
/// been decoded twice. The cache never stands in for storage when a tree
/// proves its objects exist.
#[tokio::test]
async fn shared_leaf_cache_never_hides_a_missing_leaf() {
    let config = FragmentTreeConfig::default().with_max_leaf_bytes(16 * 1024);
    let fixture = Fixture::new(300, config, SnapshotPolicy::default()).await;
    let leaves = LanceCache::with_capacity(64 * 1024 * 1024);
    let mut first = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    first.set_leaf_cache(leaves.clone());
    let mut second = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    second.set_leaf_cache(leaves);
    let expected = first.materialize().await.unwrap();
    assert_eq!(second.materialize().await.unwrap(), expected);
    assert!(first.root_child_count() >= 2);

    fixture.io.incremental_stats();
    assert_eq!(second.materialize().await.unwrap(), expected);
    let leaf_reads = fixture
        .io
        .incremental_stats()
        .requests
        .into_iter()
        .filter(|request| {
            request.method.starts_with("get_") && request.path.as_ref().contains("_bt/leaf/")
        })
        .count();
    assert_eq!(leaf_reads, 0);

    let leaf = fixture
        .tree
        .node_paths()
        .await
        .unwrap()
        .into_iter()
        .find(|path| path.starts_with("_bt/leaf/"))
        .unwrap();
    let path: Path = fixture
        .base
        .parts()
        .chain(Path::from(leaf).parts())
        .collect();
    fixture.store.inner.delete(&path).await.unwrap();
    assert_eq!(second.materialize().await.unwrap(), expected);
    assert!(second.verify_reachable().await.is_err());
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
    let actions: Vec<_> = expected.iter().map(action::upsert_fragment).collect();
    let ids: Vec<_> = (0..256).collect();
    let touched = fixture.tree.resolve_touched(&ids).await.unwrap();
    let previous = fixture.snapshot.clone();
    let io = IOTracker::default();
    let mut store = fixture.store.as_ref().clone();
    store.apply_wrapper(&io);
    fixture.tree = fixture.tree.with_object_store(Arc::new(store));
    let (snapshot, _) = fixture
        .tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(actions),
            &touched,
            &previous,
            policy,
            false,
        )
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
    fixture.snapshot = snapshot;
    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_watermarks().await.unwrap();
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
    let mut snapshot = fixture.snapshot.clone();
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
        actions.push((action::upsert_fragment(&make_fragment(8)), 1));
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
    let mut tree = FragmentTree::open_snapshot(
        fixture.store.clone(),
        fixture.base.clone(),
        fixture.scheduler.clone(),
        Arc::new(LanceCache::with_capacity(0)),
        &snapshot,
        fixture.tree.version(),
        fixture.config.clone(),
        9,
    )
    .await
    .unwrap();
    let touched = tree.resolve_touched(&[]).await.unwrap();
    let error = tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(Vec::new()),
            &touched,
            &snapshot,
            policy,
            false,
        )
        .await
        .expect_err("a removal of an absent fragment must not retire the leaf");
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
    let mut snapshot = fixture.snapshot.clone();
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
    let mut tree = fixture.open(&snapshot, fixture.tree.version()).await;
    let touched = tree.resolve_touched(&[]).await.unwrap();
    let error = tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(Vec::new()),
            &touched,
            &snapshot,
            policy,
            false,
        )
        .await
        .expect_err("a removal of an absent fragment must not retire the subtree");
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
    let mut snapshot = fixture.snapshot.clone();
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
            action: Some(action::upsert_fragment(&make_fragment_with_files(id, 2))),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        });
        root.next_action_sequence += 1;
    }
    snapshot.next_action_sequence = root.next_action_sequence;
    let mut tree = fixture.open(&snapshot, fixture.tree.version()).await;
    let touched = tree.resolve_touched(&[]).await.unwrap();
    let error = tree
        .prepare_snapshot(
            ValidatedCommit::fragment_actions(Vec::new()),
            &touched,
            &snapshot,
            policy,
            false,
        )
        .await
        .expect_err("a misstated replacement must not rebuild the leaf");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// Removing all but one fragment from a tree three or more levels deep leaves
/// single-child interiors that no sibling can absorb. The commit rebuilds in
/// bulk rather than persisting that chain.
#[tokio::test]
async fn collapsing_to_one_leaf_rebuilds_instead_of_persisting_a_chain() {
    let policy = SnapshotPolicy::default();
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
    fixture.commit(actions, policy, false).await;

    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_reachable().await.unwrap();
    reopened.verify_watermarks().await.unwrap();
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
        .map(|id| action::upsert_fragment(&make_fragment(*id)))
        .collect();
    fixture.commit(actions, policy, false).await;
    assert!(
        fixture.tree.height() >= 2,
        "twenty singleton leaves must split the root at fanout {}",
        fixture.config.max_children_per_node
    );
    assert_eq!(fixture.tree.children[0].min_key, 0);
    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    reopened.verify_reachable().await.unwrap();
    let regrown: Vec<u64> = reopened
        .materialize()
        .await
        .unwrap()
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    assert_eq!(regrown, ids);
}

/// Every read rejects a buffered action at or below its leaf's watermark,
/// including the leaf-prefetching streams that eager loads and cleanup use,
/// the set reads commits resolve through, and a point read whose newest
/// action replaces the whole record.
#[rstest]
#[case::point("point")]
#[case::point_after_reset("point_after_reset")]
#[case::stream("stream")]
#[case::offset("offset")]
#[case::prefetch_stream("prefetch_stream")]
#[case::fragment_stream("fragment_stream")]
#[case::ramped_stream("ramped_stream")]
#[case::resolve_fragments("resolve_fragments")]
#[case::resolve_touched("resolve_touched")]
#[case::materialize("materialize")]
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
            vec![action::upsert_fragment(&make_fragment_with_files(
                target, 4,
            ))],
            policy,
            true,
        )
        .await;
    let mut snapshot = fixture.snapshot.clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    let leaf = &root.children[node::child_index_for(&root.children, target)];
    assert!(leaf.materialized_through_action_sequence >= 1);
    let stale = if read == "point_after_reset" {
        action::upsert_fragment(&make_fragment_with_files(target, 4))
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
        "resolve_touched" => tree.resolve_touched(&[target]).await.err(),
        "materialize" => tree.materialize().await.err(),
        "stream" => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
        "prefetch_stream" => Arc::new(tree)
            .fragment_stream_with_prefetch(4)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "fragment_stream" => Arc::new(tree)
            .fragment_stream()
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "ramped_stream" => Arc::new(tree)
            .fragment_stream_from(0)
            .try_collect::<Vec<_>>()
            .await
            .err(),
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
            .map(|id| action::upsert_fragment(&make_fragment(id)))
            .collect();
        let stats = fixture.commit(actions, policy, path == "bulk").await;
        if path == "interior" {
            assert!(stats.max_flush_depth > 0);
            assert!(stats.splits > 0);
        }
    }
    let reopened = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    assert_eq!(
        reopened.materialize().await.unwrap(),
        (0..count + added).map(make_fragment).collect::<Vec<_>>()
    );
    reopened.verify_reachable().await.unwrap();
    reopened.verify_watermarks().await.unwrap();
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
            store,
            base,
            scheduler,
            Arc::new(LanceCache::with_capacity(0)),
            config,
            &mut (0..4).map(make_fragment).collect::<Vec<_>>(),
            1,
            policy,
        )
        .await
        .err()
    } else {
        let mut fixture = Fixture::new(2, config, policy).await;
        let actions: Vec<_> = (2..4)
            .map(|id| action::upsert_fragment(&make_fragment(id)))
            .collect();
        let touched = fixture.tree.resolve_touched(&[2, 3]).await.unwrap();
        fixture
            .tree
            .prepare_snapshot(
                ValidatedCommit::fragment_actions(actions),
                &touched,
                &fixture.snapshot.clone(),
                policy,
                false,
            )
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
            action: Some(action::upsert_fragment(&make_fragment(id))),
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
        .rewrite_tree()
        .await
        .expect_err("a leaf storing keys past its range must not be rewritten");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

#[rstest]
#[case::point("point")]
#[case::set("set")]
#[case::materialize("materialize")]
#[case::stream("stream")]
#[case::prefetch_stream("prefetch_stream")]
#[case::ramped_stream("ramped_stream")]
#[case::row_offset("row_offset")]
#[case::bulk("bulk")]
#[case::coalesce("coalesce")]
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
    let tree = &mut fixture.tree;
    match reads {
        "cached" => tree.set_leaf_cache(LanceCache::with_capacity(1024 * 1024)),
        "retained" => tree.store.retain_validation_reads().unwrap(),
        _ => {}
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
        tree.force_flush = true;
    }
    let error = match read {
        "point" => tree.resolve_fragment(60).await.err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([60]))
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "stream" => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
        "prefetch_stream" => Arc::new(fixture.tree)
            .fragment_stream_with_prefetch(4)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "ramped_stream" => Arc::new(fixture.tree)
            .fragment_stream_from(0)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "row_offset" => tree.fragment_at_row_offset(1).await.err(),
        _ => tree.rewrite_tree().await.err(),
    }
    .expect("a leaf storing keys past its range must not be read");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

#[rstest]
#[case::point("point")]
#[case::set("set")]
#[case::concurrent_set("concurrent_set")]
#[case::materialize("materialize")]
#[case::stream("stream")]
#[case::prefetch_stream("prefetch_stream")]
#[case::row_offset("row_offset")]
#[case::bulk("bulk")]
#[case::coalesce("coalesce")]
#[tokio::test]
async fn reads_reject_child_fences_past_the_parent_range(
    #[case] read: &str,
    #[values(false, true)] cached: bool,
) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(5, FragmentTreeConfig::default(), policy).await;
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
        tree.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
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
        tree.force_flush = true;
    }
    // The left interior ends at 100, so its child fence at 150 is invalid.
    let error = match read {
        "point" => tree.resolve_fragment(0).await.err(),
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([0]))
            .await
            .err(),
        "concurrent_set" => tree
            .resolve_fragments_concurrent(&RoaringBitmap::from_iter([0]), 8)
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "stream" => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
        "prefetch_stream" => Arc::new(fixture.tree)
            .fragment_stream_with_prefetch(4)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "row_offset" => tree.fragment_at_row_offset(0).await.err(),
        _ => tree.rewrite_tree().await.err(),
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
}

/// Sibling interiors under one parent are independent once the parent is
/// verified, so a concurrent set resolution reads them together while every
/// covering node is still read once and interior buffers still replay.
#[rstest]
#[case::serial(1)]
#[case::concurrent(8)]
#[tokio::test]
async fn set_resolution_overlaps_sibling_interior_reads(#[case] concurrency: usize) {
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
    let resolved = tree
        .resolve_fragments_concurrent(&requested, concurrency)
        .await
        .unwrap();

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
    if concurrency == 1 {
        assert_eq!(measured.num_stages, measured.read_iops);
    } else {
        // A serial discovery walk alone takes one stage per interior.
        assert!(
            measured.num_stages < nodes.len() as u64,
            "sibling interior reads must overlap: {measured:?}"
        );
    }
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
    #[values(1, 8)] concurrency: usize,
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
        .resolve_fragments_concurrent(&RoaringBitmap::from_iter([0, 120]), concurrency)
        .await
        .expect_err("a child fence must stay within its parent's range");
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains("range"), "{error}");
}

/// The duplicate sequence check covers the actions routed to one leaf from
/// the root and every interior on its path, whatever the read concurrency.
#[rstest]
#[case::serial(1)]
#[case::concurrent(8)]
#[tokio::test]
async fn set_resolution_rejects_a_duplicate_sequence_on_one_replay_path(
    #[case] concurrency: usize,
) {
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
        .resolve_fragments_concurrent(&RoaringBitmap::from_iter([0, 120]), concurrency)
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

/// Reads and rewrites reject an interior action targeting a sibling's range,
/// including point and set reads that name none of the forged targets and a
/// rewrite with nothing to drain.
#[rstest]
#[case::full_read("materialize")]
#[case::stream("stream")]
#[case::prefetch_stream("prefetch_stream")]
#[case::row_offset("row_offset")]
#[case::drain("drain")]
#[case::bulk_drain("bulk")]
#[case::coalesce("coalesce")]
#[case::point("point")]
#[case::point_below_range("point_below_range")]
#[case::cached_point("cached_point")]
#[case::set("set")]
#[case::touched("touched")]
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
        action: Some(action::upsert_fragment(&duplicate)),
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
        tree.force_flush = read == "bulk";
    }
    let error = match read {
        "stream" => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
        "prefetch_stream" => Arc::new(fixture.tree)
            .fragment_stream_with_prefetch(4)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "materialize" => tree.materialize().await.err(),
        "row_offset" => tree.fragment_at_row_offset(0).await.err(),
        "point" => tree.resolve_fragment(0).await.err(),
        "point_below_range" => tree.resolve_fragment(4).await.err(),
        "cached_point" => {
            tree.set_leaf_cache(LanceCache::with_capacity(1024 * 1024));
            tree.resolve_fragment(0).await.err()
        }
        "set" => tree
            .resolve_fragments(&RoaringBitmap::from_iter([0]))
            .await
            .err(),
        "touched" => tree.resolve_touched(&[0]).await.err(),
        _ => tree.rewrite_tree().await.err(),
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
    tree.children = vec![left.child_ref, right.child_ref];
    tree.buffer = root_buffer;
    tree.buffer_index.take();
    tree.next_action_sequence = 2;
    tree.store.next_action_sequence = 2;
    tree
}

/// One sequence namespace spans the tree, so an operation that decodes two
/// buffers holding the same sequence must reject the tree, even when each
/// buffer routes its copy to a different leaf.
#[rstest]
#[case::materialize("materialize", RepeatedSequence::SiblingInteriors)]
#[case::stream("stream", RepeatedSequence::SiblingInteriors)]
#[case::prefetch_stream("prefetch_stream", RepeatedSequence::SiblingInteriors)]
#[case::set("set", RepeatedSequence::SiblingInteriors)]
#[case::bulk("bulk", RepeatedSequence::SiblingInteriors)]
#[case::drain("drain", RepeatedSequence::SiblingInteriors)]
#[case::verify_watermarks("verify_watermarks", RepeatedSequence::SiblingInteriors)]
#[case::stream_below_root("stream", RepeatedSequence::RootAndInterior)]
#[case::stream_from_below_root("stream_from", RepeatedSequence::RootAndInterior)]
#[case::set_below_root("set", RepeatedSequence::RootAndInterior)]
#[case::point_below_root("point", RepeatedSequence::RootAndInterior)]
#[case::row_offset_below_root("row_offset", RepeatedSequence::RootAndInterior)]
#[case::bulk_below_root("bulk", RepeatedSequence::RootAndInterior)]
#[case::drain_below_root("drain", RepeatedSequence::RootAndInterior)]
#[case::verify_watermarks_below_root("verify_watermarks", RepeatedSequence::RootAndInterior)]
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
        tree.force_flush = read == "bulk";
    }
    let error = match read {
        "materialize" => tree.materialize().await.err(),
        "stream" => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
        // Starting at fragment 1 still descends the left interior, so the
        // lower bound path decodes both copies.
        "stream_from" => Arc::new(tree)
            .fragment_stream_from(1)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "prefetch_stream" => Arc::new(tree)
            .fragment_stream_with_prefetch(4)
            .try_collect::<Vec<_>>()
            .await
            .err(),
        "set" => tree
            .resolve_fragments_concurrent(&RoaringBitmap::from_iter([1, 3]), 4)
            .await
            .err(),
        "point" => tree.resolve_fragment(1).await.err(),
        "row_offset" => tree.fragment_at_row_offset(1).await.err(),
        "bulk" | "drain" => tree.rewrite_tree().await.err(),
        "verify_watermarks" => tree.verify_watermarks().await.err(),
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
    assert!(!tree.force_flush);

    let acc = tree.rewrite_tree().await.unwrap();
    assert!(acc.materialized > 0);
    assert!(acc.merges > 0);
    assert_eq!(
        tree.materialize().await.unwrap(),
        [0, 1, 3, 4]
            .into_iter()
            .map(make_fragment)
            .collect::<Vec<_>>()
    );
    tree.verify_watermarks().await.unwrap();
}

#[tokio::test]
async fn buffered_rewrite_admits_each_source_once_but_rechecks_its_range() {
    let tree = tree_repeating_a_sequence(RepeatedSequence::SiblingInteriors).await;
    let rewrite = rewrite::RewriteNodes::new(&tree.buffer);
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
    let acc = tree.rewrite_tree().await.unwrap();
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
#[case::stream_from_below_root("stream_from", RepeatedSequence::RootAndInterior, vec![3, 4])]
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
        "stream_from" => Arc::new(tree)
            .fragment_stream_from(3)
            .try_collect::<Vec<_>>()
            .await
            .unwrap(),
        other => unreachable!("no read named {other}"),
    };
    let expected: Vec<_> = expected_ids.into_iter().map(make_fragment).collect();
    assert_eq!(fragments, expected);
}

/// A read checks a buffered replacement's deltas against the stored record,
/// including a record a set read selects from a cached leaf.
#[rstest]
#[case::point("point")]
#[case::stream("stream")]
#[case::set_cold("set_cold")]
#[case::set_cached("set_cached")]
#[tokio::test]
async fn buffered_reset_with_a_misstated_delta_is_rejected(#[case] read: &str) {
    let policy = SnapshotPolicy::default();
    let mut fixture = Fixture::new(8, FragmentTreeConfig::default(), policy).await;
    let target = 3;
    let ids = RoaringBitmap::from_iter([target as u32]);
    let leaves = LanceCache::with_capacity(1024 * 1024);
    if read == "set_cached" {
        fixture.tree.set_leaf_cache(leaves.clone());
        for _ in 0..2 {
            let fragments = fixture.tree.resolve_fragments(&ids).await.unwrap();
            assert_eq!(fragments, vec![make_fragment(target)]);
        }
    }
    let mut snapshot = fixture.snapshot.clone();
    let Some(pb::fragment_tree::Root::InlineRoot(root)) = &mut snapshot.root else {
        panic!("expected inline root");
    };
    assert!(!root.children.is_empty());
    // The stored fragment is replaced by an identical record, so the true
    // deltas are zero. The buffer counts it as an insert.
    root.buffer.push(pb::FragmentTreeMutation {
        action_sequence: root.next_action_sequence,
        action: Some(action::upsert_fragment(&make_fragment(target))),
        fragment_count_delta: 1,
        total_rows_delta: 0,
        visible_rows_delta: 0,
    });
    root.next_action_sequence += 1;
    snapshot.next_action_sequence = root.next_action_sequence;
    let mut tree = fixture.open(&snapshot, fixture.tree.version()).await;
    if read == "set_cached" {
        tree.set_leaf_cache(leaves);
    }
    fixture.io.incremental_stats();
    let error = match read {
        "point" => tree.resolve_fragment(target).await.err(),
        "set_cold" | "set_cached" => tree.resolve_fragments(&ids).await.err(),
        _ => tree.iter_fragments().try_collect::<Vec<_>>().await.err(),
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
        fixture
            .tree
            .set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
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
