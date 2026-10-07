// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::{DeletionFile, DeletionFileType};
use object_store::ObjectStoreExt;

fn small_config() -> FragmentTreeConfig {
    FragmentTreeConfig {
        max_node_bytes: 4 * 1024,
        max_leaf_bytes: 16 * 1024,
        semantic_buffer_bytes: 2 * 1024,
        ..FragmentTreeConfig::default()
    }
}

fn deletion(id: u64, round: u64) -> pb::FragmentAction {
    action::add_deletion_file(
        id,
        &DeletionFile {
            read_version: round,
            id: round,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(1),
            base_id: None,
        },
    )
}

const FRAGMENTS: u64 = 2048;

/// A multi-level tree whose interiors hold buffered deletions.
pub(super) async fn buffered_fixture() -> Fixture {
    let mut fixture = Fixture::new(FRAGMENTS, small_config(), SnapshotPolicy::default()).await;
    for round in 1..=40u64 {
        let actions = (0..16)
            .map(|slot| deletion((round * 16 + slot * 127) % FRAGMENTS, round))
            .collect();
        fixture
            .commit(actions, SnapshotPolicy::default(), false)
            .await;
    }
    let shape = fixture.tree.shape_report().await.unwrap();
    assert!(shape.height >= 2, "{shape:?}");
    assert!(shape.node_buffer_lens.iter().sum::<u64>() > 0, "{shape:?}");
    fixture
}

fn interior_gets(stats: &lance_io::utils::tracking_store::IoStats) -> usize {
    stats
        .requests
        .iter()
        .filter(|request| {
            request.path.as_ref().contains("_bt/node/") && request.method.contains("get")
        })
        .count()
}

/// A spread of ids that routes through every subtree.
async fn resolve_each(tree: &FragmentTree) -> Vec<Result<Option<Fragment>>> {
    let mut resolved = Vec::new();
    for id in (0..FRAGMENTS).step_by(31) {
        resolved.push(tree.resolve_fragment(id).await);
    }
    resolved
}

fn same_outcomes(left: &[Result<Option<Fragment>>], right: &[Result<Option<Fragment>>]) -> bool {
    left.iter().zip(right).all(|pair| match pair {
        (Ok(left), Ok(right)) => left == right,
        (Err(left), Err(right)) => std::mem::discriminant(left) == std::mem::discriminant(right),
        _ => false,
    })
}

#[rstest]
#[case::sparse(7)]
#[case::dense(1)]
#[tokio::test]
async fn warm_reads_perform_no_interior_gets(#[case] id_step: usize) {
    let fixture = buffered_fixture().await;
    let expected = fixture.tree.materialize().await.unwrap();
    let mut tree = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    tree.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let tree = Arc::new(tree);
    let ids: Vec<u64> = (0..FRAGMENTS).step_by(id_step).collect();
    let bitmap: RoaringBitmap = ids.iter().map(|id| *id as u32).collect();

    let touched = tree.resolve_touched(&ids).await.unwrap();
    let streamed: Vec<_> = tree.clone().fragment_stream().try_collect().await.unwrap();
    assert_eq!(streamed, expected);
    fixture.io.incremental_stats();

    assert_eq!(
        tree.resolve_touched(&ids).await.unwrap().fragments,
        touched.fragments
    );
    let resolved = tree.resolve_fragments(&bitmap).await.unwrap();
    let selected: Vec<_> = expected
        .iter()
        .filter(|fragment| bitmap.contains(fragment.id as u32))
        .cloned()
        .collect();
    assert_eq!(resolved, selected);
    assert!(touched.fragments.values().eq(selected.iter()));
    for fragment in &resolved {
        assert_eq!(
            tree.resolve_fragment(fragment.id).await.unwrap().as_ref(),
            Some(fragment)
        );
    }
    let streamed: Vec<_> = tree.clone().fragment_stream().try_collect().await.unwrap();
    assert_eq!(streamed, expected);
    assert_eq!(tree.materialize().await.unwrap(), expected);
    assert_eq!(interior_gets(&fixture.io.incremental_stats()), 0);
}

/// A node cached by a newer reader is rejected by an older reader's
/// frontier, exactly as a fresh decode rejects it.
#[tokio::test]
async fn cached_interior_fails_a_stricter_frontier() {
    let mut fixture = Fixture::new(FRAGMENTS, small_config(), SnapshotPolicy::default()).await;
    let (old_snapshot, old_version) = (fixture.snapshot.clone(), fixture.tree.version());
    for round in 1..=40u64 {
        let actions = (0..16)
            .map(|slot| deletion((round * 16 + slot * 127) % FRAGMENTS, round))
            .collect();
        fixture
            .commit(actions, SnapshotPolicy::default(), false)
            .await;
    }
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let mut newer = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    newer.set_leaf_cache(cache.clone());
    assert!(resolve_each(&newer).await.iter().all(Result::is_ok));

    let forge = |tree: &mut FragmentTree| {
        tree.children = newer.children.clone();
        tree.buffer.clear();
        tree.buffer_index.take();
    };
    let mut cached = fixture.open(&old_snapshot, old_version).await;
    cached.set_leaf_cache(cache.clone());
    forge(&mut cached);
    let mut uncached = fixture.open(&old_snapshot, old_version).await;
    forge(&mut uncached);
    let hits = cache.stats().await.hits;
    let from_cache = resolve_each(&cached).await;
    assert!(cache.stats().await.hits > hits);
    let rejected: Vec<_> = from_cache.iter().filter(|result| result.is_err()).collect();
    assert!(!rejected.is_empty());
    assert!(
        rejected
            .iter()
            .all(|result| matches!(result, Err(Error::CorruptFile { .. })))
    );
    assert!(same_outcomes(&from_cache, &resolve_each(&uncached).await));

    let hits = cache.stats().await.hits;
    let from_cache = cached.materialize().await.unwrap_err();
    assert!(cache.stats().await.hits > hits);
    assert!(
        matches!(from_cache, Error::CorruptFile { .. }),
        "{from_cache}"
    );
    let fresh = uncached.materialize().await.unwrap_err();
    assert!(matches!(fresh, Error::CorruptFile { .. }), "{fresh}");
}

/// The bound of a cached interior's range that a forged right sibling moves
/// its end onto.
#[derive(Clone, Copy, Debug)]
enum NarrowedEnd {
    /// The fragment the interior buffers an insert for.
    BufferedTarget,
    /// The fence of the interior's last child.
    LastFence,
}

/// The cache key omits where an interior's range ends, so a cached interior
/// is checked again against the end of each reference that reaches it.
#[rstest]
#[case::buffered_target(NarrowedEnd::BufferedTarget, "buffers actions")]
#[case::last_fence(NarrowedEnd::LastFence, "child fence")]
#[tokio::test]
async fn cached_interior_fails_a_narrower_range(
    #[case] narrowed: NarrowedEnd,
    #[case] violation: &str,
) {
    let config = FragmentTreeConfig::default().with_semantic_buffer_bytes(1);
    let mut fixture = Fixture::new(0, config, SnapshotPolicy::default()).await;
    let mut leaves = Vec::new();
    for id in 0..3 {
        let written = fixture
            .tree
            .store
            .write_leaves(&[make_fragment(id)], None, 0, &fixture.tree.config)
            .await
            .unwrap();
        leaves.push(written.into_iter().next().unwrap().child_ref);
    }
    // The interior holds fragments 0 and 1 and buffers the insert of 2.
    let insert = pb::FragmentTreeMutation {
        action_sequence: 1,
        action: Some(action::upsert_fragment(&make_fragment(2))),
        fragment_count_delta: 1,
        total_rows_delta: 1,
        visible_rows_delta: 1,
    };
    let interior = fixture
        .tree
        .store
        .write_internal(leaves[..2].to_vec(), vec![insert])
        .await
        .unwrap()
        .child_ref;
    let tree = &mut fixture.tree;
    tree.children = vec![interior];
    tree.buffer.clear();
    tree.buffer_index.take();
    tree.total_fragments = 3;
    tree.total_rows = 3;
    tree.visible_rows = 3;
    tree.next_action_sequence = 2;
    tree.store.next_action_sequence = 2;
    let cache = LanceCache::with_capacity(1024 * 1024);
    tree.set_leaf_cache(cache.clone());
    assert!(tree.resolve_fragment(0).await.unwrap().is_some());
    assert!(tree.resolve_fragment(2).await.unwrap().is_some());

    let mut forged = tree.clone();
    forged.children.push(match narrowed {
        NarrowedEnd::BufferedTarget => leaves[2].clone(),
        NarrowedEnd::LastFence => leaves[1].clone(),
    });
    let hits = cache.stats().await.hits;
    let error = forged.resolve_fragment(0).await.unwrap_err();
    assert!(cache.stats().await.hits > hits);
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    assert!(error.to_string().contains(violation), "{error}");
}

#[rstest]
#[case::num_keys(|child: &mut pb::FragmentTreeChild| child.num_keys += 1)]
#[case::total_rows(|child: &mut pb::FragmentTreeChild| child.total_rows += 1)]
#[case::object_size(|child: &mut pb::FragmentTreeChild| child.object_size += 1)]
#[tokio::test]
async fn changed_parent_reference_is_checked_again(#[case] forge: fn(&mut pb::FragmentTreeChild)) {
    let fixture = buffered_fixture().await;
    let mut tree = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    tree.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let valid = resolve_each(&tree).await;
    assert!(valid.iter().all(Result::is_ok));
    let interior = tree
        .children
        .iter()
        .position(|child| child.height > 0)
        .unwrap();

    let mut forged = tree.clone();
    forge(&mut forged.children[interior]);
    let mut uncached = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    forge(&mut uncached.children[interior]);
    let from_cache = resolve_each(&forged).await;
    assert!(from_cache.iter().any(Result::is_err));
    assert!(same_outcomes(&from_cache, &resolve_each(&uncached).await));
    assert!(same_outcomes(&valid, &resolve_each(&tree).await));
}

/// Two datasets name their interiors by one relative path, and the source
/// reads the same object without a base id. One cache keeps all three views.
#[tokio::test]
async fn owner_views_of_one_path_stay_apart() {
    let mut fixture =
        Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let mut roots = Vec::new();
    let mut bases = HashMap::new();
    for owner in 1..=2u64 {
        let base = Path::from(format!("source-{owner}"));
        let source = NodeStore::new(
            fixture.store.clone(),
            base.clone(),
            fixture.scheduler.clone(),
            Arc::new(LanceCache::with_capacity(0)),
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
    fixture.tree.children = roots.clone();
    fixture.tree.set_foreign_bases(bases);
    fixture.tree.set_leaf_cache(cache.clone());
    let mut source_view = fixture.tree.clone();
    source_view.store = NodeStore::new(
        fixture.store.clone(),
        Path::from("source-1"),
        fixture.scheduler.clone(),
        Arc::new(LanceCache::with_capacity(0)),
    );
    source_view.store.next_action_sequence = fixture.tree.store.next_action_sequence;
    source_view.set_leaf_cache(cache.clone());
    let mut local_root = roots[0].clone();
    local_root.base_id = None;
    source_view.children = vec![local_root];

    for _ in 0..2 {
        for id in 0..4u64 {
            let fragment = fixture.tree.resolve_fragment(id).await.unwrap().unwrap();
            let owner = (id / 2 + 1) as u32;
            assert!(
                fragment
                    .files
                    .iter()
                    .all(|file| file.base_id == Some(owner))
            );
        }
        for id in 0..2u64 {
            let fragment = source_view.resolve_fragment(id).await.unwrap().unwrap();
            assert!(fragment.files.iter().all(|file| file.base_id.is_none()));
        }
    }
}

/// A budget far below the interiors forces evictions and reloads without
/// changing a single result.
#[tokio::test]
async fn evicted_interiors_reload_with_the_same_results() {
    let fixture = buffered_fixture().await;
    let expected = fixture.tree.materialize().await.unwrap();
    let mut tree = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    tree.set_leaf_cache(LanceCache::with_capacity(16 * 1024));
    let tree = Arc::new(tree);
    for _ in 0..2 {
        fixture.io.incremental_stats();
        let streamed: Vec<_> = tree.clone().fragment_stream().try_collect().await.unwrap();
        assert_eq!(streamed, expected);
        assert!(interior_gets(&fixture.io.incremental_stats()) > 0);
        for fragment in expected.iter().step_by(64) {
            assert_eq!(
                tree.resolve_fragment(fragment.id).await.unwrap().as_ref(),
                Some(fragment)
            );
        }
    }
}

/// Row ids decode as slices of the fetched object. A cached interior holds
/// its own copy, so the object it came from can be freed.
#[tokio::test]
async fn cached_interior_owns_its_row_ids() {
    let mut fixture =
        Fixture::new(0, FragmentTreeConfig::default(), SnapshotPolicy::default()).await;
    let leaves = vec![
        fixture
            .tree
            .store
            .write_leaf(&[make_fragment(0)], 0)
            .await
            .unwrap()
            .child_ref,
        fixture
            .tree
            .store
            .write_leaf(&[make_fragment(100)], 0)
            .await
            .unwrap()
            .child_ref,
    ];
    let buffer = (1..=8u64)
        .map(|id| {
            let mut fragment = make_fragment(id);
            fragment.physical_rows = Some(64);
            fragment.row_id_meta = Some(crate::format::RowIdMeta::Inline(
                crate::format::InlineRowIds::from(crate::rowids::write_row_ids(
                    &crate::rowids::RowIdSequence::from(id * 1000..id * 1000 + 64),
                )),
            ));
            pb::FragmentTreeMutation {
                action_sequence: id,
                action: Some(action::upsert_fragment(&fragment)),
                fragment_count_delta: 1,
                total_rows_delta: 64,
                visible_rows_delta: 64,
            }
        })
        .collect();
    let interior = fixture
        .tree
        .store
        .write_internal(leaves, buffer)
        .await
        .unwrap()
        .child_ref;
    fixture.tree.store.next_action_sequence = 9;
    fixture
        .tree
        .set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let cached = fixture
        .tree
        .store
        .read_internal_shared(&interior, node::ROOT_EXCLUSIVE_END)
        .await
        .unwrap();
    for mutation in &cached.node.buffer {
        let Some(pb::fragment_action::Action::UpsertFragment(fragment)) = mutation
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref())
        else {
            panic!("expected an upsert");
        };
        let Some(pb::data_fragment::RowIdSequence::InlineRowIds(ids)) = &fragment.row_id_sequence
        else {
            panic!("expected inline row ids");
        };
        assert!(ids.is_unique());
    }
}

/// A failed interior load is not cached, and a cached interior does not hide
/// a missing object from reachability checks.
#[tokio::test]
async fn missing_interior_is_neither_cached_nor_hidden_from_verification() {
    let fixture = buffered_fixture().await;
    let interior = fixture
        .tree
        .children
        .iter()
        .find(|child| child.height > 0)
        .unwrap()
        .clone();
    let location = Path::from(format!("{}/{}", fixture.base, interior.path));
    let bytes = fixture
        .store
        .inner
        .get(&location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let mut warm = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    warm.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    assert!(resolve_each(&warm).await.iter().all(Result::is_ok));
    fixture.store.inner.delete(&location).await.unwrap();
    let error = warm.verify_reachable().await.unwrap_err();
    assert!(
        matches!(error, Error::IO { .. } | Error::NotFound { .. }),
        "{error}"
    );

    let mut cold = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    cold.set_leaf_cache(LanceCache::with_capacity(64 * 1024 * 1024));
    let failed = resolve_each(&cold).await;
    let uncached = fixture
        .open(&fixture.snapshot, fixture.tree.version())
        .await;
    assert!(failed.iter().any(Result::is_err));
    assert!(same_outcomes(&failed, &resolve_each(&uncached).await));
    fixture
        .store
        .inner
        .put(&location, bytes.into())
        .await
        .unwrap();
    assert!(same_outcomes(
        &resolve_each(&cold).await,
        &resolve_each(&warm).await
    ));
}
