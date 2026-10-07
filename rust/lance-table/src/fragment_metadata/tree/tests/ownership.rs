// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
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
            Arc::new(LanceCache::with_capacity(0)),
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
    children.push(local.clone());
    crate::fragment_metadata::validation::children(&children, 1, 0).unwrap();
    fixture.tree.children = children.clone();
    fixture.tree.set_foreign_bases(bases.clone());
    assert_eq!(fixture.tree.node_paths().await.unwrap().len(), 3);
    assert_eq!(
        fixture.tree.local_node_paths().await.unwrap(),
        vec![local.path.clone()]
    );
    // Ownership follows the resolved location, so a same name leaf under
    // another base is not this owner's, and the fixture's own base id names
    // the same nodes as the local view.
    for child in &children {
        let owner = child.base_id.unwrap();
        let (store, base) = &bases[&owner];
        assert_eq!(
            fixture.tree.node_paths_owned_by(store, base).await.unwrap(),
            vec![child.path.clone()],
            "owner {owner}"
        );
    }
    assert_eq!(
        fixture
            .tree
            .node_paths_owned_by(&fixture.store, &fixture.base)
            .await
            .unwrap(),
        vec![local.path]
    );

    bases.insert(3, bases[&1].clone());
    fixture.tree.set_foreign_bases(bases);
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
    fixture.tree.children = roots;
    fixture.tree.set_foreign_bases(bases.clone());
    assert_eq!(fixture.tree.node_paths().await.unwrap().len(), 6);
    assert!(fixture.tree.local_node_paths().await.unwrap().is_empty());
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
            fixture.tree.node_paths_owned_by(store, base).await.unwrap(),
            owned
        );
    }
}
