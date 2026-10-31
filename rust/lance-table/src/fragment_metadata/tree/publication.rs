// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prepare immutable tree objects for publication by an authoritative manifest.
//! This path never writes a version-numbered root or races a second commit CAS.

use super::*;
use crate::format::pb::fragment_tree::Root;
use crate::fragment_metadata::store::{NodeLocation, inherit_refs, root_path, stamp_unset_base_id};

/// Publication byte budgets, independent of leaf and semantic-buffer budgets.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotPolicy {
    /// Embed root and descriptor only when their complete encoded contribution
    /// to the Version Manifest fits this size. Zero disables embedding.
    pub inline_root_bytes: usize,
    /// Maximum encoded cumulative suffix. Zero writes an external root every commit.
    pub max_suffix_bytes: usize,
}

impl Default for SnapshotPolicy {
    fn default() -> Self {
        Self {
            inline_root_bytes: 64 * 1024,
            max_suffix_bytes: 32 * 1024,
        }
    }
}

// A collapse retry restores routing. Rewrites leave frontiers and totals unchanged.
struct RootRouting {
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
}

/// The root a descriptor names, checked, with the version totals it derives
/// from the root and `mutations_since_root`.
pub(super) struct CheckedRoot {
    pub(super) root: pb::FragmentTreeRoot,
    pub(super) total_fragments: u64,
    pub(super) total_rows: u64,
    pub(super) visible_rows: u64,
}

/// Read the root `snapshot` names and run every descriptor check, before
/// opening squashes the root buffer and suffix together.
pub(super) async fn checked_root(
    store: &NodeStore,
    snapshot: &pb::FragmentTree,
    version: u64,
    next_fragment_id: u64,
) -> Result<CheckedRoot> {
    let root = match &snapshot.root {
        Some(Root::InlineRoot(root)) if snapshot.mutations_since_root.is_empty() => root.clone(),
        Some(Root::InlineRoot(_)) => {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree version {version} has a suffix after an inline root"
            )));
        }
        Some(Root::RootUuid(uuid)) => store.read_root_base(uuid).await?,
        None => {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree version {version} has no root base"
            )));
        }
    };
    let (base_fragments, base_rows, base_visible_rows) =
        super::super::validation::root(&root, next_fragment_id)?;
    super::super::validation::buffer(
        &snapshot.mutations_since_root,
        root.next_action_sequence,
        snapshot.next_action_sequence,
    )?;
    if next_fragment_id > u64::from(u32::MAX) + 1 {
        return Err(super::super::validation::corrupt(format!(
            "Invalid snapshot allocation frontier at version {version}"
        )));
    }
    super::super::validation::allocated_ids(
        snapshot
            .mutations_since_root
            .iter()
            .map(content::action_key),
        next_fragment_id,
        format_args!("mutations_since_root at version {version}"),
    )?;
    if snapshot.next_action_sequence < root.next_action_sequence {
        return Err(Error::invalid_input(format!(
            "fragment metadata tree version {version} has next_action_sequence {} below the root's {}",
            snapshot.next_action_sequence, root.next_action_sequence
        )));
    }
    // Mutations may be serialized in any order, so each column is summed
    // before it is applied, as buffers sum theirs.
    let suffix = &snapshot.mutations_since_root;
    let total_fragments = apply_aggregate_delta(
        base_fragments,
        sum_aggregate_deltas(
            suffix.iter().map(|action| action.fragment_count_delta),
            "Fragments",
        )?,
        "Fragments",
    )?;
    let total_rows = apply_aggregate_delta(
        base_rows,
        sum_aggregate_deltas(
            suffix.iter().map(|action| action.total_rows_delta),
            "physical rows",
        )?,
        "physical rows",
    )?;
    let visible_rows = apply_aggregate_delta(
        base_visible_rows,
        sum_aggregate_deltas(
            suffix.iter().map(|action| action.visible_rows_delta),
            "visible rows",
        )?,
        "visible rows",
    )?;
    if visible_rows > total_rows {
        return Err(super::super::validation::corrupt(format!(
            "Derived visible rows {visible_rows} exceed physical rows {total_rows} at version {version}"
        )));
    }
    Ok(CheckedRoot {
        root,
        total_fragments,
        total_rows,
        visible_rows,
    })
}

impl FragmentTree {
    /// Immutable node paths required by this snapshot. Root bases and
    /// mutations_since_root belong to the manifest descriptor.
    #[cfg(any(test, feature = "test-util"))]
    pub async fn node_paths(&self) -> Result<Vec<String>> {
        Ok(self
            .resolved_node_paths()
            .await?
            .into_iter()
            .map(|(path, _)| path)
            .collect())
    }

    /// Paths of the objects this snapshot needs that live in the dataset at
    /// `base` on `store`. Interior and leaf nodes are listed where their
    /// reference resolves, whichever base id names it. The external root is
    /// listed only for this snapshot's own dataset, because a root is read
    /// and written only there. A branch reads its source's nodes in place, so
    /// asking with the source's location lists what the branch keeps alive in
    /// the source, and asking with the branch's location lists only the
    /// branch's own objects.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_core::Result;
    /// # use lance_io::object_store::ObjectStore;
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # use object_store::path::Path;
    /// # async fn example(tree: &FragmentTree, store: &Arc<ObjectStore>, base: &Path) -> Result<()> {
    /// let retained = tree.object_paths_owned_by(store, base).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn object_paths_owned_by(
        &self,
        store: &Arc<ObjectStore>,
        base: &Path,
    ) -> Result<Vec<String>> {
        let mut owned = Vec::new();
        if let Some(root) = self.root_object_path()? {
            let own =
                self.store
                    .location_under(&self.store.object_store, self.store.base(), &root)?;
            if own == self.store.location_under(store, base, &root)? {
                owned.push(root);
            }
        }
        for (path, location) in self.resolved_node_paths().await? {
            if location == self.store.location_under(store, base, &path)? {
                owned.push(path);
            }
        }
        owned.sort();
        Ok(owned)
    }

    fn root_object_path(&self) -> Result<Option<String>> {
        match &self.descriptor.root {
            Some(Root::RootUuid(uuid)) => Ok(Some(root_path(uuid)?)),
            Some(Root::InlineRoot(_)) | None => Ok(None),
        }
    }

    async fn resolved_node_paths(&self) -> Result<Vec<(String, NodeLocation)>> {
        let mut pending: Vec<_> =
            node::with_exclusive_ends(self.children.clone(), self.next_fragment_id).collect();
        let mut locations = std::collections::BTreeSet::new();
        let mut paths = Vec::new();
        while let Some((child, end)) = pending.pop() {
            let location = self.store.child_location(&child)?;
            if !locations.insert(location.clone()) {
                return Err(Error::invalid_input(format!(
                    "fragment metadata tree version {} repeats resolved node {} (base_id={:?})",
                    self.version, child.path, child.base_id
                )));
            }
            if child.height > 0 {
                // Cleanup lists every retained version, and versions share
                // most interiors, so listing reads them through the cache
                // their walks use.
                let interior = self.store.read_internal_shared(&child, end).await?;
                pending.extend(node::with_exclusive_ends(
                    interior.node.children.clone(),
                    end,
                ));
            }
            paths.push((child.path, location));
        }
        paths.sort();
        Ok(paths)
    }

    /// Build immutable leaves and routing without publishing a version.
    /// Publish the returned tree's descriptor through the dataset's Version Manifest.
    /// A failed publication leaves only unreachable immutable objects.
    /// `fragments` is left sorted by id and without metadata size hints, so a
    /// caller that keeps it holds the published list without reading the
    /// leaves back.
    pub async fn bootstrap_snapshot(
        context: TreeContext,
        config: FragmentTreeConfig,
        fragments: &mut [Fragment],
        version: u64,
        policy: SnapshotPolicy,
    ) -> Result<(Self, BootstrapStats)> {
        fragments.sort_by_key(|fragment| fragment.id);
        for fragment in fragments.iter_mut() {
            content::clear_metadata_size_hints(fragment);
        }
        if fragments.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(Error::invalid_input(
                "fragment metadata tree bootstrap contains duplicate fragment IDs",
            ));
        }
        let store = context.into_store(&config, 1);
        Self::build(store, config, fragments, version, policy).await
    }

    /// Bootstrap an eager snapshot, retaining the same owned records written to leaves.
    /// Use [`Self::bootstrap_snapshot`] when the caller does not need residency.
    ///
    /// ```
    /// # use lance_table::{format::Fragment, fragment_metadata::{
    /// #     FragmentTree, FragmentTreeConfig, SnapshotPolicy, TreeContext,
    /// # }};
    /// # async fn example(context: TreeContext, fragments: Vec<Fragment>) -> lance_core::Result<()> {
    /// let (tree, stats) = FragmentTree::bootstrap_resident_snapshot(
    ///     context, FragmentTreeConfig::default(), fragments, 1, SnapshotPolicy::default(),
    /// ).await?;
    /// assert!(tree.resident_fragments().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn bootstrap_resident_snapshot(
        context: TreeContext,
        config: FragmentTreeConfig,
        mut fragments: Vec<Fragment>,
        version: u64,
        policy: SnapshotPolicy,
    ) -> Result<(Self, BootstrapStats)> {
        let (mut tree, stats) =
            Self::bootstrap_snapshot(context, config, &mut fragments, version, policy).await?;
        tree.resident = Some(Arc::new(fragments));
        Ok((tree, stats))
    }

    /// Open fragment state using only a manifest descriptor and its named base.
    /// `version` comes from that same authoritative manifest.
    pub async fn open_snapshot(
        context: TreeContext,
        snapshot: &pb::FragmentTree,
        version: u64,
        config: FragmentTreeConfig,
        next_fragment_id: u64,
    ) -> Result<Self> {
        config.validate()?;
        let mut store = context.into_store(&config, snapshot.next_action_sequence);
        let CheckedRoot {
            root,
            total_fragments,
            total_rows,
            visible_rows,
        } = checked_root(&store, snapshot, version, next_fragment_id).await?;
        // Interiors below the root were written before it, so the format
        // bounds them by its frontier, as validate does.
        store.next_action_sequence = root.next_action_sequence;
        let mut buffer = root.buffer;
        buffer.extend(snapshot.mutations_since_root.iter().cloned());
        let buffer = content::squash_buffer(
            buffer,
            node::combine_from(&root.children, root.next_action_sequence),
        );
        Ok(Self {
            store,
            config,
            version,
            children: root.children,
            buffer,
            buffer_index: OnceLock::new(),
            next_action_sequence: snapshot.next_action_sequence,
            contiguous_from: root.next_action_sequence,
            total_fragments,
            total_rows,
            visible_rows,
            next_fragment_id,
            descriptor: Arc::new(snapshot.clone()),
            resident: None,
        })
    }

    pub(super) async fn prepare_snapshot_inner(
        &mut self,
        commit: ValidatedCommit,
        deltas: Vec<commit::ActionDeltas>,
        policy: SnapshotPolicy,
        mode: CommitMode,
    ) -> Result<CommitStats> {
        let previous = self.descriptor.clone();
        let tagged = self.stage_commit(commit, deltas)?;
        let messages_in = tagged.len() as u64;
        // Only an external root can publish this commit as a suffix, so only
        // then does the suffix need its own copy of the actions.
        let suffix = (mode == CommitMode::Incremental
            && matches!(&previous.root, Some(Root::RootUuid(_))))
        .then(|| {
            let mut suffix = previous.mutations_since_root.clone();
            suffix.extend(tagged.iter().cloned());
            content::squash_buffer(suffix, self.contiguous_from)
        });
        self.buffer_index.take();
        self.buffer.extend(tagged);
        let before = self.buffer.len();
        self.buffer = content::squash_buffer(
            std::mem::take(&mut self.buffer),
            node::combine_from(&self.children, self.contiguous_from),
        );
        let squashed = before - self.buffer.len();
        if let Some(suffix) = suffix
            && !node::internal_overflows(&self.children, &self.buffer, &self.config)
            && node::internal_logical_bytes(&[], &suffix) <= policy.max_suffix_bytes as u64
        {
            self.descriptor = Arc::new(pb::FragmentTree {
                root: previous.root.clone(),
                mutations_since_root: suffix,
                next_action_sequence: self.next_action_sequence,
            });
            return Ok(CommitStats {
                messages_in,
                messages_squashed: squashed as u64,
                height: self.height(),
                root_buffer_len: self.buffer.len() as u64,
                ..Default::default()
            });
        }
        // A collapse starts where an ingested interior keeps a single
        // interior child, or where the root keeps a lone interior with a
        // single child. Over leaves the root flush stops once its buffer
        // fits beside its references or is empty, so the root can always
        // adopt a lone interior lifted from them. Without a root child above
        // the leaves the rewrite never falls back, so the saved copy of the
        // whole buffer would go unused.
        let before_rewrite = (mode == CommitMode::Incremental
            && self.children.iter().any(|child| child.height > 0))
        .then(|| RootRouting {
            children: self.children.clone(),
            buffer: self.buffer.clone(),
        });
        let mut acc = self.rewrite_tree(mode).await?;
        if acc.collapses > 0 {
            let Some(before_rewrite) = before_rewrite else {
                return Err(Error::internal(format!(
                    "fragment metadata tree at version {} collapsed a routing level without \
                     saved routing to retry from",
                    self.version
                )));
            };
            // Nothing written by the abandoned rewrite is referenced.
            self.buffer_index.take();
            self.children = before_rewrite.children;
            self.buffer = before_rewrite.buffer;
            let abandoned_bytes = acc.io_bytes;
            acc = self.rewrite_tree(CommitMode::Bulk).await?;
            acc.io_bytes += abandoned_bytes;
        }
        let (descriptor, bytes) =
            checkpoint(&self.store, &self.config, self.compacted_root(), policy).await?;
        self.descriptor = Arc::new(descriptor);
        // The new root bounds every interior below it, this commit's included.
        self.store.next_action_sequence = self.next_action_sequence;
        Ok(CommitStats {
            tree_write_bytes: acc.io_bytes + bytes,
            messages_in,
            messages_materialized: acc.materialized,
            messages_squashed: squashed as u64 + acc.squashed,
            flushes: acc.flushes,
            splits: acc.splits,
            merges: acc.merges,
            max_flush_depth: acc.max_flush_depth,
            height: self.height(),
            root_buffer_len: self.buffer.len() as u64,
            checkpoints: 1,
        })
    }

    /// The first snapshot of a branch whose manifest maps `source_base_id`
    /// to this snapshot's dataset, prepared for publication. The branch root
    /// keeps this snapshot's children and its buffer with the suffix folded
    /// in, so every node stays in the source. The root follows the rules of
    /// any checkpoint. It is written through `branch` only when it does not
    /// fit inline. The call fails when the root outgrows `hard_capacity_bytes`.
    /// A resident inventory stays resident, stamped as a branch read stamps it.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::{FragmentTree, SnapshotPolicy, TreeContext};
    /// # async fn example(source: FragmentTree, branch: TreeContext) -> lance_core::Result<()> {
    /// let branch = source
    ///     .branch_snapshot(branch, 7, SnapshotPolicy::default())
    ///     .await?;
    /// assert!(branch.descriptor().mutations_since_root.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn branch_snapshot(
        mut self,
        branch: TreeContext,
        source_base_id: u32,
        policy: SnapshotPolicy,
    ) -> Result<Self> {
        inherit_refs(&mut self.children, &mut self.buffer, source_base_id)?;
        self.buffer_index.take();
        if let Some(resident) = self.resident.take() {
            let mut fragments = Arc::unwrap_or_clone(resident);
            for fragment in &mut fragments {
                stamp_unset_base_id(fragment, source_base_id);
            }
            self.resident = Some(Arc::new(fragments));
        }
        self.store = branch.into_store(&self.config, self.next_action_sequence);
        let (descriptor, _) =
            checkpoint(&self.store, &self.config, self.compacted_root(), policy).await?;
        self.descriptor = Arc::new(descriptor);
        Ok(self)
    }
}

/// The descriptor that publishes `root` with an empty suffix, inline when it
/// fits `policy` and otherwise written to `store`, with the bytes written.
pub(super) async fn checkpoint(
    store: &NodeStore,
    config: &FragmentTreeConfig,
    root: pb::FragmentTreeRoot,
    policy: SnapshotPolicy,
) -> Result<(pb::FragmentTree, u64)> {
    if root.encoded_len() as u64 > config.hard_capacity_bytes {
        return Err(Error::invalid_input(format!(
            "fragment metadata root envelope requires {} encoded bytes, exceeding hard_capacity_bytes={}",
            root.encoded_len(),
            config.hard_capacity_bytes
        )));
    }
    let next_action_sequence = root.next_action_sequence;
    let mut snapshot = pb::FragmentTree {
        root: Some(Root::InlineRoot(root)),
        mutations_since_root: Vec::new(),
        next_action_sequence,
    };
    if snapshot.encoded_len() <= policy.inline_root_bytes {
        return Ok((snapshot, 0));
    }
    let Some(Root::InlineRoot(root)) = &snapshot.root else {
        return Err(Error::internal(
            "prepared fragment metadata snapshot has no inline root",
        ));
    };
    let (uuid, bytes) = store.write_root_base(root).await?;
    snapshot.root = Some(Root::RootUuid(uuid));
    Ok((snapshot, bytes))
}
