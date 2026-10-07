// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prepare immutable tree objects for publication by an authoritative manifest.
//! This path never writes a version-numbered root or races a second commit CAS.

use super::*;
use crate::format::pb::fragment_tree::Root;
use crate::fragment_metadata::store::NodeLocation;

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

impl FragmentTree {
    /// Resolve validation state while retaining complete leaf reads on local
    /// scratch storage for a subsequent bulk materialization. Scratch is owned
    /// by this writer and never referenced by a published snapshot. Repeated
    /// calls before [`Self::prepare_snapshot`] share one set of retained reads.
    pub async fn resolve_touched_for_bulk(&mut self, ids: &[u64]) -> Result<TouchedFragments> {
        self.store.retain_validation_reads()?;
        self.resolve_touched(ids).await
    }

    /// Immutable node paths required by this snapshot. Root bases and
    /// mutations_since_root belong to the manifest descriptor.
    pub async fn node_paths(&self) -> Result<Vec<String>> {
        Ok(self
            .resolved_node_paths()
            .await?
            .into_iter()
            .map(|(path, _)| path)
            .collect())
    }

    /// Node paths owned by this dataset, for local retention cleanup.
    /// Foreign nodes are traversed but are not local GC roots.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # async fn example(tree: &FragmentTree) -> Result<()> {
    /// let retained_nodes = tree.local_node_paths().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn local_node_paths(&self) -> Result<Vec<String>> {
        let store = self.store.object_store.clone();
        self.node_paths_owned_by(&store, self.store.base()).await
    }

    /// Node paths that resolve into the dataset at `base` on `store`,
    /// whichever base id names it from this tree. A shallow clone or branch
    /// reads its source's leaves and interior nodes in place, so the source's
    /// cleanup keeps every node one of its clones still reaches through this.
    /// Nodes owned by any other dataset are traversed and left out.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_io::object_store::ObjectStore;
    /// # use lance_table::fragment_metadata::FragmentTree;
    /// # use object_store::path::Path;
    /// # use std::sync::Arc;
    /// # async fn example(
    /// #     branch_tree: &FragmentTree,
    /// #     source_store: &Arc<ObjectStore>,
    /// #     source_base: &Path,
    /// # ) -> Result<()> {
    /// let retained_by_source = branch_tree
    ///     .node_paths_owned_by(source_store, source_base)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn node_paths_owned_by(
        &self,
        store: &Arc<ObjectStore>,
        base: &Path,
    ) -> Result<Vec<String>> {
        let mut owned = Vec::new();
        for (path, location) in self.resolved_node_paths().await? {
            if location == self.store.location_under(store, base, &path)? {
                owned.push(path);
            }
        }
        Ok(owned)
    }

    async fn resolved_node_paths(&self) -> Result<Vec<(String, NodeLocation)>> {
        let mut pending = self.children.clone();
        let mut locations = std::collections::BTreeSet::new();
        let mut paths = Vec::new();
        while let Some(child) = pending.pop() {
            let location = self.store.child_location(&child)?;
            if !locations.insert(location.clone()) {
                return Err(Error::invalid_input(format!(
                    "fragment metadata tree version {} repeats resolved node {} (base_id={:?})",
                    self.version, child.path, child.base_id
                )));
            }
            paths.push((child.path.clone(), location));
            if child.height > 0 {
                pending.extend(self.store.read_internal(&child).await?.children);
            }
        }
        paths.sort();
        Ok(paths)
    }

    /// Build immutable leaves and routing without publishing a version.
    /// Publish the returned snapshot through the dataset's Version Manifest.
    /// A failed publication leaves only unreachable immutable objects.
    /// `fragments` is left sorted by id, so a caller that keeps it holds the
    /// published list without reading the leaves back.
    #[allow(clippy::too_many_arguments)]
    pub async fn bootstrap_snapshot(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
        config: FragmentTreeConfig,
        fragments: &mut [Fragment],
        version: u64,
        policy: SnapshotPolicy,
    ) -> Result<(Self, pb::FragmentTree, BootstrapStats)> {
        fragments.sort_by_key(|fragment| fragment.id);
        if fragments.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(Error::invalid_input(
                "fragment metadata tree bootstrap contains duplicate fragment IDs",
            ));
        }
        let store = NodeStore::new(object_store, base, scheduler, cache);
        let (mut tree, mut stats) = Self::build(store, config, fragments).await?;
        tree.version = version;
        let (snapshot, bytes) = tree.checkpoint_snapshot(policy).await?;
        tree.snapshot = Some(Box::new(snapshot.clone()));
        stats.io_write_bytes += bytes;
        Ok((tree, snapshot, stats))
    }

    /// Open fragment state using only a manifest descriptor and its named base.
    /// `version` comes from that same authoritative manifest.
    #[allow(clippy::too_many_arguments)]
    pub async fn open_snapshot(
        object_store: Arc<ObjectStore>,
        base: Path,
        scheduler: Arc<ScanScheduler>,
        cache: Arc<LanceCache>,
        snapshot: &pb::FragmentTree,
        version: u64,
        config: FragmentTreeConfig,
        next_fragment_id: u64,
    ) -> Result<Self> {
        let mut store = NodeStore::new(object_store, base, scheduler, cache);
        store.next_action_sequence = snapshot.next_action_sequence;
        let root = match &snapshot.root {
            Some(Root::InlineRoot(root)) if snapshot.mutations_since_root.is_empty() => {
                root.clone()
            }
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
        store.hard_capacity_bytes = config.hard_capacity_bytes;
        super::super::validation::buffer(
            &snapshot.mutations_since_root,
            root.next_action_sequence,
            snapshot.next_action_sequence,
        )?;
        if next_fragment_id > u64::from(u32::MAX) + 1
            || snapshot
                .mutations_since_root
                .iter()
                .any(|tagged| node::action_key(tagged) >= next_fragment_id)
        {
            return Err(super::super::validation::corrupt(format!(
                "Invalid snapshot allocation frontier at version {version}"
            )));
        }
        if snapshot.next_action_sequence < root.next_action_sequence {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree version {version} has next_action_sequence {} below the root's {}",
                snapshot.next_action_sequence, root.next_action_sequence
            )));
        }
        if snapshot.mutations_since_root.iter().any(|action| {
            action.action_sequence < root.next_action_sequence
                || action.action_sequence >= snapshot.next_action_sequence
        }) {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree version {version} mutations_since_root crosses its root action sequence frontier"
            )));
        }
        let total_fragments = snapshot
            .mutations_since_root
            .iter()
            .try_fold(base_fragments, |value, action| {
                apply_aggregate_delta(value, action.fragment_count_delta, "Fragments")
            })?;
        let total_rows = snapshot
            .mutations_since_root
            .iter()
            .try_fold(base_rows, |value, action| {
                apply_aggregate_delta(value, action.total_rows_delta, "physical rows")
            })?;
        let visible_rows = snapshot
            .mutations_since_root
            .iter()
            .try_fold(base_visible_rows, |value, action| {
                apply_aggregate_delta(value, action.visible_rows_delta, "visible rows")
            })?;
        if visible_rows > total_rows {
            return Err(super::super::validation::corrupt(format!(
                "Derived visible rows {visible_rows} exceed physical rows {total_rows} at version {version}"
            )));
        }
        let mut buffer = root.buffer;
        buffer.extend(snapshot.mutations_since_root.iter().cloned());
        let buffer = node::squash_buffer(
            buffer,
            node::combine_from(&root.children, root.next_action_sequence),
        );
        config.validate()?;
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
            force_flush: false,
            snapshot: Some(Box::new(snapshot.clone())),
        })
    }

    /// Prepare a snapshot without publishing a version. An error or dropped future
    /// leaves this tree's fragment state unchanged. Unreferenced objects written
    /// during preparation remain until retention cleanup removes them.
    pub async fn prepare_snapshot(
        &mut self,
        commit: ValidatedCommit,
        touched: &TouchedFragments,
        previous_snapshot: &pb::FragmentTree,
        policy: SnapshotPolicy,
        bulk: bool,
    ) -> Result<(pb::FragmentTree, CommitStats)> {
        if self.snapshot.as_deref() != Some(previous_snapshot) {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree version {} received a descriptor from another validation state",
                self.version
            )));
        }
        let deltas = commit::aggregate_deltas(
            &commit.fragment_actions,
            touched,
            self.next_fragment_id,
            self.store.base(),
        )?;
        // Keep partial rewrites private if preparation fails or is cancelled.
        let mut staged = self.clone();
        self.store.clear_validation_reads();
        let (snapshot, stats) = staged
            .prepare_snapshot_inner(commit, deltas, previous_snapshot, policy, bulk)
            .await?;
        staged.force_flush = false;
        staged.store.clear_validation_reads();
        staged.snapshot = Some(Box::new(snapshot.clone()));
        *self = staged;
        Ok((snapshot, stats))
    }

    async fn prepare_snapshot_inner(
        &mut self,
        commit: ValidatedCommit,
        deltas: Vec<commit::ActionDeltas>,
        previous: &pb::FragmentTree,
        policy: SnapshotPolicy,
        bulk: bool,
    ) -> Result<(pb::FragmentTree, CommitStats)> {
        let tagged = self.stage_commit(commit, deltas)?;
        let messages_in = tagged.len() as u64;
        // Only an external root can publish this commit as a suffix, so only
        // then does the suffix need its own copy of the actions.
        let suffix = (!bulk && matches!(&previous.root, Some(Root::RootUuid(_)))).then(|| {
            let mut suffix = previous.mutations_since_root.clone();
            suffix.extend(tagged.iter().cloned());
            node::squash_buffer(suffix, self.contiguous_from)
        });
        self.buffer_index.take();
        self.buffer.extend(tagged);
        let before = self.buffer.len();
        self.buffer = node::squash_buffer(
            std::mem::take(&mut self.buffer),
            node::combine_from(&self.children, self.contiguous_from),
        );
        let squashed = before - self.buffer.len();
        if let Some(suffix) = suffix
            && !node::internal_overflows(&self.children, &self.buffer, &self.config)
            && node::internal_logical_bytes(&[], &suffix) <= policy.max_suffix_bytes as u64
        {
            return Ok((
                self.snapshot_descriptor(previous.root.clone(), suffix),
                CommitStats {
                    messages_in,
                    messages_squashed: squashed as u64,
                    height: self.height(),
                    root_buffer_len: self.buffer.len() as u64,
                    ..Default::default()
                },
            ));
        }
        self.force_flush = bulk;
        // A collapse starts only where an ingested interior keeps a single
        // interior child, which needs a root child at height two or more.
        // Below that the buffered rewrite never falls back, so the saved
        // copy of the whole buffer would go unused.
        let before_rewrite =
            (!bulk && self.children.iter().any(|child| child.height >= 2)).then(|| RootRouting {
                children: self.children.clone(),
                buffer: self.buffer.clone(),
            });
        let mut acc = self.rewrite_tree().await?;
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
            self.force_flush = true;
            acc = self.rewrite_tree().await?;
        }
        let (snapshot, bytes) = self.checkpoint_snapshot(policy).await?;
        Ok((
            snapshot,
            CommitStats {
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
            },
        ))
    }

    fn snapshot_descriptor(
        &self,
        root: Option<Root>,
        mutations_since_root: Vec<pb::FragmentTreeMutation>,
    ) -> pb::FragmentTree {
        pb::FragmentTree {
            root,
            mutations_since_root,
            next_action_sequence: self.next_action_sequence,
        }
    }

    async fn checkpoint_snapshot(&self, policy: SnapshotPolicy) -> Result<(pb::FragmentTree, u64)> {
        let root = self.compacted_root();
        if root.encoded_len() as u64 > self.config.hard_capacity_bytes {
            return Err(Error::invalid_input(format!(
                "fragment metadata root envelope requires {} encoded bytes, exceeding hard_capacity_bytes={}",
                root.encoded_len(),
                self.config.hard_capacity_bytes
            )));
        }
        let mut snapshot = self.snapshot_descriptor(Some(Root::InlineRoot(root)), Vec::new());
        if snapshot.encoded_len() <= policy.inline_root_bytes {
            return Ok((snapshot, 0));
        }
        let Some(Root::InlineRoot(root)) = &snapshot.root else {
            return Err(Error::internal(
                "prepared fragment metadata snapshot has no inline root",
            ));
        };
        let (uuid, bytes) = self.store.write_root_base(root).await?;
        snapshot.root = Some(Root::RootUuid(uuid));
        Ok((snapshot, bytes))
    }
}
