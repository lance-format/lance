// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Prepare immutable tree objects for publication by an authoritative manifest.

use super::*;
use crate::format::pb::fragment_tree::Root;
use crate::fragment_metadata::store::{NodeLocation, inherit_refs, root_path, stamp_unset_base_id};

/// Publication byte budgets, independent of the leaf limit and flush threshold.
#[derive(Debug, Clone, Copy)]
pub struct PublicationPolicy {
    /// Embed root and descriptor only when their complete encoded contribution
    /// to the Version Manifest fits this size. Zero disables embedding.
    pub inline_root_bytes: usize,
    /// Maximum encoded size of `mutations_since_root` beside an external root.
    /// An inline root carries none. A commit that would exceed this budget
    /// checkpoints, and `inline_root_bytes` decides where the new root goes. At
    /// zero, any commit with an action checkpoints.
    pub max_suffix_bytes: usize,
}

impl Default for PublicationPolicy {
    fn default() -> Self {
        Self {
            inline_root_bytes: 64 * 1024,
            max_suffix_bytes: 32 * 1024,
        }
    }
}

// A collapse retry restores routing. Rewrites leave next_fragment_id,
// next_action_sequence and totals unchanged.
struct RootRouting {
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
}

/// The root a descriptor names, checked, with the version totals it derives
/// from the root and `mutations_since_root`.
pub(super) struct CheckedRoot {
    pub(super) root: pb::FragmentTreeRoot,
    pub(super) totals: Totals,
}

/// Read the root `descriptor` names and run every descriptor check, before
/// opening squashes the root buffer and suffix together.
pub(super) async fn checked_root(
    store: &NodeStore,
    descriptor: &pb::FragmentTree,
    version: u64,
    next_fragment_id: u64,
) -> Result<CheckedRoot> {
    let root = match &descriptor.root {
        Some(Root::InlineRoot(root)) if descriptor.mutations_since_root.is_empty() => root.clone(),
        Some(Root::InlineRoot(_)) => {
            return Err(super::super::validation::corrupt(format!(
                "Inline root at version {version} has {} mutations after it",
                descriptor.mutations_since_root.len()
            )));
        }
        Some(Root::RootUuid(uuid)) => store.read_root_base(uuid).await?,
        None => {
            return Err(super::super::validation::corrupt(format!(
                "Tree descriptor at version {version} names no root"
            )));
        }
    };
    let base = super::super::validation::root(&root, next_fragment_id)?;
    super::super::validation::buffer(
        &descriptor.mutations_since_root,
        root.next_action_sequence,
        descriptor.next_action_sequence,
    )?;
    if next_fragment_id > node::ROOT_EXCLUSIVE_END {
        return Err(super::super::validation::corrupt(format!(
            "next_fragment_id {next_fragment_id} at version {version} exceeds 2^32"
        )));
    }
    super::super::validation::allocated_ids(
        descriptor
            .mutations_since_root
            .iter()
            .map(content::action_key),
        next_fragment_id,
        format_args!("mutations_since_root at version {version}"),
    )?;
    if descriptor.next_action_sequence < root.next_action_sequence {
        return Err(super::super::validation::corrupt(format!(
            "next_action_sequence {} at version {version} is below the root's {}",
            descriptor.next_action_sequence, root.next_action_sequence
        )));
    }
    let totals = base.apply(TotalsDelta::of(&descriptor.mutations_since_root)?)?;
    if totals.visible_rows > totals.rows {
        return Err(super::super::validation::corrupt(format!(
            "Derived visible rows {} exceed physical rows {} at version {version}",
            totals.visible_rows, totals.rows
        )));
    }
    Ok(CheckedRoot { root, totals })
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
                return Err(super::super::validation::corrupt(format!(
                    "Tree at version {} repeats resolved node {} (base_id={:?})",
                    self.version, child.path, child.base_id
                )));
            }
            if child.height > 0 {
                // Cleanup lists every retained version, and versions share
                // most interiors, so listing reads them through the cache
                // their walks use.
                let interior = self.store.read_interior_shared(&child, end).await?;
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
    /// [`InventoryRetention::Retain`] keeps `fragments`, sorted by id and without
    /// metadata size hints, as the snapshot's retained inventory.
    ///
    /// ```
    /// # use lance_table::{format::Fragment, fragment_metadata::{
    /// #     FragmentTree, FragmentTreeConfig, InventoryRetention, PublicationPolicy, TreeContext,
    /// # }};
    /// # async fn example(context: TreeContext, fragments: Vec<Fragment>) -> lance_core::Result<()> {
    /// let tree = FragmentTree::build(
    ///     context,
    ///     FragmentTreeConfig::default(),
    ///     fragments,
    ///     1,
    ///     PublicationPolicy::default(),
    ///     InventoryRetention::Retain,
    /// )
    /// .await?;
    /// assert!(tree.retained_inventory().is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build(
        context: TreeContext,
        config: FragmentTreeConfig,
        mut fragments: Vec<Fragment>,
        version: u64,
        policy: PublicationPolicy,
        retention: InventoryRetention,
    ) -> Result<Self> {
        fragments.sort_by_key(|fragment| fragment.id);
        if let Some(pair) = fragments.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(Error::invalid_input(format!(
                "fragment metadata tree build contains duplicate fragment ID {}",
                pair[0].id
            )));
        }
        for fragment in &mut fragments {
            content::clear_metadata_size_hints(fragment);
        }
        let store = context.into_store(&config, 1);
        // Leaf encoding matches interned lists by pointer first, so interning
        // before the build spares it comparing every list by contents.
        if retention == InventoryRetention::Retain {
            store.share_data_file_lists(&mut fragments);
        }
        let mut tree = Self::build_initial_tree(store, config, &fragments, version, policy).await?;
        if retention == InventoryRetention::Retain {
            tree.inventory = Some(Arc::new(fragments));
        }
        Ok(tree)
    }

    /// Open the tree `manifest` publishes, with that manifest's version and
    /// `next_fragment_id`. A manifest with no descriptor has no tree to open,
    /// which covers a flat manifest and a tree successor still waiting for one.
    ///
    /// ```
    /// # use lance_table::format::Manifest;
    /// # use lance_table::fragment_metadata::{FragmentTree, FragmentTreeConfig, TreeContext};
    /// # async fn example(context: TreeContext, manifest: &Manifest) -> lance_core::Result<()> {
    /// let tree = FragmentTree::open(context, manifest, FragmentTreeConfig::default()).await?;
    /// assert_eq!(tree.version(), manifest.version);
    /// assert!(tree.retained_inventory().is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open(
        context: TreeContext,
        manifest: &Manifest,
        config: FragmentTreeConfig,
    ) -> Result<Self> {
        let descriptor = manifest.fragment_tree.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "Manifest version {} publishes no fragment tree",
                manifest.version
            ))
        })?;
        Self::open_descriptor(
            context,
            descriptor,
            manifest.version,
            config,
            manifest.next_fragment_id(),
        )
        .await
    }

    /// Open fragment state using only a manifest descriptor and its named base.
    /// `version` and `next_fragment_id` come from that same authoritative
    /// manifest, because the descriptor cannot check either.
    pub(crate) async fn open_descriptor(
        context: TreeContext,
        descriptor: &pb::FragmentTree,
        version: u64,
        config: FragmentTreeConfig,
        next_fragment_id: u64,
    ) -> Result<Self> {
        config.validate()?;
        let mut store = context.into_store(&config, descriptor.next_action_sequence);
        let CheckedRoot { root, totals } =
            checked_root(&store, descriptor, version, next_fragment_id).await?;
        // Interiors below the root were written before it, so the format
        // bounds them by its frontier, as validate does.
        store.root_frontier = root.next_action_sequence;
        let mut buffer = root.buffer;
        buffer.extend(descriptor.mutations_since_root.iter().cloned());
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
            next_action_sequence: descriptor.next_action_sequence,
            contiguous_from: root.next_action_sequence,
            totals,
            next_fragment_id,
            descriptor: Arc::new(descriptor.clone()),
            inventory: None,
        })
    }

    pub(super) async fn prepare_snapshot(
        &mut self,
        changes: FragmentChanges,
        deltas: Vec<commit::ActionDeltas>,
        policy: PublicationPolicy,
        mode: CommitMode,
    ) -> Result<WriteAcc> {
        let previous = self.descriptor.clone();
        let tagged = self.stage_commit(changes, deltas)?;
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
        self.buffer = content::squash_buffer(
            std::mem::take(&mut self.buffer),
            node::combine_from(&self.children, self.contiguous_from),
        );
        if let Some(suffix) = suffix
            && !node::internal_overflows(&self.children, &self.buffer, &self.config)
            && node::internal_logical_bytes(&[], &suffix) <= policy.max_suffix_bytes as u64
        {
            self.descriptor = Arc::new(pb::FragmentTree {
                root: previous.root.clone(),
                mutations_since_root: suffix,
                next_action_sequence: self.next_action_sequence,
            });
            return Ok(WriteAcc::default());
        }
        let mut acc = self.rewrite_with_bulk_fallback(mode).await?;
        let (descriptor, bytes) =
            checkpoint(&self.store, &self.config, self.compacted_root(), policy).await?;
        self.descriptor = Arc::new(descriptor);
        // The new root bounds every interior below it, this commit's included.
        self.store.root_frontier = self.next_action_sequence;
        acc.io_bytes += bytes;
        Ok(acc)
    }

    /// Rewrite the tree in `mode`, and rebuild it in bulk if that collapses. The
    /// work returned includes what the abandoned rewrite wrote.
    async fn rewrite_with_bulk_fallback(&mut self, mode: CommitMode) -> Result<WriteAcc> {
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
        match self.rewrite_tree(mode).await? {
            Rewrite::Done(acc) => Ok(acc),
            Rewrite::Collapsed { abandoned_bytes } => {
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
                // Bulk routing is lifted over leaves in groups of at least two.
                let Rewrite::Done(mut acc) = self.rewrite_tree(CommitMode::Bulk).await? else {
                    return Err(Error::internal(format!(
                        "fragment metadata tree at version {} collapsed a bulk rewrite",
                        self.version
                    )));
                };
                acc.io_bytes += abandoned_bytes;
                Ok(acc)
            }
        }
    }

    /// The first snapshot of a branch whose manifest maps `source_base_id`
    /// to this snapshot's dataset, prepared for publication. The branch root
    /// keeps this snapshot's children and its buffer with
    /// `mutations_since_root` folded in, so every node stays in the source.
    /// The root follows the rules of any checkpoint. It is written through
    /// `branch` only when it does not fit inline. The call fails when the root
    /// outgrows `hard_capacity_bytes`.
    /// A retained inventory stays retained, stamped as a branch read stamps it.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::{FragmentTree, PublicationPolicy, TreeContext};
    /// # async fn example(source: FragmentTree, branch: TreeContext) -> lance_core::Result<()> {
    /// let branch = source
    ///     .into_branch(branch, 7, PublicationPolicy::default())
    ///     .await?;
    /// assert!(branch.descriptor().mutations_since_root.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn into_branch(
        mut self,
        branch: TreeContext,
        source_base_id: u32,
        policy: PublicationPolicy,
    ) -> Result<Self> {
        inherit_refs(&mut self.children, &mut self.buffer, source_base_id)?;
        self.buffer_index.take();
        if let Some(inventory) = self.inventory.take() {
            let mut fragments = Arc::unwrap_or_clone(inventory);
            for fragment in &mut fragments {
                stamp_unset_base_id(fragment, source_base_id);
            }
            self.inventory = Some(Arc::new(fragments));
        }
        self.rebind_store(branch.into_store(&self.config, self.next_action_sequence));
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
    policy: PublicationPolicy,
) -> Result<(pb::FragmentTree, u64)> {
    if root.encoded_len() as u64 > config.hard_capacity_bytes {
        return Err(Error::invalid_input(format!(
            "fragment metadata root envelope requires {} encoded bytes, exceeding hard_capacity_bytes={}",
            root.encoded_len(),
            config.hard_capacity_bytes
        )));
    }
    let next_action_sequence = root.next_action_sequence;
    let mut descriptor = pb::FragmentTree {
        root: Some(Root::InlineRoot(root)),
        mutations_since_root: Vec::new(),
        next_action_sequence,
    };
    if descriptor.encoded_len() <= policy.inline_root_bytes {
        return Ok((descriptor, 0));
    }
    let Some(Root::InlineRoot(root)) = &descriptor.root else {
        return Err(Error::internal(
            "prepared fragment metadata snapshot has no inline root",
        ));
    };
    let (uuid, bytes) = store.write_root_base(root).await?;
    descriptor.root = Some(Root::RootUuid(uuid));
    Ok((descriptor, bytes))
}
