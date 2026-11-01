// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Attempt-local validation and mutation of an immutable fragment snapshot.

use super::*;
use crate::fragment_metadata::action;

/// How a commit writes its fragment changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitMode {
    /// Keep changes buffered until the tree's byte budgets require a drain.
    Incremental,
    /// Materialize changes in fragment order, retaining validation reads for reuse.
    Bulk,
}

/// Whether a build or preparation keeps a complete in-memory inventory for
/// its result.
///
/// ```
/// # use lance_table::fragment_metadata::{InventoryRetention, TreeCommit};
/// # fn lazy_result(attempt: TreeCommit) -> TreeCommit {
/// attempt.with_inventory_retention(InventoryRetention::Discard)
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryRetention {
    /// Retain the complete result, reading it only when necessary.
    Retain,
    /// Return a snapshot with no inventory, even when validation used a source
    /// that retained one.
    Discard,
}

/// A single preparation attempt bound to the snapshot it began from.
///
/// Resolved state and bulk-read retention belong to this attempt. Preparing or
/// dropping it cannot modify the source snapshot. The
/// [`Transaction`](crate::transaction::Transaction) commit path still owns
/// operation validation and publication through the Version Manifest.
///
/// ```
/// # use lance_core::Result;
/// # use lance_table::fragment_metadata::{
/// #     CommitMode, FragmentChanges, FragmentTree, PublicationPolicy,
/// # };
/// # async fn example(tree: &FragmentTree) -> Result<()> {
/// let attempt = tree.begin_commit(CommitMode::Incremental);
/// let prepared = attempt
///     .prepare(FragmentChanges::new(Vec::new()), PublicationPolicy::default())
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct TreeCommit {
    tree: FragmentTree,
    touched: TouchedFragments,
    mode: CommitMode,
    is_inventory_resolved: bool,
    retain_inventory: bool,
}

impl FragmentTree {
    /// Begin an isolated commit against this exact snapshot. Bulk mode retains
    /// complete validation leaf reads in scratch storage owned by the attempt.
    pub fn begin_commit(&self, mode: CommitMode) -> TreeCommit {
        TreeCommit {
            tree: self.clone(),
            touched: TouchedFragments::default(),
            mode,
            is_inventory_resolved: false,
            retain_inventory: self.inventory.is_some(),
        }
    }
}

impl TreeCommit {
    #[cfg(test)]
    pub(super) fn has_validation_reads(&self) -> bool {
        self.tree.store.has_validation_reads()
    }

    /// Select whether the prepared snapshot retains complete records.
    /// By default a source with a retained inventory produces a result with one.
    /// Explicit retention reuses complete validation, or reads missing records
    /// during preparation.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::{InventoryRetention, TreeCommit};
    /// # fn retained_result(attempt: TreeCommit) -> TreeCommit {
    /// attempt.with_inventory_retention(InventoryRetention::Retain)
    /// # }
    /// ```
    pub fn with_inventory_retention(mut self, retention: InventoryRetention) -> Self {
        self.retain_inventory = retention == InventoryRetention::Retain;
        self
    }

    /// Resolve the requested IDs and return all state resolved by this attempt
    /// so far, including earlier calls. Missing IDs are recorded as resolved
    /// absence. The borrowed view cannot modify authoritative before-state.
    /// Callers building a sparse manifest select their desired IDs from it.
    pub async fn resolve_touched(&mut self, ids: &[u64]) -> Result<&TouchedFragments> {
        let ids: BTreeSet<_> = ids.iter().copied().collect();
        self.resolve_ids(&ids).await?;
        Ok(&self.touched)
    }

    fn retain_bulk_reads(&mut self) -> Result<()> {
        if self.mode == CommitMode::Bulk
            && self.tree.inventory.is_none()
            && !self.tree.children.is_empty()
        {
            self.tree.store.retain_validation_reads()?;
        }
        Ok(())
    }

    async fn resolve_ids(&mut self, ids: &BTreeSet<u64>) -> Result<()> {
        if !self.is_inventory_resolved {
            let next_fragment_id = self.tree.next_fragment_id;
            // Every writer of next_fragment_id bounds it by 2^32, so every id
            // below it fits a u32.
            let unresolved = ids
                .difference(&self.touched.ids)
                .copied()
                .filter(|id| *id < next_fragment_id)
                .map(|id| {
                    u32::try_from(id).map_err(|_| {
                        Error::internal(format!(
                            "fragment id {id} below next_fragment_id {next_fragment_id} does not fit u32"
                        ))
                    })
                })
                .collect::<Result<RoaringBitmap>>()?;
            if !unresolved.is_empty() {
                self.retain_bulk_reads()?;
                let resolved = self.tree.resolve_fragments(&unresolved).await?;
                self.touched
                    .fragments
                    .extend(resolved.into_iter().map(|fragment| (fragment.id, fragment)));
            }
        }
        self.touched.ids.extend(ids.iter().copied());
        Ok(())
    }

    /// Resolve and borrow the complete inventory, retaining any IDs previously
    /// resolved as absent. Any other missing ID, including a reservation below
    /// `next_fragment_id`, joins `ids` without a read when it is next
    /// resolved. Repeated calls reuse the inventory without reading or copying
    /// its fragments.
    pub async fn resolve_all(&mut self) -> Result<&TouchedFragments> {
        if !self.is_inventory_resolved {
            self.retain_bulk_reads()?;
            let fragments = self.tree.resolve_all_fragments().await?;
            self.touched.fragments = fragments
                .into_iter()
                .map(|fragment| (fragment.id, fragment))
                .collect();
            self.touched
                .ids
                .extend(self.touched.fragments.keys().copied());
            self.is_inventory_resolved = true;
        }
        Ok(&self.touched)
    }

    /// Whether the root has room to buffer actions of the given encoded size.
    /// A caller building storage actions can use this to choose compact actions
    /// for a backfill.
    pub fn can_buffer(&self, bytes: u64) -> bool {
        self.tree.can_buffer(bytes)
    }

    /// Prepare immutable objects without publishing a version. Existing action
    /// targets not already resolved by this attempt are read before mutation.
    /// New IDs need no validation read. Failed or cancelled preparation leaves
    /// only unreachable objects, which retention cleanup can remove.
    pub async fn prepare(
        self,
        changes: FragmentChanges,
        policy: PublicationPolicy,
    ) -> Result<FragmentTree> {
        let (tree, _) = self.prepare_counted(changes, policy).await?;
        Ok(tree)
    }

    /// [`Self::prepare`], also returning the write work it took.
    pub(super) async fn prepare_counted(
        mut self,
        changes: FragmentChanges,
        policy: PublicationPolicy,
    ) -> Result<(FragmentTree, WriteAcc)> {
        let targets: BTreeSet<_> = changes
            .fragment_actions
            .iter()
            .map(|fragment_action| {
                action::target_frag_id(fragment_action).ok_or_else(|| {
                    Error::invalid_input(
                        "fragment metadata tree commit contains an empty fragment action",
                    )
                })
            })
            .collect::<Result<_>>()?;
        self.resolve_ids(&targets).await?;
        let deltas = commit::aggregate_deltas(
            &changes.fragment_actions,
            &self.touched.fragments,
            self.tree.store.base(),
        )?;
        let inventory = if self.retain_inventory {
            Some(self.inventory_after(&changes).await?)
        } else {
            None
        };
        self.tree.inventory = None;
        let work = self
            .tree
            .prepare_snapshot(changes, deltas, policy, self.mode)
            .await?;
        self.tree.store.clear_validation_reads();
        self.tree.inventory = inventory;
        Ok((self.tree, work))
    }

    /// The complete inventory this commit publishes, assembled from the
    /// before-state in `touched`, which this consumes.
    async fn inventory_after(&mut self, changes: &FragmentChanges) -> Result<Arc<Vec<Fragment>>> {
        if changes.fragment_actions.is_empty()
            && let Some(inventory) = &self.tree.inventory
        {
            return Ok(inventory.clone());
        }
        if self.tree.inventory.is_none() && !self.is_inventory_resolved {
            self.resolve_all().await?;
        }
        let mut records = std::mem::take(&mut self.touched.fragments);
        content::apply_actions(&mut records, &changes.fragment_actions)?;
        self.tree.store.share_data_file_lists(records.values_mut());
        let fragments = if self.is_inventory_resolved {
            records.into_values().collect()
        } else {
            let source = self.tree.inventory.as_ref().ok_or_else(|| {
                Error::internal(
                    "retained commit has neither complete resolution nor a source inventory",
                )
            })?;
            let mut fragments = Vec::with_capacity(source.len());
            for old in source.iter() {
                if let Some(fragment) = records.remove(&old.id) {
                    fragments.push(fragment);
                } else if !self.touched.ids.contains(&old.id) {
                    fragments.push(old.clone());
                }
            }
            fragments.extend(records.into_values());
            fragments.sort_unstable_by_key(|fragment| fragment.id);
            fragments
        };
        Ok(Arc::new(fragments))
    }
}
