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

/// Whether preparation keeps a complete in-memory inventory for its result.
///
/// ```
/// # use lance_table::fragment_metadata::{InventoryRetention, TreeCommit};
/// # fn lazy_result(attempt: TreeCommit) -> TreeCommit {
/// attempt.with_inventory_retention(InventoryRetention::Discard)
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryRetention {
    /// Retain the complete result for eager APIs, reading it only when necessary.
    Retain,
    /// Return an unloaded snapshot, even when validation used a resident source.
    Discard,
}

/// A single preparation attempt bound to the snapshot it began from.
///
/// Resolved state and bulk-read retention belong to this attempt. Preparing or
/// dropping it cannot modify the source snapshot. Native transactions still own
/// operation validation and publication through the Version Manifest.
///
/// ```
/// # use lance_core::Result;
/// # use lance_table::fragment_metadata::{
/// #     CommitMode, FragmentTree, SnapshotPolicy, ValidatedCommit,
/// # };
/// # async fn example(tree: &FragmentTree) -> Result<()> {
/// let attempt = tree.begin_commit(CommitMode::Incremental);
/// let (prepared, stats) = attempt
///     .prepare(ValidatedCommit::fragment_actions(Vec::new()), SnapshotPolicy::default())
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct TreeCommit {
    tree: FragmentTree,
    touched: TouchedFragments,
    mode: CommitMode,
    is_complete: bool,
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
            is_complete: false,
            retain_inventory: self.resident.is_some(),
        }
    }
}

impl TreeCommit {
    #[cfg(test)]
    pub(super) fn has_validation_reads(&self) -> bool {
        self.tree.store.has_validation_reads()
    }

    /// Select whether the prepared snapshot retains complete records.
    /// By default a resident source produces a resident result. Explicit retention
    /// reuses complete validation, or reads missing records during preparation.
    ///
    /// ```
    /// # use lance_table::fragment_metadata::{InventoryRetention, TreeCommit};
    /// # fn eager_result(attempt: TreeCommit) -> TreeCommit {
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

    async fn resolve_ids(&mut self, ids: &BTreeSet<u64>) -> Result<()> {
        if !self.is_complete {
            let next_fragment_id = self.tree.next_fragment_id;
            // Opening, bootstrapping and committing bound the allocation
            // frontier by 2^32, so every id below it fits a u32.
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
                if self.mode == CommitMode::Bulk
                    && self.tree.resident.is_none()
                    && !self.tree.children.is_empty()
                {
                    self.tree.store.retain_validation_reads()?;
                }
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
    /// resolved as absent. This also proves absence for any other missing ID,
    /// including reservations below the allocation frontier. Repeated calls
    /// reuse the inventory without reading or copying its fragments.
    pub async fn resolve_all(&mut self) -> Result<&TouchedFragments> {
        if !self.is_complete {
            if self.mode == CommitMode::Bulk
                && self.tree.resident.is_none()
                && !self.tree.children.is_empty()
            {
                self.tree.store.retain_validation_reads()?;
            }
            let fragments = self.tree.materialize().await?;
            self.touched.fragments = fragments
                .into_iter()
                .map(|fragment| (fragment.id, fragment))
                .collect();
            self.touched
                .ids
                .extend(self.touched.fragments.keys().copied());
            self.is_complete = true;
        }
        Ok(&self.touched)
    }

    /// Whether the root has room to buffer actions of the given encoded size.
    /// Native lowering can use this to choose compact actions for a backfill.
    pub fn can_buffer(&self, bytes: u64) -> bool {
        self.tree.can_buffer(bytes)
    }

    /// Prepare immutable objects without publishing a version. Existing action
    /// targets not already resolved by this attempt are read before mutation.
    /// New IDs need no validation read. Failed or cancelled preparation leaves
    /// only unreachable objects, which retention cleanup can remove.
    pub async fn prepare(
        mut self,
        commit: ValidatedCommit,
        policy: SnapshotPolicy,
    ) -> Result<(FragmentTree, CommitStats)> {
        let targets: BTreeSet<_> = commit
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
            &commit.fragment_actions,
            &self.touched,
            self.tree.next_fragment_id,
            self.tree.store.base(),
        )?;
        let resident = if self.retain_inventory {
            if commit.fragment_actions.is_empty() && self.tree.resident.is_some() {
                self.tree.resident.clone()
            } else {
                if self.tree.resident.is_none() && !self.is_complete {
                    self.resolve_all().await?;
                }
                let mut records = std::mem::take(&mut self.touched.fragments);
                let actions = commit
                    .fragment_actions
                    .iter()
                    .enumerate()
                    .map(|(sequence, action)| pb::FragmentTreeMutation {
                        action_sequence: sequence as u64 + 1,
                        action: Some(action.clone()),
                        ..Default::default()
                    })
                    .collect();
                content::apply_actions(&mut records, actions)?;
                let mut fragments = if self.is_complete {
                    records.into_values().collect()
                } else {
                    let source = self.tree.resident.as_ref().ok_or_else(|| {
                        Error::internal(
                            "retained commit has neither complete resolution nor a resident source",
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
                self.tree.store.share_data_file_lists(&mut fragments);
                Some(Arc::new(fragments))
            }
        } else {
            None
        };
        self.tree.resident = None;
        let stats = self
            .tree
            .prepare_snapshot_inner(commit, deltas, policy, self.mode)
            .await?;
        self.tree.store.clear_validation_reads();
        self.tree.resident = resident;
        Ok((self.tree, stats))
    }
}
