// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod create_index;
mod rewrite;

use super::*;
use lance_table::transaction::action::{
    FieldRef as ActionFieldRef, Footprint, Overlap, Verdict, Window,
};
use lance_table::transaction::coverage_repair::{CommittedSince, NewCoverage, stale_coverage};

impl<'a> TransactionRebase<'a> {
    pub fn check_txn(&mut self, other_transaction: &Transaction, other_version: u64) -> Result<()> {
        // Either order: the claim was checked without the write's data.
        let ours = &self.transaction.operation;
        let theirs = &other_transaction.operation;
        if (may_alter_nullability(ours) && supplies_values(theirs))
            || (supplies_values(ours) && may_alter_nullability(theirs))
        {
            return Err(self.retryable_conflict_err(other_transaction, other_version));
        }
        // Merge and Project carry complete schemas from their read versions.
        // In either commit order, rebasing them with a metadata update can keep
        // one side's schema while discarding metadata from the other side.
        if (matches!(ours, Operation::Merge { .. } | Operation::Project { .. })
            && updates_schema_or_field_metadata(theirs))
            || (updates_schema_or_field_metadata(ours)
                && matches!(theirs, Operation::Merge { .. } | Operation::Project { .. }))
        {
            return Err(self.retryable_conflict_err(other_transaction, other_version));
        }
        // Restore replaces table metadata as well as schema/field metadata.
        // A stale provider update could otherwise select a definition that
        // restore discarded. Re-reading cannot repair the computed update.
        let updates_restored_metadata = |operation: &Operation| {
            updates_schema_or_field_metadata(operation)
                || matches!(
                    operation,
                    Operation::UpdateConfig {
                        table_metadata_updates: Some(update),
                        ..
                    } if update.replace || !update.update_entries.is_empty()
                )
        };
        if (matches!(ours, Operation::Restore { .. }) && updates_restored_metadata(theirs))
            || (updates_restored_metadata(ours) && matches!(theirs, Operation::Restore { .. }))
        {
            return Err(self.incompatible_conflict_err(other_transaction, other_version));
        }

        let op = &self.transaction.operation;
        match op {
            Operation::Delete { .. } => self.check_delete_txn(other_transaction, other_version),
            Operation::Update { .. } => self.check_update_txn(other_transaction, other_version),
            Operation::CreateIndex { .. } => {
                self.check_create_index_txn(other_transaction, other_version)
            }
            Operation::Rewrite { .. } => self.check_rewrite_txn(other_transaction, other_version),
            Operation::Overwrite { .. } => {
                self.check_overwrite_txn(other_transaction, other_version)
            }
            Operation::Append { .. } => self.check_append_txn(other_transaction, other_version),
            Operation::DataReplacement { .. } => {
                self.check_data_replacement_txn(other_transaction, other_version)
            }
            Operation::DataOverlay { .. } => {
                self.check_data_overlay_txn(other_transaction, other_version)
            }
            Operation::Merge { .. } => self.check_merge_txn(other_transaction, other_version),
            Operation::Restore { .. } => self.check_restore_txn(other_transaction, other_version),
            Operation::ReserveFragments { .. } => {
                self.check_reserve_fragments_txn(other_transaction, other_version)
            }
            Operation::Project { .. } => self.check_project_txn(other_transaction, other_version),
            Operation::UpdateConfig { .. } => {
                self.check_update_config_txn(other_transaction, other_version)
            }
            Operation::UpdateMemWalState { .. } => {
                self.check_update_mem_wal_state_txn(other_transaction, other_version)
            }
            // Clone has no checker of its own, so the cross-form rule has to be
            // stated here: it is compatible with every named operation, but a
            // concurrent action set still fails closed like everywhere else.
            Operation::Clone { .. } => {
                if matches!(
                    &other_transaction.operation,
                    Operation::CompositeOperation(_)
                ) {
                    self.check_action_txn(other_transaction, other_version)
                } else {
                    Ok(())
                }
            }
            Operation::UpdateBases { .. } => {
                self.check_add_bases_txn(other_transaction, other_version)
            }
            // An unrecognized operation cannot be checked: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
        }
    }

    /// Whether checking `other_transaction` takes the footprint of its commit
    /// from the manifests either side of it, which the caller then has to
    /// [record](Self::record_committed). Only a named operation's footprint
    /// is read off them, and only an action set's check compares one; an
    /// action set's own footprint comes from its actions.
    pub fn needs_committed_manifests(&self, other_transaction: &Transaction) -> bool {
        matches!(self.transaction.operation, Operation::CompositeOperation(_))
            && !matches!(
                other_transaction.operation,
                Operation::CompositeOperation(_)
            )
    }

    /// Whether this transaction adds index coverage over committed fragments,
    /// which `repair_index_coverage` may have to repair against every
    /// manifest committed since it read.
    pub fn adds_index_coverage(&self) -> bool {
        !new_coverage(&self.transaction, &HashSet::new()).is_empty()
    }

    /// Fold in what the footprint checks keep of `transaction`, committed as
    /// `version` on top of `before`, which is `None` when that manifest is
    /// gone. Only a named commit's footprint is kept, and only for an action
    /// set; without `before` it is left unrecorded and its check fails closed.
    pub fn fold_commit(
        &mut self,
        version: u64,
        transaction: &Transaction,
        before: Option<&Manifest>,
    ) {
        if let Some(before) = before
            && self.needs_committed_manifests(transaction)
        {
            self.record_committed(version, transaction, before);
        }
    }

    /// Record the footprint of `transaction`, which committed on top of
    /// `before` as `version`. Only the footprint is kept.
    pub fn record_committed(&mut self, version: u64, transaction: &Transaction, before: &Manifest) {
        self.footprint_basis
            .committed
            .insert(version, Footprint::of(transaction, before));
    }

    /// Record that the commits being checked took `since` to `latest`.
    pub fn record_window(&mut self, since: Arc<Manifest>, latest: Arc<Manifest>) {
        self.footprint_basis.since = Some(since);
        self.footprint_basis.latest = Some(latest);
    }

    /// Record `manifest`, committed since this transaction read, for the
    /// coverage repair to walk. See [`Self::adds_index_coverage`].
    pub fn record_history(&mut self, manifest: Arc<Manifest>) {
        self.footprint_basis
            .history
            .insert(manifest.version, manifest);
        #[cfg(test)]
        super::probe::held_history(self.footprint_basis.history.len());
    }

    /// Record what the catch-up records of one commit: `transaction`,
    /// committed as `version`, which took `before` to `after`.
    #[cfg(test)]
    pub(crate) fn record_commit(
        &mut self,
        version: u64,
        transaction: &Transaction,
        before: &Manifest,
        after: Arc<Manifest>,
    ) {
        if self.needs_committed_manifests(transaction) {
            self.record_committed(version, transaction, before);
        }
        self.record_history(after.clone());
        self.record_window(Arc::new(before.clone()), after);
    }

    /// The committed versions this rebase holds a footprint for.
    #[cfg(test)]
    pub(crate) fn recorded_versions(&self) -> Vec<u64> {
        let mut versions: Vec<u64> = self.footprint_basis.committed.keys().copied().collect();
        versions.sort_unstable();
        versions
    }

    /// Check this transaction against `committed`, every transaction
    /// committed since it read, in version order.
    ///
    /// An action set is checked once, against the change from the version the
    /// first of them landed on to the latest (see [`Footprint`]). Anything else is checked pair
    /// by pair, as [`Self::check_txn`] does.
    #[tracing::instrument(level = "debug", skip_all)]
    pub fn check_txns(&mut self, committed: &[(u64, &Transaction)]) -> Result<()> {
        let (Operation::CompositeOperation(_), Some((last, last_transaction))) =
            (&self.transaction.operation, committed.last())
        else {
            for (version, transaction) in committed {
                self.check_txn(transaction, *version)?;
            }
            return Ok(());
        };
        let Some(footprints) = committed
            .iter()
            .map(|(version, transaction)| self.committed_footprint(transaction, *version))
            .collect::<Option<Vec<_>>>()
        else {
            return Err(self.retryable_conflict_err(last_transaction, *last));
        };
        self.check_window(footprints, last_transaction, *last)
    }

    /// Whether an action set and `other_transaction`, committed as
    /// `other_version`, conflict, where at least one of them is an action set.
    ///
    /// Reached from both directions: for a pending action set checked pair by
    /// pair, and for a pending named operation over a committed action set.
    /// The window is the whole change the commits being checked made, with
    /// only `other_transaction`'s footprint: what the other
    /// commits in it changed is held against this one too, which fails safe.
    /// Fails closed when a named operation's footprint, or the latest
    /// manifest, was not recorded.
    fn check_action_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        let Some(footprint) = self.committed_footprint(other_transaction, other_version) else {
            return Err(self.retryable_conflict_err(other_transaction, other_version));
        };
        self.check_window(vec![footprint], other_transaction, other_version)
    }

    /// The footprint of `transaction`, committed as `version`: an action
    /// set's from its actions, a named operation's as recorded.
    fn committed_footprint(&self, transaction: &Transaction, version: u64) -> Option<Footprint> {
        match &transaction.operation {
            Operation::CompositeOperation(composite_operation) => {
                Some(Footprint::from(composite_operation))
            }
            _ => self.footprint_basis.committed.get(&version).cloned(),
        }
    }

    /// Check this transaction against the change the commits being checked
    /// made, by the transactions whose footprints are `committed`,
    /// the last of which is `last` at `last_version`; and rebase it past what
    /// it overlaps.
    fn check_window(
        &mut self,
        committed: Vec<Footprint>,
        last: &Transaction,
        last_version: u64,
    ) -> Result<()> {
        #[cfg(test)]
        super::probe::checked_window();
        let (Some(read), Some(since), Some(latest)) = (
            self.footprint_basis.read.clone(),
            self.footprint_basis.since.clone(),
            self.footprint_basis.latest.clone(),
        ) else {
            return Err(self.retryable_conflict_err(last, last_version));
        };
        if let Some(fragment) = self.unrecorded_move_under_new_coverage(&committed) {
            return Err(self.window_conflict_err(
                since.version,
                last_version,
                &format!(
                    "fragment {fragment}, which this one adds index coverage of, was compacted \
                     without the move being recorded"
                ),
            ));
        }
        let verdict =
            Footprint::of(&self.transaction, &read).check(&Window::new(&since, &latest, committed));
        let overlaps = match verdict {
            Verdict::Compatible => Vec::new(),
            Verdict::Rebase(overlaps) => overlaps,
            Verdict::Conflict(reason) => {
                return Err(self.window_conflict_err(since.version, last_version, &reason));
            }
        };
        // What the index coverage this transaction adds gives up is settled
        // once every committed transaction is checked, against the latest
        // manifest; see `repair_index_coverage`.
        self.footprint_basis.repairs_index_coverage = true;
        if overlaps.is_empty() {
            return Ok(());
        }
        for overlap in &overlaps {
            if !self.flag_rows_to_check(overlap) {
                return Err(self.retryable_conflict_err(last, last_version));
            }
        }
        match &mut self.transaction.operation {
            Operation::CompositeOperation(composite_operation) => {
                composite_operation.rebase(&overlaps)
            }
            // Flagged above on the marks their own checks set, so `finish`
            // settles them row by row as it does against a named operation.
            Operation::Delete { .. } | Operation::Update { .. }
                if overlaps
                    .iter()
                    .all(|overlap| matches!(overlap, Overlap::Deletions { .. })) =>
            {
                Ok(())
            }
            _ => Err(self.retryable_conflict_err(last, last_version)),
        }
    }

    fn window_conflict_err(&self, since: u64, latest: u64, reason: &str) -> Error {
        Error::retryable_commit_conflict_source(
            latest,
            format!(
                "This {} transaction was preempted by the transactions committed after \
                 version {since} up to version {latest}: {reason}. Please retry.",
                self.transaction.operation
            )
            .into(),
        )
    }

    /// Flag a fragment whose deletions `overlap` merges for `finish` to check
    /// row by row, as the legacy checks flag one through `initial_fragments`.
    /// A no-op for any other overlap.
    ///
    /// Returns false where the overlap cannot be resolved: this transaction
    /// does not say which rows it deleted, or the fragment as it read it is
    /// not known.
    fn flag_rows_to_check(&mut self, overlap: &Overlap) -> bool {
        let fragment = match overlap {
            Overlap::Deletions { fragment } if self.affected_rows.is_some() => *fragment,
            Overlap::Deletions { .. } => return false,
        };
        if let Some((_, flagged)) = self.initial_fragments.get_mut(&fragment) {
            *flagged = true;
            return true;
        }
        // The deletions this transaction's file was built on: those of the
        // version it read, unless a previous attempt already merged the file
        // with those at the start of this window. Never the handle this commit
        // goes through, which may be newer.
        let baseline = match self.merged_on_previous_attempt(fragment) {
            true => self.footprint_basis.since.as_ref(),
            false => self
                .footprint_basis
                .read
                .as_ref()
                .filter(|read| read.version == self.transaction.read_version),
        };
        let Some(initial) = baseline.and_then(|manifest| manifest.fragment_by_id(fragment)) else {
            return false;
        };
        self.initial_fragments
            .insert(fragment, (initial.clone(), true));
        true
    }

    /// A committed fragment this transaction adds index coverage of that one
    /// of `committed` compacted without the fragment reuse history recording
    /// where the rows went. Coverage of a recorded move is followed by the
    /// repair; of an unrecorded one it names rows that are gone, which legacy
    /// `CreateIndex` retries over an eager compaction too.
    fn unrecorded_move_under_new_coverage(&self, committed: &[Footprint]) -> Option<u64> {
        let compacted: HashSet<u64> = committed
            .iter()
            .flat_map(|footprint| footprint.compacted_fragments().iter().copied())
            .collect();
        if compacted.is_empty() {
            return None;
        }
        new_coverage(&self.transaction, &self.footprint_basis.ngram_segments)
            .iter()
            .flat_map(|coverage| coverage.fragments.iter().copied())
            .find(|fragment| {
                compacted.contains(fragment)
                    && self
                        .moved_to(*fragment)
                        .is_some_and(|moved| moved.is_empty())
            })
    }

    /// Whether this action set's deletion file for `fragment` is one a
    /// previous attempt's rebase merged with the deletions at the start of
    /// this window, which `settle_rows` writes stamped with that version.
    fn merged_on_previous_attempt(&self, fragment: u64) -> bool {
        let (Operation::CompositeOperation(composite_operation), Some(since)) =
            (&self.transaction.operation, &self.footprint_basis.since)
        else {
            return false;
        };
        since.version != self.transaction.read_version
            && composite_operation.iter_actions().any(|action| {
                matches!(action, Action::SetDeletionFile(set)
                    if set.fragment() == fragment
                        && set.deletion_file().read_version == since.version)
            })
    }

    /// The fragments the latest fragment reuse index records `fragment`'s
    /// rows as moved into, following one compaction after another: empty
    /// when it records no move of `fragment`, and `None` when a tagged
    /// history has it on its lineage but cannot be walked.
    ///
    /// Only a move out of `fragment` counts, never one into it, so a
    /// compaction of a fragment some recorded move produced, which recorded
    /// nothing itself, is not taken for a recorded one.
    fn moved_to(&self, fragment: u64) -> Option<Vec<u64>> {
        let Ok(fragment) = u32::try_from(fragment) else {
            return Some(Vec::new());
        };
        let moved = if let Some(lineage) = &self.current_lineage {
            if !lineage.contains(fragment) {
                return Some(Vec::new());
            }
            lineage.destinations_of(fragment)?
        } else if let Some(reuse) = &self.untagged_reuse {
            reuse.moved_to(fragment)
        } else {
            Vec::new()
        };
        Some(moved.into_iter().map(u64::from).collect())
    }

    fn check_delete_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Delete { .. } = &self.transaction.operation {
            match &other_transaction.operation {
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                Operation::CreateIndex { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Clone { .. }
                | Operation::Project { .. }
                | Operation::Append { .. }
                | Operation::UpdateConfig { .. }
                // A concurrent overlay is inert against the rows we delete
                // (deletions take precedence over overlays) and otherwise
                // preserves physical offsets, so it never conflicts.
                | Operation::DataOverlay { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Rewrite { groups, .. } => {
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataReplacement { replacements, .. } => {
                    if replacements
                        .iter()
                        .map(|r| r.0)
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::Update {
                    updated_fragments,
                    removed_fragment_ids,
                    ..
                }
                | Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids: removed_fragment_ids,
                    ..
                } => {
                    if !updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(removed_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        return Ok(());
                    }

                    if self.affected_rows.is_none() {
                        // We don't have any affected rows, so we can't
                        // do the rebase anyways.
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                    for updated in updated_fragments {
                        if let Some((fragment, needs_rewrite)) =
                            self.initial_fragments.get_mut(&updated.id)
                        {
                            // If data files, not just deletion files, are modified,
                            // then we can't rebase.
                            if fragment.files != updated.files {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }

                            // Mark any modified fragments as needing a rewrite.
                            *needs_rewrite |= updated.deletion_file != fragment.deletion_file;
                        }
                    }

                    for removed_fragment_id in removed_fragment_ids {
                        if self.initial_fragments.contains_key(removed_fragment_id) {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    Ok(())
                }
                Operation::Merge { .. } => {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                }
                Operation::Overwrite { .. }
                | Operation::Restore { .. }
                | Operation::UpdateMemWalState { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => Err(self.incompatible_conflict_err(other_transaction, other_version)),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_update_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Update {
            inserted_rows_filter: self_inserted_rows_filter,
            compacted_sstables: self_compacted_sstables,
            new_fragments: self_new_fragments,
            update_mode: self_update_mode,
            updated_fragments: self_updated_fragments,
            fields_modified: self_fields_modified,
            ..
        } = &self.transaction.operation
        {
            if let Operation::Update {
                inserted_rows_filter: other_inserted_rows_filter,
                ..
            } = &other_transaction.operation
            {
                // The presence of inserted_rows_filter means this is a primary key operation
                // and strict conflict detection should be applied.
                match (self_inserted_rows_filter, other_inserted_rows_filter) {
                    (Some(self_keys), Some(other_keys)) => {
                        if self_keys.field_ids != other_keys.field_ids {
                            // Different key columns - can't verify conflicts
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                        // Check for intersection. If the bloom filter configs don't match
                        // (e.g., different number_of_items or probability), intersects() returns
                        // an error and we treat it as a conflict to be safe.
                        let Ok((has_intersection, _maybe_false_positive)) =
                            self_keys.intersects(other_keys)
                        else {
                            // Bloom filter configs don't match - treat as conflict
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        };
                        if has_intersection {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    (Some(_), None) => {
                        // Current transaction has primary key conflict detection but
                        // the already committed transaction doesn't have a filter.
                        // We can't determine what rows were inserted by the other
                        // transaction, so we must fail to be safe.
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                    _ => {}
                }
            }

            match &other_transaction.operation {
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                Operation::CreateIndex { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Project { schema, .. } => {
                    // A project can drop fields, and this update writes
                    // replacement files for the fields it names. Committing
                    // over a projection that removed one leaves the fragment
                    // carrying a data file for a field the schema no longer
                    // has. Same rule `check_data_replacement_txn` applies to
                    // the other operation that rewrites a field in place.
                    for field in self_fields_modified {
                        if schema.field_by_id(*field as i32).is_none() {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    Ok(())
                }
                Operation::DataOverlay { groups } => {
                    // Our update recomputed rows from the pre-overlay base, so if
                    // it commits over an overlay it would silently undo the
                    // overlay's values for any cell it recomputed.
                    //
                    // An in-place column rewrite (RewriteColumns) writes a
                    // replacement file covering *every* row of each fragment it
                    // touches, filled from the snapshot it read. `build_manifest`
                    // then tombstones every overlay for the fields it rewrote, so
                    // an overlay value on a row this update never matched is
                    // dropped and the stale copied value becomes visible. Retry
                    // whenever the overlay touches a fragment we rewrote and a
                    // field we rewrote; the retry re-reads the overlaid values.
                    if matches!(self_update_mode, Some(UpdateMode::RewriteColumns)) {
                        let rewritten: HashSet<u64> = self_updated_fragments
                            .iter()
                            .map(|fragment| fragment.id)
                            .collect();
                        for group in groups {
                            if !rewritten.contains(&group.fragment_id) {
                                continue;
                            }
                            let overlaps_rewritten_field = group.overlays.iter().any(|overlay| {
                                overlay.data_file.fields.iter().any(|&field| {
                                    field >= 0 && self_fields_modified.contains(&(field as u32))
                                })
                            });
                            if overlaps_rewritten_field {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }
                        }
                        return Ok(());
                    }

                    // A row-moving update (RewriteRows) relocates the rows it
                    // touches out to new fragments; only the rows it actually
                    // moved lose their overlay, so we conflict only when the
                    // moved rows intersect the overlay's coverage.
                    let moves_rows = !self_new_fragments.is_empty()
                        && matches!(self_update_mode, Some(UpdateMode::RewriteRows) | None);
                    if !moves_rows {
                        return Ok(());
                    }
                    // `affected_rows` holds the physical offsets (per fragment)
                    // this update moved. The overlay's coverage is in the same
                    // physical-offset space, so we can intersect the two in
                    // memory. Without affected rows we cannot be precise, so we
                    // fall back to a fragment-granular conflict.
                    for group in groups {
                        if !self.modified_fragment_ids.contains(&group.fragment_id) {
                            continue;
                        }
                        let Some(affected_rows) = self.affected_rows else {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        };
                        let Some(moved) =
                            affected_rows.get_fragment_bitmap(group.fragment_id as u32)
                        else {
                            continue;
                        };
                        let coverage = overlay_group_coverage(group);
                        if !(moved & &coverage).is_empty() {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    Ok(())
                }
                Operation::Append { .. } => {
                    // If current transaction has primary key conflict detection,
                    // we can't safely commit against an Append because we don't
                    // know if the appended rows conflict with inserted rows.
                    if self_inserted_rows_filter.is_some() {
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                    Ok(())
                }
                Operation::Rewrite { groups, .. } => {
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataReplacement { replacements, .. } => {
                    if replacements
                        .iter()
                        .map(|r| r.0)
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::Update {
                    updated_fragments,
                    removed_fragment_ids,
                    ..
                }
                | Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids: removed_fragment_ids,
                    ..
                } => {
                    if !updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(removed_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        return Ok(());
                    }

                    if self.affected_rows.is_none() {
                        // We don't have any affected rows, so we can't
                        // do the rebase anyways.
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                    for updated in updated_fragments {
                        if let Some((fragment, needs_rewrite)) =
                            self.initial_fragments.get_mut(&updated.id)
                        {
                            // If data files, not just deletion files, are modified,
                            // then we can't rebase.
                            if fragment.files != updated.files {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }

                            // Mark any modified fragments as needing a rewrite.
                            *needs_rewrite |= updated.deletion_file != fragment.deletion_file;
                        }
                    }

                    for removed_fragment_id in removed_fragment_ids {
                        if self.initial_fragments.contains_key(removed_fragment_id) {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    Ok(())
                }
                Operation::Merge { .. } => {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                }
                Operation::Overwrite { .. } | Operation::Restore { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
                Operation::UpdateMemWalState {
                    compacted_sstables: other_compacted_sstables,
                    ..
                } => self.check_compacted_sstables_conflict(
                    other_compacted_sstables,
                    self_compacted_sstables,
                    other_transaction,
                    other_version,
                ),
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_overwrite_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            Operation::Overwrite { .. } => {
                if self
                    .transaction
                    .operation
                    .upsert_key_conflict(&other_transaction.operation)
                {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                } else {
                    // Concurrent overwrites are retryable so user can decide
                    // if their overwrite should still proceed
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                }
            }
            Operation::UpdateConfig { .. } => {
                if self
                    .transaction
                    .operation
                    .upsert_key_conflict(&other_transaction.operation)
                {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            Operation::Append { .. }
            | Operation::Clone { .. }
            | Operation::Delete { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::UpdateBases { .. } => Ok(()),
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_append_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        let Operation::Append { fragments } = &self.transaction.operation else {
            return Err(wrong_operation_err(&self.transaction.operation));
        };
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            // Append is not compatible with any operation that completely
            // overwrites the schema.
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            Operation::Merge { schema, .. } => {
                // Casts and dropped-and-readded columns replace field ids. Files
                // staged with the old ids would silently read back as null.
                // Negative ids are tombstones or internal fields outside the schema.
                if fragments
                    .iter()
                    .flat_map(|fragment| &fragment.files)
                    .flat_map(|file| file.fields.iter())
                    .any(|field| *field >= 0 && schema.field_by_id(*field).is_none())
                {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            Operation::Append { .. }
            | Operation::Rewrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::Delete { .. }
            | Operation::Update { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::UpdateBases { .. }
            | Operation::UpdateConfig { .. }
            | Operation::Clone { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. } => Ok(()),
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_data_replacement_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::DataReplacement { replacements, .. } = &self.transaction.operation {
            match &other_transaction.operation {
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::ReserveFragments { .. }
                // Both a column replacement and an overlay preserve physical row
                // addresses; the overlay is newer and wins its covered cells.
                | Operation::DataOverlay { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Project { schema, .. } => {
                    // A project operation can drop fields.  If the project
                    // dropped a field this operation was replacing then
                    // we have a conflict.
                    for replacement in replacements {
                        for field in replacement.1.fields.iter() {
                            if *field >= 0 && schema.field_by_id(*field).is_none() {
                                return Err(self.data_replacement_field_removed_err(
                                    *field,
                                    replacement.0,
                                    other_transaction,
                                    other_version,
                                ));
                            }
                        }
                    }
                    Ok(())
                }
                Operation::Merge { .. } => {
                    // Merge rewrites the whole fragment list; always conflict
                    // (symmetric with check_merge_txn).
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                }
                Operation::Delete {
                    deleted_fragment_ids,
                    ..
                } => {
                    // A delete only tombstones rows (deletion vector); our positional
                    // file stays aligned and the rebase preserves the deletion vector.
                    // Conflict only if our target fragment was removed outright.
                    for replacement in replacements {
                        if deleted_fragment_ids.contains(&replacement.0) {
                            return Err(self.data_replacement_target_removed_err(
                                replacement.0,
                                other_transaction,
                                other_version,
                            ));
                        }
                    }
                    Ok(())
                }
                Operation::Update {
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                    fields_modified,
                    update_mode,
                    ..
                } => {
                    for replacement in replacements {
                        if removed_fragment_ids.contains(&replacement.0) {
                            return Err(self.data_replacement_target_removed_err(
                                replacement.0,
                                other_transaction,
                                other_version,
                            ));
                        }
                        if !updated_fragments.iter().any(|f| f.id == replacement.0) {
                            continue;
                        }
                        // A row-rewriting update moves the matched rows out to
                        // new_fragments our positional file does not cover; a horizontal
                        // update may rewrite one of our fields in place. Either makes the
                        // file stale. (RewriteColumns new_fragments are unrelated inserts,
                        // not moved rows, so they stay aligned.)
                        let moved_rows = !new_fragments.is_empty()
                            && matches!(update_mode, Some(UpdateMode::RewriteRows) | None);
                        let field_rewritten = replacement
                            .1
                            .fields
                            .iter()
                            .any(|f| *f >= 0 && fields_modified.contains(&(*f as u32)));
                        if moved_rows || field_rewritten {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                    }
                    Ok(())
                }
                Operation::CreateIndex { new_indices, .. } => {
                    // A data replacement only conflicts if it is updating a field the
                    // index depends on -- whether keyed on or merely carried (see
                    // `IndexMetadata::dependency_fields`).
                    //
                    // TODO: We could potentially just drop the fragments being replaced from
                    // the index's fragment bitmap, which would lead to fewer conflicts.  However
                    // this would introduce fragment bitmaps with holes which may not be well tested
                    // yet.  For now, we don't allow this case.
                    //
                    // Values that only moved still match what the index was
                    // built from, so those never conflict.
                    if self.only_moves_values() {
                        return Ok(());
                    }
                    let newly_depended_fields = new_indices
                        .iter()
                        .flat_map(|idx| idx.dependency_fields().into_owned())
                        .collect::<HashSet<_>>();
                    for replacement in replacements {
                        for field in replacement.1.fields.iter() {
                            if newly_depended_fields.contains(field) {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }
                        }
                    }
                    Ok(())
                }
                Operation::Rewrite { groups, .. } => {
                    // These conflict if the rewrite touches any of the fragments being replaced.
                    for replacement in replacements {
                        for group in groups {
                            for old_fragment in &group.old_fragments {
                                if replacement.0 == old_fragment.id {
                                    return Err(self
                                        .retryable_conflict_err(other_transaction, other_version));
                                }
                            }
                        }
                    }

                    Ok(())
                }
                Operation::DataReplacement {
                    replacements: other_replacements,
                    ..
                } => {
                    // These conflict if there is overlap in fragment id && fields.
                    for replacement in replacements {
                        for other_replacement in other_replacements {
                            if replacement.0 != other_replacement.0 {
                                continue;
                            }

                            for field in replacement.1.fields.iter() {
                                if other_replacement.1.fields.contains(field) {
                                    return Err(self
                                        .retryable_conflict_err(other_transaction, other_version));
                                }
                            }
                        }
                    }
                    Ok(())
                }
                Operation::Overwrite { .. }
                | Operation::Restore { .. }
                | Operation::UpdateMemWalState { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => Err(self.incompatible_conflict_err(other_transaction, other_version)),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    /// Conflict checks for our DataOverlay transaction against a concurrent one.
    ///
    /// Overlays are intentionally permissive (see the Data Overlay Files spec):
    /// they stack with other overlays and tolerate appends, index builds, data
    /// replacement, deletes, and in-place column rewrites (Update with
    /// `RewriteColumns`), because overlay coverage is addressed by physical offset
    /// and the version gate keeps indexes correct. A concurrent operation
    /// conflicts when it takes precedence over the overlay for cells the overlay
    /// covers, dropping the overlay's values: retryably when it rewrites the
    /// physical layout of one of our fragments (Rewrite, Merge), drops a field
    /// we overlay (Project), or re-creates the covered rows from the pre-overlay
    /// base (a row-moving Update — checked row-by-row in `finish_data_overlay`),
    /// or removes an overlaid fragment outright (a Delete / Update that drops the
    /// fragment); and incompatibly for whole-dataset replacements (Overwrite /
    /// Restore) and MemWAL state updates (UpdateMemWalState), which do not rebase
    /// against data operations.
    fn check_data_overlay_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            Operation::Append { .. }
            | Operation::CreateIndex { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateBases { .. }
            | Operation::Clone { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. } => Ok(()),
            // A concurrent Project that dropped a field we overlay discards our
            // values for it. Project only prunes the overlays already committed,
            // so ours would otherwise land covering a field that no longer exists.
            Operation::Project { schema, .. } => {
                let Operation::DataOverlay { groups } = &self.transaction.operation else {
                    return Err(wrong_operation_err(&self.transaction.operation));
                };
                let remaining_field_ids = schema
                    .fields_pre_order()
                    .map(|f| f.id)
                    .collect::<HashSet<_>>();
                let drops_our_field = groups
                    .iter()
                    .flat_map(|g| g.overlays.iter())
                    .flat_map(|o| o.data_file.fields.iter())
                    .any(|id| !remaining_field_ids.contains(id));
                if drops_our_field {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            // A concurrent Delete only tombstones rows via a deletion vector,
            // which preserves physical offsets; the overlay value for a deleted
            // offset is simply inert. Conflict only if the whole overlaid
            // fragment was removed, orphaning the overlay.
            Operation::Delete {
                deleted_fragment_ids,
                ..
            } => {
                if deleted_fragment_ids
                    .iter()
                    .any(|id| self.modified_fragment_ids.contains(id))
                {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            // A concurrent Update that removed an overlaid fragment orphans the
            // overlay outright — conflict. A row-moving update (RewriteRows)
            // deletes the rows it touches and re-creates them in new fragments;
            // the update took precedence and the re-created rows were computed
            // from the pre-overlay base, so the overlay's values for those cells
            // are lost. That is a per-row problem, not an offset one: only the
            // moved rows are affected. Comparing the moved rows against the
            // overlay's coverage needs the update's deletion vectors, so we mark
            // the fragment here and verify row-by-row in `finish_data_overlay`.
            // An in-place column rewrite (RewriteColumns) preserves rows and just
            // tombstones the overlaid fields at build time, so it never conflicts.
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                update_mode,
                ..
            } => {
                let removed_ours = removed_fragment_ids
                    .iter()
                    .any(|id| self.modified_fragment_ids.contains(id));
                if removed_ours {
                    return Err(self.retryable_conflict_err(other_transaction, other_version));
                }
                let moves_rows = !new_fragments.is_empty()
                    && matches!(update_mode, Some(UpdateMode::RewriteRows) | None);
                if moves_rows {
                    for updated in updated_fragments {
                        if let Some((_, needs_row_check)) =
                            self.initial_fragments.get_mut(&updated.id)
                        {
                            *needs_row_check = true;
                        }
                    }
                }
                Ok(())
            }
            Operation::Rewrite { groups, .. } => {
                // A rewrite (compaction / fold) of a fragment we are overlaying
                // changes its physical row addresses, so our offsets would be
                // invalid. Conflict only if it touches one of our fragments.
                let touches_our_fragment = groups
                    .iter()
                    .flat_map(|g| g.old_fragments.iter())
                    .any(|f| self.modified_fragment_ids.contains(&f.id));
                if touches_our_fragment {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            Operation::Merge { .. } => {
                // Merge rewrites the whole fragment list; always conflict.
                Err(self.retryable_conflict_err(other_transaction, other_version))
            }
            // Overwrite/Restore replace the dataset; UpdateMemWalState does not
            // rebase against data operations (mirroring check_update_mem_wal_state_txn,
            // which likewise treats a concurrent DataOverlay as incompatible).
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_merge_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            // See the MemWAL exception in check_create_index_txn.
            Operation::CreateIndex { new_indices, .. } => {
                if new_indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                } else {
                    Ok(())
                }
            }
            Operation::ReserveFragments { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateBases { .. } => Ok(()),

            Operation::Update { .. }
            | Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::Rewrite { .. }
            | Operation::Merge { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. } => {
                Err(self.retryable_conflict_err(other_transaction, other_version))
            }
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::Project { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_restore_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::Overwrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateBases { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. } => Ok(()),
            Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_reserve_fragments_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            Operation::Overwrite { .. } | Operation::Restore { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            Operation::Append { .. }
            | Operation::Delete { .. }
            | Operation::CreateIndex { .. }
            | Operation::Rewrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
            | Operation::Merge { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Update { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::UpdateBases { .. } => Ok(()),
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_project_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        match &other_transaction.operation {
            // A mixed named/action pair fails closed; see check_action_txn.
            Operation::CompositeOperation(_) => {
                self.check_action_txn(other_transaction, other_version)
            }
            // Project is compatible with anything that doesn't change the schema
            Operation::Append { .. }
            | Operation::Update { .. }
            | Operation::Delete { .. }
            | Operation::UpdateConfig { .. }
            | Operation::CreateIndex { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
            | Operation::Rewrite { .. }
            | Operation::Clone { .. }
            | Operation::ReserveFragments { .. }
            | Operation::UpdateBases { .. } => Ok(()),
            Operation::Merge { .. } | Operation::Project { .. } => {
                // Need to recompute the schema
                Err(self.retryable_conflict_err(other_transaction, other_version))
            }
            Operation::Overwrite { .. }
            | Operation::Restore { .. }
            | Operation::UpdateMemWalState { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
            // An unrecognized operation may touch anything: assume it conflicts.
            Operation::Unknown { .. } => {
                Err(self.incompatible_conflict_err(other_transaction, other_version))
            }
        }
    }

    fn check_update_config_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::UpdateConfig {
            schema_metadata_updates,
            field_metadata_updates,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                Operation::Overwrite { .. } => {
                    // Updates to schema metadata or field metadata conflict with any kind
                    // of overwrite.
                    if schema_metadata_updates.is_some()
                        || !field_metadata_updates.is_empty()
                        || self
                            .transaction
                            .operation
                            .upsert_key_conflict(&other_transaction.operation)
                    {
                        Err(self.incompatible_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::UpdateConfig { .. } => {
                    if self
                        .transaction
                        .operation
                        .upsert_key_conflict(&other_transaction.operation)
                        || self
                            .transaction
                            .operation
                            .modifies_same_metadata(&other_transaction.operation)
                    {
                        Err(self.incompatible_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::Append { .. }
                | Operation::Clone { .. }
                | Operation::Delete { .. }
                | Operation::CreateIndex { .. }
                | Operation::Rewrite { .. }
                | Operation::DataReplacement { .. }
                | Operation::DataOverlay { .. }
                | Operation::Merge { .. }
                | Operation::Restore { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Update { .. }
                | Operation::Project { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_update_mem_wal_state_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        // Activation rebases like any other MemWAL state update; its preconditions
        // are re-checked against the rebased index list when the commit applies.
        if let Operation::UpdateMemWalState {
            compacted_sstables: self_compacted_sstables,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                Operation::UpdateMemWalState {
                    compacted_sstables: other_compacted_sstables,
                    ..
                } => {
                    // Two UpdateMemWalState transactions conflict if they're updating
                    // the same shard's compacted SSTable
                    self.check_compacted_sstables_conflict(
                        other_compacted_sstables,
                        self_compacted_sstables,
                        other_transaction,
                        other_version,
                    )
                }
                Operation::Update {
                    compacted_sstables: other_compacted_sstables,
                    ..
                } => {
                    // Update transactions with compacted_sstables can conflict
                    self.check_compacted_sstables_conflict(
                        other_compacted_sstables,
                        self_compacted_sstables,
                        other_transaction,
                        other_version,
                    )
                }
                Operation::CreateIndex { new_indices, .. } => {
                    // Check if CreateIndex has a MemWalIndex with compacted_sstables
                    if let Some(mem_wal_idx) = new_indices
                        .iter()
                        .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
                    {
                        let details = load_mem_wal_index_details(mem_wal_idx.clone())?;
                        self.check_compacted_sstables_conflict(
                            &details.compacted_sstables,
                            self_compacted_sstables,
                            other_transaction,
                            other_version,
                        )
                    } else {
                        Ok(())
                    }
                }
                Operation::UpdateConfig { .. }
                | Operation::Rewrite { .. }
                | Operation::ReserveFragments { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::Append { .. }
                | Operation::Overwrite { .. }
                | Operation::Delete { .. }
                | Operation::DataReplacement { .. }
                | Operation::DataOverlay { .. }
                | Operation::Merge { .. }
                | Operation::Restore { .. }
                | Operation::Clone { .. }
                | Operation::Project { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_add_bases_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::UpdateBases { new_bases } = &self.transaction.operation {
            match &other_transaction.operation {
                Operation::UpdateBases {
                    new_bases: committed_bases,
                } => {
                    // Check if any of the bases being added conflict with committed bases
                    for new_base in new_bases {
                        for committed_base in committed_bases {
                            // Check for ID conflicts (if both have non-zero IDs)
                            if new_base.id != 0
                                && committed_base.id != 0
                                && new_base.id == committed_base.id
                            {
                                return Err(self
                                    .incompatible_conflict_err(other_transaction, other_version));
                            }
                            // Check for name conflicts
                            if new_base.name == committed_base.name && new_base.name.is_some() {
                                return Err(self
                                    .incompatible_conflict_err(other_transaction, other_version));
                            }
                            // Check for path conflicts
                            if new_base.path == committed_base.path {
                                return Err(self
                                    .incompatible_conflict_err(other_transaction, other_version));
                            }
                        }
                    }
                    Ok(())
                }
                // A mixed named/action pair fails closed; see check_action_txn.
                Operation::CompositeOperation(_) => {
                    self.check_action_txn(other_transaction, other_version)
                }
                // UpdateBases doesn't conflict with data operations
                Operation::Append { .. }
                | Operation::Delete { .. }
                | Operation::Overwrite { .. }
                | Operation::CreateIndex { .. }
                | Operation::Rewrite { .. }
                | Operation::DataReplacement { .. }
                | Operation::DataOverlay { .. }
                | Operation::Merge { .. }
                | Operation::Restore { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Update { .. }
                | Operation::Project { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::Unknown { .. } => Ok(()),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    fn check_compacted_sstables_conflict(
        &self,
        committed: &[CompactedSsTable],
        to_commit: &[CompactedSsTable],
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        // Check if any shard has conflicting updates
        for committed_mg in committed {
            for to_commit_mg in to_commit {
                if committed_mg.shard_id == to_commit_mg.shard_id {
                    // Same shard being updated
                    // If committed >= to_commit, the SSTable is already compacted
                    // or superseded, so abort without retry.
                    // If committed < to_commit, can retry with new state
                    if committed_mg.generation >= to_commit_mg.generation {
                        return Err(
                            self.incompatible_conflict_err(other_transaction, other_version)
                        );
                    } else {
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                }
            }
        }
        Ok(())
    }

    /// Withdraw, from the index coverage this transaction adds or widens,
    /// whatever the commits since it read invalidated: a column rewritten
    /// under it, rows moved somewhere rewritten since, an overlay a
    /// compaction folded in. A no-op unless an action set was checked against
    /// it (`repairs_index_coverage`). See
    /// [`lance_table::transaction::coverage_repair`].
    pub(super) fn repair_index_coverage(&mut self) {
        if !self.footprint_basis.repairs_index_coverage {
            return;
        }
        let Some(read) = self.footprint_basis.read.clone() else {
            return;
        };
        let mut committed: Vec<Arc<Manifest>> =
            self.footprint_basis.history.values().cloned().collect();
        if let Some(latest) = &self.footprint_basis.latest
            && !self.footprint_basis.history.contains_key(&latest.version)
        {
            committed.push(latest.clone());
        }
        committed.sort_by_key(|manifest| manifest.version);
        let moved_to = |fragment| self.moved_to(fragment);
        let since = CommittedSince {
            read: &read,
            committed: committed.iter().map(|manifest| manifest.as_ref()).collect(),
            moved_to: &moved_to,
        };
        let withdrawals: Vec<(uuid::Uuid, Vec<u64>)> =
            new_coverage(&self.transaction, &self.footprint_basis.ngram_segments)
                .into_iter()
                .map(|coverage| (coverage.segment, stale_coverage(&since, &coverage)))
                .filter(|(_, stale)| !stale.is_empty())
                .collect();
        for (segment, stale) in withdrawals {
            match &mut self.transaction.operation {
                Operation::CompositeOperation(composite_operation) => {
                    for fragment in stale {
                        composite_operation.withdraw_index_coverage(segment, fragment);
                    }
                }
                Operation::CreateIndex { new_indices, .. } => {
                    if let Some(bitmap) = new_indices
                        .iter_mut()
                        .find(|index| index.uuid == segment)
                        .and_then(|index| index.fragment_bitmap.as_mut())
                    {
                        for fragment in stale {
                            if let Ok(fragment) = u32::try_from(fragment) {
                                bitmap.remove(fragment);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Whether index details describe an NGram index, whose segments cannot
/// reach rows through the fragment reuse history.
pub(super) fn is_ngram(details: Option<&Arc<prost_types::Any>>) -> bool {
    details.is_some_and(|details| details.type_url.ends_with("NGramIndexDetails"))
}

/// Every index segment `transaction` adds or widens over committed fragments,
/// where `ngram_segments` are the widened segments that are NGram segments.
/// System indices cover no rows a commit could rewrite.
fn new_coverage(
    transaction: &Transaction,
    ngram_segments: &HashSet<uuid::Uuid>,
) -> Vec<NewCoverage> {
    let committed_fields = |fields: &[ActionFieldRef], covering: &[ActionFieldRef]| {
        fields
            .iter()
            .chain(covering)
            .filter_map(|field| field.committed())
            .filter(|field| *field >= 0)
            .collect::<Vec<i32>>()
    };
    let committed_fragments = |fragments: &[ActionRef]| {
        fragments
            .iter()
            .filter_map(|fragment| fragment.committed())
            .collect::<Vec<u64>>()
    };
    match &transaction.operation {
        Operation::CreateIndex { new_indices, .. } => new_indices
            .iter()
            .filter(|index| !is_system_index(index))
            .filter_map(|index| {
                Some(NewCoverage {
                    segment: index.uuid,
                    fields: index.dependency_fields().into_owned(),
                    fragments: index
                        .fragment_bitmap
                        .as_ref()?
                        .iter()
                        .map(u64::from)
                        .collect(),
                    built_at: index.dataset_version,
                    cannot_follow_moves: is_ngram(index.index_details.as_ref()),
                })
            })
            .collect(),
        Operation::CompositeOperation(composite_operation) => composite_operation
            .iter_actions()
            .filter_map(|action| match action {
                Action::AddIndexSegment(segment) => {
                    let parts = segment.parts();
                    Some(NewCoverage {
                        segment: parts.uuid,
                        fields: committed_fields(&parts.fields, &parts.covering_fields),
                        fragments: committed_fragments(parts.covered_fragments.as_deref()?),
                        built_at: transaction.read_version,
                        cannot_follow_moves: is_ngram(parts.index_details.as_ref()),
                    })
                }
                Action::AdjustIndexCoverage(adjust) => {
                    let parts = adjust.parts();
                    Some(NewCoverage {
                        segment: parts.uuid,
                        fields: committed_fields(&parts.fields, &parts.covering_fields),
                        fragments: committed_fragments(&parts.add_fragments),
                        built_at: transaction.read_version,
                        cannot_follow_moves: ngram_segments.contains(&parts.uuid),
                    })
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
