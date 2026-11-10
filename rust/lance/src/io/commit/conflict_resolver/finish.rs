// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::provenance::revives_rows;

impl<'a> TransactionRebase<'a> {
    /// Writes
    pub async fn finish(self, dataset: &Dataset) -> Result<Transaction> {
        Ok(self.finish_with_tagged_rewrite(dataset).await?.0)
    }

    /// [`Self::finish`], also returning what a rewrite on a tagged fragment
    /// reuse history assembled for this attempt: the complete entry merged
    /// against `dataset`'s manifest, which the commit hands to the manifest
    /// build as `FragReuseUpdate::Rewrite` on the prepared index list. The transaction
    /// itself keeps the caller's entry, so the next attempt diffs the same
    /// intent again instead of treating this assembly as its own.
    pub async fn finish_with_tagged_rewrite(
        self,
        dataset: &Dataset,
    ) -> Result<(Transaction, Option<Arc<TaggedRewriteAssembly>>)> {
        match &self.transaction.operation {
            Operation::Rewrite { .. } => self.finish_rewrite(dataset).await,
            _ => Ok((self.finish_other(dataset).await?, None)),
        }
    }

    async fn finish_other(self, dataset: &Dataset) -> Result<Transaction> {
        match &self.transaction.operation {
            Operation::Delete { .. } | Operation::Update { .. } => {
                self.finish_delete_update(dataset).await
            }
            Operation::CreateIndex { .. } => self.finish_create_index(dataset).await,
            Operation::Rewrite { .. } => unreachable!("finish_with_tagged_rewrite handles Rewrite"),
            Operation::DataOverlay { .. } => self.finish_data_overlay(dataset).await,
            Operation::Append { .. }
            | Operation::Overwrite { .. }
            | Operation::DataReplacement { .. }
            | Operation::Merge { .. }
            | Operation::Restore { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::Clone { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::UpdateBases { .. } => Ok(self.transaction),
            Operation::Unknown { .. } => Err(Error::not_supported(format!(
                "Transaction {} has an operation written by a newer version of Lance \
                 and cannot be committed by this version",
                self.transaction.uuid
            ))),
        }
    }

    async fn finish_delete_update(mut self, dataset: &Dataset) -> Result<Transaction> {
        if self
            .initial_fragments
            .iter()
            .any(|(_, (_, needs_rewrite))| *needs_rewrite)
        {
            if let Some(affected_rows) = self.affected_rows {
                // Then we do the rebase

                // 1. Load the deletion files that need a rewrite.
                // 2. Validate there is no overlap with the affected rows. (if there is, return retryable conflict error)
                // 3. Write out new deletion files with existing deletes | affected rows.
                // 4. Update the transaction with the new deletion files.

                let fragments_ids_to_rewrite = self
                    .initial_fragments
                    .iter()
                    .filter_map(|(_, (fragment, needs_rewrite))| {
                        if *needs_rewrite {
                            Some(fragment.id)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                // We are rewriting the deletion files on the *current* dataset.
                let files_to_rewrite = dataset
                    .fragments()
                    .as_slice()
                    .iter()
                    .filter_map(|fragment| {
                        if fragments_ids_to_rewrite.contains(&fragment.id) {
                            Some((fragment.id, fragment.deletion_file.clone()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let existing_deletion_vecs = futures::stream::iter(files_to_rewrite)
                    .map(|(fragment_id, deletion_file)| async move {
                        read_dataset_deletion_file(
                            dataset,
                            fragment_id,
                            &deletion_file.expect("there should be a deletion file"),
                        )
                        .await
                        .map(|dv| (fragment_id, dv))
                    })
                    .buffered(dataset.object_store.as_ref().io_parallelism())
                    .try_collect::<Vec<_>>()
                    .await?;

                // Check for row-level conflicts
                let mut existing_deletions = RowAddrTreeMap::new();
                for (fragment_id, deletion_vec) in existing_deletion_vecs {
                    existing_deletions
                        .insert_bitmap(fragment_id as u32, deletion_vec.as_ref().into());
                }
                let conflicting_rows = existing_deletions.clone() & affected_rows.clone();
                if conflicting_rows.len().map(|v| v > 0).unwrap_or(true) {
                    let sample_addressed = conflicting_rows
                        .row_addrs()
                        .unwrap()
                        .take(5)
                        .collect::<Vec<_>>();
                    return Err(crate::Error::retryable_commit_conflict_source(dataset.manifest.version, format!(
                        "This {} transaction was preempted by concurrent transaction {} (both modified rows at addresses {:?}). Please retry",
                        self.transaction.uuid,
                        dataset.manifest.version,
                        sample_addressed.as_slice()
                    )
                        .into()));
                }

                let merged = existing_deletions.clone() | affected_rows.clone();

                let mut new_deleted_frag_ids = Vec::new();
                let mut new_deletion_files = HashMap::with_capacity(fragments_ids_to_rewrite.len());
                for fragment_id in fragments_ids_to_rewrite.iter() {
                    let dv = DeletionVector::from(
                        merged
                            .get_fragment_bitmap(*fragment_id as u32)
                            .unwrap()
                            .clone(),
                    );
                    // If we've deleted all rows in the fragment, we can delete it.
                    // It's acceptable if we don't handle it here, as the commit step
                    // can handle it later. Though it should be rare that physical_rows
                    // is missing.
                    if let Some(physical_rows) = self
                        .initial_fragments
                        .get(fragment_id)
                        .and_then(|(fragment, _)| fragment.physical_rows)
                        && dv.len() == physical_rows
                    {
                        new_deleted_frag_ids.push(*fragment_id);
                        continue;
                    }

                    let new_deletion_file = write_deletion_file(
                        &dataset.base,
                        *fragment_id,
                        dataset.manifest.version,
                        &dv,
                        dataset.object_store.as_ref(),
                        dataset.inline_deletions_max_bytes()?,
                    )
                    .await?;

                    // Make sure this is available in the cache for future conflict resolution.
                    let deletion_file = new_deletion_file.as_ref().unwrap();
                    let key = crate::session::caches::DeletionFileKey {
                        fragment_id: *fragment_id,
                        deletion_file,
                    };
                    dataset
                        .metadata_cache
                        .insert_with_key(&key, Arc::new(dv))
                        .await;

                    // TODO: also cleanup the old deletion file.
                    new_deletion_files.insert(*fragment_id, new_deletion_file);
                }

                match &mut self.transaction.operation {
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
                        for updated in updated_fragments {
                            if let Some(new_deletion_file) = new_deletion_files.get(&updated.id) {
                                updated.deletion_file = new_deletion_file.clone();
                            }
                        }
                        removed_fragment_ids.extend(new_deleted_frag_ids);
                    }
                    _ => {}
                }

                Ok(Transaction {
                    read_version: dataset.manifest.version,
                    ..self.transaction
                })
            } else {
                // We shouldn't hit this.
                Err(crate::Error::internal(
                    "We have a transaction that needs to be rebased, but we don't have any affected rows.",
                ))
            }
        } else {
            Ok(Transaction {
                read_version: dataset.manifest.version,
                ..self.transaction
            })
        }
    }

    /// Verify no concurrent row-moving Update dropped the values of any cell
    /// this overlay covers. `check_data_overlay_txn` flags (via the
    /// `initial_fragments` needs-check bool) each overlaid fragment on which a
    /// concurrent RewriteRows update relocated rows; here we read the deletion
    /// vectors and conflict only when the moved rows intersect the overlay's
    /// coverage.
    ///
    /// The moved rows are computed as the current deletion vector minus the
    /// read-time one. In the rare case where both a concurrent Delete and a
    /// concurrent Update touched the same flagged fragment, the Delete's rows are
    /// also counted and may trigger an unnecessary retry — never data loss. Pure
    /// concurrent deletes leave the fragment unflagged and are not examined here.
    async fn finish_data_overlay(self, dataset: &Dataset) -> Result<Transaction> {
        let fragments_to_check: HashSet<u64> = self
            .initial_fragments
            .iter()
            .filter_map(|(id, (_, needs_check))| needs_check.then_some(*id))
            .collect();
        if fragments_to_check.is_empty() {
            return Ok(Transaction {
                read_version: dataset.manifest.version,
                ..self.transaction
            });
        }

        // Coverage (physical offsets, unioned across fields) per flagged fragment.
        let Operation::DataOverlay { groups } = &self.transaction.operation else {
            return Err(wrong_operation_err(&self.transaction.operation));
        };
        let mut coverage_by_fragment: HashMap<u64, RoaringBitmap> = HashMap::new();
        for group in groups {
            if !fragments_to_check.contains(&group.fragment_id) {
                continue;
            }
            *coverage_by_fragment.entry(group.fragment_id).or_default() |=
                overlay_group_coverage(group);
        }

        for (fragment_id, coverage) in coverage_by_fragment {
            let Some(current_fragment) = dataset
                .fragments()
                .as_slice()
                .iter()
                .find(|f| f.id == fragment_id)
            else {
                // The fragment is gone entirely; the overlay is orphaned.
                return Err(crate::Error::retryable_commit_conflict_source(
                    dataset.manifest.version,
                    format!(
                        "This {} transaction was preempted: overlaid fragment {} was removed by a concurrent transaction. Please retry.",
                        self.transaction.uuid, fragment_id
                    )
                    .into(),
                ));
            };
            let current_deletions =
                read_fragment_deletion_bitmap(dataset, current_fragment).await?;
            let initial_deletions = match self.initial_fragments.get(&fragment_id) {
                Some((initial_fragment, _)) => {
                    read_fragment_deletion_bitmap(dataset, initial_fragment).await?
                }
                None => RoaringBitmap::new(),
            };
            let moved_rows = &current_deletions - &initial_deletions;
            let conflicting = &moved_rows & &coverage;
            if !conflicting.is_empty() {
                let sample: Vec<u32> = conflicting.iter().take(5).collect();
                return Err(crate::Error::retryable_commit_conflict_source(
                    dataset.manifest.version,
                    format!(
                        "This {} transaction was preempted by a concurrent update that moved overlaid rows on fragment {} (offsets {:?}). Please retry.",
                        self.transaction.uuid, fragment_id, sample.as_slice()
                    )
                    .into(),
                ));
            }
        }

        Ok(Transaction {
            read_version: dataset.manifest.version,
            ..self.transaction
        })
    }

    async fn finish_create_index(mut self, dataset: &Dataset) -> Result<Transaction> {
        // Computed before the operation is borrowed mutably below.
        let self_is_superseded_prune = is_superseded_prune_transaction(&self.transaction);
        if let Operation::CreateIndex {
            new_indices,
            removed_indices,
            ..
        } = &mut self.transaction.operation
        {
            // A tagged trim is re-derived against the CURRENT entry whenever
            // this attempt builds on a version newer than the one it read,
            // mirroring the stable-partition rewrite's assemble-at-attempt:
            // a concurrent append's records are re-filtered in, a concurrent
            // trim's result is re-trimmed instead of spliced over. When
            // nothing is left to trim after the rebase (a concurrent commit,
            // such as an index catching up mid-trim, already satisfied it),
            // the attempt aborts with a marker conflict instead of writing an
            // empty no-op version; `cleanup_frag_reuse_index` treats exactly
            // that conflict as success.
            if is_tagged_trim_operation(new_indices, removed_indices)
                && dataset.manifest.version != self.transaction.read_version
            {
                match derive_tagged_trim(dataset).await? {
                    TaggedTrimOutcome::NothingToTrim => {
                        return Err(Error::retryable_commit_conflict_source(
                            dataset.manifest.version,
                            crate::dataset::index::frag_reuse::TAGGED_TRIM_REBASED_TO_NOOP.into(),
                        ));
                    }
                    TaggedTrimOutcome::Replace {
                        new_entry,
                        current_entry,
                    } => {
                        *new_indices = vec![*new_entry];
                        *removed_indices = vec![*current_entry];
                    }
                    TaggedTrimOutcome::Delete { current_entry } => {
                        new_indices.clear();
                        *removed_indices = vec![*current_entry];
                    }
                }
                return Ok(self.transaction);
            }

            // A superseded-segment prune's removal of a segment is justified
            // by the coverage of kept same-name siblings, and a concurrent
            // commit can withdraw exactly that justification without touching
            // any segment in the removal set (e.g. a remap swap replacing a
            // kept sibling with a narrower-coverage one), so no conflict
            // predicate fires. Re-committing the original removal set after
            // such a rebase would silently drop index coverage. Mirror the
            // tagged trim: whenever this attempt builds on a version newer
            // than the one it read, re-derive the removal set wholesale
            // against the current manifest, and when nothing is superseded
            // anymore abort with a marker conflict instead of writing an
            // empty no-op version; `prune_superseded_segments` treats exactly
            // that conflict as success.
            if self_is_superseded_prune && dataset.manifest.version != self.transaction.read_version
            {
                let rederived = derive_superseded_segments(dataset).await?;
                if rederived.is_empty() {
                    return Err(Error::retryable_commit_conflict_source(
                        dataset.manifest.version,
                        crate::dataset::index::frag_reuse::SUPERSEDED_PRUNE_REBASED_TO_NOOP.into(),
                    ));
                }
                *removed_indices = rederived;
                return Ok(self.transaction);
            }

            if let Some(base) = self.frag_reuse_base.take() {
                let current = stored_frag_reuse_entry(dataset).await?;
                // A tagged history is refused by the commit gate instead.
                if !current.as_ref().is_some_and(is_tagged) && !base.is_current(current.as_ref()) {
                    return Err(base.changed_err(
                        &self.transaction.operation,
                        current.as_ref(),
                        dataset,
                        "Run the cleanup again on the latest version.",
                    ));
                }
            }

            // Cleanup can hide in-place rewrites and restores from transaction
            // checks. Added columns and further deletions keep coverage, but a
            // revived row was never indexed by this build.
            if dataset.manifest.version != self.transaction.read_version
                && let Some(read_fragments) = &self.read_fragments
            {
                let rewrites =
                    Transaction::merge_rewritten_fields(read_fragments, dataset.fragments());
                Transaction::withdraw_in_place_rewrites(
                    new_indices,
                    rewrites,
                    dataset.schema(),
                    dataset.fragment_bitmap.as_ref(),
                    self.current_lineage.as_ref(),
                );

                let covered_fragments = new_indices
                    .iter()
                    .filter(|index| !is_system_index(index))
                    .filter_map(|index| index.fragment_bitmap.as_ref())
                    .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap);
                let previously_deleted_fragments = read_fragments
                    .iter()
                    .filter(|fragment| fragment.deletion_file.is_some())
                    .map(|fragment| (fragment.id, fragment))
                    .collect::<HashMap<_, _>>();
                let mut revived_fragments = RoaringBitmap::new();
                // Both deletion files are addressable through the current dataset.
                for current in dataset.fragments().iter() {
                    if covered_fragments.contains(current.id as u32)
                        && let Some(previous) = previously_deleted_fragments.get(&current.id)
                        && revives_rows(dataset, dataset, previous, current).await?
                    {
                        revived_fragments.insert(current.id as u32);
                    }
                }
                for index in new_indices
                    .iter_mut()
                    .filter(|index| !is_system_index(index))
                {
                    if let Some(coverage) = index.fragment_bitmap.as_mut() {
                        *coverage -= &revived_fragments;
                    }
                }
            }

            // A rewrite committed since this build started retired its source
            // fragments; a segment landing with coverage of them relies on
            // translating through the history at query time (the deferred
            // rule `check_create_index` admits). A segment whose type cannot
            // do that would land addressing provenance it can never
            // translate, so the retired part of its coverage is withdrawn
            // here, in the same commit, exactly as the rewrite's own commit
            // withdraws it in the other order.
            //
            // Only a REWRITE's sources are withdrawn: a fragment the history
            // records as consumed. A fragment that merely left the manifest
            // (a whole-fragment delete) carries no rows anywhere, so no
            // segment can serve them and the coverage the reader already
            // filters to the live set needs no narrowing. Without a walkable
            // history the whole retired set stays conservative.
            if dataset.manifest.version != self.transaction.read_version
                && let (Some(lineage), Some(live)) =
                    (self.current_lineage.as_ref(), self.current_live.as_ref())
            {
                let retired = new_indices
                    .iter()
                    .filter_map(|index| index.fragment_bitmap.as_ref())
                    .fold(RoaringBitmap::new(), |retired, bitmap| {
                        retired | (bitmap - live)
                    });
                let retired: RoaringBitmap = retired
                    .iter()
                    .filter(|fragment| {
                        lineage
                            .destinations_of(*fragment)
                            .is_none_or(|destinations| !destinations.is_empty())
                    })
                    .collect();
                // The helper's `rewritten` must still be live in the liveness
                // the withdrawal is measured against: that is what makes the
                // remainder rule fire only for provenance the segment was
                // ALREADY addressing across a rewrite when it was built. Here
                // the rewrite is already committed, so restore the liveness of
                // the transaction's read snapshot for exactly the sources this
                // rewrite retired. Older provenance, retired before the build
                // read the table, stays outside and keeps the conservative
                // ancestor and remainder rules.
                // The snapshot the transaction read, not the handle it was
                // submitted through: a handle already past the rewrite does
                // not list its sources, which would leave their pre-rewrite
                // liveness unrestored and clear the untouched remainder.
                let pre_rewrite_live = match self.read_fragments.as_ref() {
                    Some(read_fragments) => {
                        let read_live: RoaringBitmap = read_fragments
                            .iter()
                            .filter_map(|fragment| u32::try_from(fragment.id).ok())
                            .collect();
                        live | (&retired & read_live)
                    }
                    None => live.clone(),
                };
                let undeferrable = crate::index::frag_reuse::undeferrable_segments_covering(
                    dataset,
                    new_indices.iter(),
                    &retired,
                    Some(lineage),
                    &pre_rewrite_live,
                )
                .await?;
                Transaction::withdraw_undeferrable_rewrite_coverage(
                    new_indices,
                    Some(lineage),
                    &pre_rewrite_live,
                    &undeferrable,
                    &retired,
                );
            }

            // Handle MEM_WAL_INDEX rebasing
            let has_mem_wal = new_indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME);

            if has_mem_wal && !self.conflicting_mem_wal_compacted_sstables.is_empty() {
                let pos = new_indices
                    .iter()
                    .position(|idx| idx.name == MEM_WAL_INDEX_NAME)
                    .unwrap();

                let mut details = load_mem_wal_index_details(new_indices[pos].clone())?;

                // Reconcile conflicting compacted_sstables by keeping each shard's higher
                // generation. Both sides are already-committed facts here, so the higher one
                // is correct; rejecting a stale proposal is the job of apply-time validation
                // against the latest state, which runs after this rebase.
                // We own self so we can consume conflicting_mem_wal_compacted_sstables directly
                for new_sstable in self.conflicting_mem_wal_compacted_sstables {
                    if let Some(existing) = details
                        .compacted_sstables
                        .iter_mut()
                        .find(|sstable| sstable.shard_id == new_sstable.shard_id)
                    {
                        if new_sstable.generation > existing.generation {
                            existing.generation = new_sstable.generation;
                        }
                    } else {
                        details.compacted_sstables.push(new_sstable);
                    }
                }

                // Replaced in place so the index list keeps its order.
                new_indices[pos] = new_mem_wal_index_meta(dataset.manifest.version, details)?;
            }

            for singleton_name in [FRAG_REUSE_INDEX_NAME, MEM_WAL_INDEX_NAME] {
                if new_indices.iter().any(|idx| idx.name == singleton_name) {
                    for existing_idx in dataset
                        .load_indices()
                        .await?
                        .iter()
                        .filter(|idx| idx.name == singleton_name)
                        .cloned()
                    {
                        if !removed_indices
                            .iter()
                            .any(|removed_idx| removed_idx.uuid == existing_idx.uuid)
                        {
                            removed_indices.push(existing_idx);
                        }
                    }
                }
            }

            Ok(self.transaction)
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }

    async fn finish_rewrite(
        mut self,
        dataset: &Dataset,
    ) -> Result<(Transaction, Option<Arc<TaggedRewriteAssembly>>)> {
        if let Operation::Rewrite {
            groups,
            frag_reuse_index,
            ..
        } = &mut self.transaction.operation
        {
            let Some(proposed) = frag_reuse_index else {
                return Ok((self.transaction, None));
            };
            // A compaction that shaped itself for a v0 table can lose the
            // race to a concurrent commit that made the table tagged (a
            // stable-partition lift, or a tagged compaction). The two record
            // forms are informationally equivalent -- a legacy reuse group
            // is exactly an ordered-compaction transition, with the same
            // digest ordering and changed_row_addrs semantics -- so instead
            // of dying at the tagged gate, the retry CONVERTS: the legacy
            // versions this transaction's entry adds relative to the entry
            // at its read version become ordered-compaction transitions
            // handed to the tagged assembly below, which appends onto the
            // CURRENT entry with full revalidation and a refreshed base
            // version. The stale versions the v0 entry carried are not
            // re-spliced: the current entry is the authoritative history,
            // concurrent trims included. The conversion keys on
            // `uses_tagged_fri` -- the sticky FLAG_FRAGMENT_REUSE_INDEX or a
            // tagged current entry -- so it still fires when a concurrent
            // trim deleted a fully drained tagged entry: the flag survives
            // the trim and remains the final authority, and the assembly
            // then restarts the history tagged. A genuine v0 table never
            // enters this branch (neither flag nor tagged entry exists
            // there), keeping pure v0 behavior untouched; writers predating
            // tagged support are fenced off by the writer feature flag
            // before this point.
            if self.reuse.added_transitions.is_none() && proposed.index_version == 0 {
                let stored = crate::index::loading::load_all_indices(dataset).await?;
                let current_uses_tagged =
                    lance_table::system_index::frag_reuse::metadata::uses_tagged_fri(
                        &dataset.manifest,
                        stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
                    );
                if current_uses_tagged {
                    let base = match self.reuse.read_entry.as_ref() {
                        Some(entry) => load_frag_reuse_records(dataset, entry).await?,
                        None => InlineContent::default(),
                    };
                    let added = records_added_since(
                        &base,
                        &load_frag_reuse_records(dataset, proposed).await?,
                    )?;
                    if !added.transitions.is_empty() {
                        return Err(Error::invalid_input(
                            "a v0 fragment reuse entry cannot carry tagged transitions",
                        ));
                    }
                    if added.legacy_versions.is_empty() {
                        return Err(Error::internal(
                            "a rewrite carrying a fragment reuse entry appended no reuse version"
                                .to_string(),
                        ));
                    }
                    self.reuse.added_transitions = Some(
                        added
                            .legacy_versions
                            .iter()
                            .flat_map(legacy_version_transitions)
                            .collect(),
                    );
                }
            }

            if let Some(transitions) = self.reuse.added_transitions.as_ref() {
                // Assembled (and re-assembled after a conflict) against the
                // FRI entry committed at the version this attempt builds on,
                // so a rebase appends onto the latest entry instead of a
                // stale one. This reassembly IS the merge with concurrent
                // history changes, stable-partition appends included: the
                // current entry already carries the concurrent writer's
                // transition (or reflects a concurrent trim), ours is
                // appended after it, and the assembly validation proves the
                // result sound -- row-map ids unique across the whole
                // ledger, per-transition conservation, lineage ordering,
                // unknown mappings rejected, destinations freshly reserved,
                // no transition recorded twice, and `build_manifest`'s
                // base-version guard at splice time.
                //
                // No separate concurrent-append detection is needed even
                // though the intent never enters a transaction file:
                // overlapping writers are rejected non-retryably by the
                // double-consumption rule in `check_rewrite_txn` from the
                // committed groups, the transaction walk cannot silently
                // skip an intervening commit (a missing transaction file
                // fails `load_new_transactions`), and even a hypothetically
                // unseen overlap still fails fragment liveness in
                // `handle_rewrite_fragments`, because its sources are no
                // longer in the manifest. What remains at this point is
                // proven disjoint, and disjoint transitions merge.
                let (entry, base_entry_version) =
                    build_frag_reuse_rewrite_entry(dataset, transitions, groups).await?;
                // Established here, against the manifest THIS attempt builds
                // on, for the same reason the entry is: a concurrent
                // CreateIndex may have added a segment that cannot translate
                // through the history, and the withdrawal has to cover it.
                let rewritten: RoaringBitmap = groups
                    .iter()
                    .flat_map(|group| group.old_fragments.iter())
                    .filter_map(|fragment| u32::try_from(fragment.id).ok())
                    .collect();
                let undeferrable_segments =
                    crate::index::frag_reuse::undeferrable_rewrite_segments(dataset, &rewritten)
                        .await?;
                let assembly = TaggedRewriteAssembly::new(
                    entry,
                    base_entry_version,
                    transitions,
                    undeferrable_segments,
                )?;
                return Ok((self.transaction, Some(Arc::new(assembly))));
            }

            // A v0 snapshot on a v0 table: the whole-history replacement contract.
            {
                let new_fri = proposed;
                let base = self.frag_reuse_base.take().ok_or_else(|| {
                    Error::internal("a v0 fragment reuse entry was rebased without its base")
                })?;
                let current = stored_frag_reuse_entry(dataset).await?;
                let mut new_fri_details =
                    Arc::unwrap_or_clone(load_frag_reuse_index_details(dataset, new_fri).await?);
                let mut own = new_fri_details.versions.pop();
                let base_versions = match &base.entry {
                    Some(entry) => load_frag_reuse_index_details(dataset, entry)
                        .await?
                        .versions
                        .clone(),
                    None => Vec::new(),
                };
                let shape_err = || {
                    Error::invalid_input(format!(
                        "the rewrite's fragment reuse entry must be the entry at version {} \
                         plus one version of its own",
                        base.version
                    ))
                };
                let is_trimmed = if base.is_current(current.as_ref()) {
                    if new_fri_details.versions != base_versions {
                        return Err(shape_err());
                    }
                    false
                } else {
                    // A cleanup only drops versions, so publishing the current ones
                    // plus our own keeps every mapping.
                    let current_versions = match (&base.entry, &current) {
                        (Some(_), Some(current_entry)) => {
                            load_frag_reuse_index_details(dataset, current_entry)
                                .await?
                                .versions
                                .clone()
                        }
                        _ => Vec::new(),
                    };
                    if base.entry.is_none()
                        || current.is_none()
                        || !is_ordered_subsequence(&current_versions, &base_versions)
                    {
                        return Err(base.changed_err(
                            &self.transaction.operation,
                            current.as_ref(),
                            dataset,
                            "Commit the rewrite again through a dataset at the latest version.",
                        ));
                    }
                    if !is_ordered_subsequence(&new_fri_details.versions, &base_versions) {
                        return Err(shape_err());
                    }
                    new_fri_details.versions = current_versions;
                    true
                };
                // Stamp the rewrite's own (last) version with the version this attempt
                // publishes on top of; a retry may land on a newer one.
                let is_restamped = match own.as_mut() {
                    Some(version) if version.dataset_version != dataset.manifest.version => {
                        version.dataset_version = dataset.manifest.version;
                        true
                    }
                    _ => false,
                };
                new_fri_details.versions.extend(own);
                if !is_trimmed && !is_restamped {
                    return Ok((self.transaction, None));
                }

                let new_frag_bitmap = new_fri_details.new_frag_bitmap();

                let new_frag_reuse_index_meta = build_frag_reuse_index_metadata(
                    dataset,
                    Some(new_fri),
                    new_fri_details,
                    new_frag_bitmap,
                )
                .await?;

                *new_fri = new_frag_reuse_index_meta;
                Ok((self.transaction, None))
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }
}

/// Read a fragment's deletion vector as a bitmap of physical offsets, or an
/// empty bitmap when the fragment has no deletion file.
async fn read_fragment_deletion_bitmap(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<RoaringBitmap> {
    match &fragment.deletion_file {
        Some(deletion_file) => {
            let dv = read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
            Ok(RoaringBitmap::from(dv.as_ref()))
        }
        None => Ok(RoaringBitmap::new()),
    }
}
