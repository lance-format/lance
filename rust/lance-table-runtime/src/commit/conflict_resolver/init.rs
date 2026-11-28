// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::datatypes::Schema;

use super::*;

impl<'a> TransactionRebase<'a> {
    pub async fn try_new(
        dataset: &Table,
        transaction: Transaction,
        affected_rows: Option<&'a RowAddrTreeMap>,
    ) -> Result<Self> {
        Self::try_new_with_inserted_row_count(dataset, transaction, affected_rows, None).await
    }

    /// Capture a staged write with its observed number of inserted rows.
    /// Pass `None` when the writer did not supply a count. Physical addresses
    /// alone identify removed rows but cannot account for untracked inserts.
    ///
    /// ```
    /// # use lance_table::transaction::Transaction;
    /// # use lance_table_runtime::{Table, commit::conflict_resolver::TransactionRebase};
    /// # async fn rebase(table: &Table, tx: Transaction, count: u64) -> lance_core::Result<()> {
    /// let rebase = TransactionRebase::try_new_with_inserted_row_count(table, tx, None, Some(count)).await?;
    /// # Ok(()) }
    /// ```
    pub async fn try_new_with_inserted_row_count(
        dataset: &Table,
        transaction: Transaction,
        affected_rows: Option<&'a RowAddrTreeMap>,
        inserted_row_count: Option<u64>,
    ) -> Result<Self> {
        let read = if staged_update::needs_row_history(&transaction, affected_rows)
            && dataset.version_id() != transaction.read_version
        {
            Cow::Owned(dataset.checkout_version(transaction.read_version).await?)
        } else {
            Cow::Borrowed(dataset)
        };
        Self::for_commit_attempt(
            &read,
            transaction,
            affected_rows,
            Some(&read),
            inserted_row_count,
        )
        .await
    }

    pub(in crate::commit) async fn for_commit_attempt(
        dataset: &Table,
        transaction: Transaction,
        affected_rows: Option<&'a RowAddrTreeMap>,
        row_read: Option<&Table>,
        inserted_row_count: Option<u64>,
    ) -> Result<Self> {
        match &transaction.operation {
            // These operations add new fragments or don't modify any.
            Operation::Append { .. }
            | Operation::Overwrite { .. }
            | Operation::CreateIndex { .. }
            | Operation::ReserveFragments { .. }
            | Operation::Project { .. }
            | Operation::UpdateConfig { .. }
            | Operation::UpdateMemWalState { .. }
            | Operation::Clone { .. }
            | Operation::Restore { .. }
            | Operation::UpdateBases { .. }
            // An action set can modify fragments, but conflicts against it are
            // settled by comparing footprints, which are derived from the
            // actions rather than from collected rebase state.
            | Operation::CompositeOperation(_) => {
                let is_create_index =
                    matches!(transaction.operation, Operation::CreateIndex { .. });
                let is_composite =
                    matches!(transaction.operation, Operation::CompositeOperation(_));
                let read_fragments = is_create_index.then(|| ReadFragments {
                    fragments: dataset.fragments().clone(),
                    moved_files: None,
                });
                let read_schema =
                    (is_create_index || is_composite).then(|| dataset.schema().clone());
                let frag_reuse_base = match &transaction.operation {
                    Operation::CreateIndex { new_indices, .. }
                        if new_indices
                            .iter()
                            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME) =>
                    {
                        Some(Box::new(FragReuseBase::load(dataset).await?))
                    }
                    _ => None,
                };
                Ok(Self {
                    transaction,
                    affected_rows,
                    initial_fragments: HashMap::new(),
                    row_history: None,
                    modified_fragment_ids: HashSet::new(),
                    frag_reuse_base,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments,
                    read_schema,
                    reuse: Default::default(),
                    staged_replay: false,
                })
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            }
            | Operation::Update {
                updated_fragments,
                removed_fragment_ids: deleted_fragment_ids,
                ..
            } => {
                let modified_fragment_ids = updated_fragments
                    .iter()
                    .map(|f| f.id)
                    .chain(deleted_fragment_ids.iter().copied())
                    .collect::<HashSet<_>>();

                // A column patch that publishes no new fragments cannot insert
                // rows, including when staged by a writer without MERGE statistics.
                let inserted_row_count = match &transaction.operation {
                    Operation::Update { new_fragments, .. } if new_fragments.is_empty() => Some(0),
                    _ => inserted_row_count,
                };
                let row_history = row_read
                    .filter(|_| staged_update::needs_row_history(&transaction, affected_rows))
                    .map(|read| staged_update::RowHistory::new(read, inserted_row_count));

                if updated_fragments.is_empty() && affected_rows.is_some() {
                    // Removed fragments conflict as a whole. A fragment-only append,
                    // however, must distinguish a supplied row map from omission.
                    let affected_rows = affected_rows.filter(|_| deleted_fragment_ids.is_empty());
                    return Ok(Self {
                        transaction,
                        row_history,
                        initial_fragments: HashMap::new(),
                        modified_fragment_ids,
                        affected_rows,
                        frag_reuse_base: None,
                        conflicting_mem_wal_compacted_sstables: Vec::new(),
                        current_lineage: None,
                        untagged_reuse: None,
                        current_live: None,
                        current_schema: None,
                        read_fragments: None,
                        read_schema: None,
                        reuse: Default::default(),
                        staged_replay: false,
                    });
                }

                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await?;
                let mut rebase = Self {
                    transaction,
                    row_history,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    frag_reuse_base: None,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments: None,
                    read_schema: None,
                    reuse: Default::default(),
                    staged_replay: false,
                };
                if rebase.row_history.is_some() {
                    rebase.validate_row_rewrite(inserted_row_count);
                }
                Ok(rebase)
            }
            Operation::Rewrite {
                groups,
                frag_reuse_index,
                ..
            } => {
                let modified_fragment_ids = groups
                    .iter()
                    .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                    .collect::<HashSet<_>>();
                let frag_reuse_base = if frag_reuse_index
                    .as_ref()
                    .is_some_and(|entry| !is_tagged(entry))
                {
                    Some(Box::new(FragReuseBase::load(dataset).await?))
                } else {
                    None
                };

                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await?;
                let reuse = rewrite_reuse_state(dataset, &transaction).await?;
                Ok(Self {
                    transaction,
                    row_history: None,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    frag_reuse_base,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments: None,
                    read_schema: None,
                    reuse,
                    staged_replay: false,
                })
            }
            Operation::DataReplacement { replacements, .. } => {
                let modified_fragment_ids =
                    replacements.iter().map(|r| r.0).collect::<HashSet<_>>();
                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await?;
                Ok(Self {
                    transaction,
                    row_history: None,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    frag_reuse_base: None,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments: None,
                    read_schema: None,
                    reuse: Default::default(),
                    staged_replay: false,
                })
            }
            Operation::DataOverlay { groups } => {
                let modified_fragment_ids =
                    groups.iter().map(|g| g.fragment_id).collect::<HashSet<_>>();
                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await?;
                Ok(Self {
                    transaction,
                    row_history: None,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    frag_reuse_base: None,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments: None,
                    read_schema: None,
                    reuse: Default::default(),
                    staged_replay: false,
                })
            }
            Operation::Merge { fragments, .. } => {
                let modified_fragment_ids = fragments.iter().map(|f| f.id).collect::<HashSet<_>>();
                let initial_fragments =
                    initial_fragments_for_rebase(dataset, &transaction, &modified_fragment_ids)
                        .await?;
                Ok(Self {
                    transaction,
                    row_history: None,
                    affected_rows,
                    initial_fragments,
                    modified_fragment_ids,
                    frag_reuse_base: None,
                    conflicting_mem_wal_compacted_sstables: Vec::new(),
                    current_lineage: None,
                    untagged_reuse: None,
                    current_live: None,
                    current_schema: None,
                    read_fragments: None,
                    read_schema: None,
                    reuse: Default::default(),
                    staged_replay: false,
                })
            }
            // An unrecognized operation cannot be rebased: we don't know what it touches.
            Operation::Unknown { .. } => Err(Error::not_supported(format!(
                "Transaction {} has an operation written by a newer version of Lance \
                 and cannot be rebased by this version",
                transaction.uuid
            ))),
        }
    }

    /// Check whether the transaction conflicts with another transaction.
    /// Mutate the current [TransactionRebase] based on `other_transaction` to be used for
    /// eventually finishing the rebase process.
    ///
    /// Will return an error if the transaction is not valid. Otherwise, it will
    /// return Ok(()).
    /// Load what a CreateIndex needs from the LATEST manifest before the
    /// per-version checks run: the lineage its tagged fragment reuse entry
    /// records. A committed rewrite is read back from its transaction file,
    /// where `frag_reuse` is never serialized, so the entry is the only
    /// durable evidence that a rewrite appended a transition (see the Rewrite
    /// arm of `check_create_index_txn`). A no-op for every other operation.
    /// Mark this rebase as the staged-segment replay; see `staged_replay`.
    pub fn for_staged_replay(mut self) -> Self {
        self.staged_replay = true;
        self
    }

    /// Use the transaction's read snapshot for every CreateIndex check, and for
    /// an action set's footprints. The submitting handle may already be newer,
    /// hiding rewritten columns or removed fields when compared with the latest
    /// version.
    pub(crate) fn with_read_version_dataset(mut self, dataset: &Table) -> Self {
        match self.transaction.operation {
            Operation::CreateIndex { .. } => {
                self.read_fragments = Some(ReadFragments {
                    fragments: dataset.fragments().clone(),
                    moved_files: None,
                });
                self.read_schema = Some(dataset.schema().clone());
            }
            Operation::CompositeOperation(_) => {
                self.read_schema = Some(dataset.schema().clone());
            }
            _ => {}
        }
        self
    }

    pub async fn load_current_lineage(&mut self, dataset: &Table) -> Result<()> {
        if matches!(self.transaction.operation, Operation::CompositeOperation(_)) {
            self.current_schema = Some(dataset.schema().clone());
            return Ok(());
        }
        if !matches!(self.transaction.operation, Operation::CreateIndex { .. }) {
            return Ok(());
        }
        let indices = load_all_indices(dataset, None).await?;
        let Some(entry) = indices
            .iter()
            .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
        else {
            self.load_untagged_reuse(dataset, &indices).await?;
            return Ok(());
        };
        let ledger = crate::frag_reuse::decode_frag_reuse_ledger(dataset, entry).await?;
        self.current_lineage = Some(TaggedLineage::new(entry, Some(Arc::new(ledger))));
        self.current_live = Some(dataset.state.fragment_bitmap.as_ref().clone());
        self.current_schema = Some(dataset.schema().clone());
        Ok(())
    }

    /// The untagged counterpart of the tagged lineage. Skipped when the
    /// CreateIndex writes the fragment reuse index itself, which
    /// `finish_create_index` checks, or the MemWAL index, which covers no
    /// fragments.
    async fn load_untagged_reuse(
        &mut self,
        dataset: &Table,
        indices: &[IndexMetadata],
    ) -> Result<()> {
        let Operation::CreateIndex { new_indices, .. } = &self.transaction.operation else {
            return Ok(());
        };
        if new_indices
            .iter()
            .any(|index| index.name == FRAG_REUSE_INDEX_NAME || index.name == MEM_WAL_INDEX_NAME)
        {
            return Ok(());
        }
        let entry = indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME);
        if lance_table::system_index::frag_reuse::metadata::uses_tagged_fri(
            &dataset.manifest,
            entry,
        ) {
            return Ok(());
        }
        let mut reuse = UntaggedReuse {
            groups: HashSet::new(),
            sources: HashMap::new(),
            manifest: dataset.manifest.clone(),
        };
        if let Some(entry) = entry {
            let details = load_frag_reuse_index_details(dataset, entry).await?;
            for group in details
                .versions
                .iter()
                .flat_map(|version| version.groups.iter())
            {
                let old: Arc<[u64]> = sorted_ids(group.old_frags.iter().map(|f| f.id)).into();
                for new in &group.new_frags {
                    reuse.sources.insert(new.id as u32, old.clone());
                }
                reuse.groups.insert(old);
            }
        }
        self.untagged_reuse = Some(Box::new(reuse));
        Ok(())
    }

    /// Record the files `data_change: false` replacements write in the
    /// transactions checked next, so that the finish keeps an index's coverage
    /// of values they only moved. Call it only when those transactions are
    /// every version since the read version: the finish compares against the
    /// read version's fragments, so a change the checks did not see could lie
    /// under any moved file.
    pub(crate) fn track_moved_files(&mut self) {
        if let Some(read) = &mut self.read_fragments {
            read.moved_files = Some(HashMap::new());
        }
    }

    /// Handle gaps in the transaction history required to safely rebase.
    pub(crate) fn check_missing_versions(
        &mut self,
        missing: &[u64],
        latest_version: u64,
        read_schema: &Schema,
        latest_schema: &Schema,
    ) -> Result<()> {
        if missing.is_empty() {
            return Ok(());
        }
        if conflicts_with_unseen_schema_change(
            &self.transaction.operation,
            read_schema,
            latest_schema,
        ) {
            return Err(Error::retryable_commit_conflict_source(
                latest_version,
                format!(
                    "versions {missing:?} between transaction read version {} and latest                      version {latest_version} were cleaned up, so {} cannot safely check                      conflicts with schema changes; retry the operation on the latest version",
                    self.transaction.read_version, self.transaction.operation
                )
                .into(),
            ));
        }
        // A missing transaction may have changed a source fragment or inserted
        // the same keys without touching any shared fragment. Neither conflict
        // check can establish safety across a gap.
        //
        // An action set is checked by comparing footprints, which needs every
        // transaction in the window: replay re-validates only what still
        // exists, not that a read, key, or definition it depends on is
        // unchanged. Nearly every action set depends on one, so refuse them all.
        if !self.modified_fragment_ids.is_empty()
            || matches!(
                &self.transaction.operation,
                Operation::Update {
                    inserted_rows_filter: Some(_),
                    ..
                } | Operation::CompositeOperation(_)
            )
        {
            return Err(Error::retryable_commit_conflict_source(
                latest_version,
                format!(
                    "versions {missing:?} between transaction read version {} and latest \
                     version {latest_version} were cleaned up, so {} cannot safely check \
                     conflicts for existing fragments or inserted keys; retry the operation \
                     on the latest version",
                    self.transaction.read_version, self.transaction.operation
                )
                .into(),
            ));
        }

        // Lineage proves where retired rows moved, but not whether their
        // indexed values changed in a missing transaction. Those fragments
        // cannot be compared with the build snapshot once their IDs retire.
        if let (Some(lineage), Some(live), Operation::CreateIndex { new_indices, .. }) = (
            &self.current_lineage,
            &self.current_live,
            &mut self.transaction.operation,
        ) {
            Transaction::withdraw_unverified_retired_coverage(new_indices, lineage, live);
        }

        // Gaps can hide changes to the source fragments an index covers, or to
        // their outputs after a deferred compaction. Compactions before the build
        // do not count: the index covers their output under its own id.
        let (Some(reuse), Operation::CreateIndex { new_indices, .. }) =
            (&self.untagged_reuse, &self.transaction.operation)
        else {
            return Ok(());
        };
        if reuse.groups.iter().any(|sources| {
            new_indices.iter().any(|index| {
                index
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|bitmap| sources.iter().any(|id| bitmap.contains(*id as u32)))
            })
        }) {
            return Err(Error::retryable_commit_conflict_source(
                latest_version,
                format!(
                    "versions {missing:?} were cleaned up, so this index cannot be carried \
                     through the deferred compaction it covers; rebuild it"
                )
                .into(),
            ));
        }
        Ok(())
    }
}

/// Whether a schema change hidden in a history gap could be undone or violated
/// by `operation`. A Project replaces the schema with one derived from its read
/// version. New rows were only validated against the read schema, and their
/// files keep the field ids they were written with: a field that was dropped,
/// recast or reused by an overwrite would read them back as nulls or as the
/// wrong type, and a column that became non-nullable could hold their nulls.
fn conflicts_with_unseen_schema_change(
    operation: &Operation,
    read_schema: &Schema,
    latest_schema: &Schema,
) -> bool {
    let new_fragments = match operation {
        Operation::Project { .. } => return true,
        Operation::Append { fragments } => fragments,
        Operation::Update { new_fragments, .. } => new_fragments,
        _ => return false,
    };
    if new_fragments.is_empty() {
        return false;
    }
    let changes_written_field = new_fragments
        .iter()
        .flat_map(|fragment| &fragment.files)
        .flat_map(|file| file.fields.iter())
        .filter(|field_id| **field_id >= 0)
        .any(|field_id| {
            match (
                read_schema.field_ancestry_by_id(*field_id),
                latest_schema.field_ancestry_by_id(*field_id),
            ) {
                // An overwrite can reuse an id for another column of the same
                // type, or put the same path under another container kind, so
                // every ancestor must keep its name and kind.
                (Some(read_path), Some(latest_path)) => {
                    !read_path
                        .iter()
                        .map(|field| (&field.name, &field.logical_type))
                        .eq(latest_path
                            .iter()
                            .map(|field| (&field.name, &field.logical_type)))
                        || read_path.last().map(|field| field.data_type())
                            != latest_path.last().map(|field| field.data_type())
                }
                _ => true,
            }
        });
    let requires_new_values = latest_schema.fields_pre_order().any(|field| {
        !field.nullable
            && read_schema
                .field_by_id(field.id)
                .is_none_or(|read_field| read_field.nullable)
    });
    changes_written_field || requires_new_values
}

/// See `RewriteReuseState`. `dataset` is the handle the transaction is committed
/// through, not its read version.
async fn rewrite_reuse_state(
    dataset: &Table,
    transaction: &Transaction,
) -> Result<RewriteReuseState> {
    let Operation::Rewrite {
        frag_reuse_index: Some(proposed),
        ..
    } = &transaction.operation
    else {
        return Ok(RewriteReuseState::default());
    };
    let read_version = if dataset.manifest.version != transaction.read_version {
        // A cleaned-up read version fails here rather than guessing the
        // base: without it the entry's additions cannot be told apart.
        Cow::Owned(dataset.checkout_version(transaction.read_version).await?)
    } else {
        Cow::Borrowed(dataset)
    };
    let read_entry = load_all_indices(&read_version, None)
        .await?
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .cloned();
    let added_transitions = if is_tagged(proposed) {
        // The contract of `Operation::Rewrite::frag_reuse_index`: a tagged
        // entry is built from the snapshot at the transaction's read
        // version, the base it is diffed against here. An entry built from
        // a later snapshot folds transitions other writers appended in
        // between into this rewrite's additions, so it is refused rather
        // than misread (a distributed compaction plans at V and must not
        // build its entry from a driver handle at V+N).
        if proposed.dataset_version != transaction.read_version {
            return Err(Error::invalid_input(format!(
                "the rewrite's fragment reuse entry was built from version {} but the \
                 transaction reads version {}; build the entry from the read-version \
                 snapshot (`frag_reuse_entry_appending` on a dataset checked out at the \
                 read version)",
                proposed.dataset_version, transaction.read_version
            )));
        }
        let base = match read_entry.as_ref() {
            Some(entry) => load_frag_reuse_records(&read_version, entry).await?,
            None => InlineContent::default(),
        };
        let added = records_added_since(&base, &load_frag_reuse_records(dataset, proposed).await?)?;
        if !added.legacy_versions.is_empty() {
            return Err(Error::invalid_input(
                "a tagged fragment reuse entry appends transitions; it cannot add legacy \
                 compaction versions",
            ));
        }
        Some(added.transitions)
    } else {
        None
    };
    Ok(RewriteReuseState {
        read_entry,
        added_transitions,
    })
}

async fn initial_fragments_for_rebase(
    dataset: &Table,
    transaction: &Transaction,
    modified_fragment_ids: &HashSet<u64>,
) -> Result<HashMap<u64, (Fragment, bool)>> {
    if modified_fragment_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let dataset = if dataset.manifest.version != transaction.read_version {
        // The read version may have been garbage-collected by a concurrent
        // `cleanup_old_versions` between the commit attempt and the rebase.
        // Propagate the error so the commit fails gracefully instead of
        // panicking (which aborts the whole process when `panic = "abort"`).
        Cow::Owned(dataset.checkout_version(transaction.read_version).await?)
    } else {
        Cow::Borrowed(dataset)
    };

    Ok(dataset
        .fragments()
        .iter()
        .filter(|fragment| {
            // Check if the fragment is modified by the transaction.
            modified_fragment_ids.contains(&fragment.id)
        })
        .map(|fragment| (fragment.id, (fragment.clone(), false)))
        .collect())
}
