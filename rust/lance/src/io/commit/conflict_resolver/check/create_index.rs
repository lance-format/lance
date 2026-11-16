// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

impl<'a> TransactionRebase<'a> {
    pub(super) fn check_create_index_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::CreateIndex {
            new_indices,
            removed_indices,
            ..
        } = &mut self.transaction.operation
        {
            let self_is_tagged_trim = is_tagged_trim_operation(new_indices, removed_indices);
            match &other_transaction.operation {
                Operation::Append { .. }
                | Operation::Clone { .. }
                // An overlay committed after this index's version is newer than
                // the index; the query path excludes its covered cells via the
                // version gate, so the build does not conflict.
                | Operation::DataOverlay { .. }
                | Operation::UpdateBases { .. } => Ok(()),
                Operation::CreateIndex {
                    new_indices: created_indices,
                    removed_indices: committed_removed_indices,
                } => {
                    if self_is_tagged_trim {
                        // Two trims of the same entry cannot merge (mirrors the
                        // v0 cleanup-vs-cleanup conflict); anything else only
                        // changes the retention inputs, which the re-derivation
                        // in `finish_create_index` recomputes from the current
                        // manifest.
                        return if created_indices
                            .iter()
                            .chain(committed_removed_indices.iter())
                            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                        {
                            Err(self.retryable_conflict_err(other_transaction, other_version))
                        } else {
                            Ok(())
                        };
                    }
                    let self_has_frag_reuse = new_indices
                        .iter()
                        .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
                    let other_has_frag_reuse = created_indices
                        .iter()
                        .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
                    let self_has_mem_wal =
                        new_indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME);
                    let other_has_mem_wal = created_indices
                        .iter()
                        .any(|idx| idx.name == MEM_WAL_INDEX_NAME);
                    let has_regular_name_conflict = new_indices
                        .iter()
                        .filter(|idx| {
                            idx.name != FRAG_REUSE_INDEX_NAME && idx.name != MEM_WAL_INDEX_NAME
                        })
                        .any(|new_index| {
                            created_indices
                                .iter()
                                .any(|created_index| created_index.name == new_index.name)
                        });
                    // An appended segment has no removed source UUID to identify the
                    // logical index it extends, so a concurrent same-name removal
                    // must conflict. Replacement-style maintenance is handled by
                    // exact source identity below, allowing disjoint segment changes
                    // under the same logical name to merge.
                    let has_append_drop_conflict = new_indices.iter().any(|new_index| {
                        !removed_indices
                            .iter()
                            .any(|removed_index| removed_index.name == new_index.name)
                            && committed_removed_indices
                                .iter()
                                .any(|removed_index| removed_index.name == new_index.name)
                    }) || created_indices.iter().any(|created_index| {
                        !committed_removed_indices
                            .iter()
                            .any(|removed_index| removed_index.name == created_index.name)
                            && removed_indices
                                .iter()
                                .any(|removed_index| removed_index.name == created_index.name)
                    });
                    // Replacement metadata depends on the exact source segments it
                    // was built from. If either side removed the same UUID, publishing
                    // a replacement would undo a drop or use a stale source identity.
                    // Concurrent pure removals remain idempotent.
                    let has_replaced_identity_conflict =
                        removed_indices.iter().any(|removed_index| {
                            committed_removed_indices.iter().any(|committed_removed| {
                                committed_removed.uuid == removed_index.uuid
                                    && (new_indices.iter().any(|new_index| {
                                        new_index.name == removed_index.name
                                            || new_index.name == committed_removed.name
                                    }) || created_indices.iter().any(|created_index| {
                                        created_index.name == removed_index.name
                                            || created_index.name == committed_removed.name
                                    }))
                            })
                        });

                    // In the staged replay a same-name CreateIndex is routine
                    // maintenance of the index the segments extend, no
                    // conflict, unless it changed what the index is.
                    let name_conflict = if self.staged_replay {
                        new_indices.iter().any(|new_index| {
                            created_indices.iter().any(|created_index| {
                                created_index.name == new_index.name
                                    && !same_logical_index(new_index, created_index)
                            })
                        })
                    } else {
                        has_regular_name_conflict || has_append_drop_conflict
                    };
                    if (self_has_frag_reuse && other_has_frag_reuse)
                        || (self_has_mem_wal && other_has_mem_wal)
                        || name_conflict
                        || has_replaced_identity_conflict
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                // Although some of the rows we indexed may have been deleted / moved,
                // row ids are still valid, so we allow this optimistically.
                Operation::Delete { .. } => Ok(()),
                Operation::Update {
                    updated_fragments,
                    fields_modified,
                    ..
                } => {
                    // An in-place column rewrite is withdrawn from an index by
                    // pruning the rewritten fragment from its bitmap. On a
                    // tagged table a segment with retired provenance covers
                    // live fragments through the history, not its bitmap, so
                    // the withdrawal follows the lineage instead
                    // (`withdraw_rewritten_coverage`, the manifest build's own
                    // rule): the transition's sources go and those rows are
                    // scanned. The build lands; nothing retries.
                    if let (Some(lineage), Some(live), Some(schema)) = (
                        self.current_lineage.as_ref(),
                        self.current_live.as_ref(),
                        self.current_schema.as_ref(),
                    ) {
                        let rewrites = Transaction::rewritten_physical_columns(
                            &other_transaction.operation,
                            schema,
                            &[],
                        );
                        Transaction::withdraw_in_place_rewrites(
                            new_indices,
                            rewrites,
                            schema,
                            live,
                            Some(lineage),
                        );
                    } else if let Some(reuse) = &self.untagged_reuse {
                        // An index built before a compaction covers its output's
                        // rows under the fragments the output was compacted from,
                        // and an index on a packed struct's child depends on the
                        // parent's physical column.
                        let updated = updated_fragments
                            .iter()
                            .map(|fragment| reuse.compacted_from(fragment.id as u32))
                            .fold(RoaringBitmap::new(), |ids, more| ids | more);
                        let fields = Transaction::with_descendants(
                            &reuse.manifest.schema,
                            fields_modified.clone(),
                        );
                        Transaction::prune_updated_fragment_ids_from_indices(
                            new_indices,
                            &updated,
                            &fields,
                        );
                    } else {
                        Transaction::prune_updated_fields_from_indices(
                            new_indices,
                            updated_fragments,
                            fields_modified,
                        );
                    }
                    Ok(())
                }
                // Merge preserves row ids. The MemWAL index's install validates
                // schema-dependent state, which a concurrent schema change invalidates.
                Operation::Merge {
                    fragments, schema, ..
                } => {
                    if new_indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
                        return Err(self.retryable_conflict_err(other_transaction, other_version));
                    }
                    // A merge that removed or retyped a field our index keys
                    // on or carries (an `alter_columns` cast rewrites the
                    // column under a new field id) leaves the index describing
                    // a column that no longer exists; the build is redone
                    // against the new field, regardless of fragment reuse history.
                    if let Some(read_schema) = self.read_schema.as_ref() {
                        let gone: HashSet<i32> = read_schema
                            .fields_pre_order()
                            .filter(|field| {
                                schema.field_by_id(field.id).is_none_or(|now| {
                                    now.data_type() != field.data_type()
                                })
                            })
                            .map(|field| field.id)
                            .collect();
                        if new_indices.iter().any(|index| {
                            index
                                .dependency_fields()
                                .iter()
                                .any(|field| gone.contains(field))
                        }) {
                            return Err(self.retryable_conflict_err(other_transaction, other_version));
                        }
                    }
                    // A merge can rewrite a column's data file in place (an
                    // `alter_columns` cast). On a tagged table that is the
                    // same case as the Update arm above: the withdrawal
                    // follows the lineage. Which columns the merge rewrote is
                    // read off the fragments as they were at our read version;
                    // a fragment that did not exist then (a destination the
                    // rewrite produced afterwards) shows an in-place rewrite
                    // by a tombstoned field in its files, and is then treated
                    // as rewritten in every dependency field.
                    if let (Some(lineage), Some(live), Some(read_fragments)) = (
                        self.current_lineage.as_ref(),
                        self.current_live.as_ref(),
                        self.read_fragments.as_ref(),
                    ) {
                        let mut rewrites = Transaction::rewritten_physical_columns(
                            &other_transaction.operation,
                            schema,
                            read_fragments,
                        );
                        let read_ids: HashSet<u64> =
                            read_fragments.iter().map(|fragment| fragment.id).collect();
                        let every_dependency_field: Vec<u32> = new_indices
                            .iter()
                            .flat_map(|index| index.dependency_fields().into_owned())
                            .filter_map(|field| u32::try_from(field).ok())
                            .collect();
                        rewrites.extend(
                            fragments
                                .iter()
                                .filter(|fragment| !read_ids.contains(&fragment.id))
                                .filter(|fragment| {
                                    fragment.files.iter().any(|file| {
                                        file.fields.iter().any(|field| {
                                            *field
                                                == lance_table::format::overlay::TOMBSTONE_FIELD_ID
                                        })
                                    })
                                })
                                .map(|fragment| (fragment.id, every_dependency_field.clone())),
                        );
                        Transaction::withdraw_in_place_rewrites(
                            new_indices,
                            rewrites,
                            schema,
                            live,
                            Some(lineage),
                        );
                    }
                    Ok(())
                }
                Operation::ReserveFragments { .. } => Ok(()),
                // A projection removes existing indices on dropped fields.
                // An index built before it must not reintroduce those dependencies.
                Operation::Project { schema, .. } => {
                    if new_indices.iter().any(|index| {
                        !is_system_index(index)
                            && index
                                .dependency_fields()
                                .iter()
                                .any(|field| schema.field_by_id(*field).is_none())
                    }) {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                // Should be compatible with rewrite if it didn't move the rows
                // we indexed. If it did, we could retry.
                // TODO: this will change with stable row ids.
                Operation::Rewrite {
                    groups,
                    frag_reuse_index,
                    ..
                } => {
                    // A tagged trim carries no state worth defending: the
                    // rewrite's appended records live in the CURRENT manifest
                    // entry (whether or not this process can see the in-memory
                    // reuse update here), and `finish_create_index`
                    // re-derives the whole trim against that entry, so the
                    // concurrently appended transition is retained naturally.
                    if self_is_tagged_trim {
                        return Ok(());
                    }
                    // if index remapping is deferred, there is no conflict with
                    // concurrent CreateIndex of column indices: the index keeps covering
                    // the rewritten rows under their old fragments. On an untagged table
                    // the remap was deferred if the rewrite carries a reuse update
                    // (visible only in the process that committed it) or the latest
                    // reuse index records it. On a stable-row-id table only the latest
                    // reuse index counts: a cleanup there may have trimmed the record
                    // since, leaving an index built before the rewrite nothing to
                    // translate through.
                    // A frag_reuse_index cleanup is checked against the latest entry in
                    // `finish_create_index`. A tagged entry (an in-process rewrite on a
                    // tagged history) takes the durable-evidence path below instead.
                    let is_recorded = self
                        .untagged_reuse
                        .as_ref()
                        .is_some_and(|reuse| reuse.records(groups));
                    let has_stable_row_ids = self
                        .untagged_reuse
                        .as_ref()
                        .is_some_and(|reuse| reuse.manifest.uses_stable_row_ids());
                    let deferred = is_recorded
                        || (!has_stable_row_ids
                            && frag_reuse_index
                                .as_ref()
                                .is_some_and(|entry| !is_tagged(entry)));
                    if deferred {
                        if let Some(reuse) = &self.untagged_reuse
                            && !reuse.withdraw_folded_overlays(groups, new_indices)
                        {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }
                        let ngram_coverage = new_indices
                            .iter()
                            .filter(|idx| {
                                idx.index_details.as_ref().is_some_and(|details| {
                                    details.type_url.ends_with("NGramIndexDetails")
                                })
                            })
                            .filter_map(|idx| idx.fragment_bitmap.as_ref())
                            .fold(RoaringBitmap::new(), |coverage, fragments| {
                                coverage | fragments
                            });
                        if groups
                            .iter()
                            .flat_map(|group| group.old_fragments.iter())
                            .any(|fragment| {
                                !compacted_from(self.untagged_reuse.as_deref(), fragment.id as u32)
                                    .is_disjoint(&ngram_coverage)
                            })
                        {
                            return Err(
                                self.retryable_conflict_err(other_transaction, other_version)
                            );
                        }

                        if new_indices
                            .iter()
                            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                        {
                            // A frag_reuse_index replacement mixed with other indices is a
                            // shape this resolver does not reason about, so refuse it.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                return Err(self
                                    .incompatible_conflict_err(other_transaction, other_version));
                            }
                        }
                        Ok(())
                    } else {
                        // `frag_reuse_index` is in-memory only: a committed
                        // rewrite read back from its transaction file always
                        // shows `None` here. On a tagged table the manifest's entry
                        // is the durable evidence: a rewrite whose sources
                        // are all on the recorded lineage appended a
                        // transition (a bare rewrite is only admitted for
                        // fragments no index, the entry included, covers), so
                        // it deferred remapping exactly like a v0 rewrite
                        // carrying an entry: our segment lands with its
                        // provenance and translates at query time. The
                        // deferred rules apply per group: a logical index (its
                        // same-name segments together) covering only some of a
                        // group's sources lands with that provenance and claims
                        // nothing for the group, and an NGram
                        // segment must be rebuilt (its committed-segment
                        // check refuses retired coverage). A group off the
                        // lineage (untagged table, trimmed entry, bare
                        // rewrite) keeps the eager rule: any overlap retries.
                        // Same-name segments are one logical index: the reader
                        // derives a transition's coverage from what they hold
                        // together, so a group's sources may be split across
                        // them (one staged segment per fragment).
                        let mut union_by_name: HashMap<&str, RoaringBitmap> = HashMap::new();
                        for index in new_indices.iter() {
                            let Some(bitmap) = &index.fragment_bitmap else {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            };
                            *union_by_name.entry(index.name.as_str()).or_default() |= bitmap;
                        }
                        for group in groups {
                            let sources: Vec<u32> =
                                group.old_fragments.iter().map(|f| f.id as u32).collect();
                            let recorded = self.current_lineage.as_ref().is_some_and(|lineage| {
                                sources.iter().all(|id| lineage.contains(*id))
                            });
                            for index in new_indices.iter() {
                                let union = &union_by_name[index.name.as_str()];
                                let covered = sources.iter().filter(|id| union.contains(**id)).count();
                                if covered == 0 {
                                    continue;
                                }
                                let is_ngram = index.index_details.as_ref().is_some_and(|details| {
                                    details.type_url.ends_with("NGramIndexDetails")
                                });
                                // Partial coverage of a recorded group is not a
                                // conflict: the segment keeps its provenance and
                                // the reader claims nothing for that transition
                                // (those rows are scanned), which a later
                                // optimize repairs. A covered fragment the
                                // history does not record (a bare rewrite), or
                                // an NGram segment, needs the rebuild.
                                if !recorded || is_ngram {
                                    return Err(
                                        self.retryable_conflict_err(other_transaction, other_version)
                                    );
                                }
                            }
                        }
                        Ok(())
                    }
                }
                Operation::UpdateConfig { .. } => Ok(()),
                Operation::DataReplacement { replacements } => {
                    // A data replacement only conflicts if it is updating a field the
                    // index depends on -- whether keyed on or merely carried (see
                    // `IndexMetadata::dependency_fields`).
                    // On a tagged table the replaced column's coverage is
                    // withdrawn along the lineage instead, like the Update arm.
                    if let (Some(lineage), Some(live), Some(schema)) = (
                        self.current_lineage.as_ref(),
                        self.current_live.as_ref(),
                        self.current_schema.as_ref(),
                    ) {
                        let rewrites = Transaction::rewritten_physical_columns(
                            &other_transaction.operation,
                            schema,
                            &[],
                        );
                        Transaction::withdraw_in_place_rewrites(
                            new_indices,
                            rewrites,
                            schema,
                            live,
                            Some(lineage),
                        );
                        return Ok(());
                    }
                    let mut newly_depended_fields = new_indices
                        .iter()
                        .flat_map(|idx| idx.dependency_fields().into_owned())
                        .collect::<Vec<_>>();
                    newly_depended_fields.sort_unstable();
                    newly_depended_fields.dedup();
                    for replacement in replacements {
                        for field in replacement.1.fields.iter() {
                            // A packed parent's file also replaces the indexed
                            // values of its children.
                            if newly_depended_fields.contains(field)
                                || self.read_schema.as_ref().is_some_and(|schema| {
                                    field_affects_index(*field, &newly_depended_fields, schema)
                                })
                            {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }
                        }
                    }
                    Ok(())
                }
                Operation::UpdateMemWalState {
                    compacted_sstables: other_compacted_sstables,
                    ..
                } => {
                    // CreateIndex of MemWalIndex is compatible with UpdateMemWalState
                    // as they can be rebased on each other
                    if new_indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
                        // Collect compacted_sstables from UpdateMemWalState for rebasing
                        self.conflicting_mem_wal_compacted_sstables
                            .extend(other_compacted_sstables.iter().cloned());
                        Ok(())
                    } else {
                        Err(self.incompatible_conflict_err(other_transaction, other_version))
                    }
                }
                Operation::Overwrite { .. } | Operation::Restore { .. } => {
                    Err(self.incompatible_conflict_err(other_transaction, other_version))
                }
                // An unrecognized operation may touch anything: assume it conflicts.
                Operation::Unknown { .. } => Err(self.incompatible_conflict_err(other_transaction, other_version)),
            }
        } else {
            Err(wrong_operation_err(&self.transaction.operation))
        }
    }
}
