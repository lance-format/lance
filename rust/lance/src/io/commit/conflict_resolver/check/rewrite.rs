// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

impl<'a> TransactionRebase<'a> {
    pub(super) fn check_rewrite_txn(
        &mut self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Result<()> {
        if let Operation::Rewrite {
            groups,
            frag_reuse_index,
            rewritten_indices,
            ..
        } = &self.transaction.operation
        {
            match &other_transaction.operation {
                // Rewrite is only compatible with operations that don't touch
                // existing fragments or update fragments we don't touch.
                Operation::Append { .. }
                | Operation::ReserveFragments { .. }
                | Operation::Project { .. }
                | Operation::Clone { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateMemWalState { .. }
                | Operation::UpdateBases { .. } => Ok(()),
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
                    if updated_fragments
                        .iter()
                        .map(|f| f.id)
                        .chain(deleted_fragment_ids.iter().copied())
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::DataOverlay { groups } => {
                    // Rewriting a fragment changes its physical row addresses, so
                    // an overlay addressed by physical offset on that fragment is
                    // invalidated and must be re-applied against the new base.
                    if groups
                        .iter()
                        .map(|g| g.fragment_id)
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        Ok(())
                    }
                }
                Operation::Rewrite {
                    groups,
                    rewritten_indices: other_rewritten_indices,
                    ..
                } => {
                    // Double consumption: the committed rewrite replaced
                    // fragments our tagged transitions read from or produce,
                    // so appending our transitions would record row movement
                    // out of (or into) fragments that no longer exist. Retry
                    // rebuilds the transitions against the surviving
                    // fragments. The committed groups are serialized in the
                    // transaction file, so this rule is live across
                    // processes. Disjoint committed rewrites fall through to
                    // rebase: `finish_rewrite` re-assembles onto the current
                    // manifest entry (picking up a v0 compaction's legacy
                    // version or a tagged compaction's transition), and its
                    // manifest-diff check turns a concurrent stable-partition
                    // append into a retryable conflict.
                    if let Some(transitions) = self.reuse.added_transitions.as_ref() {
                        let touched: HashSet<u64> = transitions
                            .iter()
                            .flat_map(|transition| {
                                transition
                                    .sources
                                    .iter()
                                    .chain(transition.destinations.iter())
                                    .map(|digest| digest.id)
                            })
                            .collect();
                        if groups
                            .iter()
                            .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
                            .any(|id| touched.contains(&id))
                        {
                            // Not retryable: the same intent can never
                            // succeed once its fragments are consumed, so a
                            // retry loop only wastes attempts. The caller
                            // must rebuild the transitions (and their row
                            // maps) against the new dataset state.
                            return Err(Error::incompatible_transaction_source(
                                format!(
                                    "A concurrent {} at version {} consumed fragments this \
                                     rewrite's transitions read from or produce; this rewrite \
                                     can never succeed as committed. Rebuild the transitions \
                                     against the new dataset state.",
                                    other_transaction.operation, other_version
                                )
                                .into(),
                            ));
                        }
                    }
                    if groups
                        .iter()
                        .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                        .any(|id| self.modified_fragment_ids.contains(&id))
                    {
                        Err(self.retryable_conflict_err(other_transaction, other_version))
                    } else {
                        let mut replacement_source_counts = HashMap::new();
                        for rewritten_index in rewritten_indices {
                            *replacement_source_counts
                                .entry(rewritten_index.new_id)
                                .or_insert(0) += 1;
                        }
                        let merged_sources = rewritten_indices
                            .iter()
                            .filter(|rewritten_index| {
                                replacement_source_counts[&rewritten_index.new_id] > 1
                            })
                            .map(|rewritten_index| rewritten_index.old_id)
                            .collect::<HashSet<_>>();
                        let replaced_source =
                            other_rewritten_indices.iter().find_map(|rewritten_index| {
                                (rewritten_index.old_id != rewritten_index.new_id
                                    && merged_sources.contains(&rewritten_index.old_id))
                                .then_some(rewritten_index.old_id)
                            });
                        if let Some(source_id) = replaced_source {
                            return Err(Error::incompatible_transaction_source(
                                format!(
                                    "A concurrent rewrite at version {} replaced source index \
                                     segment {}, which this rewrite planned to merge. The stale \
                                     compaction plan can never succeed as committed; replan \
                                     compaction from the latest dataset.",
                                    other_version, source_id
                                )
                                .into(),
                            ));
                        }
                        Ok(())
                    }
                }
                Operation::DataReplacement { replacements } => {
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
                Operation::Merge { .. } => {
                    Err(self.retryable_conflict_err(other_transaction, other_version))
                }
                Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                    ..
                } => {
                    // A stable-partition rewrite defers index remapping the
                    // same way a v0 rewrite carrying a frag_reuse_index does:
                    // the retired source ids stay in the index bitmaps as
                    // provenance and the tagged entry records the row-level
                    // translation, so it takes the deferred-remap branches
                    // below.
                    let defers_remap = frag_reuse_index.is_some();
                    match (
                        new_indices
                            .iter()
                            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
                        defers_remap,
                    ) {
                        // The other transaction replaced the FRI entry (a v0
                        // cleanup trim); `finish_rewrite` works from the
                        // CURRENT manifest entry every attempt.
                        (Some(_), true) => {
                            // A cleanup commits an FRI replacement alone. A
                            // CreateIndex mixing it with user indices is a shape
                            // this resolver does not reason about (the user-index
                            // straddle check below never runs for it), so
                            // refuse it instead of guessing.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                return Err(self
                                    .incompatible_conflict_err(other_transaction, other_version));
                            }
                            Ok(())
                        }
                        // If rewrite defers index remap, the FRI handles the
                        // post-commit bitmap update — but only if each rewrite
                        // group is fully inside or fully outside each new
                        // index's fragment bitmap. A group that straddles
                        // would produce a bitmap with a mix of indexed and
                        // non-indexed fragments, which load_indices rejects.
                        (None, true) => {
                            // The compaction planned before this index existed, so
                            // its own guard could not see it. Refuse the pair here.
                            //
                            // This should be relatively rare as we are moving indexes
                            // away from row ids
                            //
                            // `old_fragments` come from the dataset's manifest at the
                            // rewrite's read version (never freshly constructed, where
                            // `row_id_meta` would default to `None` regardless of the
                            // feature flag), so `row_id_meta.is_some()` is equivalent to
                            // the manifest's stable-row-ids flag being set. This is an
                            // inference from fragment metadata rather than reading the
                            // flag directly, and would silently stop detecting the
                            // conflict below if that equivalence ever broke.
                            let uses_stable_row_ids = groups
                                .iter()
                                .flat_map(|g| &g.old_fragments)
                                .any(|f| f.row_id_meta.is_some());
                            if uses_stable_row_ids
                                && new_indices.iter().any(corrupted_by_frag_reuse)
                            {
                                return Err(
                                    self.retryable_conflict_err(other_transaction, other_version)
                                );
                            }
                            for index in new_indices {
                                let Some(frag_bitmap) = &index.fragment_bitmap else {
                                    return Err(self
                                        .retryable_conflict_err(other_transaction, other_version));
                                };
                                for group in groups {
                                    let mut indexed = 0usize;
                                    let mut unindexed = 0usize;
                                    for frag in &group.old_fragments {
                                        if frag_bitmap.contains(frag.id as u32) {
                                            indexed += 1;
                                        } else {
                                            unindexed += 1;
                                        }
                                    }
                                    if indexed > 0 && unindexed > 0 {
                                        return Err(self.retryable_conflict_err(
                                            other_transaction,
                                            other_version,
                                        ));
                                    }
                                }
                            }
                            Ok(())
                        }
                        // Rewrite with remapping and frag_reuse_index creation can commit without conflict
                        (Some(_), false) => {
                            // As above, a replacement mixed with other indices is
                            // refused rather than reasoned about.
                            if new_indices.len() != 1 || removed_indices.len() != 1 {
                                Err(self
                                    .incompatible_conflict_err(other_transaction, other_version))
                            } else {
                                Ok(())
                            }
                        }
                        // Rewrite with remapping will conflict with
                        // index creation that touches overlapping fragments.
                        (_, false) => {
                            let mut affected_ids = HashSet::new();
                            for index in new_indices {
                                if let Some(frag_bitmap) = &index.fragment_bitmap {
                                    affected_ids.extend(frag_bitmap.iter());
                                } else {
                                    return Err(self
                                        .retryable_conflict_err(other_transaction, other_version));
                                }
                            }
                            if groups
                                .iter()
                                .flat_map(|f| f.old_fragments.iter().map(|f| f.id))
                                .any(|id| affected_ids.contains(&(id as u32)))
                            {
                                Err(self.retryable_conflict_err(other_transaction, other_version))
                            } else {
                                Ok(())
                            }
                        }
                    }
                }
                Operation::Overwrite { .. } | Operation::Restore { .. } => {
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
}
