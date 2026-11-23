// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::finish::read_fragment_deletion_bitmap;
use super::*;

impl TransactionRebase<'_> {
    pub(in crate::io::commit) fn defer_staged_update_rows(&mut self) {
        if self.affected_rows.is_none()
            && matches!(&self.transaction.operation, Operation::Update {
                update_mode: Some(UpdateMode::RewriteRows),
                fields_modified, updated_fragment_offsets: None, updated_fragments, ..
            } if fields_modified.is_empty() && !updated_fragments.is_empty())
        {
            self.affected_rows = Some(UpdateRows::Deferred);
        }
    }

    /// A staged row rewrite carries its deletion files across processes. Recover its
    /// physical offsets only when a named update or delete touched the same fragment.
    pub(in crate::io::commit) async fn recover_staged_update_rows(
        &mut self,
        dataset: &Dataset,
        window: &[(u64, Arc<Transaction>)],
    ) -> Result<()> {
        if !matches!(self.affected_rows, Some(UpdateRows::Deferred))
            || !self.initial_fragments.values().any(|(_, flagged)| *flagged)
        {
            return Ok(());
        }
        let rows = self.staged_deletion_offsets(dataset, window).await?.ok_or_else(|| {
            Error::retryable_commit_conflict_source(dataset.manifest.version, format!(
                "Cannot establish which rows in staged update {} changed across concurrent commits. Retry on the latest version",
                self.transaction.uuid
            ).into())
        })?;
        self.affected_rows = Some(UpdateRows::DeletionOverlaps(rows));
        Ok(())
    }

    async fn staged_deletion_offsets(
        &self,
        dataset: &Dataset,
        window: &[(u64, Arc<Transaction>)],
    ) -> Result<Option<RowAddrTreeMap>> {
        let Operation::Update {
            updated_fragments: staged,
            removed_fragment_ids: removed,
            ..
        } = &self.transaction.operation
        else {
            return Err(wrong_operation_err(&self.transaction.operation));
        };

        let mut previous: HashMap<u64, &Fragment> = HashMap::new();
        for (_, concurrent) in window {
            let (Operation::Update {
                updated_fragments, ..
            }
            | Operation::Delete {
                updated_fragments, ..
            }) = &concurrent.operation
            else {
                continue;
            };
            for next in updated_fragments
                .iter()
                .filter(|f| self.modified_fragment_ids.contains(&f.id))
            {
                let Some((read, _)) = self.initial_fragments.get(&next.id) else {
                    return Ok(None);
                };
                let last = previous.get(&next.id).copied().unwrap_or(read);
                if !same_fragment_except_deletions(read, next) {
                    return Ok(None);
                }
                // Checking each transition catches an undelete even when a later writer
                // fills the deletion file again before the staged commit runs.
                if last.deletion_file != next.deletion_file {
                    let (before, after) = futures::try_join!(
                        read_fragment_deletion_bitmap(dataset, last),
                        read_fragment_deletion_bitmap(dataset, next),
                    )?;
                    if !preserves_deletions(&before, &after, read.physical_rows) {
                        return Ok(None);
                    }
                }
                previous.insert(next.id, next);
            }
        }
        if previous.is_empty() {
            return Ok(None);
        }
        if removed.iter().any(|id| previous.contains_key(id)) {
            return Ok(None);
        }
        let mut affected = RowAddrTreeMap::new();
        let mut checked = HashSet::new();
        for written in staged {
            let Some(last) = previous.get(&written.id) else {
                continue;
            };
            let (read, _) = &self.initial_fragments[&written.id];
            if !checked.insert(written.id)
                || last.deletion_file.is_none()
                || dataset.find_fragment(written.id) != Some(*last)
                || !same_fragment_except_deletions(read, written)
            {
                return Ok(None);
            }
            let (before, after) = futures::try_join!(
                read_fragment_deletion_bitmap(dataset, read),
                read_fragment_deletion_bitmap(dataset, written),
            )?;
            if before == after || !preserves_deletions(&before, &after, read.physical_rows) {
                return Ok(None);
            }
            let Ok(id) = u32::try_from(written.id) else {
                return Ok(None);
            };
            affected.insert_bitmap(id, after - before);
        }
        // A whole-fragment removal has no retained post-image to compare.
        Ok((checked.len() == previous.len()).then_some(affected))
    }
}

fn preserves_deletions(
    before: &RoaringBitmap,
    after: &RoaringBitmap,
    physical_rows: Option<usize>,
) -> bool {
    physical_rows.is_some_and(|rows| after.max().is_none_or(|offset| (offset as usize) < rows))
        && before.is_subset(after)
}

fn same_fragment_except_deletions(read: &Fragment, written: &Fragment) -> bool {
    let Fragment {
        id,
        files,
        overlays,
        deletion_file: _,
        row_id_meta,
        physical_rows,
        last_updated_at_version_meta,
        created_at_version_meta,
        clustering_id,
        metadata,
    } = read;
    id == &written.id
        && files == &written.files
        && overlays == &written.overlays
        && row_id_meta == &written.row_id_meta
        && physical_rows == &written.physical_rows
        && last_updated_at_version_meta == &written.last_updated_at_version_meta
        && created_at_version_meta == &written.created_at_version_meta
        && clustering_id == &written.clustering_id
        && metadata == &written.metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::unchanged(&[1], &[1], Some(3), true)]
    #[case::added_last_row(&[1], &[1, 2], Some(3), true)]
    #[case::revived(&[1], &[2], Some(3), false)]
    #[case::past_end(&[1], &[1, 3], Some(3), false)]
    #[case::unknown_size(&[], &[0], None, false)]
    fn test_staged_merge_row_conflicts_deletion_bounds(
        #[case] before: &[u32],
        #[case] after: &[u32],
        #[case] rows: Option<usize>,
        #[case] expected: bool,
    ) {
        let before = before.iter().copied().collect();
        let after = after.iter().copied().collect();
        assert_eq!(preserves_deletions(&before, &after, rows), expected);
    }
}
