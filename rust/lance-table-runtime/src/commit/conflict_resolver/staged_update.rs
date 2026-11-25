// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::io::deletion::deletion_mask_shrank;
use arrow_schema::DataType;
use lance_core::datatypes::Schema;
use lance_select::RowAddrSelection;
use lance_table::format::DeletionFile;
use lance_table::format::key_existence::{FilterType, KeyExistenceFilter};
use std::collections::hash_map::Entry;

#[derive(Debug)]
pub(super) struct RowHistory {
    read: Table,
    inserted_row_count: Option<u64>,
    latest: HashMap<u64, (u64, Option<DeletionFile>)>,
    // First mutation of fragments absent at the staged read version.
    first_new_changes: HashMap<u64, u64>,
    superseded: Vec<(u64, u64, Option<DeletionFile>)>,
    check_intermediate_keys: Option<bool>,
    // Keep the existing precedence of incompatible transaction pairs.
    key_error: Option<Error>,
}

impl RowHistory {
    pub(super) fn new(read: &Table, inserted_row_count: Option<u64>) -> Box<Self> {
        Box::new(Self {
            read: read.clone(),
            inserted_row_count,
            latest: HashMap::new(),
            first_new_changes: HashMap::new(),
            superseded: Vec::new(),
            check_intermediate_keys: None,
            key_error: None,
        })
    }
}

pub(super) fn needs_row_history(
    transaction: &Transaction,
    affected: Option<&RowAddrTreeMap>,
) -> bool {
    matches!(
        transaction.operation,
        Operation::Update {
            inserted_rows_filter: Some(_),
            ..
        }
    ) || (affected.is_some()
        && matches!(
            transaction.operation,
            Operation::Update { .. } | Operation::Delete { .. }
        ))
}

impl TransactionRebase<'_> {
    /// Physical row addresses describe a row rewrite only while the retained
    /// fragments still refer to the data the writer read.
    pub(super) fn validate_row_rewrite(&mut self, inserted_row_count: Option<u64>) {
        let Operation::Update {
            update_mode: Some(UpdateMode::RewriteRows) | None,
            updated_fragments,
            inserted_rows_filter,
            ..
        } = &self.transaction.operation
        else {
            return;
        };
        // A matched delete plus an insert can have the same physical write set
        // and output size as a pure update. Only a known zero or a key filter
        // makes the matched-row addresses sufficient for a row-level rebase.
        let untracked_inserts = inserted_rows_filter.is_none() && inserted_row_count != Some(0);
        if self.affected_rows.is_some()
            && (untracked_inserts
                || !updated_fragments.iter().all(|written| {
                    self.initial_fragments
                        .get(&written.id)
                        .is_some_and(|(read, _)| same_row_layout(read, written))
                }))
        {
            self.affected_rows = None;
        }
    }

    /// Keep only deletion descriptors while the caller still owns the transaction.
    /// Inline bitmaps share their bytes with the manifest until the mask check finishes.
    pub(super) fn record_row_changes(&mut self, concurrent: &Transaction, version: u64) {
        let Some(history) = self.row_history.as_mut() else {
            return;
        };
        if history.key_error.is_some() {
            return;
        }
        let keys = match &self.transaction.operation {
            Operation::Update {
                inserted_rows_filter,
                ..
            } => inserted_rows_filter.as_ref(),
            _ => None,
        };
        if let Some(keys) = keys {
            history.key_error = match changes_key_domain(
                &concurrent.operation,
                keys,
                history.read.schema(),
                history.inserted_row_count,
            ) {
                Ok(false) => None,
                Ok(true) => Some(Error::retryable_commit_conflict_source(
                    version,
                    format!(
                        "Concurrent {} at version {version} may introduce a key inserted by {}",
                        concurrent.operation, self.transaction.uuid
                    )
                    .into(),
                )),
                Err(error) => Some(error),
            };
            if history.key_error.is_some() {
                return;
            }
        }
        let (Operation::Update {
            updated_fragments, ..
        }
        | Operation::Delete {
            updated_fragments, ..
        }) = &concurrent.operation
        else {
            return;
        };
        for fragment in updated_fragments {
            let id = fragment.id;
            // A concurrent restore can introduce an inserted key in any fragment.
            // Standalone callers can check the newest version first. A mask
            // equal to the read snapshot may have restored an earlier deletion.
            let touched = self.affected_rows.is_some() && self.initial_fragments.contains_key(&id);
            let may_revive_key = keys.is_some()
                && match history.read.view().find_fragment(id) {
                    Some(read) => read.deletion_file.is_some(),
                    None if history.inserted_row_count == Some(0) => false,
                    None => {
                        history
                            .first_new_changes
                            .entry(id)
                            .and_modify(|first| *first = (*first).min(version))
                            .or_insert(version);
                        true
                    }
                };
            if !touched && !may_revive_key {
                continue;
            }
            let next = (version, fragment.deletion_file.clone());
            let last = match history.latest.entry(id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(next);
                    continue;
                }
            };
            let (earlier_version, earlier_file) = if version > last.0 {
                std::mem::replace(last, next)
            } else {
                next
            };
            if earlier_file == last.1 {
                continue;
            }
            let check_keys = may_revive_key
                && keys.is_some_and(|filter| {
                    *history.check_intermediate_keys.get_or_insert_with(|| {
                        history.inserted_row_count != Some(0)
                            && !proves_no_inserts(history.read.schema(), filter)
                    })
                });
            let may_delete_row = earlier_file.is_some()
                && self
                    .affected_rows
                    .is_some_and(|rows| rows.get(&(id as u32)).is_some());
            if check_keys || may_delete_row {
                history.superseded.push((id, earlier_version, earlier_file));
            }
        }
    }

    /// Check superseded masks as well as the last mask: a restore can hide a
    /// conflicting deletion, or make a key absent at staging present again.
    pub(super) async fn check_row_history(&mut self, dataset: &Table) -> Result<()> {
        let Some(mut history) = self.row_history.take() else {
            return Ok(());
        };
        if let Some(error) = history.key_error.take() {
            return Err(error);
        }
        let parallelism = dataset.state.object_store.io_parallelism();
        let conflict = |id, version| {
            Error::retryable_commit_conflict_source(
                version,
                format!(
                    "Concurrent changes to fragment {id} at version {version} invalidate staged rows or keys of {}",
                    self.transaction.uuid
                ).into(),
            )
        };
        let read = &history.read;
        let rows = self.affected_rows;
        let check_keys = history.check_intermediate_keys.unwrap_or(false);
        let first_changes = &history.first_new_changes;
        let superseded = history
            .superseded
            .into_iter()
            .map(|(id, version, file)| (id, version, file, false));
        let latest = history
            .latest
            .into_iter()
            .map(|(id, (version, file))| (id, version, file, true));
        let check_mask = async |baseline: &Table, change| {
            let (id, version, file, latest): (u64, u64, Option<DeletionFile>, bool) = change;
            if !latest
                && let Some(moved) = rows.and_then(|rows| rows.get(&(id as u32)))
                && let Some(mask) = &file
            {
                let deleted = read_table_deletion_file(dataset, id, mask).await?;
                if overlaps_deletion(moved, &deleted) {
                    return Err(conflict(id, version));
                }
            }
            // A key can reappear and move to another fragment before the
            // final mask hides its original offset again.
            if !latest && !check_keys {
                return Ok(());
            }
            // For a fragment introduced after staging, compare against the
            // actual mask preceding this change. A delete from newly
            // written rows only shrinks the visible key domain.
            let original = baseline
                .view()
                .find_fragment(id)
                .ok_or_else(|| conflict(id, version))?;
            if deletion_mask_shrank(
                original.deletion_file.as_ref(),
                file.as_ref(),
                |file| read_table_deletion_file(baseline, id, file),
                |file| read_table_deletion_file(dataset, id, file),
            )
            .await?
            {
                return Err(conflict(id, version));
            }
            Ok(())
        };
        let mut new_changes = HashMap::<_, Vec<_>>::new();
        let read_changes = superseded.chain(latest).filter_map(|change| {
            if let Some(&version) = first_changes.get(&change.0) {
                new_changes.entry(version).or_default().push(change);
                None
            } else {
                Some(Ok(change))
            }
        });
        futures::stream::iter(read_changes)
            .try_for_each_concurrent(parallelism, |change| check_mask(read, change))
            .await?;
        // Check fragments first changed in the same commit together. Release
        // each historical manifest before loading the next one.
        for (version, changes) in new_changes {
            let before = version.checked_sub(1).ok_or_else(|| {
                Error::invalid_input("A committed fragment change must have a positive version")
            })?;
            let previous = dataset.checkout_version(before).await?;
            futures::stream::iter(changes.into_iter().map(Ok))
                .try_for_each_concurrent(parallelism, |change| check_mask(&previous, change))
                .await?;
        }
        Ok(())
    }

    /// A lost publication may have promoted a partial deletion to a whole
    /// fragment removal. Recompute those removals when the fragment changed
    /// again, preserving every removal requested by the original writer.
    pub(in crate::commit) fn recheck_fragment_removals(&mut self, submitted: &Transaction) {
        let ((
            Operation::Update {
                removed_fragment_ids: attempt,
                ..
            },
            Operation::Update {
                removed_fragment_ids: original,
                ..
            },
        )
        | (
            Operation::Delete {
                deleted_fragment_ids: attempt,
                ..
            },
            Operation::Delete {
                deleted_fragment_ids: original,
                ..
            },
        )) = (&mut self.transaction.operation, &submitted.operation)
        else {
            return;
        };
        if attempt.len() > original.len() {
            let original: HashSet<_> = original.iter().copied().collect();
            attempt.retain(|id| {
                original.contains(id) || !matches!(self.initial_fragments.get(id), Some((_, true)))
            });
        }
    }
}

// Inserted-key filters describe new rows. An in-place change can introduce the
// same key without inserting any row, even outside the fragments we rewrote.
fn changes_key_domain(
    operation: &Operation,
    filter: &KeyExistenceFilter,
    read: &Schema,
    inserted_row_count: Option<u64>,
) -> Result<bool> {
    let keys = &filter.field_ids;
    let affects_key = |field: &i32| *field >= 0 && field_affects_index(*field, keys, read);
    match operation {
        Operation::Update {
            fields_modified,
            inserted_rows_filter: Some(other),
            new_fragments,
            ..
        } => Ok(fields_modified
            .iter()
            .any(|&field| affects_key(&(field as i32)))
            || (inserted_row_count != Some(0)
                && !new_fragments.is_empty()
                && !key_filters_cover_inserts(read, filter, other))),
        Operation::DataOverlay { groups } => {
            for overlay in groups.iter().flat_map(|group| &group.overlays) {
                for (pos, field) in overlay.data_file.fields.iter().enumerate() {
                    if affects_key(field) && !overlay.coverage_for_field(pos)?.is_empty() {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
        Operation::DataReplacement {
            data_change: true,
            replacements,
        } => Ok(replacements
            .iter()
            .any(|r| r.1.fields.iter().any(affects_key))),
        Operation::Project { schema, .. } => {
            let key_ids_changed = read
                .unenforced_primary_key()
                .iter()
                .map(|f| f.id)
                .ne(schema.unenforced_primary_key().iter().map(|f| f.id));
            Ok(key_ids_changed
                || keys
                    .iter()
                    .any(|&id| match (read.field_by_id(id), schema.field_by_id(id)) {
                        (Some(before), Some(after)) => before.data_type() != after.data_type(),
                        _ => true,
                    }))
        }
        _ => Ok(false),
    }
}

// Writers silently omitted keys they could not encode. Both filters must
// represent the primary key before a negative intersection proves disjointness.
fn key_filters_cover_inserts(
    schema: &Schema,
    first: &KeyExistenceFilter,
    second: &KeyExistenceFilter,
) -> bool {
    let records_keys = |filter: &KeyExistenceFilter| match &filter.filter {
        FilterType::ExactSet(set) => !set.is_empty(),
        FilterType::Bloom { bitmap, .. } => bitmap.iter().any(|&byte| byte != 0),
    };
    !first.field_ids.is_empty()
        && first.field_ids.iter().all(|id| {
            let Some(field) = schema
                .fields
                .iter()
                .find(|field| field.id == *id && !field.nullable)
            else {
                return false;
            };
            match field.data_type() {
                DataType::Int32
                | DataType::Int64
                | DataType::UInt32
                | DataType::UInt64
                | DataType::LargeUtf8 => true,
                // JSON text and stored JSONB can encode one logical key with
                // different filter hashes. Only plain LargeBinary is bytewise.
                DataType::LargeBinary => field.logical_type.as_str() == "large_binary",
                // View arrays can populate ordinary string/binary columns without
                // being encoded in the filter. Older writers also omitted fixed
                // binary. Each batch in a merge stream has the same key types.
                DataType::Utf8 | DataType::Binary | DataType::FixedSizeBinary(_) => {
                    records_keys(first) && records_keys(second)
                }
                _ => false,
            }
        })
}

// Older protobufs can decode a missing filter payload as an empty exact set.
// Only a complete writer Bloom filter on a fully encodable key proves absence.
fn proves_no_inserts(schema: &Schema, filter: &KeyExistenceFilter) -> bool {
    let FilterType::Bloom {
        bitmap, num_bits, ..
    } = &filter.filter
    else {
        return false;
    };
    bitmap.iter().all(|&byte| byte == 0)
        && u64::from(*num_bits) == bitmap.len() as u64 * 8
        && filter.validate().is_ok()
        && schema
            .unenforced_primary_key()
            .iter()
            .map(|f| f.id)
            .eq(filter.field_ids.iter().copied())
        && key_filters_cover_inserts(schema, filter, filter)
}

fn overlaps_deletion(moved: &RowAddrSelection, deleted: &DeletionVector) -> bool {
    match (moved, deleted) {
        (RowAddrSelection::Full, _) => !deleted.is_empty(),
        (RowAddrSelection::Partial(rows), DeletionVector::Bitmap(mask)) => !rows.is_disjoint(mask),
        (RowAddrSelection::Partial(rows), _) if rows.len() < deleted.len() as u64 => {
            rows.iter().any(|row| deleted.contains(row))
        }
        (RowAddrSelection::Partial(rows), _) => deleted.iter().any(|row| rows.contains(row)),
    }
}

// Manifest assembly carries current overlays, clustering and generic metadata
// forward. None of them changes the physical offsets in a staged row set.
pub(super) fn same_row_layout(read: &Fragment, written: &Fragment) -> bool {
    let Fragment {
        id,
        files,
        overlays: _,
        deletion_file: _,
        row_id_meta,
        physical_rows,
        last_updated_at_version_meta,
        created_at_version_meta,
        clustering_id: _,
        metadata: _,
    } = read;
    id == &written.id
        && files == &written.files
        && row_id_meta == &written.row_id_meta
        && physical_rows == &written.physical_rows
        && last_updated_at_version_meta == &written.last_updated_at_version_meta
        && created_at_version_meta == &written.created_at_version_meta
}
