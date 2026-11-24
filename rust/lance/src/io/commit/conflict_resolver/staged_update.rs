// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::provenance::revives_rows;
use arrow_schema::DataType;
use lance_core::datatypes::Schema;
use lance_select::RowAddrSelection;
use lance_table::format::key_existence::{FilterType, KeyExistenceFilter};

impl TransactionRebase<'_> {
    /// Physical row addresses describe a row rewrite only while the retained
    /// fragments still refer to the data the writer read.
    pub(in crate::io::commit) fn validate_row_rewrite(&mut self) {
        if self.affected_rows.is_some()
            && let Operation::Update {
                update_mode: Some(UpdateMode::RewriteRows),
                updated_fragments,
                ..
            } = &self.transaction.operation
            && !updated_fragments.iter().all(|written| {
                self.initial_fragments
                    .get(&written.id)
                    .is_some_and(|(read, _)| same_row_layout(read, written))
            })
        {
            self.affected_rows = None;
        }
    }

    /// A later mask can hide a concurrent delete followed by an undelete.
    /// Intersect superseded masks with our write set, and keep rows absent at
    /// the read version from reappearing under a staged insert or fragment removal.
    pub(in crate::io::commit) async fn check_row_history(
        &self,
        read_snapshot: &Dataset,
        dataset: &Dataset,
        window: &[(u64, Arc<Transaction>)],
    ) -> Result<()> {
        let key_filter = match &self.transaction.operation {
            Operation::Update {
                inserted_rows_filter: Some(keys),
                ..
            } => Some(keys),
            _ => None,
        };
        if key_filter.is_none()
            && (self.affected_rows.is_none()
                || !self.initial_fragments.values().any(|(_, changed)| *changed))
        {
            return Ok(());
        }
        let mut latest = HashMap::new();
        let parallelism = dataset.table_state.object_store.io_parallelism();
        let mut earlier = futures::stream::FuturesUnordered::new();
        for (version, concurrent) in window {
            if let Some(filter) = key_filter
                && changes_key_domain(&concurrent.operation, filter, read_snapshot.schema())?
            {
                return Err(self.retryable_conflict_err(concurrent, *version));
            }
            let (Operation::Update {
                updated_fragments, ..
            }
            | Operation::Delete {
                updated_fragments, ..
            }) = &concurrent.operation
            else {
                continue;
            };
            for next in updated_fragments {
                // An inserted key was absent from the entire read snapshot.
                // Restoring a row in another fragment can introduce that key.
                if !matches!(self.initial_fragments.get(&next.id), Some((_, true)))
                    && (key_filter.is_none()
                        || read_snapshot
                            .find_fragment(next.id)
                            .is_some_and(|read| read.deletion_file.is_none()))
                {
                    continue;
                }
                if let Some((last, previous_tx, previous_version)) =
                    latest.insert(next.id, (next, concurrent, *version))
                    && last.deletion_file != next.deletion_file
                    && let Some(file) = &last.deletion_file
                    && let Some(moved) = self
                        .affected_rows
                        .and_then(|rows| rows.get(&(next.id as u32)))
                {
                    earlier.push(async move {
                        let deleted = read_dataset_deletion_file(dataset, next.id, file).await?;
                        if overlaps_deletion(moved, &deleted) {
                            return Err(self.retryable_conflict_err(previous_tx, previous_version));
                        }
                        Ok(())
                    });
                    if earlier.len() >= parallelism {
                        earlier.try_next().await?;
                    }
                }
            }
        }
        while earlier.try_next().await?.is_some() {}
        futures::stream::iter(latest)
            .map(Ok)
            .try_for_each_concurrent(parallelism, |(id, (last, tx, version))| async move {
                let Some(read) = read_snapshot.find_fragment(id) else {
                    return Err(self.retryable_conflict_err(tx, version));
                };
                if revives_rows(read_snapshot, dataset, read, last).await? {
                    return Err(self.retryable_conflict_err(tx, version));
                }
                Ok(())
            })
            .await
    }

    /// A lost publication may have promoted a partial deletion to a whole
    /// fragment removal. Recompute those removals when the fragment changed
    /// again, preserving every removal requested by the original writer.
    pub(in crate::io::commit) fn recheck_fragment_removals(&mut self, submitted: &Transaction) {
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
) -> Result<bool> {
    let keys = &filter.field_ids;
    let affects_key = |field: &i32| *field >= 0 && field_affects_index(*field, keys, read);
    match operation {
        Operation::Update {
            fields_modified,
            inserted_rows_filter: Some(other),
            ..
        } => Ok(fields_modified
            .iter()
            .any(|&field| affects_key(&(field as i32)))
            || !key_filters_cover_inserts(read, filter, other)),
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
