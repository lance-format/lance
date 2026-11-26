// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integrity checks for the stable row id invariants that the row id index and the write
//! paths rely on.

use super::{RowVersionKind, load_row_id_sequence, load_row_version_sequence};
use crate::Table;
use crate::fragment::FileFragment;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_table::rowids::RowIdSequence;
use roaring::RoaringTreemap;
use std::sync::Arc;

/// The per-fragment state a stable row id invariant is checked against.
type FragmentRowIds = (
    FileFragment,
    Arc<RowIdSequence>,
    Option<Arc<DeletionVector>>,
);

/// Check the invariants that the stable row id machinery relies on.
///
/// A no-op on tables that use physical row addresses as row ids.
pub async fn validate_stable_row_ids(table: &Table) -> Result<()> {
    if !table.manifest.uses_stable_row_ids() {
        return Ok(());
    }

    let corrupt = |message: String| Error::corrupt_file(table.state.base.clone(), message);

    for fragment in table.manifest.fragments.iter() {
        if fragment.row_id_meta.is_none() {
            return Err(corrupt(format!(
                "Fragment {} has no row id metadata, but dataset {:?} uses stable row ids",
                fragment.id, table.state.base
            )));
        }
    }

    // `buffered` preserves manifest order, which the uniqueness check below relies on to
    // report the earlier of the two fragments holding a duplicate id.
    let fragments: Vec<FragmentRowIds> = futures::stream::iter(table.get_fragments())
        .map(|fragment| async move {
            let sequence = load_row_id_sequence(table, fragment.metadata()).await?;
            let deletion_vector = fragment.get_deletion_vector().await?;
            Result::Ok((fragment, sequence, deletion_vector))
        })
        .buffered(table.state.object_store.io_parallelism())
        .try_collect()
        .await?;

    let mut live_row_ids = RoaringTreemap::new();
    let mut max_row_id: Option<u64> = None;

    for (index, (fragment, sequence, deletion_vector)) in fragments.iter().enumerate() {
        let metadata = fragment.metadata();
        let physical_rows = metadata.physical_rows.ok_or_else(|| {
            corrupt(format!(
                "Fragment {} has an unknown physical row count, but dataset {:?} uses stable row ids",
                metadata.id, table.state.base
            ))
        })? as u64;

        if sequence.len() != physical_rows {
            return Err(corrupt(format!(
                "Fragment {} has {} row ids, but {} physical rows, in dataset {:?}",
                metadata.id,
                sequence.len(),
                physical_rows,
                table.state.base
            )));
        }

        for (name, kind) in [
            ("created_at", RowVersionKind::CreatedAt),
            ("last_updated_at", RowVersionKind::LastUpdatedAt),
        ] {
            // The external form is a valid encoding nothing writes and nothing
            // can read yet, so it is the one placement not checked here.
            if let Some(versions) = load_row_version_sequence(table, metadata, kind)
                .await
                .or_else(|error| match error {
                    Error::NotSupported { .. } => Ok(None),
                    other => Err(other),
                })?
            {
                let versions = versions.len();
                if versions != physical_rows {
                    return Err(corrupt(format!(
                        "Fragment {} has {} {} versions, but {} physical rows, in dataset {:?}",
                        metadata.id, versions, name, physical_rows, table.state.base
                    )));
                }
            }
        }

        // The bounding range covers tombstoned slots too: their ids are retired and must
        // not be handed out again either.
        if let Some(range) = sequence.row_id_range() {
            max_row_id = max_row_id.max(Some(*range.end()));
        }

        for (offset, row_id) in sequence.iter().enumerate() {
            // An update rewrites a row into a new fragment under the same id and tombstones
            // the original slot, so only live ids are required to be unique.
            if deletion_vector
                .as_ref()
                .is_some_and(|deletions| deletions.contains(offset as u32))
            {
                continue;
            }
            if !live_row_ids.insert(row_id) {
                return Err(corrupt(format!(
                    "Row id {} is live in both {} and fragment {} at offset {} in dataset {:?}",
                    row_id,
                    describe_first_live_slot(&fragments[..=index], row_id),
                    metadata.id,
                    offset,
                    table.state.base
                )));
            }
        }
    }

    if let Some(max_row_id) = max_row_id
        && table.manifest.next_row_id <= max_row_id
    {
        return Err(corrupt(format!(
            "Dataset {:?} will hand out row id {} next, but row id {} is already in use",
            table.state.base, table.manifest.next_row_id, max_row_id
        )));
    }

    Ok(())
}

/// Describe where `row_id` is first live, so a uniqueness failure can name both of the
/// slots involved. Only runs on the failure path, hence the linear rescan.
fn describe_first_live_slot(fragments: &[FragmentRowIds], row_id: u64) -> String {
    for (fragment, sequence, deletion_vector) in fragments {
        // A bounding range that excludes the id rules the whole fragment out.
        if sequence
            .row_id_range()
            .is_none_or(|range| !range.contains(&row_id))
        {
            continue;
        }
        for (offset, candidate) in sequence.iter().enumerate() {
            if candidate == row_id
                && !deletion_vector
                    .as_ref()
                    .is_some_and(|deletions| deletions.contains(offset as u32))
            {
                return format!("fragment {} at offset {}", fragment.id(), offset);
            }
        }
    }
    "an earlier fragment".to_string()
}
