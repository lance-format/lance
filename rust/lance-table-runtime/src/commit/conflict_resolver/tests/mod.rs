// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZero;

use arrow_schema::{DataType, Field, Schema};
use lance_file::version::LanceFileVersion;
use lance_io::assert_io_eq;
use lance_table::format::{DataFile, InlineRowIds, RowIdMeta};
use lance_table::io::deletion::read_deletion_file;
use lance_table::transaction::UpdateMode::{RewriteColumns, RewriteRows};
use lance_table::transaction::{DataReplacementGroup, UpdateMap};
use uuid::Uuid;

use super::*;
use crate::caches::DeletionFileKey;
use crate::commit::CommitBuilder;
use crate::test_util::test_commit_table as test_dataset;

mod conflicts;
mod index_and_config;
mod rebase;

/// Helper function for tests to create UpdateConfig operations using old-style parameters
#[cfg(test)]
fn create_update_config_for_test(
    upsert_values: Option<HashMap<String, String>>,
    delete_keys: Option<Vec<String>>,
    schema_metadata: Option<HashMap<String, String>>,
    field_metadata: Option<HashMap<u32, HashMap<String, String>>>,
) -> Operation {
    use lance_table::transaction::{translate_config_updates, translate_schema_metadata_updates};

    let config_updates = if let Some(upsert) = &upsert_values {
        if let Some(delete) = &delete_keys {
            Some(translate_config_updates(upsert, delete))
        } else {
            Some(translate_config_updates(upsert, &[]))
        }
    } else {
        delete_keys
            .as_ref()
            .map(|delete| translate_config_updates(&HashMap::new(), delete))
    };

    let schema_metadata_updates = schema_metadata
        .as_ref()
        .map(translate_schema_metadata_updates);

    let field_metadata_updates = field_metadata
        .unwrap_or_default()
        .into_iter()
        .map(|(field_id, metadata)| {
            (
                field_id as i32,
                translate_schema_metadata_updates(&metadata),
            )
        })
        .collect();

    Operation::UpdateConfig {
        config_updates,
        table_metadata_updates: None,
        schema_metadata_updates,
        field_metadata_updates,
    }
}

async fn apply_deletion(delete_rows: &[u32], fragment: &mut Fragment, dataset: &Table) -> Fragment {
    let mut current_deletions = if let Some(deletion_file) = &fragment.deletion_file {
        read_deletion_file(
            fragment.id,
            deletion_file,
            // Reference deletion file should never enter this apply_deletion. So base path is fine.
            &dataset.state.base,
            dataset.state.object_store.as_ref(),
        )
        .await
        .unwrap()
    } else {
        DeletionVector::default()
    };

    current_deletions.extend(delete_rows.iter().copied());

    fragment.deletion_file = write_deletion_file(
        &dataset.state.base,
        fragment.id,
        dataset.manifest.version,
        &current_deletions,
        dataset.state.object_store.as_ref(),
        0,
    )
    .await
    .unwrap();

    let deletion_file = fragment.deletion_file.as_ref().unwrap();
    let key = DeletionFileKey {
        fragment_id: fragment.id,
        deletion_file,
    };
    dataset
        .state
        .metadata_cache
        .insert_with_key(&key, Arc::new(current_deletions))
        .await;

    fragment.clone()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConflictResult {
    Compatible,
    NotCompatible,
    Retryable,
}

/// Returns the IDs of fragments that have been modified by this operation.
///
/// This does not include new fragments.
fn modified_fragment_ids(operation: &Operation) -> Box<dyn Iterator<Item = u64> + '_> {
    match operation {
        // These operations add new fragments or don't modify any.
        Operation::Append { .. }
        | Operation::Clone { .. }
        | Operation::Overwrite { .. }
        | Operation::CreateIndex { .. }
        | Operation::ReserveFragments { .. }
        | Operation::Project { .. }
        | Operation::UpdateConfig { .. }
        | Operation::UpdateBases { .. }
        | Operation::Restore { .. }
        | Operation::UpdateMemWalState { .. }
        | Operation::CompositeOperation(_) => Box::new(std::iter::empty()),
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            ..
        } => Box::new(
            updated_fragments
                .iter()
                .map(|f| f.id)
                .chain(deleted_fragment_ids.iter().copied()),
        ),
        Operation::Rewrite { groups, .. } => Box::new(
            groups
                .iter()
                .flat_map(|f| f.old_fragments.iter().map(|f| f.id)),
        ),
        Operation::Merge { fragments, .. } => Box::new(fragments.iter().map(|f| f.id)),
        Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            ..
        } => Box::new(
            updated_fragments
                .iter()
                .map(|f| f.id)
                .chain(removed_fragment_ids.iter().copied()),
        ),
        Operation::DataReplacement { replacements, .. } => {
            Box::new(replacements.iter().map(|r| r.0))
        }
        Operation::DataOverlay { groups } => Box::new(groups.iter().map(|g| g.fragment_id)),
        Operation::Unknown { .. } => unimplemented!("modified_fragment_ids for {operation}"),
    }
}
