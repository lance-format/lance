// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::datatypes::Schema;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::{CoveringContract, ensure_can_write_manifest};
use lance_table::format::{
    IndexMetadata, Manifest, ManifestBuildConfig, operation_may_change_schema, pb,
};
use lance_table::io::commit::{CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme};
use lance_table::transaction::{Operation, Transaction, validate_operation};
use object_store::path::Path;
use prost::Message;

use crate::caches::DSMetadataCache;
use crate::{Table, TableSession};

use super::CommitHooks;
use super::migration::fix_schema;
use super::publish::{
    CommitOutcome, cleanup_spilled_clone_files, cleanup_transaction_file, verify_commit_outcome,
    write_manifest_file, write_transaction_file,
};

/// Give a new table its field IDs. `inherited` IDs are an existing table's,
/// such as a MemWAL generation's, and may exceed the new-table limit.
pub fn assign_new_table_field_ids(schema: &mut Schema, inherited: bool) -> Result<()> {
    if inherited {
        schema.try_keep_field_ids()
    } else {
        schema.try_assign_new_table_field_ids()
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn commit_new_dataset(
    object_store: &ObjectStore,
    source_store: Option<&ObjectStore>,
    commit_handler: &Arc<dyn CommitHandler>,
    base_path: &Path,
    uri: &str,
    transaction: &Transaction,
    build_config: &ManifestBuildConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    metadata_cache: &DSMetadataCache,
    session: Arc<TableSession>,
    inherited_field_ids: bool,
    hooks: CommitHooks<'_>,
) -> Result<(Manifest, ManifestLocation)> {
    let build_config = build_config
        .clone()
        .with_timestamp_nanos(hooks.timestamp_nanos())
        .with_index_is_supported(super::index::index_usability(hooks.index));
    let mut transaction = transaction.clone();
    if let Operation::Overwrite { schema, .. } = &mut transaction.operation {
        assign_new_table_field_ids(schema, inherited_field_ids)?;
        schema.verify_primary_key()?;
    }
    let transaction = &transaction;
    validate_operation(None, &transaction.operation)?;
    let pb_transaction = pb::Transaction::try_from(transaction)?;
    let inline_transaction = hooks.inlines_transaction(
        pb_transaction.encoded_len(),
        build_config.disable_transaction_file(),
    )?;
    // Classified from the operation itself. Reading it back off the inline
    // copy would tie the verdict to the payload size instead.
    let may_change_schema = operation_may_change_schema(&pb_transaction);

    // Files a shallow clone writes into the target before the manifest
    // commit; deleted again when the commit conclusively fails.
    let mut spilled_clone_files: Vec<Path> = Vec::new();
    let clone_source = if let Operation::Clone {
        is_shallow,
        ref_version,
        ref_path,
        ..
    } = &transaction.operation
    {
        // The source manifest must be read through the source store, which may differ
        // from the destination store when cloning across object stores/accounts. Falls
        // back to the destination store for same-store clones.
        let source_store = source_store.unwrap_or(object_store);
        let source_base_path =
            ObjectStore::extract_path_from_uri(session.store_registry(), ref_path.as_str())?;
        let source_manifest_location = commit_handler
            .resolve_version_location(&source_base_path, *ref_version, &source_store.inner)
            .await?;
        let source_manifest = Table::load_manifest(
            source_store,
            &source_manifest_location,
            ref_path.as_str(),
            &session,
        )
        .await?;
        ensure_can_write_manifest(&source_manifest)?;

        // Prepare the cloned index metadata now: an uncloneable tagged FRI
        // history must be rejected before anything is written to the target.
        let indices = if let Some(index_section_pos) = source_manifest.index_section {
            let reader = source_store.open(&source_manifest_location.path).await?;
            let section: pb::IndexSection =
                lance_io::utils::read_message(reader.as_ref(), index_section_pos).await?;
            let contract = CoveringContract::of(&source_manifest);
            section
                .indices
                .into_iter()
                .map(|proto| IndexMetadata::from_manifest_proto(proto, contract))
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let new_base_id =
            lance_table::format::BasePath::unused_id(source_manifest.base_paths.keys().copied())?;
        let mut updated_indices = Vec::with_capacity(indices.len());
        for mut index in indices {
            if lance_table::system_index::frag_reuse::metadata::is_tagged(&index) {
                // Restamp the entry's row-map references through the clone's
                // base mapping and move the entry itself into the clone. A
                // shallow clone leaves the row-map files where they are; a
                // deep clone copies them into its own `_fri/` (the copy loop
                // in `deep_clone`), so every reference becomes local.
                let base_remap = if *is_shallow {
                    crate::frag_reuse::CloneBaseRemap::Shallow { new_base_id }
                } else {
                    crate::frag_reuse::CloneBaseRemap::Deep
                };
                let (relocated, spilled) = crate::frag_reuse::relocate_tagged_entry_for_clone(
                    source_store,
                    &source_base_path,
                    &source_manifest,
                    session.store_registry(),
                    &index,
                    base_remap,
                    object_store,
                    base_path,
                )
                .await?;
                index = relocated;
                spilled_clone_files.extend(spilled);
            } else if !*is_shallow {
                // Deep clone: keep metadata but normalize base to local.
                index.base_id = None;
            } else if index.base_id.is_none() {
                // Same rule as the data files in `Manifest::shallow_clone`:
                // only the source's own entries get the new base; entries
                // already stamped keep their ids, which carry over into
                // the clone's `base_paths` verbatim (a chained clone must
                // not restamp an origin-based index onto the middle hop).
                index.base_id = Some(new_base_id);
            }
            updated_indices.push(index);
        }
        Some((
            source_manifest,
            source_manifest_location,
            new_base_id,
            updated_indices,
        ))
    } else {
        None
    };

    let transaction_file = if !build_config.disable_transaction_file() {
        match write_transaction_file(object_store, base_path, &pb_transaction, None).await {
            Ok(transaction_file) => transaction_file,
            Err(err) => {
                cleanup_spilled_clone_files(object_store, &spilled_clone_files).await;
                return Err(err);
            }
        }
    } else {
        String::new()
    };

    let (mut manifest, indices) = if let (
        Operation::Clone {
            is_shallow,
            ref_name,
            ref_path,
            branch_name,
            ..
        },
        Some((source_manifest, source_manifest_location, new_base_id, updated_indices)),
    ) = (&transaction.operation, clone_source)
    {
        if *is_shallow {
            let new_manifest = source_manifest.shallow_clone(
                ref_name.clone(),
                ref_path.clone(),
                new_base_id,
                branch_name.clone(),
                transaction_file.clone(),
            );
            (new_manifest, updated_indices)
        } else {
            // Deep clone: build a manifest that references local files (no external bases)
            let mut new_manifest = source_manifest.clone();
            if !source_manifest.has_managed_blobs() {
                new_manifest.base_paths.clear();
            }
            new_manifest.branch = None;
            new_manifest.tag = None;
            new_manifest.index_section = None; // will be rewritten below
            new_manifest.transaction_file =
                (!transaction_file.is_empty()).then_some(transaction_file.clone());
            let mut new_frags = new_manifest.fragments.as_ref().clone();
            for f in &mut new_frags {
                for df in f.referenced_lance_files_mut() {
                    df.base_id = None;
                }
                if let Some(d) = f.deletion_file.as_mut() {
                    d.base_id = None;
                }
            }
            new_manifest.fragments = Arc::new(new_frags);

            if source_manifest.has_managed_blobs() {
                let source_store = source_store.unwrap_or(object_store);
                let source = Table::checkout_manifest(
                    Arc::new(source_store.clone()),
                    ObjectStore::extract_path_from_uri(session.store_registry(), ref_path)?,
                    ref_path.clone(),
                    Arc::new(source_manifest.clone()),
                    source_manifest_location.clone(),
                    session.clone(),
                    commit_handler.clone(),
                    None,
                    None,
                    None,
                )
                .await?;
                let lifecycle = hooks.lifecycle.ok_or_else(|| Error::not_supported(
                    "A deep clone with managed blobs requires copying their payloads and descriptors",
                ))?;
                lifecycle
                    .copy_managed_blobs(source, object_store, base_path, uri, &mut new_manifest)
                    .await?;
            }

            (new_manifest, updated_indices)
        }
    } else {
        let (manifest, indices) =
            transaction.build_manifest(None, vec![], &transaction_file, &build_config)?;
        (manifest, indices)
    };

    if !manifest.uses_non_reusable_field_ids() {
        fix_schema(&mut manifest)?;
    }

    let result = write_manifest_file(
        object_store,
        commit_handler.as_ref(),
        base_path,
        &mut manifest,
        if indices.is_empty() {
            None
        } else {
            Some(indices.clone())
        },
        &build_config,
        manifest_naming_scheme,
        inline_transaction.then(|| pb_transaction.into()),
        may_change_schema,
        hooks,
    )
    .await;

    // TODO: Allow Append or Overwrite mode to retry using `commit_transaction`
    // if there is a conflict.
    match result {
        Ok(manifest_location) => {
            record_new_dataset_commit(metadata_cache, transaction, &manifest, &manifest_location)
                .await;
            Ok((manifest, manifest_location))
        }
        Err(CommitError::CommitConflict) => {
            // The dataset may "already exist" because this attempt's own
            // manifest write landed but returned an ambiguous error. Verify
            // before reporting a conflict (and before deleting the
            // transaction file a landed manifest would reference).
            match verify_commit_outcome(
                object_store,
                commit_handler.as_ref(),
                base_path,
                manifest.version,
                transaction,
                None,
            )
            .await
            {
                CommitOutcome::Ours {
                    manifest: committed_manifest,
                    location,
                } => {
                    let committed_manifest = *committed_manifest;
                    record_new_dataset_commit(
                        metadata_cache,
                        transaction,
                        &committed_manifest,
                        &location,
                    )
                    .await;
                    return Ok((committed_manifest, location));
                }
                CommitOutcome::Foreign { .. } | CommitOutcome::Absent => {}
                CommitOutcome::Unknown => {
                    return Err(Error::commit_status_unknown_source(
                        manifest.version,
                        "dataset creation reported a conflict but the manifest could not \
                         be read back for verification"
                            .to_string()
                            .into(),
                    ));
                }
            }
            cleanup_transaction_file(object_store, base_path, &transaction_file).await;
            cleanup_spilled_clone_files(object_store, &spilled_clone_files).await;
            Err(Error::dataset_already_exists(base_path.to_string()))
        }
        Err(CommitError::OtherError(err)) => {
            match verify_commit_outcome(
                object_store,
                commit_handler.as_ref(),
                base_path,
                manifest.version,
                transaction,
                None,
            )
            .await
            {
                CommitOutcome::Ours {
                    manifest: committed_manifest,
                    location,
                } => {
                    let committed_manifest = *committed_manifest;
                    record_new_dataset_commit(
                        metadata_cache,
                        transaction,
                        &committed_manifest,
                        &location,
                    )
                    .await;
                    if commit_handler.propagate_commit_error_after_success() {
                        return Err(err);
                    }
                    return Ok((committed_manifest, location));
                }
                CommitOutcome::Foreign { .. } | CommitOutcome::Absent => {}
                CommitOutcome::Unknown => {
                    return Err(Error::commit_status_unknown_source(
                        manifest.version,
                        Box::new(err),
                    ));
                }
            }
            cleanup_transaction_file(object_store, base_path, &transaction_file).await;
            cleanup_spilled_clone_files(object_store, &spilled_clone_files).await;
            Err(err)
        }
    }
}

/// Cache bookkeeping for a successful new-dataset commit, shared by the
/// direct-success and verified-own-commit paths of [`commit_new_dataset`].
async fn record_new_dataset_commit(
    metadata_cache: &DSMetadataCache,
    transaction: &Transaction,
    manifest: &Manifest,
    location: &ManifestLocation,
) {
    let tx_key = crate::caches::TransactionKey {
        version: manifest.version,
    };
    metadata_cache
        .insert_with_key(&tx_key, Arc::new(transaction.clone()))
        .await;

    let manifest_key = crate::caches::ManifestKey {
        version: location.version,
        e_tag: location.e_tag.as_deref(),
    };
    metadata_cache
        .insert_with_key(&manifest_key, Arc::new(manifest.clone()))
        .await;
}
