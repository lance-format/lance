// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::backoff::Backoff;

use lance_core::utils::tracing::{AUDIT_MODE_DELETE, AUDIT_TYPE_TRANSACTION, TRACE_FILE_AUDIT};
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::{
    apply_feature_flags, apply_index_object_metadata_flag, validate_non_reusable_field_id_flags,
    validate_paired_feature_flags,
};
use lance_table::format::{
    IndexMetadata, Manifest, ManifestBuildConfig, pb,
    storage_contract::finalize_manifest_storage_version,
};
use lance_table::io::commit::{
    CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, write_manifest_file_to_path,
};
use lance_table::io::manifest::read_manifest;
use lance_table::system_index::is_system_index;
use lance_table::transaction::Transaction;
use object_store::path::Path;
use prost::Message;

use crate::blob::thresholds::validate_blob_threshold_metadata;
use crate::frag_reuse::{
    apply_frag_reuse_with_stable_row_ids_flag, validate_stable_row_id_fri_shape,
};
use crate::paths::TRANSACTIONS_DIR;
use crate::write::schema::validate_decimal32_and_decimal64;

use super::CommitHooks;
use super::conflict_resolver::AppendIdempotencyKey;

/// Read the raw protobuf transaction from a transaction file.
async fn read_transaction_file_pb(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) -> Result<pb::Transaction> {
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(transaction_file);
    let result = object_store.read_object(&path).await?;
    let data = result.bytes().await?;
    Ok(pb::Transaction::decode(data)?)
}

/// Read the transaction data from a transaction file.
pub async fn read_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) -> Result<Transaction> {
    read_transaction_file_pb(object_store, base_path, transaction_file)
        .await?
        .try_into()
}

/// Best-effort delete of a transaction file that is no longer needed.
///
/// Logs a warning on failure rather than propagating the error, since the
/// primary operation has already failed and the orphaned file will eventually
/// be removed by GC.
///
/// Callers must only invoke this for attempts whose commit is confirmed to
/// have NOT landed (see [`verify_commit_outcome`]): a landed manifest
/// references its transaction file by path, so deleting it would corrupt the
/// version.
pub(super) async fn cleanup_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) {
    if transaction_file.is_empty() {
        return;
    }
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(transaction_file);
    match object_store.delete(&path).await {
        Ok(()) => {
            tracing::info!(
                target: TRACE_FILE_AUDIT,
                mode = AUDIT_MODE_DELETE,
                r#type = AUDIT_TYPE_TRANSACTION,
                path = transaction_file,
            );
        }
        Err(e) => {
            log::warn!(
                "Failed to clean up orphaned transaction file '{}': {}",
                transaction_file,
                e
            );
        }
    }
}

/// Best-effort removal of files a shallow clone staged in the target before
/// its commit conclusively failed (the relocated FRI entry's spilled
/// `details.binpb`). Never called when the commit outcome is unknown: the
/// clone may have landed and still need them.
pub(super) async fn cleanup_spilled_clone_files(
    object_store: &ObjectStore,
    spilled_files: &[Path],
) {
    for path in spilled_files {
        if let Err(e) = object_store.delete(path).await {
            log::warn!("Failed to clean up staged clone file '{}': {}", path, e);
        }
    }
}

/// Who owns the manifest at a version, checked after a failed commit attempt.
#[derive(Debug)]
pub(super) enum CommitOutcome {
    /// The manifest at the version is the one this attempt wrote: the commit
    /// actually landed even though the store reported a failure (e.g. the
    /// response to a successful conditional PUT was lost and an internal
    /// retry surfaced "already exists").
    Ours {
        manifest: Box<Manifest>,
        location: ManifestLocation,
    },
    /// A manifest exists at the version and belongs to another writer.
    Foreign { duplicate_append: bool },
    /// No manifest exists at the version: this attempt definitely did not
    /// land.
    Absent,
    /// The verification reads themselves kept failing; whether the commit
    /// landed cannot be determined. Callers must not run destructive cleanup
    /// in this state.
    Unknown,
}

/// Maximum verification read attempts in [`verify_commit_outcome`].
const COMMIT_VERIFICATION_ATTEMPTS: u32 = 3;

/// Determine whether a failed commit attempt actually landed, by comparing
/// the manifest at `version` with this attempt. A keyed append has a unique
/// transaction file, which identifies the attempt even if another writer
/// submitted the same transaction.
///
/// For unkeyed commits, compare the durable (protobuf) forms. The in-memory
/// [`Transaction`] can carry state that is intentionally not serialized (e.g.
/// the frag reuse payload attached to a rewrite).
///
/// Never returns an error. Read failures and non-definitive not-found results
/// are retried briefly, then collapse to [`CommitOutcome::Unknown`].
pub(super) async fn verify_commit_outcome(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    version: u64,
    transaction: &Transaction,
    keyed_append: Option<(AppendIdempotencyKey<'_>, &str)>,
) -> CommitOutcome {
    enum VerificationFailure {
        NotFound,
        Read(Error),
    }

    // Durable form of this attempt's transaction, matching what the commit
    // path serialized. The commit path already rejected anything that fails
    // to serialize, so this cannot fail in practice; stay conservative if it does.
    let Ok(transaction_pb) = pb::Transaction::try_from(transaction) else {
        return CommitOutcome::Unknown;
    };

    let mut backoff = Backoff::default();
    let failure = loop {
        let failure = match try_read_manifest_at(object_store, commit_handler, base_path, version)
            .await
        {
            Ok(Some((manifest, location))) => {
                if keyed_append
                    .is_some_and(|(_, file)| manifest.transaction_file.as_deref() == Some(file))
                {
                    return CommitOutcome::Ours {
                        manifest: Box::new(manifest),
                        location,
                    };
                }
                match read_manifest_transaction(object_store, base_path, &manifest, &location).await
                {
                    Ok(Some(committed_transaction)) => {
                        return if keyed_append.is_none() && committed_transaction == transaction_pb
                        {
                            CommitOutcome::Ours {
                                manifest: Box::new(manifest),
                                location,
                            }
                        } else {
                            let duplicate_append = keyed_append.is_some_and(|(key, _)| {
                                matches!(
                                    committed_transaction.operation,
                                    Some(pb::transaction::Operation::Append(_))
                                ) && key.matches_properties(
                                    &committed_transaction.transaction_properties,
                                )
                            });
                            CommitOutcome::Foreign { duplicate_append }
                        };
                    }
                    Ok(None) => {
                        return CommitOutcome::Foreign {
                            duplicate_append: false,
                        };
                    }
                    Err(error) => VerificationFailure::Read(error),
                }
            }
            Ok(None) if commit_handler.is_version_not_found_definitive() => {
                return CommitOutcome::Absent;
            }
            Ok(None) => VerificationFailure::NotFound,
            Err(error) => VerificationFailure::Read(error),
        };

        if backoff.attempt() + 1 >= COMMIT_VERIFICATION_ATTEMPTS {
            break failure;
        }
        tokio::time::sleep(backoff.next_backoff()).await;
    };

    match failure {
        VerificationFailure::NotFound => {
            log::warn!(
                "The manifest for version {} was not visible after {} commit verification \
                 attempts, and the commit handler does not guarantee definitive not-found \
                 results; treating the commit status as unknown",
                version,
                COMMIT_VERIFICATION_ATTEMPTS
            );
            CommitOutcome::Unknown
        }
        VerificationFailure::Read(error) => {
            log::warn!(
                "Could not verify the outcome of the commit attempt for version {} after {} \
                 tries; treating the commit status as unknown: {}",
                version,
                COMMIT_VERIFICATION_ATTEMPTS,
                error
            );
            CommitOutcome::Unknown
        }
    }
}

async fn read_manifest_transaction(
    object_store: &ObjectStore,
    base_path: &Path,
    manifest: &Manifest,
    location: &ManifestLocation,
) -> Result<Option<pb::Transaction>> {
    if let Some(position) = manifest.transaction_section {
        let reader = if let Some(size) = location.size {
            object_store
                .open_with_size(&location.path, size as usize)
                .await?
        } else {
            object_store.open(&location.path).await?
        };
        let transaction: pb::Transaction =
            lance_io::utils::read_message(reader.as_ref(), position).await?;
        Ok(Some(transaction))
    } else if let Some(transaction_file) = manifest.transaction_file.as_deref() {
        read_transaction_file_pb(object_store, base_path, transaction_file)
            .await
            .map(Some)
    } else {
        Ok(None)
    }
}

/// Read the manifest at `version`, distinguishing "no such version"
/// (`Ok(None)`) from transient read failures (`Err`).
async fn try_read_manifest_at(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    version: u64,
) -> Result<Option<(Manifest, ManifestLocation)>> {
    let location = match commit_handler
        .resolve_version_location(base_path, version, &object_store.inner)
        .await
    {
        Ok(location) => location,
        Err(Error::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(e),
    };
    match read_manifest(object_store, &location.path, location.size).await {
        Ok(manifest) => Ok(Some((manifest, location))),
        Err(Error::NotFound { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Write a transaction to a file and return the relative path.
pub async fn write_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &pb::Transaction,
    attempt_id: Option<uuid::Uuid>,
) -> Result<String> {
    let file_name = if let Some(attempt_id) = attempt_id {
        format!("{}-{attempt_id}.txn", transaction.read_version)
    } else {
        format!("{}-{}.txn", transaction.read_version, transaction.uuid)
    };
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(file_name.as_str());

    let buf = transaction.encode_to_vec();
    object_store.put(&path, &buf).await?;

    Ok(file_name)
}

/// Transactions serialized above this size are not inlined into the manifest.
#[cfg(not(test))]
pub const MAX_INLINE_TRANSACTION_BYTES: usize = 20 * 1024 * 1024;
/// Smaller threshold for unit tests so spill coverage does not need
/// multi-megabyte payloads.
#[cfg(test)]
pub const MAX_INLINE_TRANSACTION_BYTES: usize = 64 * 1024;

/// Commit a manifest file and create a copy at the latest manifest path.
#[allow(clippy::too_many_arguments)]
pub async fn write_manifest_file(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    config: &ManifestBuildConfig,
    naming_scheme: ManifestNamingScheme,
    transaction: Option<lance_table::format::Transaction>,
    may_change_schema: bool,
    hooks: CommitHooks<'_>,
) -> std::result::Result<ManifestLocation, CommitError> {
    manifest.update_max_field_id();
    validate_paired_feature_flags(manifest)?;
    if let Some(indices) = &indices {
        lance_table::system_index::frag_reuse::metadata::validate_flags(manifest, indices)?;
    }
    // Every manifest write funnels through here, including restore and clone,
    // which rebuild a manifest from a stored one rather than from an Arrow
    // schema. Validate names and primary keys at this boundary so all write
    // paths enforce them.
    //
    // Only for transactions that can change the schema. Released versions could
    // install a key on a nullable column through the metadata path, and
    // validating every write would make such a table read-only on upgrade --
    // including through the delete that removes the offending rows, which is
    // the first step of repairing it. A repair still has to pass: it changes
    // the schema, and the schema it produces is valid.
    //
    // The caller classifies the operation, rather than this reading it off
    // `transaction`, which is None whenever the encoded bytes were too large
    // to inline. Deriving it here would make the verdict depend on payload
    // size, so the same operation would be exempt while small and validated
    // once it spilled -- and a MemWAL table spills routinely, since its
    // transactions carry mem-table state.
    if may_change_schema {
        manifest
            .schema
            .validate_writable()
            .map_err(CommitError::OtherError)?;
        manifest
            .schema
            .verify_primary_key()
            .map_err(CommitError::OtherError)?;
        validate_blob_threshold_metadata(&manifest.schema).map_err(CommitError::OtherError)?;
    }
    if config.auto_set_feature_flags() {
        // build_manifest may have already set FLAG_STABLE_ROW_IDS on the manifest.
        // Preserve it here so this second apply_feature_flags call does not clear it
        // when config.use_stable_row_ids() is false (the ManifestBuildConfig default).
        let use_stable_row_ids = config.use_stable_row_ids() || manifest.uses_stable_row_ids();
        apply_feature_flags(
            manifest,
            use_stable_row_ids,
            config.disable_transaction_file(),
        )?;
        apply_frag_reuse_with_stable_row_ids_flag(manifest, indices.as_deref().unwrap_or_default());
        // Here as well as in `build_manifest`: the clone and restore paths reach
        // this write with their index section but without a build, and a clone
        // starts from a masked writer word.
        apply_index_object_metadata_flag(manifest, indices.as_deref().unwrap_or_default());
    }
    // Restore and clone bypass build_manifest and may copy an older snapshot
    // that predates the writer fence.
    if let Some(indices) = &indices {
        lance_table::feature_flags::apply_row_addr_index_feature_flag(manifest, indices);
    }
    // After the flag reset, so it judges the flags being published. Here rather
    // than in `build_manifest`, which restore and clone bypass.
    let index_metadata = indices.as_deref().unwrap_or_default();
    if validate_stable_row_id_fri_shape(
        manifest,
        index_metadata,
        lance_table::feature_flags::frag_reuse_with_stable_row_ids_enabled(),
    )? {
        if let Some(hook) = hooks.index {
            hook.validate_stable_row_id_indices(manifest, index_metadata)?;
        } else if index_metadata.iter().any(|index| !is_system_index(index)) {
            return Err(Error::not_supported(
                "Cannot verify user indices on a stable-row-id table with fragment reuse without a commit index hook",
            ).into());
        }
    }
    lance_table::feature_flags::validate_data_overlay_files(manifest)?;

    validate_non_reusable_field_id_flags(manifest).map_err(CommitError::OtherError)?;
    finalize_manifest_storage_version(manifest)?;
    // Adding an all-null column writes no data file, so no file writer checks it.
    // This runs after finalizing, which can raise a legacy default to the files'
    // version, so it checks the version being published.
    if may_change_schema {
        validate_decimal32_and_decimal64(
            &manifest.schema,
            manifest.data_storage_format.lance_file_format(),
        )
        .map_err(CommitError::OtherError)?;
    }

    manifest.set_timestamp(
        config
            .timestamp_nanos()
            .unwrap_or_else(|| hooks.timestamp_nanos()),
    );

    manifest.update_max_fragment_id();

    commit_handler
        .commit(
            manifest,
            indices,
            base_path,
            object_store,
            write_manifest_file_to_path,
            naming_scheme,
            transaction,
        )
        .await
}
