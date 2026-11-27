// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Instant;

use lance_core::utils::backoff::Backoff;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::ensure_can_write_manifest;
use lance_table::format::{DETACHED_VERSION_MASK, Manifest, operation_may_change_schema, pb};
use lance_table::io::commit::{CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme};
use lance_table::transaction::{
    FragReuseUpdate, Operation, Transaction, canonicalize_non_reusable_field_ids,
    validate_detached_non_reusable_field_ids, validate_non_reusable_field_id_transition,
    validate_operation,
};
use prost::Message;
use rand::{Rng, rng};

use crate::Table;
use crate::frag_reuse::history::{ReadSnapshot, validate_frag_reuse_history};

use super::index::validate_new_segments;
use super::migration::{check_fragment_ids, fix_schema, migrate_indices, migrate_manifest};
use super::publish::{
    CommitOutcome, cleanup_transaction_file, verify_commit_outcome, write_manifest_file,
    write_transaction_file,
};
use super::retry::{maybe_timeout, timeout_error};
use super::transaction::{build_config_for_attempt, finish_built_manifest, prepare_attempt};
use super::{CommitHooks, CommitTransactionOptions};

async fn do_commit_detached_transaction(
    dataset: &Table,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    options: CommitTransactionOptions<'_>,
    hooks: CommitHooks<'_>,
) -> Result<(Manifest, ManifestLocation)> {
    let CommitTransactionOptions {
        build_config,
        commit_config,
        retry_timeout,
        ..
    } = options;
    ensure_can_write_manifest(&dataset.manifest)?;
    // Definitions are allocated against the version chain: a detached
    // manifest's ids and field-identity activation would be invisible to it.
    if build_config.clustering_definitions().is_some() {
        return Err(Error::invalid_input(
            "a clustering definitions list cannot be committed detached",
        ));
    }
    let mut transaction = transaction.clone();
    canonicalize_non_reusable_field_ids(Some(&dataset.manifest), &mut transaction.operation, None)?;
    let transaction = &transaction;
    validate_detached_non_reusable_field_ids(&dataset.manifest, &transaction.operation)?;
    validate_operation(Some(&dataset.manifest), &transaction.operation)?;
    // Detached commits skip the rebase pipeline, so a rewrite's transition
    // intent would never be assembled or validated (a dummy intent plus a
    // hand-built entry would satisfy the manifest chokepoint unvalidated).
    // A detached manifest is also outside the main version chain, where an
    // appended fragment-reuse history has no meaning. Reject both shapes
    // outright.
    if let Operation::Rewrite {
        frag_reuse_index: Some(entry),
        ..
    } = &transaction.operation
        && lance_table::system_index::frag_reuse::metadata::is_tagged(entry)
    {
        return Err(Error::not_supported(
            "Detached commits cannot carry fragment reuse transition intent or a tagged \
             fragment reuse entry; commit the rewrite on the main version chain",
        ));
    }
    let pb_transaction = pb::Transaction::try_from(transaction)?;
    let inline_transaction = hooks.inlines_transaction(
        pb_transaction.encoded_len(),
        build_config.disable_transaction_file(),
    )?;
    // Classified from the operation itself. Reading it back off the inline
    // copy would tie the verdict to the payload size instead.
    let may_change_schema = operation_may_change_schema(&pb_transaction);

    // We don't strictly need a transaction file but we go ahead and create one for
    // record-keeping if nothing else.
    let transaction_file = if !build_config.disable_transaction_file() {
        write_transaction_file(object_store, &dataset.state.base, &pb_transaction, None).await?
    } else {
        String::new()
    };

    // The inline copy is moved into the first commit attempt; a retry (only on
    // a random-version collision) rebuilds it instead of cloning up front.
    let mut inline_tx: Option<lance_table::format::Transaction> =
        inline_transaction.then(|| pb_transaction.into());

    // We still do a loop since we may have conflicts in the random version we pick
    let mut backoff = Backoff::default();
    let start = Instant::now();
    while backoff.attempt() < commit_config.num_retries {
        // Pick a random u64 with the highest bit set to indicate it is detached
        let random_version = rng().random::<u64>() | DETACHED_VERSION_MASK;

        let attempt_build_config =
            build_config_for_attempt(dataset, transaction, build_config, hooks).await?;
        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
                    version,
                    &attempt_build_config,
                    &transaction_file,
                    &dataset.manifest,
                )
                .await?
            }
            _ => {
                // Nothing is settled about a tagged fragment reuse entry
                // on a detached commit: a tagged rewrite entry was refused
                // above and a trim never commits detached.
                let prepared = prepare_attempt(
                    dataset,
                    transaction,
                    &attempt_build_config,
                    FragReuseUpdate::None,
                    hooks.index,
                )
                .await?;
                let (manifest, mut indices) = transaction.build_manifest_prepared(
                    Some(dataset.manifest.as_ref()),
                    prepared,
                    &transaction_file,
                    &attempt_build_config,
                    None,
                )?;
                finish_built_manifest(dataset, transaction, &mut indices).await?;
                (manifest, indices)
            }
        };

        // Boxed with its cleanup: an error held across the cleanup's await
        // would be stored in the commit future itself.
        Box::pin(async {
            let checked = validate_new_segments(
                hooks.index,
                dataset,
                &transaction.operation,
                &manifest,
                &indices,
            )
            .await;
            if checked.is_err() {
                cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file)
                    .await;
            }
            checked
        })
        .await?;

        manifest.version = random_version;

        // recompute_stats is always false so far because detached manifests are newer than
        // the old stats bug.
        migrate_manifest(dataset, &mut manifest, /*recompute_stats=*/ false).await?;
        // Validate before the fragment-id check to preserve legacy migration
        // diagnostics. Finalization repeats this at the manifest write boundary.
        fix_schema(&mut manifest)?;
        validate_non_reusable_field_id_transition(
            &dataset.manifest,
            &manifest,
            &transaction.operation,
        )?;
        lance_table::format::storage_contract::check_manifest_storage_version_for_commit(
            &mut manifest,
        )?;
        check_fragment_ids(&manifest)?;
        // Runs after the coverage derivation and can replace a fragment bitmap
        // while keeping its UUID, so anything it narrowed loses its position.
        let recovered_coverage = migrate_indices(dataset, &mut indices, hooks.index).await?;
        Transaction::withdraw_coverage_invalidated_after_build(
            &mut indices,
            &recovered_coverage,
            manifest.version,
        )?;

        // Checked against the manifest the detached one is built on, which need
        // not be the transaction's read version.
        if let Err(error) = validate_frag_reuse_history(
            &transaction.operation,
            dataset,
            ReadSnapshot::Version {
                base: dataset,
                version: transaction.read_version,
            },
            &manifest,
            &indices,
        )
        .await
        {
            cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file).await;
            return Err(error);
        }

        // Try to commit the manifest
        dataset.view().ensure_branch_exists().await?;
        let result = write_manifest_file(
            object_store,
            commit_handler,
            &dataset.state.base,
            &mut manifest,
            if indices.is_empty() {
                None
            } else {
                Some(indices.clone())
            },
            &attempt_build_config,
            ManifestNamingScheme::V2,
            inline_tx.take(),
            may_change_schema,
            hooks,
        )
        .await;

        match result {
            Ok(location) => {
                return Ok((manifest, location));
            }
            Err(CommitError::CommitConflict) => {
                // Either an (extremely unlikely) random-version collision, or
                // our own write landed but returned an ambiguous error.
                // Verify before retrying with a new random version.
                match verify_commit_outcome(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
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
                        return Ok((*committed_manifest, location));
                    }
                    CommitOutcome::Foreign { .. } | CommitOutcome::Absent => {}
                    CommitOutcome::Unknown => {
                        return Err(Error::commit_status_unknown_source(
                            manifest.version,
                            "detached commit reported a conflict but the manifest could \
                             not be read back for verification"
                                .to_string()
                                .into(),
                        ));
                    }
                }
                if start.elapsed() > retry_timeout {
                    cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file)
                        .await;
                    return Err(timeout_error(retry_timeout, backoff.attempt() + 1));
                }
                let sleep_fut = tokio::time::sleep(backoff.next_backoff());
                if let Err(error) =
                    maybe_timeout(backoff.attempt(), start, retry_timeout, sleep_fut).await
                {
                    cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file)
                        .await;
                    return Err(error);
                }
                // The inline copy was moved into the failed attempt; rebuild
                // it for the retry with a new random version.
                inline_tx = if inline_transaction {
                    Some(lance_table::format::Transaction::try_from(transaction)?)
                } else {
                    None
                };
            }
            Err(CommitError::OtherError(err)) => {
                match verify_commit_outcome(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
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
                        if commit_handler.propagate_commit_error_after_success() {
                            return Err(err);
                        }
                        return Ok((*committed_manifest, location));
                    }
                    CommitOutcome::Foreign { .. } | CommitOutcome::Absent => {}
                    CommitOutcome::Unknown => {
                        return Err(Error::commit_status_unknown_source(
                            manifest.version,
                            Box::new(err),
                        ));
                    }
                }
                cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file)
                    .await;
                return Err(err);
            }
        }
    }

    // This should be extremely unlikely.  There should not be *that* many detached commits.  If
    // this happens then it seems more likely there is a bug in our random u64 generation.
    cleanup_transaction_file(object_store, &dataset.state.base, &transaction_file).await;
    Err(Error::commit_conflict_source(
        0,
        format!(
            "Failed find unused random u64 after {} retries.",
            commit_config.num_retries
        )
        .into(),
    ))
}

pub(super) async fn commit_detached_transaction(
    dataset: &Table,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    options: CommitTransactionOptions<'_>,
    hooks: CommitHooks<'_>,
) -> Result<(Manifest, ManifestLocation)> {
    dataset.view().ensure_branch_exists().await?;
    let result = do_commit_detached_transaction(
        dataset,
        object_store,
        commit_handler,
        transaction,
        options,
        hooks,
    )
    .await;
    dataset.view().ensure_branch_exists().await?;
    result
}
