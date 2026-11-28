// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lance_core::utils::backoff::SlotBackoff;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_select::RowAddrTreeMap;
use lance_table::feature_flags::{CoveringContract, ensure_can_write_manifest};
use lance_table::format::{
    IndexMetadata, Manifest, ManifestBuildConfig, is_detached_version, operation_may_change_schema,
    pb,
};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme,
};
use lance_table::transaction::{
    FragReuseUpdate, Operation, PreparedIndices, Transaction, canonicalize_non_reusable_field_ids,
    has_writer_placed_lineage, validate_non_reusable_field_id_transition, validate_operation,
};
use prost::Message;

use crate::Table;
use crate::caches::IndexMetadataKey;
use crate::frag_reuse::history::{ReadSnapshot, validate_frag_reuse_history};
use crate::rowids::load_spilled_row_lineage;

use super::conflict_resolver::{self, AppendIdempotencyKey, TransactionRebase};
use super::index::{index_usability, load_attempt_indices, validate_new_segments};
use super::migration::{check_fragment_ids, fix_schema, migrate_indices, migrate_manifest};
use super::publish::{
    CommitOutcome, cleanup_transaction_file, verify_commit_outcome, write_manifest_file,
    write_transaction_file,
};
use super::retry::{default_commit_retry_timeout, maybe_timeout, timeout_error};
use super::transactions::{TransactionReadSource, load_and_sort_new_transactions};
use super::{CommitHooks, CommitIndexHook, CommitLifecycleHook};

/// The build config for one commit attempt against `dataset`'s current manifest.
///
/// Resolves the commit timestamp and loads
/// spilled row lineage when the operation needs existing lineage sequences.
pub(super) async fn build_config_for_attempt(
    dataset: &Table,
    transaction: &Transaction,
    build_config: &ManifestBuildConfig,
    hooks: CommitHooks<'_>,
) -> Result<ManifestBuildConfig> {
    let mut config = build_config
        .clone()
        .with_timestamp_nanos(hooks.timestamp_nanos());
    let reads_existing_lineage = match &transaction.operation {
        // An update resolves its new fragments' created-at versions from the
        // existing fragments unless their writer placed them, and refreshes
        // the existing last-updated-at versions of the offsets it rewrote in
        // place (`updated_fragment_offsets`).
        Operation::Update {
            new_fragments,
            updated_fragment_offsets,
            ..
        } => {
            !new_fragments.iter().all(has_writer_placed_lineage)
                || updated_fragment_offsets
                    .as_ref()
                    .is_some_and(|offsets| !offsets.0.is_empty())
        }
        Operation::DataOverlay { .. } => true,
        _ => false,
    };
    if reads_existing_lineage {
        let lineage = load_spilled_row_lineage(dataset, dataset.manifest.fragments.iter()).await?;
        config = config.with_spilled_row_lineage(lineage);
    }
    Ok(config)
}

/// What an action-based transaction's attempt supplies for the fragment reuse
/// history: the history itself when it lives in an external file, which
/// applying the actions cannot read, and this build's notion of a usable
/// segment, which a trim judges coverage by. `None` for any other operation.
async fn composite_frag_reuse_update(
    dataset: &Table,
    transaction: &Transaction,
    index_hook: Option<&dyn CommitIndexHook>,
) -> Result<Option<FragReuseUpdate>> {
    if !matches!(transaction.operation, Operation::CompositeOperation(_)) {
        return Ok(None);
    }
    let history = crate::frag_reuse::preload_external_frag_reuse_history(dataset).await?;
    Ok(Some(FragReuseUpdate::Actions(Arc::new(
        lance_table::system_index::frag_reuse::FragReuseActionInputs {
            history,
            index_is_usable: index_usability(index_hook),
        },
    ))))
}

/// The IO a freshly built manifest still needs: an action-based history that
/// outgrew the inline limit moves to an external file. Keeping this behind one
/// await point keeps the commit loop's future under the `large_futures` limit.
pub(super) async fn finish_built_manifest(
    dataset: &Table,
    transaction: &Transaction,
    indices: &mut [IndexMetadata],
) -> Result<()> {
    if matches!(transaction.operation, Operation::CompositeOperation(_)) {
        crate::frag_reuse::spill_frag_reuse_history(dataset, indices).await?;
    }
    Ok(())
}

/// Step one of a commit attempt (`Transaction::prepare_indices`): the index
/// list as read from the manifest current at this attempt, and the prepared
/// list an in-place column rewrite has already withdrawn or pruned. For such
/// a rewrite on a table with a tagged fragment reuse history the entry's
/// history is decoded here, once per attempt, so the preparation can walk
/// the lineage without guessing; other operations never interpret the
/// entry: an append, or a merge that only adds columns, must carry a
/// history a newer writer recorded through untouched. `frag_reuse` is what
/// the rebase settled about the entry for this attempt; for an action-based
/// transaction, which settles nothing beforehand, it is derived here. Nothing
/// flows back into `transaction`.
pub(super) async fn prepare_attempt(
    dataset: &Table,
    transaction: &Transaction,
    build_config: &lance_table::format::ManifestBuildConfig,
    frag_reuse: FragReuseUpdate,
    index_hook: Option<&dyn CommitIndexHook>,
) -> Result<PreparedIndices> {
    let frag_reuse = match frag_reuse {
        FragReuseUpdate::None => composite_frag_reuse_update(dataset, transaction, index_hook)
            .await?
            .unwrap_or_default(),
        settled => settled,
    };
    let indices = load_attempt_indices(dataset, &transaction.operation, index_hook).await?;
    // Same rewrite set the preparation withdraws from; nothing else interprets the entry.
    let rewrites_in_place = !Transaction::rewritten_physical_columns(
        &transaction.operation,
        &dataset.manifest.schema,
        &dataset.manifest.fragments,
    )
    .is_empty();
    // A deferred rewrite withdrawing an undeferrable segment's coverage walks
    // the history too, to find the retired sources that reach the rewritten
    // fragments.
    let withdraws_undeferrable = matches!(
        &frag_reuse,
        FragReuseUpdate::Rewrite(assembly) if !assembly.undeferrable_segments.is_empty()
    );
    let ledger = if rewrites_in_place || withdraws_undeferrable {
        match indices
            .iter()
            .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
        {
            Some(entry) => Some(Arc::new(
                crate::frag_reuse::decode_frag_reuse_ledger(dataset, entry).await?,
            )),
            None => None,
        }
    } else {
        None
    };
    transaction.prepare_indices(
        Some(dataset.manifest.as_ref()),
        indices.as_ref().clone(),
        build_config,
        ledger,
        frag_reuse,
    )
}

/// verified-own-commit paths of [`commit_transaction`]: populate the session
/// caches and run the auto-cleanup hook.
async fn record_successful_commit(
    dataset: &Table,
    transaction: &Transaction,
    manifest: &Manifest,
    location: &ManifestLocation,
    indices: Vec<IndexMetadata>,
    skip_auto_cleanup: bool,
    lifecycle_hook: Option<&dyn CommitLifecycleHook>,
) -> Result<()> {
    dataset.view().ensure_branch_exists().await?;
    let tx_key = crate::caches::TransactionKey {
        version: manifest.version,
    };
    dataset
        .state
        .metadata_cache
        .insert_with_key(&tx_key, Arc::new(transaction.clone()))
        .await;

    let manifest_key = crate::caches::ManifestKey {
        version: location.version,
        e_tag: location.e_tag.as_deref(),
    };
    dataset
        .state
        .metadata_cache
        .insert_with_key(&manifest_key, Arc::new(manifest.clone()))
        .await;
    if !indices.is_empty() {
        let key = IndexMetadataKey {
            version: manifest.version,
            store_identity: &dataset.state.object_store.store_prefix,
            e_tag: location.e_tag.as_deref(),
        };
        let contract = CoveringContract::of(manifest);
        let indices = indices
            .into_iter()
            .map(|index| index.into_stored_shape(contract))
            .collect::<Vec<_>>();
        dataset
            .index_cache()
            .insert_with_key(&key, Arc::new(indices))
            .await;
    }

    if !skip_auto_cleanup && let Some(hook) = lifecycle_hook {
        hook.after_commit(dataset, manifest).await;
    }
    dataset.view().ensure_branch_exists().await
}

/// Options for [`commit_transaction`].
pub struct CommitTransactionOptions<'a> {
    pub build_config: &'a ManifestBuildConfig,
    pub commit_config: &'a CommitConfig,
    pub retry_timeout: Duration,
    pub manifest_naming_scheme: ManifestNamingScheme,
    pub affected_rows: Option<&'a RowAddrTreeMap>,
    append_idempotency_key: Option<AppendIdempotencyKey<'a>>,
    pub expected_latest_version: Option<u64>,
    /// Whether this commit is a tagged fragment-reuse-index trim derived by the
    /// maintenance path against the current manifest entry (`FragReuseUpdate::Trim`).
    pub tagged_frag_reuse_trim: bool,
}

impl<'a> CommitTransactionOptions<'a> {
    pub fn new(
        build_config: &'a ManifestBuildConfig,
        commit_config: &'a CommitConfig,
        manifest_naming_scheme: ManifestNamingScheme,
    ) -> Self {
        Self {
            build_config,
            commit_config,
            retry_timeout: default_commit_retry_timeout(),
            manifest_naming_scheme,
            affected_rows: None,
            append_idempotency_key: None,
            expected_latest_version: None,
            tagged_frag_reuse_trim: false,
        }
    }

    pub fn with_retry_timeout(mut self, retry_timeout: Duration) -> Self {
        self.retry_timeout = retry_timeout;
        self
    }

    pub fn with_affected_rows(mut self, affected_rows: Option<&'a RowAddrTreeMap>) -> Self {
        self.affected_rows = affected_rows;
        self
    }

    pub(crate) fn with_append_idempotency_key(
        mut self,
        key: Option<AppendIdempotencyKey<'a>>,
    ) -> Self {
        self.append_idempotency_key = key;
        self
    }

    pub fn with_expected_latest_version(mut self, version: u64) -> Self {
        self.expected_latest_version = Some(version);
        self
    }

    pub fn with_tagged_frag_reuse_trim(mut self) -> Self {
        self.tagged_frag_reuse_trim = true;
        self
    }
}

/// Attempt to commit a transaction, with retries and conflict resolution.
pub async fn commit_transaction(
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
        manifest_naming_scheme,
        affected_rows,
        append_idempotency_key,
        expected_latest_version,
        tagged_frag_reuse_trim,
    } = options;
    if append_idempotency_key.is_some()
        && (expected_latest_version.is_some() || build_config.disable_transaction_file())
    {
        return Err(Error::invalid_input(
            "append idempotency requires conflict resolution and transaction files",
        ));
    }
    dataset.view().ensure_branch_exists().await?;
    // Note: object_store has been configured with WriteParams, but dataset.object_store.as_ref()
    // has not necessarily. So for anything involving writing, use `object_store`.
    let read_version = transaction.read_version;
    let mut target_version = read_version + 1;
    let original_dataset = dataset.clone();
    let conditional_commit = expected_latest_version.is_some();
    if let Some(expected) = expected_latest_version
        && expected != read_version
    {
        return Err(Error::invalid_input(format!(
            "expected latest version {expected} does not match the transaction read version \
             {read_version}"
        )));
    }

    // read_version sometimes defaults to zero for overwrite.
    // If num_retries is zero, we are in "strict overwrite" mode.
    // Strict overwrites are not subject to any sort of automatic conflict resolution.
    let strict_overwrite = matches!(transaction.operation, Operation::Overwrite { .. })
        && commit_config.num_retries == 0;
    let mut dataset =
        if dataset.manifest.version != read_version && (read_version != 0 || strict_overwrite) {
            // If the dataset version is not the same as the read version, we need to
            // checkout the read version.
            dataset.checkout_version(read_version).await?
        } else {
            // If the dataset version is the same as the read version, we can use it directly.
            dataset.clone()
        };

    // The version this transaction read, captured before the retry loop moves
    // `dataset` forward. MemWAL index catch-up is derived from it: an index
    // covering every fragment live here holds every row compaction had copied
    // in by then. Fragment reuse history refusals are classified against it,
    // never against a rebased read version.
    //
    // The Arc is kept rather than cloned out: `load_all_indices` returns shared
    // cached data, so the common case is a cache hit rather than a read.
    let read_version_dataset = dataset.clone();
    let read_version_indices =
        load_attempt_indices(&read_version_dataset, &transaction.operation, hooks.index).await?;
    let read_version_state = Some(lance_table::transaction::ReadVersionState {
        manifest: read_version_dataset.manifest.as_ref(),
        indices: read_version_indices.as_slice(),
    });

    let mut transaction = transaction.clone();
    // What a rewrite on a tagged fragment reuse history assembled for the
    // attempt; handed to the manifest build, never written back into
    // `transaction`.
    let mut tagged_rewrite = None;

    let num_attempts = if conditional_commit {
        1
    } else {
        std::cmp::max(commit_config.num_retries, 1)
    };
    let mut backoff = SlotBackoff::default();
    // `retry_timeout` bounds the conflict-retry phase, which starts once the
    // first attempt has caught up with the versions committed since
    // `read_version`. A long-running writer (index build, compaction) on a
    // write-heavy table can be thousands of versions behind, and that initial
    // catch-up is work the commit must do however many retries it gets.
    let mut retry_start: Option<Instant> = None;

    // An allocating clustering submission from a reusable-id snapshot: judge
    // every transaction in its window for field-identity changes; see
    // `refuse_field_identity_change`. Decided from the version the transaction
    // read, not from the handle it was submitted through, which may already
    // be at a later, activated version.
    let identity_fence = (!read_version_dataset.manifest.uses_non_reusable_field_ids()
        && build_config.allocates_clustering_definitions(&read_version_dataset.manifest))
    .then(|| read_version_dataset.schema().clone());

    while backoff.attempt() < num_attempts {
        // We are pessimistic here and assume there may be other transactions
        // we need to check for. We could be optimistic here and blindly
        // attempt to commit, giving faster performance for sequence writes and
        // slower performance for concurrent writes. But that makes the fast path
        // faster and the slow path slower, which makes performance less predictable
        // for users. So we always check for other transactions.
        if let Some(expected_version) = expected_latest_version {
            ensure_can_write_manifest(&dataset.manifest)?;
            let (latest_manifest, _) = dataset.latest_manifest().await?;
            if latest_manifest.version != expected_version {
                return Err(Error::retryable_commit_conflict_source(
                    expected_version,
                    format!(
                        "latest version advanced from the expected {expected_version} to {} \
                         before the conditional restore could commit",
                        latest_manifest.version
                    )
                    .into(),
                ));
            }
        } else if !strict_overwrite {
            let checked_since = dataset.manifest.version;
            let source = if append_idempotency_key.is_some() {
                TransactionReadSource::Storage
            } else {
                TransactionReadSource::SessionCache
            };
            let (loaded_dataset, other_transactions) =
                load_and_sort_new_transactions(&dataset, source).await?;
            dataset = loaded_dataset;

            ensure_can_write_manifest(&dataset.manifest)?;

            // See if we can retry the commit. Try to account for all
            // transactions that have been committed since the read_version.
            // Use small amount of backoff to handle transactions that all
            // started at exact same time better.

            let mut rebase =
                TransactionRebase::try_new(&original_dataset, transaction, affected_rows)
                    .await?
                    .with_read_version_dataset(&read_version_dataset);
            rebase.load_current_lineage(&dataset).await?;
            let seen = other_transactions
                .iter()
                .map(|(version, _)| *version)
                .collect::<HashSet<_>>();
            let missing = (checked_since + 1..=dataset.manifest.version)
                .filter(|version| !seen.contains(version))
                .collect::<Vec<_>>();
            if let Some(key) = append_idempotency_key {
                if let Some(version) = key.committed_version(&other_transactions) {
                    original_dataset.view().ensure_branch_exists().await?;
                    return Err(Error::append_already_committed(version));
                }
                if !missing.is_empty() {
                    return Err(Error::commit_conflict_source(
                        dataset.manifest.version,
                        format!(
                            "Cannot check append idempotency because the transaction history is \
                             missing at versions {missing:?}"
                        )
                        .into(),
                    ));
                }
            }
            rebase.check_missing_versions(
                &missing,
                dataset.manifest.version,
                read_version_dataset.schema(),
                dataset.schema(),
            )?;
            // A retry's window starts after its previous attempt, so it would
            // miss the moves recorded there; not tracking only costs coverage.
            if checked_since == read_version && missing.is_empty() {
                rebase.track_moved_files();
            }
            // A cleaned-up version may have replaced a key field; an identity
            // window with a hole cannot be judged, so it is refused.
            if identity_fence.is_some() && !missing.is_empty() {
                return Err(Error::commit_conflict_source(
                    dataset.manifest.version,
                    format!(
                        "versions {missing:?} between this submission's read version and the \
                         latest were cleaned up, so the field identity of its clustering keys \
                         cannot be checked; reload the table and rebuild the list"
                    )
                    .into(),
                ));
            }

            for (other_version, other_transaction) in other_transactions.iter() {
                if let Some(read_schema) = &identity_fence {
                    conflict_resolver::refuse_field_identity_change(
                        read_schema,
                        other_transaction,
                        *other_version,
                    )?;
                }
                rebase.check_txn(other_transaction, *other_version)?;
            }

            let (rebased, assembly) = rebase.finish_with_index_hook(&dataset, hooks.index).await?;
            transaction = rebased;
            tagged_rewrite = assembly;
        } else {
            ensure_can_write_manifest(&dataset.manifest)?;
        }
        let attempt_start = Instant::now();
        let retry_start = *retry_start.get_or_insert(attempt_start);

        // Merge preserves fields from the read version; Overwrite replaces all
        // fields. Allocate new IDs above the latest high-water mark.
        // Each attempt remaps its own copy of the
        // staged files, so a retry never mistakes a provisional ID for a field
        // introduced by a concurrent commit.
        // Keep the per-attempt copy out of the commit future's inline state.
        let mut attempt_transaction = Box::new(transaction.clone());
        canonicalize_non_reusable_field_ids(
            Some(&dataset.manifest),
            &mut attempt_transaction.operation,
            Some(read_version_dataset.schema()),
        )?;
        validate_operation(Some(&dataset.manifest), &attempt_transaction.operation)?;

        // Recomputed every attempt: the rebase above may have rewritten the
        // transaction.
        let pb_transaction = pb::Transaction::try_from(attempt_transaction.as_ref())?;
        let inline_transaction = hooks.inlines_transaction(
            pb_transaction.encoded_len(),
            build_config.disable_transaction_file(),
        )?;
        // Classified from the operation itself. Reading it back off the inline
        // copy would tie the verdict to the payload size instead.
        let may_change_schema = operation_may_change_schema(&pb_transaction);

        let current_transaction_file = if !build_config.disable_transaction_file() {
            // Concurrent executions can submit the very same Transaction. Each
            // keyed attempt must own its file so a loser cannot delete the
            // winner's transaction record during conflict cleanup.
            let attempt_id = append_idempotency_key.map(|_| uuid::Uuid::new_v4());
            write_transaction_file(
                object_store,
                &dataset.state.base,
                &pb_transaction,
                attempt_id,
            )
            .await?
        } else {
            String::new()
        };
        let transaction_file = current_transaction_file.as_str();

        target_version = dataset.manifest.version + 1;
        if is_detached_version(target_version) {
            return Err(Error::internal(
                "more than 2^65 versions have been created and so regular version numbers are appearing as 'detached' versions.",
            ));
        }
        // Build an up-to-date manifest from the transaction and current
        // manifest: prepare the index list against this attempt's manifest
        // first, then build from the prepared result.
        let attempt_build_config =
            build_config_for_attempt(&dataset, &attempt_transaction, build_config, hooks).await?;
        let (mut manifest, mut indices) = match attempt_transaction.operation {
            Operation::Restore { version } => {
                // A restore reinstates its snapshot's index list and entry
                // whole; nothing is prepared or withdrawn a second time.
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
                    version,
                    &attempt_build_config,
                    transaction_file,
                    &dataset.manifest,
                )
                .await?
            }
            _ => {
                let frag_reuse = match tagged_rewrite.take() {
                    Some(assembly) => FragReuseUpdate::Rewrite(assembly),
                    None if tagged_frag_reuse_trim => FragReuseUpdate::Trim,
                    None => FragReuseUpdate::None,
                };
                let prepared = prepare_attempt(
                    &dataset,
                    &attempt_transaction,
                    &attempt_build_config,
                    frag_reuse,
                    hooks.index,
                )
                .await?;
                let (manifest, mut indices) = attempt_transaction.build_manifest_prepared(
                    Some(dataset.manifest.as_ref()),
                    prepared,
                    transaction_file,
                    &attempt_build_config,
                    read_version_state,
                )?;
                finish_built_manifest(&dataset, &attempt_transaction, &mut indices).await?;
                (manifest, indices)
            }
        };

        // Boxed with its cleanup: an error held across the cleanup's await
        // would be stored in the commit future itself.
        Box::pin(async {
            let checked = validate_new_segments(
                hooks.index,
                &dataset,
                &attempt_transaction.operation,
                &manifest,
                &indices,
            )
            .await;
            if checked.is_err() {
                cleanup_transaction_file(
                    object_store,
                    &dataset.state.base,
                    &current_transaction_file,
                )
                .await;
            }
            checked
        })
        .await?;

        manifest.version = target_version;

        let previous_writer_version = &dataset.manifest.writer_version;
        // The versions of Lance prior to when we started writing the writer version
        // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
        // make sure to recompute them.
        // See: https://github.com/lance-format/lance/issues/1531
        let recompute_stats = previous_writer_version.is_none();

        migrate_manifest(&dataset, &mut manifest, recompute_stats).await?;

        fix_schema(&mut manifest)?;
        validate_non_reusable_field_id_transition(
            &dataset.manifest,
            &manifest,
            &attempt_transaction.operation,
        )?;

        lance_table::format::storage_contract::check_manifest_storage_version_for_commit(
            &mut manifest,
        )?;
        check_fragment_ids(&manifest)?;

        // Runs after the coverage derivation and can replace a fragment bitmap
        // while keeping its UUID, so anything it narrowed loses its position.
        let recovered_coverage = migrate_indices(&dataset, &mut indices, hooks.index).await?;
        Transaction::withdraw_coverage_invalidated_after_build(
            &mut indices,
            &recovered_coverage,
            target_version,
        )?;

        // Before the write, so a refusal is final rather than a commit outcome
        // to verify. Every attempt checks against the manifest it replaces.
        if let Err(error) = validate_frag_reuse_history(
            &transaction.operation,
            &dataset,
            ReadSnapshot::Loaded {
                dataset: &read_version_dataset,
                indices: read_version_indices.as_slice(),
            },
            &manifest,
            &indices,
        )
        .await
        {
            cleanup_transaction_file(object_store, &dataset.state.base, &current_transaction_file)
                .await;
            return Err(error);
        }

        // Try to commit the manifest
        original_dataset.view().ensure_branch_exists().await?;
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
            manifest_naming_scheme,
            inline_transaction.then(|| pb_transaction.into()),
            may_change_schema,
            hooks,
        )
        .await;
        // Deletion can race the manifest PUT. Never acknowledge an orphaned
        // commit, including when the handler reports an ambiguous result.
        original_dataset.view().ensure_branch_exists().await?;

        let keyed_append = append_idempotency_key.map(|key| (key, transaction_file));
        match result {
            Ok(manifest_location) => {
                record_successful_commit(
                    &dataset,
                    &attempt_transaction,
                    &manifest,
                    &manifest_location,
                    indices,
                    commit_config.skip_auto_cleanup,
                    hooks.lifecycle,
                )
                .await?;
                return Ok((manifest, manifest_location));
            }
            Err(CommitError::CommitConflict) => {
                // The store may have applied this attempt's write and still
                // reported a conflict (e.g. the response to a successful
                // conditional PUT was lost and an internal retry saw
                // "already exists"). Verify who owns the version before
                // treating the attempt as lost: deleting the artifacts of a
                // commit that actually landed corrupts the version.
                let duplicate_append = match verify_commit_outcome(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
                    target_version,
                    &attempt_transaction,
                    keyed_append,
                )
                .await
                {
                    CommitOutcome::Ours {
                        manifest: committed_manifest,
                        location,
                    } => {
                        let committed_manifest = *committed_manifest;
                        record_successful_commit(
                            &dataset,
                            &attempt_transaction,
                            &committed_manifest,
                            &location,
                            indices,
                            commit_config.skip_auto_cleanup,
                            hooks.lifecycle,
                        )
                        .await?;
                        return Ok((committed_manifest, location));
                    }
                    // Confirmed loss: another writer owns the version (or,
                    // for handlers that detect conflicts before writing, the
                    // attempt never landed). Only a different key can be retried.
                    CommitOutcome::Foreign { duplicate_append } => duplicate_append,
                    CommitOutcome::Absent => false,
                    CommitOutcome::Unknown => {
                        return Err(Error::commit_status_unknown_source(
                            target_version,
                            "commit reported a conflict but the manifest at the target \
                             version could not be read back for verification"
                                .to_string()
                                .into(),
                        ));
                    }
                };
                let next_attempt_i = backoff.attempt() + 1;

                if !duplicate_append && backoff.attempt() == 0 {
                    // We add 10% buffer here, to allow concurrent writes to complete.
                    // We pass the first attempt's time to the backoff so it's used
                    // as the unit for backoff time slots. The catch-up before the
                    // attempt is excluded: it is not the window in which concurrent
                    // writers collide with this one.
                    // See SlotBackoff implementation for more details on how this works.
                    backoff =
                        backoff.with_unit((attempt_start.elapsed().as_millis() * 11 / 10) as u32);
                }

                // The attempt lost, so no committed manifest references its file.
                cleanup_transaction_file(
                    object_store,
                    &dataset.state.base,
                    &current_transaction_file,
                )
                .await;
                if duplicate_append {
                    return Err(Error::append_already_committed(target_version));
                }
                if next_attempt_i < num_attempts {
                    if retry_start.elapsed() > retry_timeout {
                        return Err(timeout_error(retry_timeout, backoff.attempt() + 1));
                    }
                    let sleep_fut = tokio::time::sleep(backoff.next_backoff());
                    maybe_timeout(backoff.attempt(), retry_start, retry_timeout, sleep_fut).await?;
                    continue;
                } else {
                    break;
                }
            }
            Err(CommitError::OtherError(err)) => {
                match verify_commit_outcome(
                    object_store,
                    commit_handler,
                    &dataset.state.base,
                    target_version,
                    &attempt_transaction,
                    keyed_append,
                )
                .await
                {
                    CommitOutcome::Ours {
                        manifest: committed_manifest,
                        location,
                    } => {
                        let committed_manifest = *committed_manifest;
                        record_successful_commit(
                            &dataset,
                            &attempt_transaction,
                            &committed_manifest,
                            &location,
                            indices,
                            commit_config.skip_auto_cleanup,
                            hooks.lifecycle,
                        )
                        .await?;
                        if commit_handler.propagate_commit_error_after_success() {
                            return Err(err);
                        }
                        return Ok((committed_manifest, location));
                    }
                    outcome @ (CommitOutcome::Foreign { .. } | CommitOutcome::Absent) => {
                        let duplicate_append = matches!(
                            outcome,
                            CommitOutcome::Foreign {
                                duplicate_append: true
                            }
                        );
                        // The attempt certainly did not land; its
                        // transaction file is orphaned.
                        cleanup_transaction_file(
                            object_store,
                            &dataset.state.base,
                            &current_transaction_file,
                        )
                        .await;
                        if duplicate_append {
                            return Err(Error::append_already_committed(target_version));
                        }
                        if conditional_commit {
                            return Err(Error::retryable_commit_conflict_source(
                                target_version,
                                Box::new(err),
                            ));
                        }
                        return Err(err);
                    }
                    CommitOutcome::Unknown => {
                        return Err(Error::commit_status_unknown_source(
                            target_version,
                            Box::new(err),
                        ));
                    }
                }
            }
        }
    }

    let message = format!(
        "Failed to commit the transaction after {} retries.",
        commit_config.num_retries
    );
    if conditional_commit {
        Err(Error::retryable_commit_conflict_source(
            target_version,
            message.into(),
        ))
    } else {
        Err(Error::commit_conflict_source(
            target_version,
            message.into(),
        ))
    }
}
