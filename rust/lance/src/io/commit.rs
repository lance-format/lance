// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for commit implementations.
//!
//! In Lance, a transaction is committed by writing the next manifest file.
//! However, care should be taken to ensure that the manifest file is written
//! only once, even if there are concurrent writers. Different stores have
//! different abilities to handle concurrent writes, so a trait is provided
//! to allow for different implementations.
//!
//! The trait [`CommitHandler`] can be implemented to provide different commit
//! strategies. The default implementation for most object stores is
//! `ConditionalPutCommitHandler`, which writes the manifest to a temporary path, then
//! renames the temporary path to the final path if no object already exists
//! at the final path.
//!
//! When providing your own commit handler, most often you are implementing in
//! terms of a lock. The trait `CommitLock` can be implemented as a simpler
//! alternative to [`CommitHandler`].

use std::sync::Arc;

use lance_io::object_store::ObjectStore;
#[cfg(test)]
use lance_table::format::ManifestBuildConfig;
use lance_table::format::{Fragment, Manifest};
#[cfg(test)]
use lance_table::io::commit::ManifestNamingScheme;
use lance_table::io::commit::{CommitHandler, ManifestLocation};
use lance_table::transaction::Transaction;
#[cfg(test)]
use object_store::path::Path;

#[cfg(test)]
use crate::session::Session;
use crate::{Dataset, Result};

use lance_table_runtime::commit as runtime;

mod hooks;

pub(crate) use hooks::LanceCommitHook;
pub use lance_table_runtime::namespace_manifest;
pub(crate) use runtime::TransactionReadSource;
pub use runtime::manifest_needs_migration;
pub(crate) use runtime::{CommitTransactionOptions, detect_overlapping_fragments, fix_schema};
#[cfg(test)]
pub(crate) use runtime::{read_transaction_file, write_transaction_file};

pub mod conflict_resolver;

pub(crate) async fn commit_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    options: CommitTransactionOptions<'_>,
) -> Result<(Manifest, ManifestLocation)> {
    let hook = LanceCommitHook::new(dataset.session.clone());
    Box::pin(runtime::commit_transaction(
        &dataset.table_view().to_table(),
        object_store,
        commit_handler,
        transaction,
        options,
        hook.hooks(),
    ))
    .await
}

pub(crate) async fn migrate_fragments(
    dataset: &Dataset,
    fragments: &[Fragment],
    recompute_stats: bool,
) -> Result<Vec<Fragment>> {
    runtime::migrate_fragments(&dataset.table_view().to_table(), fragments, recompute_stats).await
}

pub(crate) async fn load_and_sort_new_transactions(
    dataset: &Dataset,
    source: TransactionReadSource,
) -> Result<(Dataset, Vec<(u64, Arc<Transaction>)>)> {
    let (table, transactions) =
        runtime::load_and_sort_new_transactions(&dataset.table_view().to_table(), source).await?;
    let dataset = Dataset::from_table_with_index_cache(
        table,
        dataset.session.clone(),
        dataset.index_cache.clone(),
    );
    Ok((dataset, transactions))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_new_dataset(
    object_store: &ObjectStore,
    source_store: Option<&ObjectStore>,
    commit_handler: &Arc<dyn CommitHandler>,
    base_path: &Path,
    uri: &str,
    transaction: &Transaction,
    build_config: &ManifestBuildConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    metadata_cache: &crate::session::caches::DSMetadataCache,
    session: Arc<Session>,
    inherited_field_ids: bool,
) -> Result<(Manifest, ManifestLocation)> {
    let hook = LanceCommitHook::new(session.clone());
    runtime::commit_new_dataset(
        object_store,
        source_store,
        commit_handler,
        base_path,
        uri,
        transaction,
        build_config,
        manifest_naming_scheme,
        metadata_cache,
        session.table_session.clone(),
        inherited_field_ids,
        hook.hooks(),
    )
    .await
}

/// Smaller threshold so tests can exercise transaction spill with a small payload.
#[cfg(test)]
pub(crate) const MAX_INLINE_TRANSACTION_BYTES: usize = 64 * 1024;

#[cfg(test)]
mod conflict_matrix;
#[cfg(all(feature = "dynamodb_tests", test))]
mod dynamodb;
#[cfg(test)]
mod external_manifest;
#[cfg(all(feature = "dynamodb", test))]
mod s3_ddb_profile_test;
#[cfg(all(feature = "dynamodb_tests", test))]
mod s3_test;
#[cfg(test)]
mod tests;
