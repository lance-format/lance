// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lance_core::utils::tracing::{DATASET_COMMITTED_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{Error, Result};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_select::RowAddrTreeMap;
use lance_table::format::{
    ClusteringDefinition, DataStorageFormat, ManifestBuildConfig, is_detached_version,
};
use lance_table::io::commit::{
    CommitConfig, CommitHandler, ManifestNamingScheme, commit_handler_from_url,
};
use lance_table::transaction::{Operation, Transaction};
use object_store::path::Path;
use tracing::info;

use crate::branch_location::BranchLocation;
use crate::refs::Refs;
use crate::{Table, TableBuilder, TableSession, TableState};

use super::conflict_resolver::AppendIdempotencyKey;
use super::detached::commit_detached_transaction;
use super::retry::default_commit_retry_timeout;
use super::{
    CommitHooks, CommitIndexHook, CommitLifecycleHook, CommitTransactionOptions,
    commit_new_dataset, commit_transaction,
};

/// The table snapshot or URI to commit to.
#[derive(Debug, Clone)]
pub enum CommitDestination<'a> {
    /// An existing dataset to write to.
    Table(Arc<Table>),
    /// A URI to write to.
    Uri(&'a str),
}

impl CommitDestination<'_> {
    pub(crate) fn table(&self) -> Option<&Table> {
        match self {
            CommitDestination::Table(dataset) => Some(dataset.as_ref()),
            CommitDestination::Uri(_) => None,
        }
    }

    pub(crate) fn uri(&self) -> String {
        match self {
            CommitDestination::Table(dataset) => dataset.state.uri.clone(),
            CommitDestination::Uri(uri) => uri.to_string(),
        }
    }
}

impl From<Table> for CommitDestination<'_> {
    fn from(table: Table) -> Self {
        Self::Table(Arc::new(table))
    }
}

impl From<&Table> for CommitDestination<'_> {
    fn from(table: &Table) -> Self {
        Self::from(table.clone())
    }
}

impl From<Arc<Table>> for CommitDestination<'_> {
    fn from(dataset: Arc<Table>) -> Self {
        CommitDestination::Table(dataset)
    }
}

impl<'a> From<&'a str> for CommitDestination<'a> {
    fn from(uri: &'a str) -> Self {
        CommitDestination::Uri(uri)
    }
}

impl<'a> From<&'a String> for CommitDestination<'a> {
    fn from(uri: &'a String) -> Self {
        CommitDestination::Uri(uri.as_str())
    }
}

impl<'a> From<&'a Path> for CommitDestination<'a> {
    fn from(path: &'a Path) -> Self {
        CommitDestination::Uri(path.as_ref())
    }
}

/// Create a new commit from a [`Transaction`].
///
/// The transaction describes changes to a table, including any prewritten fragments.
#[derive(Clone)]
pub struct CommitBuilder<'a> {
    dest: CommitDestination<'a>,
    index_hook: Option<Arc<dyn CommitIndexHook>>,
    use_stable_row_ids: Option<bool>,
    enable_v2_manifest_paths: bool,
    manifest_config: ManifestBuildConfig,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_params: Option<ObjectStoreParams>,
    object_store: Option<Arc<ObjectStore>>,
    source_store: Option<Arc<ObjectStore>>,
    session: Option<Arc<TableSession>>,
    detached: bool,
    commit_config: CommitConfig,
    retry_timeout: Duration,
    affected_rows: Option<RowAddrTreeMap>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    append_idempotency_property: Option<String>,
    timeout: Option<Duration>,
    /// Set only by `Dataset::deep_clone`, after it has copied the source files.
    deep_clone_files_copied: bool,
    inherited_field_ids: bool,
    /// Whether the caller has opted in to committing a composite operation.
    allow_composite_operations: bool,
}

/// Default timeout applied to [`CommitBuilder::execute`] when none is set.
pub const DEFAULT_COMMIT_TIMEOUT: Duration = Duration::from_secs(1800);

impl<'a> CommitBuilder<'a> {
    /// Build a commit for a table snapshot or a table URI.
    pub fn new(dest: impl Into<CommitDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            index_hook: None,
            use_stable_row_ids: None,
            enable_v2_manifest_paths: true,
            manifest_config: ManifestBuildConfig::default(),
            commit_handler: None,
            store_params: None,
            object_store: None,
            source_store: None,
            session: None,
            detached: false,
            commit_config: Default::default(),
            retry_timeout: default_commit_retry_timeout(),
            affected_rows: None,
            transaction_properties: None,
            append_idempotency_property: None,
            timeout: Some(DEFAULT_COMMIT_TIMEOUT),
            deep_clone_files_copied: false,
            inherited_field_ids: false,
            allow_composite_operations: false,
        }
    }

    /// Whether to use stable row ids. A row then keeps the same `_rowid` for
    /// its lifetime, even when compaction, an update or a merge insert
    /// rewrites it.
    ///
    /// This is only used for new datasets. Existing datasets will use their
    /// existing setting.
    ///
    /// **Default is false.**
    pub fn use_stable_row_ids(mut self, use_stable_row_ids: bool) -> Self {
        self.use_stable_row_ids = Some(use_stable_row_ids);
        self
    }

    /// Pass the storage format to use for the dataset.
    ///
    /// On creation, this sets the default storage version. If omitted, the version
    /// is inferred from homogeneous data files, or uses the stable version for an
    /// empty table. Creating from mixed prewritten files requires an explicit default.
    ///
    /// For an existing dataset, this only sets the manifest default on overwrite;
    /// other operations preserve the existing default. If
    /// prewritten fragments introduce another exact V2 version, the commit
    /// derives the required mixed-version capability from the final manifest.
    pub fn with_storage_format(mut self, storage_format: LanceFileVersion) -> Self {
        self.manifest_config = self
            .manifest_config
            .with_storage_format(DataStorageFormat::new(storage_format.resolve()));

        self
    }

    pub fn with_exact_storage_format(mut self, storage_format: ConcreteFileVersion) -> Self {
        self.manifest_config = self
            .manifest_config
            .with_storage_format(DataStorageFormat::new(storage_format));
        self
    }

    /// Pass an object store to use.
    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.object_store = Some(object_store);
        self
    }

    /// Pass the object store of the dataset being cloned from.
    ///
    /// Only used by `Operation::Clone`: the source manifest is read through this store
    /// while the new dataset is written through the destination store. This lets a clone
    /// cross object stores/accounts (e.g. between two Azure accounts), where the source
    /// is not reachable with the destination's credentials. Defaults to the destination
    /// store when not set, preserving same-store behavior.
    pub fn with_source_store(mut self, source_store: Arc<ObjectStore>) -> Self {
        self.source_store = Some(source_store);
        self
    }

    /// Pass a commit handler to use for the dataset.
    ///
    /// Takes precedence over the destination dataset's own handler. If not
    /// set, a `Dataset` destination commits through its own handler and a
    /// `Uri` destination resolves one from the uri.
    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Commit the complete list of clustering definitions this writer wants
    /// the table to hold, planned against the manifest it read. Only an
    /// `UpdateConfig` may carry it: the list rides alongside the metadata it
    /// publishes and the commit stays a plain `UpdateConfig`; the manifest is
    /// the durable record of the definitions. Any other operation refuses this
    /// input, and so does a detached commit.
    ///
    /// The list is applied against the landing manifest on every attempt: a
    /// definition allocated or removed underneath it, or one whose identity
    /// changed, refuses the commit so the writer plans again. Every definition
    /// the list allocates (each id above the read manifest's allocation mark,
    /// not only the highest) is checked against the landing schema from the
    /// schema the writer read: every key field must still be the same typed
    /// field by id (same logical type along its path and in its subtree, a
    /// timestamp's unit and timezone included). A same-id rename keeps a key;
    /// a retype, a drop or a cast under a new id refuses the commit, and with
    /// it every other allocation in the list. Retained definitions are not
    /// checked.
    ///
    /// Definitions key on schema field ids, so allocating the first one on a
    /// table with reusable field ids activates non-reusable field ids in the
    /// same commit (see the non-reusable field ID contract): a
    /// dropped key field's id is never handed to a later column, and an
    /// overwrite assigns ids above the mark. Until that activation lands, a
    /// stricter fence in the conflict resolver refuses concurrent schema
    /// replacements that could reassign a key's id; it compares names too and
    /// is deliberately separate from the typed check above. Activation is
    /// permanent: clearing the definitions does not undo it, and under that
    /// contract the versions from before the activating commit can no longer
    /// be restored (they stay readable). A table already under the contract is
    /// unaffected.
    pub fn with_clustering_definitions(mut self, definitions: Vec<ClusteringDefinition>) -> Self {
        self.manifest_config = self
            .manifest_config
            .with_clustering_definitions(definitions);
        self
    }

    /// Pass store parameters to use for the dataset.
    ///
    /// If an object store is passed, these parameters will be ignored.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.store_params = Some(store_params);
        self
    }

    /// Pass a session to use for the dataset.
    ///
    /// If a session is not passed, but a dataset is used as the destination,
    /// then the dataset's session will be used.
    ///
    /// By passing a session or re-using a dataset, you can re-use the
    /// file metadata and index caches, which can significantly improve
    /// performance.
    pub fn with_session(mut self, session: Arc<TableSession>) -> Self {
        self.session = Some(session);
        self
    }

    ///  If set to true, and this is a new dataset, uses the new v2 manifest
    ///  paths. These allow constant-time lookups for the latest manifest on object storage.
    ///  This parameter has no effect on existing datasets. To migrate an existing
    ///  dataset, use the manifest path migration. **Default is True.**
    ///
    /// <div class="warning">
    ///  WARNING: turning this on will make the dataset unreadable for older
    ///  versions of Lance (prior to 0.17.0).
    /// </div>
    pub fn enable_v2_manifest_paths(mut self, enable: bool) -> Self {
        self.enable_v2_manifest_paths = enable;
        self
    }

    /// Commit a version that is not part of the mainline history.
    ///
    /// This commit will never show up in the dataset's history.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets
    /// whose lineage is tracked elsewhere.
    pub fn with_detached(mut self, detached: bool) -> Self {
        self.detached = detached;
        self
    }

    /// Set the maximum number of retries for commit operations.
    ///
    /// If a commit operation fails, it will be retried up to `max_retries` times.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.commit_config.num_retries = max_retries;
        self
    }

    /// Set the wall-clock budget used by commit conflict backoff.
    ///
    /// The first commit attempt is always allowed to complete. If it conflicts,
    /// each backoff sleep is bounded by the time remaining in this budget. The
    /// default is 30 seconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use lance_table_runtime::CommitBuilder;
    ///
    /// let _builder = CommitBuilder::new("memory://dataset")
    ///     .with_retry_timeout(Duration::from_secs(10));
    /// ```
    pub fn with_retry_timeout(mut self, retry_timeout: Duration) -> Self {
        self.retry_timeout = retry_timeout;
        self
    }

    pub fn with_skip_auto_cleanup(mut self, skip_auto_cleanup: bool) -> Self {
        self.commit_config.skip_auto_cleanup = skip_auto_cleanup;
        self
    }

    /// Provide the set of row addresses that were deleted or updated. This is
    /// used to perform fast conflict resolution.
    pub fn with_affected_rows(mut self, affected_rows: RowAddrTreeMap) -> Self {
        self.affected_rows = Some(affected_rows);
        self
    }

    /// Set a timeout for the commit operation.
    ///
    /// The timeout bounds the *entire* [`Self::execute`] / [`Self::execute_batch`]
    /// call, including all conflict retries — it is not applied per attempt.
    /// Pass `None` to disable the timeout entirely.
    ///
    /// The default is 30 minutes (see [`DEFAULT_COMMIT_TIMEOUT`]).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if `timeout` is `Some(Duration::ZERO)` (raised
    ///   when [`Self::execute`] is called, not here).
    /// - [`Error::Timeout`] if the operation does not complete within the
    ///   timeout.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// provide Configuration key-value pairs associated with this transaction.
    /// This is used to store metadata about the transaction, such as commit messages, engine information, etc.
    /// this properties map will be persisted as a part of the transaction object
    pub fn with_transaction_properties(
        mut self,
        transaction_properties: HashMap<String, String>,
    ) -> Self {
        self.transaction_properties = Some(Arc::new(transaction_properties));
        self
    }

    /// Deduplicate an [`Operation::Append`] by a property in
    /// [`Transaction::transaction_properties`]. The property value must be nonempty.
    ///
    /// Use the same property name, value and original `read_version` on every
    /// submission of a logical append. If an append with the same property value
    /// has already committed on this branch after `read_version`, the call returns
    /// an error with [`Error::append_already_committed_version`] and publishes no
    /// version. The first append wins even when the staged fragments differ. It
    /// may also reference the same staged files. Use distinct values for distinct
    /// appends, and retain the base version and intervening history for as long
    /// as the append can be retried. If there is a gap and no known match, the
    /// commit fails.
    ///
    /// A committed append is matched by its stored property, including appends
    /// written before this option was introduced. Every writer submitting the
    /// logical append concurrently must use the option. This is for individual,
    /// non-detached appends to an existing dataset.
    ///
    /// ```
    /// # use lance_table_runtime::CommitBuilder;
    /// let builder = CommitBuilder::new("memory://").with_append_idempotency_property("job_id");
    /// ```
    pub fn with_append_idempotency_property(mut self, property: impl Into<String>) -> Self {
        self.append_idempotency_property = Some(property.into());
        self
    }

    /// Configure this commit as the second step of a stable row ID migration.
    ///
    /// Sets `use_stable_row_ids = true` and supplies the `next_row_id` that was
    /// computed during the first migration commit. This bypasses the normal
    /// "cannot enable stable row IDs on an existing dataset" check so that the
    /// flag can be activated without creating the dataset from scratch.
    pub fn with_stable_row_id_migration_activation(mut self, next_row_id: u64) -> Self {
        self.manifest_config = self.manifest_config.with_migration_next_row_id(next_row_id);
        self
    }

    pub fn with_non_reusable_field_id_migration_activation(mut self) -> Self {
        self.manifest_config = self
            .manifest_config
            .with_activate_non_reusable_field_ids(true);
        self
    }

    /// Mark this commit as the last step of the host's deep clone, which has
    /// already copied the source's data, deletion and index files to the
    /// destination.
    ///
    /// A deep `Operation::Clone` points every file at the destination but the
    /// commit itself copies nothing, so it is rejected unless this is set.
    pub fn with_deep_clone_files_copied(mut self) -> Self {
        self.deep_clone_files_copied = true;
        self
    }

    /// Create the table keeping an existing table's field IDs.
    pub fn with_inherited_field_ids(mut self) -> Self {
        self.inherited_field_ids = true;
        self
    }

    /// Opt in to committing an action-based transaction.
    ///
    /// Transaction V2 -- a [`Transaction`] whose operation is a
    /// [`CompositeOperation`](lance_table::transaction::action::CompositeOperation)
    /// -- is experimental for now: this API may change, and not every
    /// operation can be expressed as actions yet.
    /// The format it writes is not experimental.
    ///
    /// Without this opt-in, [`Self::execute`] rejects such a transaction with
    /// [`Error::NotSupported`]. The opt-in has no effect on any other operation.
    ///
    /// # Compatibility
    ///
    /// Any reader can open and scan a dataset version committed this way: the
    /// inline transaction lives in a manifest field older readers ignore
    /// ([#9482]). Lance before v14.0.0 fails only when it decodes the
    /// transaction itself, to read the transaction history or to check a
    /// concurrent commit for conflicts. For conflict checking, that failure
    /// aborts the older writer's commit, which is the safe outcome.
    ///
    /// [#9482]: https://github.com/lance-format/lance/pull/9482
    ///
    /// ```
    /// # use lance_table_runtime::CommitBuilder;
    /// # use lance_table_runtime::Table;
    /// # use lance_core::Result;
    /// # use lance_table::transaction::Transaction;
    /// # use std::sync::Arc;
    /// # async fn commit(dataset: Arc<Table>, transaction: Transaction) -> Result<Table> {
    /// CommitBuilder::new(dataset)
    ///     .with_experimental_composite_operations(true)
    ///     .execute(transaction)
    ///     .await
    /// # }
    /// ```
    pub fn with_experimental_composite_operations(mut self, allow: bool) -> Self {
        self.allow_composite_operations = allow;
        self
    }

    /// Use the supplied user index implementation during this commit.
    pub fn with_index_hook(mut self, hook: Arc<dyn CommitIndexHook>) -> Self {
        self.index_hook = Some(hook);
        self
    }

    /// Publish the transaction and return the committed snapshot.
    pub async fn execute(self, transaction: Transaction) -> Result<Table> {
        self.execute_with_lifecycle(transaction, None).await
    }

    #[doc(hidden)]
    pub async fn execute_with_lifecycle(
        self,
        transaction: Transaction,
        lifecycle: Option<&dyn CommitLifecycleHook>,
    ) -> Result<Table> {
        if matches!(transaction.operation, Operation::CompositeOperation(_))
            && !self.allow_composite_operations
        {
            return Err(Error::not_supported_source(
                "Transaction V2 is experimental; a CompositeOperation can only be \
                 committed after opting in with \
                 CommitBuilder::with_experimental_composite_operations"
                    .into(),
            ));
        }
        let timeout = self.timeout;
        if let Some(t) = timeout
            && t.is_zero()
        {
            return Err(Error::invalid_input(
                "CommitBuilder timeout must be non-zero; pass `None` to disable",
            ));
        }
        // Box the inner future so wrapping it in `tokio::time::Timeout` does
        // not deepen the future type — downstream `async fn`s that await
        // `execute` otherwise hit the compiler's layout-query depth limit.
        let fut = Box::pin(self.execute_inner(transaction, lifecycle));
        match timeout {
            Some(t) => match tokio::time::timeout(t, fut).await {
                Ok(res) => res,
                Err(_) => Err(Error::timeout(format!(
                    "Commit timed out after {:?}. Increase the timeout via \
                     CommitBuilder::with_timeout or pass `None` to disable.",
                    t
                ))),
            },
            None => fut.await,
        }
    }

    async fn execute_inner(
        self,
        transaction: Transaction,
        lifecycle: Option<&dyn CommitLifecycleHook>,
    ) -> Result<Table> {
        let hooks = CommitHooks {
            index: self.index_hook.as_deref(),
            lifecycle,
        };
        let append_idempotency_key = if let Some(property) =
            self.append_idempotency_property.as_deref()
        {
            if self.detached
                || transaction.read_version == 0
                || is_detached_version(transaction.read_version)
                || !matches!(transaction.operation, Operation::Append { .. })
            {
                return Err(Error::invalid_input(
                    "append idempotency requires a non-detached Append based on an existing version",
                ));
            }
            let value = transaction
                .transaction_properties
                .as_deref()
                .and_then(|properties| properties.get(property))
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "append idempotency requires a nonempty transaction property {property:?}"
                    ))
                })?;
            Some(AppendIdempotencyKey { property, value })
        } else {
            None
        };

        // A definitions list is a commit-time input for UpdateConfig alone. Every
        // other operation is refused here, before any path is chosen: the
        // new-dataset paths never reach the manifest builder's own check.
        if self.manifest_config.clustering_definitions().is_some()
            && !matches!(transaction.operation, Operation::UpdateConfig { .. })
        {
            return Err(Error::invalid_input(
                "a complete clustering definitions list can only be committed with UpdateConfig",
            ));
        }
        if matches!(
            transaction.operation,
            Operation::Clone {
                is_shallow: false,
                ..
            }
        ) && !self.deep_clone_files_copied
        {
            return Err(Error::invalid_input(
                "A deep Clone cannot be committed directly: the commit does not copy the \
                 source files, so the new dataset would reference files that do not exist. \
                 Use deep_clone instead.",
            ));
        }

        let session = self
            .session
            .or_else(|| self.dest.table().map(|ds| ds.session()))
            .unwrap_or_default();

        // Store used to read the source manifest for a clone (see with_source_store).
        let source_store = self.source_store.clone();

        let (object_store, base_path, commit_handler) = match &self.dest {
            CommitDestination::Table(dataset) => (
                dataset.state.object_store.clone(),
                dataset.state.base.clone(),
                self.commit_handler
                    .clone()
                    .unwrap_or_else(|| dataset.state.commit_handler.clone()),
            ),
            CommitDestination::Uri(uri) => {
                let commit_handler = if let (Some(_), Some(commit_handler)) =
                    (&self.object_store, &self.commit_handler)
                {
                    commit_handler.clone()
                } else {
                    resolve_commit_handler(uri, self.commit_handler.clone(), &self.store_params)
                        .await?
                };
                let (object_store, base_path) = if let Some(passed_store) = self.object_store {
                    (
                        passed_store,
                        ObjectStore::extract_path_from_uri(session.store_registry(), uri)?,
                    )
                } else {
                    ObjectStore::from_uri_and_params(
                        session.store_registry(),
                        uri,
                        &self.store_params.clone().unwrap_or_default(),
                    )
                    .await?
                };
                (object_store, base_path, commit_handler)
            }
        };

        let dest = match &self.dest {
            CommitDestination::Table(dataset) => CommitDestination::Table(dataset.clone()),
            CommitDestination::Uri(uri) => {
                // Check if it already exists.
                let mut builder = TableBuilder::from_uri(uri)
                    .with_store_params(self.store_params.clone().unwrap_or_default())
                    .with_session(session.clone());
                if let Some(handler) = &self.commit_handler {
                    builder = builder.with_commit_handler(handler.clone());
                }

                // If we are using a detached version, we need to load the dataset.
                // Otherwise, we are writing to the main history, and need to check
                // out the latest version.
                if is_detached_version(transaction.read_version) {
                    builder = builder.with_version(transaction.read_version)
                }

                match builder.load().await {
                    Ok(dataset) => CommitDestination::Table(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        CommitDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        // A clone creates a new dataset. The destination may have appeared
        // after the caller's preflight; never rebase a clone onto its winner.
        if dest.table().is_some() && matches!(transaction.operation, Operation::Clone { .. }) {
            return Err(Error::dataset_already_exists(dest.uri()));
        }

        if dest.table().is_none()
            && !matches!(
                transaction.operation,
                Operation::Overwrite { .. } | Operation::Clone { .. }
            )
        {
            return Err(Error::dataset_not_found(
                base_path.to_string(),
                "The dataset must already exist unless the operation is Overwrite".into(),
            ));
        }

        let metadata_cache = match &dest {
            CommitDestination::Table(table) => table.state.metadata_cache.clone(),
            CommitDestination::Uri(uri) => Arc::new(session.metadata_cache().for_dataset(uri)),
        };

        let manifest_naming_scheme = if let Some(ds) = dest.table() {
            ds.state.manifest_location.naming_scheme
        } else if self.enable_v2_manifest_paths {
            ManifestNamingScheme::V2
        } else {
            ManifestNamingScheme::V1
        };

        let manifest_config = if self.manifest_config.migration_next_row_id().is_some() {
            // Migration activation always enables stable row IDs regardless of
            // the current dataset state (already implied by `with_migration_next_row_id`).
            self.manifest_config
        } else {
            let use_stable_row_ids = match dest.table() {
                Some(ds) => ds.manifest.uses_stable_row_ids(),
                None => self.use_stable_row_ids.unwrap_or(false),
            };
            self.manifest_config
                .with_use_stable_row_ids(use_stable_row_ids)
        };

        let (manifest, manifest_location) = if let Some(dataset) = dest.table() {
            if self.detached {
                if matches!(manifest_naming_scheme, ManifestNamingScheme::V1) {
                    return Err(Error::not_supported_source(
                        "detached commits cannot be used with v1 manifest paths".into(),
                    ));
                }
                commit_detached_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    CommitTransactionOptions::new(
                        &manifest_config,
                        &self.commit_config,
                        manifest_naming_scheme,
                    )
                    .with_retry_timeout(self.retry_timeout),
                    hooks,
                )
                .await?
            } else {
                commit_transaction(
                    dataset,
                    object_store.as_ref(),
                    commit_handler.as_ref(),
                    &transaction,
                    CommitTransactionOptions::new(
                        &manifest_config,
                        &self.commit_config,
                        manifest_naming_scheme,
                    )
                    .with_retry_timeout(self.retry_timeout)
                    .with_affected_rows(self.affected_rows.as_ref())
                    .with_append_idempotency_key(append_idempotency_key),
                    hooks,
                )
                .await?
            }
        } else if self.detached {
            // I think we may eventually want this, and we can probably handle it, but leaving a TODO for now
            return Err(Error::not_supported_source(
                "detached commits cannot currently be used to create new datasets".into(),
            ));
        } else {
            commit_new_dataset(
                object_store.as_ref(),
                source_store.as_deref(),
                &commit_handler,
                &base_path,
                &dest.uri(),
                &transaction,
                &manifest_config,
                manifest_naming_scheme,
                metadata_cache.as_ref(),
                session.clone(),
                self.inherited_field_ids,
                hooks,
            )
            .await?
        };

        info!(
            target: TRACE_DATASET_EVENTS,
            event=DATASET_COMMITTED_EVENT,
            uri=dest.uri(),
            read_version=transaction.read_version,
            committed_version=manifest.version,
            detached=self.detached,
            operation=&transaction.operation.name()
        );

        match &self.dest {
            CommitDestination::Table(dataset) => {
                let mut committed = dataset.as_ref().clone();
                committed.set_manifest(Arc::new(manifest), manifest_location);
                committed.state.session = session;
                Ok(committed)
            }
            CommitDestination::Uri(uri) => {
                let refs = Refs::new(
                    object_store.clone(),
                    commit_handler.clone(),
                    BranchLocation {
                        path: base_path.clone(),
                        uri: uri.to_string(),
                        branch: manifest.branch.clone(),
                    },
                );
                let branch_identifier = if let Some(dataset) = dest.table() {
                    dataset.view().ensure_branch_exists().await?;
                    dataset.state.branch_identifier.clone()
                } else {
                    Arc::new(
                        refs.branches()
                            .get_identifier(manifest.branch.as_deref())
                            .await?,
                    )
                };
                let mut table_state = TableState::new(
                    object_store,
                    base_path,
                    uri.to_string(),
                    &manifest,
                    manifest_location,
                    session,
                    metadata_cache,
                    commit_handler,
                    branch_identifier,
                    None,
                    self.store_params.clone(),
                    None,
                );
                table_state.is_manifest_generation_verified = true;
                Ok(Table::from_parts(table_state, Arc::new(manifest), refs))
            }
        }
    }

    /// Commit a set of transactions as a single new version.
    ///
    /// <div class="warning">
    ///   Only works for append transactions right now. Other kinds of transactions
    ///   will be supported in the future.
    /// </div>
    pub async fn execute_batch(self, transactions: Vec<Transaction>) -> Result<BatchCommitResult> {
        self.execute_batch_with_lifecycle(transactions, None).await
    }

    #[doc(hidden)]
    pub async fn execute_batch_with_lifecycle(
        self,
        transactions: Vec<Transaction>,
        lifecycle: Option<&dyn CommitLifecycleHook>,
    ) -> Result<BatchCommitResult> {
        if self.append_idempotency_property.is_some() {
            return Err(Error::invalid_input(
                "append idempotency requires an individual Append transaction",
            ));
        }
        if transactions.is_empty() {
            return Err(Error::invalid_input_source(
                "No transactions to commit".into(),
            ));
        }
        if transactions
            .iter()
            .any(|t| !matches!(t.operation, Operation::Append { .. }))
        {
            return Err(Error::not_supported_source(
                "Only append transactions are supported in batch commits".into(),
            ));
        }

        let read_version = transactions.iter().map(|t| t.read_version).min().unwrap();

        let merged = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: transactions
                    .iter()
                    .flat_map(|t| match &t.operation {
                        Operation::Append { fragments } => fragments.clone(),
                        _ => unreachable!(),
                    })
                    .collect(),
            },
            read_version,
            tag: None,
            transaction_properties: None,
        };
        let dataset = self
            .execute_with_lifecycle(merged.clone(), lifecycle)
            .await?;
        Ok(BatchCommitResult {
            table: dataset,
            merged,
        })
    }
}

/// The snapshot published by a batch commit and its combined transaction.
pub struct BatchCommitResult {
    pub table: Table,
    /// The final transaction that was committed.
    pub merged: Transaction,
    // TODO: Reject conflicts that need to be retried.
    // /// Transactions that were rejected due to conflicts.
    // pub rejected: Vec<Transaction>,
}

impl std::fmt::Debug for CommitBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitBuilder")
            .field("dest", &self.dest)
            .field("detached", &self.detached)
            .field("commit_config", &self.commit_config)
            .field("retry_timeout", &self.retry_timeout)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// Resolve the commit handler for a table URI, checking explicit store settings.
pub async fn resolve_commit_handler(
    uri: &str,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    store_options: &Option<ObjectStoreParams>,
) -> Result<Arc<dyn CommitHandler>> {
    match commit_handler {
        None => {
            #[allow(deprecated)]
            if store_options
                .as_ref()
                .map(|opts| opts.object_store.is_some())
                .unwrap_or_default()
            {
                return Err(Error::invalid_input(
                    "when creating a dataset with a custom object store the commit_handler must also be specified",
                ));
            }
            commit_handler_from_url(uri, store_options).await
        }
        Some(commit_handler) => {
            if uri.starts_with("s3+ddb") {
                Err(Error::invalid_input(
                    "`s3+ddb://` scheme and custom commit handler are mutually exclusive",
                ))
            } else {
                Ok(commit_handler)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_timeout_default_is_thirty_minutes() {
        let builder = CommitBuilder::new("memory://default-timeout");
        assert_eq!(builder.timeout, Some(DEFAULT_COMMIT_TIMEOUT));
        assert_eq!(DEFAULT_COMMIT_TIMEOUT, Duration::from_secs(1800));
    }

    #[test]
    fn test_commit_retry_timeout_default_is_thirty_seconds() {
        let builder = CommitBuilder::new("memory://default-retry-timeout");
        assert_eq!(builder.retry_timeout, default_commit_retry_timeout());
        assert_eq!(
            super::super::retry::DEFAULT_COMMIT_RETRY_TIMEOUT,
            Duration::from_secs(30)
        );
    }
}
