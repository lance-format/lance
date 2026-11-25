// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_select::RowAddrTreeMap;
use lance_table::format::ClusteringDefinition;
use lance_table::io::commit::CommitHandler;
use lance_table::transaction::Transaction;
use lance_table_runtime::CommitBuilder as RuntimeCommitBuilder;

use crate::io::commit::LanceCommitHook;
use crate::session::Session;
use crate::{Dataset, Result};

use super::WriteDestination;

/// Create a new commit from a [`Transaction`].
///
/// Transactions can be created using a write method like [`super::InsertBuilder::execute_uncommitted`].
#[derive(Debug, Clone)]
pub struct CommitBuilder<'a> {
    inner: RuntimeCommitBuilder<'a>,
    dataset: Option<Arc<Dataset>>,
    session: Option<Arc<Session>>,
}

pub use lance_table_runtime::commit::DEFAULT_COMMIT_TIMEOUT;

impl<'a> CommitBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        match dest.into() {
            WriteDestination::Dataset(dataset) => Self {
                inner: RuntimeCommitBuilder::new(Arc::new(dataset.table_view().to_table())),
                dataset: Some(dataset),
                session: None,
            },
            WriteDestination::Uri(uri) => Self {
                inner: RuntimeCommitBuilder::new(uri),
                dataset: None,
                session: None,
            },
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
        self.inner = self.inner.use_stable_row_ids(use_stable_row_ids);
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
        self.inner = self.inner.with_storage_format(storage_format);
        self
    }

    pub(crate) fn with_exact_storage_format(mut self, storage_format: ConcreteFileVersion) -> Self {
        self.inner = self.inner.with_exact_storage_format(storage_format);
        self
    }

    /// Pass an object store to use.
    pub fn with_object_store(mut self, object_store: Arc<ObjectStore>) -> Self {
        self.inner = self.inner.with_object_store(object_store);
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
        self.inner = self.inner.with_source_store(source_store);
        self
    }

    /// Pass the object store of the dataset being cloned from.
    ///
    /// The source dataset's commit handler is not used.
    #[deprecated(since = "12.0.0-beta.12", note = "use with_source_store instead")]
    pub fn with_source_dataset(self, source: &Dataset) -> Self {
        self.with_source_store(source.table_state.object_store.clone())
    }

    /// Pass a commit handler to use for the dataset.
    ///
    /// Takes precedence over the destination dataset's own handler. If not
    /// set, a `Dataset` destination commits through its own handler and a
    /// `Uri` destination resolves one from the uri.
    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.inner = self.inner.with_commit_handler(commit_handler);
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
    /// same commit (see [`Dataset::migrate_to_non_reusable_field_ids`]): a
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
        self.inner = self.inner.with_clustering_definitions(definitions);
        self
    }

    /// Pass store parameters to use for the dataset.
    ///
    /// If an object store is passed, these parameters will be ignored.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.inner = self.inner.with_store_params(store_params);
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
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    ///  If set to true, and this is a new dataset, uses the new v2 manifest
    ///  paths. These allow constant-time lookups for the latest manifest on object storage.
    ///  This parameter has no effect on existing datasets. To migrate an existing
    ///  dataset, use the [`Dataset::migrate_manifest_paths_v2`] method. **Default is True.**
    ///
    /// <div class="warning">
    ///  WARNING: turning this on will make the dataset unreadable for older
    ///  versions of Lance (prior to 0.17.0).
    /// </div>
    pub fn enable_v2_manifest_paths(mut self, enable: bool) -> Self {
        self.inner = self.inner.enable_v2_manifest_paths(enable);
        self
    }

    /// Commit a version that is not part of the mainline history.
    ///
    /// This commit will never show up in the dataset's history.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets
    /// whose lineage is tracked elsewhere.
    pub fn with_detached(mut self, detached: bool) -> Self {
        self.inner = self.inner.with_detached(detached);
        self
    }

    /// Set the maximum number of retries for commit operations.
    ///
    /// If a commit operation fails, it will be retried up to `max_retries` times.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.inner = self.inner.with_max_retries(max_retries);
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
    /// use lance::dataset::CommitBuilder;
    ///
    /// let _builder = CommitBuilder::new("memory://dataset")
    ///     .with_retry_timeout(Duration::from_secs(10));
    /// ```
    pub fn with_retry_timeout(mut self, retry_timeout: Duration) -> Self {
        self.inner = self.inner.with_retry_timeout(retry_timeout);
        self
    }

    pub fn with_skip_auto_cleanup(mut self, skip_auto_cleanup: bool) -> Self {
        self.inner = self.inner.with_skip_auto_cleanup(skip_auto_cleanup);
        self
    }

    /// Provide the set of row addresses that were deleted or updated. This is
    /// used to perform fast conflict resolution.
    pub fn with_affected_rows(mut self, affected_rows: RowAddrTreeMap) -> Self {
        self.inner = self.inner.with_affected_rows(affected_rows);
        self
    }

    /// Supply the insert count reported by the same uncommitted MERGE.
    /// A zero count lets the resolver distinguish updates from unencoded insert keys.
    ///
    /// ```
    /// # use lance::{Dataset, Result, dataset::{CommitBuilder, write::merge_insert::UncommittedMergeInsert}};
    /// # async fn commit(uri: &str, staged: UncommittedMergeInsert) -> Result<Dataset> {
    /// let mut builder = CommitBuilder::new(uri).with_merge_inserted_rows(staged.stats.num_inserted_rows);
    /// if let Some(rows) = staged.affected_rows {
    ///     builder = builder.with_affected_rows(rows);
    /// }
    /// builder.execute(staged.transaction).await
    /// # }
    /// ```
    pub fn with_merge_inserted_rows(mut self, count: u64) -> Self {
        self.inner = self.inner.with_merge_inserted_rows(count);
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
    /// - [`crate::Error::InvalidInput`] if `timeout` is `Some(Duration::ZERO)` (raised
    ///   when [`Self::execute`] is called, not here).
    /// - [`crate::Error::Timeout`] if the operation does not complete within the
    ///   timeout.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.inner = self.inner.with_timeout(timeout);
        self
    }

    /// provide Configuration key-value pairs associated with this transaction.
    /// This is used to store metadata about the transaction, such as commit messages, engine information, etc.
    /// this properties map will be persisted as a part of the transaction object
    pub fn with_transaction_properties(
        mut self,
        transaction_properties: HashMap<String, String>,
    ) -> Self {
        self.inner = self
            .inner
            .with_transaction_properties(transaction_properties);
        self
    }

    /// Deduplicate an [`Operation::Append`](lance_table::transaction::Operation::Append) by a property in
    /// [`Transaction::transaction_properties`]. The property value must be nonempty.
    ///
    /// Use the same property name, value and original `read_version` on every
    /// submission of a logical append. If an append with the same property value
    /// has already committed on this branch after `read_version`, the call returns
    /// an error with [`crate::Error::append_already_committed_version`] and publishes no
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
    /// # use lance::dataset::CommitBuilder;
    /// let builder = CommitBuilder::new("memory://").with_append_idempotency_property("job_id");
    /// ```
    pub fn with_append_idempotency_property(mut self, property: impl Into<String>) -> Self {
        self.inner = self.inner.with_append_idempotency_property(property);
        self
    }

    /// Configure this commit as the second step of a stable row ID migration.
    ///
    /// Sets `use_stable_row_ids = true` and supplies the `next_row_id` that was
    /// computed during the first migration commit. This bypasses the normal
    /// "cannot enable stable row IDs on an existing dataset" check so that the
    /// flag can be activated without creating the dataset from scratch.
    pub(crate) fn with_stable_row_id_migration_activation(mut self, next_row_id: u64) -> Self {
        self.inner = self
            .inner
            .with_stable_row_id_migration_activation(next_row_id);
        self
    }

    pub(crate) fn with_non_reusable_field_id_migration_activation(mut self) -> Self {
        self.inner = self.inner.with_non_reusable_field_id_migration_activation();
        self
    }

    /// Mark this commit as the last step of [`Dataset::deep_clone`], which has
    /// already copied the source's data, deletion and index files to the
    /// destination.
    ///
    /// A deep `Operation::Clone` points every file at the destination but the
    /// commit itself copies nothing, so it is rejected unless this is set.
    pub(crate) fn with_deep_clone_files_copied(mut self) -> Self {
        self.inner = self.inner.with_deep_clone_files_copied();
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
    /// [`crate::Error::NotSupported`]. The opt-in has no effect on any other operation.
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
    /// # use lance::dataset::CommitBuilder;
    /// # use lance::{Dataset, Result};
    /// # use lance_table::transaction::Transaction;
    /// # use std::sync::Arc;
    /// # async fn commit(dataset: Arc<Dataset>, transaction: Transaction) -> Result<Dataset> {
    /// CommitBuilder::new(dataset)
    ///     .with_experimental_composite_operations(true)
    ///     .execute(transaction)
    ///     .await
    /// # }
    /// ```
    pub fn with_experimental_composite_operations(mut self, allow: bool) -> Self {
        self.inner = self.inner.with_experimental_composite_operations(allow);
        self
    }

    pub async fn execute(self, transaction: Transaction) -> Result<Dataset> {
        let (inner, hook, session) = self.into_runtime();
        let table =
            Box::pin(inner.execute_with_lifecycle(transaction, Some(hook.as_ref()))).await?;
        Ok(Dataset::from_committed_table(table, session))
    }

    fn into_runtime(self) -> (RuntimeCommitBuilder<'a>, Arc<LanceCommitHook>, Arc<Session>) {
        let destination_session = self.dataset.map(|dataset| dataset.session.clone());
        let session = self
            .session
            .or_else(|| destination_session.clone())
            .unwrap_or_default();
        let hook = Arc::new(LanceCommitHook::new(
            destination_session.unwrap_or_else(|| session.clone()),
        ));
        let inner = self
            .inner
            .with_session(session.table_session.clone())
            .with_index_hook(hook.clone());
        (inner, hook, session)
    }

    /// Commit a set of transactions as a single new version.
    ///
    /// <div class="warning">
    ///   Only works for append transactions right now. Other kinds of transactions
    ///   will be supported in the future.
    /// </div>
    pub async fn execute_batch(self, transactions: Vec<Transaction>) -> Result<BatchCommitResult> {
        let (inner, hook, session) = self.into_runtime();
        let result =
            Box::pin(inner.execute_batch_with_lifecycle(transactions, Some(hook.as_ref()))).await?;
        let dataset = Dataset::from_committed_table(result.table, session);
        Ok(BatchCommitResult {
            dataset,
            merged: result.merged,
        })
    }
}

pub struct BatchCommitResult {
    pub dataset: Dataset,
    /// The final transaction that was committed.
    pub merged: Transaction,
    // TODO: Reject conflicts that need to be retried.
    // /// Transactions that were rejected due to conflicts.
    // pub rejected: Vec<Transaction>,
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, RecordBatch};
    use arrow_array::record_batch;
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    use crate::Error;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_io::utils::CachedFileSize;
    use lance_io::{assert_io_eq, assert_io_gt};
    use lance_table::format::is_detached_version;
    use lance_table::format::{
        DataFile, Fragment, IndexMetadata, Manifest, Transaction as TableTransaction,
    };
    use lance_table::io::commit::{
        CommitError, ManifestLocation, ManifestNamingScheme, ManifestWriter,
    };
    use lance_table::transaction::Operation;
    use lance_table::transaction::resolve_arrow_field_ids;
    use std::time::Duration;

    use object_store::throttle::ThrottleConfig;

    use crate::utils::test::ThrottledStoreWrapper;

    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};

    use super::*;

    #[tokio::test]
    async fn raw_arrow_new_dataset_preserves_user_metadata() {
        let mut field =
            lance_core::datatypes::Field::new_arrow("a", DataType::Int32, false).unwrap();
        field.id = 42;
        let metadata = HashMap::from([("source".to_string(), "user metadata".to_string())]);
        let schema = lance_core::datatypes::Schema {
            fields: vec![field],
            metadata: metadata.clone(),
        };
        let mut transaction = Transaction::new(
            0,
            Operation::Overwrite {
                schema,
                fragments: vec![],
                config_upsert_values: None,
                initial_bases: None,
            },
            None,
        );
        resolve_arrow_field_ids(None, &mut transaction.operation).unwrap();
        let dataset = CommitBuilder::new("memory://")
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(dataset.schema().field("a").unwrap().id, 42);
        assert_eq!(dataset.schema().metadata, metadata);
        let committed = dataset.read_transaction().await.unwrap().unwrap();
        let Operation::Overwrite { schema, .. } = committed.operation else {
            panic!("expected Overwrite");
        };
        assert_eq!(schema.field("a").unwrap().id, 42);
        assert_eq!(schema.metadata, metadata);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn raw_arrow_merge_resolved_before_commit(#[values(false, true)] detached: bool) {
        let batch = record_batch!(("a", Int32, [1, 2]), ("b", Int32, [3, 4])).unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                max_rows_per_file: 1,
                ..Default::default()
            })
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        dataset.migrate_to_non_reusable_field_ids().await.unwrap();
        let mut raw_schema = dataset.schema().clone();
        for field in &mut raw_schema.fields {
            field.id += 10;
        }
        let mut transaction = Transaction::new(
            dataset.version().version,
            Operation::Merge {
                schema: raw_schema,
                fragments: dataset.manifest.fragments.as_ref().clone(),
                preserves_nullability: true,
            },
            None,
        );
        resolve_arrow_field_ids(Some(&dataset.manifest), &mut transaction.operation).unwrap();
        let committed = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(detached)
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(is_detached_version(committed.version().version), detached);
        assert_eq!(committed.schema(), dataset.schema());
        assert_eq!(committed.manifest.fragments.len(), 2);
        assert_eq!(committed.manifest.fragments, dataset.manifest.fragments);
        assert_eq!(committed.scan().try_into_batch().await.unwrap(), batch);
        let persisted = committed.read_transaction().await.unwrap().unwrap();
        let Operation::Merge { schema, .. } = persisted.operation else {
            panic!("expected Merge");
        };
        assert_eq!(&schema, dataset.schema());
        assert_eq!(schema.metadata, dataset.schema().metadata);
    }

    fn sample_fragment() -> Fragment {
        let (major_version, minor_version) =
            LanceFileVersion::Stable.resolve().to_data_file_numbers();

        Fragment {
            id: 0,
            files: vec![DataFile {
                path: "file.lance".to_string(),
                fields: Arc::from([0]),
                column_indices: Arc::from([0]),
                file_major_version: major_version,
                file_minor_version: minor_version,
                file_size_bytes: CachedFileSize::new(100),
                file_metadata_size_bytes: None,
                base_id: None,
            }],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(10),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    fn sample_transaction(read_version: u64) -> Transaction {
        Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Append {
                fragments: vec![sample_fragment()],
            },
            read_version,
            tag: None,
            transaction_properties: None,
        }
    }

    #[derive(Debug)]
    struct SlowConflictingCommitHandler;

    #[async_trait::async_trait]
    impl CommitHandler for SlowConflictingCommitHandler {
        fn is_version_not_found_definitive(&self) -> bool {
            true
        }

        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base_path: &object_store::path::Path,
            _object_store: &ObjectStore,
            _manifest_writer: ManifestWriter,
            _naming_scheme: ManifestNamingScheme,
            _transaction: Option<TableTransaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Err(CommitError::CommitConflict)
        }
    }

    #[tokio::test]
    async fn test_reuse_session() {
        // Need to use in-memory for accurate IOPS tracking.
        let session = Arc::new(Session::default());
        // Create new dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .with_params(&WriteParams {
                session: Some(session.clone()),
                enable_v2_manifest_paths: true,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        let io_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_gt!(io_stats, read_iops, 0);
        assert_io_gt!(io_stats, write_iops, 0);

        // Commit transaction 5 times
        for i in 0..5 {
            let new_ds = CommitBuilder::new(dataset.clone())
                .execute(sample_transaction(1))
                .await
                .unwrap();
            assert_eq!(new_ds.manifest.version, i + 2);

            // Because we are writing transactions sequentially, and caching them,
            // we shouldn't need to read anything from disk. Except we do need
            // to check for the latest version to see if we need to do conflict
            // resolution.
            let io_stats = dataset
                .table_state
                .object_store
                .as_ref()
                .io_stats_incremental();
            assert_io_eq!(io_stats, read_iops, 1, "check latest version, i = {} ", i);
            // Should see 2 IOPs:
            // 1. Write the transaction files
            // 2. Write (conditional put) the manifest
            // (the version hint is only written on non-lexically-ordered stores)
            assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest, i = {}", i);
        }

        // Commit transaction with URI and session
        let new_ds = CommitBuilder::new("memory://test")
            .with_session(dataset.session.clone())
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 7);
        // Session should still be re-used
        // However, the dataset needs to be loaded and the read version checked out.
        // The read version's manifest body is served from the session cache (it
        // was cached when v1 was first created), so the checkout only pays the
        // version-resolution head, not a manifest read.
        let io_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_eq!(io_stats, read_iops, 3, "load dataset + check version");
        assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest");

        // Commit transaction with URI and new session. Re-use the store
        // registry so we see the same store.
        let new_session = Arc::new(Session::new(0, 0, session.store_registry()));
        let new_ds = CommitBuilder::new("memory://test")
            .with_session(new_session)
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest().version, 8);
        // Now we have to load all previous transactions.

        let io_stats = dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert_io_gt!(io_stats, read_iops, 10);
        assert_io_eq!(io_stats, write_iops, 2, "write txn + manifest");
    }

    #[tokio::test]
    async fn test_commit_iops() {
        // If there's no conflicts, we should be able to commit in 2 io requests:
        // * write txn file (this could be optional one day)
        // * write manifest
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            session: Some(session.clone()),
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![0; 5]))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();

        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental(); // Reset the stats
        let read_version = dataset.manifest().version;
        let new_ds = CommitBuilder::new(Arc::new(dataset))
            .execute(sample_transaction(read_version))
            .await
            .unwrap();

        // Assert io requests
        let io_stats = new_ds
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        // This could be zero, if we decided to be optimistic. However, that
        // would mean wasted write requests (txn + manifest) if there was
        // a conflict. We choose to be pessimistic for more consistent performance.
        assert_io_eq!(io_stats, read_iops, 1);
        assert_io_eq!(io_stats, write_iops, 2);
        // We can't write them in parallel. The transaction file must exist before
        // we can write the manifest.
        assert_io_eq!(io_stats, num_stages, 3);
    }

    #[tokio::test]
    #[rstest::rstest]
    async fn test_commit_conflict_iops(#[values(true, false)] use_cache: bool) {
        let cache_size = if use_cache { 1_000_000 } else { 0 };
        let session = Arc::new(Session::new(0, cache_size, Default::default()));
        // We need throttled to correctly count num hops. Otherwise, memory store
        // returns synchronously, and each request is 1 hop.
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(5),
                wait_get_per_call: Duration::from_millis(5),
                wait_put_per_call: Duration::from_millis(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            session: Some(session.clone()),
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                DataType::Int32,
                false,
            )])),
            vec![Arc::new(Int32Array::from(vec![0; 5]))],
        )
        .unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();
        let original_dataset = Arc::new(dataset.clone());

        // Create 3 other transactions that happen concurrently.
        let num_other_txns = 3;
        for _ in 0..num_other_txns {
            dataset = CommitBuilder::new(original_dataset.clone())
                .execute(sample_transaction(dataset.manifest().version))
                .await
                .unwrap();
        }
        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        let new_ds = CommitBuilder::new(original_dataset.clone())
            .execute(sample_transaction(original_dataset.manifest().version))
            .await
            .unwrap();

        let io_stats = new_ds
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        // If there is a conflict with two transaction, the retry should require io requests:
        // * 1 list version
        // * num_other_txns read manifests (cache-able)
        // * num_other_txns read txn files (cache-able)
        // * 1 write txn file
        // * 1 write manifest
        // For total of 3 + 2 * num_other_txns io requests. If we have caching enabled, we can skip 2 * num_other_txns
        // of those. We should be able to read in 5 hops.
        if use_cache {
            assert_io_eq!(io_stats, read_iops, 1); // Just list versions
            assert_io_eq!(io_stats, num_stages, 3);
        } else {
            // We need to read the other manifests and transactions.

            use lance_io::assert_io_lt;
            assert_io_eq!(io_stats, read_iops, 1 + num_other_txns * 2);
            // It's possible to read the txns for some versions before we
            // finish reading later versions and so the entire "read versions
            // and txs" may appear as 1 hop instead of 2.
            assert_io_lt!(io_stats, num_stages, 6);
        }
        assert_io_eq!(io_stats, write_iops, 2); // txn + manifest
    }

    #[tokio::test]
    async fn test_commit_timeout_zero_rejected() {
        let dataset = Arc::new(
            InsertBuilder::new("memory://test")
                .execute(vec![
                    RecordBatch::try_new(
                        Arc::new(ArrowSchema::new(vec![ArrowField::new(
                            "i",
                            DataType::Int32,
                            false,
                        )])),
                        vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                    )
                    .unwrap(),
                ])
                .await
                .unwrap(),
        );
        let res = CommitBuilder::new(dataset.clone())
            .with_timeout(Some(Duration::ZERO))
            .execute(sample_transaction(1))
            .await;
        assert!(
            matches!(res, Err(Error::InvalidInput { .. })),
            "got {res:?}"
        );
    }

    #[tokio::test]
    async fn test_commit_deep_clone_requires_deep_clone_api() {
        let source_dir = TempStrDir::default();
        let source = InsertBuilder::new(source_dir.as_str())
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let clone_txn = |is_shallow| {
            Transaction::new(
                source.manifest.version,
                Operation::Clone {
                    is_shallow,
                    ref_name: None,
                    ref_version: source.manifest.version,
                    ref_path: source.uri().to_string(),
                    branch_name: None,
                },
                None,
            )
        };

        // Committed directly, a deep clone would reference files never copied
        // to the destination.
        let deep_dir = TempStrDir::default();
        let res = CommitBuilder::new(deep_dir.as_str())
            .execute(clone_txn(false))
            .await;
        assert!(
            matches!(res, Err(Error::InvalidInput { .. })),
            "got {res:?}"
        );
        assert!(Dataset::open(deep_dir.as_str()).await.is_err());

        // A shallow clone reads the source files in place, so it stays allowed.
        let shallow_dir = TempStrDir::default();
        let shallow = CommitBuilder::new(shallow_dir.as_str())
            .execute(clone_txn(true))
            .await
            .unwrap();
        assert_eq!(
            shallow.scan().try_into_batch().await.unwrap().num_rows(),
            10
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_commit_timeout_triggers() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_secs(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let res = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(Some(Duration::from_millis(50)))
            .execute(sample_transaction(1))
            .await;
        let err = res.expect_err("commit should time out");
        assert!(matches!(&err, Error::Timeout { .. }), "got {err:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn test_commit_timeout_applies_to_execute_batch() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_secs(5),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://batch-timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let res = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(Some(Duration::from_millis(50)))
            .execute_batch(vec![sample_transaction(1)])
            .await;
        let Err(err) = res else {
            panic!("commit should time out");
        };
        assert!(matches!(&err, Error::Timeout { .. }), "got {err:?}");
    }

    /// `with_timeout(None)` must let a commit run unbounded. Uses a throttled
    /// store so the commit takes real wall-clock time — long enough that the
    /// 50ms timeout in `test_commit_timeout_triggers` would have fired.
    #[tokio::test(start_paused = true)]
    async fn test_commit_timeout_none_disables() {
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_put_per_call: Duration::from_millis(200),
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://no-timeout")
            .with_params(&write_params)
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let new_ds = CommitBuilder::new(Arc::new(dataset))
            .with_timeout(None)
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(new_ds.manifest.version, 2);
    }

    #[tokio::test]
    async fn test_commit_retry_timeout_interrupts_conflict_backoff() {
        let dataset = InsertBuilder::new("memory://retry-timeout")
            .execute(vec![
                RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![ArrowField::new(
                        "i",
                        DataType::Int32,
                        false,
                    )])),
                    vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let result = CommitBuilder::new(Arc::new(dataset))
            .with_commit_handler(Arc::new(SlowConflictingCommitHandler))
            .with_max_retries(3)
            .with_retry_timeout(Duration::from_millis(150))
            .with_timeout(None)
            .execute(sample_transaction(1))
            .await;

        let error = result.expect_err("conflict backoff should respect retry timeout");
        assert!(
            matches!(&error, Error::TooMuchWriteContention { message, .. } if message.contains("failed on retry_timeout")),
            "got {error:?}"
        );
    }

    /// Loses the first commit race without writing anything, then commits
    /// through the wrapped handler.
    #[derive(Debug)]
    struct ConflictOnceCommitHandler {
        inner: Arc<dyn CommitHandler>,
        conflicted: std::sync::atomic::AtomicBool,
    }

    #[async_trait::async_trait]
    impl CommitHandler for ConflictOnceCommitHandler {
        fn is_version_not_found_definitive(&self) -> bool {
            self.inner.is_version_not_found_definitive()
        }

        async fn commit(
            &self,
            manifest: &mut Manifest,
            indices: Option<Vec<IndexMetadata>>,
            base_path: &object_store::path::Path,
            object_store: &ObjectStore,
            manifest_writer: ManifestWriter,
            naming_scheme: ManifestNamingScheme,
            transaction: Option<TableTransaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            if !self
                .conflicted
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(CommitError::CommitConflict);
            }
            self.inner
                .commit(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                )
                .await
        }
    }

    /// A commit that starts far behind the latest version must not spend its
    /// conflict-retry budget catching up. Every listing is slow here, so
    /// catching up alone outlasts `retry_timeout`; the first attempt then
    /// loses a race, and the retry must still get to run and succeed.
    #[tokio::test]
    async fn test_commit_retry_timeout_excludes_initial_catch_up() {
        let catch_up_latency = Duration::from_millis(400);
        let retry_timeout = Duration::from_millis(300);
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: catch_up_latency,
                ..Default::default()
            },
        });
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                ..Default::default()
            }),
            ..Default::default()
        };
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = || {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
            )
            .unwrap()
        };
        let stale = Arc::new(
            InsertBuilder::new("memory://retry-after-catch-up")
                .with_params(&write_params)
                .execute(vec![batch()])
                .await
                .unwrap(),
        );
        // Versions committed by another writer after `stale` was read.
        let mut latest = stale.clone();
        for _ in 0..3 {
            latest = Arc::new(
                InsertBuilder::new(latest.clone())
                    .with_params(&WriteParams {
                        mode: WriteMode::Append,
                        ..write_params.clone()
                    })
                    .execute(vec![batch()])
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(latest.manifest.version, 4);

        let committed = CommitBuilder::new(stale.clone())
            .with_commit_handler(Arc::new(ConflictOnceCommitHandler {
                inner: stale.table_state.commit_handler.clone(),
                conflicted: std::sync::atomic::AtomicBool::new(false),
            }))
            .with_retry_timeout(retry_timeout)
            .with_timeout(None)
            .execute(sample_transaction(1))
            .await
            .unwrap();
        assert_eq!(committed.manifest.version, 5);
    }

    #[tokio::test]
    async fn test_commit_batch() {
        // Create a dataset
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://test")
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        // Attempting to commit empty gives error
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![])
            .await;
        assert!(matches!(res, Err(Error::InvalidInput { .. })));

        // Attempting to commit update gives error
        let update_transaction = Transaction {
            uuid: uuid::Uuid::new_v4().hyphenated().to_string(),
            operation: Operation::Update {
                updated_fragments: vec![],
                new_fragments: vec![],
                removed_fragment_ids: vec![],
                fields_modified: vec![],
                mem_wal_dir_id: None,
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            read_version: 1,
            tag: None,
            transaction_properties: None,
        };
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![update_transaction])
            .await;
        assert!(matches!(res, Err(Error::NotSupported { .. })));

        // Doing multiple appends includes all.
        let append1 = sample_transaction(1);
        let append2 = sample_transaction(2);
        let mut expected_fragments = vec![];
        if let Operation::Append { fragments } = &append1.operation {
            expected_fragments.extend(fragments.clone());
        }
        if let Operation::Append { fragments } = &append2.operation {
            expected_fragments.extend(fragments.clone());
        }
        let res = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![append1.clone(), append2.clone()])
            .await
            .unwrap();
        let transaction = res.merged;
        assert!(
            matches!(transaction.operation, Operation::Append { fragments } if fragments == expected_fragments)
        );
        assert_eq!(transaction.read_version, 1);
    }

    /// On non-lexically-ordered stores (e.g. S3 Express) a commit should use the
    /// version hint (a few HEAD probes, O(k)) instead of a full O(n) listing.
    #[tokio::test(start_paused = true)]
    async fn test_commit_uses_version_hint_on_non_lexical_store() {
        // Make `list` artificially slow per entry so a full listing would be
        // obvious; HEAD/GET/PUT stay fast.
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_entry: Duration::from_millis(50),
                wait_get_per_call: Duration::from_millis(1),
                wait_put_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());
        let write_params = WriteParams {
            store_params: Some(ObjectStoreParams {
                object_store_wrapper: Some(throttled),
                list_is_lexically_ordered: Some(false),
                ..Default::default()
            }),
            session: Some(session.clone()),
            enable_v2_manifest_paths: true,
            ..Default::default()
        };

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
        )
        .unwrap();
        let mut dataset = Arc::new(
            InsertBuilder::new("memory://test_version_hint")
                .with_params(&write_params)
                .execute(vec![batch])
                .await
                .unwrap(),
        );

        // Build up many versions so a full listing would be expensive.
        for _ in 0..50 {
            dataset = Arc::new(
                CommitBuilder::new(dataset.clone())
                    .execute(sample_transaction(dataset.manifest().version))
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(dataset.manifest().version, 51);

        dataset
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();

        let start = std::time::Instant::now();
        let new_ds = CommitBuilder::new(dataset.clone())
            .execute(sample_transaction(dataset.manifest().version))
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // A full listing of ~52 entries at 50ms each would take ~2.6s.
        assert!(
            elapsed < Duration::from_secs(1),
            "commit took {elapsed:?}; the version hint path was likely not used"
        );

        let io_stats = new_ds
            .table_state
            .object_store
            .as_ref()
            .io_stats_incremental();
        assert!(
            io_stats.read_iops < 10,
            "read_iops = {}; a full listing was likely used",
            io_stats.read_iops
        );
    }
}
