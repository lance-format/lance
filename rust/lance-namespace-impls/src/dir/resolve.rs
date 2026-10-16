// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow::record_batch::RecordBatchIterator;
use arrow_ipc::reader::StreamReader;
use bytes::Bytes;
use futures::StreamExt;
use lance::dataset::refs::check_valid_branch;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_core::{Error, Result};
use lance_io::object_store::throttle::is_throttle_error;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, ReadDirOptions};
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{DescribeTableRequest, Identity};
use object_store::Error as ObjectStoreError;
use object_store::path::Path;

use super::{DirectoryNamespace, TableStatus, manifest};

impl DirectoryNamespace {
    pub(super) fn manifest_ns_for_read(&self) -> Option<&Arc<manifest::ManifestNamespace>> {
        self.write_manifest_ns
            .get()
            .or_else(|| self.manifest_ns.get())
    }

    pub(super) async fn manifest_ns_for_write(
        &self,
    ) -> Result<Option<Arc<manifest::ManifestNamespace>>> {
        if !self.manifest_enabled {
            return Ok(None);
        }

        let manifest_ns = self
            .write_manifest_ns
            .get_or_try_init(|| async {
                manifest::ManifestNamespace::from_directory(
                    self.root.clone(),
                    self.storage_options.clone(),
                    self.session.clone(),
                    self.object_store.clone(),
                    self.base_path.clone(),
                    self.dir_listing_enabled,
                    self.inline_optimization_enabled,
                    self.commit_retries,
                )
                .await
                .map(Arc::new)
            })
            .await?;
        Ok(Some(manifest_ns.clone()))
    }

    /// Lazily open the `__manifest` dataset (read-only) into the read cell.
    ///
    /// `manifest_ns` is populated at construction only if `__manifest` already
    /// existed then. A manifest created afterwards -- e.g. by another
    /// connection's write, since Phalanx caches a connection per db and the
    /// first op on a fresh db is usually a read -- would otherwise stay
    /// invisible to this connection's reads forever, so describe/list/exists
    /// report "not found" for a table that is in fact registered. Re-open on
    /// demand so reads self-heal once the manifest exists. Idempotent and cheap
    /// once the cell is populated; unlike `manifest_ns_for_write` it never
    /// creates the manifest.
    pub(super) async fn ensure_read_manifest(&self) -> Result<()> {
        if !self.manifest_enabled
            || self.manifest_ns.get().is_some()
            || self.write_manifest_ns.get().is_some()
        {
            return Ok(());
        }
        match self
            .manifest_ns
            .get_or_try_init(|| async {
                manifest::ManifestNamespace::open_from_directory(
                    self.root.clone(),
                    self.storage_options.clone(),
                    self.session.clone(),
                    self.object_store.clone(),
                    self.base_path.clone(),
                    self.dir_listing_enabled,
                    self.inline_optimization_enabled,
                    self.commit_retries,
                )
                .await
                .map(Arc::new)
            })
            .await
        {
            Ok(_) => Ok(()),
            // Manifest still doesn't exist: leave the cell empty so a later read
            // retries once it does. A genuinely absent table is still reported
            // not-found by the callers' existing `manifest_ns_for_read() == None`
            // branch, exactly as before.
            Err(e) if manifest::ManifestNamespace::is_not_found_load_error(&e) => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub(super) fn child_namespace_requires_manifest_error(&self) -> Error {
        if self.manifest_enabled {
            NamespaceError::NamespaceNotFound {
                message: "Child namespace reads require an existing __manifest dataset".to_string(),
            }
            .into()
        } else {
            NamespaceError::Unsupported {
                message: "Child namespaces are only supported when manifest mode is enabled"
                    .to_string(),
            }
            .into()
        }
    }

    /// Apply pagination to a list of table names
    ///
    /// Sorts the list alphabetically and applies pagination using page_token (start_after) and limit.
    ///
    /// # Arguments
    /// * `names` - The vector of table names to paginate
    /// * `page_token` - Skip items until finding one greater than this value (start_after semantics)
    /// * `limit` - Maximum number of items to keep
    ///
    /// # Returns
    /// The next page token (last item in this page) if more results exist beyond the limit,
    /// or `None` if this is the last page.
    pub(super) fn apply_pagination(
        names: &mut Vec<String>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        // Sort alphabetically for consistent ordering
        names.sort();

        // Apply page_token filtering (start_after semantics)
        if let Some(start_after) = page_token {
            if let Some(index) = names
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
            {
                names.drain(0..index);
            } else {
                names.clear();
            }
        }

        // Apply limit and compute next page token
        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if names.len() > limit {
                let next_page_token = if limit > 0 {
                    Some(names[limit - 1].clone())
                } else {
                    None
                };
                names.truncate(limit);
                return next_page_token;
            }
        }

        None
    }

    /// Page size requested from [`ObjectStore::read_dir_page`] while scanning the namespace
    /// directory for tables. Only bounds the cost of one backend request on stores that push
    /// pagination down (S3, GCS, Azure); `list_directory_tables` always walks every page.
    const LIST_DIRECTORY_PAGE_SIZE: usize = 1000;

    /// List tables using directory scanning (fallback method)
    pub(super) async fn list_directory_tables(&self) -> Result<Vec<String>> {
        let mut tables = Vec::new();
        let mut page_token = None;

        loop {
            let page = self
                .object_store
                .read_dir_page(
                    self.base_path.clone(),
                    ReadDirOptions {
                        page_token,
                        // Only a hint to backends with a paginated list API (S3, GCS, Azure):
                        // it bounds the cost of one request, not the number of tables returned.
                        // Every other backend still lists (and pages through) the whole
                        // directory here regardless, same as `read_dir` always did.
                        limit: Some(Self::LIST_DIRECTORY_PAGE_SIZE),
                    },
                )
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to list directory: {:?}", e),
                    })
                })?;

            let candidates: Vec<String> = page
                .result
                .common_prefixes
                .iter()
                .chain(page.result.objects.iter().map(|o| &o.location))
                .filter_map(|p| {
                    p.filename()?
                        .trim_end_matches('/')
                        .strip_suffix(".lance")
                        .map(|name| name.to_string())
                })
                .collect();

            // Each candidate needs its own `check_table_status` round trip (a `read_dir` probe
            // for a deregistration marker), so this is linear in the number of listed entries;
            // run a bounded number concurrently rather than one at a time.
            let mut stream =
                futures::stream::iter(candidates.into_iter().map(|table_name| async move {
                    let status = self.check_table_status(&table_name).await?;
                    Ok::<Option<String>, Error>((!status.is_deregistered).then_some(table_name))
                }))
                .buffered(manifest::DECLARED_FILTER_CONCURRENCY);

            while let Some(result) = stream.next().await {
                if let Some(table_name) = result? {
                    tables.push(table_name);
                }
            }

            // A page can come back holding fewer children than the requested limit and still
            // be followed by more (a backend can spend its page budget on keys a delimiter
            // collapses away, or cap a page on its own besides) — walk until the token is
            // `None`, not until a page comes back short.
            page_token = page.page_token;
            if page_token.is_none() {
                break;
            }
        }

        Ok(tables)
    }

    /// Validate that the namespace ID represents the root namespace
    pub(super) fn validate_root_namespace_id(id: &Option<Vec<String>>) -> Result<()> {
        if let Some(id) = id
            && !id.is_empty()
        {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "Directory namespace only supports root namespace operations, but got namespace ID: {:?}. Expected empty ID.",
                    id
                ),
            }
            .into());
        }
        Ok(())
    }

    /// Extract table name from table ID
    pub(super) fn table_name_from_id(id: &Option<Vec<String>>) -> Result<String> {
        let id = id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Directory namespace table ID cannot be empty".to_string(),
            })
        })?;

        if id.len() != 1 {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "Multi-level table IDs are only supported when manifest mode is enabled, but got: {:?}",
                    id
                ),
            }
            .into());
        }

        Ok(id[0].clone())
    }

    fn format_table_id(table_id: &[String]) -> String {
        format!(
            "table id '{}'",
            manifest::ManifestNamespace::str_object_id(table_id)
        )
    }

    pub(super) fn format_table_id_from_request(id: &Option<Vec<String>>) -> String {
        id.as_ref()
            .map(|table_id| Self::format_table_id(table_id))
            .unwrap_or_else(|| "table id '<unknown>'".to_string())
    }

    pub(super) async fn resolve_table_location(&self, id: &Option<Vec<String>>) -> Result<String> {
        let mut describe_req = DescribeTableRequest::new();
        describe_req.id = id.clone();
        describe_req.load_detailed_metadata = Some(false);

        // Use internal impl to avoid counting this as an external API call
        let describe_resp = self.describe_table_impl(describe_req).await?;

        describe_resp.location.ok_or_else(|| {
            lance_core::Error::from(NamespaceError::TableNotFound {
                message: format!("Table location not found for: {:?}", id),
            })
        })
    }

    /// Map a Lance ref-related error returned by `Dataset::tags()` operations into
    /// the appropriate `NamespaceError` for tag APIs (create/get/update/delete).
    pub(super) fn map_tag_error(
        err: lance_core::Error,
        tag: &str,
        table_uri: &str,
    ) -> lance_core::Error {
        match err {
            lance_core::Error::RefNotFound { .. } => NamespaceError::TableTagNotFound {
                message: format!("tag '{}' for table at '{}'", tag, table_uri),
            }
            .into(),
            lance_core::Error::RefConflict { .. } => NamespaceError::TableTagAlreadyExists {
                message: format!("tag '{}' for table at '{}'", tag, table_uri),
            }
            .into(),
            lance_core::Error::InvalidRef { message } => NamespaceError::InvalidInput {
                message: format!("invalid tag '{}': {}", tag, message),
            }
            .into(),
            lance_core::Error::VersionNotFound { message } => {
                NamespaceError::TableVersionNotFound {
                    message: format!(
                        "version referenced by tag '{}' not found for table at '{}': {}",
                        tag, table_uri, message
                    ),
                }
                .into()
            }
            other => NamespaceError::Internal {
                message: format!(
                    "tag operation failed for tag '{}' on table at '{}': {}",
                    tag, table_uri, other
                ),
            }
            .into(),
        }
    }

    /// Map lance-core ref errors from branch operations to namespace errors.
    ///
    /// `RefConflict` is intentionally not handled here: create-time duplicates are rejected by
    /// the existence pre-check before `create_branch` runs, and delete maps its own `RefConflict`
    /// (branch still has dependents) inline.
    pub(super) fn map_branch_error(
        err: lance_core::Error,
        branch: &str,
        table_uri: &str,
    ) -> lance_core::Error {
        match err {
            lance_core::Error::RefNotFound { .. } => NamespaceError::TableBranchNotFound {
                message: format!("branch '{}' for table at '{}'", branch, table_uri),
            }
            .into(),
            lance_core::Error::InvalidRef { message } => NamespaceError::InvalidInput {
                message: format!("invalid branch '{}': {}", branch, message),
            }
            .into(),
            lance_core::Error::VersionNotFound { message } => {
                NamespaceError::TableVersionNotFound {
                    message: format!(
                        "source version for branch '{}' not found for table at '{}': {}",
                        branch, table_uri, message
                    ),
                }
                .into()
            }
            other => NamespaceError::Internal {
                message: format!(
                    "branch operation failed for branch '{}' on table at '{}': {}",
                    branch, table_uri, other
                ),
            }
            .into(),
        }
    }

    /// Map a Lance error from a table mutation (update / delete / merge-insert) into the most
    /// specific `NamespaceError` we can determine from the underlying variant.
    ///
    /// Collapsing every failure into `InvalidInput`/`Internal` hides the real cause from callers;
    /// mapping per variant lets them branch on a meaningful error code (e.g. retry on
    /// `ConcurrentModification`, surface `TableNotFound` to the user).
    ///
    /// Commit-conflict variants are mapped consistently with `convert_lance_commit_error` in
    /// `manifest.rs`: `CommitConflict` (retries exhausted, safe to retry) -> `Throttling`, while
    /// semantic conflicts (`TooMuchWriteContention` / `RetryableCommitConflict` /
    /// `IncompatibleTransaction` / `VersionConflict`) -> `ConcurrentModification`.
    pub(super) fn map_mutation_error(
        err: lance_core::Error,
        operation: &str,
        table_uri: &str,
    ) -> lance_core::Error {
        let detail = err.to_string();
        let ns_err = match &err {
            lance_core::Error::InvalidInput { .. }
            | lance_core::Error::Unprocessable { .. }
            | lance_core::Error::InvalidRef { .. } => NamespaceError::InvalidInput {
                message: format!(
                    "Invalid input for {} on table at '{}': {}",
                    operation, table_uri, detail
                ),
            },
            lance_core::Error::NotFound { .. } | lance_core::Error::DatasetNotFound { .. } => {
                NamespaceError::TableNotFound {
                    message: format!(
                        "Table at '{}' not found while running {}: {}",
                        table_uri, operation, detail
                    ),
                }
            }
            lance_core::Error::SchemaMismatch { .. } | lance_core::Error::Schema { .. } => {
                NamespaceError::TableSchemaValidationError {
                    message: format!(
                        "Schema validation failed for {} on table at '{}': {}",
                        operation, table_uri, detail
                    ),
                }
            }
            // `CommitConflict` means the version-collision retries were exhausted; the operation
            // is safe to retry as-is, so surface it as `Throttling` (kept aligned with
            // `convert_lance_commit_error` in manifest.rs).
            lance_core::Error::CommitConflict { .. } => NamespaceError::Throttling {
                message: format!(
                    "Too many concurrent writes for {} on table at '{}', please retry later: {}",
                    operation, table_uri, detail
                ),
            },
            // Semantic conflicts: a concurrent change is incompatible with this one and retrying
            // as-is would not help, so surface them as `ConcurrentModification` (kept aligned with
            // `convert_lance_commit_error` in manifest.rs).
            lance_core::Error::TooMuchWriteContention { .. }
            | lance_core::Error::RetryableCommitConflict { .. }
            | lance_core::Error::IncompatibleTransaction { .. }
            | lance_core::Error::VersionConflict { .. } => NamespaceError::ConcurrentModification {
                message: format!(
                    "Concurrent modification detected for {} on table at '{}': {}",
                    operation, table_uri, detail
                ),
            },
            lance_core::Error::NotSupported { .. } => NamespaceError::Unsupported {
                message: format!(
                    "{} is not supported on table at '{}': {}",
                    operation, table_uri, detail
                ),
            },
            _ => NamespaceError::Internal {
                message: format!(
                    "Failed to run {} on table at '{}': {}",
                    operation, table_uri, detail
                ),
            },
        };
        ns_err.into()
    }

    pub(super) async fn table_has_actual_manifests(&self, table_name: &str) -> Result<bool> {
        manifest::ManifestNamespace::path_has_actual_manifests(
            &self.object_store,
            &self.table_path(table_name),
        )
        .await
    }

    pub(super) async fn filter_declared_tables(
        &self,
        tables: Vec<String>,
        include_declared: bool,
    ) -> Result<Vec<String>> {
        if include_declared {
            return Ok(tables);
        }

        let mut stream = futures::stream::iter(tables.into_iter().map(|table_name| async move {
            // `include_declared=false` is an explicit opt-in. We still pay one `_versions/` probe
            // per table here so declared-state is derived from actual manifests. This is linear in
            // the total number of listed tables, but we probe a bounded number concurrently.
            if self.table_has_actual_manifests(&table_name).await? {
                Ok::<Option<String>, Error>(Some(table_name))
            } else {
                Ok::<Option<String>, Error>(None)
            }
        }))
        .buffered(manifest::DECLARED_FILTER_CONCURRENCY);

        let mut filtered = Vec::new();
        while let Some(result) = stream.next().await {
            if let Some(table_name) = result? {
                filtered.push(table_name);
            }
        }
        Ok(filtered)
    }

    pub(super) fn ipc_reader_from_request_data(
        request_data: &Bytes,
        operation: &str,
    ) -> Result<(
        Box<dyn arrow::record_batch::RecordBatchReader + Send>,
        usize,
    )> {
        if request_data.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Request data (Arrow IPC stream) is required for {}",
                    operation
                ),
            }
            .into());
        }

        let cursor = Cursor::new(request_data.as_ref());
        let stream_reader =
            StreamReader::try_new(cursor, None).map_err(|e| NamespaceError::InvalidInput {
                message: format!("Invalid Arrow IPC stream: {}", e),
            })?;
        let arrow_schema = stream_reader.schema();

        let mut num_rows = 0usize;
        let mut batches = Vec::new();
        for batch_result in stream_reader {
            let batch = batch_result.map_err(|e| NamespaceError::Internal {
                message: format!("Failed to read batch from IPC stream: {}", e),
            })?;
            num_rows += batch.num_rows();
            batches.push(batch);
        }

        let reader: Box<dyn arrow::record_batch::RecordBatchReader + Send> = if batches.is_empty() {
            let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], arrow_schema))
        } else {
            let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
            Box::new(RecordBatchIterator::new(batch_results, arrow_schema))
        };

        Ok((reader, num_rows))
    }

    pub(super) async fn table_uri_has_actual_manifests(&self, table_uri: &str) -> Result<bool> {
        let table_path = self.object_store_path_from_uri(table_uri)?;
        manifest::ManifestNamespace::path_has_actual_manifests(&self.object_store, &table_path)
            .await
    }

    pub(super) fn object_store_path_from_uri(&self, uri: &str) -> Result<Path> {
        let registry = self
            .session
            .as_ref()
            .map(|session| session.store_registry())
            .unwrap_or_else(|| Arc::new(ObjectStoreRegistry::default()));
        ObjectStore::extract_path_from_uri(registry, uri)
    }

    /// Normalize and validate a branch selector: `None`, empty, and `main` mean
    /// the main branch; any other name is validated with lance's
    /// `check_valid_branch` (lance skips this on the open path) so it cannot
    /// escape the table root via `..`.
    pub(super) fn normalized_branch(branch: Option<&str>) -> Result<Option<&str>> {
        match branch.filter(|b| !b.is_empty() && *b != "main") {
            Some(branch) => {
                check_valid_branch(branch).map_err(|e| {
                    lance_core::Error::from(NamespaceError::InvalidInput {
                        message: format!("invalid branch name '{}': {}", branch, e),
                    })
                })?;
                Ok(Some(branch))
            }
            None => Ok(None),
        }
    }

    pub(super) async fn open_validated_branch(
        &self,
        table_uri: &str,
        branch: &str,
    ) -> Result<Dataset> {
        let dataset = self
            .configured_builder(table_uri)
            .with_branch(branch, None)
            .load()
            .await
            .map_err(|e| {
                let message = format!(
                    "branch '{}' not found for table at '{}': {}",
                    branch, table_uri, e
                );
                Self::map_open_error(e, NamespaceError::TableNotFound { message })
            })?;
        dataset.branches().get(branch).await.map_err(|e| {
            Self::map_open_error(
                e,
                NamespaceError::TableNotFound {
                    message: format!("branch '{}' not found for table at '{}'", branch, table_uri),
                },
            )
        })?;
        Ok(dataset)
    }

    pub(super) async fn resolve_branch_location(
        &self,
        table_uri: &str,
        branch: &str,
    ) -> Result<String> {
        Ok(self
            .open_validated_branch(table_uri, branch)
            .await?
            .branch_location()
            .uri)
    }

    /// Resolves a branch to its `(uri, object-store path, parent_version)` for
    /// `create_table_version`.
    ///
    /// `BranchContents` is the source of truth, so check the ref first: a
    /// registered branch commits directly and returns its `parent_version` for
    /// empty-chain CAS. With no ref, accept the commit only on an empty chain
    /// (the `create_branch` bootstrap, whose first commit precedes its ref) and
    /// return `parent_version = None`; reject a chain that already holds
    /// committed versions as a zombie.
    pub(super) async fn resolve_branch_for_commit(
        &self,
        table_uri: &str,
        branch: &str,
    ) -> Result<(String, Path, Option<u64>)> {
        let main = self
            .configured_builder(table_uri)
            .load()
            .await
            .map_err(|e| {
                let message = format!("table at '{}' not found: {}", table_uri, e);
                Self::map_open_error(e, NamespaceError::TableNotFound { message })
            })?;
        let branch_location = main.branch_location().find_branch(Some(branch))?;
        match main.branches().get(branch).await {
            Ok(contents) => Ok((
                branch_location.uri,
                branch_location.path,
                Some(contents.parent_version),
            )),
            Err(lance_core::Error::RefNotFound { .. }) => {
                if self
                    .branch_has_committed_versions(&branch_location.path)
                    .await?
                {
                    return Err(NamespaceError::TableNotFound {
                        message: format!(
                            "branch '{}' not found for table at '{}'",
                            branch, table_uri
                        ),
                    }
                    .into());
                }
                Ok((branch_location.uri, branch_location.path, None))
            }
            Err(e) => Err(e),
        }
    }

    async fn branch_has_committed_versions(&self, branch_path: &Path) -> Result<bool> {
        Ok(!self
            .list_versions_under(branch_path, false, Some(1))
            .await?
            .is_empty())
    }

    pub(super) fn validate_dir_only_properties(
        properties: Option<&HashMap<String, String>>,
        operation: &str,
    ) -> Result<()> {
        // Dir-only mode has no metadata catalog, so non-empty table properties would be accepted
        // and then lost. Reject them instead. Request-level storage options are different: they
        // directly affect the current write and remain supported in dir-only mode.
        if properties.is_some_and(|properties| !properties.is_empty()) {
            return Err(NamespaceError::Unsupported {
                message: format!(
                    "{} with non-empty table properties requires manifest_enabled=true",
                    operation
                ),
            }
            .into());
        }
        Ok(())
    }

    pub(super) async fn write_reader_to_table(
        &self,
        table_uri: &str,
        reader: Box<dyn arrow::record_batch::RecordBatchReader + Send>,
        mode: WriteMode,
        extra_storage_options: Option<HashMap<String, String>>,
    ) -> Result<Dataset> {
        // Insert and merge-insert request models do not carry request-level storage options,
        // so these writes intentionally use the namespace-level storage options only.
        let mut merged_storage_options = self.storage_options.clone().unwrap_or_default();
        if let Some(extra_storage_options) = extra_storage_options {
            merged_storage_options.extend(extra_storage_options);
        }
        let store_params = (!merged_storage_options.is_empty()).then(|| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance_io::object_store::StorageOptionsAccessor::with_static_options(
                    merged_storage_options,
                ),
            )),
            ..Default::default()
        });

        let write_params = WriteParams {
            mode,
            store_params,
            session: self.session.clone(),
            ..Default::default()
        };

        let dataset = Dataset::write(reader, table_uri, Some(write_params))
            .await
            .map_err(|e| NamespaceError::Internal {
                message: format!("Failed to write table at '{}': {}", table_uri, e),
            })?;

        Ok(dataset)
    }

    pub(super) fn table_full_uri(&self, table_name: &str) -> String {
        format!("{}/{}.lance", self.root, table_name)
    }

    /// Get the object store path for a table (relative to base_path)
    pub(super) fn table_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
    }

    /// Get the reserved file path for a table
    pub(super) fn table_reserved_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
            .join(".lance-reserved")
    }

    /// Get the deregistered marker file path for a table
    pub(super) fn table_deregistered_file_path(&self, table_name: &str) -> Path {
        self.base_path
            .clone()
            .join(format!("{}.lance", table_name).as_str())
            .join(".lance-deregistered")
    }

    /// Atomically check table existence and deregistration status.
    ///
    /// This performs a single directory listing to get a consistent snapshot of the
    /// table's state, avoiding race conditions between checking existence and
    /// checking deregistration status.
    pub(crate) async fn check_table_status(&self, table_name: &str) -> Result<TableStatus> {
        let table_path = self.table_path(table_name);
        match self.object_store.read_dir(table_path).await {
            Ok(entries) => {
                let exists = !entries.is_empty();
                let is_deregistered = entries.iter().any(|e| e.ends_with(".lance-deregistered"));
                let has_reserved_file = entries.iter().any(|e| e.ends_with(".lance-reserved"));
                Ok(TableStatus {
                    exists,
                    is_deregistered,
                    has_reserved_file,
                })
            }
            // Local filesystems error on a missing directory where object stores
            // return an empty listing; both mean the table does not exist.
            Err(e) if manifest::ManifestNamespace::is_not_found_load_error(&e) => Ok(TableStatus {
                exists: false,
                is_deregistered: false,
                has_reserved_file: false,
            }),
            // Any other failure must propagate: collapsing it to "does not exist"
            // lets a transient error overwrite a live table via create/exist-ok
            // callers and destroys the retry evidence classifiers depend on.
            Err(e) => Err(Self::classify_storage_error(e)),
        }
    }

    /// Classify a storage error into a typed [`NamespaceError`]. The full source
    /// text is embedded in the message because the pyo3 layer flattens namespace
    /// errors to message-only (no `__cause__`), so that is the only place the
    /// 429/503 evidence survives to Python.
    pub(super) fn classify_storage_error(err: Error) -> Error {
        if matches!(&err, Error::Namespace { .. }) {
            return err;
        }
        let detail = err.to_string();
        if let Error::IO { source, .. } = &err
            && let Some(os_err) = source.downcast_ref::<ObjectStoreError>()
        {
            if is_throttle_error(os_err) {
                return NamespaceError::Throttling {
                    message: format!(
                        "Storage request was throttled while resolving table: {detail}"
                    ),
                }
                .into();
            }
            if Self::is_service_unavailable_error(os_err) {
                return NamespaceError::ServiceUnavailable {
                    message: format!("Storage service unavailable while resolving table: {detail}"),
                }
                .into();
            }
        }
        NamespaceError::Internal {
            message: format!("Storage error while resolving table: {detail}"),
        }
        .into()
    }

    /// Detect a clearly-transient 5xx not already caught by [`is_throttle_error`].
    /// `object_store` does not expose HTTP status codes, so match the (deliberately
    /// narrow) canonical status phrases in the message.
    fn is_service_unavailable_error(err: &ObjectStoreError) -> bool {
        if let ObjectStoreError::Generic { source, .. } = err {
            let message = source.to_string().to_ascii_lowercase();
            message.contains("503 service unavailable")
                || message.contains("502 bad gateway")
                || message.contains("504 gateway timeout")
        } else {
            false
        }
    }

    /// Whether a manifest error means the table is genuinely absent (rather than a
    /// storage failure while consulting the manifest). Only such errors may fall
    /// through to the directory listing; anything else must propagate.
    pub(super) fn is_manifest_table_absent_error(err: &Error) -> bool {
        if manifest::ManifestNamespace::is_not_found_load_error(err) {
            return true;
        }
        if let Error::Namespace { source, .. } = err
            && let Some(ns_err) = source.downcast_ref::<NamespaceError>()
        {
            return matches!(ns_err, NamespaceError::TableNotFound { .. });
        }
        false
    }

    /// Map a dataset/version/branch open error: a transient IO error propagates
    /// typed via [`classify_storage_error`], while a genuine not-found (missing
    /// dataset, version, or ref) keeps the caller's `not_found` variant.
    pub(super) fn map_open_error(err: Error, not_found: NamespaceError) -> Error {
        if matches!(&err, Error::IO { .. })
            && !manifest::ManifestNamespace::is_not_found_load_error(&err)
        {
            return Self::classify_storage_error(err);
        }
        not_found.into()
    }

    /// Get storage options for a table, using credential vending if configured.
    ///
    /// If credential vendor properties are configured and the table location matches
    /// a supported cloud provider, this will create an appropriate vendor and vend
    /// temporary credentials scoped to the table location. Otherwise, returns the
    /// static storage options.
    ///
    /// The vendor type is auto-selected based on the table URI:
    /// - `s3://` locations use AWS STS AssumeRole
    /// - `gs://` locations use GCP OAuth2 tokens
    /// - `az://` locations use Azure SAS tokens
    ///
    /// The permission level (Read, Write, Admin) is configured at namespace
    /// initialization time via the `credential_vendor_permission` property.
    ///
    /// # Arguments
    ///
    /// * `table_uri` - The full URI of the table
    /// * `identity` - Optional identity from the request for identity-based credential vending
    pub(super) async fn get_storage_options_for_table(
        &self,
        table_uri: &str,
        vend_credentials: bool,
        identity: Option<&Identity>,
    ) -> Result<Option<HashMap<String, String>>> {
        if vend_credentials && let Some(ref vendor) = self.credential_vendor {
            let vended = vendor.vend_credentials(table_uri, identity).await?;
            return Ok(Some(vended.storage_options));
        }
        // When vend_input_storage_options is enabled and no credential vendor is configured,
        // return the input storage options. This is useful for testing.
        if self.vend_input_storage_options {
            let mut options = self.storage_options.clone().unwrap_or_default();
            // Add expires_at_millis if refresh interval is configured
            if let Some(refresh_interval_millis) =
                self.vend_input_storage_options_refresh_interval_millis
            {
                let now_millis = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let expires_at_millis = now_millis + refresh_interval_millis;
                options.insert(
                    "expires_at_millis".to_string(),
                    expires_at_millis.to_string(),
                );
            }
            return Ok(Some(options));
        }
        // When no credential vendor is configured, return None to avoid
        // leaking the namespace's own static credentials to clients.
        Ok(None)
    }
}
