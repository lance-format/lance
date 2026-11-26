// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::refs::{Branches, Ref, Refs, check_valid_branch, normalize_branch, standardize_branch};
use crate::branch_location::BranchLocation;
use crate::namespace_manifest::LanceNamespaceExternalManifestStore;
use crate::{Table, TableSession};
use futures::{FutureExt, TryStreamExt};
use lance_core::utils::tracing::{DATASET_LOADING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{Error, Result};
use lance_file::reader::FileReaderOptions;
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, LanceNamespaceStorageOptionsProvider, ObjectStore,
    ObjectStoreParams, StorageOptions, StorageOptionsAccessor,
};
use lance_namespace::LanceNamespace;
use lance_namespace::models::DescribeTableRequest;
use lance_table::{
    feature_flags::ensure_can_read_manifest,
    format::{
        Manifest, populate_manifest_schema_dictionaries,
        storage_contract::check_manifest_storage_version,
    },
    io::commit::external_manifest::ExternalManifestCommitHandler,
    io::commit::{CommitHandler, ManifestLocation, commit_handler_from_url},
};
#[cfg(feature = "aws")]
use object_store::aws::AwsCredentialProvider;
use object_store::{DynObjectStore, path::Path};
use prost::Message;
use tracing::{info, instrument};
use url::Url;

/// Configure and open a [`Table`] snapshot.
#[derive(Clone)]
pub struct TableBuilder {
    /// Optional pre-loaded manifest to avoid loading it again.
    manifest: Option<Manifest>,
    session: Option<Arc<TableSession>>,
    commit_handler: Option<Arc<dyn CommitHandler>>,
    options: ObjectStoreParams,
    version: Option<Ref>,
    table_uri: String,
    file_reader_options: Option<FileReaderOptions>,
    /// Storage options that override user-provided options (e.g., from namespace client)
    storage_options_override: Option<HashMap<String, String>>,
    /// Runtime-only exact object store bindings keyed by base path URI.
    base_store_params: HashMap<String, ObjectStoreParams>,
    /// Namespace-managed table info `(client, table_id)`, set by `from_namespace`
    /// when the table uses managed versioning. The commit handler is built in
    /// `build_object_store`, rooted at the resolved table path; the branch a
    /// namespace request targets is derived per call from the base path the
    /// handler is handed.
    namespace_managed: Option<(Arc<dyn LanceNamespace>, Vec<String>)>,
}

impl std::fmt::Debug for TableBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableBuilder")
            .field("manifest", &self.manifest.is_some())
            .field("session", &self.session.is_some())
            .field("commit_handler", &self.commit_handler.is_some())
            .field("version", &self.version)
            .field("table_uri", &self.table_uri)
            .field("file_reader_options", &self.file_reader_options)
            .field(
                "storage_options_override",
                &self.storage_options_override.is_some(),
            )
            .field("base_store_params", &!self.base_store_params.is_empty())
            .field("namespace_managed", &self.namespace_managed.is_some())
            .finish()
    }
}

impl TableBuilder {
    /// Build an opener for a Lance table URI or local path.
    pub fn from_uri<T: AsRef<str>>(table_uri: T) -> Self {
        Self {
            table_uri: table_uri.as_ref().to_string(),
            options: ObjectStoreParams::default(),
            commit_handler: None,
            session: None,
            version: None,
            manifest: None,
            file_reader_options: None,
            storage_options_override: None,
            base_store_params: HashMap::new(),
            namespace_managed: None,
        }
    }

    /// Create a TableBuilder from a LanceNamespace client
    ///
    /// This will automatically fetch the table location and storage options from the namespace
    /// client via `describe_table()`.
    ///
    /// Storage options from the namespace client will override any user-provided storage options
    /// set via `.with_storage_options()`. This ensures the namespace client is always the source
    /// of truth for storage options.
    ///
    /// # Arguments
    /// * `namespace_client` - The namespace client implementation to fetch table info from
    /// * `table_id` - The table identifier (e.g., vec!["my_table"])
    ///
    #[allow(deprecated)]
    pub async fn from_namespace(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
    ) -> Result<Self> {
        let request = DescribeTableRequest {
            id: Some(table_id.clone()),
            vend_credentials: Some(true),
            ..Default::default()
        };

        let response = namespace_client
            .describe_table(request)
            .await
            .map_err(|e| Error::namespace_source(Box::new(e)))?;

        let table_uri = response.location.ok_or_else(|| {
            Error::namespace_source(Box::new(std::io::Error::other(
                "Table location not found in namespace response",
            )))
        })?;

        let mut builder = Self::from_uri(&table_uri);

        // Defer building the commit handler to load(): the manifest store is
        // rooted at the resolved table path, which is only known once the
        // object store is built.
        if response.managed_versioning == Some(true) {
            builder.namespace_managed = Some((namespace_client.clone(), table_id.clone()));
        }

        // Use namespace storage options if available
        let namespace_storage_options = response.storage_options;

        builder.storage_options_override = namespace_storage_options.clone();

        if let Some(initial_opts) = namespace_storage_options {
            let provider: Arc<dyn lance_io::object_store::StorageOptionsProvider> = Arc::new(
                LanceNamespaceStorageOptionsProvider::new(namespace_client, table_id),
            );
            builder.options.storage_options_accessor = Some(Arc::new(
                StorageOptionsAccessor::with_initial_and_provider(initial_opts, provider),
            ));
        }

        Ok(builder)
    }
}

// Much of this builder is directly inspired from the to delta-rs table builder implementation
// https://github.com/delta-io/delta-rs/main/crates/deltalake-core/src/table/builder.rs
impl TableBuilder {
    /// The block size passed to the underlying Object Store reader.
    ///
    /// This is used to control the minimal request size.
    /// Defaults to 4KB for local files and 64KB for others
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.options.block_size = Some(block_size);
        self
    }

    /// Validate that a transported snapshot does not conflict with configured selectors.
    #[doc(hidden)]
    fn validate_snapshot_selector(
        &self,
        snapshot: &lance_table::snapshot::SnapshotRef,
    ) -> Result<()> {
        if self.version.is_some() || self.manifest.is_some() {
            return Err(Error::invalid_input(
                "snapshot_ref cannot be combined with version, branch, tag, or serialized_manifest",
            ));
        }
        snapshot.validate_table_uri(&self.table_uri)
    }

    /// Resolve the exact snapshot and validate its commit identity, conditional
    /// ETag, or full-file digest. The caller resolves table incarnation and storage
    /// authority independently; a reference carries no credentials or retention.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_core::Result;
    /// # use lance_table::snapshot::SnapshotRef;
    /// # use lance_table_runtime::{TableBuilder, TableSession};
    /// # async fn example(reference: &SnapshotRef) -> Result<()> {
    /// let table = TableBuilder::from_uri(reference.table_uri())
    ///     .load_snapshot_with_session(reference, Arc::new(TableSession::default())).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_snapshot_with_session(
        mut self,
        snapshot: &lance_table::snapshot::SnapshotRef,
        session: Arc<TableSession>,
    ) -> Result<Table> {
        self.validate_snapshot_selector(snapshot)?;
        self.apply_storage_options_override();
        let table_uri = self.table_uri.clone();
        let file_reader_options = self.file_reader_options.clone();
        let store_params = self.options.clone();
        let base_store_params = (!self.base_store_params.is_empty())
            .then(|| Arc::new(std::mem::take(&mut self.base_store_params)));
        let (object_store, base_path, commit_handler) = self.build_object_store().await?;
        let location = BranchLocation {
            path: base_path,
            uri: table_uri,
            branch: None,
        }
        .find_branch(snapshot.branch())?;
        let mut manifest_location = commit_handler
            .resolve_version_location(&location.path, snapshot.version(), &object_store.inner)
            .await?;
        let (manifest, digest) = if snapshot.manifest_digest().is_empty() {
            if let Some(expected) = snapshot.manifest_etag() {
                // Enforce the reference on the actual reads. A local listing may
                // omit HTTP quotes, so comparing it to a HEAD token is insufficient.
                manifest_location.e_tag = Some(expected.to_owned());
            }
            let (manifest, digest) = Table::load_manifest_impl(
                &object_store,
                &manifest_location,
                &location.uri,
                &session,
                crate::manifest::ManifestRead::Metadata,
            )
            .await?;
            (
                Arc::new(manifest),
                Arc::new(tokio::sync::OnceCell::new_with(digest)),
            )
        } else {
            Table::get_manifest_with_digest(
                &object_store,
                &manifest_location,
                &location.uri,
                &session,
            )
            .await?
        };
        let mut table = Table::checkout_manifest(
            object_store,
            location.path,
            location.uri,
            manifest,
            manifest_location,
            session,
            commit_handler,
            file_reader_options,
            Some(store_params),
            base_store_params,
        )
        .await?;
        table.snapshot_digest = digest;
        if !snapshot.manifest_digest().is_empty() {
            table.ensure_snapshot_digest().await?;
        }
        table.validate_snapshot(snapshot)?;
        Ok(table)
    }

    /// Sets `version` for the builder using a version number
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = Some(Ref::from(version));
        self
    }

    /// Sets `version` for the builder using a branch and optional version number
    /// If version_number is null, checkout the latest version
    pub fn with_branch(mut self, branch: &str, version_number: Option<u64>) -> Self {
        self.version = Some(Ref::from((branch, version_number)));
        self
    }

    /// Sets `version` for the builder using a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.version = Some(Ref::from(tag));
        self
    }

    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Sets the s3 credentials refresh.
    /// This only applies to s3 storage.
    pub fn with_s3_credentials_refresh_offset(mut self, offset: Duration) -> Self {
        self.options.s3_credentials_refresh_offset = offset;
        self
    }

    /// Sets the aws credentials provider.
    /// This only applies to aws object store.
    #[cfg(feature = "aws")]
    pub fn with_aws_credentials_provider(mut self, credentials: AwsCredentialProvider) -> Self {
        self.options.aws_credentials = Some(credentials);
        self
    }

    /// Directly set the object store to use.
    #[deprecated(note = "Implement an ObjectStoreProvider instead")]
    #[allow(deprecated)]
    pub fn with_object_store(
        mut self,
        object_store: Arc<DynObjectStore>,
        location: Url,
        commit_handler: Arc<dyn CommitHandler>,
    ) -> Self {
        self.options.object_store = Some((object_store, location));
        self.commit_handler = Some(commit_handler);
        self
    }

    /// Use a serialized manifest instead of loading it from the object store.
    ///
    /// This is common when transferring a dataset across IPC boundaries.
    pub fn with_serialized_manifest(mut self, manifest: &[u8]) -> Result<Self> {
        let manifest = Manifest::try_from(lance_table::format::pb::Manifest::decode(manifest)?)?;
        self.manifest = Some(manifest);
        Ok(self)
    }

    /// Set options used to initialize storage backend
    ///
    /// Options may be passed in the HashMap or set as environment variables. See documentation of
    /// underlying object store implementation for details.
    ///
    /// - [Azure options](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants)
    /// - [S3 options](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants)
    /// - [Google options](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants)
    ///
    /// Set `scheduler_error_mode` to `fail_fast` to return the first observed
    /// scheduler error and cancel the remaining physical reads in the same
    /// logical request. The default is `best_effort`.
    ///
    /// For datasets with additional registered base paths, a key of the form
    /// `base_<id>.<key>` applies `<key>` only to the base path with that
    /// manifest id, overriding the unscoped options that every base inherits.
    /// For example `base_1.account_key = abc` makes base 1 use
    /// `account_key = abc` while all other options are shared. Exact per-base
    /// bindings set via [`Self::with_base_store_params`] take precedence over
    /// base-scoped keys.
    pub fn with_storage_options(mut self, storage_options: HashMap<String, String>) -> Self {
        // Merge with existing options if accessor exists, otherwise create new static accessor
        if let Some(existing) = self.options.storage_options_accessor.take() {
            let mut merged = existing
                .initial_storage_options()
                .cloned()
                .unwrap_or_default();
            merged.extend(storage_options);
            if let Some(provider) = existing.provider().cloned() {
                self.options.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_initial_and_provider(merged, provider),
                ));
            } else {
                self.options.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(merged),
                ));
            }
        } else {
            self.options.storage_options_accessor = Some(Arc::new(
                StorageOptionsAccessor::with_static_options(storage_options),
            ));
        }
        self
    }

    /// Set a single option used to initialize storage backend
    ///
    /// ```
    /// use lance_table_runtime::TableBuilder;
    /// let builder = TableBuilder::from_uri("s3://bucket/path")
    ///     .with_storage_option("region", "us-east-1");
    /// ```
    pub fn with_storage_option(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        let mut storage_options = self.options.storage_options().cloned().unwrap_or_default();
        storage_options.insert(key.as_ref().to_string(), value.as_ref().to_string());

        // Merge with existing accessor if present
        if let Some(existing) = self.options.storage_options_accessor.take() {
            if let Some(provider) = existing.provider().cloned() {
                self.options.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_initial_and_provider(storage_options, provider),
                ));
            } else {
                self.options.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(storage_options),
                ));
            }
        } else {
            self.options.storage_options_accessor = Some(Arc::new(
                StorageOptionsAccessor::with_static_options(storage_options),
            ));
        }
        self
    }

    /// Enable credential vending from a LanceNamespace client
    ///
    /// Credentials will be automatically refreshed from the namespace client
    /// before they expire. The namespace client should return `expires_at_millis`
    /// in the storage_options from `describe_table()`.
    ///
    /// Use `with_s3_credentials_refresh_offset()` to configure how early
    /// credentials should be refreshed before they expire (default is 5 minutes).
    ///
    /// # Arguments
    /// * `provider` - The storage options provider to fetch credentials from
    ///
    ///
    /// // With custom refresh offset (refresh 10 minutes before expiration)
    /// let dataset = TableBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_provider(provider.clone())
    ///     .with_s3_credentials_refresh_offset(Duration::from_secs(600))
    ///     .load()
    ///     .await?;
    pub fn with_storage_options_provider(
        mut self,
        provider: Arc<dyn lance_io::object_store::StorageOptionsProvider>,
    ) -> Self {
        // Preserve existing storage options if any
        if let Some(existing) = self.options.storage_options_accessor.take() {
            if let Some(initial) = existing.initial_storage_options().cloned() {
                self.options.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_initial_and_provider(initial, provider),
                ));
            } else {
                self.options.storage_options_accessor =
                    Some(Arc::new(StorageOptionsAccessor::with_provider(provider)));
            }
        } else {
            self.options.storage_options_accessor =
                Some(Arc::new(StorageOptionsAccessor::with_provider(provider)));
        }
        self
    }

    /// Set a unified storage options accessor for credential management
    ///
    /// The accessor bundles static storage options with an optional dynamic provider,
    /// handling all caching and refresh logic internally.
    ///
    /// # Arguments
    /// * `accessor` - The storage options accessor
    ///
    pub fn with_storage_options_accessor(mut self, accessor: Arc<StorageOptionsAccessor>) -> Self {
        self.options.storage_options_accessor = Some(accessor);
        self
    }

    /// Set runtime-only object store params for a specific registered base path.
    ///
    /// These params are not persisted in the manifest. They are used as-is
    /// whenever the dataset resolves an object store for the given
    /// `BasePath.path`, taking precedence over `base_<id>.<key>` storage
    /// options (see [`Self::with_storage_options`]). Table-level store params
    /// remain the fallback for bases without an explicit binding.
    pub fn with_base_store_params(
        mut self,
        base_path: impl AsRef<str>,
        store_params: ObjectStoreParams,
    ) -> Self {
        self.base_store_params
            .insert(base_path.as_ref().to_string(), store_params);
        self
    }

    /// Share registry and metadata caches with other tables.
    pub fn with_session(mut self, session: Arc<TableSession>) -> Self {
        self.session = Some(session);
        self
    }

    /// Set the file reader defaults carried by the opened snapshot.
    pub fn with_file_reader_options(mut self, options: FileReaderOptions) -> Self {
        self.file_reader_options = Some(options);
        self
    }

    /// Set exact object store params used as the dataset-level default binding.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.options = store_params;
        self
    }

    /// Build a lance object store for the given config
    pub async fn build_object_store(
        mut self,
    ) -> Result<(Arc<ObjectStore>, Path, Arc<dyn CommitHandler>)> {
        let storage_options = self
            .options
            .storage_options()
            .cloned()
            .map(StorageOptions::new)
            .unwrap_or_default();
        let download_retry_count = storage_options.download_retry_count();

        let store_registry = self
            .session
            .as_ref()
            .map(|s| s.store_registry())
            .unwrap_or_default();

        #[allow(deprecated)]
        let (object_store, base_path) = match &self.options.object_store {
            Some(store) => (
                Arc::new(ObjectStore::new(
                    store.0.clone(),
                    store.1.clone(),
                    self.options.block_size,
                    self.options.object_store_wrapper.clone(),
                    self.options.use_constant_size_upload_parts,
                    store.1.scheme() != "file",
                    // If user supplied an object store then we just assume it's probably
                    // cloud-like
                    DEFAULT_CLOUD_IO_PARALLELISM,
                    download_retry_count,
                    None, // No storage_options available here
                )),
                Path::from(store.1.path()),
            ),
            None => {
                ObjectStore::from_uri_and_params(store_registry, &self.table_uri, &self.options)
                    .await?
            }
        };

        // Resolve the commit handler: an explicitly set one wins; otherwise a
        // namespace-managed table builds a manifest store rooted at the resolved
        // table path (the branch a request targets is derived per call from the
        // base path the handler is handed); otherwise fall back to the default
        // for the uri. Resolving here (not in load) keeps this pub method
        // consistent for every caller.
        let commit_handler: Arc<dyn CommitHandler> =
            if let Some(commit_handler) = self.commit_handler.take() {
                commit_handler
            } else if let Some((namespace_client, table_id)) = self.namespace_managed.take() {
                Arc::new(ExternalManifestCommitHandler {
                    external_manifest_store: Arc::new(LanceNamespaceExternalManifestStore::new(
                        namespace_client,
                        table_id,
                        base_path.clone(),
                    )),
                })
            } else {
                commit_handler_from_url(&self.table_uri, &Some(self.options.clone())).await?
            };

        Ok((object_store, base_path, commit_handler))
    }

    /// List manifest locations without reading the manifest contents.
    ///
    /// The returned locations are not guaranteed to be ordered. This operation may list and
    /// materialize the full manifest history. Explicit version, branch, and tag targets are not
    /// supported. Custom commit handlers and externally managed version stores are also not
    /// supported because listing physical manifest objects may omit committed versions whose
    /// authoritative locations are held outside the object store.
    pub async fn list_manifest_locations(mut self) -> Result<Vec<ManifestLocation>> {
        if self.version.is_some() {
            return Err(Error::invalid_input(
                "list_manifest_locations does not support an explicit version, branch, or tag",
            ));
        }
        let uses_external_or_custom_commit_handler = self.commit_handler.is_some()
            || self.namespace_managed.is_some()
            || Url::parse(&self.table_uri).is_ok_and(|url| url.scheme() == "s3+ddb");
        if uses_external_or_custom_commit_handler {
            return Err(Error::not_supported(
                "list_manifest_locations does not support external or custom commit handlers; \
                 object-store listing may omit committed manifest locations",
            ));
        }
        self.apply_storage_options_override();
        let (object_store, base_path, commit_handler) = self.build_object_store().await?;
        commit_handler
            .list_manifest_locations(&base_path, object_store.as_ref(), false)
            .try_collect()
            .await
    }

    /// Open the configured snapshot, resolving the latest version by default.
    pub async fn load(self) -> Result<Table> {
        let session = self.session.clone().unwrap_or_default();
        self.load_with_session(session).await
    }

    /// Open with a session supplied by the embedding builder. Only an explicit
    /// builder session is used to resolve stores during build_object_store.
    #[doc(hidden)]
    #[instrument(skip_all)]
    pub async fn load_with_session(self, session: Arc<TableSession>) -> Result<Table> {
        let uri = self.table_uri.clone();
        let target_ref = self.version.clone();
        match self.load_impl(session).boxed().await {
            Ok(table) => {
                info!(target: TRACE_DATASET_EVENTS, event=DATASET_LOADING_EVENT, uri=uri, target_ref = ?target_ref, version=table.manifest.version, status="success");
                Ok(table)
            }
            Err(e) => {
                info!(target: TRACE_DATASET_EVENTS, event=DATASET_LOADING_EVENT, uri=uri, target_ref = ?target_ref, status="error");
                Err(e)
            }
        }
    }

    // Runtime per-base overrides are supplied as storage options, but the dataset
    // ultimately resolves object stores from ObjectStoreParams. Normalize once in
    // the builder so reads only need to look up the prepared params by base path.
    fn merge_store_params_with_storage_options(
        params: &ObjectStoreParams,
        override_options: &HashMap<String, String>,
    ) -> ObjectStoreParams {
        if override_options.is_empty() {
            return params.clone();
        }

        let mut merged_params = params.clone();
        let mut merged_options = merged_params.storage_options().cloned().unwrap_or_default();
        merged_options.extend(override_options.clone());

        let storage_options_accessor = match merged_params
            .storage_options_accessor
            .as_ref()
            .and_then(|accessor| accessor.provider().cloned())
        {
            Some(provider) => Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                merged_options,
                provider,
            )),
            None => Arc::new(StorageOptionsAccessor::with_static_options(merged_options)),
        };
        merged_params.storage_options_accessor = Some(storage_options_accessor);
        merged_params
    }

    fn apply_storage_options_override(&mut self) {
        if let Some(override_options) = self.storage_options_override.take() {
            self.options =
                Self::merge_store_params_with_storage_options(&self.options, &override_options);
        }
    }

    async fn load_impl(mut self, session: Arc<TableSession>) -> Result<Table> {
        // Apply storage_options_override to merge namespace client options with any existing accessor
        self.apply_storage_options_override();

        let target_ref = self.version.clone();
        let table_uri = self.table_uri.clone();

        // How do we detect which version scheme is in use?
        let manifest = self.manifest.take();

        let file_reader_options = self.file_reader_options.clone();
        let store_params = self.options.clone();
        let base_store_params = (!self.base_store_params.is_empty())
            .then(|| Arc::new(std::mem::take(&mut self.base_store_params)));

        // A namespace-managed table is always addressed at its root uri, so the
        // effective branch is resolvable before loading: the base path is
        // qualified up front and the manifest store derives the branch from it.
        // An explicitly supplied commit handler opts out of the managed flow.
        let managed_store_active =
            self.namespace_managed.is_some() && self.commit_handler.is_none();

        let (object_store, base_path, commit_handler) = self.build_object_store().await?;

        // Two cases that need to check out after loading the manifest:
        // 1. If the target is configured as a branch, we need to check the branch field in the manifest
        // and reload the right branch in case the uri is not the right one.
        // 2. If the target is configured as a tag, and we don't find the tag under the table_uri,
        // we need to get the root_location after loading the manifest and get the right version.
        // In practice, we should try best to use the right uri and avoid double loading.
        let mut need_delay_checkout = false;
        let (mut branch, mut version_number) = match target_ref.clone() {
            Some(Ref::Version(branch, version_number)) => {
                if branch.is_some() && !managed_store_active {
                    need_delay_checkout = true;
                }
                (branch, version_number)
            }
            // We don't have a current branch context, just specify the branch as main
            // But the real branch will be specified by uri
            Some(Ref::VersionNumber(version_number)) => (None, Some(version_number)),
            // Here we assume the uri and path is the root.
            // If tag not found, we need to delay checkout after loading by uri
            Some(Ref::Tag(tag_name)) => {
                let refs = Refs::new(
                    object_store.clone(),
                    commit_handler.clone(),
                    BranchLocation {
                        path: base_path.clone(),
                        uri: table_uri.clone(),
                        branch: None,
                    },
                );
                match refs.tags().get(&tag_name).await {
                    Ok(tag_content) => {
                        if tag_content.branch.is_some() && !managed_store_active {
                            // The tag's chain lives under a different base path
                            // and the unmanaged handler resolves versions by
                            // base path only, so load the root's latest first
                            // and check the tag's branch/version out from it.
                            need_delay_checkout = true;
                            (tag_content.branch, None)
                        } else {
                            (tag_content.branch.clone(), Some(tag_content.version))
                        }
                    }
                    Err(e) => {
                        // A managed table is always rooted at the namespace
                        // location, so a tag missing here is missing.
                        if managed_store_active {
                            return Err(e);
                        }
                        need_delay_checkout = true;
                        (None, None)
                    }
                }
            }
            None => (None, None),
        };

        // Reject malformed branch names at the boundary (mirroring the branch
        // CRUD paths) so they fail as InvalidRef instead of resolving oddly
        if let Some(branch_name) = branch.as_deref()
            && !Branches::is_main_branch(Some(branch_name))
        {
            check_valid_branch(branch_name)?;
        }

        // For a managed table the branch is known before loading; point the base
        // path and uri at the branch chain so the loaded dataset is rooted there
        // (data placement, refs and the path-derived store branch all follow the
        // base path).
        let (base_path, table_uri) = if managed_store_active && branch.is_some() {
            let branch_location = BranchLocation {
                path: base_path,
                uri: table_uri,
                branch: None,
            }
            .find_branch(branch.as_deref())?;
            (branch_location.path, branch_location.uri)
        } else {
            (base_path, table_uri)
        };

        // A delayed checkout resolves (branch, version) against the right chain
        // afterwards, so the uri's chain only needs its latest version here: the
        // requested version number may not exist on that chain at all.
        let load_version = if need_delay_checkout {
            None
        } else {
            version_number
        };
        let table = Self::load_by_uri(
            session,
            manifest,
            file_reader_options,
            table_uri,
            load_version,
            object_store,
            base_path,
            commit_handler,
            Some(store_params),
            base_store_params,
        )
        .await?;

        if managed_store_active {
            // The base path was qualified above, so the loaded manifest must
            // already be on the requested branch; a mismatch means the namespace
            // resolved another chain.
            let requested_branch = branch.as_deref().and_then(standardize_branch);
            if table.manifest.branch.as_deref() != requested_branch.as_deref() {
                return Err(Error::internal(format!(
                    "open of branch '{}' resolved a manifest belonging to branch '{}'",
                    normalize_branch(branch.as_deref()),
                    normalize_branch(table.manifest.branch.as_deref()),
                )));
            }
        }

        if need_delay_checkout {
            if let Some(Ref::Tag(tag_name)) = target_ref {
                let tag_content = table.tags().get(tag_name.as_str()).await?;
                branch = tag_content.branch.clone();
                version_number = Some(tag_content.version);
            }

            let branch_differs = branch.as_deref() != table.manifest.branch.as_deref();
            let version_differs =
                version_number.is_some() && version_number != Some(table.manifest.version);
            if branch_differs || version_differs {
                return table
                    .checkout_version((branch.as_deref(), version_number))
                    .await;
            }
        }
        if let Some(version_number) = version_number
            && version_number != table.manifest.version
        {
            return Err(Error::VersionNotFound {
                message: format!("version {} not found", version_number),
            });
        }
        Ok(table)
    }

    /// Open a snapshot using an already resolved store and commit handler.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn load_by_uri(
        session: Arc<TableSession>,
        manifest: Option<Manifest>,
        file_reader_options: Option<FileReaderOptions>,
        table_uri: String,
        version_number: Option<u64>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        commit_handler: Arc<dyn CommitHandler>,
        store_params: Option<ObjectStoreParams>,
        base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    ) -> Result<Table> {
        let has_serialized_manifest = manifest.is_some();
        let (manifest, location, digest) = if let Some(mut manifest) = manifest {
            ensure_can_read_manifest(&manifest)?;
            check_manifest_storage_version(&mut manifest)?;
            let location = commit_handler
                .resolve_version_location(&base_path, manifest.version, &object_store.inner)
                .await?;
            if manifest.schema.has_dictionary_types() {
                let reader = object_store.open(&location.path).await?;
                populate_manifest_schema_dictionaries(&mut manifest, reader.as_ref()).await?;
            }
            (
                Arc::new(manifest),
                location,
                Arc::new(tokio::sync::OnceCell::new()),
            )
        } else {
            let mut manifest_location = match version_number {
                Some(version) => {
                    let target_manifest_result = commit_handler
                        .resolve_version_location(&base_path, version, &object_store.inner)
                        .await;
                    // This may fail due to the uri is not the right branch
                    // In this case we should try to load the latest version and checkout the right branch and version_number
                    match target_manifest_result {
                        Ok(manifest_location) => manifest_location,
                        Err(e) => {
                            if let Error::VersionNotFound { message: _ } = e {
                                // If the version is not found, we need to try to load the latest version.
                                commit_handler
                                    .resolve_latest_location(&base_path, &object_store)
                                    .await?
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }
                None => commit_handler
                    .resolve_latest_location(&base_path, &object_store)
                    .await
                    .map_err(|e| match &e {
                        Error::NotFound { .. } => {
                            Error::dataset_not_found(base_path.to_string(), Box::new(e))
                        }
                        _ => e,
                    })?,
            };

            crate::manifest::resolve_manifest_generation(&object_store, &mut manifest_location)
                .await?;
            let (manifest, digest) = Table::get_manifest_with_digest(
                &object_store,
                &manifest_location,
                &table_uri,
                session.as_ref(),
            )
            .await?;
            (manifest, manifest_location, digest)
        };

        let mut table = Table::checkout_manifest(
            object_store,
            base_path,
            table_uri,
            manifest,
            location,
            session,
            commit_handler,
            file_reader_options,
            store_params,
            base_store_params,
        )
        .await?;
        table.snapshot_digest = digest;
        table.state.is_manifest_generation_verified = !has_serialized_manifest;
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use arrow_array::record_batch;
    use lance_core::datatypes::Schema;
    use lance_file::{version::ConcreteFileVersion, versions::v2_1::Writer};
    use lance_table::format::{DataFile, DataStorageFormat, Fragment, IndexMetadata};
    use lance_table::io::commit::{ManifestNamingScheme, write_manifest_file_to_path};
    use lance_table::transaction::{Operation, Transaction};
    use prost_types::Any;
    use uuid::Uuid;

    use crate::caches::{IndexMetadataKey, TransactionKey};
    use crate::paths::DATA_DIR;

    use async_trait::async_trait;
    use lance_io::object_store::StorageOptionsProvider;
    use lance_table::io::commit::UnsafeCommitHandler;

    use super::*;
    use lance_namespace::models::DescribeTableResponse;
    use std::sync::Mutex;

    #[tokio::test]
    async fn opens_and_checks_out_table_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let builder = TableBuilder::from_uri(uri);
        let (store, base, _) = builder.clone().build_object_store().await.unwrap();

        let batch = record_batch!(("id", Int32, [10, 20])).unwrap();
        let schema = Schema::try_from(batch.schema().as_ref()).unwrap();
        let file_version = ConcreteFileVersion::V2_1;
        let file_name = "part.lance";
        let file = store
            .create(&base.clone().join(DATA_DIR).join(file_name))
            .await
            .unwrap();
        let mut writer = Writer::try_new(file, schema.clone(), Default::default()).unwrap();
        writer.write_batch(&batch).await.unwrap();
        let summary = writer.finish().await.unwrap();
        let data_file = DataFile::new(
            file_name,
            schema.field_ids(),
            vec![0],
            file_version,
            NonZero::new(summary.size_bytes),
            None,
        );
        let fragment = Fragment {
            files: vec![data_file],
            physical_rows: Some(summary.num_rows as usize),
            ..Fragment::new(0)
        };
        let mut first = Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::new(file_version),
            HashMap::new(),
        );
        let naming = ManifestNamingScheme::V2;
        write_manifest_file_to_path(
            &store,
            &mut first,
            None,
            &naming.manifest_path(&base, 1),
            None,
        )
        .await
        .unwrap();

        let second_file = "other.lance";
        store
            .copy(
                &base.clone().join(DATA_DIR).join(file_name),
                &base.clone().join(DATA_DIR).join(second_file),
            )
            .await
            .unwrap();
        let mut second_fragment = first.fragments[0].clone();
        second_fragment.id = 1;
        second_fragment.files[0].path = second_file.into();
        let transaction = Transaction::new(
            1,
            Operation::Append {
                fragments: vec![second_fragment.clone()],
            },
            None,
        );
        let mut second = Manifest::new_from_previous(
            &first,
            first.schema.clone(),
            Arc::new(vec![first.fragments[0].clone(), second_fragment]),
        );
        second.timestamp_nanos = 2_000_000_000;
        let index = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "opaque_index".into(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 2,
            fragment_bitmap: None,
            index_details: Some(Arc::new(Any {
                type_url: "type.googleapis.com/example.OpaqueIndex".into(),
                value: vec![1, 2, 3],
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: HashMap::new(),
        };
        write_manifest_file_to_path(
            &store,
            &mut second,
            Some(vec![index.clone()]),
            &naming.manifest_path(&base, 2),
            Some((&transaction).try_into().unwrap()),
        )
        .await
        .unwrap();

        let mut table = builder.with_version(1).load().await.unwrap();
        assert_eq!(table.version_id(), 1);
        assert_eq!(table.schema().fields[0].name, "id");
        assert_eq!(table.count_fragments(), 1);
        assert_eq!(table.fragments()[0].files[0].path, file_name);
        assert_eq!(table.fragments()[0].physical_rows, Some(2));
        assert!(table.fragment_bitmap().contains(0));
        assert_eq!(table.latest_version_id().await.unwrap(), 2);
        let versions = table.versions().await.unwrap();
        assert_eq!(
            versions
                .iter()
                .map(|version| version.version)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert!(
            versions
                .iter()
                .all(|version| version.manifest_size.is_some())
        );

        let session = table.session();
        let metadata_cache = session.metadata_cache().for_dataset(uri);
        let transaction_key = TransactionKey { version: 2 };
        let record = table.read_version_transaction(2).await.unwrap();
        assert_eq!(record.timestamp.timestamp(), 2);
        assert_eq!(record.transaction, Some(transaction.clone()));
        assert!(
            metadata_cache
                .get_with_key(&transaction_key)
                .await
                .is_none()
        );

        table.checkout_latest().await.unwrap();
        assert_eq!(table.manifest().version, 2);
        assert_eq!(table.count_fragments(), 2);
        assert!(table.fragment_bitmap().contains(1));
        assert_eq!(
            table.read_transaction().await.unwrap(),
            Some(transaction.clone())
        );
        assert_eq!(
            table.get_transactions(2).await.unwrap(),
            [Some(transaction), None]
        );
        let store = table.object_store(None).await.unwrap();
        let index_key = IndexMetadataKey {
            version: 2,
            store_identity: &store.store_prefix,
            e_tag: table.manifest_location().e_tag.as_deref(),
        };
        let cached_indices = session
            .index_cache()
            .with_key_prefix(uri)
            .get_with_key(&index_key)
            .await
            .unwrap();
        assert_eq!(cached_indices.as_slice(), [index]);
    }

    #[derive(Debug)]
    struct TestStorageOptionsProvider;

    #[derive(Debug)]
    struct TestNamespace;

    #[async_trait]
    impl LanceNamespace for TestNamespace {
        fn namespace_id(&self) -> String {
            "test-namespace".to_string()
        }
    }

    #[async_trait]
    impl StorageOptionsProvider for TestStorageOptionsProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            Ok(None)
        }

        fn provider_id(&self) -> String {
            "test-storage-options-provider".to_string()
        }
    }

    #[test]
    fn test_storage_options_override_wins_and_preserves_provider() {
        let caller_options = HashMap::from([
            ("endpoint".to_string(), "caller".to_string()),
            ("caller-only".to_string(), "caller-value".to_string()),
        ]);
        let provider: Arc<dyn StorageOptionsProvider> = Arc::new(TestStorageOptionsProvider);
        let options = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                StorageOptionsAccessor::with_initial_and_provider(caller_options, provider),
            )),
            ..Default::default()
        };
        let mut builder = TableBuilder::from_uri("memory://table").with_store_params(options);
        builder.storage_options_override = Some(HashMap::from([
            ("endpoint".to_string(), "namespace".to_string()),
            ("namespace-only".to_string(), "namespace-value".to_string()),
        ]));

        builder.apply_storage_options_override();

        let merged = builder.options.storage_options().unwrap();
        assert_eq!(merged.get("endpoint").unwrap(), "namespace");
        assert_eq!(merged.get("caller-only").unwrap(), "caller-value");
        assert_eq!(merged.get("namespace-only").unwrap(), "namespace-value");
        assert_eq!(
            builder
                .options
                .get_accessor()
                .unwrap()
                .provider()
                .unwrap()
                .provider_id(),
            "test-storage-options-provider"
        );
        assert!(builder.storage_options_override.is_none());
    }

    #[tokio::test]
    async fn test_list_manifest_locations_rejects_external_and_custom_commit_handlers() {
        let mut namespace_builder = TableBuilder::from_uri("memory://namespace-table");
        namespace_builder.namespace_managed =
            Some((Arc::new(TestNamespace), vec!["namespace-table".to_string()]));

        let builders = [
            TableBuilder::from_uri("memory://custom-handler-table")
                .with_commit_handler(Arc::new(UnsafeCommitHandler)),
            TableBuilder::from_uri("s3+ddb://bucket/table.lance?ddbTableName=manifest-table"),
            namespace_builder,
        ];

        for builder in builders {
            let err = builder.list_manifest_locations().await.unwrap_err();
            assert!(matches!(err, Error::NotSupported { .. }));
            assert!(
                err.to_string()
                    .contains("does not support external or custom commit handlers")
            );
        }
    }

    /// Mock namespace that records the describe request and vends storage
    /// options only when explicitly asked -- the behavior of a fail-closed
    /// server.
    #[derive(Debug, Default)]
    struct FailClosedNamespace {
        seen: Mutex<Option<DescribeTableRequest>>,
    }

    #[async_trait]
    impl LanceNamespace for FailClosedNamespace {
        fn namespace_id(&self) -> String {
            "fail-closed-mock".to_string()
        }

        async fn describe_table(
            &self,
            request: DescribeTableRequest,
        ) -> lance_namespace::Result<DescribeTableResponse> {
            let vend = request.vend_credentials == Some(true);
            *self.seen.lock().unwrap() = Some(request);
            Ok(DescribeTableResponse {
                location: Some("memory://ns-table".to_string()),
                storage_options: vend
                    .then(|| HashMap::from([("vended".to_string(), "true".to_string())])),
                ..Default::default()
            })
        }
    }

    /// The initial namespace describe must opt into credential vending
    /// explicitly: a fail-closed server returns no storage options for an
    /// unset flag, and the refresh provider is only installed when this first
    /// response already carries them (ENT-2008).
    #[tokio::test]
    async fn test_from_namespace_requests_vended_credentials() {
        let namespace = Arc::new(FailClosedNamespace::default());
        let builder = TableBuilder::from_namespace(namespace.clone(), vec!["t".to_string()])
            .await
            .unwrap();

        let seen = namespace.seen.lock().unwrap().take().unwrap();
        assert_eq!(seen.vend_credentials, Some(true));
        assert_eq!(seen.id, Some(vec!["t".to_string()]));
        // The vended options made it onto the builder, so the refresh
        // provider path (installed only when the first response carries
        // options) is reachable.
        assert_eq!(
            builder
                .storage_options_override
                .as_ref()
                .and_then(|o| o.get("vended"))
                .map(String::as_str),
            Some("true")
        );
    }
}
