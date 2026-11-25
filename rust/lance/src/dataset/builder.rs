// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc, time::Duration};

use lance_core::cache::CacheBackend;
use lance_file::reader::FileReaderOptions;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, StorageOptionsAccessor, StorageOptionsProvider,
};
use lance_namespace::LanceNamespace;
use lance_table::{
    format::Manifest,
    io::commit::{CommitHandler, ManifestLocation},
};
use lance_table_runtime::TableBuilder;
#[cfg(feature = "aws")]
use object_store::aws::AwsCredentialProvider;
use object_store::{DynObjectStore, path::Path};
use url::Url;

use super::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE, ReadParams, WriteParams};
use crate::{Dataset, Result, session::Session};

/// Configure and open a dataset snapshot.
#[derive(Clone)]
pub struct DatasetBuilder {
    table: TableBuilder,
    index_cache_size_bytes: usize,
    metadata_cache_size_bytes: usize,
    index_cache_backend: Option<Arc<dyn CacheBackend>>,
    session: Option<Arc<Session>>,
}

impl std::fmt::Debug for DatasetBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetBuilder")
            .field("table", &self.table)
            .field("index_cache_size_bytes", &self.index_cache_size_bytes)
            .field("metadata_cache_size_bytes", &self.metadata_cache_size_bytes)
            .field("session", &self.session.is_some())
            .finish()
    }
}

impl DatasetBuilder {
    /// Build an opener for a Lance dataset URI or local path.
    pub fn from_uri<T: AsRef<str>>(table_uri: T) -> Self {
        Self::from_table_builder(TableBuilder::from_uri(table_uri))
    }

    fn from_table_builder(table: TableBuilder) -> Self {
        Self {
            table,
            index_cache_size_bytes: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size_bytes: DEFAULT_METADATA_CACHE_SIZE,
            index_cache_backend: None,
            session: None,
        }
    }

    /// Create a DatasetBuilder from a LanceNamespace client
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
    /// # Example
    /// ```ignore
    /// use lance_namespace_impls::ConnectBuilder;
    /// use lance::dataset::DatasetBuilder;
    ///
    /// // Connect to a REST namespace
    /// let namespace_client = ConnectBuilder::new("rest")
    ///     .property("uri", "http://localhost:8080")
    ///     .connect()
    ///     .await?;
    ///
    /// // Load a dataset using storage options from namespace client
    /// let dataset = DatasetBuilder::from_namespace(
    ///     namespace_client,
    ///     vec!["my_table".to_string()],
    /// )
    /// .await?
    /// .load()
    /// .await?;
    /// ```
    pub async fn from_namespace(
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
    ) -> Result<Self> {
        let builder = TableBuilder::from_namespace(namespace_client, table_id).await?;
        Ok(Self::from_table_builder(builder))
    }

    /// Set the cache size for indices. Set to zero, to disable the cache.
    pub fn with_index_cache_size_bytes(mut self, cache_size: usize) -> Self {
        self.index_cache_size_bytes = cache_size;
        self
    }

    /// Use a custom index cache backend.
    ///
    /// When set, this overrides `with_index_cache_size_bytes` — the custom
    /// backend is responsible for its own capacity management.
    pub fn with_index_cache_backend(mut self, backend: Arc<dyn CacheBackend>) -> Self {
        self.index_cache_backend = Some(backend);
        self
    }

    /// Set the cache size for indices. Set to zero, to disable the cache.
    #[deprecated(since = "0.30.0", note = "Use `with_index_cache_size_bytes` instead")]
    pub fn with_index_cache_size(mut self, cache_size: usize) -> Self {
        let assumed_entry_size = 20 * 1024 * 1024; // 20 MiB per entry
        self.index_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// Size of the metadata cache in bytes. This cache stores metadata in memory
    /// for faster open table and scans. The default is 1 GiB.
    pub fn with_metadata_cache_size_bytes(mut self, cache_size: usize) -> Self {
        self.metadata_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `with_metadata_cache_size_bytes` instead"
    )]
    pub fn with_metadata_cache_size(mut self, cache_size: usize) -> Self {
        let assumed_entry_size = 10 * 1024 * 1024; // 10MB per entry
        self.metadata_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// The block size passed to the underlying Object Store reader.
    ///
    /// This is used to control the minimal request size.
    /// Defaults to 4KB for local files and 64KB for others
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.table = self.table.with_block_size(block_size);
        self
    }

    /// Sets `version` for the builder using a version number
    pub fn with_version(mut self, version: u64) -> Self {
        self.table = self.table.with_version(version);
        self
    }

    /// Sets `version` for the builder using a branch and optional version number
    /// If version_number is null, checkout the latest version
    pub fn with_branch(mut self, branch: &str, version_number: Option<u64>) -> Self {
        self.table = self.table.with_branch(branch, version_number);
        self
    }

    /// Sets `version` for the builder using a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.table = self.table.with_tag(tag);
        self
    }

    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.table = self.table.with_commit_handler(commit_handler);
        self
    }

    /// Sets the s3 credentials refresh.
    /// This only applies to s3 storage.
    pub fn with_s3_credentials_refresh_offset(mut self, offset: Duration) -> Self {
        self.table = self.table.with_s3_credentials_refresh_offset(offset);
        self
    }

    /// Sets the aws credentials provider.
    /// This only applies to aws object store.
    #[cfg(feature = "aws")]
    pub fn with_aws_credentials_provider(mut self, credentials: AwsCredentialProvider) -> Self {
        self.table = self.table.with_aws_credentials_provider(credentials);
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
        self.table = self
            .table
            .with_object_store(object_store, location, commit_handler);
        self
    }

    /// Use a serialized manifest instead of loading it from the object store.
    ///
    /// This is common when transferring a dataset across IPC boundaries.
    pub fn with_serialized_manifest(mut self, manifest: &[u8]) -> Result<Self> {
        self.table = self.table.with_serialized_manifest(manifest)?;
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
        self.table = self.table.with_storage_options(storage_options);
        self
    }

    /// Set a single option used to initialize storage backend
    /// For example, to set the region for S3, you can use:
    ///
    /// ```ignore
    /// let builder = DatasetBuilder::from_uri("s3://bucket/path")
    ///     .with_storage_option("region", "us-east-1");
    /// ```
    pub fn with_storage_option(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Self {
        self.table = self.table.with_storage_option(key, value);
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
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use std::time::Duration;
    /// use lance_namespace_impls::ConnectBuilder;
    /// use lance_io::object_store::{StorageOptionsProvider, LanceNamespaceStorageOptionsProvider};
    ///
    /// // Connect to a REST namespace
    /// let namespace_client = ConnectBuilder::new("rest")
    ///     .property("uri", "http://localhost:8080")
    ///     .connect()
    ///     .await?;
    ///
    /// // Create a storage options provider from namespace client
    /// let provider = Arc::new(LanceNamespaceStorageOptionsProvider::new(
    ///     namespace_client,
    ///     vec!["my_table".to_string()],
    /// ));
    ///
    /// // With default settings (5 minute refresh offset)
    /// let dataset = DatasetBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_provider(provider)
    ///     .load()
    ///     .await?;
    /// ```
    ///
    /// // With custom refresh offset (refresh 10 minutes before expiration)
    /// let dataset = DatasetBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_provider(provider.clone())
    ///     .with_s3_credentials_refresh_offset(Duration::from_secs(600))
    ///     .load()
    ///     .await?;
    pub fn with_storage_options_provider(
        mut self,
        provider: Arc<dyn StorageOptionsProvider>,
    ) -> Self {
        self.table = self.table.with_storage_options_provider(provider);
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
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use std::time::Duration;
    /// use lance_io::object_store::StorageOptionsAccessor;
    ///
    /// // Create an accessor with a dynamic provider
    /// let accessor = Arc::new(StorageOptionsAccessor::with_provider(
    ///     provider,
    ///     Duration::from_secs(300), // 5 minute refresh offset
    /// ));
    ///
    /// let dataset = DatasetBuilder::from_uri("s3://bucket/table.lance")
    ///     .with_storage_options_accessor(accessor)
    ///     .load()
    ///     .await?;
    /// ```
    pub fn with_storage_options_accessor(mut self, accessor: Arc<StorageOptionsAccessor>) -> Self {
        self.table = self.table.with_storage_options_accessor(accessor);
        self
    }

    /// Set runtime-only object store params for a specific registered base path.
    ///
    /// These params are not persisted in the manifest. They are used as-is
    /// whenever the dataset resolves an object store for the given
    /// `BasePath.path`, taking precedence over `base_<id>.<key>` storage
    /// options (see [`Self::with_storage_options`]). Dataset-level store params
    /// remain the fallback for bases without an explicit binding.
    pub fn with_base_store_params(
        mut self,
        base_path: impl AsRef<str>,
        store_params: ObjectStoreParams,
    ) -> Self {
        self.table = self.table.with_base_store_params(base_path, store_params);
        self
    }

    /// Set options based on [ReadParams].
    pub fn with_read_params(mut self, read_params: ReadParams) -> Self {
        self = self
            .with_index_cache_size_bytes(read_params.index_cache_size_bytes)
            .with_metadata_cache_size_bytes(read_params.metadata_cache_size_bytes);

        if let Some(options) = read_params.store_options {
            self.table = self.table.with_store_params(options);
        }
        if let Some(session) = read_params.session {
            self = self.with_session(session);
        }
        if let Some(commit_handler) = read_params.commit_handler {
            self.table = self.table.with_commit_handler(commit_handler);
        }
        if let Some(file_reader_options) = read_params.file_reader_options {
            self.table = self.table.with_file_reader_options(file_reader_options);
        }
        self
    }

    /// Set options based on [WriteParams].
    pub fn with_write_params(mut self, write_params: WriteParams) -> Self {
        if let Some(options) = write_params.store_params {
            self.table = self.table.with_store_params(options);
        }
        if let Some(commit_handler) = write_params.commit_handler {
            self.table = self.table.with_commit_handler(commit_handler);
        }
        self
    }

    /// Re-use an existing session and its caches.
    ///
    /// An explicit session takes precedence over cache sizes and backends
    /// configured on this builder.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.table = self.table.with_session(session.table_session.clone());
        self.session = Some(session);
        self
    }

    /// Set exact object store params used as the dataset-level default binding.
    pub fn with_store_params(mut self, store_params: ObjectStoreParams) -> Self {
        self.table = self.table.with_store_params(store_params);
        self
    }

    /// Build a Lance object store and commit handler for the given config.
    pub async fn build_object_store(
        self,
    ) -> Result<(Arc<ObjectStore>, Path, Arc<dyn CommitHandler>)> {
        self.table.build_object_store().await
    }

    /// List manifest locations without reading the manifest contents.
    ///
    /// The returned locations are not guaranteed to be ordered. This operation may list and
    /// materialize the full manifest history. Explicit version, branch, and tag targets are not
    /// supported. Custom commit handlers and externally managed version stores are also not
    /// supported because listing physical manifest objects may omit committed versions whose
    /// authoritative locations are held outside the object store.
    pub async fn list_manifest_locations(self) -> Result<Vec<ManifestLocation>> {
        Box::pin(self.table.list_manifest_locations()).await
    }

    /// Open the configured snapshot, resolving the latest version by default.
    pub async fn load(self) -> Result<Dataset> {
        let session = self
            .session
            .unwrap_or_else(|| match self.index_cache_backend {
                Some(backend) => Arc::new(Session::with_index_cache_backend(
                    backend,
                    self.metadata_cache_size_bytes,
                    Default::default(),
                )),
                None => Arc::new(Session::new(
                    self.index_cache_size_bytes,
                    self.metadata_cache_size_bytes,
                    Default::default(),
                )),
            });
        let table = self
            .table
            .load_with_session(session.table_session.clone())
            .await?;
        Ok(Dataset::from_table(table, session))
    }

    /// Open a snapshot using an already resolved store and commit handler.
    #[allow(clippy::too_many_arguments)]
    pub async fn load_by_uri(
        session: Arc<Session>,
        manifest: Option<Manifest>,
        file_reader_options: Option<FileReaderOptions>,
        table_uri: String,
        version_number: Option<u64>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        commit_handler: Arc<dyn CommitHandler>,
        store_params: Option<ObjectStoreParams>,
        base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    ) -> Result<Dataset> {
        let table = TableBuilder::load_by_uri(
            session.table_session.clone(),
            manifest,
            file_reader_options,
            table_uri,
            version_number,
            object_store,
            base_path,
            commit_handler,
            store_params,
            base_store_params,
        )
        .await?;
        Ok(Dataset::from_table(table, session))
    }
}
