// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance's runtime I/O boundary.
//!
//! ObjectStore integrations resolve authorized locations, credentials, and access
//! policies through [`ObjectStoreProvider`], [`StorageOptionsProvider`], and
//! [`WrappingObjectStore`]. Dataset and commit code use [`ObjectStore`] and
//! Lance readers and writers instead of the provider's underlying store.
//! Providers supply a [`StorageBackend`]. Wrappers intercept that same interface,
//! including multipart uploads, conditional publication, and bulk operations.
//! [`backend::native`] adapts existing `object_store` implementations at the edge.
//! Paths, requests, metadata, payloads, and errors are Lance-owned values. The
//! adapter converts them without exposing the native backend to consumers.
//!
//! Backend operations and conditional publication return [`Result`] so callers
//! can distinguish conflicts, missing objects, and unsupported atomic operations.
//! Reader/writer and directory conveniences return [`lance_core::Result`], mapping
//! missing objects to core NotFound and other backend failures to core IO errors.
//! Unsupported capabilities use [`Error::NotSupported`];
//! authorization and other failures never trigger an unwrapped fallback.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::ops::Range;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
use ::object_store::{ClientOptions, HeaderMap, HeaderValue};
use ::tracing::{Span, field::Empty, instrument};
use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::Stream;
use futures::{StreamExt, TryStreamExt, future, stream::BoxStream};
use lance_core::deepsize::DeepSizeOf;
use lance_core::error::LanceOptionExt;
use lance_core::utils::parse::{parse_env_as_bool, str_is_truthy};
use list_retry::ListRetryStream;
use tokio::io::AsyncWriteExt;
use url::Url;

use super::local::LocalObjectReader;
#[cfg(target_os = "linux")]
use crate::uring::{UringCurrentThreadReader, UringReader};
pub use lance_core::storage::{Error, Path, Result, path};
mod types;
pub use types::*;

pub mod backend;
pub use backend::StorageBackend;
mod copy;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub(crate) mod dynamic_credentials;
#[cfg(any(feature = "oss", feature = "huggingface", feature = "tos"))]
pub(crate) mod dynamic_opendal;
mod list_retry;
mod local_staging;
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(any(
    feature = "aws",
    feature = "gcp",
    feature = "azure",
    feature = "oss",
    feature = "tencent",
    feature = "huggingface",
    feature = "tos",
    feature = "goosefs",
))]
pub(crate) mod opendal_store;
pub mod providers;
pub(crate) mod read_dir;
pub mod storage_options;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod throttle;
mod tracing;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub mod transport;
use crate::object_reader::SmallReader;
use crate::object_writer::{LocalWriter, WriteResult};
use crate::scheduler::{SCHEDULER_ERROR_MODE_STORAGE_OPTION, SchedulerErrorMode};
use crate::traits::{WriteExt, Writer};
use crate::utils::tracking_store::{IOTracker, IoStats};
use crate::{object_reader::CloudObjectReader, object_writer::ObjectWriter, traits::Reader};
use lance_core::{Error as LanceError, Result as LanceResult};

// Local disks tend to do fine with a few threads
// Note: the number of threads here also impacts the number of files
// we need to read in some situations.  So keeping this at 8 keeps the
// RAM on our scanner down.
pub const DEFAULT_LOCAL_IO_PARALLELISM: usize = 8;
// Cloud disks often need many many threads to saturate the network
pub const DEFAULT_CLOUD_IO_PARALLELISM: usize = 64;

const SERVER_SIDE_COPY_ENABLED_ENV: &str = "LANCE_IO_SERVER_SIDE_COPY_ENABLED";

const DEFAULT_LOCAL_BLOCK_SIZE: usize = 4 * 1024; // 4KB block size
#[cfg(any(
    feature = "aws",
    feature = "gcp",
    feature = "azure",
    feature = "oss",
    feature = "tencent",
    feature = "huggingface",
    feature = "tos",
    feature = "goosefs",
))]
const DEFAULT_CLOUD_BLOCK_SIZE: usize = 64 * 1024; // 64KB block size

pub static DEFAULT_MAX_IOP_SIZE: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    std::env::var("LANCE_MAX_IOP_SIZE")
        .map(|val| val.parse().unwrap())
        .unwrap_or(16 * 1024 * 1024)
});

pub const DEFAULT_DOWNLOAD_RETRY_COUNT: usize = 3;

/// ObjectStore option that sets the registry-wide minimum prune threshold of the
/// object store cache; see [`StorageOptions::storage_cache_prune_threshold`].
pub const STORAGE_CACHE_PRUNE_THRESHOLD_OPTION: &str = "object_store_cache_prune_threshold";

#[derive(Debug)]
struct StreamCopyError {
    stage: &'static str,
    source_path: String,
    destination_path: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for StreamCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "multipart_stream_copy failed during {} from {} to {}: {}",
            self.stage, self.source_path, self.destination_path, self.source
        )
    }
}

impl std::error::Error for StreamCopyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

fn stream_copy_error(
    stage: &'static str,
    source_path: &Path,
    destination_path: &Path,
    source: impl std::error::Error + Send + Sync + 'static,
) -> LanceError {
    LanceError::io_source(Box::new(StreamCopyError {
        stage,
        source_path: source_path.to_string(),
        destination_path: destination_path.to_string(),
        source: Box::new(source),
    }))
}

pub use providers::{BackendConfig, ObjectStoreProvider, ObjectStoreRegistry};
pub use read_dir::ReadDirOptions;
pub use storage_options::{
    BASE_SCOPED_OPTION_PREFIX, BaseScopedStorageOptionsProvider, EXPIRES_AT_MILLIS_KEY,
    LanceNamespaceStorageOptionsProvider, REFRESH_OFFSET_MILLIS_KEY, StorageOptionsAccessor,
    StorageOptionsProvider, has_base_scoped_options, parse_base_scoped_key,
    resolve_base_scoped_options,
};

#[async_trait]
pub(super) trait LocalDirOperations: std::fmt::Debug + Send + Sync {
    fn directory_to_filesystem(&self, path: &Path) -> LanceResult<PathBuf>;
    async fn remove_dir_all(&self, path: &Path) -> LanceResult<()>;
    async fn canonicalize(&self, path: &Path) -> LanceResult<Option<PathBuf>>;
}

pub(crate) fn canonicalize_local_io_result(
    result: std::io::Result<PathBuf>,
) -> LanceResult<Option<PathBuf>> {
    match result {
        Ok(path) => Ok(Some(path)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Native file IO is granted only by the built-in local provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum LocalIo {
    #[default]
    Backend,
    Standard,
    Uring,
}

struct ExternalBlobStore {
    inner: Arc<dyn StorageBackend>,
    view: std::sync::Mutex<std::sync::Weak<ObjectStore>>,
}

impl ExternalBlobStore {
    fn new(inner: Arc<dyn StorageBackend>) -> Self {
        Self {
            inner,
            view: Default::default(),
        }
    }
}

impl Clone for ExternalBlobStore {
    fn clone(&self) -> Self {
        // A cloned ObjectStore is a separate binding and must derive its own view.
        Self::new(self.inner.clone())
    }
}

/// A configured Lance I/O handle, including its provider and wrapper policies.
///
/// Keep this handle when accessing a resolved storage integration: reconstructing
/// a store from its URI alone can discard credentials, scope, and cache identity.
#[derive(Clone)]
pub struct ObjectStore {
    /// Backend used by Lance readers, writers, and object operations.
    /// Install policies through [`Self::apply_wrapper`] to also wrap external views and listing.
    inner: Arc<dyn StorageBackend>,
    has_wrappers: bool,
    // The same physical store with external-blob wrapping rules.
    external_blob_inner: Option<ExternalBlobStore>,
    // Keep the registry's weak binding alive while only an external view is held.
    _external_blob_origin: Option<Arc<Self>>,
    // Provider-owned native directory operations for rooted local stores.
    local_dir_operations: Option<Arc<dyn LocalDirOperations>>,
    // Whether local writes sync file data and directory entries before returning.
    local_fsync: bool,
    // Enabled only by the built-in file provider, never by a URI alone.
    direct_local_io: bool,
    local_io: LocalIo,
    is_local: bool,
    is_remote: bool,
    max_single_copy_size: Option<u64>,
    scheme: String,
    block_size: usize,
    max_iop_size: u64,
    /// Whether to use constant size upload parts for multipart uploads. This
    /// is only necessary for Cloudflare R2.
    use_constant_size_upload_parts: bool,
    /// Whether we can assume that the list of files is lexically ordered. This
    /// is true for object stores, but not for local filesystems.
    list_is_lexically_ordered: bool,
    io_parallelism: usize,
    /// Number of times to retry a failed download
    download_retry_count: usize,
    /// Scheduler error handling validated at the store construction boundary.
    scheduler_error_mode: SchedulerErrorMode,
    /// IO tracker for monitoring read/write operations
    io_tracker: IOTracker,
    /// The datastore prefix that uniquely identifies this object store. It encodes information
    /// which usually cannot be found in the URL such as Azure account name. The prefix plus the
    /// path uniquely identifies any object inside the store.
    store_prefix: String,
}

// Keep debug output focused on the configured binding.
impl std::fmt::Debug for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStore")
            .field("inner", &self.inner)
            .field("scheme", &self.scheme)
            .field("local_fsync", &self.local_fsync)
            .field("block_size", &self.block_size)
            .field("max_iop_size", &self.max_iop_size)
            .field(
                "use_constant_size_upload_parts",
                &self.use_constant_size_upload_parts,
            )
            .field("list_is_lexically_ordered", &self.list_is_lexically_ordered)
            .field("io_parallelism", &self.io_parallelism)
            .field("download_retry_count", &self.download_retry_count)
            .field("io_tracker", &self.io_tracker)
            .field("store_prefix", &self.store_prefix)
            .finish()
    }
}

impl DeepSizeOf for ObjectStore {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // We aren't counting `inner` here which is problematic but an ObjectStore
        // shouldn't be too big.  The only exception might be the write cache but, if
        // the writer cache has data, it means we're using it somewhere else that isn't
        // a cache and so that doesn't really count.
        self.scheme.deep_size_of_children(context) + self.block_size.deep_size_of_children(context)
    }
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

/// Apply authorization, observation, or caching to Lance backend operations.
///
/// Wrappers operate on already-configured handles and must retain provider identity.
/// Installing one disables direct local file reads and writes. Native directory
/// maintenance retains the provider's path mapping and runs outside this wrapper.
/// Pagination is part of the wrapped backend and must obey the same policy.
/// Implement middleware directly against [`StorageBackend`].
pub trait WrappingObjectStore: std::fmt::Debug + Send + Sync {
    /// Wrap an object store with additional functionality
    ///
    /// The store_prefix is a string which uniquely identifies the object
    /// store being wrapped.
    fn wrap(
        &self,
        store_prefix: &str,
        original: Arc<dyn StorageBackend>,
    ) -> Arc<dyn StorageBackend>;

    /// Wrap a store used to read mutable, externally owned blob payloads.
    ///
    /// The default retains `wrapped`, including authorization and observation.
    /// `original` is the external view before this wrapper; `wrapped` is after
    /// applying the regular wrapper to that view. Immutable caches should return
    /// `original`, or apply a cache with explicit freshness validation.
    /// Managed payloads continue through [`Self::wrap`].
    fn wrap_external_blob(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn StorageBackend>,
        wrapped: Arc<dyn StorageBackend>,
    ) -> Arc<dyn StorageBackend> {
        wrapped
    }
}

/// Parameters to create an [ObjectStore]
///
#[derive(Debug, Clone)]
pub struct ObjectStoreParams {
    pub block_size: Option<usize>,
    /// Refresh offset for AWS credentials when using the built-in AWS SDK credential chain.
    /// For StorageOptionsAccessor, use `refresh_offset_millis` storage option instead.
    pub s3_credentials_refresh_offset: Duration,
    /// Backend policies applied in order, before per-binding IO tracking.
    pub backend_wrappers: Vec<Arc<dyn WrappingObjectStore>>,
    /// Wrap or replace the configured HTTP client used by native S3, GCS and Azure.
    ///
    /// The supplied client already has the resolved timeout, proxy and TLS settings.
    /// The wrapper sees every HTTP attempt, including retries. Lance's AIMD and
    /// metrics layers remain outside it. `None` uses the configured default client.
    ///
    /// This also covers credential clients opened by these stores (IMDS, STS web
    /// identity and GCP metadata). AWS SDK credential resolution, S3 region lookup
    /// and GCS external-account token exchange open their own clients.
    ///
    /// Stores and AIMD state are shared only for the same wrapper allocation.
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    pub http_client_wrapper: Option<Arc<dyn transport::HttpClientWrapper>>,
    /// Unified storage options accessor with caching and automatic refresh
    ///
    /// Provides storage options and optionally a dynamic provider for automatic
    /// credential refresh. Use `StorageOptionsAccessor::with_static_options()` for static
    /// options or `StorageOptionsAccessor::with_initial_and_provider()` for dynamic refresh.
    /// AWS providers return `aws_access_key_id`, `aws_secret_access_key`, and optionally
    /// `aws_session_token`. Set `expires_at_millis` for expiring credentials and
    /// `refresh_offset_millis` to control early refresh. The provider ID must distinguish
    /// credential scopes: equal IDs share cached stores.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_io::object_store::{StorageOptionsAccessor, StorageOptionsProvider, ObjectStoreParams};
    /// fn params(provider: Arc<dyn StorageOptionsProvider>) -> ObjectStoreParams {
    ///     ObjectStoreParams {
    ///         storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_provider(provider))),
    ///         ..Default::default()
    ///     }
    /// }
    /// ```
    pub storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    /// Use constant size upload parts for multipart uploads. Only necessary
    /// for Cloudflare R2, which doesn't support variable size parts. When this
    /// is false, max upload size is 2.5TB. When this is true, the max size is
    /// 50GB.
    pub use_constant_size_upload_parts: bool,
    pub list_is_lexically_ordered: Option<bool>,
}

impl Default for ObjectStoreParams {
    fn default() -> Self {
        Self {
            block_size: None,
            s3_credentials_refresh_offset: Duration::from_secs(60),
            backend_wrappers: Vec::new(),
            #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
            http_client_wrapper: None,
            storage_options_accessor: None,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: None,
        }
    }
}

impl ObjectStoreParams {
    /// Get the StorageOptionsAccessor from the params
    pub fn get_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.storage_options_accessor.clone()
    }

    /// Get storage options from the accessor, if any
    ///
    /// Returns the initial storage options from the accessor without triggering refresh.
    pub fn storage_options(&self) -> Option<&HashMap<String, String>> {
        self.storage_options_accessor
            .as_ref()
            .and_then(|a| a.initial_storage_options())
    }

    /// The block size to use: the explicit `block_size` parameter, else the
    /// `block_size` storage option, else `None` for the store's default.
    pub fn resolved_block_size(&self) -> LanceResult<Option<usize>> {
        if self.block_size.is_some() {
            return Ok(self.block_size);
        }
        match self.storage_options() {
            Some(options) => StorageOptions(options.clone()).block_size(),
            None => Ok(None),
        }
    }

    pub(crate) fn scheduler_error_mode(&self) -> LanceResult<SchedulerErrorMode> {
        parse_scheduler_error_mode(self.storage_options())
    }

    /// Resolve these params for a single base path scope.
    ///
    /// ObjectStore options may carry base-scoped entries (`base_<id>.<key>`) that
    /// apply only to one registered base path; see
    /// [`StorageOptionsAccessor::scoped_to_base`]. Returns the params unchanged
    /// when the storage options contain no base-scoped entries.
    pub fn scoped_to_base(&self, base_id: Option<u32>) -> Cow<'_, Self> {
        let Some(accessor) = &self.storage_options_accessor else {
            return Cow::Borrowed(self);
        };
        let scoped = accessor.scoped_to_base(base_id);
        if Arc::ptr_eq(&scoped, accessor) {
            Cow::Borrowed(self)
        } else {
            Cow::Owned(Self {
                storage_options_accessor: Some(scoped),
                ..self.clone()
            })
        }
    }
}

fn parse_scheduler_error_mode(
    storage_options: Option<&HashMap<String, String>>,
) -> LanceResult<SchedulerErrorMode> {
    let value = storage_options.and_then(|options| {
        options
            .get(SCHEDULER_ERROR_MODE_STORAGE_OPTION)
            .or_else(|| {
                options.iter().find_map(|(key, value)| {
                    key.eq_ignore_ascii_case(SCHEDULER_ERROR_MODE_STORAGE_OPTION)
                        .then_some(value)
                })
            })
    });
    value.map_or(Ok(SchedulerErrorMode::BestEffort), |value| value.parse())
}

fn allocation_ptr<T: ?Sized>(arc: &Arc<T>) -> *const () {
    // Trait object pointers include vtable metadata, which is not stable across codegen units.
    // Cache identity must follow the Arc allocation instead.
    Arc::as_ptr(arc).cast::<()>()
}

// We implement hash for caching
impl std::hash::Hash for ObjectStoreParams {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // For hashing, we use pointer values for wrappers
        self.block_size.hash(state);
        self.s3_credentials_refresh_offset.hash(state);
        self.backend_wrappers.len().hash(state);
        for wrapper in &self.backend_wrappers {
            allocation_ptr(wrapper).hash(state);
        }
        #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
        if let Some(wrapper) = &self.http_client_wrapper {
            allocation_ptr(wrapper).hash(state);
        }
        if let Some(accessor) = &self.storage_options_accessor {
            accessor.accessor_id().hash(state);
        }
        self.use_constant_size_upload_parts.hash(state);
        self.list_is_lexically_ordered.hash(state);
    }
}

// We implement eq for caching
impl Eq for ObjectStoreParams {}
impl PartialEq for ObjectStoreParams {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
        if self.http_client_wrapper.as_ref().map(allocation_ptr)
            != other.http_client_wrapper.as_ref().map(allocation_ptr)
        {
            return false;
        }

        // For equality, we use pointer comparison for wrappers
        // For accessor, we use accessor_id() for semantic equality
        self.block_size == other.block_size
            && self.s3_credentials_refresh_offset == other.s3_credentials_refresh_offset
            && self
                .backend_wrappers
                .iter()
                .map(allocation_ptr)
                .eq(other.backend_wrappers.iter().map(allocation_ptr))
            && self
                .storage_options_accessor
                .as_ref()
                .map(|a| a.accessor_id())
                == other
                    .storage_options_accessor
                    .as_ref()
                    .map(|a| a.accessor_id())
            && self.use_constant_size_upload_parts == other.use_constant_size_upload_parts
            && self.list_is_lexically_ordered == other.list_is_lexically_ordered
    }
}

/// Convert a URI string or local path to a URL
///
/// This function handles both proper URIs (with schemes like `file://`, `s3://`, etc.)
/// and plain local filesystem paths. On Windows, it correctly handles drive letters
/// that might be parsed as URL schemes.
///
/// # Examples
///
/// ```
/// # use lance_io::object_store::uri_to_url;
/// // URIs are preserved
/// let url = uri_to_url("s3://bucket/path").unwrap();
/// assert_eq!(url.scheme(), "s3");
///
/// // Local paths are converted to file:// URIs
/// # #[cfg(unix)]
/// let url = uri_to_url("/tmp/data").unwrap();
/// # #[cfg(unix)]
/// assert_eq!(url.scheme(), "file");
/// ```
pub fn uri_to_url(uri: &str) -> LanceResult<Url> {
    match Url::parse(uri) {
        Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
            // On Windows, the drive is parsed as a scheme
            local_path_to_url(uri)
        }
        Ok(url) => Ok(url),
        Err(_) => local_path_to_url(uri),
    }
}

fn expand_path(str_path: impl AsRef<str>) -> LanceResult<std::path::PathBuf> {
    let str_path = str_path.as_ref();
    let expanded = expand_tilde_path(str_path).unwrap_or_else(|| str_path.into());

    let mut expanded_path = path_abs::PathAbs::new(expanded)
        .map_err(|err| {
            LanceError::invalid_input(format!(
                "Invalid table location: '{str_path}' could not be resolved to an absolute path: {err}"
            ))
        })?
        .as_path()
        .to_path_buf();
    // path_abs::PathAbs::new(".") returns an empty string.
    if let Some(s) = expanded_path.as_path().to_str()
        && s.is_empty()
    {
        expanded_path = std::env::current_dir()?;
    }

    Ok(expanded_path)
}

fn expand_tilde_path(path: &str) -> Option<std::path::PathBuf> {
    let home_dir = std::env::home_dir()?;
    if path == "~" {
        return Some(home_dir);
    }
    if let Some(stripped) = path.strip_prefix("~/") {
        return Some(home_dir.join(stripped));
    }
    #[cfg(windows)]
    if let Some(stripped) = path.strip_prefix("~\\") {
        return Some(home_dir.join(stripped));
    }

    None
}

fn local_path_to_url(str_path: &str) -> LanceResult<Url> {
    let expanded_path = expand_path(str_path)?;

    Url::from_directory_path(expanded_path).map_err(|_| {
        LanceError::invalid_input_source(format!("Invalid table location: '{}'", str_path).into())
    })
}

#[cfg(feature = "huggingface")]
fn parse_hf_repo_id(url: &Url) -> LanceResult<String> {
    // Accept forms with repo type prefix (models/datasets/spaces) or legacy without.
    let mut segments: Vec<String> = Vec::new();
    if let Some(host) = url.host_str() {
        segments.push(host.to_string());
    }
    segments.extend(
        url.path()
            .trim_start_matches('/')
            .split('/')
            .map(|s| s.to_string()),
    );

    if segments.len() < 2 {
        return Err(LanceError::invalid_input(
            "Huggingface URL must contain at least owner and repo",
        ));
    }

    let repo_type_candidates = ["models", "datasets", "spaces"];
    let (owner, repo_with_rev) = if repo_type_candidates.contains(&segments[0].as_str()) {
        if segments.len() < 3 {
            return Err(LanceError::invalid_input(
                "Huggingface URL missing owner/repo after repo type",
            ));
        }
        (segments[1].as_str(), segments[2].as_str())
    } else {
        (segments[0].as_str(), segments[1].as_str())
    };

    let repo = repo_with_rev
        .split_once('@')
        .map(|(r, _)| r)
        .unwrap_or(repo_with_rev);
    Ok(format!("{owner}/{repo}"))
}

impl ObjectStore {
    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    ///
    /// This uses the default [ObjectStoreRegistry] to find the object store. To
    /// allow for potential re-use of object store instances, it's recommended to
    /// create a shared [ObjectStoreRegistry] and pass that to [Self::from_uri_and_params].
    pub async fn from_uri(uri: &str) -> LanceResult<(Arc<Self>, Path)> {
        let registry = Arc::new(ObjectStoreRegistry::default());

        Self::from_uri_and_params(registry, uri, &ObjectStoreParams::default()).await
    }

    /// Parse from a string URI.
    ///
    /// Returns the ObjectStore instance and the absolute path to the object.
    pub async fn from_uri_and_params(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        params: &ObjectStoreParams,
    ) -> LanceResult<(Arc<Self>, Path)> {
        Self::from_uri_and_params_impl(registry, uri, params, true).await
    }

    /// Parse a URI and build a fresh object store outside the registry cache.
    ///
    /// The caller must retain the returned store for as long as its
    /// provider-local state should be reused.
    #[doc(hidden)]
    pub async fn from_uri_and_params_uncached(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        params: &ObjectStoreParams,
    ) -> LanceResult<(Arc<Self>, Path)> {
        Self::from_uri_and_params_impl(registry, uri, params, false).await
    }

    async fn from_uri_and_params_impl(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        params: &ObjectStoreParams,
        use_registry_cache: bool,
    ) -> LanceResult<(Arc<Self>, Path)> {
        let url = uri_to_url(uri)?;

        let store = if use_registry_cache {
            registry.get_store(url.clone(), params).await?
        } else {
            registry.new_store(url.clone(), params).await?
        };
        // We know the scheme is valid if we got a store back.
        let provider = registry.get_provider(url.scheme()).expect_ok()?;
        let path = provider.extract_path(&url)?;

        Ok((store, path))
    }

    /// Extract the path component from a URI without initializing the object store.
    ///
    /// This is a synchronous operation that only parses the URI and extracts the path,
    /// without creating or initializing any object store instance.
    ///
    /// # Arguments
    ///
    /// * `registry` - The object store registry to get the provider
    /// * `uri` - The URI to extract the path from
    ///
    /// # Returns
    ///
    /// The extracted path component
    pub fn extract_path_from_uri(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
    ) -> LanceResult<Path> {
        let url = uri_to_url(uri)?;
        let provider = registry.get_provider(url.scheme()).ok_or_else(|| {
            LanceError::invalid_input(format!("Unknown scheme: {}", url.scheme()))
        })?;
        provider.extract_path(&url)
    }

    /// Build a fresh local handle using the registry's synchronous built-in assembly.
    pub fn local() -> Self {
        ObjectStoreRegistry::local()
    }

    /// Build a fresh in-memory handle using the registry's synchronous built-in assembly.
    pub fn memory() -> Self {
        ObjectStoreRegistry::memory()
    }

    /// Whether the provider declares local directory and path-alias semantics.
    /// This does not imply that native filesystem access is available.
    pub fn is_local(&self) -> bool {
        self.is_local
    }

    /// Returns true when direct native access is allowed and paths encode absolute local filesystem paths.
    ///
    /// Local stores rooted below the filesystem root, such as UNC-backed stores, use
    /// their backend implementation instead of direct filesystem access.
    pub fn has_direct_local_paths(&self) -> bool {
        self.direct_local_io && self.local_io != LocalIo::Backend && !self.has_io_wrapper()
    }

    // Policy wrappers only intercept backend calls. Native file operations must
    // not escape them, including external views whose wrapper state lives in
    // the origin binding.
    fn has_io_wrapper(&self) -> bool {
        self.has_wrappers
    }

    /// Whether local directory cleanup must enumerate backend entries before
    /// removing subtrees. This is a provider choice, unaffected by wrappers.
    pub fn uses_backend_directory_listing(&self) -> bool {
        self.direct_local_io && matches!(self.local_io, LocalIo::Backend)
    }

    /// Canonicalize a local object path, resolving rooted and UNC stores through
    /// their filesystem root mapping.
    ///
    /// Returns `Ok(None)` for non-local stores and when the filesystem identity cannot
    /// be resolved, including missing paths and unsupported local root mappings. Other
    /// local filesystem errors are returned to the caller.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// #
    /// # async fn same_directory(store: &ObjectStore, a: &Path, b: &Path) -> lance_core::Result<bool> {
    /// let a = store.canonicalize_local_path(a).await?;
    /// let b = store.canonicalize_local_path(b).await?;
    /// # Ok(a.is_some() && a == b)
    /// # }
    /// ```
    pub async fn canonicalize_local_path(&self, path: &Path) -> LanceResult<Option<PathBuf>> {
        if let Some(local_dir_operations) = &self.local_dir_operations {
            return local_dir_operations.canonicalize(path).await;
        }
        if self.direct_local_io {
            return canonicalize_local_io_result(
                tokio::fs::canonicalize(super::local::to_local_path(path)).await,
            );
        }
        Ok(None)
    }

    pub fn is_cloud(&self) -> bool {
        self.is_remote
    }

    /// Whether this object store prefers the lite scheduler.
    ///
    /// The lite scheduler is designed for backends like io_uring where
    /// tasks should only be polled when the consumer polls them.
    pub fn prefers_lite_scheduler(&self) -> bool {
        self.local_io == LocalIo::Uring
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    /// Stable identity for cache keys; assigned by the provider and registry.
    pub fn store_prefix(&self) -> &str {
        &self.store_prefix
    }

    /// Whether multipart uploads require equally sized non-final parts.
    pub fn use_constant_size_upload_parts(&self) -> bool {
        self.use_constant_size_upload_parts
    }

    /// Whether backend listings are ordered lexically by path.
    pub fn list_is_lexically_ordered(&self) -> bool {
        self.list_is_lexically_ordered
    }

    /// Resolve a directory through the provider's local root mapping.
    /// Returns None when the provider has no native directory mapping. Native
    /// directory maintenance is independent of backend wrappers; this mapping
    /// does not enable direct file reads or writes through a wrapper.
    pub fn local_directory_path(&self, path: &Path) -> LanceResult<Option<PathBuf>> {
        if !self.direct_local_io && self.local_dir_operations.is_none() {
            return Ok(None);
        }
        match &self.local_dir_operations {
            Some(operations) => operations.directory_to_filesystem(path).map(Some),
            None => Ok(Some(super::local::to_local_path(path).into())),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes one multipart upload of `total_bytes` to this store can hold in
    /// memory at once. See [`crate::object_writer::upload_memory_bound_bytes`].
    pub fn upload_memory_bound_bytes(&self, total_bytes: u64) -> u64 {
        crate::object_writer::upload_memory_bound_bytes(
            total_bytes,
            self.use_constant_size_upload_parts,
        )
    }

    pub fn max_iop_size(&self) -> u64 {
        self.max_iop_size
    }

    /// The amount of parallelism to use for I/O operations.
    ///
    /// Honors the `LANCE_IO_THREADS` override when set, otherwise the store's configured value.
    /// Always at least 1: callers feed this straight into `buffered` / `buffer_unordered`, and a
    /// window of 0 makes those streams never poll their input — e.g. a metadata-only `count_rows`
    /// would hang rather than return.
    pub fn io_parallelism(&self) -> usize {
        std::env::var("LANCE_IO_THREADS")
            .map(|val| val.parse::<usize>().unwrap())
            .unwrap_or(self.io_parallelism)
            .max(1)
    }

    /// Set the default I/O concurrency while retaining the configured backend and policies.
    ///
    /// Zero is rejected; the process-wide `LANCE_IO_THREADS` override still takes precedence.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// let storage = ObjectStore::memory().with_io_parallelism(4).unwrap();
    /// assert!(ObjectStore::memory().with_io_parallelism(0).is_err());
    /// ```
    pub fn with_io_parallelism(mut self, parallelism: usize) -> LanceResult<Self> {
        if parallelism == 0 {
            return Err(LanceError::invalid_input(
                "I/O parallelism must be greater than zero, got 0",
            ));
        }
        self.io_parallelism = parallelism;
        Ok(self)
    }

    pub(crate) fn scheduler_error_mode(&self) -> SchedulerErrorMode {
        self.scheduler_error_mode
    }

    /// Get the IO tracker for this object store
    ///
    /// The IO tracker can be used to get statistics about read/write operations
    /// performed on this object store.
    pub fn io_tracker(&self) -> &IOTracker {
        &self.io_tracker
    }

    /// A cumulative counter named `name` on the `metrics` facade, labelled
    /// like this store's object store metrics (`base`, and `dataset` in
    /// dataset mode) so work on the store's data is attributed the way its IO
    /// is. Unlabelled for a store that is not metered (see `meter_store`); a
    /// no-op without the `metrics` feature.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # fn example(store: &ObjectStore) {
    /// let counter = store.counter("lance_example_items_total");
    /// counter.increment(1);
    /// # }
    /// ```
    #[cfg(feature = "metrics")]
    pub fn counter(&self, name: &'static str) -> StoreCounter {
        let labels = self
            .io_tracker
            .metrics_base()
            .map(metrics::store_labels)
            .unwrap_or_default();
        StoreCounter(::metrics::counter!(name, labels))
    }

    #[cfg(not(feature = "metrics"))]
    pub fn counter(&self, _name: &'static str) -> StoreCounter {
        StoreCounter()
    }

    /// Get a snapshot of current IO statistics without resetting counters
    ///
    /// Returns the current IO statistics without modifying the internal state.
    /// Use this when you need to check stats without resetting them.
    pub fn io_stats_snapshot(&self) -> IoStats {
        self.io_tracker.stats()
    }

    /// Get incremental IO statistics since the last call to this method
    ///
    /// Returns the accumulated statistics since the last call and resets the
    /// counters to zero. This is useful for tracking IO operations between
    /// different stages of processing.
    pub fn io_stats_incremental(&self) -> IoStats {
        self.io_tracker.incremental_stats()
    }

    /// Add a policy to ordinary and external-blob operations, including pagination.
    /// Existing wrappers and instrumentation retain their position in the chain.
    pub fn apply_wrapper(&mut self, wrapper: &dyn WrappingObjectStore) {
        let external = self
            .external_blob_inner
            .as_ref()
            .map(|external| &external.inner)
            .unwrap_or(&self.inner)
            .clone();
        let inner = wrapper.wrap(&self.store_prefix, self.inner.clone());
        let external_wrapped = if Arc::ptr_eq(&self.inner, &external) {
            inner.clone()
        } else {
            wrapper.wrap(&self.store_prefix, external.clone())
        };
        let external = wrapper.wrap_external_blob(&self.store_prefix, external, external_wrapped);
        self.inner = inner;
        self.has_wrappers = true;
        self.external_blob_inner = Some(ExternalBlobStore::new(external));
    }

    // Registry-created bindings track both views once, before any late wrappers.
    fn track_io(&mut self) {
        if let Some(external) = &mut self.external_blob_inner {
            external.inner = self
                .io_tracker
                .wrap(&self.store_prefix, external.inner.clone());
        }
        self.inner = self.io_tracker.wrap(&self.store_prefix, self.inner.clone());
    }

    /// Derive an external-blob read view without reopening the provider.
    ///
    /// The physical store, credentials, endpoint, and connection pool are retained.
    /// Only wrapper behavior changes through [`WrappingObjectStore::wrap_external_blob`].
    /// Managed blobs must keep using the original store. Repeated calls reuse the
    /// same view while it is held by a reader or scheduler. The cached reference
    /// is weak so the view can retain this registry binding without a cycle.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// let store = Arc::new(ObjectStore::memory());
    /// let external = store.for_external_blobs();
    /// assert_eq!(external.store_prefix(), store.store_prefix());
    /// ```
    pub fn for_external_blobs(self: &Arc<Self>) -> Arc<Self> {
        let Some(external) = self.external_blob_inner.as_ref() else {
            return self.clone();
        };
        let mut view = external
            .view
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(store) = view.upgrade() {
            return store;
        }
        let mut store = self.as_ref().clone();
        store.inner = external.inner.clone();
        store.external_blob_inner = None;
        store._external_blob_origin = Some(self.clone());
        let store = Arc::new(store);
        *view = Arc::downgrade(&store);
        store
    }

    /// Open a file for path.
    ///
    /// Parameters
    /// - ``path``: Absolute path to the file.
    pub async fn open(&self, path: &Path) -> LanceResult<Box<dyn Reader>> {
        match self.local_io {
            LocalIo::Standard if self.has_direct_local_paths() => {
                LocalObjectReader::open_with_tracker(
                    path,
                    self.block_size,
                    None,
                    Arc::new(self.io_tracker.clone()),
                )
                .await
            }
            #[cfg(target_os = "linux")]
            LocalIo::Uring if self.has_direct_local_paths() => {
                // Check if current-thread mode enabled
                let use_current_thread = std::env::var("LANCE_URING_CURRENT_THREAD")
                    .map(|v| str_is_truthy(&v))
                    .unwrap_or(false);

                if use_current_thread {
                    UringCurrentThreadReader::open(
                        path,
                        self.block_size,
                        None,
                        Arc::new(self.io_tracker.clone()),
                    )
                    .await
                } else {
                    UringReader::open(
                        path,
                        self.block_size,
                        None,
                        Arc::new(self.io_tracker.clone()),
                    )
                    .await
                }
            }
            _ => Ok(Box::new(
                CloudObjectReader::new(
                    self.inner.clone(),
                    path.clone(),
                    self.block_size,
                    None,
                    self.download_retry_count,
                )?
                .with_io_parallelism(self.io_parallelism()),
            )),
        }
    }

    /// Open a reader for a file with known size.
    ///
    /// This size may either have been retrieved from a list operation or
    /// cached metadata. By passing in the known size, we can skip a HEAD / metadata
    /// call.
    pub async fn open_with_size(
        &self,
        path: &Path,
        known_size: usize,
    ) -> LanceResult<Box<dyn Reader>> {
        // If we know the file is really small, we can read the whole thing
        // as a single request.
        if known_size <= self.block_size {
            return Ok(Box::new(SmallReader::new(
                self.inner.clone(),
                path.clone(),
                self.download_retry_count,
                known_size,
            )));
        }

        match self.local_io {
            LocalIo::Standard if self.has_direct_local_paths() => {
                LocalObjectReader::open_with_tracker(
                    path,
                    self.block_size,
                    Some(known_size),
                    Arc::new(self.io_tracker.clone()),
                )
                .await
            }
            #[cfg(target_os = "linux")]
            LocalIo::Uring if self.has_direct_local_paths() => {
                // Check if current-thread mode enabled
                let use_current_thread = std::env::var("LANCE_URING_CURRENT_THREAD")
                    .map(|v| str_is_truthy(&v))
                    .unwrap_or(false);

                if use_current_thread {
                    UringCurrentThreadReader::open(
                        path,
                        self.block_size,
                        Some(known_size),
                        Arc::new(self.io_tracker.clone()),
                    )
                    .await
                } else {
                    UringReader::open(
                        path,
                        self.block_size,
                        Some(known_size),
                        Arc::new(self.io_tracker.clone()),
                    )
                    .await
                }
            }
            _ => Ok(Box::new(
                CloudObjectReader::new(
                    self.inner.clone(),
                    path.clone(),
                    self.block_size,
                    Some(known_size),
                    self.download_retry_count,
                )?
                .with_io_parallelism(self.io_parallelism()),
            )),
        }
    }

    /// Open a file after verifying its complete length against a checkpoint.
    ///
    /// Files within the existing small-file threshold are checked and buffered
    /// by the same GET. Larger files retain the metadata check and range reader.
    /// This does not pin a version of a mutable large file across later reads.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// #
    /// # async fn example(store: &ObjectStore, path: &Path) -> lance_core::Result<()> {
    /// let reader = store.open_with_expected_size(path, 1024).await?;
    /// let bytes = reader.get_range(0..16).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open_with_expected_size(
        &self,
        path: &Path,
        expected_size: u64,
    ) -> LanceResult<Box<dyn Reader>> {
        if expected_size <= self.block_size as u64 {
            let reader = SmallReader::open_with_expected_size(
                self.inner.clone(),
                path.clone(),
                self.download_retry_count,
                expected_size as usize,
            )
            .await?;
            return Ok(Box::new(reader));
        }
        let actual_size = self.size(path).await?;
        if actual_size != expected_size {
            return Err(LanceError::invalid_input(format!(
                "object at '{path}' has {actual_size} bytes, checkpoint expects {expected_size}"
            )));
        }
        let size = usize::try_from(actual_size).map_err(|_| {
            LanceError::invalid_input(format!(
                "object at '{path}' is too large for this platform: {actual_size} bytes"
            ))
        })?;
        self.open_with_size(path, size).await
    }

    /// Create an [ObjectWriter] from local [std::path::Path]
    pub async fn create_local_writer(path: &std::path::Path) -> LanceResult<ObjectWriter> {
        let storage = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        ObjectWriter::new(&storage, &os_path).await
    }

    /// Open an [Reader] from local [std::path::Path]
    pub async fn open_local(path: &std::path::Path) -> LanceResult<Box<dyn Reader>> {
        let storage = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        storage.open(&os_path).await
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> LanceResult<Box<dyn Writer>> {
        match self.local_io {
            LocalIo::Standard if self.has_direct_local_paths() => {
                let local_path = super::local::to_local_path(path);
                let local_path = std::path::PathBuf::from(&local_path);
                let parent = local_path
                    .parent()
                    .expect("file path must have parent")
                    .to_owned();
                let local_fsync = self.local_fsync;
                let named_temp = tokio::task::spawn_blocking(move || {
                    if local_fsync {
                        super::local::create_dir_all_durable(&parent)?;
                    } else {
                        std::fs::create_dir_all(&parent)?;
                    }
                    #[cfg(unix)]
                    {
                        // NamedTempFile defaults to 0o600. Use ordinary file creation permissions so the published file honors the caller's umask.
                        tempfile::Builder::new()
                            .permissions(std::fs::Permissions::from_mode(0o666))
                            .tempfile_in(parent)
                    }
                    #[cfg(not(unix))]
                    tempfile::NamedTempFile::new_in(parent)
                })
                .await
                .map_err(|e| LanceError::io(format!("spawn_blocking failed: {}", e)))??;
                let (std_file, temp_path) = named_temp.into_parts();
                let file = tokio::fs::File::from_std(std_file);
                Ok(Box::new(
                    LocalWriter::new(
                        file,
                        path.clone(),
                        temp_path,
                        Arc::new(self.io_tracker.clone()),
                    )
                    .with_fsync(local_fsync),
                ))
            }
            _ => Ok(Box::new(ObjectWriter::new(self, path).await?)),
        }
    }

    /// Read an object with its metadata and a streaming body.
    ///
    /// The returned body may be consumed as bytes or as a stream. Requests pass
    /// through the configured backend wrappers, including external blob policies.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path, Result};
    /// # async fn example(storage: &ObjectStore, path: &Path) -> Result<()> {
    /// let response = storage.read_object(path).await?;
    /// let metadata = response.meta.clone();
    /// let body = response.into_stream();
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_object(
        &self,
        path: &Path,
    ) -> crate::object_store::Result<crate::object_store::GetResult> {
        self.inner.get(path).await
    }

    /// Begin a multipart upload through the configured backend policies.
    ///
    /// Completion publishes the object; callers must abort an unfinished upload.
    pub async fn start_upload(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> Result<Box<dyn backend::MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    /// Publish a payload through the configured backend without collecting its chunks.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, PutPayload, Result};
    /// # async fn example(storage: &ObjectStore) -> Result<()> {
    /// let written = storage.put_payload(&"data".into(), PutPayload::from_static(b"data")).await?;
    /// assert_eq!(written.size, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put_payload(
        &self,
        path: &Path,
        payload: PutPayload,
    ) -> crate::object_store::Result<WriteResult> {
        self.put_payload_opts(path, payload, PutOptions::default())
            .await
    }

    /// Publish a payload with backend preconditions, without adding staging or
    /// fallback requests. Unsupported conditions are returned to the caller.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, PutMode, PutOptions, Result};
    /// # async fn example(storage: &ObjectStore) -> Result<()> {
    /// storage.put_payload_opts(&"manifest".into(), b"data".to_vec().into(),
    ///     PutOptions { mode: PutMode::Create, ..Default::default() }).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put_payload_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> crate::object_store::Result<WriteResult> {
        let size = payload.content_length();
        let result = self.inner.put_opts(path, payload, options).await?;
        Ok(WriteResult {
            size,
            e_tag: result.e_tag,
        })
    }

    /// A helper function to create a file and write content to it.
    pub async fn put(&self, path: &Path, content: &[u8]) -> LanceResult<WriteResult> {
        let mut writer = self.create(path).await?;
        writer.write_all(content).await?;
        Writer::shutdown(writer.as_mut()).await
    }

    /// Atomically creates an object without replacing an existing object.
    ///
    /// Native file access publishes a staging object with a conditional rename.
    /// Object-backend access uses its conditional create operation. Tencent COS
    /// is rejected because it can silently ignore conditional create requests.
    ///
    /// Returns [`crate::object_store::Error::NotSupported`] when the backend cannot
    /// provide conditional publication. Staging cleanup is best effort.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// let written = store.put_if_absent(&"manifest".into(), b"data".to_vec().into()).await?;
    /// assert_eq!(written.size, 4);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put_if_absent(
        &self,
        path: &Path,
        content: PutPayload,
    ) -> crate::object_store::Result<WriteResult> {
        if self.scheme == "cos" {
            return Err(crate::object_store::Error::NotSupported {
                source: "Tencent COS cannot guarantee put-if-not-exists".into(),
            });
        }
        let size = content.content_length();
        if !matches!(self.local_io, LocalIo::Backend) || self.local_dir_operations.is_some() {
            // Preserve the parent's encoding and keep dotfile names out of the
            // staging basename, which some local backends reserve.
            let staging_name = format!("lance-io.staging.{}", uuid::Uuid::new_v4().simple());
            let staging_path = path
                .parent()
                .unwrap_or_default()
                .join(staging_name.as_str());
            self.inner.put(&staging_path, content).await?;
            let result = self.inner.rename_if_not_exists(&staging_path, path).await;
            if result.is_err()
                && let Err(error) = self.inner.delete(&staging_path).await
            {
                log::warn!(
                    "Failed to remove staging object {} after atomic create failed: {}",
                    staging_path,
                    error
                );
            }
            result.map(|()| WriteResult { size, e_tag: None })
        } else {
            self.put_payload_opts(
                path,
                content,
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
        }
    }

    /// Read object metadata through the configured provider and wrappers.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// let metadata = store.stat(&"data.lance".into()).await?;
    /// assert_eq!(metadata.location.as_ref(), "data.lance");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stat(&self, path: &Path) -> crate::object_store::Result<ObjectMeta> {
        self.inner.head(path).await
    }

    /// Move an object, replacing the destination if it exists.
    ///
    /// Atomicity follows the backend; use [`Self::rename_if_not_exists`] when
    /// publishing a new manifest without replacing a concurrent writer's object.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// store.rename(&"old".into(), &"new".into()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rename(&self, from: &Path, to: &Path) -> crate::object_store::Result<()> {
        self.inner.rename(from, to).await
    }

    /// Publish `from` at `to` only if `to` does not exist.
    ///
    /// The backend must enforce the condition atomically or report that the
    /// operation is unsupported. No existence check is used to emulate it.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// store.rename_if_not_exists(&"staging".into(), &"manifest".into()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rename_if_not_exists(
        &self,
        from: &Path,
        to: &Path,
    ) -> crate::object_store::Result<()> {
        self.inner.rename_if_not_exists(from, to).await
    }

    /// Copy without replacing an existing destination.
    ///
    /// The backend must enforce the condition atomically or return
    /// [`Error::NotSupported`]. The caller owns any publication fallback.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// store.copy_if_not_exists(&"staging".into(), &"manifest".into()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn copy_if_not_exists(
        &self,
        from: &Path,
        to: &Path,
    ) -> crate::object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }

    /// Delete an object, including a local staging file hidden from normal listings.
    ///
    /// See [`Self::read_local_staging_files`] for an example of finding and removing
    /// old staging files after an interrupted write.
    pub async fn delete(&self, path: &Path) -> LanceResult<()> {
        if let Some(local_path) = self.local_staging_path(path)? {
            // LocalFileSystem rejects its reserved staging names, even for deletion.
            let metrics = self.io_tracker.begin_io("delete");
            let result = tokio::fs::remove_file(&local_path).await.map_err(|error| {
                if error.kind() == ErrorKind::NotFound {
                    LanceError::not_found(path.to_string())
                } else {
                    LanceError::io(format!(
                        "Failed to remove local staging file {}: {error}",
                        local_path.display()
                    ))
                }
            });
            metrics.record(&result, 0);
            return result;
        }
        self.inner.delete(path).await?;
        Ok(())
    }

    /// Copy an object, streaming through multipart upload when it exceeds the
    /// provider's single-request copy limit.
    pub async fn copy(&self, from: &Path, to: &Path) -> LanceResult<()> {
        if self.has_direct_local_paths() {
            // Native filesystem copies bypass the writer's file and directory syncs.
            if self.local_fsync {
                self.copy_via_stream(from, self, to).await?;
                return Ok(());
            }
            // Use std::fs::copy for local filesystem to support cross-filesystem copies
            let metrics = self.io_tracker.begin_io("copy");
            let result = super::local::copy_file(from, to);
            metrics.record(&result, 0);
            return result;
        }
        if let Some(max_single_copy) = self.max_single_copy_size {
            // Reuse the reader for both the size lookup (a single cached HEAD)
            // and the streamed copy, avoiding a separate HEAD request.
            let reader = self.open(from).await?;
            if reader.size().await? as u64 > max_single_copy {
                let mut writer = self.create(to).await?;
                writer.copy_from_reader(reader.as_ref()).await?;
                Writer::shutdown(writer.as_mut()).await?;
                return Ok(());
            }
        }
        Ok(self.inner.copy(from, to).await?)
    }

    /// Copy an object using the policy for bulk file movement.
    ///
    /// Streaming is the default because it works across object stores and does
    /// not require provider-native copy support. Setting
    /// `LANCE_IO_SERVER_SIDE_COPY_ENABLED` to a truthy value opts same-store
    /// copies into [`Self::copy`]. Cross-store and local copies continue to use
    /// [`Self::copy_via_stream`].
    ///
    /// ```no_run
    /// # use lance_core::Result;
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// #
    /// # async fn copy(source: &ObjectStore, destination: &ObjectStore) -> lance_core::Result<()> {
    /// source
    ///     .copy_bulk(
    ///         &Path::from("staging/index.lance"),
    ///         destination,
    ///         &Path::from("index.lance"),
    ///     )
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn copy_bulk(
        &self,
        source_path: &Path,
        destination_store: &Self,
        destination_path: &Path,
    ) -> LanceResult<WriteResult> {
        self.copy_bulk_with_server_side_copy(
            source_path,
            destination_store,
            destination_path,
            self.uses_server_side_copy(destination_store),
        )
        .await
    }

    fn uses_server_side_copy(&self, destination_store: &Self) -> bool {
        parse_env_as_bool(SERVER_SIDE_COPY_ENABLED_ENV, false)
            && self.can_server_side_copy_to(destination_store)
    }

    async fn copy_bulk_with_server_side_copy(
        &self,
        source_path: &Path,
        destination_store: &Self,
        destination_path: &Path,
        server_side_copy_enabled: bool,
    ) -> LanceResult<WriteResult> {
        if !server_side_copy_enabled || !self.can_server_side_copy_to(destination_store) {
            return self
                .copy_via_stream(source_path, destination_store, destination_path)
                .await;
        }

        let source_size = self.size(source_path).await?;
        let result_size = usize::try_from(source_size).map_err(|source| {
            LanceError::io(format!(
                "server-side copy source size conversion failed from {source_path} to \
                 {destination_path}: source_size={source_size}, error={source}"
            ))
        })?;
        destination_store
            .copy(source_path, destination_path)
            .await?;
        let destination_size = destination_store.size(destination_path).await?;
        if destination_size != source_size {
            return Err(LanceError::io(format!(
                "server-side copy destination size mismatch from {source_path} to \
                 {destination_path}: source_size={source_size}, \
                 destination_size={destination_size}"
            )));
        }

        Ok(WriteResult {
            size: result_size,
            e_tag: None,
        })
    }

    fn can_server_side_copy_to(&self, destination_store: &Self) -> bool {
        // Prefixes can collide across endpoints or wrappers, where native copy could
        // read or write the wrong backend. Exact client identity is required.
        self.is_cloud()
            && destination_store.is_cloud()
            && Arc::ptr_eq(&self.inner, &destination_store.inner)
    }

    /// Copy an object by streaming its bytes through Lance's multipart-aware writer.
    ///
    /// Unlike [`Self::copy`], this never delegates to a provider-native server-side
    /// copy. The source and destination may use different object stores. The copy
    /// succeeds only after the byte count reported by the writer and a destination
    /// metadata lookup both match the source size.
    /// Local copies use the filesystem's native copy unless the destination
    /// enables `local_fsync`, in which case its durable local writer is used.
    ///
    /// ```no_run
    /// # use lance_core::Result;
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// #
    /// # async fn copy(source: &ObjectStore, destination: &ObjectStore) -> lance_core::Result<()> {
    /// source
    ///     .copy_via_stream(
    ///         &Path::from("staging/index.lance"),
    ///         destination,
    ///         &Path::from("index.lance"),
    ///     )
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(
        name = "multipart_stream_copy",
        level = "info",
        skip(self, source_path, destination_store, destination_path),
        fields(
            source = %source_path,
            destination = %destination_path,
            source_size = Empty,
            read_chunk_size = Empty,
            multipart_part_size = crate::object_writer::initial_upload_size(),
            multipart_concurrency = crate::object_writer::max_upload_parallelism(),
            part_count = Empty,
            bytes_transferred = Empty,
            destination_size = Empty,
            validation = Empty,
            elapsed_ms = Empty,
        ),
        err
    )]
    pub async fn copy_via_stream(
        &self,
        source_path: &Path,
        destination_store: &Self,
        destination_path: &Path,
    ) -> LanceResult<WriteResult> {
        self.copy_via_stream_inner(source_path, destination_store, destination_path)
            .await
    }

    /// Copy within this store without replacing an existing destination.
    ///
    /// This uses the provider's native conditional copy. An existing destination
    /// with the same size is treated as a successful retry.
    /// This only validates size; callers must ensure the destination belongs to
    /// the same immutable file publication. Metadata commits requiring content
    /// equality must use [`Self::copy_if_not_exists`] and verify conflicts.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// store.copy_if_not_exists_or_same_size(&"staging/data".into(), &"data".into()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn copy_if_not_exists_or_same_size(
        &self,
        source_path: &Path,
        destination_path: &Path,
    ) -> LanceResult<WriteResult> {
        let source_size = self.size(source_path).await?;
        match self
            .inner
            .copy_if_not_exists(source_path, destination_path)
            .await
        {
            Ok(()) => {}
            Err(
                crate::object_store::Error::AlreadyExists { .. }
                | crate::object_store::Error::Precondition { .. },
            ) => {
                let destination_size = self.size(destination_path).await?;
                if source_size != destination_size {
                    return Err(LanceError::io(format!(
                        "destination {destination_path} already exists with size \
                         {destination_size}, but source {source_path} has size {source_size}"
                    )));
                }
            }
            Err(error) => return Err(LanceError::io(error.to_string())),
        }

        Ok(WriteResult {
            size: usize::try_from(source_size).map_err(|_| {
                LanceError::io(format!(
                    "source {source_path} size {source_size} does not fit in usize"
                ))
            })?,
            e_tag: None,
        })
    }

    async fn copy_via_stream_inner(
        &self,
        source_path: &Path,
        destination_store: &Self,
        destination_path: &Path,
    ) -> LanceResult<WriteResult> {
        let started_at = Instant::now();
        if self.has_direct_local_paths()
            && destination_store.has_direct_local_paths()
            && !destination_store.local_fsync
        {
            let source_size = std::fs::metadata(super::local::to_local_path(source_path))
                .map_err(|source| {
                    let source = if source.kind() == std::io::ErrorKind::NotFound {
                        LanceError::not_found(source_path.to_string())
                    } else {
                        LanceError::from(source)
                    };
                    stream_copy_error("source metadata", source_path, destination_path, source)
                })?
                .len();
            let source_size = usize::try_from(source_size).map_err(|source| {
                stream_copy_error(
                    "source size conversion",
                    source_path,
                    destination_path,
                    source,
                )
            })?;
            Span::current().record("source_size", source_size as u64);

            let metrics = destination_store.io_tracker.begin_io("copy");
            let result = super::local::copy_file(source_path, destination_path);
            metrics.record(&result, source_size as u64);
            result.map_err(|source| {
                stream_copy_error(
                    "local filesystem copy",
                    source_path,
                    destination_path,
                    source,
                )
            })?;

            let destination_size =
                destination_store
                    .size(destination_path)
                    .await
                    .map_err(|source| {
                        stream_copy_error(
                            "destination validation",
                            source_path,
                            destination_path,
                            source,
                        )
                    })?;
            Span::current().record("bytes_transferred", source_size as u64);
            Span::current().record("destination_size", destination_size);
            if destination_size != source_size as u64 {
                Span::current().record("validation", "failed");
                return Err(LanceError::io(format!(
                    "multipart_stream_copy destination size mismatch from {source_path} to \
                     {destination_path}: source_size={source_size}, \
                     destination_size={destination_size}"
                )));
            }

            Span::current().record("validation", "passed");
            Span::current().record("elapsed_ms", started_at.elapsed().as_millis() as u64);
            return Ok(WriteResult {
                size: source_size,
                e_tag: None,
            });
        }

        let reader = self.open(source_path).await.map_err(|source| {
            stream_copy_error("source open", source_path, destination_path, source)
        })?;
        let source_size = reader.size().await.map_err(|source| {
            stream_copy_error("source metadata", source_path, destination_path, source)
        })?;
        Span::current().record("source_size", source_size as u64);

        let mut writer = destination_store
            .create(destination_path)
            .await
            .map_err(|source| {
                stream_copy_error(
                    "destination writer creation",
                    source_path,
                    destination_path,
                    source,
                )
            })?;
        let read_chunk_size = usize::try_from(self.max_iop_size())
            .unwrap_or(usize::MAX)
            .max(1);
        Span::current().record("read_chunk_size", read_chunk_size as u64);
        let mut bytes_transferred = 0usize;
        if source_size > 0 {
            let first_range = 0..read_chunk_size.min(source_size);
            let mut current_range = first_range.clone();
            let mut current_bytes = reader.get_range(first_range).await.map_err(|source| {
                stream_copy_error("source read", source_path, destination_path, source)
            })?;

            loop {
                let expected_bytes = current_range.len();
                if current_bytes.len() != expected_bytes {
                    Span::current().record("validation", "failed");
                    return Err(LanceError::io(format!(
                        "multipart_stream_copy source range size mismatch from {source_path} to \
                         {destination_path}: range={current_range:?}, \
                         expected_bytes={expected_bytes}, actual_bytes={}",
                        current_bytes.len()
                    )));
                }
                bytes_transferred = bytes_transferred
                    .checked_add(current_bytes.len())
                    .ok_or_else(|| {
                        LanceError::io(format!(
                            "multipart_stream_copy byte count overflow from {source_path} to \
                             {destination_path}"
                        ))
                    })?;

                if bytes_transferred == source_size {
                    writer.write_all(&current_bytes).await.map_err(|source| {
                        stream_copy_error(
                            "destination write",
                            source_path,
                            destination_path,
                            source,
                        )
                    })?;
                    break;
                }

                let range_end = bytes_transferred
                    .checked_add(read_chunk_size)
                    .unwrap_or(source_size)
                    .min(source_size);
                let next_range = bytes_transferred..range_end;
                let next_read = reader.get_range(next_range.clone());
                let (write_result, next_bytes) =
                    tokio::join!(writer.write_all(&current_bytes), next_read);
                write_result.map_err(|source| {
                    stream_copy_error("destination write", source_path, destination_path, source)
                })?;
                current_bytes = next_bytes.map_err(|source| {
                    stream_copy_error("source read", source_path, destination_path, source)
                })?;
                current_range = next_range;
            }
        }
        Span::current().record("bytes_transferred", bytes_transferred as u64);

        let write_result = Writer::shutdown(writer.as_mut()).await.map_err(|source| {
            stream_copy_error(
                "destination completion",
                source_path,
                destination_path,
                source,
            )
        })?;
        if write_result.size != source_size {
            Span::current().record("validation", "failed");
            return Err(LanceError::io(format!(
                "multipart_stream_copy writer size mismatch from {source_path} to \
                 {destination_path}: source_size={source_size}, \
                 writer_size={}",
                write_result.size
            )));
        }

        let destination_size =
            destination_store
                .size(destination_path)
                .await
                .map_err(|source| {
                    stream_copy_error(
                        "destination validation",
                        source_path,
                        destination_path,
                        source,
                    )
                })?;
        Span::current().record("destination_size", destination_size);
        if destination_size != source_size as u64 {
            Span::current().record("validation", "failed");
            return Err(LanceError::io(format!(
                "multipart_stream_copy destination size mismatch from {source_path} to \
                 {destination_path}: source_size={source_size}, \
                 destination_size={destination_size}"
            )));
        }

        Span::current().record("validation", "passed");
        Span::current().record("elapsed_ms", started_at.elapsed().as_millis() as u64);
        Ok(write_result)
    }

    /// Read a directory (start from base directory) and returns all sub-paths in the directory.
    ///
    /// This enumerates the whole prefix before it returns, however many children it holds.
    /// Use [`Self::read_dir_page`] to page through a directory instead.
    pub async fn read_dir(&self, dir_path: impl Into<Path>) -> LanceResult<Vec<String>> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;
        let output = self.inner.list_with_delimiter(Some(&path)).await?;
        Ok(output
            .common_prefixes
            .iter()
            .chain(output.objects.iter().map(|o| &o.location))
            .filter_map(|s| s.filename().map(|f| f.to_string()))
            .collect())
    }

    /// Non-recursive, path-segment delimited list of a single directory level.
    ///
    /// Unlike [`Self::list`], which recurses into the entire subtree, this returns
    /// only the immediate children of `prefix`: the child "directories" as
    /// [`ListResult::common_prefixes`] and the direct child files as
    /// [`ListResult::objects`].
    pub async fn list_with_delimiter(&self, prefix: Option<&Path>) -> LanceResult<ListResult> {
        Ok(self.inner.list_with_delimiter(prefix).await?)
    }

    /// List objects through the backend without adding retries or resuming a
    /// failed stream. Provider-level retries still apply.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Result};
    /// # use futures::TryStreamExt;
    /// # async fn example(storage: &ObjectStore) -> Result<()> {
    /// let objects = storage.list_objects(Some(&"versions".into()))
    ///     .try_collect::<Vec<_>>().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn list_objects(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, crate::object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    /// List objects recursively. On unordered stores, a retryable failure after
    /// the first entry is returned to the caller because resuming by lexical
    /// offset could silently omit entries.
    pub fn list(
        &self,
        path: Option<Path>,
    ) -> Pin<Box<dyn Stream<Item = LanceResult<ObjectMeta>> + Send>> {
        Box::pin(
            ListRetryStream::new(self.inner.clone(), path, 5)
                .with_lexical_ordering(self.list_is_lexically_ordered)
                .map(|m| m.map_err(|e| e.into())),
        )
    }

    /// List objects lexically after `offset`, preserving the provider's native
    /// start-after optimization when available.
    ///
    /// This filters keys, not continuation tokens. It makes no ordering
    /// guarantee beyond [`Self::list_is_lexically_ordered`].
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # use futures::TryStreamExt;
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// let objects = store.list_with_offset(Some(&"versions".into()), &"versions/d".into())
    ///     .try_collect::<Vec<_>>().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'_, LanceResult<ObjectMeta>> {
        self.inner
            .list_with_offset(prefix, offset)
            .map_err(LanceError::from)
            .boxed()
    }

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    pub fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, LanceResult<ObjectMeta>> {
        let output = self
            .inner
            .list(Some(dir_path.into()))
            .map_err(LanceError::from);
        if let Some(cutoff) = unmodified_since {
            output
                .try_filter(move |file| future::ready(file.last_modified <= cutoff))
                .boxed()
        } else {
            output.boxed()
        }
    }

    /// Remove a directory recursively.
    ///
    /// Native local stores also remove materialized directories through the
    /// provider's path mapping, independently of backend wrappers.
    pub async fn remove_dir_all(&self, dir_path: impl Into<Path>) -> LanceResult<()> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;

        if let Some(local_dir_operations) = &self.local_dir_operations {
            let metrics = self.io_tracker.begin_io("delete");
            let result = local_dir_operations.remove_dir_all(&path).await;
            metrics.record(&result, 0);
            return result;
        }
        if self.direct_local_io && self.local_io != LocalIo::Backend {
            // The local file system provider needs to delete both files and directories.
            // Counted as a single delete request, matching how `delete_stream`
            // counts one batched request regardless of how many paths it removes.
            let metrics = self.io_tracker.begin_io("delete");
            let result = super::local::remove_dir_all(&path);
            metrics.record(&result, 0);
            return result;
        }
        // The deletion stream owns its input; share only the deferred zero-size paths.
        let deferred = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let deferred_entries = deferred.clone();
        let files = self.inner.list(Some(&path)).try_filter_map(move |entry| {
            let deferred_entries = deferred_entries.clone();
            async move {
                // OpenDAL directories have size zero. Empty files can be deferred too,
                // since every child path is longer than its parent directory.
                if entry.size == 0 {
                    deferred_entries.lock().await.push(entry.location);
                    Ok(None)
                } else {
                    Ok(Some(entry.location))
                }
            }
        });
        // Feed one continuous stream so each backend retains its batching and concurrency.
        self.inner
            .delete_stream(files.boxed())
            .try_for_each(|_| future::ready(Ok(())))
            .await?;
        let mut deferred = std::mem::take(&mut *deferred.lock().await);
        deferred.sort_unstable_by_key(|location| std::cmp::Reverse(location.as_ref().len()));
        for location in deferred {
            self.inner.delete(&location).await?;
        }
        if self.direct_local_io {
            // file-object-store tries to do everything as similarly as possible to the remote
            // object stores. But we still have to delete the directory entries afterwards.
            return super::local::remove_dir_all(&path);
        }
        Ok(())
    }

    /// Remove eligible materialized empty directories below a local root.
    ///
    /// This is a no-op for object stores, which do not materialize directories.
    /// Traversal does not follow symbolic links. Directories in `retained_dirs` and their
    /// descendants are preserved. Other directories are removed only if they are empty and
    /// either appear in `verified_dirs` or predate `unmodified_since`. Passing `None` for
    /// `unmodified_since` disables the age check.
    ///
    /// ```
    /// # use std::collections::HashSet;
    /// # use chrono::Utc;
    /// # use lance_core::Result;
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn remove_stale_index_dirs(store: &ObjectStore) -> lance_core::Result<()> {
    /// store
    ///     .remove_empty_dirs(
    ///         "dataset/_indices",
    ///         HashSet::new(),
    ///         HashSet::new(),
    ///         Some(Utc::now()),
    ///     )
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn remove_empty_dirs(
        &self,
        root_path: impl Into<Path>,
        retained_dirs: HashSet<Path>,
        verified_dirs: HashSet<Path>,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> LanceResult<()> {
        if !self.direct_local_io {
            return Ok(());
        }

        let path = Path::parse(root_path.into())?;
        let metrics = self.io_tracker.begin_io("delete");
        let result = tokio::task::spawn_blocking(move || {
            super::local::remove_empty_dirs(&path, &retained_dirs, &verified_dirs, unmodified_since)
        })
        .await
        .map_err(|error| LanceError::io(format!("empty-directory cleanup task failed: {error}")))?;
        metrics.record(&result, 0);
        result
    }

    /// Delete every location in the stream, up to `io_parallelism` at a time.
    ///
    /// Deleting serially costs a round trip per object — ~30/s against a cloud store,
    /// which is days for a table with millions of unreferenced files.
    ///
    /// `ObjectStore::delete_stream` would also batch (S3: 1000 keys per request) but
    /// needs a `'static` stream of `crate::object_store::Result`; callers pass a borrowed
    /// stream of `lance_core::Result`. Using it means changing this signature.
    ///
    /// Order is not preserved; no caller may depend on it.
    ///
    /// A location that is already gone counts as removed. Callers list first and
    /// delete after, so a concurrent writer or a second cleanup can remove a path in
    /// between; failing there would abandon an entire sweep over one absent object.
    pub fn remove_stream<'a>(
        &'a self,
        locations: BoxStream<'a, LanceResult<Path>>,
    ) -> BoxStream<'a, LanceResult<Path>> {
        let store = Arc::clone(&self.inner);
        locations
            .map(move |location| {
                let store = Arc::clone(&store);
                async move {
                    let location = location?;
                    match store.delete(&location).await {
                        Ok(()) => Ok(location),
                        Err(crate::object_store::Error::NotFound { .. }) => Ok(location),
                        Err(error) => Err(error.into()),
                    }
                }
            })
            .buffer_unordered(self.io_parallelism())
            .boxed()
    }

    /// Check a file exists.
    pub async fn exists(&self, path: &Path) -> LanceResult<bool> {
        match self.inner.head(path).await {
            Ok(_) => Ok(true),
            Err(crate::object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Get file size.
    pub async fn size(&self, path: &Path) -> LanceResult<u64> {
        Ok(self.inner.head(path).await?.size)
    }

    /// Convenience function to open a reader and read all the bytes
    pub async fn read_one_all(&self, path: &Path) -> LanceResult<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_all().await?)
    }

    /// Read a byte range directly from the configured store.
    ///
    /// This preserves the backend's range response, including a shorter body
    /// when an object has shrunk since its size was observed. Manifest readers
    /// use that response to detect stale size hints and retry their metadata
    /// lookup. The range uses 64-bit object offsets without opening a Reader.
    ///
    /// ```
    /// # use lance_io::object_store::{ObjectStore, Path};
    /// # async fn example(store: &ObjectStore) -> lance_core::Result<()> {
    /// let header = store.read_range(&"manifest".into(), 0..16).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_range(&self, path: &Path, range: Range<u64>) -> LanceResult<Bytes> {
        Ok(self.inner.get_range(path, range).await?)
    }

    /// Convenience function open a reader and make a single request
    ///
    /// If you will be making multiple requests to the path it is more efficient to call [`Self::open`]
    /// and then call [`Reader::get_range`] multiple times.
    pub async fn read_one_range(&self, path: &Path, range: Range<usize>) -> LanceResult<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_range(range).await?)
    }
}

#[derive(Clone, Debug, Default)]
pub struct StorageOptions(pub HashMap<String, String>);

impl StorageOptions {
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn as_cloud_options<K>(&self) -> HashMap<K, String>
    where
        K: FromStr + AsRef<str> + Eq + std::hash::Hash,
    {
        let mut options: HashMap<K, (&str, &str)> = HashMap::new();
        for (source_key, value) in &self.0 {
            let Ok(config_key) = K::from_str(&source_key.to_ascii_lowercase()) else {
                continue;
            };
            let source_key = source_key.as_str();
            let canonical_key = config_key.as_ref();
            // Rank spellings so conflicting aliases cannot depend on HashMap iteration order.
            if let Some((selected_key, _)) = options.get(&config_key)
                && (
                    !source_key.eq_ignore_ascii_case(canonical_key),
                    source_key != canonical_key,
                    source_key,
                ) >= (
                    !selected_key.eq_ignore_ascii_case(canonical_key),
                    *selected_key != canonical_key,
                    *selected_key,
                )
            {
                continue;
            }
            options.insert(config_key, (source_key, value.as_str()));
        }
        options
            .into_iter()
            .map(|(key, (_, value))| (key, value.to_string()))
            .collect()
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn merge_env_options(
        &mut self,
        env_options: impl IntoIterator<
            Item = (impl AsRef<std::ffi::OsStr>, impl AsRef<std::ffi::OsStr>),
        >,
        canonical_key: impl Fn(&str) -> Option<String>,
    ) {
        // Aliases must block environment defaults without rewriting the caller's keys.
        let mut configured_keys: HashSet<String> = self
            .0
            .keys()
            .filter_map(|key| canonical_key(&key.to_ascii_lowercase()))
            .collect();
        for (key, value) in env_options {
            if let (Some(key), Some(value)) = (key.as_ref().to_str(), value.as_ref().to_str())
                && let Some(config_key) = canonical_key(&key.to_ascii_lowercase())
                && configured_keys.insert(config_key.clone())
            {
                self.0.insert(config_key, value.to_string());
            }
        }
    }

    /// Create a new instance of [`StorageOptions`]
    pub fn new(options: HashMap<String, String>) -> Self {
        let mut options = options;
        if let Ok(value) = std::env::var("AZURE_STORAGE_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AZURE_STORAGE_USE_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("AWS_ALLOW_HTTP") {
            options.insert("allow_http".into(), value);
        }
        if let Ok(value) = std::env::var("OBJECT_STORE_CLIENT_MAX_RETRIES") {
            options.insert("client_max_retries".into(), value);
        }
        if let Ok(value) = std::env::var("OBJECT_STORE_CLIENT_RETRY_TIMEOUT") {
            options.insert("client_retry_timeout".into(), value);
        }
        Self(options)
    }

    /// Denotes if unsecure connections via http are allowed
    pub fn allow_http(&self) -> bool {
        self.0.iter().any(|(key, value)| {
            key.to_ascii_lowercase().contains("allow_http") & str_is_truthy(value)
        })
    }

    /// Byte gap below which the I/O scheduler merges two reads of one file
    /// into a single request, overriding the store's default.
    pub fn block_size(&self) -> LanceResult<Option<usize>> {
        let Some((_, value)) = self
            .0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("block_size"))
        else {
            return Ok(None);
        };
        let block_size = value.trim().parse::<usize>().map_err(|err| {
            LanceError::invalid_input(format!(
                "storage option block_size must be a number of bytes, got `{value}`: {err}"
            ))
        })?;
        Ok(Some(block_size))
    }

    /// Minimum cache size at which the session's object store registry sweeps
    /// out entries whose store is no longer in use; `0` disables the sweep.
    /// See [`ObjectStoreRegistry::get_store()`].
    pub fn storage_cache_prune_threshold(&self) -> LanceResult<Option<usize>> {
        let Some((_, value)) = self
            .0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(STORAGE_CACHE_PRUNE_THRESHOLD_OPTION))
        else {
            return Ok(None);
        };
        let threshold = value.trim().parse::<usize>().map_err(|err| {
            LanceError::invalid_input(format!(
                "storage option {STORAGE_CACHE_PRUNE_THRESHOLD_OPTION} must be a \
                 non-negative integer, got `{value}`: {err}"
            ))
        })?;
        Ok(Some(threshold))
    }

    /// Whether local writes must sync files and directory entries before returning.
    pub(crate) fn local_fsync(&self) -> LanceResult<bool> {
        let Some((_, value)) = self
            .0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("local_fsync"))
        else {
            return Ok(false);
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(LanceError::invalid_input(format!(
                "storage option local_fsync must be `true` or `false`, got `{value}`"
            ))),
        }
    }

    /// Number of times to retry a download that fails
    pub fn download_retry_count(&self) -> usize {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("download_retry_count"))
            .map(|(_, value)| value.parse::<usize>().unwrap_or(3))
            .unwrap_or(3)
    }

    /// Max retry times to set in RetryConfig for object store client
    pub fn client_max_retries(&self) -> usize {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("client_max_retries"))
            .and_then(|(_, value)| value.parse::<usize>().ok())
            .unwrap_or(3)
    }

    /// Seconds of timeout to set in RetryConfig for object store client
    pub fn client_retry_timeout(&self) -> u64 {
        self.0
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("client_retry_timeout"))
            .and_then(|(_, value)| value.parse::<u64>().ok())
            .unwrap_or(180)
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.0.get(key)
    }

    /// Build [`ClientOptions`] with default headers extracted from `headers.*` keys.
    ///
    /// Keys prefixed with `headers.` are parsed into HTTP headers. For example,
    /// `headers.x-ms-version = 2023-11-03` results in a default header
    /// `x-ms-version: 2023-11-03`.
    ///
    /// Returns an error if any `headers.*` key has an invalid header name or value.
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn client_options(&self) -> LanceResult<ClientOptions> {
        let mut headers = HeaderMap::new();
        for (key, value) in &self.0 {
            if let Some(header_name) = key.strip_prefix("headers.") {
                let name = header_name
                    .parse::<http::header::HeaderName>()
                    .map_err(|e| {
                        LanceError::invalid_input(format!(
                            "invalid header name '{header_name}': {e}"
                        ))
                    })?;
                let val = HeaderValue::from_str(value).map_err(|e| {
                    LanceError::invalid_input(format!(
                        "invalid header value for '{header_name}': {e}"
                    ))
                })?;
                headers.insert(name, val);
            }
        }
        let mut client_options = ClientOptions::default();
        if !headers.is_empty() {
            client_options = client_options.with_default_headers(headers);
        }
        Ok(client_options)
    }

    /// Get the expiration time in milliseconds since epoch, if present
    pub fn expires_at_millis(&self) -> Option<u64> {
        self.0
            .get(EXPIRES_AT_MILLIS_KEY)
            .and_then(|s| s.parse::<u64>().ok())
    }
}

impl From<HashMap<String, String>> for StorageOptions {
    fn from(value: HashMap<String, String>) -> Self {
        Self::new(value)
    }
}

/// Wall time of a scan's decode tasks in nanoseconds, summed over tasks and
/// labelled like the store's object store metrics (`base`, and `dataset` in
/// dataset mode). Tasks run concurrently, so the counter's rate is the number
/// of batches decoding at once. Mirrors the scan node's `decode_time` plan
/// metric, as a cumulative counter a scrape can read while the scan runs.
pub const METRIC_SCAN_DECODE_TIME: &str = "lance_scan_decode_time_nanoseconds_total";
/// Time scan partitions spent waiting for their next batch's I/O to be
/// scheduled, in nanoseconds, labelled like [`METRIC_SCAN_DECODE_TIME`].
/// Mirrors the scan node's `task_wait_time` plan metric.
pub const METRIC_SCAN_TASK_WAIT_TIME: &str = "lance_scan_task_wait_time_nanoseconds_total";

/// A cumulative counter on the `metrics` facade carrying a store's labels; see
/// [`ObjectStore::counter`]. A no-op without the `metrics` feature.
pub struct StoreCounter(#[cfg(feature = "metrics")] ::metrics::Counter);

impl StoreCounter {
    /// Add `value` to the counter. This is a no-op without the `metrics`
    /// feature.
    pub fn increment(&self, value: u64) {
        #[cfg(feature = "metrics")]
        self.0.increment(value);
        #[cfg(not(feature = "metrics"))]
        let _ = value;
    }
}

/// Wrap `inner` so its operations publish metrics labelled by `store_prefix`,
/// and label `io_tracker` with the same prefix so the local reads and writes
/// that bypass `inner` publish under it too.
///
/// The two go together on purpose: a store metered on one path but not the other
/// would report a partial picture that reads like a complete one. Every
/// constructor that hands an [`ObjectStore`] to a caller must route its `inner`
/// through here, or through nothing at all.
#[cfg(feature = "metrics")]
fn meter_store(inner: &mut Arc<dyn StorageBackend>, io_tracker: &mut IOTracker, base: &str) {
    use crate::object_store::metrics::StorageBackendMetricsExt;
    io_tracker.set_metrics_base(base);
    *inner = inner.clone().metered(base.to_owned());
}

#[cfg(not(feature = "metrics"))]
fn meter_store(_inner: &mut Arc<dyn StorageBackend>, _io_tracker: &mut IOTracker, _base: &str) {}

/// The `base` metrics label for a store opened at `location`; see
/// [`metrics::metrics_base`]. Without the `metrics` feature the label is unused
/// and the prefix keeps the registry cache keyed per bucket as before.
#[cfg(feature = "metrics")]
pub(crate) fn metrics_base(store_prefix: &str, location: &Url) -> String {
    metrics::metrics_base(metrics::base_label_mode(), store_prefix, location)
}

#[cfg(not(feature = "metrics"))]
pub(crate) fn metrics_base(store_prefix: &str, _location: &Url) -> String {
    store_prefix.to_owned()
}

#[cfg(test)]
mod tests {
    use crate::object_store::Result as OSResult;
    use futures::FutureExt;

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[test]
    fn test_params_with_different_http_client_wrappers_are_not_shared() {
        use crate::object_store::test_utils::RecordingHttpClientWrapper;
        use std::hash::{BuildHasher, RandomState};

        let one: Arc<dyn transport::HttpClientWrapper> =
            Arc::new(RecordingHttpClientWrapper::default());
        let other: Arc<dyn transport::HttpClientWrapper> =
            Arc::new(RecordingHttpClientWrapper::default());
        let with = |wrapper: &Arc<dyn transport::HttpClientWrapper>| ObjectStoreParams {
            http_client_wrapper: Some(wrapper.clone()),
            ..Default::default()
        };
        let hasher = RandomState::new();
        assert_eq!(with(&one), with(&one));
        assert_eq!(hasher.hash_one(with(&one)), hasher.hash_one(with(&one)));
        assert_ne!(with(&one), with(&other));
        assert_ne!(with(&one), ObjectStoreParams::default());
    }
    use super::*;
    use crate::object_store::backend::MultipartUpload;
    use async_trait::async_trait;
    use bytes::Bytes;
    use lance_core::utils::tempfile::{TempStdDir, TempStdFile, TempStrDir};
    use object_store::memory::InMemory;
    use rstest::rstest;
    use serial_test::serial;
    use std::env::set_current_dir;
    use std::fmt::{Display, Formatter};
    use std::fs::{create_dir_all, write};
    use std::ops::Range;
    use std::path::Path as StdPath;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_canonicalize_local_io_result() {
        let path = PathBuf::from("/tmp/lance-canonical");
        assert_eq!(
            canonicalize_local_io_result(Ok(path.clone())).unwrap(),
            Some(path)
        );
        assert_eq!(
            canonicalize_local_io_result(Err(std::io::Error::from(std::io::ErrorKind::NotFound,)))
                .unwrap(),
            None
        );

        let error = canonicalize_local_io_result(Err(std::io::Error::from(
            std::io::ErrorKind::PermissionDenied,
        )))
        .unwrap_err();
        assert!(matches!(error, LanceError::IO { .. }), "{error:?}");
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn canonical_test_config_key<K: FromStr + AsRef<str>>(key: &str) -> Option<String> {
        K::from_str(key).ok().map(|key| key.as_ref().to_string())
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[rstest]
    #[cfg_attr(feature = "aws", case::aws("region", "aws_region", canonical_test_config_key::<object_store::aws::AmazonS3ConfigKey>))]
    #[cfg_attr(feature = "azure", case::azure("account_name", "azure_storage_account_name", canonical_test_config_key::<object_store::azure::AzureConfigKey>))]
    #[cfg_attr(feature = "gcp", case::gcp("service_account_path", "google_service_account", canonical_test_config_key::<object_store::gcp::GoogleConfigKey>))]
    fn test_merge_env_options(
        #[case] alias: &str,
        #[case] canonical: &str,
        #[case] canonical_key: fn(&str) -> Option<String>,
        #[values(false, true)] use_alias: bool,
        #[values(false, true)] uppercase: bool,
        #[values(None, Some(""), Some("explicit"))] explicit_value: Option<&str>,
    ) {
        let key = if use_alias { alias } else { canonical };
        let key = if uppercase {
            key.to_ascii_uppercase()
        } else {
            key.to_string()
        };
        let mut expected = HashMap::from([("lance_custom_option".into(), "unchanged".into())]);
        if let Some(value) = explicit_value {
            expected.insert(key, value.to_string());
        }
        let mut options = StorageOptions(expected.clone());
        if explicit_value.is_none() {
            expected.insert(canonical.to_string(), "environment".to_string());
        }
        let env_options = [
            (canonical.to_ascii_uppercase(), "environment"),
            (alias.to_string(), "second_environment_alias"),
            ("lance_custom_option".to_string(), "ignored"),
            ("unrecognized_env_option".to_string(), "ignored"),
        ];

        options.merge_env_options(
            env_options.iter().map(|(key, value)| (key, value)),
            canonical_key,
        );
        assert_eq!(options.0, expected);
        options.merge_env_options(
            env_options.iter().map(|(key, value)| (key, value)),
            canonical_key,
        );
        assert_eq!(options.0, expected);
    }

    /// Write test content to file.
    fn write_to_file(path_str: &str, contents: &str) -> std::io::Result<()> {
        let path = expand_path(path_str).map_err(std::io::Error::other)?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        write(path, contents)
    }

    async fn read_from_store(store: &ObjectStore, path: &Path) -> LanceResult<String> {
        let test_file_store = store.open(path).await.unwrap();
        let size = test_file_store.size().await.unwrap();
        let bytes = test_file_store.get_range(0..size).await.unwrap();
        let contents = String::from_utf8(bytes.to_vec()).unwrap();
        Ok(contents)
    }

    #[rstest]
    #[case::memory("memory")]
    #[case::local("file")]
    #[case::local_storage("file-object-store")]
    #[tokio::test]
    async fn test_put_if_absent(#[case] scheme: &str) {
        let directory = TempStdDir::default();
        let (store, base) = if scheme == "memory" {
            (Arc::new(ObjectStore::memory()), Path::from("dataset"))
        } else {
            let file_url = Url::from_directory_path(&directory).unwrap();
            let mut url = Url::parse(&format!("{scheme}:///")).unwrap();
            url.set_path(file_url.path());
            ObjectStore::from_uri(url.as_ref()).await.unwrap()
        };
        let path = base.clone().join("目录/.lance-reserved");
        let result = store
            .put_if_absent(&path, Bytes::from_static(b"first").into())
            .await
            .unwrap();
        assert_eq!(result.size, 5);
        if scheme == "memory" {
            assert!(result.e_tag.is_some());
            assert_eq!(result.e_tag, store.stat(&path).await.unwrap().e_tag);
        }
        let error = store
            .put_if_absent(&path, Bytes::from_static(b"second").into())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            crate::object_store::Error::AlreadyExists { .. }
                | crate::object_store::Error::Precondition { .. }
        ));
        assert_eq!(
            store.read_one_all(&path).await.unwrap(),
            b"first".as_slice()
        );
        let remaining = store
            .list(Some(base))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(remaining.len(), 1, "staging objects must be cleaned up");
        assert_eq!(remaining[0].location, path);
    }

    #[rstest]
    #[case::default(None, SchedulerErrorMode::BestEffort)]
    #[case::best_effort(Some("best_effort"), SchedulerErrorMode::BestEffort)]
    #[case::fail_fast(Some("fail_fast"), SchedulerErrorMode::FailFast)]
    #[case::case_insensitive(Some("FAIL_FAST"), SchedulerErrorMode::FailFast)]
    fn test_scheduler_error_mode_storage_option(
        #[case] value: Option<&str>,
        #[case] expected: SchedulerErrorMode,
    ) {
        let storage_options_accessor = value.map(|value| {
            Arc::new(StorageOptionsAccessor::with_static_options(HashMap::from(
                [(
                    SCHEDULER_ERROR_MODE_STORAGE_OPTION.to_string(),
                    value.to_string(),
                )],
            )))
        });
        let params = ObjectStoreParams {
            storage_options_accessor,
            ..Default::default()
        };

        assert_eq!(params.scheduler_error_mode().unwrap(), expected);
    }

    #[tokio::test]
    async fn test_invalid_scheduler_error_mode_storage_option() {
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(
                    SCHEDULER_ERROR_MODE_STORAGE_OPTION.to_string(),
                    "invalid".to_string(),
                )]),
            ))),
            ..Default::default()
        };

        let error = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &params,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, LanceError::InvalidInput { .. }));
        assert!(error.to_string().contains(
            "Invalid scheduler_error_mode value 'invalid'. Valid values are: best_effort, fail_fast"
        ));
    }

    #[rstest]
    #[case::zero_clamps_to_one("0", 1)]
    #[case::positive_passes_through("3", 3)]
    #[tokio::test]
    async fn test_io_parallelism_clamped_to_nonzero(
        #[case] io_threads: &str,
        #[case] expected: usize,
    ) {
        // Other tests read the process-wide `LANCE_IO_THREADS`, so instead of setting it here the
        // test re-runs itself in a child process that starts with it. libtest names the thread
        // after the running test, so the child's filter cannot drift from this test's name.
        const CHILD_ENV: &str = "LANCE_TEST_IO_PARALLELISM_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    std::thread::current().name().unwrap(),
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .env("LANCE_IO_THREADS", io_threads)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "child test failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        // `io_parallelism()` feeds `buffered`/`buffer_unordered` windows; a value of 0 makes those
        // streams never poll, hanging callers (e.g. a metadata-only `count_rows`). It must clamp.
        let store = ObjectStore::local();
        assert_eq!(
            store.io_parallelism(),
            expected,
            "LANCE_IO_THREADS={io_threads} must give {expected}"
        );

        // Readers opened by the store must advertise the store's normalized
        // effective parallelism, not the hardcoded cloud default.
        let mem_store = ObjectStore::memory();
        let path = Path::from("/io_parallelism_probe");
        mem_store.put(&path, b"x").await.unwrap();
        assert_eq!(
            mem_store.open(&path).await.unwrap().io_parallelism(),
            expected,
            "an opened reader must honor the configured request limit"
        );
        assert_eq!(
            mem_store
                .open_with_size(&path, 1024 * 1024)
                .await
                .unwrap()
                .io_parallelism(),
            expected,
            "a sized reader must honor the configured request limit"
        );
    }

    #[tokio::test]
    async fn test_absolute_paths() {
        let tmp_path = TempStrDir::default();
        write_to_file(
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "TEST_CONTENT",
        )
        .unwrap();

        // test a few variations of the same path
        for uri in &[
            format!("{tmp_path}/bar/foo.lance"),
            format!("{tmp_path}/./bar/foo.lance"),
            format!("{tmp_path}/bar/foo.lance/../foo.lance"),
        ] {
            let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store.as_ref(), &path.clone().join("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "TEST_CONTENT");
        }
    }

    #[tokio::test]
    async fn test_cloud_paths() {
        let uri = "s3://bucket/foo.lance";
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        assert_eq!(store.scheme, "s3");
        assert_eq!(path.to_string(), "foo.lance");

        let (store, path) = ObjectStore::from_uri("s3+ddb://bucket/foo.lance")
            .await
            .unwrap();
        assert_eq!(store.scheme, "s3");
        assert_eq!(path.to_string(), "foo.lance");

        let (store, path) = ObjectStore::from_uri("gs://bucket/foo.lance")
            .await
            .unwrap();
        assert_eq!(store.scheme, "gs");
        assert_eq!(path.to_string(), "foo.lance");

        let (store, path) =
            ObjectStore::from_uri("abfss://filesystem@account.dfs.core.windows.net/foo.lance")
                .await
                .unwrap();
        assert_eq!(store.scheme, "abfss");
        assert_eq!(path.to_string(), "foo.lance");
    }

    async fn test_block_size_used_test_helper(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
        default_expected_block_size: usize,
    ) {
        // Test the default
        let registry = Arc::new(ObjectStoreRegistry::default());
        let accessor = storage_options
            .clone()
            .map(|opts| Arc::new(StorageOptionsAccessor::with_static_options(opts)));
        let params = ObjectStoreParams {
            storage_options_accessor: accessor.clone(),
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, default_expected_block_size);

        // Ensure param is used
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            block_size: Some(1024),
            storage_options_accessor: accessor,
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, 1024);

        // The storage option applies when no parameter is given...
        let mut options_with_block_size = storage_options.unwrap_or_default();
        options_with_block_size.insert(String::from("block_size"), String::from("2048"));
        let accessor = Arc::new(StorageOptionsAccessor::with_static_options(
            options_with_block_size,
        ));
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            storage_options_accessor: Some(accessor.clone()),
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, 2048);

        // ...and the explicit parameter wins over it.
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            block_size: Some(1024),
            storage_options_accessor: Some(accessor),
            ..ObjectStoreParams::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(registry, uri, &params)
            .await
            .unwrap();
        assert_eq!(store.block_size, 1024);
    }

    #[tokio::test]
    async fn test_block_size_option_rejects_invalid_values() {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let accessor = Arc::new(StorageOptionsAccessor::with_static_options(HashMap::from(
            [(String::from("block_size"), String::from("64KiB"))],
        )));
        let params = ObjectStoreParams {
            storage_options_accessor: Some(accessor),
            ..ObjectStoreParams::default()
        };
        let error =
            ObjectStore::from_uri_and_params(registry, "memory:///bucket/foo.lance", &params)
                .await
                .unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("block_size"), "{error}");
    }

    #[rstest]
    #[case("s3://bucket/foo.lance", None)]
    #[case("gs://bucket/foo.lance", None)]
    #[case("az://account/bucket/foo.lance",
      Some(HashMap::from([
            (String::from("account_name"), String::from("account")),
            (String::from("container_name"), String::from("container"))
           ])))]
    #[case("abfss://filesystem@account.dfs.core.windows.net/foo.lance",
      Some(HashMap::from([
            (String::from("account_name"), String::from("account")),
            (String::from("container_name"), String::from("filesystem"))
           ])))]
    #[tokio::test]
    async fn test_block_size_used_cloud(
        #[case] uri: &str,
        #[case] storage_options: Option<HashMap<String, String>>,
    ) {
        test_block_size_used_test_helper(uri, storage_options, 64 * 1024).await;
    }

    #[rstest]
    #[case("file")]
    #[case("file-object-store")]
    #[case("memory:///bucket/foo.lance")]
    #[tokio::test]
    async fn test_block_size_used_file(#[case] prefix: &str) {
        let tmp_path = TempStrDir::default();
        let path = format!("{tmp_path}/bar/foo.lance/test_file");
        write_to_file(&path, "URL").unwrap();
        let uri = format!("{prefix}:///{path}");
        test_block_size_used_test_helper(&uri, None, 4 * 1024).await;
    }

    #[tokio::test]
    async fn test_relative_paths() {
        let tmp_path = TempStrDir::default();
        write_to_file(
            &format!("{tmp_path}/bar/foo.lance/test_file"),
            "RELATIVE_URL",
        )
        .unwrap();

        // The working directory is process state, and this one is about to be
        // removed with the temp dir, so put it back as soon as the call that
        // needs it returns.
        let original_dir = std::env::current_dir().unwrap();
        set_current_dir(StdPath::new(tmp_path.as_ref())).expect("LanceError changing current dir");
        let opened = ObjectStore::from_uri("./bar/foo.lance").await;
        set_current_dir(&original_dir).expect("LanceError restoring current dir");
        let (store, path) = opened.unwrap();

        let contents = read_from_store(store.as_ref(), &path.clone().join("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        // Resolving the path is sufficient to verify expansion and avoids writing
        // into the real user's home directory.
        let expected_path = std::env::home_dir()
            .expect("home directory should be available")
            .join("foo.lance");
        let (store, path) = ObjectStore::from_uri("~/foo.lance").await.unwrap();

        assert!(store.is_local());
        assert_eq!(
            path,
            Path::from_absolute_path(expected_path).expect("home path should be absolute")
        );
    }

    #[tokio::test]
    async fn test_read_directory() {
        let path = TempStdDir::default();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo").join("test_file").to_str().unwrap(),
            "read_dir",
        )
        .unwrap();
        let (store, base) = ObjectStore::from_uri(path.to_str().unwrap()).await.unwrap();

        let sub_dirs = store.read_dir(base.clone().join("foo")).await.unwrap();
        assert_eq!(sub_dirs, vec!["bar", "zoo", "test_file"]);
    }

    #[derive(Debug, Default)]
    struct BatchedDeleteStore {
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        deleted: Arc<AtomicUsize>,
        deferred: Arc<tokio::sync::Mutex<Vec<Path>>>,
    }

    impl Display for BatchedDeleteStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "BatchedDeleteStore")
        }
    }

    #[async_trait]
    impl crate::object_store::StorageBackend for BatchedDeleteStore {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            options: crate::object_store::PaginatedListOptions,
        ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
            let _ = (prefix, options);
            Err(crate::object_store::Error::NotSupported {
                source: "native pagination is unavailable".into(),
            })
        }

        async fn put_opts(&self, _: &Path, _: PutPayload, _: PutOptions) -> OSResult<PutResult> {
            unreachable!()
        }

        async fn put_multipart_opts(
            &self,
            _: &Path,
            _: PutMultipartOptions,
        ) -> OSResult<Box<dyn crate::object_store::backend::MultipartUpload>> {
            unreachable!()
        }

        async fn get_opts(&self, _: &Path, _: GetOptions) -> OSResult<GetResult> {
            unreachable!()
        }

        fn list(&self, _: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            let deferred = ["dataset", "dataset/data", "dataset/data/empty"]
                .into_iter()
                .map(|path| (Path::from(path), 0));
            let files = (0..20_000)
                .map(|index| (Path::from(format!("dataset/data/{index:05}.lance")), 128));
            futures::stream::iter(deferred.chain(files).map(|(location, size)| {
                Ok(ObjectMeta {
                    location,
                    size,
                    last_modified: Utc::now(),
                    e_tag: None,
                    version: None,
                })
            }))
            .boxed()
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            let active = self.active.clone();
            let peak = self.peak.clone();
            let deleted = self.deleted.clone();
            let deferred = self.deferred.clone();
            // Match S3's batching and concurrency, without depending on cloud timing.
            locations
                .try_chunks(1000)
                .map(move |chunk| {
                    let active = active.clone();
                    let peak = peak.clone();
                    let deleted = deleted.clone();
                    let deferred = deferred.clone();
                    async move {
                        let chunk = chunk.map_err(|error| error.1)?;
                        if chunk
                            .first()
                            .is_some_and(|path| path.extension() == Some("lance"))
                        {
                            assert!(chunk.iter().all(|path| path.extension() == Some("lance")));
                            let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(count, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(5)).await;
                            deleted.fetch_add(chunk.len(), Ordering::SeqCst);
                            active.fetch_sub(1, Ordering::SeqCst);
                        } else {
                            assert_eq!(active.load(Ordering::SeqCst), 0);
                            assert_eq!(deleted.load(Ordering::SeqCst), 20_000);
                            deferred.lock().await.extend(chunk.iter().cloned());
                        }
                        Ok::<_, crate::object_store::Error>(futures::stream::iter(
                            chunk.into_iter().map(Ok),
                        ))
                    }
                })
                .buffered(20)
                .try_flatten()
                .boxed()
        }

        async fn list_with_delimiter(&self, _: Option<&Path>) -> OSResult<ListResult> {
            unreachable!()
        }

        async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> OSResult<()> {
            unreachable!()
        }
        async fn get_ranges(
            &self,
            location: &crate::object_store::Path,
            ranges: &[std::ops::Range<u64>],
        ) -> crate::object_store::Result<Vec<bytes::Bytes>> {
            futures::future::try_join_all(ranges.iter().map(|range| {
                (self as &dyn crate::object_store::StorageBackend)
                    .get_range(location, range.clone())
            }))
            .await
        }
        fn list_with_offset(
            &self,
            prefix: Option<&crate::object_store::Path>,
            offset: &crate::object_store::Path,
        ) -> futures::stream::BoxStream<
            'static,
            crate::object_store::Result<crate::object_store::ObjectMeta>,
        > {
            let offset = offset.clone();
            Box::pin(futures::TryStreamExt::try_filter(
                self.list(prefix),
                move |meta| futures::future::ready(meta.location > offset),
            ))
        }
        async fn rename_opts(
            &self,
            from: &crate::object_store::Path,
            to: &crate::object_store::Path,
            options: crate::object_store::RenameOptions,
        ) -> crate::object_store::Result<()> {
            let mode = match options.target_mode {
                crate::object_store::RenameTargetMode::Create => {
                    crate::object_store::CopyMode::Create
                }
                crate::object_store::RenameTargetMode::Overwrite => {
                    crate::object_store::CopyMode::Overwrite
                }
            };
            self.copy_opts(
                from,
                to,
                crate::object_store::CopyOptions {
                    mode,
                    extensions: options.extensions,
                },
            )
            .await?;
            (self as &dyn crate::object_store::StorageBackend)
                .delete(from)
                .await
        }
    }

    #[tokio::test]
    async fn test_remove_dir_all_preserves_batch_concurrency() {
        let backend = Arc::new(BatchedDeleteStore::default());
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(|_| {
            backend.clone()
        }));
        store.remove_dir_all("dataset").await.unwrap();
        assert_eq!(backend.deleted.load(Ordering::SeqCst), 20_000);
        assert_eq!(backend.peak.load(Ordering::SeqCst), 20);
        assert_eq!(
            *backend.deferred.lock().await,
            [
                Path::from("dataset/data/empty"),
                Path::from("dataset/data"),
                Path::from("dataset")
            ]
        );
    }

    #[tokio::test]
    async fn test_delete_directory_local_store() {
        test_delete_directory("").await;
    }

    #[tokio::test]
    async fn test_delete_directory_file_storage() {
        test_delete_directory("file-object-store").await;
    }

    async fn test_delete_directory(scheme: &str) {
        let path = TempStdDir::default();
        create_dir_all(path.join("foo").join("bar")).unwrap();
        create_dir_all(path.join("foo").join("zoo")).unwrap();
        create_dir_all(path.join("foo").join("zoo").join("abc")).unwrap();
        write_to_file(
            path.join("foo")
                .join("bar")
                .join("test_file")
                .to_str()
                .unwrap(),
            "delete",
        )
        .unwrap();
        let file_url = Url::from_directory_path(&path).unwrap();
        let url = if scheme.is_empty() {
            file_url
        } else {
            let mut url = Url::parse(&format!("{scheme}:///")).unwrap();
            // Use the file:// URL's normalized path so this works on Windows too.
            url.set_path(file_url.path());
            url
        };
        let (store, base) = ObjectStore::from_uri(url.as_ref()).await.unwrap();
        store
            .remove_dir_all(base.clone().join("foo"))
            .await
            .unwrap();

        assert!(!path.join("foo").exists());
    }

    #[rstest]
    #[case("file")]
    #[case("file-object-store")]
    #[tokio::test]
    async fn test_remove_empty_directories(#[case] scheme: &str) {
        let path = TempStdDir::default();
        let stale_dir = path.join("stale");
        let nested_stale_dir = path.join("nested_stale");
        let nested_stale_child = nested_stale_dir.join("child");
        create_dir_all(&stale_dir).unwrap();
        create_dir_all(&nested_stale_child).unwrap();
        create_dir_all(path.join("retained").join("child")).unwrap();
        write_to_file(
            path.join("file_bearing")
                .join("test_file")
                .to_str()
                .unwrap(),
            "keep",
        )
        .unwrap();
        create_dir_all(path.join("file_bearing").join("empty_child")).unwrap();

        let file_url = Url::from_directory_path(&path).unwrap();
        let mut url = Url::parse(&format!("{scheme}:///")).unwrap();
        url.set_path(file_url.path());
        let (store, base) = ObjectStore::from_uri(url.as_ref()).await.unwrap();

        #[cfg(unix)]
        let unmodified_since = {
            let old_modified_time =
                std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 24 * 60 * 60);
            for directory in [&stale_dir, &nested_stale_dir, &nested_stale_child] {
                std::fs::File::open(directory)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(old_modified_time))
                    .unwrap();
            }
            DateTime::<Utc>::from(std::time::SystemTime::now())
                - chrono::TimeDelta::try_days(7).unwrap()
        };
        #[cfg(not(unix))]
        let unmodified_since = DateTime::<Utc>::from(std::time::SystemTime::now())
            + chrono::TimeDelta::try_days(1).unwrap();

        store
            .remove_empty_dirs(
                base.clone(),
                HashSet::from([base.clone().join("retained")]),
                HashSet::new(),
                Some(unmodified_since),
            )
            .await
            .unwrap();

        assert!(!path.join("stale").exists());
        assert!(!path.join("nested_stale").exists());
        assert!(path.join("retained").join("child").exists());
        assert!(path.join("file_bearing").join("empty_child").exists());

        create_dir_all(path.join("fresh")).unwrap();
        create_dir_all(path.join("verified")).unwrap();
        store
            .remove_empty_dirs(
                base.clone(),
                HashSet::from([base.clone().join("retained")]),
                HashSet::from([base.clone().join("verified")]),
                Some(
                    DateTime::<Utc>::from(std::time::SystemTime::now())
                        - chrono::TimeDelta::try_days(7).unwrap(),
                ),
            )
            .await
            .unwrap();

        assert!(path.join("fresh").exists());
        assert!(!path.join("verified").exists());
    }

    #[derive(Debug)]
    struct TestWrapper {
        calls: std::sync::atomic::AtomicUsize,

        return_value: Arc<dyn StorageBackend>,
    }

    impl WrappingObjectStore for TestWrapper {
        fn wrap(
            &self,
            _store_prefix: &str,
            _original: Arc<dyn StorageBackend>,
        ) -> Arc<dyn StorageBackend> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.return_value.clone()
        }
    }

    impl TestWrapper {
        fn called(&self) -> bool {
            self.calls.load(Ordering::Relaxed) > 0
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_short_circuit_wrapper_preserves_tracking_position(
        #[values(false, true)] apply_late: bool,
    ) {
        let cached = backend::native::from_object_store(Arc::new(InMemory::new()));
        let path = Path::from("cached");
        cached
            .put(&path, Bytes::from_static(b"cached value").into())
            .await
            .unwrap();
        let wrapper = Arc::new(TestWrapper {
            calls: Default::default(),
            return_value: cached,
        });
        let params = ObjectStoreParams {
            backend_wrappers: if apply_late {
                vec![]
            } else {
                vec![wrapper.clone()]
            },
            ..Default::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &params,
        )
        .await
        .unwrap();
        let mut store = store.as_ref().clone();
        if apply_late {
            store.apply_wrapper(wrapper.as_ref());
        }
        let store = Arc::new(store);
        for view in [store.clone(), store.for_external_blobs()] {
            store.io_stats_incremental();
            assert_eq!(
                view.read_object(&path)
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap(),
                b"cached value".as_slice()
            );
            let stats = store.io_stats_snapshot();
            assert_eq!(stats.read_iops, if apply_late { 0 } else { 1 });
            assert_eq!(stats.read_bytes, if apply_late { 0 } else { 12 });
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_builtin_constructors_do_not_install_tracking(#[values(false, true)] local: bool) {
        let dir = tempfile::tempdir().unwrap();
        let store = if local {
            ObjectStore::local()
        } else {
            ObjectStore::memory()
        };
        let path = if local {
            Path::from_absolute_path(dir.path().join("data")).unwrap()
        } else {
            Path::from("data")
        };
        store
            .put_payload(&path, Bytes::from_static(b"value").into())
            .await
            .unwrap();
        assert_eq!(
            store
                .read_object(&path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            b"value".as_slice()
        );
        assert_eq!(store.io_stats_snapshot().read_iops, 0);
        assert_eq!(store.io_stats_snapshot().write_iops, 0);
        #[cfg(feature = "metrics")]
        assert!(store.io_tracker.metrics_base().is_none());
    }

    #[tokio::test]
    async fn test_cos_restriction_only_applies_to_put_if_absent() {
        let backend = backend::native::from_object_store(Arc::new(InMemory::new()));
        let store =
            crate::utils::testing::TestObjectStoreProvider(BackendConfig::new(backend, "cos"))
                .build(
                    Url::parse("cos://bucket/").unwrap(),
                    &ObjectStoreParams::default(),
                );
        let source = Path::from("source");
        let target = Path::from("target");
        let renamed = Path::from("renamed");
        assert!(matches!(
            store
                .put_if_absent(&source, Bytes::from_static(b"data").into())
                .await,
            Err(crate::object_store::Error::NotSupported { .. })
        ));
        assert!(matches!(
            store.stat(&source).await,
            Err(crate::object_store::Error::NotFound { .. })
        ));
        store
            .put_payload_opts(
                &source,
                Bytes::from_static(b"data").into(),
                PutMode::Create.into(),
            )
            .await
            .unwrap();
        store.copy_if_not_exists(&source, &target).await.unwrap();
        store.rename_if_not_exists(&target, &renamed).await.unwrap();
        assert_eq!(
            store
                .read_object(&renamed)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            b"data".as_slice()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_external_view_counts_reads_once(#[values(false, true)] apply_late: bool) {
        let wrapper = Arc::new(IOTracker::default());
        let params = ObjectStoreParams {
            backend_wrappers: (!apply_late)
                .then(|| wrapper.clone() as Arc<dyn WrappingObjectStore>)
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let (store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            "memory:///",
            &params,
        )
        .await
        .unwrap();
        let mut store = store.as_ref().clone();
        if apply_late {
            store.apply_wrapper(wrapper.as_ref());
        }
        let store = Arc::new(store);
        let path = Path::from("payload");
        store.put(&path, b"blob").await.unwrap();
        for view in [store.clone(), store.for_external_blobs()] {
            store.io_stats_incremental();
            wrapper.incremental_stats();
            assert_eq!(
                view.read_range(&path, 0..4).await.unwrap(),
                b"blob".as_slice()
            );
            let stats = store.io_stats_snapshot();
            assert_eq!(stats.read_iops, 1);
            assert_eq!(stats.read_bytes, 4);
            assert_eq!(wrapper.stats().read_iops, 1);
        }
    }

    #[rstest]
    #[case::file("file")]
    #[case::storage("file-object-store")]
    #[cfg_attr(target_os = "linux", case::uring("file+uring"))]
    #[tokio::test]
    async fn test_local_io_uses_wrapped_backend(
        #[case] scheme: &str,
        #[values(false, true)] apply_late: bool,
    ) {
        let directory = TempStdDir::default();
        std::fs::write(directory.join("source"), b"disk").unwrap();
        let file_url = Url::from_directory_path(&directory).unwrap();
        let mut url = Url::parse(&format!("{scheme}:///")).unwrap();
        url.set_path(file_url.path());
        let replacement = backend::native::from_object_store(Arc::new(InMemory::new()));
        let wrapper = Arc::new(TestWrapper {
            calls: std::sync::atomic::AtomicUsize::new(0),
            return_value: replacement.clone(),
        });
        let params = ObjectStoreParams {
            backend_wrappers: (!apply_late)
                .then(|| wrapper.clone() as Arc<dyn WrappingObjectStore>)
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let (store, base) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            url.as_ref(),
            &params,
        )
        .await
        .unwrap();
        let store = if apply_late {
            let mut store = store.as_ref().clone();
            store.apply_wrapper(wrapper.as_ref());
            Arc::new(store)
        } else {
            store
        };
        assert!(!store.has_direct_local_paths());
        let source = base.clone().join("source");
        replacement
            .put(&source, Bytes::from_static(b"wrapped").into())
            .await
            .unwrap();

        // Both ordinary and external views must reach the selected backend,
        // including the known-size reader and local copy shortcuts.
        for view in [store.clone(), store.for_external_blobs()] {
            assert_eq!(
                view.open(&source).await.unwrap().get_all().await.unwrap(),
                b"wrapped".as_slice()
            );
            assert_eq!(
                view.open_with_size(&source, 7)
                    .await
                    .unwrap()
                    .get_all()
                    .await
                    .unwrap(),
                b"wrapped".as_slice()
            );
            let destination = base.clone().join("copy");
            view.copy(&source, &destination).await.unwrap();
            assert_eq!(
                replacement
                    .get(&destination)
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap(),
                b"wrapped".as_slice()
            );
            view.copy_via_stream(&source, &view, &base.clone().join("streamed"))
                .await
                .unwrap();
            view.put(&base.clone().join("new"), b"new").await.unwrap();
            assert_eq!(
                replacement
                    .get(&base.clone().join("new"))
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap(),
                b"new".as_slice()
            );
        }
        assert_eq!(std::fs::read(directory.join("source")).unwrap(), b"disk");
        for name in ["copy", "streamed", "new"] {
            assert!(
                !directory.join(name).exists(),
                "wrapper was bypassed for {name}"
            );
        }
        assert!(wrapper.called());
    }

    #[rstest]
    #[case::file("file")]
    #[case::storage("file-object-store")]
    #[cfg_attr(target_os = "linux", case::uring("file+uring"))]
    #[tokio::test]
    async fn test_custom_backend_does_not_inherit_native_file_access(
        #[case] scheme: &str,
        #[values(false, true)] is_local: bool,
    ) {
        let directory = TempStdDir::default();
        std::fs::write(directory.join("source"), b"disk").unwrap();
        std::fs::write(directory.join("unfinished#1"), b"staging").unwrap();
        let source = Path::from_absolute_path(directory.join("source")).unwrap();
        let base = Path::from_absolute_path(&directory).unwrap();
        let tracker = IOTracker::default();
        // This backend implements only Lance's trait, with no native ObjectStore impl.
        let backend = tracker.wrap(
            "custom",
            crate::object_store::backend::native::from_object_store(Arc::new(
                object_store::memory::InMemory::new(),
            )),
        );
        let store = {
            let url = Url::parse(&format!("{scheme}:///")).unwrap();
            let mut config = crate::object_store::BackendConfig::new(backend, url.scheme());
            config.io_parallelism = 1;
            config.is_local = is_local;
            crate::utils::testing::TestObjectStoreProvider(config).build(url, &Default::default())
        };
        assert_eq!(store.is_local(), is_local);
        assert!(!store.has_direct_local_paths());
        assert!(store.local_directory_path(&base).unwrap().is_none());
        store.put(&source, b"backend").await.unwrap();
        assert_eq!(
            store.open(&source).await.unwrap().get_all().await.unwrap(),
            b"backend".as_slice()
        );
        assert_eq!(
            store
                .open_with_size(&source, 7)
                .await
                .unwrap()
                .get_all()
                .await
                .unwrap(),
            b"backend".as_slice()
        );
        store
            .copy(&source, &base.clone().join("copy"))
            .await
            .unwrap();
        assert_eq!(
            store
                .open(&base.clone().join("copy"))
                .await
                .unwrap()
                .get_all()
                .await
                .unwrap(),
            b"backend".as_slice()
        );
        assert!(
            store
                .canonicalize_local_path(&base)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .read_local_staging_files(&base)
                .try_next()
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .local_staging_path(&Path::parse(format!("{base}/unfinished#1")).unwrap())
                .unwrap()
                .is_none()
        );
        store
            .remove_empty_dirs(base.clone(), HashSet::new(), HashSet::new(), None)
            .await
            .unwrap();
        store.remove_dir_all(base).await.unwrap();
        assert!(!store.exists(&source).await.unwrap());
        assert_eq!(std::fs::read(directory.join("source")).unwrap(), b"disk");
        assert_eq!(
            std::fs::read(directory.join("unfinished#1")).unwrap(),
            b"staging"
        );
        assert!(!directory.join("copy").exists());
    }

    #[rstest]
    #[case::file("file")]
    #[case::storage("file-object-store")]
    #[cfg_attr(target_os = "linux", case::uring("file+uring"))]
    #[tokio::test]
    async fn test_wrappers_preserve_native_directory_maintenance(
        #[case] scheme: &str,
        #[values(false, true)] apply_late: bool,
    ) {
        let directory = TempStdDir::default();
        create_dir_all(directory.join("empty")).unwrap();
        std::fs::write(directory.join("part#1"), b"staging").unwrap();
        let file_url = Url::from_directory_path(&directory).unwrap();
        let mut url = Url::parse(&format!("{scheme}:///")).unwrap();
        url.set_path(file_url.path());
        let wrapper: Arc<dyn WrappingObjectStore> = Arc::new(
            crate::utils::testing::TestWrappingObjectStore::new(|backend| backend),
        );
        let (store, base) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            url.as_ref(),
            &ObjectStoreParams {
                backend_wrappers: if apply_late {
                    vec![]
                } else {
                    vec![wrapper.clone()]
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let mut store = store.as_ref().clone();
        if apply_late {
            store.apply_wrapper(wrapper.as_ref());
        }
        store.apply_wrapper(&IOTracker::default());
        assert!(!store.has_direct_local_paths());
        let store = Arc::new(store);
        let view = store.for_external_blobs();
        assert_eq!(
            view.canonicalize_local_path(&base).await.unwrap(),
            Some(std::fs::canonicalize(&directory).unwrap())
        );
        assert_eq!(
            view.local_directory_path(&base).unwrap(),
            Some(directory.to_path_buf())
        );
        let files = view
            .read_local_staging_files(&base)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0].location,
            Path::parse(format!("{base}/part#1")).unwrap()
        );
        view.delete(&files[0].location).await.unwrap();
        assert!(!directory.join("part#1").exists());
        store
            .remove_empty_dirs(base.clone(), HashSet::new(), HashSet::new(), None)
            .await
            .unwrap();
        assert!(!directory.join("empty").exists());
        create_dir_all(directory.join("remove/child")).unwrap();
        std::fs::write(directory.join("remove/child/data"), b"data").unwrap();
        store.remove_dir_all(base.join("remove")).await.unwrap();
        assert!(!directory.join("remove").exists());
    }

    #[tokio::test]
    async fn test_external_view_retains_the_registry_binding() {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = ObjectStoreParams {
            backend_wrappers: vec![Arc::new(IOTracker::default())],
            ..Default::default()
        };
        let (store, path) =
            ObjectStore::from_uri_and_params(registry.clone(), "memory://payload", &params)
                .await
                .unwrap();
        store.put(&path, b"live").await.unwrap();
        let weak = Arc::downgrade(&store);
        let external = store.for_external_blobs();
        assert!(Arc::ptr_eq(&external, &store.for_external_blobs()));
        assert!(Arc::ptr_eq(&external, &external.for_external_blobs()));
        let cloned = Arc::new(store.as_ref().clone());
        assert!(!Arc::ptr_eq(&external, &cloned.for_external_blobs()));
        drop(cloned);
        drop(store);
        let (reopened, _) = ObjectStore::from_uri_and_params(registry, "memory://payload", &params)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&reopened, &weak.upgrade().unwrap()));
        assert_eq!(
            external
                .open(&path)
                .await
                .unwrap()
                .get_range(0..4)
                .await
                .unwrap()
                .as_ref(),
            b"live"
        );
        drop(reopened);
        drop(external);
        assert!(
            weak.upgrade().is_none(),
            "derived views must not form an Arc cycle"
        );
    }

    #[tokio::test]
    async fn test_wrapper_identity_is_stable_across_tasks() {
        let wrapper = Arc::new(TestWrapper {
            calls: std::sync::atomic::AtomicUsize::new(0),
            return_value: crate::object_store::backend::native::from_object_store(Arc::new(
                InMemory::new(),
            )),
        });
        let initial_params = ObjectStoreParams {
            backend_wrappers: vec![wrapper.clone()],
            ..ObjectStoreParams::default()
        };
        let task_params = tokio::spawn(async move {
            ObjectStoreParams {
                backend_wrappers: vec![wrapper],
                ..ObjectStoreParams::default()
            }
        })
        .await
        .unwrap();

        assert_eq!(initial_params, task_params);

        let mut initial_hasher = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(&initial_params, &mut initial_hasher);
        let mut task_hasher = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(&task_params, &mut task_hasher);
        assert_eq!(
            std::hash::Hasher::finish(&initial_hasher),
            std::hash::Hasher::finish(&task_hasher)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_wrapping_storage_option_is_used(#[values(false, true)] is_custom_store: bool) {
        // Make a store for the inner store first
        let mock_inner_store =
            crate::object_store::backend::native::from_object_store(Arc::new(InMemory::new()));
        let registry = Arc::new(ObjectStoreRegistry::default());

        assert_eq!(Arc::strong_count(&mock_inner_store), 1);

        let wrapper = Arc::new(TestWrapper {
            calls: std::sync::atomic::AtomicUsize::new(0),
            return_value: mock_inner_store.clone(),
        });

        let registry = if is_custom_store {
            crate::utils::testing::TestObjectStoreProvider(BackendConfig::new(
                backend::native::from_object_store(Arc::new(InMemory::new())),
                "memory",
            ))
            .registry()
        } else {
            registry
        };
        let params = ObjectStoreParams {
            backend_wrappers: vec![wrapper.clone()],
            ..ObjectStoreParams::default()
        };

        // not called yet
        assert!(!wrapper.called());

        let _ = ObjectStore::from_uri_and_params(registry, "memory:///", &params)
            .await
            .unwrap();

        // called after construction
        assert_eq!(wrapper.calls.load(Ordering::Relaxed), 1);

        // hard to compare two trait pointers as the point to vtables
        // using the ref count as a proxy to make sure that the store is correctly kept
        assert_eq!(Arc::strong_count(&mock_inner_store), 2);
    }

    #[tokio::test]
    async fn test_local_paths() {
        let file_path = TempStdFile::default();
        let mut writer = ObjectStore::create_local_writer(&file_path).await.unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        Writer::shutdown(&mut writer).await.unwrap();

        let reader = ObjectStore::open_local(&file_path).await.unwrap();
        let buf = reader.get_range(0..5).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_direct_local_writer_uses_standard_file_permissions() {
        let directory = TempStdDir::default();
        let reference_path = directory.join("reference");
        std::fs::File::create(&reference_path).unwrap();
        let expected_mode = std::fs::metadata(reference_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;

        let output_path = directory.join("output");
        let object_path = Path::from_absolute_path(&output_path).unwrap();
        let store = ObjectStore::local();
        let mut writer = store.create(&object_path).await.unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        Writer::shutdown(writer.as_mut()).await.unwrap();

        let actual_mode = std::fs::metadata(output_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(actual_mode, expected_mode);
    }

    #[tokio::test]
    async fn test_read_one() {
        let file_path = TempStdFile::default();
        let mut writer = ObjectStore::create_local_writer(&file_path).await.unwrap();
        writer.write_all(b"LOCAL").await.unwrap();
        Writer::shutdown(&mut writer).await.unwrap();

        let file_path_os =
            lance_core::storage::path::Path::parse(file_path.to_str().unwrap()).unwrap();
        let obj_store = ObjectStore::local();
        let buf = obj_store.read_one_all(&file_path_os).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");

        let buf = obj_store.read_one_range(&file_path_os, 0..5).await.unwrap();
        assert_eq!(buf.as_ref(), b"LOCAL");
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_windows_paths() {
        use std::path::Component;
        use std::path::Prefix;
        use std::path::Prefix::*;

        fn get_path_prefix(path: &StdPath) -> Prefix<'_> {
            match path.components().next().unwrap() {
                Component::Prefix(prefix_component) => prefix_component.kind(),
                _ => panic!(),
            }
        }

        fn get_drive_letter(prefix: Prefix) -> String {
            match prefix {
                Disk(bytes) => String::from_utf8(vec![bytes]).unwrap(),
                _ => panic!(),
            }
        }

        let tmp_path = TempStdFile::default();
        let prefix = get_path_prefix(&tmp_path);
        let drive_letter = get_drive_letter(prefix);

        write_to_file(
            &(format!("{drive_letter}:/test_folder/test.lance") + "/test_file"),
            "WINDOWS",
        )
        .unwrap();

        for uri in &[
            format!("{drive_letter}:/test_folder/test.lance"),
            format!("{drive_letter}:\\test_folder\\test.lance"),
        ] {
            let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
            let contents = read_from_store(store.as_ref(), &base.clone().join("test_file"))
                .await
                .unwrap();
            assert_eq!(contents, "WINDOWS");
        }
    }

    #[tokio::test]
    async fn test_cross_filesystem_copy() {
        // Create two temporary directories that simulate different filesystems
        let source_dir = TempStdDir::default();
        let dest_dir = TempStdDir::default();

        // Create a test file in the source directory
        let source_file_name = "test_file.txt";
        let source_file = source_dir.join(source_file_name);
        std::fs::write(&source_file, b"test content").unwrap();

        // Create ObjectStore for local filesystem
        let (store, base_path) = ObjectStore::from_uri(source_dir.to_str().unwrap())
            .await
            .unwrap();

        // Create paths relative to the ObjectStore base
        let from_path = base_path.clone().join(source_file_name);

        // Use crate::object_store::Path::parse for the destination
        let dest_file = dest_dir.join("copied_file.txt");
        let dest_str = dest_file.to_str().unwrap();
        let to_path = lance_core::storage::path::Path::parse(dest_str).unwrap();

        // Perform the copy operation
        store.copy(&from_path, &to_path).await.unwrap();

        // Verify the file was copied correctly
        assert!(dest_file.exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
    }

    #[rstest]
    #[case::copy("copy")]
    #[case::bulk("bulk")]
    #[case::stream("stream")]
    #[tokio::test]
    async fn test_copy_creates_parent_directories(
        #[case] mode: &str,
        #[values(false, true)] local_fsync: bool,
    ) {
        let source_dir = TempStdDir::default();
        let dest_dir = TempStdDir::default();

        // Create a test file in the source directory
        let source_file_name = "test_file.txt";
        let source_file = source_dir.join(source_file_name);
        std::fs::write(&source_file, b"test content").unwrap();

        let (source_store, base_path) = ObjectStore::from_uri(source_dir.to_str().unwrap())
            .await
            .unwrap();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("local_fsync".to_owned(), local_fsync.to_string())]),
            ))),
            ..Default::default()
        };
        let (destination_store, _) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            dest_dir.to_str().unwrap(),
            &params,
        )
        .await
        .unwrap();

        // Create paths
        let from_path = base_path.clone().join(source_file_name);

        // Create destination with nested directories that don't exist yet
        let dest_file = dest_dir.join("nested").join("dirs").join("copied_file.txt");
        let to_path = Path::from_absolute_path(&dest_file).unwrap();

        match mode {
            "copy" => destination_store.copy(&from_path, &to_path).await.unwrap(),
            "bulk" => {
                let result = source_store
                    .copy_bulk(&from_path, &destination_store, &to_path)
                    .await
                    .unwrap();
                assert_eq!(result.size, b"test content".len());
            }
            "stream" => {
                let result = source_store
                    .copy_via_stream(&from_path, &destination_store, &to_path)
                    .await
                    .unwrap();
                assert_eq!(result.size, b"test content".len());
            }
            _ => unreachable!(),
        }

        // Verify the file was copied correctly and directories were created
        assert!(dest_file.exists());
        assert!(dest_file.parent().unwrap().exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
        assert_eq!(std::fs::read(&source_file).unwrap(), b"test content");
    }

    /// Inner store that forwards everything to `InMemory` except single-shot
    /// server-side copy (`copy_opts`), which always fails. This lets a test
    /// prove that `ObjectStore::copy` fell back to a streaming multipart copy
    /// for an oversized source rather than issuing a single `CopyObject`.
    #[derive(Debug)]
    struct CopyFailingStore {
        inner: Arc<dyn crate::object_store::StorageBackend>,
    }

    impl Display for CopyFailingStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "CopyFailingStore")
        }
    }

    #[derive(Debug, Default)]
    struct MultipartObservations {
        part_count: AtomicUsize,
        abort_count: AtomicUsize,
        native_copy_count: AtomicUsize,
        delete_stream_count: AtomicUsize,
        delete_in_flight: AtomicUsize,
        /// High-water mark of concurrent deletions; 1 means serial.
        delete_max_in_flight: AtomicUsize,
    }

    /// Decrements the in-flight count when its stream is dropped.
    struct InFlightGuard(Arc<MultipartObservations>);

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            self.0.delete_in_flight.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Debug)]
    struct ObservedMultipartUpload {
        inner: Box<dyn MultipartUpload>,
        observations: Arc<MultipartObservations>,
        fail_parts: bool,
    }

    #[async_trait]
    impl MultipartUpload for ObservedMultipartUpload {
        fn put_part(&mut self, data: PutPayload) -> UploadPart {
            self.observations.part_count.fetch_add(1, Ordering::SeqCst);
            if self.fail_parts {
                return Box::pin(async {
                    Err(crate::object_store::Error::Generic {
                        store: "ObservedMultipartStore",
                        source: "injected multipart part failure".into(),
                    })
                });
            }
            self.inner.put_part(data)
        }

        async fn complete(&mut self) -> OSResult<PutResult> {
            self.inner.complete().await
        }

        async fn abort(&mut self) -> OSResult<()> {
            self.observations.abort_count.fetch_add(1, Ordering::SeqCst);
            self.inner.abort().await
        }
    }

    #[derive(Debug)]
    struct ObservedMultipartStore {
        inner: Arc<dyn crate::object_store::StorageBackend>,
        observations: Arc<MultipartObservations>,
        fail_parts: bool,
        destination_size_adjustment: u64,
    }

    impl Display for ObservedMultipartStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "ObservedMultipartStore")
        }
    }

    #[async_trait]
    impl crate::object_store::StorageBackend for ObservedMultipartStore {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            options: crate::object_store::PaginatedListOptions,
        ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
            let _ = (prefix, options);
            Err(crate::object_store::Error::NotSupported {
                source: "native pagination is unavailable".into(),
            })
        }

        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn crate::object_store::backend::MultipartUpload>> {
            let inner = self.inner.put_multipart_opts(location, opts).await?;
            Ok(Box::new(ObservedMultipartUpload {
                inner,
                observations: self.observations.clone(),
                fail_parts: self.fail_parts,
            }))
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            let is_head = options.head;
            let mut result = self.inner.get_opts(location, options).await?;
            if is_head && location.filename() == Some("destination.bin") {
                result.meta.size = result
                    .meta
                    .size
                    .checked_add(self.destination_size_adjustment)
                    .expect("test destination size should not overflow");
            }
            Ok(result)
        }

        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            let observations = self.observations.clone();
            observations
                .delete_stream_count
                .fetch_add(1, Ordering::SeqCst);
            let in_flight = observations.delete_in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            observations
                .delete_max_in_flight
                .fetch_max(in_flight, Ordering::SeqCst);
            let guard = InFlightGuard(observations);
            let inner = self.inner.delete_stream(locations);
            // An in-memory delete resolves on first poll, so overlap is only
            // observable if the deletion yields.
            async move {
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(5)).await;
                inner.map(move |result| {
                    let _ = &guard;
                    result
                })
            }
            .flatten_stream()
            .boxed()
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.observations
                .native_copy_count
                .fetch_add(1, Ordering::SeqCst);
            self.inner.copy_opts(from, to, opts).await
        }
        async fn rename_opts(
            &self,
            from: &crate::object_store::Path,
            to: &crate::object_store::Path,
            options: crate::object_store::RenameOptions,
        ) -> crate::object_store::Result<()> {
            let mode = match options.target_mode {
                crate::object_store::RenameTargetMode::Create => {
                    crate::object_store::CopyMode::Create
                }
                crate::object_store::RenameTargetMode::Overwrite => {
                    crate::object_store::CopyMode::Overwrite
                }
            };
            self.copy_opts(
                from,
                to,
                crate::object_store::CopyOptions {
                    mode,
                    extensions: options.extensions,
                },
            )
            .await?;
            (self as &dyn crate::object_store::StorageBackend)
                .delete(from)
                .await
        }
    }

    #[async_trait]
    impl crate::object_store::StorageBackend for CopyFailingStore {
        async fn list_paginated(
            &self,
            prefix: Option<&str>,
            options: crate::object_store::PaginatedListOptions,
        ) -> crate::object_store::Result<crate::object_store::PaginatedListResult> {
            let _ = (prefix, options);
            Err(crate::object_store::Error::NotSupported {
                source: "native pagination is unavailable".into(),
            })
        }

        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn crate::object_store::backend::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }
        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }
        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }
        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(&self, _from: &Path, _to: &Path, _opts: CopyOptions) -> OSResult<()> {
            Err(crate::object_store::Error::NotSupported {
                source: "single-shot copy disabled in test".into(),
            })
        }
        async fn rename_opts(
            &self,
            from: &crate::object_store::Path,
            to: &crate::object_store::Path,
            options: crate::object_store::RenameOptions,
        ) -> crate::object_store::Result<()> {
            let mode = match options.target_mode {
                crate::object_store::RenameTargetMode::Create => {
                    crate::object_store::CopyMode::Create
                }
                crate::object_store::RenameTargetMode::Overwrite => {
                    crate::object_store::CopyMode::Overwrite
                }
            };
            self.copy_opts(
                from,
                to,
                crate::object_store::CopyOptions {
                    mode,
                    extensions: options.extensions,
                },
            )
            .await?;
            (self as &dyn crate::object_store::StorageBackend)
                .delete(from)
                .await
        }
    }

    #[rstest]
    #[case::exceeds_limit(8, true)]
    #[case::within_limit(u64::MAX, false)]
    #[tokio::test]
    async fn test_copy_obeys_provider_size_limit(#[case] limit: u64, #[case] streams: bool) {
        // This backend rejects native copy. Only a declared limit can route the
        // public copy operation through multipart streaming.
        let backend = Arc::new(CopyFailingStore {
            inner: crate::object_store::backend::native::from_object_store(Arc::new(
                InMemory::new(),
            )),
        });
        let mut config = BackendConfig::new(backend, "memory");
        config.is_remote = false;
        config.max_single_copy_size = Some(limit);
        let store = crate::utils::testing::TestObjectStoreProvider(config)
            .build(Url::parse("memory:///").unwrap(), &Default::default());
        let from = Path::from("source.bin");
        let to = Path::from("copy.bin");
        let contents = b"streaming multipart copy payload well past the tiny threshold";
        store.put(&from, contents).await.unwrap();
        let result = store.copy(&from, &to).await;
        if streams {
            result.unwrap();
            assert_eq!(store.read_one_all(&to).await.unwrap().as_ref(), contents);
        } else {
            assert!(result.is_err());
            assert!(!store.exists(&to).await.unwrap());
        }
    }

    #[tokio::test]
    async fn test_copy_via_stream_never_uses_native_copy() {
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(CopyFailingStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"stream raw bytes instead of issuing native copy";
        store.put(&source, contents).await.unwrap();

        let result = store
            .copy_via_stream(&source, &store, &destination)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            contents
        );
    }

    #[tokio::test]
    async fn test_direct_listing_does_not_retry() {
        use crate::utils::testing::{ProxyBackend, ProxyBackendPolicy, TestWrappingObjectStore};
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = calls.clone();
        let policy = Arc::new(std::sync::Mutex::new(ProxyBackendPolicy::new()));
        policy.lock().unwrap().set_before_policy(
            "list failure",
            Arc::new(move |op, _| {
                if op == "list" {
                    observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return Err(LanceError::io("temporary list failure"));
                }
                Ok(())
            }),
        );
        let mut storage = ObjectStore::memory();
        storage.apply_wrapper(&TestWrappingObjectStore::new(|backend| {
            Arc::new(ProxyBackend::new(backend, policy.clone()))
        }));
        let error = storage
            .list_objects(None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Generic { .. }));
        assert!(error.to_string().contains("temporary list failure"));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[rstest]
    #[case::file("file", true)]
    #[case::backend("file-object-store", false)]
    #[cfg_attr(target_os = "linux", case::uring("file+uring", true))]
    #[tokio::test]
    async fn test_conditional_create_keeps_provider_strategy(
        #[case] scheme: &str,
        #[case] staged: bool,
    ) {
        use crate::utils::testing::{ProxyBackend, ProxyBackendPolicy, TestWrappingObjectStore};
        let temp = tempfile::tempdir().unwrap();
        let uri = format!(
            "{scheme}://{}{}",
            if cfg!(windows) { "/" } else { "" },
            temp.path().display()
        );
        let (store, base) = ObjectStore::from_uri(&uri).await.unwrap();
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = calls.clone();
        let policy = Arc::new(std::sync::Mutex::new(ProxyBackendPolicy::new()));
        policy.lock().unwrap().set_before_policy(
            "failure",
            Arc::new(move |op, _| {
                observed.lock().unwrap().push(op.to_string());
                match op {
                    "rename" => Err(LanceError::not_supported("conditional rename unavailable")),
                    "delete" => Err(LanceError::io("cleanup unavailable")),
                    _ => Ok(()),
                }
            }),
        );
        let mut store = (*store).clone();
        store.apply_wrapper(&TestWrappingObjectStore::new(|backend| {
            Arc::new(ProxyBackend::new(backend, policy.clone()))
        }));
        let path = base.join(".manifest");
        let result = store.put_if_absent(&path, b"data".to_vec().into()).await;
        if staged {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Generic { .. }));
            assert!(error.to_string().contains("conditional rename unavailable"));
            assert_eq!(*calls.lock().unwrap(), ["put", "rename", "delete"]);
            assert!(!temp.path().join(".manifest").exists());
        } else {
            result.unwrap();
            assert_eq!(*calls.lock().unwrap(), ["put"]);
            assert_eq!(
                std::fs::read(temp.path().join(".manifest")).unwrap(),
                b"data"
            );
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_conditional_copy_preserves_backend_result(
        #[values(false, true)] unsupported: bool,
    ) {
        let mut store = ObjectStore::memory();
        if unsupported {
            store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
                |_backend| {
                    Arc::new(CopyFailingStore {
                        inner: crate::object_store::backend::native::from_object_store(Arc::new(
                            InMemory::new(),
                        )),
                    })
                },
            ));
        }
        let source = Path::from("source");
        let destination = Path::from("destination");
        store.put(&source, b"new").await.unwrap();
        store.put(&destination, b"old").await.unwrap();
        let error = store
            .copy_if_not_exists(&source, &destination)
            .await
            .unwrap_err();
        if unsupported {
            assert!(matches!(
                error,
                crate::object_store::Error::NotSupported { .. }
            ));
        } else {
            assert!(matches!(
                error,
                crate::object_store::Error::AlreadyExists { .. }
                    | crate::object_store::Error::Precondition { .. }
            ));
        }
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            b"old"
        );
        store.delete(&destination).await.unwrap();
        let result = store.copy_if_not_exists(&source, &destination).await;
        if unsupported {
            assert!(matches!(
                result,
                Err(crate::object_store::Error::NotSupported { .. })
            ));
            assert!(matches!(
                store.stat(&destination).await,
                Err(crate::object_store::Error::NotFound { .. })
            ));
            return;
        }
        result.unwrap();
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            b"new"
        );
        let error = store
            .copy_if_not_exists(&"missing".into(), &"other".into())
            .await
            .unwrap_err();
        assert!(matches!(error, crate::object_store::Error::NotFound { .. }));
    }

    #[rstest]
    #[case::native(3, false)]
    #[case::streamed(5 * 1024 * 1024 * 1024, false)]
    #[case::upload_failure(5 * 1024 * 1024 * 1024, true)]
    #[tokio::test]
    async fn test_copy_with_known_size(#[case] size: u64, #[case] fail_parts: bool) {
        let observations = Arc::new(MultipartObservations::default());
        let mut store = ObjectStore::memory();
        let backend = ObservedMultipartStore {
            inner: crate::object_store::backend::native::from_object_store(Arc::new(
                InMemory::new(),
            )),
            observations: observations.clone(),
            fail_parts,
            destination_size_adjustment: 0,
        };
        // Report a large known size without allocating a multi-gigabyte fixture.
        backend
            .inner
            .put(&"source".into(), Bytes::from_static(b"new").into())
            .await
            .unwrap();
        let backend = Arc::new(backend);
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(|_| {
            backend.clone()
        }));
        let result = store
            .copy_with_size(&"source".into(), &"destination".into(), size)
            .await;
        if fail_parts {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("injected multipart part failure")
            );
            assert_eq!(observations.abort_count.load(Ordering::SeqCst), 1);
            assert!(matches!(
                store.stat(&"destination".into()).await,
                Err(crate::object_store::Error::NotFound { .. })
            ));
        } else {
            result.unwrap();
            assert_eq!(
                store
                    .read_one_all(&"destination".into())
                    .await
                    .unwrap()
                    .as_ref(),
                b"new"
            );
        }
        assert_eq!(
            observations.native_copy_count.load(Ordering::SeqCst),
            usize::from(size == 3)
        );
        assert_eq!(
            observations.part_count.load(Ordering::SeqCst),
            usize::from(size != 3)
        );
        assert!(matches!(
            store
                .copy_with_size(&"missing".into(), &"other".into(), size)
                .await,
            Err(crate::object_store::Error::NotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_copy_if_not_exists_or_same_size_accepts_equal_size_local_retry() {
        let directory = TempStdDir::default();
        let (store, base_path) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let source = base_path.clone().join("source.bin");
        let destination = base_path.join("destination.bin");
        store.put(&source, b"new").await.unwrap();
        store.put(&destination, b"old").await.unwrap();

        let result = store
            .copy_if_not_exists_or_same_size(&source, &destination)
            .await
            .unwrap();

        assert_eq!(result.size, 3);
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            b"old"
        );
    }

    #[tokio::test]
    async fn test_copy_if_not_exists_uses_native_copy() {
        let observations = Arc::new(MultipartObservations::default());
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = vec![42; crate::object_writer::initial_upload_size() + 1];
        store.put(&source, &contents).await.unwrap();
        observations.part_count.store(0, Ordering::SeqCst);

        let result = store
            .copy_if_not_exists_or_same_size(&source, &destination)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(observations.native_copy_count.load(Ordering::SeqCst), 1);
        assert_eq!(observations.part_count.load(Ordering::SeqCst), 0);
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            contents.as_slice()
        );
    }

    #[tokio::test]
    async fn test_remove_stream_removes_every_path() {
        let store = ObjectStore::memory();
        let paths: Vec<Path> = (0..64).map(|i| Path::from(format!("obj-{i:03}"))).collect();
        for path in &paths {
            store.put(path, b"x").await.unwrap();
        }

        let to_remove = futures::stream::iter(paths.clone().into_iter().map(Ok)).boxed();
        let mut reported = store
            .remove_stream(to_remove)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        // Concurrent deletion does not preserve order.
        reported.sort_unstable();
        let mut expected = paths.clone();
        expected.sort_unstable();
        assert_eq!(reported, expected);
        for path in &paths {
            assert!(!store.exists(path).await.unwrap(), "{path} still present");
        }
    }

    #[tokio::test]
    async fn test_remove_stream_tolerates_already_deleted() {
        // Callers list first and delete after, so a path can disappear in between.
        // Failing there abandons the whole sweep over one absent object.
        let store = ObjectStore::memory();
        let present = Path::from("present");
        let absent = Path::from("never-written");
        store.put(&present, b"x").await.unwrap();

        let to_remove =
            futures::stream::iter(vec![Ok(absent.clone()), Ok(present.clone())]).boxed();
        let mut removed = store
            .remove_stream(to_remove)
            .try_collect::<Vec<_>>()
            .await
            .expect("an already-absent path must not fail the stream");

        removed.sort_unstable();
        let mut expected = vec![absent, present.clone()];
        expected.sort_unstable();
        assert_eq!(removed, expected);
        assert!(!store.exists(&present).await.unwrap());
    }

    #[tokio::test]
    async fn test_remove_stream_deletes_concurrently() {
        // Serial deletion is a round trip per object; the high-water mark is 1
        // exactly when that regression happens.
        let observations = Arc::new(MultipartObservations::default());
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let paths: Vec<Path> = (0..64).map(|i| Path::from(format!("obj-{i:03}"))).collect();
        for path in &paths {
            store.put(path, b"x").await.unwrap();
        }

        let to_remove = futures::stream::iter(paths.clone().into_iter().map(Ok)).boxed();
        let removed = store
            .remove_stream(to_remove)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(removed.len(), paths.len());
        assert!(
            observations.delete_max_in_flight.load(Ordering::SeqCst) > 1,
            "deletions ran one at a time; remove_stream must overlap them"
        );
    }

    #[tokio::test]
    async fn test_bulk_copy_streams_when_server_side_copy_is_disabled() {
        let observations = Arc::new(MultipartObservations::default());
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"stream by default";
        store.put(&source, contents).await.unwrap();

        let result = store
            .copy_bulk_with_server_side_copy(&source, &store, &destination, false)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(observations.native_copy_count.load(Ordering::SeqCst), 0);
        assert_eq!(
            store.read_one_all(&destination).await.unwrap().as_ref(),
            contents
        );
    }

    fn remote_memory_store() -> ObjectStore {
        let config = BackendConfig::new(
            crate::object_store::backend::native::from_object_store(Arc::new(InMemory::new())),
            "test-cloud",
        );
        crate::utils::testing::TestObjectStoreProvider(config).build(
            Url::parse("test-cloud://bucket").unwrap(),
            &ObjectStoreParams::default(),
        )
    }

    #[test]
    #[serial(server_side_copy_env)]
    fn test_server_side_copy_environment_policy() {
        let previous_value = std::env::var_os(SERVER_SIDE_COPY_ENABLED_ENV);
        let store = remote_memory_store();
        let destination_store = store.clone();

        // SAFETY: this serialized test is the only test that mutates this task-specific
        // environment variable, and it restores the original value before returning.
        unsafe { std::env::remove_var(SERVER_SIDE_COPY_ENABLED_ENV) };
        assert!(!store.uses_server_side_copy(&destination_store));

        // SAFETY: see the serialized-test guarantee above.
        unsafe { std::env::set_var(SERVER_SIDE_COPY_ENABLED_ENV, "true") };
        assert!(store.uses_server_side_copy(&destination_store));

        // SAFETY: restore the process environment before the test returns.
        unsafe {
            match previous_value {
                Some(value) => std::env::set_var(SERVER_SIDE_COPY_ENABLED_ENV, value),
                None => std::env::remove_var(SERVER_SIDE_COPY_ENABLED_ENV),
            }
        }
    }

    #[tokio::test]
    async fn test_bulk_copy_uses_server_side_copy_when_enabled_for_same_store() {
        let observations = Arc::new(MultipartObservations::default());
        let mut source_store = remote_memory_store();
        source_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));
        let destination_store = source_store.clone();

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"use native copy when explicitly enabled";
        source_store.put(&source, contents).await.unwrap();

        let result = source_store
            .copy_bulk_with_server_side_copy(&source, &destination_store, &destination, true)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(observations.native_copy_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            destination_store
                .read_one_all(&destination)
                .await
                .unwrap()
                .as_ref(),
            contents
        );
    }

    #[tokio::test]
    async fn test_bulk_copy_streams_for_distinct_clients_with_same_prefix() {
        let shared_inner =
            crate::object_store::backend::native::from_object_store(Arc::new(InMemory::new()));
        let source_observations = Arc::new(MultipartObservations::default());
        let mut source_store = remote_memory_store();
        source_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: shared_inner.clone(),
                    observations: source_observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));
        let destination_observations = Arc::new(MultipartObservations::default());
        let mut destination_store = remote_memory_store();
        destination_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: shared_inner.clone(),
                    observations: destination_observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"use native copy when explicitly enabled";
        source_store.put(&source, contents).await.unwrap();

        let result = source_store
            .copy_bulk_with_server_side_copy(&source, &destination_store, &destination, true)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(
            source_observations.native_copy_count.load(Ordering::SeqCst),
            0
        );
        assert_eq!(
            destination_observations
                .native_copy_count
                .load(Ordering::SeqCst),
            0
        );
        assert_eq!(
            destination_store
                .read_one_all(&destination)
                .await
                .unwrap()
                .as_ref(),
            contents
        );
    }

    #[tokio::test]
    async fn test_bulk_copy_rejects_server_side_destination_size_mismatch() {
        let observations = Arc::new(MultipartObservations::default());
        let mut source_store = remote_memory_store();
        source_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 1,
                })
            },
        ));
        let destination_store = source_store.clone();

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        source_store
            .put(&source, b"validate native copy")
            .await
            .unwrap();

        let error = source_store
            .copy_bulk_with_server_side_copy(&source, &destination_store, &destination, true)
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("destination size mismatch"),
            "expected validation failure, got: {error}"
        );
        assert_eq!(observations.native_copy_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_bulk_copy_streams_across_stores_when_server_side_copy_is_enabled() {
        let source_store = ObjectStore::memory();
        let observations = Arc::new(MultipartObservations::default());
        let mut destination_store = ObjectStore::memory();
        destination_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"cross-store copies must stream";
        source_store.put(&source, contents).await.unwrap();

        let result = source_store
            .copy_bulk_with_server_side_copy(&source, &destination_store, &destination, true)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert_eq!(observations.native_copy_count.load(Ordering::SeqCst), 0);
        assert_eq!(
            destination_store
                .read_one_all(&destination)
                .await
                .unwrap()
                .as_ref(),
            contents
        );
    }

    #[tokio::test]
    async fn test_copy_via_stream_preserves_local_not_found() {
        let directory = TempStdDir::default();
        let (store, base_path) = ObjectStore::from_uri(directory.to_str().unwrap())
            .await
            .unwrap();
        let source = base_path.clone().join("missing.bin");
        let destination = base_path.join("destination.bin");

        let error = store
            .copy_via_stream(&source, &store, &destination)
            .await
            .unwrap_err();

        assert!(
            error.is_not_found(),
            "expected not-found error, got: {error}"
        );
    }

    #[tokio::test]
    async fn test_copy_via_stream_uses_multiple_parts() {
        let mut source_store = ObjectStore::memory();
        source_store.max_iop_size = 1024 * 1024;
        let observations = Arc::new(MultipartObservations::default());
        let mut destination_store = ObjectStore::memory();
        destination_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: false,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = vec![42; crate::object_writer::initial_upload_size() * 2 + 1];
        source_store.put(&source, &contents).await.unwrap();

        let result = source_store
            .copy_via_stream(&source, &destination_store, &destination)
            .await
            .unwrap();

        assert_eq!(result.size, contents.len());
        assert!(
            observations.part_count.load(Ordering::SeqCst) >= 2,
            "stream copy should split a large destination into multiple upload parts"
        );
        assert_eq!(
            destination_store
                .read_one_all(&destination)
                .await
                .unwrap()
                .as_ref(),
            contents.as_slice()
        );
    }

    #[tokio::test]
    async fn test_copy_via_stream_aborts_failed_upload_and_retains_source() {
        let source_store = ObjectStore::memory();
        let observations = Arc::new(MultipartObservations::default());
        let mut destination_store = ObjectStore::memory();
        destination_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: observations.clone(),
                    fail_parts: true,
                    destination_size_adjustment: 0,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = vec![7; crate::object_writer::initial_upload_size() * 2];
        source_store.put(&source, &contents).await.unwrap();

        let error = source_store
            .copy_via_stream(&source, &destination_store, &destination)
            .await
            .unwrap_err();
        let error_message = error.to_string();
        assert!(
            (error_message.contains("destination write")
                || error_message.contains("destination completion"))
                && error_message.contains("injected multipart part failure"),
            "expected upload-stage context and the underlying error, got: {error}"
        );

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if observations.abort_count.load(Ordering::SeqCst) > 0 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("multipart abort should complete");
        assert_eq!(observations.abort_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            source_store.read_one_all(&source).await.unwrap().as_ref(),
            contents.as_slice()
        );
        assert!(!destination_store.exists(&destination).await.unwrap());
    }

    #[tokio::test]
    async fn test_copy_via_stream_rejects_destination_size_mismatch() {
        let source_store = ObjectStore::memory();
        let mut destination_store = ObjectStore::memory();
        destination_store.apply_wrapper(&crate::utils::testing::TestWrappingObjectStore::new(
            |_backend| {
                Arc::new(ObservedMultipartStore {
                    inner: crate::object_store::backend::native::from_object_store(Arc::new(
                        InMemory::new(),
                    )),
                    observations: Arc::new(MultipartObservations::default()),
                    fail_parts: false,
                    destination_size_adjustment: 1,
                })
            },
        ));

        let source = Path::from("source.bin");
        let destination = Path::from("destination.bin");
        let contents = b"validate the destination after completion";
        source_store.put(&source, contents).await.unwrap();

        let error = source_store
            .copy_via_stream(&source, &destination_store, &destination)
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("destination size mismatch"),
            "expected validation failure, got: {error}"
        );
    }

    #[test]
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn test_client_options_extracts_headers() {
        let opts = StorageOptions(HashMap::from([
            ("headers.x-custom-foo".to_string(), "bar".to_string()),
            ("headers.x-ms-version".to_string(), "2023-11-03".to_string()),
            ("region".to_string(), "us-west-2".to_string()),
        ]));
        let client_options = opts.client_options().unwrap();

        // Verify non-header keys are not consumed as headers by creating
        // another StorageOptions with no headers.* keys.
        let opts_no_headers = StorageOptions(HashMap::from([(
            "region".to_string(),
            "us-west-2".to_string(),
        )]));
        opts_no_headers.client_options().unwrap();

        // Smoke test: the client_options with headers should be usable
        // in a builder (we can't inspect the headers directly, but building
        // should not fail).
        #[cfg(feature = "gcp")]
        {
            use object_store::gcp::GoogleCloudStorageBuilder;
            let _builder = GoogleCloudStorageBuilder::new()
                .with_client_options(client_options)
                .with_url("gs://test-bucket");
        }
    }

    #[test]
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn test_client_options_rejects_invalid_header_name() {
        let opts = StorageOptions(HashMap::from([(
            "headers.bad header".to_string(),
            "value".to_string(),
        )]));
        let err = opts.client_options().unwrap_err();
        assert!(err.to_string().contains("invalid header name"));
    }

    #[test]
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn test_client_options_rejects_invalid_header_value() {
        let opts = StorageOptions(HashMap::from([(
            "headers.x-good-name".to_string(),
            "bad\x01value".to_string(),
        )]));
        let err = opts.client_options().unwrap_err();
        assert!(err.to_string().contains("invalid header value"));
    }

    #[test]
    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    fn test_client_options_empty_when_no_header_keys() {
        let opts = StorageOptions(HashMap::from([
            ("region".to_string(), "us-east-1".to_string()),
            ("access_key_id".to_string(), "AKID".to_string()),
        ]));
        opts.client_options().unwrap();
    }
}
