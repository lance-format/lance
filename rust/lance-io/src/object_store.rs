// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extend [object_store::ObjectStore] functionalities

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::tracing::{Span, field::Empty, instrument};
use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{FutureExt, Stream};
use futures::{StreamExt, TryStreamExt, future, stream::BoxStream};
use lance_core::deepsize::DeepSizeOf;
use lance_core::error::LanceOptionExt;
use lance_core::utils::parse::{parse_env_as_bool, str_is_truthy};
use list_retry::ListRetryStream;
use object_store::DynObjectStore;
use object_store::ObjectStoreExt as OSObjectStoreExt;
#[cfg(feature = "aws")]
use object_store::aws::AwsCredentialProvider;
use object_store::list::PaginatedListStore;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
use object_store::{ClientOptions, HeaderMap, HeaderValue};
use object_store::{
    ListResult, ObjectMeta, ObjectStore as OSObjectStore, PutMode, PutOptions, PutPayload,
    path::Path,
};
use providers::local::FileStoreProvider;
use providers::memory::MemoryStoreProvider;
use tokio::io::AsyncWriteExt;
use url::Url;

use super::local::LocalObjectReader;
#[cfg(target_os = "linux")]
use crate::uring::{UringCurrentThreadReader, UringReader};
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub(crate) mod dynamic_credentials;
#[cfg(any(feature = "oss", feature = "huggingface", feature = "tos"))]
pub(crate) mod dynamic_opendal;
mod list_retry;
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
use crate::object_reader::SmallReader;
use crate::object_writer::{LocalWriter, WriteResult};
use crate::scheduler::{SCHEDULER_ERROR_MODE_STORAGE_OPTION, SchedulerErrorMode};
use crate::traits::{WriteExt, Writer};
use crate::utils::tracking_store::{IOTracker, IoStats};
use crate::{object_reader::CloudObjectReader, object_writer::ObjectWriter, traits::Reader};
use lance_core::{Error, Result};

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
) -> Error {
    Error::io_source(Box::new(StreamCopyError {
        stage,
        source_path: source_path.to_string(),
        destination_path: destination_path.to_string(),
        source: Box::new(source),
    }))
}

pub use providers::{ObjectStoreProvider, ObjectStoreRegistry};
pub use read_dir::ReadDirOptions;
pub use storage_options::{
    BASE_SCOPED_OPTION_PREFIX, BaseScopedStorageOptionsProvider, EXPIRES_AT_MILLIS_KEY,
    LanceNamespaceStorageOptionsProvider, REFRESH_OFFSET_MILLIS_KEY, StorageOptionsAccessor,
    StorageOptionsProvider, has_base_scoped_options, parse_base_scoped_key,
    resolve_base_scoped_options,
};

#[async_trait]
pub trait ObjectStoreExt {
    /// Returns true if the file exists.
    async fn exists(&self, path: &Path) -> Result<bool>;

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>>;
}

#[async_trait]
pub(super) trait LocalDirOperations: std::fmt::Debug + Send + Sync {
    async fn remove_dir_all(&self, path: &Path) -> Result<()>;
}

#[async_trait]
impl<O: OSObjectStore + ?Sized> ObjectStoreExt for O {
    fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>> {
        let output = self.list(Some(dir_path.into())).map_err(|e| e.into());
        if let Some(unmodified_since_val) = unmodified_since {
            output
                .try_filter(move |file| future::ready(file.last_modified <= unmodified_since_val))
                .boxed()
        } else {
            output.boxed()
        }
    }

    async fn exists(&self, path: &Path) -> Result<bool> {
        match self.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Wraps [ObjectStore](object_store::ObjectStore)
#[derive(Clone)]
pub struct ObjectStore {
    // Inner object store
    pub inner: Arc<dyn OSObjectStore>,
    // Provider-owned native directory operations for rooted local stores.
    local_dir_operations: Option<Arc<dyn LocalDirOperations>>,
    scheme: String,
    block_size: usize,
    max_iop_size: u64,
    /// Whether to use constant size upload parts for multipart uploads. This
    /// is only necessary for Cloudflare R2.
    pub use_constant_size_upload_parts: bool,
    /// Whether we can assume that the list of files is lexically ordered. This
    /// is true for object stores, but not for local filesystems.
    pub list_is_lexically_ordered: bool,
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
    pub store_prefix: String,
    /// The backend's paginated listing API, when it has one. `None` means
    /// [`Self::read_dir_page`] has to list a directory in full to page through it.
    pub(crate) paginated_lister: Option<Arc<dyn PaginatedListStore>>,
}

// Hand-written because `PaginatedListStore` is not `Debug`.
impl std::fmt::Debug for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStore")
            .field("inner", &self.inner)
            .field("scheme", &self.scheme)
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
            .field("paginated_lister", &self.paginated_lister.is_some())
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

pub trait WrappingObjectStore: std::fmt::Debug + Send + Sync {
    /// Wrap an object store with additional functionality
    ///
    /// The store_prefix is a string which uniquely identifies the object
    /// store being wrapped.
    fn wrap(&self, store_prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore>;

    /// Wrap the paginated listing API that goes with the store, if it has one.
    ///
    /// [`ObjectStore::read_dir_page`] pushes the page size and the resume position into
    /// [`PaginatedListStore`], which is a separate trait from [`OSObjectStore`] and so cannot
    /// be reached through the store [`Self::wrap`] returns. A listing that is pushed down
    /// therefore does not pass through [`Self::wrap`], and this is where a wrapper says what
    /// should happen instead:
    ///
    /// - `Some(lister)` keeps the pushdown, wrapping the lister or handing back the one
    ///   given. Right for a wrapper that observes rather than intercepts — metering, caching,
    ///   mirroring writes.
    /// - `None` gives up the pushdown, so listings go through [`Self::wrap`] as a full
    ///   directory read. Right for a wrapper that hides, rewrites or fails paths, which a
    ///   pushed-down listing would otherwise walk straight past.
    ///
    /// A wrapper that keeps the pushdown must leave the listing itself alone: setting
    /// [`offset`](object_store::list::PaginatedListOptions::offset) or changing the delimiter
    /// breaks paging, since `read_dir_page` reads one directory level and resumes by the token
    /// it got back.
    ///
    /// There is deliberately no default: getting this wrong is either a silent loss of speed
    /// or a silent loss of the wrapper, and neither announces itself.
    fn wrap_paginated(
        &self,
        store_prefix: &str,
        original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>>;
}

#[derive(Debug, Clone)]
pub struct ChainedWrappingObjectStore {
    wrappers: Vec<Arc<dyn WrappingObjectStore>>,
}

impl ChainedWrappingObjectStore {
    pub fn new(wrappers: Vec<Arc<dyn WrappingObjectStore>>) -> Self {
        Self { wrappers }
    }

    pub fn add_wrapper(&mut self, wrapper: Arc<dyn WrappingObjectStore>) {
        self.wrappers.push(wrapper);
    }
}

impl WrappingObjectStore for ChainedWrappingObjectStore {
    fn wrap(&self, store_prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        self.wrappers
            .iter()
            .fold(original, |acc, wrapper| wrapper.wrap(store_prefix, acc))
    }

    // One wrapper giving up the pushdown gives it up for the chain: the listing has to go
    // through `wrap`, which is every wrapper in the chain at once.
    fn wrap_paginated(
        &self,
        store_prefix: &str,
        original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        self.wrappers.iter().try_fold(original, |acc, wrapper| {
            wrapper.wrap_paginated(store_prefix, acc)
        })
    }
}

/// Parameters to create an [ObjectStore]
///
#[derive(Debug, Clone)]
pub struct ObjectStoreParams {
    pub block_size: Option<usize>,
    #[deprecated(note = "Implement an ObjectStoreProvider instead")]
    pub object_store: Option<(Arc<DynObjectStore>, Url)>,
    /// Refresh offset for AWS credentials when using the legacy AWS credentials path.
    /// For StorageOptionsAccessor, use `refresh_offset_millis` storage option instead.
    pub s3_credentials_refresh_offset: Duration,
    #[cfg(feature = "aws")]
    pub aws_credentials: Option<AwsCredentialProvider>,
    pub object_store_wrapper: Option<Arc<dyn WrappingObjectStore>>,
    /// Unified storage options accessor with caching and automatic refresh
    ///
    /// Provides storage options and optionally a dynamic provider for automatic
    /// credential refresh. Use `StorageOptionsAccessor::with_static_options()` for static
    /// options or `StorageOptionsAccessor::with_initial_and_provider()` for dynamic refresh.
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
        #[allow(deprecated)]
        Self {
            object_store: None,
            block_size: None,
            s3_credentials_refresh_offset: Duration::from_secs(60),
            #[cfg(feature = "aws")]
            aws_credentials: None,
            object_store_wrapper: None,
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

    pub(crate) fn scheduler_error_mode(&self) -> Result<SchedulerErrorMode> {
        parse_scheduler_error_mode(self.storage_options())
    }

    /// Resolve these params for a single base path scope.
    ///
    /// Storage options may carry base-scoped entries (`base_<id>.<key>`) that
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
) -> Result<SchedulerErrorMode> {
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

fn wrapper_allocation_ptr(wrapper: &Arc<dyn WrappingObjectStore>) -> *const () {
    // Trait object pointers include vtable metadata, which is not stable across codegen units.
    // Cache identity must follow the Arc allocation instead.
    Arc::as_ptr(wrapper) as *const ()
}

// We implement hash for caching
impl std::hash::Hash for ObjectStoreParams {
    #[allow(deprecated)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // For hashing, we use pointer values for ObjectStore, S3 credentials, wrapper
        self.block_size.hash(state);
        if let Some((store, url)) = &self.object_store {
            Arc::as_ptr(store).hash(state);
            url.hash(state);
        }
        self.s3_credentials_refresh_offset.hash(state);
        #[cfg(feature = "aws")]
        if let Some(aws_credentials) = &self.aws_credentials {
            Arc::as_ptr(aws_credentials).hash(state);
        }
        if let Some(wrapper) = &self.object_store_wrapper {
            wrapper_allocation_ptr(wrapper).hash(state);
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
    #[allow(deprecated)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(feature = "aws")]
        if self.aws_credentials.is_some() != other.aws_credentials.is_some() {
            return false;
        }

        // For equality, we use pointer comparison for ObjectStore, S3 credentials, wrapper
        // For accessor, we use accessor_id() for semantic equality
        self.block_size == other.block_size
            && self
                .object_store
                .as_ref()
                .map(|(store, url)| (Arc::as_ptr(store), url))
                == other
                    .object_store
                    .as_ref()
                    .map(|(store, url)| (Arc::as_ptr(store), url))
            && self.s3_credentials_refresh_offset == other.s3_credentials_refresh_offset
            && self
                .object_store_wrapper
                .as_ref()
                .map(wrapper_allocation_ptr)
                == other
                    .object_store_wrapper
                    .as_ref()
                    .map(wrapper_allocation_ptr)
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
pub fn uri_to_url(uri: &str) -> Result<Url> {
    match Url::parse(uri) {
        Ok(url) if url.scheme().len() == 1 && cfg!(windows) => {
            // On Windows, the drive is parsed as a scheme
            local_path_to_url(uri)
        }
        Ok(url) => Ok(url),
        Err(_) => local_path_to_url(uri),
    }
}

fn expand_path(str_path: impl AsRef<str>) -> Result<std::path::PathBuf> {
    let str_path = str_path.as_ref();
    let expanded = expand_tilde_path(str_path).unwrap_or_else(|| str_path.into());

    let mut expanded_path = path_abs::PathAbs::new(expanded)
        .unwrap()
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

fn local_path_to_url(str_path: &str) -> Result<Url> {
    let expanded_path = expand_path(str_path)?;

    Url::from_directory_path(expanded_path).map_err(|_| {
        Error::invalid_input_source(format!("Invalid table location: '{}'", str_path).into())
    })
}

#[cfg(feature = "huggingface")]
fn parse_hf_repo_id(url: &Url) -> Result<String> {
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
        return Err(Error::invalid_input(
            "Huggingface URL must contain at least owner and repo",
        ));
    }

    let repo_type_candidates = ["models", "datasets", "spaces"];
    let (owner, repo_with_rev) = if repo_type_candidates.contains(&segments[0].as_str()) {
        if segments.len() < 3 {
            return Err(Error::invalid_input(
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
    pub async fn from_uri(uri: &str) -> Result<(Arc<Self>, Path)> {
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
    ) -> Result<(Arc<Self>, Path)> {
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
    ) -> Result<(Arc<Self>, Path)> {
        Self::from_uri_and_params_impl(registry, uri, params, false).await
    }

    async fn from_uri_and_params_impl(
        registry: Arc<ObjectStoreRegistry>,
        uri: &str,
        params: &ObjectStoreParams,
        use_registry_cache: bool,
    ) -> Result<(Arc<Self>, Path)> {
        #[allow(deprecated)]
        if let Some((store, path)) = params.object_store.as_ref() {
            let mut inner = store.clone();
            let store_prefix =
                registry.calculate_object_store_prefix(uri, params.storage_options())?;

            let mut io_tracker = IOTracker::default();
            meter_store(&mut inner, &mut io_tracker, &store_prefix);

            if let Some(wrapper) = params.object_store_wrapper.as_ref() {
                inner = wrapper.wrap(&store_prefix, inner);
            }

            // Always wrap with IO tracking
            let tracked_store = io_tracker.wrap("", inner);

            let store = Self {
                inner: tracked_store,
                local_dir_operations: None,
                scheme: path.scheme().to_string(),
                block_size: params.block_size.unwrap_or(64 * 1024),
                max_iop_size: *DEFAULT_MAX_IOP_SIZE,
                use_constant_size_upload_parts: params.use_constant_size_upload_parts,
                list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or_default(),
                io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
                download_retry_count: DEFAULT_DOWNLOAD_RETRY_COUNT,
                scheduler_error_mode: params.scheduler_error_mode()?,
                io_tracker,
                store_prefix,
                // Type-erased on the way in, so there is no telling if it can paginate.
                paginated_lister: None,
            };
            let path = Path::parse(path.path())?;
            return Ok((Arc::new(store), path));
        }
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
    pub fn extract_path_from_uri(registry: Arc<ObjectStoreRegistry>, uri: &str) -> Result<Path> {
        let url = uri_to_url(uri)?;
        let provider = registry
            .get_provider(url.scheme())
            .ok_or_else(|| Error::invalid_input(format!("Unknown scheme: {}", url.scheme())))?;
        provider.extract_path(&url)
    }

    #[deprecated(note = "Use `from_uri` instead")]
    pub fn from_path(str_path: &str) -> Result<(Arc<Self>, Path)> {
        Self::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            str_path,
            &Default::default(),
        )
        .now_or_never()
        .unwrap()
    }

    /// Local object store.
    pub fn local() -> Self {
        let provider = FileStoreProvider;
        provider
            .new_store(Url::parse("file:///").unwrap(), &Default::default())
            .now_or_never()
            .unwrap()
            .unwrap()
    }

    /// Create a in-memory object store directly for testing.
    pub fn memory() -> Self {
        let provider = MemoryStoreProvider;
        provider
            .new_store(Url::parse("memory:///").unwrap(), &Default::default())
            .now_or_never()
            .unwrap()
            .unwrap()
    }

    /// Returns true if the object store pointed to a local file system.
    pub fn is_local(&self) -> bool {
        self.scheme == "file" || self.scheme == "file+uring"
    }

    /// Returns true when object paths directly encode absolute local filesystem paths.
    ///
    /// Local stores rooted below the filesystem root, such as UNC-backed stores, use
    /// their inner object-store implementation instead of direct filesystem access.
    pub fn has_direct_local_paths(&self) -> bool {
        self.is_local() && self.store_prefix == self.scheme
    }

    pub fn is_cloud(&self) -> bool {
        if self.is_local() || self.scheme == "memory" || self.scheme == "shared-memory" {
            return false;
        }
        true
    }

    /// Whether this object store prefers the lite scheduler.
    ///
    /// The lite scheduler is designed for backends like io_uring where
    /// tasks should only be polled when the consumer polls them.
    pub fn prefers_lite_scheduler(&self) -> bool {
        self.scheme == "file+uring"
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn block_size(&self) -> usize {
        self.block_size
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

    /// Apply a [`WrappingObjectStore`] to both `inner` and `paginated_lister` together.
    ///
    /// Keeps both halves in sync: a wrapper returning `None` from
    /// [`WrappingObjectStore::wrap_paginated`] clears the lister so that
    /// [`Self::read_dir_page`] falls back through the (already-wrapped) `inner`.
    pub fn apply_wrapper(&mut self, wrapper: &dyn WrappingObjectStore) {
        self.inner = wrapper.wrap(&self.store_prefix, self.inner.clone());
        self.paginated_lister = self
            .paginated_lister
            .take()
            .and_then(|lister| wrapper.wrap_paginated(&self.store_prefix, lister));
    }

    /// Open a file for path.
    ///
    /// Parameters
    /// - ``path``: Absolute path to the file.
    pub async fn open(&self, path: &Path) -> Result<Box<dyn Reader>> {
        match self.scheme.as_str() {
            "file" if self.has_direct_local_paths() => {
                LocalObjectReader::open_with_tracker(
                    path,
                    self.block_size,
                    None,
                    Arc::new(self.io_tracker.clone()),
                )
                .await
            }
            #[cfg(target_os = "linux")]
            "file+uring" => {
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
    pub async fn open_with_size(&self, path: &Path, known_size: usize) -> Result<Box<dyn Reader>> {
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

        match self.scheme.as_str() {
            "file" if self.has_direct_local_paths() => {
                LocalObjectReader::open_with_tracker(
                    path,
                    self.block_size,
                    Some(known_size),
                    Arc::new(self.io_tracker.clone()),
                )
                .await
            }
            #[cfg(target_os = "linux")]
            "file+uring" => {
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

    /// Create an [ObjectWriter] from local [std::path::Path]
    pub async fn create_local_writer(path: &std::path::Path) -> Result<ObjectWriter> {
        let object_store = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        ObjectWriter::new(&object_store, &os_path).await
    }

    /// Open an [Reader] from local [std::path::Path]
    pub async fn open_local(path: &std::path::Path) -> Result<Box<dyn Reader>> {
        let object_store = Self::local();
        let absolute_path = expand_path(path.to_string_lossy())?;
        let os_path = Path::from_absolute_path(absolute_path)?;
        object_store.open(&os_path).await
    }

    /// Create a new file.
    pub async fn create(&self, path: &Path) -> Result<Box<dyn Writer>> {
        match self.scheme.as_str() {
            "file" if self.has_direct_local_paths() => {
                let local_path = super::local::to_local_path(path);
                let local_path = std::path::PathBuf::from(&local_path);
                if let Some(parent) = local_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                let parent = local_path
                    .parent()
                    .expect("file path must have parent")
                    .to_owned();
                let named_temp = tokio::task::spawn_blocking(move || {
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
                .map_err(|e| Error::io(format!("spawn_blocking failed: {}", e)))??;
                let (std_file, temp_path) = named_temp.into_parts();
                let file = tokio::fs::File::from_std(std_file);
                Ok(Box::new(LocalWriter::new(
                    file,
                    path.clone(),
                    temp_path,
                    Arc::new(self.io_tracker.clone()),
                )))
            }
            _ => Ok(Box::new(ObjectWriter::new(self, path).await?)),
        }
    }

    /// A helper function to create a file and write content to it.
    pub async fn put(&self, path: &Path, content: &[u8]) -> Result<WriteResult> {
        let mut writer = self.create(path).await?;
        writer.write_all(content).await?;
        Writer::shutdown(writer.as_mut()).await
    }

    /// Atomically creates an object without replacing an existing object.
    ///
    /// Local stores publish a uniquely named staging object with a conditional
    /// rename. Other stores use their conditional create operation. Tencent COS
    /// is rejected because it can silently ignore conditional create requests.
    ///
    /// Returns [`object_store::Error::NotSupported`] without writing when the
    /// backend cannot reliably provide put-if-absent semantics.
    pub async fn put_if_absent(
        &self,
        path: &Path,
        content: PutPayload,
    ) -> object_store::Result<()> {
        if self.scheme == "cos" {
            return Err(object_store::Error::NotSupported {
                source: "Tencent COS does not reliably enforce put-if-absent after bucket \
                         versioning has ever been enabled"
                    .into(),
            });
        }

        if self.is_local() {
            let staging_path =
                Path::from(format!("{}.tmp.{}", path, uuid::Uuid::new_v4().simple()));
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
            result
        } else {
            self.inner
                .put_opts(
                    path,
                    content,
                    PutOptions {
                        mode: PutMode::Create,
                        ..Default::default()
                    },
                )
                .await
                .map(|_| ())
        }
    }

    pub async fn delete(&self, path: &Path) -> Result<()> {
        self.inner.delete(path).await?;
        Ok(())
    }

    /// AWS S3 and GCS reject a single-shot server-side copy whose source is
    /// larger than this; such sources are streamed through a multipart write.
    const MAX_SINGLE_COPY_BYTES: u64 = 5 * 1024 * 1024 * 1024; // 5 GiB

    pub async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        // S3 and GCS cap single-shot server-side copies at 5 GiB and object_store
        // does not fall back to a multipart copy for larger sources
        // (https://github.com/apache/arrow-rs-object-store/issues/563). Azure and
        // other blob stores don't have this limit, so we only pay for the fallback
        // (an extra size lookup) on S3 and GCS.
        let multipart_copy_fallback = matches!(self.scheme.as_str(), "s3" | "s3+ddb" | "gs");
        self.copy_impl(
            from,
            to,
            multipart_copy_fallback,
            Self::MAX_SINGLE_COPY_BYTES,
        )
        .await
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
    /// # use lance_io::object_store::ObjectStore;
    /// # use object_store::path::Path;
    /// # async fn copy(source: &ObjectStore, destination: &ObjectStore) -> Result<()> {
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
    ) -> Result<WriteResult> {
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
    ) -> Result<WriteResult> {
        if !server_side_copy_enabled || !self.can_server_side_copy_to(destination_store) {
            return self
                .copy_via_stream(source_path, destination_store, destination_path)
                .await;
        }

        let source_size = self.size(source_path).await?;
        let result_size = usize::try_from(source_size).map_err(|source| {
            Error::io(format!(
                "server-side copy source size conversion failed from {source_path} to \
                 {destination_path}: source_size={source_size}, error={source}"
            ))
        })?;
        destination_store
            .copy(source_path, destination_path)
            .await?;
        let destination_size = destination_store.size(destination_path).await?;
        if destination_size != source_size {
            return Err(Error::io(format!(
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
    ///
    /// ```no_run
    /// # use lance_core::Result;
    /// # use lance_io::object_store::ObjectStore;
    /// # use object_store::path::Path;
    /// # async fn copy(source: &ObjectStore, destination: &ObjectStore) -> Result<()> {
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
    ) -> Result<WriteResult> {
        let started_at = Instant::now();
        if self.has_direct_local_paths() && destination_store.has_direct_local_paths() {
            let source_size = std::fs::metadata(super::local::to_local_path(source_path))
                .map_err(|source| {
                    let source = if source.kind() == std::io::ErrorKind::NotFound {
                        Error::not_found(source_path.to_string())
                    } else {
                        Error::from(source)
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
                return Err(Error::io(format!(
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
                    return Err(Error::io(format!(
                        "multipart_stream_copy source range size mismatch from {source_path} to \
                         {destination_path}: range={current_range:?}, \
                         expected_bytes={expected_bytes}, actual_bytes={}",
                        current_bytes.len()
                    )));
                }
                bytes_transferred = bytes_transferred
                    .checked_add(current_bytes.len())
                    .ok_or_else(|| {
                        Error::io(format!(
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
            return Err(Error::io(format!(
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
            return Err(Error::io(format!(
                "multipart_stream_copy destination size mismatch from {source_path} to \
                 {destination_path}: source_size={source_size}, \
                 destination_size={destination_size}"
            )));
        }

        Span::current().record("validation", "passed");
        Span::current().record("elapsed_ms", started_at.elapsed().as_millis() as u64);
        Ok(write_result)
    }

    /// Copy `from` to `to`. When `multipart_copy_fallback` is set, a source
    /// larger than `max_single_copy` is streamed through a multipart write
    /// instead of a single-shot server-side copy. Both are parameters so tests
    /// can drive the streaming path without a multi-gigabyte fixture or an S3
    /// endpoint.
    async fn copy_impl(
        &self,
        from: &Path,
        to: &Path,
        multipart_copy_fallback: bool,
        max_single_copy: u64,
    ) -> Result<()> {
        if self.has_direct_local_paths() {
            // Use std::fs::copy for local filesystem to support cross-filesystem copies
            let metrics = self.io_tracker.begin_io("copy");
            let result = super::local::copy_file(from, to);
            metrics.record(&result, 0);
            return result;
        }
        if multipart_copy_fallback {
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

    /// Read a directory (start from base directory) and returns all sub-paths in the directory.
    ///
    /// This enumerates the whole prefix before it returns, however many children it holds.
    /// Use [`Self::read_dir_page`] to page through a directory instead.
    pub async fn read_dir(&self, dir_path: impl Into<Path>) -> Result<Vec<String>> {
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
    pub async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        Ok(self.inner.list_with_delimiter(prefix).await?)
    }

    pub fn list(
        &self,
        path: Option<Path>,
    ) -> Pin<Box<dyn Stream<Item = Result<ObjectMeta>> + Send>> {
        Box::pin(ListRetryStream::new(self.inner.clone(), path, 5).map(|m| m.map_err(|e| e.into())))
    }

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    pub fn read_dir_all<'a, 'b>(
        &'a self,
        dir_path: impl Into<&'b Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> BoxStream<'a, Result<ObjectMeta>> {
        self.inner.read_dir_all(dir_path, unmodified_since)
    }

    /// Remove a directory recursively.
    pub async fn remove_dir_all(&self, dir_path: impl Into<Path>) -> Result<()> {
        let path = dir_path.into();
        let path = Path::parse(&path)?;

        if let Some(local_dir_operations) = &self.local_dir_operations {
            let metrics = self.io_tracker.begin_io("delete");
            let result = local_dir_operations.remove_dir_all(&path).await;
            metrics.record(&result, 0);
            return result;
        }
        if self.has_direct_local_paths() {
            // The local file system provider needs to delete both files and directories.
            // Counted as a single delete request, matching how `delete_stream`
            // counts one batched request regardless of how many paths it removes.
            let metrics = self.io_tracker.begin_io("delete");
            let result = super::local::remove_dir_all(&path);
            metrics.record(&result, 0);
            return result;
        }
        let sub_entries = self
            .inner
            .list(Some(&path))
            .map(|m| m.map(|meta| meta.location))
            .boxed();
        self.inner
            .delete_stream(sub_entries)
            .try_collect::<Vec<_>>()
            .await?;
        if self.scheme == "file-object-store" {
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
    /// # use lance_io::object_store::ObjectStore;
    /// # async fn remove_stale_index_dirs(store: &ObjectStore) -> Result<()> {
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
    ) -> Result<()> {
        if !self.has_direct_local_paths() && self.scheme != "file-object-store" {
            return Ok(());
        }

        let path = Path::parse(root_path.into())?;
        let metrics = self.io_tracker.begin_io("delete");
        let result = tokio::task::spawn_blocking(move || {
            super::local::remove_empty_dirs(&path, &retained_dirs, &verified_dirs, unmodified_since)
        })
        .await
        .map_err(|error| Error::io(format!("empty-directory cleanup task failed: {error}")))?;
        metrics.record(&result, 0);
        result
    }

    pub fn remove_stream<'a>(
        &'a self,
        locations: BoxStream<'a, Result<Path>>,
    ) -> BoxStream<'a, Result<Path>> {
        let store = Arc::clone(&self.inner);
        locations
            .and_then(move |location| {
                let store = Arc::clone(&store);
                async move {
                    store.delete(&location).await?;
                    Ok(location)
                }
            })
            .boxed()
    }

    /// Check a file exists.
    pub async fn exists(&self, path: &Path) -> Result<bool> {
        match self.inner.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Get file size.
    pub async fn size(&self, path: &Path) -> Result<u64> {
        Ok(self.inner.head(path).await?.size)
    }

    /// Convenience function to open a reader and read all the bytes
    pub async fn read_one_all(&self, path: &Path) -> Result<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_all().await?)
    }

    /// Convenience function open a reader and make a single request
    ///
    /// If you will be making multiple requests to the path it is more efficient to call [`Self::open`]
    /// and then call [`Reader::get_range`] multiple times.
    pub async fn read_one_range(&self, path: &Path, range: Range<usize>) -> Result<Bytes> {
        let reader = self.open(path).await?;
        Ok(reader.get_range(range).await?)
    }
}

/// Options that can be set for multiple object stores
#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy)]
pub enum LanceConfigKey {
    /// Number of times to retry a download that fails
    DownloadRetryCount,
}

impl FromStr for LanceConfigKey {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "download_retry_count" => Ok(Self::DownloadRetryCount),
            _ => Err(Error::invalid_input_source(
                format!("Invalid LanceConfigKey: {}", s).into(),
            )),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StorageOptions(pub HashMap<String, String>);

impl StorageOptions {
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
    pub fn client_options(&self) -> Result<ClientOptions> {
        let mut headers = HeaderMap::new();
        for (key, value) in &self.0 {
            if let Some(header_name) = key.strip_prefix("headers.") {
                let name = header_name
                    .parse::<http::header::HeaderName>()
                    .map_err(|e| {
                        Error::invalid_input(format!("invalid header name '{header_name}': {e}"))
                    })?;
                let val = HeaderValue::from_str(value).map_err(|e| {
                    Error::invalid_input(format!("invalid header value for '{header_name}': {e}"))
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

static DEFAULT_OBJECT_STORE_REGISTRY: std::sync::LazyLock<ObjectStoreRegistry> =
    std::sync::LazyLock::new(ObjectStoreRegistry::default);

impl ObjectStore {
    /// Wrap an object store implementation with Lance I/O behavior.
    ///
    /// `storage_options` may contain `scheduler_error_mode`. Valid values are
    /// `best_effort` and `fail_fast`. Because this legacy constructor is
    /// infallible, an invalid value is logged and falls back to `best_effort`.
    /// Use [`Self::try_new`] when invalid storage options must be rejected.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<DynObjectStore>,
        location: Url,
        block_size: Option<usize>,
        wrapper: Option<Arc<dyn WrappingObjectStore>>,
        use_constant_size_upload_parts: bool,
        list_is_lexically_ordered: bool,
        io_parallelism: usize,
        download_retry_count: usize,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Self {
        let scheduler_error_mode = match parse_scheduler_error_mode(storage_options) {
            Ok(mode) => mode,
            Err(error) => {
                log::warn!(
                    "{error}; ObjectStore::new is falling back to best_effort. Use ObjectStore::try_new to reject invalid storage options"
                );
                SchedulerErrorMode::BestEffort
            }
        };
        let store_prefix = Self::direct_store_prefix(&location, storage_options).unwrap();
        Self::new_with_scheduler_error_mode(
            store,
            location,
            block_size,
            wrapper,
            use_constant_size_upload_parts,
            list_is_lexically_ordered,
            io_parallelism,
            download_retry_count,
            scheduler_error_mode,
            store_prefix,
        )
    }

    /// Wrap an object store implementation and validate Lance storage options.
    ///
    /// This is the fallible counterpart to [`Self::new`]. Custom object-store
    /// providers may use it to reject an invalid `scheduler_error_mode` before
    /// returning a store. A store returned through [`ObjectStoreRegistry`] is
    /// validated by the registry independently of which constructor its provider
    /// uses.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::{collections::HashMap, sync::Arc};
    ///
    /// use lance_io::object_store::ObjectStore;
    /// use object_store::memory::InMemory;
    /// use url::Url;
    ///
    /// # fn build_store() -> lance_core::Result<()> {
    /// let storage_options = HashMap::from([(
    ///     "scheduler_error_mode".to_string(),
    ///     "fail_fast".to_string(),
    /// )]);
    /// let _store = ObjectStore::try_new(
    ///     Arc::new(InMemory::new()),
    ///     Url::parse("memory:///").expect("valid memory URL"),
    ///     None,
    ///     None,
    ///     false,
    ///     true,
    ///     64,
    ///     3,
    ///     Some(&storage_options),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        store: Arc<DynObjectStore>,
        location: Url,
        block_size: Option<usize>,
        wrapper: Option<Arc<dyn WrappingObjectStore>>,
        use_constant_size_upload_parts: bool,
        list_is_lexically_ordered: bool,
        io_parallelism: usize,
        download_retry_count: usize,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let scheduler_error_mode = parse_scheduler_error_mode(storage_options)?;
        let store_prefix = Self::direct_store_prefix(&location, storage_options)?;
        Ok(Self::new_with_scheduler_error_mode(
            store,
            location,
            block_size,
            wrapper,
            use_constant_size_upload_parts,
            list_is_lexically_ordered,
            io_parallelism,
            download_retry_count,
            scheduler_error_mode,
            store_prefix,
        ))
    }

    fn direct_store_prefix(
        location: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        match DEFAULT_OBJECT_STORE_REGISTRY.get_provider(location.scheme()) {
            Some(provider) => provider.calculate_object_store_prefix(location, storage_options),
            None => {
                let store_prefix = format!("{}${}", location.scheme(), location.authority());
                log::warn!(
                    "Guessing that object store prefix is {}, since object store scheme is not found in registry.",
                    store_prefix
                );
                Ok(store_prefix)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_scheduler_error_mode(
        mut store: Arc<DynObjectStore>,
        location: Url,
        block_size: Option<usize>,
        wrapper: Option<Arc<dyn WrappingObjectStore>>,
        use_constant_size_upload_parts: bool,
        list_is_lexically_ordered: bool,
        io_parallelism: usize,
        download_retry_count: usize,
        scheduler_error_mode: SchedulerErrorMode,
        store_prefix: String,
    ) -> Self {
        let scheme = location.scheme();
        let block_size = block_size.unwrap_or_else(|| infer_block_size(scheme));
        let mut io_tracker = IOTracker::default();
        meter_store(&mut store, &mut io_tracker, &store_prefix);

        let store = match wrapper {
            Some(wrapper) => wrapper.wrap(&store_prefix, store),
            None => store,
        };

        // Always wrap with IO tracking
        let tracked_store = io_tracker.wrap("", store);

        Self {
            inner: tracked_store,
            local_dir_operations: None,
            scheme: scheme.into(),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts,
            list_is_lexically_ordered,
            io_parallelism,
            download_retry_count,
            scheduler_error_mode,
            io_tracker,
            store_prefix,
            // Type-erased on the way in, so there is no telling if it can paginate.
            paginated_lister: None,
        }
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
fn meter_store(inner: &mut Arc<dyn OSObjectStore>, io_tracker: &mut IOTracker, store_prefix: &str) {
    use crate::object_store::metrics::ObjectStoreMetricsExt;
    io_tracker.set_metrics_base(store_prefix);
    *inner = inner.clone().metered(store_prefix.to_owned());
}

#[cfg(not(feature = "metrics"))]
fn meter_store(
    _inner: &mut Arc<dyn OSObjectStore>,
    _io_tracker: &mut IOTracker,
    _store_prefix: &str,
) {
}

fn infer_block_size(scheme: &str) -> usize {
    // Block size: On local file systems, we use 4KB block size. On cloud
    // object stores, we use 64KB block size. This is generally the largest
    // block size where we don't see a latency penalty.
    match scheme {
        "file" => 4 * 1024,
        _ => 64 * 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use bytes::Bytes;
    use lance_core::utils::tempfile::{TempStdDir, TempStdFile, TempStrDir};
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, PutMultipartOptions,
        PutOptions, PutPayload, PutResult, Result as OSResult, UploadPart,
    };
    use rstest::rstest;
    use serial_test::serial;
    use std::env::set_current_dir;
    use std::fmt::{Display, Formatter};
    use std::fs::{create_dir_all, write};
    use std::ops::Range;
    use std::path::Path as StdPath;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Write test content to file.
    fn write_to_file(path_str: &str, contents: &str) -> std::io::Result<()> {
        let path = expand_path(path_str).map_err(std::io::Error::other)?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        write(path, contents)
    }

    async fn read_from_store(store: &ObjectStore, path: &Path) -> Result<String> {
        let test_file_store = store.open(path).await.unwrap();
        let size = test_file_store.size().await.unwrap();
        let bytes = test_file_store.get_range(0..size).await.unwrap();
        let contents = String::from_utf8(bytes.to_vec()).unwrap();
        Ok(contents)
    }

    #[tokio::test]
    async fn test_put_if_absent() {
        let temp_dir = TempStrDir::default();
        let path = Path::from(format!("{}/atomic-create", temp_dir.as_str()));
        let store = ObjectStore::local();
        store
            .put_if_absent(&path, Bytes::from_static(b"first").into())
            .await
            .unwrap();
        let error = store
            .put_if_absent(&path, Bytes::from_static(b"second").into())
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            object_store::Error::AlreadyExists { .. } | object_store::Error::Precondition { .. }
        ));
        assert_eq!(
            store.read_one_all(&path).await.unwrap(),
            b"first".as_slice()
        );
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
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(
            "Invalid scheduler_error_mode value 'invalid'. Valid values are: best_effort, fail_fast"
        ));
    }

    #[test]
    fn test_try_new_rejects_invalid_scheduler_error_mode() {
        let storage_options = HashMap::from([(
            SCHEDULER_ERROR_MODE_STORAGE_OPTION.to_string(),
            "invalid".to_string(),
        )]);

        let error = ObjectStore::try_new(
            Arc::new(InMemory::new()),
            Url::parse("memory:///").unwrap(),
            None,
            None,
            false,
            true,
            DEFAULT_CLOUD_IO_PARALLELISM,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            Some(&storage_options),
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(
            "Invalid scheduler_error_mode value 'invalid'. Valid values are: best_effort, fail_fast"
        ));
    }

    #[tokio::test]
    async fn test_put_if_absent_rejects_cos() {
        let mut store = ObjectStore::memory();
        store.scheme = "cos".to_string();
        let path = Path::from("atomic-create");

        let error = store
            .put_if_absent(&path, Bytes::from_static(b"value").into())
            .await
            .unwrap_err();

        assert!(matches!(error, object_store::Error::NotSupported { .. }));
        assert!(!store.exists(&path).await.unwrap());
    }

    #[tokio::test]
    async fn test_io_parallelism_clamped_to_nonzero() {
        // `io_parallelism()` feeds `buffered`/`buffer_unordered` windows; a value of 0 makes those
        // streams never poll, hanging callers (e.g. a metadata-only `count_rows`). It must clamp.
        let store = ObjectStore::local();
        // Readers opened by the store must advertise the store's normalized
        // effective parallelism, not the hardcoded cloud default.
        let mem_store = ObjectStore::memory();
        let path = Path::from("/io_parallelism_probe");
        mem_store.put(&path, b"x").await.unwrap();

        // SAFETY: process-global env var, set and restored within this test. `io_parallelism()`
        // only reads it, and a concurrent reader observes a valid clamped value, never 0.
        unsafe { std::env::set_var("LANCE_IO_THREADS", "0") };
        assert_eq!(
            store.io_parallelism(),
            1,
            "LANCE_IO_THREADS=0 must clamp to 1"
        );
        assert_eq!(
            mem_store.open(&path).await.unwrap().io_parallelism(),
            1,
            "an opened reader must report the store's clamped parallelism"
        );

        unsafe { std::env::set_var("LANCE_IO_THREADS", "8") };
        assert_eq!(
            store.io_parallelism(),
            8,
            "a positive override must pass through unchanged"
        );
        assert_eq!(
            mem_store.open(&path).await.unwrap().io_parallelism(),
            8,
            "an opened reader must honor the configured request limit"
        );
        assert_eq!(
            mem_store
                .open_with_size(&path, 1024 * 1024)
                .await
                .unwrap()
                .io_parallelism(),
            8,
            "a sized reader must honor the configured request limit"
        );

        unsafe { std::env::remove_var("LANCE_IO_THREADS") };
        assert!(
            store.io_parallelism() >= 1,
            "the configured default parallelism must be at least 1"
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

        set_current_dir(StdPath::new(tmp_path.as_ref())).expect("Error changing current dir");
        let (store, path) = ObjectStore::from_uri("./bar/foo.lance").await.unwrap();

        let contents = read_from_store(store.as_ref(), &path.clone().join("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "RELATIVE_URL");
    }

    #[tokio::test]
    async fn test_tilde_expansion() {
        let uri = "~/foo.lance";
        write_to_file(&format!("{uri}/test_file"), "TILDE").unwrap();
        let (store, path) = ObjectStore::from_uri(uri).await.unwrap();
        let contents = read_from_store(store.as_ref(), &path.clone().join("test_file"))
            .await
            .unwrap();
        assert_eq!(contents, "TILDE");
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

    #[tokio::test]
    async fn test_delete_directory_local_store() {
        test_delete_directory("").await;
    }

    #[tokio::test]
    async fn test_delete_directory_file_object_store() {
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
        called: AtomicBool,

        return_value: Arc<dyn OSObjectStore>,
    }

    impl WrappingObjectStore for TestWrapper {
        fn wrap(
            &self,
            _store_prefix: &str,
            _original: Arc<dyn OSObjectStore>,
        ) -> Arc<dyn OSObjectStore> {
            self.called.store(true, Ordering::Relaxed);

            // return a mocked value so we can check if the final store is the one we expect
            self.return_value.clone()
        }

        // This one swaps the store out entirely, so a listing that went around it would be
        // listing something else.
        fn wrap_paginated(
            &self,
            _store_prefix: &str,
            _original: Arc<dyn PaginatedListStore>,
        ) -> Option<Arc<dyn PaginatedListStore>> {
            None
        }
    }

    impl TestWrapper {
        fn called(&self) -> bool {
            self.called.load(Ordering::Relaxed)
        }
    }

    /// A lister that exists only to be wrapped.
    #[derive(Debug)]
    struct StubLister;

    #[async_trait]
    impl PaginatedListStore for StubLister {
        async fn list_paginated(
            &self,
            _prefix: Option<&str>,
            _opts: object_store::list::PaginatedListOptions,
        ) -> object_store::Result<object_store::list::PaginatedListResult> {
            unimplemented!("this lister exists to be wrapped, not to list")
        }
    }

    /// Records the listers it was handed, and leaves the store alone.
    #[derive(Debug)]
    struct PaginatedTestWrapper {
        name: &'static str,
        log: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl WrappingObjectStore for PaginatedTestWrapper {
        fn wrap(
            &self,
            _store_prefix: &str,
            original: Arc<dyn OSObjectStore>,
        ) -> Arc<dyn OSObjectStore> {
            original
        }

        fn wrap_paginated(
            &self,
            store_prefix: &str,
            original: Arc<dyn PaginatedListStore>,
        ) -> Option<Arc<dyn PaginatedListStore>> {
            self.log
                .lock()
                .unwrap()
                .push(format!("{}@{store_prefix}", self.name));
            Some(original)
        }
    }

    /// A chain hands the lister to each of its wrappers in turn. One wrapper giving up the
    /// pushdown gives it up for the chain, and the wrappers after it are never asked: the
    /// listing is going through `wrap` either way, which is every wrapper at once.
    #[rstest]
    #[case::every_wrapper_keeps_it(false, vec!["first@memory", "second@memory"])]
    #[case::one_wrapper_gives_it_up(true, vec!["first@memory"])]
    fn test_a_chain_wraps_the_lister_until_one_gives_it_up(
        #[case] gives_up: bool,
        #[case] expected_log: Vec<&str>,
    ) {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut wrappers: Vec<Arc<dyn WrappingObjectStore>> =
            vec![Arc::new(PaginatedTestWrapper {
                name: "first",
                log: log.clone(),
            })];
        if gives_up {
            wrappers.push(Arc::new(TestWrapper {
                called: AtomicBool::new(false),
                return_value: Arc::new(InMemory::new()),
            }));
        }
        wrappers.push(Arc::new(PaginatedTestWrapper {
            name: "second",
            log: log.clone(),
        }));

        let wrapped = ChainedWrappingObjectStore::new(wrappers)
            .wrap_paginated("memory", Arc::new(StubLister));

        assert_eq!(wrapped.is_none(), gives_up);
        assert_eq!(*log.lock().unwrap(), expected_log);
    }

    /// `apply_wrapper` keeps both halves of the store in sync. A wrapper that gives up the
    /// pushdown has to clear the lister too, or `read_dir_page` would keep talking to the
    /// backend behind the wrapper's back.
    #[rstest]
    #[case::gives_up_the_pushdown(true)]
    #[case::keeps_the_pushdown(false)]
    fn test_apply_wrapper_keeps_inner_and_the_lister_in_sync(#[case] gives_up: bool) {
        let replacement = Arc::new(InMemory::new());
        let giving_up = TestWrapper {
            called: AtomicBool::new(false),
            return_value: replacement.clone(),
        };
        let keeping = PaginatedTestWrapper {
            name: "passthrough",
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
        };
        let wrapper: &dyn WrappingObjectStore = match gives_up {
            true => &giving_up,
            false => &keeping,
        };

        let mut store = ObjectStore::memory();
        store.paginated_lister = Some(Arc::new(StubLister) as Arc<dyn PaginatedListStore>);
        store.apply_wrapper(wrapper);

        assert_eq!(
            store.paginated_lister.is_some(),
            !gives_up,
            "the lister has to follow what the wrapper said"
        );
        // The wrapper that gives up the pushdown is also the one that swaps the store out, so
        // whether `inner` was replaced says that `wrap` ran on the same wrapper.
        assert_eq!(
            Arc::ptr_eq(&store.inner, &(replacement as Arc<dyn OSObjectStore>)),
            gives_up
        );
    }

    #[tokio::test]
    async fn test_wrapper_identity_is_stable_across_tasks() {
        let wrapper = Arc::new(TestWrapper {
            called: AtomicBool::new(false),
            return_value: Arc::new(InMemory::new()),
        });
        let initial_params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..ObjectStoreParams::default()
        };
        let task_params = tokio::spawn(async move {
            ObjectStoreParams {
                object_store_wrapper: Some(wrapper),
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

    #[tokio::test]
    async fn test_wrapping_object_store_option_is_used() {
        // Make a store for the inner store first
        let mock_inner_store: Arc<dyn OSObjectStore> = Arc::new(InMemory::new());
        let registry = Arc::new(ObjectStoreRegistry::default());

        assert_eq!(Arc::strong_count(&mock_inner_store), 1);

        let wrapper = Arc::new(TestWrapper {
            called: AtomicBool::new(false),
            return_value: mock_inner_store.clone(),
        });

        let params = ObjectStoreParams {
            object_store_wrapper: Some(wrapper.clone()),
            ..ObjectStoreParams::default()
        };

        // not called yet
        assert!(!wrapper.called());

        let _ = ObjectStore::from_uri_and_params(registry, "memory:///", &params)
            .await
            .unwrap();

        // called after construction
        assert!(wrapper.called());

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

        let file_path_os = object_store::path::Path::parse(file_path.to_str().unwrap()).unwrap();
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

        // Use object_store::Path::parse for the destination
        let dest_file = dest_dir.join("copied_file.txt");
        let dest_str = dest_file.to_str().unwrap();
        let to_path = object_store::path::Path::parse(dest_str).unwrap();

        // Perform the copy operation
        store.copy(&from_path, &to_path).await.unwrap();

        // Verify the file was copied correctly
        assert!(dest_file.exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
    }

    #[tokio::test]
    async fn test_copy_creates_parent_directories() {
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

        // Create paths
        let from_path = base_path.clone().join(source_file_name);

        // Create destination with nested directories that don't exist yet
        let dest_file = dest_dir.join("nested").join("dirs").join("copied_file.txt");
        let dest_str = dest_file.to_str().unwrap();
        let to_path = object_store::path::Path::parse(dest_str).unwrap();

        // Perform the copy operation - should create parent directories
        store.copy(&from_path, &to_path).await.unwrap();

        // Verify the file was copied correctly and directories were created
        assert!(dest_file.exists());
        assert!(dest_file.parent().unwrap().exists());
        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, b"test content");
    }

    /// Inner store that forwards everything to `InMemory` except single-shot
    /// server-side copy (`copy_opts`), which always fails. This lets a test
    /// prove that `ObjectStore::copy` fell back to a streaming multipart copy
    /// for an oversized source rather than issuing a single `CopyObject`.
    #[derive(Debug)]
    struct CopyFailingStore {
        inner: InMemory,
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
                    Err(object_store::Error::Generic {
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
        inner: InMemory,
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
    impl OSObjectStore for ObservedMultipartStore {
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
        ) -> OSResult<Box<dyn MultipartUpload>> {
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

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.observations
                .native_copy_count
                .fetch_add(1, Ordering::SeqCst);
            self.inner.copy_opts(from, to, opts).await
        }
    }

    #[async_trait]
    impl OSObjectStore for CopyFailingStore {
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
        ) -> OSResult<Box<dyn MultipartUpload>> {
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
            Err(object_store::Error::Generic {
                store: "CopyFailingStore",
                source: "single-shot copy disabled in test".into(),
            })
        }
    }

    #[tokio::test]
    async fn test_copy_streams_objects_larger_than_threshold() {
        // memory:// is non-local but isn't an S3/GCS scheme, so copy() wouldn't
        // enable the fallback on its own. Drive copy_impl directly with
        // multipart_copy_fallback = true to exercise the streaming path. The
        // inner store rejects any single-shot copy, so a successful copy can only
        // have gone through the streaming branch.
        let mut store = ObjectStore::memory();
        store.inner = Arc::new(CopyFailingStore {
            inner: InMemory::new(),
        });

        let from = Path::from("source.bin");
        let contents = b"streaming multipart copy payload well past the tiny threshold";
        store.put(&from, contents).await.unwrap();

        // Source size (61 bytes) exceeds the threshold -> must stream via a
        // multipart write rather than a single-shot server-side copy.
        let streamed = Path::from("streamed.bin");
        store.copy_impl(&from, &streamed, true, 8).await.unwrap();
        let copied = store.read_one_all(&streamed).await.unwrap();
        assert_eq!(copied.as_ref(), contents.as_slice());

        // Source size below the threshold -> single-shot copy, which the inner
        // store rejects, confirming that the streaming branch (not native copy)
        // is what made the first copy succeed.
        let native = Path::from("native.bin");
        assert!(
            store
                .copy_impl(&from, &native, true, u64::MAX)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_copy_via_stream_never_uses_native_copy() {
        let mut store = ObjectStore::memory();
        store.inner = Arc::new(CopyFailingStore {
            inner: InMemory::new(),
        });

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
    async fn test_bulk_copy_streams_when_server_side_copy_is_disabled() {
        let observations = Arc::new(MultipartObservations::default());
        let mut store = ObjectStore::memory();
        store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });

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

    #[test]
    #[serial(server_side_copy_env)]
    fn test_server_side_copy_environment_policy() {
        let previous_value = std::env::var_os(SERVER_SIDE_COPY_ENABLED_ENV);
        let mut store = ObjectStore::memory();
        store.scheme = "test-cloud".to_string();
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
        let mut source_store = ObjectStore::memory();
        source_store.scheme = "test-cloud".to_string();
        source_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });
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
        let shared_inner = InMemory::new();
        let source_observations = Arc::new(MultipartObservations::default());
        let mut source_store = ObjectStore::memory();
        source_store.scheme = "test-cloud".to_string();
        source_store.store_prefix = "test-cloud$bucket".to_string();
        source_store.inner = Arc::new(ObservedMultipartStore {
            inner: shared_inner.clone(),
            observations: source_observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });
        let destination_observations = Arc::new(MultipartObservations::default());
        let mut destination_store = ObjectStore::memory();
        destination_store.scheme = "test-cloud".to_string();
        destination_store.store_prefix = "test-cloud$bucket".to_string();
        destination_store.inner = Arc::new(ObservedMultipartStore {
            inner: shared_inner,
            observations: destination_observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });

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
        let mut source_store = ObjectStore::memory();
        source_store.scheme = "test-cloud".to_string();
        source_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 1,
        });
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
        destination_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });

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
        destination_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: false,
            destination_size_adjustment: 0,
        });

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
        destination_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: observations.clone(),
            fail_parts: true,
            destination_size_adjustment: 0,
        });

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
        destination_store.inner = Arc::new(ObservedMultipartStore {
            inner: InMemory::new(),
            observations: Arc::new(MultipartObservations::default()),
            fail_parts: false,
            destination_size_adjustment: 1,
        });

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
