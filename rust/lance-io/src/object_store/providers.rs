// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, RwLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use lance_core::storage::path::Path;
use url::Url;

use crate::object_store::uri_to_url;
use crate::scheduler::SchedulerErrorMode;

use super::{ObjectStore, ObjectStoreParams, StorageOptions, tracing::StorageBackendTracingExt};
use lance_core::error::{Error, LanceOptionExt, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
#[cfg(feature = "goosefs")]
pub mod goosefs;
#[cfg(feature = "huggingface")]
pub mod huggingface;
pub mod local;
pub mod memory;
#[cfg(any(
    feature = "aws",
    feature = "azure",
    feature = "gcp",
    feature = "goosefs",
    feature = "huggingface",
    feature = "oss",
    feature = "tencent",
    feature = "tos"
))]
pub(in crate::object_store) mod opendal;
#[cfg(feature = "oss")]
pub mod oss;
pub mod shared_memory;
#[cfg(feature = "tencent")]
pub mod tencent;
#[cfg(feature = "tos")]
pub mod tos;

/// Physical backend and its IO capabilities, returned by a [`ObjectStoreProvider`].
///
/// The registry applies request policies, wrappers and instrumentation exactly once.
/// Construct with [`Self::new`] and override only capabilities the backend provides.
#[derive(Clone)]
pub struct BackendConfig {
    // Keep native capabilities tied to the backend that the provider constructed.
    pub(super) backend: Arc<dyn super::StorageBackend>,
    /// Physical storage scheme, including aliases resolved by the provider.
    pub scheme: String,
    /// Provider-resolved object identity for wrappers and the configured handle.
    /// None retains the URI-derived prefix. The registry cache key is unchanged.
    pub store_prefix: Option<String>,
    /// Preferred read block size in bytes; explicit caller configuration takes precedence.
    pub block_size: usize,
    /// Maximum suggested number of concurrent IO operations.
    pub io_parallelism: usize,
    /// Whether multipart uploads require equal-sized non-final parts.
    pub use_constant_size_upload_parts: bool,
    /// Whether listing is lexically ordered by object path.
    pub list_is_lexically_ordered: bool,
    pub(super) local_dir_operations: Option<Arc<dyn super::LocalDirOperations>>,
    pub(super) local_fsync: bool,
    pub(super) direct_local_io: bool,
    pub(super) local_io: super::LocalIo,
    /// Whether paths have local directory semantics, including possible aliases.
    /// Enables conservative cleanup when physical paths cannot be resolved.
    /// This does not grant access to the host filesystem.
    pub is_local: bool,
    /// Whether callers should apply cloud IO policies, including scheduling and bulk copy.
    pub is_remote: bool,
    /// Largest server-side copy supported in one request; None means no known limit.
    pub max_single_copy_size: Option<u64>,
}

impl BackendConfig {
    /// Describe a backend with conservative capabilities and a 64 KiB read block.
    /// Native filesystem access is reserved for the built-in file provider.
    pub fn new(backend: Arc<dyn super::StorageBackend>, scheme: impl Into<String>) -> Self {
        let scheme = scheme.into();
        Self {
            backend,
            store_prefix: None,
            block_size: 64 * 1024,
            scheme,
            io_parallelism: super::DEFAULT_CLOUD_IO_PARALLELISM,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,

            local_dir_operations: None,
            local_fsync: false,
            direct_local_io: false,
            local_io: super::LocalIo::Backend,
            is_local: false,
            is_remote: true,
            max_single_copy_size: None,
        }
    }
}

impl std::fmt::Debug for BackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendConfig")
            .field("backend", &self.backend)
            .field("scheme", &self.scheme)
            .finish_non_exhaustive()
    }
}

/// Resolve a location and its configuration into a Lance I/O binding.
///
/// ObjectStore integrations own authorization, credential refresh, and cache identity.
/// Return a [`BackendConfig`] backed by a [`super::StorageBackend`];
/// reuse its handles rather than rebuilding a backend from the URI. Native
/// implementations can be adapted with [`super::backend::native::from_object_store`].
///
/// ```
/// use std::sync::Arc;
/// use async_trait::async_trait;
/// use lance_io::object_store::{BackendConfig, ObjectStore, StorageBackend,
///     ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry};
/// use url::Url;
///
/// #[derive(Debug)]
/// struct CustomProvider(Arc<dyn StorageBackend>);
///
/// #[async_trait]
/// impl ObjectStoreProvider for CustomProvider {
///     async fn new_backend(&self, url: Url, _: &ObjectStoreParams)
///         -> lance_core::Result<BackendConfig>
///     {
///         Ok(BackendConfig::new(self.0.clone(), url.scheme()))
///     }
/// }
///
/// # async fn open(backend: Arc<dyn StorageBackend>) -> lance_core::Result<()> {
/// let registry = Arc::new(ObjectStoreRegistry::default());
/// registry.insert("custom", Arc::new(CustomProvider(backend)));
/// let (storage, path) = ObjectStore::from_uri_and_params(
///     registry, "custom://base/data.bin", &ObjectStoreParams::default(),
/// ).await?;
/// storage.put(&path, b"data").await?;
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    /// Construct the physical backend without installing caller wrappers or IO tracking.
    /// The registry applies these policies after this method returns.
    async fn new_backend(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<BackendConfig>;

    /// Whether this provider resolves native filesystem locations.
    /// Used to compare unopened locations across aliases; does not grant native IO.
    fn is_local(&self) -> bool {
        false
    }

    /// Extract the path relative to the base of the store.
    ///
    /// For example, in S3 the path is relative to the bucket. So a URL of
    /// `s3://bucket/path/to/file` would return `path/to/file`.
    ///
    /// Meanwhile, for a file store, the path is relative to the filesystem root.
    /// So a URL of `file:///path/to/file` would return `/path/to/file`.
    fn extract_path(&self, url: &Url) -> Result<Path> {
        // url.path() returns a percent-encoded string (per the WHATWG URL spec).
        // Path::from_url_path decodes it first so the Path internal representation
        // holds the raw UTF-8 string. This prevents double-encoding when the
        // object store client later percent-encodes the path for HTTP requests.
        Path::from_url_path(url.path()).map_err(|e| {
            Error::invalid_input(format!("Invalid path in URL '{}': {}", url.path(), e))
        })
    }

    /// Calculate the unique prefix that should be used for this object store.
    ///
    /// For object stores that don't have the concept of buckets, this will just be something like
    /// 'file' or 'memory'.
    ///
    /// In object stores where all bucket names are unique, like s3, this will be
    /// simply 's3$my_bucket_name' or similar.
    ///
    /// In Azure, only the combination of (account name, container name) is unique, so
    /// this will be something like 'az$account_name@container'
    ///
    /// Providers should override this if they have special requirements like Azure's.
    fn calculate_storage_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), url.authority()))
    }
}

/// Default smallest cache size at which [`ObjectStoreRegistry::get_store()`]
/// sweeps out dead entries; see [`ObjectStoreRegistry::set_min_prune_threshold()`].
pub const DEFAULT_MIN_PRUNE_THRESHOLD: usize = 64;

/// Statistics for the object store registry cache.
#[derive(Debug, Clone, Default)]
pub struct ObjectStoreRegistryStats {
    /// Number of cache hits (store was already cached and reused).
    pub hits: u64,
    /// Number of cache misses (new store had to be created).
    pub misses: u64,
    /// Number of currently active object stores in the cache.
    pub active_stores: usize,
}

/// Serializes the builds of one cached store key.
type BuildLock = Arc<tokio::sync::Mutex<()>>;

/// A registry of object store providers.
///
/// Use [`Self::default()`] to create one with the available default providers.
/// This includes (depending on features enabled):
/// - `memory`: An in-memory object store.
/// - `file`: A local file object store, with optimized code paths.
/// - `file-object-store`: A local file object store that uses the ObjectStore API,
///   for all operations. Used for testing with ObjectStore wrappers.
/// - `file+uring`: A local file object store using io_uring (Linux only).
/// - `s3`: An S3 object store.
/// - `s3+ddb`: An S3 object store with DynamoDB for metadata.
/// - `az`: An Azure Blob ObjectStore object store.
/// - `gs`: A Google Cloud ObjectStore object store.
/// - `tos`: A Volcengine TOS object store.
///
/// Use [`Self::empty()`] to create an empty registry, with no providers registered.
///
/// The registry also caches object stores that are currently in use. It holds
/// weak references to the object stores, so they are not held onto. Entries
/// whose object store is no longer in use are removed by
/// [`Self::active_stores()`], when [`Self::get_store()`] looks up the same key
/// again, and in bulk once the cache outgrows its live entries (see
/// [`Self::get_store()`]), so a long-lived registry that sees many distinct
/// keys (e.g. one per namespace table) stays bounded.
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores:
        RwLock<HashMap<(String, ObjectStoreParams, SchedulerErrorMode), Weak<ObjectStore>>>,
    // Held by `get_store` while it builds the store for a key, so concurrent
    // cold calls for that key wait for one build instead of each making one.
    building: Mutex<HashMap<(String, ObjectStoreParams, SchedulerErrorMode), BuildLock>>,
    // Floor for `prune_threshold`; 0 disables the sweep.
    min_prune_threshold: AtomicUsize,
    // Cache size at which `get_store` sweeps out dead entries before inserting.
    prune_threshold: AtomicUsize,
    // Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ObjectStoreRegistry {
    /// Create a new registry with no providers registered.
    ///
    /// Typically, you want to use [`Self::default()`] instead, so you get the
    /// default providers.
    pub fn empty() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            active_stores: RwLock::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
            min_prune_threshold: AtomicUsize::new(DEFAULT_MIN_PRUNE_THRESHOLD),
            prune_threshold: AtomicUsize::new(DEFAULT_MIN_PRUNE_THRESHOLD),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Remove providers before sharing this registry with a session.
    ///
    /// Consuming `self` prevents races with in-flight store creation. All cached
    /// store bindings are discarded so re-registering a scheme cannot resurrect
    /// an earlier provider's store, even when schemes share a cache prefix.
    /// Existing store handles held by callers remain usable; this is a
    /// construction-time restriction, not revocation of issued handles.
    /// Unknown schemes are ignored. Returns an error if a registry lock is poisoned.
    ///
    /// ```
    /// use lance_io::object_store::ObjectStoreRegistry;
    /// # fn configured() -> lance_core::Result<()> {
    /// let registry = ObjectStoreRegistry::default()
    ///     .without_providers(["file", "file-object-store", "file+uring"])?;
    /// assert!(registry.get_provider("file").is_none());
    /// assert!(registry.get_provider("memory").is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn without_providers(
        mut self,
        schemes: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self> {
        let providers = self.providers.get_mut().ok().expect_ok()?;
        for scheme in schemes {
            providers.remove(scheme.as_ref());
        }
        self.active_stores.get_mut().ok().expect_ok()?.clear();
        self.building.get_mut().ok().expect_ok()?.clear();
        Ok(self)
    }

    /// Set the smallest cache size at which [`Self::get_store()`] sweeps out
    /// dead entries (default [`DEFAULT_MIN_PRUNE_THRESHOLD`]); `0` disables the
    /// sweep. Applies to the whole registry, i.e. to every dataset sharing the
    /// session. Also settable per call through the
    /// [`STORAGE_CACHE_PRUNE_THRESHOLD_OPTION`] storage option, e.g. from
    /// the storage options a namespace returns in `describe_table`.
    ///
    /// [`STORAGE_CACHE_PRUNE_THRESHOLD_OPTION`]: crate::object_store::STORAGE_CACHE_PRUNE_THRESHOLD_OPTION
    pub fn set_min_prune_threshold(&self, threshold: usize) {
        self.min_prune_threshold.store(threshold, Ordering::Relaxed);
        self.prune_threshold.store(threshold, Ordering::Relaxed);
    }

    /// Get the object store provider for a given scheme.
    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .get(scheme)
            .cloned()
    }

    /// Get a list of all active object stores.
    ///
    /// Calling this will also clean up any weak references to object stores that
    /// are no longer valid.
    pub fn active_stores(&self) -> Vec<Arc<ObjectStore>> {
        let mut found_inactive = false;
        let output = self
            .active_stores
            .read()
            .expect("ObjectStoreRegistry lock poisoned")
            .values()
            .filter_map(|weak| match weak.upgrade() {
                Some(store) => Some(store),
                None => {
                    found_inactive = true;
                    None
                }
            })
            .collect();

        if found_inactive {
            // Clean up the cache by removing any weak references that are no longer valid
            let mut cache_lock = self
                .active_stores
                .write()
                .expect("ObjectStoreRegistry lock poisoned");
            cache_lock.retain(|_, weak| weak.upgrade().is_some());
        }
        output
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns the number of cache hits, misses, and currently active stores.
    /// This is useful for detecting configuration issues that cause excessive
    /// cache misses (e.g., storage options that vary per-request).
    pub fn stats(&self) -> ObjectStoreRegistryStats {
        let active_stores = self
            .active_stores
            .read()
            .map(|s| s.values().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0);
        ObjectStoreRegistryStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            active_stores,
        }
    }

    fn scheme_not_found_error(&self, scheme: &str) -> Error {
        let mut message = format!("No object store provider found for scheme: '{}'", scheme);
        if let Ok(providers) = self.providers.read() {
            let valid_schemes = providers.keys().cloned().collect::<Vec<_>>().join(", ");
            message.push_str(&format!("\nValid schemes: {}", valid_schemes));
        }
        Error::invalid_input(message)
    }

    async fn build_store(
        &self,
        provider: Arc<dyn ObjectStoreProvider>,
        base_path: Url,
        params: &ObjectStoreParams,
        metrics_base: &str,
        store_prefix: &str,
        scheduler_error_mode: SchedulerErrorMode,
    ) -> Result<Arc<ObjectStore>> {
        let block_size = params.resolved_block_size()?;
        let mut config = provider.new_backend(base_path, params).await?;
        if let Some(block_size) = block_size {
            config.block_size = block_size;
        }
        let mut store = Self::assemble(config, params, store_prefix, scheduler_error_mode);
        store.inner = store.inner.traced();
        crate::object_store::meter_store(&mut store.inner, &mut store.io_tracker, metrics_base);
        for wrapper in &params.backend_wrappers {
            store.apply_wrapper(wrapper.as_ref());
        }
        store.track_io();
        Ok(Arc::new(store))
    }

    // Both asynchronous providers and infallible built-ins finish through this path.
    fn assemble(
        config: BackendConfig,
        params: &ObjectStoreParams,
        store_prefix: &str,
        scheduler_error_mode: SchedulerErrorMode,
    ) -> ObjectStore {
        ObjectStore {
            inner: config.backend,
            has_wrappers: false,
            external_blob_inner: None,
            _external_blob_origin: None,
            local_dir_operations: config.local_dir_operations,
            local_fsync: config.local_fsync,
            direct_local_io: config.direct_local_io,
            local_io: config.local_io,
            is_local: config.is_local,
            is_remote: config.is_remote,
            max_single_copy_size: config.max_single_copy_size,
            scheme: config.scheme,
            block_size: config.block_size,
            max_iop_size: *super::DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: config.use_constant_size_upload_parts,
            list_is_lexically_ordered: config.list_is_lexically_ordered,
            io_parallelism: config.io_parallelism,
            download_retry_count: StorageOptions::new(
                params.storage_options().cloned().unwrap_or_default(),
            )
            .download_retry_count(),
            scheduler_error_mode,
            io_tracker: Default::default(),
            store_prefix: config
                .store_prefix
                .unwrap_or_else(|| store_prefix.to_owned()),
        }
    }

    pub(super) fn local() -> ObjectStore {
        Self::builtin(local::FileStoreProvider::default_backend())
    }

    pub(super) fn memory() -> ObjectStore {
        Self::builtin(memory::MemoryStoreProvider::default_backend())
    }

    fn builtin(config: BackendConfig) -> ObjectStore {
        let prefix = config.scheme.clone();
        Self::assemble(
            config,
            &ObjectStoreParams::default(),
            &prefix,
            SchedulerErrorMode::default(),
        )
    }

    /// Build a fresh object store without consulting or populating the cache.
    ///
    /// Callers should retain the returned [`Arc`] for as long as they want to
    /// reuse provider-local state such as HTTP clients and rate limiters.
    #[doc(hidden)]
    pub async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        // Base-scoped storage options (`base_<id>.<key>`) are directives for
        // other registered base paths; resolve them away before building a
        // store for this location.
        let params = params.scoped_to_base(None);
        let params = params.as_ref();
        let scheduler_error_mode = params.scheduler_error_mode()?;
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };
        let store_prefix =
            provider.calculate_storage_prefix(&base_path, params.storage_options())?;
        let metrics_base = crate::object_store::metrics_base(&store_prefix, &base_path);

        self.build_store(
            provider,
            base_path,
            params,
            &metrics_base,
            &store_prefix,
            scheduler_error_mode,
        )
        .await
    }

    /// Get an object store for a given base path and parameters.
    ///
    /// If the object store is already in use, it will return a strong reference
    /// to the object store. If the object store is not in use, it will create a
    /// new object store and return a strong reference to it. Concurrent calls
    /// for the same key build it once; the others wait and reuse it.
    ///
    /// A dead entry is only found again if the same key is looked up, which
    /// never happens when every key is distinct (each namespace table gets its
    /// own storage options provider, hence its own key). So before inserting,
    /// once the cache reaches twice its live size (and at least the minimum
    /// prune threshold), all dead entries are swept out; this keeps the cache
    /// bounded at amortized O(1) cost per insert.
    ///
    /// If `params` carry the `object_store_cache_prune_threshold` storage
    /// option, it is applied first via [`Self::set_min_prune_threshold()`].
    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<Arc<ObjectStore>> {
        // Base-scoped storage options (`base_<id>.<key>`) are directives for
        // other registered base paths; resolve them away before building or
        // caching a store for this location. Params already resolved for a
        // base contain no scoped entries, so this is a no-op for them.
        let params = params.scoped_to_base(None);
        let params = params.as_ref();
        // Validate before cache lookup so a cached store cannot hide an invalid
        // value. The normalized mode also forms part of the cache identity.
        let scheduler_error_mode = params.scheduler_error_mode()?;
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };
        if let Some(options) = params.storage_options()
            && let Some(threshold) =
                StorageOptions(options.clone()).storage_cache_prune_threshold()?
            && threshold != self.min_prune_threshold.load(Ordering::Relaxed)
        {
            self.set_min_prune_threshold(threshold);
        }

        // Keyed by the metrics identity rather than the bare prefix: in
        // `dataset` label mode that is the dataset URI, so datasets sharing a
        // bucket get separate stores and their metrics stay attributed.
        let store_prefix =
            provider.calculate_storage_prefix(&base_path, params.storage_options())?;
        let cache_path = crate::object_store::metrics_base(&store_prefix, &base_path);
        let cache_key = (cache_path.clone(), params.clone(), scheduler_error_mode);

        // Check if we have a cached store for this base path and params
        {
            let maybe_store = self
                .active_stores
                .read()
                .ok()
                .expect_ok()?
                .get(&cache_key)
                .cloned();
            if let Some(store) = maybe_store {
                if let Some(store) = store.upgrade() {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(store);
                } else {
                    // Remove the weak reference if it is no longer valid
                    let mut cache_lock = self
                        .active_stores
                        .write()
                        .expect("ObjectStoreRegistry lock poisoned");
                    if let Some(store) = cache_lock.get(&cache_key)
                        && store.upgrade().is_none()
                    {
                        // Remove the weak reference if it is no longer valid
                        cache_lock.remove(&cache_key);
                    }
                }
            }
        }

        // Locks that no call holds any more are swept first, so the map only
        // grows with the builds in flight.
        let build_lock = {
            let mut building = self.building.lock().ok().expect_ok()?;
            building.retain(|_, lock| Arc::strong_count(lock) > 1);
            building.entry(cache_key.clone()).or_default().clone()
        };
        let _build_guard = match build_lock.try_lock() {
            Ok(guard) => Some(guard),
            // Wait for the build in flight without then holding the lock, so if
            // that build failed, the calls that waited retry together rather
            // than one after another.
            Err(_) => {
                drop(build_lock.lock().await);
                None
            }
        };
        // Another call may have built the store while this one waited.
        let maybe_store = self
            .active_stores
            .read()
            .ok()
            .expect_ok()?
            .get(&cache_key)
            .and_then(Weak::upgrade);
        if let Some(store) = maybe_store {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(store);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);

        let store = self
            .build_store(
                provider,
                base_path,
                params,
                &cache_path,
                &store_prefix,
                scheduler_error_mode,
            )
            .await?;

        {
            // Insert the store into the cache
            let mut cache_lock = self.active_stores.write().ok().expect_ok()?;
            let min_threshold = self.min_prune_threshold.load(Ordering::Relaxed);
            if min_threshold > 0 && cache_lock.len() >= self.prune_threshold.load(Ordering::Relaxed)
            {
                cache_lock.retain(|_, weak| weak.strong_count() > 0);
                self.prune_threshold
                    .store((cache_lock.len() * 2).max(min_threshold), Ordering::Relaxed);
            }
            cache_lock.insert(cache_key, Arc::downgrade(&store));
        }

        Ok(store)
    }

    /// Calculate the datastore prefix based on the URI and the storage options.
    /// The data store prefix should uniquely identify the datastore.
    pub fn calculate_storage_prefix(
        &self,
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let url = uri_to_url(uri)?;
        match self.get_provider(url.scheme()) {
            None => {
                if url.scheme() == "file" || url.scheme().len() == 1 {
                    Ok("file".to_string())
                } else {
                    Err(self.scheme_not_found_error(url.scheme()))
                }
            }
            Some(provider) => provider.calculate_storage_prefix(&url, storage_options),
        }
    }
}

impl Default for ObjectStoreRegistry {
    fn default() -> Self {
        let mut providers: HashMap<String, Arc<dyn ObjectStoreProvider>> = HashMap::new();

        providers.insert("memory".into(), Arc::new(memory::MemoryStoreProvider));
        providers.insert(
            "shared-memory".into(),
            Arc::new(shared_memory::SharedMemoryStoreProvider::default()),
        );
        providers.insert("file".into(), Arc::new(local::FileStoreProvider));
        // The "file" scheme has special optimized code paths that bypass
        // the ObjectStore API for better performance. However, this can make it
        // hard to test when using ObjectStore wrappers, such as IOTrackingStore.
        // So we provide a "file-object-store" scheme that uses the ObjectStore API.
        // The specialized code paths are differentiated by the scheme name.
        providers.insert(
            "file-object-store".into(),
            Arc::new(local::FileStoreProvider),
        );
        #[cfg(target_os = "linux")]
        providers.insert("file+uring".into(), Arc::new(local::FileStoreProvider));

        #[cfg(feature = "aws")]
        {
            let aws = Arc::new(aws::AwsStoreProvider);
            providers.insert("s3".into(), aws.clone());
            providers.insert("s3+ddb".into(), aws);
        }
        #[cfg(feature = "azure")]
        {
            let azure = Arc::new(azure::AzureBlobStoreProvider);
            providers.insert("az".into(), azure.clone());
            providers.insert("abfss".into(), azure);
        }
        #[cfg(feature = "gcp")]
        providers.insert("gs".into(), Arc::new(gcp::GcsStoreProvider));
        #[cfg(feature = "goosefs")]
        providers.insert("goosefs".into(), Arc::new(goosefs::GooseFsStoreProvider));
        #[cfg(feature = "oss")]
        providers.insert("oss".into(), Arc::new(oss::OssStoreProvider));
        #[cfg(feature = "tencent")]
        providers.insert("cos".into(), Arc::new(tencent::TencentStoreProvider));
        #[cfg(feature = "huggingface")]
        providers.insert("hf".into(), Arc::new(huggingface::HuggingfaceStoreProvider));
        #[cfg(feature = "tos")]
        providers.insert("tos".into(), Arc::new(tos::TosStoreProvider));
        Self {
            providers: RwLock::new(providers),
            active_stores: RwLock::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
            min_prune_threshold: AtomicUsize::new(DEFAULT_MIN_PRUNE_THRESHOLD),
            prune_threshold: AtomicUsize::new(DEFAULT_MIN_PRUNE_THRESHOLD),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

impl ObjectStoreRegistry {
    /// Add a new object store provider to the registry. The provider will be used
    /// in [`Self::get_store()`] when a URL is passed with a matching scheme.
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .write()
            .expect("ObjectStoreRegistry lock poisoned")
            .insert(scheme.into(), provider);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::*;
    use crate::object_store::STORAGE_CACHE_PRUNE_THRESHOLD_OPTION;

    use crate::object_store::providers::memory::MemoryStoreProvider;
    use crate::object_store::{StorageOptionsAccessor, StorageOptionsProvider};
    use crate::scheduler::SCHEDULER_ERROR_MODE_STORAGE_OPTION;
    use rstest::rstest;

    #[rstest]
    #[case::local("file:///tmp", false)]
    #[case::local_backend("file-object-store:///tmp", false)]
    #[case::memory("memory://", true)]
    #[tokio::test]
    async fn test_provider_fixed_options_take_precedence(#[case] uri: &str, #[case] ordered: bool) {
        let params = ObjectStoreParams {
            use_constant_size_upload_parts: true,
            list_is_lexically_ordered: Some(!ordered),
            ..Default::default()
        };
        let store = ObjectStoreRegistry::default()
            .get_store(Url::parse(uri).unwrap(), &params)
            .await
            .unwrap();
        assert!(!store.use_constant_size_upload_parts());
        assert_eq!(store.list_is_lexically_ordered(), ordered);
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    #[tokio::test]
    async fn test_registry_preserves_resolved_backend_options(#[case] enabled: bool) {
        let mut config = super::memory::MemoryStoreProvider::default_backend();
        config.use_constant_size_upload_parts = enabled;
        config.list_is_lexically_ordered = enabled;
        let params = ObjectStoreParams {
            use_constant_size_upload_parts: !enabled,
            list_is_lexically_ordered: Some(!enabled),
            ..Default::default()
        };
        let storage = crate::utils::testing::TestObjectStoreProvider(config)
            .build(Url::parse("memory://").unwrap(), &params);
        assert_eq!(storage.use_constant_size_upload_parts(), enabled);
        assert_eq!(storage.list_is_lexically_ordered(), enabled);
    }

    #[tokio::test]
    async fn test_without_providers_discards_cached_bindings() {
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams::default();
        let memory_uri = Url::parse("memory://").unwrap();
        let memory = registry
            .get_store(memory_uri.clone(), &params)
            .await
            .unwrap();
        let file_uri = Url::parse("file-object-store:///tmp").unwrap();
        let file = registry.get_store(file_uri.clone(), &params).await.unwrap();
        let registry = registry
            .without_providers(["file", "file-object-store", "file+uring", "absent"])
            .unwrap();
        assert!(registry.active_stores().is_empty());
        for scheme in ["file", "file-object-store", "file+uring"] {
            assert!(registry.get_provider(scheme).is_none());
            let uri = Url::parse(&format!("{scheme}:///tmp")).unwrap();
            let error = registry.get_store(uri.clone(), &params).await.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("No object store provider found"));
            let error = registry.new_store(uri, &params).await.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("No object store provider found"));
        }
        // Re-registering cannot revive a binding created by the old provider.
        registry.insert(
            "file-object-store",
            Arc::new(SchedulerModeIgnoringStoreProvider),
        );
        let replacement = registry.get_store(file_uri, &params).await.unwrap();
        assert!(!Arc::ptr_eq(&file, &replacement));
        let reopened = registry.get_store(memory_uri, &params).await.unwrap();
        assert!(!Arc::ptr_eq(&memory, &reopened));
        assert_eq!(reopened.scheme(), "memory");
    }

    #[derive(Debug)]
    struct DummyProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for DummyProvider {
        async fn new_backend(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<BackendConfig> {
            unreachable!("This test doesn't create stores")
        }
    }

    #[derive(Debug)]
    struct StableStorageOptionsProvider;

    #[async_trait::async_trait]
    impl StorageOptionsProvider for StableStorageOptionsProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            Ok(None)
        }

        fn provider_id(&self) -> String {
            "stable-scheduler-mode-provider".to_string()
        }
    }

    #[derive(Debug)]
    struct SchedulerModeIgnoringStoreProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for SchedulerModeIgnoringStoreProvider {
        async fn new_backend(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<BackendConfig> {
            MemoryStoreProvider.new_backend(_base_path, _params).await
        }
    }

    #[test]
    fn test_calculate_storage_prefix() {
        let provider = DummyProvider;
        let url = Url::parse("dummy://blah/path").unwrap();
        assert_eq!(
            "dummy$blah",
            provider.calculate_storage_prefix(&url, None).unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_store_resolves_base_scoped_options() {
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();

        let with_scoped = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("shared".to_string(), "value".to_string()),
                    ("base_1.account_key".to_string(), "base1-key".to_string()),
                ]),
            ))),
            ..Default::default()
        };
        let without_scoped = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("shared".to_string(), "value".to_string())]),
            ))),
            ..Default::default()
        };

        // Base-scoped entries are resolved away before the store is built and
        // cached, so params with and without them yield the same cached store.
        let store_scoped = registry.get_store(url.clone(), &with_scoped).await.unwrap();
        let store_plain = registry.get_store(url, &without_scoped).await.unwrap();
        assert!(Arc::ptr_eq(&store_scoped, &store_plain));
    }

    #[tokio::test]
    async fn test_provider_cache_separates_and_validates_scheduler_error_mode() {
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory:///table").unwrap();
        let storage_options_provider = Arc::new(StableStorageOptionsProvider);
        let params_for = |value: &str| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                StorageOptionsAccessor::with_initial_and_provider(
                    HashMap::from([(
                        SCHEDULER_ERROR_MODE_STORAGE_OPTION.to_string(),
                        value.to_string(),
                    )]),
                    storage_options_provider.clone(),
                ),
            )),
            ..Default::default()
        };

        let best_effort = registry
            .get_store(url.clone(), &params_for("best_effort"))
            .await
            .unwrap();
        let fail_fast = registry
            .get_store(url.clone(), &params_for("fail_fast"))
            .await
            .unwrap();
        let best_effort_again = registry
            .get_store(url.clone(), &params_for("best_effort"))
            .await
            .unwrap();

        assert_eq!(
            best_effort.scheduler_error_mode(),
            SchedulerErrorMode::BestEffort
        );
        assert_eq!(
            fail_fast.scheduler_error_mode(),
            SchedulerErrorMode::FailFast
        );
        assert!(!Arc::ptr_eq(&best_effort, &fail_fast));
        assert!(Arc::ptr_eq(&best_effort, &best_effort_again));

        let error = registry
            .get_store(url, &params_for("invalid"))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(
            "Invalid scheduler_error_mode value 'invalid'. Valid values are: best_effort, fail_fast"
        ));

        let stats = registry.stats();
        assert_eq!((stats.hits, stats.misses, stats.active_stores), (1, 2, 2));
    }

    #[tokio::test]
    async fn test_registry_applies_scheduler_error_mode_to_custom_provider_store() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert(
            "custom-memory",
            Arc::new(SchedulerModeIgnoringStoreProvider),
        );
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(
                    SCHEDULER_ERROR_MODE_STORAGE_OPTION.to_string(),
                    "fail_fast".to_string(),
                )]),
            ))),
            ..Default::default()
        };

        let store = registry
            .get_store(Url::parse("custom-memory:///table").unwrap(), &params)
            .await
            .unwrap();

        assert_eq!(store.scheduler_error_mode(), SchedulerErrorMode::FailFast);
    }

    #[test]
    fn test_calculate_storage_scheme_not_found() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        let s = "Invalid user input: No object store provider found for scheme: 'dummy2'\nValid schemes: dummy";
        let result = registry
            .calculate_storage_prefix("dummy2://mybucket/my/long/path", None)
            .expect_err("expected error")
            .to_string();
        assert_eq!(s, &result[..s.len()]);
    }

    // Test that paths without a scheme get treated as local paths.
    #[test]
    fn test_calculate_storage_prefix_for_local() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_storage_prefix("/tmp/foobar", None)
                .unwrap()
        );
    }

    // Test that paths with a single-letter scheme that is not registered for anything get treated as local paths.
    #[test]
    fn test_calculate_storage_prefix_for_local_windows_path() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_storage_prefix("c://dos/path", None)
                .unwrap()
        );
    }

    // Test that paths with a given scheme get mapped to that storage provider.
    #[test]
    fn test_calculate_storage_prefix_for_dummy_path() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        assert_eq!(
            "dummy$mybucket",
            registry
                .calculate_storage_prefix("dummy://mybucket/my/long/path", None)
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_stats_hit_miss_tracking() {
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();

        let params1 = ObjectStoreParams::default();
        let params2 = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("k".into(), "v".into())]),
            ))),
            ..Default::default()
        };

        // (hits, misses, active)
        let cases: &[(&ObjectStoreParams, (u64, u64, usize))] = &[
            (&params1, (0, 1, 1)), // miss: new params
            (&params1, (1, 1, 1)), // hit: same params
            (&params2, (1, 2, 2)), // miss: different storage_options
        ];

        let mut stores = vec![]; // retain the stores
        for (params, (hits, misses, active)) in cases {
            stores.push(registry.get_store(url.clone(), params).await.unwrap());
            let s = registry.stats();
            assert_eq!(
                (s.hits, s.misses, s.active_stores),
                (*hits, *misses, *active)
            );
        }

        // Same params returns same instance
        assert!(Arc::ptr_eq(&stores[0], &stores[1]));
    }

    /// A memory store provider whose builds are slow enough to overlap.
    #[derive(Debug, Default)]
    struct SlowProvider {
        builds: AtomicUsize,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for SlowProvider {
        async fn new_backend(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<BackendConfig> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(100)).await;
            if self.fail {
                return Err(Error::io("build failed"));
            }
            MemoryStoreProvider.new_backend(base_path, params).await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_concurrent_get_store_builds_once() {
        let provider = Arc::new(SlowProvider::default());
        let registry = ObjectStoreRegistry::empty();
        registry.insert("slowmem", provider.clone());
        let url = Url::parse("slowmem:///").unwrap();
        let params = ObjectStoreParams::default();

        let stores =
            futures::future::try_join_all((0..8).map(|_| registry.get_store(url.clone(), &params)))
                .await
                .unwrap();

        assert_eq!(provider.builds.load(Ordering::Relaxed), 1);
        assert!(stores.iter().all(|store| Arc::ptr_eq(store, &stores[0])));
        let stats = registry.stats();
        assert_eq!((stats.hits, stats.misses), (7, 1));
    }

    #[tokio::test(start_paused = true)]
    async fn test_concurrent_get_store_failures_retry_together() {
        let provider = Arc::new(SlowProvider {
            fail: true,
            ..Default::default()
        });
        let registry = ObjectStoreRegistry::empty();
        registry.insert("slowmem", provider.clone());
        let url = Url::parse("slowmem:///").unwrap();
        let params = ObjectStoreParams::default();

        let start = tokio::time::Instant::now();
        let results =
            futures::future::join_all((0..8).map(|_| registry.get_store(url.clone(), &params)))
                .await;

        for result in &results {
            let Err(error) = result else {
                panic!("expected the failed build's error");
            };
            assert!(matches!(error, Error::IO { .. }), "{error:?}");
            assert!(error.to_string().contains("build failed"), "{error}");
        }
        // The first build fails, then the seven calls that waited retry at once.
        assert_eq!(provider.builds.load(Ordering::Relaxed), 8);
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(300), "took {elapsed:?}");
    }

    #[tokio::test]
    async fn test_get_store_prunes_dead_entries_with_distinct_keys() {
        use crate::object_store::StorageOptionsAccessor;
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();
        let params_for = |i: usize| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("table".into(), i.to_string())]),
            ))),
            ..Default::default()
        };
        let cache_len = || registry.active_stores.read().unwrap().len();

        // Every 10th store stays alive; the rest are dropped right away, like
        // datasets opened once per sample, each on a different table.
        let n = 10 * DEFAULT_MIN_PRUNE_THRESHOLD;
        let mut alive = vec![];
        for i in 0..n {
            let store = registry
                .get_store(url.clone(), &params_for(i))
                .await
                .unwrap();
            if i % 10 == 0 {
                alive.push((i, store));
            }
        }
        // Without pruning this would be `n`.
        assert!(cache_len() <= 2 * alive.len() + DEFAULT_MIN_PRUNE_THRESHOLD);
        assert_eq!(registry.stats().active_stores, alive.len());

        // Live entries survive pruning and are still reused.
        for (i, store) in &alive {
            let again = registry
                .get_store(url.clone(), &params_for(*i))
                .await
                .unwrap();
            assert!(Arc::ptr_eq(store, &again));
        }
        assert_eq!(registry.stats().hits, alive.len() as u64);
    }

    /// Opens `n` stores with distinct keys, each carrying `extra` storage
    /// options, drops them all, and returns the resulting cache size.
    async fn cache_len_after_distinct_dropped_stores(
        registry: &ObjectStoreRegistry,
        n: usize,
        extra: &[(&str, &str)],
    ) -> usize {
        use crate::object_store::StorageOptionsAccessor;
        let url = Url::parse("memory://test").unwrap();
        for i in 0..n {
            let mut options = HashMap::from([("table".to_string(), i.to_string())]);
            options.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
            let params = ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            };
            registry.get_store(url.clone(), &params).await.unwrap();
        }
        registry.active_stores.read().unwrap().len()
    }

    #[rstest]
    #[case::disabled("0", 200, 200)]
    #[case::custom("8", 200, 8)]
    #[tokio::test]
    async fn test_prune_threshold_from_storage_option(
        #[case] threshold: &str,
        #[case] n: usize,
        #[case] max_cache_len: usize,
    ) {
        // A namespace configures this through the storage options it returns
        // from `describe_table`.
        let registry = ObjectStoreRegistry::default();
        let extra = [(STORAGE_CACHE_PRUNE_THRESHOLD_OPTION, threshold)];
        let len = cache_len_after_distinct_dropped_stores(&registry, n, &extra).await;
        assert!(len <= max_cache_len, "cache has {len} entries");
        assert_eq!(registry.stats().active_stores, 0);
    }

    #[tokio::test]
    async fn test_set_min_prune_threshold() {
        let registry = ObjectStoreRegistry::default();
        registry.set_min_prune_threshold(0);
        assert_eq!(
            cache_len_after_distinct_dropped_stores(&registry, 200, &[]).await,
            200
        );
        // Re-enabling sweeps on the next insert.
        registry.set_min_prune_threshold(16);
        assert!(cache_len_after_distinct_dropped_stores(&registry, 1, &[]).await <= 16);
    }

    #[tokio::test]
    async fn test_invalid_prune_threshold_option() {
        use crate::object_store::StorageOptionsAccessor;
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(
                    STORAGE_CACHE_PRUNE_THRESHOLD_OPTION.to_string(),
                    "lots".to_string(),
                )]),
            ))),
            ..Default::default()
        };
        let err = registry
            .get_store(Url::parse("memory://test").unwrap(), &params)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains(STORAGE_CACHE_PRUNE_THRESHOLD_OPTION),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_new_store_bypasses_cache() {
        let registry = ObjectStoreRegistry::default();
        let url = Url::parse("memory://test").unwrap();
        let params = ObjectStoreParams::default();

        let first = registry.new_store(url.clone(), &params).await.unwrap();
        let second = registry.new_store(url, &params).await.unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
        let stats = registry.stats();
        assert_eq!((stats.hits, stats.misses, stats.active_stores), (0, 0, 0));
    }
}
