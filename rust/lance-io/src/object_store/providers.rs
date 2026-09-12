// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{
        Arc, RwLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use object_store::path::Path;
use url::Url;

use crate::object_store::WrappingObjectStore;
use crate::object_store::uri_to_url;

use super::{ObjectStore, ObjectStoreParams, tracing::ObjectStoreTracingExt};
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
#[cfg(feature = "oss")]
pub mod oss;
pub mod shared_memory;
#[cfg(feature = "tencent")]
pub mod tencent;
#[cfg(feature = "tos")]
pub mod tos;

/// Which built-in commit handler a provider's stores should use.
///
/// Commit-handler selection is otherwise inferred from the URL scheme. A
/// registered out-of-tree provider serves a scheme that the built-in resolver
/// does not recognize, so this lets the provider declare the guarantee its
/// store actually offers instead of falling back to a scheme-string guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommitHandlerType {
    /// Atomic put-if-not-exists. Safe for concurrent writers: a second writer
    /// racing on the same manifest version gets a conflict instead of silently
    /// overwriting. This is the default; it fits any store whose backend
    /// supports create-if-not-exists (most object stores).
    #[default]
    ConditionalPut,
    /// Blind overwrite with no concurrency protection. A store that cannot
    /// offer atomic create-if-not-exists should return this so callers get the
    /// documented "concurrent writes may lose data" behavior rather than an
    /// error at commit time.
    Unsafe,
}

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;

    /// The commit handler this provider's stores should use.
    ///
    /// Defaults to the conflict-safe [`CommitHandlerType::ConditionalPut`].
    /// Override it to return [`CommitHandlerType::Unsafe`] for a store that
    /// cannot support atomic create-if-not-exists.
    fn commit_handler(&self) -> CommitHandlerType {
        CommitHandlerType::ConditionalPut
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
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), url.authority()))
    }
}

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
/// - `az`: An Azure Blob Storage object store.
/// - `gs`: A Google Cloud Storage object store.
/// - `tos`: A Volcengine TOS object store.
///
/// Use [`Self::empty()`] to create an empty registry, with no providers registered.
///
/// The registry also caches object stores that are currently in use. It holds
/// weak references to the object stores, so they are not held onto. If an object
/// store is no longer in use, it will be removed from the cache on the next
/// call to either [`Self::active_stores()`] or [`Self::get_store()`].
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores: RwLock<HashMap<(String, ObjectStoreParams), Weak<ObjectStore>>>,
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
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
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
        store_prefix: &str,
    ) -> Result<Arc<ObjectStore>> {
        let mut store = provider.new_store(base_path, params).await?;

        // Capture the provider's declared commit-handler capability on the
        // exact store it built, so commit-handler resolution is bound to this
        // store rather than a later registry lookup.
        store.commit_handler_type = provider.commit_handler();

        store.inner = store.inner.traced();

        // Label metrics by the store's unique prefix (e.g. `s3$bucket`,
        // `az$container@account`) so multiple stores on one cloud differ.
        crate::object_store::meter_store(&mut store.inner, &mut store.io_tracker, store_prefix);

        if let Some(wrapper) = &params.object_store_wrapper {
            store.apply_wrapper(wrapper.as_ref());
        }

        // Always wrap with IO tracking
        store.inner = store.io_tracker.wrap("", store.inner);

        Ok(Arc::new(store))
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
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };
        let store_prefix =
            provider.calculate_object_store_prefix(&base_path, params.storage_options())?;

        self.build_store(provider, base_path, params, &store_prefix)
            .await
    }

    /// Get an object store for a given base path and parameters.
    ///
    /// If the object store is already in use, it will return a strong reference
    /// to the object store. If the object store is not in use, it will create a
    /// new object store and return a strong reference to it.
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
        let scheme = base_path.scheme();
        let Some(provider) = self.get_provider(scheme) else {
            return Err(self.scheme_not_found_error(scheme));
        };

        let cache_path =
            provider.calculate_object_store_prefix(&base_path, params.storage_options())?;
        let cache_key = (cache_path.clone(), params.clone());

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

        self.misses.fetch_add(1, Ordering::Relaxed);

        let store = self
            .build_store(provider, base_path, params, &cache_path)
            .await?;

        {
            // Insert the store into the cache
            let mut cache_lock = self.active_stores.write().ok().expect_ok()?;
            cache_lock.insert(cache_key, Arc::downgrade(&store));
        }

        Ok(store)
    }

    /// Calculate the datastore prefix based on the URI and the storage options.
    /// The data store prefix should uniquely identify the datastore.
    pub fn calculate_object_store_prefix(
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
            Some(provider) => provider.calculate_object_store_prefix(&url, storage_options),
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
    use std::sync::Mutex;

    use super::*;
    use object_store::ObjectStore as OSObjectStore;

    use crate::object_store::providers::memory::MemoryStoreProvider;
    use object_store::list::{PaginatedListOptions, PaginatedListResult, PaginatedListStore};
    use rstest::rstest;

    #[derive(Debug)]
    struct DummyProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for DummyProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            unreachable!("This test doesn't create stores")
        }
    }

    /// A lister that exists only to be handed to a wrapper.
    struct StubLister;

    #[async_trait::async_trait]
    impl PaginatedListStore for StubLister {
        async fn list_paginated(
            &self,
            _prefix: Option<&str>,
            _opts: PaginatedListOptions,
        ) -> object_store::Result<PaginatedListResult> {
            unimplemented!("this lister exists to be wrapped, not to list")
        }
    }

    /// A provider whose stores come with a paginated lister, which the memory store does not.
    #[derive(Debug)]
    struct PaginatedProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for PaginatedProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            let mut store = MemoryStoreProvider.new_store(base_path, params).await?;
            store.paginated_lister = Some(Arc::new(StubLister));
            Ok(store)
        }

        fn calculate_object_store_prefix(
            &self,
            _url: &Url,
            _storage_options: Option<&HashMap<String, String>>,
        ) -> Result<String> {
            Ok("memory".to_string())
        }
    }

    /// Swaps the store out for an empty one, the way a wrapper enforcing visibility would, and
    /// records the prefix each call was labelled with. `keep_pushdown` is what it answers when
    /// asked about the lister.
    #[derive(Debug)]
    struct RecordingWrapper {
        keep_pushdown: bool,
        prefixes: Mutex<Vec<String>>,
    }

    impl WrappingObjectStore for RecordingWrapper {
        fn wrap(
            &self,
            store_prefix: &str,
            _original: Arc<dyn OSObjectStore>,
        ) -> Arc<dyn OSObjectStore> {
            self.prefixes
                .lock()
                .unwrap()
                .push(format!("wrap@{store_prefix}"));
            Arc::new(object_store::memory::InMemory::new())
        }

        fn wrap_paginated(
            &self,
            store_prefix: &str,
            original: Arc<dyn PaginatedListStore>,
        ) -> Option<Arc<dyn PaginatedListStore>> {
            self.prefixes
                .lock()
                .unwrap()
                .push(format!("wrap_paginated@{store_prefix}"));
            self.keep_pushdown.then_some(original)
        }
    }

    /// A decorator supplied through [`ObjectStoreParams`] has to reach the paginated lister
    /// too, or `read_dir_page` would talk to the backend behind its back — and a decorator
    /// that gives the pushdown up gets a store with no lister, so its listings go through the
    /// wrapped `inner` and see what the wrapper allows rather than what the backend holds.
    #[rstest]
    #[case::keeps_the_pushdown(true)]
    #[case::gives_up_the_pushdown(false)]
    #[tokio::test]
    async fn test_the_registry_hands_the_lister_to_the_wrapper(#[case] keep_pushdown: bool) {
        let wrapper = Arc::new(RecordingWrapper {
            keep_pushdown,
            prefixes: Mutex::new(Vec::new()),
        });
        let registry = ObjectStoreRegistry::default();
        registry.insert("pagmem", Arc::new(PaginatedProvider));

        let store = registry
            .get_store(
                Url::parse("pagmem:///").unwrap(),
                &ObjectStoreParams {
                    object_store_wrapper: Some(wrapper.clone()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(store.paginated_lister.is_some(), keep_pushdown);
        // Both halves of the store are labelled with the same prefix.
        assert_eq!(
            *wrapper.prefixes.lock().unwrap(),
            vec!["wrap@memory", "wrap_paginated@memory"]
        );
        if !keep_pushdown {
            // `StubLister` panics if it is ever asked to list, so reaching a page at all is
            // the other half of the assertion.
            let page = store
                .read_dir_page(Path::from(""), Default::default())
                .await
                .unwrap();
            assert!(page.result.common_prefixes.is_empty());
            assert!(page.result.objects.is_empty());
        }
    }

    #[test]
    fn test_calculate_object_store_prefix() {
        let provider = DummyProvider;
        let url = Url::parse("dummy://blah/path").unwrap();
        assert_eq!(
            "dummy$blah",
            provider.calculate_object_store_prefix(&url, None).unwrap()
        );
    }

    #[tokio::test]
    async fn test_get_store_resolves_base_scoped_options() {
        use crate::object_store::StorageOptionsAccessor;

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

    #[test]
    fn test_calculate_object_store_scheme_not_found() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        let s = "Invalid user input: No object store provider found for scheme: 'dummy2'\nValid schemes: dummy";
        let result = registry
            .calculate_object_store_prefix("dummy2://mybucket/my/long/path", None)
            .expect_err("expected error")
            .to_string();
        assert_eq!(s, &result[..s.len()]);
    }

    // Test that paths without a scheme get treated as local paths.
    #[test]
    fn test_calculate_object_store_prefix_for_local() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_object_store_prefix("/tmp/foobar", None)
                .unwrap()
        );
    }

    // Test that paths with a single-letter scheme that is not registered for anything get treated as local paths.
    #[test]
    fn test_calculate_object_store_prefix_for_local_windows_path() {
        let registry = ObjectStoreRegistry::empty();
        assert_eq!(
            "file",
            registry
                .calculate_object_store_prefix("c://dos/path", None)
                .unwrap()
        );
    }

    // Test that paths with a given scheme get mapped to that storage provider.
    #[test]
    fn test_calculate_object_store_prefix_for_dummy_path() {
        let registry = ObjectStoreRegistry::empty();
        registry.insert("dummy", Arc::new(DummyProvider));
        assert_eq!(
            "dummy$mybucket",
            registry
                .calculate_object_store_prefix("dummy://mybucket/my/long/path", None)
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_stats_hit_miss_tracking() {
        use crate::object_store::StorageOptionsAccessor;
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

    /// A provider that builds a working in-memory store yet declares it can
    /// only offer unsafe commits, used to prove the declared capability is
    /// stamped onto the store the registry builds even for a known scheme.
    #[derive(Debug)]
    struct UnsafeMemoryProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for UnsafeMemoryProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            MemoryStoreProvider.new_store(base_path, params).await
        }

        fn commit_handler(&self) -> CommitHandlerType {
            CommitHandlerType::Unsafe
        }

        fn calculate_object_store_prefix(
            &self,
            _url: &Url,
            _storage_options: Option<&HashMap<String, String>>,
        ) -> Result<String> {
            Ok("memory".to_string())
        }
    }

    #[tokio::test]
    async fn test_provider_commit_handler_travels_with_store() {
        let url = Url::parse("memory://bucket/ds").unwrap();

        // A built-in provider declares the conflict-safe default, and the
        // registry stamps that capability onto the store it builds.
        let registry = ObjectStoreRegistry::default();
        let store = registry
            .new_store(url.clone(), &ObjectStoreParams::default())
            .await
            .unwrap();
        assert_eq!(
            store.commit_handler_type(),
            CommitHandlerType::ConditionalPut
        );

        // Overriding a known scheme with a provider that declares Unsafe makes
        // the store it builds carry Unsafe: the capability is bound to the
        // exact store, not inferred from the scheme name.
        let registry = ObjectStoreRegistry::default();
        registry.insert("memory", Arc::new(UnsafeMemoryProvider));
        let store = registry
            .new_store(url.clone(), &ObjectStoreParams::default())
            .await
            .unwrap();
        assert_eq!(store.commit_handler_type(), CommitHandlerType::Unsafe);

        // A cached store keeps the capability it was built with.
        let cached = registry
            .get_store(url, &ObjectStoreParams::default())
            .await
            .unwrap();
        assert_eq!(cached.commit_handler_type(), CommitHandlerType::Unsafe);
    }

    #[test]
    fn test_builtin_providers_keep_conditional_put_capability() {
        // Regression guard for existing providers: the declared capability now
        // drives commit-handler selection, so every provider the default
        // registry ships must keep declaring the conflict-safe put-if-not-exists
        // handler it resolved to before this change. A built-in that silently
        // switched to Unsafe would change the commit handler of every store on
        // that scheme.
        let registry = ObjectStoreRegistry::default();
        let providers = registry
            .providers
            .read()
            .expect("ObjectStoreRegistry lock poisoned");
        assert!(!providers.is_empty());
        for (scheme, provider) in providers.iter() {
            assert_eq!(
                provider.commit_handler(),
                CommitHandlerType::ConditionalPut,
                "built-in provider for `{scheme}` changed its declared commit handler",
            );
        }
    }
}
