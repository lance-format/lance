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
use tokio::sync::Mutex as AsyncMutex;
use url::{Host, Url};

use crate::object_store::WrappingObjectStore;
use crate::object_store::uri_to_url;

use super::{ObjectStore, ObjectStoreParams, tracing::ObjectStoreTracingExt};
use lance_core::error::{Error, Result};

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

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    /// Construct a new object store for the given base path and params.
    ///
    /// **Reentry warning**: implementations MUST NOT recursively call
    /// [`ObjectStoreRegistry::get_store`] for the same `(base_path, params)`
    /// key. The registry coalesces concurrent cold builds via a per-key
    /// async lock held across the call to `new_store`; a re-entrant call
    /// would deadlock waiting on the lock the current task already holds.
    /// Calling `get_store` for a *different* key (e.g. an underlying delegate
    /// store) is safe.
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;

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
    ///
    /// # Userinfo / IPv6 handling
    ///
    /// Overrides that build the prefix from URL authority MUST use
    /// [`sanitized_authority`] (or equivalent) instead of `Url::authority()` —
    /// the latter includes any embedded `userinfo@`, which would leak
    /// credentials into the cache key. The default impl below already does
    /// this. Azure is the only intentional exception: it parses `userinfo`
    /// position as the container name, not credentials.
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok(format!("{}${}", url.scheme(), sanitized_authority(url)))
    }
}

/// Build a cache-key authority component from a URL with two safety properties:
///
/// 1. Strips any `userinfo@` segment so URL-embedded credentials never land in
///    the cache key (which is logged via `Debug` and observed by anything that
///    calls `calculate_object_store_prefix`).
/// 2. Wraps IPv6 hosts in brackets so `[::1]:9000` is unambiguous —
///    `Url::host_str` returns `"::1"` without brackets, which combined with a
///    port would produce the ambiguous string `"::1:9000"`.
///
/// Returns just `host[:port]`. Use this in any
/// [`ObjectStoreProvider::calculate_object_store_prefix`] override that would
/// otherwise reach for `Url::authority()`.
pub fn sanitized_authority(url: &Url) -> String {
    let host = match url.host() {
        Some(Host::Ipv6(addr)) => format!("[{}]", addr),
        Some(Host::Ipv4(addr)) => addr.to_string(),
        Some(Host::Domain(d)) => d.to_string(),
        None => String::new(),
    };
    match url.port() {
        Some(p) => format!("{}:{}", host, p),
        None => host,
    }
}

type CacheKey = (String, ObjectStoreParams);
type BuildLockMap = HashMap<CacheKey, BuildLockEntry>;

/// A per-key cold-build lock plus an explicit count of the tasks currently
/// holding or queued on it.
///
/// The count is the cleanup signal — not `Arc::strong_count` of the mutex.
/// `acquire_build_lock` increments it (under the `build_locks` std mutex) the
/// instant a task claims the entry; [`BuildLockSession::drop`] decrements it on
/// every exit path. Both happen with no intervening `.await`, so the
/// increment/decrement is perfectly paired regardless of when the pending
/// `lock_owned()` future releases its own Arc — which is what makes
/// cancellation-while-queued safe without relying on async drop order.
#[derive(Debug)]
struct BuildLockEntry {
    mutex: Arc<AsyncMutex<()>>,
    /// Number of tasks holding or queued on `mutex`. The entry is removed when
    /// this reaches zero.
    waiters: usize,
}

/// Statistics for the object store registry cache.
///
/// Exposed for in-tree diagnostic consumers (e.g. the
/// `concurrent_open_bench` example, which prints hit/miss counts to prove
/// registry sharing across opens). `#[doc(hidden)]` because this is an
/// observability surface, not a stable API — field names and types may
/// change without a semver bump.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
// Adding fields to this struct must stay source-compatible: external code may
// read it via `stats()` and `..Default::default()`-construct it. `#[non_exhaustive]`
// forbids struct-literal construction outside this crate, so future field
// additions don't break downstream builds.
#[non_exhaustive]
pub struct ObjectStoreRegistryStats {
    /// Number of cache hits (store was already cached and reused).
    pub hits: u64,
    /// Number of cache misses (new store had to be created).
    pub misses: u64,
    /// Number of currently active object stores in the cache.
    pub active_stores: usize,
    /// Number of cache keys with at least one task currently holding or queued
    /// on the per-key cold-build lock.
    ///
    /// Steady-state should be 0 — non-zero indicates either an in-progress
    /// thundering herd (expected during cold start) or a stuck builder.
    pub build_locks: usize,
    /// Number of cold builds that returned an error from the underlying
    /// provider (`ObjectStoreProvider::new_store`).
    ///
    /// Per-call, not per-key: single-flight serializes failures via the
    /// per-key lock but does not cache the `Err` result. N waiters racing
    /// a failing upstream each retry in turn and bump this counter, so a
    /// stuck failure grows roughly linearly with attempts.
    pub build_failures: u64,
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
/// weak references to the object stores, so they are not held onto. Stale
/// entries are reclaimed lazily:
/// - [`Self::active_stores()`] sweeps the whole map.
/// - [`Self::get_store()`] evicts the queried key on miss, and opportunistically
///   prunes other dead entries on every cold-build insert (bounded work, keyed
///   off in-use opens).
#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: RwLock<HashMap<String, Arc<dyn ObjectStoreProvider>>>,
    // Cache of object stores currently in use. We use a weak reference so the
    // cache itself doesn't keep them alive if no object store is actually using
    // it.
    active_stores: RwLock<HashMap<CacheKey, Weak<ObjectStore>>>,
    // Per-key async build locks coalesce concurrent cold builds for the same
    // key into one `provider.new_store` call (avoids the thundering herd on
    // S3-style providers whose client construction is expensive).
    build_locks: std::sync::Mutex<BuildLockMap>,
    // Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
    build_failures: AtomicU64,
}

/// RAII session for the cold-build path. On drop it releases the per-key async
/// lock guard, then calls `release_build_lock` to drop this task's
/// [`BuildLockEntry::waiters`] hold (which removes the entry at zero). Releasing
/// the guard first lets a queued waiter wake before the build-locks map is
/// touched.
///
/// It is constructed *before* awaiting `lock_owned()`, with no `.await` between
/// `acquire_build_lock`'s increment and the construction. So a waiter cancelled
/// while still *queued* on `lock_owned()` — guard never assigned — still drops a
/// live session and decrements; without that ordering the entry would be
/// orphaned until the next open for the same key. (Cleanup tracks the explicit
/// count, not `Arc::strong_count`, so it is also immune to the drop order of the
/// pending `lock_owned()` future — see [`BuildLockEntry`].)
///
/// Drop runs on every unwinding exit — success, error, unwinding panic,
/// cancellation. (`panic = "abort"` skips destructors, but the process is dying,
/// so there is no surviving registry to leak into; see
/// `test_get_store_cleans_build_locks_after_panic`.)
///
/// `guard` is `None` until the lock is acquired. The owned `CacheKey` keeps the
/// drop independent of any local variable's lifetime.
struct BuildLockSession<'a> {
    registry: &'a ObjectStoreRegistry,
    cache_key: CacheKey,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl Drop for BuildLockSession<'_> {
    fn drop(&mut self) {
        // Release the async lock first so a queued waiter can wake before we
        // touch the build-locks map, then drop this task's waiter hold (which
        // removes the entry once the last waiter leaves).
        self.guard.take();
        self.registry.release_build_lock(&self.cache_key);
    }
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
            build_locks: std::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            build_failures: AtomicU64::new(0),
        }
    }

    /// Get the object store provider for a given scheme.
    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(scheme)
            .cloned()
    }

    /// Get a list of all active object stores.
    ///
    /// Calling this will also clean up any weak references to object stores that
    /// are no longer valid.
    ///
    /// **Caution**: the returned `Arc<ObjectStore>` instances may carry
    /// per-tenant credential state (S3 access keys, Azure SAS tokens, etc.).
    /// Callers that hand these stores out across trust boundaries are
    /// responsible for filtering — the registry itself does not know which
    /// store belongs to which tenant.
    pub fn active_stores(&self) -> Vec<Arc<ObjectStore>> {
        let mut found_inactive = false;
        let output = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
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
                .unwrap_or_else(|p| p.into_inner());
            cache_lock.retain(|_, weak| weak.upgrade().is_some());
        }
        output
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns counters for hits, misses, currently active stores,
    /// in-flight build locks, and accumulated build failures — see
    /// [`ObjectStoreRegistryStats`]. Useful for detecting configuration
    /// issues that cause excessive cache misses (e.g., storage options
    /// that vary per-request).
    ///
    /// `#[doc(hidden)]`: paired with [`ObjectStoreRegistryStats`] — this is
    /// an unstable observability surface kept `pub` only to support the
    /// in-tree `concurrent_open_bench` example.
    #[doc(hidden)]
    pub fn stats(&self) -> ObjectStoreRegistryStats {
        let active_stores = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .filter(|w| w.strong_count() > 0)
            .count();
        let build_locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        ObjectStoreRegistryStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            active_stores,
            build_locks,
            build_failures: self.build_failures.load(Ordering::Relaxed),
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
    ///
    /// Concurrent cold builds for the same key are coalesced via a per-key
    /// async lock: the first task builds the store, all others wait then
    /// re-check the cache and observe the freshly populated entry.
    ///
    /// On build *failure*, the cache is not populated, so each waiter retries
    /// in turn (serialized — not parallel-amplified, but not deduplicated
    /// either). Transient errors thus surface to operators rather than being
    /// masked by stale-error reuse.
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

        // Fast path: cache hit avoids both the std mutex and the async lock.
        if let Some(store) = self.lookup_cached(&cache_key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(store);
        }

        // Cold path: per-key single-flight. `acquire_build_lock` registers this
        // task as a waiter; the RAII `BuildLockSession` releases that hold on
        // every exit, bounding the map. The session is built before the
        // `lock_owned()` await (see `BuildLockSession`) so a waiter cancelled
        // while queued still cleans up.
        let lock = self.acquire_build_lock(&cache_key);
        let mut session = BuildLockSession {
            registry: self,
            cache_key: cache_key.clone(),
            guard: None,
        };
        // Cancellation point: `session` already exists, so its Drop cleans up
        // even if we are cancelled here before the guard is assigned.
        session.guard = Some(lock.lock_owned().await);

        // Re-check after acquiring the lock — coalesced waiters become hits.
        if let Some(store) = self.lookup_cached(&cache_key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            log::debug!(
                "ObjectStoreRegistry: coalesced wait hit for cache_key path={:?}",
                cache_key.0,
            );
            return Ok(store);
        }

        // `fetch_add` returns the prior value: this attempt's 0-based
        // sequence index. Use it (not a separate counter) to gate the
        // opportunistic sweep below: cadence is approximate because cold
        // attempts that fail still bump `misses` but skip the sweep block,
        // so a failure at a sweep-eligible index skips that sweep cycle
        // entirely; the next candidate is the next sweep-eligible cold
        // attempt that succeeds. That's fine — the sweep is a footprint
        // trim, not a correctness invariant. The raw `fetch_add` value is
        // used as-is (no `+ 1`), so there is no counter arithmetic of our
        // own that could overflow.
        let cold_attempt_index = self.misses.fetch_add(1, Ordering::Relaxed);
        log::debug!(
            "ObjectStoreRegistry: cold build starting for cache_key path={:?}",
            cache_key.0,
        );

        let cached = match self
            .build_store(provider, base_path, params, &cache_path)
            .await
        {
            Ok(store) => store,
            Err(e) => {
                self.build_failures.fetch_add(1, Ordering::Relaxed);
                // Intentionally no log here: provider error Display impls can
                // surface raw URLs whose username or query-string component
                // may carry credentials (SAS tokens, access keys), and `Url`'s
                // own `Display` only masks the password. The full error goes
                // to the caller via `Err(e)`; operators wanting an aggregate
                // signal can scrape `stats().build_failures`.
                return Err(e);
            }
        };

        // Amortized opportunistic sweep: gate the O(n) `retain` behind a
        // mod-N counter so a burst of cold builds with distinct URIs costs
        // O(n) work in aggregate, not O(n²). N=64 is small enough that the
        // map's footprint stays roughly proportional to live opens, large
        // enough that bursty cold-open paths don't pay sweep cost on every
        // insert. Per-key `lookup_cached` and full sweep via `active_stores()`
        // remain unchanged; this is purely a per-insert cost lever.
        //
        // Cadence is "every 64th cold attempt": with a 0-based index the
        // eligible attempts are indices 63, 127, 191, … so the first sweep
        // fires at the 64th cold build, not the very first, and a
        // near-empty map never pays the retain cost on its first cold
        // build.
        const SWEEP_INTERVAL: u64 = 64;
        let should_sweep = cold_attempt_index % SWEEP_INTERVAL == SWEEP_INTERVAL - 1;
        {
            let mut cache_lock = self
                .active_stores
                .write()
                .unwrap_or_else(|p| p.into_inner());
            // Safe under the write lock: no other writer can resurrect a weak
            // ref between `retain` and `insert`, and `insert` for `cache_key`
            // overwrites whatever `retain` left (live or dead) for that key.
            if should_sweep {
                cache_lock.retain(|_, weak| weak.upgrade().is_some());
            }
            cache_lock.insert(cache_key, Arc::downgrade(&cached));
        }
        Ok(cached)
    }

    /// Acquire (or create) the per-key async build lock for `cache_key`,
    /// registering this task as a waiter.
    ///
    /// Increments [`BuildLockEntry::waiters`] under the `build_locks` std
    /// mutex; the caller MUST pair this with exactly one `release_build_lock`,
    /// which the RAII [`BuildLockSession`] guarantees on every exit path.
    ///
    /// On poison: recovers via `into_inner()` rather than bricking the
    /// registry. The protected data is just a HashMap of build-lock entries;
    /// any stale entry left by a panic is reclaimed by `BuildLockSession`.
    fn acquire_build_lock(&self, cache_key: &CacheKey) -> Arc<AsyncMutex<()>> {
        let mut locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = locks
            .entry(cache_key.clone())
            .or_insert_with(|| BuildLockEntry {
                mutex: Arc::new(AsyncMutex::new(())),
                waiters: 0,
            });
        entry.waiters += 1;
        entry.mutex.clone()
    }

    /// Release this task's hold on the per-key build lock for `cache_key`,
    /// removing the entry once the last waiter leaves.
    ///
    /// Decrements [`BuildLockEntry::waiters`] and removes the entry at zero. The
    /// `build_locks` std mutex makes the decrement-and-maybe-remove atomic
    /// against concurrent `acquire_build_lock` calls, so a new waiter can't be
    /// lost between the two.
    ///
    /// The map's worst-case size is the number of distinct cache keys with an
    /// outstanding cold open, which the I/O concurrency limits already cap —
    /// RAII pairs every `acquire_build_lock` with exactly one `release_build_lock`.
    fn release_build_lock(&self, cache_key: &CacheKey) {
        let mut locks = self
            .build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // RAII pairs every `acquire_build_lock` with exactly one release, so
        // the entry must exist and its waiter count must be positive here. A
        // violation means the acquire/release accounting is broken — surface
        // it loudly in debug/test, and fall back safely (saturating, no-op on
        // a missing entry) in release so a bug can't underflow `usize` or
        // panic in production.
        if let Some(entry) = locks.get_mut(cache_key) {
            debug_assert!(
                entry.waiters > 0,
                "release_build_lock: waiter count already zero for an existing entry — \
                 acquire/release pairing is broken",
            );
            entry.waiters = entry.waiters.saturating_sub(1);
            if entry.waiters == 0 {
                locks.remove(cache_key);
            }
        } else {
            debug_assert!(
                false,
                "release_build_lock: no build-lock entry for cache_key — \
                 release called without a matching acquire",
            );
        }
    }

    #[cfg(test)]
    fn build_locks_len(&self) -> usize {
        self.build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Current waiter count for `cache_key`, or `None` if no entry exists.
    /// Lets tests observe the per-key acquire/release count directly — the
    /// signal that drives cleanup — rather than only the map's key count.
    #[cfg(test)]
    fn build_lock_waiters(&self, cache_key: &CacheKey) -> Option<usize> {
        self.build_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(cache_key)
            .map(|entry| entry.waiters)
    }

    /// Look up a cached store by key, evicting the entry if its weak ref is
    /// no longer upgradable.
    fn lookup_cached(&self, cache_key: &CacheKey) -> Option<Arc<ObjectStore>> {
        let maybe_weak = self
            .active_stores
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(cache_key)
            .cloned();
        let weak = maybe_weak?;
        if let Some(store) = weak.upgrade() {
            return Some(store);
        }
        // Stale weak; evict under the write lock.
        let mut cache_lock = self
            .active_stores
            .write()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(weak) = cache_lock.get(cache_key)
            && weak.upgrade().is_none()
        {
            cache_lock.remove(cache_key);
        }
        None
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
            build_locks: std::sync::Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            build_failures: AtomicU64::new(0),
        }
    }
}

impl ObjectStoreRegistry {
    /// Add a new object store provider to the registry. The provider will be used
    /// in [`Self::get_store()`] when a URL is passed with a matching scheme.
    pub fn insert(&self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers
            .write()
            .unwrap_or_else(|p| p.into_inner())
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

    /// `userinfo@host` (URL-embedded credentials) MUST be stripped from the
    /// cache-key prefix so the secret never lands in HashMap Debug logs or
    /// downstream state. Two URLs that differ only in `userinfo` collapse to
    /// the same prefix; multi-tenant isolation has to come from
    /// `ObjectStoreParams` (storage_options / accessor), not URL embedding.
    #[test]
    fn test_calculate_object_store_prefix_strips_userinfo() {
        let provider = DummyProvider;
        let with_creds = Url::parse("dummy://user:s3cret@blah/path").unwrap();
        let without = Url::parse("dummy://blah/path").unwrap();
        let with_prefix = provider
            .calculate_object_store_prefix(&with_creds, None)
            .unwrap();
        assert_eq!("dummy$blah", with_prefix);
        assert_eq!(
            with_prefix,
            provider
                .calculate_object_store_prefix(&without, None)
                .unwrap(),
        );
        // Defense in depth: assert the secret literally cannot appear in the
        // key, regardless of formatting changes.
        assert!(!with_prefix.contains('@'));
        assert!(!with_prefix.contains("s3cret"));
        assert!(!with_prefix.contains("user"));
    }

    #[test]
    fn test_calculate_object_store_prefix_keeps_port() {
        let provider = DummyProvider;
        let url = Url::parse("dummy://host:9000/path").unwrap();
        assert_eq!(
            "dummy$host:9000",
            provider.calculate_object_store_prefix(&url, None).unwrap()
        );
    }

    #[test]
    fn test_sanitized_authority_brackets_ipv6() {
        // Url::host_str returns "::1" without brackets; combined with a port,
        // the naive `format!("{}:{}", host, port)` would produce the
        // ambiguous string "::1:9000". sanitized_authority restores brackets.
        let url = Url::parse("dummy://[::1]:9000/path").unwrap();
        assert_eq!("[::1]:9000", sanitized_authority(&url));
        let url_no_port = Url::parse("dummy://[2001:db8::1]/path").unwrap();
        assert_eq!("[2001:db8::1]", sanitized_authority(&url_no_port));
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

        // The cache reused the entry for the second call (hits == 1 above);
        // when the same params hit the same cache slot, the returned Arcs
        // point to the same underlying ObjectStore.
        assert!(Arc::ptr_eq(&stores[0], &stores[1]));
    }

    /// A provider that counts `new_store` invocations and yields explicitly,
    /// so concurrent callers reliably contend on the registry's build lock.
    #[derive(Debug, Default)]
    struct CountingProvider {
        builds: AtomicU64,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for CountingProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            // Force a yield + delay so other tasks reach `get_store` and queue
            // on the build lock before this build completes.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            memory::MemoryStoreProvider
                .new_store(base_path, params)
                .await
        }
    }

    /// Concurrent cold-misses for the same key must coalesce into a single
    /// `provider.new_store` invocation, with all callers receiving the same Arc.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_get_store_coalesces_concurrent_misses() {
        let provider = Arc::new(CountingProvider::default());
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("counting", provider.clone());

        let url = Url::parse("counting://test").unwrap();
        let params = ObjectStoreParams::default();

        let n = 16;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let registry = registry.clone();
            let url = url.clone();
            let params = params.clone();
            handles.push(tokio::spawn(async move {
                registry.get_store(url, &params).await.unwrap()
            }));
        }
        let mut stores = Vec::with_capacity(n);
        for h in handles {
            stores.push(h.await.unwrap());
        }

        assert_eq!(
            provider.builds.load(Ordering::Relaxed),
            1,
            "single-flight must collapse N concurrent misses into 1 build"
        );
        // All concurrent callers share the same cached Arc — single-flight
        // produced exactly one ObjectStore and the registry handed it out
        // to every waiter.
        for s in &stores[1..] {
            assert!(Arc::ptr_eq(&stores[0], s));
        }
        let s = registry.stats();
        assert_eq!((s.misses, s.hits as usize, s.active_stores), (1, n - 1, 1));
    }

    /// After a successful build, the per-key build lock must be removed so
    /// `build_locks` does not grow unbounded across many distinct keys.
    #[tokio::test]
    async fn test_get_store_cleans_build_locks_after_use() {
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams::default();

        for i in 0..5 {
            let url = Url::parse(&format!("memory://cleanup-{}", i)).unwrap();
            let _store = registry.get_store(url, &params).await.unwrap();
        }
        assert_eq!(
            registry.build_locks_len(),
            0,
            "build_locks must be empty once all builds settle"
        );
    }

    /// A provider whose `new_store` panics, used to verify the RAII cleanup
    /// guard reclaims build_locks entries on panic.
    #[derive(Debug, Default)]
    struct PanickingProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for PanickingProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            panic!("boom from provider")
        }
    }

    /// A provider that returns an `Err` after a yield, simulating a normal
    /// (non-panic) build failure such as bad credentials or a network error.
    /// Counts invocations so the test can assert failures are serialized but
    /// retried once per caller.
    #[derive(Debug, Default)]
    struct FailingProvider {
        builds: AtomicU64,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for FailingProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            // Yield + delay so concurrent callers reach the build lock and
            // queue behind us — this is the configuration that exercises
            // the coalesced-failure path in `get_store`.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Err(Error::invalid_input("simulated cold-build failure"))
        }
    }

    /// Build-Err (non-panic) path under coalesced contention asserts the
    /// shipped behavior of PR#1's single-flight: failures **serialize via
    /// the per-key lock but are not cached**. Each of N waiters acquires
    /// the lock in turn, re-checks the (still-empty) cache, and retries
    /// the build. The lock keeps failures from *fanning out in parallel*
    /// (so an upstream that costs T ms sees ~N·T total, not T concurrent),
    /// but it does not collapse them into one call.
    ///
    /// Concretely this test pins:
    ///   1. `provider.new_store` fires once per caller (no Err coalescing),
    ///   2. every concurrent caller surfaces an `Err`,
    ///   3. `build_failures` increments per failed call (not per key),
    ///   4. `active_stores` stays 0 — failed builds must never be cached,
    ///   5. `build_locks` drains back to 0 via the RAII cleanup, even on
    ///      the Err return path.
    ///
    /// Caching errors (with TTL or until-eviction) is a separate design
    /// point. This test exists to lock in the current contract so a future
    /// "let's cache Err for 100ms" change is forced to update this guard
    /// deliberately rather than land silently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_get_store_serializes_build_failures_without_caching() {
        let provider = Arc::new(FailingProvider::default());
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("failing", provider.clone());

        let url = Url::parse("failing://x").unwrap();
        let params = ObjectStoreParams::default();

        let n = 8;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let registry = registry.clone();
            let url = url.clone();
            let params = params.clone();
            handles.push(tokio::spawn(async move {
                registry.get_store(url, &params).await
            }));
        }

        let mut err_count = 0;
        for h in handles {
            let result = h.await.expect("task must not panic");
            let err = result.expect_err("every waiter must surface Err");
            assert!(
                matches!(err, Error::InvalidInput { .. }),
                "expected InvalidInput from the failing provider, got: {err:?}",
            );
            assert!(
                err.to_string().contains("simulated cold-build failure"),
                "error must carry the provider's message, got: {err}",
            );
            err_count += 1;
        }
        assert_eq!(err_count, n, "all N callers must observe an error");

        assert_eq!(
            provider.builds.load(Ordering::Relaxed) as usize,
            n,
            "shipped behavior: failures are not cached, so each waiter retries the build",
        );

        let s = registry.stats();
        assert_eq!(
            s.build_failures as usize, n,
            "build_failures counts every failed call (per-call, not per-key)",
        );
        assert_eq!(
            s.active_stores, 0,
            "failed builds must never populate active_stores",
        );
        assert_eq!(
            s.build_locks, 0,
            "RAII cleanup must drain build_locks even on Err path",
        );
    }

    /// If `new_store` panics, the RAII guard must still remove the build_locks
    /// entry — otherwise a single bad URI would leak a lock per failed open.
    ///
    /// Relies on the default `panic = "unwind"` profile: the panic unwinds
    /// the spawned task and `Drop` runs on the way out, which is what triggers
    /// `BuildLockSession`. With `panic = "abort"`, the process dies before
    /// cleanup can run — that's a stricter regime where this test (and the
    /// leak it guards against) becomes moot.
    #[tokio::test]
    async fn test_get_store_cleans_build_locks_after_panic() {
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("boom", Arc::new(PanickingProvider));
        let url = Url::parse("boom://x").unwrap();
        let params = ObjectStoreParams::default();

        let registry_for_task = registry.clone();
        let result =
            tokio::task::spawn(async move { registry_for_task.get_store(url, &params).await })
                .await;
        assert!(
            result.as_ref().err().is_some_and(|e| e.is_panic()),
            "task must propagate the panic"
        );
        assert_eq!(
            registry.build_locks_len(),
            0,
            "RAII guard must clean up the build_locks entry on panic"
        );
    }

    /// A waiter cancelled while still *queued* on the per-key build lock —
    /// before its `OwnedMutexGuard` is acquired — must still reclaim the
    /// `build_locks` entry on drop. This pins that state directly: the
    /// [`BuildLockSession`] is built with `guard: None`, exactly as it exists
    /// for a task cancelled inside `lock_owned().await`. The lingering
    /// `Arc<AsyncMutex<()>>` (`_held`) stands in for the Arc that the pending
    /// `lock_owned()` future keeps alive — cleanup must not be fooled by it,
    /// because it keys off the explicit waiter count, not `Arc::strong_count`.
    #[tokio::test]
    async fn test_build_lock_session_reclaims_entry_when_cancelled_before_guard() {
        let registry = ObjectStoreRegistry::empty();
        let cache_key = ("memory$queued".to_string(), ObjectStoreParams::default());

        // Mirror the cold path: register a waiter and take the lock Arc, but
        // never acquire the guard.
        let _held = registry.acquire_build_lock(&cache_key);
        assert_eq!(
            registry.build_locks_len(),
            1,
            "acquiring the build lock must register the entry"
        );

        let session = BuildLockSession {
            registry: &registry,
            cache_key,
            guard: None,
        };
        drop(session);

        assert_eq!(
            registry.build_locks_len(),
            0,
            "dropping the session for a still-queued (guard-less) waiter must \
             reclaim the entry, even while another Arc to the lock is alive"
        );
    }

    /// A provider that signals when its build starts and blocks until released,
    /// pinning a builder in flight so other callers queue behind it.
    #[derive(Debug)]
    struct BlockingProvider {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait::async_trait]
    impl ObjectStoreProvider for BlockingProvider {
        async fn new_store(
            &self,
            base_path: Url,
            params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            self.started.notify_one();
            self.release.notified().await;
            memory::MemoryStoreProvider
                .new_store(base_path, params)
                .await
        }
    }

    /// End-to-end cancellation safety: a waiter cancelled while queued behind
    /// an in-flight builder must release its hold on the build lock. The test
    /// observes the per-key waiter count directly so the cancellation's effect
    /// is visible *before* the builder finishes — otherwise the builder's own
    /// cleanup would mask whether the cancelled waiter ever decremented.
    ///
    /// Sequence: builder A registers (count 1) and parks; waiter B queues
    /// (count 2); B is cancelled mid-wait → count must drop back to 1; A
    /// completes → entry fully reclaimed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_get_store_cleans_build_locks_when_queued_waiter_cancelled() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let provider = Arc::new(BlockingProvider {
            started: started.clone(),
            release: release.clone(),
        });
        let registry = Arc::new(ObjectStoreRegistry::empty());
        registry.insert("blocking", provider);

        let url = Url::parse("blocking://same").unwrap();
        let params = ObjectStoreParams::default();
        // The cache key the cold path computes for this URL: default prefix is
        // `scheme$sanitized_authority`, paired with the (default) params.
        let cache_key = ("blocking$same".to_string(), params.clone());

        // Builder A acquires the lock and parks inside `new_store`.
        let (ra, ua, pa) = (registry.clone(), url.clone(), params.clone());
        let builder = tokio::spawn(async move { ra.get_store(ua, &pa).await });
        started.notified().await;
        assert_eq!(
            registry.build_lock_waiters(&cache_key),
            Some(1),
            "builder A must be registered as the sole waiter"
        );

        // Waiter B targets the same key, so it queues on `lock_owned()` behind
        // A (it cannot acquire the guard — A holds it). Wait deterministically
        // for B to register rather than racing a fixed sleep; a regression that
        // never queues fails here loudly instead of hanging.
        let (rb, ub, pb) = (registry.clone(), url.clone(), params.clone());
        let waiter = tokio::spawn(async move { rb.get_store(ub, &pb).await });
        let mut queued = false;
        for _ in 0..200 {
            if registry.build_lock_waiters(&cache_key) == Some(2) {
                queued = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            queued,
            "waiter B did not queue on the build lock within timeout"
        );

        // Cancel B while it is parked, then confirm its session decremented the
        // count back to 1. This is the assertion the old strong-count cleanup
        // could not satisfy: a waiter cancelled mid-await left its hold behind.
        waiter.abort();
        assert!(
            waiter.await.unwrap_err().is_cancelled(),
            "waiter must be cancelled, not completed"
        );
        assert_eq!(
            registry.build_lock_waiters(&cache_key),
            Some(1),
            "B's cancellation must decrement the waiter count, leaving only A"
        );

        // Release A; it finishes the build and drops its session, draining the
        // entry entirely.
        release.notify_one();
        assert!(
            builder.await.unwrap().is_ok(),
            "builder must complete successfully"
        );
        assert_eq!(
            registry.build_lock_waiters(&cache_key),
            None,
            "after the builder completes, the build-lock entry must be reclaimed"
        );
        assert_eq!(
            registry.build_locks_len(),
            0,
            "no build-lock entries may leak",
        );
    }

    /// Regression: two callers resolving the same cache key must receive
    /// the same cached `Arc<ObjectStore>`. Single-flight correctness depends
    /// on this — see `test_get_store_coalesces_concurrent_misses` for the
    /// concurrent variant.
    #[tokio::test]
    async fn test_get_store_returns_shared_arc_on_hit() {
        let registry = ObjectStoreRegistry::default();
        let params = ObjectStoreParams::default();
        let url = Url::parse("memory://shared-arc").unwrap();

        let a = registry.get_store(url.clone(), &params).await.unwrap();
        let b = registry.get_store(url, &params).await.unwrap();

        let s = registry.stats();
        assert_eq!((s.misses, s.hits, s.active_stores), (1, 1, 1));
        assert!(
            Arc::ptr_eq(&a, &b),
            "cache hit must return the same Arc as the cold-build caller"
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
