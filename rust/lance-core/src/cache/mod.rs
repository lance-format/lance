// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance cache system.
//!
//! ## For cache users
//!
//! Use [`LanceCache`] (or [`WeakLanceCache`]) to store and retrieve typed
//! values. Define a [`CacheKey`] (or [`UnsizedCacheKey`] for trait objects) to
//! describe what you're caching and its type.
//!
//! To make a value type serializable (so persistent backends can store it),
//! implement [`CacheCodecImpl`] on the type, then override [`CacheKey::codec`]:
//!
//! ```ignore
//! impl CacheCodecImpl for MyData {
//!     fn serialize(&self, w: &mut dyn Write) -> Result<()> { /* ... */ }
//!     fn deserialize(data: &Bytes) -> Result<Self> { /* ... */ }
//! }
//!
//! impl CacheKey for MyDataKey {
//!     type ValueType = MyData;
//!     fn key(&self) -> Cow<'_, str> { /* ... */ }
//!     fn type_name() -> &'static str { "MyData" }
//!     fn codec() -> Option<CacheCodec> {
//!         Some(CacheCodec::from_impl::<MyData>())
//!     }
//! }
//! ```
//!
//! ## For backend implementors
//!
//! Implement [`CacheBackend`] to provide a custom storage layer (disk, Redis,
//! etc.). Backends receive opaque, fixed-size [`InternalCacheKey`] values and
//! type-erased [`CacheEntry`] values. The typed wrapping is handled by
//! [`LanceCache`]. See the [`backend`] module for migration details.
//!
//! ## Serialization flow
//!
//! When a [`CacheKey`] provides a codec via [`CacheKey::codec`]:
//!
//! 1. [`LanceCache`] wraps the [`CacheCodec`] and passes it to the backend
//!    alongside the entry on `insert` and `get` calls.
//! 2. In-memory backends (like [`MokaCacheBackend`]) ignore the codec.
//! 3. Persistent backends use `codec.serialize(entry, writer)` on insert and
//!    `codec.deserialize(reader)` on get to persist entries across restarts.

pub mod backend;
mod backend_metrics;
mod backend_uri;
pub mod codec;
mod diagnostics;
mod entry_io;
mod key;
mod moka;
mod quick;
mod registry;
pub mod telemetry;

pub use backend::{CacheBackend, CacheEntry, CacheLoader};
pub use backend_uri::{build_from_uri, parse_backend_uri};
pub use codec::{
    CacheCodec, CacheCodecImpl, CacheDecode, CacheMissReason, MAGIC, has_cache_envelope,
};
pub use diagnostics::{
    CacheActivity, CacheBackendDiagnostics, CacheBackendKind, CacheByTypeDiagnostics,
    CacheDiagnostics, CacheLoadOrigin, CacheLoadOutcome, CacheMetricsKind, CacheOccupancyByType,
    CacheOperationContext, CacheSnapshotMode, CacheTypeActivity, CacheTypeOccupancy,
    CacheWarmActivity, MAX_CACHE_TYPE_SERIES,
};
pub use entry_io::{CacheEntryReader, CacheEntryWriter};
pub use key::{CACHE_KEY_FORMAT, CacheKeySchema, CacheNamespace, InternalCacheKey, KeyBuilder};
pub use moka::{MokaCacheBackend, MokaCacheBackendBuilder};
pub use quick::{QuickCacheBackend, QuickCacheShardPolicy, recommended_cache_shards};
pub use registry::{BackendBuildFn, BackendConfig, build_from_config, register_backend};
#[cfg(feature = "metrics")]
pub use telemetry::{describe_metrics, histogram_bounds, refresh_metrics};

use std::any::TypeId;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{
    Arc, RwLock, Weak,
    atomic::{AtomicU64, Ordering},
};

use futures::Future;

use crate::{Error, Result};

pub use crate::deepsize::{Context, DeepSizeOf};

// ---------------------------------------------------------------------------
// CacheKey / UnsizedCacheKey — typed key traits for cache users
// ---------------------------------------------------------------------------

/// Typed cache key for sized value types.
///
/// Existing implementations can continue returning a logical string from
/// [`key`](Self::key). Performance-sensitive implementations should also
/// provide a stable schema and stream typed fields through
/// [`write_key`](Self::write_key), avoiding construction of that string.
///
/// # Example
///
/// ```ignore
/// struct MyKey { id: u64 }
///
/// impl CacheKey for MyKey {
///     type ValueType = MyData;
///     fn key(&self) -> Cow<'_, str> { self.id.to_string().into() }
///     fn type_name() -> &'static str { "MyData" }
/// }
/// ```
pub trait CacheKey {
    type ValueType: 'static;

    fn key(&self) -> Cow<'_, str>;

    /// Short, stable string identifying this value type.
    ///
    /// Two `CacheKey` impls that store different `ValueType`s **must** return
    /// different type names.
    ///
    /// Use a short literal (e.g. `"Vec<IndexMetadata>"`), not
    /// `std::any::type_name` — the latter is not guaranteed stable across
    /// compiler versions or build configurations.
    fn type_name() -> &'static str;

    /// Stable identity included in the physical key.
    ///
    /// The compatibility default preserves existing implementations by using
    /// their author-assigned [`type_name`](Self::type_name).
    fn stable_type_id() -> &'static str {
        Self::type_name()
    }

    /// Versioned schema for the logical key fields.
    fn schema() -> CacheKeySchema {
        CacheKeySchema::LEGACY_TEXT
    }

    /// Stream the logical key fields into the canonical key builder.
    ///
    /// The compatibility default hashes the existing string key. In-tree hot
    /// paths override this with typed, allocation-free field encoding.
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.key().as_ref());
    }

    /// Optional codec for serializing/deserializing this key's value type.
    ///
    /// Returns `None` by default. Cache backends that support persistence
    /// (e.g. disk-backed caches) use this to serialize entries on insert and
    /// deserialize on get. Types without a codec will only be stored in-memory.
    ///
    /// [`CacheCodec`] is `Copy` (two plain function pointers), so returning it
    /// by value is cheap — no allocation needed.
    fn codec() -> Option<CacheCodec> {
        None
    }
}

/// Like [`CacheKey`] but for unsized value types (e.g. `dyn Trait`).
///
/// The cache wraps values in an extra `Arc` layer internally; callers pass
/// and receive `Arc<T>` where `T: ?Sized`.
///
/// Unsized cache entries are always in-memory only (no serialization codec).
/// For serializable entries, use a sized [`CacheKey`] instead.
pub trait UnsizedCacheKey {
    type ValueType: 'static + ?Sized;

    fn key(&self) -> Cow<'_, str>;

    /// Short, stable string identifying this value type.
    /// See [`CacheKey::type_name`] for requirements.
    fn type_name() -> &'static str;

    /// Stable identity included in the physical key.
    fn stable_type_id() -> &'static str {
        Self::type_name()
    }

    /// Versioned schema for the logical key fields.
    fn schema() -> CacheKeySchema {
        CacheKeySchema::LEGACY_TEXT
    }

    /// Stream the logical key fields into the canonical key builder.
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.key().as_ref());
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Size of a cached `Arc<T>`, accounting for the Arc overhead (two atomic counters).
fn cache_entry_size<T: DeepSizeOf + ?Sized>(value: &T) -> usize {
    value.deep_size_of() + std::mem::size_of::<std::sync::atomic::AtomicUsize>() * 2
}

type CacheEntrySizeAccessor = fn(&CacheEntry, &mut Context) -> Option<usize>;

fn cache_entry_size_with_context<T>(entry: &CacheEntry, context: &mut Context) -> Option<usize>
where
    T: DeepSizeOf + Send + Sync + 'static,
{
    let value = entry.downcast_ref::<T>()?;
    let entry_ptr = Arc::as_ptr(entry) as *const () as usize;
    if !context.mark_seen(entry_ptr) {
        return Some(0);
    }
    Some(
        std::mem::size_of_val(value)
            + value.deep_size_of_children(context)
            + std::mem::size_of::<std::sync::atomic::AtomicUsize>() * 2,
    )
}

#[derive(Debug)]
struct CacheState {
    backend: Arc<dyn CacheBackend>,
    #[cfg(feature = "metrics")]
    cache_kind: CacheMetricsKind,
    #[cfg(feature = "metrics")]
    backend_kind: CacheBackendKind,
    hits: AtomicU64,
    misses: AtomicU64,
    hits_at_clear: AtomicU64,
    misses_at_clear: AtomicU64,
    activity: diagnostics::ActivityCounters,
    type_activity: diagnostics::TypeActivityRegistry,
    entry_size_accessors: RwLock<HashMap<TypeId, CacheEntrySizeAccessor>>,
}

impl CacheState {
    fn new(backend: Arc<dyn CacheBackend>, cache_kind: CacheMetricsKind) -> Self {
        let backend_kind = backend.diagnostics().kind;
        telemetry::register_backend(&backend);
        Self {
            backend,
            #[cfg(feature = "metrics")]
            cache_kind,
            #[cfg(feature = "metrics")]
            backend_kind,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hits_at_clear: AtomicU64::new(0),
            misses_at_clear: AtomicU64::new(0),
            activity: Default::default(),
            type_activity: diagnostics::TypeActivityRegistry::new(cache_kind, backend_kind),
            entry_size_accessors: RwLock::new(HashMap::new()),
        }
    }

    #[inline]
    fn record_hit(&self, by_type: diagnostics::TypeActivityHandle) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.type_activity.record_hit(by_type);
        #[cfg(feature = "metrics")]
        {
            self.type_activity
                .note_event(by_type, self.cache_kind, self.backend_kind);
            let (aggregate_key, type_key) = self.type_activity.lookup_metric_keys(by_type, true);
            telemetry::lookup(aggregate_key, type_key);
        }
    }

    #[inline]
    fn record_miss(&self, by_type: diagnostics::TypeActivityHandle) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.type_activity.record_miss(by_type);
        #[cfg(feature = "metrics")]
        {
            self.type_activity
                .note_event(by_type, self.cache_kind, self.backend_kind);
            let (aggregate_key, type_key) = self.type_activity.lookup_metric_keys(by_type, false);
            telemetry::lookup(aggregate_key, type_key);
        }
    }

    #[inline]
    fn record_lookup_error(
        &self,
        by_type: diagnostics::TypeActivityHandle,
        reason: &'static str,
        origin: CacheLoadOrigin,
    ) {
        self.activity.lookup_errors.fetch_add(1, Ordering::Relaxed);
        self.type_activity.record_lookup_error(by_type);
        if origin == CacheLoadOrigin::Warm {
            self.activity.record_warm_error(
                self.type_activity.activity(by_type),
                self.type_activity.event_metric_keys(by_type),
            );
        }
        #[cfg(feature = "metrics")]
        {
            self.type_activity
                .note_event(by_type, self.cache_kind, self.backend_kind);
            self.type_activity
                .event_metric_keys(by_type)
                .lookup_error(reason);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = reason;
    }

    #[inline]
    fn record_type_mismatch(
        &self,
        by_type: diagnostics::TypeActivityHandle,
        origin: CacheLoadOrigin,
    ) {
        self.activity
            .type_mismatches
            .fetch_add(1, Ordering::Relaxed);
        self.type_activity.record_type_mismatch(by_type);
        self.record_lookup_error(by_type, "type_mismatch", origin);
    }

    fn record_warm_attempt(
        &self,
        by_type: diagnostics::TypeActivityHandle,
        origin: CacheLoadOrigin,
    ) {
        if origin != CacheLoadOrigin::Warm {
            return;
        }
        #[cfg(feature = "metrics")]
        self.type_activity
            .note_event(by_type, self.cache_kind, self.backend_kind);
        self.activity.record_warm_attempt(
            self.type_activity.activity(by_type),
            self.type_activity.event_metric_keys(by_type),
        );
    }

    fn record_warm_hit(&self, by_type: diagnostics::TypeActivityHandle, origin: CacheLoadOrigin) {
        if origin == CacheLoadOrigin::Warm {
            self.activity.record_warm_hit(
                self.type_activity.activity(by_type),
                self.type_activity.event_metric_keys(by_type),
            );
        }
    }

    fn record_warm_insert(
        &self,
        by_type: diagnostics::TypeActivityHandle,
        origin: CacheLoadOrigin,
        size_bytes: usize,
    ) {
        if origin == CacheLoadOrigin::Warm {
            self.activity.record_warm_insert(
                self.type_activity.activity(by_type),
                self.type_activity.event_metric_keys(by_type),
                size_bytes.try_into().unwrap_or(u64::MAX),
            );
        }
    }

    fn record_clear(&self, hits: u64, misses: u64) {
        // Concurrent clears can publish samples in a different order. An older
        // sample must not restore activity excluded by a more recent clear.
        self.hits_at_clear.fetch_max(hits, Ordering::Relaxed);
        self.misses_at_clear.fetch_max(misses, Ordering::Relaxed);
    }

    fn entry_size<T>(&self, value: &T) -> usize
    where
        T: DeepSizeOf + Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();
        let is_registered = self
            .entry_size_accessors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&type_id);
        if !is_registered {
            self.entry_size_accessors
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(type_id)
                .or_insert(cache_entry_size_with_context::<T>);
        }
        cache_entry_size(value)
    }
}

// ---------------------------------------------------------------------------
// LanceCache — typed wrapper around dyn CacheBackend
// ---------------------------------------------------------------------------

/// Typed cache wrapper that handles key construction and type safety.
///
/// Internally delegates to a [`CacheBackend`]. The default backend is
/// [`MokaCacheBackend`]; pass a custom backend via [`LanceCache::with_backend`].
#[derive(Clone)]
pub struct LanceCache {
    state: Arc<CacheState>,
    namespace: key::CacheNamespace,
    origin: CacheLoadOrigin,
}

impl std::fmt::Debug for LanceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceCache")
            .field("backend", &self.state.backend)
            .finish_non_exhaustive()
    }
}

impl DeepSizeOf for LanceCache {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let state_ptr = Arc::as_ptr(&self.state) as usize;
        if !context.mark_seen(state_ptr) {
            return 0;
        }

        let accessors = self
            .state
            .entry_size_accessors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        self.state
            .backend
            .deep_size_of_entries(context, &|entry, context| {
                accessors
                    .get(&entry.as_ref().type_id())
                    .and_then(|size_of_entry| size_of_entry(entry, context))
            })
            .unwrap_or_else(|| self.state.backend.approx_size_bytes())
    }
}

impl LanceCache {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_backend(Arc::new(MokaCacheBackend::with_capacity(capacity)))
    }

    /// Create a cache backed by a custom [`CacheBackend`].
    pub fn with_backend(backend: Arc<dyn CacheBackend>) -> Self {
        Self::with_backend_and_metrics_kind(backend, CacheMetricsKind::Other)
    }

    /// Create a cache with a bounded logical classification for exported metrics.
    ///
    /// This classification affects only wrapper activity labels. Physical
    /// backend metrics omit it because one backend may be shared by caches with
    /// different logical roles.
    pub fn with_backend_and_metrics_kind(
        backend: Arc<dyn CacheBackend>,
        cache_kind: CacheMetricsKind,
    ) -> Self {
        Self {
            state: Arc::new(CacheState::new(backend, cache_kind)),
            namespace: key::CacheNamespace::root(),
            origin: CacheLoadOrigin::Demand,
        }
    }

    pub fn no_cache() -> Self {
        Self::with_backend(Arc::new(MokaCacheBackend::no_cache()))
    }

    /// Derive a child namespace for all keys in the returned cache handle.
    ///
    /// Each call adds one framed hierarchy segment. Consequently,
    /// `cache.with_key_prefix("a").with_key_prefix("b")` is deliberately
    /// distinct from `cache.with_key_prefix("a/b")`.
    pub fn with_key_prefix(&self, prefix: &str) -> Self {
        Self {
            state: self.state.clone(),
            namespace: self.namespace.child(prefix),
            origin: self.origin,
        }
    }

    /// Return a handle that attributes its cache operations to `origin`.
    ///
    /// The returned handle shares keys, entries, and diagnostics with `self`.
    /// The origin is carried by the handle so it remains correct across async
    /// task boundaries without thread-local state.
    ///
    /// ```
    /// use lance_core::cache::{CacheLoadOrigin, LanceCache};
    /// let cache = LanceCache::with_capacity(1024);
    /// let warm = cache.with_load_origin(CacheLoadOrigin::Warm);
    /// assert_eq!(warm.diagnostics().activity.warm.attempts, 0);
    /// ```
    pub fn with_load_origin(&self, origin: CacheLoadOrigin) -> Self {
        Self {
            state: self.state.clone(),
            namespace: self.namespace,
            origin,
        }
    }

    pub async fn size(&self) -> usize {
        self.state.backend.num_entries().await
    }

    pub fn approx_size(&self) -> usize {
        self.state.backend.approx_num_entries()
    }

    pub async fn size_bytes(&self) -> usize {
        self.state.backend.size_bytes().await
    }

    /// Weighted capacity in bytes, if the backend reports one.
    pub fn capacity_bytes(&self) -> Option<usize> {
        self.state.backend.capacity_bytes()
    }

    // -- Stats / clear --------------------------------------------------------

    pub async fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self
                .state
                .hits
                .load(Ordering::Relaxed)
                .saturating_sub(self.state.hits_at_clear.load(Ordering::Relaxed)),
            misses: self
                .state
                .misses
                .load(Ordering::Relaxed)
                .saturating_sub(self.state.misses_at_clear.load(Ordering::Relaxed)),
            num_entries: self.state.backend.num_entries().await,
            size_bytes: self.state.backend.size_bytes().await,
        }
    }

    pub async fn clear(&self) {
        self.state.backend.clear().await;
        // Reset the legacy view using watermarks so hot lookups still need
        // only one atomic increment for both lifetime and resettable stats.
        self.state.record_clear(
            self.state.hits.load(Ordering::Relaxed),
            self.state.misses.load(Ordering::Relaxed),
        );
    }

    /// Cheap lifetime diagnostics, shared with clones and child namespaces.
    /// No entry traversal, maintenance, or installed recorder is required.
    /// See [`CacheDiagnostics`] for byte accounting and absence semantics.
    pub fn diagnostics(&self) -> CacheDiagnostics {
        CacheDiagnostics::new(self.activity_snapshot(), self.state.backend.diagnostics())
    }

    /// Collect diagnostics with optionally refreshed backend accounting.
    ///
    /// ```
    /// # async fn example() {
    /// use lance_core::cache::{CacheSnapshotMode, LanceCache};
    /// let cache = LanceCache::with_capacity(1024);
    /// let snapshot = cache.diagnostics_with_mode(CacheSnapshotMode::Refreshed).await;
    /// assert_eq!(snapshot.backend.size_bytes, Some(0));
    /// # }
    /// ```
    pub async fn diagnostics_with_mode(&self, mode: CacheSnapshotMode) -> CacheDiagnostics {
        let backend = self.state.backend.diagnostics_with_mode(mode).await;
        CacheDiagnostics::new(self.activity_snapshot(), backend)
    }

    /// Scan backend records and collect explicit per-type diagnostics.
    ///
    /// Activity remains bounded and constant-cost to update. Occupancy requires
    /// an entry scan and is approximate under concurrent mutation. Custom
    /// backends report occupancy as unavailable unless they override the
    /// context-aware diagnostic method.
    ///
    /// ```
    /// # async fn example() {
    /// use lance_core::cache::{CacheSnapshotMode, LanceCache};
    /// let cache = LanceCache::with_capacity(1024);
    /// let snapshot = cache
    ///     .diagnostics_by_type(CacheSnapshotMode::Approximate)
    ///     .await;
    /// assert!(snapshot.by_type.is_some());
    /// # }
    /// ```
    pub async fn diagnostics_by_type(&self, mode: CacheSnapshotMode) -> CacheDiagnostics {
        let (backend, occupancy) = self.state.backend.diagnostics_with_types(mode).await;
        let aggregate = self.activity_snapshot();
        let (activity, type_label_overflow_events) = self.state.type_activity.snapshot(&aggregate);
        CacheDiagnostics::with_by_type(
            aggregate,
            backend,
            CacheByTypeDiagnostics {
                activity,
                occupancy,
                type_label_overflow_events,
            },
        )
    }

    fn activity_snapshot(&self) -> CacheActivity {
        self.state.activity.snapshot(
            self.state.hits.load(Ordering::Relaxed),
            self.state.misses.load(Ordering::Relaxed),
        )
    }

    // -- CacheKey-based methods -----------------------------------------------

    pub async fn insert_with_key<K>(&self, cache_key: &K, metadata: Arc<K::ValueType>)
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let by_type = if self.origin == CacheLoadOrigin::Warm {
            let by_type = self.state.type_activity.get(K::stable_type_id());
            self.state.record_warm_attempt(by_type, self.origin);
            Some(by_type)
        } else {
            None
        };
        let size = self.state.entry_size(metadata.as_ref());
        let key = self.sized_key(cache_key);
        self.state
            .backend
            .insert_with_context(
                &key,
                metadata,
                size,
                K::codec(),
                CacheOperationContext::new(K::stable_type_id(), self.origin),
            )
            .await;
        if let Some(by_type) = by_type {
            self.state.record_warm_insert(by_type, self.origin, size);
        }
    }

    pub async fn get_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let by_type = self.state.type_activity.get(K::stable_type_id());
        self.state.record_warm_attempt(by_type, self.origin);
        let key = self.sized_key(cache_key);
        let Some(entry) = self.state.backend.get(&key, K::codec()).await else {
            self.state.record_miss(by_type);
            return None;
        };
        match entry.downcast::<K::ValueType>() {
            Ok(value) => {
                self.state.record_hit(by_type);
                self.state.record_warm_hit(by_type, self.origin);
                Some(value)
            }
            Err(_) => {
                // Type mismatch: the backend returned a different concrete
                // type than expected (e.g. a disk cache may store
                // intermediate state). Treat as a miss.
                log::warn!(
                    "cache backend returned a value with the wrong concrete type for key type {:?}",
                    K::stable_type_id()
                );
                self.state.record_miss(by_type);
                self.state.record_type_mismatch(by_type, self.origin);
                None
            }
        }
    }

    pub async fn get_or_insert_with_key<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        self.get_or_insert_with_key_outcome(cache_key, loader)
            .await
            .map(|(value, _)| value)
    }

    /// Get or load an entry and report how the backend obtained its value.
    ///
    /// Built-in backends report every distinction their synchronization
    /// primitive can prove. Moka reports [`CacheLoadOutcome::LoaderSkippedUnknown`]
    /// when its loader was skipped, because the public Moka API cannot distinguish
    /// a resident hit from a shared concurrent load. Custom backends retain source
    /// compatibility and may report the same outcome.
    pub async fn get_or_insert_with_key_outcome<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<(Arc<K::ValueType>, CacheLoadOutcome)>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        let by_type = self.state.type_activity.get(K::stable_type_id());
        self.state.record_warm_attempt(by_type, self.origin);
        let key = self.sized_key(&cache_key);
        let state = &self.state;
        let origin = self.origin;
        let typed_loader = Box::pin(async move {
            #[cfg(feature = "metrics")]
            state
                .type_activity
                .note_event(by_type, state.cache_kind, state.backend_kind);
            let guard = state.activity.start_load(
                state.type_activity.activity(by_type),
                state.type_activity.event_metric_keys(by_type),
                origin,
            );
            let result = loader().await;
            guard.finish(result.is_ok());
            let value = Arc::new(result?);
            let size = state.entry_size(value.as_ref());
            if origin == CacheLoadOrigin::Warm {
                state.activity.record_warm_load_bytes(
                    state.type_activity.activity(by_type),
                    state.type_activity.event_metric_keys(by_type),
                    size.try_into().unwrap_or(u64::MAX),
                );
            }
            Ok((value as CacheEntry, size))
        });

        let (entry, outcome) = self
            .state
            .backend
            .get_or_insert_with_context_outcome(
                &key,
                typed_loader,
                K::codec(),
                CacheOperationContext::new(K::stable_type_id(), self.origin),
            )
            .await
            .inspect_err(|_| {
                self.state.record_lookup_error(by_type, "load", self.origin);
            })?;
        let entry = entry.downcast::<K::ValueType>().map_err(|_| {
            self.state.record_miss(by_type);
            self.state.record_type_mismatch(by_type, self.origin);
            Error::io(format!(
                "cache backend returned a value with the wrong concrete type for key type {:?}",
                K::stable_type_id()
            ))
        })?;
        if outcome.was_loader_skipped() {
            self.state.record_hit(by_type);
            self.state.record_warm_hit(by_type, self.origin);
        } else {
            self.state.record_miss(by_type);
        }
        Ok((entry, outcome))
    }

    /// Same as [`get_or_insert_with_key`](Self::get_or_insert_with_key), but
    /// also returns a boolean indicating whether the loader was skipped for
    /// this call.
    ///
    /// - `true` means this call did **not** execute the loader. That covers
    ///   both a true cache hit on an already-populated entry and a coalesced
    ///   concurrent load where an in-flight loader started by a different
    ///   caller produced the value.
    /// - `false` means the loader ran on this call (a real cache miss).
    ///
    /// Callers that need to distinguish resident hits from shared loads should
    /// use [`get_or_insert_with_key_outcome`](Self::get_or_insert_with_key_outcome).
    /// Prefer this method over rolling a caller-side `Arc<AtomicBool>` when only
    /// the legacy loader-skipped bit is needed.
    pub async fn get_or_insert_with_key_hit<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<(Arc<K::ValueType>, bool)>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        self.get_or_insert_with_key_outcome(cache_key, loader)
            .await
            .map(|(entry, outcome)| (entry, outcome.was_loader_skipped()))
    }

    pub async fn insert_unsized_with_key<K>(&self, cache_key: &K, metadata: Arc<K::ValueType>)
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let by_type = if self.origin == CacheLoadOrigin::Warm {
            let by_type = self.state.type_activity.get(K::stable_type_id());
            self.state.record_warm_attempt(by_type, self.origin);
            Some(by_type)
        } else {
            None
        };
        let metadata = Arc::new(metadata);
        let size = self.state.entry_size(metadata.as_ref());
        let key = self.unsized_key(cache_key);
        self.state
            .backend
            .insert_with_context(
                &key,
                metadata,
                size,
                None,
                CacheOperationContext::new(K::stable_type_id(), self.origin),
            )
            .await;
        if let Some(by_type) = by_type {
            self.state.record_warm_insert(by_type, self.origin, size);
        }
    }

    pub async fn get_or_insert_unsized_with_key<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<Arc<K::ValueType>>> + Send,
    {
        self.get_or_insert_unsized_with_key_outcome(cache_key, loader)
            .await
            .map(|(value, _)| value)
    }

    /// Get or load an unsized entry and report how the backend obtained it.
    pub async fn get_or_insert_unsized_with_key_outcome<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<(Arc<K::ValueType>, CacheLoadOutcome)>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<Arc<K::ValueType>>> + Send,
    {
        let by_type = self.state.type_activity.get(K::stable_type_id());
        self.state.record_warm_attempt(by_type, self.origin);
        let key = self.unsized_key(&cache_key);
        let state = &self.state;
        let origin = self.origin;
        let typed_loader = Box::pin(async move {
            #[cfg(feature = "metrics")]
            state
                .type_activity
                .note_event(by_type, state.cache_kind, state.backend_kind);
            let guard = state.activity.start_load(
                state.type_activity.activity(by_type),
                state.type_activity.event_metric_keys(by_type),
                origin,
            );
            let result = loader().await;
            guard.finish(result.is_ok());
            let value = result?;
            let size = state.entry_size(&value);
            if origin == CacheLoadOrigin::Warm {
                state.activity.record_warm_load_bytes(
                    state.type_activity.activity(by_type),
                    state.type_activity.event_metric_keys(by_type),
                    size.try_into().unwrap_or(u64::MAX),
                );
            }
            Ok((Arc::new(value) as CacheEntry, size))
        });

        let (entry, outcome) = self
            .state
            .backend
            .get_or_insert_with_context_outcome(
                &key,
                typed_loader,
                None,
                CacheOperationContext::new(K::stable_type_id(), self.origin),
            )
            .await
            .inspect_err(|_| {
                self.state.record_lookup_error(by_type, "load", self.origin);
            })?;
        let entry = entry.downcast::<Arc<K::ValueType>>().map_err(|_| {
            self.state.record_miss(by_type);
            self.state.record_type_mismatch(by_type, self.origin);
            Error::io(format!(
                "cache backend returned a value with the wrong concrete type for unsized key type {:?}",
                K::stable_type_id()
            ))
        })?;
        if outcome.was_loader_skipped() {
            self.state.record_hit(by_type);
            self.state.record_warm_hit(by_type, self.origin);
        } else {
            self.state.record_miss(by_type);
        }
        Ok((entry.as_ref().clone(), outcome))
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let by_type = self.state.type_activity.get(K::stable_type_id());
        self.state.record_warm_attempt(by_type, self.origin);
        let key = self.unsized_key(cache_key);
        let Some(entry) = self.state.backend.get(&key, None).await else {
            self.state.record_miss(by_type);
            return None;
        };
        match entry.downcast::<Arc<K::ValueType>>() {
            Ok(value) => {
                self.state.record_hit(by_type);
                self.state.record_warm_hit(by_type, self.origin);
                Some(value.as_ref().clone())
            }
            Err(_) => {
                // Type mismatch: the backend returned a different concrete
                // type than expected (e.g. a disk cache may store
                // intermediate state). Treat as a miss.
                log::warn!(
                    "cache backend returned a value with the wrong concrete type for unsized key type {:?}",
                    K::stable_type_id()
                );
                self.state.record_miss(by_type);
                self.state.record_type_mismatch(by_type, self.origin);
                None
            }
        }
    }

    fn sized_key<K: CacheKey>(&self, cache_key: &K) -> InternalCacheKey {
        let mut builder = KeyBuilder::new(self.namespace, K::stable_type_id(), K::schema());
        cache_key.write_key(&mut builder);
        builder.finish()
    }

    fn unsized_key<K: UnsizedCacheKey>(&self, cache_key: &K) -> InternalCacheKey {
        let mut builder = KeyBuilder::new(self.namespace, K::stable_type_id(), K::schema());
        cache_key.write_key(&mut builder);
        builder.finish()
    }
}

// ---------------------------------------------------------------------------
// WeakLanceCache
// ---------------------------------------------------------------------------

/// A weak reference to a LanceCache, used by indices to avoid circular references.
/// When the original cache is dropped, operations on this will gracefully no-op.
#[derive(Clone, Debug)]
pub struct WeakLanceCache {
    state: Weak<CacheState>,
    namespace: key::CacheNamespace,
    origin: CacheLoadOrigin,
}

impl WeakLanceCache {
    pub fn from(cache: &LanceCache) -> Self {
        Self {
            state: Arc::downgrade(&cache.state),
            namespace: cache.namespace,
            origin: cache.origin,
        }
    }

    pub fn with_key_prefix(&self, prefix: &str) -> Self {
        Self {
            state: self.state.clone(),
            namespace: self.namespace.child(prefix),
            origin: self.origin,
        }
    }

    /// Return a weak handle that attributes its operations to `origin`.
    ///
    /// ```
    /// use lance_core::cache::{CacheLoadOrigin, LanceCache};
    /// let cache = LanceCache::with_capacity(1024);
    /// let warm = lance_core::cache::WeakLanceCache::from(&cache)
    ///     .with_load_origin(CacheLoadOrigin::Warm);
    /// ```
    pub fn with_load_origin(&self, origin: CacheLoadOrigin) -> Self {
        Self {
            state: self.state.clone(),
            namespace: self.namespace,
            origin,
        }
    }

    /// Weighted capacity in bytes, if the cache is alive and its backend
    /// reports one.
    pub fn capacity_bytes(&self) -> Option<usize> {
        self.upgrade()?.capacity_bytes()
    }

    pub async fn get_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.upgrade()?.get_with_key(cache_key).await
    }

    pub async fn insert_with_key<K>(&self, cache_key: &K, value: Arc<K::ValueType>) -> bool
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let Some(cache) = self.upgrade() else {
            log::warn!("WeakLanceCache: cache no longer available, unable to insert item");
            return false;
        };
        cache.insert_with_key(cache_key, value).await;
        true
    }

    /// Get or insert an item, computing it if necessary.
    ///
    /// Deduplication of concurrent loads is handled by the backend.
    pub async fn get_or_insert_with_key<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<Arc<K::ValueType>>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        self.get_or_insert_with_key_outcome(cache_key, loader)
            .await
            .map(|(value, _)| value)
    }

    /// Get or load an item and report how the live backend obtained it.
    pub async fn get_or_insert_with_key_outcome<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<(Arc<K::ValueType>, CacheLoadOutcome)>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        let Some(cache) = self.upgrade() else {
            log::warn!("WeakLanceCache: cache no longer available, computing without caching");
            return loader()
                .await
                .map(|value| (Arc::new(value), CacheLoadOutcome::Loaded));
        };
        cache
            .get_or_insert_with_key_outcome(cache_key, loader)
            .await
    }

    /// Same as [`get_or_insert_with_key`](Self::get_or_insert_with_key), but
    /// also returns a boolean indicating whether the loader was skipped for
    /// this call. See [`LanceCache::get_or_insert_with_key_hit`] for the
    /// coalesced-load caveat.
    pub async fn get_or_insert_with_key_hit<K, F, Fut>(
        &self,
        cache_key: K,
        loader: F,
    ) -> Result<(Arc<K::ValueType>, bool)>
    where
        K: CacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<K::ValueType>> + Send,
    {
        self.get_or_insert_with_key_outcome(cache_key, loader)
            .await
            .map(|(value, outcome)| (value, outcome.was_loader_skipped()))
    }

    pub async fn get_unsized_with_key<K>(&self, cache_key: &K) -> Option<Arc<K::ValueType>>
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        self.upgrade()?.get_unsized_with_key(cache_key).await
    }

    pub async fn insert_unsized_with_key<K>(&self, cache_key: &K, value: Arc<K::ValueType>)
    where
        K: UnsizedCacheKey,
        K::ValueType: DeepSizeOf + Send + Sync + 'static,
    {
        let Some(cache) = self.upgrade() else {
            log::warn!("WeakLanceCache: cache no longer available, unable to insert unsized item");
            return;
        };
        cache.insert_unsized_with_key(cache_key, value).await;
    }

    fn upgrade(&self) -> Option<LanceCache> {
        Some(LanceCache {
            state: self.state.upgrade()?,
            namespace: self.namespace,
            origin: self.origin,
        })
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of times `get`, `get_unsized`, or `get_or_insert` found an item in the cache.
    pub hits: u64,
    /// Number of times `get`, `get_unsized`, or `get_or_insert` did not find an item in the cache.
    pub misses: u64,
    /// Number of entries currently in the cache.
    pub num_entries: usize,
    /// Total size in bytes of all entries in the cache.
    pub size_bytes: usize,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f32 / (self.hits + self.misses) as f32
        }
    }

    pub fn miss_ratio(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.misses as f32 / (self.hits + self.misses) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::pin::Pin;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    use std::task::Poll;
    use std::thread;
    use std::time::Duration;

    use super::*;

    async fn report_first_pending<F>(
        future: F,
        parked: tokio::sync::oneshot::Sender<()>,
    ) -> F::Output
    where
        F: Future,
    {
        tokio::pin!(future);
        let mut parked = Some(parked);
        futures::future::poll_fn(|cx| match future.as_mut().poll(cx) {
            Poll::Pending => {
                if let Some(parked) = parked.take() {
                    let _ = parked.send(());
                }
                Poll::Pending
            }
            Poll::Ready(output) => Poll::Ready(output),
        })
        .await
    }

    #[derive(Clone)]
    struct VersionedTestKey<const SCHEMA_VERSION: u32> {
        id: u64,
    }

    type TestKey = VersionedTestKey<1>;
    type TestKeyV2 = VersionedTestKey<2>;

    impl<const SCHEMA_VERSION: u32> VersionedTestKey<SCHEMA_VERSION> {
        fn new(id: u64) -> Self {
            Self { id }
        }
    }

    impl<const SCHEMA_VERSION: u32> CacheKey for VersionedTestKey<SCHEMA_VERSION> {
        type ValueType = Vec<u32>;

        fn key(&self) -> Cow<'_, str> {
            self.id.to_string().into()
        }

        fn type_name() -> &'static str {
            "test.VecU32"
        }

        fn schema() -> CacheKeySchema {
            CacheKeySchema::new("test.vec-u32-key", SCHEMA_VERSION)
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_u64(self.id);
        }
    }

    struct FirstTypedKey(u64);

    impl CacheKey for FirstTypedKey {
        type ValueType = Vec<u32>;

        fn key(&self) -> Cow<'_, str> {
            self.0.to_string().into()
        }

        fn type_name() -> &'static str {
            "test.FirstVecU32"
        }
    }

    struct SecondTypedKey(u64);

    impl CacheKey for SecondTypedKey {
        type ValueType = Vec<u32>;

        fn key(&self) -> Cow<'_, str> {
            self.0.to_string().into()
        }

        fn type_name() -> &'static str {
            "test.SecondVecU32"
        }
    }

    struct SharedTestValue {
        data: Arc<Vec<u8>>,
    }

    impl DeepSizeOf for SharedTestValue {
        fn deep_size_of_children(&self, context: &mut Context) -> usize {
            self.data.deep_size_of_children(context)
        }
    }

    struct SharedTestKey(u64);

    impl CacheKey for SharedTestKey {
        type ValueType = SharedTestValue;

        fn key(&self) -> Cow<'_, str> {
            self.0.to_string().into()
        }

        fn type_name() -> &'static str {
            "test.SharedValue"
        }

        fn schema() -> CacheKeySchema {
            CacheKeySchema::new("test.shared-value-key", 1)
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_u64(self.0);
        }
    }

    struct ReentrantValue(LanceCache);

    impl DeepSizeOf for ReentrantValue {
        fn deep_size_of_children(&self, context: &mut Context) -> usize {
            self.0.deep_size_of_children(context)
        }
    }

    struct ReentrantKey;

    impl CacheKey for ReentrantKey {
        type ValueType = ReentrantValue;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed("reentrant")
        }

        fn type_name() -> &'static str {
            "test.ReentrantValue"
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestBackendKind {
        Moka,
        Quick,
    }

    impl TestBackendKind {
        fn cache(self, capacity: usize) -> LanceCache {
            LanceCache::with_backend(self.backend(capacity))
        }

        fn backend(self, capacity: usize) -> Arc<dyn CacheBackend> {
            match self {
                Self::Moka => Arc::new(MokaCacheBackend::with_capacity(capacity)),
                Self::Quick => Arc::new(QuickCacheBackend::with_capacity(capacity)),
            }
        }
    }

    struct LegacyBridgeKey(&'static str);

    impl CacheKey for LegacyBridgeKey {
        type ValueType = Vec<u32>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "test.LegacyBridge"
        }
    }

    struct ExplicitBridgeKey(&'static str);

    impl CacheKey for ExplicitBridgeKey {
        type ValueType = Vec<u32>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "test.LegacyBridge"
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_str(self.0);
        }
    }

    trait TestDynValue: DeepSizeOf + Send + Sync {
        fn values(&self) -> &[u32];
    }

    impl TestDynValue for Vec<u32> {
        fn values(&self) -> &[u32] {
            self
        }
    }

    struct LegacyUnsizedBridgeKey(&'static str);

    impl UnsizedCacheKey for LegacyUnsizedBridgeKey {
        type ValueType = dyn TestDynValue;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "test.LegacyUnsizedBridge"
        }
    }

    struct ExplicitUnsizedBridgeKey(&'static str);

    impl UnsizedCacheKey for ExplicitUnsizedBridgeKey {
        type ValueType = dyn TestDynValue;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "test.LegacyUnsizedBridge"
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_str(self.0);
        }
    }

    #[derive(Debug, Default)]
    struct HashMapBackend {
        entries: tokio::sync::Mutex<HashMap<InternalCacheKey, (CacheEntry, usize)>>,
    }

    #[async_trait::async_trait]
    impl CacheBackend for HashMapBackend {
        async fn get(
            &self,
            key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            self.entries
                .lock()
                .await
                .get(key)
                .map(|(entry, _)| entry.clone())
        }

        async fn insert(
            &self,
            key: &InternalCacheKey,
            entry: CacheEntry,
            size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
            self.entries.lock().await.insert(*key, (entry, size_bytes));
        }

        async fn get_or_insert<'a>(
            &self,
            key: &InternalCacheKey,
            loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
            codec: Option<CacheCodec>,
        ) -> Result<(CacheEntry, bool)> {
            if let Some(entry) = self.get(key, codec).await {
                return Ok((entry, true));
            }
            let (entry, size_bytes) = loader.await?;
            self.insert(key, entry.clone(), size_bytes, codec).await;
            Ok((entry, false))
        }

        async fn clear(&self) {
            self.entries.lock().await.clear();
        }

        async fn num_entries(&self) -> usize {
            self.entries.lock().await.len()
        }

        async fn size_bytes(&self) -> usize {
            self.entries
                .lock()
                .await
                .values()
                .map(|(_, size_bytes)| size_bytes)
                .sum()
        }
    }

    #[derive(Debug)]
    struct WrongTypeBackend;

    #[async_trait::async_trait]
    impl CacheBackend for WrongTypeBackend {
        async fn get(
            &self,
            _key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            Some(Arc::new(String::from("wrong type")))
        }

        async fn insert(
            &self,
            _key: &InternalCacheKey,
            _entry: CacheEntry,
            _size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
        }

        async fn get_or_insert<'a>(
            &self,
            _key: &InternalCacheKey,
            _loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
            _codec: Option<CacheCodec>,
        ) -> Result<(CacheEntry, bool)> {
            Ok((Arc::new(String::from("wrong type")), true))
        }

        async fn clear(&self) {}

        async fn num_entries(&self) -> usize {
            0
        }

        async fn size_bytes(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn typed_roundtrip_stats_clear_and_namespace_isolation() {
        let cache = LanceCache::with_capacity(4096);
        let left = cache.with_key_prefix("left");
        let right = cache.with_key_prefix("right");
        left.insert_with_key(&TestKey::new(7), Arc::new(vec![1, 2, 3]))
            .await;

        assert_eq!(
            left.get_with_key(&TestKey::new(7)).await.as_deref(),
            Some(&vec![1, 2, 3])
        );
        assert!(right.get_with_key(&TestKey::new(7)).await.is_none());
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses, stats.num_entries), (1, 1, 1));

        cache.clear().await;
        let stats = left.stats().await;
        assert_eq!((stats.hits, stats.misses, stats.num_entries), (0, 0, 0));
    }

    #[tokio::test]
    async fn strong_and_weak_handles_share_state_and_namespace() {
        let cache = LanceCache::with_capacity(4096);
        let child = cache.with_key_prefix("child");
        let weak = WeakLanceCache::from(&child);

        assert!(
            weak.insert_with_key(&TestKey::new(1), Arc::new(vec![1]))
                .await
        );
        assert_eq!(
            child.get_with_key(&TestKey::new(1)).await.as_deref(),
            Some(&vec![1])
        );
        child
            .insert_with_key(&TestKey::new(2), Arc::new(vec![2]))
            .await;
        assert_eq!(
            weak.get_with_key(&TestKey::new(2)).await.as_deref(),
            Some(&vec![2])
        );
        assert_eq!((cache.stats().await.hits, cache.size().await), (2, 2));
    }

    #[tokio::test]
    async fn nested_namespace_segments_do_not_alias_combined_segments() {
        let cache = LanceCache::with_capacity(4096);
        let nested = cache.with_key_prefix("a").with_key_prefix("b");
        let combined = cache.with_key_prefix("a/b");
        nested
            .insert_with_key(&TestKey::new(1), Arc::new(vec![10]))
            .await;
        assert!(combined.get_with_key(&TestKey::new(1)).await.is_none());
    }

    #[tokio::test]
    async fn schema_change_produces_a_cold_miss() {
        let cache = LanceCache::with_capacity(4096);
        cache
            .insert_with_key(&TestKey::new(1), Arc::new(vec![10]))
            .await;
        assert!(cache.get_with_key(&TestKeyV2::new(1)).await.is_none());
    }

    #[tokio::test]
    async fn get_or_insert_with_key_hit_reports_loader_execution() {
        let cache = LanceCache::with_capacity(4096);

        // Cold: loader runs, was_cached = false.
        let (value, was_cached) = cache
            .get_or_insert_with_key_hit(TestKey::new(1), || async { Ok(vec![1, 2, 3]) })
            .await
            .unwrap();
        assert_eq!(*value, vec![1, 2, 3]);
        assert!(!was_cached);

        // Warm: loader must not run and was_cached = true.
        let (value, was_cached) = cache
            .get_or_insert_with_key_hit(TestKey::new(1), || async {
                panic!("should not be called")
            })
            .await
            .unwrap();
        assert_eq!(*value, vec![1, 2, 3]);
        assert!(was_cached);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn get_or_insert_reports_loaded_and_skipped_outcomes(#[case] kind: TestBackendKind) {
        let cache = kind.cache(4096);

        let (value, outcome) = cache
            .get_or_insert_with_key_outcome(TestKey::new(1), || async { Ok(vec![1, 2, 3]) })
            .await
            .unwrap();
        assert_eq!(value.as_slice(), &[1, 2, 3]);
        assert_eq!(outcome, CacheLoadOutcome::Loaded);

        let (_, outcome) = cache
            .get_or_insert_with_key_outcome(TestKey::new(1), || async {
                panic!("resident loader must not run")
            })
            .await
            .unwrap();
        assert_eq!(
            outcome,
            match kind {
                TestBackendKind::Moka => CacheLoadOutcome::LoaderSkippedUnknown,
                TestBackendKind::Quick => CacheLoadOutcome::ResidentHit,
            }
        );
    }

    #[tokio::test]
    async fn custom_backend_reports_unknown_skipped_loader_outcome() {
        let cache = LanceCache::with_backend(Arc::new(HashMapBackend::default()));

        let (_, cold) = cache
            .get_or_insert_with_key_outcome(TestKey::new(1), || async { Ok(vec![1]) })
            .await
            .unwrap();
        let (_, warm) = cache
            .get_or_insert_with_key_outcome(TestKey::new(1), || async {
                panic!("resident loader must not run")
            })
            .await
            .unwrap();

        assert_eq!(cold, CacheLoadOutcome::Loaded);
        assert_eq!(warm, CacheLoadOutcome::LoaderSkippedUnknown);
    }

    #[tokio::test]
    async fn default_string_bridge_matches_explicit_legacy_encoding() {
        let cache = LanceCache::with_capacity(4096);
        cache
            .insert_with_key(&LegacyBridgeKey("same"), Arc::new(vec![10]))
            .await;
        assert_eq!(
            cache
                .get_with_key(&ExplicitBridgeKey("same"))
                .await
                .as_deref(),
            Some(&vec![10])
        );
    }

    #[tokio::test]
    async fn unsized_default_string_bridge_matches_explicit_legacy_encoding() {
        let cache = LanceCache::with_capacity(4096);
        let value: Arc<dyn TestDynValue> = Arc::new(vec![10, 20]);
        cache
            .insert_unsized_with_key(&LegacyUnsizedBridgeKey("same"), value)
            .await;

        let cached = cache
            .get_unsized_with_key(&ExplicitUnsizedBridgeKey("same"))
            .await
            .unwrap();
        assert_eq!(cached.values(), &[10, 20]);
    }

    #[tokio::test]
    async fn custom_backend_receives_opaque_keys_and_shared_clear() {
        let backend = Arc::new(HashMapBackend::default());
        let cache = LanceCache::with_backend(backend.clone());
        let child = cache.with_key_prefix("child");
        let value = Arc::new(vec![1, 2, 3]);
        let value_size = cache_entry_size(value.as_ref());

        child.insert_with_key(&TestKey::new(7), value).await;
        assert_eq!(
            child.get_with_key(&TestKey::new(7)).await.as_deref(),
            Some(&vec![1, 2, 3])
        );
        assert_eq!(backend.entries.lock().await.len(), 1);
        assert_eq!(cache.size_bytes().await, value_size);

        cache.clear().await;
        assert!(backend.entries.lock().await.is_empty());
        assert_eq!(child.stats().await.hits, 0);
    }

    #[tokio::test]
    async fn backend_type_collisions_are_contextual_misses_or_errors() {
        let cache = LanceCache::with_backend(Arc::new(WrongTypeBackend));

        assert!(cache.get_with_key(&TestKey::new(1)).await.is_none());
        let error = cache
            .get_or_insert_with_key(TestKey::new(2), || async { Ok(vec![2]) })
            .await
            .unwrap_err();
        assert!(error.to_string().contains("test.VecU32"));
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses), (0, 2));
        let activity = cache.diagnostics().activity;
        assert_eq!((activity.lookup_errors, activity.type_mismatches), (2, 2));
        assert_eq!(activity.loads_started, 0);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn diagnostics_share_lifetime_activity_and_preserve_legacy_clear(
        #[case] kind: TestBackendKind,
    ) {
        let cache = kind.cache(4096);
        let child = cache.with_key_prefix("child");
        let weak = WeakLanceCache::from(&child);
        assert!(child.get_with_key(&TestKey::new(1)).await.is_none());
        child
            .get_or_insert_with_key(TestKey::new(1), || async { Ok(vec![1]) })
            .await
            .unwrap();
        assert!(weak.get_with_key(&TestKey::new(1)).await.is_some());
        child
            .get_or_insert_with_key_hit(TestKey::new(1), || async { panic!("resident loader") })
            .await
            .unwrap();
        let error = child
            .get_or_insert_with_key(TestKey::new(2), || async {
                Err(Error::timeout("diagnostic loader failure"))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Timeout { .. }));
        assert!(error.to_string().contains("diagnostic loader failure"));
        let before = cache.diagnostics().activity;
        assert_eq!(
            (before.hits, before.misses, before.lookup_errors),
            (2, 2, 1)
        );
        assert_eq!(
            (
                before.loads_started,
                before.loads_succeeded,
                before.loads_failed
            ),
            (2, 1, 1)
        );
        assert_eq!((before.loads_cancelled, before.loads_in_flight), (0, 0));
        cache.clear().await;
        let legacy = child.stats().await;
        assert_eq!((legacy.hits, legacy.misses), (0, 0));
        let after = cache.diagnostics().activity;
        assert_eq!((after.hits, after.misses, after.lookup_errors), (2, 2, 1));
        assert!(weak.get_with_key(&TestKey::new(1)).await.is_none());
        assert_eq!(cache.diagnostics().activity.misses, 3);
        assert_eq!(cache.stats().await.misses, 1);
        drop(child);
        let state = Arc::downgrade(&cache.state);
        drop(cache);
        assert!(state.upgrade().is_none());
        assert!(weak.get_with_key(&TestKey::new(1)).await.is_none());
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn legacy_clear_watermarks_ignore_older_samples(#[case] kind: TestBackendKind) {
        let cache = kind.cache(4096);
        for _ in 0..2 {
            assert!(cache.get_with_key(&TestKey::new(1)).await.is_none());
        }
        let older = cache.diagnostics().activity;
        cache
            .insert_with_key(&TestKey::new(1), Arc::new(vec![1]))
            .await;
        for _ in 0..2 {
            assert!(cache.get_with_key(&TestKey::new(1)).await.is_some());
            assert!(cache.get_with_key(&TestKey::new(2)).await.is_none());
        }
        let newer = cache.diagnostics().activity;
        // Model an earlier clear publishing its sample after another clear.
        // Using real lookups also verifies that lifetime counters do not reset.
        cache.state.record_clear(newer.hits, newer.misses);
        cache.state.record_clear(older.hits, older.misses);
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses), (0, 0));
        assert!(cache.get_with_key(&TestKey::new(1)).await.is_some());
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses), (1, 0));
        cache.clear().await;
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses), (0, 0));
        let activity = cache.diagnostics().activity;
        assert_eq!((activity.hits, activity.misses), (3, 4));
    }

    #[rstest::rstest]
    #[case::moka_demand(TestBackendKind::Moka, CacheLoadOrigin::Demand)]
    #[case::quick_demand(TestBackendKind::Quick, CacheLoadOrigin::Demand)]
    #[case::moka_warm(TestBackendKind::Moka, CacheLoadOrigin::Warm)]
    #[case::quick_warm(TestBackendKind::Quick, CacheLoadOrigin::Warm)]
    #[tokio::test]
    async fn diagnostics_cancel_only_executing_loaders(
        #[case] kind: TestBackendKind,
        #[case] origin: CacheLoadOrigin,
    ) {
        let cache = kind.cache(4096).with_load_origin(origin);
        let mut load = Box::pin(cache.get_or_insert_with_key(TestKey::new(1), || async {
            futures::future::pending::<Result<Vec<u32>>>().await
        }));
        assert!(futures::poll!(load.as_mut()).is_pending());
        let during = cache.diagnostics().activity;
        assert_eq!((during.loads_started, during.loads_in_flight), (1, 1));
        let mut waiter =
            Box::pin(cache.get_or_insert_with_key(TestKey::new(1), || async { Ok(vec![2]) }));
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        drop(waiter);
        assert_eq!(cache.diagnostics().activity.loads_cancelled, 0);
        drop(load);
        let after = cache.diagnostics().activity;
        assert_eq!(
            (
                after.loads_started,
                after.loads_cancelled,
                after.loads_in_flight
            ),
            (1, 1, 0)
        );
        assert_eq!((after.hits, after.misses, after.lookup_errors), (0, 0, 0));
        assert_eq!(
            (after.warm.loads_started, after.warm.loads_cancelled),
            if origin == CacheLoadOrigin::Warm {
                (1, 1)
            } else {
                (0, 0)
            }
        );
        let unpolled = cache.get_or_insert_with_key(TestKey::new(2), || async { Ok(vec![3]) });
        drop(unpolled);
        assert_eq!(cache.diagnostics().activity.loads_started, 1);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn diagnostics_instrument_unsized_values(#[case] kind: TestBackendKind) {
        let cache = kind.cache(4096);
        let value: Arc<dyn TestDynValue> = Arc::new(vec![1_u32]);
        let (_, loaded) = cache
            .get_or_insert_unsized_with_key_outcome(LegacyUnsizedBridgeKey("loaded"), || async {
                Ok(value)
            })
            .await
            .unwrap();
        assert_eq!(loaded, CacheLoadOutcome::Loaded);
        let (_, resident) = cache
            .get_or_insert_unsized_with_key_outcome(LegacyUnsizedBridgeKey("loaded"), || async {
                panic!("resident unsized loader must not run")
            })
            .await
            .unwrap();
        assert_eq!(
            resident,
            match kind {
                TestBackendKind::Moka => CacheLoadOutcome::LoaderSkippedUnknown,
                TestBackendKind::Quick => CacheLoadOutcome::ResidentHit,
            }
        );
        let error = cache
            .get_or_insert_unsized_with_key(LegacyUnsizedBridgeKey("failed"), || async {
                Err(Error::timeout("unsized loader failure"))
            })
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::Timeout { .. }));
        assert!(error.to_string().contains("unsized loader failure"));
        let activity = cache.diagnostics().activity;
        assert_eq!(
            (activity.hits, activity.misses, activity.lookup_errors),
            (1, 1, 1)
        );
        assert_eq!(
            (
                activity.loads_started,
                activity.loads_succeeded,
                activity.loads_failed
            ),
            (2, 1, 1)
        );
    }

    #[tokio::test]
    async fn custom_backend_diagnostics_default_to_unsupported() {
        let cache = LanceCache::with_backend(Arc::new(HashMapBackend::default()));
        let backend = cache.diagnostics().backend;
        assert_eq!(backend.kind, CacheBackendKind::Custom);
        assert_eq!((backend.capacity_bytes, backend.enabled), (None, None));
        assert_eq!(
            (
                backend.size_bytes,
                backend.num_entries,
                backend.write_attempts
            ),
            (None, None, None)
        );
        let refreshed = cache
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (refreshed.backend.size_bytes, refreshed.backend.num_entries),
            (Some(0), Some(0))
        );
        assert_eq!(refreshed.utilization, None);
        let by_type = cache
            .diagnostics_by_type(CacheSnapshotMode::Refreshed)
            .await
            .by_type
            .unwrap();
        assert!(by_type.occupancy.is_none());
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn diagnostics_by_type_distinguish_keys_and_track_occupancy(
        #[case] kind: TestBackendKind,
    ) {
        let backend = kind.backend(4096);
        let cache = LanceCache::with_backend(backend.clone());

        cache
            .get_or_insert_with_key(FirstTypedKey(1), || async { Ok(vec![1, 2]) })
            .await
            .unwrap();
        assert!(cache.get_with_key(&FirstTypedKey(1)).await.is_some());
        cache
            .insert_with_key(&SecondTypedKey(1), Arc::new(vec![3, 4, 5]))
            .await;
        assert!(cache.get_with_key(&SecondTypedKey(1)).await.is_some());

        // Calls made directly against the compatibility API remain visible as
        // untagged occupancy.
        backend
            .insert(
                &InternalCacheKey::from_bytes([9; 16]),
                Arc::new(9_u64),
                8,
                None,
            )
            .await;

        let snapshot = cache
            .diagnostics_by_type(CacheSnapshotMode::Refreshed)
            .await;
        let by_type = snapshot.by_type.unwrap();
        assert_eq!(by_type.activity.len(), 2);
        let first = by_type
            .activity
            .iter()
            .find(|activity| activity.type_name == "test.FirstVecU32")
            .unwrap();
        let second = by_type
            .activity
            .iter()
            .find(|activity| activity.type_name == "test.SecondVecU32")
            .unwrap();
        assert_eq!(
            (
                first.activity.hits,
                first.activity.misses,
                first.activity.loads_started,
                first.activity.loads_succeeded,
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(
            (
                second.activity.hits,
                second.activity.misses,
                second.activity.loads_started,
            ),
            (1, 0, 0)
        );
        assert_eq!(
            (
                first.activity.hits + second.activity.hits,
                first.activity.misses + second.activity.misses,
                first.activity.loads_started + second.activity.loads_started,
                first.activity.loads_succeeded + second.activity.loads_succeeded,
            ),
            (
                snapshot.activity.hits,
                snapshot.activity.misses,
                snapshot.activity.loads_started,
                snapshot.activity.loads_succeeded,
            )
        );

        let occupancy = by_type.occupancy.unwrap();
        assert_eq!(occupancy.types.len(), 2);
        assert_eq!(occupancy.untagged_num_entries, 1);
        assert_eq!(occupancy.untagged_size_bytes, 24);
        let typed_size = occupancy
            .types
            .iter()
            .map(|occupancy| occupancy.size_bytes)
            .sum::<u64>();
        let typed_entries = occupancy
            .types
            .iter()
            .map(|occupancy| occupancy.num_entries)
            .sum::<u64>();
        assert_eq!(
            typed_size + occupancy.untagged_size_bytes,
            snapshot.backend.size_bytes.unwrap()
        );
        assert_eq!(
            typed_entries + occupancy.untagged_num_entries,
            snapshot.backend.num_entries.unwrap()
        );

        cache
            .insert_with_key(&FirstTypedKey(1), Arc::new(vec![6]))
            .await;
        let replaced = cache
            .diagnostics_by_type(CacheSnapshotMode::Refreshed)
            .await;
        let replaced_occupancy = replaced.by_type.unwrap().occupancy.unwrap();
        assert_eq!(
            replaced_occupancy
                .types
                .iter()
                .map(|occupancy| occupancy.num_entries)
                .sum::<u64>(),
            2
        );

        cache.clear().await;
        let cleared = cache
            .diagnostics_by_type(CacheSnapshotMode::Refreshed)
            .await;
        let cleared_by_type = cleared.by_type.unwrap();
        assert!(cleared_by_type.occupancy.unwrap().types.is_empty());
        assert_eq!((cleared.activity.hits, cleared.activity.misses), (2, 1));
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn warm_origin_tracks_actual_work_separately_from_demand(#[case] kind: TestBackendKind) {
        let cache = LanceCache::with_backend(kind.backend(4096));
        let warm = cache.with_load_origin(CacheLoadOrigin::Warm);
        warm.get_or_insert_with_key(FirstTypedKey(1), || async { Ok(vec![1, 2]) })
            .await
            .unwrap();
        warm.get_or_insert_with_key(FirstTypedKey(1), || async {
            panic!("resident warm lookup must skip loader")
        })
        .await
        .unwrap();
        warm.insert_with_key(&SecondTypedKey(1), Arc::new(vec![3]))
            .await;
        assert!(cache.get_with_key(&SecondTypedKey(1)).await.is_some());

        let error = warm
            .get_or_insert_with_key(FirstTypedKey(2), || async {
                Err::<Vec<u32>, _>(Error::timeout("warm load failed"))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Timeout { .. }));
        assert!(error.to_string().contains("warm load failed"));

        let snapshot = cache
            .diagnostics_by_type(CacheSnapshotMode::Refreshed)
            .await;
        let warm_activity = &snapshot.activity.warm;
        assert_eq!(warm_activity.attempts, 4);
        assert_eq!(warm_activity.hits, 1);
        assert_eq!(warm_activity.loads_started, 3);
        assert_eq!(warm_activity.loads_succeeded, 2);
        assert_eq!(warm_activity.loads_failed, 1);
        assert_eq!(warm_activity.errors, 1);
        assert!(warm_activity.load_bytes > 0);
        let by_type = snapshot.by_type.unwrap();
        assert_eq!(
            by_type
                .activity
                .iter()
                .map(|row| row.activity.warm.attempts)
                .sum::<u64>(),
            warm_activity.attempts
        );
        assert_eq!(
            by_type
                .activity
                .iter()
                .map(|row| row.activity.warm.load_bytes)
                .sum::<u64>(),
            warm_activity.load_bytes
        );
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn backend_diagnostics_writes_replacement_clear_and_oversize(
        #[case] kind: TestBackendKind,
    ) {
        let backend = kind.backend(256);
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry: CacheEntry = Arc::new(());
        backend.insert(&key, entry.clone(), 48, None).await;
        let cold = backend
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!((cold.write_attempts, cold.write_bytes), (Some(1), Some(64)));
        assert_eq!((cold.num_entries, cold.size_bytes), (Some(1), Some(64)));
        assert_eq!(
            cold.size_removals,
            (kind == TestBackendKind::Quick).then_some(0)
        );
        assert!(cold.pool_id.is_some());
        assert_eq!((cold.admissions, cold.resident_evictions), (None, None));
        assert_eq!(cold.coalesced_loads, None);
        assert!(!cold.write_rejections_complete);

        backend.insert(&key, entry.clone(), 80, None).await;
        let (_, was_cached) = backend
            .get_or_insert(
                &key,
                Box::pin(async { panic!("resident lookup must not submit a write") }),
                None,
            )
            .await
            .unwrap();
        assert!(was_cached);
        let replacement = backend
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (replacement.write_attempts, replacement.write_bytes),
            (Some(2), Some(160))
        );
        assert_eq!(
            (replacement.num_entries, replacement.size_bytes),
            (Some(1), Some(96))
        );
        assert_eq!(
            replacement.size_removals,
            (kind == TestBackendKind::Quick).then_some(0)
        );

        // Reusing the same value Arc still replaces the previous submission.
        backend.insert(&key, entry.clone(), 80, None).await;
        assert_eq!(
            backend.diagnostics().size_removals,
            (kind == TestBackendKind::Quick).then_some(0)
        );

        let loader_key = InternalCacheKey::from_bytes([1; 16]);
        let value = entry.clone();
        backend
            .get_or_insert(&loader_key, Box::pin(async move { Ok((value, 8)) }), None)
            .await
            .unwrap();
        let error_key = InternalCacheKey::from_bytes([2; 16]);
        let error = backend
            .get_or_insert(
                &error_key,
                Box::pin(async { Err(Error::timeout("backend loader failure")) }),
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Timeout { .. }));
        assert!(error.to_string().contains("backend loader failure"));
        backend.clear().await;
        let cleared = backend
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (cleared.write_attempts, cleared.write_bytes),
            (Some(4), Some(280))
        );
        assert_eq!(
            (cleared.num_entries, cleared.size_bytes),
            (Some(0), Some(0))
        );
        assert_eq!(
            cleared.size_removals,
            (kind == TestBackendKind::Quick).then_some(0)
        );

        backend.insert(&key, entry, 1024, None).await;
        let oversized = backend
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (oversized.num_entries, oversized.size_bytes),
            (Some(0), Some(0))
        );
        assert_eq!(
            (oversized.size_removals, oversized.size_removed_bytes),
            if kind == TestBackendKind::Quick {
                (Some(1), Some(1040))
            } else {
                (None, None)
            }
        );
        assert_eq!(oversized.write_attempts, Some(5));
        assert_eq!(oversized.pool_id, cold.pool_id);
        assert_eq!(oversized.resident_evictions, None);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn backend_diagnostics_size_pressure(#[case] kind: TestBackendKind) {
        let backend = kind.backend(256);
        let entry: CacheEntry = Arc::new(());
        for id in 0..8 {
            backend
                .insert(
                    &InternalCacheKey::from_bytes([id; 16]),
                    entry.clone(),
                    48,
                    None,
                )
                .await;
        }
        let snapshot = backend
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(snapshot.write_attempts, Some(8));
        if kind == TestBackendKind::Quick {
            assert!(snapshot.size_removals.unwrap() > 0);
            assert_eq!(
                snapshot.size_removed_bytes.unwrap(),
                snapshot.size_removals.unwrap() * 64
            );
        } else {
            assert_eq!(
                (snapshot.size_removals, snapshot.size_removed_bytes),
                (None, None)
            );
        }
        assert!(snapshot.size_bytes.unwrap() <= 256);
        assert_eq!(snapshot.resident_evictions, None);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka, 1, 1)]
    #[case::quick(TestBackendKind::Quick, 2, 0)]
    #[tokio::test]
    async fn backend_diagnostics_disabled_submissions_and_bypasses(
        #[case] kind: TestBackendKind,
        #[case] writes: u64,
        #[case] bypasses: u64,
    ) {
        let backend = kind.backend(0);
        let cache = LanceCache::with_backend(backend);
        cache
            .insert_with_key(&TestKey::new(1), Arc::new(vec![1]))
            .await;
        cache
            .get_or_insert_with_key(TestKey::new(2), || async { Ok(vec![2]) })
            .await
            .unwrap();
        let snapshot = cache
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(snapshot.backend.enabled, Some(false));
        assert_eq!(snapshot.utilization, None);
        assert_eq!(snapshot.backend.write_attempts, Some(writes));
        assert_eq!(snapshot.backend.disabled_write_rejections, Some(writes));
        assert_eq!(snapshot.backend.disabled_bypasses, Some(bypasses));
        assert_eq!(snapshot.backend.num_entries, Some(0));
        assert_eq!(
            (
                snapshot.activity.loads_started,
                snapshot.activity.loads_succeeded
            ),
            (1, 1)
        );
    }

    #[tokio::test]
    async fn backend_diagnostics_shared_pool_has_separate_wrapper_activity() {
        let backend = Arc::new(QuickCacheBackend::with_capacity(4096));
        let first = LanceCache::with_backend(backend.clone());
        let second = LanceCache::with_backend(backend);
        first
            .insert_with_key(&TestKey::new(1), Arc::new(vec![1]))
            .await;
        assert!(second.get_with_key(&TestKey::new(1)).await.is_some());
        let first = first.diagnostics();
        let second = second.diagnostics();
        assert!(first.backend.pool_id.is_some());
        assert_eq!(first.backend.pool_id, second.backend.pool_id);
        assert_eq!(first.backend.write_attempts, second.backend.write_attempts);
        assert_eq!((first.activity.hits, second.activity.hits), (0, 1));
        assert_eq!(first.backend.num_entries, Some(1));
    }

    #[tokio::test]
    async fn moka_weight_includes_the_fixed_physical_key() {
        let value = Arc::new(vec![0_u32; 3]);
        let expected = cache_entry_size(value.as_ref())
            .checked_add(std::mem::size_of::<InternalCacheKey>())
            .unwrap();
        let cache = LanceCache::with_capacity(expected * 2);
        cache.insert_with_key(&TestKey::new(1), value).await;
        assert_eq!(cache.size_bytes().await, expected);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test]
    async fn deep_size_deduplicates_shared_entry_allocations(
        #[case] backend_kind: TestBackendKind,
    ) {
        let cache = backend_kind.cache(1 << 20);
        let shared_data = Arc::new(vec![0_u8; 1024]);

        for id in 0..2 {
            let data = shared_data.clone();
            cache
                .get_or_insert_with_key(SharedTestKey(id), || async move {
                    Ok(SharedTestValue { data })
                })
                .await
                .unwrap();
        }

        let arc_overhead = std::mem::size_of::<AtomicUsize>() * 2;
        let shared_allocation = std::mem::size_of::<Vec<u8>>() + shared_data.capacity();
        let expected_entries = 2 * std::mem::size_of::<InternalCacheKey>()
            + 2 * (std::mem::size_of::<SharedTestValue>() + arc_overhead)
            + shared_allocation;

        let weighted_size = cache.size_bytes().await;
        assert_eq!(weighted_size, expected_entries + shared_allocation);
        assert_eq!(
            cache.deep_size_of(),
            std::mem::size_of::<LanceCache>() + expected_entries
        );

        let mut context = Context::new();
        assert_eq!(cache.deep_size_of_children(&mut context), expected_entries);
        assert_eq!(
            cache
                .with_key_prefix("another-handle")
                .deep_size_of_children(&mut context),
            0
        );
    }

    #[test]
    fn sizing_can_reenter_the_same_cache() {
        let (done_tx, done_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let cache = LanceCache::with_capacity(4096);
                cache
                    .insert_with_key(&ReentrantKey, Arc::new(ReentrantValue(cache.clone())))
                    .await;
                done_tx.send(()).unwrap();
            });
        });

        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cache insertion deadlocked during sizing");
        worker.join().unwrap();
    }

    #[tokio::test]
    async fn no_cache_computes_each_time() {
        let cache = LanceCache::no_cache();
        let loads = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let loads = loads.clone();
            let value = cache
                .get_or_insert_with_key(TestKey::new(1), move || async move {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![42])
                })
                .await
                .unwrap();
            assert_eq!(value.as_slice(), &[42]);
        }
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert_eq!(cache.size().await, 0);
    }

    #[rstest::rstest]
    #[case::moka(TestBackendKind::Moka)]
    #[case::quick(TestBackendKind::Quick)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_flight_coalesces_success_after_contenders_are_parked(
        #[case] kind: TestBackendKind,
    ) {
        const CONTENDERS: usize = 4;

        let cache = Arc::new(kind.cache(4096));
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let owner = {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            let release = release.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert_with_key_outcome(TestKey::new(10), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        let _ = started_tx.send(());
                        release.notified().await;
                        Ok(vec![10])
                    })
                    .await
            })
        };
        started_rx.await.unwrap();

        let mut contenders = Vec::new();
        let mut parked = Vec::new();
        for _ in 0..CONTENDERS {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
            parked.push(parked_rx);
            contenders.push(tokio::spawn(async move {
                report_first_pending(
                    cache.get_or_insert_with_key_outcome(TestKey::new(10), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![99])
                    }),
                    parked_tx,
                )
                .await
            }));
        }
        for parked in parked {
            parked
                .await
                .expect("contender completed instead of parking behind owner");
        }
        assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
        assert!(contenders.iter().all(|handle| !handle.is_finished()));

        release.notify_one();
        let (owner_value, owner_outcome) = owner.await.unwrap().unwrap();
        assert_eq!(owner_value.as_slice(), &[10]);
        assert_eq!(owner_outcome, CacheLoadOutcome::Loaded);
        for contender in contenders {
            let (value, outcome) = contender.await.unwrap().unwrap();
            assert_eq!(value.as_slice(), &[10]);
            assert_eq!(outcome, CacheLoadOutcome::LoaderSkippedUnknown);
        }
        let (resident, outcome) = cache
            .get_or_insert_with_key_outcome(TestKey::new(10), || async {
                panic!("a resident lookup must skip its loader")
            })
            .await
            .unwrap();
        assert_eq!(resident.as_slice(), &[10]);
        assert_eq!(
            outcome,
            match kind {
                TestBackendKind::Moka => CacheLoadOutcome::LoaderSkippedUnknown,
                TestBackendKind::Quick => CacheLoadOutcome::ResidentHit,
            }
        );
        let stats = cache.stats().await;
        assert_eq!((stats.hits, stats.misses), (CONTENDERS as u64 + 1, 1));
        let activity = cache.diagnostics().activity;
        assert_eq!(
            (
                activity.loads_started,
                activity.loads_succeeded,
                activity.loads_in_flight
            ),
            (1, 1, 0)
        );
        assert_eq!(cache.diagnostics().backend.coalesced_loads, None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_flight_coalesces_errors_after_contenders_are_parked() {
        const CONTENDERS: usize = 4;

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let owner = {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            let release = release.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert_with_key(TestKey::new(20), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        let _ = started_tx.send(());
                        release.notified().await;
                        Err(Error::timeout("owner loader timed out"))
                    })
                    .await
            })
        };
        started_rx.await.unwrap();

        let mut contenders = Vec::new();
        let mut parked = Vec::new();
        for _ in 0..CONTENDERS {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
            parked.push(parked_rx);
            contenders.push(tokio::spawn(async move {
                report_first_pending(
                    cache.get_or_insert_with_key(TestKey::new(20), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        Err(Error::timeout("contender loader timed out"))
                    }),
                    parked_tx,
                )
                .await
            }));
        }
        for parked in parked {
            parked
                .await
                .expect("contender completed instead of parking behind owner");
        }
        assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
        assert!(contenders.iter().all(|handle| !handle.is_finished()));

        release.notify_one();
        assert!(matches!(owner.await.unwrap(), Err(Error::Timeout { .. })));
        for contender in contenders {
            assert!(matches!(
                contender.await.unwrap(),
                Err(Error::Timeout { .. })
            ));
        }
        assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
        let activity = cache.diagnostics().activity;
        assert_eq!(
            (
                activity.loads_started,
                activity.loads_failed,
                activity.loads_in_flight
            ),
            (1, 1, 0)
        );
        assert_eq!(activity.lookup_errors, CONTENDERS as u64 + 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_flight_retries_after_the_owner_is_cancelled() {
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let owner = {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert_with_key(TestKey::new(30), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        let _ = started_tx.send(());
                        std::future::pending::<()>().await;
                        Ok(vec![30])
                    })
                    .await
            })
        };
        started_rx.await.unwrap();

        let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
        let contender = {
            let cache = cache.clone();
            let loader_calls = loader_calls.clone();
            tokio::spawn(async move {
                report_first_pending(
                    cache.get_or_insert_with_key(TestKey::new(30), move || async move {
                        loader_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![31])
                    }),
                    parked_tx,
                )
                .await
            })
        };
        parked_rx
            .await
            .expect("contender completed instead of parking behind owner");
        assert_eq!(loader_calls.load(Ordering::SeqCst), 1);
        assert!(!contender.is_finished());

        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        let value = tokio::time::timeout(std::time::Duration::from_secs(5), contender)
            .await
            .expect("contender remained parked after owner cancellation")
            .unwrap()
            .unwrap();
        assert_eq!(value.as_slice(), &[31]);
        assert_eq!(loader_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn expired_weak_cache_degrades_without_retaining_state() {
        let cache = LanceCache::with_capacity(4096);
        let weak = WeakLanceCache::from(&cache);
        drop(cache);

        assert!(weak.get_with_key(&TestKey::new(1)).await.is_none());
        assert!(
            !weak
                .insert_with_key(&TestKey::new(1), Arc::new(vec![1]))
                .await
        );
        let value = weak
            .get_or_insert_with_key(TestKey::new(1), || async { Ok(vec![7]) })
            .await
            .unwrap();
        assert_eq!(value.as_slice(), &[7]);
    }
}
