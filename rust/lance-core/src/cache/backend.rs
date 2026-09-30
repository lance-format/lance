// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Backend interface for cache implementors.
//!
//! This module defines the trait that custom cache backends must implement,
//! along with the entry type they operate on. Most callers should
//! use [`LanceCache`](super::LanceCache) instead of interacting with
//! backends directly.
//!
//! # Migrating custom backends
//!
//! Cache keys are opaque 16-byte values. Store
//! [`InternalCacheKey::as_bytes`] directly instead of decomposing a logical
//! prefix, key string, and Rust type name. The physical namespace must also
//! include [`CACHE_KEY_FORMAT`](super::CACHE_KEY_FORMAT), so a future key
//! protocol produces cold misses instead of aliases. Persistent or tiered
//! backends can route serializable values with [`CacheCodec::type_id`].
//!
//! Prefix invalidation and key inventory are intentionally not part of this
//! interface: one-way digests cannot support either operation without
//! retaining the logical strings that fixed-size keys are designed to remove.
//!
//! # Pinned entries
//!
//! A backend may keep entries in RAM while callers lease them (see the
//! [`pin`](super::pin) module): [`insert_pinned`](CacheBackend::insert_pinned),
//! [`get_or_insert_pinned`](CacheBackend::get_or_insert_pinned) and
//! [`get_leased`](CacheBackend::get_leased). The defaults store such entries
//! like any other, charged and evictable, so existing backends keep working;
//! a pinning backend records each admission with [`CachePin::record`] in a
//! [`PinBudget`](super::PinBudget) and skips pinned entries on eviction.
//! Existing callers should migrate removed symbols as follows:
//! - replace `with_backend_and_prefix(backend, prefix)` with
//!   [`LanceCache::with_backend`](super::LanceCache::with_backend) followed by
//!   [`LanceCache::with_key_prefix`](super::LanceCache::with_key_prefix);
//! - replace `invalidate_prefix` with [`LanceCache::clear`](super::LanceCache::clear)
//!   when clearing the shared backend is acceptable, or rotate a versioned
//!   namespace to leave older entries to age out;
//! - remove uses of `prefix`, `keys`, and session key-inventory methods; opaque
//!   keys have no readable or enumerable equivalent.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Future;

use crate::Result;
use crate::deepsize::Context;

use super::pin::{CachePin, PinnedStats};
use super::{CacheCodec, InternalCacheKey};

/// A type-erased cache entry.
pub type CacheEntry = Arc<dyn Any + Send + Sync>;

/// The loader [`CacheBackend::get_or_insert_pinned`] receives: the entry,
/// its size in bytes and the pin its leases hold.
pub type PinnedEntryLoader<'a> =
    Pin<Box<dyn Future<Output = Result<(CacheEntry, usize, Arc<CachePin>)>> + Send + 'a>>;

/// The tier that would serve a read of an entry, as
/// [`CacheBackend::peek_tier`] predicts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// In RAM: a read is a memory hit.
    Resident,
    /// Not in RAM but held by a local tier below it, such as a persistent
    /// store or a buffer of recently evicted entries: a read costs local I/O
    /// at most, never the caller's loader.
    Local,
    /// Held by no tier: a read runs the caller's loader.
    Absent,
}

/// Low-level pluggable cache backend.
///
/// Implementations store entries keyed by [`InternalCacheKey`] and return
/// type-erased [`CacheEntry`] values.
/// [`LanceCache`](super::LanceCache) handles key construction and type safety;
/// backend authors only need to implement storage and eviction.
#[async_trait]
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
    /// Look up RAM only, without reading or promoting a persistent entry.
    /// Backends without a resident lookup return a miss safely.
    async fn get_resident(&self, _key: &InternalCacheKey) -> Option<CacheEntry> {
        None
    }

    /// Report whether `key` is resident in RAM without counting as an access,
    /// so a residency probe does not change what the backend evicts next.
    /// The default uses [`get_resident`](Self::get_resident), which backends
    /// with recency state may treat as an access.
    async fn peek_resident(&self, key: &InternalCacheKey) -> bool {
        self.get_resident(key).await.is_some()
    }

    /// Predict which tier would serve a read of `key`, without reading the
    /// entry or counting as an access, so the check changes neither what the
    /// backend evicts next nor its hit statistics. The default knows RAM only,
    /// through [`peek_resident`](Self::peek_resident), and reports every other
    /// entry as [`CacheTier::Absent`]; backends with a local tier below RAM
    /// override it.
    async fn peek_tier(&self, key: &InternalCacheKey) -> CacheTier {
        if self.peek_resident(key).await {
            CacheTier::Resident
        } else {
            CacheTier::Absent
        }
    }

    /// Whether layered index planes should gate lower-plane RAM admission on
    /// their sign plane being resident. Backends that admit plane entries
    /// through their ordinary policy return `false`, so lower planes are
    /// loaded and admitted like any other entry.
    fn plane_admission_gated(&self) -> bool {
        true
    }

    /// Read an entry without admitting a persistent hit into RAM.
    /// Backends without this capability safely fall back to resident entries.
    async fn get_without_promotion(
        &self,
        key: &InternalCacheKey,
        _codec: Option<CacheCodec>,
    ) -> Option<CacheEntry> {
        self.get_resident(key).await
    }

    /// Gather selected rows from persistent storage, without RAM admission.
    async fn get_rows(
        &self,
        _key: &InternalCacheKey,
        _rows: &[u32],
        _codec: Option<CacheCodec>,
    ) -> Option<CacheEntry> {
        None
    }

    /// Look up an entry by its key.
    ///
    /// `codec` is provided so that persistent backends can deserialize the
    /// entry from storage. In-memory backends can ignore it. When `codec`
    /// is `None`, the entry type does not support serialization yet and
    /// must be stored in-memory.
    ///
    /// The goal is for all cache entry types to eventually have codecs,
    /// at which point the `Option` will be removed.
    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry>;

    /// Store an entry. `size_bytes` is used for eviction accounting.
    ///
    /// See [`get`](Self::get) for codec semantics.
    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    );

    /// Get an existing entry or compute it from `loader`.
    ///
    /// Implementations should deduplicate concurrent loads for the same key
    /// so the loader runs at most once, unless caching is disabled. Disabled
    /// caches may invoke each caller's loader independently.
    ///
    /// Returns `(entry, was_cached)` where `was_cached` is `true` if the entry
    /// was already present in the cache (the loader was not invoked).
    ///
    /// See [`get`](Self::get) for codec semantics.
    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)>;

    /// Look up RAM only, as an access that refreshes the entry's recency, so
    /// that a caller can lease an entry it finds (see
    /// [`CachePin::lease`]): never reads a persistent tier and counts no miss.
    /// The default uses [`get_resident`](Self::get_resident).
    async fn get_leased(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        self.get_resident(key).await
    }

    /// Store a RAM-only entry that stays resident while `pin` is leased and
    /// the backend's pin budget has room for it. The default stores it like
    /// any entry without a codec: charged, evictable, never pinned.
    async fn insert_pinned(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _pin: &Arc<CachePin>,
    ) {
        self.insert(key, entry, size_bytes, None).await
    }

    /// Get a RAM-only entry or compute it from `loader`, admitting a loaded
    /// entry as [`insert_pinned`](Self::insert_pinned) does. Loads are
    /// deduplicated as in [`get_or_insert`](Self::get_or_insert), which the
    /// default uses, storing the entry without a pin.
    async fn get_or_insert_pinned<'a>(
        &self,
        key: &InternalCacheKey,
        loader: PinnedEntryLoader<'a>,
    ) -> Result<(CacheEntry, bool)> {
        let loader = Box::pin(async move { loader.await.map(|(entry, size, _)| (entry, size)) });
        self.get_or_insert(key, loader, None).await
    }

    /// Size in bytes of the largest entry the backend admits to RAM, or
    /// `None` when it sets no limit below its capacity. A caller can keep a
    /// value that would take most of it out of the cache.
    fn max_entry_bytes(&self) -> Option<u64> {
        None
    }

    /// What the backend's pin budget holds. The default pins nothing.
    fn pinned_stats(&self) -> PinnedStats {
        PinnedStats::default()
    }

    /// Remove all entries.
    async fn clear(&self);

    /// Number of entries currently stored (may flush pending operations).
    async fn num_entries(&self) -> usize;

    /// Total weighted size in bytes of all stored entries (may flush pending operations).
    async fn size_bytes(&self) -> usize;

    /// Approximate number of entries, callable from synchronous contexts.
    /// Backends that cannot provide this cheaply should return 0.
    fn approx_num_entries(&self) -> usize {
        0
    }

    /// Approximate weighted size in bytes, callable from synchronous contexts.
    /// Used as a `DeepSizeOf` fallback when exact entry traversal is unavailable.
    /// Backends that cannot provide this cheaply should return 0.
    ///
    /// Assumes entries do not share underlying buffers; if they do, the
    /// returned total may overcount.
    fn approx_size_bytes(&self) -> usize {
        0
    }

    /// Computes the size of the entries currently held in memory.
    ///
    /// `size_of_entry` threads a shared [`Context`] through each value so
    /// allocations shared by multiple entries are counted once. It returns
    /// `None` when the value's concrete type was not registered by
    /// [`LanceCache`](super::LanceCache); implementations should use the
    /// entry's declared eviction size as a fallback in that case.
    ///
    /// Backends that can enumerate their in-memory entries should include the
    /// physical key footprint in the returned total. The default returns
    /// `None`, causing `LanceCache` to use [`approx_size_bytes`](Self::approx_size_bytes).
    fn deep_size_of_entries(
        &self,
        _context: &mut Context,
        _size_of_entry: &dyn Fn(&CacheEntry, &mut Context) -> Option<usize>,
    ) -> Option<usize> {
        None
    }
}
