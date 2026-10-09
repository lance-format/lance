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

use super::{
    CacheBackendDiagnostics, CacheCodec, CacheLoadOutcome, CacheOccupancyByType,
    CacheOperationContext, CacheSnapshotMode, InternalCacheKey,
};

/// A type-erased cache entry.
pub type CacheEntry = Arc<dyn Any + Send + Sync>;

/// Boxed loader accepted by cache backends.
pub type CacheLoader<'a> = Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>;

/// Low-level pluggable cache backend.
///
/// Implementations store entries keyed by [`InternalCacheKey`] and return
/// type-erased [`CacheEntry`] values.
/// [`LanceCache`](super::LanceCache) handles key construction and type safety;
/// backend authors only need to implement storage and eviction.
#[async_trait]
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
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

    /// Context-aware lookup used by typed cache wrappers.
    ///
    /// The compatibility default delegates to [`get`](Self::get). Override it
    /// only when the backend needs the stable type identity during lookup.
    async fn get_with_context(
        &self,
        key: &InternalCacheKey,
        codec: Option<CacheCodec>,
        _context: CacheOperationContext,
    ) -> Option<CacheEntry> {
        self.get(key, codec).await
    }

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

    /// Context-aware insertion used by typed cache wrappers.
    ///
    /// The compatibility default delegates to [`insert`](Self::insert).
    async fn insert_with_context(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
        _context: CacheOperationContext,
    ) {
        self.insert(key, entry, size_bytes, codec).await;
    }

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
        loader: CacheLoader<'a>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)>;

    /// Context-aware get-or-load used by typed cache wrappers.
    ///
    /// The compatibility default delegates to
    /// [`get_or_insert`](Self::get_or_insert).
    async fn get_or_insert_with_context<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        codec: Option<CacheCodec>,
        _context: CacheOperationContext,
    ) -> Result<(CacheEntry, bool)> {
        self.get_or_insert(key, loader, codec).await
    }

    /// Outcome-aware get-or-load used by typed cache wrappers.
    ///
    /// The compatibility default preserves custom backends and reports a
    /// skipped loader as [`CacheLoadOutcome::LoaderSkippedUnknown`]. Backends
    /// should override this only when their synchronization primitive can
    /// positively distinguish resident and shared results.
    async fn get_or_insert_with_context_outcome<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) -> Result<(CacheEntry, CacheLoadOutcome)> {
        self.get_or_insert_with_context(key, loader, codec, context)
            .await
            .map(|(entry, was_cached)| {
                let outcome = if was_cached {
                    CacheLoadOutcome::LoaderSkippedUnknown
                } else {
                    CacheLoadOutcome::Loaded
                };
                (entry, outcome)
            })
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

    /// Weighted capacity in bytes, or `None` when the backend has no fixed
    /// capacity or cannot report it. Lets callers skip loading entries that
    /// could never be retained.
    fn capacity_bytes(&self) -> Option<usize> {
        None
    }

    /// Cheap diagnostics, with conservative absence for unsupported accounting.
    /// Override this to advertise cheap occupancy and lifecycle measurements.
    /// Custom backends implementing only the original trait remain supported.
    fn diagnostics(&self) -> CacheBackendDiagnostics {
        let capacity_bytes = self.capacity_bytes().map(|capacity| capacity as u64);
        CacheBackendDiagnostics {
            capacity_bytes,
            enabled: capacity_bytes.map(|capacity| capacity > 0),
            ..Default::default()
        }
    }

    /// Request maintenance when collecting a refreshed snapshot. This does not
    /// prevent concurrent writes or make the returned fields atomic.
    async fn diagnostics_with_mode(&self, mode: CacheSnapshotMode) -> CacheBackendDiagnostics {
        if mode == CacheSnapshotMode::Approximate {
            return self.diagnostics();
        }
        let num_entries = self.num_entries().await as u64;
        let size_bytes = self.size_bytes().await as u64;
        let mut snapshot = self.diagnostics();
        snapshot.num_entries = Some(num_entries);
        snapshot.size_bytes = Some(size_bytes);
        snapshot
    }

    /// Collect aggregate diagnostics and optional per-type occupancy.
    ///
    /// The compatibility default keeps custom backends source-compatible and
    /// reports per-type occupancy as unavailable.
    async fn diagnostics_with_types(
        &self,
        mode: CacheSnapshotMode,
    ) -> (CacheBackendDiagnostics, Option<CacheOccupancyByType>) {
        (self.diagnostics_with_mode(mode).await, None)
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
