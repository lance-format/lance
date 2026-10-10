// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! [`CacheBackend`] backed by [quick_cache](https://crates.io/crates/quick_cache).
//! A hit takes a shard read lock, clones the cached value, and marks it as
//! accessed. There is no read-operation channel or inline eviction work.
//!
//! Entries admitted with a [`CachePin`] stay while leased: quick_cache skips
//! them on eviction ([`PinLifecycle`]), and the strict priority tier does
//! too. A [`PinBudget`] with one partition per shard caps what pins hold, so
//! a shard always has unpinned weight to evict.

use super::{PINNED_PRIORITY, PriorityEntries};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::Future;

use super::backend::{CacheBackend, CacheEntry, PinnedEntryLoader};
use super::moka::key_footprint;
use super::pin::{CachePin, PinBudget, PinRecord, PinnedStats};
use super::{CacheCodec, InternalCacheKey};
use crate::Result;
use crate::deepsize::Context;

#[derive(Clone)]
struct QuickEntry {
    entry: CacheEntry,
    size_bytes: usize,
    /// The admission of a pinned-kind entry in the backend's pin budget,
    /// shared by every copy of the entry, so that it lasts while any tier
    /// or in-flight load still holds the entry.
    pin: Option<Arc<PinRecord>>,
}

impl QuickEntry {
    fn is_pinned(&self) -> bool {
        self.pin.as_ref().is_some_and(|record| record.is_pinned())
    }
}

/// Keeps pinned entries through quick_cache's eviction passes, which check
/// every candidate as they visit it.
#[derive(Clone, Copy, Debug, Default)]
struct PinLifecycle;

impl quick_cache::Lifecycle<InternalCacheKey, QuickEntry> for PinLifecycle {
    type RequestState = ();

    fn is_pinned(&self, _key: &InternalCacheKey, value: &QuickEntry) -> bool {
        value.is_pinned()
    }

    fn begin_request(&self) -> Self::RequestState {}
}

#[derive(Clone)]
struct EntryWeighter;

impl quick_cache::Weighter<InternalCacheKey, QuickEntry> for EntryWeighter {
    fn weight(&self, key: &InternalCacheKey, value: &QuickEntry) -> u64 {
        // Same accounting as the moka backend.
        key_footprint(key).saturating_add(value.size_bytes).max(1) as u64
    }
}

type QuickCache = quick_cache::sync::Cache<
    InternalCacheKey,
    QuickEntry,
    EntryWeighter,
    quick_cache::DefaultHashBuilder,
    PinLifecycle,
>;

pub struct QuickCacheBackend {
    capacity: usize,
    generation: AtomicU64,
    priority_active: AtomicBool,
    priority: Mutex<PriorityEntries<QuickEntry>>,
    cache: QuickCache,
    /// Caps the bytes leased entries pin, per shard.
    pins: Arc<PinBudget>,
}

/// Controls how a [`QuickCacheBackend`] divides its weight budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QuickCacheShardPolicy {
    /// Choose a shard count from the cache capacity and available parallelism.
    #[default]
    Recommended,
    /// Give every entry access to one shared weight budget.
    ///
    /// This avoids capacity fragmentation for a small number of large,
    /// unequal entries. Concurrent hits can still take the shard read lock.
    Single,
}

impl std::fmt::Debug for QuickCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuickCacheBackend")
            .field("entry_count", &self.cache.len())
            .finish()
    }
}

/// Minimum weight budget (4 GiB) per shard: shards don't borrow capacity, and
/// an entry heavier than ~its shard's budget is silently refused admission.
const MIN_SHARD_SHARE: usize = 4 << 30;

/// Recommended shard count: `min(cpus / 2, capacity / 4 GiB)`, power of two
/// in `[1, 1024]`. The cpu term bounds lock contention; the capacity term
/// keeps each shard's budget >= 4 GiB so large entries stay admissible.
/// Rounded down because quick_cache rounds requests up.
pub fn recommended_cache_shards(capacity: usize) -> usize {
    let available_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    recommended_cache_shards_for_parallelism(capacity, available_parallelism)
}

fn recommended_cache_shards_for_parallelism(
    capacity: usize,
    available_parallelism: usize,
) -> usize {
    let by_cpu = available_parallelism / 2;
    let shards = (capacity / MIN_SHARD_SHARE).min(by_cpu).max(1);
    let shards = if shards.is_power_of_two() {
        shards
    } else {
        shards.next_power_of_two() / 2
    };
    shards.clamp(1, 1024)
}

/// Assumed average entry size for pre-allocation sizing.
const ESTIMATED_AVG_ENTRY_BYTES: usize = 64 << 10;

impl QuickCacheBackend {
    /// Create a backend holding up to `capacity` bytes of weighted entries
    /// (weight = key footprint + declared size), sharded per
    /// [`recommended_cache_shards`].
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_shard_policy(capacity, QuickCacheShardPolicy::Recommended)
    }

    /// Create a weighted cache with an explicit shard policy.
    ///
    /// `capacity` bounds the sum of key footprints and declared entry sizes.
    /// Each shard has an independent share of that bound. In addition, the
    /// default Quick admission policy rejects an unpinned entry heavier than
    /// approximately 97% of one shard's share.
    ///
    /// # Example
    ///
    /// ```
    /// use lance_core::cache::{QuickCacheBackend, QuickCacheShardPolicy};
    ///
    /// let cache = QuickCacheBackend::with_shard_policy(
    ///     8 << 30,
    ///     QuickCacheShardPolicy::Single,
    /// );
    /// ```
    pub fn with_shard_policy(capacity: usize, shard_policy: QuickCacheShardPolicy) -> Self {
        let shards = match shard_policy {
            QuickCacheShardPolicy::Recommended => recommended_cache_shards(capacity),
            QuickCacheShardPolicy::Single => 1,
        };
        Self::with_shards(capacity, shards)
    }

    fn with_shards(capacity: usize, shards: usize) -> Self {
        // Floor protects the shard count from quick_cache's items-per-shard
        // heuristic; ceiling bounds pre-allocation.
        let estimated_items = (capacity / ESTIMATED_AVG_ENTRY_BYTES).clamp(shards * 32, 1_000_000);
        let options = quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(estimated_items)
            .weight_capacity(capacity as u64)
            .shards(shards)
            .build()
            // Only errors when weight/item capacity is missing; both are set.
            .expect("quick_cache options");
        let cache =
            QuickCache::with_options(options, EntryWeighter, Default::default(), PinLifecycle);
        let pins = Arc::new(PinBudget::new(capacity as u64, cache.num_shards()));
        Self {
            cache,
            capacity,
            generation: AtomicU64::new(0),
            priority_active: AtomicBool::new(false),
            priority: Mutex::new(PriorityEntries::default()),
            pins,
        }
    }

    /// An admission of a pinned-kind entry, charged what the weigher charges.
    fn pin_record(
        &self,
        key: &InternalCacheKey,
        size_bytes: usize,
        pin: &Arc<CachePin>,
    ) -> Arc<PinRecord> {
        let bytes = key_footprint(key).saturating_add(size_bytes) as u64;
        Arc::new(CachePin::record(
            pin,
            &self.pins,
            self.cache.shard_index(key),
            bytes,
        ))
    }

    /// Priority of `item` in the strict tier: pinned-kind entries outrank
    /// every plane, and entries without a plane priority rank with signs.
    fn strict_priority(item: &QuickEntry, priority: u8) -> u8 {
        match (item.pin.is_some(), priority) {
            (true, _) => PINNED_PRIORITY,
            (false, 0) => 3,
            (false, priority) => priority,
        }
    }
    fn admit_priority(
        &self,
        key: InternalCacheKey,
        item: QuickEntry,
        priority: u8,
        generation: u64,
    ) {
        self.cache.remove(&key);
        let dropped = {
            let mut entries = self.priority.lock().unwrap_or_else(|e| e.into_inner());
            if self.generation.load(Ordering::Acquire) != generation {
                return;
            }
            let mut dropped = Vec::new();
            if !self.priority_active.swap(true, Ordering::AcqRel) {
                // Metadata and existing entries compete with signs, ahead of ex planes.
                // Reserving the whole budget for planes would otherwise evict the IVF model.
                let existing: Vec<_> = self.cache.iter().collect();
                for (key, value) in existing {
                    let size = key_footprint(&key).saturating_add(value.size_bytes);
                    if value.pin.is_some() {
                        // Eviction below cannot move a pinned entry, so move
                        // it here rather than hold it in both tiers.
                        dropped.extend(
                            self.cache
                                .remove_if(&key, |held| held.pin.is_some())
                                .map(|(_, held)| held),
                        );
                    }
                    let priority = Self::strict_priority(&value, 0);
                    dropped.extend(entries.insert_with(
                        key,
                        value,
                        size,
                        priority,
                        self.capacity,
                        QuickEntry::is_pinned,
                    ));
                }
                // Evict resident values without invalidating single-flight
                // placeholders: their loaders still belong to this generation.
                self.cache.set_capacity(0);
            }
            let size = key_footprint(&key).saturating_add(item.size_bytes);
            let priority = Self::strict_priority(&item, priority);
            dropped.extend(entries.insert_with(
                key,
                item,
                size,
                priority,
                self.capacity,
                QuickEntry::is_pinned,
            ));
            dropped
        };
        drop(dropped);
    }

    #[cfg(test)]
    fn num_shards(&self) -> usize {
        self.cache.num_shards()
    }

    #[cfg(test)]
    fn shard_index(&self, key: &InternalCacheKey) -> usize {
        self.cache.shard_index(key)
    }
}

#[async_trait]
impl CacheBackend for QuickCacheBackend {
    async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        self.priority
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .or_else(|| self.cache.get(key))
            .map(|r| r.entry)
    }

    /// Membership checks only: priority stamps and quick_cache's reference
    /// bits stay unchanged, so a probed entry is evicted as if never probed.
    async fn peek_resident(&self, key: &InternalCacheKey) -> bool {
        self.priority
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(key)
            || self.cache.contains_key(key)
    }

    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry> {
        if (self.priority_active.load(Ordering::Acquire)
            || codec.is_some_and(|c| c.memory_priority() > 0))
            && let Some(value) = self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(key)
        {
            return Some(value.entry);
        }
        self.cache.get(key).map(|v| v.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    ) {
        let priority = codec.map(|c| c.memory_priority()).unwrap_or(0);
        let item = QuickEntry {
            entry,
            size_bytes,
            pin: None,
        };
        if priority > 0 || self.priority_active.load(Ordering::Acquire) {
            self.admit_priority(
                *key,
                item,
                priority,
                self.generation.load(Ordering::Acquire),
            );
        } else {
            self.cache.insert(*key, item);
        }
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let priority = codec.map(|c| c.memory_priority()).unwrap_or(0);
        if (priority > 0 || self.priority_active.load(Ordering::Acquire))
            && let Some(value) = self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(key)
        {
            return Ok((value.entry, true));
        }
        let generation = self.generation.load(Ordering::Acquire);
        match self.cache.get_value_or_guard_async(key).await {
            Ok(value) => Ok((value.entry, true)),
            Err(guard) => {
                let (entry, size_bytes) = loader.await?;
                let item = QuickEntry {
                    entry: entry.clone(),
                    size_bytes,
                    pin: None,
                };
                if guard.insert(item.clone()).is_ok()
                    && (priority > 0 || self.priority_active.load(Ordering::Acquire))
                {
                    self.admit_priority(*key, item, priority, generation);
                }
                Ok((entry, false))
            }
        }
    }

    /// A RAM hit refreshes the entry's recency like [`get`](Self::get): a
    /// quick_cache hit sets its reference bit, which an eviction pass would
    /// otherwise have cleared while the entry was pinned.
    async fn get_leased(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        self.get_resident(key).await
    }

    async fn insert_pinned(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        pin: &Arc<CachePin>,
    ) {
        let item = QuickEntry {
            entry,
            size_bytes,
            pin: Some(self.pin_record(key, size_bytes, pin)),
        };
        if self.priority_active.load(Ordering::Acquire) {
            self.admit_priority(
                *key,
                item,
                PINNED_PRIORITY,
                self.generation.load(Ordering::Acquire),
            );
        } else {
            self.cache.insert(*key, item);
        }
    }

    async fn get_or_insert_pinned<'a>(
        &self,
        key: &InternalCacheKey,
        loader: PinnedEntryLoader<'a>,
    ) -> Result<(CacheEntry, bool)> {
        if self.priority_active.load(Ordering::Acquire)
            && let Some(value) = self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(key)
        {
            return Ok((value.entry, true));
        }
        let generation = self.generation.load(Ordering::Acquire);
        match self.cache.get_value_or_guard_async(key).await {
            Ok(value) => Ok((value.entry, true)),
            Err(guard) => {
                let (entry, size_bytes, pin) = loader.await?;
                let item = QuickEntry {
                    entry: entry.clone(),
                    size_bytes,
                    pin: Some(self.pin_record(key, size_bytes, &pin)),
                };
                if guard.insert(item.clone()).is_ok()
                    && self.priority_active.load(Ordering::Acquire)
                {
                    self.admit_priority(*key, item, PINNED_PRIORITY, generation);
                }
                Ok((entry, false))
            }
        }
    }

    /// A shard's budget: quick_cache refuses an unpinned entry heavier than
    /// about its shard's weight.
    fn max_entry_bytes(&self) -> Option<u64> {
        Some((self.capacity / self.cache.num_shards()) as u64)
    }

    fn pinned_stats(&self) -> PinnedStats {
        self.pins.stats()
    }

    async fn clear(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let dropped = {
            let mut entries = self.priority.lock().unwrap_or_else(|e| e.into_inner());
            let dropped = entries.clear();
            self.priority_active.store(false, Ordering::Release);
            self.cache.set_capacity(self.capacity as u64);
            dropped
        };
        drop(dropped);
        self.cache.clear();
    }

    async fn num_entries(&self) -> usize {
        self.cache.len()
            + self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
    }

    async fn size_bytes(&self) -> usize {
        self.cache.weight() as usize
            + self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .bytes()
    }

    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity)
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.len()
            + self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len()
    }

    fn approx_size_bytes(&self) -> usize {
        self.cache.weight() as usize
            + self
                .priority
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .bytes()
    }

    fn deep_size_of_entries(
        &self,
        context: &mut Context,
        size_of_entry: &dyn Fn(&CacheEntry, &mut Context) -> Option<usize>,
    ) -> Option<usize> {
        let prioritized: usize = self
            .priority
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot()
            .into_iter()
            .map(|(key, _, value)| {
                key_footprint(&key)
                    + size_of_entry(&value.entry, context).unwrap_or(value.size_bytes)
            })
            .sum();
        Some(
            prioritized
                + self
                    .cache
                    .iter()
                    .map(|(key, record)| {
                        key_footprint(&key)
                            + size_of_entry(&record.entry, context).unwrap_or(record.size_bytes)
                    })
                    .sum::<usize>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::cache::{CacheKey, CacheLease, CacheTier, LanceCache};

    const TEST_CAPACITY: usize = 1_000;

    struct TestKey<T: 'static> {
        key: String,
        _phantom: PhantomData<T>,
    }

    impl<T: 'static> TestKey<T> {
        fn new(key: &str) -> Self {
            Self {
                key: key.to_string(),
                _phantom: PhantomData,
            }
        }
    }

    impl<T: 'static> CacheKey for TestKey<T> {
        type ValueType = T;
        fn key(&self) -> std::borrow::Cow<'_, str> {
            std::borrow::Cow::Borrowed(&self.key)
        }
        fn type_name() -> &'static str {
            std::any::type_name::<T>()
        }
    }

    #[tokio::test]
    async fn priority_transition_preserves_inflight_loads() {
        let cache = Arc::new(QuickCacheBackend::with_capacity(1024));
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let codec = CacheCodec::new("test.priority", 1, |_, _| Ok(()), |_| Ok(Arc::new(())))
            .with_memory_priority(3);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let loading_cache = cache.clone();
        let loading = tokio::spawn(async move {
            loading_cache
                .get_or_insert(
                    &key(1),
                    Box::pin(async move {
                        started_tx.send(()).unwrap();
                        release_rx.await.unwrap();
                        Ok((Arc::new(1u64) as CacheEntry, 32))
                    }),
                    Some(codec),
                )
                .await
                .unwrap()
        });
        started_rx.await.unwrap();
        cache.insert(&key(2), Arc::new(2u64), 32, Some(codec)).await;
        release_tx.send(()).unwrap();
        loading.await.unwrap();
        assert!(cache.get_resident(&key(1)).await.is_some());
        assert!(cache.get_resident(&key(2)).await.is_some());
    }

    #[rstest::rstest]
    #[case::recommended(QuickCacheShardPolicy::Recommended)]
    #[case::single(QuickCacheShardPolicy::Single)]
    #[tokio::test]
    async fn priority_budget_retains_metadata_and_singleflight_entries(
        #[case] shard_policy: QuickCacheShardPolicy,
    ) {
        let cache = QuickCacheBackend::with_shard_policy(160, shard_policy);
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let codec = CacheCodec::new("test.priority", 1, |_, _| Ok(()), |_| Ok(Arc::new(())));
        cache.insert(&key(0), Arc::new(0u64), 32, None).await;
        for (id, priority) in [(1, 3), (2, 2), (3, 1)] {
            cache
                .insert(
                    &key(id),
                    Arc::new(id),
                    32,
                    Some(codec.with_memory_priority(priority)),
                )
                .await;
        }
        assert!(cache.get_resident(&key(0)).await.is_some());
        assert!(cache.get_resident(&key(1)).await.is_some());
        assert!(cache.get_resident(&key(2)).await.is_some());
        assert!(cache.get_resident(&key(3)).await.is_none());
        // Priority admission disables Quick's ordinary budget, but callers
        // must still see the configured budget for the active priority tier.
        assert_eq!(cache.capacity_bytes(), Some(160));
        let (_, hit) = cache
            .get_or_insert(
                &key(4),
                Box::pin(async { Ok((Arc::new(4u64) as CacheEntry, 32)) }),
                None,
            )
            .await
            .unwrap();
        assert!(!hit);
        let (_, hit) = cache
            .get_or_insert(
                &key(4),
                Box::pin(async { panic!("resident entry reloaded") }),
                None,
            )
            .await
            .unwrap();
        assert!(hit);
        assert!(cache.get_resident(&key(2)).await.is_none());
        assert!(cache.size_bytes().await <= 160);
        cache.clear().await;
        assert_eq!(cache.num_entries().await, 0);
        assert_eq!(cache.capacity_bytes(), Some(160));
        cache.insert(&key(0), Arc::new(0u64), 32, None).await;
        assert!(cache.get_resident(&key(0)).await.is_some());
    }

    /// Fill a single-shard cache so that `victim` is its only cold entry,
    /// then peek at its tier or get it, and insert one more entry.
    async fn cold_victim_after(touch_with_get: bool) -> QuickCacheBackend {
        const ENTRY_BYTES: usize = 100;
        const HOT_ENTRIES: u8 = 9;
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let entry_weight = key_footprint(&key(0)) + ENTRY_BYTES;
        let capacity = entry_weight * (usize::from(HOT_ENTRIES) + 1);
        let cache = QuickCacheBackend::with_capacity(capacity);
        for id in 0..HOT_ENTRIES {
            cache
                .insert(&key(id), Arc::new(id), ENTRY_BYTES, None)
                .await;
        }
        let victim = key(HOT_ENTRIES);
        cache
            .insert(&victim, Arc::new(0u8), ENTRY_BYTES, None)
            .await;
        if touch_with_get {
            assert!(cache.get_resident(&victim).await.is_some());
        } else {
            assert_eq!(cache.peek_tier(&victim).await, CacheTier::Resident);
        }
        cache
            .insert(&key(HOT_ENTRIES + 1), Arc::new(0u8), ENTRY_BYTES, None)
            .await;
        cache
    }

    #[tokio::test]
    async fn tier_peek_leaves_eviction_order_unchanged() {
        let victim = InternalCacheKey::from_bytes([9; 16]);
        let peeked = cold_victim_after(false).await;
        assert_eq!(peeked.peek_tier(&victim).await, CacheTier::Absent);
        assert!(!peeked.cache.contains_key(&victim));

        // A real access sets the reference bit, so the same victim survives.
        let touched = cold_victim_after(true).await;
        assert_eq!(touched.peek_tier(&victim).await, CacheTier::Resident);
    }

    /// The priority tier evicts its oldest stamp first; a tier peek leaves
    /// the stamp alone where a get refreshes it.
    #[tokio::test]
    async fn tier_peek_leaves_priority_order_unchanged() {
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        let codec = CacheCodec::new("test.priority", 1, |_, _| Ok(()), |_| Ok(Arc::new(())))
            .with_memory_priority(1);
        for touch_with_get in [false, true] {
            // Room for three 48-byte entries (32 bytes plus the 16-byte key).
            let cache = QuickCacheBackend::with_capacity(160);
            for id in 1..=3 {
                cache
                    .insert(&key(id), Arc::new(u64::from(id)), 32, Some(codec))
                    .await;
            }
            if touch_with_get {
                assert!(cache.get_resident(&key(1)).await.is_some());
            } else {
                assert_eq!(cache.peek_tier(&key(1)).await, CacheTier::Resident);
            }
            cache.insert(&key(4), Arc::new(4u64), 32, Some(codec)).await;
            let (survivor, evicted) = if touch_with_get {
                (key(1), key(2))
            } else {
                (key(2), key(1))
            };
            assert_eq!(
                cache.peek_tier(&evicted).await,
                CacheTier::Absent,
                "get={touch_with_get}"
            );
            assert_eq!(
                cache.peek_tier(&survivor).await,
                CacheTier::Resident,
                "get={touch_with_get}"
            );
        }
    }

    /// A value leased through its own pin, charged `bytes` beyond its struct.
    struct Pinned {
        pin: Arc<CachePin>,
        bytes: usize,
    }

    impl crate::deepsize::DeepSizeOf for Pinned {
        fn deep_size_of_children(&self, _context: &mut Context) -> usize {
            self.bytes
        }
    }

    impl crate::cache::PinnedValue for Pinned {
        fn cache_pin(&self) -> &Arc<CachePin> {
            &self.pin
        }
    }

    /// Capacity of the caches the pinning tests churn: one shard, as the
    /// benchmark's caches below 4 GiB have.
    const PIN_TEST_CAPACITY: usize = 1 << 20;
    /// Bytes of each churned entry, as a small plane.
    const CHURN_ENTRY_BYTES: usize = 4096;

    fn single_shard_cache() -> (Arc<QuickCacheBackend>, LanceCache) {
        let backend = Arc::new(QuickCacheBackend::with_shard_policy(
            PIN_TEST_CAPACITY,
            QuickCacheShardPolicy::Single,
        ));
        assert_eq!(backend.cache.num_shards(), 1);
        assert_eq!(backend.capacity_bytes(), Some(PIN_TEST_CAPACITY));
        (backend.clone(), LanceCache::with_backend(backend))
    }

    /// Admit `bytes` of entries under `prefix`, reading each back so that it
    /// is promoted to the hot ring and pushes older hot entries out, as the
    /// planes queries read do.
    async fn churn(cache: &LanceCache, prefix: &str, bytes: usize) {
        for i in 0..bytes / CHURN_ENTRY_BYTES {
            let key = TestKey::<Vec<u8>>::new(&format!("{prefix}-{i}"));
            cache
                .insert_with_key(&key, Arc::new(vec![0u8; CHURN_ENTRY_BYTES]))
                .await;
            cache.get_with_key(&key).await;
        }
    }

    /// Load `name`, a pinned value of `bytes`, and lease it.
    async fn leased(cache: &LanceCache, name: &str, bytes: usize) -> (Arc<Pinned>, CacheLease) {
        let (value, lease, _) = cache
            .get_or_insert_leased_with_key(TestKey::<Pinned>::new(name), || async move {
                Ok(Pinned {
                    pin: CachePin::new(),
                    bytes,
                })
            })
            .await
            .unwrap();
        (value, lease)
    }

    async fn is_resident(cache: &LanceCache, name: &str) -> bool {
        cache
            .peek_resident_with_key(&TestKey::<Pinned>::new(name))
            .await
    }

    /// A store of 30% of a single-shard cache stays through ten times the
    /// capacity of plane churn while leased, with planes and the store
    /// within the capacity; unleased, the churn evicts it.
    #[tokio::test]
    async fn pinned_entry_survives_pressure_while_leased() {
        let (_, cache) = single_shard_cache();
        let store_bytes = PIN_TEST_CAPACITY * 3 / 10;
        let (_store, lease) = leased(&cache, "store", store_bytes).await;
        assert!(lease.is_pinned());
        for round in 0..10 {
            churn(&cache, &format!("planes-{round}"), PIN_TEST_CAPACITY).await;
            assert!(is_resident(&cache, "store").await, "round {round}");
            assert!(
                cache.size_bytes().await <= PIN_TEST_CAPACITY,
                "round {round}"
            );
        }
        let stats = cache.pinned_stats();
        assert_eq!((stats.pinned_entries, stats.overflow), (1, 0), "{stats:?}");
        assert!(stats.pinned_bytes >= store_bytes as u64, "{stats:?}");

        drop(lease);
        churn(&cache, "idle", 10 * PIN_TEST_CAPACITY).await;
        assert!(!is_resident(&cache, "store").await);
        assert_eq!(cache.pinned_stats().pinned_bytes, 0);
    }

    /// Entries admitted once the hot ring holds its target enter the cold
    /// ring, where churn evicts an idle one first; a leased one stays.
    #[tokio::test]
    async fn pinned_entry_admitted_leased_above_hot_target() {
        let (backend, cache) = single_shard_cache();
        for i in 0..PIN_TEST_CAPACITY / CHURN_ENTRY_BYTES {
            let key = TestKey::<Vec<u8>>::new(&format!("fill-{i}"));
            cache
                .insert_with_key(&key, Arc::new(vec![0u8; CHURN_ENTRY_BYTES]))
                .await;
        }
        let (_store, lease) = leased(&cache, "store", PIN_TEST_CAPACITY / 5).await;
        assert!(lease.is_pinned());
        let idle = Arc::new(Pinned {
            pin: CachePin::new(),
            bytes: PIN_TEST_CAPACITY / 5,
        });
        cache
            .insert_pinned_with_key(&TestKey::<Pinned>::new("idle"), idle)
            .await;
        assert!(is_resident(&cache, "idle").await);
        churn(&cache, "planes", 10 * PIN_TEST_CAPACITY).await;
        assert!(is_resident(&cache, "store").await);
        assert!(!is_resident(&cache, "idle").await);
        assert!(backend.size_bytes().await <= PIN_TEST_CAPACITY);
    }

    /// A leased lookup is an access, as a get is: an idle entry found
    /// through it keeps the reference bit a peek leaves clear, so it
    /// survives the eviction pass that takes a peeked entry.
    #[tokio::test]
    async fn leased_get_refreshes_referenced() {
        const ENTRY_BYTES: usize = 100;
        const HOT_ENTRIES: u8 = 9;
        let key = |id| InternalCacheKey::from_bytes([id; 16]);
        for lease_with_get in [false, true] {
            let entry_weight = key_footprint(&key(0)) + ENTRY_BYTES;
            let cache =
                QuickCacheBackend::with_capacity(entry_weight * (usize::from(HOT_ENTRIES) + 1));
            for id in 0..HOT_ENTRIES {
                cache
                    .insert(&key(id), Arc::new(id), ENTRY_BYTES, None)
                    .await;
            }
            // The only cold entry, admitted leased and then left idle.
            let victim = key(HOT_ENTRIES);
            let pin = CachePin::new();
            let lease = CachePin::lease(&pin);
            cache
                .insert_pinned(&victim, Arc::new(0u8), ENTRY_BYTES, &pin)
                .await;
            assert!(lease.is_pinned());
            drop(lease);
            if lease_with_get {
                assert!(cache.get_leased(&victim).await.is_some());
                drop(CachePin::lease(&pin));
            } else {
                assert!(cache.peek_resident(&victim).await);
            }
            cache
                .insert(&key(HOT_ENTRIES + 1), Arc::new(0u8), ENTRY_BYTES, None)
                .await;
            assert_eq!(
                cache.peek_resident(&victim).await,
                lease_with_get,
                "get={lease_with_get}"
            );
        }
    }

    /// Two leased stores of 30% each under the 50% cap: the second counts
    /// an overflow and churn evicts it, and it is pinned again when admitted
    /// once the first is released.
    #[tokio::test]
    async fn pinned_cap_leaves_overflow_unpinned() {
        let (_, cache) = single_shard_cache();
        let store_bytes = PIN_TEST_CAPACITY * 3 / 10;
        let (_first, first) = leased(&cache, "first", store_bytes).await;
        let (second_store, second) = leased(&cache, "second", store_bytes).await;
        assert!(first.is_pinned());
        assert!(!second.is_pinned() && second.pin().is_overflowed());
        let stats = cache.pinned_stats();
        assert_eq!(stats.cap_bytes, PIN_TEST_CAPACITY as u64 / 2);
        assert_eq!(
            (stats.pinned_entries, stats.leased_entries, stats.overflow),
            (1, 2, 1)
        );
        churn(&cache, "planes", 10 * PIN_TEST_CAPACITY).await;
        assert!(is_resident(&cache, "first").await);
        assert!(!is_resident(&cache, "second").await);

        drop(first);
        let key = TestKey::<Pinned>::new("second");
        assert!(
            cache
                .ensure_pinned_with_key(&key, || second_store.clone())
                .await
        );
        assert!(second.is_pinned());
    }

    /// However many stores are leased, pins hold at most the cap, and
    /// churn stays within the capacity plus one entry.
    #[tokio::test]
    async fn pins_never_push_inserts_over_capacity() {
        let (_, cache) = single_shard_cache();
        let store_bytes = PIN_TEST_CAPACITY / 5;
        let mut stores = Vec::new();
        for store in 0..4 {
            stores.push(leased(&cache, &format!("store-{store}"), store_bytes).await);
        }
        let pinned = stores.iter().filter(|(_, lease)| lease.is_pinned()).count();
        assert_eq!(pinned, 2);
        let slack = CHURN_ENTRY_BYTES + 1024;
        for round in 0..10 {
            churn(&cache, &format!("planes-{round}"), PIN_TEST_CAPACITY).await;
            let bytes = cache.size_bytes().await;
            assert!(bytes <= PIN_TEST_CAPACITY + slack, "round {round}: {bytes}");
            let stats = cache.pinned_stats();
            assert!(stats.pinned_bytes <= stats.cap_bytes, "{stats:?}");
        }
        for (store, (_, lease)) in stores.iter().enumerate() {
            let name = format!("store-{store}");
            assert_eq!(
                is_resident(&cache, &name).await,
                lease.is_pinned(),
                "{name}"
            );
        }
    }

    /// Concurrent first callers load a pinned entry once and all lease it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn get_or_insert_pinned_is_single_flight() {
        const CALLERS: usize = 32;
        let (_, cache) = single_shard_cache();
        let loads = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(tokio::sync::Barrier::new(CALLERS));
        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                let (cache, loads, barrier) = (cache.clone(), loads.clone(), barrier.clone());
                tokio::spawn(async move {
                    barrier.wait().await;
                    cache
                        .get_or_insert_leased_with_key(
                            TestKey::<Pinned>::new("store"),
                            || async move {
                                loads.fetch_add(1, Ordering::SeqCst);
                                tokio::task::yield_now().await;
                                Ok(Pinned {
                                    pin: CachePin::new(),
                                    bytes: 1024,
                                })
                            },
                        )
                        .await
                        .unwrap()
                })
            })
            .collect();
        let mut results = Vec::with_capacity(CALLERS);
        for caller in callers {
            results.push(caller.await.unwrap());
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        let store = results[0].0.clone();
        assert!(
            results
                .iter()
                .all(|(value, lease, _)| Arc::ptr_eq(value, &store) && lease.is_pinned())
        );
        assert_eq!(results.iter().filter(|(_, _, hit)| !hit).count(), 1);
        assert_eq!(store.pin.holders(), CALLERS);
        drop(results);
        assert_eq!(store.pin.holders(), 0);
        assert!(!store.pin.is_pinned());
    }

    /// The largest admissible entry is a shard's weight budget, and the pin
    /// cap is half of every shard's.
    #[test]
    fn max_entry_bytes_is_shard_budget() {
        for capacity in [0, PIN_TEST_CAPACITY, 16 << 30] {
            let backend = QuickCacheBackend::with_capacity(capacity);
            let shards = backend.cache.num_shards();
            let shard_bytes = (capacity / shards) as u64;
            assert_eq!(backend.max_entry_bytes(), Some(shard_bytes), "{capacity}");
            let cap = (shard_bytes as f64 * crate::cache::PINNED_CAP_FRACTION) as u64;
            assert_eq!(
                backend.pinned_stats().cap_bytes,
                cap * shards as u64,
                "{capacity}"
            );
        }
    }

    /// A clear drops a leased entry and its pin; admitting it again pins it
    /// again through the leases it still has.
    #[tokio::test]
    async fn recharge_after_clear() {
        let (_, cache) = single_shard_cache();
        let (store, lease) = leased(&cache, "store", 1 << 16).await;
        assert!(lease.is_pinned());
        cache.clear().await;
        assert!(!is_resident(&cache, "store").await);
        assert!(!lease.is_pinned());
        assert_eq!(cache.pinned_stats().leased_entries, 0);
        let key = TestKey::<Pinned>::new("store");
        assert!(cache.ensure_pinned_with_key(&key, || store.clone()).await);
        assert!(lease.is_pinned());
        assert!(
            !cache
                .ensure_pinned_with_key(&key, || panic!("admitted twice"))
                .await
        );
        let stats = cache.pinned_stats();
        assert_eq!((stats.pinned_entries, stats.leased_entries), (1, 1));
    }

    /// Once sign-priority planes activate the strict tier, a leased store
    /// stays through their churn, moved there with the pinned priority, and
    /// an overflowed store outranks the planes too.
    #[tokio::test]
    async fn strict_tier_keeps_leased_pinned_entries() {
        let (backend, cache) = single_shard_cache();
        let store_bytes = PIN_TEST_CAPACITY * 3 / 10;
        let (_store, lease) = leased(&cache, "store", store_bytes).await;
        let (_other, other) = leased(&cache, "other", store_bytes).await;
        assert!(lease.is_pinned() && !other.is_pinned());
        let codec = CacheCodec::new("test.sign", 1, |_, _| Ok(()), |_| Ok(Arc::new(())))
            .with_memory_priority(3);
        for id in 0..(10 * PIN_TEST_CAPACITY / CHURN_ENTRY_BYTES) as u64 {
            let mut bytes = [0xAB; 16];
            bytes[..8].copy_from_slice(&id.to_le_bytes());
            let key = InternalCacheKey::from_bytes(bytes);
            backend
                .insert(&key, Arc::new(id), CHURN_ENTRY_BYTES, Some(codec))
                .await;
        }
        assert!(backend.priority_active.load(Ordering::Acquire));
        assert!(is_resident(&cache, "store").await);
        assert!(is_resident(&cache, "other").await);
        assert!(backend.size_bytes().await <= PIN_TEST_CAPACITY);
        assert_eq!(cache.pinned_stats().pinned_entries, 1);
    }

    async fn pin_budget_cache(is_strict: bool) -> QuickCacheBackend {
        let backend =
            QuickCacheBackend::with_shard_policy(TEST_CAPACITY, QuickCacheShardPolicy::Single);
        if is_strict {
            let codec = CacheCodec::new("test.sign", 1, |_, _| Ok(()), |_| Ok(Arc::new(())))
                .with_memory_priority(3);
            backend
                .insert(
                    &InternalCacheKey::from_bytes([u8::MAX; 16]),
                    Arc::new(()),
                    1,
                    Some(codec),
                )
                .await;
        }
        assert_eq!(backend.priority_active.load(Ordering::Acquire), is_strict);
        backend
    }

    #[rstest::rstest]
    #[case::quick(false)]
    #[case::strict(true)]
    #[tokio::test]
    async fn shared_pin_aliases_obey_capacity(#[case] is_strict: bool) {
        let backend = pin_budget_cache(is_strict).await;
        let pin = CachePin::new();
        let _lease = CachePin::lease(&pin);
        let entry_bytes = TEST_CAPACITY * 3 / 10;
        let value = Arc::new(vec![0u8; entry_bytes]);
        for id in 0..4 {
            backend
                .insert_pinned(
                    &InternalCacheKey::from_bytes([id; 16]),
                    value.clone(),
                    entry_bytes,
                    &pin,
                )
                .await;
        }
        assert!(
            backend.size_bytes().await <= TEST_CAPACITY,
            "resident bytes={}, pin stats={:?}",
            backend.size_bytes().await,
            backend.pinned_stats(),
        );
        let stats = backend.pinned_stats();
        assert_eq!(stats.pinned_entries, 1);
        assert_eq!(
            stats.pinned_bytes,
            (entry_bytes + key_footprint(&InternalCacheKey::from_bytes([0; 16]))) as u64,
        );
    }

    #[rstest::rstest]
    #[case::quick(false)]
    #[case::strict(true)]
    #[tokio::test]
    async fn shared_pin_backends_obey_capacity(#[case] is_strict: bool) {
        let first = pin_budget_cache(is_strict).await;
        let second =
            QuickCacheBackend::with_shard_policy(10 * TEST_CAPACITY, QuickCacheShardPolicy::Single);
        let entry_bytes = TEST_CAPACITY * 3 / 10;
        let mut leases = Vec::with_capacity(4);
        for id in 0..4 {
            let pin = CachePin::new();
            leases.push(CachePin::lease(&pin));
            let value = Arc::new(vec![0u8; entry_bytes]);
            let key = InternalCacheKey::from_bytes([id; 16]);
            first
                .insert_pinned(&key, value.clone(), entry_bytes, &pin)
                .await;
            second.insert_pinned(&key, value, entry_bytes, &pin).await;
        }
        assert!(leases.iter().all(CacheLease::is_pinned));
        assert!(
            first.size_bytes().await <= TEST_CAPACITY,
            "first bytes={}, first stats={:?}, second stats={:?}",
            first.size_bytes().await,
            first.pinned_stats(),
            second.pinned_stats(),
        );
        let first_stats = first.pinned_stats();
        assert_eq!(first_stats.pinned_entries, 1);
        assert_eq!(second.pinned_stats().pinned_entries, 4);
        first.clear().await;
        assert_eq!(first.pinned_stats().pinned_bytes, 0);
        assert_eq!(second.pinned_stats().pinned_entries, 4);
    }

    #[rstest::rstest]
    #[case::quick(false)]
    #[case::strict(true)]
    #[tokio::test]
    async fn shared_pin_replacement_reacquires_its_reservation(#[case] is_strict: bool) {
        let backend = pin_budget_cache(is_strict).await;
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry_bytes = TEST_CAPACITY * 3 / 10;
        let value = Arc::new(vec![0u8; entry_bytes]);
        let pin = CachePin::new();
        let lease = CachePin::lease(&pin);
        for _ in 0..4 {
            backend
                .insert_pinned(&key, value.clone(), entry_bytes, &pin)
                .await;
            assert!(backend.peek_resident(&key).await);
            assert!(lease.is_pinned());
            assert_eq!(pin.admissions(), 1);
            let stats = backend.pinned_stats();
            assert_eq!((stats.pinned_entries, stats.leased_entries), (1, 1));
            assert_eq!(
                stats.pinned_bytes,
                (entry_bytes + key_footprint(&key)) as u64
            );
        }
        backend.clear().await;
        assert_eq!(pin.admissions(), 0);
        assert!(!lease.is_pinned());
        assert_eq!(backend.pinned_stats().leased_bytes, 0);
    }

    #[test]
    fn entry_weight_includes_fixed_key() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry = QuickEntry {
            entry: Arc::new(()),
            size_bytes: 7,
            pin: None,
        };
        assert_eq!(
            quick_cache::Weighter::weight(&EntryWeighter, &key, &entry),
            23
        );
    }

    #[test]
    fn capacity_bytes_reports_configured_capacity() {
        assert_eq!(
            QuickCacheBackend::with_capacity(1 << 20).capacity_bytes(),
            Some(1 << 20)
        );
    }

    fn keys_in_shard(
        cache: &QuickCacheBackend,
        shard_index: usize,
        count: usize,
    ) -> Vec<InternalCacheKey> {
        let mut keys = Vec::with_capacity(count);
        for value in 0_u128.. {
            let key = InternalCacheKey::from_bytes(value.to_le_bytes());
            if cache.shard_index(&key) == shard_index {
                keys.push(key);
                if keys.len() == count {
                    return keys;
                }
            }
        }
        unreachable!("the key space must contain enough keys for every shard")
    }

    async fn load_declared_value(
        cache: &QuickCacheBackend,
        key: &InternalCacheKey,
        value: usize,
        size_bytes: usize,
        loads: &AtomicUsize,
    ) -> CacheEntry {
        let (entry, _) = cache
            .get_or_insert(
                key,
                Box::pin(async {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok((Arc::new(value) as CacheEntry, size_bytes))
                }),
                None,
            )
            .await
            .unwrap();
        assert_eq!(*entry.downcast_ref::<usize>().unwrap(), value);
        entry
    }

    #[test]
    fn recommended_shards_cover_large_capacity_boundaries() {
        assert_eq!(recommended_cache_shards_for_parallelism(8 << 30, 4), 2);
        assert_eq!(
            recommended_cache_shards_for_parallelism((8 << 30) - 1, 4),
            1
        );
        assert_eq!(recommended_cache_shards_for_parallelism(16 << 30, 8), 4);
        assert_eq!(
            recommended_cache_shards_for_parallelism((16 << 30) - 1, 8),
            2
        );
    }

    #[tokio::test]
    async fn single_shard_avoids_fragmentation_for_unequal_entries() {
        // Declared entry weights, including the 16-byte key, total 660. They
        // fit the whole cache but not one 500-byte share of a two-shard cache.
        let sizes = [168, 188, 256];
        let sharded = QuickCacheBackend::with_shards(TEST_CAPACITY, 2);
        assert_eq!(sharded.num_shards(), 2);
        let sharded_keys = keys_in_shard(&sharded, 0, sizes.len());
        let sharded_loads = AtomicUsize::new(0);
        for _ in 0..3 {
            for (value, (key, size_bytes)) in sharded_keys.iter().zip(sizes).enumerate() {
                load_declared_value(&sharded, key, value, size_bytes, &sharded_loads).await;
            }
        }
        assert!(sharded_loads.load(Ordering::SeqCst) > sizes.len());
        assert!(sharded.num_entries().await < sizes.len());
        assert!(sharded.size_bytes().await <= TEST_CAPACITY);

        let single =
            QuickCacheBackend::with_shard_policy(TEST_CAPACITY, QuickCacheShardPolicy::Single);
        let single_loads = AtomicUsize::new(0);
        for _ in 0..3 {
            for (value, (key, size_bytes)) in sharded_keys.iter().zip(sizes).enumerate() {
                load_declared_value(&single, key, value, size_bytes, &single_loads).await;
            }
        }
        assert_eq!(single_loads.load(Ordering::SeqCst), sizes.len());
        assert_eq!(single.num_entries().await, sizes.len());
        assert_eq!(single.size_bytes().await, 660);
    }

    #[tokio::test]
    async fn direct_insert_obeys_the_same_shard_budget() {
        let sizes = [168, 188, 256];
        for (shards, expected_entries) in [(2, 2), (1, 3)] {
            let cache = QuickCacheBackend::with_shards(TEST_CAPACITY, shards);
            let keys = keys_in_shard(&cache, 0, sizes.len());
            for (value, (key, size_bytes)) in keys.iter().zip(sizes).enumerate() {
                cache.insert(key, Arc::new(value), size_bytes, None).await;
            }
            assert_eq!(cache.num_entries().await, expected_entries);
            assert!(cache.size_bytes().await <= TEST_CAPACITY);
        }
    }

    #[tokio::test]
    async fn single_shard_preserves_capacity_and_hot_admission_limits() {
        let cache = QuickCacheBackend::with_shards(TEST_CAPACITY, 1);
        let keys = keys_in_shard(&cache, 0, 4);

        // The default hot target is 97% of capacity. Weight includes the key,
        // so a declared size of 954 is admitted at weight 970, while 955 is not.
        cache.insert(&keys[0], Arc::new(0_usize), 954, None).await;
        assert!(cache.get(&keys[0], None).await.is_some());
        cache.clear().await;
        cache.insert(&keys[1], Arc::new(1_usize), 955, None).await;
        assert!(cache.get(&keys[1], None).await.is_none());

        // Individually admissible entries whose working set exceeds the total
        // budget must still reload and remain bounded.
        let loads = AtomicUsize::new(0);
        for _ in 0..2 {
            for (value, key) in keys[1..].iter().enumerate() {
                load_declared_value(&cache, key, value, 384, &loads).await;
            }
        }
        assert!(loads.load(Ordering::SeqCst) > 3);
        assert!(cache.size_bytes().await <= TEST_CAPACITY);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_misses_share_a_load_and_retry_failure() {
        let cache = Arc::new(QuickCacheBackend::with_shards(4096, 1));
        let key = InternalCacheKey::from_bytes([42; 16]);
        let loads = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let owner = {
            let cache = cache.clone();
            let loads = loads.clone();
            let release = release.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert(
                        &key,
                        Box::pin(async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            let _ = started_tx.send(());
                            release.notified().await;
                            Err(crate::Error::timeout("test loader failed"))
                        }),
                        None,
                    )
                    .await
            })
        };
        started_rx.await.unwrap();

        let contender = {
            let cache = cache.clone();
            let loads = loads.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert(
                        &key,
                        Box::pin(async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            Ok((Arc::new(7_usize) as CacheEntry, 8))
                        }),
                        None,
                    )
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        release.notify_one();
        assert!(owner.await.unwrap().is_err());
        let (value, is_hit) = contender.await.unwrap().unwrap();
        assert_eq!(*value.downcast_ref::<usize>().unwrap(), 7);
        assert!(!is_hit);
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_loader_releases_concurrent_miss() {
        let cache = Arc::new(QuickCacheBackend::with_shards(4096, 1));
        let key = InternalCacheKey::from_bytes([24; 16]);
        let loads = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let owner = {
            let cache = cache.clone();
            let loads = loads.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert(
                        &key,
                        Box::pin(async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            let _ = started_tx.send(());
                            std::future::pending::<()>().await;
                            Ok((Arc::new(1_usize) as CacheEntry, 8))
                        }),
                        None,
                    )
                    .await
            })
        };
        started_rx.await.unwrap();

        let contender = {
            let cache = cache.clone();
            let loads = loads.clone();
            tokio::spawn(async move {
                cache
                    .get_or_insert(
                        &key,
                        Box::pin(async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            Ok((Arc::new(2_usize) as CacheEntry, 8))
                        }),
                        None,
                    )
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        let (value, is_hit) = tokio::time::timeout(std::time::Duration::from_secs(1), contender)
            .await
            .expect("contender remained blocked after loader cancellation")
            .unwrap()
            .unwrap();
        assert_eq!(*value.downcast_ref::<usize>().unwrap(), 2);
        assert!(!is_hit);
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_quick_backend_roundtrip_singleflight_and_eviction() {
        // Capacity must be large relative to one entry: quick_cache shards
        // its weight budget, and an entry heavier than its shard's share is
        // not admitted at all.
        const CAPACITY: usize = 1 << 20;
        let item = Arc::new(vec![1u8, 2, 3]);
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(CAPACITY)));

        // insert + get roundtrip and weighted accounting
        cache
            .insert_with_key(&TestKey::<Vec<u8>>::new("a"), item.clone())
            .await;
        assert_eq!(
            cache
                .get_with_key(&TestKey::<Vec<u8>>::new("a"))
                .await
                .as_deref(),
            Some(&vec![1u8, 2, 3])
        );
        assert_eq!(cache.approx_size(), 1);
        assert!(cache.size_bytes().await > 0);

        // get_or_insert runs the loader only on a miss
        let loads = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let loads = loads.clone();
            let value = cache
                .get_or_insert_with_key(TestKey::<Vec<u8>>::new("b"), || async move {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![7u8])
                })
                .await
                .unwrap();
            assert_eq!(value.as_ref(), &vec![7u8]);
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        // capacity is enforced: overfill with 4x capacity of 16KiB entries
        // and confirm eviction kept the weighted size within budget
        for i in 0..256 {
            cache
                .insert_with_key(
                    &TestKey::<Vec<u8>>::new(&format!("fill-{i}")),
                    Arc::new(vec![0u8; 16 << 10]),
                )
                .await;
        }
        assert!(cache.size_bytes().await <= CAPACITY);
        assert!(cache.size().await < 258);

        cache.clear().await;
        assert_eq!(cache.size().await, 0);
    }

    #[tokio::test]
    async fn test_quick_backend_tiny_capacity() {
        // A tiny cache must not over-provision item metadata and must still
        // admit and evict correctly within its weight budget.
        const CAPACITY: usize = 64 << 10;
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(CAPACITY)));
        for i in 0..64 {
            cache
                .insert_with_key(
                    &TestKey::<Vec<u8>>::new(&format!("k-{i}")),
                    Arc::new(vec![0u8; 4 << 10]),
                )
                .await;
        }
        assert!(cache.size_bytes().await <= CAPACITY);
        assert!(cache.size().await >= 1);
        let hit = cache
            .get_with_key(&TestKey::<Vec<u8>>::new("k-63"))
            .await
            .is_some()
            || cache
                .get_with_key(&TestKey::<Vec<u8>>::new("k-62"))
                .await
                .is_some();
        assert!(hit, "recently inserted entries should be resident");
    }
}
