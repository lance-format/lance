// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! [`CacheBackend`] backed by [quick_cache](https://crates.io/crates/quick_cache).
//! A hit takes a shard read lock, clones the cached value, and marks it as
//! accessed. There is no read-operation channel or inline eviction work.

use super::PriorityEntries;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use async_trait::async_trait;
use futures::Future;

use super::backend::{CacheBackend, CacheEntry};
use super::moka::key_footprint;
use super::{CacheCodec, InternalCacheKey};
use crate::Result;
use crate::deepsize::Context;

#[derive(Clone)]
struct QuickEntry {
    entry: CacheEntry,
    size_bytes: usize,
}

#[derive(Clone)]
struct EntryWeighter;

impl quick_cache::Weighter<InternalCacheKey, QuickEntry> for EntryWeighter {
    fn weight(&self, key: &InternalCacheKey, value: &QuickEntry) -> u64 {
        // Same accounting as the moka backend.
        key_footprint(key).saturating_add(value.size_bytes).max(1) as u64
    }
}

pub struct QuickCacheBackend {
    capacity: usize,
    generation: AtomicU64,
    priority_active: AtomicBool,
    priority: Mutex<PriorityEntries<QuickEntry>>,
    cache: quick_cache::sync::Cache<InternalCacheKey, QuickEntry, EntryWeighter>,
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
        let cache = quick_cache::sync::Cache::with_options(
            options,
            EntryWeighter,
            Default::default(),
            Default::default(),
        );
        Self {
            cache,
            capacity,
            generation: AtomicU64::new(0),
            priority_active: AtomicBool::new(false),
            priority: Mutex::new(PriorityEntries::default()),
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
                    dropped.extend(entries.insert(key, value, size, 3, self.capacity));
                }
                // Evict resident values without invalidating single-flight
                // placeholders: their loaders still belong to this generation.
                self.cache.set_capacity(0);
            }
            let size = key_footprint(&key).saturating_add(item.size_bytes);
            dropped.extend(entries.insert(
                key,
                item,
                size,
                if priority == 0 { 3 } else { priority },
                self.capacity,
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
        let item = QuickEntry { entry, size_bytes };
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

    fn max_entry_bytes(&self) -> Option<u64> {
        Some((self.capacity / self.cache.num_shards()) as u64)
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
    use crate::cache::{CacheKey, CacheTier, LanceCache};

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

    #[test]
    fn entry_weight_includes_fixed_key() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry = QuickEntry {
            entry: Arc::new(()),
            size_bytes: 7,
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
        for capacity in [0, 1 << 20, 16 << 30] {
            let backend = QuickCacheBackend::with_capacity(capacity);
            assert_eq!(
                backend.max_entry_bytes(),
                Some((capacity / backend.cache.num_shards()) as u64)
            );
        }
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
