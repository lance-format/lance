// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! [`CacheBackend`] backed by [quick_cache](https://crates.io/crates/quick_cache),
//! whose hit path is one atomic bit — no read-op channel or inline
//! housekeeping. Used for the session index and metadata caches; the index
//! cache sees thousands of cache reads per query.

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
    let by_cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        / 2;
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
        let shards = recommended_cache_shards(capacity);
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
                self.cache.clear();
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
    use crate::cache::{CacheKey, LanceCache};

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
    async fn priority_budget_retains_metadata_and_singleflight_entries() {
        let cache = QuickCacheBackend::with_capacity(160);
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
        cache.insert(&key(0), Arc::new(0u64), 32, None).await;
        assert!(cache.get_resident(&key(0)).await.is_some());
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
