// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! [`CacheBackend`] backed by [quick_cache](https://crates.io/crates/quick_cache).
//! A hit takes a shard read lock, clones the cached value, and marks it as
//! accessed. There is no read-operation channel or inline eviction work.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use async_trait::async_trait;

use super::backend::{CacheBackend, CacheEntry, CacheLoader};
use super::backend_metrics::BackendCounters;
use super::moka::key_footprint;
use super::{
    CacheBackendDiagnostics, CacheBackendKind, CacheCodec, CacheLoadOutcome, CacheOccupancyByType,
    CacheOperationContext, CacheSnapshotMode, CacheTypeOccupancy, InternalCacheKey,
};
use crate::Result;
use crate::deepsize::Context;

#[derive(Clone)]
struct QuickEntry {
    entry: CacheEntry,
    size_bytes: usize,
    submission_id: u64,
    type_name: Option<&'static str>,
}

#[derive(Clone)]
struct EntryWeighter;

impl quick_cache::Weighter<InternalCacheKey, QuickEntry> for EntryWeighter {
    fn weight(&self, key: &InternalCacheKey, value: &QuickEntry) -> u64 {
        // Same accounting as the moka backend.
        key_footprint(key).saturating_add(value.size_bytes).max(1) as u64
    }
}

#[derive(Clone)]
struct EntryLifecycle(Arc<BackendCounters>);

#[derive(Default)]
struct RemovalSummary {
    count: u64,
    bytes: u128,
    first: Option<(InternalCacheKey, u64, u64)>,
}

impl RemovalSummary {
    #[allow(clippy::collapsible_if)]
    fn finish(mut self, metrics: &BackendCounters, submitted: Option<(InternalCacheKey, u64)>) {
        // In quick_cache 0.6.24 insert_existing reports the replaced value
        // first, despite Lifecycle's documentation saying replacements do
        // not notify. Submission identity distinguishes that old value from
        // a rejected candidate, even when both writes reuse the same Arc.
        if let (Some((key, id, bytes)), Some((submitted_key, submitted_id))) =
            (self.first, submitted)
        {
            if key == submitted_key && id != submitted_id {
                self.count -= 1;
                self.bytes -= bytes as u128;
            }
        }
        if self.count > 0 {
            metrics.size_removal(self.count, self.bytes.try_into().unwrap_or(u64::MAX));
        }
    }
}

impl quick_cache::Lifecycle<InternalCacheKey, QuickEntry> for EntryLifecycle {
    type RequestState = RemovalSummary;

    fn begin_request(&self) -> RemovalSummary {
        RemovalSummary::default()
    }

    fn on_evict(&self, state: &mut RemovalSummary, key: InternalCacheKey, value: QuickEntry) {
        let weight = quick_cache::Weighter::weight(&EntryWeighter, &key, &value);
        if state.first.is_none() {
            state.first = Some((key, value.submission_id, weight));
        }
        state.count += 1;
        state.bytes += weight as u128;
    }

    fn end_request(&self, state: RemovalSummary) {
        state.finish(&self.0, None);
    }
}

pub struct QuickCacheBackend {
    cache: quick_cache::sync::Cache<
        InternalCacheKey,
        QuickEntry,
        EntryWeighter,
        quick_cache::DefaultHashBuilder,
        EntryLifecycle,
    >,
    capacity: usize,
    metrics: Arc<BackendCounters>,
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
    /// Admission also depends on each shard's hot-weight target; available
    /// total capacity does not guarantee that a particular entry fits.
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
        let metrics = Arc::new(BackendCounters::new(CacheBackendKind::Quick, true));
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
            EntryLifecycle(metrics.clone()),
        );
        Self {
            cache,
            capacity,
            metrics,
        }
    }

    fn record_write(&self, key: &InternalCacheKey, size_bytes: usize) -> u64 {
        let bytes = key_footprint(key).saturating_add(size_bytes).max(1) as u64;
        let submission_id = self.metrics.write(bytes);
        if self.capacity == 0 {
            self.metrics.disabled_rejection();
        }
        if key_footprint(key).checked_add(size_bytes).is_none() {
            self.metrics
                .weight_saturations
                .fetch_add(1, Ordering::Relaxed);
        }
        submission_id
    }

    #[cfg(test)]
    fn num_shards(&self) -> usize {
        self.cache.num_shards()
    }

    #[cfg(test)]
    fn shard_index(&self, key: &InternalCacheKey) -> usize {
        self.cache.shard_index(key)
    }

    async fn insert_tagged(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        type_name: Option<&'static str>,
    ) {
        let submission_id = self.record_write(key, size_bytes);
        let removals = self.cache.insert_with_lifecycle(
            *key,
            QuickEntry {
                entry,
                size_bytes,
                submission_id,
                type_name,
            },
        );
        removals.finish(&self.metrics, Some((*key, submission_id)));
    }

    async fn get_or_insert_tagged<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        type_name: Option<&'static str>,
    ) -> Result<(CacheEntry, bool)> {
        match self.cache.get_value_or_guard_async(key).await {
            Ok(value) => Ok((value.entry, true)),
            Err(guard) => {
                let (entry, size_bytes) = loader.await?;
                let submission_id = self.record_write(key, size_bytes);
                match guard.insert_with_lifecycle(QuickEntry {
                    entry: entry.clone(),
                    size_bytes,
                    submission_id,
                    type_name,
                }) {
                    Ok(removals) => removals.finish(&self.metrics, Some((*key, submission_id))),
                    Err(_) => {
                        self.metrics.lost_placeholder();
                    }
                }
                Ok((entry, false))
            }
        }
    }

    fn occupancy_by_type(&self) -> CacheOccupancyByType {
        let mut types = BTreeMap::<&'static str, (u64, u64)>::new();
        let mut untagged_size_bytes = 0_u64;
        let mut untagged_num_entries = 0_u64;
        for (key, record) in self.cache.iter() {
            let weight = quick_cache::Weighter::weight(&EntryWeighter, &key, &record);
            if let Some(type_name) = record.type_name {
                let totals = types.entry(type_name).or_default();
                totals.0 = totals.0.saturating_add(weight);
                totals.1 = totals.1.saturating_add(1);
            } else {
                untagged_size_bytes = untagged_size_bytes.saturating_add(weight);
                untagged_num_entries = untagged_num_entries.saturating_add(1);
            }
        }
        CacheOccupancyByType {
            types: types
                .into_iter()
                .map(
                    |(type_name, (size_bytes, num_entries))| CacheTypeOccupancy {
                        type_name: type_name.to_string(),
                        size_bytes,
                        num_entries,
                    },
                )
                .collect(),
            untagged_size_bytes,
            untagged_num_entries,
        }
    }
}

#[async_trait]
impl CacheBackend for QuickCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).map(|v| v.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.insert_tagged(key, entry, size_bytes, None).await;
    }

    async fn insert_with_context(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) {
        self.insert_tagged(key, entry, size_bytes, context.type_name())
            .await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        self.get_or_insert_tagged(key, loader, None).await
    }

    async fn get_or_insert_with_context<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) -> Result<(CacheEntry, bool)> {
        // Keep resident hits on quick_cache's synchronous one-bit hit path.
        // The context is needed only if the following guarded lookup wins the
        // right to insert a loaded value.
        if let Some(value) = self.cache.get(key) {
            return Ok((value.entry, true));
        }
        self.get_or_insert_tagged(key, loader, context.type_name())
            .await
    }

    async fn get_or_insert_with_context_outcome<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) -> Result<(CacheEntry, CacheLoadOutcome)> {
        // Keep resident hits on quick_cache's synchronous one-bit hit path.
        if let Some(value) = self.cache.get(key) {
            return Ok((value.entry, CacheLoadOutcome::ResidentHit));
        }
        self.get_or_insert_tagged(key, loader, context.type_name())
            .await
            .map(|(entry, was_cached)| {
                let outcome = if was_cached {
                    // quick_cache does not reveal whether this value was found
                    // immediately or received after waiting on a placeholder.
                    CacheLoadOutcome::LoaderSkippedUnknown
                } else {
                    CacheLoadOutcome::Loaded
                };
                (entry, outcome)
            })
    }

    async fn clear(&self) {
        self.cache.clear();
    }

    async fn num_entries(&self) -> usize {
        self.cache.len()
    }

    async fn size_bytes(&self) -> usize {
        self.cache.weight() as usize
    }

    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity)
    }

    fn diagnostics(&self) -> CacheBackendDiagnostics {
        self.metrics.snapshot(
            CacheBackendKind::Quick,
            self.capacity,
            self.approx_size_bytes(),
            self.approx_num_entries(),
        )
    }

    async fn diagnostics_with_types(
        &self,
        _mode: CacheSnapshotMode,
    ) -> (CacheBackendDiagnostics, Option<CacheOccupancyByType>) {
        (self.diagnostics(), Some(self.occupancy_by_type()))
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.len()
    }

    fn approx_size_bytes(&self) -> usize {
        self.cache.weight() as usize
    }

    fn deep_size_of_entries(
        &self,
        context: &mut Context,
        size_of_entry: &dyn Fn(&CacheEntry, &mut Context) -> Option<usize>,
    ) -> Option<usize> {
        Some(
            self.cache
                .iter()
                .map(|(key, record)| {
                    key_footprint(&key)
                        + size_of_entry(&record.entry, context).unwrap_or(record.size_bytes)
                })
                .sum(),
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

    #[test]
    fn entry_weight_includes_fixed_key() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry = QuickEntry {
            entry: Arc::new(()),
            size_bytes: 7,
            submission_id: 0,
            type_name: None,
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

    #[tokio::test]
    async fn diagnostics_report_superseded_loader_placeholder() {
        let backend = QuickCacheBackend::with_capacity(4096);
        let key = InternalCacheKey::from_bytes([0; 16]);
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let mut load = Box::pin(backend.get_or_insert(
            &key,
            Box::pin(async move {
                release_rx.await.unwrap();
                Ok((Arc::new(1_u64) as CacheEntry, 8))
            }),
            None,
        ));
        assert!(futures::poll!(load.as_mut()).is_pending());
        backend.insert(&key, Arc::new(2_u64), 8, None).await;
        release_tx.send(()).unwrap();
        let (result, was_cached) = load.await.unwrap();
        assert!(!was_cached);
        assert_eq!(*result.downcast_ref::<u64>().unwrap(), 1);
        assert_eq!(
            *backend
                .get(&key, None)
                .await
                .unwrap()
                .downcast_ref::<u64>()
                .unwrap(),
            2
        );
        let snapshot = backend.diagnostics();
        assert_eq!(snapshot.write_attempts, Some(2));
        assert_eq!(snapshot.lost_placeholder_rejections, Some(1));
        assert_eq!(snapshot.size_removals, Some(0));
    }

    #[cfg(target_pointer_width = "64")]
    #[tokio::test]
    async fn diagnostics_report_saturated_declared_weight() {
        let backend = QuickCacheBackend::with_capacity(4096);
        let key = InternalCacheKey::from_bytes([0; 16]);
        backend.insert(&key, Arc::new(()), usize::MAX, None).await;
        let snapshot = backend.diagnostics();
        assert_eq!(snapshot.weight_saturations, Some(1));
        assert_eq!(snapshot.write_bytes, Some(u64::MAX));
        assert_eq!(snapshot.size_removed_bytes, Some(u64::MAX));
        backend.insert(&key, Arc::new(()), 8, None).await;
        assert_eq!(backend.diagnostics().write_bytes, Some(u64::MAX));
    }
}
