// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::IVFIndex;
use super::native_storage_tests::native_fixture;
use super::*;
use async_trait::async_trait;
use lance_core::cache::{
    CacheCodec, CacheEntry, CacheKey, CacheLoader, CacheNamespace, InternalCacheKey, KeyBuilder,
    MokaCacheBackend,
};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::planes::PlaneKey;
use std::sync::Mutex;
use tokio::sync::Semaphore;

#[derive(Debug)]
struct PausedPlaneCache {
    inner: MokaCacheBackend,
    gated: bool,
    permits: Semaphore,
    starts: Mutex<Vec<(InternalCacheKey, u8)>>,
    rejected: Option<InternalCacheKey>,
    active: AtomicUsize,
    fail_next: AtomicBool,
}

struct ActiveLoad<'a>(&'a AtomicUsize);
impl Drop for ActiveLoad<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[async_trait]
impl CacheBackend for PausedPlaneCache {
    fn plane_admission_gated(&self) -> bool {
        self.gated
    }
    async fn peek_resident(&self, key: &InternalCacheKey) -> bool {
        self.inner.peek_resident(key).await
    }
    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.inner.get(key, codec).await
    }
    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size: usize,
        codec: Option<CacheCodec>,
    ) {
        self.inner.insert(key, entry, size, codec).await;
    }
    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let plane = codec.as_ref().unwrap().plane_tag().unwrap();
        let load = Box::pin(async {
            self.starts.lock().unwrap().push((*key, plane));
            self.active.fetch_add(1, Ordering::Relaxed);
            let _active = ActiveLoad(&self.active);
            self.permits.acquire().await.unwrap().forget();
            if self.fail_next.swap(false, Ordering::Relaxed) {
                return Err(lance_core::Error::io("injected plane read failure"));
            }
            loader.await
        });
        if self.rejected == Some(*key) {
            return Ok((load.await?.0, false));
        }
        self.inner.get_or_insert(key, load, codec).await
    }
    async fn clear(&self) {
        self.inner.clear().await;
    }
    async fn num_entries(&self) -> usize {
        self.inner.num_entries().await
    }
    async fn size_bytes(&self) -> usize {
        self.inner.size_bytes().await
    }
}

#[rstest]
#[case::ordinary(false)]
#[case::sign_gated(true)]
#[tokio::test]
async fn native_plane_prewarm_overlaps_partitions(#[case] gated: bool) {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap();
    let backend = Arc::new(PausedPlaneCache {
        inner: MokaCacheBackend::with_capacity(1024 * 1024),
        gated,
        permits: Semaphore::new(0),
        starts: Mutex::new(Vec::new()),
        rejected: None,
        active: AtomicUsize::new(0),
        fail_next: AtomicBool::new(false),
    });
    let cache = LanceCache::with_backend(backend.clone());
    let weak = WeakLanceCache::from(&cache);
    let warm = index.storage.prewarm_planes(&weak);
    tokio::pin!(warm);
    assert!(futures::poll!(&mut warm).is_pending());
    let starts = backend.starts.lock().unwrap().clone();
    let parallelism = lance_core::utils::tokio::get_num_compute_intensive_cpus().min(4);
    assert_eq!(
        starts.len(),
        parallelism,
        "independent partitions must overlap before any read completes"
    );
    assert!(starts.iter().all(|(_, plane)| *plane == 0));
    backend.permits.add_permits(16);
    let (first, second) = tokio::join!(warm, index.storage.prewarm_planes(&weak));
    first.unwrap();
    second.unwrap();
    let starts = backend.starts.lock().unwrap().clone();
    assert_eq!(starts.len(), 8);
    assert_eq!(
        starts
            .iter()
            .map(|(key, _)| *key)
            .collect::<HashSet<_>>()
            .len(),
        8
    );
    if gated {
        assert!(starts[..4].iter().all(|(_, plane)| *plane == 0));
    }
    // Both concurrent and later prewarms share existing immutable keys/loaders.
    let (first, second) = tokio::join!(
        index.storage.prewarm_planes(&weak),
        index.storage.prewarm_planes(&weak)
    );
    first.unwrap();
    second.unwrap();
    assert_eq!(backend.starts.lock().unwrap().len(), 8);
}

#[rstest]
#[case::ordinary(false, 8)]
#[case::gated(true, 7)]
#[tokio::test]
async fn native_plane_prewarm_preserves_rejected_sign_policy(
    #[case] gated: bool,
    #[case] loads: usize,
) {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap();
    let mut key = KeyBuilder::new(
        CacheNamespace::root(),
        PlaneKey::stable_type_id(),
        PlaneKey::schema(),
    );
    index.storage.plane_key(0, 0).write_key(&mut key);
    let backend = Arc::new(PausedPlaneCache {
        inner: MokaCacheBackend::with_capacity(1024 * 1024),
        gated,
        permits: Semaphore::new(16),
        starts: Mutex::new(Vec::new()),
        rejected: Some(key.finish()),
        active: AtomicUsize::new(0),
        fail_next: AtomicBool::new(false),
    });
    let cache = LanceCache::with_backend(backend.clone());
    let weak = WeakLanceCache::from(&cache);
    index.storage.prewarm_planes(&weak).await.unwrap();
    assert_eq!(backend.starts.lock().unwrap().len(), loads);
    assert!(
        !weak
            .peek_resident_with_key(&index.storage.plane_key(0, 0))
            .await
    );
    assert_eq!(
        weak.peek_resident_with_key(&index.storage.plane_key(0, 1))
            .await,
        !gated
    );
    for part in 1..4 {
        for plane in [0, 1] {
            assert!(
                weak.peek_resident_with_key(&index.storage.plane_key(part, plane))
                    .await
            );
        }
    }
}

#[rstest]
#[case::cancel(false)]
#[case::read_error(true)]
#[tokio::test]
async fn native_plane_prewarm_releases_loaders_and_can_retry(#[case] fail: bool) {
    let dir = TempStrDir::default();
    let (dataset, _) = native_fixture(dir.as_str()).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let index = index
        .as_any()
        .downcast_ref::<IVFIndex<FlatIndex, RabitQuantizer>>()
        .unwrap();
    let backend = Arc::new(PausedPlaneCache {
        inner: MokaCacheBackend::with_capacity(1024 * 1024),
        gated: true,
        permits: Semaphore::new(0),
        starts: Mutex::new(Vec::new()),
        rejected: None,
        active: AtomicUsize::new(0),
        fail_next: AtomicBool::new(fail),
    });
    let cache = LanceCache::with_backend(backend.clone());
    let weak = WeakLanceCache::from(&cache);
    let mut warm = Box::pin(index.storage.prewarm_planes(&weak));
    assert!(futures::poll!(&mut warm).is_pending());
    assert!(backend.active.load(Ordering::Relaxed) > 0);
    if fail {
        backend.permits.add_permits(16);
        let error = warm.await.unwrap_err();
        assert!(matches!(error.find_root(), lance_core::Error::IO { .. }));
        assert!(error.to_string().contains("injected plane read failure"));
    } else {
        drop(warm);
        backend.permits.add_permits(16);
    }
    assert_eq!(backend.active.load(Ordering::Relaxed), 0);
    tokio::time::timeout(
        std::time::Duration::from_millis(500),
        index.storage.prewarm_planes(&weak),
    )
    .await
    .expect("cancelled cache loaders must not block a retry")
    .unwrap();
    assert_eq!(backend.active.load(Ordering::Relaxed), 0);
    for part in 0..4 {
        for plane in [0, 1] {
            assert!(
                weak.peek_resident_with_key(&index.storage.plane_key(part, plane))
                    .await
            );
        }
    }
}
