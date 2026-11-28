// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance::index::vector::ivf::v2::rabit_scan::{
    RabitCacheLayout, RabitScan, RabitScanContext, RabitScanFactory,
};
use lance::session::{CacheSpec, OriginLatencyClass, ResidentColumnsSetting, Session};
use lance_core::cache::{
    CacheBackend, CacheCodec, CacheKey, CacheSnapshotMode, LanceCache, QuickCacheBackend,
};
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{Dimension, RowCount, Seed, array, gen_batch};
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::storage::{IndexFileKey, ResidentColumnsKey};
use lance_index::{INDEX_AUXILIARY_FILE_NAME, IndexType};
use lance_linalg::distance::DistanceType;

const RABIT_INDEX: &str = "vector_idx";

struct UnrelatedEntry(u8, bool);

impl CacheKey for UnrelatedEntry {
    type ValueType = Vec<u8>;

    fn type_name() -> &'static str {
        "UnrelatedEntry"
    }

    fn key(&self) -> Cow<'_, str> {
        self.0.to_string().into()
    }

    fn codec_for_key(&self) -> Option<CacheCodec> {
        self.1.then(|| {
            CacheCodec::new(
                "test.unrelated",
                1,
                |_, _| Ok(()),
                |_| Ok(Arc::new(Vec::<u8>::new())),
            )
            .with_memory_priority(3)
        })
    }
}

#[rstest::rstest]
#[case::clock(false)]
#[case::priority(true)]
#[tokio::test]
async fn cold_index_discovery_preserves_unrelated_entries(#[case] priority: bool) {
    let dir = TempStrDir::default();
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..64).map(|row| Some([Some(row as f32), Some((row % 7) as f32)])),
        2,
    );
    let batch = RecordBatch::try_from_iter([("vector", Arc::new(vectors) as _)]).unwrap();
    let batches = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(batches, dir.as_str(), None).await.unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(DistanceType::L2, IvfBuildParams::new(4));
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_owned()),
            &params,
            true,
        )
        .await
        .unwrap();

    let payload = Arc::new(vec![7u8; 8192]);
    let sizing_cache =
        LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(1 << 20)));
    sizing_cache
        .insert_with_key(&UnrelatedEntry(0, priority), payload.clone())
        .await;
    let entry_weight = sizing_cache.size_bytes().await;

    let backend = Arc::new(QuickCacheBackend::with_capacity(2 * entry_weight));
    let dataset = DatasetBuilder::from_uri(dir.as_str())
        .with_index_cache_backend(backend.clone())
        .load()
        .await
        .unwrap();
    // Force a cache miss when the rebuilt index is first listed.
    backend.clear().await;
    let unrelated = LanceCache::with_backend(backend.clone()).with_key_prefix("other-table");
    for id in [0, 1] {
        unrelated
            .insert_with_key(&UnrelatedEntry(id, priority), payload.clone())
            .await;
    }
    assert_eq!(backend.size_bytes().await, 2 * entry_weight);

    let error = dataset
        .prewarm_index_with_cache_budget("vector_idx")
        .await
        .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains("prewarm"));
    for id in [0, 1] {
        assert_eq!(
            unrelated
                .get_with_key(&UnrelatedEntry(id, priority))
                .await
                .as_deref(),
            Some(payload.as_ref()),
            "capacity-limited prewarm evicted unrelated entry {id}",
        );
    }
}

#[derive(Debug)]
struct PlaneScan;

impl RabitScanFactory for PlaneScan {
    fn create(&self, _: &IndexFileKey, _: OriginLatencyClass) -> lance::Result<Arc<dyn RabitScan>> {
        Ok(Arc::new(Self))
    }
}

#[async_trait::async_trait]
impl RabitScan for PlaneScan {
    fn cache_layout(&self) -> RabitCacheLayout {
        RabitCacheLayout::Planes
    }

    async fn search(&self, _: Arc<RabitScanContext>) -> lance::Result<Option<RecordBatch>> {
        Ok(None)
    }
}

fn rabit_session(backend: Arc<QuickCacheBackend>, layout: RabitCacheLayout) -> Arc<Session> {
    let session = Session::with_cache_backends(
        CacheSpec::Backend(backend),
        CacheSpec::Default,
        Default::default(),
    )
    .with_index_resident_columns(ResidentColumnsSetting::On);
    Arc::new(match layout {
        RabitCacheLayout::Planes => session.with_rabit_scan_factory(Arc::new(PlaneScan)),
        RabitCacheLayout::Partition => session,
    })
}

async fn open_at(
    uri: &str,
    version: Option<u64>,
    session: Arc<Session>,
    cache: &LanceCache,
) -> Dataset {
    let builder = DatasetBuilder::from_uri(uri).with_session(session);
    let builder = match version {
        Some(version) => builder.with_version(version),
        None => builder,
    };
    // Isolate scoped prewarm from ordinary cache admissions during dataset open.
    cache.without_admission(builder.load()).await.unwrap()
}

async fn resident_file(dataset: &Dataset) -> IndexFileKey {
    let indices = dataset.load_indices().await.unwrap();
    let [segment] = indices.as_slice() else {
        panic!("expected a single RaBitQ index segment, got {indices:?}");
    };
    let uuid = segment.uuid.to_string();
    let store = dataset.object_store(None).await.unwrap();
    let path = dataset
        .indices_dir()
        .join(uuid.as_str())
        .join(INDEX_AUXILIARY_FILE_NAME);
    IndexFileKey::new(&uuid, &store.store_prefix, path.as_ref())
}

async fn cache_weight(uri: &str, layout: RabitCacheLayout, entry_type: &str) -> (usize, usize) {
    let backend = Arc::new(QuickCacheBackend::with_capacity(1 << 25));
    let cache = LanceCache::with_backend(backend.clone());
    let dataset = open_at(uri, None, rabit_session(backend.clone(), layout), &cache).await;
    dataset
        .prewarm_index_with_cache_budget(RABIT_INDEX)
        .await
        .unwrap();
    let (stats, by_type) = backend
        .diagnostics_with_types(CacheSnapshotMode::Refreshed)
        .await;
    let by_type = by_type.unwrap();
    assert!(
        by_type.types.iter().any(|row| {
            row.type_name == ResidentColumnsKey::type_name() && row.num_entries == 1
        })
    );
    let raw_bytes = by_type
        .types
        .iter()
        .find(|row| row.type_name == entry_type)
        .unwrap()
        .size_bytes;
    (stats.size_bytes.unwrap() as usize, raw_bytes as usize)
}

async fn raw_loads(session: &Session, entry_type: &str) -> u64 {
    session
        .cache_diagnostics_by_type(CacheSnapshotMode::Refreshed)
        .await
        .index
        .by_type
        .unwrap()
        .activity
        .into_iter()
        .find(|row| row.type_name == entry_type)
        .unwrap()
        .activity
        .warm
        .loads_started
}

#[rstest::rstest]
#[case::planes(RabitCacheLayout::Planes, "RabitPlane")]
#[case::codes(RabitCacheLayout::Partition, "IVFPartitionCodes")]
#[tokio::test]
async fn rabit_rebuild_protects_shared_cache_and_reclaims_exclusive_cache(
    #[case] layout: RabitCacheLayout,
    #[case] entry_type: &str,
) {
    let dir = TempStrDir::default();
    let clone_dir = TempStrDir::default();
    let batch = gen_batch()
        .with_seed(Seed::from(42))
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(64)),
        )
        .into_batch_rows(RowCount::from(4096))
        .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut writer = Dataset::write(reader, dir.as_str(), None).await.unwrap();
    let vectors = batch["vector"].as_fixed_size_list();
    let params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(4, Arc::new(vectors.slice(0, 4))).unwrap(),
        RQBuildParams::new(7),
    );
    let backend = Arc::new(QuickCacheBackend::with_capacity(1 << 25));
    let file_cache = LanceCache::with_backend(backend.clone()).with_key_prefix("ivf-file");
    let source_session = rabit_session(backend.clone(), layout);
    let mut prior_file = None;
    let mut generations = Vec::new();
    let mut capacity = 0;
    for rebuild in 0..3 {
        writer
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(RABIT_INDEX.to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();
        let version = writer.version_id();
        if rebuild == 0 {
            writer
                .shallow_clone(clone_dir.as_str(), version, None)
                .await
                .unwrap();
        }
        let (size, raw_size) = cache_weight(dir.as_str(), layout, entry_type).await;
        // Grow only to the largest one-generation footprint measured so far.
        // No cache entries are removed by resizing between consecutive rebuilds.
        capacity = capacity.max(size);
        backend.set_capacity(capacity);
        let head = open_at(dir.as_str(), None, source_session.clone(), &file_cache).await;
        head.prewarm_index_with_cache_budget(RABIT_INDEX)
            .await
            .unwrap();
        let file = resident_file(&writer).await;
        assert!(
            file_cache
                .peek_resident_with_key(&ResidentColumnsKey::new(&file))
                .await
        );
        if let Some(prior_file) = prior_file.replace(file) {
            assert!(
                !file_cache
                    .peek_resident_with_key(&ResidentColumnsKey::new(&prior_file))
                    .await
            );
        }
        generations.push((version, size, raw_size));
    }

    let (old_version, old_size, _) = generations[0];
    let (_, new_size, new_raw) = generations[2];
    // Account for the clone's own index state while keeping pressure on the
    // physical entries both tables use.
    let backend = Arc::new(QuickCacheBackend::with_capacity(
        old_size + new_size - new_raw / 2,
    ));
    let file_cache = LanceCache::with_backend(backend.clone()).with_key_prefix("ivf-file");
    let source_session = rabit_session(backend.clone(), layout);
    let clone_session = rabit_session(backend, layout);
    let old = open_at(
        dir.as_str(),
        Some(old_version),
        source_session.clone(),
        &file_cache,
    )
    .await;
    let new = open_at(dir.as_str(), None, source_session, &file_cache).await;
    let clone = open_at(clone_dir.as_str(), None, clone_session.clone(), &file_cache).await;
    let old_file = file_cache.without_admission(resident_file(&old)).await;
    let old_columns = ResidentColumnsKey::new(&old_file);
    old.prewarm_index_with_cache_budget(RABIT_INDEX)
        .await
        .unwrap();
    clone
        .prewarm_index_with_cache_budget(RABIT_INDEX)
        .await
        .unwrap();
    assert!(file_cache.peek_resident_with_key(&old_columns).await);
    let err = new
        .prewarm_index_with_cache_budget(RABIT_INDEX)
        .await
        .unwrap_err();
    assert!(matches!(err, lance::Error::InvalidInput { .. }), "{err}");
    assert!(err.to_string().contains("cache capacity"));
    assert!(file_cache.peek_resident_with_key(&old_columns).await);
    let cached_loads = raw_loads(&clone_session, entry_type).await;
    clone
        .prewarm_index_with_cache_budget(RABIT_INDEX)
        .await
        .unwrap();
    assert_eq!(raw_loads(&clone_session, entry_type).await, cached_loads);
}
