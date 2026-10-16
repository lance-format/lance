// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use bytes::Bytes;
use lance_core::cache::{CacheCodec, CacheEntry, CacheTier, InternalCacheKey, QuickCacheBackend};
use lance_index::vector::ApproxMode;
use lance_index::vector::bq::layered::{PlaneBatch, RQPrecision};
use lance_io::object_store::ObjectStoreRegistry;

use crate::session::Session;

const PARTITIONS: usize = 8;
const PLANES: usize = 3;
const BITS: u8 = 7;
const K: usize = 10;
/// Holds every entry, so the planes' total size can be measured.
const LARGE_RAM_BYTES: usize = 256 * 1024 * 1024;
const METADATA_CACHE_BYTES: usize = 64 * 1024 * 1024;
/// The small RAM tier holds 1 / this of the plane bytes.
const SMALL_RAM_DIVISOR: usize = 10;

type DiskEntries = HashMap<InternalCacheKey, (Bytes, CacheCodec, usize)>;

/// RAM tier plus an unbounded serialized "disk" tier, like a tiered
/// cache: loads are written through and disk hits are admitted to
/// RAM. `gated` is what the backend reports for plane admission.
struct PlaneTierTestBackend {
    ram: QuickCacheBackend,
    disk: std::sync::Mutex<DiskEntries>,
    gated: bool,
}

impl std::fmt::Debug for PlaneTierTestBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneTierTestBackend")
            .field("gated", &self.gated)
            .finish_non_exhaustive()
    }
}

impl PlaneTierTestBackend {
    fn new(ram_bytes: usize, gated: bool) -> Self {
        Self {
            ram: QuickCacheBackend::with_capacity(ram_bytes),
            disk: Default::default(),
            gated,
        }
    }

    fn read_disk(&self, key: &InternalCacheKey) -> Option<(CacheEntry, usize)> {
        let (bytes, codec, size) = self.disk.lock().unwrap().get(key).cloned()?;
        codec.deserialize(&bytes).hit().map(|entry| (entry, size))
    }

    /// Persisted plane entries and their accounted bytes.
    fn persisted_planes(&self) -> (usize, usize) {
        self.disk
            .lock()
            .unwrap()
            .values()
            .filter(|(_, codec, _)| codec.type_id() == <PlaneBatch as CacheCodecImpl>::TYPE_ID)
            .fold((0, 0), |(entries, bytes), (_, _, size)| {
                (entries + 1, bytes + size)
            })
    }
}

#[async_trait::async_trait]
impl CacheBackend for PlaneTierTestBackend {
    async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        self.ram.get_resident(key).await
    }

    /// RAM as `peek_resident` reports it, then the serialized tier.
    async fn peek_tier(&self, key: &InternalCacheKey) -> CacheTier {
        if self.peek_resident(key).await {
            CacheTier::Resident
        } else if self.disk.lock().unwrap().contains_key(key) {
            CacheTier::Local
        } else {
            CacheTier::Absent
        }
    }

    fn plane_admission_gated(&self) -> bool {
        self.gated
    }

    async fn get_without_promotion(
        &self,
        key: &InternalCacheKey,
        _codec: Option<CacheCodec>,
    ) -> Option<CacheEntry> {
        match self.ram.get_resident(key).await {
            Some(entry) => Some(entry),
            None => self.read_disk(key).map(|(entry, _)| entry),
        }
    }

    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        if let Some(entry) = self.ram.get(key, None).await {
            return Some(entry);
        }
        let (entry, size) = self.read_disk(key)?;
        self.ram.insert(key, entry.clone(), size, None).await;
        Some(entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    ) {
        if let Some(codec) = codec {
            let mut bytes = Vec::new();
            codec.serialize(&entry, &mut bytes).unwrap();
            self.disk
                .lock()
                .unwrap()
                .insert(*key, (Bytes::from(bytes), codec, size_bytes));
        }
        self.ram.insert(key, entry, size_bytes, None).await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: std::pin::Pin<
            Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
        >,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        if let Some(entry) = self.get(key, codec).await {
            return Ok((entry, true));
        }
        let (entry, size) = loader.await?;
        self.insert(key, entry.clone(), size, codec).await;
        Ok((entry, false))
    }

    async fn clear(&self) {
        self.ram.clear().await;
        self.disk.lock().unwrap().clear();
    }

    async fn num_entries(&self) -> usize {
        self.ram.num_entries().await
    }

    async fn size_bytes(&self) -> usize {
        self.ram.size_bytes().await
    }
}

async fn open_prewarmed(
    uri: &str,
    backend: Arc<PlaneTierTestBackend>,
) -> (Dataset, Arc<dyn VectorIndex>) {
    let session = Session::with_index_cache_backend(
        backend,
        METADATA_CACHE_BYTES,
        Arc::new(ObjectStoreRegistry::default()),
    );
    let dataset = crate::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    index.prewarm().await.unwrap();
    (dataset, index)
}

fn full_query(key: ArrayRef) -> Query {
    Query {
        search_effort: 0.5,
        column: "vector".to_string(),
        key,
        k: K,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: PARTITIONS,
        maximum_nprobes: Some(PARTITIONS),
        ef: None,
        refine_factor: None,
        metric_type: None,
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: ApproxMode::Normal,
        rq_precision: RQPrecision::Full,
        rq_cascade_factor: None,
    }
}

/// Row ids and distance bits of every probed partition, in result order.
async fn search_all_partitions(
    index: &Arc<dyn VectorIndex>,
    query: &Query,
) -> (Vec<u64>, Vec<u32>) {
    let (partitions, dists) = index.find_partitions(query).unwrap();
    let probes = partitions.len();
    let batches = index
        .clone()
        .search_partitions(
            query.clone(),
            Arc::new(partitions),
            Arc::new(dists),
            0,
            probes,
            Arc::new(NoFilter),
            None,
            Arc::new(NoOpMetricsCollector),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut row_ids = Vec::new();
    let mut distance_bits = Vec::new();
    for batch in &batches {
        row_ids.extend(batch[ROW_ID].as_primitive::<UInt64Type>().values());
        distance_bits.extend(
            batch[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|dist| dist.to_bits()),
        );
    }
    (row_ids, distance_bits)
}

/// A backend that admits planes like any entry is warmed partition by
/// partition, so every plane is persisted although RAM holds only a
/// fraction of them, and warm full-precision queries read nothing from
/// the index file. A sign-gated backend of the same size persists only
/// the lower planes of partitions whose sign plane stayed resident.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_ungated_prewarm_persists_every_plane() {
    let dir = TempStrDir::default();
    let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(dir.as_str(), 0.0..1.0).await;
    let params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::new(PARTITIONS),
        RQBuildParams::new(BITS).with_layered(true),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    let query = full_query(vectors.value(0));

    let resident = Arc::new(PlaneTierTestBackend::new(LARGE_RAM_BYTES, false));
    let (_, index) = open_prewarmed(dir.as_str(), resident.clone()).await;
    let (entries, plane_bytes) = resident.persisted_planes();
    assert_eq!(entries, PLANES * PARTITIONS);
    let expected = search_all_partitions(&index, &query).await;
    assert_eq!(expected.0.len(), K);

    let small_ram = plane_bytes / SMALL_RAM_DIVISOR;
    let ungated = Arc::new(PlaneTierTestBackend::new(small_ram, false));
    let (dataset, index) = open_prewarmed(dir.as_str(), ungated.clone()).await;
    assert_eq!(ungated.persisted_planes(), (entries, plane_bytes));
    assert!(ungated.ram.size_bytes().await <= small_ram);
    assert_eq!(search_all_partitions(&index, &query).await, expected);
    dataset.object_store.as_ref().io_stats_incremental();
    for _ in 0..2 {
        assert_eq!(search_all_partitions(&index, &query).await, expected);
    }
    let io = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io, read_iops, 0, "warm layered queries read no plane");

    let gated = Arc::new(PlaneTierTestBackend::new(small_ram, true));
    let (_, index) = open_prewarmed(dir.as_str(), gated.clone()).await;
    let (gated_entries, _) = gated.persisted_planes();
    assert!(
        (PARTITIONS..PLANES * PARTITIONS).contains(&gated_entries),
        "gated prewarm persisted {gated_entries} plane entries"
    );
    assert_eq!(search_all_partitions(&index, &query).await, expected);
}
