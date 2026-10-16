// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::vector::ivf::v2::rabit_scan::RabitCacheLayout;
use crate::io::exec::utils::IndexMetrics;
use async_trait::async_trait;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use lance_core::Error;
use lance_index::metrics::MetricsCollector;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use arrow::compute::concat_batches;
use arrow_array::UInt32Array;
use bytes::Bytes;
use lance_arrow::RecordBatchExt;
use lance_core::cache::{CacheCodec, CacheEntry, CacheTier, InternalCacheKey, QuickCacheBackend};
use lance_encoding::decoder::FilterExpression;
use lance_file::version::LanceFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_index::metrics::LocalMetricsCollector;
use lance_index::vector::bq::layered::{
    EntryColumns, FULL_BOUNDS_COLUMN, PlaneBatch, RQPrecision, SIGN_BOUNDS_PLANE, SignBounds,
    plane_columns, plane_entry_columns,
};
use lance_index::vector::bq::partition_codes::{PartitionCodes, PartitionCodesKey};
use lance_index::vector::bq::scan_stats;
use lance_index::vector::bq::storage::RABIT_CODE_COLUMN;
use lance_index::vector::storage::{
    GatherPlan, IndexFileKey, OriginLatencyClass, PlaneSource, ResidentAttach, ResidentColumns,
    ResidentColumnsEntry, ResidentColumnsKey, ResidentColumnsSetting, ResidentLifetime,
    ResidentStoreSize, ResidentStoreViews, entry_columns_setting, origin_latency_setting,
    resident_columns_setting, resident_lifetime_setting, resident_store_is_live,
    resident_store_views, shares_resident_store, sign_bounds_setting,
};
use lance_index::vector::{ApproxMode, VECTOR_RESULT_SCHEMA};
use lance_io::ReadBatchParams;
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::scheduler::IoStats;
use rand::seq::SliceRandom;

use crate::session::Session;

type IvfRq = super::super::IVFIndex<FlatIndex, RabitQuantizer>;

const LAZY_DIM: usize = 128;
const LAZY_PARTITIONS: usize = 64;
const LAZY_ROWS: usize = 20_000;
/// `(partition, rows)` of partitions that are empty or smaller than
/// one 32-row FastScan block, or straddle its boundary.
const LAZY_SMALL_PARTITIONS: [(usize, usize); 8] = [
    (3, 0),
    (11, 0),
    (7, 1),
    (13, 5),
    (19, 17),
    (23, 31),
    (29, 32),
    (31, 33),
];
/// About a tenth of the index's plane bytes, so plane entries churn.
const LAZY_SMALL_CACHE_BYTES: usize = 512 * 1024;
const LAZY_LARGE_CACHE_BYTES: usize = 256 * 1024 * 1024;
const LAZY_METADATA_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// These tests assert deltas of process-wide counters, so they run one at a time.
static LAZY_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(Default::default);

/// RAM tier plus a serialized "disk" tier, like a tiered cache: disk
/// hits are admitted to RAM, loads are written through, and selected
/// rows are read from the serialized entry.
struct TieredPlaneTestBackend {
    ram: QuickCacheBackend,
    disk: std::sync::Mutex<HashMap<InternalCacheKey, (Bytes, CacheCodec, usize)>>,
    gated: bool,
    serve_rows: AtomicBool,
    /// Report the plane entries of empty partitions as not resident,
    /// as if the tier had evicted them.
    hide_empty_planes: AtomicBool,
    /// When set, every RAM admission through `get_or_insert` waits
    /// for a slot in this bounded channel, like a tiered cache whose
    /// admissions queue their evictions for the disk.
    spill: std::sync::Mutex<Option<tokio::sync::mpsc::Sender<()>>>,
    /// When set, plane loads through `get_or_insert` are recorded
    /// here, and each yields once after it starts so that loads its
    /// caller issues together overlap.
    plane_loads: std::sync::Mutex<Option<Vec<PlaneLoad>>>,
    /// How long every row read waits before it is served or misses,
    /// like a slow tier, so that sparse gathers stay in flight.
    row_read_delay: std::sync::Mutex<Duration>,
    /// Keys of the cached index states written through, so that a
    /// test can evict the states alone from RAM.
    state_keys: std::sync::Mutex<HashSet<InternalCacheKey>>,
    /// Keys whose RAM copy `get` treats as evicted until it admits
    /// the persistent copy again.
    evicted_from_ram: std::sync::Mutex<HashSet<InternalCacheKey>>,
    /// Bytes of churn the next admission of an entry evicted with
    /// `evict_states_from_ram` runs first, as another query's planes
    /// admitted meanwhile would.
    churn_on_admission: AtomicUsize,
    /// Admissions of evicted entries that ran churn first.
    churned_admissions: AtomicUsize,
    /// Source of the keys churned entries take.
    churn_keys: AtomicU64,
}

/// The loader a backend's `get_or_insert` receives.
type EntryLoader<'a> =
    std::pin::Pin<Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>;

/// A plane load through the backend's `get_or_insert`, by plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaneLoad {
    Started(u8),
    Finished(u8),
}

impl std::fmt::Debug for TieredPlaneTestBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredPlaneTestBackend")
            .field("gated", &self.gated)
            .finish_non_exhaustive()
    }
}

impl TieredPlaneTestBackend {
    fn new(ram_bytes: usize, gated: bool) -> Self {
        Self {
            ram: QuickCacheBackend::with_capacity(ram_bytes),
            disk: Default::default(),
            gated,
            serve_rows: AtomicBool::new(true),
            hide_empty_planes: AtomicBool::new(false),
            spill: Default::default(),
            plane_loads: Default::default(),
            row_read_delay: Default::default(),
            state_keys: Default::default(),
            evicted_from_ram: Default::default(),
            churn_on_admission: AtomicUsize::new(0),
            churned_admissions: AtomicUsize::new(0),
            churn_keys: AtomicU64::new(0),
        }
    }

    /// Evict the cached index states from RAM alone, keeping their
    /// persistent copies: the next open reads its state back.
    fn evict_states_from_ram(&self) {
        let states = self.state_keys.lock().unwrap().clone();
        self.evicted_from_ram.lock().unwrap().extend(states);
    }

    /// Run `bytes` of churn before the next admission of an entry
    /// evicted with `evict_states_from_ram`.
    fn churn_on_next_admission(&self, bytes: usize) {
        self.churn_on_admission.store(bytes, Ordering::SeqCst);
    }

    /// Admit `bytes` of entries to RAM, each read back so that it is
    /// promoted to the hot ring and pushes older entries out, as the
    /// planes of another index's queries do.
    async fn churn_ram(&self, bytes: usize) {
        const CHURN_ENTRY_BYTES: usize = 16 * 1024;
        for _ in 0..bytes / CHURN_ENTRY_BYTES {
            let id = self.churn_keys.fetch_add(1, Ordering::Relaxed);
            let mut key = [0xC5; 16];
            key[..8].copy_from_slice(&id.to_le_bytes());
            let key = InternalCacheKey::from_bytes(key);
            let entry = Arc::new(vec![0u8; CHURN_ENTRY_BYTES]);
            self.ram.insert(&key, entry, CHURN_ENTRY_BYTES, None).await;
            self.ram.get(&key, None).await;
        }
    }

    fn record_plane_loads(&self) {
        *self.plane_loads.lock().unwrap() = Some(Vec::new());
    }

    /// The plane loads recorded since `record_plane_loads`, which
    /// stops recording.
    fn take_plane_loads(&self) -> Vec<PlaneLoad> {
        self.plane_loads.lock().unwrap().take().unwrap_or_default()
    }

    /// Record `load` when recording; returns whether it did.
    fn record_plane_load(&self, load: PlaneLoad) -> bool {
        match self.plane_loads.lock().unwrap().as_mut() {
            Some(loads) => {
                loads.push(load);
                true
            }
            None => false,
        }
    }

    async fn get_or_insert_entry(
        &self,
        key: &InternalCacheKey,
        loader: EntryLoader<'_>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        if let Some(entry) = self.ram.get(key, None).await {
            return Ok((entry, true));
        }
        let admitted = match self.read_disk(key) {
            Some((entry, size)) => {
                self.ram.insert(key, entry.clone(), size, None).await;
                (entry, true)
            }
            None => {
                let (entry, size) = loader.await?;
                self.insert(key, entry.clone(), size, codec).await;
                (entry, false)
            }
        };
        let spill = self.spill.lock().unwrap().clone();
        if let Some(spill) = spill {
            // A closed channel only means the test stopped draining it.
            let _ = spill.send(()).await;
        }
        Ok(admitted)
    }

    fn persist(
        &self,
        key: &InternalCacheKey,
        entry: &CacheEntry,
        size: usize,
        codec: Option<CacheCodec>,
    ) {
        let Some(codec) = codec else {
            return;
        };
        let mut bytes = Vec::new();
        codec.serialize(entry, &mut bytes).unwrap();
        self.disk
            .lock()
            .unwrap()
            .insert(*key, (Bytes::from(bytes), codec, size));
    }

    fn read_disk(&self, key: &InternalCacheKey) -> Option<(CacheEntry, usize)> {
        let (bytes, codec, size) = self.disk.lock().unwrap().get(key).cloned()?;
        codec.deserialize(&bytes).hit().map(|entry| (entry, size))
    }

    fn persisted_entries(&self, type_id: &str) -> usize {
        self.disk
            .lock()
            .unwrap()
            .values()
            .filter(|(_, codec, _)| codec.type_id() == type_id)
            .count()
    }
}

#[async_trait::async_trait]
impl CacheBackend for TieredPlaneTestBackend {
    async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        let entry = self.ram.get_resident(key).await?;
        let empty_plane = entry
            .downcast_ref::<PlaneBatch>()
            .is_some_and(|plane| plane.0.num_rows() == 0);
        if empty_plane && self.hide_empty_planes.load(Ordering::Relaxed) {
            return None;
        }
        Some(entry)
    }

    /// RAM as `peek_resident` reports it, hidden empty planes
    /// included, then the serialized tier.
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

    async fn get_rows(
        &self,
        key: &InternalCacheKey,
        rows: &[u32],
        _codec: Option<CacheCodec>,
    ) -> Option<CacheEntry> {
        let delay = *self.row_read_delay.lock().unwrap();
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        if !self.serve_rows.load(Ordering::Relaxed) {
            return None;
        }
        let (bytes, codec, _) = self.disk.lock().unwrap().get(key).cloned()?;
        let reader = |range: Range<usize>| -> Result<Bytes> { Ok(bytes.slice(range)) };
        codec.deserialize_rows(&reader, rows).hit()
    }

    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        let evicted = self.evicted_from_ram.lock().unwrap().remove(key);
        if !evicted && let Some(entry) = self.ram.get(key, None).await {
            return Some(entry);
        }
        let (entry, size) = self.read_disk(key)?;
        let churn = if evicted {
            self.churn_on_admission.swap(0, Ordering::SeqCst)
        } else {
            0
        };
        if churn > 0 {
            self.churn_ram(churn).await;
            self.churned_admissions.fetch_add(1, Ordering::SeqCst);
        }
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
        if entry.downcast_ref::<IvfStateEntryBox>().is_some() {
            self.state_keys.lock().unwrap().insert(*key);
        }
        self.persist(key, &entry, size_bytes, codec);
        self.ram.insert(key, entry, size_bytes, None).await;
    }

    fn max_entry_bytes(&self) -> Option<u64> {
        self.ram.max_entry_bytes()
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: std::pin::Pin<
            Box<dyn futures::Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>,
        >,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let plane = codec
            .and_then(|codec| codec.plane_tag())
            .filter(|&plane| self.record_plane_load(PlaneLoad::Started(plane)));
        if plane.is_some() {
            tokio::task::yield_now().await;
        }
        let result = self.get_or_insert_entry(key, loader, codec).await;
        if let Some(plane) = plane {
            self.record_plane_load(PlaneLoad::Finished(plane));
        }
        result
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

/// Prefilter over a row-address mask. `collapses` makes the scan ask
/// for partition coverage and replace the filter with `NoFilter`.
struct LazyTestFilter {
    mask: Arc<RowAddrMask>,
    collapses: bool,
}

#[async_trait::async_trait]
impl PreFilter for LazyTestFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn needs_partition_row_ids(&self) -> bool {
        self.collapses
    }

    fn is_empty_for(&self, _rows: &RowAddrTreeMap) -> bool {
        self.collapses
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.mask.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask.selected_indices(row_ids)
    }
}

fn lazy_test_filters() -> Vec<(&'static str, Arc<dyn PreFilter>)> {
    let rows = |keep: fn(u64) -> bool| -> RowAddrTreeMap {
        (0..LAZY_ROWS as u64)
            .filter(|&row| keep(row))
            .collect::<Vec<_>>()
            .iter()
            .collect()
    };
    let filter = |mask: RowAddrMask, collapses: bool| -> Arc<dyn PreFilter> {
        Arc::new(LazyTestFilter {
            mask: Arc::new(mask),
            collapses,
        })
    };
    vec![
        ("none", Arc::new(NoFilter) as Arc<dyn PreFilter>),
        (
            "sparse",
            filter(RowAddrMask::from_allowed(rows(|row| row % 20 == 3)), false),
        ),
        (
            "dense",
            filter(RowAddrMask::from_block(rows(|row| row % 20 == 7)), false),
        ),
        (
            "alternating",
            filter(RowAddrMask::from_allowed(rows(|row| row % 2 == 0)), false),
        ),
        (
            "single",
            filter(RowAddrMask::from_allowed(rows(|row| row == 42)), false),
        ),
        (
            "empty",
            filter(RowAddrMask::from_allowed(RowAddrTreeMap::default()), false),
        ),
        ("collapse", filter(RowAddrMask::all_rows(), true)),
    ]
}

/// Clustered vectors around seeded centroids, in shuffled row order,
/// with the partition sizes of [`LAZY_SMALL_PARTITIONS`].
fn lazy_test_data() -> (RecordBatch, Arc<FixedSizeListArray>) {
    let mut rng = StdRng::seed_from_u64(0x1a2e_7ed5);
    let centroids: Vec<f32> = (0..LAZY_PARTITIONS * LAZY_DIM)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let small_rows: usize = LAZY_SMALL_PARTITIONS.iter().map(|(_, rows)| rows).sum();
    let regular_rows = (LAZY_ROWS - small_rows) / (LAZY_PARTITIONS - LAZY_SMALL_PARTITIONS.len());
    let mut sizes: Vec<usize> = (0..LAZY_PARTITIONS)
        .map(|partition| {
            LAZY_SMALL_PARTITIONS
                .iter()
                .find(|(small, _)| *small == partition)
                .map_or(regular_rows, |(_, rows)| *rows)
        })
        .collect();
    sizes[0] += LAZY_ROWS - sizes.iter().sum::<usize>();
    let mut partitions: Vec<usize> = sizes
        .iter()
        .enumerate()
        .flat_map(|(partition, &rows)| repeat_n(partition, rows))
        .collect();
    partitions.shuffle(&mut rng);
    let mut values = Vec::with_capacity(LAZY_ROWS * LAZY_DIM);
    for partition in partitions {
        for dim in 0..LAZY_DIM {
            values.push(centroids[partition * LAZY_DIM + dim] + rng.random_range(-0.05f32..0.05));
        }
    }
    let vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), LAZY_DIM as i32)
            .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("vector", vectors.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from_iter_values(0..LAZY_ROWS as u64)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let centroids =
        FixedSizeListArray::try_new_from_values(Float32Array::from(centroids), LAZY_DIM as i32)
            .unwrap();
    (batch, Arc::new(centroids))
}

async fn write_lazy_test_dataset(
    uri: &str,
    bits: u8,
    distance_type: DistanceType,
) -> (Dataset, RecordBatch) {
    write_rq_test_dataset(uri, bits, distance_type, true).await
}

/// The lazy test data with a layered or a native IVF_RQ index.
async fn write_rq_test_dataset(
    uri: &str,
    bits: u8,
    distance_type: DistanceType,
    layered: bool,
) -> (Dataset, RecordBatch) {
    let (batch, centroids) = lazy_test_data();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, uri, None).await.unwrap();
    let params = VectorIndexParams::with_ivf_rq_params(
        distance_type,
        IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
        RQBuildParams::new(bits).with_layered(layered),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    (dataset, batch)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LazyTestCache {
    /// No capacity: every plane comes from the origin file.
    Origin,
    /// Everything resident after prewarm.
    Resident,
    /// A tiered backend without sign gating, prewarmed to disk.
    Ungated,
    /// A tiered backend without sign gating that holds every plane.
    TieredResident,
    /// An empty tiered backend without sign gating that can hold every plane.
    ColdUngated,
    /// An empty tiered backend with sign gating that can hold every plane.
    ColdGated,
}

async fn open_lazy_test_index(
    uri: &str,
    cache: LazyTestCache,
) -> (
    Dataset,
    Arc<dyn VectorIndex>,
    Option<Arc<TieredPlaneTestBackend>>,
) {
    let (dataset, tiered) = open_lazy_test_dataset(uri, cache).await;
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let index = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    if cache.is_warmed() {
        index.prewarm().await.unwrap();
    }
    (dataset, index, tiered)
}

impl LazyTestCache {
    /// Whether the tests warm an index opened over this cache.
    fn is_warmed(self) -> bool {
        !matches!(self, Self::Origin | Self::ColdUngated | Self::ColdGated)
    }
}

/// The dataset at `uri` over a new session whose index cache is
/// `cache`, and the tiered backend of that cache, if it is one.
async fn open_lazy_test_dataset(
    uri: &str,
    cache: LazyTestCache,
) -> (Dataset, Option<Arc<TieredPlaneTestBackend>>) {
    let registry = Arc::new(ObjectStoreRegistry::default());
    let (session, tiered) = match cache {
        LazyTestCache::Origin => (Session::new(0, LAZY_METADATA_CACHE_BYTES, registry), None),
        LazyTestCache::Resident => (
            Session::new(LAZY_LARGE_CACHE_BYTES, LAZY_METADATA_CACHE_BYTES, registry),
            None,
        ),
        LazyTestCache::Ungated
        | LazyTestCache::TieredResident
        | LazyTestCache::ColdUngated
        | LazyTestCache::ColdGated => {
            let (ram_bytes, gated) = match cache {
                LazyTestCache::TieredResident | LazyTestCache::ColdUngated => {
                    (LAZY_LARGE_CACHE_BYTES, false)
                }
                LazyTestCache::ColdGated => (LAZY_LARGE_CACHE_BYTES, true),
                _ => (LAZY_SMALL_CACHE_BYTES, false),
            };
            let backend = Arc::new(TieredPlaneTestBackend::new(ram_bytes, gated));
            (
                Session::with_index_cache_backend(
                    backend.clone(),
                    LAZY_METADATA_CACHE_BYTES,
                    registry,
                ),
                Some(backend),
            )
        }
    };
    let dataset = crate::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    (dataset, tiered)
}

fn lazy_index(index: &Arc<dyn VectorIndex>) -> &IvfRq {
    index
        .as_any()
        .downcast_ref::<IvfRq>()
        .expect("layered IVF_RQ index")
}

#[derive(Debug, Clone)]
struct RecordingScanFactory {
    opens: Arc<AtomicUsize>,
    searches: Arc<AtomicUsize>,
    behavior: TestScanBehavior,
    cache_layout: RabitCacheLayout,
}

#[derive(Debug, Clone, Copy)]
enum TestScanBehavior {
    Full,
    SearchError,
    OpenError,
}

impl super::super::rabit_scan::RabitScanFactory for RecordingScanFactory {
    fn create(
        &self,
        _file: &IndexFileKey,
        _origin: OriginLatencyClass,
    ) -> Result<Arc<dyn super::super::rabit_scan::RabitScan>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, TestScanBehavior::OpenError) {
            return Err(Error::invalid_input("injected RaBitQ factory failure"));
        }
        Ok(Arc::new(self.clone()))
    }
}

#[async_trait]
impl super::super::rabit_scan::RabitScan for RecordingScanFactory {
    fn cache_layout(&self) -> RabitCacheLayout {
        self.cache_layout
    }

    async fn search(
        &self,
        _context: Arc<super::super::rabit_scan::RabitScanContext>,
    ) -> Result<Option<RecordBatch>> {
        self.searches.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, TestScanBehavior::SearchError) {
            Err(Error::internal("injected RaBitQ execution failure"))
        } else {
            Ok(None)
        }
    }
}

#[rstest]
#[case::fallback(TestScanBehavior::Full)]
#[case::execution_error(TestScanBehavior::SearchError)]
#[case::open_error(TestScanBehavior::OpenError)]
#[tokio::test]
async fn test_rabit_scan_factory_rebinds_from_current_session(
    #[case] behavior: TestScanBehavior,
    #[values(true, false)] layered: bool,
    #[values(RabitCacheLayout::Partition, RabitCacheLayout::Planes)] cache_layout: RabitCacheLayout,
) {
    let batch = gen_batch()
        .with_seed(Seed::from(42))
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .into_batch_rows(RowCount::from(256))
        .unwrap();
    let dir = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, dir.as_str(), None).await.unwrap();
    let vectors = batch["vector"].as_fixed_size_list();
    let centroids = Arc::new(vectors.slice(0, 4));
    let params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(4, centroids).unwrap(),
        RQBuildParams::new(7).with_layered(layered),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let query = lazy_test_query(vectors.value(0), 8, 4);
    let truth =
        super::ground_truth(&dataset, "vector", query.key.as_ref(), 8, DistanceType::L2).await;
    let backend = Arc::new(QuickCacheBackend::with_capacity(4 * 1024 * 1024));
    let session = || {
        Session::with_index_cache_backend(
            backend.clone(),
            LAZY_METADATA_CACHE_BYTES,
            Arc::default(),
        )
    };
    let original = Arc::new(RecordingScanFactory {
        opens: Arc::default(),
        searches: Arc::default(),
        behavior: TestScanBehavior::Full,
        cache_layout,
    });
    let installed = open_with_session(
        dir.as_str(),
        session().with_rabit_scan_factory(original.clone()),
    )
    .await;
    let mut expected = None;
    for opening in 1..=2 {
        let index = installed
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            lazy_index(&index).uses_rabit_plane_cache(),
            layered || cache_layout == RabitCacheLayout::Planes
        );
        let result = search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap();
        let actual = result_bits(&result);
        assert_eq!(original.opens.load(Ordering::Relaxed), opening);
        assert_eq!(original.searches.load(Ordering::Relaxed), opening);
        assert!(
            actual.0.iter().filter(|row| truth.contains(row)).count() >= 4,
            "recall must be >= 0.5"
        );
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected);
        }
        expected = Some(actual);
    }
    let expected = expected.unwrap();

    // Sharing serialized file state never installs another session's strategy.
    let ordinary = open_with_session(dir.as_str(), session()).await;
    let index = ordinary
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(lazy_index(&index).rabit_scan.is_none());
    assert_eq!(lazy_index(&index).uses_rabit_plane_cache(), layered);
    assert_eq!(
        result_bits(
            &search_global(&index, &query, Arc::new(NoFilter))
                .await
                .unwrap()
        ),
        expected
    );
    assert_eq!(original.searches.load(Ordering::Relaxed), 2);

    let replacement_layout = match cache_layout {
        RabitCacheLayout::Partition => RabitCacheLayout::Planes,
        RabitCacheLayout::Planes => RabitCacheLayout::Partition,
    };
    let replacement = Arc::new(RecordingScanFactory {
        behavior,
        cache_layout: replacement_layout,
        opens: Arc::default(),
        searches: Arc::default(),
    });
    let replaced = open_with_session(
        dir.as_str(),
        session().with_rabit_scan_factory(replacement.clone()),
    )
    .await;
    let opened = replaced
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await;
    assert_eq!(replacement.opens.load(Ordering::Relaxed), 1);
    if matches!(behavior, TestScanBehavior::OpenError) {
        let error = opened.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("injected RaBitQ factory failure")
        );
        return;
    }
    let opened = opened.unwrap();
    assert_eq!(
        lazy_index(&opened).uses_rabit_plane_cache(),
        layered || replacement_layout == RabitCacheLayout::Planes
    );
    let result = search_global(&opened, &query, Arc::new(NoFilter)).await;
    assert_eq!(replacement.searches.load(Ordering::Relaxed), 1);
    match behavior {
        TestScanBehavior::Full => assert_eq!(result_bits(&result.unwrap()), expected),
        TestScanBehavior::SearchError => {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("injected RaBitQ execution failure")
            );
        }
        TestScanBehavior::OpenError => unreachable!(),
    }
}

fn lazy_test_query(key: ArrayRef, k: usize, nprobes: usize) -> Query {
    Query {
        search_effort: 0.5,
        rq_cascade_factor: None,
        rq_precision: RQPrecision::Full,
        column: "vector".to_string(),
        key,
        k,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: nprobes,
        maximum_nprobes: Some(nprobes),
        ef: None,
        refine_factor: None,
        metric_type: None,
        use_index: true,
        query_parallelism: DEFAULT_QUERY_PARALLELISM,
        dist_q_c: 0.0,
        approx_mode: ApproxMode::Normal,
    }
}

async fn search_global(
    index: &Arc<dyn VectorIndex>,
    query: &Query,
    filter: Arc<dyn PreFilter>,
) -> Result<RecordBatch> {
    let (partitions, dists) = index.find_partitions(query)?;
    let probes = partitions.len();
    let batches = index
        .clone()
        .search_partitions(
            query.clone(),
            Arc::new(partitions),
            Arc::new(dists),
            0,
            probes,
            filter,
            None,
            Arc::new(NoOpMetricsCollector),
        )
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    Ok(concat_batches(&VECTOR_RESULT_SCHEMA, batches.iter())?)
}

/// Row ids and distance bits in heap order.
fn result_bits(batch: &RecordBatch) -> (Vec<u64>, Vec<u32>) {
    (
        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec(),
        batch[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|dist| dist.to_bits())
            .collect(),
    )
}

/// `auto` makes cloud object stores class high and local files and
/// memory class low, unless the session declares a class; an explicit
/// class applies to every store, whatever the session declares.
#[test]
fn test_origin_latency_class_resolves_by_store() {
    let s3 = ObjectStore::new(
        Arc::new(object_store::memory::InMemory::new()),
        url::Url::parse("s3://bucket/").unwrap(),
        None,
        None,
        false,
        true,
        1,
        lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
        None,
    );
    for (store, auto) in [
        (ObjectStore::local(), OriginLatencyClass::Low),
        (ObjectStore::memory(), OriginLatencyClass::Low),
        (s3, OriginLatencyClass::High),
    ] {
        let scheme = store.scheme().to_string();
        assert_eq!(OriginLatencyClass::of_store(&store), auto, "{scheme}");
        assert_eq!(OriginLatencyClass::resolve(None, &store), auto, "{scheme}");
        assert_eq!(
            OriginLatencyClass::resolve_with_hint(None, None, &store),
            auto,
            "{scheme}"
        );
        for class in [OriginLatencyClass::Low, OriginLatencyClass::High] {
            assert_eq!(
                OriginLatencyClass::resolve(Some(class), &store),
                class,
                "{scheme}"
            );
            assert_eq!(
                OriginLatencyClass::resolve_with_hint(None, Some(class), &store),
                class,
                "{scheme} hint={class}"
            );
            for hint in [OriginLatencyClass::Low, OriginLatencyClass::High] {
                assert_eq!(
                    OriginLatencyClass::resolve_with_hint(Some(class), Some(hint), &store),
                    class,
                    "{scheme} setting={class} hint={hint}"
                );
            }
        }
    }
}

/// The class is resolved when the index opens and carried into its
/// storage, and through a reconstruction from the cached state. The
/// test dataset is a local file, so the class is what the environment
/// sets, or low.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_origin_latency_survives_reconstruction() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
    let expected =
        OriginLatencyClass::resolve(origin_latency_setting().unwrap(), &ObjectStore::local());
    let ivf = lazy_index(&index);
    assert_eq!(ivf.origin_latency(), expected);
    assert_eq!(ivf.storage.origin_latency(), expected);

    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
    let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
    assert!(
        dataset.index_cache.get_with_key(&state_key).await.is_some(),
        "the reopen must reconstruct from the cached state"
    );
    let reopened = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let reconstructed = lazy_index(&reopened);
    assert_eq!(reconstructed.origin_latency(), expected);
    assert_eq!(reconstructed.storage.origin_latency(), expected);
}

/// A session's origin latency hint is the class of the IVF_RQ indexes
/// it opens, when they open and when they are reconstructed from the
/// cached state, and the lazy origin gap and the promotion policy
/// follow it; residency does not. Without a hint an index takes its
/// store's class, low for the test's local files, and an IVF_PQ
/// index the session opens keeps its store's class whatever the
/// hint. Results do not depend on the class. A class the environment
/// sets overrides every hint, so the hint takes effect only where it
/// sets none.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_session_origin_latency_hint_reaches_rq_index() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let rq_dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(rq_dir.as_str(), 7, DistanceType::L2).await;
    let pq_dir = TempStrDir::default();
    let (mut pq_dataset, _) = generate_test_dataset::<Float32Type>(pq_dir.as_str(), 0.0..1.0).await;
    let pq_params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams::new(LIGHTWEIGHT_PQ_PARTITIONS),
        lightweight_pq_params(),
    );
    pq_dataset
        .create_index(&["vector"], IndexType::Vector, None, &pq_params, true)
        .await
        .unwrap();

    let setting = origin_latency_setting().unwrap();

    let resident_setting = resident_columns_setting().unwrap();
    let entry_setting = entry_columns_setting().unwrap();
    let vectors = batch["vector"].as_fixed_size_list();
    let queries = [
        lazy_test_query(vectors.value(0), 10, 8),
        lazy_test_query(vectors.value(777), 100, LAZY_PARTITIONS),
    ];
    let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);

    let mut first_results = vec![None; queries.len()];
    for hint in [
        Some(OriginLatencyClass::High),
        Some(OriginLatencyClass::Low),
        None,
    ] {
        let session = Arc::new(
            Session::new(
                LAZY_LARGE_CACHE_BYTES,
                LAZY_METADATA_CACHE_BYTES,
                Arc::new(ObjectStoreRegistry::default()),
            )
            .with_index_origin_latency(hint),
        );
        let dataset = crate::DatasetBuilder::from_uri(rq_dir.as_str())
            .with_session(session.clone())
            .load()
            .await
            .unwrap();
        let store_class = OriginLatencyClass::of_store(&dataset.object_store);
        assert_eq!(store_class, OriginLatencyClass::Low);
        let expected = setting.or(hint).unwrap_or(store_class);

        let uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let opened = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
        let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
        assert!(
            dataset.index_cache.get_with_key(&state_key).await.is_some(),
            "the reopen must reconstruct from the cached state"
        );
        let reconstructed = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        for (path, index) in [("open", &opened), ("reconstruct", &reconstructed)] {
            let context = format!("hint={hint:?} {path}");
            let ivf = lazy_index(index);
            assert_eq!(ivf.origin_latency(), expected, "{context}");
            assert_eq!(ivf.storage.origin_latency(), expected, "{context}");
            // Residency does not depend on the class: the cache
            // decides it, whatever the hint.
            assert_eq!(
                ivf.resident_columns_enabled(),
                resident_setting.admits(
                    ivf.resident_store_size().unwrap(),
                    dataset.index_cache.max_entry_bytes(),
                ),
                "{context}"
            );
            // Code-only entries follow residency.
            assert_eq!(
                ivf.entry_columns(),
                entry_setting.resolve(ivf.resident_columns_enabled()),
                "{context}"
            );

            for (position, query) in queries.iter().enumerate() {
                let context = format!("{context} query={position}");

                let eager = search_global(index, query, filter.clone()).await.unwrap();
                let bits = result_bits(&eager);
                let first = first_results[position].get_or_insert_with(|| bits.clone());
                assert_eq!(&bits, first, "{context}");
            }
        }

        let pq = crate::DatasetBuilder::from_uri(pq_dir.as_str())
            .with_session(session)
            .load()
            .await
            .unwrap();
        let pq_class = OriginLatencyClass::of_store(&pq.object_store);
        let pq_uuid = pq.load_indices().await.unwrap()[0].uuid;
        let pq_frag_reuse_uuid = pq.frag_reuse_index_uuid().await;
        let pq_state_key =
            crate::index::IvfIndexStateCacheKey::new(&pq_uuid, pq_frag_reuse_uuid.as_ref());
        for path in ["open", "reconstruct"] {
            let cached = pq.index_cache.get_with_key(&pq_state_key).await.is_some();
            assert_eq!(cached, path == "reconstruct", "hint={hint:?} {path}");
            let index = pq
                .open_vector_index("vector", &pq_uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            let ivf_pq = index
                .as_any()
                .downcast_ref::<IvfPq>()
                .expect("IVF_PQ index");
            assert_eq!(ivf_pq.origin_latency(), pq_class, "hint={hint:?} {path}");
            assert_eq!(ivf_pq.entry_columns(), EntryColumns::All);
        }
    }
}

/// A plane's tier follows where the backend holds it: RAM, then only
/// the serialized tier, then nowhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_plane_tier_follows_cache_tiers() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (_dataset, index, tiered) =
        open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
    let tiered = tiered.unwrap();
    let ivf = lazy_index(&index);
    let cache = &ivf.index_cache;
    let partition = (0..LAZY_PARTITIONS)
        .find(|&partition| ivf.storage.partition_size(partition) > 0)
        .unwrap();
    let high = ivf.storage.plane_tier(partition, 1, cache).await;
    assert_eq!(high, CacheTier::Absent);

    ivf.storage
        .load_plane_entry(partition, 1, cache, None)
        .await
        .unwrap();
    let high = ivf.storage.plane_tier(partition, 1, cache).await;
    let low = ivf.storage.plane_tier(partition, 2, cache).await;
    assert_eq!((high, low), (CacheTier::Resident, CacheTier::Absent));

    // A tier peek of a persisted plane does not admit it to RAM.
    tiered.ram.clear().await;
    let high = ivf.storage.plane_tier(partition, 1, cache).await;
    assert_eq!(high, CacheTier::Local);
    assert_eq!(tiered.ram.num_entries().await, 0);

    tiered.clear().await;
    let high = ivf.storage.plane_tier(partition, 1, cache).await;
    assert_eq!(high, CacheTier::Absent);
}

/// Open the index of `dataset` over the dataset's caches, keeping its
/// bounds columns where `sign_bounds` places them.
async fn open_with_sign_bounds(dataset: &Dataset, sign_bounds: SignBounds) -> Arc<dyn VectorIndex> {
    let index = dataset.load_indices().await.unwrap()[0].clone();
    let ivf = IvfRq::try_new(
        dataset.object_store.clone(),
        dataset.indice_files_dir(&index).unwrap(),
        index.uuid,
        None,
        &dataset.metadata_cache,
        dataset.index_cache.for_index(&index.uuid, None),
        index.file_size_map(),
        IvfOpenContext::default(),
    )
    .await
    .unwrap()
    .with_sign_bounds_for_test(sign_bounds);
    Arc::new(ivf)
}

/// Every precision in both approximation modes, the cascade, and the
/// lazy scan over every partition, labelled for assertion messages.
fn sign_bounds_test_queries(key: &ArrayRef) -> Vec<(String, Query)> {
    let mut queries = Vec::new();
    for rq_precision in [RQPrecision::Sign, RQPrecision::High, RQPrecision::Full] {
        for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
            let mut query = lazy_test_query(key.clone(), 100, 8);
            query.rq_precision = rq_precision;
            query.approx_mode = approx_mode;
            queries.push((
                format!("precision={rq_precision:?} approx={approx_mode:?}"),
                query,
            ));
        }
    }
    let mut cascade = lazy_test_query(key.clone(), 100, 8);
    cascade.rq_cascade_factor = Some(4);
    queries.push(("cascade".to_string(), cascade));
    queries.push((
        "full scan".to_string(),
        lazy_test_query(key.clone(), 100, LAZY_PARTITIONS),
    ));
    queries
}

/// Prewarm both placements into their shared cache, where each finds
/// its own sign plane: only the eager placement's holds the bounds.
async fn assert_sign_planes_coexist(placements: &[Arc<dyn VectorIndex>]) {
    for index in placements {
        index.prewarm().await.unwrap();
    }
    for index in placements {
        let ivf = lazy_index(index);
        let partition = (0..LAZY_PARTITIONS)
            .find(|&partition| ivf.storage.partition_size(partition) > 0)
            .unwrap();
        let sign = ivf
            .index_cache
            .get_resident_with_key(&ivf.storage.plane_key(partition, 0))
            .await
            .expect("prewarmed sign plane");
        let bounds_in_sign = sign.0.column_by_name(FULL_BOUNDS_COLUMN).is_some();
        assert_eq!(bounds_in_sign, ivf.sign_bounds() == SignBounds::Eager);
    }
}

/// Keeping the bounds columns in their own plane changes only which
/// entries hold them: every precision, the cascade and the lazy scan
/// return the batch of the sign plane with bounds, bit for bit, from
/// the origin file and from one cache that holds both placements.
#[rstest]
#[case::rq5_l2(5, DistanceType::L2)]
#[case::rq7_cosine(7, DistanceType::Cosine)]
#[case::rq9_dot(9, DistanceType::Dot)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_sign_bounds_placements_match(
    #[case] bits: u8,
    #[case] distance_type: DistanceType,
) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), bits, distance_type).await;
    let vectors = batch["vector"].as_fixed_size_list();
    let keys = [vectors.value(0), vectors.value(777)];
    for cache in [LazyTestCache::Origin, LazyTestCache::TieredResident] {
        let (dataset, _, _) = open_lazy_test_index(dir.as_str(), cache).await;
        let placements = [
            open_with_sign_bounds(&dataset, SignBounds::Eager).await,
            open_with_sign_bounds(&dataset, SignBounds::Lazy).await,
        ];
        if cache == LazyTestCache::TieredResident {
            assert_sign_planes_coexist(&placements).await;
        }
        for key in &keys {
            for (label, query) in sign_bounds_test_queries(key) {
                let context = format!("bits={bits} {distance_type:?} cache={cache:?} {label}");
                let mut results = Vec::new();
                for index in &placements {
                    let result = search_global(index, &query, Arc::new(NoFilter)).await;

                    results.push(result_bits(&result.unwrap()));
                }
                assert!(!results[0].0.is_empty(), "{context}");
                assert_eq!(results[1], results[0], "{context}");
            }
        }
    }
}

/// The bounds plane is read only by the precisions that prune with
/// the bounds, and only when the sign plane leaves them out. Full
/// precision prunes with the error factors every layered file has.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_bounds_plane_loads_only_when_read() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (dataset, _, tiered) = open_lazy_test_index(dir.as_str(), LazyTestCache::ColdUngated).await;
    let tiered = tiered.unwrap();
    for (sign_bounds, high) in [
        (SignBounds::Lazy, vec![0, 1, SIGN_BOUNDS_PLANE]),
        (SignBounds::Eager, vec![0, 1]),
    ] {
        let index = open_with_sign_bounds(&dataset, sign_bounds).await;
        let ivf = lazy_index(&index);
        let partition = (0..LAZY_PARTITIONS)
            .find(|&partition| ivf.storage.partition_size(partition) > 0)
            .unwrap();
        for (precision, expected) in [
            (RQPrecision::Sign, vec![0]),
            (RQPrecision::High, high.clone()),
            (RQPrecision::Full, vec![0, 1, 2]),
        ] {
            tiered.clear().await;
            tiered.record_plane_loads();
            ivf.storage
                .load_partition_at_precision(partition, precision, &ivf.index_cache, None)
                .await
                .unwrap();
            let mut loaded: Vec<u8> = tiered
                .take_plane_loads()
                .into_iter()
                .filter_map(|load| match load {
                    PlaneLoad::Started(plane) => Some(plane),
                    PlaneLoad::Finished(_) => None,
                })
                .collect();
            loaded.sort_unstable();
            assert_eq!(loaded, expected, "{sign_bounds} {precision:?}");
        }
    }
}

/// The bounds placement is what the environment sets, resolved when
/// the index opens and again when it is reconstructed from the
/// cached state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_sign_bounds_survive_reconstruction() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
    let expected = sign_bounds_setting().unwrap();
    let ivf = lazy_index(&index);
    assert!(ivf.is_layered_rq());
    assert_eq!(ivf.sign_bounds(), expected);
    assert_eq!(ivf.storage.sign_bounds(), expected);

    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
    let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
    assert!(
        dataset.index_cache.get_with_key(&state_key).await.is_some(),
        "the reopen must reconstruct from the cached state"
    );
    let reopened = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(lazy_index(&reopened).is_layered_rq());
    assert_eq!(lazy_index(&reopened).sign_bounds(), expected);
}

/// A backend that admits planes like any entry loads a missed
/// partition's three planes at once, so it pays one round trip; a
/// gated backend admits the sign plane before it starts the ex planes,
/// whose admission depends on it. Results are identical.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_plane_loads_overlap_unless_gated() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let key = batch["vector"].as_fixed_size_list().value(17);
    let filter: Arc<dyn PreFilter> = Arc::new(NoFilter);
    let single = lazy_test_query(key.clone(), 10, 1);
    let every = lazy_test_query(key, 100, LAZY_PARTITIONS);
    let (_resident_dataset, resident, _) =
        open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
    let expected = [
        result_bits(
            &search_global(&resident, &single, filter.clone())
                .await
                .unwrap(),
        ),
        result_bits(
            &search_global(&resident, &every, filter.clone())
                .await
                .unwrap(),
        ),
    ];
    for cache in [LazyTestCache::ColdUngated, LazyTestCache::ColdGated] {
        let (_dataset, index, tiered) = open_lazy_test_index(dir.as_str(), cache).await;
        let tiered = tiered.unwrap();
        tiered.record_plane_loads();
        let result = search_global(&index, &single, filter.clone())
            .await
            .unwrap();
        let loads = tiered.take_plane_loads();
        assert_eq!(result_bits(&result), expected[0], "{cache:?}");
        assert_eq!(loads.len(), 6, "{cache:?} {loads:?}");
        let position = |load: PlaneLoad| loads.iter().position(|&seen| seen == load).unwrap();
        let sign_loaded = position(PlaneLoad::Finished(0));
        for plane in [1, 2] {
            let started = position(PlaneLoad::Started(plane));
            if cache == LazyTestCache::ColdGated {
                assert!(sign_loaded < started, "{cache:?} {loads:?}");
            } else {
                assert!(started < sign_loaded, "{cache:?} {loads:?}");
            }
        }
        let result = search_global(&index, &every, filter.clone()).await.unwrap();
        assert_eq!(result_bits(&result), expected[1], "{cache:?}");
    }
}

/// Rows per page of a storage file that [`rewrite_rq_storage`] cuts
/// into small pages: prime, so page ends fall inside partitions.
const RESIDENT_TEST_PAGE_ROWS: usize = 97;
/// First reads issued at once against a storage whose resident
/// columns have not loaded.
const RESIDENT_TEST_CONCURRENT_READS: usize = 32;

/// A reader of `source`'s storage file rewritten as a `version` file:
/// in pages of `page_rows` rows when set, so that partitions straddle
/// pages, or else in the writer's default pages, as the index writer
/// lays it out.
async fn rewrite_rq_storage(
    source: &IvfQuantizationStorage<RabitQuantizer>,
    version: LanceFileVersion,
    page_rows: Option<usize>,
) -> FileReader {
    let schema = source.schema();
    let batches = source
        .reader()
        .read_stream(
            ReadBatchParams::RangeFull,
            u32::MAX,
            1,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch = concat_batches(&schema, batches.iter()).unwrap();
    let store = Arc::new(ObjectStore::memory());
    let path = object_store::path::Path::from("resident/storage.lance");
    // Without a cache, every write flushes a page of each column.
    let options = FileWriterOptions {
        data_cache_bytes: page_rows.map(|_| 1),
        ..Default::default()
    };
    let mut writer = lance_file::versions::create_writer(
        version.resolve(),
        store.create(&path).await.unwrap(),
        lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap(),
        options,
    )
    .unwrap();
    let rows = batch.num_rows();
    let step = page_rows.unwrap_or(rows).max(1);
    for offset in (0..rows).step_by(step) {
        let slice = batch.slice(offset, step.min(rows - offset));
        writer.write_batch(&slice).await.unwrap();
    }
    writer.finish().await.unwrap();
    let scheduler = ScanScheduler::new(store, SchedulerConfig::default_for_testing());
    let file = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

/// A storage over `reader`, a file of `source`'s rows, that takes the
/// small columns from its resident store when `resident`.
fn rq_storage_over(
    source: &IvfQuantizationStorage<RabitQuantizer>,
    reader: &FileReader,
    resident: bool,
) -> IvfQuantizationStorage<RabitQuantizer> {
    IvfQuantizationStorage::<RabitQuantizer>::from_cached_with_remapper(
        reader.clone(),
        source.ivf().clone(),
        source.metadata().clone(),
        source.distance_type(),
        None,
    )
    .with_sign_bounds(source.sign_bounds())
    .with_resident_columns_enabled(resident)
}

/// Whether the file rows `range` span more than one page of the first
/// column of `reader`'s file.
fn straddles_page(reader: &FileReader, range: Range<usize>) -> bool {
    let pages = &reader.metadata().column_infos[0].page_infos;
    let mut end = 0;
    pages.iter().any(|page| {
        end += page.num_rows as usize;
        range.start < end && end < range.end
    })
}

/// Ascending offsets into a partition of `rows` rows, some in runs and
/// some alone, as a sparse gather reads them.
fn resident_test_rows(rows: usize) -> Vec<u32> {
    (0..rows as u32)
        .filter(|row| row % 5 != 3 && row % 7 != 1)
        .collect()
}

/// Open the index of `dataset` over the dataset's caches, with its
/// small columns resident or read from the file, attached to whole
/// partitions and planes as `attach` says.
async fn open_with_resident_columns(
    dataset: &Dataset,
    resident: bool,
    attach: ResidentAttach,
) -> Arc<dyn VectorIndex> {
    let index = dataset.load_indices().await.unwrap()[0].clone();
    let ivf = IvfRq::try_new(
        dataset.object_store.clone(),
        dataset.indice_files_dir(&index).unwrap(),
        index.uuid,
        None,
        &dataset.metadata_cache,
        dataset.index_cache.for_index(&index.uuid, None),
        index.file_size_map(),
        IvfOpenContext::default(),
    )
    .await
    .unwrap()
    .with_resident_columns_for_test(resident)
    .await
    .unwrap()
    .with_resident_attach_for_test(attach);
    Arc::new(ivf)
}

/// Reads that take the small columns from the resident store return
/// the batches that reads of the file return, bit for bit: whole
/// partitions and planes, and planes at selected rows, of every
/// partition, empty ones included, in the writer's pages and in pages
/// that partitions straddle, in either file version. They are charged
/// the same bytes, except that selected rows of a v2.0 file may be
/// charged less.
#[rstest]
#[case::native_v2_0(false, LanceFileVersion::V2_0)]
#[case::native_v2_2(false, LanceFileVersion::V2_2)]
#[case::layered_v2_0(true, LanceFileVersion::V2_0)]
#[case::layered_v2_2(true, LanceFileVersion::V2_2)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_columns_reads_match_file_reads(
    #[case] layered: bool,
    #[case] version: LanceFileVersion,
) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    // A native file has the sign plane's columns too.
    let mut planes = vec![0];
    if layered {
        planes.extend([1, 2]);
        if source.sign_bounds() == SignBounds::Lazy {
            planes.push(SIGN_BOUNDS_PLANE);
        }
    }
    for page_rows in [None, Some(RESIDENT_TEST_PAGE_ROWS)] {
        let reader = rewrite_rq_storage(source, version, page_rows).await;
        let file = rq_storage_over(source, &reader, false);
        let resident = rq_storage_over(source, &reader, true);
        let straddled = (0..source.num_partitions())
            .any(|partition| straddles_page(&reader, source.ivf().row_range(partition)));
        assert_eq!(straddled, page_rows.is_some(), "pages={page_rows:?}");
        for partition in 0..source.num_partitions() {
            let context =
                format!("layered={layered} {version:?} pages={page_rows:?} partition={partition}");
            let expected = file.load_partition(partition, None).await.unwrap();
            let actual = resident.load_partition(partition, None).await.unwrap();
            assert_eq!(
                actual.to_batches().unwrap().collect::<Vec<_>>(),
                expected.to_batches().unwrap().collect::<Vec<_>>(),
                "{context}"
            );
            assert_eq!(actual.deep_size_of(), expected.deep_size_of(), "{context}");
            let rows = resident_test_rows(source.partition_size(partition));
            for &plane in &planes {
                for rows in [None, Some(rows.clone())] {
                    let sparse = rows.is_some();
                    let context = format!("{context} plane={plane} sparse={sparse}");
                    let expected = file
                        .read_plane(partition, plane, rows.clone(), None)
                        .await
                        .unwrap();
                    let actual = resident
                        .read_plane(partition, plane, rows, None)
                        .await
                        .unwrap();
                    assert_eq!(actual, expected, "{context}");
                    let (actual_bytes, expected_bytes) =
                        (actual.deep_size_of(), expected.deep_size_of());
                    if sparse && version == LanceFileVersion::V2_0 {
                        // A v2.0 read of several row runs in a page copies a
                        // fixed-width column's runs into a buffer that reserves
                        // the bytes after the first run only, so it reallocates
                        // to about twice their bytes, while the resident copies
                        // hold exactly theirs. The two reads lay out every
                        // other column, and pages, the same way. Only whole
                        // planes are cached as plane entries, where the charge
                        // must match; selected rows are scored and dropped.
                        assert!(
                            actual_bytes <= expected_bytes,
                            "{context}: {actual_bytes} > {expected_bytes}"
                        );
                    } else {
                        assert_eq!(actual_bytes, expected_bytes, "{context}");
                    }
                }
            }
        }
        assert_eq!(
            resident.resident_columns().loaded_bytes(),
            Some(source.resident_columns_bytes())
        );
    }
}

/// With its small columns resident, a storage reads only the code and
/// bounds columns from the file. In the writer's pages, as the
/// benchmark indexes are laid out, a native partition miss makes 2
/// requests instead of 8, and a layered whole plane 1. The read that
/// loads the store counts the load's requests too.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_columns_cut_origin_requests(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    assert_eq!(lazy_index(&index).is_layered_rq(), layered);
    let source = &lazy_index(&index).storage;
    let reader = rewrite_rq_storage(source, LanceFileVersion::V2_0, None).await;
    // Lazy bounds whatever `LANCE_RQ_SIGN_BOUNDS` sets: a whole sign
    // plane then reads its code column alone from the file, where
    // eager bounds would add the two bounds columns, never resident.
    let file = rq_storage_over(source, &reader, false).with_sign_bounds(SignBounds::Lazy);
    let resident = rq_storage_over(source, &reader, true).with_sign_bounds(SignBounds::Lazy);
    let schema = source.schema();
    // The code and bounds columns are those without a fixed width.
    let file_columns = schema
        .fields()
        .iter()
        .filter(|field| field.data_type().primitive_width().is_none())
        .count() as u64;
    let columns = schema.fields().len() as u64;
    if !layered {
        assert_eq!((columns, file_columns), (8, 2));
    }
    let partition = (0..source.num_partitions())
        .find(|&partition| source.partition_size(partition) > 1)
        .unwrap();
    let requests = |stats: &IoStats| stats.snapshot().iops;

    scan_stats::snapshot_and_reset();
    let first = IoStats::new();
    resident
        .load_partition(partition, Some(first.clone()))
        .await
        .unwrap();
    let load = scan_stats::snapshot_and_reset();
    assert_eq!(load.resident_columns_read_loads, 1, "{load:?}");
    // The read made the load's requests besides its file columns'.
    // Other tests' opens load stores of their own meanwhile, so the
    // process-wide count only bounds this load's from above.
    let load_requests = requests(&first).checked_sub(file_columns).unwrap();
    assert!(load_requests > 0, "{load:?}");
    assert!(
        load.resident_columns_load_requests >= load_requests,
        "{load:?}"
    );

    for (storage, expected) in [(&file, columns), (&resident, file_columns)] {
        let stats = IoStats::new();
        storage
            .load_partition(partition, Some(stats.clone()))
            .await
            .unwrap();
        let enabled = storage.resident_columns_enabled();
        assert_eq!(requests(&stats), expected, "resident={enabled}");
    }
    if layered {
        let rows = resident_test_rows(source.partition_size(partition));
        for plane in [0, 1, 2] {
            let columns = plane_columns(plane, SignBounds::Lazy).len() as u64;
            let mut sparse = Vec::new();
            for (storage, expected) in [(&file, columns), (&resident, 1)] {
                let enabled = storage.resident_columns_enabled();
                let whole = IoStats::new();
                storage
                    .read_plane(partition, plane, None, Some(whole.clone()))
                    .await
                    .unwrap();
                assert_eq!(
                    requests(&whole),
                    expected,
                    "plane={plane} resident={enabled}"
                );
                let stats = IoStats::new();
                storage
                    .read_plane(partition, plane, Some(rows.clone()), Some(stats.clone()))
                    .await
                    .unwrap();
                sparse.push(requests(&stats));
            }
            // A sparse read fetches the row runs of the code column alone.
            assert!(sparse[1] < sparse[0], "plane={plane} {sparse:?}");
        }
    }
    // Only the first read loaded the store: no read loaded one since.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
}

/// First reads issued at once through a storage built outside an index
/// open, which loaded no store for it, load the resident store once,
/// as a fallback counted as a read load, and a storage that shares
/// the store, as a reconstruction of the index does, does not load it
/// again. The load holds the bytes that `resident_columns_bytes`
/// gives without I/O.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_columns_load_once() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    // Row ids and the add, scale and error factors of the sign codes
    // and the add and scale factors of both ex planes.
    let per_row: u64 = source
        .schema()
        .fields()
        .iter()
        .map(|field| match field.data_type() {
            DataType::UInt64 => 8,
            DataType::Float32 => 4,
            _ => 0,
        })
        .sum();
    assert_eq!(per_row, 36);
    let bytes = source.resident_columns_bytes();
    assert_eq!(bytes, per_row * LAZY_ROWS as u64);

    let storage = Arc::new(rq_storage_over(source, source.reader(), true));
    scan_stats::snapshot_and_reset();
    let reads: Vec<_> = (0..RESIDENT_TEST_CONCURRENT_READS)
        .map(|read| {
            let storage = storage.clone();
            tokio::spawn(async move {
                let partition = read % LAZY_PARTITIONS;
                if read % 2 == 0 {
                    storage
                        .read_plane(partition, 0, None, None)
                        .await
                        .map(|plane| plane.num_rows())
                } else {
                    storage
                        .load_partition(partition, None)
                        .await
                        .map(|loaded| loaded.len())
                }
            })
        })
        .collect();
    for read in reads {
        read.await.unwrap().unwrap();
    }
    // One read load, which only this storage could make; other tests'
    // opens load stores of their own meanwhile, so the process-wide
    // bytes only bound this load's from below.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 1, "{stats:?}");
    assert!(stats.resident_columns_bytes >= bytes, "{stats:?}");
    // The arrays hold at least their values.
    assert!(stats.resident_columns_alloc_bytes >= bytes, "{stats:?}");
    assert!(stats.resident_columns_load_requests > 0, "{stats:?}");
    assert!(stats.resident_columns_load_bytes > 0, "{stats:?}");
    assert_eq!(storage.resident_columns().loaded_bytes(), Some(bytes));

    let shared = rq_storage_over(source, source.reader(), true)
        .with_resident_columns(storage.resident_columns().clone());
    let partition = (0..LAZY_PARTITIONS)
        .find(|&partition| source.partition_size(partition) > 0)
        .unwrap();
    shared.load_partition(partition, None).await.unwrap();
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
}

/// Searches return the same batches whether the small columns are
/// resident or read from the origin file, and whether reads view or
/// copy the resident rows: every precision, the cascade and the lazy
/// scan of a layered index, and a native index in both approximation
/// modes.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_columns_search_parity(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let vectors = batch["vector"].as_fixed_size_list();
    let (dataset, _, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    scan_stats::snapshot_and_reset();
    let file = open_with_resident_columns(&dataset, false, ResidentAttach::Share).await;
    let shared = open_with_resident_columns(&dataset, true, ResidentAttach::Share).await;
    let copied = open_with_resident_columns(&dataset, true, ResidentAttach::Copy).await;
    for key in [vectors.value(0), vectors.value(777)] {
        let queries = if layered {
            sign_bounds_test_queries(&key)
        } else {
            [ApproxMode::Normal, ApproxMode::Accurate]
                .into_iter()
                .map(|approx_mode| {
                    let mut query = lazy_test_query(key.clone(), 100, 8);
                    query.approx_mode = approx_mode;
                    let label = format!("approx={approx_mode:?}");
                    (label, query)
                })
                .collect()
        };
        for (label, query) in queries {
            let mut results = Vec::new();
            for index in [&file, &shared, &copied] {
                let result = search_global(index, &query, Arc::new(NoFilter)).await;

                results.push(result_bits(&result.unwrap()));
            }
            let context = format!("layered={layered} {label}");
            assert!(!results[0].0.is_empty(), "{context}");
            assert_eq!(results[1], results[0], "{context} share");
            assert_eq!(results[2], results[0], "{context} copy");
        }
    }
    // Each resident twin holds a store of its own, which other tests
    // cannot load, so its bytes are checked on the twin.
    let bytes = lazy_index(&shared).resident_columns_bytes();
    for (index, attach) in [
        (&shared, ResidentAttach::Share),
        (&copied, ResidentAttach::Copy),
    ] {
        let ivf = lazy_index(index);
        assert_eq!(ivf.resident_attach(), attach);
        assert_eq!(
            ivf.storage.resident_columns().loaded_bytes(),
            Some(bytes),
            "{attach}"
        );
    }
    let stats = scan_stats::snapshot_and_reset();
    assert!(stats.resident_columns_bytes >= 2 * bytes, "{stats:?}");
    let file_store = lazy_index(&file).storage.resident_columns();
    assert_eq!(file_store.loaded_bytes(), None);
}

/// Whether the small columns are resident is what the environment
/// resolves for the index cache, whatever the origin, when the index
/// opens, which loads the store it keeps, and when it is
/// reconstructed from the cached state, and a reconstruction while
/// the index lives shares its store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_columns_survive_reconstruction() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Resident).await;
    let ivf = lazy_index(&index);
    let bytes = ivf.resident_columns_bytes();
    let size = ivf.storage.resident_store_size().unwrap();
    assert_eq!(size.bytes, bytes);
    let setting = resident_columns_setting().unwrap();
    let expected = setting.admits(size, dataset.index_cache.max_entry_bytes());
    assert_eq!(ivf.resident_columns_enabled(), expected);
    assert_eq!(ivf.storage.resident_columns_enabled(), expected);
    let loaded = expected.then_some(bytes);
    assert_eq!(ivf.storage.resident_columns().loaded_bytes(), loaded);

    let uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
    let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
    assert!(
        dataset.index_cache.get_with_key(&state_key).await.is_some(),
        "the reopen must reconstruct from the cached state"
    );
    let reopened = dataset
        .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let reopened = lazy_index(&reopened);
    assert_eq!(reopened.resident_columns_enabled(), expected);
    // A resident store is the one the live index loaded; an index
    // that keeps none resident reads the file.
    let store = reopened.storage.resident_columns();
    assert_eq!(store.loaded_bytes(), loaded, "shared store");
}

/// Scheme of the stores [`CloudTestStoreProvider`] makes.
const CLOUD_TEST_SCHEME: &str = "rqcloud";

/// In-memory stores over one backend, under a scheme that `auto`
/// takes for a cloud object store: a dataset written through a
/// registry holding the provider opens as class high, from any
/// session over that registry.
#[derive(Debug)]
struct CloudTestStoreProvider(Arc<object_store::memory::InMemory>);

#[async_trait::async_trait]
impl lance_io::object_store::ObjectStoreProvider for CloudTestStoreProvider {
    async fn new_store(
        &self,
        base_path: url::Url,
        _params: &ObjectStoreParams,
    ) -> Result<ObjectStore> {
        Ok(ObjectStore::new(
            self.0.clone(),
            base_path,
            None,
            None,
            false,
            true,
            lance_io::object_store::DEFAULT_CLOUD_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ))
    }
}

/// A registry that resolves [`CLOUD_TEST_SCHEME`] URIs to a new
/// in-memory backend.
fn cloud_test_registry() -> Arc<ObjectStoreRegistry> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    let backend = Arc::new(object_store::memory::InMemory::new());
    registry.insert(CLOUD_TEST_SCHEME, Arc::new(CloudTestStoreProvider(backend)));
    registry
}

/// The lazy test data with a layered index, written through
/// `registry` to a new dataset under [`CLOUD_TEST_SCHEME`]: the
/// dataset's URI and the data.
async fn write_cloud_lazy_test_dataset(
    registry: Arc<ObjectStoreRegistry>,
) -> (String, RecordBatch) {
    let uri = format!("{CLOUD_TEST_SCHEME}://bucket/{}", Uuid::new_v4());
    let (batch, centroids) = lazy_test_data();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let params = WriteParams {
        session: Some(Arc::new(Session::new(
            0,
            LAZY_METADATA_CACHE_BYTES,
            registry,
        ))),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, uri.as_str(), Some(params))
        .await
        .unwrap();
    let params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
        RQBuildParams::new(7).with_layered(true),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    (uri, batch)
}

/// The dataset at `uri` over `session`.
async fn open_with_session(uri: &str, session: Session) -> Dataset {
    crate::DatasetBuilder::from_uri(uri)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap()
}

/// Open the index of `dataset` with UUID `uuid`, as a query does, and
/// search it for `query` with the lazy scan settings `config`.
async fn open_and_search(
    dataset: &Dataset,
    uuid: &Uuid,
    query: &Query,
) -> Result<Arc<dyn VectorIndex>> {
    let index = dataset
        .open_vector_index("vector", uuid, &NoOpMetricsCollector)
        .await?;
    search_global(&index, query, Arc::new(NoFilter)).await?;
    Ok(index)
}

/// Indexes of one file opened at once on a fresh session, as a
/// benchmark's workers open a cold index, bind to one resident store
/// and one far gather pool: the opens load the store once, before any
/// search, the index cache charges and pins it while they live, their
/// searches load nothing, and a permit taken through one index is
/// taken from every index's pool. The file is on a store that `auto`
/// takes for a cloud object store, so the index opens as class high.
/// It keeps its small columns resident unless the environment sets
/// otherwise; an index cache with no room admits no store, so there
/// `auto` keeps them in the file.
#[rstest]
#[case::charged(LAZY_LARGE_CACHE_BYTES)]
#[case::zero_capacity(0)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_cold_opens_load_once(#[case] cache_bytes: usize) {
    const OPENS: usize = 8;
    let _serial = LAZY_TEST_LOCK.lock().await;
    let registry = cloud_test_registry();
    let (uri, batch) = write_cloud_lazy_test_dataset(registry.clone()).await;
    let session = Session::new(cache_bytes, LAZY_METADATA_CACHE_BYTES, registry);
    let dataset = open_with_session(&uri, session).await;
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let uuid = index_meta.uuid;
    let file = index_file_key(&dataset, &index_meta);
    let key = batch["vector"].as_fixed_size_list().value(0);
    let query = lazy_test_query(key, 10, 8);

    assert!(!resident_store_is_live(&file));
    scan_stats::snapshot_and_reset();
    let indexes = futures::future::try_join_all(
        (0..OPENS).map(|_| dataset.open_vector_index("vector", &uuid, &NoOpMetricsCollector)),
    )
    .await
    .unwrap();
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    let first = lazy_index(&indexes[0]);
    if origin_latency_setting().unwrap().is_none() {
        assert_eq!(first.origin_latency(), OriginLatencyClass::High);
    }
    let setting = resident_columns_setting().unwrap();
    let bytes = first.resident_columns_bytes();
    let max_entry_bytes = dataset.index_cache.max_entry_bytes();
    assert_eq!(max_entry_bytes, Some(cache_bytes as u64));
    let resident = first.resident_columns_enabled();
    assert_eq!(
        resident,
        setting.admits(
            first.storage.resident_store_size().unwrap(),
            max_entry_bytes,
        )
    );
    if cache_bytes == 0 && setting == ResidentColumnsSetting::Auto {
        assert!(!resident);
        assert!(stats.resident_columns_oversize > 0, "{stats:?}");
    }
    let loaded = resident.then_some(bytes);
    // The opens loaded the file's store, live nowhere before them,
    // exactly when the index keeps it. Other tests open IVF_RQ indexes
    // at the same time, so the process-wide counters only bound this
    // load from below; the file's store is this test's alone.
    assert_eq!(resident_store_is_live(&file), resident);
    assert!(
        stats.resident_columns_loads >= u64::from(resident),
        "{stats:?}"
    );
    if resident {
        assert!(stats.resident_columns_bytes >= bytes, "{stats:?}");
        assert!(stats.resident_columns_load_requests > 0, "{stats:?}");
    }
    for index in &indexes {
        let store = lazy_index(index).storage.resident_columns();
        assert_eq!(store.loaded_bytes(), loaded);
    }
    // The searches read through the loaded store, and load nothing.
    futures::future::try_join_all(
        indexes
            .iter()
            .map(|index| search_global(index, &query, Arc::new(NoFilter))),
    )
    .await
    .unwrap();
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
}

/// A cached state lost while an older index of its file still runs
/// rebinds to that index's resident store and far gather pool, both
/// when the files are opened anew and when the state is read back
/// from the persistent tier, so neither loads the store again, and
/// the open admits the store the cache lost again. The cache then
/// holds it idle, though no state holds it: it stays loaded once
/// every index of the file is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_lost_state_rebinds_index_file_handles() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let registry = cloud_test_registry();
    let (uri, batch) = write_cloud_lazy_test_dataset(registry.clone()).await;
    let tiered = Arc::new(TieredPlaneTestBackend::new(LAZY_LARGE_CACHE_BYTES, false));
    let session =
        Session::with_index_cache_backend(tiered.clone(), LAZY_METADATA_CACHE_BYTES, registry);
    let dataset = open_with_session(&uri, session).await;
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let uuid = index_meta.uuid;
    let file = index_file_key(&dataset, &index_meta);
    let store_key = ResidentColumnsKey::new(&file);
    let file_cache = dataset.index_cache.for_index(&uuid, None);
    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
    let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
    let key = batch["vector"].as_fixed_size_list().value(0);
    let query = lazy_test_query(key, 10, 8);

    assert!(!resident_store_is_live(&file));
    scan_stats::snapshot_and_reset();
    let old = open_and_search(&dataset, &uuid, &query).await.unwrap();
    let ivf = lazy_index(&old);
    let loaded = ivf
        .resident_columns_enabled()
        .then(|| ivf.resident_columns_bytes());
    // Other tests open IVF_RQ indexes at the same time, so what the
    // opens load is checked on the file's store, this test's alone:
    // `old` loaded it, and later opens reused it without loading.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert_eq!(resident_store_is_live(&file), loaded.is_some());
    assert_eq!(ivf.storage.resident_columns().loaded_bytes(), loaded);

    // Evicted from every tier: the next open reads the files anew.
    tiered.clear().await;
    assert!(dataset.index_cache.get_with_key(&state_key).await.is_none());
    let reopened = open_and_search(&dataset, &uuid, &query).await.unwrap();
    // The open found the store gone from the cleared RAM and admitted
    // it again.
    assert_eq!(
        file_cache.peek_resident_with_key(&store_key).await,
        loaded.is_some()
    );
    // Evicted from RAM only: the next open reads the state back.
    assert_eq!(tiered.persisted_entries(IvfStateEntryBox::TYPE_ID), 1);
    tiered.ram.clear().await;
    let read_back = open_and_search(&dataset, &uuid, &query).await.unwrap();
    assert_eq!(
        file_cache.peek_resident_with_key(&store_key).await,
        loaded.is_some()
    );
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    for (label, index) in [("reopened", &reopened), ("read back", &read_back)] {
        let storage = &lazy_index(index).storage;
        assert_eq!(storage.resident_columns().loaded_bytes(), loaded, "{label}");
    }

    // The cache holds the store idle, though no index does.
    drop((old, reopened, read_back));
    assert_eq!(resident_store_is_live(&file), loaded.is_some());
    let again = open_and_search(&dataset, &uuid, &query).await.unwrap();
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    let store = lazy_index(&again).storage.resident_columns();
    assert_eq!(store.loaded_bytes(), loaded);
}

/// RAM of the tiered backend the resident store lifecycle tests use:
/// room for the lazy test index's store under `auto`, which takes at
/// most its entry limit, whatever the origin, and for a query's planes,
/// and small enough that churn evicts idle entries quickly.
const RESIDENT_TEST_RAM_BYTES: usize = 4 * 1024 * 1024;
/// Bytes of the lazy test index's resident store: row ids and seven
/// four-byte factors per row.
const LAZY_RESIDENT_BYTES: u64 = 36 * LAZY_ROWS as u64;
/// Room a store's cache entry may take beyond its arrays' buffers.
const RESIDENT_ENTRY_OVERHEAD_BYTES: u64 = 64 * 1024;

/// Whether the lazy test index, opened over an index cache admitting
/// entries up to `max_entry_bytes` keeps its small columns resident as the
/// environment resolves it, for the default `index` lifetime. The
/// origin plays no part. The resident store lifecycle tests have
/// nothing to check when the environment keeps the columns in the
/// file, and check the `process` lifetime elsewhere.
fn keeps_resident_store(max_entry_bytes: Option<u64>) -> bool {
    let setting = resident_columns_setting().unwrap();
    let index_lifetime = resident_lifetime_setting().unwrap() == ResidentLifetime::Index;
    index_lifetime
        && setting.admits(
            ResidentStoreSize {
                bytes: LAZY_RESIDENT_BYTES,
                charge: LAZY_RESIDENT_BYTES + RESIDENT_ENTRY_OVERHEAD_BYTES,
            },
            max_entry_bytes,
        )
}

/// Whether the lazy test index keeps its small columns resident in
/// an index cache over `backend`; see [`keeps_resident_store`].
fn keeps_resident_store_in(backend: &dyn CacheBackend) -> bool {
    keeps_resident_store(backend.max_entry_bytes())
}

/// The layered lazy test index at `uri`, opened by a session over
/// the index cache `backend` that declares class high, as on S3, for
/// the lazy scan's policies (residency does not depend on the
/// class): the dataset, the index's UUID and the key of its storage
/// file.
struct ResidentTestIndex {
    dataset: Dataset,
    uuid: Uuid,
    file: IndexFileKey,
    open_metrics: IndexMetrics,
}

impl ResidentTestIndex {
    async fn open(uri: &str, backend: Arc<dyn CacheBackend>) -> Self {
        Self::open_with(
            uri,
            backend,
            Some(OriginLatencyClass::High),
            ResidentColumnsSetting::Auto,
        )
        .await
    }

    /// [`Self::open`] by a session that declares `hint` and sets
    /// `resident_columns` instead.
    async fn open_with(
        uri: &str,
        backend: Arc<dyn CacheBackend>,
        hint: Option<OriginLatencyClass>,
        resident_columns: ResidentColumnsSetting,
    ) -> Self {
        let session = Session::with_index_cache_backend(
            backend,
            LAZY_METADATA_CACHE_BYTES,
            Arc::new(ObjectStoreRegistry::default()),
        )
        .with_index_origin_latency(hint)
        .with_index_resident_columns(resident_columns);
        let dataset = open_with_session(uri, session).await;
        let index = dataset.load_indices().await.unwrap()[0].clone();
        let index_dir = dataset.indice_files_dir(&index).unwrap();
        let file = super::super::aux_file_key(&dataset.object_store, &index_dir, &index.uuid);
        Self {
            dataset,
            uuid: index.uuid,
            file,
            open_metrics: IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        }
    }

    async fn open_index(&self) -> Arc<dyn VectorIndex> {
        self.dataset
            .open_vector_index("vector", &self.uuid, &self.open_metrics)
            .await
            .unwrap()
    }

    /// Open the index as a query does, search it and drop it.
    async fn search(&self, query: &Query) -> (Vec<u64>, Vec<u32>) {
        let index = self.open_index().await;
        let result = search_global(&index, query, Arc::new(NoFilter))
            .await
            .unwrap();
        result_bits(&result)
    }

    /// The index's namespace of the index cache that charges the store.
    fn file_cache(&self) -> LanceCache {
        self.dataset.index_cache.for_index(&self.uuid, None)
    }

    /// Whether RAM holds the store, checked without an access.
    async fn store_cached(&self) -> bool {
        self.file_cache()
            .peek_resident_with_key(&ResidentColumnsKey::new(&self.file))
            .await
    }

    /// The store's cache entry, held weakly so that the cache alone
    /// keeps it: an eviction drops it, and a reload or an admission
    /// again caches another.
    async fn cached_store_entry(&self) -> std::sync::Weak<ResidentColumnsEntry> {
        let entry = self
            .file_cache()
            .get_resident_with_key(&ResidentColumnsKey::new(&self.file))
            .await
            .expect("the store is cached");
        Arc::downgrade(&entry)
    }

    /// Assert that RAM still holds `entry`, the entry
    /// [`Self::cached_store_entry`] took: the store was never evicted
    /// since, and so never reloaded or admitted again. Unlike the
    /// process-wide counters, other tests cannot move this.
    async fn assert_store_entry_kept(&self, entry: &std::sync::Weak<ResidentColumnsEntry>) {
        let kept = entry
            .upgrade()
            .expect("the cache never dropped the store's entry");
        let cached = self
            .file_cache()
            .get_resident_with_key(&ResidentColumnsKey::new(&self.file))
            .await
            .expect("the store is cached");
        assert!(Arc::ptr_eq(&cached, &kept));
    }
}

/// Polls of a condition that tasks a search spawned settle, which may
/// hold its index for a moment after the search returned.
const DRAIN_POLLS: usize = 500;
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Wait until no loaded store of `file` is alive.
async fn wait_until_store_freed(file: &IndexFileKey) -> bool {
    for _ in 0..DRAIN_POLLS {
        if !resident_store_is_live(file) {
            return true;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
    }
    false
}

/// The store is an entry of the index's namespace of the index
/// cache, charged what its arrays allocate; no storage charges it
/// again. Index handles keep it alive independently of cache eviction.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_charged_in_index_cache() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

    assert!(!resident_store_is_live(&test.file));
    scan_stats::snapshot_and_reset();
    let index = test.open_index().await;
    search_global(&index, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    // Other tests open IVF_RQ indexes at the same time, so the load is
    // checked on the file's store, this test's alone, and the
    // process-wide counters only bound it from below.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    assert!(
        stats.resident_columns_alloc_bytes >= LAZY_RESIDENT_BYTES,
        "{stats:?}"
    );
    let ivf = lazy_index(&index);
    assert_eq!(
        ivf.storage.resident_columns().loaded_bytes(),
        Some(LAZY_RESIDENT_BYTES)
    );
    let entry = test
        .file_cache()
        .get_resident_with_key(&ResidentColumnsKey::new(&test.file))
        .await
        .expect("the store is cached");
    assert_eq!(entry.loaded_bytes(), Some(LAZY_RESIDENT_BYTES));
    // Measure the entry through ordinary cache admission, including
    // the key and Arc overhead, independently of the other entries.
    let accounting_cache = LanceCache::with_capacity(RESIDENT_TEST_RAM_BYTES);
    accounting_cache
        .insert_with_key(&ResidentColumnsKey::new(&test.file), entry.clone())
        .await;
    let charged = accounting_cache.size_bytes().await as u64;
    assert!(charged >= LAZY_RESIDENT_BYTES, "{charged}");
    drop(accounting_cache);
    assert!(
        charged <= LAZY_RESIDENT_BYTES + RESIDENT_ENTRY_OVERHEAD_BYTES,
        "{charged}"
    );
    assert!((ivf.storage.deep_size_of() as u64) < LAZY_RESIDENT_BYTES);
    assert_eq!(charged, ivf.storage.resident_store_size().unwrap().charge);
    assert!(test.dataset.index_cache.size_bytes().await as u64 >= charged);

    drop((entry, index));
    assert!(test.store_cached().await);
}

/// A cached state read back from a persistent tier is admitted to RAM
/// before the index binds its store, and that admission can evict:
/// the open holds a strong reference first. With the state on the persistent
/// tier alone, the store in RAM idle and no index live, ten times the
/// RAM of churn during the state's admission evicts the cache entry
/// without reloading its store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_state_promotion_from_nvme_keeps_store() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

    assert!(!resident_store_is_live(&test.file));
    scan_stats::snapshot_and_reset();
    let expected = test.search(&query).await;
    let entry = test.cached_store_entry().await;
    let open_io = test.open_metrics.io_stats().unwrap();
    let opened_bytes = open_io.snapshot().bytes_read;
    backend.evict_states_from_ram();
    backend.churn_on_next_admission(10 * RESIDENT_TEST_RAM_BYTES);
    let index = test.open_index().await;
    assert_eq!(
        open_io.snapshot().bytes_read,
        opened_bytes,
        "the cached state must reuse the live store without I/O"
    );
    assert_eq!(
        lazy_index(&index).storage.resident_columns().loaded_bytes(),
        Some(LAZY_RESIDENT_BYTES),
    );
    let actual = result_bits(
        &search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap(),
    );
    assert_eq!(actual, expected);
    assert_eq!(backend.churned_admissions.load(Ordering::SeqCst), 1);
    // Churn evicts the entry while the opener holds its allocation.
    assert!(entry.upgrade().is_none());
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    assert!(stats.resident_store_evictions > 0, "{stats:?}");
    assert!(resident_store_is_live(&test.file));
    drop(index);
    backend.ram.clear().await;
    assert!(wait_until_store_freed(&test.file).await);
}

/// The V12 regression: a cached state evicted while no index of its
/// file is live leaves the store cached, since the state holds none,
/// and the next open reads the state back and binds the same store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_state_eviction_keeps_store() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 100, 16);

    assert!(!resident_store_is_live(&test.file));
    scan_stats::snapshot_and_reset();
    let expected = test.search(&query).await;
    let entry = test.cached_store_entry().await;
    let open_io = test.open_metrics.io_stats().unwrap();
    let opened_bytes = open_io.snapshot().bytes_read;
    for _ in 0..3 {
        backend.evict_states_from_ram();
        let index = test.open_index().await;
        assert_eq!(open_io.snapshot().bytes_read, opened_bytes);
        let actual = search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap();
        assert_eq!(result_bits(&actual), expected);
    }
    // As in `test_state_promotion_from_nvme_keeps_store`: the store is
    // the entry the first search cached, never evicted, reloaded or
    // admitted again.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    test.assert_store_entry_kept(&entry).await;
}

/// Cached index state does not own resident buffers: after every query
/// finishes, evicting the store frees it even while that state is held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cached_state_does_not_own_resident_store() {
    const CONCURRENT_QUERIES: usize = 4;
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let vectors = batch["vector"].as_fixed_size_list();

    let queries: Vec<_> = (0..CONCURRENT_QUERIES)
        .map(|query| lazy_test_query(vectors.value(query * 97), 10, 8))
        .collect();
    futures::future::try_join_all(
        queries
            .iter()
            .map(|query| open_and_search(&test.dataset, &test.uuid, query)),
    )
    .await
    .unwrap();
    test.search(&queries[0]).await;

    let entry = test
        .file_cache()
        .get_resident_with_key(&ResidentColumnsKey::new(&test.file))
        .await
        .expect("the idle store stays cached");
    let frag_reuse_uuid = test.dataset.frag_reuse_index_uuid().await;
    let state_key = crate::index::IvfIndexStateCacheKey::new(&test.uuid, frag_reuse_uuid.as_ref());
    let state = test
        .dataset
        .index_cache
        .get_with_key(&state_key)
        .await
        .unwrap();
    drop(entry);
    backend.churn_ram(10 * RESIDENT_TEST_RAM_BYTES).await;
    assert!(wait_until_store_freed(&test.file).await);
    drop(state);
}

/// An idle store is an ordinary entry: another index's churn evicts
/// it once no index of its file is live, freeing it, and the next
/// open of the file loads it once more, before its query reads, with
/// the same results.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_idle_store_evicted_by_other_index_reloads_once() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

    // Other tests open IVF_RQ indexes at the same time, so the
    // process-wide counters only bound the loads and the eviction
    // from below; the file's store, this test's alone, shows them.
    assert!(!resident_store_is_live(&test.file));
    scan_stats::snapshot_and_reset();
    let expected = test.search(&query).await;
    assert!(test.store_cached().await);
    backend.churn_ram(10 * RESIDENT_TEST_RAM_BYTES).await;
    assert!(!test.store_cached().await);
    assert!(wait_until_store_freed(&test.file).await);
    // The churn pushed the index's planes off the persistent tier
    // too, so the next query reads them from the file.
    backend.disk.lock().unwrap().clear();
    let stats = scan_stats::snapshot_and_reset();
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    assert!(stats.resident_store_evictions >= 1, "{stats:?}");

    // The open loads the freed store again, before any read: no store
    // of the file was live or cached for it to bind.
    let index = test.open_index().await;
    let stats = scan_stats::snapshot_and_reset();
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(resident_store_is_live(&test.file));
    let store = lazy_index(&index).storage.resident_columns();
    assert_eq!(store.loaded_bytes(), Some(LAZY_RESIDENT_BYTES));
    // The cache may refuse readmission after pressure; the index's
    // Arc still owns the loaded store for the query.
    let result = search_global(&index, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(result_bits(&result), expected);
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    drop(index);
    backend.ram.clear().await;
    assert!(wait_until_store_freed(&test.file).await);
}

/// The key of the storage file of the index of `dataset` with
/// metadata `index`, which the index's resident store is shared by.
fn index_file_key(dataset: &Dataset, index: &IndexMetadata) -> IndexFileKey {
    let index_dir = dataset.indice_files_dir(index).unwrap();
    super::super::aux_file_key(&dataset.object_store, &index_dir, &index.uuid)
}

/// An index loads its resident store when it opens, whatever origin
/// class its session declares: before any search the store of its
/// file, live nowhere before the open, is loaded and shared, and the
/// searches then load nothing, with the results of an open that keeps
/// no store. Other tests open IVF_RQ indexes at the same time, so
/// only the read loads, which no open counts, are counted exactly;
/// the file is this test's alone.
#[rstest]
#[case::native_no_hint(false, None)]
#[case::native_low(false, Some(OriginLatencyClass::Low))]
#[case::native_high(false, Some(OriginLatencyClass::High))]
#[case::layered_no_hint(true, None)]
#[case::layered_low(true, Some(OriginLatencyClass::Low))]
#[case::layered_high(true, Some(OriginLatencyClass::High))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_loads_at_open(
    #[case] layered: bool,
    #[case] hint: Option<OriginLatencyClass>,
) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() != ResidentColumnsSetting::Auto
        || resident_lifetime_setting().unwrap() != ResidentLifetime::Index
    {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let vectors = batch["vector"].as_fixed_size_list();
    let queries = [
        lazy_test_query(vectors.value(0), 10, 8),
        lazy_test_query(vectors.value(777), 100, LAZY_PARTITIONS),
    ];
    let mut results = Vec::with_capacity(2);
    for setting in [ResidentColumnsSetting::Auto, ResidentColumnsSetting::Off] {
        let resident = setting == ResidentColumnsSetting::Auto;
        let context = format!("layered={layered} hint={hint:?} {setting}");
        let session = Session::new(
            LAZY_LARGE_CACHE_BYTES,
            LAZY_METADATA_CACHE_BYTES,
            Arc::new(ObjectStoreRegistry::default()),
        )
        .with_index_origin_latency(hint)
        .with_index_resident_columns(setting);
        let dataset = open_with_session(dir.as_str(), session).await;
        let index_meta = dataset.load_indices().await.unwrap()[0].clone();
        let file = index_file_key(&dataset, &index_meta);
        assert!(!resident_store_is_live(&file), "{context}");

        scan_stats::snapshot_and_reset();
        let index = dataset
            .open_vector_index("vector", &index_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let opened = scan_stats::snapshot_and_reset();
        let ivf = lazy_index(&index);
        assert_eq!(ivf.resident_columns_enabled(), resident, "{context}");
        assert!(
            opened.resident_columns_loads >= u64::from(resident),
            "{context} {opened:?}"
        );
        assert_eq!(
            opened.resident_columns_read_loads, 0,
            "{context} {opened:?}"
        );
        let store = ivf.storage.resident_columns();
        assert_eq!(
            store.loaded_bytes(),
            resident.then(|| ivf.resident_columns_bytes()),
            "{context}"
        );
        assert_eq!(resident_store_is_live(&file), resident, "{context}");

        let mut bits = Vec::with_capacity(queries.len());
        for query in &queries {
            let result = search_global(&index, query, Arc::new(NoFilter))
                .await
                .unwrap();
            bits.push(result_bits(&result));
        }
        // A search of an opened index could load a store only as a
        // read's fallback.
        let searched = scan_stats::snapshot_and_reset();
        assert_eq!(
            searched.resident_columns_read_loads, 0,
            "{context} {searched:?}"
        );
        if resident {
            assert!(searched.resident_attach_calls > 0, "{context} {searched:?}");
        }
        results.push(bits);
        // The next open loads a store of its own.
        drop((index, dataset));
        assert!(wait_until_store_freed(&file).await, "{context}");
    }
    assert!(!results[0][0].0.is_empty());
    assert_eq!(results[0], results[1]);
}

/// The open loads the store, not prewarm: an index's prewarm loads
/// nothing and leaves the store and the entries cached, and a
/// dataset's prewarm on a cold cache loads the store through the open
/// it starts with, after which a query binds the prewarmed store and
/// loads nothing. Only the read loads are counted exactly, as in
/// `test_resident_store_loads_at_open`.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_loads_at_open_not_prewarm(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() != ResidentColumnsSetting::Auto
        || entry_columns_setting().unwrap() != EntryColumns::Codes
        || resident_lifetime_setting().unwrap() != ResidentLifetime::Index
    {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let query = lazy_test_query(
        batch["vector"].as_fixed_size_list().value(0),
        100,
        LAZY_PARTITIONS,
    );
    let session = Session::new(
        LAZY_LARGE_CACHE_BYTES,
        LAZY_METADATA_CACHE_BYTES,
        Arc::new(ObjectStoreRegistry::default()),
    );
    let dataset = open_with_session(dir.as_str(), session).await;
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let file = index_file_key(&dataset, &index_meta);
    let store_key = ResidentColumnsKey::new(&file);
    let file_cache = dataset.index_cache.for_index(&index_meta.uuid, None);

    assert!(!resident_store_is_live(&file));
    scan_stats::snapshot_and_reset();
    let index = dataset
        .open_generic_index("vector", &index_meta.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let opened = scan_stats::snapshot_and_reset();
    assert!(opened.resident_columns_loads >= 1, "{opened:?}");
    assert_eq!(opened.resident_columns_read_loads, 0, "{opened:?}");
    let ivf = index.as_any().downcast_ref::<IvfRq>().unwrap();
    let bytes = ivf.resident_columns_bytes();
    assert_eq!(ivf.storage.resident_columns().loaded_bytes(), Some(bytes));
    index.prewarm().await.unwrap();
    let prewarmed = scan_stats::snapshot_and_reset();
    assert_eq!(prewarmed.resident_columns_read_loads, 0, "{prewarmed:?}");
    assert!(file_cache.peek_resident_with_key(&store_key).await);
    for partition in 0..LAZY_PARTITIONS {
        if ivf.storage.partition_size(partition) == 0 {
            continue;
        }
        let cached = if layered {
            let key = ivf.storage.plane_key(partition, 0);
            ivf.index_cache.peek_resident_with_key(&key).await
        } else {
            let key = PartitionCodesKey { partition };
            ivf.index_cache.peek_resident_with_key(&key).await
        };
        assert!(cached, "layered={layered} partition={partition}");
    }

    // A cold cache: no index of the file live and nothing cached.
    drop(index);
    dataset.index_cache.clear().await;
    assert!(wait_until_store_freed(&file).await);
    scan_stats::snapshot_and_reset();
    dataset.prewarm_index(&index_meta.name).await.unwrap();
    let prewarmed = scan_stats::snapshot_and_reset();
    assert!(prewarmed.resident_columns_loads >= 1, "{prewarmed:?}");
    assert_eq!(prewarmed.resident_columns_read_loads, 0, "{prewarmed:?}");
    let cached = file_cache
        .get_resident_with_key(&store_key)
        .await
        .expect("the prewarm's open loaded and cached the store");
    assert_eq!(cached.loaded_bytes(), Some(bytes));

    // The query's open binds the store the prewarm's open loaded and
    // cached, so it has none to load.
    let index = dataset
        .open_vector_index("vector", &index_meta.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(
        lazy_index(&index).storage.resident_columns().loaded_bytes(),
        cached.loaded_bytes()
    );
    let result = search_global(&index, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert!(result.num_rows() > 0);
    let queried = scan_stats::snapshot_and_reset();
    assert_eq!(queried.resident_columns_read_loads, 0, "{queried:?}");
}

/// A resident store that fails to load fails the index open, which
/// leaves the store neither cached nor live, and the next open loads
/// it, with the results of the first. The opens after the first
/// reconstruct the index from its cached state and the files'
/// cached metadata, so the store's load is their only read of the
/// storage file.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_load_failure_fails_open() {
    const INJECTED: &str = "injected resident store load failure";
    const READS: [&str; 2] = ["get_opts", "get_ranges"];
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
    // Reads through the object store crate's local store, which the
    // wrapper sees.
    let path_prefix = if dir.as_str().starts_with('/') {
        ""
    } else {
        "/"
    };
    let uri = format!("file-object-store://{path_prefix}{}", dir.as_str());
    let failing = Arc::new(crate::utils::test::FailingProxyStore::new());
    let session = Session::with_index_cache_backend(
        backend.clone(),
        LAZY_METADATA_CACHE_BYTES,
        Arc::new(ObjectStoreRegistry::default()),
    );
    let dataset = crate::DatasetBuilder::from_uri(&uri)
        .with_session(Arc::new(session))
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(failing.clone()),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let file = index_file_key(&dataset, &index_meta);
    let store_key = ResidentColumnsKey::new(&file);
    let file_cache = dataset.index_cache.for_index(&index_meta.uuid, None);
    let open = || dataset.open_vector_index("vector", &index_meta.uuid, &NoOpMetricsCollector);

    let index = open().await.unwrap();
    let expected = result_bits(
        &search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap(),
    );
    // RAM loses the store; the state stays on the persistent tier.
    drop(index);
    backend.ram.clear().await;
    assert!(wait_until_store_freed(&file).await);

    for read in READS {
        failing.fail_when(read, INDEX_AUXILIARY_FILE_NAME, INJECTED);
    }
    scan_stats::snapshot_and_reset();
    let error = open().await.unwrap_err();
    assert!(error.to_string().contains(INJECTED), "{error}");
    // The open bound the store, then failed to load it.
    let failed = scan_stats::snapshot_and_reset();
    assert!(failed.resident_columns_binds >= 1, "{failed:?}");
    assert!(!file_cache.peek_resident_with_key(&store_key).await);
    assert!(!resident_store_is_live(&file));

    for read in READS {
        failing.clear_fail_when(read, INDEX_AUXILIARY_FILE_NAME);
    }
    let index = open().await.unwrap();
    let reopened = scan_stats::snapshot_and_reset();
    assert!(reopened.resident_columns_loads >= 1, "{reopened:?}");
    assert_eq!(reopened.resident_columns_read_loads, 0, "{reopened:?}");
    let store = lazy_index(&index).storage.resident_columns();
    assert_eq!(store.loaded_bytes(), Some(LAZY_RESIDENT_BYTES));
    assert!(file_cache.peek_resident_with_key(&store_key).await);
    let result = search_global(&index, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(result_bits(&result), expected);
}

/// A state read back from the persistent tier after RAM lost both it
/// and the store, while an older index of the file still runs, binds
/// that index's store through the registry and admits it again,
/// without loading its columns again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_state_from_persistent_tier_binds_resident_store() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);

    let old = test.open_index().await;
    let expected = result_bits(
        &search_global(&old, &query, Arc::new(NoFilter))
            .await
            .unwrap(),
    );
    backend.ram.clear().await;
    assert!(!test.store_cached().await);
    scan_stats::snapshot_and_reset();
    let open_io = test.open_metrics.io_stats().unwrap();
    let opened_bytes = open_io.snapshot().bytes_read;
    let read_back = test.open_index().await;
    assert_eq!(open_io.snapshot().bytes_read, opened_bytes);
    let actual = search_global(&read_back, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(result_bits(&actual), expected);
    // Other tests open IVF_RQ indexes at the same time, so the
    // process-wide counters only bound the reuse and the admission
    // from below; loaded bytes and unchanged load counts verify reuse.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(stats.resident_columns_registry_reuses >= 1, "{stats:?}");
    assert!(test.store_cached().await);
    assert_eq!(
        lazy_index(&read_back)
            .storage
            .resident_columns()
            .loaded_bytes(),
        lazy_index(&old).storage.resident_columns().loaded_bytes(),
    );
}

/// The store is charged in the index's namespace without a fragment
/// reuse segment, so opens under two fragment reuse namespaces, as
/// before and after a compaction with deferred remapping, share one
/// cached store and load it once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_survives_frag_reuse_change() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend.clone()).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
    let index = test.dataset.load_indices().await.unwrap()[0].clone();

    assert!(!resident_store_is_live(&test.file));
    scan_stats::snapshot_and_reset();
    let mut results = Vec::new();
    let mut loaded_stores = Vec::new();
    for frag_reuse_uuid in [Uuid::new_v4(), Uuid::new_v4()] {
        let context = IvfOpenContext {
            rabit_scan_factory: None,
            origin_latency_hint: Some(OriginLatencyClass::High),
            resident_columns: ResidentColumnsSetting::Auto,
            file_cache: Some(test.file_cache()),
            resident_columns_handle: None,
            io_stats: None,
        };
        let opened: Arc<dyn VectorIndex> = Arc::new(
            IvfRq::try_new(
                test.dataset.object_store.clone(),
                test.dataset.indice_files_dir(&index).unwrap(),
                index.uuid,
                None,
                &test.dataset.metadata_cache,
                test.dataset
                    .index_cache
                    .for_index(&index.uuid, Some(&frag_reuse_uuid)),
                index.file_size_map(),
                context,
            )
            .await
            .unwrap(),
        );
        let result = search_global(&opened, &query, Arc::new(NoFilter))
            .await
            .unwrap();
        results.push(result_bits(&result));
        let store = lazy_index(&opened).storage.resident_columns();
        loaded_stores.push(store.loaded_bytes());
    }
    assert_eq!(results[0], results[1]);
    // Other tests open IVF_RQ indexes at the same time, so the
    // process-wide counters only bound the loads and binds from
    // below; both opens have the expected resident data.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert!(stats.resident_columns_loads >= 1, "{stats:?}");
    assert!(stats.resident_columns_binds >= 2, "{stats:?}");
    assert_eq!(loaded_stores, vec![Some(LAZY_RESIDENT_BYTES); 2]);
    assert!(test.store_cached().await);
}

/// The store lives no longer than its session: once every index, the
/// dataset and the session are dropped, it is freed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_resident_store_freed_on_session_drop() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = Arc::new(TieredPlaneTestBackend::new(RESIDENT_TEST_RAM_BYTES, false));
    if !keeps_resident_store_in(backend.as_ref()) {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let test = ResidentTestIndex::open(dir.as_str(), backend).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
    test.search(&query).await;
    assert!(resident_store_is_live(&test.file));
    let file = test.file.clone();
    drop(test);
    assert!(wait_until_store_freed(&file).await);
}

/// A store that exceeds the index cache's
/// largest admissible entry stays in the file under `auto`, which
/// counts it, whatever the origin class the session declares, with
/// the same results as a resident store; a session that turns the
/// store off counts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_oversize_store_resolves_off_under_auto() {
    const SMALL_RAM_BYTES: usize = 512 * 1024;
    let _serial = LAZY_TEST_LOCK.lock().await;
    let backend = |ram_bytes| Arc::new(TieredPlaneTestBackend::new(ram_bytes, false));
    let setting = resident_columns_setting().unwrap();
    if setting != ResidentColumnsSetting::Auto
        || !keeps_resident_store_in(backend(RESIDENT_TEST_RAM_BYTES).as_ref())
    {
        return;
    }
    let small = backend(SMALL_RAM_BYTES);
    assert!(LAZY_RESIDENT_BYTES > small.max_entry_bytes().unwrap());
    let dir = TempStrDir::default();
    let (_, batch) = write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 100, 16);
    let mut results = Vec::new();
    for hint in [
        Some(OriginLatencyClass::High),
        Some(OriginLatencyClass::Low),
        None,
    ] {
        for (ram_bytes, resident_columns) in [
            (SMALL_RAM_BYTES, ResidentColumnsSetting::Auto),
            (RESIDENT_TEST_RAM_BYTES, ResidentColumnsSetting::Auto),
            // A session that turns the store off sizes nothing, so
            // it counts nothing on the cache the store overflows.
            (SMALL_RAM_BYTES, ResidentColumnsSetting::Off),
        ] {
            let resident = ram_bytes == RESIDENT_TEST_RAM_BYTES;
            let oversize = !resident && resident_columns == ResidentColumnsSetting::Auto;
            let context = format!("hint={hint:?} {resident_columns} ram={ram_bytes}");
            let test = ResidentTestIndex::open_with(
                dir.as_str(),
                backend(ram_bytes),
                hint,
                resident_columns,
            )
            .await;
            assert!(!resident_store_is_live(&test.file), "{context}");
            scan_stats::snapshot_and_reset();
            let index = test.open_index().await;
            let ivf = lazy_index(&index);
            assert_eq!(ivf.resident_columns_enabled(), resident, "{context}");
            // The open loaded the file's store, live nowhere before
            // it, exactly when it keeps one.
            assert_eq!(resident_store_is_live(&test.file), resident, "{context}");
            assert_eq!(
                ivf.storage.resident_columns().loaded_bytes(),
                resident.then_some(LAZY_RESIDENT_BYTES),
                "{context}"
            );
            let result = search_global(&index, &query, Arc::new(NoFilter))
                .await
                .unwrap();
            results.push(result_bits(&result));
            // Other tests open IVF_RQ indexes at the same time, so
            // the process-wide counters only bound this open's from
            // below; the off session is also checked through its loaded store.
            let stats = scan_stats::snapshot_and_reset();
            assert!(
                stats.resident_columns_oversize >= u64::from(oversize),
                "{context} {stats:?}"
            );
            assert!(
                stats.resident_columns_loads >= u64::from(resident),
                "{context} {stats:?}"
            );
            assert_eq!(stats.resident_columns_read_loads, 0, "{context} {stats:?}");
            // The next open loads a store of its own.
            let file = test.file.clone();
            drop((index, test));
            assert!(wait_until_store_freed(&file).await, "{context}");
        }
    }
    for result in &results {
        assert_eq!(result, &results[0]);
    }
}

/// `auto` uses the ordinary entry limit on both QuickCache and Moka.
/// With zero capacity it keeps columns in the file; explicit `on`
/// still holds the store through the index. Results are identical.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_auto_residency_follows_the_cache_backend(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() != ResidentColumnsSetting::Auto
        || entry_columns_setting().unwrap() != EntryColumns::Codes
    {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 100, 16);
    let moka = |backend: lance_core::cache::MokaCacheBackend| {
        Session::with_index_cache_backend(
            Arc::new(backend),
            LAZY_METADATA_CACHE_BYTES,
            Arc::new(ObjectStoreRegistry::default()),
        )
    };
    let moka_with_capacity = || {
        moka(lance_core::cache::MokaCacheBackend::with_capacity(
            LAZY_LARGE_CACHE_BYTES,
        ))
    };
    let sessions = [
        ("quick", Session::default(), true),
        ("moka", moka_with_capacity(), true),
        (
            "moka without capacity",
            moka(lance_core::cache::MokaCacheBackend::no_cache()),
            false,
        ),
        (
            "moka without capacity with on",
            moka(lance_core::cache::MokaCacheBackend::no_cache())
                .with_index_resident_columns(ResidentColumnsSetting::On),
            true,
        ),
    ];
    let mut results = Vec::new();
    for (label, session, resident) in sessions {
        let context = format!("layered={layered} {label}");
        let dataset = open_with_session(dir.as_str(), session).await;
        let uuid = dataset.load_indices().await.unwrap()[0].uuid;
        scan_stats::snapshot_and_reset();
        let index = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let opened = scan_stats::snapshot_and_reset();
        let ivf = lazy_index(&index);
        assert_eq!(ivf.resident_columns_enabled(), resident, "{context}");
        let entry_columns = if resident {
            EntryColumns::Codes
        } else {
            EntryColumns::All
        };
        assert_eq!(ivf.entry_columns(), entry_columns, "{context}");
        // Other tests open IVF_RQ indexes at the same time, so the
        // process-wide counters only bound this open's from below; the
        // zero cases are left to the harness, whose helper runs one
        // case per process.
        assert_eq!(
            opened.resident_columns_read_loads, 0,
            "{context} {opened:?}"
        );
        // The open loaded the store before any search.
        let loaded = ivf.storage.resident_columns().loaded_bytes();
        let bytes = ivf.resident_columns_bytes();
        assert_eq!(loaded, resident.then_some(bytes), "{context}");
        let result = search_global(&index, &query, Arc::new(NoFilter))
            .await
            .unwrap();
        results.push(result_bits(&result));
        // A search of an opened index could load a store only as a
        // read's fallback.
        let searched = scan_stats::snapshot_and_reset();
        assert_eq!(
            searched.resident_columns_read_loads, 0,
            "{context} {searched:?}"
        );
    }
    assert!(!results[0].0.is_empty());
    for result in &results {
        assert_eq!(result, &results[0]);
    }
}

/// A session's residency setting reaches the IVF_RQ indexes it
/// opens, when they open and when they are reconstructed from the
/// cached state: `off` keeps the small columns in the file, with
/// full entries, and `auto`, the default, keeps them resident, with
/// code-only entries, when the store fits the cache entry limit. Results
/// do not depend on it. The environment's `on` or `off` overrides
/// every session, so the test checks the sessions only where it
/// sets none.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_session_resident_columns_reach_rq_index(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() != ResidentColumnsSetting::Auto
        || entry_columns_setting().unwrap() != EntryColumns::Codes
    {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let vectors = batch["vector"].as_fixed_size_list();
    let queries = [
        lazy_test_query(vectors.value(0), 10, 8),
        lazy_test_query(vectors.value(777), 100, LAZY_PARTITIONS),
    ];
    let mut first_results = vec![None; queries.len()];
    for (setting, resident) in [
        (ResidentColumnsSetting::Off, false),
        (ResidentColumnsSetting::Auto, true),
    ] {
        let session = Session::new(
            LAZY_LARGE_CACHE_BYTES,
            LAZY_METADATA_CACHE_BYTES,
            Arc::new(ObjectStoreRegistry::default()),
        );
        assert_eq!(
            session.index_resident_columns(),
            ResidentColumnsSetting::Auto
        );
        let session = session.with_index_resident_columns(setting);
        let dataset = open_with_session(dir.as_str(), session).await;
        let uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let opened = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
        let state_key = crate::index::IvfIndexStateCacheKey::new(&uuid, frag_reuse_uuid.as_ref());
        assert!(
            dataset.index_cache.get_with_key(&state_key).await.is_some(),
            "the reopen must reconstruct from the cached state"
        );
        let reconstructed = dataset
            .open_vector_index("vector", &uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        for (path, index) in [("open", &opened), ("reconstruct", &reconstructed)] {
            let context = format!("layered={layered} {setting} {path}");
            let ivf = lazy_index(index);
            assert_eq!(ivf.resident_columns_enabled(), resident, "{context}");
            let entry_columns = if resident {
                EntryColumns::Codes
            } else {
                EntryColumns::All
            };
            assert_eq!(ivf.entry_columns(), entry_columns, "{context}");
            for (position, query) in queries.iter().enumerate() {
                let context = format!("{context} query={position}");
                let result = search_global(index, query, Arc::new(NoFilter))
                    .await
                    .unwrap();
                let bits = result_bits(&result);
                assert!(!bits.0.is_empty(), "{context}");
                let first = first_results[position].get_or_insert_with(|| bits.clone());
                assert_eq!(&bits, first, "{context}");
            }
            let loaded = ivf.storage.resident_columns().loaded_bytes();
            let bytes = ivf.resident_columns_bytes();
            assert_eq!(loaded, resident.then_some(bytes), "{context}");
        }
    }
}

/// Open the index of the dataset at `uri` over a new session whose
/// index cache is `cache`, as if its origin were of `class`, with its
/// small columns resident or read from the file, asking for
/// `entry_columns` and attaching the resident rows as `attach` says
/// (`None` keeps what the environment sets); warm it when the tests
/// warm that cache.
async fn open_entry_columns_test_index(
    uri: &str,
    cache: LazyTestCache,
    class: OriginLatencyClass,
    resident: bool,
    entry_columns: EntryColumns,
    attach: Option<ResidentAttach>,
) -> (
    Dataset,
    Arc<dyn VectorIndex>,
    Option<Arc<TieredPlaneTestBackend>>,
) {
    let (dataset, tiered) = open_lazy_test_dataset(uri, cache).await;
    let index = open_entry_columns_index(&dataset, class, resident, entry_columns, attach).await;
    if cache.is_warmed() {
        index.prewarm().await.unwrap();
    }
    (dataset, index, tiered)
}

/// Open the index of `dataset` over the dataset's caches, as
/// [`open_entry_columns_test_index`] does.
async fn open_entry_columns_index(
    dataset: &Dataset,
    class: OriginLatencyClass,
    resident: bool,
    entry_columns: EntryColumns,
    attach: Option<ResidentAttach>,
) -> Arc<dyn VectorIndex> {
    let (store, index_dir) = index_files(dataset).await;
    let mut ivf = open_rq_index(dataset, store, index_dir)
        .await
        .with_origin_latency_for_test(class)
        .unwrap()
        .with_resident_columns_for_test(resident)
        .await
        .unwrap()
        .with_entry_columns_for_test(entry_columns);
    if let Some(attach) = attach {
        ivf = ivf.with_resident_attach_for_test(attach);
    }
    Arc::new(ivf)
}

/// Bytes the columns of `full` that `entry` does not hold add to the
/// charge of `full`: each array and its buffers.
fn bytes_beyond(full: &RecordBatch, entry: &RecordBatch) -> usize {
    full.schema()
        .fields()
        .iter()
        .zip(full.columns())
        .filter(|(field, _)| entry.column_by_name(field.name()).is_none())
        .map(|(_, column)| column.deep_size_of_children(&mut lance_core::deepsize::Context::new()))
        .sum()
}

fn column_names(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect()
}

/// A code-only entry holds the file columns of its plane or native
/// partition, and is charged the bytes of a read of the file but the
/// resident columns'. Attaching the resident columns to it, whole or
/// at selected rows, gives the batch a read of the file gives, with or
/// without the resident store, bit for bit, and whole planes and
/// partitions byte for byte: every partition, empty ones included, in
/// the writer's pages and in pages that partitions straddle, in either
/// file version, whether whole reads view or copy the resident rows.
/// Whole partitions and planes view the store under `share` alone,
/// and gathered rows never do. An entry of other rows is an error.
#[rstest]
#[case(LanceFileVersion::V2_0)]
#[case(LanceFileVersion::V2_2)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_batch_remapping(
    #[case] version: LanceFileVersion,
    #[values(ResidentAttach::Copy, ResidentAttach::Share)] attach: ResidentAttach,
) {
    use lance_index::scalar::BatchRowIdRemapper;

    const ROW_ID_OFFSET: u64 = 1 << 40;
    #[derive(Debug)]
    struct TestBatchRemapper;

    #[async_trait::async_trait]
    impl BatchRowIdRemapper for TestBatchRemapper {
        async fn remap_row_ids(&self, row_ids: &[u64]) -> lance_core::Result<Vec<Option<u64>>> {
            Ok(row_ids
                .iter()
                .map(|&id| (!id.is_multiple_of(2)).then_some(id + ROW_ID_OFFSET))
                .collect())
        }
    }

    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, false).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    let reader = rewrite_rq_storage(source, version, None).await;
    let file =
        rq_storage_over(source, &reader, false).with_row_id_remapping(Arc::new(TestBatchRemapper));
    let codes = rq_storage_over(source, &reader, true)
        .with_entry_columns(EntryColumns::Codes)
        .with_row_id_remapping(Arc::new(TestBatchRemapper));
    let original = rq_storage_over(source, &reader, true).with_entry_columns(EntryColumns::Codes);
    let cache = lance_core::cache::LanceCache::with_capacity(16 * 1024 * 1024);
    let weak = lance_core::cache::WeakLanceCache::from(&cache);
    let expected = file.load_partition(0, None).await.unwrap();
    let unmapped = original.load_partition(0, None).await.unwrap();
    assert!(!expected.is_empty());
    assert!(expected.len() < unmapped.len());
    for expected_hit in [false, true] {
        let (actual, hit) = codes
            .load_partition_cached(0, &weak, true, attach, None)
            .await
            .unwrap();
        assert_eq!(hit, expected_hit);
        assert_eq!(
            actual.to_batches().unwrap().collect::<Vec<_>>(),
            expected.to_batches().unwrap().collect::<Vec<_>>(),
        );
    }
    // The code cache and shared resident columns keep source rows.
    let (reloaded, hit) = original
        .load_partition_cached(0, &weak, true, attach, None)
        .await
        .unwrap();
    assert!(hit);
    assert_eq!(
        reloaded.to_batches().unwrap().collect::<Vec<_>>(),
        unmapped.to_batches().unwrap().collect::<Vec<_>>(),
    );
}

#[rstest]
#[case::native_v2_0(false, LanceFileVersion::V2_0)]
#[case::native_v2_2(false, LanceFileVersion::V2_2)]
#[case::layered_v2_0(true, LanceFileVersion::V2_0)]
#[case::layered_v2_2(true, LanceFileVersion::V2_2)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_match_file_reads(
    #[case] layered: bool,
    #[case] version: LanceFileVersion,
    #[values(ResidentAttach::Share, ResidentAttach::Copy)] attach: ResidentAttach,
) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    let sign_bounds = source.sign_bounds();
    // A native file has the sign plane's columns too.
    let mut planes = vec![0];
    if layered {
        planes.extend([1, 2]);
        if sign_bounds == SignBounds::Lazy {
            planes.push(SIGN_BOUNDS_PLANE);
        }
    }
    for page_rows in [None, Some(RESIDENT_TEST_PAGE_ROWS)] {
        let reader = rewrite_rq_storage(source, version, page_rows).await;
        let file = rq_storage_over(source, &reader, false);
        let resident = rq_storage_over(source, &reader, true);
        // A store of the process's registry, as an index open binds,
        // so that `shares_resident_store` knows it.
        let store_file = IndexFileKey::new(
            &format!("code-only-{layered}-{version:?}-{page_rows:?}-{attach}"),
            "memory",
            "resident/storage.lance",
        );
        let codes = rq_storage_over(source, &reader, true)
            .with_entry_columns(EntryColumns::Codes)
            .with_resident_attach(attach)
            .with_resident_columns(ResidentColumns::shared(&store_file));
        let shares = attach == ResidentAttach::Share;
        assert_eq!(resident.entry_columns(), EntryColumns::All);
        assert_eq!(codes.entry_columns(), EntryColumns::Codes);
        // Code-only entries need the resident store.
        let unresident =
            rq_storage_over(source, &reader, false).with_entry_columns(EntryColumns::Codes);
        assert_eq!(unresident.entry_columns(), EntryColumns::All);
        for partition in 0..source.num_partitions() {
            let context =
                format!("layered={layered} {version:?} pages={page_rows:?} partition={partition}");
            let size = source.partition_size(partition);
            if !layered {
                let expected = file.load_partition(partition, None).await.unwrap();
                let entry = codes.read_partition_codes(partition, None).await.unwrap();
                assert_eq!(
                    column_names(&entry.0),
                    [RABIT_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_COLUMN],
                    "{context}"
                );
                scan_stats::snapshot_and_reset();
                let actual = codes
                    .partition_from_codes(partition, &entry, attach, None)
                    .await
                    .unwrap();
                let built = scan_stats::snapshot_and_reset();
                assert!(built.code_only_partition_builds >= 1, "{context} {built:?}");
                assert_eq!(
                    actual.to_batches().unwrap().collect::<Vec<_>>(),
                    expected.to_batches().unwrap().collect::<Vec<_>>(),
                    "{context}"
                );
                assert_eq!(
                    actual
                        .to_batches()
                        .unwrap()
                        .any(|batch| shares_resident_store(&batch)),
                    shares && size > 0,
                    "{context}"
                );
                assert_eq!(actual.deep_size_of(), expected.deep_size_of(), "{context}");
                // The entry is charged what the partition built from
                // it is but the copies of the resident rows.
                let actual_batch = actual.to_batches().unwrap().next().unwrap();
                assert_eq!(
                    entry.0.deep_size_of(),
                    actual_batch.deep_size_of() - bytes_beyond(&actual_batch, &entry.0),
                    "{context}"
                );
            }
            let rows = resident_test_rows(size);
            let every_row: Vec<u32> = (0..size as u32).collect();
            for &plane in &planes {
                let context = format!("{context} plane={plane}");
                let full = file.read_plane(partition, plane, None, None).await.unwrap();
                assert_eq!(
                    resident
                        .read_plane(partition, plane, None, None)
                        .await
                        .unwrap(),
                    full,
                    "{context}"
                );
                let entry = codes
                    .read_plane_entry(partition, plane, None)
                    .await
                    .unwrap();
                assert_eq!(
                    column_names(&entry),
                    plane_entry_columns(plane, sign_bounds, EntryColumns::Codes),
                    "{context}"
                );
                let whole = codes
                    .attach_resident(partition, plane, &entry, None, None)
                    .await
                    .unwrap();
                assert_eq!(whole, full, "{context}");
                assert_eq!(whole.deep_size_of(), full.deep_size_of(), "{context}");
                // The bounds plane keeps no resident column.
                let attaches = whole.num_columns() > entry.num_columns();
                assert_eq!(
                    shares_resident_store(&whole),
                    shares && attaches && size > 0,
                    "{context}"
                );
                // The entry is charged what the plane is but the
                // copies of the resident rows.
                assert_eq!(
                    entry.deep_size_of(),
                    whole.deep_size_of() - bytes_beyond(&whole, &entry),
                    "{context}"
                );
                for selected in [&rows, &every_row] {
                    let expected = file
                        .read_plane(partition, plane, Some(selected.clone()), None)
                        .await
                        .unwrap();
                    let entry_rows = entry.take(&UInt32Array::from(selected.clone())).unwrap();
                    let actual = codes
                        .attach_resident(partition, plane, &entry_rows, Some(selected), None)
                        .await
                        .unwrap();
                    assert_eq!(actual, expected, "{context} rows={}", selected.len());
                    assert!(
                        !shares_resident_store(&actual),
                        "{context} rows={}",
                        selected.len()
                    );
                }
                if !rows.is_empty() && rows.len() < size {
                    let error = codes
                        .attach_resident(partition, plane, &entry, Some(&rows), None)
                        .await
                        .unwrap_err();
                    assert!(
                        matches!(error, lance_core::Error::Internal { .. }),
                        "{context}: {error}"
                    );
                    assert!(
                        error
                            .to_string()
                            .contains(&format!("partition {partition} plane {plane}")),
                        "{context}: {error}"
                    );
                }
            }
        }
    }
}

/// Searches return the same batches, and assemble partitions of the
/// same sizes, whether the index caches code-only or full entries,
/// with its small columns resident or not: every precision, the
/// cascade and the lazy scan under every setting of the parity tests
/// of a layered index, reading its planes from the origin file or
/// from a tiered cache that serves selected rows of persisted
/// entries, and a native index in both approximation modes and
/// under every prefilter.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_search_parity(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let key = batch["vector"].as_fixed_size_list().value(777);
    let mut queries: Vec<(String, Query, Arc<dyn PreFilter>)> = Vec::new();
    if layered {
        for (label, query) in sign_bounds_test_queries(&key) {
            queries.push((label, query, Arc::new(NoFilter)));
        }
    } else {
        for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
            for (filter_name, filter) in lazy_test_filters() {
                let mut query = lazy_test_query(key.clone(), 100, 8);
                query.approx_mode = approx_mode;
                let label = format!("approx={approx_mode:?} filter={filter_name}");
                queries.push((label, query, filter));
            }
        }
    }
    // Only code-only entries attach the resident rows, so only they
    // run both attach modes.
    let variants = [
        (EntryColumns::All, false, ResidentAttach::Share),
        (EntryColumns::All, true, ResidentAttach::Share),
        (EntryColumns::Codes, false, ResidentAttach::Share),
        (EntryColumns::Codes, true, ResidentAttach::Share),
        (EntryColumns::Codes, true, ResidentAttach::Copy),
    ];
    for cache in [LazyTestCache::Origin, LazyTestCache::Ungated] {
        let mut expected: Option<Vec<(Vec<u64>, Vec<u32>)>> = None;
        let mut expected_sizes: Option<Vec<usize>> = None;
        for (entry_columns, resident, attach) in variants {
            let context = format!(
                "layered={layered} cache={cache:?} entries={entry_columns} resident={resident} attach={attach}"
            );
            scan_stats::snapshot_and_reset();
            let (_dataset, index, _) = open_entry_columns_test_index(
                dir.as_str(),
                cache,
                OriginLatencyClass::High,
                resident,
                entry_columns,
                Some(attach),
            )
            .await;
            let ivf = lazy_index(&index);
            assert_eq!(
                ivf.entry_columns(),
                entry_columns.resolve(resident),
                "{context}"
            );
            let mut results = Vec::new();
            for (label, query, filter) in &queries {
                let result = search_global(&index, query, filter.clone()).await;

                let result = result.unwrap_or_else(|error| panic!("{context} {label}: {error}"));
                results.push(result_bits(&result));
            }
            assert!(results.iter().any(|(ids, _)| !ids.is_empty()), "{context}");

            let stats = scan_stats::snapshot_and_reset();
            if entry_columns.resolve(resident) == EntryColumns::Codes {
                assert!(stats.resident_attach_calls > 0, "{context} {stats:?}");
            }
            // The tiered cache persisted every plane, so the lazy
            // scan gathers selected rows from the persistent tier and
            // none from the origin file, and those rows, attached to
            // the resident columns, are the rows a read of the file
            // returns.
            if layered && cache == LazyTestCache::Ungated {
                assert_eq!(stats.origin_row_reads, 0, "{context} {stats:?}");
                let partition = (0..LAZY_PARTITIONS)
                    .find(|&partition| ivf.storage.partition_size(partition) > 1)
                    .unwrap();
                let rows = resident_test_rows(ivf.storage.partition_size(partition));
                let plan = GatherPlan {
                    planes: [PlaneSource::Sparse; 2],
                    origin_whole: [false; 2],
                };
                let gathered = ivf
                    .storage
                    .gather_ex_rows(partition, &rows, plan, None, &ivf.index_cache, None)
                    .await
                    .unwrap();
                assert_eq!(gathered.sources, [Some(PlaneSource::Sparse); 2]);
                assert_eq!(gathered.origin_row_reads, 0, "{context}");
                for (plane, batch) in [(1, &gathered.high), (2, &gathered.low)] {
                    let expected = ivf
                        .storage
                        .read_plane(partition, plane, Some(rows.clone()), None)
                        .await
                        .unwrap();
                    // A row read of a full entry decodes a schema
                    // without the file's metadata; the columns match.
                    assert_eq!(
                        column_names(batch),
                        column_names(&expected),
                        "{context} plane={plane}"
                    );
                    assert_eq!(
                        batch.columns(),
                        expected.columns(),
                        "{context} plane={plane}"
                    );
                }
            }
            match &expected {
                None => expected = Some(results),
                Some(expected) => {
                    for ((label, ..), (actual, expected)) in
                        queries.iter().zip(results.iter().zip(expected))
                    {
                        assert_eq!(actual, expected, "{context} {label}");
                    }
                }
            }
            // A partition is assembled of the same arrays from the
            // origin file whatever the entries hold.
            if cache == LazyTestCache::Origin {
                let mut sizes = Vec::with_capacity(LAZY_PARTITIONS);
                for partition in 0..LAZY_PARTITIONS {
                    let entry = if layered {
                        ivf.load_query_partition(
                            partition,
                            RQPrecision::Full,
                            &NoOpMetricsCollector,
                        )
                        .await
                    } else {
                        ivf.load_partition(partition, true, &NoOpMetricsCollector)
                            .await
                    };
                    sizes.push(entry.unwrap().size_bytes());
                }
                match &expected_sizes {
                    None => expected_sizes = Some(sizes),
                    Some(expected) => assert_eq!(&sizes, expected, "{context}"),
                }
            }
        }
    }
}

/// Whether every fixed-width column of `batch`, the row ids and
/// factors a resident store keeps, holds exactly its values: no
/// validity buffer, and a values buffer that allocates its rows'
/// bytes and no more, as the index cache charges an entry. A batch
/// of no rows, such as an empty partition's prewarmed entry, which
/// may keep what the prewarm's read decoded, is not checked.
fn fixed_width_columns_exact(batch: &RecordBatch) -> bool {
    batch.num_rows() == 0
        || batch.columns().iter().all(|column| {
            let data = column.to_data();
            match (column.data_type().primitive_width(), data.buffers()) {
                (Some(width), [values]) => {
                    data.nulls().is_none() && values.capacity() == data.len() * width
                }
                _ => true,
            }
        })
}

/// Wait until no view of the store that `store_views` counts is
/// alive, as the tasks a search spawned drop what they hold.
async fn wait_until_views_dropped(store_views: &ResidentStoreViews) -> u64 {
    for _ in 0..DRAIN_POLLS {
        if store_views.live() == 0 {
            return 0;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
    }
    store_views.live()
}

/// A search of `index` per query of `queries`, sparse gathers of
/// every partition's ex planes of a layered index, from the origin
/// file and from its cached planes, and every partition's stream
/// with and without vectors: every read path that can load a cache
/// entry.
async fn read_every_way(
    index: &Arc<dyn VectorIndex>,
    queries: &[(String, Query, Arc<dyn PreFilter>)],
    label: &str,
) {
    let ivf = lazy_index(index);
    for (name, query, filter) in queries {
        let result = search_global(index, query, filter.clone()).await;

        result.unwrap_or_else(|error| panic!("{label} {name}: {error}"));
    }
    if ivf.is_layered_rq() {
        for partition in 0..LAZY_PARTITIONS {
            let rows = resident_test_rows(ivf.storage.partition_size(partition));
            if rows.is_empty() {
                continue;
            }
            for source in [PlaneSource::Sparse, PlaneSource::Resident] {
                let plan = GatherPlan {
                    planes: [source; 2],
                    origin_whole: [false; 2],
                };
                let gathered = ivf
                    .storage
                    .gather_ex_rows(partition, &rows, plan, None, &ivf.index_cache, None)
                    .await
                    .unwrap();
                // Gathered rows are copies.
                if source == PlaneSource::Sparse {
                    for batch in [&gathered.high, &gathered.low] {
                        assert!(!shares_resident_store(batch), "{label} {partition}");
                    }
                }
            }
        }
    }

    for partition in 0..LAZY_PARTITIONS {
        for with_vector in [false, true] {
            let batches = index
                .partition_reader(partition, with_vector, &NoOpMetricsCollector)
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            assert_eq!(
                rows,
                ivf.storage.partition_size(partition),
                "{label} {partition} with_vector={with_vector}"
            );
            // The stream leaves the index, so it holds copies,
            // whatever reads attach.
            assert!(
                !batches.iter().any(shares_resident_store),
                "{label} {partition} with_vector={with_vector}"
            );
        }
    }
}

/// Probe every key an entry of `ivf` can take in its index cache,
/// which cannot be walked, and check that each entry found holds no
/// memory of a resident store and its fixed-width columns exactly
/// their values; returns how many entries it found.
async fn check_cached_entries(ivf: &IvfRq, label: &str) -> usize {
    let mut probed = 0;
    for partition in 0..LAZY_PARTITIONS {
        let mut entries: Vec<(String, Vec<RecordBatch>)> = Vec::new();
        let key = IVFPartitionKey::<FlatIndex, RabitQuantizer>::new(partition);
        if let Some(entry) = ivf.index_cache.get_resident_with_key(&key).await {
            let batches = entry.storage.to_batches().unwrap().collect();
            entries.push((format!("partition {partition}"), batches));
        }
        let key = PartitionCodesKey { partition };
        if let Some(codes) = ivf.index_cache.get_resident_with_key(&key).await {
            entries.push((format!("codes {partition}"), vec![codes.0.clone()]));
        }
        if ivf.is_layered_rq() {
            for plane in [0, 1, 2, SIGN_BOUNDS_PLANE] {
                let key = ivf.storage.plane_key(partition, plane);
                if let Some(entry) = ivf.index_cache.get_resident_with_key(&key).await {
                    let name = format!("plane {partition}/{plane}");
                    entries.push((name, vec![entry.0.clone()]));
                }
            }
        }
        for (name, batches) in entries {
            probed += 1;
            for batch in &batches {
                assert!(!shares_resident_store(batch), "{label} {name}");
                assert!(fixed_width_columns_exact(batch), "{label} {name}");
            }
        }
    }
    probed
}

/// No cache entry holds memory of the resident store, whatever the
/// entries hold and however reads attach the resident rows: after a
/// prewarm, and again after the cache was cleared so that the reads
/// load every entry, eager, lazy, cascade and prefiltered searches,
/// sparse gathers and partition streams with and without vectors
/// leave every partition, code-only and plane entry the index cache
/// holds in buffers of its own, its fixed-width columns of exactly
/// their values, and every view the reads took has dropped. A read
/// of a code-only entry views the store under `share` alone, and a
/// partition stream, which leaves the index, never does.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cached_entries_never_hold_resident_memory(
    #[case] layered: bool,
    #[values(EntryColumns::All, EntryColumns::Codes)] entry_columns: EntryColumns,
    #[values(ResidentAttach::Share, ResidentAttach::Copy)] attach: ResidentAttach,
) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() == ResidentColumnsSetting::Off {
        return;
    }
    let label = format!("layered={layered} entries={entry_columns} attach={attach}");
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let key = batch["vector"].as_fixed_size_list().value(777);
    let (dataset, _) = open_lazy_test_dataset(dir.as_str(), LazyTestCache::Resident).await;
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let (store, index_dir) = index_files(&dataset).await;
    // Open as a session that keeps the store does, so that the store
    // is the process's and `shares_resident_store` knows it.
    let file_cache = dataset.index_cache.for_index(&index_meta.uuid, None);
    let ivf = IvfRq::try_new(
        store,
        index_dir,
        index_meta.uuid,
        None,
        &dataset.metadata_cache,
        file_cache.clone(),
        index_meta.file_size_map(),
        IvfOpenContext {
            resident_columns: ResidentColumnsSetting::On,
            file_cache: Some(file_cache),
            ..Default::default()
        },
    )
    .await
    .unwrap()
    .with_entry_columns_for_test(entry_columns)
    .with_resident_attach_for_test(attach);
    assert!(ivf.resident_columns_enabled(), "{label}");
    let index: Arc<dyn VectorIndex> = Arc::new(ivf);
    let ivf = lazy_index(&index);
    assert_eq!(ivf.entry_columns(), entry_columns, "{label}");
    let store_views = ivf
        .storage
        .resident_columns()
        .store_views()
        .expect("the open loaded the store");

    let mut queries: Vec<(String, Query, Arc<dyn PreFilter>)> = Vec::new();
    if layered {
        for (name, query) in sign_bounds_test_queries(&key) {
            queries.push((name, query, Arc::new(NoFilter)));
        }
    } else {
        for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
            let mut query = lazy_test_query(key.clone(), 100, 8);
            query.approx_mode = approx_mode;
            let name = format!("approx={approx_mode:?}");
            queries.push((name, query, Arc::new(NoFilter)));
        }
    }
    // A prefiltered query caches the whole partitions it probes
    // (`cache_partition_rows`) where the index caches whole entries.
    for (name, filter) in lazy_test_filters() {
        if name == "sparse" || name == "dense" {
            let query = lazy_test_query(key.clone(), 100, LAZY_PARTITIONS);
            queries.push((format!("filter={name}"), query, filter));
        }
    }

    index.prewarm().await.unwrap();
    read_every_way(&index, &queries, &label).await;
    let warmed = check_cached_entries(ivf, &label).await;
    assert!(warmed >= LAZY_PARTITIONS, "{label} {warmed}");
    // The reads load every entry they need into an empty cache.
    dataset.index_cache.clear().await;
    read_every_way(&index, &queries, &label).await;
    let loaded = check_cached_entries(ivf, &label).await;
    assert!(loaded > 0, "{label}");
    assert_eq!(wait_until_views_dropped(&store_views).await, 0, "{label}");

    // A read of a whole partition or plane, held: it views the store
    // exactly where the index caches code-only entries and shares.
    let partition = (0..LAZY_PARTITIONS)
        .find(|&partition| ivf.storage.partition_size(partition) > 1)
        .unwrap();
    let held = if layered {
        ivf.storage
            .load_plane(partition, 0, &ivf.index_cache, None)
            .await
            .unwrap()
    } else {
        let entry = ivf
            .load_partition(partition, true, &NoOpMetricsCollector)
            .await
            .unwrap();
        entry.storage.to_batches().unwrap().next().unwrap()
    };
    let shares = entry_columns == EntryColumns::Codes && attach == ResidentAttach::Share;
    assert_eq!(shares_resident_store(&held), shares, "{label}");
    assert_eq!(store_views.live() > 0, shares, "{label}");
    drop(held);
    assert_eq!(store_views.live(), 0, "{label}");
}

/// A view keeps its store column's memory alive, not the store: a
/// native partition built for one read from its code-only entry
/// still holds the store's rows, readable and equal to a new read's,
/// after its index dropped and the cache evicted the store, which is
/// then no longer live while the partition's views are. Once the
/// partition drops they are gone. The reopen loads the store again
/// when it opens, and its searches load nothing and find what the
/// first index found.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_shared_rows_outlive_store_eviction() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    if resident_columns_setting().unwrap() != ResidentColumnsSetting::Auto
        || entry_columns_setting().unwrap() != EntryColumns::Codes
        || resident_lifetime_setting().unwrap() != ResidentLifetime::Index
    {
        return;
    }
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, false).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(0), 10, 8);
    let session = Session::new(
        LAZY_LARGE_CACHE_BYTES,
        LAZY_METADATA_CACHE_BYTES,
        Arc::new(ObjectStoreRegistry::default()),
    );
    let dataset = open_with_session(dir.as_str(), session).await;
    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    let file = index_file_key(&dataset, &index_meta);
    let open = || dataset.open_vector_index("vector", &index_meta.uuid, &NoOpMetricsCollector);

    let index = open().await.unwrap();
    let ivf = lazy_index(&index);
    assert_eq!(ivf.entry_columns(), EntryColumns::Codes);
    let expected = search_global(&index, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    let partition = (0..LAZY_PARTITIONS)
        .find(|&partition| ivf.storage.partition_size(partition) > 1)
        .unwrap();
    // Built for this read from the cached code-only entry, not cached.
    let held = ivf
        .load_partition(partition, true, &NoOpMetricsCollector)
        .await
        .unwrap();
    let held_batches: Vec<RecordBatch> = held.storage.to_batches().unwrap().collect();
    assert!(held_batches.iter().any(shares_resident_store));
    let store_views = ivf.storage.resident_columns().store_views().unwrap();
    let views = store_views.live();
    assert!(views > 0);

    drop(index);
    dataset.index_cache.clear().await;
    assert!(wait_until_store_freed(&file).await);
    assert_eq!(store_views.live(), views);
    assert!(resident_store_views() >= views);

    scan_stats::snapshot_and_reset();
    let reopened = open().await.unwrap();
    let opened = scan_stats::snapshot_and_reset();
    assert!(opened.resident_columns_loads >= 1, "{opened:?}");
    assert_eq!(opened.resident_columns_read_loads, 0, "{opened:?}");
    let ivf = lazy_index(&reopened);
    let bytes = ivf.resident_columns_bytes();
    assert_eq!(ivf.storage.resident_columns().loaded_bytes(), Some(bytes));
    let fresh = ivf
        .load_partition(partition, true, &NoOpMetricsCollector)
        .await
        .unwrap();
    let fresh_batches: Vec<RecordBatch> = fresh.storage.to_batches().unwrap().collect();
    assert_eq!(held_batches, fresh_batches);
    drop((fresh, fresh_batches));

    drop((held, held_batches));
    assert_eq!(store_views.live(), 0);
    let result = search_global(&reopened, &query, Arc::new(NoFilter))
        .await
        .unwrap();
    assert_eq!(result_bits(&result), result_bits(&expected));
    let searched = scan_stats::snapshot_and_reset();
    assert_eq!(searched.resident_columns_read_loads, 0, "{searched:?}");
}

/// With its entries cached, a storage with code-only entries reads
/// nothing from the file; a miss reads the code columns alone, one
/// request per column in the writer's pages: 1 for a layered sign,
/// high or low plane, 2 for a native partition. Reading the entries
/// loads no store, and the first attaches, issued at once, load it
/// once.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_cut_origin_requests(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    let reader = rewrite_rq_storage(source, LanceFileVersion::V2_0, None).await;
    // Lazy bounds whatever `LANCE_RQ_SIGN_BOUNDS` sets: a sign plane
    // entry then holds the code column alone.
    let codes = Arc::new(
        rq_storage_over(source, &reader, true)
            .with_sign_bounds(SignBounds::Lazy)
            .with_entry_columns(EntryColumns::Codes),
    );
    let planes: &[u8] = if layered { &[0, 1, 2] } else { &[] };
    let warm = LanceCache::with_capacity(LAZY_LARGE_CACHE_BYTES);
    let requests = |stats: &IoStats| stats.snapshot().iops;

    scan_stats::snapshot_and_reset();
    for partition in 0..LAZY_PARTITIONS {
        if layered {
            for &plane in planes {
                codes
                    .load_plane_entry(partition, plane, &WeakLanceCache::from(&warm), None)
                    .await
                    .unwrap();
            }
        } else {
            let codes = &codes;
            warm.get_or_insert_with_key(PartitionCodesKey { partition }, || async move {
                codes.read_partition_codes(partition, None).await
            })
            .await
            .unwrap();
        }
    }
    // The storage was built outside an index open, so its loads are
    // read loads, which other tests' opens never count.
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
    assert_eq!(codes.resident_columns().loaded_bytes(), None);

    let reads: Vec<_> = (0..RESIDENT_TEST_CONCURRENT_READS)
        .map(|read| {
            let codes = codes.clone();
            let cache = WeakLanceCache::from(&warm);
            tokio::spawn(async move {
                let partition = read % LAZY_PARTITIONS;
                if layered {
                    let plane = (read % 3) as u8;
                    codes
                        .load_plane(partition, plane, &cache, None)
                        .await
                        .map(|plane| (plane.num_rows(), true))
                } else {
                    codes
                        .load_partition_cached(
                            partition,
                            &cache,
                            true,
                            codes.resident_attach(),
                            None,
                        )
                        .await
                        .map(|(storage, hit)| (storage.len(), hit))
                }
            })
        })
        .collect();
    for read in reads {
        let (_, hit) = read.await.unwrap().unwrap();
        assert!(hit);
    }
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 1, "{stats:?}");
    assert_eq!(
        codes.resident_columns().loaded_bytes(),
        Some(source.resident_columns_bytes())
    );
    assert!(stats.resident_attach_calls > 0, "{stats:?}");

    let partition = (0..LAZY_PARTITIONS)
        .find(|&partition| source.partition_size(partition) > 1)
        .unwrap();
    let cold = LanceCache::with_capacity(LAZY_LARGE_CACHE_BYTES);
    for (cache, hit) in [(&warm, true), (&cold, false)] {
        let cache = WeakLanceCache::from(cache);
        if layered {
            for &plane in planes {
                let stats = IoStats::new();
                codes
                    .load_plane(partition, plane, &cache, Some(stats.clone()))
                    .await
                    .unwrap();
                let expected = if hit { 0 } else { 1 };
                assert_eq!(requests(&stats), expected, "plane={plane} hit={hit}");
            }
        } else {
            let stats = IoStats::new();
            let (_, was_hit) = codes
                .load_partition_cached(
                    partition,
                    &cache,
                    true,
                    codes.resident_attach(),
                    Some(stats.clone()),
                )
                .await
                .unwrap();
            assert_eq!(was_hit, hit);
            let expected = if hit { 0 } else { 2 };
            assert_eq!(requests(&stats), expected, "hit={hit}");
        }
    }
    let stats = scan_stats::snapshot_and_reset();
    assert_eq!(stats.resident_columns_read_loads, 0, "{stats:?}");
}

/// Code-only and full entries of one index persist under keys of
/// their own: an index of either kind finds none of the other's
/// entries in a cache the other warmed, and reads its own, every one
/// of which decodes, with the same results.
#[rstest]
#[case::native(false)]
#[case::layered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_persist_apart_from_full_entries(#[case] layered: bool) {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, layered).await;
    let key = batch["vector"].as_fixed_size_list().value(0);
    let query = lazy_test_query(key, 100, LAZY_PARTITIONS);
    let partition_entry_type =
        <PartitionEntry<FlatIndex, RabitQuantizer> as CacheCodecImpl>::TYPE_ID;
    for order in [
        [EntryColumns::All, EntryColumns::Codes],
        [EntryColumns::Codes, EntryColumns::All],
    ] {
        let context = format!("layered={layered} order={order:?}");
        let (dataset, tiered) =
            open_lazy_test_dataset(dir.as_str(), LazyTestCache::ColdUngated).await;
        let tiered = tiered.unwrap();
        let persisted = |entry_columns| match (layered, entry_columns) {
            (true, _) => tiered.persisted_entries(<PlaneBatch as CacheCodecImpl>::TYPE_ID),
            (false, EntryColumns::All) => tiered.persisted_entries(partition_entry_type),
            (false, EntryColumns::Codes) => {
                tiered.persisted_entries(<PartitionCodes as CacheCodecImpl>::TYPE_ID)
            }
        };
        let mut indexes = Vec::new();
        for entry_columns in order {
            let index = open_entry_columns_index(
                &dataset,
                OriginLatencyClass::High,
                true,
                entry_columns,
                None,
            )
            .await;
            assert_eq!(lazy_index(&index).entry_columns(), entry_columns);
            indexes.push(index);
        }
        indexes[0].prewarm().await.unwrap();
        let first_entries = persisted(order[0]);
        assert!(first_entries > 0, "{context}");
        // The second kind finds none of the first kind's entries.
        let second = lazy_index(&indexes[1]);
        for partition in 0..LAZY_PARTITIONS {
            let tiers = if layered {
                let mut tiers = Vec::new();
                for plane in [0, 1, 2] {
                    tiers.push(
                        second
                            .storage
                            .plane_tier(partition, plane, &second.index_cache)
                            .await,
                    );
                }
                tiers
            } else if order[1] == EntryColumns::Codes {
                vec![
                    second
                        .index_cache
                        .peek_tier_with_key(&PartitionCodesKey { partition })
                        .await,
                ]
            } else {
                vec![
                    second
                        .index_cache
                        .peek_tier_with_key(&IVFPartitionKey::<FlatIndex, RabitQuantizer>::new(
                            partition,
                        ))
                        .await,
                ]
            };
            assert!(
                tiers.iter().all(|&tier| tier == CacheTier::Absent),
                "{context} partition={partition} {tiers:?}"
            );
        }
        indexes[1].prewarm().await.unwrap();
        if layered {
            // Both kinds hold the sign, high and low planes.
            assert_eq!(persisted(order[1]), 2 * first_entries, "{context}");
        } else {
            assert_eq!(persisted(order[1]), LAZY_PARTITIONS, "{context}");
            assert_eq!(first_entries, LAZY_PARTITIONS, "{context}");
        }
        for (bytes, codec, _) in tiered.disk.lock().unwrap().values() {
            assert!(
                codec.deserialize(bytes).hit().is_some(),
                "{context}: a persisted {} entry does not decode",
                codec.type_id()
            );
        }
        // Each kind reads its own entries back from the persistent
        // tier, as it wrote them.
        tiered.ram.clear().await;
        let mut results = Vec::new();
        for (index, entry_columns) in indexes.iter().zip(order) {
            let ivf = lazy_index(index);
            let partition = (0..LAZY_PARTITIONS)
                .find(|&partition| ivf.storage.partition_size(partition) > 0)
                .unwrap();
            let (held, expected): (Vec<String>, Vec<&str>) = if layered {
                let entry = ivf
                    .index_cache
                    .get_with_key(&ivf.storage.plane_key(partition, 1))
                    .await
                    .unwrap();
                let sign_bounds = ivf.sign_bounds();
                (
                    column_names(&entry.0),
                    plane_entry_columns(1, sign_bounds, entry_columns),
                )
            } else if entry_columns == EntryColumns::Codes {
                let entry = ivf
                    .index_cache
                    .get_with_key(&PartitionCodesKey { partition })
                    .await
                    .unwrap();
                (
                    column_names(&entry.0),
                    vec![RABIT_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_COLUMN],
                )
            } else {
                (Vec::new(), Vec::new())
            };
            assert_eq!(held, expected, "{context} {entry_columns}");
            let result = search_global(index, &query, Arc::new(NoFilter))
                .await
                .unwrap();
            results.push(result_bits(&result));
        }
        assert_eq!(results[0], results[1], "{context}");
    }
}

/// A native index with code-only entries warms them, one per
/// partition and no whole partition, and a search that follows reads
/// every partition from them: no cache miss and no partition load,
/// with the results of full entries.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_native_prewarm_inserts_partition_codes() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (_, batch) = write_rq_test_dataset(dir.as_str(), 7, DistanceType::L2, false).await;
    let query = lazy_test_query(batch["vector"].as_fixed_size_list().value(3), 100, 16);
    let partition_entry_type =
        <PartitionEntry<FlatIndex, RabitQuantizer> as CacheCodecImpl>::TYPE_ID;
    let mut results = Vec::new();
    for entry_columns in [EntryColumns::Codes, EntryColumns::All] {
        let (_dataset, index, tiered) = open_entry_columns_test_index(
            dir.as_str(),
            LazyTestCache::TieredResident,
            OriginLatencyClass::High,
            true,
            entry_columns,
            None,
        )
        .await;
        let tiered = tiered.unwrap();
        let codes = tiered.persisted_entries(<PartitionCodes as CacheCodecImpl>::TYPE_ID);
        let whole = tiered.persisted_entries(partition_entry_type);
        match entry_columns {
            EntryColumns::Codes => assert_eq!((codes, whole), (LAZY_PARTITIONS, 0)),
            EntryColumns::All => assert_eq!((codes, whole), (0, LAZY_PARTITIONS)),
        }
        let metrics = Arc::new(LocalMetricsCollector::default());
        let (partitions, dists) = index.find_partitions(&query).unwrap();
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
                metrics.clone(),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let result = concat_batches(&VECTOR_RESULT_SCHEMA, batches.iter()).unwrap();
        assert_eq!(metrics.index_cache_misses(), 0, "{entry_columns}");
        assert!(metrics.index_cache_hits() >= probes, "{entry_columns}");
        assert_eq!(
            metrics.parts_loaded.load(Ordering::Relaxed),
            0,
            "{entry_columns}"
        );
        results.push(result_bits(&result));
    }
    assert_eq!(results[0], results[1]);
}

/// A native index whose file stores its sign codes unpacked keeps
/// them packed in its code-only entries: every partition built from a
/// cached entry uses the entry's codes as they are, and searches
/// return the results of full entries, which pack the codes on every
/// construction.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_code_only_entries_keep_unpacked_codes_packed() {
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    let (batch, centroids) = lazy_test_data();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader, dir.as_str(), None).await.unwrap();
    let mut params = VectorIndexParams::with_ivf_rq_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(LAZY_PARTITIONS, centroids).unwrap(),
        RQBuildParams::new(7),
    );
    params.skip_transpose(true);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    let vectors = batch["vector"].as_fixed_size_list();
    let mut results = Vec::new();
    for entry_columns in [EntryColumns::All, EntryColumns::Codes] {
        let (_dataset, index, _) = open_entry_columns_test_index(
            dir.as_str(),
            LazyTestCache::ColdUngated,
            OriginLatencyClass::High,
            true,
            entry_columns,
            None,
        )
        .await;
        let ivf = lazy_index(&index);
        assert!(!ivf.storage.metadata().packed);
        let mut bits = Vec::new();
        for key in [vectors.value(0), vectors.value(777)] {
            for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                let mut query = lazy_test_query(key.clone(), 100, 8);
                query.approx_mode = approx_mode;
                let result = search_global(&index, &query, Arc::new(NoFilter))
                    .await
                    .unwrap();
                bits.push(result_bits(&result));
            }
        }
        results.push(bits);
        if entry_columns == EntryColumns::Codes {
            for partition in 0..LAZY_PARTITIONS {
                if ivf.storage.partition_size(partition) == 0 {
                    continue;
                }
                let loaded = ivf
                    .load_partition(partition, true, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                let cached = ivf
                    .index_cache
                    .get_with_key(&PartitionCodesKey { partition })
                    .await
                    .unwrap();
                let storage_batch = loaded.storage.to_batches().unwrap().next().unwrap();
                let values = |batch: &RecordBatch| {
                    batch[RABIT_CODE_COLUMN]
                        .as_fixed_size_list()
                        .values()
                        .to_data()
                        .buffers()[0]
                        .as_ptr()
                };
                assert_eq!(
                    values(&storage_batch),
                    values(&cached.0),
                    "partition {partition} rewrote its cached codes"
                );
            }
        }
    }
    assert_eq!(results[0], results[1]);
}

/// Increasing gaps exercise touching reads, block-sized coalescing,
/// wider ranges, and a whole column page.
const COALESCE_TEST_GAPS: [u64; 5] = [0, 64 * 1024, 256 * 1024, 1024 * 1024, u64::MAX];
/// Spacing of the single rows of a sparse selection.
const COALESCE_TEST_ROW_STEP: usize = 41;

/// Sparse plane reads return the same batch at every coalescing gap:
/// every plane of every partition, empty ones included, with the
/// small columns read from the file or resident, in the writer's
/// pages and in pages that partitions straddle, in either file
/// version. In a v2.0 file, as the benchmark indexes are laid out, a
/// wider gap makes no more requests, gap 0 makes more than a gap that
/// covers the partition, and that makes no more than a whole read,
/// which is one request per column page.
#[rstest]
#[case::v2_0(LanceFileVersion::V2_0)]
#[case::v2_2(LanceFileVersion::V2_2)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_layered_read_plane_matches_across_coalesce_gaps(#[case] version: LanceFileVersion) {
    assert!(
        COALESCE_TEST_GAPS.is_sorted_by(|narrower, wider| narrower < wider),
        "{COALESCE_TEST_GAPS:?}"
    );
    let _serial = LAZY_TEST_LOCK.lock().await;
    let dir = TempStrDir::default();
    write_lazy_test_dataset(dir.as_str(), 7, DistanceType::L2).await;
    let (_dataset, index, _) = open_lazy_test_index(dir.as_str(), LazyTestCache::Origin).await;
    let source = &lazy_index(&index).storage;
    let mut planes = vec![0, 1, 2];
    if source.sign_bounds() == SignBounds::Lazy {
        planes.push(SIGN_BOUNDS_PLANE);
    }
    // A v2.0 file reads each column page with one request per range
    // group and no page initialization, so its requests are exact.
    let counts_requests = version == LanceFileVersion::V2_0;
    let requests = |stats: &IoStats| stats.snapshot().iops;
    for (page_rows, resident) in [
        (None, false),
        (Some(RESIDENT_TEST_PAGE_ROWS), false),
        (Some(RESIDENT_TEST_PAGE_ROWS), true),
    ] {
        let reader = rewrite_rq_storage(source, version, page_rows).await;
        let storage = rq_storage_over(source, &reader, resident);
        let straddled = (0..source.num_partitions())
            .any(|partition| straddles_page(&reader, source.ivf().row_range(partition)));
        assert_eq!(straddled, page_rows.is_some(), "pages={page_rows:?}");
        for &plane in &planes {
            let layout =
                format!("{version:?} pages={page_rows:?} resident={resident} plane={plane}");
            // Requests at gap 0 and at the covering gap, summed over
            // the partitions and row selections.
            let (mut separate, mut covering) = (0, 0);
            for partition in 0..source.num_partitions() {
                let partition_rows = source.partition_size(partition);
                let selections: [Vec<u32>; 2] = [
                    resident_test_rows(partition_rows),
                    (0..partition_rows as u32)
                        .step_by(COALESCE_TEST_ROW_STEP)
                        .collect(),
                ];
                let whole = IoStats::new();
                if counts_requests {
                    storage
                        .read_plane(partition, plane, None, Some(whole.clone()))
                        .await
                        .unwrap();
                }
                for rows in selections {
                    let context = format!("{layout} partition={partition} rows={}", rows.len());
                    let expected = storage
                        .read_plane(partition, plane, Some(rows.clone()), None)
                        .await
                        .unwrap();
                    let mut gap_requests = Vec::with_capacity(COALESCE_TEST_GAPS.len());
                    for gap in COALESCE_TEST_GAPS {
                        let stats = IoStats::new();
                        let actual = storage
                            .read_plane_with_coalesce_gap(
                                partition,
                                plane,
                                Some(rows.clone()),
                                Some(stats.clone()),
                                Some(gap),
                            )
                            .await
                            .unwrap();
                        assert_eq!(actual, expected, "{context} gap={gap}");
                        gap_requests.push(requests(&stats));
                    }
                    if counts_requests {
                        let context = format!(
                            "{context} requests={gap_requests:?} whole={}",
                            requests(&whole)
                        );
                        assert!(
                            gap_requests.windows(2).all(|pair| pair[1] <= pair[0]),
                            "{context}"
                        );
                        let last = *gap_requests.last().unwrap();
                        assert!(last <= requests(&whole), "{context}");
                        separate += gap_requests[0];
                        covering += last;
                    }
                }
            }
            if counts_requests {
                assert!(separate > covering, "{layout}: {separate} vs {covering}");
            }
        }
    }
}

/// The object store holding the files of `dataset`'s index, and their
/// directory.
async fn index_files(dataset: &Dataset) -> (Arc<ObjectStore>, object_store::path::Path) {
    let index = dataset.load_indices().await.unwrap()[0].clone();
    (
        dataset.object_store.clone(),
        dataset.indice_files_dir(&index).unwrap(),
    )
}

/// Open the index of `dataset` from its files in `index_dir` of
/// `store`, over the dataset's caches.
async fn open_rq_index(
    dataset: &Dataset,
    store: Arc<ObjectStore>,
    index_dir: object_store::path::Path,
) -> IvfRq {
    let index = dataset.load_indices().await.unwrap()[0].clone();
    IvfRq::try_new(
        store,
        index_dir,
        index.uuid,
        None,
        &dataset.metadata_cache,
        dataset.index_cache.for_index(&index.uuid, None),
        index.file_size_map(),
        IvfOpenContext::default(),
    )
    .await
    .unwrap()
}
