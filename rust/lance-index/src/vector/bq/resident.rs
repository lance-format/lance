// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The small columns of an IVF_RQ storage file, kept in memory.
//!
//! A read of a partition or plane from the file costs at least one request
//! per column, yet every column but the codes and the estimator bounds holds
//! a few bytes per row: row ids and factors. With those columns read once
//! per index, later reads fetch only the codes (and bounds) from the file:
//! on an object store a native partition miss takes 2 requests instead of
//! 8, and a layered one 3 instead of 13. Reads attach the resident rows, so
//! a batch holds, and is charged, what a file read returns. Cache entries
//! hold that batch, or only the columns reads fetch from the file
//! (`EntryColumns::Codes`), and every read of such an entry attaches the
//! resident rows again: a whole partition or plane gets views of the store's
//! buffers, each of exactly its rows' bytes, unless
//! `LANCE_RQ_RESIDENT_ATTACH=copy`; gathered rows, and whole reads that
//! become cache entries, get copies. A view keeps its store column's whole
//! allocation alive until it drops, so views live within a read and never
//! reach a cache entry; [`resident_store_views`] counts the live ones and
//! [`shares_resident_store`] tells whether a batch holds store memory.
//!
//! An index loads the store when it opens, so that no read waits for it.
//! The store is an entry of the index cache ([`ResidentColumnsKey`]),
//! charged what its arrays allocate. The handles of live indexes lease it
//! ([`ResidentColumns`]), so a backend that pins entries keeps it in RAM while
//! an index of the file is in use; an idle store is an ordinary entry,
//! evicted under pressure and loaded again by the next open of the file.
//! Every live handle of a file shares one store through a weak registry, so
//! indexes opened at once load it once and an evicted store still in use is
//! admitted again rather than loaded again. A read loads the store only as a
//! fallback, for a storage built outside an index open, and counts it in
//! `resident_columns_read_loads`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use std::time::Instant;

use arrow::array::ArrayData;
use arrow::buffer::Buffer;
use arrow_array::{
    Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array, make_array, new_empty_array,
};
use arrow_schema::{Field, Schema, SchemaRef};
use arrow_select::take::take;
use futures::TryStreamExt;
use lance_core::cache::{
    CacheKey, CacheKeySchema, CacheLease, CachePin, InternalCacheKey, KeyBuilder, LanceCache,
    PinnedValue, WeakLanceCache,
};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_encoding::decoder::{FilterExpression, PageInfo};
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_io::scheduler::IoStats;
use tokio::sync::OnceCell;

use super::layered::{FULL_BOUNDS_COLUMN, HIGH_BOUNDS_COLUMN};
use super::layered_stats;
use super::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN, RABIT_CODE_COLUMN,
    RABIT_EX_CODE_COLUMN,
};
use crate::vector::exact_buffers::{exact_array, exact_batch};
use crate::vector::storage::{
    IndexFileKey, ResidentAttach, ResidentLifetime, WeakRegistry, shared_by_key,
};

/// Columns that reads always fetch from the file: the codes, which hold most
/// of its bytes, and the estimator bounds, which only some scans read.
const FILE_COLUMNS: [&str; 6] = [
    RABIT_CODE_COLUMN,
    RABIT_EX_CODE_COLUMN,
    RABIT_BLOCKED_EX_CODE_COLUMN,
    RABIT_BLOCKED_EX_CODE_LO_COLUMN,
    HIGH_BOUNDS_COLUMN,
    FULL_BOUNDS_COLUMN,
];

/// Byte width of `field`'s values when the store keeps the column: every
/// fixed-width primitive column but [`FILE_COLUMNS`].
fn resident_width(field: &Field) -> Option<usize> {
    if FILE_COLUMNS.contains(&field.name().as_str()) {
        return None;
    }
    field.data_type().primitive_width()
}

/// Whether the store keeps column `field` of an IVF_RQ storage file.
pub(crate) fn is_resident(field: &Field) -> bool {
    resident_width(field).is_some()
}

/// Whether reads always fetch column `name` of an IVF_RQ storage file from
/// the file: the columns a code-only cache entry holds. Every other column
/// of a plane or partition is a fixed-width one the store keeps.
pub(crate) fn is_file_column(name: &str) -> bool {
    FILE_COLUMNS.contains(&name)
}

/// Bytes of values the resident store holds for an IVF_RQ storage file with
/// `schema` and `num_rows` rows: every row of every column it keeps, at the
/// column's fixed width. It reads nothing, so an open can size the store
/// against the index cache before loading it. The store's allocations, which
/// the cache charges, can exceed it by a few bytes per column.
pub fn resident_columns_bytes(schema: &Schema, num_rows: u64) -> u64 {
    schema
        .fields()
        .iter()
        .filter_map(|field| resident_width(field))
        .map(|width| width as u64 * num_rows)
        .sum()
}

/// The stores of the index files with a live handle or cache entry, held
/// weakly: only dedupes loads, as the handles and the index cache own them.
static RESIDENT_SLOTS: LazyLock<WeakRegistry<IndexFileKey, ResidentSlot>> =
    LazyLock::new(Default::default);

/// Handles kept for the life of the process under
/// [`ResidentLifetime::Process`], one per index file, each with its lease.
static PROCESS_STORES: LazyLock<Mutex<HashMap<IndexFileKey, ResidentColumns>>> =
    LazyLock::new(Default::default);

/// What starts a load of a resident store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentLoadTrigger {
    /// An index opening on the file, which loads its store before it serves
    /// any read.
    Open,
    /// A read through a storage whose store has not loaded: one built
    /// outside an index open, such as a test's, since every index loads its
    /// store when it opens. Counted in `resident_columns_read_loads`.
    Read,
}

/// The file rows a read attaches the resident rows of.
#[derive(Debug, Clone)]
pub(crate) enum ResidentRows<'a> {
    /// Every file row of a whole partition or plane.
    Range(Range<usize>),
    /// The sorted, unique file rows of a sparse gather.
    Rows(&'a UInt64Array),
}

impl ResidentRows<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Range(range) => range.len(),
            Self::Rows(rows) => rows.len(),
        }
    }
}

/// One index file's store, shared by every handle of the file and by its
/// index cache entry, and loaded by the first index of the file to open (or,
/// as a fallback, by the first read).
#[derive(Default)]
struct ResidentSlot {
    store: OnceCell<ResidentColumnStore>,
    /// The pin the handles lease the store's cache entry through.
    pin: Arc<CachePin>,
}

/// Prints the store's columns and leases, not its values.
impl std::fmt::Debug for ResidentSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentSlot")
            .field("store", &self.store.get())
            .field("leases", &self.pin.holders())
            .finish()
    }
}

impl ResidentSlot {
    /// The store of `reader`'s file, loaded by the first caller, which
    /// `trigger` says started the load. Concurrent first callers share one
    /// load, whose I/O is added to the loading caller's `io_stats`; a failed
    /// or dropped load leaves the store for the next caller to load. The
    /// load runs once per store, so it is boxed rather than inlined into
    /// every read's future.
    async fn get_or_load(
        &self,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
        trigger: ResidentLoadTrigger,
    ) -> Result<&ResidentColumnStore> {
        self.store
            .get_or_try_init(|| Box::pin(ResidentColumnStore::load(reader, io_stats, trigger)))
            .await
    }

    fn is_loaded(&self) -> bool {
        self.store.initialized()
    }
}

/// The live store of `file`, or a new one registered for the next caller,
/// and whether it was live.
fn shared_slot(file: &IndexFileKey) -> (Arc<ResidentSlot>, bool) {
    let mut created = false;
    let slot = shared_by_key(&RESIDENT_SLOTS, file, || {
        created = true;
        Arc::default()
    });
    (slot, !created)
}

/// The live stores that hold their columns.
fn live_slots() -> Vec<Arc<ResidentSlot>> {
    RESIDENT_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter_map(Weak::upgrade)
        .filter(|slot| slot.is_loaded())
        .collect()
}

/// Loaded resident stores alive in the process, cached or held by a live
/// index. 0 once every index of every file is dropped and the index caches
/// evicted or cleared the stores, unless the `process` lifetime keeps them.
pub fn resident_store_count() -> usize {
    live_slots().len()
}

/// Leases on the loaded resident stores alive in the process: one per live
/// index handle of their files (and one per store under the `process`
/// lifetime). A cached index state holds none.
pub fn resident_store_leases() -> usize {
    live_slots().iter().map(|slot| slot.pin.holders()).sum()
}

/// Whether a loaded resident store of `file` is alive in the process.
pub fn resident_store_is_live(file: &IndexFileKey) -> bool {
    RESIDENT_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(file)
        .and_then(Weak::upgrade)
        .is_some_and(|slot| slot.is_loaded())
}

/// Live views of resident store rows in the process; see
/// [`resident_store_views`].
static LIVE_STORE_VIEWS: AtomicU64 = AtomicU64::new(0);

/// Live views of resident store rows in the process: buffers that per-read
/// batches of whole partitions and planes share with a store
/// ([`ResidentAttach::Share`]), one per resident column of each. A gauge,
/// not reset by `layered_stats::snapshot_and_reset`. Each view keeps its
/// store column's whole allocation alive, even after the store's eviction,
/// so it returns to 0 once the reads that hold views have dropped them:
/// a nonzero value after every query is done means a view leaked, such as
/// into a cache entry.
pub fn resident_store_views() -> u64 {
    LIVE_STORE_VIEWS.load(Ordering::Relaxed)
}

/// The live views of one resident store's rows, which outlives the store
/// while any of them does; see [`ResidentColumns::store_views`].
#[derive(Debug, Clone, Default)]
pub struct ResidentStoreViews(Arc<AtomicU64>);

impl ResidentStoreViews {
    /// Views of the store's rows alive now: one per resident column of each
    /// batch that shares the store.
    pub fn live(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Rows of a resident store column that a view shares. It holds the
/// column's buffer, so the column's whole allocation lives until the last
/// view of it drops, and it counts itself in [`resident_store_views`] and in
/// its store's [`ResidentStoreViews`] while it lives.
struct StoreRows {
    rows: Buffer,
    store_views: ResidentStoreViews,
}

impl StoreRows {
    fn new(rows: Buffer, store_views: &ResidentStoreViews) -> Self {
        LIVE_STORE_VIEWS.fetch_add(1, Ordering::Relaxed);
        store_views.0.fetch_add(1, Ordering::Relaxed);
        Self {
            rows,
            store_views: store_views.clone(),
        }
    }
}

impl AsRef<[u8]> for StoreRows {
    fn as_ref(&self) -> &[u8] {
        self.rows.as_slice()
    }
}

impl Drop for StoreRows {
    fn drop(&mut self) {
        LIVE_STORE_VIEWS.fetch_sub(1, Ordering::Relaxed);
        self.store_views.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Whether any buffer of `batch` lies in the memory of a loaded resident
/// store alive in the process: a view of the store's rows, or a slice of a
/// store column. A cache entry must never hold such a buffer, since it
/// would keep the store column's whole allocation alive while charged its
/// rows alone; the loaders of cache entries check it in debug builds. It
/// knows the stores of the process's registry, which every index open
/// binds, and not those of storages built with a store of their own.
pub fn shares_resident_store(batch: &RecordBatch) -> bool {
    let slots = live_slots();
    let stores: Vec<&ResidentColumnStore> =
        slots.iter().filter_map(|slot| slot.store.get()).collect();
    !stores.is_empty()
        && batch
            .columns()
            .iter()
            .any(|column| array_in_stores(&column.to_data(), &stores))
}

/// Whether a buffer of `data` (or of its children) lies in the memory of
/// one of `stores`.
fn array_in_stores(data: &ArrayData, stores: &[&ResidentColumnStore]) -> bool {
    data.buffers()
        .iter()
        .chain(data.nulls().map(|nulls| nulls.inner().inner()))
        .any(|buffer| stores.iter().any(|store| store.holds(buffer)))
        || data
            .child_data()
            .iter()
            .any(|child| array_in_stores(child, stores))
}

/// The allocation `buffer` lies in, as a range of addresses.
fn allocation(buffer: &Buffer) -> Range<usize> {
    let start = buffer.data_ptr().as_ptr() as usize;
    start..start + buffer.capacity()
}

/// An index open's lease on the cached resident store of its file
/// ([`resident_store_preopen_lease`]), which also holds the store: a store
/// the cache could not pin (its lease overflowed, or the backend never pins)
/// and evicted during the open stays loaded, so the open's handle admits it
/// again rather than loading it again.
#[derive(Debug, Clone)]
pub struct ResidentPreopen {
    slot: Arc<ResidentSlot>,
    lease: CacheLease,
}

impl ResidentPreopen {
    /// The lease on the store.
    pub fn lease(&self) -> &CacheLease {
        &self.lease
    }
}

/// Lease the resident store of `file` if `cache`, the index's namespace of
/// the index cache without a fragment reuse segment, holds it in RAM. An
/// index open takes it before its first index-cache access, whose admissions
/// (a cached state promoted from a persistent tier) would otherwise find the
/// store unleased and evictable; the open then hands it to
/// [`ResidentColumns::in_index_cache`]. The lookup is a RAM-only access that
/// refreshes the store's recency and counts no miss; any other index misses.
pub async fn resident_store_preopen_lease(
    cache: &LanceCache,
    file: &IndexFileKey,
) -> Option<ResidentPreopen> {
    let (entry, lease) = cache
        .get_resident_leased_with_key(&ResidentColumnsKey::new(file))
        .await?;
    layered_stats::counters()
        .resident_columns_preopen_leases
        .incr();
    Some(ResidentPreopen {
        slot: entry.slot.clone(),
        lease,
    })
}

/// Cache key of the resident store of an IVF_RQ index file. An index keeps
/// it in its namespace of the index cache without a fragment reuse segment,
/// so that a new fragment reuse index, which changes the namespace of the
/// index's other entries, keeps the store.
pub struct ResidentColumnsKey<'a> {
    file: &'a IndexFileKey,
}

impl<'a> ResidentColumnsKey<'a> {
    /// The key of the store of `file`.
    pub fn new(file: &'a IndexFileKey) -> Self {
        Self { file }
    }
}

impl CacheKey for ResidentColumnsKey<'_> {
    type ValueType = ResidentColumnsEntry;

    fn key(&self) -> Cow<'_, str> {
        format!(
            "rq-resident-columns/{}/{}/{}",
            self.file.index_uuid(),
            self.file.store_prefix(),
            self.file.path()
        )
        .into()
    }

    fn type_name() -> &'static str {
        "IvfRqResidentColumns"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.rq-resident-columns-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.file.index_uuid());
        builder.write_str(self.file.store_prefix());
        builder.write_str(self.file.path());
    }
}

/// The index cache's entry of a loaded resident store: RAM only, charged
/// what the store's arrays allocate, and leased through the store's pin.
pub struct ResidentColumnsEntry {
    slot: Arc<ResidentSlot>,
}

impl ResidentColumnsEntry {
    fn new(slot: Arc<ResidentSlot>) -> Self {
        Self { slot }
    }

    /// Bytes of values the store holds; see [`resident_columns_bytes`].
    pub fn loaded_bytes(&self) -> Option<u64> {
        self.slot.store.get().map(|store| store.bytes)
    }
}

impl std::fmt::Debug for ResidentColumnsEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ResidentColumnsEntry")
            .field(&self.slot)
            .finish()
    }
}

impl DeepSizeOf for ResidentColumnsEntry {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        if !context.mark_seen(Arc::as_ptr(&self.slot) as usize) {
            return 0;
        }
        std::mem::size_of::<ResidentSlot>()
            + self
                .slot
                .store
                .get()
                .map_or(0, ResidentColumnStore::heap_bytes)
    }
}

impl PinnedValue for ResidentColumnsEntry {
    fn cache_pin(&self) -> &Arc<CachePin> {
        &self.slot.pin
    }
}

/// An entry drops once the index cache no longer holds it: evicted,
/// cleared or refused at admission.
impl Drop for ResidentColumnsEntry {
    fn drop(&mut self) {
        if self.slot.is_loaded() {
            layered_stats::counters().resident_store_evictions.incr();
        }
    }
}

/// Where a handle charges its store.
#[derive(Debug, Clone)]
struct ResidentCharge {
    file: IndexFileKey,
    /// The index's namespace of the index cache without a fragment reuse
    /// segment; see [`ResidentColumnsKey`].
    cache: WeakLanceCache,
    lifetime: ResidentLifetime,
}

/// A storage's handle on the resident columns of its file, which the index
/// loads when it opens. A handle bound to an index cache
/// ([`in_index_cache`](Self::in_index_cache)) shares the store with every
/// live handle of the file, admits it to the cache and leases it once it is
/// loaded, until the handle (and so its index) drops; clones are holders
/// too. A default handle holds a store of its own, charged nowhere.
#[derive(Debug, Clone, Default)]
pub struct ResidentColumns {
    slot: Arc<ResidentSlot>,
    /// `None` for a store charged nowhere.
    charge: Option<ResidentCharge>,
    lease: OnceLock<CacheLease>,
}

impl ResidentColumns {
    /// The store of index file `file` that every live handle of it shares,
    /// charged nowhere, for an index opened without an index cache.
    pub fn shared(file: &IndexFileKey) -> Self {
        Self {
            slot: shared_slot(file).0,
            charge: None,
            lease: OnceLock::new(),
        }
    }

    /// The handle of an index opening index file `file` with its small
    /// columns resident: the store every live handle of the file shares,
    /// charged in `cache`, the index's namespace of the index cache without
    /// a fragment reuse segment. `preopen` is the lease the open took on the
    /// cached store ([`resident_store_preopen_lease`]), whose lease the
    /// handle keeps and whose store it binds, loaded even if the cache
    /// evicted it since. A store already loaded is leased now and admitted
    /// again if the cache lost it; otherwise the opening index loads, admits
    /// and leases it next
    /// ([`IvfQuantizationStorage::load_resident_store`](crate::vector::storage::IvfQuantizationStorage::load_resident_store)).
    /// Under [`ResidentLifetime::Process`] the store also keeps a lease for
    /// the life of the process once leased.
    pub async fn in_index_cache(
        cache: &LanceCache,
        file: &IndexFileKey,
        preopen: Option<ResidentPreopen>,
        lifetime: ResidentLifetime,
    ) -> Self {
        // The pre-open lease's store is the file's live store, which the
        // registry would hand out too, kept alive even if the cache evicted
        // it during the open.
        let (slot, live) = match &preopen {
            Some(preopen) => (preopen.slot.clone(), true),
            None => shared_slot(file),
        };
        let stats = layered_stats::counters();
        stats.resident_columns_binds.incr();
        if live && slot.is_loaded() {
            stats.resident_columns_registry_reuses.incr();
        }
        let handle = Self {
            slot,
            charge: Some(ResidentCharge {
                file: file.clone(),
                cache: WeakLanceCache::from(cache),
                lifetime,
            }),
            lease: OnceLock::new(),
        };
        if let Some(preopen) = preopen {
            handle.adopt(preopen.lease);
        }
        handle.ensure_charged().await;
        handle
    }

    /// Keep `lease` as this handle's lease when it leases this handle's
    /// store and the handle holds none yet; returns whether it kept it.
    pub fn adopt(&self, lease: CacheLease) -> bool {
        Arc::ptr_eq(lease.pin(), &self.slot.pin) && self.lease.set(lease).is_ok()
    }

    /// Lease the loaded store and make sure the index cache holds it,
    /// admitting it again if the cache lost it: evicted while overflowed,
    /// cleared, refused, or held only by a backend that never pins. The
    /// check refreshes the store's recency. Returns whether it admitted the
    /// store again; a store not loaded yet, or a handle charged nowhere, is
    /// left alone.
    pub async fn ensure_charged(&self) -> bool {
        let Some(charge) = &self.charge else {
            return false;
        };
        if !self.slot.is_loaded() {
            return false;
        }
        self.lease.get_or_init(|| CachePin::lease(&self.slot.pin));
        let slot = &self.slot;
        let recharged = charge
            .cache
            .ensure_pinned_with_key(&ResidentColumnsKey::new(&charge.file), || {
                Arc::new(ResidentColumnsEntry::new(slot.clone()))
            })
            .await;
        if recharged {
            layered_stats::counters().resident_columns_recharges.incr();
        }
        self.leased(charge);
        recharged
    }

    /// Bytes of values the store holds, `None` until it has loaded. Equals
    /// [`resident_columns_bytes`] of its file.
    pub fn loaded_bytes(&self) -> Option<u64> {
        self.slot.store.get().map(|store| store.bytes)
    }

    /// The live views of the store's rows (see [`resident_store_views`]),
    /// `None` until it has loaded. The count outlives the store: views keep
    /// the store's memory, not the store.
    pub fn store_views(&self) -> Option<ResidentStoreViews> {
        self.slot.store.get().map(|store| store.views.clone())
    }

    /// The lease this handle holds on its store, once the store is loaded
    /// and charged.
    pub fn lease(&self) -> Option<&CacheLease> {
        self.lease.get()
    }

    /// The store of `reader`'s file, loaded by the first caller if nothing
    /// has loaded it, with `trigger` saying what started the load: see
    /// [`ResidentSlot::get_or_load`]. A handle bound to an index cache
    /// that holds no lease yet admits the store and leases it first.
    pub(crate) async fn get_or_load(
        &self,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
        trigger: ResidentLoadTrigger,
    ) -> Result<&ResidentColumnStore> {
        if self.lease.get().is_none()
            && let Some(charge) = &self.charge
        {
            self.charge_on_load(charge, reader, io_stats, trigger)
                .await?;
        }
        self.slot.get_or_load(reader, io_stats, trigger).await
    }

    /// Load the store if nothing has, admit it leased so that it enters the
    /// cache pinned, and keep the lease; a store the cache holds is leased
    /// as found. A dropped cache leaves the store charged nowhere.
    async fn charge_on_load(
        &self,
        charge: &ResidentCharge,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
        trigger: ResidentLoadTrigger,
    ) -> Result<()> {
        let slot = self.slot.clone();
        let loaded = charge
            .cache
            .get_or_insert_leased_with_key(ResidentColumnsKey::new(&charge.file), || async move {
                slot.get_or_load(reader, io_stats, trigger).await?;
                Ok(ResidentColumnsEntry::new(slot))
            })
            .await?;
        if let Some((_, lease, _)) = loaded {
            self.adopt(lease);
            self.leased(charge);
        }
        Ok(())
    }

    /// Account for this handle's lease: count it when the cache's pinned
    /// cap left the store evictable, and under the `process` lifetime keep a
    /// handle on the store until the process exits.
    fn leased(&self, charge: &ResidentCharge) {
        let Some(lease) = self.lease.get() else {
            return;
        };
        if lease.pin().is_overflowed() {
            layered_stats::counters().pinned_overflow.incr();
        }
        if charge.lifetime == ResidentLifetime::Process {
            PROCESS_STORES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(charge.file.clone())
                .or_insert_with(|| self.clone());
        }
    }
}

/// Every row of the resident columns of one file.
pub(crate) struct ResidentColumnStore {
    columns: HashMap<String, ResidentColumn>,
    num_rows: u64,
    /// See [`ResidentColumns::loaded_bytes`].
    bytes: u64,
    /// See [`ResidentColumns::store_views`].
    views: ResidentStoreViews,
}

/// Names the columns instead of printing hundreds of megabytes of values.
impl std::fmt::Debug for ResidentColumnStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut columns: Vec<&String> = self.columns.keys().collect();
        columns.sort();
        f.debug_struct("ResidentColumnStore")
            .field("columns", &columns)
            .field("num_rows", &self.num_rows)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl ResidentColumnStore {
    async fn load(
        reader: &FileReader,
        io_stats: Option<&IoStats>,
        trigger: ResidentLoadTrigger,
    ) -> Result<Self> {
        let started = Instant::now();
        let schema = Schema::from(reader.schema().as_ref());
        let fields: Vec<(&Field, usize)> = schema
            .fields()
            .iter()
            .filter_map(|field| Some((field.as_ref(), resident_width(field)?)))
            .collect();
        let num_rows = reader.num_rows();
        let load_stats = IoStats::new();
        let batch = if fields.is_empty() || num_rows == 0 {
            None
        } else {
            let names: Vec<&str> = fields
                .iter()
                .map(|(field, _)| field.name().as_str())
                .collect();
            let projection = lance_file::versions::reader_projection_from_column_names(
                reader.metadata().version(),
                reader.schema(),
                &names,
            )?;
            let projected = Arc::new(Schema::from(projection.schema.as_ref()));
            let stats_reader = reader.with_io_stats(load_stats.recorder());
            let batches = stats_reader
                .read_stream_projected(
                    ReadBatchParams::Range(0..num_rows as usize),
                    u32::MAX,
                    1,
                    projection,
                    FilterExpression::no_filter(),
                )
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            // Buffers of exactly the values, so the cache charges the store
            // what `resident_columns_bytes` sizes it at without reading it.
            Some(exact_batch(&projected, &batches, false)?)
        };
        let mut columns = HashMap::with_capacity(fields.len());
        let mut bytes = 0u64;
        // What the arrays hold, which a zero-copy slice of a larger decoded
        // or read buffer would push past `bytes`.
        let mut alloc_bytes = 0u64;
        for (field, width) in fields {
            let name = field.name();
            let values = match &batch {
                Some(batch) => batch.column_by_name(name).cloned().ok_or_else(|| {
                    Error::internal(format!("resident column {name} is missing from its read"))
                })?,
                None => new_empty_array(field.data_type()),
            };
            if values.len() as u64 != num_rows {
                return Err(Error::internal(format!(
                    "resident column {name} read {} of the file's {num_rows} rows",
                    values.len()
                )));
            }
            bytes += width as u64 * num_rows;
            alloc_bytes += values.get_buffer_memory_size() as u64;
            let page_ends = column_page_ends(reader, name)?;
            columns.insert(name.clone(), ResidentColumn { values, page_ends });
        }
        let loaded = load_stats.snapshot();
        let stats = layered_stats::counters();
        stats.resident_columns_loads.incr();
        if trigger == ResidentLoadTrigger::Read {
            stats.resident_columns_read_loads.incr();
        }
        stats.resident_columns_bytes.add(bytes);
        stats.resident_columns_alloc_bytes.add(alloc_bytes);
        stats.resident_columns_load_requests.add(loaded.iops);
        stats.resident_columns_load_bytes.add(loaded.bytes_read);
        stats.resident_columns_load_ns.add_elapsed(started);
        if let Some(io_stats) = io_stats {
            io_stats.add_scan_stats(&loaded);
        }
        Ok(Self {
            columns,
            num_rows,
            bytes,
            views: ResidentStoreViews::default(),
        })
    }

    /// Resident column `name`, `None` for a column reads fetch from the file.
    pub(crate) fn column(&self, name: &str) -> Option<&ResidentColumn> {
        self.columns.get(name)
    }

    /// Rows of the file.
    pub(crate) fn num_rows(&self) -> u64 {
        self.num_rows
    }

    /// `schema`'s columns at the file rows `rows`: this store's rows of the
    /// columns it keeps and, by name, the other columns of `file_batch`,
    /// which holds them at the same rows, read from the file or a code-only
    /// cache entry. The batch is the one a read of every column from the
    /// file returns, bit for bit and byte for byte. `mode` says how a whole
    /// partition or plane ([`ResidentRows::Range`]) takes the store's rows:
    /// views of its buffers ([`ResidentColumn::share_rows`]) or copies;
    /// sparse rows ([`ResidentRows::Rows`]) are always copied
    /// ([`ResidentColumn::copy_rows`]). A batch with views must never be
    /// cached. The mode has no default, so every caller states it.
    pub(crate) fn attach(
        &self,
        schema: SchemaRef,
        file_batch: Option<&RecordBatch>,
        rows: ResidentRows<'_>,
        mode: ResidentAttach,
    ) -> Result<RecordBatch> {
        let num_rows = rows.len();
        if let Some(batch) = file_batch
            && batch.num_rows() != num_rows
        {
            return Err(Error::internal(format!(
                "resident columns attach to {num_rows} rows of a batch of {} rows",
                batch.num_rows()
            )));
        }
        if let ResidentRows::Range(range) = &rows
            && (range.start > range.end || range.end as u64 > self.num_rows)
        {
            return Err(Error::invalid_input(format!(
                "resident rows {range:?} are not within the {} rows of the file",
                self.num_rows
            )));
        }
        // Resolve the rows before the timer: a copy of a whole range copies
        // the rows of a `UInt64Array` built here, as callers built it before
        // reads could share the store, so `resident_attach_ns` times the
        // same work in `copy` mode.
        let whole = matches!(rows, ResidentRows::Range(_));
        // What each resident column takes: a range to view, or rows to copy.
        let range_rows;
        let taken = match (rows, mode) {
            (ResidentRows::Range(range), ResidentAttach::Share) => ResidentRows::Range(range),
            (ResidentRows::Range(range), ResidentAttach::Copy) => {
                range_rows = UInt64Array::from_iter_values(range.start as u64..range.end as u64);
                ResidentRows::Rows(&range_rows)
            }
            (ResidentRows::Rows(rows), _) => ResidentRows::Rows(rows),
        };
        let started = Instant::now();
        let mut attached_bytes = 0u64;
        let mut copied_bytes = 0u64;
        let mut every_column_shared = true;
        let mut columns = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let column = match self.column(field.name()) {
                Some(column) => {
                    let (array, shared) = match &taken {
                        ResidentRows::Range(range) => {
                            column.share_rows(range.clone(), &self.views)?
                        }
                        ResidentRows::Rows(rows) => (column.copy_rows(rows)?, false),
                    };
                    let bytes = column.row_bytes(num_rows);
                    attached_bytes += bytes;
                    if !shared {
                        copied_bytes += bytes;
                        every_column_shared = false;
                    }
                    array
                }
                None => file_batch
                    .and_then(|batch| batch.column_by_name(field.name()))
                    .cloned()
                    .ok_or_else(|| Error::internal(format!("unread column {}", field.name())))?,
            };
            columns.push(column);
        }
        let batch = RecordBatch::try_new_with_options(
            schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(num_rows)),
        )?;
        let stats = layered_stats::counters();
        stats.resident_attach_calls.incr();
        stats.resident_attach_rows.add(num_rows as u64);
        stats.resident_attach_bytes.add(attached_bytes);
        stats.resident_attach_copied_bytes.add(copied_bytes);
        match (whole, every_column_shared) {
            (true, true) => stats.resident_attach_shares.incr(),
            (true, false) => stats.resident_attach_whole_copies.incr(),
            (false, _) => stats.resident_attach_gathers.incr(),
        }
        stats.resident_attach_ns.add_elapsed(started);
        Ok(batch)
    }

    /// Whether `buffer`'s bytes lie in the allocation of one of this
    /// store's columns.
    fn holds(&self, buffer: &Buffer) -> bool {
        let start = buffer.as_ptr() as usize;
        let bytes = start..start + buffer.len();
        !bytes.is_empty()
            && self.columns.values().any(|column| {
                column.values.to_data().buffers().iter().any(|own| {
                    let own = allocation(own);
                    bytes.start < own.end && own.start < bytes.end
                })
            })
    }

    /// Heap memory the store holds: its arrays' buffers, which the
    /// `resident_columns_alloc_bytes` of its load count, plus its pages and
    /// its map of columns.
    fn heap_bytes(&self) -> usize {
        self.columns.capacity() * std::mem::size_of::<(String, ResidentColumn)>()
            + self
                .columns
                .iter()
                .map(|(name, column)| {
                    name.capacity()
                        + column.values.get_buffer_memory_size()
                        + column.page_ends.capacity() * std::mem::size_of::<u64>()
                })
                .sum::<usize>()
    }
}

/// The pages of column `name` of `reader`'s file.
fn column_pages<'a>(reader: &'a FileReader, name: &str) -> Result<&'a [PageInfo]> {
    let metadata = reader.metadata();
    let projection = lance_file::versions::reader_projection_from_column_names(
        metadata.version(),
        reader.schema(),
        &[name],
    )?;
    match projection.column_indices.as_slice() {
        [column] => metadata.column_infos.get(*column as usize),
        _ => None,
    }
    .map(|column| column.page_infos.as_ref())
    .ok_or_else(|| {
        Error::internal(format!(
            "resident column {name} is not one column of the file: {:?}",
            projection.column_indices
        ))
    })
}

/// The row at which each page of column `name` of `reader`'s file ends, one
/// per page in a vector of exactly that capacity, as
/// [`resident_store_charge`] counts them.
fn column_page_ends(reader: &FileReader, name: &str) -> Result<Vec<u64>> {
    let pages = column_pages(reader, name)?;
    let mut page_ends = Vec::with_capacity(pages.len());
    let mut end = 0u64;
    for page in pages {
        end += page.num_rows;
        page_ends.push(end);
    }
    Ok(page_ends)
}

/// Bytes an index cache charges for the resident store of `reader`'s file
/// once loaded, known without reading it: what its cache entry
/// ([`ResidentColumnsEntry`]) takes, which is the store's columns in buffers
/// of exactly their values ([`resident_columns_bytes`]), their page ends,
/// names and map, the shared slot and the entry's `Arc`, and the cost a
/// backend adds for the entry's key. An open compares it with the cache's
/// pinned cap before keeping a store resident, as a lease compares the
/// charged entry with it.
pub fn resident_store_charge(reader: &FileReader) -> Result<u64> {
    let schema = Schema::from(reader.schema().as_ref());
    let columns = schema
        .fields()
        .iter()
        .filter_map(|field| Some((field.name(), resident_width(field)?)))
        .map(|(name, width)| Ok((name.as_str(), width, column_pages(reader, name)?.len())))
        .collect::<Result<Vec<_>>>()?;
    Ok(store_charge(&columns, reader.num_rows()))
}

/// [`resident_store_charge`] of a store of `num_rows` rows of `columns`,
/// each a name, a value width and a number of pages.
fn store_charge(columns: &[(&str, usize, usize)], num_rows: u64) -> u64 {
    let map_bytes = HashMap::<String, ResidentColumn>::with_capacity(columns.len()).capacity()
        * std::mem::size_of::<(String, ResidentColumn)>();
    let column_bytes: u64 = columns
        .iter()
        .map(|&(name, width, pages)| {
            (name.len() + pages * std::mem::size_of::<u64>()) as u64 + width as u64 * num_rows
        })
        .sum();
    let fixed_bytes = std::mem::size_of::<ResidentColumnsEntry>()
        + std::mem::size_of::<ResidentSlot>()
        + 2 * std::mem::size_of::<std::sync::atomic::AtomicUsize>()
        + std::mem::size_of::<InternalCacheKey>();
    (map_bytes + fixed_bytes) as u64 + column_bytes
}

/// Every row of one resident column, with the column's pages in the file.
pub(crate) struct ResidentColumn {
    values: ArrayRef,
    /// The row at which each page of the column ends, ascending.
    page_ends: Vec<u64>,
}

impl ResidentColumn {
    /// Bytes of `rows` rows of the column's values, which have a fixed width
    /// (see [`resident_width`]).
    fn row_bytes(&self, rows: usize) -> u64 {
        let width = self
            .values
            .data_type()
            .primitive_width()
            .unwrap_or_default();
        (rows * width) as u64
    }

    /// The file rows `range` as a view of the column's buffer: an array
    /// whose one buffer is exactly those rows' bytes, which it shares with
    /// the column rather than copies, so it is charged what a copy is
    /// (an `ArrayRef::slice` would be charged the whole column). The view
    /// keeps the column's whole allocation alive until it drops, and counts
    /// itself in [`resident_store_views`] and in `store_views` meanwhile.
    /// Returns whether the array is a view: a column a view cannot cover
    /// (with a validity buffer, or more than one buffer) is copied as
    /// [`Self::copy_rows`] copies, and so is a buffer Arrow refuses for the
    /// column's type. An empty range is an empty array, which is no copy.
    pub(crate) fn share_rows(
        &self,
        range: Range<usize>,
        store_views: &ResidentStoreViews,
    ) -> Result<(ArrayRef, bool)> {
        if range.start > range.end || range.end > self.values.len() {
            return Err(Error::invalid_input(format!(
                "resident rows {range:?} are not within the {} rows of the file",
                self.values.len()
            )));
        }
        if range.is_empty() {
            return Ok((new_empty_array(self.values.data_type()), true));
        }
        let data = self.values.to_data();
        if let (None, [values], Some(width)) = (
            data.nulls(),
            data.buffers(),
            data.data_type().primitive_width(),
        ) {
            // The store's buffers come from `exact_batch`, aligned for their
            // type, so the rows' first byte keeps that alignment.
            let rows = values
                .slice_with_length((data.offset() + range.start) * width, range.len() * width);
            let view = Buffer::from(bytes::Bytes::from_owner(StoreRows::new(rows, store_views)));
            if let Ok(view) = ArrayData::builder(data.data_type().clone())
                .len(range.len())
                .add_buffer(view)
                .build()
            {
                return Ok((make_array(view), true));
            }
        }
        let rows = UInt64Array::from_iter_values(range.start as u64..range.end as u64);
        Ok((self.copy_rows(&rows)?, false))
    }

    /// A copy of the ascending file rows `rows` in a buffer of exactly their
    /// values, as a whole read of the file returns them (see
    /// `read_projected`), so that a cache entry holding it is charged the
    /// same bytes, and so that it never keeps the store's memory alive: what
    /// a sparse gather, a whole read that becomes a cache entry and every
    /// attach under [`ResidentAttach::Copy`] take. A slice would instead
    /// keep, and be charged, the whole column.
    pub(crate) fn copy_rows(&self, rows: &UInt64Array) -> Result<ArrayRef> {
        let offsets = rows.values();
        // The page split below relies on the order.
        if !offsets.is_sorted() {
            return Err(Error::invalid_input(
                "rows read from resident columns must ascend",
            ));
        }
        if let Some(&last) = offsets.last()
            && last >= self.values.len() as u64
        {
            return Err(Error::invalid_input(format!(
                "resident row {last} is past the {} rows of the file",
                self.values.len()
            )));
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut page = 0;
        while start < offsets.len() {
            let first = offsets[start];
            page += self.page_ends[page..].partition_point(|&end| end <= first);
            let page_end = self.page_ends.get(page).copied().unwrap_or(u64::MAX);
            let len = offsets[start..].partition_point(|&row| row < page_end);
            let piece = take(self.values.as_ref(), &rows.slice(start, len), None)?;
            pieces.push(piece);
            start += len;
        }
        match pieces.len() {
            0 => Ok(new_empty_array(self.values.data_type())),
            1 => Ok(pieces.swap_remove(0)),
            _ => exact_array(&pieces.iter().collect::<Vec<_>>(), false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, cast::AsArray, types::Float32Type};
    use arrow_schema::DataType;
    use lance_core::cache::QuickCacheBackend;
    use lance_encoding::decoder::DecoderPlugins;
    use lance_file::reader::FileReaderOptions;
    use lance_file::version::LanceFileVersion;
    use lance_file::writer::FileWriterOptions;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;

    use crate::vector::bq::layered::{HIGH_ADD_FACTORS_COLUMN, HIGH_SCALE_FACTORS_COLUMN};
    use crate::vector::bq::transform::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
    };
    use crate::vector::storage::{ResidentColumnsSetting, ResidentStoreSize};

    fn list_field(name: &str, item: DataType, width: i32) -> Field {
        Field::new(
            name,
            DataType::FixedSizeList(Arc::new(Field::new("item", item, true)), width),
            true,
        )
    }

    fn factor_field(name: &str) -> Field {
        Field::new(name, DataType::Float32, true)
    }

    #[test]
    fn resident_columns_bytes_counts_the_small_fixed_width_columns() {
        let native = vec![
            Field::new(lance_core::ROW_ID, DataType::UInt64, true),
            list_field(RABIT_CODE_COLUMN, DataType::UInt8, 16),
            factor_field(ADD_FACTORS_COLUMN),
            factor_field(SCALE_FACTORS_COLUMN),
            factor_field(ERROR_FACTORS_COLUMN),
            list_field(RABIT_BLOCKED_EX_CODE_COLUMN, DataType::UInt8, 96),
            factor_field(EX_ADD_FACTORS_COLUMN),
            factor_field(EX_SCALE_FACTORS_COLUMN),
        ];
        let mut layered = native.clone();
        layered.extend([
            list_field(RABIT_BLOCKED_EX_CODE_LO_COLUMN, DataType::UInt8, 64),
            factor_field(HIGH_ADD_FACTORS_COLUMN),
            factor_field(HIGH_SCALE_FACTORS_COLUMN),
            list_field(HIGH_BOUNDS_COLUMN, DataType::Float32, 3),
            list_field(FULL_BOUNDS_COLUMN, DataType::Float32, 3),
        ]);
        // Row ids (8 bytes) and four-byte factors: 3 for the sign codes and
        // 2 for each ex level.
        assert_eq!(
            resident_columns_bytes(&Schema::new(native.clone()), 1000),
            28_000
        );
        assert_eq!(resident_columns_bytes(&Schema::new(layered), 1000), 36_000);
        assert_eq!(resident_columns_bytes(&Schema::new(native.clone()), 0), 0);
        // Legacy ex codes and columns without a fixed width stay in the file.
        let mut other = native;
        other.push(list_field(RABIT_EX_CODE_COLUMN, DataType::UInt8, 96));
        other.push(Field::new("label", DataType::Utf8, true));
        assert_eq!(resident_columns_bytes(&Schema::new(other), 10), 280);
    }

    /// A file key unique to one test, so tests that share the process-wide
    /// registry do not share stores.
    fn test_file(test: &str) -> IndexFileKey {
        let path = format!("t.lance/_indices/{test}/auxiliary.idx");
        IndexFileKey::new(test, "s3$bucket", &path)
    }

    /// A loaded store of `rows` rows of one four-byte column in one page.
    fn test_store(rows: usize) -> ResidentColumnStore {
        let values: ArrayRef = Arc::new(Float32Array::from(vec![1.0; rows]));
        let column = ResidentColumn {
            values,
            page_ends: vec![rows as u64],
        };
        ResidentColumnStore {
            columns: HashMap::from([("factor".to_string(), column)]),
            num_rows: rows as u64,
            bytes: 4 * rows as u64,
            views: ResidentStoreViews::default(),
        }
    }

    /// The slot of `file`, loaded with a test store, and held by the caller.
    fn loaded_slot(file: &IndexFileKey, rows: usize) -> Arc<ResidentSlot> {
        let (slot, _) = shared_slot(file);
        slot.store.set(test_store(rows)).unwrap();
        slot
    }

    fn pinning_cache() -> LanceCache {
        LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(1 << 20)))
    }

    /// Handles of one file share its store while any holds it, and a store
    /// no handle or cache holds is freed, so the next handle gets a new one;
    /// the same path in another bucket is another file.
    #[test]
    fn slot_registry_shares_live_store_and_frees_dropped() {
        let file = test_file("slot-registry");
        let store = ResidentColumns::shared(&file);
        let shared = ResidentColumns::shared(&file);
        assert!(Arc::ptr_eq(&store.slot, &shared.slot));
        let path = file.path().to_string();
        let other = IndexFileKey::new(file.index_uuid(), "s3$other", &path);
        assert!(!Arc::ptr_eq(
            &store.slot,
            &ResidentColumns::shared(&other).slot
        ));
        store.slot.store.set(test_store(10)).unwrap();
        assert!(resident_store_is_live(&file));
        let held = Arc::downgrade(&store.slot);
        drop((store, shared));
        assert!(held.upgrade().is_none());
        assert!(!resident_store_is_live(&file));
        let reopened = ResidentColumns::shared(&file);
        assert_eq!(reopened.loaded_bytes(), None);
    }

    /// The cache entry of a store is charged its heap memory: what its
    /// arrays allocate, and less than 64 KiB more.
    #[test]
    fn store_deep_size_matches_alloc() {
        const OVERHEAD_BYTES: usize = 64 * 1024;
        let slot = Arc::new(ResidentSlot::default());
        let rows = 10_000;
        slot.store.set(test_store(rows)).unwrap();
        let entry = ResidentColumnsEntry::new(slot.clone());
        let alloc = rows * 4;
        let size = entry.deep_size_of();
        assert!(size >= alloc, "{size} < {alloc}");
        assert!(size <= alloc + OVERHEAD_BYTES, "{size}");
        // Two entries of one store are counted once.
        let twin = ResidentColumnsEntry::new(slot);
        let mut context = Context::new();
        let both =
            entry.deep_size_of_children(&mut context) + twin.deep_size_of_children(&mut context);
        assert_eq!(both, entry.deep_size_of_children(&mut Context::new()));
    }

    /// A handle bound to an index cache leases and admits a loaded store,
    /// and releases it when dropped: the idle store stays cached until
    /// evicted. The next open's pre-open lease is adopted, and a store the
    /// cache lost is admitted again.
    #[tokio::test]
    async fn in_index_cache_leases_and_charges_a_loaded_store() {
        let file = test_file("in-index-cache");
        let slot = loaded_slot(&file, 1000);
        let cache = pinning_cache();
        let key = ResidentColumnsKey::new(&file);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        assert!(handle.lease().is_some_and(CacheLease::is_pinned));
        assert!(cache.get_resident_with_key(&key).await.is_some());
        assert_eq!(cache.pinned_stats().pinned_entries, 1);
        assert_eq!(slot.pin.holders(), 1);
        let clone = handle.clone();
        assert_eq!(slot.pin.holders(), 2);
        drop((handle, clone));
        assert_eq!(slot.pin.holders(), 0);
        assert_eq!(cache.pinned_stats().pinned_entries, 0);
        assert!(cache.peek_resident_with_key(&key).await);

        let preopen = resident_store_preopen_lease(&cache, &file).await.unwrap();
        let preopen_pin = preopen.lease().pin().clone();
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, Some(preopen), ResidentLifetime::Index)
                .await;
        assert!(Arc::ptr_eq(handle.lease().unwrap().pin(), &preopen_pin));
        assert_eq!(slot.pin.holders(), 1);

        cache.clear().await;
        assert!(!handle.lease().unwrap().is_pinned());
        assert!(handle.ensure_charged().await);
        assert!(handle.lease().unwrap().is_pinned());
        assert!(!handle.ensure_charged().await);
        // Another file's cached store is not this one.
        assert!(
            resident_store_preopen_lease(&cache, &test_file("in-index-cache-other"))
                .await
                .is_none()
        );
    }

    /// A store of `rows` rows of `columns`, each a name and a value width
    /// of 4 or 8 bytes, in `pages` pages, laid out as a load lays it out.
    fn store_of(columns: &[(&str, usize)], rows: usize, pages: usize) -> ResidentColumnStore {
        let mut map = HashMap::with_capacity(columns.len());
        let mut bytes = 0;
        for &(name, width) in columns {
            let values: ArrayRef = match width {
                4 => Arc::new(Float32Array::from(vec![1.0; rows])),
                _ => Arc::new(UInt64Array::from(vec![1; rows])),
            };
            let mut page_ends = Vec::with_capacity(pages);
            page_ends.extend((1..=pages).map(|page| (rows * page / pages) as u64));
            map.insert(name.to_string(), ResidentColumn { values, page_ends });
            bytes += (width * rows) as u64;
        }
        ResidentColumnStore {
            columns: map,
            num_rows: rows as u64,
            bytes,
            views: ResidentStoreViews::default(),
        }
    }

    /// The slot of `file`, loaded with `store`, and held by the caller.
    fn slot_with(file: &IndexFileKey, store: ResidentColumnStore) -> Arc<ResidentSlot> {
        let (slot, _) = shared_slot(file);
        slot.store.set(store).unwrap();
        slot
    }

    const CHARGE_TEST_COLUMNS: [(&str, usize); 3] =
        [("_rowid", 8), ("__add_factors", 4), ("__scale_factors", 4)];

    /// What [`resident_store_charge`] estimates without reading the store
    /// is what a pinning cache charges for it once loaded.
    #[tokio::test]
    async fn store_charge_is_what_the_cache_charges() {
        for (rows, pages) in [(1, 1), (1000, 1), (10_000, 7), (100_000, 64)] {
            let columns: Vec<(&str, usize, usize)> = CHARGE_TEST_COLUMNS
                .iter()
                .map(|&(name, width)| (name, width, pages))
                .collect();
            let charge = store_charge(&columns, rows as u64);
            let file = test_file(&format!("store-charge-{rows}-{pages}"));
            let _slot = slot_with(&file, store_of(&CHARGE_TEST_COLUMNS, rows, pages));
            let cache =
                LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(64 << 20)));
            let handle =
                ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
            assert!(handle.lease().is_some_and(CacheLease::is_pinned));
            assert_eq!(cache.pinned_stats().pinned_bytes, charge, "{rows} {pages}");
        }
    }

    /// `auto` keeps a store resident exactly when the cache pins it: where
    /// the store's charge fits the pinned cap, a lease pins it, and a cache
    /// a byte smaller resolves it off rather than leaving it to overflow, as
    /// it would at a cap that fits its values but not its charge.
    #[tokio::test]
    async fn auto_admits_a_store_where_the_cache_pins_it() {
        let rows = 10_000;
        let pages = 3;
        let columns: Vec<(&str, usize, usize)> = CHARGE_TEST_COLUMNS
            .iter()
            .map(|&(name, width)| (name, width, pages))
            .collect();
        let store = ResidentStoreSize {
            bytes: 16 * rows as u64,
            charge: store_charge(&columns, rows as u64),
        };
        assert!(store.charge > store.bytes);
        let (bytes, charge) = (store.bytes as usize, store.charge as usize);
        for capacity in [
            2 * charge - 2,
            2 * charge - 1,
            2 * charge,
            2 * charge + 1,
            2 * bytes,
            2 * bytes + 1,
        ] {
            let cache =
                LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(capacity)));
            assert_eq!(cache.max_entry_bytes(), Some(capacity as u64));
            let has_pin_budget = cache.pinned_stats().cap_bytes > 0;
            assert!(has_pin_budget, "{capacity}");
            let admitted =
                ResidentColumnsSetting::Auto.admits(store, cache.max_entry_bytes(), has_pin_budget);
            assert_eq!(admitted, capacity >= 2 * charge, "{capacity}");
            // Lease the store as an index that keeps it resident does.
            let file = test_file(&format!("auto-admits-{capacity}"));
            let _slot = slot_with(&file, store_of(&CHARGE_TEST_COLUMNS, rows, pages));
            let handle =
                ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
            let pinned = handle.lease().is_some_and(CacheLease::is_pinned);
            assert_eq!(pinned, admitted, "{capacity}");
            assert_eq!(cache.pinned_stats().overflow > 0, !admitted, "{capacity}");
        }
    }

    /// A pre-open lease holds its store: when the cache drops the store's
    /// entry between the lease and the bind, as an open's admissions can
    /// evict a store the cache could not pin, the bound handle gets the
    /// loaded store and admits it again rather than loading it again.
    #[tokio::test]
    async fn preopen_lease_keeps_a_store_evicted_during_the_open() {
        let file = test_file("preopen-evicted");
        let cache = pinning_cache();
        let key = ResidentColumnsKey::new(&file);
        // A loaded store, idle in the cache, which alone holds it.
        let slot = loaded_slot(&file, 1000);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        let held = Arc::downgrade(&slot);
        drop((handle, slot));
        assert!(held.upgrade().is_some());

        let preopen = resident_store_preopen_lease(&cache, &file).await.unwrap();
        cache.clear().await;
        assert!(!cache.peek_resident_with_key(&key).await);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, Some(preopen), ResidentLifetime::Index)
                .await;
        assert_eq!(handle.loaded_bytes(), Some(4000));
        assert!(Arc::ptr_eq(&handle.slot, &held.upgrade().unwrap()));
        assert!(cache.peek_resident_with_key(&key).await);
        assert!(handle.lease().is_some_and(CacheLease::is_pinned));
        assert_eq!(handle.slot.pin.holders(), 1);
    }

    /// Under the `process` lifetime a store keeps a lease once its first
    /// handle leased it: pinned after every handle dropped, and alive after
    /// the cache lost it, so the next open admits it again rather than
    /// loading it.
    #[tokio::test]
    async fn resident_lifetime_process_keeps_store_pinned() {
        let file = test_file("process-lifetime");
        let slot = loaded_slot(&file, 1000);
        let cache = pinning_cache();
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Process).await;
        assert_eq!(slot.pin.holders(), 2);
        drop(handle);
        let held = Arc::downgrade(&slot);
        drop(slot);
        let slot = held.upgrade().expect("the process keeps the store");
        assert_eq!(slot.pin.holders(), 1);
        assert!(slot.pin.is_pinned());
        cache.clear().await;
        assert!(resident_store_is_live(&file));
        let reopened =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Process).await;
        assert!(Arc::ptr_eq(&reopened.slot, &slot));
        assert!(slot.pin.is_pinned());
        assert_eq!(slot.pin.holders(), 2);
    }

    /// A 100-row column in pages ending at rows 40, 40 (an empty page) and 100.
    fn column() -> ResidentColumn {
        let values: Vec<f32> = (0..100).map(|row| row as f32).collect();
        ResidentColumn {
            values: Arc::new(Float32Array::from(values)),
            page_ends: vec![40, 40, 100],
        }
    }

    fn copy(column: &ResidentColumn, rows: &[u64]) -> ArrayRef {
        column.copy_rows(&UInt64Array::from(rows.to_vec())).unwrap()
    }

    fn values(array: &ArrayRef) -> Vec<f32> {
        array.as_primitive::<Float32Type>().values().to_vec()
    }

    fn as_values(rows: &[u64]) -> Vec<f32> {
        rows.iter().map(|&row| row as f32).collect()
    }

    #[test]
    fn copy_rows_takes_the_rows_into_their_own_buffers() {
        let column = column();
        let store = column.values.as_primitive::<Float32Type>();
        let store_values = store.values().as_ptr();
        let rows: Vec<u64> = (10..30).collect();
        let copied = copy(&column, &rows);
        assert_eq!(values(&copied), as_values(&rows));
        assert!(copied.nulls().is_none());
        // One page: a buffer of exactly the rows, apart from the store's.
        assert_eq!(copied.get_buffer_memory_size(), rows.len() * 4);
        let copied_values = copied.as_primitive::<Float32Type>().values().as_ptr();
        assert_ne!(copied_values, store_values);

        let sparse = copy(&column, &[3, 7, 8, 39]);
        assert_eq!(values(&sparse), as_values(&[3, 7, 8, 39]));
        assert_eq!(sparse.get_buffer_memory_size(), 4 * 4);
        assert_eq!(copy(&column, &[]).len(), 0);
        for rows in [vec![5, 100], vec![7, 5]] {
            let error = column.copy_rows(&UInt64Array::from(rows)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        }
    }

    /// Serializes the tests that assert exact deltas of the attach counters
    /// and the view gauges, which only these tests move in this crate.
    static ATTACH_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// A store of [`column`], 100 rows of `factor`, live in the registry
    /// under a key of its own, as an index open binds it.
    fn live_column_store(test: &str, column: ResidentColumn) -> Arc<ResidentSlot> {
        slot_with(
            &test_file(test),
            ResidentColumnStore {
                columns: HashMap::from([("factor".to_string(), column)]),
                num_rows: 100,
                bytes: 400,
                views: ResidentStoreViews::default(),
            },
        )
    }

    /// The attach counters, in a fixed order: calls, rows, bytes, shares,
    /// whole copies, gathers and copied bytes.
    fn attach_counters() -> [u64; 7] {
        let counters = layered_stats::counters();
        [
            counters.resident_attach_calls.get(),
            counters.resident_attach_rows.get(),
            counters.resident_attach_bytes.get(),
            counters.resident_attach_shares.get(),
            counters.resident_attach_whole_copies.get(),
            counters.resident_attach_gathers.get(),
            counters.resident_attach_copied_bytes.get(),
        ]
    }

    fn counter_deltas(before: [u64; 7]) -> [u64; 7] {
        let after = attach_counters();
        std::array::from_fn(|counter| after[counter] - before[counter])
    }

    /// A view of a range of rows shares the store column's buffer, at the
    /// range's first row, and is charged exactly its rows' bytes; it lives,
    /// readable, after the column drops, and the gauges count it until it
    /// drops. An empty range is an empty array, a range past the file an
    /// error.
    #[test]
    fn share_rows_views_the_store_without_copying() {
        let _serial = ATTACH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let column = column();
        let store_values = column
            .values
            .as_primitive::<Float32Type>()
            .values()
            .as_ptr();
        let store_views = ResidentStoreViews::default();
        let baseline = resident_store_views();
        let mut held = Vec::new();
        for range in [10..30, 30..60, 0..100, 99..100] {
            let context = format!("{range:?}");
            let (view, shared) = column.share_rows(range.clone(), &store_views).unwrap();
            assert!(shared, "{context}");
            let rows: Vec<u64> = (range.start as u64..range.end as u64).collect();
            assert_eq!(values(&view), as_values(&rows), "{context}");
            let view_values = view.as_primitive::<Float32Type>().values().as_ptr();
            assert_eq!(
                view_values,
                store_values.wrapping_add(range.start),
                "{context}"
            );
            assert!(view_values.is_aligned(), "{context}");
            assert!(view.nulls().is_none(), "{context}");
            assert_eq!(view.get_buffer_memory_size(), range.len() * 4, "{context}");
            assert_eq!(
                view.as_ref().deep_size_of_children(&mut Context::new()),
                range.len() * 4,
                "{context}"
            );
            held.push(view);
            assert_eq!(store_views.live(), held.len() as u64, "{context}");
            assert_eq!(
                resident_store_views(),
                baseline + held.len() as u64,
                "{context}"
            );
        }
        let (empty, _) = column.share_rows(100..100, &store_views).unwrap();
        assert_eq!(empty.len(), 0);
        let error = column.share_rows(90..101, &store_views).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert_eq!(store_views.live(), held.len() as u64);

        drop(column);
        let every_row: Vec<u64> = (0..100).collect();
        assert_eq!(values(&held[2]), as_values(&every_row));
        drop(held);
        assert_eq!(store_views.live(), 0);
        assert_eq!(resident_store_views(), baseline);
    }

    /// A column with a validity buffer cannot be viewed as one buffer, so
    /// `share_rows` copies it, and the attach counts a whole copy. Arrow
    /// keeps a validity buffer only where a value is null, so the column
    /// has one null, past the rows read.
    #[test]
    fn share_rows_copies_a_column_with_nulls() {
        let _serial = ATTACH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let factors: Vec<Option<f32>> = (0..100)
            .map(|row| (row != 50).then_some(row as f32))
            .collect();
        let column = ResidentColumn {
            values: Arc::new(Float32Array::from(factors)),
            page_ends: vec![40, 40, 100],
        };
        assert!(column.values.to_data().nulls().is_some());
        let store_views = ResidentStoreViews::default();
        let (copied, shared) = column.share_rows(10..30, &store_views).unwrap();
        assert!(!shared);
        let rows: Vec<u64> = (10..30).collect();
        assert_eq!(values(&copied), as_values(&rows));
        assert_eq!(store_views.live(), 0);

        let slot = live_column_store("share-rows-nulls", column);
        let store = slot.store.get().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "factor",
            DataType::Float32,
            true,
        )]));
        let before = attach_counters();
        let batch = store
            .attach(
                schema,
                None,
                ResidentRows::Range(10..30),
                ResidentAttach::Share,
            )
            .unwrap();
        // One call of 20 rows of 4 bytes, a whole copy of them.
        assert_eq!(counter_deltas(before), [1, 20, 80, 0, 1, 0, 80]);
        assert!(!shares_resident_store(&batch));
        assert_eq!(values(batch.column(0)), as_values(&rows));
    }

    /// Attaching the resident columns to a batch of the other columns at
    /// the same rows lays the columns out in the schema's order, takes the
    /// resident rows as the mode says and shares the others' buffers. Under
    /// `share`, a whole range views the store, charged its rows alone; a
    /// copy, under `copy` or of sparse rows, is a buffer of exactly its
    /// rows. Either way the batch is the same, and the store's cache entry
    /// is charged the same. A batch of other rows, a column the store does
    /// not keep missing from it, or a range past the file is an error.
    #[rstest::rstest]
    fn attach_shares_or_copies_resident_rows(
        #[values(ResidentAttach::Share, ResidentAttach::Copy)] mode: ResidentAttach,
        #[values(false, true)] sparse: bool,
    ) {
        let _serial = ATTACH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let slot = live_column_store(&format!("attach-{mode}-{sparse}"), column());
        let store = slot.store.get().unwrap();
        let entry_bytes = ResidentColumnsEntry::new(slot.clone()).deep_size_of();
        assert!(is_file_column(RABIT_CODE_COLUMN));
        assert!(!is_file_column("factor"));
        let schema = Arc::new(Schema::new(vec![
            Field::new(RABIT_CODE_COLUMN, DataType::UInt8, true),
            Field::new("factor", DataType::Float32, true),
        ]));
        let rows: Vec<u64> = if sparse {
            vec![5, 39, 40, 99]
        } else {
            (5..45).collect()
        };
        let file_rows = UInt64Array::from(rows.clone());
        let resident_rows = || {
            if sparse {
                ResidentRows::Rows(&file_rows)
            } else {
                ResidentRows::Range(5..45)
            }
        };
        let codes: ArrayRef = Arc::new(arrow_array::UInt8Array::from_iter_values(
            (0..rows.len()).map(|row| row as u8),
        ));
        let file_batch = RecordBatch::try_from_iter([(RABIT_CODE_COLUMN, codes.clone())]).unwrap();
        let shared = mode == ResidentAttach::Share && !sparse;

        let before = attach_counters();
        let batch = store
            .attach(schema.clone(), Some(&file_batch), resident_rows(), mode)
            .unwrap();
        let bytes = 4 * rows.len() as u64;
        let copied = if shared { 0 } else { bytes };
        let (shares, whole_copies, gathers) = match (sparse, shared) {
            (true, _) => (0, 0, 1),
            (false, true) => (1, 0, 0),
            (false, false) => (0, 1, 0),
        };
        assert_eq!(
            counter_deltas(before),
            [
                1,
                rows.len() as u64,
                bytes,
                shares,
                whole_copies,
                gathers,
                copied
            ]
        );
        assert_eq!(batch.schema(), schema);
        assert_eq!(values(batch.column(1)), as_values(&rows));
        assert_eq!(
            batch.column(0).to_data().buffers()[0].as_ptr(),
            codes.to_data().buffers()[0].as_ptr()
        );
        assert_eq!(shares_resident_store(&batch), shared);
        assert_eq!(store.views.live(), u64::from(shared));
        assert_eq!(batch.column(1).get_buffer_memory_size(), rows.len() * 4);
        assert_eq!(
            ResidentColumnsEntry::new(slot.clone()).deep_size_of(),
            entry_bytes
        );

        // The other mode attaches the same batch, of the same bytes.
        let other = match mode {
            ResidentAttach::Share => ResidentAttach::Copy,
            ResidentAttach::Copy => ResidentAttach::Share,
        };
        let twin = store
            .attach(schema.clone(), Some(&file_batch), resident_rows(), other)
            .unwrap();
        assert_eq!(twin, batch);
        assert_eq!(
            twin.column(1).get_buffer_memory_size(),
            batch.column(1).get_buffer_memory_size()
        );
        drop((batch, twin));
        assert_eq!(store.views.live(), 0);

        let short = file_batch.slice(0, 2);
        for (batch, rows) in [(Some(&short), resident_rows()), (None, resident_rows())] {
            let error = store.attach(schema.clone(), batch, rows, mode).unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error}");
        }
        let error = store
            .attach(schema, None, ResidentRows::Range(90..101), mode)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    /// Rows across pages are copied into one buffer of exactly their
    /// values, as a whole read of the file returns them wherever its pages
    /// end.
    #[test]
    fn copy_rows_across_pages_holds_exactly_the_rows() {
        let column = column();
        for rows in [
            (30..60).collect::<Vec<u64>>(),
            vec![5, 39, 40, 99],
            (0..100).collect(),
        ] {
            let copied = copy(&column, &rows);
            assert_eq!(values(&copied), as_values(&rows));
            assert!(copied.nulls().is_none());
            let (first, second): (Vec<u64>, Vec<u64>) = rows.iter().partition(|&&row| row < 40);
            let pages = [first, second]
                .map(|page| take(column.values.as_ref(), &UInt64Array::from(page), None).unwrap());
            let read =
                arrow_select::concat::concat(&[pages[0].as_ref(), pages[1].as_ref()]).unwrap();
            assert_eq!(copied.as_ref(), read.as_ref(), "{rows:?}");
            assert_eq!(copied.get_buffer_memory_size(), rows.len() * 4, "{rows:?}");
        }
    }

    /// A reader of a file of `rows` rows of one four-byte column the store
    /// keeps, in memory.
    async fn factor_file(rows: usize) -> FileReader {
        let values: Vec<f32> = (0..rows).map(|row| row as f32).collect();
        let factors: ArrayRef = Arc::new(Float32Array::from(values));
        let batch = RecordBatch::try_from_iter([("factor", factors)]).unwrap();
        let store = Arc::new(ObjectStore::memory());
        let path = object_store::path::Path::from("resident/factors.lance");
        let mut writer = lance_file::versions::create_writer(
            LanceFileVersion::default().resolve(),
            store.create(&path).await.unwrap(),
            lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        writer.finish().await.unwrap();
        let scheduler = ScanScheduler::new(store, SchedulerConfig::default_for_testing());
        FileReader::try_open(
            scheduler
                .open_file(&path, &CachedFileSize::unknown())
                .await
                .unwrap(),
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .unwrap()
    }

    /// A read that finds the store not loaded loads it, as a fallback for
    /// a storage no index open loaded it for, and counts that load in
    /// `resident_columns_read_loads` too. The counters are process-wide, so
    /// only their growth is bounded here; the index tests check that opens
    /// count no read loads.
    #[tokio::test]
    async fn read_triggered_load_counts_a_read_load() {
        const ROWS: usize = 100;
        let reader = factor_file(ROWS).await;
        let counters = layered_stats::counters();
        let (loads, read_loads) = (
            counters.resident_columns_loads.get(),
            counters.resident_columns_read_loads.get(),
        );
        let handle = ResidentColumns::default();
        let store = handle
            .get_or_load(&reader, None, ResidentLoadTrigger::Read)
            .await
            .unwrap();
        assert_eq!(store.num_rows(), ROWS as u64);
        assert_eq!(handle.loaded_bytes(), Some(4 * ROWS as u64));
        assert!(counters.resident_columns_loads.get() > loads);
        assert!(counters.resident_columns_read_loads.get() > read_loads);
    }
}
