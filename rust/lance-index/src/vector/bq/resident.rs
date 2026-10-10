// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The small columns of an IVF_RQ storage file, kept in memory.
//!
//! A read of a partition or plane from the file costs at least one request
//! per column, yet every column but the codes and the estimator bounds holds
//! a few bytes per row: row ids and factors. With those columns read once
//! per index, later reads fetch only the codes (and bounds) from the file:
//! on an object store a native partition miss takes 2 requests instead of
//! 8, and a layered one 3 instead of 13. Reads attach copies of the resident
//! rows, so a batch holds, and is charged, what a file read returns. Cache
//! entries hold that batch, or only the columns reads fetch from the file
//! (`EntryColumns::Codes`), and every read of such an entry attaches the
//! resident rows again.
//!
//! The store is an entry of the index cache ([`ResidentColumnsKey`]),
//! charged what its arrays allocate and evictable under ordinary cache
//! pressure. Each live index holds an `Arc` to its store independently of
//! the cache. A weak registry shares that allocation across index handles,
//! even after eviction; concurrent handles load it once. An idle store is
//! freed when the cache evicts it and no index holds it anymore.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Instant;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array, new_empty_array};
use arrow_schema::{Field, Schema, SchemaRef};
use arrow_select::take::take;
use futures::TryStreamExt;
use lance_core::cache::{
    CacheKey, CacheKeySchema, InternalCacheKey, KeyBuilder, LanceCache, WeakLanceCache,
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
use crate::vector::storage::{IndexFileKey, ResidentLifetime, WeakRegistry, shared_by_key};

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
/// [`ResidentLifetime::Process`], one strong reference per index file.
static PROCESS_STORES: LazyLock<Mutex<HashMap<IndexFileKey, Arc<ResidentSlot>>>> =
    LazyLock::new(Default::default);

/// One index file's store, shared by every handle of the file and by its
/// index cache entry, and loaded on first use.
#[derive(Default)]
struct ResidentSlot {
    store: OnceCell<ResidentColumnStore>,
}

/// Prints the store's columns, not its values.
impl std::fmt::Debug for ResidentSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentSlot")
            .field("store", &self.store.get())
            .finish()
    }
}

impl ResidentSlot {
    /// The store of `reader`'s file, loaded on first use. Concurrent first
    /// callers share one load, whose I/O is added to the loading caller's
    /// `io_stats`; a failed or dropped load leaves the store for the next
    /// caller to load. The load runs once per store, so it is boxed rather
    /// than inlined into every read's future.
    async fn get_or_load(
        &self,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
    ) -> Result<&ResidentColumnStore> {
        self.store
            .get_or_try_init(|| Box::pin(ResidentColumnStore::load(reader, io_stats)))
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

/// Whether a loaded resident store of `file` is alive in the process.
pub fn resident_store_is_live(file: &IndexFileKey) -> bool {
    RESIDENT_SLOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(file)
        .and_then(Weak::upgrade)
        .is_some_and(|slot| slot.is_loaded())
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
/// what the store's arrays allocate. Eviction releases only this reference;
/// live index handles continue to own the store.
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

/// A storage's handle on the resident columns of its file, loaded on first
/// use. A handle bound to an index cache
/// ([`in_index_cache`](Self::in_index_cache)) shares the store with every
/// live handle of the file and caches it once loaded. The handle's strong
/// reference keeps the data alive through cache eviction. A default handle
/// holds a store of its own, charged nowhere.
#[derive(Debug, Clone, Default)]
pub struct ResidentColumns {
    slot: Arc<ResidentSlot>,
    /// `None` for a store charged nowhere.
    charge: Option<ResidentCharge>,
    /// Cache the loaded store once per handle, without consulting the cache
    /// on every partition read. Reopening an index can cache it again.
    cached: OnceCell<()>,
}

impl ResidentColumns {
    /// The store of index file `file` that every live handle of it shares,
    /// charged nowhere, for an index opened without an index cache.
    pub fn shared(file: &IndexFileKey) -> Self {
        Self {
            slot: shared_slot(file).0,
            charge: None,
            cached: OnceCell::new(),
        }
    }

    /// The handle of an index opening index file `file` with its small
    /// columns resident: the store every live handle of the file shares,
    /// charged in `cache`, the index's namespace of the index cache without
    /// a fragment reuse segment. `preopen` holds the shared store through
    /// cache accesses performed while opening the index. Loaded stores are
    /// cached now; otherwise the first read loads and caches the store.
    /// Under [`ResidentLifetime::Process`] a separate strong reference keeps
    /// the loaded store alive until the process exits.
    pub async fn in_index_cache(
        cache: &LanceCache,
        file: &IndexFileKey,
        preopen: Option<Self>,
        lifetime: ResidentLifetime,
    ) -> Self {
        // Keep the same allocation even if opening the index evicted it.
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
            cached: OnceCell::new(),
        };
        handle.cache_loaded().await;
        handle
    }

    /// Offer the loaded store to the ordinary cache once per handle. Cache
    /// admission and eviction do not control the store's lifetime.
    async fn cache_loaded(&self) {
        let Some(charge) = &self.charge else {
            return;
        };
        if !self.slot.is_loaded() {
            return;
        }
        self.cached
            .get_or_init(|| async {
                let key = ResidentColumnsKey::new(&charge.file);
                if charge.cache.get_with_key(&key).await.is_none() {
                    charge
                        .cache
                        .insert_with_key(
                            &key,
                            Arc::new(ResidentColumnsEntry::new(self.slot.clone())),
                        )
                        .await;
                }
                if charge.lifetime == ResidentLifetime::Process {
                    PROCESS_STORES
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(charge.file.clone())
                        .or_insert_with(|| self.slot.clone());
                }
            })
            .await;
    }

    /// Bytes of values the store holds, `None` until it has loaded. Equals
    /// [`resident_columns_bytes`] of its file.
    pub fn loaded_bytes(&self) -> Option<u64> {
        self.slot.store.get().map(|store| store.bytes)
    }

    /// The store of `reader`'s file, loaded on first use: see
    /// [`ResidentSlot::get_or_load`]. Cache the loaded allocation without
    /// making subsequent reads depend on whether it stays in the cache.
    pub(crate) async fn get_or_load(
        &self,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
    ) -> Result<&ResidentColumnStore> {
        let store = self.slot.get_or_load(reader, io_stats).await?;
        self.cache_loaded().await;
        Ok(store)
    }
}

/// Every row of the resident columns of one file.
pub(crate) struct ResidentColumnStore {
    columns: HashMap<String, ResidentColumn>,
    num_rows: u64,
    /// See [`ResidentColumns::loaded_bytes`].
    bytes: u64,
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
    async fn load(reader: &FileReader, io_stats: Option<&IoStats>) -> Result<Self> {
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

    /// `schema`'s columns at the ascending file rows `rows`: copies of this
    /// store's rows of the columns it keeps (see [`ResidentColumn::copy_rows`])
    /// and, by name, the other columns of `file_batch`, which holds them at
    /// the same rows, read from the file or a code-only cache entry. The
    /// batch is the one a read of every column from the file returns.
    pub(crate) fn attach(
        &self,
        schema: SchemaRef,
        file_batch: Option<&RecordBatch>,
        rows: &UInt64Array,
    ) -> Result<RecordBatch> {
        let started = Instant::now();
        if let Some(batch) = file_batch
            && batch.num_rows() != rows.len()
        {
            return Err(Error::internal(format!(
                "resident columns attach to {} rows of a batch of {} rows",
                rows.len(),
                batch.num_rows()
            )));
        }
        let mut copied_bytes = 0;
        let mut columns = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let column = match self.column(field.name()) {
                Some(column) => {
                    let copy = column.copy_rows(rows)?;
                    copied_bytes += copy.get_buffer_memory_size() as u64;
                    copy
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
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )?;
        let stats = layered_stats::counters();
        stats.resident_attach_calls.incr();
        stats.resident_attach_rows.add(rows.len() as u64);
        stats.resident_attach_bytes.add(copied_bytes);
        stats.resident_attach_ns.add_elapsed(started);
        Ok(batch)
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
/// largest admissible entry before automatically keeping a store resident.
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
    /// A copy of the ascending file rows `rows` in a buffer of exactly their
    /// values, as a whole read of the file returns them (see
    /// `read_projected`), so that a cache entry holding it is charged the
    /// same bytes. A slice would instead keep, and be charged, the whole
    /// column.
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

    use crate::vector::bq::layered::{HIGH_ADD_FACTORS_COLUMN, HIGH_SCALE_FACTORS_COLUMN};
    use crate::vector::bq::transform::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
    };

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
        }
    }

    /// The slot of `file`, loaded with a test store, and held by the caller.
    fn loaded_slot(file: &IndexFileKey, rows: usize) -> Arc<ResidentSlot> {
        let (slot, _) = shared_slot(file);
        slot.store.set(test_store(rows)).unwrap();
        slot
    }

    fn test_cache() -> LanceCache {
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

    #[tokio::test]
    async fn in_index_cache_retains_idle_store() {
        let file = test_file("in-index-cache");
        let slot = loaded_slot(&file, 1000);
        let held = Arc::downgrade(&slot);
        let cache = test_cache();
        let key = ResidentColumnsKey::new(&file);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        assert!(cache.get_with_key(&key).await.is_some());
        let clone = handle.clone();
        drop((handle, clone, slot));
        assert!(
            held.upgrade().is_some(),
            "the idle cache entry owns the store"
        );
        cache.clear().await;
        assert!(
            held.upgrade().is_none(),
            "the weak registry must not retain data"
        );
    }

    #[tokio::test]
    async fn live_store_is_evictable_and_reused_after_eviction() {
        let file = test_file("active-eviction");
        let slot = loaded_slot(&file, 1000);
        let held = Arc::downgrade(&slot);
        let cache = LanceCache::with_backend(Arc::new(QuickCacheBackend::with_capacity(8192)));
        let key = ResidentColumnsKey::new(&file);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        drop(slot);
        assert!(cache.peek_resident_with_key(&key).await);
        for id in 0..16 {
            let other = test_file(&format!("eviction-pressure-{id}"));
            let slot = loaded_slot(&other, 1000);
            cache
                .insert_with_key(
                    &ResidentColumnsKey::new(&other),
                    Arc::new(ResidentColumnsEntry::new(slot)),
                )
                .await;
        }
        assert!(!cache.peek_resident_with_key(&key).await);
        assert!(cache.size_bytes().await <= 8192);
        assert_eq!(handle.loaded_bytes(), Some(4000));
        let reopened =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        assert!(Arc::ptr_eq(&handle.slot, &reopened.slot));
        assert_eq!(reopened.loaded_bytes(), Some(4000));
        let values = reopened.slot.store.get().unwrap().columns["factor"]
            .values
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(values.values().iter().all(|value| *value == 1.0));
        drop((handle, reopened));
        cache.clear().await;
        assert!(held.upgrade().is_none());
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
    /// is what an ordinary cache charges for it once loaded.
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
            let _handle =
                ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
            assert_eq!(cache.size_bytes().await as u64, charge, "{rows} {pages}");
        }
    }

    #[tokio::test]
    async fn preopen_handle_keeps_a_store_evicted_during_the_open() {
        let file = test_file("preopen-evicted");
        let cache = test_cache();
        let key = ResidentColumnsKey::new(&file);
        let slot = loaded_slot(&file, 1000);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Index).await;
        let held = Arc::downgrade(&slot);
        drop((handle, slot));
        let preopen = ResidentColumns::shared(&file);
        cache.clear().await;
        assert!(!cache.peek_resident_with_key(&key).await);
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, Some(preopen), ResidentLifetime::Index)
                .await;
        assert_eq!(handle.loaded_bytes(), Some(4000));
        assert!(Arc::ptr_eq(&handle.slot, &held.upgrade().unwrap()));
        assert!(cache.peek_resident_with_key(&key).await);
        drop(handle);
        cache.clear().await;
        assert!(held.upgrade().is_none());
    }

    #[tokio::test]
    async fn resident_lifetime_process_keeps_store_alive() {
        let file = test_file("process-lifetime");
        let slot = loaded_slot(&file, 1000);
        let held = Arc::downgrade(&slot);
        let cache = test_cache();
        let handle =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Process).await;
        drop((handle, slot));
        cache.clear().await;
        let slot = held
            .upgrade()
            .expect("the process owns the store after eviction");
        let reopened =
            ResidentColumns::in_index_cache(&cache, &file, None, ResidentLifetime::Process).await;
        assert!(Arc::ptr_eq(&reopened.slot, &slot));
        PROCESS_STORES.lock().unwrap().remove(&file);
        drop((slot, reopened));
        cache.clear().await;
        assert!(held.upgrade().is_none());
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

    /// Attaching the resident columns to a batch of the other columns at
    /// the same rows lays the columns out in the schema's order, copies the
    /// resident rows and shares the others' buffers; a batch of other rows
    /// or without a column the store does not keep is an error.
    #[test]
    fn attach_copies_resident_rows_and_takes_file_columns() {
        let store = ResidentColumnStore {
            columns: HashMap::from([("factor".to_string(), column())]),
            num_rows: 100,
            bytes: 400,
        };
        assert!(is_file_column(RABIT_CODE_COLUMN));
        assert!(!is_file_column("factor"));
        let schema = Arc::new(Schema::new(vec![
            Field::new(RABIT_CODE_COLUMN, DataType::UInt8, true),
            Field::new("factor", DataType::Float32, true),
        ]));
        let rows = [5u64, 39, 40, 99];
        let codes: ArrayRef = Arc::new(arrow_array::UInt8Array::from(vec![1u8, 2, 3, 4]));
        let file_batch = RecordBatch::try_from_iter([(RABIT_CODE_COLUMN, codes.clone())]).unwrap();
        let calls = || layered_stats::counters().resident_attach_calls.get();
        let before = calls();
        let batch = store
            .attach(
                schema.clone(),
                Some(&file_batch),
                &UInt64Array::from(rows.to_vec()),
            )
            .unwrap();
        assert!(calls() > before);
        assert_eq!(batch.schema(), schema);
        assert_eq!(values(batch.column(1)), as_values(&rows));
        assert_eq!(
            batch.column(0).to_data().buffers()[0].as_ptr(),
            codes.to_data().buffers()[0].as_ptr()
        );
        for (batch, rows) in [(Some(&file_batch), vec![5, 39]), (None, rows.to_vec())] {
            let error = store
                .attach(schema.clone(), batch, &UInt64Array::from(rows))
                .unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error}");
        }
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
}
