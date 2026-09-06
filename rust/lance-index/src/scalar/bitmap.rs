// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::{
    any::Any,
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fmt::Debug,
    ops::Bound,
    sync::Arc,
};

use arrow::array::BinaryBuilder;
use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array, new_null_array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{
    Error, ROW_ID, Result,
    cache::{
        CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
        KeyBuilder, LanceCache, WeakLanceCache,
    },
    error::LanceOptionExt,
    utils::tokio::get_num_compute_intensive_cpus,
};
use lance_io::object_store::ObjectStore;
use lance_select::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};

use super::{AnyQuery, IndexFile, IndexStore, ScalarIndex, SearchOptions};
use super::{
    BuiltinIndexType, SargableQuery, ScalarIndexParams, SearchResult, btree::OrderableScalarValue,
};
use crate::pbold;
use crate::{Index, IndexType, metrics::MetricsCollector};
use crate::{
    progress::IndexBuildProgress,
    scalar::{
        CreatedIndex, RowIdRemapper, UpdateCriteria,
        expression::SargableQueryParser,
        registry::{
            BasicTrainer, ScalarIndexLoad, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering,
            TrainingRequest, VALUE_COLUMN_NAME, single_flight_open,
        },
    },
};
use crate::{scalar::IndexReader, scalar::expression::ScalarQueryParser};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
pub const INDEX_STATS_METADATA_KEY: &str = "lance:index_stats";
const BITMAP_PART_LOOKUP_PREFIX: &str = "part_";
const BITMAP_PART_LOOKUP_SUFFIX: &str = "_bitmap_page_lookup.lance";
const EXPLICIT_SHARD_ID_TAG: u64 = 0;
const IMPLICIT_FRAGMENT_ID_TAG: u64 = 1;

/// Maximum bytes a [`BitmapBatchWriter`] buffers before flushing a record
/// batch.
///
/// Charged for the keys as well as the serialized bitmaps, so it limits this
/// writer's buffered state independently of how many keys the index has. A
/// flush temporarily makes another copy to build the Arrow arrays, and a single
/// entry can exceed the threshold because it is checked after serialization.
/// Memory held by the caller, input pipeline, caches, or merge state is outside
/// this writer limit.
///
/// It also keeps both output columns far below the `i32` offset ceiling of
/// Arrow's `Binary`/`Utf8` layouts. The previous threshold was that ceiling
/// itself, which charged the bitmap column only: a high-cardinality column with
/// tiny bitmaps could overflow the keys column's offsets before it ever tripped.
const MAX_BUFFERED_BYTES: usize = 32 * 1024 * 1024;

const MAX_ROWS_PER_CHUNK: usize = 2 * 1024;
// Smaller than MAX_ROWS_PER_CHUNK to cap the rows retained per cursor during a
// k-way merge (N cursors x chunk), while still amortising I/O over a reasonable
// number of rows per read. This is not a byte limit because bitmap sizes vary.
const MERGE_ROWS_PER_CHUNK: usize = 512;

const BITMAP_INDEX_VERSION: u32 = 0;

// We only need to open a file reader if we need to load a bitmap. If all
// bitmaps are cached we don't open it. If we do open it we should only open it once.
#[derive(Clone)]
struct LazyIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn IndexReader>>>>,
    store: Arc<dyn IndexStore>,
}

impl std::fmt::Debug for LazyIndexReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyIndexReader")
            .field("store", &self.store)
            .finish()
    }
}

impl LazyIndexReader {
    fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
        }
    }

    async fn get(&self) -> Result<Arc<dyn IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let index_reader = self.store.open_index_file(BITMAP_LOOKUP_NAME).await?;
            *reader = Some(index_reader);
        }
        Ok(reader.as_ref().unwrap().clone())
    }
}

/// A scalar index that stores a bitmap for each possible value
///
/// This index works best for low-cardinality columns, where the number of unique values is small.
/// The bitmap stores a list of row ids where the value is present.
#[derive(Clone, Debug)]
pub struct BitmapIndex {
    /// Maps each unique value to its bitmap location in the index file
    /// The usize value is the row offset in the bitmap_page_lookup.lance file
    /// for quickly locating the row and reading it out
    index_map: Arc<BTreeMap<OrderableScalarValue, usize>>,

    null_map: Arc<RowAddrTreeMap>,

    value_type: DataType,

    store: Arc<dyn IndexStore>,

    index_cache: WeakLanceCache,

    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,

    lazy_reader: LazyIndexReader,
}

#[derive(Debug, Clone)]
pub struct BitmapKey {
    row_offset: u64,
}

impl BitmapKey {
    fn try_new(row_offset: usize) -> Result<Self> {
        let row_offset = u64::try_from(row_offset).map_err(|_| {
            Error::internal(format!(
                "bitmap row offset {row_offset} does not fit in u64"
            ))
        })?;
        Ok(Self { row_offset })
    }
}

impl CacheKey for BitmapKey {
    type ValueType = RowAddrTreeMap;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.row_offset.to_string().into()
    }

    fn type_name() -> &'static str {
        "Bitmap"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.bitmap-row-offset-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.row_offset);
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<RowAddrTreeMap>())
    }
}

/// The serializable state of a [`BitmapIndex`].
///
/// `BitmapIndex` holds non-serializable infrastructure (an `IndexStore`, a
/// cache handle, a lazy reader, a fragment-reuse index). `BitmapIndexState`
/// captures just the data needed to rebuild it: the value→file-offset map,
/// the null bitmap, and the value type.
#[derive(Debug, Clone)]
pub struct BitmapIndexState {
    /// Value-to-row-offset lookup, encoded as an Arrow `RecordBatch` so we can
    /// reuse the existing IPC utilities for zero-copy round trips.
    ///
    /// Schema: `keys: <value_type>`, `offsets: UInt64`. Iteration order of
    /// `index_map` is preserved on serialize and the `BTreeMap` resorts the
    /// entries on deserialize, so the wire form does not need to be sorted.
    lookup_batch: RecordBatch,
    /// Already-remapped null bitmap (remapping is applied during load, so the
    /// cached state matches the in-memory representation).
    null_map: Arc<RowAddrTreeMap>,
    /// Cached separately from the schema for the empty-index case where the
    /// `lookup_batch` is empty but we still need to remember the column type.
    value_type: DataType,
    /// Parsed form of `lookup_batch`. Not serialized — populated eagerly in
    /// both [`BitmapIndexState::from_index`] and [`CacheCodecImpl::deserialize`].
    /// Stored as `Arc` so cloning into a new [`BitmapIndex`] is O(1).
    index_map: Arc<BTreeMap<OrderableScalarValue, usize>>,
}

impl DeepSizeOf for BitmapIndexState {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.lookup_batch.get_array_memory_size()
            + self.null_map.deep_size_of_children(context)
            + self.index_map.deep_size_of_children(context)
    }
}

impl BitmapIndexState {
    pub(crate) fn from_index(index: &BitmapIndex) -> Result<Self> {
        Ok(Self {
            lookup_batch: build_lookup_batch(&index.index_map, &index.value_type)?,
            null_map: index.null_map.clone(),
            value_type: index.value_type.clone(),
            index_map: index.index_map.clone(),
        })
    }

    fn from_scalar_index(index: &dyn ScalarIndex) -> Result<Self> {
        let bitmap = index
            .as_any()
            .downcast_ref::<BitmapIndex>()
            .ok_or_else(|| {
                Error::internal(
                    "BitmapIndexState::from_scalar_index called with a non-bitmap index",
                )
            })?;
        Self::from_index(bitmap)
    }

    pub(crate) fn to_bitmap_index(
        &self,
        store: Arc<dyn IndexStore>,
        index_cache: &LanceCache,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Arc<BitmapIndex>> {
        Ok(Arc::new(BitmapIndex::new(
            self.index_map.clone(),
            self.null_map.clone(),
            self.value_type.clone(),
            store,
            WeakLanceCache::from(index_cache),
            frag_reuse_index,
        )))
    }

    /// Build a state directly from its parts, for codec tests in sibling
    /// modules (e.g. the label-list index, which nests a bitmap state).
    #[cfg(test)]
    pub(crate) fn new_for_test(
        index_map: BTreeMap<OrderableScalarValue, usize>,
        null_map: RowAddrTreeMap,
        value_type: DataType,
    ) -> Result<Self> {
        Ok(Self {
            lookup_batch: build_lookup_batch(&index_map, &value_type)?,
            null_map: Arc::new(null_map),
            value_type,
            index_map: Arc::new(index_map),
        })
    }

    #[cfg(test)]
    pub(crate) fn lookup_batch(&self) -> &RecordBatch {
        &self.lookup_batch
    }

    #[cfg(test)]
    pub(crate) fn null_map(&self) -> &RowAddrTreeMap {
        &self.null_map
    }
}

fn build_lookup_batch(
    index_map: &BTreeMap<OrderableScalarValue, usize>,
    value_type: &DataType,
) -> Result<RecordBatch> {
    let keys = if index_map.is_empty() {
        arrow_array::new_empty_array(value_type)
    } else {
        ScalarValue::iter_to_array(index_map.keys().map(|k| k.0.clone()))?
    };
    let offsets = Arc::new(UInt64Array::from_iter_values(
        index_map.values().map(|v| *v as u64),
    ));
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", value_type.clone(), true),
        Field::new("offsets", DataType::UInt64, false),
    ]));
    Ok(RecordBatch::try_new(schema, vec![keys, offsets])?)
}

fn parse_lookup_batch(batch: &RecordBatch) -> Result<BTreeMap<OrderableScalarValue, usize>> {
    let keys = batch.column(0);
    let offsets = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            Error::internal("BitmapIndexState: expected UInt64 offsets column".to_string())
        })?;
    let mut index_map = BTreeMap::new();
    for idx in 0..batch.num_rows() {
        let value = OrderableScalarValue(ScalarValue::try_from_array(keys, idx)?);
        index_map.insert(value, offsets.value(idx) as usize);
    }
    Ok(index_map)
}

impl CacheCodecImpl for BitmapIndexState {
    const TYPE_ID: &'static str = "lance.scalar.BitmapIndexState";
    const CURRENT_VERSION: u32 = 1;

    /// Wire format:
    /// ```text
    /// RAW_BLOB  : null_map (roaring tree map, portable encoding)
    /// ARROW_IPC : (keys: <value_type>, offsets: UInt64)
    /// ```
    /// The value type is recovered from the IPC section schema.
    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let mut null_bytes = Vec::with_capacity(self.null_map.serialized_size());
        self.null_map.serialize_into(&mut null_bytes)?;
        w.write_raw(&null_bytes)?;
        w.write_ipc(&self.lookup_batch)?;
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let null_bytes = r.read_raw()?;
        let null_map = Arc::new(RowAddrTreeMap::deserialize_from(null_bytes.as_ref())?);
        let lookup_batch = r.read_ipc()?;
        let value_type = lookup_batch.schema().field(0).data_type().clone();
        let index_map = Arc::new(parse_lookup_batch(&lookup_batch)?);
        Ok(Self {
            lookup_batch,
            null_map,
            value_type,
            index_map,
        })
    }
}

/// Cache key for a [`BitmapIndexState`]. The cache is already namespaced
/// per-index by the caller, so a constant key suffices.
struct BitmapIndexStateKey;

impl CacheKey for BitmapIndexStateKey {
    type ValueType = BitmapIndexState;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "state".into()
    }

    fn type_name() -> &'static str {
        "BitmapIndexState"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.bitmap-index-state-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_variant(0);
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<BitmapIndexState>())
    }
}

impl BitmapIndex {
    fn new(
        index_map: Arc<BTreeMap<OrderableScalarValue, usize>>,
        null_map: Arc<RowAddrTreeMap>,
        value_type: DataType,
        store: Arc<dyn IndexStore>,
        index_cache: WeakLanceCache,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Self {
        let lazy_reader = LazyIndexReader::new(store.clone());
        Self {
            index_map,
            null_map,
            value_type,
            store,
            index_cache,
            frag_reuse_index,
            lazy_reader,
        }
    }

    pub(crate) async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        let total_rows = page_lookup_file.num_rows();

        if total_rows == 0 {
            let schema = page_lookup_file.schema();
            let data_type = schema.fields[0].data_type();
            return Ok(Arc::new(Self::new(
                Arc::new(BTreeMap::new()),
                Arc::new(RowAddrTreeMap::default()),
                data_type,
                store,
                WeakLanceCache::from(index_cache),
                frag_reuse_index,
            )));
        }

        let mut index_map: BTreeMap<OrderableScalarValue, usize> = BTreeMap::new();
        let mut null_map = Arc::new(RowAddrTreeMap::default());
        let mut null_location: Option<usize> = None;
        let value_type = page_lookup_file.schema().fields[0].data_type();

        // Stream keys in bounded batches to avoid loading the entire keys
        // column into memory at once.
        let mut keys_stream = page_lookup_file
            .read_range_stream(0..total_rows, Some(&["keys"]))
            .await?;
        let mut row_offset: usize = 0;
        while let Some(keys_batch) = keys_stream.try_next().await? {
            let dict_keys = keys_batch.column(0);
            for idx in 0..keys_batch.num_rows() {
                let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);
                if key.0.is_null() {
                    null_location = Some(row_offset);
                } else {
                    index_map.insert(key, row_offset);
                }
                row_offset += 1;
            }
        }

        if let Some(null_loc) = null_location {
            let batch = page_lookup_file
                .read_range(null_loc..null_loc + 1, Some(&["bitmaps"]))
                .await?;

            let binary_bitmaps = batch
                .column(0)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::internal("Invalid bitmap column type".to_string()))?;
            let bitmap_bytes = binary_bitmaps.value(0);
            let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

            // Apply fragment remapping if needed
            if let Some(fri) = &frag_reuse_index {
                bitmap = fri.remap_row_addrs_tree_map(&bitmap);
            }

            null_map = Arc::new(bitmap);
        }

        Ok(Arc::new(Self::new(
            Arc::new(index_map),
            null_map,
            value_type,
            store,
            WeakLanceCache::from(index_cache),
            frag_reuse_index,
        )))
    }

    async fn load_bitmap(
        &self,
        key: &OrderableScalarValue,
        metrics: Option<&dyn MetricsCollector>,
    ) -> Result<Arc<RowAddrTreeMap>> {
        if key.0.is_null() {
            return Ok(self.null_map.clone());
        }

        // A value that isn't in `index_map` never reaches the loader or the
        // cache, so it should not touch the per-query cache counters either.
        // Checking here (before the cached-lookup fast path) also avoids
        // returning an unmapped-value response as a spurious cache hit if a
        // prior insert somehow ended up under `cache_key`.
        let row_offset = match self.index_map.get(key) {
            Some(loc) => *loc,
            None => return Ok(Arc::new(RowAddrTreeMap::default())),
        };
        let cache_key = BitmapKey::try_new(row_offset)?;

        if let Some(cached) = self.index_cache.get_with_key(&cache_key).await {
            if let Some(metrics) = metrics {
                metrics.record_index_cache_hit();
            }
            return Ok(cached);
        }

        // Record that we're loading a partition from disk
        if let Some(metrics) = metrics {
            metrics.record_index_cache_miss();
            metrics.record_part_load();
        }

        let page_lookup_file = self.lazy_reader.get().await?;
        let batch = page_lookup_file
            .read_range(row_offset..row_offset + 1, Some(&["bitmaps"]))
            .await?;

        let binary_bitmaps = batch
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| Error::internal("Invalid bitmap column type".to_string()))?;
        let bitmap_bytes = binary_bitmaps.value(0); // First (and only) row
        let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

        if let Some(fri) = &self.frag_reuse_index {
            bitmap = fri.remap_row_addrs_tree_map(&bitmap);
        }

        self.index_cache
            .insert_with_key(&cache_key, Arc::new(bitmap.clone()))
            .await;

        Ok(Arc::new(bitmap))
    }

    pub(crate) fn value_type(&self) -> &DataType {
        &self.value_type
    }
}

impl DeepSizeOf for BitmapIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.index_map.deep_size_of_children(context) + self.store.deep_size_of_children(context)
    }
}

#[derive(Serialize)]
struct BitmapStatistics {
    num_bitmaps: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BitmapParameters {
    /// Optional shard identifier for distributed bitmap builds spanning
    /// multiple fragments.
    pub shard_id: Option<u32>,
}

struct BitmapTrainingRequest {
    parameters: BitmapParameters,
    criteria: TrainingCriteria,
}

impl BitmapTrainingRequest {
    fn new(parameters: BitmapParameters) -> Self {
        Self {
            parameters,
            criteria: TrainingCriteria::new(TrainingOrdering::Values).with_row_id(),
        }
    }
}

impl TrainingRequest for BitmapTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl Index for BitmapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        let page_lookup_file = self.lazy_reader.get().await?;
        let total_rows = page_lookup_file.num_rows();

        if total_rows == 0 {
            return Ok(());
        }

        for start_row in (0..total_rows).step_by(MAX_ROWS_PER_CHUNK) {
            let end_row = (start_row + MAX_ROWS_PER_CHUNK).min(total_rows);
            let chunk = page_lookup_file
                .read_range(start_row..end_row, None)
                .await?;

            if chunk.num_rows() == 0 {
                continue;
            }

            let dict_keys = chunk.column(0);
            let binary_bitmaps = chunk.column(1);
            let bitmap_binary_array = binary_bitmaps
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();

            for idx in 0..chunk.num_rows() {
                let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);

                if key.0.is_null() {
                    continue;
                }

                let bitmap_bytes = bitmap_binary_array.value(idx);
                let mut bitmap = RowAddrTreeMap::deserialize_from(bitmap_bytes).unwrap();

                if let Some(frag_reuse_index_ref) = self.frag_reuse_index.as_ref() {
                    bitmap = frag_reuse_index_ref.remap_row_addrs_tree_map(&bitmap);
                }

                let row_offset = start_row.checked_add(idx).ok_or_else(|| {
                    Error::internal(format!(
                        "bitmap row offset overflow: start_row={start_row}, idx={idx}"
                    ))
                })?;
                let cache_key = BitmapKey::try_new(row_offset)?;
                self.index_cache
                    .insert_with_key(&cache_key, Arc::new(bitmap))
                    .await;
            }
        }

        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Bitmap
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = BitmapStatistics {
            num_bitmaps: self.index_map.len() + if !self.null_map.is_empty() { 1 } else { 0 },
        };
        serde_json::to_value(stats).map_err(|e| {
            Error::internal(format!(
                "failed to serialize bitmap index statistics: {}",
                e
            ))
        })
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for BitmapIndex {
    #[instrument(name = "bitmap_search", level = "debug", skip_all)]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        self.search_with_options(query, SearchOptions::default(), metrics)
            .await
    }

    async fn search_with_options(
        &self,
        query: &dyn AnyQuery,
        options: SearchOptions,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let tracked_null_rows = || {
            if options.track_nulls() && !self.null_map.is_empty() {
                Some((*self.null_map).clone())
            } else {
                None
            }
        };

        let (row_ids, null_row_ids) = match query {
            SargableQuery::Equals(val) => {
                metrics.record_comparisons(1);
                if val.is_null() {
                    // Querying FOR nulls - they are the TRUE result, not NULL result
                    ((*self.null_map).clone(), None)
                } else {
                    let key = OrderableScalarValue(val.clone());
                    let bitmap = self.load_bitmap(&key, Some(metrics)).await?;
                    ((*bitmap).clone(), tracked_null_rows())
                }
            }
            SargableQuery::Range(start, end) => {
                let range_start = match start {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let range_end = match end {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                // Empty range if lower > upper, or if any bound is excluded and lower >= upper.
                let empty_range = match (&range_start, &range_end) {
                    (Bound::Included(lower), Bound::Included(upper)) => lower > upper,
                    (Bound::Included(lower), Bound::Excluded(upper))
                    | (Bound::Excluded(lower), Bound::Included(upper))
                    | (Bound::Excluded(lower), Bound::Excluded(upper)) => lower >= upper,
                    _ => false,
                };

                let keys: Vec<_> = if empty_range {
                    Vec::new()
                } else {
                    self.index_map
                        .range((range_start, range_end))
                        .map(|(k, _v)| k.clone())
                        .collect()
                };

                metrics.record_comparisons(keys.len());

                let result = if keys.is_empty() {
                    RowAddrTreeMap::default()
                } else {
                    let bitmaps: Vec<_> = stream::iter(
                        keys.into_iter()
                            .map(|key| async move { self.load_bitmap(&key, Some(metrics)).await }),
                    )
                    .buffer_unordered(get_num_compute_intensive_cpus())
                    .try_collect()
                    .await?;

                    let bitmap_refs: Vec<_> = bitmaps.iter().map(|b| b.as_ref()).collect();
                    RowAddrTreeMap::union_all(&bitmap_refs)
                };

                (result, tracked_null_rows())
            }
            SargableQuery::IsIn(values) => {
                metrics.record_comparisons(values.len());

                // Collect keys that exist in the index, tracking if we need nulls
                let mut has_null = false;
                let keys: Vec<_> = values
                    .iter()
                    .filter_map(|val| {
                        if val.is_null() {
                            has_null = true;
                            None
                        } else {
                            let key = OrderableScalarValue(val.clone());
                            if self.index_map.contains_key(&key) {
                                Some(key)
                            } else {
                                None
                            }
                        }
                    })
                    .collect();

                // Load bitmaps in parallel
                let mut bitmaps: Vec<_> = stream::iter(
                    keys.into_iter()
                        .map(|key| async move { self.load_bitmap(&key, Some(metrics)).await }),
                )
                .buffer_unordered(get_num_compute_intensive_cpus())
                .try_collect()
                .await?;

                // Add null bitmap if needed
                if has_null && !self.null_map.is_empty() {
                    bitmaps.push(self.null_map.clone());
                }

                let result = if bitmaps.is_empty() {
                    RowAddrTreeMap::default()
                } else {
                    // Convert Arc<RowAddrTreeMap> to &RowAddrTreeMap for union_all
                    let bitmap_refs: Vec<_> = bitmaps.iter().map(|b| b.as_ref()).collect();
                    RowAddrTreeMap::union_all(&bitmap_refs)
                };

                // If the query explicitly includes null, then nulls are TRUE (not NULL)
                // Otherwise, nulls remain NULL (unknown)
                let null_rows = if has_null { None } else { tracked_null_rows() };
                (result, null_rows)
            }
            SargableQuery::IsNull() => {
                metrics.record_comparisons(1);
                // Querying FOR nulls - they are the TRUE result, not NULL result
                ((*self.null_map).clone(), None)
            }
            SargableQuery::FullTextSearch(_) => {
                return Err(Error::not_supported_source(
                    "full text search is not supported for bitmap indexes".into(),
                ));
            }
            SargableQuery::LikePrefix(_) => {
                return Err(Error::not_supported_source(
                    "LIKE prefix queries are not supported for bitmap indexes".into(),
                ));
            }
        };

        let selection = NullableRowAddrSet::new(row_ids, null_row_ids.unwrap_or_default());
        Ok(SearchResult::Exact(selection))
    }

    fn can_remap(&self) -> bool {
        true
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let mut writer =
            new_bitmap_batch_writer(dest_store, BITMAP_LOOKUP_NAME, &self.value_type).await?;
        remap_index_map(self, mapping, &mut writer).await?;
        let file = writer.finish().await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
            files: vec![file],
        })
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let file = BitmapIndexPlugin::streaming_build_and_write(
            new_data,
            Some(self),
            dest_store,
            BITMAP_LOOKUP_NAME,
            old_data_filter.as_ref(),
        )
        .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
            files: vec![file],
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::Values).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap))
    }
}

/// Buffers serialized (key, bitmap) pairs and flushes them as record batches
/// to the index file once they reach [`MAX_BUFFERED_BYTES`].
pub(crate) struct BitmapBatchWriter {
    file: Box<dyn super::IndexWriter>,
    keys: Vec<ScalarValue>,
    serialized: Vec<Vec<u8>>,
    bytes: usize,
    num_bitmaps: usize,
    /// Flush threshold. A field rather than [`MAX_BUFFERED_BYTES`] directly only
    /// so that tests can drive the multi-batch path without writing 32 MiB.
    max_buffered_bytes: usize,
    /// Record batches handed to `file` so far, so tests can assert the writer
    /// actually flushed rather than buffering everything.
    #[cfg(test)]
    batches_written: usize,
    /// Global-buffer keys and the buffer index each was written to. Recorded
    /// as file metadata at finish so readers can find them.
    buffer_indices: HashMap<String, String>,
}

impl BitmapBatchWriter {
    fn new(file: Box<dyn super::IndexWriter>) -> Self {
        Self {
            file,
            keys: Vec::new(),
            serialized: Vec::new(),
            bytes: 0,
            num_bitmaps: 0,
            max_buffered_bytes: MAX_BUFFERED_BYTES,
            #[cfg(test)]
            batches_written: 0,
            buffer_indices: HashMap::new(),
        }
    }

    #[cfg(test)]
    fn with_max_buffered_bytes(mut self, bytes: usize) -> Self {
        self.max_buffered_bytes = bytes;
        self
    }

    #[cfg(test)]
    pub(crate) fn batches_written(&self) -> usize {
        self.batches_written
    }

    /// Attach a global buffer to the file, recording its index under `key` so
    /// that a reader can find it from the file metadata.
    ///
    /// Callable at any point: the underlying writer records the current offset
    /// and writes the buffer immediately, so its position relative to the data
    /// pages does not matter. Callers here do it first only to keep the metadata
    /// setup in one place.
    pub(crate) async fn add_global_buffer(&mut self, key: String, data: Bytes) -> Result<()> {
        let buffer_idx = self.file.add_global_buffer(data).await?;
        self.buffer_indices.insert(key, buffer_idx.to_string());
        Ok(())
    }

    /// Serialize and buffer a single (key, bitmap) pair, flushing the current
    /// batch to disk if adding it would exceed [`MAX_BUFFERED_BYTES`].
    pub(crate) async fn emit(&mut self, key: ScalarValue, bitmap: &RowAddrTreeMap) -> Result<()> {
        let mut buf = Vec::new();
        bitmap.serialize_into(&mut buf).unwrap();
        let size = buf.len() + key.size();

        if self.bytes + size > self.max_buffered_bytes {
            self.flush().await?;
        }

        self.keys.push(key);
        self.serialized.push(buf);
        self.bytes += size;
        self.num_bitmaps += 1;
        Ok(())
    }

    /// Write the current batch to disk.
    async fn flush(&mut self) -> Result<()> {
        if self.keys.is_empty() {
            return Ok(());
        }
        let keys_array =
            ScalarValue::iter_to_array(self.keys.drain(..).collect::<Vec<_>>()).unwrap();
        let total_size: usize = self.serialized.iter().map(|b| b.len()).sum();
        let mut binary_builder = BinaryBuilder::with_capacity(self.serialized.len(), total_size);
        for b in self.serialized.drain(..) {
            binary_builder.append_value(&b);
        }
        let bitmaps_array = Arc::new(binary_builder.finish()) as Arc<dyn Array>;
        let batch = BitmapIndexPlugin::get_batch_from_arrays(keys_array, bitmaps_array)?;
        self.file.write_record_batch(batch).await?;
        self.bytes = 0;
        #[cfg(test)]
        {
            self.batches_written += 1;
        }
        Ok(())
    }

    /// Flush any remaining data, write index statistics and any global-buffer
    /// indices, and finalize the file.
    pub(crate) async fn finish(mut self) -> Result<IndexFile> {
        self.flush().await?;
        let stats_json = serde_json::to_string(&BitmapStatistics {
            num_bitmaps: self.num_bitmaps,
        })
        .map_err(|e| Error::internal(format!("failed to serialize bitmap statistics: {e}")))?;
        let mut metadata = std::mem::take(&mut self.buffer_indices);
        metadata.insert(INDEX_STATS_METADATA_KEY.to_string(), stats_json);
        self.file.finish_with_metadata(metadata).await
    }
}

fn bitmap_shard_file_name(partition_id: u64) -> String {
    format!("{BITMAP_PART_LOOKUP_PREFIX}{partition_id}{BITMAP_PART_LOOKUP_SUFFIX}")
}

fn tagged_bitmap_partition_id(id: u32, tag: u64) -> u64 {
    ((id as u64) << 32) | tag
}

fn bitmap_shard_partition_id(fragment_ids: &[u32], shard_id: Option<u32>) -> Result<u64> {
    if fragment_ids.is_empty() {
        return Err(Error::invalid_input(
            "Bitmap shard build requires at least one fragment id".to_string(),
        ));
    }

    if let Some(shard_id) = shard_id {
        return Ok(tagged_bitmap_partition_id(shard_id, EXPLICIT_SHARD_ID_TAG));
    }

    let [fragment_id] = fragment_ids else {
        return Err(Error::invalid_input(format!(
            "Bitmap distributed build over multiple fragments requires an explicit shard_id. \
             Received {} fragment ids: {:?}. Please assign mutually exclusive shard_id values \
             to disjoint fragment groups.",
            fragment_ids.len(),
            fragment_ids
        )));
    };

    Ok(tagged_bitmap_partition_id(
        *fragment_id,
        IMPLICIT_FRAGMENT_ID_TAG,
    ))
}

fn extract_bitmap_shard_id(filename: &str) -> Result<u64> {
    let partition_id = filename
        .strip_prefix(BITMAP_PART_LOOKUP_PREFIX)
        .and_then(|name| name.strip_suffix(BITMAP_PART_LOOKUP_SUFFIX))
        .ok_or_else(|| {
            Error::internal(format!("Invalid bitmap shard file name format: {filename}"))
        })?;
    partition_id.parse::<u64>().map_err(|_| {
        Error::internal(format!(
            "Failed to parse bitmap partition id from file name: {filename}"
        ))
    })
}

fn deserialize_bitmap(bitmap_bytes: &[u8], file_name: &str) -> Result<RowAddrTreeMap> {
    RowAddrTreeMap::deserialize_from(bitmap_bytes).map_err(|error| {
        Error::corrupt_file(
            Path::from(file_name),
            format!("Failed to deserialize bitmap bytes: {error}"),
        )
    })
}

pub(crate) async fn new_bitmap_batch_writer(
    index_store: &dyn IndexStore,
    file_name: &str,
    value_type: &DataType,
) -> Result<BitmapBatchWriter> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", value_type.clone(), true),
        Field::new("bitmaps", DataType::Binary, true),
    ]));
    let index_file = index_store.new_index_file(file_name, schema).await?;
    Ok(BitmapBatchWriter::new(index_file))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BitmapHeapItem {
    key: OrderableScalarValue,
    shard_idx: usize,
}

impl Ord for BitmapHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.shard_idx.cmp(&other.shard_idx))
    }
}

impl PartialOrd for BitmapHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct BitmapShardCursor {
    file_name: String,
    reader: Arc<dyn IndexReader>,
    total_rows: usize,
    next_row_offset: usize,
    batch: Option<RecordBatch>,
    batch_row_idx: usize,
}

impl BitmapShardCursor {
    async fn try_new(file_name: String, reader: Arc<dyn IndexReader>) -> Result<Option<Self>> {
        let total_rows = reader.num_rows();
        if total_rows == 0 {
            return Ok(None);
        }

        let mut cursor = Self {
            file_name,
            reader,
            total_rows,
            next_row_offset: 0,
            batch: None,
            batch_row_idx: 0,
        };
        if cursor.advance().await? {
            Ok(Some(cursor))
        } else {
            Ok(None)
        }
    }

    fn peek_key(&self) -> Result<OrderableScalarValue> {
        let batch = self.batch.as_ref().ok_or_else(|| {
            Error::internal(format!(
                "Bitmap shard {} has no active batch",
                self.file_name
            ))
        })?;
        let key = ScalarValue::try_from_array(batch.column(0), self.batch_row_idx)?;
        Ok(OrderableScalarValue(key))
    }

    fn take_current(&mut self) -> Result<(ScalarValue, RowAddrTreeMap)> {
        let batch = self.batch.as_ref().ok_or_else(|| {
            Error::internal(format!(
                "Bitmap shard {} has no active batch",
                self.file_name
            ))
        })?;
        let keys = batch.column(0);
        let binary_bitmaps = batch
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| {
                Error::corrupt_file(
                    Path::from(self.file_name.as_str()),
                    "Bitmap shard batch has non-binary bitmap column".to_string(),
                )
            })?;
        let key = ScalarValue::try_from_array(keys, self.batch_row_idx)?;
        let bitmap = deserialize_bitmap(binary_bitmaps.value(self.batch_row_idx), &self.file_name)?;
        self.batch_row_idx += 1;
        Ok((key, bitmap))
    }

    async fn advance(&mut self) -> Result<bool> {
        loop {
            if let Some(batch) = &self.batch
                && self.batch_row_idx < batch.num_rows()
            {
                return Ok(true);
            }

            if self.next_row_offset >= self.total_rows {
                self.batch = None;
                return Ok(false);
            }

            let end_row = (self.next_row_offset + MERGE_ROWS_PER_CHUNK).min(self.total_rows);
            let batch = self
                .reader
                .read_range(self.next_row_offset..end_row, None)
                .await?;
            self.next_row_offset = end_row;
            self.batch = Some(batch);
            self.batch_row_idx = 0;
        }
    }
}

async fn advance_cursor_and_push(
    cursors: &mut [BitmapShardCursor],
    heap: &mut BinaryHeap<Reverse<BitmapHeapItem>>,
    shard_idx: usize,
) -> Result<()> {
    if cursors[shard_idx].advance().await? {
        heap.push(Reverse(BitmapHeapItem {
            key: cursors[shard_idx].peek_key()?,
            shard_idx,
        }));
    }
    Ok(())
}

async fn drain_same_key_bitmaps(
    cursors: &mut [BitmapShardCursor],
    heap: &mut BinaryHeap<Reverse<BitmapHeapItem>>,
    item: BitmapHeapItem,
) -> Result<(ScalarValue, RowAddrTreeMap)> {
    let (key, mut merged_bitmap) = cursors[item.shard_idx].take_current()?;
    let merged_key = OrderableScalarValue(key);
    advance_cursor_and_push(cursors, heap, item.shard_idx).await?;

    while let Some(Reverse(next_item)) = heap.peek() {
        if next_item.key != merged_key {
            break;
        }

        let shard_idx = next_item.shard_idx;
        let _ = heap.pop();
        let (_, bitmap) = cursors[shard_idx].take_current()?;
        merged_bitmap |= &bitmap;
        advance_cursor_and_push(cursors, heap, shard_idx).await?;
    }

    Ok((merged_key.0, merged_bitmap))
}

/// Open a set of key-sorted bitmap files as merge cursors, seeded into a
/// min-heap on their first key, and confirm every file shares a value type.
///
/// Returns `None` for the value type only when every file was empty.
pub(crate) async fn open_sorted_bitmap_cursors(
    store: &dyn IndexStore,
    files: &[String],
) -> Result<(
    Vec<BitmapShardCursor>,
    BinaryHeap<Reverse<BitmapHeapItem>>,
    Option<DataType>,
)> {
    let mut cursors = Vec::with_capacity(files.len());
    let mut heap = BinaryHeap::with_capacity(files.len());
    let mut value_type: Option<DataType> = None;

    for file_name in files {
        let reader = store.open_index_file(file_name).await?;
        let file_value_type = reader.schema().fields[0].data_type().clone();
        if let Some(existing_type) = &value_type {
            if existing_type != &file_value_type {
                return Err(Error::invalid_input(format!(
                    "Bitmap shard {} has value type {:?}, expected {:?}",
                    file_name, file_value_type, existing_type
                )));
            }
        } else {
            value_type = Some(file_value_type);
        }
        if let Some(cursor) = BitmapShardCursor::try_new(file_name.clone(), reader).await? {
            let key = cursor.peek_key()?;
            let shard_idx = cursors.len();
            cursors.push(cursor);
            heap.push(Reverse(BitmapHeapItem { key, shard_idx }));
        }
    }

    Ok((cursors, heap, value_type))
}

/// Drain cursors opened by [`open_sorted_bitmap_cursors`] into `writer`,
/// emitting each key once in ascending order with the row sets of duplicate
/// keys unioned.
///
/// The merge's working state is one row-bounded record batch per cursor plus the
/// bitmap currently being merged, independent of the total number of keys. This
/// does not include the output writer or other state retained by the caller, and
/// the cursor batches are not byte-bounded.
pub(crate) async fn drain_sorted_bitmap_cursors(
    cursors: &mut [BitmapShardCursor],
    heap: &mut BinaryHeap<Reverse<BitmapHeapItem>>,
    writer: &mut BitmapBatchWriter,
    progress: Option<(&dyn IndexBuildProgress, &str)>,
) -> Result<()> {
    let mut merged_keys = 0u64;
    while let Some(Reverse(item)) = heap.pop() {
        let (key, merged_bitmap) = drain_same_key_bitmaps(cursors, heap, item).await?;
        writer.emit(key, &merged_bitmap).await?;
        merged_keys += 1;
        if let Some((progress, stage)) = progress {
            progress.stage_progress(stage, merged_keys).await?;
        }
    }
    Ok(())
}

async fn list_bitmap_shard_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    progress: &dyn IndexBuildProgress,
) -> Result<Vec<String>> {
    let mut shard_files = Vec::new();
    let mut list_stream = object_store.list(Some(index_dir.clone()));
    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let file_name = meta.location.filename().unwrap_or_default();
                if file_name.starts_with(BITMAP_PART_LOOKUP_PREFIX)
                    && file_name.ends_with(BITMAP_PART_LOOKUP_SUFFIX)
                {
                    shard_files.push(file_name.to_string());
                    progress
                        .stage_progress("scan_bitmap_shards", shard_files.len() as u64)
                        .await?;
                }
            }
            Err(err) => {
                return Err(Error::io(format!(
                    "Failed to list bitmap shard files in {}: {err}",
                    index_dir
                )));
            }
        }
    }
    let mut shard_files = shard_files
        .into_iter()
        .map(|file_name| extract_bitmap_shard_id(&file_name).map(|shard_id| (shard_id, file_name)))
        .collect::<Result<Vec<_>>>()?;
    shard_files.sort_unstable_by_key(|(shard_id, _)| *shard_id);
    let shard_files = shard_files
        .into_iter()
        .map(|(_, file_name)| file_name)
        .collect::<Vec<_>>();
    if shard_files.is_empty() {
        return Err(Error::invalid_input(format!(
            "No bitmap shard files found in index directory: {}; \
             call build_index for each fragment before calling merge_index_metadata",
            index_dir
        )));
    }
    Ok(shard_files)
}

async fn cleanup_bitmap_shard_files(store: &dyn IndexStore, shard_files: &[String]) {
    for file_name in shard_files {
        if let Err(error) = store.delete_index_file(file_name).await {
            warn!(
                "Failed to delete bitmap shard file '{}': {}. \
                 This does not affect the merged bitmap index, but the shard file \
                 may need manual cleanup.",
                file_name, error
            );
        }
    }
}

#[derive(Debug, Default)]
pub struct BitmapIndexPlugin;

/// Drop the rows an old posting should no longer expose -- rows whose fragment
/// was removed, or (under stable row ids) rows rewritten by an update -- keeping
/// only those `filter` still considers valid. A no-op when `filter` is `None`.
fn retain_valid(
    mut bitmap: RowAddrTreeMap,
    filter: Option<&super::OldIndexDataFilter>,
) -> RowAddrTreeMap {
    if let Some(filter) = filter {
        filter.retain_old_rows(&mut bitmap);
    }
    bitmap
}

impl BitmapIndexPlugin {
    fn get_batch_from_arrays(
        keys: Arc<dyn Array>,
        binary_bitmaps: Arc<dyn Array>,
    ) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("keys", keys.data_type().clone(), true),
            Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
        ]));

        let columns = vec![keys, binary_bitmaps];

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub async fn train_bitmap_index(
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
    ) -> Result<IndexFile> {
        Self::streaming_build_and_write(data, None, index_store, BITMAP_LOOKUP_NAME, None).await
    }

    async fn train_bitmap_shard(
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        fragment_ids: &[u32],
        shard_id: Option<u32>,
        progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<IndexFile> {
        let partition_id = bitmap_shard_partition_id(fragment_ids, shard_id)?;
        let file_name = bitmap_shard_file_name(partition_id);
        progress
            .stage_start("build_bitmap_shard", None, "rows")
            .await?;
        let file =
            Self::streaming_build_and_write(data, None, index_store, &file_name, None).await?;
        progress.stage_complete("build_bitmap_shard").await?;
        Ok(file)
    }

    /// Builds and writes a bitmap index in a streaming fashion from value-sorted
    /// input. Only one new value's aggregate bitmap is held at a time instead of
    /// an aggregate map containing every value. The input pipeline, an existing
    /// index and its cache, and the output writer retain separate memory.
    ///
    /// If `old_index` is provided, its existing bitmaps are merged with the new
    /// data via a sorted merge-join (the old index_map is a BTreeMap, already
    /// sorted by value).
    async fn streaming_build_and_write(
        mut data_source: SendableRecordBatchStream,
        old_index: Option<&BitmapIndex>,
        index_store: &dyn IndexStore,
        output_file_name: &str,
        old_data_filter: Option<&super::OldIndexDataFilter>,
    ) -> Result<IndexFile> {
        let value_type = data_source.schema().field(0).data_type().clone();

        let mut writer =
            new_bitmap_batch_writer(index_store, output_file_name, &value_type).await?;

        // Collect old index keys (already in memory as BTreeMap keys — this is
        // just a Vec of references, not a copy of the bitmaps themselves).
        let old_keys: Vec<OrderableScalarValue> = old_index
            .map(|idx| idx.index_map.keys().cloned().collect())
            .unwrap_or_default();
        let mut old_pos: usize = 0;

        // Current value being accumulated from the new data stream.
        let mut current_key: Option<ScalarValue> = None;
        let mut current_bitmap = RowAddrTreeMap::default();
        // Track whether we emitted a null bitmap (old index stores nulls
        // separately in null_map, not in index_map).
        let mut emitted_null = false;

        while let Some(batch) = data_source.try_next().await? {
            let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
            let row_ids = batch.column_by_name(ROW_ID).expect_ok()?;
            debug_assert_eq!(row_ids.data_type(), &DataType::UInt64);
            let row_id_column = row_ids.as_any().downcast_ref::<UInt64Array>().unwrap();

            for i in 0..values.len() {
                let row_id = row_id_column.value(i);
                let key = ScalarValue::try_from_array(values.as_ref(), i)?;

                match &current_key {
                    Some(cur) if *cur == key => {
                        current_bitmap.insert(row_id);
                    }
                    _ => {
                        // Value changed — flush the previous run.
                        if let Some(prev_key) = current_key.take() {
                            let mut prev_bitmap = std::mem::take(&mut current_bitmap);
                            Self::finish_run(
                                prev_key,
                                &mut prev_bitmap,
                                old_index,
                                &old_keys,
                                &mut old_pos,
                                &mut emitted_null,
                                &mut writer,
                                old_data_filter,
                            )
                            .await?;
                        }
                        current_key = Some(key);
                        current_bitmap = RowAddrTreeMap::default();
                        current_bitmap.insert(row_id);
                    }
                }
            }
        }

        // Flush the last accumulated run from new data.
        if let Some(last_key) = current_key.take() {
            let mut last_bitmap = std::mem::take(&mut current_bitmap);
            Self::finish_run(
                last_key,
                &mut last_bitmap,
                old_index,
                &old_keys,
                &mut old_pos,
                &mut emitted_null,
                &mut writer,
                old_data_filter,
            )
            .await?;
        }

        // Emit any remaining old-only entries.
        if let Some(idx) = old_index {
            while old_pos < old_keys.len() {
                let old_bitmap = retain_valid(
                    idx.load_bitmap(&old_keys[old_pos], None)
                        .await?
                        .as_ref()
                        .clone(),
                    old_data_filter,
                );
                writer
                    .emit(old_keys[old_pos].0.clone(), &old_bitmap)
                    .await?;
                old_pos += 1;
            }
        }

        // Emit old null bitmap if we didn't already merge it with new nulls.
        if !emitted_null
            && let Some(idx) = old_index
            && !idx.null_map.is_empty()
        {
            let null_key = new_null_array(&value_type, 1);
            let null_key = ScalarValue::try_from_array(null_key.as_ref(), 0)?;
            let null_bitmap = retain_valid((*idx.null_map).clone(), old_data_filter);
            writer.emit(null_key, &null_bitmap).await?;
        }

        writer.finish().await
    }

    /// Flush a completed value-run from the new data stream, emitting any
    /// old-only entries that sort before it and merging the old bitmap if the
    /// key exists in both old and new.
    #[allow(clippy::too_many_arguments)]
    async fn finish_run(
        key: ScalarValue,
        bitmap: &mut RowAddrTreeMap,
        old_index: Option<&BitmapIndex>,
        old_keys: &[OrderableScalarValue],
        old_pos: &mut usize,
        emitted_null: &mut bool,
        writer: &mut BitmapBatchWriter,
        old_data_filter: Option<&super::OldIndexDataFilter>,
    ) -> Result<()> {
        if key.is_null() {
            // Null values are stored separately in the old index's null_map.
            if let Some(idx) = old_index
                && !idx.null_map.is_empty()
            {
                *bitmap |= &retain_valid((*idx.null_map).clone(), old_data_filter);
            }
            *emitted_null = true;
            writer.emit(key, bitmap).await?;
        } else if let Some(idx) = old_index {
            let orderable = OrderableScalarValue(key.clone());

            // Emit old-only entries that sort before this key.
            while *old_pos < old_keys.len() && old_keys[*old_pos] < orderable {
                let old_bitmap = retain_valid(
                    idx.load_bitmap(&old_keys[*old_pos], None)
                        .await?
                        .as_ref()
                        .clone(),
                    old_data_filter,
                );
                writer
                    .emit(old_keys[*old_pos].0.clone(), &old_bitmap)
                    .await?;
                *old_pos += 1;
            }

            // If the old index also has this key, merge its bitmap.
            if *old_pos < old_keys.len() && old_keys[*old_pos] == orderable {
                *bitmap |= &retain_valid(
                    idx.load_bitmap(&old_keys[*old_pos], None)
                        .await?
                        .as_ref()
                        .clone(),
                    old_data_filter,
                );
                *old_pos += 1;
            }

            writer.emit(key, bitmap).await?;
        } else {
            writer.emit(key, bitmap).await?;
        }
        Ok(())
    }

    /// Merge per-shard bitmap lookup files into a single bitmap index file.
    ///
    /// Each shard file is already sorted by key and can contain many distinct keys.
    /// This method does not materialize an entire shard in memory. Instead, it keeps
    /// one cursor per shard, where each cursor tracks the shard's current row within
    /// a small in-memory batch. A min-heap stores the current key for each shard.
    ///
    /// The merge then proceeds as a streaming K-way merge:
    /// - pop the smallest current key across all shards
    /// - union the bitmap for that key with any other shards currently positioned on
    ///   the same key
    /// - advance only those shards that participated in the union and push their next
    ///   keys back into the heap
    ///
    /// The merge-specific working state is proportional to the number of shards
    /// plus the bitmaps currently being merged, instead of the total number of
    /// keys across all shards. This is not a total-memory or byte-bound claim.
    async fn merge_shards(
        store: &dyn IndexStore,
        shard_files: &[String],
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<IndexFile> {
        progress
            .stage_start("merge_bitmap_shards", None, "bitmaps")
            .await?;

        let (mut cursors, mut heap, value_type) =
            open_sorted_bitmap_cursors(store, shard_files).await?;

        let value_type = value_type.ok_or_else(|| {
            Error::invalid_input("Bitmap shard merge requires at least one shard file".to_string())
        })?;
        let mut writer = new_bitmap_batch_writer(store, BITMAP_LOOKUP_NAME, &value_type).await?;

        drain_sorted_bitmap_cursors(
            &mut cursors,
            &mut heap,
            &mut writer,
            Some((progress.as_ref(), "merge_bitmap_shards")),
        )
        .await?;

        progress.stage_complete("merge_bitmap_shards").await?;
        progress
            .stage_start("write_bitmap_index", Some(1), "files")
            .await?;
        let file = writer.finish().await?;
        progress.stage_progress("write_bitmap_index", 1).await?;
        progress.stage_complete("write_bitmap_index").await?;
        Ok(file)
    }
}

pub async fn merge_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<()> {
    progress
        .stage_start("scan_bitmap_shards", None, "files")
        .await?;
    let shard_files = list_bitmap_shard_files(object_store, index_dir, progress.as_ref()).await?;
    progress.stage_complete("scan_bitmap_shards").await?;

    BitmapIndexPlugin::merge_shards(store.as_ref(), &shard_files, progress).await?;
    cleanup_bitmap_shard_files(store.as_ref(), &shard_files).await;
    Ok(())
}

/// Apply `mapping` to every row address in `index` without materializing every
/// bitmap payload at once, writing the result through `writer`.
///
/// This helper's transient aggregation state is one bitmap at a time. The
/// loaded `index_map`, index cache, mapping, and output writer remain resident
/// separately. Nulls live outside `index_map`, in `null_map`, so they are
/// remapped separately and emitted first -- a null sorts below every value.
pub(crate) async fn remap_index_map(
    index: &BitmapIndex,
    mapping: &RowAddrRemap,
    writer: &mut BitmapBatchWriter,
) -> Result<()> {
    if !index.null_map.is_empty() {
        let null_key = new_null_array(index.value_type(), 1);
        let null_key = ScalarValue::try_from_array(null_key.as_ref(), 0)?;
        writer
            .emit(null_key, &remap_row_addrs(&index.null_map, mapping))
            .await?;
    }

    for key in index.index_map.keys() {
        let bitmap = index.load_bitmap(key, None).await?;
        writer
            .emit(key.0.clone(), &remap_row_addrs(&bitmap, mapping))
            .await?;
    }

    Ok(())
}

pub(crate) fn remap_row_addrs(bitmap: &RowAddrTreeMap, mapping: &RowAddrRemap) -> RowAddrTreeMap {
    RowAddrTreeMap::from_iter(bitmap.row_addrs().unwrap().filter_map(|addr| {
        let addr_as_u64 = u64::from(addr);
        mapping.get(addr_as_u64).unwrap_or(Some(addr_as_u64))
    }))
}

/// Total source entries a merge of `sources` will consume.
///
/// The exact denominator for the merge's progress: known before the merge
/// starts, at no I/O cost since every `index_map` is already loaded, and reached
/// exactly, because the merge drains each source's keys and advances every one
/// of them once. Counting entries rather than emitted keys is what makes it
/// exact -- a key held by three sources costs three loads and yields one output
/// row, and a key whose rows are all retired by `old_data_filter` yields none.
///
/// Nulls are excluded, matching how the merge treats them everywhere else: they
/// live in `null_map`, outside `index_map`, and are unioned in one step.
pub(crate) fn merge_source_entry_count(sources: &[Arc<BitmapIndex>]) -> u64 {
    sources.iter().map(|s| s.index_map.len() as u64).sum()
}

/// Merge loaded bitmap indexes into `writer` without materializing all source
/// bitmap payloads at once.
///
/// Drives each source through its `index_map` -- a sorted `BTreeMap` rebuilt at
/// load time -- rather than through the rows of its file. LabelList index files
/// written before spill-based builds landed are unsorted on disk, so file order
/// cannot be trusted for them; `index_map` order can, for old and new files
/// alike, which is what makes this work without an index version bump.
///
/// Null keys live outside `index_map`, in each source's `null_map`, so they are
/// unioned separately and emitted first -- a null sorts below every value.
///
/// The merge's transient aggregation state is the merged bitmap for the current
/// key plus one loaded bitmap per participating source. Each source `index_map`,
/// any bitmaps retained by the index cache, and the output writer remain outside
/// that state.
///
/// `progress` reports source entries consumed, against the total from
/// [`merge_source_entry_count`]. Not segments: the merge is key-driven and
/// touches every source on every key, so no source is ever "done" to report.
/// Not emitted keys either: those have no denominator that can be known up front
/// or reached exactly. Entries have both, and are the unit the merge's cost is
/// actually in, since it loads one bitmap per entry.
pub(crate) async fn merge_index_maps(
    sources: &[Arc<BitmapIndex>],
    old_data_filter: Option<&super::OldIndexDataFilter>,
    writer: &mut BitmapBatchWriter,
    progress: Option<(&dyn IndexBuildProgress, &str)>,
) -> Result<()> {
    let Some(first) = sources.first() else {
        return Ok(());
    };
    let value_type = first.value_type().clone();

    let mut merged_nulls = RowAddrTreeMap::default();
    for source in sources {
        merged_nulls |= source.null_map.as_ref();
    }
    let merged_nulls = retain_valid(merged_nulls, old_data_filter);
    if !merged_nulls.is_empty() {
        let null_key = new_null_array(&value_type, 1);
        let null_key = ScalarValue::try_from_array(null_key.as_ref(), 0)?;
        writer.emit(null_key, &merged_nulls).await?;
    }

    let mut consumed = 0u64;

    let mut key_iters: Vec<_> = sources
        .iter()
        .map(|source| source.index_map.keys())
        .collect();

    // Every source is sorted, so the smallest key any of them is currently
    // positioned on is the next key overall. A min-heap holding one entry per
    // live source finds it in `log(sources)`, where scanning every source's
    // current key -- twice, once to select and once to consume -- cost
    // `O(keys x sources)`: with one segment per fragment that turns a linear
    // merge into billions of comparisons. It is the same merge
    // `drain_sorted_bitmap_cursors` runs over file-backed cursors.
    //
    // Entries borrow their key from the source's `index_map` rather than cloning
    // it. Seeding the heap touches every source key, so cloning here would cost
    // more than the scan it replaces whenever there are only a few sources.
    let mut heap: BinaryHeap<Reverse<(&OrderableScalarValue, usize)>> =
        BinaryHeap::with_capacity(key_iters.len());
    for (source_idx, keys) in key_iters.iter_mut().enumerate() {
        if let Some(key) = keys.next() {
            heap.push(Reverse((key, source_idx)));
        }
    }

    while let Some(Reverse((next_key, _))) = heap.peek().copied() {
        let mut merged = RowAddrTreeMap::default();

        // Drain the sources positioned on this key -- only those, where the
        // previous scan visited every source on every key. A source's next key is
        // strictly greater, so re-pushing it cannot re-enter this loop.
        while let Some(Reverse((key, source_idx))) = heap.peek().copied() {
            if key != next_key {
                break;
            }
            heap.pop();
            consumed += 1;
            merged |= sources[source_idx].load_bitmap(key, None).await?.as_ref();
            if let Some(next) = key_iters[source_idx].next() {
                heap.push(Reverse((next, source_idx)));
            }
        }

        let merged = retain_valid(merged, old_data_filter);
        if !merged.is_empty() {
            writer.emit(next_key.0.clone(), &merged).await?;
        }

        // Reported outside the guard above: a key the filter emptied still
        // consumed its source entries, and skipping it would stall the count.
        if let Some((progress, stage)) = progress {
            progress.stage_progress(stage, consumed).await?;
        }
    }

    Ok(())
}

pub async fn merge_bitmap_indices(
    source_indices: &[Arc<BitmapIndex>],
    dest_store: &dyn IndexStore,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<CreatedIndex> {
    if source_indices.is_empty() {
        return Err(Error::invalid_input(
            "Bitmap segment merge requires at least one source segment".to_string(),
        ));
    }

    let value_type = source_indices[0].value_type().clone();

    progress
        .stage_start(
            "merge_bitmap_segments",
            Some(merge_source_entry_count(source_indices)),
            "index entries",
        )
        .await?;
    for source_index in source_indices.iter() {
        if source_index.value_type() != &value_type {
            return Err(Error::invalid_input(format!(
                "Bitmap segment has value type {:?}, expected {:?}",
                source_index.value_type(),
                value_type
            )));
        }
    }

    let mut writer = new_bitmap_batch_writer(dest_store, BITMAP_LOOKUP_NAME, &value_type).await?;
    merge_index_maps(
        source_indices,
        None,
        &mut writer,
        Some((progress.as_ref(), "merge_bitmap_segments")),
    )
    .await?;
    progress.stage_complete("merge_bitmap_segments").await?;

    progress
        .stage_start("write_bitmap_index", Some(1), "files")
        .await?;
    let file = writer.finish().await?;
    progress.stage_progress("write_bitmap_index", 1).await?;
    progress.stage_complete("write_bitmap_index").await?;

    Ok(CreatedIndex {
        index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default()).unwrap(),
        index_version: BITMAP_INDEX_VERSION,
        files: vec![file],
    })
}

#[async_trait]
impl BasicTrainer for BitmapIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::invalid_input_source(
                "A bitmap index can only be created on a non-nested field.".into(),
            ));
        }
        let params = if params.is_empty() {
            BitmapParameters::default()
        } else {
            serde_json::from_str::<BitmapParameters>(params)?
        };
        Ok(Box::new(BitmapTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = request
            .as_any()
            .downcast_ref::<BitmapTrainingRequest>()
            .ok_or_else(|| {
                Error::internal(
                    "BitmapIndexPlugin::train_index received a non-bitmap training request"
                        .to_string(),
                )
            })?;
        let file = if let Some(fragment_ids) = fragment_ids.as_ref() {
            Self::train_bitmap_shard(
                data,
                index_store,
                fragment_ids,
                request.parameters.shard_id,
                progress,
            )
            .await?
        } else if request.parameters.shard_id.is_some() {
            return Err(Error::invalid_input(
                "Bitmap shard_id requires fragment_ids and is only supported for distributed shard builds"
                    .to_string(),
            ));
        } else {
            Self::train_bitmap_index(data, index_store).await?
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default())
                .unwrap(),
            index_version: BITMAP_INDEX_VERSION,
            files: vec![file],
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for BitmapIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "Bitmap"
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        BITMAP_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // Bitmap indexes cannot answer `LikePrefix` queries (see `search`), so the parser
        // is configured to skip them and let such predicates fall back to ordinary filtering.
        Some(Box::new(
            SargableQueryParser::new(index_name, self.name().to_string(), false)
                .without_like_prefix(),
        ))
    }

    /// Load an index from storage
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(BitmapIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }

    async fn get_from_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Option<Arc<dyn ScalarIndex>>> {
        let Some(state) = cache.get_with_key(&BitmapIndexStateKey).await else {
            return Ok(None);
        };
        let index = state.to_bitmap_index(index_store, cache, frag_reuse_index)?;
        Ok(Some(index as Arc<dyn ScalarIndex>))
    }

    async fn put_in_cache(&self, cache: &LanceCache, index: Arc<dyn ScalarIndex>) -> Result<()> {
        let state = BitmapIndexState::from_scalar_index(index.as_ref())?;
        cache
            .insert_with_key(&BitmapIndexStateKey, Arc::new(state))
            .await;
        Ok(())
    }

    async fn get_or_insert_in_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
        load: ScalarIndexLoad<'_>,
    ) -> Result<Arc<dyn ScalarIndex>> {
        single_flight_open(
            cache,
            BitmapIndexStateKey,
            load,
            BitmapIndexState::from_scalar_index,
            move |state| {
                Ok(state.to_bitmap_index(index_store, cache, frag_reuse_index)?
                    as Arc<dyn ScalarIndex>)
            },
        )
        .await
    }

    async fn load_statistics(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
    ) -> Result<Option<serde_json::Value>> {
        let reader = index_store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        if let Some(value) = reader.schema().metadata.get(INDEX_STATS_METADATA_KEY) {
            let stats = serde_json::from_str(value).map_err(|e| {
                Error::internal(format!("failed to parse bitmap statistics metadata: {e}"))
            })?;
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }
}

/// Fixtures shared by the tests of every module that writes this file format.
#[cfg(test)]
pub(crate) mod test_util {
    use std::sync::Arc;

    use arrow_array::{Array, BinaryArray};
    use datafusion_common::ScalarValue;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use lance_select::RowAddrTreeMap;

    use crate::scalar::IndexStore;
    use crate::scalar::lance_format::LanceIndexStore;

    /// A local index store in a fresh temporary directory. The directory is
    /// returned because dropping it deletes the store.
    pub fn index_store() -> (TempObjDir, Arc<dyn IndexStore>) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        (tmpdir, store)
    }

    /// Every `(key, bitmap)` row of a bitmap-shaped index file, in file order.
    ///
    /// File order matters to callers: the build path emits keys ascending, while
    /// indexes written before spill-based builds are in arbitrary order, and some
    /// tests assert on which of the two they are looking at.
    pub async fn read_key_bitmaps(
        store: &dyn IndexStore,
        file_name: &str,
    ) -> Vec<(Option<String>, RowAddrTreeMap)> {
        let reader = store.open_index_file(file_name).await.unwrap();
        let total = reader.num_rows();
        if total == 0 {
            return Vec::new();
        }
        let batch = reader.read_range(0..total, None).await.unwrap();
        let bitmaps = batch
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        (0..batch.num_rows())
            .map(|idx| {
                let key = match ScalarValue::try_from_array(batch.column(0), idx).unwrap() {
                    ScalarValue::Utf8(value) => value,
                    other => panic!("unexpected key type {other:?}"),
                };
                let bitmap = RowAddrTreeMap::deserialize_from(bitmaps.value(idx)).unwrap();
                (key, bitmap)
            })
            .collect()
    }

    /// A bitmap's row addresses, ascending. Empty for a bitmap with no
    /// enumerable addresses.
    pub fn row_addrs(bitmap: &RowAddrTreeMap) -> Vec<u64> {
        bitmap
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow_array::{RecordBatch, StringArray, UInt64Array, record_batch};
    use arrow_schema::{DataType, Field, Schema};

    /// Sort a (value, row_id) RecordBatch by the value column so that unit tests
    /// match the ordering the production scanner applies via TrainingOrdering::Values.
    fn sort_batch_by_value(batch: &RecordBatch) -> RecordBatch {
        use arrow::compute::SortOptions;
        let values = batch.column(0);
        let row_ids = batch.column(1);
        let options = SortOptions {
            descending: false,
            nulls_first: true,
        };
        let indices = arrow::compute::sort_to_indices(values, Some(options), None).unwrap();
        let sorted_values = arrow::compute::take(values.as_ref(), &indices, None).unwrap();
        let sorted_row_ids = arrow::compute::take(row_ids.as_ref(), &indices, None).unwrap();
        RecordBatch::try_new(batch.schema(), vec![sorted_values, sorted_row_ids]).unwrap()
    }
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::utils::{address::RowAddress, tempfile::TempObjDir};
    use lance_io::object_store::ObjectStore;
    use lance_select::RowSetOps;
    use rstest::rstest;

    fn assert_state_roundtrips(state: &BitmapIndexState) {
        let mut buf = Vec::new();
        state
            .serialize(&mut CacheEntryWriter::new(&mut buf))
            .unwrap();
        let data = bytes::Bytes::from(buf);
        let mut reader = CacheEntryReader::new(&data, 0, BitmapIndexState::CURRENT_VERSION);
        let restored = BitmapIndexState::deserialize(&mut reader).unwrap();
        assert_eq!(restored.lookup_batch, state.lookup_batch);
        assert_eq!(&*restored.null_map, &*state.null_map);
        assert_eq!(restored.value_type, state.value_type);
    }

    #[test]
    fn test_bitmap_index_state_codec_roundtrip() {
        // Non-empty state with a few keys and a populated null map.
        let mut index_map = BTreeMap::new();
        index_map.insert(OrderableScalarValue(ScalarValue::Int32(Some(1))), 0);
        index_map.insert(OrderableScalarValue(ScalarValue::Int32(Some(7))), 1);
        index_map.insert(OrderableScalarValue(ScalarValue::Int32(Some(42))), 2);
        let mut null_map = RowAddrTreeMap::new();
        null_map.insert(RowAddress::new_from_parts(0, 3).into());
        null_map.insert(RowAddress::new_from_parts(0, 5).into());
        let state = BitmapIndexState {
            lookup_batch: build_lookup_batch(&index_map, &DataType::Int32).unwrap(),
            null_map: Arc::new(null_map),
            value_type: DataType::Int32,
            index_map: Arc::new(index_map),
        };
        assert_state_roundtrips(&state);

        // Empty state: no keys, empty null map. Schema still carries the type.
        let empty_state = BitmapIndexState {
            lookup_batch: build_lookup_batch(&BTreeMap::new(), &DataType::Utf8).unwrap(),
            null_map: Arc::new(RowAddrTreeMap::new()),
            value_type: DataType::Utf8,
            index_map: Arc::new(BTreeMap::new()),
        };
        assert_state_roundtrips(&empty_state);
    }

    /// The lookup batch must decode zero-copy through the full envelope-bearing
    /// [`CacheCodec`] even though the envelope pushes the IPC section to a
    /// non-aligned starting offset.
    #[test]
    fn test_bitmap_index_state_lookup_is_zero_copy() {
        const ALIGN: usize = 64;
        let mut index_map = BTreeMap::new();
        for k in 0..32i32 {
            index_map.insert(
                OrderableScalarValue(ScalarValue::Int32(Some(k))),
                k as usize,
            );
        }
        let state = BitmapIndexState {
            lookup_batch: build_lookup_batch(&index_map, &DataType::Int32).unwrap(),
            null_map: Arc::new(RowAddrTreeMap::new()),
            value_type: DataType::Int32,
            index_map: Arc::new(index_map),
        };

        let codec = CacheCodec::from_impl::<BitmapIndexState>();
        let any: Arc<dyn std::any::Any + Send + Sync> = Arc::new(state);
        let mut buf = Vec::new();
        codec.serialize(&any, &mut buf).unwrap();

        // Model a backend reading into a 64-byte-aligned buffer.
        let mut v = vec![0u8; buf.len() + ALIGN];
        let pad = (ALIGN - (v.as_ptr() as usize % ALIGN)) % ALIGN;
        v[pad..pad + buf.len()].copy_from_slice(&buf);
        let data = bytes::Bytes::from(v).slice(pad..pad + buf.len());

        let restored = codec.deserialize(&data).hit().unwrap();
        let restored = restored.downcast::<BitmapIndexState>().unwrap();

        let base = data.as_ptr() as usize;
        let end = base + data.len();
        for col in restored.lookup_batch.columns() {
            for buffer in col.to_data().buffers() {
                let ptr = buffer.as_ptr() as usize;
                assert!(
                    ptr >= base && ptr < end,
                    "lookup batch buffer was realigned out of the input — misaligned IPC section",
                );
            }
        }
    }

    #[tokio::test]
    async fn test_bitmap_cache_key_uses_row_offset_identity() {
        let cache = LanceCache::with_capacity(1024);
        let first = BitmapKey::try_new(3).unwrap();
        let second = BitmapKey::try_new(4).unwrap();

        cache
            .insert_with_key(&first, Arc::new(RowAddrTreeMap::default()))
            .await;

        assert!(cache.get_with_key(&second).await.is_none());
    }

    #[tokio::test]
    async fn test_bitmap_lazy_loading_and_cache() {
        // Create a temporary directory for the index
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data with low cardinality column
        let colors = vec![
            "red", "blue", "green", "red", "yellow", "blue", "red", "green", "blue", "yellow",
            "red", "red", "blue", "green", "yellow",
        ];

        let row_ids = (0u64..15u64).collect::<Vec<_>>();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(colors.clone())),
                Arc::new(UInt64Array::from(row_ids.clone())),
            ],
        )
        .unwrap();

        let batch = sort_batch_by_value(&batch);
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        // Create a cache with limited capacity
        let cache = LanceCache::with_capacity(1024 * 1024); // 1MB cache

        // Load the index (should only load metadata, not bitmaps)
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        assert_eq!(index.index_map.len(), 4); // 4 non-null unique values (red, blue, green, yellow)
        assert!(index.null_map.is_empty()); // No nulls in test data

        // Test 1: Search for "red"
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("red".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Verify results
        let expected_red_rows = vec![0u64, 3, 6, 10, 11];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(|id| id.into())
                .collect();
            actual.sort();
            assert_eq!(actual, expected_red_rows);
        } else {
            panic!("Expected exact search result");
        }

        // Test 2: Search for "red" again - should hit cache
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(|id| id.into())
                .collect();
            actual.sort();
            assert_eq!(actual, expected_red_rows);
        }

        // Test 3: Range query
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Utf8(Some("blue".to_string()))),
            std::ops::Bound::Included(ScalarValue::Utf8(Some("green".to_string()))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        let expected_range_rows = vec![1u64, 2, 5, 7, 8, 12, 13];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(|id| id.into())
                .collect();
            actual.sort();
            assert_eq!(actual, expected_range_rows);
        }

        // Test 3b: Inverted range query should return empty result
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Utf8(Some("green".to_string()))),
            std::ops::Bound::Included(ScalarValue::Utf8(Some("blue".to_string()))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        if let SearchResult::Exact(row_ids) = result {
            assert!(row_ids.true_rows().is_empty());
        } else {
            panic!("Expected exact search result");
        }

        // Test 4: IsIn query
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Utf8(Some("red".to_string())),
            ScalarValue::Utf8(Some("yellow".to_string())),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        let expected_in_rows = vec![0u64, 3, 4, 6, 9, 10, 11, 14];
        if let SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(|id| id.into())
                .collect();
            actual.sort();
            assert_eq!(actual, expected_in_rows);
        }
    }

    /// Regression test for the review fix that gates `load_bitmap` on
    /// `index_map.contains_key` before recording a miss: a value that is
    /// not present in the index must short-circuit before touching the
    /// per-query cache counters. Previously an Equals query for a missing
    /// value would silently bump `index_cache_misses` and `parts_loaded`
    /// on every call even though no bitmap page was actually loaded.
    #[tokio::test]
    async fn test_bitmap_absent_value_records_no_cache_activity() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let colors = vec!["red", "blue", "green", "yellow"];
        let row_ids = (0u64..4u64).collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(colors)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();
        let batch = sort_batch_by_value(&batch);
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        // Keep the `LanceCache` alive in test scope so the `WeakLanceCache`
        // inside `BitmapIndex` can upgrade during search.
        let cache = LanceCache::with_capacity(1024 * 1024);
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        // Equals on a value that is not in `index_map` must not touch
        // the cache counters and must not report a part load.
        let metrics = LocalMetricsCollector::default();
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("purple".to_string())));
        let result = index.search(&query, &metrics).await.unwrap();
        if let SearchResult::Exact(row_ids) = result {
            assert!(row_ids.true_rows().is_empty());
        } else {
            panic!("Expected exact search result");
        }
        assert_eq!(
            metrics.index_cache_hits(),
            0,
            "absent value must not record any cache hits",
        );
        assert_eq!(
            metrics.index_cache_misses(),
            0,
            "absent value must not record a cache miss (no loader ran)",
        );

        // IsIn covering only absent values also stays at 0/0.
        let metrics = LocalMetricsCollector::default();
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Utf8(Some("purple".to_string())),
            ScalarValue::Utf8(Some("teal".to_string())),
        ]);
        let result = index.search(&query, &metrics).await.unwrap();
        if let SearchResult::Exact(row_ids) = result {
            assert!(row_ids.true_rows().is_empty());
        } else {
            panic!("Expected exact search result");
        }
        assert_eq!(metrics.index_cache_hits(), 0);
        assert_eq!(metrics.index_cache_misses(), 0);

        // Sanity: a present value on the same cold cache still records
        // exactly one miss, proving the counters are wired up and the
        // absent-value path above is not silently no-op.
        let metrics = LocalMetricsCollector::default();
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("red".to_string())));
        index.search(&query, &metrics).await.unwrap();
        assert_eq!(metrics.index_cache_hits(), 0);
        assert_eq!(metrics.index_cache_misses(), 1);
    }

    // Regression test for the O(N log N) warm-cache rebuild introduced in
    // commit 4de5ce67d.  BitmapIndexState now caches the parsed Arc<BTreeMap>
    // so that get_from_cache skips parse_lookup_batch on warm hits.
    // IS NULL is the worst case: the actual bitmap lookup is O(1) but
    // reconstruction of the BTreeMap touched every row in the lookup batch.
    #[tokio::test]
    async fn test_bitmap_cache_fast_path() {
        use arrow_array::Int32Array;

        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // High-cardinality: 1 000 unique integers + 5 null rows.
        const N: u64 = 1_000;
        const NULL_COUNT: u64 = 5;
        // nulls first (sorted batch: nulls precede values)
        let null_values: Vec<Option<i32>> =
            std::iter::repeat_n(None, NULL_COUNT as usize).collect();
        let non_null_values: Vec<Option<i32>> = (0..N as i32).map(Some).collect();
        let all_values: Vec<Option<i32>> = null_values.into_iter().chain(non_null_values).collect();
        let all_row_ids: Vec<u64> = (0..N + NULL_COUNT).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(all_values)),
                Arc::new(UInt64Array::from(all_row_ids)),
            ],
        )
        .unwrap();
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        let cache = LanceCache::with_capacity(16 * 1024 * 1024);
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        let plugin = BitmapIndexPlugin;
        let index_arc: Arc<dyn ScalarIndex> = index.clone() as Arc<dyn ScalarIndex>;
        plugin.put_in_cache(&cache, index_arc).await.unwrap();

        // get_from_cache must return Some, and the BitmapIndexState's OnceLock
        // must have been populated by put_in_cache so no parse_lookup_batch occurs.
        let cached = plugin
            .get_from_cache(store.clone(), None, &cache)
            .await
            .unwrap()
            .expect("get_from_cache must return Some after put_in_cache");

        // IS NULL: trivial work once the index is in hand.
        let query = SargableQuery::IsNull();
        match cached.search(&query, &NoOpMetricsCollector).await.unwrap() {
            SearchResult::Exact(row_set) => {
                let mut null_rows: Vec<u64> = row_set
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                null_rows.sort();
                let expected: Vec<u64> = (0..NULL_COUNT).collect();
                assert_eq!(null_rows, expected);
            }
            _ => panic!("Expected Exact result for IS NULL"),
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_big_bitmap_index() {
        // WARNING: This test allocates a huge state to force overflow over int32 on BinaryArray
        // You must run it only on a machine with enough resources (or skip it normally).
        use super::{BITMAP_LOOKUP_NAME, BitmapIndex};
        use crate::scalar::IndexStore;
        use crate::scalar::lance_format::LanceIndexStore;
        use arrow_schema::DataType;
        use datafusion_common::ScalarValue;
        use lance_core::cache::LanceCache;
        use lance_io::object_store::ObjectStore;
        use lance_select::RowAddrTreeMap;
        use std::sync::Arc;

        // Adjust these numbers so that:
        //     m * (serialized size per bitmap) > 2^31 bytes.
        //
        // For example, if we assume each bitmap serializes to ~1000 bytes,
        // you need m > 2.1e6.
        let m: u32 = 2_500_000;
        let per_bitmap_size = 1000; // assumed bytes per bitmap

        // Create a temporary store.
        let tmpdir = TempObjDir::default();
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        );

        // This should never trigger a "byte array offset overflow" error, since
        // the writer flushes a batch once it reaches MAX_BUFFERED_BYTES, which is
        // far below the i32 offset ceiling of either output column.
        let mut writer =
            new_bitmap_batch_writer(&test_store, BITMAP_LOOKUP_NAME, &DataType::UInt32)
                .await
                .unwrap();
        for i in 0..m {
            // Create a bitmap that contains, say, 1000 row IDs.
            let bitmap = RowAddrTreeMap::from_iter(0..per_bitmap_size);
            writer
                .emit(ScalarValue::UInt32(Some(i)), &bitmap)
                .await
                .unwrap();
        }
        let result = writer.finish().await;

        assert!(
            result.is_ok(),
            "Failed to write bitmap index: {:?}",
            result.err()
        );

        // Verify the index file exists
        let index_file = test_store.open_index_file(BITMAP_LOOKUP_NAME).await;
        assert!(
            index_file.is_ok(),
            "Failed to open index file: {:?}",
            index_file.err()
        );
        let index_file = index_file.unwrap();

        // Print stats about the index file
        tracing::info!(
            "Index file contains {} rows in total",
            index_file.num_rows()
        );

        // Load the index using BitmapIndex::load
        tracing::info!("Loading index from disk...");
        let loaded_index = BitmapIndex::load(Arc::new(test_store), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load bitmap index");

        // Verify the loaded index has the correct number of entries
        assert_eq!(
            loaded_index.index_map.len(),
            m as usize,
            "Loaded index has incorrect number of keys (expected {}, got {})",
            m,
            loaded_index.index_map.len()
        );

        // Manually verify specific keys without using search()
        let test_keys = [0, m / 2, m - 1]; // Beginning, middle, and end
        for &key_val in &test_keys {
            let key = OrderableScalarValue(ScalarValue::UInt32(Some(key_val)));
            // Load the bitmap for this key
            let bitmap = loaded_index
                .load_bitmap(&key, None)
                .await
                .unwrap_or_else(|_| panic!("Key {} should exist", key_val));

            // Convert RowAddrTreeMap to a vector for easier assertion
            let row_addrs: Vec<u64> = bitmap.row_addrs().unwrap().map(u64::from).collect();

            // Verify length
            assert_eq!(
                row_addrs.len(),
                per_bitmap_size as usize,
                "Bitmap for key {} has wrong size",
                key_val
            );

            // Verify first few and last few elements
            for i in 0..5.min(per_bitmap_size) {
                assert!(
                    row_addrs.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            for i in (per_bitmap_size - 5)..per_bitmap_size {
                assert!(
                    row_addrs.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            // Verify exact range
            let expected_range: Vec<u64> = (0..per_bitmap_size).collect();
            assert_eq!(
                row_addrs, expected_range,
                "Bitmap for key {} doesn't contain expected values",
                key_val
            );

            tracing::info!(
                "✓ Verified bitmap for key {}: {} rows as expected",
                key_val,
                row_addrs.len()
            );
        }

        tracing::info!("Test successful! Index properly contains {} keys", m);
    }

    #[tokio::test]
    async fn test_bitmap_prewarm() {
        // Create a temporary directory for the index
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data with low cardinality
        let colors = vec![
            "red", "blue", "green", "red", "yellow", "blue", "red", "green", "blue", "yellow",
            "red", "red", "blue", "green", "yellow",
        ];

        let row_ids = (0u64..15u64).collect::<Vec<_>>();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(colors.clone())),
                Arc::new(UInt64Array::from(row_ids.clone())),
            ],
        )
        .unwrap();

        let batch = sort_batch_by_value(&batch);
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        // Create a cache with metrics tracking
        let cache = LanceCache::with_capacity(1024 * 1024); // 1MB cache

        // Load the index (should only load metadata, not bitmaps)
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        // Verify no bitmaps are cached yet
        let red = OrderableScalarValue(ScalarValue::Utf8(Some("red".to_string())));
        let blue = OrderableScalarValue(ScalarValue::Utf8(Some("blue".to_string())));
        let cache_key_red = BitmapKey::try_new(*index.index_map.get(&red).unwrap()).unwrap();
        let cache_key_blue = BitmapKey::try_new(*index.index_map.get(&blue).unwrap()).unwrap();

        assert!(
            cache
                .get_with_key::<BitmapKey>(&cache_key_red)
                .await
                .is_none()
        );
        assert!(
            cache
                .get_with_key::<BitmapKey>(&cache_key_blue)
                .await
                .is_none()
        );

        // Call prewarm
        index.prewarm().await.unwrap();

        // Verify all bitmaps are now cached
        assert!(
            cache
                .get_with_key::<BitmapKey>(&cache_key_red)
                .await
                .is_some()
        );
        assert!(
            cache
                .get_with_key::<BitmapKey>(&cache_key_blue)
                .await
                .is_some()
        );

        // Verify cached bitmaps have correct content
        let cached_red = cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .unwrap();
        let red_rows: Vec<u64> = cached_red.row_addrs().unwrap().map(u64::from).collect();
        assert_eq!(red_rows, vec![0, 3, 6, 10, 11]);

        // Call prewarm again - should be idempotent
        index.prewarm().await.unwrap();

        // Verify cache still contains the same items
        let cached_red_2 = cache
            .get_with_key::<BitmapKey>(&cache_key_red)
            .await
            .unwrap();
        let red_rows_2: Vec<u64> = cached_red_2.row_addrs().unwrap().map(u64::from).collect();
        assert_eq!(red_rows_2, vec![0, 3, 6, 10, 11]);
    }

    // frags 1 and 2 (3 rows each) are compacted into frag 3: the 6 rows are
    // rewritten in order to frag 3 offsets 0..6.
    fn bitmap_remap_compact() -> RowAddrRemap {
        use lance_core::utils::row_addr_remap::GroupInput;
        use roaring::RoaringTreemap;
        RowAddrRemap::compact([GroupInput {
            rewritten_old_row_addrs: RoaringTreemap::from_iter(
                (0..3)
                    .map(|o| u64::from(RowAddress::new_from_parts(1, o)))
                    .chain((0..3).map(|o| u64::from(RowAddress::new_from_parts(2, o)))),
            ),
            old_frag_ids: vec![1, 2],
            new_frags: vec![(3, 6)],
        }])
        .unwrap()
    }

    fn bitmap_remap_explicit() -> RowAddrRemap {
        // The same mapping, listed out explicitly.
        RowAddrRemap::direct(
            (0..6u32)
                .map(|i| {
                    let (f, o) = if i < 3 { (1, i) } else { (2, i - 3) };
                    (
                        u64::from(RowAddress::new_from_parts(f, o)),
                        Some(u64::from(RowAddress::new_from_parts(3, i))),
                    )
                })
                .collect(),
        )
    }

    // remap must behave identically whether the mapping is compact or explicit.
    #[rstest]
    #[case(bitmap_remap_compact())]
    #[case(bitmap_remap_explicit())]
    #[tokio::test]
    async fn test_remap_bitmap_with_null(#[case] remap: RowAddrRemap) {
        use arrow_array::UInt32Array;

        // Create a temporary store.
        let tmpdir = TempObjDir::default();
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data that simulates:
        // frag 1 - { 0: null, 1: null, 2: 1 }
        // frag 2 - { 0: 1, 1: 2, 2: 2 }
        // We'll create this data with specific row addresses
        let values = vec![
            None,       // row 0: null (will be at address (1,0))
            None,       // row 1: null (will be at address (1,1))
            Some(1u32), // row 2: 1    (will be at address (1,2))
            Some(1u32), // row 3: 1    (will be at address (2,0))
            Some(2u32), // row 4: 2    (will be at address (2,1))
            Some(2u32), // row 5: 2    (will be at address (2,2))
        ];

        // Create row IDs with specific fragment addresses
        let row_ids: Vec<u64> = vec![
            RowAddress::new_from_parts(1, 0).into(),
            RowAddress::new_from_parts(1, 1).into(),
            RowAddress::new_from_parts(1, 2).into(),
            RowAddress::new_from_parts(2, 0).into(),
            RowAddress::new_from_parts(2, 1).into(),
            RowAddress::new_from_parts(2, 2).into(),
        ];

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::UInt32, true),
            Field::new("_rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();

        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Create the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, test_store.as_ref())
            .await
            .unwrap();

        // Load the index
        let index = BitmapIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load bitmap index");

        // Verify initial state
        assert_eq!(index.index_map.len(), 2); // 2 non-null values (1 and 2)
        assert!(!index.null_map.is_empty()); // Should have null values

        // Perform remap
        index.remap(&remap, test_store.as_ref()).await.unwrap();

        // Reload and check
        let reloaded_idx = BitmapIndex::load(test_store, None, &LanceCache::no_cache())
            .await
            .expect("Failed to load remapped bitmap index");

        // Verify the null bitmap was remapped correctly
        let expected_null_addrs: Vec<u64> = vec![
            RowAddress::new_from_parts(3, 0).into(),
            RowAddress::new_from_parts(3, 1).into(),
        ];
        let actual_null_addrs: Vec<u64> = reloaded_idx
            .null_map
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        assert_eq!(
            actual_null_addrs, expected_null_addrs,
            "Null bitmap not remapped correctly"
        );

        // Search for value 1 and verify remapped addresses
        let query = SargableQuery::Equals(ScalarValue::UInt32(Some(1)));
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            actual.sort();
            let expected: Vec<u64> = vec![
                RowAddress::new_from_parts(3, 2).into(),
                RowAddress::new_from_parts(3, 3).into(),
            ];
            assert_eq!(actual, expected, "Value 1 bitmap not remapped correctly");
        }

        // Search for value 2 and verify remapped addresses
        let query = SargableQuery::Equals(ScalarValue::UInt32(Some(2)));
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            actual.sort();
            let expected: Vec<u64> = vec![
                RowAddress::new_from_parts(3, 4).into(),
                RowAddress::new_from_parts(3, 5).into(),
            ];
            assert_eq!(actual, expected, "Value 2 bitmap not remapped correctly");
        }

        // Search for null values
        let query = SargableQuery::IsNull();
        let result = reloaded_idx
            .search(&query, &NoOpMetricsCollector)
            .await
            .unwrap();
        if let crate::scalar::SearchResult::Exact(row_ids) = result {
            let mut actual: Vec<u64> = row_ids
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect();
            actual.sort();
            assert_eq!(
                actual, expected_null_addrs,
                "Null search results not correct"
            );
        }
    }

    /// Remap must emit exactly what the pre-streaming path did: one row per
    /// source key, nulls included, every address put through the same mapping.
    ///
    /// The old path materialized the index into a
    /// `HashMap<ScalarValue, RowAddrTreeMap>`, remapped each entry and wrote the
    /// whole map, so a key whose rows were all deleted still produced a row with
    /// an empty bitmap. `remap_index_map` streams key-by-key instead and emits
    /// unconditionally to preserve that -- deliberately unlike `merge_index_maps`,
    /// which drops keys its filter empties.
    #[tokio::test]
    async fn test_bitmap_remap_matches_materialized_path() {
        // frag 1 - { 0: null, 1: "a", 2: "b" }
        // frag 2 - { 0: "a",  1: "c", 2: null }
        let addrs: Vec<u64> = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect();
        let values = [None, Some("a"), Some("b"), Some("a"), Some("c"), None];

        let (_src_dir, src_store) = test_util::index_store();
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from_iter(values)),
                Arc::new(UInt64Array::from(addrs.clone())),
            ],
        )
        .unwrap();
        let batch = sort_batch_by_value(&batch);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(async move { Ok(batch) }),
        ));
        BitmapIndexPlugin::train_bitmap_index(stream, src_store.as_ref())
            .await
            .unwrap();
        let index = BitmapIndex::load(src_store, None, &LanceCache::no_cache())
            .await
            .unwrap();

        // One key per arm of the mapping's tri-state: "a" keeps a remapped row
        // and loses a deleted one, "b" loses its only row, "c" is absent from the
        // mapping and passes through unchanged, and the null bitmap -- which lives
        // outside `index_map` -- sees both a remap and a delete.
        let mapping = RowAddrRemap::direct(HashMap::from([
            (addrs[0], Some(RowAddress::new_from_parts(3, 0).into())),
            (addrs[1], None),
            (addrs[2], None),
            (addrs[3], Some(RowAddress::new_from_parts(3, 1).into())),
            (addrs[5], None),
        ]));

        // The old path, restated: materialize every key including the null one,
        // remap each bitmap, write them all out.
        let mut old_path = Vec::new();
        if !index.null_map.is_empty() {
            old_path.push((None, remap_row_addrs(&index.null_map, &mapping)));
        }
        for key in index.index_map.keys() {
            let bitmap = index.load_bitmap(key, None).await.unwrap();
            let ScalarValue::Utf8(key) = key.0.clone() else {
                panic!("keys are utf8")
            };
            old_path.push((key, remap_row_addrs(&bitmap, &mapping)));
        }
        let mut old_path: Vec<(Option<String>, Vec<u64>)> = old_path
            .into_iter()
            .map(|(key, bitmap)| (key, test_util::row_addrs(&bitmap)))
            .collect();
        old_path.sort();

        // Guard the oracle: a remap that emitted nothing, or that dropped the
        // emptied key, would otherwise agree with an equally broken expectation.
        let frag_3 =
            |offset: u32| -> Vec<u64> { vec![RowAddress::new_from_parts(3, offset).into()] };
        assert_eq!(
            old_path,
            vec![
                (None, frag_3(0)),
                (Some("a".to_string()), frag_3(1)),
                (Some("b".to_string()), Vec::new()),
                (Some("c".to_string()), vec![addrs[4]]),
            ]
        );

        let (_dest_dir, dest_store) = test_util::index_store();
        index.remap(&mapping, dest_store.as_ref()).await.unwrap();
        assert_eq!(old_path, read_bitmap_contents(dest_store.as_ref()).await);

        // The old path wrote a `HashMap`, in no particular order. The streaming
        // one emits the null key first and then ascending keys.
        let written_keys: Vec<Option<String>> =
            test_util::read_key_bitmaps(dest_store.as_ref(), BITMAP_LOOKUP_NAME)
                .await
                .into_iter()
                .map(|(key, _)| key)
                .collect();
        assert_eq!(
            written_keys,
            vec![
                None,
                Some("a".to_string()),
                Some("b".to_string()),
                Some("c".to_string())
            ]
        );

        // The emptied key survives the round trip as a key rather than vanishing.
        let reloaded = BitmapIndex::load(dest_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert_eq!(reloaded.index_map.len(), 3);
    }

    #[tokio::test]
    async fn test_bitmap_null_handling_in_queries() {
        // Test that bitmap index correctly returns null_list for queries
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create test data: [0, 5, null]
        let batch = record_batch!(
            ("value", Int64, [Some(0), Some(5), None]),
            ("_rowid", UInt64, [0, 1, 2])
        )
        .unwrap();
        let schema = batch.schema();
        let stream = stream::once(async move { Ok(batch) });
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));

        // Train and write the bitmap index
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();

        let cache = LanceCache::with_capacity(1024 * 1024);
        let index = BitmapIndex::load(store.clone(), None, &cache)
            .await
            .unwrap();

        // Test 1: A caller that does not need NULL bookkeeping should receive
        // the same true rows without cloning the null bitmap.
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(5)));
        let result = index
            .search_with_options(
                &query,
                SearchOptions::default().with_track_nulls(false),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        match result {
            SearchResult::Exact(row_ids) => {
                let actual_rows: Vec<u64> = row_ids
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(actual_rows, vec![1]);
                assert!(row_ids.null_rows().is_empty());
            }
            _ => panic!("Expected Exact search result"),
        }

        // The existing API keeps NULL rows for three-valued logic.
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        match result {
            SearchResult::Exact(row_ids) => {
                let actual_rows: Vec<u64> = row_ids
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(actual_rows, vec![1], "Should find row 1 where value == 5");

                let null_row_ids = row_ids.null_rows();
                // Check that null_row_ids contains row 2
                assert!(!null_row_ids.is_empty(), "null_row_ids should be Some");
                let null_rows: Vec<u64> =
                    null_row_ids.row_addrs().unwrap().map(u64::from).collect();
                assert_eq!(null_rows, vec![2], "Should report row 2 as null");
            }
            _ => panic!("Expected Exact search result"),
        }
        let entries_after_value_lookup = cache.size().await;

        // Test 2: Search for null values - should return allow=[2], null=None
        let query = SargableQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        match result {
            SearchResult::Exact(row_addrs) => {
                let actual_rows: Vec<u64> = row_addrs
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(
                    actual_rows,
                    vec![2],
                    "IsNull should find row 2 where value is null"
                );

                let null_row_ids = row_addrs.null_rows();
                // When querying FOR nulls, null_row_ids should be None (nulls are the TRUE result)
                assert!(
                    null_row_ids.is_empty(),
                    "null_row_ids should be None for IsNull query"
                );
            }
            _ => panic!("Expected Exact search result"),
        }
        assert_eq!(
            cache.size().await,
            entries_after_value_lookup,
            "null bitmap lookup should bypass the per-value cache"
        );

        // Test 3: Range query - should return matching rows and null_list
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Int64(Some(0))),
            std::ops::Bound::Included(ScalarValue::Int64(Some(3))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        match result {
            SearchResult::Exact(row_addrs) => {
                let actual_rows: Vec<u64> = row_addrs
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(u64::from)
                    .collect();
                assert_eq!(actual_rows, vec![0], "Should find row 0 where value == 0");

                // Should report row 2 as null
                let null_row_ids = row_addrs.null_rows();
                assert!(!null_row_ids.is_empty(), "null_row_ids should be Some");
                let null_rows: Vec<u64> =
                    null_row_ids.row_addrs().unwrap().map(u64::from).collect();
                assert_eq!(null_rows, vec![2], "Should report row 2 as null");
            }
            _ => panic!("Expected Exact search result"),
        }
    }

    /// Merging bitmap segments must equal a single build over the same rows,
    /// including the null bitmap, which lives outside `index_map`.
    #[tokio::test]
    async fn test_bitmap_segment_merge_matches_single_build() {
        let values: Vec<Option<String>> = (0..600)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some(format!("v-{:03}", i % 50))
                }
            })
            .collect();

        async fn build(
            values: &[Option<String>],
            offset: u64,
        ) -> (TempObjDir, Arc<dyn IndexStore>) {
            let (tmpdir, store) = test_util::index_store();
            let schema = Arc::new(Schema::new(vec![
                Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
                Field::new(ROW_ID, DataType::UInt64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from_iter(values.iter().cloned())),
                    Arc::new(UInt64Array::from_iter_values(
                        (0..values.len() as u64).map(|i| i + offset),
                    )),
                ],
            )
            .unwrap();
            let batch = sort_batch_by_value(&batch);
            let stream = Box::pin(RecordBatchStreamAdapter::new(
                schema,
                stream::once(async move { Ok(batch) }),
            ));
            BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
                .await
                .unwrap();
            (tmpdir, store)
        }

        let (_all_dir, all_store) = build(&values, 0).await;
        let expected = read_bitmap_contents(all_store.as_ref()).await;
        assert!(
            expected.iter().any(|(key, _)| key.is_none()),
            "fixture must exercise the null bitmap"
        );

        let (_left_dir, left_store) = build(&values[..300], 0).await;
        let (_right_dir, right_store) = build(&values[300..], 300).await;
        let left = BitmapIndex::load(left_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        let right = BitmapIndex::load(right_store, None, &LanceCache::no_cache())
            .await
            .unwrap();

        let (_dest_dir, dest_store) = test_util::index_store();
        merge_bitmap_indices(
            &[left, right],
            dest_store.as_ref(),
            crate::progress::noop_progress(),
        )
        .await
        .unwrap();

        assert_eq!(expected, read_bitmap_contents(dest_store.as_ref()).await);
    }

    /// The keys column counts toward the flush threshold, not just the bitmaps.
    /// A column with a very large number of tiny bitmaps used to buffer without
    /// limit: the old threshold charged the bitmap column only, so the keys could
    /// grow until the writer held gigabytes and their Arrow i32 offsets overflowed.
    #[tokio::test]
    async fn test_batch_writer_charges_keys_against_flush_threshold() {
        // Small enough to cross with a handful of keys; the production threshold
        // is MAX_BUFFERED_BYTES, which no test wants to write out.
        const THRESHOLD: usize = 4 * 1024;
        const NUM_KEYS: u64 = 64;

        let (_tmpdir, store) = test_util::index_store();
        let mut writer =
            new_bitmap_batch_writer(store.as_ref(), BITMAP_LOOKUP_NAME, &DataType::Utf8)
                .await
                .unwrap()
                .with_max_buffered_bytes(THRESHOLD);

        // Every bitmap holds one row, so all the bitmaps together stay under the
        // threshold; only the keys can push the writer past it.
        for i in 0..NUM_KEYS {
            let key = format!("{i:08}{}", "k".repeat(512));
            writer
                .emit(
                    ScalarValue::Utf8(Some(key)),
                    &RowAddrTreeMap::from_iter([i]),
                )
                .await
                .unwrap();
        }
        assert!(
            writer.batches_written() > 1,
            "keys must be charged against the flush threshold, but the writer \
             buffered all {NUM_KEYS} keys in {} batch(es)",
            writer.batches_written()
        );
        writer.finish().await.unwrap();

        // Flushing part-way through must not disturb the file's contents.
        let contents = read_bitmap_contents(store.as_ref()).await;
        assert_eq!(contents.len() as u64, NUM_KEYS);
        for (idx, (key, addrs)) in contents.iter().enumerate() {
            let key = key.as_deref().expect("keys are non-null");
            assert_eq!(&key[..8], format!("{idx:08}"), "keys must stay ascending");
            assert_eq!(addrs, &vec![idx as u64]);
        }
    }

    /// Key to sorted row addresses, read from the file and sorted so the
    /// comparison does not depend on the order keys happen to be written in.
    async fn read_bitmap_contents(store: &dyn IndexStore) -> Vec<(Option<String>, Vec<u64>)> {
        let mut out: Vec<(Option<String>, Vec<u64>)> =
            test_util::read_key_bitmaps(store, BITMAP_LOOKUP_NAME)
                .await
                .into_iter()
                .map(|(key, bitmap)| (key, test_util::row_addrs(&bitmap)))
                .collect();
        out.sort();
        out
    }
}
