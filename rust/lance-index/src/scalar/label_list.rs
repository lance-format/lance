// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::{
    any::Any,
    fmt::Debug,
    pin::Pin,
    sync::{Arc, Mutex},
};

use arrow::array::AsArray;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, SortOptions};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::RecordBatchStream;
use datafusion::physical_plan::{
    ExecutionPlan, SendableRecordBatchStream, sorts::sort::SortExec,
    stream::RecordBatchStreamAdapter,
};
use datafusion_common::ScalarValue;
use datafusion_physical_expr::{PhysicalSortExpr, expressions::Column};
use futures::{StreamExt, TryStream, TryStreamExt, stream::BoxStream};
use lance_core::cache::{
    CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
    KeyBuilder, LanceCache,
};
use lance_core::deepsize::DeepSizeOf;
use lance_core::error::LanceOptionExt;
use lance_core::{Error, ROW_ID, Result};
use lance_datafusion::exec::{
    HardCapBatchSizeExec, LanceExecutionOptions, OneShotExec, execute_plan,
};
use lance_select::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use roaring::RoaringBitmap;
use tracing::instrument;

use super::{
    AnyQuery, IndexFile, IndexStore, LabelListQuery, OldIndexDataFilter, ScalarIndex,
    bitmap::BitmapIndex,
};
use super::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
use super::{MetricsCollector, SearchResult};
use crate::pbold;
use crate::scalar::bitmap::{
    BitmapIndexState, build_index_map, merge_index_maps, merge_source_entry_count,
    new_bitmap_batch_writer, remap_index_map, remap_row_addrs,
};
use crate::scalar::expression::{LabelListQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    BasicTrainer, DefaultTrainingRequest, ScalarIndexLoad, ScalarIndexPlugin, TrainingCriteria,
    TrainingOrdering, TrainingRequest, VALUE_COLUMN_NAME, single_flight_open,
};
use crate::scalar::{CreatedIndex, RowIdRemapper, UpdateCriteria};
use crate::{Index, IndexType};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
pub const LABEL_LIST_NULLS_METADATA_KEY: &str = "lance:label_list_nulls";
pub const LABEL_LIST_NULLS_MIN_VERSION: i32 = 1;
const LABEL_LIST_INDEX_VERSION: u32 = 1;

#[async_trait]
trait LabelListSubIndex: ScalarIndex + DeepSizeOf {
    async fn search_exact(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableRowAddrSet> {
        let result = self.search(query, metrics).await?;
        match result {
            SearchResult::Exact(row_ids) => {
                // Label list semantics treat NULL elements as non-matches, so only TRUE/FALSE
                // results should remain for array_has_any/array_has_all when the list itself
                // is non-NULL. Clear nulls to avoid propagating element-level NULLs.
                Ok(row_ids.with_nulls(RowAddrTreeMap::new()))
            }
            _ => Err(Error::internal(
                "Label list sub-index should return exact results".to_string(),
            )),
        }
    }
}

impl<T: ScalarIndex + DeepSizeOf> LabelListSubIndex for T {}

/// A scalar index that can be used on `List<T>` columns to
/// accelerate list membership filters such as `array_has_all`, `array_has_any`,
/// and `array_has` / `array_contains`, using an underlying bitmap index.
#[derive(Clone, Debug, DeepSizeOf)]
pub struct LabelListIndex {
    values_index: Arc<BitmapIndex>,
    list_nulls: Arc<RowAddrTreeMap>,
}

impl LabelListIndex {
    fn new(values_index: Arc<BitmapIndex>, list_nulls: Arc<RowAddrTreeMap>) -> Self {
        Self {
            values_index,
            list_nulls,
        }
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let values_index =
            BitmapIndex::load(store.clone(), frag_reuse_index.clone(), index_cache).await?;
        let list_nulls = read_list_nulls(store, frag_reuse_index).await?;
        Ok(Arc::new(Self::new(values_index, Arc::new(list_nulls))))
    }
}

#[async_trait]
impl Index for LabelListIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        self.values_index.prewarm().await
    }

    fn index_type(&self) -> IndexType {
        IndexType::LabelList
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.values_index.statistics()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl LabelListIndex {
    fn search_values<'a>(
        &'a self,
        values: &'a Vec<ScalarValue>,
        metrics: &'a dyn MetricsCollector,
    ) -> BoxStream<'a, Result<NullableRowAddrSet>> {
        futures::stream::iter(values)
            .then(move |value| {
                let value_query = SargableQuery::Equals(value.clone());
                async move { self.values_index.search_exact(&value_query, metrics).await }
            })
            .boxed()
    }

    async fn set_union<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = NullableRowAddrSet, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<NullableRowAddrSet> {
        let mut union_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(union_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            union_bitmap |= &next;
        }
        Ok(union_bitmap)
    }

    async fn set_intersection<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = NullableRowAddrSet, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<NullableRowAddrSet> {
        let mut intersect_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(intersect_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            intersect_bitmap &= &next;
        }
        Ok(intersect_bitmap)
    }
}

#[async_trait]
impl ScalarIndex for LabelListIndex {
    #[instrument(skip_all, level = "debug")]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<LabelListQuery>().unwrap();

        let row_ids = match query {
            LabelListQuery::HasAllLabels(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_intersection(values_results, labels.len() == 1)
                    .await
            }
            LabelListQuery::HasAnyLabel(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_union(values_results, labels.len() == 1).await
            }
        }?;
        let row_ids = if self.list_nulls.as_ref().is_empty() {
            row_ids
        } else {
            let mut nulls = row_ids.null_rows().clone();
            nulls |= self.list_nulls.as_ref();
            row_ids.with_nulls(nulls)
        };
        Ok(SearchResult::Exact(row_ids))
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
        let remapped_nulls = remap_row_addrs(&self.list_nulls, mapping)?;
        let mut writer = new_bitmap_batch_writer(
            dest_store,
            BITMAP_LOOKUP_NAME,
            self.values_index.value_type(),
        )
        .await?;
        writer
            .add_global_buffer(
                LABEL_LIST_NULLS_METADATA_KEY.to_string(),
                serialize_list_nulls(&remapped_nulls)?,
            )
            .await?;
        remap_index_map(&self.values_index, mapping, &mut writer).await?;
        let file = writer.finish().await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
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
        // Not applied, matching every other derived-key scalar index (ngram,
        // fmindex, bloomfilter, zonemap, rtree). Only btree and bitmap -- one
        // key per row -- prune retired rows here. The cost is real: postings for
        // retired rows survive, so an update that rewrites rows in place can
        // leave this index returning them. Preserved as-is rather than fixed,
        // since changing it is a correctness change this memory work should not
        // carry.
        let _ = old_data_filter;
        let file = update_label_list_index(self, new_data, dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
            files: vec![file],
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList))
    }
}

fn extract_flatten_indices(list_arr: &dyn Array) -> UInt64Array {
    if let Some(list_arr) = list_arr.as_list_opt::<i32>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else if let Some(list_arr) = list_arr.as_list_opt::<i64>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else {
        unreachable!(
            "Should verify that the first column is a list earlier. Got array of type: {}",
            list_arr.data_type()
        )
    }
}

/// Collect row_ids for list-level NULLs before unnest; unnest drops NULL lists entirely.
fn track_list_nulls(
    source: SendableRecordBatchStream,
    list_nulls: Arc<Mutex<RowAddrTreeMap>>,
) -> SendableRecordBatchStream {
    let schema = source.schema();
    let stream = source.try_filter_map(move |batch| {
        let list_nulls = list_nulls.clone();
        async move {
            record_list_nulls(&batch, &list_nulls)?;
            Ok(Some(batch))
        }
    });

    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

fn record_list_nulls(
    batch: &RecordBatch,
    list_nulls: &Arc<Mutex<RowAddrTreeMap>>,
) -> datafusion_common::Result<()> {
    let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
    let row_ids = batch.column_by_name(ROW_ID).expect_ok()?;
    let row_ids = row_ids.as_any().downcast_ref::<UInt64Array>().unwrap();

    let mut local_nulls = RowAddrTreeMap::new();
    for i in 0..values.len() {
        if values.is_null(i) {
            local_nulls.insert(row_ids.value(i));
        }
    }
    if !local_nulls.is_empty() {
        let mut guard = list_nulls.lock().unwrap();
        *guard |= &local_nulls;
    }
    Ok(())
}

fn unnest_schema(schema: &Schema) -> SchemaRef {
    let mut fields_iter = schema.fields.iter().cloned();
    let key_field = fields_iter.next().unwrap();
    let remaining_fields = fields_iter.collect::<Vec<_>>();

    let new_key_field = match key_field.data_type() {
        DataType::List(item_field) | DataType::LargeList(item_field) => Field::new(
            key_field.name(),
            item_field.data_type().clone(),
            item_field.is_nullable() || key_field.is_nullable(),
        ),
        other_type => {
            unreachable!(
                "The first field in the schema must be a List or LargeList type. \
                Found: {}. This should have been verified earlier in the code.",
                other_type
            )
        }
    };

    let all_fields = vec![Arc::new(new_key_field)]
        .into_iter()
        .chain(remaining_fields)
        .collect::<Vec<_>>();

    Arc::new(Schema::new(Fields::from(all_fields)))
}

fn unnest_batch(
    batch: arrow::record_batch::RecordBatch,
    unnest_schema: SchemaRef,
) -> datafusion_common::Result<RecordBatch> {
    let mut columns_iter = batch.columns().iter().cloned();
    let key_col = columns_iter.next().unwrap();
    let remaining_cols = columns_iter.collect::<Vec<_>>();

    let remaining_fields = unnest_schema
        .fields
        .iter()
        .skip(1)
        .cloned()
        .collect::<Vec<_>>();

    let remaining_batch = RecordBatch::try_new(
        Arc::new(Schema::new(Fields::from(remaining_fields))),
        remaining_cols,
    )?;

    let flatten_indices = extract_flatten_indices(key_col.as_ref());

    let flattened_remaining =
        arrow_select::take::take_record_batch(&remaining_batch, &flatten_indices)?;

    let new_key_values = if let Some(key_list) = key_col.as_list_opt::<i32>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else if let Some(key_list) = key_col.as_list_opt::<i64>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else {
        unreachable!("Should verify that the first column is a list earlier")
    };

    let all_columns = vec![new_key_values]
        .into_iter()
        .chain(flattened_remaining.columns().iter().cloned())
        .collect::<Vec<_>>();

    datafusion_common::Result::Ok(arrow::record_batch::RecordBatch::try_new(
        unnest_schema,
        all_columns,
    )?)
}

fn unnest_chunks(
    source: Pin<Box<dyn RecordBatchStream + Send>>,
) -> Result<SendableRecordBatchStream> {
    let unnest_schema = unnest_schema(source.schema().as_ref());
    let unnest_schema_copy = unnest_schema.clone();
    let source = source.try_filter_map(move |batch| {
        std::future::ready(Some(unnest_batch(batch, unnest_schema.clone())).transpose())
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        unnest_schema_copy,
        source,
    )))
}

async fn read_list_nulls(
    store: Arc<dyn IndexStore>,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
) -> Result<RowAddrTreeMap> {
    let reader = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
    if let Some(buffer_idx_str) = reader.schema().metadata.get(LABEL_LIST_NULLS_METADATA_KEY) {
        let buffer_idx = buffer_idx_str.parse::<u32>().map_err(|err| {
            Error::internal(format!(
                "LabelList metadata key {} had invalid global buffer index {}: {}",
                LABEL_LIST_NULLS_METADATA_KEY, buffer_idx_str, err
            ))
        })?;
        let bytes = reader.read_global_buffer(buffer_idx).await?;
        let null_map = RowAddrTreeMap::deserialize_from(bytes.as_ref())?;
        return if let Some(frag_reuse_index) = frag_reuse_index {
            Ok(frag_reuse_index.remap_row_addrs_tree_map(&null_map))
        } else {
            Ok(null_map)
        };
    }
    Ok(RowAddrTreeMap::default())
}

fn serialize_list_nulls(null_map: &RowAddrTreeMap) -> Result<Bytes> {
    let mut bytes = Vec::new();
    null_map.serialize_into(&mut bytes)?;
    Ok(Bytes::from(bytes))
}

/// Sort unnested `(label, row_addr)` pairs by label so that
/// [`build_index_map`] can stream one label at a time.
///
/// This is DataFusion's sort, so it spills under `LANCE_MEM_POOL_SIZE` to
/// `LANCE_MAX_TEMP_DIRECTORY_SIZE` rather than through a mechanism belonging to
/// this index. `JsonIndexPlugin::sort_stream_by_value` sorts its training
/// stream the same way. Note this is a full blocking sort, which buffers or
/// spills the whole unnested stream -- unlike `BTreeIndexPlugin::merge_segments`,
/// whose `SortPreservingMergeExec` streams over already-ordered inputs and never
/// buffers the data volume at all.
///
/// Nulls sort first because `OrderableScalarValue` orders them below every
/// value, so [`build_index_map`]'s ascending-input `debug_assert!` rejects a
/// stream that puts them last. Its runtime path would in fact tolerate a null
/// run anywhere -- `finish_run` never advances the old-keys cursor for a null
/// key -- but the assert is the contract, and null-first is also what
/// [`remap_index_map`] emits and what the plain bitmap index's training scan
/// produces.
///
/// `mem_pool_size`, when set, overrides the session's memory pool for this
/// sort instead of leaving it to `LANCE_MEM_POOL_SIZE`. Production callers
/// pass `None`; the spilling-equivalence test passes an explicit size so it
/// can force a spill without mutating the process-wide environment variable
/// that every other cached session shares.
///
/// Also returns the `SortExec` node itself (not just its output stream), so a
/// caller can inspect its metrics -- e.g. `spill_count` -- once the stream is
/// fully drained. Production call sites have no use for it and bind it as
/// `_`; the spilling-equivalence test uses it to confirm a forced-spill build
/// actually spilled, rather than trusting the pool-size math alone.
fn sort_labels_by_value(
    unnested: SendableRecordBatchStream,
    mem_pool_size: Option<u64>,
) -> Result<(SendableRecordBatchStream, Arc<dyn ExecutionPlan>)> {
    let input: Arc<dyn ExecutionPlan> = Arc::new(OneShotExec::new(unnested));
    // Resolved by name: `Column::evaluate` only bounds-checks the index it is
    // given and never consults the name, so a hardcoded index would silently
    // sort by whatever column ends up there if the unnest schema's column
    // order ever changed.
    let value_idx = input.schema().index_of(VALUE_COLUMN_NAME)?;
    let sort_expr = PhysicalSortExpr {
        expr: Arc::new(Column::new(VALUE_COLUMN_NAME, value_idx)),
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    };

    let options = LanceExecutionOptions {
        use_spilling: true,
        mem_pool_size,
        ..Default::default()
    };

    // Unnesting multiplies rows by list length, so one scan batch becomes a batch
    // far larger than any other operator produces: 8,192 rows x 512 labels is
    // 4.2M rows at once. DataFusion's sort cannot spill a batch it has not first
    // admitted, so an oversized batch fails the build outright instead of
    // spilling -- reproduced as `ResourcesExhausted` from `ExternalSorter` with
    // only 512 distinct labels. Cap by bytes before the sort. Byte-aware rather
    // than row-aware because list width and label width both vary, and
    // deep-copying, because a sliced batch still reports its parent buffer's
    // size to memory accounting.
    //
    // A sixth of the pool, derived rather than fixed: the sort reserves
    // `min(pool / 3, 40 MiB)` up front, so at worst two thirds of the pool is
    // left to admit a batch into, and a sixth is a quarter of that worst case --
    // room for the sort to work in, not just to hold the batch. Deriving it also
    // avoids restating that reservation formula, which lives in `lance-datafusion`
    // and would silently drift. At the 150 MiB default this is 25 MiB, the value
    // the merge-insert write path fixes above its own sort.
    let max_batch_bytes = (options.mem_pool_size() / 6).max(1) as usize;
    let capped: Arc<dyn ExecutionPlan> =
        Arc::new(HardCapBatchSizeExec::new(input, max_batch_bytes));

    let sorted: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new([sort_expr].into(), capped));
    let stream = execute_plan(sorted.clone(), options)?;
    Ok((stream, sorted))
}

/// Write a LabelList index: a bitmap index over `sorted_labels`, plus the
/// `list_nulls` set as a global buffer in the same file.
///
/// `list_nulls` is a closure, not a value, because it is only correct to read
/// after `sorted_labels` is drained -- `track_list_nulls` fills its set as rows
/// flow through it, and the sort upstream means no row reaches here until every
/// input row has passed it. The buffer is attached after the data pages for the
/// same reason; `add_global_buffer` records its index in the file metadata, so
/// position does not matter to readers.
async fn write_label_list_index(
    store: &dyn IndexStore,
    value_type: &DataType,
    sorted_labels: SendableRecordBatchStream,
    old_index: Option<&BitmapIndex>,
    list_nulls: impl FnOnce() -> Result<RowAddrTreeMap>,
) -> Result<IndexFile> {
    let mut writer = new_bitmap_batch_writer(store, BITMAP_LOOKUP_NAME, value_type).await?;
    // `None`: LabelList does not apply the old-data filter. See `update`.
    build_index_map(sorted_labels, old_index, None, &mut writer).await?;
    writer
        .add_global_buffer(
            LABEL_LIST_NULLS_METADATA_KEY.to_string(),
            serialize_list_nulls(&list_nulls()?)?,
        )
        .await?;
    writer.finish().await
}

/// Build a LabelList index from a `(list column, row_addr)` stream, also
/// returning the sort's `SortExec` node.
///
/// The list column is unnested and the resulting labels sorted, so the bitmap
/// builder holds one label's bitmap at a time instead of the whole map. The sort
/// is DataFusion's and spills under `LANCE_MEM_POOL_SIZE`.
///
/// [`train_label_list_index`] is a thin wrapper over this that discards the
/// plan handle and always passes `None` for `mem_pool_size` -- production has
/// no use for either. The spilling-equivalence test calls this directly,
/// passing an explicit `mem_pool_size`, so its forced-spill build can assert
/// on `spill_count` while still running the exact sequence production does,
/// rather than a hand-maintained copy of it that could quietly drift out of
/// step.
async fn train_label_list_index_with_plan(
    data: SendableRecordBatchStream,
    index_store: &dyn IndexStore,
    mem_pool_size: Option<u64>,
) -> Result<(IndexFile, Arc<dyn ExecutionPlan>)> {
    let list_nulls = Arc::new(Mutex::new(RowAddrTreeMap::new()));
    let data = track_list_nulls(data, list_nulls.clone());
    let data = unnest_chunks(data)?;
    let value_type = data.schema().field(0).data_type().clone();
    let (sorted, sort_plan) = sort_labels_by_value(data, mem_pool_size)?;

    let file = write_label_list_index(index_store, &value_type, sorted, None, || {
        Ok(list_nulls.lock().unwrap().clone())
    })
    .await?;
    Ok((file, sort_plan))
}

/// Build a LabelList index from a `(list column, row_addr)` stream.
async fn train_label_list_index(
    data: SendableRecordBatchStream,
    index_store: &dyn IndexStore,
) -> Result<IndexFile> {
    train_label_list_index_with_plan(data, index_store, None)
        .await
        .map(|(file, _sort_plan)| file)
}

/// Add `new_data` to an existing LabelList index, writing the result to
/// `dest_store`.
///
/// The existing index joins the new labels through bitmap's merge-join, which
/// walks its `index_map` and loads one old bitmap at a time. Files written before
/// this change have their keys in arbitrary on-disk order, which does not matter:
/// `BitmapIndex::load` builds `index_map` as a `BTreeMap` recording each key's
/// file offset, so iteration is sorted regardless.
async fn update_label_list_index(
    existing: &LabelListIndex,
    new_data: SendableRecordBatchStream,
    dest_store: &dyn IndexStore,
) -> Result<IndexFile> {
    let list_nulls = Arc::new(Mutex::new(RowAddrTreeMap::new()));
    let new_data = track_list_nulls(new_data, list_nulls.clone());
    let new_data = unnest_chunks(new_data)?;

    // The destination file's schema is declared from the existing index while
    // every new key comes from the stream. If the two disagree the written
    // batches would not match the schema they are declared under, so reject it
    // here rather than emit a file whose schema lies about its key type.
    let value_type = existing.values_index.value_type().clone();
    let new_value_type = new_data.schema().field(0).data_type().clone();
    if new_value_type != value_type {
        return Err(Error::invalid_input(format!(
            "Cannot update a LabelList index with value type {value_type} \
             from new data whose list items are {new_value_type}"
        )));
    }

    let (sorted, _sort_plan) = sort_labels_by_value(new_data, None)?;
    let existing_nulls = existing.list_nulls.clone();

    write_label_list_index(
        dest_store,
        &value_type,
        sorted,
        Some(&existing.values_index),
        || {
            let mut merged = (*existing_nulls).clone();
            merged |= &*list_nulls.lock().unwrap();
            Ok(merged)
        },
    )
    .await
}

/// Merge multiple LabelList index segments into a single index.
///
/// A [`LabelListIndex`] is a [`BitmapIndex`] over the unnested list values plus a
/// separate `list_nulls` row set. Because distributed segments cover disjoint rows
/// (distinct fragments), merging streams and unions the bitmap payloads by key
/// and separately unions the `list_nulls` sets; no source-data re-scan is
/// required. This mirrors [`crate::scalar::bitmap::merge_bitmap_indices`] but
/// also carries the per-segment `list_nulls`. When `old_data_filter` is provided,
/// rows from retired fragments are removed from both the value bitmaps and
/// `list_nulls`.
pub async fn merge_label_list_indices(
    source_indices: &[Arc<LabelListIndex>],
    dest_store: &dyn IndexStore,
    old_data_filter: Option<OldIndexDataFilter>,
    progress: Arc<dyn crate::progress::IndexBuildProgress>,
) -> Result<CreatedIndex> {
    if source_indices.is_empty() {
        return Err(Error::invalid_input(
            "LabelList segment merge requires at least one source segment".to_string(),
        ));
    }

    let value_type = source_indices[0].values_index.value_type().clone();
    let mut merged_nulls = RowAddrTreeMap::new();

    let mut values_indices = Vec::with_capacity(source_indices.len());
    for source_index in source_indices.iter() {
        if source_index.values_index.value_type() != &value_type {
            return Err(Error::invalid_input(format!(
                "LabelList segment has value type {:?}, expected {:?}",
                source_index.values_index.value_type(),
                value_type
            )));
        }
        values_indices.push(source_index.values_index.clone());

        // `list_nulls` records whole rows whose list was null, so it is a row
        // set per segment rather than a per-label bitmap. It stays outside both
        // the key merge and the aggregation-state budget.
        // Only copy when there is a filter to apply: an unfiltered merge is the
        // common case, and each copy is a whole segment's row set.
        if let Some(filter) = old_data_filter.as_ref() {
            let mut list_nulls = source_index.list_nulls.as_ref().clone();
            filter.retain_old_rows(&mut list_nulls);
            merged_nulls |= &list_nulls;
        } else {
            merged_nulls |= source_index.list_nulls.as_ref();
        }
    }

    progress
        .stage_start(
            "merge_label_list_segments",
            Some(merge_source_entry_count(&values_indices)),
            "entries",
        )
        .await?;

    let mut writer = new_bitmap_batch_writer(dest_store, BITMAP_LOOKUP_NAME, &value_type).await?;
    writer
        .add_global_buffer(
            LABEL_LIST_NULLS_METADATA_KEY.to_string(),
            serialize_list_nulls(&merged_nulls)?,
        )
        .await?;
    merge_index_maps(
        &values_indices,
        old_data_filter.as_ref(),
        &mut writer,
        Some((progress.as_ref(), "merge_label_list_segments")),
    )
    .await?;
    progress.stage_complete("merge_label_list_segments").await?;

    progress
        .stage_start("write_label_list_index", Some(1), "files")
        .await?;
    let file = writer.finish().await?;
    progress.stage_progress("write_label_list_index", 1).await?;
    progress.stage_complete("write_label_list_index").await?;

    Ok(CreatedIndex {
        index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
            .unwrap(),
        index_version: LABEL_LIST_INDEX_VERSION,
        files: vec![file],
    })
}

/// The serializable state of a [`LabelListIndex`].
///
/// `LabelListIndex` is a thin wrapper around a [`BitmapIndex`] plus a separate
/// row bitmap tracking which list values were `NULL` (lost by unnest at build
/// time). Its cache state is the corresponding [`BitmapIndexState`] plus the
/// already-loaded `list_nulls`.
#[derive(Debug, Clone)]
pub struct LabelListIndexState {
    bitmap_state: BitmapIndexState,
    list_nulls: Arc<RowAddrTreeMap>,
}

impl DeepSizeOf for LabelListIndexState {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.bitmap_state.deep_size_of_children(context)
            + self.list_nulls.deep_size_of_children(context)
    }
}

impl LabelListIndexState {
    fn from_index(index: &LabelListIndex) -> Result<Self> {
        Ok(Self {
            bitmap_state: BitmapIndexState::from_index(&index.values_index)?,
            list_nulls: index.list_nulls.clone(),
        })
    }

    fn from_scalar_index(index: &dyn ScalarIndex) -> Result<Self> {
        let label_list = index
            .as_any()
            .downcast_ref::<LabelListIndex>()
            .ok_or_else(|| {
                Error::internal(
                    "LabelListIndexState::from_scalar_index called with a non-label-list index",
                )
            })?;
        Self::from_index(label_list)
    }

    fn into_label_list_index(
        self,
        store: Arc<dyn IndexStore>,
        index_cache: &LanceCache,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Arc<LabelListIndex>> {
        let bitmap = self
            .bitmap_state
            .to_bitmap_index(store, index_cache, frag_reuse_index)?;
        Ok(Arc::new(LabelListIndex::new(bitmap, self.list_nulls)))
    }
}

impl CacheCodecImpl for LabelListIndexState {
    const TYPE_ID: &'static str = "lance.scalar.LabelListIndexState";
    const CURRENT_VERSION: u32 = 1;

    /// Wire format:
    /// ```text
    /// RAW_BLOB : list_nulls (roaring tree map, portable encoding)
    /// <nested BitmapIndexState body (self-delimiting)>
    /// ```
    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let mut nulls_bytes = Vec::with_capacity(self.list_nulls.serialized_size());
        self.list_nulls.serialize_into(&mut nulls_bytes)?;
        w.write_raw(&nulls_bytes)?;
        // The bitmap state writes its own self-delimiting body inline.
        self.bitmap_state.serialize(w)?;
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let nulls_bytes = r.read_raw()?;
        let list_nulls = Arc::new(RowAddrTreeMap::deserialize_from(nulls_bytes.as_ref())?);
        // The bitmap state is self-delimiting (length-prefixed null map +
        // Arrow IPC stream with EOS marker); it continues reading the body
        // from where the null map left off.
        let bitmap_state = BitmapIndexState::deserialize(r)?;
        Ok(Self {
            bitmap_state,
            list_nulls,
        })
    }
}

struct LabelListIndexStateKey;

impl CacheKey for LabelListIndexStateKey {
    type ValueType = LabelListIndexState;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "state".into()
    }

    fn type_name() -> &'static str {
        "LabelListIndexState"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.label-list-index-state-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_variant(0);
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<LabelListIndexState>())
    }
}

#[derive(Debug, Default)]
pub struct LabelListIndexPlugin;

pub(super) fn validate_label_list_data_type(data_type: &DataType) -> Result<()> {
    let item_type = match data_type {
        DataType::List(item_field) | DataType::LargeList(item_field) => item_field.data_type(),
        _ => {
            return Err(Error::invalid_input_source(
                format!(
                    "LabelList index can only be created on List or LargeList type columns. Column has type {:?}",
                    data_type
                )
                .into(),
            ));
        }
    };

    if item_type.is_nested() {
        return Err(Error::invalid_input_source(
            format!(
                "LabelList index item type must be non-nested. Column has type {:?}",
                data_type
            )
            .into(),
        ));
    }

    Ok(())
}

#[async_trait]
impl BasicTrainer for LabelListIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        validate_label_list_data_type(field.data_type())?;

        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        )))
    }

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`
    /// and the plugin can rely on this fact.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        // Training over a fragment subset is supported for distributed builds: the
        // provided `data` stream is already scoped to those fragments, so a partial
        // index covering exactly those rows is produced. Segments are recombined by
        // `merge_label_list_indices`.
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let schema = data.schema();
        let field = schema
            .column_with_name(VALUE_COLUMN_NAME)
            .ok_or_else(|| {
                Error::invalid_input_source(
                    "Index training data missing value column"
                        .to_string()
                        .into(),
                )
            })?
            .1;

        validate_label_list_data_type(field.data_type())?;

        let file = train_label_list_index(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
            files: vec![file],
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for LabelListIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "LabelList"
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        LABEL_LIST_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(LabelListQueryParser::new(
            index_name,
            self.name().to_string(),
        )))
    }

    /// Load an index from storage
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            LabelListIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }

    async fn get_from_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Option<Arc<dyn ScalarIndex>>> {
        let Some(state) = cache.get_with_key(&LabelListIndexStateKey).await else {
            return Ok(None);
        };
        let state = (*state).clone();
        let index = state.into_label_list_index(index_store, cache, frag_reuse_index)?;
        Ok(Some(index as Arc<dyn ScalarIndex>))
    }

    async fn put_in_cache(
        &self,
        _index_store: Arc<dyn IndexStore>,
        cache: &LanceCache,
        index: Arc<dyn ScalarIndex>,
    ) -> Result<()> {
        let state = LabelListIndexState::from_scalar_index(index.as_ref())?;
        cache
            .insert_with_key(&LabelListIndexStateKey, Arc::new(state))
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
            LabelListIndexStateKey,
            load,
            LabelListIndexState::from_scalar_index,
            move |state| {
                Ok((*state)
                    .clone()
                    .into_label_list_index(index_store, cache, frag_reuse_index)?
                    as Arc<dyn ScalarIndex>)
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use datafusion_common::ScalarValue;
    use lance_core::cache::CacheCodec;
    use lance_core::utils::address::RowAddress;
    use rstest::rstest;

    use super::super::bitmap::BitmapIndexState;
    use super::super::btree::OrderableScalarValue;
    use super::*;
    use crate::scalar::bitmap::test_util::{self, row_addrs};
    use lance_core::utils::tempfile::TempObjDir;

    #[rstest]
    #[case::list(DataType::List(Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        true,
    ))))]
    #[case::large_list(DataType::LargeList(Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        true,
    ))))]
    fn test_rejects_nested_item_type(#[case] data_type: DataType) {
        let field = Field::new(VALUE_COLUMN_NAME, data_type, true);
        let error = LabelListIndexPlugin
            .new_training_request("", &field)
            .err()
            .expect("nested item type should be rejected");

        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected invalid input error, got: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("LabelList index item type must be non-nested"),
            "unexpected error: {error}"
        );
    }

    fn sample_state() -> LabelListIndexState {
        let mut index_map = BTreeMap::new();
        for k in 0..32i32 {
            index_map.insert(
                OrderableScalarValue(ScalarValue::Int32(Some(k))),
                k as usize,
            );
        }
        let mut bitmap_nulls = RowAddrTreeMap::new();
        bitmap_nulls.insert(RowAddress::new_from_parts(0, 3).into());
        let bitmap_state =
            BitmapIndexState::new_for_test(index_map, bitmap_nulls, DataType::Int32).unwrap();

        let mut list_nulls = RowAddrTreeMap::new();
        list_nulls.insert(RowAddress::new_from_parts(0, 9).into());
        LabelListIndexState {
            bitmap_state,
            list_nulls: Arc::new(list_nulls),
        }
    }

    #[test]
    fn test_label_list_state_codec_roundtrip() {
        let state = sample_state();
        let mut buf = Vec::new();
        state
            .serialize(&mut CacheEntryWriter::new(&mut buf))
            .unwrap();
        let data = Bytes::from(buf);
        let mut reader = CacheEntryReader::new(&data, 0, LabelListIndexState::CURRENT_VERSION);
        let restored = LabelListIndexState::deserialize(&mut reader).unwrap();

        assert_eq!(&*restored.list_nulls, &*state.list_nulls);
        assert_eq!(
            restored.bitmap_state.lookup_batch(),
            state.bitmap_state.lookup_batch()
        );
        assert_eq!(
            restored.bitmap_state.null_map(),
            state.bitmap_state.null_map()
        );
    }

    /// The nested bitmap lookup batch must decode zero-copy through the full
    /// envelope, proving the leading `list_nulls` RAW_BLOB does not knock the
    /// nested IPC section off its 64-byte boundary.
    #[test]
    fn test_label_list_nested_lookup_is_zero_copy() {
        const ALIGN: usize = 64;
        let codec = CacheCodec::from_impl::<LabelListIndexState>();
        let any: Arc<dyn std::any::Any + Send + Sync> = Arc::new(sample_state());
        let mut buf = Vec::new();
        codec.serialize(&any, &mut buf).unwrap();

        let mut v = vec![0u8; buf.len() + ALIGN];
        let pad = (ALIGN - (v.as_ptr() as usize % ALIGN)) % ALIGN;
        v[pad..pad + buf.len()].copy_from_slice(&buf);
        let data = Bytes::from(v).slice(pad..pad + buf.len());

        let restored = codec.deserialize(&data).hit().unwrap();
        let restored = restored.downcast::<LabelListIndexState>().unwrap();

        let base = data.as_ptr() as usize;
        let end = base + data.len();
        for col in restored.bitmap_state.lookup_batch().columns() {
            for buffer in col.to_data().buffers() {
                let ptr = buffer.as_ptr() as usize;
                assert!(
                    ptr >= base && ptr < end,
                    "nested bitmap lookup buffer was realigned — misaligned IPC section",
                );
            }
        }
    }

    /// A wide list column must build whatever the memory pool is set to.
    ///
    /// `unnest_chunks` maps one input batch to one output batch, multiplying rows
    /// by list length, so a scan's 8,192-row batch becomes 4.2M unnested rows in
    /// a single batch. DataFusion's sort cannot spill a batch it has not first
    /// admitted, so without the byte cap in `sort_labels_by_value` this fails
    /// outright with `ResourcesExhausted` -- and with only 512 distinct labels,
    /// so it is unrelated to the cardinality this index's memory work targets.
    ///
    /// Both pool sizes matter. The default case is the originally reported
    /// failure; the small-pool case is what a cap fixed at 25 MiB gets wrong,
    /// since 25 MiB does not fit in a 32 MiB pool's ~21 MiB of admittable space.
    /// Each fixture is sized to exceed its own pool, so a cap that stopped
    /// tracking the pool fails one case or the other.
    ///
    /// Deliberately one large batch: every other fixture here chunks input at 64
    /// rows, which is why none of them can reach this. The default-pool case is
    /// also the one test here that exceeds the one-second guideline (~1.6s), and
    /// deliberately: it has to put more than ~110 MiB through a single batch, and
    /// trimming it closer to the threshold would let it pass without the cap on a
    /// machine with slightly different accounting. A memory test that quietly
    /// stops reproducing is worse than a slow one.
    #[rstest]
    #[case::default_pool(None, 1024)]
    #[case::small_pool(Some(32 * 1024 * 1024), 256)]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_train_index_handles_a_wide_list_in_one_scan_batch(
        #[case] mem_pool_size: Option<u64>,
        #[case] rows: usize,
    ) {
        // Sized for bytes, not rows: a wide label keeps the single batch over the
        // pool while sorting far fewer rows than an 8,192-row scan batch would.
        const LABELS_PER_ROW: usize = 512;
        const LABEL_PAD: usize = 200;

        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new(
                VALUE_COLUMN_NAME,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let mut builder =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new());
        for _ in 0..rows {
            for label in 0..LABELS_PER_ROW {
                builder
                    .values()
                    .append_value(format!("label-{label:04}{}", "p".repeat(LABEL_PAD)));
            }
            builder.append(true);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(builder.finish()),
                Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
            ],
        )
        .unwrap();
        let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(vec![Ok(batch)]),
        ));

        let (_tmpdir, store) = test_util::index_store();
        train_label_list_index_with_plan(stream, store.as_ref(), mem_pool_size)
            .await
            .expect("a wide list column must build whatever the pool is set to");

        let contents = read_index_contents(store.as_ref()).await;
        assert_eq!(
            contents.labels.len(),
            LABELS_PER_ROW,
            "every label in the lists must be indexed"
        );
        // Every row carries every label, so each posting covers all rows.
        assert!(
            contents.labels.iter().all(|(_, addrs)| addrs.len() == rows),
            "each label must cover all {rows} rows"
        );
        assert!(contents.list_nulls.is_empty());
    }

    // ---- shared fixtures for the spill-build tests ------------------------

    /// One test row: its address, and either a null list or a list whose
    /// elements may individually be null. Both null shapes matter — a null list
    /// is recorded in `list_nulls`, a null element survives unnesting as a null
    /// index key.
    type SampleRow = (u64, Option<Vec<Option<String>>>);

    /// Deterministic rows with heavy label reuse, a null list every 17th row,
    /// and a null element every 11th, so every code path sees both.
    fn sample_label_list_rows(count: usize, addr_offset: u64) -> Vec<SampleRow> {
        let rows: Vec<SampleRow> = (0..count)
            .map(|i| {
                let addr = addr_offset + i as u64;
                if i % 17 == 0 {
                    return (addr, None);
                }
                let mut labels: Vec<Option<String>> = (0..(i % 4) + 1)
                    .map(|k| Some(format!("label-{:04}", (i * 7 + k * 13) % 200)))
                    .collect();
                if i % 11 == 0 {
                    labels.push(None);
                }
                (addr, Some(labels))
            })
            .collect();

        // Guarded here rather than in each caller: several tests rely on this
        // fixture carrying both null shapes, and a change to the pattern above
        // would otherwise silently stop exercising them everywhere at once.
        // That the shapes then reach the right places in the index is
        // `test_null_element_null_list_and_empty_list_stay_distinct`'s job.
        assert!(
            rows.iter().any(|(_, labels)| labels.is_none()),
            "fixture must contain a null list; {count} rows is too few"
        );
        assert!(
            rows.iter()
                .any(|(_, labels)| labels.as_ref().is_some_and(|l| l.contains(&None))),
            "fixture must contain a null label; {count} rows is too few"
        );
        rows
    }

    fn label_list_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(
                VALUE_COLUMN_NAME,
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]))
    }

    /// Chunk the rows into several batches so the builder crosses batch
    /// boundaries the way a real scan does.
    fn sample_rows_to_stream(rows: &[SampleRow]) -> SendableRecordBatchStream {
        let schema = label_list_schema();
        let batches: Vec<_> = rows
            .chunks(64)
            .map(|chunk| {
                let mut builder = arrow_array::builder::ListBuilder::new(
                    arrow_array::builder::StringBuilder::new(),
                );
                for (_, labels) in chunk {
                    match labels {
                        Some(labels) => {
                            for label in labels {
                                builder.values().append_option(label.as_deref());
                            }
                            builder.append(true);
                        }
                        None => builder.append(false),
                    }
                }
                let addrs = UInt64Array::from_iter_values(chunk.iter().map(|(addr, _)| *addr));
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(builder.finish()), Arc::new(addrs)],
                )
                .unwrap()
            })
            .collect();

        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    /// The index as a comparable value: labels (nulls included) each mapped to
    /// their sorted row addresses, plus the separately-stored `list_nulls`.
    /// Read from the file rather than the loaded index so that the on-disk
    /// format itself is what the assertions compare.
    #[derive(Debug, PartialEq, Eq)]
    struct IndexContents {
        labels: Vec<(Option<String>, Vec<u64>)>,
        list_nulls: Vec<u64>,
    }

    /// Also asserts that the directory holds nothing but the index. Spill files
    /// belong in local scratch, and every path that writes an index here --
    /// build, update, remap, merge -- must leave none of them behind, so the
    /// check lives where all of them are already read back.
    async fn read_index_contents(store: &dyn IndexStore) -> IndexContents {
        let files: Vec<String> = store
            .list_files_with_sizes()
            .await
            .unwrap()
            .into_iter()
            .map(|file| file.path)
            .collect();
        assert_eq!(files, vec![BITMAP_LOOKUP_NAME.to_string()]);

        let mut labels: Vec<(Option<String>, Vec<u64>)> =
            test_util::read_key_bitmaps(store, BITMAP_LOOKUP_NAME)
                .await
                .into_iter()
                .map(|(key, bitmap)| (key, row_addrs(&bitmap)))
                .collect();
        // The build path emits keys in sorted order; older indexes on disk are
        // in arbitrary order. Sort so the comparison is order-independent.
        labels.sort();

        let list_nulls = read_list_nulls(store.clone_arc(), None).await.unwrap();
        IndexContents {
            labels,
            list_nulls: row_addrs(&list_nulls),
        }
    }

    async fn build_label_list_index(rows: &[SampleRow]) -> IndexContents {
        let (_tmpdir, store) = test_util::index_store();
        train_label_list_index(sample_rows_to_stream(rows), store.as_ref())
            .await
            .unwrap();
        read_index_contents(store.as_ref()).await
    }

    // Every LabelList build or update now sorts through DataFusion with spilling
    // enabled. Most tests below get that from the default, process-cached
    // session, whose pool `LANCE_MEM_POOL_SIZE` governs, and so share that
    // resource group -- see the same pattern in btree.rs, json.rs and rtree.rs.
    // The spill-forcing test below passes an explicit `mem_pool_size` for the
    // spilling half, but its baseline build goes through the default session
    // like everything else, so it joins the group too.
    /// A pool this small forces the sort below to spill: `new_session_context`
    /// reserves `(pool / 3).min(40 MiB)` up front, leaving only `2 * pool / 3`
    /// (about 171 KiB here) for in-memory accumulation, which the fixture's
    /// unnested data clears. A smaller pool (tried down to 64 KiB) instead
    /// fails the final spill-merge with `ResourcesExhausted`, since that step
    /// also draws on the same budget -- this size was picked empirically to
    /// leave it enough room. Measured at this size: 3 spills, 108,096 bytes and
    /// 4,878 rows spilled, consistent across repeated runs.
    const SPILL_FORCING_POOL_BYTES: u64 = 256 * 1024;

    /// The sort spills through DataFusion's memory pool, so a build that spills
    /// must produce the same index as one that does not. Forced via an explicit
    /// `mem_pool_size` rather than by mutating the process-wide
    /// `LANCE_MEM_POOL_SIZE` that every other cached session shares, so this
    /// test needs no serialization against them.
    ///
    /// A small pool alone does not prove a spill happened -- the sort could stay
    /// under budget, or a regression could silently drop `use_spilling` -- so this
    /// also reads the `SortExec`'s own metrics after the build completes and
    /// asserts `spill_count > 0`. Calling [`train_label_list_index_with_plan`]
    /// directly (rather than the equivalent hand-written unnest/sort/write
    /// sequence) is what lets the forced-spill build assert on the plan while
    /// still running the exact production build path -- the same one `generous`
    /// below runs through `build_label_list_index`, which uses the shared
    /// default session and is why this joins the serial group.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_train_index_identical_when_the_sort_spills() {
        // Large enough that the unnested data clears the spill-forcing pool's
        // remaining budget with room to spare; small enough to sort and spill
        // well within the one-second test budget.
        const ROWS: usize = 2_000;
        let rows = sample_label_list_rows(ROWS, 0);

        let generous = build_label_list_index(&rows).await;

        let (_tmpdir, store) = test_util::index_store();
        let (_file, sort_plan) = train_label_list_index_with_plan(
            sample_rows_to_stream(&rows),
            store.as_ref(),
            Some(SPILL_FORCING_POOL_BYTES),
        )
        .await
        .unwrap();

        let metrics = sort_plan.metrics().expect("SortExec must report metrics");
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "fixture and pool size must force a real spill, or this test \
             cannot tell a working spill path from a broken one; got \
             metrics: {metrics:?}"
        );

        let tiny = read_index_contents(store.as_ref()).await;

        assert_eq!(generous, tiny, "a spilling sort must not change the index");
    }

    async fn build_label_list_segment(rows: &[SampleRow]) -> (TempObjDir, Arc<LabelListIndex>) {
        let (tmpdir, store) = test_util::index_store();
        train_label_list_index(sample_rows_to_stream(rows), store.as_ref())
            .await
            .unwrap();
        let index = LabelListIndex::load(store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        (tmpdir, index)
    }

    /// `remap` carries `list_nulls` in a global buffer written through the
    /// streaming writer, and nothing else covers that path. If the buffer or its
    /// metadata key were dropped, `read_list_nulls` would return an empty set and
    /// the index would be classified as pre-nulls, so `NOT` filters over nullable
    /// list columns would return wrong rows with only a warning -- no error.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_remap_rewrites_addresses_and_preserves_list_nulls() {
        const OLD_FRAGMENT: u64 = 1 << 32;
        const NEW_FRAGMENT: u64 = 3 << 32;
        const ROWS: usize = 120;

        let rows = sample_label_list_rows(ROWS, OLD_FRAGMENT);
        // Built here rather than via build_label_list_segment so the source
        // store stays reachable, to compare against after the remap.
        let (_src_dir, src_store) = test_util::index_store();
        train_label_list_index(sample_rows_to_stream(&rows), src_store.as_ref())
            .await
            .unwrap();
        let index = LabelListIndex::load(src_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        let before = read_index_contents(src_store.as_ref()).await;

        // Move every row to a new fragment, and delete one so the dropped-row
        // path is covered too.
        let deleted = OLD_FRAGMENT + 1;
        let mapping = RowAddrRemap::direct(
            (0..ROWS as u64)
                .map(|i| {
                    let old = OLD_FRAGMENT + i;
                    (old, (old != deleted).then_some(NEW_FRAGMENT + i))
                })
                .collect(),
        );

        let (_dest_dir, dest_store) = test_util::index_store();
        index.remap(&mapping, dest_store.as_ref()).await.unwrap();
        let after = read_index_contents(dest_store.as_ref()).await;

        let remapped = |addrs: &[u64]| -> Vec<u64> {
            addrs
                .iter()
                .filter(|addr| **addr != deleted)
                .map(|addr| addr - OLD_FRAGMENT + NEW_FRAGMENT)
                .collect()
        };

        // The nulls buffer must survive, remapped, not silently become empty.
        assert_eq!(
            after.list_nulls,
            remapped(&before.list_nulls),
            "list_nulls must be carried through remap with its addresses rewritten"
        );

        assert_eq!(
            after.labels,
            before
                .labels
                .iter()
                .map(|(key, addrs)| (key.clone(), remapped(addrs)))
                .collect::<Vec<_>>(),
            "every label must survive with its addresses rewritten to the new fragment"
        );
    }

    /// Many segments whose keys interleave without overlapping is the case key
    /// selection must handle in `log(segments)` rather than by scanning every
    /// segment for every key. Covers both shapes at once: a vocabulary unique to
    /// each segment, plus one label every segment holds.
    ///
    /// The merge must equal a single build over the same rows, and being
    /// key-driven it must also report progress as it goes -- reporting once on
    /// return leaves a long merge indistinguishable from a hung one -- against a
    /// denominator it actually reaches. Source entries give one; emitted keys do
    /// not, since the shared label emits one row for `NUM_SEGMENTS` entries.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_segment_merge_matches_single_build_and_reports_progress() {
        const NUM_SEGMENTS: usize = 5;
        const ROWS_PER_SEGMENT: usize = 40;
        const SHARED_LABEL: &str = "shared";

        fn segment_rows(segment: usize) -> Vec<SampleRow> {
            (0..ROWS_PER_SEGMENT)
                .map(|i| {
                    let addr = ((segment as u64) << 32) | i as u64;
                    // A null list every 7th row and a null label every 5th, so
                    // the merge carries both null shapes as well.
                    if i % 7 == 0 {
                        return (addr, None);
                    }
                    // Striding by the segment count interleaves the vocabularies,
                    // so each segment's keys fall between its neighbours' rather
                    // than alongside them.
                    let mut labels = vec![
                        Some(format!("label-{:04}", i * NUM_SEGMENTS + segment)),
                        Some(SHARED_LABEL.to_string()),
                    ];
                    if i % 5 == 0 {
                        labels.push(None);
                    }
                    (addr, Some(labels))
                })
                .collect()
        }

        let mut all_rows = Vec::new();
        let mut segments = Vec::new();
        let mut _dirs = Vec::new();
        for segment in 0..NUM_SEGMENTS {
            let rows = segment_rows(segment);
            all_rows.extend(rows.clone());
            let (dir, index) = build_label_list_segment(&rows).await;
            _dirs.push(dir);
            segments.push(index);
        }

        let (_dest_dir, dest_store) = test_util::index_store();
        let progress = Arc::new(RecordingProgress::default());
        merge_label_list_indices(&segments, dest_store.as_ref(), None, progress.clone())
            .await
            .unwrap();

        let merged = read_index_contents(dest_store.as_ref()).await;
        assert_eq!(
            merged,
            build_label_list_index(&all_rows).await,
            "merging {NUM_SEGMENTS} interleaved segments must equal a single build"
        );
        assert!(
            merged.labels.iter().any(|(key, _)| key.is_none()),
            "fixture must exercise null labels"
        );
        assert!(
            !merged.list_nulls.is_empty(),
            "fixture must exercise null lists"
        );

        let merge_counts = progress.counts("merge_label_list_segments");
        let declared_total = progress.declared_total("merge_label_list_segments");
        let merged_keys = merged.labels.len() as u64;
        // The shared label makes entries outnumber output rows. This is what an
        // emitted-key denominator cannot express, and reverting to one fails here
        // rather than silently under-reporting.
        assert!(
            declared_total > merged_keys,
            "fixture must have segments sharing labels so entries ({declared_total}) \
             exceed emitted keys ({merged_keys})"
        );
        // Pins "reports as it goes", which the monotonicity check below cannot:
        // a throttle that collapsed to a single terminal report would satisfy
        // both that and the declared-total check while leaving a long merge
        // indistinguishable from a hung one.
        assert!(
            merge_counts.len() > 1,
            "the merge must report incrementally, got {} report(s)",
            merge_counts.len()
        );
        assert!(
            merge_counts.windows(2).all(|pair| pair[0] <= pair[1]),
            "reported progress must never go backwards: {merge_counts:?}"
        );
        assert_eq!(
            merge_counts.last().copied(),
            Some(declared_total),
            "the merge must finish on exactly the total it declared"
        );
    }

    /// Records what a build reported, so progress can be asserted on.
    #[derive(Debug, Default)]
    struct RecordingProgress {
        /// `(stage, completed)` for every progress call, in order.
        calls: std::sync::Mutex<Vec<(String, u64)>>,
        /// `(stage, total)` for every stage start.
        totals: std::sync::Mutex<Vec<(String, Option<u64>)>>,
    }

    impl RecordingProgress {
        fn counts(&self, stage: &str) -> Vec<u64> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(reported, _)| reported == stage)
                .map(|(_, completed)| *completed)
                .collect()
        }

        fn declared_total(&self, stage: &str) -> u64 {
            self.totals
                .lock()
                .unwrap()
                .iter()
                .find(|(reported, _)| reported == stage)
                .unwrap_or_else(|| panic!("stage {stage} was never started"))
                .1
                .unwrap_or_else(|| panic!("stage {stage} declared no total"))
        }
    }

    #[async_trait]
    impl crate::progress::IndexBuildProgress for RecordingProgress {
        async fn stage_start(&self, stage: &str, total: Option<u64>, _: &str) -> Result<()> {
            self.totals.lock().unwrap().push((stage.to_string(), total));
            Ok(())
        }
        async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push((stage.to_string(), completed));
            Ok(())
        }
        async fn stage_complete(&self, _: &str) -> Result<()> {
            Ok(())
        }
    }

    /// Segment merge prunes rows from retired fragments. This is existing
    /// behaviour and the streaming merge must not quietly drop it.
    ///
    /// A key whose rows are all retired emits nothing but still consumed its
    /// source entries. Progress has to count it, or a merge that drops a
    /// segment's exclusive labels ends below the total it declared -- the
    /// stalled bar this reporting exists to remove.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_segment_merge_applies_old_data_filter() {
        const FRAGMENT_1: u64 = 1 << 32;
        const RETIRED_ONLY: &str = "retired-only-";

        let kept = sample_label_list_rows(150, 0);
        let retired: Vec<SampleRow> = sample_label_list_rows(150, FRAGMENT_1)
            .into_iter()
            .enumerate()
            .map(|(i, (addr, labels))| {
                // A label the kept segment does not have, so the filter empties
                // that key outright rather than merely trimming it. Rows with a
                // null list keep it, so retired `list_nulls` are pruned too.
                let labels = labels.map(|mut labels| {
                    labels.push(Some(format!("{RETIRED_ONLY}{i:04}")));
                    labels
                });
                (addr, labels)
            })
            .collect();

        let expected = build_label_list_index(&kept).await;

        let (_kept_dir, kept_index) = build_label_list_segment(&kept).await;
        let (_retired_dir, retired_index) = build_label_list_segment(&retired).await;
        let (_dest_dir, dest_store) = test_util::index_store();
        let progress = Arc::new(RecordingProgress::default());
        merge_label_list_indices(
            &[kept_index, retired_index],
            dest_store.as_ref(),
            Some(OldIndexDataFilter::Fragments {
                to_keep: RoaringBitmap::from_iter([0u32]),
                to_remove: RoaringBitmap::from_iter([1u32]),
            }),
            progress.clone(),
        )
        .await
        .unwrap();

        assert!(
            retired.iter().any(|(_, labels)| labels
                .as_ref()
                .is_some_and(|l| l.iter().flatten().any(|k| k.starts_with(RETIRED_ONLY)))),
            "the retired segment must own labels the kept one lacks, or the filter \
             only trims keys instead of emptying them outright"
        );

        let merged = read_index_contents(dest_store.as_ref()).await;
        assert_eq!(
            expected, merged,
            "rows from the retired fragment must not survive the merge"
        );

        let declared_total = progress.declared_total("merge_label_list_segments");
        assert_eq!(
            progress.counts("merge_label_list_segments").last().copied(),
            Some(declared_total),
            "entries consumed by keys the filter emptied must still be reported"
        );
    }

    /// The spill and destination file schemas are declared from the existing
    /// index while the keys come from the new stream, so a disagreement must be
    /// rejected rather than written as a file whose schema lies about its keys.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_update_rejects_a_mismatched_value_type() {
        let (_src_dir, index) = build_label_list_segment(&sample_label_list_rows(64, 0)).await;
        let (_dest_dir, dest_store) = test_util::index_store();

        // The index above is List<Utf8>; feed the update List<LargeUtf8>.
        let mut list_builder =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::LargeStringBuilder::new());
        list_builder.values().append_value("a");
        list_builder.append(true);
        let labels = list_builder.finish();
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, labels.data_type().clone(), true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(labels),
                Arc::new(UInt64Array::from(vec![1u64 << 32])),
            ],
        )
        .unwrap();
        let new_data: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(vec![Ok(batch)]),
        ));

        let error = update_label_list_index(index.as_ref(), new_data, dest_store.as_ref())
            .await
            .expect_err("a value-type mismatch must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("value type Utf8") && message.contains("items are LargeUtf8"),
            "the error must name both value types, got: {message}"
        );
    }

    /// Every LabelList index written before spill-based builds landed has its
    /// keys in arbitrary order on disk, because the old writer iterated a HashMap.
    /// Nothing added here may assume sorted files: reading, updating and
    /// merging such an index must all still be correct.
    ///
    /// This is also the primary update-equals-rebuild coverage. A separate test
    /// over a current-format index would assert the same equality through the
    /// same merge-join over strictly easier input, since `index_map` is sorted
    /// regardless of file order.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_reads_updates_and_merges_an_unsorted_legacy_index() {
        const FRAGMENT_1: u64 = 1 << 32;

        /// Write the same index the old HashMap-consuming writer would have:
        /// identical contents, keys in an order that is not sorted.
        async fn write_unsorted_index(rows: &[SampleRow], store: &dyn IndexStore) {
            let sorted = build_label_list_index(rows).await;
            let mut writer = new_bitmap_batch_writer(store, BITMAP_LOOKUP_NAME, &DataType::Utf8)
                .await
                .unwrap();
            writer
                .add_global_buffer(
                    LABEL_LIST_NULLS_METADATA_KEY.to_string(),
                    serialize_list_nulls(&RowAddrTreeMap::from_iter(
                        sorted.list_nulls.iter().copied(),
                    ))
                    .unwrap(),
                )
                .await
                .unwrap();
            // Reverse order is sorted-ness's clearest counterexample.
            for (key, addrs) in sorted.labels.iter().rev() {
                writer
                    .emit(
                        ScalarValue::Utf8(key.clone()),
                        &RowAddrTreeMap::from_iter(addrs.iter().copied()),
                    )
                    .await
                    .unwrap();
            }
            writer.finish().await.unwrap();
        }

        let initial = sample_label_list_rows(200, 0);
        let expected_initial = build_label_list_index(&initial).await;

        let (_legacy_dir, legacy_store) = test_util::index_store();
        write_unsorted_index(&initial, legacy_store.as_ref()).await;

        // Reading: the loaded index must match one built by the current path.
        assert_eq!(
            expected_initial,
            read_index_contents(legacy_store.as_ref()).await,
            "an unsorted file must carry the same contents"
        );
        let legacy = LabelListIndex::load(legacy_store.clone(), None, &LanceCache::no_cache())
            .await
            .unwrap();

        // Updating: streaming the unsorted index in as a merge input must still
        // produce the same index as a full rebuild.
        let additional = sample_label_list_rows(100, FRAGMENT_1);
        let mut all = initial.clone();
        all.extend(additional.clone());
        let (_updated_dir, updated_store) = test_util::index_store();
        update_label_list_index(
            legacy.as_ref(),
            sample_rows_to_stream(&additional),
            updated_store.as_ref(),
        )
        .await
        .unwrap();
        assert_eq!(
            build_label_list_index(&all).await,
            read_index_contents(updated_store.as_ref()).await,
            "update over an unsorted index diverged from a full rebuild"
        );

        // Merging: an unsorted segment must merge correctly with a sorted one.
        let (_other_dir, other_index) = build_label_list_segment(&additional).await;
        let (_merged_dir, merged_store) = test_util::index_store();
        merge_label_list_indices(
            &[legacy, other_index],
            merged_store.as_ref(),
            None,
            crate::progress::noop_progress(),
        )
        .await
        .unwrap();
        assert_eq!(
            build_label_list_index(&all).await,
            read_index_contents(merged_store.as_ref()).await,
            "merging an unsorted segment diverged from a full build"
        );
    }

    /// A null *element* inside a list, a null *list*, and an empty list are three
    /// different things and must land in three different places. `track_list_nulls`
    /// runs ahead of the unnest, where the list column's validity is still present,
    /// which is the only reason they are separable -- after unnesting, a null list
    /// and an empty list are both zero rows.
    ///
    /// Asserted through every path that writes an index, so moving the unnest
    /// upstream of the null tracking fails here rather than silently misfiling
    /// rows and breaking NOT filters on nullable list columns.
    #[rstest]
    #[case::build("build")]
    #[case::update("update")]
    #[case::merge("merge")]
    #[case::remap("remap")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_null_element_null_list_and_empty_list_stay_distinct(#[case] path: &str) {
        let rows: Vec<SampleRow> = vec![
            (0, Some(vec![Some("b".to_string()), Some("a".to_string())])),
            (1, None),                                    // null list
            (2, Some(vec![])),                            // empty list
            (3, Some(vec![Some("a".to_string()), None])), // null element
        ];

        // `merge_label_list_indices` is documented to assume its segments cover
        // disjoint fragments, and every other merge test in this file honors
        // that by giving each segment its own fragment rather than by relying
        // on addresses 0-3 happening not to collide. Follow the same
        // convention here: the merge case's second segment (rows 2 and 3)
        // moves to fragment 1, taking the null-element row's expected address
        // with it. Every other path keeps the original addresses.
        const FRAGMENT_1: u64 = 1 << 32;
        let null_element_addr = if path == "merge" { FRAGMENT_1 + 1 } else { 3 };

        let expected_labels = vec![
            (None, vec![null_element_addr]), // the null element
            (Some("a".to_string()), vec![0, null_element_addr]),
            (Some("b".to_string()), vec![0]),
        ];

        let actual = match path {
            "build" => build_label_list_index(&rows).await,
            "update" => {
                // Split so both halves carry one of the null shapes.
                let (first, second) = rows.split_at(2);
                let (_src_dir, index) = build_label_list_segment(first).await;
                let (_dest_dir, dest_store) = test_util::index_store();
                update_label_list_index(
                    index.as_ref(),
                    sample_rows_to_stream(second),
                    dest_store.as_ref(),
                )
                .await
                .unwrap();
                read_index_contents(dest_store.as_ref()).await
            }
            "merge" => {
                let (first, second) = rows.split_at(2);
                let second: Vec<SampleRow> = second
                    .iter()
                    .map(|(addr, labels)| (*addr - 2 + FRAGMENT_1, labels.clone()))
                    .collect();
                let (_a_dir, a) = build_label_list_segment(first).await;
                let (_b_dir, b) = build_label_list_segment(&second).await;
                let (_dest_dir, dest_store) = test_util::index_store();
                merge_label_list_indices(
                    &[a, b],
                    dest_store.as_ref(),
                    None,
                    crate::progress::noop_progress(),
                )
                .await
                .unwrap();
                read_index_contents(dest_store.as_ref()).await
            }
            "remap" => {
                let (_src_dir, src_store) = test_util::index_store();
                train_label_list_index(sample_rows_to_stream(&rows), src_store.as_ref())
                    .await
                    .unwrap();
                let index = LabelListIndex::load(src_store, None, &LanceCache::no_cache())
                    .await
                    .unwrap();
                // Identity mapping: the addresses must survive untouched.
                let (_dest_dir, dest_store) = test_util::index_store();
                let mapping =
                    RowAddrRemap::direct((0..4u64).map(|addr| (addr, Some(addr))).collect());
                index.remap(&mapping, dest_store.as_ref()).await.unwrap();
                read_index_contents(dest_store.as_ref()).await
            }
            other => panic!("unknown path {other}"),
        };

        assert_eq!(
            actual.labels, expected_labels,
            "the null element belongs to the null key, and nothing else does"
        );
        assert_eq!(
            actual.list_nulls,
            vec![1],
            "only the null list belongs to list_nulls -- not the empty list, \
             and not the row with a null element"
        );
    }
}
