// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read one B-tree index segment in key order.
//!
//! [`OrderedIndexScanExec`] streams
//! [`OrderedScalarIndex::search_ordered`](lance_index::scalar::OrderedScalarIndex::search_ordered)
//! and keeps only the rows the dataset still serves. Each index row falls in
//! one of three groups, and only live rows count toward `fetch`:
//!
//! ```text
//! index row      fragment            offset         result
//! (key, addr) -> outside the scope                -> rows_excluded
//!                in the scope, gone               -> rows_deleted
//!                in the scope     -> deleted      -> rows_deleted
//!                                 -> not deleted  -> emitted
//! ```
//!
//! The scope is the segment's fragment coverage, narrowed by an optional
//! fragment allow-list.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt32Array, cast::AsArray, types::UInt64Type};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use arrow_select::take::take_record_batch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
    metrics::{Count, ExecutionPlanMetricsSet, MetricsSet},
};
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalSortExpr, expressions::Column};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use lance_core::{Error, ROW_ADDR_FIELD, Result, utils::address::RowAddress};
use lance_datafusion::utils::{ExecutionPlanMetricsSetExt, ROWS_SCANNED_METRIC};
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::scalar::{AnyQuery, ScalarIndex};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::staleness::overlay_exclusion_offsets;
use roaring::RoaringBitmap;

use super::utils::InstrumentedRecordBatchStreamAdapter;
use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::io::deletion::read_dataset_deletion_file;
use crate::{Dataset, index::DatasetIndexInternalExt};

/// Streams one segment of a B-tree index in key order as two columns: the key,
/// named after the indexed column, then `_rowaddr`.
///
/// Deleted rows and rows outside the fragment scope never reach the output, so
/// they do not use up `fetch`. Rows with equal keys come in the order the
/// index yields them. A fragment's deletion vector loads the first time the
/// scan reaches one of its rows. Without a `fetch`, the scan reaches whole
/// batches. With one, it first reaches the in-scope rows of a batch that the
/// fetch still needs, then doubles the in-scope rows it has reached each time
/// deleted rows use them up.
///
/// ```
/// # use std::sync::Arc;
/// # use lance::deps::arrow_schema::SortOptions;
/// # use lance::deps::datafusion::physical_plan::ExecutionPlan;
/// # use lance::index::DatasetIndexExt;
/// # use lance::io::exec::ordered_index_scan::OrderedIndexScanExec;
/// # use lance::{Dataset, Result};
/// # async fn test(dataset: Arc<Dataset>) -> Result<()> {
/// let segment = dataset.load_indices_by_name("ts_idx").await?.remove(0);
/// let newest_first = SortOptions { descending: true, nulls_first: false };
/// let scan = OrderedIndexScanExec::try_new(dataset, segment, None, newest_first, None).await?;
/// let newest_ten = scan.with_fetch(Some(10));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct OrderedIndexScanExec {
    dataset: Arc<Dataset>,
    segment: IndexMetadata,
    index: Arc<dyn ScalarIndex>,
    column: String,
    query: Option<Arc<dyn AnyQuery>>,
    options: SortOptions,
    scope: RoaringBitmap,
    fetch: Option<usize>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl OrderedIndexScanExec {
    /// Plans an ordered read of `segment`, one segment of a single-column
    /// B-tree index on `dataset`.
    ///
    /// `query` limits the read to the rows it selects, and `options` is the key
    /// order. `fragments`, when set, limits the output to rows in those
    /// fragments.
    ///
    /// This opens the segment and checks that it can stream in key order
    /// without reading a data page, so a planner that gets an error can keep
    /// its sort.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the segment indexes more or fewer
    /// than one column, names a field the dataset does not have, or records no
    /// fragment coverage, or when the index rejects `query`.
    ///
    /// Returns [`Error::NotSupported`] in these cases:
    ///
    /// - The dataset uses stable row ids and the segment stores them instead
    ///   of row addresses. Segments written before the row address migration
    ///   do.
    /// - A data overlay committed after the segment changes the indexed column
    ///   in a fragment the scan reads.
    /// - The index cannot search in key order, because it is not a B-tree or
    ///   its pages are not in key order.
    ///
    /// Returns [`Error::Index`] when the index returns a key type other than
    /// the column's.
    pub async fn try_new(
        dataset: Arc<Dataset>,
        segment: IndexMetadata,
        query: Option<Arc<dyn AnyQuery>>,
        options: SortOptions,
        fragments: Option<&RoaringBitmap>,
    ) -> Result<Self> {
        let [field_id] = segment.fields[..] else {
            return Err(Error::invalid_input(format!(
                "ordered index scan reads a single-column index, but segment {} of index '{}' \
                 has field ids {:?}",
                segment.uuid, segment.name, segment.fields
            )));
        };
        let Some(coverage) = &segment.fragment_bitmap else {
            return Err(Error::invalid_input(format!(
                "ordered index scan needs fragment coverage, but segment {} of index '{}' \
                 records none",
                segment.uuid, segment.name
            )));
        };
        let ids_are_addresses =
            !dataset.manifest.uses_stable_row_ids() || segment.results_are_row_addrs();
        if !ids_are_addresses {
            return Err(Error::not_supported(format!(
                "ordered index scan needs row addresses, but segment {} of index '{}' stores \
                 stable row ids; rebuild the index to read it in order",
                segment.uuid, segment.name
            )));
        }
        let schema = dataset.schema();
        let ancestry = schema.field_ancestry_by_id(field_id).unwrap_or_default();
        let [.., key] = ancestry.as_slice() else {
            return Err(Error::invalid_input(format!(
                "ordered index scan reads field id {field_id}, but segment {} of index '{}' \
                 names a field the dataset schema does not have",
                segment.uuid, segment.name
            )));
        };
        let column = schema.field_path_minimal(field_id)?;
        let nullable = ancestry.iter().any(|field| field.nullable);
        let output_schema = Arc::new(Schema::new(vec![
            Field::new(&column, key.data_type(), nullable),
            ROW_ADDR_FIELD.clone(),
        ]));
        let ordering = PhysicalSortExpr::new(Arc::new(Column::new(&column, 0)), options);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new_with_orderings(output_schema, [[ordering]]),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let scope = match fragments {
            Some(fragments) => coverage & fragments,
            None => coverage.clone(),
        };

        // Dropping the stale rows is not enough: a row whose overlay moved its
        // key into the requested range would be missing from the output.
        if let Some((version, fragment_id)) = newer_key_overlay(&dataset, &segment, &scope)? {
            return Err(Error::not_supported(format!(
                "ordered index scan cannot read segment {} of index '{}' in key order: a data \
                 overlay committed at version {version} changed column '{column}' in fragment \
                 {fragment_id} after the segment was built at version {}; sort the rows instead",
                segment.uuid, segment.name, segment.dataset_version
            )));
        }

        let index = dataset
            .open_scalar_index(&column, &segment.uuid, &NoOpMetricsCollector)
            .await?;
        let plan = Self {
            dataset,
            segment,
            index,
            column,
            query,
            options,
            scope,
            fetch: None,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        };
        plan.checked_ordered_stream(Arc::new(NoOpMetricsCollector))
            .await?;
        Ok(plan)
    }

    async fn checked_ordered_stream(
        &self,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(index) = self.index.as_ordered() else {
            return Err(Error::not_supported(format!(
                "ordered index scan cannot read segment {} of index '{}' in key order: its {} \
                 index on column '{}' cannot stream rows in key order",
                self.segment.uuid,
                self.segment.name,
                self.index.index_type(),
                self.column
            )));
        };
        let batches = index.search_ordered(self.query.as_deref(), self.options, metrics)?;
        let schema = self.schema();
        let key_type = schema.field(0).data_type();
        let returned = batches.schema();
        let returned_types = returned.fields().iter().map(|field| field.data_type());
        if !returned_types.eq([key_type, &DataType::UInt64]) {
            let returned_fields = returned
                .fields()
                .iter()
                .map(|field| format!("{}: {}", field.name(), field.data_type()))
                .collect::<Vec<_>>();
            return Err(Error::index(format!(
                "ordered index scan expects segment {} of index '{}' to return column '{}' as \
                 {key_type} and row ids as UInt64, but it returned [{}]",
                self.segment.uuid,
                self.segment.name,
                self.column,
                returned_fields.join(", ")
            )));
        }
        Ok(batches)
    }

    async fn live_rows(
        self,
        partition: usize,
    ) -> Result<impl Stream<Item = Result<RecordBatch>> + Send> {
        let pages_loaded = Arc::new(PagesLoaded(
            self.metrics.new_count("pages_loaded", partition),
        ));
        let batches = self.checked_ordered_stream(pages_loaded).await?;
        Ok(LiveRows {
            schema: self.schema(),
            batches,
            dataset: self.dataset,
            scope: self.scope,
            loaded_fragments: RoaringBitmap::new(),
            deleted: RowAddrTreeMap::new(),
            remaining: self.fetch,
            index_name: self.segment.name,
            metrics: RowMetrics::new(&self.metrics, partition),
        }
        .into_stream())
    }
}

/// The version and fragment of the newest data overlay that was committed
/// after `segment` and changes indexed rows in a fragment in `scope`.
fn newer_key_overlay(
    dataset: &Dataset,
    segment: &IndexMetadata,
    scope: &RoaringBitmap,
) -> Result<Option<(u64, u32)>> {
    let mut overlaid = overlaid_fragments(dataset.fragments());
    overlaid.retain(|fragment_id, _| scope.contains(*fragment_id));
    let mut stale = HashMap::new();
    collect_overlay_stale_rows_for_segment(segment, &overlaid, &mut stale, dataset.schema())?;
    let mut newest = None;
    for (&fragment_id, fragment) in &overlaid {
        if !stale.contains_key(&fragment_id) {
            continue;
        }
        for overlay in &fragment.overlays {
            let changed_rows = overlay_exclusion_offsets(
                std::slice::from_ref(overlay),
                &segment.fields,
                segment.dataset_version,
                dataset.schema(),
            )?;
            if !changed_rows.is_empty() {
                newest = newest.max(Some((overlay.committed_version, fragment_id)));
            }
        }
    }
    Ok(newest)
}

struct PagesLoaded(Count);

impl MetricsCollector for PagesLoaded {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.0.add(num_parts);
    }
    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
}

struct RowMetrics {
    scanned: Count,
    deleted: Count,
    excluded: Count,
    deletion_vectors_loaded: Count,
}

impl RowMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            scanned: metrics.new_count(ROWS_SCANNED_METRIC, partition),
            deleted: metrics.new_count("rows_deleted", partition),
            excluded: metrics.new_count("rows_excluded", partition),
            deletion_vectors_loaded: metrics.new_count("deletion_vectors_loaded", partition),
        }
    }
}

struct LiveRows {
    batches: SendableRecordBatchStream,
    dataset: Arc<Dataset>,
    scope: RoaringBitmap,
    loaded_fragments: RoaringBitmap,
    deleted: RowAddrTreeMap,
    remaining: Option<usize>,
    schema: SchemaRef,
    index_name: String,
    metrics: RowMetrics,
}

impl LiveRows {
    fn into_stream(self) -> impl Stream<Item = Result<RecordBatch>> + Send {
        stream::try_unfold(self, |mut rows| async move {
            while rows.remaining != Some(0) {
                let Some(batch) = rows.batches.try_next().await? else {
                    break;
                };
                let live = rows.filter(&batch).await?;
                if live.num_rows() > 0 {
                    return Ok(Some((live, rows)));
                }
            }
            Ok(None)
        })
    }

    async fn load_deletions(&mut self, addrs: &[u64]) -> Result<()> {
        let reached = RoaringBitmap::from_iter(
            addrs
                .iter()
                .map(|&addr| RowAddress::from(addr).fragment_id()),
        );
        let unloaded = (reached & &self.scope) - &self.loaded_fragments;
        let mut deletion_files = Vec::new();
        for fragment_id in &unloaded {
            match self.dataset.find_fragment(fragment_id.into()) {
                None => self.deleted.insert_fragment(fragment_id),
                Some(fragment) => {
                    if let Some(deletion_file) = &fragment.deletion_file {
                        deletion_files.push((fragment_id, deletion_file.clone()));
                    }
                }
            }
        }
        let mut vectors = stream::iter(deletion_files)
            .map(|(fragment_id, deletion_file)| {
                let dataset = self.dataset.clone();
                async move {
                    let vector =
                        read_dataset_deletion_file(&dataset, fragment_id.into(), &deletion_file)
                            .await?;
                    Result::Ok((fragment_id, vector))
                }
            })
            .buffer_unordered(self.dataset.object_store.io_parallelism());
        while let Some((fragment_id, vector)) = vectors.try_next().await? {
            self.deleted
                .insert_bitmap(fragment_id, RoaringBitmap::from(vector.as_ref()));
            self.metrics.deletion_vectors_loaded.add(1);
        }
        self.loaded_fragments |= unloaded;
        Ok(())
    }

    async fn filter(&mut self, batch: &RecordBatch) -> Result<RecordBatch> {
        let addrs = batch.column(1).as_primitive::<UInt64Type>();
        if addrs.null_count() > 0 {
            return Err(Error::index(format!(
                "index '{}' returned a null row address in the batch that starts at key {}",
                self.index_name,
                ScalarValue::try_from_array(batch.column(0), 0)?
            )));
        }
        let addrs = addrs.values();
        let mut live = Vec::new();
        let (mut num_deleted, mut num_excluded) = (0, 0);
        let mut prefix_end = 0;
        for (row, &addr) in addrs.iter().enumerate() {
            if self.remaining == Some(live.len()) {
                break;
            }
            if !self.scope.contains(RowAddress::from(addr).fragment_id()) {
                num_excluded += 1;
                continue;
            }
            if row >= prefix_end {
                prefix_end = match self.remaining {
                    None => addrs.len(),
                    Some(remaining) => {
                        let checked = live.len() + num_deleted;
                        let window = if checked == 0 { remaining } else { checked };
                        addrs[row..]
                            .iter()
                            .enumerate()
                            .filter(|&(_, &addr)| {
                                self.scope.contains(RowAddress::from(addr).fragment_id())
                            })
                            .nth(window - 1)
                            .map_or(addrs.len(), |(offset, _)| row + offset + 1)
                    }
                };
                self.load_deletions(&addrs[row..prefix_end]).await?;
            }
            if self.deleted.contains(addr) {
                num_deleted += 1;
            } else {
                live.push(row as u32);
            }
        }
        self.metrics
            .scanned
            .add(live.len() + num_deleted + num_excluded);
        self.metrics.deleted.add(num_deleted);
        self.metrics.excluded.add(num_excluded);
        self.remaining = self.remaining.map(|remaining| remaining - live.len());
        let live = take_record_batch(batch, &UInt32Array::from(live))?;
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            live.columns().to_vec(),
        )?)
    }
}

impl DisplayAs for OrderedIndexScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let separator = match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "OrderedIndexScan: ")?;
                ", "
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "OrderedIndexScan")?;
                "\n"
            }
        };
        write!(
            f,
            "index={}{separator}order=[{} {}]",
            self.segment.name, self.column, self.options
        )?;
        if let Some(fetch) = self.fetch {
            write!(f, "{separator}fetch={fetch}")?;
        }
        if let Some(query) = &self.query {
            write!(f, "{separator}query={}", query.format(&self.column))?;
        }
        Ok(())
    }
}

impl ExecutionPlan for OrderedIndexScanExec {
    fn name(&self) -> &str {
        "OrderedIndexScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "OrderedIndexScanExec does not have children".to_string(),
            ))
        }
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.as_ref().clone()
        }))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let rows = stream::once(self.clone().live_rows(partition))
            .try_flatten()
            .map_err(DataFusionError::from);
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            rows,
            partition,
            &self.metrics,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            fetch,
            ..self.clone()
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::{HashMap, HashSet};
    use std::iter::StepBy;
    use std::ops::{Bound, Range};

    use arrow_array::{
        ArrayRef, Int32Array, RecordBatchIterator, StringArray, StructArray, UInt64Array,
        types::Int32Type,
    };
    use arrow_buffer::NullBuffer;
    use datafusion::physical_plan::metrics::MetricValue;
    use datafusion::physical_plan::{
        ExecutionPlanProperties, displayable, stream::RecordBatchStreamAdapter,
    };
    use datafusion_physical_expr::LexOrdering;
    use lance_core::ROW_ADDR;
    use lance_datafusion::exec::{LanceExecutionOptions, analyze_plan};
    use lance_index::IndexType;
    use lance_index::scalar::btree::{
        BTreeParameters, DEFAULT_BTREE_BATCH_SIZE, train_btree_index,
    };
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::registry::VALUE_COLUMN_NAME;
    use lance_index::scalar::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
    use rstest::rstest;
    use uuid::Uuid;

    use super::*;
    use crate::dataset::index::LanceIndexStoreExt;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::{WriteDestination, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::copy_test_data_to_tmp;

    const ROWS: i32 = 240;
    const FRAGMENT_ROWS: usize = 60;
    const PAGE_ROWS: u64 = 16;
    const INDEX_NAME: &str = "key_idx";

    const ASC_NULLS_FIRST: SortOptions = SortOptions {
        descending: false,
        nulls_first: true,
    };
    const ASC_NULLS_LAST: SortOptions = SortOptions {
        descending: false,
        nulls_first: false,
    };
    const DESC_NULLS_FIRST: SortOptions = SortOptions {
        descending: true,
        nulls_first: true,
    };
    const DESC_NULLS_LAST: SortOptions = SortOptions {
        descending: true,
        nulls_first: false,
    };

    type Row = (Option<i32>, u64);

    fn fixture_key(row: i32, nulls: bool) -> Option<i32> {
        let unique_key_out_of_address_order = row * 7 % ROWS;
        let null_row = nulls && row % 20 == 3;
        (!null_row).then_some(unique_key_out_of_address_order)
    }

    fn fixture_addr(row: i32) -> u64 {
        let (fragment, offset) = (row as usize / FRAGMENT_ROWS, row as usize % FRAGMENT_ROWS);
        RowAddress::new_from_parts(fragment as u32, offset as u32).into()
    }

    async fn fixture(stable_row_ids: bool, nulls: bool, delete: Option<&str>) -> Arc<Dataset> {
        let keys = Int32Array::from_iter((0..ROWS).map(|row| fixture_key(row, nulls)));
        let mut dataset = indexed_dataset(key_column("key", keys), "key", stable_row_ids).await;
        if let Some(predicate) = delete {
            dataset.delete(predicate).await.unwrap();
        }
        Arc::new(dataset)
    }

    fn key_column(name: &str, keys: Int32Array) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(keys)]).unwrap()
    }

    async fn indexed_dataset(batch: RecordBatch, column: &str, stable_row_ids: bool) -> Dataset {
        let params = WriteParams {
            max_rows_per_file: FRAGMENT_ROWS,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(params),
        )
        .await
        .unwrap();
        let btree = BTreeParameters {
            zone_size: Some(PAGE_ROWS),
            range_id: None,
        };
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree)
            .with_params(&serde_json::to_value(btree).unwrap());
        dataset
            .create_index(
                &[column],
                IndexType::BTree,
                Some(INDEX_NAME.into()),
                &params,
                false,
            )
            .await
            .unwrap();
        dataset
    }

    async fn key_segment(dataset: &Dataset) -> IndexMetadata {
        dataset
            .load_indices_by_name(INDEX_NAME)
            .await
            .unwrap()
            .remove(0)
    }

    async fn live_keys(dataset: &Dataset) -> HashMap<u64, Option<i32>> {
        let batch = dataset
            .scan()
            .project(&["key"])
            .unwrap()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let keys = batch["key"].as_primitive::<Int32Type>();
        let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
        addrs.values().iter().copied().zip(keys).collect()
    }

    fn key_order(a: &Option<i32>, b: &Option<i32>, options: SortOptions) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if options.nulls_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) if options.nulls_first => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) if options.descending => b.cmp(a),
            (Some(a), Some(b)) => a.cmp(b),
        }
    }

    async fn run(plan: Arc<dyn ExecutionPlan>) -> Vec<Row> {
        let batches: Vec<RecordBatch> = plan
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|batch| {
                let keys = batch.column(0).as_primitive::<Int32Type>();
                let addrs = batch[ROW_ADDR].as_primitive::<UInt64Type>();
                keys.iter()
                    .zip(addrs.values().iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
        plan.metrics()
            .unwrap()
            .sum_by_name(name)
            .map_or(0, |value| value.as_usize())
    }

    #[rstest]
    #[case::deletes_in_first_page(Some("key < 4"), None, None, ASC_NULLS_LAST, Some(10), false)]
    #[case::deletes_across_pages(Some("key % 3 = 0"), None, None, DESC_NULLS_LAST, Some(50), false)]
    #[case::fetch_zero(None, None, None, ASC_NULLS_FIRST, Some(0), false)]
    #[case::fetch_past_end(Some("key % 5 = 1"), None, None, ASC_NULLS_LAST, Some(1000), false)]
    #[case::every_row_deleted(Some("key >= 0"), None, None, ASC_NULLS_LAST, Some(10), false)]
    #[case::allow_list(
        Some("key < 30"),
        Some(&[0, 2][..]),
        None,
        ASC_NULLS_LAST,
        Some(30),
        false
    )]
    #[case::asc_nulls_first(None, None, None, ASC_NULLS_FIRST, None, true)]
    #[case::asc_nulls_last(None, None, None, ASC_NULLS_LAST, None, true)]
    #[case::desc_nulls_first(None, None, None, DESC_NULLS_FIRST, None, true)]
    #[case::desc_nulls_last(None, None, None, DESC_NULLS_LAST, None, true)]
    #[case::range_query(
        Some("key % 3 = 0"),
        None,
        Some(40..120),
        DESC_NULLS_LAST,
        Some(20),
        false
    )]
    #[tokio::test]
    async fn scan_returns_live_rows_in_key_order(
        #[case] delete: Option<&str>,
        #[case] fragments: Option<&[u32]>,
        #[case] range: Option<Range<i32>>,
        #[case] options: SortOptions,
        #[case] fetch: Option<usize>,
        #[case] nulls: bool,
        #[values(false, true)] stable_row_ids: bool,
    ) {
        let dataset = fixture(stable_row_ids, nulls, delete).await;
        let allowed = fragments.map(|ids| RoaringBitmap::from_iter(ids.iter().copied()));
        let query = range.clone().map(|range| {
            Arc::new(SargableQuery::Range(
                Bound::Included(ScalarValue::Int32(Some(range.start))),
                Bound::Excluded(ScalarValue::Int32(Some(range.end))),
            )) as Arc<dyn AnyQuery>
        });
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            query,
            options,
            allowed.as_ref(),
        )
        .await
        .unwrap()
        .with_fetch(fetch)
        .unwrap();

        let in_scope = |addr: u64| {
            allowed
                .as_ref()
                .is_none_or(|ids| ids.contains(RowAddress::from(addr).fragment_id()))
        };
        let selected = |key: Option<i32>| {
            range
                .as_ref()
                .is_none_or(|range| key.is_some_and(|key| range.contains(&key)))
        };
        let live = live_keys(&dataset).await;
        let mut expected = live
            .iter()
            .filter(|(addr, key)| in_scope(**addr) && selected(**key))
            .map(|(_, key)| *key)
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| key_order(a, b, options));
        expected.truncate(fetch.unwrap_or(usize::MAX));

        let rows = run(plan.clone()).await;
        let keys = rows.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        assert_eq!(keys, expected, "keys differ from the sorted scan");
        for (key, addr) in &rows {
            assert_eq!(
                live.get(addr),
                Some(key),
                "row {addr} is not live with key {key:?}"
            );
            assert!(in_scope(*addr), "row {addr} is outside the allow-list");
        }
        let addrs = rows.iter().map(|(_, addr)| *addr).collect::<HashSet<_>>();
        assert_eq!(addrs.len(), rows.len(), "a row was emitted twice");

        let mut index_rows = (0..ROWS)
            .map(|row| (fixture_key(row, nulls), fixture_addr(row)))
            .filter(|(key, _)| selected(*key))
            .collect::<Vec<_>>();
        index_rows.sort_by(|a, b| key_order(&a.0, &b.0, options));
        let (mut emitted, mut deleted, mut excluded) = (0, 0, 0);
        for (_, addr) in index_rows {
            if fetch == Some(emitted) {
                break;
            }
            if !in_scope(addr) {
                excluded += 1;
            } else if !live.contains_key(&addr) {
                deleted += 1;
            } else {
                emitted += 1;
            }
        }
        assert_eq!(metric(plan.as_ref(), "rows_deleted"), deleted);
        assert_eq!(metric(plan.as_ref(), "rows_excluded"), excluded);
        assert_eq!(
            metric(plan.as_ref(), "rows_scanned"),
            rows.len() + deleted + excluded
        );
    }

    #[tokio::test]
    async fn fetch_backfills_past_deleted_rows() {
        let dataset = fixture(false, false, Some("key < 4")).await;
        let segment = key_segment(&dataset).await;
        let plan = || async {
            OrderedIndexScanExec::try_new(
                dataset.clone(),
                segment.clone(),
                None,
                ASC_NULLS_LAST,
                None,
            )
            .await
            .unwrap()
            .with_fetch(Some(10))
            .unwrap()
        };

        let keys = run(plan().await)
            .await
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(keys, (4..14).map(Some).collect::<Vec<_>>());

        let analyzed = analyze_plan(plan().await, LanceExecutionOptions::default())
            .await
            .unwrap();
        let line = analyzed
            .lines()
            .find(|line| line.contains("OrderedIndexScan"))
            .unwrap();
        assert!(
            line.contains("index=key_idx, order=[key ASC NULLS LAST], fetch=10"),
            "{analyzed}"
        );
        let metrics = line
            .split_once("metrics=[")
            .unwrap()
            .1
            .trim_end_matches(']')
            .split(", ")
            .collect::<HashSet<_>>();
        for expected in [
            "output_rows=10",
            "rows_deleted=4",
            "rows_scanned=14",
            "pages_loaded=1",
        ] {
            assert!(metrics.contains(expected), "{expected} in {analyzed}");
        }
    }

    #[rstest]
    #[case::asc_nulls_first(ASC_NULLS_FIRST, "key ASC NULLS FIRST")]
    #[case::asc_nulls_last(ASC_NULLS_LAST, "key ASC NULLS LAST")]
    #[case::desc_nulls_first(DESC_NULLS_FIRST, "key DESC NULLS FIRST")]
    #[case::desc_nulls_last(DESC_NULLS_LAST, "key DESC NULLS LAST")]
    #[tokio::test]
    async fn plan_declares_requested_order_and_fetch(
        #[case] options: SortOptions,
        #[case] order: &str,
    ) {
        let dataset = fixture(false, false, None).await;
        let segment = key_segment(&dataset).await;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            OrderedIndexScanExec::try_new(dataset.clone(), segment.clone(), None, options, None)
                .await
                .unwrap(),
        );
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            Arc::new(Column::new("key", 0)),
            options,
        )]);
        assert_eq!(plan.output_ordering(), ordering.as_ref());
        assert_eq!(
            plan.schema().field(0),
            &Field::new("key", DataType::Int32, true)
        );
        assert_eq!(plan.schema().field(1), &*ROW_ADDR_FIELD);
        assert_eq!(plan.fetch(), None);

        let limited = plan.with_fetch(Some(7)).unwrap();
        assert_eq!(limited.fetch(), Some(7));
        assert_eq!(limited.output_ordering(), ordering.as_ref());
        assert_eq!(limited.schema(), plan.schema());
        assert_eq!(limited.with_fetch(None).unwrap().fetch(), None);
        assert_eq!(
            displayable(limited.as_ref()).one_line().to_string().trim(),
            format!("OrderedIndexScan: index=key_idx, order=[{order}], fetch=7")
        );

        let range = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(40))),
            Bound::Excluded(ScalarValue::Int32(Some(120))),
        );
        let filtered =
            OrderedIndexScanExec::try_new(dataset, segment, Some(Arc::new(range)), options, None)
                .await
                .unwrap();
        assert_eq!(
            displayable(&filtered).one_line().to_string().trim(),
            format!(
                "OrderedIndexScan: index=key_idx, order=[{order}], query=key >= 40 && key < 120"
            )
        );
    }

    #[rstest]
    #[case::no_fragment_coverage(|segment: &mut IndexMetadata| segment.fragment_bitmap = None)]
    #[case::two_columns(|segment: &mut IndexMetadata| segment.fields.push(segment.fields[0]))]
    #[case::unknown_field(|segment: &mut IndexMetadata| segment.fields = vec![999])]
    #[tokio::test]
    async fn malformed_segment_is_rejected(#[case] malform: fn(&mut IndexMetadata)) {
        let dataset = fixture(false, false, None).await;
        let mut segment = key_segment(&dataset).await;
        malform(&mut segment);
        let error =
            OrderedIndexScanExec::try_new(dataset, segment.clone(), None, ASC_NULLS_FIRST, None)
                .await
                .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("index 'key_idx'"), "{error}");
        assert!(
            error.to_string().contains(&segment.uuid.to_string()),
            "{error}"
        );
    }

    #[tokio::test]
    async fn key_type_mismatch_names_the_index_segment_column_and_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let names = StringArray::from_iter_values((0..ROWS).map(|row| row.to_string()));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..ROWS)),
                Arc::new(names),
            ],
        )
        .unwrap();
        let dataset = Arc::new(indexed_dataset(batch, "key", false).await);
        let mut segment = key_segment(&dataset).await;
        segment.fields = vec![dataset.schema().field("name").unwrap().id];

        let error =
            OrderedIndexScanExec::try_new(dataset, segment.clone(), None, ASC_NULLS_FIRST, None)
                .await
                .unwrap_err();
        assert!(matches!(error, Error::Index { .. }), "{error}");
        for expected in [
            "index 'key_idx'",
            &segment.uuid.to_string(),
            "column 'name' as Utf8",
            "it returned [value: Int32, ",
        ] {
            assert!(error.to_string().contains(expected), "{expected}: {error}");
        }
    }

    #[tokio::test]
    async fn stable_row_id_segment_is_rejected() {
        let dir = copy_test_data_to_tmp("v14.0.0-beta.15/stable_row_id_btree.lance").unwrap();
        let dataset = Arc::new(Dataset::open(&dir.path_str()).await.unwrap());
        let segment = dataset
            .load_indices_by_name("value_idx")
            .await
            .unwrap()
            .remove(0);
        let error = OrderedIndexScanExec::try_new(dataset, segment, None, ASC_NULLS_FIRST, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("index 'value_idx'"), "{error}");
    }

    #[tokio::test]
    async fn unordered_index_segment_is_rejected() {
        let batch = key_column("key", Int32Array::from_iter_values(0..ROWS));
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["key"],
                IndexType::Bitmap,
                Some(INDEX_NAME.into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                false,
            )
            .await
            .unwrap();
        let segment = key_segment(&dataset).await;

        let error = OrderedIndexScanExec::try_new(
            Arc::new(dataset),
            segment.clone(),
            None,
            ASC_NULLS_FIRST,
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        for expected in [
            "index 'key_idx'",
            &segment.uuid.to_string(),
            "Bitmap index on column 'key'",
        ] {
            assert!(error.to_string().contains(expected), "{expected}: {error}");
        }
    }

    #[tokio::test]
    async fn pages_out_of_key_order_fail_at_construction() {
        let dataset = fixture(false, false, None).await;
        let segment = key_segment(&dataset).await;
        let mut rows = live_keys(&dataset)
            .await
            .into_iter()
            .map(|(addr, key)| (key.unwrap(), addr))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        let (low_partition, high_partition) = rows.split_at(8 * PAGE_ROWS as usize);
        let merged_out_of_order = [high_partition, low_partition].concat();
        let page_schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let pages = RecordBatch::try_new(
            page_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    merged_out_of_order.iter().map(|row| row.0),
                )),
                Arc::new(UInt64Array::from_iter_values(
                    merged_out_of_order.iter().map(|row| row.1),
                )),
            ],
        )
        .unwrap();
        let uuid = Uuid::new_v4();
        let store = LanceIndexStore::from_dataset_for_new(&dataset, &uuid).unwrap();
        train_btree_index(
            Box::pin(RecordBatchStreamAdapter::new(
                page_schema,
                stream::iter([Ok(pages)]),
            )),
            &store,
            PAGE_ROWS,
            None,
            None,
        )
        .await
        .unwrap();
        let out_of_order = IndexMetadata {
            uuid,
            files: None,
            ..segment.clone()
        };
        let mut dataset = dataset.as_ref().clone();
        let operation = Operation::CreateIndex {
            new_indices: vec![out_of_order],
            removed_indices: vec![segment],
        };
        dataset
            .apply_commit(
                Transaction::new(dataset.manifest.version, operation, None),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let segment = key_segment(&dataset).await;
        let error =
            OrderedIndexScanExec::try_new(Arc::new(dataset), segment, None, ASC_NULLS_LAST, None)
                .await
                .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains("previous max above current min"),
            "{error}"
        );
    }

    async fn overlay_value(
        dataset: Arc<Dataset>,
        column: &str,
        row_key: i32,
        new_value: i32,
    ) -> Arc<Dataset> {
        let (addr, _) = live_keys(&dataset)
            .await
            .into_iter()
            .find(|(_, key)| *key == Some(row_key))
            .unwrap();
        let fragment_id = RowAddress::from(addr).fragment_id() as usize;
        let field_id = dataset.schema().field(column).unwrap().id;
        let mut overlay = dataset
            .get_fragment(fragment_id)
            .unwrap()
            .write_overlay(&dataset.schema().project_by_ids(&[field_id], true))
            .await
            .unwrap();
        let values = Arc::new(Schema::new(vec![
            ROW_ADDR_FIELD.clone(),
            Field::new(column, DataType::Int32, true),
        ]));
        let values = RecordBatch::try_new(
            values,
            vec![
                Arc::new(UInt64Array::from(vec![addr])),
                Arc::new(Int32Array::from(vec![new_value])),
            ],
        )
        .unwrap();
        overlay.write_batch(&values).await.unwrap();
        let group = overlay.finish().await.unwrap().unwrap();
        let read_version = dataset.version().version;
        let dataset = Dataset::commit(
            WriteDestination::Dataset(dataset),
            Operation::DataOverlay {
                groups: vec![group],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();
        Arc::new(dataset)
    }

    #[rstest]
    #[case::key_moves_into_range(10, 50)]
    #[case::key_moves_out_of_range(50, 10)]
    #[tokio::test]
    async fn newer_key_overlay_is_not_supported(#[case] old_key: i32, #[case] new_key: i32) {
        let dataset =
            overlay_value(fixture(false, false, None).await, "key", old_key, new_key).await;
        let segment = key_segment(&dataset).await;
        let range = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(40))),
            Bound::Excluded(ScalarValue::Int32(Some(120))),
        );
        let error = OrderedIndexScanExec::try_new(
            dataset.clone(),
            segment.clone(),
            Some(Arc::new(range)),
            ASC_NULLS_LAST,
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        for expected in [
            "index 'key_idx'".to_string(),
            segment.uuid.to_string(),
            "column 'key'".to_string(),
            format!("version {}", dataset.version().version),
        ] {
            assert!(error.to_string().contains(&expected), "{expected}: {error}");
        }
    }

    #[tokio::test]
    async fn key_overlay_outside_the_fragment_scope_is_read() {
        let dataset = overlay_value(fixture(false, false, None).await, "key", 10, 50).await;
        let overlaid = dataset
            .fragments()
            .iter()
            .find(|fragment| !fragment.overlays.is_empty())
            .unwrap()
            .id as u32;
        let mut allowed = dataset.fragment_bitmap().as_ref().clone();
        allowed.remove(overlaid);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            ASC_NULLS_LAST,
            Some(&allowed),
        )
        .await
        .unwrap();

        let rows = run(Arc::new(plan)).await;
        let mut expected = (0..ROWS)
            .filter(|row| allowed.contains(RowAddress::from(fixture_addr(*row)).fragment_id()))
            .map(|row| (fixture_key(row, false), fixture_addr(row)))
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(rows, expected);
    }

    #[rstest]
    #[case::other_column("other", false)]
    #[case::index_rebuilt_after_the_overlay("key", true)]
    #[tokio::test]
    async fn overlay_the_segment_already_reflects_is_read(
        #[case] column: &str,
        #[case] rebuild_index: bool,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, true),
            Field::new("other", DataType::Int32, true),
        ]));
        let keys = Int32Array::from_iter((0..ROWS).map(|row| fixture_key(row, false)));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(keys),
                Arc::new(Int32Array::from_iter_values(0..ROWS)),
            ],
        )
        .unwrap();
        let dataset = Arc::new(indexed_dataset(batch, "key", false).await);
        let mut dataset = overlay_value(dataset, column, 10, 1000)
            .await
            .as_ref()
            .clone();
        if rebuild_index {
            dataset
                .create_index(
                    &["key"],
                    IndexType::BTree,
                    Some(INDEX_NAME.into()),
                    &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                    true,
                )
                .await
                .unwrap();
        }
        let dataset = Arc::new(dataset);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            ASC_NULLS_LAST,
            None,
        )
        .await
        .unwrap();

        let mut expected = live_keys(&dataset)
            .await
            .into_iter()
            .map(|(addr, key)| (key, addr))
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(run(Arc::new(plan)).await, expected);
    }

    #[rstest]
    #[case::asc_nulls_first(ASC_NULLS_FIRST)]
    #[case::asc_nulls_last(ASC_NULLS_LAST)]
    #[case::desc_nulls_first(DESC_NULLS_FIRST)]
    #[case::desc_nulls_last(DESC_NULLS_LAST)]
    #[tokio::test]
    async fn key_under_a_nullable_parent_is_nullable(#[case] options: SortOptions) {
        let null_parent = |row: i32| row % 20 == 3;
        let parent = StructArray::new(
            vec![Field::new("key", DataType::Int32, false)].into(),
            vec![Arc::new(Int32Array::from_iter_values(0..ROWS)) as ArrayRef],
            Some(NullBuffer::from_iter(
                (0..ROWS).map(|row| !null_parent(row)),
            )),
        );
        let schema = Schema::new(vec![Field::new("parent", parent.data_type().clone(), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(parent)]).unwrap();
        let dataset = Arc::new(indexed_dataset(batch, "parent.key", false).await);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            options,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            plan.schema().field(0),
            &Field::new("parent.key", DataType::Int32, true)
        );

        let mut expected = (0..ROWS)
            .map(|row| (!null_parent(row)).then_some(row))
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| key_order(a, b, options));
        let keys = run(Arc::new(plan))
            .await
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(keys, expected);
    }

    #[tokio::test]
    async fn hyphenated_column_keeps_its_name() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "my-col",
            DataType::Int32,
            false,
        )]));
        let keys = Int32Array::from_iter_values((0..ROWS).rev());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(keys)]).unwrap();
        let dataset = Arc::new(indexed_dataset(batch, "my-col", false).await);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            ASC_NULLS_LAST,
            None,
        )
        .await
        .unwrap()
        .with_fetch(Some(3))
        .unwrap();

        assert_eq!(
            plan.schema().field(0),
            &Field::new("my-col", DataType::Int32, false)
        );
        assert_eq!(
            displayable(plan.as_ref()).one_line().to_string().trim(),
            "OrderedIndexScan: index=key_idx, order=[my-col ASC NULLS LAST], fetch=3"
        );
        let keys = run(plan)
            .await
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        assert_eq!(keys, [Some(0), Some(1), Some(2)]);
    }

    fn count_metrics(plan: &dyn ExecutionPlan) -> Vec<(String, usize)> {
        let mut counts = plan
            .metrics()
            .unwrap()
            .aggregate_by_name()
            .iter()
            .map(|metric| metric.value())
            .filter(|value| {
                matches!(
                    value,
                    MetricValue::OutputRows(_)
                        | MetricValue::OutputBatches(_)
                        | MetricValue::Count { .. }
                )
            })
            .map(|value| (value.name().to_string(), value.as_usize()))
            .collect::<Vec<_>>();
        counts.sort_unstable();
        counts
    }

    #[tokio::test]
    async fn reset_state_starts_fresh_metrics() {
        let dataset = fixture(false, false, Some("key < 4")).await;
        let segment = key_segment(&dataset).await;
        let plan = || async {
            OrderedIndexScanExec::try_new(
                dataset.clone(),
                segment.clone(),
                None,
                ASC_NULLS_LAST,
                None,
            )
            .await
            .unwrap()
            .with_fetch(Some(10))
            .unwrap()
        };

        let fresh = plan().await;
        run(fresh.clone()).await;
        let executed = plan().await;
        run(executed.clone()).await;
        let reset = executed.reset_state().unwrap();
        run(reset.clone()).await;

        assert_eq!(metric(fresh.as_ref(), "rows_deleted"), 4);
        assert_eq!(count_metrics(reset.as_ref()), count_metrics(fresh.as_ref()));
    }

    fn key_in_fragment_order(row: i32) -> i32 {
        row
    }

    fn key_interleaving_the_first_two_fragments(row: i32) -> i32 {
        let fragment_rows = FRAGMENT_ROWS as i32;
        if row < 2 * fragment_rows {
            row % fragment_rows * 2 + row / fragment_rows
        } else {
            row
        }
    }

    #[rstest]
    #[case::fetch_zero(key_in_fragment_order, Some(0), 0)]
    #[case::keys_in_fragment_order(key_in_fragment_order, Some(10), 1)]
    #[case::keys_interleaving_two_fragments(key_interleaving_the_first_two_fragments, Some(10), 2)]
    #[case::no_fetch(key_in_fragment_order, None, 4)]
    #[tokio::test]
    async fn deletion_vectors_load_when_a_batch_first_reaches_their_fragment(
        #[case] key: fn(i32) -> i32,
        #[case] fetch: Option<usize>,
        #[case] vectors_loaded: usize,
    ) {
        let keys = Int32Array::from_iter_values((0..ROWS).map(key));
        let mut dataset = indexed_dataset(key_column("key", keys), "key", false).await;
        dataset.delete("key % 5 = 0").await.unwrap();
        let dataset = Arc::new(dataset);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            ASC_NULLS_LAST,
            None,
        )
        .await
        .unwrap()
        .with_fetch(fetch)
        .unwrap();

        let keys = run(plan.clone())
            .await
            .into_iter()
            .map(|(key, _)| key.unwrap())
            .collect::<Vec<_>>();
        let expected = (0..ROWS)
            .filter(|key| key % 5 != 0)
            .take(fetch.unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        assert_eq!(keys, expected);
        assert_eq!(
            metric(plan.as_ref(), "deletion_vectors_loaded"),
            vectors_loaded
        );
    }

    #[rstest]
    #[case::fetch_one(0, None, Some(1), 2)]
    #[case::fetch_ten(0, None, Some(10), 20)]
    #[case::fetch_hundred(0, None, Some(100), 64)]
    #[case::no_fetch(0, None, None, 64)]
    #[case::fetch_one_after_five_deleted_keys(5, None, Some(1), 8)]
    #[case::fetch_one_after_excluded_fragments(0, Some((32..64).step_by(1)), Some(1), 1)]
    #[case::fetch_one_with_every_other_fragment_excluded(3, Some((0..64).step_by(2)), Some(1), 4)]
    #[tokio::test]
    async fn deletion_vectors_load_for_the_rows_a_fetch_needs(
        #[case] deleted_below: i32,
        #[case] allowed: Option<StepBy<Range<u32>>>,
        #[case] fetch: Option<usize>,
        #[case] vectors_loaded: usize,
    ) {
        const FRAGMENTS: i32 = 64;
        let fragment_rows = DEFAULT_BTREE_BATCH_SIZE as i32 / FRAGMENTS;
        assert_eq!(
            fragment_rows, FRAGMENTS,
            "each fragment holds as many rows as there are fragments"
        );
        let keys = Int32Array::from_iter_values((0..FRAGMENTS).flat_map(|fragment| {
            (0..fragment_rows).map(move |offset| offset * FRAGMENTS + fragment)
        }));
        let batch = key_column("key", keys);
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: fragment_rows as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["key"],
                IndexType::BTree,
                Some(INDEX_NAME.into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
        let one_per_fragment = FRAGMENTS + 1;
        dataset
            .delete(&format!(
                "key % {one_per_fragment} = 0 OR key < {deleted_below}"
            ))
            .await
            .unwrap();
        let dataset = Arc::new(dataset);
        let allowed = allowed.map(RoaringBitmap::from_iter);
        let plan = OrderedIndexScanExec::try_new(
            dataset.clone(),
            key_segment(&dataset).await,
            None,
            ASC_NULLS_LAST,
            allowed.as_ref(),
        )
        .await
        .unwrap()
        .with_fetch(fetch)
        .unwrap();

        let keys = run(plan.clone())
            .await
            .into_iter()
            .map(|(key, _)| key.unwrap())
            .collect::<Vec<_>>();
        let live = live_keys(&dataset).await;
        let in_scope = |addr: u64| {
            allowed
                .as_ref()
                .is_none_or(|ids| ids.contains(RowAddress::from(addr).fragment_id()))
        };
        let mut expected = live
            .iter()
            .filter(|(addr, _)| in_scope(**addr))
            .map(|(_, key)| key.unwrap())
            .collect::<Vec<_>>();
        expected.sort_unstable();
        expected.truncate(fetch.unwrap_or(usize::MAX));
        assert_eq!(keys, expected);

        let (mut emitted, mut deleted, mut excluded) = (0, 0, 0);
        for key in 0..FRAGMENTS * fragment_rows {
            if fetch == Some(emitted) {
                break;
            }
            let addr = u64::from(RowAddress::new_from_parts(
                (key % FRAGMENTS) as u32,
                (key / FRAGMENTS) as u32,
            ));
            if !in_scope(addr) {
                excluded += 1;
            } else if live.contains_key(&addr) {
                emitted += 1;
            } else {
                deleted += 1;
            }
        }
        assert_eq!(metric(plan.as_ref(), "rows_excluded"), excluded);
        assert_eq!(metric(plan.as_ref(), "rows_deleted"), deleted);
        assert_eq!(
            metric(plan.as_ref(), "rows_scanned"),
            emitted + deleted + excluded
        );
        assert_eq!(
            metric(plan.as_ref(), "pages_loaded"),
            1,
            "the index is one page"
        );
        assert_eq!(
            metric(plan.as_ref(), "deletion_vectors_loaded"),
            vectors_loaded
        );
    }

    #[rstest]
    #[case::nothing_deleted(RowAddrTreeMap::new())]
    #[case::rows_deleted(RowAddrTreeMap::from_iter([5_u64]))]
    #[tokio::test]
    async fn null_row_address_is_an_error_naming_the_index(#[case] deleted: RowAddrTreeMap) {
        let page_schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, true),
        ]));
        let page = RecordBatch::try_new(
            page_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![11, 12])),
                Arc::new(UInt64Array::from(vec![None, Some(7)])),
            ],
        )
        .unwrap();
        let rows = LiveRows {
            batches: Box::pin(RecordBatchStreamAdapter::new(
                page_schema,
                stream::iter([Ok(page)]),
            )),
            dataset: fixture(false, false, None).await,
            scope: RoaringBitmap::from_iter([0]),
            loaded_fragments: RoaringBitmap::from_iter([0]),
            deleted,
            remaining: Some(1),
            schema: Arc::new(Schema::new(vec![
                Field::new("key", DataType::Int32, true),
                ROW_ADDR_FIELD.clone(),
            ])),
            index_name: "key_idx".into(),
            metrics: RowMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
        };
        let error = rows
            .into_stream()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains(
                "index 'key_idx' returned a null row address in the batch that starts at key 11"
            ),
            "{error}"
        );
    }
}
