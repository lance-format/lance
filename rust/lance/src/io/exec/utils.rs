// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::datatypes::Field as LanceField;
use crate::datatypes::Schema as LanceSchema;
use lance_datafusion::utils::{
    BYTES_READ_METRIC, ExecutionPlanMetricsSetExt, INDEX_CACHE_HITS_METRIC,
    INDEX_CACHE_MISSES_METRIC, INDEX_COMPARISONS_METRIC, INDICES_LOADED_METRIC, IOPS_METRIC,
    PARTS_LOADED_METRIC, REQUESTS_METRIC,
};
use lance_encoding::decoder::estimate_bytes_per_row;
use lance_index::metrics::MetricsCollector;
use lance_io::scheduler::{IoStats, ScanScheduler, ScanStats};
use lance_table::format::IndexMetadata;
use pin_project::pin_project;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricValue, Time,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use futures::future::{BoxFuture, Shared};
use futures::stream::FuturesUnordered;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use lance_arrow::DataTypeExt as _;
use lance_core::error::{CloneableResult, Error};
use lance_core::utils::futures::{Capacity, SharedStreamExt};
use lance_core::{ROW_ID, Result};
use lance_index::prefilter::FilterLoader;
use lance_select::{RowAddrMask, RowAddrTreeMap, result::IndexExprResult};
use tracing::Instrument;

use super::row_addr_mask::MaskAndLoader;
use crate::Dataset;
use crate::index::prefilter::DatasetPreFilter;

/// Open fragments on cancellation-safe tasks while preserving the stream's
/// ordering and readahead bound.
pub(crate) fn buffered_fragment_opens<S, Open, OpenFuture, Reader>(
    fragments: S,
    fragment_readahead: usize,
    mut open: Open,
) -> impl Stream<Item = DataFusionResult<Reader>>
where
    S: Stream + Send,
    Open: FnMut(S::Item) -> OpenFuture + Send,
    OpenFuture: Future<Output = DataFusionResult<Reader>> + Send + 'static,
    Reader: Send + 'static,
{
    fragments
        .map(move |fragment| {
            SpawnedTask::spawn(open(fragment).in_current_span()).map(|task_result| {
                task_result.map_err(|error| DataFusionError::External(Box::new(error)))?
            })
        })
        .buffered(fragment_readahead)
}

#[derive(Debug, Clone)]
pub enum PreFilterSource {
    /// The prefilter input is an array of row ids that match the filter condition
    FilteredRowIds(Arc<dyn ExecutionPlan>),
    /// The prefilter input is a selection vector from an index query
    ScalarIndexQuery(Arc<dyn ExecutionPlan>),
    /// There is no prefilter
    None,
}

type SharedPreFilterFuture = Shared<BoxFuture<'static, CloneableResult<Arc<RowAddrMask>>>>;

struct SharedPreFilterEntry {
    context: std::sync::Weak<datafusion::execution::TaskContext>,
    future: SharedPreFilterFuture,
    waiters: usize,
    is_complete: bool,
    generation: u64,
}

/// Query-plan-local materialization state for a MultiMatch base prefilter.
///
/// Entries are keyed by task-context identity and partition. This prevents a
/// reused physical plan from carrying a mask into a later query and keeps an
/// accidental multi-partition execution from sharing across input partitions.
/// The mutex is held only while installing or cloning a future; prefilter
/// execution never runs under it.
struct SharedPreFilterMaterialization {
    queries: Mutex<HashMap<(usize, usize), SharedPreFilterEntry>>,
    next_generation: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for SharedPreFilterMaterialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let queries = self
            .queries
            .lock()
            .map(|queries| queries.len())
            .unwrap_or_default();
        f.debug_struct("SharedPreFilterMaterialization")
            .field("queries", &queries)
            .finish()
    }
}

impl SharedPreFilterMaterialization {
    fn new() -> Self {
        Self {
            queries: Mutex::new(HashMap::new()),
            next_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct SharedPreFilterExec {
    source: Arc<dyn ExecutionPlan>,
    materialization: Arc<SharedPreFilterMaterialization>,
    properties: Arc<PlanProperties>,
}

impl SharedPreFilterExec {
    fn new(
        source: Arc<dyn ExecutionPlan>,
        materialization: Arc<SharedPreFilterMaterialization>,
    ) -> Self {
        Self {
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(source.schema()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            source,
            materialization,
        }
    }
}

impl DisplayAs for SharedPreFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SharedMultiMatchPrefilter")
    }
}

impl ExecutionPlan for SharedPreFilterExec {
    fn name(&self) -> &str {
        "SharedPreFilterExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.source]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let source = match children.len() {
            1 => children.pop().ok_or_else(|| {
                DataFusionError::Internal(
                    "shared MultiMatch prefilter lost its source child".to_string(),
                )
            })?,
            count => {
                return Err(DataFusionError::Internal(format!(
                    "shared MultiMatch prefilter expected one child, got {count}"
                )));
            }
        };
        Ok(Arc::new(Self::new(source, self.materialization.clone())))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Err(DataFusionError::Internal(
            "shared MultiMatch prefilter must be materialized by its FTS consumer".to_string(),
        ))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

pub(crate) struct PreFilterMasks {
    pub overlay_block: Option<RowAddrMask>,
    pub external_mask: Option<Arc<RowAddrMask>>,
}

impl PreFilterSource {
    /// Return a plan-local shared form for a MultiMatch with multiple fields.
    /// No-filter and already-shared sources retain their existing identity.
    pub(crate) fn shared_for_multimatch_fields(&self, field_count: usize) -> Vec<Self> {
        if field_count <= 1 {
            return vec![self.clone(); field_count];
        }
        match self {
            Self::FilteredRowIds(source) | Self::ScalarIndexQuery(source) => {
                let materialization = Arc::new(SharedPreFilterMaterialization::new());
                (0..field_count)
                    .map(|_| {
                        let shared = Arc::new(SharedPreFilterExec::new(
                            source.clone(),
                            materialization.clone(),
                        ));
                        if matches!(self, Self::FilteredRowIds(_)) {
                            Self::FilteredRowIds(shared)
                        } else {
                            Self::ScalarIndexQuery(shared)
                        }
                    })
                    .collect()
            }
            Self::None => vec![self.clone(); field_count],
        }
    }

    pub(crate) fn execution_plan(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        match self {
            Self::FilteredRowIds(source) | Self::ScalarIndexQuery(source) => Some(source),
            Self::None => None,
        }
    }

    pub(crate) fn with_execution_plan(
        &self,
        source: Arc<dyn ExecutionPlan>,
    ) -> DataFusionResult<Self> {
        match self {
            Self::FilteredRowIds(_) => Ok(Self::FilteredRowIds(source)),
            Self::ScalarIndexQuery(_) => Ok(Self::ScalarIndexQuery(source)),
            Self::None => Err(DataFusionError::Internal(
                "prefilter source received an unexpected execution-plan child".to_string(),
            )),
        }
    }
}

struct SharedPreFilterWaiter {
    materialization: Arc<SharedPreFilterMaterialization>,
    key: (usize, usize),
    generation: u64,
}

impl SharedPreFilterWaiter {
    fn mark_complete(&self) {
        if let Ok(mut queries) = self.materialization.queries.lock()
            && let Some(entry) = queries.get_mut(&self.key)
            && entry.generation == self.generation
        {
            entry.is_complete = true;
        }
    }
}

impl Drop for SharedPreFilterWaiter {
    fn drop(&mut self) {
        let Ok(mut queries) = self.materialization.queries.lock() else {
            return;
        };
        let should_remove = if let Some(entry) = queries.get_mut(&self.key)
            && entry.generation == self.generation
        {
            let Some(waiters) = entry.waiters.checked_sub(1) else {
                debug_assert!(false, "shared prefilter waiter count underflowed");
                return;
            };
            entry.waiters = waiters;
            entry.waiters == 0 && !entry.is_complete
        } else {
            false
        };
        if should_remove {
            queries.remove(&self.key);
        }
    }
}

fn shared_prefilter_future(
    materialization: Arc<SharedPreFilterMaterialization>,
    source: Arc<dyn ExecutionPlan>,
    is_scalar_index_query: bool,
    context: Arc<datafusion::execution::TaskContext>,
    partition: usize,
) -> BoxFuture<'static, Result<Arc<RowAddrMask>>> {
    async move {
        let context_id = Arc::as_ptr(&context) as usize;
        let key = (context_id, partition);
        let (future, generation) = {
            let mut queries = materialization.queries.lock().map_err(|_| {
                Error::internal("MultiMatch prefilter materialization lock was poisoned")
            })?;
            queries.retain(|_, entry| entry.context.strong_count() > 0);
            if let Some(entry) = queries.get_mut(&key) {
                entry.waiters = entry.waiters.checked_add(1).ok_or_else(|| {
                    Error::internal("MultiMatch prefilter waiter count overflowed")
                })?;
                (entry.future.clone(), entry.generation)
            } else {
                let generation = materialization
                    .next_generation
                    .fetch_update(
                        std::sync::atomic::Ordering::Relaxed,
                        std::sync::atomic::Ordering::Relaxed,
                        |generation| generation.checked_add(1),
                    )
                    .map_err(|_| {
                        Error::internal("MultiMatch prefilter generation counter overflowed")
                    })?;
                let entry = SharedPreFilterEntry {
                    context: Arc::downgrade(&context),
                    future: {
                        async move {
                            let result = async move {
                                let stream = source.execute(partition, context)?;
                                if is_scalar_index_query {
                                    Box::new(SelectionVectorToPrefilter(stream)).load().await
                                } else {
                                    Box::new(FilteredRowIdsToPrefilter::new(stream))
                                        .load()
                                        .await
                                }
                            }
                            .await;
                            CloneableResult::from(result.map(Arc::new))
                        }
                        .boxed()
                        .shared()
                    },
                    waiters: 1,
                    is_complete: false,
                    generation,
                };
                let future = entry.future.clone();
                queries.insert(key, entry);
                (future, generation)
            }
        };
        let waiter = SharedPreFilterWaiter {
            materialization,
            key,
            generation,
        };
        let CloneableResult(result) = future.await;
        waiter.mark_complete();
        result.map_err(|error| error.0)
    }
    .boxed()
}

pub(crate) fn build_prefilter(
    context: Arc<datafusion::execution::TaskContext>,
    partition: usize,
    prefilter_source: &PreFilterSource,
    ds: Arc<Dataset>,
    index_meta: &[IndexMetadata],
    masks: PreFilterMasks,
    metrics: &ExecutionPlanMetricsSet,
) -> Result<Arc<DatasetPreFilter>> {
    let mut shared_filter = None;
    let prefilter_loader = match &prefilter_source {
        PreFilterSource::FilteredRowIds(src_node) => {
            if let Some(shared) = src_node.downcast_ref::<SharedPreFilterExec>() {
                shared_filter = Some(shared_prefilter_future(
                    shared.materialization.clone(),
                    shared.source.clone(),
                    false,
                    context,
                    partition,
                ));
                None
            } else {
                let stream = src_node.execute(partition, context)?;
                // Attribute physical materialization to this FTS node. Shared loaders
                // need a separate owner so racing consumers do not receive arbitrary metrics.
                Some(Box::new(
                    FilteredRowIdsToPrefilter::new(stream).with_metrics(metrics, partition),
                ) as Box<dyn FilterLoader>)
            }
        }
        PreFilterSource::ScalarIndexQuery(src_node) => {
            if let Some(shared) = src_node.downcast_ref::<SharedPreFilterExec>() {
                shared_filter = Some(shared_prefilter_future(
                    shared.materialization.clone(),
                    shared.source.clone(),
                    true,
                    context,
                    partition,
                ));
                None
            } else {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(SelectionVectorToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
        }
        PreFilterSource::None => None,
    };
    // Combine the external row-address mask (logical AND) with whatever the
    // filter produced, so an FTS prefilter restricts BM25 scoring to masked rows
    // (mirrors the ANN path). Independent of `overlay_block`, which the prefilter
    // applies separately to drop index entries staled by a data overlay.
    let mut prefilter = if let Some(shared_filter) = shared_filter {
        let shared_filter = match masks.external_mask {
            Some(mask) => async move {
                Ok(Arc::new(
                    mask.as_ref().clone() & shared_filter.await?.as_ref().clone(),
                ))
            }
            .boxed(),
            None => shared_filter,
        };
        DatasetPreFilter::new_with_filter_future(ds, index_meta, Some(shared_filter))
    } else {
        let prefilter_loader = match masks.external_mask {
            Some(mask) => {
                Some(Box::new(MaskAndLoader::new(mask, prefilter_loader)) as Box<dyn FilterLoader>)
            }
            None => prefilter_loader,
        };
        DatasetPreFilter::new(ds, index_meta, prefilter_loader)
    };
    if let Some(overlay_block) = masks.overlay_block {
        prefilter = prefilter.with_overlay_block(overlay_block);
    }
    Ok(Arc::new(prefilter))
}

struct RowIdPrefilterMetrics {
    loads: Count,
    input_rows: Count,
    input_batches: Count,
    row_ids: Count,
    load_time: Time,
    input_time: Time,
    build_time: Time,
}

// Utility to convert an input (containing row ids) into a prefilter.
pub(crate) struct FilteredRowIdsToPrefilter {
    stream: SendableRecordBatchStream,
    metrics: Option<RowIdPrefilterMetrics>,
}

impl FilteredRowIdsToPrefilter {
    pub(crate) fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream,
            metrics: None,
        }
    }

    // Count physical loader executions: batch ANN shares one loader, whereas
    // multi-vector ANN can materialize one per query node.
    pub(crate) fn with_metrics(
        mut self,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        self.metrics = Some(RowIdPrefilterMetrics {
            loads: metrics.new_count("prefilter_loads", partition),
            input_rows: metrics.new_count("prefilter_input_rows", partition),
            input_batches: metrics.new_count("prefilter_input_batches", partition),
            row_ids: metrics.new_count("prefilter_row_ids", partition),
            load_time: metrics.new_time("prefilter_load_time", partition),
            input_time: metrics.new_time("prefilter_input_time", partition),
            build_time: metrics.new_time("prefilter_build_time", partition),
        });
        self
    }
}

#[async_trait]
impl FilterLoader for FilteredRowIdsToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowAddrMask> {
        let metrics = self.metrics.as_ref();
        let _load_timer = metrics.map(|m| m.load_time.timer());
        if let Some(metrics) = metrics {
            metrics.loads.add(1);
        }
        let mut allow_list = RowAddrTreeMap::new();
        loop {
            // Input polling can include I/O, decoding and scheduling. Keep it separate
            // from set insertion, and time batches rather than individual row IDs.
            let batch = {
                let _input_timer = metrics.map(|m| m.input_time.timer());
                self.stream.next().await
            };
            let Some(batch) = batch else { break };
            let batch = batch?;
            let row_ids = batch.column_by_name(ROW_ID).ok_or_else(|| Error::internal("input batch missing row id column even though it is in the schema for the stream"))?;
            let row_ids = row_ids
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| {
                    Error::internal("row id column in prefilter input must be UInt64")
                })?;
            if let Some(metrics) = metrics {
                metrics.input_batches.add(1);
                metrics.input_rows.add(row_ids.len() - row_ids.null_count());
            }
            let _build_timer = metrics.map(|m| m.build_time.timer());
            allow_list.extend(row_ids.iter().flatten());
        }
        let mask = RowAddrMask::from_allowed(allow_list);
        if let Some(metrics) = metrics
            && let Some(row_ids) = mask.max_len()
        {
            metrics.row_ids.add(row_ids as usize);
        }
        Ok(mask)
    }
}

// Utility to convert a serialized selection vector into a prefilter
pub(crate) struct SelectionVectorToPrefilter(pub SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for SelectionVectorToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowAddrMask> {
        let batch = self.0.try_next().await?.ok_or_else(|| {
            Error::internal("Selection vector source for prefilter did not yield any batches")
        })?;
        // The vector-search prefilter wants the set of rows the search is
        // allowed to consider — the `upper` bound of the index expression
        // result. Rows outside the upper bound are guaranteed not to match,
        // so the vector search can skip them.
        //
        // Use deserialize() here (rather than indexing "upper" directly) to
        // support both the TwoMask and the legacy ThreeVariant wire formats
        // that ScalarIndexExec may emit.
        let (result, _) = IndexExprResult::deserialize(&batch)?;
        Ok(result.upper)
    }
}

struct InnerState {
    cached: Option<SendableRecordBatchStream>,
    taken: bool,
}

impl std::fmt::Debug for InnerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerState")
            .field("cached", &self.cached.is_some())
            .field("taken", &self.taken)
            .finish()
    }
}

/// An execution node that can be used as an input twice
///
/// This can be used to broadcast an input to multiple outputs.
///
/// Note: this is done by caching the results.  If one output is consumed
/// more quickly than the other, this can lead to increased memory usage.
/// The `capacity` parameter can bound this, by blocking the faster output
/// when the cache is full.  Take care not to cause deadlock.
///
/// For example, if both outputs are fed to a HashJoinExec then one side
/// of the join will be fully consumed before the other side is read.  In
/// this case, you should probably use an unbounded capacity.
#[derive(Debug)]
pub struct ReplayExec {
    capacity: Capacity,
    input: Arc<dyn ExecutionPlan>,
    inner_state: Arc<Mutex<InnerState>>,
}

impl ReplayExec {
    pub fn new(capacity: Capacity, input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            capacity,
            input,
            inner_state: Arc::new(Mutex::new(InnerState {
                cached: None,
                taken: false,
            })),
        }
    }
}

impl DisplayAs for ReplayExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "Replay: capacity={:?}", self.capacity)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "Replay\ncapacity={:?}", self.capacity)
            }
        }
    }
}

// There's some annoying adapter-work that needs to happen here.  In order
// to share a stream we need its items to be Clone and DataFusionError is
// not Clone.  So we wrap errors in Arc<DataFusionError> (which is Clone).
// In order for that shared stream to be a SendableRecordBatchStream it must
// use DataFusionError, so the adapter unwraps the Arc via DataFusionError::Shared,
// which preserves the typed source chain for both consumers.
pub struct ShareableRecordBatchStream(pub SendableRecordBatchStream);

type SharedBatchResult = std::result::Result<RecordBatch, std::sync::Arc<DataFusionError>>;

impl Stream for ShareableRecordBatchStream {
    type Item = SharedBatchResult;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.0.poll_next_unpin(cx) {
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Some(res)) => {
                std::task::Poll::Ready(Some(res.map_err(std::sync::Arc::new)))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

pub struct ShareableRecordBatchStreamAdapter<S: Stream<Item = SharedBatchResult> + Unpin> {
    schema: SchemaRef,
    stream: S,
}

impl<S: Stream<Item = SharedBatchResult> + Unpin> ShareableRecordBatchStreamAdapter<S> {
    pub fn new(schema: SchemaRef, stream: S) -> Self {
        Self { schema, stream }
    }
}

impl<S: Stream<Item = SharedBatchResult> + Unpin> Stream for ShareableRecordBatchStreamAdapter<S> {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.stream.poll_next_unpin(cx) {
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Some(res)) => {
                std::task::Poll::Ready(Some(res.map_err(DataFusionError::Shared)))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl<S: Stream<Item = SharedBatchResult> + Unpin> RecordBatchStream
    for ShareableRecordBatchStreamAdapter<S>
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[pin_project]
pub struct InstrumentedRecordBatchStreamAdapter<S> {
    schema: SchemaRef,

    #[pin]
    stream: S,
    baseline_metrics: BaselineMetrics,
    batch_count: Count,
}

impl<S> InstrumentedRecordBatchStreamAdapter<S> {
    pub fn new(
        schema: SchemaRef,
        stream: S,
        partition: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        let batch_count = Count::new();
        MetricBuilder::new(metrics)
            .with_partition(partition)
            .build(MetricValue::OutputBatches(batch_count.clone()));
        Self {
            schema,
            stream,
            baseline_metrics: BaselineMetrics::new(metrics, partition),
            batch_count,
        }
    }
}

impl<S> Stream for InstrumentedRecordBatchStreamAdapter<S>
where
    S: Stream<Item = DataFusionResult<RecordBatch>>,
{
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.as_mut().project();
        let timer = this.baseline_metrics.elapsed_compute().timer();
        let poll = this.stream.poll_next(cx);
        timer.done();
        if let Poll::Ready(Some(Ok(_))) = &poll {
            this.batch_count.add(1);
        }
        this.baseline_metrics.record_poll(poll)
    }
}

impl<S> RecordBatchStream for InstrumentedRecordBatchStreamAdapter<S>
where
    S: Stream<Item = DataFusionResult<RecordBatch>>,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Stream wrapper for an `ExecutionPlan` node that pulls from a child input and
/// applies a per-batch async transform.
///
/// `elapsed_compute` measures only the time spent driving the transform
/// futures -- never the time spent polling the child input -- so wrapping a
/// chain of nodes does not double-count child CPU. `output_rows` and
/// `output_batches` are recorded as the transform produces batches.
///
/// `concurrency` caps how many transform futures may be in flight at once.
/// Use `1` for sequential transforms; larger values parallelize per-batch
/// work (e.g., KNN distance computation).
///
/// For leaf nodes (no child input), use [`InstrumentedRecordBatchStreamAdapter`]
/// instead.
pub struct InstrumentedChildInputStream<F, Fut> {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    transform: F,
    concurrency: usize,
    in_flight: FuturesUnordered<Fut>,
    input_done: bool,
    baseline_metrics: BaselineMetrics,
    batch_count: Count,
}

impl<F, Fut> InstrumentedChildInputStream<F, Fut>
where
    F: FnMut(RecordBatch) -> Fut,
    Fut: Future<Output = DataFusionResult<RecordBatch>>,
{
    pub fn new(
        input: SendableRecordBatchStream,
        schema: SchemaRef,
        transform: F,
        concurrency: usize,
        partition: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Self {
        assert!(concurrency >= 1, "concurrency must be >= 1");
        let batch_count = Count::new();
        MetricBuilder::new(metrics)
            .with_partition(partition)
            .build(MetricValue::OutputBatches(batch_count.clone()));
        Self {
            schema,
            input,
            transform,
            concurrency,
            in_flight: FuturesUnordered::new(),
            input_done: false,
            baseline_metrics: BaselineMetrics::new(metrics, partition),
            batch_count,
        }
    }
}

impl<F, Fut> Stream for InstrumentedChildInputStream<F, Fut>
where
    F: FnMut(RecordBatch) -> Fut + Unpin,
    Fut: Future<Output = DataFusionResult<RecordBatch>>,
{
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Fill in-flight transforms up to `concurrency` from the input.
        // Polling the input does not count toward `elapsed_compute`.
        while !this.input_done && this.in_flight.len() < this.concurrency {
            match this.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    this.in_flight.push((this.transform)(batch));
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    this.input_done = true;
                }
                Poll::Pending => break,
            }
        }

        // Drive in-flight transforms; their poll time is counted.
        if !this.in_flight.is_empty() {
            let timer = this.baseline_metrics.elapsed_compute().timer();
            let poll = this.in_flight.poll_next_unpin(cx);
            timer.done();
            match poll {
                Poll::Ready(Some(result)) => {
                    if result.is_ok() {
                        this.batch_count.add(1);
                    }
                    return this.baseline_metrics.record_poll(Poll::Ready(Some(result)));
                }
                // `FuturesUnordered::poll_next` returns `Ready(None)` only
                // when empty, and we just checked `!is_empty` above.
                Poll::Ready(None) => unreachable!("non-empty transform queue yielded None"),
                Poll::Pending => return Poll::Pending,
            }
        }

        if this.input_done {
            return Poll::Ready(None);
        }

        Poll::Pending
    }
}

impl<F, Fut> RecordBatchStream for InstrumentedChildInputStream<F, Fut>
where
    F: FnMut(RecordBatch) -> Fut + Unpin,
    Fut: Future<Output = DataFusionResult<RecordBatch>>,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl ExecutionPlan for ReplayExec {
    fn name(&self) -> &str {
        "ReplayExec"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // We aren't doing any work here, and it would be a little confusing
        // to have multiple replay queues.
        vec![false]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let mut inner_state = self.inner_state.lock().unwrap();
        if let Some(cached) = inner_state.cached.take() {
            if inner_state.taken {
                panic!("ReplayExec can only be executed twice");
            }
            inner_state.taken = true;
            Ok(cached)
        } else {
            let input = self.input.execute(partition, context)?;
            let schema = input.schema();
            let input = ShareableRecordBatchStream(input);
            let (to_return, to_cache) = input.boxed().share(self.capacity);
            inner_state.cached = Some(Box::pin(ShareableRecordBatchStreamAdapter {
                schema: schema.clone(),
                stream: to_cache,
            }));
            Ok(Box::pin(ShareableRecordBatchStreamAdapter {
                schema,
                stream: to_return,
            }))
        }
    }

    fn properties(&self) -> &Arc<datafusion::physical_plan::PlanProperties> {
        self.input.properties()
    }
}

#[derive(Debug, Clone)]
pub struct IoMetrics {
    // We use gauge and not counter here because the underlying ScanScheduler
    // reports cumulative stats, not deltas. We use set_max to ensure the gauge
    // always shows the highest value seen.
    iops: Gauge,
    requests: Gauge,
    bytes_read: Gauge,
}

impl IoMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let iops = metrics.new_gauge(IOPS_METRIC, partition);
        let requests = metrics.new_gauge(REQUESTS_METRIC, partition);
        let bytes_read = metrics.new_gauge(BYTES_READ_METRIC, partition);
        Self {
            iops,
            requests,
            bytes_read,
        }
    }

    pub fn record(&self, scan_scheduler: &ScanScheduler) {
        self.record_stats(scan_scheduler.stats());
    }

    /// Record a snapshot of cumulative I/O statistics.
    ///
    /// Uses `set_max` because the underlying counters are cumulative; the gauge
    /// always reflects the highest (i.e. final) value seen.
    pub fn record_stats(&self, stats: ScanStats) {
        self.iops.set_max(stats.iops as usize);
        self.requests.set_max(stats.requests as usize);
        self.bytes_read.set_max(stats.bytes_read as usize);
    }
}

#[derive(Clone)]
pub struct IndexMetrics {
    indices_loaded: Count,
    parts_loaded: Count,
    index_comparisons: Count,
    index_cache_hits: Count,
    index_cache_misses: Count,
    /// Per-query sink that accumulates exact index-file I/O as partitions are
    /// loaded from storage.  Shared by all clones of this `IndexMetrics`, so
    /// concurrent partition loads all funnel into the same counters.  Published
    /// to `io_metrics` for display via [`IndexMetrics::flush_io`].
    io_stats: IoStats,
    io_metrics: IoMetrics,
}

impl IndexMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            indices_loaded: metrics.new_count(INDICES_LOADED_METRIC, partition),
            parts_loaded: metrics.new_count(PARTS_LOADED_METRIC, partition),
            index_comparisons: metrics.new_count(INDEX_COMPARISONS_METRIC, partition),
            index_cache_hits: metrics.new_count(INDEX_CACHE_HITS_METRIC, partition),
            index_cache_misses: metrics.new_count(INDEX_CACHE_MISSES_METRIC, partition),
            io_stats: IoStats::new(),
            io_metrics: IoMetrics::new(metrics, partition),
        }
    }

    /// Publish the I/O accumulated in the per-query sink to the displayed
    /// `iops`/`requests`/`bytes_read` metrics.  Call once when the operator's
    /// stream finishes; the sink only accumulates on cache misses, so a fully
    /// cache-resident query publishes zeros.
    pub fn flush_io(&self) {
        self.io_metrics.record_stats(self.io_stats.snapshot());
    }
}

impl MetricsCollector for IndexMetrics {
    fn record_parts_loaded(&self, num_shards: usize) {
        self.parts_loaded.add(num_shards);
    }
    fn record_index_loads(&self, num_indexes: usize) {
        self.indices_loaded.add(num_indexes);
    }
    fn record_comparisons(&self, num_comparisons: usize) {
        self.index_comparisons.add(num_comparisons);
    }
    fn record_index_cache_hits(&self, num_hits: usize) {
        self.index_cache_hits.add(num_hits);
    }
    fn record_index_cache_misses(&self, num_misses: usize) {
        self.index_cache_misses.add(num_misses);
    }
    fn io_stats(&self) -> Option<IoStats> {
        Some(self.io_stats.clone())
    }
}

/// Minimum estimated row width, matching DataFusion's default collect thresholds:
/// 1 MiB / 128 Ki rows = 8 bytes per row. DataFusion uses the byte threshold instead
/// of the row threshold whenever a byte estimate is available. This floor keeps
/// narrow schemas from admitting more rows than the default row threshold allows.
///
/// Custom threshold ratios can break that agreement. This is a planning estimate,
/// not a bound on runtime memory, which also includes allocation and join overhead.
const MIN_BYTES_PER_ROW: f64 = 8.0;

/// Whether the columns a node emits include a materialized blob payload.
///
/// A payload has no schema-determined width and is not bounded by the row count,
/// so a node carrying one reports no size at all.
///
/// Decided from what the node emits, matched against the dataset's own fields,
/// because the public output schema strips the blob marker and a projection's blob
/// mode describes what it fetches rather than what it carries. Descriptors remain
/// eligible for estimation, including a seeded width for any URI field.
fn carries_blob_payload(emitted: &arrow_schema::Fields, dataset: &[LanceField]) -> bool {
    emitted.iter().any(|field| {
        let Some(lance) = dataset.iter().find(|f| f.name == *field.name()) else {
            return false;
        };
        carried(field.data_type(), lance)
    })
}

/// Whether `emitted` holds a blob payload described by `lance`, at any nesting.
fn carried(emitted: &DataType, lance: &LanceField) -> bool {
    match emitted {
        DataType::Binary | DataType::LargeBinary => lance.is_blob(),
        DataType::Struct(children) => carries_blob_payload(children, &lance.children),
        // A list's payload is its child's. The dataset field nests the same way, so
        // the child is matched positionally rather than by name.
        DataType::List(child)
        | DataType::LargeList(child)
        | DataType::ListView(child)
        | DataType::LargeListView(child)
        | DataType::FixedSizeList(child, _) => lance
            .children
            .first()
            .is_some_and(|lance_child| carried(child.data_type(), lance_child)),
        DataType::Map(entries, _) => lance
            .children
            .first()
            .is_some_and(|lance_child| carried(entries.data_type(), lance_child)),
        _ => false,
    }
}

/// The average list length [`estimate_bytes_per_row`] assumes when it sizes a
/// list's values. Mirrored here so a list's child buffers are counted as many
/// times as its values are.
const ASSUMED_LIST_LENGTH: f64 = 5.0;

/// Values per row the decoder's estimate would charge, with dictionaries counted
/// as their keys rather than expanded to what they decode to.
///
/// Unlike the decoder's estimate, this charges only dictionary keys. Shared
/// dictionary values are omitted because their cardinality and size are unknown.
fn seeded_value_bytes_per_row(data_type: &DataType) -> f64 {
    match data_type {
        DataType::Dictionary(key, _) => key.byte_width_opt().unwrap_or(1) as f64,
        DataType::List(child) | DataType::LargeList(child) => {
            ASSUMED_LIST_LENGTH * seeded_value_bytes_per_row(child.data_type())
        }
        DataType::FixedSizeList(child, dim) => {
            *dim as f64 * seeded_value_bytes_per_row(child.data_type())
        }
        DataType::Map(entries, _) => {
            ASSUMED_LIST_LENGTH * seeded_value_bytes_per_row(entries.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| seeded_value_bytes_per_row(field.data_type()))
            .sum(),
        other => estimate_bytes_per_row(other),
    }
}

/// Bytes per row of the Arrow buffers that hold no values: validity bitmaps and
/// offsets. [`seeded_value_bytes_per_row`] covers the values, so the two are summed.
///
/// Allocation padding and per-array object size are omitted; the schema describes
/// neither buffer capacities nor batch counts.
fn arrow_overhead_bytes_per_row(field: &arrow_schema::Field) -> f64 {
    let validity = if field.is_nullable() { 1.0 / 8.0 } else { 0.0 };
    // Variable-width types carry `n + 1` offsets; the extra one is a per-batch
    // constant this ignores.
    let buffers = match field.data_type() {
        DataType::Utf8 | DataType::Binary => 4.0,
        DataType::LargeUtf8 | DataType::LargeBinary => 8.0,
        DataType::List(child) => 4.0 + ASSUMED_LIST_LENGTH * arrow_overhead_bytes_per_row(child),
        DataType::Map(entries, _) => {
            4.0 + ASSUMED_LIST_LENGTH * arrow_overhead_bytes_per_row(entries)
        }
        DataType::LargeList(child) => {
            8.0 + ASSUMED_LIST_LENGTH * arrow_overhead_bytes_per_row(child)
        }
        DataType::FixedSizeList(child, dim) => *dim as f64 * arrow_overhead_bytes_per_row(child),
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| arrow_overhead_bytes_per_row(field))
            .sum(),
        // Keys are already charged; the shared dictionary values are omitted.
        DataType::Dictionary(_, _) => 0.0,
        _ => 0.0,
    };
    validity + buffers
}

/// Estimated Arrow bytes per row of one field.
///
/// Uses schema widths for fixed-size values and assumes a validity bitmap for
/// nullable fields. Dictionary values, allocation padding, and per-batch overhead
/// are omitted. Variable-width values use decoder seeds, such as 64 bytes for a
/// string and five items for a list, rather than measured sizes.
fn arrow_bytes_per_row(field: &arrow_schema::Field) -> f64 {
    // A `NullArray` is a row count and nothing else: no values buffer and no
    // validity bitmap whatever the field's nullability says.
    if matches!(field.data_type(), DataType::Null) {
        return 0.0;
    }
    let validity = if field.is_nullable() { 1.0 / 8.0 } else { 0.0 };
    match field.data_type() {
        DataType::Boolean => validity + 1.0 / 8.0,
        DataType::Struct(fields) => {
            validity
                + fields
                    .iter()
                    .map(|field| arrow_bytes_per_row(field))
                    .sum::<f64>()
        }
        // Fixed-size lists store child values without an offset buffer.
        DataType::FixedSizeList(child, dim) => validity + *dim as f64 * arrow_bytes_per_row(child),
        // Count one key per row. The shared values buffer can be substantial,
        // but its size and cardinality are not described by the schema.
        DataType::Dictionary(key, _) => validity + key.byte_width_opt().unwrap_or(1) as f64,
        // Arrow covers fixed-width types missing from `byte_width_opt`.
        other => match other.byte_width_opt().or_else(|| other.primitive_width()) {
            Some(width) => validity + width as f64,
            // `arrow_overhead_bytes_per_row` carries the validity term for these.
            None => seeded_value_bytes_per_row(other) + arrow_overhead_bytes_per_row(field),
        },
    }
}

/// Estimated Arrow bytes per row, floored at [`MIN_BYTES_PER_ROW`].
///
/// Returns `None` for a carried blob payload or a nonpositive estimated width.
/// Uses schema widths where available and decoder seeds otherwise. Callers cache
/// this at construction; dataset metadata identifies blobs in the output schema.
pub(crate) fn estimated_bytes_per_row(
    schema: &arrow_schema::Schema,
    dataset_schema: &LanceSchema,
) -> Option<f64> {
    if carries_blob_payload(schema.fields(), &dataset_schema.fields) {
        return None;
    }
    let bytes_per_row: f64 = schema
        .fields()
        .iter()
        .map(|field| arrow_bytes_per_row(field))
        .sum();
    if bytes_per_row <= 0.0 {
        return None;
    }
    Some(bytes_per_row.max(MIN_BYTES_PER_ROW))
}

/// A row count scaled by a width from [`estimated_bytes_per_row`], always inexact.
///
/// Returns `Absent` when either the row count or width is unavailable.
pub(crate) fn estimated_total_byte_size(
    num_rows: Precision<usize>,
    bytes_per_row: Option<f64>,
) -> Precision<usize> {
    let (Some(rows), Some(bytes_per_row)) = (num_rows.get_value(), bytes_per_row) else {
        return Precision::Absent;
    };
    // A float-to-int cast saturates at `usize::MAX` rather than wrapping, so a
    // huge row count degrades to an enormous estimate instead of a tiny one.
    Precision::Inexact((*rows as f64 * bytes_per_row).ceil() as usize)
}

#[cfg(test)]
mod tests {
    use super::LanceField;
    use super::LanceSchema;
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_META_KEY, BLOB_V2_EXT_NAME};

    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, RecordBatchReader, UInt64Array, types::UInt32Type};
    use arrow_schema::{DataType, Field, Fields, Schema, SortOptions};
    use datafusion::common::NullEquality;
    use datafusion::common::stats::Precision;
    use datafusion::config::ConfigOptions;
    use datafusion::error::{DataFusionError, Result as DataFusionResult};
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::{
        logical_expr::JoinType,
        physical_expr::expressions::Column,
        physical_plan::{
            ExecutionPlan, joins::SortMergeJoinExec, stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{StreamExt, TryStreamExt, stream};
    use lance_core::{ROW_ID, utils::futures::Capacity};
    use lance_datafusion::exec::OneShotExec;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::prefilter::FilterLoader;
    use lance_select::result::IndexExprResultWireFormat;
    use lance_select::{RowAddrMask, RowAddrTreeMap, RowSetOps, result::IndexExprResult};
    use roaring::RoaringBitmap;
    use rstest::rstest;

    use super::{
        FilteredRowIdsToPrefilter, InstrumentedChildInputStream, PreFilterSource, ReplayExec,
        SharedPreFilterExec, SharedPreFilterMaterialization, shared_prefilter_future,
    };

    #[tokio::test]
    async fn test_row_id_prefilter_metrics() {
        let batch = RecordBatch::try_from_iter(vec![(
            "_rowid",
            Arc::new(UInt64Array::from(vec![Some(1), None, Some(2), Some(1)])) as ArrayRef,
        )])
        .unwrap();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::iter(vec![Ok(batch)]),
        ));
        let metrics = ExecutionPlanMetricsSet::new();
        let loader = FilteredRowIdsToPrefilter::new(stream).with_metrics(&metrics, 0);
        let mask = Box::new(loader).load().await.unwrap();
        assert_eq!(mask.max_len(), Some(2));
        assert!(mask.selected(1));
        assert!(!mask.selected(3));
        let collected = metrics.clone_inner();
        for (name, expected) in [
            ("prefilter_loads", 1),
            ("prefilter_input_rows", 3),
            ("prefilter_input_batches", 1),
            ("prefilter_row_ids", 2),
        ] {
            assert_eq!(collected.sum_by_name(name).unwrap().as_usize(), expected);
        }
        for name in [
            "prefilter_load_time",
            "prefilter_input_time",
            "prefilter_build_time",
        ] {
            assert!(collected.sum_by_name(name).unwrap().as_usize() > 0);
        }
    }

    fn prefilter_source(is_scalar_index_query: bool, is_empty: bool) -> PreFilterSource {
        let mask = if is_empty {
            RowAddrMask::allow_nothing()
        } else {
            RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(0_u64..4))
        };
        let batch = if is_scalar_index_query {
            IndexExprResult::exact(mask)
                .serialize(
                    &RoaringBitmap::from_iter([0_u32]),
                    IndexExprResultWireFormat::TwoMask,
                )
                .unwrap()
        } else {
            let row_ids = if is_empty {
                UInt64Array::from(Vec::<u64>::new())
            } else {
                UInt64Array::from_iter_values(0_u64..4)
            };
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    ROW_ID,
                    DataType::UInt64,
                    false,
                )])),
                vec![Arc::new(row_ids)],
            )
            .unwrap()
        };
        // A duplicate source execution fails, so successful concurrent
        // materialization verifies sharing without production metrics.
        let source = Arc::new(OneShotExec::from_batch(batch));
        if is_scalar_index_query {
            PreFilterSource::ScalarIndexQuery(source)
        } else {
            PreFilterSource::FilteredRowIds(source)
        }
    }

    fn shared_materialization(source: &PreFilterSource) -> Arc<SharedPreFilterMaterialization> {
        match source {
            PreFilterSource::FilteredRowIds(source) | PreFilterSource::ScalarIndexQuery(source) => {
                source
                    .downcast_ref::<SharedPreFilterExec>()
                    .expect("expected a shared prefilter source")
                    .materialization
                    .clone()
            }
            _ => panic!("expected a shared prefilter source"),
        }
    }

    fn shared_source(source: &PreFilterSource) -> Arc<dyn ExecutionPlan> {
        match source {
            PreFilterSource::FilteredRowIds(source) | PreFilterSource::ScalarIndexQuery(source) => {
                source
                    .downcast_ref::<SharedPreFilterExec>()
                    .expect("expected a shared prefilter source")
                    .source
                    .clone()
            }
            _ => panic!("expected a shared prefilter source"),
        }
    }

    #[rstest]
    #[case::two_fields(2)]
    #[case::four_fields(4)]
    #[case::eight_fields(8)]
    #[tokio::test]
    async fn shared_multimatch_prefilter_materializes_once(
        #[case] field_count: usize,
        #[values(false, true)] is_scalar_index_query: bool,
        #[values(false, true)] is_empty: bool,
    ) {
        let shared_sources = prefilter_source(is_scalar_index_query, is_empty)
            .shared_for_multimatch_fields(field_count);
        assert_eq!(
            shared_sources
                .iter()
                .filter(|source| source.execution_plan().is_some())
                .count(),
            field_count,
            "every field must declare its shared source dependency"
        );
        let context = Arc::new(datafusion::execution::TaskContext::default());
        let masks = futures::future::try_join_all(shared_sources.iter().map(|source| {
            shared_prefilter_future(
                shared_materialization(source),
                shared_source(source),
                is_scalar_index_query,
                context.clone(),
                0,
            )
        }))
        .await
        .unwrap();

        assert!(masks.windows(2).all(|pair| Arc::ptr_eq(&pair[0], &pair[1])));
        assert_eq!(masks[0].allow_list().unwrap().is_empty(), is_empty);
    }

    #[test]
    fn no_filter_and_single_field_do_not_install_sharing() {
        let no_filter = PreFilterSource::None.shared_for_multimatch_fields(8);
        assert!(
            no_filter
                .iter()
                .all(|source| matches!(source, PreFilterSource::None))
        );

        let single = prefilter_source(false, false).shared_for_multimatch_fields(1);
        assert!(matches!(
            single.as_slice(),
            [PreFilterSource::FilteredRowIds(_)]
        ));
    }

    #[tokio::test]
    async fn shared_multimatch_prefilter_caches_source_error() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            ROW_ID,
            DataType::UInt64,
            false,
        )]));
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Err(DataFusionError::Execution(
                "shared prefilter failure".to_string(),
            ))]),
        ));
        let source = PreFilterSource::FilteredRowIds(Arc::new(OneShotExec::new(stream)));
        let shared_sources = source.shared_for_multimatch_fields(2);
        let context = Arc::new(datafusion::execution::TaskContext::default());
        let left = shared_prefilter_future(
            shared_materialization(&shared_sources[0]),
            shared_source(&shared_sources[0]),
            false,
            context.clone(),
            0,
        );
        let right = shared_prefilter_future(
            shared_materialization(&shared_sources[1]),
            shared_source(&shared_sources[1]),
            false,
            context,
            0,
        );
        let (left, right) = tokio::join!(left, right);

        assert!(
            left.unwrap_err()
                .to_string()
                .contains("shared prefilter failure")
        );
        assert!(
            right
                .unwrap_err()
                .to_string()
                .contains("shared prefilter failure")
        );
    }

    #[tokio::test]
    async fn shared_multimatch_prefilter_survives_waiter_cancellation() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                ROW_ID,
                DataType::UInt64,
                false,
            )])),
            vec![Arc::new(UInt64Array::from_iter_values(0_u64..4))],
        )
        .unwrap();
        let schema = batch.schema();
        let (started, has_started) = tokio::sync::oneshot::channel::<()>();
        let (release, wait) = tokio::sync::oneshot::channel::<()>();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(async move {
                started.send(()).map_err(|_| {
                    DataFusionError::Execution(
                        "shared prefilter startup receiver dropped".to_string(),
                    )
                })?;
                wait.await.map_err(|error| {
                    DataFusionError::Execution(format!(
                        "shared prefilter release sender dropped: {error}"
                    ))
                })?;
                Ok(batch)
            }),
        ));
        let source = PreFilterSource::FilteredRowIds(Arc::new(OneShotExec::new(stream)));
        let shared_sources = source.shared_for_multimatch_fields(2);
        let materialization = shared_materialization(&shared_sources[0]);
        let context = Arc::new(datafusion::execution::TaskContext::default());
        let first = tokio::spawn(shared_prefilter_future(
            materialization.clone(),
            shared_source(&shared_sources[0]),
            false,
            context.clone(),
            0,
        ));
        tokio::time::timeout(std::time::Duration::from_secs(5), has_started)
            .await
            .expect("shared prefilter source should start")
            .expect("shared prefilter startup sender should remain alive");
        let second = tokio::spawn(shared_prefilter_future(
            materialization.clone(),
            shared_source(&shared_sources[1]),
            false,
            context,
            0,
        ));
        loop {
            let waiters = materialization
                .queries
                .lock()
                .unwrap()
                .values()
                .map(|entry| entry.waiters)
                .sum::<usize>();
            if waiters == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        first.abort();
        release.send(()).unwrap();
        let mask = tokio::time::timeout(std::time::Duration::from_secs(5), second)
            .await
            .expect("replacement waiter should resume the shared source")
            .unwrap()
            .unwrap();
        assert_eq!(mask.allow_list().unwrap().len(), Some(4));
    }

    #[tokio::test]
    async fn shared_multimatch_prefilter_drops_fully_canceled_query() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            ROW_ID,
            DataType::UInt64,
            false,
        )]));
        let (started, has_started) = tokio::sync::oneshot::channel::<()>();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(async move {
                started.send(()).map_err(|_| {
                    DataFusionError::Execution(
                        "shared prefilter startup receiver dropped".to_string(),
                    )
                })?;
                std::future::pending::<DataFusionResult<RecordBatch>>().await
            }),
        ));
        let source = PreFilterSource::FilteredRowIds(Arc::new(OneShotExec::new(stream)));
        let shared_sources = source.shared_for_multimatch_fields(2);
        let materialization = shared_materialization(&shared_sources[0]);
        let waiter = tokio::spawn(shared_prefilter_future(
            materialization.clone(),
            shared_source(&shared_sources[0]),
            false,
            Arc::new(datafusion::execution::TaskContext::default()),
            0,
        ));
        tokio::time::timeout(std::time::Duration::from_secs(5), has_started)
            .await
            .expect("shared prefilter source should start")
            .expect("shared prefilter startup sender should remain alive");
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(materialization.queries.lock().unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn instrumented_child_input_stream_excludes_child_poll_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::task::Poll;
        use std::time::Duration;

        use arrow_array::Int32Array;
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::physical_plan::SendableRecordBatchStream;
        use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let n_batches: usize = 3;
        let child_delay = Duration::from_millis(150);

        let counter = Arc::new(AtomicUsize::new(0));
        let s = schema.clone();
        let child = futures::stream::poll_fn(move |_cx| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            if n >= n_batches {
                return Poll::Ready(None);
            }
            std::thread::sleep(child_delay);
            let batch = arrow_array::RecordBatch::try_new(
                s.clone(),
                vec![Arc::new(Int32Array::from(vec![n as i32]))],
            )
            .unwrap();
            Poll::Ready(Some(Ok(batch)))
        });
        let child: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), child));

        let metrics = ExecutionPlanMetricsSet::new();
        let stream = InstrumentedChildInputStream::new(
            child,
            schema,
            move |batch| async move { Ok(batch) },
            1,
            0,
            &metrics,
        );

        let batches: Vec<_> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), n_batches);

        let elapsed_ns = metrics
            .clone_inner()
            .elapsed_compute()
            .expect("elapsed_compute should be recorded");
        let elapsed = Duration::from_nanos(elapsed_ns as u64);

        // The transform is immediate, so `elapsed_compute` should stay well
        // below even one child poll delay. A version that double-counts child
        // input time would include roughly `child_delay * n_batches`.
        let upper = child_delay;
        assert!(
            elapsed < upper,
            "elapsed_compute={:?} >= {:?}; child input time was double-counted",
            elapsed,
            upper,
        );
    }

    #[tokio::test]
    async fn instrumented_child_input_stream_propagates_child_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::task::Poll;

        use arrow_array::Int32Array;
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::error::DataFusionError;
        use datafusion::physical_plan::SendableRecordBatchStream;
        use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let s = schema.clone();
        let step = Arc::new(AtomicUsize::new(0));
        // Yield one OK batch, then an Err, then None.
        let child = futures::stream::poll_fn(move |_cx| {
            let n = step.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => {
                    let batch = arrow_array::RecordBatch::try_new(
                        s.clone(),
                        vec![Arc::new(Int32Array::from(vec![1]))],
                    )
                    .unwrap();
                    Poll::Ready(Some(Ok(batch)))
                }
                1 => Poll::Ready(Some(Err(DataFusionError::Execution("boom".into())))),
                _ => Poll::Ready(None),
            }
        });
        let child: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), child));

        let metrics = ExecutionPlanMetricsSet::new();
        let stream = InstrumentedChildInputStream::new(
            child,
            schema,
            move |batch| async move { Ok(batch) },
            1,
            0,
            &metrics,
        );

        let mut stream = Box::pin(stream);
        let first = stream.next().await.expect("first item present");
        assert!(first.is_ok(), "expected first batch ok, got {:?}", first);

        let second = stream.next().await.expect("error item present");
        let err = second.expect_err("expected propagated error");
        assert!(err.to_string().contains("boom"), "got {}", err);
    }

    #[tokio::test]
    async fn test_replay() {
        let data = lance_datagen::gen_batch()
            .col("x", array::step::<UInt32Type>())
            .into_reader_rows(RowCount::from(1024), BatchCount::from(16));
        let schema = data.schema();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(data).map_err(datafusion::error::DataFusionError::from),
        ));

        let input = Arc::new(OneShotExec::new(data));
        let shared = Arc::new(ReplayExec::new(Capacity::Bounded(4), input));

        let joined = Arc::new(
            SortMergeJoinExec::try_new(
                shared.clone(),
                shared,
                vec![(Arc::new(Column::new("x", 0)), Arc::new(Column::new("x", 0)))],
                None,
                JoinType::Inner,
                vec![SortOptions::default()],
                NullEquality::NullEqualsNull,
            )
            .unwrap(),
        );

        let mut join_stream = joined
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap();

        while let Some(batch) = join_stream.next().await {
            // We don't test much here but shouldn't really need to.  The join and stream sharing
            // are tested on their own.  We just need to make sure they get hooked up correctly
            assert_eq!(batch.unwrap().num_columns(), 2);
        }
    }

    /// Verify that a typed error survives both consumers of a `ReplayExec`.
    #[tokio::test]
    async fn test_replay_preserves_typed_error() {
        use datafusion::error::DataFusionError;
        use datafusion::physical_plan::SendableRecordBatchStream;

        // A marker type that we will look for in the source chain.
        #[derive(Debug)]
        struct MarkerError;
        impl std::fmt::Display for MarkerError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "marker error")
            }
        }
        impl std::error::Error for MarkerError {}

        let schema = Arc::new(arrow_schema::Schema::empty());

        // Build a stream that immediately yields a typed external DataFusion error.
        let typed_err = DataFusionError::External(Box::new(MarkerError));
        let err_stream: SendableRecordBatchStream = Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::once(async move { Err(typed_err) }),
            ),
        );

        let input = Arc::new(OneShotExec::new(err_stream));
        let shared = Arc::new(ReplayExec::new(Capacity::Bounded(4), input));

        let ctx = Arc::new(datafusion::execution::TaskContext::default());

        // Both consumers must receive an error whose source chain includes MarkerError.
        for partition in 0..2 {
            let mut stream = shared.execute(partition, ctx.clone()).unwrap();
            let err = stream
                .next()
                .await
                .expect("stream should yield an error item")
                .expect_err("expected error");

            let mut found = false;
            let mut src: Option<&dyn std::error::Error> = Some(&err);
            while let Some(e) = src {
                if e.downcast_ref::<MarkerError>().is_some() {
                    found = true;
                    break;
                }
                src = e.source();
            }
            assert!(
                found,
                "partition {partition}: MarkerError not found in source chain: {err}"
            );
        }
    }

    #[rstest]
    // The estimate is inexact whatever the row count's precision.
    #[case::exact_row_count(Precision::Exact(10), Some(72.0), Precision::Inexact(720))]
    #[case::inexact_row_count(Precision::Inexact(10), Some(72.0), Precision::Inexact(720))]
    #[case::no_row_count(Precision::Absent, Some(72.0), Precision::Absent)]
    #[case::no_width(Precision::Exact(1_000_000_000), None, Precision::Absent)]
    fn estimated_byte_size_needs_both_a_row_count_and_a_width(
        #[case] num_rows: Precision<usize>,
        #[case] bytes_per_row: Option<f64>,
        #[case] expected: Precision<usize>,
    ) {
        assert_eq!(
            super::estimated_total_byte_size(num_rows, bytes_per_row),
            expected
        );
    }

    #[test]
    fn row_width_combines_fixed_and_seeded_fields() {
        let blobless = LanceSchema::default();
        let mut fields = vec![
            Arc::new(Field::new("a", DataType::UInt32, false)),
            Arc::new(Field::new("b", DataType::Float64, false)),
        ];
        assert_eq!(
            super::estimated_bytes_per_row(&Schema::new(fields.clone()), &blobless),
            Some(12.0)
        );

        // Empty and null-only schemas have no value or validity buffers.
        assert_eq!(
            super::estimated_bytes_per_row(&Schema::empty(), &blobless),
            None
        );
        assert_eq!(
            super::estimated_bytes_per_row(
                &Schema::new(vec![Field::new("null", DataType::Null, true)]),
                &blobless
            ),
            None
        );

        // A partly measurable schema is seeded rather than withdrawn: 12 bytes of
        // fixed width, plus 64 of string, 4 of offset and a validity bit.
        fields.push(Arc::new(Field::new("note", DataType::Utf8, true)));
        assert_eq!(
            super::estimated_bytes_per_row(&Schema::new(fields), &blobless),
            Some(80.125)
        );
    }

    /// Nested blob payloads must suppress the whole output estimate.
    #[test]
    fn a_blob_nested_in_a_list_is_still_carried() {
        let emitted = Schema::new(vec![Field::new(
            "blobs",
            DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
            true,
        )]);
        let mut dataset = LanceSchema::try_from(&emitted).unwrap();
        dataset.fields[0].children[0]
            .metadata
            .insert(BLOB_META_KEY.to_string(), "true".to_string());

        assert_eq!(super::estimated_bytes_per_row(&emitted, &dataset), None);
    }

    /// The guard keys on what a node emits, matched against the dataset's fields.
    /// A v1 blob carries `BLOB_META_KEY` and no v2 extension name, so a v2-only
    /// check misses it and bills an unbounded payload the seed width.
    #[rstest]
    #[case::v2_payload(ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME, DataType::LargeBinary, true)]
    #[case::v1_payload(BLOB_META_KEY, "true", DataType::LargeBinary, true)]
    // A descriptor is a fixed-width column, not a payload, so it stays measurable.
    #[case::v1_descriptor(
        BLOB_META_KEY,
        "true",
        DataType::Struct(Fields::from(vec![
            Field::new("position", DataType::UInt64, false),
            Field::new("size", DataType::UInt64, false),
        ])),
        false
    )]
    // An ordinary binary column that is not a blob is measurable.
    #[case::plain_binary("unrelated-key", "true", DataType::LargeBinary, false)]
    fn a_carried_blob_payload_is_recognised_whatever_marked_it(
        #[case] key: &str,
        #[case] value: &str,
        #[case] emitted: DataType,
        #[case] expected: bool,
    ) {
        let mut lance =
            LanceField::try_from(&Field::new("blob", DataType::LargeBinary, true)).unwrap();
        lance.metadata.insert(key.to_string(), value.to_string());
        let emitted = Schema::new(vec![Field::new("blob", emitted, true)]);

        assert_eq!(
            super::carries_blob_payload(emitted.fields(), std::slice::from_ref(&lance)),
            expected
        );
    }

    /// Covers schema widths, nullable overhead, and variable-width seeds.
    #[rstest]
    #[case::fixed_width(Field::new("a", DataType::Int64, false), 8.0)]
    // A nullable field pays one validity bit a row.
    #[case::fixed_width_nullable(Field::new("a", DataType::Int64, true), 8.125)]
    // Boolean values are a bit a row as well, so validity doubles the column.
    #[case::boolean_nullable(Field::new("a", DataType::Boolean, true), 0.25)]
    #[case::fixed_size_binary(Field::new("a", DataType::FixedSizeBinary(12), false), 12.0)]
    // Children pay their own validity once per value, not once per row: a 4-dim
    // vector of nullable floats carries four bits a row.
    #[case::fixed_size_list(
        Field::new(
            "a",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            false,
        ),
        16.5
    )]
    #[case::nested_struct(
        Field::new(
            "a",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int64, true),
                Field::new("y", DataType::Float32, false),
            ])),
            false,
        ),
        12.125
    )]
    #[case::null(Field::new("a", DataType::Null, true), 0.0)]
    // Below: everything whose per-row cost the schema is silent about, seeded from
    // the decoder's estimate plus the buffers around it.
    // 64 bytes of value and 4 of offset.
    #[case::utf8(Field::new("a", DataType::Utf8, false), 68.0)]
    // 64 and an 8-byte offset.
    #[case::large_binary(Field::new("a", DataType::LargeBinary, false), 72.0)]
    // The decoder's catch-all, with no offset buffer of its own.
    #[case::utf8_view(Field::new("a", DataType::Utf8View, false), 64.0)]
    #[case::list(
        Field::new(
            "a",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        // Five 8-byte items, a 4-byte offset, and a validity bit for the list and
        // each assumed item.
        44.75
    )]
    // One Int8 key a row; shared dictionary values are omitted.
    #[case::dictionary(
        Field::new(
            "a",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        1.0
    )]
    // Nested dictionaries use keys too: five keys plus a four-byte list offset.
    #[case::nested_dictionary(
        Field::new(
            "a",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ))),
            false,
        ),
        9.0
    )]
    // A fixed-width type `byte_width_opt` does not enumerate still reports exactly,
    // via Arrow's own width rather than the 64-byte seed.
    #[case::decimal64(Field::new("a", DataType::Decimal64(18, 2), false), 8.0)]
    // A struct sums its children, seeded child included.
    #[case::struct_with_a_string(
        Field::new(
            "a",
            DataType::Struct(Fields::from(vec![
                Field::new("x", DataType::Int64, false),
                Field::new("y", DataType::Utf8, false),
            ])),
            false,
        ),
        76.0
    )]
    fn width_includes_schema_sizes_and_seeds(#[case] field: Field, #[case] expected: f64) {
        assert_eq!(super::arrow_bytes_per_row(&field), expected);
    }

    /// The floor reaches the default byte threshold at the default row threshold.
    #[test]
    fn a_narrow_row_is_floored_to_the_row_guard() {
        // Read DataFusion's guards rather than copy them. The floor is derived from
        // both, so an upgrade that moves either default has to fail here instead of
        // leaving behind a floor that no longer reproduces the row cap.
        let optimizer = ConfigOptions::default().optimizer;
        let collect_bytes = optimizer.hash_join_single_partition_threshold;
        let collect_rows = optimizer.hash_join_single_partition_threshold_rows;
        assert_eq!(
            super::MIN_BYTES_PER_ROW,
            collect_bytes as f64 / collect_rows as f64,
            "the floor is the byte guard spread across the row guard"
        );

        // A quarter byte a row: a bit of value and a bit of validity. Unfloored,
        // four million rows of this would still pass for less than 1 MiB.
        let narrow = Schema::new(vec![Field::new("flag", DataType::Boolean, true)]);

        let width = super::estimated_bytes_per_row(&narrow, &LanceSchema::default());
        assert_eq!(width, Some(super::MIN_BYTES_PER_ROW), "the floor applies");

        let under = super::estimated_total_byte_size(Precision::Exact(collect_rows - 1), width);
        assert!(
            under
                .get_value()
                .is_some_and(|bytes| *bytes < collect_bytes),
            "a row short of the row guard has to stay under the byte guard: {under:?}"
        );

        let over = super::estimated_total_byte_size(Precision::Exact(collect_rows), width);
        assert!(
            over.get_value()
                .is_some_and(|bytes| *bytes >= collect_bytes),
            "the row count the row guard rejects has to fail the byte guard too: {over:?}"
        );
    }
}
