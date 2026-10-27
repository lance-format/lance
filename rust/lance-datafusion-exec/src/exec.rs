// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion execution plans

use std::{
    collections::HashMap,
    num::NonZero,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use datafusion_common::{DataFusionError, Statistics, utils::get_available_parallelism};
use datafusion_execution::{
    TaskContext,
    config::SessionConfig,
    disk_manager::DiskManagerBuilder,
    memory_pool::{FairSpillPool, TrackConsumersPool},
    runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
    coalesce_partitions::CoalescePartitionsExec,
    display::DisplayableExecutionPlan,
    execution_plan::{Boundedness, CardinalityEffect, EmissionType},
    sorts::sort_preserving_merge::SortPreservingMergeExec,
    stream::RecordBatchStreamAdapter,
};

use futures::{StreamExt, stream};
use lance_arrow::SchemaExt;
use lance_core::{
    Error, Result,
    utils::{
        futures::FinallyStreamExt,
        tracing::{EXECUTION_PLAN_RUN, TRACE_EXECUTION},
    },
};
use log::{debug, info, warn};

use crate::{
    chunker::StrictBatchSizeStream,
    utils::{
        BYTES_READ_METRIC, INDEX_CACHE_HITS_METRIC, INDEX_CACHE_MISSES_METRIC,
        INDEX_COMPARISONS_METRIC, INDICES_LOADED_METRIC, IOPS_METRIC, MetricsExt,
        PARTS_LOADED_METRIC, REQUESTS_METRIC,
    },
};

/// An source execution node created from an existing stream
///
/// It can only be used once, and will return the stream.  After that the node
/// is exhausted.
///
/// Note: the stream should be finite, otherwise we will report datafusion properties
/// incorrectly.
pub struct OneShotExec {
    stream: Mutex<Option<SendableRecordBatchStream>>,
    // We save off a copy of the schema to speed up formatting and so ExecutionPlan::schema & display_as
    // can still function after exhausted
    schema: Arc<ArrowSchema>,
    properties: Arc<PlanProperties>,
}

impl OneShotExec {
    /// Create a new instance from a given stream
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        let schema = stream.schema();
        Self {
            stream: Mutex::new(Some(stream)),
            schema: schema.clone(),
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
        }
    }

    pub fn from_batch(batch: RecordBatch) -> Self {
        let schema = batch.schema();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        ));
        Self::new(stream)
    }
}

impl std::fmt::Debug for OneShotExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = self.stream.lock().unwrap();
        f.debug_struct("OneShotExec")
            .field("exhausted", &stream.is_none())
            .field("schema", self.schema.as_ref())
            .finish()
    }
}

impl DisplayAs for OneShotExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let stream = self.stream.lock().unwrap();
        let exhausted = if stream.is_some() { "" } else { "EXHAUSTED" };
        let columns = self
            .schema
            .field_names()
            .iter()
            .cloned()
            .cloned()
            .collect::<Vec<_>>();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "OneShotStream: {}columns=[{}]",
                    exhausted,
                    columns.join(",")
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "OneShotStream\nexhausted={}\ncolumns=[{}]",
                    exhausted,
                    columns.join(",")
                )
            }
        }
    }
}

impl ExecutionPlan for OneShotExec {
    fn name(&self) -> &str {
        "OneShotExec"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        // OneShotExec has no children, so this should only be called with an empty vector
        if !children.is_empty() {
            return Err(datafusion_common::DataFusionError::Internal(
                "OneShotExec does not support children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let stream = self
            .stream
            .lock()
            .map_err(|err| DataFusionError::Execution(err.to_string()))?
            .take();
        if let Some(stream) = stream {
            Ok(stream)
        } else {
            Err(DataFusionError::Execution(
                "OneShotExec has already been executed".to_string(),
            ))
        }
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

/// Callback for reporting statistics after a scan
pub type ExecutionStatsCallback = Arc<dyn Fn(&ExecutionSummaryCounts) + Send + Sync>;

#[derive(Default, Clone)]
pub struct LanceExecutionOptions {
    pub use_spilling: bool,
    pub mem_pool_size: Option<u64>,
    pub max_temp_directory_size: Option<u64>,
    pub batch_size: Option<usize>,
    pub target_partition: Option<usize>,
    pub execution_stats_callback: Option<ExecutionStatsCallback>,
    pub skip_logging: bool,
}

impl std::fmt::Debug for LanceExecutionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceExecutionOptions")
            .field("use_spilling", &self.use_spilling)
            .field("mem_pool_size", &self.mem_pool_size)
            .field("max_temp_directory_size", &self.max_temp_directory_size)
            .field("batch_size", &self.batch_size)
            .field("target_partition", &self.target_partition)
            .field("skip_logging", &self.skip_logging)
            .field(
                "execution_stats_callback",
                &self.execution_stats_callback.is_some(),
            )
            .finish()
    }
}

const DEFAULT_LANCE_MEM_POOL_SIZE_PER_PARTITION: u64 = 150 * 1024 * 1024;
/// Maximum headroom reserved for one external sort's non-spillable merge phase.
const MAX_SORT_SPILL_RESERVATION_BYTES: u64 = 40 * 1024 * 1024;
const DEFAULT_LANCE_MAX_TEMP_DIRECTORY_SIZE: u64 = 100 * 1024 * 1024 * 1024; // 100GB

impl LanceExecutionOptions {
    /// The number of partitions the DataFusion session will actually run with.
    ///
    /// When `target_partition` is not set we do not override the session config,
    /// so the session falls back to DataFusion's default `target_partitions`,
    /// which is the available parallelism (number of CPU cores). The memory pool
    /// must be sized for that effective partition count, not for a single
    /// partition, or sort-heavy plans exhaust the pool.
    fn effective_target_partition(&self) -> u64 {
        self.target_partition
            .unwrap_or_else(get_available_parallelism) as u64
    }

    pub fn mem_pool_size(&self) -> u64 {
        let num_partitions = self.effective_target_partition();
        self.mem_pool_size.unwrap_or_else(|| {
            std::env::var("LANCE_MEM_POOL_SIZE")
                .map(|s| match s.parse::<u64>() {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("Failed to parse LANCE_MEM_POOL_SIZE: {}, using default", e);
                        DEFAULT_LANCE_MEM_POOL_SIZE_PER_PARTITION * num_partitions
                    }
                })
                .unwrap_or(DEFAULT_LANCE_MEM_POOL_SIZE_PER_PARTITION * num_partitions)
        })
    }

    pub fn max_temp_directory_size(&self) -> u64 {
        self.max_temp_directory_size.unwrap_or_else(|| {
            std::env::var("LANCE_MAX_TEMP_DIRECTORY_SIZE")
                .map(|s| match s.parse::<u64>() {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(
                            "Failed to parse LANCE_MAX_TEMP_DIRECTORY_SIZE: {}, using default",
                            e
                        );
                        DEFAULT_LANCE_MAX_TEMP_DIRECTORY_SIZE
                    }
                })
                .unwrap_or(DEFAULT_LANCE_MAX_TEMP_DIRECTORY_SIZE)
        })
    }

    pub fn use_spilling(&self) -> bool {
        if !self.use_spilling {
            return false;
        }
        // Presence enables the bypass; the value is not parsed as a boolean.
        std::env::var("LANCE_BYPASS_SPILLING")
            .map(|_| {
                info!("Bypassing spilling because LANCE_BYPASS_SPILLING is set");
                false
            })
            .unwrap_or(true)
    }

    /// The batch size override for one execution, if any.
    ///
    /// Task contexts write this straight into the session config, which skips
    /// the `batch_size > 0` check that `SessionConfig::with_batch_size`
    /// asserts, so a zero is rejected here instead.
    pub fn validated_batch_size(&self) -> Result<Option<usize>> {
        match self.batch_size {
            Some(0) => Err(Error::invalid_input(
                "batch_size must be greater than 0, got 0",
            )),
            batch_size => Ok(batch_size),
        }
    }
}

/// Builds the session config for executing plans with `options`.
///
/// This ignores [`LanceExecutionOptions::batch_size`], which applies to one
/// execution rather than to a session.
pub fn new_session_config(options: &LanceExecutionOptions) -> SessionConfig {
    let mut session_config = SessionConfig::new();
    if let Some(target_partition) = options.target_partition {
        session_config = session_config.with_target_partitions(target_partition);
    }
    if options.use_spilling() {
        // Reserve sort/merge headroom for each spillable sort, using up to 40 MiB
        // instead of DataFusion's 10 MiB default. Limit it to one third of the pool
        // to leave room for input batches in small pools. This reservation comes
        // out of the same pool; it does not guarantee that every batch will fit.
        let sort_spill_reservation_bytes =
            (options.mem_pool_size() / 3).min(MAX_SORT_SPILL_RESERVATION_BYTES) as usize;
        session_config =
            session_config.with_sort_spill_reservation_bytes(sort_spill_reservation_bytes);
    }
    session_config
}

/// Builds the runtime environment, including the memory pool and the spill
/// directory, for executing plans with `options`.
pub fn new_runtime_env(options: &LanceExecutionOptions) -> Arc<RuntimeEnv> {
    let mut runtime_env_builder = RuntimeEnvBuilder::new();
    if options.use_spilling() {
        let disk_manager_builder = DiskManagerBuilder::default()
            .with_max_temp_directory_size(options.max_temp_directory_size());
        runtime_env_builder = runtime_env_builder
            .with_disk_manager_builder(disk_manager_builder)
            .with_memory_pool(Arc::new(TrackConsumersPool::new(
                FairSpillPool::new(options.mem_pool_size() as usize),
                NonZero::try_from(16).unwrap(),
            )));
    }
    // Without spilling, DataFusion's default UnboundedMemoryPool accepts all
    // reservations. This bypasses the configured pool limit, not actual RAM limits.
    runtime_env_builder.build_arc().unwrap()
}

/// Returns a new task context to execute one plan with `options`.
///
/// The context has no registered functions, so it can only execute plans
/// whose expressions are already resolved. Each call gets its own runtime
/// environment, so concurrent spilling executions do not share a memory pool.
pub fn new_task_context(options: &LanceExecutionOptions) -> Result<Arc<TaskContext>> {
    let mut session_config = new_session_config(options);
    if let Some(batch_size) = options.validated_batch_size()? {
        session_config.options_mut().execution.batch_size = batch_size;
    }
    Ok(Arc::new(
        TaskContext::default()
            .with_session_config(session_config)
            .with_runtime(new_runtime_env(options)),
    ))
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct ExecutionSummaryCounts {
    /// The number of I/O operations performed
    pub iops: usize,
    /// The number of requests made to the storage layer (may be larger or smaller than iops
    /// depending on coalescing configuration)
    pub requests: usize,
    /// The number of bytes read during the execution of the plan
    pub bytes_read: usize,
    /// The number of top-level indices loaded
    pub indices_loaded: usize,
    /// The number of index partitions loaded
    pub parts_loaded: usize,
    /// The number of index comparisons performed (the exact meaning depends on the index type)
    pub index_comparisons: usize,
    /// Additional metrics for more detailed statistics.  These are subject to change in the future
    /// and should only be used for debugging purposes.
    ///
    /// Newer metrics (e.g. [`INDEX_CACHE_HITS_METRIC`], [`INDEX_CACHE_MISSES_METRIC`]) are added
    /// here rather than as `pub` fields, so this struct stays backwards compatible for callers
    /// that construct or destructure it. Prefer the typed accessors below.
    pub all_counts: HashMap<String, usize>,
    /// Additional time metrics for more detailed statistics, stored in nanoseconds.
    /// Operator baseline times use `<ExecutionPlan::name()>_elapsed_compute` keys.
    /// Timings may be nested and accumulate across concurrent work; they are not
    /// exclusive stages that can be added to recover query wall time.
    /// These are subject to change in the future and should only be used for debugging purposes.
    pub all_times: HashMap<String, usize>,
}

impl ExecutionSummaryCounts {
    /// Number of index cache page lookups where the loader was not executed
    /// (per-page granularity).
    ///
    /// A "hit" is any page-level lookup at an instrumented cache boundary that
    /// did not run the loader on this call. That covers both a true cache hit
    /// on an already-populated entry and a coalesced concurrent load where an
    /// in-flight loader started by a different caller produced the value.
    ///
    /// Instrumented boundaries in this release:
    /// BTree page, IVF partition (v2, `write_cache=true` scan path), inverted
    /// posting list (grouped and per-token), inverted per-token metadata
    /// (`PostingMetadataKey`), inverted phrase positions (`PositionKey`),
    /// bitmap posting (Equals / Range / IsIn), ngram posting, and rtree page
    /// / null slot.
    ///
    /// Caveats:
    /// * IVF v2 streaming scans and legacy v1 IVF partitions run
    ///   `load_partition` with `write_cache=false`. Those loads always execute
    ///   the loader and never write the result back, so they are reported as a
    ///   miss on every call. See [`Self::index_cache_hit_ratio`].
    /// * A cold posting-list lookup on the grouped inverted layout can record
    ///   up to two misses (posting-list group + per-token metadata) for a
    ///   single term.
    ///
    /// Other index cache boundaries such as HNSW graph pages and quantizer
    /// codebooks are not yet instrumented; a scan that only touches those
    /// paths returns `0` here.
    pub fn index_cache_hits(&self) -> usize {
        self.all_counts
            .get(INDEX_CACHE_HITS_METRIC)
            .copied()
            .unwrap_or(0)
    }

    /// Number of index cache page lookups that had to execute the loader
    /// (per-page granularity).
    ///
    /// A "miss" is any page-level lookup at an instrumented cache boundary
    /// where the loader ran, i.e. the page was not resident and had to be
    /// materialised (typically from storage). See
    /// [`Self::index_cache_hits`] for the paired counter and the list of
    /// instrumented boundaries.
    pub fn index_cache_misses(&self) -> usize {
        self.all_counts
            .get(INDEX_CACHE_MISSES_METRIC)
            .copied()
            .unwrap_or(0)
    }

    /// Ratio of index cache hits to total lookups. Returns `0.0` when no lookups
    /// were recorded in this scan.
    ///
    /// This ratio only reflects paths that write their result back to the
    /// index cache. Streaming scans (IVF v2 `write_cache=false` and legacy v1
    /// IVF `load_partition_stream`) intentionally bypass the cache and are
    /// counted as misses on every call, so a workload dominated by streaming
    /// vector scans will report a hit ratio near `0.0` regardless of cache
    /// size.
    pub fn index_cache_hit_ratio(&self) -> f32 {
        // Widen to u128 before summing so a pathological (hits + misses)
        // overflow can't panic in debug builds nor wrap in release builds.
        let hits = self.index_cache_hits() as u128;
        let total = hits + self.index_cache_misses() as u128;
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }
}

pub fn collect_execution_metrics(node: &dyn ExecutionPlan, counts: &mut ExecutionSummaryCounts) {
    if let Some(metrics) = node.metrics() {
        for (metric_name, count) in metrics.iter_counts() {
            match metric_name.as_ref() {
                IOPS_METRIC => counts.iops += count.value(),
                REQUESTS_METRIC => counts.requests += count.value(),
                BYTES_READ_METRIC => counts.bytes_read += count.value(),
                INDICES_LOADED_METRIC => counts.indices_loaded += count.value(),
                PARTS_LOADED_METRIC => counts.parts_loaded += count.value(),
                INDEX_COMPARISONS_METRIC => counts.index_comparisons += count.value(),
                _ => {
                    let existing = counts
                        .all_counts
                        .entry(metric_name.as_ref().to_string())
                        .or_insert(0);
                    *existing += count.value();
                }
            }
        }
        for (metric_name, time) in metrics.iter_times() {
            let existing = counts
                .all_times
                .entry(metric_name.as_ref().to_string())
                .or_insert(0);
            *existing += time.value();
        }
        // Keep operator baselines separate: ANN elapsed time includes asynchronous waits,
        // while operators such as SortExec report compute time. Summing them hides the stages.
        if let Some(elapsed) = metrics.elapsed_compute() {
            *counts
                .all_times
                .entry(format!("{}_elapsed_compute", node.name()))
                .or_default() += elapsed;
        }
        // Include gauge-based I/O metrics (some nodes record I/O as gauges)
        for (metric_name, gauge) in metrics.iter_gauges() {
            match metric_name.as_ref() {
                IOPS_METRIC => counts.iops += gauge.value(),
                REQUESTS_METRIC => counts.requests += gauge.value(),
                BYTES_READ_METRIC => counts.bytes_read += gauge.value(),
                _ => {}
            }
        }
    }
    for child in node.children() {
        collect_execution_metrics(child.as_ref(), counts);
    }
}

fn report_plan_summary_metrics(plan: &dyn ExecutionPlan, options: &LanceExecutionOptions) {
    let output_rows = plan
        .metrics()
        .map(|m| m.output_rows().unwrap_or(0))
        .unwrap_or(0);
    let mut counts = ExecutionSummaryCounts::default();
    collect_execution_metrics(plan, &mut counts);
    if !options.skip_logging {
        tracing::info!(
            target: TRACE_EXECUTION,
            r#type = EXECUTION_PLAN_RUN,
            plan_summary = display_plan_one_liner(plan),
            output_rows,
            iops = counts.iops,
            requests = counts.requests,
            bytes_read = counts.bytes_read,
            indices_loaded = counts.indices_loaded,
            parts_loaded = counts.parts_loaded,
            index_comparisons = counts.index_comparisons,
            index_cache_hits = counts.index_cache_hits(),
            index_cache_misses = counts.index_cache_misses(),
        );
    }
    if let Some(callback) = options.execution_stats_callback.as_ref() {
        callback(&counts);
    }
}

/// Create a one-line rough summary of the given execution plan.
///
/// The summary just shows the name of the operators in the plan. It omits any
/// details such as parameters or schema information.
///
/// Example: `Projection(Take(CoalesceBatches(Filter(LanceScan))))`
fn display_plan_one_liner(plan: &dyn ExecutionPlan) -> String {
    let mut output = String::new();

    display_plan_one_liner_impl(plan, &mut output);

    output
}

fn display_plan_one_liner_impl(plan: &dyn ExecutionPlan, output: &mut String) {
    // Remove the "Exec" suffix from the plan name if present for brevity
    let name = plan.name().trim_end_matches("Exec");
    output.push_str(name);

    let children = plan.children();
    if !children.is_empty() {
        output.push('(');
        for (i, child) in children.iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            display_plan_one_liner_impl(child.as_ref(), output);
        }
        output.push(')');
    }
}

/// Executes a plan under a task context built by [`new_task_context`].
///
/// Plans that need registered functions at execution time should use
/// `lance_datafusion::exec::execute_plan`, which runs under a session context.
pub fn execute_plan(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    let task_context = new_task_context(&options)?;
    execute_plan_with_context(plan, options, task_context)
}

/// Executes a plan under `task_context`, reporting its metrics through `options`
/// once the stream finishes.
///
/// A plan with more than one partition is merged into one.
pub fn execute_plan_with_context(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
    task_context: Arc<TaskContext>,
) -> Result<SendableRecordBatchStream> {
    if !options.skip_logging {
        debug!(
            "Executing plan:\n{}",
            DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
        );
    }

    // Coalesce to a single partition if the optimizer left more than one.
    // EnforceDistribution may remove RepartitionExec(1) nodes when the parent
    // declares UnspecifiedDistribution, leaving multi-partition plans here.
    //
    // If the plan carries an output ordering (e.g. a top-k `SortExec` whose
    // result was later repartitioned to parallelize downstream operators),
    // a plain `CoalescePartitionsExec` would scramble that order because it
    // merges partitions in scheduling-dependent order. Use an order-preserving
    // merge in that case instead, mirroring what `EnforceDistribution` itself
    // does when it needs to merge an ordered, multi-partition plan.
    let plan: Arc<dyn ExecutionPlan> = if plan.properties().partitioning.partition_count() == 1 {
        plan
    } else if let Some(ordering) = plan.output_ordering() {
        Arc::new(SortPreservingMergeExec::new(ordering.clone(), plan))
    } else {
        Arc::new(CoalescePartitionsExec::new(plan))
    };

    let stream = plan.execute(0, task_context)?;

    let schema = stream.schema();
    let stream = stream.finally(move || {
        if !options.skip_logging || options.execution_stats_callback.is_some() {
            report_plan_summary_metrics(plan.as_ref(), &options);
        }
    });
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
}

#[derive(Clone, Debug)]
pub struct StrictBatchSizeExec {
    input: Arc<dyn ExecutionPlan>,
    batch_size: usize,
}

impl StrictBatchSizeExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, batch_size: usize) -> Self {
        Self { input, batch_size }
    }
}

impl DisplayAs for StrictBatchSizeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "StrictBatchSizeExec")
    }
}

impl ExecutionPlan for StrictBatchSizeExec {
    fn name(&self) -> &str {
        "StrictBatchSizeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: children[0].clone(),
            batch_size: self.batch_size,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context)?;
        let schema = stream.schema();
        let stream = StrictBatchSizeStream::new(stream, self.batch_size);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion_common::Result<std::sync::Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

/// Exec node that rechunks batches so no output batch exceeds `max_bytes`.
///
/// # Why this exists
///
/// DataFusion's sort operator must reserve memory for an entire input batch,
/// including estimated sort/merge overhead. It can spill buffered batches and
/// retry, but still fails if the new batch's reservation cannot fit. This is
/// separate from the historical allocation issues fixed by DataFusion PR #14644
/// (<https://github.com/apache/datafusion/pull/14644>).
///
/// This node bounds input batch sizes before sorting, at the cost of potentially
/// expensive deep copies. It does not guarantee success for every memory pool:
/// spill/merge reservations and other consumers also need room in the pool.
///
/// # Deep copy
///
/// After slicing a RecordBatch, `get_array_memory_size` still reports the
/// size of the *original* backing buffers, not the slice.  To get accurate
/// sizes the slices must be deep-copied.  This is a last resort and can be
/// expensive for large batches, but the deep copy is only performed when a
/// batch actually needs to be sliced — batches that are already within the
/// target range pass through at zero cost.
///
/// If a single row exceeds `max_bytes`, execution fails with an error.
#[derive(Clone, Debug)]
pub struct HardCapBatchSizeExec {
    input: Arc<dyn ExecutionPlan>,
    max_bytes: usize,
}

impl HardCapBatchSizeExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, max_bytes: usize) -> Self {
        Self { input, max_bytes }
    }
}

impl DisplayAs for HardCapBatchSizeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "HardCapBatchSizeExec(max_bytes={})", self.max_bytes)
    }
}

impl ExecutionPlan for HardCapBatchSizeExec {
    fn name(&self) -> &str {
        "HardCapBatchSizeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: children[0].clone(),
            max_bytes: self.max_bytes,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context)?;
        let schema = stream.schema();
        let max_bytes = self.max_bytes;
        let rechunked = lance_arrow::stream::rechunk_stream_by_size_deep_copy(
            stream,
            schema.clone(),
            0,
            max_bytes,
        );
        // Check that no single-row batch exceeds the limit.
        let validated = rechunked.map(move |result| {
            let batch = result?;
            if batch.num_rows() == 1 && batch.get_array_memory_size() > max_bytes {
                return Err(DataFusionError::External(Box::new(Error::invalid_input(
                    format!(
                        "a single row is {} bytes which exceeds the maximum allowed batch \
                         size of {} bytes",
                        batch.get_array_memory_size(),
                        max_bytes,
                    ),
                ))));
            }
            Ok(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, validated)))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion_common::Result<std::sync::Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, cast::AsArray, record_batch, types::Int32Type};
    use arrow_select::concat::concat_batches;
    use datafusion_execution::memory_pool::{MemoryConsumer, MemoryLimit};
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
    use datafusion_physical_plan::sorts::sort::SortExec;
    use futures::TryStreamExt;
    use rstest::rstest;

    #[test]
    fn test_new_task_context() {
        let options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(64 * 1024 * 1024),
            target_partition: Some(3),
            batch_size: Some(17),
            ..Default::default()
        };
        let task_ctx = new_task_context(&options).unwrap();
        assert_eq!(task_ctx.session_config().batch_size(), 17);
        assert_eq!(task_ctx.session_config().target_partitions(), 3);
        assert!(matches!(
            task_ctx.memory_pool().memory_limit(),
            MemoryLimit::Finite(limit) if limit == 64 * 1024 * 1024
        ));
        // The batch size applies to one execution, not to the session config.
        assert_ne!(new_session_config(&options).batch_size(), 17);

        let zero_batch_size = LanceExecutionOptions {
            batch_size: Some(0),
            ..Default::default()
        };
        let err = new_task_context(&zero_batch_size).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains("batch_size must be greater than 0")
        );

        // Each reservation fits its own pool, but any two exceed a shared pool.
        let reservations: Vec<_> = (0..3)
            .map(|_| {
                let task_ctx = new_task_context(&options).unwrap();
                let reservation =
                    MemoryConsumer::new("ExternalSorterMerge[0]").register(task_ctx.memory_pool());
                reservation.try_grow(40 * 1024 * 1024).unwrap();
                reservation
            })
            .collect();
        assert_eq!(reservations.len(), 3);
    }

    #[rstest]
    #[case::session_batch_size(None, &[6])]
    #[case::batch_size_override(Some(4), &[4, 2])]
    #[tokio::test]
    async fn test_execute_plan_batch_size(
        #[case] batch_size: Option<usize>,
        #[case] expected_batch_rows: &[usize],
    ) {
        let batch = record_batch!(("x", Int32, [5, 3, 1, 4, 2, 0])).unwrap();
        let sort_expr = PhysicalSortExpr::new_default(col("x", &batch.schema()).unwrap());
        let plan = Arc::new(SortExec::new(
            LexOrdering::new([sort_expr]).unwrap(),
            Arc::new(OneShotExec::from_batch(batch.clone())),
        ));
        let options = LanceExecutionOptions {
            batch_size,
            ..Default::default()
        };
        let batches: Vec<RecordBatch> = execute_plan(plan, options)
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let batch_rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(batch_rows, expected_batch_rows);
        let sorted = concat_batches(&batch.schema(), &batches).unwrap();
        assert_eq!(
            sorted["x"].as_primitive::<Int32Type>(),
            &Int32Array::from(vec![0, 1, 2, 3, 4, 5])
        );
    }

    #[test]
    fn test_mem_pool_size_scales_with_partitions() {
        let default_per_partition = DEFAULT_LANCE_MEM_POOL_SIZE_PER_PARTITION;

        // No partitions specified → the session runs with DataFusion's default
        // target_partitions (available parallelism), so the pool must be sized
        // for that effective partition count.
        let opts = LanceExecutionOptions::default();
        assert_eq!(
            opts.mem_pool_size(),
            default_per_partition * get_available_parallelism() as u64
        );

        // 4 partitions → 4x the per-partition size
        let opts = LanceExecutionOptions {
            target_partition: Some(4),
            ..Default::default()
        };
        assert_eq!(opts.mem_pool_size(), default_per_partition * 4);

        // 8 partitions → 8x the per-partition size
        let opts = LanceExecutionOptions {
            target_partition: Some(8),
            ..Default::default()
        };
        assert_eq!(opts.mem_pool_size(), default_per_partition * 8);

        // Explicit mem_pool_size is not scaled
        let opts = LanceExecutionOptions {
            mem_pool_size: Some(50 * 1024 * 1024),
            target_partition: Some(8),
            ..Default::default()
        };
        assert_eq!(opts.mem_pool_size(), 50 * 1024 * 1024);
    }

    #[test]
    fn test_default_pool_fits_sort_merge_reservations_per_partition() {
        let options = LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        };
        let num_partitions = options.effective_target_partition() as usize;
        let task_ctx = new_task_context(&options).unwrap();
        let pool = task_ctx.memory_pool();

        // External sort merges cannot spill their own reservations. Keep three
        // merge reservations per execution partition while leaving 30 MiB per
        // partition for input batches and other operators. Before the default
        // pool followed the effective partition count, the fourth reservation
        // exhausted the single 150 MiB pool on machines with multiple cores.
        let num_reservations = num_partitions * 3;
        let mut reservations = Vec::with_capacity(num_reservations);
        for _ in 0..num_reservations {
            let reservation = MemoryConsumer::new("ExternalSorterMerge[0]").register(pool);
            reservation
                .try_grow(MAX_SORT_SPILL_RESERVATION_BYTES as usize)
                .unwrap();
            reservations.push(reservation);
        }

        assert_eq!(
            pool.reserved(),
            num_reservations * MAX_SORT_SPILL_RESERVATION_BYTES as usize
        );
    }
}
