// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion execution plans

use std::{
    collections::HashMap,
    fmt::{self, Formatter},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use chrono::{DateTime, Utc};

use arrow_schema::Schema as ArrowSchema;
use datafusion::{
    catalog::{TableProvider, streaming::StreamingTable},
    dataframe::DataFrame,
    execution::{TaskContext, context::SessionContext},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        analyze::AnalyzeExec, coalesce_partitions::CoalescePartitionsExec, metrics::MetricType,
        metrics::MetricValue, stream::RecordBatchStreamAdapter, streaming::PartitionStream,
    },
};
use futures::StreamExt;
use lance_core::{Error, Result, utils::tracing::StreamTracingExt};
use tracing::Span;

pub use lance_datafusion_exec::exec::{
    ExecutionStatsCallback, ExecutionSummaryCounts, HardCapBatchSizeExec, LanceExecutionOptions,
    OneShotExec, StrictBatchSizeExec, collect_execution_metrics,
};
use lance_datafusion_exec::exec::{execute_plan_with_context, new_runtime_env, new_session_config};

use crate::udf::register_functions;

struct TracedExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    span: Span,
}

impl TracedExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, span: Span) -> Self {
        Self {
            properties: input.properties().clone(),
            input,
            span,
        }
    }
}

impl DisplayAs for TracedExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "TracedExec")
            }
        }
    }
}

impl std::fmt::Debug for TracedExec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "TracedExec")
    }
}
impl ExecutionPlan for TracedExec {
    fn name(&self) -> &str {
        "TracedExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
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
            properties: self.properties.clone(),
            span: self.span.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let _guard = self.span.enter();
        let stream = self.input.execute(partition, context)?;
        let schema = stream.schema();
        let stream = stream.stream_in_span(self.span.clone());
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

pub fn new_session_context(options: &LanceExecutionOptions) -> SessionContext {
    let ctx =
        SessionContext::new_with_config_rt(new_session_config(options), new_runtime_env(options));
    register_functions(&ctx);
    ctx
}

/// Cache key for session contexts based on resolved configuration values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SessionContextCacheKey {
    mem_pool_size: u64,
    max_temp_directory_size: u64,
    target_partition: Option<usize>,
}

impl SessionContextCacheKey {
    fn from_options(options: &LanceExecutionOptions) -> Self {
        Self {
            mem_pool_size: options.mem_pool_size(),
            max_temp_directory_size: options.max_temp_directory_size(),
            target_partition: options.target_partition,
        }
    }
}

struct CachedSessionContext {
    context: SessionContext,
    last_access: std::time::Instant,
}

fn get_session_cache() -> &'static Mutex<HashMap<SessionContextCacheKey, CachedSessionContext>> {
    static SESSION_CACHE: OnceLock<Mutex<HashMap<SessionContextCacheKey, CachedSessionContext>>> =
        OnceLock::new();
    SESSION_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_max_cache_size() -> usize {
    const DEFAULT_CACHE_SIZE: usize = 4;
    static MAX_CACHE_SIZE: OnceLock<usize> = OnceLock::new();
    *MAX_CACHE_SIZE.get_or_init(|| {
        std::env::var("LANCE_SESSION_CACHE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_CACHE_SIZE)
    })
}

/// Reuses unbounded sessions, while giving each spilling caller a fresh bounded
/// memory pool so concurrent sorts cannot consume each other's headroom.
pub fn get_session_context(options: &LanceExecutionOptions) -> SessionContext {
    if options.use_spilling() {
        return new_session_context(options);
    }

    let key = SessionContextCacheKey::from_options(options);
    let mut cache = get_session_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // If key exists, update access time and return
    if let Some(entry) = cache.get_mut(&key) {
        entry.last_access = std::time::Instant::now();
        return entry.context.clone();
    }

    // Evict least recently used entry if cache is full
    if cache.len() >= get_max_cache_size()
        && let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, _)| k.clone())
    {
        cache.remove(&lru_key);
    }

    let context = new_session_context(options);
    cache.insert(
        key,
        CachedSessionContext {
            context: context.clone(),
            last_access: std::time::Instant::now(),
        },
    );
    context
}

/// Returns a new task context to execute one plan with `options`.
///
/// Every execution needs its own context: Lance operators treat a task
/// context's identity as the identity of one plan execution, for example to
/// share a MultiMatch prefilter mask only between the fields of one query.
fn get_task_context(
    session_ctx: &SessionContext,
    options: &LanceExecutionOptions,
) -> Result<Arc<TaskContext>> {
    // Build from the session state in place. `SessionContext::state` would clone
    // the whole state (every function map, rule list and option) only to drop
    // it, which is a measurable share of CPU for short queries.
    let task_ctx = TaskContext::from(session_ctx);
    let Some(batch_size) = options.validated_batch_size()? else {
        return Ok(Arc::new(task_ctx));
    };
    let mut session_config = task_ctx.session_config().clone();
    session_config.options_mut().execution.batch_size = batch_size;
    Ok(Arc::new(task_ctx.with_session_config(session_config)))
}

/// Executes a plan under a task context of the cached session for `options`,
/// which has Lance's functions registered.
///
/// See [`execute_plan_with_context`] for how partitions and metrics are handled.
pub fn execute_plan(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
) -> Result<SendableRecordBatchStream> {
    let task_context = get_task_context(&get_session_context(&options), &options)?;
    execute_plan_with_context(plan, options, task_context)
}

pub async fn analyze_plan(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
) -> Result<String> {
    analyze_plan_with_context(plan, options, None).await
}

/// Analyze a plan, optionally under a caller-provided [`TaskContext`].
///
/// When `task_context` is `Some`, the plan executes under it instead of the
/// context derived from `options`. Callers whose nodes read session-config
/// extensions at execution time (e.g. distributed routing identity) must pass
/// the context carrying those extensions; otherwise the nodes error during
/// `execute` and `AnalyzeExec` reports an empty, unexecuted plan tree instead
/// of surfacing the error.
pub async fn analyze_plan_with_context(
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
    task_context: Option<Arc<TaskContext>>,
) -> Result<String> {
    // This is needed as AnalyzeExec launches a thread task per
    // partition, and we want these to be connected to the parent span
    let plan = Arc::new(TracedExec::new(plan, Span::current()));

    let schema = plan.schema();
    // TODO(tsaucer) I chose SUMMARY here but do we also want DEV?
    let analyze = Arc::new(AnalyzeExec::new(
        true,
        true,
        vec![MetricType::Summary],
        None,
        plan,
        schema,
    ));

    let session_ctx = get_session_context(&options);
    let task_context = match task_context {
        Some(task_context) => task_context,
        None => get_task_context(&session_ctx, &options)?,
    };
    assert_eq!(analyze.properties().partitioning.partition_count(), 1);
    let mut stream = analyze
        .execute(0, task_context)
        .map_err(|err| Error::io(format!("Failed to execute analyze plan: {}", err)))?;

    // fully execute the plan
    while (stream.next().await).is_some() {}

    let result = format_plan(analyze);
    Ok(result)
}

pub fn format_plan(plan: Arc<dyn ExecutionPlan>) -> String {
    /// A visitor which calculates additional metrics for all the plans.
    struct CalculateVisitor {
        highest_index: usize,
        index_to_elapsed: HashMap<usize, Duration>,
    }

    /// Result of calculating metrics for a subtree
    struct SubtreeMetrics {
        min_start: Option<DateTime<Utc>>,
        max_end: Option<DateTime<Utc>>,
    }

    impl CalculateVisitor {
        fn calculate_metrics(&mut self, plan: &Arc<dyn ExecutionPlan>) -> SubtreeMetrics {
            self.highest_index += 1;
            let plan_index = self.highest_index;

            // Get timestamps for this node
            let (mut min_start, mut max_end) = Self::node_timerange(plan);

            // Accumulate from children
            for child in plan.children() {
                let child_metrics = self.calculate_metrics(child);
                min_start = Self::min_option(min_start, child_metrics.min_start);
                max_end = Self::max_option(max_end, child_metrics.max_end);
            }

            // Calculate wall clock duration for this subtree (only if we have timestamps)
            let elapsed = match (min_start, max_end) {
                (Some(start), Some(end)) => Some((end - start).to_std().unwrap_or_default()),
                _ => None,
            };

            if let Some(e) = elapsed {
                self.index_to_elapsed.insert(plan_index, e);
            }

            SubtreeMetrics { min_start, max_end }
        }

        fn node_timerange(
            plan: &Arc<dyn ExecutionPlan>,
        ) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
            let Some(metrics) = plan.metrics() else {
                return (None, None);
            };
            let min_start = metrics
                .iter()
                .filter_map(|m| match m.value() {
                    MetricValue::StartTimestamp(ts) => ts.value(),
                    _ => None,
                })
                .min();
            let max_end = metrics
                .iter()
                .filter_map(|m| match m.value() {
                    MetricValue::EndTimestamp(ts) => ts.value(),
                    _ => None,
                })
                .max();
            (min_start, max_end)
        }

        fn min_option(a: Option<DateTime<Utc>>, b: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
            [a, b].into_iter().flatten().min()
        }

        fn max_option(a: Option<DateTime<Utc>>, b: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
            [a, b].into_iter().flatten().max()
        }
    }

    /// A visitor which prints out all the plans.
    struct PrintVisitor {
        highest_index: usize,
        indent: usize,
    }
    impl PrintVisitor {
        fn write_output(
            &mut self,
            plan: &Arc<dyn ExecutionPlan>,
            f: &mut Formatter,
            calcs: &CalculateVisitor,
        ) -> std::fmt::Result {
            self.highest_index += 1;
            write!(f, "{:indent$}", "", indent = self.indent * 2)?;

            // Format the plan description
            let displayable =
                datafusion::physical_plan::display::DisplayableExecutionPlan::new(plan.as_ref());
            let plan_str = displayable.one_line().to_string();
            let plan_str = plan_str.trim();

            // Write operator with elapsed time inserted after the name
            match calcs.index_to_elapsed.get(&self.highest_index) {
                Some(elapsed) => match plan_str.find(": ") {
                    Some(i) => write!(
                        f,
                        "{}: elapsed={elapsed:?}, {}",
                        &plan_str[..i],
                        &plan_str[i + 2..]
                    )?,
                    None => write!(f, "{plan_str}, elapsed={elapsed:?}")?,
                },
                None => write!(f, "{plan_str}")?,
            }

            if let Some(metrics) = plan.metrics() {
                let metrics = metrics
                    .aggregate_by_name()
                    .sorted_for_display()
                    .timestamps_removed();

                write!(f, ", metrics=[{metrics}]")?;
            } else {
                write!(f, ", metrics=[]")?;
            }
            writeln!(f)?;
            self.indent += 1;
            for child in plan.children() {
                self.write_output(child, f, calcs)?;
            }
            self.indent -= 1;
            std::fmt::Result::Ok(())
        }
    }
    // A wrapper which prints out a plan.
    struct PrintWrapper {
        plan: Arc<dyn ExecutionPlan>,
    }
    impl fmt::Display for PrintWrapper {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            let mut calcs = CalculateVisitor {
                highest_index: 0,
                index_to_elapsed: HashMap::new(),
            };
            calcs.calculate_metrics(&self.plan);
            let mut prints = PrintVisitor {
                highest_index: 0,
                indent: 0,
            };
            prints.write_output(&self.plan, f, &calcs)
        }
    }
    let wrapper = PrintWrapper { plan };
    format!("{}", wrapper)
}

pub trait SessionContextExt {
    /// Creates a DataFrame for reading a stream of data
    ///
    /// This dataframe may only be queried once, future queries will fail
    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame>;
}

pub struct OneShotPartitionStream {
    data: Arc<Mutex<Option<SendableRecordBatchStream>>>,
    schema: Arc<ArrowSchema>,
}

impl std::fmt::Debug for OneShotPartitionStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.lock().unwrap();
        f.debug_struct("OneShotPartitionStream")
            .field("exhausted", &data.is_none())
            .field("schema", self.schema.as_ref())
            .finish()
    }
}

impl OneShotPartitionStream {
    pub fn new(data: SendableRecordBatchStream) -> Self {
        let schema = data.schema();
        Self {
            data: Arc::new(Mutex::new(Some(data))),
            schema,
        }
    }
}

impl PartitionStream for OneShotPartitionStream {
    fn schema(&self) -> &arrow_schema::SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let mut stream = self.data.lock().unwrap();
        stream
            .take()
            .expect("Attempt to consume a one shot dataframe multiple times")
    }
}

impl SessionContextExt for SessionContext {
    fn read_one_shot(
        &self,
        data: SendableRecordBatchStream,
    ) -> datafusion::common::Result<DataFrame> {
        let schema = data.schema();
        let part_stream = Arc::new(OneShotPartitionStream::new(data));
        let provider = StreamingTable::try_new(schema, vec![part_stream])?;
        self.read_table(Arc::new(provider))
    }
}

/// Scan a [`TableProvider`] into a single-partition [`SendableRecordBatchStream`].
///
/// Multi-partition providers are coalesced into a single partition. This adapts a
/// re-scannable provider back into the one stream the writer pipeline consumes;
/// re-scanning the same provider (e.g. on a write retry) yields a fresh stream.
///
/// # Examples
///
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::{Int32Array, RecordBatch};
/// # use arrow_schema::{DataType, Field, Schema};
/// # use datafusion::catalog::TableProvider;
/// # use datafusion::datasource::MemTable;
/// # use futures::TryStreamExt;
/// # use lance_datafusion::exec::provider_to_stream;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
/// let batch =
///     RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])?;
/// let provider: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);
///
/// // A re-scannable provider yields a fresh stream on each call.
/// let batches: Vec<RecordBatch> = provider_to_stream(provider).await?.try_collect().await?;
/// assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
/// # Ok(())
/// # }
/// ```
pub async fn provider_to_stream(
    provider: Arc<dyn TableProvider>,
) -> Result<SendableRecordBatchStream> {
    let ctx = SessionContext::new();
    let plan = provider.scan(&ctx.state(), None, &[], None).await?;
    let plan: Arc<dyn ExecutionPlan> =
        if plan.properties().output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(plan))
        } else {
            plan
        };
    Ok(plan.execute(0, ctx.task_ctx())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::record_batch;
    use datafusion::execution::memory_pool::MemoryConsumer;
    use datafusion_common::DataFusionError;

    use datafusion::execution::memory_pool::MemoryLimit;
    use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
    use futures::stream;

    // Serialize cache tests since they share global state
    static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_session_context_cache() {
        let _lock = CACHE_TEST_LOCK.lock().unwrap();
        let cache = get_session_cache();

        // Clear any existing entries from other tests
        cache.lock().unwrap().clear();

        // Create first session with default options
        let opts1 = LanceExecutionOptions::default();
        let _ctx1 = get_session_context(&opts1);

        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.len(), 1);
        }

        // Same options should reuse cached session (no new entry)
        let _ctx1_again = get_session_context(&opts1);
        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.len(), 1);
        }

        // Different non-spilling options should create a new entry.
        let opts2 = LanceExecutionOptions {
            target_partition: Some(1),
            ..Default::default()
        };
        let _ctx2 = get_session_context(&opts2);
        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.len(), 2);
        }
    }

    #[test]
    fn test_spilling_executions_have_independent_memory_pools() {
        let options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(150 * 1024 * 1024),
            target_partition: Some(1),
            ..Default::default()
        };
        let contexts: Vec<_> = (0..3).map(|_| get_session_context(&options)).collect();
        let mut reservations = Vec::with_capacity(contexts.len());

        // Each reservation fits its own pool, but any two exceed a shared pool.
        for context in &contexts {
            let task_context = context.task_ctx();
            let pool = task_context.memory_pool();
            let reservation = MemoryConsumer::new("ExternalSorterMerge[0]").register(pool);
            reservation.try_grow(100 * 1024 * 1024).unwrap();
            assert_eq!(pool.reserved(), 100 * 1024 * 1024);
            reservations.push(reservation);
        }

        assert_eq!(reservations.len(), 3);
    }

    #[test]
    fn test_session_context_cache_lru_eviction() {
        let _lock = CACHE_TEST_LOCK.lock().unwrap();
        let cache = get_session_cache();

        // Clear any existing entries from other tests
        cache.lock().unwrap().clear();

        // Create 4 different configurations to fill the cache
        let configs: Vec<LanceExecutionOptions> = (0..4)
            .map(|i| LanceExecutionOptions {
                mem_pool_size: Some((i + 1) as u64 * 1024 * 1024),
                ..Default::default()
            })
            .collect();

        for config in &configs {
            let _ctx = get_session_context(config);
        }

        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.len(), 4);
        }

        // Access config[0] to make it more recently used than config[1]
        // (config[0] was inserted first, so without this access it would be evicted)
        std::thread::sleep(std::time::Duration::from_millis(1));
        let _ctx = get_session_context(&configs[0]);

        // Add a 5th configuration - should evict config[1] (now least recently used)
        let opts5 = LanceExecutionOptions {
            mem_pool_size: Some(5 * 1024 * 1024),
            ..Default::default()
        };
        let _ctx5 = get_session_context(&opts5);

        {
            let cache_guard = cache.lock().unwrap();
            assert_eq!(cache_guard.len(), 4);

            // config[0] should still be present (was accessed recently)
            let key0 = SessionContextCacheKey::from_options(&configs[0]);
            assert!(
                cache_guard.contains_key(&key0),
                "config[0] should still be cached after recent access"
            );

            // config[1] should be evicted (was least recently used)
            let key1 = SessionContextCacheKey::from_options(&configs[1]);
            assert!(
                !cache_guard.contains_key(&key1),
                "config[1] should have been evicted"
            );

            // New config should be present
            let key5 = SessionContextCacheKey::from_options(&opts5);
            assert!(
                cache_guard.contains_key(&key5),
                "new config should be cached"
            );
        }
    }

    #[test]
    fn test_task_context_per_execution() {
        let _lock = CACHE_TEST_LOCK.lock().unwrap();

        let default_options = LanceExecutionOptions::default();
        let session_ctx = get_session_context(&default_options);
        let task_ctx = get_task_context(&session_ctx, &default_options).unwrap();
        // Lance operators key per-execution state on the task context's identity.
        assert!(!Arc::ptr_eq(
            &task_ctx,
            &get_task_context(&session_ctx, &default_options).unwrap()
        ));
        assert_eq!(task_ctx.session_id(), session_ctx.session_id());
        assert_eq!(
            task_ctx.session_config().batch_size(),
            session_ctx.copied_config().batch_size()
        );
        assert!(Arc::ptr_eq(
            &task_ctx.runtime_env(),
            &session_ctx.runtime_env()
        ));
        assert!(task_ctx.scalar_functions().contains_key("contains_tokens"));

        let spill_options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(64 * 1024 * 1024),
            target_partition: Some(3),
            batch_size: Some(17),
            ..Default::default()
        };
        let spill_session_ctx = get_session_context(&spill_options);
        let spill_task_ctx = get_task_context(&spill_session_ctx, &spill_options).unwrap();
        assert_eq!(spill_task_ctx.session_config().batch_size(), 17);
        assert_eq!(spill_task_ctx.session_config().target_partitions(), 3);
        assert!(
            spill_task_ctx
                .scalar_functions()
                .contains_key("contains_tokens")
        );
        // The batch size override applies to this execution only, and the
        // execution still shares the session's memory pool.
        assert_ne!(spill_session_ctx.copied_config().batch_size(), 17);
        assert!(Arc::ptr_eq(
            &spill_task_ctx.runtime_env(),
            &spill_session_ctx.runtime_env()
        ));
        assert!(matches!(
            spill_task_ctx.memory_pool().memory_limit(),
            MemoryLimit::Finite(limit) if limit == 64 * 1024 * 1024
        ));
    }

    #[tokio::test]
    async fn test_zero_batch_size_is_rejected() {
        let batch = record_batch!(("x", Int32, [1, 2, 3])).unwrap();
        let options = LanceExecutionOptions {
            batch_size: Some(0),
            ..Default::default()
        };

        let plan = Arc::new(OneShotExec::from_batch(batch.clone()));
        let Err(err) = execute_plan(plan, options.clone()) else {
            panic!("execute_plan accepted batch_size 0");
        };
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains("batch_size must be greater than 0")
        );

        let plan = Arc::new(OneShotExec::from_batch(batch));
        let err = analyze_plan(plan, options).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains("batch_size must be greater than 0")
        );
    }

    /// A marker a node reads from the session-config extensions at execute time.
    #[derive(Debug)]
    struct RequiredExtension;

    /// Execution node that only succeeds when [`RequiredExtension`] is present
    /// on the task context's session config. This mirrors distributed routing
    /// nodes that read a session-config identity extension during `execute`.
    #[derive(Debug)]
    struct NeedsExtensionExec {
        properties: Arc<PlanProperties>,
        /// Set once the node reaches execution with the extension present.
        /// Observed by the test so that dropping context forwarding (which
        /// makes `execute` error before this point) is detectable.
        executed: Arc<std::sync::atomic::AtomicBool>,
    }

    impl NeedsExtensionExec {
        fn new(executed: Arc<std::sync::atomic::AtomicBool>) -> Self {
            let schema = Arc::new(ArrowSchema::empty());
            Self {
                properties: Arc::new(PlanProperties::new(
                    EquivalenceProperties::new(schema),
                    Partitioning::UnknownPartitioning(1),
                    EmissionType::Incremental,
                    Boundedness::Bounded,
                )),
                executed,
            }
        }
    }

    impl DisplayAs for NeedsExtensionExec {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
            write!(f, "NeedsExtensionExec")
        }
    }

    impl ExecutionPlan for NeedsExtensionExec {
        fn name(&self) -> &str {
            "NeedsExtensionExec"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(
            &self,
            _partition: usize,
            context: Arc<TaskContext>,
        ) -> datafusion_common::Result<SendableRecordBatchStream> {
            if context
                .session_config()
                .get_extension::<RequiredExtension>()
                .is_none()
            {
                return Err(DataFusionError::Execution(
                    "missing required session-config extension".to_string(),
                ));
            }
            self.executed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let schema = self.schema();
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                stream::empty(),
            )))
        }
    }

    // Regression: analyze must run under a caller-provided TaskContext so nodes
    // that read a session-config extension at execute time see it. Without the
    // context the node errors and AnalyzeExec would otherwise report an empty,
    // unexecuted plan tree.
    #[tokio::test]
    async fn test_analyze_plan_uses_provided_task_context() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(NeedsExtensionExec::new(executed.clone()));

        // Default context lacks the extension: the node errors during execute
        // (never reaching the `executed` flag), but AnalyzeExec absorbs that
        // per-partition failure and reports an empty, unexecuted plan tree
        // rather than propagating the error. This is the regression symptom.
        let report = analyze_plan(plan.clone(), LanceExecutionOptions::default())
            .await
            .expect("AnalyzeExec swallows the node's execute error into an Ok report");
        assert!(
            report.contains("NeedsExtensionExec, metrics=[]"),
            "expected an empty, unexecuted NeedsExtensionExec node, got: {report}"
        );
        assert!(
            !executed.load(Ordering::SeqCst),
            "node must not execute successfully without the extension"
        );

        // A context carrying the extension executes the node successfully.
        let options = LanceExecutionOptions::default();
        let session_ctx = get_session_context(&options);
        let config = session_ctx
            .task_ctx()
            .session_config()
            .clone()
            .with_extension(Arc::new(RequiredExtension));
        let task_ctx = session_ctx.task_ctx();
        let task_ctx = Arc::new(TaskContext::new(
            task_ctx.task_id(),
            task_ctx.session_id(),
            config,
            task_ctx.scalar_functions().clone(),
            task_ctx.higher_order_functions().clone(),
            task_ctx.aggregate_functions().clone(),
            task_ctx.window_functions().clone(),
            task_ctx.runtime_env(),
        ));
        let report = analyze_plan_with_context(plan, options, Some(task_ctx))
            .await
            .expect("analyze should succeed when the extension is present");
        assert!(report.contains("NeedsExtensionExec"));
        // The node only reaches this flag when the supplied context is actually
        // forwarded to `execute`; dropping the forwarding fails this assertion.
        assert!(
            executed.load(Ordering::SeqCst),
            "supplied context must be forwarded so the node executes"
        );
    }
}
