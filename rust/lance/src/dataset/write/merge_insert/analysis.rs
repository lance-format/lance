// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Operation-owned diagnostics for plans that are not children of the merge sink.

use std::collections::BTreeSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use crate::io::exec::{LanceScanExec, filtered_read::FilteredReadExec};
use datafusion::{
    catalog::TableProvider,
    execution::context::SessionContext,
    physical_plan::{
        ExecutionPlan, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricValue},
    },
};
use lance_core::utils::tracing::StreamTracingExt;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_datafusion::exec::{
    ExecutionAnalysis, ExecutionMetricValue, ExecutionSummaryCounts, LanceExecutionOptions,
    execute_plan_with_session_context, execute_plan_with_session_diagnostics, format_plan,
};
use lance_datafusion::spill::{
    ReplaySpillMetrics, spilling_table_provider_with_job_disk_manager,
    spilling_table_provider_with_metrics,
};

use super::{MergeInsertJoin, collect_merge_joins};
mod wire;

/// A separately executed plan. Paths are local to this subplan, identified by `id`.
#[derive(Clone, Debug)]
pub struct MergeSubplan {
    /// Unique within the operation; zero is reserved for the root execution.
    pub id: usize,
    /// Execution boundary, such as index probing or sorting fragment updates.
    pub purpose: String,
    /// Snapshot taken before the stream releases its plan, possibly partial on error/drop.
    pub execution: ExecutionAnalysis,
    /// Physical joins belonging to this subplan.
    pub joins: Vec<MergeInsertJoin>,
}

/// DataFusion spill totals across root/subplans and indexed source materialization.
#[derive(Clone, Debug, Default)]
pub struct MergeOperatorSpills {
    /// Spill events, including measured zero when no operator spills.
    pub spill_count: usize,
    /// Bytes written by spill operators; excludes replay, scratch, and data files.
    pub bytes_written: usize,
    /// Rows spilled by operators, including repeated spills of the same rows.
    pub rows: usize,
    /// Bytes read, absent when operators do not expose this measurement.
    pub bytes_read: Option<usize>,
    /// Cumulative spill write time, absent when unmeasured.
    pub write_duration: Option<Duration>,
    /// Cumulative spill read time, absent when unmeasured.
    pub read_duration: Option<Duration>,
}

#[derive(Debug, Default)]
pub struct AnalysisScope {
    pub target_uri: String,
    target_reads: Mutex<(Option<usize>, Option<usize>)>,
    pub materializations: Mutex<Vec<ExecutionPlanMetricsSet>>,
    pub blobs: crate::dataset::blob::BlobPayloadInventory,
    pub patches: Mutex<Vec<lance_table::format::Fragment>>,
    pub replay: Arc<ReplaySpillMetrics>,
    pub segments: Arc<Mutex<BTreeSet<String>>>,
    next_plan: AtomicUsize,
    plans: Mutex<Vec<MergeSubplan>>,
}

impl AnalysisScope {
    pub fn new(target_uri: impl Into<String>) -> Self {
        Self {
            target_uri: target_uri.into(),
            ..Default::default()
        }
    }

    pub fn record_target_reads(&self, plan: &dyn ExecutionPlan) {
        let dataset = if let Some(scan) = plan.downcast_ref::<FilteredReadExec>() {
            Some(scan.dataset())
        } else {
            plan.downcast_ref::<LanceScanExec>()
                .map(|scan| scan.dataset())
        };
        // Merge target scans carry row identities for the writer. Native source
        // scans do not, even when the source is the target dataset itself.
        if dataset.is_some_and(|dataset| dataset.uri() == self.target_uri)
            && (plan.schema().index_of(ROW_ID).is_ok() || plan.schema().index_of(ROW_ADDR).is_ok())
            && let Some(metrics) = plan.metrics()
        {
            let mut reads = self.target_reads.lock().unwrap_or_else(|e| e.into_inner());
            for metric in metrics.iter() {
                if let MetricValue::Count { name, count } = metric.value() {
                    let target = match name.as_ref() {
                        "rows_scanned" => &mut reads.0,
                        "fragments_scanned" => &mut reads.1,
                        _ => continue,
                    };
                    *target.get_or_insert(0) += count.value();
                }
            }
        }
        for child in plan.children() {
            self.record_target_reads(child.as_ref());
        }
    }

    pub fn target_reads(&self) -> (Option<usize>, Option<usize>) {
        *self.target_reads.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn materialization_summaries(&self) -> Vec<ExecutionSummaryCounts> {
        self.materializations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|metrics| {
                let mut summary = ExecutionSummaryCounts::default();
                let metrics = metrics.clone_inner();
                for metric in metrics.iter() {
                    match metric.value() {
                        MetricValue::SpillCount(count)
                        | MetricValue::SpilledBytes(count)
                        | MetricValue::SpilledRows(count)
                        | MetricValue::Count { count, .. } => {
                            *summary
                                .all_counts
                                .entry(metric.value().name().to_owned())
                                .or_default() += count.value();
                        }
                        MetricValue::Time { time, .. } => {
                            *summary
                                .all_times
                                .entry(metric.value().name().to_owned())
                                .or_default() += time.value();
                        }
                        _ => {}
                    }
                }
                summary
            })
            .collect()
    }

    pub fn plans(&self) -> Vec<MergeSubplan> {
        let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner()).clone();
        plans.sort_by_key(|plan| plan.id);
        plans
    }
}

pub(super) fn scope(session: &SessionContext) -> Option<Arc<AnalysisScope>> {
    session.copied_config().get_extension::<AnalysisScope>()
}

pub(super) fn execute(
    purpose: &'static str,
    plan: Arc<dyn ExecutionPlan>,
    options: LanceExecutionOptions,
    session: &SessionContext,
) -> Result<SendableRecordBatchStream> {
    let Some(scope) = scope(session) else {
        return execute_plan_with_session_context(plan, options, session);
    };
    let id = scope
        .next_plan
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| Error::internal("merge diagnostic subplan id overflow"))?
        + 1;
    let captured = plan.clone();
    let span = tracing::info_span!("merge_analysis_subplan", subplan_id = id, purpose);
    let callback_span = span.clone();
    let guard = span.enter();
    let stream = execute_plan_with_session_diagnostics(
        plan,
        options,
        session,
        Arc::new(move |report| {
            let _guard = callback_span.enter();
            scope.record_target_reads(captured.as_ref());
            tracing::debug!(?report.outcome, "merge analysis subplan finalized");
            let subplan = MergeSubplan {
                id,
                purpose: purpose.to_owned(),
                execution: ExecutionAnalysis {
                    plan: format_plan(captured.clone()),
                    outcome: report.outcome,
                    error: report.error.clone(),
                    elapsed: report.elapsed,
                    // Only the operation report owns the pool peak.
                    pool_peak_reserved_bytes: None,
                    summary: report.summary.clone(),
                    operators: report.operators.clone(),
                },
                joins: collect_merge_joins(captured.as_ref())
                    .into_iter()
                    .map(|mut join| {
                        join.subplan_id = id;
                        join
                    })
                    .collect(),
            };
            scope
                .plans
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(subplan);
        }),
    )?;
    drop(guard);
    let schema = stream.schema();
    Ok(Box::pin(
        datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
            schema,
            stream.stream_in_span(span),
        ),
    ))
}

pub(super) async fn replay(
    stream: SendableRecordBatchStream,
    memory_limit: usize,
    session: &SessionContext,
) -> Result<Arc<dyn TableProvider>> {
    let manager = session.runtime_env().disk_manager.clone();
    Ok(if let Some(scope) = scope(session) {
        spilling_table_provider_with_metrics(stream, memory_limit, manager, scope.replay.clone())
            .await?
    } else {
        spilling_table_provider_with_job_disk_manager(stream, memory_limit, manager).await?
    })
}

pub(super) fn operator_spills<'a>(
    summaries: impl Iterator<Item = &'a ExecutionSummaryCounts>,
) -> MergeOperatorSpills {
    let mut spills = MergeOperatorSpills::default();
    for counts in summaries {
        spills.spill_count += counts.spill_count().unwrap_or(0);
        spills.bytes_written += counts.spilled_bytes().unwrap_or(0);
        spills.rows += counts.spilled_rows().unwrap_or(0);
        if let Some(bytes) = counts.all_counts.get("spill_read_bytes") {
            *spills.bytes_read.get_or_insert(0) += bytes;
        }
        for (name, value) in &counts.all_times {
            let target = match name.as_str() {
                "spill_write_time" => &mut spills.write_duration,
                "spill_read_time" => &mut spills.read_duration,
                _ => continue,
            };
            *target.get_or_insert(Duration::ZERO) += Duration::from_nanos(*value as u64);
        }
    }
    spills
}

pub(super) fn join_measurements<'a>(
    executions: impl Iterator<Item = &'a ExecutionAnalysis>,
) -> (Option<Duration>, Option<usize>) {
    let mut duration = None;
    let mut rows = None;
    for execution in executions {
        for node in &execution.operators {
            if !matches!(node.name.as_str(), "HashJoinExec" | "SortMergeJoinExec") {
                continue;
            }
            for metric in &node.metrics {
                match (&*metric.name, &metric.value) {
                    ("join_time", ExecutionMetricValue::Time(time)) => {
                        *duration.get_or_insert(Duration::ZERO) += *time
                    }
                    ("output_rows", ExecutionMetricValue::Count(count)) => {
                        *rows.get_or_insert(0) += count
                    }
                    _ => {}
                }
            }
        }
    }
    (duration, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{cast::AsArray, record_batch, types::Int32Type};
    use datafusion::physical_plan::{sorts::sort::SortExec, stream::RecordBatchStreamAdapter};
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
    use futures::TryStreamExt;
    use lance_datafusion::exec::{OneShotExec, new_session_context_with_memory_tracking};

    #[tokio::test]
    async fn test_materialization_spills_are_attributed_once() {
        let batch = record_batch!(("x", Int32, [1, 2, 3])).unwrap();
        let input = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::iter([Ok(batch.clone()), Ok(batch)]),
        ));
        let session =
            new_session_context_with_memory_tracking(&LanceExecutionOptions::default()).unwrap();
        let (mut output, metrics) = lance_datafusion::spill::materialize_stream(
            input,
            session.runtime_env(),
            0,
            "testing merge source materialization",
        )
        .await
        .unwrap();
        let mut rows = 0;
        while let Some(batch) = output.try_next().await.unwrap() {
            rows += batch.num_rows();
        }
        assert_eq!(rows, 6);
        let scope = AnalysisScope::default();
        scope.materializations.lock().unwrap().push(metrics);
        let summaries = scope.materialization_summaries();
        let spills = operator_spills(summaries.iter());
        assert!(spills.bytes_written > 0);
        assert_eq!(spills.spill_count, 1);
        assert_eq!(spills.rows, 6);
        assert_eq!(spills.bytes_written, summaries[0].spilled_bytes().unwrap());
        assert_eq!(scope.replay.snapshot().bytes_written, 0);
    }

    #[tokio::test]
    async fn test_nested_sort_spills_are_attributed_once() {
        let batch = record_batch!(("x", Int32, (0..128).rev().collect::<Vec<i32>>())).unwrap();
        let input = OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::iter((0..128).map(|_| Ok(batch.clone())).collect::<Vec<_>>()),
        )));
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(
            col("x", &batch.schema()).unwrap(),
        )])
        .unwrap();
        let options = LanceExecutionOptions {
            use_spilling: true,
            mem_pool_size: Some(64 * 1024),
            target_partition: Some(1),
            batch_size: Some(128),
            ..Default::default()
        };
        let session = new_session_context_with_memory_tracking(&options).unwrap();
        let scope = Arc::new(AnalysisScope::default());
        let session = SessionContext::new_with_config_rt(
            session.copied_config().with_extension(scope.clone()),
            session.runtime_env(),
        );
        let mut output = execute(
            "fragment_update_sort",
            Arc::new(SortExec::new(ordering, Arc::new(input))),
            options,
            &session,
        )
        .unwrap();
        let mut rows = 0;
        let mut previous = 0;
        while let Some(batch) = output.try_next().await.unwrap() {
            for value in batch["x"].as_primitive::<Int32Type>().values() {
                assert!(*value >= previous);
                previous = *value;
            }
            rows += batch.num_rows();
        }
        // Repeated EOF polling must not duplicate the subplan or its spill traffic.
        assert!(output.try_next().await.unwrap().is_none());
        drop(output);
        let plans = scope.plans();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].id, 1);
        assert_eq!(rows, 128 * 128);
        let spills = operator_spills(plans.iter().map(|plan| &plan.execution.summary));
        assert!(spills.bytes_written > 0);
        assert!(spills.spill_count > 0);
        assert_eq!(
            spills.bytes_written,
            plans[0].execution.summary.spilled_bytes().unwrap()
        );
        assert_eq!(scope.replay.snapshot().bytes_written, 0);
    }
}
