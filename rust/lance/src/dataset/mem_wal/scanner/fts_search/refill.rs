// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded per-source search across primary-key updates.

use std::{fmt, sync::Arc};

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{TaskContext, memory_pool::MemoryConsumer};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricValue, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchReceiverStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    execute_stream,
};
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_datafusion::utils::{BYTES_READ_METRIC, IOPS_METRIC, REQUESTS_METRIC};
use lance_index::scalar::FullTextSearchQuery;

use super::super::block_list::GenMembership;
use super::super::exec::filter_blocked_keys;
use super::super::generation_read::GenerationRead;
use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;
use crate::dataset::scanner::Scanner;

pub(super) enum SourceScanner {
    Dataset(Box<Scanner>),
    MemTable(Box<MemTableScanner>),
}

/// The dataset version or memtable prefix, projection and filters stay fixed
/// across candidate budgets. A transient memtable index is also reused.
pub(super) struct SourceSearch {
    pub(super) scanner: SourceScanner,
    pub(super) query: FullTextSearchQuery,
    pub(super) reconciliation: Option<(GenerationRead, Option<Expr>)>,
}

impl SourceSearch {
    pub(super) async fn plan(&self, budget: Option<usize>) -> Result<Arc<dyn ExecutionPlan>> {
        let limit = budget.map(i64::try_from).transpose().map_err(|_| {
            Error::invalid_input(format!("FTS candidate limit {budget:?} exceeds i64::MAX"))
        })?;
        let query = self.query.clone().limit(limit);
        let plan = match &self.scanner {
            SourceScanner::Dataset(scanner) => {
                let mut scanner = scanner.as_ref().clone();
                scanner.full_text_search(query)?;
                Box::pin(scanner.create_plan()).await?
            }
            SourceScanner::MemTable(scanner) => {
                let mut scanner = scanner.as_ref().clone();
                scanner.full_text_search(query)?;
                Box::pin(scanner.create_plan()).await?
            }
        };
        match &self.reconciliation {
            Some((generation, above)) => generation.reconcile_above(plan, above),
            None => Ok(plan),
        }
    }
}

#[derive(Clone)]
pub(super) struct FtsRefillExec {
    first: Arc<dyn ExecutionPlan>,
    search: Arc<SourceSearch>,
    k: usize,
    pk_columns: Arc<[String]>,
    blocked: Arc<[GenMembership]>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl FtsRefillExec {
    pub(super) fn new(
        first: Arc<dyn ExecutionPlan>,
        search: SourceSearch,
        k: usize,
        pk_columns: Vec<String>,
        blocked: Vec<GenMembership>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(first.schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            first,
            search: Arc::new(search),
            k,
            pk_columns: pk_columns.into(),
            blocked: blocked.into(),
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl fmt::Debug for FtsRefillExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FtsRefillExec")
            .field("k", &self.k)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for FtsRefillExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FtsRefillExec: k={}", self.k)
    }
}

impl ExecutionPlan for FtsRefillExec {
    fn name(&self) -> &str {
        "FtsRefillExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.first]
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let [first] = children.try_into().map_err(|_| {
            DataFusionError::Internal("FtsRefillExec requires one child".to_owned())
        })?;
        Ok(Arc::new(Self {
            first,
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.as_ref().clone()
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "FtsRefillExec expected partition 0, got {partition}"
            )));
        }
        let mut result = RecordBatchReceiverStream::builder(self.schema(), 1);
        let tx = result.tx();
        let mut plan = self.first.clone();
        let (search, pk_columns, blocked) = (
            self.search.clone(),
            self.pk_columns.clone(),
            self.blocked.clone(),
        );
        let k = self.k;
        let candidates = MetricBuilder::new(&self.metrics).counter("candidates_read", partition);
        let metrics = self.metrics.clone();
        let reservation = MemoryConsumer::new("FtsRefillExec").register(ctx.memory_pool());
        result.spawn(async move {
            let mut budget = k;
            loop {
                let mut stream = execute_stream(plan.clone(), ctx.clone())?;
                let mut seen = 0;
                let mut live = 0;
                let mut pending = Vec::new();
                while let Some(batch) = stream.try_next().await? {
                    seen += batch.num_rows();
                    candidates.add(batch.num_rows());
                    let batch = filter_blocked_keys(batch, &pk_columns, &blocked).await?;
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    live += batch.num_rows();
                    let held = reservation.new_empty();
                    if live < k {
                        held.try_grow(batch.get_array_memory_size())?;
                    }
                    pending.push((batch, held));
                    // Once this round will win, stream the rest into the outer
                    // score sort. Until then buffer fewer than k survivors.
                    if live >= k {
                        for (batch, _held) in pending.drain(..) {
                            if tx.send(Ok(batch)).await.is_err() {
                                return Ok(());
                            }
                        }
                    }
                }
                if budget != k {
                    // The first plan is a visible child. Preserve the storage
                    // and index metrics from dynamically planned retries too.
                    plan.apply(|node| {
                        if let Some(set) = node.metrics() {
                            for metric in set.iter() {
                                let keep = match metric.value() {
                                    MetricValue::Count { .. } | MetricValue::Time { .. } => true,
                                    MetricValue::Gauge { name, .. } => matches!(
                                        name.as_ref(),
                                        IOPS_METRIC | REQUESTS_METRIC | BYTES_READ_METRIC
                                    ),
                                    _ => false,
                                };
                                if keep {
                                    metrics.register(metric.clone());
                                }
                            }
                        }
                        Ok(TreeNodeRecursion::Continue)
                    })?;
                }
                if live >= k || seen < budget || budget == i64::MAX as usize {
                    for (batch, _held) in pending {
                        if tx.send(Ok(batch)).await.is_err() {
                            break;
                        }
                    }
                    return Ok(());
                }
                // Re-run the wider top-k on the same snapshot. Discard the
                // entire earlier prefix: ties need not break in the same order.
                drop(pending);
                budget = budget.saturating_mul(2).min(i64::MAX as usize);
                plan = Box::pin(search.plan(Some(budget))).await?;
            }
        });
        Ok(result.build())
    }
}
