// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::{
    ChildStats, ChildrenPropertiesMode, ReplaceChildrenOptions, StatisticsArgs,
};
use std::sync::Arc;

use datafusion::{catalog::Session, execution::TaskContext, logical_expr::Expr};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    Statistics, filter::FilterExec, metrics::MetricsSet,
};
use lance_core::{Result, error::DataFusionResult};
use lance_datafusion::planner::Planner;

#[derive(Debug)]
// LanceFilterExec is a wrapper around FilterExec that includes the original
// expression for the filter node. In comparison to a FilterExec, this makes it
// possible for an optimization rule to serialize the filter to substrait and
// send it to a remote worker.
pub struct LanceFilterExec {
    expr: Expr,
    pub filter: Arc<FilterExec>,
}

impl DisplayAs for LanceFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.filter.fmt_as(t, f)
    }
}

impl LanceFilterExec {
    pub fn try_new(expr: Expr, input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let planner = Planner::new(input.schema());
        let predicate = planner.create_physical_expr(&expr)?;
        Self::try_new_with_predicate(expr, predicate, input)
    }

    pub fn try_new_with_session(
        expr: Expr,
        input: Arc<dyn ExecutionPlan>,
        session: &dyn Session,
    ) -> Result<Self> {
        let planner = Planner::new(input.schema());
        let predicate = planner.create_physical_expr_with_session(&expr, session)?;
        Self::try_new_with_predicate(expr, predicate, input)
    }

    fn try_new_with_predicate(
        expr: Expr,
        predicate: Arc<dyn datafusion_physical_plan::PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let filter_exec = FilterExec::try_new(predicate.clone(), input)?;
        Ok(Self {
            expr,
            filter: Arc::new(filter_exec),
        })
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }
}

impl ExecutionPlan for LanceFilterExec {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        self.filter.apply_expressions(f)
    }
    fn name(&self) -> &str {
        "LanceFilterExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.filter.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.filter.children()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.filter.maintains_input_order()
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        options: ReplaceChildrenOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        // Rewrap the result in a LanceFilterExec to preserve the logical expression
        let new_filter_plan = self.filter.clone().replace_children(children, options)?;
        let new_filter = new_filter_plan
            .downcast_ref::<FilterExec>()
            .expect("FilterExec::replace_children should return FilterExec")
            .clone();
        Ok(Arc::new(Self {
            expr: self.expr.clone(),
            filter: Arc::new(new_filter),
        }))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        self.filter.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.filter.metrics()
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        self.filter.child_stats_requests(partition)
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        self.filter.statistics_from_inputs(input_stats, args)
    }

    fn cardinality_effect(&self) -> datafusion_physical_plan::execution_plan::CardinalityEffect {
        self.filter.cardinality_effect()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &datafusion_physical_plan::projection::ProjectionExec,
    ) -> datafusion::error::Result<Option<Arc<dyn ExecutionPlan>>> {
        self.filter.try_swapping_with_projection(projection)
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
