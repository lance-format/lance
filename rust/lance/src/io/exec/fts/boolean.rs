// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Boost and Boolean query composition nodes.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::NullEquality;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use futures::stream::{self};
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{ROW_ID, Result, utils::tracing::StreamTracingExt};
use lance_datafusion::utils::{MetricsExt, PARTITIONS_SEARCHED_METRIC};

use lance_index::scalar::inverted::query::BooleanQuery;
use lance_index::scalar::inverted::query::{BoostQuery, FtsSearchParams};
use lance_index::scalar::inverted::{DOC_INDEX_COL, FTS_SCHEMA};
use tracing::instrument;

use super::matches::{
    batch_document_keys, batch_scored_document_keys, batch_scored_document_keys_sum_scores,
    compare_scored_documents, document_key_scores_batch,
};
use super::selection::FtsIndexMetrics;

#[derive(Debug)]

pub struct BoostQueryExec {
    query: BoostQuery,
    params: FtsSearchParams,
    positive: Arc<dyn ExecutionPlan>,
    negative: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BoostQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BoostQuery: negative_boost={}",
                    self.query.negative_boost
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "BoostQuery\nnegative_boost={}",
                    self.query.negative_boost
                )
            }
        }
    }
}

impl BoostQueryExec {
    pub fn new(
        query: BoostQuery,
        params: FtsSearchParams,
        positive: Arc<dyn ExecutionPlan>,
        negative: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let schema = positive.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            positive,
            negative,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn query(&self) -> &BoostQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn positive(&self) -> &Arc<dyn ExecutionPlan> {
        &self.positive
    }

    pub fn negative(&self) -> &Arc<dyn ExecutionPlan> {
        &self.negative
    }
}

impl ExecutionPlan for BoostQueryExec {
    fn name(&self) -> &str {
        "BoostQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.positive, &self.negative]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.
        // It must be run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }

        let negative = children.pop().unwrap();
        let positive = children.pop().unwrap();
        Ok(Arc::new(Self {
            query: self.query.clone(),
            params: self.params.clone(),
            positive,
            negative,
            schema: self.schema.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "boost_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let positive = self.positive.execute(partition, context.clone())?;
        let negative = self.negative.execute(partition, context)?;
        let schema = self.schema.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let positive = positive.try_collect::<Vec<_>>().await?;
            let negative = negative.try_collect::<Vec<_>>().await?;

            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let mut res = HashMap::new();
            for batch in positive {
                for (key, score) in batch_scored_document_keys(&batch)? {
                    res.insert(key, score);
                }
            }
            for batch in negative {
                for (key, neg_score) in batch_scored_document_keys(&batch)? {
                    if let Some(score) = res.get_mut(&key) {
                        *score -= query.negative_boost * neg_score;
                    }
                }
            }

            let documents = res
                .into_iter()
                .sorted_unstable_by(compare_scored_documents)
                .take(params.limit.unwrap_or(usize::MAX))
                .collect::<Vec<_>>();
            metrics.baseline_metrics.record_output(documents.len());

            let batch = document_key_scores_batch(schema, documents)?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

/// Identifies which clause of a [`BooleanQuery`] a list of child execs
/// belongs to. Used by [`build_boolean_query_children`] to pick the
/// right exec shape per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolSlot {
    Should,
    Must,
    MustNot,
}

/// Combine N children into the per-slot exec shape that
/// [`BooleanQueryExec::new`] expects. Used by `Scanner::plan_fts` to
/// assemble the per-slot exec shape:
///
/// | slot      | 0 children                 | 1 child       | N children                                          |
/// |-----------|----------------------------|---------------|-----------------------------------------------------|
/// | Should    | `Some(EmptyExec(FTS))`     | `Some(child)` | `Some(Union -> Repartition(RoundRobinBatch(1)))`    |
/// | Must      | `None`                     | `Some(child)` | `Some(chained HashJoin on row_id)`                  |
/// | MustNot   | `Some(EmptyExec(FTS))`     | `Some(child)` | `Some(Union -> Repartition(RoundRobinBatch(1)))`    |
///
/// Errors only on internal invariants (HashJoin construction, Schema
/// lookups). Returns `Result<Option<Arc<dyn ExecutionPlan>>>` so the
/// `Must` slot's `None` case is naturally expressible.
pub fn build_boolean_query_children(
    slot: BoolSlot,
    children: Vec<Arc<dyn ExecutionPlan>>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    build_boolean_query_children_with_schema(slot, children, FTS_SCHEMA.clone())
}

pub fn build_boolean_query_children_with_schema(
    slot: BoolSlot,
    mut children: Vec<Arc<dyn ExecutionPlan>>,
    schema: SchemaRef,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    match slot {
        BoolSlot::Should | BoolSlot::MustNot => {
            if children.is_empty() {
                Ok(Some(Arc::new(EmptyExec::new(schema))))
            } else if children.len() == 1 {
                Ok(Some(children.pop().unwrap()))
            } else {
                let unioned = UnionExec::try_new(children)?;
                Ok(Some(Arc::new(RepartitionExec::try_new(
                    unioned,
                    Partitioning::RoundRobinBatch(1),
                )?)))
            }
        }
        BoolSlot::Must => {
            let mut joined: Option<Arc<dyn ExecutionPlan>> = None;
            for plan in children {
                if let Some(left) = joined {
                    let mut on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = vec![(
                        Arc::new(Column::new_with_schema(ROW_ID, &schema)?),
                        Arc::new(Column::new_with_schema(ROW_ID, &schema)?),
                    )];
                    if schema.field_with_name(DOC_INDEX_COL).is_ok() {
                        on.push((
                            Arc::new(Column::new_with_schema(DOC_INDEX_COL, &schema)?),
                            Arc::new(Column::new_with_schema(DOC_INDEX_COL, &schema)?),
                        ));
                    }
                    joined = Some(Arc::new(HashJoinExec::try_new(
                        left,
                        plan,
                        on,
                        None,
                        &datafusion_expr::JoinType::Inner,
                        None,
                        PartitionMode::CollectLeft,
                        NullEquality::NullEqualsNothing,
                        false,
                    )?) as _);
                } else {
                    joined = Some(plan);
                }
            }
            Ok(joined)
        }
    }
}

#[derive(Debug)]
pub struct BooleanQueryExec {
    query: BooleanQuery,
    params: FtsSearchParams,
    should: Arc<dyn ExecutionPlan>,
    must: Option<Arc<dyn ExecutionPlan>>,
    must_not: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BooleanQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BooleanQuery: should={:?}, must={:?}, must_not={:?}",
                    self.query.should, self.query.must, self.query.must_not,
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "BooleanQuery")?;
                if !self.query.should.is_empty() {
                    write!(f, "\nshould={:?}", self.query.should)?;
                }
                if !self.query.must.is_empty() {
                    write!(f, "\nmust={:?}", self.query.must)?;
                }
                if !self.query.must_not.is_empty() {
                    write!(f, "\nmust_not={:?}", self.query.must_not)?;
                }
                std::fmt::Result::Ok(())
            }
        }
    }
}

impl BooleanQueryExec {
    pub fn new(
        query: BooleanQuery,
        params: FtsSearchParams,
        should: Arc<dyn ExecutionPlan>,
        must: Option<Arc<dyn ExecutionPlan>>,
        must_not: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let schema = should.schema();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            must,
            should,
            must_not,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn query(&self) -> &BooleanQuery {
        &self.query
    }

    pub fn params(&self) -> &FtsSearchParams {
        &self.params
    }

    pub fn should(&self) -> &Arc<dyn ExecutionPlan> {
        &self.should
    }

    pub fn must(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.must.as_ref()
    }

    pub fn must_not(&self) -> &Arc<dyn ExecutionPlan> {
        &self.must_not
    }
}

impl ExecutionPlan for BooleanQueryExec {
    fn name(&self) -> &str {
        "BooleanQueryExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.must {
            Some(must) => vec![&self.should, &self.must_not, must],
            None => vec![&self.should, &self.must_not],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.
        // It must be run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => {
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not: self.must_not.clone(),
                    schema: self.schema.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            2 => {
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not,
                    schema: self.schema.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            3 => {
                let must = children.pop().unwrap();
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: Some(must),
                    must_not,
                    schema: self.schema.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            _ => Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            )),
        }
    }

    #[instrument(name = "bool_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let params = self.params.clone();
        let should_plan = self.should.clone();
        let must_plan = self.must.clone();
        let must_not_plan = self.must_not.clone();
        let must = self
            .must
            .as_ref()
            .map(|m| m.execute(partition, context.clone()))
            .transpose()?;
        let mut should = self.should.execute(partition, context.clone())?;
        let mut must_not = self.must_not.execute(partition, context)?;
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let schema = self.schema.clone();

        let stream = stream::once(async move {
            let elapsed_time = metrics.baseline_metrics.elapsed_compute();

            let mut res = HashMap::new();
            let has_must = must.is_some();
            if let Some(mut must) = must {
                while let Some(batch) = must.try_next().await? {
                    let _timer = elapsed_time.timer();
                    res.extend(batch_scored_document_keys_sum_scores(&batch)?);
                }
            }

            // add the scores from the should clause
            while let Some(batch) = should.try_next().await? {
                let _timer = elapsed_time.timer();
                for (key, score) in batch_scored_document_keys(&batch)? {
                    let entry = res.entry(key).and_modify(|value| *value += score);
                    if !has_must {
                        entry.or_insert(score);
                    }
                }
            }

            // remove the results from the must_not clause
            while let Some(batch) = must_not.try_next().await? {
                let _timer = elapsed_time.timer();
                for key in batch_document_keys(&batch)? {
                    res.remove(&key);
                }
            }

            let mut partitions_searched = 0;
            for plan in [Some(&should_plan), must_plan.as_ref(), Some(&must_not_plan)] {
                let Some(plan) = plan else {
                    continue;
                };
                let Some(metrics) = plan.metrics() else {
                    continue;
                };
                for (metric_name, count) in metrics.iter_counts() {
                    if metric_name.as_ref() == PARTITIONS_SEARCHED_METRIC {
                        partitions_searched += count.value();
                    }
                }
            }
            metrics.record_parts_searched(partitions_searched);

            // sort the results and take the top k
            let _timer = elapsed_time.timer();
            let documents = res
                .into_iter()
                .sorted_unstable_by(compare_scored_documents)
                .take(params.limit.unwrap_or(usize::MAX))
                .collect::<Vec<_>>();
            metrics.baseline_metrics.record_output(documents.len());
            let batch = document_key_scores_batch(schema, documents)?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}
