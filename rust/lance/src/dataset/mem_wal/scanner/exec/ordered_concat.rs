// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! OrderedConcatExec - Merges a plan's partitions into one, in partition order.

use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::common::spawn_buffered;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::StreamExt;

/// Batches each partition may read ahead of the one being emitted.
const READ_AHEAD_BATCHES: usize = 2;

/// Emits every batch of partition 0, then partition 1, and so on, as a single
/// partition.
///
/// Unlike `CoalescePartitionsExec`, which interleaves partitions in arrival
/// order, the output order is fixed by the input's, so a limit / offset above
/// it selects the same rows on every run. All partitions start at once and
/// read ahead into a bounded buffer, so they still scan concurrently.
#[derive(Debug)]
pub struct OrderedConcatExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl OrderedConcatExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self { input, properties }
    }
}

impl DisplayAs for OrderedConcatExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => write!(f, "OrderedConcatExec"),
        }
    }
}

impl ExecutionPlan for OrderedConcatExec {
    fn name(&self) -> &str {
        "OrderedConcatExec"
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "OrderedConcatExec takes one child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "OrderedConcatExec has one partition, asked for {partition}"
            )));
        }
        let streams = (0..self.input.output_partitioning().partition_count())
            .map(|p| {
                self.input
                    .execute(p, context.clone())
                    .map(|s| spawn_buffered(s, READ_AHEAD_BATCHES))
            })
            .collect::<DFResult<Vec<_>>>()?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::iter(streams).flatten(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::collect;
    use datafusion::physical_plan::union::UnionExec;

    fn source(values: &[&[i32]]) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batches = values
            .iter()
            .map(|v| {
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(v.to_vec()))])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        MemorySourceConfig::try_new_exec(&[batches], schema, None).unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emits_partitions_in_order() {
        let union = UnionExec::try_new(vec![
            source(&[&[1, 2], &[3]]),
            source(&[&[4]]),
            source(&[&[5, 6], &[7]]),
        ])
        .unwrap();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(OrderedConcatExec::new(union));
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        for _ in 0..20 {
            let batches = collect(plan.clone(), Arc::new(TaskContext::default()))
                .await
                .unwrap();
            let values: Vec<i32> = batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7]);
        }
    }
}
