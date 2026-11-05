// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::Formatter;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::stream;

/// Replays one materialized batch while preserving its batch boundary.
///
/// Some inputs are control messages: the two rows of a serialized index result
/// must reach the reader together, even when the query's data batch size is one.
/// Each execution gets a fresh stream sharing the immutable Arrow buffers.
/// Encoding can inspect [`Self::batch`] without executing or consuming a stream.
///
/// ```
/// use arrow_array::record_batch;
/// use lance_datafusion_exec::exec::RecordBatchExec;
/// let batch = record_batch!(("part", Int32, [0, 1])).unwrap();
/// let input = RecordBatchExec::new(batch);
/// let reconstructed = RecordBatchExec::new(input.batch().clone());
/// assert_eq!(input.batch(), reconstructed.batch());
/// ```
#[derive(Debug)]
pub struct RecordBatchExec {
    batch: RecordBatch,
    properties: Arc<PlanProperties>,
}

impl RecordBatchExec {
    /// Retain this already materialized batch; no data is read or collected.
    pub fn new(batch: RecordBatch) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(batch.schema()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self { batch, properties }
    }

    /// The complete literal payload, including its schema and batch boundary.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
}

impl DisplayAs for RecordBatchExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RecordBatchExec: rows={}, columns={}",
            self.batch.num_rows(),
            self.batch.num_columns()
        )
    }
}

impl ExecutionPlan for RecordBatchExec {
    fn name(&self) -> &str {
        "RecordBatchExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "RecordBatchExec expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "RecordBatchExec expects partition 0, got {partition}"
            )));
        }
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.batch.schema(),
            stream::iter([Ok(self.batch.clone())]),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{LanceExecutionOptions, new_task_context};
    use arrow_array::record_batch;
    use datafusion_physical_plan::collect;
    use futures::TryStreamExt;
    use rstest::rstest;

    #[rstest]
    #[case::empty(0)]
    #[case::two_row_frame(2)]
    #[tokio::test]
    async fn record_batch_exec_replays_atomic_batches(#[case] rows: i32) {
        let batch = record_batch!(("part", Int32, (0..rows).collect::<Vec<_>>())).unwrap();
        let exec = Arc::new(RecordBatchExec::new(batch.clone()));
        let context = new_task_context(&LanceExecutionOptions {
            batch_size: Some(1),
            ..Default::default()
        })
        .unwrap();
        // Dropping one execution must not consume the next execution's payload.
        drop(exec.execute(0, context.clone()).unwrap());
        for _ in 0..2 {
            let actual = exec
                .execute(0, context.clone())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            assert_eq!(actual, vec![batch.clone()]);
        }
        let (left, right) = futures::try_join!(
            collect(exec.clone(), context.clone()),
            collect(exec.clone(), context.clone())
        )
        .unwrap();
        assert_eq!(left, vec![batch.clone()]);
        assert_eq!(right, vec![batch]);
        let Err(error) = exec.execute(1, context) else {
            panic!("invalid partition accepted")
        };
        assert!(matches!(error, DataFusionError::Internal(_)));
        assert!(error.to_string().contains("partition 0, got 1"));
    }
}
