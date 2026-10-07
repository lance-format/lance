// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Applying a [`Plan`] to a source's batches.

use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

use crate::dataset::mem_wal::reconcile::Plan;

/// Brings one source's batches to the schema the scan reads in.
///
/// The same [`Plan`] replay applies to a WAL entry, so a generation and an
/// entry written under the same schema are reconciled the same way.
pub struct ReconcileExec {
    input: Arc<dyn ExecutionPlan>,
    plan: Arc<Plan>,
    properties: Arc<PlanProperties>,
}

impl ReconcileExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, plan: Arc<Plan>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(plan.target())),
            input.properties().output_partitioning().clone(),
            input.properties().emission_type,
            input.properties().boundedness,
        ));
        Self {
            input,
            plan,
            properties,
        }
    }
}

impl fmt::Debug for ReconcileExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReconcileExec")
    }
}

impl DisplayAs for ReconcileExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ReconcileExec")
    }
}

impl ExecutionPlan for ReconcileExec {
    fn name(&self) -> &str {
        "ReconcileExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(self.plan.target())
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
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            children
                .into_iter()
                .next()
                .ok_or_else(|| DataFusionError::Internal("ReconcileExec needs one child".into()))?,
            Arc::clone(&self.plan),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        Ok(Box::pin(ReconcileStream {
            input: self.input.execute(partition, context)?,
            plan: Arc::clone(&self.plan),
        }))
    }
}

struct ReconcileStream {
    input: SendableRecordBatchStream,
    plan: Arc<Plan>,
}

impl Stream for ReconcileStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(
                self.plan
                    .apply(&batch)
                    .map_err(|e| DataFusionError::External(Box::new(e))),
            )),
            other => other,
        }
    }
}

impl RecordBatchStream for ReconcileStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(self.plan.target())
    }
}
