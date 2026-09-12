// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep the first row per primary key from an ordered stream.
//!
//! Collapses the duplicates a cross-column full-text search produces: a row
//! whose text matches in two columns is scored once per column, so it arrives
//! twice. A cross-column `MultiMatch` scores each field independently and takes
//! the **maximum** per row (`DisjunctionScore::Max` on the base-table path), and
//! over an input already sorted by `_score` descending, "first wins" *is* that
//! maximum — which is why this is a filter rather than a grouped aggregate, and
//! why it keeps the streaming top-k shape the FTS planner is built around.
//!
//! The input ordering is load-bearing. Feeding this an unordered stream keeps an
//! arbitrary duplicate instead of the best-scoring one.

use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::compute::filter_record_batch;
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

use super::pk::resolve_pk_indices;

/// Emits the first row seen for each primary key, preserving input order.
#[derive(Debug)]
pub struct FirstByPkExec {
    input: Arc<dyn ExecutionPlan>,
    pk_columns: Vec<String>,
    properties: Arc<PlanProperties>,
}

impl FirstByPkExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, pk_columns: Vec<String>) -> Self {
        // A filter: same schema, same partitioning, and the input's ordering
        // survives because rows are only dropped.
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Self {
            input,
            pk_columns,
            properties,
        }
    }
}

impl DisplayAs for FirstByPkExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "FirstByPk: pk={:?}", self.pk_columns)
            }
            DisplayFormatType::TreeRender => write!(f, "FirstByPk"),
        }
    }
}

impl ExecutionPlan for FirstByPkExec {
    fn name(&self) -> &str {
        "FirstByPkExec"
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
        let input = children.into_iter().next().ok_or_else(|| {
            DataFusionError::Internal("FirstByPkExec requires one child".to_string())
        })?;
        Ok(Arc::new(Self::new(input, self.pk_columns.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        Ok(Box::pin(FirstByPkStream {
            input: self.input.execute(partition, context)?,
            pk_columns: self.pk_columns.clone(),
            schema: self.schema(),
            seen: HashSet::new(),
        }))
    }
}

struct FirstByPkStream {
    input: SendableRecordBatchStream,
    pk_columns: Vec<String>,
    schema: SchemaRef,
    seen: HashSet<Vec<ScalarValue>>,
}

impl FirstByPkStream {
    /// Mask out rows whose PK was already emitted. Sequential by construction:
    /// the first occurrence in input order wins, so the mask depends on every
    /// row before it.
    fn keep_first(&mut self, batch: &RecordBatch) -> DFResult<RecordBatch> {
        if self.pk_columns.is_empty() || batch.num_rows() == 0 {
            return Ok(batch.clone());
        }
        let pk_indices = resolve_pk_indices(batch, &self.pk_columns)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        let mut keep = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let key = pk_indices
                .iter()
                .map(|&col| ScalarValue::try_from_array(batch.column(col), row))
                .collect::<DFResult<Vec<_>>>()?;
            keep.push(self.seen.insert(key));
        }
        filter_record_batch(batch, &BooleanArray::from(keep)).map_err(DataFusionError::from)
    }
}

impl Stream for FirstByPkStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match futures::ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    let filtered = self.keep_first(&batch)?;
                    // An all-duplicate batch yields nothing; keep polling rather
                    // than emitting empties.
                    if filtered.num_rows() > 0 {
                        return Poll::Ready(Some(Ok(filtered)));
                    }
                }
                other => return Poll::Ready(other),
            }
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for FirstByPkStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
