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
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_schema::SchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

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
        let schema = self.schema();
        // Resolve the PK columns once per stream rather than once per batch:
        // every batch carries the plan's schema, so positions and types are
        // fixed for the life of the stream.
        let mut pk_indices = Vec::with_capacity(self.pk_columns.len());
        let mut sort_fields = Vec::with_capacity(self.pk_columns.len());
        for col in &self.pk_columns {
            let (idx, field) = schema.column_with_name(col).ok_or_else(|| {
                DataFusionError::Internal(format!("Primary key column '{col}' not found"))
            })?;
            pk_indices.push(idx);
            sort_fields.push(SortField::new(field.data_type().clone()));
        }

        Ok(Box::pin(FirstByPkStream {
            input: self.input.execute(partition, context)?,
            converter: RowConverter::new(sort_fields)?,
            pk_indices,
            schema,
            seen: HashSet::new(),
        }))
    }
}

struct FirstByPkStream {
    input: SendableRecordBatchStream,
    /// Positions of the primary-key columns within the input schema.
    pk_indices: Vec<usize>,
    schema: SchemaRef,
    /// Encodes a row's primary key into one comparable byte string. Column-at-a-time
    /// and exact, so the dedup key costs a single allocation and a `memcmp` rather
    /// than a boxed `ScalarValue` per key column.
    converter: RowConverter,
    seen: HashSet<OwnedRow>,
}

impl FirstByPkStream {
    /// Mask out rows whose PK was already emitted. Sequential by construction:
    /// the first occurrence in input order wins, so the mask depends on every
    /// row before it.
    fn keep_first(&mut self, batch: &RecordBatch) -> DFResult<RecordBatch> {
        if self.pk_indices.is_empty() || batch.num_rows() == 0 {
            return Ok(batch.clone());
        }
        let pk_columns = self
            .pk_indices
            .iter()
            .map(|&idx| batch.column(idx).clone())
            .collect::<Vec<ArrayRef>>();
        let rows = self.converter.convert_columns(&pk_columns)?;
        let mut keep = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            keep.push(self.seen.insert(rows.row(row).owned()));
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    /// Run the exec over a single partition holding `batches` and return the
    /// `tag` of every surviving row, in output order. `tag` identifies which
    /// occurrence of a duplicated key was kept.
    async fn kept_tags(schema: SchemaRef, batches: Vec<RecordBatch>, pk: &[&str]) -> Vec<i32> {
        let input = TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap();
        let exec = FirstByPkExec::new(input, pk.iter().map(|c| c.to_string()).collect());
        let stream = exec.execute(0, SessionContext::new().task_ctx()).unwrap();
        let out: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        out.iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("tag")
                    .expect("tag is projected")
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("tag is Int32")
                    .values()
                    .to_vec()
            })
            .collect()
    }

    fn string_pk_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("pk", DataType::Utf8, true),
            Field::new("tag", DataType::Int32, false),
        ]))
    }

    fn string_pk_batch(schema: &SchemaRef, pks: Vec<Option<&str>>, tags: &[i32]) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(pks)),
                Arc::new(Int32Array::from(tags.to_vec())),
            ],
        )
        .unwrap()
    }

    /// `seen` spans the whole stream, so a key repeated in a later batch is
    /// dropped even though nothing in that batch repeats.
    #[tokio::test]
    async fn duplicates_are_dropped_across_batch_boundaries() {
        let schema = string_pk_schema();
        let batches = vec![
            string_pk_batch(&schema, vec![Some("a"), Some("b")], &[1, 2]),
            string_pk_batch(&schema, vec![Some("a"), Some("c")], &[3, 4]),
        ];

        assert_eq!(
            kept_tags(schema, batches, &["pk"]).await,
            vec![1, 2, 4],
            "the second 'a' must be dropped and the first kept"
        );
    }

    /// A composite key collapses only on the whole tuple — sharing one component
    /// is not a duplicate.
    #[tokio::test]
    async fn composite_keys_collapse_on_the_whole_tuple() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("pk_a", DataType::Utf8, true),
            Field::new("pk_b", DataType::Int32, true),
            Field::new("tag", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("x"), Some("x"), Some("x")])),
                Arc::new(Int32Array::from(vec![1, 2, 1])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();

        assert_eq!(
            kept_tags(schema, vec![batch], &["pk_a", "pk_b"]).await,
            vec![1, 2],
            "('x', 2) is a distinct key; the repeated ('x', 1) is not"
        );
    }

    /// Null keys collapse with each other and stay distinct from the empty
    /// string, which the row encoding separates by a leading sentinel.
    #[tokio::test]
    async fn null_keys_collapse_and_stay_distinct_from_empty() {
        let schema = string_pk_schema();
        let batch = string_pk_batch(
            &schema,
            vec![None, Some(""), None, Some("a")],
            &[1, 2, 3, 4],
        );

        assert_eq!(
            kept_tags(schema, vec![batch], &["pk"]).await,
            vec![1, 2, 4],
            "the repeated null drops; the empty string is its own key"
        );
    }
}
