// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Final-hit highlighting. The child owns retrieval and snapshot-consistent source reads.

use std::sync::Arc;

use arrow_array::{RecordBatch, cast::AsArray, types::UInt32Type};
use arrow_schema::{Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use futures::TryStreamExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_index::scalar::inverted::DOC_INDEX_COL;
use lance_index::scalar::inverted::highlight::{
    HighlightDocument, PreparedHighlightQuery, highlights_array, highlights_field,
};

use crate::index::scalar::inverted::ResolvedFtsField;

/// Appends `_highlights` to already selected, materialized search results.
#[derive(Debug)]
pub struct HighlightExec {
    input: Arc<dyn ExecutionPlan>,
    query: Arc<PreparedHighlightQuery>,
    sources: Arc<Vec<ResolvedFtsField>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl HighlightExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        query: Arc<PreparedHighlightQuery>,
        sources: Arc<Vec<ResolvedFtsField>>,
    ) -> Self {
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(highlights_field()));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema().metadata().clone(),
        ));
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema.clone())),
        );
        Self {
            input,
            query,
            sources,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for HighlightExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "HighlightExec: fields={}",
            self.sources
                .iter()
                .map(|source| source.canonical_path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl ExecutionPlan for HighlightExec {
    fn name(&self) -> &str {
        "HighlightExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "HighlightExec requires one child".into(),
            ));
        }
        let input = children
            .pop()
            .ok_or_else(|| DataFusionError::Internal("HighlightExec child is missing".into()))?;
        Ok(Arc::new(Self::new(
            input,
            self.query.clone(),
            self.sources.clone(),
        )))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let query = self.query.clone();
        let sources = self.sources.clone();
        let schema = self.schema.clone();
        let output_schema = schema.clone();
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let memory_pool = context.memory_pool().clone();
        let stream = self
            .input
            .execute(partition, context)?
            .and_then(move |batch| {
                let (query, sources, schema, metrics) = (
                    query.clone(),
                    sources.clone(),
                    schema.clone(),
                    metrics.clone(),
                );
                let memory_pool = memory_pool.clone();
                async move {
                    // Account for the materialized batch and retain its reservation while
                    // CPU work is in flight, including when the consumer drops the stream.
                    let reservation =
                        Arc::new(MemoryConsumer::new("HighlightExec").register(&memory_pool));
                    reservation.try_grow(batch.get_array_memory_size())?;
                    let in_flight = reservation.clone();
                    let active = Arc::new(());
                    let is_active = Arc::downgrade(&active);
                    let compute_metrics = metrics.clone();
                    let result = spawn_cpu(move || -> lance_core::Result<RecordBatch> {
                        let _reservation = in_flight;
                        let _timer = compute_metrics.elapsed_compute().timer();
                        let mut rows = Vec::with_capacity(batch.num_rows());
                        for row in 0..batch.num_rows() {
                            if is_active.strong_count() == 0 {
                                return Err(lance_core::Error::execution(
                                    "highlight execution cancelled",
                                ));
                            }
                            let coordinate = batch
                                .column_by_name(DOC_INDEX_COL)
                                .map(|array| array.as_list::<i32>().value(row));
                            let doc_index = coordinate
                                .as_ref()
                                .map(|array| array.as_primitive::<UInt32Type>().values().as_ref())
                                .unwrap_or(&[]);
                            let documents = sources
                                .iter()
                                .map(|source| {
                                    Ok(HighlightDocument {
                                        field: source.canonical_path.clone(),
                                        values: source.highlight_values(&batch, row, doc_index)?,
                                    })
                                })
                                .collect::<lance_core::Result<Vec<_>>>()?;
                            rows.push(query.highlight(&documents).map_err(|error| {
                                let message = format!("highlight batch row {row}: {error}");
                                match error {
                                    lance_core::Error::InvalidInput { .. } => {
                                        lance_core::Error::invalid_input(message)
                                    }
                                    _ => lance_core::Error::internal(message),
                                }
                            })?);
                        }
                        let mut columns = batch.columns().to_vec();
                        columns.push(highlights_array(&rows)?);
                        Ok(RecordBatch::try_new(schema, columns)?)
                    })
                    .await
                    .map_err(DataFusionError::from)?;
                    drop(active);
                    reservation.try_grow(
                        result
                            .column(result.num_columns() - 1)
                            .get_array_memory_size(),
                    )?;
                    metrics.record_output(result.num_rows());
                    Ok(result)
                }
            });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::scalar::inverted::resolve_fts_field;
    use crate::io::exec::fts::default_text_tokenizer;
    use arrow_array::record_batch;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::highlight::HighlightLeaf;
    use lance_index::scalar::inverted::query::{
        FtsSearchParams, MatchQuery, Operator, try_collect_query_tokens,
    };

    fn plan(batch: RecordBatch) -> HighlightExec {
        let source = resolve_fts_field(
            &lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap(),
            "body",
            DocumentGranularity::Row,
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone(), batch.clone()]],
            batch.schema(),
            None,
        )
        .unwrap();
        let mut tokenizer = default_text_tokenizer();
        let tokens = try_collect_query_tokens("rust", &mut tokenizer).unwrap();
        let leaf = HighlightLeaf::try_new(
            vec!["body".into()],
            tokenizer.into(),
            tokens,
            Operator::Or,
            None,
            &FtsSearchParams::new().with_fuzziness(Some(0)),
        )
        .unwrap();
        let query = PreparedHighlightQuery::new(
            MatchQuery::new("rust".into())
                .with_column(Some("body".into()))
                .into(),
            vec![leaf],
        );
        HighlightExec::new(input, Arc::new(query), Arc::new(vec![source]))
    }

    #[tokio::test]
    async fn rejects_unadmitted_batches_and_releases_reservations() {
        let batch = record_batch!(("body", Utf8, ["rust"])).unwrap();
        // One limit rejects the input, the other rejects the additional result array.
        for limit in [0, batch.get_array_memory_size()] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(pool.clone())
                .build_arc()
                .unwrap();
            let context = Arc::new(TaskContext::default().with_runtime(runtime));
            let exec = plan(batch.clone());
            let error = exec
                .execute(0, context)
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap_err();
            assert!(
                matches!(error, DataFusionError::ResourcesExhausted(_)),
                "{error}"
            );
            assert_eq!(pool.reserved(), 0);
        }
    }

    #[tokio::test]
    async fn dropping_stream_does_not_analyze_subsequent_batches() {
        let exec = plan(record_batch!(("body", Utf8, ["rust"])).unwrap());
        let mut stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        assert_eq!(stream.try_next().await.unwrap().unwrap().num_rows(), 1);
        drop(stream);
        assert_eq!(exec.metrics().unwrap().output_rows(), Some(1));
    }
}
