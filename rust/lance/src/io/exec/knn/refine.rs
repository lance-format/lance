// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read shared candidates once, then rank only each query's own candidates.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{Array, Float32Array, Int32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, SchemaRef};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties};
use futures::{StreamExt, stream};
use lance_core::ROW_ID;
use lance_core::utils::tokio::spawn_cpu;
use lance_datafusion::utils::ExecutionPlanMetricsSetExt;
use lance_index::vector::Query;

use crate::dataset::{Dataset, ProjectionRequest};
use crate::index::vector::utils::get_vector_type;
use crate::io::exec::utils::InstrumentedRecordBatchStreamAdapter;
use crate::{Error, Result};

use super::{
    BatchKnnCandidate, BatchKnnExtra, KNNVectorDistanceExec, QUERY_INDEX_COL,
    knn_empty_result_schema, would_enter_heap,
};

/// Limit materialized vectors independently of batch width and refine factor.
/// Scoring gathers at most another chunk's worth of vectors for one query at a
/// time. A single vector wider than the budget is still read on its own.
const VECTOR_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
pub struct BatchRefineExec {
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    query: Query,
    query_count: usize,
    rows_per_take: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl BatchRefineExec {
    pub(crate) fn try_new(
        dataset: Arc<Dataset>,
        input: Arc<dyn ExecutionPlan>,
        query: Query,
        query_count: usize,
    ) -> Result<Self> {
        if query_count == 0 || query.k == 0 || !query.key.len().is_multiple_of(query_count) {
            return Err(Error::invalid_input(format!(
                "Batch refinement requires positive k and a key divisible by query_count: k={}, key_length={}, query_count={query_count}",
                query.k,
                query.key.len()
            )));
        }
        let (vector_type, element_type) = get_vector_type(dataset.schema(), &query.column)?;
        let DataType::FixedSizeList(_, dim) = vector_type else {
            return Err(Error::invalid_input(format!(
                "Batch refinement requires fixed-size vectors, got {vector_type} for '{}'",
                query.column
            )));
        };
        let width = element_type.primitive_width().ok_or_else(|| {
            Error::invalid_input(format!(
                "Batch refinement requires primitive vector elements, got {element_type}"
            ))
        })?;
        let row_bytes = (dim as usize)
            .checked_mul(width)
            .and_then(|bytes| bytes.checked_add(1))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vector byte size overflows: dimension={dim}, element_width={width}"
                ))
            })?;
        if dim <= 0 || query.key.len() / query_count != dim as usize || query.metric_type.is_none()
        {
            return Err(Error::invalid_input(format!(
                "Invalid batch refinement query: dimension={dim}, key_length={}, query_count={query_count}, metric={:?}",
                query.key.len(),
                query.metric_type
            )));
        }
        for (name, expected) in [
            (ROW_ID, DataType::UInt64),
            (QUERY_INDEX_COL, DataType::Int32),
        ] {
            let schema = input.schema();
            let field = schema.field_with_name(name)?;
            if field.data_type() != &expected {
                return Err(Error::invalid_input(format!(
                    "Batch refinement requires {name}: {expected}, got {}",
                    field.data_type()
                )));
            }
        }
        Ok(Self {
            dataset,
            input,
            query,
            query_count,
            rows_per_take: (VECTOR_BATCH_BYTES / row_bytes).max(1),
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(knn_empty_result_schema(true)),
                Partitioning::RoundRobinBatch(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    async fn refine(
        self: Arc<Self>,
        mut input: SendableRecordBatchStream,
        vectors_read: Count,
        read_batches: Count,
        peak_vector_bytes: Count,
    ) -> Result<RecordBatch> {
        let mut heaps = (0..self.query_count)
            .map(|_| BinaryHeap::<BatchKnnCandidate>::new())
            .collect::<Vec<_>>();
        let projection = ProjectionRequest::from_columns(
            [self.query.column.as_str(), ROW_ID],
            self.dataset.schema(),
        );
        let mut peak_bytes = 0;
        while let Some(batch) = input.next().await {
            let batch = batch?;
            let query_count = self.query_count;
            let candidates = spawn_cpu(move || -> Result<Vec<(u64, usize)>> {
                let ids = batch
                    .column_by_name(ROW_ID)
                    .ok_or_else(|| Error::internal("Batch refinement input missing _rowid"))?
                    .as_primitive::<UInt64Type>();
                let queries = batch
                    .column_by_name(QUERY_INDEX_COL)
                    .ok_or_else(|| Error::internal("Batch refinement input missing query_index"))?
                    .as_primitive::<Int32Type>();
                let mut candidates = Vec::with_capacity(batch.num_rows());
                for (id, query) in ids.iter().zip(queries.iter()) {
                    let (Some(id), Some(query)) = (id, query) else {
                        return Err(Error::internal(
                            "Batch refinement input has NULL candidate identity",
                        ));
                    };
                    if query < 0 || query as usize >= query_count {
                        return Err(Error::internal(format!(
                            "Candidate query_index={query} is outside query_count={query_count}"
                        )));
                    }
                    candidates.push((id, query as usize));
                }
                candidates.sort_unstable();
                Ok(candidates)
            })
            .await?;
            let mut offset = 0;
            while offset < candidates.len() {
                let start = offset;
                let mut row_ids =
                    Vec::with_capacity(self.rows_per_take.min(candidates.len() - offset));
                while offset < candidates.len() {
                    let row_id = candidates[offset].0;
                    if row_ids.last() != Some(&row_id) {
                        if row_ids.len() == self.rows_per_take {
                            break;
                        }
                        row_ids.push(row_id);
                    }
                    offset += 1;
                }
                let batch = self.dataset.take_rows(&row_ids, projection.clone()).await?;
                vectors_read.add(batch.num_rows());
                read_batches.add(1);
                let vectors =
                    KNNVectorDistanceExec::resolve_vector_column(&batch, &self.query.column)?;
                let bytes = vectors.get_array_memory_size();
                if bytes > peak_bytes {
                    peak_vector_bytes.add(bytes - peak_bytes);
                    peak_bytes = bytes;
                }
                // The ANN prefilter excludes deleted rows. Stable-ID lookup
                // can also omit missing IDs: join by returned identity so an
                // omitted row never shifts a candidate onto another vector.
                let returned_ids = batch
                    .column_by_name(ROW_ID)
                    .ok_or_else(|| Error::internal("Batch refinement take missing _rowid"))?
                    .as_primitive::<UInt64Type>();
                let positions: HashMap<u64, u32> = returned_ids
                    .values()
                    .iter()
                    .enumerate()
                    .map(|(index, id)| (*id, index as u32))
                    .collect();
                let mut groups = vec![Vec::new(); self.query_count];
                for &(id, query_index) in &candidates[start..offset] {
                    if let Some(&position) = positions.get(&id) {
                        groups[query_index].push(position);
                    }
                }
                let query = self.query.clone();
                let query_count = self.query_count;
                let returned_ids = returned_ids.clone();
                let rows_per_take = self.rows_per_take;
                heaps = spawn_cpu(move || -> Result<_> {
                    let metric = query
                        .metric_type
                        .ok_or_else(|| Error::internal("Batch refinement metric is missing"))?;
                    let dim = query.key.len() / query_count;
                    for (query_index, indices) in groups.into_iter().enumerate() {
                        if indices.is_empty() {
                            continue;
                        }
                        // Do not score the union against every query: only this
                        // query's original ANN candidates are eligible for top-k.
                        // Repeated candidates within one query must not expand
                        // the scoring buffer beyond the vector read budget.
                        for indices in indices.chunks(rows_per_take) {
                            let indices = UInt32Array::from(indices.to_vec());
                            let selected =
                                arrow_select::take::take(vectors.as_ref(), &indices, None)?;
                            let key = query.key.slice(query_index * dim, dim);
                            let distances = metric.arrow_batch_func()(
                                key.as_ref(),
                                selected.as_fixed_size_list(),
                            )?;
                            let heap = &mut heaps[query_index];
                            for (position, distance) in distances.iter().enumerate() {
                                let Some(distance) = distance else { continue };
                                if distance.is_nan()
                                    || query.lower_bound.is_some_and(|bound| distance < bound)
                                    || query.upper_bound.is_some_and(|bound| distance >= bound)
                                {
                                    continue;
                                }
                                let row_id = returned_ids.value(indices.value(position) as usize);
                                if would_enter_heap(
                                    heap,
                                    query.k,
                                    distance,
                                    row_id,
                                    query_index as i32,
                                ) {
                                    if heap.len() == query.k {
                                        heap.pop();
                                    }
                                    heap.push(BatchKnnCandidate {
                                        query_index: query_index as i32,
                                        distance,
                                        row_id,
                                        extra: BatchKnnExtra::RowIdOnly,
                                    });
                                }
                            }
                        }
                    }
                    Ok(heaps)
                })
                .await?;
            }
        }
        let mut results = heaps
            .into_iter()
            .flat_map(BinaryHeap::into_vec)
            .collect::<Vec<_>>();
        results.sort_unstable_by(|left, right| {
            left.query_index
                .cmp(&right.query_index)
                .then_with(|| left.cmp(right))
        });
        Ok(RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(Int32Array::from(
                    results
                        .iter()
                        .map(|candidate| candidate.query_index)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from(
                    results
                        .iter()
                        .map(|candidate| candidate.distance)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(UInt64Array::from(
                    results
                        .iter()
                        .map(|candidate| candidate.row_id)
                        .collect::<Vec<_>>(),
                )),
            ],
        )?)
    }
}

impl DisplayAs for BatchRefineExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "BatchRefine: query_count={}, k={}, rows_per_take={}",
            self.query_count, self.query.k, self.rows_per_take
        )
    }
}

impl ExecutionPlan for BatchRefineExec {
    fn name(&self) -> &str {
        "BatchRefineExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn schema(&self) -> SchemaRef {
        knn_empty_result_schema(true)
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "BatchRefineExec requires one child, got {}",
                children.len()
            )));
        }
        let input = children.remove(0);
        let mut node = Self::try_new(
            self.dataset.clone(),
            input,
            self.query.clone(),
            self.query_count,
        )?;
        node.rows_per_take = self.rows_per_take;
        Ok(Arc::new(node))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let node = Arc::new(Self {
            dataset: self.dataset.clone(),
            input: self.input.clone(),
            query: self.query.clone(),
            query_count: self.query_count,
            rows_per_take: self.rows_per_take,
            properties: self.properties.clone(),
            metrics: self.metrics.clone(),
        });
        let vectors_read = self.metrics.new_count("refine_vectors_read", partition);
        let read_batches = self.metrics.new_count("refine_read_batches", partition);
        let peak_vector_bytes = self
            .metrics
            .new_count("refine_peak_vector_bytes", partition);
        let result = async move {
            node.refine(input, vectors_read, read_batches, peak_vector_bytes)
                .await
                .map_err(DataFusionError::from)
        };
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            stream::once(result).boxed(),
            partition,
            &self.metrics,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::record_batch;
    use arrow_array::types::Float32Type;
    use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount, array, gen_batch};
    use lance_linalg::distance::DistanceType;
    use rstest::rstest;

    use crate::dataset::WriteParams;
    use crate::io::exec::testing::TestingExec;

    #[rstest]
    #[tokio::test]
    async fn test_batch_refine_shared_candidates(
        #[values(1, 100)] rows_per_take: usize,
        #[values(false, true)] stable_row_ids: bool,
        #[values("normal", "deleted", "bounded", "all_null", "empty")] scenario: &str,
    ) {
        let nulls = if scenario == "all_null" {
            vec![true; 8]
        } else {
            vec![false, false, false, false, false, false, false, true]
        };
        let data = gen_batch()
            .col(
                "vec",
                array::cycle_vec(
                    array::cycle::<Float32Type>(vec![
                        0.0,
                        2.0,
                        3.0,
                        10.0,
                        15.0,
                        f32::NAN,
                        30.0,
                        0.0,
                    ]),
                    Dimension::from(1),
                )
                .with_nulls(&nulls),
            )
            .col("id", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(8), BatchCount::from(1));
        let mut dataset = Dataset::write(
            data,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let ids = dataset
            .scan()
            .with_row_id()
            .project::<&str>(&[])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = ids[ROW_ID].as_primitive::<UInt64Type>().values();
        // Both queries share invalid rows. Each also has a better neighbor in
        // the other query's candidates, which must never become eligible.
        let mut candidates = record_batch!(
            (QUERY_INDEX_COL, Int32, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            (
                ROW_ID,
                UInt64,
                [
                    ids[0], ids[0], ids[2], ids[4], ids[5], ids[7], ids[1], ids[3], ids[6], ids[5],
                    ids[7]
                ]
            )
        )
        .unwrap();
        if scenario == "deleted" {
            dataset.delete("id = 2").await.unwrap();
            // Physical-ID takes require live rows, as guaranteed by the ANN
            // prefilter. Stable-ID takes can additionally omit missing IDs.
            if !stable_row_ids {
                let mask = arrow_array::BooleanArray::from_iter(
                    candidates[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .iter()
                        .map(|id| id.map(|id| id != ids[2])),
                );
                candidates = arrow_select::filter::filter_record_batch(&candidates, &mask).unwrap();
            }
        }
        let query = Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![9.0, 0.0])),
            k: 1,
            lower_bound: (scenario == "bounded").then_some(5.0),
            upper_bound: (scenario == "bounded").then_some(101.0),
            minimum_nprobes: 1,
            maximum_nprobes: Some(1),
            ef: None,
            refine_factor: Some(2),
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: 1,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        };
        let input = if scenario == "empty" {
            candidates.slice(0, 0)
        } else {
            candidates
        };
        let input = Arc::new(TestingExec::new(vec![input]));
        let mut node = BatchRefineExec::try_new(Arc::new(dataset), input, query, 2).unwrap();
        node.rows_per_take = rows_per_take;
        let node = Arc::new(node);
        let output = node
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        if matches!(scenario, "empty" | "all_null") {
            assert_eq!(output.num_rows(), 0);
        } else {
            assert_eq!(
                output[QUERY_INDEX_COL].as_primitive::<Int32Type>().values(),
                &[0, 1]
            );
            let expected = [
                ids[if scenario == "deleted" { 4 } else { 2 }],
                ids[if scenario == "bounded" { 3 } else { 1 }],
            ];
            assert_eq!(
                output[ROW_ID].as_primitive::<UInt64Type>().values(),
                &expected
            );
            let distances = [36.0, if scenario == "bounded" { 100.0 } else { 4.0 }];
            assert_eq!(
                output["_distance"].as_primitive::<Float32Type>().values(),
                &distances
            );
        }
        let metrics = node.metrics().unwrap();
        let read = metrics
            .sum_by_name("refine_vectors_read")
            .unwrap()
            .as_usize();
        assert_eq!(
            read,
            if scenario == "empty" {
                0
            } else if scenario == "deleted" {
                7
            } else {
                8
            }
        );
        let batches = metrics
            .sum_by_name("refine_read_batches")
            .unwrap()
            .as_usize();
        assert_eq!(
            batches,
            if scenario == "empty" {
                0
            } else {
                let candidates = if scenario == "deleted" && !stable_row_ids {
                    7_usize
                } else {
                    8
                };
                candidates.div_ceil(rows_per_take)
            }
        );
    }
}
