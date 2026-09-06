// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use arrow::array::{Float32Builder, Int32Builder};
use arrow::datatypes::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{Array, Float32Array, UInt32Array, UInt64Array};
use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, StringArray,
    builder::{ListBuilder, UInt32Builder},
    cast::AsArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::PlanProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
    Statistics,
};
use datafusion::{common::ColumnStatistics, physical_plan::metrics::ExecutionPlanMetricsSet};
use datafusion::{
    common::stats::Precision,
    physical_plan::execution_plan::{Boundedness, EmissionType},
};
use datafusion::{
    error::{DataFusionError, Result as DataFusionResult},
    physical_plan::metrics::MetricsSet,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count, Time};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt, future, stream};
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_core::utils::futures::FinallyStreamExt;
use lance_core::{
    ROW_ID_FIELD,
    utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu},
};
use lance_datafusion::utils::{
    DELTAS_SEARCHED_METRIC, ExecutionPlanMetricsSetExt, FIND_PARTITIONS_ELAPSED_METRIC,
    PARTITIONS_RANKED_METRIC, PARTITIONS_SEARCHED_METRIC,
};
use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::prefilter::PreFilter;
use lance_index::vector::DIST_Q_C_COLUMN;
use lance_index::vector::{
    DIST_COL, INDEX_UUID_COLUMN, PART_ID_COLUMN, PartitionSearchControl, Query, VectorIndex,
    flat::compute_distance,
};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use crate::index::DatasetIndexInternalExt;
use crate::index::covering::{effective_covering, effective_covering_with_ids};
use crate::index::prefilter::{DatasetPreFilter, FilterLoader};
use crate::index::vector::utils::{get_vector_type, validate_distance_type_for};
use crate::{Error, Result};
use lance_arrow::*;

use super::row_addr_mask::MaskAndLoader;
use super::utils::{
    FilteredRowIdsToPrefilter, IndexMetrics, InstrumentedRecordBatchStreamAdapter, PreFilterSource,
    SelectionVectorToPrefilter,
};

pub const QUERY_INDEX_COL: &str = "query_index";

pub fn query_index_field() -> Field {
    Field::new(QUERY_INDEX_COL, DataType::Int32, false)
}

pub struct AnnPartitionMetrics {
    index_metrics: IndexMetrics,
    partitions_ranked: Count,
    deltas_searched: Count,
    find_partitions_elapsed: Time,
    baseline_metrics: BaselineMetrics,
}

impl AnnPartitionMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
            partitions_ranked: metrics.new_count(PARTITIONS_RANKED_METRIC, partition),
            deltas_searched: metrics.new_count(DELTAS_SEARCHED_METRIC, partition),
            find_partitions_elapsed: metrics.new_time(FIND_PARTITIONS_ELAPSED_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }
}

pub struct AnnIndexMetrics {
    index_metrics: IndexMetrics,
    partitions_searched: Count,
    baseline_metrics: BaselineMetrics,
}

impl AnnIndexMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
            partitions_searched: metrics.new_count(PARTITIONS_SEARCHED_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }
}

async fn find_partitions_on_cpu(
    index: Arc<dyn VectorIndex>,
    query: Query,
) -> DataFusionResult<(UInt32Array, Float32Array)> {
    spawn_cpu(move || index.find_partitions(&query))
        .await
        .map_err(|e| DataFusionError::Execution(format!("Failed to find partitions: {}", e)))
}

fn normalize_query_for_index(index: &dyn VectorIndex, query: Query) -> DataFusionResult<Query> {
    if index.metric_type() != DistanceType::Cosine {
        return Ok(query);
    }

    let mut query = query;
    let key = normalize_arrow(&query.key)
        .map_err(|e| DataFusionError::Execution(format!("Failed to normalize query: {e}")))?
        .0;
    query.key = key;
    Ok(query)
}

/// [ExecutionPlan] compute vector distance from a query vector.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector column.
///
/// WARNING: Internal API with no stability guarantees.
#[derive(Debug)]
pub struct KNNVectorDistanceExec {
    /// Inner input node.
    pub input: Arc<dyn ExecutionPlan>,

    /// The vector query to execute.
    pub query: ArrayRef,
    pub is_batch: bool,
    pub query_count: usize,
    pub k: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub column: String,
    pub distance_type: DistanceType,
    retain_vector: bool,

    input_schema: SchemaRef,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,

    metrics: ExecutionPlanMetricsSet,
}

pub struct KnnBatchParams {
    pub is_batch: bool,
    pub query_count: usize,
    pub k: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub distance_type: DistanceType,
    pub retain_vector: bool,
}

struct BatchKnnConfig {
    stored_schema: SchemaRef,
    output_schema: SchemaRef,
    column: String,
    query: ArrayRef,
    query_count: usize,
    k: usize,
    lower_bound: Option<f32>,
    upper_bound: Option<f32>,
    distance_type: DistanceType,
    retain_vector: bool,
}

impl DisplayAs for KNNVectorDistanceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                if self.is_batch {
                    write!(
                        f,
                        "KNNVectorDistance: queries={}, k={}, metric={}",
                        self.query_count, self.k, self.distance_type,
                    )
                } else {
                    write!(f, "KNNVectorDistance: metric={}", self.distance_type,)
                }
            }
            DisplayFormatType::TreeRender => {
                if self.is_batch {
                    write!(
                        f,
                        "KNNVectorDistance\nqueries={}\nk={}\nmetric={}",
                        self.query_count, self.k, self.distance_type,
                    )
                } else {
                    write!(f, "KNNVectorDistance\nmetric={}", self.distance_type,)
                }
            }
        }
    }
}

impl KNNVectorDistanceExec {
    fn remove_field_path_from_fields(
        fields: &[Arc<Field>],
        path: &[String],
    ) -> DataFusionResult<Vec<Arc<Field>>> {
        if path.is_empty() {
            return Ok(fields.to_vec());
        }
        let mut removed = false;
        let mut new_fields = Vec::with_capacity(fields.len());
        for field in fields {
            if field.name() != &path[0] {
                new_fields.push(field.clone());
                continue;
            }
            removed = true;
            if path.len() == 1 {
                continue;
            }
            match field.data_type() {
                DataType::Struct(children) => {
                    let child_fields = children.iter().cloned().collect::<Vec<_>>();
                    let projected_children =
                        Self::remove_field_path_from_fields(&child_fields, &path[1..])?;
                    if projected_children.is_empty() {
                        continue;
                    }
                    let updated = Field::new(
                        field.name(),
                        DataType::Struct(projected_children.into()),
                        field.is_nullable(),
                    )
                    .with_metadata(field.metadata().clone());
                    new_fields.push(Arc::new(updated));
                }
                _ => {
                    return Err(DataFusionError::Internal(format!(
                        "batch KNN cannot remove nested path '{}': '{}' is not a struct",
                        path.join("."),
                        field.name()
                    )));
                }
            }
        }
        if !removed {
            return Err(DataFusionError::Internal(format!(
                "batch KNN expected vector column '{}' in scan batch schema",
                path.join(".")
            )));
        }
        Ok(new_fields)
    }

    fn remove_vector_from_schema(schema: &Schema, column: &str) -> DataFusionResult<Schema> {
        let path = lance_core::datatypes::parse_field_path(column).map_err(|err| {
            DataFusionError::Internal(format!(
                "batch KNN failed to parse vector column path '{column}': {err}"
            ))
        })?;
        let fields = schema.fields().iter().cloned().collect::<Vec<_>>();
        let updated_fields = Self::remove_field_path_from_fields(&fields, &path)?;
        Ok(Schema::new_with_metadata(
            updated_fields,
            schema.metadata().clone(),
        ))
    }

    fn remove_vector_from_batch(
        batch: &RecordBatch,
        column: &str,
    ) -> DataFusionResult<RecordBatch> {
        let slim_schema = Self::remove_vector_from_schema(batch.schema().as_ref(), column)?;
        batch
            .project_by_schema(&slim_schema)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }

    fn resolve_vector_column(batch: &RecordBatch, column: &str) -> DataFusionResult<ArrayRef> {
        if let Some(col) = batch.column_by_name(column) {
            return Ok(col.clone());
        }
        let parts = lance_core::datatypes::parse_field_path(column).map_err(|e| {
            DataFusionError::Internal(format!(
                "batch KNN failed to parse vector column path '{column}': {e}"
            ))
        })?;
        if parts.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "batch KNN has invalid empty vector column path '{column}'"
            )));
        }
        let mut current = batch.column_by_name(&parts[0]).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "batch KNN expected vector column '{column}' in scan batch (missing root field '{}')",
                parts[0]
            ))
        })?;
        for part in &parts[1..] {
            let struct_array = current
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "batch KNN expected struct while resolving '{column}', but parent of '{part}' was not a struct"
                    ))
                })?;
            current = struct_array.column_by_name(part).cloned().ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "batch KNN expected vector column '{column}' in scan batch (missing nested field '{part}')"
                ))
            })?;
        }
        Ok(current)
    }

    /// Create a new [`KNNVectorDistanceExec`] node.
    ///
    /// Returns an error if the preconditions are not met.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        column: &str,
        query: ArrayRef,
        distance_type: DistanceType,
    ) -> Result<Self> {
        Self::try_new_batch(
            input,
            column,
            query,
            KnnBatchParams {
                is_batch: false,
                query_count: 1,
                k: 0,
                lower_bound: None,
                upper_bound: None,
                distance_type,
                retain_vector: false,
            },
        )
    }

    pub(crate) fn try_new_batch(
        input: Arc<dyn ExecutionPlan>,
        column: &str,
        query: ArrayRef,
        params: KnnBatchParams,
    ) -> Result<Self> {
        let KnnBatchParams {
            is_batch,
            query_count,
            k,
            lower_bound,
            upper_bound,
            distance_type,
            retain_vector,
        } = params;
        if query_count == 0 {
            return Err(Error::invalid_input(
                "query_count must be positive for KNN".to_string(),
            ));
        }
        if !query.len().is_multiple_of(query_count) {
            return Err(Error::invalid_input(format!(
                "query length ({}) must be divisible by query_count ({})",
                query.len(),
                query_count
            )));
        }
        if is_batch && k == 0 {
            return Err(Error::invalid_input(
                "k must be positive for batch KNN".to_string(),
            ));
        }

        let mut input_schema = input.schema().as_ref().clone();
        let (_, element_type) = get_vector_type(&(&input_schema).try_into()?, column)?;
        validate_distance_type_for(distance_type, &element_type)?;

        // FlatExec appends a distance column to the input schema. The input
        // may already have a distance column (possibly in the wrong position), so
        // we need to remove it before adding a new one.
        if input_schema.column_with_name(DIST_COL).is_some() {
            input_schema = input_schema.without_column(DIST_COL);
        }
        if is_batch && input_schema.column_with_name(QUERY_INDEX_COL).is_some() {
            return Err(Error::invalid_input(format!(
                "batch KNN cannot run when the input already contains reserved column '{QUERY_INDEX_COL}'"
            )));
        }

        let stored_schema = if is_batch && !retain_vector {
            Arc::new(Self::remove_vector_from_schema(&input_schema, column)?)
        } else {
            Arc::new(input_schema)
        };

        let output_schema = if is_batch {
            stored_schema
                .as_ref()
                .try_with_column_at(0, query_index_field())?
        } else {
            stored_schema.as_ref().clone()
        };
        let output_schema = Arc::new(output_schema.try_with_column(Field::new(
            DIST_COL,
            DataType::Float32,
            true,
        ))?);

        // This node has the same partitioning & boundedness as the input node
        // but it destroys any ordering.
        let properties = if is_batch {
            Arc::new(PlanProperties::new(
                EquivalenceProperties::new(output_schema.clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            ))
        } else {
            Arc::new(
                input
                    .properties()
                    .as_ref()
                    .clone()
                    .with_eq_properties(EquivalenceProperties::new(output_schema.clone())),
            )
        };

        Ok(Self {
            input,
            query,
            is_batch,
            query_count,
            k,
            lower_bound,
            upper_bound,
            column: column.to_string(),
            distance_type,
            retain_vector,
            input_schema: stored_schema,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    fn take_vector_row(vectors: &dyn Array, row_index: u32) -> DataFusionResult<ArrayRef> {
        let indices = UInt32Array::from_iter([Some(row_index)]);
        arrow_select::take::take(vectors, &indices, None)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }

    fn take_slim_batch_field(
        results: &[BatchKnnCandidate],
        field_name: &str,
    ) -> DataFusionResult<ArrayRef> {
        Self::take_slim_batch_field_if_present(results, field_name)?.ok_or_else(|| {
            DataFusionError::Internal(format!("column '{field_name}' missing from slim batch"))
        })
    }

    fn take_slim_batch_field_if_present(
        results: &[BatchKnnCandidate],
        field_name: &str,
    ) -> DataFusionResult<Option<ArrayRef>> {
        use std::collections::HashMap;

        type SlimBatchGroup = (Arc<RecordBatch>, Vec<(usize, u32)>);
        let mut groups: HashMap<*const RecordBatch, SlimBatchGroup> = HashMap::new();
        for (result_index, candidate) in results.iter().enumerate() {
            let BatchKnnExtra::WithSlimBatch {
                slim_batch,
                row_index,
                ..
            } = &candidate.extra
            else {
                return Err(DataFusionError::Internal(
                    "batch KNN expected slim batch in candidate heap".to_string(),
                ));
            };
            groups
                .entry(Arc::as_ptr(slim_batch))
                .or_insert_with(|| (Arc::clone(slim_batch), Vec::new()))
                .1
                .push((result_index, *row_index));
        }

        let mut ordered: Vec<Option<ArrayRef>> = vec![None; results.len()];
        for (_, (slim_batch, entries)) in groups {
            let indices =
                UInt32Array::from_iter(entries.iter().map(|(_, row_index)| Some(*row_index)));
            let taken = arrow_select::take::take_record_batch(slim_batch.as_ref(), &indices)
                .map_err(|e| {
                    DataFusionError::ArrowError(Box::new(e), Some("take top-k rows".to_string()))
                })?;
            let Some(column) = taken.column_by_name(field_name) else {
                continue;
            };
            for (offset, (result_index, _)) in entries.iter().enumerate() {
                ordered[*result_index] = Some(column.slice(offset, 1));
            }
        }
        if ordered.iter().all(Option::is_none) {
            return Ok(None);
        }
        if ordered.iter().any(Option::is_none) {
            return Err(DataFusionError::Internal(format!(
                "column '{field_name}' inconsistently present in slim batches"
            )));
        }

        let row_arrays: Vec<&dyn Array> = ordered
            .iter()
            .map(|array| {
                array
                    .as_ref()
                    .expect("every result mapped from slim batch")
                    .as_ref()
            })
            .collect();
        arrow::compute::concat(&row_arrays)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
            .map(Some)
    }

    fn build_struct_column_for_path(
        field: &Field,
        path: &[String],
        leaf_column: ArrayRef,
        slim_column: Option<&dyn Array>,
    ) -> DataFusionResult<ArrayRef> {
        if path.is_empty() {
            return Ok(leaf_column);
        }
        let DataType::Struct(children) = field.data_type() else {
            return Err(DataFusionError::Internal(format!(
                "batch KNN expected struct field '{}' while rebuilding nested vector path '{}'",
                field.name(),
                path.join(".")
            )));
        };
        let slim_struct = slim_column
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<arrow_array::StructArray>()
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "batch KNN expected slim column '{}' to be a struct while rebuilding nested vector path '{}'",
                            field.name(),
                            path.join(".")
                        ))
                    })
            })
            .transpose()?;
        let mut columns = Vec::with_capacity(children.len());
        for child in children.iter() {
            if child.name() == &path[0] {
                if path.len() == 1 {
                    columns.push(leaf_column.clone());
                } else {
                    let child_slim_column = slim_struct
                        .and_then(|struct_array| struct_array.column_by_name(child.name()));
                    columns.push(Self::build_struct_column_for_path(
                        child,
                        &path[1..],
                        leaf_column.clone(),
                        child_slim_column.map(|column| column.as_ref()),
                    )?);
                }
            } else if let Some(column) =
                slim_struct.and_then(|struct_array| struct_array.column_by_name(child.name()))
            {
                columns.push(column.clone());
            } else {
                columns.push(arrow_array::new_null_array(
                    child.data_type(),
                    leaf_column.len(),
                ));
            }
        }
        let struct_array = arrow_array::StructArray::try_new(
            children.clone(),
            columns,
            slim_struct.and_then(|struct_array| struct_array.nulls().cloned()),
        )
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        Ok(Arc::new(struct_array))
    }

    fn take_retained_vector_column(
        results: &[BatchKnnCandidate],
        field: &Field,
        field_path: &[String],
    ) -> DataFusionResult<ArrayRef> {
        let vector_rows: Vec<&dyn Array> = results
            .iter()
            .map(|candidate| {
                let BatchKnnExtra::WithSlimBatch {
                    vector_row: Some(vector_row),
                    ..
                } = &candidate.extra
                else {
                    return Err(DataFusionError::Internal(
                        "batch KNN expected vector rows in candidate heap".to_string(),
                    ));
                };
                Ok(vector_row.as_ref())
            })
            .collect::<DataFusionResult<Vec<_>>>()?;
        let leaf_column = arrow::compute::concat(&vector_rows)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        if field_path.len() <= 1 {
            Ok(leaf_column)
        } else {
            let slim_column = Self::take_slim_batch_field_if_present(results, field.name())?;
            Self::build_struct_column_for_path(
                field,
                &field_path[1..],
                leaf_column,
                slim_column.as_deref(),
            )
        }
    }

    fn assemble_batch_output(
        results: &[BatchKnnCandidate],
        stored_schema: &Schema,
        column: &str,
        retain_vector: bool,
    ) -> DataFusionResult<RecordBatch> {
        let field_path = lance_core::datatypes::parse_field_path(column).map_err(|e| {
            DataFusionError::Internal(format!(
                "batch KNN failed to parse vector column path '{column}': {e}"
            ))
        })?;
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(stored_schema.fields().len());
        for field in stored_schema.fields() {
            if field.name() == ROW_ID {
                let row_ids =
                    UInt64Array::from_iter(results.iter().map(|candidate| Some(candidate.row_id)));
                columns.push(Arc::new(row_ids));
            } else if retain_vector && !field_path.is_empty() && field.name() == &field_path[0] {
                columns.push(Self::take_retained_vector_column(
                    results,
                    field,
                    &field_path,
                )?);
            } else {
                columns.push(Self::take_slim_batch_field(results, field.name())?);
            }
        }
        RecordBatch::try_new(Arc::new(stored_schema.clone()), columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }

    async fn execute_batch(
        input: SendableRecordBatchStream,
        config: BatchKnnConfig,
    ) -> DataFusionResult<RecordBatch> {
        let BatchKnnConfig {
            stored_schema,
            output_schema,
            column,
            query,
            query_count,
            k,
            lower_bound,
            upper_bound,
            distance_type,
            retain_vector,
        } = config;
        let query_dim = query.len() / query_count;
        let needs_slim_batch = stored_schema.fields().iter().any(|f| f.name() != ROW_ID);
        let mut heaps = (0..query_count)
            .map(|_| BinaryHeap::<BatchKnnCandidate>::with_capacity(k))
            .collect::<Vec<_>>();
        let mut input = input;

        while let Some(batch) = input.next().await {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }

            let row_ids = batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "KNNVectorDistanceExec batch mode requires _rowid in input".to_string(),
                    )
                })?
                .as_primitive::<UInt64Type>()
                .clone();

            let mut slim_batch: Option<Arc<RecordBatch>> = None;
            let vectors = if retain_vector {
                Some(Self::resolve_vector_column(&batch, &column)?)
            } else {
                None
            };

            for (query_index, heap) in heaps.iter_mut().enumerate().take(query_count) {
                let key = query.slice(query_index * query_dim, query_dim);
                let with_distances = compute_distance(key, distance_type, &column, batch.clone())
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                let distances = with_distances[DIST_COL].as_primitive::<Float32Type>();
                let distance_values = distances.values();
                for row_index in 0..distances.len() {
                    if !distances.is_valid(row_index) {
                        continue;
                    }
                    let distance = distance_values[row_index];
                    if distance.is_nan() {
                        continue;
                    }
                    // Single-query flat KNN applies distance_range as a plan filter.
                    // Batch mode filters before insertion so top-k stays per query.
                    if lower_bound.is_some_and(|lower_bound| distance < lower_bound)
                        || upper_bound.is_some_and(|upper_bound| distance >= upper_bound)
                    {
                        continue;
                    }
                    let query_index = query_index as i32;
                    let row_id = row_ids.value(row_index);
                    if !would_enter_heap(heap, k, distance, row_id, query_index) {
                        continue;
                    }

                    let extra = if retain_vector || needs_slim_batch {
                        let row_index = row_index as u32;
                        if slim_batch.is_none() {
                            let slim = Self::remove_vector_from_batch(&batch, &column)?;
                            slim_batch = Some(Arc::new(slim));
                        }
                        let slim_batch = slim_batch.as_ref().expect("slim batch");
                        let vector_row = if retain_vector {
                            Some(Self::take_vector_row(
                                vectors.as_ref().expect("vectors"),
                                row_index,
                            )?)
                        } else {
                            None
                        };
                        BatchKnnExtra::WithSlimBatch {
                            slim_batch: Arc::clone(slim_batch),
                            row_index,
                            vector_row,
                        }
                    } else {
                        BatchKnnExtra::RowIdOnly
                    };
                    let candidate = BatchKnnCandidate {
                        query_index,
                        distance,
                        row_id,
                        extra,
                    };
                    if heap.len() < k {
                        heap.push(candidate);
                    } else {
                        heap.pop();
                        heap.push(candidate);
                    }
                }
            }
        }

        let mut results = heaps
            .into_iter()
            .flat_map(BinaryHeap::into_vec)
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            left.query_index
                .cmp(&right.query_index)
                .then_with(|| left.distance.total_cmp(&right.distance))
                .then_with(|| left.row_id.cmp(&right.row_id))
        });

        if results.is_empty() {
            return Ok(RecordBatch::new_empty(output_schema));
        }

        let mut query_indices = Int32Builder::with_capacity(results.len());
        let mut distances = Float32Builder::with_capacity(results.len());
        for result in &results {
            query_indices.append_value(result.query_index);
            distances.append_value(result.distance);
        }

        let output =
            Self::assemble_batch_output(&results, stored_schema.as_ref(), &column, retain_vector)?;

        output
            .try_with_column_at(0, query_index_field(), Arc::new(query_indices.finish()))
            .and_then(|batch| {
                batch.try_with_column(
                    Field::new(DIST_COL, DataType::Float32, true),
                    Arc::new(distances.finish()),
                )
            })
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl ExecutionPlan for KNNVectorDistanceExec {
    fn name(&self) -> &str {
        "KNNVectorDistanceExec"
    }

    /// Flat KNN inherits the schema from input node, and add one distance column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "KNNVectorDistanceExec node must have exactly one child".to_string(),
            ));
        }

        Ok(Arc::new(Self::try_new_batch(
            children.pop().expect("length checked"),
            &self.column,
            self.query.clone(),
            KnnBatchParams {
                is_batch: self.is_batch,
                query_count: self.query_count,
                k: self.k,
                lower_bound: self.lower_bound,
                upper_bound: self.upper_bound,
                distance_type: self.distance_type,
                retain_vector: self.retain_vector,
            },
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        if self.is_batch {
            let stream = stream::once(Self::execute_batch(
                input_stream,
                BatchKnnConfig {
                    stored_schema: self.input_schema.clone(),
                    output_schema: self.output_schema.clone(),
                    column: self.column.clone(),
                    query: self.query.clone(),
                    query_count: self.query_count,
                    k: self.k,
                    lower_bound: self.lower_bound,
                    upper_bound: self.upper_bound,
                    distance_type: self.distance_type,
                    retain_vector: self.retain_vector,
                },
            ));
            let schema = self.schema();
            return Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
                schema,
                stream.boxed(),
                partition,
                &self.metrics,
            )) as SendableRecordBatchStream);
        }
        let key = self.query.clone();
        let column = self.column.clone();
        let dt = self.distance_type;
        let schema = self.schema();

        // Empty batches don't have a vector column to score; filter them out
        // before the transform so it always sees real work.
        let filtered_input = input_stream.try_filter(|batch| future::ready(batch.num_rows() > 0));

        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let elapsed_compute = baseline.elapsed_compute().clone();

        let stream = filtered_input
            .map(move |batch_result| {
                let key = key.clone();
                let column = column.clone();
                let elapsed_compute = elapsed_compute.clone();
                async move {
                    let batch = batch_result?;
                    // Time around the .await to capture the spawn_blocking
                    // distance work, which otherwise runs while this future is
                    // Pending and is missed by a poll-time timer.
                    let start = Instant::now();
                    let batch = compute_distance(key, dt, &column, batch)
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    elapsed_compute.add_duration(start.elapsed());

                    let _t = elapsed_compute.timer();
                    let distances = batch[DIST_COL].as_primitive::<Float32Type>();
                    let distance_values = distances.values();
                    let mask = BooleanArray::from_iter((0..distances.len()).map(|row_index| {
                        Some(distances.is_valid(row_index) && !distance_values[row_index].is_nan())
                    }));
                    arrow::compute::filter_record_batch(&batch, &mask)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                }
            })
            .buffer_unordered(get_num_compute_intensive_cpus());

        let stream = stream.map(move |batch| {
            let poll = baseline.record_poll(std::task::Poll::Ready(Some(batch)));
            match poll {
                std::task::Poll::Ready(Some(b)) => b,
                _ => unreachable!("record_poll preserves Ready(Some) input"),
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        let inner_stats = self.input.partition_statistics(partition)?;
        let input_schema = self.input.schema();
        let input_stats_by_name = inner_stats
            .column_statistics
            .iter()
            .zip(input_schema.fields())
            .map(|(stats, field)| (field.name().as_str(), stats.clone()))
            .collect::<HashMap<_, _>>();
        let vector_root = lance_core::datatypes::parse_field_path(&self.column)
            .ok()
            .and_then(|parts| parts.first().cloned())
            .unwrap_or_else(|| self.column.clone());
        let dist_stats = input_stats_by_name
            .get(vector_root.as_str())
            .map(|stats| ColumnStatistics {
                null_count: stats.null_count,
                ..Default::default()
            })
            .unwrap_or_default();
        let column_statistics = self
            .output_schema
            .fields()
            .iter()
            .map(|field| {
                if field.name() == QUERY_INDEX_COL {
                    ColumnStatistics::default()
                } else if field.name() == DIST_COL {
                    dist_stats.clone()
                } else {
                    input_stats_by_name
                        .get(field.name().as_str())
                        .cloned()
                        .unwrap_or_default()
                }
            })
            .collect::<Vec<_>>();
        Ok(Arc::new(Statistics {
            num_rows: inner_stats.num_rows,
            column_statistics,
            ..Statistics::new_unknown(self.schema().as_ref())
        }))
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

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Both batch and non-batch modes execute a single input partition at a time,
        // so all input must be coalesced to one partition before distance computation.
        vec![Distribution::SinglePartition]
    }
}

#[derive(Clone)]
struct BatchKnnCandidate {
    query_index: i32,
    distance: f32,
    row_id: u64,
    extra: BatchKnnExtra,
}

#[derive(Clone)]
enum BatchKnnExtra {
    RowIdOnly,
    WithSlimBatch {
        slim_batch: Arc<RecordBatch>,
        row_index: u32,
        vector_row: Option<ArrayRef>,
    },
}

fn would_enter_heap(
    heap: &BinaryHeap<BatchKnnCandidate>,
    k: usize,
    distance: f32,
    row_id: u64,
    query_index: i32,
) -> bool {
    if heap.len() < k {
        return true;
    }
    let worst = heap.peek().expect("heap non-empty when len >= k");
    let probe = BatchKnnCandidate {
        query_index,
        distance,
        row_id,
        extra: BatchKnnExtra::RowIdOnly,
    };
    probe.cmp(worst).is_lt()
}

impl PartialEq for BatchKnnCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.query_index == other.query_index
            && self.distance == other.distance
            && self.row_id == other.row_id
    }
}

impl Eq for BatchKnnCandidate {}

impl PartialOrd for BatchKnnCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for BatchKnnCandidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.row_id.cmp(&other.row_id))
            .then_with(|| self.query_index.cmp(&other.query_index))
    }
}

pub static KNN_INDEX_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| knn_empty_result_schema(false));

/// Schema for empty vector-search results (e.g. `fast_search` with no index).
pub fn knn_empty_result_schema(include_query_index: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ];
    if include_query_index {
        fields.insert(0, query_index_field());
    }
    Arc::new(Schema::new(fields))
}

pub static KNN_PARTITION_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new(
            PART_ID_COLUMN,
            DataType::List(Field::new("item", DataType::UInt32, false).into()),
            false,
        ),
        Field::new(
            DIST_Q_C_COLUMN,
            DataType::List(Field::new("item", DataType::Float32, false).into()),
            false,
        ),
        Field::new(INDEX_UUID_COLUMN, DataType::Utf8, false),
    ]))
});

fn covering_fields_match(expected: &Field, physical: &Field) -> bool {
    expected.name() == physical.name()
        && expected.data_type() == physical.data_type()
        && expected.is_nullable() == physical.is_nullable()
        && expected.metadata() == physical.metadata()
}

/// Intersect a requested manifest declaration with the capabilities proven by every
/// selected physical segment.
///
/// The output remains in declaration order because that is the schema the ANN exec
/// exposes. If a segment stores the surviving columns in a different relative order then
/// none are served: storage emits in physical order, and pairing those values with a
/// declaration-ordered schema would be incorrect.
fn common_physical_covering(
    mut requested: Vec<(i32, Field)>,
    physical_by_segment: &[Vec<(i32, Field)>],
) -> Vec<(i32, Field)> {
    if physical_by_segment.is_empty() {
        return Vec::new();
    }

    for physical in physical_by_segment {
        let before = requested.len();
        requested.retain(|(expected_id, expected_field)| {
            let kept = physical.iter().any(|(physical_id, physical_field)| {
                expected_id == physical_id && covering_fields_match(expected_field, physical_field)
            });
            if !kept {
                // Withdrawal is silent by design -- results stay correct, the plan just
                // grows a base-table read -- which makes a covered index that quietly
                // stopped eliding takes indistinguishable from one that never worked.
                // `covering_fields_match` compares name, data type, nullability AND
                // metadata, so the cause is usually a drift too small to guess at.
                log::debug!(
                    "Covering column '{}' (field id {}) is declared but not servable by a \
                     segment: no physical column matches it on name, type, nullability and \
                     metadata. Its values will come from a base-table read.",
                    expected_field.name(),
                    expected_id
                );
            }
            kept
        });
        if requested.is_empty() {
            if before > 0 {
                log::debug!(
                    "No declared covering column is servable by every selected segment; \
                     this query reads all of them from the base table."
                );
            }
            return requested;
        }
    }

    let requested_ids = requested.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    for physical in physical_by_segment {
        let physical_ids = physical
            .iter()
            .filter(|(physical_id, physical_field)| {
                requested.iter().any(|(expected_id, expected_field)| {
                    expected_id == physical_id
                        && covering_fields_match(expected_field, physical_field)
                })
            })
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        if requested_ids != physical_ids {
            // Segments disagree on the order or set they can serve. Storage emits in
            // physical order while the schema is declaration-ordered, so pairing them
            // would be wrong -- none are served rather than some.
            log::debug!(
                "Selected segments do not agree on the covering columns they can serve \
                 ({requested_ids:?} vs {physical_ids:?}); this query reads all of them from \
                 the base table."
            );
            return Vec::new();
        }
    }
    requested
}

/// Resolve the query's manifest-requested covering columns against every selected
/// segment's physical storage capability.
///
/// The returned query always has an explicit covering projection. A missing physical
/// column is omitted from it, causing the normal scanner take to fetch that value from
/// the base table instead.
pub async fn resolve_physical_covering_query(
    dataset: Arc<Dataset>,
    indices: &[IndexMetadata],
    query: &Query,
) -> Result<Query> {
    let mut resolved_query = query.clone();
    let declared_ids = indices
        .first()
        .map(|index| index.covering_fields.as_slice())
        .unwrap_or_default();
    if let Some(mismatch) = indices
        .iter()
        .find(|index| index.covering_fields != declared_ids)
    {
        return Err(Error::index(format!(
            "ANN index delta segments of index '{}' disagree on covering fields ({:?} vs {:?}); \
             the index metadata is inconsistent -- rebuild the index",
            mismatch.name, declared_ids, mismatch.covering_fields
        )));
    }

    let requested = effective_covering_with_ids(
        declared_ids,
        query.covering_projection.as_deref(),
        dataset.schema(),
    )?;
    if requested.is_empty() {
        resolved_query.covering_projection = Some(Arc::from(Vec::<String>::new()));
        return Ok(resolved_query);
    }

    let physical_by_segment = future::try_join_all(indices.iter().map(|index| {
        let dataset = dataset.clone();
        let column = query.column.clone();
        let uuid = index.uuid;
        async move {
            dataset
                .open_vector_index(&column, &uuid, &NoOpMetricsCollector)
                .await?
                .physical_covering_fields()
        }
    }))
    .await?;
    let names = common_physical_covering(requested, &physical_by_segment)
        .into_iter()
        .map(|(_, field)| field.name().clone())
        .collect::<Vec<_>>();
    resolved_query.covering_projection = Some(Arc::from(names));
    Ok(resolved_query)
}

/// Create a new ANN execution node. `overlay_block`, when `Some`, excludes rows whose index
/// entries may be stale due to a newer data overlay (see [`ANNIvfSubIndexExec::with_overlay_block`]).
/// `external_mask`, when `Some`, additionally restricts the scan to a caller-supplied
/// allow/block set (see [`ANNIvfSubIndexExec::with_external_mask`]).
pub async fn new_knn_exec(
    dataset: Arc<Dataset>,
    indices: &[IndexMetadata],
    query: &Query,
    prefilter_source: PreFilterSource,
    overlay_block: Option<RowAddrMask>,
    external_mask: Option<Arc<RowAddrMask>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let query = resolve_physical_covering_query(dataset.clone(), indices, query).await?;
    let ivf_node = ANNIvfPartitionExec::try_new(
        dataset.clone(),
        indices.iter().map(|idx| idx.uuid).collect_vec(),
        query.clone(),
    )?;

    let mut sub_index = ANNIvfSubIndexExec::try_new_with_resolved_covering(
        Arc::new(ivf_node),
        dataset,
        indices.to_vec(),
        query,
        prefilter_source,
    )?;
    if let Some(overlay_block) = overlay_block {
        sub_index = sub_index.with_overlay_block(overlay_block);
    }
    if external_mask.is_some() {
        sub_index = sub_index.with_external_mask(external_mask);
    }

    Ok(Arc::new(sub_index))
}

/// [ExecutionPlan] to execute the find the closest IVF partitions.
///
/// It searches the partition IDs using the input query.
///
/// It searches all index deltas in parallel.  For each delta it returns a
/// single batch with the partition IDs and the delta index `uuid`:
///
/// The number of partitions returned is at most `maximum_nprobes`.  If
/// `maximum_nprobes` is not set, it will return all partitions.  The partitions
/// are returned in sorted order from closest to farthest.
///
/// Typically, all partition ids will be identical for each delta index (since delta
/// indices have identical partitions) but the downstream nodes do not rely on this.
///
/// TODO: We may want to search the partitions once instead of once per delta index
/// since the centroids are the same.
///
/// ```text
/// {
///    "__ivf_part_id": List<UInt32>,
///    "__index_uuid": String,
/// }
/// ```
#[derive(Debug)]
pub struct ANNIvfPartitionExec {
    pub dataset: Arc<Dataset>,

    /// The vector query to execute.
    pub query: Query,

    /// The UUIDs of the indices to search.
    pub index_uuids: Vec<Uuid>,

    pub properties: Arc<PlanProperties>,

    pub metrics: ExecutionPlanMetricsSet,
}

impl ANNIvfPartitionExec {
    pub fn try_new(dataset: Arc<Dataset>, index_uuids: Vec<Uuid>, query: Query) -> Result<Self> {
        let dataset_schema = dataset.schema();
        get_vector_type(dataset_schema, &query.column)?;
        if index_uuids.is_empty() {
            return Err(Error::execution(
                "ANNIVFPartitionExec node: no index found for query".to_string(),
            ));
        }

        let schema = KNN_PARTITION_SCHEMA.clone();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            dataset,
            query,
            index_uuids,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for ANNIvfPartitionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNIvfPartition: uuid={}, minimum_nprobes={}, maximum_nprobes={:?}, deltas={}",
                    self.index_uuids[0],
                    self.query.minimum_nprobes,
                    self.query.maximum_nprobes,
                    self.index_uuids.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ANNIvfPartition\nuuid={}\nminimum_nprobes={}\nmaximum_nprobes={:?}\ndeltas={}",
                    self.index_uuids[0],
                    self.query.minimum_nprobes,
                    self.query.maximum_nprobes,
                    self.index_uuids.len()
                )
            }
        }
    }
}

impl ExecutionPlan for ANNIvfPartitionExec {
    fn name(&self) -> &str {
        "ANNIVFPartitionExec"
    }

    fn schema(&self) -> SchemaRef {
        KNN_PARTITION_SCHEMA.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics {
            num_rows: Precision::Exact(self.query.minimum_nprobes),
            ..Statistics::new_unknown(self.schema().as_ref())
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            Err(DataFusionError::Internal(
                "ANNIVFPartitionExec node does not accept children".to_string(),
            ))
        } else {
            Ok(self)
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let timer = Instant::now();

        let target_partitions = context.session_config().target_partitions();
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let metrics = Arc::new(AnnPartitionMetrics::new(&self.metrics, partition));
        metrics.deltas_searched.add(self.index_uuids.len());
        let metrics_clone = metrics.clone();

        let stream = stream::iter(self.index_uuids.clone())
            .map(move |uuid| {
                let query = query.clone();
                let ds = ds.clone();
                let metrics = metrics.clone();
                async move {
                    let index = ds
                        .open_vector_index(&query.column, &uuid, &metrics.index_metrics)
                        .await?;
                    // Normalize cosine queries once before partition ranking.
                    let query = normalize_query_for_index(index.as_ref(), query.clone())?;

                    metrics.partitions_ranked.add(index.total_partitions());

                    let (partitions, dist_q_c) = {
                        let _timer = metrics.find_partitions_elapsed.timer();
                        find_partitions_on_cpu(index, query).await?
                    };

                    let mut part_list_builder = ListBuilder::new(UInt32Builder::new())
                        .with_field(Field::new("item", DataType::UInt32, false));
                    part_list_builder.append_value(partitions.iter());
                    let partition_col = part_list_builder.finish();

                    let mut dist_q_c_list_builder = ListBuilder::new(Float32Builder::new())
                        .with_field(Field::new("item", DataType::Float32, false));
                    dist_q_c_list_builder.append_value(dist_q_c.iter());
                    let dist_q_c_col = dist_q_c_list_builder.finish();

                    let uuid_col = StringArray::from(vec![uuid.to_string()]);
                    let batch = RecordBatch::try_new(
                        KNN_PARTITION_SCHEMA.clone(),
                        vec![
                            Arc::new(partition_col),
                            Arc::new(dist_q_c_col),
                            Arc::new(uuid_col),
                        ],
                    )?;
                    metrics.baseline_metrics.record_output(batch.num_rows());
                    Ok::<_, DataFusionError>(batch)
                }
            })
            .buffered(self.index_uuids.len().min(target_partitions).max(1))
            .finally(move || {
                // Partition ranking reads centroids from memory, so this is
                // typically zero; flushed for symmetry with ANNSubIndex.
                metrics_clone.index_metrics.flush_io();
                metrics_clone.baseline_metrics.done();
                metrics_clone
                    .baseline_metrics
                    .elapsed_compute()
                    .add_duration(timer.elapsed());
            });
        let schema = self.schema();
        Ok(
            Box::pin(RecordBatchStreamAdapter::new(schema, stream.boxed()))
                as SendableRecordBatchStream,
        )
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

/// Datafusion [ExecutionPlan] to run search on vector index partitions.
///
/// A IVF-{PQ/SQ/HNSW} query plan is:
///
/// ```text
/// AnnSubIndexExec: k=10
///   AnnPartitionExec: nprobes=20
/// ```
///
/// The partition index returns one batch per delta with `maximum_nprobes` partitions in sorted order.
///
/// The sub-index then runs a KNN search on each partition in parallel.
///
/// First, the index will search `minimum_probes` partitions on each delta.  If there are enough results
/// at that point to satisfy K then the results will be sorted and returned.
///
/// If there are not enough results then the prefilter will be consulted to determine how many potential
/// results exist.  If the number is smaller than K then those additional results will be fetched directly
/// and given maximum distance.
///
/// If the number of results is larger then additional partitions will be searched in batches of
/// `cpu_parallelism` until min(K, num_filtered_results) are found or `maximum_nprobes` partitions
/// have been searched.
///
/// TODO: In the future, if we can know that a filter will be highly selective (through cost estimation or
/// user-provided hints) we wait for the prefilter results before we load any partitions.  If there are less
/// than K (or some threshold) results then we can return without search.
#[derive(Debug)]
pub struct ANNIvfSubIndexExec {
    /// Inner input source node.
    input: Arc<dyn ExecutionPlan>,

    dataset: Arc<Dataset>,

    indices: Vec<IndexMetadata>,

    /// Vector Query.
    query: Query,

    /// Prefiltering input
    prefilter_source: PreFilterSource,

    /// Row addresses whose index entries are stale due to a newer data overlay. Blocked from
    /// index results at execution time via [`DatasetPreFilter::with_overlay_block`].
    overlay_block: Option<RowAddrMask>,

    /// Optional external row-address allow/block mask, combined with the
    /// prefilter using logical AND.
    external_mask: Option<Arc<RowAddrMask>>,
    /// Output schema: `[_distance, _rowid]` plus any included/covering columns
    /// the index emits.
    output_schema: SchemaRef,

    /// Whether this plan emits covering columns, resolved from the index
    /// declaration at construction. Kept as its own field rather than re-derived
    /// from the output schema's width at execution: width-based inference is
    /// exact only while covering is the sole thing that widens the schema, and
    /// the first unrelated extra column would silently arm per-query emitted-row
    /// tracking and the covered recovery take on every plain query (or mask
    /// covering if a column were dropped) with results staying correct -- no test
    /// would catch it.
    has_covered: bool,

    /// Datafusion Plan Properties
    properties: Arc<PlanProperties>,

    metrics: ExecutionPlanMetricsSet,
}

impl ANNIvfSubIndexExec {
    /// Construct an ANN sub-index exec without a verified physical covering capability.
    ///
    /// This synchronous entry point cannot open segment storage, so it conservatively
    /// disables covering and leaves declared values to the base-table take. Dataset
    /// planning and protobuf reconstruction use the storage-aware async resolution path
    /// before calling `Self::try_new_with_resolved_covering` (private, so not linked).
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        mut query: Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        query.covering_projection = Some(Arc::from(Vec::<String>::new()));
        Self::try_new_with_resolved_covering(input, dataset, indices, query, prefilter_source)
    }

    pub(crate) fn try_new_with_resolved_covering(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        query: Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if input.schema().field_with_name(PART_ID_COLUMN).is_err() {
            return Err(Error::index(format!(
                "ANNSubIndexExec node: input schema does not have \"{}\" column",
                PART_ID_COLUMN
            )));
        }
        // Start from the logical declaration, narrowed to the explicit projection already
        // proven against every selected segment's physical storage. The per-partition
        // search emits only this subset, so TakeExec may skip the matching base-table fields.
        let mut fields = KNN_INDEX_SCHEMA.fields().to_vec();
        let included_ids = indices
            .first()
            .map(|idx| idx.covering_fields.clone())
            .unwrap_or_default();
        // This exec declares one output schema but executes every delta segment,
        // each emitting covering columns from its own storage -- so all deltas
        // must agree on the covering set. The commit boundary enforces this; the
        // check here defends against already-committed inconsistent metadata,
        // failing with a clear error instead of a downstream schema mismatch.
        if let Some(mismatch) = indices
            .iter()
            .find(|idx| idx.covering_fields != included_ids)
        {
            return Err(Error::index(format!(
                "ANNIvfSubIndexExec: delta segments of index '{}' disagree on covering fields \
                 ({:?} vs {:?}); the index metadata is inconsistent -- rebuild the index",
                mismatch.name, included_ids, mismatch.covering_fields
            )));
        }
        // Narrow only AFTER the agreement check above: narrowing first could reduce
        // segments that genuinely disagree into a subset on which they happen to agree,
        // masking metadata corruption as a working query. The manifest remains
        // authoritative for logical identity and schema dependencies; the projection is
        // authoritative for physical availability.
        let covering = effective_covering(
            &included_ids,
            query.covering_projection.as_deref(),
            dataset.schema(),
        )?;
        // Resolved-at-construction covered flag (see the `has_covered` field): from
        // the NARROWED set, not the declaration -- a projection that needs none of
        // the declared columns (`Some(&[])`) makes this plan non-covered.
        let has_covered = !covering.is_empty();
        fields.extend(covering.into_iter().map(Arc::new));
        let output_schema = Arc::new(arrow_schema::Schema::new(fields));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            dataset,
            indices,
            query,
            prefilter_source,
            overlay_block: None,
            external_mask: None,
            output_schema,
            has_covered,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Block stale row addresses (see the `overlay_block` field) from index results.
    pub fn with_overlay_block(mut self, overlay_block: RowAddrMask) -> Self {
        self.overlay_block = Some(overlay_block);
        self
    }

    /// Restrict the ANN search to a caller-supplied row-address allow/block set.
    /// Intersected with the prefilter, so top-k is computed over surviving rows
    /// rather than filtered afterwards. No-op when `mask` is `None`.
    pub fn with_external_mask(mut self, mask: Option<Arc<RowAddrMask>>) -> Self {
        self.external_mask = mask;
        self
    }

    /// Returns a reference to the vector query.
    pub fn query(&self) -> &Query {
        &self.query
    }

    /// Returns a reference to the dataset.
    pub fn dataset(&self) -> &Arc<Dataset> {
        &self.dataset
    }

    /// Returns a reference to the index metadata.
    pub fn indices(&self) -> &[IndexMetadata] {
        &self.indices
    }

    /// Returns a reference to the prefilter source.
    pub fn prefilter_source(&self) -> &PreFilterSource {
        &self.prefilter_source
    }
}

impl DisplayAs for ANNIvfSubIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let metric_str = self
            .query
            .metric_type
            .map(|m| format!("{:?}", m))
            .unwrap_or_else(|| "default".to_string());
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNSubIndex: name={}, k={}, deltas={}, metric={}",
                    self.indices[0].name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.indices.len(),
                    metric_str
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ANNSubIndex\nname={}\nk={}\ndeltas={}\nmetric={}",
                    self.indices[0].name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.indices.len(),
                    metric_str
                )
            }
        }
    }
}

struct ANNIvfEarlySearchResults {
    k: usize,
    initial_ids: Mutex<Vec<u64>>,
    num_results_found: AtomicUsize,
    deltas_remaining: AtomicUsize,
    all_deltas_done: Notify,
    took_no_rows_shortcut: AtomicBool,
    /// One entry per covered delta whose late search reached the no-rows shortcut:
    /// that delta's per-segment ownership mask (`None` when the delta owned the
    /// whole global mask). The exec-level covered recovery emits ONLY rows these
    /// scopes select, so covered result sets stay identical to non-covered ones in
    /// both directions: a delta that never armed (it skipped its late search, e.g.
    /// probing fewer partitions than `minimum_nprobes`) would not have emitted its
    /// not-found rows on the non-covered path either, so recovery must not emit
    /// them for it -- and an empty list means no delta armed, so recovery emits
    /// nothing at all.
    covered_recovery_scopes: Mutex<Vec<Option<Arc<RowAddrMask>>>>,
}

impl ANNIvfEarlySearchResults {
    fn new(deltas_remaining: usize, k: usize) -> Self {
        Self {
            k,
            initial_ids: Mutex::new(Vec::with_capacity(k)),
            num_results_found: AtomicUsize::new(0),
            deltas_remaining: AtomicUsize::new(deltas_remaining),
            all_deltas_done: Notify::new(),
            took_no_rows_shortcut: AtomicBool::new(false),
            covered_recovery_scopes: Mutex::new(Vec::new()),
        }
    }

    fn record_batch(&self, batch: &RecordBatch) -> DataFusionResult<()> {
        let row_ids = batch.column_by_name(ROW_ID).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "ANNIvfEarlySearchResults::record_batch: batch is missing the {ROW_ID} column"
            ))
        })?;
        let mut initial_ids = self.initial_ids.lock().unwrap();
        let ids_to_record = (self.k - initial_ids.len()).min(batch.num_rows());
        initial_ids.extend(
            row_ids
                .as_primitive::<UInt64Type>()
                .values()
                .iter()
                .take(ids_to_record),
        );
        Ok(())
    }

    fn record_late_batch(&self, num_rows: usize) {
        self.num_results_found
            .fetch_add(num_rows, Ordering::Relaxed);
    }

    async fn wait_for_minimum_to_finish(&self) -> usize {
        if self.deltas_remaining.fetch_sub(1, Ordering::Relaxed) == 1 {
            {
                let new_num_results_found = self.initial_ids.lock().unwrap().len();
                self.num_results_found
                    .store(new_num_results_found, Ordering::Relaxed);
            }
            self.all_deltas_done.notify_waiters();
        } else {
            self.all_deltas_done.notified().await;
        }
        self.num_results_found.load(Ordering::Relaxed)
    }
}

struct LatePartitionSearchControl {
    state: Arc<ANNIvfEarlySearchResults>,
    max_results: usize,
    seg_mask: Option<Arc<RowAddrMask>>,
}

/// A query prefilter restricted to the fragments owned by one physical index segment.
///
/// The shared dataset prefilter covers the union of every segment.  When an in-place
/// update removes a fragment from an older segment's metadata, this additional mask
/// keeps that segment's stale physical rows out of its local top-k.
struct SegmentPreFilter {
    base: Arc<DatasetPreFilter>,
    ownership_mask: Arc<RowAddrMask>,
    final_mask: Mutex<Option<Arc<RowAddrMask>>>,
}

impl SegmentPreFilter {
    fn new(base: Arc<DatasetPreFilter>, ownership_mask: Arc<RowAddrMask>) -> Self {
        Self {
            base,
            ownership_mask,
            final_mask: Mutex::new(None),
        }
    }
}

#[async_trait::async_trait]
impl PreFilter for SegmentPreFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        self.base.wait_for_ready().await?;
        let mut final_mask = self.final_mask.lock().unwrap();
        final_mask.get_or_insert_with(|| {
            Arc::new(self.base.mask().as_ref().clone() & self.ownership_mask.as_ref().clone())
        });
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn needs_partition_row_ids(&self) -> bool {
        self.base.is_empty()
    }

    fn is_empty_for(&self, rows: &RowAddrTreeMap) -> bool {
        self.base.is_empty() && self.ownership_mask.selects_all(rows)
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.final_mask
            .lock()
            .unwrap()
            .as_ref()
            .expect("mask called without call to wait_for_ready")
            .clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask().selected_indices(row_ids)
    }
}

async fn prefilter_for_segment(
    dataset: Arc<Dataset>,
    index: &IndexMetadata,
    base: Arc<DatasetPreFilter>,
) -> Result<Arc<dyn PreFilter>> {
    let Some(owned_fragments) = index.fragment_bitmap.clone() else {
        return Ok(base);
    };
    let Some(ownership_mask) =
        DatasetPreFilter::create_restricted_deletion_mask(dataset, owned_fragments)
    else {
        return Ok(base);
    };
    let ownership_mask = ownership_mask.await?;
    Ok(Arc::new(SegmentPreFilter::new(base, ownership_mask)))
}

impl PartitionSearchControl for LatePartitionSearchControl {
    fn should_stop(&self) -> bool {
        self.state.num_results_found.load(Ordering::Relaxed) >= self.max_results
    }

    fn record_batch(&self, batch: &RecordBatch) {
        // The batch this sees is the raw partition result; the stream applies the
        // segment restriction afterwards, so only the rows that survive it may count
        // towards the shared budget.
        let num_rows = match self.seg_mask.as_ref() {
            Some(seg_mask) => num_rows_in_segment(batch, seg_mask),
            None => batch.num_rows(),
        };
        self.state.record_late_batch(num_rows);
    }
}

/// How many of `batch`'s rows belong to fragments the segment still owns.
fn num_rows_in_segment(batch: &RecordBatch, seg_mask: &RowAddrMask) -> usize {
    batch[ROW_ID]
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .filter(|&&id| seg_mask.selected(id))
        .count()
}

/// Drop the rows of `batch` whose fragment the segment no longer owns.
///
/// A segment's index file can still hold rows for fragments that were pruned from its
/// `fragment_bitmap`, for example after an in-place column update. Once a newer delta
/// owns such a fragment, those rows must not reach the query, and they must not reach
/// the shared search accounting either: a stale row that is counted and only dropped
/// later can satisfy the `k` budget on its own, so the segment that owns the fresh
/// copy stops probing and the query returns fewer than `k` current rows.
fn restrict_to_segment(
    batch: RecordBatch,
    seg_mask: Option<&RowAddrMask>,
) -> DataFusionResult<RecordBatch> {
    let Some(seg_mask) = seg_mask else {
        return Ok(batch);
    };
    if batch.num_rows() == 0 {
        return Ok(batch);
    }
    let keep = BooleanArray::from_iter(
        batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .map(|&id| Some(seg_mask.selected(id))),
    );
    if keep.false_count() == 0 {
        return Ok(batch);
    }
    arrow::compute::filter_record_batch(&batch, &keep)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

fn effective_query_parallelism(
    query: &Query,
    index: &dyn VectorIndex,
    target_partitions: usize,
) -> usize {
    let cpu_pool_size = get_num_compute_intensive_cpus()
        .min(target_partitions)
        .max(1);
    effective_query_parallelism_for(
        query,
        cpu_pool_size,
        index.auto_query_parallelism(cpu_pool_size),
    )
}

fn effective_query_parallelism_for(
    query: &Query,
    cpu_pool_size: usize,
    auto_parallelism: usize,
) -> usize {
    let cpu_pool_size = cpu_pool_size.max(1);
    match query.query_parallelism {
        -1 => cpu_pool_size,
        0 => auto_parallelism.clamp(1, cpu_pool_size),
        n if n > 0 => (n as usize).min(cpu_pool_size).max(1),
        _ => 1,
    }
}

impl ANNIvfSubIndexExec {
    async fn search_partition(
        index: Arc<dyn VectorIndex>,
        query: Query,
        part_id: usize,
        pre_filter: Arc<dyn PreFilter>,
        metrics: Arc<AnnIndexMetrics>,
        seg_mask: Option<Arc<RowAddrMask>>,
    ) -> DataFusionResult<RecordBatch> {
        let batch = index
            .search_in_partition(part_id, &query, pre_filter, &metrics.index_metrics)
            .map_err(|e| DataFusionError::Execution(format!("Failed to calculate KNN: {}", e)))
            .await?;
        let batch = restrict_to_segment(batch, seg_mask.as_deref())?;
        metrics.baseline_metrics.record_output(batch.num_rows());
        Ok(batch)
    }

    fn instrument_sequential_partition_stream(
        stream: stream::BoxStream<'static, DataFusionResult<RecordBatch>>,
        metrics: Arc<AnnIndexMetrics>,
        state: Arc<ANNIvfEarlySearchResults>,
        record_initial: bool,
        record_partition_per_batch: bool,
        seg_mask: Option<Arc<RowAddrMask>>,
    ) -> stream::BoxStream<'static, DataFusionResult<RecordBatch>> {
        stream
            .map(move |batch| {
                let batch = restrict_to_segment(batch?, seg_mask.as_deref())?;
                if record_partition_per_batch {
                    metrics.partitions_searched.add(1);
                }
                metrics.baseline_metrics.record_output(batch.num_rows());
                if record_initial {
                    state.record_batch(&batch)?;
                }
                Ok(batch)
            })
            .boxed()
    }

    #[allow(clippy::too_many_arguments)]
    fn late_search(
        index: Arc<dyn VectorIndex>,
        query: Query,
        partitions: Arc<UInt32Array>,
        q_c_dists: Arc<Float32Array>,
        prefilter: Arc<dyn PreFilter>,
        global_prefilter: Arc<DatasetPreFilter>,
        metrics: Arc<AnnIndexMetrics>,
        state: Arc<ANNIvfEarlySearchResults>,
        target_partitions: usize,
        seg_mask: Option<Arc<RowAddrMask>>,
        has_covered: bool,
    ) -> impl Stream<Item = DataFusionResult<RecordBatch>> {
        let stream = futures::stream::once(async move {
            let max_nprobes = query
                .maximum_nprobes
                .unwrap_or(partitions.len())
                .min(partitions.len());
            let min_nprobes = query.minimum_nprobes.min(max_nprobes);

            // Every delta must reach the barrier, even if it has no partitions left
            // to search, so that siblings waiting for the initial search can proceed.
            let found_so_far = state.wait_for_minimum_to_finish().await;
            if max_nprobes <= min_nprobes {
                // We've already searched all partitions, no late search needed
                return futures::stream::empty().boxed();
            }

            if found_so_far >= query.k {
                // We found enough results, no need for late search
                return futures::stream::empty().boxed();
            }

            if seg_mask
                .as_ref()
                .is_some_and(|mask| mask.max_len() == Some(0))
            {
                // Every fragment this segment used to own now belongs to a newer delta,
                // so probing it can neither produce a row nor move the shared budget that
                // stops the late search. Skip it instead of scanning to maximum_nprobes.
                return futures::stream::empty().boxed();
            }

            // We know the prefilter should be ready at this point so we shouldn't
            // need to call wait_for_ready
            let prefilter_mask = global_prefilter.mask();

            let max_results = prefilter_mask.max_len().map(|x| x as usize);

            if let Some(max_results) = max_results
                && found_so_far < max_results
                && max_results <= query.k
            {
                // In this case there are fewer than k results matching the prefilter so
                // just return the prefilter ids and don't bother searching any further

                // The shortcut batch emits only [_distance, _rowid]; a covering
                // index's declared schema carries more columns, and their values
                // cannot be fabricated here. Emit nothing instead: the
                // end-of-stream covered recovery (`covered_not_found_batch`)
                // returns every unemitted prefilter row with INFINITY distance
                // and its covering columns via a take bounded by the prefilter
                // size -- mirroring the non-covered shortcut's semantics without
                // searching (a prefilter row with no index entry, e.g. a null
                // vector, could otherwise never be found and would drive the
                // search through every partition).
                if has_covered {
                    // Arm recovery with THIS delta's ownership scope: recovery may
                    // emit only rows the arming deltas own, or it diverges from the
                    // non-covered result set whenever a sibling delta skipped its
                    // late search without arming (see `covered_recovery_scopes`).
                    state
                        .covered_recovery_scopes
                        .lock()
                        .unwrap()
                        .push(seg_mask.clone());
                    return futures::stream::empty().boxed();
                }

                // This next if check should be true, because we wouldn't get max_results otherwise
                if let Some(iter_addrs) = prefilter_mask.iter_addrs() {
                    // Emit the prefilter rows that the partition search did not reach.
                    //
                    // The prefilter mask is shared by all deltas. When a per-segment
                    // restriction is in effect (`seg_mask` is `Some`) each delta emits only
                    // the addresses its own segment owns; the segments partition the
                    // fragments, so each address is emitted by exactly one delta. Without a
                    // restriction the mask is global, so only the first delta emits (guarded
                    // by a shared flag) to avoid duplicating rows across deltas.
                    let should_emit = seg_mask.is_some()
                        || state
                            .took_no_rows_shortcut
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok();
                    if should_emit {
                        let initial_addrs = state.initial_ids.lock().unwrap();
                        let found_addrs = HashSet::<_>::from_iter(initial_addrs.iter().copied());
                        drop(initial_addrs);
                        let not_found_addrs = UInt64Array::from_iter_values(
                            iter_addrs.map(u64::from).filter(|addr| {
                                !found_addrs.contains(addr)
                                    && seg_mask.as_ref().is_none_or(|m| m.selected(*addr))
                            }),
                        );
                        let not_found_distance =
                            Float32Array::from_value(f32::INFINITY, not_found_addrs.len());
                        let not_found_batch = RecordBatch::try_new(
                            KNN_INDEX_SCHEMA.clone(),
                            vec![Arc::new(not_found_distance), Arc::new(not_found_addrs)],
                        )
                        .unwrap();
                        return futures::stream::once(async move { Ok(not_found_batch) }).boxed();
                    } else {
                        // We meet all the criteria for an early exit, but we aren't the first
                        // delta and the mask is global, so skip to avoid duplicate rows.
                        return futures::stream::empty().boxed();
                    }
                }
            }

            // Stop searching if we have k results or we've found all the results
            // that could possible match the prefilter
            let max_results = max_results.unwrap_or(usize::MAX).min(query.k);

            let state_clone = state.clone();

            let query_parallelism =
                effective_query_parallelism(&query, index.as_ref(), target_partitions);
            if query_parallelism <= 1 {
                return stream::once(async move {
                    let prefilter: Arc<dyn PreFilter> = prefilter;
                    let index_metrics: Arc<dyn MetricsCollector> =
                        Arc::new(metrics.index_metrics.clone());
                    let stream = index
                        .search_partitions(
                            query,
                            partitions,
                            q_c_dists,
                            min_nprobes,
                            max_nprobes,
                            prefilter,
                            Some(Arc::new(LatePartitionSearchControl {
                                state: state.clone(),
                                max_results,
                                seg_mask: seg_mask.clone(),
                            })),
                            index_metrics,
                        )
                        .await
                        .map_err(|e| {
                            DataFusionError::Execution(format!("Failed to search partitions: {e}"))
                        })?;
                    Ok::<stream::BoxStream<'static, DataFusionResult<RecordBatch>>, DataFusionError>(
                        Self::instrument_sequential_partition_stream(
                            stream.boxed(),
                            metrics,
                            state,
                            false,
                            true,
                            seg_mask,
                        ),
                    )
                })
                .try_flatten()
                .boxed();
            }

            futures::stream::iter(min_nprobes..max_nprobes)
                .map(move |idx| {
                    let part_id = partitions.value(idx);
                    let mut query = query.clone();
                    query.dist_q_c = q_c_dists.value(idx);
                    let metrics = metrics.clone();
                    let pre_filter = prefilter.clone();
                    let state = state.clone();
                    let index = index.clone();
                    let seg_mask = seg_mask.clone();
                    async move {
                        metrics.partitions_searched.add(1);
                        let batch = Self::search_partition(
                            index,
                            query,
                            part_id as usize,
                            pre_filter,
                            metrics,
                            seg_mask,
                        )
                        .await?;
                        state.record_late_batch(batch.num_rows());
                        Ok(batch)
                    }
                })
                .take_while(move |_| {
                    let found_so_far = state_clone.num_results_found.load(Ordering::Relaxed);
                    std::future::ready(found_so_far < max_results)
                })
                .buffered(query_parallelism)
                .boxed()
        });
        stream.flatten()
    }

    #[allow(clippy::too_many_arguments)]
    fn initial_search(
        index: Arc<dyn VectorIndex>,
        query: Query,
        partitions: Arc<UInt32Array>,
        q_c_dists: Arc<Float32Array>,
        prefilter: Arc<dyn PreFilter>,
        metrics: Arc<AnnIndexMetrics>,
        state: Arc<ANNIvfEarlySearchResults>,
        target_partitions: usize,
        seg_mask: Option<Arc<RowAddrMask>>,
    ) -> impl Stream<Item = DataFusionResult<RecordBatch>> {
        let minimum_nprobes = query.minimum_nprobes.min(partitions.len());

        let query_parallelism =
            effective_query_parallelism(&query, index.as_ref(), target_partitions);
        if query_parallelism <= 1 {
            metrics.partitions_searched.add(minimum_nprobes);
            return stream::once(async move {
                let prefilter: Arc<dyn PreFilter> = prefilter;
                let index_metrics: Arc<dyn MetricsCollector> =
                    Arc::new(metrics.index_metrics.clone());
                let stream = index
                    .search_partitions(
                        query,
                        partitions,
                        q_c_dists,
                        0,
                        minimum_nprobes,
                        prefilter,
                        None,
                        index_metrics,
                    )
                    .await
                    .map_err(|e| {
                        DataFusionError::Execution(format!("Failed to search partitions: {e}"))
                    })?;
                Ok::<stream::BoxStream<'static, DataFusionResult<RecordBatch>>, DataFusionError>(
                    Self::instrument_sequential_partition_stream(
                        stream.boxed(),
                        metrics,
                        state,
                        true,
                        false,
                        seg_mask,
                    ),
                )
            })
            .try_flatten()
            .boxed();
        }

        metrics.partitions_searched.add(minimum_nprobes);
        futures::stream::iter(0..minimum_nprobes)
            .map(move |idx| {
                let part_id = partitions.value(idx);
                let mut query = query.clone();
                query.dist_q_c = q_c_dists.value(idx);
                let metrics = metrics.clone();
                let index = index.clone();
                let pre_filter = prefilter.clone();
                let state = state.clone();
                let seg_mask = seg_mask.clone();
                async move {
                    let batch = Self::search_partition(
                        index,
                        query,
                        part_id as usize,
                        pre_filter,
                        metrics,
                        seg_mask,
                    )
                    .await?;
                    state.record_batch(&batch)?;
                    Ok(batch)
                }
            })
            .buffered(query_parallelism)
            .boxed()
    }

    /// Recover rows that match a bounded, selective prefilter but have no index entry
    /// (their vector is null or an empty multivector), which the covered search never
    /// returns. The non-covered path recovers them via its not-found shortcut with
    /// `INFINITY` distance; a covered index must do the same but also supply the covering
    /// columns, which it fetches from the base table for these few missing rows. Only
    /// rows selected by an arming delta's ownership scope in `scopes` are recovered
    /// (`None` = that delta owned the whole global mask): a delta that skipped its late
    /// search without arming never emits its not-found rows on the non-covered path, so
    /// recovering them here would make the covered result set diverge. Returns `None`
    /// when there is nothing to recover (non-covered, unbounded/non-selective prefilter,
    /// or every prefilter row was already emitted or unowned).
    async fn covered_not_found_batch(
        has_covered: bool,
        prefilter: Arc<DatasetPreFilter>,
        dataset: Arc<Dataset>,
        output_schema: SchemaRef,
        k: usize,
        emitted: Arc<std::sync::Mutex<roaring::RoaringTreemap>>,
        scopes: Vec<Option<Arc<RowAddrMask>>>,
    ) -> DataFusionResult<Option<RecordBatch>> {
        if !has_covered {
            return Ok(None);
        }
        prefilter
            .wait_for_ready()
            .await
            .map_err(|e| DataFusionError::Execution(format!("prefilter not ready: {e}")))?;
        let mask = prefilter.mask();
        // Only a bounded, selective prefilter (at most k allowed rows) guarantees every
        // matching row belongs in the result; an unbounded prefilter has no not-found set.
        let Some(max_len) = mask.max_len() else {
            return Ok(None);
        };
        if max_len as usize > k {
            return Ok(None);
        }
        let Some(addrs) = mask.iter_addrs() else {
            return Ok(None);
        };
        let not_found: Vec<u64> = {
            let emitted = emitted.lock().unwrap();
            addrs
                .map(u64::from)
                .filter(|addr| !emitted.contains(*addr))
                .filter(|addr| {
                    // Mirror the non-covered shortcut's per-segment ownership rule:
                    // each armed delta may recover only the rows its segment owns.
                    scopes
                        .iter()
                        .any(|scope| scope.as_ref().is_none_or(|m| m.selected(*addr)))
                })
                .collect()
        };
        if not_found.is_empty() {
            return Ok(None);
        }

        // Fetch the covering columns (the fields after `[_distance, _rowid]`) from the
        // base table for the missing rows; the take is bounded by `not_found.len()`.
        // Carry `_rowid` through as well: the take resolves ids with
        // `MissingRowPolicy::Ignore` and silently returns FEWER rows than requested when an
        // id no longer resolves through the row-id index -- a stale prefilter listing rows
        // of a fragment a later delete removed entirely. Rebuilding the batch from
        // `not_found` would then pair N distance/rowid values with < N covering values and
        // fail with "all columns in a record batch must have the same length". Carrying
        // `_rowid` lets the batch be rebuilt from the rows the take actually returned,
        // which is what the non-covered path does when its `TakeExec` drops the same ids.
        //
        // A row that still resolves but is tombstoned would shorten the batch the same way.
        // That is not reachable today, so the handling below is defensive for that case, not
        // load-bearing. The invariant is enforced by `DatasetPreFilter`, not by the individual
        // prefilter sources: its `deleted_ids` mask (`prefilter.rs`) is intersected into the final
        // mask and covers every fragment in the manifest -- fragments inside the index bitmap
        // contribute their deletion vectors, fragments outside it are blocked wholesale, and
        // the mask is elided only when no fragment carries a deletion file at all. So a
        // tombstoned address never enters `mask` regardless of which source produced it.
        // Check that invariant there, not here, when adding a new `PreFilterSource`.
        //
        // `include_row_id()`, NOT `_rowid` in the projected column list: naming it there
        // sets `must_add_row_offset`, which both arms the take's "must not target deleted
        // rows" check (turning any shortfall into a hard error) and computes a row-offset
        // column per batch that `project_batch` then discards.
        let covered_names: Vec<String> = output_schema
            .fields()
            .iter()
            .skip(2)
            .map(|field| field.name().clone())
            .collect();
        let projection = ProjectionRequest::from_columns(covered_names, dataset.schema());
        // `not_found` holds row ids (the `_rowid` space): stable logical ids on a
        // stable-row-id dataset, physical addresses otherwise. `try_new_from_ids`
        // resolves them through the row-id index when one exists and falls back to
        // identity (id == address) when it does not, so it is correct in both cases --
        // unlike `try_new_from_addresses`, which would misread a stable id as
        // `frag = id >> 32, offset = id` and take the wrong physical row (or error).
        let covered = TakeBuilder::try_new_from_ids(dataset, not_found, projection)?
            .include_row_id()
            .execute()
            .await?;

        // Realign on what came back, not on what was asked for.
        let returned_ids = covered
            .column_by_name(ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "covered recovery take did not return the requested {ROW_ID} column"
                ))
            })?
            .clone();
        let n = covered.num_rows();
        if n == 0 {
            return Ok(None);
        }
        // Assemble by name. `include_row_id` appends `_rowid` to the take's output rather
        // than placing it first, so no position in `covered` is contractual; the declared
        // `[_distance, _rowid, <covering...>]` order comes from `output_schema` alone.
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
        columns.push(Arc::new(Float32Array::from_value(f32::INFINITY, n)));
        columns.push(returned_ids);
        for field in output_schema.fields().iter().skip(2) {
            let column = covered.column_by_name(field.name()).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "covered recovery take did not return the requested column '{}'",
                    field.name()
                ))
            })?;
            columns.push(column.clone());
        }
        let batch = RecordBatch::try_new(output_schema.clone(), columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
        Ok(Some(batch))
    }
}

impl ExecutionPlan for ANNIvfSubIndexExec {
    fn name(&self) -> &str {
        "ANNSubIndexExec"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![&self.input],
            PreFilterSource::FilteredRowIds(src) => vec![&self.input, &src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&self.input, &src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = if children.len() == 1 || children.len() == 2 {
            let prefilter_source = if children.len() == 2 {
                let prefilter = children.pop().expect("length checked");
                match &self.prefilter_source {
                    PreFilterSource::None => PreFilterSource::None,
                    PreFilterSource::FilteredRowIds(_) => {
                        PreFilterSource::FilteredRowIds(prefilter)
                    }
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(prefilter)
                    }
                }
            } else {
                self.prefilter_source.clone()
            };

            Self {
                input: children.pop().expect("length checked"),
                dataset: self.dataset.clone(),
                indices: self.indices.clone(),
                query: self.query.clone(),
                prefilter_source,
                overlay_block: self.overlay_block.clone(),
                external_mask: self.external_mask.clone(),
                output_schema: self.output_schema.clone(),
                has_covered: self.has_covered,
                properties: self.properties.clone(),
                metrics: ExecutionPlanMetricsSet::new(),
            }
        } else {
            return Err(DataFusionError::Internal(
                "ANNSubIndexExec node must have exactly one or two (prefilter) child".to_string(),
            ));
        };
        Ok(Arc::new(plan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema();
        // When the index emits covering columns, the late-search no-rows shortcut
        // (which emits only [_distance, _rowid]) is disabled in favour of the
        // end-of-stream covered recovery. Resolved at construction; see the field.
        let has_covered = self.has_covered;
        let target_partitions = context.session_config().target_partitions();
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let column = self.query.column.clone();
        let indices = self.indices.clone();
        // Per-segment fragment restriction, applied to every partition result before
        // the shared search accounting sees it. Only enabled when every segment has a
        // fragment_bitmap, mirroring the `all_have_bitmaps` gate in
        // DatasetPreFilter::new so we never restrict more aggressively than the
        // shared prefilter's fallback.
        let segment_bitmaps: Arc<HashMap<Uuid, RoaringBitmap>> =
            Arc::new(if indices.iter().all(|idx| idx.fragment_bitmap.is_some()) {
                indices
                    .iter()
                    .map(|idx| (idx.uuid, idx.fragment_bitmap.clone().unwrap()))
                    .collect()
            } else {
                HashMap::new()
            });
        let prefilter_source = self.prefilter_source.clone();
        let metrics = Arc::new(AnnIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let timer = Instant::now();

        // Per-delta-index stream:
        //   Stream<(parttitions, index uuid)>
        let per_index_stream = input_stream
            .and_then(move |batch| {
                let part_id_col = batch.column_by_name(PART_ID_COLUMN).unwrap_or_else(|| {
                    panic!("ANNSubIndexExec: input missing {} column", PART_ID_COLUMN)
                });
                let part_id_arr = part_id_col.as_list::<i32>().clone();
                let dist_q_c_col = batch.column_by_name(DIST_Q_C_COLUMN).unwrap_or_else(|| {
                    panic!("ANNSubIndexExec: input missing {} column", DIST_Q_C_COLUMN)
                });
                let dist_q_c_arr = dist_q_c_col.as_list::<i32>().clone();
                let index_uuid_col = batch.column_by_name(INDEX_UUID_COLUMN).unwrap_or_else(|| {
                    panic!(
                        "ANNSubIndexExec: input missing {} column",
                        INDEX_UUID_COLUMN
                    )
                });
                let index_uuid = index_uuid_col.as_string::<i32>().clone();

                let plan: Vec<DataFusionResult<(_, _, _)>> = part_id_arr
                    .iter()
                    .zip(dist_q_c_arr.iter())
                    .zip(index_uuid.iter())
                    .map(|((part_id, dist_q_c), uuid)| {
                        let partitions =
                            Arc::new(part_id.unwrap().as_primitive::<UInt32Type>().clone());
                        let dist_q_c =
                            Arc::new(dist_q_c.unwrap().as_primitive::<Float32Type>().clone());
                        let uuid = Uuid::parse_str(uuid.unwrap()).map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Invalid UUID in __index_uuid column: {e}"
                            ))
                        })?;
                        Ok((partitions, dist_q_c, uuid))
                    })
                    .collect_vec();
                async move { DataFusionResult::Ok(stream::iter(plan)) }
            })
            .try_flatten();
        let prefilter_loader = match &prefilter_source {
            PreFilterSource::FilteredRowIds(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(FilteredRowIdsToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::ScalarIndexQuery(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(SelectionVectorToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::None => None,
        };

        // AND the external row-address mask into whatever the filter produced.
        let prefilter_loader = match self.external_mask.clone() {
            Some(mask) => {
                Some(Box::new(MaskAndLoader::new(mask, prefilter_loader)) as Box<dyn FilterLoader>)
            }
            None => prefilter_loader,
        };
        let pre_filter = {
            let mut pf = DatasetPreFilter::new(ds.clone(), &indices, prefilter_loader);
            if let Some(block) = self.overlay_block.clone() {
                pf = pf.with_overlay_block(block);
            }
            Arc::new(pf)
        };
        let indices_by_uuid = Arc::new(
            indices
                .iter()
                .cloned()
                .map(|index| (index.uuid, index))
                .collect::<HashMap<_, _>>(),
        );

        let state = Arc::new(ANNIvfEarlySearchResults::new(indices.len(), query.k));

        // Covered null-vector recovery: track which row ids the search emits, then after
        // it finishes emit any prefilter-matched rows that had no index entry, with their
        // covering columns taken from the base table (see `covered_not_found_batch`).
        // This must NOT be gated on `prefilter_source`: `DatasetPreFilter` builds a bounded
        // allow-list from the deletion mask alone, so an unfiltered query over a dataset with
        // deletions (or a fragment outside the index bitmap) still arms recovery. Skipping the
        // bookkeeping there left `emitted` empty, and recovery re-emitted every live row on top
        // of the ones the search had already returned. The cost is already confined to covered
        // plans, and recovery itself stays gated behind `has_covered`.
        let track_emitted = has_covered;
        let emitted_row_ids = Arc::new(std::sync::Mutex::new(roaring::RoaringTreemap::new()));
        let emitted_for_inspect = emitted_row_ids.clone();
        let recon_prefilter = pre_filter.clone();
        let recon_state = state.clone();
        let recon_ds = ds.clone();
        let recon_schema = schema.clone();
        let recon_k = query.k;
        let declared_schema = schema.clone();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            per_index_stream
                .and_then(move |(part_ids, q_c_dists, index_uuid)| {
                    let ds = ds.clone();
                    let column = column.clone();
                    let metrics = metrics.clone();
                    let pre_filter = pre_filter.clone();
                    let indices_by_uuid = indices_by_uuid.clone();
                    let state = state.clone();
                    let segment_bitmaps = segment_bitmaps.clone();
                    let mut query = query.clone();
                    let pruned_nprobes = early_pruning(q_c_dists.values(), query.k);
                    adjust_probes(&mut query, pruned_nprobes);
                    async move {
                        let index_metadata = indices_by_uuid.get(&index_uuid).ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "ANNSubIndexExec: input referenced unknown index segment {index_uuid}"
                            ))
                        })?;
                        let segment_pre_filter = prefilter_for_segment(
                            ds.clone(),
                            index_metadata,
                            pre_filter.clone(),
                        )
                        .await?;
                        let raw_index = ds
                            .open_vector_index(&column, &index_uuid, &metrics.index_metrics)
                            .await?;
                        let query = normalize_query_for_index(raw_index.as_ref(), query)?;

                        // A segment's index file may still physically contain rows for
                        // fragments that were pruned from its fragment_bitmap (e.g. after an
                        // in-place column update via update_columns). Once a newer delta
                        // segment owns such a fragment, the stale rows in this segment must
                        // not be returned, otherwise the same row is emitted by two segments.
                        // Build a per-segment restriction mask, reusing the scheme-aware
                        // helper so it is correct for both row-address and stable-row-id
                        // datasets. The shared prefilter is built from the union of all
                        // segment bitmaps and cannot express this per-segment rule.
                        let seg_mask = match segment_bitmaps.get(&index_uuid).cloned() {
                            Some(bitmap) => {
                                match DatasetPreFilter::create_restricted_deletion_mask(
                                    ds.clone(),
                                    bitmap,
                                ) {
                                    Some(fut) => Some(fut.await?),
                                    None => None,
                                }
                            }
                            None => None,
                        };

                        let early_search = Self::initial_search(
                            raw_index.clone(),
                            query.clone(),
                            part_ids.clone(),
                            q_c_dists.clone(),
                            segment_pre_filter.clone(),
                            metrics.clone(),
                            state.clone(),
                            target_partitions,
                            seg_mask.clone(),
                        );
                        let late_search = Self::late_search(
                            raw_index.clone(),
                            query,
                            part_ids,
                            q_c_dists,
                            segment_pre_filter,
                            pre_filter,
                            metrics,
                            state,
                            target_partitions,
                            seg_mask,
                            has_covered,
                        );
                        DataFusionResult::Ok(early_search.chain(late_search).boxed())
                    }
                })
                // Must use flatten_unordered to avoid deadlock.
                // Each delta stream is split into an early and late search.  The late search
                // will not start until the early search is complete across all deltas.
                .try_flatten_unordered(None)
                // The search emits covering columns read from index STORAGE, while this node
                // declares them from the manifest (`IndexMetadata.covering_fields`). The two can
                // disagree: `FLAG_COVERED_INDEX_METADATA` is a best-effort writer fence, so a
                // client predating it can drop the declaration while the index files keep the
                // payload. Without narrowing, this stream mixes batch widths -- a wider search
                // batch next to the narrower non-covered shortcut batch -- which panics inside
                // DataFusion's TopK (`interleave_record_batch` reads every batch through the first
                // batch's schema). Only ever narrows: the opposite desync (declared but absent from
                // storage) is prevented at build time (covering requires the V3 index format and
                // rejects columns the storage cannot emit) and otherwise fails loudly, which is the
                // right outcome.
                //
                // Caveat worth knowing: the test below is cardinality-based, so it cannot detect a
                // desync in which storage holds the same *number* of covering columns as the
                // declaration but different ones.
                .map(move |batch| {
                    let batch = batch?;
                    if batch.num_columns() > declared_schema.fields().len() {
                        return batch
                            .project_by_schema(declared_schema.as_ref())
                            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None));
                    }
                    Ok(batch)
                })
                .inspect_ok(move |batch| {
                    // Record emitted row ids so the reconciliation below can find the
                    // prefilter rows the search never returned (no index entry).
                    if track_emitted && let Some(col) = batch.column_by_name(ROW_ID) {
                        emitted_for_inspect
                            .lock()
                            .unwrap()
                            .extend(col.as_primitive::<UInt64Type>().values().iter().copied());
                    }
                })
                .chain(
                    stream::once(async move {
                        // Only recover when a covered late search actually took the
                        // no-rows shortcut; otherwise the non-covered path would not
                        // have emitted these rows either, and covered must match.
                        // Each armed delta contributes its ownership scope, and the
                        // recovery emits only rows those scopes select.
                        let scopes = std::mem::take(
                            &mut *recon_state.covered_recovery_scopes.lock().unwrap(),
                        );
                        if scopes.is_empty() {
                            return Ok(None);
                        }
                        Self::covered_not_found_batch(
                            has_covered,
                            recon_prefilter,
                            recon_ds,
                            recon_schema,
                            recon_k,
                            emitted_row_ids,
                            scopes,
                        )
                        .await
                    })
                    .try_filter_map(|opt| future::ready(Ok(opt))),
                )
                .finally(move || {
                    // Publish the exact index-file I/O measured for this query
                    // (cache misses only) to the iops/requests/bytes_read gauges.
                    metrics_clone.index_metrics.flush_io();
                    metrics_clone
                        .baseline_metrics
                        .elapsed_compute()
                        .add_duration(timer.elapsed());
                    metrics_clone.baseline_metrics.done();
                })
                .boxed(),
        )))
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> DataFusionResult<Arc<datafusion::physical_plan::Statistics>> {
        Ok(Arc::new(Statistics {
            num_rows: Precision::Exact(
                self.query.k
                    * self.query.refine_factor.unwrap_or(1) as usize
                    * self
                        .input
                        .partition_statistics(partition)?
                        .num_rows
                        .get_value()
                        .unwrap_or(&1),
            ),
            ..Statistics::new_unknown(self.schema().as_ref())
        }))
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

fn adjust_probes(query: &mut Query, pruned_nprobes: usize) {
    query.minimum_nprobes = query.minimum_nprobes.max(pruned_nprobes);
    if let Some(maximum) = query.maximum_nprobes
        && query.minimum_nprobes > maximum
    {
        query.minimum_nprobes = maximum;
    }
}

fn early_pruning(dists: &[f32], k: usize) -> usize {
    if dists.is_empty() {
        return 0;
    }

    const PRUNING_FACTORS: [f32; 3] = [0.6, 7.0, 81.0];
    let factor = match k {
        ..=1 => PRUNING_FACTORS[0],
        2..=10 => PRUNING_FACTORS[1],
        11.. => PRUNING_FACTORS[2],
    };
    let dist_threshold = dists[0] * factor;
    dists.partition_point(|dist| *dist <= dist_threshold)
}

#[derive(Debug)]
pub struct MultivectorScoringExec {
    // the inputs are sorted ANN search results
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    query: Query,
    /// Output schema: `[_distance, _rowid]` plus any covering columns the ANN children
    /// emit. The covered payload is per source row (replicated across its sub-vectors),
    /// so it is carried through the per-row-id reduction below.
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl MultivectorScoringExec {
    pub fn try_new(inputs: Vec<Arc<dyn ExecutionPlan>>, query: Query) -> Result<Self> {
        // Inherit the children's schema so covering columns flow through scoring instead
        // of being dropped (a covered multivector query would otherwise still take from
        // the base table).
        let output_schema = inputs
            .first()
            .map(|input| input.schema())
            .unwrap_or_else(|| KNN_INDEX_SCHEMA.clone());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));

        Ok(Self {
            inputs,
            query,
            output_schema,
            properties,
        })
    }
}

impl DisplayAs for MultivectorScoringExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MultivectorScoring: k={}", self.query.k)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "MultivectorScoring\nk={}", self.query.k)
            }
        }
    }
}

impl ExecutionPlan for MultivectorScoringExec {
    fn name(&self) -> &str {
        "MultivectorScoringExec"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.  It must be
        // run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = Self::try_new(children, self.query.clone())?;
        Ok(Arc::new(plan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let inputs = self
            .inputs
            .iter()
            .map(|input| input.execute(partition, context.clone()))
            .collect::<DataFusionResult<Vec<_>>>()?;

        // Covering columns (if any) follow `[_distance, _rowid]` in the child schema.
        // They are per source row -- identical across a row's sub-vectors -- so we carry
        // them through the per-row-id reduction and re-attach one copy per output row.
        let output_schema = self.output_schema.clone();
        let n_covered = output_schema.fields().len().saturating_sub(2);

        // collect the top k results from each stream,
        // and max-reduce for each query,
        // records the minimum distance for each query as estimation.
        let mut reduced_inputs = stream::select_all(inputs.into_iter().map(|stream| {
            let output_schema = output_schema.clone();
            stream.map(move |batch| {
                let batch = batch?;
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let dists = batch[DIST_COL].as_primitive::<Float32Type>();
                debug_assert_eq!(dists.null_count(), 0);

                // max-reduce for the same row id
                let min_sim = dists
                    .values()
                    .last()
                    .map(|dist| 1.0 - *dist)
                    .unwrap_or_default();
                let mut new_row_ids = Vec::with_capacity(row_ids.len());
                let mut new_sims = Vec::with_capacity(row_ids.len());
                let mut keep_indices: Vec<u32> = Vec::with_capacity(row_ids.len());
                let mut visited_row_ids = HashSet::with_capacity(row_ids.len());

                for (i, (row_id, dist)) in row_ids
                    .values()
                    .iter()
                    .zip(dists.values().iter())
                    .enumerate()
                {
                    // the results are sorted by distance, so we can skip if we have seen this row id before
                    if visited_row_ids.contains(row_id) {
                        continue;
                    }
                    visited_row_ids.insert(row_id);
                    new_row_ids.push(*row_id);
                    // it's cosine distance, so we need to convert it to similarity
                    new_sims.push(1.0 - *dist);
                    keep_indices.push(i as u32);
                }

                let mut columns: Vec<ArrayRef> = vec![
                    Arc::new(Float32Array::from(new_sims)),
                    Arc::new(UInt64Array::from(new_row_ids)),
                ];
                if n_covered > 0 {
                    let keep = arrow_array::UInt32Array::from(keep_indices);
                    for covered_col in batch.columns().iter().skip(2) {
                        columns.push(arrow::compute::take(covered_col, &keep, None)?);
                    }
                }
                let batch = RecordBatch::try_new(output_schema.clone(), columns)?;

                Ok::<_, DataFusionError>((min_sim, batch))
            })
        }));

        let k = self.query.k;
        let refactor = self.query.refine_factor.unwrap_or(1) as usize;
        let num_queries = self.inputs.len() as f32;
        let stream = stream::once(async move {
            // at most, we will have k * refine_factor results for each query
            let mut results = HashMap::with_capacity(k * refactor);
            // Covering columns of every reduced batch, kept alive (cheap Arc clones)
            // for one zero-copy-per-cell `interleave` gather at emit. The map records
            // where each row id's covering values live: (batch position, row position),
            // first-seen (the values are per source row, identical across batches).
            // Materializing per-cell `ScalarValue`s here instead would copy every
            // candidate cell once into the map and again through `iter_to_array` --
            // several MB per query for wide payloads.
            let mut covered_batches: Vec<Vec<ArrayRef>> = Vec::new();
            let mut payload_locs: HashMap<u64, (usize, usize)> = HashMap::new();
            let mut missed_sim_sum = 0.0;
            while let Some((min_sim, batch)) = reduced_inputs.try_next().await? {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let sims = batch[DIST_COL].as_primitive::<Float32Type>();

                if n_covered > 0 {
                    let batch_pos = covered_batches.len();
                    for (i, row_id) in row_ids.values().iter().enumerate() {
                        payload_locs.entry(*row_id).or_insert((batch_pos, i));
                    }
                    covered_batches.push(batch.columns().iter().skip(2).cloned().collect());
                }

                let query_results = row_ids
                    .values()
                    .iter()
                    .copied()
                    .zip(sims.values().iter().copied())
                    .collect::<HashMap<_, _>>();

                // for a row `r`:
                // if `r` is in only `results``, then `results[r] += min_sim`
                // if `r` is in only `query_results`, then `results[r] = query_results[r] + missed_similarities`,
                // here `missed_similarities` is the sum of `min_sim` from previous iterations
                // if `r` is in both, then `results[r] += query_results[r]`
                results.iter_mut().for_each(|(row_id, sim)| {
                    if let Some(new_dist) = query_results.get(row_id) {
                        *sim += new_dist;
                    } else {
                        *sim += min_sim;
                    }
                });
                query_results.into_iter().for_each(|(row_id, sim)| {
                    results.entry(row_id).or_insert(sim + missed_sim_sum);
                });
                missed_sim_sum += min_sim;
            }

            let (row_ids, sims): (Vec<u64>, Vec<f32>) = results.into_iter().unzip();
            let dists = sims
                .iter()
                // it's similarity, so we need to convert it back to distance
                .map(|sim| num_queries - sim)
                .collect::<Vec<_>>();
            let mut columns: Vec<ArrayRef> = vec![
                Arc::new(Float32Array::from(dists)),
                Arc::new(UInt64Array::from(row_ids.clone())),
            ];
            if n_covered > 0 && !row_ids.is_empty() {
                // Every emitted row id was recorded while scanning the reduced
                // batches (both maps are fed from the same iteration), so a miss
                // here is a results/payloads desync. Fail loudly rather than
                // fall back to nulls: a silent null would masquerade as a real
                // covered value and hide the desync from every result-based test.
                let locations = row_ids
                    .iter()
                    .map(|row_id| {
                        payload_locs.get(row_id).copied().ok_or_else(|| {
                            DataFusionError::Internal(format!(
                                "multivector covering: row id {row_id} missing from \
                                 covering source"
                            ))
                        })
                    })
                    .collect::<DataFusionResult<Vec<_>>>()?;
                for covered_pos in 0..n_covered {
                    let arrays: Vec<&dyn Array> = covered_batches
                        .iter()
                        .map(|cols| cols[covered_pos].as_ref())
                        .collect();
                    columns.push(arrow::compute::interleave(&arrays, &locations)?);
                }
            } else {
                for covered_pos in 0..n_covered {
                    let field = output_schema.field(2 + covered_pos);
                    columns.push(arrow::array::new_empty_array(field.data_type()));
                }
            }
            let batch = RecordBatch::try_new(output_schema.clone(), columns)?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::any::Any;

    use crate::index::DatasetIndexExt;
    use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
    use arrow::datatypes::Float32Type;
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatchIterator,
        RecordBatchReader, StringArray, StructArray,
    };
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use async_trait::async_trait;
    use datafusion::error::Result as DataFusionResult;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
    use lance_datafusion::utils::FIND_PARTITIONS_ELAPSED_METRIC;
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::optimize::OptimizeOptions;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::{DEFAULT_QUERY_PARALLELISM, PreparedPartitionSearchHandle};
    use lance_index::{Index, IndexType};
    use lance_io::traits::Reader;
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::index::vector::ivf::v2::STREAMING_SEARCH_BATCH_SIZE;
    use crate::io::exec::testing::TestingExec;

    fn base_query() -> Query {
        Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0f32])) as ArrayRef,
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: None,
        }
    }

    fn covering_field(id: i32, name: &str, data_type: DataType) -> (i32, Field) {
        (id, Field::new(name, data_type, true))
    }

    #[test]
    fn test_common_physical_covering_requires_every_segment_to_prove_the_field() {
        let price = covering_field(17, "price", DataType::Int64);
        let payload = covering_field(23, "payload", DataType::Utf8);

        let common = common_physical_covering(
            vec![price.clone(), payload.clone()],
            &[vec![price.clone(), payload.clone()], vec![price.clone()]],
        );

        assert_eq!(common, vec![price.clone()]);

        let common = common_physical_covering(
            vec![price.clone(), payload.clone()],
            &[vec![payload, price.clone()], vec![price.clone()]],
        );
        assert_eq!(
            common,
            vec![price],
            "order is checked after finding the plan-wide subset"
        );
    }

    #[test]
    fn test_common_physical_covering_rejects_wrong_identity_shape_and_order() {
        let price = covering_field(17, "price", DataType::Int64);
        let payload = covering_field(23, "payload", DataType::Utf8);

        assert!(
            common_physical_covering(
                vec![price.clone()],
                &[vec![covering_field(99, "price", DataType::Int64)]],
            )
            .is_empty(),
            "a matching name and type cannot substitute for the source field id"
        );
        assert!(
            common_physical_covering(
                vec![price.clone()],
                &[vec![covering_field(17, "price", DataType::UInt64)]],
            )
            .is_empty(),
            "a matching source field id cannot substitute for the physical Arrow type"
        );
        assert!(
            common_physical_covering(
                vec![price.clone(), payload.clone()],
                &[vec![payload, price]],
            )
            .is_empty(),
            "a physical order that disagrees with the declared output schema is not servable"
        );
    }

    /// The manifest remains authoritative for logical dependencies: if an index declares
    /// a covering field id the current schema cannot resolve, construction must fail loudly
    /// rather than hiding schema/metadata corruption behind physical fallback.
    #[tokio::test]
    async fn test_ann_sub_index_rejects_unresolvable_covering_field() {
        use lance_datafusion::exec::OneShotExec;

        // Minimal input plan carrying the required PART_ID_COLUMN schema.
        let input = Arc::new(OneShotExec::from_batch(RecordBatch::new_empty(
            KNN_PARTITION_SCHEMA.clone(),
        )));

        // A dataset whose schema has no field with id 9999.
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                "memory://d2-unresolvable-covering",
                None,
            )
            .await
            .unwrap(),
        );

        // A legal covering shape (`covering_fields` is the trailing slice of `fields`,
        // which keeps at least one keyed entry -- see
        // `IndexMetadata::validate_covering_fields`) whose carried id 9999 the schema
        // cannot resolve. Keeping the shape legal is what makes this a test of the
        // resolution failure rather than of metadata no commit can produce.
        let index = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![0, 9999],
            name: "vector_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::new()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            covering_fields: vec![9999],
        };
        index
            .validate_covering_fields()
            .expect("the fixture must be a legal covering declaration");

        let result = ANNIvfSubIndexExec::try_new(
            input,
            dataset,
            vec![index],
            base_query(),
            PreFilterSource::None,
        );
        let err = result.expect_err("unresolvable covering field id must be rejected");
        assert!(
            err.to_string().contains("covering field id 9999"),
            "expected a manifest/schema inconsistency error, got: {err}"
        );
    }

    /// The exec's declared output schema is the seam through which a query's covering
    /// projection reaches the rest of the plan: `knn_combined` unions the flat paths
    /// against this schema, so whatever is declared here is what every unindexed
    /// fragment pays to read.
    ///
    /// All three states of [`Query::covering_projection`] must be visible in the
    /// declaration, and the `Some(&[])` case must NOT collapse into the `None` case.
    /// The two are indistinguishable in query results -- a covering column is
    /// semantically transparent, so the values simply arrive via a take instead -- which
    /// is why this is asserted on the declared schema rather than on any result.
    #[tokio::test]
    async fn test_ann_sub_index_declares_only_the_projected_covering_columns() {
        use lance_datafusion::exec::OneShotExec;

        let input = Arc::new(OneShotExec::from_batch(RecordBatch::new_empty(
            KNN_PARTITION_SCHEMA.clone(),
        )));

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("vec", DataType::Int32, false),
            ArrowField::new("price", DataType::Int32, true),
            ArrowField::new("payload", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![Some(10), Some(20), Some(30)])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                "memory://d2-projected-covering",
                None,
            )
            .await
            .unwrap(),
        );
        let vec_id = dataset.schema().field("vec").unwrap().id;
        let price_id = dataset.schema().field("price").unwrap().id;
        let payload_id = dataset.schema().field("payload").unwrap().id;

        // Declared in the reverse of field-id order, so the assertions below distinguish
        // declaration order from schema order.
        let index = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![vec_id, payload_id, price_id],
            name: "vector_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::new()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            covering_fields: vec![payload_id, price_id],
        };
        index
            .validate_covering_fields()
            .expect("the fixture must be a legal covering declaration");

        let declared_for = |projection: Option<Vec<String>>| {
            let mut query = base_query();
            query.covering_projection = projection.map(Arc::from);
            let exec = ANNIvfSubIndexExec::try_new_with_resolved_covering(
                input.clone(),
                dataset.clone(),
                vec![index.clone()],
                query,
                PreFilterSource::None,
            )
            .unwrap();
            exec.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            declared_for(None),
            vec![DIST_COL, ROW_ID, "payload", "price"],
            "no projection means no narrowing: every declared covering column"
        );
        assert_eq!(
            declared_for(Some(vec!["price".to_string()])),
            vec![DIST_COL, ROW_ID, "price"],
            "only the covering column this query reads is declared"
        );
        assert_eq!(
            declared_for(Some(vec!["price".to_string(), "payload".to_string()])),
            vec![DIST_COL, ROW_ID, "payload", "price"],
            "covering columns are declared in the index's order, not the request's"
        );
        assert_eq!(
            declared_for(Some(vec![])),
            vec![DIST_COL, ROW_ID],
            "a query needing no covering column must declare a plain-index schema, which \
             is what lets the flat paths skip the covering read entirely"
        );

        let mut unverified_query = base_query();
        unverified_query.covering_projection = Some(Arc::from(["price".to_string()]));
        let unverified = ANNIvfSubIndexExec::try_new(
            input,
            dataset,
            vec![index],
            unverified_query,
            PreFilterSource::None,
        )
        .unwrap();
        assert_eq!(
            unverified
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            vec![DIST_COL, ROW_ID],
            "the synchronous constructor cannot verify storage and must disable covering"
        );
    }

    /// The exec declares one output schema but executes every delta segment, so delta
    /// segments that disagree on covering fields must be rejected up front with a clear
    /// error, not fail later with a stream/plan schema mismatch. (The commit boundary
    /// rejects new mixed sets; this defends against already-committed metadata.)
    #[tokio::test]
    async fn test_ann_sub_index_rejects_mixed_covering_deltas() {
        use lance_datafusion::exec::OneShotExec;

        let input = Arc::new(OneShotExec::from_batch(RecordBatch::new_empty(
            KNN_PARTITION_SCHEMA.clone(),
        )));

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                "memory://d2-mixed-covering-deltas",
                None,
            )
            .await
            .unwrap(),
        );
        let id_field_id = dataset.schema().field("id").unwrap().id;
        // This exec path only resolves `covering_fields` against the live schema, so a
        // key id that is not itself a real schema field is harmless here; it stands in
        // for the vector key column a real covering index would also carry in `fields`.
        let key_field_id = id_field_id + 1;

        // `covering_fields` is always the trailing entries of `fields`.
        let covered_delta = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![key_field_id, id_field_id],
            name: "vector_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(RoaringBitmap::new()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            covering_fields: vec![id_field_id],
        };
        let plain_delta = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            covering_fields: Vec::new(),
            ..covered_delta.clone()
        };

        let result = ANNIvfSubIndexExec::try_new(
            input,
            dataset,
            vec![covered_delta, plain_delta],
            base_query(),
            PreFilterSource::None,
        );
        let err = result.expect_err("delta segments with mixed covering must be rejected");
        assert!(
            err.to_string().contains("disagree on covering fields"),
            "expected a delta covering mismatch error, got: {err}"
        );
    }

    #[test]
    fn test_effective_query_parallelism_clamps_to_cpu_pool() {
        let mut query = base_query();

        query.query_parallelism = -1;
        assert_eq!(effective_query_parallelism_for(&query, 16, 1), 16);

        query.query_parallelism = 0;
        assert_eq!(effective_query_parallelism_for(&query, 16, 1), 1);
        assert_eq!(effective_query_parallelism_for(&query, 16, 8), 8);
        assert_eq!(effective_query_parallelism_for(&query, 16, 128), 16);

        query.query_parallelism = 1;
        assert_eq!(effective_query_parallelism_for(&query, 16, 8), 1);

        query.query_parallelism = 4;
        assert_eq!(effective_query_parallelism_for(&query, 16, 1), 4);

        query.query_parallelism = 128;
        assert_eq!(effective_query_parallelism_for(&query, 16, 1), 16);
    }

    #[test]
    fn test_effective_query_parallelism_respects_target_partitions() {
        // effective_query_parallelism caps cpu_pool_size at target_partitions before
        // passing it to effective_query_parallelism_for, so the ceiling is
        // min(cpu_pool_size, target_partitions).
        let mut query = base_query();
        let cpu_pool_size = 16;

        // use-all-cpus mode: capped at target_partitions
        query.query_parallelism = -1;
        assert_eq!(
            effective_query_parallelism_for(&query, cpu_pool_size.min(4), 1),
            4
        );

        // auto mode: auto_parallelism also clamped to the reduced cpu_pool_size
        query.query_parallelism = 0;
        assert_eq!(
            effective_query_parallelism_for(&query, cpu_pool_size.min(4), 8),
            4
        );

        // explicit parallelism > target_partitions: clamped down
        query.query_parallelism = 16;
        assert_eq!(
            effective_query_parallelism_for(&query, cpu_pool_size.min(4), 1),
            4
        );
    }

    #[derive(Debug, DeepSizeOf)]
    struct ThreadCapturingIndex {
        thread_name: Arc<Mutex<Option<String>>>,
        row_ids: Vec<u64>,
    }

    #[derive(Debug, DeepSizeOf)]
    struct PreparedThreadCapturingIndex {
        prepared_partitions: Arc<Mutex<Vec<usize>>>,
        searched_partitions: Arc<Mutex<Vec<usize>>>,
        search_threads: Arc<Mutex<Vec<String>>>,
        /// The rows each partition returns, indexed by partition id.
        row_ids: Vec<Vec<u64>>,
    }

    #[async_trait]
    impl Index for ThreadCapturingIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait]
    impl VectorIndex for ThreadCapturingIndex {
        async fn search(
            &self,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn find_partitions(&self, _query: &Query) -> Result<(UInt32Array, Float32Array)> {
            let thread_name = std::thread::current()
                .name()
                .unwrap_or("unknown")
                .to_string();
            *self.thread_name.lock().unwrap() = Some(thread_name);
            Ok((UInt32Array::from(vec![0]), Float32Array::from(vec![0.0])))
        }

        fn total_partitions(&self) -> usize {
            1
        }

        async fn search_in_partition(
            &self,
            _partition_id: usize,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn is_loadable(&self) -> bool {
            false
        }

        fn use_residual(&self) -> bool {
            false
        }

        async fn load(
            &self,
            _reader: Arc<dyn Reader>,
            _offset: usize,
            _length: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            unimplemented!()
        }

        fn num_rows(&self) -> u64 {
            0
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
            Box::new(self.row_ids.iter())
        }

        async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
            Ok(())
        }

        async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
            unimplemented!()
        }

        fn ivf_model(&self) -> &lance_index::vector::ivf::storage::IvfModel {
            unimplemented!()
        }

        fn quantizer(&self) -> lance_index::vector::quantizer::Quantizer {
            unimplemented!()
        }

        fn partition_size(&self, _part_id: usize) -> usize {
            unimplemented!()
        }

        fn sub_index_type(
            &self,
        ) -> (
            lance_index::vector::v3::subindex::SubIndexType,
            lance_index::vector::quantizer::QuantizationType,
        ) {
            unimplemented!()
        }

        fn metric_type(&self) -> DistanceType {
            DistanceType::L2
        }
    }

    #[async_trait]
    impl Index for PreparedThreadCapturingIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({}))
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait]
    impl VectorIndex for PreparedThreadCapturingIndex {
        async fn search(
            &self,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn find_partitions(&self, _query: &Query) -> Result<(UInt32Array, Float32Array)> {
            // No additional partitions to reveal; the covered re-rank in
            // late_search treats an empty result as "nothing more to search".
            Ok((
                UInt32Array::from(Vec::<u32>::new()),
                Float32Array::from(Vec::<f32>::new()),
            ))
        }

        fn total_partitions(&self) -> usize {
            self.row_ids.len()
        }

        async fn search_in_partition(
            &self,
            partition_id: usize,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<RecordBatch> {
            // Only the parallel path reaches this. Tests that must stay on the sequential
            // path assert that every partition went through prepare_partition_search,
            // which this entry point never records.
            self.search_prepared_partition(
                Box::new(partition_id),
                &lance_index::metrics::NoOpMetricsCollector,
            )
        }

        async fn prepare_partition_search(
            &self,
            partition_id: usize,
            _query: &Query,
            _pre_filter: Arc<dyn PreFilter>,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<PreparedPartitionSearchHandle> {
            self.prepared_partitions.lock().unwrap().push(partition_id);
            Ok(Box::new(partition_id))
        }

        fn search_prepared_partition(
            &self,
            prepared: PreparedPartitionSearchHandle,
            _metrics: &dyn lance_index::metrics::MetricsCollector,
        ) -> Result<RecordBatch> {
            let partition_id = *prepared.downcast::<usize>().unwrap();
            self.searched_partitions.lock().unwrap().push(partition_id);
            self.search_threads.lock().unwrap().push(
                std::thread::current()
                    .name()
                    .unwrap_or("unknown")
                    .to_string(),
            );
            let row_ids = &self.row_ids[partition_id];
            // Distances stay distinct within a partition so a test can tell whether the
            // distance column survived a filter aligned with its row ids.
            let dists = (0..row_ids.len())
                .map(|offset| partition_id as f32 + offset as f32 * 0.5)
                .collect::<Vec<_>>();
            Ok(RecordBatch::try_new(
                KNN_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(Float32Array::from(dists)),
                    Arc::new(UInt64Array::from(row_ids.clone())),
                ],
            )?)
        }

        fn supports_prepared_partition_search(&self) -> bool {
            true
        }

        #[allow(clippy::too_many_arguments)]
        async fn search_partitions(
            self: Arc<Self>,
            _query: Query,
            partitions: Arc<UInt32Array>,
            _q_c_dists: Arc<Float32Array>,
            start_idx: usize,
            end_idx: usize,
            _pre_filter: Arc<dyn PreFilter>,
            control: Option<Arc<dyn PartitionSearchControl>>,
            _metrics: Arc<dyn lance_index::metrics::MetricsCollector>,
        ) -> Result<SendableRecordBatchStream> {
            let (batch_tx, batch_rx) = mpsc::channel(1);
            let prepared_partition_ids = (start_idx..end_idx)
                .map(|idx| partitions.value(idx) as usize)
                .collect::<Vec<_>>();
            self.prepared_partitions
                .lock()
                .unwrap()
                .extend(prepared_partition_ids.iter().copied());
            // Mirror the production streaming path (v2.rs): search prepared partitions
            // in batches of STREAMING_SEARCH_BATCH_SIZE, one `spawn_cpu` per batch, with
            // the channel send in async code so no CPU-pool thread parks (#7642).
            tokio::spawn(async move {
                for chunk in prepared_partition_ids.chunks(*STREAMING_SEARCH_BATCH_SIZE) {
                    if control
                        .as_ref()
                        .is_some_and(|control| control.should_stop())
                        || batch_tx.is_closed()
                    {
                        return;
                    }
                    let chunk = chunk.to_vec();
                    let index = self.clone();
                    let control_for_search = control.clone();
                    let cancel_probe = batch_tx.clone();
                    let search_output = spawn_cpu(move || {
                        let mut outputs: Vec<DataFusionResult<RecordBatch>> =
                            Vec::with_capacity(chunk.len());
                        let mut stopped = false;
                        for partition_id in chunk {
                            if control_for_search
                                .as_ref()
                                .is_some_and(|control| control.should_stop())
                                || cancel_probe.is_closed()
                            {
                                stopped = true;
                                break;
                            }
                            match index
                                .search_prepared_partition(
                                    Box::new(partition_id),
                                    &lance_index::metrics::NoOpMetricsCollector,
                                )
                                .map_err(datafusion::error::DataFusionError::from)
                            {
                                Ok(batch) => {
                                    if let Some(control) = control_for_search.as_ref() {
                                        control.record_batch(&batch);
                                    }
                                    outputs.push(Ok(batch));
                                }
                                Err(err) => {
                                    outputs.push(Err(err));
                                    stopped = true;
                                    break;
                                }
                            }
                        }
                        Ok::<_, datafusion::error::DataFusionError>((outputs, stopped))
                    })
                    .await;

                    let (outputs, stopped) = match search_output {
                        Ok(output) => output,
                        Err(err) => {
                            let _ = batch_tx.send(Err(err)).await;
                            return;
                        }
                    };
                    for output in outputs {
                        if batch_tx.send(output).await.is_err() {
                            return;
                        }
                    }
                    if stopped {
                        return;
                    }
                }
            });

            Ok(Box::pin(RecordBatchStreamAdapter::new(
                KNN_INDEX_SCHEMA.clone(),
                ReceiverStream::new(batch_rx),
            )))
        }

        fn is_loadable(&self) -> bool {
            false
        }

        fn use_residual(&self) -> bool {
            false
        }

        async fn load(
            &self,
            _reader: Arc<dyn Reader>,
            _offset: usize,
            _length: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            unimplemented!()
        }

        fn num_rows(&self) -> u64 {
            self.row_ids.iter().map(|ids| ids.len() as u64).sum()
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
            Box::new(self.row_ids.iter().flatten())
        }

        async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
            Ok(())
        }

        async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
            unimplemented!()
        }

        fn ivf_model(&self) -> &lance_index::vector::ivf::storage::IvfModel {
            unimplemented!()
        }

        fn quantizer(&self) -> lance_index::vector::quantizer::Quantizer {
            unimplemented!()
        }

        fn partition_size(&self, _part_id: usize) -> usize {
            unimplemented!()
        }

        fn sub_index_type(
            &self,
        ) -> (
            lance_index::vector::v3::subindex::SubIndexType,
            lance_index::vector::quantizer::QuantizationType,
        ) {
            unimplemented!()
        }

        fn metric_type(&self) -> DistanceType {
            DistanceType::L2
        }
    }

    async fn empty_prefilter() -> Arc<DatasetPreFilter> {
        static NEXT_PREFILTER_DATASET_ID: AtomicUsize = AtomicUsize::new(0);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let uri = format!(
            "memory://sequential-prefilter-{}",
            NEXT_PREFILTER_DATASET_ID.fetch_add(1, Ordering::Relaxed)
        );
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
                &uri,
                None,
            )
            .await
            .unwrap(),
        );
        let mut indexed_fragments = RoaringBitmap::new();
        for fragment in dataset.manifest.fragments.iter() {
            indexed_fragments.insert(fragment.id as u32);
        }
        let index = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![],
            covering_fields: vec![],
            name: "test".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(indexed_fragments),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };
        let prefilter = Arc::new(DatasetPreFilter::new(dataset, &[index], None));
        prefilter.wait_for_ready().await.unwrap();
        prefilter
    }

    #[tokio::test]
    async fn test_append_only_deltas_keep_empty_prefilter_fast_path() {
        let first = lance_datagen::gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(4)),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let first_schema = first.schema();
        let mut dataset = Dataset::write(first, "memory://", None).await.unwrap();
        let first_version = dataset.manifest.version;
        let first_fragments = dataset.fragment_bitmap.as_ref().clone();
        let field_id = dataset.schema().field("vector").unwrap().id;

        let second = lance_datagen::gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(4)),
            )
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        assert_eq!(second.schema(), first_schema);
        dataset.append(second, None).await.unwrap();
        let appended_fragments = dataset.fragment_bitmap.as_ref() - &first_fragments;
        let dataset = Arc::new(dataset);
        let old_segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            covering_fields: vec![],
            name: "vector_idx".to_string(),
            dataset_version: first_version,
            fragment_bitmap: Some(first_fragments.clone()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };
        let new_segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![field_id],
            covering_fields: vec![],
            name: "vector_idx".to_string(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(appended_fragments.clone()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
        };
        assert!(
            !appended_fragments.is_empty(),
            "the append fixture must create at least one new fragment"
        );
        let base = Arc::new(DatasetPreFilter::new(
            dataset.clone(),
            &[old_segment.clone(), new_segment.clone()],
            None,
        ));
        base.wait_for_ready().await.unwrap();
        assert!(base.is_empty(), "the combined delta coverage is unfiltered");
        let segment_prefilter = prefilter_for_segment(dataset.clone(), &old_segment, base)
            .await
            .unwrap();
        segment_prefilter.wait_for_ready().await.unwrap();

        assert!(
            !segment_prefilter.is_empty(),
            "the segment ownership restriction is not globally empty"
        );
        assert!(segment_prefilter.needs_partition_row_ids());
        let old_partition_rows = first_fragments
            .iter()
            .flat_map(|fragment_id| {
                (0..20_u64).map(move |offset| (u64::from(fragment_id) << 32) | offset)
            })
            .collect::<RowAddrTreeMap>();
        assert!(
            segment_prefilter.is_empty_for(&old_partition_rows),
            "an append-only segment must preserve the unfiltered partition fast path"
        );

        let appended_fragment_id = appended_fragments.iter().next().unwrap();
        let mut rows_with_unowned_entry = old_partition_rows;
        rows_with_unowned_entry.insert(u64::from(appended_fragment_id) << 32);
        assert!(!segment_prefilter.is_empty_for(&rows_with_unowned_entry));

        let ordinary_base = Arc::new(
            DatasetPreFilter::new(dataset, &[old_segment, new_segment], None)
                .with_overlay_block(RowAddrMask::allow_nothing()),
        );
        let ordinary_segment =
            SegmentPreFilter::new(ordinary_base, Arc::new(RowAddrMask::all_rows()));
        assert!(
            !ordinary_segment.needs_partition_row_ids(),
            "a user filter cannot take the no-filter fast path, so partition coverage is unused"
        );
    }

    /// A `FilterLoader` that yields a fixed allow-list of row addresses, so the
    /// resulting prefilter mask is a bounded `AllowList` (`max_len()` is `Some`).
    struct StaticAllowList(Vec<u64>);

    #[async_trait]
    impl FilterLoader for StaticAllowList {
        async fn load(self: Box<Self>) -> lance_core::Result<lance_select::RowAddrMask> {
            Ok(lance_select::RowAddrMask::from_allowed(
                lance_select::RowAddrTreeMap::from_iter(self.0.iter().copied()),
            ))
        }
    }

    /// Like `empty_prefilter`, but the prefilter mask is a bounded allow-list of
    /// the given row addresses, which drives the late-search no-rows shortcut.
    async fn bounded_prefilter(allow: Vec<u64>) -> Arc<DatasetPreFilter> {
        static NEXT_PREFILTER_DATASET_ID: AtomicUsize = AtomicUsize::new(100_000);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let uri = format!(
            "memory://bounded-prefilter-{}",
            NEXT_PREFILTER_DATASET_ID.fetch_add(1, Ordering::Relaxed)
        );
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
                &uri,
                None,
            )
            .await
            .unwrap(),
        );
        let mut indexed_fragments = RoaringBitmap::new();
        for fragment in dataset.manifest.fragments.iter() {
            indexed_fragments.insert(fragment.id as u32);
        }
        let index = IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            fields: vec![],
            name: "test".to_string(),
            dataset_version: 1,
            fragment_bitmap: Some(indexed_fragments),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            covering_fields: Vec::new(),
        };
        let loader: Box<dyn FilterLoader> = Box::new(StaticAllowList(allow));
        let prefilter = Arc::new(DatasetPreFilter::new(dataset, &[index], Some(loader)));
        prefilter.wait_for_ready().await.unwrap();
        prefilter
    }

    fn prepared_metrics() -> Arc<AnnIndexMetrics> {
        Arc::new(AnnIndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0))
    }

    type PreparedIndexState = (
        Arc<dyn VectorIndex>,
        Arc<Mutex<Vec<usize>>>,
        Arc<Mutex<Vec<usize>>>,
        Arc<Mutex<Vec<String>>>,
    );

    /// One partition per row id, each returning that single row.
    fn prepared_index(row_ids: Vec<u64>) -> PreparedIndexState {
        prepared_index_multi(row_ids.into_iter().map(|row_id| vec![row_id]).collect())
    }

    /// One partition per entry, each returning the rows in that entry.
    fn prepared_index_multi(row_ids: Vec<Vec<u64>>) -> PreparedIndexState {
        let prepared_partitions = Arc::new(Mutex::new(Vec::new()));
        let searched_partitions = Arc::new(Mutex::new(Vec::new()));
        let search_threads = Arc::new(Mutex::new(Vec::new()));
        let index: Arc<dyn VectorIndex> = Arc::new(PreparedThreadCapturingIndex {
            prepared_partitions: prepared_partitions.clone(),
            searched_partitions: searched_partitions.clone(),
            search_threads: search_threads.clone(),
            row_ids,
        });
        (
            index,
            prepared_partitions,
            searched_partitions,
            search_threads,
        )
    }

    #[test]
    fn test_adjust_probes_rules() {
        let mut query = base_query();
        adjust_probes(&mut query, 10);
        assert_eq!(query.minimum_nprobes, 10);
        assert_eq!(query.maximum_nprobes, None);

        let mut query = base_query();
        query.minimum_nprobes = 20;
        adjust_probes(&mut query, 10);
        assert_eq!(query.minimum_nprobes, 20);
        assert_eq!(query.maximum_nprobes, None);

        let mut query = base_query();
        query.maximum_nprobes = Some(25);
        adjust_probes(&mut query, 10);
        assert_eq!(query.minimum_nprobes, 10);
        assert_eq!(query.maximum_nprobes, Some(25));

        let mut query = base_query();
        query.maximum_nprobes = Some(5);
        adjust_probes(&mut query, 10);
        assert_eq!(query.minimum_nprobes, 5);
        assert_eq!(query.maximum_nprobes, Some(5));

        let mut query = base_query();
        query.minimum_nprobes = 30;
        query.maximum_nprobes = Some(50);
        adjust_probes(&mut query, 10);
        assert_eq!(query.minimum_nprobes, 30);
        assert_eq!(query.maximum_nprobes, Some(50));
    }

    #[tokio::test]
    async fn test_find_partitions_runs_on_cpu_runtime() {
        let thread_name = Arc::new(Mutex::new(None));
        let index: Arc<dyn VectorIndex> = Arc::new(ThreadCapturingIndex {
            thread_name: thread_name.clone(),
            row_ids: Vec::new(),
        });

        let (_partitions, _distances) = find_partitions_on_cpu(index, base_query()).await.unwrap();

        let thread_name = thread_name.lock().unwrap().clone().unwrap();
        assert!(
            thread_name.contains("lance-cpu"),
            "expected find_partitions to run on the dedicated cpu runtime, got thread {thread_name}",
        );
    }

    // All partitions fit in a single search batch, so they are searched in one
    // `spawn_cpu` dispatch and therefore share one cpu thread. The partition count
    // adapts to the configured batch size so the single-batch property holds under
    // any valid `LANCE_IVF_STREAMING_SEARCH_BATCH_SIZE`, including 1.
    #[tokio::test]
    async fn test_sequential_initial_search_prepares_all_then_searches_on_one_cpu_thread() {
        let num_partitions = 3.min(*STREAMING_SEARCH_BATCH_SIZE);
        let row_ids = (0..num_partitions).map(|i| 10 + i as u64).collect();
        let (index, prepared_partitions, searched_partitions, search_threads) =
            prepared_index(row_ids);
        let mut query = base_query();
        query.minimum_nprobes = num_partitions;
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));

        let partition_idx = (0..num_partitions as u32).collect::<Vec<_>>();
        let q_c_dists = (0..num_partitions)
            .map(|i| i as f32 * 0.1)
            .collect::<Vec<_>>();
        let batches = ANNIvfSubIndexExec::initial_search(
            index,
            query,
            Arc::new(UInt32Array::from(partition_idx)),
            Arc::new(Float32Array::from(q_c_dists)),
            empty_prefilter().await,
            prepared_metrics(),
            state,
            usize::MAX,
            None,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        let expected: Vec<usize> = (0..num_partitions).collect();
        assert_eq!(batches.len(), num_partitions);
        assert_eq!(*prepared_partitions.lock().unwrap(), expected);
        assert_eq!(*searched_partitions.lock().unwrap(), expected);
        let search_threads = search_threads.lock().unwrap().clone();
        assert_eq!(search_threads.len(), num_partitions);
        assert!(
            search_threads.iter().all(|name| name.contains("lance-cpu")),
            "expected prepared searches to run on the cpu runtime, got threads {search_threads:?}",
        );
        assert!(
            search_threads.iter().all(|name| name == &search_threads[0]),
            "expected all prepared searches to reuse one cpu thread, got threads {search_threads:?}",
        );
    }

    // Regression guard for the batched streaming search (#7642): with more partitions
    // than a single batch, the search spans multiple `spawn_cpu` dispatches. Verify that
    // every partition is still prepared and searched in order across the batch boundary,
    // and that all search work stays on the cpu runtime.
    //
    // Note: this does not reproduce the single-thread-pool deadlock the async recv/send
    // fixes -- that requires a 1-thread CPU pool, which is a process-global singleton and
    // impractical to force in a unit test (same limitation noted for the #7423 fix).
    #[tokio::test]
    async fn test_sequential_search_spans_multiple_cpu_batches() {
        let num_partitions = *STREAMING_SEARCH_BATCH_SIZE + 3;
        let row_ids = (0..num_partitions).map(|i| i as u64 * 10).collect();
        let (index, prepared_partitions, searched_partitions, search_threads) =
            prepared_index(row_ids);
        let mut query = base_query();
        query.minimum_nprobes = num_partitions;
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));

        let partition_idx = (0..num_partitions as u32).collect::<Vec<_>>();
        let q_c_dists = (0..num_partitions).map(|i| i as f32).collect::<Vec<_>>();
        let batches = ANNIvfSubIndexExec::initial_search(
            index,
            query,
            Arc::new(UInt32Array::from(partition_idx.clone())),
            Arc::new(Float32Array::from(q_c_dists)),
            empty_prefilter().await,
            prepared_metrics(),
            state,
            usize::MAX,
            None,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        let expected: Vec<usize> = (0..num_partitions).collect();
        assert_eq!(batches.len(), num_partitions);
        assert_eq!(*prepared_partitions.lock().unwrap(), expected);
        assert_eq!(*searched_partitions.lock().unwrap(), expected);
        let search_threads = search_threads.lock().unwrap().clone();
        assert_eq!(search_threads.len(), num_partitions);
        assert!(
            search_threads.iter().all(|name| name.contains("lance-cpu")),
            "expected prepared searches to run on the cpu runtime, got threads {search_threads:?}",
        );
    }

    #[tokio::test]
    async fn test_sequential_late_search_prepares_all_then_stops_search_early() {
        let (index, prepared_partitions, searched_partitions, _search_threads) =
            prepared_index(vec![21, 22, 23]);
        let mut query = base_query();
        query.k = 2;
        query.minimum_nprobes = 0;
        query.maximum_nprobes = Some(3);
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));
        state
            .record_batch(
                &RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![
                        Arc::new(Float32Array::from(vec![0.0])),
                        Arc::new(UInt64Array::from(vec![999])),
                    ],
                )
                .unwrap(),
            )
            .unwrap();

        let prefilter = empty_prefilter().await;
        let batches = ANNIvfSubIndexExec::late_search(
            index,
            query,
            Arc::new(UInt32Array::from(vec![0, 1, 2])),
            Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])),
            prefilter.clone(),
            prefilter,
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            None,
            false,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(*prepared_partitions.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(*searched_partitions.lock().unwrap(), vec![0]);
        assert_eq!(state.num_results_found.load(Ordering::Relaxed), 2);
    }

    /// `record_batch` must resolve `_rowid` by name, not by position: a covered ANN
    /// batch appends covering columns after `_rowid`, but nothing here should assume
    /// that specific layout. Puts another UInt64 column ahead of `_rowid` so a
    /// positional read would silently record the decoy column's values instead.
    #[test]
    fn test_early_search_results_record_batch_resolves_row_id_by_name() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(DIST_COL, DataType::Float32, true),
            ArrowField::new("decoy_u64_col", DataType::UInt64, true),
            ROW_ID_FIELD.clone(),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(vec![0.0, 1.0])),
                Arc::new(UInt64Array::from(vec![111, 222])),
                Arc::new(UInt64Array::from(vec![7, 8])),
            ],
        )
        .unwrap();

        let state = ANNIvfEarlySearchResults::new(1, 2);
        state.record_batch(&batch).unwrap();
        assert_eq!(*state.initial_ids.lock().unwrap(), vec![7, 8]);
    }

    fn row_ids_of(batches: &[RecordBatch]) -> Vec<u64> {
        batches
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn dists_of(batches: &[RecordBatch]) -> Vec<f32> {
        batches
            .iter()
            .flat_map(|batch| {
                batch[DIST_COL]
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// A probe can reach a row whose fragment the segment no longer owns. Such a row
    /// must be dropped before the shared accounting sees it: counting it and only
    /// dropping it downstream lets it consume the `k` budget on its own, so the segment
    /// stops probing early and the query returns fewer than `k` current rows.
    ///
    /// Partitions 0 and 2 hold rows this segment no longer owns and 1 and 3 hold rows
    /// it does, so the restriction is exercised in both the initial and the late search.
    /// The two parallelism settings pick different code paths: the sequential one counts
    /// inside the index via `LatePartitionSearchControl`, the parallel one counts in
    /// `search_partition`.
    #[rstest]
    #[tokio::test]
    async fn test_unowned_row_does_not_fill_the_shared_budget(
        #[values(1, 2)] query_parallelism: i32,
    ) {
        // Every partition mixes owned and unowned rows differently: partition 0 loses its
        // first row, partition 1 its last, partition 2 all of them and partition 3 its
        // last, so the restriction has to keep part of a batch rather than all or nothing.
        let (index, prepared_partitions, searched_partitions, _search_threads) =
            prepared_index_multi(vec![vec![21, 22, 24], vec![25, 26], vec![20], vec![23, 27]]);
        let seg_mask = Arc::new(RowAddrMask::from_allowed(
            lance_select::RowAddrTreeMap::from_iter([22u64, 23, 24, 25]),
        ));
        let partitions = Arc::new(UInt32Array::from(vec![0, 1, 2, 3]));
        let q_c_dists = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4]));

        let mut query = base_query();
        query.k = 4;
        query.minimum_nprobes = 2;
        query.maximum_nprobes = Some(4);
        query.query_parallelism = query_parallelism;
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));

        let early = ANNIvfSubIndexExec::initial_search(
            index.clone(),
            query.clone(),
            partitions.clone(),
            q_c_dists.clone(),
            empty_prefilter().await,
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            Some(seg_mask.clone()),
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(
            row_ids_of(&early),
            vec![22, 24, 25],
            "the initial search must emit the owned rows and only those"
        );
        assert_eq!(
            dists_of(&early),
            vec![0.5, 1.0, 1.0],
            "the distance column must stay aligned with the surviving row ids"
        );
        assert_eq!(
            *state.initial_ids.lock().unwrap(),
            vec![22, 24, 25],
            "unowned rows must not take up the initial result budget"
        );

        let prefilter = empty_prefilter().await;
        let late = ANNIvfSubIndexExec::late_search(
            index,
            query,
            partitions,
            q_c_dists,
            prefilter.clone(),
            prefilter,
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            Some(seg_mask),
            false,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(
            *searched_partitions.lock().unwrap(),
            vec![0, 1, 2, 3],
            "the all-unowned partition 2 must not stop the late search"
        );
        assert_eq!(
            row_ids_of(&late),
            vec![23],
            "the late search must emit only rows the segment owns"
        );
        assert_eq!(
            state.num_results_found.load(Ordering::Relaxed),
            4,
            "only rows that survive the segment restriction may be counted"
        );
        // A parallelism setting the cpu pool cannot honour would silently rerun the
        // sequential path, leaving `search_partition`'s restriction untested.
        let prepared_partitions = prepared_partitions.lock().unwrap();
        if query_parallelism == 1 {
            assert_eq!(*prepared_partitions, vec![0, 1, 2, 3]);
        } else {
            assert!(
                prepared_partitions.is_empty(),
                "the parallel path must not prepare partitions, got {prepared_partitions:?}",
            );
        }
    }

    /// Every fragment the segment used to own now belongs to a newer delta, so it can
    /// never contribute a row nor move the shared budget that ends the late search.
    /// Probing it to `maximum_nprobes` would be pure waste.
    #[tokio::test]
    async fn test_segment_owning_nothing_skips_the_late_search() {
        let (index, _prepared_partitions, searched_partitions, _search_threads) =
            prepared_index(vec![21, 22, 23, 24]);
        let seg_mask = Arc::new(RowAddrMask::from_allowed(
            lance_select::RowAddrTreeMap::new(),
        ));
        let partitions = Arc::new(UInt32Array::from(vec![0, 1, 2, 3]));
        let q_c_dists = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4]));

        let mut query = base_query();
        query.k = 4;
        query.minimum_nprobes = 1;
        query.maximum_nprobes = Some(4);
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));

        ANNIvfSubIndexExec::initial_search(
            index.clone(),
            query.clone(),
            partitions.clone(),
            q_c_dists.clone(),
            empty_prefilter().await,
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            Some(seg_mask.clone()),
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        let prefilter = empty_prefilter().await;
        let late = ANNIvfSubIndexExec::late_search(
            index,
            query,
            partitions,
            q_c_dists,
            prefilter.clone(),
            prefilter,
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            Some(seg_mask),
            false,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert!(late.is_empty());
        assert_eq!(
            *searched_partitions.lock().unwrap(),
            vec![0],
            "only the initial probe may run; the late search must not probe at all"
        );
    }

    /// With a bounded selective prefilter, the late-search no-rows shortcut must fire
    /// for covered indexes too -- but emit nothing: the shortcut's `[_distance, _rowid]`
    /// batch would drop the covering columns, so the end-of-stream covered recovery
    /// (`covered_not_found_batch`) emits the missing rows instead, with a take bounded
    /// by the prefilter size. Searching instead of shortcutting is the failure mode this
    /// pins down: a prefilter-matched row with no index entry (null vector) can never be
    /// found, so the search degenerates to scoring every partition on every query.
    #[tokio::test]
    async fn test_late_search_covered_skips_search_on_selective_prefilter() {
        // Bounded prefilter (one allowed row) + k=2 => max_results=1 <= k, and an
        // empty initial state => found_so_far=0 < max_results: shortcut criteria.
        async fn run_late_search(has_covered: bool) -> (Vec<RecordBatch>, Vec<usize>) {
            let (index, _prepared, searched_partitions, _threads) =
                prepared_index(vec![21, 22, 23]);
            let mut query = base_query();
            query.k = 2;
            query.minimum_nprobes = 0;
            query.maximum_nprobes = Some(3);
            let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));
            let prefilter = bounded_prefilter(vec![0]).await;

            let batches = ANNIvfSubIndexExec::late_search(
                index,
                query,
                Arc::new(UInt32Array::from(vec![0, 1, 2])),
                Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])),
                prefilter.clone(),
                prefilter,
                prepared_metrics(),
                state,
                usize::MAX,
                None,
                has_covered,
            )
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
            let searched = searched_partitions.lock().unwrap().clone();
            (batches, searched)
        }

        // Control: a non-covered index takes the shortcut, so no partition is
        // searched and the only batch is the [_distance, _rowid] not-found batch.
        let (batches, searched) = run_late_search(false).await;
        assert!(
            searched.is_empty(),
            "non-covered query should take the shortcut and skip the search, searched={searched:?}"
        );
        // The non-covered control for the covered/non-covered contrast below: guard
        // against a batch count of zero, which would make `all(..)` vacuously true
        // regardless of whether the shortcut actually emitted anything.
        assert_eq!(batches.len(), 1, "the shortcut emits exactly one batch");
        assert!(
            batches.iter().all(|b| b.num_columns() == 2),
            "the shortcut emits only [_distance, _rowid]"
        );

        // Covered: the shortcut fires too (no partition searched), but emits nothing --
        // the end-of-stream covered recovery supplies the not-found rows with their
        // covering columns.
        let (batches, searched) = run_late_search(true).await;
        assert!(
            searched.is_empty(),
            "covered query must take the shortcut and skip the search, searched={searched:?}"
        );
        assert!(
            batches.is_empty(),
            "the covered shortcut emits nothing; recovery happens at end of stream"
        );
    }

    /// Arming must record the arming delta's ownership scope, and the recovery must
    /// emit only rows an armed scope selects. A sibling delta that skipped its late
    /// search without arming never emits its not-found rows on the non-covered path
    /// (its `seg_mask` filters them out of the shortcut batch), so recovering them
    /// for a covered index would make the covered and non-covered result sets
    /// diverge -- e.g. null-vector rows owned by the skipping delta surfacing at
    /// INFINITY distance only when the index is covered.
    #[tokio::test]
    async fn test_covered_recovery_honors_arming_delta_ownership() {
        // Delta B owns row 0 only; rows 1 and 2 belong to a sibling delta that will
        // NOT arm (it skipped its late search entirely).
        let seg_mask_b = Arc::new(RowAddrMask::from_allowed(
            lance_select::RowAddrTreeMap::from_iter([0u64]),
        ));

        // Delta B takes the covered no-rows shortcut: bounded prefilter of 3 rows,
        // k = 4, nothing found so far.
        let (index, _prepared, searched_partitions, _threads) = prepared_index(vec![21, 22, 23]);
        let mut query = base_query();
        query.k = 4;
        query.minimum_nprobes = 0;
        query.maximum_nprobes = Some(3);
        let state = Arc::new(ANNIvfEarlySearchResults::new(1, query.k));
        let prefilter = bounded_prefilter(vec![0, 1, 2]).await;

        let batches = ANNIvfSubIndexExec::late_search(
            index,
            query,
            Arc::new(UInt32Array::from(vec![0, 1, 2])),
            Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])),
            prefilter.clone(),
            prefilter.clone(),
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            Some(seg_mask_b.clone()),
            true,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
        assert!(batches.is_empty(), "covered shortcut emits nothing inline");
        assert!(
            searched_partitions.lock().unwrap().is_empty(),
            "the shortcut must not search"
        );

        // Arming recorded delta B's scope (and only that one).
        let scopes = std::mem::take(&mut *state.covered_recovery_scopes.lock().unwrap());
        assert_eq!(scopes.len(), 1, "exactly one delta armed");
        assert!(scopes[0].is_some(), "the armed scope is delta B's seg_mask");

        // The recovery over a real dataset must return ONLY delta B's row 0 --
        // rows 1 and 2 are owned by the never-armed sibling.
        let payload_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "payload",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            payload_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 11, 12]))],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), payload_schema),
                "memory://covered-recovery-ownership",
                None,
            )
            .await
            .unwrap(),
        );
        let mut fields = KNN_INDEX_SCHEMA.fields().to_vec();
        fields.push(Arc::new(ArrowField::new("payload", DataType::Int32, false)));
        let output_schema = Arc::new(ArrowSchema::new(fields));

        let emitted = Arc::new(std::sync::Mutex::new(roaring::RoaringTreemap::new()));
        let recovered = ANNIvfSubIndexExec::covered_not_found_batch(
            true,
            prefilter.clone(),
            dataset.clone(),
            output_schema.clone(),
            4,
            emitted.clone(),
            scopes,
        )
        .await
        .unwrap()
        .expect("delta B's unemitted row must be recovered");
        assert_eq!(
            recovered
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            &[0],
            "recovery must emit only rows the arming delta owns"
        );
        assert_eq!(
            recovered
                .column_by_name("payload")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[10],
            "the covering column value comes from the base table"
        );

        // A global (unrestricted) arming scope recovers every unemitted row -- the
        // single-delta behaviour is unchanged.
        let recovered = ANNIvfSubIndexExec::covered_not_found_batch(
            true,
            prefilter,
            dataset,
            output_schema,
            4,
            emitted,
            vec![None],
        )
        .await
        .unwrap()
        .expect("a global scope recovers all unemitted rows");
        assert_eq!(
            recovered
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            &[0, 1, 2],
            "a global arming scope keeps the old recover-everything behaviour"
        );
    }

    #[tokio::test]
    async fn test_delta_skipping_late_search_releases_sibling() {
        let prefilter = empty_prefilter().await;
        let state = Arc::new(ANNIvfEarlySearchResults::new(2, 4));

        let (index_a, _prepared_a, searched_a, _threads_a) = prepared_index(vec![21]);
        let mut query_a = base_query();
        query_a.k = 4;
        query_a.minimum_nprobes = 1;
        let delta_a = ANNIvfSubIndexExec::late_search(
            index_a,
            query_a,
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(Float32Array::from(vec![0.1])),
            prefilter.clone(),
            prefilter.clone(),
            prepared_metrics(),
            state.clone(),
            usize::MAX,
            None,
            false,
        )
        .try_collect::<Vec<_>>();

        let (index_b, _prepared_b, searched_b, _threads_b) = prepared_index(vec![31, 32, 33, 34]);
        let mut query_b = base_query();
        query_b.k = 4;
        query_b.minimum_nprobes = 1;
        query_b.maximum_nprobes = Some(4);
        let delta_b = ANNIvfSubIndexExec::late_search(
            index_b,
            query_b,
            Arc::new(UInt32Array::from(vec![0, 1, 2, 3])),
            Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4])),
            prefilter.clone(),
            prefilter,
            prepared_metrics(),
            state,
            usize::MAX,
            None,
            false,
        )
        .try_collect::<Vec<_>>();

        let (result_a, result_b) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            futures::future::join(delta_a, delta_b),
        )
        .await
        .expect("late search deadlocked because one delta skipped the shared barrier");

        result_a.unwrap();
        result_b.unwrap();
        assert!(searched_a.lock().unwrap().is_empty());
        assert!(!searched_b.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn knn_flat_search() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(
                            FixedSizeListArray::try_new_from_values(
                                generate_random_array(128 * 20),
                                128,
                            )
                            .unwrap(),
                        ),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|i| format!("s3://bucket/file-{}", i)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let vector_arr = batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stream = dataset
            .scan()
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();

        assert!(results[0].schema().column_with_name(DIST_COL).is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let all_with_distances = stream
            .and_then(|batch| compute_distance(q.clone(), DistanceType::L2, "vector", batch))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let all_with_distances =
            concat_batches(&results[0].schema(), all_with_distances.iter()).unwrap();
        let dist_arr = all_with_distances.column_by_name(DIST_COL).unwrap();
        let distances = dist_arr.as_primitive::<Float32Type>();
        let indices = sort_to_indices(distances, None, Some(10)).unwrap();
        let expected = take_record_batch(&all_with_distances, &indices).unwrap();
        assert_eq!(expected, results[0]);
    }

    #[test]
    fn test_create_knn_flat() {
        let dim: usize = 128;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema);

        let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));

        let idx = KNNVectorDistanceExec::try_new(
            input,
            "vector",
            Arc::new(generate_random_array(dim)),
            DistanceType::L2,
        )
        .unwrap();
        assert_eq!(
            idx.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("key", DataType::Int32, false),
                ArrowField::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    true,
                ),
                ArrowField::new("uri", DataType::Utf8, true),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
        );
    }

    #[test]
    fn test_batch_partition_statistics_aligns_with_output_schema() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    4,
                ),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]));
        let batch = RecordBatch::new_empty(schema);
        let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));
        let query = Arc::new(Float32Array::from(vec![0.0, 1.0, 2.0, 3.0])) as ArrayRef;
        let plan = KNNVectorDistanceExec::try_new_batch(
            input,
            "vec",
            query,
            KnnBatchParams {
                is_batch: true,
                query_count: 1,
                k: 2,
                lower_bound: None,
                upper_bound: None,
                distance_type: DistanceType::L2,
                retain_vector: false,
            },
        )
        .unwrap();
        let stats = plan.partition_statistics(None).unwrap();
        assert_eq!(
            stats.column_statistics.len(),
            plan.schema().fields().len(),
            "partition stats must align with output schema"
        );
        let schema = plan.schema();
        let query_index_pos = schema
            .column_with_name(QUERY_INDEX_COL)
            .expect("query_index must exist")
            .0;
        let dist_pos = schema
            .column_with_name(DIST_COL)
            .expect("distance must exist")
            .0;
        assert_eq!(
            stats.column_statistics[query_index_pos],
            ColumnStatistics::default(),
        );
        assert_eq!(
            stats.column_statistics[dist_pos].null_count,
            stats.column_statistics[schema.column_with_name("i").unwrap().0].null_count,
            "distance null-count should be derived from vector/input nullability and remain aligned"
        );
    }

    #[test]
    fn test_remove_vector_from_schema_nested_path() {
        let payload_field = ArrowField::new(
            "payload",
            DataType::Struct(
                vec![
                    ArrowField::new(
                        "vec",
                        DataType::FixedSizeList(
                            Arc::new(ArrowField::new("item", DataType::Float32, true)),
                            4,
                        ),
                        true,
                    ),
                    ArrowField::new("tag", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        );
        let schema = ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            payload_field,
            ROW_ID_FIELD.clone(),
        ]);
        let without_vec =
            KNNVectorDistanceExec::remove_vector_from_schema(&schema, "payload.vec").unwrap();
        let payload = without_vec.field_with_name("payload").unwrap();
        let DataType::Struct(children) = payload.data_type() else {
            panic!("payload should remain struct");
        };
        assert!(children.iter().all(|f| f.name() != "vec"));
        assert!(children.iter().any(|f| f.name() == "tag"));
    }

    #[test]
    fn test_take_vector_row_copies_single_row() {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..12).map(|v| v as f32).collect::<Vec<_>>()),
            4,
        )
        .unwrap();
        let row = KNNVectorDistanceExec::take_vector_row(&vectors, 2).unwrap();
        assert_eq!(row.len(), 1);
        assert_eq!(
            row.to_data().offset(),
            0,
            "take/copy should not retain row offset into the full input buffer"
        );
    }

    #[test]
    fn test_resolve_vector_column_supports_escaped_nested_path() {
        let vec_field = ArrowField::new(
            "vec.with.dot",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        );
        let payload_field = ArrowField::new(
            "payload",
            DataType::Struct(vec![vec_field.clone()].into()),
            true,
        );
        let schema = Arc::new(ArrowSchema::new(vec![payload_field]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..8).map(|v| v as f32).collect::<Vec<_>>()),
            4,
        )
        .unwrap();
        let payload = StructArray::from(vec![(Arc::new(vec_field), Arc::new(vectors) as ArrayRef)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(payload)]).unwrap();
        let vector =
            KNNVectorDistanceExec::resolve_vector_column(&batch, "payload.`vec.with.dot`").unwrap();
        assert_eq!(vector.len(), 2);
    }

    #[test]
    fn test_remove_vector_from_batch_nested_keeps_siblings() {
        let vec_field = ArrowField::new(
            "vec.with.dot",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        );
        let tag_field = ArrowField::new("tag", DataType::Utf8, true);
        let payload_field = ArrowField::new(
            "payload",
            DataType::Struct(vec![vec_field.clone(), tag_field.clone()].into()),
            true,
        );
        let schema = Arc::new(ArrowSchema::new(vec![payload_field]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..8).map(|v| v as f32).collect::<Vec<_>>()),
            4,
        )
        .unwrap();
        let tags = StringArray::from(vec!["a", "b"]);
        let payload = StructArray::from(vec![
            (Arc::new(vec_field), Arc::new(vectors) as ArrayRef),
            (Arc::new(tag_field), Arc::new(tags) as ArrayRef),
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(payload)]).unwrap();

        let slim =
            KNNVectorDistanceExec::remove_vector_from_batch(&batch, "payload.`vec.with.dot`")
                .unwrap();
        let payload = slim.column_by_name("payload").unwrap().as_struct();
        assert!(payload.column_by_name("vec.with.dot").is_none());
        assert!(payload.column_by_name("tag").is_some());
    }

    #[test]
    fn test_assemble_batch_output_retained_nested_vector_keeps_sibling_values() {
        let vec_field = ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        );
        let tag_field = ArrowField::new("tag", DataType::Utf8, true);
        let payload_field = ArrowField::new(
            "payload",
            DataType::Struct(vec![vec_field.clone(), tag_field.clone()].into()),
            true,
        );
        let schema = Arc::new(ArrowSchema::new(vec![payload_field, ROW_ID_FIELD.clone()]));
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..12).map(|v| v as f32).collect::<Vec<_>>()),
            4,
        )
        .unwrap();
        let tags = StringArray::from(vec!["a", "b", "c"]);
        let payload = StructArray::from(vec![
            (Arc::new(vec_field), Arc::new(vectors) as ArrayRef),
            (Arc::new(tag_field), Arc::new(tags) as ArrayRef),
        ]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(payload) as ArrayRef,
                Arc::new(UInt64Array::from(vec![10, 11, 12])) as ArrayRef,
            ],
        )
        .unwrap();
        let slim_batch = Arc::new(
            KNNVectorDistanceExec::remove_vector_from_batch(&batch, "payload.vec").unwrap(),
        );
        let vectors = KNNVectorDistanceExec::resolve_vector_column(&batch, "payload.vec").unwrap();
        let results = [2, 0]
            .into_iter()
            .map(|row_index| BatchKnnCandidate {
                query_index: 0,
                distance: row_index as f32,
                row_id: 10 + row_index as u64,
                extra: BatchKnnExtra::WithSlimBatch {
                    slim_batch: Arc::clone(&slim_batch),
                    row_index,
                    vector_row: Some(
                        KNNVectorDistanceExec::take_vector_row(vectors.as_ref(), row_index)
                            .unwrap(),
                    ),
                },
            })
            .collect::<Vec<_>>();

        let output = KNNVectorDistanceExec::assemble_batch_output(
            &results,
            schema.as_ref(),
            "payload.vec",
            true,
        )
        .unwrap();

        let payload = output.column_by_name("payload").unwrap().as_struct();
        let tags = payload.column_by_name("tag").unwrap().as_string::<i32>();
        assert!(tags.is_valid(0));
        assert!(tags.is_valid(1));
        assert_eq!(tags.value(0), "c");
        assert_eq!(tags.value(1), "a");
        let vectors = payload.column_by_name("vec").unwrap();
        assert_eq!(vectors.len(), 2);
    }

    #[tokio::test]
    async fn test_multivector_score() {
        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(generate_random_array(1)),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::Cosine),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: None,
        };

        async fn multivector_scoring(
            inputs: Vec<Arc<dyn ExecutionPlan>>,
            query: Query,
        ) -> Result<HashMap<u64, f32>> {
            let ctx = Arc::new(datafusion::execution::context::TaskContext::default());
            let plan = MultivectorScoringExec::try_new(inputs, query.clone())?;
            let batches = plan
                .execute(0, ctx.clone())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await?;
            let mut results = HashMap::new();
            for batch in batches {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let dists = batch[DIST_COL].as_primitive::<Float32Type>();
                for (row_id, dist) in row_ids.values().iter().zip(dists.values().iter()) {
                    results.insert(*row_id, *dist);
                }
            }
            Ok(results)
        }

        let batches = (0..3)
            .map(|i| {
                RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![
                        Arc::new(Float32Array::from(vec![i as f32 + 1.0, i as f32 + 2.0])),
                        Arc::new(UInt64Array::from(vec![i + 1, i + 2])),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let mut res: Option<HashMap<_, _>> = None;
        for perm in batches.into_iter().permutations(3) {
            let inputs = perm
                .into_iter()
                .map(|batch| {
                    let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));
                    input
                })
                .collect::<Vec<_>>();
            let new_res = multivector_scoring(inputs, query.clone()).await.unwrap();
            assert_eq!(new_res.len(), 4);
            if let Some(res) = &res {
                for (row_id, dist) in new_res.iter() {
                    assert_eq!(res.get(row_id).unwrap(), dist)
                }
            } else {
                res = Some(new_res);
            }
        }
    }

    /// Covered multivector scoring must carry each row's covering values through
    /// the per-row-id reduction, aligned by row id -- and must FAIL LOUDLY (not
    /// emit nulls) if an emitted row id ever has no recorded covering source.
    #[tokio::test]
    async fn test_multivector_score_carries_covering_values() {
        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(generate_random_array(1)),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::Cosine),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            covering_projection: None,
        };

        let mut fields = KNN_INDEX_SCHEMA.fields().to_vec();
        fields.push(Arc::new(ArrowField::new("payload", DataType::Int32, false)));
        let covered_schema = Arc::new(ArrowSchema::new(fields));

        // Three sub-query streams over overlapping row ids 1..=4; the covering value
        // is a pure function of the row id (100 + row_id), identical across streams,
        // as the contract requires.
        let inputs = (0..3u64)
            .map(|i| {
                let batch = RecordBatch::try_new(
                    covered_schema.clone(),
                    vec![
                        Arc::new(Float32Array::from(vec![i as f32 + 1.0, i as f32 + 2.0])),
                        Arc::new(UInt64Array::from(vec![i + 1, i + 2])),
                        Arc::new(Int32Array::from(vec![
                            100 + (i + 1) as i32,
                            100 + (i + 2) as i32,
                        ])),
                    ],
                )
                .unwrap();
                let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));
                input
            })
            .collect::<Vec<_>>();

        let ctx = Arc::new(datafusion::execution::context::TaskContext::default());
        let plan = MultivectorScoringExec::try_new(inputs, query).unwrap();
        let batches = plan
            .execute(0, ctx)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut seen = 0;
        for batch in batches {
            assert_eq!(batch.num_columns(), 3, "covering column must flow through");
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            let payloads = batch
                .column_by_name("payload")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            assert_eq!(payloads.null_count(), 0, "no fabricated nulls");
            for (row_id, payload) in row_ids.values().iter().zip(payloads.values().iter()) {
                assert_eq!(
                    *payload,
                    100 + *row_id as i32,
                    "covering value must stay aligned with its row id"
                );
                seen += 1;
            }
        }
        assert_eq!(seen, 4, "all four distinct row ids are emitted");
    }

    /// A test dataset for testing the nprobes parameter.
    ///
    /// The dataset has 100 partitions and filterable columns setup to easily create
    /// filters whose results are spread across the partitions evenly.
    struct NprobesTestFixture {
        dataset: Dataset,
        centroids: Arc<dyn Array>,
        _tmp_dir: TempStrDir,
    }

    impl NprobesTestFixture {
        pub async fn new(num_centroids: usize, num_deltas: usize) -> Self {
            let tempdir = TempStrDir::default();
            let tmppath = tempdir.as_str();

            // We create 100 centroids
            // We generate 10,000 vectors evenly divided (100 vectors per centroid)
            // We assign labels 0..60 to the vectors so each label has ~164 vectors
            //   spread out through all of the centroids
            let centroids = array::cycle_unit_circle(num_centroids as u32)
                .generate_default(RowCount::from(num_centroids as u64))
                .unwrap();

            // Let's not deal with fractions
            assert!(100 % num_deltas == 0, "num_deltas must divide 100");
            let rows_per_frag = 100;
            let num_frags = 100;
            let frags_per_delta = num_frags / num_deltas;

            let batches = lance_datagen::gen_batch()
                .col("vector", array::jitter_centroids(centroids.clone(), 0.0001))
                .col("label", array::cycle::<UInt32Type>(Vec::from_iter(0..61)))
                .col("userid", array::step::<UInt64Type>())
                .into_reader_rows(
                    RowCount::from(rows_per_frag),
                    BatchCount::from(num_frags as u32),
                )
                .collect::<Vec<_>>();
            let schema = batches[0].as_ref().unwrap().schema();

            let mut first = true;
            for batches in batches.chunks(frags_per_delta) {
                let delta_batches = batches
                    .iter()
                    .map(|maybe_batch| Ok(maybe_batch.as_ref().unwrap().clone()))
                    .collect::<Vec<_>>();
                let reader = RecordBatchIterator::new(delta_batches, schema.clone());
                let mut dataset = Dataset::write(
                    reader,
                    tmppath,
                    Some(WriteParams {
                        mode: WriteMode::Append,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();

                let ivf_params = IvfBuildParams::try_with_centroids(
                    num_centroids,
                    Arc::new(centroids.as_fixed_size_list().clone()),
                )
                .unwrap();

                let codebook = array::rand::<Float32Type>()
                    .generate_default(RowCount::from(256 * 2))
                    .unwrap();
                let pq_params = PQBuildParams::with_codebook(2, 8, codebook);
                let index_params =
                    VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

                if first {
                    first = false;
                    dataset
                        .create_index(&["vector"], IndexType::Vector, None, &index_params, false)
                        .await
                        .unwrap();
                } else {
                    dataset
                        .optimize_indices(&OptimizeOptions::append())
                        .await
                        .unwrap();
                }
            }

            let dataset = Dataset::open(tmppath).await.unwrap();
            Self {
                dataset,
                centroids,
                _tmp_dir: tempdir,
            }
        }

        pub fn get_centroid(&self, idx: usize) -> Arc<dyn Array> {
            let centroids = self.centroids.as_fixed_size_list();
            centroids.value(idx).clone()
        }
    }

    #[derive(Default)]
    struct StatsHolder {
        pub collected_stats: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
    }

    impl StatsHolder {
        fn get_setter(&self) -> ExecutionStatsCallback {
            let collected_stats = self.collected_stats.clone();
            Arc::new(move |stats| {
                *collected_stats.lock().unwrap() = Some(stats.clone());
            })
        }

        fn consume(self) -> ExecutionSummaryCounts {
            self.collected_stats.lock().unwrap().take().unwrap()
        }
    }

    fn assert_find_partitions_elapsed_recorded(stats: &ExecutionSummaryCounts) {
        assert!(
            stats
                .all_times
                .get(FIND_PARTITIONS_ELAPSED_METRIC)
                .copied()
                .unwrap_or_default()
                > 0
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_no_max_nprobes(#[values(1, 20)] num_deltas: usize) {
        let fixture = NprobesTestFixture::new(100, num_deltas).await;

        let q = fixture.get_centroid(0);
        let stats_holder = StatsHolder::default();

        let results = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 50)
            .unwrap()
            .minimum_nprobes(10)
            .prefilter(true)
            .scan_stats_callback(stats_holder.get_setter())
            .filter("label = 17")
            .unwrap()
            .project(&Vec::<String>::new())
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(results.num_rows(), 50);

        let stats = stats_holder.consume();

        // We should not search _all_ partitions because we should hit 50 results partway
        // through the late search.
        // The exact number here is deterministic but depends on the number of CPUs
        if get_num_compute_intensive_cpus() <= 32 {
            assert!(*stats.all_counts.get(PARTITIONS_SEARCHED_METRIC).unwrap() < 100 * num_deltas);
        }
        assert_find_partitions_elapsed_recorded(&stats);
    }

    /// The ANN operators report the exact index-file I/O performed for a query
    /// (bytes_read / iops), measured only on cache misses.  A cold search loads
    /// partitions from storage and reports non-zero I/O; an immediately
    /// following warm search serves every partition from the index cache and
    /// reports zero -- which is the cache-effectiveness signal the metric adds.
    #[tokio::test]
    async fn test_io_metrics_cold_vs_warm() {
        let fixture = NprobesTestFixture::new(100, 1).await;
        let q = fixture.get_centroid(0);

        let run = |holder: &StatsHolder| {
            let setter = holder.get_setter();
            async {
                fixture
                    .dataset
                    .scan()
                    .nearest("vector", q.as_ref(), 10)
                    .unwrap()
                    .minimum_nprobes(10)
                    .scan_stats_callback(setter)
                    .project(&Vec::<String>::new())
                    .unwrap()
                    .with_row_id()
                    .try_into_batch()
                    .await
                    .unwrap()
            }
        };

        // Cold: a freshly opened dataset has an empty index cache, so the
        // sub-index search must read partitions (and their quantization storage)
        // from disk.  Those reads flow through the per-query I/O sink.
        let cold_holder = StatsHolder::default();
        run(&cold_holder).await;
        let cold = cold_holder.consume();
        assert!(
            cold.parts_loaded > 0,
            "cold search should load partitions, got parts_loaded={}",
            cold.parts_loaded
        );
        assert!(
            cold.bytes_read > 0,
            "cold search should report index-file I/O, got bytes_read={}",
            cold.bytes_read
        );
        assert!(
            cold.iops > 0,
            "cold search should report index-file IOPS, got iops={}",
            cold.iops
        );

        // Warm: the same query on the same dataset finds every partition it
        // needs already cached, so no index-file I/O is performed.
        let warm_holder = StatsHolder::default();
        run(&warm_holder).await;
        let warm = warm_holder.consume();
        assert_eq!(
            warm.parts_loaded, 0,
            "warm search should not reload partitions, got parts_loaded={}",
            warm.parts_loaded
        );
        assert_eq!(
            warm.bytes_read, 0,
            "warm search should report no index-file I/O, got bytes_read={}",
            warm.bytes_read
        );
    }

    /// The new I/O metrics must actually surface in `EXPLAIN ANALYZE` text on
    /// the ANN operators: non-zero on a cold query (partition reads on
    /// `ANNSubIndex`, index-open reads on `ANNIvfPartition`) and zero on a warm
    /// query (everything served from the index cache).
    #[tokio::test]
    async fn test_io_metrics_visible_in_explain_analyze() {
        // Returns the value of `metric=` from the analyzed-plan line for `node`.
        fn node_metric<'a>(plan: &'a str, node: &str, metric: &str) -> &'a str {
            let line = plan
                .lines()
                .find(|l| l.trim_start().starts_with(node))
                .unwrap_or_else(|| panic!("plan missing node {node}:\n{plan}"));
            let after = line
                .split_once(&format!("{metric}="))
                .unwrap_or_else(|| panic!("node {node} line missing {metric}=:\n{line}"))
                .1;
            after.split([',', ']']).next().unwrap().trim()
        }

        let fixture = NprobesTestFixture::new(100, 1).await;
        let q = fixture.get_centroid(0);

        // Cold: a freshly opened dataset must show real index-file I/O.
        let cold = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 10)
            .unwrap()
            .minimum_nprobes(10)
            .analyze_plan()
            .await
            .unwrap();
        // Sub-index partition reads.
        assert_ne!(node_metric(&cold, "ANNSubIndex", "bytes_read"), "0");
        assert_ne!(node_metric(&cold, "ANNSubIndex", "iops"), "0");
        // Index-open reads (centroids/metadata) now attributed to the partition
        // operator -- the value this part of the change adds.
        assert_ne!(node_metric(&cold, "ANNIvfPartition", "bytes_read"), "0");
        assert_ne!(node_metric(&cold, "ANNIvfPartition", "iops"), "0");

        // Warm: same query, everything cache-resident -> zero index-file I/O.
        let warm = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 10)
            .unwrap()
            .minimum_nprobes(10)
            .analyze_plan()
            .await
            .unwrap();
        assert_eq!(node_metric(&warm, "ANNSubIndex", "bytes_read"), "0");
        assert_eq!(node_metric(&warm, "ANNIvfPartition", "bytes_read"), "0");
    }

    #[rstest]
    #[tokio::test]
    async fn test_no_prefilter_results(#[values(1, 20)] num_deltas: usize) {
        let fixture = NprobesTestFixture::new(100, num_deltas).await;

        let q = fixture.get_centroid(0);
        let stats_holder = StatsHolder::default();

        let results = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 50)
            .unwrap()
            .minimum_nprobes(10)
            .prefilter(true)
            .scan_stats_callback(stats_holder.get_setter())
            .filter("label = 17 AND label = 18")
            .unwrap()
            .project(&Vec::<String>::new())
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(results.num_rows(), 0);

        let stats = stats_holder.consume();
        // We still do the early search because we don't wait for the prefilter to execute
        // We skip the late search because by then we know there are no results
        assert_eq!(
            stats.all_counts.get(PARTITIONS_SEARCHED_METRIC).unwrap(),
            &(10 * num_deltas)
        );
        assert_find_partitions_elapsed_recorded(&stats);
    }

    #[rstest]
    #[tokio::test]
    async fn test_some_max_nprobes(#[values(1, 20)] num_deltas: usize) {
        let fixture = NprobesTestFixture::new(100, num_deltas).await;

        for (max_nprobes, expected_results) in [(10, 16), (20, 33), (30, 48)] {
            let q = fixture.get_centroid(0);
            let stats_holder = StatsHolder::default();
            let results = fixture
                .dataset
                .scan()
                .nearest("vector", q.as_ref(), 50)
                .unwrap()
                .minimum_nprobes(max_nprobes)
                .maximum_nprobes(max_nprobes)
                .prefilter(true)
                .filter("label = 17")
                .unwrap()
                .scan_stats_callback(stats_holder.get_setter())
                .project(&Vec::<String>::new())
                .unwrap()
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap();

            let stats = stats_holder.consume();

            assert_eq!(results.num_rows(), expected_results);
            assert_eq!(
                stats.all_counts.get(PARTITIONS_SEARCHED_METRIC).unwrap(),
                &(max_nprobes * num_deltas)
            );
            assert_eq!(
                stats.all_counts.get(PARTITIONS_RANKED_METRIC).unwrap(),
                &(100 * num_deltas)
            );
            assert_find_partitions_elapsed_recorded(&stats);
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_fewer_than_k_results(#[values(1, 20)] num_deltas: usize) {
        let fixture = NprobesTestFixture::new(100, num_deltas).await;

        let q = fixture.get_centroid(0);
        let stats_holder = StatsHolder::default();
        let results = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 50)
            .unwrap()
            .minimum_nprobes(10)
            .prefilter(true)
            .filter("userid < 20")
            .unwrap()
            .scan_stats_callback(stats_holder.get_setter())
            .project(&Vec::<String>::new())
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let stats = stats_holder.consume();

        // We should only search minimum_nprobes before we look at the prefilter and realize
        // we can cheaply stop early.
        assert_eq!(
            stats.all_counts.get(PARTITIONS_SEARCHED_METRIC).unwrap(),
            &(10 * num_deltas)
        );
        assert_find_partitions_elapsed_recorded(&stats);
        assert_eq!(results.num_rows(), 20);

        // 15 of the results come from beyond the closest 10 partitions and these will have infinite
        // distance.
        let num_infinite_results = results
            .column(0)
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .filter(|val| val.is_infinite())
            .count();
        assert_eq!(num_infinite_results, 15);

        // If we set a refine factor then the distance should not be infinite.
        let results = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 50)
            .unwrap()
            .minimum_nprobes(10)
            .prefilter(true)
            .refine(1)
            .filter("userid < 20")
            .unwrap()
            .project(&Vec::<String>::new())
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        assert_eq!(results.num_rows(), 20);
        let num_infinite_results = results
            .column(0)
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .filter(|val| val.is_infinite())
            .count();
        assert_eq!(num_infinite_results, 0);
    }

    #[rstest]
    #[tokio::test]
    async fn test_dataset_too_small(#[values(1, 20)] num_deltas: usize) {
        let fixture = NprobesTestFixture::new(100, num_deltas).await;

        let q = fixture.get_centroid(0);
        let stats_holder = StatsHolder::default();
        // There is no filter but we only have 10K rows.  Since maximum_nprobes is not set
        // we will search all partitions.
        let results = fixture
            .dataset
            .scan()
            .nearest("vector", q.as_ref(), 40000)
            .unwrap()
            .minimum_nprobes(10)
            .scan_stats_callback(stats_holder.get_setter())
            .project(&Vec::<String>::new())
            .unwrap()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let stats = stats_holder.consume();

        assert_eq!(
            stats.all_counts.get(PARTITIONS_SEARCHED_METRIC).unwrap(),
            &(100 * num_deltas)
        );
        assert_find_partitions_elapsed_recorded(&stats);
        assert_eq!(results.num_rows(), 10000);
    }
}
