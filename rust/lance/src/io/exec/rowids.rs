// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type};
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::ColumnStatistics;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ChildStats;
use datafusion::physical_plan::StatisticsArgs;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::projection::project_ordering;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{ExecutionPlanProperties, Statistics};
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{
    Error as LanceError, ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, ROW_OFFSET, ROW_OFFSET_FIELD,
    Result as LanceResult,
};

use crate::Dataset;
use crate::dataset::rowids::row_ids_to_row_addrs;

/// Add a `_rowaddr` column to a stream of record batches that have a `_rowid`.
///
/// It's generally more efficient to scan the `_rowaddr` column, but this can be
/// useful when reading secondary indices, which only have the `_rowid` column.
/// Missing or deleted stable IDs become null addresses; input order and row count
/// are preserved.
#[derive(Clone)]
pub struct AddRowAddrExec {
    input: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    /// Position in the input schema where the rowids are located
    rowid_pos: usize,
    /// Position in the output schema where to insert the row address
    rowaddr_pos: usize,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,

    metrics: ExecutionPlanMetricsSet,
}

impl std::fmt::Debug for AddRowAddrExec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("AddRowAddrExec")
            .field("input", &self.input)
            .field("dataset", &self.dataset)
            .field("rowid_pos", &self.rowid_pos)
            .field("rowaddr_pos", &self.rowaddr_pos)
            .field("output_schema", &self.output_schema)
            .field("properties", &self.properties)
            .finish()
    }
}

impl AddRowAddrExec {
    /// Create a new AddRowAddrExec node.
    ///
    /// This adds a `_rowaddr` column to streams where there is a `_rowid`
    /// column.
    ///
    /// # Errors
    ///
    /// If the `_rowid` field is not found in the input schema.
    ///
    /// # Arguments
    /// * `input` - The input plan to add row addresses to.
    /// * `dataset` - The dataset to get the row id index from.
    /// * `rowaddr_pos` - The position in the output schema where to insert the row address.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        rowaddr_pos: usize,
    ) -> Result<Self> {
        // Need to know the physical position of the row id field, so we don't
        // have to do a schema lookup for every batch.
        let input_schema = input.schema();
        let rowid_pos = input_schema
            .fields()
            .iter()
            .position(|f| f.name() == ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Internal("rowid field not found in input schema".into())
            })?;

        let mut fields = input_schema.fields().iter().cloned().collect::<Vec<_>>();
        fields.insert(rowaddr_pos, Arc::new(ROW_ADDR_FIELD.clone()));
        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        let orderings = input
            .equivalence_properties()
            .oeq_class()
            .iter()
            .filter_map(|ordering| project_ordering(ordering, &output_schema));
        let equivalence_properties =
            EquivalenceProperties::new_with_orderings(output_schema.clone(), orderings);
        // Lookup preserves the driving side's order, including repeated keys.
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(equivalence_properties),
        );

        Ok(Self {
            input,
            dataset,
            rowid_pos,
            rowaddr_pos,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    async fn compute_row_addrs(dataset: &Dataset, row_ids: &ArrayRef) -> Result<ArrayRef> {
        let row_id_values = row_ids.as_primitive_opt::<UInt64Type>().ok_or_else(|| {
            DataFusionError::Internal("AddRowAddrExec: rowid column is not a UInt64Array".into())
        })?;
        if !dataset.manifest.uses_stable_row_ids() {
            return Ok(row_ids.clone());
        }
        let addresses = row_ids_to_row_addrs(dataset, row_id_values).await?;
        Ok(Arc::new(UInt64Array::from(
            addresses
                .into_iter()
                .map(|address| address.map(u64::from))
                .collect::<Vec<_>>(),
        )))
    }

    fn do_execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();

        let input_stream = self.input.execute(partition, context)?;

        let rowid_pos = self.rowid_pos;
        let rowaddr_pos = self.rowaddr_pos;
        let output_schema = self.output_schema.clone();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let elapsed_compute = baseline.elapsed_compute().clone();
        let stream = input_stream.then(move |batch_result| {
            let output_schema = output_schema.clone();
            let dataset = dataset.clone();
            let elapsed_compute = elapsed_compute.clone();
            async move {
                let batch = batch_result?;
                let row_addr = Self::compute_row_addrs(&dataset, batch.column(rowid_pos)).await?;
                let _t = elapsed_compute.timer();

                let mut columns = Vec::with_capacity(batch.num_columns() + 1);
                let existing_columns = batch.columns();
                columns.extend_from_slice(&existing_columns[..rowaddr_pos]);
                columns.push(row_addr);
                columns.extend_from_slice(&existing_columns[rowaddr_pos..]);

                Ok(RecordBatch::try_new(output_schema.clone(), columns)?)
            }
        });
        let stream = stream.map(move |batch| {
            let poll = baseline.record_poll(std::task::Poll::Ready(Some(batch)));
            match poll {
                std::task::Poll::Ready(Some(b)) => b,
                _ => unreachable!("record_poll preserves Ready(Some) input"),
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }
}

impl DisplayAs for AddRowAddrExec {
    fn fmt_as(
        &self,
        _format_type: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "AddRowAddrExec")
    }
}

impl ExecutionPlan for AddRowAddrExec {
    fn name(&self) -> &str {
        "AddRowAddrExec"
    }

    fn schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // We aren't doing much work here, best to avoid the thread overhead
        vec![false]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            Err(DataFusionError::Internal(
                "AddRowAddrExec: invalid number of children".into(),
            ))
        } else {
            Ok(Arc::new(Self::try_new(
                children.into_iter().next().unwrap(),
                self.dataset.clone(),
                self.rowaddr_pos,
            )?))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let schema = self.schema();
        let this = self.clone();
        let stream = futures::stream::once(async move { this.do_execute(partition, context) });
        let stream = stream.try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let mut stats = Statistics::clone(&input_stats[0]);

        let row_id_col_stats = stats.column_statistics.get(self.rowid_pos).ok_or_else(|| {
            DataFusionError::Internal("RowAddrExec: rowid column stats not found".into())
        })?;
        let row_addr_col_stats = ColumnStatistics {
            null_count: if self.dataset.manifest.uses_stable_row_ids() {
                Precision::Absent
            } else {
                row_id_col_stats.null_count
            },
            distinct_count: if self.dataset.manifest.uses_stable_row_ids() {
                Precision::Absent
            } else {
                row_id_col_stats.distinct_count
            },
            sum_value: Precision::Absent,
            max_value: Precision::Absent,
            min_value: Precision::Absent,
            byte_size: Precision::Absent,
        };

        let base_size = std::mem::size_of::<UInt64Array>();
        // Buffer size is the number of rows times 8 bytes per row, but there
        // is a minimum size of 64 bytes.
        let mut added_byte_size = stats
            .num_rows
            .map(|n| n * 8)
            .add(&Precision::Exact(base_size));
        if self.dataset.manifest.uses_stable_row_ids()
            || row_id_col_stats
                .null_count
                .get_value()
                .map(|v| *v > 0)
                .unwrap_or_default()
        {
            // Account for null buffer.
            added_byte_size = added_byte_size.add(&stats.num_rows.map(|n| n.div_ceil(8).max(64)));
        }
        stats.total_byte_size = stats.total_byte_size.add(&added_byte_size);
        if self.dataset.manifest.uses_stable_row_ids() {
            // Misses can introduce a validity bitmap even when every input ID
            // is non-null; its presence is only known after lookup.
            stats.total_byte_size = stats.total_byte_size.to_inexact();
        }
        stats
            .column_statistics
            .insert(self.rowaddr_pos, row_addr_col_stats);

        Ok(Arc::new(stats))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

#[derive(Debug)]
struct FragInfo {
    // The dataset offset of the first row in the fragment
    row_offset: u64,
    // The key is the cumulative row offset in the fragment, the value is the number of rows deleted so far
    deletion_vector: Option<Arc<DeletionVector>>,
}

/// Add a `_rowoffset` column to a stream of record batches that have a `_rowaddr` column.
///
/// The row offset is the number of rows between the current row and the first row in the dataset.
#[derive(Clone, Debug)]
pub struct AddRowOffsetExec {
    input: Arc<dyn ExecutionPlan>,
    row_addr_pos: usize,
    frag_id_to_offset: Arc<HashMap<u32, FragInfo>>,
    properties: Arc<PlanProperties>,
}

impl AddRowOffsetExec {
    fn internal_new(
        input: Arc<dyn ExecutionPlan>,
        frag_id_to_offset: Arc<HashMap<u32, FragInfo>>,
    ) -> LanceResult<Self> {
        let input_schema = input.schema();
        let row_addr_pos = input_schema.index_of(ROW_ADDR).map_err(|_| {
            LanceError::internal(format!("Input plan does not have a {} column", ROW_ADDR))
        })?;

        if input_schema.field_with_name(ROW_OFFSET).is_ok() {
            return Err(LanceError::internal(format!(
                "Input plan already has a {} column",
                ROW_OFFSET
            )));
        }

        let mut fields = input.schema().fields().iter().cloned().collect::<Vec<_>>();
        fields.push(Arc::new(ROW_OFFSET_FIELD.clone()));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema().metadata().clone(),
        ));

        let new_eq_props =
            EquivalenceProperties::new(schema).extend(input.properties().eq_properties.clone())?;
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(new_eq_props),
        );

        Ok(Self {
            input,
            row_addr_pos,
            frag_id_to_offset,
            properties,
        })
    }

    pub async fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
    ) -> LanceResult<Self> {
        let frag_id_to_offset = Self::compute_frag_id_to_offset(dataset).await?;
        Self::internal_new(input, frag_id_to_offset)
    }

    async fn compute_frag_id_to_offset(
        dataset: Arc<Dataset>,
    ) -> LanceResult<Arc<HashMap<u32, FragInfo>>> {
        let mut frag_id_to_offset = HashMap::new();
        let mut row_offset = 0;
        for frag in dataset.get_fragments() {
            let deletion_vector = frag.get_deletion_vector().await?;
            frag_id_to_offset.insert(
                frag.id() as u32,
                FragInfo {
                    row_offset,
                    deletion_vector,
                },
            );
            // Should be sync unless the dataset was written by an extremely old lance version
            row_offset += frag.count_rows(None).await? as u64;
        }

        Ok(Arc::new(frag_id_to_offset))
    }

    pub async fn compute_row_offset_array(
        row_addr: &ArrayRef,
        dataset: Arc<Dataset>,
    ) -> Result<ArrayRef> {
        let frag_id_to_offset = Self::compute_frag_id_to_offset(dataset).await?;
        Self::compute_row_offsets(row_addr, frag_id_to_offset.as_ref())
    }

    fn compute_row_offsets(
        row_addr: &ArrayRef,
        frag_id_to_offset: &HashMap<u32, FragInfo>,
    ) -> Result<ArrayRef> {
        let row_addr_values = row_addr.as_primitive::<UInt64Type>().values();
        let mut row_offsets = vec![0; row_addr_values.len()];
        // The deletion iterator only moves forward, so compute in address order and
        // scatter the offsets back into input order.
        let mut sorted_row_indices = (0..row_addr_values.len()).collect::<Vec<_>>();
        sorted_row_indices.sort_unstable_by_key(|index| row_addr_values[*index]);

        let mut last_frag_id = u32::MAX;
        let mut last_frag_offset = 0;
        let mut last_frag_delete_count = 0;
        let mut dv_iter = None;

        for row_index in sorted_row_indices {
            let addr = RowAddress::new_from_u64(row_addr_values[row_index]);
            let frag_id = addr.fragment_id();
            if frag_id != last_frag_id {
                last_frag_id = frag_id;
                let Some(frag_info) = frag_id_to_offset.get(&frag_id) else {
                    return Err(DataFusionError::External(Box::new(LanceError::internal(
                        format!(
                            "A row address referred to a fragment {} that wasn't in the frag_id_to_offset map",
                            frag_id
                        ),
                    ))));
                };
                last_frag_offset = frag_info.row_offset;
                last_frag_delete_count = 0;
                dv_iter = frag_info
                    .deletion_vector
                    .as_ref()
                    .map(|dv| dv.to_sorted_iter().peekable());
            }

            let row_offset = addr.row_offset();
            if let Some(dv_iter) = &mut dv_iter {
                while dv_iter.peek().is_some() {
                    if *dv_iter.peek().unwrap() < row_offset {
                        dv_iter.next();
                        last_frag_delete_count += 1;
                    } else {
                        break;
                    }
                }
                row_offsets[row_index] =
                    last_frag_offset + row_offset as u64 - last_frag_delete_count as u64;
            } else {
                row_offsets[row_index] = last_frag_offset + row_offset as u64;
            }
        }

        Ok(Arc::new(UInt64Array::from(row_offsets)))
    }
}

impl DisplayAs for AddRowOffsetExec {
    fn fmt_as(
        &self,
        _format_type: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "AddRowOffsetExec")
    }
}

impl ExecutionPlan for AddRowOffsetExec {
    fn name(&self) -> &str {
        "AddRowOffsetExec"
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

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let mut stats = Statistics::clone(&input_stats[0]);
        stats
            .column_statistics
            .push(ColumnStatistics::new_unknown());
        // The input's byte count excludes the appended row-offset array.
        stats.total_byte_size = Precision::Absent;
        Ok(Arc::new(stats))
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            Err(DataFusionError::Internal(
                "AddRowOffsetExec: invalid number of children".into(),
            ))
        } else {
            Ok(Arc::new(Self::internal_new(
                children.into_iter().next().unwrap(),
                self.frag_id_to_offset.clone(),
            )?))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let schema = self.schema();
        let row_addr_pos = self.row_addr_pos;
        let frag_id_to_offset = self.frag_id_to_offset.clone();
        let stream = input_stream.then(move |batch| {
            let schema = schema.clone();
            let row_addr_pos = row_addr_pos;
            let frag_id_to_offset = frag_id_to_offset.clone();
            async move {
                let batch = batch?;
                let row_addr = batch.column(row_addr_pos);
                let row_offsets = Self::compute_row_offsets(row_addr, frag_id_to_offset.as_ref())?;
                let mut columns = batch.columns().to_vec();
                columns.push(Arc::new(row_offsets));
                Ok(RecordBatch::try_new(schema.clone(), columns)?)
            }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::{Array, Int32Array, RecordBatchIterator, record_batch};
    use arrow_schema::{DataType, Field};
    use datafusion::physical_optimizer::{
        PhysicalOptimizerRule, ensure_requirements::EnsureRequirements,
    };
    use datafusion::physical_plan::{StatisticsArgs, StatisticsContext};
    use datafusion::{
        config::ConfigOptions, datasource::memory::MemorySourceConfig, prelude::SessionContext,
    };
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col};
    use datafusion_physical_plan::sorts::sort::SortExec;
    use futures::TryStreamExt;
    use lance_core::{ROW_ADDR, ROW_ID_FIELD};
    use lance_datafusion::exec::OneShotExec;
    use rstest::rstest;

    use crate::dataset::WriteParams;

    use super::*;

    async fn apply_to_batch(batch: RecordBatch, dataset: Arc<Dataset>) -> Result<RecordBatch> {
        let memory_exec = OneShotExec::from_batch(batch);
        let exec = AddRowAddrExec::try_new(Arc::new(memory_exec), dataset, 0)?;
        let session = SessionContext::new();
        let task_ctx = session.task_ctx();
        let stream = exec.execute(0, task_ctx)?;
        let batches = stream.try_collect::<Vec<_>>().await?;
        assert_eq!(batches.len(), 1);
        Ok(batches.into_iter().next().unwrap())
    }

    #[tokio::test]
    async fn test_address_style_ids() {
        // Creating a dataset with no stable row ids means that the row address
        // will be the same as the row id.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let reader = RecordBatchIterator::new(vec![], schema.clone());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: false,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let dataset = Arc::new(dataset);

        let rowids = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(result.as_ref(), rowids.as_ref() as &dyn Array);
        // The array should be just a copy of the _rowid array pointer.
        assert_eq!(Arc::as_ptr(&result), Arc::as_ptr(&rowids));
    }

    async fn sample_dataset_with_rowid_index() -> Arc<Dataset> {
        // Create a row id index
        // 0 -> 0
        // 1 -> 1 << 32
        // 2 -> 2 << 32
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);
        Arc::new(dataset)
    }

    #[tokio::test]
    async fn test_row_ids_no_nulls() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![0, 1 << 32, 2 << 32])).as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_row_ids_with_nulls() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![
            Some(2),
            None,
            Some(0),
            Some(2),
            Some(99),
        ]));
        let schema = Schema::new(vec![ROW_ID_FIELD.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await.unwrap();
        let result = result[ROW_ADDR].clone();

        assert_eq!(
            result.as_ref(),
            Arc::new(UInt64Array::from(vec![
                Some(2 << 32),
                None,
                Some(0),
                Some(2 << 32),
                None
            ]))
            .as_ref() as &dyn Array
        );
    }

    #[tokio::test]
    async fn test_invalid_schema() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(Int32Array::from(vec![0, 1, 2]));
        let schema = Schema::new(vec![Field::new("invalid", DataType::Int32, true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![rowids.clone()]).unwrap();

        let result = apply_to_batch(batch, dataset).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stats() {
        let dataset = sample_dataset_with_rowid_index().await;

        let rowids = Arc::new(UInt64Array::from(vec![Some(0), None, Some(2)]));
        let schema = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()]));
        let batch = RecordBatch::try_new(schema.clone(), vec![rowids.clone()]).unwrap();

        let memory_exec =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], schema, None).unwrap();
        let exec = AddRowAddrExec::try_new(memory_exec, dataset.clone(), 0).unwrap();
        let stats = StatisticsContext::new()
            .compute(&exec, &StatisticsArgs::new())
            .unwrap();
        let result = apply_to_batch(batch, dataset).await.unwrap();

        assert_eq!(stats.num_rows, Precision::Exact(3));
        assert_eq!(stats.column_statistics.len(), 2);
        assert_eq!(stats.column_statistics[0].null_count, Precision::Absent);
        assert_eq!(stats.column_statistics[1].null_count, Precision::Exact(1));

        let actual_byte_size = result
            .columns()
            .iter()
            .fold(0, |acc, col| acc + col.get_array_memory_size());
        assert_eq!(stats.total_byte_size, Precision::Inexact(actual_byte_size));
    }

    #[tokio::test]
    async fn test_row_ids_preserve_input_ordering() {
        let dataset = sample_dataset_with_rowid_index().await;
        let batch = arrow_array::record_batch!(("_rowid", UInt64, [2, 0, 1])).unwrap();
        let input = Arc::new(OneShotExec::from_batch(batch));
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(
            col(ROW_ID, &input.schema()).unwrap(),
        )])
        .unwrap();
        let sorted = Arc::new(SortExec::new(ordering, input));
        let exec = AddRowAddrExec::try_new(sorted, dataset, 0).unwrap();
        let expected = LexOrdering::new([PhysicalSortExpr::new_default(
            col(ROW_ID, &exec.schema()).unwrap(),
        )])
        .unwrap();
        assert_eq!(exec.maintains_input_order(), vec![true]);
        assert_eq!(exec.properties().output_ordering(), Some(&expected));

        let redundant = Arc::new(SortExec::new(expected, Arc::new(exec)));
        let optimized = EnsureRequirements::new()
            .optimize(redundant, &ConfigOptions::new())
            .unwrap();
        let rendered = datafusion_physical_plan::displayable(optimized.as_ref())
            .indent(false)
            .to_string();
        assert_eq!(rendered.matches("SortExec:").count(), 1, "{rendered}");
        let batches =
            datafusion_physical_plan::collect(optimized, SessionContext::new().task_ctx())
                .await
                .unwrap();
        let ids = batches
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[rstest]
    #[case::global(None, false)]
    #[case::first_partition(Some(0), false)]
    #[case::second_partition(Some(1), false)]
    #[case::unknown(None, true)]
    fn test_row_offset_statistics(
        #[case] partition: Option<usize>,
        #[case] has_unknown_statistics: bool,
    ) {
        let batch = record_batch!(
            ("payload", Utf8, [Some("a"), None]),
            (ROW_ADDR, UInt64, [0, 1])
        )
        .unwrap();
        let input: Arc<dyn ExecutionPlan> = if has_unknown_statistics {
            Arc::new(OneShotExec::from_batch(batch))
        } else {
            MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()], vec![batch.slice(0, 1)]],
                batch.schema(),
                None,
            )
            .unwrap()
        };
        let args = StatisticsArgs::new().with_partition(partition);
        let input_stats = StatisticsContext::new()
            .compute(input.as_ref(), &args)
            .unwrap();
        let exec = AddRowOffsetExec::internal_new(input.clone(), Arc::new(HashMap::new())).unwrap();
        let stats = StatisticsContext::new().compute(&exec, &args).unwrap();

        assert_eq!(stats.column_statistics.len(), exec.schema().fields().len());
        assert_eq!(stats.num_rows, input_stats.num_rows);
        assert_eq!(
            stats.column_statistics[..input_stats.column_statistics.len()],
            input_stats.column_statistics
        );
        assert_eq!(
            stats.column_statistics.last().unwrap(),
            &ColumnStatistics::new_unknown()
        );
        // Even an exact input size excludes the newly allocated row-offset array.
        assert_eq!(stats.total_byte_size, Precision::Absent);
        if !has_unknown_statistics {
            assert!(matches!(input_stats.total_byte_size, Precision::Exact(_)));
        }
        assert_eq!(
            StatisticsContext::new()
                .compute(input.as_ref(), &args)
                .unwrap(),
            input_stats
        );
    }

    #[test]
    fn test_row_offsets_with_unsorted_addresses() {
        let row_addrs: ArrayRef = Arc::new(UInt64Array::from(vec![
            u64::from(RowAddress::new_from_parts(0, 100)),
            u64::from(RowAddress::new_from_parts(0, 50)),
        ]));
        let frag_id_to_offset = HashMap::from([(
            0,
            FragInfo {
                row_offset: 1_000,
                deletion_vector: Some(Arc::new(DeletionVector::from_iter([10, 60]))),
            },
        )]);

        let row_offsets =
            AddRowOffsetExec::compute_row_offsets(&row_addrs, &frag_id_to_offset).unwrap();

        assert_eq!(
            row_offsets.as_primitive::<UInt64Type>().values(),
            &[1_098, 1_049]
        );
    }
}
