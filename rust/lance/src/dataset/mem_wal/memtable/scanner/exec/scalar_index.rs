// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ScalarIndexExec` — scalar-index queries over a memtable, with MVCC visibility.
//!
//! Serves the built-in B-tree and any index a registered plugin maintains: the
//! predicate resolves to row positions, and everything after that — position to
//! batch, projection, row id and row address — is the same work either way.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::Result as DataFusionResult;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExprRef};
use futures::stream::{self, StreamExt};
use lance_core::Result;

use lance_index::scalar::expression::ScalarIndexExpr;

use crate::dataset::mem_wal::index::{SearchContext, evaluate_index_filter, positions};
use crate::dataset::mem_wal::memtable::scanner::exec::{scan_record_batch, take_projected_columns};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Execution-plan node answering a filter from the memtable's indexes,
/// filtered by visibility.
pub struct ScalarIndexExec {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    /// The index searches the filter was split into: a tree of `AND`/`OR` over
    /// per-index queries, built by the same pass the base table's scan uses.
    index_expr: ScalarIndexExpr,
    /// The whole filter, compiled.
    ///
    /// Applied to the rows the indexes narrowed to. It is the filter itself
    /// rather than a re-reading of the queries, because the two are not always
    /// the same question — an R-tree narrows a spatial relation down to a
    /// bounding box, and only the filter knows the relation that box stands in
    /// for. `None` when the indexes answered exactly and nothing is left.
    recheck: Option<PhysicalExprRef>,
    /// Whether the index tree is the whole filter. When some condition had no
    /// index to answer it, an exact index answer is still only a superset.
    is_filter_covered: bool,
    readable_count: usize,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    /// Whether to include _rowid column (row position) in output.
    with_row_id: bool,
    /// Whether to include _rowaddr column (same as row position) in output.
    with_row_address: bool,
}

impl Debug for ScalarIndexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarIndexExec")
            .field("index_expr", &self.index_expr)
            .field("rechecked", &self.recheck.is_some())
            .field("readable_count", &self.readable_count)
            .field("with_row_id", &self.with_row_id)
            .field("with_row_address", &self.with_row_address)
            .finish()
    }
}

impl ScalarIndexExec {
    /// Create a new ScalarIndexExec.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - Lock-free batch store containing data
    /// * `indexes` - Index registry holding the scalar index for the column
    /// * `index_expr` - The index searches the filter was split into
    /// * `recheck` - The whole filter, compiled, for rows an index only narrowed
    /// * `readable_count` - Exclusive count of batch positions this scan may read
    /// * `projection` - Optional column indices to project
    /// * `output_schema` - Schema after projection (should include _rowid/_rowaddr if requested)
    /// * `with_row_id` - Whether to include _rowid column (row position)
    /// * `with_row_address` - Whether to include _rowaddr column (same as row position)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        index_expr: ScalarIndexExpr,
        recheck: Option<PhysicalExprRef>,
        is_filter_covered: bool,
        readable_count: usize,
        projection: Option<Vec<usize>>,
        output_schema: SchemaRef,
        with_row_id: bool,
        with_row_address: bool,
    ) -> Result<Self> {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            batch_store,
            indexes,
            index_expr,
            recheck,
            is_filter_covered,
            readable_count,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            with_row_id,
            with_row_address,
        })
    }

    /// Last row position within `readable_count`, or None if nothing is
    /// readable.
    fn compute_max_readable_row(&self) -> Option<u64> {
        let mut max_readable_row_exclusive: u64 = 0;
        let mut current_row: u64 = 0;

        for (batch_position, stored_batch) in self.batch_store.iter().enumerate() {
            let batch_end = current_row + stored_batch.num_rows as u64;
            if batch_position < self.readable_count {
                max_readable_row_exclusive = batch_end;
            }
            current_row = batch_end;
        }

        if max_readable_row_exclusive > 0 {
            Some(max_readable_row_exclusive - 1)
        } else {
            None
        }
    }

    /// Evaluate the index searches and return matching row positions, filtered
    /// by visibility, with whether the answer still needs the filter applied.
    fn query_index(&self) -> (Vec<u64>, bool) {
        let Some(max_readable_row) = self.compute_max_readable_row() else {
            return (vec![], true);
        };
        let ctx = SearchContext::new(max_readable_row);
        match evaluate_index_filter(&self.index_expr, &self.indexes, &ctx) {
            Ok(result) => positions(result),
            // A failing index must not silently answer "no rows". Hand back
            // every visible row and let the filter decide, which is a scan
            // done in place.
            Err(_) => ((0..=max_readable_row).collect(), false),
        }
    }

    /// Keep only the candidate rows the filter accepts.
    ///
    /// Evaluated once per stored batch rather than once per row: the candidates
    /// from one batch arrive together, and a spatial or string predicate costs
    /// far more per call than the mask does per row.
    fn retain_matching_rows(
        &self,
        batch_rows: Vec<(usize, usize, u64)>,
        recheck: &PhysicalExprRef,
    ) -> DataFusionResult<Vec<(usize, usize, u64)>> {
        let mut kept = Vec::with_capacity(batch_rows.len());
        let mut current: Option<(usize, BooleanArray)> = None;
        for (batch_id, row_in_batch, position) in batch_rows {
            if current.as_ref().is_none_or(|(id, _)| *id != batch_id) {
                let Some(stored) = self.batch_store.get(batch_id) else {
                    continue;
                };
                let evaluated = recheck.evaluate(&stored.data)?;
                let mask = evaluated.into_array(stored.data.num_rows())?;
                let mask = mask
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Internal(
                            "a filter must evaluate to a boolean".to_string(),
                        )
                    })?
                    .clone();
                current = Some((batch_id, mask));
            }
            // A null result is not a match, as it is not in a full scan.
            if current
                .as_ref()
                .is_some_and(|(_, mask)| mask.is_valid(row_in_batch) && mask.value(row_in_batch))
            {
                kept.push((batch_id, row_in_batch, position));
            }
        }
        Ok(kept)
    }

    /// Convert row positions to batch_id, row_within_batch, and original row_position tuples.
    fn positions_to_batch_rows(&self, positions: &[u64]) -> Vec<(usize, usize, u64)> {
        // Build a map of batch_id -> (start_row, end_row)
        let mut batch_ranges = Vec::new();
        let mut current_row = 0usize;

        for stored_batch in self.batch_store.iter() {
            let batch_start = current_row;
            let batch_end = current_row + stored_batch.num_rows;
            batch_ranges.push((batch_start, batch_end));
            current_row = batch_end;
        }

        // Batch ranges are contiguous and ascending, so the owning batch is a
        // binary search rather than a walk. A linear scan here cost one pass
        // over every batch for every matching row.
        let mut result = Vec::with_capacity(positions.len());
        for &pos in positions {
            let pos_usize = pos as usize;
            let found = batch_ranges.partition_point(|(start, _)| *start <= pos_usize);
            if found == 0 {
                continue;
            }
            let batch_id = found - 1;
            let (start, end) = batch_ranges[batch_id];
            if pos_usize < end {
                result.push((batch_id, pos_usize - start, pos));
            }
        }
        result
    }

    /// Materialize rows from batch store.
    fn materialize_rows(
        &self,
        batch_rows: &[(usize, usize, u64)],
    ) -> DataFusionResult<Vec<RecordBatch>> {
        if batch_rows.is_empty() {
            return Ok(vec![]);
        }

        // Group rows by batch, preserving row_position for _rowid
        let mut batches_to_rows: std::collections::HashMap<usize, Vec<(usize, u64)>> =
            std::collections::HashMap::new();
        for &(batch_id, row_in_batch, row_position) in batch_rows {
            batches_to_rows
                .entry(batch_id)
                .or_default()
                .push((row_in_batch, row_position));
        }

        let mut results = Vec::new();
        for (batch_id, rows_with_positions) in batches_to_rows {
            if let Some(stored) = self.batch_store.get(batch_id) {
                let data = scan_record_batch(&stored.data)?;
                // Extract row indices and row positions
                let row_indices: Vec<u32> = rows_with_positions
                    .iter()
                    .map(|&(row_in_batch, _)| row_in_batch as u32)
                    .collect();
                let row_positions: Vec<u64> = rows_with_positions
                    .iter()
                    .map(|&(_, row_position)| row_position)
                    .collect();

                // Use take to select specific rows
                let indices = arrow_array::UInt32Array::from(row_indices);

                let columns: std::result::Result<Vec<_>, datafusion::error::DataFusionError> = data
                    .columns()
                    .iter()
                    .map(|col| {
                        arrow_select::take::take(col.as_ref(), &indices, None).map_err(|e| {
                            datafusion::error::DataFusionError::ArrowError(Box::new(e), None)
                        })
                    })
                    .collect();

                let columns = columns?;

                // Apply projection
                let source_schema = data.schema();
                let mut final_columns: Vec<Arc<dyn arrow_array::Array>> =
                    if let Some(ref proj_indices) = self.projection {
                        take_projected_columns(
                            &columns,
                            source_schema.fields(),
                            proj_indices,
                            self.output_schema.as_ref(),
                            row_positions.len(),
                        )?
                    } else {
                        columns
                    };

                // Add _rowid column if requested
                if self.with_row_id {
                    final_columns.push(Arc::new(UInt64Array::from(row_positions.clone())));
                }

                // Add _rowaddr column if requested (same value as row position)
                if self.with_row_address {
                    final_columns.push(Arc::new(UInt64Array::from(row_positions)));
                }

                let batch = RecordBatch::try_new(self.output_schema.clone(), final_columns)?;
                results.push(batch);
            }
        }

        Ok(results)
    }
}

impl DisplayAs for ScalarIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ScalarIndexExec: query={}, rechecked={}, with_row_id={}, with_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    self.with_row_id,
                    self.with_row_address
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ScalarIndexExec\nquery={}\nrechecked={}\nwith_row_id={}\nwith_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    self.with_row_id,
                    self.with_row_address
                )
            }
        }
    }
}

impl ExecutionPlan for ScalarIndexExec {
    fn name(&self) -> &str {
        "ScalarIndexExec"
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        // Query the index
        let (positions, exact) = self.query_index();

        // Convert positions to batch/row pairs with visibility filtering
        let mut batch_rows = self.positions_to_batch_rows(&positions);

        // An index that only narrows hands back candidates, and so does an
        // exact index answer to part of the filter, so the filter decides here.
        // Dropping this would surface rows that do not match.
        if !exact || !self.is_filter_covered {
            let Some(recheck) = &self.recheck else {
                return Err(datafusion::error::DataFusionError::Internal(
                    "the indexes did not decide the filter, but no filter was given to re-check with"
                        .to_string(),
                ));
            };
            batch_rows = self.retain_matching_rows(batch_rows, recheck)?;
        }

        // Materialize the rows
        let batches = self.materialize_rows(&batch_rows)?;

        let stream = stream::iter(batches.into_iter().map(Ok)).boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        // We can't know the exact count without querying the index
        Ok(Arc::new(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use futures::TryStreamExt;
    use lance_index::scalar::SargableQuery;
    use lance_index::scalar::expression::ScalarIndexSearch;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &Schema, start_id: i32, count: usize) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|id| format!("name_{}", id)).collect();

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    /// One index search, the way the expression pass produces it.
    fn search(index_name: &str, column: &str, query: SargableQuery) -> ScalarIndexExpr {
        ScalarIndexExpr::Query(ScalarIndexSearch {
            column: column.to_string(),
            index_name: index_name.to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(query),
            needs_recheck: false,
            fragment_bitmap: None,
        })
    }

    #[tokio::test]
    async fn test_btree_index_eq_query() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        // Create index registry with btree index on "id" (field_id = 0)
        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert test data and update index
        let batch = create_test_batch(&schema, 0, 10);
        registry.insert(&batch, 0).unwrap();
        batch_store.append(batch).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1, // readable_count (batch at position 0)
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find one row with id=5
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_btree_index_in_query() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        let batch = create_test_batch(&schema, 0, 10);
        registry.insert(&batch, 0).unwrap();
        batch_store.append(batch).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::IsIn(vec![
                ScalarValue::Int32(Some(2)),
                ScalarValue::Int32(Some(5)),
                ScalarValue::Int32(Some(8)),
            ]),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1,
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find 3 rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn test_btree_index_visibility() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert two batches at positions 0 and 1
        let batch1 = create_test_batch(&schema, 0, 10);
        let batch2 = create_test_batch(&schema, 10, 10);
        registry.insert(&batch1, 0).unwrap();
        registry.insert(&batch2, 10).unwrap();
        batch_store.append(batch1).unwrap();
        batch_store.append(batch2).unwrap();

        let indexes = Arc::new(registry);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(15))),
        );

        // Query with max_readable=0 should not see batch at position 1
        let exec = ScalarIndexExec::new(
            batch_store.clone(),
            indexes.clone(),
            index_expr.clone(),
            None,
            true,
            1,
            None,
            schema.clone(),
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);

        // Query with max_readable=1 should see both batches
        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            2,
            None,
            schema,
            false,
            false,
        )
        .unwrap();

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_btree_index_with_row_id() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Insert batch with 10 rows at position 0
        let batch = create_test_batch(&schema, 0, 10);
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();

        let indexes = Arc::new(indexes);

        // Add _rowid to schema
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        let exec = ScalarIndexExec::new(
            batch_store,
            indexes,
            index_expr,
            None,
            true,
            1,
            None,
            schema_with_rowid.clone(),
            true,
            false,
        )
        .unwrap();

        // Verify the plan output
        let debug_str = format!("{:?}", exec);
        assert!(debug_str.contains("with_row_id: true"));
        assert!(debug_str.contains("with_row_address: false"));

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // Should find one row with id=5
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);

        // Verify _rowid column is present and has correct value
        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(2).name(), "_rowid");

        let row_ids = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 5); // Row position for id=5 is 5
    }

    #[tokio::test]
    async fn test_btree_plan_display() {
        use crate::utils::test::assert_plan_node_equals;
        use datafusion::physical_plan::ExecutionPlan;

        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());

        let batch = create_test_batch(&schema, 0, 10);
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();

        let indexes = Arc::new(indexes);

        let index_expr = search(
            "id_idx",
            "id",
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        // Test plan display without _rowid
        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            ScalarIndexExec::new(
                batch_store.clone(),
                indexes.clone(),
                index_expr.clone(),
                None,
                true,
                1,
                None,
                schema.clone(),
                false,
                false,
            )
            .unwrap(),
        );

        assert_plan_node_equals(
            exec,
            "ScalarIndexExec: query=id = Int32(5), rechecked=false, with_row_id=false, with_row_address=false",
        )
        .await
        .unwrap();

        // Test plan display with _rowid
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));

        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            ScalarIndexExec::new(
                batch_store,
                indexes,
                index_expr,
                None,
                true,
                1,
                None,
                schema_with_rowid,
                true,
                false,
            )
            .unwrap(),
        );

        assert_plan_node_equals(
            exec,
            "ScalarIndexExec: query=id = Int32(5), rechecked=false, with_row_id=true, with_row_address=false",
        )
        .await
        .unwrap();
    }
}
