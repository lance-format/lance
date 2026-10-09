// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ScalarMemIndexExec`: a filter answered from a memtable's indexes, with MVCC
//! visibility.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow::compute::{filter_record_batch, prep_null_mask_filter, take_record_batch};
use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
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

use crate::dataset::mem_wal::index::{SearchContext, evaluate_index_filter};
use crate::dataset::mem_wal::memtable::scanner::exec::{scan_record_batch, take_projected_columns};
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// An index may decline a search matching more than `1 / MATCH_BUDGET_SHARE`
/// of the visible rows; reading them all is then cheaper.
const MATCH_BUDGET_SHARE: u64 = 16;

/// Matches always worth listing, whatever the share.
const MIN_MATCH_BUDGET: u64 = 4096;

/// Execution-plan node answering a filter from the memtable's indexes,
/// filtered by visibility.
pub struct ScalarMemIndexExec {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    /// The index searches the filter was split into.
    index_expr: ScalarIndexExpr,
    /// The whole filter, compiled. Applied to the candidates unless the
    /// indexes decided the filter alone.
    recheck: Option<PhysicalExprRef>,
    /// Whether the index searches are the whole filter.
    is_whole_filter: bool,
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

impl Debug for ScalarMemIndexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarMemIndexExec")
            .field("index_expr", &self.index_expr)
            .field("rechecked", &self.recheck.is_some())
            .field("readable_count", &self.readable_count)
            .field("with_row_id", &self.with_row_id)
            .field("with_row_address", &self.with_row_address)
            .finish()
    }
}

impl ScalarMemIndexExec {
    /// Read the rows `index_expr` selects from the first `readable_count`
    /// batches. `output_schema` is the projected schema, including `_rowid` and
    /// `_rowaddr` when requested.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        index_expr: ScalarIndexExpr,
        recheck: Option<PhysicalExprRef>,
        is_whole_filter: bool,
        readable_count: usize,
        projection: Option<Vec<usize>>,
        output_schema: SchemaRef,
        with_row_id: bool,
        with_row_address: bool,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Self {
            batch_store,
            indexes,
            index_expr,
            recheck,
            is_whole_filter,
            readable_count,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            with_row_id,
            with_row_address,
        }
    }

    /// Evaluate the index searches: the candidate positions, or `None` when
    /// every visible row is one, and whether they are the exact answer.
    fn query_index(&self) -> Result<(Option<Vec<u64>>, bool)> {
        let Some(max_readable_row) = self.batch_store.max_visible_row(self.readable_count) else {
            return Ok((Some(Vec::new()), true));
        };
        let visible_rows = max_readable_row + 1;
        let mut ctx = SearchContext::new(max_readable_row);
        // A declined search needs `recheck` to filter the rows, so set a budget
        // only when there is one.
        if self.recheck.is_some() {
            ctx = ctx.with_match_budget((visible_rows / MATCH_BUDGET_SHARE).max(MIN_MATCH_BUDGET));
        }
        let result = evaluate_index_filter(&self.index_expr, &self.indexes, &ctx)?;
        let exact = result.is_exact();
        Ok(if result.at_most.len() == visible_rows {
            (None, exact)
        } else {
            (Some(result.at_most.into()), exact)
        })
    }

    /// Read the candidate rows, one stored batch at a time: those at
    /// `candidates` (ascending), or every visible row when it is `None`. A
    /// batch without a candidate is not read, and `recheck` sees only the
    /// candidates.
    fn read_rows(
        &self,
        candidates: Option<&[u64]>,
        recheck: Option<&PhysicalExprRef>,
    ) -> DataFusionResult<Vec<RecordBatch>> {
        let mut results = Vec::new();
        let mut next = 0;
        for stored in self.batch_store.iter().take(self.readable_count) {
            let start = stored.row_offset;
            let end = start + stored.num_rows as u64;
            // The rows read, and their positions; `None` when they are the
            // whole batch.
            let (mut data, mut positions) = match candidates {
                Some(candidates) => {
                    if next == candidates.len() {
                        break;
                    }
                    let first = next;
                    while next < candidates.len() && candidates[next] < end {
                        next += 1;
                    }
                    let in_batch = &candidates[first..next];
                    if in_batch.is_empty() {
                        continue;
                    }
                    if in_batch.len() == stored.num_rows {
                        (scan_record_batch(&stored.data)?, None)
                    } else {
                        let rows = UInt32Array::from_iter_values(
                            in_batch.iter().map(|position| (position - start) as u32),
                        );
                        let taken = take_record_batch(&stored.data, &rows)?;
                        (scan_record_batch(&taken)?, Some(in_batch.to_vec()))
                    }
                }
                None => (scan_record_batch(&stored.data)?, None),
            };
            if let Some(recheck) = recheck {
                let keep = recheck.evaluate(&data)?.into_array(data.num_rows())?;
                let keep = keep.as_boolean_opt().ok_or_else(|| {
                    DataFusionError::Internal("a filter must evaluate to a boolean".to_string())
                })?;
                // A null result is not a match, as it is not in a full scan.
                let keep = if keep.null_count() > 0 {
                    prep_null_mask_filter(keep)
                } else {
                    keep.clone()
                };
                (data, positions) = retain(data, positions, start, &keep)?;
            }
            if data.num_rows() == 0 {
                continue;
            }

            let mut columns: Vec<Arc<dyn Array>> = match &self.projection {
                Some(projection) => take_projected_columns(
                    data.columns(),
                    data.schema().fields(),
                    projection,
                    self.output_schema.as_ref(),
                    data.num_rows(),
                )?,
                None => data.columns().to_vec(),
            };
            if self.with_row_id || self.with_row_address {
                let row_positions: Arc<dyn Array> = Arc::new(match positions {
                    Some(positions) => UInt64Array::from(positions),
                    None => UInt64Array::from_iter_values(start..end),
                });
                if self.with_row_id {
                    columns.push(row_positions.clone());
                }
                // A memtable row's address is its position.
                if self.with_row_address {
                    columns.push(row_positions);
                }
            }
            results.push(RecordBatch::try_new(self.output_schema.clone(), columns)?);
        }
        Ok(results)
    }
}

/// `data` and its row `positions` narrowed to the rows `keep` selects;
/// `positions` is `None` while `data` is the whole batch starting at `start`.
fn retain(
    data: RecordBatch,
    positions: Option<Vec<u64>>,
    start: u64,
    keep: &BooleanArray,
) -> DataFusionResult<(RecordBatch, Option<Vec<u64>>)> {
    if keep.true_count() == data.num_rows() {
        return Ok((data, positions));
    }
    let kept = keep.values().set_indices();
    let positions = match positions {
        Some(positions) => kept.map(|row| positions[row]).collect(),
        None => kept.map(|row| start + row as u64).collect(),
    };
    Ok((filter_record_batch(&data, keep)?, Some(positions)))
}

impl DisplayAs for ScalarMemIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ScalarMemIndexExec: query={}, whole_filter={}, with_row_id={}, with_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    self.with_row_id,
                    self.with_row_address
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ScalarMemIndexExec\nquery={}\nwhole_filter={}\nwith_row_id={}\nwith_row_address={}",
                    self.index_expr.to_expr(),
                    self.recheck.is_some(),
                    self.with_row_id,
                    self.with_row_address
                )
            }
        }
    }
}

impl ExecutionPlan for ScalarMemIndexExec {
    fn name(&self) -> &str {
        "ScalarMemIndexExec"
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
            return Err(DataFusionError::Internal(
                "ScalarMemIndexExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let (positions, exact) = self.query_index()?;

        // Candidates from a narrowing or partial answer still need the filter.
        let recheck = if !exact || !self.is_whole_filter {
            let Some(recheck) = &self.recheck else {
                return Err(DataFusionError::Internal(
                    "the indexes did not decide the filter, but no filter was given to re-check with"
                        .to_string(),
                ));
            };
            Some(recheck)
        } else {
            None
        };
        let batches = self.read_rows(positions.as_deref(), recheck)?;

        let stream = stream::iter(batches.into_iter().map(Ok)).boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: Statistics::unknown_column(&self.schema()),
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

    /// Past the match budget every visible row is read with the filter, and a
    /// batch past the readable count stays unread. The filter is wider than the
    /// index query, so only a declined search returns ids below 100.
    #[tokio::test]
    async fn a_broad_answer_reads_every_visible_row_with_the_filter() {
        let schema = create_test_schema();
        let (batch_store, indexes) = broad_memtable(&schema);
        let visible_rows = (READABLE_BATCHES * ROWS_PER_BATCH) as u64;

        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Int32(Some(100))),
            std::ops::Bound::Unbounded,
        );
        let budget = (visible_rows / MATCH_BUDGET_SHARE).max(MIN_MATCH_BUDGET);
        let ctx = SearchContext::new(visible_rows - 1).with_match_budget(budget);
        assert!(
            indexes
                .get_index("id_idx")
                .unwrap()
                .search(&query, &ctx)
                .unwrap()
                .is_none(),
            "the B-tree must decline this many matches"
        );

        let planner = lance_datafusion::planner::Planner::new(schema.clone());
        let recheck = planner
            .create_physical_expr(&planner.parse_filter("id >= 50").unwrap())
            .unwrap();
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));
        let exec = ScalarMemIndexExec::new(
            batch_store,
            Arc::new(indexes),
            search("id_idx", "id", query),
            Some(recheck),
            true,
            READABLE_BATCHES,
            None,
            schema_with_rowid,
            true,
            false,
        );
        let batches: Vec<RecordBatch> = exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let (ids, row_ids) = ids_and_row_ids(&batches);
        let expected: Vec<i32> = (50..visible_rows as i32).collect();
        assert_eq!(ids, expected);
        assert_eq!(
            row_ids,
            expected.iter().map(|id| *id as u64).collect::<Vec<_>>()
        );
    }

    const ROWS_PER_BATCH: usize = 1_000;
    const READABLE_BATCHES: usize = 20;

    /// `READABLE_BATCHES` readable batches of `ROWS_PER_BATCH` ids counting up
    /// from 0, plus one more batch past the readable count, with a B-tree on
    /// `id`.
    fn broad_memtable(schema: &SchemaRef) -> (Arc<BatchStore>, IndexStore) {
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());
        for n in 0..=READABLE_BATCHES {
            let start = n * ROWS_PER_BATCH;
            let batch = create_test_batch(schema, start as i32, ROWS_PER_BATCH);
            batch_store.append(batch.clone()).unwrap();
            indexes
                .insert_with_batch_position(&batch, start as u64, Some(n))
                .unwrap();
        }
        (batch_store, indexes)
    }

    /// With no filter to fall back on, a search past the budget is still
    /// answered.
    #[tokio::test]
    async fn a_search_with_no_filter_to_fall_back_on_is_never_declined() {
        let schema = create_test_schema();
        let (batch_store, indexes) = broad_memtable(&schema);
        let query = SargableQuery::Range(
            std::ops::Bound::Included(ScalarValue::Int32(Some(100))),
            std::ops::Bound::Unbounded,
        );
        let exec = ScalarMemIndexExec::new(
            batch_store,
            Arc::new(indexes),
            search("id_idx", "id", query),
            None,
            true,
            READABLE_BATCHES,
            None,
            schema,
            false,
            false,
        );
        let batches: Vec<RecordBatch> = exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, READABLE_BATCHES * ROWS_PER_BATCH - 100);
    }

    /// Candidates an index only narrowed to are checked against the filter,
    /// both when they fill a batch and when they are a few of its rows, and
    /// each kept row keeps its own row id.
    #[tokio::test]
    async fn narrowed_candidates_are_checked_against_the_filter() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(4));
        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());
        for n in 0..2 {
            let batch = create_test_batch(&schema, n * 10, 10);
            batch_store.append(batch.clone()).unwrap();
            indexes
                .insert_with_batch_position(&batch, (n * 10) as u64, Some(n as usize))
                .unwrap();
        }

        // All of the first batch and half of the second are candidates.
        let ScalarIndexExpr::Query(mut narrowing) = search(
            "id_idx",
            "id",
            SargableQuery::Range(
                std::ops::Bound::Unbounded,
                std::ops::Bound::Excluded(ScalarValue::Int32(Some(15))),
            ),
        ) else {
            unreachable!()
        };
        narrowing.needs_recheck = true;
        let planner = lance_datafusion::planner::Planner::new(schema.clone());
        let filter = planner
            .optimize_expr(planner.parse_filter("id < 15 AND id % 2 = 0").unwrap())
            .unwrap();
        let recheck = planner.create_physical_expr(&filter).unwrap();
        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));
        let exec = ScalarMemIndexExec::new(
            batch_store,
            Arc::new(indexes),
            ScalarIndexExpr::Query(narrowing),
            Some(recheck),
            true,
            2,
            None,
            schema_with_rowid,
            true,
            false,
        );
        let batches: Vec<RecordBatch> = exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let (ids, row_ids) = ids_and_row_ids(&batches);
        assert_eq!(ids, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(row_ids, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    }

    /// The `id` and `_rowid` columns of `batches`, in order.
    fn ids_and_row_ids(batches: &[RecordBatch]) -> (Vec<i32>, Vec<u64>) {
        let ids = batches
            .iter()
            .flat_map(|batch| {
                batch["id"]
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let row_ids = batches
            .iter()
            .flat_map(|batch| {
                batch["_rowid"]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        (ids, row_ids)
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
    async fn an_equality_returns_its_row() {
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
            SargableQuery::Equals(ScalarValue::Int32(Some(5))),
        );

        let exec = ScalarMemIndexExec::new(
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
        );

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn a_list_returns_each_listed_row() {
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

        let exec = ScalarMemIndexExec::new(
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
        );

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn a_row_past_the_readable_batches_is_not_returned() {
        let schema = create_test_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(100));

        let mut registry = IndexStore::new();
        registry.add_btree("id_idx".to_string(), 0, "id".to_string());

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

        // Only the first batch is readable; id 15 is in the second.
        let exec = ScalarMemIndexExec::new(
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
        );

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0);

        // Both batches readable.
        let exec = ScalarMemIndexExec::new(
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
        );

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn a_row_id_is_the_rows_position() {
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

        let exec = ScalarMemIndexExec::new(
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
        );

        let debug_str = format!("{:?}", exec);
        assert!(debug_str.contains("with_row_id: true"));
        assert!(debug_str.contains("with_row_address: false"));

        let ctx = Arc::new(TaskContext::default());
        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);

        let batch = &batches[0];
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(2).name(), "_rowid");

        let row_ids = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 5);
    }

    #[tokio::test]
    async fn the_plan_shows_the_query_and_its_flags() {
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

        let exec: Arc<dyn ExecutionPlan> = Arc::new(ScalarMemIndexExec::new(
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
        ));

        assert_plan_node_equals(
            exec,
            "ScalarMemIndexExec: query=id = Int32(5), whole_filter=false, with_row_id=false, with_row_address=false",
        )
        .await
        .unwrap();

        let schema_with_rowid = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_rowid", DataType::UInt64, true),
        ]));

        let exec: Arc<dyn ExecutionPlan> = Arc::new(ScalarMemIndexExec::new(
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
        ));

        assert_plan_node_equals(
            exec,
            "ScalarMemIndexExec: query=id = Int32(5), whole_filter=false, with_row_id=true, with_row_address=false",
        )
        .await
        .unwrap();
    }
}
