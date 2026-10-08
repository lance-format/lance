// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `ScalarMemIndexExec`: a filter answered from a memtable's indexes, with MVCC
//! visibility.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow::compute::{filter_record_batch, prep_null_mask_filter, take_record_batch};
use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt32Array, UInt64Array};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
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

/// An index may decline a search matching more than the visible rows divided
/// by this; reading them all is then cheaper.
const MATCH_BUDGET_DIVISOR: u64 = 16;

/// Matches always worth listing, however few rows are visible.
const MIN_MATCH_BUDGET: u64 = 4096;

/// The divisor for a read that keeps only each key's newest version, where
/// each match also costs a seek in the primary-key index.
const NEWEST_CHECK_MATCH_BUDGET_DIVISOR: u64 = 8;

/// Metric counting the matches checked for being their key's newest version.
pub const NEWEST_CHECKS_METRIC: &str = "newest_checks";

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
    /// Keep only the newest visible version of each primary key, given the key
    /// columns' positions in the stored batches. An older version can match a
    /// filter its key's newest version fails, and must not be returned.
    newest_of: Option<Vec<usize>>,
    /// Run instead when the indexes match too many rows to be worth listing.
    /// Required with `newest_of`.
    broad_fallback: Option<Arc<dyn ExecutionPlan>>,
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
            newest_of: None,
            broad_fallback: None,
        }
    }

    /// Return only the newest visible version of each primary key, reading
    /// with `broad_fallback` when the indexes match too many rows to list.
    pub fn with_newest_check(
        mut self,
        pk_indices: Vec<usize>,
        broad_fallback: Arc<dyn ExecutionPlan>,
    ) -> Self {
        self.newest_of = Some(pk_indices);
        self.broad_fallback = Some(broad_fallback);
        self
    }

    /// How many matches are worth listing out of `visible_rows`.
    fn match_budget(&self, visible_rows: u64) -> u64 {
        if self.newest_of.is_some() {
            visible_rows / NEWEST_CHECK_MATCH_BUDGET_DIVISOR
        } else {
            (visible_rows / MATCH_BUDGET_DIVISOR).max(MIN_MATCH_BUDGET)
        }
    }

    /// Evaluate the index searches: the candidate positions, or `None` when
    /// every visible row is one, and whether they are the exact answer.
    fn query_index(&self) -> Result<(Option<Vec<u64>>, bool)> {
        let Some(max_readable_row) = self.batch_store.max_visible_row(self.readable_count) else {
            return Ok((Some(Vec::new()), true));
        };
        let visible_rows = max_readable_row + 1;
        let budget = self.match_budget(visible_rows);
        let mut ctx = SearchContext::new(max_readable_row);
        // A declined search needs `recheck` or the fallback to read the rows,
        // so set a budget only when there is one.
        if self.recheck.is_some() || self.broad_fallback.is_some() {
            ctx = ctx.with_match_budget(budget);
        }
        let result = evaluate_index_filter(&self.index_expr, &self.indexes, &ctx)?;
        let exact = result.is_exact();
        // An index may list past its budget; the fallback is then cheaper.
        let past_budget = self.broad_fallback.is_some() && result.at_most.len() > budget;
        Ok(if past_budget || result.at_most.len() == visible_rows {
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
        newest_checks: &Count,
    ) -> DataFusionResult<Vec<RecordBatch>> {
        let max_readable_row = self.batch_store.max_visible_row(self.readable_count);
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
            if let (Some(pk_indices), Some(max_readable_row)) = (&self.newest_of, max_readable_row)
            {
                newest_checks.add(data.num_rows());
                let keep = self.keep_newest(
                    &data,
                    positions.as_deref(),
                    start,
                    pk_indices,
                    max_readable_row,
                )?;
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

    /// Which of `data`'s rows are their key's newest visible version: one seek
    /// in the primary-key index per row. `positions` are the rows' positions,
    /// or `None` when `data` is the whole batch starting at `start`.
    fn keep_newest(
        &self,
        data: &RecordBatch,
        positions: Option<&[u64]>,
        start: u64,
        pk_indices: &[usize],
        max_readable_row: u64,
    ) -> DataFusionResult<BooleanArray> {
        let rows = data.num_rows();
        let mut keep = BooleanBufferBuilder::new(rows);
        let mut values = Vec::with_capacity(pk_indices.len());
        for row in 0..rows {
            values.clear();
            for &column in pk_indices {
                values.push(ScalarValue::try_from_array(data.column(column), row)?);
            }
            let position = positions.map_or(start + row as u64, |positions| positions[row]);
            keep.append(
                self.indexes
                    .pk_is_newest(&values, position, max_readable_row),
            );
        }
        Ok(BooleanArray::new(keep.finish(), None))
    }
}

/// `data` and its row `positions` narrowed to the rows `keep` selects;
/// `positions` is `None` while `data` is the whole batch starting at `start`.
/// `keep` has no nulls: its bits pick the positions.
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
                    "ScalarMemIndexExec: query={}, whole_filter={}{}, with_row_id={}, with_row_address={}",
                    self.index_expr.to_expr(),
                    self.is_whole_filter,
                    if self.newest_of.is_some() {
                        ", newest_only=true"
                    } else {
                        ""
                    },
                    self.with_row_id,
                    self.with_row_address
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ScalarMemIndexExec\nquery={}\nwhole_filter={}\nnewest_only={}\nwith_row_id={}\nwith_row_address={}",
                    self.index_expr.to_expr(),
                    self.is_whole_filter,
                    self.newest_of.is_some(),
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
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let newest_checks =
            MetricBuilder::new(&self.metrics).counter(NEWEST_CHECKS_METRIC, partition);
        let (positions, exact) = self.query_index()?;
        if positions.is_none()
            && let Some(fallback) = &self.broad_fallback
        {
            return fallback.execute(partition, context);
        }

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
        let batches = self.read_rows(positions.as_deref(), recheck, &newest_checks)?;

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
    use std::ops::Bound;

    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::common::ScalarValue;
    use futures::TryStreamExt;
    use lance_index::scalar::SargableQuery;
    use lance_index::scalar::expression::ScalarIndexSearch;

    const ROWS_PER_BATCH: usize = 1_000;
    const READABLE_BATCHES: usize = 20;

    type Memtable = (Arc<BatchStore>, Arc<IndexStore>);

    /// `id` and `name`, plus `_rowid` when `with_row_id`.
    fn schema(with_row_id: bool) -> SchemaRef {
        let mut fields = vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ];
        if with_row_id {
            fields.push(Field::new("_rowid", DataType::UInt64, true));
        }
        Arc::new(Schema::new(fields))
    }

    /// `batches` batches of `rows` ids counting up from 0, so a row's id is its
    /// position, with a B-tree `id_idx` on `id`.
    fn memtable(batches: usize, rows: usize) -> Memtable {
        let batch_store = Arc::new(BatchStore::with_capacity(batches));
        let mut indexes = IndexStore::new();
        indexes.add_btree("id_idx".to_string(), 0, "id".to_string());
        for position in 0..batches {
            let start = position * rows;
            let ids: Vec<i32> = (start as i32..(start + rows) as i32).collect();
            let names: Vec<String> = ids.iter().map(|id| format!("name_{id}")).collect();
            let batch = RecordBatch::try_new(
                schema(false),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(StringArray::from(names)),
                ],
            )
            .unwrap();
            batch_store.append(batch.clone()).unwrap();
            indexes
                .insert_with_batch_position(&batch, start as u64, Some(position))
                .unwrap();
        }
        (batch_store, Arc::new(indexes))
    }

    /// One search of `id_idx`, the way the expression pass produces it.
    fn search(query: SargableQuery) -> ScalarIndexExpr {
        ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "id".to_string(),
            index_name: "id_idx".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(query),
            needs_recheck: false,
            fragment_bitmap: None,
        })
    }

    /// Ids from `start` up.
    fn from(start: i32) -> SargableQuery {
        SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(start))),
            Bound::Unbounded,
        )
    }

    /// `filter` compiled for the memtable's rows.
    fn recheck(filter: &str) -> PhysicalExprRef {
        let planner = lance_datafusion::planner::Planner::new(schema(false));
        let filter = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        planner.create_physical_expr(&filter).unwrap()
    }

    /// The exec answering `index_expr` from the first `readable` batches, with
    /// row ids.
    fn exec(
        memtable: &Memtable,
        index_expr: ScalarIndexExpr,
        recheck: Option<PhysicalExprRef>,
        readable: usize,
    ) -> ScalarMemIndexExec {
        ScalarMemIndexExec::new(
            memtable.0.clone(),
            memtable.1.clone(),
            index_expr,
            recheck,
            true,
            readable,
            None,
            schema(true),
            true,
            false,
        )
    }

    /// The ids and row ids `exec` returns, in order.
    async fn read(exec: ScalarMemIndexExec) -> (Vec<i32>, Vec<u64>) {
        let batches: Vec<RecordBatch> = exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut ids = Vec::new();
        let mut row_ids = Vec::new();
        for batch in &batches {
            ids.extend(
                batch["id"]
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values(),
            );
            row_ids.extend(
                batch["_rowid"]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values(),
            );
        }
        (ids, row_ids)
    }

    fn as_row_ids(ids: &[i32]) -> Vec<u64> {
        ids.iter().map(|id| *id as u64).collect()
    }

    /// A search returns exactly its rows from the readable batches, each with
    /// its position as its row id.
    #[rstest::rstest]
    #[case::equality(SargableQuery::Equals(ScalarValue::Int32(Some(5))), 2, vec![5])]
    #[case::list(
        SargableQuery::IsIn([2, 5, 8].map(|id| ScalarValue::Int32(Some(id))).to_vec()),
        2,
        vec![2, 5, 8]
    )]
    #[case::past_the_readable_batches(SargableQuery::Equals(ScalarValue::Int32(Some(15))), 1, vec![])]
    #[case::in_the_readable_batches(SargableQuery::Equals(ScalarValue::Int32(Some(15))), 2, vec![15])]
    #[tokio::test]
    async fn a_search_returns_its_readable_rows(
        #[case] query: SargableQuery,
        #[case] readable: usize,
        #[case] expected: Vec<i32>,
    ) {
        let (ids, row_ids) = read(exec(&memtable(2, 10), search(query), None, readable)).await;
        assert_eq!(ids, expected);
        assert_eq!(row_ids, as_row_ids(&expected));
    }

    /// Past the match budget every visible row is read with the filter, and a
    /// batch past the readable count stays unread. The filter is wider than the
    /// index query, so only a declined search returns ids below 100.
    #[tokio::test]
    async fn a_broad_answer_reads_every_visible_row_with_the_filter() {
        let memtable = memtable(READABLE_BATCHES + 1, ROWS_PER_BATCH);
        let visible_rows = (READABLE_BATCHES * ROWS_PER_BATCH) as u64;
        let budget = (visible_rows / MATCH_BUDGET_DIVISOR).max(MIN_MATCH_BUDGET);
        let ctx = SearchContext::new(visible_rows - 1).with_match_budget(budget);
        assert!(
            memtable
                .1
                .get_index("id_idx")
                .unwrap()
                .search(&from(100), &ctx)
                .unwrap()
                .is_none(),
            "the B-tree must decline this many matches"
        );

        let (ids, row_ids) = read(exec(
            &memtable,
            search(from(100)),
            Some(recheck("id >= 50")),
            READABLE_BATCHES,
        ))
        .await;
        let expected: Vec<i32> = (50..visible_rows as i32).collect();
        assert_eq!(ids, expected);
        assert_eq!(row_ids, as_row_ids(&expected));
    }

    /// With no filter to fall back on, a search past the budget is still
    /// answered.
    #[tokio::test]
    async fn a_search_with_no_filter_to_fall_back_on_is_never_declined() {
        let memtable = memtable(READABLE_BATCHES + 1, ROWS_PER_BATCH);
        let (ids, _) = read(exec(&memtable, search(from(100)), None, READABLE_BATCHES)).await;
        let expected: Vec<i32> = (100..(READABLE_BATCHES * ROWS_PER_BATCH) as i32).collect();
        assert_eq!(ids, expected);
    }

    /// Candidates an index only narrowed to are checked against the filter,
    /// both when they fill a batch and when they are a few of its rows, and
    /// each kept row keeps its own row id.
    #[tokio::test]
    async fn narrowed_candidates_are_checked_against_the_filter() {
        // All of the first batch and half of the second are candidates.
        let ScalarIndexExpr::Query(mut narrowing) = search(SargableQuery::Range(
            Bound::Unbounded,
            Bound::Excluded(ScalarValue::Int32(Some(15))),
        )) else {
            unreachable!()
        };
        narrowing.needs_recheck = true;
        let (ids, row_ids) = read(exec(
            &memtable(2, 10),
            ScalarIndexExpr::Query(narrowing),
            Some(recheck("id < 15 AND id % 2 = 0")),
            2,
        ))
        .await;
        assert_eq!(ids, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(row_ids, as_row_ids(&ids));
    }

    /// The plan shows the query and whether the searches are the whole filter.
    #[tokio::test]
    async fn the_plan_shows_the_query_and_its_flags() {
        use crate::utils::test::assert_plan_node_equals;

        let (batch_store, indexes) = memtable(1, 10);
        let equals_5 = || search(SargableQuery::Equals(ScalarValue::Int32(Some(5))));
        let whole = ScalarMemIndexExec::new(
            batch_store.clone(),
            indexes.clone(),
            equals_5(),
            None,
            true,
            1,
            None,
            schema(false),
            false,
            false,
        );
        assert_plan_node_equals(
            Arc::new(whole),
            "ScalarMemIndexExec: query=id = Int32(5), whole_filter=true, with_row_id=false, with_row_address=false",
        )
        .await
        .unwrap();

        // Part of the filter is left for the re-check.
        let partial = ScalarMemIndexExec::new(
            batch_store,
            indexes,
            equals_5(),
            Some(recheck("id = 5 AND name = 'x'")),
            false,
            1,
            None,
            schema(true),
            true,
            false,
        );
        assert_plan_node_equals(
            Arc::new(partial),
            "ScalarMemIndexExec: query=id = Int32(5), whole_filter=false, with_row_id=true, with_row_address=false",
        )
        .await
        .unwrap();
    }
}
