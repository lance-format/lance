// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemTableBruteForceVectorExec — KNN over the active memtable without an HNSW.
//!
//! Mirrors [`super::VectorIndexExec`]'s output contract (same schema, same row
//! shape, same `_distance` / `_rowid` semantics) so the LSM caller can swap one
//! for the other based on whether the memtable's `IndexStore` has an HNSW for
//! the queried column. The active memtable is the LSM's unindexed-rows path:
//! whenever the HNSW config is absent (cold-start before the Indexer commits,
//! or new rows in the window between commit and next memtable rotation), this
//! exec keeps KNN correct by computing exact distances row-by-row.

use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Float32Array, RecordBatch, UInt64Array, cast::AsArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::ScalarValue;
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
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use lance_index::scalar::expression::ScalarIndexExpr;
use lance_linalg::distance::DistanceType;

use super::super::builder::VectorQuery;
use super::newest_pk_positions;
use super::vector::DISTANCE_COLUMN;
use crate::dataset::mem_wal::index::{SearchContext, evaluate_index_filter};
use crate::dataset::mem_wal::memtable::scanner::exec::{scan_record_batch, take_projected_columns};
use crate::dataset::mem_wal::scanner::exec::resolve_pk_indices;
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Distance metric used when [`VectorQuery::distance_type`] is `None`. The
/// indexed path defers to the index's own metric, but with no index there is
/// no inherent default — L2 matches what most callers configure and what the
/// SSTable/base arms use when re-ranking unindexed candidates.
const DEFAULT_DISTANCE_TYPE: DistanceType = DistanceType::L2;

/// Past `1 / NEWEST_SEEK_SHARE` of the visible rows, checking each candidate's
/// key with a seek costs more than one pass hashing every visible key. The
/// same crossover the filter indexes' newest-only reads use.
const NEWEST_SEEK_SHARE: u64 = 8;

/// The part of the prefilter the memtable's filter indexes answer.
#[derive(Debug, Clone)]
struct IndexFilter {
    /// The index searches the filter was split into.
    expr: ScalarIndexExpr,
    /// Whether those searches are the whole filter. When some condition had no
    /// index, an exact index answer is still only a superset.
    covers_filter: bool,
}

/// How a search keeps only each primary key's newest visible version.
enum Newest {
    /// Every row is its key's newest version, or there is no key.
    Any,
    /// The newest versions' positions, from a pass over every visible key.
    Among(HashSet<u64>),
    /// A seek in the key index per row.
    Seek,
}

/// Brute-force KNN over an active memtable without an HNSW. Produces the same
/// output schema as [`super::VectorIndexExec`].
#[derive(Clone)]
pub struct MemTableBruteForceVectorExec {
    batch_store: Arc<BatchStore>,
    query: VectorQuery,
    readable_count: usize,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    with_row_id: bool,
    /// Optional prefilter predicate, compiled against the memtable schema.
    /// Applied per row before the top-k cut so the KNN only ranks matching
    /// rows (true prefilter, not a lossy post-filter on the top-k).
    filter: Option<PhysicalExprRef>,
    /// Primary-key columns. When set, only the newest version of each PK is
    /// eligible for top-k. With a filter, this evaluates the predicate against
    /// the current PK version instead of falling back to a stale older version.
    pk_columns: Option<Vec<String>>,
    /// The memtable's indexes: the primary-key index tells whether any key was
    /// rewritten and which version is newest, and the filter indexes narrow the
    /// rows the prefilter is applied to.
    indexes: Option<Arc<IndexStore>>,
    /// The prefilter as searches in `indexes`, when they answer some of it.
    index_filter: Option<IndexFilter>,
}

impl Debug for MemTableBruteForceVectorExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemTableBruteForceVectorExec")
            .field("column", &self.query.column)
            .field("k", &self.query.k)
            .field("readable_count", &self.readable_count)
            .field("with_row_id", &self.with_row_id)
            .finish()
    }
}

impl MemTableBruteForceVectorExec {
    /// Build the exec. `base_schema` is the post-projection row schema (no
    /// `_distance`, no `_rowid`); `_distance` is appended unconditionally and
    /// `_rowid` only when `with_row_id` is set, matching [`VectorIndexExec`].
    pub fn new(
        batch_store: Arc<BatchStore>,
        query: VectorQuery,
        readable_count: usize,
        projection: Option<Vec<usize>>,
        base_schema: SchemaRef,
        with_row_id: bool,
    ) -> Result<Self> {
        let mut fields: Vec<Field> = base_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(DISTANCE_COLUMN, DataType::Float32, true));
        if with_row_id {
            fields.push(Field::new(lance_core::ROW_ID, DataType::UInt64, true));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            batch_store,
            query,
            readable_count,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            with_row_id,
            filter: None,
            pk_columns: None,
            indexes: None,
            index_filter: None,
        })
    }

    /// Attach an optional prefilter predicate (compiled against the memtable
    /// schema). Rows that fail the predicate are excluded before the top-k cut.
    pub fn with_filter(mut self, filter: Option<PhysicalExprRef>) -> Self {
        self.filter = filter;
        self
    }

    /// Provide the primary-key columns so search keeps only the newest version
    /// of each PK (see `pk_columns`).
    pub fn with_pk_columns(mut self, pk_columns: Option<Vec<String>>) -> Self {
        self.pk_columns = pk_columns.filter(|columns| !columns.is_empty());
        self
    }

    /// Provide the memtable's indexes, so a search skips the newest-version
    /// pass when no key was ever rewritten.
    pub fn with_indexes(mut self, indexes: Arc<IndexStore>) -> Self {
        self.indexes = Some(indexes);
        self
    }

    /// Narrow the prefilter's rows with searches in the filter indexes:
    /// `expr` is what they answer of it, and `covers_filter` whether that is
    /// the whole filter. Needs [`Self::with_indexes`]; the filter itself is
    /// still applied to every candidate the indexes did not settle.
    pub fn with_index_filter(mut self, expr: ScalarIndexExpr, covers_filter: bool) -> Self {
        self.index_filter = Some(IndexFilter {
            expr,
            covers_filter,
        });
        self
    }

    /// Evaluate the prefilter predicate against a memtable batch, returning a
    /// keep-mask (`true` = retain). `Ok(None)` when no filter is configured.
    fn filter_mask(&self, batch: &RecordBatch) -> Result<Option<BooleanArray>> {
        let Some(ref predicate) = self.filter else {
            return Ok(None);
        };
        let values = predicate
            .evaluate(batch)
            .and_then(|v| v.into_array(batch.num_rows()))
            .map_err(|e| {
                Error::invalid_input(format!("vector prefilter evaluation failed: {}", e))
            })?;
        let mask = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "vector prefilter predicate did not evaluate to boolean".to_string(),
                )
            })?
            .clone();
        Ok(Some(mask))
    }

    /// Last row position within `readable_count`, or `None` if nothing is
    /// readable. Identical to `VectorIndexExec`'s helper so both arms cut at
    /// the same bound.
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

    /// Extract the flat per-element query vector. `arrow_batch_func` wants
    /// `(from: &dyn Array, to: &FixedSizeListArray)` where `from` is the raw
    /// primitive array of one vector (NOT an FSL), so unwrap an FSL with one
    /// row if that's how the caller built it.
    fn query_as_flat(&self) -> Result<Arc<dyn Array>> {
        let query_array = self.query.query_vector.as_ref();
        if let Some(fsl) = query_array.as_fixed_size_list_opt() {
            if fsl.len() != 1 {
                return Err(Error::invalid_input(format!(
                    "brute-force vector search expects a single query vector, got {}",
                    fsl.len()
                )));
            }
            return Ok(fsl.value(0));
        }
        Ok(self.query.query_vector.clone())
    }

    /// The filter indexes' candidates for the prefilter, ascending, or `None`
    /// when every visible row is one; and whether those candidates are exactly
    /// the rows the filter keeps, so it need not be applied to them.
    fn index_candidates(&self, max_readable_row: u64) -> (Option<Vec<u64>>, bool) {
        let (Some(indexes), Some(index_filter)) = (&self.indexes, &self.index_filter) else {
            return (None, false);
        };
        let ctx = SearchContext::new(max_readable_row);
        match evaluate_index_filter(&index_filter.expr, indexes, &ctx) {
            Ok(result) => {
                let settled = result.is_exact() && index_filter.covers_filter;
                if result.at_most.len() == max_readable_row + 1 {
                    (None, settled)
                } else {
                    (Some(result.at_most.into()), settled)
                }
            }
            // A failing index must not answer "no rows": every row is a
            // candidate and the filter decides.
            Err(error) => {
                log::warn!(
                    "a memtable index failed to search {}; filtering every row instead: {error}",
                    index_filter.expr
                );
                (None, false)
            }
        }
    }

    /// How to keep each primary key's newest visible version, given how many
    /// rows passed the prefilter: not at all when no key was ever rewritten, a
    /// seek in the key index per row when few passed, and otherwise a pass over
    /// every visible key.
    fn newest_rule(&self, max_readable_row: u64, passed: u64) -> Result<Newest> {
        let Some(pk_columns) = &self.pk_columns else {
            return Ok(Newest::Any);
        };
        if let Some(indexes) = &self.indexes
            && indexes.has_pk_index()
        {
            // Read at execution, not planning: the flag is set before a rewrite
            // becomes visible and is never cleared, so a later read can only
            // err toward checking.
            if !indexes.pk_has_overrides() {
                return Ok(Newest::Any);
            }
            if passed <= (max_readable_row + 1) / NEWEST_SEEK_SHARE {
                return Ok(Newest::Seek);
            }
        }
        newest_pk_positions(
            &self.batch_store,
            pk_columns,
            self.readable_count,
            max_readable_row,
        )
        .map(Newest::Among)
        .map_err(|e| Error::invalid_input(e.to_string()))
    }

    /// The visible rows the filter indexes leave and the prefilter keeps, per
    /// stored batch: the batch's first row position, the batch, and the kept
    /// rows' offsets in it. Rows are tracked by offset rather than copied out,
    /// so a broad answer costs no more than reading every row.
    fn prefiltered_rows(&self, max_readable_row: u64) -> Result<Vec<(u64, RecordBatch, Vec<u32>)>> {
        let (index_candidates, settled) = self.index_candidates(max_readable_row);
        let mut current_row: u64 = 0;
        let mut next_candidate = 0usize;
        let mut passed = Vec::new();
        for (batch_position, stored_batch) in self.batch_store.iter().enumerate() {
            let n = stored_batch.num_rows;
            let start = current_row;
            current_row += n as u64;
            if n == 0 || batch_position >= self.readable_count || start > max_readable_row {
                continue;
            }
            let end = current_row.min(max_readable_row + 1);

            let mut rows: Vec<u32> = match &index_candidates {
                Some(positions) => {
                    let first = next_candidate;
                    while next_candidate < positions.len() && positions[next_candidate] < end {
                        next_candidate += 1;
                    }
                    positions[first..next_candidate]
                        .iter()
                        .map(|&pos| (pos - start) as u32)
                        .collect()
                }
                None => (0..(end - start) as u32).collect(),
            };
            if rows.is_empty() {
                continue;
            }
            let scan_batch = scan_record_batch(&stored_batch.data)?;

            // Prefilter: drop rows that fail the predicate before they reach the
            // top-k heap (a NULL predicate result excludes the row, matching SQL).
            if !settled && let Some(mask) = self.filter_mask(&scan_batch)? {
                rows.retain(|&row| mask.is_valid(row as usize) && mask.value(row as usize));
            }
            if !rows.is_empty() {
                passed.push((start, scan_batch, rows));
            }
        }
        Ok(passed)
    }

    /// Compute `(distance, row_position)` for every visible row that passes the
    /// prefilter and is its key's newest version, then top-k by distance
    /// ascending. Distances are computed over whole batches, since copying a
    /// vector out costs far more than its distance. Rows where
    /// the vector column is null or where the computed distance is non-finite
    /// are skipped — same convention as the HNSW search (which filters on
    /// `result.distance.is_finite()`).
    fn compute_topk(&self) -> Result<Vec<(f32, u64)>> {
        if self.query.k == 0 {
            return Ok(Vec::new());
        }
        let Some(max_readable_row) = self.compute_max_readable_row() else {
            return Ok(Vec::new());
        };
        let query_flat = self.query_as_flat()?;
        let column_name = self.query.column.as_str();
        let distance_type = self.query.distance_type.unwrap_or(DEFAULT_DISTANCE_TYPE);
        let batch_func = distance_type.arrow_batch_func();

        let passed = self.prefiltered_rows(max_readable_row)?;
        // When PK columns are configured, only the newest version of each PK is
        // eligible. This keeps top-k slots from being consumed by superseded
        // rows and makes filtered search evaluate the predicate against the
        // current version of the PK.
        let passed_rows = passed.iter().map(|(_, _, rows)| rows.len() as u64).sum();
        let newest = self.newest_rule(max_readable_row, passed_rows)?;

        let mut candidates: Vec<(f32, u64)> = Vec::new();
        for (start, scan_batch, mut rows) in passed {
            // Skip superseded versions: only the newest version of each PK is
            // eligible, so a newer non-matching version excludes the PK.
            match &newest {
                Newest::Any => {}
                Newest::Among(positions) => {
                    rows.retain(|&row| positions.contains(&(start + row as u64)));
                }
                Newest::Seek => {
                    let (Some(indexes), Some(pk_columns)) = (&self.indexes, &self.pk_columns)
                    else {
                        unreachable!("a seek is chosen only with a key index and key columns");
                    };
                    let pk_indices = resolve_pk_indices(&scan_batch, pk_columns)?;
                    let mut kept = Vec::with_capacity(rows.len());
                    for row in rows {
                        let values = pk_indices
                            .iter()
                            .map(|&column| {
                                ScalarValue::try_from_array(scan_batch.column(column), row as usize)
                            })
                            .collect::<DataFusionResult<Vec<_>>>()?;
                        if indexes.pk_is_newest(&values, start + row as u64, max_readable_row) {
                            kept.push(row);
                        }
                    }
                    rows = kept;
                }
            }
            if rows.is_empty() {
                continue;
            }

            let column = scan_batch.column_by_name(column_name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vector column '{}' not found in memtable schema",
                    column_name
                ))
            })?;
            let column_fsl = column.as_fixed_size_list_opt().ok_or_else(|| {
                Error::invalid_input(format!(
                    "Vector column '{}' must be FixedSizeList; got {:?}",
                    column_name,
                    column.data_type()
                ))
            })?;

            let distances = batch_func(query_flat.as_ref(), column_fsl).map_err(|e| {
                Error::invalid_input(format!(
                    "brute-force distance computation failed for column '{}': {}",
                    column_name, e
                ))
            })?;

            for row in rows {
                if distances.is_null(row as usize) {
                    continue;
                }
                let dist = distances.value(row as usize);
                if !dist.is_finite() {
                    continue;
                }
                candidates.push((dist, start + row as u64));
            }
        }

        // `partial_cmp` defaults Equal on NaN; we filtered non-finite above so
        // every remaining value compares deterministically.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if self.query.distance_lower_bound.is_some() || self.query.distance_upper_bound.is_some() {
            candidates.retain(|&(dist, _)| {
                let above_lower = self.query.distance_lower_bound.is_none_or(|lb| dist >= lb);
                let below_upper = self.query.distance_upper_bound.is_none_or(|ub| dist < ub);
                above_lower && below_upper
            });
        }

        candidates.truncate(self.query.k);
        Ok(candidates)
    }

    /// Materialize the top-k rows from the batch store, mirroring
    /// `VectorIndexExec::materialize_rows`. Groups by batch so the per-batch
    /// `take` is amortized; emits one output batch per source batch that
    /// contributes.
    fn materialize_rows(&self, results: &[(f32, u64)]) -> DataFusionResult<Vec<RecordBatch>> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        let mut batch_ranges = Vec::new();
        let mut current_row = 0usize;
        for stored_batch in self.batch_store.iter() {
            let start = current_row;
            let end = current_row + stored_batch.num_rows;
            batch_ranges.push((start, end));
            current_row = end;
        }

        let mut batches_data: std::collections::HashMap<usize, Vec<(usize, f32, u64)>> =
            std::collections::HashMap::new();
        for &(distance, pos) in results {
            let pos_usize = pos as usize;
            for (batch_id, &(start, end)) in batch_ranges.iter().enumerate() {
                if pos_usize >= start && pos_usize < end {
                    batches_data.entry(batch_id).or_default().push((
                        pos_usize - start,
                        distance,
                        pos,
                    ));
                    break;
                }
            }
        }

        let mut all_batches = Vec::new();
        for (batch_id, rows_with_dist) in batches_data {
            if let Some(stored) = self.batch_store.get(batch_id) {
                let data = scan_record_batch(&stored.data)?;
                let rows: Vec<u32> = rows_with_dist.iter().map(|&(r, _, _)| r as u32).collect();
                let distances: Vec<f32> = rows_with_dist.iter().map(|&(_, d, _)| d).collect();
                let row_positions: Vec<u64> =
                    rows_with_dist.iter().map(|&(_, _, pos)| pos).collect();

                let indices = arrow_array::UInt32Array::from(rows);

                let mut columns: Vec<Arc<dyn arrow_array::Array>> = data
                    .columns()
                    .iter()
                    .map(|col| arrow_select::take::take(col.as_ref(), &indices, None).unwrap())
                    .collect();

                columns.push(Arc::new(Float32Array::from(distances)));

                let source_schema = data.schema();
                let mut final_columns = if let Some(ref proj_indices) = self.projection {
                    let mut projected: Vec<_> = take_projected_columns(
                        &columns,
                        source_schema.fields(),
                        proj_indices,
                        self.output_schema.as_ref(),
                        row_positions.len(),
                    )?;
                    // Distance was just pushed onto `columns`; keep it last.
                    projected.push(columns.last().unwrap().clone());
                    projected
                } else {
                    columns
                };

                if self.with_row_id {
                    final_columns.push(Arc::new(UInt64Array::from(row_positions)));
                }

                let batch = RecordBatch::try_new(self.output_schema.clone(), final_columns)?;
                all_batches.push(batch);
            }
        }

        Ok(all_batches)
    }
}

impl DisplayAs for MemTableBruteForceVectorExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MemTableBruteForceVectorExec: column={}, k={}, with_row_id={}",
                    self.query.column, self.query.k, self.with_row_id
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MemTableBruteForceVectorExec\ncolumn={}\nk={}\nwith_row_id={}",
                    self.query.column, self.query.k, self.with_row_id
                )
            }
        }
    }
}

impl ExecutionPlan for MemTableBruteForceVectorExec {
    fn name(&self) -> &str {
        "MemTableBruteForceVectorExec"
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
                "MemTableBruteForceVectorExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        // Ranking reads every candidate row, so it runs on the CPU pool from
        // inside the stream rather than on the caller's thread in `execute`.
        let exec = self.clone();
        let batches = stream::once(spawn_cpu(move || {
            let results = exec
                .compute_topk()
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
            exec.materialize_rows(&results)
        }))
        .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
        .try_flatten()
        .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            batches,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics {
            num_rows: Precision::Exact(self.query.k),
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
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::common::collect;
    use datafusion::prelude::{Expr, SessionContext, col, lit};
    use lance_datafusion::planner::Planner;

    fn make_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
        ]))
    }

    fn make_schema_with_active() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
            Field::new("active", DataType::Boolean, true),
        ]))
    }

    fn make_batch(schema: SchemaRef, ids: &[i32], vectors: &[[f32; 2]]) -> RecordBatch {
        let id_array = Arc::new(Int32Array::from(ids.to_vec())) as Arc<dyn Array>;
        let values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let inner = Arc::new(Float32Array::from(values));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vec_array =
            Arc::new(FixedSizeListArray::try_new(field, 2, inner, None).expect("build fsl"))
                as Arc<dyn Array>;
        RecordBatch::try_new(schema, vec![id_array, vec_array]).expect("build batch")
    }

    fn make_batch_with_active(
        schema: SchemaRef,
        ids: &[i32],
        vectors: &[[f32; 2]],
        active: &[Option<bool>],
    ) -> RecordBatch {
        let id_array = Arc::new(Int32Array::from(ids.to_vec())) as Arc<dyn Array>;
        let values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let inner = Arc::new(Float32Array::from(values));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vec_array =
            Arc::new(FixedSizeListArray::try_new(field, 2, inner, None).expect("build fsl"))
                as Arc<dyn Array>;
        let active_array = Arc::new(BooleanArray::from(active.to_vec())) as Arc<dyn Array>;
        RecordBatch::try_new(schema, vec![id_array, vec_array, active_array]).expect("build batch")
    }

    fn store_with_batches(batches: Vec<RecordBatch>) -> Arc<BatchStore> {
        let store = Arc::new(BatchStore::with_capacity(batches.len().max(1)));
        for batch in batches {
            store.append(batch).expect("append batch");
        }
        store
    }

    fn query_for(vector: [f32; 2], k: usize) -> VectorQuery {
        let values = Arc::new(Float32Array::from(vector.to_vec())) as Arc<dyn Array>;
        VectorQuery {
            column: "vector".to_string(),
            query_vector: values,
            k,
            nprobes: 1,
            maximum_nprobes: None,
            distance_type: Some(DistanceType::L2),
            ef: None,
            refine_factor: None,
            distance_lower_bound: None,
            distance_upper_bound: None,
        }
    }

    fn physical_filter(schema: SchemaRef, expr: Expr) -> PhysicalExprRef {
        let planner = Planner::new(schema);
        let optimized = planner.optimize_expr(expr).expect("optimize filter");
        planner
            .create_physical_expr(&optimized)
            .expect("create physical filter")
    }

    async fn execute_to_batches(exec: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).expect("execute");
        collect(stream).await.expect("collect")
    }

    fn ids_from_batches(batches: &[RecordBatch]) -> Vec<i32> {
        let mut ids = Vec::new();
        for batch in batches {
            let id_arr = batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            for row in 0..batch.num_rows() {
                ids.push(id_arr.value(row));
            }
        }
        ids
    }

    #[tokio::test]
    async fn top_k_by_distance() {
        // Five rows; query at (0,0); L2 distances are id² each — expect ids in
        // ascending order, capped at k=3.
        let schema = make_schema();
        let batch = make_batch(
            schema.clone(),
            &[0, 1, 2, 3, 4],
            &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        );
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 3);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(
                store,
                query,
                /* readable_count = */ usize::MAX,
                None,
                schema,
                false,
            )
            .expect("ctor"),
        );
        let out = execute_to_batches(exec).await;
        // Concat and check ids + distances in order.
        let total: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3, "k=3 cap not honored: got {total} rows");

        let mut id_dist: Vec<(i32, f32)> = Vec::new();
        for batch in &out {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            let dists = batch
                .column_by_name(DISTANCE_COLUMN)
                .unwrap()
                .as_primitive::<arrow_array::types::Float32Type>();
            for i in 0..batch.num_rows() {
                id_dist.push((ids.value(i), dists.value(i)));
            }
        }
        id_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        assert_eq!(id_dist[0].0, 0);
        assert_eq!(id_dist[1].0, 1);
        assert_eq!(id_dist[2].0, 2);
    }

    #[tokio::test]
    async fn empty_memtable_returns_empty_with_distance_schema() {
        let schema = make_schema();
        let store = Arc::new(BatchStore::with_capacity(4));
        let query = query_for([0.5, 0.5], 10);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(store, query, usize::MAX, None, schema, false)
                .expect("ctor"),
        );
        let out_schema = exec.schema();
        assert!(
            out_schema.field_with_name(DISTANCE_COLUMN).is_ok(),
            "output schema must contain `{DISTANCE_COLUMN}` even with empty memtable; got {:?}",
            out_schema
        );
        let out = execute_to_batches(exec).await;
        let total: usize = out.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn respects_indexed_count() {
        // Two batches of two rows. Freeze at batch 0 — only ids 0,1 are
        // visible candidates; the (closer) ids 2,3 in batch 1 are excluded.
        let schema = make_schema();
        let b0 = make_batch(schema.clone(), &[0, 1], &[[5.0, 0.0], [6.0, 0.0]]);
        let b1 = make_batch(schema.clone(), &[2, 3], &[[1.0, 0.0], [2.0, 0.0]]);
        let store = store_with_batches(vec![b0, b1]);
        let query = query_for([0.0, 0.0], 4);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(
                store, query, /* readable_count = */ 1, None, schema, false,
            )
            .expect("ctor"),
        );
        let out = execute_to_batches(exec).await;
        let mut returned_ids: Vec<i32> = Vec::new();
        for batch in &out {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            for i in 0..batch.num_rows() {
                returned_ids.push(ids.value(i));
            }
        }
        returned_ids.sort();
        assert_eq!(returned_ids, vec![0, 1]);
    }

    #[tokio::test]
    async fn applies_distance_bounds() {
        let schema = make_schema();
        let batch = make_batch(
            schema.clone(),
            &[0, 1, 2, 3],
            &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        );
        let store = store_with_batches(vec![batch]);
        let mut query = query_for([0.0, 0.0], 10);
        // L2² distances: 0, 1, 4, 9. Keep only distances in [1, 5) — ids 1, 2.
        query.distance_lower_bound = Some(1.0);
        query.distance_upper_bound = Some(5.0);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(store, query, usize::MAX, None, schema, false)
                .expect("ctor"),
        );
        let out = execute_to_batches(exec).await;
        let mut ids: Vec<i32> = Vec::new();
        for batch in &out {
            let id_arr = batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            for i in 0..batch.num_rows() {
                ids.push(id_arr.value(i));
            }
        }
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[tokio::test]
    async fn populates_row_id_when_requested() {
        let schema = make_schema();
        let batch = make_batch(
            schema.clone(),
            &[10, 11, 12],
            &[[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        );
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 3);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(
                store,
                query,
                usize::MAX,
                None,
                schema,
                /* with_row_id = */ true,
            )
            .expect("ctor"),
        );
        let out_schema = exec.schema();
        assert!(out_schema.field_with_name(lance_core::ROW_ID).is_ok());

        let out = execute_to_batches(exec).await;
        let mut pairs: Vec<(i32, u64)> = Vec::new();
        for batch in &out {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>();
            let rowids = batch
                .column_by_name(lance_core::ROW_ID)
                .unwrap()
                .as_primitive::<arrow_array::types::UInt64Type>();
            for i in 0..batch.num_rows() {
                pairs.push((ids.value(i), rowids.value(i)));
            }
        }
        // Row offsets are insert-order: id=10 → 0, id=11 → 1, id=12 → 2.
        pairs.sort_by_key(|(id, _)| *id);
        assert_eq!(pairs, vec![(10, 0), (11, 1), (12, 2)]);
    }

    #[tokio::test]
    async fn prefilter_null_predicate_excludes_rows() {
        let schema = make_schema_with_active();
        let batch = make_batch_with_active(
            schema.clone(),
            &[1, 2, 3],
            &[[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]],
            &[None, Some(true), Some(false)],
        );
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 3);
        let filter = physical_filter(schema.clone(), col("active").eq(lit(true)));
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(store, query, usize::MAX, None, schema, false)
                .expect("ctor")
                .with_filter(Some(filter)),
        );

        let out = execute_to_batches(exec).await;
        assert_eq!(
            ids_from_batches(&out),
            vec![2],
            "NULL predicate results must be excluded from vector prefilter candidates"
        );
    }

    #[tokio::test]
    async fn prefilter_with_pk_columns_drops_stale_matching_version() {
        let schema = make_schema_with_active();
        let batch = make_batch_with_active(
            schema.clone(),
            &[5, 5],
            &[[0.0, 0.0], [10.0, 0.0]],
            &[Some(true), Some(false)],
        );
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 10);
        let filter = physical_filter(schema.clone(), col("active").eq(lit(true)));
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(store, query, usize::MAX, None, schema, false)
                .expect("ctor")
                .with_filter(Some(filter))
                .with_pk_columns(Some(vec!["id".to_string()])),
        );

        let out = execute_to_batches(exec).await;
        assert!(
            out.iter().all(|batch| batch.num_rows() == 0),
            "the older matching vector version must not leak when the newest PK fails the filter"
        );
    }

    #[tokio::test]
    async fn pk_columns_keep_newest_version_without_filter() {
        let schema = make_schema();
        let batch = make_batch(schema.clone(), &[5, 5], &[[0.0, 0.0], [10.0, 0.0]]);
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 10);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(
                store,
                query,
                usize::MAX,
                None,
                schema,
                /* with_row_id = */ true,
            )
            .expect("ctor")
            .with_pk_columns(Some(vec!["id".to_string()])),
        );

        let out = execute_to_batches(exec).await;
        let row_ids: Vec<u64> = out
            .iter()
            .flat_map(|batch| {
                let row_ids = batch
                    .column_by_name(lance_core::ROW_ID)
                    .unwrap()
                    .as_primitive::<arrow_array::types::UInt64Type>();
                (0..batch.num_rows())
                    .map(|row| row_ids.value(row))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            row_ids,
            vec![1],
            "brute-force vector PK recency must keep only the newest duplicate"
        );
    }

    #[tokio::test]
    async fn empty_pk_columns_do_not_collapse_results() {
        let schema = make_schema();
        let batch = make_batch(
            schema.clone(),
            &[1, 2, 3],
            &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        );
        let store = store_with_batches(vec![batch]);
        let query = query_for([0.0, 0.0], 3);
        let exec = Arc::new(
            MemTableBruteForceVectorExec::new(store, query, usize::MAX, None, schema, false)
                .expect("ctor")
                .with_pk_columns(Some(vec![])),
        );

        let out = execute_to_batches(exec).await;
        let mut ids = ids_from_batches(&out);
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2, 3],
            "empty PK columns should behave like no PK columns, not one empty tuple key"
        );
    }
}
